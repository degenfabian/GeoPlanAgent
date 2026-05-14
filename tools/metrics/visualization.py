"""
Visualization Tools for Planning Document Digitization

This module provides tools for visualizing GeoJSON boundaries on maps,
allowing the agent to verify and refine extracted boundaries by comparing
them visually to the source map.

The visualization uses OpenStreetMap tiles as a basemap, making it easy
to see if the extracted boundary aligns correctly with real-world features
like roads, rivers, and buildings.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import base64
from pathlib import Path
from typing import Dict, Any, Optional

import geopandas as gpd
import contextily as ctx
from shapely.geometry import shape
from shapely.ops import unary_union
from PIL import Image


def visualize_geojson_boundary(
    geojson_data: Dict[str, Any],
    padding: float = 2.0,
    max_size: int = 1024,
) -> Dict[str, Any]:
    """
    Visualize a GeoJSON boundary on an OpenStreetMap basemap.

    Uses GeoPandas and contextily for visualization with automatic
    zoom level calculation.

    Args:
        geojson_data: GeoJSON Feature object containing the boundary polygon.
        padding: Padding around boundary as fraction of boundary size (default: 3.0).
                 A value of 1.0 means 100% extra space on each side.
        max_size: Maximum dimension (width or height) in pixels (default: 1024).
                  Images larger than this will be resized to fit API constraints.

    Returns:
        Dict containing:
        - "success" (bool): Whether visualization succeeded
        - "image_base64" (str): Base64-encoded PNG image of the map
        - "bbox" (Dict): Bounding box with min/max lat/lon
        - "error" (str): Error message if failed
    """
    try:
        plt.close("all")

        # Extract geometry and validate type
        geometry = geojson_data["geometry"]
        geom_type = geometry["type"]

        if geom_type not in ("Polygon", "MultiPolygon"):
            return {
                "success": False,
                "error": f"Unsupported geometry type: {geom_type}",
            }

        # Create GeoDataFrame from GeoJSON (in WGS84 / EPSG:4326)
        geom = shape(geometry)
        gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")

        # Get bounding box in original CRS for return value
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        lon_min, lat_min, lon_max, lat_max = bounds

        if lon_min == lon_max or lat_min == lat_max:
            return {"success": False, "error": "No valid extent found in GeoJSON"}

        # Reproject to Web Mercator (EPSG:3857) for contextily
        gdf_mercator = gdf.to_crs(epsg=3857)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(14, 12))

        # Plot the boundary
        gdf_mercator.plot(
            ax=ax,
            facecolor="red",
            edgecolor="red",
            alpha=0.15,
            linewidth=2,
        )
        # Plot boundary outline again for better visibility
        gdf_mercator.boundary.plot(ax=ax, color="red", linewidth=2, label="Boundary")

        # Calculate extent with padding
        bounds_mercator = gdf_mercator.total_bounds
        minx, miny, maxx, maxy = bounds_mercator

        x_pad = (maxx - minx) * padding
        y_pad = (maxy - miny) * padding

        ax.set_xlim(minx - x_pad, maxx + x_pad)
        ax.set_ylim(miny - y_pad, maxy + y_pad)

        # Add OpenStreetMap basemap (contextily auto-calculates zoom)
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        # Add title and legend
        ax.set_title("Planning Area Boundary", fontsize=14, pad=10)
        ax.legend(loc="upper right")

        # Remove axis labels (they're in Web Mercator units, not useful)
        ax.set_axis_off()

        plt.tight_layout()

        # Convert to image buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        # Load image and resize if needed
        img = Image.open(buf)
        if img.width > max_size or img.height > max_size:
            ratio = min(max_size / img.width, max_size / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to base64
        output_buf = io.BytesIO()
        img.save(output_buf, format="PNG")
        output_buf.seek(0)
        img_base64 = base64.b64encode(output_buf.read()).decode("utf-8")
        buf.close()
        output_buf.close()

        return {
            "success": True,
            "image_base64": img_base64,
            "bbox": {
                "min_lon": lon_min,
                "max_lon": lon_max,
                "min_lat": lat_min,
                "max_lat": lat_max,
            },
        }

    except Exception as e:
        plt.close("all")
        return {"success": False, "error": f"Visualization failed: {str(e)}"}


def visualize_comparison(
    predicted_geojson: Dict[str, Any],
    ground_truth_geojson: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    padding: float = 1.5,
) -> Dict[str, Any]:
    """
    Visualize predicted (and optionally ground truth) GeoJSON on OSM tiles.

    Renders both boundaries on the same map:
      - Green = predicted / extracted boundary
      - Blue  = ground truth boundary (if provided)

    Saves the result as a PNG file. Useful for quick visual verification
    after running the pipeline.

    Args:
        predicted_geojson: Extracted GeoJSON Feature (MultiPolygon or Polygon).
        ground_truth_geojson: Optional ground truth GeoJSON Feature.
        output_path: Where to save the PNG. If None, saves to "results/comparison.png".
        title: Optional title for the plot.
        padding: Padding around boundaries as fraction of combined extent.

    Returns:
        Dict with "success", "output_path", and optionally "error".
    """
    try:
        plt.close("all")

        # Build GeoDataFrames
        pred_geom = shape(predicted_geojson["geometry"])
        pred_gdf = gpd.GeoDataFrame({"geometry": [pred_geom]}, crs="EPSG:4326")

        gt_gdf = None
        if ground_truth_geojson:
            gt_geom = shape(ground_truth_geojson["geometry"])
            gt_gdf = gpd.GeoDataFrame({"geometry": [gt_geom]}, crs="EPSG:4326")

        # Compute combined extent (union of both shapes for padding)
        all_shapes = [pred_geom]
        if gt_gdf is not None:
            all_shapes.append(gt_geom)
        combined = unary_union(all_shapes)
        combined_gdf = gpd.GeoDataFrame({"geometry": [combined]}, crs="EPSG:4326")

        # Reproject to Web Mercator
        pred_merc = pred_gdf.to_crs(epsg=3857)
        combined_merc = combined_gdf.to_crs(epsg=3857)
        gt_merc = gt_gdf.to_crs(epsg=3857) if gt_gdf is not None else None

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))

        # Plot ground truth first (blue, underneath)
        if gt_merc is not None:
            gt_merc.plot(ax=ax, facecolor="blue", edgecolor="blue", alpha=0.15, linewidth=2)
            gt_merc.boundary.plot(ax=ax, color="blue", linewidth=2.5)

        # Plot predicted (green, on top)
        pred_merc.plot(ax=ax, facecolor="green", edgecolor="green", alpha=0.15, linewidth=2)
        pred_merc.boundary.plot(ax=ax, color="green", linewidth=2.5)

        # Set extent with padding
        bounds = combined_merc.total_bounds  # [minx, miny, maxx, maxy]
        minx, miny, maxx, maxy = bounds
        x_pad = (maxx - minx) * padding
        y_pad = (maxy - miny) * padding
        ax.set_xlim(minx - x_pad, maxx + x_pad)
        ax.set_ylim(miny - y_pad, maxy + y_pad)

        # Add OSM basemap
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        # Legend
        legend_handles = [
            mpatches.Patch(facecolor="green", edgecolor="green", alpha=0.4, label="Extracted"),
        ]
        if gt_merc is not None:
            legend_handles.insert(
                0, mpatches.Patch(facecolor="blue", edgecolor="blue", alpha=0.4, label="Ground Truth")
            )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=12)

        # Title
        if title:
            ax.set_title(title, fontsize=14, pad=10)
        elif gt_merc is not None:
            ax.set_title("Extracted vs Ground Truth", fontsize=14, pad=10)
        else:
            ax.set_title("Extracted Boundary", fontsize=14, pad=10)

        ax.set_axis_off()
        plt.tight_layout()

        # Save
        if output_path is None:
            Path("results").mkdir(exist_ok=True)
            output_path = "results/comparison.png"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"Visualization saved: {output_path}")
        return {"success": True, "output_path": output_path}

    except Exception as e:
        plt.close("all")
        return {"success": False, "error": f"Comparison visualization failed: {e}"}


# Tool definitions for LLM function calling
VISUALIZATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "visualize_geojson_boundary",
            "description": """Visualize a GeoJSON boundary on an OpenStreetMap basemap.

Use this after transforming coordinates to verify the boundary aligns correctly
with real-world features (roads, rivers, buildings).

PADDING (optional):
- 1.0 (default): Shows boundary with 100% extra space on each side
- 0.5: Tighter view with 50% padding
- 2.0: Wider view with more context""",
            "parameters": {
                "type": "object",
                "properties": {
                    "geojson_data": {
                        "type": "object",
                        "description": "GeoJSON Feature with 'geometry' containing Polygon/MultiPolygon",
                    },
                    "padding": {
                        "type": "number",
                        "description": "Padding around boundary as fraction of boundary size. Default: 1.0",
                    },
                },
                "required": ["geojson_data"],
            },
        },
    },
]
