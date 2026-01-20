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
import io
import base64
from typing import Dict, Any

import geopandas as gpd
import contextily as ctx
from shapely.geometry import shape
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
