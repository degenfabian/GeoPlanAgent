"""Render predicted/GT boundaries on OSM basemap tiles."""

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
    """Render the GeoJSON polygon on an OSM basemap. Returns success/image_base64/bbox."""
    try:
        plt.close("all")

        geometry = geojson_data["geometry"]
        geom_type = geometry["type"]

        if geom_type not in ("Polygon", "MultiPolygon"):
            return {
                "success": False,
                "error": f"Unsupported geometry type: {geom_type}",
            }

        geom = shape(geometry)
        gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")

        bounds = gdf.total_bounds
        lon_min, lat_min, lon_max, lat_max = bounds

        if lon_min == lon_max or lat_min == lat_max:
            return {"success": False, "error": "No valid extent found in GeoJSON"}

        # Web Mercator for contextily.
        gdf_mercator = gdf.to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(14, 12))
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
    """Render predicted (green) and optional GT (blue) on an OSM basemap; save PNG."""
    try:
        plt.close("all")

        pred_geom = shape(predicted_geojson["geometry"])
        pred_gdf = gpd.GeoDataFrame({"geometry": [pred_geom]}, crs="EPSG:4326")

        gt_gdf = None
        if ground_truth_geojson:
            gt_geom = shape(ground_truth_geojson["geometry"])
            gt_gdf = gpd.GeoDataFrame({"geometry": [gt_geom]}, crs="EPSG:4326")

        all_shapes = [pred_geom]
        if gt_gdf is not None:
            all_shapes.append(gt_geom)
        combined = unary_union(all_shapes)
        combined_gdf = gpd.GeoDataFrame({"geometry": [combined]}, crs="EPSG:4326")

        pred_merc = pred_gdf.to_crs(epsg=3857)
        combined_merc = combined_gdf.to_crs(epsg=3857)
        gt_merc = gt_gdf.to_crs(epsg=3857) if gt_gdf is not None else None

        fig, ax = plt.subplots(figsize=(14, 12))

        if gt_merc is not None:
            gt_merc.plot(ax=ax, facecolor="blue", edgecolor="blue", alpha=0.15, linewidth=2)
            gt_merc.boundary.plot(ax=ax, color="blue", linewidth=2.5)

        pred_merc.plot(ax=ax, facecolor="green", edgecolor="green", alpha=0.15, linewidth=2)
        pred_merc.boundary.plot(ax=ax, color="green", linewidth=2.5)

        minx, miny, maxx, maxy = combined_merc.total_bounds
        x_pad = (maxx - minx) * padding
        y_pad = (maxy - miny) * padding
        ax.set_xlim(minx - x_pad, maxx + x_pad)
        ax.set_ylim(miny - y_pad, maxy + y_pad)

        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        legend_handles = [
            mpatches.Patch(facecolor="green", edgecolor="green", alpha=0.4, label="Extracted"),
        ]
        if gt_merc is not None:
            legend_handles.insert(
                0, mpatches.Patch(facecolor="blue", edgecolor="blue", alpha=0.4, label="Ground Truth")
            )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=12)

        if title:
            ax.set_title(title, fontsize=14, pad=10)
        elif gt_merc is not None:
            ax.set_title("Extracted vs Ground Truth", fontsize=14, pad=10)
        else:
            ax.set_title("Extracted Boundary", fontsize=14, pad=10)

        ax.set_axis_off()
        plt.tight_layout()

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
