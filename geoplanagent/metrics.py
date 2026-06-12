"""Scoring of predicted boundaries against ground truth.

calculate_spatial_metrics() produces the per-case numbers stored in
metrics.json (iou, precision, recall, f1_score, positioning_error_m):
geometries are loaded from GeoJSON (load_geojson + geojson_to_shape,
with buffer(0) repair of invalid polygons), overlap is computed with
shapely on raw WGS84 coordinates (planar degrees — locally affine, so
the IoU ratio is unaffected; verified <2e-4 vs projected), and centroid
error is the haversine distance between polygon centroids in metres.

visualize_comparison() renders the predicted-vs-GT overlay on an OSM
basemap (geopandas + contextily) that the benchmark saves per case as
viz_comparison.png.
"""

import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from shapely.geometry import shape, Polygon, MultiPolygon
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import contextily as ctx
from shapely.ops import unary_union


log = logging.getLogger(__name__)


def load_geojson(geojson_path: str) -> Optional[Dict[str, Any]]:
    path = Path(geojson_path)
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def validate_geojson_format(geojson_data: Dict[str, Any]) -> tuple[bool, str]:
    if geojson_data.get("type") != "Feature":
        return False, f"Expected 'Feature', got '{geojson_data.get('type')}'"
    geometry = geojson_data.get("geometry")
    if not geometry:
        return False, "Missing 'geometry' field"
    geom_type = geometry.get("type")
    if geom_type not in ["MultiPolygon", "Polygon"]:
        return False, f"Expected 'MultiPolygon' or 'Polygon', got '{geom_type}'"
    return True, ""




def geojson_to_shape(geojson_data: Dict[str, Any]) -> Optional[Polygon | MultiPolygon]:
    is_valid, error_msg = validate_geojson_format(geojson_data)
    if not is_valid:
        raise ValueError(f"Invalid GeoJSON format: {error_msg}")

    try:
        s = shape(geojson_data["geometry"])
        if not s.is_valid:
            s = s.buffer(0)
        return s if s.is_valid else None
    except Exception as e:
        raise ValueError(f"Error converting GeoJSON to shape: {e}")


def calculate_positioning_error_m(pred_geojson, gt_geojson):
    from geoplanagent.utils import haversine_km
    try:
        pred_shape = geojson_to_shape(pred_geojson)
        gt_shape = geojson_to_shape(gt_geojson)
        if pred_shape is None or gt_shape is None:
            return None
        if pred_shape.is_empty or gt_shape.is_empty:
            return None
        pc, gc = pred_shape.centroid, gt_shape.centroid
        return haversine_km(gc.y, gc.x, pc.y, pc.x) * 1000.0
    except Exception:
        return None


def calculate_spatial_metrics(
    ground_truth_geojson: Dict[str, Any], predicted_geojson: Dict[str, Any]
) -> Dict[str, Any]:
    metrics = {
        "valid_ground_truth": False,
        "valid_prediction": False,
        "validation_error": None,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
    }

    try:
        gt_shape = geojson_to_shape(ground_truth_geojson)
        metrics["valid_ground_truth"] = gt_shape is not None and gt_shape.is_valid
    except ValueError as e:
        metrics["validation_error"] = f"Ground truth error: {e}"
        return metrics

    try:
        pred_shape = geojson_to_shape(predicted_geojson)
        metrics["valid_prediction"] = pred_shape is not None and pred_shape.is_valid
    except ValueError as e:
        metrics["validation_error"] = f"Prediction error: {e}"
        return metrics

    if not metrics["valid_ground_truth"] or not metrics["valid_prediction"]:
        metrics["validation_error"] = "Invalid geometry"
        return metrics

    try:
        gt_area = gt_shape.area
        pred_area = pred_shape.area
        intersection_area = gt_shape.intersection(pred_shape).area
        union_area = gt_shape.union(pred_shape).area

        iou = intersection_area / union_area if union_area > 0 else 0.0
        precision = intersection_area / pred_area if pred_area > 0 else 0.0
        recall = intersection_area / gt_area if gt_area > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics.update({
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "positioning_error_m": calculate_positioning_error_m(
                predicted_geojson, ground_truth_geojson
            ),
        })

    except Exception as e:
        metrics["validation_error"] = f"Calculation error: {e}"

    return metrics


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
