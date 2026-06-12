"""Scoring of predicted boundaries against ground truth."""

import json
from typing import Dict, Any, Optional
from pathlib import Path
from shapely.geometry import shape, Polygon, MultiPolygon
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import contextily as ctx
from shapely.ops import unary_union
from geoplanagent.utils import haversine_km


def load_geojson(geojson_path: str) -> Optional[Dict[str, Any]]:
    return json.loads(Path(geojson_path).read_text())


def validate_geojson_format(geojson_data: Dict[str, Any]) -> tuple[bool, str]:
    """Checks the GeoJSON format is valid i.e. a Feature with Polygon or MultiPolygon geometry."""
    if geojson_data.get("type") != "Feature":
        return False, f"Expected 'Feature', got '{geojson_data.get('type')}'"
    geometry = geojson_data.get("geometry")
    if not geometry:
        return False, "Missing 'geometry' field"
    geom_type = geometry.get("type")
    if geom_type not in ["MultiPolygon", "Polygon"]:
        return False, f"Expected 'MultiPolygon' or 'Polygon', got '{geom_type}'"
    return True, ""


def geojson_to_shape(geojson_data: Dict[str, Any]) -> Polygon | MultiPolygon:
    """GeoJSON Feature -> valid shapely geometry, repairing invalid polygons.

    Raises ValueError on anything outside the benchmark's output contract
    (a Feature with Polygon/MultiPolygon geometry), on conversion errors,
    and when the geometry is empty or irreparably invalid — so a returned
    shape is always a valid, non-empty boundary.
    """
    is_valid, error_msg = validate_geojson_format(geojson_data)
    if not is_valid:
        raise ValueError(f"Invalid GeoJSON format: {error_msg}")

    # Only the shapely calls live in the try: their exceptions get
    # translated to the contract's ValueError. Our own checks come after.
    try:
        s = shape(geojson_data["geometry"])
        if not s.is_valid:
            # Zero-width buffer: the standard shapely repair for
            # self-intersecting polygons, which mask tracing produces.
            s = s.buffer(0)
    except Exception as e:
        raise ValueError(f"Error converting GeoJSON to shape: {e}") from e

    if not s.is_valid:
        raise ValueError("geometry invalid even after buffer(0) repair")
    if s.is_empty:
        raise ValueError("geometry is empty")
    return s


def calculate_spatial_metrics(
    ground_truth_geojson: Dict[str, Any], predicted_geojson: Dict[str, Any]
) -> Dict[str, Any]:
    """Per-case scores stored in metrics.json: iou, precision, recall,
    f1_score (+ validity flags), with positioning_error_m added below.

    Overlap is computed with shapely on raw WGS84 coordinates. Planar
    degrees distort areas, but the distortion is locally affine and hits
    numerator and denominator alike, so the IoU/precision/recall ratios
    are unaffected (verified <2e-4 against EPSG:27700). Invalid polygons
    are repaired with buffer(0) in geojson_to_shape.
    """
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
        metrics["valid_ground_truth"] = True
    except ValueError as e:
        metrics["validation_error"] = f"Ground truth error: {e}"
        return metrics

    try:
        pred_shape = geojson_to_shape(predicted_geojson)
        metrics["valid_prediction"] = True
    except ValueError as e:
        metrics["validation_error"] = f"Prediction error: {e}"
        return metrics

    try:
        gt_area = gt_shape.area
        pred_area = pred_shape.area
        intersection_area = gt_shape.intersection(pred_shape).area
        union_area = gt_shape.union(pred_shape).area

        iou = intersection_area / union_area if union_area > 0 else 0.0
        precision = intersection_area / pred_area if pred_area > 0 else 0.0
        recall = intersection_area / gt_area if gt_area > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics.update(
            {
                "iou": float(iou),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                # Centroid distance in metres (haversine, WGS84 centroids).
                "positioning_error_m": haversine_km(
                    gt_shape.centroid.y, gt_shape.centroid.x,
                    pred_shape.centroid.y, pred_shape.centroid.x,
                ) * 1000.0,
            }
        )

    except Exception as e:
        metrics["validation_error"] = f"Calculation error: {e}"

    return metrics


# Fraction of the combined bounding box added on each side of the plot.
_VIZ_PADDING = 1.5


def visualize_comparison(
    predicted_geojson: Dict[str, Any],
    ground_truth_geojson: Optional[Dict[str, Any]] = None,
    *,
    output_path: str,
) -> None:
    """Render predicted (green) and optional GT (blue) on an OSM basemap; save PNG.

    Raises on render failure so the caller's stub-image fallback can fire.
    """
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
    x_pad = (maxx - minx) * _VIZ_PADDING
    y_pad = (maxy - miny) * _VIZ_PADDING
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

    if gt_merc is not None:
        ax.set_title("Extracted vs Ground Truth", fontsize=14, pad=10)
    else:
        ax.set_title("Extracted Boundary", fontsize=14, pad=10)

    ax.set_axis_off()
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Visualization saved: {output_path}")
