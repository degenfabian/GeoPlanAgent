"""Scoring of predicted boundaries against ground truth."""

import json
from typing import Dict, Any
from pathlib import Path
from shapely.errors import GEOSException
from shapely.geometry import shape, Polygon, MultiPolygon
from geoplanagent.utils import haversine_km


def load_geojson(geojson_path: str) -> Dict[str, Any]:
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
    """
    Converts a GeoJSON to a shapely geometry, repairing invalid polygons.

    Raises ValueError on:
    - Anything outside the benchmark's expected output format (a Feature with Polygon/MultiPolygon geometry)
    - Conversion errors
    - When the geometry is empty or irreparably invalid
    """
    is_valid, error_msg = validate_geojson_format(geojson_data)
    if not is_valid:
        raise ValueError(f"Invalid GeoJSON format: {error_msg}")

    try:
        geom = shape(geojson_data["geometry"])
        if not geom.is_valid:
            # Zero-width buffer: the standard shapely repair for
            # self-intersecting polygons, which mask tracing produces.
            geom = geom.buffer(0)
    except Exception as e:
        raise ValueError(f"Error converting GeoJSON to shape: {e}") from e

    if not geom.is_valid:
        raise ValueError("geometry invalid even after buffer(0) repair")
    if geom.is_empty:
        raise ValueError("geometry is empty")
    return geom


def calculate_spatial_metrics(
    ground_truth_geojson: Dict[str, Any], predicted_geojson: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculates the spatial metrics: IoU, precision, recall,
    and positioning error in metres.
    """
    metrics = {
        "valid_ground_truth": False,
        "valid_prediction": False,
        "validation_error": None,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
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

        metrics.update(
            {
                "iou": float(iou),
                "precision": float(precision),
                "recall": float(recall),
                # Centroid distance in metres (haversine, WGS84 centroids).
                "positioning_error_m": haversine_km(
                    gt_shape.centroid.y, gt_shape.centroid.x,
                    pred_shape.centroid.y, pred_shape.centroid.x,
                ) * 1000.0,
            }
        )

    except GEOSException as e:
        # Overlay ops can fail on pathological geometry pairs; that is a
        # scoring verdict. Anything else (a code defect) propagates loudly.
        metrics["validation_error"] = f"Calculation error: {e}"

    return metrics
