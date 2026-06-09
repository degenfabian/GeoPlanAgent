"""IoU and related spatial metrics for predicted vs. GT planning boundaries."""

import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from shapely.geometry import shape, Polygon, MultiPolygon

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


def calculate_iou(
    ground_truth: Polygon | MultiPolygon, prediction: Polygon | MultiPolygon
) -> float:
    try:
        if not ground_truth.is_valid:
            ground_truth = ground_truth.buffer(0)
        if not prediction.is_valid:
            prediction = prediction.buffer(0)

        intersection = ground_truth.intersection(prediction)
        union = ground_truth.union(prediction)

        if union.area == 0:
            return 0.0

        return float(intersection.area / union.area)

    except Exception:
        log.warning("IoU computation failed, scoring 0", exc_info=True)
        return 0.0


def calculate_positioning_error_m(pred_geojson, gt_geojson):
    from tools.geo.coords import haversine_km
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


