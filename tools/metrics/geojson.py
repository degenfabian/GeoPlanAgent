"""
GeoJSON Metrics - Calculate IoU and other spatial metrics for planning boundary extraction
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path
from shapely.geometry import shape, Polygon, MultiPolygon


def load_geojson(geojson_path: str) -> Optional[Dict[str, Any]]:
    """
    Load GeoJSON from file.

    Args:
        geojson_path: Path to GeoJSON file

    Returns:
        GeoJSON dict or None if file doesn't exist
    """
    path = Path(geojson_path)
    if not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)


def validate_geojson_format(geojson_data: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate that GeoJSON matches expected format: Feature with MultiPolygon.

    Args:
        geojson_data: GeoJSON dict to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
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
    """
    Convert GeoJSON Feature to Shapely geometry.

    Expected format: Feature with MultiPolygon or Polygon geometry.

    Args:
        geojson_data: GeoJSON dict (must be Feature with Polygon/MultiPolygon)

    Returns:
        Shapely Polygon or MultiPolygon, or None if invalid

    Raises:
        ValueError: If GeoJSON format is incorrect
    """
    is_valid, error_msg = validate_geojson_format(geojson_data)
    if not is_valid:
        raise ValueError(f"Invalid GeoJSON format: {error_msg}")

    try:
        geometry = geojson_data["geometry"]
        # shape() converts a GeoJSON geometry dict into a Shapely geometry object
        s = shape(geometry)

        # buffer(0) is a common trick to fix invalid geometries
        # It rebuilds the geometry, fixing issues like self-intersections,
        # duplicate points, or incorrectly wound rings
        if not s.is_valid:
            s = s.buffer(0)

        return s if s.is_valid else None

    except Exception as e:
        raise ValueError(f"Error converting GeoJSON to shape: {e}")


def calculate_iou(
    ground_truth: Polygon | MultiPolygon, prediction: Polygon | MultiPolygon
) -> float:
    """
    Calculate Intersection over Union (IoU) between two polygons.

    IoU = Area of Intersection / Area of Union

    Args:
        ground_truth: Ground truth polygon
        prediction: Predicted polygon

    Returns:
        IoU score between 0 and 1
    """
    try:
        if not ground_truth.is_valid:
            ground_truth = ground_truth.buffer(0)
        if not prediction.is_valid:
            prediction = prediction.buffer(0)

        intersection = ground_truth.intersection(prediction)
        union = ground_truth.union(prediction)

        if union.area == 0:
            return 0.0

        iou = intersection.area / union.area
        return float(iou)

    except Exception as e:
        print(f"Error calculating IoU: {e}")
        return 0.0


def calculate_positioning_error_m(pred_geojson, gt_geojson):
    """Haversine distance (meters) between centroids of predicted and GT polygons."""
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
    """
    Calculate comprehensive spatial metrics between ground truth and prediction.

    Args:
        ground_truth_geojson: Ground truth GeoJSON (Feature with MultiPolygon)
        predicted_geojson: Predicted GeoJSON (Feature with MultiPolygon/Polygon)

    Returns:
        Dict containing IoU and other spatial metrics, plus validation info
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

        intersection = gt_shape.intersection(pred_shape)
        union = gt_shape.union(pred_shape)

        intersection_area = intersection.area
        union_area = union.area

        # IoU: measures overall overlap relative to total area covered by either shape
        # Range: 0 (no overlap) to 1 (perfect match)
        iou = intersection_area / union_area if union_area > 0 else 0.0

        # Precision: of all area the model predicted, how much was correct?
        # Low precision = model predicted too much area (false positives)
        precision = intersection_area / pred_area if pred_area > 0 else 0.0

        # Recall: of all area that should be covered, how much did the model find?
        # Low recall = model missed area it should have predicted (false negatives)
        recall = intersection_area / gt_area if gt_area > 0 else 0.0

        # F1: harmonic mean of precision and recall
        # Balances both metrics - penalizes sacrificing one for the other
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics.update(
            {
                "iou": float(iou),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "positioning_error_m": calculate_positioning_error_m(
                    predicted_geojson, ground_truth_geojson
                ),
            }
        )

    except Exception as e:
        metrics["validation_error"] = f"Calculation error: {e}"

    return metrics


