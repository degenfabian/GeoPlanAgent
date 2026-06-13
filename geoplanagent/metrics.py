"""Scoring of predicted boundaries against ground truth."""

import json
from typing import Dict, Any
from pathlib import Path
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
) -> Dict[str, float]:
    """ Computes IoU, precision, recall, and centroid distance (metres).

    Raises ValueError if either geometry can't be built or repaired.
    """
    gt = geojson_to_shape(ground_truth_geojson)
    pred = geojson_to_shape(predicted_geojson)

    intersection = gt.intersection(pred).area
    union = gt.union(pred).area
    return {
        "iou": intersection / union if union else 0.0,
        "precision": intersection / pred.area if pred.area else 0.0,
        "recall": intersection / gt.area if gt.area else 0.0,
        "centroid_distance_m": haversine_km(
            gt.centroid.y, gt.centroid.x, pred.centroid.y, pred.centroid.x
        )
        * 1000.0,
    }
