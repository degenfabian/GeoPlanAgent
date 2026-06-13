"""Scoring of predicted boundaries against ground truth."""

import json
from itertools import combinations
from typing import Any, Dict, Sequence
from pathlib import Path

import numpy as np
from shapely.geometry import shape, Polygon, MultiPolygon

from geoplanagent.utils import haversine_m


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
        "recall": intersection / gt.area if gt.area else 0.0,#
        # Computes the haversine distance (see utils.py) between the two polygon centroids in metres.
        "centroid_distance_m": haversine_m(
            gt.centroid.y, gt.centroid.x, pred.centroid.y, pred.centroid.x
        ),
    }


def feret_diameter_m(geom) -> float:
    """Feret diameter: the largest distance (metres) between any two points of
    the geometry — its widest span. Computed over the convex hull."""
    hull = geom.convex_hull
    if hull.geom_type == "Point":
        return 0.0
    pts = list(hull.exterior.coords)[:-1] if hull.geom_type == "Polygon" else list(hull.coords)
    return max(
        (
            # shapely coords are (lon, lat); haversine_m takes (lat, lon).
            haversine_m(lat1, lon1, lat2, lon2)
            for (lon1, lat1), (lon2, lat2) in combinations(pts, 2)
        ),
        default=0.0,
    )


def aggregate_stats(values: Sequence[float]) -> Dict[str, float]:
    """Aggregates the statistics: mean, median, std, min, max of a sequence of values."""
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def aggregate_spatial_metrics(ious, centroid_distances, feret_diameters) -> Dict[str, float]:
    """Summarise per-case spatial metrics across a set of cases.

    Inputs are three parallel lists, one entry per case:
        ious                IoU of each prediction against its ground truth
        centroid_distances  gt-centroid to prediction-centroid distance in metres
        feret_diameters     GT Feret diameter (widest span) in metres

    Returns the paper's headline aggregates:
        n_cases     number of cases
        pct_grt_0   % of cases with IoU > 0 (any overlap with the ground truth)
        mean_IoU    mean IoU
        median_IoU  median IoU
        pct_grt_08  % of cases with IoU >= 0.8 (high-quality matches)
        median_centroid_distance_m  median centroid distance over all cases
        acc_01d     % of cases whose centroid distance is within 0.1 x the GT Feret
                    diameter
    """
    ious = np.asarray(ious, float)
    centroid_distances = np.asarray(
        [d if d is not None else np.inf for d in centroid_distances], float
    ) # None values are counted as a miss
    feret_diameters = np.asarray(feret_diameters, float)
    return {
        "n_cases": len(ious),
        "pct_grt_0": 100 * np.mean(ious > 0),
        "mean_IoU": float(np.mean(ious)),
        "median_IoU": float(np.median(ious)),
        "pct_grt_08": 100 * np.mean(ious >= 0.8),
        "median_centroid_distance_m": float(np.median(centroid_distances)),
        "acc_01d": 100 * np.mean(centroid_distances <= 0.1 * feret_diameters),
    }
