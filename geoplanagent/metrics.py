"""Scoring of predicted boundaries against ground truth."""

import csv
import json
from itertools import combinations
from typing import Any, Dict, Sequence
from pathlib import Path

import numpy as np
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union

from geoplanagent.utils import haversine_m, aggregate_pages_to_cases
from geoplanagent.paths import SAM_KFOLD_PREDICTIONS


def load_geojson(geojson_path: str) -> Dict[str, Any]:
    return json.loads(Path(geojson_path).read_text())


def load_case_ground_truth(case_dir) -> Dict[str, Any] | None:
    """Load a case's ground-truth boundary as a single GeoJSON Feature.

    Most cases store one GeoJSON. The "merged" cases store each constituent
    sub-area as its own file — the true boundary is their union, so every
    *.geojson in the folder is dissolved into one (Multi)Polygon.

    Returns None when the folder holds no GeoJSON.
    """
    gt_files = sorted(Path(case_dir).glob("*.geojson"))
    if not gt_files:
        return None
    if len(gt_files) == 1:
        return load_geojson(str(gt_files[0]))  # common case, unchanged
    merged = unary_union([geojson_to_shape(load_geojson(str(path))) for path in gt_files])
    return {"type": "Feature", "properties": {}, "geometry": mapping(merged)}


def case_dirs(run_dir) -> list[Path]:
    """The per-case subdirs of a benchmark run dir that hold a metrics.json."""
    return sorted(
        case_dir
        for case_dir in Path(run_dir).iterdir()
        if case_dir.is_dir() and (case_dir / "metrics.json").exists()
    )


def load_run_metrics(run_dir) -> Dict[str, dict]:
    """{case_name: metrics dict} for every case under a benchmark run dir."""
    return {
        case_dir.name: json.loads((case_dir / "metrics.json").read_text())
        for case_dir in case_dirs(run_dir)
    }


def worker_first(case_metrics: dict) -> tuple[float, float | None]:
    """Pre-critic (iou, err_m) for one case.

    The benchmark ran with the critic enabled; metrics.json keeps the
    pre-critic result in worker_first_*. Where worker_first_iou is null
    the critic never changed anything, so the final value is already the
    pre-critic value.
    """
    if case_metrics.get("worker_first_iou") is None:
        return case_metrics["iou"], case_metrics.get("centroid_distance_m")
    wf_metrics = case_metrics.get("worker_first_metrics") or {}
    return case_metrics["worker_first_iou"], wf_metrics.get("centroid_distance_m")


def pre_critic_iou_by_case(run_dir) -> Dict[str, float]:
    """{case folder: pre-critic (worker-first) IoU} for a run. Shared by Figure 4's table and figure."""
    return {
        folder: worker_first(case_metrics)[0]
        for folder, case_metrics in load_run_metrics(run_dir).items()
    }


def load_sam_iou_by_case() -> Dict[str, float]:
    """Per-case SAM3-LoRA semantic IoU from the cached k-fold predictions
    (SAM_KFOLD_PREDICTIONS); multi-page cases are averaged to one score, matching
    the end-to-end aggregation."""
    data = json.loads(SAM_KFOLD_PREDICTIONS.read_text())
    return aggregate_pages_to_cases({key: entry["sem_iou"] for key, entry in data.items()})


def seg_iou_by_case(csv_path: Path) -> np.ndarray:
    """Per-case IoU from a SAM/VLM-seg results.csv (per-page rows averaged to
    cases). The `filename` column (a per-page image) is the unique page key so
    multi-page cases survive to be averaged."""
    per_page = {
        Path(row["filename"]).stem: float(row["iou"])
        for row in csv.DictReader(open(csv_path))
        if row.get("iou") not in (None, "")
    }
    return np.asarray(list(aggregate_pages_to_cases(per_page).values()))


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
    except Exception as error:
        raise ValueError(f"Error converting GeoJSON to shape: {error}") from error

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
    points = list(hull.exterior.coords)[:-1] if hull.geom_type == "Polygon" else list(hull.coords)
    return max(
        (
            # shapely coords are (lon, lat); haversine_m takes (lat, lon).
            haversine_m(lat1, lon1, lat2, lon2)
            for (lon1, lat1), (lon2, lat2) in combinations(points, 2)
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
        [distance if distance is not None else np.inf for distance in centroid_distances], float
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
