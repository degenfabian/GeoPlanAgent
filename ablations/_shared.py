"""Helpers shared across ablation harnesses.

GT-centroid extraction + nearest-part scoring are used by every harness
that compares a picked (lat, lon) against a planning-boundary GT
geometry — currently :mod:`ablations.locate_only_eval` and
:mod:`ablations.locate_vlm_direct`, with the future locate-vs-VLM
aggregation script as a third caller.

Keeping these in one place prevents byte-drift between harnesses; if
the metric definition ever changes, it changes once.
"""
from __future__ import annotations

from typing import Optional

from tools.geo.coords import haversine_km
from tools.metrics.geojson import geojson_to_shape


# Canonical CSV column schema for any per-case scoring CSV produced by
# the ablation harnesses. Holding it here lets the aggregation step
# union all configs' CSVs without column-by-column reconciliation.
# Fields a particular approach has no value for (e.g. VLM-direct has no
# sigma_m / confidence / verified_inside_admin_region) stay empty.
CSV_FIELDNAMES = [
    "case",
    "err_km",
    "picked_lat",
    "picked_lon",
    "picked_source",
    "confidence",
    "sigma_m",
    "verified_inside_admin_region",
    "n_gt_parts",
    "evidence",
    "error",
]


def gt_part_centroids(gt_geojson: dict) -> list[tuple[float, float]]:
    """Return one (lat, lon) per Polygon part of the GT geometry.

    Multi-area planning documents have MultiPolygon GTs; the first-pick
    scoring takes the MIN haversine distance over part centroids, so a
    multi-area case is scored by whichever component the agent landed
    nearest to.

    Returns an empty list when the geojson can't be parsed or the
    geometry can't be repaired (``geojson_to_shape`` returns None).
    """
    shape = geojson_to_shape(gt_geojson)
    if shape is None:
        return []
    polys = list(shape.geoms) if hasattr(shape, "geoms") else [shape]
    return [(p.centroid.y, p.centroid.x) for p in polys]


def nearest_part_err_km(
    pick_lat: float,
    pick_lon: float,
    centroids: list[tuple[float, float]],
) -> Optional[float]:
    """Min haversine km from a picked (lat, lon) to any GT-part centroid.

    Returns ``None`` when ``centroids`` is empty (no parsable GT), so
    the caller can record the failure instead of silently scoring 0 km.
    """
    if not centroids:
        return None
    return min(
        haversine_km(pick_lat, pick_lon, c_lat, c_lon)
        for c_lat, c_lon in centroids
    )


__all__ = [
    "CSV_FIELDNAMES",
    "gt_part_centroids",
    "nearest_part_err_km",
]
