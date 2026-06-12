"""GT-centroid extraction + nearest-part scoring; shared by locate ablations."""
from __future__ import annotations

from typing import Optional

from geoplanagent.utils import haversine_km
from geoplanagent.metrics import geojson_to_shape


# Canonical per-case scoring CSV schema, shared across harnesses.
# Note: the older ``verified_inside_admin_region`` column was dropped when
# the field was removed from LocatePick (its production value was always
# False since la_check is disabled by default, and the LLM was setting it
# to True regardless — see the ablation hallucination analysis). The
# already-saved CSVs in ablations/locate_only_eval/<config>/locate_picks.csv
# still carry the column; readers should treat it as optional.
CSV_FIELDNAMES = [
    "case",
    "err_km",
    "picked_lat",
    "picked_lon",
    "picked_source",
    "confidence",
    "sigma_m",
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
