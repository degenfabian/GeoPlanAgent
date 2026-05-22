"""Tolerant GeoJSON loaders for paths on disk.

The strict, schema-validating GeoJSON helper lives in
:mod:`tools.metrics.geojson` (`geojson_to_shape`, `calculate_iou`,
`calculate_positioning_error_m`) — that one raises ``ValueError`` on
malformed input and is used by the official metrics pipeline.

This module provides the *tolerant* counterpart used by scripts and
ad-hoc analysis tools that just want "the polygon from this file, or
None if it can't be parsed":

* Accepts either a ``Feature``, a ``FeatureCollection`` (unioned), or a
  raw ``Geometry``.
* Calls ``buffer(0)`` to repair invalid geometries silently.
* Returns ``None`` on any error (no exceptions).
* Path-based — load + parse + shapely in one call.

Consolidated 2026-05-22 from inline duplicates in
``scripts/monitor_lucky_shot.py`` and ``scripts/sigma_signal_analysis.py``
plus ad-hoc copies in other scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Union


def load_geojson_polygon(path: Union[str, Path]):
    """Load a GeoJSON file as a shapely polygon (or MultiPolygon).

    Handles three top-level shapes:
      * ``Feature``           → returns ``shape(d['geometry'])``
      * ``FeatureCollection`` → unioned shape of every feature's geometry
      * raw ``Geometry``      → returns ``shape(d)`` directly

    Always applies ``buffer(0)`` to repair invalid geometries silently.
    Returns ``None`` on any error (file missing, parse failure, empty
    feature collection, irreparably invalid).
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    return _shape_from_geojson_dict(d)


def _shape_from_geojson_dict(d):
    """Internal: take a parsed GeoJSON dict, return shapely geometry or None."""
    if not isinstance(d, dict):
        return None
    try:
        from shapely.geometry import shape
        from shapely.ops import unary_union
    except Exception:
        return None
    try:
        t = d.get("type")
        if t == "Feature":
            geom = d.get("geometry")
            if not geom:
                return None
            s = shape(geom)
        elif t == "FeatureCollection":
            feats = d.get("features") or []
            geoms = []
            for f in feats:
                g = f.get("geometry") if isinstance(f, dict) else None
                if g:
                    geoms.append(shape(g))
            if not geoms:
                return None
            s = unary_union(geoms)
        else:
            # Assume it's already a raw Geometry dict.
            s = shape(d)
        if not s.is_valid:
            s = s.buffer(0)
        return s if s.is_valid and not s.is_empty else None
    except Exception:
        return None


def centroid_latlon(path: Union[str, Path]) -> Optional[Tuple[float, float]]:
    """Return ``(lat, lon)`` of the polygon's centroid, or ``None`` on failure.

    Thin wrapper over :func:`load_geojson_polygon` for the common
    "where is this polygon roughly?" use case in scripts.
    """
    g = load_geojson_polygon(path)
    if g is None:
        return None
    try:
        c = g.centroid
        return float(c.y), float(c.x)
    except Exception:
        return None
