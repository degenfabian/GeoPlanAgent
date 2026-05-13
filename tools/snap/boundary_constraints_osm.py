"""Idea-A boundary-text constraint refiner — OSM/osmnx variant.

Same public API as `tools/snap/boundary_constraints.py`, but uses OSM
(via osmnx) instead of OS Open Zoomstack for the named-feature lookup.

Why this variant exists:
  Zoomstack is cartographic-display data: small residential roads often
  have no name attribute in `roads_local` (they're not rendered with a
  label at zoom-level cartography). For boundary-text constraint snapping,
  we want EXHAUSTIVE road-name coverage — and OSM provides that, at the
  cost of slightly noisier centerlines (community-mapped, not OS-grade).

This module is an A/B comparison target — not currently the production
default. Runs offline against cached v18 predictions; writes to
`overnight/idea_a_v18_results_osm.json` (separate from the Zoomstack
results file).
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, Point
from shapely.ops import unary_union, nearest_points, transform

# Reuse helpers from the Zoomstack module
from tools.snap.boundary_constraints import (
    _get_transformers, _to_bng, _to_wgs, _bbox_pad_wgs, _normalize_name,
    _names_match, _snap_vertices_to_line,
)


def _find_named_lines_osm(name: str, center_lat: float, center_lon: float,
                            radius_m: float = 1500,
                            include_waterways: bool = False) -> List[LineString]:
    """Query OSM for line features within `radius_m` of (lat, lon) whose
    name matches `name`. Returns LineStrings in BNG / EPSG:27700.

    Used both for road snapping (highway= tag) and river snapping
    (waterway= tag). Pass include_waterways=True to query water linestrings
    instead of highways.
    """
    try:
        import osmnx as ox
    except ImportError:
        return []

    tags = {"waterway": True} if include_waterways else {"highway": True}
    try:
        # features_from_point returns a GeoDataFrame in WGS84
        gdf = ox.features_from_point((center_lat, center_lon), tags=tags,
                                       dist=int(radius_m))
    except Exception:
        return []
    if gdf is None or len(gdf) == 0:
        return []

    name_col = None
    for col in ("name", "name:en"):
        if col in gdf.columns:
            name_col = col
            break
    if name_col is None:
        return []

    # Fuzzy match by name
    matched = gdf[gdf[name_col].apply(
        lambda x: _names_match(name, str(x)) if x else False
    )]
    if len(matched) == 0:
        return []

    # Convert WGS84 → BNG; flatten MultiLineStrings
    out = []
    bng, _ = _get_transformers()
    for geom in matched.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            geom_bng = _to_bng(geom)
            out.append(geom_bng)
        elif geom.geom_type == "MultiLineString":
            geom_bng = _to_bng(geom)
            out.extend(list(geom_bng.geoms))
        # Polygon waterways (lakes, etc.) — convert to boundary lines
        elif geom.geom_type in ("Polygon", "MultiPolygon"):
            geom_bng = _to_bng(geom)
            if geom_bng.geom_type == "Polygon":
                out.append(LineString(geom_bng.exterior.coords))
            else:
                for p in geom_bng.geoms:
                    out.append(LineString(p.exterior.coords))
    return out


def apply_constraints_osm(predicted_geom, constraints: List[Dict[str, Any]],
                            verbose: bool = False) -> Dict[str, Any]:
    """OSM-backed constraint application. Same return shape as
    `tools.snap.boundary_constraints.apply_constraints`."""
    if predicted_geom is None or predicted_geom.is_empty:
        return {"refined_geom": predicted_geom, "applied": 0, "skipped": 0,
                "rejected_guard": 0, "log": []}

    if predicted_geom.geom_type == "Polygon":
        polys_bng = [_to_bng(predicted_geom)]
    elif predicted_geom.geom_type == "MultiPolygon":
        polys_bng = [_to_bng(p) for p in predicted_geom.geoms]
    else:
        return {"refined_geom": predicted_geom, "applied": 0, "skipped": 0,
                "rejected_guard": 0,
                "log": [f"unsupported geom: {predicted_geom.geom_type}"]}

    centroid = predicted_geom.centroid

    log = []
    applied = 0
    skipped = 0
    rejected_guard = 0

    for c in constraints:
        ctype = c.get("type", "")
        target = c.get("target", "")
        if not target:
            log.append({"type": ctype, "target": target, "outcome": "no_target"})
            skipped += 1
            continue

        # Decide tag-set
        if ctype in ("follows_road", "along_centerline", "bounded_by"):
            lines = _find_named_lines_osm(
                target, centroid.y, centroid.x, radius_m=1500
            )
            data_source = "osm:highway"
        elif ctype == "touches_river":
            lines = _find_named_lines_osm(
                target, centroid.y, centroid.x, radius_m=1500,
                include_waterways=True
            )
            data_source = "osm:waterway"
        else:
            log.append({"type": ctype, "target": target,
                         "outcome": "type_not_supported"})
            skipped += 1
            continue

        if not lines:
            log.append({"type": ctype, "target": target,
                         "outcome": "no_match_in_OSM", "source": data_source})
            skipped += 1
            continue

        new_polys = []
        any_changed = False
        for poly_bng in polys_bng:
            old_area = poly_bng.area
            new_poly, n_moved, n_total = _snap_vertices_to_line(
                poly_bng, lines, max_snap_m=30.0
            )
            new_area = new_poly.area if new_poly else 0
            move_frac = n_moved / max(n_total, 1)
            if new_area <= 0:
                new_polys.append(poly_bng)
                rejected_guard += 1
                log.append({"type": ctype, "target": target,
                             "outcome": "rejected_zero_area",
                             "n_moved": n_moved, "n_total": n_total,
                             "source": data_source})
            elif old_area > 0 and abs(new_area - old_area) / old_area > 0.25:
                new_polys.append(poly_bng)
                rejected_guard += 1
                log.append({"type": ctype, "target": target,
                             "outcome": "rejected_area_drift",
                             "old_area_m2": int(old_area),
                             "new_area_m2": int(new_area),
                             "n_moved": n_moved, "n_total": n_total,
                             "source": data_source})
            elif move_frac < 0.05:
                new_polys.append(poly_bng)
                skipped += 1
                log.append({"type": ctype, "target": target,
                             "outcome": "no_vertices_in_snap_range",
                             "n_moved": n_moved, "n_total": n_total,
                             "source": data_source})
            else:
                new_polys.append(new_poly)
                any_changed = True
                applied += 1
                log.append({"type": ctype, "target": target,
                             "outcome": "applied",
                             "n_moved": n_moved, "n_total": n_total,
                             "area_change_pct": round(
                                 (new_area - old_area) / max(old_area, 1) * 100, 1
                             ),
                             "source": data_source,
                             "n_lines_matched": len(lines)})

        if any_changed:
            polys_bng = new_polys

    if len(polys_bng) == 1:
        refined_bng = polys_bng[0]
    else:
        refined_bng = MultiPolygon(polys_bng)
    refined_wgs = _to_wgs(refined_bng)

    return {
        "refined_geom": refined_wgs,
        "applied": applied,
        "skipped": skipped,
        "rejected_guard": rejected_guard,
        "log": log,
    }
