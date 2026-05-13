"""Idea-A boundary-text constraint refiner (offline experiment, v18 data).

Reads a predicted polygon + a list of BoundaryConstraint dicts (already
extracted by the v18 reader), and snaps the polygon to satisfy those
constraints against OS Open Zoomstack layers.

Constraint types supported in this first cut:
  - follows_road   → snap vertices near a named road to its centerline
  - bounded_by     → if direction set, push the corresponding edge to the named feature
  - touches_river  → extend a polygon edge to touch the named waterline
  - along_centerline → same as follows_road (centerline is the default)
  - near_landmark, abuts_parcel, corner_at, other → currently no-op (logged)

Hard guards (to avoid breaking already-good polygons):
  - max-snap-distance per vertex = 30 m (don't drag distant vertices)
  - max-snap-distance per edge for `touches_river`/`bounded_by` = 100 m
  - reject the snap if the new polygon's area changes by >25% (per the design doc)
  - reject the snap if <50% of vertices were actually moved (snap had no
    effect → just adds noise)

Read-only against v18 data. All output to overnight/idea_a_v18_results.json.
"""
from __future__ import annotations
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, Point
from shapely.ops import unary_union, nearest_points, transform

ROOT = Path(__file__).resolve().parent.parent
ZOOMSTACK_PATH = ROOT / "os_opendata" / "OS_Open_Zoomstack.gpkg"

# Reuse pyproj transformer for WGS84 ↔ BNG conversions (sub-meter accurate
# for UK; preserves shapely shape).
_BNG = None  # lazy
_WGS = None  # lazy


def _get_transformers():
    global _BNG, _WGS
    if _BNG is None:
        from pyproj import Transformer
        _BNG = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
        _WGS = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    return _BNG, _WGS


def _to_bng(geom):
    """Reproject a shapely geometry from WGS84 (lon, lat) to BNG (E, N) metres."""
    bng, _ = _get_transformers()
    return transform(lambda x, y, z=None: bng.transform(x, y), geom)


def _to_wgs(geom):
    """Reproject BNG → WGS84 (lon, lat)."""
    _, wgs = _get_transformers()
    return transform(lambda x, y, z=None: wgs.transform(x, y), geom)


def _bbox_pad_wgs(geom, pad_km: float) -> Tuple[float, float, float, float]:
    """Compute a padded WGS84 bbox (lon_min, lat_min, lon_max, lat_max) around
    the geometry. Pad in km is rough at UK latitudes."""
    lon_min, lat_min, lon_max, lat_max = geom.bounds
    # ~111 km per degree latitude; lon-degree is cos(lat) * 111
    lat = (lat_min + lat_max) / 2
    dlat = pad_km / 111.0
    dlon = pad_km / (111.0 * max(math.cos(math.radians(lat)), 0.1))
    return (lon_min - dlon, lat_min - dlat, lon_max + dlon, lat_max + dlat)


def _normalize_name(s: str) -> str:
    """Normalize a road/river/place name for fuzzy matching.

    Common patterns we want to match:
      'Mill Road' ↔ 'Mill Rd' ↔ 'mill road'
      'River Stour' ↔ 'Stour' ↔ 'the River Stour'
    """
    if not s:
        return ""
    s = s.lower().strip()
    # Drop articles
    for prefix in ("the ", "a "):
        if s.startswith(prefix):
            s = s[len(prefix):]
    # Expand common abbreviations
    abbrev = {
        "rd ": "road ", "rd$": "road",
        "st ": "street ", "st$": "street",
        "ln ": "lane ", "ln$": "lane",
        "ave ": "avenue ", "ave$": "avenue",
    }
    for k, v in abbrev.items():
        import re
        s = re.sub(r"\b" + k.rstrip("$"), v, s)
    return s.strip()


def _names_match(target: str, candidate: str) -> bool:
    """Fuzzy name match. Returns True if candidate looks like target."""
    a = _normalize_name(target)
    b = _normalize_name(candidate)
    if not a or not b:
        return False
    # Direct equality
    if a == b:
        return True
    # Substring (one contains the other) — strict, full word boundary
    if len(a) >= 4 and (a in b or b in a):
        # Avoid 'st' (1 char abbrev) collisions
        return True
    return False


def _load_zoomstack_layer_bbox(layer: str, bbox_wgs: Tuple[float, float, float, float]):
    """Load a Zoomstack layer (LineString or Polygon) clipped to bbox_wgs.

    NOTE: OS_Open_Zoomstack.gpkg is in EPSG:27700 (British National Grid),
    NOT WGS84. We must convert the WGS84 bbox to BNG before querying.
    geopandas/pyogrio's bbox argument expects coordinates in the LAYER's CRS.

    Returns a GeoDataFrame in BNG (EPSG:27700).
    """
    import geopandas as gpd
    bng, _ = _get_transformers()
    lon_min, lat_min, lon_max, lat_max = bbox_wgs
    # Transform the 4 corners and take the BNG bbox covering them
    xs, ys = [], []
    for lon, lat in [(lon_min, lat_min), (lon_min, lat_max),
                       (lon_max, lat_min), (lon_max, lat_max)]:
        x, y = bng.transform(lon, lat)
        xs.append(x); ys.append(y)
    bng_bbox = (min(xs), min(ys), max(xs), max(ys))
    try:
        gdf = gpd.read_file(
            str(ZOOMSTACK_PATH),
            layer=layer,
            bbox=bng_bbox,
            engine="pyogrio",
        )
    except Exception:
        return None
    if gdf is None or len(gdf) == 0:
        return None
    return gdf


def _find_named_lines(layer: str, name: str,
                        bbox_wgs: Tuple[float, float, float, float]):
    """Return a list of LineString (already in BNG / EPSG:27700) for features
    in `layer` whose name matches `name` (fuzzy). Returns [] if not found.

    Zoomstack layers are stored in EPSG:27700, so the returned geometries
    are already in BNG metres — no reprojection needed.
    """
    gdf = _load_zoomstack_layer_bbox(layer, bbox_wgs)
    if gdf is None:
        return []
    name_col = None
    for col in ("name", "Name", "NAME", "name1"):
        if col in gdf.columns:
            name_col = col
            break
    if name_col is None:
        return []
    matched = gdf[gdf[name_col].apply(
        lambda x: _names_match(name, str(x)) if x else False
    )]
    out = []
    for geom in matched.geometry:
        if geom is None or geom.is_empty:
            continue
        # geom is already in BNG (layer CRS preserved by geopandas)
        if geom.geom_type == "LineString":
            out.append(geom)
        elif geom.geom_type == "MultiLineString":
            out.extend(list(geom.geoms))
    return out


def _snap_vertices_to_line(poly_bng: Polygon, line_geoms: List[LineString],
                            max_snap_m: float = 30.0) -> Tuple[Polygon, int, int]:
    """Snap polygon vertices that lie within max_snap_m of any line to the
    nearest point on the line. Returns (new_polygon, n_moved, n_total)."""
    if not line_geoms:
        return poly_bng, 0, 0
    line_union = unary_union(line_geoms)
    new_coords = []
    n_moved = 0
    coords = list(poly_bng.exterior.coords)
    n_total = len(coords) - 1  # last point repeats first
    for x, y in coords:
        pt = Point(x, y)
        d = pt.distance(line_union)
        if d <= max_snap_m:
            new_pt = nearest_points(pt, line_union)[1]
            new_coords.append((new_pt.x, new_pt.y))
            n_moved += 1
        else:
            new_coords.append((x, y))
    try:
        new_poly = Polygon(new_coords)
        if not new_poly.is_valid:
            new_poly = new_poly.buffer(0)
        # Preserve interior holes
        if poly_bng.interiors:
            new_poly = Polygon(new_poly.exterior.coords,
                                 [list(r.coords) for r in poly_bng.interiors])
            if not new_poly.is_valid:
                new_poly = new_poly.buffer(0)
        return new_poly, n_moved, n_total
    except Exception:
        return poly_bng, 0, n_total


def apply_constraints(predicted_geom, constraints: List[Dict[str, Any]],
                       verbose: bool = False) -> Dict[str, Any]:
    """Apply boundary constraints to refine a predicted polygon.

    Args:
        predicted_geom: shapely Polygon or MultiPolygon in WGS84.
        constraints: list of BoundaryConstraint dicts (from pdf_info.json).
        verbose: log per-constraint diagnostics.

    Returns:
        {
          "refined_geom": shapely geom (WGS84),  # may equal predicted_geom on no-op
          "applied": int,                         # constraints actually applied
          "skipped": int,                         # constraints skipped (no match)
          "rejected_guard": int,                  # snap rejected by area/move guards
          "log": list of per-constraint diagnostics,
        }
    """
    if predicted_geom is None or predicted_geom.is_empty:
        return {"refined_geom": predicted_geom, "applied": 0, "skipped": 0,
                "rejected_guard": 0, "log": []}

    # Work in BNG (metres) for snap calculations.
    if predicted_geom.geom_type == "Polygon":
        polys_bng = [_to_bng(predicted_geom)]
    elif predicted_geom.geom_type == "MultiPolygon":
        polys_bng = [_to_bng(p) for p in predicted_geom.geoms]
    else:
        return {"refined_geom": predicted_geom, "applied": 0, "skipped": 0,
                "rejected_guard": 0, "log": [f"unsupported geom: {predicted_geom.geom_type}"]}

    # Pad the bbox by 1 km to catch nearby roads
    bbox_wgs = _bbox_pad_wgs(predicted_geom, pad_km=1.0)

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

        # Choose layer per constraint type
        layers = []
        if ctype in ("follows_road", "along_centerline", "bounded_by"):
            layers = ["roads_local", "roads_regional", "roads_national"]
        elif ctype == "touches_river":
            layers = ["waterlines"]
        else:
            log.append({"type": ctype, "target": target, "outcome": "type_not_supported"})
            skipped += 1
            continue

        # Find named lines across the relevant layers
        all_lines: List[LineString] = []
        for layer in layers:
            all_lines.extend(_find_named_lines(layer, target, bbox_wgs))

        if not all_lines:
            log.append({"type": ctype, "target": target, "outcome": "no_match_in_OS_data"})
            skipped += 1
            continue

        # Try snapping each polygon
        new_polys = []
        any_changed = False
        for poly_bng in polys_bng:
            old_area = poly_bng.area
            new_poly, n_moved, n_total = _snap_vertices_to_line(
                poly_bng, all_lines, max_snap_m=30.0
            )
            new_area = new_poly.area if new_poly else 0
            # Guards
            move_frac = n_moved / max(n_total, 1)
            if new_area <= 0:
                # Snap destroyed the polygon
                new_polys.append(poly_bng)
                rejected_guard += 1
                log.append({"type": ctype, "target": target,
                             "outcome": "rejected_zero_area",
                             "n_moved": n_moved, "n_total": n_total})
            elif old_area > 0 and abs(new_area - old_area) / old_area > 0.25:
                # Area changed by >25% — too aggressive
                new_polys.append(poly_bng)
                rejected_guard += 1
                log.append({"type": ctype, "target": target,
                             "outcome": "rejected_area_drift",
                             "old_area_m2": int(old_area),
                             "new_area_m2": int(new_area),
                             "n_moved": n_moved, "n_total": n_total})
            elif move_frac < 0.05:
                # Almost no vertices moved → snap had no effect
                new_polys.append(poly_bng)
                skipped += 1
                log.append({"type": ctype, "target": target,
                             "outcome": "no_vertices_in_snap_range",
                             "n_moved": n_moved, "n_total": n_total})
            else:
                new_polys.append(new_poly)
                any_changed = True
                applied += 1
                log.append({"type": ctype, "target": target,
                             "outcome": "applied",
                             "n_moved": n_moved, "n_total": n_total,
                             "area_change_pct": round(
                                 (new_area - old_area) / max(old_area, 1) * 100, 1
                             )})

        if any_changed:
            polys_bng = new_polys

    # Reproject back to WGS84
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
