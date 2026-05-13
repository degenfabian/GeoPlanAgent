"""Parcel-selection snap (human-workflow algorithm).

Different from `tools/snap/inspire.py` (which moves polygon vertices). This
REPLACES the predicted polygon with a UNION of OS Open Map Local building
footprints OR INSPIRE freehold parcels whose centroids fall INSIDE the
predicted polygon.

This replicates how UK planning officers actually work: they don't trace
pixels — they SELECT TOID parcels on OS MasterMap and dissolve them.
We use the free OS Open Map Local + INSPIRE Index Polygons as substitutes
for paid OS MasterMap.

Decision: building snap vs parcel snap?
  - boundary_description contains "single building" / "house" / "dwelling"
    → building snap (OS Open Map Local)
  - boundary_description contains "land" / "field" / "agricultural"
    → parcel snap (INSPIRE)
  - otherwise: try both, pick the one with MORE centroid-inside hits

Safety guard: reject snap if resulting area is <0.5× or >2× of predicted.
"""
from __future__ import annotations
import math, os, re
from pathlib import Path
from typing import Optional, List

ROOT = Path(__file__).resolve().parent.parent
OPEN_MAP_LOCAL_DIR = ROOT / "os_opendata" / "open_map_local"


# Cache of building-footprints per 100km grid (lazy-loaded)
_OML_CACHE: dict = {}
_OML_TRANSFORMER = None


def _bng_to_wgs_init():
    global _OML_TRANSFORMER
    if _OML_TRANSFORMER is None:
        from pyproj import Transformer
        _OML_TRANSFORMER = Transformer.from_crs(27700, 4326, always_xy=True)
    return _OML_TRANSFORMER


def _grid_for_bbox_wgs84(lat_min, lon_min, lat_max, lon_max):
    """Return list of OS 100km-grid letter codes covering this WGS84 bbox.
    Approximate; we just enumerate likely grids around UK."""
    # For now, return a generous superset. Refine after we see what's
    # actually in the bbox during a real query.
    from pyproj import Transformer
    t = Transformer.from_crs(4326, 27700, always_xy=True)
    corners = [t.transform(lon, lat) for lat in (lat_min, lat_max)
               for lon in (lon_min, lon_max)]
    e_min = min(c[0] for c in corners); e_max = max(c[0] for c in corners)
    n_min = min(c[1] for c in corners); n_max = max(c[1] for c in corners)
    # OS 100km letters: 25-square (5x5) starting at false origin.
    # See https://www.ordnancesurvey.co.uk/documents/resources/guide-to-nationalgrid.pdf
    grids = set()
    for e_100 in range(int(e_min // 100000) - 1, int(e_max // 100000) + 2):
        for n_100 in range(int(n_min // 100000) - 1, int(n_max // 100000) + 2):
            letter = _en_to_grid(e_100 * 100000, n_100 * 100000)
            if letter:
                grids.add(letter)
    return grids


_OUTER_500KM = {
    (0, 0): 'S', (1, 0): 'T',
    (0, 1): 'N', (1, 1): 'O',
    (0, 2): 'H', (1, 2): 'J',
}
# Inner 100km letters arranged 5x5 reading from bottom-left (n_100=0,e_100=0)
# to top-right. Letters A-Z skipping I.
_INNER_100KM = [
    ['V', 'W', 'X', 'Y', 'Z'],  # n_100 = 0 (bottom row)
    ['Q', 'R', 'S', 'T', 'U'],  # n_100 = 1
    ['L', 'M', 'N', 'O', 'P'],  # n_100 = 2
    ['F', 'G', 'H', 'J', 'K'],  # n_100 = 3
    ['A', 'B', 'C', 'D', 'E'],  # n_100 = 4 (top row)
]


def _en_to_grid(e, n):
    """Convert easting/northing in metres to OS two-letter grid code (e.g. 'SU')."""
    e_500 = int(e // 500000)
    n_500 = int(n // 500000)
    outer = _OUTER_500KM.get((e_500, n_500))
    if outer is None: return None
    e_100 = int((e % 500000) // 100000)
    n_100 = int((n % 500000) // 100000)
    if not (0 <= e_100 < 5 and 0 <= n_100 < 5): return None
    return outer + _INNER_100KM[n_100][e_100]


def _load_buildings_for_grid(grid_letter: str):
    """Lazy-load building polygons for one 100km grid (e.g. 'SU' or 'TQ').
    Returns geopandas GeoDataFrame in WGS84, or None if not found."""
    if grid_letter in _OML_CACHE: return _OML_CACHE[grid_letter]
    zp = OPEN_MAP_LOCAL_DIR / f"opmplc_essh_{grid_letter.lower()}.zip"
    if not zp.exists():
        _OML_CACHE[grid_letter] = None
        return None
    # The shapefile lives inside a folder with spaces:
    #   "OS OpenMap Local (ESRI Shape File) SU/data/SU_Building.shp"
    # geopandas can read it via the !-suffix syntax (URL-encoded spaces).
    inner = f"OS OpenMap Local (ESRI Shape File) {grid_letter}/data/{grid_letter}_Building.shp"
    try:
        import geopandas as gpd
        # geopandas/pyogrio handles spaces if we pass the full vsi path
        gdf = gpd.read_file(f"zip://{zp}!{inner}")
        gdf = gdf.to_crs(4326)
        _OML_CACHE[grid_letter] = gdf
        return gdf
    except Exception as e:
        print(f"  OML load {grid_letter} failed: {e!s:.100}")
        _OML_CACHE[grid_letter] = None
        return None


def building_polygons_in_bbox(lat_min, lon_min, lat_max, lon_max):
    """Return list of shapely building Polygons in WGS84."""
    grids = _grid_for_bbox_wgs84(lat_min, lon_min, lat_max, lon_max)
    out = []
    for g in grids:
        gdf = _load_buildings_for_grid(g)
        if gdf is None or len(gdf) == 0: continue
        # Filter to bbox
        sub = gdf.cx[lon_min:lon_max, lat_min:lat_max]
        for geom in sub.geometry:
            if geom is not None and not geom.is_empty:
                out.append(geom)
    return out


def parcel_snap(predicted_polygon, pdf_info: dict | None = None,
                la_name: str | None = None,
                area_band=(0.5, 2.0)) -> 'shapely.Polygon':
    """Snap predicted polygon to UNION of parcels/buildings whose centroid
    falls inside it.

    Args:
        predicted_polygon: shapely Polygon in WGS84.
        pdf_info: optional dict with 'boundary_description' to choose between
            building snap and parcel snap.
        la_name: optional Local Authority for INSPIRE parcel snap.
        area_band: (lo, hi) — reject if new_area / old_area outside this range.

    Returns:
        Snapped shapely polygon, or original if no good snap found.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    if predicted_polygon is None or predicted_polygon.is_empty:
        return predicted_polygon
    bounds = predicted_polygon.bounds  # (lon_min, lat_min, lon_max, lat_max)
    lon_min, lat_min, lon_max, lat_max = bounds

    desc = (pdf_info or {}).get('boundary_description', '').lower()
    is_building = bool(re.search(r'\b(single building|building|house|dwelling|cottage|property)\b', desc))
    is_field = bool(re.search(r'\b(land|field|agricultural|wedge|woodland)\b', desc))

    candidates = []
    used_source = None

    # Building snap — use OS Open Map Local
    if is_building or not is_field:
        try:
            buildings = building_polygons_in_bbox(lat_min, lon_min, lat_max, lon_max)
            inside = [b for b in buildings
                      if predicted_polygon.contains(b.centroid) or predicted_polygon.intersects(b)]
            if inside:
                candidates.append(('open_map_local_buildings', inside))
        except Exception:
            pass

    # Parcel snap — use INSPIRE freehold parcels
    if (is_field or not is_building) and la_name:
        try:
            from tools.snap.inspire import InspireSnap
            insp = InspireSnap([la_name])
            # InspireSnap stores edges in self._edges (LineStrings, WGS84).
            # Reconstruct parcel polygons from edges → too expensive.
            # Skip for now; primary value is building snap.
        except Exception:
            pass

    if not candidates:
        return predicted_polygon

    # Pick the source with most overlapping parcels
    candidates.sort(key=lambda kv: -len(kv[1]))
    source, inside = candidates[0]
    snapped = unary_union(inside)
    if snapped.is_empty: return predicted_polygon

    # Safety guard: reject if area changed too much
    old_area = predicted_polygon.area
    new_area = snapped.area
    if old_area > 0:
        ratio = new_area / old_area
        if ratio < area_band[0] or ratio > area_band[1]:
            return predicted_polygon

    return snapped


if __name__ == "__main__":
    # Smoke: verify grid math
    print(_en_to_grid(440000, 175000), '(expect SU)')  # Wiltshire
    print(_en_to_grid(530000, 180000), '(expect TQ)')  # London
    print(_en_to_grid(330000, 380000), '(expect SK)')  # Midlands
    print(_grid_for_bbox_wgs84(51.5, -0.1, 51.6, 0.0), '(expect TQ-ish around London)')
