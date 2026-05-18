"""LA-polygon resolution helpers (OS BoundaryLine).

Resolves a UK administrative-area name to its boundary polygon using
local OS BoundaryLine shapefiles. Called by:
- the locate sub-agent's `la_check` tool,
- the worker's `lookup_district` tool,
- the smart-commit gate's inside-LA filter.

Public surface:
- _resolve_la(query)        → shapely (Multi)Polygon | None
- _load_la_polygons()       → Dict[name_variant → polygon]
- _normalize_la_name(s)     → canonicalized lowercase name

Data source: OS BoundaryLine shapefiles under
``os_opendata/boundary_line/`` (district_borough_unitary + county +
ceremonial-counties layers).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent

# LA-polygon dict, lazy-loaded
_LA_POLYGONS = None
_LA_NAMES = None


def _normalize_la_name(s: str) -> str:
    """Normalize LA name for matching: lowercase, strip periods, suffixes,
    'City Council', 'District Council', 'London Boro', etc."""
    if not s:
        return ""
    out = str(s).lower().strip()
    out = out.replace(".", "")  # "St. Albans" → "St Albans"
    # Strip parenthetical suffixes "(b)", "(district)", "(unitary)", "(county)"
    import re as _re
    out = _re.sub(r"\s*\(b\)$", "", out)
    out = _re.sub(r"\s*\((?:district|borough|county|unitary|metro)\)$", "", out)
    # Strip trailing administrative suffixes
    for suffix in (" district council", " borough council", " city council",
                    " county council", " metropolitan borough council",
                    " london borough council",
                    " district", " borough", " london boro", " london borough",
                    " metropolitan borough", " unitary", " unitary authority",
                    " council"):
        if out.endswith(suffix):
            out = out[:-len(suffix)].strip()
    # Strip leading prefixes
    for prefix in ("city of ", "london borough of ", "borough of ",
                    "the london borough of ", "royal borough of "):
        if out.startswith(prefix):
            out = out[len(prefix):].strip()
    return out


def _add_la_variants(out: Dict[str, Any], name: str, geom):
    """Insert (lower, normalized) → geom mappings, including useful aliases."""
    nm = name.lower()
    out[nm] = geom
    norm = _normalize_la_name(name)
    if norm and norm not in out:
        out[norm] = geom
    # Drop any "(b)" / "(district)" but keep the rest
    if " (b)" in nm:
        out[nm.replace(" (b)", "")] = geom
    # Strip trailing district/borough
    for suffix in (" district", " borough", " london boro", " county"):
        if nm.endswith(suffix):
            short = nm[:-len(suffix)].strip()
            if short and short not in out:
                out[short] = geom


def _load_la_polygons() -> Dict[str, Any]:
    """Load OS BoundaryLine LA polygons into name→shapely dict.
    Loads district_borough_unitary_region AND county_region (counties like Kent
    aren't in the district layer)."""
    global _LA_POLYGONS, _LA_NAMES
    if _LA_POLYGONS is not None:
        return _LA_POLYGONS
    bdir = ROOT / "os_opendata" / "boundary_line"
    if not bdir.exists():
        _LA_POLYGONS = {}
        _LA_NAMES = []
        return _LA_POLYGONS
    try:
        import geopandas as gpd
        out: Dict[str, Any] = {}
        # Load all relevant layers
        layer_paths = []
        for fname in ("district_borough_unitary_region.shp",
                       "county_region.shp",
                       "boundary-line-ceremonial-counties_region.shp"):
            paths = list(bdir.rglob(fname))
            layer_paths.extend(paths)
        # Also extract county shapefile from zip if not already
        county_shp = bdir / "Data" / "GB" / "county_region.shp"
        if not county_shp.exists():
            zp = bdir / "bdline_essh.zip"
            if zp.exists():
                import zipfile
                with zipfile.ZipFile(zp) as z:
                    for member in z.namelist():
                        if "county_region" in member or "ceremonial-counties" in member:
                            try:
                                z.extract(member, str(bdir))
                            except Exception:
                                pass
                paths = list(bdir.rglob("county_region.shp"))
                layer_paths.extend(paths)
                paths = list(bdir.rglob("boundary-line-ceremonial-counties_region.shp"))
                layer_paths.extend(paths)

        layer_paths = list(set(layer_paths))  # dedup
        if not layer_paths:
            print(f"  BoundaryLine: no LA shapefiles under {bdir}")
            _LA_POLYGONS = {}
            _LA_NAMES = []
            return _LA_POLYGONS
        for path in layer_paths:
            try:
                gdf = gpd.read_file(str(path)).to_crs(4326)
            except Exception:
                continue
            name_col = next((c for c in gdf.columns if c.lower() == "name"), None)
            if name_col is None:
                continue
            for _, row in gdf.iterrows():
                nm = str(row[name_col]).strip()
                if nm and row.geometry is not None and not row.geometry.is_empty:
                    _add_la_variants(out, nm, row.geometry)
        _LA_POLYGONS = out
        _LA_NAMES = sorted(set(out.keys()))
        return out
    except Exception as e:
        print(f"  BoundaryLine load failed: {e!s:.200}")
        _LA_POLYGONS = {}
        _LA_NAMES = []
        return _LA_POLYGONS


def _resolve_la(query: str):
    """Return the best-matching LA polygon, or None.
    Tries: exact, normalized, substring (with priority for longer matches)."""
    q = (query or "").strip().lower()
    if not q:
        return None
    polys = _load_la_polygons()
    # Try exact (lowercased) first
    if q in polys:
        return polys[q]
    # Try normalized form (strip "City Council", periods, etc.)
    qn = _normalize_la_name(query)
    if qn and qn in polys:
        return polys[qn]
    # Substring match — prefer longest match
    best = None
    best_len = 0
    for k, v in polys.items():
        if q in k or qn in k or k in q or (qn and k in qn):
            if len(k) > best_len:
                best = v
                best_len = len(k)
    return best
