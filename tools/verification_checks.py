"""Resolve a UK administrative-area name to its OS BoundaryLine polygon."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent

_LA_POLYGONS = None
_LA_NAMES = None

# District > county > ceremonial: first layer to claim a name wins.
_LAYER_ORDER = (
    "district_borough_unitary_region.shp",
    "county_region.shp",
    "boundary-line-ceremonial-counties_region.shp",
)


def _normalize_la_name(s: str) -> str:
    if not s:
        return ""
    out = str(s).lower().strip().replace(".", "")
    out = re.sub(r"\s*\(b\)$", "", out)
    out = re.sub(r"\s*\((?:district|borough|county|unitary|metro)\)$", "", out)
    for suffix in (" district council", " borough council", " city council",
                    " county council", " metropolitan borough council",
                    " london borough council",
                    " district", " borough", " london boro", " london borough",
                    " metropolitan borough", " unitary", " unitary authority",
                    " council"):
        if out.endswith(suffix):
            out = out[:-len(suffix)].strip()
    for prefix in ("city of ", "london borough of ", "borough of ",
                    "the london borough of ", "royal borough of "):
        if out.startswith(prefix):
            out = out[len(prefix):].strip()
    return out


def _add_la_variants(out: Dict[str, Any], name: str, geom):
    nm = name.lower()
    if nm not in out:
        out[nm] = geom
    norm = _normalize_la_name(name)
    if norm and norm not in out:
        out[norm] = geom
    if " (b)" in nm:
        bare = nm.replace(" (b)", "")
        if bare not in out:
            out[bare] = geom
    for suffix in (" district", " borough", " london boro", " county"):
        if nm.endswith(suffix):
            short = nm[:-len(suffix)].strip()
            if short and short not in out:
                out[short] = geom


def _load_la_polygons() -> Dict[str, Any]:
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

        # Case-insensitive shapefile lookup — Linux is case-sensitive and the
        # ceremonial-counties file ships with a capital B.
        def _find_layer(fname: str):
            lower = fname.lower()
            for p in sorted(bdir.rglob("*.shp")):
                if p.name.lower() == lower:
                    return p
            return None

        if any(_find_layer(f) is None for f in _LAYER_ORDER):
            zp = bdir / "bdline_essh.zip"
            if zp.exists():
                import zipfile
                with zipfile.ZipFile(zp) as z:
                    for member in z.namelist():
                        ml = member.lower()
                        if ("county_region" in ml
                                or "ceremonial-counties" in ml
                                or "district_borough_unitary" in ml):
                            try:
                                z.extract(member, str(bdir))
                            except Exception:
                                pass

        layer_paths = []
        seen = set()
        for fname in _LAYER_ORDER:
            p = _find_layer(fname)
            if p is not None and p not in seen:
                seen.add(p)
                layer_paths.append(p)
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
    q = (query or "").strip().lower()
    if not q:
        return None
    polys = _load_la_polygons()
    if q in polys:
        return polys[q]
    qn = _normalize_la_name(query)
    if qn and qn in polys:
        return polys[qn]
    best = None
    best_len = 0
    for k, v in polys.items():
        if q in k or qn in k or k in q or (qn and k in qn):
            if len(k) > best_len:
                best = v
                best_len = len(k)
    return best
