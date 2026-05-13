"""Verification cross-checks (post-positioning sanity gates).

Each check is a pure function returning (confidence ∈ [0,1], reason: str).
A reason string of "" means the check did not apply (insufficient data).

Aggregated by `verification_score` into a single confidence the critic uses
to ROUTE retries. Below 0.45 → force retry_in_worker with diagnosis +
Reflexion memory. Below 0.25 → flag_low_confidence. Above 0.7 with
≥2 checks passing → straight approve.

Designed against the verification checklist a UK planning officer runs:
  - Polygon area is plausible for the description ("single building", "field")
  - Postcode (when single-property) sits inside or near the polygon
  - Polygon is inside the named admin region
  - MINIMA inliers spread across the matched window (not clustered on legend)
  - Scale factor ∈ reasonable band

References: PAS National Validation Guide, MHCLG Extract weeknotes 2026W04/W10,
Set-of-Mark prompting (Yang 2023), CRITIC (Gou ICLR 2024), Reflexion (Shinn 2023).
"""
from __future__ import annotations
import os, re
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

ROOT = Path(__file__).resolve().parent.parent

# Per-check return type
CheckResult = Tuple[float, str]  # (confidence in [0,1], reason)


# ────────────────────────────────────────────────────────────────────────────
# Description → expected-area-band heuristics
# ────────────────────────────────────────────────────────────────────────────

# Negative signals (override single-property even if positive matches)
_NEG_LARGE = re.compile(
    r"\b(conservation area|article 4 (?:area|direction)|borough|district|"
    r"estate|town centre|village|parish|airfield|park\b|woodland|farm|"
    r"hamlet|whole|entire|all|various|multiple|several)\b",
    re.I,
)
_NEG_ROW = re.compile(
    r"\b(row of|terrace|terraced|properties|dwellinghouses|houses|"
    r"semi-detached)\b",
    re.I,
)
_POS_SINGLE = re.compile(
    r"\b(single (?:building|dwelling|house|cottage|property|footprint)|"
    r"the (?:building|property|dwelling|cottage|house|shop)|"
    r"cottage|bungalow)\b",
    re.I,
)
_POS_FIELD = re.compile(
    r"\b(field|land|agricultural|paddock|grassland)\b", re.I,
)
_POS_BUILDING_LARGER = re.compile(
    r"\b(theatre|hall|church|hotel|station|warehouse|chapel|school|"
    r"college|university|hospital|leisure|public house|pub)\b", re.I,
)
_POS_LARGE = re.compile(
    r"\b(conservation area|article 4|borough|district|estate|"
    r"town centre|village|parish|airfield|park|woodland|whole|entire)\b",
    re.I,
)

# Explicit area extraction
_AREA_PAT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(hectares?|ha\b|acres?|sq\.?\s*m(?:etres?)?|m²|"
    r"square metres?)",
    re.I,
)


def _explicit_area_m2(pdf_info: Dict[str, Any]) -> Optional[float]:
    """Extract an explicit area mention from boundary_description / notes.
    Returns m² or None."""
    text = " ".join(
        (pdf_info.get(k) or "") for k in ("boundary_description", "notes", "site_address")
    )
    matches = list(_AREA_PAT.finditer(text))
    if not matches:
        return None
    # Prefer first hit that's plausibly small (filter out OS parcel codes
    # like "077 Ha" appearing in case IDs)
    for m in matches:
        val = float(m.group(1))
        unit = m.group(2).lower().replace(" ", "").replace(".", "")
        if unit.startswith("hect") or unit == "ha":
            m2 = val * 10_000
        elif unit.startswith("acre"):
            m2 = val * 4046.86
        else:
            m2 = val
        if 5 <= m2 <= 50_000_000:
            return m2
    return None


def _expected_area_band_m2(pdf_info: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Heuristic expected-area band from boundary_description.
    Returns (lo_m2, hi_m2) or None if unclear.

    Negative signals dominate (large/row override single-property).
    """
    explicit = _explicit_area_m2(pdf_info)
    if explicit is not None:
        return (0.5 * explicit, 2.0 * explicit)
    desc = pdf_info.get("boundary_description") or ""
    if not desc:
        return None
    # Negative signals first — they trump
    if _NEG_LARGE.search(desc) or _POS_LARGE.search(desc):
        return (50_000, 50_000_000)  # 5ha → 5000ha
    if _NEG_ROW.search(desc):
        return (200, 30_000)  # row of houses → terrace
    if _POS_BUILDING_LARGER.search(desc):
        return (200, 50_000)  # theatre, school, hospital
    if _POS_FIELD.search(desc):
        return (1_000, 1_000_000)  # field, land
    if _POS_SINGLE.search(desc):
        return (50, 3_000)  # single cottage/dwelling
    return None  # truly unclear


# ────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ────────────────────────────────────────────────────────────────────────────

_BNG_TRANSFORMER = None
def _to_bng(geom):
    """Transform WGS84 shapely geom to BNG (m)."""
    global _BNG_TRANSFORMER
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer
    if _BNG_TRANSFORMER is None:
        _BNG_TRANSFORMER = Transformer.from_crs(4326, 27700, always_xy=True)
    return shp_transform(lambda x, y, z=None: _BNG_TRANSFORMER.transform(x, y), geom)


def _polygon_diameter_m(geom_bng) -> float:
    minx, miny, maxx, maxy = geom_bng.bounds
    return max(maxx - minx, maxy - miny)


def _haversine_m(lon1, lat1, lon2, lat2) -> float:
    """Approximate ground distance in metres."""
    import math
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    return R * math.hypot(dlat, dlon)


# ────────────────────────────────────────────────────────────────────────────
# Check 1 — Polygon area consistency
# ────────────────────────────────────────────────────────────────────────────

def check_area_consistency(pdf_info: Dict[str, Any], predicted_geom) -> CheckResult:
    """Compare polygon area to expected from description (and explicit area)."""
    band = _expected_area_band_m2(pdf_info)
    if band is None:
        return (0.5, "")  # neutral — no info
    if predicted_geom is None or predicted_geom.is_empty:
        return (0.0, "polygon empty")
    pred_m2 = _to_bng(predicted_geom).area
    lo, hi = band
    if lo <= pred_m2 <= hi:
        return (1.0, f"area={pred_m2:.0f}m² in expected band [{lo:.0f}, {hi:.0f}]")
    if pred_m2 < lo:
        ratio = lo / max(1.0, pred_m2)
        conf = max(0.0, 1.0 - 0.3 * (ratio - 1))  # graceful decay
        return (conf, f"area={pred_m2:.0f}m² is {ratio:.1f}× too small for expected ≥{lo:.0f}")
    # too big
    ratio = pred_m2 / max(1.0, hi)
    conf = max(0.0, 1.0 - 0.3 * (ratio - 1))
    return (conf, f"area={pred_m2:.0f}m² is {ratio:.1f}× too big for expected ≤{hi:.0f}")


# ────────────────────────────────────────────────────────────────────────────
# Check 2 — Postcode-in-polygon (single-property gated)
# ────────────────────────────────────────────────────────────────────────────

def check_postcode_in_polygon(pdf_info: Dict[str, Any], predicted_geom) -> CheckResult:
    """When boundary is a single property, postcode centroid should be near
    the polygon. Skipped for large/row/area types (postcode is metadata then).
    """
    desc = pdf_info.get("boundary_description") or ""
    if _NEG_LARGE.search(desc) or _NEG_ROW.search(desc) or _POS_LARGE.search(desc):
        return (0.5, "")  # not single-property; skip
    if not _POS_SINGLE.search(desc) and not _POS_BUILDING_LARGER.search(desc):
        return (0.5, "")  # ambiguous; skip
    pcs = pdf_info.get("postcodes") or []
    if not pcs or predicted_geom is None or predicted_geom.is_empty:
        return (0.5, "")
    try:
        from tools.code_point import lookup_postcode
    except Exception:
        return (0.5, "")
    hit = None
    for pc in pcs[:3]:
        h = lookup_postcode(pc)
        if h:
            hit = h; break
    if hit is None:
        return (0.5, "")  # outward-only postcode, can't lookup
    geom_bng = _to_bng(predicted_geom)
    diam_m = _polygon_diameter_m(geom_bng)
    cent = predicted_geom.centroid
    dist_m = _haversine_m(cent.x, cent.y, hit["lon"], hit["lat"])
    if dist_m < 0.5 * diam_m:
        return (1.0, f"postcode {pcs[0]} {dist_m:.0f}m from centroid (≤0.5×diam)")
    if dist_m < 2.0 * diam_m:
        return (0.5, f"postcode {pcs[0]} {dist_m:.0f}m from centroid (≤2×diam)")
    return (0.0, f"postcode {pcs[0]} is {dist_m:.0f}m from centroid (>2×diam={2*diam_m:.0f}m)")


# ────────────────────────────────────────────────────────────────────────────
# Check 3 — Polygon inside named LA admin region
# ────────────────────────────────────────────────────────────────────────────

# LA-polygon dict, lazy-loaded
_LA_POLYGONS = None
_LA_NAMES = None


def _normalize_la_name(s: str) -> str:
    """Normalize LA name for matching: lowercase, strip periods, suffixes,
    'City Council', 'District Council', 'London Boro', etc."""
    if not s: return ""
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
        _LA_POLYGONS = {}; _LA_NAMES = []
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
                            try: z.extract(member, str(bdir))
                            except Exception: pass
                paths = list(bdir.rglob("county_region.shp"))
                layer_paths.extend(paths)
                paths = list(bdir.rglob("boundary-line-ceremonial-counties_region.shp"))
                layer_paths.extend(paths)

        layer_paths = list(set(layer_paths))  # dedup
        if not layer_paths:
            print(f"  BoundaryLine: no LA shapefiles under {bdir}")
            _LA_POLYGONS = {}; _LA_NAMES = []
            return _LA_POLYGONS
        for path in layer_paths:
            try:
                gdf = gpd.read_file(str(path)).to_crs(4326)
            except Exception as e:
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
        _LA_POLYGONS = {}; _LA_NAMES = []
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
    best = None; best_len = 0
    for k, v in polys.items():
        if q in k or qn in k or k in q or (qn and k in qn):
            if len(k) > best_len:
                best = v; best_len = len(k)
    return best


def check_la_boundary(pdf_info: Dict[str, Any], predicted_geom) -> CheckResult:
    """Predicted polygon must be ≥80% inside the named admin region's LA."""
    name = pdf_info.get("admin_region")
    if not name or predicted_geom is None or predicted_geom.is_empty:
        return (0.5, "")
    la_geom = _resolve_la(name)
    if la_geom is None:
        return (0.5, f"unknown admin_region {name!r}")
    try:
        inside = predicted_geom.intersection(la_geom).area / max(1e-12, predicted_geom.area)
    except Exception:
        return (0.5, "intersection failed")
    if inside >= 0.95:
        return (1.0, f"polygon {inside*100:.0f}% inside {name}")
    if inside >= 0.50:
        return (0.5, f"polygon {inside*100:.0f}% inside {name} (partial)")
    return (0.0, f"polygon only {inside*100:.0f}% inside {name} — likely WRONG town")


# ────────────────────────────────────────────────────────────────────────────
# Check 4 — Inlier geometric scatter (cached match_info has summary only;
#           needs inlier_coords logged for full implementation)
# ────────────────────────────────────────────────────────────────────────────

def check_inlier_scatter(match_info: Dict[str, Any]) -> CheckResult:
    """Convex-hull area of MINIMA inliers / window area. Tightly clustered
    inliers (hull/window < 0.05) signal title-block/legend lock.
    Falls back to neutral if inlier coords aren't in match_info.
    """
    if not match_info:
        return (0.5, "")
    inliers = (match_info.get("inlier_coords") or
               match_info.get("inliers") or
               match_info.get("inlier_pts"))
    window = match_info.get("window") or match_info.get("matched_window")
    if not inliers or not window:
        return (0.5, "")  # not logged yet
    try:
        import numpy as np
        from scipy.spatial import ConvexHull
        pts = np.asarray(inliers, dtype=float).reshape(-1, 2)
        if len(pts) < 4:
            return (0.0, f"only {len(pts)} inliers, can't form hull")
        hull = ConvexHull(pts)
        hull_area = float(hull.volume)  # 2D ConvexHull.volume is the area
        if isinstance(window, (list, tuple)) and len(window) == 4:
            w_w = window[2] - window[0]; w_h = window[3] - window[1]
            window_area = max(1.0, w_w * w_h)
        else:
            window_area = float(window)
        ratio = hull_area / window_area
        if ratio >= 0.30:
            return (1.0, f"inlier hull spread {ratio*100:.0f}% of window")
        if ratio >= 0.10:
            return (0.5, f"inlier hull spread only {ratio*100:.0f}% of window")
        return (0.0, f"inlier hull tight ({ratio*100:.0f}% of window) — likely legend lock")
    except Exception as e:
        return (0.5, f"inlier scatter calc failed: {e!s:.50}")


# ────────────────────────────────────────────────────────────────────────────
# Check 5 — Scale factor sanity
# ────────────────────────────────────────────────────────────────────────────

def check_scale_factor(match_info: Dict[str, Any]) -> CheckResult:
    """MINIMA's scale_factor (or avg_scale) outside [0.5, 2.0] is suspect."""
    if not match_info:
        return (0.5, "")
    sf = match_info.get("scale_factor") or match_info.get("avg_scale")
    if sf is None:
        return (0.5, "")
    sf = float(sf)
    if 0.7 <= sf <= 1.5:
        return (1.0, f"scale_factor={sf:.2f} normal")
    if 0.5 <= sf <= 2.0:
        return (0.5, f"scale_factor={sf:.2f} unusual")
    return (0.0, f"scale_factor={sf:.2f} extreme — wrong-zoom or wrong-rotation")


# ────────────────────────────────────────────────────────────────────────────
# Check 6 — OS Open Map Local building proximity
# ────────────────────────────────────────────────────────────────────────────

def check_building_overlap(predicted_geom) -> CheckResult:
    """Predicted polygon should contain or border at least one OS building
    when the surrounding area is urban (>10 buildings within 500m of polygon
    centroid). Rural sites without buildings nearby are not penalized.

    This catches catastrophic positioning where the polygon lands in empty
    space far from any OS building footprint despite being in an urban
    setting (e.g. wrong-town homonym matches).

    Returns confidence:
      1.0 — rural (no buildings within 500m, gate not applicable)
      1.0 — urban AND ≥1 OS building within 30m of polygon (overlap or border)
      0.5 — urban AND nearest OS building 30-100m away
      0.0 — urban AND no OS building within 100m of polygon
    """
    if predicted_geom is None or predicted_geom.is_empty:
        return (0.5, "")
    try:
        from tools.snap.parcel import building_polygons_in_bbox
    except Exception:
        return (0.5, "parcel_snap not available")
    try:
        # 500m bbox around polygon centroid for urban-context detection
        c = predicted_geom.centroid
        # Rough degree-to-meter at UK latitudes: 1 deg lat ≈ 111km;
        # 1 deg lon ≈ 111km × cos(lat). 500m → ~0.0045 deg lat,
        # ~0.0072 deg lon at lat 53.
        import math
        dlat = 0.005
        dlon = 0.005 / max(math.cos(math.radians(c.y)), 0.1)
        bbox_lat_min, bbox_lat_max = c.y - dlat, c.y + dlat
        bbox_lon_min, bbox_lon_max = c.x - dlon, c.x + dlon
        nearby = building_polygons_in_bbox(
            bbox_lat_min, bbox_lon_min, bbox_lat_max, bbox_lon_max
        )
        if len(nearby) < 10:
            return (1.0, f"rural ({len(nearby)} OS buildings in 500m bbox)")
        # Urban context — measure distance from polygon to nearest building
        # Convert to BNG for accurate metres
        pred_bng = _to_bng(predicted_geom)
        if pred_bng is None or pred_bng.is_empty:
            return (0.5, "couldn't reproject polygon to BNG")
        from shapely.ops import unary_union
        bldg_union = unary_union(nearby)
        bldg_bng = _to_bng(bldg_union)
        if bldg_bng is None or bldg_bng.is_empty:
            return (0.5, "couldn't reproject buildings to BNG")
        d = pred_bng.distance(bldg_bng)
        if d <= 30:
            return (1.0, f"urban: {len(nearby)} buildings nearby, polygon "
                          f"{'contains/borders' if d == 0 else f'{d:.0f}m from'} buildings")
        if d <= 100:
            return (0.5, f"urban: {len(nearby)} buildings nearby, polygon "
                          f"{d:.0f}m from nearest")
        return (0.0, f"urban: {len(nearby)} buildings within 500m but polygon "
                      f"{d:.0f}m from any of them — likely wrong location")
    except Exception as e:
        return (0.5, f"building_overlap calc failed: {e!s:.50}")


# ────────────────────────────────────────────────────────────────────────────
# Check 7 — Multi-zoom coherence
# ────────────────────────────────────────────────────────────────────────────

def check_multi_zoom_coherence(match_info: Dict[str, Any]) -> CheckResult:
    """When multiple zoom levels were tried in the same MINIMA sweep, their
    chosen centers should agree within ~200m. Significant disagreement
    suggests the match is locked to a single zoom on noise rather than
    actual map content.

    Reads match_info["candidates_per_zoom"] (an optional list MINIMA can log)
    or falls back to neutral if the data isn't there. To enable, the caller
    of `sliding_window_position` must pass `return_candidates=True` and the
    result must propagate the per-zoom centers into match_info.

    Returns confidence:
      1.0 — top-2 chosen centers agree within 200m (coherent)
      0.5 — agree within 1km (mild disagreement)
      0.0 — disagree by >1km (likely zoom-locked on noise)
    """
    if not match_info:
        return (0.5, "")
    cands = match_info.get("candidates_per_zoom")
    if not cands or not isinstance(cands, list) or len(cands) < 2:
        return (0.5, "")  # not enough zooms tried OR data not logged
    # Take the top 2 by inlier count
    sorted_cands = sorted(cands, key=lambda c: -(c.get("n_inliers") or 0))[:2]
    if len(sorted_cands) < 2:
        return (0.5, "")
    c1, c2 = sorted_cands
    ll1 = c1.get("center_latlon") or c1.get("chosen_center_latlon")
    ll2 = c2.get("center_latlon") or c2.get("chosen_center_latlon")
    if not (ll1 and ll2):
        return (0.5, "")
    d_m = _haversine_m(ll1[1], ll1[0], ll2[1], ll2[0])
    if d_m <= 200:
        return (1.0, f"top-2 zooms agree (drift {d_m:.0f}m)")
    if d_m <= 1000:
        return (0.5, f"top-2 zooms drift {d_m:.0f}m — mild disagreement")
    return (0.0, f"top-2 zooms drift {d_m:.0f}m — likely zoom-locked on noise")


# ────────────────────────────────────────────────────────────────────────────
# Aggregator
# ────────────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "area_consistency": 0.20,
    "postcode_in_polygon": 0.10,
    "la_boundary": 0.15,
    "inlier_scatter": 0.15,
    "scale_factor": 0.05,
    "building_overlap": 0.20,
    "multi_zoom_coherence": 0.15,
}


def verification_score(
    pdf_info: Dict[str, Any],
    predicted_geom,
    match_info: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Run all checks and return a combined confidence + per-check breakdown.

    Returns:
        {
          "score": float in [0,1],
          "checks": {name: {"confidence": float, "reason": str}, ...},
          "diagnosis": str       # one of {WRONG_TOWN, BAD_SCALE, OOB_AREA, BAD_MASK, OK}
        }
    """
    weights = weights or DEFAULT_WEIGHTS
    checks = {
        "area_consistency": check_area_consistency(pdf_info, predicted_geom),
        "postcode_in_polygon": check_postcode_in_polygon(pdf_info, predicted_geom),
        "la_boundary": check_la_boundary(pdf_info, predicted_geom),
        "inlier_scatter": check_inlier_scatter(match_info or {}),
        "scale_factor": check_scale_factor(match_info or {}),
        "building_overlap": check_building_overlap(predicted_geom),
        "multi_zoom_coherence": check_multi_zoom_coherence(match_info or {}),
    }
    total_w = 0.0; total = 0.0
    for name, (conf, reason) in checks.items():
        # Skip neutral (0.5 with empty reason = "not applicable")
        if conf == 0.5 and reason == "":
            continue
        w = weights.get(name, 0.0)
        total += conf * w
        total_w += w
    score = total / total_w if total_w > 0 else 0.5

    # Diagnosis: which check failed hardest
    diagnosis = "OK"
    if checks["la_boundary"][0] < 0.3:
        diagnosis = "WRONG_TOWN"
    elif checks["building_overlap"][0] < 0.3:
        diagnosis = "WRONG_LOCATION"   # urban context, polygon far from buildings
    elif checks["multi_zoom_coherence"][0] < 0.3:
        diagnosis = "ZOOM_LOCKED"      # match unstable across zooms
    elif checks["inlier_scatter"][0] < 0.3:
        diagnosis = "BAD_MASK_LOCK"    # legend / title-block lock
    elif checks["area_consistency"][0] < 0.3:
        diagnosis = "OOB_AREA"
    elif checks["postcode_in_polygon"][0] < 0.3:
        diagnosis = "PC_FAR_FROM_POLY"
    elif checks["scale_factor"][0] < 0.3:
        diagnosis = "BAD_SCALE"

    return {
        "score": float(score),
        "checks": {n: {"confidence": float(c), "reason": r} for n, (c, r) in checks.items()},
        "diagnosis": diagnosis,
        # Hard-gate trigger flag for callers (v18 critic-replacement):
        # True when at least one of the three v18 hard gates failed.
        "hard_gate_failed": (
            checks["inlier_scatter"][0] < 0.3
            or checks["building_overlap"][0] < 0.3
            or checks["multi_zoom_coherence"][0] < 0.3
        ),
    }


if __name__ == "__main__":
    import json, glob
    from shapely.geometry import shape

    print("=== verification_checks smoke test on sub-0.3 cases ===")
    n_loaded = len(_load_la_polygons())
    print(f"Loaded {n_loaded} LA polygons from BoundaryLine")
    files = sorted(glob.glob("results/benchmark_v13/gemini-flash/*/metrics.json"))
    n = 0
    for mf in files:
        case = mf.split("/")[-2]
        try:
            m = json.load(open(mf))
        except Exception:
            continue
        iou = m.get("iou")
        if not isinstance(iou, (int, float)) or iou >= 0.3:
            continue
        try:
            pi = json.load(open(mf.replace("metrics.json", "pdf_info.json")))
            pred_p = mf.replace("metrics.json", "predicted.geojson")
            pred = shape(json.load(open(pred_p))["geometry"])
            if not pred.is_valid:
                pred = pred.buffer(0)
        except Exception:
            continue
        result = verification_score(pi, pred, m.get("match_info"))
        print(f"\n{case[:32]:<32} IoU={iou:.3f}  score={result['score']:.2f}  diag={result['diagnosis']}")
        for cn, cr in result["checks"].items():
            if cr["reason"]:
                print(f"  {cn:<22} {cr['confidence']:.2f}  {cr['reason']}")
        n += 1
        if n >= 8:
            break
