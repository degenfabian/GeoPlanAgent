"""Offline OS Open Names lookup — TRULY FREE, no signup, no card.

OS Open Names is the FREE downloadable dataset (vs. OS Names API which is
on the paid Premium plan). 2.5M GB place names, postcodes, roads,
settlements with sub-metre BNG coordinates. Same data as the API.

Setup (one-time):
  Already downloaded to os_opendata/open_names/csv/Data/ (819 CSVs).

  If missing, fetch (no auth needed):
    curl -L -o opname_csv_gb.zip \\
       "https://api.os.uk/downloads/v1/products/OpenNames/downloads?area=GB&format=CSV&redirect"
    unzip opname_csv_gb.zip -d os_opendata/open_names/csv

Usage:
    from tools.geo.os_names import lookup, search
    hit = lookup("East Langdon")
    # → {'name_full': 'East Langdon', 'type': 'village',
    #    'lat': 51.171, 'lon': 1.345, 'sigma_m': 800,
    #    'source': 'os_open_names:village'}

A first call lazy-loads all CSVs (~3-5s, in-memory ~150MB) and builds a
name index. Subsequent calls are O(log n) for exact-match, O(n) for
fuzzy. Cache is process-local; restart re-loads.
"""
from __future__ import annotations
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Three .parent steps because this file lives at tools.geo.os_names.py
# after the 2026-05-13 reorg (was tools/os_names.py before).
DATA_DIR = (Path(__file__).resolve().parent.parent.parent
            / "os_opendata" / "open_names" / "csv" / "Data")

_HEADER = [
    "ID", "NAMES_URI", "NAME1", "NAME1_LANG", "NAME2", "NAME2_LANG",
    "TYPE", "LOCAL_TYPE", "GEOMETRY_X", "GEOMETRY_Y",
    "MOST_DETAIL_VIEW_RES", "LEAST_DETAIL_VIEW_RES",
    "MBR_XMIN", "MBR_YMIN", "MBR_XMAX", "MBR_YMAX",
    "POSTCODE_DISTRICT", "POSTCODE_DISTRICT_URI",
    "POPULATED_PLACE", "POPULATED_PLACE_URI", "POPULATED_PLACE_TYPE",
    "DISTRICT_BOROUGH", "DISTRICT_BOROUGH_URI", "DISTRICT_BOROUGH_TYPE",
    "COUNTY_UNITARY", "COUNTY_UNITARY_URI", "COUNTY_UNITARY_TYPE",
    "REGION", "REGION_URI", "COUNTRY", "COUNTRY_URI",
    "RELATED_SPATIAL_OBJECT", "SAME_AS_DBPEDIA", "SAME_AS_GEONAMES",
]

_KEEP_COLS = ["NAME1", "TYPE", "LOCAL_TYPE",
              "GEOMETRY_X", "GEOMETRY_Y",
              "DISTRICT_BOROUGH", "COUNTY_UNITARY", "COUNTRY",
              # POPULATED_PLACE is the village/hamlet-level context
              # column (filled on ~16% of rows). The context filter
              # in ``search()`` iterates over POPULATED_PLACE, but
              # without it in _KEEP_COLS the column was silently
              # missing → the village-level disambiguation branch was
              # a no-op (e.g. ``place("Manor Road", la="Cullivoe")``
              # could not match by Cullivoe).
              "POPULATED_PLACE"]

# Sigma (meters) by LOCAL_TYPE — uncertainty about the planning-doc site
# location given this gazetteer hit. The OS BLPU centroid is sub-metre
# accurate, but a "City" feature has 5km extent and the site within it
# could be anywhere; "Section of Named Road" is much tighter.
_SIGMA_BY_TYPE = {
    "city": 4000, "town": 1500, "suburban area": 800, "village": 600,
    "hamlet": 400, "other settlement": 800,
    "section of named road": 200, "named road": 250,
    "postcode": 300,
    "named place": 500, "named area": 1000,
    "spot height": 200, "valley": 1500, "wood or forest": 800,
}
_DEFAULT_SIGMA = 1000


_TABLE: Optional[pd.DataFrame] = None
_NAME_INDEX: Optional[Dict[str, np.ndarray]] = None  # lowercase NAME1 -> row idxs


def _load() -> pd.DataFrame:
    """Lazy-load the full Open Names dataset into a single DataFrame.
    Memory: ~150MB. Loads in ~3-5s on first call. Idempotent."""
    global _TABLE, _NAME_INDEX
    if _TABLE is not None:
        return _TABLE
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"OS Open Names not found at {DATA_DIR}. Run:\n"
            f"  curl -L -o opname.zip 'https://api.os.uk/downloads/v1/products"
            f"/OpenNames/downloads?area=GB&format=CSV&redirect' && unzip "
            f"opname.zip -d os_opendata/open_names/csv"
        )
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {DATA_DIR}")
    parts = []
    for f in files:
        try:
            df = pd.read_csv(f, names=_HEADER, header=None,
                                  usecols=_KEEP_COLS, encoding="utf-8",
                                  on_bad_lines="skip")
            parts.append(df)
        except Exception:
            continue
    _TABLE = pd.concat(parts, ignore_index=True)
    _TABLE["LOCAL_TYPE"] = _TABLE["LOCAL_TYPE"].fillna("").str.lower()
    _TABLE["TYPE"] = _TABLE["TYPE"].fillna("").str.lower()
    # Build lowercase index for O(1) exact lookup
    _TABLE["_NAME_LC"] = _TABLE["NAME1"].fillna("").str.lower().str.strip()
    name_idx: Dict[str, list] = {}
    for i, n in enumerate(_TABLE["_NAME_LC"].values):
        if not n: continue
        name_idx.setdefault(n, []).append(i)
    _NAME_INDEX = {k: np.asarray(v, dtype=np.int64) for k, v in name_idx.items()}
    return _TABLE


def _bng_to_wgs84(easting: float, northing: float) -> Tuple[float, float]:
    """BNG (EPSG:27700) → WGS84 lat/lon. Cached transformer is module-level."""
    global _TRANSFORMER
    try:
        return _TRANSFORMER.transform(easting, northing)[::-1]
    except NameError:
        from pyproj import Transformer
        _TRANSFORMER = Transformer.from_crs(27700, 4326, always_xy=True)
        lon, lat = _TRANSFORMER.transform(easting, northing)
        return lat, lon


def _sigma_for_type(local_type: str) -> int:
    lt = (local_type or "").lower()
    for k, sig in _SIGMA_BY_TYPE.items():
        if k in lt:
            return sig
    return _DEFAULT_SIGMA


def _safe_str(v) -> str:
    """Coerce CSV-loaded values to str; pd reads empty fields as NaN floats."""
    if v is None: return ""
    if isinstance(v, float) and (v != v):  # NaN
        return ""
    return str(v).strip()


def _row_to_hit(row: pd.Series) -> Dict:
    name = _safe_str(row.get("NAME1"))
    bo = _safe_str(row.get("DISTRICT_BOROUGH"))
    co = _safe_str(row.get("COUNTY_UNITARY"))
    name_full = ", ".join(p for p in (name, bo, co) if p)
    lat, lon = _bng_to_wgs84(float(row["GEOMETRY_X"]), float(row["GEOMETRY_Y"]))
    lt = row.get("LOCAL_TYPE", "") or ""
    # Surface ``name``, ``admin_district`` and ``county`` as their own
    # keys. The locate-agent's ``place`` tool reads them under those
    # names — previously ``_row_to_hit`` only set ``name_full`` and
    # folded district/county into it, so the LLM got nulls for all
    # three disambiguation fields and had to spend extra ``la_check``
    # calls.
    return {
        "name_full": name_full or name,
        "name": name or None,
        "type": lt,
        "lat": float(lat), "lon": float(lon),
        "sigma_m": _sigma_for_type(lt),
        "source": f"os_open_names:{lt or 'unknown'}",
        "admin_district": bo or None,
        "county": co or None,
    }


_QUALIFIER_SUFFIXES = (
    " village", " town", " city", " hamlet", " road", " street",
    " lane", " avenue", " way", " borough", " district",
)


def _normalize_query(q: str) -> List[str]:
    """Return [original, stripped-of-qualifier-suffix] candidates."""
    base = q.strip().lower()
    cands = [base]
    for suf in _QUALIFIER_SUFFIXES:
        if base.endswith(suf) and len(base) > len(suf) + 2:
            cands.append(base[:-len(suf)].strip())
    return cands


def _wgs84_bbox_to_bng(lat_min: float, lon_min: float,
                          lat_max: float, lon_max: float) -> Tuple[float, float, float, float]:
    """Convert a WGS84 bbox to a BNG bbox (axis-aligned, slight inflation
    for safety since BNG isn't axis-aligned with WGS84)."""
    from pyproj import Transformer
    t = Transformer.from_crs(4326, 27700, always_xy=True)
    corners = [t.transform(lon, lat) for lat in (lat_min, lat_max)
               for lon in (lon_min, lon_max)]
    xs = [c[0] for c in corners]; ys = [c[1] for c in corners]
    return min(xs), min(ys), max(xs), max(ys)


def search(query: str, max_results: int = 10,
           context: Optional[str] = None,
           bbox_wgs84: Optional[Tuple[float, float, float, float]] = None,
           bbox_radius_km: Optional[float] = None,
           bbox_center: Optional[Tuple[float, float]] = None) -> List[Dict]:
    """Return up to `max_results` hits ranked by exact > prefix > fuzzy.

    Args:
        query: name to look up (case-insensitive). Common qualifier suffixes
            ('village', 'road', 'street', etc.) are stripped before lookup
            so 'East Langdon village' matches NAME1='East Langdon'.
        max_results: cap on returned hits.
        context: optional UK county/region/district to disambiguate. Pre-FILTERS
            the search to rows whose DISTRICT_BOROUGH/COUNTY_UNITARY/COUNTRY
            contains any context token. Falls back to global if no rows match.
        bbox_wgs84: optional (lat_min, lon_min, lat_max, lon_max) to spatially
            constrain the search. Stronger than `context` for disambiguating
            common road names (Manor Road, West Street). Cases where every
            postcode/parish hit is within ~5km of one location should pass
            the bbox derived from those.
        bbox_center, bbox_radius_km: alternative to bbox_wgs84; if both given,
            constructs a bbox of ±(radius/111) degrees around (lat, lon).
    """
    if not query or not query.strip():
        return []
    df = _load()

    rows_pool = df

    # 1. Spatial bbox filter (strongest disambiguator — UK postcodes give
    #    sub-borough precision).
    # 111 km/°: rough mean length of one degree of latitude on the WGS84
    # ellipsoid (varies by ±0.5% with latitude — fine for a filter bbox).
    # 1.6× lon half-width: at mid-UK latitude (~54°), cos(54°) ≈ 0.588,
    # so one degree of longitude is ~65 km. Symmetric-in-km coverage
    # therefore needs lon-degree half-width ≈ 1/0.588 ≈ 1.7× the lat
    # half-width; 1.6 is the safe rounded approximation.
    if bbox_wgs84 is None and bbox_center is not None and bbox_radius_km:
        clat, clon = bbox_center
        d = bbox_radius_km / 111.0
        bbox_wgs84 = (clat - d, clon - 1.6 * d, clat + d, clon + 1.6 * d)
    if bbox_wgs84 is not None:
        lat_min, lon_min, lat_max, lon_max = bbox_wgs84
        x_min, y_min, x_max, y_max = _wgs84_bbox_to_bng(
            lat_min, lon_min, lat_max, lon_max)
        # 500 m BNG inflation: handles place-name records whose
        # GEOMETRY_X/Y is the centroid of a feature whose extent crosses
        # the bbox boundary (parks, large estates). 500 m comfortably
        # exceeds the largest such offset in Open Names while staying
        # tight enough that the resulting candidate pool is small.
        x_min -= 500; y_min -= 500; x_max += 500; y_max += 500
        spatial_mask = (
            (df["GEOMETRY_X"] >= x_min) & (df["GEOMETRY_X"] <= x_max) &
            (df["GEOMETRY_Y"] >= y_min) & (df["GEOMETRY_Y"] <= y_max)
        )
        if spatial_mask.any():
            rows_pool = df[spatial_mask]

    # 2. Pre-filter by context if provided (cumulative with spatial filter).
    #    Strip "District"/"Borough" suffixes so "South Norfolk District" matches
    #    DISTRICT_BOROUGH="South Norfolk".
    if context:
        ctx = context.strip().lower()
        for suf in (" district", " borough", " unitary", " county", " council"):
            if ctx.endswith(suf):
                ctx = ctx[:-len(suf)].strip()
        ctx_tokens = [t.strip() for t in ctx.replace(",", " ").split() if len(t.strip()) > 2]
        if ctx_tokens:
            mask = pd.Series(False, index=rows_pool.index)
            for col in ("DISTRICT_BOROUGH", "COUNTY_UNITARY", "POPULATED_PLACE"):
                if col not in rows_pool.columns:
                    continue
                col_lc = rows_pool[col].fillna("").astype(str).str.lower()
                for tok in ctx_tokens:
                    mask = mask | col_lc.str.contains(tok, na=False, regex=False)
            if mask.any():
                rows_pool = rows_pool[mask]

    pool_name_lc = rows_pool["_NAME_LC"]
    qcands = _normalize_query(query)
    idxs: List[int] = []
    seen_idx: set = set()

    for q in qcands:
        # 1. Exact match within filtered pool
        exact = rows_pool.index[pool_name_lc == q].tolist()
        for i in exact:
            if i not in seen_idx:
                idxs.append(int(i)); seen_idx.add(i)
        if len(idxs) >= max_results:
            break
        # 2. Prefix match within filtered pool
        prefix = rows_pool.index[pool_name_lc.str.startswith(q, na=False)].tolist()
        for i in prefix:
            if i not in seen_idx:
                idxs.append(int(i)); seen_idx.add(i)
        if len(idxs) >= max_results:
            break

    # 3. Fuzzy fallback within filtered pool (only if nothing exact)
    if not idxs:
        try:
            from rapidfuzz import process, fuzz
            pool_names = pool_name_lc.unique().tolist()
            for q in qcands:
                cands = process.extract(q, pool_names, scorer=fuzz.WRatio,
                                          limit=max_results, score_cutoff=85)
                for cn, score, _ in cands:
                    matches = rows_pool.index[pool_name_lc == cn].tolist()
                    for i in matches:
                        if i not in seen_idx:
                            idxs.append(int(i)); seen_idx.add(i)
                if idxs: break
        except Exception:
            pass

    if not idxs:
        # Context filter eliminated everything? Retry globally — but
        # KEEP the bbox / radius constraints. A caller that supplied
        # both context and a bbox is using the bbox as the stronger
        # spatial signal (it's typically sub-borough); dropping it on
        # the fallback would silently widen the recursive search to
        # the whole UK and could return wrong-LA hits the bbox was
        # supposed to prevent.
        if context and rows_pool is not df:
            return search(query, max_results=max_results, context=None,
                           bbox_wgs84=bbox_wgs84,
                           bbox_center=bbox_center,
                           bbox_radius_km=bbox_radius_km)
        return []

    rows = df.iloc[idxs]
    seen_keys = set()
    hits = []
    for _, row in rows.head(max_results * 3).iterrows():
        key = (_safe_str(row.get("NAME1")).lower(),
               round(float(row["GEOMETRY_X"]) / 100),
               round(float(row["GEOMETRY_Y"]) / 100))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        hits.append(_row_to_hit(row))
        if len(hits) >= max_results:
            break
    return hits


def lookup(query: str, context: Optional[str] = None,
           bbox_center: Optional[Tuple[float, float]] = None,
           bbox_radius_km: Optional[float] = None) -> Optional[Dict]:
    """Single-best-hit lookup. Returns the top hit or None."""
    hits = search(query, max_results=1, context=context,
                       bbox_center=bbox_center, bbox_radius_km=bbox_radius_km)
    return hits[0] if hits else None


def lookup_postcode(postcode: str) -> Optional[Dict]:
    """Lookup a UK postcode — returns the postcode centroid (BLPU sub-metre
    BNG). OS Open Names stores 1.74M full postcodes with NAME1='NW3 7QR'
    etc. Falls back to averaging all postcodes sharing the outward code if
    only outward (e.g. 'NW3') was provided.
    """
    if not postcode: return None
    df = _load()
    pc = postcode.strip().upper()
    # Normalize: insert space if missing ("NW37QR" -> "NW3 7QR")
    norm = pc.replace(" ", "")
    if len(norm) > 4:
        # Full postcode: outward (3-4 chars) + inward (3 chars)
        full = f"{norm[:-3]} {norm[-3:]}"
    else:
        full = norm
    # Try full postcode exact
    if full.lower() in (_NAME_INDEX or {}):
        idxs = _NAME_INDEX[full.lower()]
        return _row_to_hit(df.iloc[idxs[0]])
    # Try outward-only: NAME1 starts with the outward + space
    outward = norm[:-3] if len(norm) > 4 else norm
    mask = (df["_NAME_LC"].str.startswith(outward.lower() + " ", na=False) &
            df["LOCAL_TYPE"].str.contains("postcode", na=False, case=False))
    idxs = df.index[mask].tolist()
    if not idxs: return None
    # Return centroid of all matching postcodes (outward area centroid)
    sub = df.iloc[idxs]
    cx = float(sub["GEOMETRY_X"].mean())
    cy = float(sub["GEOMETRY_Y"].mean())
    lat, lon = _bng_to_wgs84(cx, cy)
    return {
        "name_full": f"{outward} (outward area)",
        "type": "postcode_outward",
        "lat": float(lat), "lon": float(lon),
        "sigma_m": 1500,  # outward postcodes are ~1-3 sq km
        "source": "os_open_names:postcode_outward",
        "n_subcodes": len(idxs),
    }


def is_loaded() -> bool:
    return _TABLE is not None


if __name__ == "__main__":
    import sys, time
    if len(sys.argv) < 2:
        print("usage: python -m tools.geo.os_names <query> [context]")
        sys.exit(1)
    t0 = time.time()
    df = _load()
    print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    q = sys.argv[1]
    ctx = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"Query: {q!r}  context={ctx!r}")
    for h in search(q, max_results=5, context=ctx):
        print(f"  {h['name_full']:50s} {h['type']:25s} "
              f"({h['lat']:.5f}, {h['lon']:.5f}) σ={h['sigma_m']}m")
