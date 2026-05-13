"""
Enhanced geocoding utilities: Photon API, place-name centers, and center filtering.

All APIs used are open-source / open-data compatible with academic research:
  - Photon (Apache 2.0, OSM data under ODbL)
  - Postcodes.io (MIT, ONS open data)
  - Nominatim (ODbL)
  - OS Open Zoomstack names layer (Open Government Licence v3.0,
    Crown copyright and database right)
"""

import json
import math
import os
import sqlite3
import time

# Module-level Nominatim throttle. Free tier requires ≥1 req/sec, so we
# enforce a 1.1s gap between any two calls. Updated by nominatim_structured.
_NOMINATIM_LAST_CALL: float = 0.0
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional, Dict

# Center tuple: (name, lat, lon, sigma_m)
Center = Tuple[str, float, float, Optional[float]]

PHOTON_URL = "https://photon.komoot.io/api/"
COUNCIL_CACHE_DIR = "cache/council_boundaries"

# UK bounding box for coordinate validation
UK_BBOX = {"lat_min": 49.0, "lat_max": 61.0, "lon_min": -8.5, "lon_max": 2.0}


# ── OS Zoomstack places gazetteer (offline, free OGL v3) ───────────────────

_ZOOMSTACK_GPKG = Path(__file__).resolve().parent.parent / "os_opendata" / "OS_Open_Zoomstack.gpkg"

# Specificity ranking — lower rank = more specific / preferred for place queries.
# Populated places first (usually what planning docs reference), then named
# buildings/sites, then natural features, then way-too-broad admin areas.
_TYPE_RANK = {
    "City": 1, "Town": 1, "Village": 1, "Hamlet": 1,
    "Suburban Area": 2, "Small Settlements": 2,
    "Sites": 3, "Motorway Junctions": 3,
    "Water": 4, "Landform": 4, "Woodland": 4,
    "Greenspace": 5, "Landcover": 5,
    "National Park": 9, "Country": 10,
}

# Lazy-loaded transformer (OSGB36 → WGS84)
_ZS_TRANSFORMER = None
def _osgb_to_wgs84():
    global _ZS_TRANSFORMER
    if _ZS_TRANSFORMER is None:
        from pyproj import Transformer
        _ZS_TRANSFORMER = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    return _ZS_TRANSFORMER


def _parse_gpkg_point(blob: bytes) -> Optional[Tuple[float, float]]:
    """Parse a GeoPackage binary POINT blob to (easting_m, northing_m) in OSGB36.
    Returns None if blob is unexpected or empty.
    """
    if not blob or len(blob) < 8 or blob[:2] != b"GP":
        return None
    flags = blob[3]
    envelope_type = (flags >> 1) & 0x07
    env_size = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get(envelope_type, 0)
    wkb_bytes = bytes(blob[8 + env_size:])
    # WKB: byte order (1) + type (4, little-endian=1 for POINT) + x (8) + y (8)
    if len(wkb_bytes) < 21:
        return None
    import struct
    byte_order = wkb_bytes[0]
    endian = "<" if byte_order == 1 else ">"
    geom_type = struct.unpack(endian + "I", wkb_bytes[1:5])[0]
    if geom_type != 1:  # not a POINT
        return None
    x, y = struct.unpack(endian + "dd", wkb_bytes[5:21])
    return x, y


def gpkg_place_search(
    query: str,
    parent_lat: Optional[float] = None,
    parent_lon: Optional[float] = None,
    max_parent_distance_km: float = 30.0,
    type_filter: Optional[List[str]] = None,
    limit: int = 10,
    gpkg_path: Optional[Path] = None,
) -> List[Dict]:
    """Search the OS Open Zoomstack `names` table for a place name.

    Sorts results by:
      1. Exact match beats LIKE match
      2. Distance from parent_lat/lon (if given) — closer first
      3. Specificity rank (populated places > buildings > natural features)

    Args:
        query: Place name to search for.
        parent_lat/parent_lon: Optional anchor point (e.g. district center)
            to disambiguate. Results further than max_parent_distance_km
            are filtered out.
        max_parent_distance_km: If parent given, drop results beyond this.
        type_filter: If set, only return entries of these types.
        limit: Max results to return.

    Returns:
        List of {"name", "type", "lat", "lon", "exact", "specificity",
                 "distance_from_parent_km"} dicts sorted best-first.
    """
    gpkg = Path(gpkg_path) if gpkg_path else _ZOOMSTACK_GPKG
    if not gpkg.exists():
        return []
    q = (query or "").strip()
    if len(q) < 2:
        return []

    results = []
    conn = sqlite3.connect(str(gpkg))
    try:
        cur = conn.cursor()
        # Pull exact and LIKE candidates together; we'll rank them.
        sql = ("SELECT name1, name2, type, geom FROM names "
               "WHERE UPPER(name1) LIKE UPPER(?) OR UPPER(name2) LIKE UPPER(?) "
               "LIMIT 200")
        cur.execute(sql, (f"%{q}%", f"%{q}%"))
        rows = cur.fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()

    if not rows:
        return []

    transformer = _osgb_to_wgs84()
    q_upper = q.upper()
    for name1, name2, type_, geom_blob in rows:
        if type_filter and type_ not in type_filter:
            continue
        if not geom_blob:
            continue
        pt = _parse_gpkg_point(geom_blob)
        if pt is None:
            continue
        x_osgb, y_osgb = pt
        lon, lat = transformer.transform(x_osgb, y_osgb)
        if not _is_valid_uk_coord(lat, lon):
            continue

        # Exact match check (name1 or name2 matches the query exactly)
        is_exact = (
            (name1 and name1.upper() == q_upper) or
            (name2 and name2.upper() == q_upper)
        )
        specificity = _TYPE_RANK.get(type_, 6)

        # Distance from parent (if given)
        parent_dist_km = None
        if parent_lat is not None and parent_lon is not None:
            parent_dist_km = _distance_m(parent_lat, parent_lon, lat, lon) / 1000.0
            if parent_dist_km > max_parent_distance_km:
                continue

        results.append({
            "name": name1,
            "type": type_,
            "lat": lat,
            "lon": lon,
            "exact": bool(is_exact),
            "specificity": specificity,
            "distance_from_parent_km": parent_dist_km,
        })

    # Sort: exact matches first, then by parent distance (if known), then by
    # specificity rank. Absent parent, specificity dominates.
    def sort_key(r):
        return (
            0 if r["exact"] else 1,
            r["distance_from_parent_km"] if r["distance_from_parent_km"] is not None else 10.0,
            r["specificity"],
        )
    results.sort(key=sort_key)
    return results[:limit]


def _is_valid_uk_coord(lat: float, lon: float) -> bool:
    """Check if coordinates fall within the UK bounding box."""
    return (UK_BBOX["lat_min"] <= lat <= UK_BBOX["lat_max"] and
            UK_BBOX["lon_min"] <= lon <= UK_BBOX["lon_max"])


def _retry_urlopen(req, timeout=10, retries=3, label="request"):
    """URL fetch with exponential backoff retry on transient failures."""
    last_error = None
    for attempt in range(retries):
        if attempt > 0:
            wait = 2 ** attempt  # 2s, 4s
            print(f"    WARN:Retry {attempt}/{retries-1} for {label} after {wait}s...")
            time.sleep(wait)
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                last_error = e
                print(f"    WARN:{label}: HTTP {e.code}, retrying...")
                continue
            raise  # 400, 404 etc. are not retryable
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = e
            print(f"    WARN:{label}: {e}, retrying...")
            continue
    raise last_error or Exception(f"{label}: all {retries} retries failed")


# ── Photon geocoding ─────────────────────────────────────────────────────────

# File-backed cache for Wikidata place-search results. Wikidata calls are
# ~0.3s each (two HTTP round-trips per query), so for a 182-case benchmark
# with ≤5 place_names per case the cache saves up to 5 minutes per rerun.
# Keyed by normalised query + rounded parent coords + parent distance cap.
_WIKIDATA_CACHE_PATH = Path("cache/wikidata_place.json")
_WIKIDATA_CACHE: Optional[Dict[str, List[Dict]]] = None


def _load_wikidata_cache() -> Dict[str, List[Dict]]:
    global _WIKIDATA_CACHE
    if _WIKIDATA_CACHE is not None:
        return _WIKIDATA_CACHE
    try:
        if _WIKIDATA_CACHE_PATH.exists():
            _WIKIDATA_CACHE = json.loads(_WIKIDATA_CACHE_PATH.read_text())
        else:
            _WIKIDATA_CACHE = {}
    except Exception:
        _WIKIDATA_CACHE = {}
    return _WIKIDATA_CACHE


def _save_wikidata_cache() -> None:
    if _WIKIDATA_CACHE is None:
        return
    try:
        _WIKIDATA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _WIKIDATA_CACHE_PATH.write_text(json.dumps(_WIKIDATA_CACHE, indent=2))
    except Exception:
        pass


def _wd_cache_key(query: str, parent_lat: Optional[float],
                   parent_lon: Optional[float], max_km: float,
                   limit: int) -> str:
    p_lat = f"{parent_lat:.2f}" if parent_lat is not None else "-"
    p_lon = f"{parent_lon:.2f}" if parent_lon is not None else "-"
    return f"{query.strip().lower()}|{p_lat}|{p_lon}|{max_km:.0f}|{limit}"


def wikidata_place_search(
    query: str,
    parent_lat: Optional[float] = None,
    parent_lon: Optional[float] = None,
    max_parent_distance_km: float = 30.0,
    limit: int = 5,
) -> List[Dict]:
    """Query Wikidata for a named feature with UK + parent-distance filter.

    Useful for named buildings (Colney Hall), conservation areas
    (Belsize Park), historic landmarks — entries that aren't in OS Open
    Names because they aren't gazetteer-listed places.

    Two-step (wbsearchentities → wbgetentities). ~0.3s per call. Results
    cached to cache/wikidata_place.json; the empty list is cached too, so
    "no match" queries aren't re-sent on rerun.

    Args:
        query: free-text place name.
        parent_lat/lon: optional anchor for disambiguation.
        max_parent_distance_km: drop hits beyond this from parent.
        limit: max results to return.

    Returns:
        List of {"name", "lat", "lon", "qid", "distance_from_parent_km"},
        sorted by parent distance (if given) else by Wikidata's own
        relevance ordering.
    """
    if not query or len(query.strip()) < 3:
        return []
    q = query.strip()

    cache = _load_wikidata_cache()
    ck = _wd_cache_key(q, parent_lat, parent_lon, max_parent_distance_km, limit)
    if ck in cache:
        return cache[ck] or []

    # Step 1: search for entity IDs
    try:
        params = urllib.parse.urlencode({
            "action": "wbsearchentities", "search": q,
            "language": "en", "format": "json", "type": "item",
            "limit": str(min(limit * 2, 10)),
        })
        url = f"https://www.wikidata.org/w/api.php?{params}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "GeoMapAgent-research/0.1 (UK planning)"})
        resp = _retry_urlopen(req, timeout=8, label=f"WD-search({q[:30]})")
        data = json.loads(resp.read())
        resp.close()
        entities = data.get("search", [])
    except Exception:
        # Network error — don't cache (let future runs retry)
        return []

    if not entities:
        cache[ck] = []
        _save_wikidata_cache()
        return []

    qids = [e["id"] for e in entities[:10]]

    # Step 2: fetch coordinates for those entities
    try:
        params = urllib.parse.urlencode({
            "action": "wbgetentities", "ids": "|".join(qids),
            "format": "json", "props": "claims|labels", "languages": "en",
        })
        url = f"https://www.wikidata.org/w/api.php?{params}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "GeoMapAgent-research/0.1 (UK planning)"})
        resp = _retry_urlopen(req, timeout=12, label=f"WD-coords({q[:30]})")
        ent_data = json.loads(resp.read()).get("entities", {})
        resp.close()
    except Exception:
        return []

    out = []
    for qid, ent in ent_data.items():
        claims = ent.get("claims", {})
        p625 = claims.get("P625") or []
        if not p625:
            continue
        try:
            v = p625[0]["mainsnak"]["datavalue"]["value"]
            lat = float(v["latitude"]); lon = float(v["longitude"])
        except Exception:
            continue
        if not _is_valid_uk_coord(lat, lon):
            continue
        label = ent.get("labels", {}).get("en", {}).get("value", qid)
        parent_dist_km = None
        if parent_lat is not None and parent_lon is not None:
            parent_dist_km = _distance_m(parent_lat, parent_lon, lat, lon) / 1000.0
            if parent_dist_km > max_parent_distance_km:
                continue
        out.append({
            "name": label, "qid": qid, "lat": lat, "lon": lon,
            "distance_from_parent_km": parent_dist_km,
        })

    # Sort: closest to parent first if given, else preserve Wikidata ranking
    out.sort(key=lambda r: r["distance_from_parent_km"]
             if r["distance_from_parent_km"] is not None else 999.0)
    out = out[:limit]
    cache[ck] = out
    _save_wikidata_cache()
    return out


# File-backed cache for Nominatim results. Both production and offline testing
# benefit — reruns don't hit the API again and avoid 429 rate limiting. Keyed
# by normalised query params. Stored next to the GPKG so it travels with the
# repo's OS data.
_NOMINATIM_CACHE_PATH = Path("cache/nominatim_structured.json")
_NOMINATIM_CACHE = None


def _load_nominatim_cache():
    global _NOMINATIM_CACHE
    if _NOMINATIM_CACHE is not None:
        return _NOMINATIM_CACHE
    try:
        if _NOMINATIM_CACHE_PATH.exists():
            _NOMINATIM_CACHE = json.loads(_NOMINATIM_CACHE_PATH.read_text())
        else:
            _NOMINATIM_CACHE = {}
    except Exception:
        _NOMINATIM_CACHE = {}
    return _NOMINATIM_CACHE


def _save_nominatim_cache():
    if _NOMINATIM_CACHE is None:
        return
    try:
        _NOMINATIM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _NOMINATIM_CACHE_PATH.write_text(json.dumps(_NOMINATIM_CACHE, indent=2))
    except Exception:
        pass


def _nominatim_cache_key(street, city, county, country):
    return "|".join(x.strip().lower() for x in (street, city, county, country))


def nominatim_structured(street: str = "", city: str = "",
                         county: str = "", country: str = "UK",
                         limit: int = 2,
                         viewbox: Optional[Tuple[float, float, float, float]] = None,
                         bounded: bool = False) -> Optional[Dict]:
    """Query OSM Nominatim with structured fields for precise UK addresses.

    Better than Photon for concrete house-numbered addresses because it
    uses OSM's address index directly and returns one definitive result.
    Rate limited to 1 req/sec politely; adds a User-Agent as required.
    Results are cached to cache/nominatim_structured.json so reruns skip
    the HTTP call.

    Args:
        street: e.g. "123 High Street"
        city: e.g. "London" or "Rossendale"
        county: optional county
        country: default "UK"
        limit: max results (usually take first)

    Returns:
        {"lat", "lon", "display_name", "osm_type"} or None if no match.
    """
    # Viewbox is included in the cache key so bounded/unbounded queries
    # don't collide.
    vb_key = "-"
    if viewbox is not None and len(viewbox) == 4:
        vb_key = f"{viewbox[0]:.2f},{viewbox[1]:.2f},{viewbox[2]:.2f},{viewbox[3]:.2f}|{int(bool(bounded))}"
    cache = _load_nominatim_cache()
    ck = _nominatim_cache_key(street, city, county, country) + "|" + vb_key
    if ck in cache:
        v = cache[ck]
        # Cached None = no match, avoid re-querying
        return v if v else None

    params = {"format": "json", "limit": str(limit), "addressdetails": "1"}
    if street: params["street"] = street
    if city: params["city"] = city
    if county: params["county"] = county
    if country: params["country"] = country
    # Nominatim viewbox: x1,y1,x2,y2 as min_lon,max_lat,max_lon,min_lat.
    # With bounded=1 the API returns ONLY results inside the box.
    if viewbox is not None and len(viewbox) == 4:
        params["viewbox"] = ",".join(f"{v:.6f}" for v in viewbox)
        if bounded:
            params["bounded"] = "1"
    qs = urllib.parse.urlencode(params)
    url = f"https://nominatim.openstreetmap.org/search?{qs}"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "GeoMapAgent-research/0.1 (UK planning)"})
        # Nominatim free tier rate limit: ≥1 req/sec. Without a global
        # throttle, firing N fresh calls in tight succession hits 429 on
        # all but the first. Sleep until the previous call was ≥1.1s ago.
        global _NOMINATIM_LAST_CALL
        try:
            elapsed = time.time() - _NOMINATIM_LAST_CALL
        except NameError:
            elapsed = 999.0
        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)
        _NOMINATIM_LAST_CALL = time.time()
        resp = _retry_urlopen(req, timeout=10, label=f"Nominatim({street[:30]})")
        data = json.loads(resp.read())
        resp.close()
        for r in data:
            try:
                lat = float(r["lat"]); lon = float(r["lon"])
            except Exception:
                continue
            if not _is_valid_uk_coord(lat, lon):
                continue
            # City-verification: reject results whose display_name doesn't
            # mention the requested city or county. Catches wrong-sense
            # matches like Nominatim picking "Pipers Lane" in a different
            # region when the requested city was "Heswall". We allow a
            # short-token fuzzy check (case-insensitive substring match).
            disp = (r.get("display_name") or "").lower()
            if city and len(city) >= 3:
                city_tokens = [t for t in city.lower().split()
                               if len(t) >= 3 and t not in
                               {"the", "and", "of", "district", "council",
                                "borough", "city", "town"}]
                if city_tokens and not any(t in disp for t in city_tokens):
                    continue  # try next candidate
            out = {
                "lat": lat, "lon": lon,
                "display_name": r.get("display_name", ""),
                "osm_type": r.get("osm_type", ""),
            }
            cache[ck] = out
            _save_nominatim_cache()
            return out
        # No valid match — cache the null so we don't re-query
        cache[ck] = None
        _save_nominatim_cache()
        return None
    except Exception as e:
        print(f"    WARN:Nominatim failed for '{street[:30]}, {city}': {e}")
        # Do NOT cache on exception (network error, 429 etc.) — might be
        # transient, let a future run retry.
        return None


def query_photon(address: str, limit: int = 3) -> list:
    """Query Photon geocoder with retry. Returns list of {lat, lon, osm_type, name}."""
    params = urllib.parse.urlencode({"q": address, "limit": limit, "lang": "en"})
    url = f"{PHOTON_URL}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GeoMapAgent/1.0"})
        resp = _retry_urlopen(req, timeout=10, label=f"Photon({address[:40]})")
        data = json.loads(resp.read())
        resp.close()
        results = []
        for feat in data.get("features", []):
            coords = feat.get("geometry", {}).get("coordinates", [])
            props = feat.get("properties", {})
            if len(coords) >= 2:
                lat, lon = coords[1], coords[0]
                if not _is_valid_uk_coord(lat, lon):
                    print(f"    WARN:Photon: skipping non-UK result "
                          f"({lat:.2f}, {lon:.2f}) for '{address[:40]}'")
                    continue
                results.append({
                    "lat": lat,
                    "lon": lon,
                    "osm_type": props.get("osm_type", ""),
                    "name": props.get("name", ""),
                    "city": props.get("city", ""),
                    "country": props.get("country", ""),
                })
        return results
    except Exception as e:
        print(f"    WARN:Photon failed for '{address[:40]}': {e}")
        return []


def _distance_m(lat1, lon1, lat2, lon2):
    """Approximate distance in meters between two lat/lon points."""
    dlat = (lat2 - lat1) * 111111
    dlon = (lon2 - lon1) * 111111 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.sqrt(dlat**2 + dlon**2)


def cross_validate_centers(centers: List[Center], max_outlier_km: float = 10) -> List[Center]:
    """Drop centers that are >threshold from the median center.

    Uses adaptive thresholding: when many centers agree tightly (IQR < 2km),
    the threshold tightens to 3x IQR (min 2km). Otherwise uses max_outlier_km.

    Median is computed from the highest-specificity subset when present
    (Nominatim street/addr, grid_refs, postcode, Zoomstack Town/City/Village)
    so a wrong-sense Sites/Greenspace/county hit can't drag the median off
    and cause the correct street-level anchor to be dropped as an "outlier".

    If fewer than 3 centers but a rank-≤1 anchor exists (Nominatim street,
    grid-ref, postcode), still drop centers >max_outlier_km from it —
    otherwise a wrong-sense gpkg hit 100+km away (e.g. gpkg:Camdentown in
    Hampshire while we mean Camden in London) slips through and confuses
    MINIMA.
    """
    # Source-prefix ranking: lower = more precise. Delegates to the canonical
    # `_center_specificity` table in tools.matching (consolidated 2026-05-11).
    # Lazy import to avoid a circular dependency at module load time.
    from tools.matching import _center_specificity as _spec  # noqa: F401

    # Fast path for small center lists: anchor-based drop only
    if len(centers) < 3:
        anchors = [c for c in centers if _spec(c[0]) <= 1]
        if not anchors:
            return centers
        # Use the single most-specific anchor
        anchors.sort(key=lambda c: _spec(c[0]))
        a_lat, a_lon = anchors[0][1], anchors[0][2]
        kept = []
        for c in centers:
            if _spec(c[0]) <= 1:
                kept.append(c)
                continue
            d = _distance_m(c[1], c[2], a_lat, a_lon)
            if d <= max_outlier_km * 1000:
                kept.append(c)
            else:
                print(f"  Cross-validate: dropped {c[0]} ({d/1000:.1f}km "
                      f"from anchor {anchors[0][0]!r})")
        return kept if kept else centers

    # Prefer specific centers when computing the median. If ≥1 rank-≤2
    # centers exist, use only those for the median anchor. Else fall back
    # to all centers (legacy behaviour).
    specific = [c for c in centers if _spec(c[0]) <= 2]
    median_source = specific if len(specific) >= 1 else centers

    lats = [c[1] for c in median_source]
    lons = [c[2] for c in median_source]
    med_lat = np.median(lats)
    med_lon = np.median(lons)

    # Compute distances from median for adaptive threshold
    dists = [_distance_m(c[1], c[2], med_lat, med_lon) for c in centers]
    dists_sorted = sorted(dists)
    q1 = dists_sorted[len(dists_sorted) // 4]
    q3 = dists_sorted[3 * len(dists_sorted) // 4]
    iqr = q3 - q1

    # Adaptive threshold: if centers are tightly clustered (IQR < 2km),
    # use 3x IQR (min 2km) instead of the default 10km
    if len(centers) >= 5 and iqr < 2000:
        threshold_m = max(2000, 3 * iqr)
        print(f"  Cross-validate: adaptive threshold={threshold_m:.0f}m "
              f"(IQR={iqr:.0f}m, {len(centers)} centers)")
    else:
        threshold_m = max_outlier_km * 1000

    kept = []
    dropped = []
    for c, d in zip(centers, dists):
        if d <= threshold_m:
            kept.append(c)
        else:
            dropped.append((c[0], d / 1000))

    if dropped:
        for name, dist_km in dropped:
            print(f"  Cross-validate: dropped {name} ({dist_km:.1f}km from median)")

    # If we'd remove everything, return originals
    return kept if kept else centers


def parse_scale_ratio(analysis, ve=None):
    """Extract scale_ratio from analysis.json, falling back to visual_extract."""
    scale_ratio = analysis.get("scale_ratio")
    if scale_ratio is not None:
        return scale_ratio
    if ve and ve.get("scale"):
        m = re.search(r"1\s*[:/]\s*([\d,]+)", str(ve["scale"]))
        if m:
            try:
                return int(m.group(1).replace(",", ""))
            except ValueError:
                pass
    return None


def query_postcodes_io_bulk(postcodes, cache):
    """Bulk query postcodes.io API, using and updating cache.

    Returns dict of postcode -> {lat, lon, admin_district}.
    """
    results = {}
    for pc in postcodes:
        pc_norm = pc.strip().upper().replace(" ", "")
        if pc_norm in cache and cache[pc_norm]:
            results[pc] = cache[pc_norm]

    to_query = [pc.strip() for pc in postcodes
                if pc.strip().upper().replace(" ", "") not in cache]
    if not to_query:
        return results

    if len(to_query) > 100:
        print(f"    WARN:WARNING: {len(to_query)} postcodes, querying first 100 only")
    payload = json.dumps({"postcodes": to_query[:100]}).encode()
    req = urllib.request.Request(
        "https://api.postcodes.io/postcodes",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = _retry_urlopen(req, timeout=15, label="postcodes.io")
        data = json.loads(resp.read())
        resp.close()
        for item in data.get("result", []):
            pc_norm = item["query"].strip().upper().replace(" ", "")
            if item["result"]:
                entry = {
                    "lat": item["result"]["latitude"],
                    "lon": item["result"]["longitude"],
                    "admin_district": item["result"].get("admin_district", ""),
                }
                cache[pc_norm] = entry
                results[item["query"]] = entry
    except Exception as e:
        print(f"  WARN:postcodes.io failed after retries: {e}")
    return results


