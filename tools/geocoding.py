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

    Two-step:
      1. wbsearchentities to find candidate entity IDs by label.
      2. wbgetentities to fetch coordinate (P625) and label.

    Both APIs are free (MediaWiki API on wikidata.org), no auth, no
    license restriction. Polite User-Agent. ~0.3s per request.

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
        return []

    if not entities:
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
    return out[:limit]


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
                         limit: int = 2) -> Optional[Dict]:
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
    cache = _load_nominatim_cache()
    ck = _nominatim_cache_key(street, city, county, country)
    if ck in cache:
        v = cache[ck]
        # Cached None = no match, avoid re-querying
        return v if v else None

    params = {"format": "json", "limit": str(limit), "addressdetails": "1"}
    if street: params["street"] = street
    if city: params["city"] = city
    if county: params["county"] = county
    if country: params["country"] = country
    qs = urllib.parse.urlencode(params)
    url = f"https://nominatim.openstreetmap.org/search?{qs}"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "GeoMapAgent-research/0.1 (UK planning)"})
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


def photon_centers(visual_extract: dict) -> List[Center]:
    """Generate centers from Photon geocoding of address data.

    Tries site_address, then geocode_queries, then road+place combos.
    """
    centers = []
    seen = set()

    def _add(name, lat, lon, sigma):
        key = (round(lat, 4), round(lon, 4))
        if key not in seen:
            seen.add(key)
            centers.append((name, lat, lon, sigma))

    def _try_query(query, prefix, limit=3):
        time.sleep(0.5)
        results = query_photon(query, limit=limit)
        if results:
            for j, r in enumerate(results):
                # First result gets tight sigma, others get wider
                sigma = 200 if j == 0 else 500
                suffix = "" if j == 0 else f"_r{j}"
                _add(f"{prefix}{suffix}_{sigma}", r["lat"], r["lon"], sigma)
                if j == 0:
                    _add(f"{prefix}_500", r["lat"], r["lon"], 500)
            return True
        return False

    # 1. Site address
    site_addr = visual_extract.get("site_address", "")
    if site_addr:
        _try_query(site_addr + ", UK", "photon_addr")

    # 2. Geocode queries — query first 4 (gq0-gq3); gq4 is harmful per ablation
    for i, gq in enumerate(visual_extract.get("geocode_queries", [])[:4]):
        _try_query(gq, f"photon_gq{i}")

    # 3. Road name + place name combos
    roads = visual_extract.get("road_names_on_map", [])[:3]
    places = visual_extract.get("place_names", [])[:2]
    if roads and places:
        combo = f"{roads[0]}, {places[0]}, UK"
        _try_query(combo, "photon_road_place")

    # 4. Council/authority name qualifier (disambiguates common place names)
    council = visual_extract.get("council_name") or visual_extract.get("local_authority")
    if council and roads:
        combo = f"{roads[0]}, {council}, UK"
        _try_query(combo, "photon_council_road")

    return centers


def place_name_centers(visual_extract: dict) -> List[Center]:
    """Disabled: ablation showed zero primary/oracle picks across 179 cases."""
    return []


# ── Center filtering ─────────────────────────────────────────────────────────

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
    # Source-prefix ranking: lower = more precise. Keep in sync with
    # tools.positioning._center_specificity.
    _HIGH_SPEC = {"Town", "City", "Village", "Hamlet", "Suburb"}

    def _spec(name):
        n = (name or "").lower()
        if n.startswith("nominatim:addr:"): return 0
        if n.startswith("nominatim:"): return 1
        if n.startswith("grid_refs_centroid") or n.startswith("gridref:"): return 1
        if n.startswith("postcode:"): return 1
        if n.startswith("gpkg:") and "(" in (name or "") and ")" in (name or ""):
            t = name.rsplit("(", 1)[-1].rstrip(")")
            if t in _HIGH_SPEC: return 2
        return 5

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


def road_name_precheck(center_lat: float, center_lon: float,
                       road_names: list, osm_data: dict,
                       radius_m: float = 2000) -> bool:
    """Check if any LLM-extracted road names exist in OSM near this center.

    Returns True if at least one road name matches within radius.
    Returns True if no road names provided (can't filter).
    """
    if not road_names or not osm_data:
        return True

    llm_lower = {n.lower().strip() for n in road_names if n.strip()}
    if not llm_lower:
        return True

    lat_tol = radius_m / 111111.0
    cos_lat = math.cos(math.radians(center_lat))
    lon_tol = radius_m / (111111.0 * cos_lat) if cos_lat > 0 else lat_tol

    for el in osm_data.get("elements", []):
        if el.get("type") != "way":
            continue
        name = el.get("tags", {}).get("name", "").lower().strip()
        if not name:
            continue

        # Check if this way has geometry near center
        near = False
        for pt in el.get("geometry", []):
            if (abs(pt["lat"] - center_lat) <= lat_tol and
                    abs(pt["lon"] - center_lon) <= lon_tol):
                near = True
                break
        if not near:
            continue

        # Check name match
        for ln in llm_lower:
            if ln in name or name in ln:
                return True

    return False


def postcode_district_filter(centers: List[Center], postcodes: list,
                             pc_cache: dict, max_dist_km: float = 15) -> List[Center]:
    """Drop centers that are too far from any known postcode centroid.

    If no postcodes available, returns all centers.
    """
    if not postcodes or not pc_cache:
        return centers

    # Collect postcode centroids
    pc_coords = []
    for pc in postcodes:
        pc_str = pc if isinstance(pc, str) else pc.get("postcode", "")
        pc_norm = pc_str.strip().upper().replace(" ", "")
        entry = pc_cache.get(pc_norm)
        if entry and "lat" in entry:
            pc_coords.append((entry["lat"], entry["lon"]))

    if not pc_coords:
        return centers

    kept = []
    for c in centers:
        min_dist = min(_distance_m(c[1], c[2], pc[0], pc[1]) for pc in pc_coords)
        if min_dist <= max_dist_km * 1000:
            kept.append(c)

    return kept if kept else centers


def council_boundary_filter(centers: List[Center], council_name: str,
                            buffer_km: float = 2) -> List[Center]:
    """Drop centers outside the named council boundary + buffer.

    Fetches boundary from Nominatim (cached). Falls through if boundary
    can't be fetched.
    """
    if not council_name or not centers:
        return centers

    boundary = _fetch_council_boundary(council_name)
    if boundary is None:
        return centers

    # boundary is a list of (lat, lon) polygon vertices
    # Use simple point-in-polygon + buffer check
    kept = []
    for c in centers:
        if _point_near_polygon(c[1], c[2], boundary, buffer_km * 1000):
            kept.append(c)

    return kept if kept else centers


def _fetch_council_boundary(council_name: str) -> Optional[list]:
    """Fetch council boundary polygon from Nominatim (cached)."""
    os.makedirs(COUNCIL_CACHE_DIR, exist_ok=True)
    safe_name = council_name.lower().replace(" ", "_").replace("/", "_")[:60]
    cache_path = f"{COUNCIL_CACHE_DIR}/{safe_name}.json"

    if os.path.exists(cache_path):
        data = json.load(open(cache_path))
        return data if data else None

    try:
        params = urllib.parse.urlencode({
            "q": council_name + ", UK",
            "format": "json",
            "polygon_geojson": 1,
            "limit": 1,
        })
        url = f"https://nominatim.openstreetmap.org/search?{params}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "GeoMapAgent/1.0 (academic research)"
        })
        time.sleep(1.1)  # Nominatim rate limit
        resp = _retry_urlopen(req, timeout=15, label=f"Nominatim({council_name[:30]})")
        results = json.loads(resp.read())
        resp.close()

        if not results:
            with open(cache_path, "w") as f:
                json.dump(None, f)
            return None

        geojson = results[0].get("geojson", {})
        gtype = geojson.get("type", "")
        coords = geojson.get("coordinates", [])

        # Extract outer ring of polygon
        ring = None
        if gtype == "Polygon" and coords:
            ring = coords[0]
        elif gtype == "MultiPolygon" and coords:
            # Use largest polygon
            largest = max(coords, key=lambda p: len(p[0]) if p else 0)
            ring = largest[0] if largest else None

        if ring:
            # Convert [lon, lat] → (lat, lon)
            boundary = [(pt[1], pt[0]) for pt in ring]
            with open(cache_path, "w") as f:
                json.dump(boundary, f)
            return boundary

        with open(cache_path, "w") as f:
            json.dump(None, f)
        return None

    except Exception:
        return None


def _point_near_polygon(lat: float, lon: float, polygon: list,
                        buffer_m: float) -> bool:
    """Check if point is inside polygon or within buffer_m of its boundary.

    Uses ray casting for point-in-polygon, then min-distance for buffer.
    """
    # Ray casting
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        yi, xi = polygon[i]
        yj, xj = polygon[j]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i

    if inside:
        return True

    # Check buffer: min distance to any edge
    min_dist = float("inf")
    for i in range(n):
        yi, xi = polygon[i]
        dist = _distance_m(lat, lon, yi, xi)
        if dist < min_dist:
            min_dist = dist

    return min_dist <= buffer_m


# ── Scale and postcode extraction from LLM analysis ─────────────────────────

import re

POSTCODE_RE = re.compile(r"\b([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})\b", re.IGNORECASE)


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


def collect_postcodes(analysis, ve=None):
    """Collect all postcodes from analysis and visual_extract."""
    postcodes = []
    for src in [analysis, ve]:
        if src is None:
            continue
        for p in src.get("postcodes", []):
            if isinstance(p, str):
                postcodes.append(p)
            elif isinstance(p, dict) and "postcode" in p:
                postcodes.append(p["postcode"])
    postcodes += POSTCODE_RE.findall(json.dumps(analysis))
    return list(set(pc.strip().upper() for pc in postcodes))


def load_postcode_cache(cache_path=None):
    """Load postcodes.io cache from disk."""
    if cache_path is None:
        cache_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "cache", "postcodes_io.json",
        )
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return {}


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


def compute_all_centers(analysis, geocode, ve, postcode_results):
    """Compute candidate centers from all geocoding sources.

    Sources: OS grid ref, postcodes, Nominatim, visual extract grid ref,
    Photon geocoding, place names.

    Args:
        analysis: LLM analysis dict (has os_grid_ref, postcodes, etc.)
        geocode: Nominatim geocode result dict (has lat, lon)
        ve: Visual extract dict (has geocode_queries, road names, etc.)
        postcode_results: Dict from query_postcodes_io_bulk()

    Returns list of (name, lat, lon, sigma_m) tuples.
    """
    from tools.geo_tools import os_grid_ref_to_latlon

    centers = []
    seen = set()

    def add(name, lat, lon, sigma):
        key = (round(lat, 4), round(lon, 4))
        if key not in seen:
            seen.add(key)
            centers.append((name, lat, lon, sigma))

    # Grid reference
    grid_ref = analysis.get("os_grid_ref")
    if grid_ref:
        parsed = os_grid_ref_to_latlon(grid_ref)
        if parsed:
            gr_lat, gr_lon = parsed
            add("gridref_250", gr_lat, gr_lon, 250)
            add("gridref_500", gr_lat, gr_lon, 500)
            add("gridref_none", gr_lat, gr_lon, None)

    # Postcodes
    if postcode_results:
        for pc, data in postcode_results.items():
            if data and "lat" in data:
                add(f"postcode_{pc[:4]}_150", data["lat"], data["lon"], 150)
                add(f"postcode_{pc[:4]}_300", data["lat"], data["lon"], 300)

    # Nominatim
    nom_lat, nom_lon = geocode.get("lat"), geocode.get("lon")
    if nom_lat and nom_lon:
        add("nom_500", nom_lat, nom_lon, 500)
        add("nom_800", nom_lat, nom_lon, 800)
        add("nom_none", nom_lat, nom_lon, None)

    # Visual extract grid ref
    ve_grid_ref = analysis.get("grid_ref_from_text") or (ve.get("os_grid_ref") if ve else None)
    if ve_grid_ref and ve_grid_ref != grid_ref:
        parsed = os_grid_ref_to_latlon(ve_grid_ref)
        if parsed:
            vgr_lat, vgr_lon = parsed
            add("ve_gridref_250", vgr_lat, vgr_lon, 250)
            add("ve_gridref_500", vgr_lat, vgr_lon, 500)

    # Photon centers
    if ve:
        phot = photon_centers(ve)
        for c in phot:
            add(c[0], c[1], c[2], c[3])

        # Place name centers
        place = place_name_centers(ve)
        for c in place:
            add(c[0], c[1], c[2], c[3])

    return centers
