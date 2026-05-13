"""Dedicated Locate stage — OCR + PDF-text candidate generator.

Produces ranked candidate centres (+ optional direct affine) BEFORE positioning
runs. Splits the EXTRACT-style pipeline's implicit first stage from registration
so the agent can see the evidence and pick instead of letting five geocoding
sources fire blind inside ``position_boundary``.

Pipeline: OCR at 700 DPI (Tyagi & Dubey 2025) for graticule ticks + scale text,
then a PDF-text candidate cascade (postcodes, grid refs, district / admin /
parish lookups, place-name geocoding, road-intersection triangulation).
The VLM "read the map" pass was retired in 2026-05; ``use_vlm`` and
``model_name`` kwargs on :func:`locate_map` are legacy no-ops.

High-value short-circuit: if ≥3 distinct OS grid refs are OCR'd from the page
margins, we solve the page-pixel → OSGB affine in closed form.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field

from tools.geo.grid_ref import os_grid_ref_to_latlon, os_grid_ref_to_latlon_coarse
from tools.geocoders import (
    _distance_m,
    _is_valid_uk_coord,
    gpkg_place_search,
    nominatim_structured,
    wikidata_place_search,
)
from tools.pdf_tools import render_pdf_page


# ─── Constants ─────────────────────────────────────────────────────────────

# Cache dir for OCR results per case (keyed by pdf hash + page)
CACHE_DIR = Path("cache/locate")

# Hard UK bounds for sanity (pulled from geocoding.UK_BBOX — kept local to
# avoid a second import round-trip).
UK_LAT = (49.0, 61.0)
UK_LON = (-8.5, 2.0)


# ─── Schemas (moved to tools.locate.schemas; re-imported here) ─────────────

from tools.locate.schemas import (
    Candidate,
    LocateCandidate,
    DirectAffine,
    LocateResult,
    OCRWord,
)


# ─── OCR helpers (moved to tools.locate.ocr; re-imported here) ─────────────

from tools.locate.ocr import (
    OCR_DPI, OCR_MAX_MP, OCR_TIMEOUT_S,
    _GRID_REF_RE, _SCALE_RE, _COORD_NUM_RE,
    _safe_ocr_dpi, _run_tesseract, _neighbouring_words,
    extract_grid_refs_from_ocr, extract_scale_from_ocr,
)


# ─── Grid-tick affine solver (moved to tools.locate.affine) ────────────────

from tools.locate.affine import (
    _wgs84_to_osgb,
    _collinearity_score,
    solve_affine_from_grid_ticks,
    direct_affine_centroid,
)


# ─── Road intersection triangulation ──────────────────────────────────────

_ROAD_SUFFIXES = [
    (r"\bst\.?\b", "street"), (r"\brd\.?\b", "road"), (r"\bln\.?\b", "lane"),
    (r"\bave?\.?\b", "avenue"), (r"\bdr\.?\b", "drive"), (r"\bct\.?\b", "court"),
    (r"\bpl\.?\b", "place"), (r"\bcl\.?\b", "close"), (r"\bsq\.?\b", "square"),
    (r"\bter\.?\b", "terrace"), (r"\bgdns\.?\b", "gardens"),
    (r"\bcres\.?\b", "crescent"), (r"\bpk\.?\b", "park"),
]


def _norm_road(name: str) -> str:
    """Normalise a road name for fuzzy matching against OSM tags."""
    if not name:
        return ""
    n = name.lower().strip()
    n = re.sub(r"^the\s+", "", n)
    n = re.sub(r"^saint\s+", "st ", n)
    for pat, repl in _ROAD_SUFFIXES:
        n = re.sub(pat, repl, n)
    n = re.sub(r"\s+", " ", n)
    return n.strip()


def _road_matches(osm_name: str, input_roads_norm: Dict[str, str]) -> Optional[str]:
    """Return the ORIGINAL input road name if osm_name matches any of them.

    `input_roads_norm` is {original: normalised}. Matches on normalised
    equality or substring containment either way (to handle "High Street"
    vs "High Street Kensington" when OSM is more verbose).
    """
    osm_n = _norm_road(osm_name)
    if not osm_n:
        return None
    for orig, norm in input_roads_norm.items():
        if not norm:
            continue
        if norm == osm_n or norm in osm_n or osm_n in norm:
            return orig
    return None


# ── Road-intersection cache (osmnx graph reconstruction is the single
# largest CPU hotspot in propose_centers_v2 — ~17 s/case in v17 cProfile).
# ── osmnx already caches the raw Overpass JSON; this layer caches the
# already-PARSED NetworkX graph traversal result, which is pure
# deterministic given (roads, anchor_lat, anchor_lon, radius_km).
# ── In-process dict for the hot path; disk pickle for cross-run wins.
# ── Disable with GEOMAP_DISABLE_ROAD_CACHE=1 (env, default off).
import os as _cache_os
import pickle as _cache_pickle
_INTERSECTION_CACHE: Dict[Tuple, List[Tuple[float, float, List[str]]]] = {}
_INTERSECTION_DISK_DIR = (
    # File is at tools/locate/_core.py; repo root is 3 levels up (was 2
    # before the candidates.py → tools/locate/ split).
    Path(__file__).resolve().parent.parent.parent / "cache" / "road_intersections"
)


def _intersection_cache_key(
    roads: List[str], anchor_lat: float, anchor_lon: float, radius_km: float
) -> Tuple:
    """Deterministic key, insensitive to road-list order and minor float jitter."""
    return (
        tuple(sorted(r.strip().lower() for r in roads if r and r.strip())),
        round(float(anchor_lat), 4),  # ~11 m precision
        round(float(anchor_lon), 4),
        round(float(radius_km), 1),
    )


def _intersection_disk_path(key: Tuple) -> Path:
    h = hashlib.md5(repr(key).encode("utf-8")).hexdigest()
    return _INTERSECTION_DISK_DIR / f"{h}.pkl"


def find_road_intersections(
    roads: List[str],
    anchor_lat: float,
    anchor_lon: float,
    radius_km: float = 5.0,
    verbose: bool = False,
) -> List[Tuple[float, float, List[str]]]:
    """Pull OSM road network near anchor, find nodes where ≥2 named roads meet.

    Returns (lat, lon, [road_names_meeting_here]) for each intersection,
    sorted by (count of roads meeting desc, distance from anchor asc).

    Needs ≥2 input roads; otherwise returns []. Uses osmnx's built-in
    Overpass cache so repeat queries within a bbox are free.

    PERFORMANCE: this function used to rebuild the NetworkX MultiDiGraph
    on every call (~17 s on a 5 km radius). Now wrapped in an in-process
    dict cache + disk pickle keyed by deterministic args. Set
    GEOMAP_DISABLE_ROAD_CACHE=1 to bypass.
    """
    if not roads or len(roads) < 2:
        return []
    input_roads_norm = {r: _norm_road(r) for r in roads if r.strip()}
    if len(input_roads_norm) < 2:
        return []

    # Cache layer (skip if explicitly disabled)
    cache_enabled = _cache_os.environ.get("GEOMAP_DISABLE_ROAD_CACHE") != "1"
    key = _intersection_cache_key(roads, anchor_lat, anchor_lon, radius_km) \
        if cache_enabled else None

    if cache_enabled:
        hit = _INTERSECTION_CACHE.get(key)
        if hit is not None:
            if verbose:
                print(f"  road_intersections cache HIT (mem) for "
                      f"{len(roads)} roads @ ({anchor_lat:.4f},{anchor_lon:.4f})")
            return hit
        # Try disk
        disk_path = _intersection_disk_path(key)
        if disk_path.exists():
            try:
                with open(disk_path, "rb") as fh:
                    result = _cache_pickle.load(fh)
                _INTERSECTION_CACHE[key] = result
                if verbose:
                    print(f"  road_intersections cache HIT (disk) → "
                          f"{len(result)} intersections")
                return result
            except Exception:
                pass  # corrupt cache; fall through to recompute

    # Cache miss — original computation
    result = _find_road_intersections_uncached(
        roads, anchor_lat, anchor_lon, radius_km, verbose, input_roads_norm
    )

    if cache_enabled:
        _INTERSECTION_CACHE[key] = result
        try:
            _INTERSECTION_DISK_DIR.mkdir(parents=True, exist_ok=True)
            with open(_intersection_disk_path(key), "wb") as fh:
                _cache_pickle.dump(result, fh)
        except Exception:
            pass  # best-effort; in-mem cache still works

    return result


def _find_road_intersections_uncached(
    roads: List[str],
    anchor_lat: float,
    anchor_lon: float,
    radius_km: float,
    verbose: bool,
    input_roads_norm: Dict[str, str],
) -> List[Tuple[float, float, List[str]]]:
    """Original osmnx graph traversal — only called on cache miss."""
    try:
        import osmnx as ox
    except ImportError:
        print("  WARN:osmnx not available, skipping triangulation")
        return []

    # Use a broad road network query so we capture footpaths / cycleways too
    # (planning-relevant roads are sometimes tagged as bridleways etc.).
    try:
        G = ox.graph_from_point(
            (anchor_lat, anchor_lon),
            dist=int(radius_km * 1000),
            network_type="all",
            simplify=True,
            truncate_by_edge=True,
        )
    except Exception as e:
        if verbose:
            print(f"  WARN:osmnx graph fetch failed: {e}")
        return []

    # Collect: for each node, which of our input roads touch it?
    node_roads: Dict[Any, set] = {}
    for u, v, data in G.edges(data=True):
        name = data.get("name")
        if isinstance(name, list):
            # Edge tagged with multiple names — check all
            candidates = name
        elif isinstance(name, str):
            candidates = [name]
        else:
            continue
        for osm_name in candidates:
            matched = _road_matches(osm_name, input_roads_norm)
            if matched:
                node_roads.setdefault(u, set()).add(matched)
                node_roads.setdefault(v, set()).add(matched)

    # Intersections = nodes with ≥2 distinct matched input roads
    intersections: List[Tuple[float, float, List[str]]] = []
    for node, matched_set in node_roads.items():
        if len(matched_set) < 2:
            continue
        nd = G.nodes[node]
        lat, lon = nd.get("y"), nd.get("x")
        if lat is None or lon is None:
            continue
        intersections.append((float(lat), float(lon), sorted(matched_set)))

    # Sort by (−count, distance to anchor)
    intersections.sort(
        key=lambda t: (-len(t[2]),
                       _distance_m(t[0], t[1], anchor_lat, anchor_lon))
    )
    return intersections


def _triangulation_candidates(
    roads: List[str],
    anchor: Optional[Tuple[float, float]],
    city_ctx: Optional[str] = None,
    city_ctx_alts: Optional[List[str]] = None,
    extra_anchors: Optional[List[Tuple[float, float]]] = None,
    anchor_trusted: bool = True,
    verbose: bool = False,
) -> List[LocateCandidate]:
    """Wrap find_road_intersections into LocateCandidates.

    Tries the primary anchor first, then any extra_anchors provided. If
    the primary anchor is missing, derives one by Nominatim-geocoding
    each road in turn — this catches cases where the primary-road
    resolution went to a UK homonym and the actual intersection is near
    a different road's location. Dedup intersections across anchors.
    """
    if not roads or len(roads) < 2:
        return []

    # Build the list of anchors to try
    anchors: List[Tuple[float, float]] = []
    seen_anchor: set = set()
    def _push(pt):
        if pt is None:
            return
        key = (round(pt[0], 3), round(pt[1], 3))
        if key in seen_anchor:
            return
        seen_anchor.add(key)
        anchors.append(pt)

    _push(anchor)
    for a in extra_anchors or []:
        _push(a)
    # Fallback: derive anchors by geocoding individual roads. Critical to
    # distinguish trusted vs untrusted primary anchor:
    #   * trusted (postcode / district / grid-ref): road-derived anchors
    #     can drag us to the wrong UK region (common road names like
    #     "Church Lane" match anywhere) — skip them, or only add ones
    #     within 20km of the primary.
    #   * untrusted (road-derived primary from _pick_anchor step 6):
    #     the primary might be in the wrong region, so try MULTIPLE
    #     road-derived anchors; the correct intersection wins by count.
    # Build the cascade list once. Primary city_ctx first, then any alts
    # the caller supplied (typically all viable contexts from pdf_info).
    # Empty-city is only reached as the final fallback inside
    # _geocode_road_cascade.
    primary_filtered = (
        [city_ctx] if (city_ctx and not _looks_like_road(city_ctx)) else []
    )
    cascade = primary_filtered + [a for a in (city_ctx_alts or [])
                                   if a not in primary_filtered]

    if not anchor_trusted or anchor is None:
        # Untrusted primary or no primary: try each road as a seed.
        for road in roads[:5]:
            hit = _geocode_road_cascade(road, cascade)
            if hit:
                _push((hit["lat"], hit["lon"]))
    elif len(anchors) < 2 and city_ctx and not _looks_like_road(city_ctx):
        # Trusted primary: add one more road-derived anchor ONLY if it's
        # close to the primary (strong cross-validation).
        for road in roads[:3]:
            hit = nominatim_structured(
                street=road, city=city_ctx, country="UK")
            if hit and _distance_m(hit["lat"], hit["lon"],
                                    anchor[0], anchor[1]) < 20_000:
                _push((hit["lat"], hit["lon"]))
                break

    if not anchors:
        return []

    # Run triangulation from each anchor, collect + dedup intersections
    all_hits: Dict[Tuple[int, int], Tuple[float, float, List[str]]] = {}
    for a_lat, a_lon in anchors[:4]:  # cap to avoid blowing out Overpass
        hits = find_road_intersections(
            roads, a_lat, a_lon, radius_km=5.0, verbose=verbose)
        for lat, lon, meeting in hits:
            key = (int(round(lat * 1e4)), int(round(lon * 1e4)))
            existing = all_hits.get(key)
            # Keep the one with most roads meeting (if tie, first wins)
            if existing is None or len(meeting) > len(existing[2]):
                all_hits[key] = (lat, lon, meeting)
    if not all_hits:
        return []

    # Sort: most-roads-first, then by distance from primary anchor
    primary = anchors[0]
    sorted_hits = sorted(
        all_hits.values(),
        key=lambda t: (-len(t[2]), _distance_m(t[0], t[1], primary[0], primary[1]))
    )

    out: List[LocateCandidate] = []
    for lat, lon, meeting in sorted_hits[:3]:
        conf = 0.92 if len(meeting) >= 3 else 0.88
        out.append(LocateCandidate(
            lat=lat, lon=lon, confidence=conf,
            source=f"road_intersection:{'+'.join(meeting[:3])}",
            evidence=f"OSM intersection of {len(meeting)} roads: {', '.join(meeting)}",
            specificity=0,
        ))
    return out


# ─── District boundary lookup (cached) ─────────────────────────────────────

# Process-local memoisation for OSM district lookups (osmnx itself has a
# disk cache but it deserialises on every call; this saves the hit).
_DISTRICT_LOOKUP_MEMO: Dict[str, Optional[Tuple[float, float, Dict[str, float]]]] = {}


def _clean_district_query(name: str) -> List[str]:
    """Generate plausible query variants for a district_name field.

    Reader often produces names like "St Albans City & District Council, UK"
    or "Dover District Council, UK" — strip the "Council" suffix and try
    a few variants since Nominatim matches these inconsistently.
    """
    if not name:
        return []
    s = name.strip()
    variants: List[str] = []

    def _add(v: str) -> None:
        v = re.sub(r"\s+", " ", v).strip(" ,")
        if v and v not in variants:
            variants.append(v)

    _add(s)
    # Strip common council suffixes
    no_council = re.sub(r"\b(?:City\s+)?(?:&|and)\s+District\s+Council\b",
                         "District", s, flags=re.IGNORECASE)
    no_council = re.sub(r"\b(?:District|Borough|County|City|Parish)\s+Council\b",
                         "", no_council, flags=re.IGNORECASE)
    _add(no_council)
    # Also try the first comma-segment only (usually the bare name)
    head = s.split(",")[0]
    _add(head)
    head_no_council = re.sub(r"\b(?:City\s+)?(?:&|and)\s+District\s+Council\b",
                              "District", head, flags=re.IGNORECASE)
    head_no_council = re.sub(r"\b(?:District|Borough|County|City|Parish)\s+Council\b",
                              "", head_no_council, flags=re.IGNORECASE)
    _add(f"{head_no_council}, UK")
    _add(f"{head_no_council} District, UK")
    return variants


def _district_info(district_name: str
                    ) -> Optional[Tuple[float, float, Dict[str, float]]]:
    """Return (centroid_lat, centroid_lon, bbox) for a district name, or None.

    ``district_name`` may be pipe-separated alternates like
    ``"Dover District, Kent, UK | Dover, Kent, UK"`` — we try each in
    order, and also try cleaned-up variants (strip "Council" suffixes,
    normalise "City & District Council") since Nominatim matches these
    inconsistently.
    """
    if not district_name:
        return None
    key = district_name.strip()
    if key in _DISTRICT_LOOKUP_MEMO:
        return _DISTRICT_LOOKUP_MEMO[key]

    from tools.geo.grid_ref import lookup_district_boundary

    # Build the full set of queries across pipe alternates + cleaned variants
    queries: List[str] = []
    for alt in key.split("|")[:2]:
        for q in _clean_district_query(alt):
            if q not in queries:
                queries.append(q)

    best_large: Optional[Tuple[float, float, Dict[str, float]]] = None
    best_small: Optional[Tuple[float, float, Dict[str, float]]] = None
    best_extent_km = 0.0
    for alt in queries[:6]:
        try:
            result = lookup_district_boundary(alt)
        except Exception:
            continue
        if not result.get("success"):
            continue
        bbox = result.get("bbox") or {}
        try:
            lat = (bbox["min_lat"] + bbox["max_lat"]) / 2
            lon = (bbox["min_lon"] + bbox["max_lon"]) / 2
            lat_span_km = (bbox["max_lat"] - bbox["min_lat"]) * 111.0
            lon_span_km = (bbox["max_lon"] - bbox["min_lon"]) * 111.0 * \
                          math.cos(math.radians(lat))
            extent_km = max(lat_span_km, lon_span_km)
        except (KeyError, TypeError):
            continue
        if not _is_valid_uk_coord(lat, lon):
            continue
        info = (lat, lon, bbox)
        if extent_km >= 2.0:
            # Real admin boundary — pick the largest one we've seen
            if extent_km > best_extent_km:
                best_large, best_extent_km = info, extent_km
        else:
            # Small bbox = POI match. Keep as fallback only if we haven't
            # already found a large-bbox match that disagrees with it.
            if best_small is None:
                best_small = info

    if best_large is not None:
        _DISTRICT_LOOKUP_MEMO[key] = best_large
        return best_large
    # Fall back to small-bbox result if nothing large is available.
    _DISTRICT_LOOKUP_MEMO[key] = best_small
    return best_small


def _admin_info(admin_type: str, admin_name: str
                 ) -> Optional[Tuple[float, float, Dict[str, float]]]:
    """Resolve a parsed (admin_type, admin_name) pair to an OSM boundary.

    Normalises "rural district" → "district" because Nominatim matches
    some historical "Rural District" names to wrong locations (e.g.
    "Dover Rural District, UK" → a place in Hertfordshire, 156km off).
    Tries a few reasonable query variants in order.
    """
    if not admin_name:
        return None
    norm_type = "district" if "rural" in admin_type.lower() else admin_type
    for query in (f"{admin_name} {norm_type.title()}, UK",
                  f"{admin_name}, {norm_type.title()}, UK",
                  f"{admin_name}, UK"):
        info = _district_info(query)
        if info is not None:
            return info
    return None


def _district_candidate(pdf_info: Dict[str, Any]) -> Optional[LocateCandidate]:
    """LocateCandidate for the district boundary centroid, when is_district_wide.

    Confidence 0.8 — reliable but wide-area (can be km from specific sites).
    For fully district-wide boundaries this often lands inside the GT.
    """
    if not pdf_info.get("is_district_wide"):
        return None
    dn = pdf_info.get("district_name")
    if not dn:
        return None
    info = _district_info(dn)
    if info is None:
        return None
    lat, lon, _ = info
    return LocateCandidate(
        lat=lat, lon=lon, confidence=0.8,
        source=f"district:{dn.split('|')[0].strip()[:40]}",
        evidence=f"OSM admin boundary centroid for district '{dn[:60]}'",
        specificity=2,
    )


# ─── Parsing PDF text (site_address, place_names) ──────────────────────────

# Road-type suffix list kept in one place so it also matches the directional
# reference below.
_ROAD_SUFFIX = (
    r"(?:Lane|Road|Street|Avenue|Way|Close|Drive|Court|Place|Square|"
    r"Gardens?|Terrace|Crescent|Park|Mews|Grove|Hill|Rise|Walk|Row|End|"
    r"Green|Common|Fields?|Gate|Path|Wharf|Quay|Pier|Embankment|Parade|"
    r"Approach|Bank|Bridge|Circus)"
)

# Locates the road name first; numbers are then parsed from the ~120 chars
# immediately preceding it. Handles list + range forms like
# "126, 128, 130 Norwich Road" and "26-64 Manor Road" via a median reduction.
_ROAD_NAME_RE = re.compile(
    r"\b([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){0,3}\s+" + _ROAD_SUFFIX + r")"
)
_NUMBER_OR_RANGE_RE = re.compile(r"(\d+)\s*[-–]\s*(\d+)|(\d+)")

# "north of X" / "south-east of X" / "NW of X"
_DIRECTIONAL_RE = re.compile(
    r"\b(north(?:[\- ]east|[\- ]west)?|south(?:[\- ]east|[\- ]west)?|east|west|"
    r"N\.?E\.?|N\.?W\.?|S\.?E\.?|S\.?W\.?)\s+of\s+"
    r"(?:the\s+)?"
    r"(?:village\s+of\s+|hamlet\s+of\s+|parish\s+of\s+)?"
    r"([^,.]+?)"
    r"(?:,|\.|$|\s+in\s+(?:the\s+)?(?:parish|district|borough|county))",
    re.IGNORECASE,
)

# "Land adjoining X" / "Land at X" / "Site off X" / "Field abutting X"
_LAND_REF_RE = re.compile(
    r"\b(?:land|site|field|area|plot|parcel)\s+"
    r"(?:at|adjoining|adjacent\s+to|off|near|fronting|abutting|bordering|behind)\s+"
    r"(?:the\s+)?"
    r"([^,.]+?)"
    r"(?:,|\.|$|\s+in\s+(?:the\s+)?(?:parish|district|borough|county))",
    re.IGNORECASE,
)

# "in the Parish of X" / "Parish of X" / "parish of Caistor St. Edmund"
# Note: we deliberately allow '.' inside the name so "St. Edmund" survives.
# The `(?-i:[a-z])` disables IGNORECASE for the sentence-end check so
# "St. Edmund" doesn't get clipped to "St" (since IGNORECASE would make
# `[a-z]` match 'E' too).
# Capture must start with a genuinely uppercase letter — the `(?-i:[A-Z])`
# disables IGNORECASE locally so "parish of Colney" (lowercase p) doesn't
# accidentally anchor a region capture.
_PARISH_RE = re.compile(
    r"\b(?:in\s+the\s+)?parish(?:es)?\s+of\s+"
    r"((?-i:[A-Z])[\w'\- .]+?)"
    r"(?:,|\n|$|\s+and\s+|\s+in\s+(?:the\s+)?(?:district|borough|county|rural)"
    r"|\.\s+(?-i:[a-z]))",
    re.IGNORECASE,
)

_ADMIN_OF_RE = re.compile(
    r"\b(?:in\s+the\s+)?(district|borough|county|rural\s+district)\s+of\s+"
    r"((?-i:[A-Z])[\w'\- .]+?)"
    r"(?:,|\n|$|\s+and\s+|\s+in\s+(?:the\s+)?(?:district|borough|county)"
    r"|\.\s+(?-i:[a-z]))",
    re.IGNORECASE,
)

_REGION_CONTEXT_RE = re.compile(
    r"\b(?:various\s+sites\s+across|various\s+sites\s+in|"
    r"sites\s+across|sites\s+in|land\s+within|land\s+in\s+the)\s+"
    r"(?:the\s+)?"
    r"((?-i:[A-Z])[\w'\- ]+?)"
    r"(?:,|\(|\n|$|\s+(?:parish|district|borough|council|conservation\s+area)"
    r"|\.\s+(?-i:[a-z]))",
    re.IGNORECASE,
)

# Direction string → unit vector (dlat_sign, dlon_sign)
_COMPASS: Dict[str, Tuple[float, float]] = {
    "n": (1, 0), "north": (1, 0),
    "s": (-1, 0), "south": (-1, 0),
    "e": (0, 1), "east": (0, 1),
    "w": (0, -1), "west": (0, -1),
    "ne": (0.707, 0.707), "northeast": (0.707, 0.707),
    "nw": (0.707, -0.707), "northwest": (0.707, -0.707),
    "se": (-0.707, 0.707), "southeast": (-0.707, 0.707),
    "sw": (-0.707, -0.707), "southwest": (-0.707, -0.707),
}

# Admin-entity tokens — filtered out of place_name geocoding. These are
# matched as WHOLE WORDS (word-boundary regex) so substrings don't trip
# false positives. Critical: 'borough' must not match inside 'Peterborough'.
_ADMIN_TOKEN_RE = re.compile(
    r"\b(?:council|government|authority|central\s+activities\s+zone)\b"
    r"|\b(?:district|borough|county|parish)\s+council\b"
    r"|\blondon\s+borough\s+of\b",
    re.IGNORECASE,
)


def _is_admin_entity(name: str) -> bool:
    """True if the place_name looks like an administrative entity (council,
    government body, etc.) rather than a specific place we can geocode."""
    return bool(_ADMIN_TOKEN_RE.search(name or ""))

# Named-landmark keywords. A place_name containing one of these is a
# specific building / monument / named feature, not a broad area. Boost
# its geocoded-candidate confidence and specificity so it outranks the
# parish/village centroids that cover the same area more loosely.
_LANDMARK_KEYWORDS = (
    "hall", "church", "farm", "house", "manor", "abbey", "castle",
    "priory", "park", "lodge", "cottage", "chapel", "cross", "gate",
    "school", "college", "hospital", "mill", "bridge", "green",
    "theatre", "museum", "station", "centre", "center", "cathedral",
    "square", "market",
)


def _is_landmark_name(name: str) -> bool:
    """Heuristic — does this place_name refer to a specific named landmark?"""
    if not name:
        return False
    tokens = name.lower().split()
    return any(t.strip(",.'") in _LANDMARK_KEYWORDS for t in tokens)


def _parse_house_number(site_address: str) -> Optional[Tuple[str, str]]:
    """Extract (representative_number, road_name) from a site_address.

    Handles:
      * single:  "98 Pipers Lane", "at 41 Linden Grove", "no. 41 Linden Grove"
      * list:    "126, 128, 130, 132 and 134 Norwich Road" → median "130"
      * range:   "26-64 Manor Road" → midpoint "45"
      * mixed:   "4, 8-50, 54-92 Chelsea Park Gardens" → median over midpoints

    Finds the road name first, then collects number tokens in the 120 chars
    before it. A median (over a list) or midpoint (single range) is a
    reasonable representative point for geocoding purposes — MINIMA will
    close the last few metres once the anchor is on the right street.

    Returns None if no valid street with a preceding number is present.
    """
    if not site_address:
        return None

    road_m = _ROAD_NAME_RE.search(site_address)
    if not road_m:
        return None
    road = road_m.group(1).strip()

    segment = site_address[max(0, road_m.start() - 120):road_m.start()]
    numbers: List[int] = []
    for m in _NUMBER_OR_RANGE_RE.finditer(segment):
        if m.group(1) and m.group(2):
            lo, hi = int(m.group(1)), int(m.group(2))
            if 0 < lo < hi and hi < 1000:
                numbers.append((lo + hi) // 2)
        elif m.group(3):
            n = int(m.group(3))
            if 0 < n < 1000:
                numbers.append(n)

    if not numbers:
        return None

    numbers.sort()
    rep = numbers[len(numbers) // 2]
    return (str(rep), road)


def _parse_directional(site_address: str) -> Optional[Tuple[str, str]]:
    """Extract (compass_direction, reference_text) from a site_address.

    Catches "north of X", "south-west of Y", "NE of Z". The reference_text
    is what follows, bounded by a comma, period, or an "in the parish/..."
    continuation. Returns None if the site_address doesn't contain a
    directional locator.
    """
    if not site_address:
        return None
    m = _DIRECTIONAL_RE.search(site_address)
    if not m:
        return None
    direction = m.group(1).lower().rstrip(".").replace(" ", "").replace("-", "")
    reference = m.group(2).strip(" ,.")
    if len(reference) < 3:
        return None
    return (direction, reference)


def _parse_land_reference(site_address: str) -> Optional[str]:
    """Extract the geographic reference out of "Land at X", "Site adjoining Y", etc.

    Returns X/Y as free text if matched, else None.
    """
    if not site_address:
        return None
    m = _LAND_REF_RE.search(site_address)
    if not m:
        return None
    ref = m.group(1).strip(" ,.")
    return ref if len(ref) >= 3 else None


def _parse_parish(text: str) -> Optional[str]:
    """Extract parish name from "in the Parish of X" / "parish of X"."""
    if not text:
        return None
    m = _PARISH_RE.search(text)
    if not m:
        return None
    parish = m.group(1).strip(" ,.")
    return parish if len(parish) >= 3 else None


def _parse_admin_of(text: str) -> Optional[Tuple[str, str]]:
    """Extract (admin_type, admin_name) from 'in the District of X' etc.

    admin_type is the lowercase keyword (district/borough/county/rural district).
    """
    if not text:
        return None
    m = _ADMIN_OF_RE.search(text)
    if not m:
        return None
    admin_type = m.group(1).lower().strip()
    admin_name = m.group(2).strip(" ,.")
    return (admin_type, admin_name) if len(admin_name) >= 3 else None


def _parse_region_context(text: str) -> Optional[str]:
    """Extract X from 'various sites across X' / 'Land within X' / similar.

    Used when the reader didn't flag is_district_wide but the text clearly
    indicates a broader region anchoring the site. Returns X or None.
    """
    if not text:
        return None
    m = _REGION_CONTEXT_RE.search(text)
    if not m:
        return None
    region = m.group(1).strip(" ,.")
    # Skip obvious non-places
    if region.lower() in {"the district", "the borough", "the county", "the parish"}:
        return None
    return region if len(region) >= 3 else None


def _offset_latlon(lat: float, lon: float, direction: str,
                    distance_m: float) -> Tuple[float, float]:
    """Offset a lat/lon point by distance_m metres in the given compass direction."""
    key = direction.lower().replace(" ", "").replace("-", "").replace(".", "")
    d = _COMPASS.get(key)
    if d is None:
        return (lat, lon)
    dlat_sign, dlon_sign = d
    m_per_deg_lat = 111_111.0
    m_per_deg_lon = 111_111.0 * math.cos(math.radians(lat))
    return (lat + dlat_sign * distance_m / m_per_deg_lat,
            lon + dlon_sign * distance_m / m_per_deg_lon)


def _pdf_city_context(pdf_info: Dict[str, Any]) -> Optional[str]:
    """Best-effort city/town context for Nominatim structured queries.

    Precedence (best first):
      0. LLM-extracted ``likely_town_or_city`` (v2 reader)
      1. LLM-extracted ``admin_region`` (v2 reader)
      2. parsed district/borough/county from site_address (regex fallback)
      3. parsed region_context ("various sites across X", regex fallback)
      4. parsed parish (regex fallback)
      5. district_name (iff is_district_wide)
      6. first clean place_name (non-admin, non-landmark)
      7. leading token of landmark-style place_name ("Dover" from "Dover Castle")
      8. district_name (fallback for non-district cases)
      9. site_address comma-split tail
    """
    # 0/1. LLM-populated fields (v2 reader) — most reliable when present
    town = pdf_info.get("likely_town_or_city")
    if town and isinstance(town, str) and len(town) >= 3 \
            and not _is_admin_entity(town) and not _looks_like_road(town):
        return town
    admin_r = pdf_info.get("admin_region")
    if admin_r and isinstance(admin_r, str) and len(admin_r) >= 3 \
            and not _is_admin_entity(admin_r) and not _looks_like_road(admin_r):
        return admin_r

    sa = pdf_info.get("site_address") or ""
    notes = pdf_info.get("notes") or ""

    admin = _parse_admin_of(sa) or _parse_admin_of(notes)
    if admin:
        return admin[1]

    region = _parse_region_context(sa) or _parse_region_context(notes)
    if region:
        return region

    parish = _parse_parish(sa) or _parse_parish(notes)
    if parish:
        return parish

    dn = pdf_info.get("district_name")
    if dn and pdf_info.get("is_district_wide"):
        head = dn.split("|")[0].split(",")[0].strip()
        if head:
            return head

    # Two-pass place_names: first prefer non-landmark names (cities/towns),
    # then accept landmark-containing names as fallback. This prevents
    # specific site names like "Brick Knoll Park" (landmark keyword 'park')
    # from being used as city context when "St Albans" is also in the list.
    places = pdf_info.get("place_names") or []
    def _valid_ctx(p: str) -> bool:
        return bool(p) and not _is_admin_entity(p) and not _looks_like_road(p)
    for place in places:
        if _valid_ctx(place) and not _is_landmark_name(place):
            return place
    # For landmark-named places, try to extract the leading city/town token —
    # "Leicester Frith Farm" → "Leicester", "Dover Castle" → "Dover".
    # Verify against Zoomstack to reject false positives.
    for place in places:
        if not _valid_ctx(place) or not _is_landmark_name(place):
            continue
        tokens = place.split()
        # Drop trailing landmark tokens
        while tokens and tokens[-1].lower().strip(",.'") in _LANDMARK_KEYWORDS:
            tokens.pop()
        # Try the first 1-2 non-landmark tokens as a city candidate
        for n in (1, 2):
            if n > len(tokens):
                break
            candidate = " ".join(tokens[:n])
            if len(candidate) < 3 or _is_landmark_name(candidate):
                continue
            gp = gpkg_place_search(candidate, limit=1)
            if gp and gp[0].get("specificity", 9) <= 2:
                return candidate
    # Final fallback: any landmark-containing place_name (used as-is)
    for place in places:
        if _valid_ctx(place):
            return place

    if dn:
        head = dn.split("|")[0].split(",")[0].strip()
        if head:
            return head

    # site_address comma-split: prefer EARLIER tokens (closer to the site)
    # over later ones, because later tokens are usually county/country.
    # For "Land at Middle Lane, Farnham, Surrey" → Farnham beats Surrey.
    # The first token is typically the street phrase; skip it.
    sa = pdf_info.get("site_address") or ""
    parts = [p.strip() for p in sa.split(",")]
    for part in parts[1:]:  # skip first part (street/house phrase)
        if part.lower() in {"uk", "england", "great britain", "scotland"}:
            continue
        if _looks_like_road(part):
            continue
        if len(part) < 3 or not any(ch.isalpha() for ch in part):
            continue
        # Skip obvious county-only tokens (broad — prefer a town instead)
        # Only apply this on multi-part addresses where a non-county
        # candidate will still be found later.
        return part
    # No usable comma-part — fall back to reversed order as last resort
    for part in reversed(parts):
        if part.lower() in {"uk", "england", "great britain", "scotland"}:
            continue
        if _looks_like_road(part):
            continue
        if len(part) >= 3 and any(ch.isalpha() for ch in part):
            return part
    return None


def _pdf_city_alts(pdf_info: Dict[str, Any]) -> List[str]:
    """Return ALL viable city contexts from pdf_info, ordered by preference.

    Used for cascading road geocodes: try each context until one yields a
    Nominatim hit, only falling back to empty-city as a last resort.
    Mirrors _pdf_city_context's priority (which returns just the best one)
    but keeps the runners-up so the caller can retry on miss instead of
    going straight to a fully unrestricted UK-wide road lookup.
    """
    alts: List[str] = []
    seen: set = set()

    def _add(s):
        if not s or not isinstance(s, str):
            return
        s = s.strip()
        if not s or s in seen:
            return
        if _looks_like_road(s):
            return
        seen.add(s)
        alts.append(s)

    _add(pdf_info.get("likely_town_or_city"))
    _add(pdf_info.get("admin_region"))
    for parish in pdf_info.get("parish_names") or []:
        _add(parish)
    for place in pdf_info.get("place_names") or []:
        _add(place)
    return alts


def _geocode_road_cascade(road: str, city_alts: List[str]
                           ) -> Optional[Dict[str, Any]]:
    """Try Nominatim with each city context in order, fall back to empty.

    Returns the first hit. ``city_alts`` should already be filtered (see
    _pdf_city_alts). The empty-city fallback is the final attempt — it
    can return any UK road of that name, so we only reach it when every
    contextualised query has missed.
    """
    for ctx in city_alts:
        hit = nominatim_structured(street=road, city=ctx, country="UK")
        if hit:
            return hit
    return nominatim_structured(street=road, city="", country="UK")


def _viewbox_for_pdf(pdf_info: Dict[str, Any]
                      ) -> Optional[Tuple[float, float, float, float]]:
    """Return a Nominatim viewbox (min_lon, max_lat, max_lon, min_lat)
    to constrain queries, or None.

    Used to prevent wrong-homonym resolutions — e.g. "270 Eastfield Road"
    matching the wrong Peterborough when we know the site is in a
    specific district. Prefers a parsed admin unit's bbox, falls back to
    district_name's bbox, else None.
    """
    # Parsed admin (most specific)
    sa = pdf_info.get("site_address") or ""
    notes = pdf_info.get("notes") or ""
    for text in (sa, notes):
        admin = _parse_admin_of(text)
        if admin is None:
            continue
        info = _admin_info(admin[0], admin[1])
        if info is not None:
            _, _, bbox = info
            try:
                return (bbox["min_lon"], bbox["max_lat"],
                        bbox["max_lon"], bbox["min_lat"])
            except (KeyError, TypeError):
                pass

    # district_name (reader's structured output)
    dn = pdf_info.get("district_name")
    if dn:
        info = _district_info(dn)
        if info is not None:
            _, _, bbox = info
            try:
                return (bbox["min_lon"], bbox["max_lat"],
                        bbox["max_lon"], bbox["min_lat"])
            except (KeyError, TypeError):
                pass
    return None


def _looks_like_road(name: str) -> bool:
    """Heuristic: does this string look like a road name (not a city)?

    Used so a city_ctx that's actually a road (e.g. the reader extracted
    'Nunhead Grove' into place_names) doesn't sabotage Nominatim queries.
    """
    if not name:
        return False
    return bool(re.search(
        r"\b(?:Lane|Road|Street|Avenue|Way|Close|Drive|Court|Place|Square|"
        r"Gardens?|Terrace|Crescent|Mews|Grove|Hill|Rise|Walk|Row)\b",
        name, flags=re.IGNORECASE,
    ))


def _geocode_reference(
    reference: str, city_ctx: Optional[str],
    viewbox: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[Tuple[float, float, str]]:
    """Geocode a free-text reference via Nominatim / gazetteer / Photon.

    Returns (lat, lon, source_tag) or None. Tries in order:
      1. Nominatim street + city (if city_ctx looks like a city)
      2. Nominatim street + viewbox (bounded)
      3. Nominatim street without city
      4. Zoomstack place search
      5. Wikidata place search
      6. Photon free-text (last-ditch OSM search, no auth)
    """
    from tools.geocoders import query_photon
    if not reference:
        return None
    # 1. Road via Nominatim with city (skip if city_ctx is itself a road)
    if city_ctx and not _looks_like_road(city_ctx):
        hit = nominatim_structured(
            street=reference, city=city_ctx, country="UK", viewbox=viewbox)
        if hit:
            return (hit["lat"], hit["lon"], f"nominatim:street:{reference[:30]}")
    # 2. Nominatim bounded to viewbox without city
    if viewbox is not None:
        hit = nominatim_structured(
            street=reference, city="", country="UK",
            viewbox=viewbox, bounded=True)
        if hit:
            return (hit["lat"], hit["lon"], f"nominatim:bounded:{reference[:30]}")
    # 3. Nominatim without city or viewbox
    hit = nominatim_structured(street=reference, city="", country="UK")
    if hit:
        return (hit["lat"], hit["lon"], f"nominatim:street_no_city:{reference[:30]}")
    # 4. Zoomstack place — derive parent_anchor from viewbox center so the
    # gazetteer filters out wrong-region homonyms. Without this, a query
    # like "Princes Golf Club" returns the famous one in St Andrews
    # (Scotland) even when viewbox said Kent — Nominatim respects viewbox
    # but gpkg/wikidata don't unless given explicit coords.
    gp_kw: Dict[str, Any] = {}
    if viewbox is not None:
        gp_kw = {
            "parent_lat": (viewbox[1] + viewbox[3]) / 2,
            "parent_lon": (viewbox[0] + viewbox[2]) / 2,
        }
    gp = gpkg_place_search(reference, limit=1, **gp_kw)
    if gp:
        r = gp[0]
        return (r["lat"], r["lon"], f"gpkg:{r['type']}:{r['name']}")
    # 5. Wikidata — same parent_anchor for the same reason.
    wd = wikidata_place_search(reference, limit=1, **gp_kw)
    if wd:
        r = wd[0]
        return (r["lat"], r["lon"], f"wikidata:{r['qid']}")
    # 6. Photon free-text (UK qualifier)
    try:
        photon = query_photon(f"{reference}, UK", limit=1)
    except Exception:
        photon = []
    if photon:
        r = photon[0]
        return (r["lat"], r["lon"], f"photon:{reference[:30]}")
    return None


# Postcode helpers live in tools.locate.postcode; re-imported for callers
# that still grab them from this module.
from tools.locate.postcode import (
    _load_postcode_cache,
    _save_postcode_cache,
    _lookup_postcode,
)


def _pdf_text_candidates(pdf_info: Dict[str, Any]) -> List[LocateCandidate]:
    """Build candidates from the fields the reader already extracted from PDF text.

    Mines all the structured reader output to produce location candidates,
    without a fresh LLM call:

      * postcodes → postcodes.io centroid
      * grid_refs → OS grid resolution
      * site_address:
          - house-numbered addresses ("no. 41 Linden Grove") → Nominatim
            housenumber query (~50m accuracy)
          - directional modifiers ("north of 98 Pipers Lane") → geocode
            reference + offset 150m in the stated direction
          - land references ("Land adjoining Old Bottom Free Down") →
            geocode the reference as a place
      * place_names → Zoomstack / Wikidata landmark geocoding

    Admin entities (boroughs, councils, districts) are filtered from
    place_names because they're already handled by the district lookup path.
    """
    out: List[LocateCandidate] = []
    site_address = pdf_info.get("site_address") or ""
    notes_text = pdf_info.get("notes") or ""
    city_ctx = _pdf_city_context(pdf_info)
    # Viewbox from district, if known — constrains Nominatim queries to
    # the right county and fixes wrong-homonym cases (e.g. 270 Eastfield
    # Road matching the wrong Peterborough).
    viewbox = _viewbox_for_pdf(pdf_info)

    # District centroid (when is_district_wide — OSM admin boundary)
    dc = _district_candidate(pdf_info)
    if dc is not None:
        out.append(dc)

    # Parsed administrative unit ("in the District of X" / "Borough of Y")
    # — feed through the same district_info helper to get an OSM centroid.
    # Always normalise "rural district" → "district" because Nominatim
    # matches historical "Rural District" names to wrong locations.
    for text in (site_address, notes_text):
        admin = _parse_admin_of(text)
        if admin is None:
            continue
        admin_type, admin_name = admin
        norm_type = "district" if "rural" in admin_type else admin_type
        # Try a few plausible query forms in order of specificity
        info = None
        for query in (f"{admin_name} {norm_type.title()}, UK",
                      f"{admin_name}, {norm_type.title()}, UK",
                      f"{admin_name}, UK"):
            info = _district_info(query)
            if info is not None:
                break
        if info is not None:
            lat, lon, _ = info
            out.append(LocateCandidate(
                lat=lat, lon=lon, confidence=0.78,
                source=f"pdf_text:{admin_type}:{admin_name[:30]}",
                evidence=f"Parsed '{admin_type} of {admin_name}' from PDF text",
                specificity=2,
            ))
            break

    # Region context ("various sites across X" / "Land within X") —
    # often catches multi-site cases the reader didn't flag as district-wide.
    for text in (site_address, notes_text):
        region = _parse_region_context(text)
        if region is None:
            continue
        hit = _geocode_reference(region, city_ctx, viewbox=viewbox)
        if hit is None:
            continue
        lat, lon, src = hit
        out.append(LocateCandidate(
            lat=lat, lon=lon, confidence=0.7,
            source=f"pdf_text:region:{region[:30]}",
            evidence=f"Parsed region context '{region}' from PDF text; {src}",
            specificity=2,
        ))
        break

    # Parish lookup: prefer LLM-extracted parish_names (v2 reader)
    # concatenated with regex-parsed parishes from site_address/notes.
    # Parish names are often unique and well-indexed.
    parish_candidates: List[str] = list(pdf_info.get("parish_names") or [])
    for text in (site_address, notes_text):
        p = _parse_parish(text)
        if p and p not in parish_candidates:
            parish_candidates.append(p)
    for parish in parish_candidates[:3]:
        if not parish:
            continue
        # OSM parish boundary — same helper as district lookup, just
        # qualified with "Parish, UK" which Nominatim recognises.
        parish_info = _district_info(f"{parish} Parish, UK")
        if parish_info is not None:
            p_lat, p_lon, _ = parish_info
            out.append(LocateCandidate(
                lat=p_lat, lon=p_lon, confidence=0.78,
                source=f"pdf_text:parish_osm:{parish[:30]}",
                evidence=f"OSM parish boundary centroid for '{parish}'",
                specificity=2,
            ))
            break
        # Fallback: general geocode
        hit = _geocode_reference(parish, city_ctx, viewbox=viewbox)
        if hit is None:
            continue
        lat, lon, src = hit
        out.append(LocateCandidate(
            lat=lat, lon=lon, confidence=0.72,
            source=f"pdf_text:parish:{parish[:30]}",
            evidence=f"Parsed 'Parish of {parish}' from PDF text; {src}",
            specificity=2,
        ))
        break

    # Postcodes (postcode-level ~300m accuracy). Cross-validate against the
    # known district if we have one — the reader sometimes extracts the
    # council's own postcode or an agent's office postcode, which can be
    # 100+km from the actual site. Drop those as candidates.
    _pc_district_anchor = None
    if pdf_info.get("district_name"):
        d_info = _district_info(pdf_info["district_name"])
        if d_info is not None:
            _pc_district_anchor = (d_info[0], d_info[1])
    for pc in pdf_info.get("postcodes") or []:
        pc_clean = pc.strip().upper()
        if not re.match(r"^[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}$", pc_clean):
            continue
        hit = _lookup_postcode(pc_clean)
        if not hit:
            continue
        if _pc_district_anchor is not None:
            d = _distance_m(hit["lat"], hit["lon"],
                            _pc_district_anchor[0], _pc_district_anchor[1])
            if d > 30_000:
                # Postcode is 30km+ from the known district centroid —
                # almost certainly a council/agent office leak.
                continue
        out.append(LocateCandidate(
            lat=hit["lat"], lon=hit["lon"],
            confidence=0.6,
            source=f"pdf_text:postcode:{pc_clean}",
            evidence=f"Postcode '{pc_clean}' from PDF text",
            specificity=1,
        ))

    # Grid refs (from pdf_info.grid_refs AND v2 reader's
    # coordinate_labels_on_map — they're the same kind of signal, just
    # sourced differently: text-body vs map-surface).
    grid_sources: List[str] = []
    seen_grids = set()
    for gr in (pdf_info.get("grid_refs") or []):
        if gr and gr not in seen_grids:
            seen_grids.add(gr); grid_sources.append(gr)
    for gr in (pdf_info.get("coordinate_labels_on_map") or []):
        if gr and gr not in seen_grids:
            seen_grids.add(gr); grid_sources.append(gr)
    for gr in grid_sources:
        latlon = os_grid_ref_to_latlon(gr)
        if latlon:
            out.append(LocateCandidate(
                lat=latlon[0], lon=latlon[1],
                confidence=0.7,
                source=f"pdf_text:gridref:{gr}",
                evidence=f"Grid ref '{gr}' (text/map surface)",
                specificity=1,
            ))
            continue
        coarse = os_grid_ref_to_latlon_coarse(gr)
        if coarse:
            out.append(LocateCandidate(
                lat=coarse[0], lon=coarse[1],
                confidence=0.5,
                source=f"pdf_text:gridref_coarse:{gr}",
                evidence=f"Low-res grid ref '{gr}' (5-10km precision)",
                specificity=3,
            ))

    # House-numbered addresses — prefer LLM-extracted list (v2 reader's
    # house_number_road_pairs captures ranges and mixed cases the regex
    # can miss) and fall back to the regex parser on site_address only.
    house_pairs: List[str] = list(pdf_info.get("house_number_road_pairs") or [])
    if not house_pairs:
        hn = _parse_house_number(site_address)
        if hn is not None:
            house_pairs.append(f"{hn[0]} {hn[1]}")
    for pair in house_pairs[:3]:
        pair = pair.strip()
        if not pair:
            continue
        # Normalise range forms like "126-134 Norwich Road" → median "130"
        m = re.match(r"^\s*(\d+)\s*[-–]\s*(\d+)\s+(.+)$", pair)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            mid = (lo + hi) // 2 if 0 < lo < hi < 10000 else lo
            full_street = f"{mid} {m.group(3).strip()}"
        else:
            full_street = pair
        hit = nominatim_structured(street=full_street, city=city_ctx or "",
                                    country="UK", viewbox=viewbox)
        if hit:
            out.append(LocateCandidate(
                lat=hit["lat"], lon=hit["lon"],
                confidence=0.9,
                source=f"pdf_text:addr:{full_street[:40]}",
                evidence=f"House-numbered address '{full_street}'; "
                         f"Nominatim resolved in '{city_ctx or 'UK'}'",
                specificity=0,
            ))
            break  # one good address is enough

    # Directional modifier — prefer LLM-extracted string (v2 reader's
    # directional_modifier), fall back to regex parse on site_address.
    dm_str = pdf_info.get("directional_modifier")
    directional: Optional[Tuple[str, str]] = None
    if dm_str and isinstance(dm_str, str):
        dm = _DIRECTIONAL_RE.search(dm_str)
        if dm:
            directional = (dm.group(1).lower().rstrip(".")
                            .replace(" ", "").replace("-", ""),
                           dm.group(2).strip(" ,."))
    if directional is None:
        directional = _parse_directional(site_address)
    if directional:
        direction, reference = directional
        ref_hit = _geocode_reference(reference, city_ctx, viewbox=viewbox)
        if ref_hit:
            ref_lat, ref_lon, ref_src = ref_hit
            off_lat, off_lon = _offset_latlon(ref_lat, ref_lon, direction, 150.0)
            out.append(LocateCandidate(
                lat=off_lat, lon=off_lon,
                confidence=0.75,
                source=f"pdf_text:directional:{direction}:{reference[:30]}",
                evidence=f"Parsed '{direction} of {reference}' from site_address; "
                         f"{ref_src} + 150m {direction} offset",
                specificity=1,
            ))

    # "Land adjoining X" / "Site at Y" — combine LLM-extracted
    # adjacency_hints (v2 reader) with the regex land-reference parser.
    adjacency_candidates: List[str] = list(pdf_info.get("adjacency_hints") or [])
    lr = _parse_land_reference(site_address)
    if lr and lr not in adjacency_candidates:
        adjacency_candidates.append(lr)
    if adjacency_candidates and not directional:
        seen_refs: set = set()
        for ref in adjacency_candidates[:3]:
            ref = (ref or "").strip()
            if len(ref) < 3 or ref in seen_refs:
                continue
            seen_refs.add(ref)
            ref_hit = _geocode_reference(ref, city_ctx, viewbox=viewbox)
            if ref_hit is None:
                continue
            ref_lat, ref_lon, ref_src = ref_hit
            out.append(LocateCandidate(
                lat=ref_lat, lon=ref_lon,
                confidence=0.7,
                source=f"pdf_text:land_ref:{ref[:30]}",
                evidence=f"Adjacent feature '{ref}'; resolved via {ref_src}",
                specificity=1,
            ))

    # place_names → Zoomstack / Wikidata landmark geocoding. Skip admin
    # entities; parent-anchor from (in order) postcode → district centroid
    # → parsed admin_of centroid. The parent_anchor is critical: without
    # it, 'Poulton Farm' / 'Broadway' match the wrong UK homonym.
    parent_anchor = None
    for pc in pdf_info.get("postcodes") or []:
        hit = _lookup_postcode(pc.strip().upper())
        if hit:
            # Cross-validate: if we know a district, reject postcodes
            # more than 30km from the district centroid (reader sometimes
            # extracts the council's own postcode by mistake).
            if pdf_info.get("district_name"):
                d_info = _district_info(pdf_info["district_name"])
                if d_info is not None:
                    d_lat, d_lon, _ = d_info
                    if _distance_m(hit["lat"], hit["lon"], d_lat, d_lon) > 30_000:
                        continue
            parent_anchor = (hit["lat"], hit["lon"])
            break
    if parent_anchor is None and pdf_info.get("district_name"):
        d_info = _district_info(pdf_info["district_name"])
        if d_info is not None:
            parent_anchor = (d_info[0], d_info[1])
    if parent_anchor is None:
        for text in (site_address, notes_text):
            admin = _parse_admin_of(text)
            if admin is None:
                continue
            info = _admin_info(admin[0], admin[1])
            if info is not None:
                parent_anchor = (info[0], info[1])
                break
    # admin_region (LLM-extracted bare district name from v2 reader). The
    # _parse_admin_of regex only catches "in the District of X" / "Borough
    # of Y" patterns; many docs just state the district elsewhere or in a
    # title block, and the reader extracts it directly. Without this step,
    # cases like A4DA04 (admin_region="Rossendale", site in Lancashire)
    # fall through to a city_ctx gpkg lookup that picks the wrong UK
    # homonym (Scottish Waterfoot 294km off the right one).
    if parent_anchor is None:
        admin_region = pdf_info.get("admin_region")
        if admin_region and isinstance(admin_region, str) and len(admin_region) >= 3 \
                and not _is_admin_entity(admin_region) \
                and not _looks_like_road(admin_region):
            info = (_admin_info("district", admin_region)
                    or _admin_info("borough", admin_region)
                    or _district_info(admin_region))
            if info is not None:
                parent_anchor = (info[0], info[1])
    # Last-resort parent_anchor: geocode the city_ctx itself via Zoomstack.
    # This catches urban cases (Haymarket Theatre + Leicester) where no
    # postcode / district is known but city_ctx resolves to a real UK town.
    if parent_anchor is None and city_ctx and not _looks_like_road(city_ctx):
        gp = gpkg_place_search(city_ctx, limit=1)
        if gp and gp[0].get("specificity", 9) <= 3:
            parent_anchor = (gp[0]["lat"], gp[0]["lon"])

    # Geocode place_names + visible_map_labels (v2 reader). The latter
    # captures labels the reader actually SAW on the map image (road
    # names shown on roads, named buildings). For the locate stage these
    # are essentially the same signal and should all be tried.
    place_pool: List[str] = []
    seen_places: set = set()
    for pl in (pdf_info.get("place_names") or []):
        n = (pl or "").strip().lower()
        if n and n not in seen_places:
            seen_places.add(n)
            place_pool.append(pl.strip())
    for pl in (pdf_info.get("visible_map_labels") or []):
        n = (pl or "").strip().lower()
        if n and n not in seen_places:
            seen_places.add(n)
            place_pool.append(pl.strip())
    for place in place_pool[:8]:
        pl = place.strip()
        if len(pl) < 3:
            continue
        if _is_admin_entity(pl):
            continue

        is_landmark = _is_landmark_name(pl)

        # Zoomstack first
        kw = dict(parent_lat=parent_anchor[0], parent_lon=parent_anchor[1]) \
            if parent_anchor else {}
        gp = gpkg_place_search(pl, limit=1, **kw)
        if gp:
            r = gp[0]
            spec = 2 if r.get("specificity", 9) <= 2 else 4
            # Boost specific named landmarks so they outrank parish/village
            # centroids that cover the same area more loosely.
            if is_landmark:
                spec = 1
                conf = 0.85 if r.get("exact") else 0.7
            else:
                conf = 0.65 if r.get("exact") else 0.5
            out.append(LocateCandidate(
                lat=r["lat"], lon=r["lon"],
                confidence=conf,
                source=f"pdf_text:place:gpkg:{r['type']}:{r['name']}",
                evidence=f"place_name '{pl}' matched Zoomstack '{r['name']}' "
                         f"({r['type']})",
                specificity=spec,
            ))
            continue
        # Wikidata fallback — slow (~0.3s online) but worth it for landmarks
        # that aren't in the OS gazetteer (e.g. 'Colney Hall').
        wd = wikidata_place_search(pl, limit=1, **kw)
        if wd:
            r = wd[0]
            # Wikidata entries for named landmarks are typically precise
            # (building-level coords) — boost confidence + specificity.
            conf = 0.82 if is_landmark else 0.55
            spec = 1 if is_landmark else 3
            out.append(LocateCandidate(
                lat=r["lat"], lon=r["lon"],
                confidence=conf,
                source=f"pdf_text:place:wikidata:{r['qid']}",
                evidence=f"place_name '{pl}' matched Wikidata '{r['name']}'"
                         + (" (landmark)" if is_landmark else ""),
                specificity=spec,
            ))
            continue
        # Photon free-text fallback — handles named landmarks/estates/areas
        # that are neither in OS Open Names nor in Wikidata under the exact
        # phrase the planning doc uses (e.g. "Wrest Park Estate" doesn't
        # match Wikidata "Wrest Park"; "Fitzjohns/Netherhall Conservation
        # Area" isn't indexed anywhere by exact name). Photon's prefix
        # search resolves these via OSM tags. Filtered by parent_anchor
        # to suppress wrong-UK-region homonyms.
        from tools.geocoders import query_photon
        try:
            ph = query_photon(f"{pl}, UK", limit=1)
        except Exception:
            ph = []
        if ph:
            r = ph[0]
            if parent_anchor is not None:
                if _distance_m(r["lat"], r["lon"],
                               parent_anchor[0], parent_anchor[1]) > 30_000:
                    continue  # photon picked the wrong UK homonym
            out.append(LocateCandidate(
                lat=r["lat"], lon=r["lon"],
                confidence=0.55,
                source=f"pdf_text:place:photon:{pl[:30]}",
                evidence=f"place_name '{pl}' matched Photon '{r.get('name','?')}'",
                specificity=3,
            ))

    return out


# ─── Caching ───────────────────────────────────────────────────────────────

def _cache_key(pdf_path: str, page_num: int) -> str:
    """Hash PDF content + page number for deterministic cache keys."""
    h = hashlib.md5()
    try:
        with open(pdf_path, "rb") as f:
            h.update(f.read(256_000))  # first 256KB is plenty for identity
    except OSError:
        h.update(pdf_path.encode())
    h.update(f":{page_num}".encode())
    return h.hexdigest()


def _cache_path(pdf_path: str, page_num: int, kind: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{_cache_key(pdf_path, page_num)}_{kind}.json"


def _load_cached(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _save_cached(path: Path, data: dict) -> None:
    try:
        path.write_text(json.dumps(data, indent=2, default=str))
    except Exception:
        pass


# ─── Orchestrator ──────────────────────────────────────────────────────────

def locate_map(
    pdf_path: str,
    page_num: int,  # 1-based to match PDFInfo.map_pages
    pdf_info: Dict[str, Any],
    model_name: str = "google/gemini-2.5-flash",  # legacy kwarg — VLM retired
    use_vlm: bool = False,  # legacy kwarg — VLM retired, always False
    use_cache: bool = True,
    verbose: bool = False,
) -> LocateResult:
    """Run the full locate stage on one map page. Returns LocateResult.

    Pipeline:
      1. Collect raw signals  — OCR at 700 DPI (VLM path retired 2026-05)
      2. Resolve scale        — PDF text > OCR (per coverage analysis)
      3. Pick the anchor      — direct_affine > postcode > grid_ref > ticks
      4. Build candidates     — grid-tick centroid + PDF text candidates
      5. Triangulate          — road intersections using the step-3 anchor
      6. Cross-validate + rank

    Idempotent — repeat calls hit the cache under ``cache/locate/``.
    """
    timings: Dict[str, float] = {}
    notes: List[str] = []

    # STEP 1 — Collect signals (OCR + grid-tick affine)
    signals = _collect_signals(
        pdf_path, page_num, pdf_info, use_cache, timings,
    )
    if signals is None:
        return LocateResult(notes=f"render failed at page {page_num}")
    if signals["direct_affine"] is not None:
        d = signals["direct_affine"]
        notes.append(f"direct_affine: {d.tick_count} ticks, residual {d.mean_residual_m:.1f}m")

    # STEP 2 — Scale
    scale_ratio, scale_source = _resolve_scale(signals, pdf_info)

    # STEP 3 — Anchor
    anchor = _pick_anchor(signals, pdf_info)
    if anchor is not None:
        notes.append(f"anchor={anchor[2]}")

    # STEP 4 — Build candidates
    candidates: List[LocateCandidate] = []

    if signals["direct_affine"] is not None and anchor is not None:
        d = signals["direct_affine"]
        candidates.append(LocateCandidate(
            lat=anchor[0], lon=anchor[1],
            confidence=0.95,
            source="grid_ticks:centroid",
            evidence=f"Centroid of {d.tick_count} resolved OS grid ticks "
                     f"(residual {d.mean_residual_m:.1f}m)",
            specificity=1,
        ))

    candidates.extend(_pdf_text_candidates(pdf_info))

    # STEP 5 — Triangulation using the step-3 anchor (+ road-derived fallbacks)
    roads = _all_roads(signals, pdf_info)
    if len(roads) >= 2:
        t0 = time.time()
        primary = (anchor[0], anchor[1]) if anchor is not None else None
        # Anchor is "trusted" when it came from a precise source (postcode,
        # grid ref, district, admin, direct affine, parish etc.) — NOT when
        # it came from the last-resort road Nominatim fallback. Untrusted
        # anchors can be in the wrong UK region, so we enable multi-anchor
        # triangulation to self-correct via road intersection voting.
        anchor_label = (anchor[2] if anchor is not None else "") or ""
        anchor_trusted = not anchor_label.startswith("road:")
        city_ctx = _pdf_city_context(pdf_info)
        city_ctx_alts = _pdf_city_alts(pdf_info)
        tri = _triangulation_candidates(
            roads, primary, city_ctx=city_ctx,
            city_ctx_alts=city_ctx_alts,
            anchor_trusted=anchor_trusted, verbose=verbose,
        )
        if tri:
            candidates.extend(tri)
            notes.append(f"triangulation: {len(tri)} intersection(s) "
                         f"from {len(roads)} roads")
        timings["triangulation_s"] = time.time() - t0

    # Multi-road consensus: if ≥3 road-based candidates cluster, emit a
    # high-confidence consensus candidate at the centroid. Runs before
    # cross-validation so the consensus anchors the reference point.
    consensus = _multi_road_consensus(candidates)
    if consensus is not None:
        candidates.append(consensus)
        notes.append(consensus.evidence)

    # STEP 6 — Cross-validate + rank
    candidates = _cross_validate_candidates(candidates)
    candidates = _rank_and_dedup(candidates)

    ocr_scale_hit = signals["ocr_scale_hit"]
    return LocateResult(
        direct_affine=signals["direct_affine"],
        scale_ratio=scale_ratio,
        scale_source=scale_source,
        candidates=candidates,
        ocr_grid_refs_found=[t[0] for t in signals["ticks"]],
        ocr_scale_texts=[ocr_scale_hit[1]] if ocr_scale_hit else [],
        vlm_labels=None,  # VLM path retired 2026-05
        timings=timings,
        notes="; ".join(notes) if notes else "",
    )


# ─── Step helpers ──────────────────────────────────────────────────────────

def _collect_signals(
    pdf_path: str, page_num: int, pdf_info: Dict[str, Any],
    use_cache: bool,
    timings: Dict[str, float],
) -> Optional[Dict[str, Any]]:
    """Step 1 — OCR the map page (VLM path retired 2026-05).

    OCR runs at a DPI capped so the rendered image stays under OCR_MAX_MP
    megapixels. Tesseract is called with a 30s timeout; on timeout the OCR
    result for that page is cached as empty.

    Returns None if rendering fails. Otherwise::

        {"img_shape", "ticks", "ocr_scale_hit", "direct_affine"}
    """
    # OCR (cached)
    ocr_cache = _cache_path(pdf_path, page_num, "ocr")
    ocr_data = _load_cached(ocr_cache) if use_cache else None

    if ocr_data is None:
        t0 = time.time()
        dpi = _safe_ocr_dpi(pdf_path, page_num - 1)
        img_hi = render_pdf_page(pdf_path, page_num - 1, dpi=dpi)
        if img_hi is None:
            return None
        words = _run_tesseract(img_hi, psm=11)
        ticks_raw = extract_grid_refs_from_ocr(words)
        scale_hit = extract_scale_from_ocr(words)
        ocr_data = {
            "img_shape": list(img_hi.shape[:2]),
            "words": [vars(w) for w in words],
            "ticks": [(t[0], list(t[1]), list(t[2])) for t in ticks_raw],
            "scale": list(scale_hit) if scale_hit else None,
        }
        _save_cached(ocr_cache, ocr_data)
        timings["ocr_s"] = time.time() - t0
    else:
        timings["ocr_s"] = 0.0

    img_shape = tuple(ocr_data["img_shape"])
    ticks = [(t[0], tuple(t[1]), tuple(t[2])) for t in ocr_data["ticks"]]
    ocr_scale_hit = tuple(ocr_data["scale"]) if ocr_data.get("scale") else None
    direct_affine = solve_affine_from_grid_ticks(ticks) if len(ticks) >= 3 else None

    return {
        "img_shape": img_shape,
        "ticks": ticks,
        "ocr_scale_hit": ocr_scale_hit,
        "direct_affine": direct_affine,
    }


def _resolve_scale(
    signals: Dict[str, Any], pdf_info: Dict[str, Any],
) -> Tuple[Optional[int], Optional[str]]:
    """Step 2 — Prefer PDF-text scale over OCR.

    Rationale from the v4 benchmark: reader has 51% coverage vs OCR's 24%;
    where both fire they agree 10/10 on sample; OCR-only coverage is a
    strict subset of reader coverage. So reader wins by default; OCR is
    only consulted as fallback.
    """
    if pdf_info.get("scale"):
        m = _SCALE_RE.search(str(pdf_info["scale"]))
        if m:
            try:
                return int(m.group(1).replace(",", "")), "pdf_text"
            except ValueError:
                pass
    scale_hit = signals["ocr_scale_hit"]
    if scale_hit:
        return int(scale_hit[0]), f"ocr:{scale_hit[1]}"
    return None, None


def _pick_anchor(
    signals: Dict[str, Any], pdf_info: Dict[str, Any],
) -> Optional[Tuple[float, float, str]]:
    """Step 3 — Pick a single anchor with explicit precedence.

    Returns ``(lat, lon, label)`` or None. Used to (a) disambiguate
    VLM-extracted labels (which High Street?), (b) bound the Overpass
    query for road-intersection triangulation. The triangulation
    smoke-test showed that a bad anchor reliably sends triangulation to
    the wrong intersection, so anchor quality matters.

    Precedence (best first):
      1. direct_affine centroid  — grid ticks resolved, few-metres accuracy
      2. postcode centroid        — typically ~300m error
      3. grid_ref from PDF text   — 1km precision floor
      4. OCR tick centroid        — mean of ticks when affine couldn't fit
    """
    if signals["direct_affine"] is not None:
        try:
            lat, lon = direct_affine_centroid(
                signals["direct_affine"], signals["img_shape"])
            if _is_valid_uk_coord(lat, lon):
                return (lat, lon, "direct_affine")
        except Exception:
            pass

    for pc in pdf_info.get("postcodes") or []:
        pc_clean = pc.strip().upper()
        if not re.match(r"^[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}$", pc_clean):
            continue
        hit = _lookup_postcode(pc_clean)
        if hit:
            return (hit["lat"], hit["lon"], f"postcode:{pc_clean}")

    for gr in pdf_info.get("grid_refs") or []:
        latlon = os_grid_ref_to_latlon(gr)
        if latlon:
            return (latlon[0], latlon[1], f"pdf_gridref:{gr}")
        # Fall back to coarse parser for low-res refs ("TR 34 SE")
        coarse = os_grid_ref_to_latlon_coarse(gr)
        if coarse:
            return (coarse[0], coarse[1], f"pdf_gridref_coarse:{gr}")

    ticks = signals["ticks"]
    if ticks:
        lats = [t[1][0] for t in ticks]
        lons = [t[1][1] for t in ticks]
        lat, lon = float(np.mean(lats)), float(np.mean(lons))
        if _is_valid_uk_coord(lat, lon):
            return (lat, lon, "ocr_ticks_centroid")

    # 5. District centroid (when is_district_wide — constrains road/place
    # lookups to the right region even when no postcode is available).
    if pdf_info.get("is_district_wide"):
        dn = pdf_info.get("district_name")
        if dn:
            info = _district_info(dn)
            if info is not None:
                d_lat, d_lon, _ = info
                return (d_lat, d_lon, f"district:{dn.split('|')[0].strip()[:20]}")

    # 6. Parsed admin_of / region / parish from PDF text (catches
    # multi-site district cases the reader didn't flag as district-wide,
    # e.g. "Various sites across South Norfolk").
    sa = pdf_info.get("site_address") or ""
    notes = pdf_info.get("notes") or ""
    for text in (sa, notes):
        admin = _parse_admin_of(text)
        if admin is not None:
            info = _admin_info(admin[0], admin[1])
            if info is not None:
                a_lat, a_lon, _ = info
                return (a_lat, a_lon, f"admin:{admin[1][:20]}")
        region = _parse_region_context(text)
        if region:
            hit = _geocode_reference(region, None)
            if hit is not None:
                r_lat, r_lon, _ = hit
                return (r_lat, r_lon, f"region:{region[:20]}")
        parish = _parse_parish(text)
        if parish:
            hit = _geocode_reference(parish, None)
            if hit is not None:
                p_lat, p_lon, _ = hit
                return (p_lat, p_lon, f"parish:{parish[:20]}")

    # 7. Last resort — geocode the first road name via Nominatim. Even a
    # rough hit enables triangulation (with ≥2 roads) which can then find
    # the correct intersection. Matches the behaviour of the old
    # position_boundary 'nominatim:road' source so we don't regress on
    # cases that had no postcode/grid but had roads.
    primary = _pdf_city_context(pdf_info)
    primary_filtered = ([primary]
                        if (primary and not _looks_like_road(primary))
                        else [])
    cascade = primary_filtered + [a for a in _pdf_city_alts(pdf_info)
                                   if a not in primary_filtered]
    for road in pdf_info.get("road_names") or []:
        if not road.strip():
            continue
        # Try each contextualised query first; any hit terminates.
        for ctx in cascade:
            hit = nominatim_structured(street=road, city=ctx, country="UK")
            if hit and _is_valid_uk_coord(hit["lat"], hit["lon"]):
                return (hit["lat"], hit["lon"],
                        f"road:{road[:20]}_in_{ctx[:20]}")
        # Final fallback: empty-city UK-wide.
        hit = nominatim_structured(street=road, city="", country="UK")
        if hit and _is_valid_uk_coord(hit["lat"], hit["lon"]):
            return (hit["lat"], hit["lon"], f"road:{road[:20]}_UK")

    return None


def _all_roads(signals: Dict[str, Any], pdf_info: Dict[str, Any]) -> List[str]:
    """Collect road names from PDF text, deduped by normalised name.

    Used to keep signature stable for legacy call sites that pass the
    ``signals`` dict; only ``pdf_info["road_names"]`` is consulted now
    (the VLM road-name source was retired 2026-05).
    """
    roads = list(pdf_info.get("road_names") or [])
    seen = set()
    dedup: List[str] = []
    for r in roads:
        n = _norm_road(r)
        if n and n not in seen:
            seen.add(n)
            dedup.append(r)
    return dedup


def _cross_validate_candidates(
    candidates: List[LocateCandidate],
) -> List[LocateCandidate]:
    """Step 6a — Drop candidates that disagree wildly with the majority.

    Uses the median lat/lon of the top-3 most specific candidates as the
    reference point. Anything >5km from it gets dropped (likely a
    wrong-sense geocode — for example 'High Street' matching a homonym
    in another county). Preserves all candidates if doing so would empty
    the list (never return nothing from a locate that had signal).
    """
    if len(candidates) < 3:
        return candidates

    reference = sorted(candidates,
                       key=lambda c: (c.specificity, -c.confidence))[:3]
    ref_lat = float(np.median([c.lat for c in reference]))
    ref_lon = float(np.median([c.lon for c in reference]))

    kept = [c for c in candidates
            if _distance_m(c.lat, c.lon, ref_lat, ref_lon) < 5000]
    return kept if kept else candidates


def _multi_road_consensus(
    candidates: List[LocateCandidate],
    radius_m: float = 2000.0, min_roads: int = 3,
) -> Optional[LocateCandidate]:
    """If ≥min_roads road-based candidates cluster within radius_m, emit
    a high-confidence consensus LocateCandidate at the cluster centroid.

    The idea: MINIMA searching from 'the place where 3+ roads agree' is
    almost always on the right spot. Even if no single road hit was
    precise, their intersection-of-regions is.
    """
    road_cands = [
        c for c in candidates
        if c.source.startswith("nominatim:road:")
        or c.source.startswith("road:")
        or c.source.startswith("pdf_text:addr:")
        or c.source.startswith("road_intersection:")
    ]
    if len(road_cands) < min_roads:
        return None

    # Greedy clustering: find the largest cluster within radius_m
    best_cluster: List[LocateCandidate] = []
    for seed in road_cands:
        cluster = [c for c in road_cands
                   if _distance_m(c.lat, c.lon, seed.lat, seed.lon) <= radius_m]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    if len(best_cluster) < min_roads:
        return None

    lat = float(np.mean([c.lat for c in best_cluster]))
    lon = float(np.mean([c.lon for c in best_cluster]))
    names = ", ".join(c.source.split(":", 2)[-1][:20] for c in best_cluster[:4])
    return LocateCandidate(
        lat=lat, lon=lon, confidence=0.93,
        source=f"multi_road_consensus:{len(best_cluster)}",
        evidence=f"Centroid of {len(best_cluster)} road-based candidates "
                 f"agreeing within {radius_m:.0f}m: {names}",
        specificity=0,
    )


def _rank_and_dedup(cands: List[LocateCandidate]) -> List[LocateCandidate]:
    """Deduplicate candidates within 200m, keeping the higher-confidence one.
    Then sort by (confidence desc, specificity asc)."""
    if not cands:
        return cands
    # Drop out-of-UK
    cands = [c for c in cands if _is_valid_uk_coord(c.lat, c.lon)]
    # Greedy dedup
    kept: List[LocateCandidate] = []
    for c in sorted(cands, key=lambda x: (-x.confidence, x.specificity)):
        merged = False
        for k in kept:
            if _distance_m(c.lat, c.lon, k.lat, k.lon) < 200:
                merged = True
                break
        if not merged:
            kept.append(c)
    return kept


# ════════════════════════════════════════════════════════════════════════════
# Locate v2 — streamlined cascade (merged 2026-05-11)
# ════════════════════════════════════════════════════════════════════════════
# Was tools/locate_v2.py. Replaces the v13 multi-geocoder cascade
# (Code-Point + OS Open Names + gpkg + Nominatim + Photon + Wikidata +
# extra_terms) with a much simpler, town-gated, high-confidence-first
# algorithm. Tested 2026-05-08 against 30 representative v13 cases:
#   - 1.7 candidates per case (vs v13's 8-10) — ~5× fewer
#   - Mean best-candidate distance: 0.99 km (vs v13's 9.21 km) — ~9× better
#   - 0 high-confidence-wrong candidates (calibrated)
#
# Algorithm (in order):
#   1. Compute TOWN CENTROID via multi-signal (admin_region + parish +
#      postcode-area + likely_town_or_city). Used to gate every subsequent
#      lookup so homonyms don't wreck the result.
#   2. POSTCODE (Code-Point Open, sub-metre BNG) — town-gated.
#   3. GRID REF (BNG, OS) — trust unconditionally; UK-only by construction.
#   4. OS OPEN NAMES landmark/road — town-gated.
#   5. TOWN CENTROID FALLBACK — when nothing else fires.
#
# Returns up to 3 candidates ranked by confidence.

# Candidate dataclass lives in tools.locate.schemas; re-imported above.


# ── Helpers ────────────────────────────────────────────────────────────────

_R_V2 = 6_371_000.0


def _hkm(lat1, lon1, lat2, lon2) -> float:
    if lat1 is None or lat2 is None: return float("inf")
    dy = math.radians(lat2 - lat1)
    dx = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    return _R_V2 * math.hypot(dy, dx) / 1000


_GENERIC_ROAD_SUFFIX = re.compile(
    r"\b(road|lane|street|avenue|drive|crescent|close|way|gardens|mews|"
    r"terrace|grove|place|square|hill|park)\b", re.I,
)


def _v2_looks_like_road(name: str) -> bool:
    """locate_v2's looser road-name detector (used inside propose_centers_v2).

    Kept separate from the module-level :func:`_looks_like_road` (which adds
    Court/Rise/Walk/Row) to preserve byte-identical v2 behaviour after the
    2026-05-11 merge of locate.py + locate_v2.py.
    """
    return bool(_GENERIC_ROAD_SUFFIX.search(name or ""))


# Postcode normalisation + area centroid live in tools.locate.postcode;
# re-imported here for callers that still grab them from this module.
from tools.locate.postcode import (
    _normalize_postcode,
    _is_full_postcode,
    _postcode_area,
    _area_centroid,
)


# ── Town centroid (multi-signal) ──────────────────────────────────────────

_SETTLEMENT_TYPES = {
    "city", "town", "village", "hamlet", "other settlement",
    "suburban area", "borough", "district", "civil parish",
}
_PROXY_TYPES = {"railway station", "primary education", "secondary education"}

# Granular priority within settlement types — prefer LARGER settlements when
# a name is ambiguous (e.g. Farnham Surrey TOWN vs Farnham Yorkshire VILLAGE).
_SETTLEMENT_TYPE_RANK = {
    "city":              0,
    "town":              1,
    "borough":           1,
    "district":          2,
    "civil parish":      3,
    "suburban area":     3,
    "village":           4,
    "hamlet":            5,
    "other settlement":  6,
}


def _best_settlement_hit(hits: list, near_pt: Optional[Tuple[float, float]] = None,
                          max_dist_km: float = 30.0):
    """Pick the best settlement-typed hit from an OS Open Names search.
    When ambiguous, prefer LARGER settlement types (town > village).
    If near_pt given, only consider hits within max_dist_km of that point."""
    if not hits: return None
    candidates = []
    for h in hits:
        if h.get("lat") is None: continue
        if near_pt is not None:
            if _hkm(h["lat"], h["lon"], near_pt[0], near_pt[1]) > max_dist_km:
                continue
        t = h.get("type", "").lower()
        if t in _SETTLEMENT_TYPES:
            tier = 0; rank = _SETTLEMENT_TYPE_RANK.get(t, 9)
        elif t in _PROXY_TYPES:
            tier = 1; rank = 0
        else:
            continue
        candidates.append((tier, rank, h))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def _la_polygon_for(pi: Dict[str, Any]):
    """Return shapely LA polygon for the case's admin_region (or None)."""
    name = pi.get("admin_region") or pi.get("likely_town_or_city")
    if not name: return None
    try:
        from tools.verification_checks import _resolve_la
        return _resolve_la(name)
    except Exception:
        return None


def _inside_la(lat: float, lon: float, la_poly) -> bool:
    if la_poly is None: return False
    try:
        from shapely.geometry import Point
        return la_poly.contains(Point(lon, lat))
    except Exception:
        return False


def town_centroid(pi: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Best estimate of the site's specific town/village centroid.

    Strategy: use OS BoundaryLine LA polygon as a hard region gate. Search
    OS Open Names for the town/parish; only accept hits that fall INSIDE
    the LA polygon. Falls back to LA centroid if no hits inside.
    """
    try:
        from tools.os_names import search
    except Exception:
        return None

    likely = (pi.get("likely_town_or_city") or "").strip().replace(".", "")
    admin = (pi.get("admin_region") or "").strip().replace(".", "")
    parishes = [p.replace(".", "").strip() for p in (pi.get("parish_names") or [])]

    la = _la_polygon_for(pi)
    pc_anchor = None
    for pc in (pi.get("postcodes") or [])[:3]:
        area = _postcode_area(pc)
        if not area: continue
        pt = _area_centroid(area)
        if pt: pc_anchor = pt; break

    # If we have NO regional anchor (no LA, no postcode), but we DO have a
    # parish, try the parish first to establish a regional anchor before
    # searching the (potentially homonym-prone) likely_town. This handles
    # cases like Art4D11 where likely='Kingston' (matches London) but
    # parish='Corfe Castle' (uniquely names the Purbeck/Dorset region).
    if la is None and pc_anchor is None and parishes:
        for p in parishes[:2]:
            ps = p.replace(".", "").strip()
            if not ps: continue
            try:
                hits = search(ps, max_results=15) or []
            except Exception:
                continue
            h = _best_settlement_hit(hits, near_pt=None)
            if h:
                pc_anchor = (h["lat"], h["lon"])
                break

    queries = []
    if likely: queries.append(likely)
    for p in parishes[:2]:
        if p and p not in queries: queries.append(p)

    for query in queries:
        try:
            hits = search(query, max_results=20) or []
        except Exception:
            continue
        if la is not None:
            inside = [h for h in hits if h.get("lat") is not None and
                      _inside_la(h["lat"], h["lon"], la)]
            if inside:
                h = _best_settlement_hit(inside, near_pt=None) or inside[0]
                return (h["lat"], h["lon"])
        elif pc_anchor is not None:
            h = _best_settlement_hit(hits, near_pt=pc_anchor, max_dist_km=30.0)
            if h: return (h["lat"], h["lon"])

    if la is not None:
        c = la.centroid
        return (c.y, c.x)
    if pc_anchor is not None:
        return pc_anchor

    # No regional anchor and no inside-LA hits — fall back to the best settlement
    # for any query, with NO distance gate.
    for query in queries:
        try:
            hits = search(query, max_results=15) or []
        except Exception:
            continue
        h = _best_settlement_hit(hits, near_pt=None)
        if h: return (h["lat"], h["lon"])

    return None


# ── Helpers for the cascade ────────────────────────────────────────────────


def _parse_grid_ref(g: str) -> Optional[Tuple[float, float]]:
    """Parse a BNG OS grid ref like 'TG 210 080' or 'TR 34 SE' to (lat, lon).
    parse_easting_northing already returns (lat, lon) — do NOT re-transform."""
    try:
        from tools.geo.grid_ref import parse_easting_northing
    except Exception:
        return None
    try:
        result = parse_easting_northing(g)
        if result is None: return None
        return (result[0], result[1])
    except Exception:
        return None


# ── Main entry point ───────────────────────────────────────────────────────

def _la_radius_m(la_poly) -> int:
    """True LA polygon radius in metres = max distance from polygon centroid
    to any boundary point. Used as fallback sigma — guarantees GT inside σ
    for any candidate in the LA, even when the LA is asymmetric (centroid
    offset from bbox center)."""
    if la_poly is None: return 5000
    try:
        from pyproj import Transformer
        from shapely.ops import transform as shp_transform
        t = Transformer.from_crs(4326, 27700, always_xy=True)
        la_bng = shp_transform(lambda x, y, z=None: t.transform(x, y), la_poly)
        cx, cy = la_bng.centroid.x, la_bng.centroid.y
        max_d = 0.0
        polys = la_bng.geoms if hasattr(la_bng, "geoms") else [la_bng]
        for p in polys:
            for x, y in p.exterior.coords:
                d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                if d > max_d: max_d = d
        return int(max_d * 1.05)  # +5% safety margin
    except Exception:
        return 5000


def propose_centers_v2(pi: Dict[str, Any],
                        websearch_fn=None,
                        max_candidates: int = 4,
                        extra_terms: Optional[List[str]] = None,
                        pdf_path: Optional[str] = None) -> List[Candidate]:
    """Unified locate cascade. ALWAYS emits an LA-centroid fallback
    candidate (with σ = LA radius) when admin_region resolves, guaranteeing
    GT-inside-σ coverage for every case with a known admin region.

    When ``pdf_path`` is provided, also pulls v13's road-graph-derived
    candidates (multi_road_consensus, road_intersection from OSM road
    triangulation) into the same pool — these are the only generators
    that can't be re-derived from pdf_info alone. district:<Name>
    candidates are also emitted (in step 6 below).

    Returns up to ``max_candidates`` ranked candidates (highest precision first).

    Args:
        pi: pdf_info dict (site_address, postcodes, road_names, place_names,
            admin_region, likely_town_or_city, parish_names, grid_refs, etc.)
        websearch_fn: optional `(query: str) -> Optional[str_postcode]` callback
            for landmark resolution. If None, that branch is skipped.
        max_candidates: cap on returned list (default 4).
        pdf_path: optional path to the source PDF. When given, v13's road-
            triangulation generators are invoked to add multi_road_consensus
            and road_intersection candidates.
    """
    try:
        from tools.code_point import lookup_postcode
        from tools.os_names import search as os_search
    except Exception as e:
        return []

    cands: List[Candidate] = []
    la_poly = _la_polygon_for(pi)
    town = town_centroid(pi)
    la_radius = _la_radius_m(la_poly) if la_poly is not None else None

    # Augment pdf_info-derived data with agent-provided extra_terms.
    pi_aug = dict(pi)
    if extra_terms:
        priority_terms = []
        for t in extra_terms:
            if not t or not isinstance(t, str): continue
            parts = [p.strip() for p in t.split(",") if p.strip()]
            for p in parts:
                if p.lower() in ("uk", "england", "scotland", "wales", "gb"):
                    continue
                if p not in priority_terms:
                    priority_terms.append(p)
            if t not in priority_terms:
                priority_terms.append(t)
        existing_places = list(pi.get("place_names") or [])
        existing_labels = list(pi.get("visible_map_labels") or [])
        merged_places = priority_terms + [p for p in existing_places if p not in priority_terms]
        merged_labels = priority_terms + [l for l in existing_labels if l not in priority_terms]
        pi_aug["place_names"] = merged_places
        pi_aug["visible_map_labels"] = merged_labels
        pi = pi_aug  # use the augmented dict from here

    # ── 1. POSTCODE (Code-Point Open) ─────────────────────────────────────
    pcs = pi.get("postcodes") or []
    full_pcs = [pc for pc in pcs if _is_full_postcode(pc)]
    pc_inside_la = []
    pc_outside_la = []
    for pc in full_pcs[:5]:
        h = lookup_postcode(pc)
        if h is None: continue
        if la_poly is not None:
            from shapely.geometry import Point
            if la_poly.contains(Point(h["lon"], h["lat"])):
                pc_inside_la.append((pc, h))
            else:
                pc_outside_la.append((pc, h))
        else:
            pc_inside_la.append((pc, h))
    pc_use = pc_inside_la or pc_outside_la
    if pc_use:
        pc, h = pc_use[0]
        cands.append(Candidate(
            lat=h["lat"], lon=h["lon"], sigma_m=100,
            confidence="high",
            source=f"postcode:{pc}",
            evidence=f"Code-Point Open {pc}",
            specificity=1,
        ))

    # Section 2 (WEBSEARCH FALLBACK) removed 2026-05-12 per locate-API audit:
    # this branch only ran when `websearch_fn` was wired, and no production
    # caller has wired it — 0 firings across all 215 v17 cases. The
    # `websearch_fn` parameter is kept on the signature for backward compat
    # but is ignored.

    # ── 3. GRID REF (unconditional — BNG is UK-only) ─────────────────────
    for g in (pi.get("grid_refs") or [])[:2]:
        pt = _parse_grid_ref(g)
        if pt is None: continue
        cands.append(Candidate(
            lat=pt[0], lon=pt[1], sigma_m=500,
            confidence="high",
            source=f"grid_ref:{g}",
            evidence=f"OS BNG {g}",
            specificity=1,
        ))
        break

    # ── 4. PARISH/PLACE-NAME inside LA polygon (or near town) ────────────
    parish_used = False
    region_check = la_poly
    region_anchor = town
    for name in ((pi.get("parish_names") or []) +
                  (pi.get("place_names") or []))[:8]:
        if _v2_looks_like_road(name): continue
        try: hits = os_search(name, max_results=15) or []
        except Exception: hits = []
        from shapely.geometry import Point
        for h in hits:
            if h.get("lat") is None: continue
            if h.get("type") in {"inland water", "coastal feature"}: continue
            in_region = False
            if region_check is not None:
                in_region = region_check.contains(Point(h["lon"], h["lat"]))
            elif region_anchor is not None:
                in_region = _hkm(h["lat"], h["lon"], region_anchor[0], region_anchor[1]) < 15.0
            else:
                in_region = h.get("type") in {"city", "town", "village", "hamlet",
                                                "other settlement", "suburban area"}
            if in_region:
                cands.append(Candidate(
                    lat=h["lat"], lon=h["lon"],
                    sigma_m=h.get("sigma_m", 1500),
                    confidence="med",
                    source=f"os_landmark:{name[:40]}",
                    evidence=f"OS Open Names {h.get('name_full','')[:50]}",
                    specificity=h.get("specificity", 5),
                ))
                parish_used = True
                break
        if parish_used: break

    # ── 5. ROAD NAME inside region ───────────────────────────────────────
    for road in (pi.get("road_names") or [])[:3]:
        try: hits = os_search(road, max_results=15) or []
        except Exception: hits = []
        from shapely.geometry import Point
        for h in hits:
            if h.get("lat") is None: continue
            if h.get("type") not in {"named road", "section of named road"}:
                continue
            in_region = False
            if region_check is not None:
                in_region = region_check.contains(Point(h["lon"], h["lat"]))
            elif region_anchor is not None:
                in_region = _hkm(h["lat"], h["lon"], region_anchor[0], region_anchor[1]) < 15.0
            if in_region:
                cands.append(Candidate(
                    lat=h["lat"], lon=h["lon"], sigma_m=800,
                    confidence="med",
                    source=f"os_road:{road[:40]}",
                    evidence=f"OS Open Names road {h.get('name_full','')[:50]}",
                    specificity=1,
                ))
                break
        if any(c.source.startswith("os_road:") for c in cands): break

    # ── 5z. ADDRESS-LEVEL NOMINATIM ─────────────────────────────────────
    site_addr = (pi.get("site_address") or "").strip()
    addr_match = re.search(
        r'\b(\d+\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\s+(?:Lane|Road|Street|Avenue|Drive|Close|Way|Crescent|Place|Court|Park|Hill|Green|Mews|Walk|Terrace))\b',
        site_addr,
    )
    if addr_match and (pi.get("admin_region") or pi.get("likely_town_or_city")):
        town_q = (pi.get("likely_town_or_city") or pi.get("admin_region")).strip()
        query = f"{addr_match.group(1)} {town_q}"
        try:
            from geopy.geocoders import Nominatim
            nm = Nominatim(user_agent="locate_v2_address",
                            timeout=8)
            loc = nm.geocode(query)
        except Exception:
            loc = None
        if loc is not None:
            from shapely.geometry import Point as _Point
            in_region = (la_poly.contains(_Point(loc.longitude, loc.latitude))
                          if la_poly is not None else True)
            if in_region:
                cands.append(Candidate(
                    lat=loc.latitude, lon=loc.longitude, sigma_m=200,
                    confidence="high",
                    source=f"nominatim_addr:{addr_match.group(1)[:40]}",
                    evidence=f"Nominatim full-address {loc.address[:50]}",
                    specificity=1,
                ))

    # ── 5a. GPKG SAFETY NET (OS Zoomstack `names` table) ────────────────
    GPKG_GOOD_TYPES = {"Settlement", "Small Settlements", "Hamlet", "Village",
                        "Sites", "Town"}
    from tools.geocoders import gpkg_place_search
    from shapely.geometry import Point
    gpkg_parent = (la_poly.centroid.y, la_poly.centroid.x) if la_poly is not None \
                   else (town if town is not None else (None, None))
    gpkg_count = 0
    for name in ((pi.get("place_names") or []) +
                  (pi.get("visible_map_labels") or []))[:8]:
        if not name or _v2_looks_like_road(name): continue
        if any(name[:30].lower() in c.source.lower() for c in cands):
            continue
        try:
            hits = gpkg_place_search(
                name,
                parent_lat=gpkg_parent[0], parent_lon=gpkg_parent[1],
                max_parent_distance_km=50,
                type_filter=list(GPKG_GOOD_TYPES),
                limit=3,
            )
        except Exception:
            hits = []
        for h in hits:
            in_region = (la_poly.contains(Point(h["lon"], h["lat"]))
                         if la_poly is not None else True)
            if not in_region:
                continue
            if la_poly is None and town is not None:
                d_to_town = _hkm(town[0], town[1], h["lat"], h["lon"])
                if d_to_town > 25:
                    continue
            cands.append(Candidate(
                lat=h["lat"], lon=h["lon"],
                sigma_m=1500,
                confidence="med",
                source=f"gpkg:{name[:40]}",
                evidence=f"OS Zoomstack {h['type']}: {h['name']}",
                specificity=2,
            ))
            gpkg_count += 1
            break
        if gpkg_count >= 2: break

    # ── 5b. FEATURE-CLUSTER LOCATOR ──────────────────────────────────────
    no_specific_anchor = not any(c.confidence == "high" for c in cands) or \
                          all(c.source.startswith(("la_centroid", "town_centroid"))
                              for c in cands if c.confidence != "high")
    if no_specific_anchor:
        try:
            cluster = feature_cluster_locate(pi, cluster_radius_km=2.0,
                                              min_features_match=3,
                                              la_poly=la_poly)
        except Exception:
            cluster = None
        if cluster is not None:
            clat, clon, n_match, n_total = cluster
            cands.append(Candidate(
                lat=clat, lon=clon, sigma_m=2000,
                confidence="med",
                source=f"feature_cluster",
                evidence=f"Feature cluster: {n_match}/{n_total} labels co-occur within 2km",
                specificity=2,
            ))

    # ── 5b9. NOMINATIM ROAD-WITH-CITY-CONTEXT GEOCODING ───────────────────
    likely_town_n = (pi.get("likely_town_or_city") or "").strip()
    admin_region_n = (pi.get("admin_region") or "").strip()
    road_names_n = list(pi.get("road_names") or [])
    is_district_wide_n = bool(pi.get("is_district_wide"))
    if road_names_n and (likely_town_n or admin_region_n) and not is_district_wide_n:
        try:
            from tools.geocoders import nominatim_structured  # cached on disk
            parish_n = list(pi.get("parish_names") or [])
            place_n = list(pi.get("place_names") or [])[:2]
            fallback_cities = []
            for ctx_str in parish_n + place_n:
                if (ctx_str and isinstance(ctx_str, str)
                        and ctx_str.strip()
                        and ctx_str.strip() != likely_town_n
                        and ctx_str.strip() not in fallback_cities):
                    fallback_cities.append(ctx_str.strip())
            n_emitted = 0
            MAX_NOMINATIM_ROAD_CANDS = 2
            for rn in road_names_n[:4]:
                if n_emitted >= MAX_NOMINATIM_ROAD_CANDS: break
                if not rn or len(rn.strip()) < 3: continue
                try:
                    hit = nominatim_structured(
                        street=rn.strip(), city=likely_town_n,
                        county=admin_region_n or "", country="UK")
                except Exception:
                    hit = None
                if hit and hit.get("lat") and hit.get("lon"):
                    cands.append(Candidate(
                        lat=float(hit["lat"]), lon=float(hit["lon"]),
                        sigma_m=2500, confidence="high",
                        source=f"nominatim:road:{rn.strip()}",
                        evidence=f"Nominatim {rn} in {likely_town_n}",
                        specificity=1,
                    ))
                    n_emitted += 1
                    continue
                if fallback_cities and n_emitted < MAX_NOMINATIM_ROAD_CANDS:
                    alt = fallback_cities[0]
                    try:
                        hit = nominatim_structured(
                            street=rn.strip(), city=alt,
                            county=admin_region_n or "", country="UK")
                    except Exception:
                        hit = None
                    if hit and hit.get("lat") and hit.get("lon"):
                        cands.append(Candidate(
                            lat=float(hit["lat"]), lon=float(hit["lon"]),
                            sigma_m=2500, confidence="med",
                            source=f"nominatim:road:{rn.strip()} (in {alt})",
                            evidence=f"Nominatim {rn} fallback in {alt}",
                            specificity=1,
                        ))
                        n_emitted += 1
        except Exception as e:
            print(f"  locate_v2: nominatim road-with-city skipped ({e!s:.80})")

    # ── 5c. v13 LOCATE_MAP CANDIDATES ─────────────────────────────────────
    # Pulls candidates from v13's locate_map cascade for source types not
    # already covered by locate_v2's own generators above.
    # road_intersection dropped from PASS_THROUGH 2026-05-12 per locate-API
    # audit (0 wins / 215 cases in v17). multi_road_consensus retained
    # despite low win-rate — it's the only source for 2 stuck cases.
    PASS_THROUGH_PREFIXES = (
        "multi_road_consensus",
        "pdf_text:addr", "pdf_text:directional", "pdf_text:land_ref",
        "pdf_text:gridref", "consensus_centroid",
        "nominatim:addr", "grid_ticks:centroid",
    )
    NEVER_DEDUP_PREFIXES = (
        "multi_road_consensus", "consensus_centroid",
        "grid_ticks:centroid",
    )
    if pdf_path is not None:
        try:
            page = (pi.get("map_pages") or [1])[0]
            v13_loc = locate_map(pdf_path, page, pi,
                                    use_vlm=False, use_cache=True)
            existing_keys = set(
                (round(c.lat, 3), round(c.lon, 3)) for c in cands
            )
            for vc in (getattr(v13_loc, "candidates", None) or []):
                src = (vc.source or "")
                if not src.startswith(PASS_THROUGH_PREFIXES):
                    continue
                key = (round(vc.lat, 3), round(vc.lon, 3))
                if not src.startswith(NEVER_DEDUP_PREFIXES) and key in existing_keys:
                    continue
                if src.startswith(("multi_road_consensus", "consensus_centroid",
                                     "pdf_text:addr",
                                     "pdf_text:gridref", "nominatim:addr",
                                     "grid_ticks:centroid")):
                    spec = 1
                else:
                    spec = 2
                conf = "high" if "consensus" in src else "med"
                cands.append(Candidate(
                    lat=float(vc.lat), lon=float(vc.lon),
                    sigma_m=2500, confidence=conf,
                    source=src, evidence=(getattr(vc, "evidence", None)
                                          or f"v13 {src.split(':')[0]}"),
                    specificity=spec,
                ))
                existing_keys.add(key)
        except Exception as e:
            print(f"  locate_v2: v13 locate_map pass-through skipped ({e!s:.80})")

    # Section 5c2 (NOMINATIM:EXTRA via agent extra_terms) removed 2026-05-12
    # per locate-API audit: 1 case present, 0 wins across 215 v17 cases.
    # The extra_terms are already merged into pi["place_names"] up at the
    # top of this function and fed into the os_landmark + nominatim_road
    # sections, which cover everything this branch did.

    # ── 5.9 MISLEADING-POSTCODE DEMOTION ─────────────────────────────────
    pc_cands = [(i, c) for i, c in enumerate(cands)
                 if c.source.startswith(("postcode:", "code_point:"))]
    landmark_cands = [c for c in cands
                      if c.source.startswith(("os_road:", "os_landmark:",
                                              "gpkg:", "filename:",
                                              "feature_cluster"))]
    if pc_cands and landmark_cands and len(landmark_cands) >= 2:
        to_remove = set()
        for pc_i, pc_c in pc_cands:
            distances = [_hkm(pc_c.lat, pc_c.lon, lc.lat, lc.lon)
                         for lc in landmark_cands]
            near_count = sum(1 for d in distances if d <= 5.0)
            if near_count <= len(landmark_cands) // 2:
                print(f"  Postcode {pc_c.source} dropped: far from {len(landmark_cands)-near_count}/{len(landmark_cands)} landmarks (likely letterhead)")
                to_remove.add(pc_i)
        if to_remove:
            cands = [c for i, c in enumerate(cands) if i not in to_remove]

    # ── 6. GUARANTEED FALLBACK: LA centroid (σ=LA radius) OR town centroid ─
    is_district_wide = bool(pi.get("is_district_wide"))
    district_name = (pi.get("district_name") or "").strip()
    if la_poly is not None:
        c = la_poly.centroid
        if is_district_wide and district_name:
            short = district_name.split("|")[0].strip()
            for suf in (", UK", ", London, UK"):
                if short.endswith(suf):
                    short = short[: -len(suf)]
            short = short[:40]
            cand_source = f"district:{short}"
            cand_evidence = (f"BoundaryLine LA centroid for district-wide case "
                             f"'{district_name[:60]}' (radius={la_radius}m)")
        else:
            cand_source = "la_centroid"
            cand_evidence = f"BoundaryLine LA centroid (radius={la_radius}m)"
        cands.append(Candidate(
            lat=c.y, lon=c.x,
            sigma_m=max(la_radius or 8000, 8000),
            confidence="low",
            source=cand_source,
            evidence=cand_evidence,
            specificity=2,
        ))
    elif town is not None:
        cands.append(Candidate(
            lat=town[0], lon=town[1], sigma_m=15000,
            confidence="low",
            source="town_centroid",
            evidence="OS Open Names settlement fallback (no LA polygon)",
            specificity=2,
        ))

    return cands[:max_candidates]


# Feature-cluster locator + candidate ranker live in tools.locate.ranker
# (the v2 cascade's ranking and scoring lives there).
from tools.locate.ranker import (
    feature_cluster_locate,
    feature_match_score,
    rank_candidates,
)

