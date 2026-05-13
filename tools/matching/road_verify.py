"""Road-name and directional-modifier verification helpers.

Two cross-checks used by :func:`sliding_window_position` when picking between
candidates that have similar inlier counts but plausibly different anchors:

* **Road-name verifier** — query the OS Zoomstack GeoPackage for road names
  near each candidate's predicted centre, fuzzy-match against the road names
  the reader extracted from the PDF, and prefer the candidate with the best
  road-name overlap. Overrides the metric-best candidate only when the
  override has at least 60 % road-name match AND at least 2× the top
  candidate's ratio AND ≥70 % of the top candidate's metric.

* **Directional verifier** — when the reader extracts a directional modifier
  like ``"south of East Langdon village"``, the predicted centre must lie
  roughly south of the geocoded anchor. Candidates on the opposite side get
  a metric penalty inside the main loop. This module exposes the parser +
  bearing primitives; the penalty itself is applied in
  :mod:`tools.matching._core`.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ─── Road-name verifier ────────────────────────────────────────────────────

def _query_gpkg_road_names(lat, lon, radius_m=1500):
    """Query OS GeoPackage for road names near a point. Fully offline."""
    try:
        import geopandas as gpd
        import pyproj

        gpkg_path = BASE_DIR / "os_opendata" / "OS_Open_Zoomstack.gpkg"
        if not gpkg_path.exists():
            return []

        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", "EPSG:27700", always_xy=True)
        x, y = transformer.transform(lon, lat)

        names = set()
        for layer in ["roads_local", "roads_regional", "roads_national"]:
            try:
                gdf = gpd.read_file(
                    str(gpkg_path), layer=layer,
                    bbox=(x - radius_m, y - radius_m,
                          x + radius_m, y + radius_m))
                for _, row in gdf.iterrows():
                    name = row.get("name")
                    if name and str(name).strip() and str(name) != "None":
                        names.add(str(name).strip())
            except Exception:
                pass
        return list(names)
    except ImportError:
        return []


def _fuzzy_road_match(llm_name, reference_names):
    """Check if an LLM-extracted road name matches any reference name."""
    llm_lower = llm_name.lower().strip()
    for ref in reference_names:
        ref_lower = ref.lower().strip()
        if llm_lower == ref_lower:
            return True
        if llm_lower in ref_lower or ref_lower in llm_lower:
            return True
        # Handle common abbreviations
        llm_norm = (llm_lower
                    .replace(" street", " st").replace(" road", " rd")
                    .replace(" lane", " ln").replace(" avenue", " ave")
                    .replace(" drive", " dr").replace(" close", " cl"))
        ref_norm = (ref_lower
                    .replace(" street", " st").replace(" road", " rd")
                    .replace(" lane", " ln").replace(" avenue", " ave")
                    .replace(" drive", " dr").replace(" close", " cl"))
        if llm_norm == ref_norm:
            return True
    return False


def _verify_candidates_with_road_names(ranked_candidates, road_names):
    """Pick the best candidate where nearby OSM roads match LLM road names.

    Only overrides the top candidate if:
    - Top candidate has NO road name matches AND a lower-ranked one DOES, AND
    - The override has ≥70% of top's metric score, OR
    - The override has ≥60% road match AND ≥2× top's ratio AND ≥70% top metric.

    Returns the verified candidate dict, or None to use default.
    """
    if not road_names:
        return None

    top_metric = ranked_candidates[0][0]
    min_metric = top_metric * 0.5  # candidate must be at least 50% as good

    results = []
    for metric, _seq, candidate in ranked_candidates:
        if metric < min_metric:
            break

        center_ll = candidate["match_info"].get("center_latlon")
        if not center_ll:
            results.append((metric, candidate, 0, 0))
            continue

        lat, lon = center_ll
        nearby_roads = _query_gpkg_road_names(lat, lon, radius_m=1500)

        if not nearby_roads:
            results.append((metric, candidate, 0, 0))
            continue

        matches = sum(1 for rn in road_names
                      if _fuzzy_road_match(rn, nearby_roads))
        results.append((metric, candidate, matches, len(road_names)))

    if not results:
        return None

    # Log verification results
    for metric, cand, matches, total in results:
        cname = cand["match_info"]["center"]
        inliers = cand["match_info"]["n_inliers"]
        print(f"    Road verify: {cname} inl={inliers} metric={metric:.1f} "
              f"roads={matches}/{total}")

    # Analyse all candidates by road-match quality.
    top_metric_v = results[0][0]
    top_cand = results[0][1]
    top_matches = results[0][2]
    top_total = results[0][3]
    top_ratio = top_matches / max(1, top_total)

    # Find the candidate with the best road-match ratio (ties broken by
    # higher metric). We compare this alternative to the top candidate.
    best_by_roads = max(results, key=lambda r: (r[2] / max(1, r[3]), r[0]))
    br_metric_v, br_cand, br_matches, br_total = best_by_roads
    br_ratio = br_matches / max(1, br_total)

    # Override rule: a candidate with DRAMATICALLY more road matches should
    # win even if its raw metric is slightly lower. Fixes cases where a
    # postcode-centroid or wikidata borough wins by raw inliers (1/9 roads)
    # but a Nominatim street anchor would win by 8/9 with perfect local
    # alignment. Conditions (all required):
    #   - it's NOT already the top candidate
    #   - its road-match ratio is ≥60% AND ≥ 2× top ratio
    #   - its metric is ≥70% of the top metric (not drastically worse)
    if (br_cand is not top_cand
        and br_ratio >= 0.6 and br_ratio >= 2 * top_ratio + 0.01
        and br_metric_v >= 0.7 * top_metric_v):
        cname = br_cand["match_info"]["center"]
        print(f"    Road verify: OVERRIDE → {cname} "
              f"({br_matches}/{br_total}={br_ratio:.0%} roads vs top "
              f"{top_matches}/{top_total}={top_ratio:.0%}, "
              f"metric={br_metric_v:.1f} vs {top_metric_v:.1f})")
        return br_cand

    # Fallback: legacy override when top has 0 matches but another does.
    # Only fire if the alternative's metric is close to top's (≥70%). Road
    # names in gpkg data can be sparse/noisy, especially in rural areas —
    # a partial road match (e.g. 2/4) is not enough to override a clearly-
    # better MINIMA match (e.g. inliers 39 vs 20 = 2x ratio).
    if top_matches == 0:
        for metric, candidate, matches, total in results[1:]:
            if matches == 0:
                continue
            if metric < 0.7 * top_metric_v:
                continue
            cname = candidate["match_info"]["center"]
            print(f"    Road verify: OVERRIDE (top had 0) → {cname} "
                  f"({matches}/{total} roads matched, "
                  f"metric={metric:.1f} vs top={top_metric_v:.1f})")
            return candidate
        print("    Road verify: top had 0 matches but alternatives "
              "too weak metric-wise, keeping top")
        return None

    print("    Road verify: top candidate confirmed")
    return None


# ─── Directional verifier ──────────────────────────────────────────────────
# Regex-based parser + great-circle bearing check. When the reader extracts
# a `directional_modifier` like "south of the village", the predicted center
# is expected to lie south of the geocoder anchor.

# Direction must appear at the start of the modifier string (after optional
# preposition strip). Compound directions listed first so "south-east" beats
# "south". `(ern)?` handles "northern"/"south western"/etc.
_DIRECTION_PATTERNS_ANCHORED = [
    (r'^north[\s-]?east(ern)?\b', 45),
    (r'^south[\s-]?east(ern)?\b', 135),
    (r'^south[\s-]?west(ern)?\b', 225),
    (r'^north[\s-]?west(ern)?\b', 315),
    (r'^north(ern)?\b', 0),
    (r'^south(ern)?\b', 180),
    (r'^east(ern)?\b', 90),
    (r'^west(ern)?\b', 270),
]

# For multi-directional detection ("north, south and east of X") — search
# anywhere with word boundaries.
_DIRECTION_ANYWHERE = (
    r'\b(north[\s-]?east(?:ern)?|south[\s-]?east(?:ern)?|'
    r'south[\s-]?west(?:ern)?|north[\s-]?west(?:ern)?|'
    r'north(?:ern)?|south(?:ern)?|east(?:ern)?|west(?:ern)?)\b'
)


def _parse_directional_bearing(text: Optional[str]) -> Optional[int]:
    """Parse 'south of village center' → 180° (site is S of anchor).

    Returns expected bearing FROM anchor TO predicted center, in degrees
    from north (clockwise). Returns None when the modifier doesn't contain
    a parseable compass direction (e.g. 'rear of X', 'opposite X',
    'between A and B'); the verifier then skips this case safely.

    Anchors at the start of the string so place names containing cardinal
    words ("rear of South Hill Park", "off Collinge Fold Lane") do not
    spuriously match.
    """
    if not text:
        return None
    t = str(text).lower().strip()
    # Strip common leading prepositions
    t = re.sub(r'^(to the\s+|on the\s+|at the\s+|just\s+|directly\s+)', '', t)
    # Only consider the part BEFORE " of " — what follows is place names that
    # may contain cardinal-word collisions ("east of East Langdon").
    if ' of ' in t:
        t = t.split(' of ', 1)[0]
    # Ambiguous multi-directional ("north, south and east of X")
    found = re.findall(_DIRECTION_ANYWHERE, t)

    def _normalize(m):
        s = re.sub(r'[\s-]', '', m)
        s = s.replace('ern', '')
        return s

    normalized = set(_normalize(m) for m in found)
    if len(normalized) > 1:
        return None  # multi-directional, skip safely
    for pattern, bearing in _DIRECTION_PATTERNS_ANCHORED:
        if re.match(pattern, t):
            return bearing
    return None


def _bearing_deg(lat1, lon1, lat2, lon2):
    """Great-circle initial bearing from (lat1,lon1) to (lat2,lon2) in
    degrees from north, clockwise. 0=N, 90=E, 180=S, 270=W.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(dlon))
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360.0) % 360.0


def _angular_diff_deg(a, b):
    """Smallest absolute angular difference between two bearings (0-180°)."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d
