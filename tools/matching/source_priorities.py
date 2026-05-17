"""Canonical source-registry for geocoding candidates.

Extracted 2026-05-11 from `tools/matching.py` (which had grown to 1521 LOC).
The registry centralises two complementary lookups keyed by a candidate's
`source` prefix (e.g. ``"nominatim:addr:..."``, ``"gpkg:Camden (Town)"``):

  - `sigma_from_source(name)`     — empirical p95 candidate→GT distance,
    used as the MINIMA search-window radius.
  - `source_priority(name)`       — preference for capping candidate count
    (postcodes/code_point/INSPIRE rank 0; admin/parish rank 9).

`tools/matching.py` re-exports every public + private name in this module
so existing imports like::

    from tools.matching import (sigma_from_source, candidate_passes_la_filter,
                                    _FILTERABLE_SOURCES, _SOURCE_SIGMA_M)

keep working unmodified.

Constants and functions kept here are PURE — no I/O, no module-level state
beyond the dict/set tables. The only exception is `candidate_passes_la_filter`,
which lazy-imports `tools.verification_checks` to consult OS BoundaryLine.
"""

from __future__ import annotations

import math
from typing import Dict, Optional


# ── Scale-driven sigma ──────────────────────────────────────────────────────


def sigma_from_scale(scale_ratio, page_mm=(297, 210)):
    """Compute MAP-SCALE-DRIVEN search sigma (meters).

    This is the lower bound on sigma — the area MINIMA must search to fit
    the planning map's visible extent against OS tiles. Source-driven
    sigma (`sigma_from_source`) is the OTHER lower bound. Effective sigma
    is the max of the two.

    Args:
        scale_ratio: Map scale denominator (e.g., 2500 for 1:2500). None if unknown.
        page_mm: Paper size (default A4 landscape).

    Returns:
        Sigma in metres = half-diagonal of the printed map's real-world extent.
        For 1:1250 → 226m, 1:2500 → 454m, 1:10000 → 1815m, 1:25000 → 4540m.
        If scale unknown, returns 2500m (a sensible default for site plans).
    """
    if scale_ratio is None:
        return 2500
    diag_mm = math.sqrt(page_mm[0] ** 2 + page_mm[1] ** 2)
    half_diag_m = 0.5 * (diag_mm / 1000.0) * scale_ratio
    return max(150, int(half_diag_m))  # tiny floor — only covers numerical safety


# ── Per-source sigma table ──────────────────────────────────────────────────

# Per-source sigma (metres) calibrated to empirical p90 candidate→GT distance.
# Measured 2026-05-08 against 138 v13 cached candidates.
# Derivation:
#   nominatim:    p90 = 3.77 km → σ = 4000m (covers 90%)
#   wikidata:     p90 = 9.51 km, but the >5km tail is ~all wrong-region; σ = 5000m
#                 with LA-polygon candidate filter rejecting the catastrophic tail
#   gpkg:         p90 = 110 km — unusable without LA filter; σ = 5000m post-filter
#   nominatim cottages need full 4-5km (SSA* failures had Nominatim road anchors
#   1900m from GT); 4000m covers p85 — combine with LA filter to catch outliers.
_SOURCE_SIGMA_M = {
    # High-precision (sub-metre to ~300m): trust the source, tight window.
    # σ from source-paper documented precision (postcodes ~50m, grid refs
    # 100-1000m by digit count).
    "postcode":         100,    # Code-Point Open postcode unit (~50m radius)
    "code_point":       100,
    "grid_ref":         300,    # parsed BNG grid ref
    "agent_websearch": 3000,    # WebSearch → Code-Point. σ=3km covers ~80% of
    "websearch_pc":    3000,    # WebSearch hit (alias of agent_websearch)
    "inspire":          100,    # INSPIRE freehold parcel
    "consensus_centroid": 500,  # cluster of agreeing candidates
    "feature_cluster":  2000,   # location where multiple pdf_info features co-occur

    # Street-address Nominatim (sub-100m precision typical when the address
    # geocodes to a building). Two prefix variants exist in the codebase:
    # "nominatim:addr:..." (legacy v13 path) and "nominatim_addr:..." (v2
    # cascade). Both must be explicit so neither falls through to the bare
    # "nominatim" prefix at 4000m. F4 (v18, 2026-05-10): added after
    # case 69 regressed 0.779→0.000 with 570m drift because σ=5000 default
    # made MINIMA's search window too wide.
    "nominatim_addr":  2500,
    "nominatim:addr":  2500,    # not actually keyed (split[0] → "nominatim"); kept for documentation

    # Medium-precision: AFTER LA filter, the catastrophic tail is gone.
    # σ calibrated to cover edge-of-village cases (rural sites can be at the
    # edge of a named landmark, not the centroid).
    "nominatim":       4000,    # cottage drift can hit 4km
    "os_landmark":     3000,    # named landmark — village edges drift 2-3km
    "os_road":         3000,    # road centroid + site can be 1-2km along the road
    "os_names":        3000,
    "wikidata":        4000,    # post-LA filter, drift bounded
    "photon":          4000,
    "osm":             4000,
    "extra_centers":   4000,
    "gpkg":            4000,    # post-LA filter.
    "carryover":       4000,

    # Broad / fallback (no LA filter or filter not applicable)
    "admin":           8000,
    "parish_centroid": 8000,
    "town_centroid":   8000,
}

# Sources where the LA-polygon filter SHOULD reject candidates outside the
# named admin region. Includes postcode + grid_ref because council-letterhead
# postcodes can resolve to a different LA, and parse errors can produce
# wildly wrong grid_ref coords.
# F3 (v18, 2026-05-10): added road_intersection after CB:82 regressed
# 0.724→0.000 because an extra_terms-supplied road_intersection candidate
# resolved 5km outside the South Bedfordshire LA polygon.
_FILTERABLE_SOURCES = {
    "wikidata", "gpkg", "carryover", "extra_centers",
    "nominatim", "photon", "osm", "os_landmark", "os_road", "os_names",
    "postcode", "code_point", "grid_ref", "grid_refs_centroid",
    "websearch_pc", "agent_websearch", "feature_cluster",
    "road_intersection",
}


def candidate_passes_la_filter(source: str, lat: float, lon: float,
                                admin_region: Optional[str]) -> bool:
    """Return True if candidate is inside the named LA polygon (or filter
    not applicable). Returns True if no admin_region or no LA polygon, so
    this is fail-open."""
    if not admin_region: return True
    src = (source or "").split(":")[0].lower()
    if src not in _FILTERABLE_SOURCES: return True  # exempt high-conf sources
    try:
        from tools.verification_checks import _resolve_la, _load_la_polygons
        _load_la_polygons()
        la = _resolve_la(admin_region)
        if la is None: return True
        from shapely.geometry import Point
        return la.contains(Point(lon, lat))
    except Exception:
        return True


# ── Canonical source priority table ─────────────────────────────────────────
#
# SOURCE_PRIORITY ranks candidate sources for capping the candidate list
# (postcodes/code_point rank 0 = "trust most"; admin/parish rank 9 = "trust
# least"). Lower = preferred.
#
# Empirically derived 2026-05-08: postcodes & grid refs are sub-metre
# accurate; nominatim averages 1km drift; gpkg/wikidata have catastrophic
# tails that the LA filter catches but which still warrant lower priority
# when LA filter is unavailable.
SOURCE_PRIORITY: Dict[str, int] = {
    "postcode": 0, "code_point": 0, "agent_websearch": 0, "inspire": 0,
    "grid_ref": 1, "grid_refs_centroid": 1,
    "consensus_centroid": 2,
    "nominatim": 3, "photon": 3,
    "os_landmark": 4, "os_road": 4, "os_names": 4,
    "wikidata": 5,
    "gpkg": 6,
    "extra_centers": 7,
    "carryover": 8,
    "admin": 9, "parish_centroid": 9, "town_centroid": 9,
}

# Legacy alias retained for callers that imported the underscored name.
_SOURCE_PRIORITY = SOURCE_PRIORITY


def source_priority(source: str) -> int:
    """Lower is better. Used to rank candidates before capping."""
    if not source:
        return 99
    return SOURCE_PRIORITY.get(source.split(":")[0].lower(), 50)


def sigma_from_source(source: str) -> int:
    """Source-driven sigma in metres (empirical p95 candidate→GT distance)."""
    if not source:
        return 5000
    s = source.lower().split(":")[0]
    return _SOURCE_SIGMA_M.get(s, 5000)


def effective_sigma(source: str, scale_ratio: Optional[int]) -> int:
    """Effective MINIMA search sigma = max(source-driven, scale-driven).

    Both lower bounds must be satisfied:
      - Source-driven covers candidate→GT drift
      - Scale-driven covers map's visible-area extent
    """
    return max(sigma_from_source(source), sigma_from_scale(scale_ratio))


