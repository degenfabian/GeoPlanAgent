"""Canonical source-registry for geocoding candidates.

Extracted 2026-05-11 from `tools/matching.py` (which had grown to 1521 LOC).
The registry centralises three complementary lookups keyed by a candidate's
`source` prefix (e.g. ``"nominatim:addr:..."``, ``"gpkg:Camden (Town)"``):

  - `sigma_from_source(name)`     — empirical p95 candidate→GT distance,
    used as the MINIMA search-window radius.
  - `source_priority(name)`       — preference for capping candidate count
    (postcodes/code_point/INSPIRE rank 0; admin/parish rank 9).
  - `_center_specificity(name)`   — geocode precision on the ground
    (rank 0 = housenumber, rank 6 = county). Used by outlier
    cross-validation and broad-area centre filtering.

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


# ── Canonical source priority / specificity tables ──────────────────────────
#
# Consolidated 2026-05-11. Previously three tables had drifted apart:
#   - `_SOURCE_PRIORITY`        (this file, used by `source_priority`)
#   - `_center_specificity`     (this file, deep if/elif ladder)
#   - `_spec`                   (geocoding.py:cross_validate_centers nested fn)
#
# We keep TWO tables because the two consumers want different rankings:
#   - SOURCE_PRIORITY: for capping candidate count (postcodes/code_point
#     rank 0 as "trust most"). Lower = preferred.
#   - SOURCE_SPECIFICITY: for outlier-anchor selection and cross-validation
#     (Nominatim house-number addresses rank 0 as "most precise on the
#     ground"). Lower = more precise. Used by `_center_specificity`.
#
# Both tables live here as the single source of truth. `geocoding._spec`
# now delegates to `_center_specificity`.
#
# SOURCE_PRIORITY empirically derived 2026-05-08: postcodes & grid refs are
# sub-metre accurate; nominatim averages 1km drift; gpkg/wikidata have
# catastrophic tails that the LA filter catches but which still warrant
# lower priority when LA filter is unavailable.
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


# ── Zoomstack-type specificity sets ─────────────────────────────────────────

# Lower rank = more specific. Used to deprioritise broad-area centers when
# street-level anchors are available.
# Ordered by expected geocode precision:
#   0  house-numbered Nominatim address (best)
#   1  Nominatim street+city, grid_refs_centroid, parsed grid ref, postcode
#   2  Zoomstack Town/City/Village/Hamlet/Suburb (point-level settlement)
#   3  Wikidata named settlement (moderate; can be administrative area)
#   4  Zoomstack Suburban Area/Small Settlements (broader neighbourhood)
#   5  Zoomstack Sites/Greenspace/Landform/Water (POI-style, often wrong-sense)
#   6  Zoomstack District/Borough/County/National Park (large admin area)
#   9  unknown (treated as broad)
_HIGH_SPECIFICITY_ZOOMSTACK = {"Town", "City", "Village", "Hamlet", "Suburb"}
_MID_SPECIFICITY_ZOOMSTACK = {"Suburban Area", "Small Settlements"}
_BROAD_ZOOMSTACK = {"District", "County", "Borough", "National Park",
                    "Region", "Country", "Unitary Authority", "Capital",
                    "Metropolitan County"}
_POI_ZOOMSTACK = {"Sites", "Greenspace", "Landform", "Water", "Woodland",
                  "Wetland"}


# Prefix → specificity rank. Lower = more precise on the ground.
# Distinct from SOURCE_PRIORITY (which is for candidate-capping ordering),
# but driven by the same source-prefix taxonomy.
#
# Specificity scale:
#   0  Nominatim address with house number (best)
#   1  Nominatim street+city, grid_refs, postcode, code_point, road centroids
#   2  os_landmark, feature_cluster, filename-derived, Zoomstack
#      Town/City/Village/Hamlet/Suburb
#   3  Wikidata named settlement (moderate; can be administrative area)
#   4  Zoomstack Suburban Area/Small Settlements; bare gpkg; extra_centers
#   5  Zoomstack Sites/Greenspace/Landform/Water (POI-style, wrong-sense risk)
#   6  Zoomstack District/Borough/County/National Park; la_centroid;
#      town_centroid (large admin area)
#   9  unknown (treated as broad)
SOURCE_SPECIFICITY: Dict[str, int] = {
    # Rank 0
    "nominatim:addr": 0,           # synthetic prefix; match handled below
    "nominatim_addr": 0,
    # Rank 1
    "nominatim": 1,
    "grid_refs_centroid": 1, "gridref": 1, "grid_ref": 1,
    "postcode": 1, "code_point": 1,
    "os_road": 1,
    # Rank 2
    "os_landmark": 2,
    "feature_cluster": 2,
    "filename": 2,
    # Rank 3
    "wikidata": 3,
    # Rank 4 (default for gpkg/extra_centers, see lookup below)
    "extra_centers": 4,
    # Rank 6
    "la_centroid": 6,
    "town_centroid": 6,
}


def _center_specificity(name: str) -> int:
    """Map a center's source-encoded name to a specificity rank (lower=better).

    Consults the canonical `SOURCE_SPECIFICITY` table. The only special case
    is gpkg names with a parenthesised Zoomstack type suffix
    (e.g. "gpkg:Camden (Town)"), where the type controls the rank.
    """
    if not isinstance(name, str):
        return 9
    n = name.lower()

    # gpkg with type suffix gets per-type ranking
    if n.startswith("gpkg:") and "(" in name and ")" in name:
        t = name.rsplit("(", 1)[-1].rstrip(")")
        if t in _HIGH_SPECIFICITY_ZOOMSTACK:
            return 2
        if t in _MID_SPECIFICITY_ZOOMSTACK:
            return 4
        if t in _BROAD_ZOOMSTACK:
            return 6
        if t in _POI_ZOOMSTACK:
            return 5
        return 4  # unknown gpkg type → treat as mid
    if n.startswith("gpkg:"):
        return 4  # legacy name without type suffix

    # Address-prefixed Nominatim is special (compound prefix)
    if n.startswith("nominatim:addr:") or n.startswith("nominatim_addr:"):
        return 0

    # Single-prefix lookup
    prefix = n.split(":", 1)[0]
    if prefix in SOURCE_SPECIFICITY:
        return SOURCE_SPECIFICITY[prefix]
    return 9


def filter_centers_by_specificity(centers, anchor_threshold=2,
                                   drop_above=4, min_keep=1):
    """When at least one center has specificity ≤ anchor_threshold (i.e. a
    street-level or grid-ref-quality anchor), drop centers with specificity
    > drop_above (broad-area admin/POI types). Leaves ≥ min_keep centers
    so MINIMA isn't starved.

    Rationale: in v3_flash, the dominant failure mode for IoU=0 accepted
    cases was MINIMA locking onto a broad-area / wrong-sense center
    (gpkg:Presbytery(Greenspace), gpkg:St Albans Church(Sites),
    wikidata:London Borough of Camden) even when a Nominatim street-level
    anchor was in the candidate list. With drop_above=4, ranks 5-6 are
    dropped (Zoomstack Sites/Greenspace/Woodland/Water/Landform and
    District/Borough/County/National Park). Wikidata (rank 3) and
    Zoomstack Town/City/Village/Hamlet/Suburb (rank 2) and Nominatim
    (rank 0-1) survive.
    """
    if len(centers) <= min_keep:
        return centers
    ranked = [(c, _center_specificity(c[0])) for c in centers]
    min_spec = min(s for _, s in ranked)
    if min_spec > anchor_threshold:
        # No high-confidence anchor — keep everything; MINIMA needs every
        # signal it can get.
        return centers
    kept = [c for c, s in ranked if s <= drop_above]
    dropped = [c for c, s in ranked if s > drop_above]
    if len(kept) < min_keep:
        # Filter was too aggressive; add back the least-broad dropped
        # centers until we hit min_keep.
        dropped_ranked = sorted(
            [(c, _center_specificity(c[0])) for c in dropped], key=lambda x: x[1])
        for c, _ in dropped_ranked:
            if len(kept) >= min_keep:
                break
            kept.append(c)
    if dropped:
        dropped_names = ", ".join(c[0] for c in dropped
                                  if c in (dropped[:6]))
        if len(dropped) > 6:
            dropped_names += f" +{len(dropped)-6} more"
        print(f"  Specificity filter: kept {len(kept)}/{len(centers)} "
              f"(dropped broad-area: {dropped_names})")
    return kept
