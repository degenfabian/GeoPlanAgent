"""Locate stage — candidate-centre generation for the matching pipeline.

The public entry points are :func:`propose_centers_v2` (the cascade that
produces the candidates ``agent_tools_locate.propose_centers`` returns to the
agent) and :func:`rank_candidates` (post-cascade re-ranker).

The implementation lives in :mod:`tools.locate._core` for now; it will be
split into themed sub-modules (parsing, postcode, roads, district, town,
signals, pipeline, ranker) over subsequent refactor passes. Re-exports here
keep ``from tools.locate import …`` stable across that work.
"""

# ── Schemas ────────────────────────────────────────────────────────────────
from tools.locate._core import (
    Candidate,
    DirectAffine,
    LocateCandidate,
    LocateResult,
    OCRWord,
)

# ── Public cascade entry points ─────────────────────────────────────────────
from tools.locate._core import (
    propose_centers_v2,
    rank_candidates,
    locate_map,           # v13 legacy entry — kept for overnight scripts
    town_centroid,
    feature_cluster_locate,
    feature_match_score,
)

# ── Public helpers used by callers outside this package ────────────────────
from tools.locate._core import (
    # OCR helpers — exposed for offline analysis scripts
    extract_grid_refs_from_ocr,
    extract_scale_from_ocr,
    # Affine solver — used by analytical-match path
    solve_affine_from_grid_ticks,
    direct_affine_centroid,
    # Road utilities — used by overnight reproducibility scripts
    find_road_intersections,
    # Postcode helpers
    _lookup_postcode,
    # District lookups
    _district_info,
)

__all__ = [
    "Candidate", "DirectAffine", "LocateCandidate", "LocateResult", "OCRWord",
    "propose_centers_v2", "rank_candidates", "locate_map",
    "town_centroid", "feature_cluster_locate", "feature_match_score",
    "extract_grid_refs_from_ocr", "extract_scale_from_ocr",
    "solve_affine_from_grid_ticks", "direct_affine_centroid",
    "find_road_intersections",
    "_lookup_postcode", "_district_info",
]
