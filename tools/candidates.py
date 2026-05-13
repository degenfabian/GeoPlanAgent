"""Backward-compatibility shim — re-exports the locate package.

The candidate-generation code lived in this file historically; it was moved
to the :mod:`tools.locate` package on 2026-05-12. This shim keeps the old
``from tools.candidates import propose_centers_v2`` imports working.
"""

# Star-import the package so every public symbol the module used to expose
# (including the private helpers that overnight/ scripts grab) stays accessible.
from tools.locate._core import *  # noqa: F401, F403
from tools.locate._core import (  # noqa: F401
    # Explicit re-exports for symbols that don't follow __all__ conventions
    Candidate,
    DirectAffine,
    LocateCandidate,
    LocateResult,
    OCRWord,
    propose_centers_v2,
    rank_candidates,
    locate_map,
    town_centroid,
    feature_cluster_locate,
    feature_match_score,
    extract_grid_refs_from_ocr,
    extract_scale_from_ocr,
    solve_affine_from_grid_ticks,
    direct_affine_centroid,
    find_road_intersections,
    # Private helpers (kept exported for overnight scripts and offline tests)
    _lookup_postcode,
    _normalize_postcode,
    _is_full_postcode,
    _postcode_area,
    _area_centroid,
)
