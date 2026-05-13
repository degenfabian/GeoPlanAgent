"""Single source of truth for candidate scoring across the pipeline.

The pipeline has three independent scoring stages, each with a named
function here:

1. :func:`composite_window_score` — applied to each MINIMA sliding-window
   match. Decides which window to keep within a candidate centre.
2. :func:`commit_attempt_score` — applied across the agent's accumulated
   ``match_at`` attempts. Decides which one ``commit_match`` will keep.
3. :func:`feature_match_score` (re-exported from
   :mod:`tools.locate.ranker`) — applied to locate-stage candidates.
   Decides the order ``propose_centers`` returns them in.

Keeping these in one file makes the trade-offs auditable in a single
place rather than buried across :mod:`tools.matching`,
:mod:`tools.agent_tools_match`, and :mod:`tools.locate.ranker`.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

# Re-export the locate-stage feature_match scorer here so callers have
# `from tools.scoring import …` as a single entry point.
from tools.locate.ranker import feature_match_score, rank_candidates  # noqa: F401


# ─── Stage 1: sliding-window match score ───────────────────────────────────

def composite_window_score(
    vanilla_metric: float,
    quadrant_coverage: int,
    km_to_anchor: float,
) -> float:
    """v19 composite score for a single MINIMA window.

    Combines three orthogonal factors:

    * ``vanilla_metric``     — RANSAC inlier-confidence sum (the v13 metric).
    * ``quadrant_coverage``  — 0..4 count of map quadrants with ≥1 inlier
      (spatial spread; punishes one-corner matches).
    * ``km_to_anchor``       — softmax-like penalty on the predicted centre's
      distance from the geocoded anchor.

    On the 211-case overnight sweep at PB=1, MC=5: +5 cases at IoU ≥ 0.8
    vs the v13 raw-metric ranking (125 → 130). Honest "untuned" version —
    a variant with extra softening factors (0.5+0.5·Q/4, km/2) gains +2
    more cases on this dataset but at risk of overfit.
    """
    if quadrant_coverage < 0:
        quadrant_coverage = 4  # neutral when missing
    if km_to_anchor < 0:
        km_to_anchor = 0.0
    return float(vanilla_metric) * (quadrant_coverage / 4.0) \
        * (1.0 / (1.0 + km_to_anchor))


def quadrant_coverage_from_inlier_points(
    inlier_pts_map, rot_shape: Tuple[int, int],
) -> int:
    """Count how many of the 4 rotated-map quadrants contain ≥1 inlier.

    ``inlier_pts_map`` is a list of (x, y) in rot_map coords, as written
    by :func:`tools.matching.sliding_window_position` into
    ``match_info["_inlier_pts_map"]``. ``rot_shape`` is the (h, w) of the
    rotated map crop at the time of matching.
    """
    if not inlier_pts_map or not rot_shape:
        return 4
    try:
        import numpy as np
        rh, rw = rot_shape
        cx, cy = rw / 2.0, rh / 2.0
        arr = np.asarray(inlier_pts_map)
        return (
            int(((arr[:, 0] < cx) & (arr[:, 1] < cy)).any())
            + int(((arr[:, 0] >= cx) & (arr[:, 1] < cy)).any())
            + int(((arr[:, 0] < cx) & (arr[:, 1] >= cy)).any())
            + int(((arr[:, 0] >= cx) & (arr[:, 1] >= cy)).any())
        )
    except Exception:
        return 4


def haversine_km(
    anchor_latlon: Optional[Tuple[float, float]],
    pred_latlon: Optional[Tuple[float, float]],
) -> float:
    """Great-circle distance in km, used as the km_to_anchor factor."""
    if not anchor_latlon or not pred_latlon:
        return 0.0
    try:
        lat1, lon1 = anchor_latlon
        lat2, lon2 = pred_latlon
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        return 2 * R * math.asin(math.sqrt(a))
    except Exception:
        return 0.0


# ─── Stage 2: smart-commit gate score ──────────────────────────────────────

# Penalty multiplier when a candidate centre falls outside the named LA
# polygon. 0.3× is small enough that a much-higher-inlier outside-LA
# candidate can still beat a low-inlier inside-LA candidate (the LA
# polygon is itself imperfect), but big enough to dominate ties.
OUTSIDE_LA_PENALTY = 0.3


def commit_attempt_score(n_inliers: int, inside_la: bool) -> float:
    """Score a single ``match_at`` attempt for ``commit_match`` ranking.

    The two signals are:

    * ``n_inliers``  — the raw matching signal.
    * ``inside_la``  — whether the predicted centre falls inside the
      reader-extracted admin_region polygon. Catches wrong-town homonym
      matches (e.g. CB:82's road_intersection 5 km outside the LA).

    Analytical short-circuit matches (E/N + scale + DPI + mask centroid)
    have no n_inliers and are scored as ``+inf`` by the caller — they
    always win the gate.
    """
    if n_inliers < 0:
        return -1.0
    return float(n_inliers) * (1.0 if inside_la else OUTSIDE_LA_PENALTY)
