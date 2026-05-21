"""Single source of truth for match-stage candidate scoring.

Two stages, each with a named function:

1. :func:`composite_window_score` — applied to each MINIMA sliding-window
   match. Decides which window to keep within a candidate centre.
2. :func:`commit_attempt_score` — applied across the agent's accumulated
   ``match_at`` attempts. Decides which one ``commit_match`` will keep.

Locate-stage scoring lives inside :mod:`tools.agent.locate_agent` (the
live LLM-locate sub-agent picks one center directly).
"""

from __future__ import annotations

from typing import Tuple


# ─── Stage 1: sliding-window match score ───────────────────────────────────

def composite_window_score(
    vanilla_metric: float,
    quadrant_coverage: int,
) -> float:
    """Composite score for a single MINIMA window.

    Combines two factors:

    * ``vanilla_metric``     — RANSAC inlier-confidence sum (the v13 metric).
    * ``quadrant_coverage``  — 0..4 count of map quadrants with ≥1 inlier
      (spatial spread; punishes one-corner matches).
    """
    if quadrant_coverage < 0:
        quadrant_coverage = 4  # neutral when missing
    return float(vanilla_metric) * (quadrant_coverage / 4.0)


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
    """
    if n_inliers < 0:
        return -1.0
    penalty = 1.0 if inside_la else OUTSIDE_LA_PENALTY
    return float(n_inliers) * penalty
