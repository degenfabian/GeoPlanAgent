"""Single source of truth for match-stage candidate scoring.

:func:`composite_window_score` is applied to each MINIMA sliding-window
match to decide which window to keep within a candidate centre.

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

    * ``vanilla_metric``     — RANSAC inlier count from the MINIMA window
      match (the per-window primary metric; earlier v13 builds used a
      confidence-weighted sum here, switched to raw count later).
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


