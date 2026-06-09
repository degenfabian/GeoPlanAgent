"""Match-stage candidate scoring.

composite_window_score() ranks MINIMA sliding-window matches within a
candidate centre. Locate-stage scoring is separate: the locate sub-agent
picks one centre directly (see tools.agent.locate_agent).
"""

from __future__ import annotations

import logging
from typing import Tuple

log = logging.getLogger(__name__)


def composite_window_score(vanilla_metric: float,
                           quadrant_coverage: int) -> float:
    """RANSAC inlier count weighted by spatial spread of the inliers.

    quadrant_coverage counts map quadrants with at least one inlier
    (0..4), which penalises matches whose support sits in one corner.
    """
    if quadrant_coverage < 0:
        quadrant_coverage = 4  # unknown coverage shouldn't penalise
    return float(vanilla_metric) * (quadrant_coverage / 4.0)


def quadrant_coverage_from_inlier_points(
    inlier_pts_map, rot_shape: Tuple[int, int],
) -> int:
    """How many of the rotated map's 4 quadrants contain an inlier.

    inlier_pts_map is the list of (x, y) points that
    tools.matching.sliding_window_position stores in
    match_info["_inlier_pts_map"]; rot_shape is the (h, w) of the rotated
    map crop at match time.
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
        log.warning("quadrant coverage failed for %d pts, shape %s; "
                    "treating as full coverage", len(inlier_pts_map), rot_shape,
                    exc_info=True)
        return 4
