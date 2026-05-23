"""Scale- and fallback-driven sigma helpers for the MINIMA sliding-window
search radius.

The live locate sub-agent always supplies its own σ on every candidate
it returns, so the matcher trusts that value directly. These helpers
fire only on the fallback path inside ``sliding_window_position`` —
when the worker passes ``sigma_m=None`` or a non-positive value:

  - ``sigma_from_scale(scale_ratio)`` — half-diagonal of the printed
    map's real-world extent. The lower bound MINIMA must search to fit
    the planning map's visible area against OS tiles.
  - ``effective_sigma(scale_ratio)`` — ``max(_FALLBACK_SIGMA_M,
    sigma_from_scale(scale_ratio))``. Adds a generic source-side floor
    so a tiny printed-scale (e.g. a 1:500 inset) doesn't collapse the
    search window below candidate→GT drift.

The historical multi-candidate cascade (nominatim / wikidata / gpkg /
etc.) was retired when the live locate sub-agent landed; the
per-source σ table, per-source priority table, and LA-polygon
candidate filter that lived here were removed once the cascade was
gone (no live callers).

Pure module — no I/O, no module-level state.
"""

from __future__ import annotations

import math
from typing import Optional


# Generic source-side σ floor used by ``effective_sigma`` when the
# worker omits σ. The live locate sub-agent's picks always carry a σ
# directly, so this only matters for the rare fallback path.
_FALLBACK_SIGMA_M = 5000


def sigma_from_scale(scale_ratio, page_mm=(297, 210)):
    """Compute MAP-SCALE-DRIVEN search sigma (meters).

    Lower bound on σ — the area MINIMA must search to fit the planning
    map's visible extent against OS tiles.

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


def effective_sigma(scale_ratio: Optional[int]) -> int:
    """Fallback MINIMA search sigma when the worker omits σ.

    Returns ``max(_FALLBACK_SIGMA_M, sigma_from_scale(scale_ratio))`` —
    conservative floor covering both candidate→GT drift and the map's
    visible extent. Fires almost never in practice because the live
    locate sub-agent always supplies σ on its picks.
    """
    return max(_FALLBACK_SIGMA_M, sigma_from_scale(scale_ratio))
