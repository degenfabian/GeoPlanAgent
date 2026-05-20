"""Per-axis consistency reward for positioning candidates.

A single match attempt produces an affine_H + tile_info + match_info.
This module evaluates that attempt along three independent consistency
axes — no ground truth, no learned model — and returns both numeric
scores and textual verdicts the LLM agent can reason over.

Axes (each returns a score in [0, 1] plus a 1-line verdict):

  inlier_strength       n_inliers, bucketed
  scale_consistency     does recovered affine scale match reader's stated scale?
                        (avg_scale ≈ 1.0 means the assumed scale was right)
  road_name_agreement   do reader-extracted road names actually appear in
                        the OS road network at the matched window?

No composite/aggregate score: empirical analysis (2026-05-20) showed
inlier-count alone is a sharp predictor at the n_inliers=50 threshold,
and per-axis rules outperform any composite. keypoint_spread + the
geometric-mean overall_score were removed when nothing in code used
them for decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Axis primitives ─────────────────────────────────────────────────────────

@dataclass
class AxisResult:
    score: float                 # in [0, 1]
    verdict: str                 # 1-line human-readable verdict
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardResult:
    axes: Dict[str, AxisResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axes": {
                name: {"score": ax.score, "verdict": ax.verdict,
                       "evidence": ax.evidence}
                for name, ax in self.axes.items()
            },
        }


# ── Axis implementations ────────────────────────────────────────────────────

def axis_inlier_strength(n_inliers: int, score: float = 0.0) -> AxisResult:
    """Inlier-count quality, bucketed.

    n_inliers is from RANSAC after MINIMA matching. Higher = more
    correspondences support the chosen affine. Score is the sum of
    matcher confidences over inliers (already in match_info).
    """
    n = int(n_inliers or 0)
    if n < 25:
        s, v = 0.10, "very weak (<25 inliers — unreliable affine)"
    elif n < 100:
        s, v = 0.40, "borderline (25-100 inliers — needs verification)"
    elif n < 300:
        s, v = 0.75, "decent (100-300 inliers)"
    elif n < 800:
        s, v = 0.90, "strong (300-800 inliers)"
    else:
        s, v = 1.00, "very strong (800+ inliers)"
    return AxisResult(score=s, verdict=v,
                       evidence={"n_inliers": n, "ransac_score": float(score)})


def axis_scale_consistency(
    avg_scale: float, reader_scale_text: Optional[str] = None,
) -> AxisResult:
    """Does the recovered affine scale agree with the assumed scale?

    avg_scale ≈ 1.0 means the resize-to-tile-pixel-scale was correct,
    which means the assumed map scale was right. Far from 1.0 indicates
    the assumed scale was wrong AND/OR MINIMA found a coincidental match
    at a different scale.

    When the reader did NOT provide a scale, positioning.py iterates
    through a coarse list of common scales [1:1250, 1:2500, 1:5000,
    1:10000, ...] and picks the best. avg_scale up to ~2.0 in that case
    just means the actual scale lies between two grid points — it's NOT
    a mismatch signal. We weaken the penalty accordingly.
    """
    s = float(avg_scale or 0.0)
    if s <= 0:
        return AxisResult(score=0.0, verdict="invalid (avg_scale ≤ 0)",
                           evidence={"avg_scale": s})

    reader_provided = bool(
        reader_scale_text and "not" not in str(reader_scale_text).lower())
    deviation = abs(s - 1.0)

    if reader_provided:
        # Reader gave a scale — deviation IS the signal.
        score = max(0.0, 1.0 - deviation / 0.5)
        if deviation < 0.10:
            v = (f"consistent (avg_scale={s:.2f}, "
                 f"reader said {reader_scale_text!r})")
        elif deviation < 0.25:
            v = (f"mild deviation (avg_scale={s:.2f}, off by "
                 f"{deviation*100:.0f}% from reader's {reader_scale_text!r})")
        elif deviation < 0.50:
            v = (f"significant deviation (avg_scale={s:.2f} — reader said "
                 f"{reader_scale_text!r}, recovered scale disagrees)")
        else:
            v = (f"scale mismatch (avg_scale={s:.2f} — reader's "
                 f"{reader_scale_text!r} appears wrong, OR this is the wrong area)")
    else:
        # No reader scale; common-scales fallback was used. Tolerate up to 2x.
        score = max(0.0, 1.0 - max(0.0, deviation - 1.0) / 1.0)
        if deviation < 1.0:
            v = (f"plausible (avg_scale={s:.2f}, no reader scale — "
                 f"common-scale fallback in expected range)")
        elif deviation < 2.0:
            v = (f"large recovered scale (avg_scale={s:.2f}, no reader scale "
                 f"— could be between common-scale grid points)")
        else:
            v = (f"extreme recovered scale (avg_scale={s:.2f}) — "
                 f"likely a wrong match")

    return AxisResult(score=score, verdict=v,
                       evidence={"avg_scale": s,
                                  "reader_scale": reader_scale_text,
                                  "reader_provided_scale": reader_provided})


def axis_road_name_agreement(
    chosen_lat: float, chosen_lon: float,
    reader_road_names: List[str],
    radius_m: float = 1500.0,
) -> AxisResult:
    """Are the reader-extracted road names present in the OS road network
    at the matched location?

    Uses the offline OS Open Zoomstack GeoPackage (no network calls).

    Three regimes (distinguishes "no data" from "data disagrees"):
      * `reader_road_names` empty           → 0.5 neutral (no signal to test)
      * OS has no roads in radius           → 0.5 neutral (sparse cartography,
        common in rural villages — NOT a wrong-area signal)
      * OS has roads, but none match reader → 0.0 strong wrong-area signal
      * Some / all match                    → matched / total
    """
    n_total = len(reader_road_names or [])
    if n_total == 0:
        return AxisResult(
            score=0.5, verdict="no road names extracted by reader (no signal)",
            evidence={"reader_roads": [], "matched_roads": []})

    # Reuse positioning.py's helpers — they're already there.
    try:
        from tools.matching import _query_gpkg_road_names, _fuzzy_road_match
    except Exception:
        return AxisResult(
            score=0.5, verdict="gpkg helpers unavailable (no signal)",
            evidence={"reader_roads": list(reader_road_names),
                      "matched_roads": []})

    nearby = _query_gpkg_road_names(chosen_lat, chosen_lon, radius_m=radius_m)
    if not nearby:
        # Sparse OS cartography (rural villages often render no road labels
        # at z15). Geometry might be perfectly correct — we just can't test
        # road agreement here. Neutral, NOT a wrong-area signal.
        return AxisResult(
            score=0.5,
            verdict=("no OS roads within radius — sparse cartography "
                     "(rural / unlabelled), neutral signal"),
            evidence={"reader_roads": list(reader_road_names),
                      "matched_roads": [], "radius_m": radius_m})

    matched: List[str] = []
    for rn in reader_road_names:
        if _fuzzy_road_match(rn, nearby):
            matched.append(rn)

    n_matched = len(matched)
    score = n_matched / n_total
    if score >= 0.6:
        v = f"strong agreement ({n_matched}/{n_total} reader roads found in OS)"
    elif score >= 0.3:
        v = f"partial agreement ({n_matched}/{n_total} reader roads found)"
    elif score > 0:
        v = f"weak agreement ({n_matched}/{n_total} reader roads found)"
    else:
        v = (f"OS roads present here but ZERO of {n_total} reader roads "
             f"match — strong wrong-area signal")

    return AxisResult(
        score=score, verdict=v,
        evidence={"reader_roads": list(reader_road_names),
                  "matched_roads": matched, "radius_m": radius_m})


# ── Top-level entry point ───────────────────────────────────────────────────

def compute_match_reward(
    *,
    match_info: Dict[str, Any],
    pdf_info: Dict[str, Any],
) -> RewardResult:
    """Compute the per-axis consistency reward for a single match.

    Args:
        match_info: dict from sliding_window_position with at least
            n_inliers, score, aspect, avg_scale, center_latlon, zoom.
        pdf_info: PDFInfo dict from the reader (scale, road_names, etc.).
    """
    n_inliers = int(match_info.get("n_inliers", 0) or 0)
    ransac_score = float(match_info.get("score", 0.0) or 0.0)
    avg_scale = float(match_info.get("avg_scale", 0.0) or 0.0)
    center_ll = match_info.get("center_latlon")

    axes: Dict[str, AxisResult] = {
        "inlier_strength": axis_inlier_strength(n_inliers, ransac_score),
        "scale_consistency": axis_scale_consistency(
            avg_scale, reader_scale_text=pdf_info.get("scale")),
    }

    # Road-name axis needs a chosen lat/lon.
    if center_ll and len(center_ll) == 2:
        axes["road_name_agreement"] = axis_road_name_agreement(
            float(center_ll[0]), float(center_ll[1]),
            list(pdf_info.get("road_names") or []),
        )
    else:
        axes["road_name_agreement"] = AxisResult(
            score=0.5, verdict="no center_latlon (no signal)", evidence={})

    return RewardResult(axes=axes)
