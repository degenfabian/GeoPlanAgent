"""Multi-axis consistency reward for positioning candidates.

A single match attempt produces an affine_H + tile_info + match_info.
This module evaluates that attempt along several independent consistency
axes — no ground truth, no learned model — and returns both numeric
scores and textual verdicts the LLM agent can reason over.

Axes (each returns a score in [0, 1] plus a 1-line verdict):

  inlier_strength       n_inliers, bucketed
  scale_consistency     does recovered affine scale match reader's stated scale?
                        (avg_scale ≈ 1.0 means the assumed scale was right)
  road_name_agreement   do reader-extracted road names actually appear in
                        the OS road network at the matched window?
  keypoint_spread       are inlier keypoints spread across the map or
                        clumped in one corner? (clumped = local match)

Aggregate:
  overall_score        weighted geometric mean (penalises any axis
                        being terrible)

Format:
  format_for_agent(reward) → multi-line text for the LLM's next prompt.
                              Each line states the axis, score, verdict,
                              and one piece of supporting evidence.
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
    overall_score: float         # in [0, 1]
    summary: str                 # multi-line text for the agent prompt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "axes": {
                name: {"score": ax.score, "verdict": ax.verdict,
                       "evidence": ax.evidence}
                for name, ax in self.axes.items()
            },
            "summary": self.summary,
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
    Returns score = matched / total reader roads. Score 0 with
    reader_road_names empty is treated as "no signal" (score 0.5).
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
        return AxisResult(
            score=0.0,
            verdict="no roads found in OS data within radius — likely wrong area",
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
        v = (f"NO reader roads found in OS at this location "
             f"({n_total} expected, 0 matched) — strong wrong-area signal")

    return AxisResult(
        score=score, verdict=v,
        evidence={"reader_roads": list(reader_road_names),
                  "matched_roads": matched, "radius_m": radius_m})


def axis_keypoint_spread(
    inlier_pts_in_map: Optional[np.ndarray],
    map_shape_hw: Optional[Tuple[int, int]],
) -> AxisResult:
    """Are the inlier keypoints spread across the map area, or clumped?

    Computes bbox-of-inliers / map-bbox area ratio. Spread across most
    of the map → the affine fits the WHOLE map. Clumped in one corner
    → the affine fits just that corner; the rest of the map may be
    misaligned.

    Pass mkpts0 (the keypoints in MAP image coordinates) restricted to
    inliers, plus the map's (h, w).
    """
    if (inlier_pts_in_map is None or len(inlier_pts_in_map) < 3
            or map_shape_hw is None):
        return AxisResult(score=0.5, verdict="insufficient data (no signal)",
                           evidence={})
    h, w = map_shape_hw
    pts = np.asarray(inlier_pts_in_map, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0 or h <= 0 or w <= 0:
        return AxisResult(score=0.5, verdict="insufficient data (no signal)",
                           evidence={})
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    bbox_area = max(1.0, (x_max - x_min) * (y_max - y_min))
    map_area = float(h) * float(w)
    ratio = float(bbox_area / map_area)
    ratio = max(0.0, min(1.0, ratio))
    score = math.sqrt(ratio)  # gentler — 0.25 ratio → 0.5 score

    if ratio >= 0.60:
        v = f"well spread (covers {ratio*100:.0f}% of map area)"
    elif ratio >= 0.30:
        v = f"moderately spread (covers {ratio*100:.0f}% of map area)"
    elif ratio >= 0.10:
        v = f"clumped (only {ratio*100:.0f}% of map area)"
    else:
        v = (f"highly clumped (only {ratio*100:.0f}% of map area — "
             f"likely a local match, not a whole-map alignment)")

    return AxisResult(score=score, verdict=v,
                       evidence={"bbox_to_map_ratio": ratio})


# ── Aggregator ──────────────────────────────────────────────────────────────

# Weights for the geometric-mean aggregate. Each axis is raised to its
# weight; the weights sum to 1.0. A geometric mean penalises any axis
# being very low much more than an arithmetic mean would, which is what
# we want — a candidate with 1900 inliers but zero road agreement is
# suspect, not "great."
DEFAULT_WEIGHTS: Dict[str, float] = {
    "inlier_strength":     0.35,
    "scale_consistency":   0.25,
    "road_name_agreement": 0.30,
    "keypoint_spread":     0.10,
}


def aggregate(axes: Dict[str, AxisResult],
              weights: Optional[Dict[str, float]] = None) -> float:
    weights = weights or DEFAULT_WEIGHTS
    eps = 1e-3  # avoid log(0)
    log_sum = 0.0
    w_sum = 0.0
    for name, w in weights.items():
        if name not in axes:
            continue
        s = max(eps, min(1.0, axes[name].score))
        log_sum += w * math.log(s)
        w_sum += w
    if w_sum <= 0:
        return 0.0
    return float(math.exp(log_sum / w_sum))


# ── Top-level entry point ───────────────────────────────────────────────────

def compute_match_reward(
    *,
    match_info: Dict[str, Any],
    pdf_info: Dict[str, Any],
    inlier_pts_in_map: Optional[np.ndarray] = None,
    map_shape_hw: Optional[Tuple[int, int]] = None,
) -> RewardResult:
    """Compute the multi-axis consistency reward for a single match.

    Args:
        match_info: dict from sliding_window_position with at least
            n_inliers, score, aspect, avg_scale, center_latlon, zoom.
        pdf_info: PDFInfo dict from the reader (scale, road_names, etc.).
        inlier_pts_in_map: optional (N,2) array of inlier keypoint
            coordinates in MAP-image space (for keypoint-spread axis).
        map_shape_hw: optional (h, w) of the map image.
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

    axes["keypoint_spread"] = axis_keypoint_spread(
        inlier_pts_in_map, map_shape_hw)

    overall = aggregate(axes)
    return RewardResult(axes=axes, overall_score=overall,
                          summary=format_for_agent(axes, overall))


# ── Formatting for the agent's prompt ──────────────────────────────────────

def format_for_agent(axes: Dict[str, AxisResult], overall: float) -> str:
    """Render the reward as a structured block of text for the LLM's
    next prompt. Each axis: name, score, 1-line verdict.
    """
    lines = [f"Match-quality evaluation (overall score: {overall:.2f} / 1.00)"]
    order = ["inlier_strength", "scale_consistency", "road_name_agreement",
              "keypoint_spread"]
    for name in order:
        if name not in axes:
            continue
        ax = axes[name]
        lines.append(f"  • {name:<22} {ax.score:>4.2f}  {ax.verdict}")

    # Decision hints
    hints: List[str] = []
    inl = axes.get("inlier_strength")
    sc = axes.get("scale_consistency")
    rn = axes.get("road_name_agreement")
    sp = axes.get("keypoint_spread")
    if inl and inl.score < 0.4 and sc and sc.score < 0.5:
        hints.append("Both inlier count AND scale are weak — reject this candidate.")
    if rn and rn.score == 0.0:
        hints.append("Zero reader road names found in OS data here — strong "
                     "signal this is the wrong area; reject unless other "
                     "axes are unusually strong.")
    if inl and inl.score >= 0.75 and sc and sc.score >= 0.7 and rn and rn.score >= 0.5:
        hints.append("Strong inliers AND consistent scale AND reasonable road "
                     "agreement — this is a high-confidence accept.")
    if sp and sp.score < 0.3 and inl and inl.score >= 0.75:
        hints.append("Inliers are high but clumped — verify visually before "
                     "accepting; the affine may fit only one quadrant.")
    if hints:
        lines.append("")
        lines.append("Decision hints:")
        for h in hints:
            lines.append(f"  - {h}")

    return "\n".join(lines)
