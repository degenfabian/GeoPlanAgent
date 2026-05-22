"""Per-axis consistency reward for positioning candidates.

A single match attempt produces an affine_H + tile_info + match_info.
This module evaluates that attempt along two independent consistency
axes — no ground truth, no learned model — and returns both numeric
scores and textual verdicts the LLM agent can reason over.

Axes (each returns a score in [0, 1] plus a 1-line verdict):

  scale_consistency     does recovered affine scale match reader's stated scale?
                        (avg_scale ≈ 1.0 means the assumed scale was right)
  road_name_agreement   do reader-extracted road names actually appear in
                        the OS road network at the matched window?

Raw n_inliers (from match_info) is itself the primary RANSAC strength
signal — the worker and critic read it directly, and the smart-commit
gate uses it as the base factor before multiplying in scale_consistency
and quadrant coverage. We no longer wrap it in a redundant
"inlier_strength" axis whose score nothing in the pipeline consumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# A reader-provided scale is signaled by a real "1:N" / "1/N" pattern in
# the extracted text — not by the absence of a few stop-words. The old
# substring check "not in reader_scale_text.lower()" mis-classified valid
# scales like "1:2500 (note: ...)" or "1:2500 cannot be guaranteed" as
# "no reader scale" because "not" appears as a substring.
_SCALE_PATTERN_RE = re.compile(r"\b1\s*[:/]\s*\d")


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

def axis_scale_consistency(
    avg_scale: float, reader_scale_text: Optional[str] = None,
) -> AxisResult:
    """Does the recovered affine scale agree with the assumed scale?

    avg_scale ≈ 1.0 means the resize-to-tile-pixel-scale was correct,
    which means the assumed map scale was right. Far from 1.0 indicates
    the assumed scale was wrong AND/OR MINIMA found a coincidental match
    at a different scale.

    Score is ``min(s, 1/s) ** 2`` — symmetric about identity (treats
    "stretched 31% more" and "compressed by 24%" as equally suspicious),
    returns 1.0 at s=1, smoothly decays toward 0. The squaring is the
    only knob (``p=2``); it sharpens the slope near identity enough that
    a 10–30% deviation produces a decisive ranking difference at
    smart-commit time.
    """
    s = float(avg_scale or 0.0)
    if s <= 0:
        return AxisResult(score=0.0, verdict="invalid (avg_scale ≤ 0)",
                           evidence={"avg_scale": s})

    reader_provided = bool(
        reader_scale_text
        and _SCALE_PATTERN_RE.search(str(reader_scale_text)))
    score = min(s, 1.0 / s) ** 2

    if reader_provided:
        v = (f"scale_consistency={score:.2f} (avg_scale={s:.3f}, "
             f"reader said {reader_scale_text!r})")
    else:
        v = (f"scale_consistency={score:.2f} (avg_scale={s:.3f}, "
             f"no reader scale)")

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
            score=0.5,
            verdict=("no OS roads within radius — sparse cartography "
                     "(rural / unlabelled), neutral signal"),
            evidence={"reader_roads": list(reader_road_names),
                      "matched_roads": [], "radius_m": radius_m})

    matched: List[str] = []
    for rn in reader_road_names:
        if _fuzzy_road_match(rn, nearby):
            matched.append(rn)

    # Score is the raw match ratio; the verdict is just human-readable
    # context. Resist the urge to add tier thresholds here — the critic
    # reads the score directly, and any "strong/partial/weak" labels would
    # be arbitrary cutoffs masquerading as principled signal.
    n_matched = len(matched)
    score = n_matched / n_total
    if score == 0:
        v = (f"OS roads present here but ZERO of {n_total} reader roads "
             f"match — possible wrong-area signal (trust strong inliers "
             f"over this)")
    else:
        v = f"{n_matched}/{n_total} reader roads found in OS"

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
            n_inliers, avg_scale, center_latlon.
        pdf_info: PDFInfo dict from the reader (scale, road_names, …).
    """
    avg_scale = float(match_info.get("avg_scale", 0.0) or 0.0)
    center_ll = match_info.get("center_latlon")

    axes: Dict[str, AxisResult] = {
        "scale_consistency": axis_scale_consistency(
            avg_scale, reader_scale_text=pdf_info.get("scale")),
    }

    if center_ll and len(center_ll) == 2:
        axes["road_name_agreement"] = axis_road_name_agreement(
            float(center_ll[0]), float(center_ll[1]),
            list(pdf_info.get("road_names") or []),
        )
    else:
        axes["road_name_agreement"] = AxisResult(
            score=0.5, verdict="no center_latlon (no signal)", evidence={})

    return RewardResult(axes=axes)
