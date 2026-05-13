"""tools/critic.py — Deterministic boundary verifier (v18+).

Runs AFTER the worker agent has returned an "accepted" BoundaryOutcome.
Computes a deterministic verdict via `tools.verification_checks.verification_score`
(inlier_scatter, building_overlap, multi_zoom_coherence, etc.) and labels the
result as `approve` or `flag_low_confidence`. Never nullifies the GeoJSON —
flag_low_confidence is the strongest negative decision and only labels the
result for downstream logging.

Why deterministic-only as of v18: the v17 effectiveness audit
(`overnight/v17_critic_effectiveness.md`) showed the legacy LLM critic
rubber-stamped 95% of cases, and its 3.3% interventions had mean IoU 0.256
vs 0.750 on rubber-stamps — i.e. interventions correlated with failures
and didn't rescue any of them. The LLM critic was removed entirely in this
commit; the dispatcher path now goes straight to `run_deterministic_critic`.

Public API:
    - run_critic_loop(state, worker_agent, worker_result, model, sam3,
                       minima_matcher, verbose, max_super, max_inner) -> dict
    - run_deterministic_critic(state, worker_result, verbose) -> dict

Both return the same dict shape so downstream tooling (benchmark_runner,
watch scripts) keeps working unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from tools.agent import AgentState  # noqa: F401


# ── Visual helpers (kept for offline panel-rendering scripts) ───────────────

def _resize_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / h
    return cv2.resize(img, (max(1, int(w * scale)), target_h))


def _add_label_bar(img: np.ndarray, label: str) -> np.ndarray:
    """Prepend a 35px white bar with the label in black."""
    h, w = img.shape[:2]
    bar = np.full((35, w, 3), 255, dtype=np.uint8)
    cv2.putText(bar, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


# ── Deterministic critic ────────────────────────────────────────────────────

def run_deterministic_critic(
    state: Any,
    worker_result: Any,
    verbose: bool = True,
) -> Dict[str, Any]:
    """v18+ default critic: deterministic verdict via verification_score.

    Three hard gates via `tools.verification_checks`: inlier_scatter,
    building_overlap, multi_zoom_coherence. Empirically, the LLM critic in
    v17 rubber-stamped 95% of cases and the 5% it intervened on had MEAN
    IoU 0.256 vs 0.750 on rubber-stamps — i.e. interventions correlated
    with failures and didn't rescue any of them. Per-case audit memory
    file: `overnight/v17_critic_effectiveness.md`.

    The deterministic critic still produces the same `iterations` log
    shape so downstream tooling (benchmark_runner, watch scripts) keeps
    working unchanged. The single iteration's `decision` is one of
    'approve' or 'flag_low_confidence'. No retries are attempted — when
    a hard gate fails, we record the diagnosis but accept the GeoJSON for
    partial-IoU credit (consistent with the LLM critic's "never nullify"
    rule).

    Saves: ~$0.04 per case in LLM tokens, ~10 s per case in wall time
    compared to the now-removed LLM critic super-loop.
    """
    from tools.verification_checks import verification_score
    from shapely.geometry import shape

    iterations: List[Dict[str, Any]] = []
    geojson = (state.current_result or {}).get("geojson") if state.current_result else None
    match_info = (state.current_result or {}).get("match_info") if state.current_result else None
    pdf_info = state.pdf_info or {}

    # If no geojson, nothing to critique
    if geojson is None:
        if verbose:
            print("  det-critic: no geojson, skipping")
        return {
            "iterations": [],
            "final_decision": None,
            "changed_mask": False,
            "tokens_used": {"request": 0, "response": 0},
            "panel_iter0": None,
            "applied_rotation_deg": None,
        }

    try:
        pred_geom = shape(geojson.get("geometry") or geojson)
        if not pred_geom.is_valid:
            pred_geom = pred_geom.buffer(0)
    except Exception as e:
        if verbose:
            print(f"  det-critic: bad geojson ({e!s:.50}), skipping")
        return {
            "iterations": [],
            "final_decision": None,
            "changed_mask": False,
            "tokens_used": {"request": 0, "response": 0},
            "panel_iter0": None,
            "applied_rotation_deg": None,
        }

    score = verification_score(pdf_info, pred_geom, match_info)
    diag = score["diagnosis"]
    hard_failed = score.get("hard_gate_failed", False)
    if hard_failed:
        decision_kind = "flag_low_confidence"
    elif score["score"] < 0.5:
        decision_kind = "flag_low_confidence"
    else:
        decision_kind = "approve"

    # Single-iteration log
    iterations.append({
        "iter_idx": 0,
        "decision": decision_kind,
        "diagnosis": diag,
        "score": score["score"],
        "checks": score["checks"],
        "fix_applied": "",
    })

    if verbose:
        triggered = [n for n, c in score["checks"].items()
                     if c.get("confidence", 0.5) < 0.3 and c.get("reason")]
        print(f"  det-critic: {decision_kind} (diag={diag}, score={score['score']:.2f}, "
              f"triggered={triggered})")

    return {
        "iterations": iterations,
        "final_decision": decision_kind,
        "changed_mask": False,  # deterministic critic never modifies state
        "tokens_used": {"request": 0, "response": 0},
        "panel_iter0": None,
        "applied_rotation_deg": None,
    }


def run_critic_loop(
    state: Any,
    worker_agent: Any,
    worker_result: Any,
    model: Any,
    sam3: Dict[str, Any],
    minima_matcher: Any,
    verbose: bool = True,
    max_super: int = 2,
    max_inner: int = 2,
) -> Dict[str, Any]:
    """v18+: only the deterministic critic ships. The legacy LLM critic
    was removed after the v17 effectiveness audit showed it was net
    neutral-to-negative (95% rubber-stamp, 3.3% intervention with mean
    IoU 0.256 vs 0.750 on rubber-stamps).

    The `worker_agent`, `model`, `sam3`, `minima_matcher`, `max_super`,
    and `max_inner` parameters are preserved in the signature for caller
    compatibility but are no longer used.
    """
    return run_deterministic_critic(state, worker_result, verbose=verbose)
