"""Phase 3 critic — visual review + structured directive, optionally
rehanded to the worker.

`run_critic_agent` runs a separate Gemini-Flash agent that sees a 2-panel
image (planning map + SAM mask | OS render + projected polygon) plus
deterministic-verification metrics, and emits a structured CriticDirective
(approve / retry_extract_bbox / retry_match_at). retry-* directives are
rehanded to the worker; the worker is prompted to comply via its CRITIC
DIRECTIVE protocol.

`run_critic_loop` is the public entry point; benchmark_runner and watch
scripts call it.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent

from tools.agent._model import resolve_model


# ── Critic output schema (structured directive) ────────────────────────────

class CriticDirective(BaseModel):
    """Structured directive the worker is forced to execute."""
    diagnosis: str = Field(
        description="2-3 sentences on what the critic SEES in the panel. "
                    "Specifically: can you trace named roads / settlement "
                    "shape between the planning map (left) and OS render "
                    "(right)? Does the projected polygon align with road "
                    "junctions or building blocks in the OS render? "
                    "Refer to concrete visual features, not vibes."
    )
    action: str = Field(
        description="One of: approve | retry_extract_bbox | retry_match_at. "
                    "Choose 'approve' when the panel shows clear road/feature "
                    "correspondence between map and OS render AND the polygon "
                    "outline sits where the boundary appears on the map. "
                    "Otherwise pick the retry that addresses the specific "
                    "failure mode."
    )
    bbox: Optional[List[int]] = Field(
        default=None,
        description="Required when action='retry_extract_bbox'. Format "
                    "[x1, y1, x2, y2] in PLANNING-MAP pixel coordinates "
                    "(origin top-left). Tightly bound the area where the "
                    "boundary appears on the map; SAM3 will be re-run with "
                    "this bbox to focus its segmentation."
    )
    center_idx: Optional[int] = Field(
        default=None,
        description="Required when action='retry_match_at'. Zero-based index "
                    "into the CENTRES list shown in the metrics block. Pick "
                    "an untried centre that better matches the planning map's "
                    "named features."
    )
    reason: str = Field(
        description="≥30 chars. Specifically reference the visual signal or "
                    "metric that drove this decision (e.g. 'inlier_scatter "
                    "check failed AND mask covers a 32% blob that includes "
                    "title-block text'). Do NOT say 'looks fine' or "
                    "'reasonable'."
    )


CRITIC_INSTRUCTIONS = """\
You are reviewing one extracted planning-boundary prediction. Your job is to
decide whether the agent's polygon prediction matches the planning map's
intended boundary, and if not, dispatch ONE concrete retry action that a
downstream worker will execute.

WHAT YOU SEE
- A 2-panel image. LEFT: the planning map with the agent's SAM mask
  overlaid in translucent green. RIGHT: the OS map render at the matched
  location with the projected polygon outlined in red. Both panels are at
  the same orientation and roughly the same scale.
- A structured metrics block (match scores, mask coverage, deterministic
  verification check results, a numbered list of CENTRES the locator
  proposed, and PDF hints from the document).

WHAT "GOOD" LOOKS LIKE
Trace named roads or recognisable settlement features between the two
panels. The same road in the planning map should be findable in the OS
render at roughly the same position. The boundary in the planning map
(usually a coloured line — red, green, dashed, hatched, etc.) should sit
where the projected red outline sits in the OS render. Building blocks,
junctions, road bends, water bodies should line up.

Additional positive signals:
- n_inliers ≥ 50 with aspect close to 1.0 (square-ish affine, no shear)
- Deterministic overall_score ≥ 0.6
The scale_factor is shown as a raw number — interpret it from context
(planning-map scale, what the panel shows), not as a hard threshold.

WHAT "BAD" LOOKS LIKE
- No road or settlement correspondence between left and right panel.
  Planning map shows urban streets; OS render shows farmland or a totally
  different street pattern.
- The SAM mask covers something that isn't a boundary: a title block,
  legend, scale bar, north arrow, or wide region of text/whitespace.
- The planning map has multiple plausible candidate polygons and the
  mask picked the wrong one.
- The projected polygon (red outline in OS render) lands well outside the
  planning map's bounded area, or its shape doesn't match the planning
  map's boundary shape.
- Deterministic verification flagged hard gates (look at "failed checks"
  block) — esp. inlier_scatter BAD, multi_zoom_coherence BAD, la_boundary
  BAD.

ACTIONS (pick exactly one)
- approve
    Use when the panel shows clear road/feature correspondence AND the
    polygon outline aligns with the boundary as drawn on the map. Required:
    a `diagnosis` field naming the specific features you matched between
    panels.

- retry_extract_bbox
    Use when the SAM mask covers the wrong region of the planning map
    (e.g. it grabbed text / a legend block / a whole-map blob), BUT the
    matching looks plausible. The worker will re-run SAM3 segmentation on
    the planning map with your bbox to focus its attention.
    Required: `bbox=[x1, y1, x2, y2]` in planning-map pixel coordinates.
    Coordinates: origin (0, 0) is top-left of the planning map. (x1, y1)
    is the upper-left of your target region; (x2, y2) is the lower-right.

- retry_match_at
    Use when the OS render simply doesn't correspond to the planning map
    geometry — different roads, different settlement pattern. The worker
    will re-run matching at a different centre.
    Required: `center_idx` = 0-based index of one of the UNTRIED centres
    from the CENTRES list.

OUTPUT
A single structured response with: diagnosis, action, optional bbox /
center_idx, and reason. The reason MUST cite a concrete observation —
either a specific feature you matched (or failed to match) between the
panels, or a specific metric value. Never use vague language ('looks
fine', 'reasonable', 'I think').
"""


_critic_agent: Optional[Agent] = None
_critic_agent_model: Optional[str] = None


def _ensure_agent(model_name: str) -> Agent:
    """Build (or return cached) critic Agent. Rebuilds if the model_name
    changes between calls — important when a benchmark sweep tests
    different worker/critic models."""
    global _critic_agent, _critic_agent_model
    if _critic_agent is not None and _critic_agent_model == model_name:
        return _critic_agent
    _critic_agent = Agent[None, CriticDirective](
        model=resolve_model(model_name),
        output_type=CriticDirective,
        instructions=CRITIC_INSTRUCTIONS,
    )
    _critic_agent_model = model_name
    return _critic_agent


# ── State snapshot helper ──────────────────────────────────────────────────

def _committed_primary_view(state) -> tuple:
    """Return (page, map_img, mask) for the worker's committed primary
    group, or (None, None, None) when nothing is committed."""
    from tools.agent.state import committed_primary_page
    page = committed_primary_page(state)
    if page is None:
        return None, None, None
    return page, state.rendered_pages.get(page), state.sam_masks_by_page.get(page)


def _snapshot_state(state) -> Dict[str, Any]:
    _, _, mask = _committed_primary_view(state)
    cr = state.current_result or {}
    aff = cr.get("affine_H")
    return {
        "mask": (mask.copy() if isinstance(mask, np.ndarray) else None),
        "geojson": cr.get("geojson"),
        "affine_H": (aff.copy() if isinstance(aff, np.ndarray) else aff),
        "tile_info": cr.get("tile_info"),
        "match_info": (cr.get("match_info") or {}).copy() if cr.get("match_info") else None,
    }


# ── Visual panel ───────────────────────────────────────────────────────────

def build_critic_panel(state) -> Optional[np.ndarray]:
    """LEFT: planning map + current SAM mask (translucent green).
    RIGHT: OS tile canvas + projected polygon outline (red).
    Both side-by-side, labelled. None if state is missing essentials."""
    _, map_img, mask = _committed_primary_view(state)
    cr = state.current_result or {}
    tile_info = cr.get("tile_info") or {}
    affine_H = cr.get("affine_H")
    if map_img is None:
        return None
    target_h = 600

    left = map_img.copy()
    if mask is not None and mask.sum() > 0:
        mb = (mask > 0).astype(np.uint8)
        if mb.shape != left.shape[:2]:
            mb = cv2.resize(mb, (left.shape[1], left.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        layer = left.copy()
        layer[mb > 0] = (0, 255, 0)
        left = cv2.addWeighted(left, 0.55, layer, 0.45, 0)
    h_l, w_l = left.shape[:2]
    left_resized = cv2.resize(left, (int(w_l * target_h / h_l), target_h))

    tile_img = tile_info.get("image") if isinstance(tile_info, dict) else None
    right_resized = None
    if tile_img is not None and affine_H is not None:
        if tile_img.shape[2] == 3 and tile_info.get("_was_rgb", True):
            tile_bgr = cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR)
        else:
            tile_bgr = tile_img.copy()
        if mask is not None and mask.sum() > 0:
            mb = (mask > 0).astype(np.uint8)
            contours, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) < 3:
                    continue
                pts = cnt.reshape(-1, 2).astype(np.float32)
                pts_h = np.concatenate([pts, np.ones((len(pts), 1),
                                                       dtype=np.float32)], axis=1)
                proj = pts_h @ affine_H.T
                proj_int = np.round(proj).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(tile_bgr, [proj_int], isClosed=True,
                              color=(0, 0, 255),
                              thickness=max(2, tile_bgr.shape[0] // 250))
        h_r, w_r = tile_bgr.shape[:2]
        right_resized = cv2.resize(tile_bgr, (int(w_r * target_h / h_r), target_h))

    def _label(img, text):
        bar = np.full((36, img.shape[1], 3), 25, dtype=np.uint8)
        cv2.putText(bar, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([bar, img])

    left_lab = _label(left_resized, "PLANNING MAP + SAM mask (green)")
    if right_resized is not None:
        right_lab = _label(right_resized,
                            f"OS RENDER @ z={tile_info.get('zoom','?')} "
                            f"+ projected polygon (red)")
        panel = np.hstack([left_lab, right_lab])
    else:
        panel = left_lab
    if panel.shape[1] > 2000:
        s = 2000 / panel.shape[1]
        panel = cv2.resize(panel, (2000, int(panel.shape[0] * s)))
    return panel


# ── Metrics text block ─────────────────────────────────────────────────────

def format_metrics_text(state, det_score: Dict[str, Any]) -> str:
    cr = state.current_result or {}
    mi = cr.get("match_info") or {}
    _, _, mask = _committed_primary_view(state)
    pi = state.pdf_info or {}
    mask_pct = (float((mask > 0).mean()) * 100
                if isinstance(mask, np.ndarray) and mask.size else 0.0)

    lines = ["=== MATCH METRICS ==="]
    cll = mi.get("center_latlon") or ["?", "?"]
    lines.append(f"  center: {mi.get('center', '?')} @ ({cll[0]}, {cll[1]})")
    lines.append(
        f"  n_inliers={mi.get('n_inliers', 0)}  score={mi.get('score', 0)}  "
        f"aspect={mi.get('aspect', 0)}  scale_factor={mi.get('scale_factor', 0)}  "
        f"zoom={mi.get('zoom', '?')}  rotation={mi.get('rotation', 0)}°")
    lines.append(f"  mask coverage: {mask_pct:.2f}% of planning-map image")

    lines.append("\n=== DETERMINISTIC VERIFICATION ===")
    lines.append(
        f"  overall_score: {det_score.get('score', 0.5):.2f}  "
        f"diagnosis: {det_score.get('diagnosis', '?')}")
    checks = det_score.get("checks") or {}
    triggered = [(n, c.get("confidence", 0.5), c.get("reason", "")[:80])
                 for n, c in checks.items()
                 if c.get("confidence", 0.5) < 0.5 and c.get("reason")]
    if triggered:
        lines.append("  failed checks:")
        for name, conf, reason in triggered:
            lines.append(f"    {name}={conf:.2f}: {reason}")

    lines.append("\n=== CENTRES (for retry_match_at center_idx) ===")
    for i, p in enumerate((state.proposed_centers or [])[:8]):
        name = p.get("name") or p.get("source") or f"candidate_{i}"
        sigma = p.get("sigma_m", "?")
        lines.append(f"  [{i}] {name}  σ={sigma}m")

    lines.append("\n=== PDF HINTS ===")
    lines.append(
        f"  admin_region: {pi.get('admin_region', '?')}  "
        f"scale: {pi.get('scale', '?')}")
    if pi.get("postcodes"):
        lines.append(f"  postcodes: {pi.get('postcodes')[:3]}")
    if pi.get("road_names"):
        lines.append(f"  road_names: {pi.get('road_names')[:5]}")
    return "\n".join(lines)


# ── LLM critic: one iteration ──────────────────────────────────────────────

def _run_critic_once(state, model_name: str):
    """One critic LLM call. Returns (directive, panel, det_score, wall, in_tokens, out_tokens)."""
    from tools.verification_checks import verification_score
    from shapely.geometry import shape

    panel = build_critic_panel(state)
    try:
        pred_geom = shape(state.current_result.get("geojson", {}).get("geometry") or {})
        if not pred_geom.is_valid:
            pred_geom = pred_geom.buffer(0)
        det = verification_score(state.pdf_info or {}, pred_geom,
                                   state.current_result.get("match_info"))
    except Exception:
        det = {"score": 0.5, "diagnosis": "?", "checks": {}, "hard_gate_failed": False}
    metrics_text = format_metrics_text(state, det)

    if panel is not None:
        _, buf = cv2.imencode(".png", panel)
        user_input = [
            metrics_text,
            BinaryContent(data=buf.tobytes(), media_type="image/png"),
        ]
    else:
        user_input = [metrics_text]

    agent = _ensure_agent(model_name)
    in_tokens = 0
    out_tokens = 0
    t0 = time.time()
    from tools.agent.state import _run_sync_with_retry
    try:
        result = _run_sync_with_retry(agent, user_input, label="critic")
        directive = result.output
        try:
            usage = result.usage()
            in_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            out_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        except Exception:
            pass
    except Exception as e:
        directive = CriticDirective(
            diagnosis="LLM call failed",
            action="approve",
            reason=f"critic failure: {e!s:.80}",
        )
    wall = time.time() - t0
    return directive, panel, det, wall, in_tokens, out_tokens


def _rehand_to_worker(state, worker_result, directive: CriticDirective,
                       verbose: bool = True):
    """Re-invoke the worker with a CRITIC DIRECTIVE message instructing it
    to comply. Mutates state.sam_masks_by_page and
    state.current_result["geojson"] via the worker's tools. Returns the
    new pydantic-ai result or the prior worker_result on failure."""
    from tools.agent.worker_agent import _agent

    action = directive.action
    bbox = directive.bbox
    center_idx = directive.center_idx

    if action == "retry_extract_bbox":
        if not (isinstance(bbox, list) and len(bbox) == 4):
            if verbose:
                print(f"  critic rehand: bad bbox {bbox}, skipping")
            return worker_result
        instruction = (
            f"CRITIC DIRECTIVE — you MUST comply.\n"
            f"Diagnosis: {directive.diagnosis}\n"
            f"Action: re-run extract_boundary with bbox=[{bbox[0]}, {bbox[1]}, "
            f"{bbox[2]}, {bbox[3]}] in planning-map pixel coordinates, then "
            f"project_boundary, then submit a new BoundaryOutcome.\n"
            f"Reason: {directive.reason}"
        )
    elif action == "retry_match_at":
        if center_idx is None:
            if verbose:
                print(f"  critic rehand: missing center_idx, skipping")
            return worker_result
        proposed = state.proposed_centers or []
        if center_idx < 0 or center_idx >= len(proposed):
            if verbose:
                print(f"  critic rehand: center_idx {center_idx} out of range")
            return worker_result
        cand = proposed[center_idx]
        instruction = (
            f"CRITIC DIRECTIVE — you MUST comply.\n"
            f"Diagnosis: {directive.diagnosis}\n"
            f"Action: re-run match_at(name={cand.get('name')!r}, "
            f"lat={cand.get('lat')}, lon={cand.get('lon')}, "
            f"sigma_m={cand.get('sigma_m', 2500)}). If the new score beats "
            f"the current committed match, commit_match the new candidate "
            f"and re-run extract_boundary + project_boundary. Submit a new "
            f"BoundaryOutcome.\n"
            f"Reason: {directive.reason}"
        )
    else:
        return worker_result

    if verbose:
        print(f"  critic rehand: re-invoking worker with {action}")
    try:
        history = worker_result.all_messages() if worker_result is not None else None
        sub_result = _agent.run_sync(instruction, deps=state,
                                       message_history=history)
        return sub_result
    except Exception as e:
        if verbose:
            print(f"  critic rehand: worker re-invoke failed: {e!s:.120}")
        return worker_result


# ── LLM critic: outer loop ─────────────────────────────────────────────────

def run_critic_agent(
    state: Any,
    worker_result: Any,
    model_name: str,
    verbose: bool = True,
    max_iters: int = 2,
) -> Dict[str, Any]:
    """LLM critic + worker-rehand loop. Up to `max_iters` outer iterations
    of (critic → maybe rehand → critic re-check → ...).

    Args:
        state: AgentState carrying the worker's commit + tile_info.
        worker_result: pydantic-ai result from the worker (used for
            message_history when re-handing).
        model_name: OpenRouter model identifier (full ID, no alias). Pass
            through from benchmark_runner so the critic uses the SAME
            model family as the worker — keeps ablation experiments
            single-variable.
        max_iters: max outer (critic → rehand) loops.

    Each iteration captures:
      - panel image the critic saw
      - pre-fix mask + geojson (state BEFORE this iter's action)
      - post-fix mask + geojson (state AFTER this iter's action, if any)
      - the critic's directive (diagnosis, action, reason, args)
    """

    pre_snapshot = _snapshot_state(state)
    if pre_snapshot["geojson"] is None:
        if verbose:
            print("  critic: no geojson, skipping")
        return {
            "iterations": [], "final_decision": None, "changed_mask": False,
            "tokens_used": {"request": 0, "response": 0},
            "panel_iter0": None, "applied_rotation_deg": None,
            "pre_snapshot": pre_snapshot, "final_snapshot": pre_snapshot,
            "per_iteration_panels": [], "per_iteration_snapshots": [],
            "directive": None,
        }

    iterations: List[Dict[str, Any]] = []
    panels: List[Optional[np.ndarray]] = []
    snapshots: List[Dict[str, Any]] = []
    total_in = 0
    total_out = 0
    current_worker_result = worker_result

    for it_idx in range(max_iters):
        snap_before = _snapshot_state(state)
        directive, panel, det, wall, in_t, out_t = _run_critic_once(state, model_name)
        total_in += in_t
        total_out += out_t
        panels.append(panel)

        if verbose:
            print(f"  critic iter{it_idx}: {directive.action}  "
                  f"reason={directive.reason[:80]!r}  wall={wall:.1f}s")

        iter_entry = {
            "iter_idx": it_idx,
            "decision": directive.action,
            "diagnosis": directive.diagnosis,
            "score": det.get("score", 0.5),
            "checks": det.get("checks", {}),
            "fix_applied": "",
            "reason": directive.reason,
            "directive": directive.model_dump(),
            "wall_s": round(wall, 1),
        }
        snap = {
            "pre_fix_mask": snap_before.get("mask"),
            "pre_fix_geojson": snap_before.get("geojson"),
            "pre_fix_affine_H": snap_before.get("affine_H"),
        }

        if directive.action == "approve":
            iter_entry["fix_applied"] = "approve"
            iterations.append(iter_entry)
            snapshots.append(snap)
            break

        new_worker_result = _rehand_to_worker(state, current_worker_result,
                                                directive, verbose=verbose)
        current_worker_result = new_worker_result

        snap_after = _snapshot_state(state)
        snap["post_fix_mask"] = snap_after.get("mask")
        snap["post_fix_geojson"] = snap_after.get("geojson")
        snap["post_fix_affine_H"] = snap_after.get("affine_H")
        iter_entry["fix_applied"] = directive.action
        iterations.append(iter_entry)
        snapshots.append(snap)

        if (snap_before.get("geojson") == snap_after.get("geojson")
                and snap_after.get("mask") is not None):
            if verbose:
                print(f"  critic rehand: state unchanged, worker may not have "
                      f"complied — exiting loop")
            break

    final_snapshot = _snapshot_state(state)
    changed_mask = pre_snapshot.get("geojson") != final_snapshot.get("geojson")
    final_decision = iterations[-1]["decision"] if iterations else "approve"

    return {
        "iterations": iterations,
        "final_decision": final_decision,
        "changed_mask": changed_mask,
        "tokens_used": {"request": total_in, "response": total_out},
        "panel_iter0": panels[0] if panels else None,
        "applied_rotation_deg": None,
        "pre_snapshot": pre_snapshot,
        "final_snapshot": final_snapshot,
        "per_iteration_panels": panels,
        "per_iteration_snapshots": snapshots,
        "directive": iterations[-1]["directive"] if iterations else None,
        "worker_reentered": any(i.get("fix_applied") and
                                 i["fix_applied"] != "approve"
                                 for i in iterations),
    }


# ── Entry point ────────────────────────────────────────────────────────────

def run_critic_loop(
    state: Any,
    worker_result: Any,
    model_name: str,
    verbose: bool = True,
    max_iters: int = 2,
) -> Dict[str, Any]:
    """Run the LLM critic. Thin wrapper around `run_critic_agent`."""
    return run_critic_agent(
        state, worker_result, model_name=model_name,
        verbose=verbose, max_iters=max_iters,
    )
