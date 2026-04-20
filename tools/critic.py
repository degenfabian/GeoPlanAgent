"""tools/critic.py — Commenter VLM critic loop (Paper2Poster-style Phase 3).

Runs AFTER the worker agent has returned an "accepted" BoundaryOutcome.
Sees the final mask + geojson + OS-tile overlay and either approves,
auto-fixes via code (retry_sam / retry_projection / retry_rotation), or
re-enters the worker agent with feedback (retry_in_worker).

The critic NEVER nullifies the GeoJSON — flag_low_confidence is the strongest
negative decision and only labels the result for downstream logging.

Public API:
    - CriticDecision: pydantic output schema
    - run_critic_loop(state, worker_agent, worker_result, model, sam3,
                       minima_matcher, verbose) -> dict
"""

from __future__ import annotations

import os
import tempfile
import traceback
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

import cv2
import numpy as np
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.usage import UsageLimits

if TYPE_CHECKING:
    from tools.agent import AgentState  # noqa: F401


# ── Schema ──────────────────────────────────────────────────────────────────

class CriticDecision(BaseModel):
    decision: Literal[
        "approve",
        "retry_sam",
        "retry_projection",
        "retry_rotation",
        "retry_in_worker",
        "flag_low_confidence",
    ] = Field(description="What the critic wants to do. See system prompt.")
    reasoning: str = Field(
        ..., min_length=30,
        description="Concrete observation of what LEFT/RIGHT panels show and why "
                    "this decision. ≥30 chars. 'looks wrong' is not acceptable — "
                    "say what specifically doesn't match.",
    )
    suggested_sam_query: Optional[str] = Field(
        default=None,
        description="For retry_sam: new SAM query, preferring the Reader's "
                    "boundary_color if it was identified (e.g. 'land edged red').",
    )
    suggested_select_indices: Optional[List[int]] = Field(
        default=None,
        description="For retry_sam: 0..4 indices of SAM candidates to combine "
                    "(union). Omit to take top-1.",
    )
    apply_hole_fill: bool = Field(
        default=False,
        description="For retry_projection: morphological close to fill holes.",
    )
    apply_dilation: bool = Field(
        default=False,
        description="For retry_projection: expand thin outline masks.",
    )
    suggested_rotation_deg: Optional[Literal[90, 180, 270]] = Field(
        default=None,
        description="For retry_rotation: clockwise degrees to rotate the map "
                    "so the RIGHT polygon matches LEFT road layout.",
    )
    worker_should_skip_sources: Optional[List[str]] = Field(
        default=None,
        description="For retry_in_worker: geocoding sources to disable on next "
                    "position_boundary. Valid: grid_refs_centroid, gpkg, wikidata, "
                    "nominatim:addr, nominatim:road.",
    )
    worker_suggestion: Optional[str] = Field(
        default=None,
        description="For retry_in_worker: free-text guidance for the worker.",
    )
    suspected_wrong_location: bool = Field(
        default=False,
        description="True if OS tile settlement pattern doesn't match planning map.",
    )
    confidence: Literal["high", "medium", "low"] = "medium"


# ── Critic Agent ────────────────────────────────────────────────────────────

_CRITIC_SYSTEM_PROMPT = """You are an independent boundary verifier.

You see a composite image with TWO panels side-by-side:
- LEFT: the planning PDF map with the extracted boundary mask overlaid in red.
- RIGHT: OS tiles at the positioned location with the projected polygon outlined in red.

The context text lists n_inliers, which geocoding sources produced centers
(centers_tried with [picked] marking the winner), the Reader's boundary_color
and detected map rotation, the Worker's reasoning and tool-call summary, and
any prior critic iterations + code fixes applied.

Pick exactly one decision:

- approve: red mask on LEFT follows the colored boundary the planning map
  actually draws, AND the polygon on RIGHT roughly aligns with the road /
  settlement pattern on the map. This is the default when evidence is ambiguous.

- retry_sam: mask on LEFT is clearly wrong shape — grabbed far too much
  (whole map), far too little (tiny speck), or the wrong region entirely.
  Fill suggested_sam_query (use the Reader's boundary_color, e.g. 'land edged
  red' / 'pink shaded area' / 'red tick marks along buildings'). If you see
  multiple candidate regions fill suggested_select_indices (0..4).

- retry_projection: mask outline on LEFT roughly matches the colored boundary
  but the polygon on RIGHT has big internal holes, spikes, or fragmentation.
  Set apply_hole_fill=true (closes gaps in the outline) and/or
  apply_dilation=true (expands a thin outline into a filled shape).

- retry_rotation: polygon on RIGHT is clearly rotated 90 / 180 / 270° vs the
  road layout on LEFT — the shapes match but oriented wrong. Fill
  suggested_rotation_deg with the clockwise correction. Do NOT pick this for
  general shape mismatches — only when rotation is visibly the issue.

- retry_in_worker: the fix needs tool orchestration that a simple code fix
  can't do. Examples: OS tiles show a fundamentally different settlement
  (wrong geocoding source won); centers_tried shows only one weak source;
  multiple prior code fixes haven't helped. Fill worker_should_skip_sources
  with sources you think produced a bad center (e.g. ['wikidata'] if wikidata
  is the winning source and the location clearly doesn't match) and/or
  worker_suggestion with a free-text hint. Prefer code fixes first — only
  pick this on the first iteration if it's clearly a geocoding problem.

- flag_low_confidence: you've seen prior iterations fail to improve things,
  or the output is clearly wrong but you have no actionable fix. The GeoJSON
  is kept regardless — this is a warning label. Use sparingly.

Always-allowed flags (fill independently of decision):
- suspected_wrong_location: true if OS tile features are completely unrelated
  to the planning map.

Rules:
- Default to 'approve' when evidence is ambiguous. We optimise to not regress
  currently-good cases.
- reasoning must be ≥30 chars and concrete. "polygon rotated 90° vs roads"
  is good; "looks off" is not acceptable.
- If a prior iteration's fix was applied, comment on whether it helped.
- Do not pick retry_in_worker as the first decision unless it's clearly a
  geocoding-source problem — code fixes are cheaper, try them first.
- If a prior fix_applied string ends with '_failed', '_no_candidates', or
  '_invalid', that code path could not execute. Do not repeat the same
  decision — pick retry_in_worker or flag_low_confidence instead. (In
  practice the runtime will escalate for you on hard failures, but do not
  rely on it.)
"""

_critic_agent = Agent(
    "test",  # placeholder, overridden at run time via model= kwarg
    output_type=CriticDecision,
    retries=2,
    output_retries=2,
    instructions=_CRITIC_SYSTEM_PROMPT,
)


# ── Visual panel builder ────────────────────────────────────────────────────

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


def build_critic_panel(state: Any) -> Optional[np.ndarray]:
    """Two-panel composite the critic sees. None if required state is missing."""
    if state.map_img is None or state.current_mask is None:
        return None

    # Late imports to avoid circular deps (agent.py imports this module inside
    # run_agent).
    from tools.agent import _create_boundary_overlay, _draw_geojson_on_tiles
    from tools.os_opendata_tiles import fetch_os_opendata_grid

    left = _create_boundary_overlay(state.map_img, state.current_mask)
    left = _resize_height(left, 600)
    left = _add_label_bar(left, "PLANNING MAP + MASK (red)")

    mi = (state.current_result or {}).get("match_info") or {}
    center_ll = mi.get("center_latlon")
    geojson = (state.current_result or {}).get("geojson")

    if center_ll is None or geojson is None:
        right = np.full((600, 800, 3), 240, dtype=np.uint8)
        cv2.putText(right, "(no positioning)", (50, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    else:
        try:
            tile_info = fetch_os_opendata_grid(center_ll[0], center_ll[1], 17, 5, 5)
            right = cv2.cvtColor(tile_info["image"], cv2.COLOR_RGB2BGR)
            right = _draw_geojson_on_tiles(right, geojson, tile_info)
        except Exception:
            right = np.full((600, 800, 3), 240, dtype=np.uint8)
            cv2.putText(right, "(OS tile fetch failed)", (50, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    right = _resize_height(right, 600)
    right = _add_label_bar(right, "OS TILES + POLYGON (red)")

    # Align heights (both panels have 35px label bar + 600px content = 635)
    if left.shape[0] != right.shape[0]:
        target = min(left.shape[0], right.shape[0])
        left = cv2.resize(left, (left.shape[1], target))
        right = cv2.resize(right, (right.shape[1], target))

    panel = np.hstack([left, right])
    if panel.shape[1] > 1600:
        scale = 1600 / panel.shape[1]
        panel = cv2.resize(panel, (1600, int(panel.shape[0] * scale)))
    return panel


# ── Context text builder ────────────────────────────────────────────────────

def _summarize_tool_calls(worker_result: Any) -> str:
    try:
        counts: Dict[str, int] = {}
        for msg in worker_result.all_messages():
            parts = getattr(msg, "parts", None) or []
            for p in parts:
                kind = str(getattr(p, "kind", "")).lower()
                if "toolcall" in kind:
                    name = getattr(p, "tool_name", "?")
                    counts[name] = counts.get(name, 0) + 1
        return ", ".join(f"{k} ×{v}" for k, v in sorted(counts.items())) or "(none)"
    except Exception:
        return "(unavailable)"


def build_context_text(state: Any, worker_result: Any,
                        prior_iterations: List[Dict[str, Any]]) -> str:
    mi = (state.current_result or {}).get("match_info") or {}
    pdf_info = state.pdf_info or {}

    reasoning = ""
    if state.last_output is not None:
        reasoning = (state.last_output.reasoning or "")[:400]

    centers_lines: List[str] = []
    for c in (state.centers_tried or [])[:10]:
        mark = " [picked]" if c.get("was_picked") else ""
        centers_lines.append(f"  - {c['source']}: {c['name']}{mark}")
    centers_block = "\n".join(centers_lines) if centers_lines else "  (none tracked)"

    prior_block = ""
    if prior_iterations:
        lines = []
        for pi in prior_iterations:
            line = (f"  iter {pi.get('super_iter', 0)}.{pi.get('inner_iter', 0)}: "
                    f"{pi.get('decision')}")
            if pi.get("fix_applied"):
                line += f" → fix_applied={pi['fix_applied']}"
            line += f" — {(pi.get('reasoning') or '')[:120]}"
            lines.append(line)
        prior_block = "\nPRIOR ITERATIONS:\n" + "\n".join(lines) + "\n"

    return (
        f"Context:\n"
        f"- Worker's reasoning: {reasoning or '(empty)'}\n"
        f"- Worker tool-call summary: {_summarize_tool_calls(worker_result)}\n"
        f"- n_inliers: {mi.get('n_inliers', 0)}  score: {float(mi.get('score', 0) or 0):.1f}  "
        f"aspect: {float(mi.get('aspect', 0) or 0):.3f}\n"
        f"- Reader boundary_color: {pdf_info.get('boundary_color') or '(not identified)'}\n"
        f"- Reader map_rotation: {pdf_info.get('map_rotation', 0)}° (already applied if non-zero)\n"
        f"- centers_tried:\n{centers_block}\n"
        f"{prior_block}"
    )


# ── Code fixes (no LLM) ─────────────────────────────────────────────────────

def _mask_to_uint8(m: np.ndarray) -> np.ndarray:
    if m.dtype == np.uint8:
        return m
    if m.dtype == bool:
        return (m.astype(np.uint8)) * 255
    return (m > 0).astype(np.uint8) * 255


def _reproject(state: Any, mask: np.ndarray) -> Optional[dict]:
    """Re-project mask using existing affine_H + tile_info. Returns geojson or None."""
    from tools.positioning import mask_to_geojson_affine
    affine_H = (state.current_result or {}).get("affine_H")
    tile_info = (state.current_result or {}).get("tile_info")
    if affine_H is None or tile_info is None:
        return None
    return mask_to_geojson_affine(mask, affine_H, tile_info)


def _apply_retry_sam(state: Any, decision: CriticDecision,
                      sam3_processor: Any, sam3_model: Any, device: Any) -> str:
    from tools.sam3_boundary import extract_candidates

    if state.map_crop_path is None:
        return "retry_sam_no_map"
    query = decision.suggested_sam_query or "planning boundary"
    try:
        candidates = extract_candidates(
            state.map_crop_path, sam3_processor, sam3_model, device,
            query=query, top_k=5,
        )
    except Exception as e:
        return f"retry_sam_failed: {e!s:.80}"
    if not candidates:
        return "retry_sam_no_candidates"

    masks = [c["mask"] for c in candidates]
    state.instance_masks = masks

    indices = decision.suggested_select_indices
    if indices:
        valid = [i for i in indices if 0 <= i < len(masks)]
        if not valid:
            valid = [0]
        stacked = np.any(np.stack([masks[i] for i in valid], axis=0), axis=0)
        new_mask = _mask_to_uint8(stacked)
        state.selected_indices = valid
    else:
        new_mask = _mask_to_uint8(masks[0])
        state.selected_indices = [0]

    state.current_mask = new_mask
    new_geojson = _reproject(state, new_mask)
    if new_geojson is not None:
        state.current_result["geojson"] = new_geojson
    return f"retry_sam(query={query!r}, indices={state.selected_indices})"


def _apply_retry_projection(state: Any, decision: CriticDecision) -> str:
    if state.current_mask is None:
        return "retry_projection_no_mask"

    mask = state.current_mask.copy()
    actions: List[str] = []

    if decision.apply_hole_fill:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        actions.append("hole_fill")

    if decision.apply_dilation:
        try:
            from tools.positioning import _expand_thin_mask
            mask = _expand_thin_mask(mask)
            actions.append("dilation")
        except (ImportError, AttributeError):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask = cv2.dilate(mask, kernel, iterations=2)
            actions.append("dilation_fallback")

    if not actions:
        return "retry_projection_noop"

    state.current_mask = mask
    new_geojson = _reproject(state, mask)
    if new_geojson is not None:
        state.current_result["geojson"] = new_geojson
    return f"retry_projection({', '.join(actions)})"


def _apply_retry_rotation(state: Any, decision: CriticDecision,
                           sam3_processor: Any, sam3_model: Any, device: Any,
                           minima_matcher: Any) -> str:
    """Rotate map → re-SAM → re-MINIMA → re-project. Atomic commit."""
    from tools.sam3_boundary import extract_candidates
    from tools.positioning import sliding_window_position, mask_to_geojson_affine

    deg = decision.suggested_rotation_deg
    if deg not in (90, 180, 270) or state.map_img is None:
        return "retry_rotation_invalid"

    rot_code = {90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE}[deg]
    rotated = cv2.rotate(state.map_img, rot_code)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        new_crop_path = tmp.name
        cv2.imwrite(tmp.name, rotated)

    def _cleanup_and(msg: str) -> str:
        try:
            os.unlink(new_crop_path)
        except OSError:
            pass
        return msg

    try:
        candidates = extract_candidates(
            new_crop_path, sam3_processor, sam3_model, device,
            query="planning boundary", top_k=5,
        )
    except Exception as e:
        return _cleanup_and(f"retry_rotation_sam_failed: {e!s:.80}")
    if not candidates:
        return _cleanup_and("retry_rotation_no_sam_candidates")

    new_masks = [c["mask"] for c in candidates]
    new_mask = _mask_to_uint8(new_masks[0])

    try:
        result = sliding_window_position(
            matcher=minima_matcher,
            map_img=rotated,
            sam3_mask=new_mask,
            centers=list(state.centers),
            scale_ratio=state.scale_ratio,
            dpi=state.dpi,
            tile_fetcher=None,
            grayscale=False,
        )
    except Exception as e:
        return _cleanup_and(f"retry_rotation_minima_failed: {e!s:.80}")

    affine_H = result.get("affine_H")
    tile_info = result.get("tile_info")
    if affine_H is None or tile_info is None:
        return _cleanup_and("retry_rotation_no_affine")

    new_geojson = mask_to_geojson_affine(new_mask, affine_H, tile_info)
    if new_geojson is None:
        return _cleanup_and("retry_rotation_projection_failed")

    # All steps succeeded — commit
    old_crop_path = state.map_crop_path
    state.map_img = rotated
    state.map_crop_path = new_crop_path
    state.current_mask = new_mask
    state.instance_masks = new_masks
    state.selected_indices = [0]
    state.current_result = result
    state.current_result["geojson"] = new_geojson
    state.critic_applied_rotation_deg = deg

    if old_crop_path and old_crop_path != new_crop_path \
            and os.path.exists(old_crop_path):
        try:
            os.unlink(old_crop_path)
        except OSError:
            pass

    mi = result.get("match_info") or {}
    return f"retry_rotation({deg}deg, new_inliers={mi.get('n_inliers', 0)})"


# ── Worker re-entry ─────────────────────────────────────────────────────────

def _build_worker_feedback_prompt(decision: CriticDecision) -> str:
    pieces = [
        "The independent critic reviewed your previous result.",
        f"Critic's reasoning: {decision.reasoning}",
    ]
    if decision.worker_should_skip_sources:
        pieces.append(
            f"Critic suggests retrying position_boundary with "
            f"skip_sources={decision.worker_should_skip_sources}. This disables "
            f"those automatic geocoding sources so a different center can win. "
            f"Call position_boundary(skip_sources=..., scale_ratio=..., road_names=...) "
            f"then re-run extract_boundary + project_boundary."
        )
    if decision.worker_suggestion:
        pieces.append(f"Additional guidance: {decision.worker_suggestion}")
    pieces.append(
        "Please (1) call the appropriate tools to improve the result, and "
        "(2) resubmit an updated BoundaryOutcome. If you genuinely cannot "
        "improve, submit status='rejected_visual_mismatch' with a reject_reason "
        "— the downstream code keeps your partial GeoJSON regardless."
    )
    return "\n\n".join(pieces)


def _worker_reentry(worker_agent: Any, worker_result: Any, state: Any,
                     decision: CriticDecision, model: Any,
                     verbose: bool) -> tuple:
    """Returns (new_worker_result, ok)."""
    try:
        msg = _build_worker_feedback_prompt(decision)
        if verbose:
            print("  Critic: re-entering worker with feedback (request_limit=10)")
        new_result = worker_agent.run_sync(
            [msg],
            deps=state,
            model=model,
            message_history=worker_result.all_messages(),
            usage_limits=UsageLimits(request_limit=10),
        )
        return new_result, True
    except Exception as e:
        if verbose:
            print(f"  Critic: worker re-entry failed: {e}")
            traceback.print_exc()
        return worker_result, False


# ── Main orchestrator ───────────────────────────────────────────────────────

_TERMINAL_FIX_SUFFIXES = (
    "_failed", "_noop", "_invalid", "_no_mask", "_no_candidates",
    "_no_sam_candidates", "_no_affine", "_projection_failed", "_no_map",
)


def _fix_changed_mask(fix_str: Optional[str]) -> bool:
    if not fix_str:
        return False
    return not any(fix_str.endswith(suf) for suf in _TERMINAL_FIX_SUFFIXES)


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
    """Critic super-loop. See module docstring and plan file for semantics."""
    iterations: List[Dict[str, Any]] = []
    tokens = {"request": 0, "response": 0}
    panel_iter0: Optional[np.ndarray] = None
    changed_mask = False
    changed_geojson = False
    worker_reentered = False
    suspected_wrong_location = False
    last_decision: Optional[CriticDecision] = None

    sam3_proc = sam3.get("processor")
    sam3_model = sam3.get("model")
    device = sam3.get("device", "cpu")

    def _run_critic_once() -> Optional[CriticDecision]:
        nonlocal panel_iter0
        panel = build_critic_panel(state)
        if panel is None:
            if verbose:
                print("  Critic: panel build failed (no map_img/mask)")
            return None
        if panel_iter0 is None:
            panel_iter0 = panel.copy()
        context = build_context_text(state, worker_result, iterations)

        ok, buf = cv2.imencode(".png", panel)
        if not ok:
            if verbose:
                print("  Critic: panel encode failed")
            return None
        panel_bc = BinaryContent(data=buf.tobytes(), media_type="image/png")

        try:
            result = _critic_agent.run_sync(
                [context, panel_bc],
                model=model,
                usage_limits=UsageLimits(request_limit=4),
            )
        except Exception as e:
            if verbose:
                print(f"  Critic call failed: {e}")
            return None

        try:
            usage = result.usage()
            tokens["request"] += usage.request_tokens or 0
            tokens["response"] += usage.response_tokens or 0
        except Exception:
            pass
        return result.output

    super_iter = 0
    while super_iter < max_super:
        escalate_to_worker = False
        inner_iter = 0
        while inner_iter < max_inner:
            decision = _run_critic_once()
            if decision is None:
                iterations.append({
                    "super_iter": super_iter, "inner_iter": inner_iter,
                    "decision": "approve", "reasoning": "critic_call_failed",
                    "confidence": "low",
                    "suspected_wrong_location": False,
                    "fix_applied": None, "changed_mask": False,
                })
                last_decision = None
                break

            last_decision = decision
            suspected_wrong_location = (
                suspected_wrong_location or decision.suspected_wrong_location
            )

            entry: Dict[str, Any] = {
                "super_iter": super_iter, "inner_iter": inner_iter,
                "decision": decision.decision, "reasoning": decision.reasoning,
                "confidence": decision.confidence,
                "suspected_wrong_location": decision.suspected_wrong_location,
                "fix_applied": None, "changed_mask": False,
            }
            if verbose:
                print(f"  Critic iter {super_iter}.{inner_iter}: "
                      f"{decision.decision} ({decision.confidence}) — "
                      f"{decision.reasoning[:100]}")

            if decision.decision == "approve":
                iterations.append(entry)
                return _finalize(iterations, tokens, panel_iter0, "approve",
                                  changed_mask, changed_geojson,
                                  worker_reentered, suspected_wrong_location)
            if decision.decision == "flag_low_confidence":
                iterations.append(entry)
                return _finalize(iterations, tokens, panel_iter0,
                                  "flag_low_confidence", changed_mask,
                                  changed_geojson, worker_reentered,
                                  suspected_wrong_location)
            if decision.decision == "retry_in_worker":
                iterations.append(entry)
                escalate_to_worker = True
                break

            # Code-fix paths
            fix_applied: Optional[str] = None
            if decision.decision == "retry_sam":
                fix_applied = _apply_retry_sam(
                    state, decision, sam3_proc, sam3_model, device)
            elif decision.decision == "retry_projection":
                fix_applied = _apply_retry_projection(state, decision)
            elif decision.decision == "retry_rotation":
                fix_applied = _apply_retry_rotation(
                    state, decision, sam3_proc, sam3_model, device,
                    minima_matcher)

            entry["fix_applied"] = fix_applied
            if _fix_changed_mask(fix_applied):
                entry["changed_mask"] = True
                changed_mask = True
                changed_geojson = True
            iterations.append(entry)

            # Hard failure: the code path couldn't execute (SAM crashed,
            # MINIMA crashed, no candidates produced, projection returned
            # None, bad rotation arg). Re-running the critic on the same
            # state is pointless — short-circuit and escalate.
            if fix_applied and any(fix_applied.endswith(suf) for suf in
                                   ("_failed", "_no_candidates",
                                    "_no_sam_candidates", "_invalid",
                                    "_no_affine", "_projection_failed")):
                if verbose:
                    print(f"  Critic: hard fix failure ({fix_applied}); "
                          f"escalating to worker re-entry")
                escalate_to_worker = True
                break

            inner_iter += 1

        # Inner loop exited
        if escalate_to_worker and super_iter + 1 < max_super:
            worker_result, ok = _worker_reentry(
                worker_agent, worker_result, state,
                last_decision or decision, model, verbose,
            )
            worker_reentered = worker_reentered or ok
            super_iter += 1
            continue

        # No escalation path or budget exhausted
        break

    final = last_decision.decision if last_decision else "approve"
    return _finalize(iterations, tokens, panel_iter0, final,
                      changed_mask, changed_geojson, worker_reentered,
                      suspected_wrong_location)


def _finalize(iterations: List[Dict[str, Any]],
              tokens: Dict[str, int],
              panel: Optional[np.ndarray],
              final_decision: str,
              changed_mask: bool,
              changed_geojson: bool,
              worker_reentered: bool,
              suspected_wrong_location: bool) -> Dict[str, Any]:
    return {
        "iterations": iterations,
        "final_decision": final_decision,
        "changed_mask": changed_mask,
        "changed_geojson": changed_geojson,
        "worker_reentered": worker_reentered,
        "suspected_wrong_location": suspected_wrong_location,
        "panel_img_iter0": panel,
        "tokens": tokens,
    }
