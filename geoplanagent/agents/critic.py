"""Pairwise LLM critic: reviews stored match_attempts post-worker-commit."""

import copy
import time
from typing import Any, Dict, List, Literal, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent

from geoplanagent.utils import resolve_model, result_tokens, run_sync_with_retry
from geoplanagent.prompts import CRITIC_INSTRUCTIONS


class CriticDirective(BaseModel):
    """Pairwise judgement across all stored match_attempts."""

    chosen_candidate_id: int = Field(
        description="The candidate_id you believe is the best match. This "
        "may equal the worker's committed candidate (if you "
        "agree) or a different stored candidate (if you would "
        "switch). For retry_locate, set to the worker's "
        "committed_id (placeholder)."
    )
    action: Literal["approve", "switch", "retry_locate"] = Field(
        description="One of three actions: "
        "'approve' — the worker's committed candidate is the "
        "best stored option and looks correct visually. "
        "'switch' — a different stored candidate is a better "
        "fit; specify it in chosen_candidate_id. "
        "'retry_locate' — none of the stored candidates appear "
        "to be in the right region; the agent should re-locate "
        "from a different geocoding signal."
    )
    reasoning: str = Field(
        description="2-3 sentences naming concrete visual features: which "
        "named road, settlement shape, building block, or "
        "junction you used to judge alignment. Cite per-"
        "candidate metrics where relevant (n_inliers, "
        "road_name_agreement, scale_consistency). Do NOT say "
        "'looks reasonable' or 'I think' — point at features."
    )


_critic_agent: Optional[Agent] = None
_critic_agent_model: Optional[str] = None


def _ensure_agent(model_name: str) -> Agent:
    """Build (or return cached) critic Agent."""
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


def _label_strip(image: np.ndarray, text: str, height: int = 32) -> np.ndarray:
    """Black label strip above image; font auto-shrinks to fit width."""
    bar = np.full((height, image.shape[1], 3), 25, dtype=np.uint8)
    available_width = max(20, image.shape[1] - 16)  # 8px margin each side
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    (text_width, _), _ = cv2.getTextSize(text, font, scale, 1)
    while text_width > available_width and scale > 0.3:
        scale -= 0.05
        (text_width, _), _ = cv2.getTextSize(text, font, scale, 1)
    cv2.putText(bar, text, (8, height - 10), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, image])


def _binary_mask_at(mask: np.ndarray, shape: tuple) -> np.ndarray:
    """Binarise mask and nearest-neighbour resize it to (height, width) shape."""
    binary = (mask > 0).astype(np.uint8)
    if binary.shape != shape:
        binary = cv2.resize(binary, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return binary


def _build_one_group_panel(
    map_img: np.ndarray,
    mask: Optional[np.ndarray],
    tile_info: Optional[Dict[str, Any]],
    affine_H: Optional[np.ndarray],
    left_label: str,
    right_label: str,
    target_h: int = 480,
) -> Optional[np.ndarray]:
    """One LEFT|RIGHT row: map + SAM mask | OS tiles + projected polygon."""
    if map_img is None or tile_info is None or "image" not in tile_info:
        return None
    left = map_img.copy()
    if isinstance(mask, np.ndarray) and mask.sum() > 0:
        mask_binary = _binary_mask_at(mask, left.shape[:2])
        layer = left.copy()
        layer[mask_binary > 0] = (0, 255, 0)
        left = cv2.addWeighted(left, 0.55, layer, 0.45, 0)
    left_h, left_w = left.shape[:2]
    left = cv2.resize(left, (max(1, int(left_w * target_h / left_h)), target_h))

    tile_img = tile_info["image"]
    tile_bgr = cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR)

    # Project SAM-mask contour through affine_H onto the OS tile.
    if isinstance(mask, np.ndarray) and mask.sum() > 0 and affine_H is not None:
        mask_binary = _binary_mask_at(mask, map_img.shape[:2])
        contours, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            if len(contour) < 3:
                continue
            points = contour.reshape(-1, 2).astype(np.float32)
            points_homogeneous = np.concatenate(
                [points, np.ones((len(points), 1), dtype=np.float32)], axis=1
            )
            projected = points_homogeneous @ affine_H.T
            projected_int = np.round(projected).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                tile_bgr,
                [projected_int],
                isClosed=True,
                color=(0, 0, 255),
                thickness=max(2, tile_bgr.shape[0] // 250),
            )
    right_h, right_w = tile_bgr.shape[:2]
    right = cv2.resize(tile_bgr, (max(1, int(right_w * target_h / right_h)), target_h))

    return np.hstack([_label_strip(left, left_label), _label_strip(right, right_label)])


def _build_candidate_panel(
    state: Any, attempt: Dict[str, Any], is_committed: bool
) -> Optional[np.ndarray]:
    """Build the visual panel for ONE candidate (possibly multiple groups)."""
    candidate_id = attempt.get("candidate_id")
    badge = f"CANDIDATE {candidate_id}" + (" [COMMITTED]" if is_committed else "")
    rows: List[np.ndarray] = []
    for group in attempt.get("per_group") or []:
        page = group.get("page")
        map_img = state.rendered_pages.get(page) if state.rendered_pages else None
        mask = (state.sam_masks_by_page or {}).get(page)
        tile_info = group.get("tile_info")
        affine_H = group.get("affine_H")
        group_id = group.get("area_group", "?")
        zoom = (tile_info or {}).get("zoom", "?")
        # LEFT identifies the row; numeric metrics live in the text block.
        left_label = f"{badge}  group {group_id}  page {page}"
        right_label = f"OS tile @ zoom={zoom}"
        row = _build_one_group_panel(
            map_img, mask, tile_info, affine_H, left_label=left_label, right_label=right_label
        )
        if row is not None:
            rows.append(row)
    if not rows:
        return None
    # Pad rows to common width.
    max_width = max(row.shape[1] for row in rows)
    padded = []
    for row in rows:
        if row.shape[1] < max_width:
            pad = np.full((row.shape[0], max_width - row.shape[1], 3), 220, dtype=np.uint8)
            row = np.hstack([row, pad])
        padded.append(row)
        padded.append(np.full((4, max_width, 3), 220, dtype=np.uint8))
    return np.vstack(padded[:-1])


def _format_metrics_text(attempts: List[Dict[str, Any]], committed_ids: set) -> str:
    """Per-candidate metrics block; tags [COMMITTED] for each id in committed_ids."""
    lines = ["=== CANDIDATES ==="]
    if not committed_ids:
        lines.append("  worker's committed_ids: (none — nothing committed yet)")
    elif len(committed_ids) == 1:
        lines.append(f"  worker's committed_id: {next(iter(committed_ids))}")
    else:
        lines.append(f"  worker's committed_ids (one per area_group): {sorted(committed_ids)}")
    lines.append("")
    for attempt in attempts:
        candidate_id = attempt.get("candidate_id")
        tag = "  [COMMITTED]" if candidate_id in committed_ids else ""
        per_group = attempt.get("per_group") or []
        for group in per_group:
            match_info = group.get("match_info") or {}
            reward = group.get("reward") or {}
            n_inliers = int(match_info.get("n_inliers") or 0)
            axes = reward.get("axes") or {}
            road_agreement = axes.get("road_name_agreement") or {}
            scale = axes.get("scale_consistency") or {}
            road_score = road_agreement.get("score")
            road_verdict = road_agreement.get("verdict") or ""
            scale_score = scale.get("score")
            road_str = f"{road_score:.2f}" if isinstance(road_score, (int, float)) else "?"
            scale_str = f"{scale_score:.2f}" if isinstance(scale_score, (int, float)) else "?"
            verdict_str = f" ({road_verdict})" if road_verdict else ""
            lines.append(
                f"  cand {candidate_id}  group {group.get('area_group', '?')}  "
                f"page {group.get('page', '?')}  "
                f"n_inliers={n_inliers}  "
                f"road_name_agreement={road_str}{verdict_str}  "
                f"scale_consistency={scale_str}{tag}"
            )
    return "\n".join(lines)


def _run_critic_once(state: Any, model_name: str, message_history: Optional[list] = None) -> tuple:
    """One critic LLM call.

    If ``message_history`` is provided, the critic sees its own prior
    iteration(s) — lets iter 2 reason about whether the worker's
    response to iter 1's directive actually helped.

    Returns (directive, wall_s, in_tokens, out_tokens, llm_error,
    new_history).
    - new_history:  updated message list for next iteration (None on error)
    """
    attempts = sorted(
        state.match_attempts.values(), key=lambda a: int(a.get("candidate_id") or 0)
    )
    # SET of committed candidate_ids — one per area_group. Single-area
    # docs have one entry; multi-area docs can have several.
    committed_ids: set = set(
        int(c) for c in (getattr(state, "committed_groups", {}) or {}).values()
    )

    # Rank candidates by n_inliers (each attempt covers one area_group
    # so this is the per-group strength). Tie-break by candidate_id
    # for reproducibility.
    def _attempt_n_inliers(attempt: Dict[str, Any]) -> int:
        per_group = attempt.get("per_group") or [{}]
        match_info = (per_group[0] or {}).get("match_info") or {}
        return int(match_info.get("n_inliers") or 0)

    total_n_attempts = len(attempts)
    by_inliers = sorted(
        attempts,
        key=lambda a: (-_attempt_n_inliers(a), int(a.get("candidate_id") or 0)),
    )
    shown = list(by_inliers[:3])
    shown_ids = {a.get("candidate_id") for a in shown}
    # Always include every committed candidate, even if not in top-3.
    for candidate_id in committed_ids:
        if candidate_id in shown_ids:
            continue
        committed_attempt = state.match_attempts.get(candidate_id)
        if committed_attempt is not None:
            shown.append(committed_attempt)
            shown_ids.add(candidate_id)
    # Display order: by candidate_id (stable across iterations).
    shown.sort(key=lambda a: int(a.get("candidate_id") or 0))

    # Build ONE panel per shown candidate (sent as separate images, so
    # the VLM sees each candidate at full resolution rather than
    # downscaled inside a tall vertical stack).
    cand_panels_with_id = []
    for attempt in shown:
        panel_image = _build_candidate_panel(
            state, attempt, is_committed=(attempt.get("candidate_id") in committed_ids)
        )
        if panel_image is not None:
            cand_panels_with_id.append((attempt.get("candidate_id"), panel_image))

    metrics_text = _format_metrics_text(shown, committed_ids)
    selection_note = (
        f"Showing top-{len(shown)} of {total_n_attempts} stored candidates "
        f"(ranked by n_inliers; the worker's committed candidate(s) are "
        f"always included). On multi-area documents the worker commits "
        f"one candidate per area_group, so multiple panels may be tagged "
        f"[COMMITTED]. Each candidate is sent as a separate image."
    )

    # On follow-up iterations the message_history already contains the
    # critic's prior directive; an explicit header makes the meta-state
    # unambiguous so the model doesn't have to infer "this is a re-look"
    # from the conversation structure alone.
    if message_history is not None:
        header = (
            "FOLLOW-UP REVIEW. This is a subsequent iteration of your "
            "pairwise judgement on the same case. Your earlier directive "
            "is in the conversation above; the worker has responded "
            "(switched candidate, re-located, or attempted to). The "
            "panels and metrics below reflect the CURRENT state. "
            "Decide whether the response addressed your prior directive "
            "and whether the now-committed candidate is correct, or "
            "whether a further switch / retry_locate is warranted."
        )
        text_block = header + "\n\n" + selection_note + "\n\n" + metrics_text
    else:
        text_block = selection_note + "\n\n" + metrics_text

    # Multi-image input: one BinaryContent per shown candidate.
    user_input: List[Any] = [text_block]
    for _candidate_id, panel_image in cand_panels_with_id:
        _, encoded_png = cv2.imencode(".png", panel_image)
        user_input.append(BinaryContent(data=encoded_png.tobytes(), media_type="image/png"))

    agent = _ensure_agent(model_name)
    in_tokens = 0
    out_tokens = 0
    llm_error: Optional[str] = None
    new_history: Optional[list] = None
    start_time = time.time()
    try:
        result = run_sync_with_retry(
            agent,
            user_input,
            label="critic",
            message_history=message_history,
        )
        directive = result.output
        try:
            new_history = result.all_messages()
        except Exception:
            new_history = None
        in_tokens, out_tokens = result_tokens(result)
    except Exception as error:
        # Surface the failure: emit an approve directive (so the loop
        # exits cleanly) but tag it so downstream can distinguish a
        # genuine approve from a critic-LLM crash. Pick an arbitrary
        # committed_id when available; fall back to 0 (a valid id) when
        # nothing has been committed yet.
        llm_error = f"{type(error).__name__}: {str(error)[:100]}"
        directive = CriticDirective(
            chosen_candidate_id=(next(iter(committed_ids)) if committed_ids else 0),
            action="approve",
            reasoning=f"CRITIC_LLM_ERROR (treated as approve): {llm_error}",
        )
    wall_s = time.time() - start_time
    return (
        directive,
        wall_s,
        in_tokens,
        out_tokens,
        llm_error,
        new_history,
    )


def _direct_switch_commit(
    state: Any, chosen_id: int, directive_reasoning: str, verbose: bool = True
) -> None:
    """Commit ``chosen_id`` directly into state — no worker LLM call.

    The critic has already decided which candidate to commit; the
    candidate is already in ``state.match_attempts`` (it was put there
    by an earlier ``match_at`` call, and the caller has verified the
    id is present). An LLM re-invocation here would just be a
    middleman that types out ``commit_match(N)`` — wasted cost and
    deterministic risk (the worker could rationalise around the pick).

    Mutates ``state.current_result`` / ``state.last_output`` /
    ``state.accepted`` / ``state.accept_reason`` to mirror what a
    successful ``commit_match`` would have produced. The critic loop
    breaks on ``switch``, so we never need fresh worker messages
    downstream.
    """
    candidate = state.match_attempts.get(int(chosen_id))

    # Strict gate: refuse to commit a candidate that produced no
    # valid affine for any group — mirrors commit_match's check at
    # positioning.py around the n_groups_committed==0 path. The critic
    # *should* never pick such a candidate (it has no tile_info /
    # rendered panel), but guard against it: a candidate with no
    # geojson would silently null-out state.current_result["geojson"]
    # and crash downstream IoU scoring.
    n_groups_committed = int(candidate.get("n_groups_committed") or 0)
    if n_groups_committed == 0 or candidate.get("geojson") is None:
        if verbose:
            print(
                f"  direct_switch: id {chosen_id} has no usable affine / "
                f"geojson (n_groups_committed={n_groups_committed}); skipping"
            )
        return

    # Per-group commit: replace the entry for THIS candidate's
    # area_group, then rebuild current_result by re-unioning every
    # committed group's geojson. For single-area docs (99% of cases)
    # this just replaces the one entry. For multi-area docs the
    # critic's switch affects only the group it picked; the other
    # groups stay committed to whatever the worker chose for them.
    from geoplanagent.tools.positioning import _recompute_current_result

    group_id = int(candidate.get("requested_group", 0))
    state.committed_groups[group_id] = int(chosen_id)
    _recompute_current_result(state)
    # Mirror commit_match's n_commits increment so agent_stats
    # accurately reflects total commits, including critic-driven
    # switches. Without this, the metric undercounts commits whenever
    # the critic flips the worker's pick.
    state.n_commits += 1

    # Synthesise a BoundaryOutcome for downstream consumers that read
    # ``state.last_output``. Preserve the prior status when available
    # — district_lookup outcomes never reach the critic, so this is
    # almost always "accepted", but we keep the carry-forward to be
    # safe against schema changes.
    from geoplanagent.schemas import BoundaryOutcome

    previous_output = getattr(state, "last_output", None)
    previous_status = previous_output.status if previous_output is not None else "accepted"
    new_reasoning = f"[critic-switch to candidate {chosen_id}] {(directive_reasoning or '')[:900]}"
    # final_n_inliers reflects the SUM across all committed groups,
    # which is what state.current_result["total_inliers"] holds after
    # _recompute_current_result ran above.
    state.last_output = BoundaryOutcome(
        status=previous_status,
        final_n_inliers=int(state.current_result.get("total_inliers") or 0),
        rotation_checked=state.rotation_checked,
        reasoning=new_reasoning,
    )
    state.accepted = previous_status in ("accepted", "district_lookup")
    state.accept_reason = f"[{previous_status}] {new_reasoning[:160]}"

    if verbose:
        print(
            f"  direct_switch: committed candidate {chosen_id} "
            f"(sum_n_inliers={state.current_result['total_inliers']}) "
            f"without worker re-invoke"
        )


def _rehand_to_worker(
    state: Any, worker_result: Any, directive: CriticDirective, verbose: bool = True
) -> Any:
    """Apply the critic's directive to state.

    Two paths, with very different cost profiles:

    * ``switch``       → handled entirely in Python by
      ``_direct_switch_commit``. No LLM call. The critic already chose
      the candidate; we just copy it into state.
    * ``retry_locate`` → worker IS re-invoked: it has to call
      ``propose_centers`` + ``match_at`` on a NEW location, which
      requires LLM reasoning about why the previous locate was wrong.

    Returns ``(worker_result, rehand_error_or_None)``.
    """
    from geoplanagent.agents.worker import _worker_agent

    action = directive.action
    current_result = state.current_result or {}
    worker_committed_id = current_result.get("candidate_id")

    rehand_error: Optional[str] = None
    if action == "switch":
        chosen_id = directive.chosen_candidate_id
        if chosen_id == worker_committed_id:
            if verbose:
                print(f"  critic switch: chose committed_id {chosen_id} — treating as approve")
            return worker_result, None
        if chosen_id not in state.match_attempts:
            if verbose:
                print(f"  critic switch: id {chosen_id} not in stored candidates, skipping")
            return worker_result, None
        _direct_switch_commit(
            state,
            chosen_id,
            directive.reasoning or "",
            verbose=verbose,
        )
        return worker_result, None

    elif action == "retry_locate":
        # Refill the worker's match_at budget and partial-clear the
        # dedup set so the rehand can actually call propose_centers +
        # match_at again. Without this, a worker that exhausted its
        # 5-call budget during the first pass crashes on its next
        # match_at, the exception is swallowed below, and the critic's
        # intended fix is lost with no on-disk trace.
        state.match_at_budget = max(state.match_at_budget, 0) + 2
        state.seen_call_keys = set()
        instruction = (
            f"Reconsider your commit. None of your stored match "
            f"candidates appear to align with the boundary drawn on the "
            f"planning map — the projected polygons sit in the wrong "
            f"region. {directive.reasoning}\n\n"
            f"Call propose_centers with a match_context describing what "
            f"went wrong (in your own words), then match_at on the new "
            f"candidate, then commit_match on the new candidate's id, "
            f"then re-submit BoundaryOutcome."
        )

    else:
        # 'approve' shouldn't reach here; nothing to rehand.
        return worker_result, None

    if verbose:
        print(f"  critic rehand: re-invoking worker with action={action}")
    try:
        history = (
            worker_result.all_messages()
            if worker_result is not None and hasattr(worker_result, "all_messages")
            else None
        )
        sub_result = run_sync_with_retry(
            _worker_agent,
            instruction,
            deps=state,
            message_history=history,
            label="critic-rehand",
        )
        # Propagate the new worker outcome into state so downstream sees
        # the latest accepted/last_output/accept_reason.
        try:
            new_outcome = sub_result.output
            if new_outcome is not None:
                state.last_output = new_outcome
                state.accepted = new_outcome.status in ("accepted", "district_lookup")
                state.accept_reason = f"[{new_outcome.status}] {new_outcome.reasoning[:160]}"
        except Exception:
            pass
        return sub_result, None
    except Exception as error:
        rehand_error = f"{type(error).__name__}: {str(error)[:160]}"
        if verbose:
            print(f"  critic rehand: worker re-invoke failed: {rehand_error}")
        return worker_result, rehand_error


def _snapshot_geojson(state: Any) -> Optional[dict]:
    """Snapshot the current committed geojson.

    Returns a deep copy so a later commit_match (which builds a fresh
    state.current_result dict but might be refactored to mutate in
    place) cannot retroactively corrupt the snapshot.
    """
    current_result = state.current_result or {}
    geojson = current_result.get("geojson")
    return copy.deepcopy(geojson) if geojson is not None else None


def run_critic_loop(
    state: Any,
    worker_result: Any,
    model_name: str,
    max_iters: int = 2,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the LLM critic loop after the worker's first submission.

    Args:
        state: AgentState with worker's commit + stored match_attempts.
        worker_result: pydantic-ai result from the worker (for
            message_history when re-handing).
        model_name: OpenRouter model id passed through to the critic
            (keeps worker/critic on the same family for ablation).
        max_iters: max critic-rejection iterations before forcing accept.
        verbose: print per-iter status.

    Returns a dict with:
        - worker_first_geojson : worker's commit BEFORE any critic intervention
        - iterations           : list of per-iter records (action, reasoning, ...)
        - n_rejections         : count of switch/retry_locate actions issued
        - tokens               : {request, response} totals across critic calls
        - final_worker_result  : latest worker result (for token/tool stats)
    """
    # Snapshot the worker's first commit BEFORE any critic intervention.
    worker_first_geojson = _snapshot_geojson(state)
    if worker_first_geojson is None:
        if verbose:
            print("  critic: no geojson committed, skipping")
        return {
            "worker_first_geojson": None,
            "iterations": [],
            "n_rejections": 0,
            "tokens": {"request": 0, "response": 0},
        }

    iterations: List[Dict[str, Any]] = []
    total_in_tokens = 0
    total_out_tokens = 0
    current_worker_result = worker_result
    # Critic's own message history across iterations — lets iter 2
    # reason about whether iter 1's directive was followed and helped.
    # Each iteration's user_input still rebuilds the current panels +
    # metrics from state, so the critic always sees current candidates;
    # the history just adds its own prior reasoning to that context.
    critic_message_history: Optional[list] = None

    for iter_idx in range(max_iters):
        (
            directive,
            wall_s,
            iter_in_tokens,
            iter_out_tokens,
            llm_error,
            new_history,
        ) = _run_critic_once(state, model_name, message_history=critic_message_history)
        # Update history for next iter (only if the call succeeded; on
        # LLM error new_history is None and we keep prior history).
        if new_history is not None:
            critic_message_history = new_history
        total_in_tokens += iter_in_tokens
        total_out_tokens += iter_out_tokens

        if verbose:
            err_tag = " [LLM_ERROR]" if llm_error else ""
            print(
                f"  critic iter{iter_idx}: action={directive.action}{err_tag}  "
                f"chose={directive.chosen_candidate_id}  "
                f"reason={directive.reasoning[:80]!r}  wall={wall_s:.1f}s"
            )

        iter_entry = {
            "iter_idx": iter_idx,
            "action": directive.action,
            "chosen_candidate_id": directive.chosen_candidate_id,
            "reasoning": directive.reasoning,
            "wall_s": round(wall_s, 1),
            "llm_error": llm_error,  # None on success, error string on crash
        }

        if directive.action == "approve":
            iterations.append(iter_entry)
            break

        # Apply the directive to state (switch = direct Python commit,
        # no worker LLM; retry_locate = worker re-invoked).
        before = _snapshot_geojson(state)
        prev_worker_result = current_worker_result
        current_worker_result, rehand_error = _rehand_to_worker(
            state, current_worker_result, directive, verbose=verbose
        )
        # A retry_locate rehand re-invokes the worker, yielding a NEW result
        # object; fold its tokens into the critic total so the critic's
        # reported cost includes the worker turns it triggered. switch / failed
        # rehands return the SAME object, so the identity check avoids
        # double-counting the original worker run (already in worker_*_tokens).
        if rehand_error is None and current_worker_result is not prev_worker_result:
            rehand_in, rehand_out = result_tokens(current_worker_result)
            total_in_tokens += rehand_in
            total_out_tokens += rehand_out
        after = _snapshot_geojson(state)
        iter_entry["geojson_changed"] = before != after
        if rehand_error is not None:
            iter_entry["rehand_error"] = rehand_error
        iterations.append(iter_entry)

        # ``switch`` is one-and-done. The critic already chose the
        # candidate; re-asking on iter 1 would just show the SAME
        # panels with only the 'committed' marker moved to the new
        # candidate — rubber-stamp noise that costs another LLM call.
        # Only ``retry_locate`` produces NEW candidates worth a fresh
        # look on iter 1.
        if directive.action == "switch":
            break

        # If a retry_locate rehand didn't change state, the worker
        # couldn't comply — break to avoid spinning on a stuck case.
        if not iter_entry["geojson_changed"]:
            if verbose:
                print("  critic rehand: geojson unchanged after rehand, exiting loop")
            break

    # A genuine rejection is any iteration where the critic asked us
    # to change state. The crashed-LLM path synthesises a directive
    # with action="approve" + llm_error set; we must NOT count those
    # as approvals, otherwise a run where 100% of critic calls
    # crashed would report n_rejections=0 (= 100% agreement), which
    # is the opposite of reality.
    n_rejections = sum(
        1
        for iteration in iterations
        if iteration["action"] != "approve" or iteration.get("llm_error")
    )

    return {
        "worker_first_geojson": worker_first_geojson,
        "iterations": iterations,
        "n_rejections": n_rejections,
        "tokens": {"request": total_in_tokens, "response": total_out_tokens},
        # The latest worker result — carries the full conversation including
        # all critic-triggered rehand sub-turns. Used for agent-stats
        # extraction so the per-run token/tool counts include those
        # interventions; otherwise the rehand turns are lost.
        "final_worker_result": current_worker_result,
    }
