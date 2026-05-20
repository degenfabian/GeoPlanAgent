"""Pairwise LLM critic — independent reviewer of the worker's commit.

The critic runs AFTER the worker submits a BoundaryOutcome. It sees:
  - a panel stack (one row per stored match_attempt) showing planning map
    + SAM mask overlay on the left and OS tiles + projected polygon
    (red) on the right
  - per-candidate metrics (n_inliers, road_name_agreement,
    scale_consistency)
  - which candidate the worker committed

The critic emits a CriticDirective specifying:
  - chosen_candidate_id : critic's pick (may equal worker's commit)
  - action              : approve | switch | retry_locate
  - reasoning           : 2-3 sentences naming concrete visual features

System templates the directive into a fixed user message and re-invokes
the worker. The worker is opaque to the critic's existence: its system
prompt is unchanged. Up to ``max_iters`` critic rejections per case.

Two-in-one ablation: the worker's first-commit polygon is the no-critic
baseline; the post-loop polygon is the with-critic outcome. Both IoUs
are reported.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent
from typing import Literal

from tools.agent._model import resolve_model


# ── Critic output schema ───────────────────────────────────────────────────


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


# Critic instructions — pairwise framing across candidates.
CRITIC_INSTRUCTIONS = """\
You are an independent reviewer of a UK planning-boundary extraction
pipeline. The agent has matched a planning map to OS map tiles, generated
candidate match attempts at different locations, projected SAM3 boundary
masks through those matches, and committed one candidate as its final
answer. Your job is pairwise comparison across the stored candidates and
a single directive on whether to accept or redirect.

WHAT YOU SEE
- ONE image per candidate (sent as separate images so each renders at
  full resolution rather than getting downscaled inside a tall stack).
  Each image is LEFT|RIGHT, with labels acting purely as identifiers
  (numeric metrics live in the text block — see "INTERPRETING THE
  METRICS" below):
    LEFT  = planning map with the SAM mask overlaid in translucent green.
            Label: "CANDIDATE {id} [COMMITTED] group {g} page {p}".
            The COMMITTED tag marks the worker's choice.
            "page" = the 1-based PDF page this row's match was run on.
            "group" = area_group index (see area_groups note below).
    RIGHT = OS tile render at the matched window, with the projected
            polygon outlined in red. Label: "OS tile @ zoom={z}".
            "OS" = Ordnance Survey, the UK national mapping agency
            whose vector + raster tiles the agent matches against.
            "zoom" = web-mercator tile zoom level (z17 ≈ 0.6m/px,
            z18 ≈ 0.3m/px, z19 ≈ 0.15m/px) — different candidates
            may have matched at different zooms, so each panel
            reports its own.
- Only the TOP-3 candidates by total_inliers are shown, plus the worker's
  committed candidate if it falls outside the top-3 (so you always see
  the worker's pick alongside the strongest alternatives). A note at the
  top of the metrics block tells you how many total candidates exist and
  how many are shown.
- area_groups: a single planning document can cover MULTIPLE separate
  geographic areas (e.g., a multi-site Article 4 direction). When this
  happens, ONE candidate image contains stacked sub-rows (one per area-
  group) and the metrics block has one line per (candidate, area-group).
- A metrics block listing per-candidate {n_inliers, road_name_agreement,
  scale_consistency} for the SHOWN candidates.
- The worker's committed candidate_id is also stated explicitly.

WHAT "GOOD" LOOKS LIKE — for a candidate
Trace named roads, settlement shapes, or distinctive features between
the planning map (left) and OS render (right). The boundary line on the
planning map (any colour / hatch) should sit where the red outlined
polygon sits in the OS render. Road junctions, building blocks, water
bodies should line up.

WHAT "BAD" LOOKS LIKE
- No road / feature correspondence: planning map shows urban streets;
  OS render shows farmland or a different street pattern.
- The SAM mask covers something other than the boundary (title block,
  legend, scale bar, large blob of text).
- The polygon outline lands well outside the planning map's drawn
  boundary, or its shape clearly doesn't match.

INTERPRETING THE METRICS
- n_inliers ≥ 50 is a strong signal that the affine is correct;
  < 25 is too weak to trust.
- scale_consistency near 1.0 means the recovered map scale matches
  the document's stated scale. < 0.5 hints at a possibly poor match,
  but if n_inliers is strong (≥ 80) the match can still be right.
- road_name_agreement = 0.0 means OS roads at this location exist
  but don't match the reader's road names — possible wrong-area
  signal. But be careful: if n_inliers is strong (≥ 80) and
  scale_consistency is reasonable, trust the inlier count over this
  signal.
- These numbers are supporting evidence — the visual panels are the
  primary signal for your decision.

DECISION (pick exactly one action)

- approve
    The worker's committed candidate shows clear road/feature
    correspondence AND the polygon outline aligns with the drawn
    boundary. Set chosen_candidate_id = committed_id.

- switch
    A DIFFERENT stored candidate looks visually better (clearer
    correspondence, better polygon alignment) than the committed one.
    Set chosen_candidate_id to that candidate's id. Cite the specific
    visual feature that swayed you.

- retry_locate
    None of the stored candidates show good correspondence — they all
    appear to be in the wrong region (no road / feature match, polygons
    in totally different terrain). The agent will be asked to re-locate
    from a different geocoding signal (postcode vs place vs road, etc.).
    Set chosen_candidate_id = committed_id (placeholder).

OUTPUT
A single CriticDirective. Reasoning MUST cite a concrete observation —
specific feature you saw matched or mismatched, or a specific metric.
Never use vague language ('looks fine', 'reasonable').
"""


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


# ── Panel-building helpers ─────────────────────────────────────────────────


def _label_strip(img: np.ndarray, text: str, height: int = 32) -> np.ndarray:
    """Render a black label strip above `img`, auto-shrinking the font
    so the text fits the image width (cv2.putText doesn't auto-wrap;
    fixed font sizes truncate at the panel boundary when the planning
    map ends up narrower than the label needs)."""
    bar = np.full((height, img.shape[1], 3), 25, dtype=np.uint8)
    available_w = max(20, img.shape[1] - 16)  # 8px margin each side
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    (text_w, _), _ = cv2.getTextSize(text, font, scale, 1)
    while text_w > available_w and scale > 0.3:
        scale -= 0.05
        (text_w, _), _ = cv2.getTextSize(text, font, scale, 1)
    cv2.putText(bar, text, (8, height - 10), font, scale,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _build_one_group_panel(map_img: np.ndarray,
                            mask: Optional[np.ndarray],
                            tile_info: Optional[Dict[str, Any]],
                            affine_H: Optional[np.ndarray],
                            left_label: str,
                            right_label: str,
                            target_h: int = 480) -> Optional[np.ndarray]:
    """Build a single LEFT|RIGHT row for one group of one candidate.

    LEFT: planning map with SAM mask overlay (translucent green).
    RIGHT: OS tile render with projected polygon outline (red).

    Labels are passed in separately (LEFT identifies the candidate,
    RIGHT reports per-match metrics) so neither label has to fit the
    full info — cv2.putText doesn't auto-wrap, so cramming all info
    into a single label gets it truncated at the image boundary.
    """
    if map_img is None or tile_info is None or "image" not in tile_info:
        return None
    left = map_img.copy()
    if isinstance(mask, np.ndarray) and mask.sum() > 0:
        mb = (mask > 0).astype(np.uint8)
        if mb.shape != left.shape[:2]:
            mb = cv2.resize(mb, (left.shape[1], left.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        layer = left.copy()
        layer[mb > 0] = (0, 255, 0)
        left = cv2.addWeighted(left, 0.55, layer, 0.45, 0)
    h_l, w_l = left.shape[:2]
    left = cv2.resize(left, (max(1, int(w_l * target_h / h_l)), target_h))

    tile_img = tile_info["image"]
    if tile_img.shape[2] == 3 and tile_info.get("_was_rgb", True):
        tile_bgr = cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR)
    else:
        tile_bgr = tile_img.copy()

    # Project SAM-mask contour through affine_H onto the OS tile.
    if isinstance(mask, np.ndarray) and mask.sum() > 0 and affine_H is not None:
        mb = (mask > 0).astype(np.uint8)
        if mb.shape != map_img.shape[:2]:
            mb = cv2.resize(mb, (map_img.shape[1], map_img.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
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
    right = cv2.resize(tile_bgr,
                       (max(1, int(w_r * target_h / h_r)), target_h))

    return np.hstack([_label_strip(left, left_label),
                      _label_strip(right, right_label)])


def _build_candidate_panel(state: Any, attempt: Dict[str, Any],
                            is_committed: bool) -> Optional[np.ndarray]:
    """Build the visual panel for ONE candidate (possibly multiple groups)."""
    cid = attempt.get("candidate_id")
    badge = f"CANDIDATE {cid}" + (" [COMMITTED]" if is_committed else "")
    rows: List[np.ndarray] = []
    for g in attempt.get("per_group") or []:
        page = g.get("page")
        map_img = state.rendered_pages.get(page) if state.rendered_pages else None
        mask = (state.sam_masks_by_page or {}).get(page)
        mi = g.get("match_info") or {}
        tile_info = g.get("tile_info") or mi.get("tile_info")
        affine_H = g.get("affine_H")
        group_id = g.get("area_group", "?")
        zoom = (tile_info or {}).get("zoom", "?")
        # Labels split LEFT|RIGHT so neither truncates:
        #   LEFT  = identifier of THIS image (which candidate / area_group /
        #           PDF page).
        #   RIGHT = zoom of the OS-tile crop in this image (the only
        #           per-image property of the OS render side; numeric
        #           metrics like n_inliers / road_name_agreement /
        #           scale_consistency live in the text-block, not on the
        #           panel label).
        left_label = f"{badge}  group {group_id}  page {page}"
        right_label = f"OS tile @ zoom={zoom}"
        row = _build_one_group_panel(map_img, mask, tile_info, affine_H,
                                       left_label=left_label,
                                       right_label=right_label)
        if row is not None:
            rows.append(row)
    if not rows:
        return None
    # Pad rows to common width.
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.full((r.shape[0], max_w - r.shape[1], 3), 220,
                          dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)
        padded.append(np.full((4, max_w, 3), 220, dtype=np.uint8))
    return np.vstack(padded[:-1])


def _stack_candidate_panels(panels: List[np.ndarray]) -> Optional[np.ndarray]:
    """Vertical stack of per-candidate blocks with spacers."""
    panels = [p for p in panels if p is not None]
    if not panels:
        return None
    max_w = max(p.shape[1] for p in panels)
    out = []
    for p in panels:
        if p.shape[1] < max_w:
            pad = np.full((p.shape[0], max_w - p.shape[1], 3), 240,
                          dtype=np.uint8)
            p = np.hstack([p, pad])
        out.append(p)
        out.append(np.full((10, max_w, 3), 80, dtype=np.uint8))  # darker spacer
    big = np.vstack(out[:-1])
    if big.shape[1] > 2000:
        s = 2000 / big.shape[1]
        big = cv2.resize(big, (2000, int(big.shape[0] * s)))
    return big


def _format_metrics_text(state: Any,
                          attempts: List[Dict[str, Any]],
                          committed_id: int) -> str:
    """Per-candidate compact metrics block."""
    lines = ["=== CANDIDATES ==="]
    lines.append(f"  worker's committed_id: {committed_id}")
    lines.append("")
    for a in attempts:
        cid = a.get("candidate_id")
        tag = "  [COMMITTED]" if cid == committed_id else ""
        per_group = a.get("per_group") or []
        # Aggregate per-group axis numbers — show committed-group(s) only
        # if multi-group, else single line.
        for g in per_group:
            mi = g.get("match_info") or {}
            rwd = g.get("reward") or {}
            n_inl = int(mi.get("n_inliers") or 0)
            # reward.to_dict() returns {"axes": {...}}; the per-axis dicts
            # live one level down. Reading them at the top level (old bug)
            # silently returned empty {} → "?" in the metrics block.
            axes = rwd.get("axes") or {}
            rna = axes.get("road_name_agreement") or {}
            sc = axes.get("scale_consistency") or {}
            road_v = rna.get("score") if isinstance(rna, dict) else None
            scale_v = sc.get("score") if isinstance(sc, dict) else None
            road_str = f"{road_v:.2f}" if isinstance(road_v, (int, float)) else "?"
            scale_str = f"{scale_v:.2f}" if isinstance(scale_v, (int, float)) else "?"
            lines.append(
                f"  cand {cid}  group {g.get('area_group','?')}  "
                f"page {g.get('page','?')}  "
                f"n_inliers={n_inl}  "
                f"road_name_agreement={road_str}  "
                f"scale_consistency={scale_str}{tag}"
            )
    return "\n".join(lines)


# ── Critic single-call + rehand ────────────────────────────────────────────


def _run_critic_once(state: Any, model_name: str,
                      message_history: Optional[list] = None) -> tuple:
    """One critic LLM call.

    If ``message_history`` is provided, the critic sees its own prior
    iteration(s) — lets iter 2 reason about whether the worker's
    response to iter 1's directive actually helped.

    Returns (directive, panel, cand_panels, wall_s, in_tokens, out_tokens,
    llm_error, new_history).
    - panel:        the stacked top-N view (for disk save / human audit)
    - cand_panels:  list of (candidate_id, panel) for each shown candidate
                    — same images sent to the LLM as separate inputs
    - new_history:  updated message list for next iteration (None on error)
    """
    attempts = sorted(state.match_attempts.values(),
                       key=lambda a: int(a.get("candidate_id") or 0))
    cr = state.current_result or {}
    committed_id = cr.get("candidate_id")
    if committed_id is None and attempts:
        committed_id = attempts[-1].get("candidate_id")

    # Pick top-3 candidates by total_inliers. Always include the
    # worker's committed candidate (so the critic can decide whether
    # the worker's pick was good vs. a stronger stored alternative),
    # even if its inlier count puts it outside the top-3 by raw inliers.
    total_n_attempts = len(attempts)
    by_inliers = sorted(attempts,
                         key=lambda a: -int(a.get("total_inliers") or 0))
    shown = list(by_inliers[:3])
    shown_ids = {a.get("candidate_id") for a in shown}
    if committed_id is not None and committed_id not in shown_ids:
        committed_attempt = next(
            (a for a in attempts if a.get("candidate_id") == committed_id),
            None)
        if committed_attempt is not None:
            shown.append(committed_attempt)
    # Display order: by candidate_id (stable across iterations).
    shown.sort(key=lambda a: int(a.get("candidate_id") or 0))

    # Build ONE panel per shown candidate (sent as separate images, so
    # the VLM sees each candidate at full resolution rather than
    # downscaled inside a tall vertical stack).
    cand_panels_with_id = []
    for a in shown:
        p = _build_candidate_panel(
            state, a,
            is_committed=(a.get("candidate_id") == committed_id))
        if p is not None:
            cand_panels_with_id.append((a.get("candidate_id"), p))

    # Aggregated stack — for disk save only, not sent to the LLM.
    panel = _stack_candidate_panels([p for _, p in cand_panels_with_id])

    # Use explicit None check — committed_id=0 is a valid value
    # (the first match_at attempt has candidate_id=0). The old
    # 'committed_id or -1' incorrectly treated 0 as missing.
    metrics_text = _format_metrics_text(
        state, shown,
        committed_id if committed_id is not None else -1)
    selection_note = (
        f"Showing top-{len(shown)} of {total_n_attempts} stored candidates "
        f"(ranked by total_inliers; the worker's committed candidate is "
        f"always included). Each candidate is sent as a separate image."
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
    for cid, p in cand_panels_with_id:
        _, buf = cv2.imencode(".png", p)
        user_input.append(
            BinaryContent(data=buf.tobytes(), media_type="image/png"))

    agent = _ensure_agent(model_name)
    in_tokens = 0
    out_tokens = 0
    llm_error: Optional[str] = None
    new_history: Optional[list] = None
    t0 = time.time()
    try:
        from tools.agent._retry import _run_sync_with_retry
        result = _run_sync_with_retry(
            agent, user_input, label="critic",
            message_history=message_history,
        )
        directive = result.output
        try:
            new_history = result.all_messages()
        except Exception:
            new_history = None
        try:
            usage = result.usage()
            # Prefer the modern pydantic-ai field names; fall back to the
            # legacy aliases the rest of the codebase uses elsewhere.
            in_tokens = int(getattr(usage, "input_tokens", None)
                            or getattr(usage, "request_tokens", 0) or 0)
            out_tokens = int(getattr(usage, "output_tokens", None)
                             or getattr(usage, "response_tokens", 0) or 0)
        except Exception:
            pass
    except Exception as e:
        # Surface the failure: emit an approve directive (so the loop
        # exits cleanly) but tag it so downstream can distinguish a
        # genuine approve from a critic-LLM crash.
        llm_error = f"{type(e).__name__}: {str(e)[:100]}"
        directive = CriticDirective(
            # Explicit-None check matches the fix at the metrics-text
            # site; candidate_id=0 is a valid value.
            chosen_candidate_id=(committed_id if committed_id is not None else 0),
            action="approve",
            reasoning=f"CRITIC_LLM_ERROR (treated as approve): {llm_error}",
        )
    wall = time.time() - t0
    return (directive, panel, cand_panels_with_id, wall, in_tokens,
            out_tokens, llm_error, new_history)


def _rehand_to_worker(state: Any,
                      worker_result: Any,
                      directive: CriticDirective,
                      verbose: bool = True) -> Any:
    """Re-invoke worker with a fixed-template user message based on the
    critic's directive. Returns the new pydantic-ai result, or the prior
    result on failure / unknown action.
    """
    from tools.agent.worker_agent import _agent

    action = directive.action
    cr = state.current_result or {}
    worker_committed_id = cr.get("candidate_id")

    if action == "switch":
        chosen = directive.chosen_candidate_id
        if chosen == worker_committed_id:
            if verbose:
                print(f"  critic switch: chose committed_id {chosen} — "
                      f"treating as approve")
            return worker_result
        if chosen not in state.match_attempts:
            if verbose:
                print(f"  critic switch: id {chosen} not in stored "
                      f"candidates, skipping")
            return worker_result
        # Neutral framing — do NOT reveal a "reviewer" / "critic" to the
        # worker. The worker stays opaque to the critic's existence.
        instruction = (
            f"Reconsider your commit. Candidate {chosen} appears to be a "
            f"better match than your committed candidate "
            f"{worker_committed_id} based on visual alignment with the "
            f"planning map. {directive.reasoning}\n\n"
            f"Call commit_match({chosen}) and re-submit BoundaryOutcome."
        )

    elif action == "retry_locate":
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
        return worker_result

    if verbose:
        print(f"  critic rehand: re-invoking worker with action={action}")
    try:
        history = (worker_result.all_messages()
                   if worker_result is not None and
                   hasattr(worker_result, "all_messages") else None)
        from tools.agent._retry import _run_sync_with_retry
        sub_result = _run_sync_with_retry(
            _agent, instruction, deps=state, message_history=history,
            label="critic-rehand",
        )
        # Propagate the new worker outcome into state so downstream sees
        # the latest accepted/last_output/accept_reason.
        try:
            new_outcome = sub_result.output
            if new_outcome is not None:
                state.last_output = new_outcome
                state.accepted = (new_outcome.status in
                                  ("accepted", "district_lookup"))
                state.accept_reason = (
                    f"[{new_outcome.status}] {new_outcome.reasoning[:160]}"
                )
        except Exception:
            pass
        return sub_result
    except Exception as e:
        if verbose:
            print(f"  critic rehand: worker re-invoke failed: {str(e)[:120]}")
        return worker_result


# ── Outer loop ─────────────────────────────────────────────────────────────


def _snapshot_geojson(state: Any) -> Optional[dict]:
    """Snapshot the current committed geojson.

    Returns a deep copy so a later commit_match (which builds a fresh
    state.current_result dict but might be refactored to mutate in
    place) cannot retroactively corrupt the snapshot.
    """
    import copy
    cr = state.current_result or {}
    gj = cr.get("geojson")
    return copy.deepcopy(gj) if gj is not None else None


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
        - critic_final_geojson : geojson AFTER the critic loop
        - iterations           : list of per-iter records (action, reasoning, ...)
        - n_rejections         : count of switch/retry_locate actions issued
        - tokens               : {request, response} totals across critic calls
        - panel_iter0          : first iteration's panel (np.ndarray) for viz
    """
    # Snapshot the worker's first commit BEFORE any critic intervention.
    worker_first_geojson = _snapshot_geojson(state)
    if worker_first_geojson is None:
        if verbose:
            print("  critic: no geojson committed, skipping")
        return {
            "worker_first_geojson": None,
            "critic_final_geojson": None,
            "iterations": [],
            "n_rejections": 0,
            "tokens": {"request": 0, "response": 0},
            "panel_iter0": None,
        }

    iterations: List[Dict[str, Any]] = []
    total_in = 0
    total_out = 0
    panels_by_iter: List[Optional[np.ndarray]] = []
    # per-iteration list of (candidate_id, panel) tuples — same images
    # the LLM saw, saved separately so we can audit exactly what the
    # critic was looking at when it made each decision.
    per_cand_panels_by_iter: List[List[tuple]] = []
    current_worker_result = worker_result
    # Critic's own message history across iterations — lets iter 2
    # reason about whether iter 1's directive was followed and helped.
    # Each iteration's user_input still rebuilds the current panels +
    # metrics from state, so the critic always sees current candidates;
    # the history just adds its own prior reasoning to that context.
    critic_message_history: Optional[list] = None

    for it_idx in range(max_iters):
        (directive, panel, cand_panels, wall, in_t, out_t, llm_error,
         new_history) = _run_critic_once(
             state, model_name, message_history=critic_message_history)
        # Update history for next iter (only if the call succeeded; on
        # LLM error new_history is None and we keep prior history).
        if new_history is not None:
            critic_message_history = new_history
        panels_by_iter.append(panel)
        per_cand_panels_by_iter.append(cand_panels)
        total_in += in_t
        total_out += out_t

        if verbose:
            err_tag = " [LLM_ERROR]" if llm_error else ""
            print(f"  critic iter{it_idx}: action={directive.action}{err_tag}  "
                  f"chose={directive.chosen_candidate_id}  "
                  f"reason={directive.reasoning[:80]!r}  wall={wall:.1f}s")

        iter_entry = {
            "iter_idx": it_idx,
            "action": directive.action,
            "chosen_candidate_id": directive.chosen_candidate_id,
            "reasoning": directive.reasoning,
            "wall_s": round(wall, 1),
            "llm_error": llm_error,  # None on success, error string on crash
        }

        if directive.action == "approve":
            iterations.append(iter_entry)
            break

        # Re-hand to worker.
        before = _snapshot_geojson(state)
        current_worker_result = _rehand_to_worker(
            state, current_worker_result, directive, verbose=verbose
        )
        after = _snapshot_geojson(state)
        iter_entry["geojson_changed"] = before != after
        iterations.append(iter_entry)

        # If state didn't change despite a rehand, the worker probably
        # couldn't comply — break to avoid loop on a stuck case.
        if not iter_entry["geojson_changed"] and directive.action != "approve":
            if verbose:
                print(f"  critic rehand: geojson unchanged after rehand, "
                      f"exiting loop")
            break

    critic_final_geojson = _snapshot_geojson(state)
    n_rej = sum(1 for it in iterations if it["action"] != "approve")

    return {
        "worker_first_geojson": worker_first_geojson,
        "critic_final_geojson": critic_final_geojson,
        "iterations": iterations,
        "n_rejections": n_rej,
        "tokens": {"request": total_in, "response": total_out},
        # Per-iteration stacked panel (for human-friendly debugging) +
        # per-iteration per-candidate panels (the actual images the LLM
        # received). Caller writes both to disk.
        "panels_by_iter": panels_by_iter,
        "per_cand_panels_by_iter": per_cand_panels_by_iter,
        # The latest worker result — carries the full conversation
        # including all critic-triggered rehand sub-turns. Use this for
        # message_log extraction so the on-disk log includes critic
        # interventions; otherwise the rehand turns are lost.
        "final_worker_result": current_worker_result,
    }
