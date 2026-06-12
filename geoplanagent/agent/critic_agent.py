"""Pairwise LLM critic: reviews stored match_attempts post-worker-commit."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent
from typing import Literal

from geoplanagent.agent._model import resolve_model


# Critic output schema


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
  Each candidate covers exactly ONE area_group and one page. On
  multi-area documents the worker commits group-by-group, producing
  one candidate per (group, attempt); they appear here as separate
  panels.
  Each image is LEFT|RIGHT, with labels acting purely as identifiers
  (numeric metrics live in the text block — see "INTERPRETING THE
  METRICS" below):
    LEFT  = planning map with the SAM mask overlaid in translucent green.
            Label: "CANDIDATE {id} [COMMITTED] group {g} page {p}".
            The COMMITTED tag marks the worker's choice for THIS group
            (on multi-area docs more than one candidate can be tagged).
            "page" = the 1-based PDF page this match was run on.
            "group" = area_group index.
    RIGHT = OS tile render at the matched window, with the projected
            polygon outlined in red. Label: "OS tile @ zoom={z}".
            "OS" = Ordnance Survey, the UK national mapping agency
            whose vector + raster tiles the agent matches against.
            "zoom" = web-mercator tile zoom level (z17 ≈ 0.6m/px,
            z18 ≈ 0.3m/px, z19 ≈ 0.15m/px) — different candidates
            may have matched at different zooms, so each panel
            reports its own.
- Only the TOP-3 candidates by n_inliers are shown, plus the worker's
  committed candidate(s) if they fall outside the top-3 (so you always
  see the worker's pick alongside the strongest alternatives). A note
  at the top of the metrics block tells you how many total candidates
  exist and how many are shown.
- A metrics block accompanying the images, with one line per candidate.
  Each line is:
    "cand {id}  group {g}  page {p}  n_inliers={N}  "
    "road_name_agreement={r}  scale_consistency={s}  [COMMITTED]?"
  The "cand", "group", "page" fields match the corresponding panel-
  label identifiers, so you can map every metrics row back to its
  image. The [COMMITTED] tag appears on the worker's committed
  candidate(s).
- For multi-area documents the worker's committed candidate_ids
  (one per area_group) are listed at the top of the metrics block.

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

INTERPRETING THE METRICS (same tiers the worker uses)

  n_inliers (RANSAC match strength, integer ≥ 0):
    ≥ 100   STRONG     — the affine is almost always correct here.
    50-99   OK         — borderline; lean on the visual panels.
    25-49   WEAK       — likely wrong; needs visual confirmation.
    < 25    TOO WEAK   — almost certainly wrong location.

  scale_consistency (range 0..1):
    ≥ 0.8   GOOD       — recovered scale matches the document's stated scale.
    0.5-0.8 MARGINAL   — scale stretched; suspect alternative is better.
    < 0.5   BAD        — scale very off; trust only if n_inliers ≥ 100.

  road_name_agreement (range 0..1):
    ≥ 0.6   STRONG     — reader's road names found at the matched location.
    0.0     CONFLICT   — OS has roads here but NONE match the reader;
                         possible wrong-area signal. Trust only if
                         n_inliers ≥ 100.
    0.5     NEUTRAL    — verdict says "no OS roads within radius"
                         (sparse cartography); no signal.
    other   PARTIAL    — some roads matched; weak corroboration.

These numbers are supporting evidence — the visual panels are the
primary signal for your decision.

DECISION (pick exactly one action)

- approve
    The worker's committed candidate(s) show clear road/feature
    correspondence AND the polygon outline aligns with the drawn
    boundary. Set chosen_candidate_id = committed_id (use any one of
    the committed ids on multi-area docs; the action applies to the
    whole set).

- switch
    A DIFFERENT stored candidate looks visually better (clearer
    correspondence, better polygon alignment) than the committed one
    FOR ITS area_group. Set chosen_candidate_id to that candidate's
    id; the worker's commit for that group will be replaced. Cite
    the specific visual feature that swayed you.

- retry_locate
    None of the stored candidates show good correspondence — they all
    appear to be in the wrong region (no road / feature match, polygons
    in totally different terrain). The worker will be asked to
    re-locate from a different geocoding signal (postcode vs place vs
    road, etc.). Set chosen_candidate_id = committed_id (placeholder).

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


# Panel-building helpers


def _label_strip(img: np.ndarray, text: str, height: int = 32) -> np.ndarray:
    """Black label strip above img; font auto-shrinks to fit width."""
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
    """One LEFT|RIGHT row: map + SAM mask | OS tiles + projected polygon."""
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
        # LEFT identifies the row; numeric metrics live in the text block.
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
                          committed_ids: set) -> str:
    """Per-candidate metrics block; tags [COMMITTED] for each id in committed_ids."""
    lines = ["=== CANDIDATES ==="]
    if not committed_ids:
        lines.append("  worker's committed_ids: (none — nothing committed yet)")
    elif len(committed_ids) == 1:
        lines.append(f"  worker's committed_id: {next(iter(committed_ids))}")
    else:
        lines.append(
            f"  worker's committed_ids (one per area_group): "
            f"{sorted(committed_ids)}")
    lines.append("")
    for a in attempts:
        cid = a.get("candidate_id")
        tag = "  [COMMITTED]" if cid in committed_ids else ""
        per_group = a.get("per_group") or []
        for g in per_group:
            mi = g.get("match_info") or {}
            rwd = g.get("reward") or {}
            n_inl = int(mi.get("n_inliers") or 0)
            axes = rwd.get("axes") or {}
            rna = axes.get("road_name_agreement") or {}
            sc = axes.get("scale_consistency") or {}
            road_v = rna.get("score") if isinstance(rna, dict) else None
            road_verdict = (rna.get("verdict") if isinstance(rna, dict)
                            else None) or ""
            scale_v = sc.get("score") if isinstance(sc, dict) else None
            road_str = f"{road_v:.2f}" if isinstance(road_v, (int, float)) else "?"
            scale_str = f"{scale_v:.2f}" if isinstance(scale_v, (int, float)) else "?"
            verdict_str = f" ({road_verdict})" if road_verdict else ""
            lines.append(
                f"  cand {cid}  group {g.get('area_group','?')}  "
                f"page {g.get('page','?')}  "
                f"n_inliers={n_inl}  "
                f"road_name_agreement={road_str}{verdict_str}  "
                f"scale_consistency={scale_str}{tag}"
            )
    return "\n".join(lines)


# Critic single-call + rehand


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
    # SET of committed candidate_ids — one per area_group. Single-area
    # docs have one entry; multi-area docs can have several.
    committed_ids: set = set(int(c) for c in (
        getattr(state, "committed_groups", {}) or {}).values())
    # Legacy fallback: if committed_groups wasn't populated (very old
    # state shape), read from current_result["candidate_id"].
    if not committed_ids:
        cr = state.current_result or {}
        if cr.get("candidate_id") is not None:
            committed_ids = {int(cr["candidate_id"])}

    # Rank candidates by n_inliers (each attempt covers one area_group
    # so this is the per-group strength). Tie-break by candidate_id
    # for reproducibility.
    def _attempt_n_inliers(a: Dict[str, Any]) -> int:
        per = a.get("per_group") or [{}]
        mi = (per[0] or {}).get("match_info") or {}
        return int(mi.get("n_inliers") or 0)

    total_n_attempts = len(attempts)
    by_inliers = sorted(
        attempts,
        key=lambda a: (-_attempt_n_inliers(a),
                       int(a.get("candidate_id") or 0)),
    )
    shown = list(by_inliers[:3])
    shown_ids = {a.get("candidate_id") for a in shown}
    # Always include every committed candidate, even if not in top-3.
    for cid in committed_ids:
        if cid in shown_ids:
            continue
        committed_attempt = next(
            (a for a in attempts if a.get("candidate_id") == cid), None)
        if committed_attempt is not None:
            shown.append(committed_attempt)
            shown_ids.add(cid)
    # Display order: by candidate_id (stable across iterations).
    shown.sort(key=lambda a: int(a.get("candidate_id") or 0))

    # Build ONE panel per shown candidate (sent as separate images, so
    # the VLM sees each candidate at full resolution rather than
    # downscaled inside a tall vertical stack).
    cand_panels_with_id = []
    for a in shown:
        p = _build_candidate_panel(
            state, a,
            is_committed=(a.get("candidate_id") in committed_ids))
        if p is not None:
            cand_panels_with_id.append((a.get("candidate_id"), p))

    # Aggregated stack — for disk save only, not sent to the LLM.
    panel = _stack_candidate_panels([p for _, p in cand_panels_with_id])

    metrics_text = _format_metrics_text(state, shown, committed_ids)
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
        from geoplanagent.agent.state import _run_sync_with_retry
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
        # genuine approve from a critic-LLM crash. Pick an arbitrary
        # committed_id when available; fall back to 0 (a valid id) when
        # nothing has been committed yet.
        llm_error = f"{type(e).__name__}: {str(e)[:100]}"
        directive = CriticDirective(
            chosen_candidate_id=(next(iter(committed_ids))
                                  if committed_ids else 0),
            action="approve",
            reasoning=f"CRITIC_LLM_ERROR (treated as approve): {llm_error}",
        )
    wall = time.time() - t0
    return (directive, panel, cand_panels_with_id, wall, in_tokens,
            out_tokens, llm_error, new_history)


def _direct_switch_commit(state: Any,
                           worker_result: Any,
                           chosen_id: int,
                           directive_reasoning: str,
                           verbose: bool = True) -> Any:
    """Commit ``chosen_id`` directly into state — no worker LLM call.

    The critic has already decided which candidate to commit; the
    candidate is already in ``state.match_attempts`` (it was put there
    by an earlier ``match_at`` call). An LLM re-invocation here would
    just be a middleman that types out ``commit_match(N)`` — wasted
    cost and deterministic risk (the worker could rationalise around
    the pick).

    Mutates ``state.current_result`` / ``state.last_output`` /
    ``state.accepted`` / ``state.accept_reason`` to mirror what a
    successful ``commit_match`` would have produced. Returns the
    ORIGINAL ``worker_result`` unchanged — the critic loop breaks on
    ``switch``, so we never need fresh worker messages downstream.
    """
    cand = state.match_attempts.get(int(chosen_id))
    if cand is None:
        if verbose:
            print(f"  direct_switch: id {chosen_id} not in match_attempts")
        return worker_result

    # Strict gate: refuse to commit a candidate that produced no
    # valid affine for any group — mirrors commit_match's check at
    # match.py around the n_groups_committed==0 path. The critic
    # *should* never pick such a candidate (it has no tile_info /
    # rendered panel), but guard against it: a candidate with no
    # geojson would silently null-out state.current_result["geojson"]
    # and crash downstream IoU scoring.
    n_committed = int(cand.get("n_groups_committed") or 0)
    if n_committed == 0 or cand.get("geojson") is None:
        if verbose:
            print(f"  direct_switch: id {chosen_id} has no usable affine / "
                  f"geojson (n_groups_committed={n_committed}); skipping")
        return worker_result

    # Per-group commit: replace the entry for THIS candidate's
    # area_group, then rebuild current_result by re-unioning every
    # committed group's geojson. For single-area docs (99% of cases)
    # this just replaces the one entry. For multi-area docs the
    # critic's switch affects only the group it picked; the other
    # groups stay committed to whatever the worker chose for them.
    from geoplanagent.agent.worker_tools import _recompute_current_result
    group_id = int(cand.get("requested_group", 0))
    state.committed_groups[group_id] = int(chosen_id)
    _recompute_current_result(state)
    # Mirror commit_match's position_calls increment so agent_stats
    # accurately reflects total commits, including critic-driven
    # switches. Without this, the metric undercounts commits whenever
    # the critic flips the worker's pick.
    state.position_calls += 1

    # Synthesise a BoundaryOutcome for downstream consumers that read
    # ``state.last_output``. Preserve the prior status when available
    # — district_lookup outcomes never reach the critic, so this is
    # almost always "accepted", but we keep the carry-forward to be
    # safe against schema changes.
    from geoplanagent.agent.schemas import BoundaryOutcome
    prev = getattr(state, "last_output", None)
    prev_status = prev.status if prev is not None else "accepted"
    new_reasoning = (
        f"[critic-switch to candidate {chosen_id}] "
        f"{(directive_reasoning or '')[:900]}"
    )
    # final_n_inliers reflects the SUM across all committed groups,
    # which is what state.current_result["total_inliers"] holds after
    # _recompute_current_result ran above.
    state.last_output = BoundaryOutcome(
        status=prev_status,
        final_n_inliers=int(state.current_result.get("total_inliers") or 0),
        rotation_checked=state.rotation_checked,
        reasoning=new_reasoning,
    )
    state.accepted = prev_status in ("accepted", "district_lookup")
    state.accept_reason = f"[{prev_status}] {new_reasoning[:160]}"

    if verbose:
        print(f"  direct_switch: committed candidate {chosen_id} "
              f"(sum_n_inliers={state.current_result['total_inliers']}) "
              f"without worker re-invoke")
    return worker_result


def _rehand_to_worker(state: Any,
                      worker_result: Any,
                      directive: CriticDirective,
                      verbose: bool = True) -> Any:
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
    from geoplanagent.agent.worker_agent import _agent

    action = directive.action
    cr = state.current_result or {}
    worker_committed_id = cr.get("candidate_id")

    rehand_error: Optional[str] = None
    if action == "switch":
        chosen = directive.chosen_candidate_id
        if chosen == worker_committed_id:
            if verbose:
                print(f"  critic switch: chose committed_id {chosen} — "
                      f"treating as approve")
            return worker_result, None
        if chosen not in state.match_attempts:
            if verbose:
                print(f"  critic switch: id {chosen} not in stored "
                      f"candidates, skipping")
            return worker_result, None
        new_result = _direct_switch_commit(
            state, worker_result, chosen,
            directive.reasoning or "", verbose=verbose,
        )
        return new_result, None

    elif action == "retry_locate":
        # Refill the worker's match_at budget and partial-clear the
        # dedup set so the rehand can actually call propose_centers +
        # match_at again. Without this, a worker that exhausted its
        # 5-call budget during the first pass crashes on its next
        # match_at, the exception is swallowed below, and the critic's
        # intended fix is lost with no on-disk trace.
        state.match_at_budget = max(state.match_at_budget, 0) + 2
        state.recent_calls = set()
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
        history = (worker_result.all_messages()
                   if worker_result is not None and
                   hasattr(worker_result, "all_messages") else None)
        from geoplanagent.agent.state import _run_sync_with_retry
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
        return sub_result, None
    except Exception as e:
        rehand_error = f"{type(e).__name__}: {str(e)[:160]}"
        if verbose:
            print(f"  critic rehand: worker re-invoke failed: {rehand_error}")
        return worker_result, rehand_error


# Outer loop


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

        # Apply the directive to state (switch = direct Python commit,
        # no worker LLM; retry_locate = worker re-invoked).
        before = _snapshot_geojson(state)
        current_worker_result, rehand_error = _rehand_to_worker(
            state, current_worker_result, directive, verbose=verbose
        )
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
        if not iter_entry["geojson_changed"] and directive.action != "approve":
            if verbose:
                print("  critic rehand: geojson unchanged after rehand, "
                      "exiting loop")
            break

    critic_final_geojson = _snapshot_geojson(state)
    # A genuine rejection is any iteration where the critic asked us
    # to change state. The crashed-LLM path synthesises a directive
    # with action="approve" + llm_error set; we must NOT count those
    # as approvals, otherwise a run where 100% of critic calls
    # crashed would report n_rejections=0 (= 100% agreement), which
    # is the opposite of reality.
    n_rej = sum(1 for it in iterations
                 if it["action"] != "approve" or it.get("llm_error"))

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
