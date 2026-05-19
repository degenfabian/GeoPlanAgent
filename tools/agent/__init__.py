"""Planning-boundary extraction agent — thin orchestrator.

The pipeline is three LLM agents:

  Phase 1 — Reader. One-shot pydantic-ai call (output_type=PDFInfo) over
            the raw PDF + per-page text from fitz/OCR. Each map_page is
            tagged with category (match/discard), area_group, and
            boundary clarity/zoom. Pre-rendered for every match page.

  Phase 2 — Worker. PydanticAI loop with tools registered against
            tools.agent.worker_agent._agent. The canonical loop is:
              propose_centers → match_at(page=X) → commit_match
              → verify_position (when borderline) → submit BoundaryOutcome.
            match_at takes the page explicitly. For multi-area_group
            documents one match_at call handles all groups at the same
            centre (per-group MINIMA + per-page SAM3 cache) and unions
            the resulting polygons; commit_match commits the unioned
            candidate.

The pure functions for each phase live in tools.agent.runtime; this
module is just the coordinator + tool-module import (which triggers the
@_agent.tool decorators at import time so all worker tools are
registered before run_agent is called).
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# Re-exported schemas for callers that import from tools.agent directly.
from tools.agent.schemas import (
    BoundaryOutcome,
    PDFInfo,
)
from tools.agent.state import AgentState

# Importing these modules triggers the `@_agent.tool` decorators inside
# them, registering every worker tool against the shared _agent. Order
# between tool modules doesn't matter, but they MUST be imported AFTER
# tools.agent.worker_agent (the file that defines _agent itself).
from tools.agent.tools import (  # noqa: F401  (decorator side-effects)
    locate as _locate_tool,
    match as _match_tool,
    verify as _verify_tool,
    refine as _refine_tool,
)

from tools.agent import runtime as _rt


def run_agent(
    pdf_path: str,
    models_state: dict,
    model_name: str = "google/gemini-3.1-pro-preview",
    max_iterations: int = 8,
    dpi: int = 200,
    verbose: bool = True,
    case_name: Optional[str] = None,
    case_dir: Optional[Path] = None,
    enable_critic: bool = False,
    critic_max_iters: int = 2,
) -> Dict[str, Any]:
    """Run reader → worker on one planning PDF.

    Args:
        pdf_path: Path to the planning PDF.
        models_state: Dict with sam3_ft (or sam3_base) and minima models.
        model_name: OpenRouter model identifier or alias (gemini-flash, …).
        max_iterations: Soft cap on worker turns. Floor of 25 requests
            covers healthy hard cases.
        dpi: PDF rendering DPI for the planning map pages.
        verbose: Print phase headers and progress.
        case_name: Override for the case identifier (used by k-fold SAM3
            adapter routing). Defaults to the PDF's parent directory.
        case_dir: Where to flush pdf_info.json + partial_state.json on
            crash. Optional.
        enable_critic: If True, after the worker submits, run an independent
            LLM critic that compares all stored match_attempts and may
            instruct the worker to switch or re-locate. The worker's first
            committed polygon is also captured (snapshot) so the
            returned dict carries BOTH no-critic and with-critic outcomes
            from the same run (two-in-one ablation).
        critic_max_iters: max critic-rejection iterations before forcing
            accept. Ignored when enable_critic is False.

    Returns:
        Dict with keys including geojson, mask, match_info, agent_accepted,
        agent_reason, agent_stats, message_log. When enable_critic is
        True, also includes worker_first_geojson (snapshot) so downstream
        can score the no-critic baseline.
    """
    from tools.agent._model import resolve_model_name
    model_name = resolve_model_name(model_name)

    if "sam3_ft" in models_state:
        sam3 = models_state["sam3_ft"]
    elif "sam3_base" in models_state:
        sam3 = models_state["sam3_base"]
    else:
        return {"success": False, "error": "No SAM3 model loaded"}

    # ── Phase 1: read the PDF ──────────────────────────────────────────────
    pdf_info = _rt.read_pdf_phase(pdf_path, model_name, verbose=verbose)

    # Flush pdf_info.json so it survives a Phase 2 crash.
    if case_dir is not None:
        try:
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "pdf_info.json").write_text(
                json.dumps({k: v for k, v in pdf_info.items()
                            if not k.startswith("_")},
                           indent=2, default=str)
            )
        except Exception as _e:
            if verbose:
                print(f"  Warning: failed to flush pdf_info.json: {_e}")

    # ── Phase 2 setup: state + worker user_parts ──────────────────────────
    state, user_parts = _rt.prepare_worker_state(
        pdf_path=pdf_path, sam3=sam3, minima_matcher=models_state["minima"],
        pdf_info=pdf_info, dpi=dpi, case_name=case_name, verbose=verbose,
    )

    if verbose:
        print(f"  Running agent ({model_name}, max {max_iterations} turns)")

    from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded

    # ── Phase 2: invoke the worker ────────────────────────────────────────
    result = None
    outcome: Optional[BoundaryOutcome] = None  # may stay None on exception path
    try:
        result = _rt.invoke_worker(state, user_parts, model_name,
                                     max_iterations, verbose)
    except (UnexpectedModelBehavior, UsageLimitExceeded) as e:
        if verbose:
            print(f"  Agent loop ended: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
        if not state.accepted:
            state.accepted = True
            state.accept_reason = f"Loop ended: {type(e).__name__}"
    except Exception as e:
        if verbose:
            print(f"  Agent error: {e}")
            traceback.print_exc()
        partial_stats = _rt.dump_partial_state(state, pdf_info, e, case_dir, verbose)
        from tools.agent.state import committed_primary_page
        pp = committed_primary_page(state)
        return {
            "success": False,
            "error": str(e),
            "geojson": state.current_result.get("geojson"),
            "match_info": state.current_result.get("match_info", {}),
            "mask": state.sam_masks_by_page.get(pp) if pp else None,
            "agent_stats": partial_stats,
        }
    else:
        outcome = result.output
        state.last_output = outcome
        state.accepted = (outcome.status in ("accepted", "district_lookup"))
        state.accept_reason = f"[{outcome.status}] {outcome.reasoning[:160]}"
        if verbose:
            print(f"  Worker outcome: status={outcome.status} "
                  f"inliers={outcome.final_n_inliers} "
                  f"verify={outcome.verify_position_called} "
                  f"rotation_checked={outcome.rotation_checked}")

    # ── Phase 3 (optional): independent critic loop ───────────────────────
    # We snapshot the worker's first committed polygon BEFORE entering
    # the critic loop, so that even if the critic crashes mid-loop, the
    # no-critic baseline is preserved in the return dict (distinguishes
    # critic-crash from critic-disabled in downstream telemetry).
    critic_result: Optional[Dict[str, Any]] = None
    worker_first_geojson_snapshot: Optional[dict] = None
    can_run_critic = (
        enable_critic
        and state.accepted
        and outcome is not None
        and outcome.status != "district_lookup"
    )
    if can_run_critic:
        worker_first_geojson_snapshot = state.current_result.get("geojson")
        try:
            from tools.agent.critic_agent import run_critic_loop
            if verbose:
                print(f"  Phase 3: running LLM critic loop "
                      f"(max_iters={critic_max_iters})...")
            critic_result = run_critic_loop(
                state, result, model_name=model_name,
                max_iters=critic_max_iters, verbose=verbose,
            )
            if verbose:
                n_rej = critic_result.get("n_rejections", 0)
                its = critic_result.get("iterations") or [{}]
                final_action = its[-1].get("action", "?") if its else "n/a"
                print(f"  Phase 3 done: {n_rej} rejection(s), "
                      f"final_decision={final_action}")
        except Exception as e:
            if verbose:
                print(f"  Phase 3 critic failed: {type(e).__name__}: {e}")
                traceback.print_exc()
            critic_result = {
                "error": str(e)[:200],
                "worker_first_geojson": worker_first_geojson_snapshot,
            }

    # ── Cleanup, stats, soft quality gate, return ─────────────────────────
    _rt.cleanup_temp_pages(state)

    if verbose:
        mi = state.current_result.get("match_info", {})
        print(f"  Agent done: accepted={state.accepted}, "
              f"inliers={mi.get('n_inliers', 0)}, "
              f"reason={state.accept_reason[:100]}")

    message_log = []
    extracted_stats: dict = {}
    if result is not None:
        try:
            message_log, extracted_stats = _rt.extract_message_log(result)
        except Exception:
            pass

    agent_stats = _rt.collect_agent_stats(state, pdf_info, result, extracted_stats)

    # Embed critic telemetry in agent_stats (tokens, iterations, n_rejections).
    # The actual geojsons are passed separately so build_run_agent_return
    # can include them at the top level.
    if critic_result is not None and "error" not in critic_result:
        agent_stats["critic"] = {
            "n_rejections": critic_result.get("n_rejections", 0),
            "iterations": [
                {k: v for k, v in it.items() if k != "panel"}
                for it in critic_result.get("iterations") or []
            ],
            "tokens": critic_result.get("tokens", {}),
        }
    elif critic_result is not None:
        agent_stats["critic"] = {"error": critic_result.get("error")}

    agent_rejected = (state.accept_reason or "").upper().lstrip().startswith("REJECTED")
    _rt.apply_quality_gate(state, agent_rejected, verbose)

    return _rt.build_run_agent_return(
        state, agent_stats, message_log,
        critic_result=critic_result,
    )
