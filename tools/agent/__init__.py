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

  Phase 3 — Critic. tools.agent.critic_agent.run_critic_loop: LLM visual review
            + structured directive, optionally rehanded to the worker.
            Off by default for production benchmark runs; opt-in for
            ablations via enable_critic=True.

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
    max_iterations: int = 6,
    dpi: int = 200,
    verbose: bool = True,
    enable_critic: bool = False,
    case_name: Optional[str] = None,
    case_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run reader → worker → (optional) critic on one planning PDF.

    Args:
        pdf_path: Path to the planning PDF.
        models_state: Dict with sam3_ft (or sam3_base) and minima models.
        model_name: OpenRouter model identifier or alias (gemini-flash, …).
        max_iterations: Soft cap on worker turns. Floor of 25 requests
            covers healthy hard cases.
        dpi: PDF rendering DPI for the planning map pages.
        verbose: Print phase headers and progress.
        enable_critic: Run the Phase 3 critic loop after the worker
            finishes. Off by default; opt-in for ablations.
        case_name: Override for the case identifier (used by k-fold SAM3
            adapter routing). Defaults to the PDF's parent directory.
        case_dir: Where to flush pdf_info.json + partial_state.json on
            crash. Optional.

    Returns:
        Dict with keys including geojson, mask, match_info, agent_accepted,
        agent_reason, agent_stats, message_log, critic_*.
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
        outcome: BoundaryOutcome = result.output
        state.last_output = outcome
        state.accepted = (outcome.status in ("accepted", "district_lookup"))
        state.accept_reason = f"[{outcome.status}] {outcome.reasoning[:160]}"
        if verbose:
            print(f"  Worker outcome: status={outcome.status} "
                  f"inliers={outcome.final_n_inliers} "
                  f"verify={outcome.verify_position_called} "
                  f"rotation_checked={outcome.rotation_checked}")

    # ── Phase 3: critic (optional) ────────────────────────────────────────
    critic_result = None
    if enable_critic and result is not None:
        critic_result = _rt.apply_critic_loop(
            state=state, worker_result=result, model_name=model_name,
            verbose=verbose,
        )

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

    agent_rejected = (state.accept_reason or "").upper().lstrip().startswith("REJECTED")
    _rt.apply_quality_gate(state, agent_rejected, verbose)

    return _rt.build_run_agent_return(state, agent_stats, message_log, critic_result)
