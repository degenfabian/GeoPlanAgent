"""run_agent: reader → worker (→ optional critic) over one planning PDF."""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from tools.agent.schemas import (
    BoundaryOutcome,
    PDFInfo,
)
from tools.agent.state import AgentState

# Imports trigger @_agent.tool decorator side-effects; must come after worker_agent.
from tools.agent.tools import (  # noqa: F401
    locate as _locate_tool,
    match as _match_tool,
    submit as _submit_tool,
    verify as _verify_tool,
)

from tools.agent import runtime as _rt


def run_agent(
    pdf_path: str,
    models_state: dict,
    model_name: str = "google/gemini-3.1-pro-preview",
    max_iterations: int = 12,
    dpi: int = 200,
    verbose: bool = True,
    case_name: Optional[str] = None,
    case_dir: Optional[Path] = None,
    enable_critic: bool = False,
    critic_max_iters: int = 2,
    locate_model: str = "google/gemini-3-flash-preview",
    locate_disabled_tools: frozenset = frozenset(
        {"postcode", "grid_ref", "road", "intersect", "la_check"}
    ),
    folded: bool = False,
) -> Dict[str, Any]:
    """Run reader → worker on one planning PDF. Returns geojson, mask, stats.

    folded=True runs the folded-reader ablation: a single agent does both
    PDFInfo extraction and positioning. Phase 1 (the dedicated reader
    call) is skipped; the PDF binary is attached to the worker's first
    user message and the worker is forced to call submit_pdf_info before
    any other tool. Everything downstream of pdf_info is identical.
    """
    from tools.agent._model import resolve_model_name
    model_name = resolve_model_name(model_name)

    if "sam3_ft" in models_state:
        sam3 = models_state["sam3_ft"]
    elif "sam3_base" in models_state:
        sam3 = models_state["sam3_base"]
    else:
        return {"success": False, "error": "No SAM3 model loaded"}

    # ── Phase 1: read the PDF (skipped in folded ablation) ───────────────
    if folded:
        # In folded mode pdf_info is populated by the worker's first tool
        # call (submit_pdf_info); start empty.
        pdf_info: Dict[str, Any] = {}
        state, user_parts = _rt.prepare_folded_state(
            pdf_path=pdf_path, sam3=sam3,
            minima_matcher=models_state["minima"],
            dpi=dpi, case_name=case_name, verbose=verbose,
            locate_model=locate_model,
            locate_disabled_tools=locate_disabled_tools,
        )
    else:
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

        # ── Phase 2 setup: state + worker user_parts ──────────────────────
        state, user_parts = _rt.prepare_worker_state(
            pdf_path=pdf_path, sam3=sam3, minima_matcher=models_state["minima"],
            pdf_info=pdf_info, dpi=dpi, case_name=case_name, verbose=verbose,
            locate_model=locate_model,
            locate_disabled_tools=locate_disabled_tools,
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
                  f"rotation_checked={outcome.rotation_checked}")

    # ── Phase 3 (optional): critic loop ───────────────────────────────────
    # Snapshot the worker's first commit so critic-crash and critic-disabled
    # stay distinguishable downstream.
    critic_result: Optional[Dict[str, Any]] = None
    worker_first_geojson_snapshot: Optional[dict] = None
    can_run_critic = (
        enable_critic
        and state.accepted
        and outcome is not None
        and outcome.status != "district_lookup"
    )
    if can_run_critic:
        # Deep-copy: protect the snapshot from any future in-place mutation.
        import copy as _copy
        worker_first_geojson_snapshot = _copy.deepcopy(
            state.current_result.get("geojson"))
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

    # Write critic-panel PNGs to disk for post-hoc debugging.
    if (critic_result is not None
            and "error" not in critic_result
            and case_dir is not None):
        try:
            import cv2 as _cv2
            case_dir.mkdir(parents=True, exist_ok=True)
            for _i, _p in enumerate(critic_result.get("panels_by_iter") or []):
                if _p is None:
                    continue
                _path = case_dir / f"critic_panel_iter{_i}.png"
                _cv2.imwrite(str(_path), _p)
            for _i, _cands in enumerate(
                    critic_result.get("per_cand_panels_by_iter") or []):
                for _cid, _cp in _cands or []:
                    if _cp is None:
                        continue
                    _path = case_dir / f"critic_panel_iter{_i}_cand{_cid}.png"
                    _cv2.imwrite(str(_path), _cp)
        except Exception as _e:
            if verbose:
                print(f"  Warning: failed to save critic panels: {_e}")

    # ── Cleanup, stats, soft quality gate, return ─────────────────────────
    _rt.cleanup_temp_pages(state)

    # In folded mode pdf_info was populated by the worker's submit_pdf_info
    # tool call rather than Phase 1. Pull it back from state so downstream
    # stats + the on-disk pdf_info.json reflect what the worker submitted.
    if folded:
        pdf_info = dict(state.pdf_info or {})
        if case_dir is not None and pdf_info:
            try:
                case_dir.mkdir(parents=True, exist_ok=True)
                (case_dir / "pdf_info.json").write_text(
                    json.dumps(pdf_info, indent=2, default=str)
                )
            except Exception as _e:
                if verbose:
                    print(f"  Warning: failed to flush pdf_info.json "
                          f"(folded): {_e}")

    if verbose:
        mi = state.current_result.get("match_info", {})
        print(f"  Agent done: accepted={state.accepted}, "
              f"inliers={mi.get('n_inliers', 0)}, "
              f"reason={state.accept_reason[:100]}")

    # If the critic triggered rehands, the post-critic result has the full
    # conversation including those sub-turns — use it for log extraction.
    log_source_result = result
    if critic_result is not None and "error" not in critic_result:
        final_wr = critic_result.get("final_worker_result")
        if final_wr is not None:
            log_source_result = final_wr

    message_log = []
    extracted_stats: dict = {}
    if log_source_result is not None:
        try:
            message_log, extracted_stats = _rt.extract_message_log(log_source_result)
        except Exception:
            pass

    agent_stats = _rt.collect_agent_stats(state, pdf_info, result, extracted_stats)

    # Critic telemetry into agent_stats; geojsons go through build_run_agent_return.
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

    return _rt.build_run_agent_return(
        state, agent_stats, message_log,
        critic_result=critic_result,
    )
