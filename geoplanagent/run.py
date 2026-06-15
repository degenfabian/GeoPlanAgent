"""Per-case orchestration: run_agent drives the reader phase, map-page
rendering with auto-rotation, the worker tool loop, and the optional
critic pass, returning the final GeoJSON plus telemetry.
"""

import copy
import json
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pydantic_ai import BinaryContent
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.usage import UsageLimits
from geoplanagent.schemas import BoundaryOutcome, PDFInfo
from geoplanagent.utils import (
    AgentState,
    img_to_binary,
    run_sync_with_retry,
    resolve_model,
    resolve_model_name,
    result_tokens,
)
from geoplanagent.agents.reader import _reader_agent
from geoplanagent.agents.worker import _worker_agent

# Side-effect import: importing positioning.py runs its @_worker_agent.tool
# decorators, which is what registers the worker's tools on the agent.
# Nothing here uses `positioning` directly, so it looks unused to the linter
# (the trailing noqa) — but deleting it would leave the worker with no tools.
from geoplanagent.tools import positioning as _positioning  # noqa: F401


def _public(pdf_info: dict) -> dict:
    """Drop private (underscore-prefixed) keys such as _reader_tokens."""
    return {k: v for k, v in pdf_info.items() if not k.startswith("_")}


def run_agent(
    pdf_path: str,
    models_state: dict,
    model_name: str = "google/gemini-3-flash-preview",
    max_requests: int = 30,
    dpi: int = 200,
    verbose: bool = True,
    case_name: Optional[str] = None,
    enable_critic: bool = False,
    critic_max_iters: int = 2,
    locate_model_name: str = "google/gemini-3-flash-preview",
    folded: bool = False,
) -> Dict[str, Any]:
    """Run reader → worker (and optionally critic) on one planning PDF. Returns geojson, mask, stats.

    model_name drives the reader, worker, and critic (they share one model);
    locate_model_name is a separate model for the locate sub-agent.

    folded=True runs the folded-reader ablation: a single agent does both
    PDFInfo extraction and positioning. Phase 1 (the dedicated reader
    call) is skipped; the PDF binary is attached to the worker's first
    user message and the worker is forced to call submit_pdf_info before
    any other tool. Everything downstream of pdf_info is identical.
    """
    model_name = resolve_model_name(model_name)

    sam3 = models_state.get("sam3_ft")
    if sam3 is None:
        return _crashed("No SAM3 model loaded")

    # Phase 1: read the PDF (skipped in folded ablation)
    if folded:
        # In folded mode pdf_info is populated by the worker's first tool
        # call (submit_pdf_info); start empty.
        pdf_info: Dict[str, Any] = {}
        state, user_parts = prepare_folded_state(
            pdf_path=pdf_path,
            sam3=sam3,
            minima_matcher=models_state["minima"],
            dpi=dpi,
            case_name=case_name,
            verbose=verbose,
            locate_model_name=locate_model_name,
        )
    else:
        pdf_info = read_pdf_phase(pdf_path, model_name, verbose=verbose)
        if pdf_info.get("error"):
            # The reader couldn't parse the PDF, so the worker has no map
            # pages or location signals to act on — a full worker run would
            # just burn tokens for a guaranteed non-result. Fail fast with a
            # crashed status (carries the reader error) so the case is flagged
            # for a later rerun.
            if verbose:
                print(f"  Reader failed — skipping worker: {pdf_info['error']}")
            return _crashed(f"reader failed: {pdf_info['error']}")

        # Phase 2 setup: state + worker user_parts
        state, user_parts = prepare_worker_state(
            pdf_path=pdf_path,
            sam3=sam3,
            minima_matcher=models_state["minima"],
            pdf_info=pdf_info,
            dpi=dpi,
            case_name=case_name,
            verbose=verbose,
            locate_model_name=locate_model_name,
        )

    if verbose:
        print(f"  Running agent ({model_name}, max {max_requests} requests)")

    # Phase 2: invoke the worker
    result = None
    outcome: Optional[BoundaryOutcome] = None  # may stay None on exception path
    try:
        result = invoke_worker(state, user_parts, model_name, max_requests)
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
        return _crashed(
            str(e),
            geojson=state.current_result.get("geojson"),
            tb=traceback.format_exc(),
        )
    else:
        outcome = result.output
        state.last_output = outcome
        state.accepted = outcome.status in ("accepted", "district_lookup")
        state.accept_reason = f"[{outcome.status}] {outcome.reasoning[:160]}"
        if verbose:
            print(
                f"  Worker outcome: status={outcome.status} "
                f"inliers={outcome.final_n_inliers} "
                f"rotation_checked={outcome.rotation_checked}"
            )

    # Phase 3 (optional): critic loop
    critic_result = run_critic_phase(
        state,
        result,
        outcome,
        enable_critic=enable_critic,
        model_name=model_name,
        critic_max_iters=critic_max_iters,
        verbose=verbose,
    )

    # Stats and return
    if verbose:
        match_info = state.current_result.get("match_info", {})
        print(
            f"  Agent done: accepted={state.accepted}, "
            f"inliers={match_info.get('n_inliers', 0)}, "
            f"reason={state.accept_reason[:100]}"
        )

    # If the critic triggered rehands, the post-critic result has the full
    # conversation including those sub-turns — use it for stats extraction.
    log_source_result = result
    if critic_result is not None and "error" not in critic_result:
        final_worker_result = critic_result.get("final_worker_result")
        if final_worker_result is not None:
            log_source_result = final_worker_result

    agent_stats = collect_agent_stats(state, pdf_info, result, log_source_result, critic_result)

    return build_run_agent_return(
        state,
        agent_stats,
        critic_result=critic_result,
    )


# Phase 1: read the PDF


def read_pdf_phase(pdf_path: str, model_name: str, verbose: bool = True) -> dict:
    """Reader → PDFInfo dict (+ _reader_tokens). {'error': str} on failure (run_agent then fails the case)."""
    pdf_bytes = Path(pdf_path).read_bytes()

    if verbose:
        print(f"  Phase 1: reading PDF ({len(pdf_bytes) // 1024} KB)...")

    model = resolve_model(model_name)

    try:
        result = run_sync_with_retry(
            _reader_agent,
            [
                BinaryContent(data=pdf_bytes, media_type="application/pdf"),
                "Read this UK planning PDF and populate the PDFInfo schema "
                "with all geographic information you can find.",
            ],
            model=model,
            usage_limits=UsageLimits(request_limit=5),
            label="reader",
        )
        info_model: PDFInfo = result.output
        info = info_model.model_dump()

        if verbose:
            print(
                f"  Phase 1: map_pages={info['map_pages']}, "
                f"postcodes={info['postcodes']}, "
                f"roads={len(info['road_names'])}, "
                f"scale={info['scale']}, "
                f"district={info['is_district_wide']}"
            )

        req_tokens, resp_tokens = result_tokens(result)
        info["_reader_tokens"] = {"request": req_tokens, "response": resp_tokens}
        return info

    except UnexpectedModelBehavior as e:
        if verbose:
            print(f"  Phase 1 failed: {e}")
        return {"error": str(e)}


# Phase 2 setup: pre-render map pages, build worker user prompt


def prepare_worker_state(
    pdf_path: str,
    sam3: Dict[str, Any],
    minima_matcher: Any,
    pdf_info: dict,
    dpi: int,
    case_name: Optional[str],
    verbose: bool,
    locate_model_name: str,
) -> Tuple[AgentState, list]:
    """Build AgentState + worker user_parts (summary JSON + primary page image)."""
    state = AgentState(
        pdf_path=str(pdf_path),
        minima_matcher=minima_matcher,
        dpi=dpi,
        sam3_state=sam3,
        case_name=case_name,
        locate_model_name=locate_model_name,
    )
    public_info = _public(pdf_info)
    state.pdf_info = public_info

    map_pages = pdf_info.get("map_pages", []) or []
    map_page_details = pdf_info.get("map_page_details", []) or []

    if map_pages:
        from geoplanagent.tools.pdf import render_map_page

        for page_1based in map_pages:
            rendered = render_map_page(
                str(pdf_path), int(page_1based), dpi=dpi, verbose=verbose, case_name=case_name
            )
            if rendered is None:
                continue
            page_img, rot_info = rendered
            if rot_info.get("applied") and page_1based == map_pages[0]:
                state.rotation_checked = True
            state.rendered_pages[int(page_1based)] = page_img

    summary_text = json.dumps(public_info, indent=2)
    roles_line = _build_map_roles_line(map_page_details, map_pages)
    user_parts: list = [
        f"PDF EXTRACTION SUMMARY:\n{summary_text}\n{roles_line}\n"
        f"Use this information to geolocate and extract the planning boundary. "
        f"Page {map_pages[0] if map_pages else '?'} (the top-ranked match "
        f"page) is pre-rendered as your default working map. For multi-area "
        f"docs, iterate the propose_centers → match_at → commit_match loop "
        f"once per area_group; each commit_match unions its group's polygon "
        f"into the running final result."
    ]
    primary_img = state.rendered_pages.get(int(map_pages[0])) if map_pages else None
    if primary_img is not None:
        user_parts.append(f"Map page {map_pages[0]}:")
        user_parts.append(img_to_binary(primary_img))
    return state, user_parts


def _build_map_roles_line(map_page_details: list, map_pages: list) -> str:
    """Worker-prompt line (verbatim, LLM-visible) describing each match page
    and grouping the pages by area_group, so the worker knows which page to
    pass to match_at per group. Returns "" when there is no map metadata.
    """
    if not map_page_details:
        return ""
    roles = ", ".join(
        f"page {detail.get('page', '?')}=["
        f"{detail.get('category', '?')}, "
        f"grp={detail.get('area_group', '?')}, "
        f"{detail.get('boundary_clarity', '?')}/"
        f"{detail.get('detail_level', '?')}"
        f"] {(detail.get('caption') or '')[:60]!r}"
        for detail in map_page_details
    )
    roles_line = (
        "\nMap-page metadata (only category='match' pages are pre-"
        "rendered; pass the page number you want as match_at's "
        f"`page` argument): {roles}\n"
    )
    by_group: dict[int, list[int]] = {}
    page_to_group = {
        int(detail["page"]): int(detail["area_group"])
        for detail in map_page_details
        if detail.get("category") == "match"
    }
    for page in map_pages:
        group = page_to_group.get(int(page))
        if group is None:
            continue
        by_group.setdefault(group, []).append(int(page))
    if by_group:
        grouped = "; ".join(
            f"Group {group}: pages {pages} (primary={pages[0]}"
            + (f", alternates={pages[1:]}" if len(pages) > 1 else "")
            + ")"
            for group, pages in sorted(by_group.items())
        )
        roles_line += (
            "\nMatch pages by area_group (each match_at call covers "
            "ONE group — iterate propose_centers → match_at → "
            "commit_match per group; to retry a specific group, "
            "pass `page=<next alternate in that group>`): "
            f"{grouped}\n"
        )
    return roles_line


# Folded ablation: no reader phase, worker fills PDFInfo itself


def prepare_folded_state(
    pdf_path: str,
    sam3: Dict[str, Any],
    minima_matcher: Any,
    dpi: int,
    case_name: Optional[str],
    verbose: bool,
    locate_model_name: str,
) -> Tuple[AgentState, list]:
    """Build AgentState + worker user_parts for the folded ablation.

    Skips the dedicated reader phase. The user prompt attaches the raw
    PDF binary; the system prompt requires submit_pdf_info as the first
    tool call. After that tool runs, state.pdf_info is populated and
    state.rendered_pages is filled with the identified map pages — the
    same end-state prepare_worker_state arrives at, just by a different
    route.
    """
    state = AgentState(
        pdf_path=str(pdf_path),
        minima_matcher=minima_matcher,
        dpi=dpi,
        sam3_state=sam3,
        case_name=case_name,
        locate_model_name=locate_model_name,
        folded_mode=True,
    )

    pdf_bytes = Path(pdf_path).read_bytes()
    if verbose:
        print(f"  Folded mode: attaching PDF ({len(pdf_bytes) // 1024} KB), no reader phase.")

    user_parts: list = [
        BinaryContent(data=pdf_bytes, media_type="application/pdf"),
        "The UK planning permission PDF is attached above. Your first "
        "tool call must be submit_pdf_info(info=<PDFInfo>) — read every "
        "page, populate the PDFInfo schema, and submit. Only after that "
        "may you call propose_centers, match_at, commit_match, or "
        "lookup_district. The pipeline always produces a polygon — never "
        "refuse a case.",
    ]
    return state, user_parts


# Phase 2: invoke the worker


def invoke_worker(
    state: AgentState,
    user_parts: list,
    model_name: str,
    max_requests: int,
):
    """Run the worker tool loop. Returns the pydantic-ai result or raises.

    ``max_requests`` is pydantic-ai's request_limit: the cap on worker model
    calls (LLM requests) for the case. Real cases use ~4 (median) to ~14 (max);
    the default leaves comfortable headroom while still bounding a runaway loop.
    """
    model = resolve_model(model_name)
    return run_sync_with_retry(
        _worker_agent,
        user_parts,
        deps=state,
        model=model,
        usage_limits=UsageLimits(request_limit=max_requests),
        label="worker",
    )


# Phase 3 (optional): critic loop


def run_critic_phase(
    state: AgentState,
    result: Any,
    outcome: Optional[BoundaryOutcome],
    enable_critic: bool,
    model_name: str,
    critic_max_iters: int,
    verbose: bool,
) -> Optional[Dict[str, Any]]:
    """Run the LLM critic loop when it is enabled.

    Returns the critic_result dict, or None when the critic is skipped
    (disabled or a district lookup — nothing to review).
    On a critic crash, returns {"error", "worker_first_geojson"} so downstream
    can tell a mid-run critic failure apart from the critic being disabled.
    """
    if not (
        enable_critic
        and state.accepted
        and outcome is not None
        and outcome.status != "district_lookup"
    ):
        return None

    # Deep-copy: protect the snapshot from any future in-place mutation.
    worker_first_geojson_snapshot = copy.deepcopy(state.current_result.get("geojson"))
    try:
        from geoplanagent.agents.critic import run_critic_loop

        if verbose:
            print(f"  Phase 3: running LLM critic loop (max_iters={critic_max_iters})...")
        critic_result = run_critic_loop(
            state,
            result,
            model_name=model_name,
            max_iters=critic_max_iters,
            verbose=verbose,
        )
        if verbose:
            n_rejections = critic_result.get("n_rejections", 0)
            iterations = critic_result.get("iterations") or [{}]
            final_action = iterations[-1].get("action", "?")
            print(f"  Phase 3 done: {n_rejections} rejection(s), final_decision={final_action}")
        return critic_result
    except Exception as e:
        if verbose:
            print(f"  Phase 3 critic failed: {type(e).__name__}: {e}")
            traceback.print_exc()
        return {
            "error": str(e)[:200],
            "worker_first_geojson": worker_first_geojson_snapshot,
        }


# Message log + stats extraction


def extract_agent_stats_from_msgs(messages: list) -> dict:
    """Aggregate lightweight telemetry from a pydantic-ai message list.

    Returns counts only — ``tool_calls`` (name → invocation count),
    ``n_turns`` (number of messages), and ``validator_retries`` (how many
    ModelRetry / validation prompts fired).
    """
    tool_calls: Dict[str, int] = {}
    n_turns = 0
    validator_retries = 0

    for msg in messages:
        n_turns += 1
        for part in getattr(msg, "parts", None) or []:
            kind_lower = getattr(part, "kind", type(part).__name__).lower()
            if "toolcall" in kind_lower:
                name = getattr(part, "tool_name", "?")
                tool_calls[name] = tool_calls.get(name, 0) + 1
            elif kind_lower.startswith("retryprompt"):
                validator_retries += 1

    return {
        "tool_calls": tool_calls,
        "n_turns": n_turns,
        "validator_retries": validator_retries,
    }


def collect_agent_stats(
    state: AgentState,
    pdf_info: dict,
    result: Any,
    log_source_result: Any = None,
    critic_result: Optional[dict] = None,
) -> dict:
    """Assemble the agent_stats dict that benchmark_runner persists.

    ``result`` supplies the worker token usage; ``log_source_result`` is the
    conversation mined for tool-call / turn counts — normally the same object,
    but the critic's post-rehand result when the critic intervened.
    ``critic_result`` (when the critic ran) adds the ``critic`` sub-record.
    """
    agent_stats: dict = {
        "n_commits": state.n_commits,
    }
    reader_tokens = pdf_info.get("_reader_tokens", {}) or {}
    if reader_tokens:
        agent_stats["reader_request_tokens"] = reader_tokens.get("request", 0)
        agent_stats["reader_response_tokens"] = reader_tokens.get("response", 0)

    # Locate sub-agent telemetry
    # Populated by ``geoplanagent.tools.positioning.propose_centers`` via the
    # ``usage_sink=state.locate_calls`` kwarg threaded into ``run_locate``.
    locate_calls = state.locate_calls
    locate_req = sum(int(c.get("request_tokens", 0) or 0) for c in locate_calls)
    locate_resp = sum(int(c.get("response_tokens", 0) or 0) for c in locate_calls)
    agent_stats["locate_n_calls"] = len(locate_calls)
    agent_stats["locate_request_tokens"] = locate_req
    agent_stats["locate_response_tokens"] = locate_resp

    if state.last_output is not None:
        output = state.last_output
        agent_stats["outcome_status"] = output.status
        agent_stats["rotation_checked"] = output.rotation_checked

    if log_source_result is not None:
        try:
            agent_stats.update(extract_agent_stats_from_msgs(log_source_result.all_messages()))
        except Exception:
            pass

    if result is not None:
        worker_req, worker_resp = result_tokens(result)
        agent_stats["worker_request_tokens"] = worker_req
        agent_stats["worker_response_tokens"] = worker_resp
        reader_total = sum(reader_tokens.values()) if reader_tokens else 0
        agent_stats["total_tokens"] = (
            reader_total + worker_req + worker_resp + locate_req + locate_resp
        )

    # Critic sub-record (geojsons go through build_run_agent_return, not here).
    if critic_result is not None and "error" not in critic_result:
        agent_stats["critic"] = {
            "n_rejections": critic_result.get("n_rejections", 0),
            "iterations": list(critic_result.get("iterations") or []),
            "tokens": critic_result.get("tokens", {}),
        }
    elif critic_result is not None:
        agent_stats["critic"] = {"error": critic_result.get("error")}

    return agent_stats


# Return-dict assembly


def _crashed(error: str, geojson: Optional[dict] = None, tb: Optional[str] = None) -> Dict[str, Any]:
    """run_agent return dict for a hard failure (status='crashed') — the
    crashed-path counterpart to build_run_agent_return. ``tb`` is the formatted
    traceback for an actual exception; the guard paths pass None.
    """
    result: Dict[str, Any] = {
        "status": "crashed",
        "error": error,
        "geojson": geojson,
        "agent_stats": {},
    }
    if tb is not None:
        result["traceback"] = tb
    return result


def build_run_agent_return(
    state: AgentState,
    agent_stats: dict,
    critic_result: Optional[dict] = None,
) -> dict:
    """Assemble the dict that run_agent returns to benchmark_runner.

    When critic_result is supplied (enable_critic=True path), the returned
    dict also contains ``worker_first_geojson`` — the polygon at the
    worker's first commit, BEFORE any critic intervention. The top-level
    ``geojson`` always reflects the final state (post-critic if critic ran,
    worker's commit otherwise). This lets downstream score both
    conditions from a single run.
    """
    geojson = state.current_result.get("geojson")
    result_dict: Dict[str, Any] = {
        "status": "ok" if geojson else "no_prediction",
        "geojson": geojson,
        "match_info": state.current_result.get("match_info", {}),
        "agent_reason": state.accept_reason,
        "agent_stats": agent_stats,
    }
    # Surface worker_first_geojson on BOTH the success path AND the
    # critic-error path — lets downstream distinguish "critic ran and
    # produced a paired result" from "critic disabled" without confusing
    # either with a critic crash mid-run.
    if critic_result is not None:
        worker_first = critic_result.get("worker_first_geojson")
        if worker_first is not None:
            result_dict["worker_first_geojson"] = worker_first
    return result_dict
