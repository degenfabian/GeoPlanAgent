"""Phase orchestration for one case: render + rotate the map pages, run the
Phase-1 reader (defined at the bottom of this module), then drive the
worker loop and the optional critic pass.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import cv2
from pydantic_ai import BinaryContent
from pydantic_ai.usage import UsageLimits
from tools.agent.state import _img_to_binary
from tools.agent._model import resolve_model
from tools.agent.state import _run_sync_with_retry
from tools.agent.worker_agent import _agent
from tools.agent.schemas import PDFInfo
from tools.agent.state import AgentState
from dotenv import load_dotenv
from pydantic_ai import Agent
from tools.agent.prompts import READER_SYSTEM_PROMPT


# Phase 1: read the PDF

def read_pdf_phase(pdf_path: str, model_name: str, verbose: bool = True) -> dict:
    """Reader → PDFInfo dict (+ _reader_tokens). Empty dict + 'error' on failure."""
    pdf_bytes = Path(pdf_path).read_bytes()

    if verbose:
        print(f"  Phase 1: reading PDF ({len(pdf_bytes) // 1024} KB)...")

    model = resolve_model(model_name)

    from pydantic_ai.exceptions import UnexpectedModelBehavior

    try:
        result = _run_sync_with_retry(
            _reader_agent,
            [
                BinaryContent(data=pdf_bytes, media_type='application/pdf'),
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
            print(f"  Phase 1: map_pages={info['map_pages']}, "
                  f"postcodes={info['postcodes']}, "
                  f"roads={len(info['road_names'])}, "
                  f"scale={info['scale']}, "
                  f"district={info['is_district_wide']}")

        usage = result.usage()
        info["_reader_tokens"] = {
            "request": usage.request_tokens,
            "response": usage.response_tokens,
        }
        return info

    except UnexpectedModelBehavior as e:
        if verbose:
            print(f"  Phase 1 failed: {e}")
        empty = PDFInfo().model_dump()
        empty["error"] = str(e)
        return empty


# Phase 2 setup: pre-render map pages, build worker user prompt

def prepare_worker_state(
    pdf_path: str,
    sam3: Dict[str, Any],
    minima_matcher: Any,
    pdf_info: dict,
    dpi: int,
    case_name: Optional[str],
    verbose: bool,
    locate_model: str = "google/gemini-3-flash-preview",
    locate_disabled_tools: frozenset = frozenset(
        {"postcode", "grid_ref", "road", "intersect", "la_check"}
    ),
) -> Tuple[AgentState, list]:
    """Build AgentState + worker user_parts (summary JSON + primary page image)."""
    state = AgentState(
        pdf_path=str(pdf_path),
        sam3_processor=sam3["processor"],
        sam3_model=sam3["model"],
        device=sam3["device"],
        minima_matcher=minima_matcher,
        dpi=dpi,
        sam3_state=sam3,
        case_name=case_name,
        locate_model=locate_model,
        locate_disabled_tools=locate_disabled_tools,
    )
    state.pdf_info = {k: v for k, v in pdf_info.items() if not k.startswith("_")}

    map_pages = pdf_info.get("map_pages", []) or []
    map_page_details = pdf_info.get("map_page_details", []) or []

    if map_pages:
        from tools.io.pdf import render_map_page
        for page_1based in map_pages:
            rendered = render_map_page(str(pdf_path), int(page_1based),
                                         dpi=dpi, verbose=verbose,
                                         case_name=case_name)
            if rendered is None:
                continue
            page_img, rot_info = rendered
            if rot_info.get("applied") and page_1based == map_pages[0]:
                state.rotation_checked = True
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            cv2.imwrite(tmp_path, page_img)
            state.rendered_pages[int(page_1based)] = page_img
            state.rendered_page_paths[int(page_1based)] = tmp_path

    summary_text = json.dumps(
        {k: v for k, v in pdf_info.items() if not k.startswith("_")},
        indent=2)
    roles_line = ""
    if map_page_details:
        roles = ", ".join(
            f"page {d.get('page', '?')}=["
            f"{d.get('category', '?')}, "
            f"grp={d.get('area_group', '?')}, "
            f"{d.get('boundary_clarity', '?')}/"
            f"{d.get('detail_level', '?')}"
            f"] {(d.get('caption') or '')[:60]!r}"
            for d in map_page_details
        )
        roles_line = (
            "\nMap-page metadata (only category='match' pages are pre-"
            "rendered; pass the page number you want as match_at's "
            f"`page` argument): {roles}\n"
        )
        by_group: dict[int, list[int]] = {}
        page_to_group = {int(d["page"]): int(d.get("area_group", 0))
                         for d in map_page_details
                         if d.get("category") == "match"}
        for p in map_pages:
            g = page_to_group.get(int(p))
            if g is None:
                continue
            by_group.setdefault(g, []).append(int(p))
        if by_group:
            grouped = "; ".join(
                f"Group {g}: pages {pages} (primary={pages[0]}"
                + (f", alternates={pages[1:]}" if len(pages) > 1 else "")
                + ")"
                for g, pages in sorted(by_group.items())
            )
            roles_line += (
                "\nMatch pages by area_group (each match_at call covers "
                "ONE group — iterate propose_centers → match_at → "
                "commit_match per group; to retry a specific group, "
                "pass `page=<next alternate in that group>`): "
                f"{grouped}\n"
            )
    user_parts: list = [
        f"PDF EXTRACTION SUMMARY:\n{summary_text}\n{roles_line}\n"
        f"Use this information to geolocate and extract the planning boundary. "
        f"Page {map_pages[0] if map_pages else '?'} (the top-ranked match "
        f"page) is pre-rendered as your default working map. For multi-area "
        f"docs, iterate the propose_centers → match_at → commit_match loop "
        f"once per area_group; each commit_match unions its group's polygon "
        f"into the running final result."
    ]
    primary_img = (state.rendered_pages.get(int(map_pages[0]))
                   if map_pages else None)
    if primary_img is not None:
        user_parts.append(f"Map page {map_pages[0]}:")
        user_parts.append(_img_to_binary(primary_img))
    return state, user_parts


# Folded ablation: no reader phase, worker fills PDFInfo itself

def prepare_folded_state(
    pdf_path: str,
    sam3: Dict[str, Any],
    minima_matcher: Any,
    dpi: int,
    case_name: Optional[str],
    verbose: bool,
    locate_model: str = "google/gemini-3-flash-preview",
    locate_disabled_tools: frozenset = frozenset(),
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
        sam3_processor=sam3["processor"],
        sam3_model=sam3["model"],
        device=sam3["device"],
        minima_matcher=minima_matcher,
        dpi=dpi,
        sam3_state=sam3,
        case_name=case_name,
        locate_model=locate_model,
        locate_disabled_tools=locate_disabled_tools,
        folded_mode=True,
    )

    pdf_bytes = Path(pdf_path).read_bytes()
    if verbose:
        print(f"  Folded mode: attaching PDF ({len(pdf_bytes) // 1024} KB), "
              f"no reader phase.")

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
    state: AgentState, user_parts: list, model_name: str,
    max_iterations: int, verbose: bool,
):
    """Run the worker tool loop. Returns the pydantic-ai result or raises."""
    model = resolve_model(model_name)
    return _run_sync_with_retry(
        _agent,
        user_parts,
        deps=state,
        model=model,
        usage_limits=UsageLimits(request_limit=max(max_iterations * 4, 25)),
        label="worker",
    )


# Phase 2 error path: dump partial state

def dump_partial_state(state: AgentState, pdf_info: dict, exc: Exception,
                        case_dir: Optional[Path], verbose: bool) -> dict:
    """Write partial_state.json for post-hoc debug on a mid-run worker error."""
    partial_stats = {
        "pdf_info": state.pdf_info,
        "method": "error",
        "error": str(exc),
        "error_type": type(exc).__name__,
        "current_match_info": state.current_result.get("match_info", {}),
        "match_attempts_summary": {
            cid: {
                "name": a.get("name"),
                "lat": a.get("lat"), "lon": a.get("lon"),
                "area_group": a.get("requested_group"),
                "page": a.get("requested_page"),
                "n_inliers": (
                    ((a.get("per_group") or [{}])[0].get("match_info") or {})
                    .get("n_inliers")
                ),
            }
            for cid, a in (state.match_attempts or {}).items()
        },
        "position_calls": state.position_calls,
        "rotation_checked": state.rotation_checked,
        "last_output": (state.last_output.model_dump()
                        if state.last_output is not None else None),
    }
    if case_dir is not None:
        try:
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "partial_state.json").write_text(
                json.dumps(partial_stats, indent=2, default=str)
            )
        except Exception as _e:
            if verbose:
                print(f"  Warning: failed to flush partial_state.json: {_e}")
    return partial_stats


# Cleanup

def cleanup_temp_pages(state: AgentState) -> None:
    """Unlink every pre-rendered page tempfile."""
    seen_paths = set()
    for p in state.rendered_page_paths.values():
        if not p or p in seen_paths:
            continue
        seen_paths.add(p)
        try:
            os.unlink(p)
        except OSError:
            pass
        try:
            for rot_path in Path(p).parent.glob(Path(p).stem + "_rot*.png"):
                try:
                    rot_path.unlink()
                except OSError:
                    pass
        except OSError:
            pass


# Message log + stats extraction

def extract_message_log(result: Any) -> Tuple[list, dict]:
    return extract_message_log_from_msgs(result.all_messages())


def extract_message_log_from_msgs(messages: list) -> Tuple[list, dict]:
    """Return (message_log, stats) for a pydantic-ai message list.

    stats keys: tool_calls, total_tool_calls, n_turns, validator_retries.
    Binary content is summarised so the log is JSON-safe.
    """
    message_log: list = []
    tool_calls: Dict[str, int] = {}
    turn_idx = 0

    for msg in messages:
        role = getattr(msg, 'kind', type(msg).__name__)
        parts = getattr(msg, 'parts', None)
        if not parts:
            turn_idx += 1
            continue
        for part in parts:
            kind = getattr(part, 'kind', type(part).__name__)
            kind_lower = kind.lower()
            entry = {"turn": turn_idx, "role": role, "kind": kind}

            if 'toolcall' in kind_lower:
                name = getattr(part, 'tool_name', '?')
                tool_calls[name] = tool_calls.get(name, 0) + 1
                entry["tool"] = name
                entry["args"] = _coerce_args(getattr(part, 'args', None))

            elif 'toolreturn' in kind_lower:
                entry["tool"] = getattr(part, 'tool_name', '?')
                entry["return"] = _coerce_return(getattr(part, 'content', None))

            elif 'retry' in kind_lower:
                rc = getattr(part, 'content', None)
                entry["retry_content"] = str(rc)[:1000] if rc else ""

            elif 'userprompt' in kind_lower:
                c = getattr(part, 'content', None)
                if isinstance(c, list):
                    n_images = sum(1 for x in c if hasattr(x, 'media_type'))
                    n_text = sum(1 for x in c if isinstance(x, str))
                    entry["user_summary"] = f"{n_text} text + {n_images} images"
                elif isinstance(c, str):
                    entry["text"] = c[:500]

            elif 'text' in kind_lower or 'thinking' in kind_lower:
                entry["text"] = str(getattr(part, 'content', ''))[:2000]

            message_log.append(entry)
        turn_idx += 1

    validator_retries = 0
    for e in message_log:
        if e.get("kind", "").lower().startswith("retryprompt"):
            validator_retries += 1

    extracted = {
        "tool_calls": tool_calls,
        "total_tool_calls": sum(tool_calls.values()),
        "n_turns": turn_idx,
        "validator_retries": validator_retries,
    }
    return message_log, extracted


def _coerce_args(args: Any) -> Any:
    if args is None:
        return {}
    if isinstance(args, dict):
        return {k: (v if not isinstance(v, (bytes, bytearray))
                    else f"<bytes:{len(v)}>")
                for k, v in args.items()}
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else str(args)[:500]
        except Exception:
            return args[:500]
    return str(args)[:500]


def _coerce_return(content: Any) -> Any:
    if isinstance(content, dict):
        return {k: (v if not isinstance(v, (bytes, bytearray))
                    else f"<bytes:{len(v)}>")
                for k, v in content.items()}
    if isinstance(content, str):
        return content[:1000]
    return str(content)[:1000]


def collect_agent_stats(
    state: AgentState, pdf_info: dict, result: Any,
    message_log_extracted: Optional[dict] = None,
) -> dict:
    """Assemble the agent_stats dict that benchmark_runner persists."""
    agent_stats: dict = {
        "position_calls": state.position_calls,
        "pdf_info": {k: v for k, v in pdf_info.items() if not k.startswith("_")},
    }
    reader_tokens = pdf_info.get("_reader_tokens", {}) or {}
    if reader_tokens:
        agent_stats["reader_request_tokens"] = reader_tokens.get("request", 0)
        agent_stats["reader_response_tokens"] = reader_tokens.get("response", 0)

    # Locate sub-agent telemetry
    # Populated by ``tools.agent.tools.locate.propose_centers`` via the
    # ``usage_sink=state.locate_calls`` kwarg threaded into ``run_locate``.
    # On runs that pre-date the telemetry patch, ``state.locate_calls`` is
    # absent / empty and these fields stay 0 — safe to compare with
    # legacy ``metrics.json`` files.
    locate_calls = getattr(state, "locate_calls", None) or []
    locate_req = sum(int(c.get("request_tokens", 0) or 0) for c in locate_calls)
    locate_resp = sum(int(c.get("response_tokens", 0) or 0) for c in locate_calls)
    agent_stats["locate_n_calls"] = len(locate_calls)
    agent_stats["locate_request_tokens"] = locate_req
    agent_stats["locate_response_tokens"] = locate_resp
    if locate_calls:
        # Keep per-call records (small) so the audit script can query
        # OpenRouter's /v1/generation?id=<gen_id> for exact billed cost.
        agent_stats["locate_calls"] = locate_calls

    if state.last_output is not None:
        out = state.last_output
        agent_stats["outcome_status"] = out.status
        agent_stats["outcome_reasoning"] = out.reasoning
        agent_stats["rotation_checked"] = out.rotation_checked

    if message_log_extracted:
        agent_stats.update(message_log_extracted)

    if result is not None:
        try:
            usage = result.usage()
            agent_stats["worker_request_tokens"] = usage.request_tokens
            agent_stats["worker_response_tokens"] = usage.response_tokens
            reader_total = sum(reader_tokens.values()) if reader_tokens else 0
            worker_total = (usage.request_tokens or 0) + (usage.response_tokens or 0)
            # NB: totals now INCLUDE locate_* if telemetry is present.
            # Old cached metrics.json files that pre-date this patch will
            # have locate_* = 0, so their totals are reader + worker only
            # (matching the paper's $/doc computation).
            agent_stats["request_tokens"] = (
                (reader_tokens.get("request", 0) or 0)
                + (usage.request_tokens or 0)
                + locate_req)
            agent_stats["response_tokens"] = (
                (reader_tokens.get("response", 0) or 0)
                + (usage.response_tokens or 0)
                + locate_resp)
            agent_stats["total_tokens"] = (reader_total + worker_total
                                            + locate_req + locate_resp)
        except Exception:
            pass
    return agent_stats


# Return-dict assembly

def build_run_agent_return(
    state: AgentState,
    agent_stats: dict,
    message_log: list,
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
    from tools.agent.state import committed_primary_page
    primary_page = committed_primary_page(state)
    primary_img = state.rendered_pages.get(primary_page) if primary_page else None
    primary_mask = state.sam_masks_by_page.get(primary_page) if primary_page else None
    primary_overlay = None
    if primary_img is not None and primary_mask is not None:
        sel = primary_img.copy()
        sel[primary_mask > 0] = [0, 255, 0]
        primary_overlay = cv2.addWeighted(primary_img, 0.5, sel, 0.5, 0)
    out: Dict[str, Any] = {
        "success": True,
        "geojson": state.current_result.get("geojson"),
        "match_info": state.current_result.get("match_info", {}),
        "mask": primary_mask,
        "affine_H": state.current_result.get("affine_H"),
        "tile_info_meta": {
            k: v for k, v in (state.current_result.get("tile_info") or {}).items()
            if k != "image"
        },
        "agent_accepted": state.accepted,
        "agent_reason": state.accept_reason,
        "agent_stats": agent_stats,
        "message_log": message_log,
        "selected_overlay": primary_overlay,
    }
    # Surface worker_first_geojson on BOTH the success path AND the
    # critic-error path — lets downstream distinguish "critic ran and
    # produced a paired result" from "critic disabled" without confusing
    # either with a critic crash mid-run.
    if critic_result is not None:
        wf = critic_result.get("worker_first_geojson")
        if wf is not None:
            out["worker_first_geojson"] = wf
    return out


load_dotenv()

# Production runs at temperature 0 for reproducibility; the
# GEOMAP_TEMPERATURE env var lets the appendix ablation re-run at 1.0
# without disturbing the cached benchmarks.
_TEMPERATURE = float(os.environ.get("GEOMAP_TEMPERATURE", "0"))


_reader_agent = Agent(
    "test",  # placeholder, overridden at runtime via model= kwarg
    output_type=PDFInfo,
    retries=2,
    output_retries=2,
    model_settings={"temperature": _TEMPERATURE},
    instructions=READER_SYSTEM_PROMPT,
)
