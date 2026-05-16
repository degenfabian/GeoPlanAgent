"""Runtime helpers for the main run_agent orchestrator.

run_agent itself is a thin coordinator in tools.agent.__init__; the
phase-specific logic (reader call, pre-render, prompt build, message
log assembly, critic, cleanup, return-dict packaging) lives here so
the orchestrator stays under 100 lines and each phase is independently
testable.
"""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from pydantic_ai import BinaryContent
from pydantic_ai.usage import UsageLimits

from tools.agent._helpers import _img_to_binary
from tools.agent._model import resolve_model
from tools.agent._retry import _run_sync_with_retry
from tools.agent.agents import _agent, _reader_agent
from tools.agent.schemas import BoundaryOutcome, PDFInfo
from tools.agent.state import AgentState


# ── Phase 1: read the PDF ─────────────────────────────────────────────────

def read_pdf_phase(pdf_path: str, model_name: str, verbose: bool = True) -> dict:
    """Send the PDF to the reader agent, get a structured PDFInfo dict.

    Augments the prompt with per-page text extracted via fitz (digital
    pages) or OCR (scanned pages, ~60% of the eval). Gemini still does
    its own PDF processing for vision-required fields (boundary_color,
    rotation, map labels), but exact-string fields benefit from being
    given the ground-truth text. Text extraction is cached on disk under
    cache/text_extraction/.

    Returns the PDFInfo dict plus a "_reader_tokens" key. On reader
    failure: an empty PDFInfo dict with an "error" key set.
    """
    pdf_bytes = Path(pdf_path).read_bytes()

    if verbose:
        print(f"  Phase 1: reading PDF ({len(pdf_bytes) // 1024} KB)...")

    try:
        from tools.io.text_extraction import (extract_text_per_page,
                                                format_for_reader_prompt)
        page_texts = extract_text_per_page(pdf_path, use_cache=True, verbose=verbose)
        text_block = format_for_reader_prompt(page_texts)
        if verbose:
            methods: Dict[str, int] = {}
            for p in page_texts:
                methods[p["method"]] = methods.get(p["method"], 0) + 1
            print(f"  Phase 1: text extraction {dict(methods)}, "
                  f"total {sum(p['chars'] for p in page_texts)} chars")
    except Exception as e:
        if verbose:
            print(f"  Phase 1: text extraction failed ({e!s:.80}); "
                  f"reader will rely on PDF binary only")
        text_block = "(per-page text extraction unavailable)"

    model = resolve_model(model_name)

    from pydantic_ai.exceptions import UnexpectedModelBehavior

    try:
        result = _run_sync_with_retry(
            _reader_agent,
            [
                BinaryContent(data=pdf_bytes, media_type='application/pdf'),
                "Read this UK planning PDF and populate the PDFInfo schema "
                "with all geographic information you can find.\n\n"
                "Below is the PDF text already extracted by a dedicated OCR "
                "pipeline (fitz for digital pages — 100% accurate; PaddleOCR "
                "for scanned pages — ~95-98% accurate). For ANY exact-string "
                "field — postcodes, grid_refs, site_address, "
                "house_number_road_pairs, parish_names, admin_region, "
                "district_name, scale — this TEXT BLOCK is the source of "
                "truth. If the PDF image and the text block disagree on a "
                "character (e.g. you 'see' NR15 2XE in the image but the "
                "text block says NR16 1DJ), trust the TEXT BLOCK — it is "
                "more accurate than re-reading the image.\n\n"
                "If a page below says '(extraction failed; rely on PDF "
                "image)', do your normal vision-OCR for that page only.\n\n"
                "TEXT BLOCK (per page):\n\n"
                f"{text_block}",
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


# ── Phase 2 setup: pre-render map pages, build worker user prompt ──────────

def prepare_worker_state(
    pdf_path: str,
    sam3: Dict[str, Any],
    minima_matcher: Any,
    pdf_info: dict,
    dpi: int,
    case_name: Optional[str],
    verbose: bool,
) -> Tuple[AgentState, list]:
    """Create AgentState, pre-render every map_page from the reader, and
    build the worker's user_parts (JSON summary + active page image)."""
    state = AgentState(
        pdf_path=str(pdf_path),
        sam3_processor=sam3["processor"],
        sam3_model=sam3["model"],
        device=sam3["device"],
        minima_matcher=minima_matcher,
        dpi=dpi,
        sam3_state=sam3,
        case_name=case_name,
    )
    state.pdf_info = {k: v for k, v in pdf_info.items() if not k.startswith("_")}

    map_pages = pdf_info.get("map_pages", []) or []
    map_page_details = pdf_info.get("map_page_details", []) or []
    if map_pages:
        from tools.io.map_page import render_map_page
        for page_1based in map_pages:
            rendered = render_map_page(str(pdf_path), int(page_1based),
                                         dpi=dpi, verbose=verbose)
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
            if page_1based == map_pages[0]:
                state.map_img = page_img
                state.map_crop_path = tmp_path

    summary_text = json.dumps(
        {k: v for k, v in pdf_info.items() if not k.startswith("_")},
        indent=2)
    roles_line = ""
    if map_page_details:
        roles = ", ".join(
            f"page {d.get('page', '?')}=[{d.get('role', '?')}] "
            f"{(d.get('caption') or '')[:60]!r}"
            for d in map_page_details
        )
        roles_line = f"\nMap-page roles (cached, switch via render_page(N)): {roles}\n"
    user_parts: list = [
        f"PDF EXTRACTION SUMMARY:\n{summary_text}\n{roles_line}\n"
        f"Use this information to geolocate and extract the planning boundary. "
        f"Page {map_pages[0] if map_pages else '?'} (role='detail', the "
        f"top-ranked map) is pre-rendered as your working map."
    ]
    if state.map_img is not None:
        user_parts.append(f"Map page {map_pages[0]}:")
        user_parts.append(_img_to_binary(state.map_img))
    return state, user_parts


# ── Phase 2: invoke the worker ────────────────────────────────────────────

def invoke_worker(
    state: AgentState, user_parts: list, model_name: str,
    max_iterations: int, verbose: bool,
):
    """Run the worker agent's tool loop. Returns the pydantic-ai result
    (BoundaryOutcome on success), or raises on a non-retryable error.

    The caller handles UnexpectedModelBehavior / UsageLimitExceeded /
    other exceptions and writes a partial_state.json dump."""
    model = resolve_model(model_name)
    return _run_sync_with_retry(
        _agent,
        user_parts,
        deps=state,
        model=model,
        usage_limits=UsageLimits(request_limit=max(max_iterations * 4, 25)),
        label="worker",
    )


# ── Phase 2 error path: dump partial state ────────────────────────────────

def dump_partial_state(state: AgentState, pdf_info: dict, exc: Exception,
                        case_dir: Optional[Path], verbose: bool) -> dict:
    """Build and write partial_state.json so post-hoc debugging works
    when the worker errors mid-conversation."""
    partial_stats = {
        "pdf_info": state.pdf_info,
        "method": "error",
        "error": str(exc),
        "error_type": type(exc).__name__,
        "current_match_info": state.current_result.get("match_info", {}),
        "centers_tried": state.centers_tried,
        "match_attempts_summary": {
            cid: {
                "name": a.get("name"),
                "lat": a.get("lat"), "lon": a.get("lon"),
                "overall_score": a.get("overall_score"),
                "n_inliers": (a.get("match_info") or {}).get("n_inliers"),
            }
            for cid, a in (state.match_attempts or {}).items()
        },
        "position_calls": state.position_calls,
        "verify_position_called": state.verify_position_called,
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


# ── Phase 3: critic ───────────────────────────────────────────────────────

def apply_critic_loop(
    state: AgentState, worker_result: Any, model_name: str,
    sam3: Dict[str, Any], minima_matcher: Any, verbose: bool,
) -> Optional[Dict[str, Any]]:
    """Run the critic loop and propagate results onto state. Returns the
    critic_result dict (or None when the critic was skipped/failed)."""
    if (state.last_output is None
            or state.last_output.status != "accepted"
            or state.current_result.get("geojson") is None
            or state.current_mask is None):
        return None
    try:
        from tools.agent.critic import run_critic_loop
        critic_result = run_critic_loop(
            state=state, worker_agent=_agent,
            worker_result=worker_result,
            model=resolve_model(model_name),
            sam3=sam3, minima_matcher=minima_matcher, verbose=verbose,
        )
    except Exception as _critic_err:
        if verbose:
            print(f"  Critic loop failed (continuing): {_critic_err}")
            traceback.print_exc()
        return None

    state.critic_iterations = critic_result["iterations"]
    state.critic_final_decision = critic_result["final_decision"]
    state.critic_changed_mask = critic_result["changed_mask"]
    state.critic_suspected_wrong_location = critic_result.get(
        "suspected_wrong_location", False)
    state.critic_worker_reentered = critic_result.get("worker_reentered", False)

    if critic_result["final_decision"] == "flag_low_confidence":
        state.accepted = False
        last_reason = (critic_result["iterations"][-1].get("reason", "")[:160]
                       if critic_result["iterations"] else "")
        state.accept_reason = (
            f"CRITIC_LOW_CONFIDENCE: {last_reason} | prior: "
            f"{(state.accept_reason or '')[:100]}"
        )
    if verbose:
        print(f"  Critic final: {critic_result['final_decision']} "
              f"(changed_mask={state.critic_changed_mask}, "
              f"worker_reentered={state.critic_worker_reentered})")
    return critic_result


# ── Cleanup ───────────────────────────────────────────────────────────────

def cleanup_temp_pages(state: AgentState) -> None:
    """Unlink every pre-rendered page tempfile (and rotated variants)."""
    seen_paths = set()
    for p in list(state.rendered_page_paths.values()) + [state.map_crop_path]:
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


# ── Message log + stats extraction ────────────────────────────────────────

def extract_message_log(result: Any) -> Tuple[list, dict]:
    """Walk pydantic-ai's message history and return (message_log,
    extracted_stats). extracted_stats is a dict with keys: tool_calls,
    total_tool_calls, n_turns, geocode_types, validator_retries.

    Each message_log entry is {turn, role, kind, ...} where the extras
    depend on the part kind (tool, args, return, text, etc.).
    """
    message_log: list = []
    tool_calls: Dict[str, int] = {}
    turn_idx = 0

    for msg in result.all_messages():
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

    geocode_types: Dict[str, int] = {}
    validator_retries = 0
    for e in message_log:
        if e.get("tool") == "geocode":
            args = e.get("args", {})
            if isinstance(args, dict):
                t = args.get("type", "?")
                geocode_types[t] = geocode_types.get(t, 0) + 1
        if e.get("kind", "").lower().startswith("retryprompt"):
            validator_retries += 1

    extracted = {
        "tool_calls": tool_calls,
        "total_tool_calls": sum(tool_calls.values()),
        "n_turns": turn_idx,
        "geocode_types": geocode_types,
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

    if state.last_output is not None:
        out = state.last_output
        agent_stats["outcome_status"] = out.status
        agent_stats["outcome_reasoning"] = out.reasoning
        agent_stats["visual_check_notes"] = out.visual_check_notes
        agent_stats["verify_position_called"] = out.verify_position_called
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
            agent_stats["request_tokens"] = (
                (reader_tokens.get("request", 0) or 0)
                + (usage.request_tokens or 0))
            agent_stats["response_tokens"] = (
                (reader_tokens.get("response", 0) or 0)
                + (usage.response_tokens or 0))
            agent_stats["total_tokens"] = reader_total + worker_total
        except Exception:
            pass
    return agent_stats


# ── Soft quality gate ─────────────────────────────────────────────────────

def apply_quality_gate(state: AgentState, agent_rejected: bool,
                        verbose: bool) -> None:
    """Soft LOW_QUALITY gate: flag the result but never null the geojson —
    partial IoU always beats no prediction.

    Mutates state.accepted / state.accept_reason if the gate trips.
    """
    final_mi = state.current_result.get("match_info") or {}
    if not (final_mi or agent_rejected):
        return
    _inl = final_mi.get("n_inliers", 0) or 0
    _score = final_mi.get("score", 0) or 0
    quant_reject = (_inl < 25 and _score < 15)
    if not (quant_reject or agent_rejected):
        return
    prev_reason = state.accept_reason[:160] if state.accept_reason else ""
    if agent_rejected and not quant_reject:
        gate_reason = (
            f"LOW_QUALITY (agent visual check flagged) "
            f"(inliers={_inl}, score={_score:.1f}): {prev_reason}")
    else:
        gate_reason = (
            f"LOW_QUALITY (inliers={_inl} < 25, score={_score:.1f} < 15). "
            f"Agent said: {prev_reason}")
    state.accepted = False
    state.accept_reason = gate_reason
    if verbose:
        src = "agent visual" if agent_rejected and not quant_reject else "quality gate"
        print(f"  {src.upper()}: flagging low-quality (inliers={_inl}, "
              f"score={_score:.1f}) - keeping geojson for partial IoU")


# ── Return-dict assembly ──────────────────────────────────────────────────

def build_run_agent_return(
    state: AgentState,
    agent_stats: dict,
    message_log: list,
    critic_result: Optional[Dict[str, Any]],
) -> dict:
    """Assemble the dict that run_agent returns to benchmark_runner."""
    return {
        "success": True,
        "geojson": state.current_result.get("geojson"),
        "match_info": state.current_result.get("match_info", {}),
        "mask": state.current_mask,
        "affine_H": state.current_result.get("affine_H"),
        "tile_info_meta": {
            k: v for k, v in (state.current_result.get("tile_info") or {}).items()
            if k != "image"
        },
        "agent_accepted": state.accepted,
        "agent_reason": state.accept_reason,
        "agent_stats": agent_stats,
        "message_log": message_log,
        "candidate_overlays": state.candidate_overlays,
        "selected_overlay": state.selected_overlay,
        "selected_indices": state.selected_indices,
        "critic_iterations": state.critic_iterations,
        "critic_final_decision": state.critic_final_decision,
        "critic_changed_mask": state.critic_changed_mask,
        "critic_applied_rotation_deg": state.critic_applied_rotation_deg,
        "critic_suspected_wrong_location": state.critic_suspected_wrong_location,
        "critic_worker_reentered": state.critic_worker_reentered,
        "critic_panel_img": (critic_result.get("panel_iter0")
                              if critic_result else None),
        "critic_tokens": (critic_result.get("tokens_used")
                           if critic_result else None),
        "critic_pre_snapshot": (critic_result.get("pre_snapshot")
                                  if critic_result else None),
        "critic_final_snapshot": (critic_result.get("final_snapshot")
                                    if critic_result else None),
        "critic_iteration_panels": (critic_result.get("per_iteration_panels")
                                      if critic_result else []),
        "critic_iteration_snapshots": (critic_result.get("per_iteration_snapshots")
                                         if critic_result else []),
        "centers_tried": state.centers_tried,
    }
