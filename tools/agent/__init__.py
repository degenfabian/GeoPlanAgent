"""Unified planning boundary extraction agent — orchestrator.

PydanticAI-based agent with 10 tools that handles the full pipeline:
PDF reading → geocoding → positioning → boundary extraction → verification.

This module is the orchestrator. The agent definition, state, helpers
and tools live in dedicated modules (extracted 2026-05-11 in stage 2 of
the agent split):

* :mod:`tools.agent_core`           — ``_agent`` / ``_reader_agent`` /
                                      ``AgentState`` / shared helpers
* :mod:`tools.agent_tools_render`   — ``render_page``
* :mod:`tools.agent_tools_locate`   — ``geocode`` / ``propose_centers``
* :mod:`tools.agent_tools_match`    — ``match_at`` / ``commit_match``
* :mod:`tools.agent_tools_extract`  — ``extract_boundary`` /
                                      ``project_boundary``
* :mod:`tools.agent_tools_verify`   — ``verify_position`` /
                                      ``lookup_district`` / ``visualize``

The tool modules use ``@_agent.tool`` / ``@_agent.tool_plain``
decorators that register against the shared ``_agent`` instance at
import time. The orchestrator must therefore import them BEFORE
``run_agent`` is invoked.

Tools (registered at import time across the modules above):
     1. render_page       — render a PDF page as an image
     2. geocode           — look up coordinates (postcode or grid_ref only)
     3. propose_centers   — get ranked candidate centers from text fields
     4. match_at          — run MINIMA at a specific (lat, lon)
     5. commit_match      — accept the chosen match
     6. extract_boundary  — SAM3 boundary segmentation
     7. project_boundary  — project mask to GeoJSON via affine
     8. verify_position   — visual inspection on OS tiles
     9. lookup_district   — get district boundary from OSM
    10. visualize         — show boundary overlay + positioned GeoJSON

Rotation: handled automatically by a trained ResNet50 classifier inside
render_page. The agent does NOT decide rotation.
"""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
from pydantic_ai import BinaryContent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.usage import UsageLimits

# Structured I/O schemas live in their own module (extracted 2026-05-11).
# Re-exported from this module for backwards-compatibility — existing
# imports of `from tools.agent import PDFInfo` etc. continue to work.
from tools.agent.schemas import (
    PDFInfo,
    BoundaryOutcome,
)
# Large system prompts live in their own module (extracted 2026-05-11).
from tools.agent.prompts import READER_SYSTEM_PROMPT, WORKER_SYSTEM_PROMPT

# Agent core (extracted stage 2, 2026-05-11). The tool-registration
# decorators inside the agent_tools_* modules attach to `_agent` at
# import time; we MUST import the core first, then the tool modules,
# before run_agent is callable.
from tools.agent.state import (
    _agent,
    _reader_agent,
    AgentState,
    MODEL_ALIASES,
    resize_for_api,
    _img_to_binary,
    _dedup_check,
    _create_boundary_overlay,
    _draw_geojson_on_tiles,
    _strip_old_images,
    _RETRYABLE_STATUS,
    _is_retryable_http_error,
    _run_sync_with_retry,
    validate_boundary_outcome,
    build_system_prompt,
)

# Importing these modules triggers the `@_agent.tool` decorators inside
# them, registering the 10 worker tools against `_agent`. Without these
# imports the agent would have NO tools when run_sync is called.
# Order is not important among the tool modules (decorators are
# idempotent), but they MUST be imported AFTER `_agent` is created.
from tools.agent.tools import (    # noqa: F401  (decorator side-effects)
    render as _render_tool,
    locate as _locate_tool,
    match as _match_tool,
    extract as _extract_tool,
    verify as _verify_tool,
    refine as _refine_tool,
)



def _read_pdf_phase(pdf_path: str, model_name: str, verbose: bool = True) -> dict:
    """Phase 1: Send the PDF to the reader agent, get structured extraction.

    The reader agent's output_type=PDFInfo, so pydantic-ai enforces the schema.
    No JSON parsing or markdown fence stripping needed.

    Augments the prompt with per-page text extracted via fitz (born-digital
    pages) or OCR (scanned pages, ~60% of the eval). Gemini still does its
    own PDF processing for vision-required fields (boundary_color, rotation,
    map labels), but exact-string fields (postcodes, grid_refs, addresses)
    benefit from being given the ground-truth text alongside the image. The
    text extraction is cached on disk under ``cache/text_extraction/``.

    Returns:
        Dict of extracted info (from PDFInfo.model_dump()), plus "_reader_tokens".
        On failure: {"error": ..., defaulted fields...}.
    """
    pdf_bytes = Path(pdf_path).read_bytes()

    if verbose:
        print(f"  Phase 1: reading PDF ({len(pdf_bytes) // 1024} KB)...")

    # Per-page text: fitz where the PDF is digital, OCR where it isn't.
    # Cached so reruns are instant. Wrapped in try/except — extraction
    # failure is non-fatal, the reader can still rely on the PDF binary.
    try:
        from tools.io.text_extraction import extract_text_per_page, format_for_reader_prompt
        page_texts = extract_text_per_page(pdf_path, use_cache=True, verbose=verbose)
        text_block = format_for_reader_prompt(page_texts)
        if verbose:
            methods = {p["method"]: 0 for p in page_texts}
            for p in page_texts:
                methods[p["method"]] += 1
            print(f"  Phase 1: text extraction {dict(methods)}, "
                  f"total {sum(p['chars'] for p in page_texts)} chars")
    except Exception as e:
        if verbose:
            print(f"  Phase 1: text extraction failed ({e!s:.80}); "
                  f"reader will rely on PDF binary only")
        text_block = "(per-page text extraction unavailable)"

    model = OpenRouterModel(model_name)

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
            usage_limits=UsageLimits(request_limit=5),  # allow a validator retry
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

        # Extract usage from reader phase
        usage = result.usage()
        info["_reader_tokens"] = {
            "request": usage.request_tokens,
            "response": usage.response_tokens,
        }

        return info

    except UnexpectedModelBehavior as e:
        if verbose:
            print(f"  Phase 1 failed: {e}")
        # Fallback: return an empty-ish PDFInfo so downstream doesn't crash.
        empty = PDFInfo().model_dump()
        empty["error"] = str(e)
        return empty


# NOTE: critic-driven worker rehand lives inside tools/agent/critic_v2.py
# (function `_rehand_to_worker`) — it loops critic ↔ rehand up to
# `max_iters` and tracks compliance. run_agent below just calls
# run_critic_loop and consumes the returned dict.


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
    """Run the two-phase agent on a single planning document.

    Phase 1: Reader agent reads the full PDF once, extracts structured info (JSON).
    Phase 2: Worker agent receives only the JSON summary + rendered map image.
             The full PDF is never in the worker's context, saving ~90% of tokens
             on multi-turn conversations.
    Phase 3 (if enable_critic=True, OPT-IN): Commenter VLM critic reviews the
             worker's output, auto-fixes simple issues in code, or re-enters
             the worker with feedback. Never nullifies the GeoJSON — partial
             IoU > 0. Skipped for multi-page and district_lookup cases.
             Default is OFF — measured to add minimal IoU over the base agent
             on the v20 dataset; only kept as an opt-in ablation cell.

    Args:
        pdf_path: Path to the planning PDF.
        models_state: Dict with sam3_ft/sam3_base and minima models.
        model_name: OpenRouter model identifier.
        max_iterations: Maximum number of agent turns.
        dpi: PDF rendering DPI.
        verbose: Print progress.
        enable_critic: Run Phase 3 critic loop after worker finishes.

    Returns:
        Dict with: geojson, match_info, mask, agent_accepted, agent_reason, etc.
    """
    # Resolve model alias
    model_name = MODEL_ALIASES.get(model_name, model_name)

    # Get SAM3 model (prefer fine-tuned)
    if "sam3_ft" in models_state:
        sam3 = models_state["sam3_ft"]
    elif "sam3_base" in models_state:
        sam3 = models_state["sam3_base"]
    else:
        return {"success": False, "error": "No SAM3 model loaded"}

    # ── Phase 1: Read the PDF ──────────────────────────────────────────────
    pdf_info = _read_pdf_phase(pdf_path, model_name, verbose=verbose)

    # Incremental flush: write pdf_info.json now so it survives a Phase 2
    # crash. Without this, Phase 2 errors lose all reader output and we
    # can't tell post-hoc what the agent had to work with.
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

    # ── Phase 2: Work from the summary ─────────────────────────────────────
    state = AgentState(
        pdf_path=str(pdf_path),
        sam3_processor=sam3["processor"],
        sam3_model=sam3["model"],
        device=sam3["device"],
        minima_matcher=models_state["minima"],
        dpi=dpi,
        sam3_state=sam3,
        case_name=case_name,
    )
    # Give the output_validator access to the reader's extraction (for
    # multi-page counting and district-wide checks).
    state.pdf_info = {k: v for k, v in pdf_info.items() if not k.startswith("_")}

    # Pre-render EVERY map page from the reader (auto-rotate + map_crop applied).
    # Each is cached on state.rendered_pages so render_page(N) becomes a free
    # state-pointer flip rather than re-rendering. The first page is the
    # active one when the worker starts.
    map_pages = pdf_info.get("map_pages", []) or []
    map_page_details = pdf_info.get("map_page_details", []) or []
    if map_pages:
        from tools.io.pdf import render_pdf_page
        try:
            from tools.io.rotation_classifier import auto_rotate
        except Exception:
            auto_rotate = None
        try:
            from tools.io.map_crop import detect_title_block_crop
        except Exception:
            detect_title_block_crop = None
        for page_1based in map_pages:
            page_idx = max(0, int(page_1based) - 1)
            try:
                page_img = render_pdf_page(str(pdf_path), page_idx, dpi=dpi)
            except IndexError:
                page_img = None
            if page_img is None:
                continue
            if auto_rotate is not None:
                try:
                    page_img, rot_info = auto_rotate(page_img, verbose=verbose)
                    if rot_info.get("applied") and page_1based == map_pages[0]:
                        state.rotation_checked = True
                except Exception as e:
                    if verbose:
                        print(f"  rotation_classifier failed for page "
                              f"{page_1based} ({e!s:.80}); proceeding raw")
            if detect_title_block_crop is not None:
                try:
                    cropped, _xo, _yo, _info = detect_title_block_crop(page_img)
                    if _info.get("cropped"):
                        page_img = cropped
                except Exception:
                    pass
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            cv2.imwrite(tmp_path, page_img)
            state.rendered_pages[int(page_1based)] = page_img
            state.rendered_page_paths[int(page_1based)] = tmp_path
            if page_1based == map_pages[0]:
                state.map_img = page_img
                state.map_crop_path = tmp_path

    # No fast-path on is_district_wide. The reader over-flags this on
    # conservation areas, named neighbourhoods, and small sites in
    # documents that mention an admin region — all of which should be
    # positioned, not replaced with a borough polygon. The worker has
    # lookup_district as a tool and falls back to it when match_at
    # attempts all score below the system-prompt threshold.

    # Build the worker's user prompt: JSON summary + active map page image.
    # Pages with role != "detail" are described in the summary but not sent
    # as images to keep token usage flat; the worker can render_page(N) to
    # switch (free, since we pre-rendered).
    summary_text = json.dumps({k: v for k, v in pdf_info.items()
                                if not k.startswith("_")}, indent=2)
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

    # Attach the active map image
    if state.map_img is not None:
        user_parts.append(f"Map page {map_pages[0]}:")
        user_parts.append(_img_to_binary(state.map_img))

    if verbose:
        print(f"  Running agent ({model_name}, max {max_iterations} turns)")

    model = OpenRouterModel(model_name)

    from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded

    result = None
    agent_rejected = False  # set True if the agent's DONE message starts with REJECTED
    try:
        if verbose:
            print("  Phase 2: sending summary + map image to worker agent...")
            print(f"  Agent tools: {[t.name for t in _agent._toolset.tools()]}"
                  if hasattr(_agent, '_toolset') else "")
        # Floor of 25 model requests covers healthy hard cases (15-22 requests
        # typically), above that caps bad runs.
        result = _run_sync_with_retry(
            _agent,
            user_parts,
            deps=state,
            model=model,
            usage_limits=UsageLimits(request_limit=max(max_iterations * 4, 25)),
            label="worker",
        )
        if verbose:
            print("  Agent completed normally")
            # Debug: show message history
            for msg in result.all_messages():
                role = getattr(msg, 'kind', type(msg).__name__)
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        kind = getattr(part, 'kind', type(part).__name__)
                        if kind == 'tool-call':
                            print(f"    [{role}] tool-call: {part.tool_name}({str(part.args)[:80]})")
                        elif kind == 'tool-return':
                            print(f"    [{role}] tool-return: {str(part.content)[:80]}")
                        elif kind == 'text':
                            print(f"    [{role}] text: {str(part.content)[:100]}")
                        else:
                            print(f"    [{role}] {kind}")
                else:
                    print(f"    [{role}] {str(msg)[:100]}")
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
        # Dump whatever partial state we have so post-hoc debugging works.
        # Pre-fix, errored cases lost everything (no pdf_info, no message_log,
        # no centers_tried) and we couldn't tell if the 400 hit at turn 1
        # or turn 30. With this dump, the case dir at minimum has:
        # pdf_info.json (Phase 1), partial_state.json (Phase 2 progress).
        partial_stats = {
            "pdf_info": state.pdf_info,
            "method": "error",
            "error": str(e),
            "error_type": type(e).__name__,
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
        return {
            "success": False,
            "error": str(e),
            "geojson": state.current_result.get("geojson"),
            "match_info": state.current_result.get("match_info", {}),
            "mask": state.current_mask,
            "agent_stats": partial_stats,
        }
    else:
        # Structured output: result.output is a BoundaryOutcome (validated).
        outcome: BoundaryOutcome = result.output
        state.last_output = outcome
        state.accepted = (outcome.status in ("accepted", "district_lookup"))
        state.accept_reason = f"[{outcome.status}] {outcome.reasoning[:160]}"
        if verbose:
            print(f"  Worker outcome: status={outcome.status} "
                  f"inliers={outcome.final_n_inliers} "
                  f"verify={outcome.verify_position_called} "
                  f"rotation_checked={outcome.rotation_checked}")

    # ── Phase 3: Commenter VLM critic loop ─────────────────────────────────
    # Runs whenever the worker produced an "accepted" result with a geojson
    # + mask. district_lookup is skipped (no mask/affine to critique).
    critic_result = None
    if enable_critic and result is not None \
            and state.last_output is not None \
            and state.last_output.status == "accepted" \
            and state.current_result.get("geojson") is not None \
            and state.current_mask is not None:
        try:
            from tools.agent.critic import run_critic_loop
            critic_result = run_critic_loop(
                state=state,
                worker_agent=_agent,
                worker_result=result,
                model=model,
                sam3=sam3,
                minima_matcher=models_state["minima"],
                verbose=verbose,
            )
            state.critic_iterations = critic_result["iterations"]
            state.critic_final_decision = critic_result["final_decision"]
            state.critic_changed_mask = critic_result["changed_mask"]
            state.critic_suspected_wrong_location = critic_result.get(
                "suspected_wrong_location", False)
            state.critic_worker_reentered = critic_result.get(
                "worker_reentered", False)

            if critic_result["final_decision"] == "flag_low_confidence":
                state.accepted = False
                last_reason = (
                    critic_result["iterations"][-1].get("reason", "")[:160]
                    if critic_result["iterations"] else ""
                )
                state.accept_reason = (
                    f"CRITIC_LOW_CONFIDENCE: {last_reason} | prior: "
                    f"{(state.accept_reason or '')[:100]}"
                )
            if verbose:
                print(f"  Critic final: {critic_result['final_decision']} "
                      f"(changed_mask={state.critic_changed_mask}, "
                      f"worker_reentered={state.critic_worker_reentered})")
        except Exception as _critic_err:
            if verbose:
                print(f"  Critic loop failed (continuing): {_critic_err}")
                traceback.print_exc()

    # Agent rejection paths removed 2026-05-14. The BoundaryOutcome schema
    # only allows status="accepted" or "district_lookup" now, so the only
    # remaining "rejection-like" signal is an exception path where the
    # output never validated and accept_reason was populated by the runner
    # with a REJECTED:... prefix.
    agent_rejected = (state.accept_reason or "").upper().lstrip().startswith("REJECTED")

    # Clean up temp files: every pre-rendered page + any rotated variants.
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

    if verbose:
        mi = state.current_result.get("match_info", {})
        print(f"  Agent done: accepted={state.accepted}, "
              f"inliers={mi.get('n_inliers', 0)}, "
              f"reason={state.accept_reason[:100]}")

    # Extract comprehensive stats from message history
    agent_stats = {
        "position_calls": state.position_calls,
        "pdf_info": {k: v for k, v in pdf_info.items() if not k.startswith("_")},
    }
    # Include reader phase token usage
    reader_tokens = pdf_info.get("_reader_tokens", {})
    if reader_tokens:
        agent_stats["reader_request_tokens"] = reader_tokens.get("request", 0)
        agent_stats["reader_response_tokens"] = reader_tokens.get("response", 0)

    # Surface BoundaryOutcome fields directly so we don't have to rummage
    # through message_log.json to see WHY the agent accepted/rejected.
    if state.last_output is not None:
        out = state.last_output
        agent_stats["outcome_status"] = out.status
        agent_stats["outcome_reasoning"] = out.reasoning
        agent_stats["visual_check_notes"] = out.visual_check_notes
        agent_stats["verify_position_called"] = out.verify_position_called
        agent_stats["rotation_checked"] = out.rotation_checked

    # Extract full message history for offline analysis.
    # pydantic-ai 1.81 part kinds: SystemPromptPart, UserPromptPart, ToolCallPart,
    # ToolReturnPart, RetryPromptPart, TextPart, ThinkingPart.
    message_log = []
    try:
        if result is not None:
            tool_calls = {}
            turn_idx = 0
            for msg in result.all_messages():
                role = getattr(msg, 'kind', type(msg).__name__)
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        kind = getattr(part, 'kind', type(part).__name__)
                        # Normalize to lowercase for easier matching
                        kind_lower = kind.lower()
                        entry = {"turn": turn_idx, "role": role, "kind": kind}

                        if 'toolcall' in kind_lower:
                            name = getattr(part, 'tool_name', '?')
                            tool_calls[name] = tool_calls.get(name, 0) + 1
                            entry["tool"] = name
                            try:
                                args = getattr(part, 'args', None)
                                if args is None:
                                    entry["args"] = {}
                                elif isinstance(args, dict):
                                    entry["args"] = {
                                        k: (v if not isinstance(v, (bytes, bytearray))
                                            else f"<bytes:{len(v)}>")
                                        for k, v in args.items()
                                    }
                                elif isinstance(args, str):
                                    # Args can be a JSON string
                                    try:
                                        import json as _j
                                        parsed = _j.loads(args)
                                        entry["args"] = parsed if isinstance(parsed, dict) else str(args)[:500]
                                    except Exception:
                                        entry["args"] = args[:500]
                                else:
                                    entry["args"] = str(args)[:500]
                            except Exception as _e:
                                entry["args"] = f"<err:{_e}>"

                        elif 'toolreturn' in kind_lower:
                            entry["tool"] = getattr(part, 'tool_name', '?')
                            content = getattr(part, 'content', None)
                            if isinstance(content, dict):
                                entry["return"] = {
                                    k: (v if not isinstance(v, (bytes, bytearray))
                                        else f"<bytes:{len(v)}>")
                                    for k, v in content.items()
                                }
                            elif isinstance(content, str):
                                entry["return"] = content[:1000]
                            else:
                                entry["return"] = str(content)[:1000]

                        elif 'retry' in kind_lower:
                            # RetryPromptPart — the validator fired
                            rc = getattr(part, 'content', None)
                            entry["retry_content"] = str(rc)[:1000] if rc else ""

                        elif 'userprompt' in kind_lower:
                            # User prompts sometimes contain images — note their presence
                            c = getattr(part, 'content', None)
                            if isinstance(c, list):
                                n_images = sum(1 for x in c
                                               if hasattr(x, 'media_type'))
                                n_text = sum(1 for x in c if isinstance(x, str))
                                entry["user_summary"] = f"{n_text} text + {n_images} images"
                            elif isinstance(c, str):
                                entry["text"] = c[:500]

                        elif 'text' in kind_lower or 'thinking' in kind_lower:
                            entry["text"] = str(getattr(part, 'content', ''))[:2000]

                        message_log.append(entry)
                turn_idx += 1

            agent_stats["tool_calls"] = tool_calls
            agent_stats["total_tool_calls"] = sum(tool_calls.values())
            agent_stats["n_turns"] = turn_idx

            # Per-case geocode-type breakdown: which `type=` did the agent
            # actually pass to geocode()? Tells us whether agent prefers
            # gpkg_place over place, used wikidata, etc.
            geocode_types = {}
            validator_retries = 0
            for e in message_log:
                if e.get("tool") == "geocode":
                    args = e.get("args", {})
                    if isinstance(args, dict):
                        t = args.get("type", "?")
                        geocode_types[t] = geocode_types.get(t, 0) + 1
                if e.get("kind", "").lower().startswith("retryprompt"):
                    validator_retries += 1
            agent_stats["geocode_types"] = geocode_types
            agent_stats["validator_retries"] = validator_retries
            # Extract worker phase usage
            usage = result.usage()
            agent_stats["worker_request_tokens"] = usage.request_tokens
            agent_stats["worker_response_tokens"] = usage.response_tokens
            # Total = reader + worker
            reader_total = sum(reader_tokens.values()) if reader_tokens else 0
            worker_total = (usage.request_tokens or 0) + (usage.response_tokens or 0)
            agent_stats["request_tokens"] = (reader_tokens.get("request", 0) or 0) + (usage.request_tokens or 0)
            agent_stats["response_tokens"] = (reader_tokens.get("response", 0) or 0) + (usage.response_tokens or 0)
            agent_stats["total_tokens"] = reader_total + worker_total
    except Exception:
        pass

    # Single-page final geojson: just use the agent's last current_result.
    final_geojson = state.current_result.get("geojson")

    # Soft quality gate: flag LOW_QUALITY but never null the geojson —
    # partial IoU always beats no prediction.
    final_mi = state.current_result.get("match_info") or {}
    if final_mi or agent_rejected:
        _inl = final_mi.get("n_inliers", 0) or 0
        _score = final_mi.get("score", 0) or 0
        quant_reject = (_inl < 25 and _score < 15)
        if quant_reject or agent_rejected:
            prev_reason = state.accept_reason[:160] if state.accept_reason else ""
            if agent_rejected and not quant_reject:
                gate_reason = (
                    f"LOW_QUALITY (agent visual check flagged) "
                    f"(inliers={_inl}, score={_score:.1f}): {prev_reason}"
                )
            else:
                gate_reason = (
                    f"LOW_QUALITY (inliers={_inl} < 25, score={_score:.1f} "
                    f"< 15). Agent said: {prev_reason}"
                )
            state.accepted = False
            state.accept_reason = gate_reason
            if verbose:
                src = "agent visual" if agent_rejected and not quant_reject else "quality gate"
                print(f"  {src.upper()}: flagging low-quality (inliers={_inl}, "
                      f"score={_score:.1f}) - keeping geojson for partial IoU")

    # M2 (v18): the FALLBACK_ANCHOR path was deleted along with
    # `_position_boundary_disabled` (which was the only thing populating
    # `state.centers`). Under v17/v18 `state.centers` is always empty, so
    # this block was dead code. If MINIMA fails entirely under v18, we
    # surface that — no synthesised partial-IoU fallback.
    return {
        "success": True,
        "geojson": final_geojson,
        "match_info": state.current_result.get("match_info", {}),
        "mask": state.current_mask,
        "affine_H": state.current_result.get("affine_H"),
        "tile_info_meta": {
            k: v for k, v in (state.current_result.get("tile_info") or {}).items()
            if k != "image"  # don't return the tile image array
        },
        "agent_accepted": state.accepted,
        "agent_reason": state.accept_reason,
        "agent_stats": agent_stats,
        "message_log": message_log,
        "candidate_overlays": state.candidate_overlays,
        "selected_overlay": state.selected_overlay,
        "selected_indices": state.selected_indices,
        # Phase 3 (Commenter critic) artifacts
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
        # Rigorous-analysis artefacts (written to critic_debug/ by benchmark)
        "critic_pre_snapshot": (critic_result.get("pre_snapshot")
                                  if critic_result else None),
        "critic_final_snapshot": (critic_result.get("final_snapshot")
                                    if critic_result else None),
        "critic_iteration_panels": (critic_result.get("per_iteration_panels")
                                      if critic_result else []),
        "critic_iteration_snapshots": (critic_result.get("per_iteration_snapshots")
                                         if critic_result else []),
        # Geocoding transparency
        "centers_tried": state.centers_tried,
    }
