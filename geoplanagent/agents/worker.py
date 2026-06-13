"""Phase 2 worker Agent + output validator + history processor."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic_ai import Agent, ModelRetry, RunContext

from geoplanagent.prompts import FOLDED_SYSTEM_PROMPT, WORKER_SYSTEM_PROMPT
from geoplanagent.schemas import BoundaryOutcome
from geoplanagent.utils import AgentState

load_dotenv()

# Production runs at temperature 0 for reproducible tool selection,
# schema-constrained outputs, and commit decisions. Overridable via the
# GEOMAP_TEMPERATURE env var for the appendix temperature ablation.
_TEMPERATURE = float(os.environ.get("GEOMAP_TEMPERATURE", "0"))


def _strip_old_images(messages):
    """Drop images from messages older than KEEP_RECENT to bound token cost.

    Rebinds ``part.content`` rather than mutating in place — the same list
    objects are passed back into pydantic-ai across critic rehands, and an
    in-place strip would cascade and corrupt the on-disk message_log.
    """
    KEEP_RECENT = 4
    if len(messages) <= KEEP_RECENT:
        return messages
    cutoff = len(messages) - KEEP_RECENT

    for index, message in enumerate(messages):
        if index >= cutoff:
            continue
        parts = getattr(message, "parts", None)
        if not parts:
            continue
        for part in parts:
            content = getattr(part, "content", None)
            if not isinstance(content, list):
                continue

            def _is_strippable(item) -> bool:
                if not (hasattr(item, "media_type") and hasattr(item, "data")):
                    return False
                media_type = getattr(item, "media_type", "") or ""
                # PDFs also get stripped — in folded_mode the PDF lives in
                # the first user turn and is no longer needed once
                # submit_pdf_info has populated state.pdf_info.
                return media_type.startswith("image/") or media_type == "application/pdf"

            has_binary = any(_is_strippable(item) for item in content)
            if not has_binary:
                continue
            new_content = list(content)
            for content_index, item in enumerate(new_content):
                if _is_strippable(item):
                    kind = (
                        "PDF" if getattr(item, "media_type", "") == "application/pdf" else "image"
                    )
                    new_content[content_index] = (
                        f"[{kind} omitted from older history; "
                        f"was {item.media_type}, {len(item.data)} bytes]"
                    )
            try:
                part.content = new_content
            except Exception:
                # Frozen part: fall back to in-place to avoid a token blowup.
                content[:] = new_content
    return messages


_agent = Agent(
    "test",  # overridden at runtime via model= kwarg
    deps_type=AgentState,
    output_type=BoundaryOutcome,
    retries=5,
    output_retries=3,
    history_processors=[_strip_old_images],
    model_settings={"temperature": _TEMPERATURE},
)


@_agent.output_validator
async def validate_boundary_outcome(
    ctx: RunContext[AgentState], out: BoundaryOutcome
) -> BoundaryOutcome:
    """Raise ModelRetry if accepted/district_lookup is submitted without the prerequisite tool calls."""
    state = ctx.deps
    state.last_output = out

    # Prefer the union total across area_groups; fall back to primary group's n_inliers.
    current_result = state.current_result or {}
    match_info = current_result.get("match_info") or {}
    final_n_inliers = (
        current_result.get("total_inliers")
        if current_result.get("total_inliers") is not None
        else (match_info.get("n_inliers", 0) or 0)
    )
    final_n_inliers = int(final_n_inliers or 0)

    if out.rotation_checked != state.rotation_checked:
        out.rotation_checked = state.rotation_checked
    if out.final_n_inliers != final_n_inliers:
        out.final_n_inliers = final_n_inliers

    # Folded mode: refuse to finalise until submit_pdf_info has populated
    # state.pdf_info. Standard pipeline already has pdf_info from Phase 1.
    if getattr(state, "folded_mode", False) and not state.pdf_info:
        raise ModelRetry(
            "Cannot submit BoundaryOutcome yet — you have not called "
            "submit_pdf_info. The PDF binary is attached to your first "
            "user message; read it, populate the PDFInfo schema, and "
            "call submit_pdf_info(info=<PDFInfo>) before any other tool."
        )

    # district_lookup: polygon must come from a successful lookup_district call.
    if out.status == "district_lookup":
        if state.current_result.get("geojson") is None:
            raise ModelRetry(
                "status='district_lookup' requires a successful lookup_district "
                "call that produced a GeoJSON. Call lookup_district with the "
                "district_name from the PDFInfo and retry."
            )
        return out

    # status == "accepted" from here on.

    if final_n_inliers == 0 and state.current_result.get("geojson") is None:
        raise ModelRetry(
            "Cannot accept: no successful match_at + commit_match has produced "
            "a result. Run positioning to completion (propose_centers → "
            "match_at → commit_match). Even if all match_at attempts are weak "
            "(low n_inliers), commit the highest-n_inliers one anyway "
            "and proceed — the pipeline always produces a polygon."
        )

    return out


@_agent.system_prompt
def build_system_prompt(ctx: RunContext[AgentState]) -> str:
    """Pick the worker's system prompt: folded variant when the run
    skips the dedicated reader phase (--no-reader), standard otherwise."""
    if getattr(ctx.deps, "folded_mode", False):
        return FOLDED_SYSTEM_PROMPT
    return WORKER_SYSTEM_PROMPT
