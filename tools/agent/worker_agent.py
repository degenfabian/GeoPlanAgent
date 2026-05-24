"""Phase 2 worker Agent + output validator + history processor."""

from __future__ import annotations

from dotenv import load_dotenv
from pydantic_ai import Agent, ModelRetry, RunContext

from tools.agent.prompts import WORKER_SYSTEM_PROMPT
from tools.agent.schemas import BoundaryOutcome
from tools.agent.state import AgentState

load_dotenv()


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

    for i, msg in enumerate(messages):
        if i >= cutoff:
            continue
        parts = getattr(msg, 'parts', None)
        if not parts:
            continue
        for part in parts:
            content = getattr(part, 'content', None)
            if not isinstance(content, list):
                continue
            has_image = any(
                hasattr(it, 'media_type') and hasattr(it, 'data')
                and getattr(it, 'media_type', '').startswith('image/')
                for it in content
            )
            if not has_image:
                continue
            new_content = list(content)
            for j, item in enumerate(new_content):
                if (hasattr(item, 'media_type')
                        and hasattr(item, 'data')
                        and getattr(item, 'media_type', '').startswith('image/')):
                    new_content[j] = (
                        f"[image omitted from older history; "
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
    model_settings={"temperature": 0},
)


@_agent.output_validator
async def validate_boundary_outcome(
    ctx: RunContext[AgentState], out: BoundaryOutcome
) -> BoundaryOutcome:
    """Raise ModelRetry if accepted/district_lookup is submitted without the prerequisite tool calls."""
    state = ctx.deps
    state.last_output = out

    # Prefer the union total across area_groups; fall back to primary group's n_inliers.
    cr = state.current_result or {}
    mi = cr.get("match_info") or {}
    final_inl = (
        cr.get("total_inliers")
        if cr.get("total_inliers") is not None
        else (mi.get("n_inliers", 0) or 0)
    )
    final_inl = int(final_inl or 0)

    if out.rotation_checked != state.rotation_checked:
        out.rotation_checked = state.rotation_checked
    if out.final_n_inliers != final_inl:
        out.final_n_inliers = final_inl

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

    if final_inl == 0 and state.current_result.get("geojson") is None:
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
    return WORKER_SYSTEM_PROMPT
