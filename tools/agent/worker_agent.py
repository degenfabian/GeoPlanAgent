"""Phase 2 worker: PydanticAI Agent with all tools registered.

The worker `_agent` is decorated by the tool modules under
tools.agent.tools.{render,locate,match,extract,verify,refine} at import
time via @_agent.tool — so this module must be importable BEFORE those
tool modules.

Defines:
  - _agent — the worker Agent instance
  - _strip_old_images — history processor that drops binary images from
    older messages to keep token cost flat
  - validate_boundary_outcome — output validator enforcing tool-call
    preconditions (verify_position when 25≤inliers≤100, etc.)
  - build_system_prompt — registers WORKER_SYSTEM_PROMPT
"""

from __future__ import annotations

from dotenv import load_dotenv
from pydantic_ai import Agent, ModelRetry, RunContext

from tools.agent.prompts import WORKER_SYSTEM_PROMPT
from tools.agent.schemas import BoundaryOutcome
from tools.agent.state import AgentState

load_dotenv()


def _strip_old_images(messages):
    """Replace BinaryContent in messages older than KEEP_RECENT with a
    placeholder. Without this, multi-image tool returns (e.g. match_at
    panels) get replayed every subsequent turn — token cost grows
    quadratically."""
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
            if isinstance(content, list):
                for j, item in enumerate(content):
                    if (hasattr(item, 'media_type')
                            and hasattr(item, 'data')
                            and getattr(item, 'media_type', '').startswith('image/')):
                        content[j] = (
                            f"[image omitted from older history; "
                            f"was {item.media_type}, {len(item.data)} bytes]"
                        )
    return messages


_agent = Agent(
    "test",  # overridden at runtime via model= kwarg
    deps_type=AgentState,
    output_type=BoundaryOutcome,
    retries=5,
    output_retries=3,
    history_processors=[_strip_old_images],
    # temperature=0 for reproducible runs.
    model_settings={"temperature": 0},
)


@_agent.output_validator
async def validate_boundary_outcome(
    ctx: RunContext[AgentState], out: BoundaryOutcome
) -> BoundaryOutcome:
    """Enforce that required tool calls happened before accepting an outcome.

    Pydantic-AI raises ModelRetry on failure and the agent has to submit
    again after filling the gap. This is what makes verify_position
    actually mandatory rather than suggested.
    """
    state = ctx.deps
    state.last_output = out

    mi = state.current_result.get("match_info") or {}
    final_inl = mi.get("n_inliers", 0) or 0

    # The model sometimes hallucinates that it called verify_position. Override
    # the schema fields with real state.
    if out.verify_position_called != state.verify_position_called:
        out.verify_position_called = state.verify_position_called
    if out.rotation_checked != state.rotation_checked:
        out.rotation_checked = state.rotation_checked
    if out.final_n_inliers != final_inl:
        out.final_n_inliers = final_inl

    # district_lookup requires verify_position — catches cases where the
    # reader mis-flagged district-wide and lookup_district returned a 900 km²
    # polygon when the real boundary is a single site.
    if out.status == "district_lookup":
        if state.current_result.get("geojson") is None:
            raise ModelRetry(
                "status='district_lookup' requires a successful lookup_district "
                "call that produced a GeoJSON. Call lookup_district with the "
                "district_name from the PDFInfo and retry."
            )
        if not state.verify_position_called:
            raise ModelRetry(
                "status='district_lookup' requires you to call verify_position "
                "first. Look at the OS tile with the district polygon overlaid, "
                "then compare against the planning map. If the district polygon "
                "is dramatically larger than what the map shows, note your "
                "concern in visual_check_notes but still submit. The pipeline "
                "always produces a polygon. Call verify_position now, fill "
                "visual_check_notes, then resubmit."
            )
        if len(out.visual_check_notes.strip()) < 20:
            raise ModelRetry(
                "district_lookup requires visual_check_notes (≥20 chars) "
                "describing whether the district polygon matches the planning "
                "map's apparent scope."
            )
        return out

    # status == "accepted" from here on.

    if final_inl == 0 and state.current_result.get("geojson") is None:
        raise ModelRetry(
            "Cannot accept: no successful match_at + commit_match has produced "
            "a result. Run positioning to completion (propose_centers → "
            "match_at → commit_match → extract_boundary → project_boundary). "
            "Even if all match_at scores are low, commit the highest-scoring "
            "one anyway and proceed — the pipeline always produces a polygon."
        )

    # Borderline positioning (25-100 inliers) must be manually verified.
    if 25 <= final_inl <= 100:
        if not state.verify_position_called:
            raise ModelRetry(
                f"Positioning produced {final_inl} inliers (borderline band 25-100). "
                f"You MUST call verify_position to visually compare the OS tile "
                f"against the planning map before accepting. Call verify_position "
                f"now, compare the road/feature patterns, then resubmit with "
                f"verify_position_called=True and visual_check_notes describing "
                f"the comparison. If features do NOT match, still submit with "
                f"status='accepted' — note the mismatch in visual_check_notes. "
                f"The pipeline always produces a polygon."
            )
        if len(out.visual_check_notes.strip()) < 20:
            raise ModelRetry(
                f"verify_position was called but visual_check_notes is too short "
                f"(len={len(out.visual_check_notes.strip())}). Describe in at "
                f"least 20 characters whether the OS tile features match the "
                f"planning map (road patterns, settlement shape, named roads)."
            )

    return out


@_agent.system_prompt
def build_system_prompt(ctx: RunContext[AgentState]) -> str:
    return WORKER_SYSTEM_PROMPT
