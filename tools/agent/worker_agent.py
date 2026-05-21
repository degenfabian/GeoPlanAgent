"""Phase 2 worker: PydanticAI Agent with all tools registered.

The worker `_agent` is decorated by the tool modules under
tools.agent.tools.{locate,match,verify} at import time via
@_agent.tool — so this module must be importable BEFORE those tool
modules.

Defines:
  - _agent — the worker Agent instance
  - _strip_old_images — history processor that drops binary images from
    older messages to keep token cost flat
  - validate_boundary_outcome — output validator enforcing that
    status='accepted' has a committed geojson, and that
    status='district_lookup' produced a geojson via lookup_district.
    Post-commit visual review is delegated to the optional critic
    (enable_critic=True), so the worker no longer self-verifies.
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
    quadratically.

    We REBIND ``part.content`` to a fresh list rather than mutating the
    existing list in place. pydantic-ai's history-processor mechanism
    is called every turn (and the critic rehand passes
    ``worker_result.all_messages()`` back in as ``message_history`` of
    the next call), so the same list object is reached repeatedly. An
    in-place mutation cascades — each rehand iteration strips more
    history off the ORIGINAL result, and the on-disk message_log
    written later (in ``runtime.extract_message_log``) sees the
    stripped state for every prior turn. The rebind keeps the original
    list intact and only the freshly-built copy carries placeholders.
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
                # If the part is frozen (e.g. pydantic immutable
                # model), fall back to in-place mutation — better
                # one-time stripping than a token-cost blowup.
                content[:] = new_content
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
    again after filling the gap. Post-commit visual review is no longer
    the worker's responsibility — the optional independent critic
    (enable_critic=True) handles that role.
    """
    state = ctx.deps
    state.last_output = out

    # Use the union total (summed across all area_groups) when
    # available — primary-group n_inliers alone underreports
    # multi-group docs. ``commit_match`` writes ``total_inliers``
    # into ``state.current_result`` for every successful commit; for
    # old/legacy cached state we fall back to the primary-group's
    # ``match_info.n_inliers``.
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

    # district_lookup requires only that lookup_district succeeded.
    # The polygon comes from OS BoundaryLine and cannot be refined via
    # SAM3 or re-projection. If the agent suspects the wrong district
    # was looked up, the recovery is to call lookup_district again with
    # a different '|'-alternate name.
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
            "(low total_inliers), commit the highest-total_inliers one anyway "
            "and proceed — the pipeline always produces a polygon."
        )

    return out


@_agent.system_prompt
def build_system_prompt(ctx: RunContext[AgentState]) -> str:
    return WORKER_SYSTEM_PROMPT
