"""Phase 2 worker: PydanticAI Agent with all tools registered.

The worker `_agent` is decorated by the tool modules under
tools.agent.tools.{locate,match,verify,refine} at import time via
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
    again after filling the gap. Post-commit visual review is no longer
    the worker's responsibility — the optional independent critic
    (enable_critic=True) handles that role.
    """
    state = ctx.deps
    state.last_output = out

    mi = state.current_result.get("match_info") or {}
    final_inl = mi.get("n_inliers", 0) or 0

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
            "match_at → commit_match). Even if all match_at scores are low, "
            "commit the highest-scoring one anyway and proceed — the pipeline "
            "always produces a polygon."
        )

    return out


def _strip_lookup_district_mentions(prompt: str) -> str:
    """Remove every mention of lookup_district / district_lookup /
    is_district_wide from the worker prompt — used for the ablation
    where the tool is unregistered (GEOMAP_DISABLE_LOOKUP_DISTRICT=1).
    Five targeted string replacements; each MUST match the canonical
    prompt text verbatim, so a prompt edit upstream will break this
    helper and force an update.
    """
    REPLACEMENTS = [
        # 1. Tool surface line
        ("  propose_centers → match_at(page=N, …) → commit_match → return BoundaryOutcome\n"
         "plus lookup_district, reader_refine for fallback/recovery.",
         "  propose_centers → match_at(page=N, …) → commit_match → return BoundaryOutcome\n"
         "plus reader_refine for fallback/recovery."),
        # 2. OUTPUT validator block — drop district_lookup status
        ('OUTPUT: a BoundaryOutcome. The output_validator enforces:\n'
         '• status="accepted" → a commit_match call must have produced a geojson.\n'
         '• status="district_lookup" → lookup_district() must have succeeded.\n'
         'The status enum is just ["accepted", "district_lookup"] — refusing a case\n'
         'is not supported, the pipeline always produces a polygon.',
         'OUTPUT: a BoundaryOutcome. The output_validator enforces:\n'
         '• status="accepted" → a commit_match call must have produced a geojson.\n'
         'Refusing a case is not supported, the pipeline always produces a polygon.'),
        # 3. WORKFLOW step 1 caveat about district_wide
        ('   Always try positioning first, even when PDFInfo.is_district_wide=True.\n'
         '   The reader over-flags district_wide on conservation areas and named\n'
         '   neighbourhoods — positioning will find the correct sub-area. Only call\n'
         '   lookup_district as a LAST RESORT (every match_at < 0.40 AND\n'
         '   is_district_wide=True).\n\n', ''),
        # 4. WORKFLOW step 4 — drop district_lookup path
        ('4. Return BoundaryOutcome with status="accepted" (or\n'
         '   status="district_lookup" if you took the lookup_district path).\n'
         '   The pipeline always produces a polygon — downstream measures IoU on\n'
         '   whatever you commit, so don\'t refuse a case. If you suspect the\n'
         '   wrong district was looked up, call lookup_district again with a\n'
         '   different \'|\'-alternate name (or call reader_refine to confirm the\n'
         '   right district name) before submitting status="district_lookup".\n'
         '   rotation_checked is auto-overwritten from state — leave at default.',
         '4. Return BoundaryOutcome with status="accepted". The pipeline always\n'
         '   produces a polygon — downstream measures IoU on whatever you commit,\n'
         '   so don\'t refuse a case. rotation_checked is auto-overwritten from\n'
         '   state — leave at default.'),
        # 5. RE-CALLING propose_centers section
        ('This is the right move BEFORE calling lookup_district or accepting a\n'
         '0.4-score commit.',
         'This is the right move BEFORE accepting a 0.4-score commit.'),
    ]
    out = prompt
    n_replaced = 0
    for src, dst in REPLACEMENTS:
        if src in out:
            out = out.replace(src, dst)
            n_replaced += 1
    if n_replaced != len(REPLACEMENTS):
        import warnings
        warnings.warn(
            f"_strip_lookup_district_mentions matched {n_replaced}/"
            f"{len(REPLACEMENTS)} replacements — prompt may have changed "
            f"upstream. Update the helper before trusting the ablation.",
            stacklevel=2,
        )
    return out


@_agent.system_prompt
def build_system_prompt(ctx: RunContext[AgentState]) -> str:
    import os
    prompt = WORKER_SYSTEM_PROMPT
    if os.environ.get("GEOMAP_DISABLE_LOOKUP_DISTRICT") == "1":
        prompt = _strip_lookup_district_mentions(prompt)
    return prompt
