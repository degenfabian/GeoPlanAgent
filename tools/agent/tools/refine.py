"""reader_refine worker tool: focused re-consultation of the source PDF.

When the worker discovers it needs information the reader didn't surface
(e.g. a missed postcode, an unclear scale, a north-arrow check), it can
call reader_refine(question) to spawn a fresh small Gemini-Flash call
on the PDF binary. One LLM call per invocation; bounded to ≤3 calls per
case to keep cost predictable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic_ai import Agent, BinaryContent, ModelRetry, RunContext
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.usage import UsageLimits

from tools.agent.state import _agent, AgentState


REFINE_BUDGET_PER_CASE = 3


_refine_instructions = """You are a focused PDF-reading helper.

You receive ONE planning-document PDF and ONE question from a downstream
worker agent. Read whatever pages are relevant and answer the question
directly.

- If the question references a specific page (e.g. "on page 4"), focus
  there but read other pages if the answer crosses pages.
- If the question is open-ended (e.g. "any postcodes you can find"),
  scan the whole document.
- Be terse. Two-to-three sentences max. Quote the source verbatim when
  the worker is asking for an exact string (postcodes, grid refs, scale
  text, road names, etc.).
- If the answer is genuinely not in the PDF, say so plainly: "Not
  found in this PDF." Do not invent.
- Do not guess at locations or coordinates. Geocoding is the worker's job.
"""


_refine_agent: Optional[Agent] = None


def _ensure_refine_agent(model_name: str) -> Agent:
    global _refine_agent
    if _refine_agent is None:
        _refine_agent = Agent(
            "test",
            output_type=str,
            instructions=_refine_instructions,
            retries=2,
            model_settings={"temperature": 0},
        )
    return _refine_agent


@_agent.tool
def reader_refine(
    ctx: RunContext[AgentState],
    question: str,
    page_hint: Optional[int] = None,
) -> dict:
    """Ask a focused question of the source PDF when the reader missed something.

    Use this when you need information the reader did not extract — e.g.:
      - "What's the printed scale text on page 4?"
      - "Are there any postcodes anywhere in the document that PDFInfo
        is missing?"
      - "Does the map on page 3 have a north arrow, and if so which
        direction does it point?"
      - "What text appears in the title block of page 2?"

    Do NOT use this for geocoding or to ask the helper to locate places.
    The helper only reads the PDF.

    Budget: 3 refinements per case. After that, work with what you have.

    Args:
        question: Specific question to ask. Be concrete.
        page_hint: Optional 1-based page number to focus the helper on.

    Returns:
        {"success": True, "answer": str, "budget_remaining": int}
    """
    state = ctx.deps
    used = getattr(state, "refine_calls", 0)
    remaining = REFINE_BUDGET_PER_CASE - used
    if remaining <= 0:
        raise ModelRetry(
            f"reader_refine budget exhausted ({REFINE_BUDGET_PER_CASE} calls). "
            f"Proceed with the information you have."
        )

    if not question or not question.strip():
        raise ModelRetry("question is required and must not be empty.")

    pdf_path = state.pdf_path
    if not pdf_path or not Path(pdf_path).exists():
        return {"success": False, "error": "PDF binary unavailable in state."}
    pdf_bytes = Path(pdf_path).read_bytes()

    model_name = os.environ.get(
        "GEOMAP_REFINE_MODEL", "google/gemini-3-flash-preview")
    model = OpenRouterModel(model_name)
    agent = _ensure_refine_agent(model_name)

    prompt = question.strip()
    if page_hint is not None:
        prompt = f"(Focus on page {int(page_hint)}.) {prompt}"

    try:
        result = agent.run_sync(
            [BinaryContent(data=pdf_bytes, media_type="application/pdf"),
             prompt],
            model=model,
            usage_limits=UsageLimits(request_limit=3),
        )
        answer = result.output if isinstance(result.output, str) else str(result.output)
    except Exception as e:
        return {"success": False, "error": f"refine call failed: {e!s:.180}"}

    state.refine_calls = used + 1
    return {
        "success": True,
        "answer": answer.strip()[:1500],
        "budget_remaining": remaining - 1,
    }
