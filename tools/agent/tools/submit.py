"""submit_pdf_info tool: initialise per-case PDFInfo from the PDF binary.

Registered globally on `_agent`. The tool's behaviour is mode-agnostic:
when state.pdf_info is empty (folded mode), the agent reads the attached
PDF and submits a PDFInfo via this tool; pydantic-ai validates against
the schema, the result is stored on state, and the identified map_pages
are pre-rendered for the positioning tools — mirroring what
`prepare_worker_state` does in the standard reader→worker path. When
state.pdf_info is already populated (standard mode, where the reader
phase ran), the "already populated" gate errors harmlessly — the
standard worker prompt does not reference this tool so the gate is
defensive against accidental calls.
"""

from __future__ import annotations

import tempfile

import cv2
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.tools import ToolDefinition

from tools.agent.schemas import PDFInfo
from tools.agent.state import _agent, AgentState


async def _hide_unless_folded(
    ctx: RunContext[AgentState], tool_def: ToolDefinition
) -> ToolDefinition | None:
    """Make submit_pdf_info invisible to the LLM unless folded_mode is set.

    pydantic-ai calls this before each model request and uses the returned
    ToolDefinition (or None) to decide what tools to expose. Returning None
    in standard mode means the standard worker sees the same 4-tool surface
    it had before the folded ablation was added — full bit-exact parity.
    """
    if getattr(ctx.deps, "folded_mode", False):
        return tool_def
    return None


def _is_empty_pdfinfo(info: PDFInfo) -> bool:
    """True iff every PDFInfo field is at its default — i.e. the agent
    submitted essentially `PDFInfo()` without actually reading anything.

    Used as a folded-mode "did you actually look at the PDF?" gate. A
    legitimate UK planning doc always yields at least one non-default
    field (an address, postcode, road name, place name, district name,
    or map_page_details entry); an all-default submission is the
    agent punting.
    """
    return (
        not info.site_address
        and not info.postcodes
        and not info.grid_refs
        and not info.scale
        and not info.map_pages
        and not info.map_page_details
        and not info.road_names
        and not info.place_names
        and not info.is_district_wide
        and not info.district_name
        and not info.house_number_road_pairs
        and not info.parish_names
        and not info.admin_region
        and not info.likely_town_or_city
        and not info.visible_map_labels
        and not info.adjacency_hints
    )


@_agent.tool(prepare=_hide_unless_folded)
def submit_pdf_info(ctx: RunContext[AgentState], info: PDFInfo) -> dict:
    """Initialise PDFInfo for this case. One-shot per case — this tool
    populates the PDFInfo that the positioning tools (propose_centers,
    match_at, commit_match, lookup_district) read from. It is the
    required first action whenever PDFInfo is not yet populated.

    The PDF binary is attached to your first user message. Read every
    page, populate the PDFInfo schema (the full schema, including
    field descriptions and validators, is sent to you as this tool's
    parameter spec), and submit. The system validates against the
    schema, stores the result on case state, and pre-renders the
    map_pages you identified.

    If PDFInfo is already populated for this case, this tool errors —
    use the positioning tools directly. Submitting a PDFInfo with
    every field at its default also errors (it means you did not
    actually read the PDF).

    Args:
        info: PDFInfo instance with every applicable field populated by
            reading the attached PDF. See the schema for field
            semantics — postcodes, grid_refs, road_names, place_names,
            map_page_details, etc. are all required to be filled when
            present in the document.

    Returns:
        {"success": True, "map_pages_rendered": [page numbers],
         "next_step": short instruction string}
    """
    state = ctx.deps
    if state.pdf_info:
        raise ModelRetry(
            "PDFInfo is already populated for this case — do not call "
            "submit_pdf_info again. Proceed with propose_centers → "
            "match_at → commit_match."
        )

    # pydantic-ai has already validated `info` against the PDFInfo schema
    # by the time we get here (typed parameter). The remaining gate is
    # the "did you actually read the PDF?" check.
    if _is_empty_pdfinfo(info):
        raise ModelRetry(
            "You submitted a PDFInfo with every field at its default — "
            "no address, postcodes, road names, place names, district, "
            "map_page_details, or anything else. That means you did not "
            "actually read the PDF binary attached to your first user "
            "message. Open the PDF, look at every page, and extract: "
            "(a) map_page_details for EVERY page that contains map-like "
            "content (category 'match' or 'discard'), (b) the site "
            "address / road names / place names / postcodes visible in "
            "the text and on the maps, (c) is_district_wide + "
            "district_name if the document covers an entire borough. "
            "Then call submit_pdf_info again with the populated PDFInfo."
        )

    state.pdf_info = info.model_dump()

    # Mirror prepare_worker_state's render loop. We can't import
    # prepare_worker_state here without a cycle, so the render code is
    # duplicated (small, stable).
    from tools.io.map_page import render_map_page

    map_pages = state.pdf_info.get("map_pages") or []
    rendered: list[int] = []
    for page_1based in map_pages:
        result = render_map_page(
            str(state.pdf_path), int(page_1based),
            dpi=state.dpi, verbose=False, case_name=state.case_name,
        )
        if result is None:
            continue
        page_img, rot_info = result
        if rot_info.get("applied") and page_1based == map_pages[0]:
            state.rotation_checked = True
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        cv2.imwrite(tmp_path, page_img)
        state.rendered_pages[int(page_1based)] = page_img
        state.rendered_page_paths[int(page_1based)] = tmp_path
        rendered.append(int(page_1based))

    if not rendered:
        return {
            "success": True,
            "map_pages_rendered": [],
            "next_step": (
                "No map_pages identified. If you took status='district_lookup' "
                "path, call lookup_district(district_name=...). Otherwise "
                "re-examine the PDF — at least one page must be category='match'."
            ),
        }

    return {
        "success": True,
        "map_pages_rendered": rendered,
        "next_step": (
            f"Primary match page is {rendered[0]}. Now run "
            f"propose_centers → match_at(page={rendered[0]}, ...) → "
            f"commit_match → return BoundaryOutcome. The locate sub-agent "
            f"reads the rendered map image directly from state."
        ),
    }
