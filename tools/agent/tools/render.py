"""render_page worker tool.

Switches the active map page. All map_pages from the reader are pre-
rendered into state.rendered_pages during _read_pdf_phase (with
auto-rotate + title-block crop applied), so this tool is a free
state-pointer flip in the common case. Falls back to a fresh render
only for pages outside that cache.
"""

from __future__ import annotations

import tempfile

import cv2
from pydantic_ai import ModelRetry, RunContext, ToolReturn

from tools.agent.state import _agent, AgentState, _img_to_binary


@_agent.tool
def render_page(ctx: RunContext[AgentState], page: int) -> ToolReturn:
    """Switch the active map page.

    Use this when the pre-rendered first map page isn't the right one and
    map_page_details lists a better alternative (typically a 'detail' page
    deeper in the document). The page becomes the working map for all
    subsequent tools (extract_boundary, match_at, propose_centers,
    verify_position).

    Args:
        page: Page number (1-based) to switch to.

    Returns:
        Image of the page (shown to you), plus:
        {"success": true, "width": int, "height": int, "page": int,
         "from_cache": bool}
    """
    state = ctx.deps
    page = int(page)

    cached = state.rendered_pages.get(page)
    if cached is not None:
        state.map_img = cached
        state.map_crop_path = state.rendered_page_paths[page]
        h, w = cached.shape[:2]
        return ToolReturn(
            return_value={"success": True, "width": w, "height": h,
                          "page": page, "from_cache": True},
            content=[
                f"Switched to page {page} ({w}x{h}, pre-rendered):",
                _img_to_binary(cached),
            ],
        )

    from tools.io.map_page import render_map_page
    rendered = render_map_page(state.pdf_path, page, dpi=state.dpi, verbose=True)
    if rendered is None:
        raise ModelRetry(f"Page {page} could not be rendered (out of range?).")
    map_img, rot_info, _crop_info = rendered
    if rot_info.get("applied"):
        state.rotation_checked = True

    state.map_img = map_img
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        state.map_crop_path = tmp.name
        cv2.imwrite(tmp.name, map_img)
    state.rendered_pages[page] = map_img
    state.rendered_page_paths[page] = state.map_crop_path

    h, w = map_img.shape[:2]
    return ToolReturn(
        return_value={"success": True, "width": w, "height": h,
                      "page": page, "from_cache": False},
        content=[
            f"Rendered page {page} ({w}x{h}, fresh render):",
            _img_to_binary(map_img),
        ],
    )
