"""render_page worker tool.

Extracted from ``tools/agent.py`` (stage-2 split, 2026-05-11). Registers
``render_page`` against the shared ``_agent`` instance at import time.
"""

from __future__ import annotations

import os
import tempfile

import cv2
from pydantic_ai import ModelRetry, RunContext, ToolReturn

from tools.agent.state import _agent, AgentState, _img_to_binary


# ── Tool 1: render_page ────────────────────────────────────────────────────

@_agent.tool
def render_page(ctx: RunContext[AgentState], page: int) -> ToolReturn:
    """Render a page from the planning PDF as an image.

    Use this after reading the PDF to render the page containing the site/location
    map. The map is usually on the LAST page or near the end. Use 1-based page
    numbering (first page = 1).

    The rendered image becomes the working map for all subsequent tools
    (extract_boundary, match_at, visualize). Auto-rotation is
    applied during rendering so the map you see is upright.

    Args:
        page: Page number (1-based) to render.

    Returns:
        Image of the rendered page (shown to you), plus:
        {"success": true, "width": int, "height": int, "page": int}
    """
    state = ctx.deps
    # No dedup on render_page: rendering is cheap (~100ms) and idempotent.
    # The agent legitimately needs to re-render the same page after touching
    # a different one (multi-page workflows). Dedup'ing here meant the
    # SECOND render_page(N) was a no-op even when the agent expected it
    # to switch back, so subsequent extract/position calls used whatever
    # page was rendered in between — silent page-swap bug.

    page_idx = max(0, page - 1)  # convert 1-based to 0-based
    from tools.io.pdf import render_pdf_page
    try:
        map_img = render_pdf_page(state.pdf_path, page_idx, dpi=state.dpi)
    except IndexError as e:
        raise ModelRetry(str(e))

    # Auto-rotate via the trained classifier. Returns the same image
    # if confidence is below threshold (safer to leave a map alone
    # than rotate it wrongly). The classifier replaces the agent's
    # old rotate_map tool — agent decisions on rotation were a footgun
    # that destroyed several v9 cases vs v7.
    try:
        from tools.io.rotation_classifier import auto_rotate
        map_img, rot_info = auto_rotate(map_img, verbose=True)
        if rot_info["applied"]:
            state.rotation_checked = True
    except Exception as e:
        # Don't fail rendering if the classifier is unavailable; log and
        # proceed with the unrotated map (same as before the classifier
        # existed).
        print(f"  rotation_classifier unavailable ({e!s:.80}); "
              f"proceeding with raw render")

    # Crop title-block / signature panel if there's a clear vertical border
    # in the right half of the page AND the region right of it is ink-sparse
    # (text/whitespace, not map content). Targets v10 case 5B1 where the
    # title block at x=2755 (60% of page width) created MINIMA-noise features
    # that prevented the correct match from winning. Conservative: returns
    # unchanged image when no clear title block is detected. Run AFTER
    # rotation so the detection works in the upright frame.
    try:
        from tools.io.map_crop import detect_title_block_crop
        cropped, _x_off, _y_off, _crop_info = detect_title_block_crop(map_img)
        if _crop_info.get("cropped"):
            print(f"  map_crop: {_crop_info['reason']}")
            map_img = cropped
    except Exception as e:
        print(f"  map_crop unavailable ({e!s:.80}); proceeding without crop")

    state.map_img = map_img

    # Save to temp file for SAM3
    if state.map_crop_path and os.path.exists(state.map_crop_path):
        os.unlink(state.map_crop_path)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        state.map_crop_path = tmp.name
        cv2.imwrite(tmp.name, map_img)

    h, w = map_img.shape[:2]

    return ToolReturn(
        return_value={"success": True, "width": w, "height": h, "page": page},
        content=[
            f"Rendered page {page} ({w}x{h} pixels):",
            _img_to_binary(map_img),
        ],
    )


# Tool 1b (rotate_map) removed. Auto-rotation is now done by a trained
# ResNet50 classifier (TTA + confidence threshold) inside render_page.
# The agent making rotation decisions visually was a footgun: across
# v9 it rotated 5+ cases wrongly that v7 had handled correctly. The
# trained classifier is right >95% of its kept predictions and abstains
# (no-op) when uncertain.
