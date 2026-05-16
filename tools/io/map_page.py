"""Render one map page from a planning PDF into the canonical working image.

The pipeline is render → auto_rotate → title-block crop. Every caller in
the pipeline needs this exact sequence; centralising it here removes the
4-way copy (run_agent loop, render_page tool, replay scripts).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def render_map_page(
    pdf_path: str,
    page_1based: int,
    dpi: int = 200,
    verbose: bool = False,
) -> Optional[Tuple[np.ndarray, dict, dict]]:
    """Render one page of a planning PDF into the canonical working image.

    Pipeline:
      1. fitz render at the requested DPI
      2. auto_rotate via the trained ResNet50 classifier (no-op if
         confidence is below threshold)
      3. detect_title_block_crop (no-op if no clear title block found)

    Args:
        pdf_path: path to the PDF.
        page_1based: 1-based page number to render.
        dpi: render DPI (default 200).
        verbose: pass through to auto_rotate's logger.

    Returns:
        (img_bgr, rot_info, crop_info) on success, or None if rendering
        failed (e.g. page index out of range). rot_info / crop_info are
        the dicts returned by auto_rotate / detect_title_block_crop —
        the caller can read e.g. rot_info["applied"] to know whether
        rotation was performed.
    """
    from tools.io.pdf import render_pdf_page

    page_idx = max(0, int(page_1based) - 1)
    try:
        img = render_pdf_page(str(pdf_path), page_idx, dpi=dpi)
    except IndexError:
        return None
    if img is None:
        return None

    rot_info: dict = {"applied": False}
    try:
        from tools.io.rotation_classifier import auto_rotate
        img, rot_info = auto_rotate(img, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"  rotation_classifier failed ({e!s:.80}); raw render")

    crop_info: dict = {"cropped": False}
    try:
        from tools.io.map_crop import detect_title_block_crop
        cropped, _x_off, _y_off, info = detect_title_block_crop(img)
        crop_info = info
        if info.get("cropped"):
            img = cropped
            if verbose:
                print(f"  map_crop: {info.get('reason','cropped')}")
    except Exception as e:
        if verbose:
            print(f"  map_crop failed ({e!s:.80}); no crop")

    return img, rot_info, crop_info
