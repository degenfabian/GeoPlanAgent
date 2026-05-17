"""Render one map page from a planning PDF into the canonical working image.

The pipeline is render → auto_rotate. (Title-block cropping was removed:
the heuristic hurt as often as it helped, and SAM3 + MINIMA handle
title-block presence robustly without explicit cropping.)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def render_map_page(
    pdf_path: str,
    page_1based: int,
    dpi: int = 200,
    verbose: bool = False,
    case_name: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, dict]]:
    """Render one page of a planning PDF into the canonical working image.

    Pipeline:
      1. fitz render at the requested DPI
      2. auto_rotate via the trained ResNet50 classifier (no-op if
         confidence is below threshold). When ``case_name`` is given AND
         a k-fold rotation checkpoint dir is available, the case is
         routed to the fold that did NOT see it during training.

    Args:
        pdf_path: path to the PDF.
        page_1based: 1-based page number to render.
        dpi: render DPI (default 200).
        verbose: pass through to auto_rotate's logger.
        case_name: optional case identifier for k-fold rotation routing.

    Returns:
        (img_bgr, rot_info) on success, or None if rendering failed
        (e.g. page index out of range). rot_info is the dict returned by
        auto_rotate — the caller can read rot_info["applied"] to know
        whether rotation was performed.
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
        img, rot_info = auto_rotate(img, case_name=case_name, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"  rotation_classifier failed ({e!s:.80}); raw render")

    return img, rot_info
