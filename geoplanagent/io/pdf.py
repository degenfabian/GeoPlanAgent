"""PDF and case-file handling: page rendering via PyMuPDF (full-MediaBox at
200 DPI), map-page selection + auto-rotation for the worker, and
locating the source PDF inside an evaluation-case folder.
"""

from __future__ import annotations

import cv2
import numpy as np
from pdf2image import convert_from_path
from typing import Optional, Tuple
from pathlib import Path


def render_pdf_page(pdf_path, page_index, dpi=200):
    """Render a single PDF page as a numpy BGR image at full resolution.

    Uses PyMuPDF (fitz) for fast rendering. Falls back to pdf2image when
    fitz isn't available. Raises IndexError if page_index is out of range.
    """
    try:
        import fitz
        doc = fitz.open(pdf_path)
        try:
            if page_index < 0 or page_index >= len(doc):
                raise IndexError(
                    f"page_index {page_index} out of range "
                    f"(PDF has {len(doc)} pages)")
            page = doc[page_index]
            # Force the full MediaBox to be rendered. By default PyMuPDF
            # honours the page's CropBox, which on some PDFs is set a few
            # points smaller than the MediaBox and silently clips real map
            # content at the edges (e.g. case 3DA282…: cropbox 595×841 vs
            # mediabox 603×847, losing ~11 px on each side of the planning
            # map). set_cropbox goes through the standard rotation pipeline
            # and is a no-op when cropbox already equals mediabox.
            #
            # Some PDFs have a MediaBox in a different coordinate space than
            # the CropBox (e.g. case 5FA84190 page 6: media=(0,-1920,864,0),
            # crop=(0,0,864,1920) — Y inverted). PyMuPDF rejects with
            # "CropBox not in MediaBox" when the rects don't overlap. In
            # that case the existing CropBox is already correct (matches
            # the page's effective rect); fall through and render that.
            try:
                page.set_cropbox(page.mediabox)
            except ValueError:
                pass
            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n)
        finally:
            doc.close()
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except ImportError:
        pages = convert_from_path(
            pdf_path, dpi=dpi,
            first_page=page_index + 1, last_page=page_index + 1,
        )
        if not pages:
            return None
        return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)


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
    from geoplanagent.io.pdf import render_pdf_page

    page_idx = max(0, int(page_1based) - 1)
    try:
        img = render_pdf_page(str(pdf_path), page_idx, dpi=dpi)
    except IndexError:
        return None
    if img is None:
        return None

    rot_info: dict = {"applied": False}
    try:
        from geoplanagent.io.rotation_classifier import auto_rotate
        img, rot_info = auto_rotate(img, case_name=case_name, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"  rotation_classifier failed ({e!s:.80}); raw render")

    return img, rot_info


# Filename tokens that hint at a dedicated map/plan PDF (vs. notice or
# form documents in the same folder). The first PDF whose lowercase
# name contains any of these tokens wins. "plan" catches cases like
# A4Da2 where one file is "..._Direction_Plan.pdf" and another is a
# notice.
_MAP_TOKENS = ("map", "plan")


def resolve_case_pdf(folder_path: Path) -> Optional[Path]:
    """Pick the canonical PDF for a single evaluation case folder.

    Prefers PDFs whose filename contains 'map' or 'plan' (case-insensitive);
    falls back to the first PDF in the folder if none match.

    Returns ``None`` if the folder doesn't exist or has no PDFs.
    """
    if not folder_path.is_dir():
        return None
    pdf_files = list(folder_path.glob("*.pdf"))
    if not pdf_files:
        return None
    map_pdfs = [p for p in pdf_files
                if any(tok in p.name.lower() for tok in _MAP_TOKENS)]
    return map_pdfs[0] if map_pdfs else pdf_files[0]
