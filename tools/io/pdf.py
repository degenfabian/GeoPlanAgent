"""PDF rendering helpers used by the agent + dataset scripts."""

import os

import cv2
import numpy as np
from pdf2image import convert_from_path


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
            page.set_cropbox(page.mediabox)
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


def find_pdf_for_case(case_folder, eval_dir=None):
    """Find the PDF file in evaluation_data/<case>/."""
    if eval_dir is None:
        eval_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "evaluation_data",
        )
    case_dir = os.path.join(eval_dir, case_folder)
    if not os.path.isdir(case_dir):
        return None
    pdfs = [f for f in os.listdir(case_dir) if f.lower().endswith(".pdf")]
    if not pdfs:
        return None
    return os.path.join(case_dir, pdfs[0])
