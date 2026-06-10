"""PDF rendering helpers used by the agent + dataset scripts."""


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
