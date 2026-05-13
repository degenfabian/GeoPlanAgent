"""OCR primitives for the locate stage.

Reads a PDF page at high DPI, runs Tesseract, and extracts OS grid refs
and printed scale text from the recognised words. Used by the v13
:func:`locate_map` cascade for analytical-affine short-circuits; the v2
cascade lives in :mod:`tools.locate.pipeline` and currently does not
re-run OCR (it reuses pdf_info from the reader).
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tools.geo.grid_ref import os_grid_ref_to_latlon
from tools.locate.schemas import OCRWord


# ─── Constants ─────────────────────────────────────────────────────────────

OCR_DPI = 700          # Tyagi & Dubey NCVPRIPG 2025 — best for fine graticule text
OCR_MAX_MP = 40        # cap for oversized pages — actual DPI scaled down
OCR_TIMEOUT_S = 30     # hard kill for tesseract; empty result cached on timeout

# OS grid ref regex: 2 letters + 2-10 digits, optional space-separation.
# Examples: "TG 210 080", "TG210080", "TG 2108", "SK 3425 6712".
_GRID_REF_RE = re.compile(
    r"\b([A-HJ-Z]{2})\s*(\d{2,5})\s+(\d{2,5})\b"
    r"|\b([A-HJ-Z]{2})(\d{4}|\d{6}|\d{8}|\d{10})\b"
)

# Scale regex: "1:2500", "1 : 2,500", "Scale 1/1250".
_SCALE_RE = re.compile(r"1\s*[:/]\s*([\d][\d,]{2,7})")

# OSGB easting/northing tick (6-digit numbers in a column/row pattern, margin-only).
_COORD_NUM_RE = re.compile(r"\b([1-9]\d{4,5})\b")


# ─── OCR runners ───────────────────────────────────────────────────────────

def _safe_ocr_dpi(pdf_path: str, page_idx: int) -> int:
    """Pick a DPI that keeps the rendered page under OCR_MAX_MP megapixels.

    A4 at 700 DPI is ~47 MP (fine). Oversized planning sheets (A2, A1, or
    odd historic OS sheets) explode at 700 DPI — a 157 MP tesseract input
    takes ~15 min. We never go below 250 DPI (below that, graticule labels
    become unreadable).
    """
    try:
        import fitz
        doc = fitz.open(pdf_path)
        page = doc[page_idx]
        # page.rect is in points (72 per inch). MP-at-1-DPI = in² / 144.
        sq_inches = (page.rect.width / 72.0) * (page.rect.height / 72.0)
        doc.close()
    except Exception:
        return OCR_DPI
    if sq_inches <= 0:
        return OCR_DPI
    max_dpi_by_mp = (OCR_MAX_MP * 1e6 / sq_inches) ** 0.5
    return max(250, min(OCR_DPI, int(max_dpi_by_mp)))


def _run_tesseract(img_bgr: np.ndarray, psm: int = 11) -> List[OCRWord]:
    """Run Tesseract and return word-level boxes. PSM 11 = sparse text.

    PSM 11 is robust for map annotations (scattered text labels, margin
    ticks) where layout analysis would fail. Low-confidence words are
    kept — downstream filters by pattern, not score.
    """
    import pytesseract
    config = f"--psm {psm} -c preserve_interword_spaces=1"
    try:
        data = pytesseract.image_to_data(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            output_type=pytesseract.Output.DICT,
            config=config,
            timeout=OCR_TIMEOUT_S,
        )
    except RuntimeError:
        # pytesseract raises RuntimeError on timeout; it also kills the child process.
        print(f"  WARN:tesseract timed out after {OCR_TIMEOUT_S}s — skipping OCR for this page")
        return []
    except Exception as e:
        print(f"  WARN:tesseract failed: {e}")
        return []
    words: List[OCRWord] = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1
        words.append(OCRWord(
            text=txt,
            x=int(data["left"][i]), y=int(data["top"][i]),
            w=int(data["width"][i]), h=int(data["height"][i]),
            conf=conf,
        ))
    return words


def _neighbouring_words(words: List[OCRWord], i: int,
                        max_gap_px: int = 40) -> str:
    """Join word i with the next word if they're on the same baseline and close.

    Helps recover OCR outputs like ['TG', '210', '080'] into 'TG 210 080'.
    """
    parts = [words[i].text]
    j = i + 1
    while j < len(words):
        prev = words[j - 1]
        curr = words[j]
        same_line = abs((prev.y + prev.h / 2) - (curr.y + curr.h / 2)) < max(prev.h, curr.h) * 0.6
        gap = curr.x - (prev.x + prev.w)
        if not (same_line and 0 <= gap <= max_gap_px):
            break
        parts.append(curr.text)
        j += 1
        if len(parts) >= 4:  # enough for "TG 21234 08765"
            break
    return " ".join(parts)


# ─── Extractors ────────────────────────────────────────────────────────────

def extract_grid_refs_from_ocr(
    words: List[OCRWord],
) -> List[Tuple[str, Tuple[float, float], Tuple[int, int]]]:
    """Find OS grid refs in OCR output. Returns (ref_text, (lat, lon), (x, y))
    for each resolved tick, with (x, y) being the grid ref text centre in
    page pixels."""
    results = []
    seen = set()
    for i, w in enumerate(words):
        # Try single token then two- and three-token joins
        for combined in (w.text, _neighbouring_words(words, i, 30),
                         _neighbouring_words(words, i, 60)):
            m = _GRID_REF_RE.search(combined)
            if not m:
                continue
            candidate = m.group(0).strip()
            if candidate in seen:
                continue
            latlon = os_grid_ref_to_latlon(candidate)
            if latlon is None:
                continue
            # Grid ref text centre — use just the first token's bbox as anchor.
            cx = w.x + w.w // 2
            cy = w.y + w.h // 2
            seen.add(candidate)
            results.append((candidate, latlon, (cx, cy)))
            break
    return results


def extract_scale_from_ocr(words: List[OCRWord]) -> Optional[Tuple[int, str]]:
    """Find the printed scale as an integer ratio (e.g. 2500 for 1:2500).

    Looks for '1:NNNN' across single words and 3-word joins. Returns
    (ratio_int, raw_text) or None.
    """
    for i, w in enumerate(words):
        for combined in (w.text, _neighbouring_words(words, i, 20),
                         _neighbouring_words(words, i, 40)):
            m = _SCALE_RE.search(combined)
            if not m:
                continue
            try:
                ratio = int(m.group(1).replace(",", ""))
            except ValueError:
                continue
            if 100 <= ratio <= 100000:
                return ratio, combined
    return None
