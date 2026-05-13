"""Per-page text extraction for the reader.

For each page in a PDF:
  * If born-digital (fitz returns substantial text), use it — perfect.
  * If scanned (fitz returns < OCR_FALLBACK_THRESHOLD chars), OCR the page.
    Default backend is macOS Vision framework (Apple Neural Engine
    accelerated, ~0.5-2s per page, quality ≥ PaddleOCR). Other backends
    are kept as fallbacks for portability or when Vision returns empty.

Backend ordering for OCR (controlled by ``backend`` arg):
  "vision"    — vision → easyocr → paddleocr → tesseract  (default; macOS-only,
                ANE-accelerated, fastest + highest quality)
  "easyocr"   — easyocr → paddleocr → tesseract  (PyTorch/MPS, cross-platform)
  "paddle"    — paddleocr → easyocr → tesseract  (CPU-only, slow on Mac)
  "tesseract" — tesseract only  (lowest quality, no GPU)

Results are cached on disk under ``cache/text_extraction/`` keyed by PDF
content hash, so reruns are instant.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

OCR_FALLBACK_THRESHOLD = 50    # chars below which we OCR the page
OCR_DPI = 250                  # body-text OCR — lower than the 700 DPI used
                               # for graticule labels in tools.candidates
OCR_TIMEOUT_S = 30
MAX_TEXT_PER_PAGE = 8000       # truncate per-page text to this many chars

CACHE_DIR = Path("cache/text_extraction")


def _cache_path(pdf_path: str) -> Path:
    """Hash the first 256KB of the PDF for a deterministic cache key."""
    h = hashlib.md5()
    try:
        with open(pdf_path, "rb") as f:
            h.update(f.read(256_000))
    except OSError:
        h.update(pdf_path.encode())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{h.hexdigest()}.json"


def _ocr_page_tesseract(img_bgr) -> str:
    """OCR a page image with Tesseract. PSM 6 (uniform block of text) is
    the right mode for body content; PSM 11 is better for sparse map labels
    but worse for paragraphs. Returns "" on failure (timeout / engine error)."""
    try:
        import pytesseract
        import cv2
        text = pytesseract.image_to_string(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            config="--psm 6",
            timeout=OCR_TIMEOUT_S,
        )
        return text or ""
    except RuntimeError:
        return ""  # tesseract timeout
    except Exception:
        return ""


def _ocr_page_vision(img_bgr) -> Optional[str]:
    """OCR via the macOS Vision framework (Apple Neural Engine accelerated).
    Returns None if pyobjc-framework-Vision isn't installed (e.g. on
    non-macOS platforms); "" on runtime error.

    Vision's text recognition runs on the ANE/GPU and is the fastest
    high-quality OCR on Apple Silicon — typically 0.5-2s per page with
    accuracy comparable to or better than PaddleOCR. UK English language
    hint enables postcode/road-name correction.
    """
    try:
        from Vision import (
            VNRecognizeTextRequest,
            VNImageRequestHandler,
            VNRequestTextRecognitionLevelAccurate,
        )
        from Quartz import CIImage
        from Cocoa import NSData
    except ImportError:
        return None
    try:
        import cv2
        ok, buf = cv2.imencode(".png", img_bgr)
        if not ok:
            return ""
        ns_data = NSData.dataWithBytes_length_(buf.tobytes(), len(buf.tobytes()))
        ci_image = CIImage.imageWithData_(ns_data)
        if ci_image is None:
            return ""

        request = VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)
        try:
            request.setRecognitionLanguages_(["en-GB", "en-US"])
        except Exception:
            pass  # language hint is best-effort

        handler = VNImageRequestHandler.alloc().initWithCIImage_options_(
            ci_image, None)
        success, _err = handler.performRequests_error_([request], None)
        if not success:
            return ""
        results = request.results() or []
        lines = []
        for r in results:
            top = r.topCandidates_(1)
            if top and len(top) > 0:
                lines.append(str(top[0].string()))
        return "\n".join(lines)
    except Exception:
        return ""


def _ocr_page_easyocr(img_bgr) -> Optional[str]:
    """OCR via EasyOCR (PyTorch backend, MPS-accelerated on Apple Silicon).
    Returns None if not installed; "" on runtime error.

    EasyOCR auto-detects available device with gpu=True (MPS on Mac, CUDA
    on Linux/NVIDIA, falls back to CPU). The reader is a singleton — model
    load is ~20s on first call, then ~1-3s per page on MPS.
    """
    try:
        import easyocr
    except ImportError:
        return None
    try:
        global _EASYOCR_READER
        if "_EASYOCR_READER" not in globals():
            _EASYOCR_READER = easyocr.Reader(["en"], gpu=True, verbose=False)
        # detail=0 returns just the text strings in detection (≈ reading) order
        lines = _EASYOCR_READER.readtext(img_bgr, detail=0, paragraph=False)
        return "\n".join(l for l in lines if l)
    except Exception:
        return ""


def _ocr_page_paddle(img_bgr) -> Optional[str]:
    """OCR via PaddleOCR (3.x API) if installed. Returns None if PaddleOCR
    is missing so the caller can fall back to Tesseract; returns "" on
    runtime error.

    PaddleOCR 3.x changed the API: ``predict(img)`` instead of ``ocr(img)``,
    init kwargs renamed (``use_textline_orientation`` instead of
    ``use_angle_cls``, no ``show_log``), result is a list of OCRResult
    dicts with ``rec_texts`` / ``rec_scores`` keys.
    """
    try:
        import os
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR
    except ImportError:
        return None
    try:
        global _PADDLE_READER
        if "_PADDLE_READER" not in globals():
            _PADDLE_READER = PaddleOCR(lang="en", use_textline_orientation=True)
        result = _PADDLE_READER.predict(img_bgr)
        if not result:
            return ""
        page0 = result[0]
        # OCRResult is dict-like; rec_texts is a list of strings in detection
        # (≈ reading) order. Filter very-low-confidence results.
        try:
            texts = list(page0.get("rec_texts", []))
            scores = list(page0.get("rec_scores", []))
        except Exception:
            return ""
        lines = [t for t, s in zip(texts, scores)
                 if t and (s is None or s >= 0.5)]
        return "\n".join(lines)
    except Exception:
        return ""


def _render_page_for_ocr(page, dpi: int):
    """Render a fitz page to a BGR ndarray at the given DPI."""
    import cv2
    import numpy as np
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if pix.n == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def extract_text_per_page(pdf_path: str, use_cache: bool = True,
                           backend: str = "vision",
                           verbose: bool = False) -> List[Dict[str, Any]]:
    """Return per-page text with extraction-method labels.

    Args:
        pdf_path: Path to the PDF.
        use_cache: If True, check ``cache/text_extraction/<hash>.json`` first.
        backend: Preferred OCR backend for scanned pages.
            "vision" (default) — vision → easyocr → paddleocr → tesseract.
                macOS Vision framework, ANE-accelerated (~0.5-2s per page,
                quality ≥ PaddleOCR).
            "easyocr" — easyocr → paddleocr → tesseract.
                MPS-accelerated PyTorch (~1-3s per page on Apple Silicon).
            "paddle" — paddleocr → easyocr → tesseract.
                CPU-only, slightly higher quality on hard scans
                (~25-55s per page on CPU).
            "tesseract" — tesseract only. Lowest quality, no GPU.
        verbose: Print per-page progress.

    Returns:
        List of ``{"page": int (1-based), "text": str, "method":
         "fitz"|"ocr_vision"|"ocr_easyocr"|"ocr_paddle"|"ocr_tesseract"|"failed",
         "chars": int}``. ``text`` truncated at MAX_TEXT_PER_PAGE chars.
    """
    cache_p = _cache_path(pdf_path)
    if use_cache and cache_p.exists():
        try:
            return json.loads(cache_p.read_text())
        except Exception:
            pass  # corrupted cache, re-extract

    # Build OCR backend cascade by preference
    cascades = {
        "vision":    [("ocr_vision",    _ocr_page_vision),
                      ("ocr_easyocr",   _ocr_page_easyocr),
                      ("ocr_paddle",    _ocr_page_paddle),
                      ("ocr_tesseract", _ocr_page_tesseract)],
        "easyocr":   [("ocr_easyocr",   _ocr_page_easyocr),
                      ("ocr_paddle",    _ocr_page_paddle),
                      ("ocr_tesseract", _ocr_page_tesseract)],
        "paddle":    [("ocr_paddle",    _ocr_page_paddle),
                      ("ocr_easyocr",   _ocr_page_easyocr),
                      ("ocr_tesseract", _ocr_page_tesseract)],
        "tesseract": [("ocr_tesseract", _ocr_page_tesseract)],
    }
    cascade = cascades.get(backend, cascades["vision"])

    import fitz
    out: List[Dict[str, Any]] = []
    doc = fitz.open(str(pdf_path))
    try:
        for i, page in enumerate(doc):
            t0 = time.time()
            digital_text = (page.get_text() or "").strip()
            if len(digital_text) >= OCR_FALLBACK_THRESHOLD:
                method = "fitz"
                text = digital_text
            else:
                img = _render_page_for_ocr(page, OCR_DPI)
                text = ""
                method = "failed"
                for label, fn in cascade:
                    result = fn(img)
                    if result is None:
                        # Backend not installed — skip
                        continue
                    result = result.strip()
                    if result:
                        text = result
                        method = label
                        break
                    # Empty result — try next backend
            if len(text) > MAX_TEXT_PER_PAGE:
                text = text[:MAX_TEXT_PER_PAGE] + "\n[…truncated]"
            entry = {
                "page": i + 1,
                "text": text,
                "method": method,
                "chars": len(text),
            }
            out.append(entry)
            if verbose:
                print(f"    page {i+1}: {method} ({len(text)} chars, "
                      f"{time.time()-t0:.1f}s)")
    finally:
        doc.close()

    if use_cache:
        try:
            cache_p.write_text(json.dumps(out, indent=2))
        except Exception:
            pass
    return out


def format_for_reader_prompt(pages: List[Dict[str, Any]]) -> str:
    """Render the per-page text as a single string block for the reader prompt.

    Includes the extraction method per page so the reader can weight its
    confidence (fitz = perfect; ocr_tesseract = noisy; failed = ignore).
    """
    if not pages:
        return "(no per-page text extracted)"
    sections = []
    for p in pages:
        if p["method"] == "failed" or not p["text"]:
            sections.append(f"--- Page {p['page']} (extraction failed; rely on PDF image) ---")
        else:
            method_note = {
                "fitz": "exact text from digital PDF",
                "ocr_vision": "OCR (macOS Vision — high accuracy)",
                "ocr_easyocr": "OCR (EasyOCR — high accuracy)",
                "ocr_paddle": "OCR (PaddleOCR — high accuracy)",
                "ocr_tesseract": "OCR (Tesseract — may have character errors)",
            }.get(p["method"], p["method"])
            sections.append(
                f"--- Page {p['page']} ({method_note}) ---\n{p['text']}"
            )
    return "\n\n".join(sections)
