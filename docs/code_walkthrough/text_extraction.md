# `tools/text_extraction.py`

**317 lines.** Extracts text from a PDF page-by-page. Tries fitz's native
text extraction first (fast, works on PDFs with embedded text) and falls
back to a configurable OCR cascade (Vision → easyocr → paddle → tesseract)
for image-only or scanned pages.

## Public API

| Function | Purpose |
|---|---|
| `extract_text_per_page(pdf_path, ...)` | per-page text + extraction-method tag |
| `format_for_reader_prompt(pages)` | format pages as a single prompt string |

The agent's reader phase uses both: it calls `extract_text_per_page` to
get text, formats it via `format_for_reader_prompt`, and feeds that to
the reader LLM along with the rendered images.

## OCR cascade

The four OCR backends are tried in order, falling through to the next on
failure:

### `_ocr_page_vision(img_bgr)` (line 69)

macOS Vision framework via pyobjc. ANE-accelerated, ~0.5-2s per page,
quality on par with PaddleOCR. **Default backend on macOS.** Returns None
on non-macOS or if pyobjc isn't installed.

The function:
1. Encodes the image as PNG bytes.
2. Wraps in NSData → CIImage.
3. Builds a `VNRecognizeTextRequest` with accurate level + UK English
   language hint.
4. Calls the handler.
5. Extracts the top-1 candidate string per recognised text observation.

The recovery effort showed Vision is much better than tesseract on map
labels — it found 3 distinct road names where tesseract found 1.

### `_ocr_page_easyocr(img_bgr)` (line 123)

PyTorch-based, MPS-accelerated. ~1-3s per page. Used as the second
fallback. Returns None if easyocr isn't installed.

### `_ocr_page_paddle(img_bgr)` (line 146)

PaddleOCR. CPU-only, slower (~25-55s per page) but robust on hard scans
where Vision/easyocr fail.

### `_ocr_page_tesseract(img_bgr)` (line 50)

The classical fallback. Lowest quality, no GPU, but always available.

## `_render_page_for_ocr(page, dpi)` (line 184)

Render a fitz `page` object (already opened) to a BGR numpy array. Used
in the iteration over the doc — re-opening the PDF each time would be
wasteful. Handles 4-channel and 1-channel pixmaps in addition to standard
RGB.

## `_safe_dpi(pdf_path, page_idx)` and similar

Pick a DPI that keeps the rendered page under ~50 megapixels. Some
historic OS sheets are A1 or larger; rendering them at 700 DPI would
produce a 150 MP image that takes 15 minutes for tesseract. Floor of 250
DPI (below that, OCR can't read graticule labels).

## `extract_text_per_page(pdf_path, use_cache=True, backend="vision", verbose=False)` (line 198)

The main entry point.

### Cache (lines 223-228)

Computed via `_cache_path(pdf_path)` — md5 of the PDF path → JSON file
under `cache/text_extraction/`. If the cached file exists and parses,
return it immediately. Skips re-OCR on benchmark re-runs.

### Cascade selection (lines 230-244)

```python
cascades = {
    "vision":    [vision, easyocr, paddle, tesseract],
    "easyocr":   [easyocr, paddle, tesseract],
    "paddle":    [paddle, easyocr, tesseract],
    "tesseract": [tesseract],
}
```

Caller picks the preferred backend; the cascade falls through automatically.

### Per-page loop (lines 246-290)

For each page:
1. Try fitz's `page.get_text()` first. If it returns ≥ `OCR_FALLBACK_THRESHOLD`
   characters, use that — fast and exact.
2. Otherwise, render the page and run the OCR cascade until one succeeds.
3. Tag the result with the method used (`"fitz"`, `"ocr_vision"`, etc.)
   so downstream code can know whether to trust it.
4. Truncate at `MAX_TEXT_PER_PAGE` to keep prompt sizes reasonable.

Returns `[{"page": int, "text": str, "method": str, "chars": int}, ...]`.

### Cache write (line 293)

After all pages are extracted, write the result to the cache.

## `format_for_reader_prompt(pages)` (line 294)

Format the page list as a single block of text for the reader LLM:

```
=== Page 1 (fitz, 4823 chars) ===
TOWN AND COUNTRY PLANNING (GENERAL PERMITTED ...
[content]

=== Page 2 (ocr_vision, 312 chars) ===
[content]
```

The method tag is shown so the LLM can adjust trust level (fitz text is
exact; ocr_* text may have OCR artefacts).

## Why this design

**Why a cascade?** No single OCR engine is best on every page type.
Vision wins on map labels but isn't available on Linux. easyocr is fast
on most scans. Paddle wins on extreme cases. Falling through means we
get the best available extraction without manual tuning per case.

**Why caching?** OCR is expensive (~5-30s per page). Benchmark re-runs
on the same dataset would re-OCR every PDF without it. The cache key is
md5 of the path, so different PDFs at the same path get fresh results.

**Why fitz first instead of OCR?** Fitz reads the embedded text layer
when one exists — exact, instant, no OCR errors. Many UK planning PDFs
have a text layer; only the map images need OCR. The threshold check
(`>= OCR_FALLBACK_THRESHOLD`) handles cases where the text layer is
present but trivial (e.g. just a page number).

**Why per-page method tags?** Lets the agent prompt distinguish high-trust
text from possibly-wrong OCR. Saw this matter for case 11 in the v13 run
— the agent picked a worse anchor partly because OCR misread a postcode.
