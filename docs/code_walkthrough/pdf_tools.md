# `tools/pdf_tools.py`

**55 lines.** Two helpers for getting at PDF content: render a page to an
image, and find the PDF for a given case folder. The "render" function is
called from many places (agent.py, locate.py, dataset scripts) — this file
exists to keep that one operation in one spot instead of duplicating the
fitz/pixmap dance.

## Public API

- `render_pdf_page(pdf_path, page_index, dpi=200)` → BGR ndarray.
- `find_pdf_for_case(case_folder, eval_dir=None)` → str path or `None`.

## Function walkthroughs

### `render_pdf_page(pdf_path, page_index, dpi=200)`

```
Input:  PDF path, 0-indexed page number, render DPI
Output: numpy array, shape (H, W, 3), BGR colour order, dtype uint8
Raises: IndexError if page_index is out of range
```

Two implementations under one signature:

1. **PyMuPDF (`fitz`) path** — preferred. `fitz.open(path).get_pixmap(dpi=dpi)`
   gives a `Pixmap` with raw RGB(A) bytes. Reshape to a 3D array, then
   convert to BGR (OpenCV's preferred channel order):
   - 4 channels → drop alpha via `COLOR_RGBA2BGR`
   - 3 channels → swap R/B via `COLOR_RGB2BGR`
   - The `try / finally doc.close()` ensures the file handle is released
     even if reshape or conversion blows up.

2. **`pdf2image` fallback** — only triggers if the `import fitz` fails
   (e.g. in environments where MuPDF isn't installed). `convert_from_path`
   is slower (forks a subprocess for ImageMagick) but works as a last
   resort. Used basically never in practice.

### `find_pdf_for_case(case_folder, eval_dir=None)`

Looks inside `evaluation_data/<case_folder>/` and returns the path to the
first `.pdf` file there. Returns `None` if the folder doesn't exist or has
no PDF.

If `eval_dir` is `None`, defaults to `<repo>/evaluation_data/` — computed
relative to this file's location, so it works regardless of which cwd the
caller invokes it from.

Used by `scripts/auto_label_boundary_dataset.py` and similar batch scripts
that iterate over cases.

## Why this design

**One function, multiple callers.** Before consolidation there were 4
different sites in the codebase doing roughly the same thing
(`fitz.open + get_pixmap + cv2.cvtColor`). Putting it in one helper means:

- **Bug fixes apply everywhere** — e.g. if the RGBA→BGR conversion was
  wrong, you fix it once.
- **Consistent error handling** — out-of-range page indices raise
  `IndexError` uniformly; callers can catch one type.
- **Easier to swap implementation** — if you wanted to switch to a
  different PDF library, this is the one place to change.

**Why not return RGB?** OpenCV (used downstream by SAM3 wrappers, mask
operations, drawing functions) expects BGR. Doing the conversion at the
render boundary means downstream code never has to think about channel
order. The cost is one extra `cvtColor` call (~ms).

**Why default DPI 200?** Empirically the sweet spot for UK planning maps:
- Below 150: small road labels become unreadable for OCR.
- Above 300: file size + memory blow up; SAM3 slows linearly with pixel
  count.
- 200 is what the production agent uses by default. Higher DPIs (e.g. 400
  for OCR-heavy paths in `locate.py`) are passed explicitly when needed.

## Tests

There aren't any unit tests for this file. It's exercised through every
benchmark run — if it broke, the whole pipeline would fail loudly.
