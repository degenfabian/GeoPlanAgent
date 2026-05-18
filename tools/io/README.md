# tools/io/

I/O for the pipeline's image and text inputs. PDF rendering, OS tile
composition, per-page OCR, and the rotation classifier wrapper.

## Public surface

| Module | Function | Purpose |
|---|---|---|
| `pdf` | `render_pdf_page(pdf_path, page_index, dpi=200)` | Render one PDF page to a numpy BGR image via PyMuPDF (fitz). Falls back to pdf2image when fitz isn't available. Forces the full `MediaBox` so PDFs with a smaller `CropBox` don't silently clip map content. |
| `text_extraction` | `extract_text_per_page(pdf_path, use_cache=True, verbose=False)` | Per-page text for the reader prompt. fitz for born-digital pages, macOS Vision (PaddleOCR fallback) for scanned ones. Cached on disk under `cache/text_extraction/` keyed by PDF content hash. |
| `text_extraction` | `format_for_reader_prompt(pages)` | Format the cache output as the `TEXT BLOCK (per page):` string the reader + `reader_refine` consume. |
| `os_tiles` | `fetch_os_opendata_grid(lat, lon, zoom, n_tiles_x, n_tiles_y)` | Render an `N×M` tile canvas centred on `(lat, lon)` from `OS_Open_Zoomstack.gpkg`. Returns `{image, zoom, tx_min, ty_min, tile_size_px, …}`. Lazy raster cache keyed by tile coords. No API key — OS OpenData is OGL v3. |
| `rotation_classifier` | `predict_rotation_cw(map_bgr, case_name=None)` | Returns 0/90/180/270 CW degrees to upright. Uses k-fold checkpoints when present (`models/rotation_classifier_kfold/`) and routes by case via `fold_assignment.json`; falls back to a legacy single checkpoint. 4-rotation TTA + 0.50 softmax-confidence abstain (returns 0 if below threshold). |
| `rotation_classifier` | `predict_rotation_with_confidence(...)` | Same prediction with the confidence + fold-routing metadata exposed. |
| `map_page` | `render_map_page(pdf_path, page_1based, dpi=200, case_name=None)` | The single source of truth for `render → auto_rotate`. Called from `prepare_worker_state` and `_get_or_render_page` in the worker. Returns `(map_img, rot_info)`. |

## How they fit together

### Reader phase

```python
from tools.io.text_extraction import extract_text_per_page, format_for_reader_prompt

pages = extract_text_per_page("doc.pdf", use_cache=True)
# → [{"page": 1, "method": "fitz", "chars": 1234, "text": "…"}, ...]

block = format_for_reader_prompt(pages)
# fed to the reader along with the raw PDF binary
```

The reader gets BOTH the OCR text block AND the PDF binary — the LLM
prefers the text block for exact strings (postcodes, grid refs, road
names) but uses the PDF image for visual fields (boundary geometry,
north arrow direction).

### Worker phase — page rendering

```python
from tools.io.map_page import render_map_page

img, rot_info = render_map_page(
    "doc.pdf", page_1based=3, dpi=200, case_name="12:00116:ART4"
)
# rot_info: {"applied": 90, "confidence": 0.93, "fold": 2, ...}
```

Title-block cropping is intentionally absent — the heuristic hurt as
often as it helped, and SAM3 + MINIMA both tolerate title-block
presence without it.

### Worker phase — tile fetching (called inside `match_at`)

```python
from tools.io.os_tiles import fetch_os_opendata_grid

tile_info = fetch_os_opendata_grid(lat, lon, zoom=17, n_tiles_x=5, n_tiles_y=5)
# tile_info["image"]: HxWx3 BGR canvas of the requested tile grid
# tile_info["tx_min"], tile_info["ty_min"]: BNG tile indices of the top-left
```

## Per-page OCR backends

`text_extraction` chooses the backend per page:

1. **fitz** (PyMuPDF) first — returns text directly from the PDF if
   the page is born-digital. ~100% accurate, no inference cost.
2. If fitz returns < `OCR_FALLBACK_THRESHOLD=50` chars → page is
   scanned. Render to PNG at `OCR_DPI=250` and OCR.
3. **macOS Vision** is preferred (Apple Neural Engine, ~0.5-2 s/page).
4. **PaddleOCR** falls back when Vision isn't available (Linux/Windows
   builds) or returns empty.
5. Pages where every backend fails are tagged
   `"(extraction failed; rely on PDF image)"` so the reader prompt
   knows to vision-OCR them itself.

Cache layout:

```
cache/text_extraction/
└── <sha1_of_pdf_bytes>.json   # {pages: [{page, method, chars, text}, ...]}
```

Re-runs hit the cache and skip OCR entirely — important for the
benchmark (~60% of cases are scanned).

## Rotation classifier (`rotation_classifier`)

- **Model**: ResNet50 (ImageNet pretrained), 4-way head (0°/90°/180°/270° CW).
- **TTA**: predict on the input AND its 90/180/270° CW rotations,
  cyclically shift each rotated prediction back to the original frame,
  ensemble (mean softmax), return top class.
- **Confidence abstain**: if the top class softmax probability is
  below `threshold=0.50`, return `0` (don't rotate). Safer than
  rotating wrongly — a few-degree off page is still positionable, a
  90°-wrong page never is.
- **Fold routing**: when `case_name` is given and the k-fold dir is
  present, route the case to the fold that did NOT see it in training
  (mirrors the SAM3 k-fold loader). Falls back to the single-
  checkpoint legacy path otherwise.

## OS-tile rendering (`os_tiles`)

Tiles are rendered from `os_opendata/OS_Open_Zoomstack.gpkg` at
inference time, not pre-rasterised. Styling matches UK planning-map
conventions — pink buildings, road casings, water, woodland — so
MINIMA / LoFTR can match scanned planning maps against tile imagery
with minimal cross-modal gap.

Disk cache: rasterised tiles are persisted under `cache/zoomstack/`
keyed by `(zoom, tx, ty)`. The full benchmark at production scale
populates ~200 GB of tile cache; subsequent runs are network-free.
