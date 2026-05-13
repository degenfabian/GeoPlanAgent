# `tools/boundary_color.py`

**75 lines.** Detects an explicit red/blue/magenta boundary line drawn on a
planning map and returns it as a filled mask. Used as a fallback when SAM3
misfires — many UK planning maps have a hand-drawn coloured boundary that's
much more reliably found by colour thresholding than learned segmentation.

## Public API

- `extract_color_boundary(img_bgr, colors=("red","blue","magenta"), min_area_px=500)`
  — only thing other code calls. Returns a uint8 mask (255 inside, 0 outside)
  the same shape as the input image, or `None` if no closed coloured region
  was found in any of the requested colours.

## How it works

### `_COLOR_RANGES` (module-level dict)

For each colour name maps to a list of `(h_lo, h_hi, s_min, v_min, s_max, v_max)`
tuples. These are HSV thresholds. Tuned empirically on UK planning PDFs:

- **red wraps around H=0/180** in HSV, so two ranges (0-12 and 165-180) are
  combined.
- **blue** is roughly 95-135.
- **magenta** is 140-170.
- All three require S≥80 and V≥60 to filter out anti-aliasing edges around
  black ink (which has low saturation).

Tweak these if you find a colour the function misses on your PDFs.

### `_color_mask(img_bgr, color)`

1. Converts the BGR image to HSV.
2. ORs together every threshold range for the requested colour into a
   boolean mask.
3. Returns it as uint8 with 255 inside, 0 outside.

The "OR over multiple ranges" is what handles red wrapping around H=0/180.

### `_largest_contour_filled(binary, min_area_px=500)`

Takes a noisy binary mask of coloured pixels and tries to extract a single
filled boundary polygon:

1. **Morphological close** with a 15×15 ellipse kernel, 2 iterations —
   bridges small gaps in the line where the colour pixels aren't perfectly
   connected (anti-aliasing, JPEG artefacts).
2. **Connected components** — find every connected blob.
3. **Keep only the largest** if its area is ≥ `min_area_px`. Anything smaller
   is annotation arrows / colour ticks in the legend, not the boundary.
4. **`findContours` + `drawContours(...FILLED)`** — turn the line into a
   filled region.

Returns `None` if no blob is large enough — that's the signal to fall back
to SAM3.

### `extract_color_boundary(img_bgr, colors, min_area_px)`

Tries each colour in `colors` (default `("red", "blue", "magenta")`) in
order. Returns the first one that produced a usable filled region. Returns
`None` if all three fail — caller should then fall back to SAM3 results.

The "first match wins" order matters: red is the most common annotation
colour on UK planning maps, so it's tried first. If your dataset has
predominantly blue boundaries, swap the order.

## Why this design

**Self-gating.** The function returns nothing if no closed coloured region
exists, so callers can plug it in as a fallback unconditionally:

```python
sam_mask = extract_candidates(...)
if not sam_mask_looks_good(sam_mask):
    color_mask = extract_color_boundary(plan_img)
    if color_mask is not None:
        use color_mask
    else:
        use the best of the SAM candidates
```

**No GT.** The whole pipeline is purely image-based — no ground truth
involved. Deployable as-is.

**Why it's a separate module from `sam3_boundary.py`.** SAM3 is a heavy
neural-net dependency (PyTorch, MPS, hundreds of MB of weights). Colour
extraction is pure OpenCV and runs in milliseconds. Keeping them apart
means you can load + run colour extraction without touching SAM3 — useful
for quick batch processing or smoke tests.

## Recovery context

Phase 30 of the recovery experiment showed colour-boundary extraction
rescued 5 stuck cases in the v12 benchmark (e.g. `A4D6` 0.000 → 0.974,
`FDBC0FDC` 0.045 → 0.936) where SAM3 had picked the wrong region but the
red line on the original PDF was clear and closed. The integration into
production was guided by this experiment.
