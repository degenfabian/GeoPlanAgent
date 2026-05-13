# `tools/map_crop.py`

**155 lines.** Detects the "Article 4 Direction" title block / legend
panel that occupies the right ~30% of many UK planning maps, and crops it
off so MINIMA only sees the map portion. The rationale (from the file's
docstring): keypoints inside the title block are dense, text-shaped, and
match against noise in OS tiles — diluting the inlier count and pushing
wrong-area windows up the ranking.

## Public API

- `detect_title_block_crop(bgr, *, debug=False)` →
  `(cropped_bgr, x_offset, y_offset, info_dict)`

That's the entire surface.

## How it works

The function looks for two specific signals:

### 1. A long vertical Hough line in the right half of the page (lines 60-100)

1. **Edge detect** with Canny.
2. **Hough transform** for vertical lines (theta close to 0).
3. **Filter** for lines that are:
   - In the right half of the page (`x > w/2`).
   - At least 40% of the page height tall.
   - Closer than 20% of the width to a perfectly vertical orientation.
4. Pick the leftmost qualifying line as the candidate left-edge of the
   title block.

This catches the typical Article 4 layout where the title block has a
clean vertical border separating it from the map.

### 2. Right-region ink density check (lines 100-130)

1. Take the region to the right of the candidate vertical line.
2. Count dark pixels (text/borders) divided by region area = ink density.
3. Compare to the overall page mean ink density.
4. **Reject** the crop if the right-region density is ≥ 80% of the page
   mean — that means the right region is *map content* (similar density to
   the map), not text/legend.

This second check prevents false positives where a normal map feature
(e.g. a road running vertically near the right edge) gets misidentified as
a title-block border.

### Optional horizontal top crop (lines 130-150)

If a horizontal Hough line is detected near the top of the page (above
20% of the page height), also crop above it. Same density check applies —
this catches title bars that span the top.

### Return shape

`(cropped_bgr, x_offset, y_offset, info)`:

- `cropped_bgr` — the kept region, possibly the original image.
- `(x_offset, y_offset)` — pixel offset of the kept region's top-left in
  the original image. Used by callers if they need to adjust an affine
  built on the cropped image back to the full-page coordinate space.
- `info["cropped"]` — bool, did we actually crop?
- `info["reason"]` — diagnostic string (e.g. "vertical line at x=2760,
  density ratio 0.18").
- `info["crop_box"]` — `(x0, y0, x1, y1)` of the kept region.

## Why this design

**Conservative by default.** All thresholds are set so the function would
rather no-op than crop real map content. False negatives (missing a title
block) cost a few percent of MINIMA accuracy; false positives (cropping
the actual map) destroy positioning entirely.

**Why a vertical-line detector instead of OCR-based?** The title block has
a *clean vertical border* in 90% of cases — that's a strong, fast,
language-independent signal. OCR would be slower and more language-dependent.

**Why density check?** Hough alone has too many false positives. Adding
"the region to the right of this line must look like text" gates the crop
on the actual content, not just an artefact line.

**Why is this called from multiple places (agent.py, sam3_boundary.py,
verifier.py, critic.py)?** Each component wants to view the "map portion"
specifically — SAM3 doesn't want to grab the title-block legend as the
boundary, the verifier wants to compare like-for-like with OS tiles, etc.

## Recovery context

Originally added to fix the v10 5B10B5A8 case (mentioned in the docstring)
where a noisy title block was producing junk MINIMA matches. Now applied
universally during render, with the no-op fallback handling the cases
where there's no detectable title block.
