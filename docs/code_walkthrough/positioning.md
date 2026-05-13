# `tools/positioning.py`

**1305 lines.** The geometric heart of the pipeline. Three responsibilities:

1. Run **MINIMA** (a cross-modal feature matcher) to find where on an OS
   tile a planning page sits.
2. Build **affine transforms** mapping page pixels to tile-canvas pixels
   (and ultimately to lat/lon).
3. Project **SAM3 masks** through those affines into GeoJSON polygons.

Plus two new things added during recovery integration:
- `analytical_affine_from_anchor` — bypass MINIMA when an exact site
  anchor is known.
- Mask post-processing (`_keep_dominant_components`, `_expand_thin_mask`,
  `_fill_mask_holes`) that runs inside `mask_to_geojson_affine`.

## Public API

| Function | Purpose |
|---|---|
| `load_minima(base_dir)` | load MINIMA-LoFTR matcher |
| `sliding_window_position(matcher, map, sam3_mask, centers, ...)` | end-to-end MINIMA loop |
| `mask_to_geojson_affine(mask, affine_H, tile_info)` | project mask → GeoJSON |
| `analytical_affine_from_anchor(...)` | build affine without MINIMA |
| `compute_map_mpp(scale_ratio, dpi)` | metres per page pixel |
| `best_zoom_for_scale(map_mpp, lat)` | OS tile zoom for a map scale |
| `sigma_from_scale(scale_ratio)` | search radius for MINIMA |
| `osm_pixel_to_latlon(px, py, zoom, tx_min, ty_min)` | inverse projection |

## MINIMA loading and inference

### `load_minima(base_dir=None)` (line 30)

Loads MINIMA-LoFTR from `MINIMA/weights/minima_loftr.ckpt`. Sets up the
PyTorch model on MPS/CUDA/CPU, runs in eval mode. Singleton-ish (the
loaded matcher object is passed around).

### `run_minima(matcher, map_img, tile_img, grayscale=False)` (line 55)

One forward pass: takes a planning-page crop and an OS-tile crop, returns
matched keypoints `(mkpts0, mkpts1)` and per-match confidences `mconf`.

If `grayscale=True`, both images are converted to grayscale first — useful
for very stylised planning maps where colour confuses the matcher.

### `estimate_affine(mkpts0, mkpts1, mconf, reproj_thresh=10.0)` (line 88)

Take the keypoints and fit a 2D affine via `cv2.estimateAffinePartial2D`
(similarity transform: rotation + uniform scale + translation, no shear).
RANSAC handles outliers.

Returns `(affine_H, n_inliers)` or `(None, 0)` on failure.

## Scale + zoom math

### `compute_map_mpp(scale_ratio, dpi=200)` (line 122)

Metres per page pixel:
```
mm_per_px = 25.4 / dpi             # 25.4 = 1 inch in mm
m_per_px  = mm_per_px / 1000 * S   # at scale 1:S
```

### `best_zoom_for_scale(map_mpp, lat)` (line 134)

OS tile pixels are roughly `156543 * cos(lat) / 2^zoom` metres each.
Choose the zoom that gives a tile pixel size closest to `map_mpp`.
Clamped to [15, 19] — outside this range OS Zoomstack tiles aren't
useful.

### `sigma_from_scale(scale_ratio, page_mm=(297, 210))` (line 142)

How far MINIMA should search around a candidate center. The geocoded
center might be at an arbitrary point on the map; the diagonal of an A4
page at 1:S is `0.36 * S` metres, so the search needs to cover at least
half-diagonal.

Formula: `max(2500m, 0.18 * S)`. Floor of 2500m was raised after rural
v10 cases (geocoded village centre 1900m from GT) failed at smaller
sigmas.

### `_latlon_to_global_tile_pixel(lat, lon, zoom, tile_size)` (line 179)

Slippy-map projection: WGS84 → global tile pixel coordinates. Used by
`analytical_affine_from_anchor` to compute where an anchor lat/lon falls
in the OS tile coordinate system.

### `osm_pixel_to_latlon(px, py, zoom, tx_min, ty_min, tile_size)` (line 274)

Inverse: tile pixel → WGS84. Used by `mask_to_geojson_affine` to convert
each contour vertex back to lat/lon for the GeoJSON output.

## Analytical affine (recovery-integration addition)

### `analytical_affine_from_anchor(plan_shape, mask_centroid_xy, anchor_lat, anchor_lon, scale_ratio, dpi=200, rotation_deg=0.0, zoom=None, tile_size=256, n_tiles=35)` (line 189)

Constructs the affine geometrically — no MINIMA needed.

Math:
- `s = map_mpp / tile_mpp` — tile pixels per page pixel.
- `R = rotation matrix * s` — combined scale and rotation.
- `t = anchor_pixel - R @ centroid` — translation that puts the SAM mask
  centroid at the anchor's lat/lon.

Returns `(affine_H, tile_info)` with the same shape as `sliding_window_position`'s
output, so `mask_to_geojson_affine` works unchanged.

Triggered by the agent's `_try_analytical_affine` when:
- `pdf_info.grid_refs` parses as exact OS easting/northing (via
  `parse_easting_northing` in `geo_tools.py`).
- `pdf_info.scale` parses as a numeric ratio.
- A SAM mask is set so we have a centroid.

If any condition fails → `None` → caller falls through to MINIMA.

## Mask post-processing

These three functions run inside `mask_to_geojson_affine`. They were
crystallised from the recovery experiment (especially Phase 19's mask-cleanup
sweep) — found big IoU gains.

### `_keep_dominant_components(mask, min_frac_of_largest=0.05)` (line 390)

Drop noise blobs:
1. Connected-components label.
2. Find the largest area.
3. Drop any blob smaller than `min_frac_of_largest * largest_area`.

Default 0.05 means "anything < 5% of the largest is noise". Was the
single biggest individual win during recovery (Phase 1, +1pp on its own).

### `_expand_thin_mask(mask)` (line 328)

If the mask is a thin outline (low fill ratio), dilate it into a filled
region. Same idea as `try_fill_boundary_outline` in `sam3_boundary.py`
but lives here because it's run on every projection regardless of which
candidate-source produced the mask.

### `_fill_mask_holes(mask)` (line 294)

Morphological close + fill internal holes. Catches the "boundary outline
with text gaps inside" case where the mask has the right outline but
hollow.

## Mask → GeoJSON

### `mask_to_geojson_affine(mask, affine_H, tile_info, simplify_px=3.0)` (line 434)

The complete mask → GeoJSON conversion:

1. Run `_keep_dominant_components` (drop noise).
2. Run `_expand_thin_mask` (fill outlines).
3. Run `_fill_mask_holes` (close gaps).
4. `cv2.findContours` on the cleaned mask.
5. For each contour:
   - Skip if `cv2.contourArea < 100` (tiny noise).
   - `cv2.approxPolyDP(contour, simplify_px, True)` (Douglas-Peucker
     simplification — reduces vertex count).
   - For each vertex: `osm_pixel = affine_H @ [px, py, 1]`.
   - `osm_pixel_to_latlon` → `(lat, lon)`.
   - Close the polygon (first point = last point).
6. Wrap as GeoJSON `Feature` (single Polygon if 1 contour, MultiPolygon
   if more).

This is the function downstream of EVERY positioning path (MINIMA, fresh
SAM, color, analytical) — they all eventually feed a (mask, affine, tile_info)
through it.

## Center filtering / dedup helpers

These are small utilities used by `sliding_window_position`:

- `filter_centers(centers, max_centers, max_dist_km)` (line 495) — limit
  centers to top-K by some criterion + drop centers far from a "primary"
  one if there's an obvious cluster.
- `_deduplicate_centers(centers, min_dist_m=500)` (line 586) — merge
  centers within 500m of each other.
- `_center_specificity(name)` (line 629) — score a center label's
  precision: postcodes/grid-refs are most specific (3), road names (2),
  district names (1), etc.
- `filter_centers_by_specificity(centers, anchor_threshold=2)` (line 661)
  — prefer specific anchors when both are available.

## Road-verification

Two functions used during candidate ranking:

- `_query_gpkg_road_names(lat, lon, radius_m=1500)` (line 709) —
  pull road names from OS Open Zoomstack within a radius.
- `_fuzzy_road_match(llm_name, reference_names)` (line 741) — match
  a name from the PDF (might be misspelled or use a different style)
  against the OS reference names.
- `_verify_candidates_with_road_names(ranked_candidates, road_names)`
  (line 764) — boost candidates whose matched window has the expected
  roads nearby.

## `sliding_window_position(matcher, map_img, sam3_mask, centers, scale_ratio, dpi=200, rotations=None, road_names=None, tile_fetcher=None, grayscale=False, return_candidates=False)` (line 869)

The big one. End-to-end MINIMA loop:

1. **For each center** (lat, lon, sigma):
   - **For each candidate zoom** (best-for-scale + 1-2 neighbours):
     - **Fetch OS tile grid** centered on the candidate.
     - **For each rotation** in `rotations` (default `[0, 90, 180, 270]`):
       - Resize the map to match tile MPP.
       - **For each window position** in a sliding pattern within sigma:
         - Run MINIMA on (map_resized_rotated, tile_window).
         - Estimate affine from keypoints.
         - Score the match (n_inliers + spread).
   - **Track the best window** for this center.
2. **Aggregate across centers** — pick the winning (center, zoom, rotation,
   window) by `n_inliers`.
3. **Construct the canonical affine** by combining the winning rotation
   with `_build_scale_H` (resize back to original page).
4. **Project the SAM mask** through the affine via `mask_to_geojson_affine`.
5. **Return** a dict with `affine_H`, `tile_info`, `match_info` (n_inliers,
   score, scale, etc.), and the resulting `geojson`.

This is what the legacy `position_boundary` tool calls. The v2 path
(`match_at` + `commit_match`) calls it once per probe center.

## Why this design

**Why does `mask_to_geojson_affine` always run mask cleanup?** The mask
post-processing changes IoU by ~5pp on average. Skipping it for callers
that "have a clean mask" saves no time and breaks the universal contract
that a mask-shaped numpy array maps to a clean polygon.

**Why is `analytical_affine_from_anchor` in this file and not its own?**
It produces `(affine_H, tile_info)` — same return shape as the MINIMA
path. Putting it in `positioning.py` lets callers swap one for the other
without a separate import.

**Why so many center-filter functions?** Different stages need different
filters. `propose_centers` collects ALL candidates from many sources;
`filter_centers_by_specificity` picks the top-N for the agent to probe;
`_deduplicate_centers` merges near-duplicates. Each one is small and
single-purpose.

**Why the `sliding_window_position` mega-function instead of breaking it
up?** It's read top-to-bottom as a pipeline. Splitting into 5 helper
functions would mean the reader has to follow indirection and reconstruct
the flow mentally. Single linear function is easier to debug.
