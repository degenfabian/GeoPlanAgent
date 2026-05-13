# `tools/visualization_tools.py`

**295 lines.** Renders predicted vs ground-truth boundaries on top of an
OS basemap as PNG. Two functions — one to draw a single GeoJSON
boundary on a tile, one to draw pred + GT side-by-side for comparison.
Used by the agent's `visualize` tool and by `benchmark_runner.py` to
produce per-case `viz_comparison.png` files.

## Public API

- `visualize_geojson_boundary(geojson, ...)` — single boundary render
- `visualize_comparison(pred_geojson, gt_geojson, ...)` — side-by-side

Both return PNGs as numpy arrays (BGR uint8) so callers can save with
`cv2.imwrite` or include them in agent prompts.

## `visualize_geojson_boundary(geojson, tile_fetcher=None, padding_m=200, output_size=1024, line_color=(0,0,255), line_thickness=4)` (line 27)

Steps:
1. **Compute the bbox** of the GeoJSON polygon in WGS84.
2. **Pad** it by `padding_m` (default 200m) so there's context around
   the boundary.
3. **Compute the right zoom level** for the bbox at `output_size` pixels
   — uses the standard slippy-map zoom formula.
4. **Fetch tiles** covering the padded bbox via `tile_fetcher` (default:
   OS Open Zoomstack via `os_opendata_tiles.fetch_os_opendata_grid`).
5. **Project polygon vertices** from WGS84 → tile pixel space.
6. **Draw** with `cv2.polylines`.
7. **Crop** to the bbox + padding so the output isn't full-tile-grid sized.

The `line_color=(0, 0, 255)` is BGR red — boundaries are conventionally
shown in red on UK maps.

## `visualize_comparison(pred_geojson, gt_geojson, ...)` (line 152)

Draws prediction (red) and GT (green) on the same OS tile background,
with a small text label showing the IoU.

Steps:
1. **Bbox = union** of pred + GT bboxes, padded.
2. **Zoom + tile fetch** as in `visualize_geojson_boundary`.
3. **Draw GT first** (green, slightly thicker) so it's visible even
   where pred overlaps.
4. **Draw pred** (red, thinner) on top.
5. **Compute IoU** and overlay it as a text annotation top-left.

Result is the standard panel you see in benchmark output: a single OS
tile with two coloured polygons and "IoU: 0.83" stamped on it.

## Why this design

**Why fetch tiles inline instead of taking them as input?** Caller
shouldn't have to know about tile coordinates and zoom levels. Pass a
GeoJSON, get a PNG.

**Why BGR colours?** OpenCV convention; saves a `cv2.cvtColor` round-trip
when writing PNGs.

**Why the IoU annotation?** When you eyeball 200 comparison panels in a
row, the score makes it instantly clear which is which (pred-good vs
pred-bad vs partial). Without it you'd be staring at vertex overlap.

**Why tile_fetcher injectable?** Lets test code swap in a fake fetcher
that returns a blank image (no GeoPackage required). Production callers
just pass `None` and get the OS Open Zoomstack default.
