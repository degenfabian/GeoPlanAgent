# tools/matching/

MINIMA-based sliding-window georeferencing. Given a rendered planning
map + a candidate centre coordinate, find the affine transform that
maps page pixels to OS-tile pixels, RANSAC the inlier set, and emit
the projected boundary as a WGS84 GeoJSON polygon.

## Public API

```python
from tools.matching import (
    load_minima,                # one-time MINIMA-LoFTR matcher load
    sliding_window_position,    # the main entry — search centres × zooms × windows
    mask_to_geojson_affine,     # project a binary mask through a committed affine
    sigma_from_scale,           # σ default given a map's stated scale
    sigma_from_source,          # σ default given a geocode source label
    effective_sigma,            # max(provided, source-default)
    candidate_passes_la_filter, # locate-stage outside-LA filter (fail-open)
)
```

The above are re-exported from `_core.py` and `source_priorities.py`
respectively — `from tools.matching import …` is the stable surface.

## How it fits in the pipeline

`tools.agent.tools.match.match_at` is the worker tool that drives this
package. Per `match_at` call (for each area_group):

1. **`load_minima()`** (once at process start) returns the LoFTR-based
   MINIMA matcher.
2. **`sliding_window_position(matcher, map_img, sam3_mask, centers,
   scale_ratio, ...)`** is the master entry:
   - Filters / dedups input centres, computes scale-aware zoom configs
     from `scale_ratio + dpi`.
   - For each centre × zoom × rotation, resizes the map to match the
     tile pixel-scale (`resize_map_to_match_zoom`), fetches an OS
     OpenData tile canvas (`tools.io.os_tiles.fetch_os_opendata_grid`),
     and slides the map across the canvas at a stride of
     `WINDOW_STRIDE_TARGET=100 px`.
   - At every window position calls `run_minima(matcher, map_img,
     tile_img)` to compute LoFTR matches; `estimate_affine` recovers
     a 2×3 RANSAC affine.
   - Keeps the best per-bucket window via a composite reranker
     (`tools.scoring.composite_window_score`).
3. **`mask_to_geojson_affine(mask, affine_H, tile_info)`** projects the
   SAM3 mask through the winning affine into a WGS84 GeoJSON
   `Feature` with `MultiPolygon` geometry. Mask cleanup primitives
   (`tools.extraction.mask_ops.*`) run inline before vectorisation.

## RANSAC affine (`estimate_affine`)

- **4-DOF similarity only** — rotation + uniform scale + translation
  via `cv2.estimateAffinePartial2D`.
- (2026-05-21) The 6-DOF full-affine fallback was removed after a
  25-case ablation showed it nets to -0.01 mean IoU and rescues only
  ~2 cases at the cost of code complexity.
- (2026-05-21) The optional Delaunay-consistency post-filter was
  removed after a 15-case ablation showed it provided zero mean
  benefit and was actively hurting the highest-inlier stress case.

## Sigma / source-priority registry (`source_priorities.py`)

Each geocoded candidate carries a `source` prefix (e.g.
`"code_point:AL1 3JE"`, `"gpkg:Camden (Town)"`,
`"emergency_la_centroid"`). The registry centralises:

| Function | Returns |
|---|---|
| `sigma_from_source(name)` | Empirical p95 candidate→GT distance for this source. Postcode lookups → ~50-300 m; place names → ~800 m; LA centroid → kilometres. Used as the search-window radius when the worker didn't supply one. |
| `source_priority(name)` | Lower = preferred. Postcodes / code_point rank 0; admin / parish rank 9. Used when capping candidate count. |
| `effective_sigma(provided, source)` | `max(provided, default-for-source)` so a worker-supplied σ never goes below the source's empirical floor. |
| `candidate_passes_la_filter(source, lat, lon, admin_region)` | Lazy-imports `tools.verification_checks._resolve_la` and checks LA-polygon containment. Returns True (fail-open) when no `admin_region` is provided or the source is exempt (postcodes, grid_refs, etc.). Used at the locate stage to drop catastrophic wrong-region picks. |

## Output of `sliding_window_position`

A dict with:

| Key | Type | Meaning |
|---|---|---|
| `affine_H` | `np.ndarray (2, 3)` or None | Page-pixel → tile-pixel affine. None when no centre passed RANSAC. |
| `tile_info` | dict | `{"image", "zoom", "tx_min", "ty_min", "tile_size_px", …}` for the winning tile canvas. |
| `match_info` | dict | `{"n_inliers", "score", "aspect", "center_latlon", "zoom", "window", "scale_factor", …}` |
| `geojson` | dict or None | Final GeoJSON (computed inline if affine + tile_info are set). |

## Tunable constants

| Constant | Home | Value | Purpose |
|---|---|---|---|
| `WINDOW_STRIDE_TARGET` | `_core.py` | 100 | Sliding-window stride target (px) |
| `OUTSIDE_LA_PENALTY` | `tools/scoring.py` | 0.3 | Smart-commit penalty (applied via `candidate_passes_la_filter`) |
