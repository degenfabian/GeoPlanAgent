# geoplanagent/matching/

MINIMA-based sliding-window georeferencing. Given a rendered planning
map + a candidate centre coordinate, find the affine transform that
maps page pixels to OS-tile pixels, RANSAC the inlier set, and emit
the projected boundary as a WGS84 GeoJSON polygon.

## Public API

```python
from geoplanagent.matching import (
    load_minima,                # one-time MINIMA-LoFTR matcher load
    sliding_window_position,    # the main entry — search centres × zooms × windows
    mask_to_geojson_affine,     # project a binary mask through a committed affine
    sigma_from_scale,           # σ default given a map's stated scale
    effective_sigma,            # fallback σ when the worker omits it
)
```

The above are re-exported from `_core.py` and `source_priorities.py`
respectively — `from geoplanagent.matching import …` is the stable surface.

## How it fits in the pipeline

`geoplanagent.agent.worker_tools.match_at` is the worker tool that drives this
package. Each `match_at` call covers ONE page (= one area_group):

1. **`load_minima()`** (once at process start) returns the LoFTR-based
   MINIMA matcher.
2. **`sliding_window_position(matcher, map_img, sam3_mask, centers,
   scale_ratio, ...)`** is the master entry:
   - Trusts the locate sub-agent's σ when one is supplied; otherwise
     falls back to `effective_sigma(scale_ratio)` (max of 5 km and the
     half-diagonal of the printed map's real-world extent).
   - Computes scale-aware zoom configs from `scale_ratio + dpi` —
     `best_zoom_for_scale ± 1` plus two `±15%` scale perturbations
     when the printed scale is known, or six canonical UK planning-map
     scales (1:1250 → 1:25000) when it isn't.
   - For each centre × zoom × rotation, resizes the map to match the
     tile pixel-scale (`resize_map_to_match_zoom`, AREA for downscale
     and CUBIC for upscale, skipped outside `[0.3, 3.0]`), fetches an
     OS OpenData tile canvas (`geoplanagent.io.os_tiles.fetch_os_opendata_grid`,
     sized by σ and clamped to a 3×3 to 17×17 odd-dimensioned grid),
     and slides the map across the canvas at a stride targeting
     `WINDOW_STRIDE_TARGET=100` windows per configuration (32-px floor
     — MINIMA's spatial-accuracy limit).
   - At every window position calls `run_minima(matcher, map_img,
     tile_img)` to compute LoFTR matches; `estimate_affine` recovers
     a 2×3 RANSAC similarity (4-DOF).
   - Keeps the best per-bucket window via a composite reranker — first
     `composite_window_score = vanilla_metric × Q/4` (Q = number of
     map quadrants with ≥1 inlier; from `geoplanagent.matching.scoring`), then
     `_verify_candidates_with_road_names` re-weights by
     `metric × (1 + road_match_ratio)²` when the reader extracted any
     road names. Sparse-rural candidates with no nearby OS roads get a
     neutral 1× multiplier.
3. **`mask_to_geojson_affine(mask, affine_H, tile_info)`** projects the
   SAM3 mask through the winning affine into a WGS84 GeoJSON
   `Feature` with `Polygon` (single contour) or `MultiPolygon`
   geometry. No morphological mask cleanup is applied — a 177-case
   ablation 2026-05-22 showed the old `keep_dominant_components →
   expand_thin_mask → fill_mask_holes` chain was a +0.001 IoU wash
   and was removed along with the `mask_ops` module.

## RANSAC affine (`estimate_affine`)

- **4-DOF similarity only** — rotation + uniform scale + translation
  via `cv2.estimateAffinePartial2D`.
- (2026-05-21) The 6-DOF full-affine fallback was removed after a
  25-case ablation showed it nets to -0.01 mean IoU and rescues only
  ~2 cases at the cost of code complexity.
- (2026-05-21) The optional Delaunay-consistency post-filter was
  removed after a 15-case ablation showed it provided zero mean
  benefit and was actively hurting the highest-inlier stress case.

## Sigma helpers (`source_priorities.py`)

The live locate sub-agent always supplies its own σ on every candidate,
and `sliding_window_position` trusts that value directly. These helpers
fire only on the fallback path when the worker passes `sigma_m=None` or
a non-positive value:

| Function | Returns |
|---|---|
| `sigma_from_scale(scale_ratio)` | Half-diagonal of the printed map's real-world extent (A4 landscape default). 1:1250 → 226 m, 1:2500 → 454 m, 1:25000 → 4540 m. Falls back to 2500 m when scale is unknown. |
| `effective_sigma(scale_ratio)` | `max(_FALLBACK_SIGMA_M, sigma_from_scale(scale_ratio))` with `_FALLBACK_SIGMA_M = 5000`. Conservative floor used only when the worker omits σ. |

The historical multi-candidate cascade (per-source σ tables,
LA-polygon candidate filter, source-priority capping) was retired when
the live locate sub-agent landed; its `la_check` tool already
validates LA containment on the single pick, so the cascade helpers
had no live callers and were removed.

## Output of `sliding_window_position`

A dict with:

| Key | Type | Meaning |
|---|---|---|
| `affine_H` | `np.ndarray (2, 3)` or None | Page-pixel → tile-pixel affine. None when no centre passed RANSAC. |
| `tile_info` | dict | `{"image", "zoom", "tx_min", "ty_min", "tile_size_px", …}` for the winning tile canvas. |
| `match_info` | dict | `{"n_inliers", "score", "aspect", "center_latlon", "zoom", "window", "scale_factor", …}` |
| `geojson` | dict or None | Final GeoJSON (computed inline if affine + tile_info are set). |

## Road-name verification (`road_verify.py`)

`_verify_candidates_with_road_names(ranked, road_names)` reads road
names from OS Open Zoomstack (offline GPKG) within 1500 m of each
candidate's recovered centre, fuzzy-matches against the reader's
`pdf_info.road_names` (with Street/Road/Lane → St/Rd/Ln
normalisation), and re-ranks by `metric × (1 + ratio)²`. The
quadratic boost is a single knob (exponent `p=2`); candidates with no
nearby OS roads (sparse rural cartography) get a neutral 1.0
multiplier — the metric fully decides for them. Warns once per
process when `os_opendata/OS_Open_Zoomstack.gpkg` is missing.

## Tunable constants

| Constant | Home | Value | Purpose |
|---|---|---|---|
| `WINDOW_STRIDE_TARGET` | `_core.py` | 100 | Sliding-window stride target (px) |
| `MAX_CANDIDATES` / `PER_BUCKET` | `_core.py` | 5 / 1 | Diversity-capped top-K within a single sliding-window pass |
| `_FALLBACK_SIGMA_M` | `source_priorities.py` | 5000 | Sigma floor when the worker omits σ (the live locate sub-agent always supplies one) |
| RANSAC reproj threshold | `_core.estimate_affine` | 10 px | Passed to `cv2.estimateAffinePartial2D` |
| RANSAC RNG seed | `_core.estimate_affine` | 42 | Inner RANSAC is run-to-run reproducible |
