# `tools/locate_eval.py`

**336 lines.** Evaluation utilities for the `tools.locate` candidate
generator: load GT polygons, compute distance from a candidate point to
the GT boundary, decide which "bucket" of accuracy a candidate falls in
based on the map scale. Not used in production runtime — only by the
training/dataset scripts (`scripts/auto_label_boundary_dataset.py`).

## Public API

| Function | Purpose |
|---|---|
| `load_gt_polygon(case_name, eval_dir)` | load + clean GT shape |
| `distance_to_boundary_m(lat, lon, gt_geom)` | metres from point to polygon edge |
| `scale_aware_tolerance_m(scale_ratio, ...)` | "good" tolerance per scale |
| `bucket(dist_m, tolerance_m)` | classify into "exact", "near", "far", etc. |
| `centroid_latlon(gt_geom)` | centroid as (lat, lon) |
| `predicted_map_footprint(...)` | polygon describing where a match places the map |

## Function walkthroughs

### `_osgb()` (line 32)

Lazy-loaded `pyproj.Transformer` for WGS84 → EPSG:27700 (OSGB). Cached
because `Transformer` construction is non-trivial (~100ms). Returns the
same instance on every call after the first.

### `load_gt_polygon(case_name, eval_dir)` (line 39)

Find the GT GeoJSON file in `evaluation_data/<case>/` and return it as a
shapely geometry in WGS84.

1. Look for any `*.geojson` in the case dir that's NOT a
   `location_*.geojson` (those are bbox helpers, not the boundary).
2. Parse, extract the geometry dict (handling `Feature` / `FeatureCollection`
   wrappers).
3. Build a shapely object via `shapely.geometry.shape`.
4. If invalid, `.buffer(0)` to repair.

Returns the shape, or `None` if the file's missing/malformed.

### `_extract_geometry_dict(data)` (line 60)

Helper for the wrapper-handling step in `load_gt_polygon`. Pulls the raw
geometry dict out of any GeoJSON shape. For `FeatureCollection`, takes
just the first feature (acceptable since most GTs have one feature).

### `distance_to_boundary_m(lat, lon, gt_geom_wgs84)` (line 79)

Distance from a (lat, lon) point to the nearest point on the GT polygon
boundary, in metres. Uses OSGB for accurate metric distance:

1. Project the GT polygon to EPSG:27700 (metric).
2. Project the point to OSGB.
3. shapely's `distance(point, polygon.boundary)`.

Used to evaluate locate-stage candidates: if a proposed center is 50m
from the GT boundary, it's a great anchor; 5km is useless.

### `scale_aware_tolerance_m(scale_ratio, ...)` (line 100)

What counts as "good enough" for a given scale:
- 1:1250 → 50m tolerance
- 1:2500 → 100m
- 1:10000 → 500m
- ...

The intuition: at small scales (large area on the page), you can be
further off and still match correctly because the map covers more
ground. At big scales (zoomed in), the tolerance is tight.

The numbers are tuned empirically — see the comment block in the function.

### `bucket(dist_m, tolerance_m)` (line 121)

Classify a candidate into:
- `"on_target"` — within tolerance
- `"close"` — within 3× tolerance
- `"nearby"` — within 10× tolerance
- `"far"` — beyond

Used for accuracy reporting in the training-set assembly scripts.

### `centroid_latlon(gt_geom_wgs84)` (line 138)

`gt.centroid` → `(lat, lon)`. Returns `None` if the geometry is empty.

### `predicted_map_footprint(affine_H, tile_info, map_shape)` (line 150)

Project the rectangle of the rendered map (its 4 corners) through the
affine to get a polygon describing **where the match places the map** in
WGS84. Useful for visualising candidate matches and computing things
like "what fraction of the GT is covered by where MINIMA placed the map".

Inverse of `mask_to_geojson_affine` in spirit — that function projects a
mask, this one projects the map's bbox.

## Why this design

**Why a separate file from `locate.py`?** `locate.py` is the production
candidate generator (only `locate_map` is called from the agent). The
evaluation helpers here are training-time only — keeping them separate
prevents the production bundle from pulling in shapely/pyproj
unnecessarily.

**Why scale-aware tolerance?** A 50m-off match at 1:10000 (where 50m =
2 page mm) is very accurate. A 50m-off match at 1:500 (50m = 10 cm on
page) is wildly off. A single fixed tolerance can't capture both — hence
the per-scale lookup.

**Why metres in OSGB instead of WGS84 degrees?** Distances on a sphere
aren't a single number — 0.001 degree of latitude is 111m, but 0.001
degree of longitude varies from 111m at the equator to ~50m in the UK.
Projecting to OSGB gives a flat metric coordinate where shapely's
distance is just euclidean and consistent across the UK.
