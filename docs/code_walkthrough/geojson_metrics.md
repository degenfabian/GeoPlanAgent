# `tools/geojson_metrics.py`

**261 lines.** GeoJSON helpers + IoU computation. Used by the benchmark
runner to compute the headline metric (predicted polygon vs ground-truth
polygon IoU) and by the agent for in-loop sanity checks.

## Public API

| Function | Purpose |
|---|---|
| `load_geojson(path)` | parse a GeoJSON file → dict |
| `validate_geojson_format(data)` | shape check (returns `(ok, reason)`) |
| `geojson_to_shape(data)` | dict → shapely Polygon/MultiPolygon |
| `calculate_iou(pred, gt)` | IoU between two GeoJSON dicts |
| `calculate_positioning_error_m(pred, gt)` | metres between centroids |
| `calculate_spatial_metrics(pred, gt)` | full metrics dict |
| `gt_centroid(path)` | centroid of GT polygon |

## Function walkthroughs

### `load_geojson(path)` (line 11)

`json.load` + minimal sanity check (returns `None` if the file's missing
or doesn't parse). Used everywhere a GeoJSON path is taken.

### `validate_geojson_format(data)` (line 29)

Confirms the dict has a recognised shape:
- `Feature` with a `geometry` of type `Polygon` or `MultiPolygon`
- `FeatureCollection` (will be unioned later)
- Bare `Polygon` / `MultiPolygon`

Returns `(True, "")` if valid, `(False, "reason")` otherwise. Used to
fail fast with a clear error if the agent or pipeline produces malformed
output.

### `geojson_to_shape(data)` (line 53)

Convert a GeoJSON dict to a shapely geometry:
1. Pull the geometry out (handles `Feature` and `FeatureCollection`
   wrappers).
2. For `FeatureCollection`, union all features into one shape.
3. Build the shapely object via `shapely.geometry.shape(...)`.
4. If invalid (e.g. self-intersecting polygons from imperfect drawing),
   call `.buffer(0)` to repair — a standard shapely trick.

Returns `None` if the dict is unparseable.

### `calculate_iou(pred, gt)` (line 89)

The headline metric:
1. Convert both to shapely shapes.
2. Compute `intersection.area / union.area`.
3. Returns 0.0 if either polygon is empty or shapes can't be built.

This is what every benchmark report measures success against.

### `calculate_positioning_error_m(pred, gt)` (line 124)

Distance between predicted and GT centroids in metres. Useful for
sub-IoU diagnostics — even when IoU is 0, a 100m error is very different
from a 100km error.

Uses haversine via shapely's centroid + a manual lat/lon → metres
conversion (`111111 * cos(lat)` for longitude, `111111` for latitude).

### `calculate_spatial_metrics(pred, gt)` (line 142)

Returns the full metrics bundle:
- `iou`
- `positioning_error_m`
- `predicted_area_m2`, `gt_area_m2`
- `area_ratio` (pred/gt — catches scale errors)
- `centroid_distance_m` (alias for positioning_error_m)

This is what `metrics.json` serialises for each benchmark case.

### `gt_centroid(path)` (line 233)

Convenience: load a GeoJSON file and return its centroid as `(lat, lon)`.
Used by the agent's reward axes that need the GT location for sanity
checks (during training-time eval; never during deployment, since GT
isn't available).

## Why this design

**Why custom format check?** Different GeoJSON producers emit slightly
different shapes (`Feature` vs raw `Polygon`, single vs multipolygon
wrapped vs unwrapped). The check normalises so callers don't have to
case-split.

**Why metres for positioning error?** Lat/lon degrees are not
intuitive — 0.001 degrees is 111m at the equator but ~70m at UK
latitudes. Converting to metres up front means downstream code (logging,
thresholds, agent prompts) can use a single intuitive unit.

**Why `.buffer(0)` for invalid polygons?** Hand-drawn boundaries often
have self-intersections (the artist crossed their own line). shapely
flags these as invalid and refuses to compute IoU. `buffer(0)` is the
standard repair — it computes the polygon's exterior cleanly. Alternative
would be `make_valid` (newer shapely) but `buffer(0)` works on any
version.
