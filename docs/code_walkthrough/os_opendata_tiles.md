# `tools/os_opendata_tiles.py`

**812 lines.** Renders OS Open Zoomstack tiles from a local GeoPackage
on demand — instead of fetching from a tile server. The OS data is shipped
as a single ~12 GB `.gpkg` file containing all UK basemap layers
(buildings, roads, water, etc.); this module turns it into rasterised
tiles that match the slippy-map XYZ scheme.

## Public API

| Function | Purpose |
|---|---|
| `render_tile(zoom, tx, ty, gpkg_path)` | render one tile from layers |
| `fetch_tile_cached(zoom, tx, ty)` | render-or-cache one tile |
| `fetch_os_opendata_grid(lat, lon, zoom, nx, ny)` | render a grid centred on lat/lon |
| `fetch_historical_grid(lat, lon, zoom, nx, ny)` | same but for historical OS sheets |
| `fetch_os_opendata_roads_grid(...)` | roads-only render (for matchers) |

`fetch_os_opendata_grid` is the main one — called by
`positioning.sliding_window_position`.

## Tile coordinate plumbing

### `_lat_lon_to_tile(lat, lon, zoom)` (line 33)

Standard slippy-map projection: `lat/lon → (tx, ty)` integer tile
coordinates. Floor of the precise float values.

### `_tile_to_bounds_3857(zoom, tx, ty, tile_size)` (line 42)

Each XYZ tile covers a known bbox in EPSG:3857 (Web Mercator metres).
This function computes that bbox so we know what slice of the gpkg to
render.

### `_transform_3857_to_27700(x_min, y_min, x_max, y_max)` (line 94)

OS Open Zoomstack stores geometries in EPSG:27700 (OSGB36); slippy tiles
are EPSG:3857. Reproject the bbox so we can query the gpkg.

### `_transform_27700_to_pixels(geom, bounds_3857, tile_size)` (line 108)

For each polygon vertex pulled from the gpkg (in OSGB metres), compute
the corresponding (x, y) pixel inside the tile we're rendering. Combines
27700→3857 + 3857→tile-pixel transforms.

## Layer reading

### `_read_layer(layer_name, bounds_27700, gpkg_path=None)` (line 62)

Run a SQL query against the GeoPackage:

```sql
SELECT geom FROM <layer_name>
WHERE ST_Intersects(geom, BuildMbr(?, ?, ?, ?))
```

ST_Intersects + the spatial R-tree index makes this fast even on a 12 GB
file. Returns a list of (geometry-blob, attrs) tuples.

The `gpkg_path` defaults to `os_opendata/OS_Open_Zoomstack.gpkg` if not
provided.

## Drawing primitives

### `_draw_polygon(canvas, pixel_geom, fill_color, outline=None, outline_width=1)` (line 297)

Fill a polygon on the canvas with `cv2.fillPoly` and optionally outline
with `cv2.polylines`.

### `_draw_line(canvas, pixel_geom, color, width)` (line 329)

Draw a polyline (for roads, rivers, etc.) with `cv2.polylines`.

## Single-tile rendering

### `render_tile(zoom, tx, ty, gpkg_path=None, tile_size=256)` (line 163)

The full pipeline for one tile:

1. **Compute bounds** of the tile in 3857 and 27700.
2. **Initialise canvas** (256×256, fill with `STYLE["background"]`).
3. **For each layer** (in z-order: water → buildings → roads → labels):
   - Read polygons/lines intersecting the tile's bounds.
   - Project each to tile pixels.
   - Draw with the appropriate `_draw_*` function.
4. **Return** the canvas as a `(256, 256, 3)` RGB ndarray.

The `STYLE` dict (defined elsewhere in the file) maps each layer name to
its colour + draw type (fill vs. outline, line width, etc.). Style
choices loosely follow the OS Open Zoomstack default style.

## Bulk grid rendering (the fast path)

### `_render_canvas_bulk(zoom, tx_min, ty_min, n_tiles_x, n_tiles_y, gpkg_path=None)` (line 379)

The optimisation: rendering 35×35 tiles via `render_tile` × 1225 calls
would be slow because each call re-queries the gpkg. Instead, this
function:

1. **Compute the full grid's bounds** (one big bbox).
2. **Read each layer once** for the whole grid.
3. **Project each geometry once** to canvas pixels (huge canvas =
   `nx*256 × ny*256`).
4. **Draw** all layers into the giant canvas.

End result: ~5-7 seconds for a 35×35 grid (versus 1+ minute for per-tile
calls).

### `fetch_os_opendata_grid(lat, lon, zoom, n_tiles_x, n_tiles_y, gpkg_path=None)` (line 760)

Top-level grid fetcher:
1. Compute centre tile coords from `(lat, lon, zoom)`.
2. `tx_min = cx - half_x`, `ty_min = cy - half_y`.
3. Check the on-disk grid cache (`_grid_cache_path` + `_grid_cache_*`).
4. If hit: load PNG, return.
5. If miss: call `_render_canvas_bulk`, cache the result.
6. Return `{"image": canvas_rgb, "zoom": ..., "tx_min": ..., ...}` so
   downstream code knows how to project pixel coords back to lat/lon.

## Historical sheets

### `fetch_historical_grid(lat, lon, zoom, n_tiles_x, n_tiles_y, layer="newpopular")` (line 539)

Same interface as the modern grid but renders from a different style
(historical OS New Popular Edition). Used as a fallback when modern
Zoomstack tiles don't match the planning map's cartography (the
"grayscale" path in `sliding_window_position`).

## Roads-only rendering

### `_render_roads_only_canvas`, `fetch_os_opendata_roads_grid`, `fetch_os_opendata_roads_for_tile_info` (lines 623-758)

Render only the roads layer. Used by some matcher-precheck paths that
just need to know "is there a road network at this location" without
the noise of buildings + landuse polygons.

## Caching

Two cache layers:

1. **Per-tile** (`_tile_cache_path`, `fetch_tile_cached`) — single 256×256
   tiles, used for tools that fetch one tile at a time.
2. **Per-grid** (`_grid_cache_path`) — pre-rendered grids of common sizes
   centred on common locations. The big speed win for benchmark re-runs.

Cache files are PNGs in `cache/os_tiles/`.

## Why this design

**Why render from gpkg instead of using a tile server?** Two reasons:

- **Reliability** — public OS tile servers rate-limit and occasionally go
  down. Local rendering is always available.
- **Speed** — for the same area at z18, local rendering with caching beats
  remote fetching after the first run.

**Why two cache layers?** Per-tile is fine for one-off lookups (e.g.
visualising a single boundary). Per-grid is way faster for MINIMA
sliding-window matching which always needs grids of consistent size.

**Why bulk render in `_render_canvas_bulk`?** Profiling showed that 90%
of tile-render time was the gpkg query + projection setup. Doing it once
for a whole grid amortises that cost across all 256×256 sub-canvases.
