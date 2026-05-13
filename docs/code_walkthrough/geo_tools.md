# `tools/geo_tools.py`

**676 lines.** Pure geography helpers: parsing OS grid references in their
many UK formats, transforming OSGB ⇄ WGS84, looking up administrative
boundaries via OSM, and a thin wrapper around Nominatim. No image/ML —
pure coordinate arithmetic and remote queries.

## Public API

| Function | Purpose |
|---|---|
| `parse_easting_northing(text)` | exact OSGB easting/northing → (lat, lon) |
| `os_grid_ref_to_latlon_coarse(grid_ref)` | low-resolution grid ref → (lat, lon) |
| `os_grid_ref_to_latlon(grid_ref)` | precise grid ref (1km+) → (lat, lon) |
| `pixels_to_geo_linear(...)` | pixel → lat/lon via centre + scale |
| `lookup_district_boundary(query)` | OSM admin polygon for a place name |
| `try_district_boundary(analysis)` | wraps lookup with case-specific filters |
| `geocode_address(address)` | Nominatim address lookup with retries |

## Module-level setup

### `_OS_GRID_LETTERS` (lines 32-42)

The OS National Grid uses 2-letter prefixes (e.g. `TQ`, `SU`) where each
prefix names a 100km×100km square. The lookup table maps each prefix to its
(easting, northing) base in metres.

The math (lines 33-42) implements the standard formula. Two key oddities:

- **Letters skip "I"** (lines 34, 37). The OS grid was designed to avoid
  visual confusion between I and 1.
- **The arithmetic** uses the letter's position in a 5×5 super-grid and a
  5×5 sub-grid: easting = `((c1-2) % 5) * 5 + (c2 % 5) * 100km` etc.

You don't need to follow it; just trust that `_OS_GRID_LETTERS["TQ"]` is
the metric coordinates of TQ's south-west corner (510,000m E, 100,000m N
for TQ).

### `_OSGB_TO_WGS84` (line 44)

A pyproj transformer. EPSG:27700 is the UK's OSGB36 projection (in metres);
EPSG:4326 is WGS84 lat/lon. `always_xy=True` means it takes (x, y) order
i.e. (easting, northing) → (lon, lat).

### `_EN_RE` (line 47)

Regex for "528942 E 184544 N" style coordinates. Allows 4-7 digits per
axis, case-insensitive E/N markers, optional whitespace.

## Function walkthroughs

### `parse_easting_northing(text) -> (lat, lon) | None`

The highest-precision anchor extractor. Many UK planning PDFs print the
site centre as explicit OSGB metric coordinates somewhere in the body
text. If we find them, we know the site location to ~1m accuracy.

```
"528942 E 184544 N"  →  (51.545, -0.142)
```

Steps:
1. Run `_EN_RE` against the input text.
2. If it matches, the two capture groups are easting/northing in metres.
3. Hand them to `_OSGB_TO_WGS84.transform(east, north)` → (lon, lat).
4. Note the swap: pyproj returns (x, y) = (lon, lat), so we flip when
   returning to the standard (lat, lon) order.

This is a key input to `tools/positioning.analytical_affine_from_anchor` —
when both this AND a numeric scale are available, the affine is fully
determined and we can skip MINIMA entirely.

### `os_grid_ref_to_latlon_coarse(grid_ref) -> (lat, lon) | None`

Handles low-resolution grid refs like `TR 34` (10km tile, centre) or
`TR 34 SE` (5km quadrant, centre). These are common on planning maps that
just want to indicate the rough area.

The "centre of the tile" return is intentional. If the user wants a corner,
they can offset by ±5km themselves.

### `os_grid_ref_to_latlon(grid_ref) -> (lat, lon) | None`

The precise version. Accepts:
- `TG 210 080` (3+3 digits = 100m precision)
- `TG 21 08` (2+2 = 1km)
- `TG2108` (compact form)
- `TR 35-3656` (range — collapses to lower bound)
- Trailing compass directions (`SE`, `NW`) are stripped before parsing.

Internally:
1. Strip whitespace, uppercase, strip trailing compass.
2. Try `LL ddd ddd` format with regex.
3. If that fails, try `LLddd...` (compact).
4. Look up the prefix in `_OS_GRID_LETTERS` to get the 100km base.
5. Append the digit pairs (left-padded with zeros if asymmetric) as
   easting/northing offsets within the 100km tile.
6. Transform to WGS84.

Returns None if anything malformed.

The 4-digit minimum (line 148) is a deliberate floor: OS grid refs with
just 2-3 digits are too coarse to be useful as match anchors.

### `pixels_to_geo_linear(pixel_xy, image_shape, center_lat_lon, scale_ratio, dpi)`

Linear pixel→lat/lon transform when you trust:
- the image is north-up
- `center_lat_lon` is at the image centre
- `scale_ratio` (e.g. 2500 for 1:2500) is correct
- `dpi` is the rendering DPI

Math:
- 1 page pixel = `25.4/dpi/1000` metres on paper → multiply by scale_ratio
  for ground metres.
- Convert metres east/north to lon/lat deltas using
  `METERS_PER_DEGREE_LAT = 111111` and a cosine adjustment for longitude
  (the earth gets narrower near the poles).

This is the simplest possible projection — no rotation correction, no skew.
Used as a sanity check / quick estimate; the real production path uses
MINIMA-derived affines (`tools/positioning.py`) or analytical affines
(`tools/positioning.analytical_affine_from_anchor`).

### `lookup_district_boundary(query) -> dict`

Hits OSM via osmnx (`ox.geocode_to_gdf`) for a place name like
"Wokingham Borough Council" or "Conservation Area: Toddington". Returns
the bounding box and a representative point.

Used by the legacy `position_boundary` path when the planning area equals
an admin region. Most cases don't fall through here — we extract a
boundary mask and project it through an affine instead.

### `try_district_boundary(analysis) -> dict | None`

Wrapper that takes a `pdf_info`-style analysis dict and tries district
lookup if `is_district_wide=True` was set by the reader. Falls through to
None if the lookup fails or the case isn't district-wide.

The reader (`_reader_agent` in `agent.py`) over-flags this — the worker
agent has a tool (`lookup_district`) that can pull the polygon when needed
but doesn't apply it automatically.

### `geocode_address(address) -> dict`

Calls Nominatim (geopy wrapper) with an address string. 3 retries with
exponential backoff because Nominatim's free service rate-limits
aggressively (1 req/sec).

Returns `{"success": bool, "latitude": ..., "longitude": ..., ...}` or
`{"success": False, "error": "..."}`. The shape is consistent with other
geocoders so the agent can switch backends without restructuring.

The User-Agent string `"planning_doc_extractor"` is required by Nominatim's
ToS — they reject requests with no UA.

## Why this design

**Separation of concerns.** All OSGB ⇄ WGS84 logic lives here, separate
from the geocoder cascade in `geocoding.py`. The geocoders return lat/lon;
this file converts other coordinate systems to lat/lon.

**Why a coarse vs precise grid-ref function?** Some planning PDFs only
quote the 10km tile (`TR 34`) — coarse but better than nothing. Mixing
them in one function would force an arbitrary precision-acceptance flag.
Two functions with explicit names = caller chooses.

**Why module-level lookup table for `_OS_GRID_LETTERS`?** The grid is
fixed, only ~325 entries, computing it on import is microseconds. Caching
it as a class or with `lru_cache` would just add complexity.
