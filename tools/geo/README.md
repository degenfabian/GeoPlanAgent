# tools/geo/

Offline geographic primitives. Everything here works without a network
connection or an API key. The geocoder data files live under
`os_opendata/`; setup commands are in each module's docstring.

## Public surface

| Module | Function | Purpose |
|---|---|---|
| `code_point` | `lookup_postcode(pc)` | Full UK postcode → BNG → WGS84. Sub-100 m precision. From OS Code-Point Open (1.6 M postcode unit centroids). |
| `os_names` | `lookup(name)` / `search(query, max_results, context?)` | Place / settlement / road / landmark name search. From OS Open Names (2.5 M GB entries). Sub-metre BNG. |
| `grid_ref` | `os_grid_ref_to_latlon(gr)` | Parse an OS BNG grid reference in any common format → `(lat, lon)`. |
| `grid_ref` | `lookup_district_boundary(district_name)` | OS BoundaryLine offline lookup of a UK admin district → GeoJSON feature. Supports `'|'`-separated name alternates and common-suffix normalisation. |
| `coords` | (helpers) | Web-Mercator / tile-pixel math, BNG ↔ WGS84, `haversine_km`. |

These are the building blocks the locate sub-agent's six geocoders are
built from (see `tools.agent.locate_agent`). The same modules are
imported directly by the worker tool `lookup_district` and by
verification checks.

## Data assets (offline)

Place under `os_opendata/`. All are OGL v3, no API key, no
registration.

| Asset | Source | Approximate size | Setup |
|---|---|---|---|
| Code-Point Open CSVs | OS Code-Point Open | ~250 MB | `curl -L .../CodePointOpen → unzip os_opendata/code_point_open/csv` |
| OS Open Names CSVs | OS Open Names | ~750 MB | `curl -L .../OpenNames → unzip os_opendata/open_names/csv` |
| OS BoundaryLine | OS BoundaryLine | ~100 MB | Download from OS OpenData portal → `os_opendata/boundary_line/` |
| OS OpenMap Local | OS OpenMap Local | ~2.5 GB | Used by the `road` / `intersect` locate tools via `oml_road_index.json` |
| OS Open Zoomstack | OS OpenData Zoomstack | ~1.5 GB | `OS_Open_Zoomstack.gpkg` at `os_opendata/`. Used by `tools.io.os_tiles` for tile rendering. |

Full URLs in each module's docstring.

## Code-Point Open (`code_point.py`)

```python
from tools.geo.code_point import lookup_postcode

hit = lookup_postcode("AL1 3JE")
# → {"lat": 51.7534, "lon": -0.3361, "easting": 515387, "northing": 206398,
#    "sigma_m": 50, "source": "code_point_open", "admin_district": "St Albans"}
```

Use as the highest-priority anchor when the reader extracts a full
postcode (outward + inward, e.g. `"AL1 3JE"`). Drops σ to ~50 m vs
the 800-2500 m floor that other sources pay.

## OS Open Names (`os_names.py`)

```python
from tools.geo.os_names import lookup, search

hit = lookup("East Langdon")
# → {"name_full": "East Langdon", "type": "village", "lat": 51.171,
#    "lon": 1.345, "sigma_m": 800, "source": "os_open_names:village"}

hits = search("Manor", max_results=5, context="St Albans")  # LA-disambiguated
```

Covers villages, hamlets, suburbs, named roads, churches, schools,
hospitals, recreation grounds, named buildings, etc. Same data as the
paid OS Names API — just offline.

## OS BNG grid references (`grid_ref.py`)

```python
from tools.geo.grid_ref import os_grid_ref_to_latlon, lookup_district_boundary

pt = os_grid_ref_to_latlon("TL 150 067")
# → (51.7534, -0.3361)

pt = os_grid_ref_to_latlon("TR3559")        # also valid
pt = os_grid_ref_to_latlon("485700 148600") # raw BNG easting-northing

dist = lookup_district_boundary(
    "City of Westminster, UK | Westminster, UK"
)
# → {"success": True, "geojson": {"type": "Feature", "geometry": {…}, ...},
#    "matched_variant": "City of Westminster, UK"}
```

`lookup_district_boundary` uses `tools.verification_checks._resolve_la`
under the hood. Name normalisation handles `"London Borough of X"
→ "X"` and strips trailing `"District" / "Borough" / "Council"`. The
`|`-alternates syntax lets the worker hedge between ambiguous names.

## Coordinate utilities (`coords.py`)

All Web-Mercator / tile-pixel / BNG ↔ WGS84 math lives here. Single
source of truth; the same formula was previously duplicated across
six modules in the repo. Notable:

- `haversine_km(lat1, lon1, lat2, lon2)` — great-circle distance in
  kilometres (multiply by 1000 for metres).
- BNG transforms via `pyproj.Transformer.from_crs("EPSG:4326",
  "EPSG:27700")`.
- Web-Mercator tile-pixel projection: `156543.03 * cos(lat) / 2**zoom`
  metres per pixel.

`tools/matching` re-exports the names it needs (`compute_map_mpp`,
`best_zoom_for_scale`, etc.) for callers that import from
`tools.matching` directly.
