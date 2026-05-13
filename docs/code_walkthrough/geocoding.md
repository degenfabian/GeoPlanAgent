# `tools/geocoding.py`

**1063 lines.** A cascade of geocoders that turn a name like "Wokingham"
or "Maumbury Rings, Dorchester" into (lat, lon). Tries multiple sources
because each has different coverage and biases:

1. **gpkg** — local OS Open Names GeoPackage (fastest, UK-only,
   excellent for small UK villages and hamlets that web geocoders miss).
2. **Wikidata** — SPARQL queries (good for landmarks and unusual names).
3. **Photon** — public OSM-based geocoder (broadest coverage, occasional
   wrong-country hits).
4. **Nominatim** — official OSM geocoder (slow, rate-limited).

Each source has the same return shape so the agent can fall through
without restructuring.

## Public API

| Function | Source | Purpose |
|---|---|---|
| `gpkg_place_search(name, parent_lat, parent_lon, limit)` | local gpkg | UK villages, towns, geo features |
| `wikidata_place_search(name, parent_lat, parent_lon, limit)` | Wikidata | landmarks |
| `nominatim_structured(street, city, county, country)` | Nominatim | structured address |
| `query_photon(address, limit)` | Photon | freeform address |
| `photon_centers(visual_extract)` | Photon | extract centers from VLM output |
| `place_name_centers(visual_extract)` | gpkg | extract centers from VLM output |
| `cross_validate_centers(centers, max_outlier_km)` | local | drop centers far from cluster |
| `road_name_precheck(...)` | OSM | verify candidate has expected roads |
| `postcode_district_filter(...)` | local | drop centers in wrong postcode district |
| `parse_scale_ratio(analysis)` | local | extract `1:N` from analysis dict |
| `collect_postcodes(analysis)` | local | gather postcodes from various fields |

## Module setup

### `_osgb_to_wgs84` (line 56)

Lazy `pyproj.Transformer`. Same pattern as `geo_tools.py` and
`locate_eval.py` — built on first call, cached.

### `_parse_gpkg_point(blob)` (line 64)

GPKG stores point geometries as binary blobs. This decodes the blob's
header + raw little-endian doubles to `(lat, lon)`. Used internally by
`gpkg_place_search`.

## gpkg (OS Open Names)

### `gpkg_place_search(name, parent_lat=None, parent_lon=None, limit=5, ...)` (line 87)

Search the local OS Open Names GeoPackage for a place name.

1. **Open** `os_opendata/OS_Open_Zoomstack.gpkg` (singleton SQLite
   connection).
2. **Run a fuzzy SQL query**:
   ```sql
   SELECT name, type, geom FROM places WHERE name LIKE ? COLLATE NOCASE
   ```
3. **Decode each result's geometry** to (lat, lon).
4. **If `parent_lat/lon` is given**, sort results by distance from the
   parent — disambiguates "Waterfoot" (3 in the UK) by picking the one
   nearest the case's anchor.
5. **Return** up to `limit` results, each as a dict with `name`, `type`,
   `lat`, `lon`, `distance_from_parent_km`.

This is the cleanest geocoder because OS Open Names is UK-only and
high-precision. Used as the primary source whenever the place is in the
UK gazetteer.

## Wikidata

### `_load_wikidata_cache() / _save_wikidata_cache()` (lines 231, 245)

JSON file at `cache/wikidata_cache.json` mapping query keys to results.
Wikidata SPARQL is slow and rate-limited; caching repeat queries (which
are common when re-running benchmarks) saves minutes per case.

### `wikidata_place_search(name, parent_lat, parent_lon, limit=3, ...)` (line 263)

SPARQL query against the Wikidata Query Service:

```sparql
SELECT ?item ?coords WHERE {
  ?item rdfs:label ?label .
  FILTER(LCASE(STR(?label)) = LCASE("...")).
  ?item wdt:P625 ?coords .
  ...
}
```

Filters to UK results via P17 (country). Parent-distance disambiguation
works the same as in gpkg.

Used for cases where the place is too small for OS Open Names (e.g.
specific listed buildings, conservation areas, monuments).

## Nominatim

### `nominatim_structured(street, city, county, country, ...)` (line 411)

The **structured** Nominatim endpoint (better than `?q=` for parsed
addresses). Pass `street="123 Main Street"`, `city="Bath"`,
`country="GB"` and you get back a single best result.

Has its own cache (`cache/nominatim_cache.json`) keyed on the structured
fields. Built-in retry + exponential backoff for the inevitable
rate-limit hits (Nominatim's free tier is 1 req/sec, no concurrent).

## Photon

### `query_photon(address, limit=3)` (line 516)

Hits `photon.komoot.io/api?q=...`. Returns up to `limit` results as
list of dicts.

Has a hardcoded UK bbox filter (`_is_valid_uk_coord`) — Photon's free
endpoint sometimes returns hits in the US or Eastern Europe for
ambiguous queries; we drop them.

### `photon_centers(visual_extract)` (line 549)

Extract a list of `Center` candidates (named tuple of name, lat, lon, sigma)
from a VLM's parsed output. Calls `query_photon` for each named place
the VLM extracted, filters to UK-valid, returns the list.

## Candidate filtering

### `_distance_m(lat1, lon1, lat2, lon2)` (line 609)

Haversine distance in metres. Pure numpy, no external deps. Used
everywhere candidate filtering is done.

### `cross_validate_centers(centers, max_outlier_km=10)` (line 616)

Drop outlier centers: compute the geometric median of all centers,
remove anyone more than `max_outlier_km` away. Catches the case where
one geocoder picked a wrong UK region (e.g. Photon got "Linden Grove,
Wisconsin" instead of "Linden Grove, SE15").

### `road_name_precheck(center_lat, center_lon, road_names, ...)` (line 712)

For a candidate center, query OSM for road names within a radius.
Return True iff a fuzzy match exists for at least one of the expected
`road_names`. Used to drop candidates that look correct geographically
but are missing the expected roads (= probably the wrong area within a
city).

### `postcode_district_filter(centers, postcodes, ...)` (line 756)

If we have postcodes from the PDF, filter centers to those in the
correct postcode district (the alphanumeric prefix, e.g. `SW1` or
`PE5`). Catches "wrong city in same county" failures.

### `council_boundary_filter(centers, council_name, ...)` (line 786)

For cases where the PDF mentions a specific council (Local Planning
Authority), drop centers outside that council's polygon. Helper
`_fetch_council_boundary` does the OSM lookup; `_point_near_polygon`
does the inclusion test.

## Helpers

### `parse_scale_ratio(analysis, ve=None)` (line 907)

Extract the scale denominator from a PDF analysis dict. Looks at
`analysis.scale`, falls back to `ve.scale` (visual extraction). Handles
formats like `1:2500`, `1/2500`, `1 : 2,500` (commas in big numbers).
Returns `None` if no numeric scale.

### `collect_postcodes(analysis, ve=None)` (line 922)

Pull postcodes from many possible PDFInfo fields (`postcodes`,
`site_address`, etc.) and return a deduplicated list.

## Why this design

**Why so many geocoders?** No single source covers everything:

- gpkg has every UK village ≥100 inhabitants, but no street-level data.
- Photon has streets but variable quality.
- Nominatim is authoritative for structured addresses but slow.
- Wikidata has landmarks (monuments, conservation areas) that aren't in
  any of the others.

The cascade lets the agent try the cheap-and-fast sources first, fall
through to slower ones only when they fail.

**Why disk caches?** Photon and Nominatim have rate limits. Wikidata
SPARQL is slow. Caching makes benchmark re-runs fast and protects
against transient API failures.

**Why disambiguate by parent_lat/lon?** UK has many duplicate place names
("Newport", "Whitchurch", etc.). Without a parent anchor, the geocoder
picks one essentially at random. With one, the closest match wins.
Originated this design after recovery experiments showed wrong-city
matches were the second-most-common failure mode.

**Why a `Center` named tuple instead of a class?** It's used as a
4-tuple `(name, lat, lon, sigma)` literally everywhere; making it a
class would mean 200+ refactors of `c.name` instead of `c[0]`. The
tuple shape is a stable API across the codebase.
