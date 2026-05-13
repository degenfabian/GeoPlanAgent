# `tools/locate.py`

**2486 lines.** The legacy GCP (Ground Control Point) candidate generator.
Originally pre-dated the LLM-driven agent flow — at the time it was the
primary path for figuring out where on the UK a planning map sits. Now
it's a single tool the worker agent can call (`locate_map`) when other
sources of centers fail, and its outputs are merged with everything else
in `propose_centers`.

Only `locate_map` is called externally. The other 50+ functions are
internal helpers.

## Public API

| Function | Purpose |
|---|---|
| `locate_map(pdf_path, page_idx, pdf_info, ...)` | top-level: scan PDF + image → candidates |
| `to_positioning_centers(result, min_confidence)` | convert result → `Center` tuples |

Plus the data classes:
- `VLMMapLabels` — schema for VLM map-label extraction.
- `LocateCandidate` — one ranked candidate (lat, lon, confidence, evidence).
- `DirectAffine` — affine fitted directly from grid ticks.
- `LocateResult` — wraps a list of candidates + the optional direct affine.
- `OCRWord` — OCR text + bbox.

## High-level flow

`locate_map` runs five strategies in parallel (or near-parallel) and
returns the union of all candidates ranked by confidence:

### 1. Postcode lookup

Pull postcodes from `pdf_info.postcodes` + OCR-extracted ones, geocode
each (Photon → Nominatim), produce one candidate per postcode.

### 2. Grid reference solving

Tesseract OCR the rendered map page. Find OS grid-tick labels (e.g.
`520`, `184500`, `52'30"`) and their pixel positions. If ≥3 are found:

- `solve_affine_from_grid_ticks(ticks)` (line 319) — fit a 2D affine
  via `cv2.estimateAffine2D` mapping page pixels → OSGB metres.
- Reject near-collinear configurations (singular).
- Compute mean reprojection residual on inliers.

If the affine is good, use `direct_affine_centroid` (line 379) to compute
the page-centre lat/lon and emit it as a high-confidence candidate.

### 3. Road triangulation

Get road names from `pdf_info.road_names` + VLM extraction. For each
combination of ≥2 roads:

- `find_road_intersections(roads, anchor_lat, anchor_lon)` (line 488) —
  query OSMnx for the road network around an anchor; find nodes where
  ≥2 of the input roads meet.
- Each intersection becomes a candidate.

This catches cases where postcode + grid-ref both fail but the map has
clear named-road labels (e.g. older OS sheets).

### 4. VLM map reading

If `_VLM_SYS_PROMPT` is enabled, call a VLM (`_vlm_read`, line 406) on
the rendered map crop with a prompt that asks it to read every label.
Returns a `VLMMapLabels` schema with structured fields (place names,
roads, grid refs, scale).

Then geocode the VLM-extracted place names + roads as additional
candidates.

### 5. District lookup

If the planning area is district-wide (`pdf_info.is_district_wide`),
look up the admin polygon via `_district_candidate` (line 911) — Nominatim
+ GeoJSON for the district. The boundary itself becomes the prediction.

## Internal data classes

### `OCRWord` (line 158, dataclass)

```python
@dataclass
class OCRWord:
    text: str
    x: int           # bbox top-left
    y: int
    w: int           # bbox dimensions
    h: int
    conf: float      # 0-100 from pytesseract
```

Used everywhere downstream of `_run_tesseract`. Confidence scores from
pytesseract aren't perfectly calibrated but useful for filtering noise.

### `LocateCandidate` (line 121, Pydantic BaseModel)

```python
{
  "name": "98 Pipers Lane CH60 9HN",
  "lat": 53.3334,
  "lon": -3.1242,
  "confidence": 0.85,
  "evidence_type": "postcode" | "grid_ref" | "road_intersection" | "vlm" | "district",
  "evidence": {...}  # source-specific details
}
```

Confidence is on a 0-1 scale, calibrated per evidence type:
- Grid-ref affine: 0.95+ (very precise)
- Postcode: 0.8-0.9
- Road intersection (≥3 roads meeting): 0.75
- VLM extraction: 0.4-0.6 (LLM might hallucinate)
- District: 0.3 (very coarse)

### `DirectAffine` (line 132, Pydantic BaseModel)

The output of `solve_affine_from_grid_ticks`:

```python
{
  "matrix_2x3": [[a, b, c], [d, e, f]],  # page-px → OSGB-metres
  "tick_count": 5,
  "mean_residual_m": 12.3,  # avg distance between predicted & actual tick positions
}
```

If `mean_residual_m` is small (< 50m), the affine is trustworthy and the
caller can use it for the entire mask projection — bypassing MINIMA and
SAM3-derived projection. Otherwise it's just a coarse anchor.

### `LocateResult` (line 142, Pydantic BaseModel)

Wraps the candidate list + optional direct affine + scale (parsed from
either the OCR or the VLM output).

## Selected internal helpers

### `_safe_ocr_dpi(pdf_path, page_idx)` (line 167)

Pick a DPI that keeps the rendered page under ~50 megapixels. Some
historic OS sheets are A1 or larger; 700 DPI on those would give 150 MP
images that take 15+ minutes for tesseract.

### `_run_tesseract(img_bgr, psm=11)` (line 190)

PSM 11 = sparse text mode. Robust for map annotations (scattered labels,
margin ticks) where layout analysis would fail. Returns a list of
`OCRWord`s.

### `extract_grid_refs_from_ocr(words, pdf_text)` (line 254)

Find OS grid-tick labels in OCR words. The trick is that grid ticks are
short numeric strings (3-5 digits) at known positions on the map margin.
Filter by:
- Pure-digit text (or digits + a small set of allowed chars).
- Position near the page edge.
- Plausible OSGB easting/northing range when interpreted as metres.

### `extract_scale_from_ocr(words)` (line 284)

Find a `1:N` token in the OCR. Returns `(scale, units_str)` if found.

### `_collinearity_score(points)` (line 368)

`1 - (smallest_singular_value / largest_singular_value)` of the centred
point matrix. Closer to 1 = more collinear. Used to reject affine fits
where ticks all sit on one line.

### `_norm_road(name)` (line 455)

Normalise a road name for fuzzy matching — lowercase, strip "the", strip
" Road"/" Street"/etc. suffix → just the distinctive part. So "Marsham
Street" matches "Marsham St" matches "MARSHAM ROAD".

### `_road_matches(osm_name, input_roads_norm)` (line 469)

Take an OSM road name and a dict of normalised input roads; return the
first match (or None). Allows partial matching.

### `find_road_intersections(roads, anchor_lat, anchor_lon, radius_km=5.0)` (line 488)

The triangulation core:
1. Query OSMnx for the road network within `radius_km` of the anchor.
2. For each edge in the graph, check if its name fuzzy-matches any
   `roads[i]`.
3. Collect, per node, which input roads touch it.
4. Return nodes where ≥2 distinct input roads meet, sorted by
   (count desc, distance from anchor asc).

Output: `[(lat, lon, [matched_road_names]), ...]`.

### `_triangulation_candidates(roads, ...)` (line 565)

Wraps `find_road_intersections` and converts intersections to
`LocateCandidate`s with calibrated confidences (≥3 roads meeting:
high; 2 roads: medium).

### `_geocode_vlm_labels(labels, ...)` (line 674)

Geocode the place names and roads extracted by the VLM. Calls
`gpkg_place_search` first, falls through to Photon.

### `_district_candidate(pdf_info)` (line 911)

If `pdf_info.is_district_wide` and `district_name` is set, look up the
district boundary via Nominatim → return its centroid as a (low-confidence)
candidate.

### `_clean_district_query(name)` (line 783) and `_district_info(...)` (line 819)

Helpers for building Nominatim queries from district names like
"Bath and North East Somerset Council" — strip " Council", try both with
and without administrative suffixes.

### `_is_admin_entity(name)`, `_is_landmark_name(name)` (lines 1031, 1049)

Classify a VLM-extracted name as an admin entity (use district lookup)
vs landmark (use Wikidata/Nominatim).

### `_parse_house_number`, `_parse_directional`, `_parse_land_reference`,
`_parse_parish` (lines 1057-1135)

Various parsers for site-address strings: "Land north of 50 Main Street,
Hometown" → `(direction="north", reference="50 Main Street", parish="Hometown")`.
Used to build alternative queries when the literal address doesn't geocode.

## `locate_map(pdf_path, page_idx, pdf_info, model_name=None, verbose=False)` (line 2021)

The top-level orchestrator. Roughly:

1. **Render the page** at safe DPI.
2. **Run tesseract** to get OCR words.
3. **Extract grid-ref ticks** and scale from OCR.
4. **If ≥3 grid ticks**: solve `DirectAffine`, emit a high-confidence
   candidate, store the affine.
5. **Run postcode lookup** (synchronous).
6. **Run road triangulation** (uses OSMnx).
7. **Run VLM map-label extraction** if `model_name` is set (skipped in
   benchmark mode where API calls are off).
8. **Geocode VLM places + roads** as more candidates.
9. **Run district lookup** if `is_district_wide`.
10. **Combine + rank** all candidates by confidence.
11. **Return** a `LocateResult`.

### `to_positioning_centers(result, min_confidence=0.3)` (line 2475)

Adapter from `LocateResult` to the `Center` named-tuple format used by
`tools.positioning.sliding_window_position`. Filters by min_confidence.

## Why this design

**Why so many strategies in one function?** Each strategy has different
failure modes. Grid-ticks fail on PDFs with no margin ticks; postcodes
fail on rural plans without any postcode; road triangulation fails on
plans with no road labels. Running them all and unioning means we get
candidates from whichever sources DO have data.

**Why isn't this in the agent loop?** It IS — `locate_map` is wrapped
as a fallback agent tool. But it does heavy work (OSMnx queries, VLM
calls) and would over-fire if called on every case. The agent uses
`propose_centers` (which has cheaper sources) first; `locate_map` is
only invoked as a last resort.

**Why preserve the `DirectAffine` separately from candidates?** When the
grid-tick affine is precise (mean_residual_m < 50m), the caller can use
that affine to project the SAM mask directly — skipping MINIMA. Same
shortcut as `analytical_affine_from_anchor` but derived from on-page
grid ticks instead of explicit easting/northing in PDF text.

**Why so many name parsers (`_parse_house_number`, etc.)?** UK address
strings have extreme syntactic variety: "Land north of 50 Main Street",
"Plot 4 of Field 117 (Smith's Field)", "The Old Mill, Bend in the River".
Each parser handles one common form; combined they cover ~80% of the
real cases.

**Why is this file the second-largest in the repo?** Mostly the parsers
+ disambiguation logic + multiple strategies + the data class definitions
+ the long VLM prompt. None of it is dead — but it's the kind of code
that grows organically as you encounter new edge cases. A from-scratch
rewrite could probably halve the line count, but at risk of regressing
on cases that the current version handles correctly.
