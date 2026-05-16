# tools/

Core modules for the boundary extraction pipeline. `tools.agent` orchestrates
the LLM tool calls; `tools.agent.critic` runs afterwards to verify and correct.

## Entry points

- `tools.agent.run_agent(pdf_path, models_state, model_name, enable_critic=True)`
  — full pipeline for a single case.
- `tools.agent.critic.run_critic_loop(...)` — Phase 3 standalone, invoked by
  `run_agent` after the worker submits.

## Package map

### `tools/agent/` — Reader + Worker agents

PydanticAI agent with two phases:

- **Reader** — one-shot structured extraction. Output schema: `PDFInfo` (site
  address, postcodes, grid refs, scale, boundary colour, map rotation, map
  pages, district-wide flag).
- **Worker** — tool-calling agent producing a validated `BoundaryOutcome`.

Worker tools (one module per logical step under `tools/agent/tools/`):

| Tool | Module | Purpose |
|---|---|---|
| `render_page` | `agent/tools/render.py` | Render a PDF page as a BGR image |
| `propose_centers` | `agent/tools/locate.py` | Delegates to the live LLM-locate sub-agent in `agent/locate_agent.py` |
| `match_at` | `agent/tools/match.py` | MINIMA sliding-window match at one centre |
| `commit_match` | `agent/tools/match.py` | Smart commit gate over all `match_at` attempts |
| `extract_boundary` | `agent/tools/extract.py` | SAM3 mask → GeoJSON polygon (+ INSPIRE snap) |

Shared state lives on `AgentState` (`tools/agent/state.py`). The
`output_validator` enforces preconditions (borderline matches need visual
checks, etc.) and re-prompts on violations.

### `tools/agent/critic.py` — Phase 3 Commenter

Independent VLM agent that reviews the worker's output. Decisions:

| Decision | Action |
|---|---|
| `approve` | Proceed, no changes |
| `retry_sam` | Re-extract SAM3 with a new query / candidate, re-project |
| `retry_projection` | Apply hole-fill (`MORPH_CLOSE`) and/or `_expand_thin_mask` dilation, re-project |
| `retry_rotation` | Rotate map 90/180/270°, re-SAM, re-MINIMA at existing centres, re-project |
| `retry_in_worker` | Re-enter the worker with `message_history` replay + critic feedback |
| `flag_low_confidence` | Keep GeoJSON, label `CRITIC_LOW_CONFIDENCE` in `accept_reason` |

Budget: 2 inner critic iterations + 1 worker re-entry per case. The critic
**never nullifies** the GeoJSON.

**Hard-failure escalation.** Each `_apply_retry_*` function returns a status
string. Suffixes `_failed`, `_no_candidates`, `_no_sam_candidates`,
`_invalid`, `_no_affine`, `_projection_failed` mean the path could not execute
against the current state. The runtime short-circuits the inner critic loop
on those and escalates to worker re-entry. Soft statuses (`_noop`, `_no_mask`,
`_no_map`) stay in the inner loop so the critic can pick a different decision.

`build_critic_panel` composes the image the critic reasons over: planning map
+ SAM mask on the left, the MINIMA-matched OS tile canvas with the mask warped
through `affine_H` on the right. Both panels depict the same region at the
same scale and orientation. `build_context_text` appends worker reasoning,
tool-call counts, `centers_tried`, and prior iteration decisions.

### `tools/matching/` — Map georeferencing

- `sliding_window_position(matcher, map_img, sam3_mask, centers, scale_ratio, ...)`
  — production entry. Searches centres × zooms × rotations × window positions.
  Returns `match_info` with `n_inliers`, `score`, `aspect`, `center_latlon`,
  `zoom`, and an `affine_H`. Background-colour windows are skipped early
  (module-level `_BG_RGB`, `_BG_TOL`, `_BG_FRAC_THR` constants).
- `run_minima(matcher, map_img, tile_img, grayscale=False)` — LoFTR feature
  match.
- `estimate_affine(mkpts0, mkpts1, mconf)` — RANSAC affine fit (single-seed,
  4-DOF with a 6-DOF fallback gated on inlier improvement + shear sanity).
- `mask_to_geojson_affine(mask, affine_H, tile_info)` — project a binary mask
  to GeoJSON.
- `_expand_thin_mask` — re-exported alias of `tools.extraction.mask_ops.expand_thin_mask`
  (used by `retry_projection`).
- `compute_map_mpp`, `best_zoom_for_scale`, `sigma_from_scale`,
  `candidate_passes_la_filter` — scale / zoom / search-radius / LA-filter
  helpers.

### Locate sub-agent (`tools/agent/locate_agent.py`)

`propose_centers` in the worker calls `run_locate(pdf_info, map_img_bytes,
model_name)`, which spawns a live LLM-locate pydantic-ai agent with 6
offline geocoder tools:

- `postcode(pc)` — Code-Point Open lookup
- `grid_ref(gr)` — OS BNG parse
- `place(query, la?)` — OS Open Names settlements / landmarks
- `road(query, la?)` — OS OpenMap Local road centroid (LA-bbox filtered)
- `intersect(road_a, road_b, la?, road_c?)` — geometric junction (≤100 m)
- `la_check(lat, lon, la)` — LA polygon containment + distance

The agent views the rendered map image, runs ≤8 geocoder calls, then
returns ONE `LocatePick` (top_lat, top_lon, sigma_m, confidence, source,
evidence, la_check_passed). Pydantic-ai enforces the schema; on agent-loop
failure `run_locate` emits an emergency LA-centroid LocatePick — never
returns None.

### `tools/extraction/` — Boundary segmentation

- `sam3.load_sam3_ft()` — base SAM3 + LoRA adapter.
- `sam3.extract_boundary_sam3_semantic(image_path, processor, model, device, query)`
  — single best mask via semantic segmentation.
- `sam3.extract_candidates(image_path, …, top_k=5)` — multi-candidate
  extraction ranked by confidence.
- `sam3.select_best_candidate(candidates)` — compactness + area picker.
- `sam3.try_fill_boundary_outline(mask)` — morphological close + flood fill
  for thin-outline masks.
- `boundary_color` — HSV detection of red / coloured site outlines.
- `mask_ops` — re-usable mask-cleanup primitives (`expand_thin_mask`,
  `fill_mask_holes`, `keep_dominant_components`, `cleanup_mask_pipeline`).

### `tools/geocoding/` — Geocoding sources

- `dispatchers.gpkg_place_search` — OS Open Zoomstack gazetteer (offline,
  OGL v3).
- `dispatchers.wikidata_place_search` — conservation areas / historic
  buildings missing from OS data.
- `dispatchers.nominatim_structured` — OSM ODbL house-number / street
  lookups.
- `dispatchers.query_photon` — free-text OSM search (Apache 2.0).
- `dispatchers.cross_validate_centers`, `_is_valid_uk_coord`, `_distance_m`
  — shared utilities. `postcodes.io` is dispatched through
  `tools.geocoding.code_point`.
- `code_point` — Code-Point Open sub-metre postcode lookup.
- `os_names` — OS Open Names search.
- `positioning_sources` — anchor cascade primitives.

### `tools/io/` — PDF, tile, page-frame I/O

- `io.pdf` — PyMuPDF rendering with pdf2image fallback.
- `io.text_extraction` — structured PDFInfo extraction from rendered pages.
- `io.os_tiles.fetch_os_opendata_grid(lat, lon, zoom, nx, ny)` — NxN tile
  canvas + tile metadata (`zoom`, `tx_min`, `ty_min`, `tile_size`). Styled
  OS-planning-style raster from the Zoomstack GeoPackage. No API key
  required; OGL v3 licensed.
- `io.os_tiles.render_tile(zoom, tx, ty)` — single 256×256 tile.
- `io.rotation_classifier.auto_rotate` — VLM-judged 0/90/180/270° page
  rotation.
- `io.map_crop.detect_title_block_crop` — crop title block / legend so the
  matcher sees only the map.

### `tools/metrics/` — Evaluation metrics & visualisation

- `geojson.calculate_spatial_metrics(gt_geojson, predicted_geojson)` — IoU,
  precision, recall, F1, centroid positioning error.
- `geojson.load_geojson(path)` — validated GeoJSON loader.
- `visualization.visualize_comparison` — GeoPandas + contextily helpers for
  predicted-vs-GT overlays (`viz_comparison.png`).
- `reward` — MINIMA-axis reward shaping used by `agent.tools.match`.

### Other modules

- `delaunay_filter.py` — optional Delaunay-consistency RANSAC filter.
- `verification_checks.py` — cross-checks (LA polygon, scale, area) for the
  critic context.
- `scoring.py` — `composite_window_score`, `commit_attempt_score`.
- `build_oml_road_index.py` — regenerate the OML road index/geometry caches.
