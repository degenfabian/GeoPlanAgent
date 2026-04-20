# tools/

Core modules for the boundary extraction pipeline. `agent.py` orchestrates the pipeline via LLM tool calls; `critic.py` runs afterwards to verify and correct.

## Entry Points

- `tools.agent.run_agent(pdf_path, models_state, model_name, enable_critic=True)` — full pipeline for a single case.
- `tools.critic.run_critic_loop(...)` — Phase 3 standalone, invoked by `run_agent` after the worker submits.

## Module Map

### `agent.py` — Reader + Worker agents

PydanticAI agent with two phases:

- **Reader** (`_reader_agent`) — one-shot structured extraction. Output schema: `PDFInfo` (site address, postcodes, grid refs, scale, boundary colour, map rotation, map pages, district-wide flag).
- **Worker** (`_agent`) — tool-calling agent that produces a validated `BoundaryOutcome`.

Worker tools:

| Tool | Purpose |
|---|---|
| `render_page` | Render a PDF page as a BGR image |
| `geocode` | Postcode or OS grid-ref lookup (place names are auto-geocoded inside `position_boundary`) |
| `position_boundary` | MINIMA sliding-window match against OS tiles; auto-geocodes from five sources |
| `extract_boundary` | SAM3 segmentation (semantic or instance mode) |
| `project_boundary` | Affine-project the mask to a GeoJSON polygon |
| `accumulate_boundary` | Save current page, reset per-page state for multi-page maps |
| `verify_position` | Fetch OS tiles at the matched centre, draw the polygon, return for visual inspection |
| `lookup_district` | Pull an OSM administrative boundary (district-wide cases) |
| `visualize` | Render debug overlays for the agent |

Shared state lives on `AgentState`. The `output_validator` enforces preconditions (borderline positioning needs `verify_position`, multi-page needs `accumulate_boundary`, etc.) and re-prompts the agent on violations.

The `_strip_old_images` history processor replaces `BinaryContent` older than 4 messages with a text placeholder, preventing quadratic token growth when the agent calls `extract_boundary(mode='instance')` — which attaches 5 candidate images per call.

### `critic.py` — Phase 3 Commenter

Independent VLM agent that reviews the worker's output. Decisions:

| Decision | Action |
|---|---|
| `approve` | Proceed, no changes |
| `retry_sam` | Re-extract SAM3 with a new query / candidate selection, re-project |
| `retry_projection` | Apply hole-fill (`MORPH_CLOSE`) and/or `_expand_thin_mask` dilation, re-project |
| `retry_rotation` | Rotate map 90/180/270°, re-SAM, re-run MINIMA with the existing centres, re-project |
| `retry_in_worker` | Re-enter the worker with `message_history` replay + critic feedback |
| `flag_low_confidence` | Keep GeoJSON, label `CRITIC_LOW_CONFIDENCE` in `accept_reason` |

Budget: 2 inner critic iterations + 1 worker re-entry per case. Critic is skipped for multi-page and district-lookup cases. The critic **never nullifies** the GeoJSON.

**Hard-failure escalation.** Each `_apply_retry_*` function returns a status string. Suffixes `_failed`, `_no_candidates`, `_no_sam_candidates`, `_invalid`, `_no_affine`, `_projection_failed` mean the code path could not execute against the current state. When the runtime sees one of those, it short-circuits the inner critic loop and escalates to worker re-entry — re-polling the critic on unchanged state would just reproduce the same failure. Soft statuses (`_noop`, `_no_mask`, `_no_map`) stay in the inner loop so the critic can pick a different decision. The system prompt also instructs the critic to switch away from a decision whose prior `fix_applied` ends in a hard-failure suffix, as a belt-and-braces backstop.

`build_critic_panel` composes the image the critic reasons over: planning map with SAM mask on the left, OS tiles with projected polygon on the right. `build_context_text` appends worker reasoning, tool-call counts, `centers_tried`, and prior iteration decisions (including their `fix_applied` strings, so the critic can see what failed).

### `positioning.py` — Map georeferencing

- `sliding_window_position(matcher, map_img, sam3_mask, centers, scale_ratio, ...)` — production entry. Searches across centres × zooms × rotations × window positions. Returns `match_info` with `n_inliers`, `score`, `aspect`, `center_latlon`, `zoom`, and an `affine_H`.
- `run_minima(matcher, map_img, tile_img, grayscale=False)` — LoFTR feature match.
- `estimate_affine(mkpts0, mkpts1, mconf)` — RANSAC affine fit.
- `mask_to_geojson_affine(mask, affine_H, tile_info)` — project a binary mask to a GeoJSON polygon.
- `_expand_thin_mask` — dilates outline-only masks into filled shapes (used by `retry_projection`).
- `compute_map_mpp`, `best_zoom_for_scale`, `sigma_from_scale` — scale / zoom / search-radius helpers.

Quality metric is `n_inliers`; the scoring combines inlier count with per-match confidence.

### `sam3_boundary.py` — Boundary segmentation

- `load_sam3_ft()` — load base SAM3 + LoRA adapter.
- `extract_boundary_sam3_semantic(image_path, processor, model, device, query)` — single best mask via semantic segmentation.
- `extract_candidates(image_path, ..., top_k=5)` — multi-candidate extraction ranked by confidence. Used by the worker's `instance` mode.
- `select_best_candidate(candidates)` — compactness + area-based picker.
- `try_fill_boundary_outline(mask)` — morphological close + flood fill for thin-outline masks.

### `geocoding.py` — Geocoding sources

Central dispatcher plus source-specific clients:

- `gpkg_place_search` — OS Open Zoomstack gazetteer (offline, OGL v3).
- `wikidata_place_search` — conservation areas, historic buildings missing from OS data.
- `nominatim_structured` — house-number and street-level lookups (OSM ODbL).
- `query_photon` — free-text OSM search (Apache 2.0).
- `postcodes.io` lookups are handled inline in `agent.geocode(type="postcode")`.
- `cross_validate_centers`, `_is_valid_uk_coord`, `_distance_m` — shared utilities for sanity-filtering and deduplication.

### `geo_tools.py` — Geographic utilities

- `os_grid_ref_to_latlon(ref)` — parse OS grid references (letter-pair + 2–10 digits, or easting/northing form) into WGS84.
- `lookup_district_boundary(name)` — OSM admin boundary via osmnx for district-wide cases.
- `pixels_to_geo_linear(...)` — linear pixel→geo fallback when MINIMA cannot match.

### `os_opendata_tiles.py` — Offline OS tile rendering

Renders styled OS-planning-style tiles from the Zoomstack GeoPackage. No API key; OGL v3 licensed.

- `fetch_os_opendata_grid(lat, lon, zoom, nx, ny)` — returns an NxN tile canvas + tile metadata (`zoom`, `tx_min`, `ty_min`, `tile_size`).
- `render_tile(zoom, tx, ty)` — single 256×256 tile from GeoPackage layers.

### `geojson_metrics.py` — Evaluation metrics

Shapely-based spatial metrics for benchmark analysis.

- `calculate_spatial_metrics(gt_geojson, predicted_geojson)` — IoU, precision, recall, F1, positioning error (haversine centroid distance).
- `load_geojson(path)` — validated GeoJSON loader.

### `pdf_tools.py` — PDF rendering

PyMuPDF with a pdf2image fallback. Used by `render_page` and the reader phase.

### `visualization_tools.py` — Visualisation

GeoPandas + contextily helpers for comparing predicted vs ground-truth boundaries. Outputs `viz_comparison.png`.
