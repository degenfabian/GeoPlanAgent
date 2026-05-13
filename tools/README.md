# tools/

Core modules for the boundary extraction pipeline. `agent.py` orchestrates the
LLM tool calls; `critic.py` runs afterwards to verify and correct.

## Entry points

- `tools.agent.run_agent(pdf_path, models_state, model_name, enable_critic=True)`
  — full pipeline for a single case.
- `tools.critic.run_critic_loop(...)` — Phase 3 standalone, invoked by
  `run_agent` after the worker submits.

## Module map

### `agent.py` — Reader + Worker agents

PydanticAI agent with two phases:

- **Reader** — one-shot structured extraction. Output schema: `PDFInfo` (site
  address, postcodes, grid refs, scale, boundary colour, map rotation, map
  pages, district-wide flag).
- **Worker** — tool-calling agent producing a validated `BoundaryOutcome`.

Worker tools (one tool per `agent_tools_*.py` module):

| Tool | Module | Purpose |
|---|---|---|
| `render_page` | `agent_tools_render.py` | Render a PDF page as a BGR image |
| `propose_centers` | `agent_tools_locate.py` | locate_v2 cascade of candidate centres |
| `match_at` | `agent_tools_match.py` | MINIMA sliding-window match at one centre |
| `commit_match` | `agent_tools_match.py` | Smart commit gate over all `match_at` attempts |
| `extract_boundary` | `agent_tools_extract.py` | SAM3 mask → GeoJSON polygon (+ INSPIRE snap) |

Shared state lives on `AgentState`. The `output_validator` enforces
preconditions (borderline matches need visual checks, etc.) and re-prompts on
violations.

### `critic.py` — Phase 3 Commenter

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

### `matching.py` — Map georeferencing

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
- `_expand_thin_mask` — dilates outline-only masks (used by `retry_projection`).
- `compute_map_mpp`, `best_zoom_for_scale`, `sigma_from_scale`,
  `candidate_passes_la_filter` — scale / zoom / search-radius / LA-filter
  helpers.

### `candidates.py` — `propose_centers` cascade

`propose_centers_v2(pdf_info, …)` builds the candidate list pulled by
`agent_tools_locate.propose_centers`. Sources: postcode (Code-Point Open +
postcodes.io), OS grid_ref, parish / landmark / road inside the LA polygon
(OS Open Names with Nominatim address-level fallback), feature_cluster,
`la_centroid`, `multi_road_consensus`, `road_intersection`, and
district-lookup. `rank_candidates` scores them via feature-cluster overlap.

### `sam3_boundary.py` — Boundary segmentation

- `load_sam3_ft()` — base SAM3 + LoRA adapter.
- `extract_boundary_sam3_semantic(image_path, processor, model, device, query)`
  — single best mask via semantic segmentation.
- `extract_candidates(image_path, …, top_k=5)` — multi-candidate extraction
  ranked by confidence.
- `select_best_candidate(candidates)` — compactness + area picker.
- `try_fill_boundary_outline(mask)` — morphological close + flood fill for
  thin-outline masks.

### `geocoders.py` — Geocoding sources

- `gpkg_place_search` — OS Open Zoomstack gazetteer (offline, OGL v3).
- `wikidata_place_search` — conservation areas / historic buildings missing
  from OS data.
- `nominatim_structured` — OSM ODbL house-number / street lookups.
- `query_photon` — free-text OSM search (Apache 2.0).
- `cross_validate_centers`, `_is_valid_uk_coord`, `_distance_m` — shared
  utilities. `postcodes.io` is dispatched through `tools.code_point`.

### `os_opendata_tiles.py` — Offline OS tile rendering

Renders styled OS-planning-style tiles from the Zoomstack GeoPackage. No API
key required; OGL v3 licensed.

- `fetch_os_opendata_grid(lat, lon, zoom, nx, ny)` — NxN tile canvas + tile
  metadata (`zoom`, `tx_min`, `ty_min`, `tile_size`).
- `render_tile(zoom, tx, ty)` — single 256×256 tile.

### `geojson_metrics.py` — Evaluation metrics

- `calculate_spatial_metrics(gt_geojson, predicted_geojson)` — IoU, precision,
  recall, F1, centroid positioning error.
- `load_geojson(path)` — validated GeoJSON loader.

### `pdf_tools.py`, `text_extraction.py`

PyMuPDF rendering with pdf2image fallback; structured PDFInfo extraction from
the rendered pages.

### `visualization_tools.py`

GeoPandas + contextily helpers for predicted-vs-GT overlays
(`viz_comparison.png`).

### Other modules

- `snap/inspire.py` — INSPIRE freehold-parcel boundary snap post-processor.
- `delaunay_filter.py` — optional Delaunay-consistency RANSAC filter.
- `verification_checks.py` — cross-checks (LA polygon, scale, area) for the
  critic context.
- `code_point.py`, `os_names.py`, `positioning_sources.py` — locate-cascade
  primitives.
- `logging_utils.py`, `map_crop.py`, `mask_ops.py`, `scale_bar_ocr.py`,
  `rotation_classifier.py`, `boundary_color.py`, `locate_eval.py`,
  `reward.py` — supporting utilities.
