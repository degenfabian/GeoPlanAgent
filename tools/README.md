# tools/

Core modules for the boundary-extraction pipeline. `tools.agent` orchestrates
two LLM agents (reader → worker) plus two sub-agents (locate, reader-refine).
Matching is MINIMA (LoFTR); segmentation is SAM3 + LoRA; geocoding is offline
(Code-Point Open + OS Open Names + OML road index).

## Entry points

- `tools.agent.run_agent(pdf_path, models_state, model_name)` — full pipeline
  for a single case.

## Package map

### `tools/agent/` — Two top-level agents + sub-agents

| File | Role |
|---|---|
| `reader_agent.py` | Phase 1: one-shot read of the PDF binary → `PDFInfo` |
| `worker_agent.py` | Phase 2: tool-calling agent → `BoundaryOutcome` |
| `locate_agent.py` | Sub-agent called by the worker's `propose_centers` — six offline geocoder tools, one `LocatePick` out |
| `runtime.py` | Phase orchestration helpers called by `run_agent` |
| `state.py` | `AgentState` (mutable per-case state passed as deps) |
| `schemas.py` | `PDFInfo`, `BoundaryOutcome`, `MapPageMeta`, `LocatePick` |
| `prompts.py` | `READER_SYSTEM_PROMPT`, `WORKER_SYSTEM_PROMPT` |
| `_model.py` | `resolve_model` + alias table |
| `_helpers.py` | Image/dedup/overlay helpers |
| `_retry.py` | Transient-HTTP-error retry helper |

### `tools/agent/tools/` — Worker tools

Each module registers its tool against `_agent` via `@_agent.tool` at import
time.

| Tool | Module | Purpose |
|---|---|---|
| `render_page` | `render.py` | Switch active map page (state-pointer flip; falls back to fresh fitz render) |
| `geocode` | `locate.py` | Postcode / grid_ref lookups the worker spots on the map after `propose_centers` |
| `propose_centers` | `locate.py` | Calls `tools.agent.locate_agent.run_locate` — locate sub-agent picks ONE center |
| `match_at` | `match.py` | MINIMA sliding-window match at one centre |
| `commit_match` | `match.py` | Smart-commit gate over accumulated `match_at` attempts |
| `extract_boundary` | `extract.py` | SAM3 semantic mask (optional bbox) |
| `project_boundary` | `extract.py` | Mask → GeoJSON via committed affine |
| `verify_position` | `verify.py` | Render OS tiles with predicted polygon for visual check |
| `lookup_district` | `verify.py` | OSM district polygon fallback (used when `is_district_wide`) |
| `visualize` | `verify.py` | Show current mask + tile overlay |
| `reader_refine` | `refine.py` | Fresh small-context call on the PDF binary for a focused question (budget 3/case) |

### `tools/matching/` — Map georeferencing

- `sliding_window_position(matcher, map_img, sam3_mask, centers, scale_ratio, ...)`
  — production entry. Searches centres × zooms × rotations × window
  positions. Returns `match_info` with `n_inliers`, `score`, `aspect`,
  `center_latlon`, `zoom`, and `affine_H`.
- `run_minima(matcher, map_img, tile_img, grayscale=False)` — LoFTR
  feature match.
- `estimate_affine(mkpts0, mkpts1, mconf)` — RANSAC affine. 4-DOF
  similarity by default, 6-DOF full-affine fallback gated on inlier
  improvement (`GATE_RATIO_6DOF=1.3`) + scale band
  (`SCALE_6DOF_MIN=0.3, SCALE_6DOF_MAX=3.0`) + det/shear sanity.
- `mask_to_geojson_affine(mask, affine_H, tile_info)` — project a binary
  mask through the committed affine.
- `source_priorities` — `_SOURCE_SIGMA_M`, `effective_sigma`,
  `sigma_from_scale`, `candidate_passes_la_filter` — per-source σ
  defaults + LA-polygon filter used by `sliding_window_position`.
- `road_verify._verify_candidates_with_road_names` — OSM road-name
  cross-check of MINIMA candidates.

### `tools/agent/locate_agent.py` — Locate sub-agent

`propose_centers` calls `run_locate(pdf_info, map_img_bytes, model_name,
match_context, prior_messages)`. The sub-agent has 6 offline geocoder
tools:

- `postcode(pc)` — Code-Point Open lookup (sub-100 m).
- `grid_ref(gr)` — OS BNG parse.
- `place(query, la?)` — OS Open Names settlements / landmarks / churches.
- `road(query, la?)` — OS OpenMap Local road centroid (LA-bbox filtered).
- `intersect(road_a, road_b, la?, road_c?)` — geometric junction (≤100 m).
- `la_check(lat, lon, la)` — LA polygon containment + distance.

It views the rendered map image, runs ≤8 tool calls, then returns one
`LocatePick(top_lat, top_lon, sigma_m, confidence, picked_source,
evidence, la_check_passed)`. Pydantic-ai enforces the schema; on
agent-loop failure `run_locate` emits an emergency LA-centroid pick
rather than returning None.

`prior_messages` carries the agent's full conversation across worker
re-calls so the sub-agent SEES its own previous reasoning + the new
`match_context` (worker's feedback after a poor match_at).

### `tools/extraction/` — Boundary segmentation

- `sam3.load_sam3_ft()` — base SAM3 + LoRA adapter (k-fold).
- `sam3.extract_boundary_sam3_semantic(image_path, processor, model, device, query, bbox)`
  — single best mask via semantic segmentation. The query is locked to
  `"planning boundary"` (the LoRA was trained against it).
- `sam3.try_fill_boundary_outline(mask)` — morphological close + flood
  fill for thin-outline masks.
- `mask_ops` — `expand_thin_mask`, `fill_mask_holes`,
  `keep_dominant_components`, `cleanup_mask_pipeline`.

### `tools/geo/` — Geographic primitives

- `coords` — Web-Mercator / tile-pixel math, BNG ↔ WGS84, `haversine_m`.
- `grid_ref` — OS BNG grid-reference parser; `lookup_district_boundary`
  via OS BoundaryLine (offline) for the worker's `lookup_district` tool.
- `code_point` — Code-Point Open sub-metre postcode lookup.
- `os_names` — OS Open Names settlement / landmark / road search.

### `tools/io/` — PDF, tile, page-frame I/O

- `io.pdf` — PyMuPDF page rendering.
- `io.text_extraction` — fitz/PaddleOCR per-page text for the reader prompt
  (cached on disk).
- `io.os_tiles.fetch_os_opendata_grid(lat, lon, zoom, nx, ny)` — NxN tile
  canvas from the Zoomstack GeoPackage (OGL v3, no API key).
- `io.rotation_classifier.auto_rotate` — trained ResNet50 classifier
  (TTA + abstain) for 0/90/180/270° page rotation.
- `io.map_page.render_map_page` — single source of truth for the
  `render → auto_rotate` pipeline used everywhere a planning page is
  rendered.

### `tools/metrics/` — Evaluation + visualisation

- `geojson.calculate_spatial_metrics(gt_geojson, predicted_geojson)` —
  IoU, precision, recall, F1, centroid positioning error.
- `geojson.load_geojson(path)` — validated GeoJSON loader.
- `visualization.visualize_comparison` — GeoPandas + contextily
  predicted-vs-GT overlay (`viz_comparison.png`).
- `reward.compute_match_reward` — multi-axis MINIMA reward used by
  `match_at`.

### Other top-level modules

- `scoring.py` — `composite_window_score`, `commit_attempt_score`
  (single source of truth for match-stage candidate ranking).
- `delaunay_filter.py` — Delaunay-consistency RANSAC filter applied
  inside `estimate_affine`.
- `verification_checks.py` — OS BoundaryLine LA-polygon resolver
  (`_resolve_la`, `_load_la_polygons`) used by the worker's
  `lookup_district`, locate sub-agent's `la_check`, and the smart-commit
  LA filter.
- `build_oml_road_index.py` — script to regenerate the OML road indexes
  consumed by the locate sub-agent's `road` / `intersect` tools.

## Empirically-tuned constants (single source of truth)

| Constant | Home | Value | Note |
|---|---|---|---|
| `GATE_RATIO_6DOF` | `tools/matching/_core.py` | 1.3 | 6-DOF affine fallback threshold |
| `SCALE_6DOF_MIN/MAX` | `tools/matching/_core.py` | 0.3 / 3.0 | 6-DOF affine scale-sanity band |
| `WINDOW_STRIDE_TARGET` | `tools/matching/_core.py` | 100 | Sliding-window stride target |
| `REFINE_BUDGET_PER_CASE` | `tools/agent/tools/refine.py` | 3 | Cap on `reader_refine` calls |
| `OUTSIDE_LA_PENALTY` | `tools/scoring.py` | 0.3 | Smart-commit penalty for outside-LA picks |
