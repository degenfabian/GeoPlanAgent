# Code Walkthrough

Per-file walkthroughs of every Python module in production. Read these in
order if you want to understand the pipeline; jump to a specific file if
you're debugging or extending one part.

## Pipeline order (how data flows)

1. **`benchmark_runner.py`** — entry point. Loads the case list, instantiates
   the worker, calls the agent per case, writes metrics.
2. **`tools/agent.py`** — the LLM agent loop and the 9 tools the LLM calls.
   This is the orchestrator — every other tool exists to be called from here.
3. **PDF intake**:
   - **`tools/pdf_tools.py`** — render a page to an image
   - **`tools/text_extraction.py`** — extract text via fitz + OCR cascade
   - **`tools/locate.py`** — OCR with positions, GCP candidate generation
4. **Geocoding**:
   - **`tools/geocoding.py`** — Photon / Nominatim / Wikidata / OS gpkg lookups
   - **`tools/geo_tools.py`** — OSGB grid parsing, easting/northing → lat/lon
5. **Map matching**:
   - **`tools/positioning.py`** — MINIMA sliding-window matcher; mask projection
   - **`tools/os_opendata_tiles.py`** — render OS tiles from GeoPackage
   - **`tools/rotation_classifier.py`** — fix planning-page orientation
   - **`tools/map_crop.py`** — strip title-block from rendered page
6. **Boundary extraction**:
   - **`tools/sam3_boundary.py`** — SAM3 inference + multi-prompt
   - **`tools/boundary_color.py`** — colour-line fallback
7. **Quality + output**:
   - **`tools/reward.py`** — multi-axis match scoring
   - **`tools/verifier.py`** — visual confirmation panels
   - **`tools/critic.py`** — Phase-3 VLM that reviews worker output
   - **`tools/geojson_metrics.py`** — IoU computation against ground truth
   - **`tools/visualization_tools.py`** — produce the comparison overlays
8. **Training** (separate from runtime):
   - **`training/train_sam3_kfold.py`** — produces `models/sam3_lora_v7_both/`
   - **`training/train_rotation_classifier.py`** — produces `models/rotation_classifier/`
   - **`scripts/train_verifier.py`** — produces `models/verifier_v3/`
   - **`scripts/build_curated_training_set.py`** — assembles `training/dataset_v5/`

## Per-file walkthroughs (alphabetical)

| File | Walkthrough |
|---|---|
| `tools/agent.py` | [agent.md](agent.md) |
| `tools/boundary_color.py` | [boundary_color.md](boundary_color.md) |
| `tools/critic.py` | [critic.md](critic.md) |
| `tools/geocoding.py` | [geocoding.md](geocoding.md) |
| `tools/geo_tools.py` | [geo_tools.md](geo_tools.md) |
| `tools/geojson_metrics.py` | [geojson_metrics.md](geojson_metrics.md) |
| `tools/locate.py` | [locate.md](locate.md) |
| `tools/locate_eval.py` | [locate_eval.md](locate_eval.md) |
| `tools/map_crop.py` | [map_crop.md](map_crop.md) |
| `tools/os_opendata_tiles.py` | [os_opendata_tiles.md](os_opendata_tiles.md) |
| `tools/pdf_tools.py` | [pdf_tools.md](pdf_tools.md) |
| `tools/positioning.py` | [positioning.md](positioning.md) |
| `tools/reward.py` | [reward.md](reward.md) |
| `tools/rotation_classifier.py` | [rotation_classifier.md](rotation_classifier.md) |
| `tools/sam3_boundary.py` | [sam3_boundary.md](sam3_boundary.md) |
| `tools/text_extraction.py` | [text_extraction.md](text_extraction.md) |
| `tools/verifier.py` | [verifier.md](verifier.md) |
| `tools/visualization_tools.py` | [visualization_tools.md](visualization_tools.md) |
| `benchmark_runner.py` | [benchmark_runner.md](benchmark_runner.md) |

## Reading style

Each walkthrough has the same structure:

- **What it's for** — one paragraph
- **Public API** — what other files call from here
- **Function-by-function** — name, purpose, inputs, gotchas
- **Why this design** — for the parts that aren't obvious

Code that's self-evident from variable names is skipped. Math (affines,
projections) is explained. Recovery-experiment context is referenced where
relevant (e.g. "this fallback was added after Phase 21 of the recovery
showed cached SAM masks were sometimes wrong").
