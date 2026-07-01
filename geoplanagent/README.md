# geoplanagent - main pipeline

Everything the benchmark runs lives here: two LLM agents, one sub-agent, an
optional critic, and the offline geospatial tools they call. Section 3 of the paper
describes how this module works in depth.

## Call flow

`run.py` orchestrates one case end to end: the **Reader** turns the PDF into a
typed `PDFInfo`; the **Worker** loops over its four tools — `propose_centers`
(which invokes the **Locate** sub-agent), `match_at`, `commit_match`,
`lookup_district` — until it submits a `BoundaryOutcome`; the optional
**Critic** reviews the committed candidate against other alternatives that were considered by the worker.
All LLM calls are done via the OpenRouter API, using pydantic-ai to validate intermediate LLM outputs.

## Files

| File | Role |
|---|---|
| `run.py` | Runs one case end to end — calls the agents in order, retries provider errors, records per-case telemetry (`agent_stats`), and makes sure a crashing case doesn't take down the whole run |
| `schemas.py` | The pydantic models passed between the stages: `PDFInfo`, `LocatePick`, `BoundaryOutcome`, `CriticDirective` |
| `prompts.py` | Every system prompt (Reader, Worker, Locate, Critic) |
| `paths.py` | All dataset, OS-data, model, and result paths in one place |
| `utils.py` | Shared helpers: model-alias resolution, fold routing, page→case aggregation, label normalisation, haversine distance calculation|
| `metrics.py` | IoU, precision/recall, centroid error and Feret diameter, plus the helpers that load and aggregate finished runs |

### agents/

| File | Role |
|---|---|
| `reader.py` | Phase 1 — one-shot `PDFInfo` extraction over the raw PDF binary |
| `locate.py` | The Locate sub-agent — an LLM whose only production tool is `place`, an offline OS Open Names gazetteer lookup. Reads `pdf_info` + the rendered map page, returns one `LocatePick` (lat, lon, σ, confidence, evidence). Five further offline geocoders exist for the all-tools ablation |
| `worker.py` | Phase 2 — the tool-calling agent definition and its output validator |
| `critic.py` | Phase 3 (optional) — builds per-candidate visual panels, asks a fresh LLM to approve / switch / retry_locate |

### tools/

| File | Role |
|---|---|
| `pdf.py` | PDF page rendering and map-page extraction |
| `tiles.py` | OS Open Zoomstack tile rendering — the basemap canvas MINIMA matches against |
| `geocode.py` | Offline UK geocoding: OS Open Names, Code-Point Open postcodes, BNG grid references, local-authority resolution |
| `matching.py` | MINIMA-LoFTR sliding-window registration, candidate re-ranking (inlier coverage, scale consistency, road-name agreement) |
| `positioning.py` | The `propose_centers` / `match_at` / `commit_match` state machine and mask→GeoJSON projection through the recovered affine |
| `rotation_classifier.py` | Auto-rotation of scanned map pages (ResNet50, 4-way TTA, k-fold routed) |
| `segment.py` | SAM3 + LoRA boundary segmentation, fold-routed per case so no case is scored by an adapter that saw its ground truth |
