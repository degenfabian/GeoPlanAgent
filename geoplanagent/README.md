# geoplanagent/

The pipeline package. Start at [`run.py`](run.py)`::run_agent` — the
per-case entry point: reader phase → map-page rendering with
auto-rotation → worker tool loop → optional critic.

| Module | Role |
|---|---|
| [`run.py`](run.py) | Orchestration: `run_agent`, phase helpers, telemetry assembly |
| [`agents/reader.py`](agents/reader.py) | Phase 1 Agent — raw PDF → `PDFInfo` |
| [`agents/worker.py`](agents/worker.py) | Phase 2 Agent + output validator + history processor |
| [`agents/locate.py`](agents/locate.py) | Locate sub-agent: geocoder tool wrappers, dynamic prompt assembly, output validator, LA-centroid fallback |
| [`agents/critic.py`](agents/critic.py) | Optional Phase 3 reviewer — approve / switch / retry_locate |
| [`tools/positioning.py`](tools/positioning.py) | The worker's tool surface, registration order: `propose_centers`, `match_at` (= `_segment_boundary` → `_search_window` → `_project_candidate`), `commit_match`, `submit_pdf_info` (folded only), `lookup_district` |
| [`tools/matching.py`](tools/matching.py) | MINIMA-LoFTR engine: sliding-window search, RANSAC affine, composite rerank, road verification, reward signals |
| [`tools/geocode.py`](tools/geocode.py) | Offline UK geocoding: OS Open Names, Code-Point Open, BNG grid refs, OS BoundaryLine |
| [`tools/segment.py`](tools/segment.py) | SAM3 + k-fold LoRA loader and semantic-head inference |
| [`tools/tiles.py`](tools/tiles.py) | OS Open Zoomstack canvas renderer + disk cache |
| [`tools/pdf.py`](tools/pdf.py) | PyMuPDF page rendering, case-PDF resolution, rotation classifier (4-way TTA) |
| [`prompts.py`](prompts.py) | Every system prompt and prompt section (LLM-visible) |
| [`schemas.py`](schemas.py) | Pydantic contracts; field descriptions are LLM-visible |
| [`utils.py`](utils.py) | `AgentState`, model aliases, HTTP retry, geodesy/tile math, k-fold routing |
| [`metrics.py`](metrics.py) | IoU / precision / recall / centroid metrics + predicted-vs-GT visualisation |
