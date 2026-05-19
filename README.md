# GeoPlanAgent

Autonomous extraction of the application-site boundary from UK planning
permission PDFs. An LLM agent reads each PDF, geocodes likely locations,
positions the planning map against Ordnance Survey OpenData tiles via
learned feature matching (MINIMA / LoFTR), segments the boundary with a
fine-tuned SAM3 model, and projects the result to a WGS84 GeoJSON polygon.

## Pipeline

Two LLM phases plus one inline sub-agent. There is no critic; the pipeline
always emits a polygon and downstream scores it via IoU against
ground-truth GeoJSON.

```
                Phase 1                Phase 2 — Worker agent loop
                Reader                 (5 tools)
                                       ─────────────────────────────
 PDF ────►  PDFInfo (JSON)  ────►  propose_centers  ──┐
            • site_address          (calls locate     │
            • postcodes              sub-agent —      │
            • grid_refs              6 geocoders)     │
            • map_pages                                ▼
              (ranked per       match_at(page=N) ──► commit_match
              area_group)         (MINIMA at one      (smart-commit
            • district info       centre, automatic    gate, projects
            • text & visual       SAM3 + projection    SAM mask →
              cues for locate     across all groups)   GeoJSON)
                                                       │
                                                       ▼
                                       BoundaryOutcome → final GeoJSON
                                                       │
                                       (optional) LLM critic — pairwise
                                       review across stored candidates;
                                       may direct switch / retry_locate
                                       (enable_critic=True; default off)
```

## Project structure

```
GeoMapAgent_autonomous/
├── README.md
├── benchmark_runner.py        # Evaluation driver across the dataset
├── pyproject.toml             # Dependencies (uv-managed)
├── uv.lock
│
├── tools/                     # Core pipeline (see tools/README.md)
│   ├── agent/                 # Reader + worker + locate sub-agent
│   ├── matching/              # MINIMA sliding-window + RANSAC
│   ├── extraction/            # SAM3 + LoRA k-fold + mask ops
│   ├── geo/                   # Offline geocoders + BNG ↔ WGS84
│   ├── io/                    # PDF render, OS tiles, OCR, rotation
│   ├── metrics/               # IoU/F1, MINIMA reward, viz
│   ├── scoring.py             # commit_attempt_score, composite_window_score
│   └── verification_checks.py # OS BoundaryLine LA-polygon resolver
│
├── ablations/                 # Paper ablation scripts (see ablations/README.md)
├── training/                  # SAM3 LoRA fine-tune (see training/README.md)
├── scripts/                   # One-off data-prep + eval scripts
│
├── MINIMA/                    # LoFTR matcher (external, gitignored)
├── evaluation_data/           # Test dataset (PDFs + GT GeoJSON, gitignored)
├── models/                    # Model weights (gitignored, see models/README.md)
├── os_opendata/               # OS OpenData (Zoomstack, BoundaryLine,
│                              # OpenNames, OpenMapLocal, Code-Point Open)
│                              # — gitignored
├── cache/                     # text_extraction + tile caches
└── results/                   # Benchmark outputs (gitignored)
```

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv isn't installed
uv sync
```

## Configuration

```bash
cp .env.template .env
```

| Variable | Required | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | yes | LLM access via OpenRouter |
| `HF_TOKEN` | yes | HuggingFace — SAM3 base-weight download on first run |

## Usage

### Full benchmark

```bash
uv run benchmark_runner.py \
    --model gemini-flash \
    --max-iterations 12 \
    --output-dir results/benchmark_v1 \
    --force
```

### Common flags

```bash
uv run benchmark_runner.py --max-cases 10              # quick smoke test
uv run benchmark_runner.py --cases 12:00116:ART4       # single case
uv run benchmark_runner.py --hard-first \              # failing cases first
    --prev-results results/benchmark_v0
uv run benchmark_runner.py --start-from 50             # resume after N
uv run benchmark_runner.py --force                     # bypass per-case cache
```

Results go to `results/<output-dir>/<model_id_with_slashes→underscores>/`.

### Model aliases

| Alias | OpenRouter model ID |
|---|---|
| `gemini-pro` | `google/gemini-3.1-pro-preview` |
| `gemini-flash` | `google/gemini-3-flash-preview` |
| `gemini-flash-lite` | `google/gemini-3.1-flash-lite-preview` |
| `claude-sonnet` | `anthropic/claude-sonnet-4-6` |
| `claude-opus` | `anthropic/claude-opus-4.6` |
| `gpt-5.4` | `openai/gpt-5.4` |
| `gpt-5.4-mini` / `gpt-5.4-nano` | corresponding OpenAI IDs |

Any OpenRouter ID is accepted directly. Defined in
[`tools/agent/_model.py`](tools/agent/_model.py).

### Programmatic single-case usage

```python
from tools.agent import run_agent
from tools.extraction.sam3 import load_sam3_ft
from tools.matching import load_minima

models = {"sam3_ft": load_sam3_ft(), "minima": load_minima()}

result = run_agent(
    pdf_path="evaluation_data/12:00116:ART4/document.pdf",
    models_state=models,
    model_name="gemini-flash",
    max_iterations=12,
    case_name="12:00116:ART4",   # for SAM3 k-fold adapter routing
)

if result["geojson"]:
    print(f"Inliers: {result['match_info'].get('n_inliers')}")
    # result['geojson'] is a Feature dict with MultiPolygon geometry
```

## Phase details

### Phase 1 — Reader

One-shot pydantic-ai call (`output_type=PDFInfo`). Sees the raw PDF binary
plus a per-page text block extracted by fitz (digital pages) or macOS
Vision / PaddleOCR (scanned pages, cached on disk). Populates a
schema covering: site address, postcodes, grid references, scale,
ranked map pages with per-page area-group / boundary-clarity /
detail-level metadata, district info, and locate-stage signals
(road names, place names, parishes, admin region, visible map
labels). Cached for speed.

### Phase 2 — Worker

Tool-calling pydantic-ai agent. Four worker tools:

1. **`propose_centers(extra_terms?, match_context?)`** — invokes the
   locate sub-agent (see `tools/agent/README.md`), which uses six
   offline geocoders (postcode, grid_ref, place, road, intersect,
   la_check) to return ONE picked centre with σ + confidence + provenance.
2. **`match_at(page=N, name, lat, lon)`** — runs MINIMA at the supplied
   centre. For multi-area-group documents one call handles every group
   automatically (per-group MINIMA on each group's primary page, per-page
   SAM3 mask caching, polygons UNIONed). Returns a multi-axis reward
   (overall_score, total_inliers, road_name_agreement, scale_consistency)
   — numbers only.
3. **`commit_match(candidate_id)`** — picks the best stored match_at
   attempt as the active result and projects the SAM mask through the
   committed affine to GeoJSON. The smart-commit gate combines
   `total_inliers` with an outside-LA penalty so a worse pick gets
   redirected; the strict gate rejects commits where MINIMA produced
   no usable affine for any group.
4. **`lookup_district(district_name)`** — OS BoundaryLine offline lookup
   for documents that cover an entire admin district (e.g. Article 4
   directions, conservation-area-wide documents). When this succeeds
   the worker submits `status="district_lookup"` and the polygon comes
   straight from BoundaryLine — no SAM, no positioning.

Plus `reader_refine(question, page_hint?)` — re-consults the source PDF
(binary + cached OCR text) for a focused question. Budget 3 per case.

The worker output (`BoundaryOutcome`) is validated by an
`output_validator` that re-reads tool-call state, so the worker can't
report flags it didn't actually set.

### Phase 3 (optional) — Independent LLM critic

Opt-in via `--enable-critic` (or `enable_critic=True` to `run_agent`).
After the worker submits, a separate LLM call sees the visual panels
for ALL stored match candidates plus per-candidate metrics
(`n_inliers`, `road_name_agreement`, `scale_consistency`) and emits a
`CriticDirective` with `action ∈ {approve, switch, retry_locate}`. On
switch / retry_locate the worker is re-invoked via a templated user
message (neutral framing — the worker stays opaque to the critic's
existence during initial exploration). Max 2 rejections per case.

The worker's first-committed polygon is snapshotted before the loop
fires, so a single run produces paired no-critic / with-critic IoUs
for the ablation. Defined in `tools/agent/critic_agent.py`.

## Output layout (per case)

```
results/<output-dir>/<model>/<case>/
├── predicted.geojson       # Extracted boundary (Feature with MultiPolygon)
├── metrics.json            # IoU, precision, recall, F1, agent_stats,
│                           # match_info, processing_time
├── message_log.json        # Full worker conversation trace
├── pdf_info.json           # Phase 1 structured extraction
├── boundary_mask.png       # Binary SAM3 mask on the committed page
├── affine_H.npy            # 2×3 committed affine (page → tile pixel)
├── tile_info.json          # Zoom / tx_min / ty_min for affine_H
├── selected_boundary.png   # Final SAM-overlay on planning page
└── viz_comparison.png      # Predicted vs ground truth (geopandas + contextily)
```

A run-level `summary.json` lands at `results/<output-dir>/<model>/summary.json`
with aggregate IoU stats (production-honest and polygon-only) plus
per-case rows.

## Positioning quality signals

| `n_inliers` (RANSAC) | Meaning |
|---|---|
| ≥ 100 | Strong match |
| 50–100 | Decent — should still pass smart-commit gate |
| 25–50 | Borderline — worker must try at least one more candidate |
| < 25 | Weak — worker should try another centre |

Multi-axis reward axes (returned in each `match_at` per-group entry):

- `overall_score` — weighted geometric mean of the axes (0-1)
- `n_inliers` / `score` — RANSAC quality
- `road_name_agreement` (+ verdict) — do reader-extracted road names
  appear in OS Open Zoomstack at the matched location?
- `scale_consistency` — does the recovered affine scale agree with the
  reader's stated map scale?

## External dependencies (offline)

- **MINIMA** — LoFTR-based map-to-tile matcher. Clone into `MINIMA/`;
  weights in `MINIMA/weights/`.
- **SAM3 + LoRA** — Facebook SAM3 base weights auto-downloaded from
  HuggingFace on first run (`HF_TOKEN`). The fine-tuned LoRA adapter is
  shipped per fold in `models/sam3_lora/fold_*/best.pt`.
- **OS OpenData** — `OS_Open_Zoomstack.gpkg`, BoundaryLine, OpenNames,
  OpenMapLocal, Code-Point Open. All OGL v3, no API key, no rate limit.
  Placed under `os_opendata/`. Setup instructions live in the relevant
  geocoder modules.

## Requirements

- Python 3.10+ (managed via `uv`)
- macOS with MPS or Linux with CUDA for GPU acceleration
- ~10 GB disk: SAM3 base weights (~3 GB) + LoRA adapters (~1 GB) +
  OS OpenData (~5 GB) + tile cache (~200 GB at full benchmark scale,
  but cached lazily)
