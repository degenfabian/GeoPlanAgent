# GeoPlanAgent

Autonomous extraction of the application-site boundary from UK planning
permission PDFs. An LLM agent reads each PDF, geocodes likely locations,
positions the planning map against Ordnance Survey OpenData tiles via
learned feature matching (MINIMA / LoFTR), segments the boundary with a
fine-tuned SAM3 model, and projects the result to a WGS84 GeoJSON polygon.

## Pipeline

Two LLM phases plus an inline locate sub-agent, with an optional third
LLM critic phase. The pipeline always emits a polygon and downstream
scores it via IoU against ground-truth GeoJSON.

```
                Phase 1                Phase 2 — Worker agent loop
                Reader                 (4 tools)
                                       ─────────────────────────────
 PDF ────►  PDFInfo (JSON)  ────►  propose_centers  ──┐
            • site_address          (calls locate     │
            • postcodes              sub-agent —      │
            • grid_refs              6 geocoders)     │
            • map_pages                                ▼
              (ranked per       match_at(page=N) ──► commit_match
              area_group)         (MINIMA + SAM3       (commits this
            • district info       on ONE page; one     candidate for
            • text & visual       area_group at a      its area_group;
              cues for locate     time)                unions running
                                                       result. Loop
                                                       once per group
                                                       on multi-area
                                                       documents.)
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
│   ├── agent/                 # Reader + worker + locate sub-agent + critic
│   ├── matching/              # MINIMA sliding-window + RANSAC
│   ├── extraction/            # SAM3 + LoRA k-fold loader (single module)
│   ├── geo/                   # Offline geocoders + BNG ↔ WGS84
│   ├── io/                    # PDF render, OS tiles, rotation classifier
│   ├── metrics/               # IoU/F1, MINIMA reward, viz
│   ├── scoring.py             # composite_window_score (sliding-window reranker)
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
├── cache/                     # OS tile cache (gitignored)
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

One-shot pydantic-ai call (`output_type=PDFInfo`) over the raw PDF
binary — no OCR pipeline, the VLM reads the PDF directly. Populates a
schema covering: site address, postcodes, grid references, scale,
ranked map pages with per-page area-group / boundary-clarity /
detail-level metadata, district info, and locate-stage signals
(road names, place names, parishes, admin region, visible map
labels).

### Phase 2 — Worker

Tool-calling pydantic-ai agent. Four worker tools:

1. **`propose_centers(extra_terms?, match_context?)`** — invokes the
   locate sub-agent (see `tools/agent/README.md`), which uses six
   offline geocoders (postcode, grid_ref, place, road, intersect,
   la_check) to return ONE picked centre with σ + confidence + provenance.
2. **`match_at(page=N, name, lat, lon)`** — runs MINIMA + SAM3 on the
   supplied centre and ONE page (one area_group). Stores one
   candidate covering that group. Returns numeric signals only
   (`n_inliers`, `scale_consistency`, `road_name_agreement`,
   `area_group`, `page`).
3. **`commit_match(candidate_id)`** — commits this candidate for its
   area_group. The polygon was already projected inside match_at;
   commit_match adds (or replaces) the group's slot in the running
   result and unions all committed groups into the final geojson.
   A strict gate rejects commits where MINIMA produced no valid affine.
4. **`lookup_district(district_name)`** — OS BoundaryLine offline lookup
   for documents that cover an entire admin district (e.g. Article 4
   directions, conservation-area-wide documents). When this succeeds
   the worker submits `status="district_lookup"` and the polygon comes
   straight from BoundaryLine — no SAM, no positioning.

For multi-area documents the worker iterates the
`propose_centers → match_at → commit_match` loop once per
area_group; the final geojson is the union of every committed
group's polygon. Most documents are single-area, in which case the
loop runs once.

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

The worker prompt formalises four explicit tiers per signal (matched
exactly in the critic prompt for cross-phase consistency):

| `n_inliers` (RANSAC) | Tier | Meaning |
|---|---|---|
| ≥ 100 | STRONG    | Commit on this attempt unless another signal disagrees |
| 50–99 | OK        | Commit only after trying at least one more candidate |
| 25–49 | WEAK      | Keep exploring; don't commit yet |
| < 25  | TOO WEAK  | Try another candidate; never commit unless budget exhausted |

| `scale_consistency` | Tier | Meaning |
|---|---|---|
| ≥ 0.8   | GOOD     | Recovered scale matches stated map scale |
| 0.5–0.8 | MARGINAL | Stretched; prefer an alternative if you have one |
| < 0.5   | BAD      | Scale very off; trust only if `n_inliers ≥ 100` |

| `road_name_agreement` | Tier | Meaning |
|---|---|---|
| ≥ 0.6 | STRONG   | Reader's road names found at the matched location |
| 0.0   | CONFLICT | OS has roads but none match reader; possible wrong area |
| 0.5   | NEUTRAL  | "No OS roads in radius" (sparse cartography); no signal |
| other | PARTIAL  | Some roads matched; weak corroboration |

Tie-break order across candidates (within the same area_group):
`n_inliers` → `scale_consistency` (closer to 1.0 wins) → `road_name_agreement`.

## External dependencies (offline)

- **MINIMA** — LoFTR-based map-to-tile matcher. Clone into `MINIMA/`;
  weights in `MINIMA/weights/`.
- **SAM3 + LoRA** — Facebook SAM3 base weights auto-downloaded from
  HuggingFace on first run (`HF_TOKEN`). The fine-tuned LoRA adapter is
  shipped per fold in `models/sam3_lora/fold_*/` as PEFT-format
  `adapter_config.json` + `adapter_model.safetensors` (~76 MB / fold).
- **OS OpenData** — `OS_Open_Zoomstack.gpkg`, BoundaryLine, OpenNames,
  OpenMapLocal, Code-Point Open. All OGL v3, no API key, no rate limit.
  Placed under `os_opendata/`. Setup instructions live in the relevant
  geocoder modules.

## Requirements

- Python 3.10+ (managed via `uv`)
- macOS with MPS or Linux with CUDA for GPU acceleration
- ~10 GB disk: SAM3 base weights (~3 GB) + LoRA + rotation
  adapters (~830 MB) + OS OpenData (~5 GB) + tile cache (~200 GB at
  full benchmark scale, but cached lazily)
