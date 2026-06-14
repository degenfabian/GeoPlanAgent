# GeoPlanAgent

Autonomous extraction of the application-site boundary from UK planning
permission PDFs. An LLM agent reads each PDF, geocodes a likely centre
via an offline OS Open Names sub-agent, positions the planning map
against Ordnance Survey OpenData tiles using MINIMA-LoFTR feature
matching, segments the drawn boundary with a fine-tuned SAM3 model,
and projects the result to a WGS84 GeoJSON polygon. Reference baseline
for the [Plan2Map benchmark](https://arxiv.org/abs/2606.02747).

## Pipeline

Two LLM phases plus an inline locate sub-agent, with an optional third
LLM critic phase. The pipeline always emits a polygon; downstream
scoring is IoU against ground-truth GeoJSON.

```
                Phase 1                Phase 2 — Worker agent loop
                Reader                 (4 tools)
                                       ─────────────────────────────
 PDF ────►  PDFInfo (JSON)  ────►  propose_centers ──┐
            • site_address          (invokes locate   │
            • postcodes              sub-agent —      │
            • grid_refs              OS Open Names    │
            • map_pages              by default)      │
              (one entry per                          ▼
              area_group)       match_at(page=N) ──► commit_match
            • district info       (MINIMA + SAM3        (commits ONE
            • text & visual       on the supplied       candidate for
              cues for locate     page; one             its area_group;
                                  area_group per        unions running
                                  call)                 result. Loop
                                                        once per group
                                                        on multi-area
                                                        documents.)
                                                        │
                                                        ▼
                                       BoundaryOutcome → final GeoJSON
                                                        │
                                       (optional) LLM critic — pairwise
                                       review across stored candidates;
                                       may approve / switch / retry_locate
                                       (--enable-critic; default off)
```

Documents that cover an entire administrative area (Article 4 directions,
borough-wide conservation, etc.) take a shortcut: `lookup_district`
returns the OS BoundaryLine polygon directly and the worker submits
`status="district_lookup"` — no MINIMA, no SAM.

## Project structure

```
GeoMapAgent_autonomous/
├── README.md
├── benchmark_runner.py        # Evaluation driver across the dataset
├── pyproject.toml             # Dependencies (uv-managed)
├── uv.lock
│
├── geoplanagent/              # Core pipeline (see geoplanagent/README.md)
│   ├── run.py                 # run_agent: reader → worker loop → critic
│   ├── agents/                # reader.py, worker.py, locate.py, critic.py
│   ├── tools/                 # positioning, matching, geocode, segment, tiles, pdf
│   ├── prompts.py             # every system prompt
│   ├── schemas.py             # pydantic contracts (LLM-visible)
│   ├── utils.py               # AgentState, model aliases, retry, geodesy, folds
│   └── metrics.py             # IoU/centroid scoring + viz
│
├── ablations/                 # Paper ablation scripts (see ablations/README.md)
├── training/                  # SAM3 LoRA + rotation classifier (see training/README.md)
├── scripts/                   # reproduce_paper.py, compute_costs.py, utilities
├── docs/                      # GitHub Pages demo site (see docs/README.md)
├── LICENSE
│
├── MINIMA/                    # LoFTR matcher (external, gitignored)
├── evaluation_data/           # Test dataset (PDFs + GT GeoJSON, gitignored)
├── boundary_annotations/      # Per-case annotated map + mask (tracked)
├── models/                    # Model weights (gitignored, see models/README.md)
├── os_opendata/               # OS OpenData (Zoomstack, BoundaryLine,
│                              # OpenNames, OpenMapLocal, Code-Point Open)
│                              # — gitignored
├── cache/                     # Rendered tile-canvas cache (gitignored, disposable)
├── results/                   # Benchmark outputs (gitignored)
└── figures/                   # Paper figures (regenerable)
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

| Flag | Default | Purpose |
|---|---|---|
| `--model` | `gemini-flash` | Reader + worker model (alias or full OpenRouter ID); default is the paper configuration |
| `--locate-model` | `google/gemini-3-flash-preview` | Locate sub-agent model (separate from `--model`) |
| `--max-iterations` | `12` | Max worker turns per case |
| `--max-cases` | — | Cap dataset size (quick smoke test) |
| `--cases <ids…>` | — | Run only the listed case folder names |
| `--start-from` | `0` | Skip the first N cases |
| `--dpi` | `200` | PDF render DPI |
| `--output-dir` | `results/benchmark` | Result root (model subdir appended) |
| `--force` | off | Re-run even if `metrics.json` is cached |
| `--enable-critic` | off | Run the LLM critic after worker submit (snapshots paired no-critic / with-critic IoUs for ablation) |
| `--critic-max-iters` | `2` | Max critic-rejection iterations (only with `--enable-critic`) |
| `--no-reader` | off | Folded ablation: skip the dedicated reader phase; the worker calls `submit_pdf_info` as its first tool |

Results land in `results/<output-dir>/<model_id_with_slashes→underscores>/`.
The benchmark loads the case list from `evaluation_data/0_planning_dataset_list.xlsx`
and auto-injects any `*_merged` folders not in the spreadsheet.

### Model aliases

The aliases listed below are the only ones resolved in
[`geoplanagent/utils.py`](geoplanagent/utils.py); any other string is
treated as a full OpenRouter ID and passed through unchanged. So
`--model openai/gpt-4o-mini` works directly; `--model claude-sonnet`
would be sent to OpenRouter literally and fail.

| Alias | OpenRouter model ID |
|---|---|
| `gemini-pro` | `google/gemini-3.1-pro-preview` |
| `gemini-flash` | `google/gemini-3-flash-preview` |
| `claude-opus` | `anthropic/claude-opus-4.7` |
| `gpt-5.5-pro` | `openai/gpt-5.5-pro` |

### Programmatic single-case usage

```python
from geoplanagent.run import run_agent
from geoplanagent.tools.segment import load_sam3_ft
from geoplanagent.tools.matching import load_minima

models = {"sam3_ft": load_sam3_ft(), "minima": load_minima()}

result = run_agent(
    pdf_path="evaluation_data/12:00116:ART4/document.pdf",
    models_state=models,
    model_name="gemini-flash",
    max_iterations=12,
    case_name="12:00116:ART4",   # for SAM3 + rotation k-fold routing
)

if result["geojson"]:
    print(f"Inliers: {result['match_info'].get('n_inliers')}")
    # result['geojson'] is a Feature dict with Polygon or MultiPolygon geometry
```

`run_agent` also accepts `enable_critic=True`, `critic_max_iters=2`,
`locate_model="..."`, and `folded=True` (folded ablation).

## Phase details

### Phase 1 — Reader

One-shot pydantic-ai call (`output_type=PDFInfo`) over the raw PDF
binary — no OCR pipeline, the VLM reads the PDF directly. Populates a
schema covering: site address, postcodes, grid references, scale,
ranked map pages with per-page area-group / boundary-clarity /
detail-level metadata, district info, and locate-stage signals
(road names, place names, parishes, admin region, visible map
labels). Schema lives in [`geoplanagent/schemas.py`](geoplanagent/schemas.py);
prompt in [`geoplanagent/prompts.py`](geoplanagent/prompts.py).

In the `--no-reader` folded variant this phase is skipped: the PDF
binary is attached to the worker's first user message and the worker
is required to call `submit_pdf_info(info=<PDFInfo>)` before any
positioning tool. The downstream state is identical.

### Phase 2 — Worker

Tool-calling pydantic-ai agent. Four worker tools:

1. **`propose_centers(extra_terms?, match_context?)`** — invokes the
   locate sub-agent (see [`geoplanagent/agents/locate.py`](geoplanagent/agents/locate.py)).
   The sub-agent reads pdf_info + the rendered map image and returns
   ONE `LocatePick` (lat, lon, σ, confidence, source, evidence).
   **In production the sub-agent ships with a single geocoder tool —
   `place` (OS Open Names).** Five additional offline geocoders
   (`postcode`, `grid_ref`, `road`, `intersect`, `la_check`) are
   implemented and used only by the locate ablation's all-tools agent
   (`ablations/locate_only_eval.py --config all_tools`).
2. **`match_at(page=N, name, lat, lon, sigma_m?, scale_ratio?)`** —
   runs MINIMA on ONE page (one `area_group`) at the supplied centre.
   Returns one candidate with `n_inliers`, `scale_consistency`,
   `road_name_agreement`, `road_name_verdict`, `area_group`, `page`,
   `candidate_id`, `budget_remaining`, and the list of already-
   committed groups. SAM3 segmentation runs automatically on first
   need per page and is cached. Budget: 5 `match_at` calls per case.
3. **`commit_match(candidate_id)`** — commits ONE candidate (and
   therefore one `area_group`). The polygon was already projected
   inside `match_at`; `commit_match` writes (or replaces) that
   group's slot in the running result and unions all committed groups
   into the final GeoJSON. A strict gate rejects commits where MINIMA
   produced no valid affine.
4. **`lookup_district(district_name)`** — OS BoundaryLine offline
   lookup for documents whose boundary IS an entire admin region
   (Article 4 directions, conservation-area-wide planning, etc.). On
   success the worker submits `status="district_lookup"` and the
   polygon comes straight from BoundaryLine — no SAM, no positioning.
   Accepts `|`-separated alternates (`"Westminster, UK | City of Westminster, UK"`).

In `--no-reader` mode the worker additionally has `submit_pdf_info` as
its required first tool call; otherwise that tool is hidden from the
LLM via a `prepare` callback.

For multi-area documents the worker iterates the
`propose_centers → match_at → commit_match` loop once per
`area_group`; the final GeoJSON is the shapely-union of every
committed group's polygon. Most documents are single-area, in which
case the loop runs exactly once.

The worker output (`BoundaryOutcome`, status ∈ {`accepted`,
`district_lookup`}) is validated by an `output_validator` that
re-reads tool-call state — the worker can't report flags it didn't
actually set, and `final_n_inliers` / `rotation_checked` are auto-
corrected to the truth from state.

### Phase 3 (optional) — Independent LLM critic

Opt-in via `--enable-critic` (or `enable_critic=True` to `run_agent`).
After the worker submits, a separate LLM call sees the visual panels
for the top-3 (by `n_inliers`) stored match candidates plus every
committed candidate, plus per-candidate metrics (`n_inliers`,
`scale_consistency`, `road_name_agreement`), and emits a
`CriticDirective` with `action ∈ {approve, switch, retry_locate}`. A
`switch` is applied directly in Python (no worker re-invoke — the
critic has already chosen the id, an LLM round-trip just to type
`commit_match(N)` would be wasted cost). A `retry_locate` does
re-invoke the worker with a templated instruction.

The worker's first-committed polygon is deep-copied before the loop
fires, so a single run with `--enable-critic` produces paired
no-critic / with-critic IoUs (saved as `worker_first_iou` in
`metrics.json`). Implementation in [`geoplanagent/agents/critic.py`](geoplanagent/agents/critic.py).
Max 2 rejection iterations per case by default.

## Output layout (per case)

```
results/<output-dir>/<model>/<case>/
├── predicted.geojson                   # Final boundary (Feature, Polygon | MultiPolygon)
├── predicted_worker_first.geojson      # (with --enable-critic) pre-critic snapshot
├── metrics.json                        # IoU, precision, recall, centroid_distance_m,
│                                       # match_info, agent_stats, processing_time,
│                                       # worker_first_{iou,metrics} when critic ran
├── message_log.json                    # Full worker conversation trace (binary parts summarised)
├── pdf_info.json                       # Phase 1 / submit_pdf_info structured extraction
├── boundary_mask.png                   # Binary SAM3 mask on the committed primary page
├── affine_H.npy                        # 2×3 committed affine (page → tile pixel)
├── tile_info.json                      # zoom / tx_min / ty_min / nx / ny / tile_size
├── selected_boundary.png               # SAM-overlay (50/50) on the committed page
├── viz_comparison.png                  # Predicted vs GT (geopandas + contextily)
├── critic_panel_iter0.png …            # (with --enable-critic) stacked top-N panel per iter
├── critic_panel_iter0_cand<id>.png …   # (with --enable-critic) per-candidate panels sent to the LLM
└── partial_state.json                  # (on a mid-run crash) snapshot of state for debugging
```

A run-level `summary.json` lands at `results/<output-dir>/<model>/summary.json`
with aggregate IoU stats (production-honest — no-polygon cases count
as IoU 0 — and polygon-only) plus the full per-case rows. The
benchmark detects a cache-mode mismatch (`worker_first_iou` present
implies the cached run had `--enable-critic`; mismatched calls force
re-run rather than silently mixing modes).

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

Tie-break order across candidates (within the same `area_group`):
`n_inliers` → `scale_consistency` (closer to 1.0 wins) → `road_name_agreement`.
Scoring formulas live in [`geoplanagent/tools/matching.py`](geoplanagent/tools/matching.py).

## Headline numbers (paper, Gemini 3 Flash)

| Stage | N | Metric | Value |
|---|---|---|---|
| Full pipeline | 208 | mean IoU | 0.736 |
| Full pipeline | 208 | median IoU | 0.904 |
| Full pipeline | 208 | cases with IoU > 0 | 89.4% |
| Full pipeline | 208 | cases with IoU ≥ 0.8 | 67.8% |
| Full pipeline | 208 | median centroid error | 4.6 m |
| Full pipeline | 208 | Acc@0.1D | 78.8% |
| Full pipeline | 208 | mean cost / doc | $0.043 |
| Full pipeline | 208 | mean wall-clock / case | 153 s |
| + Critic | 208 | mean IoU | 0.740 |
| SAM3-LoRA only | 208 | mean pixel IoU (5-fold OOF, case-level) | 0.912 |
| Rotation classifier (TTA) | 208 | 5-fold mean top-1 acc (case-level) | 0.981 |

The full-pipeline rows are pre-critic (the paper's main row); the cost
includes the locate sub-agent's LLM calls, which are roughly half of it
(reader + worker alone come to about $0.020/doc — see
`scripts/compute_costs.py`).

VLM-direct PDF-to-GeoJSON on the strongest of four models
(Gemini-3.1-Pro, 40-case stratified subset): mean IoU 0.112.
See [the paper](https://arxiv.org/abs/2606.02747) for the full table
and the four-model breakdown.

## Reproducing the paper

Every number in the paper is recomputed from the cached per-case run
artifacts (`metrics.json`, `results.csv`), which are tracked in this
repo under `results/`, `ablations/` and `training/eval/` — no API calls
needed:

```bash
uv run scripts/reproduce_paper.py all          # everything
uv run scripts/reproduce_paper.py table1 fig3  # or individual sections
```

Sections: `table1` `table2` `table4` `table9` `table11` `table12`
`fig3` `fig4` `costs` `dataset`. Each line prints the recomputed value
next to the value reported in the paper.

From a bare clone (without `evaluation_data/`), the sections that need
only run artifacts work immediately: `table2` `table4` `table9`
`table11` `table12` `fig3` `costs`. The remaining sections (`table1`,
`fig4`, `dataset`) additionally read the ground-truth GeoJSONs and the
metadata spreadsheet from `evaluation_data/` — place the dataset
release there first.

To re-run the underlying experiments rather than re-aggregate them
(these call OpenRouter and cost API credits), the main benchmark is
`benchmark_runner.py` and every ablation goes through one entry point,
`ablations/run.py` — see [`ablations/README.md`](ablations/README.md)
for the exact command behind each published row:

| What | Command |
|---|---|
| Main benchmark (Table 1) | `uv run benchmark_runner.py --model gemini-flash --enable-critic --output-dir results/<name>` |
| Collapsed-reader ablation | `uv run ablations/run.py collapsed-reader --model gemini-flash --output-dir ablations/no_reader` |
| Locate-stage ablations (Table 2) | `uv run ablations/run.py locate …` / `uv run ablations/run.py locate-vlm …` |
| VLM end-to-end baselines | `uv run ablations/run.py vlm-e2e --vlm-model <alias>` |
| VLM-direct segmentation | `uv run ablations/run.py vlm-seg --model <alias>` |
| Vanilla-SAM prompt sweep | `uv run ablations/run.py sam-prompts` |
| SAM3-LoRA / rotation k-fold eval (offline) | `uv run training/eval/eval_sam_kfold.py` / `eval_rotation_kfold.py [--tta]` |
| Cost decomposition (offline) | `uv run scripts/compute_costs.py results/cost_audit_v1` |
| Paper figures (offline) | `uv run figures/make_section5_figures.py` |

The test suite is offline and instant: `uv run pytest`.

## External dependencies (offline)

- **MINIMA** — LoFTR-based map-to-tile matcher. Clone
  [LSXI7/MINIMA](https://github.com/LSXI7/MINIMA) into `MINIMA/` and
  place its released LoFTR checkpoint at
  `MINIMA/weights/minima_loftr.ckpt` (download link in the MINIMA
  README). Apache-2.0, not vendored here.
- **SAM3 + LoRA** — Facebook SAM3 base weights auto-downloaded from
  HuggingFace on first run (`HF_TOKEN`). The fine-tuned LoRA adapters
  ship per fold in `models/sam3_lora/fold_*/` as PEFT-format
  `adapter_config.json` + `adapter_model.safetensors` (~76 MB / fold).
- **OS OpenData** — `OS_Open_Zoomstack.gpkg`, BoundaryLine, OpenNames,
  OpenMapLocal, Code-Point Open. All OGL v3, no API key, no rate limit.
  Placed under `os_opendata/`. Per-dataset setup instructions live in
  the relevant geocoder docstrings (see [`geoplanagent/README.md`](geoplanagent/README.md)).

## Requirements

- Python 3.10+ (managed via `uv`)
- macOS with MPS or Linux with CUDA for GPU acceleration
- ~10 GB disk: SAM3 base weights (~3 GB) + LoRA + rotation
  adapters (~830 MB) + OS OpenData (~5 GB) + rendered tile-canvas
  cache (`cache/`, lazily populated; ≈250 MB after a full benchmark
  run, safe to delete at any time)

## Data

`evaluation_data/` holds the 208-case Plan2Map benchmark: one folder
per case containing the planning PDF and the ground-truth GeoJSON,
plus two spreadsheets — `0_planning_dataset_list.xlsx` (the case list
the benchmark runner loads) and `new_updated.xlsx` (the full metadata
table behind the dataset statistics and figures). Ground-truth
polygons originate from planning.data.gov.uk and the source PDFs from
the issuing local planning authorities; see the paper's dataset
section for the release. `boundary_annotations/` and
`training/dataset/` hold the 211 hand-annotated map-page masks used to
fine-tune and evaluate SAM3-LoRA.

## License

Code is MIT-licensed (see [LICENSE](LICENSE)). Ordnance Survey
OpenData and planning.data.gov.uk ground truth are used under the
[Open Government Licence v3](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
