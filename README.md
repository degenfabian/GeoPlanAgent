# GeoMapAgent

Autonomous planning-boundary extraction from UK planning-document PDFs. An LLM
agent reads each PDF, identifies the site map, geocodes locations, positions
the map against Ordnance Survey tiles via learned feature matching (MINIMA),
extracts the boundary with SAM3, projects it to WGS84 GeoJSON, and has a
separate VLM critic review the result and either approve it, apply a
deterministic code fix, or re-enter the worker agent with targeted feedback.

## Pipeline

```
                Phase 1               Phase 2                       Phase 3
                (Reader)              (Worker agent, 5 tools)       (Critic agent)
 PDF ─────────>  structured JSON ──>  geocode + MINIMA + SAM3 ──>   VLM review
                (site addr,           positioning + mask             approve /
                 postcodes,           extraction + projection        code fix /
                 scale, rotation)                                    worker re-entry /
                                                                     flag low-confidence
                                                                            │
                                                                            v
                                                                     final GeoJSON
                                                                     (always emitted)
```

## Project structure

```
GeoMapAgent_autonomous/
├── README.md
├── benchmark_runner.py        # Evaluation driver across the dataset
├── check_credits.py           # OpenRouter credit check utility
├── pyproject.toml             # Dependencies (uv-managed)
├── uv.lock
│
├── tools/                     # Core pipeline modules (see tools/__init__.py)
│   ├── agent/                 # PydanticAI orchestrator + tool impls
│   │   ├── __init__.py        #   run_agent() entry point
│   │   ├── state.py           #   shared AgentState
│   │   ├── schemas.py         #   Pydantic models for tool I/O
│   │   ├── prompts.py         #   reader / worker / critic prompts
│   │   ├── critic.py          #   Phase 3 critic loop
│   │   └── tools/             #   render / locate / match / extract / verify
│   ├── locate/                # propose_centers_v2 cascade + ranker
│   ├── matching/              # MINIMA sliding-window + road-name verifier
│   ├── extraction/            # sam3, boundary_color, mask_ops
│   ├── geocoding/             # code_point, os_names, dispatchers, positioning
│   ├── io/                    # pdf, os_tiles, rotation_classifier, map_crop,
│   │                          #   text_extraction
│   ├── metrics/               # geojson (IoU/F1), visualization, reward
│   ├── snap/                  # INSPIRE freehold-parcel snap
│   ├── scoring.py             # composite_window_score, commit_attempt_score
│   ├── candidates.py          # backwards-compat shim → tools.locate
│   ├── delaunay_filter.py     # optional Delaunay-consistency RANSAC filter
│   └── verification_checks.py # critic cross-checks (LA poly, scale, area)
│
├── scripts/run_benchmark.sh   # One-shot full-benchmark wrapper
├── MINIMA/                    # LoFTR matcher (external, gitignored)
│   └── third_party/LoFTR_minima/
├── evaluation_data/           # Test dataset (PDFs + GT GeoJSON) — gitignored
├── models/                    # Model weights — gitignored
├── os_opendata/               # OS Zoomstack GeoPackage — gitignored
└── results/                   # Benchmark outputs — gitignored
```

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # Install uv if needed
uv sync
```

## Configuration

```bash
cp .env.template .env
```

| Variable | Required | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | yes | LLM API via OpenRouter |
| `HF_TOKEN` | yes | HuggingFace — SAM3 model download |

## Usage

### Full benchmark

```bash
scripts/run_benchmark.sh                       # writes to results/benchmark
scripts/run_benchmark.sh results/my_run        # custom output dir
```

Equivalent to: `uv run benchmark_runner.py --model gemini-flash
--max-iterations 12 --output-dir results/benchmark --force`.

### Subsetting

```bash
uv run benchmark_runner.py --max-cases 10
uv run benchmark_runner.py --cases 12:00116:ART4 A4Ba1
uv run benchmark_runner.py --hard-first \
    --prev-results results/benchmark
```

### Model aliases

| Alias | OpenRouter model ID |
|---|---|
| `gemini-pro` | `google/gemini-3.1-pro-preview` |
| `gemini-flash` | `google/gemini-3-flash-preview` |
| `gemini-flash-lite` | `google/gemini-3.1-flash-lite-preview` |
| `claude-sonnet` | `anthropic/claude-sonnet-4-6` |
| `claude-opus` | `anthropic/claude-opus-4.6` |
| `gpt-5.4` | `openai/gpt-5.4` |

Any OpenRouter model ID is also accepted directly.

### Run a single case programmatically

```python
from tools.agent import run_agent
from tools.extraction.sam3 import load_sam3_ft
from tools.matching import load_minima

models = {"sam3_ft": load_sam3_ft(), "minima": load_minima()}

result = run_agent(
    pdf_path="evaluation_data/12:00116:ART4/document.pdf",
    models_state=models,
    model_name="gemini-flash",
    enable_critic=True,
)

if result["success"]:
    print(f"Inliers: {result['match_info']['n_inliers']}")
    print(f"Critic: {result['critic_final_decision']}")
    # result["geojson"] — Feature dict with MultiPolygon geometry
```

## Phase details

### Phase 1 — Reader

One-shot PDF read that populates `PDFInfo`: site address, postcodes, grid
refs, scale, boundary colour, map rotation, map page numbers, district-wide
flag. The summary — not the full PDF — is passed to the worker, so multi-turn
conversations stay cheap.

### Phase 2 — Worker

Five tools orchestrated by the LLM:

1. `render_page` — render a PDF page as a BGR image.
2. `propose_centers` — locate_v2 cascade: pulls candidate centres from
   postcode, grid_ref, parish/landmark/road geocodes constrained to the LA
   polygon, feature_cluster, and multi-road consensus.
3. `match_at` — MINIMA sliding-window match at one candidate centre. Returns
   `n_inliers`, `score`, and a composite reranker score.
4. `commit_match` — pick the best `match_at` to commit. Gated by a smart
   commit gate (inliers × inside-LA × distance-to-anchor) and a strict
   evidence floor (`n_inliers ≥ 18`, `mask_frac ≥ 0.002`).
5. `extract_boundary` — SAM3 segmentation in semantic mode, mask projection
   to a GeoJSON polygon, INSPIRE freehold-snap post-processing.

Output is a validated `BoundaryOutcome`; an `output_validator` enforces
preconditions on it (visual checks for borderline positions, etc.).

### Phase 3 — Commenter critic

An independent VLM agent runs after the worker submits `accepted`. It sees
a composite image (planning map + SAM mask on the left, OS tiles + projected
polygon on the right) plus context (inlier counts, which geocoders fired,
worker reasoning) and chooses:

- `approve` — proceed.
- `retry_sam` — re-run SAM3 with a new query/candidate, re-project.
- `retry_projection` — morphological hole-fill or thin-mask dilation.
- `retry_rotation` — rotate 90/180/270°, re-SAM, re-MINIMA at the existing
  centres, re-project.
- `retry_in_worker` — re-enter the worker with the critic's feedback as
  a new user message; supports `worker_should_skip_sources`.
- `flag_low_confidence` — keep the GeoJSON, label `CRITIC_LOW_CONFIDENCE`.

Budget: 2 inner critic iterations and 1 worker re-entry per case. The
critic never nullifies the GeoJSON.

## Output layout

```
results/<run>/<model>/<case>/
├── predicted.geojson         # Extracted boundary
├── metrics.json              # IoU, precision, recall, F1, agent_stats
├── message_log.json          # Full worker conversation trace
├── pdf_info.json             # Phase 1 structured extraction
├── boundary_mask.png         # Final binary mask
├── affine_H.npy              # 2×3 affine matrix
├── tile_info.json            # Tile grid metadata matching affine_H
├── candidate_*.png           # Per-candidate SAM overlays
├── selected_boundary.png     # Final selected overlay
├── critic_log.json           # Critic iterations + decisions
├── critic_panel.png          # Composite image the critic saw
├── centers_tried.json        # Which geocoders fired / won
└── viz_comparison.png        # Predicted vs ground truth overlay
```

## Positioning quality signals

| `n_inliers` | Meaning |
|---|---|
| ≥ 100 | Strong match |
| 50–100 | Decent; verify visually |
| 25–50 | Borderline; often wrong location |
| < 18 | Strict commit gate refuses to commit |

## External dependencies

- **MINIMA** — LoFTR-based map-to-tile matcher. Clone separately into
  `MINIMA/`; weights in `MINIMA/weights/`.
- **SAM3** — Facebook's Segment Anything 3 from HuggingFace. Auto-downloaded
  on first run (`HF_TOKEN` required).
- **OS OpenData Zoomstack** — Free OGL-licensed GeoPackage; place in
  `os_opendata/`. No API key needed.

## Requirements

- Python 3.10+
- macOS (MPS) or Linux (CUDA) for GPU acceleration
- ~8 GB disk for model weights
