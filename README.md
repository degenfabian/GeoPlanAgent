# GeoMapAgent

Autonomous planning-boundary extraction from UK planning-document PDFs. An LLM agent reads each PDF, identifies the site map, geocodes locations, positions the map against Ordnance Survey tiles via learned feature matching (MINIMA), extracts the boundary with SAM3, projects it to WGS84 GeoJSON, and — in Phase 3 — has a separate VLM critic review the result and either approve it, apply a deterministic code fix (re-run SAM, post-process the mask, or rotate + re-match), or re-enter the worker agent with targeted feedback.

## Pipeline

```
                 Phase 1                Phase 2                       Phase 3
                 (Reader)               (Worker agent, 9 tools)       (Critic agent)
 PDF ─────────>  structured JSON  ──>   geocode + MINIMA + SAM3 ──>   VLM review
                 (site addr,             positioning + mask             approve /
                  postcodes,             extraction + projection        code fix /
                  scale, rotation)                                      worker re-entry /
                                                                        flag low-confidence
                                                                              │
                                                                              v
                                                                        final GeoJSON
                                                                        (always emitted)
```

The LLM acts as the reasoning engine, making tool calls to orchestrate the pipeline. The reader (Phase 1) is a one-shot structured extraction. The worker (Phase 2) iterates tools until it submits a validated `BoundaryOutcome`. The critic (Phase 3) provides independent visual verification.

## Project Structure

```
GeoMapAgent_autonomous/
├── README.md                   # This file
├── benchmark_runner.py         # Evaluation driver across the dataset
├── check_credits.py            # OpenRouter credit check utility
├── pyproject.toml              # Dependencies (uv-managed)
├── uv.lock
├── .env.template
│
├── tools/                      # Core pipeline modules (see tools/README.md)
│   ├── agent.py                # Reader + Worker agents, 9 tools
│   ├── critic.py               # Phase 3 Commenter VLM critic loop
│   ├── positioning.py          # MINIMA sliding-window matching
│   ├── sam3_boundary.py        # SAM3 boundary segmentation
│   ├── geocoding.py            # Multi-source geocoding
│   ├── geo_tools.py            # OS grid refs, district lookup
│   ├── os_opendata_tiles.py    # Offline OS tile rendering
│   ├── pdf_tools.py            # PDF rendering
│   ├── geojson_metrics.py      # IoU / precision / recall / F1
│   └── visualization_tools.py  # Boundary visualisation helpers
│
│
├── training/                   # SAM3 LoRA training pipeline (see training/README.md)
│
├── evaluation_data/            # Test dataset (PDFs + GT GeoJSON) — gitignored
├── boundary_annotation_dataset/ # SAM3 training set — gitignored
├── road_annotation_dataset/    # Road-extractor training set — gitignored
├── MINIMA/                     # LoFTR matcher (external, gitignored)
├── models/                     # Model weights — gitignored
├── os_opendata/                # OS Zoomstack GeoPackage — gitignored
└── results/                    # Benchmark outputs — gitignored
```

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # Install uv if needed
uv sync                                           # Install dependencies
uv sync --extra training                          # Optional: training scripts
```

## Configuration

```bash
cp .env.template .env
```

| Variable | Required | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | LLM API via OpenRouter |
| `HF_TOKEN` | Yes | HuggingFace — SAM3 model download |

## Usage

### Benchmark against the eval dataset

```bash
# Default: Gemini Pro, all cases, critic enabled
uv run benchmark_runner.py

# Specific model + case set
uv run benchmark_runner.py --model gemini-flash --max-cases 10

# Run named cases only
uv run benchmark_runner.py --cases 12:00116:ART4 A4Ba1

# Re-run previously failing cases first
uv run benchmark_runner.py --hard-first --prev-results results/benchmark_v4_flash/gemini-flash
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
from tools.sam3_boundary import load_sam3_ft
from tools.positioning import load_minima

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

## Phase Details

### Phase 1 — Reader

One-shot PDF read. The reader populates a `PDFInfo` schema: site address, postcodes, grid refs, scale, boundary colour, map rotation, map page numbers, whether the boundary is district-wide. This JSON summary — not the full PDF — is passed to the worker, so multi-turn conversations stay cheap.

### Phase 2 — Worker

Nine tools orchestrated by the LLM:

1. `render_page` — render a PDF page as a BGR image.
2. `geocode` — look up a postcode or OS grid ref the reader missed.
3. `position_boundary` — MINIMA sliding-window match against OS OpenData tiles. Auto-geocodes from five sources (postcodes.io, OS Zoomstack gazetteer, Wikidata, two Nominatim paths). Returns `n_inliers` and a `centers_summary` showing which source won. Accepts `skip_sources` for retries.
4. `extract_boundary` — SAM3 segmentation with semantic or instance mode.
5. `project_boundary` — affine-project the mask to a GeoJSON polygon.
6. `accumulate_boundary` — save the current page and reset state for multi-page maps.
7. `verify_position` — fetch OS tiles at the matched centre and draw the polygon for visual inspection.
8. `lookup_district` — pull an OSM administrative boundary for district-wide cases.
9. `visualize` — render debug overlays.

An `output_validator` enforces preconditions on the final `BoundaryOutcome`: borderline positioning (25–100 inliers) must be visually verified with explanatory notes, multi-page documents must accumulate every page except the last, and so on. Violations trigger `ModelRetry` and the agent has to resubmit.

Positioning quality gate: results with `n_inliers < 25 AND score < 15` are flagged `LOW_QUALITY` but the GeoJSON is **kept**. Partial overlap beats no prediction.

### Phase 3 — Commenter critic

An independent VLM agent runs after the worker submits `accepted`. It sees a composite image — planning map + SAM mask on the left, OS tiles + projected polygon on the right — plus a context block with inlier counts, which geocoding sources produced which centres, and the worker's reasoning.

Decisions:

- `approve` — proceed.
- `retry_sam` — re-run SAM3 with a new query or candidate selection, re-project. No MINIMA re-run.
- `retry_projection` — apply morphological hole-fill or thin-mask dilation, re-project.
- `retry_rotation` — rotate the map 90 / 180 / 270°, re-SAM, re-MINIMA with the existing centres, re-project. No LLM involvement.
- `retry_in_worker` — re-enter the worker agent with `message_history` replay, passing the critic's feedback (e.g. `worker_should_skip_sources=["wikidata"]`) as a new user message.
- `flag_low_confidence` — keep the GeoJSON, label `CRITIC_LOW_CONFIDENCE` in `accept_reason`.

After each successful code fix the critic re-runs on the post-fix state and picks a fresh decision. If a code fix **hard-fails** — SAM3 returns no candidates, MINIMA crashes, projection returns `None`, or rotation gets bad args — the runtime detects the `_failed` / `_no_candidates` / `_invalid` suffix on the fix string and short-circuits the inner loop straight to worker re-entry, because re-running the critic on identical state would pick the same doomed fix. Soft failures (`_noop`, `_no_mask`) stay in the inner loop for another critic pass.

Budget: up to 2 inner critic iterations and 1 worker re-entry per case. Multi-page and district-lookup cases skip the critic for now.

The critic **never nullifies** the GeoJSON. Its strongest negative decision only affects the `accept_reason` label.

## Output Layout

```
results/<run>/<model>/<case>/
├── predicted.geojson          # Extracted boundary (always present if one was produced)
├── metrics.json               # IoU, precision, recall, F1, agent_stats
├── message_log.json           # Full worker agent conversation trace
├── pdf_info.json              # Phase 1 structured extraction
├── boundary_mask.png          # Final binary mask
├── affine_H.npy               # 2x3 affine matrix (for offline re-projection)
├── tile_info.json             # Tile grid metadata matching affine_H
├── candidate_*.png            # Per-candidate SAM overlays
├── selected_boundary.png      # Final selected overlay
├── selected_indices.json      # Which candidates were combined
├── critic_log.json            # Phase 3: all iterations, decisions, fixes applied
├── critic_panel.png           # The composite image the critic reasoned over
├── centers_tried.json         # Geocoding sources: which fired, which won
└── viz_comparison.png         # Predicted vs ground truth overlay
```

## Positioning Quality Signals

| `n_inliers` | Meaning |
|---|---|
| ≥ 100 | Strong match |
| 50–100 | Decent; worker must call `verify_position` and fill `visual_check_notes` |
| 25–50 | Borderline; often wrong location |
| < 25 AND `score < 15` | Quality gate fires: labelled `LOW_QUALITY`, GeoJSON kept |

## External Dependencies

- **MINIMA** — LoFTR-based map-to-tile matcher. Clone separately into `MINIMA/`; weights in `MINIMA/weights/`.
- **SAM3** — Facebook's Segment Anything 3 from HuggingFace. Auto-downloaded on first run (`HF_TOKEN` required).
- **OS OpenData Zoomstack** — Free OGL-licensed GeoPackage, downloaded into `os_opendata/`. No API key needed.

## Requirements

- Python 3.10+
- macOS (MPS) or Linux (CUDA) for GPU acceleration
- ~8 GB disk for model weights