# ablations/

Paper-ablation harnesses that run **outside** the main agent pipeline.
Everything is driven through one entry point:

```bash
uv run ablations/run.py <subcommand> [flags...]
uv run ablations/run.py <subcommand> -h     # full flags per harness
```

Each harness is self-contained, evaluates against the same ground-truth
assets as the main benchmark, supports `--max-cases N` for a cheap
smoke run, and is safe to run in parallel with a live benchmark (no
shared mutable state in `results/` or `models/`).

## Reproducing the published rows

| Paper row | Command |
|---|---|
| Table 1, VLM end-to-end (per model) | `uv run ablations/run.py vlm-e2e --vlm-model <gemini-flash\|gemini-pro\|claude-opus\|gpt-5.5-pro>` |
| Table 1, Collapsed Reader | `uv run ablations/run.py collapsed-reader --model gemini-flash --enable-critic --output-dir ablations/no_reader` |
| Segmentation table, VLM-direct rows | `uv run ablations/run.py vlm-seg --model <gemini-flash\|gemini-pro>` |
| Segmentation table, vanilla-SAM3 row + prompt-sweep appendix | `uv run ablations/run.py sam-prompts` |
| Locate table, place-only (production) | `uv run ablations/run.py locate --config production` |
| Locate table, all 6 geocoder tools | `uv run ablations/run.py locate --config all_tools` |
| Locate table, VLM-direct | `uv run ablations/run.py locate-vlm --vlm-model gemini-flash` |

Aggregation back into paper numbers is `scripts/reproduce_paper.py`
(offline — reads the cached per-case outputs, never calls an API).

## Subcommands

| Subcommand | Module | Purpose |
|---|---|---|
| `vlm-e2e` | [`vlm_e2e_pdf_to_geojson.py`](vlm_e2e_pdf_to_geojson.py) | Single-shot VLM PDF → GeoJSON with a strict pydantic schema; scored with the production `calculate_spatial_metrics`, so VLM and pipeline IoUs are byte-identical on the same input. |
| `vlm-seg` | [`vlm_segmentation.py`](vlm_segmentation.py) | VLM-direct boundary segmentation, pixel IoU vs the 211 hand-annotated masks. |
| `sam-prompts` | [`sam_base_prompt_search.py`](sam_base_prompt_search.py) | Base SAM3 (no LoRA) with five candidate text prompts on the same 211 masks. |
| `locate` | [`locate_only_eval.py`](locate_only_eval.py) | Calls the locate sub-agent once per case (no worker / MINIMA / SAM3) and scores its pick against the nearest GT polygon-part centroid. `--config {production,all_tools}` selects the place-only or six-geocoder agent. |
| `locate-vlm` | [`locate_vlm_direct.py`](locate_vlm_direct.py) | Single-shot VLM-direct geocoder (no tools): sends the PDF, asks for one (lat, lon). |
| `collapsed-reader` | `benchmark_runner.py --no-reader` | Folded ablation — the worker fills `PDFInfo` itself via a first tool call instead of a dedicated Reader phase. |
| `build-subset` | [`build_vlm_e2e_subset.py`](build_vlm_e2e_subset.py) | Deterministic (seed 42) stratified 40-case subset over document-quality × shape-complexity. Offline. |
| `reader-cache` | [`extract_pdf_info_cache.py`](extract_pdf_info_cache.py) | Freezes per-case `PDFInfo` from an existing benchmark run into `cached_pdf_info_for_locate_ablations.json` (no reader execution, offline), so every locate variant reuses identical reader output. |
| `audit-locate` | [`audit_locate_results.py`](audit_locate_results.py) | Post-hoc audit of locate trajectories: flags LA-centroid emergency fallbacks and picks > 5 km from every tool return. Drives [`locate_only_eval/AUDIT.md`](locate_only_eval/AUDIT.md). Offline. |

[`_shared.py`](_shared.py) holds the GT-centroid extraction and
nearest-part scoring shared by `locate` and `locate-vlm`, so the metric
is byte-identical across both harnesses.

## Checked-in artifacts

| Path | Produced by | Contents |
|---|---|---|
| `locate_only_eval/<config>/` | `locate`, `locate-vlm` | `locate_picks.csv` per config (`min_1_tool`, `full`, `vlm_direct_*`). |
| `no_reader/<model>/` | `collapsed-reader` | Per-case outputs for the Collapsed Reader row. |
| `vlm_e2e_pdf_to_geojson/` | `build-subset` + `vlm-e2e` | `subset_40.json`, pipeline baselines, summaries, and per-model `results.csv` / `pred_geojsons/` / `trajectories/`. |
| `cached_pdf_info_for_locate_ablations.json` | `reader-cache` | Frozen reader output shared by all locate configs. |
| `prompts/` | `vlm-* --dump-prompt` | Verbatim VLM-prompt snapshots. (The locate prompts are frozen constants in `geoplanagent/prompts.py`.) |

## Conventions for adding a new ablation

- One self-contained harness module, registered as a subcommand in
  [`run.py`](run.py).
- Load the same ground truth and report the same summary shape as the
  existing harness for that stage, so the row drops into the paper
  table without re-aggregation.
- Module docstring with usage examples, model defaults, and a
  `--dump-prompt` / `--max-cases` path for cheap pre-run verification.
