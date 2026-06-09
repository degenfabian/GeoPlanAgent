# ablations/

Paper-ablation scripts that run **outside** the main agent pipeline.
Each ablation is a self-contained driver that loads what it needs,
evaluates against the same ground-truth assets the main benchmark
uses, and reports metrics in a shape that drops into the paper table.

All ablations are safe to run in parallel with a live benchmark — they
do not mutate shared state in `results/` or `models/`.

## Scripts

| File | Stage isolated | Purpose |
|---|---|---|
| [`vlm_segmentation.py`](vlm_segmentation.py) | segmentation | VLM-direct boundary segmentation (Gemini/Claude/etc. via OpenRouter). Pixel IoU vs `training/dataset/boundary_masks/`, same shape as `training/eval/eval_sam_kfold.py`. Goes into the segmentation comparison (`tab:abl-finetune` / Figure 3). |
| [`sam_base_prompt_search.py`](sam_base_prompt_search.py) | segmentation | Base SAM3 (no LoRA) with five candidate text prompts, scored against the 211 hand-annotated masks. Goes into the prompt-sweep appendix table. |
| [`vlm_e2e_pdf_to_geojson.py`](vlm_e2e_pdf_to_geojson.py) | end-to-end | Single-shot VLM PDF → GeoJSON (strict pydantic schema). Goes into the VLM end-to-end rows of Table 1 (`tab:main-result`) + the four-model breakdown (`tab:vlm-models`). Subset built by `build_vlm_e2e_subset.py`. |
| [`build_vlm_e2e_subset.py`](build_vlm_e2e_subset.py) | — | Builds the stratified 40-case subset for the VLM-direct ablation from `evaluation_data/new_updated.xlsx`. Deterministic (seed 42). |
| [`locate_only_eval.py`](locate_only_eval.py) | locate | Calls the locate sub-agent once per case (no worker / MINIMA / SAM3 / commit / critic) and scores its pick against the nearest GT polygon-part centroid (haversine km). Use `--disabled-tools <tools>` for the LOO variants; `--dump-prompts` writes the seven prompt variants to disk without LLM calls. Goes into the locate table (`tab:abl-locate`) along with `locate_vlm_direct.py`. |
| [`locate_vlm_direct.py`](locate_vlm_direct.py) | locate | Single-shot VLM-direct geocoder (no tools) — sends the PDF and asks for one (lat, lon). Comparator row in the locate table. |
| [`audit_locate_results.py`](audit_locate_results.py) | locate | Post-hoc audit of the LOO outputs: flags cases that fell back to the LA-centroid emergency or whose pick is > 5 km from every tool return. Drives [`locate_only_eval/AUDIT.md`](locate_only_eval/AUDIT.md). |
| [`extract_pdf_info_cache.py`](extract_pdf_info_cache.py) | reader | Runs the reader phase once per case and caches the PDFInfo JSON to `cached_pdf_info_for_locate_ablations.json`. Lets every LOO variant reuse the same reader output, isolating locate-side variation from reader-side noise. |
| [`repopulate_pdf_info_cache.py`](repopulate_pdf_info_cache.py) | reader | One-shot helper to re-extract the cache after a schema change. |
| [`rerun_reader_only.py`](rerun_reader_only.py) | reader | Targeted reader-rerun helper for cases that failed reader validation in earlier runs; writes into `reader_rerun_post_fix/`. |
| [`diff_reader_output.py`](diff_reader_output.py) | reader | Compares two reader-output snapshots, writing per-field diffs to `reader_diff_report.json`. |
| [`_shared.py`](_shared.py) | — | GT-centroid extraction + nearest-part scoring (`gt_part_centroids`, `nearest_part_err_km`). Shared by `locate_only_eval.py` and `locate_vlm_direct.py` so the metric is byte-identical across harnesses. |

## Output subdirs

| Subdir | Produced by | Contents |
|---|---|---|
| `locate_only_eval/` | `locate_only_eval.py` | One subdir per config (`full`, `min_1_tool`, `no_<tool>`): `locate_picks.csv`, `run.log`, `trajectories/<case>.json`. `AUDIT.md` is the post-hoc summary across configs. |
| `locate_iou_subset/` | manual | 11-case regression test that re-runs the full pipeline (not just locate) with `--locate-disabled-tools` to verify the place-only kit doesn't break IoU on cross-1km cases. See [`locate_iou_subset/README.md`](locate_iou_subset/README.md). |
| `no_reader/<model>/` | `benchmark_runner.py --no-reader --output-dir ablations/no_reader/` | Folded-ablation per-case outputs (collapses reader → worker). Goes into the "Collapsed Reader" row of Table 1. |
| `vlm_e2e_pdf_to_geojson/` | `build_vlm_e2e_subset.py` + `vlm_e2e_pdf_to_geojson.py` | `subset_40.json`, `subset_40_pipeline_baseline.json`, `subset_40_summary.md`, plus per-model subdirs with `results.csv`, `summary.json`, `pred_geojsons/`, `trajectories/`. Also has a `subset_208/` set. |
| `reader_rerun_post_fix/` | `rerun_reader_only.py` | Per-case re-runs of just the reader phase after a fix. |
| `prompts/` | `locate_only_eval.py --dump-prompts` and `vlm_*.py --dump-prompt` | Verbatim prompt snapshots committed to git so each ablation row in the paper has a reviewable source-of-truth prompt. |

The `run_*.sh` driver scripts alongside are run by hand; they just
loop the Python harnesses over configs.

## Conventions for adding a new ablation

- **Self-contained**: one script, no shared mutable state with the
  main pipeline.
- **Same evaluation surface**: load the same ground truth and report
  the same summary shape (`mean / median / ≥0.50 / ≥0.70 / ≥0.80 /
  ≥0.90` for segmentation; centroid-error km bands for locate; full
  spatial-metrics shape for end-to-end) so the row drops into the
  paper table without re-aggregation.
- **Self-documenting**: a module docstring at the top with `Usage`
  examples, the model defaults, what it's comparing against, and any
  `--dump-prompts` / `--dry-run` flag for pre-run verification.
- **Cost-safe**: support `--max-cases N` for smoke tests and a
  dry-run mode that writes the prompt to disk without an LLM call.
