# `benchmark_runner.py`

**676 lines.** The benchmark entry point. Reads the dataset manifest
(Excel sheet of all UK planning cases), filters to existing cases on
disk + non-training cases, optionally filters by user-specified case
list or hard-first ordering, then iterates and calls `run_agent` per
case. Writes per-case `metrics.json` + a final aggregate report.

```bash
uv run benchmark_runner.py \
  --model gemini-flash --max-iterations 12 \
  --output-dir results/benchmark_v13 --force
```

## Public API (CLI)

| Flag | Effect |
|---|---|
| `--model NAME` | OpenRouter model id (default `claude-sonnet`; user prefers `gemini-flash`) |
| `--max-iterations N` | Max agent turns per case (default 12) |
| `--output-dir DIR` | Where per-case results go (default `results/benchmark`) |
| `--cases A B C` | Run only the specified case folder names |
| `--max-cases N` | Limit to first N cases |
| `--start-from N` | Skip first N cases |
| `--dpi N` | Override render DPI (default 200) |
| `--force` | Overwrite existing case results |
| `--hard-first` | Sort cases by prior-run worst-IoU first |
| `--prev-results DIR` | Source for `--hard-first` IoUs |
| `--no-critic` | Skip Phase-3 VLM critic |
| `--include-training-cases` | Include all 27 training-set cases (use with k-fold inference) |

## High-level flow

```
1. Parse CLI args → settings
2. Load Excel manifest → case list
3. Filter:
   - cases that exist in eval_dir
   - inject *_merged folders (added during cleanup)
   - (unless --include-training-cases) drop training set
   - (if --cases) keep only specified
   - (if --hard-first) reorder by prior IoU
4. Build models once (SAM3+LoRA, MINIMA, verifier)
5. For each case:
   - Pre-flight: skip if metrics.json exists and not --force
   - run_agent(case, ...) → result + metrics
   - Write per-case files: metrics.json, predicted.geojson, viz_*.png, etc.
   - Catch and log per-case errors
6. Build aggregate: mean IoU, threshold buckets, regressions
7. Write summary.json
```

## Section walkthrough

### `_parse_args()` (~line 30)

Standard argparse. Notable flags:
- `--include-training-cases` — only safe with k-fold SAM3, otherwise the
  fine-tune leaks. The flag's help string is a long warning about this.
- `--hard-first` + `--prev-results` — ordering hack: when iterating on
  the worst cases during a debug session, you don't want to wait through
  150 trivial cases first.

### `EXCLUDE_SL_NOS` (line 28)

Set of 27 serial numbers from the Excel manifest that correspond to
hand-annotated training cases. Excluded by default to prevent training
contamination.

### `run_benchmark(model_name, output_dir, max_cases, start_from, dpi, max_iterations, only_cases, force, hard_first, prev_results_dir, enable_critic, include_training_cases)` (line ~196)

The core function:

#### Dataset loading (lines 218-245)

1. `pd.read_excel("planning-decisions-dataset.xlsx", sheet_name="0_planning_dataset_list")`
   → DataFrame with one row per case.
2. Filter to rows whose `Unique ID (Folder_Name)` exists in
   `eval_dir/`.
3. **Inject `*_merged` folders** that exist on disk but aren't in the
   Excel (added during cleanup — these are post-hoc consolidated cases
   whose own PDF + GT geojson live in their `_merged` folder). Synthetic
   `Sl no` 9001+ keeps them out of the training-exclude filter.
4. Apply `EXCLUDE_SL_NOS` filter unless `--include-training-cases`.

#### Filtering modes (lines 247-310)

- `--cases A B C` → keep only those folder names.
- `--max-cases N` + `--start-from N` → window into the dataset.
- `--hard-first`:
  - Read `prev_results_dir/<case>/metrics.json` for each case.
  - Sort cases by ascending IoU.
  - "Hardest first" — useful when iterating on hard-case fixes.

#### Model loading (line ~315)

Calls `tools.agent.load_models()` (or equivalent inline) — loads SAM3 +
LoRA, MINIMA-LoFTR, verifier. Done once before the loop so each case
doesn't pay the load cost.

#### Per-case loop (line ~340)

```python
for case_idx, (_, row) in enumerate(dataset.iterrows()):
    folder = str(row["Unique ID (Folder_Name)"])
    sl_no = int(row["Sl no"])

    # skip if already done and not --force
    out_dir = output_path / model_name / folder
    if (out_dir / "metrics.json").exists() and not force:
        all_results.append({...prior result...})
        continue

    # find the PDF
    pdf_files = list(folder_path.glob("*.pdf"))
    if not pdf_files: continue

    # run the agent
    try:
        result = run_agent(
            pdf_path=str(pdf), case_name=folder,
            model=model_name, sam3=sam3, ...)
    except Exception as e:
        # log + record
        all_results.append({"folder": folder, "error": str(e)})
        continue

    # compute metrics if GT exists
    metrics = compute_spatial_metrics(result["geojson"], gt_path)
    all_results.append({**result, **metrics})

    # write artefacts
    json.dump(metrics, ...)         # metrics.json
    json.dump(result["geojson"], ...) # predicted.geojson
    cv2.imwrite(viz_path, viz_img)   # viz_comparison.png
```

The try/except is critical: one case crashing shouldn't bring down a
200-case run. Errors are logged with the case folder + traceback.

#### Aggregation (line ~554)

```python
def compute_aggregate(all_results):
    iou_results = [r for r in all_results
                    if isinstance(r.get("iou"), (int, float))]
    n = len(iou_results)
    return {
        "n_cases": n,
        "mean_iou": sum(r["iou"] for r in iou_results) / n,
        "median_iou": median(r["iou"] for r in iou_results),
        "above_0.5_count": sum(1 for r in iou_results if r["iou"] >= 0.5),
        "above_0.7_count": ...,
        "above_0.9_count": ...,
        "errors": [r for r in all_results if "error" in r],
    }
```

Also produces a `comparison_to_prev.json` if `prev_results_dir` is set —
shows per-case wins/regressions vs the prior run.

#### Output structure

```
results/<output_dir>/<model_name>/
├── <case_1>/
│   ├── metrics.json
│   ├── predicted.geojson
│   ├── boundary_mask.png
│   ├── tile_info.json
│   ├── affine_H.npy
│   ├── viz_comparison.png       # pred vs GT side-by-side
│   ├── pdf_info.json            # reader output
│   ├── centers_tried.json       # geocoding transparency
│   ├── message_log.json         # full agent transcript
│   ├── critic_log.json          # critic decisions
│   ├── critic_panel.png         # critic's visual context
│   └── selected_boundary.png    # SAM3 candidate that won
├── <case_2>/
│   └── ...
└── summary.json                 # mean_iou, threshold_counts, errors
```

Per-case dirs are self-contained — you can re-evaluate or visualise any
case from disk without re-running the agent.

## `EXCLUDE_SL_NOS` (line 28)

Hardcoded list of 27 Sl numbers corresponding to training cases. Pulled
from the spreadsheet during initial setup; preserved here so re-runs
without `--include-training-cases` give honest test-set evaluation.

## Why this design

**Why Excel as the manifest?** The dataset was originally curated in
Excel by the planning-team annotators. Reading it directly avoids a
round-trip through CSV (and the inevitable encoding bugs).

**Why `Sl no` based exclusion instead of folder-name based?** The Excel
has the canonical Sl-no → case-name mapping. Folder names changed a few
times during dataset evolution; Sl numbers didn't.

**Why is the `*_merged` injection in `benchmark_runner.py` and not the
Excel?** The user explicitly asked to keep the Excel untouched (it's the
shared annotation source-of-truth). Injecting in Python keeps the Excel
clean while still running those cases.

**Why catch every case error?** A 200-case run takes 2-3 hours. A single
crash should not lose the other 199 runs. Caught errors are logged
verbosely in summary.json so you can fix and re-run only the failures.

**Why per-case dirs with all the artefacts?** Debugging a single
problematic case requires the full intermediate state (mask, affine,
tile_info, message log). Without saving everything, you'd have to re-run
the whole agent every time. With everything saved, post-hoc analysis is
fast (load tile_info + mask + affine, run `mask_to_geojson_affine` with
different cleanup variants — exactly what the recovery scripts did).
