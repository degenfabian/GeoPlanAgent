# ablations/

Paper-ablation scripts that run **outside** the main agent pipeline.
Each ablation is a self-contained driver that loads what it needs,
evaluates against the same ground-truth assets the main benchmark
uses, and reports metrics in a shape that drops directly into the
paper table.

See [`PLAN.md`](PLAN.md) for the full ablation tier list (A: supervisor-
requested, B: paper-headline, C: design-justification, D: skip-unless-
time-spare) plus the suggested 8-day execution sequence and
already-staged code.

## Currently in this directory

| File | Tier | Purpose |
|---|---|---|
| [`vlm_segmentation.py`](vlm_segmentation.py) | B1 / B2 | VLM-direct boundary segmentation baseline. Asks Gemini Flash/Pro to trace the boundary as a polygon and computes pixel IoU vs `training/dataset/boundary_masks/`. |
| [`PLAN.md`](PLAN.md) | — | Captured ablation plan with time / cost / scope per ablation. |

## VLM-direct segmentation (`vlm_segmentation.py`)

Pure single-shot VLM inference — no SAM3, no MINIMA, no agent loop. The
script:

1. Builds the per-case manifest in-memory from
   `training/dataset/maps/*.png` + `training/dataset/fold_assignment.json`
   (no `manifest.json` on disk — see `training/train_sam3_kfold._build_manifest_from_disk`)
   and iterates each case.
2. Sends each `training/dataset/maps/<case>.png` to the VLM with a
   segmentation prompt asking for polygon vertices.
3. Rasterises the returned polygons to a binary mask at the source
   image's resolution.
4. Computes pixel IoU vs `training/dataset/boundary_masks/<case>.png`
   (the same ground truth the SAM3 fine-tune trained against).
5. Aggregates mean / median / `≥0.50` / `≥0.70` / `≥0.80` / `≥0.90`
   bands — identical shape to `scripts/eval_sam_kfold_v2.py:summarise`.

### Coordinate convention

Vertices are emitted as `(y, x)` **integers in `[0, 1000]`** — Gemini
2.5/3's native bounding-box / segmentation convention per Google's
docs. The rasterizer scales x by `W/1000` and y by `H/1000`
separately (anisotropic for non-square pages) and flips the y-first
order to PIL's (x, y) before drawing.

A range sanity check fires on every call: if any returned vertex is
outside `[0, 1000]`, the script prints a `WARN` line with the
offending value. This catches the silent failure mode where the model
ignores the convention and emits pixel coords (huge values) or `[0,
1]` floats (tiny values).

### Running

```bash
# Dry run — no API call, verifies prompt + image dims
uv run python ablations/vlm_segmentation.py \
    --model gemini-flash --max-cases 3 --dry-run

# 3-case smoke test (~$0.01, ~2 min)
uv run python ablations/vlm_segmentation.py \
    --model gemini-flash --max-cases 3

# Full run (208 cases, ~$0.50, 30-60 min)
uv run python ablations/vlm_segmentation.py --model gemini-flash

# Prompt A/B via --prompt-file (no code edit)
uv run python ablations/vlm_segmentation.py \
    --model gemini-flash --prompt-file my_alternate_prompt.txt
```

Output: `results/ablation_vlm_seg/<model_id_with_slashes→underscores>/`:

```
results/ablation_vlm_seg/google_gemini-3-flash-preview/
├── results.csv              # per-case: iou, n_polygons, call_seconds, error
├── summary.json             # aggregate metrics in the SAM3-eval shape
└── pred_masks/<case>.png    # rasterised VLM-predicted mask per case
```

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--model` | `gemini-flash` | OpenRouter alias or full ID |
| `--max-cases` | none | Cap for quick iteration |
| `--fold` | none | Only evaluate cases in this fold (matches SAM3 fold-by-fold output) |
| `--held-out-only` | off | No-op for VLM (every case is "held out" since there's no training). Kept for API parity. |
| `--out-dir` | `results/ablation_vlm_seg` | Output base |
| `--throttle-s` | 1.0 | Sleep between API calls (rate-limit safety) |
| `--prompt-file` | none | Custom prompt for A/B experiments |
| `--dry-run` | off | Print what would be sent for case 0 and exit (no API call) |

## Adding a new ablation

Each ablation should be:

- **Self-contained**: one script, no shared mutable state with the
  main pipeline (it should be safe to run in parallel with a live
  benchmark).
- **Same evaluation surface**: load the same ground truth and report
  the same summary shape (`mean / median / ≥0.50 / ≥0.70 / ≥0.80 /
  ≥0.90`) so the row drops into the paper table without
  re-aggregation.
- **Self-documenting**: a module docstring at the top with `Usage`
  examples, the model defaults, and what it's comparing against. The
  benchmark runner doesn't need to know about it.

Add the new script here and add a short entry to [`PLAN.md`](PLAN.md)
under the appropriate tier.
