# `tools/sam3_boundary.py`

**620 lines.** Wraps SAM3 (Facebook's text-prompted segmentation model)
with a LoRA fine-tune for "planning boundary" extraction. Handles the
k-fold inference setup (different LoRA adapter per case to avoid train/eval
contamination), candidate ranking, and the multi-prompt pipeline added
during the integration of recovery wins.

## Public API

| Function | Purpose |
|---|---|
| `extract_candidates(...)` | one prompt → top-K candidate masks |
| `extract_candidates_multi_prompt(...)` | several prompts → merged candidates |
| `select_best_candidate(candidates)` | pick from a candidate list |
| `try_fill_boundary_outline(mask)` | fill thin outline masks |
| `extract_boundary_sam3_semantic(...)` | single-mask semantic mode |
| `load_sam3_ft(kfold_dir=...)` | model loader (singleton) |
| `set_fold_for_case(sam_state, case_name)` | switch active LoRA adapter |

Module-level constants:
- `DEFAULT_KFOLD_DIR = "models/sam3_lora_v7_both"` — production LoRA
- `N_FOLDS = 5`

## Boundary-mask post-processing

### `try_fill_boundary_outline(mask)` (lines 21-67)

Some SAM3 candidates trace just the outline of a region (a thin closed
ring) rather than the filled interior. This function detects that and
fills the interior so the projected polygon has the right area:

1. **Skip if already filled** — `fill_ratio > 0.4` means the mask is
   substantively filled; leave alone.
2. **Skip if too sparse** — `< 0.001` ratio is just specks; flood-filling
   would mess things up.
3. **Morphological close** with a 7×7 ellipse (lines 41-42) — bridges
   small gaps in a not-quite-closed outline.
4. **Flood-fill from each corner** (lines 49-56) — paints the exterior
   with value 128, leaving the original outline (255) and any interior
   holes (0) untouched.
5. **Interior = pixels still at 0** after the flood-fill (line 59).
6. **Combine outline + interior** with `np.maximum`.
7. **Sanity check**: only return the filled version if it's meaningfully
   bigger AND less than 85% of the page (line 65). Otherwise return the
   original — protects against flood-filling escaping through a gap and
   filling the entire image.

### `_compactness(mask_uint8)` (lines 72-83)

Compactness `4πA/P²` of the largest contour. A circle has compactness 1;
a long thin shape has compactness near 0. Used to filter out clearly-wrong
candidates (long noise streaks).

## Single-prompt candidate extraction

### `extract_candidates(image_path, processor, model, device, query, bbox, top_k)` (lines 88-173)

The core SAM3 inference call:

1. **Truncate the query to 6 words** (lines 113-119) — CLIP's text encoder
   has a ~32-token limit. Longer prompts get silently clipped, which is
   confusing; we truncate explicitly with a warning.
2. **Load the image** as PIL RGB (SAM3 expects PIL, not OpenCV).
3. **Build inputs** — with or without a bbox prompt. The bbox is in
   pixel coords on the input image and steers SAM3 to focus there.
4. **Forward pass** — `model(**inputs)` returns `pred_masks` (multi-mask
   logits) and `pred_logits` (per-mask confidence scores).
5. **Get top-K by score** — `scores.topk(top_k)`.
6. **For each top-K mask**:
   - Sigmoid + bilinear-interpolate to the original image size.
   - Threshold at 0.5 → binary mask.
   - Compute area %.
   - Filter out absurdly small (< 0.01%) or absurdly big (> 90%) masks.
   - Compute compactness.
7. **Return** a list of dicts with `mask`, `score`, `area_pct`, `compactness`.

## Multi-prompt extraction (added during recovery integration)

### `extract_candidates_multi_prompt(...)` (lines 176-232)

Runs `extract_candidates` for each prompt in a list, then merges and
deduplicates. Default prompts:

```python
("planning boundary", "site outline", "red line boundary")
```

The trained prompt is "planning boundary" but recovery experiment Phase 26
showed alternatives win on a non-trivial fraction of cases. Each candidate
gets a `query` field tagging its origin.

Pipeline:
1. **Run each prompt** through `extract_candidates` with `top_k_per_query`.
2. **Pool all candidates**, sort by score (cross-prompt scores aren't
   strictly comparable but are close enough for ranking).
3. **NMS dedup**: for each candidate (in score order), if any kept
   candidate has mask-IoU ≥ `dedupe_iou` (default 0.7), drop it.
4. **Cap at `total_top_k`** (default 8).

Returns the deduplicated list. Empty list if no prompt produced anything.

### `_mask_iou(a, b)` (lines 234-242)

Standard mask IoU: `intersection / union` over uint8 masks. Returns 0 if
the shapes differ (defensive — shouldn't happen in normal flow).

## `select_best_candidate(candidates)` (lines 244-262)

Pick a single mask from a candidate list:

1. **Filter** for "good" candidates: compactness > 0.05 AND
   0.5% < area_pct < 70%.
2. **Pick the highest-scored** of the filtered set, or the highest-scored
   overall if none pass the filter.
3. **Warn** if the chosen score is < 0.3 (low confidence — boundary may
   be unreliable).
4. **Run `try_fill_boundary_outline`** on the chosen mask before returning.

This is the heuristic used when callers want a single mask; the agent
typically uses `extract_candidates_multi_prompt` and lets the LLM pick via
`select_indices`, bypassing this function.

## k-fold LoRA loading

The trick: SAM3 was fine-tuned with 5-fold cross-validation. Each fold's
adapter was trained on 80% of cases and validated on the other 20%. To
avoid contamination at inference, each case is routed to the fold that
*didn't* see it during training.

### `_normalise_case_name(case_name)` (lines 280-296)

Replaces `:` and `/` with `_` so case names are valid as filesystem keys
and produce stable hashes regardless of which form is passed in.

### `_fold_for_case(case_name, n_folds=5)` (lines 294-313)

Deterministic md5-based fold assignment. Mirrors the trainer's logic
exactly — a case that was in fold k's val set during training routes to
fold k at inference. The `_normalise_case_name` step ensures `12:00114:ART4`
and `12_00114_ART4` hash to the same fold.

### `_load_kfold(kfold_dir, hf_token, device)` (lines 311-475)

The complex loader. Loads SAM3 base + every fold's LoRA adapter as named
PEFT adapters (`fold_0`, `fold_1`, etc.):

1. **Read `fold_assignment.json`** to know which cases go to which fold.
2. **Find available folds** by checking for `best.pt` in each fold dir.
3. **Load the first available fold** as a `PeftModel` to bootstrap.
4. **For each remaining fold**, call `model.load_adapter(name=f"fold_{k}")`
   to add it as a named adapter on the same model object.
5. **Rename keys** (`_rename_default_to_fold` at lines 303-329) — the
   trainer saves all adapters under `default`; we need them under
   `fold_K`. Without this rename, `load_state_dict(strict=False)` silently
   drops every key.
6. **Verify**: count how many `fold_K`-named weights actually changed
   value after loading. If zero, raise a hard error — a silent-failed
   load was the bug that nullified every v6/v7 inference path before the
   rename was added.

Returns a dict with `processor`, `model`, `device`, `kind="kfold"`, plus
`fold_assignment`, `available_folds`, `current_fold`.

The verbose loading logs you see (`fold 0 (raw): 964 fold-specific weights
updated, 0 missing, 0 unexpected`) come from `_load_raw_into` (line 331).
The "missing" counts grow per fold because each fold's checkpoint only
contains its own fold's weights — not a bug, expected behaviour.

### `set_fold_for_case(sam_state, case_name)` (lines 482-521)

Called per-case before inference. Looks up the case's fold via
`_fold_for_case`, then `model.set_adapter(f"fold_{k}")` to make that
adapter active. If the case's intended fold isn't available (partial
training state), falls back to the lowest available fold.

The agent calls this in both `extract_boundary` paths (semantic + instance)
right before invoking SAM3.

## `load_sam3_ft(kfold_dir=DEFAULT_KFOLD_DIR)` (lines 524-560)

The top-level entry point. Tries k-fold loading; if that fails (no
adapter dir, no `fold_assignment.json`), falls back to plain SAM3 base
with a warning.

Returns a dict shaped consistently with the k-fold path so callers don't
need to branch.

## `extract_boundary_sam3_semantic(...)` (lines 562+)

Runs SAM3 in semantic-segmentation mode (single best mask, no per-instance
candidates). Used by the `extract_boundary(mode="semantic")` agent tool.
Wraps `processor.post_process_semantic_segmentation` from HuggingFace.

Filters the same generic too-small / too-big criteria as `extract_candidates`.

## Why this design

**Why k-fold inference?** Test contamination. The fine-tune dataset
overlaps with the eval set. Routing each case to the fold that didn't see
it gives an honest evaluation. In a true production deployment with a
separate test set this isn't needed — could just train one adapter on
everything.

**Why multi-prompt + NMS dedup?** The LoRA was trained on "planning
boundary", but Phase 26 of the recovery showed `'site outline'` and
`'red line boundary'` win on cases the trained prompt misses. Pooling and
deduplicating gives the agent a richer menu without overwhelming it with
duplicates.

**Why separate `try_fill_boundary_outline` from candidate generation?**
Some downstream callers want the raw mask (e.g. for visualisation), some
want the filled version (e.g. for IoU). Keeping the fill step independent
lets the caller choose.

**Why the explicit fold-rename + verify?** This was a real bug: the early
v6/v7 LoRA loader silently produced base SAM3 because the adapter keys
didn't match. Catching it via the changed-weight count check would have
saved a week of confused debugging. Worth keeping as a tripwire.
