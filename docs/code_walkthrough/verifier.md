# `tools/verifier.py`

**430 lines.** A small learned classifier that takes a (planning map crop,
OS tile crop, mask) triplet and predicts whether the match is correct.
Uses a frozen DINOv2 backbone for visual features + an MLP head for
classification. Works alongside the multi-axis reward (`reward.py`) — the
verifier is the visual axis, the reward axes are tabular.

## Public API

| Symbol | Purpose |
|---|---|
| `Verifier` (class) | the model + score interface |
| `get_verifier(ckpt_path)` | singleton accessor |
| `get_verifier_for_case(case_name)` | k-fold-aware accessor |
| `fold_assignment(case_name, n_folds)` | which fold to use |
| `render_disagreement_panel(score_result)` | visual debug panel |
| `format_features_for_context(score_result)` | text summary for the agent |

Constants:
- `VERIFIER_DEFAULT_CKPT = "models/verifier_v3/head.pt"`

## How it works (high level)

The verifier predicts `P(match is correct | crops, mask, tabular features)`
by combining:

1. **Visual features** — DINOv2 ViT-B applied to the map and tile crops
   independently, then comparing the [CLS] tokens (cosine sim, L2
   distance, a couple of fixed transforms).
2. **Tabular features** — handcrafted scalars from the match_info /
   mask geometry (inlier count, scale, mask coverage, etc.).
3. **MLP head** — concatenates visual + tabular, feeds through a small
   fully-connected network, outputs a logit in [0,1].

Trained via `scripts/train_verifier.py` on a hand-labelled dataset
("`verifier_dataset/`").

## k-fold setup

Same idea as SAM3's k-fold: each case is assigned to a fold via
`fold_assignment`, and inference uses the fold that DIDN'T see the case
during training. Avoids contamination on the eval set.

`get_verifier_for_case(case_name)` looks up the fold, then loads
`models/verifier_v3/folds/fold_K/head.pt`. If folds aren't present,
falls back to the global `head.pt`.

## Function walkthroughs (selected)

### `_bbox_of_mask(mask, margin_frac=0.15)` (line 64)

Tight bbox around the mask's nonzero pixels, expanded by 15% on each side
so DINOv2 sees a bit of context around the boundary. Returns
`(x0, y0, x1, y1)`.

### `_crop_and_resize(img, bbox, size)` (line 80)

Crop to bbox, then resize to a fixed square (typical 224 for DINOv2).
Uses BORDER_REFLECT for the resize so the model doesn't see padding.

### `_compute_visual_sim(map_cls, tile_cls, ...)` (line 104)

Builds the visual feature vector from the [CLS] tokens of the map and
tile DINOv2 forward passes. Includes:
- Cosine similarity
- L2 distance
- Element-wise abs diff (for the head to learn dimension-specific
  signals)

### `_compute_tabular(meta_like, mask)` (line 128)

Builds the scalar features:
- Mask coverage (fraction of crop)
- Mask compactness
- Aspect ratio of mask bbox
- MINIMA n_inliers, score, avg_scale (read from `meta_like`)
- Plus a few normalised log/abs transforms

### `Verifier` class (lines 186-302)

```python
v = Verifier(ckpt_path="models/verifier_v3/head.pt", device=...)
result = v.score(map_bgr, tile_bgr, mask, meta_like)
# result = {"score": 0.83, "feats": {...}, "panels": {...}}
```

The `score` method:
1. Crops both images via the mask bbox.
2. Runs DINOv2 on both crops.
3. Computes visual + tabular features.
4. Concatenates → MLP head → sigmoid.
5. Returns the score plus all features (for debugging) and rendered crops
   (for `render_disagreement_panel`).

### `render_disagreement_panel(score_result, target_h=600)` (line 348)

For low-confidence verifier scores, render a side-by-side panel showing
the map crop, tile crop, and mask overlay. Used by the critic agent to
visualise borderline cases.

### `format_features_for_context(score_result)` (line 402)

Pretty-prints the verifier's scalar features as a multi-line string for
inclusion in the agent's prompt context.

## Why this design

**Why DINOv2 + MLP instead of fine-tuning DINO end-to-end?** A frozen
DINOv2 + small head trains in minutes on the available data; full
fine-tuning would need 10× more data and tens of GPU-hours. The frozen
backbone also generalises better to unseen UK regions.

**Why both visual and tabular features?** Inlier counts and scale ratios
catch geometric failures that visuals can't (e.g. matched a logo at the
wrong scale); visuals catch wrong-region cases where the geometry looks
fine. Combined, they cover both.

**Why k-fold inference like SAM3?** Same reason — the verifier dataset
overlaps with the eval set. Routing each case to the fold that didn't see
it gives an honest measurement.

**Why expose both `score` and `format_features_for_context`?** The agent
is given the score AND the feature breakdown so it can sanity-check the
verifier's output (e.g. "score 0.85 but mask coverage is only 2% — that's
suspicious").
