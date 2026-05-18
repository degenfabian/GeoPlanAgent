# tools/extraction/

SAM3 boundary segmentation + binary-mask cleanup primitives. The
fine-tuned model loaded here is what runs inside `match_at` (lazily,
per page) to produce the planning-boundary mask that gets projected
through the committed affine.

## Public API

```python
from tools.extraction.sam3 import (
    load_sam3_ft,                    # one-time loader at process start
    set_fold_for_case,               # swap in fold k's LoRA for this case
    extract_boundary_sam3_semantic,  # run the segmentation
    try_fill_boundary_outline,       # morphological hole-fill for thin outlines
)
from tools.extraction.mask_ops import (
    fill_mask_holes,
    expand_thin_mask,
    keep_dominant_components,
    cleanup_mask_pipeline,           # all three chained
)
```

## How it fits in the pipeline

1. `benchmark_runner.load_models()` calls `load_sam3_ft()` once. The
   returned state dict carries the base SAM3 + LoRA-wrapped model, the
   processor, the device, and k-fold metadata (`adapters_by_fold`,
   `fold_assignment`).
2. `tools.agent.tools.match._get_or_compute_mask` is called per page
   inside `match_at`. It:
   - Looks up the case's fold via `set_fold_for_case(state.sam3_state,
     state.case_name)` and swaps in `fold_<k>/best.pt` if not already
     active.
   - Calls `extract_boundary_sam3_semantic(image_path, processor,
     model, device, query="planning boundary")`.
   - Caches the resulting mask on `state.sam_masks_by_page[page]`.
3. `mask_to_geojson_affine` (in `tools.matching`) feeds the cached mask
   through `cleanup_mask_pipeline` before vectorising to GeoJSON.

## SAM3 + LoRA (`sam3.py`)

- **Base model**: `facebook/sam3` (HuggingFace, ~3 GB, requires `HF_TOKEN`).
- **Fine-tune**: LoRA r=16 on both heads (semantic + instance).
  Shipped per-fold in `models/sam3_lora/fold_<k>/best.pt`. If
  `models/sam3_lora/` is missing, the loader falls through to base
  SAM3 — accuracy drops materially.
- **Text query** is locked to the literal phrase `"planning boundary"`
  (`_SAM3_QUERY` in `tools.agent.tools.match`). The LoRA was trained
  against this exact string; using a paraphrase silently regresses.
- **Inference mode**: semantic segmentation only. `pred_masks` from the
  instance head is unused at inference (training kept it diverse via
  best-IoU-only loss so it'd be available, but the worker currently
  takes the semantic-head mask).

### Fold dispatch

```python
state = load_sam3_ft()               # base + processor + adapters_by_fold
set_fold_for_case(state, case_name)  # swap in fold k's LoRA

# Then to run:
mask = extract_boundary_sam3_semantic(
    image_path, state["processor"], state["model"], state["device"],
    query="planning boundary",
)
```

`set_fold_for_case` canonicalises the case name (`replace(":", "_").
replace("/", "_")`), looks up `fold_assignment.json[case_name]`,
falls back to `md5(canonical) % 5` for unseen cases, and applies the
matching LoRA. Important: hash on the canonical form, otherwise
`md5("12:00114:ART4")` and `md5("12_00114_ART4")` resolve to
different folds — silent leakage. The normaliser fixes this.

### `try_fill_boundary_outline` (legacy)

Pre-LoRA, the model occasionally returned a thin outline-style mask
(just the boundary line, not the filled interior). This helper
morphologically closes the outline and floodfills the exterior from
border pixels to recover the interior. Rarely needed against the
LoRA, which is trained on filled masks, but kept for ablations and
the base-SAM3 fallback path.

## Mask cleanup primitives (`mask_ops.py`)

`mask_to_geojson_affine` in `tools.matching` runs these in order
before vectorisation. Each is a small pure function on a binary
mask:

| Function | What it does |
|---|---|
| `keep_dominant_components(mask, min_area_frac=0.05)` | Drops noise blobs. Keeps connected components whose area is at least `min_area_frac` of the largest blob. |
| `expand_thin_mask(mask)` | Thickens hollow outlines (rare against the LoRA, common against base SAM3). Detects boundary-style masks via fill-ratio and dilates. |
| `fill_mask_holes(mask)` | Plugs interior gaps. Morphological close + external-contour fill — SAM3 sometimes returns masks with road/text-shaped holes inside the boundary, producing fragmented polygons instead of one. |
| `cleanup_mask_pipeline(mask)` | All three chained: `keep_dominant_components → expand_thin_mask → fill_mask_holes`. |

Kernel sizes auto-scale to the input dimensions (~1% of the shorter
side, clamped to `[5, 31]` and forced odd) so the chain works on
either ~256 px previews or full-res ~2000 px maps.
