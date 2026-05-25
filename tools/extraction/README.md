# tools/extraction/

SAM3 boundary segmentation. The fine-tuned model loaded here is what
runs inside `match_at` (lazily, per page) to produce the
planning-boundary mask that gets projected through the committed
affine.

## Public API

```python
from tools.extraction.sam3 import (
    load_sam3_ft,                    # one-time loader at process start
    set_fold_for_case,               # swap in fold k's LoRA for this case
    extract_boundary_sam3_semantic,  # run the segmentation
)
```

## How it fits in the pipeline

1. `benchmark_runner.load_models()` calls `load_sam3_ft()` once. The
   returned state dict carries the base SAM3 + LoRA-wrapped model (the
   PEFT `PeftModel` with one named adapter per fold), the processor,
   the device, and k-fold metadata (`fold_assignment`,
   `available_folds`, `current_fold`).
2. `tools.agent.tools.match._get_or_compute_mask` is called per page
   inside `match_at`. It:
   - Looks up the case's fold via `set_fold_for_case(state.sam3_state,
     state.case_name)` and switches the active adapter to
     `fold_<k>` if not already active.
   - Calls `extract_boundary_sam3_semantic(image_path, processor,
     model, device, query="planning boundary")`.
   - Caches the resulting mask on `state.sam_masks_by_page[page]`.
3. `mask_to_geojson_affine` (in `tools.matching._core`) vectorises the
   cached mask directly via `cv2.findContours` — **no morphological
   cleanup**. A 177-case ablation 2026-05-22 showed the old
   `keep_dominant_components → expand_thin_mask → fill_mask_holes`
   chain was a +0.001 IoU wash (2 wins / 2 losses) and was deleted
   along with the `mask_ops` module.

## SAM3 + LoRA (`sam3.py`)

- **Base model**: `facebook/sam3` (HuggingFace, ~3 GB, requires `HF_TOKEN`).
- **Fine-tune**: LoRA r=16 (`lora_alpha=32`, dropout 0.05, bias none)
  on `q_proj` / `k_proj` / `v_proj` / `o_proj` / `fc1` / `fc2`
  across every transformer subsystem (vision_encoder, text_encoder,
  geometry_encoder, detr_encoder, detr_decoder, mask_decoder), plus
  fully-trained head modules (`mask_embedder`, `presence_head`,
  `semantic_projection`) carried via PEFT's `modules_to_save`.
- **Checkpoint formats** — the loader (`_load_kfold`) supports both
  PEFT-format dirs (`fold_<k>/adapter_config.json` +
  `adapter_model.safetensors`) and raw `fold_<k>/best.pt` files; the
  shipped checkpoints are PEFT format (~76 MB / fold). When loading
  raw state dicts the loader renames `default` adapter slots to
  per-fold names (`.lora_A.default. → .lora_A.fold_K.`) and verifies
  that at least one fold-specific weight actually changed value — a
  silent name-mismatch was the bug that nullified every v6/v7
  inference path until 2026-04.
- **Loader fallbacks** — if `fold_assignment.json` is missing the
  loader prints a warning and falls through to base SAM3 (no LoRA);
  if the entire `models/sam3_lora/` dir is missing, it also falls
  through. Both paths cost materially on accuracy, so the warning is
  loud.
- **Text query** is locked to the literal phrase `"planning boundary"`
  (`_SAM3_QUERY` in `tools.agent.tools.match`). The LoRA was trained
  against this exact string; using a paraphrase silently regresses.
- **Inference mode**: semantic segmentation only. `pred_masks` from the
  instance head is unused at inference (training kept it diverse via
  best-IoU-only loss so it'd be available, but the worker currently
  takes the semantic-head mask).

### Fold dispatch

```python
state = load_sam3_ft()               # base SAM3 + PEFT wrapper w/ all folds loaded
set_fold_for_case(state, case_name)  # switch active adapter to fold k

# Then to run:
mask = extract_boundary_sam3_semantic(
    image_path, state["processor"], state["model"], state["device"],
    query="planning boundary",
)
```

`set_fold_for_case` canonicalises the case name (`replace(":", "_").
replace("/", "_")`), looks up `fold_assignment.json[case_name]`, and
calls `model.set_adapter(f"fold_{k}")` to swap the active LoRA. The
shared helper is `tools.core.fold_routing.resolve_fold`. For cases not
in the training pool (only possible for an external deployment on a
fresh case), it returns `min(available_folds)` deterministically — no
fold "owns" an unseen case, so any adapter is equally valid; an
earlier md5-hash fallback added no signal and was removed.
