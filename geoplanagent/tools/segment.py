"""Extracts planning boundaries from map images via SAM3 semantic segmentation with k-fold LoRA routing."""

import numpy as np
import torch

from geoplanagent.utils import (
    N_FOLDS,
    resolve_fold as _resolve_fold,
)

# Production model: k-fold both-head LoRA, mean val_iou 0.908 ± 0.016 across
# folds. If this directory is missing the loader falls through to base SAM3
# (no LoRA) — useful for environments where the LoRA isn't shipped, but
# accuracy will drop noticeably.
DEFAULT_KFOLD_DIR = "models/sam3_lora"


def _load_kfold(kfold_dir: str, hf_token: str | None, device: torch.device) -> dict | None:
    """Load SAM3 base + all available fold adapters as named PEFT adapters.

    Returns the same dict shape as load_sam3_ft plus three extras:
      fold_assignment   {case_name: fold_idx}
      available_folds   set of fold indices that actually have a best.pt
      current_fold      int — fold currently active (0 by default)

    Missing folds are silently dropped — the run continues with whatever
    is on disk so partial-training states (e.g. only fold 0 finished)
    still work for the cases that hash to a trained fold. Cases that
    hash to a missing fold fall back to the lowest available fold at
    set_fold_for_case() time.
    """
    import json
    from pathlib import Path
    from peft import PeftModel
    from transformers import Sam3Model, Sam3Processor

    kfold_dir = Path(kfold_dir)
    fold_assignment_path = kfold_dir / "fold_assignment.json"
    if not fold_assignment_path.exists():
        # If adapter dirs ARE present but fold_assignment.json is missing,
        # the caller silently falls all the way back to base SAM3 (no
        # LoRA), which tanks inference accuracy. Surface that loudly so
        # a half-populated k-fold dir doesn't look like a clean run.
        adapter_dirs = [
            d
            for d in kfold_dir.glob("fold_*")
            if (d / "adapter_config.json").exists() or (d / "best.pt").exists()
        ]
        if adapter_dirs:
            print(
                f"  sam3 loader: WARNING — {fold_assignment_path} missing but "
                f"{len(adapter_dirs)} adapter dir(s) present "
                f"({[d.name for d in adapter_dirs]}). Falling back to "
                f"base SAM3 with NO LoRA — accuracy will drop. Restore "
                f"fold_assignment.json to use the trained adapters."
            )
        return None  # signal caller to use legacy path

    fold_assignment = {}
    try:
        fold_assignment = json.loads(fold_assignment_path.read_text())
    except Exception as e:
        print(
            f"  sam3 loader: WARNING — failed to parse {fold_assignment_path.name} "
            f"({e!r}); k-fold routing falls back to min(available_folds)"
        )

    # Find which folds actually have a usable adapter on disk.
    available_adapters = []
    for fold in range(N_FOLDS):
        adapter_dir = kfold_dir / f"fold_{fold}"
        # PEFT looks for adapter_config.json + adapter_model.safetensors
        # alongside the checkpoint. Our trainer saves to best.pt and
        # latest.pt as full PyTorch state_dict files. Provide both: if
        # PEFT-format dir exists, use it; otherwise we'll need to load
        # from best.pt manually below.
        if (adapter_dir / "adapter_config.json").exists():
            available_adapters.append((fold, adapter_dir, "peft"))
        elif (adapter_dir / "best.pt").exists():
            available_adapters.append((fold, adapter_dir / "best.pt", "raw"))
        # If neither, fold isn't trained yet; skip it.

    if not available_adapters:
        return None  # fall back to legacy path

    print(f"Loading SAM3 + k-fold adapters from {kfold_dir}")
    print(f"  available folds: {[fold for fold, _, _ in available_adapters]}")

    processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)
    base = Sam3Model.from_pretrained("facebook/sam3", token=hf_token)

    def rename_default_to_fold(state_dict: dict, fold_key: str) -> dict:
        """Rename PEFT 'default' adapter slots to per-fold names.

        The trainer saves all checkpoints with the active adapter named
        'default' (PEFT's single-adapter convention). The k-fold loader
        builds the model with adapter_name=f'fold_{k}' so we have one
        named adapter per fold. Without renaming, every weight in the
        checkpoint sits at `.lora_A.default.` / `.lora_B.default.` /
        `modules_to_save.default.` and the loaded model expects
        `.lora_A.fold_K.` / etc — load_state_dict(strict=False) silently
        drops every mismatch, leaving the model at base SAM3.

        That bug nullified every v6/v7 inference path until this fix.
        """
        renamed_state_dict = {}
        for weight_name, weight in state_dict.items():
            renamed_name = (
                weight_name.replace(".lora_A.default.", f".lora_A.{fold_key}.")
                .replace(".lora_B.default.", f".lora_B.{fold_key}.")
                .replace(".lora_embedding_A.default.", f".lora_embedding_A.{fold_key}.")
                .replace(".lora_embedding_B.default.", f".lora_embedding_B.{fold_key}.")
                .replace("modules_to_save.default.", f"modules_to_save.{fold_key}.")
            )
            renamed_state_dict[renamed_name] = weight
        return renamed_state_dict

    def load_raw_into(model, checkpoint_state: dict, fold_key: str, label: str) -> None:
        """Rename keys, load, and verify enough weights matched.

        Raises RuntimeError when the load updated zero fold-specific weights.
        """
        renamed = rename_default_to_fold(checkpoint_state, fold_key)
        before = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if fold_key in name
        }
        result = model.load_state_dict(renamed, strict=False)
        # Sanity-check: count how many of OUR expected fold_key weights
        # actually changed value after load. If 0, the load silently failed.
        n_changed = 0
        for name, param in model.named_parameters():
            if fold_key in name and name in before:
                if not torch.equal(before[name].to(param.device), param):
                    n_changed += 1
        print(
            f"    {label}: {n_changed} fold-specific weights updated, "
            f"{len(result.missing_keys)} missing, "
            f"{len(result.unexpected_keys)} unexpected"
        )
        if n_changed == 0:
            raise RuntimeError(
                f"{label}: state_dict load updated zero {fold_key} weights — "
                f"adapter name mismatch or empty checkpoint? "
                f"Missing example: {result.missing_keys[:1]}, "
                f"unexpected example: {result.unexpected_keys[:1]}"
            )

    from peft import LoraConfig, get_peft_model

    def build_lora_config(rank: int) -> LoraConfig:
        # modules_to_save MUST be set for PEFT to construct the wrapped
        # paths the checkpoint saved its trained head weights to.
        return LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
            lora_dropout=0.05,
            bias="none",
            modules_to_save=["mask_embedder", "presence_head", "semantic_projection"],
        )

    # Take the first available fold to construct the PeftModel; the
    # remaining ones load via .load_adapter(name=...).
    first_fold, first_source, first_kind = available_adapters[0]
    if first_kind == "peft":
        model = PeftModel.from_pretrained(base, str(first_source), adapter_name=f"fold_{first_fold}")
    else:
        # Raw best.pt: load state_dict into a freshly-configured PeftModel.
        checkpoint = torch.load(first_source, map_location="cpu", weights_only=False)
        rank = checkpoint.get("config", {}).get("rank", 16)
        fold_key = f"fold_{first_fold}"
        model = get_peft_model(base, build_lora_config(rank), adapter_name=fold_key)
        load_raw_into(model, checkpoint["state_dict"], fold_key, f"fold {first_fold} (raw)")

    # Add the remaining adapters under their own names.
    for fold, source, kind in available_adapters[1:]:
        if kind == "peft":
            model.load_adapter(str(source), adapter_name=f"fold_{fold}")
        else:
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            rank = checkpoint.get("config", {}).get("rank", 16)
            fold_key = f"fold_{fold}"
            model.add_adapter(fold_key, build_lora_config(rank))
            load_raw_into(model, checkpoint["state_dict"], fold_key, f"fold {fold} (raw)")

    model = model.to(device).eval()
    available_folds = {fold for fold, _, _ in available_adapters}
    initial_fold = sorted(available_folds)[0]
    model.set_adapter(f"fold_{initial_fold}")
    print(f"  SAM3 k-fold loaded on {device} (initial fold={initial_fold})")
    return {
        "processor": processor,
        "model": model,
        "device": device,
        "fold_assignment": fold_assignment,
        "available_folds": available_folds,
        "current_fold": initial_fold,
        "kind": "kfold",
    }


def set_fold_for_case(sam_state: dict, case_name: str | None) -> None:
    """Switch the active LoRA adapter to the fold that excluded this case
    from training. No-op when the loaded model is the legacy single
    adapter or when ``case_name`` resolves to the currently-active fold.

    Routing is delegated to :func:`geoplanagent.utils.resolve_fold`:
    look up by case name, then by canonical underscore form, then fall
    back to ``min(available_folds)`` for cases the training pool didn't
    contain.
    """
    if not isinstance(sam_state, dict) or sam_state.get("kind") != "kfold":
        return  # legacy single-adapter path; nothing to switch
    if not case_name:
        return
    fold_assignment = sam_state.get("fold_assignment") or {}
    available_folds = sam_state.get("available_folds") or set()
    if not available_folds:
        return
    fold = _resolve_fold(case_name, fold_assignment, available_folds)
    if sam_state.get("current_fold") == fold:
        return
    try:
        sam_state["model"].set_adapter(f"fold_{fold}")
        sam_state["current_fold"] = fold
    except Exception as e:
        print(f"  SAM3 set_adapter(fold_{fold}) failed: {e}")


def load_sam3_ft() -> dict:
    """Load SAM3 base model + k-fold LoRA adapters.

    Resolution order:
      1) k-fold adapters at `DEFAULT_KFOLD_DIR` if present — returns a dict
         that supports per-case fold switching via `set_fold_for_case`.
      2) Base SAM3 with no LoRA (warns; production accuracy will drop).

    The returned dict has the same shape in either case (processor, model,
    device, kind, optional kfold metadata), so callers don't branch.
    """
    import os
    from transformers import Sam3Processor, Sam3Model

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print(
            "  WARNING: HF_TOKEN not set. SAM3 download may fail if model "
            "is gated. Set: export HF_TOKEN=hf_xxx"
        )

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    kfold_state = _load_kfold(DEFAULT_KFOLD_DIR, hf_token, device)
    if kfold_state is not None:
        return kfold_state

    print(
        f"  WARNING: k-fold dir ({DEFAULT_KFOLD_DIR}) missing. Falling back to base SAM3 (no LoRA)."
    )
    processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)
    model = Sam3Model.from_pretrained("facebook/sam3", token=hf_token)
    model = model.to(device).eval()
    print(f"SAM3 (base, no LoRA) loaded on {device}")
    return {"processor": processor, "model": model, "device": device, "kind": "base"}


def extract_boundary_sam3_semantic(
    map_crop_path: str,
    processor,
    model,
    device: torch.device,
    query: str = "planning boundary",
    bbox: list[float] | None = None,
) -> np.ndarray | None:
    """Extract boundary using semantic segmentation mode.

    Uses post_process_semantic_segmentation() for a single best mask.
    Works with both base SAM3 and SAM3-FT (LoRA).

    Args:
        map_crop_path: Path to the map crop image.
        processor: Sam3Processor.
        model: SAM3 model (base or LoRA).
        device: torch device.
        query: Text prompt for segmentation.
        bbox: Optional [x1, y1, x2, y2] bounding box to focus segmentation.

    Returns:
        Binary mask (0/255 uint8) or None if extraction failed.
    """
    from PIL import Image

    image = Image.open(map_crop_path).convert("RGB")
    width, height = image.size

    # The LoRA was trained on the literal phrase "planning boundary"
    # (the default value of `query`). Other phrasings still work via the
    # underlying SAM3 + CLIP, but slot quality is best on the trained
    # phrase. Offline callers (e.g. the prompt-search ablation) may
    # override the default — we just truncate to fit CLIP's 32-token limit.
    if isinstance(query, str):
        words = query.split()
        if len(words) > 6:
            truncated = " ".join(words[:6])
            print(
                f"  SAM3 query truncated: {query!r} → {truncated!r} "
                f"(was {len(words)} words, CLIP limit ≈32 tokens)"
            )
            query = truncated

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        inputs = processor(
            images=image,
            text=query,
            input_boxes=[[[float(x1), float(y1), float(x2), float(y2)]]],
            input_boxes_labels=[[1]],
            return_tensors="pt",
        )
    else:
        inputs = processor(images=image, text=query, return_tensors="pt")

    inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    with torch.no_grad():
        outputs = model(**inputs)
    masks = processor.post_process_semantic_segmentation(outputs, target_sizes=[(height, width)])
    if len(masks) == 0:
        return None
    mask = masks[0].cpu().numpy().astype(np.uint8)
    mask = (mask > 0).astype(np.uint8) * 255
    mask_fraction_pct = np.sum(mask > 0) / (height * width) * 100
    bbox_suffix = f", bbox={bbox}" if bbox is not None else ""
    print(f"  SAM3 semantic: mask {mask_fraction_pct:.1f}% of image (query={query!r}{bbox_suffix})")
    return mask
