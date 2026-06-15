"""Extracts planning boundaries from map images via SAM3 semantic segmentation with k-fold LoRA routing."""

import cv2
import numpy as np
import torch

from geoplanagent.utils import (
    N_FOLDS,
    resolve_fold as _resolve_fold,
)

# Production model: k-fold both-head LoRA, mean val_iou 0.908 ± 0.016 across
# folds. These adapters are required — load_sam3_ft raises if they're missing.
DEFAULT_KFOLD_DIR = "models/sam3_lora"


def _load_kfold(kfold_dir: str, hf_token: str | None, device: torch.device) -> dict | None:
    """Load SAM3 base + every trained fold adapter as named PEFT adapters.

    Returns ``{processor, model, device, fold_assignment, available_folds,
    current_fold}``, or ``None`` when no adapters are on disk (the caller
    turns ``None`` into a hard error).

    Missing folds are silently dropped — the run continues with whatever
    is on disk so partial-training states (e.g. only fold 0 finished)
    still work for the cases that hash to a trained fold. Cases that hash
    to a missing fold fall back to the lowest available fold at
    set_fold_for_case() time.
    """
    import json
    from pathlib import Path
    from peft import PeftModel
    from transformers import Sam3Model, Sam3Processor

    kfold_dir = Path(kfold_dir)
    fold_assignment_path = kfold_dir / "fold_assignment.json"
    if not fold_assignment_path.exists():
        return None

    fold_assignment = {}
    try:
        fold_assignment = json.loads(fold_assignment_path.read_text())
    except Exception as e:
        print(
            f"  sam3 loader: WARNING — failed to parse {fold_assignment_path.name} "
            f"({e!r}); k-fold routing falls back to min(available_folds)"
        )

    # Each trained fold is a PEFT adapter dir (adapter_config.json +
    # adapter_model.safetensors) written by train_sam3_kfold's save_pretrained.
    available_adapters = [
        (fold, kfold_dir / f"fold_{fold}")
        for fold in range(N_FOLDS)
        if (kfold_dir / f"fold_{fold}" / "adapter_config.json").exists()
    ]
    if not available_adapters:
        return None

    print(f"Loading SAM3 + k-fold adapters from {kfold_dir}")
    print(f"  available folds: {[fold for fold, _ in available_adapters]}")

    processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)
    base = Sam3Model.from_pretrained("facebook/sam3", token=hf_token)

    # The first adapter constructs the PeftModel; the rest attach as named
    # adapters so set_fold_for_case can switch between them per case.
    first_fold, first_source = available_adapters[0]
    model = PeftModel.from_pretrained(base, str(first_source), adapter_name=f"fold_{first_fold}")
    for fold, source in available_adapters[1:]:
        model.load_adapter(str(source), adapter_name=f"fold_{fold}")

    model = model.to(device).eval()
    available_folds = {fold for fold, _ in available_adapters}
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
    }


def set_fold_for_case(sam_state: dict, case_name: str | None) -> None:
    """Switch the active LoRA adapter to the fold that excluded this case
    from training. No-op when ``case_name`` is empty or already resolves
    to the currently-active fold.

    Routing is delegated to :func:`geoplanagent.utils.resolve_fold`:
    look up by case name, then by canonical underscore form, then fall
    back to ``min(available_folds)`` for cases the training pool didn't
    contain.
    """
    if not case_name:
        return
    available_folds = sam_state.get("available_folds") or set()
    if not available_folds:
        return
    fold_assignment = sam_state.get("fold_assignment") or {}
    fold = _resolve_fold(case_name, fold_assignment, available_folds)
    if sam_state.get("current_fold") == fold:
        return
    try:
        sam_state["model"].set_adapter(f"fold_{fold}")
        sam_state["current_fold"] = fold
    except Exception as e:
        print(f"  SAM3 set_adapter(fold_{fold}) failed: {e}")


def load_sam3_ft() -> dict:
    """Load SAM3 base + k-fold LoRA adapters from ``DEFAULT_KFOLD_DIR``.

    Returns a dict (processor, model, device, + k-fold metadata) that
    supports per-case fold switching via ``set_fold_for_case``. Raises
    ``RuntimeError`` when no trained adapters are found: they are required
    for correct segmentation, so we fail loudly rather than silently
    running base SAM3.
    """
    from huggingface_hub import get_token

    # HF token from the HF_TOKEN env var (.env is loaded at the entry point) or
    # a `huggingface-cli login` cached credential. It is required to fetch the
    # gated facebook/sam3 base, so fail loud here rather than later with a
    # confusing download error.
    hf_token = get_token()
    if not hf_token:
        raise RuntimeError(
            "No Hugging Face token found. The gated facebook/sam3 base model "
            "requires authentication: set HF_TOKEN (e.g. in .env) or run "
            "`huggingface-cli login`."
        )

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    kfold_state = _load_kfold(DEFAULT_KFOLD_DIR, hf_token, device)
    if kfold_state is None:
        raise RuntimeError(
            f"No SAM3 k-fold adapters found at {DEFAULT_KFOLD_DIR} (expected "
            f"fold_assignment.json + fold_*/adapter_config.json). The "
            f"fine-tuned adapters are required for correct segmentation; "
            f"aborting rather than silently falling back to base SAM3."
        )
    return kfold_state


def extract_boundary_sam3_semantic(
    image: np.ndarray,
    processor,
    model,
    device: torch.device,
    query: str = "planning boundary",
) -> np.ndarray | None:
    """Extract boundary using semantic segmentation mode.

    Uses post_process_semantic_segmentation() for a single best mask.
    Works with both base SAM3 and SAM3-FT (LoRA).

    Args:
        image: the map crop as a cv2 BGR uint8 array.
        processor: Sam3Processor.
        model: SAM3 model (base or LoRA).
        device: torch device.
        query: Text prompt for segmentation.

    Returns:
        Binary mask (0/255 uint8) or None if extraction failed.
    """
    from PIL import Image

    # SAM3 wants a PIL RGB image; rendered pages are cv2 BGR arrays.
    # Reversing the channels here is byte-identical to the old
    # cv2.imwrite + Image.open().convert("RGB") round-trip.
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    width, height = pil_image.size

    # The LoRA was trained on the literal phrase "planning boundary"
    # (the default value of `query`). Other phrasings still work via the
    # underlying SAM3 + CLIP, but slot quality is best on the trained
    # phrase. Offline callers (e.g. the prompt-search ablation) may
    # override the default — we cap at 6 words, comfortably under CLIP's
    # ~32-token limit.
    if isinstance(query, str):
        words = query.split()
        if len(words) > 6:
            truncated = " ".join(words[:6])
            print(
                f"  SAM3 query truncated: {query!r} → {truncated!r} "
                f"(was {len(words)} words, CLIP limit ≈32 tokens)"
            )
            query = truncated

    inputs = processor(images=pil_image, text=query, return_tensors="pt")

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
    print(f"  SAM3 semantic: mask {mask_fraction_pct:.1f}% of image (query={query!r})")
    return mask
