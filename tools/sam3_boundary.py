"""
SAM3 Boundary Extraction
========================
Extracts planning boundaries from map images using SAM3 segmentation.

Two modes used in production:
- Semantic: post_process_semantic_segmentation for a single best mask
- Instance: multi-candidate extraction with area/compactness filtering

Supports both base SAM3 and SAM3-FT (LoRA fine-tuned) models.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F


# ── Flood fill ──────────────────────────────────────────────────────────────

def try_fill_boundary_outline(mask):
    """Try to fill gaps in a boundary outline mask using morphological closing.

    If the mask is a thin outline (low fill ratio), applies morphological close
    to bridge small gaps, then floodfills from all border pixels to find the
    exterior, and returns the interior as a filled mask. If the mask is already
    filled or closing doesn't help, returns the original mask.
    """
    if mask is None:
        return None

    h, w = mask.shape[:2]
    total_pixels = h * w
    filled_pixels = np.sum(mask > 0)
    fill_ratio = filled_pixels / total_pixels

    # Already well-filled or too sparse — return as-is
    if fill_ratio > 0.4 or fill_ratio < 0.001:
        return mask

    # Morphological close to bridge small gaps in outline
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Floodfill from ALL border pixels (not just corners) to find exterior.
    # This handles boundaries that touch edges or corners correctly.
    flood = closed.copy()
    fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    # Top and bottom rows
    for x in range(0, w, 4):
        if flood[0, x] == 0:
            cv2.floodFill(flood, fill_mask, (x, 0), 128)
        if flood[h - 1, x] == 0:
            cv2.floodFill(flood, fill_mask, (x, h - 1), 128)
    # Left and right columns
    for y in range(0, h, 4):
        if flood[y, 0] == 0:
            cv2.floodFill(flood, fill_mask, (0, y), 128)
        if flood[y, w - 1] == 0:
            cv2.floodFill(flood, fill_mask, (w - 1, y), 128)

    # Interior = pixels that are neither boundary (255) nor exterior (128)
    interior = np.where((flood != 128) & (flood != 255), 255, 0).astype(np.uint8)
    # Combine: boundary outline + filled interior
    filled = np.maximum(closed, interior)

    filled_after = np.sum(filled > 0) / total_pixels
    # Only use filled version if it meaningfully increased coverage
    if filled_after > fill_ratio * 1.2 and filled_after < 0.85:
        return filled
    return mask


# ── Compactness ─────────────────────────────────────────────────────────────

def _compactness(mask_uint8):
    """Compute compactness (4*pi*area/perim^2) of the largest contour. Circle=1."""
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perim = cv2.arcLength(largest, True)
    if perim <= 0:
        return 0.0
    return 4 * np.pi * area / (perim * perim + 1e-8)


# ── Shared candidate extraction ────────────────────────────────────────────

def extract_candidates(image_path, processor, model, device,
                       query="planning boundary", bbox=None, top_k=5):
    """Extract instance mask candidates from SAM3.

    Shared logic for all multi-candidate extraction modes. Runs the model,
    extracts top-k masks by confidence, filters by area.

    Args:
        image_path: Path to the map crop image.
        processor: Sam3Processor.
        model: SAM3 model (base or LoRA).
        device: torch device.
        query: Text prompt for segmentation.
        bbox: Optional [x1, y1, x2, y2] bounding box to focus segmentation.
        top_k: Number of top candidates to consider.

    Returns:
        List of dicts with 'mask' (uint8), 'score', 'area_pct', 'compactness'.
    """
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        inputs = processor(
            images=image, text=query,
            input_boxes=[[[float(x1), float(y1), float(x2), float(y2)]]],
            input_boxes_labels=[[1]],
            return_tensors="pt",
        )
    else:
        inputs = processor(images=image, text=query, return_tensors="pt")

    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    pred_masks = outputs.pred_masks[0]
    pred_logits = outputs.pred_logits
    if pred_logits.dim() == 3:
        scores = torch.sigmoid(pred_logits[0].squeeze(-1))
    else:
        scores = torch.sigmoid(pred_logits.squeeze())

    top_scores, top_indices = scores.topk(min(top_k, len(scores)))

    candidates = []
    for score, idx in zip(top_scores, top_indices):
        mask = torch.sigmoid(pred_masks[idx])
        if mask.dim() == 3:
            mask = mask[0]
        mask_up = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0), size=(h, w),
            mode="bilinear", align_corners=False,
        ).squeeze().cpu().numpy()

        mask_uint8 = (mask_up > 0.5).astype(np.uint8) * 255
        area_pct = np.sum(mask_uint8 > 0) / (h * w) * 100

        if area_pct < 0.01 or area_pct > 90:
            continue

        candidates.append({
            "mask": mask_uint8,
            "score": score.item(),
            "area_pct": area_pct,
            "compactness": _compactness(mask_uint8),
        })

    return candidates


def select_best_candidate(candidates):
    """Select the best candidate mask using compactness + area filters.

    Prefers candidates with reasonable compactness (>0.05) and area (0.5-70%).
    Falls back to highest raw confidence if none pass filters.

    Returns:
        Binary mask (uint8) or None.
    """
    if not candidates:
        return None

    good = [c for c in candidates
            if c["compactness"] > 0.05 and 0.5 < c["area_pct"] < 70]
    chosen = max(good if good else candidates, key=lambda c: c["score"])

    if chosen["score"] < 0.3:
        print(f"  SAM3 low confidence ({chosen['score']:.2f}) — boundary may be unreliable")

    print(f"  SAM3: {len(candidates)} candidates, "
          f"chose score={chosen['score']:.3f} area={chosen['area_pct']:.1f}% "
          f"compact={chosen['compactness']:.2f}")

    return try_fill_boundary_outline(chosen["mask"])


# ── Model loading ───────────────────────────────────────────────────────────

def load_sam3_ft(lora_path="models/sam3_lora_v4/checkpoint_latest"):
    """Load SAM3 base model + fine-tuned LoRA adapter.

    Falls back to base SAM3 (without LoRA) if the adapter path doesn't exist.

    Returns:
        dict with 'processor', 'model', 'device'.
    """
    import os
    from transformers import Sam3Processor, Sam3Model

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("  WARNING: HF_TOKEN not set. SAM3 download may fail if model "
              "is gated. Set: export HF_TOKEN=hf_xxx")

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(lora_path):
        print(f"  WARNING: LoRA path not found: {lora_path}")
        print("    Will use base SAM3 without fine-tuning")
        processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)
        model = Sam3Model.from_pretrained("facebook/sam3", token=hf_token)
        model = model.to(device).eval()
        print(f"SAM3 (base, no LoRA) loaded on {device}")
        return {"processor": processor, "model": model, "device": device}

    from peft import PeftModel

    print(f"Loading SAM3-FT (LoRA from {lora_path})...")
    processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)
    base_model = Sam3Model.from_pretrained("facebook/sam3", token=hf_token)
    model = PeftModel.from_pretrained(base_model, str(lora_path))
    model = model.to(device).eval()
    print(f"SAM3-FT loaded on {device}")
    return {"processor": processor, "model": model, "device": device}


# ── Semantic segmentation ───────────────────────────────────────────────────

def extract_boundary_sam3_semantic(map_crop_path, processor, model, device,
                                   query="planning boundary", bbox=None):
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
    w, h = image.size

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        inputs = processor(
            images=image, text=query,
            input_boxes=[[[float(x1), float(y1), float(x2), float(y2)]]],
            input_boxes_labels=[[1]],
            return_tensors="pt",
        )
    else:
        inputs = processor(images=image, text=query, return_tensors="pt")

    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    masks = processor.post_process_semantic_segmentation(outputs, target_sizes=[(h, w)])
    if len(masks) == 0:
        return None
    mask = masks[0].cpu().numpy().astype(np.uint8)
    mask = (mask > 0).astype(np.uint8) * 255
    pct = np.sum(mask > 0) / (h * w) * 100
    bbox_str = f", bbox={bbox}" if bbox is not None else ""
    print(f"  SAM3 semantic: mask {pct:.1f}% of image (query={query!r}{bbox_str})")
    return try_fill_boundary_outline(mask)
