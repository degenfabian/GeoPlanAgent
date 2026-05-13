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

    If the mask is a thin outline (low fill ratio), applies morphological
    close to bridge small gaps, then floodfills from border pixels to mark
    the exterior, and returns the interior as a filled mask.
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

    # Flood-fill from border pixels to mark the exterior, then invert.
    flood = closed.copy()
    flood_h, flood_w = flood.shape[:2]
    ff_mask = np.zeros((flood_h + 2, flood_w + 2), dtype=np.uint8)

    border_seeds = []
    if flood[0, 0] == 0: border_seeds.append((0, 0))
    if flood[0, flood_w - 1] == 0: border_seeds.append((flood_w - 1, 0))
    if flood[flood_h - 1, 0] == 0: border_seeds.append((0, flood_h - 1))
    if flood[flood_h - 1, flood_w - 1] == 0: border_seeds.append((flood_w - 1, flood_h - 1))

    for seed in border_seeds:
        cv2.floodFill(flood, ff_mask, seed, 128)

    # Pixels that are 0 (not exterior, not original outline) are interior.
    filled = (flood == 0).astype(np.uint8) * 255
    # Combine the original outline with the new interior fill.
    filled = np.maximum(filled, closed)

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

    # The LoRA was trained on the literal phrase "planning boundary"
    # (default). Other queries still work via the underlying SAM3 + CLIP,
    # but quality is best on the trained phrase. The agent is free to
    # override; we just truncate to fit CLIP's 32-token limit.
    if isinstance(query, str):
        words = query.split()
        if len(words) > 6:
            truncated = " ".join(words[:6])
            print(f"  SAM3 query truncated: {query!r} → {truncated!r} "
                  f"(was {len(words)} words, CLIP limit ≈32 tokens)")
            query = truncated

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


def extract_candidates_multi_prompt(
    image_path, processor, model, device,
    queries=("planning boundary", "site outline", "red line boundary"),
    bbox=None, top_k_per_query=3, total_top_k=8, dedupe_iou=0.7,
):
    """Run SAM3 with multiple text prompts; merge and dedupe candidates.

    The LoRA was trained on "planning boundary" but recovery experiments
    showed alternative prompts beat the default on a non-trivial fraction
    of cases (`'site outline'`, `'red line boundary'`). Running them all
    and reranking gives a more robust candidate pool than a single prompt.

    Each candidate keeps a `query` field tagging which prompt produced it.
    Candidates with mask IoU > `dedupe_iou` are merged (highest-score wins).

    Args:
        image_path, processor, model, device: as in extract_candidates.
        queries: ordered iterable of text prompts to try.
        bbox: optional [x1,y1,x2,y2] focus box, applied to every query.
        top_k_per_query: candidates kept per prompt before dedup.
        total_top_k: total candidates returned after dedup, score-sorted desc.
        dedupe_iou: pairwise mask IoU above which candidates are deduped.

    Returns:
        List of dicts with the same shape as `extract_candidates` plus
        a `'query'` key. Empty list if no prompt produced anything usable.
    """
    pool = []
    for q in queries:
        try:
            cands = extract_candidates(
                image_path, processor, model, device,
                query=q, bbox=bbox, top_k=top_k_per_query,
            )
        except Exception as e:
            print(f"  multi-prompt SAM3: query {q!r} failed ({e})")
            continue
        for c in cands:
            c["query"] = q
            pool.append(c)
    if not pool:
        return []

    pool.sort(key=lambda c: -c["score"])
    kept: list = []
    for cand in pool:
        is_dup = False
        for k in kept:
            if _mask_iou(cand["mask"], k["mask"]) >= dedupe_iou:
                is_dup = True
                break
        if not is_dup:
            kept.append(cand)
        if len(kept) >= total_top_k:
            break
    return kept


def _mask_iou(a, b):
    if a.shape != b.shape:
        return 0.0
    a_b = a > 0
    b_b = b > 0
    inter = np.logical_and(a_b, b_b).sum()
    union = np.logical_or(a_b, b_b).sum()
    return float(inter) / float(union) if union else 0.0


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


# ── Callout-aware ranking (small-site disambiguation) ────────────────
# For small sites with "THE SITE" callouts, SAM3 often picks the
# rectangular callout BOX over the actual red site polygon. These
# helpers reorder candidates to prefer the polygon nearest the red
# callout target (the small irregular polygon, not the box itself).

def find_callout_target_centroid(plan_img_bgr, masks=None):
    """Find the small red SITE polygon (NOT the THE SITE callout box).

    Strategy: extract red blobs via HSV. Drop blobs that look like
    rectangular callout boxes (aspect > 2.0, fill_ratio > 0.85) or
    huge stamps (area > 5% of image). Return centroid of largest
    remaining red blob (the actual site polygon). Returns None if
    nothing matches.
    """
    # Direct HSV red detection (more permissive than tools.boundary_color
    # for synthetic / muted reds typical in scanned planning maps)
    hsv = cv2.cvtColor(plan_img_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv,
                       np.array([0, 70, 50], dtype=np.uint8),
                       np.array([10, 255, 255], dtype=np.uint8))
    m2 = cv2.inRange(hsv,
                       np.array([170, 70, 50], dtype=np.uint8),
                       np.array([180, 255, 255], dtype=np.uint8))
    red = cv2.bitwise_or(m1, m2)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(red, connectivity=8)
    H, W = red.shape
    img_area = H * W
    keep = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 50 or a > 0.05 * img_area:  # skip noise + huge stamps/boxes
            continue
        fill = a / max(1, w * h)
        ar = max(w, h) / max(1, min(w, h))
        # Reject rectangular callout-box blobs
        if fill > 0.85 and ar > 2.0:
            continue
        keep.append((a, cents[i]))
    if not keep:
        return None
    _, c = max(keep, key=lambda t: t[0])
    return (float(c[0]), float(c[1]))


def rank_by_callout_proximity(masks, target_xy):
    """Return mask index whose centroid is nearest target_xy. None if no masks."""
    if not masks or target_xy is None:
        return None
    tx, ty = target_xy
    best_i, best_d = None, float("inf")
    for i, m in enumerate(masks):
        ys, xs = np.where(m > 0)
        if xs.size == 0:
            continue
        d = (xs.mean() - tx) ** 2 + (ys.mean() - ty) ** 2
        if d < best_d:
            best_d, best_i = d, i
    return best_i


# ── Model loading ───────────────────────────────────────────────────────────

# Production model: k-fold both-head LoRA, mean val_iou 0.908 ± 0.016 across
# folds. If this directory is missing the loader falls through to base SAM3
# (no LoRA) — useful for environments where the LoRA isn't shipped, but
# accuracy will drop noticeably.
DEFAULT_KFOLD_DIR = "models/sam3_lora_v7_both"
N_FOLDS = 5


def _normalise_case_name(case_name: str) -> str:
    """Convert a case identifier to the canonical underscore form used as
    the key in fold_assignment.json. Idempotent.

    Both the auto-labeller and the curated-dataset builder use a
    'safe filename' convention that replaces ':' with '_'. The
    benchmark runner passes the original eval-data folder name (with
    colons), so we have to translate before any lookup or hash. Without
    this, lookups miss and fall back to a hash on the colon form, which
    differs from the hash on the underscore form — silent leakage.
    """
    return (case_name or "").replace(":", "_").replace("/", "_")


def _fold_for_case(case_name: str, n_folds: int = N_FOLDS) -> int:
    """Deterministic fold assignment via md5(canonical_case_name) % n_folds.

    Mirrors `scripts/build_curated_training_set.py:fold_for` so a case
    that was in fold k's val set during training also routes to fold k
    at inference (= the model that did NOT see this case during training).

    IMPORTANT: hash on the canonical (underscore) form so that
    `md5("12:00114:ART4")` and `md5("12_00114_ART4")` resolve to the
    same fold — both are aliases for the same case.
    """
    import hashlib
    canonical = _normalise_case_name(case_name)
    h = hashlib.md5(canonical.encode()).hexdigest()
    return int(h, 16) % n_folds


def _load_kfold(kfold_dir, hf_token, device):
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
    fa_path = kfold_dir / "fold_assignment.json"
    if not fa_path.exists():
        return None  # signal caller to use legacy path

    fold_assignment = {}
    try:
        fold_assignment = json.loads(fa_path.read_text())
    except Exception:
        pass

    # Find which folds actually have a usable adapter on disk.
    available = []
    for k in range(N_FOLDS):
        adapter_dir = kfold_dir / f"fold_{k}"
        # PEFT looks for adapter_config.json + adapter_model.safetensors
        # alongside the checkpoint. Our trainer saves to best.pt and
        # latest.pt as full PyTorch state_dict files. Provide both: if
        # PEFT-format dir exists, use it; otherwise we'll need to load
        # from best.pt manually below.
        if (adapter_dir / "adapter_config.json").exists():
            available.append((k, adapter_dir, "peft"))
        elif (adapter_dir / "best.pt").exists():
            available.append((k, adapter_dir / "best.pt", "raw"))
        # If neither, fold isn't trained yet; skip it.

    if not available:
        return None  # fall back to legacy path

    print(f"Loading SAM3 + k-fold adapters from {kfold_dir}")
    print(f"  available folds: {[k for k, _, _ in available]}")

    processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)
    base = Sam3Model.from_pretrained("facebook/sam3", token=hf_token)

    def _rename_default_to_fold(sd: dict, fold_key: str) -> dict:
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
        out = {}
        for k, v in sd.items():
            new_k = (k
                     .replace(".lora_A.default.", f".lora_A.{fold_key}.")
                     .replace(".lora_B.default.", f".lora_B.{fold_key}.")
                     .replace(".lora_embedding_A.default.",
                              f".lora_embedding_A.{fold_key}.")
                     .replace(".lora_embedding_B.default.",
                              f".lora_embedding_B.{fold_key}.")
                     .replace("modules_to_save.default.",
                              f"modules_to_save.{fold_key}."))
            out[new_k] = v
        return out

    def _load_raw_into(model, ckpt_state, fold_key: str, label: str):
        """Rename keys, load, and verify enough weights matched."""
        renamed = _rename_default_to_fold(ckpt_state, fold_key)
        before = {n: p.detach().clone()
                  for n, p in model.named_parameters()
                  if fold_key in n}
        result = model.load_state_dict(renamed, strict=False)
        # Sanity-check: count how many of OUR expected fold_key weights
        # actually changed value after load. If 0, the load silently failed.
        changed = 0
        for n, p in model.named_parameters():
            if fold_key in n and n in before:
                if not torch.equal(before[n].to(p.device), p):
                    changed += 1
        print(f"    {label}: {changed} fold-specific weights updated, "
              f"{len(result.missing_keys)} missing, "
              f"{len(result.unexpected_keys)} unexpected")
        if changed == 0:
            raise RuntimeError(
                f"{label}: state_dict load updated zero {fold_key} weights — "
                f"adapter name mismatch or empty checkpoint? "
                f"Missing example: {result.missing_keys[:1]}, "
                f"unexpected example: {result.unexpected_keys[:1]}")

    # Take the first available fold to construct the PeftModel; the
    # remaining ones load via .load_adapter(name=...).
    first_k, first_src, first_kind = available[0]
    if first_kind == "peft":
        model = PeftModel.from_pretrained(base, str(first_src),
                                            adapter_name=f"fold_{first_k}")
    else:
        # Raw best.pt: load state_dict into a freshly-configured PeftModel.
        ckpt = torch.load(first_src, map_location="cpu", weights_only=False)
        from peft import LoraConfig, get_peft_model
        cfg = ckpt.get("config", {})
        rank = cfg.get("rank", 16)
        # modules_to_save MUST be set for PEFT to construct the wrapped
        # paths the checkpoint saved its trained head weights to.
        lora_cfg = LoraConfig(
            r=rank, lora_alpha=rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "fc1", "fc2"],
            lora_dropout=0.05, bias="none",
            modules_to_save=["mask_embedder", "presence_head",
                              "semantic_projection"],
        )
        fold_key = f"fold_{first_k}"
        model = get_peft_model(base, lora_cfg, adapter_name=fold_key)
        _load_raw_into(model, ckpt["state_dict"], fold_key,
                       f"fold {first_k} (raw)")

    # Add the remaining adapters under their own names.
    for k, src, kind in available[1:]:
        if kind == "peft":
            model.load_adapter(str(src), adapter_name=f"fold_{k}")
        else:
            ckpt = torch.load(src, map_location="cpu", weights_only=False)
            cfg = ckpt.get("config", {})
            rank = cfg.get("rank", 16)
            from peft import LoraConfig
            lora_cfg = LoraConfig(
                r=rank, lora_alpha=rank * 2,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "fc1", "fc2"],
                lora_dropout=0.05, bias="none",
                modules_to_save=["mask_embedder", "presence_head",
                                  "semantic_projection"],
            )
            fold_key = f"fold_{k}"
            model.add_adapter(fold_key, lora_cfg)
            _load_raw_into(model, ckpt["state_dict"], fold_key,
                           f"fold {k} (raw)")

    model = model.to(device).eval()
    available_folds = {k for k, _, _ in available}
    initial = sorted(available_folds)[0]
    model.set_adapter(f"fold_{initial}")
    print(f"  SAM3 k-fold loaded on {device} (initial fold={initial})")
    return {
        "processor": processor,
        "model": model,
        "device": device,
        "fold_assignment": fold_assignment,
        "available_folds": available_folds,
        "current_fold": initial,
        "kind": "kfold",
    }


def set_fold_for_case(sam_state, case_name):
    """Switch the active LoRA adapter to the fold that excluded this case
    from training. No-op when the loaded model is the legacy single
    adapter.

    Routing:
      - Look up `case_name` in fold_assignment.json (training cases).
      - If not present, hash via md5 % N_FOLDS (new cases at deployment).
      - If the chosen fold isn't trained yet, fall back to the lowest
        available fold and log it.
    """
    if not isinstance(sam_state, dict) or sam_state.get("kind") != "kfold":
        return  # legacy single-adapter path; nothing to switch
    if not case_name:
        return
    fa = sam_state.get("fold_assignment") or {}
    avail = sam_state.get("available_folds") or set()
    if not avail:
        return
    canonical = _normalise_case_name(case_name)
    # Look up under both as-given and canonical form so this works
    # regardless of whether fa has colon-keys (legacy) or underscore-keys
    # (current). The build script uses underscore-keys.
    fold = fa.get(case_name)
    if fold is None:
        fold = fa.get(canonical)
    if fold is None:
        fold = _fold_for_case(case_name)  # _fold_for_case canonicalises internally
    if fold not in avail:
        fold_chosen = sorted(avail)[0]
        print(f"  SAM3 fold {fold} not trained yet for case {case_name!r}; "
              f"falling back to fold {fold_chosen}")
        fold = fold_chosen
    if sam_state.get("current_fold") == fold:
        return
    try:
        sam_state["model"].set_adapter(f"fold_{fold}")
        sam_state["current_fold"] = fold
    except Exception as e:
        print(f"  SAM3 set_adapter(fold_{fold}) failed: {e}")


def load_sam3_ft(kfold_dir=DEFAULT_KFOLD_DIR):
    """Load SAM3 base model + k-fold LoRA adapters.

    Resolution order:
      1) k-fold adapters at `kfold_dir` if present — returns a dict that
         supports per-case fold switching via `set_fold_for_case`.
      2) Base SAM3 with no LoRA (warns; production accuracy will drop).

    The returned dict has the same shape in either case (processor, model,
    device, kind, optional kfold metadata), so callers don't branch.
    """
    import os
    from transformers import Sam3Processor, Sam3Model

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("  WARNING: HF_TOKEN not set. SAM3 download may fail if model "
              "is gated. Set: export HF_TOKEN=hf_xxx")

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    out = _load_kfold(kfold_dir, hf_token, device)
    if out is not None:
        return out

    print(f"  WARNING: k-fold dir ({kfold_dir}) missing. "
          f"Falling back to base SAM3 (no LoRA).")
    processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)
    model = Sam3Model.from_pretrained("facebook/sam3", token=hf_token)
    model = model.to(device).eval()
    print(f"SAM3 (base, no LoRA) loaded on {device}")
    return {"processor": processor, "model": model, "device": device,
            "kind": "base"}


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

    # The LoRA was trained on the literal phrase "planning boundary"
    # (the default value of `query`). Other phrasings still work via the
    # underlying SAM3 + CLIP, but slot quality is best on the trained
    # phrase. The agent is free to override the default if a particular
    # case needs it — we just truncate to fit CLIP's 32-token limit.
    if isinstance(query, str):
        words = query.split()
        if len(words) > 6:
            truncated = " ".join(words[:6])
            print(f"  SAM3 query truncated: {query!r} → {truncated!r} "
                  f"(was {len(words)} words, CLIP limit ≈32 tokens)")
            query = truncated

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
