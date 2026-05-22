"""Rotation classifier inference for planning maps.

Two checkpoint layouts supported, in order of preference:

  1. K-fold (preferred when present): ``models/rotation_classifier_kfold/``
     with one ``fold_K/best.pt`` per fold plus a ``fold_assignment.json``
     mapping case_name → fold_idx. Each case is routed to the fold that
     did NOT see it during training (mirrors the SAM3 k-fold loader's
     ``set_fold_for_case``). Caller passes ``case_name``.

  2. Legacy single checkpoint: ``models/rotation_classifier/best.pt``.
     Used when the k-fold dir is missing or ``case_name`` is None.

Both paths apply 4-rotation TTA + confidence threshold:
  1. Predict on the input AND its 90/180/270 CW rotations
  2. Cyclically shift each rotated-frame prediction back to the
     original frame and ensemble (mean softmax)
  3. If the top class probability exceeds the threshold (default 0.50),
     return it; otherwise return 0 (don't rotate) — safer to leave a
     map alone than rotate it wrongly

Public API:
  predict_rotation_cw(map_bgr, case_name=None) -> int
      Returns CW degrees to rotate `map_bgr` to upright (0/90/180/270).
      Returns 0 if confidence is below threshold (abstain = no rotation).

  predict_rotation_with_confidence(map_bgr, case_name=None, threshold=0.50) -> dict
      Same prediction with explicit metadata for callers that want to
      log / decide based on confidence.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T


# Repo root. After the 2026-05-13 tools/ reorganization this file moved from
# tools/rotation_classifier.py to tools/io/rotation_classifier.py — three
# .parent steps now to reach the repo root (was two before the reorg).
_REPO = Path(__file__).resolve().parent.parent.parent
_CKPT_PATH = _REPO / "models" / "rotation_classifier" / "best.pt"
_KFOLD_DIR = _REPO / "models" / "rotation_classifier_kfold"

_DEFAULT_CONFIDENCE_THRESHOLD = 0.50

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Class index -> CW rotation in degrees needed to make the input upright.
_CLASS_TO_DEGREES = {0: 0, 1: 90, 2: 180, 3: 270}

_CV2_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


# Singleton state — load once on first call, keep cached.
_state_lock = threading.Lock()
_state: dict | None = None
_kfold_state: dict | None = None


class _RotationClassifier(torch.nn.Module):
    """Match the trainer's wrapper exactly so state_dict keys load
    cleanly. Trainer saves under `backbone.*`; bare ResNet50 wouldn't."""
    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.backbone = tv_models.resnet50(weights=None)
        self.backbone.fc = torch.nn.Linear(
            self.backbone.fc.in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)


def _build_model(n_classes: int = 4) -> torch.nn.Module:
    return _RotationClassifier(n_classes=n_classes)


def _device() -> torch.device:
    return torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu")


def _make_transform(img_size: int):
    return T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size), antialias=True),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def _load_state() -> dict:
    """Load the legacy single-checkpoint model + transform.
    Thread-safe singleton; used when the k-fold dir is unavailable or
    no case_name was passed."""
    global _state
    if _state is not None:
        return _state
    with _state_lock:
        if _state is not None:
            return _state
        if not _CKPT_PATH.exists():
            raise FileNotFoundError(
                f"rotation classifier checkpoint not found at {_CKPT_PATH}. "
                f"Train it via training/train_rotation_classifier.py.")
        ckpt = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config") or {}
        img_size = int(cfg.get("img_size", 768))

        device = _device()
        model = _build_model(n_classes=int(cfg.get("n_classes", 4)))
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device).eval()

        _state = {
            "models": {None: model},  # None = "any case" (legacy)
            "device": device,
            "img_size": img_size,
            "transform": _make_transform(img_size),
            "fold_assignment": None,
            "kind": "legacy",
            "best_val_acc": float(ckpt.get("best_val_acc", 0.0)),
        }
        return _state


def _load_kfold_state() -> Optional[dict]:
    """Load all available fold_K/best.pt checkpoints + fold_assignment.json.

    Returns the kfold state dict, or None if the k-fold dir is missing
    (caller should fall back to the legacy single-checkpoint loader).
    Thread-safe singleton."""
    global _kfold_state
    if _kfold_state is not None:
        return _kfold_state
    with _state_lock:
        if _kfold_state is not None:
            return _kfold_state
        fa_path = _KFOLD_DIR / "fold_assignment.json"
        if not fa_path.exists():
            return None
        try:
            fa = json.loads(fa_path.read_text())
        except Exception:
            return None

        device = _device()
        models: dict = {}
        # Track each fold's img_size so we can detect inconsistency.
        # Previously the loop overwrote a single `img_size` variable with
        # whichever fold was loaded last, so a mismatched fold would
        # silently use the wrong transform resolution for the others.
        per_fold_img_size: dict[int, int] = {}
        for fold_dir in sorted(_KFOLD_DIR.glob("fold_*")):
            ckpt_path = fold_dir / "best.pt"
            if not ckpt_path.exists():
                continue
            try:
                fold_k = int(fold_dir.name.split("_")[-1])
            except ValueError:
                continue
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = ckpt.get("config") or {}
            per_fold_img_size[fold_k] = int(cfg.get("img_size", 768))
            model = _build_model(n_classes=int(cfg.get("n_classes", 4)))
            model.load_state_dict(ckpt["state_dict"])
            model = model.to(device).eval()
            models[fold_k] = model
        if not models:
            return None

        # Pick a single transform resolution for inference. All folds were
        # trained at the same img_size in practice (768); if a future
        # checkpoint disagrees, warn loudly rather than silently using
        # whichever happened to load last. Use the most common value.
        sizes = list(per_fold_img_size.values())
        img_size = max(set(sizes), key=sizes.count)
        mismatched = {f: s for f, s in per_fold_img_size.items() if s != img_size}
        if mismatched:
            print(f"  rotation_classifier: WARNING — fold img_size mismatch "
                  f"(folds with non-default img_size: {mismatched}). Using "
                  f"img_size={img_size} for all folds; mismatched folds may "
                  f"see degraded accuracy because their training resolution "
                  f"differs from the inference transform.")

        _kfold_state = {
            "models": models,
            "device": device,
            "img_size": img_size,
            "transform": _make_transform(img_size),
            "fold_assignment": fa,
            "kind": "kfold",
            # Set, not list — `_resolve_fold` does `fold not in available_folds`
            # which is O(1) on sets, O(n) on lists. Matches tools.extraction.sam3.
            "available_folds": set(models.keys()),
        }
        print(f"  rotation_classifier: loaded {len(models)} k-fold adapter(s) "
              f"from {_KFOLD_DIR.name}/ "
              f"({len(fa)} cases routed via fold_assignment.json)")
        return _kfold_state


# Fold-routing helpers live in tools.core.fold_routing (shared with
# tools.extraction.sam3). The module-level aliases below keep the
# historical private names available to call sites in this file.
from tools.core.fold_routing import (
    resolve_fold as _resolve_fold,
)


def _model_for_case(case_name: Optional[str]) -> tuple[torch.nn.Module, dict]:
    """Pick the right model for `case_name`. Routing order:
       1. k-fold state (preferred when available + case_name given)
       2. legacy single-checkpoint state

    Returns (model, state_dict) where state_dict carries device/transform/
    img_size for the prediction loop."""
    if case_name is not None:
        kf = _load_kfold_state()
        if kf is not None:
            fold = _resolve_fold(case_name, kf["fold_assignment"],
                                  kf["available_folds"])
            return kf["models"][fold], kf
    # Legacy path
    st = _load_state()
    return st["models"][None], st


def _preprocess(map_bgr: np.ndarray, transform) -> torch.Tensor:
    """BGR uint8 -> normalised RGB tensor of shape (3, H, W)."""
    if map_bgr.ndim == 2:
        map_bgr = cv2.cvtColor(map_bgr, cv2.COLOR_GRAY2BGR)
    elif map_bgr.shape[2] == 4:
        map_bgr = cv2.cvtColor(map_bgr, cv2.COLOR_BGRA2BGR)
    rgb = cv2.cvtColor(map_bgr, cv2.COLOR_BGR2RGB)
    return transform(rgb)


@torch.no_grad()
def predict_rotation_with_confidence(
    map_bgr: np.ndarray,
    case_name: Optional[str] = None,
    threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """Predict CW rotation (0/90/180/270) needed to make `map_bgr` upright.

    Uses 4-view TTA + confidence threshold. If the ensemble's top class
    probability is below `threshold`, returns 0 ("don't rotate") — safer
    than risking a wrong rotation on a map that's already upright.

    Args:
        map_bgr: HxWx3 BGR uint8 numpy array (the format cv2 produces).
        case_name: optional case identifier. When provided AND a k-fold
            checkpoint dir is available, the case is routed to the fold
            that did NOT see it during training (clean inference). When
            absent the legacy single-checkpoint model is used.
        threshold: confidence below which we abstain (return 0).

    Returns:
        {
            "rotation_cw_degrees": int (0/90/180/270),
            "applied": bool — true if we recommend rotating
                       (rotation_cw_degrees != 0),
            "confidence": float (top-class softmax prob 0..1),
            "all_probs": list[float] of len 4 — class probabilities in
                       original-frame order [0°, 90°, 180°, 270°],
            "abstained_low_confidence": bool — true if confidence < threshold,
            "raw_class": int — argmax class before threshold (for logging),
            "fold": int | None — which k-fold model handled this case
                       (or None if the legacy path was used),
        }
    """
    model, state = _model_for_case(case_name)
    device = state["device"]
    transform = state["transform"]
    fold = None
    if state["kind"] == "kfold" and case_name is not None:
        fold = _resolve_fold(case_name, state["fold_assignment"],
                              state["available_folds"])

    base = _preprocess(map_bgr, transform).unsqueeze(0).to(device)  # (1, 3, H, W)

    # 4 TTA views: input + 90/180/270° CW rotations of the input.
    # torch.rot90 conventions:
    #   k=1 → 90° CCW
    #   k=2 → 180°
    #   k=3 → 270° CCW = 90° CW
    # We want CW augmentations:
    #   90° CW  = rot90(x, 3)
    #   180°    = rot90(x, 2)
    #   270° CW = rot90(x, 1)
    aug_torch_k = {0: 0, 1: 3, 2: 2, 3: 1}

    ensemble = torch.zeros(1, 4, device=device)
    for k_cw in (0, 1, 2, 3):
        x = base if k_cw == 0 else torch.rot90(base, aug_torch_k[k_cw],
                                                 dims=(-2, -1))
        logits = model(x)
        probs = F.softmax(logits, dim=-1)
        # Convert back to original frame: rotated-frame class C' on an
        # input we rotated k_cw further CW corresponds to original class
        # C = (C' + k_cw) mod 4. In torch.roll semantics
        # (new[i] = old[(i - shifts) mod 4]), use shifts=k_cw.
        if k_cw != 0:
            probs = torch.roll(probs, shifts=k_cw, dims=-1)
        ensemble = ensemble + probs
    ensemble = ensemble / 4.0

    probs_np = ensemble.squeeze(0).cpu().numpy().astype(float)
    top_class = int(np.argmax(probs_np))
    confidence = float(probs_np[top_class])

    abstained = confidence < threshold
    rotation = 0 if abstained else _CLASS_TO_DEGREES[top_class]

    return {
        "rotation_cw_degrees": rotation,
        "applied": rotation != 0,
        "confidence": confidence,
        "all_probs": probs_np.tolist(),
        "abstained_low_confidence": abstained,
        "raw_class": top_class,
        "fold": fold,
    }


def predict_rotation_cw(
    map_bgr: np.ndarray,
    case_name: Optional[str] = None,
    threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
) -> int:
    """Convenience wrapper: returns just the CW degrees to rotate (0 if abstained)."""
    return predict_rotation_with_confidence(
        map_bgr, case_name=case_name, threshold=threshold,
    )["rotation_cw_degrees"]


def auto_rotate(
    map_bgr: np.ndarray,
    case_name: Optional[str] = None,
    threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """Predict + apply rotation. Returns (rotated_map, info_dict).

    The returned map is the input rotated CW by the predicted amount
    (or unchanged if abstained / class 0). info_dict is the same as
    predict_rotation_with_confidence's return. Pass `case_name` to route
    the prediction through k-fold (excludes the case from training).
    """
    info = predict_rotation_with_confidence(
        map_bgr, case_name=case_name, threshold=threshold)
    rot = info["rotation_cw_degrees"]
    if rot == 0:
        if verbose:
            if info["abstained_low_confidence"]:
                print(f"  rotation_classifier: abstained "
                      f"(conf={info['confidence']:.2f} < {threshold:.2f}); "
                      f"raw_class={info['raw_class']} -> "
                      f"{_CLASS_TO_DEGREES[info['raw_class']]}°. "
                      f"Leaving map unrotated.")
            else:
                print(f"  rotation_classifier: 0° (already upright, "
                      f"conf={info['confidence']:.2f})")
        return map_bgr, info
    rotated = cv2.rotate(map_bgr, _CV2_ROTATE_CODES[rot])
    if verbose:
        fold_str = (f" fold={info['fold']}" if info.get("fold") is not None
                    else "")
        print(f"  rotation_classifier: rotating {rot}° CW "
              f"(conf={info['confidence']:.2f}{fold_str})")
    return rotated, info
