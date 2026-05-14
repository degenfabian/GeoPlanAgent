"""Rotation classifier inference for planning maps.

Loads the trained ResNet50 checkpoint from
`models/rotation_classifier/best.pt` and predicts the CW rotation
needed to make a planning-map image upright.

Uses 4-rotation TTA + confidence threshold:
  1. Predict on the input AND its 90/180/270 CW rotations
  2. Cyclically shift each rotated-frame prediction back to the
     original frame and ensemble (mean softmax)
  3. If the top class probability exceeds the threshold (default 0.80),
     return it; otherwise return 0 (don't rotate) — safer to leave a
     map alone than rotate it wrongly

Public API:
  predict_rotation_cw(map_bgr) -> int
      Returns CW degrees to rotate `map_bgr` to upright (0/90/180/270).
      Returns 0 if confidence is below threshold (abstain = no rotation).

  predict_rotation_with_confidence(map_bgr, threshold=0.80) -> dict
      Same prediction with explicit metadata for callers that want to
      log / decide based on confidence.
"""

from __future__ import annotations

import threading
from pathlib import Path

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

_DEFAULT_CONFIDENCE_THRESHOLD = 0.80

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


def _load_state() -> dict:
    """Load the model + transform on first call. Thread-safe singleton."""
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

        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu")
        model = _build_model(n_classes=int(cfg.get("n_classes", 4)))
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device).eval()

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size), antialias=True),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

        _state = {
            "model": model,
            "device": device,
            "img_size": img_size,
            "transform": transform,
            "best_val_acc": float(ckpt.get("best_val_acc", 0.0)),
        }
        return _state


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
    threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """Predict CW rotation (0/90/180/270) needed to make `map_bgr` upright.

    Uses 4-view TTA + confidence threshold. If the ensemble's top class
    probability is below `threshold`, returns 0 ("don't rotate") — safer
    than risking a wrong rotation on a map that's already upright.

    Args:
        map_bgr: HxWx3 BGR uint8 numpy array (the format cv2 produces).
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
        }
    """
    state = _load_state()
    model = state["model"]
    device = state["device"]
    transform = state["transform"]

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
    }


def predict_rotation_cw(
    map_bgr: np.ndarray,
    threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
) -> int:
    """Convenience wrapper: returns just the CW degrees to rotate (0 if abstained)."""
    return predict_rotation_with_confidence(map_bgr, threshold=threshold)[
        "rotation_cw_degrees"]


def auto_rotate(
    map_bgr: np.ndarray,
    threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """Predict + apply rotation. Returns (rotated_map, info_dict).

    The returned map is the input rotated CW by the predicted amount
    (or unchanged if abstained / class 0). info_dict is the same as
    predict_rotation_with_confidence's return.
    """
    info = predict_rotation_with_confidence(map_bgr, threshold=threshold)
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
        print(f"  rotation_classifier: rotating {rot}° CW "
              f"(conf={info['confidence']:.2f})")
    return rotated, info
