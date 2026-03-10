"""The auto-rotation classifier, end to end: the ResNet50 model + ImageNet
normalisation, the k-fold checkpoint loading and no-leakage fold routing,
4-view-TTA inference with a confidence abstain, and applying the predicted
correction to an image.

Inference (geoplanagent.tools.pdf) calls ``auto_rotate``; training
(training/train_rotation.py, training/eval/eval_rotation_kfold.py) reuses the
model, transform and constants.
"""

import cv2
import json
import threading
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T

from geoplanagent.paths import FOLD_ASSIGNMENT, ROTATION_KFOLD_DIR
from geoplanagent.utils import device as _device, resolve_fold as _resolve_fold

# Standard ImageNet RGB normalisation. The ResNet50 backbone is ImageNet
# pre-trained, so every consumer must normalise inputs with these exact stats.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class index → clockwise degrees to rotate the image upright.
CLASS_DEGREES = [0, 90, 180, 270]

# Abstain (leave the map unrotated) when the top-class softmax prob is below this.
_DEFAULT_CONFIDENCE_THRESHOLD = 0.50

_CV2_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class RotationClassifier(torch.nn.Module):
    """ResNet50 with a 4-way rotation head, full fine-tune.

    ImageNet-classification pretraining, preserves the
    rotation-sensitive features this task needs.

    The ``backbone.*`` key layout matches the saved checkpoints, so a state_dict
    loads cleanly (a bare ResNet50 wouldn't). ``pretrained=True`` seeds the
    backbone with ImageNet weights for training; inference passes False and
    loads the fine-tuned checkpoint over it.
    """

    def __init__(self, n_classes: int = 4, pretrained: bool = False):
        super().__init__()
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = tv_models.resnet50(weights=weights)
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)


def make_transform(img_size: int, train: bool = False):
    """Rotation-classifier image preprocessing (numpy/tensor input via ToPILImage).

    train=True: RandomResizedCrop + ColorJitter + RandomErasing kill the
    case-identity shortcut, so the model must use rotation-discriminative
    features (text orientation, north arrows, scale-bar position) rather than
    memorising layouts. No horizontal flip — that would change the rotation
    label. train=False (inference / val): plain resize + ImageNet-normalise.
    """
    if train:
        return T.Compose(
            [
                T.ToPILImage(),
                T.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.85, 1.18), antialias=True),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.05),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                T.RandomErasing(p=0.4, scale=(0.02, 0.18), ratio=(0.3, 3.3), value=0),
            ]
        )
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize((img_size, img_size), antialias=True),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# Singleton state — load once on first call, keep cached.
_state_lock = threading.Lock()
_kfold_state: dict | None = None


def _load_kfold_state() -> Optional[dict]:
    """Load all available fold_K/best.pt checkpoints + fold_assignment.json.

    Returns the kfold state dict, or None if the k-fold dir is missing
    (the caller then raises — the fine-tuned adapters are required).
    Thread-safe singleton."""
    global _kfold_state
    if _kfold_state is not None:
        return _kfold_state
    with _state_lock:
        if _kfold_state is not None:
            return _kfold_state
        fold_assignment_path = FOLD_ASSIGNMENT
        if not fold_assignment_path.exists():
            return None
        try:
            fold_assignment = json.loads(fold_assignment_path.read_text())
        except Exception:
            return None

        device = _device()
        models: dict = {}
        # All folds train at one resolution, so any checkpoint's img_size is
        # authoritative for the single inference transform (default 768).
        img_size = 768
        for fold_dir in sorted(ROTATION_KFOLD_DIR.glob("fold_*")):
            checkpoint_path = fold_dir / "best.pt"
            if not checkpoint_path.exists():
                continue
            try:
                fold = int(fold_dir.name.split("_")[-1])
            except ValueError:
                continue
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            config = checkpoint.get("config") or {}
            img_size = int(config.get("img_size", 768))
            model = RotationClassifier(n_classes=int(config.get("n_classes", 4)))
            model.load_state_dict(checkpoint["state_dict"])
            model = model.to(device).eval()
            models[fold] = model
        if not models:
            return None

        _kfold_state = {
            "models": models,
            "device": device,
            "transform": make_transform(img_size),
            "fold_assignment": fold_assignment,
            "available_folds": set(models.keys()),
        }
        print(
            f"  rotation_classifier: loaded {len(models)} k-fold adapter(s) "
            f"from {ROTATION_KFOLD_DIR.name}/ "
            f"({len(fold_assignment)} cases routed via fold_assignment.json)"
        )
        return _kfold_state


def _model_for_case(case_name: Optional[str]) -> tuple[torch.nn.Module, dict]:
    """K-fold rotation model for case_name; an unknown or missing case_name
    falls back to min(available_folds) inside resolve_fold."""
    kfold_state = _load_kfold_state()
    if kfold_state is None:
        raise FileNotFoundError(
            f"rotation classifier k-fold dir not found at {ROTATION_KFOLD_DIR}. "
            f"Make sure you downloaded the k-fold checkpoints."
        )
    fold = _resolve_fold(
        case_name or "", kfold_state["fold_assignment"], kfold_state["available_folds"]
    )
    return kfold_state["models"][fold], kfold_state


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
    tta: bool = True,
) -> dict:
    """CW rotation (0/90/180/270) to make `map_bgr` upright.

    With ``tta=True`` (the deployed default) the classifier is run on the base
    view and its 90/180/270 CW rotations and the four softmaxes are averaged
    after realigning each to the original frame; ``tta=False`` uses the single
    base view. Abstains (returns 0) when the (averaged) top-class probability is
    below `_DEFAULT_CONFIDENCE_THRESHOLD`. Returns dict: rotation_cw_degrees,
    applied, confidence, abstained_low_confidence, raw_class, fold.
    """
    model, state = _model_for_case(case_name)
    device = state["device"]
    transform = state["transform"]
    fold = None
    if case_name is not None:
        fold = _resolve_fold(case_name, state["fold_assignment"], state["available_folds"])

    base_tensor = _preprocess(map_bgr, transform).unsqueeze(0).to(device)  # (1, 3, H, W)

    if tta:
        # CW → torch.rot90 (CCW) k-arg mapping.
        rot90_k_by_cw = {0: 0, 1: 3, 2: 2, 3: 1}
        ensemble_probs = torch.zeros(1, 4, device=device)
        for cw_views in (0, 1, 2, 3):
            view = (
                base_tensor
                if cw_views == 0
                else torch.rot90(base_tensor, rot90_k_by_cw[cw_views], dims=(-2, -1))
            )
            probs = F.softmax(model(view), dim=-1)
            # Convert back to original frame: rotated-frame class C' on an
            # input we rotated cw_views further CW corresponds to original class
            # C = (C' + cw_views) mod 4. In torch.roll semantics
            # (new[i] = old[(i - shifts) mod 4]), use shifts=cw_views.
            if cw_views != 0:
                probs = torch.roll(probs, shifts=cw_views, dims=-1)
            ensemble_probs = ensemble_probs + probs
        ensemble_probs = ensemble_probs / 4.0
    else:
        ensemble_probs = F.softmax(model(base_tensor), dim=-1)

    probs = ensemble_probs.squeeze(0).cpu().numpy().astype(float)
    top_class = int(np.argmax(probs))
    confidence = float(probs[top_class])

    abstained = confidence < _DEFAULT_CONFIDENCE_THRESHOLD
    rotation = 0 if abstained else CLASS_DEGREES[top_class]

    return {
        "rotation_cw_degrees": rotation,
        "applied": rotation != 0,
        "confidence": confidence,
        "abstained_low_confidence": abstained,
        "raw_class": top_class,
        "fold": fold,
    }


def auto_rotate(
    map_bgr: np.ndarray,
    case_name: Optional[str] = None,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """Predict + apply rotation. Returns (rotated_map, info_dict).

    The returned map is the input rotated CW by the predicted amount
    (or unchanged if abstained / class 0). info_dict is the same as
    predict_rotation_with_confidence's return. Pass `case_name` to route
    the prediction through k-fold (excludes the case from training).
    """
    rotation_info = predict_rotation_with_confidence(map_bgr, case_name=case_name)
    rotation_degrees = rotation_info["rotation_cw_degrees"]
    if rotation_degrees == 0:
        if verbose:
            if rotation_info["abstained_low_confidence"]:
                print(
                    f"  rotation_classifier: abstained "
                    f"(conf={rotation_info['confidence']:.2f} < "
                    f"{_DEFAULT_CONFIDENCE_THRESHOLD:.2f}); "
                    f"raw_class={rotation_info['raw_class']} -> "
                    f"{CLASS_DEGREES[rotation_info['raw_class']]}°. "
                    f"Leaving map unrotated."
                )
            else:
                print(
                    f"  rotation_classifier: 0° (already upright, "
                    f"conf={rotation_info['confidence']:.2f})"
                )
        return map_bgr, rotation_info
    rotated_map = cv2.rotate(map_bgr, _CV2_ROTATE_CODES[rotation_degrees])
    if verbose:
        fold_suffix = (
            f" fold={rotation_info['fold']}" if rotation_info.get("fold") is not None else ""
        )
        print(
            f"  rotation_classifier: rotating {rotation_degrees}° CW "
            f"(conf={rotation_info['confidence']:.2f}{fold_suffix})"
        )
    return rotated_map, rotation_info
