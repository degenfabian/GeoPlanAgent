"""PDF and page handling: PyMuPDF rendering (full MediaBox at 200 DPI),
evaluation-case PDF resolution, and worker map-page preparation including
the k-fold ResNet50 auto-rotation classifier with 4-way TTA.
"""

from __future__ import annotations

import cv2
import numpy as np
from pdf2image import convert_from_path
from typing import Optional, Tuple
from pathlib import Path
import json
import threading
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T
from geoplanagent.utils import resolve_fold as _resolve_fold


def render_pdf_page(pdf_path: str, page_index: int, dpi: int = 200) -> Optional[np.ndarray]:
    """Render a single PDF page as a numpy BGR image at full resolution.

    Uses PyMuPDF (fitz) for fast rendering. Falls back to pdf2image when
    fitz isn't available. Raises IndexError if page_index is out of range.
    """
    try:
        import fitz

        document = fitz.open(pdf_path)
        try:
            if page_index < 0 or page_index >= len(document):
                raise IndexError(
                    f"page_index {page_index} out of range (PDF has {len(document)} pages)"
                )
            page = document[page_index]
            # Force the full MediaBox to be rendered. By default PyMuPDF
            # honours the page's CropBox, which on some PDFs is set a few
            # points smaller than the MediaBox and silently clips real map
            # content at the edges (e.g. case 3DA282…: cropbox 595×841 vs
            # mediabox 603×847, losing ~11 px on each side of the planning
            # map). set_cropbox goes through the standard rotation pipeline
            # and is a no-op when cropbox already equals mediabox.
            #
            # Some PDFs have a MediaBox in a different coordinate space than
            # the CropBox (e.g. case 5FA84190 page 6: media=(0,-1920,864,0),
            # crop=(0,0,864,1920) — Y inverted). PyMuPDF rejects with
            # "CropBox not in MediaBox" when the rects don't overlap. In
            # that case the existing CropBox is already correct (matches
            # the page's effective rect); fall through and render that.
            try:
                page.set_cropbox(page.mediabox)
            except ValueError:
                pass
            pixmap = page.get_pixmap(dpi=dpi)
            rgb_image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n
            )
        finally:
            document.close()
        if rgb_image.shape[2] == 4:
            return cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    except ImportError:
        pages = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_index + 1,
            last_page=page_index + 1,
        )
        if not pages:
            return None
        return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)


def render_map_page(
    pdf_path: str,
    page_1based: int,
    dpi: int = 200,
    verbose: bool = False,
    case_name: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, dict]]:
    """Render one page of a planning PDF into the canonical working image.

    Pipeline:
      1. fitz render at the requested DPI
      2. auto_rotate via the trained ResNet50 classifier (no-op if
         confidence is below threshold). When ``case_name`` is given AND
         a k-fold rotation checkpoint dir is available, the case is
         routed to the fold that did NOT see it during training.

    Args:
        pdf_path: path to the PDF.
        page_1based: 1-based page number to render.
        dpi: render DPI (default 200).
        verbose: pass through to auto_rotate's logger.
        case_name: optional case identifier for k-fold rotation routing.

    Returns:
        (map_bgr, rotation_info) on success, or None if rendering failed
        (e.g. page index out of range). rotation_info is the dict returned by
        auto_rotate — the caller can read rotation_info["applied"] to know
        whether rotation was performed.
    """
    page_index = max(0, int(page_1based) - 1)
    try:
        map_bgr = render_pdf_page(str(pdf_path), page_index, dpi=dpi)
    except IndexError:
        return None
    if map_bgr is None:
        return None

    rotation_info: dict = {"applied": False}
    try:
        map_bgr, rotation_info = auto_rotate(map_bgr, case_name=case_name, verbose=verbose)
    except Exception as error:
        if verbose:
            print(f"  rotation_classifier failed ({error!s:.80}); raw render")

    return map_bgr, rotation_info


# Filename tokens that hint at a dedicated map/plan PDF (vs. notice or
# form documents in the same folder). The first PDF whose lowercase
# name contains any of these tokens wins. "plan" catches cases like
# A4Da2 where one file is "..._Direction_Plan.pdf" and another is a
# notice.
_MAP_TOKENS = ("map", "plan")


def resolve_case_pdf(folder_path: Path) -> Optional[Path]:
    """Pick the canonical PDF for a single evaluation case folder.

    Prefers PDFs whose filename contains 'map' or 'plan' (case-insensitive);
    falls back to the first PDF in the folder if none match.

    Returns ``None`` if the folder doesn't exist or has no PDFs.
    """
    if not folder_path.is_dir():
        return None
    pdf_files = list(folder_path.glob("*.pdf"))
    if not pdf_files:
        return None
    map_pdfs = [
        pdf_file
        for pdf_file in pdf_files
        if any(token in pdf_file.name.lower() for token in _MAP_TOKENS)
    ]
    return map_pdfs[0] if map_pdfs else pdf_files[0]


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
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)


def _device() -> torch.device:
    return torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def _make_transform(img_size: int) -> T.Compose:
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize((img_size, img_size), antialias=True),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


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
                f"Train it via training/train_rotation_classifier.py."
            )
        checkpoint = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
        config = checkpoint.get("config") or {}
        img_size = int(config.get("img_size", 768))

        device = _device()
        model = _RotationClassifier(n_classes=int(config.get("n_classes", 4)))
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device).eval()

        _state = {
            "models": {None: model},  # None = "any case" (legacy)
            "device": device,
            "img_size": img_size,
            "transform": _make_transform(img_size),
            "kind": "legacy",
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
        fold_assignment_path = _KFOLD_DIR / "fold_assignment.json"
        if not fold_assignment_path.exists():
            return None
        try:
            fold_assignment = json.loads(fold_assignment_path.read_text())
        except Exception:
            return None

        device = _device()
        models: dict = {}
        # Per-fold img_size: detect inconsistency rather than silently using
        # whichever fold loaded last.
        img_size_by_fold: dict[int, int] = {}
        for fold_dir in sorted(_KFOLD_DIR.glob("fold_*")):
            checkpoint_path = fold_dir / "best.pt"
            if not checkpoint_path.exists():
                continue
            try:
                fold = int(fold_dir.name.split("_")[-1])
            except ValueError:
                continue
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            config = checkpoint.get("config") or {}
            img_size_by_fold[fold] = int(config.get("img_size", 768))
            model = _RotationClassifier(n_classes=int(config.get("n_classes", 4)))
            model.load_state_dict(checkpoint["state_dict"])
            model = model.to(device).eval()
            models[fold] = model
        if not models:
            return None

        # Use the modal img_size; warn on mismatch.
        fold_img_sizes = list(img_size_by_fold.values())
        img_size = max(set(fold_img_sizes), key=fold_img_sizes.count)
        mismatched = {
            fold: size for fold, size in img_size_by_fold.items() if size != img_size
        }
        if mismatched:
            print(
                f"  rotation_classifier: WARNING — fold img_size mismatch "
                f"(folds with non-default img_size: {mismatched}). Using "
                f"img_size={img_size} for all folds; mismatched folds may "
                f"see degraded accuracy because their training resolution "
                f"differs from the inference transform."
            )

        _kfold_state = {
            "models": models,
            "device": device,
            "img_size": img_size,
            "transform": _make_transform(img_size),
            "fold_assignment": fold_assignment,
            "kind": "kfold",
            "available_folds": set(models.keys()),
        }
        print(
            f"  rotation_classifier: loaded {len(models)} k-fold adapter(s) "
            f"from {_KFOLD_DIR.name}/ "
            f"({len(fold_assignment)} cases routed via fold_assignment.json)"
        )
        return _kfold_state


def _model_for_case(case_name: Optional[str]) -> tuple[torch.nn.Module, dict]:
    """K-fold model for case_name, or the legacy single checkpoint."""
    if case_name is not None:
        kfold_state = _load_kfold_state()
        if kfold_state is not None:
            fold = _resolve_fold(
                case_name, kfold_state["fold_assignment"], kfold_state["available_folds"]
            )
            return kfold_state["models"][fold], kfold_state
    # Legacy path
    legacy_state = _load_state()
    return legacy_state["models"][None], legacy_state


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
) -> dict:
    """CW rotation (0/90/180/270) to make `map_bgr` upright, with 4-view TTA.

    Abstains (returns 0) when the top-class softmax prob is below
    `_DEFAULT_CONFIDENCE_THRESHOLD`.
    Returns dict: rotation_cw_degrees, applied, confidence,
    abstained_low_confidence, raw_class, fold.
    """
    model, state = _model_for_case(case_name)
    device = state["device"]
    transform = state["transform"]
    fold = None
    if state["kind"] == "kfold" and case_name is not None:
        fold = _resolve_fold(case_name, state["fold_assignment"], state["available_folds"])

    base_tensor = _preprocess(map_bgr, transform).unsqueeze(0).to(device)  # (1, 3, H, W)

    # CW → torch.rot90 (CCW) k-arg mapping.
    rot90_k_by_cw = {0: 0, 1: 3, 2: 2, 3: 1}

    ensemble_probs = torch.zeros(1, 4, device=device)
    for cw_views in (0, 1, 2, 3):
        view = (
            base_tensor
            if cw_views == 0
            else torch.rot90(base_tensor, rot90_k_by_cw[cw_views], dims=(-2, -1))
        )
        logits = model(view)
        probs = F.softmax(logits, dim=-1)
        # Convert back to original frame: rotated-frame class C' on an
        # input we rotated cw_views further CW corresponds to original class
        # C = (C' + cw_views) mod 4. In torch.roll semantics
        # (new[i] = old[(i - shifts) mod 4]), use shifts=cw_views.
        if cw_views != 0:
            probs = torch.roll(probs, shifts=cw_views, dims=-1)
        ensemble_probs = ensemble_probs + probs
    ensemble_probs = ensemble_probs / 4.0

    probs = ensemble_probs.squeeze(0).cpu().numpy().astype(float)
    top_class = int(np.argmax(probs))
    confidence = float(probs[top_class])

    abstained = confidence < _DEFAULT_CONFIDENCE_THRESHOLD
    rotation = 0 if abstained else _CLASS_TO_DEGREES[top_class]

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
                    f"{_CLASS_TO_DEGREES[rotation_info['raw_class']]}°. "
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
