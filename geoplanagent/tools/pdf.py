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


def render_pdf_page(pdf_path, page_index, dpi=200):
    """Render a single PDF page as a numpy BGR image at full resolution.

    Uses PyMuPDF (fitz) for fast rendering. Falls back to pdf2image when
    fitz isn't available. Raises IndexError if page_index is out of range.
    """
    try:
        import fitz

        doc = fitz.open(pdf_path)
        try:
            if page_index < 0 or page_index >= len(doc):
                raise IndexError(f"page_index {page_index} out of range (PDF has {len(doc)} pages)")
            page = doc[page_index]
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
            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        finally:
            doc.close()
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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
        (img_bgr, rot_info) on success, or None if rendering failed
        (e.g. page index out of range). rot_info is the dict returned by
        auto_rotate — the caller can read rot_info["applied"] to know
        whether rotation was performed.
    """
    page_idx = max(0, int(page_1based) - 1)
    try:
        img = render_pdf_page(str(pdf_path), page_idx, dpi=dpi)
    except IndexError:
        return None
    if img is None:
        return None

    rot_info: dict = {"applied": False}
    try:
        img, rot_info = auto_rotate(img, case_name=case_name, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"  rotation_classifier failed ({e!s:.80}); raw render")

    return img, rot_info


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
    map_pdfs = [p for p in pdf_files if any(tok in p.name.lower() for tok in _MAP_TOKENS)]
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


def _make_transform(img_size: int):
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
        ckpt = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config") or {}
        img_size = int(cfg.get("img_size", 768))

        device = _device()
        model = _RotationClassifier(n_classes=int(cfg.get("n_classes", 4)))
        model.load_state_dict(ckpt["state_dict"])
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
        fa_path = _KFOLD_DIR / "fold_assignment.json"
        if not fa_path.exists():
            return None
        try:
            fa = json.loads(fa_path.read_text())
        except Exception:
            return None

        device = _device()
        models: dict = {}
        # Per-fold img_size: detect inconsistency rather than silently using
        # whichever fold loaded last.
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
            model = _RotationClassifier(n_classes=int(cfg.get("n_classes", 4)))
            model.load_state_dict(ckpt["state_dict"])
            model = model.to(device).eval()
            models[fold_k] = model
        if not models:
            return None

        # Use the modal img_size; warn on mismatch.
        sizes = list(per_fold_img_size.values())
        img_size = max(set(sizes), key=sizes.count)
        mismatched = {f: s for f, s in per_fold_img_size.items() if s != img_size}
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
            "fold_assignment": fa,
            "kind": "kfold",
            "available_folds": set(models.keys()),
        }
        print(
            f"  rotation_classifier: loaded {len(models)} k-fold adapter(s) "
            f"from {_KFOLD_DIR.name}/ "
            f"({len(fa)} cases routed via fold_assignment.json)"
        )
        return _kfold_state


def _model_for_case(case_name: Optional[str]) -> tuple[torch.nn.Module, dict]:
    """K-fold model for case_name, or the legacy single checkpoint."""
    if case_name is not None:
        kf = _load_kfold_state()
        if kf is not None:
            fold = _resolve_fold(case_name, kf["fold_assignment"], kf["available_folds"])
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

    base = _preprocess(map_bgr, transform).unsqueeze(0).to(device)  # (1, 3, H, W)

    # CW → torch.rot90 (CCW) k-arg mapping.
    aug_torch_k = {0: 0, 1: 3, 2: 2, 3: 1}

    ensemble = torch.zeros(1, 4, device=device)
    for k_cw in (0, 1, 2, 3):
        x = base if k_cw == 0 else torch.rot90(base, aug_torch_k[k_cw], dims=(-2, -1))
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
    info = predict_rotation_with_confidence(map_bgr, case_name=case_name)
    rot = info["rotation_cw_degrees"]
    if rot == 0:
        if verbose:
            if info["abstained_low_confidence"]:
                print(
                    f"  rotation_classifier: abstained "
                    f"(conf={info['confidence']:.2f} < "
                    f"{_DEFAULT_CONFIDENCE_THRESHOLD:.2f}); "
                    f"raw_class={info['raw_class']} -> "
                    f"{_CLASS_TO_DEGREES[info['raw_class']]}°. "
                    f"Leaving map unrotated."
                )
            else:
                print(f"  rotation_classifier: 0° (already upright, conf={info['confidence']:.2f})")
        return map_bgr, info
    rotated = cv2.rotate(map_bgr, _CV2_ROTATE_CODES[rot])
    if verbose:
        fold_str = f" fold={info['fold']}" if info.get("fold") is not None else ""
        print(
            f"  rotation_classifier: rotating {rot}° CW (conf={info['confidence']:.2f}{fold_str})"
        )
    return rotated, info
