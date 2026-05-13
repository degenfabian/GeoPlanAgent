# `tools/rotation_classifier.py`

**243 lines.** Detects whether a planning map is rotated 0/90/180/270° from
upright and returns the rotation needed to fix it. Uses a fine-tuned
ResNet50 with 4-rotation Test-Time Augmentation (TTA) and a confidence
threshold to avoid making bad calls. Replaces the pre-existing DocTR-based
orientation detector which was confidently wrong on planning-map content.

## Public API

| Function | Purpose |
|---|---|
| `predict_rotation_cw(map_bgr)` | int (0/90/180/270) — CW degrees to upright |
| `predict_rotation_with_confidence(...)` | full prediction dict with metadata |
| `auto_rotate(map_bgr, ...)` | rotate the image AND return rotation info |

## Module-level constants (lines 38-53)

- **`_CKPT_PATH`** — repo-relative path to `models/rotation_classifier/best.pt`.
- **`_DEFAULT_CONFIDENCE_THRESHOLD = 0.80`** — below this, the classifier
  abstains (returns 0 = "don't rotate"). Tuned to be safe — wrong rotation
  destroys downstream MINIMA matching, no rotation just leaves things as-is.
- **`_IMAGENET_MEAN/STD`** — standard ResNet50 normalisation. Matches the
  trainer exactly.
- **`_CLASS_TO_DEGREES`** — class 0/1/2/3 → 0/90/180/270 CW. Per the trainer.
- **`_CV2_ROTATE_CODES`** — OpenCV rotation flag for each angle.

## Singleton state (lines 56-58)

`_state` holds the loaded model + transform. Lazy-initialised on first call,
guarded by a lock so concurrent agent calls don't double-load the ResNet.
Saves ~3s of load time per call after the first.

## `_RotationClassifier(torch.nn.Module)` (lines 61-71)

A thin wrapper that mirrors the trainer's class. Has a single attribute
`backbone` (a ResNet50 with the final FC swapped for 4-class output). The
`backbone.*` prefix in checkpoint keys requires this wrapper to load
cleanly — without it, `state_dict` would mismatch.

## `_load_state()` (lines 78-115)

First-call setup:

1. Acquire lock (only one thread loads at a time).
2. Re-check `_state` inside the lock (double-checked locking — handles the
   race where two callers passed the first check simultaneously).
3. Load the checkpoint with `weights_only=False` (the ckpt embeds a config
   dict so `weights_only=True` wouldn't work).
4. Read config: `img_size` (typically 768) and `n_classes` (4).
5. Pick device: MPS > CUDA > CPU.
6. Build the model, load weights, move to device, set to eval mode.
7. Build a torchvision transform: resize → centre-crop → ToTensor →
   normalise. The size matches whatever the trainer used.
8. Cache everything in `_state`.

The first call takes ~2-3s; subsequent calls are instant.

## `_preprocess(map_bgr, transform)` (lines 118-126)

Standard image preprocessing chain:

1. BGR → RGB (OpenCV → PyTorch convention)
2. Convert to PIL Image (torchvision transforms expect PIL).
3. Apply the transform from `_state["transform"]` → tensor.
4. Add batch dimension and move to device.

## `predict_rotation_with_confidence(map_bgr, threshold=0.80, return_logits=False)` (lines 129-201)

The core inference function. Implements 4-rotation TTA:

1. **Run 4 predictions** — on the original image and its 90/180/270 CW
   rotations. (Lines 145-156)
2. **Cyclically shift each rotated prediction** so all four "see" the
   same canonical orientation. (Lines 159-170) Example: when we feed the
   model an image rotated 90° CW, its prediction for "input is at 0°" is
   actually "input is at 90°" — so we shift the probability vector by 1
   class.
3. **Average the 4 softmax distributions** — TTA reduces single-pass noise.
4. **Apply confidence threshold** — if max(probs) ≥ threshold, return that
   class; otherwise return 0 (abstain).
5. **Build the result dict** with all probabilities, the chosen class, the
   threshold, and an `abstained_low_confidence` flag.

Why TTA: a single forward pass on an ambiguous map can be 60/40 between
two rotations. Averaging 4 rotated views typically resolves that into 95/5
or pushes it below threshold (correctly abstaining instead of guessing).

## `predict_rotation_cw(map_bgr)` (lines 205-211)

Thin wrapper around `predict_rotation_with_confidence` that returns just
the CW degrees as an int. Used by callers who don't need the metadata.

## `auto_rotate(map_bgr, threshold=0.80, verbose=False)` (lines 214-243)

End-to-end "give me a corrected image" function. Returns
`(corrected_image, info_dict)`:

1. Get the prediction + confidence.
2. If `info["applied"]` (i.e. a non-zero rotation passed the threshold),
   apply it via `cv2.rotate(...)`. Otherwise return the original image
   unchanged.
3. Print a one-liner if `verbose=True` so benchmark logs show what
   happened ("rotation_classifier: 90° applied (conf 0.91)" or
   "rotation_classifier: abstained (conf 0.62)").

This is the function the agent's `render_page` tool calls.

## Why this design

**Why a confidence threshold?** Wrong rotation is much worse than no
rotation. A 180°-flipped map will fail MINIMA matching outright; a
correctly-oriented (but unrotated) map just runs MINIMA's internal rotation
search and finds the match anyway. So abstaining when uncertain is strictly
safer than always rotating.

**Why TTA?** The classifier was trained on per-image rotation prediction;
TTA at inference is ~free (same model, 4 forward passes) and consistently
reduces noise on borderline cases. The trainer uses TTA in eval too, so
production uses the same setup.

**Why a singleton with a lock?** Loading a ResNet50 takes 2-3s and ~100MB.
Doing it on every `auto_rotate` call would dominate a benchmark's runtime.
The lock is needed because tools may run in parallel during the agent loop.

**Why save the transform in `_state` instead of rebuilding?** The trainer
might use specific augmentations or sizes that aren't trivial to
reconstruct from config. Using the trainer's serialised transform exactly
guarantees train/inference consistency.
