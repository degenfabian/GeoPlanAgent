"""Style-transfer augmentation for SAM3 fine-tuning: redraws the filled
ground-truth boundary in a random annotation style (solid/dashed/dotted
outline, hatching, or a recoloured fill) and roughens it to look scanned;
the ground-truth mask is unchanged. Used by train_sam3_kfold.py.

Run directly to write example augmentations for eyeballing:

    uv run python training/boundary_augmentations.py
"""

import random

import cv2
import numpy as np
from PIL import Image


# BGR palette for boundary recoloring.
BOUNDARY_COLORS_BGR = [
    (0, 0, 255),  # red
    (0, 0, 200),  # dark red
    (0, 0, 150),  # maroon
    (30, 30, 30),  # black
    (60, 60, 60),  # dark gray
    (255, 0, 0),  # blue
    (0, 128, 0),  # green
    (0, 100, 0),  # dark green
    (200, 0, 200),  # magenta
    (0, 140, 255),  # orange
    (100, 50, 50),  # dark blue
]

BOUNDARY_STYLES = [
    "solid_outline",
    "thick_outline",
    "thin_outline",
    "dashed",
    "dotted",
    "filled_transparent",  # semi-transparent fill (like the original but diff color)
    "filled_opaque",  # opaque fill
    "hatched",  # diagonal hatching
]


def _extract_contours(mask):
    """Extract contours from a binary mask."""
    binary = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _fade_boundary(image_bgr, mask):
    """Fade/desaturate the existing boundary fill without trying to remove it.

    Instead of full inpainting (which looks weird), we just reduce the
    color saturation and blend the masked region towards the local mean.
    This leaves a subtle ghost of the original that looks natural, then
    new boundary styles are drawn on top.
    """
    result = image_bgr.copy()
    binary = (mask > 127).astype(np.uint8)
    if np.sum(binary) == 0:
        return result

    # Convert to HSV, kill saturation in masked region
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Reduce saturation by 70-100%
    fade_amount = random.uniform(0.7, 1.0)
    hsv[:, :, 1] = hsv[:, :, 1] * (1.0 - binary.astype(np.float32) * fade_amount)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Blend masked pixels towards local background mean
    # Get mean color of pixels just outside the mask
    kernel = np.ones((25, 25), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    ring = cv2.bitwise_and(dilated, cv2.bitwise_not(mask))
    ring_pixels = image_bgr[ring > 127]
    if len(ring_pixels) > 50:
        bg_mean = ring_pixels.mean(axis=0)
        alpha = random.uniform(0.3, 0.7)
        mask_float = binary[:, :, None].astype(np.float32)
        bg_layer = np.full_like(result, bg_mean, dtype=np.float32)
        result = (
            result.astype(np.float32) * (1 - mask_float * alpha) + bg_layer * mask_float * alpha
        ).astype(np.uint8)

    return result


def _draw_solid_outline(image, contours, color, thickness):
    """Draw solid outline on image."""
    cv2.drawContours(image, contours, -1, color, thickness=thickness)
    return image


def _draw_dashed_outline(image, contours, color, thickness):
    """Draw dashed outline along contours."""
    for contour in contours:
        total_pts = len(contour)
        if total_pts < 4:
            continue

        # Dash length scales with the contour perimeter
        perimeter = cv2.arcLength(contour, True)
        dash_len = max(8, int(perimeter / 60))
        gap_len = max(4, dash_len // 2)

        i = 0
        drawing = True
        while i < total_pts:
            if drawing:
                end = min(i + dash_len, total_pts)
                pts = contour[i:end]
                if len(pts) > 1:
                    cv2.polylines(image, [pts], False, color, thickness=thickness)
            else:
                end = min(i + gap_len, total_pts)
            i = end
            drawing = not drawing
    return image


def _draw_dotted_outline(image, contours, color, dot_radius):
    """Draw dotted outline along contours."""
    for contour in contours:
        total_pts = len(contour)
        if total_pts < 2:
            continue

        perimeter = cv2.arcLength(contour, True)
        spacing = max(8, int(perimeter / 80))

        # Walk along contour at regular intervals
        accumulated = 0.0
        for i in range(1, total_pts):
            p1 = contour[i - 1][0].astype(float)
            p2 = contour[i][0].astype(float)
            seg_len = np.linalg.norm(p2 - p1)

            while accumulated < seg_len:
                t = accumulated / max(seg_len, 1e-6)
                pt = (p1 + t * (p2 - p1)).astype(int)
                cv2.circle(image, tuple(pt), dot_radius, color, -1)
                accumulated += spacing
            accumulated -= seg_len
    return image


def _draw_hatching(image, contours, mask, color, thickness):
    """Draw diagonal hatching inside the boundary region."""
    h, w = mask.shape[:2]
    # Spacing scales with the mask area
    area = np.sum(mask > 127)
    spacing = max(8, int(np.sqrt(area) / 8))

    # Create hatching pattern
    hatch_mask = np.zeros((h, w), dtype=np.uint8)

    # Diagonal lines at 45 degrees
    angle = random.choice([45, -45, 30, -30, 60, -60])
    max_dim = max(h, w) * 2

    for offset in range(-max_dim, max_dim, spacing):
        if angle == 45 or angle == -45:
            sign = 1 if angle == 45 else -1
            pt1 = (offset, 0)
            pt2 = (offset + sign * max_dim, max_dim)
        elif angle == 30 or angle == -30:
            sign = 1 if angle == 30 else -1
            pt1 = (offset, 0)
            pt2 = (offset + sign * max_dim // 2, max_dim)
        else:  # 60 or -60
            sign = 1 if angle == 60 else -1
            pt1 = (offset, 0)
            pt2 = (offset + sign * max_dim * 2, max_dim)
        cv2.line(hatch_mask, pt1, pt2, 255, thickness=thickness)

    # Mask hatching to boundary region
    boundary_mask = (mask > 127).astype(np.uint8) * 255
    hatch_in_boundary = cv2.bitwise_and(hatch_mask, boundary_mask)

    # Draw hatching on image
    image[hatch_in_boundary > 0] = color

    # Also draw the outline
    cv2.drawContours(image, contours, -1, color, thickness=thickness)
    return image


def _roughen_boundary(image, drawn_mask):
    """Make a drawn boundary look like a real scanned map annotation.

    Applies blur, noise, fade, and uneven-edge roughening to the boundary
    pixels so they don't look artificially clean.
    """
    if np.sum(drawn_mask > 0) == 0:
        return image

    result = image.copy()
    mask_bool = drawn_mask > 0

    # 1. Slight Gaussian blur on boundary pixels (simulates scan blur)
    sigma = random.uniform(0.5, 2.0)
    blurred = cv2.GaussianBlur(result, (0, 0), sigma)
    # Dilate mask slightly so blur bleeds naturally
    blur_mask = cv2.dilate(drawn_mask, np.ones((3, 3), np.uint8), iterations=1)
    blur_float = (blur_mask > 0).astype(np.float32)
    blur_float = cv2.GaussianBlur(blur_float, (5, 5), 1.5)
    result = (blurred * blur_float[:, :, None] + result * (1 - blur_float[:, :, None])).astype(
        np.uint8
    )

    # 2. Add noise to boundary pixels (scan artifacts)
    if random.random() > 0.3:
        noise = np.random.normal(0, random.uniform(5, 20), result.shape).astype(np.float32)
        noisy = np.clip(result.astype(np.float32) + noise, 0, 255)
        result[mask_bool] = noisy[mask_bool].astype(np.uint8)

    # 3. Random fade/lighten (old ink, faded boundary)
    if random.random() > 0.4:
        fade = random.uniform(0.5, 0.9)
        faded = (result.astype(np.float32) * fade + 255.0 * (1 - fade)).astype(np.uint8)
        result[mask_bool] = faded[mask_bool]

    # 4. Slight morphological roughening (uneven edges)
    if random.random() > 0.5:
        small_kernel = np.ones((2, 2), np.uint8)
        if random.random() > 0.5:
            roughened = cv2.erode(drawn_mask, small_kernel, iterations=1)
        else:
            roughened = cv2.dilate(drawn_mask, small_kernel, iterations=1)
        # Only apply to random patches (not uniform)
        noise_mask = (np.random.random(drawn_mask.shape) > 0.3).astype(np.uint8)
        final_mask = cv2.bitwise_and(roughened, noise_mask * 255)
        # Where the roughened mask differs from the drawn one, restore the
        # input pixels, so the edge degrades patchily rather than uniformly
        diff = cv2.absdiff(drawn_mask, final_mask)
        result[diff > 0] = image[diff > 0]

    return result


def style_transfer_augment(image_pil, mask_pil, p=0.5):
    """Apply style transfer augmentation to a boundary sample.

    Takes a filled boundary and converts it to a random style (solid, dashed,
    or dotted outline, hatching, or a re-coloured fill) with a random color.
    The ground truth mask remains the filled interior.

    Args:
        image_pil: PIL RGB image with boundary annotation
        mask_pil: PIL L-mode binary mask (filled boundary region)
        p: probability of applying augmentation

    Returns:
        (augmented_image_pil, mask_pil) — mask is unchanged
    """
    if random.random() > p:
        return image_pil, mask_pil

    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    mask = np.array(mask_pil)

    # Check mask has content
    if np.sum(mask > 127) < 100:
        return image_pil, mask_pil

    contours = _extract_contours(mask)
    if not contours:
        return image_pil, mask_pil

    # Fade rather than fully remove the existing boundary fill.
    faded_map = _fade_boundary(image_bgr, mask)

    style = random.choice(BOUNDARY_STYLES)
    color = random.choice(BOUNDARY_COLORS_BGR)

    result = faded_map.copy()

    # Track which pixels we drew so we can roughen them
    drawn_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

    if style == "solid_outline":
        thickness = random.randint(3, 10)
        result = _draw_solid_outline(result, contours, color, thickness)
        cv2.drawContours(drawn_mask, contours, -1, 255, thickness=thickness)

    elif style == "thick_outline":
        thickness = random.randint(10, 20)
        result = _draw_solid_outline(result, contours, color, thickness)
        cv2.drawContours(drawn_mask, contours, -1, 255, thickness=thickness)

    elif style == "thin_outline":
        thickness = random.randint(1, 3)
        result = _draw_solid_outline(result, contours, color, thickness)
        cv2.drawContours(drawn_mask, contours, -1, 255, thickness=thickness)

    elif style == "dashed":
        thickness = random.randint(2, 8)
        before = result.copy()
        result = _draw_dashed_outline(result, contours, color, thickness)
        drawn_mask = (np.any(result != before, axis=2) * 255).astype(np.uint8)

    elif style == "dotted":
        dot_radius = random.randint(2, 5)
        before = result.copy()
        result = _draw_dotted_outline(result, contours, color, dot_radius)
        drawn_mask = (np.any(result != before, axis=2) * 255).astype(np.uint8)

    elif style == "filled_transparent":
        # Semi-transparent fill in new color
        alpha = random.uniform(0.2, 0.5)
        fill_overlay = result.copy()
        cv2.drawContours(fill_overlay, contours, -1, color, thickness=-1)
        result = cv2.addWeighted(result, 1 - alpha, fill_overlay, alpha, 0)
        outline_thickness = random.randint(1, 4)
        cv2.drawContours(result, contours, -1, color, thickness=outline_thickness)
        cv2.drawContours(drawn_mask, contours, -1, 255, thickness=-1)

    elif style == "filled_opaque":
        cv2.drawContours(result, contours, -1, color, thickness=-1)
        cv2.drawContours(drawn_mask, contours, -1, 255, thickness=-1)

    elif style == "hatched":
        hatch_thickness = random.randint(1, 3)
        before = result.copy()
        result = _draw_hatching(result, contours, mask, color, thickness=hatch_thickness)
        drawn_mask = (np.any(result != before, axis=2) * 255).astype(np.uint8)

    # Roughen the drawn boundary to mimic scan artefacts.
    result = _roughen_boundary(result, drawn_mask)

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb), mask_pil


# Quick visual test: writes augmented samples for eyeballing.
if __name__ == "__main__":
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from geoplanagent.paths import RESULTS_DIR, TRAINING_DATASET_DIR

    out_dir = RESULTS_DIR / "augment_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    maps_dir = TRAINING_DATASET_DIR / "maps"
    masks_dir = TRAINING_DATASET_DIR / "boundary_masks"

    files = sorted(os.listdir(maps_dir))

    # Test style transfer on first 5 samples, 4 variants each
    print("=== Style Transfer Augmentation ===")
    for fname in files[:5]:
        img = Image.open(maps_dir / fname).convert("RGB")
        mask = Image.open(masks_dir / fname).convert("L")

        for variant in range(4):
            aug_img, _ = style_transfer_augment(img, mask, p=1.0)
            aug_bgr = cv2.cvtColor(np.array(aug_img), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"style_{fname[:-4]}_v{variant}.png"), aug_bgr)
        print(f"  {fname}: 4 variants saved")

    print(f"\nAll saved to {out_dir}")
