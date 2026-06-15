"""Boundary augmentations for SAM3 fine-tuning: style transfer (filled→outline) + copy-paste."""

import cv2
import numpy as np
import random
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

# Boundary styles
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
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


def _draw_dashed_outline(image, contours, color, thickness, dash_len=None):
    """Draw dashed outline along contours."""
    for contour in contours:
        total_pts = len(contour)
        if total_pts < 4:
            continue

        # Auto-compute dash length based on contour perimeter
        if dash_len is None:
            perimeter = cv2.arcLength(contour, True)
            dash_len_actual = max(8, int(perimeter / 60))
        else:
            dash_len_actual = dash_len
        gap_len = max(4, dash_len_actual // 2)

        i = 0
        drawing = True
        while i < total_pts:
            if drawing:
                end = min(i + dash_len_actual, total_pts)
                pts = contour[i:end]
                if len(pts) > 1:
                    cv2.polylines(image, [pts], False, color, thickness=thickness)
            else:
                end = min(i + gap_len, total_pts)
            i = end
            drawing = not drawing
    return image


def _draw_dotted_outline(image, contours, color, dot_radius=3, spacing=None):
    """Draw dotted outline along contours."""
    for contour in contours:
        total_pts = len(contour)
        if total_pts < 2:
            continue

        perimeter = cv2.arcLength(contour, True)
        if spacing is None:
            spacing_actual = max(8, int(perimeter / 80))
        else:
            spacing_actual = spacing

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
                accumulated += spacing_actual
            accumulated -= seg_len
    return image


def _draw_hatching(image, contours, mask, color, thickness=2, spacing=None):
    """Draw diagonal hatching inside the boundary region."""
    h, w = mask.shape[:2]
    if spacing is None:
        # Scale spacing based on mask area
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

    Applies blur, noise, uneven ink, and fade to the boundary pixels
    so they don't look artificially clean.
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
        # Where roughened differs from original, blend
        diff = cv2.absdiff(drawn_mask, final_mask)
        result[diff > 0] = image[diff > 0]

    return result


def style_transfer_augment(image_pil, mask_pil, p=0.5):
    """Apply style transfer augmentation to a boundary sample.

    Takes a filled boundary and converts it to a random style (outline,
    dashed, dotted, hatched) with a random color. The ground truth mask
    remains the filled interior.

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
    h, w = mask.shape[:2]

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


def copy_paste_augment(image_pil, mask_pil, donor_image_pil, donor_mask_pil, p=0.5):
    """Copy a boundary shape from a donor sample and paste onto this image.

    Takes the boundary contour from the donor sample, optionally transforms it
    (scale, style), and draws it onto the current image's map background.
    The ground truth mask is replaced with the donor's boundary (filled contour).

    Args:
        image_pil: PIL RGB image (recipient map background)
        mask_pil: PIL L-mode mask (recipient's original mask — will be replaced)
        donor_image_pil: PIL RGB image (donor — used only for contour extraction)
        donor_mask_pil: PIL L-mode mask (donor's boundary mask)
        p: probability of applying augmentation

    Returns:
        (augmented_image_pil, new_mask_pil)
    """
    if random.random() > p:
        return image_pil, mask_pil

    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    mask = np.array(mask_pil)
    donor_mask = np.array(donor_mask_pil)
    h, w = image_bgr.shape[:2]

    # Extract donor contours
    donor_contours = _extract_contours(donor_mask)
    if not donor_contours:
        return image_pil, mask_pil

    # Get bounding rect of donor boundary
    all_pts = np.vstack(donor_contours)
    dx, dy, dw, dh = cv2.boundingRect(all_pts)

    if dw < 10 or dh < 10:
        return image_pil, mask_pil

    faded_map = _fade_boundary(image_bgr, mask)

    # Scale the donor contour to 30-80% of the recipient image.
    target_fraction = random.uniform(0.3, 0.8)
    scale_w = (w * target_fraction) / dw
    scale_h = (h * target_fraction) / dh
    scale = min(scale_w, scale_h)

    # Scale contours
    scaled_contours = []
    for c in donor_contours:
        sc = c.copy().astype(np.float64)
        sc[:, :, 0] = (sc[:, :, 0] - dx) * scale
        sc[:, :, 1] = (sc[:, :, 1] - dy) * scale
        scaled_contours.append(sc.astype(np.int32))

    # Get new bounding rect after scaling
    all_scaled = np.vstack(scaled_contours)
    _, _, sw, sh = cv2.boundingRect(all_scaled)

    # Random position (ensure it fits)
    max_x = max(0, w - sw - 10)
    max_y = max(0, h - sh - 10)
    offset_x = random.randint(10, max(10, max_x))
    offset_y = random.randint(10, max(10, max_y))

    # Translate contours
    for c in scaled_contours:
        c[:, :, 0] += offset_x
        c[:, :, 1] += offset_y

    new_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(new_mask, scaled_contours, -1, 255, thickness=-1)

    # Check mask is reasonable
    mask_pct = np.sum(new_mask > 0) / (h * w) * 100
    if mask_pct < 0.1 or mask_pct > 60:
        return image_pil, mask_pil

    color = random.choice(BOUNDARY_COLORS_BGR)
    style = random.choice(BOUNDARY_STYLES)
    result = faded_map.copy()
    drawn_mask = np.zeros((h, w), dtype=np.uint8)

    if style == "solid_outline":
        thickness = random.randint(3, 10)
        result = _draw_solid_outline(result, scaled_contours, color, thickness)
        cv2.drawContours(drawn_mask, scaled_contours, -1, 255, thickness=thickness)

    elif style == "thick_outline":
        thickness = random.randint(10, 20)
        result = _draw_solid_outline(result, scaled_contours, color, thickness)
        cv2.drawContours(drawn_mask, scaled_contours, -1, 255, thickness=thickness)

    elif style == "thin_outline":
        thickness = random.randint(1, 3)
        result = _draw_solid_outline(result, scaled_contours, color, thickness)
        cv2.drawContours(drawn_mask, scaled_contours, -1, 255, thickness=thickness)

    elif style == "dashed":
        thickness = random.randint(2, 8)
        before = result.copy()
        result = _draw_dashed_outline(result, scaled_contours, color, thickness)
        drawn_mask = (np.any(result != before, axis=2) * 255).astype(np.uint8)

    elif style == "dotted":
        dot_radius = random.randint(2, 5)
        before = result.copy()
        result = _draw_dotted_outline(result, scaled_contours, color, dot_radius)
        drawn_mask = (np.any(result != before, axis=2) * 255).astype(np.uint8)

    elif style == "filled_transparent":
        alpha = random.uniform(0.2, 0.5)
        fill_overlay = result.copy()
        cv2.drawContours(fill_overlay, scaled_contours, -1, color, thickness=-1)
        result = cv2.addWeighted(result, 1 - alpha, fill_overlay, alpha, 0)
        outline_thickness = random.randint(1, 4)
        cv2.drawContours(result, scaled_contours, -1, color, thickness=outline_thickness)
        cv2.drawContours(drawn_mask, scaled_contours, -1, 255, thickness=-1)

    elif style == "filled_opaque":
        cv2.drawContours(result, scaled_contours, -1, color, thickness=-1)
        cv2.drawContours(drawn_mask, scaled_contours, -1, 255, thickness=-1)

    elif style == "hatched":
        hatch_thickness = random.randint(1, 3)
        before = result.copy()
        result = _draw_hatching(result, scaled_contours, new_mask, color, thickness=hatch_thickness)
        drawn_mask = (np.any(result != before, axis=2) * 255).astype(np.uint8)

    # Roughen to look like a real scan
    result = _roughen_boundary(result, drawn_mask)

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb), Image.fromarray(new_mask)


# Quick visual test: writes augmented samples for eyeballing.
if __name__ == "__main__":
    import os
    from pathlib import Path

    data_dir = Path(__file__).parent.parent / "boundary_annotation_dataset"
    out_dir = Path(__file__).parent.parent / "results" / "augment_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    maps_dir = data_dir / "maps"
    masks_dir = data_dir / "boundary_masks"

    files = sorted(os.listdir(maps_dir))

    # Test style transfer on first 5 samples, 3 variants each
    print("=== Style Transfer Augmentation ===")
    for fname in files[:5]:
        img = Image.open(maps_dir / fname).convert("RGB")
        mask = Image.open(masks_dir / fname).convert("L")

        for v in range(4):
            aug_img, aug_mask = style_transfer_augment(img, mask, p=1.0)
            aug_bgr = cv2.cvtColor(np.array(aug_img), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_dir / f"style_{fname[:-4]}_v{v}.png"), aug_bgr)
        print(f"  {fname}: 4 variants saved")

    # Test copy-paste
    print("\n=== Copy-Paste Augmentation ===")
    for i in range(5):
        recipient_idx = i
        donor_idx = (i + 3) % len(files)

        img = Image.open(maps_dir / files[recipient_idx]).convert("RGB")
        mask = Image.open(masks_dir / files[recipient_idx]).convert("L")
        donor_img = Image.open(maps_dir / files[donor_idx]).convert("RGB")
        donor_mask = Image.open(masks_dir / files[donor_idx]).convert("L")

        for v in range(3):
            aug_img, aug_mask = copy_paste_augment(img, mask, donor_img, donor_mask, p=1.0)
            aug_bgr = cv2.cvtColor(np.array(aug_img), cv2.COLOR_RGB2BGR)
            aug_m = np.array(aug_mask)
            cv2.imwrite(
                str(out_dir / f"cp_{files[recipient_idx][:-4]}_d{donor_idx}_v{v}.png"), aug_bgr
            )
            cv2.imwrite(
                str(out_dir / f"cp_{files[recipient_idx][:-4]}_d{donor_idx}_v{v}_mask.png"), aug_m
            )
        print(f"  {files[recipient_idx]} + donor {files[donor_idx]}: 3 variants saved")

    print(f"\nAll saved to {out_dir}")
