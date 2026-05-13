"""Shared binary-mask cleanup primitives used by SAM3 post-processing.

Extracted 2026-05-11 from `tools/matching.py` and `tools/sam3_boundary.py`.
The four primitives are called in a chain by `mask_to_geojson_affine`:

    keep_dominant_components(mask)   # drop noise blobs
        -> expand_thin_mask(mask)    # thicken hollow outlines
        -> fill_mask_holes(mask)     # plug interior gaps

`cleanup_mask_pipeline(mask)` runs that chain end-to-end as a single call.

`tools/matching.py` re-exports these with underscore-prefixed aliases
(`_fill_mask_holes`, `_expand_thin_mask`, `_keep_dominant_components`) so
existing internal imports keep working.
"""

from __future__ import annotations

import cv2
import numpy as np


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill internal holes in a binary mask.

    SAM3 often returns masks with road/text-shaped gaps inside the boundary,
    producing many small fragmented polygons instead of one solid one. This
    morphologically closes small gaps and then fills remaining holes by
    finding external contours and drawing them filled.

    Returns a cleaned mask where small internal holes are filled.
    """
    binary = (mask > 0).astype(np.uint8)

    # Morphological close to bridge small gaps (roads, text)
    h, w = binary.shape[:2]
    # Scale kernel to image size: ~1% of the smaller dimension
    kernel_size = max(5, min(31, min(h, w) // 100))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find external contours and fill them to eliminate interior holes
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    filled = np.zeros_like(binary)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)

    return (filled * 255).astype(np.uint8)


def expand_thin_mask(mask: np.ndarray) -> np.ndarray:
    """Expand a thin outline mask into a filled region.

    SAM3 often returns boundary outlines rather than filled areas. This
    detects thin masks (low fill ratio relative to bounding box) and applies
    aggressive dilation to produce a solid region.

    Strategy:
    - If mask fill < 10% of its bounding box area: dilate aggressively
    - Otherwise: return as-is (already a filled region)
    """
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape[:2]
    total = h * w
    fill_pct = np.sum(binary > 0) / total

    if fill_pct < 0.001 or fill_pct > 0.05:
        return mask  # Too sparse (noise) or already reasonably filled

    # Check if the mask is thin by comparing fill area to bounding box area
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    # Get bounding rect of all contours
    all_pts = np.vstack(contours)
    x, y, bw, bh = cv2.boundingRect(all_pts)
    bbox_area = bw * bh
    if bbox_area == 0:
        return mask

    fill_vs_bbox = np.sum(binary > 0) / bbox_area
    if fill_vs_bbox > 0.10:
        return mask  # Already reasonably filled within its bbox

    # Thin outline detected — dilate to fill
    # Scale kernel to ~1.5% of smaller bbox dimension
    kernel_size = max(7, min(31, min(bw, bh) // 60))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernel_size, kernel_size))
    dilated = cv2.dilate(binary, kernel, iterations=3)

    # Fill holes in the dilated result
    contours_d, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    if contours_d:
        filled = np.zeros_like(binary)
        cv2.drawContours(filled, contours_d, -1, 1, cv2.FILLED)
        result = (filled * 255).astype(np.uint8)
        new_fill = np.sum(result > 0) / total
        # Only use if expansion is reasonable (not too big)
        if new_fill < 0.5:
            print(f"  Mask expansion: {fill_pct*100:.1f}% -> {new_fill*100:.1f}% "
                  f"(thin outline detected, dilated)")
            return result

    return mask


def keep_dominant_components(
    mask: np.ndarray, min_frac_of_largest: float = 0.05,
) -> np.ndarray:
    """Drop tiny noise blobs from a SAM3 mask, keep the dominant region(s).

    SAM3 occasionally returns masks with one main blob plus dozens of tiny
    scattered noise components (e.g. v12 case 8FB7BF90: 1 component at 65k px
    + 77 components mostly under 500 px). The noise inflates the predicted
    polygon area without contributing to GT overlap, tanking precision.

    This filter keeps only connected components whose area is >= 5% of the
    largest component's area. For 8FB7's noise (2nd largest = 3.4% of
    largest), this drops all 77 noise blobs. For legitimate multi-region
    cases (e.g. 12:00126:ART4 where multiple sub-sites are genuinely
    separate at 20-80% of largest), all real regions are preserved.

    Tuned at 0.05 after empirical eval: 0.30 threshold caused 10 regressions
    on multi-region cases vs 7 wins. 0.05 keeps the 8FB7-class wins while
    preserving multi-region masks.

    Returns the cleaned binary mask, same dtype as input.
    """
    binary = (mask > 0).astype(np.uint8)
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n_lab <= 2:  # background + 0 or 1 component — nothing to filter
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]  # exclude background (label 0)
    largest = int(areas.max())
    # Threshold: 5% of largest, with absolute floor of 100 px (also stricter
    # than the contourArea<100 filter that already runs downstream).
    threshold = max(100, int(largest * min_frac_of_largest))
    keep = np.zeros_like(binary, dtype=np.uint8)
    n_kept, n_dropped = 0, 0
    for i in range(1, n_lab):
        if int(stats[i, cv2.CC_STAT_AREA]) >= threshold:
            keep[lab == i] = 1
            n_kept += 1
        else:
            n_dropped += 1
    if n_dropped > 0:
        print(f"  Mask cleanup: kept {n_kept} dominant component(s), "
              f"dropped {n_dropped} noise blob(s) (largest={largest}px, "
              f"threshold={threshold}px)")
    return (keep * 255).astype(np.uint8) if mask.dtype == np.uint8 else keep


def cleanup_mask_pipeline(
    mask: np.ndarray, min_frac_of_largest: float = 0.05,
) -> np.ndarray:
    """Run the standard SAM3 mask cleanup pipeline as a single call.

    Equivalent to:
        keep_dominant_components(mask)
            -> expand_thin_mask(...)
            -> fill_mask_holes(...)

    This is the same chain that `mask_to_geojson_affine` applies before
    contour extraction. Use this when you need a cleaned mask for some
    purpose other than GeoJSON conversion (e.g. visualisation, alternative
    consumers like a SAM-second-opinion comparator).

    Args:
        mask: Binary mask (uint8, anything > 0 treated as foreground).
        min_frac_of_largest: Threshold for keep_dominant_components.

    Returns:
        Cleaned uint8 mask.
    """
    cleaned = keep_dominant_components(mask, min_frac_of_largest=min_frac_of_largest)
    cleaned = expand_thin_mask(cleaned)
    cleaned = fill_mask_holes(cleaned)
    return cleaned
