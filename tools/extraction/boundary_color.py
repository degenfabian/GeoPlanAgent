"""Detect a planning-area boundary from coloured strokes drawn on the map.

Many UK planning maps have an explicit red (occasionally blue / magenta) line
drawn on top of the OS basemap. When SAM3 misfires we can fall back to a
colour threshold + contour fill, which projects through the cached affine
exactly like a SAM3 mask. Used as a fallback in `extract_boundary`.
"""
from __future__ import annotations
from typing import Iterable, Optional
import cv2
import numpy as np


# HSV ranges tuned on UK planning maps. Saturation/value floors filter out
# anti-aliasing edges around black ink.
_COLOR_RANGES: dict[str, list[tuple[int, int, int, int, int, int]]] = {
    # red wraps around 0/180 in HSV
    "red":     [(0, 12, 80, 60, 180, 255), (165, 180, 80, 60, 180, 255)],
    "blue":    [(95, 135, 80, 60, 180, 255)],
    "magenta": [(140, 170, 80, 60, 180, 255)],
}


def _color_mask(img_bgr: np.ndarray, color: str) -> np.ndarray:
    """Binary mask of pixels in the named colour band (255 inside, 0 outside)."""
    if color not in _COLOR_RANGES:
        raise ValueError(f"unknown color {color!r}; pick from {list(_COLOR_RANGES)}")
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    out = np.zeros(img_bgr.shape[:2], dtype=bool)
    for h_lo, h_hi, s_min, v_min, s_max, v_max in _COLOR_RANGES[color]:
        out |= (
            (hsv[:, :, 0] >= h_lo) & (hsv[:, :, 0] <= h_hi)
            & (hsv[:, :, 1] >= s_min) & (hsv[:, :, 1] <= s_max)
            & (hsv[:, :, 2] >= v_min) & (hsv[:, :, 2] <= v_max)
        )
    return (out.astype(np.uint8)) * 255


def _largest_contour_filled(binary: np.ndarray, min_area_px: int = 500) -> Optional[np.ndarray]:
    """Close gaps in the line, take the largest connected component, fill it."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(closed)
    if n_lab <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.max() < min_area_px:
        return None
    keep = ((lab == int(np.argmax(areas)) + 1)).astype(np.uint8)
    contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    out = np.zeros_like(keep)
    cv2.drawContours(out, [max(contours, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    return (out * 255).astype(np.uint8)


def extract_color_boundary(
    img_bgr: np.ndarray,
    colors: Iterable[str] = ("red", "blue", "magenta"),
    min_area_px: int = 500,
) -> Optional[np.ndarray]:
    """Try each colour in turn; return the first filled boundary mask found.

    Returns a uint8 mask (255 inside, 0 outside) suitable for
    `mask_to_geojson_affine`, or None if no colour produced a usable closed
    region. Self-gating: if no colour gives a sufficiently large blob the
    caller should fall back to SAM3 (which is the normal path).
    """
    for color in colors:
        binary = _color_mask(img_bgr, color)
        filled = _largest_contour_filled(binary, min_area_px=min_area_px)
        if filled is not None:
            return filled
    return None
