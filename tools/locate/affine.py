"""Closed-form affine solver from OS grid-tick OCR.

When ≥3 OS grid refs are OCR'd from a page (typically in the margin tick
marks), solve a page-pixel → OSGB easting/northing affine in closed form,
then project the image centre to get a confidence-1.0 candidate centre.

This is the "analytical short-circuit" path: MINIMA is skipped entirely
when this succeeds.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from tools.locate.schemas import DirectAffine


# OSGB36 transformer (re-used per process)
_WGS_TO_OSGB = None


def _wgs84_to_osgb(lat: float, lon: float) -> Tuple[float, float]:
    """Convert (lat, lon) → (easting_m, northing_m) in OSGB36."""
    global _WGS_TO_OSGB
    if _WGS_TO_OSGB is None:
        from pyproj import Transformer
        _WGS_TO_OSGB = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    east, north = _WGS_TO_OSGB.transform(lon, lat)
    return east, north


def _collinearity_score(points: np.ndarray) -> float:
    """0..1 — higher means more collinear. Uses smallest SVD singular value ratio."""
    if len(points) < 3:
        return 1.0
    centred = points - points.mean(axis=0)
    _, s, _ = np.linalg.svd(centred, full_matrices=False)
    if s[0] < 1e-9:
        return 1.0
    return 1.0 - s[-1] / s[0]


def solve_affine_from_grid_ticks(
    ticks: List[Tuple[str, Tuple[float, float], Tuple[int, int]]],
) -> Optional[DirectAffine]:
    """Solve the 2×3 affine mapping (page_px_x, page_px_y) → (easting_m, northing_m).

    Needs ≥3 non-collinear ticks for a full 6-DoF affine, ≥2 for a 4-DoF
    similarity. We only return on ≥3 (cheap and keeps the signal honest —
    an affine from 2 ticks would coincide with the data and give zero
    residual even when both are wrong).
    """
    if len(ticks) < 3:
        return None

    src = np.array([t[2] for t in ticks], dtype=np.float64)
    dst_en = np.array(
        [_wgs84_to_osgb(t[1][0], t[1][1]) for t in ticks],
        dtype=np.float64,
    )

    # Reject near-collinear configurations (degenerate affine)
    if _collinearity_score(src) > 0.98 or _collinearity_score(dst_en) > 0.98:
        return None

    A, inliers = cv2.estimateAffine2D(
        src.reshape(-1, 1, 2).astype(np.float32),
        dst_en.reshape(-1, 1, 2).astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,  # metres, generous — OCR bbox centre is noisy
        confidence=0.99,
    )
    if A is None:
        return None

    # Reprojection residual on inliers only
    homog = np.hstack([src, np.ones((len(src), 1))])
    pred = homog @ A.T
    residuals = np.linalg.norm(pred - dst_en, axis=1)
    if inliers is not None:
        mask = inliers.flatten().astype(bool)
        if mask.sum() >= 2:
            residuals = residuals[mask]

    return DirectAffine(
        matrix_2x3=A.tolist(),
        tick_count=int(inliers.sum()) if inliers is not None else len(ticks),
        mean_residual_m=float(residuals.mean()),
    )


def direct_affine_centroid(affine: DirectAffine, img_shape: Tuple[int, int]) -> Tuple[float, float]:
    """Project the image centre through the direct affine and return (lat, lon).

    Used as a confidence-1.0 candidate when grid ticks resolve.
    """
    A = np.array(affine.matrix_2x3, dtype=np.float64)
    h, w = img_shape[:2]
    cx, cy = w / 2.0, h / 2.0
    east, north = A @ np.array([cx, cy, 1.0])

    from pyproj import Transformer
    osgb_to_wgs = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    lon, lat = osgb_to_wgs.transform(east, north)
    return lat, lon
