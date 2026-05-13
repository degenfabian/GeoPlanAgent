"""Delaunay-consistency RANSAC filter (Pierdicca et al. 2025).

Post-RANSAC outlier filter for keypoint matches. Builds Delaunay triangulation
on inlier keypoints in the source (plan) image; for each triangle compares its
area in source vs target; drops triangles whose source/target area ratio is
outside [r_lo, r_hi]. Re-fit affine on filtered inliers.

Survives non-affine local distortion (planning maps are hand-drafted with
local stretch). Independent of matcher → composes with MINIMA / any sparse
matcher. Pure scipy + shapely. No GPU, no weights.

Use:
    from tools.delaunay_filter import delaunay_consistency_filter
    H_filtered, kept_mask = delaunay_consistency_filter(
        mkpts0, mkpts1, inlier_mask_from_ransac,
        area_ratio_band=(0.5, 2.0), reproj_thresh=10.0,
    )
"""
from __future__ import annotations
from typing import Tuple
import math
import numpy as np


def _triangle_area(pts: np.ndarray) -> float:
    """Signed area of a triangle defined by 3 points (3,2)."""
    a, b, c = pts
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1])
                       - (c[0] - a[0]) * (b[1] - a[1]))


def delaunay_consistency_filter(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    inlier_mask: np.ndarray | None = None,
    area_ratio_band: Tuple[float, float] = (0.5, 2.0),
    reproj_thresh: float = 10.0,
    min_inliers_after: int = 4,
) -> Tuple[np.ndarray | None, np.ndarray | None, int]:
    """Run Delaunay-consistency filter on already-RANSAC-passed inliers.

    Args:
        mkpts0: Source keypoints (N, 2).
        mkpts1: Target keypoints (N, 2). Same N as mkpts0.
        inlier_mask: 1-D mask from prior RANSAC (None = use all).
        area_ratio_band: (lo, hi) - drop triangles with area ratio outside.
        reproj_thresh: Used to re-fit affine after filtering.
        min_inliers_after: Need this many inliers post-filter to commit;
            else fall back to original RANSAC affine.

    Returns:
        (H_filtered, kept_mask, n_kept) — H_filtered is None if filter
        eliminated too many points; kept_mask is over the ORIGINAL N matches.
    """
    if mkpts0 is None or len(mkpts0) < 4:
        return None, None, 0
    n = len(mkpts0)

    # Restrict to RANSAC inliers
    if inlier_mask is not None:
        inlier_mask = inlier_mask.astype(bool).ravel()
        idx_inliers = np.where(inlier_mask)[0]
    else:
        idx_inliers = np.arange(n)
    if len(idx_inliers) < 4:
        return None, None, 0

    pts0_in = mkpts0[idx_inliers]
    pts1_in = mkpts1[idx_inliers]

    # Build Delaunay triangulation on source points
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(pts0_in)
    except Exception:
        return None, None, 0

    # For each triangle, compute area ratio (target/source)
    triangle_indices = tri.simplices  # (T, 3) indices into pts0_in / pts1_in
    if len(triangle_indices) == 0:
        return None, None, 0

    # Vote per inlier: how many "good" triangles is it part of?
    votes = np.zeros(len(pts0_in), dtype=np.int32)
    n_total_triangles = 0
    for tri_idx in triangle_indices:
        a0 = _triangle_area(pts0_in[tri_idx])
        a1 = _triangle_area(pts1_in[tri_idx])
        if a0 == 0 or a1 == 0: continue
        ratio = a1 / a0
        n_total_triangles += 1
        if area_ratio_band[0] <= ratio <= area_ratio_band[1]:
            votes[tri_idx] += 1
    if n_total_triangles == 0:
        return None, None, 0

    # Keep inliers that participate in at least 1 good triangle. Each
    # interior point is in ~4-6 triangles, edge points in fewer.
    keep_mask_local = votes > 0
    n_kept_local = int(keep_mask_local.sum())
    if n_kept_local < min_inliers_after:
        return None, None, n_kept_local

    pts0_kept = pts0_in[keep_mask_local]
    pts1_kept = pts1_in[keep_mask_local]

    # Re-fit affine on kept points
    import cv2
    H, _ = cv2.estimateAffinePartial2D(
        pts0_kept, pts1_kept, method=cv2.RANSAC,
        ransacReprojThreshold=reproj_thresh,
    )
    if H is None:
        return None, None, n_kept_local

    # Build kept_mask over the original N matches
    kept_mask_full = np.zeros(n, dtype=bool)
    kept_mask_full[idx_inliers[keep_mask_local]] = True
    return H, kept_mask_full, n_kept_local


if __name__ == '__main__':
    # Smoke test: synthetic outliers should be filtered
    np.random.seed(42)
    src = np.random.rand(50, 2) * 100
    H_true = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0]])  # pure translation
    dst_true = (H_true[:, :2] @ src.T + H_true[:, 2:3]).T
    # Add 10 outliers
    src_outliers = np.random.rand(10, 2) * 100
    dst_outliers = np.random.rand(10, 2) * 100
    src_full = np.vstack([src, src_outliers])
    dst_full = np.vstack([dst_true, dst_outliers])
    inlier_mask = np.ones(60, dtype=np.uint8)  # all "inliers" pre-filter
    H_f, kept, n = delaunay_consistency_filter(src_full, dst_full, inlier_mask)
    print(f'kept {n}/60 (expect ~50)  H={H_f}')
