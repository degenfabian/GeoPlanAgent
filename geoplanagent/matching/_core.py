"""MINIMA model management, the sliding-window matcher, affine recovery and
GeoJSON projection, plus the scale/sigma priors and the composite
window-score reranker it ranks candidates with.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from geoplanagent.geo.coords import (
    best_zoom_for_scale,
    compute_map_mpp,
    latlon_to_global_tile_pixel,
    osm_pixel_to_latlon,
    tile_mpp as _tile_mpp_at,
)
from geoplanagent.matching.road_verify import _verify_candidates_with_road_names
import logging
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent.parent.parent


# Constants empirically tuned against the 211-case cached MINIMA sweep on v3
# benchmark output. Per-case stats are in
# results/benchmark_v3/gemini-flash/<case>/metrics.json.

# Target sliding-window count per (center, zoom, rotation).
WINDOW_STRIDE_TARGET = 100


# MINIMA model management

def load_minima(base_dir=None):
    """Load MINIMA LoFTR matcher model."""
    from argparse import Namespace

    if base_dir is None:
        base_dir = BASE_DIR
    minima_dir = os.path.join(str(base_dir), "MINIMA")
    prev_dir = os.getcwd()
    try:
        os.chdir(minima_dir)
        sys.path.insert(0, minima_dir)
        from load_model import load_model
        args = Namespace(
            ckpt=os.path.join(minima_dir, "weights", "minima_loftr.ckpt"),
            thr=0.2,
        )
        return load_model("loftr", args, use_path=False)
    finally:
        os.chdir(prev_dir)


def run_minima(matcher, map_img, tile_img, grayscale=False):
    """MINIMA match map↔tile. Returns (mkpts0, mkpts1, mconf).

    `grayscale=True` helps B&W or sepia maps against coloured tiles.
    """
    if len(map_img.shape) == 2:
        map_bgr = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
    elif map_img.shape[2] == 4:
        map_bgr = cv2.cvtColor(map_img, cv2.COLOR_RGBA2BGR)
    else:
        map_bgr = map_img.copy()
    if len(tile_img.shape) == 2:
        tile_bgr = cv2.cvtColor(tile_img, cv2.COLOR_GRAY2BGR)
    else:
        tile_bgr = tile_img.copy()

    if grayscale:
        map_gray = cv2.cvtColor(map_bgr, cv2.COLOR_BGR2GRAY)
        tile_gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
        map_bgr = cv2.cvtColor(map_gray, cv2.COLOR_GRAY2BGR)
        tile_bgr = cv2.cvtColor(tile_gray, cv2.COLOR_GRAY2BGR)

    result = matcher(map_bgr, tile_bgr)
    return result["mkpts0"], result["mkpts1"], result["mconf"]


def estimate_affine(mkpts0, mkpts1, mconf=None, reproj_thresh=10.0):
    """Estimate a 4-DOF similarity transform (rotation + uniform scale +
    translation) via RANSAC.

    Similarity is the right prior for map-to-map matching: shear only
    appears as a photography/photocopy artifact. A 6-DOF affine
    fallback was tried and netted out slightly negative on mean IoU,
    so we keep this deliberately simple.

    Returns (H, n_inliers, score, inlier_mask). H is shape (2, 3).
    """
    if len(mkpts0) < 4:
        return None, 0, 0.0, None

    try:
        cv2.setRNGSeed(42)
    except Exception:
        pass
    H, inlier_mask = cv2.estimateAffinePartial2D(
        mkpts0, mkpts1, method=cv2.RANSAC,
        ransacReprojThreshold=reproj_thresh,
    )
    if H is None or inlier_mask is None:
        return None, 0, 0.0, None
    n_inliers = int(inlier_mask.sum())

    if mconf is not None and n_inliers > 0:
        inlier_flags = inlier_mask.ravel().astype(bool)
        score = float(np.sum(mconf[inlier_flags]))
    else:
        score = float(n_inliers)

    return H, n_inliers, score, inlier_mask


# Scale and zoom utilities


_latlon_to_global_tile_pixel = latlon_to_global_tile_pixel


def resize_map_to_match_zoom(map_img, map_mpp, zoom, lat):
    """Resize map so its pixel scale matches the tile pixel scale at given zoom.

    Returns (resized_img, scale_factor) where scale_factor is the resize ratio.
    Returns (None, scale_factor) if the scale difference is too large.
    """
    tmpp = _tile_mpp_at(lat, zoom)
    scale_factor = map_mpp / tmpp
    if scale_factor < 0.3 or scale_factor > 3.0:
        return None, scale_factor
    new_h = int(map_img.shape[0] * scale_factor)
    new_w = int(map_img.shape[1] * scale_factor)
    if new_h < 64 or new_w < 64:
        return None, scale_factor
    # AREA for downscale, CUBIC for upscale: blurry upscale hurts keypoint repeatability.
    interp = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(map_img, (new_w, new_h), interpolation=interp)
    return resized, scale_factor


# Coordinate transform and GeoJSON

def affine_center_to_latlon(affine_H, map_h, map_w, tile_info):
    """Apply affine to map center, convert to lat/lon."""
    cp = affine_H @ np.array([map_w / 2, map_h / 2, 1.0])
    return osm_pixel_to_latlon(
        cp[0], cp[1], tile_info["zoom"],
        tile_info["tx_min"], tile_info["ty_min"],
    )


def mask_to_geojson_affine(mask, affine_H, tile_info):
    """SAM3 mask → GeoJSON Feature via the 2x3 affine. None if no contours."""
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    zoom = tile_info["zoom"]
    tx_min = tile_info["tx_min"]
    ty_min = tile_info["ty_min"]

    all_polys = []
    for contour in contours:
        coords = []
        for pt in contour:
            px, py = float(pt[0][0]), float(pt[0][1])
            osm_pt = affine_H @ np.array([px, py, 1.0])
            lat, lon = osm_pixel_to_latlon(osm_pt[0], osm_pt[1], zoom, tx_min, ty_min)
            coords.append([lon, lat])
        if len(coords) < 4:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        all_polys.append([coords])

    if not all_polys:
        return None
    if len(all_polys) == 1:
        return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": all_polys[0]}, "properties": {}}
    return {"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": all_polys}, "properties": {}}


# Internal helpers

def _build_scale_H(affine_H, wx, wy, sf):
    """Build final affine: shift by window offset, scale for map resize.

    Original pixel (px, py) -> resized (px*sf, py*sf) -> canvas via affine.
    mask_to_geojson_affine does: H @ [px, py, 1], so we absorb sf into H.
    """
    adjusted_H = affine_H.copy()
    adjusted_H[0, 2] += wx
    adjusted_H[1, 2] += wy
    scale_H = adjusted_H.copy()
    scale_H[0, 0] *= sf
    scale_H[0, 1] *= sf
    scale_H[1, 0] *= sf
    scale_H[1, 1] *= sf
    return scale_H


# Road-name verification helper (directional verifier remains ripped).


# Main entry point

def sliding_window_position(
    matcher,
    map_img,
    sam3_mask=None,
    centers=None,
    scale_ratio=None,
    dpi=200,
    rotations=None,
    road_names=None,
    tile_fetcher=None,
    grayscale=False,
    return_candidates=False,
    pdf_path=None,
    map_pages=None,
):
    """Sliding-window MINIMA positioning on OS tiles. Production entry point.

    centers is `[(name, lat, lon, sigma_m)]` from the locate sub-agent (single
    entry; list shape is historical). scale_ratio=None tries common scales.
    Returns: geojson, affine_H, tile_info, match_info, n_windows.
    """
    if tile_fetcher is None:
        from geoplanagent.io.os_tiles import fetch_os_opendata_grid
        tile_fetcher = fetch_os_opendata_grid

    if not centers:
        return {
            "geojson": None, "affine_H": None, "tile_info": None,
            "match_info": {}, "n_windows": 0,
        }

    # Trust the locate sub-agent's σ; fallback only for offline/test callers.
    name, lat, lon, sigma_in = centers[0]
    if sigma_in is None or float(sigma_in) <= 0:
        sigma_in = effective_sigma(scale_ratio)
    centers = [(name, lat, lon, float(sigma_in))]

    # Diversity-bucketed top-K: PER_BUCKET per (anchor, zoom), MAX_CANDIDATES global.
    # Prevents one (center, zoom) sweep from filling every slot with near-duplicates.
    import heapq
    MAX_CANDIDATES = 5
    PER_BUCKET = 1
    per_bucket: Dict[Tuple[str, int], List[Tuple[float, int, dict]]] = {}
    _seq = 0  # tiebreaker for heap
    best_metric = 0
    best_result = None
    total_windows = 0

    map_mpp = compute_map_mpp(scale_ratio, dpi)
    map_h, map_w = map_img.shape[:2]

    # Determine (zoom, mpp) configs. Explores best_z + neighbours plus
    # ±15% scale perturbations to absorb DPI/metadata errors.
    ref_lat = centers[0][1]
    if map_mpp is not None:
        best_z = best_zoom_for_scale(map_mpp, ref_lat)
        zoom_mpp_configs = [
            (z, map_mpp)
            for z in sorted(set([best_z, max(15, best_z - 1), min(19, best_z + 1)]))
        ]
        # ±15% scale perturbation handles DPI/metadata error.
        zoom_mpp_configs.append((best_z, map_mpp * 0.85))
        zoom_mpp_configs.append((best_z, map_mpp * 1.15))
    else:
        # Unknown scale: sweep canonical UK planning-map scales 1:1250–1:25000.
        common_scales = [1250, 2500, 5000, 10000, 15000, 25000]
        zoom_mpp_configs = []
        seen = set()
        for sr in common_scales:
            mpp = compute_map_mpp(sr, dpi)
            z = best_zoom_for_scale(mpp, ref_lat)
            if z not in seen:
                seen.add(z)
                zoom_mpp_configs.append((z, mpp))

        # Modal-scale ±15% catches between-grid scales (1:3500, 1:7000…).
        modal_mpp = compute_map_mpp(2500, dpi)
        modal_z = best_zoom_for_scale(modal_mpp, ref_lat)
        zoom_mpp_configs.append((modal_z, modal_mpp * 0.85))
        zoom_mpp_configs.append((modal_z, modal_mpp * 1.15))

    if rotations is None:
        # Image arrives upright (auto-rotation runs in render_map_page).
        rotations = [0]

    # Tightest sigma first.
    centers = sorted(centers, key=lambda x: x[3] if x[3] is not None else 9e9)
    if centers:
        print(f"  Centers sorted by sigma: {centers[0][0]}(σ={centers[0][3]}) "
              f"→ {centers[-1][0]}(σ={centers[-1][3]})")

    for cname, clat, clon, sigma in centers:
        for zoom, cur_mpp in zoom_mpp_configs:
            tmpp = _tile_mpp_at(clat, zoom)

            resized_map, sf = resize_map_to_match_zoom(map_img, cur_mpp, zoom, clat)
            if resized_map is None:
                continue

            rh, rw = resized_map.shape[:2]

            # Tile grid sized by sigma — no hardcoded floor.
            search_m = sigma if sigma else 1000
            margin_tiles = max(2, int(math.ceil(search_m / (256 * tmpp))))
            nx_needed = int(math.ceil(rw / 256)) + 2 * margin_tiles
            ny_needed = int(math.ceil(rh / 256)) + 2 * margin_tiles
            nx = max(3, min(17, nx_needed))
            ny = max(3, min(17, ny_needed))
            if nx % 2 == 0:
                nx += 1
            if ny % 2 == 0:
                ny += 1

            tile_info = tile_fetcher(clat, clon, zoom, nx, ny)
            os_canvas = tile_info["image"]
            ch, cw = os_canvas.shape[:2]

            if rh >= ch or rw >= cw:
                continue

            n_windows = 0
            for rot_angle in rotations:
                if rot_angle == 0:
                    rot_map = resized_map
                    cur_mask = sam3_mask
                    rot_h, rot_w = rh, rw
                else:
                    rot_codes = {
                        90: cv2.ROTATE_90_CLOCKWISE,
                        180: cv2.ROTATE_180,
                        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                    }
                    if rot_angle not in rot_codes:
                        continue
                    rot_map = cv2.rotate(resized_map, rot_codes[rot_angle])
                    cur_mask = (cv2.rotate(sam3_mask, rot_codes[rot_angle])
                                if sam3_mask is not None else None)
                    if rot_angle in (90, 270):
                        rot_h, rot_w = rw, rh
                    else:
                        rot_h, rot_w = rh, rw

                if rot_h >= ch or rot_w >= cw:
                    continue

                # Stride targets ~WINDOW_STRIDE_TARGET windows; 32 px floor
                # (~48 m at z18) is the spatial-accuracy limit of MINIMA.
                _area_available = max(1, (ch - rot_h) * (cw - rot_w))
                _target_stride = int(math.sqrt(_area_available / WINDOW_STRIDE_TARGET))
                step_x = max(32, min(_target_stride, max(1, cw - rot_w)))
                step_y = max(32, min(_target_stride, max(1, ch - rot_h)))

                for wy in range(0, ch - rot_h + 1, step_y):
                    for wx in range(0, cw - rot_w + 1, step_x):
                        window = os_canvas[wy:wy + rot_h, wx:wx + rot_w]
                        mkpts0, mkpts1, mconf = run_minima(matcher, rot_map, window, grayscale=grayscale)
                        affine_H, n_inliers, score, inlier_mask = estimate_affine(
                            mkpts0, mkpts1, mconf=mconf)
                        n_windows += 1
                        total_windows += 1

                        if affine_H is None or n_inliers < 5:
                            continue

                        # avg_scale (column-norm mean) feeds the scale_consistency reward.
                        a, b = affine_H[0, 0], affine_H[0, 1]
                        c_a, d = affine_H[1, 0], affine_H[1, 1]
                        sx = math.sqrt(a * a + c_a * c_a)
                        sy = math.sqrt(b * b + d * d)
                        avg_scale_now = (sx + sy) / 2

                        metric = float(n_inliers)
                        if metric > best_metric:
                            best_metric = metric

                        # Keep top-N candidates for post-verification
                        if metric > 0:
                            scale_H = _build_scale_H(affine_H, wx, wy, sf)
                            center_ll = affine_center_to_latlon(
                                scale_H, map_h, map_w, tile_info)
                            avg_scale = avg_scale_now
                            # Inlier keypoints (rot_map coords) for the composite reranker.
                            inlier_pts_map = None
                            if inlier_mask is not None:
                                try:
                                    flag = inlier_mask.ravel().astype(bool)
                                    in0 = mkpts0[flag]
                                    if len(in0) > 0:
                                        inlier_pts_map = in0.tolist()
                                except Exception:
                                    inlier_pts_map = None
                            candidate = {
                                "geojson": None,  # defer mask projection
                                "affine_H": scale_H,
                                "tile_info": tile_info,
                                "match_info": {
                                    "center": cname,
                                    "zoom": zoom,
                                    "rotation": rot_angle,
                                    "n_inliers": n_inliers,
                                    "score": round(score, 2),
                                    "scale_factor": round(sf, 3),
                                    "avg_scale": round(avg_scale, 4),
                                    "window": (wx, wy),
                                    "center_latlon": center_ll,
                                    "anchor_latlon": (float(clat), float(clon)),
                                    "_inlier_pts_map": inlier_pts_map,
                                    "_rot_map_shape": (rot_h, rot_w),
                                },
                                "n_windows": 0,
                                "_metric": metric,
                                "_sam3_mask": cur_mask,
                            }
                            _seq += 1
                            bucket_key = (cname, zoom)
                            bucket = per_bucket.setdefault(bucket_key, [])
                            if len(bucket) < PER_BUCKET:
                                heapq.heappush(bucket, (metric, _seq, candidate))
                            elif metric > bucket[0][0]:
                                heapq.heapreplace(bucket, (metric, _seq, candidate))

            if n_windows > 0:
                print(f"    z{zoom}:{cname}: {n_windows}w, "
                      f"best={best_metric:.1f}", flush=True)

    # Flatten buckets → global top-K.
    all_candidates: List[Tuple[float, int, dict]] = []
    for bucket in per_bucket.values():
        all_candidates.extend(bucket)
    if not all_candidates:
        return {
            "geojson": None, "affine_H": None, "tile_info": None,
            "match_info": {}, "n_windows": total_windows,
        }
    all_candidates.sort(key=lambda x: -x[0])

    top_candidates = all_candidates[:MAX_CANDIDATES]

    # Sort candidates best-first by raw metric
    ranked = sorted(top_candidates, key=lambda x: -x[0])

    # Composite rescore: pick by V × Q/4 (composite_window_score, this module).
    if ranked:
        rescored = []
        for metric, seq, cand in ranked:
            mi = cand.get("match_info") or {}
            q = quadrant_coverage_from_inlier_points(
                mi.get("_inlier_pts_map") or [],
                mi.get("_rot_map_shape"),
            )
            composite_score = composite_window_score(metric, q)
            cand["_vanilla_metric"] = metric
            cand["_composite_score"] = composite_score
            cand["_quadrant_cov"] = q
            rescored.append((composite_score, seq, cand))
        rescored.sort(key=lambda x: -x[0])
        ranked = rescored
        if ranked:
            top = ranked[0][2]
            print(f"  Composite rerank: top score={ranked[0][0]:.2f} "
                  f"(V={top.get('_vanilla_metric',0):.2f} "
                  f"Q={top.get('_quadrant_cov',0)})")

    # Road-name verifier: re-rank by metric * (1 + road_match_ratio)^2.
    best_result = None
    if road_names and len(road_names) >= 1:
        best_result = _verify_candidates_with_road_names(ranked, road_names)
    if best_result is None:
        _, _, best_result = ranked[0]

    # Project mask now (deferred from inner loop).
    cur_mask = best_result.get("_sam3_mask") if return_candidates \
        else best_result.pop("_sam3_mask", None)
    if not return_candidates:
        best_result.pop("_metric", None)
    if sam3_mask is not None and cur_mask is not None:
        best_result["geojson"] = mask_to_geojson_affine(
            cur_mask, best_result["affine_H"], best_result["tile_info"])

    best_result["n_windows"] = total_windows

    if return_candidates:
        out_candidates = []
        for metric, _, cand in ranked:
            cand = dict(cand)
            cand["sam3_mask"] = cand.pop("_sam3_mask", None)
            cand["metric"] = cand.pop("_metric", metric)
            out_candidates.append(cand)
        best_result["candidates"] = out_candidates

    return best_result


log = logging.getLogger(__name__)


def composite_window_score(vanilla_metric: float,
                           quadrant_coverage: int) -> float:
    """RANSAC inlier count weighted by spatial spread of the inliers.

    quadrant_coverage counts map quadrants with at least one inlier
    (0..4), which penalises matches whose support sits in one corner.
    """
    if quadrant_coverage < 0:
        quadrant_coverage = 4  # unknown coverage shouldn't penalise
    return float(vanilla_metric) * (quadrant_coverage / 4.0)


def quadrant_coverage_from_inlier_points(
    inlier_pts_map, rot_shape: Tuple[int, int],
) -> int:
    """How many of the rotated map's 4 quadrants contain an inlier.

    inlier_pts_map is the list of (x, y) points that
    geoplanagent.matching.sliding_window_position stores in
    match_info["_inlier_pts_map"]; rot_shape is the (h, w) of the rotated
    map crop at match time.
    """
    if not inlier_pts_map or not rot_shape:
        return 4
    try:
        import numpy as np
        rh, rw = rot_shape
        cx, cy = rw / 2.0, rh / 2.0
        arr = np.asarray(inlier_pts_map)
        return (
            int(((arr[:, 0] < cx) & (arr[:, 1] < cy)).any())
            + int(((arr[:, 0] >= cx) & (arr[:, 1] < cy)).any())
            + int(((arr[:, 0] < cx) & (arr[:, 1] >= cy)).any())
            + int(((arr[:, 0] >= cx) & (arr[:, 1] >= cy)).any())
        )
    except Exception:
        log.warning("quadrant coverage failed for %d pts, shape %s; "
                    "treating as full coverage", len(inlier_pts_map), rot_shape,
                    exc_info=True)
        return 4


# Generic source-side σ floor used by ``effective_sigma`` when the
# worker omits σ. The live locate sub-agent's picks always carry a σ
# directly, so this only matters for the rare fallback path.
_FALLBACK_SIGMA_M = 5000


def sigma_from_scale(scale_ratio, page_mm=(297, 210)):
    """Compute MAP-SCALE-DRIVEN search sigma (meters).

    Lower bound on σ — the area MINIMA must search to fit the planning
    map's visible extent against OS tiles.

    Args:
        scale_ratio: Map scale denominator (e.g., 2500 for 1:2500). None if unknown.
        page_mm: Paper size (default A4 landscape).

    Returns:
        Sigma in metres = half-diagonal of the printed map's real-world extent.
        For 1:1250 → 226m, 1:2500 → 454m, 1:10000 → 1815m, 1:25000 → 4540m.
        If scale unknown, returns 2500m (a sensible default for site plans).
    """
    if scale_ratio is None:
        return 2500
    diag_mm = math.sqrt(page_mm[0] ** 2 + page_mm[1] ** 2)
    half_diag_m = 0.5 * (diag_mm / 1000.0) * scale_ratio
    return max(150, int(half_diag_m))  # tiny floor — only covers numerical safety


def effective_sigma(scale_ratio: Optional[int]) -> int:
    """Fallback MINIMA search sigma when the worker omits σ.

    Returns ``max(_FALLBACK_SIGMA_M, sigma_from_scale(scale_ratio))`` —
    conservative floor covering both candidate→GT drift and the map's
    visible extent. Fires almost never in practice because the live
    locate sub-agent always supplies σ on its picks.
    """
    return max(_FALLBACK_SIGMA_M, sigma_from_scale(scale_ratio))
