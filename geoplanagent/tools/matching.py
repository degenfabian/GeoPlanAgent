"""MINIMA-LoFTR registration of a planning map against OS tiles, plus match-quality rewards."""

from __future__ import annotations

import heapq
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from geoplanagent.utils import (
    best_zoom_for_scale,
    compute_map_mpp,
    osm_pixel_to_latlon,
    tile_mpp,
)


BASE_DIR = Path(__file__).resolve().parent.parent.parent


# Constants empirically tuned against the 211-case cached MINIMA sweep on v3
# benchmark output.

# Target sliding-window count per (center, zoom, rotation).
WINDOW_STRIDE_TARGET = 100


# MINIMA model management


def load_minima():
    """Load MINIMA LoFTR matcher model."""
    from argparse import Namespace

    minima_dir = os.path.join(str(BASE_DIR), "MINIMA")
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


def run_minima(matcher, map_img, tile_img):
    """MINIMA match map↔tile. Returns (mkpts0, mkpts1, mconf)."""
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

    result = matcher(map_bgr, tile_bgr)
    return result["mkpts0"], result["mkpts1"], result["mconf"]


def estimate_affine(mkpts0, mkpts1):
    """Estimate a 4-DOF similarity transform (rotation + uniform scale +
    translation) via RANSAC.

    Similarity is the right prior for map-to-map matching: shear only
    appears as a photography/photocopy artifact. A 6-DOF affine
    fallback was tried and netted out slightly negative on mean IoU,
    so we keep this deliberately simple.

    Returns (H, n_inliers, inlier_mask). H is shape (2, 3).
    """
    if len(mkpts0) < 4:
        return None, 0, None

    try:
        cv2.setRNGSeed(42)
    except Exception:
        pass
    H, inlier_mask = cv2.estimateAffinePartial2D(
        mkpts0,
        mkpts1,
        method=cv2.RANSAC,
        ransacReprojThreshold=10.0,
    )
    if H is None or inlier_mask is None:
        return None, 0, None
    n_inliers = int(inlier_mask.sum())

    return H, n_inliers, inlier_mask


# Scale and zoom utilities


def resize_map_to_match_zoom(map_img, map_mpp, zoom, lat):
    """Resize map so its pixel scale matches the tile pixel scale at given zoom.

    Returns (resized_img, scale_factor) where scale_factor is the resize ratio.
    Returns (None, scale_factor) if the scale difference is too large.
    """
    tile_mpp_here = tile_mpp(lat, zoom)
    scale_factor = map_mpp / tile_mpp_here
    if scale_factor < 0.3 or scale_factor > 3.0:
        return None, scale_factor
    new_height = int(map_img.shape[0] * scale_factor)
    new_width = int(map_img.shape[1] * scale_factor)
    if new_height < 64 or new_width < 64:
        return None, scale_factor
    # AREA for downscale, CUBIC for upscale: blurry upscale hurts keypoint repeatability.
    interpolation = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(map_img, (new_width, new_height), interpolation=interpolation)
    return resized, scale_factor


# Coordinate transform and GeoJSON


def affine_center_to_latlon(affine_H, map_h, map_w, tile_info):
    """Apply affine to map center, convert to lat/lon."""
    center_pixel = affine_H @ np.array([map_w / 2, map_h / 2, 1.0])
    return osm_pixel_to_latlon(
        center_pixel[0],
        center_pixel[1],
        tile_info["zoom"],
        tile_info["tx_min"],
        tile_info["ty_min"],
    )


def mask_to_geojson_affine(mask, affine_H, tile_info):
    """SAM3 mask → GeoJSON Feature via the 2x3 affine. None if no contours."""
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    zoom = tile_info["zoom"]
    tx_min = tile_info["tx_min"]
    ty_min = tile_info["ty_min"]

    polygons = []
    for contour in contours:
        ring = []
        for point in contour:
            pixel_x, pixel_y = float(point[0][0]), float(point[0][1])
            osm_pixel = affine_H @ np.array([pixel_x, pixel_y, 1.0])
            lat, lon = osm_pixel_to_latlon(osm_pixel[0], osm_pixel[1], zoom, tx_min, ty_min)
            ring.append([lon, lat])
        if len(ring) < 4:
            continue
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        polygons.append([ring])

    if not polygons:
        return None
    if len(polygons) == 1:
        return {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": polygons[0]},
            "properties": {},
        }
    return {
        "type": "Feature",
        "geometry": {"type": "MultiPolygon", "coordinates": polygons},
        "properties": {},
    }


# Internal helpers


def _build_scale_H(affine_H, window_x, window_y, scale_factor):
    """Build final affine: shift by window offset, scale for map resize.

    Original pixel (px, py) -> resized (px*sf, py*sf) -> canvas via affine.
    mask_to_geojson_affine does: H @ [px, py, 1], so we absorb sf into H.
    """
    adjusted_H = affine_H.copy()
    adjusted_H[0, 2] += window_x
    adjusted_H[1, 2] += window_y
    scale_H = adjusted_H.copy()
    scale_H[0, 0] *= scale_factor
    scale_H[0, 1] *= scale_factor
    scale_H[1, 0] *= scale_factor
    scale_H[1, 1] *= scale_factor
    return scale_H


# Main entry point


def sliding_window_position(
    matcher,
    map_img,
    sam3_mask=None,
    centers=None,
    scale_ratio=None,
    dpi=200,
    road_names=None,
):
    """Sliding-window MINIMA positioning on OS tiles. Production entry point.

    centers is `[(name, lat, lon, sigma_m)]` from the locate sub-agent (single
    entry; list shape is historical). scale_ratio=None tries common scales.
    Returns: geojson, affine_H, tile_info, match_info.
    """
    from geoplanagent.tools.tiles import fetch_os_opendata_grid

    if not centers:
        return {
            "geojson": None,
            "affine_H": None,
            "tile_info": None,
            "match_info": {},
        }

    # Trust the locate sub-agent's σ; fallback only for offline/test callers.
    name, lat, lon, sigma_in = centers[0]
    if sigma_in is None or float(sigma_in) <= 0:
        # Conservative σ floor; the live locate sub-agent always supplies σ.
        sigma_in = max(5000, sigma_from_scale(scale_ratio))
    centers = [(name, lat, lon, float(sigma_in))]

    # Diversity-bucketed top-K: PER_BUCKET per (anchor, zoom), MAX_CANDIDATES global.
    # Prevents one (center, zoom) sweep from filling every slot with near-duplicates.
    MAX_CANDIDATES = 5
    PER_BUCKET = 1
    candidates_per_bucket: Dict[Tuple[str, int], List[Tuple[float, int, dict]]] = {}
    sequence_number = 0  # tiebreaker for heap
    best_metric = 0

    map_mpp = compute_map_mpp(scale_ratio, dpi)
    map_h, map_w = map_img.shape[:2]

    # Determine (zoom, mpp) configs. Explores best_z + neighbours plus
    # ±15% scale perturbations to absorb DPI/metadata errors.
    ref_lat = centers[0][1]
    if map_mpp is not None:
        best_zoom = best_zoom_for_scale(map_mpp, ref_lat)
        zoom_mpp_configs = [
            (zoom, map_mpp)
            for zoom in sorted(set([best_zoom, max(15, best_zoom - 1), min(19, best_zoom + 1)]))
        ]
        # ±15% scale perturbation handles DPI/metadata error.
        zoom_mpp_configs.append((best_zoom, map_mpp * 0.85))
        zoom_mpp_configs.append((best_zoom, map_mpp * 1.15))
    else:
        # Unknown scale: sweep canonical UK planning-map scales 1:1250–1:25000.
        common_scales = [1250, 2500, 5000, 10000, 15000, 25000]
        zoom_mpp_configs = []
        seen_zooms = set()
        for scale in common_scales:
            mpp = compute_map_mpp(scale, dpi)
            zoom = best_zoom_for_scale(mpp, ref_lat)
            if zoom not in seen_zooms:
                seen_zooms.add(zoom)
                zoom_mpp_configs.append((zoom, mpp))

        # Modal-scale ±15% catches between-grid scales (1:3500, 1:7000…).
        modal_mpp = compute_map_mpp(2500, dpi)
        modal_zoom = best_zoom_for_scale(modal_mpp, ref_lat)
        zoom_mpp_configs.append((modal_zoom, modal_mpp * 0.85))
        zoom_mpp_configs.append((modal_zoom, modal_mpp * 1.15))

    for center_name, center_lat, center_lon, sigma in centers:
        for zoom, config_mpp in zoom_mpp_configs:
            tile_mpp_here = tile_mpp(center_lat, zoom)

            resized_map, scale_factor = resize_map_to_match_zoom(
                map_img, config_mpp, zoom, center_lat
            )
            if resized_map is None:
                continue

            resized_height, resized_width = resized_map.shape[:2]

            # Tile grid sized by sigma — no hardcoded floor.
            search_radius_m = sigma if sigma else 1000
            margin_tiles = max(2, int(math.ceil(search_radius_m / (256 * tile_mpp_here))))
            tiles_x_needed = int(math.ceil(resized_width / 256)) + 2 * margin_tiles
            tiles_y_needed = int(math.ceil(resized_height / 256)) + 2 * margin_tiles
            tiles_x = max(3, min(17, tiles_x_needed))
            tiles_y = max(3, min(17, tiles_y_needed))
            if tiles_x % 2 == 0:
                tiles_x += 1
            if tiles_y % 2 == 0:
                tiles_y += 1

            tile_info = fetch_os_opendata_grid(center_lat, center_lon, zoom, tiles_x, tiles_y)
            os_canvas = tile_info["image"]
            canvas_height, canvas_width = os_canvas.shape[:2]

            if resized_height >= canvas_height or resized_width >= canvas_width:
                continue

            n_windows = 0

            # Stride targets ~WINDOW_STRIDE_TARGET windows; 32 px floor
            # (~48 m at z18) is the spatial-accuracy limit of MINIMA.
            available_area = max(
                1, (canvas_height - resized_height) * (canvas_width - resized_width)
            )
            target_stride = int(math.sqrt(available_area / WINDOW_STRIDE_TARGET))
            step_x = max(32, min(target_stride, max(1, canvas_width - resized_width)))
            step_y = max(32, min(target_stride, max(1, canvas_height - resized_height)))

            for window_y in range(0, canvas_height - resized_height + 1, step_y):
                for window_x in range(0, canvas_width - resized_width + 1, step_x):
                    window = os_canvas[
                        window_y : window_y + resized_height,
                        window_x : window_x + resized_width,
                    ]
                    mkpts0, mkpts1, _ = run_minima(matcher, resized_map, window)
                    affine_H, n_inliers, inlier_mask = estimate_affine(mkpts0, mkpts1)
                    n_windows += 1

                    if affine_H is None or n_inliers < 5:
                        continue

                    # avg_scale (column-norm mean) feeds the scale_consistency reward.
                    scale_x = math.sqrt(affine_H[0, 0] * affine_H[0, 0] + affine_H[1, 0] * affine_H[1, 0])
                    scale_y = math.sqrt(affine_H[0, 1] * affine_H[0, 1] + affine_H[1, 1] * affine_H[1, 1])
                    avg_scale = (scale_x + scale_y) / 2

                    metric = float(n_inliers)
                    if metric > best_metric:
                        best_metric = metric

                    # Keep top-N candidates for post-verification
                    scale_H = _build_scale_H(affine_H, window_x, window_y, scale_factor)
                    center_latlon = affine_center_to_latlon(scale_H, map_h, map_w, tile_info)
                    # Inlier keypoints (map coords) for the composite reranker.
                    inlier_points_map = None
                    if inlier_mask is not None:
                        try:
                            inlier_flags = inlier_mask.ravel().astype(bool)
                            inlier_keypoints = mkpts0[inlier_flags]
                            if len(inlier_keypoints) > 0:
                                inlier_points_map = inlier_keypoints.tolist()
                        except Exception:
                            inlier_points_map = None
                    candidate = {
                        "geojson": None,  # defer mask projection
                        "affine_H": scale_H,
                        "tile_info": tile_info,
                        "match_info": {
                            "center": center_name,
                            "zoom": zoom,
                            "n_inliers": n_inliers,
                            "scale_factor": round(scale_factor, 3),
                            "avg_scale": round(avg_scale, 4),
                            "window": (window_x, window_y),
                            "center_latlon": center_latlon,
                            "anchor_latlon": (float(center_lat), float(center_lon)),
                            "_inlier_pts_map": inlier_points_map,
                            "_rot_map_shape": (resized_height, resized_width),
                        },
                        "_sam3_mask": sam3_mask,
                    }
                    sequence_number += 1
                    bucket_key = (center_name, zoom)
                    bucket = candidates_per_bucket.setdefault(bucket_key, [])
                    if len(bucket) < PER_BUCKET:
                        heapq.heappush(bucket, (metric, sequence_number, candidate))
                    elif metric > bucket[0][0]:
                        heapq.heapreplace(bucket, (metric, sequence_number, candidate))

            if n_windows > 0:
                print(
                    f"    z{zoom}:{center_name}: {n_windows}w, best={best_metric:.1f}", flush=True
                )

    # Flatten buckets → global top-K.
    all_candidates: List[Tuple[float, int, dict]] = []
    for bucket in candidates_per_bucket.values():
        all_candidates.extend(bucket)
    if not all_candidates:
        return {
            "geojson": None,
            "affine_H": None,
            "tile_info": None,
            "match_info": {},
        }
    all_candidates.sort(key=lambda c: -c[0])

    ranked = all_candidates[:MAX_CANDIDATES]

    # Composite rescore: pick by V × Q/4 (composite_window_score, this module).
    rescored = []
    for metric, seq, candidate in ranked:
        match_info = candidate.get("match_info") or {}
        quadrant_coverage = quadrant_coverage_from_inlier_points(
            match_info.get("_inlier_pts_map") or [],
            match_info.get("_rot_map_shape"),
        )
        composite_score = composite_window_score(metric, quadrant_coverage)
        candidate["_vanilla_metric"] = metric
        candidate["_quadrant_cov"] = quadrant_coverage
        rescored.append((composite_score, seq, candidate))
    rescored.sort(key=lambda c: -c[0])
    ranked = rescored
    top_candidate = ranked[0][2]
    print(
        f"  Composite rerank: top score={ranked[0][0]:.2f} "
        f"(V={top_candidate.get('_vanilla_metric', 0):.2f} "
        f"Q={top_candidate.get('_quadrant_cov', 0)})"
    )

    # Road-name verifier: re-rank by metric * (1 + road_match_ratio)^2.
    best_result = None
    if road_names and len(road_names) >= 1:
        best_result = _verify_candidates_with_road_names(ranked, road_names)
    if best_result is None:
        _, _, best_result = ranked[0]

    # Project mask now (deferred from inner loop).
    deferred_mask = best_result.pop("_sam3_mask", None)
    if sam3_mask is not None and deferred_mask is not None:
        best_result["geojson"] = mask_to_geojson_affine(
            deferred_mask, best_result["affine_H"], best_result["tile_info"]
        )

    return best_result


def composite_window_score(vanilla_metric: float, quadrant_coverage: int) -> float:
    """RANSAC inlier count weighted by spatial spread of the inliers.

    quadrant_coverage counts map quadrants with at least one inlier
    (0..4), which penalises matches whose support sits in one corner.
    """
    if quadrant_coverage < 0:
        quadrant_coverage = 4  # unknown coverage shouldn't penalise
    return float(vanilla_metric) * (quadrant_coverage / 4.0)


def quadrant_coverage_from_inlier_points(
    inlier_pts_map,
    rot_shape: Tuple[int, int],
) -> int:
    """How many of the rotated map's 4 quadrants contain an inlier.

    inlier_pts_map is the list of (x, y) points that
    geoplanagent.tools.matching.sliding_window_position stores in
    match_info["_inlier_pts_map"]; rot_shape is the (h, w) of the rotated
    map crop at match time.
    """
    if not inlier_pts_map or not rot_shape:
        return 4
    try:
        rotated_height, rotated_width = rot_shape
        center_x, center_y = rotated_width / 2.0, rotated_height / 2.0
        points = np.asarray(inlier_pts_map)
        return (
            int(((points[:, 0] < center_x) & (points[:, 1] < center_y)).any())
            + int(((points[:, 0] >= center_x) & (points[:, 1] < center_y)).any())
            + int(((points[:, 0] < center_x) & (points[:, 1] >= center_y)).any())
            + int(((points[:, 0] >= center_x) & (points[:, 1] >= center_y)).any())
        )
    except Exception as e:
        print(
            f"  warn: quadrant coverage failed for {len(inlier_pts_map)} pts, "
            f"shape {rot_shape} ({e!s:.80}); treating as full coverage"
        )
        return 4


def sigma_from_scale(scale_ratio):
    """Compute MAP-SCALE-DRIVEN search sigma (meters).

    Lower bound on σ — the area MINIMA must search to fit the planning
    map's visible extent against OS tiles, assuming an A4-landscape page.

    Args:
        scale_ratio: Map scale denominator (e.g., 2500 for 1:2500). None if unknown.

    Returns:
        Sigma in metres = half-diagonal of the printed map's real-world extent.
        For 1:1250 → 226m, 1:2500 → 454m, 1:10000 → 1815m, 1:25000 → 4540m.
        If scale unknown, returns 2500m (a sensible default for site plans).
    """
    if scale_ratio is None:
        return 2500
    diagonal_mm = math.sqrt(297**2 + 210**2)  # A4 landscape
    half_diagonal_m = 0.5 * (diagonal_mm / 1000.0) * scale_ratio
    return max(150, int(half_diagonal_m))  # tiny floor — only covers numerical safety


# A reader-provided scale is signaled by a real "1:N" / "1/N" pattern in
# the extracted text — not by the absence of a few stop-words. The old
# substring check "not in reader_scale_text.lower()" mis-classified valid
# scales like "1:2500 (note: ...)" or "1:2500 cannot be guaranteed" as
# "no reader scale" because "not" appears as a substring.
_SCALE_PATTERN_RE = re.compile(r"\b1\s*[:/]\s*\d")


# Axis primitives


@dataclass
class AxisResult:
    score: float  # in [0, 1]
    verdict: str  # 1-line human-readable verdict


@dataclass
class RewardResult:
    axes: Dict[str, AxisResult]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form stored in metrics.json under "reward"."""
        return {
            "axes": {
                name: {"score": ax.score, "verdict": ax.verdict} for name, ax in self.axes.items()
            },
        }


# Axis implementations


def axis_scale_consistency(
    avg_scale: float,
    reader_scale_text: Optional[str] = None,
) -> AxisResult:
    """Does the recovered affine scale agree with the assumed scale?

    avg_scale ≈ 1.0 means the resize-to-tile-pixel-scale was correct,
    which means the assumed map scale was right. Far from 1.0 indicates
    the assumed scale was wrong AND/OR MINIMA found a coincidental match
    at a different scale.

    Score is ``min(s, 1/s) ** 2`` — symmetric about identity (treats
    "stretched 31% more" and "compressed by 24%" as equally suspicious),
    returns 1.0 at s=1, smoothly decays toward 0. The squaring is the
    only knob (``p=2``); it sharpens the slope near identity enough
    that a 10–30% deviation produces a decisive ranking difference
    when the worker or critic compares candidates.
    """
    scale = float(avg_scale or 0.0)
    if scale <= 0:
        return AxisResult(score=0.0, verdict="invalid (avg_scale ≤ 0)")

    reader_provided = bool(reader_scale_text and _SCALE_PATTERN_RE.search(str(reader_scale_text)))
    score = min(scale, 1.0 / scale) ** 2

    if reader_provided:
        verdict = (
            f"scale_consistency={score:.2f} "
            f"(avg_scale={scale:.3f}, reader said {reader_scale_text!r})"
        )
    else:
        verdict = f"scale_consistency={score:.2f} (avg_scale={scale:.3f}, no reader scale)"

    return AxisResult(score=score, verdict=verdict)


def axis_road_name_agreement(
    chosen_lat: float,
    chosen_lon: float,
    reader_road_names: List[str],
) -> AxisResult:
    """Are the reader-extracted road names present in the OS road network
    at the matched location?

    Uses the offline OS Open Zoomstack GeoPackage (no network calls).

    Three regimes (distinguishes "no data" from "data disagrees"):
      * `reader_road_names` empty           → 0.5 neutral (no signal to test)
      * OS has no roads in radius           → 0.5 neutral (sparse cartography,
        common in rural villages — NOT a wrong-area signal)
      * OS has roads, but none match reader → 0.0 strong wrong-area signal
      * Some / all match                    → matched / total
    """
    n_road_names = len(reader_road_names or [])
    if n_road_names == 0:
        return AxisResult(score=0.5, verdict="no road names extracted by reader (no signal)")

    nearby_road_names = _query_gpkg_road_names(chosen_lat, chosen_lon, radius_m=1500.0)
    if not nearby_road_names:
        return AxisResult(
            score=0.5,
            verdict=(
                "no OS roads within radius — sparse cartography "
                "(rural / unlabelled), neutral signal"
            ),
        )

    matched_names: List[str] = []
    for road_name in reader_road_names:
        if _fuzzy_road_match(road_name, nearby_road_names):
            matched_names.append(road_name)

    # Score is the raw match ratio; the verdict is just human-readable
    # context. Resist the urge to add tier thresholds here — the critic
    # reads the score directly, and any "strong/partial/weak" labels would
    # be arbitrary cutoffs masquerading as principled signal.
    n_matched = len(matched_names)
    score = n_matched / n_road_names
    if score == 0:
        verdict = (
            f"OS roads present here but ZERO of {n_road_names} reader roads "
            f"match — possible wrong-area signal (trust strong inliers "
            f"over this)"
        )
    else:
        verdict = f"{n_matched}/{n_road_names} reader roads found in OS"

    return AxisResult(score=score, verdict=verdict)


# Top-level entry point


def compute_match_reward(
    *,
    match_info: Dict[str, Any],
    pdf_info: Dict[str, Any],
) -> RewardResult:
    """Compute the per-axis consistency reward for a single match.

    Args:
        match_info: dict from sliding_window_position with at least
            n_inliers, avg_scale, center_latlon.
        pdf_info: PDFInfo dict from the reader (scale, road_names, …).
    """
    avg_scale = float(match_info.get("avg_scale", 0.0) or 0.0)
    center_latlon = match_info.get("center_latlon")

    axes: Dict[str, AxisResult] = {
        "scale_consistency": axis_scale_consistency(
            avg_scale, reader_scale_text=pdf_info.get("scale")
        ),
    }

    if center_latlon and len(center_latlon) == 2:
        axes["road_name_agreement"] = axis_road_name_agreement(
            float(center_latlon[0]),
            float(center_latlon[1]),
            list(pdf_info.get("road_names") or []),
        )
    else:
        axes["road_name_agreement"] = AxisResult(score=0.5, verdict="no center_latlon (no signal)")

    return RewardResult(axes=axes)


# Set to True the first time we notice the OS Zoomstack file is missing,
# so the verifier prints exactly one warning per process instead of
# spamming every candidate iteration.
_ZOOMSTACK_WARNED = False


# Road-name verifier


def _query_gpkg_road_names(lat, lon, radius_m=1500):
    """Query OS GeoPackage for road names near a point. Fully offline."""
    try:
        import geopandas as gpd
        import pyproj

        gpkg_path = BASE_DIR / "os_opendata" / "OS_Open_Zoomstack.gpkg"
        if not gpkg_path.exists():
            # Warn ONCE per process, not per-call (this fires inside the
            # per-candidate verifier loop and the per-call critic axis).
            global _ZOOMSTACK_WARNED
            if not _ZOOMSTACK_WARNED:
                print(
                    f"  road_verify: WARNING — {gpkg_path} not found; "
                    f"road-name verification disabled. Download from "
                    f"OS Open Zoomstack and place at this path."
                )
                _ZOOMSTACK_WARNED = True
            return []

        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
        x, y = transformer.transform(lon, lat)

        names = set()
        for layer in ["roads_local", "roads_regional", "roads_national"]:
            try:
                gdf = gpd.read_file(
                    str(gpkg_path),
                    layer=layer,
                    bbox=(x - radius_m, y - radius_m, x + radius_m, y + radius_m),
                )
                for _, row in gdf.iterrows():
                    name = row.get("name")
                    if name and str(name).strip() and str(name) != "None":
                        names.add(str(name).strip())
            except Exception:
                pass
        return list(names)
    except ImportError:
        return []


def _fuzzy_road_match(llm_name, reference_names):
    """Check if an LLM-extracted road name matches any reference name."""
    llm_lower = llm_name.lower().strip()
    for ref in reference_names:
        ref_lower = ref.lower().strip()
        if llm_lower == ref_lower:
            return True
        if llm_lower in ref_lower or ref_lower in llm_lower:
            return True
        # Handle common abbreviations
        llm_norm = (
            llm_lower.replace(" street", " st")
            .replace(" road", " rd")
            .replace(" lane", " ln")
            .replace(" avenue", " ave")
            .replace(" drive", " dr")
            .replace(" close", " cl")
        )
        ref_norm = (
            ref_lower.replace(" street", " st")
            .replace(" road", " rd")
            .replace(" lane", " ln")
            .replace(" avenue", " ave")
            .replace(" drive", " dr")
            .replace(" close", " cl")
        )
        if llm_norm == ref_norm:
            return True
    return False


def _verify_candidates_with_road_names(ranked_candidates, road_names):
    """Re-rank candidates by metric × (1 + road_match_ratio) ** 2.

    Each candidate's metric is multiplied by a quadratic boost from its
    road-name overlap ratio: 0 matches → 1× (no change), all matches →
    4× (full boost). The quadratic shape is symmetric with the squared
    scale_consistency penalty — both treat the relevant signal as
    multiplicatively-quadratic in their evidence.

    Single knob (exponent ``p = 2``); replaces the previous triple-gated
    scheme (5 magic numbers: 0.5 / 0.6 / 2× / 0.7 / 0.01) with one
    decisive multiplicative form.

    Candidates with no nearby OS roads (sparse cartography) get a
    neutral boost of 1.0 — neither helped nor penalised — so the metric
    fully decides for them.
    """
    if not road_names:
        return None

    n_road_names = len(road_names)
    scored = []
    for metric, _seq, candidate in ranked_candidates:
        center_latlon = candidate["match_info"].get("center_latlon")
        if not center_latlon:
            scored.append((metric, metric, candidate, None))
            continue
        lat, lon = center_latlon
        nearby_road_names = _query_gpkg_road_names(lat, lon, radius_m=1500)
        if not nearby_road_names:
            scored.append((metric, metric, candidate, None))
            continue
        matches = sum(
            1 for road_name in road_names if _fuzzy_road_match(road_name, nearby_road_names)
        )
        ratio = matches / n_road_names
        boosted = metric * (1.0 + ratio) ** 2
        scored.append((boosted, metric, candidate, matches))

    if not scored:
        return None

    for boosted, original_metric, candidate, matches in scored:
        center_name = candidate["match_info"]["center"]
        inliers = candidate["match_info"]["n_inliers"]
        matches_text = "n/a" if matches is None else f"{matches}/{n_road_names}"
        print(
            f"    Road verify: {center_name} inl={inliers} "
            f"metric={original_metric:.1f} boosted={boosted:.1f} roads={matches_text}"
        )

    scored.sort(key=lambda r: -r[0])
    top_candidate = scored[0][2]
    metric_best_candidate = ranked_candidates[0][2]
    if top_candidate is metric_best_candidate:
        print("    Road verify: top candidate confirmed")
        return None
    center_name = top_candidate["match_info"]["center"]
    print(f"    Road verify: OVERRIDE → {center_name} (boosted {scored[0][0]:.1f})")
    return top_candidate
