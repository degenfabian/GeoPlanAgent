"""MINIMA-LoFTR map-to-tile registration: model management, the sliding-window
search with RANSAC affine fit and diversity-bucketed re-ranking, the
scale/sigma priors, road-name verification, and the per-axis match-quality
reward signals fed to the worker's commit policy.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from geoplanagent.utils import (
    best_zoom_for_scale,
    compute_map_mpp,
    osm_pixel_to_latlon,
    tile_mpp as _tile_mpp_at,
)
from typing import Optional
import re
from dataclasses import dataclass
from typing import Any


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
        cp[0],
        cp[1],
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
        return {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": all_polys[0]},
            "properties": {},
        }
    return {
        "type": "Feature",
        "geometry": {"type": "MultiPolygon", "coordinates": all_polys},
        "properties": {},
    }


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

    map_mpp = compute_map_mpp(scale_ratio, dpi)
    map_h, map_w = map_img.shape[:2]

    # Determine (zoom, mpp) configs. Explores best_z + neighbours plus
    # ±15% scale perturbations to absorb DPI/metadata errors.
    ref_lat = centers[0][1]
    if map_mpp is not None:
        best_z = best_zoom_for_scale(map_mpp, ref_lat)
        zoom_mpp_configs = [
            (z, map_mpp) for z in sorted(set([best_z, max(15, best_z - 1), min(19, best_z + 1)]))
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

            tile_info = fetch_os_opendata_grid(clat, clon, zoom, nx, ny)
            os_canvas = tile_info["image"]
            ch, cw = os_canvas.shape[:2]

            if rh >= ch or rw >= cw:
                continue

            n_windows = 0

            # Stride targets ~WINDOW_STRIDE_TARGET windows; 32 px floor
            # (~48 m at z18) is the spatial-accuracy limit of MINIMA.
            _area_available = max(1, (ch - rh) * (cw - rw))
            _target_stride = int(math.sqrt(_area_available / WINDOW_STRIDE_TARGET))
            step_x = max(32, min(_target_stride, max(1, cw - rw)))
            step_y = max(32, min(_target_stride, max(1, ch - rh)))

            for wy in range(0, ch - rh + 1, step_y):
                for wx in range(0, cw - rw + 1, step_x):
                    window = os_canvas[wy : wy + rh, wx : wx + rw]
                    mkpts0, mkpts1, _ = run_minima(matcher, resized_map, window)
                    affine_H, n_inliers, inlier_mask = estimate_affine(mkpts0, mkpts1)
                    n_windows += 1

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
                    scale_H = _build_scale_H(affine_H, wx, wy, sf)
                    center_ll = affine_center_to_latlon(scale_H, map_h, map_w, tile_info)
                    # Inlier keypoints (map coords) for the composite reranker.
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
                            "n_inliers": n_inliers,
                            "scale_factor": round(sf, 3),
                            "avg_scale": round(avg_scale_now, 4),
                            "window": (wx, wy),
                            "center_latlon": center_ll,
                            "anchor_latlon": (float(clat), float(clon)),
                            "_inlier_pts_map": inlier_pts_map,
                            "_rot_map_shape": (rh, rw),
                        },
                        "_sam3_mask": sam3_mask,
                    }
                    _seq += 1
                    bucket_key = (cname, zoom)
                    bucket = per_bucket.setdefault(bucket_key, [])
                    if len(bucket) < PER_BUCKET:
                        heapq.heappush(bucket, (metric, _seq, candidate))
                    elif metric > bucket[0][0]:
                        heapq.heapreplace(bucket, (metric, _seq, candidate))

            if n_windows > 0:
                print(f"    z{zoom}:{cname}: {n_windows}w, best={best_metric:.1f}", flush=True)

    # Flatten buckets → global top-K.
    all_candidates: List[Tuple[float, int, dict]] = []
    for bucket in per_bucket.values():
        all_candidates.extend(bucket)
    if not all_candidates:
        return {
            "geojson": None,
            "affine_H": None,
            "tile_info": None,
            "match_info": {},
        }
    all_candidates.sort(key=lambda x: -x[0])

    ranked = all_candidates[:MAX_CANDIDATES]

    # Composite rescore: pick by V × Q/4 (composite_window_score, this module).
    rescored = []
    for metric, seq, cand in ranked:
        mi = cand.get("match_info") or {}
        q = quadrant_coverage_from_inlier_points(
            mi.get("_inlier_pts_map") or [],
            mi.get("_rot_map_shape"),
        )
        composite_score = composite_window_score(metric, q)
        cand["_vanilla_metric"] = metric
        cand["_quadrant_cov"] = q
        rescored.append((composite_score, seq, cand))
    rescored.sort(key=lambda x: -x[0])
    ranked = rescored
    top = ranked[0][2]
    print(
        f"  Composite rerank: top score={ranked[0][0]:.2f} "
        f"(V={top.get('_vanilla_metric', 0):.2f} "
        f"Q={top.get('_quadrant_cov', 0)})"
    )

    # Road-name verifier: re-rank by metric * (1 + road_match_ratio)^2.
    best_result = None
    if road_names and len(road_names) >= 1:
        best_result = _verify_candidates_with_road_names(ranked, road_names)
    if best_result is None:
        _, _, best_result = ranked[0]

    # Project mask now (deferred from inner loop).
    cur_mask = best_result.pop("_sam3_mask", None)
    if sam3_mask is not None and cur_mask is not None:
        best_result["geojson"] = mask_to_geojson_affine(
            cur_mask, best_result["affine_H"], best_result["tile_info"]
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
        rh, rw = rot_shape
        cx, cy = rw / 2.0, rh / 2.0
        arr = np.asarray(inlier_pts_map)
        return (
            int(((arr[:, 0] < cx) & (arr[:, 1] < cy)).any())
            + int(((arr[:, 0] >= cx) & (arr[:, 1] < cy)).any())
            + int(((arr[:, 0] < cx) & (arr[:, 1] >= cy)).any())
            + int(((arr[:, 0] >= cx) & (arr[:, 1] >= cy)).any())
        )
    except Exception as e:
        print(
            f"  warn: quadrant coverage failed for {len(inlier_pts_map)} pts, "
            f"shape {rot_shape} ({e!s:.80}); treating as full coverage"
        )
        return 4


# Generic source-side σ floor used by ``effective_sigma`` when the
# worker omits σ. The live locate sub-agent's picks always carry a σ
# directly, so this only matters for the rare fallback path.
_FALLBACK_SIGMA_M = 5000


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
    diag_mm = math.sqrt(297**2 + 210**2)  # A4 landscape
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
    s = float(avg_scale or 0.0)
    if s <= 0:
        return AxisResult(score=0.0, verdict="invalid (avg_scale ≤ 0)")

    reader_provided = bool(reader_scale_text and _SCALE_PATTERN_RE.search(str(reader_scale_text)))
    score = min(s, 1.0 / s) ** 2

    if reader_provided:
        v = f"scale_consistency={score:.2f} (avg_scale={s:.3f}, reader said {reader_scale_text!r})"
    else:
        v = f"scale_consistency={score:.2f} (avg_scale={s:.3f}, no reader scale)"

    return AxisResult(score=score, verdict=v)


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
    n_total = len(reader_road_names or [])
    if n_total == 0:
        return AxisResult(score=0.5, verdict="no road names extracted by reader (no signal)")

    nearby = _query_gpkg_road_names(chosen_lat, chosen_lon, radius_m=1500.0)
    if not nearby:
        return AxisResult(
            score=0.5,
            verdict=(
                "no OS roads within radius — sparse cartography "
                "(rural / unlabelled), neutral signal"
            ),
        )

    matched: List[str] = []
    for rn in reader_road_names:
        if _fuzzy_road_match(rn, nearby):
            matched.append(rn)

    # Score is the raw match ratio; the verdict is just human-readable
    # context. Resist the urge to add tier thresholds here — the critic
    # reads the score directly, and any "strong/partial/weak" labels would
    # be arbitrary cutoffs masquerading as principled signal.
    n_matched = len(matched)
    score = n_matched / n_total
    if score == 0:
        v = (
            f"OS roads present here but ZERO of {n_total} reader roads "
            f"match — possible wrong-area signal (trust strong inliers "
            f"over this)"
        )
    else:
        v = f"{n_matched}/{n_total} reader roads found in OS"

    return AxisResult(score=score, verdict=v)


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
    center_ll = match_info.get("center_latlon")

    axes: Dict[str, AxisResult] = {
        "scale_consistency": axis_scale_consistency(
            avg_scale, reader_scale_text=pdf_info.get("scale")
        ),
    }

    if center_ll and len(center_ll) == 2:
        axes["road_name_agreement"] = axis_road_name_agreement(
            float(center_ll[0]),
            float(center_ll[1]),
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

    n_road = len(road_names)
    scored = []
    for metric, _seq, candidate in ranked_candidates:
        center_ll = candidate["match_info"].get("center_latlon")
        if not center_ll:
            scored.append((metric, metric, candidate, None))
            continue
        lat, lon = center_ll
        nearby = _query_gpkg_road_names(lat, lon, radius_m=1500)
        if not nearby:
            scored.append((metric, metric, candidate, None))
            continue
        matches = sum(1 for rn in road_names if _fuzzy_road_match(rn, nearby))
        ratio = matches / n_road
        boosted = metric * (1.0 + ratio) ** 2
        scored.append((boosted, metric, candidate, matches))

    if not scored:
        return None

    for boosted, orig, cand, matches in scored:
        cname = cand["match_info"]["center"]
        inliers = cand["match_info"]["n_inliers"]
        m_str = "n/a" if matches is None else f"{matches}/{n_road}"
        print(
            f"    Road verify: {cname} inl={inliers} "
            f"metric={orig:.1f} boosted={boosted:.1f} roads={m_str}"
        )

    scored.sort(key=lambda r: -r[0])
    top_cand = scored[0][2]
    metric_best_cand = ranked_candidates[0][2]
    if top_cand is metric_best_cand:
        print("    Road verify: top candidate confirmed")
        return None
    cname = top_cand["match_info"]["center"]
    print(f"    Road verify: OVERRIDE → {cname} (boosted {scored[0][0]:.1f})")
    return top_cand
