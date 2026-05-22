"""MINIMA sliding-window positioning on OS OpenData tiles.

Production pipeline for georeferencing planning maps:
  1. Filter geocoding centers (UK bbox, outlier removal, dedup)
  2. Compute scale/zoom configs from scale_ratio + DPI
  3. For each center x zoom x rotation:
     - Resize map to match tile pixel scale
     - Fetch OS OpenData tile grid
     - Slide map across tile canvas
     - Run MINIMA feature matching at each window
     - Estimate affine via RANSAC (4-DOF similarity)
     - Score: n_inliers, re-ranked by quadrant coverage
  4. Best scoring match -> build affine -> convert mask to GeoJSON
"""

import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Repo root is three levels up from tools/matching/_core.py
# (was two before the matching.py → tools/matching/ package split).
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# OS Zoomstack background-colour skip removed 2026-05-12: it regressed rural
# cases where 100% of candidate windows have >=85% background pixels (the
# loop body then never called MINIMA, so no matches were found at all). The
# claimed "30% wall-time saving" came from the v19 simulation, but that sim
# re-scored CACHED match attempts — it never exercised the live skip loop.
# Restoring the full sliding-window search; live wall-clock cost is bearable.


# ── Tuned constants (empirical) ──────────────────────────────────────────────
#
# These thresholds were empirically tuned against the 211-case cached MINIMA
# sweep on v3 benchmark output. The original tuning scripts lived in
# `overnight/` (gitignored, since deleted); the v3 per-case stats remain in
# `results/benchmark_v3/gemini-flash/<case>/metrics.json` and the bands here
# can be re-derived from them. See per-constant comments for the specific
# regression / case the value was calibrated against.

# (2026-05-21) 6-DOF affine fallback removed entirely after a 25-case
# ablation showed it nets to -0.01 mean IoU and rescues only ~2 cases
# (Ar4.15, Art4D04) at the cost of code complexity. The 4-DOF
# similarity transform is the geometrically correct prior for
# map-to-map matching anyway — shear is an artifact of photography or
# photocopying, not of real-world geometry.

# Sliding-window stride target: ~100 windows per (center, zoom, rotation).
# Same coverage density independent of rotation; conditional rotation is
# what bounds compute. 32-px floor below = ~48 m at z18, fine enough for
# MINIMA's spatial accuracy. The prior 128-px floor evaluated only ~1
# window per 192 m × 192 m square, leaving sub-tile match positions
# untested.
WINDOW_STRIDE_TARGET = 100


# ── MINIMA model management ──────────────────────────────────────────────────

def load_minima(base_dir=None):
    """Load MINIMA LoFTR matcher model.

    Args:
        base_dir: Repository root directory. Defaults to parent of tools/.
    """
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
    """Run MINIMA matching between map and tile images.

    Args:
        matcher: MINIMA matcher from load_minima().
        map_img: Map image (BGR, RGBA, or grayscale).
        tile_img: Tile image (BGR or grayscale).
        grayscale: If True, convert both images to grayscale before matching.
            Improves matching for B&W or sepia-tinted maps against coloured tiles.

    Returns (mkpts0, mkpts1, mconf) — matched keypoints and confidence.
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

    (2026-05-21) The 6-DOF fallback was removed after a 25-case
    ablation showed it nets to -0.01 mean IoU. The 4-DOF similarity is
    the geometrically correct prior for map-to-map matching; shear
    only shows up as an artifact of photography/photocopying and the
    rescue cases (~2 cases on the 208-case eval) don't justify the
    code complexity.

    (2026-05-21) The Delaunay-consistency refit used to live here too,
    deleted in the same pass — it provided zero mean benefit and was
    actively hurting the highest-inlier stress case.

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


# ── Scale and zoom utilities ─────────────────────────────────────────────────

# Web-Mercator math and lat/lon <-> tile-pixel projections moved to
# `tools/geo/coords.py` (2026-05-11) to deduplicate the formula that was repeated
# in 6 places across positioning, agent, and os_opendata_tiles. Re-exported
# here so existing `from tools.matching import compute_map_mpp` callers
# keep working.
from tools.geo.coords import (
    WEB_MERCATOR_C,
    best_zoom_for_scale,
    compute_map_mpp,
    haversine_m,
    latlon_to_global_tile_pixel,
    osm_pixel_to_latlon,
    tile_mpp as _tile_mpp_at,
)

# Legacy underscore-prefixed alias retained for internal callers.
_latlon_to_global_tile_pixel = latlon_to_global_tile_pixel


# Source-registry tables and lookups live in tools/matching/source_priorities.py
# (matching config, not geocoding). Re-exported here so existing imports such
# as `from tools.matching import sigma_from_source, _SOURCE_SIGMA_M,
# _FILTERABLE_SOURCES` keep working.
from tools.matching.source_priorities import (
    _FILTERABLE_SOURCES,
    _SOURCE_PRIORITY,
    _SOURCE_SIGMA_M,
    SOURCE_PRIORITY,
    candidate_passes_la_filter,
    candidate_la_distance_km,
    effective_sigma,
    sigma_from_scale,
    sigma_from_source,
    source_priority,
)

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
    # INTER_AREA optimal for downscale (sf<1); INTER_CUBIC for upscale (sf>1).
    # The previous code always used INTER_AREA, which blurs upscaled output
    # and hurts SuperPoint keypoint repeatability — per offline audit, roughly
    # half of "unknown scale" configs upscale, and the 0.85 / 1.15 perturb
    # paths often upscale too.
    interp = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(map_img, (new_w, new_h), interpolation=interp)
    return resized, scale_factor


# ── Coordinate transform and GeoJSON ─────────────────────────────────────────

# `osm_pixel_to_latlon` moved to `tools/geo/coords.py` and re-imported above so
# `from tools.matching import osm_pixel_to_latlon` keeps working.


def affine_center_to_latlon(affine_H, map_h, map_w, tile_info):
    """Apply affine to map center, convert to lat/lon."""
    cp = affine_H @ np.array([map_w / 2, map_h / 2, 1.0])
    return osm_pixel_to_latlon(
        cp[0], cp[1], tile_info["zoom"],
        tile_info["tx_min"], tile_info["ty_min"],
    )


# Mask cleanup primitives moved to `tools/extraction/mask_ops.py`
# (originally extracted from this file on 2026-05-11). The legacy
# underscore-prefixed names are kept as module-level aliases so
# external imports like `from tools.matching import _expand_thin_mask`
# keep working.
from tools.extraction.mask_ops import (
    cleanup_mask_pipeline,
    expand_thin_mask as _expand_thin_mask,
    fill_mask_holes as _fill_mask_holes,
    keep_dominant_components as _keep_dominant_components,
)


def mask_to_geojson_affine(mask, affine_H, tile_info, simplify_px=0.0):
    """Convert SAM3 mask to GeoJSON Feature using affine transform.

    Args:
        mask: Binary boundary mask (uint8).
        affine_H: 2x3 affine matrix mapping mask pixels to tile canvas pixels.
        tile_info: Dict with zoom, tx_min, ty_min from fetch_os_opendata_grid.
        simplify_px: Douglas-Peucker epsilon in pixels. Default 0.0 means
            no simplification — the polygon retains every contour vertex.
            The previous 3.0 default was an arbitrary "clean segments"
            smoothing that had no principled basis.

    Returns GeoJSON Feature dict, or None if no valid contours.
    """
    # Drop tiny noise components before any other processing. Targets cases
    # like v12 8FB7 where SAM returns 1 main blob + dozens of scattered noise
    # specks; the noise inflates predicted area without GT overlap.
    mask = _keep_dominant_components(mask)

    # Expand thin outline masks into filled regions before hole-filling.
    # SAM3 often traces boundary lines rather than selecting filled areas.
    mask = _expand_thin_mask(mask)

    # Fill internal holes (roads, text gaps) before extracting contours.
    # This prevents fragmented multi-polygon output.
    filled_mask = _fill_mask_holes(mask)

    contours, _ = cv2.findContours(
        (filled_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    zoom = tile_info["zoom"]
    tx_min = tile_info["tx_min"]
    ty_min = tile_info["ty_min"]

    all_polys = []
    for contour in contours:
        # Any contour the mask cleanup left intact is projected. The
        # previous "< 100 pixels" floor was an arbitrary noise filter
        # already covered by _keep_dominant_components above; keeping
        # both was redundant.
        if simplify_px > 0:
            contour = cv2.approxPolyDP(contour, simplify_px, True)
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


# ── Internal helpers ─────────────────────────────────────────────────────────

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
from tools.matching.road_verify import _verify_candidates_with_road_names


# ── Main entry point ─────────────────────────────────────────────────────────

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
    """Position a map on OS tiles using sliding-window MINIMA matching.

    This is the production entry point. No ground truth, no IoU computation.

    Args:
        matcher: MINIMA matcher from load_minima().
        map_img: Map image (numpy array, BGR or RGBA).
        sam3_mask: Binary boundary mask from SAM3 (uint8). Optional — if None,
                   matching still runs but geojson will be None. Use
                   mask_to_geojson_affine() afterwards to project masks.
        tile_fetcher: Function(lat, lon, zoom, nx, ny) → tile_info dict.
                      Default: fetch_os_opendata_grid (modern OS tiles).
        centers: 1-element list ``[(name, lat, lon, sigma_m)]`` returned by
                 the agentic locate sub-agent and passed through by the
                 worker's ``match_at`` tool. The list shape is historical;
                 only the first entry is used.
        scale_ratio: Map scale ratio (e.g. 2500 for 1:2500). None = try common scales.
        dpi: DPI used to render the PDF page.
        rotations: List of rotation angles to try, e.g. [0] or [0, 90, 270].
                   None defaults to [0].

    Returns:
        dict with keys:
            geojson: GeoJSON Feature or None (None if sam3_mask not provided)
            affine_H: final 2x3 affine matrix (or None)
            tile_info: tile grid metadata dict
            match_info: dict with center, zoom, rotation, n_inliers, score, etc.
            n_windows: total windows evaluated
    """
    if tile_fetcher is None:
        from tools.io.os_tiles import fetch_os_opendata_grid
        tile_fetcher = fetch_os_opendata_grid

    if not centers:
        return {
            "geojson": None, "affine_H": None, "tile_info": None,
            "match_info": {}, "n_windows": 0,
        }

    # Respect the input σ — the locate sub-agent's σ has Spearman ρ=+0.629
    # against actual pick→GT error on v3 (tight σ → small error, wide σ →
    # large error). The previous default-to-effective_sigma() overwrite
    # always landed at the 5km fallback because `live_locate:*` isn't
    # registered in _SOURCE_SIGMA_M. Fall back to effective_sigma only
    # when σ is missing or non-positive.
    name, lat, lon, sigma_in = centers[0]
    if sigma_in is None or float(sigma_in) <= 0:
        sigma_in = effective_sigma(name, scale_ratio)
    centers = [(name, lat, lon, float(sigma_in))]

    # Track top-N candidates for post-verification (road name check).
    # Per-(center, zoom) bucket caps to PER_BUCKET. Without this cap one
    # (center, zoom) sweep can fill all 5 slots with near-duplicate
    # wrong-area windows, hiding a correct-area window from a different
    # config (seen in Ar4.17/Ar4.18/ED3ECD0D where the heap held 5
    # variants of the same wrong window). Final top-5 is the union of
    # buckets sorted by metric.
    import heapq
    # Top-K cap on diversity-bucketed candidates. PER_BUCKET=1 (top per
    # (anchor, zoom)) + MAX_CANDIDATES=5 (global top) gives diverse pool
    # without dup candidates from same location. Validated +5 cases at IoU≥0.8
    # on the 211-case cached sweep vs old PB=2 baseline.
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
        # +/-15% scale perturbation at best zoom (handles DPI/metadata errors)
        zoom_mpp_configs.append((best_z, map_mpp * 0.85))
        zoom_mpp_configs.append((best_z, map_mpp * 1.15))
    else:
        # Unknown-scale path: sweep across common UK planning-map scales
        # (1:1250 to 1:25000), one canonical zoom per scale.
        common_scales = [1250, 2500, 5000, 10000, 15000, 25000]
        zoom_mpp_configs = []
        seen = set()
        for sr in common_scales:
            mpp = compute_map_mpp(sr, dpi)
            z = best_zoom_for_scale(mpp, ref_lat)
            if z not in seen:
                seen.add(z)
                zoom_mpp_configs.append((z, mpp))

        # Add ±15% scale perturbations on the modal scale (1:2500 by default).
        # Scales like 1:3500 or 1:7000 fall between the canonical grid points.
        modal_mpp = compute_map_mpp(2500, dpi)
        modal_z = best_zoom_for_scale(modal_mpp, ref_lat)
        zoom_mpp_configs.append((modal_z, modal_mpp * 0.85))
        zoom_mpp_configs.append((modal_z, modal_mpp * 1.15))

    if rotations is None:
        # Single orientation. Rotation detection happens upstream:
        # tools.io.map_page.render_map_page runs the auto-rotation classifier
        # (k-fold ResNet50, TTA) so by the time MINIMA sees the image it is
        # already upright. There is no per-window rotation search at match time.
        rotations = [0]

    # No early termination — the EARLY_STOP_METRIC=75.0 threshold was
    # a tuning knob calibrated to a single sweep with no principled basis.
    # Running all (centre, zoom) configurations is slower but removes a
    # magic-constant cutoff.

    # Sort centers by sigma (tightest first) so the most-confident
    # anchors are explored first.
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

            # Tile grid: cover map + search margin. Use the center's sigma
            # directly (NO hardcoded floor). For small-scale maps (plot-level)
            # sigma is 300m, for large-area maps (rural/district) sigma is
            # 2-4km. The 2000m floor that was here before negated scale-aware
            # sigma entirely — MINIMA always searched a ~2km window regardless
            # of map scale, causing wrong matches far from good geocoded centers.
            search_m = sigma if sigma else 1000
            margin_tiles = max(2, int(math.ceil(search_m / (256 * tmpp))))
            nx_needed = int(math.ceil(rw / 256)) + 2 * margin_tiles
            ny_needed = int(math.ceil(rh / 256)) + 2 * margin_tiles
            # Grid floor/ceiling. The 211-case sweep showed 3/17 is tight
            # enough for all observed (zoom, sigma) combinations; ~5% wall
            # time saved vs the loose 5/35 we originally used.
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

                # Stride is determined by canvas size alone — target ~100
                # windows per (center, zoom, rotation). Same sampling density
                # regardless of rotation; the rotation angle doesn't change
                # what "enough coverage" looks like. Conditional rotation
                # (above) is what keeps compute bounded by only sweeping
                # 90/180/270 when rotation=0 fails.
                #
                # Floor at 32 px (was 128). 128 px at z18 = ~192 m of ground;
                # the previous floor meant we evaluated only ~1 window inside
                # any 192 m × 192 m square, so sub-tile match positions were
                # never tested. 32 px = ~48 m at z18, fine enough for the
                # MINIMA matcher's spatial accuracy.
                _area_available = max(1, (ch - rot_h) * (cw - rot_w))
                _target_stride = int(math.sqrt(_area_available / WINDOW_STRIDE_TARGET))
                step_x = max(32, min(_target_stride, max(1, cw - rot_w)))
                step_y = max(32, min(_target_stride, max(1, ch - rot_h)))

                # Background-window skipping removed 2026-05-12. The
                # threshold (>=85% OS-background pixels) was excluding 100% of
                # candidate windows for rural cases (e.g. 1D1 East Langdon),
                # so the matcher tested zero windows and regressed. The v19
                # sim that "validated" bg-skip was re-scoring cached match
                # attempts and never exercised this loop. Skip the early-exit;
                # let MINIMA evaluate every window.
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

                        # avg_scale is the geometric mean of the affine's
                        # column norms. With 4-DOF similarity (the only
                        # path now) sx == sy and avg_scale equals the
                        # uniform scale factor. Kept around because the
                        # scale_consistency reward axis reads it.
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
                            # Inlier keypoints in MAP coords feed the composite
                            # reranker (quadrant coverage). Reprojection errors
                            # were removed: only stale MAGSAC offline tests
                            # read them, and the norm+tolist hot allocation
                            # paid per successful window adds up.
                            inlier_pts_map = None
                            if inlier_mask is not None:
                                try:
                                    flag = inlier_mask.ravel().astype(bool)
                                    in0 = mkpts0[flag]
                                    if len(in0) > 0:
                                        # in0 is in rot_map coords (post-resize,
                                        # post-rotate); save as (x, y) pairs.
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
                                    "_inlier_pts_map": inlier_pts_map,  # offline-only
                                    "_rot_map_shape": (rot_h, rot_w),  # offline-only
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

    # Flatten per-(center, zoom) buckets and take the global top-N.
    # This is the "diversity-capped top-K" — at most PER_BUCKET (2) entries
    # from any one (center, zoom) sweep survive into post-verification.
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

    # Composite rescoring: pick by V × Q/4 instead of raw vanilla.
    # See tools.scoring.composite_window_score for the formula and history.
    # The km_to_anchor factor was dropped 2026-05-21 after the ablation
    # showed it contributed zero mean Δ across the distance-stress sample.
    if ranked:
        from tools.scoring import (
            composite_window_score,
            quadrant_coverage_from_inlier_points,
        )
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

    # Road-name verifier: if the reader extracted road names, prefer the
    # candidate whose nearby OSM road names overlap with them. Only fires
    # under the triple-gated override (≥60% match AND ≥2× top ratio AND
    # ≥70% top metric). Catches wrong-LA picks where MINIMA finds a
    # high-inlier window in the wrong town. Directional verifier remains
    # ripped — only road-name verification is wired here.
    best_result = None
    if road_names and len(road_names) >= 1:
        best_result = _verify_candidates_with_road_names(ranked, road_names)
    if best_result is None:
        _, _, best_result = ranked[0]

    # Project mask now (deferred from inner loop for efficiency).
    # When return_candidates=True we keep the per-candidate masks on `ranked`
    # so the caller can project each one independently — so we take a ref
    # to best_result's mask before popping it off.
    cur_mask = best_result.get("_sam3_mask") if return_candidates \
        else best_result.pop("_sam3_mask", None)
    if not return_candidates:
        best_result.pop("_metric", None)
    if sam3_mask is not None and cur_mask is not None:
        best_result["geojson"] = mask_to_geojson_affine(
            cur_mask, best_result["affine_H"], best_result["tile_info"])

    best_result["n_windows"] = total_windows

    if return_candidates:
        # Offline/analysis hook: expose the top-K ranked candidates so callers
        # can evaluate them externally (e.g. per-candidate IoU vs ground truth,
        # or re-ranking with a verifier). Each candidate carries its own mask
        # (already rotated) and affine. `ranked` is post-specificity re-rank.
        out_candidates = []
        for metric, _, cand in ranked:
            cand = dict(cand)
            cand["sam3_mask"] = cand.pop("_sam3_mask", None)
            cand["metric"] = cand.pop("_metric", metric)
            out_candidates.append(cand)
        best_result["candidates"] = out_candidates

    return best_result
