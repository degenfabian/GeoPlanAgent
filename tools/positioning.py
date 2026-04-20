"""MINIMA sliding-window positioning on OS OpenData tiles.

Production pipeline for georeferencing planning maps:
  1. Filter geocoding centers (UK bbox, outlier removal, dedup)
  2. Compute scale/zoom configs from scale_ratio + DPI
  3. For each center x zoom x rotation:
     - Resize map to match tile pixel scale
     - Fetch OS OpenData tile grid
     - Slide map across tile canvas
     - Run MINIMA feature matching at each window
     - Estimate affine via RANSAC
     - Score: n_inliers x aspect (GT-free metric)
  4. Best scoring match -> build affine -> convert mask to GeoJSON
"""

import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent


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
    """Estimate affine transform via RANSAC.

    Returns (H, n_inliers, score, inlier_mask) where score is the sum of
    MINIMA confidence values for inlier matches.
    """
    if len(mkpts0) < 4:
        return None, 0, 0.0, None
    H, inlier_mask = cv2.estimateAffine2D(
        mkpts0, mkpts1, method=cv2.RANSAC,
        ransacReprojThreshold=reproj_thresh,
    )
    if H is None:
        return None, 0, 0.0, None
    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0

    if mconf is not None and inlier_mask is not None and n_inliers > 0:
        inlier_flags = inlier_mask.ravel().astype(bool)
        score = float(np.sum(mconf[inlier_flags]))
    else:
        score = float(n_inliers)

    return H, n_inliers, score, inlier_mask


# ── Scale and zoom utilities ─────────────────────────────────────────────────

def compute_map_mpp(scale_ratio, dpi=200):
    """Compute meters per pixel for a map at given scale ratio and DPI.

    At 1:2500 and 200 DPI: 1 pixel = 25.4/200 mm = 0.127mm on paper.
    Ground meters per pixel = (25.4 / dpi) / 1000 * scale_ratio.
    """
    if scale_ratio is None:
        return None
    mm_per_px = 25.4 / dpi
    return mm_per_px / 1000.0 * scale_ratio


def best_zoom_for_scale(map_mpp, lat):
    """Find the tile zoom level closest to the map's meters per pixel."""
    if map_mpp is None:
        return None
    z = math.log2(156543.03 * math.cos(math.radians(lat)) / map_mpp)
    return max(15, min(19, round(z)))


def sigma_from_scale(scale_ratio, page_mm=(297, 210)):
    """Compute recommended search sigma (meters) for a given map scale.

    Derivation: for a 1:N map printed on A4 (297×210mm), the real-world extent
    is (0.297 × N) × (0.21 × N) meters. Half of the diagonal =
    0.5 × sqrt(0.297² + 0.21²) × N ≈ 0.18 × N. This is how far the map CENTER
    can be from an exactly-on-site geocoded point (site at the corner of the
    map). Sigma must cover that distance for MINIMA to find the match.

    Validated against empirical offset measurements on 108 passing cases:
      - zoom 19 maps: observed p90 offset ~105m → sigma ~230m ok
      - zoom 17 maps: observed p90 offset ~350m → sigma ~450m ok
      - zoom 15 maps: observed p90 offset ~2000m → sigma ~4500m needed

    Args:
        scale_ratio: Map scale denominator (e.g., 2500 for 1:2500). None if unknown.
        page_mm: Paper size (default A4 landscape).

    Returns:
        Recommended sigma in meters. Floor of 300m, no explicit upper cap
        (10km is the practical max for 1:55000 maps).
    """
    if scale_ratio is None:
        # Unknown scale: use a conservative default. Covers ~1:8000 which is
        # the median in our dataset when scale IS known.
        return 1500
    diag_mm = math.sqrt(page_mm[0] ** 2 + page_mm[1] ** 2)
    half_diag_m = 0.5 * (diag_mm / 1000.0) * scale_ratio
    return max(300, int(half_diag_m))


def resize_map_to_match_zoom(map_img, map_mpp, zoom, lat):
    """Resize map so its pixel scale matches the tile pixel scale at given zoom.

    Returns (resized_img, scale_factor) where scale_factor is the resize ratio.
    Returns (None, scale_factor) if the scale difference is too large.
    """
    tile_mpp = 156543.03 * math.cos(math.radians(lat)) / (2 ** zoom)
    scale_factor = map_mpp / tile_mpp
    if scale_factor < 0.3 or scale_factor > 3.0:
        return None, scale_factor
    new_h = int(map_img.shape[0] * scale_factor)
    new_w = int(map_img.shape[1] * scale_factor)
    if new_h < 64 or new_w < 64:
        return None, scale_factor
    resized = cv2.resize(map_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale_factor


# ── Coordinate transform and GeoJSON ─────────────────────────────────────────

def osm_pixel_to_latlon(px, py, zoom, tx_min, ty_min, tile_size=256):
    """Convert pixel position on tile canvas to lat/lon."""
    n = 2 ** zoom
    global_px = tx_min * tile_size + px
    global_py = ty_min * tile_size + py
    lon = global_px / (n * tile_size) * 360 - 180
    lat = math.degrees(math.atan(math.sinh(
        math.pi * (1 - 2 * global_py / (n * tile_size)))))
    return lat, lon


def affine_center_to_latlon(affine_H, map_h, map_w, tile_info):
    """Apply affine to map center, convert to lat/lon."""
    cp = affine_H @ np.array([map_w / 2, map_h / 2, 1.0])
    return osm_pixel_to_latlon(
        cp[0], cp[1], tile_info["zoom"],
        tile_info["tx_min"], tile_info["ty_min"],
    )


def _fill_mask_holes(mask):
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


def _expand_thin_mask(mask):
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


def mask_to_geojson_affine(mask, affine_H, tile_info, simplify_px=3.0):
    """Convert SAM3 mask to GeoJSON Feature using affine transform.

    Args:
        mask: Binary boundary mask (uint8).
        affine_H: 2x3 affine matrix mapping mask pixels to tile canvas pixels.
        tile_info: Dict with zoom, tx_min, ty_min from fetch_os_opendata_grid.
        simplify_px: Douglas-Peucker epsilon in pixels (3.0 = clean segments).

    Returns GeoJSON Feature dict, or None if no valid contours.
    """
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
        if cv2.contourArea(contour) < 100:
            continue
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


# ── Center filtering ─────────────────────────────────────────────────────────

def filter_centers(centers, max_centers=5, max_dist_km=50):
    """Filter outlier centers and cap to max_centers. Production-safe (no GT).

    Source-aware: trusted sources (gridref, postcode, nominatim, photon) are
    used to build consensus. Untrusted sources (place names) are filtered
    against the consensus.

    Steps:
    1. Classify centers as trusted vs untrusted
    2. Compute consensus median from trusted sources
    3. Validate gridrefs against non-gridref consensus; drop untrusted
       centers that are >max_dist_km from consensus
    4. Sort by sigma, cap at max_centers
    """
    if len(centers) <= 1:
        return centers

    from tools.geocoding import _distance_m

    trusted = []
    untrusted = []
    for c in centers:
        name = c[0].lower()
        if name.startswith("place_") or name.startswith("ve_"):
            untrusted.append(c)
        else:
            trusted.append(c)

    if len(trusted) >= 2:
        ref_lats = [c[1] for c in trusted]
        ref_lons = [c[2] for c in trusted]
        ref_lat = float(np.median(ref_lats))
        ref_lon = float(np.median(ref_lons))

        non_gridref_trusted = [c for c in trusted if "gridref" not in c[0]]
        if len(non_gridref_trusted) >= 2:
            ngr_lat = float(np.median([c[1] for c in non_gridref_trusted]))
            ngr_lon = float(np.median([c[2] for c in non_gridref_trusted]))
            kept_trusted = list(non_gridref_trusted)
            for c in trusted:
                if "gridref" in c[0]:
                    d = _distance_m(c[1], c[2], ngr_lat, ngr_lon)
                    if d < max_dist_km * 1000:
                        kept_trusted.append(c)
            trusted = kept_trusted
            ref_lat = float(np.median([c[1] for c in trusted]))
            ref_lon = float(np.median([c[2] for c in trusted]))

        kept_untrusted = []
        for c in untrusted:
            d = _distance_m(c[1], c[2], ref_lat, ref_lon)
            if d < max_dist_km * 1000:
                kept_untrusted.append(c)
        untrusted = kept_untrusted

    elif len(trusted) == 1:
        ref_lat, ref_lon = trusted[0][1], trusted[0][2]
        kept_untrusted = []
        for c in untrusted:
            d = _distance_m(c[1], c[2], ref_lat, ref_lon)
            if d < max_dist_km * 1000:
                kept_untrusted.append(c)
        untrusted = kept_untrusted

    all_valid = trusted + untrusted
    if not all_valid:
        return centers[:1]

    all_valid.sort(key=lambda c: c[3] if c[3] else 9999)
    return all_valid[:max_centers]


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


def _deduplicate_centers(centers, min_dist_m=500):
    """Remove centers within min_dist_m of each other.

    High-specificity anchors (rank ≤ 1: Nominatim street/addr, grid_refs,
    postcode) are exempt from dedup — each street returns a slightly
    different point and having 3 tight anchors is MORE informative than
    1 (their spatial consistency confirms the right geography). The
    A4_094:LL:013 regression (Leicester cluster of 3 nominatim hits within
    500m collapsing to a single point and starving MINIMA of signal) is
    exactly the reason for this carve-out.
    """
    from tools.geocoding import _distance_m
    deduped = []
    for c in centers:
        if _center_specificity(c[0]) <= 1:
            # Always keep rank-≤1 anchors; they're each their own signal
            deduped.append(c)
            continue
        if not any(_distance_m(c[1], c[2], d[1], d[2]) < min_dist_m for d in deduped):
            deduped.append(c)
    return deduped


# Lower rank = more specific. Used to deprioritise broad-area centers when
# street-level anchors are available.
# Ordered by expected geocode precision:
#   0  house-numbered Nominatim address (best)
#   1  Nominatim street+city, grid_refs_centroid, parsed grid ref, postcode
#   2  Zoomstack Town/City/Village/Hamlet/Suburb (point-level settlement)
#   3  Wikidata named settlement (moderate; can be administrative area)
#   4  Zoomstack Suburban Area/Small Settlements (broader neighbourhood)
#   5  Zoomstack Sites/Greenspace/Landform/Water (POI-style, often wrong-sense)
#   6  Zoomstack District/Borough/County/National Park (large admin area)
#   9  unknown (treated as broad)
_HIGH_SPECIFICITY_ZOOMSTACK = {"Town", "City", "Village", "Hamlet", "Suburb"}
_MID_SPECIFICITY_ZOOMSTACK = {"Suburban Area", "Small Settlements"}
_BROAD_ZOOMSTACK = {"District", "County", "Borough", "National Park",
                    "Region", "Country", "Unitary Authority", "Capital",
                    "Metropolitan County"}
_POI_ZOOMSTACK = {"Sites", "Greenspace", "Landform", "Water", "Woodland",
                  "Wetland"}


def _center_specificity(name: str) -> int:
    """Map a center's source-encoded name to a specificity rank (lower=better)."""
    if not isinstance(name, str):
        return 9
    n = name.lower()
    if n.startswith("nominatim:addr:"):
        return 0
    if n.startswith("nominatim:"):
        return 1
    if n.startswith("grid_refs_centroid") or n.startswith("gridref:"):
        return 1
    if n.startswith("postcode:"):
        return 1
    if n.startswith("gpkg:") and "(" in name and ")" in name:
        # Extract "(TYPE)" suffix
        t = name.rsplit("(", 1)[-1].rstrip(")")
        if t in _HIGH_SPECIFICITY_ZOOMSTACK:
            return 2
        if t in _MID_SPECIFICITY_ZOOMSTACK:
            return 4
        if t in _BROAD_ZOOMSTACK:
            return 6
        if t in _POI_ZOOMSTACK:
            return 5
        return 4  # unknown gpkg type → treat as mid
    if n.startswith("gpkg:"):
        return 4  # legacy name without type suffix
    if n.startswith("wikidata:"):
        return 3
    return 9


def filter_centers_by_specificity(centers, anchor_threshold=2,
                                   drop_above=4, min_keep=1):
    """When at least one center has specificity ≤ anchor_threshold (i.e. a
    street-level or grid-ref-quality anchor), drop centers with specificity
    > drop_above (broad-area admin/POI types). Leaves ≥ min_keep centers
    so MINIMA isn't starved.

    Rationale: in v3_flash, the dominant failure mode for IoU=0 accepted
    cases was MINIMA locking onto a broad-area / wrong-sense center
    (gpkg:Presbytery(Greenspace), gpkg:St Albans Church(Sites),
    wikidata:London Borough of Camden) even when a Nominatim street-level
    anchor was in the candidate list. With drop_above=4, ranks 5-6 are
    dropped (Zoomstack Sites/Greenspace/Woodland/Water/Landform and
    District/Borough/County/National Park). Wikidata (rank 3) and
    Zoomstack Town/City/Village/Hamlet/Suburb (rank 2) and Nominatim
    (rank 0-1) survive.
    """
    if len(centers) <= min_keep:
        return centers
    ranked = [(c, _center_specificity(c[0])) for c in centers]
    min_spec = min(s for _, s in ranked)
    if min_spec > anchor_threshold:
        # No high-confidence anchor — keep everything; MINIMA needs every
        # signal it can get.
        return centers
    kept = [c for c, s in ranked if s <= drop_above]
    dropped = [c for c, s in ranked if s > drop_above]
    if len(kept) < min_keep:
        # Filter was too aggressive; add back the least-broad dropped
        # centers until we hit min_keep.
        dropped_ranked = sorted(
            [(c, _center_specificity(c[0])) for c in dropped], key=lambda x: x[1])
        for c, _ in dropped_ranked:
            if len(kept) >= min_keep:
                break
            kept.append(c)
    if dropped:
        dropped_names = ", ".join(c[0] for c in dropped
                                  if c in (dropped[:6]))
        if len(dropped) > 6:
            dropped_names += f" +{len(dropped)-6} more"
        print(f"  Specificity filter: kept {len(kept)}/{len(centers)} "
              f"(dropped broad-area: {dropped_names})")
    return kept


# ── Road name verification ───────────────────────────────────────────────────

def _query_gpkg_road_names(lat, lon, radius_m=1500):
    """Query OS GeoPackage for road names near a point. Fully offline."""
    try:
        import geopandas as gpd
        import pyproj

        gpkg_path = BASE_DIR / "os_opendata" / "OS_Open_Zoomstack.gpkg"
        if not gpkg_path.exists():
            return []

        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", "EPSG:27700", always_xy=True)
        x, y = transformer.transform(lon, lat)

        names = set()
        for layer in ["roads_local", "roads_regional", "roads_national"]:
            try:
                gdf = gpd.read_file(
                    str(gpkg_path), layer=layer,
                    bbox=(x - radius_m, y - radius_m,
                          x + radius_m, y + radius_m))
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
        llm_norm = (llm_lower
                    .replace(" street", " st").replace(" road", " rd")
                    .replace(" lane", " ln").replace(" avenue", " ave")
                    .replace(" drive", " dr").replace(" close", " cl"))
        ref_norm = (ref_lower
                    .replace(" street", " st").replace(" road", " rd")
                    .replace(" lane", " ln").replace(" avenue", " ave")
                    .replace(" drive", " dr").replace(" close", " cl"))
        if llm_norm == ref_norm:
            return True
    return False


def _verify_candidates_with_road_names(ranked_candidates, road_names):
    """Pick the best candidate where nearby OSM roads match LLM road names.

    Only overrides the top candidate if:
    - Top candidate has NO road name matches AND
    - A lower-ranked candidate DOES have matches AND
    - That candidate has >= 50% of the top candidate's metric score

    Returns the verified candidate dict, or None to use default.
    """
    if not road_names:
        return None

    top_metric = ranked_candidates[0][0]
    min_metric = top_metric * 0.5  # candidate must be at least 50% as good

    results = []
    for metric, _seq, candidate in ranked_candidates:
        if metric < min_metric:
            break

        center_ll = candidate["match_info"].get("center_latlon")
        if not center_ll:
            results.append((metric, candidate, 0, 0))
            continue

        lat, lon = center_ll
        nearby_roads = _query_gpkg_road_names(lat, lon, radius_m=1500)

        if not nearby_roads:
            results.append((metric, candidate, 0, 0))
            continue

        matches = sum(1 for rn in road_names
                      if _fuzzy_road_match(rn, nearby_roads))
        results.append((metric, candidate, matches, len(road_names)))

    if not results:
        return None

    # Log verification results
    for metric, cand, matches, total in results:
        cname = cand["match_info"]["center"]
        inliers = cand["match_info"]["n_inliers"]
        print(f"    Road verify: {cname} inl={inliers} metric={metric:.1f} "
              f"roads={matches}/{total}")

    # Analyse all candidates by road-match quality.
    top_metric_v = results[0][0]
    top_cand = results[0][1]
    top_matches = results[0][2]
    top_total = results[0][3]
    top_ratio = top_matches / max(1, top_total)

    # Find the candidate with the best road-match ratio (ties broken by
    # higher metric). We compare this alternative to the top candidate.
    best_by_roads = max(results, key=lambda r: (r[2] / max(1, r[3]), r[0]))
    br_metric_v, br_cand, br_matches, br_total = best_by_roads
    br_ratio = br_matches / max(1, br_total)

    # Override rule: a candidate with DRAMATICALLY more road matches should
    # win even if its raw metric is slightly lower. Fixes cases where a
    # postcode-centroid or wikidata borough wins by raw inliers (1/9 roads)
    # but a Nominatim street anchor would win by 8/9 with perfect local
    # alignment. Conditions (all required):
    #   - it's NOT already the top candidate
    #   - its road-match ratio is ≥60% AND ≥ 2× top ratio
    #   - its metric is ≥70% of the top metric (not drastically worse)
    if (br_cand is not top_cand
        and br_ratio >= 0.6 and br_ratio >= 2 * top_ratio + 0.01
        and br_metric_v >= 0.7 * top_metric_v):
        cname = br_cand["match_info"]["center"]
        print(f"    Road verify: OVERRIDE → {cname} "
              f"({br_matches}/{br_total}={br_ratio:.0%} roads vs top "
              f"{top_matches}/{top_total}={top_ratio:.0%}, "
              f"metric={br_metric_v:.1f} vs {top_metric_v:.1f})")
        return br_cand

    # Fallback: legacy override when top has 0 matches but another does.
    # Only fire if the alternative's metric is close to top's (≥70%). Road
    # names in gpkg data can be sparse/noisy, especially in rural areas —
    # a partial road match (e.g. 2/4) is not enough to override a clearly-
    # better MINIMA match (e.g. inliers 39 vs 20 = 2x ratio).
    if top_matches == 0:
        for metric, candidate, matches, total in results[1:]:
            if matches == 0:
                continue
            if metric < 0.7 * top_metric_v:
                # This alternative is too weak metric-wise to prefer over top
                continue
            cname = candidate["match_info"]["center"]
            print(f"    Road verify: OVERRIDE (top had 0) → {cname} "
                  f"({matches}/{total} roads matched, "
                  f"metric={metric:.1f} vs top={top_metric_v:.1f})")
            return candidate
        print("    Road verify: top had 0 matches but alternatives "
              "too weak metric-wise, keeping top")
        return None

    print("    Road verify: top candidate confirmed")
    return None


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
        centers: List of (name, lat, lon, sigma_m) tuples from geocoding.
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
        from tools.os_opendata_tiles import fetch_os_opendata_grid
        tile_fetcher = fetch_os_opendata_grid

    # UK bounding box filter
    UK_LAT_MIN, UK_LAT_MAX = 49.0, 61.0
    UK_LON_MIN, UK_LON_MAX = -8.5, 2.0
    uk_centers = [c for c in centers
                  if UK_LAT_MIN <= c[1] <= UK_LAT_MAX and UK_LON_MIN <= c[2] <= UK_LON_MAX]
    if not uk_centers:
        uk_centers = centers[:1]

    # Cross-validate: drop centers >5km from median (catches bogus geocode
    # results AND bad grid-ref centroids). Tightened from 10km — for site-level
    # matching, anything >5km from the cluster of other centers is almost
    # certainly wrong. The adaptive IQR path still kicks in when ≥5 centers
    # cluster tightly.
    from tools.geocoding import cross_validate_centers
    uk_centers = cross_validate_centers(uk_centers, max_outlier_km=5)

    # Filter outliers, cap at 5
    centers = filter_centers(uk_centers, max_centers=5)

    # Deduplicate within 500m
    centers = _deduplicate_centers(centers, min_dist_m=500)

    # Specificity filter: when a street-level anchor exists, drop broad-area
    # admin/POI centers that tend to produce coincidental high-inlier matches
    # at wrong locations. Targets v3_flash IoU=0 failures like
    # gpkg:Presbytery(Greenspace), gpkg:St Albans Church(Sites),
    # gpkg:Wirral(District), wikidata:London Borough of Camden.
    centers = filter_centers_by_specificity(centers,
                                             anchor_threshold=2,
                                             drop_above=4, min_keep=1)

    if not centers:
        return {
            "geojson": None, "affine_H": None, "tile_info": None,
            "match_info": {}, "n_windows": 0,
        }

    # Scale-aware sigma: compute per-center based on both map extent AND
    # center specificity. Rationale:
    #   - High-specificity anchors (Nominatim street, grid-ref) are known
    #     to sit inside the mapped area, so map-center-to-anchor offset is
    #     bounded by ~half_map_diagonal. Use scale-derived sigma.
    #   - When scale is unknown, default sigma=1500m is too loose for
    #     street-level anchors and lets MINIMA wander to false positives
    #     1+ km away (seen on A4FNa1: nominatim anchor 0.2km from GT but
    #     MINIMA picked a match 1.3km off). Cap sigma at 800m for high-
    #     specificity anchors when scale is None.
    #   - Broad-area centers (gpkg Town, wikidata admin) need the full
    #     scale sigma because the mapped site may be anywhere in the
    #     admin area, not necessarily near the center.
    _scale_sigma = sigma_from_scale(scale_ratio)
    _high_spec_no_scale_cap = 800  # metres; only used when scale unknown
    new_centers = []
    for (n, la, lo, _) in centers:
        spec = _center_specificity(n)
        if spec <= 1 and scale_ratio is None:
            s = min(_scale_sigma, _high_spec_no_scale_cap)
        else:
            s = _scale_sigma
        new_centers.append((n, la, lo, s))
    centers = new_centers

    # Center clustering: if the surviving centers all agree tightly (within
    # ~500m of each other), collapse them to a single centroid to avoid 5-7×
    # redundant MINIMA searches around essentially-the-same-spot. If they
    # disagree (e.g., two true-positive clusters at different geographic sites),
    # keep them all so MINIMA picks the right one.
    if len(centers) >= 2:
        lats = [c[1] for c in centers]
        lons = [c[2] for c in centers]
        # Pairwise max distance (rough — use bounding-box diagonal)
        from tools.geocoding import _distance_m as _dist_m
        _max_pair = 0.0
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = _dist_m(centers[i][1], centers[i][2],
                            centers[j][1], centers[j][2])
                if d > _max_pair:
                    _max_pair = d
        # Collapse if spread is small relative to sigma; 500m is a tight
        # agreement threshold for site-level matching.
        if _max_pair <= 500:
            lat_c = sum(lats) / len(lats)
            lon_c = sum(lons) / len(lons)
            print(f"  Center clustering: {len(centers)} centers agree within "
                  f"{_max_pair:.0f}m — collapsing to centroid "
                  f"({lat_c:.5f}, {lon_c:.5f})")
            centers = [("consensus_centroid", lat_c, lon_c, _scale_sigma)]
        else:
            print(f"  Center clustering: centers spread {_max_pair:.0f}m > 500m, "
                  f"keeping all {len(centers)} for independent search")

    # Track top-N candidates for post-verification (road name check)
    import heapq
    MAX_CANDIDATES = 5
    top_candidates = []  # min-heap of (metric, seq, result_dict)
    _seq = 0  # tiebreaker for heap
    best_metric = 0
    best_result = None
    total_windows = 0

    map_mpp = compute_map_mpp(scale_ratio, dpi)
    map_h, map_w = map_img.shape[:2]

    # Determine (zoom, mpp) configs.
    # Env var GEOMAP_FAST=1 forces a single best-guess zoom per center —
    # used by offline tests to speed up iteration. Default (production)
    # explores multiple zooms and ±15% scale perturbations.
    _FAST = os.environ.get("GEOMAP_FAST") == "1"
    ref_lat = centers[0][1]
    if map_mpp is not None:
        best_z = best_zoom_for_scale(map_mpp, ref_lat)
        if _FAST:
            zoom_mpp_configs = [(best_z, map_mpp)]
        else:
            zoom_mpp_configs = [
                (z, map_mpp)
                for z in sorted(set([best_z, max(15, best_z - 1), min(19, best_z + 1)]))
            ]
            # +/-15% scale perturbation at best zoom (handles DPI/metadata errors)
            zoom_mpp_configs.append((best_z, map_mpp * 0.85))
            zoom_mpp_configs.append((best_z, map_mpp * 1.15))
    else:
        common_scales = [2500, 5000, 10000] if _FAST else \
                        [1250, 2500, 5000, 10000, 15000, 25000]
        zoom_mpp_configs = []
        seen = set()
        for sr in common_scales:
            mpp = compute_map_mpp(sr, dpi)
            z = best_zoom_for_scale(mpp, ref_lat)
            if z not in seen:
                seen.add(z)
                zoom_mpp_configs.append((z, mpp))

    if rotations is None:
        # Single orientation. Rotation detection happens upstream — the reader
        # phase populates PDFInfo.map_rotation, and run_agent pre-rotates the
        # map image once before SAM3/MINIMA see it. By the time we get here,
        # the map is already correctly oriented.
        rotations = [0]

    # Early termination: once we find an excellent match, skip remaining
    # centers/zooms to save time. The threshold is high enough to avoid
    # false positives (200 inliers with good aspect is very reliable).
    EARLY_STOP_METRIC = 150  # ~200 inliers * 0.95 aspect * 0.8 scale

    early_stopped = False
    for cname, clat, clon, sigma in centers:
        if early_stopped:
            break
        for zoom, cur_mpp in zoom_mpp_configs:
            if early_stopped:
                break
            tile_mpp = 156543.03 * math.cos(math.radians(clat)) / (2 ** zoom)

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
            margin_tiles = max(2, int(math.ceil(search_m / (256 * tile_mpp))))
            nx_needed = int(math.ceil(rw / 256)) + 2 * margin_tiles
            ny_needed = int(math.ceil(rh / 256)) + 2 * margin_tiles
            nx = max(5, min(35, nx_needed))
            ny = max(5, min(35, ny_needed))
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
                _target = 100
                _area_available = max(1, (ch - rot_h) * (cw - rot_w))
                _target_stride = int(math.sqrt(_area_available / _target))
                step_x = max(128, min(_target_stride, max(1, cw - rot_w)))
                step_y = max(128, min(_target_stride, max(1, ch - rot_h)))

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

                        # Aspect ratio from affine decomposition
                        a, b = affine_H[0, 0], affine_H[0, 1]
                        c_a, d = affine_H[1, 0], affine_H[1, 1]
                        sx = math.sqrt(a * a + c_a * c_a)
                        sy = math.sqrt(b * b + d * d)
                        aspect = min(sx, sy) / max(sx, sy) if sx > 0 and sy > 0 else 0

                        # Scoring: n_inliers * aspect, with rotation and scale penalties
                        rot_penalty = 1.0 if rot_angle == 0 else 1.1
                        scale_penalty = max(0.5, 1.0 - abs(sf - 1.0) * 0.5)
                        metric = (n_inliers / rot_penalty) * aspect * scale_penalty
                        if metric > best_metric:
                            best_metric = metric

                        # Keep top-N candidates for post-verification
                        if metric > 0:
                            scale_H = _build_scale_H(affine_H, wx, wy, sf)
                            center_ll = affine_center_to_latlon(
                                scale_H, map_h, map_w, tile_info)
                            avg_scale = (sx + sy) / 2
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
                                    "aspect": round(aspect, 4),
                                    "scale_factor": round(sf, 3),
                                    "avg_scale": round(avg_scale, 4),
                                    "window": (wx, wy),
                                    "center_latlon": center_ll,
                                },
                                "n_windows": 0,
                                "_metric": metric,
                                "_sam3_mask": cur_mask,
                            }
                            _seq += 1
                            if len(top_candidates) < MAX_CANDIDATES:
                                heapq.heappush(top_candidates, (metric, _seq, candidate))
                            elif metric > top_candidates[0][0]:
                                heapq.heapreplace(top_candidates, (metric, _seq, candidate))

            if n_windows > 0:
                print(f"    z{zoom}:{cname}: {n_windows}w, "
                      f"best={best_metric:.1f}", flush=True)

            # Early termination: skip remaining centers/zooms if we have
            # an excellent match (saves significant time on easy cases)
            if best_metric >= EARLY_STOP_METRIC:
                print(f"    Early stop: metric {best_metric:.1f} >= {EARLY_STOP_METRIC}")
                early_stopped = True

    if not top_candidates:
        return {
            "geojson": None, "affine_H": None, "tile_info": None,
            "match_info": {}, "n_windows": total_windows,
        }

    # Sort candidates best-first by raw metric
    ranked = sorted(top_candidates, key=lambda x: -x[0])

    # Specificity-aware re-ranking: if the top candidate is from a
    # broad-area center (rank ≥ 3, e.g. wikidata admin boundary, gpkg
    # Suburban Area) but another candidate in the top-5 is from a
    # street-level anchor (rank ≤ 1) with a metric ≥ 0.5x the top, prefer
    # the street-level one. This fixes cases where MINIMA locks onto a
    # wikidata borough/admin centroid with marginally more inliers than a
    # correctly-placed Nominatim street hit.
    if len(ranked) >= 2:
        top_metric, _, top_cand = ranked[0]
        top_center = (top_cand.get("match_info") or {}).get("center", "")
        top_rank = _center_specificity(top_center)
        if top_rank >= 3:
            for metric, _, cand in ranked[1:]:
                if metric < 0.5 * top_metric:
                    break
                c_center = (cand.get("match_info") or {}).get("center", "")
                c_rank = _center_specificity(c_center)
                if c_rank <= 1:
                    print(f"  Specificity re-rank: "
                          f"{top_center!r} (m={top_metric:.1f}, rank={top_rank}) "
                          f"→ {c_center!r} (m={metric:.1f}, rank={c_rank})")
                    # Move the chosen candidate to top
                    ranked = [(metric, 0, cand)] + [r for r in ranked
                                                     if r[2] is not cand]
                    break

    # Road name verification: if road names available, prefer candidates
    # where nearby OSM roads match the LLM-extracted road names
    best_result = None
    if road_names and len(road_names) >= 1:
        best_result = _verify_candidates_with_road_names(
            ranked, road_names)

    # Fallback: use best-scoring candidate
    if best_result is None:
        _, _, best_result = ranked[0]

    # Project mask now (deferred from inner loop for efficiency)
    cur_mask = best_result.pop("_sam3_mask", None)
    best_result.pop("_metric", None)
    if sam3_mask is not None and cur_mask is not None:
        best_result["geojson"] = mask_to_geojson_affine(
            cur_mask, best_result["affine_H"], best_result["tile_info"])

    best_result["n_windows"] = total_windows
    return best_result
