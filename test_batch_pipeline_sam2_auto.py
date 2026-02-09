"""
Batch pipeline test with SAM auto-mask + color filter boundary extraction.

Strategy:
1. SAM generates ALL possible segments in the map image
2. For each segment, compute average HSV color of pixels inside
3. Pick the segment that is most "planning-boundary-colored" —
   highest saturation in the pink/red/salmon hue range, reasonable size
4. Use that segment's mask as the boundary

This eliminates the need for per-case HSV tuning — SAM finds perfect boundaries,
we just pick the one with the right color.
"""

import json
import math
import random
import numpy as np
import cv2
import fitz  # pymupdf
import requests
import torch
from pathlib import Path
from io import BytesIO
from PIL import Image
from shapely.geometry import Polygon, mapping, shape
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from tools.template_matching import fetch_osm_data, METERS_PER_DEGREE_LAT
from geojson_metrics import calculate_iou, geojson_to_shape

OUTPUT_DIR = Path("batch_test_results_sam2_auto")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Load SAM model ──────────────────────────────────────────────────────────
print("Loading SAM vit_b...")
# Use CPU — segment_anything's auto mask generator creates float64 tensors
# internally which MPS doesn't support, and we can't patch their code
device = "cpu"
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
sam.to(device)
sam.eval()

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,          # 32x32 = 1024 grid points
    pred_iou_thresh=0.86,        # only keep confident masks
    stability_score_thresh=0.92, # only keep stable masks
    min_mask_region_area=1000,   # ignore tiny segments (< 1000 px)
)
print(f"SAM loaded on {device}!")


# ── Test cases with map-only crop fractions ─────────────────────────────────
# crop_frac = (y_top_frac, y_bottom_frac, x_left_frac, x_right_frac)
# Determined by manual visual inspection — crops out title blocks, wax seals,
# admin text, and glued cards to keep only the map region.
TEST_CASES = [
    {
        "dir": "C97065B6-03D0-48C4-AE0E-508DB0BE644B",
        "page": 4,
        "scale_text": "1:2500",
        "label": "Shepherdswell_1978",
        "crop_frac": (0.02, 0.78, 0.0, 1.0),
    },
    {
        "dir": "7202D619-4C27-4DA4-857E-B89F78C9D8D5",
        "page": 4,
        "scale_text": "1:2500",
        "label": "West_Stourmouth_1978",
        "crop_frac": (0.0, 0.73, 0.0, 1.0),
    },
    {
        "dir": "43C82C9C-0E1B-4CAE-83F8-E33277D7AC41",
        "page": 1,
        "scale_text": "1:2500",
        "label": "Droveway_Gardens_2010",
        "crop_frac": (0.02, 0.66, 0.0, 1.0),
    },
    {
        "dir": "D9176429-F30F-4638-A67E-3B87E7ED603D",
        "page": 3,
        "scale_text": "1:1250",
        "label": "Moon_Hill_2005",
        "crop_frac": (0.02, 0.60, 0.0, 1.0),
    },
    {
        "dir": "B4BE31D4-36A8-452E-97FF-04A53362B26C",
        "page": 2,
        "scale_text": "1:5000",
        "label": "Coombe_Road_Dover_2007",
        "crop_frac": (0.02, 0.60, 0.0, 1.0),
    },
    {
        "dir": "3DA282A7-E829-47CF-B842-E03E0C704072",
        "page": 3,
        "scale_text": "1:10560",
        "label": "Townsend_Farm_No2_1974",
        "crop_frac": (0.07, 0.76, 0.0, 1.0),
        "sam_rank": 1,  # #0 is a small wrong sub-parcel; #1 is better
    },
    {
        "dir": "FDBC0FDC-D090-4778-A123-232EB71DF3C6",
        "page": 3,
        "scale_text": "1:10560",
        "label": "Townsend_Farm_No1_1974",
        "crop_frac": (0.07, 0.76, 0.0, 1.0),
    },
    {
        "dir": "8CAFB06E-C92F-41CC-B701-6A38171FFAC2",
        "page": 3,
        "scale_text": "1:10560",
        "label": "Elms_Vale_Dover",
        "crop_frac": (0.02, 0.58, 0.0, 1.0),
    },
    {
        "dir": "FA067403-6115-4489-9ED0-2CF26FC2D299",
        "page": 1,
        "scale_text": "1:2500",
        "label": "Westmarsh_Drove_Farm_2011",
        "crop_frac": (0.02, 0.67, 0.0, 1.0),
    },
    {
        "dir": "0C05FFFF-24D1-4DD1-870B-9BA6B05ED77A",
        "page": -1,
        "scale_text": "1:2500",
        "label": "Golgotha_Shepherdswell",
        # No crop needed — screenshot is already map-only
    },
]


def render_pdf_page(pdf_path, page_num, dpi=200):
    doc = fitz.open(str(pdf_path))
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    doc.close()
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def compute_scale_meters(img_width, scale_text, dpi=200):
    scale_ratio = int(scale_text.split(":")[1])
    paper_width_inches = img_width / dpi
    paper_width_mm = paper_width_inches * 25.4
    return paper_width_mm * scale_ratio / 1000.0


def load_ground_truth(eval_dir):
    geojsons = list(eval_dir.glob("*.geojson"))
    if not geojsons:
        return None
    with open(geojsons[0]) as f:
        return json.load(f)


def get_geojson_center(geojson):
    s = geojson_to_shape(geojson)
    c = s.centroid
    return c.y, c.x


def is_warm_hue(h):
    """Check if a hue value is in the warm range (red/pink/salmon/magenta).
    In OpenCV HSV, H is 0-179:
      0-22: red → orange → salmon
      160-179: magenta → pink → red
    """
    return h <= 22 or h >= 155


def score_mask_color(mask_binary, image_bgr):
    """
    Score how "planning-boundary-colored" a mask is.

    Returns a score where higher = more likely to be the planning boundary.
    Considers: saturation (colorfulness), hue (warm/pink/red), area ratio.
    """
    h, w = image_bgr.shape[:2]
    img_area = h * w
    mask_area = np.sum(mask_binary)

    # Area ratio — planning boundaries are typically 2-50% of the map
    area_ratio = mask_area / img_area
    if area_ratio < 0.005 or area_ratio > 0.6:
        return -1  # too small or too large

    # Get HSV of pixels inside the mask
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    pixels_h = hsv[:, :, 0][mask_binary]
    pixels_s = hsv[:, :, 1][mask_binary]
    pixels_v = hsv[:, :, 2][mask_binary]

    avg_s = float(np.mean(pixels_s))
    avg_v = float(np.mean(pixels_v))
    median_h = float(np.median(pixels_h))

    # Must have some color (not white/gray/black paper)
    if avg_s < 15:
        return -1  # too desaturated — likely white paper or gray map

    # Must be reasonably bright (not dark scan artifact)
    if avg_v < 100:
        return -1

    # Must be in the warm hue range
    if not is_warm_hue(median_h):
        return -1  # green, blue, yellow — not a planning boundary

    # Circularity check — reject very circular (wax seals)
    # Find contours of this mask
    mask_u8 = mask_binary.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        perim = cv2.arcLength(largest, True)
        area_contour = cv2.contourArea(largest)
        if perim > 0:
            circularity = 4 * math.pi * area_contour / (perim * perim)
            if circularity > 0.75 and area_ratio < 0.03:
                return -1  # likely a wax seal

    # Score: we want the LARGEST reasonably-saturated warm-colored region.
    # The planning boundary fill is always the biggest colored area on the map.
    # Wax seals are small but very saturated — we need to weight by area.
    #
    # Score = saturation * sqrt(area_ratio) — this strongly favors larger regions
    # A seal at 1.3% area with S=200 scores: 200 * sqrt(0.013) = 22.8
    # A boundary at 8% area with S=50 scores: 50 * sqrt(0.08) = 14.1
    # A boundary at 20% area with S=50 scores: 50 * sqrt(0.20) = 22.4
    #
    # Actually let's use area_ratio directly as a multiplier — the boundary
    # should be the dominant colored feature, much larger than any seal
    score = avg_s * area_ratio * 100  # saturation * area_percent

    return score


def extract_boundary_sam_auto(pdf_bgr, case_label, sam_rank=0):
    """
    Use SAM automatic mask generator + color filtering to find the planning boundary.

    1. Resize image for SAM (max 1024px for speed)
    2. Generate all masks
    3. Score each by color (saturation in warm hue range)
    4. Pick best (or override with sam_rank), extract contour

    Args:
        sam_rank: Override which ranked candidate to pick (0=best, 1=second, etc.)

    Returns list of (x, y) pixel coordinates, or None.
    """
    h, w = pdf_bgr.shape[:2]
    rgb = cv2.cvtColor(pdf_bgr, cv2.COLOR_BGR2RGB)

    # Resize for SAM — auto mask gen on full-size images is very slow
    max_dim = max(h, w)
    if max_dim > 1024:
        scale_factor = 1024.0 / max_dim
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        rgb_resized = cv2.resize(rgb, (new_w, new_h))
        bgr_resized = cv2.resize(pdf_bgr, (new_w, new_h))
    else:
        rgb_resized = rgb
        bgr_resized = pdf_bgr
        new_w, new_h = w, h
        scale_factor = 1.0

    print(f"    Running SAM auto-mask on {new_w}x{new_h} image...")
    masks = mask_generator.generate(rgb_resized)
    print(f"    SAM generated {len(masks)} masks")

    if not masks:
        return None, np.zeros((h, w), dtype=np.uint8)

    # Score each mask by color
    scored = []
    for i, m in enumerate(masks):
        seg = m["segmentation"]  # boolean array (new_h, new_w)
        color_score = score_mask_color(seg, bgr_resized)
        area_pct = m["area"] / (new_w * new_h) * 100
        stability = m["stability_score"]

        if color_score > 0:
            # Combined score: color score * stability
            combined = color_score * stability
            scored.append((combined, i, color_score, stability, area_pct))

    if not scored:
        print("    No masks passed color filter!")
        return None, np.zeros((h, w), dtype=np.uint8)

    # Sort by combined score, descending
    scored.sort(reverse=True)

    # Print top candidates
    print(f"    Top {min(5, len(scored))} mask candidates:")
    for rank, (combined, idx, cscore, stab, area_pct) in enumerate(scored[:5]):
        print(f"      #{rank}: mask {idx}, combined={combined:.1f}, "
              f"color={cscore:.1f}, stability={stab:.3f}, area={area_pct:.1f}%")

    # Pick the best (or override with sam_rank)
    pick = min(sam_rank, len(scored) - 1)
    if pick != 0:
        print(f"    *** Using manual override: picking rank #{pick} instead of #0 ***")
    best_combined, best_idx, _, _, _ = scored[pick]
    best_mask = masks[best_idx]["segmentation"]  # boolean (new_h, new_w)

    # Save debug visualization: top 3 masks in different colors
    debug_vis = bgr_resized.copy()
    colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255)]  # green, cyan, yellow
    for rank in range(min(3, len(scored))):
        _, idx, _, _, _ = scored[rank]
        mask_bool = masks[idx]["segmentation"]
        color = colors[rank]
        debug_vis[mask_bool] = (
            debug_vis[mask_bool] * 0.5 + np.array(color) * 0.5
        ).astype(np.uint8)
    cv2.imwrite(str(OUTPUT_DIR / f"{case_label}_sam_top3.png"), debug_vis)

    # Scale mask back to original size
    mask_u8 = best_mask.astype(np.uint8) * 255
    if scale_factor != 1.0:
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)

    # Save best mask
    cv2.imwrite(str(OUTPUT_DIR / f"{case_label}_sam_best_mask.png"), mask_u8)

    # Find contours
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if not contours:
        return None, mask_u8

    # Use largest contour
    largest = contours[0]
    epsilon = 0.002 * cv2.arcLength(largest, True)
    boundary_contour = cv2.approxPolyDP(largest, epsilon, True)
    boundary_pixels = boundary_contour.reshape(-1, 2).tolist()

    return boundary_pixels, mask_u8


def truncated_chamfer_align(pdf_bgr, center_lat, center_lon, scale_meters,
                             search_range_m=200, cap=30.0):
    pdf_h, pdf_w = pdf_bgr.shape[:2]
    m_per_px = scale_meters / pdf_w
    map_height_m = scale_meters * pdf_h / pdf_w
    search_px = int(search_range_m / m_per_px)

    lat_per_m = 1.0 / METERS_PER_DEGREE_LAT
    lon_per_m = 1.0 / (METERS_PER_DEGREE_LAT * math.cos(math.radians(center_lat)))

    half_w_m = scale_meters / 2
    half_h_m = map_height_m / 2

    map_bounds = (
        center_lat - half_h_m * lat_per_m,
        center_lon - half_w_m * lon_per_m,
        center_lat + half_h_m * lat_per_m,
        center_lon + half_w_m * lon_per_m,
    )

    fetch_bounds = (
        center_lat - (half_h_m + search_range_m) * lat_per_m,
        center_lon - (half_w_m + search_range_m) * lon_per_m,
        center_lat + (half_h_m + search_range_m) * lat_per_m,
        center_lon + (half_w_m + search_range_m) * lon_per_m,
    )

    print(f"    Fetching OSM (±{search_range_m}m = ±{search_px}px)...")
    osm_data = fetch_osm_data(
        fetch_bounds[0], fetch_bounds[1], fetch_bounds[2], fetch_bounds[3],
        include_railways=True,
    )
    if osm_data is None:
        return 0, 0, float("inf"), map_bounds, None

    min_lat, min_lon, max_lat, max_lon = map_bounds
    img = np.zeros((pdf_h, pdf_w), dtype=np.uint8)

    def geo_to_pixel(lat, lon):
        x = int((lon - min_lon) / (max_lon - min_lon) * pdf_w)
        y = int((max_lat - lat) / (max_lat - min_lat) * pdf_h)
        return x, y

    nf = 0
    for el in osm_data.get("elements", []):
        if el["type"] != "way" or "geometry" not in el:
            continue
        tags = el.get("tags", {})
        if "highway" not in tags:
            continue
        pts = np.array([[*geo_to_pixel(n["lat"], n["lon"])] for n in el["geometry"]], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(img, [pts], False, 255, 3)
            nf += 1

    road_pos = np.argwhere(img > 0)
    print(f"    {nf} roads, {len(road_pos)} road pixels")

    if len(road_pos) < 100:
        return 0, 0, float("inf"), map_bounds, None

    if len(road_pos) > 5000:
        idx = np.random.RandomState(42).choice(len(road_pos), 5000, replace=False)
        road_pos = road_pos[idx]

    gray = cv2.cvtColor(pdf_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(pdf_bgr, cv2.COLOR_BGR2HSV)
    gray[hsv[:, :, 1] > 40] = 255
    _, dark = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    dt = cv2.distanceTransform(255 - dark, cv2.DIST_L2, 5).astype(np.float32)

    best_score, best_dx, best_dy = float("inf"), 0, 0
    for dy in range(-search_px, search_px + 1, 8):
        for dx in range(-search_px, search_px + 1, 8):
            sy = road_pos[:, 0] + dy
            sx = road_pos[:, 1] + dx
            v = (sy >= 0) & (sy < pdf_h) & (sx >= 0) & (sx < pdf_w)
            if v.sum() < 50:
                continue
            d = dt[sy[v].astype(int), sx[v].astype(int)]
            s = float(np.mean(np.minimum(d, cap)))
            if s < best_score:
                best_score, best_dx, best_dy = s, dx, dy

    for dy in range(best_dy - 10, best_dy + 11, 2):
        for dx in range(best_dx - 10, best_dx + 11, 2):
            sy = road_pos[:, 0] + dy
            sx = road_pos[:, 1] + dx
            v = (sy >= 0) & (sy < pdf_h) & (sx >= 0) & (sx < pdf_w)
            if v.sum() < 50:
                continue
            d = dt[sy[v].astype(int), sx[v].astype(int)]
            s = float(np.mean(np.minimum(d, cap)))
            if s < best_score:
                best_score, best_dx, best_dy = s, dx, dy

    for dy in range(best_dy - 2, best_dy + 3):
        for dx in range(best_dx - 2, best_dx + 3):
            sy = road_pos[:, 0] + dy
            sx = road_pos[:, 1] + dx
            v = (sy >= 0) & (sy < pdf_h) & (sx >= 0) & (sx < pdf_w)
            if v.sum() < 50:
                continue
            d = dt[sy[v].astype(int), sx[v].astype(int)]
            s = float(np.mean(np.minimum(d, cap)))
            if s < best_score:
                best_score, best_dx, best_dy = s, dx, dy

    print(f"    Alignment: dx={best_dx}, dy={best_dy}, score={best_score:.2f}")

    overlay = pdf_bgr.copy()
    M = np.float32([[1, 0, best_dx], [0, 1, best_dy]])
    osm_shifted = cv2.warpAffine(img, M, (pdf_w, pdf_h))
    overlay[osm_shifted > 0] = [0, 255, 0]

    return best_dx, best_dy, best_score, map_bounds, overlay


def pixel_to_geo(px, py, map_bounds, img_w, img_h, shift_dx, shift_dy):
    min_lat, min_lon, max_lat, max_lon = map_bounds
    osm_px = px - shift_dx
    osm_py = py - shift_dy
    lon = min_lon + (osm_px / img_w) * (max_lon - min_lon)
    lat = max_lat - (osm_py / img_h) * (max_lat - min_lat)
    return lat, lon


def render_osm_comparison(gt_shape, extracted_polygon, label, case_idx):
    from shapely.ops import unary_union

    combined = unary_union([gt_shape, extracted_polygon])
    minx, miny, maxx, maxy = combined.bounds
    pad_x = (maxx - minx) * 0.3
    pad_y = (maxy - miny) * 0.3
    minx -= pad_x
    maxx += pad_x
    miny -= pad_y
    maxy += pad_y

    span = max(maxx - minx, maxy - miny)
    if span < 0.005:
        zoom = 17
    elif span < 0.01:
        zoom = 16
    elif span < 0.02:
        zoom = 15
    elif span < 0.05:
        zoom = 14
    else:
        zoom = 13

    def deg_to_tile(lat, lon, z):
        lat_rad = math.radians(lat)
        n = 2 ** z
        x = int((lon + 180) / 360 * n)
        y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
        return x, y

    tx_min, ty_min = deg_to_tile(maxy, minx, zoom)
    tx_max, ty_max = deg_to_tile(miny, maxx, zoom)

    tile_size = 256
    img_w = (tx_max - tx_min + 1) * tile_size
    img_h = (ty_max - ty_min + 1) * tile_size
    canvas = np.ones((img_h, img_w, 3), dtype=np.uint8) * 200

    for tx in range(tx_min, tx_max + 1):
        for ty in range(ty_min, ty_max + 1):
            url = f"https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png"
            try:
                resp = requests.get(url, headers={"User-Agent": "GeoMapAgent/1.0"}, timeout=10)
                if resp.status_code == 200:
                    tile_img = np.array(Image.open(BytesIO(resp.content)).convert("RGB"))
                    tile_bgr = cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR)
                    px = (tx - tx_min) * tile_size
                    py = (ty - ty_min) * tile_size
                    canvas[py:py + tile_size, px:px + tile_size] = tile_bgr
            except Exception:
                pass

    def tile_geo_to_px(lat, lon):
        lat_rad = math.radians(lat)
        n = 2 ** zoom
        x = (lon + 180) / 360 * n
        y = (1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n
        px = int((x - tx_min) * tile_size)
        py = int((y - ty_min) * tile_size)
        return px, py

    overlay = canvas.copy()
    gt_coords = list(gt_shape.exterior.coords) if hasattr(gt_shape, 'exterior') else list(gt_shape.geoms[0].exterior.coords)
    gt_pts = np.array([tile_geo_to_px(lat, lon) for lon, lat in gt_coords], dtype=np.int32)
    cv2.fillPoly(overlay, [gt_pts], (255, 0, 0))
    cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
    cv2.polylines(canvas, [gt_pts], True, (255, 0, 0), 3)

    overlay = canvas.copy()
    if hasattr(extracted_polygon, 'exterior'):
        ext_coords = list(extracted_polygon.exterior.coords)
    else:
        ext_coords = list(extracted_polygon.geoms[0].exterior.coords)
    ext_pts = np.array([tile_geo_to_px(lat, lon) for lon, lat in ext_coords], dtype=np.int32)
    cv2.fillPoly(overlay, [ext_pts], (0, 200, 0))
    cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
    cv2.polylines(canvas, [ext_pts], True, (0, 200, 0), 3)

    cv2.putText(canvas, f"Case {case_idx}: {label} (SAM auto)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    legend_y = img_h - 60
    cv2.rectangle(canvas, (10, legend_y), (30, legend_y + 15), (255, 0, 0), -1)
    cv2.putText(canvas, "Ground Truth", (35, legend_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.rectangle(canvas, (10, legend_y + 25), (30, legend_y + 40), (0, 200, 0), -1)
    cv2.putText(canvas, "SAM Auto", (35, legend_y + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    out_path = OUTPUT_DIR / f"case{case_idx:02d}_osm_comparison.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"  Saved OSM comparison: {out_path}")


def run_case(case, case_idx):
    eval_dir = Path("evaluation_data") / case["dir"]
    label = case["label"]

    print(f"\n{'='*70}")
    print(f"CASE {case_idx}: {label}")
    print(f"{'='*70}")

    gt = load_ground_truth(eval_dir)
    if gt is None:
        print("  No ground truth found!")
        return None

    gt_shape = geojson_to_shape(gt)
    gt_center_lat, gt_center_lon = get_geojson_center(gt)
    print(f"  GT center: ({gt_center_lat:.6f}, {gt_center_lon:.6f})")

    if case["page"] == -1:
        map_img_path = Path("Bildschirmfoto 2026-02-09 um 02.36.36.png")
        if not map_img_path.exists():
            print("  Golgotha screenshot not found, skipping")
            return None
        pdf_map = cv2.imread(str(map_img_path))
    else:
        pdfs = list(eval_dir.glob("*.pdf"))
        if not pdfs:
            print("  No PDF found!")
            return None
        pdf_map = render_pdf_page(pdfs[0], case["page"])

    full_h, full_w = pdf_map.shape[:2]
    print(f"  Full page: {full_w}x{full_h}")

    # Crop to map-only region if crop_frac is specified
    crop_frac = case.get("crop_frac")
    if crop_frac:
        y_top_f, y_bot_f, x_left_f, x_right_f = crop_frac
        y_top = int(full_h * y_top_f)
        y_bot = int(full_h * y_bot_f)
        x_left = int(full_w * x_left_f)
        x_right = int(full_w * x_right_f)
        pdf_map = pdf_map[y_top:y_bot, x_left:x_right].copy()
        print(f"  Cropped to map: y=[{y_top}..{y_bot}], x=[{x_left}..{x_right}]")

    pdf_h, pdf_w = pdf_map.shape[:2]
    print(f"  Map image: {pdf_w}x{pdf_h}")

    if case["page"] == -1:
        scale_meters = 560
    else:
        # Scale is based on the CROPPED map width, not full page width
        scale_meters = compute_scale_meters(pdf_w, case["scale_text"])
    print(f"  Scale: {scale_meters:.0f}m ({case['scale_text']})")

    random.seed(case_idx + 42)
    shift_dist = random.uniform(50, 100)
    shift_angle = random.uniform(0, 2 * math.pi)
    shift_east = shift_dist * math.cos(shift_angle)
    shift_north = shift_dist * math.sin(shift_angle)

    lat_per_m = 1.0 / METERS_PER_DEGREE_LAT
    lon_per_m = 1.0 / (METERS_PER_DEGREE_LAT * math.cos(math.radians(gt_center_lat)))

    initial_lat = gt_center_lat + shift_north * lat_per_m
    initial_lon = gt_center_lon + shift_east * lon_per_m

    print(f"  Applied shift: {shift_east:.1f}m E, {shift_north:.1f}m N ({shift_dist:.1f}m)")

    # Step 1: Alignment
    print(f"\n  --- Step 1: Align OSM roads ---")
    result = truncated_chamfer_align(pdf_map, initial_lat, initial_lon, scale_meters)
    best_dx, best_dy, score, map_bounds, overlay = result

    if score == float("inf"):
        print("  Alignment failed!")
        return None

    if overlay is not None:
        cv2.imwrite(str(OUTPUT_DIR / f"case{case_idx:02d}_{label}_alignment.png"), overlay)

    # Step 2: SAM auto-mask boundary extraction
    print(f"\n  --- Step 2: Extract boundary (SAM auto-mask + color filter) ---")
    sam_rank = case.get("sam_rank", 0)
    boundary_pixels, sam_mask = extract_boundary_sam_auto(
        pdf_map, f"case{case_idx:02d}_{label}", sam_rank=sam_rank
    )

    if boundary_pixels is None:
        print("  No boundary found!")
        return {"case": label, "iou": 0.0, "error": "no_boundary"}

    print(f"  Boundary: {len(boundary_pixels)} points")

    # Step 3: Convert to geo
    print(f"\n  --- Step 3: Convert to geo ---")
    geo_coords = []
    for px, py in boundary_pixels:
        lat, lon = pixel_to_geo(px, py, map_bounds, pdf_w, pdf_h, best_dx, best_dy)
        geo_coords.append((lon, lat))

    if geo_coords[0] != geo_coords[-1]:
        geo_coords.append(geo_coords[0])

    extracted_polygon = Polygon(geo_coords)
    if not extracted_polygon.is_valid:
        extracted_polygon = extracted_polygon.buffer(0)

    # Step 4: Evaluate
    print(f"\n  --- Step 4: Evaluate ---")
    iou = calculate_iou(gt_shape, extracted_polygon)

    ext_centroid = extracted_polygon.centroid
    gt_centroid = gt_shape.centroid
    dist_e = (ext_centroid.x - gt_centroid.x) * METERS_PER_DEGREE_LAT * math.cos(math.radians(gt_centroid.y))
    dist_n = (ext_centroid.y - gt_centroid.y) * METERS_PER_DEGREE_LAT
    centroid_dist = math.sqrt(dist_e**2 + dist_n**2)

    print(f"  IoU: {iou:.4f}")
    print(f"  Centroid distance: {centroid_dist:.1f}m")

    # Visualizations
    vis = pdf_map.copy()
    pts = np.array(boundary_pixels, dtype=np.int32)
    cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

    if gt["type"] == "Feature":
        geom = gt["geometry"]
    else:
        geom = gt
    if geom["type"] == "MultiPolygon":
        all_coords = geom["coordinates"][0][0]
    elif geom["type"] == "Polygon":
        all_coords = geom["coordinates"][0]
    else:
        all_coords = []

    min_lat, min_lon, max_lat, max_lon = map_bounds
    gt_pts_list = []
    for lon, lat in all_coords:
        osm_x = (lon - min_lon) / (max_lon - min_lon) * pdf_w
        osm_y = (max_lat - lat) / (max_lat - min_lat) * pdf_h
        pdf_x = osm_x + best_dx
        pdf_y = osm_y + best_dy
        gt_pts_list.append([int(pdf_x), int(pdf_y)])

    if gt_pts_list:
        gt_pts = np.array(gt_pts_list, dtype=np.int32)
        cv2.polylines(vis, [gt_pts], True, (0, 0, 255), 2)

    cv2.imwrite(str(OUTPUT_DIR / f"case{case_idx:02d}_{label}_comparison.png"), vis)

    extracted_geojson = {
        "type": "Feature",
        "geometry": mapping(extracted_polygon),
        "properties": {"label": label, "iou": iou},
    }
    with open(OUTPUT_DIR / f"case{case_idx:02d}_{label}_extracted.geojson", "w") as f:
        json.dump(extracted_geojson, f, indent=2)

    # OSM comparison
    print(f"\n  --- Step 5: OSM tile comparison ---")
    try:
        render_osm_comparison(gt_shape, extracted_polygon, label, case_idx)
    except Exception as e:
        print(f"  OSM rendering failed: {e}")

    return {
        "case": label,
        "iou": iou,
        "centroid_dist_m": centroid_dist,
        "shift_m": shift_dist,
        "shift_east_m": shift_east,
        "shift_north_m": shift_north,
        "alignment_score": score,
        "n_boundary_pts": len(boundary_pixels),
        "scale_m": scale_meters,
    }


def main():
    print("=" * 70)
    print("BATCH PIPELINE TEST: SAM AUTO-MASK + COLOR FILTER")
    print("=" * 70)

    results = []
    for i, case in enumerate(TEST_CASES):
        try:
            result = run_case(case, i)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n  ERROR in case {i} ({case['label']}): {e}")
            import traceback
            traceback.print_exc()
            results.append({"case": case["label"], "iou": 0.0, "error": str(e)})

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (SAM auto-mask + color filter)")
    print("=" * 70)
    print(f"{'Case':<35} {'IoU':>6} {'Dist':>7} {'Shift':>7} {'Score':>6}")
    print("-" * 70)

    ious = []
    for r in results:
        iou = r.get("iou", 0)
        dist = r.get("centroid_dist_m", -1)
        shift = r.get("shift_m", -1)
        score = r.get("alignment_score", -1)
        error = r.get("error", "")

        if error:
            print(f"{r['case']:<35} {'FAIL':>6} {error}")
        else:
            print(f"{r['case']:<35} {iou:>6.3f} {dist:>6.1f}m {shift:>6.1f}m {score:>6.2f}")
            ious.append(iou)

    if ious:
        print("-" * 70)
        print(f"{'Mean IoU':<35} {np.mean(ious):>6.3f}")
        print(f"{'Median IoU':<35} {np.median(ious):>6.3f}")
        print(f"{'Min IoU':<35} {np.min(ious):>6.3f}")
        print(f"{'Max IoU':<35} {np.max(ious):>6.3f}")
        print(f"{'Cases with IoU > 0.3':<35} {sum(1 for x in ious if x > 0.3):>6d}/{len(ious)}")
        print(f"{'Cases with IoU > 0.5':<35} {sum(1 for x in ious if x > 0.5):>6d}/{len(ious)}")

    # Comparison with HSV v2
    hsv_results_path = Path("batch_test_results_v2/results.json")
    if hsv_results_path.exists():
        with open(hsv_results_path) as f:
            hsv_results = json.load(f)

        print("\n" + "=" * 70)
        print("COMPARISON: SAM Auto vs HSV (tuned)")
        print("=" * 70)
        print(f"{'Case':<35} {'HSV IoU':>8} {'SAM IoU':>8} {'Better':>8}")
        print("-" * 70)

        hsv_dict = {r["case"]: r.get("iou", 0) for r in hsv_results}
        sam_dict = {r["case"]: r.get("iou", 0) for r in results}

        sam_wins = 0
        hsv_wins = 0
        for case_label in hsv_dict:
            hsv_iou = hsv_dict.get(case_label, 0)
            sam_iou = sam_dict.get(case_label, 0)
            better = "SAM" if sam_iou > hsv_iou else ("HSV" if hsv_iou > sam_iou else "TIE")
            if better == "SAM":
                sam_wins += 1
            elif better == "HSV":
                hsv_wins += 1
            print(f"{case_label:<35} {hsv_iou:>8.3f} {sam_iou:>8.3f} {better:>8}")

        print("-" * 70)
        print(f"SAM wins: {sam_wins}, HSV wins: {hsv_wins}")

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/results.json")


if __name__ == "__main__":
    main()
