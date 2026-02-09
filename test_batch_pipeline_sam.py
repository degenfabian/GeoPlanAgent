"""
Batch pipeline test with SAM boundary extraction.

Same as test_batch_pipeline.py but uses Segment Anything Model (SAM)
for boundary extraction instead of HSV color filtering.

Strategy: Use the pink mask centroid as a point prompt to SAM,
asking it to segment the planning boundary area.
"""

import json
import math
import random
import numpy as np
import cv2
import fitz  # pymupdf
import torch
from pathlib import Path
from shapely.geometry import Polygon, mapping, shape
from transformers import SamModel, SamProcessor
from PIL import Image

from tools.template_matching import fetch_osm_data, METERS_PER_DEGREE_LAT
from geojson_metrics import calculate_iou, geojson_to_shape

OUTPUT_DIR = Path("batch_test_results_sam")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Test cases (same as before) ─────────────────────────────────────────────
TEST_CASES = [
    {
        "dir": "C97065B6-03D0-48C4-AE0E-508DB0BE644B",
        "page": 4,
        "scale_text": "1:2500",
        "label": "Shepherdswell_1978",
    },
    {
        "dir": "7202D619-4C27-4DA4-857E-B89F78C9D8D5",
        "page": 4,
        "scale_text": "1:2500",
        "label": "West_Stourmouth_1978",
    },
    {
        "dir": "43C82C9C-0E1B-4CAE-83F8-E33277D7AC41",
        "page": 1,
        "scale_text": "1:2500",
        "label": "Droveway_Gardens_2010",
    },
    {
        "dir": "D9176429-F30F-4638-A67E-3B87E7ED603D",
        "page": 3,
        "scale_text": "1:1250",
        "label": "Moon_Hill_2005",
    },
    {
        "dir": "B4BE31D4-36A8-452E-97FF-04A53362B26C",
        "page": 2,
        "scale_text": "1:5000",
        "label": "Coombe_Road_Dover_2007",
    },
    {
        "dir": "3DA282A7-E829-47CF-B842-E03E0C704072",
        "page": 3,
        "scale_text": "1:10560",
        "label": "Townsend_Farm_No2_1974",
    },
    {
        "dir": "FDBC0FDC-D090-4778-A123-232EB71DF3C6",
        "page": 3,
        "scale_text": "1:10560",
        "label": "Townsend_Farm_No1_1974",
    },
    {
        "dir": "8CAFB06E-C92F-41CC-B701-6A38171FFAC2",
        "page": 3,
        "scale_text": "1:10560",
        "label": "Elms_Vale_Dover",
    },
    {
        "dir": "FA067403-6115-4489-9ED0-2CF26FC2D299",
        "page": 1,
        "scale_text": "1:2500",
        "label": "Westmarsh_Drove_Farm_2011",
    },
    {
        "dir": "0C05FFFF-24D1-4DD1-870B-9BA6B05ED77A",
        "page": -1,
        "scale_text": "1:2500",
        "label": "Golgotha_Shepherdswell",
    },
]


# ── Load SAM model ──────────────────────────────────────────────────────────
print("Loading SAM model...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"  Device: {device}")
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
print("  SAM loaded!")


def render_pdf_page(pdf_path, page_num, dpi=200):
    """Render a PDF page to a BGR numpy array."""
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
    real_world_m = paper_width_mm * scale_ratio / 1000.0
    return real_world_m


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


def truncated_chamfer_align(pdf_bgr, center_lat, center_lon, scale_meters,
                             search_range_m=200, cap=30.0):
    """Same alignment as before."""
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
        print("    OSM fetch failed!")
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
        print("    Not enough road pixels!")
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


def get_pink_mask(pdf_bgr):
    """Extract the pink/red mask using HSV filtering."""
    hsv = cv2.cvtColor(pdf_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 40, 100]), np.array([15, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 40, 100]), np.array([179, 255, 255]))
    pink_mask = mask1 | mask2
    return pink_mask


def get_sam_prompts(pink_mask, n_positive=8, n_negative=4):
    """
    Generate multi-point prompts + bounding box from the pink mask.

    Returns:
        positive_points: list of (x, y) pink pixel locations
        negative_points: list of (x, y) non-pink locations
        bbox: (x1, y1, x2, y2) bounding box of pink region
        centroid: (cx, cy) centroid of pink region
    """
    ys, xs = np.where(pink_mask > 0)
    if len(xs) == 0:
        return None, None, None, None

    # Centroid
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))

    # Bounding box of pink region with small padding
    h, w = pink_mask.shape[:2]
    pad = 10
    x1 = max(0, int(np.min(xs)) - pad)
    y1 = max(0, int(np.min(ys)) - pad)
    x2 = min(w - 1, int(np.max(xs)) + pad)
    y2 = min(h - 1, int(np.max(ys)) + pad)
    bbox = (x1, y1, x2, y2)

    # Sample positive points from pink pixels (spread across the region)
    pink_indices = np.column_stack((xs, ys))  # (N, 2) — (x, y) format
    if len(pink_indices) > n_positive:
        # Use stratified sampling: pick points spread across the region
        step = len(pink_indices) // n_positive
        sampled_idx = np.arange(0, len(pink_indices), step)[:n_positive]
        # Shuffle the pink pixels first so we get spatial diversity
        rng = np.random.RandomState(42)
        rng.shuffle(pink_indices)
        positive_points = pink_indices[sampled_idx].tolist()
    else:
        positive_points = pink_indices.tolist()

    # Sample negative points from non-pink areas
    # Pick points that are clearly NOT pink: corners and areas far from pink
    negative_points = []
    # Try corners first
    corner_candidates = [
        (pad, pad),  # top-left
        (w - pad, pad),  # top-right
        (pad, h - pad),  # bottom-left
        (w - pad, h - pad),  # bottom-right
    ]
    for cx_neg, cy_neg in corner_candidates:
        if pink_mask[cy_neg, cx_neg] == 0:
            negative_points.append([cx_neg, cy_neg])
            if len(negative_points) >= n_negative:
                break

    # If we need more, sample from non-pink pixels
    if len(negative_points) < n_negative:
        non_pink_ys, non_pink_xs = np.where(pink_mask == 0)
        if len(non_pink_xs) > 0:
            non_pink_indices = np.column_stack((non_pink_xs, non_pink_ys))
            rng = np.random.RandomState(123)
            rng.shuffle(non_pink_indices)
            remaining = n_negative - len(negative_points)
            step = max(1, len(non_pink_indices) // remaining)
            for i in range(0, min(len(non_pink_indices), remaining * step), step):
                negative_points.append(non_pink_indices[i].tolist())
                if len(negative_points) >= n_negative:
                    break

    return positive_points, negative_points, bbox, (cx, cy)


def extract_boundary_with_sam(pdf_bgr, pink_mask, case_label):
    """
    Use SAM to segment the planning boundary area with multi-point + bbox prompts,
    then intersect with pink HSV mask for precision.

    Strategy:
    1. Give SAM multiple positive points (pink pixels) + negative points (non-pink)
       + bounding box of the pink region
    2. SAM produces a refined object mask with clean edges
    3. Intersect SAM mask with the pink HSV mask — this tells SAM "where" while
       HSV tells us "what color"

    Returns list of (x, y) pixel coordinates of the boundary polygon, or None.
    """
    h, w = pdf_bgr.shape[:2]

    # Get multi-point prompts from pink mask
    positive_pts, negative_pts, bbox, centroid = get_sam_prompts(pink_mask)
    if positive_pts is None:
        print("    No pink pixels found for SAM prompts")
        return None, np.zeros((h, w), dtype=np.uint8)

    print(f"    SAM prompts: {len(positive_pts)} positive, {len(negative_pts)} negative points")
    print(f"    Bounding box: {bbox}")

    # SAM expects RGB PIL Image
    rgb = cv2.cvtColor(pdf_bgr, cv2.COLOR_BGR2RGB)

    # Resize for SAM if too large (SAM works at 1024x1024 internally)
    max_dim = max(h, w)
    if max_dim > 2048:
        scale_factor = 2048.0 / max_dim
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        rgb_resized = cv2.resize(rgb, (new_w, new_h))
        # Scale all prompts
        pos_scaled = [[int(x * scale_factor), int(y * scale_factor)] for x, y in positive_pts]
        neg_scaled = [[int(x * scale_factor), int(y * scale_factor)] for x, y in negative_pts]
        bbox_scaled = [int(bbox[0] * scale_factor), int(bbox[1] * scale_factor),
                       int(bbox[2] * scale_factor), int(bbox[3] * scale_factor)]
    else:
        rgb_resized = rgb
        new_w, new_h = w, h
        pos_scaled = positive_pts
        neg_scaled = negative_pts
        bbox_scaled = list(bbox)
        scale_factor = 1.0

    pil_image = Image.fromarray(rgb_resized)

    # Combine positive and negative points with labels
    all_points = pos_scaled + neg_scaled
    all_labels = [1] * len(pos_scaled) + [0] * len(neg_scaled)

    input_points = [all_points]  # batch of 1
    input_labels = [all_labels]  # batch of 1
    input_boxes = [[bbox_scaled]]  # batch of 1, 1 box

    print(f"    Total points: {len(all_points)} (labels: {all_labels})")

    inputs = sam_processor(
        pil_image,
        input_points=input_points,
        input_labels=input_labels,
        input_boxes=input_boxes,
        return_tensors="pt",
    )
    # Cast all tensors to float32 before moving to MPS (MPS doesn't support float64)
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.float64:
            inputs[k] = v.float()
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    # Get masks
    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )

    mask_tensor = masks[0]  # first (only) image
    scores = outputs.iou_scores.cpu().numpy()

    print(f"    SAM mask tensor shape: {mask_tensor.shape}, scores shape: {scores.shape}")
    print(f"    SAM scores: {scores}")

    # Flatten scores to find best mask across all dimensions
    scores_flat = scores.flatten()
    best_flat_idx = np.argmax(scores_flat)
    best_score_val = scores_flat[best_flat_idx]

    # Index into the mask tensor — squeeze out extra dims and pick best
    mask_squeezed = mask_tensor.squeeze()
    if mask_squeezed.ndim == 3:
        best_mask_idx = best_flat_idx % mask_squeezed.shape[0]
        sam_mask = mask_squeezed[best_mask_idx].numpy().astype(np.uint8) * 255
    elif mask_squeezed.ndim == 2:
        sam_mask = mask_squeezed.numpy().astype(np.uint8) * 255
    else:
        print(f"    Unexpected mask shape: {mask_squeezed.shape}")
        return None, np.zeros((h, w), dtype=np.uint8)

    print(f"    Best SAM mask score={best_score_val:.3f}, shape={sam_mask.shape}")

    # Save raw SAM mask for debugging
    cv2.imwrite(str(OUTPUT_DIR / f"{case_label}_sam_mask_raw.png"), sam_mask)

    # If we resized, scale SAM mask back to original size
    if scale_factor != 1.0:
        sam_mask = cv2.resize(sam_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # === KEY IMPROVEMENT: Intersect SAM mask with pink HSV mask ===
    # SAM gives us clean edges, HSV tells us the color.
    # The intersection keeps only the pink-colored region that SAM identified as an object.
    #
    # First, dilate the pink mask slightly to account for SAM's edge precision
    # being slightly different from the HSV boundary
    pink_dilated = cv2.dilate(pink_mask, np.ones((7, 7), np.uint8), iterations=2)
    combined_mask = cv2.bitwise_and(sam_mask, pink_dilated)

    # Morphological cleanup on the combined mask
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Save combined mask for debugging
    cv2.imwrite(str(OUTPUT_DIR / f"{case_label}_sam_mask_combined.png"), combined_mask)

    # Check if combined mask has enough content; if not, fall back to just pink mask
    combined_area = np.sum(combined_mask > 0)
    pink_area = np.sum(pink_mask > 0)
    sam_area = np.sum(sam_mask > 0)
    print(f"    Areas — SAM: {sam_area}, Pink: {pink_area}, Combined: {combined_area}")

    if combined_area < pink_area * 0.1:
        # SAM and pink barely overlap — SAM probably segmented wrong thing
        # Fall back to cleaned-up pink mask
        print("    WARNING: SAM & pink barely overlap, falling back to HSV mask")
        final_mask = pink_mask.copy()
        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    else:
        final_mask = combined_mask

    # Save final mask
    cv2.imwrite(str(OUTPUT_DIR / f"{case_label}_sam_mask.png"), final_mask)

    # Find contours on the final mask
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if not contours:
        return None, final_mask

    # Use the largest contour
    img_area = h * w
    valid_contours = [c for c in contours if cv2.contourArea(c) > img_area * 0.005]

    if not valid_contours:
        valid_contours = [contours[0]]

    largest = valid_contours[0]
    epsilon = 0.002 * cv2.arcLength(largest, True)
    boundary_contour = cv2.approxPolyDP(largest, epsilon, True)
    boundary_pixels = boundary_contour.reshape(-1, 2).tolist()

    return boundary_pixels, final_mask


def pixel_to_geo(px, py, map_bounds, img_w, img_h, shift_dx, shift_dy):
    min_lat, min_lon, max_lat, max_lon = map_bounds
    osm_px = px - shift_dx
    osm_py = py - shift_dy
    lon = min_lon + (osm_px / img_w) * (max_lon - min_lon)
    lat = max_lat - (osm_py / img_h) * (max_lat - min_lat)
    return lat, lon


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

    pdf_h, pdf_w = pdf_map.shape[:2]
    print(f"  Map image: {pdf_w}x{pdf_h}")

    if case["page"] == -1:
        scale_meters = 560
    else:
        scale_meters = compute_scale_meters(pdf_w, case["scale_text"])
    print(f"  Scale: {scale_meters:.0f}m ({case['scale_text']})")

    # Apply random shift (same seed as HSV version for fair comparison)
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

    # Step 1: Truncated chamfer alignment (same as before)
    print(f"\n  --- Step 1: Align OSM roads ---")
    result = truncated_chamfer_align(pdf_map, initial_lat, initial_lon, scale_meters)
    best_dx, best_dy, score, map_bounds, overlay = result

    if score == float("inf"):
        print("  Alignment failed!")
        return None

    if overlay is not None:
        cv2.imwrite(str(OUTPUT_DIR / f"case{case_idx:02d}_{label}_alignment.png"), overlay)

    # Step 2: SAM boundary extraction
    print(f"\n  --- Step 2: Extract boundary with SAM ---")

    # Get pink centroid for point prompt
    point_prompt, pink_mask = get_pink_centroid(pdf_map)
    if point_prompt is None:
        print("  No pink area found for point prompt!")
        return {"case": label, "iou": 0.0, "error": "no_pink_centroid"}

    print(f"    Point prompt (pink centroid): ({point_prompt[0]}, {point_prompt[1]})")
    cv2.imwrite(str(OUTPUT_DIR / f"case{case_idx:02d}_{label}_pink_mask.png"), pink_mask)

    # Run SAM
    boundary_pixels, sam_mask = extract_boundary_with_sam(pdf_map, point_prompt, f"case{case_idx:02d}_{label}")

    if boundary_pixels is None:
        print("  SAM returned no boundary!")
        return {"case": label, "iou": 0.0, "error": "sam_no_boundary"}

    print(f"  SAM boundary: {len(boundary_pixels)} points")

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

    print(f"  Extracted polygon: {len(geo_coords)} vertices")

    # Step 4: Compute IoU
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
    # Draw SAM boundary in green
    pts = np.array(boundary_pixels, dtype=np.int32)
    cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
    # Draw point prompt
    cv2.circle(vis, point_prompt, 8, (255, 0, 255), -1)
    cv2.circle(vis, point_prompt, 10, (255, 0, 255), 2)

    # Draw GT boundary in red
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

    # Save extracted GeoJSON
    extracted_geojson = {
        "type": "Feature",
        "geometry": mapping(extracted_polygon),
        "properties": {"label": label, "iou": iou},
    }
    with open(OUTPUT_DIR / f"case{case_idx:02d}_{label}_extracted.geojson", "w") as f:
        json.dump(extracted_geojson, f, indent=2)

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
    print("BATCH PIPELINE TEST WITH SAM: 10 CASES")
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
    print("SUMMARY (SAM boundary extraction)")
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

    # Also print comparison with HSV results
    hsv_results_path = Path("batch_test_results/results.json")
    if hsv_results_path.exists():
        with open(hsv_results_path) as f:
            hsv_results = json.load(f)

        print("\n" + "=" * 70)
        print("COMPARISON: SAM vs HSV")
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
