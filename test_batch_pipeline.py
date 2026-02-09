"""
Batch pipeline test: Run the alignment + boundary extraction pipeline on 10 test cases.

For each case:
1. Render the PDF map page to an image
2. Compute scale from image size and known map scale (from metadata)
3. Shift the GT center by 50-100m to simulate an imprecise initial guess
4. Run truncated chamfer alignment to find OSM-PDF road alignment
5. Extract pink/red boundary from the PDF map
6. Convert boundary pixels to geo coordinates using alignment
7. Compute IoU vs ground truth
8. Save visualizations
"""

import json
import math
import random
import numpy as np
import cv2
import fitz  # pymupdf
import requests
from pathlib import Path
from io import BytesIO
from PIL import Image
from shapely.geometry import Polygon, mapping, shape

from tools.template_matching import fetch_osm_data, METERS_PER_DEGREE_LAT
from geojson_metrics import calculate_iou, geojson_to_shape

OUTPUT_DIR = Path("batch_test_results_v2")
OUTPUT_DIR.mkdir(exist_ok=True)


def render_osm_comparison(gt_shape, extracted_polygon, label, case_idx):
    """Render GT and extracted polygons on OSM tile background."""
    from shapely.ops import unary_union

    combined = unary_union([gt_shape, extracted_polygon])
    minx, miny, maxx, maxy = combined.bounds

    # Add padding
    pad_x = (maxx - minx) * 0.3
    pad_y = (maxy - miny) * 0.3
    minx -= pad_x
    maxx += pad_x
    miny -= pad_y
    maxy += pad_y

    # Determine zoom level
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

    # Calculate tile range
    def deg_to_tile(lat, lon, z):
        lat_rad = math.radians(lat)
        n = 2 ** z
        x = int((lon + 180) / 360 * n)
        y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
        return x, y

    tx_min, ty_min = deg_to_tile(maxy, minx, zoom)
    tx_max, ty_max = deg_to_tile(miny, maxx, zoom)

    # Fetch tiles
    tile_size = 256
    img_w = (tx_max - tx_min + 1) * tile_size
    img_h = (ty_max - ty_min + 1) * tile_size
    canvas = np.ones((img_h, img_w, 3), dtype=np.uint8) * 200  # light gray

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

    # Geo to pixel conversion for tiles
    def tile_geo_to_px(lat, lon):
        lat_rad = math.radians(lat)
        n = 2 ** zoom
        x = (lon + 180) / 360 * n
        y = (1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n
        px = int((x - tx_min) * tile_size)
        py = int((y - ty_min) * tile_size)
        return px, py

    # Draw GT (blue, semi-transparent fill)
    overlay = canvas.copy()
    gt_coords = list(gt_shape.exterior.coords) if hasattr(gt_shape, 'exterior') else list(gt_shape.geoms[0].exterior.coords)
    gt_pts = np.array([tile_geo_to_px(lat, lon) for lon, lat in gt_coords], dtype=np.int32)
    cv2.fillPoly(overlay, [gt_pts], (255, 0, 0))  # blue
    cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
    cv2.polylines(canvas, [gt_pts], True, (255, 0, 0), 3)

    # Draw extracted (green, semi-transparent fill)
    overlay = canvas.copy()
    if hasattr(extracted_polygon, 'exterior'):
        ext_coords = list(extracted_polygon.exterior.coords)
    else:
        ext_coords = list(extracted_polygon.geoms[0].exterior.coords)
    ext_pts = np.array([tile_geo_to_px(lat, lon) for lon, lat in ext_coords], dtype=np.int32)
    cv2.fillPoly(overlay, [ext_pts], (0, 200, 0))  # green
    cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
    cv2.polylines(canvas, [ext_pts], True, (0, 200, 0), 3)

    # Legend
    cv2.putText(canvas, f"Case {case_idx}: {label}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    legend_y = img_h - 60
    cv2.rectangle(canvas, (10, legend_y), (30, legend_y + 15), (255, 0, 0), -1)
    cv2.putText(canvas, "Ground Truth", (35, legend_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.rectangle(canvas, (10, legend_y + 25), (30, legend_y + 40), (0, 200, 0), -1)
    cv2.putText(canvas, "Extracted", (35, legend_y + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    out_path = OUTPUT_DIR / f"case{case_idx:02d}_osm_comparison.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"  Saved OSM comparison: {out_path}")

# ── Test cases ──────────────────────────────────────────────────────────────
# Each: (dir_name, map_page, scale_ratio, label)
# scale_ratio: the "1:X" scale → we compute real-world width from image size
TEST_CASES = [
    {
        "dir": "C97065B6-03D0-48C4-AE0E-508DB0BE644B",
        "page": 4,
        "scale_text": "1:2500",
        "label": "Shepherdswell_1978",
        # Salmon/orange fill, H median=14, S median=49, V median=251
        # Need S>30 to avoid yellowish paper, V>180 to keep bright salmon only
        "hsv_ranges": [([5, 30, 180], [20, 255, 255])],
    },
    {
        "dir": "7202D619-4C27-4DA4-857E-B89F78C9D8D5",
        "page": 4,
        "scale_text": "1:2500",
        "label": "West_Stourmouth_1978",
        # Salmon/orange fill, H median=14, S median=39, V median=250
        "hsv_ranges": [([5, 30, 180], [20, 255, 255])],
    },
    {
        "dir": "43C82C9C-0E1B-4CAE-83F8-E33277D7AC41",
        "page": 1,
        "scale_text": "1:2500",
        "label": "Droveway_Gardens_2010",
        # Magenta/pink, H median=166, S median=80
        "hsv_ranges": [([158, 40, 100], [179, 255, 255])],
    },
    {
        "dir": "D9176429-F30F-4638-A67E-3B87E7ED603D",
        "page": 3,
        "scale_text": "1:1250",
        "label": "Moon_Hill_2005",
        # Magenta/pink, H median=174, S median=102
        "hsv_ranges": [([160, 50, 100], [179, 255, 255])],
    },
    {
        "dir": "B4BE31D4-36A8-452E-97FF-04A53362B26C",
        "page": 2,
        "scale_text": "1:5000",
        "label": "Coombe_Road_Dover_2007",
        # Magenta/pink, H median=171, S median=80
        "hsv_ranges": [([160, 40, 100], [179, 255, 255])],
    },
    {
        "dir": "3DA282A7-E829-47CF-B842-E03E0C704072",
        "page": 3,
        "scale_text": "1:10560",
        "label": "Townsend_Farm_No2_1974",
        # Orange-red fill, H median=17, S median=70
        # Tighter range: H=8-22, S>50 to avoid brown scan edges, V>150
        "hsv_ranges": [([8, 50, 150], [22, 255, 255])],
    },
    {
        "dir": "FDBC0FDC-D090-4778-A123-232EB71DF3C6",
        "page": 3,
        "scale_text": "1:10560",
        "label": "Townsend_Farm_No1_1974",
        # Orange-red fill, H median=18, S median=67
        "hsv_ranges": [([8, 50, 150], [22, 255, 255])],
    },
    {
        "dir": "8CAFB06E-C92F-41CC-B701-6A38171FFAC2",
        "page": 3,
        "scale_text": "1:10560",
        "label": "Elms_Vale_Dover",
        # Salmon fill, H median=16, S median=37 — very subtle
        # Need careful range to avoid the glued seal page at bottom
        "hsv_ranges": [([5, 28, 200], [22, 255, 255])],
    },
    {
        "dir": "FA067403-6115-4489-9ED0-2CF26FC2D299",
        "page": 1,
        "scale_text": "1:2500",
        "label": "Westmarsh_Drove_Farm_2011",
        # Magenta/pink, H median=169, S median=69
        "hsv_ranges": [([158, 35, 100], [179, 255, 255])],
    },
    {
        "dir": "0C05FFFF-24D1-4DD1-870B-9BA6B05ED77A",
        "page": -1,  # special: use screenshot
        "scale_text": "1:2500",
        "label": "Golgotha_Shepherdswell",
        # Pink/magenta from web screenshot — original range that worked well
        "hsv_ranges": [([0, 40, 100], [15, 255, 255]), ([160, 40, 100], [179, 255, 255])],
    },
]


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
    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def compute_scale_meters(img_width, scale_text, dpi=200):
    """
    Compute the real-world width (meters) that the image covers.

    For a map at 1:X printed on paper and scanned at 'dpi':
    - Paper width = img_width / dpi (inches)
    - Paper width_mm = paper_width * 25.4
    - Real-world width = paper_width_mm * X / 1000 (meters)
    """
    scale_ratio = int(scale_text.split(":")[1])
    paper_width_inches = img_width / dpi
    paper_width_mm = paper_width_inches * 25.4
    real_world_m = paper_width_mm * scale_ratio / 1000.0
    return real_world_m


def load_ground_truth(eval_dir):
    """Load the GeoJSON ground truth from an evaluation directory."""
    geojsons = list(eval_dir.glob("*.geojson"))
    if not geojsons:
        return None
    with open(geojsons[0]) as f:
        return json.load(f)


def get_geojson_center(geojson):
    """Get the centroid of a GeoJSON Feature."""
    s = geojson_to_shape(geojson)
    c = s.centroid
    return c.y, c.x


def truncated_chamfer_align(pdf_bgr, center_lat, center_lon, scale_meters,
                             search_range_m=200, cap=30.0):
    """
    Find the pixel shift that aligns OSM roads with PDF map roads.
    Returns (best_dx, best_dy, score, map_bounds).
    """
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
        return 0, 0, float("inf"), map_bounds

    # Render OSM roads
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
        return 0, 0, float("inf"), map_bounds

    if len(road_pos) > 5000:
        idx = np.random.RandomState(42).choice(len(road_pos), 5000, replace=False)
        road_pos = road_pos[idx]

    # PDF distance transform
    gray = cv2.cvtColor(pdf_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(pdf_bgr, cv2.COLOR_BGR2HSV)
    gray[hsv[:, :, 1] > 40] = 255  # mask out colored areas
    _, dark = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    dt = cv2.distanceTransform(255 - dark, cv2.DIST_L2, 5).astype(np.float32)

    # Coarse search
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

    # Fine search
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

    # Pixel-level
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

    # Visualize alignment
    overlay = pdf_bgr.copy()
    M = np.float32([[1, 0, best_dx], [0, 1, best_dy]])
    osm_shifted = cv2.warpAffine(img, M, (pdf_w, pdf_h))
    overlay[osm_shifted > 0] = [0, 255, 0]

    return best_dx, best_dy, best_score, map_bounds, overlay, osm_data


def extract_boundary(pdf_bgr, hsv_ranges):
    """
    Extract the colored boundary from a PDF map image using case-specific HSV ranges.

    Args:
        pdf_bgr: BGR image
        hsv_ranges: list of (lower, upper) HSV tuples for the boundary color

    Returns list of (x, y) pixel coordinates of the boundary polygon, or None.
    """
    h, w = pdf_bgr.shape[:2]
    hsv = cv2.cvtColor(pdf_bgr, cv2.COLOR_BGR2HSV)

    # Build mask from all provided HSV ranges
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for lo, hi in hsv_ranges:
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        combined_mask = combined_mask | mask

    # Morphological cleanup — close small gaps
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if not contours:
        return None, combined_mask

    # Filter: keep contours that are at least 0.5% of image area
    img_area = h * w
    valid_contours = [c for c in contours if cv2.contourArea(c) > img_area * 0.005]

    if not valid_contours:
        return None, combined_mask

    # Filter out circular contours (wax seals) — seals are roughly circular
    # with high circularity (4π·area / perimeter²) close to 1.0
    filtered = []
    for c in valid_contours:
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circularity = 4 * math.pi * area / (perim * perim)
        # Wax seals: circularity > 0.7, and area < 5% of image
        if circularity > 0.7 and area < img_area * 0.05:
            print(f"    Filtered out circular contour (circularity={circularity:.2f}, "
                  f"area={area:.0f} = {area/img_area*100:.1f}% of image) — likely wax seal")
            continue
        filtered.append(c)

    if not filtered:
        # All contours were seals — fall back to largest original
        filtered = [valid_contours[0]]
        print("    WARNING: All contours looked like seals, using largest anyway")

    # Use the largest non-seal contour
    largest = filtered[0]
    epsilon = 0.002 * cv2.arcLength(largest, True)
    boundary_contour = cv2.approxPolyDP(largest, epsilon, True)
    boundary_pixels = boundary_contour.reshape(-1, 2).tolist()

    return boundary_pixels, combined_mask


def pixel_to_geo(px, py, map_bounds, img_w, img_h, shift_dx, shift_dy):
    """Convert PDF pixel to geographic coordinates accounting for alignment shift."""
    min_lat, min_lon, max_lat, max_lon = map_bounds
    osm_px = px - shift_dx
    osm_py = py - shift_dy
    lon = min_lon + (osm_px / img_w) * (max_lon - min_lon)
    lat = max_lat - (osm_py / img_h) * (max_lat - min_lat)
    return lat, lon


def run_case(case, case_idx):
    """Run the full pipeline on a single test case."""
    eval_dir = Path("evaluation_data") / case["dir"]
    label = case["label"]

    print(f"\n{'='*70}")
    print(f"CASE {case_idx}: {label}")
    print(f"{'='*70}")

    # Load ground truth
    gt = load_ground_truth(eval_dir)
    if gt is None:
        print("  No ground truth found!")
        return None

    gt_shape = geojson_to_shape(gt)
    gt_center_lat, gt_center_lon = get_geojson_center(gt)
    print(f"  GT center: ({gt_center_lat:.6f}, {gt_center_lon:.6f})")

    # Load/render map image
    if case["page"] == -1:
        # Special case: Golgotha uses a screenshot
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

    # Compute scale
    if case["page"] == -1:
        # Golgotha: cropped screenshot, use empirical scale
        scale_meters = 560  # Best found from previous calibration
    else:
        scale_meters = compute_scale_meters(pdf_w, case["scale_text"])
    print(f"  Scale: {scale_meters:.0f}m ({case['scale_text']})")

    # Apply random shift (50-100m)
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

    # Step 1: Truncated chamfer alignment
    print(f"\n  --- Step 1: Align OSM roads ---")
    result = truncated_chamfer_align(pdf_map, initial_lat, initial_lon, scale_meters)
    if len(result) == 4:
        best_dx, best_dy, score, map_bounds = result
        overlay = None
        osm_data = None
    else:
        best_dx, best_dy, score, map_bounds, overlay, osm_data = result

    if score == float("inf"):
        print("  Alignment failed!")
        return None

    # Save alignment overlay
    if overlay is not None:
        cv2.imwrite(str(OUTPUT_DIR / f"case{case_idx:02d}_{label}_alignment.png"), overlay)

    # Step 2: Extract boundary with per-case HSV
    print(f"\n  --- Step 2: Extract boundary ---")
    hsv_ranges = case.get("hsv_ranges", [([0, 40, 100], [15, 255, 255]),
                                          ([160, 40, 100], [179, 255, 255])])
    print(f"    HSV ranges: {hsv_ranges}")
    boundary_pixels, color_mask = extract_boundary(pdf_map, hsv_ranges)
    cv2.imwrite(str(OUTPUT_DIR / f"case{case_idx:02d}_{label}_color_mask.png"), color_mask)

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

    print(f"  Extracted polygon: {len(geo_coords)} vertices")

    # Step 4: Compute IoU
    print(f"\n  --- Step 4: Evaluate ---")
    iou = calculate_iou(gt_shape, extracted_polygon)

    # Centroid distance
    ext_centroid = extracted_polygon.centroid
    gt_centroid = gt_shape.centroid
    dist_e = (ext_centroid.x - gt_centroid.x) * METERS_PER_DEGREE_LAT * math.cos(math.radians(gt_centroid.y))
    dist_n = (ext_centroid.y - gt_centroid.y) * METERS_PER_DEGREE_LAT
    centroid_dist = math.sqrt(dist_e**2 + dist_n**2)

    print(f"  IoU: {iou:.4f}")
    print(f"  Centroid distance: {centroid_dist:.1f}m")

    # Step 5: Visualizations
    # a) Boundary on PDF map (green=extracted, red=GT)
    vis = pdf_map.copy()
    pts = np.array(boundary_pixels, dtype=np.int32)
    cv2.polylines(vis, [pts], True, (0, 255, 0), 2)

    # Draw GT on PDF
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
    print(f"  Saved comparison (green=extracted, red=GT)")

    # Save extracted GeoJSON
    extracted_geojson = {
        "type": "Feature",
        "geometry": mapping(extracted_polygon),
        "properties": {"label": label, "iou": iou},
    }
    with open(OUTPUT_DIR / f"case{case_idx:02d}_{label}_extracted.geojson", "w") as f:
        json.dump(extracted_geojson, f, indent=2)

    # Step 6: OSM tile comparison
    print(f"\n  --- Step 6: OSM tile comparison ---")
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
    print("BATCH PIPELINE TEST: 10 CASES")
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
    print("SUMMARY")
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

    # Save results
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/results.json")


if __name__ == "__main__":
    main()
