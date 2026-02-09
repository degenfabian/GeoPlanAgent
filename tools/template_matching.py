"""
Template Matching Utilities for Georeferencing Refinement

Shared functions for:
- Fetching OSM data via Overpass API
- Rendering OSM features as black lines on white background
- Edge-detecting PDF map images
- Multi-scale template matching (baseline and improved)
- Coordinate conversions

This module eliminates code duplication across openrouter_client.py,
test_template_matching.py, test_refinement.py, and visualize_pipeline_steps.py.
"""

import math
import time
import requests
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional, List


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METERS_PER_DEGREE_LAT = 111_000.0


# ---------------------------------------------------------------------------
# OSM Data Fetching
# ---------------------------------------------------------------------------

def fetch_osm_data(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    include_railways: bool = False,
    include_waterways: bool = False,
    retries: int = 3,
) -> Optional[dict]:
    """
    Fetch raw OSM data via Overpass API with retry logic.

    Args:
        min_lat, min_lon, max_lat, max_lon: Bounding box.
        include_railways: Also fetch railway ways.
        include_waterways: Also fetch waterway ways.
        retries: Number of retry attempts.

    Returns:
        Parsed JSON dict or None on failure.
    """
    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    extra_queries = ""
    if include_railways:
        extra_queries += f'way["railway"]({bbox});'
    if include_waterways:
        extra_queries += f'way["waterway"]({bbox});'

    query = f"""
    [out:json][timeout:30];
    (
      way["building"]({bbox});
      way["highway"]({bbox});
      {extra_queries}
    );
    out geom;
    """

    for attempt in range(retries):
        try:
            response = requests.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": query},
                timeout=60,
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code in (429, 503, 504):
                wait = 2 ** attempt
                print(f"  Overpass API {response.status_code}, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Overpass API error: {response.status_code}")
                return None
        except Exception as e:
            print(f"  Overpass request failed: {e}")
            if attempt < retries - 1:
                time.sleep(1)

    return None


# ---------------------------------------------------------------------------
# OSM Rendering
# ---------------------------------------------------------------------------

def render_osm_lines(
    osm_data: dict,
    bounds: Tuple[float, float, float, float],
    img_size: Tuple[int, int] = (1000, 1000),
) -> Tuple[np.ndarray, int, int, int, int]:
    """
    Render OSM ways as black lines on a white background.

    Args:
        osm_data: Parsed Overpass JSON.
        bounds: (min_lat, min_lon, max_lat, max_lon).
        img_size: (height, width) of the output image.

    Returns:
        (image, num_buildings, num_roads, num_railways, num_waterways)
    """
    min_lat, min_lon, max_lat, max_lon = bounds
    h, w = img_size

    img = np.ones((h, w), dtype=np.uint8) * 255

    def geo_to_pixel(lat: float, lon: float) -> Tuple[int, int]:
        x = int((lon - min_lon) / (max_lon - min_lon) * w)
        y = int((max_lat - lat) / (max_lat - min_lat) * h)
        return x, y

    num_buildings = 0
    num_roads = 0
    num_railways = 0
    num_waterways = 0

    for element in osm_data.get("elements", []):
        if element["type"] != "way" or "geometry" not in element:
            continue

        points = []
        for node in element["geometry"]:
            px, py = geo_to_pixel(node["lat"], node["lon"])
            points.append([px, py])

        if len(points) < 2:
            continue

        pts = np.array(points, dtype=np.int32)
        tags = element.get("tags", {})

        if "building" in tags:
            cv2.polylines(img, [pts], isClosed=True, color=0, thickness=1)
            num_buildings += 1
        elif "highway" in tags:
            hw = tags.get("highway", "")
            if hw in ("motorway", "trunk", "primary"):
                thickness = 3
            elif hw in ("secondary", "tertiary"):
                thickness = 2
            else:
                thickness = 1
            cv2.polylines(img, [pts], isClosed=False, color=0, thickness=thickness)
            num_roads += 1
        elif "railway" in tags:
            cv2.polylines(img, [pts], isClosed=False, color=0, thickness=2)
            num_railways += 1
        elif "waterway" in tags:
            cv2.polylines(img, [pts], isClosed=False, color=0, thickness=2)
            num_waterways += 1

    return img, num_buildings, num_roads, num_railways, num_waterways


# ---------------------------------------------------------------------------
# Edge Detection
# ---------------------------------------------------------------------------

def _suppress_text_edges(edges: np.ndarray, min_component_area: int = 40) -> np.ndarray:
    """
    Remove small connected components that likely represent text/noise.

    Text in scanned maps produces many small, isolated edge fragments.
    Structural features (roads, buildings) form larger connected regions.
    By removing components smaller than `min_component_area` pixels,
    we keep structural edges and discard text noise.

    Args:
        edges: Binary edge image from Canny.
        min_component_area: Minimum number of edge pixels in a connected
            component to keep it. Components below this are zeroed out.
            Default 40 works well for 1000-2000px images.

    Returns:
        Cleaned edge image.
    """
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        edges, connectivity=8
    )

    # Build mask: keep only components with area >= min_component_area
    # Label 0 is the background
    cleaned = np.zeros_like(edges)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            cleaned[labels == label_id] = 255

    return cleaned


def edge_detect_pdf(
    image: np.ndarray,
    margin: int = 50,
    denoise: bool = False,
    suppress_text: bool = False,
    min_component_area: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Canny edge detection to a PDF map image.

    Args:
        image: Input image (BGR or grayscale).
        margin: Pixel margin to crop from each side for the template.
            A larger margin (e.g. 80) removes more border noise like
            page edges, legends, and title blocks.
        denoise: If True, use bilateral filtering before Canny to
            suppress text and fine-grained noise while preserving
            structural edges (roads, buildings). Also raises Canny
            thresholds slightly.
        suppress_text: If True, remove small connected components from
            edges (text labels, point annotations). Keeps only larger
            connected structures (roads, buildings, field boundaries).
        min_component_area: Minimum connected component size (in pixels)
            to keep when suppress_text=True.

    Returns:
        (full_edges, cropped_template)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if denoise:
        # Bilateral filter: preserves edges while smoothing texture/text.
        # d=9: filter diameter, sigmaColor=75: color range, sigmaSpace=75: spatial range
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        # Slightly higher Canny thresholds to further suppress noise
        edges = cv2.Canny(filtered, 80, 200)
    else:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)

    # Remove small connected components (text/noise) if requested
    if suppress_text:
        edges = _suppress_text_edges(edges, min_component_area)

    h, w = edges.shape
    template = edges[margin : h - margin, margin : w - margin]

    return edges, template


# ---------------------------------------------------------------------------
# Template Matching — Baseline (original approach)
# ---------------------------------------------------------------------------

def template_match_baseline(
    template: np.ndarray,
    search_img: np.ndarray,
    scales: Optional[List[float]] = None,
    score_threshold: float = 0.01,
) -> Tuple[float, Optional[Tuple[int, int]], Optional[float]]:
    """
    Original multi-scale template matching (no rotation).

    Args:
        template: Edge-detected PDF template (cropped).
        search_img: Inverted OSM edge image (white lines on black).
        scales: Scale factors to try. Default: [0.7..1.2].
        score_threshold: Minimum acceptable match score.

    Returns:
        (best_score, best_location, best_scale)
        Location is (x, y) of the top-left corner of the match.
        Returns (score, None, None) if no valid match found.
    """
    if scales is None:
        scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    best_score = -1.0
    best_loc = None
    best_scale = None

    for scale in scales:
        new_w = int(template.shape[1] * scale)
        new_h = int(template.shape[0] * scale)

        if new_w >= search_img.shape[1] or new_h >= search_img.shape[0]:
            continue
        if new_w < 50 or new_h < 50:
            continue

        resized = cv2.resize(template, (new_w, new_h))
        result = cv2.matchTemplate(search_img, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            best_loc = max_loc
            best_scale = scale

    if best_score < score_threshold:
        return best_score, None, None

    return best_score, best_loc, best_scale


# ---------------------------------------------------------------------------
# Template Matching — Improved
# ---------------------------------------------------------------------------

def _dilate_edges(
    image: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Dilate edge pixels to create "fuzzy" edges for more tolerant matching.

    Dilating both template and search image creates a tolerance band so
    features that are close-but-not-pixel-aligned still contribute to
    the match score. Improves NCC scores without sacrificing positional
    accuracy.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    return cv2.dilate(image, kernel, iterations=iterations)


def template_match_improved(
    template: np.ndarray,
    search_img: np.ndarray,
    scales: Optional[List[float]] = None,
    score_threshold: float = 0.01,
    dilate: bool = True,
    dilation_kernel: int = 3,
    dilation_iterations: int = 1,
) -> Tuple[float, Optional[Tuple[int, int]], Optional[float]]:
    """
    Improved template matching: fuzzy dilation + coarse-to-fine scale search.

    Two improvements over baseline:
    1. Dilate both template and search image edges before matching, creating
       tolerance bands that bridge the visual gap between PDF maps and OSM.
    2. Fine-scale refinement: after the coarse pass, search ±0.05 around
       the best scale in 0.01 steps for a more precise fit.

    Args:
        template: Edge-detected PDF template (cropped).
        search_img: Inverted OSM edge image (white lines on black).
        scales: Coarse scale factors to try. Default: [0.7..1.2].
        score_threshold: Minimum acceptable match score.
        dilate: Dilate both images before matching.
        dilation_kernel: Size of the dilation kernel.
        dilation_iterations: Number of dilation passes.

    Returns:
        (best_score, best_location, best_scale)
        Same signature as template_match_baseline for easy swapping.
    """
    if scales is None:
        scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    # Fuzzy matching via dilation
    if dilate:
        template = _dilate_edges(template, dilation_kernel, dilation_iterations)
        search_img = _dilate_edges(search_img, dilation_kernel, dilation_iterations)

    best_score = -1.0
    best_loc = None
    best_scale = None

    # Coarse pass — same scales as baseline
    for scale in scales:
        new_w = int(template.shape[1] * scale)
        new_h = int(template.shape[0] * scale)

        if new_w >= search_img.shape[1] or new_h >= search_img.shape[0]:
            continue
        if new_w < 50 or new_h < 50:
            continue

        resized = cv2.resize(template, (new_w, new_h))
        result = cv2.matchTemplate(search_img, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            best_loc = max_loc
            best_scale = scale

    # Fine pass — ±0.05 around best coarse scale in 0.01 steps
    min_scale = min(scales)
    max_scale = max(scales)
    if best_scale is not None:
        fine_scales = [
            best_scale + delta
            for delta in [-0.05, -0.04, -0.03, -0.02, -0.01,
                          0.01, 0.02, 0.03, 0.04, 0.05]
            if min_scale <= best_scale + delta <= max_scale
        ]
        for scale in fine_scales:
            new_w = int(template.shape[1] * scale)
            new_h = int(template.shape[0] * scale)
            if new_w >= search_img.shape[1] or new_h >= search_img.shape[0]:
                continue
            if new_w < 50 or new_h < 50:
                continue
            resized = cv2.resize(template, (new_w, new_h))
            result = cv2.matchTemplate(search_img, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best_loc = max_loc
                best_scale = scale

    if best_score < score_threshold:
        return best_score, None, None

    return best_score, best_loc, best_scale


# ---------------------------------------------------------------------------
# Coordinate Helpers
# ---------------------------------------------------------------------------

def calculate_search_bounds(
    lat: float,
    lon: float,
    scale_meters: float,
    image_height: int,
    image_width: int,
    search_margin: float = 200.0,
) -> Tuple[float, float, float, float]:
    """
    Calculate the geographic bounding box for the OSM search area.

    Args:
        lat, lon: Estimated center of the map.
        scale_meters: Real-world width of the map in meters.
        image_height, image_width: PDF image dimensions in pixels.
        search_margin: Extra margin in meters around the map area.

    Returns:
        (min_lat, min_lon, max_lat, max_lon)
    """
    search_width_m = scale_meters + search_margin * 2
    search_height_m = (scale_meters * image_height / image_width) + search_margin * 2

    lat_per_m = 1.0 / METERS_PER_DEGREE_LAT
    lon_per_m = 1.0 / (METERS_PER_DEGREE_LAT * math.cos(math.radians(lat)))

    half_lat = (search_height_m / 2) * lat_per_m
    half_lon = (search_width_m / 2) * lon_per_m

    return (lat - half_lat, lon - half_lon, lat + half_lat, lon + half_lon)


def calculate_osm_image_size(
    search_width_m: float,
    search_height_m: float,
    pdf_m_per_px: float,
    min_size: int = 500,
    max_size: int = 3000,
) -> Tuple[int, int]:
    """
    Calculate OSM rendering image size to match PDF resolution.

    Args:
        search_width_m, search_height_m: Search area in meters.
        pdf_m_per_px: Meters per pixel of the PDF image.
        min_size, max_size: Clamp dimensions.

    Returns:
        (height, width)
    """
    w = int(search_width_m / pdf_m_per_px)
    h = int(search_height_m / pdf_m_per_px)
    w = max(min_size, min(max_size, w))
    h = max(min_size, min(max_size, h))
    return h, w


def match_location_to_geo(
    loc: Tuple[int, int],
    match_w: int,
    match_h: int,
    osm_img_shape: Tuple[int, int],
    bounds: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """
    Convert a template match pixel location to geographic coordinates.

    Args:
        loc: Top-left (x, y) of the match rectangle.
        match_w, match_h: Size of the matched template.
        osm_img_shape: (height, width) of the OSM image.
        bounds: (min_lat, min_lon, max_lat, max_lon).

    Returns:
        (latitude, longitude) of the match center.
    """
    min_lat, min_lon, max_lat, max_lon = bounds
    img_h, img_w = osm_img_shape

    center_x = loc[0] + match_w // 2
    center_y = loc[1] + match_h // 2

    matched_lon = min_lon + (center_x / img_w) * (max_lon - min_lon)
    matched_lat = max_lat - (center_y / img_h) * (max_lat - min_lat)

    return matched_lat, matched_lon
