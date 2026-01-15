"""
Boundary Detection Tools for Planning Document Digitization

Two methods for extracting planning area boundaries from map images:

1. extract_color_boundary: HSV color filtering
   - Best for maps with a distinct colored boundary line (orange, red, blue, etc.)
   - The boundary color must be visually different from the background map

2. extract_region_boundary: Grayscale edge detection
   - Best for maps with black/dark boundary lines or low-contrast boundaries
   - Works when the boundary is drawn with a thin dark line on a light background
   - Does not rely on color, uses edge detection instead
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import base64


def extract_color_boundary(
    image_base64: str,
    lower_hsv: List[int],
    upper_hsv: List[int],
) -> Dict[str, Any]:
    """
    Extract boundary from a map image using HSV color filtering.

    USE THIS METHOD WHEN:
    - The boundary is drawn with a distinct colored line (orange, red, blue, green, etc.)
    - The boundary color stands out from the background map imagery
    - You can clearly see a colored line marking the planning area

    DO NOT USE THIS METHOD WHEN:
    - The boundary is black or very dark (use extract_region_boundary instead)
    - The boundary color is similar to other map features
    - The boundary line is very thin or faint

    Args:
        image_base64: Base64-encoded image string
        lower_hsv: Lower HSV bound [H, S, V]. OpenCV uses H:0-179, S:0-255, V:0-255
        upper_hsv: Upper HSV bound [H, S, V]

    Returns:
        Dict with 'success', 'image_height', 'image_width', 'boundary_pixels'
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"success": False, "error": "Failed to decode image"}

        image_height, image_width = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Extract color region using HSV filtering
        lower = np.array(lower_hsv)
        upper = np.array(upper_hsv)
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological operations to connect broken lines
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=2)
        mask_closed = cv2.morphologyEx(
            mask_dilated, cv2.MORPH_CLOSE, kernel, iterations=2
        )

        # Fill internal regions to ensure closed boundary
        mask_filled = mask_closed.copy()
        h, w = mask_filled.shape[:2]
        mask_temp = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(mask_filled, mask_temp, (0, 0), 255)
        mask_filled_inv = cv2.bitwise_not(mask_filled)
        mask_final = mask_closed | mask_filled_inv

        # Find contours and select the largest one
        contours, _ = cv2.findContours(
            mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return {
                "success": False,
                "error": "No colored boundary detected. Try adjusting HSV range.",
                "image_height": image_height,
                "image_width": image_width,
            }

        # Sort by area (largest first) and return all as candidates
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        candidates = []
        for contour in sorted_contours:
            # Smooth the boundary
            epsilon = 0.002 * cv2.arcLength(contour, True)
            smooth_contour = cv2.approxPolyDP(contour, epsilon, True)
            candidates.append(smooth_contour.reshape(-1, 2).tolist())

        return {
            "success": True,
            "image_height": image_height,
            "image_width": image_width,
            "candidates": candidates,
        }

    except Exception as e:
        return {"success": False, "error": f"Color boundary extraction failed: {e}"}


def extract_region_boundary(image_base64: str) -> Dict[str, Any]:
    """
    Extract boundary from a map image using grayscale edge detection.

    USE THIS METHOD WHEN:
    - The boundary is drawn with a black or dark line
    - The boundary color is not distinct (similar to other map features)
    - The boundary is a thin dark line on a light background
    - Color-based extraction (extract_color_boundary) doesn't work

    DO NOT USE THIS METHOD WHEN:
    - The boundary is a bright/vivid color (use extract_color_boundary instead)
    - The map has many dark lines that could be confused with the boundary
    - The background is dark

    Args:
        image_base64: Base64-encoded image string

    Returns:
        Dict with 'success', 'image_height', 'image_width', 'boundary_pixels'
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"success": False, "error": "Failed to decode image"}

        image_height, image_width = img.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

        # Edge detection (Canny)
        edges = cv2.Canny(opened, threshold1=50, threshold2=150)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find external contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return {
                "success": False,
                "error": "No boundary contour detected.",
                "image_height": image_height,
                "image_width": image_width,
            }

        # Find the largest contour (usually the main boundary)
        largest_contour = max(contours, key=cv2.contourArea)

        # Convert to list of [x, y] coordinates
        boundary_pixels = largest_contour.reshape(-1, 2).tolist()

        return {
            "success": True,
            "image_height": image_height,
            "image_width": image_width,
            "boundary_pixels": boundary_pixels,
            "num_points": len(boundary_pixels),
        }

    except Exception as e:
        return {"success": False, "error": f"Region boundary extraction failed: {e}"}


# Tool definitions for LLM function calling
BOUNDARY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_color_boundary",
            "description": """Extract boundary using HSV color filtering.

USE THIS WHEN the boundary is a distinct colored line (orange, red, blue, green, etc.)
that stands out from the background map.

DO NOT USE when the boundary is black/dark - use extract_region_boundary instead.

HSV COLOR GUIDE (OpenCV ranges):
- H (Hue): 0-179
  * Red: 0-10 (also 170-179 for darker reds)
  * Orange: 5-25
  * Yellow: 25-35
  * Green: 35-85
  * Blue: 90-130
  * Purple: 130-170
- S (Saturation): 0-255. Use 70+ to exclude grays.
- V (Value/Brightness): 0-255. Use 80+ to exclude dark areas.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_base64": {
                        "type": "string",
                        "description": "Base64-encoded map image",
                    },
                    "lower_hsv": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Lower HSV bound [H, S, V]. Example: [5, 80, 80] for orange",
                    },
                    "upper_hsv": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Upper HSV bound [H, S, V]. Example: [25, 255, 255] for orange",
                    },
                },
                "required": ["image_base64", "lower_hsv", "upper_hsv"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_region_boundary",
            "description": """Extract boundary using grayscale edge detection.

USE THIS WHEN:
- The boundary is black or dark colored
- Color-based extraction doesn't work
- The boundary is a thin dark line on a light background

DO NOT USE when the boundary is a bright/vivid color - use extract_color_boundary instead.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_base64": {
                        "type": "string",
                        "description": "Base64-encoded map image",
                    },
                },
                "required": ["image_base64"],
            },
        },
    },
]
