"""
OpenRouter Client - Unified LLM interface supporting multiple models
Supports Claude Opus 4.5, GPT-5.2, Gemini 3 Pro, and other models via OpenRouter API

This module provides multiple approaches for extracting GeoJSON from UK planning documents:
1. Baseline: Direct LLM extraction (extract_geojson_from_pdf)
2. Linear Transformation: CV boundary detection + center/scale estimation (extract_geojson_linear_transform)
   - Includes automatic district lookup for documents covering entire districts
3. Tool-based Agentic: LLM decides which tools to use (extract_geojson_agentic)
"""

import os
import time
import json
import base64
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import requests
from dotenv import load_dotenv

import cv2
import numpy as np

from tools import (
    # PDF tools
    get_pdf_page_as_image,
    # Boundary tools
    extract_color_boundary,
    extract_region_boundary,
    # Geo tools
    geocode_address,
    lookup_district_boundary,
    pixels_to_geo_linear,
    # Visualization tools
    visualize_geojson_boundary,
    # All tool definitions for agentic workflow
    ALL_TOOLS,
)

# Load environment variables from .env file (e.g., API keys)
load_dotenv()


class OpenRouterClient:
    """
    Unified client for accessing multiple LLM providers through OpenRouter.

    Supports:
    - Text generation
    - Vision/multimodal (PDF, images)
    - Structured JSON output
    - Multiple model providers (Claude, GPT-4o, Gemini, etc.)
    """

    # Shorthand mappings to full model identifiers
    # Allows users to write "claude-opus" instead of "anthropic/claude-opus-4.5"
    MODELS = {
        "claude-opus": "anthropic/claude-opus-4.5",
        "gpt-5.2": "openai/gpt-5.2-pro",
        "gemini-pro": "google/gemini-3-pro-preview",
    }

    DEFAULT_HSV_LOWER_RANGE = [5, 80, 80]  # Orange-ish
    DEFAULT_HSV_UPPER_RANGE = [25, 255, 255]  # Orange-ish
    DEFAULT_CENTER_LAT = 51.5  # London
    DEFAULT_CENTER_LON = -0.15  # London
    DEFAULT_SCALE_METERS = 1000
    DEFAULT_SCALE_SOURCE = "scale_bar"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "anthropic/claude-opus-4.5",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model: Model identifier (use MODELS dict or full model path)
            base_url: OpenRouter API endpoint
        """
        # Use provided key, or fall back to environment variable
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable."
            )

        # dict.get() returns the value if key exists, otherwise returns the key itself
        # This allows both "claude-opus" and "anthropic/claude-opus-4.5" to work
        self.model = self.MODELS.get(model, model)
        self.base_url = base_url

        print(f"OpenRouterClient initialized with model: {self.model}")

    def _parse_json_from_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response, handling markdown code blocks."""
        # Try direct parse first (response might be pure JSON)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # LLMs often wrap JSON in markdown code blocks
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",  # ```json ... ```
            r"```geojson\s*([\s\S]*?)\s*```",  # ```geojson ... ```
            r"```\s*([\s\S]*?)\s*```",  # ``` ... ``` (no lang)
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)  # Case insensitive
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

        # Last resort: find JSON object in text using brace matching
        # Find first { and last matching }
        start = content.find("{")
        if start != -1:
            brace_count = 0
            for i, char in enumerate(content[start:], start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(content[start : i + 1])
                        except json.JSONDecodeError:
                            break

        return None

    # Core API Methods
    def _chat_with_attachment(
        self,
        prompt: str,
        attachment: Dict[str, Any],
        system_message: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Internal method to send a prompt with an attachment (image or file).

        Args:
            prompt: Question or instruction
            attachment: Content block for the attachment
            system_message: Optional system instruction
            max_tokens: Maximum response length

        Returns:
            Response dict with 'content', 'model', 'tokens', 'processing_time', 'success'
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})

        user_content = [{"type": "text", "text": prompt}, attachment]
        messages.append({"role": "user", "content": user_content})

        return self._send_completion(messages=messages, max_tokens=max_tokens)

    def chat_with_images(
        self,
        images_base64: List[str],
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Send a prompt along with an aribtrary number of images.

        Args:
            images_base64: List of base64-encoded image strings
            prompt: Question or instruction about the images
            system_message: Optional system instruction
            max_tokens: Maximum response length

        Returns:
            Response dict with 'content', 'model', 'tokens', 'processing_time', 'success'
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})

        user_content = [{"type": "text", "text": prompt}]
        for img_b64 in images_base64:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                }
            )
        messages.append({"role": "user", "content": user_content})

        return self._send_completion(messages=messages, max_tokens=max_tokens)

    def chat_with_pdf(
        self,
        pdf_path: str,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Send a prompt along with a PDF document.

        Args:
            pdf_path: Path to PDF file
            prompt: Question or instruction about the PDF
            system_message: Optional system instruction
            max_tokens: Maximum response length

        Returns:
            Response dict with 'content', 'model', 'tokens', 'processing_time', 'success'
        """
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        with open(pdf_path, "rb") as f:
            pdf_data = base64.b64encode(f.read()).decode("utf-8")

        attachment = {
            "type": "file",
            "file": {
                "filename": os.path.basename(pdf_path),
                "file_data": f"data:application/pdf;base64,{pdf_data}",
            },
        }
        return self._chat_with_attachment(
            prompt, attachment, system_message, max_tokens
        )

    def _send_completion(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """Send a chat completion request to OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        start_time = time.time()

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,  # Automatically serializes dict to JSON
                timeout=120,  # Seconds before request times out
            )
            # Raises HTTPError for 4xx/5xx status codes
            response.raise_for_status()
            data = response.json()

            # OpenRouter follows OpenAI's response format:
            # choices[0].message.content contains the model's text response
            content = data["choices"][0]["message"]["content"]

            result = {
                "content": content,
                "model": data.get("model", self.model),
                "tokens": data.get("usage", {}),
                "processing_time": time.time() - start_time,
                "success": True,
            }

            # Attempt to extract structured JSON from the response
            parsed = self._parse_json_from_response(content)
            if parsed:
                result["parsed_json"] = parsed
            else:
                result["error"] = "Could not extract valid JSON from response"
                result["success"] = False

            return result

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    # Method 1: Extract GeoJSON from PDF directly

    def extract_geojson_from_pdf(
        self, pdf_path: str, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract planning area boundary as GeoJSON from PDF.

        Args:
            pdf_path: Path to planning document PDF
            context: Additional context about the document

        Returns:
            Dict with GeoJSON and metadata
        """
        # f-string with conditional: only includes context line if context is provided
        prompt = f"""You are a GIS analyst specializing in UK planning documents.

Analyze this planning document and extract the geographic boundary of the planning area as GeoJSON.

Task:
1. Identify if the document contains a map showing the planning area boundary
2. Look for coordinate information (latitude/longitude, OS grid references, or textual descriptions)
3. Extract or infer the boundary polygon(s)

Requirements:
- Output a valid GeoJSON Feature with MultiPolygon geometry
- Always use "MultiPolygon" type (even for single connected areas, wrap it in MultiPolygon format)
- Coordinates must be in [longitude, latitude] format (WGS84)
- If the document shows multiple disconnected areas, include all of them in the MultiPolygon
- If the document lacks precise coordinates, make a reasonable estimate based on:
  * Street names and landmarks mentioned
  * The map boundaries if visible
  * The administrative area description

{f"Additional context: {context}" if context else ""}

Output only valid GeoJSON in this exact format:
{{
"type": "Feature",
"geometry": {{
    "type": "MultiPolygon",
    "coordinates": [
        [[[lon1, lat1], [lon2, lat2], [lon3, lat3], [lon1, lat1]]],
        [[[lon4, lat4], [lon5, lat5], [lon6, lat6], [lon4, lat4]]]
    ]
}},
"properties": {{
    "source": "planning_document",
    "confidence": "high|medium|low",
    "method": "description of extraction method"
}}
}}

Notes:
- For a single connected area, use one polygon in the MultiPolygon array
- For multiple disconnected areas, use multiple polygons in the array
- Each polygon must close (first and last coordinate pairs must be identical)"""

        system_message = (
            "You are a GIS analyst. Always respond with valid GeoJSON only."
        )

        result = self.chat_with_pdf(
            pdf_path=pdf_path,
            prompt=prompt,
            system_message=system_message,
            max_tokens=8192,
        )

        return result

    # Method 2: Linear Transformation (CV + Center/Scale Estimation) -> Proposed workflow

    def _analyze_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Combined document analysis: identify map page, center/scale, boundary color,
        and whether the planning area covers an entire administrative district.

        Performs all PDF analysis in a single API call to reduce costs.

        Returns:
            Dict with:
            - has_map: bool
            - page_index: int (0-indexed) or None
            - covers_district: bool (if True, use district lookup instead of linear transform)
            - district_name: str or None (full OSM-compatible district name)
            - center_place: str
            - center_lat: float
            - center_lon: float
            - scale_meters: float
            - scale_source: str
            - boundary_method: "color" or "edge"
            - lower_hsv: [H, S, V] (if color method)
            - upper_hsv: [H, S, V] (if color method)
            - confidence: str
        """
        prompt = """Analyze this UK planning document for GeoJSON extraction.

TASK 1: FIND THE MAP PAGE

Look through the PDF for a page containing a planning area map with a boundary.

Use ZERO-BASED page indexing (first page = 0, second page = 1, etc.)

If NO MAP exists, set has_map=false and SKIP to TASK 2 only.

TASK 2: CHECK IF PLANNING AREA COVERS AN ENTIRE DISTRICT

This is CRITICAL for documents WITHOUT maps - it's our only way to extract a boundary.

Set covers_district=true if the document explicitly states the planning area is:
- An entire London Borough (e.g., "London Borough of Barking and Dagenham")
- An entire Royal Borough (e.g., "Royal Borough of Kensington and Chelsea")
- An entire district/city (e.g., "City of Westminster")
- A whole ward or parish

Look for phrases like:
- "The land comprising the entire borough of..."
- "All of [district name]"
- "The whole area of..."
- Document title containing just a borough/district name

If covers_district=true, you MUST provide district_name in OSM-geocodable format:
- "London Borough of Barking and Dagenham, London, UK"
- "Royal Borough of Kensington and Chelsea, London, UK"
- "City of Westminster, London, UK"

The district_name MUST be specific enough to look up in OpenStreetMap.

If the planning area is a SPECIFIC SITE within a district, set covers_district=false.

TASK 3: CENTER, SCALE, AND BOUNDARY METHOD (ONLY if has_map=true AND covers_district=false)

Skip this entire task if has_map=false OR covers_district=true.

CENTER: Find a place name representing the CENTER of the planning area.
- Choose a location IN OR NEAR THE CENTER
- Prefer names that geocode well in OpenStreetMap
- AVOID generic names like "High Street"
- Include city/town for context (e.g., "Peckham, London")

SCALE: The scale_meters value = TOTAL WIDTH of map area in meters.
Scale clues (in order of reliability):
1. Scale bar - measure how many times it fits across the map, multiply
2. Scale ratio (e.g., "1:2500") - for A4: 210mm × 2500 = 525m
3. Known features - estimate distances between recognizable landmarks

BOUNDARY METHOD: Look at the planning area boundary on the map.

COLOR-BASED (boundary_method="color"): Use when boundary is colored
- Specify HSV range (OpenCV: H=0-179, S=0-255, V=0-255)

FOR FILLED REGIONS (lower saturation ~30-50):
* Pink/Magenta: H=140-175, S=30-150, V=150-255
* Light red/salmon: H=0-10, S=30-150, V=150-255
* Light orange: H=5-25, S=30-150, V=150-255
* Light blue: H=90-130, S=30-150, V=150-255

FOR BOLD LINES (higher saturation ~70+):
* Red line: H=0-10, S=70-255, V=80-255
* Orange line: H=5-25, S=70-255, V=80-255
* Blue line: H=90-130, S=70-255, V=80-255

EDGE-BASED (boundary_method="edge"): Use when boundary is BLACK or DARK

OUTPUT FORMAT (valid JSON only)

Case 1 - No map, covers entire district (OSM lookup from text):
{
    "has_map": false,
    "page_index": null,
    "covers_district": true,
    "district_name": "London Borough of X, London, UK"
}

Case 2 - Has map, covers entire district (OSM lookup, map for reference only):
{
    "has_map": true,
    "page_index": 0,
    "covers_district": true,
    "district_name": "London Borough of X, London, UK"
}

Case 3 - Has map, specific site (linear transform):
{
    "has_map": true,
    "page_index": 0,
    "covers_district": false,
    "district_name": null,
    "center_place": "Specific place, City",
    "center_lat": 51.5,
    "center_lon": -0.15,
    "scale_meters": 1000,
    "scale_source": "scale_bar|scale_ratio|estimated",
    "boundary_method": "color",
    "lower_hsv": [H, S, V],
    "upper_hsv": [H, S, V],
    "boundary_reason": "pink shaded region",
    "confidence": "high|medium|low"
}

Case 4 - No map, no district coverage (CANNOT extract):
{
    "has_map": false,
    "page_index": null,
    "covers_district": false,
    "district_name": null
}"""

        result = self.chat_with_pdf(
            pdf_path=pdf_path,
            prompt=prompt,
            system_message="You are a GIS analyst and computer vision expert specializing in UK planning documents. Output only valid JSON.",
        )

        if result.get("parsed_json"):
            data = result["parsed_json"]
            if data.get("covers_district"):
                print(
                    f"Analysis: page {data.get('page_index')}, district={data.get('district_name')}"
                )
            else:
                print(
                    f"Analysis: page {data.get('page_index')}, center={data.get('center_place')}, scale={data.get('scale_meters')}m, method={data.get('boundary_method')}"
                )
            return data

        print("Analysis failed, no map found")
        return {"has_map": False, "page_index": None}

    def _refine_placement_with_llm(
        self,
        map_image_b64: str,
        current_geojson: Dict[str, Any],
        current_center: Tuple[float, float],
        current_scale: float,
        boundary_pixels: List[List[int]],
        image_height: int,
        image_width: int,
        max_iterations: int = 5,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Iteratively refine boundary placement by comparing landmarks in PDF vs OSM.

        The LLM identifies a landmark visible in both images, describes where the
        boundary is relative to that landmark in each image, and suggests a shift.
        """

        lat, lon = current_center
        scale = current_scale
        geojson = current_geojson
        refinement_log = []

        METERS_PER_DEGREE_LAT = 111111.0

        for i in range(max_iterations):
            # Visualize current placement on OSM
            viz = visualize_geojson_boundary(geojson)
            if not viz.get("success"):
                refinement_log.append(
                    {
                        "iteration": i + 1,
                        "status": "viz_failed",
                        "error": viz.get("error"),
                    }
                )
                break

            # Refinement prompt - landmark-based comparison
            prompt = f"""You are comparing two maps to check if the red boundary is correctly positioned.

IMAGE 1 (FIRST): Original PDF map showing the CORRECT boundary (pink/red shaded area)
IMAGE 2 (SECOND): OSM map showing our CURRENT boundary placement (red outline)

Map scale: ~{scale:.0f}m width

TASK: Check if the boundary aligns correctly with landmarks

1. Find a LANDMARK visible in BOTH images (road, junction, building, railway)

2. In the PDF: Describe where the pink boundary is relative to that landmark
   Example: "The boundary's southern edge touches Main Road"

3. In the OSM: Describe where the red boundary is relative to that same landmark
   Example: "The boundary is 100m north of Main Road, not touching it"

4. If they don't match, calculate how to shift the OSM boundary to match the PDF

DIRECTION REFERENCE (this is critical!):

Think about it this way:
- If the boundary in OSM is TOO FAR NORTH compared to PDF → move it SOUTH → shift_north_m = NEGATIVE
- If the boundary in OSM is TOO FAR SOUTH compared to PDF → move it NORTH → shift_north_m = POSITIVE
- If the boundary in OSM is TOO FAR EAST compared to PDF → move it WEST → shift_east_m = NEGATIVE
- If the boundary in OSM is TOO FAR WEST compared to PDF → move it EAST → shift_east_m = POSITIVE

OUTPUT (JSON only):

{{
    "assessment": "good" or "needs_adjustment",
    "landmark": "the landmark you used for comparison",
    "pdf_boundary_position": "where is boundary relative to landmark in PDF",
    "osm_boundary_position": "where is boundary relative to landmark in OSM",
    "direction_reasoning": "The OSM boundary is too far [NORTH/SOUTH/EAST/WEST] compared to PDF, so I need to shift it [opposite direction]",
    "shift_north_m": <number in meters, NEGATIVE to move south, POSITIVE to move north>,
    "shift_east_m": <number in meters, NEGATIVE to move west, POSITIVE to move east>
}}

Remember: shift values move the boundary. NEGATIVE shift_north_m moves it SOUTH."""

            # Send to LLM
            result = self.chat_with_images(
                images_base64=[map_image_b64, viz["image_base64"]],
                prompt=prompt,
                system_message="You are a geospatial expert. Compare the boundary positions carefully and output valid JSON.",
            )

            parsed = result.get("parsed_json")
            if not parsed:
                refinement_log.append({"iteration": i + 1, "status": "parse_failed"})
                continue

            refinement_log.append({"iteration": i + 1, **parsed})

            assessment = parsed.get("assessment", "unknown")
            landmark = parsed.get("landmark", "unknown")

            # Stop if alignment is good
            if assessment == "good":
                print(f"Refinement {i + 1}: good (landmark: {landmark})")
                break

            # Get shift values
            shift_north_m = parsed.get("shift_north_m", 0)
            shift_east_m = parsed.get("shift_east_m", 0)

            # Skip tiny adjustments
            if abs(shift_north_m) < 5 and abs(shift_east_m) < 5:
                print(f"Refinement {i + 1}: converged (shift <5m)")
                break

            print(
                f"Refinement {i + 1}: shift {shift_north_m:+.0f}m N, {shift_east_m:+.0f}m E (landmark: {landmark})"
            )

            # Apply shifts
            lat += shift_north_m / METERS_PER_DEGREE_LAT
            lon += shift_east_m / (METERS_PER_DEGREE_LAT * np.cos(np.radians(lat)))

            # Re-transform boundary with new center
            transform_result = pixels_to_geo_linear(
                boundary_pixels=boundary_pixels,
                image_height=image_height,
                image_width=image_width,
                center_lat=lat,
                center_lon=lon,
                scale_meters=scale,
            )

            if transform_result.get("success"):
                geojson = transform_result["geojson"]
                if geojson.get("geometry", {}).get("type") == "Polygon":
                    geojson["geometry"] = {
                        "type": "MultiPolygon",
                        "coordinates": [geojson["geometry"]["coordinates"]],
                    }
            else:
                refinement_log.append(
                    {
                        "iteration": i + 1,
                        "status": "transform_failed",
                        "error": transform_result.get("error"),
                    }
                )
                break

        geojson["properties"]["refinement_iterations"] = len(refinement_log)
        return geojson, refinement_log

    def _select_boundary_candidate(
        self,
        image_base64: str,
        candidates: List[List[List[int]]],
    ) -> List[List[int]]:
        """
        Use LLM to select the correct planning boundary from multiple candidates.
        This is done because the boundary extraction tools may not always extract the correct boundary,
        for example if the planning area in the original map is drawn in red but there is also a large
        red circular seal on that same pdf page, the boundary extraction tools may extract the large red circular seal instead of the planning area.
        Therefore, we need to use the LLM to select the correct boundary from the multiple candidates.

        Args:
            image_base64: Original map image
            candidates: List of boundary pixel arrays

        Returns:
            The selected boundary pixels
        """
        # Visualize each candidate with a different color
        color = (
            0,
            255,
            0,
        )  # Green (planning boundaries in original document are usually not green so it will be easy for the model to distinguish the proposed boundary)

        visualized_images = []
        for i, candidate in enumerate(candidates[:5]):  # Limit to 5 candidates
            viz = self.visualize_boundary_on_image(
                image_base64,
                candidate,
                line_color=color,
                fill_color=(*color, 50),
            )

            if viz.get("success"):
                visualized_images.append(viz["image_base64"])

        if not visualized_images:
            # Default to the first candidate (which is boundary with largest area because the candidates are sorted by area in descending order)
            return candidates[0]

        # Include original image first, then the candidate visualizations
        images = [image_base64] + visualized_images

        prompt = f"""IMAGE 1: The original planning map showing the actual planning boundary.

IMAGES 2-{len(visualized_images) + 1}: Each shows a different detected boundary candidate overlaid on the map.
The candidates are colored: Green (0), Blue (1), Yellow (2), Magenta (2).

Compare the candidates to the original map. Which candidate correctly matches the planning boundary shown in the original?

Ignore any seals, stamps, logos, or decorative elements.

Return JSON: {{"selected_index": <0-based index of correct candidate>}}"""

        result = self.chat_with_images(images, prompt)

        if result.get("parsed_json") and "selected_index" in result["parsed_json"]:
            idx = result["parsed_json"]["selected_index"]
            if 0 <= idx < len(candidates):
                return candidates[idx]

        # Default to the first candidate (which is boundary with largest area because the candidates are sorted by area in descending order)
        return candidates[0]

    def visualize_boundary_on_image(
        self,
        image_base64: str,
        boundary_pixels: List[List[int]],
        output_path: Optional[str] = None,
        line_color: Tuple[int, int, int] = (255, 0, 0),
        line_thickness: int = 2,
        fill_color: Optional[Tuple[int, int, int, int]] = (255, 0, 0, 50),
    ) -> Dict[str, Any]:
        """
        Visualize the extracted boundary overlaid on the original map image.

        Args:
            image_base64: Base64-encoded original map image
            boundary_pixels: List of [x, y] pixel coordinates forming the boundary
            output_path: Optional path to save the visualization (PNG format)
            line_color: RGB color for the boundary line (default: red)
            line_thickness: Thickness of the boundary line in pixels
            fill_color: Optional RGBA color to fill the boundary area (default: semi-transparent red)

        Returns:
            Dict with:
            - "success": bool
            - "image_base64": Base64-encoded visualization image
            - "output_path": Path where image was saved (if output_path provided)
        """
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return {"success": False, "error": "Failed to decode image"}

            # Convert boundary to numpy array
            pts = np.array(boundary_pixels, dtype=np.int32)

            # Create overlay for semi-transparent fill
            if fill_color is not None:
                overlay = img.copy()
                # OpenCV uses BGR, so convert RGB to BGR
                fill_bgr = (fill_color[2], fill_color[1], fill_color[0])
                cv2.fillPoly(overlay, [pts], fill_bgr)
                # Blend with original (alpha from fill_color)
                alpha = fill_color[3] / 255.0
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

            # Draw boundary outline
            line_bgr = (line_color[2], line_color[1], line_color[0])
            cv2.polylines(
                img, [pts], isClosed=True, color=line_bgr, thickness=line_thickness
            )

            # Draw vertex markers
            for x, y in boundary_pixels[:: max(1, len(boundary_pixels) // 20)]:
                cv2.circle(img, (int(x), int(y)), 4, line_bgr, -1)

            # Convert back to base64
            _, buffer = cv2.imencode(".png", img)
            result_base64 = base64.b64encode(buffer).decode("utf-8")

            result = {
                "success": True,
                "image_base64": result_base64,
                "image_height": img.shape[0],
                "image_width": img.shape[1],
                "num_boundary_points": len(boundary_pixels),
            }

            # Save to file if path provided
            if output_path:
                cv2.imwrite(output_path, img)
                result["output_path"] = output_path

            return result

        except Exception as e:
            return {"success": False, "error": f"Visualization failed: {e}"}

    def extract_geojson_linear_transform(
        self,
        pdf_path: str,
        iterative_refinement: bool = False,
        max_refinement_iterations: int = 5,
    ) -> Dict[str, Any]:
        """
        Extract GeoJSON using the best available method.

        Workflow:
        1. Analyze document (map page, center/scale, boundary color, district check)
        2. If planning area covers entire district → use OSM district lookup
        3. Otherwise → extract boundary and apply linear transformation
        4. (Optional) Iterative refinement using visual feedback

        Args:
            pdf_path: Path to the PDF file
            iterative_refinement: If True, uses LLM to iteratively refine placement (default: False)
            max_refinement_iterations: Maximum number of refinement iterations (default: 5)

        This method reuses the tools/ package to avoid code duplication.
        """
        start_time = time.time()

        try:
            # Combined analysis: single API call for all document info
            analysis = self._analyze_document(pdf_path)

            # Try district lookup first (works even without a map)
            if analysis.get("covers_district") and analysis.get("district_name"):
                district_name = analysis["district_name"]
                lookup_result = lookup_district_boundary(district_name)

                if lookup_result.get("success"):
                    geojson = lookup_result["geojson"]
                    geom = geojson.get("geometry", {})
                    if geom.get("type") == "Polygon":
                        geojson["geometry"] = {
                            "type": "MultiPolygon",
                            "coordinates": [geom["coordinates"]],
                        }
                    geojson["properties"]["source"] = "osm_district_lookup"
                    geojson["properties"]["district_name"] = district_name
                    print(
                        f"Output: district_lookup, time={time.time() - start_time:.1f}s"
                    )
                    return {
                        "success": True,
                        "parsed_json": geojson,
                        "method": "district_lookup",
                        "processing_time": time.time() - start_time,
                    }
                elif not analysis.get("has_map"):
                    # No map and district lookup failed - cannot proceed
                    return {
                        "success": False,
                        "error": f"District lookup failed for '{district_name}' and no map available for boundary extraction",
                        "processing_time": time.time() - start_time,
                    }
                else:
                    # District lookup failed but we have a map - fall back to linear transform
                    print(f"District lookup failed, falling back to linear transform")

            # Check if we have a map for linear transform
            if not analysis.get("has_map"):
                return {
                    "success": False,
                    "error": "No planning map found in PDF and no district boundary identified",
                    "processing_time": time.time() - start_time,
                }

            # Linear transformation path
            map_page = analysis.get("page_index")
            center_place = analysis.get("center_place")
            lat_center = analysis.get("center_lat", self.DEFAULT_CENTER_LAT)
            lon_center = analysis.get("center_lon", self.DEFAULT_CENTER_LON)
            center_detection_method = "combined_analysis"

            if center_place:
                geocode_result = geocode_address(center_place)
                if geocode_result.get("success"):
                    lat_center = geocode_result["latitude"]
                    lon_center = geocode_result["longitude"]
                    center_detection_method = f"geocoded:{center_place}"
                    print(
                        f"Geocoded: {center_place} -> ({lat_center:.4f}, {lon_center:.4f})"
                    )

            scale_m = analysis.get("scale_meters", self.DEFAULT_SCALE_METERS)
            scale_source = analysis.get("scale_source", "unknown")
            center_detection_method = (
                f"{center_detection_method} | scale:{scale_source}"
            )

            # Boundary extraction
            boundary_method = analysis.get("boundary_method", "color")
            lower_hsv = analysis.get("lower_hsv", self.DEFAULT_HSV_LOWER_RANGE)
            upper_hsv = analysis.get("upper_hsv", self.DEFAULT_HSV_UPPER_RANGE)

            pdf_result = get_pdf_page_as_image(pdf_path, page_index=map_page)
            if not pdf_result.get("success"):
                return {
                    "success": False,
                    "error": pdf_result.get("error", "PDF conversion failed"),
                }
            img_b64 = pdf_result["image_base64"]

            if boundary_method == "edge":
                result = extract_region_boundary(img_b64)
            else:
                result = extract_color_boundary(img_b64, lower_hsv, upper_hsv)

            if not result.get("success"):
                return {
                    "success": False,
                    "error": result.get("error", "Boundary extraction failed"),
                    "processing_time": time.time() - start_time,
                }

            image_height = result["image_height"]
            image_width = result["image_width"]
            candidates = result["candidates"]
            if len(candidates) < 1:
                return {
                    "success": False,
                    "error": f"Expected 1 or more boundary candidates, got {len(candidates)}",
                    "processing_time": time.time() - start_time,
                }
            boundary_pixels = self._select_boundary_candidate(img_b64, candidates)

            print(f"Boundary: {len(boundary_pixels)} points extracted")

            # Create center tuple for refinement step
            center = (lat_center, lon_center)

            # Transform to geographic coordinates using tools package
            transform_result = pixels_to_geo_linear(
                boundary_pixels=boundary_pixels,
                image_height=image_height,
                image_width=image_width,
                center_lat=lat_center,
                center_lon=lon_center,
                scale_meters=scale_m,
            )

            if not transform_result.get("success"):
                return {
                    "success": False,
                    "error": transform_result.get("error", "Transform failed"),
                    "processing_time": time.time() - start_time,
                }

            geojson = transform_result["geojson"]

            # Ensure MultiPolygon format for consistency
            geom = geojson.get("geometry", {})
            if geom.get("type") == "Polygon":
                geojson["geometry"] = {
                    "type": "MultiPolygon",
                    "coordinates": [geom["coordinates"]],
                }

            # Add center detection method (not in tools output)
            geojson["properties"]["center_detection_method"] = center_detection_method

            # Optional iterative refinement
            refinement_log = []
            if iterative_refinement:
                geojson, refinement_log = self._refine_placement_with_llm(
                    map_image_b64=img_b64,
                    current_geojson=geojson,
                    current_center=center,
                    current_scale=scale_m,
                    boundary_pixels=boundary_pixels,
                    image_height=image_height,
                    image_width=image_width,
                    max_iterations=max_refinement_iterations,
                )

            print(f"Output: linear_transform, time={time.time() - start_time:.1f}s")
            return {
                "success": True,
                "parsed_json": geojson,
                "method": "linear_transform",
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    # =========================================================================
    # Method 3: Tool-Based Agentic Approach
    # LLM decides which tools to use with full control over the extraction process
    # Uses ALL_TOOLS imported from tools package at top of file
    # =========================================================================

    def _execute_agentic_tool(
        self, tool_name: str, arguments: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool call from the agent.

        Dispatches to the appropriate function from the tools package (imported at top of file).
        Stores image data in context for convenience (so boundary tools
        can automatically use the last converted PDF page).
        """
        # PDF Tools
        if tool_name == "get_pdf_page_as_image":
            # Always use context path - LLM doesn't know the actual file path
            pdf_path = context.get("pdf_path")
            if not pdf_path:
                return {"success": False, "error": "Missing pdf_path in context"}
            result = get_pdf_page_as_image(
                pdf_path=pdf_path,
                page_index=arguments.get("page_index", 0),
                dpi=arguments.get("dpi", 200),
            )
            # Store image in context for boundary tools
            if result.get("success"):
                context["last_image_base64"] = result["image_base64"]
                context["last_image_width"] = result["image_width"]
                context["last_image_height"] = result["image_height"]
            return result

        # Boundary Tools
        elif tool_name == "extract_color_boundary":
            image_b64 = arguments.get("image_base64") or context.get(
                "last_image_base64"
            )
            if not image_b64:
                return {
                    "success": False,
                    "error": "Missing image_base64. Call get_pdf_page_as_image first.",
                }
            if "lower_hsv" not in arguments:
                return {
                    "success": False,
                    "error": "Missing required argument: lower_hsv",
                }
            if "upper_hsv" not in arguments:
                return {
                    "success": False,
                    "error": "Missing required argument: upper_hsv",
                }
            return extract_color_boundary(
                image_b64,
                arguments["lower_hsv"],
                arguments["upper_hsv"],
            )

        elif tool_name == "extract_region_boundary":
            image_b64 = arguments.get("image_base64") or context.get(
                "last_image_base64"
            )
            if not image_b64:
                return {
                    "success": False,
                    "error": "Missing image_base64. Call get_pdf_page_as_image first.",
                }
            return extract_region_boundary(image_b64)

        # Geo Tools
        elif tool_name == "geocode_address":
            if "address" not in arguments:
                return {"success": False, "error": "Missing required argument: address"}
            return geocode_address(arguments["address"])

        elif tool_name == "lookup_district_boundary":
            if "district_name" not in arguments:
                return {
                    "success": False,
                    "error": "Missing required argument: district_name",
                }
            return lookup_district_boundary(
                arguments["district_name"],
                arguments.get("include_parent", True),
            )

        elif tool_name == "pixels_to_geo_linear":
            required = [
                "boundary_pixels",
                "image_height",
                "image_width",
                "center_lat",
                "center_lon",
                "scale_meters",
            ]
            missing = [k for k in required if k not in arguments]
            if missing:
                return {
                    "success": False,
                    "error": f"Missing required arguments: {missing}",
                }
            return pixels_to_geo_linear(
                arguments["boundary_pixels"],
                arguments["image_height"],
                arguments["image_width"],
                arguments["center_lat"],
                arguments["center_lon"],
                arguments["scale_meters"],
            )

        # Visualization Tools
        elif tool_name == "visualize_geojson_boundary":
            if "geojson_data" not in arguments:
                return {
                    "success": False,
                    "error": "Missing required argument: geojson_data",
                }
            return visualize_geojson_boundary(
                arguments["geojson_data"],
                arguments.get("padding", 1.0),
            )

        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

    def _get_agentic_system_prompt(self) -> str:
        """
        Get the system prompt for the agentic extractor.
        Documents all tools available from the tools package.
        """
        return """You are an expert GIS AI agent specialized in extracting geographic boundaries from UK planning documents (Article 4 directions, conservation areas, etc.).

                Your task is to extract the planning area boundary as GeoJSON coordinates.

                ═══════════════════════════════════════════════════════════════════════════════
                AVAILABLE TOOLS
                ═══════════════════════════════════════════════════════════════════════════════

                PDF TOOLS:
                ──────────
                1. get_pdf_page_as_image
                Convert a PDF page to a base64-encoded image.
                PARAMS: pdf_path (optional, uses context), page_index (0-indexed), dpi (default 200)
                OUTPUT: image_base64, image_width, image_height, total_pages

                BOUNDARY EXTRACTION TOOLS:
                ──────────────────────────
                2. extract_color_boundary
                Extract boundary pixels using HSV color filtering.
                USE WHEN: Boundary is a colored line/region (red, orange, blue, pink, etc.)
                PARAMS: image_base64, lower_hsv [H,S,V], upper_hsv [H,S,V]

                HSV COLOR GUIDE (OpenCV: H=0-179, S=0-255, V=0-255):
                • Red (low hue):   lower=[0, 70, 150],   upper=[10, 255, 255]
                • Red (high hue):  lower=[170, 70, 150], upper=[179, 255, 255]
                • Orange:          lower=[5, 80, 80],    upper=[25, 255, 255]
                • Blue:            lower=[90, 70, 150],  upper=[130, 255, 255]
                • Green:           lower=[35, 70, 150],  upper=[85, 255, 255]
                • Pink/Magenta:    lower=[140, 30, 150], upper=[175, 150, 255]

                OUTPUT: boundary_pixels, image_height, image_width, num_points

                3. extract_region_boundary
                Extract boundary using grayscale edge detection.
                USE WHEN: Boundary is BLACK or dark colored, or color extraction fails.
                PARAMS: image_base64 (auto-uses last PDF page)
                OUTPUT: boundary_pixels, image_height, image_width, num_points

                GEO TOOLS:
                ──────────
                4. geocode_address
                Convert an address or place name to lat/lon coordinates.
                USE TO: Find the center coordinates for linear transformation.
                PARAMS: address (e.g., "Chelsea Embankment, London, UK")
                OUTPUT: latitude, longitude, display_name

                5. lookup_district_boundary
                Look up official boundary from OpenStreetMap for an entire district.
                USE WHEN: Planning area covers a whole borough/ward/parish.
                PARAMS: district_name (e.g., "London Borough of Barnet, London, UK")
                OUTPUT: geojson, coordinates, bbox

                6. pixels_to_geo_linear
                Transform pixel coordinates to geographic coordinates.
                USE AFTER: Extracting boundary pixels.
                PARAMS:
                • boundary_pixels: From extract_color_boundary or extract_region_boundary
                • image_height, image_width: From boundary extraction result
                • center_lat, center_lon: From geocode_address
                • scale_meters: Real-world width the map covers

                SCALE GUIDE:
                • 1:1250 on A4 → ~262m width
                • 1:2500 on A4 → ~525m width
                • 1:5000 on A4 → ~1050m width

                OUTPUT: geojson, coordinates, bbox

                VISUALIZATION:
                ──────────────
                7. visualize_geojson_boundary
                Render GeoJSON boundary on OpenStreetMap basemap.
                USE TO: Verify alignment with real-world features.
                PARAMS: geojson_data, padding (default 1.0)
                OUTPUT: image_base64, bbox

                ═══════════════════════════════════════════════════════════════════════════════
                WORKFLOW
                ═══════════════════════════════════════════════════════════════════════════════

                APPROACH A: Boundary Extraction (most common)
                ─────────────────────────────────────────────
                1. Examine document, find map page, identify boundary color and scale
                2. get_pdf_page_as_image → get the map page
                3. extract_color_boundary OR extract_region_boundary → get boundary pixels
                4. geocode_address → get center coordinates
                5. pixels_to_geo_linear → transform to GeoJSON
                6. visualize_geojson_boundary → verify alignment
                7. If misaligned, adjust center/scale and repeat 4-6

                APPROACH B: District Lookup (when area = entire district)
                ─────────────────────────────────────────────────────────
                1. If document covers entire borough/ward/parish
                2. lookup_district_boundary → get official OSM boundary
                3. visualize_geojson_boundary → verify

                ═══════════════════════════════════════════════════════════════════════════════
                IMPORTANT NOTES
                ═══════════════════════════════════════════════════════════════════════════════

                • GeoJSON coordinate order: [longitude, latitude] not (lat, lon)
                • Image coordinates: Y increases downward; geo Y increases northward
                • The boundary tools auto-use the last PDF page image from context
                • Always visualize to verify before returning final result

                When done, return the final GeoJSON."""

    def extract_geojson_agentic(
        self, pdf_path: str, max_iterations: int = 7
    ) -> Dict[str, Any]:
        """
        Extract GeoJSON using a fully agentic approach where the LLM has complete
        control over the extraction pipeline.

        The LLM agent receives:
        - The PDF document
        - A prompt explaining the task
        - All tools from tools package:
          PDF: get_pdf_page_as_image
          Boundary: extract_color_boundary, extract_region_boundary
          Geo: pixels_to_geo_linear, lookup_district_boundary, geocode_address
          Visualization: visualize_geojson_boundary

        The agent decides the best approach:
        - Boundary extraction + linear transformation for site-specific areas
        - District lookup for areas covering entire boroughs/wards
        """
        start_time = time.time()
        iterations = []
        context = {"pdf_path": pdf_path}

        try:
            # Read PDF and encode as base64
            with open(pdf_path, "rb") as f:
                pdf_data = base64.b64encode(f.read()).decode("utf-8")
            pdf_data_uri = f"data:application/pdf;base64,{pdf_data}"

            system_prompt = self._get_agentic_system_prompt()

            # Build user content with PDF
            user_content = [
                {
                    "type": "text",
                    "text": """Analyze this UK planning document and extract the planning area boundary as GeoJSON.

                                YOUR TASK:
                                1. Find the page containing the planning map with the boundary
                                2. Use get_pdf_page_as_image to get the map page as base64
                                3. Identify the boundary color and determine appropriate HSV values (if deciding to use extract_color_boundary in step 4)
                                4. Extract the boundary pixels using extract_color_boundary (for colored boundaries) or extract_region_boundary (for black/dark boundaries)
                                5. Identify a place name at the center of the map and geocode it
                                6. Determine the scale from the scale bar or ratio on the map
                                7. Use pixels_to_geo_linear to transform pixel coordinates to geographic coordinates
                                8. Visualize to verify, refine if needed
                                9. Return the final GeoJSON

                                ALTERNATIVE: If the document covers an entire district/borough, use lookup_district_boundary instead of boundary extraction.

                                Now analyze the document and extract the planning boundary as GeoJSON using the tools provided.""",
                },
                {
                    "type": "file",
                    "file": {
                        "filename": os.path.basename(pdf_path),
                        "file_data": pdf_data_uri,
                    },
                },
            ]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            final_geojson = None

            # Agentic loop - allow multiple tool calls
            for iteration in range(max_iterations):
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "tools": ALL_TOOLS,
                    "tool_choice": "auto",
                    "max_tokens": 8192,
                }

                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=300,
                )
                response.raise_for_status()
                data = response.json()

                message = data["choices"][0]["message"]
                messages.append(message)  # Add assistant response to history

                # Check if LLM wants to call tools
                if message.get("tool_calls"):
                    tool_results = []

                    for tool_call in message["tool_calls"]:
                        tool_name = tool_call["function"]["name"]
                        tool_args_str = tool_call["function"]["arguments"]
                        try:
                            tool_args = (
                                json.loads(tool_args_str) if tool_args_str else {}
                            )
                        except json.JSONDecodeError:
                            tool_args = {}

                        iterations.append(
                            {
                                "iteration": iteration,
                                "tool": tool_name,
                                "arguments": {
                                    k: v[:100] + "..."
                                    if isinstance(v, str) and len(v) > 100
                                    else v
                                    for k, v in tool_args.items()
                                },
                            }
                        )

                        # Execute the tool
                        result = self._execute_agentic_tool(
                            tool_name, tool_args, context
                        )

                        # Store successful geojson results
                        if result.get("success") and result.get("geojson"):
                            final_geojson = result["geojson"]
                            context["last_geojson"] = final_geojson

                        # Add tool result to messages
                        tool_results.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": json.dumps(result, default=str)[
                                    :10000
                                ],  # Truncate large results
                            }
                        )

                    messages.extend(tool_results)

                else:
                    # No tool calls - LLM is done or providing final answer
                    content = message.get("content", "")

                    # Try to extract GeoJSON from the response
                    parsed = self._parse_json_from_response(content)
                    if parsed:
                        final_geojson = parsed

                    # Check if we have a valid result
                    if final_geojson:
                        # Ensure MultiPolygon format
                        geom = final_geojson.get("geometry", {})
                        if geom.get("type") == "Polygon":
                            final_geojson["geometry"] = {
                                "type": "MultiPolygon",
                                "coordinates": [geom["coordinates"]],
                            }

                        return {
                            "success": True,
                            "parsed_json": final_geojson,
                            "method": "agentic",
                            "agent_iterations": iterations,
                            "num_iterations": iteration + 1,
                            "processing_time": time.time() - start_time,
                        }

                    # If no geojson and no tool calls, we're stuck
                    break

            # If we exhausted iterations without a result
            if final_geojson:
                geom = final_geojson.get("geometry", {})
                if geom.get("type") == "Polygon":
                    final_geojson["geometry"] = {
                        "type": "MultiPolygon",
                        "coordinates": [geom["coordinates"]],
                    }

                return {
                    "success": True,
                    "parsed_json": final_geojson,
                    "method": "agentic",
                    "agent_iterations": iterations,
                    "num_iterations": max_iterations,
                    "processing_time": time.time() - start_time,
                }
            else:
                return {
                    "success": False,
                    "error": "No GeoJSON found",
                    "method": "agentic",
                    "agent_iterations": iterations,
                    "processing_time": time.time() - start_time,
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "agentic",
                "agent_iterations": iterations,
                "processing_time": time.time() - start_time,
            }

    # Unified Extraction Method

    def extract_geojson(
        self,
        pdf_path: str,
        method: str = "baseline",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Unified method to extract GeoJSON from a planning PDF.

        Args:
            pdf_path: Path to the planning document PDF
            method: Extraction method to use:
                - "baseline": Direct LLM extraction (default)
                - "linear_transform": CV boundary detection + center/scale estimation
                - "agentic": LLM chooses which tools to use
            **kwargs: Additional arguments passed to the specific method

        Returns:
            Dict with 'success', 'parsed_json' (the GeoJSON), and metadata
        """
        methods = {
            "baseline": self.extract_geojson_from_pdf,
            "linear_transform": self.extract_geojson_linear_transform,
            "agentic": self.extract_geojson_agentic,
        }

        if method not in methods:
            return {
                "success": False,
                "error": f"Unknown method: {method}. Available: {list(methods.keys())}",
            }

        return methods[method](pdf_path, **kwargs)


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract GeoJSON from UK planning PDFs"
    )
    parser.add_argument("pdf_path", help="Path to the planning document PDF")
    parser.add_argument(
        "--method",
        choices=["baseline", "linear_transform", "agentic"],
        default="linear_transform",
        help="Extraction method to use (default: linear_transform)",
    )
    parser.add_argument(
        "--model",
        default="claude-opus",
        help="LLM model to use (default: claude-opus)",
    )
    parser.add_argument(
        "--output",
        default="output.geojson",
        help="Output file path (default: output.geojson)",
    )
    parser.add_argument(
        "--iterative",
        action="store_true",
        help="Enable iterative refinement for linear_transform method (uses visual feedback to adjust placement)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum iterations for iterative refinement (default: 5)",
    )

    args = parser.parse_args()

    client = OpenRouterClient(model=args.model)

    print(f"Extracting: {args.pdf_path} (method={args.method})")

    # Build kwargs for the extraction method
    extraction_kwargs = {}
    if args.iterative:
        extraction_kwargs["iterative_refinement"] = True
        extraction_kwargs["max_refinement_iterations"] = args.max_iterations

    response = client.extract_geojson(
        args.pdf_path, method=args.method, **extraction_kwargs
    )

    if response.get("success") and response.get("parsed_json"):
        geojson = response["parsed_json"]
        with open(args.output, "w") as f:
            json.dump(geojson, f, indent=2)
        print(f"Success: {args.output} ({response.get('processing_time', 0):.1f}s)")
    else:
        print(f"Failed: {response.get('error')}")
