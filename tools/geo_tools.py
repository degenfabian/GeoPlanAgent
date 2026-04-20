"""
Geographic Transformation Tools for Planning Document Digitization

This module provides tools for converting pixel coordinates from planning maps
to geographic coordinates (WGS84 latitude/longitude) and for looking up
administrative boundaries.

LINEAR TRANSFORMATION: Uses center point + scale to transform coordinates.
Best when you know the map center location and scale ratio.

Additionally, a DISTRICT LOOKUP tool is provided for when the planning area
corresponds to an entire administrative district.
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import osmnx as ox
from pyproj import Transformer

# Average meters per degree of latitude
METERS_PER_DEGREE_LAT = 111111.0

# ── OS Grid Reference → WGS84 ───────────────────────────────────────────────

# OS National Grid: 2-letter prefix → (easting, northing) base in metres.
# Standard formula: each letter is 0-24 (A-Z skipping I).
_OS_GRID_LETTERS = {}
for _c1 in range(26):
    if _c1 == 8: continue  # skip I
    _l1 = _c1 - (1 if _c1 > 8 else 0)  # 0-24 index
    for _c2 in range(26):
        if _c2 == 8: continue
        _l2 = _c2 - (1 if _c2 > 8 else 0)
        e = ((_l1 - 2) % 5) * 5 + (_l2 % 5)
        n = 19 - 5 * (_l1 // 5) - (_l2 // 5)
        if 0 <= e <= 9 and 0 <= n <= 24:  # valid GB range
            _OS_GRID_LETTERS[chr(_c1 + 65) + chr(_c2 + 65)] = (e * 100000, n * 100000)

_OSGB_TO_WGS84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)


def os_grid_ref_to_latlon(grid_ref: str) -> Optional[Tuple[float, float]]:
    """Convert OS grid reference string to (lat, lon) WGS84.

    Accepts formats like "TG 210 080", "TG 2105 0803", "TG2108",
    "TG 21 08", "TR 206 48" (asymmetric), with or without spaces.
    Strips trailing compass directions like "SE", "NW" etc.

    Returns (lat, lon) or None if parsing fails.
    """
    s = grid_ref.strip().upper().replace(",", "").replace("  ", " ")
    # Strip trailing compass directions (e.g., "TR 34 SE" → "TR 34")
    s = re.sub(r"\s+[NSEW]{1,2}$", "", s)

    # Reject range-style refs (e.g., "TR 2562-63", "TG 20-22 08-10")
    if "-" in s:
        return None

    # Try 2-part format: "TG 210 080" or "TG210 080"
    m = re.match(r"([A-Z]{2})\s*(\d+)\s+(\d+)$", s)
    if not m:
        # Try compact form (with or without space): "TG210080" or "TG2108" or "TG 2638"
        m = re.match(r"([A-Z]{2})\s*(\d+)$", s)
        if not m:
            return None
        letters, digits = m.group(1), m.group(2)
        if len(digits) % 2 != 0:
            # Odd number of digits — try dropping last digit
            digits = digits[:-1]
        if len(digits) < 2:
            return None
        half = len(digits) // 2
        east_digits, north_digits = digits[:half], digits[half:]
    else:
        letters = m.group(1)
        east_digits, north_digits = m.group(2), m.group(3)

    if letters not in _OS_GRID_LETTERS:
        return None

    # Reject low-resolution refs: need at least 4 total digits (2+2 = 1km)
    total_digits = len(east_digits) + len(north_digits)
    if total_digits < 4:
        return None

    # Handle asymmetric digit counts by padding shorter to match longer
    max_len = max(len(east_digits), len(north_digits))
    east_digits = east_digits.ljust(max_len, "0")
    north_digits = north_digits.ljust(max_len, "0")

    # Pad to 5 digits (1m resolution)
    east_digits = east_digits.ljust(5, "0")
    north_digits = north_digits.ljust(5, "0")

    base_e, base_n = _OS_GRID_LETTERS[letters]
    easting = base_e + int(east_digits)
    northing = base_n + int(north_digits)

    # Convert OSGB36 → WGS84
    lon, lat = _OSGB_TO_WGS84.transform(easting, northing)

    # Validate result is within UK bounding box (49-61°N, -8.5-2°E)
    if not (49.0 <= lat <= 61.0 and -8.5 <= lon <= 2.0):
        return None

    return lat, lon

# Minimum points required to form a valid polygon
MIN_POLYGON_POINTS = 3


def pixels_to_geo_linear(
    boundary_pixels: List[List[float]],
    image_height: int,
    image_width: int,
    center_lat: float,
    center_lon: float,
    scale_meters: float,
) -> Dict[str, Any]:
    """
    Transform pixel coordinates to geographic coordinates using linear transformation.

    This tool converts boundary pixel coordinates to WGS84 (latitude/longitude)
    coordinates using a simple linear transformation based on a known center
    point and scale.

    HOW IT WORKS:
    1. Calculates meters-per-pixel ratio based on scale and image width
    2. Converts pixel offsets from image center to meter offsets
    3. Converts meter offsets to lat/lon offsets using approximate conversion
    4. Applies the offsets to the center coordinates

    WHEN TO USE:
    - When you know (or can estimate) the center location of the map
    - When you know (or can read from the map) the scale ratio
    - When the map has minimal rotation/distortion
    - Quick transformation when high precision isn't critical

    COORDINATE SYSTEM NOTES:
    - Pixel coordinates: Origin at top-left, Y increases downward
    - Geographic coordinates: Latitude increases northward, longitude increases eastward
    - This function handles the Y-axis flip automatically

    SCALE INTERPRETATION:
    - scale_meters is the real-world width covered by the map
    - For a 1:2500 scale map on A4 paper (210mm wide): 210mm × 2500 / 1000 = 525m
    - For a 1:1250 scale map on A4 paper: 210mm × 1250 / 1000 = 262.5m

    Args:
        boundary_pixels (List[List[float]]):
            List of [x, y] pixel coordinates forming the boundary polygon.
            These should come from extract_boundary_with_hsv or similar.

        image_height (int):
            Height of the source image in pixels. Used for Y-axis flip.

        image_width (int):
            Width of the source image in pixels. Used to calculate scale.

        center_lat (float):
            Latitude of the map's center point in decimal degrees.
            Example: 51.5074 for central London

        center_lon (float):
            Longitude of the map's center point in decimal degrees.
            Example: -0.1278 for central London

        scale_meters (float):
            The real-world width that the map represents, in meters.
            Example: 500.0 for a map covering 500 meters width

    Returns:
        Dict containing:
        - "success" (bool): Whether transformation succeeded
        - "geojson" (Dict): Complete GeoJSON Feature object with the boundary
        - "coordinates" (List): List of [lon, lat] coordinate pairs
        - "bbox" (Dict): Bounding box with min/max lat/lon
        - "transformation_info" (Dict): Details about the transformation applied

    EXAMPLE:
        result = pixels_to_geo_linear(
            boundary_pixels=[[100, 100], [200, 100], [200, 200], [100, 200]],
            image_height=1000,
            image_width=800,
            center_lat=51.5074,
            center_lon=-0.1278,
            scale_meters=500.0
        )
    """
    try:
        if not boundary_pixels or len(boundary_pixels) < MIN_POLYGON_POINTS:
            return {
                "success": False,
                "error": f"Need at least {MIN_POLYGON_POINTS} boundary points to create a polygon",
            }

        # Calculate meters per pixel
        meters_per_pixel = scale_meters / image_width

        # Image center in pixels
        center_x = image_width / 2
        center_y = image_height / 2

        # Longitude degrees vary with latitude: 1° lon ≈ METERS_PER_DEGREE_LAT * cos(lat)
        lat_rad = np.radians(center_lat)
        meters_per_degree_lon = METERS_PER_DEGREE_LAT * np.cos(lat_rad)

        # Transform each point
        geo_coords = []
        for px, py in boundary_pixels:
            # Flip Y-axis (image Y increases down, geo Y increases up)
            py_flipped = image_height - py

            # Calculate offset from center in pixels
            dx_pixels = px - center_x
            dy_pixels = py_flipped - center_y

            # Convert to meters
            dx_meters = dx_pixels * meters_per_pixel
            dy_meters = dy_pixels * meters_per_pixel

            # Convert to degrees
            d_lon = dx_meters / meters_per_degree_lon
            d_lat = dy_meters / METERS_PER_DEGREE_LAT

            # Apply to center
            lon = center_lon + d_lon
            lat = center_lat + d_lat

            geo_coords.append([lon, lat])

        # Close the polygon if not already closed
        if geo_coords[0] != geo_coords[-1]:
            geo_coords.append(geo_coords[0])

        # Calculate bounding box
        lons = [c[0] for c in geo_coords]
        lats = [c[1] for c in geo_coords]

        # Create GeoJSON
        geojson = {
            "type": "Feature",
            "properties": {
                "transformation_method": "linear",
                "center_lat": center_lat,
                "center_lon": center_lon,
                "scale_meters": scale_meters,
                "source": "planning_document_extraction",
            },
            "geometry": {"type": "Polygon", "coordinates": [geo_coords]},
        }

        return {
            "success": True,
            "geojson": geojson,
            "coordinates": geo_coords,
            "bbox": {
                "min_lon": min(lons),
                "max_lon": max(lons),
                "min_lat": min(lats),
                "max_lat": max(lats),
            },
            "transformation_info": {
                "method": "linear",
                "meters_per_pixel": meters_per_pixel,
                "image_dimensions": [image_width, image_height],
                "center": [center_lon, center_lat],
                "scale_meters": scale_meters,
            },
        }

    except Exception as e:
        return {"success": False, "error": f"Linear transformation failed: {str(e)}"}


def lookup_district_boundary(
    district_name: str,
) -> Dict[str, Any]:
    """
    Look up the boundary of an administrative district from OpenStreetMap.

    This tool retrieves the official boundary polygon for a named administrative
    area (borough, ward, parish, etc.) from OpenStreetMap data.

    WHEN TO USE:
    - When the planning document covers an entire administrative district
    - When you can identify the district name from the document
    - As a fallback when boundary extraction from the map image fails
    - To verify extracted boundaries against known district shapes

    NAMING CONVENTIONS:
    - UK Borough: "Royal Borough of Kensington and Chelsea, London"
    - UK Ward: "Rowley Green, London Borough of Barnet, London"
    - Include parent areas for disambiguation: "Kensington, London, UK"

    Args:
        district_name (str):
            The full name of the administrative district.
            Be as specific as possible to avoid ambiguous matches.
            Examples:
            - "Royal Borough of Kensington and Chelsea, London, UK"
            - "Rowley Green, London Borough of Barnet, London, UK"
            - "City of Westminster, London, UK"

    Returns:
        Dict containing:
        - "success" (bool): Whether lookup succeeded
        - "geojson" (Dict): GeoJSON Feature with the boundary
        - "coordinates" (List): Boundary coordinates
        - "bbox" (Dict): Bounding box
        - "osm_info" (Dict): OpenStreetMap metadata about the area
        - "error" (str): Error message if lookup failed

    LIMITATIONS:
    - Requires internet connection
    - OSM data quality varies by region
    - Some small areas may not have boundary data
    - Rate-limited by Nominatim usage policy
    """
    try:
        # Query OSM for the district boundary
        gdf = ox.geocode_to_gdf(district_name)

        if gdf.empty:
            return {
                "success": False,
                "error": f"No boundary found for: {district_name}",
            }

        # Convert to GeoJSON
        geojson_dict = json.loads(gdf.to_json())
        feature = geojson_dict["features"][0]

        # Extract geometry
        geometry = feature["geometry"]
        geom_type = geometry["type"]

        # Normalize to consistent coordinate format
        if geom_type == "Polygon" or geom_type == "MultiPolygon":
            coordinates = geometry["coordinates"]
        else:
            return {"success": False, "error": f"Unexpected geometry type: {geom_type}"}

        # Calculate bounding box
        all_coords = []
        if geom_type == "Polygon":
            for ring in coordinates:
                all_coords.extend(ring)
        else:  # MultiPolygon
            for polygon in coordinates:
                for ring in polygon:
                    all_coords.extend(ring)

        lons = [c[0] for c in all_coords]
        lats = [c[1] for c in all_coords]

        # Build result GeoJSON
        result_geojson = {
            "type": "Feature",
            "properties": {
                "source": "openstreetmap",
                "query": district_name,
                "osm_type": feature.get("properties", {}).get("osm_type", "unknown"),
                "display_name": feature.get("properties", {}).get(
                    "display_name", district_name
                ),
            },
            "geometry": geometry,
        }

        return {
            "success": True,
            "geojson": result_geojson,
            "coordinates": coordinates,
            "geometry_type": geom_type,
            "bbox": {
                "min_lon": min(lons),
                "max_lon": max(lons),
                "min_lat": min(lats),
                "max_lat": max(lats),
            },
            "osm_info": feature.get("properties", {}),
        }

    except Exception as e:
        return {"success": False, "error": f"District lookup failed: {str(e)}"}


def try_district_boundary(analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """If analysis says covers_district, look up boundary and normalize to MultiPolygon.

    Returns GeoJSON Feature dict on success, None otherwise.
    """
    if not (analysis.get("covers_district") and analysis.get("district_name")):
        return None
    lookup = lookup_district_boundary(analysis["district_name"])
    if not lookup.get("success"):
        return None
    geojson = lookup["geojson"]
    geom = geojson.get("geometry", {})
    if geom.get("type") == "Polygon":
        geojson["geometry"] = {
            "type": "MultiPolygon",
            "coordinates": [geom["coordinates"]],
        }
    geojson["properties"]["source"] = "osm_district_lookup"
    return geojson


def geocode_address(address: str) -> Dict[str, Any]:
    """
    Convert an address or place name to geographic coordinates.

    This tool uses Nominatim (OpenStreetMap's geocoding service) to find
    the latitude and longitude of a named location.

    WHEN TO USE:
    - To find the center coordinates for pixels_to_geo_linear
    - To verify locations mentioned in planning documents

    Args:
        address (str):
            The address or place name to geocode.
            Examples:
            - "Chelsea Embankment, London, UK"
            - "Notting Hill Gate Station, London"
            - "51.5074, -0.1278" (will parse as coordinates)

    Returns:
        Dict containing:
        - "success" (bool): Whether geocoding succeeded
        - "latitude" (float): Latitude in decimal degrees
        - "longitude" (float): Longitude in decimal degrees
        - "display_name" (str): Full name returned by geocoder
        - "address_components" (Dict): Parsed address components
    """
    geolocator = Nominatim(user_agent="planning_doc_extractor")
    last_error = None

    for attempt in range(3):
        if attempt > 0:
            wait = 2 ** attempt
            print(f"    WARN:Nominatim retry {attempt}/2 after {wait}s...")
            import time as _time
            _time.sleep(wait)
        try:
            location = geolocator.geocode(address, timeout=10)

            if location is None:
                return {"success": False, "error": f"Could not geocode: {address}"}

            return {
                "success": True,
                "latitude": location.latitude,
                "longitude": location.longitude,
                "display_name": location.address,
                "raw": location.raw,
            }

        except GeocoderTimedOut:
            last_error = "Geocoding request timed out"
            continue
        except Exception as e:
            return {"success": False, "error": f"Geocoding failed: {str(e)}"}

    print(f"    WARN:Nominatim: all retries failed for '{address[:40]}'")
    return {"success": False, "error": f"Nominatim timed out after 3 retries: {address}"}


# Tool definitions for LLM function calling
GEO_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "pixels_to_geo_linear",
            "description": """Transform pixel boundary coordinates to geographic coordinates using LINEAR transformation.

Uses center point + scale to transform. Best when:
- You know the map center location (from document text or geocoding a mentioned place)
- You know the scale (e.g., 1:2500 means ~525m width on A4 paper)
- The map has minimal rotation

SCALE EXAMPLES:
- 1:1250 on A4 → ~262m width
- 1:2500 on A4 → ~525m width
- 1:5000 on A4 → ~1050m width

Simple and effective for most planning documents.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "boundary_pixels": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "List of [x, y] pixel coordinates from boundary extraction",
                    },
                    "image_height": {
                        "type": "integer",
                        "description": "Image height in pixels (from boundary extraction result)",
                    },
                    "image_width": {
                        "type": "integer",
                        "description": "Image width in pixels (from boundary extraction result)",
                    },
                    "center_lat": {
                        "type": "number",
                        "description": "Latitude of map center in decimal degrees (e.g., 51.5074)",
                    },
                    "center_lon": {
                        "type": "number",
                        "description": "Longitude of map center in decimal degrees (e.g., -0.1278)",
                    },
                    "scale_meters": {
                        "type": "number",
                        "description": "Real-world width covered by the map in meters (e.g., 500.0)",
                    },
                },
                "required": [
                    "boundary_pixels",
                    "image_height",
                    "image_width",
                    "center_lat",
                    "center_lon",
                    "scale_meters",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_district_boundary",
            "description": """Look up the official boundary of an administrative district from OpenStreetMap.

Use when:
- The planning area is an entire district/ward/borough
- You can identify the district name from the document
- As verification for extracted boundaries
- As fallback when image extraction fails

NAMING TIPS:
- Be specific: "Royal Borough of Kensington and Chelsea, London, UK"
- Include parent areas: "Rowley Green, London Borough of Barnet, London"
- UK boroughs, wards, parishes are usually available""",
            "parameters": {
                "type": "object",
                "properties": {
                    "district_name": {
                        "type": "string",
                        "description": "Full name of the district (be specific to avoid ambiguity)",
                    },
                },
                "required": ["district_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "geocode_address",
            "description": """Convert an address or place name to latitude/longitude coordinates.

Use to:
- Find center coordinates for pixels_to_geo_linear
- Geocode landmarks for pixels_to_geo_affine control points
- Verify locations mentioned in documents

Returns latitude and longitude in decimal degrees.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {
                        "type": "string",
                        "description": "Address or place name to geocode (e.g., 'Chelsea Embankment, London, UK')",
                    }
                },
                "required": ["address"],
            },
        },
    },
]
