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


_EN_RE = re.compile(r"(\d{4,7})\s*E\s*(\d{4,7})\s*N", re.IGNORECASE)


def parse_easting_northing(text: str) -> Optional[Tuple[float, float]]:
    """Return (lat, lon) parsed from explicit OS easting/northing in metres.

    Accepts formats like "528942 E 184544 N" (typical OS grid coords printed
    on UK planning maps). This is the highest-precision anchor we ever get
    from a PDF — site centre to within ~1 m. Used as a high-confidence
    geocoder candidate via the locate sub-agent's `grid_ref` tool.
    """
    if not isinstance(text, str):
        return None
    m = _EN_RE.search(text)
    if not m:
        return None
    east, north = int(m.group(1)), int(m.group(2))
    lon, lat = _OSGB_TO_WGS84.transform(east, north)
    return float(lat), float(lon)


def os_grid_ref_to_latlon_coarse(grid_ref: str) -> Optional[Tuple[float, float]]:
    """Convert a low-resolution OS grid ref (10km or 5km square) to (lat, lon).

    Handles ``TR 34`` (10km tile centre) and ``TR 34 SE`` (south-east 5km
    quadrant centre). These are too imprecise for the strict
    ``os_grid_ref_to_latlon`` (which requires 1km resolution) but still
    make a useful coarse anchor when no better signal exists.

    Returns (lat, lon) at the CENTRE of the referenced tile / quadrant,
    or None if parsing fails.
    """
    if not grid_ref:
        return None
    s = grid_ref.strip().upper().replace(",", "").replace("  ", " ")

    # Optional trailing compass quadrant
    quad = None
    m = re.search(r"\s+(NE|NW|SE|SW)$", s)
    if m:
        quad = m.group(1)
        s = s[:m.start()]

    # Parse letters + 2 digits (10km tile)
    m = re.match(r"^([A-Z]{2})\s*(\d)\s*(\d)$", s) or \
        re.match(r"^([A-Z]{2})\s*(\d)(\d)$", s)
    if not m:
        return None
    letters, e_dig, n_dig = m.group(1), m.group(2), m.group(3)
    if letters not in _OS_GRID_LETTERS:
        return None

    base_e, base_n = _OS_GRID_LETTERS[letters]
    # 10km tile: digit × 10000 metres, centred at +5000 within the tile
    easting = base_e + int(e_dig) * 10_000
    northing = base_n + int(n_dig) * 10_000
    easting += 5_000
    northing += 5_000

    # Quadrant = 5km sub-square; offset from 10km centre by ±2500 m
    if quad == "NE":
        easting += 2_500;  northing += 2_500
    elif quad == "NW":
        easting -= 2_500;  northing += 2_500
    elif quad == "SE":
        easting += 2_500;  northing -= 2_500
    elif quad == "SW":
        easting -= 2_500;  northing -= 2_500

    try:
        lon, lat = _OSGB_TO_WGS84.transform(easting, northing)
    except Exception:
        return None
    if not (49.0 <= lat <= 61.0 and -8.5 <= lon <= 2.0):
        return None
    return lat, lon


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

    # Range-style refs like "TR3559-60" or "TR 2562-63" or "TG 20-22 08-10":
    # replace the hyphenated tail with just the lower bound so we still get
    # a usable (if slightly coarse) anchor. "TR3559-60" → "TR3559".
    if "-" in s:
        s = re.sub(r"(\d+)-\d+", r"\1", s)
        s = s.strip()

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

    # Pad to 5 digits (1m resolution). The first padding char is "5" so the
    # resolved metres land at the CENTROID of the precision-defined tile,
    # not its SW corner. Without this, a 4-digit ref like "TR 2048" (1km
    # tile) resolves to the SW corner — up to 1414m from a GT elsewhere
    # in the tile. Centroid bounds worst-case error at the half-diagonal
    # (~707m) and gives ~250-400m for typical GT-in-tile.
    def _centroid_pad(d: str) -> str:
        if len(d) >= 5:
            return d
        return d + "5" + "0" * (5 - len(d) - 1)
    east_digits = _centroid_pad(east_digits)
    north_digits = _centroid_pad(north_digits)

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
    Look up the boundary polygon of a UK administrative district from
    OS BoundaryLine (offline, OS Open Data).

    Resolves the name through tools.verification_checks._resolve_la, which
    normalises common UK admin variants (strips "District", "Borough",
    "London Borough of", "City of …", "Council", etc.) and tries all
    "|"-separated alternates in order until one resolves.

    Used by the worker's `lookup_district` tool for district-wide planning
    documents (entire LA / borough / ward covered).

    Args:
        district_name: e.g. "Camden, UK", "Royal Borough of Kensington
            and Chelsea, UK", "Broadland District, Norfolk, UK", or
            "City of Westminster, UK | Westminster, UK".

    Returns:
        Dict with:
          - success (bool)
          - geojson (Feature with MultiPolygon geometry; source =
            "os_boundaryline")
          - coordinates, geometry_type, bbox
          - resolved_variant (which "|" alternate matched)
          - error (on failure)
    """
    from shapely.geometry import mapping, MultiPolygon, Polygon
    from tools.verification_checks import _resolve_la

    variants = [v.strip() for v in district_name.split("|") if v.strip()]
    if not variants:
        variants = [district_name]

    poly = None
    resolved_variant = None
    for v in variants:
        try:
            poly = _resolve_la(v)
        except Exception:
            poly = None
        if poly is not None:
            resolved_variant = v
            break

    if poly is None:
        return {
            "success": False,
            "error": f"No OS BoundaryLine polygon found for: {district_name}",
        }

    # Normalise to MultiPolygon for downstream consistency
    if isinstance(poly, Polygon):
        poly = MultiPolygon([poly])
    elif not isinstance(poly, MultiPolygon):
        return {
            "success": False,
            "error": f"Unexpected geometry type from _resolve_la: "
                     f"{type(poly).__name__}",
        }

    geometry = mapping(poly)
    minx, miny, maxx, maxy = poly.bounds

    return {
        "success": True,
        "geojson": {
            "type": "Feature",
            "properties": {
                "source": "os_boundaryline",
                "query": district_name,
                "resolved_variant": resolved_variant,
            },
            "geometry": geometry,
        },
        "coordinates": geometry["coordinates"],
        "geometry_type": geometry["type"],
        "bbox": {
            "min_lon": float(minx),
            "max_lon": float(maxx),
            "min_lat": float(miny),
            "max_lat": float(maxy),
        },
        "resolved_variant": resolved_variant,
    }


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
    geojson["properties"]["source"] = "os_boundaryline_district_lookup"
    return geojson
