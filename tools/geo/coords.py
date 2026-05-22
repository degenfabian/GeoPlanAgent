"""Shared Web-Mercator and BNG <-> WGS84 coordinate utilities.

Extracted 2026-05-11 to consolidate the Web-Mercator math that had drifted
across `tools/matching.py`, `tools/agent.py`, and `tools/os_opendata_tiles.py`.
The same `156543.03 * cos(lat) / 2**zoom` formula (and the symmetric
lat/lon -> tile-pixel projection) was duplicated 6 times.

All callers should import from this module. `tools/matching.py` re-exports
the names so existing `from tools.matching import compute_map_mpp` imports
keep working.

References:
- https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
- https://en.wikipedia.org/wiki/Web_Mercator_projection
"""

from __future__ import annotations

import math
from typing import Tuple

# Earth circumference (m) at the equator divided by 256 (the tile pixel size)
# = 2 * pi * 6378137 / 256 ≈ 156543.0339.
# This is the ground distance covered by one tile pixel at zoom 0 at the
# equator. Multiply by cos(lat) to account for the latitude foreshortening
# in Web Mercator, and divide by 2**zoom to get the pixel scale at any zoom.
WEB_MERCATOR_C: float = 156543.03


# Mean Earth radius (km); WGS84 conventional spherical-Earth value used
# everywhere in the repo. 6371.0 km is the standard "spherical Earth"
# approximation — accurate to ~0.3% vs the proper ellipsoid for UK-scale
# distances.
_EARTH_R_KM = 6371.0


def haversine_km(lat1: float, lon1: float,
                  lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two (lat, lon) points.

    Standard haversine formula on a spherical Earth (R = 6371 km). The
    inner ``min(1.0, ...)`` guards against floating-point noise pushing
    the argument of ``asin`` above 1.0 for near-coincident points.

    The repo previously also exposed a separate ``haversine_m`` flat-
    earth approximation (kept "for speed in hot paths"), but the
    callers were dead imports or sub-percent-precision use cases. One
    true function for everyone now; multiply by 1000 if you want metres.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2.0 * _EARTH_R_KM * math.asin(min(1.0, math.sqrt(a)))


def tile_mpp(lat: float, zoom: int) -> float:
    """Meters per pixel of a Web Mercator tile at the given latitude and zoom.

    Args:
        lat: Latitude in degrees (WGS84).
        zoom: Tile zoom level (typically 15-19 for UK planning maps).

    Returns:
        Ground meters per tile pixel.
    """
    return WEB_MERCATOR_C * math.cos(math.radians(lat)) / (2 ** zoom)


def compute_map_mpp(scale_ratio, dpi: int = 200):
    """Compute meters per pixel for a printed map at given scale ratio and DPI.

    At 1:2500 and 200 DPI: 1 pixel = 25.4/200 mm = 0.127 mm on paper.
    Ground meters per pixel = (25.4 / dpi) / 1000 * scale_ratio.

    Args:
        scale_ratio: Map scale denominator (e.g., 2500 for 1:2500). None -> None.
        dpi: Render DPI of the PDF page.

    Returns:
        Meters per pixel of the rendered map, or None if scale_ratio is None.
    """
    if scale_ratio is None:
        return None
    mm_per_px = 25.4 / dpi
    return mm_per_px / 1000.0 * scale_ratio


def best_zoom_for_scale(map_mpp, lat: float):
    """Find the OSM tile zoom level whose pixel scale most closely matches
    the printed map's pixel scale at the given latitude.

    Args:
        map_mpp: Meters per pixel of the rendered map (`compute_map_mpp`).
                 None -> None.
        lat: Latitude in degrees (WGS84).

    Returns:
        Integer zoom level in [15, 19], or None if map_mpp is None.
    """
    if map_mpp is None:
        return None
    z = math.log2(WEB_MERCATOR_C * math.cos(math.radians(lat)) / map_mpp)
    return max(15, min(19, round(z)))


def latlon_to_global_tile_pixel(
    lat: float, lon: float, zoom: int, tile_size: int = 256,
) -> Tuple[float, float]:
    """Project WGS84 lat/lon to global Web-Mercator tile-pixel coordinates.

    The origin is the top-left of the tile grid for the given zoom level.
    The result spans [0, 2**zoom * tile_size) in both x (lon) and y (lat).

    Args:
        lat, lon: WGS84 coordinates in degrees.
        zoom: Tile zoom level.
        tile_size: Tile size in pixels (default 256).

    Returns:
        (px, py) global tile pixel coordinates as floats.
    """
    n = 2 ** zoom
    lat_rad = math.radians(lat)
    px = (lon + 180.0) / 360.0 * n * tile_size
    py = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad))
          / math.pi) / 2.0 * n * tile_size
    return px, py


def osm_pixel_to_latlon(
    px: float, py: float, zoom: int, tx_min: int, ty_min: int,
    tile_size: int = 256,
) -> Tuple[float, float]:
    """Inverse of `latlon_to_global_tile_pixel`, but with a tile-canvas offset.

    Convert a pixel position on a tile canvas back to WGS84 lat/lon. The
    canvas is assumed to start at tile (tx_min, ty_min) at the given zoom.

    Args:
        px, py: Pixel coordinates on the tile canvas.
        zoom: Tile zoom level.
        tx_min, ty_min: Tile indices of the canvas origin (top-left).
        tile_size: Tile size in pixels (default 256).

    Returns:
        (lat, lon) in WGS84 degrees.
    """
    n = 2 ** zoom
    global_px = tx_min * tile_size + px
    global_py = ty_min * tile_size + py
    lon = global_px / (n * tile_size) * 360 - 180
    lat = math.degrees(math.atan(math.sinh(
        math.pi * (1 - 2 * global_py / (n * tile_size)))))
    return lat, lon


def latlon_to_tile_xy(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """Convert lat/lon to integer tile (x, y) indices at the given zoom.

    Convenience wrapper that returns integer tile coordinates rather than
    sub-pixel canvas coordinates. Used by tile fetchers.
    """
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad))
            / math.pi) / 2.0 * n)
    return x, y
