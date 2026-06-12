"""Web-Mercator and WGS84 coordinate utilities.

Refs:
- https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
- https://en.wikipedia.org/wiki/Web_Mercator_projection
"""

from __future__ import annotations

import math
from typing import Tuple

# Ground metres per zoom-0 tile pixel at the equator: 2π·6378137 / 256.
WEB_MERCATOR_C: float = 156543.03

# Spherical Earth (~0.3% off vs ellipsoid at UK scale).
_EARTH_R_KM = 6371.0


def haversine_km(lat1: float, lon1: float,
                  lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points (haversine, R=6371 km)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2.0 * _EARTH_R_KM * math.asin(min(1.0, math.sqrt(a)))


def tile_mpp(lat: float, zoom: int) -> float:
    """Ground metres per pixel for a Web-Mercator tile at (lat, zoom)."""
    return WEB_MERCATOR_C * math.cos(math.radians(lat)) / (2 ** zoom)


def compute_map_mpp(scale_ratio, dpi: int = 200):
    """Ground metres per pixel for a 1:scale_ratio map rendered at dpi. None passes through."""
    if scale_ratio is None:
        return None
    mm_per_px = 25.4 / dpi
    return mm_per_px / 1000.0 * scale_ratio


def best_zoom_for_scale(map_mpp, lat: float):
    """OSM zoom in [15, 19] whose pixel scale most closely matches map_mpp at lat."""
    if map_mpp is None:
        return None
    z = math.log2(WEB_MERCATOR_C * math.cos(math.radians(lat)) / map_mpp)
    return max(15, min(19, round(z)))


def latlon_to_global_tile_pixel(
    lat: float, lon: float, zoom: int, tile_size: int = 256,
) -> Tuple[float, float]:
    """WGS84 → global Web-Mercator tile-pixel (px, py). Origin = top-left of zoom grid."""
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
    """Inverse of latlon_to_global_tile_pixel, offset by canvas origin (tx_min, ty_min)."""
    n = 2 ** zoom
    global_px = tx_min * tile_size + px
    global_py = ty_min * tile_size + py
    lon = global_px / (n * tile_size) * 360 - 180
    lat = math.degrees(math.atan(math.sinh(
        math.pi * (1 - 2 * global_py / (n * tile_size)))))
    return lat, lon


def latlon_to_tile_xy(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """WGS84 → integer (tx, ty) tile indices at zoom."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad))
            / math.pi) / 2.0 * n)
    return x, y
