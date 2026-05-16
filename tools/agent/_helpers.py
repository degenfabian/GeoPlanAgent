"""Pure helpers shared across the agent tools.

Image conversion, mask overlay, dedup tracking — anything that doesn't
need to know about the Agent instance or pydantic-ai itself.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import cv2
import numpy as np
from pydantic_ai import BinaryContent, ModelRetry

if TYPE_CHECKING:
    from tools.agent.state import AgentState


def resize_for_api(img: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """Resize image so largest dimension is max_dim."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def _img_to_binary(img: np.ndarray) -> BinaryContent:
    """Convert numpy BGR image to PydanticAI BinaryContent."""
    _, buf = cv2.imencode('.png', resize_for_api(img))
    return BinaryContent(data=buf.tobytes(), media_type='image/png')


def _dedup_check(state: "AgentState", tool_name: str, args: dict) -> None:
    """Raise ModelRetry if this exact tool+args was already called."""
    key = tool_name + ":" + hashlib.md5(
        json.dumps(args, sort_keys=True, default=str).encode()
    ).hexdigest()
    if key in state.recent_calls:
        raise ModelRetry(
            "You already called this tool with the same arguments. "
            "Try different arguments or respond with DONE."
        )
    state.recent_calls.add(key)


def _create_boundary_overlay(map_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay boundary mask on map image (red tint, 40% opacity)."""
    overlay = map_img.copy()
    if mask is not None and mask.shape[:2] == map_img.shape[:2]:
        overlay[mask > 0] = [0, 0, 255]
    return cv2.addWeighted(map_img, 0.6, overlay, 0.4, 0)


def _draw_geojson_on_tiles(tile_bgr, geojson, tile_info):
    """Draw GeoJSON boundary outline on tile canvas."""
    geom = geojson.get("geometry", {})
    coord_rings = []
    if geom.get("type") == "Polygon":
        coord_rings = [geom["coordinates"][0]]
    elif geom.get("type") == "MultiPolygon":
        coord_rings = [poly[0] for poly in geom["coordinates"]]

    zoom = tile_info.get("zoom", 17)
    tx_min = tile_info.get("tx_min", 0)
    ty_min = tile_info.get("ty_min", 0)
    tile_size = tile_info.get("tile_size", 256)

    from tools.geo.coords import latlon_to_global_tile_pixel
    for ring in coord_rings:
        pts = []
        for lon_c, lat_c in ring:
            abs_px, abs_py = latlon_to_global_tile_pixel(
                lat_c, lon_c, zoom, tile_size)
            px = abs_px - tx_min * tile_size
            py = abs_py - ty_min * tile_size
            pts.append([int(px), int(py)])
        if len(pts) >= 3:
            cv2.polylines(tile_bgr, [np.array(pts, dtype=np.int32)],
                          True, (0, 0, 255), 2)
    return tile_bgr
