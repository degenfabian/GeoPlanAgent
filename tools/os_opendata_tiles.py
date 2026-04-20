"""
OS OpenData Tile Renderer
=========================
Render raster tiles from OS Open Zoomstack GeoPackage (free, OGL licensed).
Produces tiles styled to match UK planning map conventions — pink buildings,
road casings, water, woodland — so LoFTR/MINIMA can match scanned planning
maps against them with minimal cross-modal gap.

No API key required. Data: https://osdatahub.os.uk/downloads/open/OpenZoomstack
Contains OS data © Crown Copyright and database right.

Usage:
    from tools.os_opendata_tiles import fetch_os_opendata_grid

    tile_info = fetch_os_opendata_grid(lat, lon, zoom, n_tiles_x, n_tiles_y)
    # Returns dict compatible with existing pipeline:
    # {"image": np.array, "zoom": int, "tx_min": int, "ty_min": int, ...}
"""

import math
import os
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
GPKG_PATH = BASE / "os_opendata" / "OS_Open_Zoomstack.gpkg"
TILE_CACHE_DIR = BASE / "cache" / "os_opendata_tiles"

# ── Coordinate transforms ────────────────────────────────────────────────────

def _lat_lon_to_tile(lat, lon, zoom):
    """Convert lat/lon to tile coordinates (Web Mercator)."""
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    lat_rad = math.radians(lat)
    y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
    return x, y


def _tile_to_bounds_3857(zoom, tx, ty, tile_size=256):
    """Convert tile coordinates to bounds in EPSG:3857 (Web Mercator meters)."""
    n = 2 ** zoom
    # World extent in EPSG:3857
    origin = 20037508.342789244
    tile_extent = 2 * origin / n

    x_min = -origin + tx * tile_extent
    x_max = x_min + tile_extent
    y_max = origin - ty * tile_extent  # y is inverted in tile coords
    y_min = y_max - tile_extent

    return x_min, y_min, x_max, y_max


# ── GeoPackage reader (lazy, cached) ─────────────────────────────────────────

_gpkg_cache = {}


def _read_layer(layer_name, bounds_27700, gpkg_path=None):
    """Read features from a GeoPackage layer within BNG bounds.

    Args:
        layer_name: GeoPackage layer name (e.g., 'buildings', 'roads')
        bounds_27700: (xmin, ymin, xmax, ymax) in EPSG:27700
        gpkg_path: Path to GeoPackage file

    Returns:
        GeoDataFrame with features, or None if layer doesn't exist / no features
    """
    import geopandas as gpd
    from shapely.geometry import box

    if gpkg_path is None:
        gpkg_path = str(GPKG_PATH)

    bbox = box(*bounds_27700)
    try:
        gdf = gpd.read_file(
            gpkg_path,
            layer=layer_name,
            bbox=bbox,
            engine="pyogrio",
        )
        if gdf.empty:
            return None
        return gdf
    except Exception:
        return None


def _transform_3857_to_27700(x_min, y_min, x_max, y_max):
    """Convert EPSG:3857 bounds to EPSG:27700 (British National Grid)."""
    import pyproj
    transformer = pyproj.Transformer.from_crs(
        "EPSG:3857", "EPSG:27700", always_xy=True
    )
    # Transform corners
    x1, y1 = transformer.transform(x_min, y_min)
    x2, y2 = transformer.transform(x_max, y_max)
    # Add buffer to handle projection distortion at edges
    buf = max(abs(x2 - x1), abs(y2 - y1)) * 0.05
    return min(x1, x2) - buf, min(y1, y2) - buf, max(x1, x2) + buf, max(y1, y2) + buf


def _transform_27700_to_pixels(geom, bounds_3857, tile_size=256):
    """Transform a shapely geometry from BNG to pixel coordinates.

    We go BNG → 3857 → pixel so tile pixels align with Web Mercator grid.
    """
    import pyproj
    from shapely.ops import transform as shapely_transform

    transformer_to_3857 = pyproj.Transformer.from_crs(
        "EPSG:27700", "EPSG:3857", always_xy=True
    )
    x_min, y_min, x_max, y_max = bounds_3857
    x_extent = x_max - x_min
    y_extent = y_max - y_min

    def _to_pixel(x, y):
        # BNG → 3857
        mx, my = transformer_to_3857.transform(x, y)
        # 3857 → pixel (y is inverted)
        px = (mx - x_min) / x_extent * tile_size
        py = (1 - (my - y_min) / y_extent) * tile_size
        return px, py

    return shapely_transform(_to_pixel, geom)


# ── Tile renderer ─────────────────────────────────────────────────────────────

# UK planning map style colors (BGR for cv2)
STYLE = {
    "background":   (232, 240, 245),   # light cream/buff
    "building":     (179, 179, 255),   # salmon/pink fill
    "building_outline": (100, 100, 100),
    "road_fill":    (255, 255, 255),   # white
    "road_casing":  (160, 160, 160),   # gray
    "motorway":     (180, 200, 255),   # light orange-pink
    "a_road":       (200, 220, 255),   # light salmon
    "water":        (255, 217, 179),   # light blue
    "woodland":     (192, 230, 200),   # light green
    "greenspace":   (216, 240, 224),   # very light green
    "rail":         (120, 120, 120),   # dark gray
}

# Road widths by type (pixels at z17)
ROAD_WIDTHS_Z17 = {
    "Motorway": (6, 8),       # (fill, casing)
    "A Road": (4, 6),
    "B Road": (3, 5),
    "Minor Road": (2, 3),
    "Local Street": (1.5, 2.5),
    "Alley": (1, 1.5),
    "Pedestrianised Street": (1, 1.5),
}


def render_tile(zoom, tx, ty, gpkg_path=None, tile_size=256):
    """Render a single 256x256 tile from OS Open Zoomstack data.

    Styling mimics UK planning map conventions for LoFTR matching.

    Args:
        zoom: Web Mercator zoom level
        tx, ty: Tile coordinates
        gpkg_path: Path to OS Open Zoomstack GeoPackage
        tile_size: Output tile size in pixels (default 256)

    Returns:
        numpy RGB array (tile_size x tile_size x 3) or None on failure
    """
    if gpkg_path is None:
        gpkg_path = str(GPKG_PATH)

    if not os.path.exists(gpkg_path):
        raise FileNotFoundError(
            f"OS Open Zoomstack GeoPackage not found at {gpkg_path}. "
            f"Download from https://osdatahub.os.uk/downloads/open/OpenZoomstack"
        )

    bounds_3857 = _tile_to_bounds_3857(zoom, tx, ty, tile_size)
    bounds_27700 = _transform_3857_to_27700(*bounds_3857)

    # Scale factor for road widths relative to z17
    scale = 2 ** (zoom - 17)

    # Create canvas
    canvas = np.full((tile_size, tile_size, 3), STYLE["background"], dtype=np.uint8)

    def _geom_to_pixel_coords(geom):
        """Convert geometry to pixel coordinates as numpy array."""
        pixel_geom = _transform_27700_to_pixels(geom, bounds_3857, tile_size)
        return pixel_geom

    # ── Layer 1: Greenspaces ──────────────────────────────────────────────
    gdf = _read_layer("greenspace", bounds_27700, gpkg_path)
    if gdf is not None:
        for _, row in gdf.iterrows():
            pixel_geom = _geom_to_pixel_coords(row.geometry)
            _draw_polygon(canvas, pixel_geom, STYLE["greenspace"])

    # ── Layer 2: Woodland ─────────────────────────────────────────────────
    gdf = _read_layer("woodland", bounds_27700, gpkg_path)
    if gdf is not None:
        for _, row in gdf.iterrows():
            pixel_geom = _geom_to_pixel_coords(row.geometry)
            _draw_polygon(canvas, pixel_geom, STYLE["woodland"])

    # ── Layer 3: Water (surfacewater) ─────────────────────────────────────
    gdf = _read_layer("surfacewater", bounds_27700, gpkg_path)
    if gdf is not None:
        for _, row in gdf.iterrows():
            pixel_geom = _geom_to_pixel_coords(row.geometry)
            _draw_polygon(canvas, pixel_geom, STYLE["water"])

    # ── Layer 4: Water lines ──────────────────────────────────────────────
    gdf = _read_layer("waterlines", bounds_27700, gpkg_path)
    if gdf is not None:
        width = max(1, int(2 * scale))
        for _, row in gdf.iterrows():
            pixel_geom = _geom_to_pixel_coords(row.geometry)
            _draw_line(canvas, pixel_geom, STYLE["water"], width)

    # ── Layer 5: Buildings (local_buildings for detail) ───────────────────
    gdf = _read_layer("local_buildings", bounds_27700, gpkg_path)
    if gdf is not None:
        for _, row in gdf.iterrows():
            pixel_geom = _geom_to_pixel_coords(row.geometry)
            _draw_polygon(canvas, pixel_geom, STYLE["building"],
                         outline=STYLE["building_outline"], outline_width=1)

    # ── Layer 6: Roads (all three layers: local, regional, national) ─────
    # Combine all road layers with appropriate type labels
    import pandas as pd
    road_gdfs = []
    for layer_name, default_type in [
        ("roads_local", "Local Street"),
        ("roads_regional", "B Road"),
        ("roads_national", "A Road"),
    ]:
        gdf = _read_layer(layer_name, bounds_27700, gpkg_path)
        if gdf is not None:
            if "type" not in gdf.columns:
                gdf["type"] = default_type
            road_gdfs.append(gdf)

    if road_gdfs:
        all_roads = pd.concat(road_gdfs, ignore_index=True)
        # Sort by road type so major roads draw on top
        road_order = {"Motorway": 0, "A Road": 1, "B Road": 2,
                      "Minor Road": 3, "Local Street": 4,
                      "Alley": 5, "Pedestrianised Street": 5}
        all_roads["_order"] = all_roads["type"].map(
            lambda t: road_order.get(t, 4)
        )
        all_roads = all_roads.sort_values("_order", ascending=False)  # minor first

        # Pass 1: casings
        for _, row in all_roads.iterrows():
            road_type = row.get("type", "Local Street")
            widths = ROAD_WIDTHS_Z17.get(road_type, (1.5, 2.5))
            casing_w = max(1, int(widths[1] * scale))
            pixel_geom = _geom_to_pixel_coords(row.geometry)
            color = STYLE["road_casing"]
            _draw_line(canvas, pixel_geom, color, casing_w)

        # Pass 2: fills
        for _, row in all_roads.iterrows():
            road_type = row.get("type", "Local Street")
            widths = ROAD_WIDTHS_Z17.get(road_type, (1.5, 2.5))
            fill_w = max(1, int(widths[0] * scale))
            pixel_geom = _geom_to_pixel_coords(row.geometry)
            if road_type == "Motorway":
                color = STYLE["motorway"]
            elif road_type == "A Road":
                color = STYLE["a_road"]
            else:
                color = STYLE["road_fill"]
            _draw_line(canvas, pixel_geom, color, fill_w)

    # ── Layer 7: Railways ─────────────────────────────────────────────────
    gdf = _read_layer("rail", bounds_27700, gpkg_path)
    if gdf is not None:
        width = max(1, int(1.5 * scale))
        for _, row in gdf.iterrows():
            pixel_geom = _geom_to_pixel_coords(row.geometry)
            _draw_line(canvas, pixel_geom, STYLE["rail"], width)

    return canvas


def _draw_polygon(canvas, pixel_geom, fill_color, outline=None, outline_width=1):
    """Draw a polygon geometry on the canvas."""
    from shapely.geometry import Polygon, MultiPolygon

    if pixel_geom.is_empty:
        return

    polys = []
    if isinstance(pixel_geom, Polygon):
        polys = [pixel_geom]
    elif isinstance(pixel_geom, MultiPolygon):
        polys = list(pixel_geom.geoms)
    else:
        return

    for poly in polys:
        if poly.is_empty:
            continue
        exterior = np.array(poly.exterior.coords, dtype=np.int32)
        if len(exterior) < 3:
            continue
        cv2.fillPoly(canvas, [exterior], fill_color)
        if outline is not None:
            cv2.polylines(canvas, [exterior], True, outline, outline_width,
                         lineType=cv2.LINE_AA)
        # Draw holes
        for interior in poly.interiors:
            hole = np.array(interior.coords, dtype=np.int32)
            if len(hole) >= 3:
                cv2.fillPoly(canvas, [hole], STYLE["background"])


def _draw_line(canvas, pixel_geom, color, width):
    """Draw a line geometry on the canvas."""
    from shapely.geometry import LineString, MultiLineString

    if pixel_geom.is_empty:
        return

    lines = []
    if isinstance(pixel_geom, LineString):
        lines = [pixel_geom]
    elif isinstance(pixel_geom, MultiLineString):
        lines = list(pixel_geom.geoms)
    else:
        return

    for line in lines:
        if line.is_empty:
            continue
        pts = np.array(line.coords, dtype=np.int32)
        if len(pts) < 2:
            continue
        cv2.polylines(canvas, [pts], False, color, width, lineType=cv2.LINE_AA)


# ── Cached tile fetching ──────────────────────────────────────────────────────

def _tile_cache_path(zoom, tx, ty):
    return TILE_CACHE_DIR / f"{zoom}" / f"{tx}" / f"{ty}.png"


def fetch_tile_cached(zoom, tx, ty, gpkg_path=None):
    """Fetch a single OS OpenData tile with file-based caching.

    Returns numpy RGB array (256x256x3) or None on failure.
    """
    cache_path = _tile_cache_path(zoom, tx, ty)
    if cache_path.exists():
        img = cv2.imread(str(cache_path))
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tile = render_tile(zoom, tx, ty, gpkg_path=gpkg_path)
    if tile is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Store as BGR for cv2
        cv2.imwrite(str(cache_path), tile)
        return tile
    return None


def _render_canvas_bulk(zoom, tx_min, ty_min, n_tiles_x, n_tiles_y, gpkg_path=None):
    """Render an entire tile grid as one canvas with single spatial queries per layer.

    Instead of 169 separate render_tile() calls (each doing 7+ spatial queries),
    this does 7 spatial queries total for the whole grid. ~20x faster on cold cache.
    """
    import pandas as pd

    if gpkg_path is None:
        gpkg_path = str(GPKG_PATH)
    if not os.path.exists(gpkg_path):
        raise FileNotFoundError(f"GeoPackage not found at {gpkg_path}")

    tile_size = 256
    canvas_w = n_tiles_x * tile_size
    canvas_h = n_tiles_y * tile_size

    # Compute full bounds in 3857 for the entire grid
    bounds_3857 = (
        _tile_to_bounds_3857(zoom, tx_min, ty_min, tile_size)[0],          # x_min (left edge)
        _tile_to_bounds_3857(zoom, tx_min, ty_min + n_tiles_y - 1, tile_size)[1],  # y_min (bottom edge)
        _tile_to_bounds_3857(zoom, tx_min + n_tiles_x - 1, ty_min, tile_size)[2],  # x_max (right edge)
        _tile_to_bounds_3857(zoom, tx_min, ty_min, tile_size)[3],          # y_max (top edge)
    )
    bounds_27700 = _transform_3857_to_27700(*bounds_3857)

    scale = 2 ** (zoom - 17)
    canvas = np.full((canvas_h, canvas_w, 3), STYLE["background"], dtype=np.uint8)

    def _geom_to_pixel(geom):
        return _transform_27700_to_pixels(geom, bounds_3857, canvas_w)

    # Pixel transform needs to know canvas dimensions for both axes
    x_min_3857, y_min_3857, x_max_3857, y_max_3857 = bounds_3857
    x_extent = x_max_3857 - x_min_3857
    y_extent = y_max_3857 - y_min_3857

    import pyproj
    from shapely.ops import transform as shapely_transform
    transformer_to_3857 = pyproj.Transformer.from_crs("EPSG:27700", "EPSG:3857", always_xy=True)

    def _geom_to_pixels(geom):
        def _to_px(x, y):
            mx, my = transformer_to_3857.transform(x, y)
            px = (mx - x_min_3857) / x_extent * canvas_w
            py = (1 - (my - y_min_3857) / y_extent) * canvas_h
            return px, py
        return shapely_transform(_to_px, geom)

    # ── Layer 1: Greenspaces
    gdf = _read_layer("greenspace", bounds_27700, gpkg_path)
    if gdf is not None:
        for _, row in gdf.iterrows():
            _draw_polygon(canvas, _geom_to_pixels(row.geometry), STYLE["greenspace"])

    # ── Layer 2: Woodland
    gdf = _read_layer("woodland", bounds_27700, gpkg_path)
    if gdf is not None:
        for _, row in gdf.iterrows():
            _draw_polygon(canvas, _geom_to_pixels(row.geometry), STYLE["woodland"])

    # ── Layer 3: Water (surfacewater)
    gdf = _read_layer("surfacewater", bounds_27700, gpkg_path)
    if gdf is not None:
        for _, row in gdf.iterrows():
            _draw_polygon(canvas, _geom_to_pixels(row.geometry), STYLE["water"])

    # ── Layer 4: Water lines
    gdf = _read_layer("waterlines", bounds_27700, gpkg_path)
    if gdf is not None:
        width = max(1, int(2 * scale))
        for _, row in gdf.iterrows():
            _draw_line(canvas, _geom_to_pixels(row.geometry), STYLE["water"], width)

    # ── Layer 5: Buildings
    gdf = _read_layer("local_buildings", bounds_27700, gpkg_path)
    if gdf is not None:
        for _, row in gdf.iterrows():
            _draw_polygon(canvas, _geom_to_pixels(row.geometry), STYLE["building"],
                         outline=STYLE["building_outline"], outline_width=1)

    # ── Layer 6: Roads (all three layers combined)
    road_gdfs = []
    for layer_name, default_type in [
        ("roads_local", "Local Street"),
        ("roads_regional", "B Road"),
        ("roads_national", "A Road"),
    ]:
        gdf = _read_layer(layer_name, bounds_27700, gpkg_path)
        if gdf is not None:
            if "type" not in gdf.columns:
                gdf["type"] = default_type
            road_gdfs.append(gdf)

    if road_gdfs:
        all_roads = pd.concat(road_gdfs, ignore_index=True)
        road_order = {"Motorway": 0, "A Road": 1, "B Road": 2,
                      "Minor Road": 3, "Local Street": 4,
                      "Alley": 5, "Pedestrianised Street": 5}
        all_roads["_order"] = all_roads["type"].map(lambda t: road_order.get(t, 4))
        all_roads = all_roads.sort_values("_order", ascending=False)

        # Pass 1: casings
        for _, row in all_roads.iterrows():
            road_type = row.get("type", "Local Street")
            widths = ROAD_WIDTHS_Z17.get(road_type, (1.5, 2.5))
            casing_w = max(1, int(widths[1] * scale))
            _draw_line(canvas, _geom_to_pixels(row.geometry), STYLE["road_casing"], casing_w)

        # Pass 2: fills
        for _, row in all_roads.iterrows():
            road_type = row.get("type", "Local Street")
            widths = ROAD_WIDTHS_Z17.get(road_type, (1.5, 2.5))
            fill_w = max(1, int(widths[0] * scale))
            if road_type == "Motorway":
                color = STYLE["motorway"]
            elif road_type == "A Road":
                color = STYLE["a_road"]
            else:
                color = STYLE["road_fill"]
            _draw_line(canvas, _geom_to_pixels(row.geometry), color, fill_w)

    # ── Layer 7: Railways
    gdf = _read_layer("rail", bounds_27700, gpkg_path)
    if gdf is not None:
        width = max(1, int(1.5 * scale))
        for _, row in gdf.iterrows():
            _draw_line(canvas, _geom_to_pixels(row.geometry), STYLE["rail"], width)

    return canvas


def _grid_cache_path(zoom, tx_min, ty_min, n_tiles_x, n_tiles_y):
    """Cache path for a full rendered grid."""
    return TILE_CACHE_DIR / "grids" / f"z{zoom}_{tx_min}_{ty_min}_{n_tiles_x}x{n_tiles_y}.png"


NLS_TILE_LAYERS = {
    "newpopular": {
        "url": "https://mapseries-tilesets.s3.amazonaws.com/os/newpopular/{z}/{x}/{y}.png",
        "name": "OS New Popular 1940s-1950s",
        "max_zoom": 16,
        "min_zoom": 10,
    },
    "6inch": {
        "url": "https://mapseries-tilesets.s3.amazonaws.com/os/6inchsecond/{z}/{x}/{y}.png",
        "name": "OS 6-inch 2nd ed 1888-1913",
        "max_zoom": 17,
        "min_zoom": 10,
    },
    "10k": {
        "url": "https://mapseries-tilesets.s3.amazonaws.com/os/britain10knatgrid/{z}/{x}/{y}.png",
        "name": "OS 10k National Grid 1950s-60s",
        "max_zoom": 16,
        "min_zoom": 10,
    },
}
NLS_CACHE_DIR = BASE / "cache" / "nls_historical_tiles"


def fetch_historical_grid(lat, lon, zoom, n_tiles_x, n_tiles_y, layer="newpopular"):
    """Fetch a grid of NLS historical OS tiles.

    Free, no API key. Tiles hosted on S3 by National Library of Scotland.

    Available layers:
      - "newpopular": OS New Popular 1940s-1950s (zoom 10-16) — best for mid-century docs
      - "6inch": OS 6-inch 2nd edition 1888-1913 (zoom 10-17) — best for Victorian docs
      - "10k": OS 10k National Grid 1950s-60s (zoom 10-16) — similar era to newpopular

    Args:
        lat, lon: Center coordinates (WGS84).
        zoom: Tile zoom level (will be clamped to layer's range).
        n_tiles_x, n_tiles_y: Grid dimensions.
        layer: Which historical layer to use (default "newpopular").

    Returns:
        Dict with 'image' (numpy RGB), 'zoom', 'tx_min', 'ty_min', etc.
        Same format as fetch_os_opendata_grid for drop-in use.
    """
    import requests
    from PIL import Image
    from io import BytesIO

    layer_cfg = NLS_TILE_LAYERS.get(layer, NLS_TILE_LAYERS["newpopular"])
    tile_url_template = layer_cfg["url"]

    # Clamp zoom to layer's available range
    zoom = max(layer_cfg["min_zoom"], min(layer_cfg["max_zoom"], zoom))

    cx, cy = _lat_lon_to_tile(lat, lon, zoom)
    half_x = n_tiles_x // 2
    half_y = n_tiles_y // 2
    tx_min = cx - half_x
    ty_min = cy - half_y
    tile_size = 256

    # Check cache
    cache_dir = NLS_CACHE_DIR / layer / "grids"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"z{zoom}_{tx_min}_{ty_min}_{n_tiles_x}x{n_tiles_y}.png"
    if cache_path.exists():
        img = cv2.imread(str(cache_path))
        if img is not None:
            canvas = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"  NLS {layer_cfg['name']}: loaded cached grid z{zoom} ({n_tiles_x}x{n_tiles_y})")
            return {
                "image": canvas, "zoom": zoom,
                "tx_min": tx_min, "ty_min": ty_min,
                "nx": n_tiles_x, "ny": n_tiles_y, "tile_size": tile_size,
            }

    # Fetch tiles
    canvas = np.ones((n_tiles_y * tile_size, n_tiles_x * tile_size, 3),
                     dtype=np.uint8) * 255  # white background

    fetched = 0
    for dy in range(n_tiles_y):
        for dx in range(n_tiles_x):
            tx = tx_min + dx
            ty = ty_min + dy
            url = tile_url_template.format(z=zoom, x=tx, y=ty)
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    tile_img = np.array(Image.open(BytesIO(r.content)).convert("RGB"))
                    canvas[dy*tile_size:(dy+1)*tile_size,
                           dx*tile_size:(dx+1)*tile_size] = tile_img
                    fetched += 1
            except Exception:
                pass

    # Cache the grid
    cv2.imwrite(str(cache_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"  NLS {layer_cfg['name']}: fetched {fetched}/{n_tiles_x*n_tiles_y} tiles "
          f"z{zoom} ({n_tiles_x}x{n_tiles_y})")

    return {
        "image": canvas, "zoom": zoom,
        "tx_min": tx_min, "ty_min": ty_min,
        "nx": n_tiles_x, "ny": n_tiles_y, "tile_size": tile_size,
    }


def fetch_os_opendata_grid(lat, lon, zoom, n_tiles_x, n_tiles_y, gpkg_path=None):
    """Fetch a grid of OS OpenData tiles using bulk rendering.

    Same interface as tools.os_tiles.fetch_tile_grid for drop-in replacement.
    Uses single spatial query per layer for the entire grid (~20x faster than
    per-tile rendering on cold cache).

    Args:
        lat, lon: Center coordinates (WGS84).
        zoom: Tile zoom level (15-19 recommended).
        n_tiles_x, n_tiles_y: Grid dimensions.
        gpkg_path: Path to GeoPackage (default: os_opendata/OS_Open_Zoomstack.gpkg)

    Returns:
        Dict with 'image' (numpy RGB), 'zoom', 'tx_min', 'ty_min', 'nx', 'ny', 'tile_size'.
    """
    cx, cy = _lat_lon_to_tile(lat, lon, zoom)
    half_x = n_tiles_x // 2
    half_y = n_tiles_y // 2
    tx_min = cx - half_x
    ty_min = cy - half_y

    tile_size = 256

    # Check grid cache first
    cache_path = _grid_cache_path(zoom, tx_min, ty_min, n_tiles_x, n_tiles_y)
    if cache_path.exists():
        img = cv2.imread(str(cache_path))
        if img is not None:
            canvas = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"  OS OpenData: loaded cached grid z{zoom} ({n_tiles_x}x{n_tiles_y})")
            return {
                "image": canvas, "zoom": zoom,
                "tx_min": tx_min, "ty_min": ty_min,
                "nx": n_tiles_x, "ny": n_tiles_y, "tile_size": tile_size,
            }

    # Bulk render
    import time
    t0 = time.time()
    canvas = _render_canvas_bulk(zoom, tx_min, ty_min, n_tiles_x, n_tiles_y, gpkg_path)
    elapsed = time.time() - t0
    print(f"  OS OpenData: bulk rendered z{zoom} ({n_tiles_x}x{n_tiles_y}) in {elapsed:.1f}s")

    # Cache the grid
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cache_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    return {
        "image": canvas, "zoom": zoom,
        "tx_min": tx_min, "ty_min": ty_min,
        "nx": n_tiles_x, "ny": n_tiles_y, "tile_size": tile_size,
    }
