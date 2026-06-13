"""Render raster tiles from OS Open Zoomstack, styled to match UK planning maps for cross-modal matching."""

from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GPKG_PATH = REPO_ROOT / "os_opendata" / "OS_Open_Zoomstack.gpkg"
TILE_CACHE_DIR = REPO_ROOT / "cache" / "os_opendata_tiles"


def _tile_to_bounds_3857(
    zoom: int, tile_x: int, tile_y: int
) -> tuple[float, float, float, float]:
    """Convert tile coordinates to bounds in EPSG:3857 (Web Mercator metres)."""
    tiles_per_axis = 2**zoom
    world_extent_m = 20037508.342789244
    tile_extent_m = 2 * world_extent_m / tiles_per_axis

    x_min = -world_extent_m + tile_x * tile_extent_m
    x_max = x_min + tile_extent_m
    # Tile y increases southward, so the top edge is the larger y value.
    y_max = world_extent_m - tile_y * tile_extent_m
    y_min = y_max - tile_extent_m

    return x_min, y_min, x_max, y_max


def _read_layer(layer_name: str, bounds_27700: tuple[float, float, float, float]):
    """Read GeoPackage layer features within BNG bounds; None if empty/missing."""
    import geopandas as gpd
    from shapely.geometry import box

    bounding_box = box(*bounds_27700)
    try:
        layer = gpd.read_file(
            str(GPKG_PATH),
            layer=layer_name,
            bbox=bounding_box,
            engine="pyogrio",
        )
        if layer.empty:
            return None
        return layer
    except Exception:
        return None


def _transform_3857_to_27700(
    x_min: float, y_min: float, x_max: float, y_max: float
) -> tuple[float, float, float, float]:
    """Convert EPSG:3857 bounds to EPSG:27700 (British National Grid)."""
    import pyproj

    transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:27700", always_xy=True)
    corner_x_1, corner_y_1 = transformer.transform(x_min, y_min)
    corner_x_2, corner_y_2 = transformer.transform(x_max, y_max)
    # Pad the box to absorb projection distortion that bends the straight edges.
    margin_m = max(abs(corner_x_2 - corner_x_1), abs(corner_y_2 - corner_y_1)) * 0.05
    return (
        min(corner_x_1, corner_x_2) - margin_m,
        min(corner_y_1, corner_y_2) - margin_m,
        max(corner_x_1, corner_x_2) + margin_m,
        max(corner_y_1, corner_y_2) + margin_m,
    )


# UK planning map style colours (BGR for cv2)
STYLE = {
    "background": (232, 240, 245),  # light cream/buff
    "building": (179, 179, 255),  # salmon/pink fill
    "building_outline": (100, 100, 100),
    "road_fill": (255, 255, 255),  # white
    "road_casing": (160, 160, 160),  # gray
    "motorway": (180, 200, 255),  # light orange-pink
    "a_road": (200, 220, 255),  # light salmon
    "water": (255, 217, 179),  # light blue
    "woodland": (192, 230, 200),  # light green
    "greenspace": (216, 240, 224),  # very light green
    "rail": (120, 120, 120),  # dark gray
}

# Road widths by type (pixels at z17)
ROAD_WIDTHS_Z17 = {
    "Motorway": (6, 8),  # (fill, casing)
    "A Road": (4, 6),
    "B Road": (3, 5),
    "Minor Road": (2, 3),
    "Local Street": (1.5, 2.5),
    "Alley": (1, 1.5),
    "Pedestrianised Street": (1, 1.5),
}


def _draw_polygon(
    canvas: np.ndarray,
    pixel_geom,
    fill_color: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
    outline_width: int = 1,
) -> None:
    """Draw a polygon geometry on the canvas."""
    from shapely.geometry import Polygon, MultiPolygon

    if pixel_geom.is_empty:
        return

    if isinstance(pixel_geom, Polygon):
        polygons = [pixel_geom]
    elif isinstance(pixel_geom, MultiPolygon):
        polygons = list(pixel_geom.geoms)
    else:
        return

    for polygon in polygons:
        if polygon.is_empty:
            continue
        exterior_points = np.array(polygon.exterior.coords, dtype=np.int32)
        if len(exterior_points) < 3:
            continue
        cv2.fillPoly(canvas, [exterior_points], fill_color)
        if outline is not None:
            cv2.polylines(
                canvas, [exterior_points], True, outline, outline_width, lineType=cv2.LINE_AA
            )
        # Punch holes back to the background colour.
        for interior in polygon.interiors:
            hole_points = np.array(interior.coords, dtype=np.int32)
            if len(hole_points) >= 3:
                cv2.fillPoly(canvas, [hole_points], STYLE["background"])


def _draw_line(
    canvas: np.ndarray, pixel_geom, color: tuple[int, int, int], width: int
) -> None:
    """Draw a line geometry on the canvas."""
    from shapely.geometry import LineString, MultiLineString

    if pixel_geom.is_empty:
        return

    if isinstance(pixel_geom, LineString):
        lines = [pixel_geom]
    elif isinstance(pixel_geom, MultiLineString):
        lines = list(pixel_geom.geoms)
    else:
        return

    for line in lines:
        if line.is_empty:
            continue
        line_points = np.array(line.coords, dtype=np.int32)
        if len(line_points) < 2:
            continue
        cv2.polylines(canvas, [line_points], False, color, width, lineType=cv2.LINE_AA)


def _render_canvas_bulk(
    zoom: int, tx_min: int, ty_min: int, n_tiles_x: int, n_tiles_y: int
) -> np.ndarray:
    """Render the whole tile grid as one canvas, querying each layer once for the full extent.

    Raises FileNotFoundError if the source GeoPackage is missing.
    """
    import pandas as pd

    if not GPKG_PATH.exists():
        raise FileNotFoundError(f"GeoPackage not found at {GPKG_PATH}")

    tile_size = 256
    canvas_w = n_tiles_x * tile_size
    canvas_h = n_tiles_y * tile_size

    # Stitch the four edges of the grid into a single 3857 bounding box.
    bounds_3857 = (
        _tile_to_bounds_3857(zoom, tx_min, ty_min)[0],  # x_min (left edge)
        _tile_to_bounds_3857(zoom, tx_min, ty_min + n_tiles_y - 1)[1],  # y_min (bottom edge)
        _tile_to_bounds_3857(zoom, tx_min + n_tiles_x - 1, ty_min)[2],  # x_max (right edge)
        _tile_to_bounds_3857(zoom, tx_min, ty_min)[3],  # y_max (top edge)
    )
    bounds_27700 = _transform_3857_to_27700(*bounds_3857)

    scale = 2 ** (zoom - 17)
    canvas = np.full((canvas_h, canvas_w, 3), STYLE["background"], dtype=np.uint8)

    x_min_3857, y_min_3857, x_max_3857, y_max_3857 = bounds_3857
    x_extent = x_max_3857 - x_min_3857
    y_extent = y_max_3857 - y_min_3857

    import pyproj
    from shapely.ops import transform as shapely_transform

    transformer_to_3857 = pyproj.Transformer.from_crs("EPSG:27700", "EPSG:3857", always_xy=True)

    def _geom_to_pixels(geom):
        def _to_pixel(x, y):
            mercator_x, mercator_y = transformer_to_3857.transform(x, y)
            pixel_x = (mercator_x - x_min_3857) / x_extent * canvas_w
            pixel_y = (1 - (mercator_y - y_min_3857) / y_extent) * canvas_h
            return pixel_x, pixel_y

        return shapely_transform(_to_pixel, geom)

    def _draw_polygon_layer(layer_name: str, fill_color: tuple[int, int, int]) -> None:
        layer = _read_layer(layer_name, bounds_27700)
        if layer is not None:
            for _, row in layer.iterrows():
                _draw_polygon(canvas, _geom_to_pixels(row.geometry), fill_color)

    def _draw_line_layer(
        layer_name: str, color: tuple[int, int, int], width: int
    ) -> None:
        layer = _read_layer(layer_name, bounds_27700)
        if layer is not None:
            for _, row in layer.iterrows():
                _draw_line(canvas, _geom_to_pixels(row.geometry), color, width)

    _draw_polygon_layer("greenspace", STYLE["greenspace"])
    _draw_polygon_layer("woodland", STYLE["woodland"])
    _draw_polygon_layer("surfacewater", STYLE["water"])
    _draw_line_layer("waterlines", STYLE["water"], max(1, int(2 * scale)))

    buildings = _read_layer("local_buildings", bounds_27700)
    if buildings is not None:
        for _, row in buildings.iterrows():
            _draw_polygon(
                canvas,
                _geom_to_pixels(row.geometry),
                STYLE["building"],
                outline=STYLE["building_outline"],
                outline_width=1,
            )

    # The three road layers are merged so painting order is by road class, not source layer.
    road_layers = []
    for layer_name, default_type in [
        ("roads_local", "Local Street"),
        ("roads_regional", "B Road"),
        ("roads_national", "A Road"),
    ]:
        layer = _read_layer(layer_name, bounds_27700)
        if layer is not None:
            if "type" not in layer.columns:
                layer["type"] = default_type
            road_layers.append(layer)

    if road_layers:
        all_roads = pd.concat(road_layers, ignore_index=True)
        road_order = {
            "Motorway": 0,
            "A Road": 1,
            "B Road": 2,
            "Minor Road": 3,
            "Local Street": 4,
            "Alley": 5,
            "Pedestrianised Street": 5,
        }
        all_roads["_order"] = all_roads["type"].map(lambda t: road_order.get(t, 4))
        all_roads = all_roads.sort_values("_order", ascending=False)

        # Casings underneath, then fills on top, so neighbouring roads share a clean edge.
        for _, row in all_roads.iterrows():
            road_type = row.get("type", "Local Street")
            fill_width, casing_width = ROAD_WIDTHS_Z17.get(road_type, (1.5, 2.5))
            _draw_line(
                canvas,
                _geom_to_pixels(row.geometry),
                STYLE["road_casing"],
                max(1, int(casing_width * scale)),
            )

        for _, row in all_roads.iterrows():
            road_type = row.get("type", "Local Street")
            fill_width, casing_width = ROAD_WIDTHS_Z17.get(road_type, (1.5, 2.5))
            if road_type == "Motorway":
                color = STYLE["motorway"]
            elif road_type == "A Road":
                color = STYLE["a_road"]
            else:
                color = STYLE["road_fill"]
            _draw_line(
                canvas, _geom_to_pixels(row.geometry), color, max(1, int(fill_width * scale))
            )

    _draw_line_layer("rail", STYLE["rail"], max(1, int(1.5 * scale)))

    return canvas


def _grid_cache_path(
    zoom: int, tx_min: int, ty_min: int, n_tiles_x: int, n_tiles_y: int
) -> Path:
    """Cache path for a full rendered grid."""
    return TILE_CACHE_DIR / "grids" / f"z{zoom}_{tx_min}_{ty_min}_{n_tiles_x}x{n_tiles_y}.png"


def fetch_os_opendata_grid(
    lat: float, lon: float, zoom: int, n_tiles_x: int, n_tiles_y: int
) -> dict:
    """Bulk-render a tile grid from OS OpenData; returns dict with image, zoom, tx_min, ty_min, nx, ny, tile_size."""
    from geoplanagent.utils import latlon_to_tile_xy

    center_tile_x, center_tile_y = latlon_to_tile_xy(lat, lon, zoom)
    tx_min = center_tile_x - n_tiles_x // 2
    ty_min = center_tile_y - n_tiles_y // 2

    tile_size = 256

    cache_path = _grid_cache_path(zoom, tx_min, ty_min, n_tiles_x, n_tiles_y)
    if cache_path.exists():
        cached_bgr = cv2.imread(str(cache_path))
        if cached_bgr is not None:
            canvas = cv2.cvtColor(cached_bgr, cv2.COLOR_BGR2RGB)
            print(f"  OS OpenData: loaded cached grid z{zoom} ({n_tiles_x}x{n_tiles_y})")
            return {
                "image": canvas,
                "zoom": zoom,
                "tx_min": tx_min,
                "ty_min": ty_min,
                "nx": n_tiles_x,
                "ny": n_tiles_y,
                "tile_size": tile_size,
            }

    import time

    render_start_s = time.time()
    canvas = _render_canvas_bulk(zoom, tx_min, ty_min, n_tiles_x, n_tiles_y)
    elapsed_s = time.time() - render_start_s
    print(f"  OS OpenData: bulk rendered z{zoom} ({n_tiles_x}x{n_tiles_y}) in {elapsed_s:.1f}s")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cache_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    return {
        "image": canvas,
        "zoom": zoom,
        "tx_min": tx_min,
        "ty_min": ty_min,
        "nx": n_tiles_x,
        "ny": n_tiles_y,
        "tile_size": tile_size,
    }
