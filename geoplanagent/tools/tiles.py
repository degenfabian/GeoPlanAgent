"""Render raster tiles from OS Open Zoomstack, styled to match UK planning maps for cross-modal matching."""

import cv2
import numpy as np

from geoplanagent.paths import OS_ZOOMSTACK_GPKG

# Web Mercator (EPSG:3857) world half-extent in metres: π · 6378137 (the WGS84
# equatorial radius). The projection spans ±this on both the x and y axes.
WEB_MERCATOR_EXTENT_M = 20037508.342789244


def _tile_to_bounds_3857(
    zoom: int, tile_x: int, tile_y: int
) -> tuple[float, float, float, float]:
    """Convert a slippy-map tile (zoom/x/y) to its ground bounds in EPSG:3857.

    Args:
        zoom: slippy-map zoom level; the world is split into 2**zoom tiles per axis.
        tile_x: tile column index (0 at the western edge, increasing east).
        tile_y: tile row index (0 at the northern edge, increasing south).

    Returns:
        ``(x_min, y_min, x_max, y_max)`` bounds in Web Mercator metres.
    """
    tiles_per_axis = 2**zoom
    tile_extent_m = 2 * WEB_MERCATOR_EXTENT_M / tiles_per_axis

    x_min = -WEB_MERCATOR_EXTENT_M + tile_x * tile_extent_m
    x_max = x_min + tile_extent_m
    # Tile y increases southward, so the top edge is the larger y value.
    y_max = WEB_MERCATOR_EXTENT_M - tile_y * tile_extent_m
    y_min = y_max - tile_extent_m

    return x_min, y_min, x_max, y_max


def _read_layer(layer_name: str, bounds_27700: tuple[float, float, float, float]):
    """Read one OS Open Zoomstack layer's features inside a bounding box.

    Args:
        layer_name: GeoPackage layer to read (e.g. ``"roads_local"``,
            ``"local_buildings"``, ``"surfacewater"``, ``"woodland"``).
        bounds_27700: ``(x_min, y_min, x_max, y_max)`` in British National Grid
            (EPSG:27700) metres; only features intersecting this box are read.

    Returns:
        A GeoDataFrame of the matching features, or ``None`` if the layer is
        empty, missing, or the read fails.
    """
    import geopandas as gpd
    from shapely.geometry import box

    bounding_box = box(*bounds_27700)
    try:
        layer = gpd.read_file(
            str(OS_ZOOMSTACK_GPKG),
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
    """Convert a Web Mercator (EPSG:3857) bounding box to British National Grid (EPSG:27700).

    Args:
        x_min, y_min, x_max, y_max: bounding-box corners in EPSG:3857 metres.

    Returns:
        ``(x_min, y_min, x_max, y_max)`` in EPSG:27700 metres, padded by 5% to
        absorb projection distortion that bows the straight box edges (so the
        BNG box safely covers the Mercator one).
    """
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
    """Fill a polygon area feature (building, water, woodland) onto the canvas.

    MultiPolygons are split into parts; each part is filled inside its exterior
    ring but NOT inside its interior rings (holes), so any feature already drawn
    underneath shows through the holes. An optional outline strokes the edge.

    Args:
        canvas: the BGR image being drawn on (mutated in place).
        pixel_geom: a shapely Polygon/MultiPolygon whose coords are already in
            PIXEL space (non-polygon geometries are ignored).
        fill_color: BGR fill colour for the polygon interior.
        outline: BGR colour to stroke the polygon edge, or None for no outline.
        outline_width: edge thickness in pixels (used only when outline is given).
    """
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
        hole_contours = [
            np.array(interior.coords, dtype=np.int32)
            for interior in polygon.interiors
            if len(interior.coords) >= 3
        ]
        if hole_contours:
            # Paint only (exterior minus holes) via a mask, so anything already
            # drawn under a hole shows through instead of being overwritten with
            # the background.
            mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [exterior_points], 255)
            cv2.fillPoly(mask, hole_contours, 0)
            canvas[mask == 255] = fill_color
        else:
            cv2.fillPoly(canvas, [exterior_points], fill_color)
        if outline is not None:
            cv2.polylines(
                canvas, [exterior_points], True, outline, outline_width, lineType=cv2.LINE_AA
            )


def _draw_line(
    canvas: np.ndarray, pixel_geom, color: tuple[int, int, int], width: int
) -> None:
    """Stroke a line feature (road, railway, watercourse) onto the canvas.

    MultiLineStrings are split into parts; each is drawn as an anti-aliased polyline.

    Args:
        canvas: the BGR image being drawn on (mutated in place).
        pixel_geom: a shapely LineString/MultiLineString whose coords are already
            in PIXEL space (non-line geometries are ignored).
        color: BGR colour for the line.
        width: line thickness in pixels.
    """
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

    Paints the OS layers in cartographic order (greenspace/woodland/water →
    buildings → road casings then fills by class → rail). Reading the full grid
    extent once, rather than per tile, avoids re-opening the GeoPackage for every
    tile. Feature widths scale with zoom off the z17 baseline (2 ** (zoom - 17)).

    Args:
        zoom: slippy-map zoom level to render at (sets the feature widths).
        tx_min: tile column index of the grid's left edge.
        ty_min: tile row index of the grid's top edge.
        n_tiles_x: grid width in tiles.
        n_tiles_y: grid height in tiles.

    Returns:
        The styled basemap as one BGR canvas of shape
        ``(n_tiles_y * 256, n_tiles_x * 256, 3)``.

    Raises:
        FileNotFoundError: if the source GeoPackage is missing.
    """
    import pandas as pd

    if not OS_ZOOMSTACK_GPKG.exists():
        raise FileNotFoundError(f"GeoPackage not found at {OS_ZOOMSTACK_GPKG}")

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
        all_roads["_order"] = all_roads["type"].map(lambda road_type: road_order.get(road_type, 4))
        all_roads = all_roads.sort_values("_order", ascending=False)

        # Casings underneath, then fills on top, so neighbouring roads share a clean edge.
        for _, row in all_roads.iterrows():
            road_type = row.get("type", "Local Street")
            _, casing_width = ROAD_WIDTHS_Z17.get(road_type, (1.5, 2.5))
            _draw_line(
                canvas,
                _geom_to_pixels(row.geometry),
                STYLE["road_casing"],
                max(1, int(casing_width * scale)),
            )

        for _, row in all_roads.iterrows():
            road_type = row.get("type", "Local Street")
            fill_width, _ = ROAD_WIDTHS_Z17.get(road_type, (1.5, 2.5))
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


def fetch_os_opendata_grid(
    lat: float, lon: float, zoom: int, n_tiles_x: int, n_tiles_y: int
) -> dict:
    """Render a styled OS-OpenData basemap grid around a point.

    Centres an ``n_tiles_x × n_tiles_y`` tile grid on (lat, lon) at the given
    zoom and renders it via ``_render_canvas_bulk``.

    Args:
        lat: centre latitude (WGS84).
        lon: centre longitude (WGS84).
        zoom: slippy-map zoom level to render at.
        n_tiles_x: grid width in tiles.
        n_tiles_y: grid height in tiles.

    Returns:
        dict with ``image`` (the rendered grid), ``zoom``, ``tx_min``, ``ty_min``,
        ``nx``, ``ny``, and ``tile_size`` (256) — the geo-anchoring the matcher
        needs to map result pixels back to coordinates.
    """
    import time

    from geoplanagent.utils import latlon_to_tile_xy

    center_tile_x, center_tile_y = latlon_to_tile_xy(lat, lon, zoom)
    tx_min = center_tile_x - n_tiles_x // 2
    ty_min = center_tile_y - n_tiles_y // 2

    render_start_s = time.time()
    canvas = _render_canvas_bulk(zoom, tx_min, ty_min, n_tiles_x, n_tiles_y)
    elapsed_s = time.time() - render_start_s
    print(f"  OS OpenData: bulk rendered z{zoom} ({n_tiles_x}x{n_tiles_y}) in {elapsed_s:.1f}s")

    return {
        "image": canvas,
        "zoom": zoom,
        "tx_min": tx_min,
        "ty_min": ty_min,
        "nx": n_tiles_x,
        "ny": n_tiles_y,
        "tile_size": 256,
    }
