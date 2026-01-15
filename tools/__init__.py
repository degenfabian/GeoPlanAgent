"""
Planning Document Digitization Tools

This package provides tools for extracting planning area boundaries from
map images and converting them to geographic coordinates (GeoJSON).

PDF PROCESSING (pdf_tools.py):

  get_pdf_page_as_image: Convert a PDF page to base64 image (also returns total_pages)

BOUNDARY EXTRACTION (boundary_tools.py):

  extract_color_boundary(image_base64, lower_hsv, upper_hsv)
    - Use when the boundary is a distinct colored line (orange, red, blue, etc.)
    - LLM should determine the HSV color range based on what it sees

  extract_region_boundary(image_base64)
    - Use when the boundary is black/dark or color extraction doesn't work
    - Uses grayscale edge detection, no color parameters needed

GEOGRAPHIC TRANSFORMATION (geo_tools.py):

  pixels_to_geo_linear: Linear transformation (center + scale)
  lookup_district_boundary: Look up district boundary from OSM
  geocode_address: Convert address to coordinates

VISUALIZATION (visualization_tools.py):

  visualize_geojson_boundary: Visualize boundary on OSM map
"""

# Import PDF tools
from .pdf_tools import (
    get_pdf_page_as_image,
    PDF_TOOLS,
)

# Import boundary tools
from .boundary_tools import (
    extract_color_boundary,
    extract_region_boundary,
    BOUNDARY_TOOLS,
)

# Import geo tools
from .geo_tools import (
    pixels_to_geo_linear,
    lookup_district_boundary,
    geocode_address,
    GEO_TOOLS,
)

# Import visualization tools
from .visualization_tools import (
    visualize_geojson_boundary,
    VISUALIZATION_TOOLS,
)

# Combined tool definitions for LLM function calling
ALL_TOOLS = PDF_TOOLS + BOUNDARY_TOOLS + GEO_TOOLS + VISUALIZATION_TOOLS

# Export all
__all__ = [
    # PDF tools
    "get_pdf_page_as_image",
    "PDF_TOOLS",
    # Boundary tools
    "extract_color_boundary",
    "extract_region_boundary",
    "BOUNDARY_TOOLS",
    # Geo tools
    "pixels_to_geo_linear",
    "lookup_district_boundary",
    "geocode_address",
    "GEO_TOOLS",
    # Visualization tools
    "visualize_geojson_boundary",
    "VISUALIZATION_TOOLS",
    # Combined
    "ALL_TOOLS",
]
