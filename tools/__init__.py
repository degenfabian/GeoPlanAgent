"""
Planning Document Boundary Extraction Tools

Core modules used by the unified agent (tools/agent.py):

  agent.py              — PydanticAI agent with 9 tools (main entry point)
  critic.py             — Phase 3 Commenter VLM critic loop (Paper2Poster-style)
  sam3_boundary.py      — SAM3 boundary segmentation (semantic + instance)
  positioning.py        — MINIMA sliding-window matching + mask->GeoJSON projection
  geocoding.py          — Multi-source geocoding (Photon, postcodes.io, Nominatim)
  geo_tools.py          — OS grid ref parsing, district boundary lookup
  os_opendata_tiles.py  — OS OpenData tile rendering (offline, no API key)
  geojson_metrics.py    — IoU, precision, recall, F1, positioning error
  pdf_tools.py          — PDF page rendering
  visualization_tools.py — GeoPandas boundary visualization

See tools/README.md for detailed module documentation.
"""
