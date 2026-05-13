"""
Planning Document Boundary Extraction Tools

Pipeline entry point:

  agent.py              — PydanticAI agent (5 tools) and main control loop
  agent_core.py         — Shared agent state + utilities
  agent_prompts.py      — Reader / locator / matcher / critic system prompts
  agent_schemas.py      — Pydantic models for tool inputs/outputs

Agent-tool modules (one per logical step):

  agent_tools_render.py  — render_page (PDF -> map crop)
  agent_tools_locate.py  — propose_centers (locate_v2 cascade)
  agent_tools_match.py   — match_at + commit_match (MINIMA sliding window)
  agent_tools_extract.py — extract_boundary (SAM3 -> mask -> GeoJSON)
  agent_tools_verify.py  — critic loop helpers

Core pipeline modules:

  matching.py            — MINIMA sliding-window position search + scoring
  candidates.py          — locate_v2 candidate generation + ranking
  sam3_boundary.py       — SAM3 boundary segmentation (semantic + instance)
  critic.py              — VLM critic loop
  pdf_tools.py           — PDF page rendering
  text_extraction.py     — PDF OCR + structured info extraction
  os_opendata_tiles.py   — OS OpenData tile rendering (offline)
  geocoders.py           — Multi-source geocoding (OS Names, postcodes.io, Nominatim, gpkg)
  code_point.py          — Code-Point Open postcode lookup
  os_names.py            — OS Open Names search
  positioning_sources.py — Anchor cascade primitives
  geojson_metrics.py     — IoU, precision/recall/F1
  visualization_tools.py — Map-overlay rendering for the benchmark report

Snap / verification post-processors:

  snap/inspire.py        — INSPIRE freehold-parcel boundary snap
  delaunay_filter.py     — Delaunay-consistency RANSAC filter (optional)
  verification_checks.py — Cross-checks (LA polygon, scale, area) for critic

Utilities:

  logging_utils.py, map_crop.py, mask_ops.py, scale_bar_ocr.py,
  rotation_classifier.py, boundary_color.py, locate_eval.py, reward.py
"""
