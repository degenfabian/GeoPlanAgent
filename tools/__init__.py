"""Planning-document boundary extraction toolkit.

Top-level packages:

  agent/      — PydanticAI orchestrator (reader, worker, critic) + the
                live LLM-locate sub-agent (locate_agent.py) called from the
                worker's propose_centers tool.
  matching/   — MINIMA sliding-window matcher + road-name verifier
  scoring.py  — composite_window_score, commit_attempt_score
  extraction/ — SAM3 boundary segmentation + colour primitives + mask ops
  geocoding/  — Code-Point Open, OS Open Names, postcodes.io dispatch,
                positioning-source primitives
  io/         — PDF render, OS tile render, page rotation, title-block crop,
                text extraction
  metrics/    — IoU/F1/positioning metrics, viz overlays, MINIMA reward
  geo/        — Read-only geospatial data adapters
  data/       — Cached geocoder responses (websearch landmark, adjacency, …)

Top-level helpers:

  delaunay_filter.py     — optional Delaunay-consistency RANSAC filter
  verification_checks.py — cross-checks (LA polygon, scale, area) fed to the
                           critic context
  build_oml_road_index.py — script to regenerate oml_road_index.json /
                            oml_road_geom_subset.json from OS OpenMap Local
"""
