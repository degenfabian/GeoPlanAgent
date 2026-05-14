"""Planning-document boundary extraction toolkit.

Top-level packages (each has its own ``__init__.py`` and re-exports):

  agent/      — PydanticAI orchestrator + tool implementations
                (__init__: run_agent; tools/render, locate, match, extract,
                verify; state, schemas, prompts, critic)
  locate/     — propose_centers_v2 cascade + ranker (was candidates.py)
  matching/   — MINIMA sliding-window matcher + road-name verifier
  scoring.py  — composite_window_score, commit_attempt_score
  extraction/ — SAM3 boundary segmentation + colour primitives + mask ops
                (sam3, boundary_color, mask_ops)
  geocoding/  — Code-Point Open, OS Open Names, postcodes.io dispatch,
                positioning-source primitives
  io/         — PDF render, OS tile render, page rotation, title-block crop,
                text extraction (pdf, os_tiles, rotation_classifier,
                map_crop, text_extraction)
  metrics/    — IoU/F1/positioning metrics, viz overlays, MINIMA reward
                (geojson, visualization, reward)
  snap/       — INSPIRE freehold-parcel boundary snap post-processor
  geo/, os_opendata/ — Read-only geospatial data adapters

Top-level helpers:

  candidates.py          — backwards-compat shim re-exporting tools.locate
  delaunay_filter.py     — optional Delaunay-consistency RANSAC filter
  verification_checks.py — cross-checks (LA polygon, scale, area) fed to the
                           critic context
"""
