"""Planning-document boundary extraction toolkit.

Top-level packages:

  agent/      — PydanticAI orchestrator (reader, worker) + the live
                LLM-locate sub-agent (locate_agent.py) called from the
                worker's propose_centers tool.
  matching/   — MINIMA sliding-window matcher + RANSAC affine fit
  extraction/ — SAM3 boundary segmentation + colour primitives + mask ops
  geo/        — Geographic primitives: lat/lon math, BNG grid-ref parsing,
                OS BoundaryLine LA-polygon resolution, Code-Point Open
                postcode lookup, OS Open Names search.
  io/         — PDF render, OS tile render, page rotation, text extraction
  metrics/    — IoU/F1/positioning metrics, viz overlays, MINIMA reward

Top-level helpers:

  build_oml_road_index.py — script to regenerate oml_road_index.json /
                            oml_road_geom_subset.json from OS OpenMap Local
"""
