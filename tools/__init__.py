"""Planning-document boundary extraction toolkit.

Start reading at tools/agent/__init__.py::run_agent — the per-case entry
point the benchmark drives. It calls runtime.py (phases: render pages →
reader → worker loop → optional critic); the worker's four tools live in
agent/worker_tools.py and fan out into the subsystem packages:

  agent/      — the LLM layer: runtime, worker agent + validator,
                locate sub-agent, critic, prompts, pydantic schemas,
                shared state/support, model-alias resolution.
  matching/   — MINIMA sliding-window matcher, RANSAC affine, composite
                rerank, match-quality reward signals, road-name check.
  extraction/ — SAM3 boundary segmentation (k-fold LoRA loader).
  geo/        — one module per offline data source: coordinate math,
                BNG grid refs, OS BoundaryLine, Code-Point Open,
                OS Open Names.
  io/         — PDF/page rendering + case files, OS tile rendering,
                rotation classifier.
  metrics/    — IoU/F1/centroid metrics and comparison visualisation.

Top-level helpers:

  fold_routing.py — k-fold case→fold routing shared by SAM3 and the
                    rotation classifier.
  build_oml_road_index.py — script to regenerate oml_road_index.json /
                            oml_road_geom_subset.json from OS OpenMap Local
"""
