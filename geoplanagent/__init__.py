"""GeoPlanAgent: planning-document boundary extraction.

Start reading at run.py::run_agent — the per-case entry point the
benchmark drives (reader phase → map-page rendering with auto-rotation →
worker tool loop → optional critic).

  agents/     — one file per LLM agent: reader, worker (+ output
                validator), locate sub-agent, critic.
  tools/      — what the agents call: positioning.py (the worker's tool
                surface), matching.py (MINIMA registration engine),
                geocode.py (offline UK geocoders), segment.py (SAM3
                k-fold), tiles.py (OS Zoomstack renderer), pdf.py
                (rendering + rotation).
  prompts.py  — every system prompt and prompt section.
  schemas.py  — the pydantic contracts (LLM-visible field docs).
  utils.py    — AgentState, model aliases, retry, geodesy, fold routing.
  metrics.py  — IoU/precision/recall/centroid scoring + metric aggregation.
"""
