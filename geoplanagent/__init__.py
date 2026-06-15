"""GeoPlanAgent: planning-document boundary extraction.

  agents/     — one file per LLM agent: reader, worker, locate sub-agent, critic.
  tools/      — what the agents call: positioning.py (the worker's tool
                surface), matching.py (MINIMA registration engine),
                geocode.py (offline UK geocoders), segment.py (SAM3
                k-fold), tiles.py (OS Zoomstack renderer), pdf.py
                (rendering + rotation).
  prompts.py  — every system prompt.
  schemas.py  — the pydantic schemas (LLM-visible field docs).
  utils.py    — AgentState, model aliases, retry, geodesy, fold routing.
  metrics.py  — IoU/precision/recall/centroid scoring + metric aggregation.
"""
