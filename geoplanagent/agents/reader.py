"""Phase 1 reader Agent: one multimodal call over the raw PDF -> PDFInfo."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic_ai import Agent

from geoplanagent.prompts import READER_SYSTEM_PROMPT
from geoplanagent.schemas import PDFInfo

load_dotenv()

# Production runs at temperature 0 for reproducibility; the
# GEOMAP_TEMPERATURE env var lets the appendix ablation re-run at 1.0
# without disturbing the cached benchmarks.
_TEMPERATURE = float(os.environ.get("GEOMAP_TEMPERATURE", "0"))


_reader_agent = Agent(
    "test",  # placeholder, overridden at runtime via model= kwarg
    output_type=PDFInfo,
    retries=2,
    output_retries=2,
    model_settings={"temperature": _TEMPERATURE},
    instructions=READER_SYSTEM_PROMPT,
)
