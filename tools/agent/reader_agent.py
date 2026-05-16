"""Phase 1 reader: one-shot pydantic-ai Agent over the PDF binary.

Output is locked to PDFInfo (schema in tools.agent.schemas). The actual
read call lives in tools.agent.runtime.read_pdf_phase — this module just
defines the Agent instance and its system prompt.
"""

from __future__ import annotations

from dotenv import load_dotenv
from pydantic_ai import Agent

from tools.agent.prompts import READER_SYSTEM_PROMPT
from tools.agent.schemas import PDFInfo

load_dotenv()


_reader_agent = Agent(
    "test",  # placeholder, overridden at runtime via model= kwarg
    output_type=PDFInfo,
    retries=2,
    output_retries=2,
    model_settings={"temperature": 0},
    instructions=READER_SYSTEM_PROMPT,
)
