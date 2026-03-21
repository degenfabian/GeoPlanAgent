"""Phase 1 reader: one multimodal LLM call over the raw planning PDF.

Fills the PDFInfo schema — the map page(s) to position, plus the textual
location hints (place names, grid references, addresses) later stages anchor to.
"""

from dotenv import load_dotenv
from pydantic_ai import Agent

from geoplanagent.prompts import READER_SYSTEM_PROMPT
from geoplanagent.schemas import PDFInfo

load_dotenv()


_reader_agent = Agent(
    "test",  # placeholder, overridden at runtime via model= kwarg
    output_type=PDFInfo,
    retries=2,
    output_retries=2,
    # Temperature 0 for reproducible production runs;
    model_settings={"temperature": 0},
    instructions=READER_SYSTEM_PROMPT,
)
