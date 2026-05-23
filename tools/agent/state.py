"""Per-case mutable AgentState passed to the worker as deps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tools.agent.schemas import BoundaryOutcome


class AgentState:
    """Mutable state shared across all tool calls."""

    def __init__(self, pdf_path, sam3_processor, sam3_model, device,
                 minima_matcher, dpi=200, sam3_state=None, case_name=None,
                 locate_model: str = "google/gemini-3-flash-preview"):
        self.pdf_path = pdf_path
        self.sam3_processor = sam3_processor
        self.sam3_model = sam3_model
        self.device = device
        self.minima_matcher = minima_matcher
        self.dpi = dpi
        self.locate_model: str = locate_model
        self.sam3_state: Optional[Dict[str, Any]] = sam3_state
        # Case folder name; needed for k-fold adapter routing.
        self.case_name: Optional[str] = case_name
        if self.case_name is None and pdf_path:
            try:
                self.case_name = Path(pdf_path).parent.name
            except Exception:
                pass

        # Pre-rendered match pages, keyed by 1-based page number.
        self.rendered_pages: Dict[int, np.ndarray] = {}
        self.rendered_page_paths: Dict[int, str] = {}

        # Lazily computed in match_at on first need per page.
        self.sam_masks_by_page: Dict[int, np.ndarray] = {}

        self.current_result: dict = {}

        self.accepted = False
        self.accept_reason = ""
        self.recent_calls: set = set()
        self.position_calls: int = 0

        self.pdf_info: Dict[str, Any] = {}
        self.rotation_checked: bool = False
        self.last_output: Optional["BoundaryOutcome"] = None

        # Locate sub-agent's picked candidates (one entry usually).
        self.proposed_centers: List[Dict[str, Any]] = []
        # Full message history from the most recent run_locate call.
        # When the worker re-invokes propose_centers, this is passed back
        # to run_locate as `prior_messages` so the locate sub-agent sees
        # its previous reasoning + tool calls + pick.
        self.locate_message_history: List[Any] = []
        # Each match attempt covers one area_group; commit_match references by id.
        self.match_attempts: Dict[int, Dict[str, Any]] = {}
        self._match_attempt_counter: int = 0
        # area_group → committed candidate_id. Multi-area docs accumulate;
        # current_result["geojson"] is the union across groups.
        self.committed_groups: Dict[int, int] = {}
        self.match_at_budget: int = 5


# ── Page-of-interest helpers ─────────────────────────────────────────────

def primary_match_page(state: "AgentState") -> Optional[int]:
    """Highest-ranked map_page from the reader, or None."""
    pages = ((state.pdf_info or {}).get("map_pages") or [])
    return int(pages[0]) if pages else None


def committed_primary_page(state: "AgentState") -> Optional[int]:
    """Page of the worker's committed primary group, else the default match page."""
    cr = state.current_result or {}
    per_group = cr.get("per_group") or []
    if per_group:
        requested = cr.get("requested_group")
        primary = next(
            (g for g in per_group if g.get("area_group") == requested),
            per_group[0],
        )
        page = primary.get("page")
        if page is not None:
            return int(page)
    return primary_match_page(state)


from tools.agent._helpers import (  # noqa: E402, F401
    _img_to_binary,
    _dedup_check,
    _create_boundary_overlay,
    _draw_geojson_on_tiles,
)
from tools.agent._retry import _run_sync_with_retry  # noqa: E402, F401
from tools.agent.worker_agent import _agent  # noqa: E402, F401
