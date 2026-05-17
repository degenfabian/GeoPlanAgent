"""AgentState — mutable per-case state passed to the worker as deps.

The Agent instances themselves live in tools.agent.reader_agent and
tools.agent.worker_agent (the latter decorates _agent with tools and the
output validator). Pure helpers live in tools.agent._helpers and the
HTTP retry helper in tools.agent._retry.

Re-exports `_agent` and a few helpers so the worker-tool modules under
`tools/agent/tools/` can keep doing `from tools.agent.state import
_agent, AgentState, _img_to_binary, _dedup_check, ...`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tools.agent.schemas import BoundaryOutcome


class AgentState:
    """Mutable state shared across all tool calls."""

    def __init__(self, pdf_path, sam3_processor, sam3_model, device,
                 minima_matcher, dpi=200, sam3_state=None, case_name=None):
        self.pdf_path = pdf_path
        self.sam3_processor = sam3_processor
        self.sam3_model = sam3_model
        self.device = device
        self.minima_matcher = minima_matcher
        self.dpi = dpi

        # Full SAM3 loader output (incl. k-fold metadata when available),
        # used by tools.extraction.sam3.set_fold_for_case to switch the
        # active LoRA adapter per case at inference time.
        self.sam3_state: Optional[Dict[str, Any]] = sam3_state
        # Case identifier (folder name in evaluation_data). Used for
        # k-fold adapter routing. If None, derived from pdf_path's parent
        # directory.
        self.case_name: Optional[str] = case_name
        if self.case_name is None and pdf_path:
            try:
                self.case_name = Path(pdf_path).parent.name
            except Exception:
                pass

        # Set by render_page (active page)
        self.map_img: Optional[np.ndarray] = None
        self.map_crop_path: Optional[str] = None

        # Pre-rendered cache of every map_page from the reader.
        # Populated by _read_pdf_phase; render_page(N) does a state-pointer
        # flip into these rather than re-rendering. Keyed by 1-based page.
        self.rendered_pages: Dict[int, np.ndarray] = {}
        self.rendered_page_paths: Dict[int, str] = {}

        # Set by extract_boundary
        self.current_mask: Optional[np.ndarray] = None

        # Set by match_at + commit_match
        self.current_result: dict = {}

        # Cache for offline analysis: candidate overlays + final selection
        self.candidate_overlays: List[np.ndarray] = []
        self.selected_overlay: Optional[np.ndarray] = None
        self.selected_indices: Optional[List[int]] = None

        # Agent metadata
        self.accepted = False
        self.accept_reason = ""
        self.recent_calls: set = set()
        self.position_calls: int = 0
        # reader_refine call counter (bounded per case in tools/refine.py).
        self.refine_calls: int = 0

        # Structured-output validator tracking
        self.pdf_info: Dict[str, Any] = {}  # populated from reader phase
        self.verify_position_called: bool = False
        self.rotation_checked: bool = False
        self.last_output: Optional["BoundaryOutcome"] = None

        # Critic (Phase 3) — filled by tools/agent/critic_agent.py
        self.critic_iterations: List[dict] = []
        self.critic_final_decision: Optional[str] = None
        self.critic_changed_mask: bool = False
        self.critic_applied_rotation_deg: Optional[int] = None
        self.critic_suspected_wrong_location: bool = False
        self.critic_worker_reentered: bool = False

        # Geocoding transparency — written by tools that emit centers.
        self.centers_tried: List[Dict[str, Any]] = []

        # Locate sub-agent's picked candidates (one entry usually).
        self.proposed_centers: List[Dict[str, Any]] = []
        # Full message history from the most recent run_locate call.
        # When the worker re-invokes propose_centers, this is passed back
        # to run_locate as `prior_messages` so the locate sub-agent sees
        # its previous reasoning + tool calls + pick.
        self.locate_message_history: List[Any] = []
        # match_at stores each match attempt by integer candidate_id so
        # commit_match can refer to it later.
        self.match_attempts: Dict[int, Dict[str, Any]] = {}
        self._match_attempt_counter: int = 0
        # Per-case budget — agent can call match_at up to this many times
        # before being forced to commit.
        self.match_at_budget: int = 5


# ── Re-exports used by tools/agent/tools/*.py ─────────────────────────────
# The worker-tool modules import `_agent` (for the decorator) plus a few
# image/dedup helpers from here rather than reaching across the package.

from tools.agent._helpers import (  # noqa: E402, F401
    _img_to_binary,
    _dedup_check,
    _create_boundary_overlay,
    _draw_geojson_on_tiles,
)
from tools.agent._retry import _run_sync_with_retry  # noqa: E402, F401
# worker_agent.py imports AgentState from here, so import _agent AFTER
# AgentState is defined.
from tools.agent.worker_agent import _agent  # noqa: E402, F401
