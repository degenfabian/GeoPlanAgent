"""Core agent objects shared across the tool modules.

Extracted from ``tools/agent.py`` (stage-2 split, 2026-05-11). Defines:

* The PydanticAI ``Agent`` instances (``_agent``, ``_reader_agent``)
* The shared ``AgentState`` carried as ``deps``
* History processor (``_strip_old_images``)
* Common helpers (``resize_for_api``, ``_img_to_binary``, ``_dedup_check``,
  ``_create_boundary_overlay``, ``_draw_geojson_on_tiles``)
* The output validator and the worker system-prompt registrar
  (both decorate ``_agent`` and therefore must live with the agent
  instance)
* Model alias table + the transient-HTTP-error retry helper

Tool modules (``agent_tools_render``, ``agent_tools_locate``,
``agent_tools_match``, ``agent_tools_extract``, ``agent_tools_verify``)
import ``_agent`` and ``AgentState`` from here. They register their
tools at import time via the ``@_agent.tool`` / ``@_agent.tool_plain``
decorators.

Backward compatibility: ``tools.agent`` re-exports everything defined
here so existing ``from tools.agent import …`` call sites continue to
work.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from pydantic_ai import Agent, BinaryContent, ModelRetry, RunContext

# Structured I/O schemas and prompt text live in their own modules
# (extracted in stage 1 of the agent split, 2026-05-11).
from tools.agent_schemas import (
    BoundaryConstraint,
    PDFInfo,
    CenterInput,
    BoundaryOutcome,
)
from tools.agent_prompts import READER_SYSTEM_PROMPT, WORKER_SYSTEM_PROMPT

load_dotenv()


# ── Helpers ─────────────────────────────────────────────────────────────────

def resize_for_api(img: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """Resize image so largest dimension is max_dim."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def _img_to_binary(img: np.ndarray) -> BinaryContent:
    """Convert numpy BGR image to PydanticAI BinaryContent."""
    _, buf = cv2.imencode('.png', resize_for_api(img))
    return BinaryContent(data=buf.tobytes(), media_type='image/png')


def _dedup_check(state: "AgentState", tool_name: str, args: dict) -> None:
    """Raise ModelRetry if this exact tool+args was already called."""
    key = tool_name + ":" + hashlib.md5(
        json.dumps(args, sort_keys=True, default=str).encode()
    ).hexdigest()
    if key in state.recent_calls:
        raise ModelRetry(
            "You already called this tool with the same arguments. "
            "Try different arguments or respond with DONE."
        )
    state.recent_calls.add(key)


def _create_boundary_overlay(map_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay boundary mask on map image (red tint, 40% opacity)."""
    overlay = map_img.copy()
    if mask is not None and mask.shape[:2] == map_img.shape[:2]:
        overlay[mask > 0] = [0, 0, 255]
    return cv2.addWeighted(map_img, 0.6, overlay, 0.4, 0)


def _draw_geojson_on_tiles(tile_bgr, geojson, tile_info):
    """Draw GeoJSON boundary outline on tile canvas."""
    geom = geojson.get("geometry", {})
    coord_rings = []
    if geom.get("type") == "Polygon":
        coord_rings = [geom["coordinates"][0]]
    elif geom.get("type") == "MultiPolygon":
        coord_rings = [poly[0] for poly in geom["coordinates"]]

    zoom = tile_info.get("zoom", 17)
    tx_min = tile_info.get("tx_min", 0)
    ty_min = tile_info.get("ty_min", 0)
    tile_size = tile_info.get("tile_size", 256)

    from tools.geo.coords import latlon_to_global_tile_pixel
    for ring in coord_rings:
        pts = []
        for lon_c, lat_c in ring:
            abs_px, abs_py = latlon_to_global_tile_pixel(
                lat_c, lon_c, zoom, tile_size)
            px = abs_px - tx_min * tile_size
            py = abs_py - ty_min * tile_size
            pts.append([int(px), int(py)])
        if len(pts) >= 3:
            cv2.polylines(tile_bgr, [np.array(pts, dtype=np.int32)],
                          True, (0, 0, 255), 2)
    return tile_bgr


# ── Agent State ─────────────────────────────────────────────────────────────

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
        # used by tools/sam3_boundary.set_fold_for_case to switch the
        # active LoRA adapter per case at inference time. None means a
        # legacy single-adapter or base SAM3, which set_fold_for_case
        # treats as a no-op.
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

        # Set by render_page
        self.map_img: Optional[np.ndarray] = None
        self.map_crop_path: Optional[str] = None

        # Set by extract_boundary
        self.current_mask: Optional[np.ndarray] = None
        self.instance_masks: List[np.ndarray] = []

        # Set by match_at + commit_match
        self.current_result: dict = {}
        # NOTE: state.centers and state.scale_ratio were removed in v18.
        # They were populated only by `_position_boundary_disabled` (now
        # deleted) and read by `_fallback_geojson_at_anchor` (now deleted).
        # If you need the committed center, read it from
        # `state.current_result["match_info"]["chosen_center_latlon"]`.

        # Cache for offline analysis: candidate overlays + final selection
        self.candidate_overlays: List[np.ndarray] = []  # per-candidate viz images
        self.selected_overlay: Optional[np.ndarray] = None  # final combined mask viz
        self.selected_indices: Optional[List[int]] = None  # which candidates were chosen

        # Agent metadata
        self.accepted = False
        self.accept_reason = ""
        self.recent_calls: set = set()
        self.position_calls: int = 0

        # Structured-output validator tracking
        self.pdf_info: Dict[str, Any] = {}  # populated from reader phase
        self.verify_position_called: bool = False
        self.rotation_checked: bool = False
        self.last_output: Optional["BoundaryOutcome"] = None  # set by validator

        # Critic (Phase 3) — filled by tools/critic.run_critic_loop
        self.critic_iterations: List[dict] = []
        self.critic_final_decision: Optional[str] = None
        self.critic_changed_mask: bool = False
        self.critic_applied_rotation_deg: Optional[int] = None
        self.critic_suspected_wrong_location: bool = False
        self.critic_worker_reentered: bool = False

        # Geocoding transparency — filled by position_boundary
        self.centers_tried: List[Dict[str, Any]] = []

        # ── v2 agentic positioning (the default path) ───────────────────
        # propose_centers populates this with ranked candidate centers from
        # locate.py + positioning.py's internal geocoders, unified.
        self.proposed_centers: List[Dict[str, Any]] = []
        # match_at stores each match attempt by integer candidate_id so the
        # agent can refer to it in subsequent score_match / commit_match calls.
        self.match_attempts: Dict[int, Dict[str, Any]] = {}
        self._match_attempt_counter: int = 0
        # Per-case budget — agent can call match_at up to this many times
        # before being forced to commit or reject.
        self.match_at_budget: int = 5


# ── Phase 1: PDF Reader Agent ──────────────────────────────────────────────

_reader_agent = Agent(
    "test",  # placeholder, overridden at runtime
    output_type=PDFInfo,
    retries=2,
    output_retries=2,
    model_settings={"temperature": 0},
    instructions=READER_SYSTEM_PROMPT,
)


# ── Phase 2: Worker Agent Definition ──────────────────────────────────────

def _strip_old_images(messages):
    """Replace BinaryContent in messages older than KEEP_RECENT with a placeholder.

    Without this, each extract_boundary(mode='instance') call attaches 5 candidate
    images that get replayed every subsequent turn — token cost grows quadratically.
    """
    KEEP_RECENT = 4
    if len(messages) <= KEEP_RECENT:
        return messages
    cutoff = len(messages) - KEEP_RECENT

    for i, msg in enumerate(messages):
        if i >= cutoff:
            continue
        parts = getattr(msg, 'parts', None)
        if not parts:
            continue
        for part in parts:
            content = getattr(part, 'content', None)
            if isinstance(content, list):
                for j, item in enumerate(content):
                    if (hasattr(item, 'media_type')
                            and hasattr(item, 'data')
                            and getattr(item, 'media_type', '').startswith('image/')):
                        content[j] = (
                            f"[image omitted from older history; "
                            f"was {item.media_type}, {len(item.data)} bytes]"
                        )
    return messages


_agent = Agent(
    "test",  # overridden at runtime via model= kwarg
    deps_type=AgentState,
    output_type=BoundaryOutcome,
    retries=5,
    output_retries=3,
    history_processors=[_strip_old_images],
    # temperature=0 for reproducible runs. The recovery experiments showed
    # ~+8pp of "best of N" gain came from LLM nondeterminism alone; setting
    # temperature=0 isolates the effect of the deterministic improvements
    # (multi-prompt SAM3, color fallback, analytical affine) from that.
    model_settings={"temperature": 0},
)


# ── Output validator ────────────────────────────────────────────────────────

@_agent.output_validator
async def validate_boundary_outcome(
    ctx: RunContext[AgentState], out: BoundaryOutcome
) -> BoundaryOutcome:
    """Enforce that required tool calls happened before accepting an outcome.

    Pydantic-AI raises ModelRetry on failure and the agent has to submit again
    after filling the gap. This is what makes verify_position and multi-page
    accumulation actually mandatory rather than suggested.
    """
    state = ctx.deps
    state.last_output = out

    mi = state.current_result.get("match_info") or {}
    final_inl = mi.get("n_inliers", 0) or 0

    # The model sometimes hallucinates that it called verify_position / rotated
    # the map. Override the schema fields with real state.
    if out.verify_position_called != state.verify_position_called:
        out.verify_position_called = state.verify_position_called
    if out.rotation_checked != state.rotation_checked:
        out.rotation_checked = state.rotation_checked
    if out.final_n_inliers != final_inl:
        out.final_n_inliers = final_inl

    # District_lookup still requires verify_position — catches cases where the
    # reader mis-flagged district-wide and lookup_district returned a 900 km²
    # polygon when the real boundary is a single site.
    if out.status == "district_lookup":
        if state.current_result.get("geojson") is None:
            raise ModelRetry(
                "status='district_lookup' requires a successful lookup_district "
                "call that produced a GeoJSON. Call lookup_district with the "
                "district_name from the PDFInfo and retry."
            )
        if not state.verify_position_called:
            raise ModelRetry(
                "status='district_lookup' requires you to call verify_position "
                "first. Look at the OS tile with the district polygon overlaid, "
                "then compare against the planning map. If the district polygon "
                "is dramatically larger than what the map shows, set status to "
                "'rejected_no_match' instead. Call verify_position now, fill "
                "visual_check_notes, then resubmit."
            )
        if len(out.visual_check_notes.strip()) < 20:
            raise ModelRetry(
                "district_lookup requires visual_check_notes (≥20 chars) "
                "describing whether the district polygon matches the planning "
                "map's apparent scope."
            )
        return out

    if out.status.startswith("rejected_"):
        if not (out.reject_reason and out.reject_reason.strip()):
            raise ModelRetry(
                f"status='{out.status}' requires a reject_reason explaining why "
                f"(1-2 sentences). Fill reject_reason and resubmit."
            )
        return out

    # status == "accepted" from here on — preconditions below.

    if final_inl == 0 and state.current_result.get("geojson") is None:
        raise ModelRetry(
            "Cannot accept: no successful match_at + commit_match has produced "
            "a result. Either run positioning to completion (propose_centers → "
            "match_at → commit_match → extract_boundary → project_boundary), "
            "or set status to 'rejected_no_match' with a reject_reason."
        )

    # Borderline positioning (25-100 inliers) must be manually verified.
    if 25 <= final_inl <= 100:
        if not state.verify_position_called:
            raise ModelRetry(
                f"Positioning produced {final_inl} inliers (borderline band 25-100). "
                f"You MUST call verify_position to visually compare the OS tile "
                f"against the planning map before accepting. Call verify_position "
                f"now, compare the road/feature patterns, then resubmit with "
                f"verify_position_called=True and visual_check_notes describing "
                f"the comparison. If features do NOT match, set status to "
                f"'rejected_visual_mismatch' instead."
            )
        if len(out.visual_check_notes.strip()) < 20:
            raise ModelRetry(
                f"verify_position was called but visual_check_notes is too short "
                f"(len={len(out.visual_check_notes.strip())}). Describe in at "
                f"least 20 characters whether the OS tile features match the "
                f"planning map (road patterns, settlement shape, named roads)."
            )

    return out


# ── System Prompt ───────────────────────────────────────────────────────────

@_agent.system_prompt
def build_system_prompt(ctx: RunContext[AgentState]) -> str:
    return WORKER_SYSTEM_PROMPT


# ── Model Aliases ──────────────────────────────────────────────────────────

MODEL_ALIASES = {
    "claude-opus": "anthropic/claude-opus-4.6",
    "claude-sonnet": "anthropic/claude-sonnet-4-6",
    "gpt-5.4": "openai/gpt-5.4",
    "gpt-5.4-mini": "openai/gpt-5.4-mini",
    "gpt-5.4-nano": "openai/gpt-5.4-nano",
    "gemini-pro": "google/gemini-3.1-pro-preview",
    "gemini-flash": "google/gemini-3-flash-preview",
    "gemini-flash-lite": "google/gemini-3.1-flash-lite-preview",
}


# ── Transient-HTTP-error retry helper ───────────────────────────────────────

# Status codes that are typically transient and worth retrying. 400 is
# included because OpenRouter routinely surfaces upstream Gemini hiccups
# (rate limit, model overload, transient safety-check backend failures)
# as a generic 400 with body "Provider returned error".
_RETRYABLE_STATUS = {400, 408, 425, 429, 500, 502, 503, 504}


def _is_retryable_http_error(exc: Exception) -> bool:
    """True if this exception looks like a transient OpenRouter/provider hiccup."""
    try:
        from pydantic_ai.exceptions import ModelHTTPError
    except Exception:
        return False
    if not isinstance(exc, ModelHTTPError):
        return False
    # ModelHTTPError stringifies as "status_code: NNN, model_name: ..., body: ..."
    s = str(exc)
    import re
    m = re.search(r"status_code:\s*(\d+)", s)
    if not m:
        return False
    return int(m.group(1)) in _RETRYABLE_STATUS


def _run_sync_with_retry(agent_obj, *args, max_retries: int = 2,
                          backoff_s: float = 5.0, label: str = "agent",
                          **kwargs):
    """Wrap Agent.run_sync with retries on transient HTTP errors.

    Why: gemini-3-flash-preview hits transient 400s (~22% of v11 cases —
    "Provider returned error" with no body) that recover on retry.
    Without this, a single hiccup mid-conversation kills the whole case
    and we lose the agent's accumulated state. Each retry re-runs the
    whole conversation from scratch — expensive but the only option since
    pydantic-ai doesn't expose mid-stream resume.

    Non-retryable errors (auth, bad input, ModelRetry / UnexpectedModelBehavior)
    are re-raised immediately so we don't waste cycles.
    """
    import time as _time
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return agent_obj.run_sync(*args, **kwargs)
        except Exception as e:
            if not _is_retryable_http_error(e) or attempt == max_retries:
                raise
            wait = backoff_s * (2 ** attempt)
            print(f"  {label}: transient HTTP error (attempt "
                  f"{attempt + 1}/{max_retries + 1}): {str(e)[:140]}"
                  f" — retrying in {wait:.0f}s")
            _time.sleep(wait)
            last_exc = e
    # unreachable; raise above covers the no-retry-left path
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{label}: retry loop fell through without error")
