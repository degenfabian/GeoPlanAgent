"""Shared support for the whole package: the per-case AgentState blackboard,
image/mask helpers, the transient-HTTP retry wrapper, OpenRouter model-alias
resolution, geodesy/tile-pixel math, and k-fold case routing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np
import hashlib
import json
import cv2
from pydantic_ai import BinaryContent, ModelRetry
import re
import time as _time
from pydantic_ai.models.openrouter import OpenRouterModel
import math
from typing import Tuple


if TYPE_CHECKING:
    from geoplanagent.schemas import BoundaryOutcome


# Production locate sub-agent ships with `place` only — see the rationale on
# AgentState.locate_disabled_tools below.
PRODUCTION_LOCATE_DISABLED_TOOLS: frozenset = frozenset(
    {"postcode", "grid_ref", "road", "intersect", "la_check"}
)


class AgentState:
    """Mutable state shared across all tool calls."""

    def __init__(self, pdf_path, sam3_processor, sam3_model, device,
                 minima_matcher, dpi, sam3_state, case_name,
                 locate_model: str,
                 locate_disabled_tools: frozenset,
                 folded_mode: bool = False):
        self.pdf_path = pdf_path
        self.sam3_processor = sam3_processor
        self.sam3_model = sam3_model
        self.device = device
        self.minima_matcher = minima_matcher
        self.dpi = dpi
        self.locate_model: str = locate_model
        # Production ships the locate sub-agent with `place` only — the
        # locate-stage ablation showed 1-tool ≈ 6-tool in IoU (Δmean = +0.001
        # on the 11 cross-1km regression-risk cases), and dropping the 5
        # other tools shrinks the prompt + tool schema sent to the LLM.
        #
        # The other 5 tool wrappers + the factory pattern are RETAINED in
        # geoplanagent.agents.locate for paper-ablation reproducibility — the
        # ablation harness calls run_locate(disabled_tools=…) directly with
        # the LOO/min_N kits. Override via benchmark_runner's
        # --locate-disabled-tools to run those kits in production too.
        self.locate_disabled_tools: frozenset = locate_disabled_tools
        # Ablation flag: when True the worker is also responsible for
        # PDFInfo extraction (no separate reader phase). Drives the
        # system_prompt branch, the submit_pdf_info tool gate, and the
        # validator's pdf_info-empty check.
        self.folded_mode: bool = folded_mode
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
        # Per-invocation token telemetry from the locate sub-agent. Each
        # propose_centers call appends one dict:
        #   {request_tokens: int, response_tokens: int,
        #    generation_id: Optional[str]}
        # Collected here (rather than aggregated inline) so the audit
        # script can attribute per-call costs to the right area_group
        # if needed and so the cost telemetry stays additive across
        # re-invocations within the same case.
        self.locate_calls: List[Dict[str, Any]] = []
        # Each match attempt covers one area_group; commit_match references by id.
        self.match_attempts: Dict[int, Dict[str, Any]] = {}
        self._match_attempt_counter: int = 0
        # area_group → committed candidate_id. Multi-area docs accumulate;
        # current_result["geojson"] is the union across groups.
        self.committed_groups: Dict[int, int] = {}
        self.match_at_budget: int = 5


# Page-of-interest helpers

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


# Status codes that are typically transient and worth retrying. 400 is
# included because OpenRouter routinely surfaces upstream Gemini hiccups
# (rate limit, model overload, transient safety-check backend failures)
# as a generic 400 with body "Provider returned error"; 413 (payload too
# large, e.g. an oversized image) is likewise retried.
_RETRYABLE_STATUS = {400, 408, 413, 425, 429, 500, 502, 503, 504}


def _is_retryable_http_error(exc: Exception) -> bool:
    """True if this exception looks like a transient OpenRouter/provider hiccup."""
    try:
        from pydantic_ai.exceptions import ModelHTTPError
    except Exception:
        return False
    if not isinstance(exc, ModelHTTPError):
        return False
    s = str(exc)
    m = re.search(r"status_code:\s*(\d+)", s)
    if not m:
        return False
    return int(m.group(1)) in _RETRYABLE_STATUS


# Substring forms of the same canonical status set, for raw transport
# exceptions that aren't wrapped in ModelHTTPError.
_TRANSIENT_HTTP_MARKERS = tuple(
    f"status_code: {s}" for s in sorted(_RETRYABLE_STATUS)
)


def is_transient_http_error(e: Exception) -> bool:
    """Substring variant of _is_retryable_http_error for raw transport
    exceptions that aren't wrapped in ModelHTTPError (used by the locate
    sub-agent's own retry loop)."""
    s = str(e).lower()
    return any(m in s for m in (x.lower() for x in _TRANSIENT_HTTP_MARKERS))


def _run_sync_with_retry(agent_obj, *args, max_retries: int = 2,
                          backoff_s: float = 5.0, label: str = "agent",
                          **kwargs):
    """Wrap Agent.run_sync with retries on transient HTTP errors.

    Non-retryable errors (auth, bad input, ModelRetry / UnexpectedModelBehavior)
    are re-raised immediately so we don't waste cycles.
    """
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



MODEL_ALIASES = {
    "claude-opus": "anthropic/claude-opus-4.7",
    "gpt-5.5-pro": "openai/gpt-5.5-pro",
    "gemini-pro": "google/gemini-3.1-pro-preview",
    "gemini-flash": "google/gemini-3-flash-preview",
}


def resolve_model_name(name: str) -> str:
    """Map a short alias (gemini-flash, claude-opus, …) to a full
    OpenRouter model identifier. Already-qualified IDs (containing "/")
    or unknown aliases pass through unchanged."""
    return MODEL_ALIASES.get(name, name)


def resolve_model(name: str) -> OpenRouterModel:
    """Convenience: resolve alias + construct OpenRouterModel."""
    return OpenRouterModel(resolve_model_name(name))


# Ground metres per zoom-0 tile pixel at the equator: 2π·6378137 / 256.
WEB_MERCATOR_C: float = 156543.03

# Spherical Earth (~0.3% off vs ellipsoid at UK scale).
_EARTH_R_KM = 6371.0


def haversine_km(lat1: float, lon1: float,
                  lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points (haversine, R=6371 km)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2.0 * _EARTH_R_KM * math.asin(min(1.0, math.sqrt(a)))


def tile_mpp(lat: float, zoom: int) -> float:
    """Ground metres per pixel for a Web-Mercator tile at (lat, zoom)."""
    return WEB_MERCATOR_C * math.cos(math.radians(lat)) / (2 ** zoom)


def compute_map_mpp(scale_ratio, dpi: int = 200):
    """Ground metres per pixel for a 1:scale_ratio map rendered at dpi. None passes through."""
    if scale_ratio is None:
        return None
    mm_per_px = 25.4 / dpi
    return mm_per_px / 1000.0 * scale_ratio


def best_zoom_for_scale(map_mpp, lat: float):
    """OSM zoom in [15, 19] whose pixel scale most closely matches map_mpp at lat."""
    if map_mpp is None:
        return None
    z = math.log2(WEB_MERCATOR_C * math.cos(math.radians(lat)) / map_mpp)
    return max(15, min(19, round(z)))


def latlon_to_global_tile_pixel(
    lat: float, lon: float, zoom: int, tile_size: int = 256,
) -> Tuple[float, float]:
    """WGS84 → global Web-Mercator tile-pixel (px, py). Origin = top-left of zoom grid."""
    n = 2 ** zoom
    lat_rad = math.radians(lat)
    px = (lon + 180.0) / 360.0 * n * tile_size
    py = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad))
          / math.pi) / 2.0 * n * tile_size
    return px, py


def osm_pixel_to_latlon(
    px: float, py: float, zoom: int, tx_min: int, ty_min: int,
    tile_size: int = 256,
) -> Tuple[float, float]:
    """Inverse of latlon_to_global_tile_pixel, offset by canvas origin (tx_min, ty_min)."""
    n = 2 ** zoom
    global_px = tx_min * tile_size + px
    global_py = ty_min * tile_size + py
    lon = global_px / (n * tile_size) * 360 - 180
    lat = math.degrees(math.atan(math.sinh(
        math.pi * (1 - 2 * global_py / (n * tile_size)))))
    return lat, lon


def latlon_to_tile_xy(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """WGS84 → integer (tx, ty) tile indices at zoom."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad))
            / math.pi) / 2.0 * n)
    return x, y


N_FOLDS = 5


def normalise_case_name(case_name: str) -> str:
    """Map a case name to the safe-filename form used in fold_assignment.json.

    The dataset builder replaces ':' and '/' with '_', so e.g. the eval
    folder '12:00114:ART4' is keyed as '12_00114_ART4'.
    """
    return (case_name or "").replace(":", "_").replace("/", "_")


def resolve_fold(case_name: str, fold_assignment: dict,
                 available_folds: set[int]) -> int:
    """Return the fold whose checkpoint should serve `case_name`.

    Lookup order: exact key, then the normalised safe-filename form, then
    page-suffixed keys (multi-page cases are stored per page, e.g.
    'A108P_p4', but the benchmark asks for 'A108P'). Cases outside the
    training pool were never seen by any fold, so any checkpoint is fine;
    we pick min(available_folds) for determinism.
    """
    norm = normalise_case_name(case_name)
    fold = fold_assignment.get(case_name)
    if fold is None:
        fold = fold_assignment.get(norm)
    if fold is None:
        # Multi-page cases: pages of one document always share a fold
        # (the split is grouped by case), so the first hit is enough.
        prefix = norm + "_p"
        for key, val in fold_assignment.items():
            if key.startswith(prefix) and key[len(prefix):].isdigit():
                fold = val
                break
    if fold is None or fold not in available_folds:
        return min(available_folds)
    return int(fold)
