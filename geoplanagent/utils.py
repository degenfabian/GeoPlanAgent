"""Utility functions for the GeoPlanAgent pipeline."""

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic_ai import BinaryContent, ModelRetry
from pydantic_ai.models.openrouter import OpenRouterModel

from geoplanagent.schemas import BoundaryOutcome


class AgentState:
    """Mutable state shared across all tool calls."""

    def __init__(
        self,
        pdf_path,
        minima_matcher,
        dpi,
        sam3_state,
        case_name,
        locate_model_name: str,
        folded_mode: bool = False,
    ):
        """The shared state for one case.

        Built once per document and handed to every tool, so each tool reads
        the models, paths, and options it needs from one place.

        Args:
            pdf_path: path to the PDF of the current Article 4 planning application.
            minima_matcher: shared MINIMA-LoFTR matcher (loaded once, reused
                across all cases).
            dpi: render resolution for map pages.
            sam3_state: the loaded SAM3 segmentation model and everything it
                needs to run, bundled together by the loader.
            case_name: case folder name; drives k-fold adapter routing and
                telemetry. Derived from ``pdf_path`` when not given.
            locate_model_name: model id for the locate sub-agent.
            folded_mode: ablation flag — when True the worker also extracts
                PDFInfo (no separate reader phase), which gates the system
                prompt, the submit_pdf_info tool, and the pdf_info-empty
                validator.
        """
        self.pdf_path = pdf_path
        self.minima_matcher = minima_matcher
        self.dpi = dpi
        self.sam3_state: Optional[Dict[str, Any]] = sam3_state
        self.case_name: Optional[str] = case_name
        if self.case_name is None and pdf_path:
            try:
                self.case_name = Path(pdf_path).parent.name
            except Exception:
                pass
        self.locate_model_name: str = locate_model_name
        self.folded_mode: bool = folded_mode

        # Rendered match-page images, keyed by 1-based page number.
        self.rendered_pages: Dict[int, np.ndarray] = {}

        # SAM3 masks, keyed by page. Lazily computed in match_at on first need.
        self.sam_masks_by_page: Dict[int, np.ndarray] = {}

        self.current_result: dict = {}

        self.accepted = False
        self.accept_reason = ""
        # Hashes of (tool, args) already issued this case; _dedup_check
        # blocks an exact repeat with a ModelRetry.
        self.seen_call_keys: set = set()
        # Count of committed matches (incremented in commit_match); a
        # metrics.json telemetry field.
        self.position_calls: int = 0

        self.pdf_info: Dict[str, Any] = {}
        self.rotation_checked: bool = False
        # The current BoundaryOutcome. The critic can update this, so it may
        # differ from the worker's original result.output.
        self.last_output: Optional["BoundaryOutcome"] = None

        # Locate sub-agent's picked candidates (one entry usually).
        self.proposed_centers: List[Dict[str, Any]] = []
        # Full message history from the most recent run_locate call.
        # When the worker re-invokes propose_centers, this is passed back
        # to run_locate as `prior_messages` so the locate sub-agent sees
        # its previous reasoning + tool calls + pick.
        self.locate_message_history: List[Any] = []
        # One {request_tokens, response_tokens, generation_id} dict per
        # locate call; summed into agent_stats for cost telemetry.
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
    pages = (state.pdf_info or {}).get("map_pages") or []
    return int(pages[0]) if pages else None


def committed_primary_page(state: "AgentState") -> Optional[int]:
    """Page of the worker's committed primary group, else the default match page."""
    current_result = state.current_result or {}
    per_group = current_result.get("per_group") or []
    if per_group:
        requested = current_result.get("requested_group")
        primary = next(
            (group for group in per_group if group.get("area_group") == requested),
            per_group[0],
        )
        page = primary.get("page")
        if page is not None:
            return int(page)
    return primary_match_page(state)


def resize_for_api(img: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """Resize image so largest dimension is max_dim."""
    height, width = img.shape[:2]
    if max(height, width) <= max_dim:
        return img
    scale = max_dim / max(height, width)
    return cv2.resize(img, (int(width * scale), int(height * scale)))


def _img_to_binary(img: np.ndarray) -> BinaryContent:
    """Convert numpy BGR image to PydanticAI BinaryContent."""
    _, encoded = cv2.imencode(".png", resize_for_api(img))
    return BinaryContent(data=encoded.tobytes(), media_type="image/png")


def _dedup_check(state: "AgentState", tool_name: str, args: dict) -> None:
    """Raise ModelRetry if this exact tool+args was already called."""
    key = (
        tool_name
        + ":"
        + hashlib.md5(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()
    )
    if key in state.seen_call_keys:
        raise ModelRetry(
            "You already called this tool with the same arguments. "
            "Try different arguments or respond with DONE."
        )
    state.seen_call_keys.add(key)


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
    status_match = re.search(r"status_code:\s*(\d+)", str(exc))
    if not status_match:
        return False
    return int(status_match.group(1)) in _RETRYABLE_STATUS


# Substring forms of the same canonical status set, for raw transport
# exceptions that aren't wrapped in ModelHTTPError.
_TRANSIENT_HTTP_MARKERS = tuple(f"status_code: {s}" for s in sorted(_RETRYABLE_STATUS))


def is_transient_http_error(e: Exception) -> bool:
    """Substring variant of _is_retryable_http_error for raw transport
    exceptions that aren't wrapped in ModelHTTPError (used by the locate
    sub-agent's own retry loop)."""
    message = str(e).lower()
    return any(marker.lower() in message for marker in _TRANSIENT_HTTP_MARKERS)


def _run_sync_with_retry(
    agent_obj, *args, max_retries: int = 2, backoff_s: float = 5.0, label: str = "agent", **kwargs
):
    """Wrap Agent.run_sync with retries on transient HTTP errors.

    Non-retryable errors (auth, bad input, ModelRetry / UnexpectedModelBehavior)
    are re-raised immediately so we don't waste cycles.
    """
    for attempt in range(max_retries + 1):
        try:
            return agent_obj.run_sync(*args, **kwargs)
        except Exception as error:
            if not _is_retryable_http_error(error) or attempt == max_retries:
                raise
            wait_s = backoff_s * (2**attempt)
            print(
                f"  {label}: transient HTTP error (attempt "
                f"{attempt + 1}/{max_retries + 1}): {str(error)[:140]}"
                f" — retrying in {wait_s:.0f}s"
            )
            time.sleep(wait_s)


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


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points (haversine, R=6371 km)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2.0 * _EARTH_R_KM * math.asin(min(1.0, math.sqrt(a)))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres — a metres wrapper over haversine_km so
    metric code doesn't hand-multiply by 1000."""
    return haversine_km(lat1, lon1, lat2, lon2) * 1000.0


def tile_mpp(lat: float, zoom: int) -> float:
    """Ground metres per pixel for a Web-Mercator tile at (lat, zoom)."""
    return WEB_MERCATOR_C * math.cos(math.radians(lat)) / (2**zoom)


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
    zoom = math.log2(WEB_MERCATOR_C * math.cos(math.radians(lat)) / map_mpp)
    return max(15, min(19, round(zoom)))


def osm_pixel_to_latlon(
    px: float,
    py: float,
    zoom: int,
    tx_min: int,
    ty_min: int,
    tile_size: int = 256,
) -> Tuple[float, float]:
    """Global Web-Mercator tile-pixel → WGS84, offset by canvas origin (tx_min, ty_min)."""
    n = 2**zoom
    global_px = tx_min * tile_size + px
    global_py = ty_min * tile_size + py
    lon = global_px / (n * tile_size) * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * global_py / (n * tile_size)))))
    return lat, lon


def latlon_to_tile_xy(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """WGS84 → integer (tx, ty) tile indices at zoom."""
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


N_FOLDS = 5


def normalise_case_name(case_name: str) -> str:
    """Map a case name to the safe-filename form used in fold_assignment.json.

    The dataset builder replaces ':' and '/' with '_', so e.g. the eval
    folder '12:00114:ART4' is keyed as '12_00114_ART4'.
    """
    return (case_name or "").replace(":", "_").replace("/", "_")


def resolve_fold(case_name: str, fold_assignment: dict, available_folds: set[int]) -> int:
    """Return the fold whose checkpoint should serve `case_name`.

    Lookup order: exact key, then the normalised safe-filename form, then
    page-suffixed keys (multi-page cases are stored per page, e.g.
    'A108P_p4', but the benchmark asks for 'A108P'). Cases outside the
    training pool were never seen by any fold, so any checkpoint is fine;
    we pick min(available_folds) for determinism.
    """
    normalised = normalise_case_name(case_name)
    fold = fold_assignment.get(case_name)
    if fold is None:
        fold = fold_assignment.get(normalised)
    if fold is None:
        # Multi-page cases: pages of one document always share a fold
        # (the split is grouped by case), so the first hit is enough.
        prefix = normalised + "_p"
        for key, fold_value in fold_assignment.items():
            if key.startswith(prefix) and key[len(prefix) :].isdigit():
                fold = fold_value
                break
    if fold is None or fold not in available_folds:
        return min(available_folds)
    return int(fold)
