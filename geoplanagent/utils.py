"""Utility functions for the GeoPlanAgent pipeline."""

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic_ai import BinaryContent, ModelRetry
from pydantic_ai.models.openrouter import OpenRouterModel

from geoplanagent.schemas import BoundaryOutcome

MODEL_ALIASES = {
    "claude-opus": "anthropic/claude-opus-4.7",
    "gpt-5.5-pro": "openai/gpt-5.5-pro",
    "gemini-pro": "google/gemini-3.1-pro-preview",
    "gemini-flash": "google/gemini-3-flash-preview",
}


# Web Mercator ground resolution: metres per pixel at zoom 0 on the
# equator = equatorial circumference (2π · 6_378_137 m, the WGS84
# equatorial radius) / 256 px per tile. At zoom z and latitude lat the
# resolution is WEB_MERCATOR_C · cos(lat) / 2**z — used to pick the OS
# tile zoom matching a printed map's scale (see best_zoom_for_scale).
WEB_MERCATOR_C: float = 156543.03

# Mean spherical Earth radius (km) for haversine great-circle distances.
# A sphere vs the WGS84 ellipsoid is ~0.3% off — negligible at UK scale.
_EARTH_R_KM = 6371.0

# Millimetres per inch — turns a render DPI (dots per inch) into a
# physical mm-per-pixel in compute_map_mpp.
MM_PER_INCH = 25.4

# k-fold cross-validation folds: each case is segmented by the SAM3
# adapter whose fold excluded it from training (no train/test leakage).
N_FOLDS = 5


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

        # The current extraction result, rebuilt by _recompute_current_result
        # on every commit. geojson is the union of every committed group's
        # polygon. affine_H / tile_info / match_info are single-page values 
        # (can't be unioned), so they come from the primary (highest-inlier)
        # group for visualisation only; the other groups still live in per_group,
        # and total_inliers sums n_inliers across all of them.
        self.current_result: dict = {}

        self.accepted = False
        self.accept_reason = ""
        # Hashes of (tool, args) already issued this case; dedup_check
        # blocks an exact repeat with a ModelRetry.
        self.seen_call_keys: set = set()
        # Count of committed matches (incremented in commit_match); a
        # metrics.json telemetry field.
        self.n_commits: int = 0

        self.pdf_info: Dict[str, Any] = {}
        self.rotation_checked: bool = False
        # The worker's structured verdict (status + reasoning + inlier /
        # rotation telemetry), i.e. result.output. The geometry lives in
        # current_result, not here. The critic can update this, so it may
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


def resize_for_api(img: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """Resize image so largest dimension is max_dim."""
    height, width = img.shape[:2]
    if max(height, width) <= max_dim:
        return img
    scale = max_dim / max(height, width)
    return cv2.resize(img, (int(width * scale), int(height * scale)))


def img_to_binary(img: np.ndarray) -> BinaryContent:
    """Convert numpy BGR image to PydanticAI BinaryContent."""
    _, encoded = cv2.imencode(".png", resize_for_api(img))
    return BinaryContent(data=encoded.tobytes(), media_type="image/png")


def dedup_check(state: "AgentState", tool_name: str, args: dict) -> None:
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


def is_http_error(e: Exception) -> bool:
    """True if this is a provider HTTP error. In this case, we will retry the request."""
    return "status_code:" in str(e).lower()


def run_sync_with_retry(
    agent_obj,
    user_prompt,
    max_retries: int = 2,
    backoff_s: float = 5.0,
    label: str = "agent",
    **run_kwargs,
):
    """
    Wrap Agent.run_sync with retries on transient HTTP errors.

    ``user_prompt`` and ``run_kwargs`` (model, deps, usage_limits,
    message_history, …) are passed straight through to
    ``agent_obj.run_sync``. Non-retryable errors (auth, bad input,
    ModelRetry / UnexpectedModelBehavior) are re-raised immediately so we
    don't waste cycles.
    """
    for attempt in range(max_retries + 1):
        try:
            return agent_obj.run_sync(user_prompt, **run_kwargs)
        except Exception as error:
            if not is_http_error(error) or attempt == max_retries:
                raise
            # Increases the wait time exponentially.
            wait_s = backoff_s * (2**attempt)
            print(
                f"  {label}: transient HTTP error (attempt "
                f"{attempt + 1}/{max_retries + 1}): {str(error)[:140]}"
                f" — retrying in {wait_s:.0f}s"
            )
            time.sleep(wait_s)


def resolve_model_name(name: str) -> str:
    """Map a short alias (gemini-flash, claude-opus, …) to a full
    OpenRouter model identifier. Already-qualified IDs (containing "/")
    or unknown aliases pass through unchanged."""
    return MODEL_ALIASES.get(name, name)


def resolve_model(name: str) -> OpenRouterModel:
    """Convenience: resolve alias + construct OpenRouterModel."""
    return OpenRouterModel(resolve_model_name(name))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points (haversine, R=6371 km)."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2.0 * _EARTH_R_KM * math.asin(min(1.0, math.sqrt(a)))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres — a metres wrapper over haversine_km."""
    return haversine_km(lat1, lon1, lat2, lon2) * 1000.0


def tile_mpp(lat: float, zoom: int) -> float:
    """Ground metres per pixel for a Web-Mercator tile at (lat, zoom)."""
    return WEB_MERCATOR_C * math.cos(math.radians(lat)) / (2**zoom)


def compute_map_mpp(scale_ratio, dpi: int = 200):
    """Ground metres per pixel for a 1:scale_ratio map rendered at dpi. None passes through."""
    if scale_ratio is None:
        return None
    # paper mm per pixel → ground metres (× scale_ratio), with mm→m (÷1000).
    mm_per_px = MM_PER_INCH / dpi
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



def normalise_case_name(case_name: str) -> str:
    """Map a case name to safe filenames (no ":" or "/").

    The dataset builder replaces ':' and '/' with '_', so e.g. the eval
    folder '12:00114:ART4' is keyed as '12_00114_ART4'.
    """
    return (case_name or "").replace(":", "_").replace("/", "_")


def resolve_fold(case_name: str, fold_assignment: dict, available_folds: set[int]) -> int:
    """Return the fold whose checkpoint should serve `case_name`.

    Keys and query are both normalised (':' / '/' → '_'), so the
    filename-safety swap never matters — one lookup covers every form.
    Multi-page cases are keyed per page ('A108P_p4'), and pages of one
    case share a fold, so a page-suffixed hit is enough. A case the
    training pool never saw falls back to min(available_folds): any
    checkpoint is fine, and min is deterministic.
    """
    by_fold = {normalise_case_name(k): v for k, v in fold_assignment.items()}
    key = normalise_case_name(case_name)
    # NB: fold 0 is valid, so test `is None`, never `a or b` (0 is falsy).
    fold = by_fold.get(key)
    if fold is None:
        prefix = key + "_p"
        fold = next(
            (v for k, v in by_fold.items() if k.startswith(prefix) and k[len(prefix) :].isdigit()),
            None,
        )
    if fold is None or fold not in available_folds:
        return min(available_folds)
    return int(fold)
