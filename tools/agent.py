"""Unified planning boundary extraction agent.

PydanticAI-based agent with 10 tools that handles the full pipeline:
PDF reading → geocoding → positioning → boundary extraction → verification.

The LLM reads the PDF directly, reasons about it, and makes tool calls.
No pre-computed analysis step — the agent IS the reasoning engine.

Tools:
    1. render_page — render a PDF page as an image
    2. geocode — look up coordinates (postcode or grid_ref only)
    3. position_boundary — MINIMA sliding-window matching
    4. extract_boundary — SAM3 boundary segmentation
    5. project_boundary — project mask to GeoJSON via affine
    6. accumulate_boundary — save current page's result for multi-page maps
    7. verify_position — visual inspection on OS tiles
    8. lookup_district — get district boundary from OSM
    9. visualize — show boundary overlay + positioned GeoJSON
   (rotate_map removed — rotation is reader-detected and pre-applied by
   the wrapper)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import cv2
import fitz
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, ModelRetry, RunContext, ToolReturn
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.usage import UsageLimits

load_dotenv()


# ── Structured Outputs (Pydantic models enforced via pydantic-ai) ─────────

class PDFInfo(BaseModel):
    """Structured output for the reader agent. pydantic-ai enforces the schema;
    the model physically cannot return a string — it must fill these fields."""
    site_address: str = Field(
        default="",
        description="The SITE address (location of the planning boundary). "
                    "Prefer 'Site Address', 'Location', or 'Land at...' fields. "
                    "IGNORE council/agent/architect office addresses."
    )
    postcodes: List[str] = Field(
        default_factory=list,
        description="All UK postcodes found (format 'XX1 2YZ')."
    )
    grid_refs: List[str] = Field(
        default_factory=list,
        description="OS grid references (e.g. 'TG 210 080')."
    )
    scale: Optional[str] = Field(
        default=None,
        description="Printed map scale (e.g. '1:2500')."
    )
    map_pages: List[int] = Field(
        default_factory=list,
        description="1-based page numbers containing maps. Include ALL map pages."
    )
    n_pages: int = 0
    road_names: List[str] = Field(default_factory=list)
    place_names: List[str] = Field(default_factory=list)
    boundary_color: Optional[str] = Field(
        default=None,
        description="Color of the planning boundary line (red, blue, pink, etc.)."
    )
    boundary_description: str = ""
    is_district_wide: bool = Field(
        default=False,
        description="TRUE if the boundary covers an ENTIRE borough/district/ward/"
                    "parish/conservation area. Patterns that trigger TRUE: "
                    "'Land within the X of Y', 'Various sites across X', "
                    "'The X Conservation Area', 'Land in the Urban District of X'. "
                    "When unsure, prefer true — downstream falls through if lookup fails."
    )
    district_name: Optional[str] = Field(
        default=None,
        description="If is_district_wide, the OSM-format name with 'UK' suffix. "
                    "Provide '|' alternates if ambiguous (e.g. 'Dover District, Kent, UK | Dover, Kent, UK')."
    )
    multiple_map_areas: bool = Field(
        default=False,
        description="True if different map pages show different geographic areas. "
                    "Set true whenever map_pages has more than one entry unless all "
                    "pages are zoomed views of the same site."
    )
    map_rotation: int = Field(
        default=0,
        description="Rotation in degrees CLOCKWISE needed to make the map's north "
                    "point UP. Set 0 if the map is already correctly oriented. "
                    "Set 90 if the map is rotated 90° counterclockwise (i.e., "
                    "north points right and you'd rotate it 90° clockwise to fix it). "
                    "Set 180 if upside-down. Set 270 if rotated 90° clockwise "
                    "(north points left). Look at the north arrow if visible, "
                    "or the orientation of place-name labels and the scale bar. "
                    "Most modern maps are 0; planning maps and historic OS sheets "
                    "can be 90, 180, or 270."
    )
    notes: str = ""


class CenterInput(BaseModel):
    """A geocoded search center for position_boundary. All three fields required."""
    name: str = Field(description="A short label for this center (e.g. 'postcode NR15 2XE').")
    lat: float = Field(description="Latitude in decimal degrees (e.g. 52.4774).")
    lon: float = Field(description="Longitude in decimal degrees (e.g. 1.3854).")


class BoundaryOutcome(BaseModel):
    """Structured output for the worker agent. Includes mandatory checklist
    fields so the output_validator can enforce that required tools were called."""
    status: Literal["accepted", "rejected_low_quality", "rejected_visual_mismatch",
                    "rejected_no_match", "district_lookup"] = Field(
        description="accepted = produce GeoJSON; rejected_* = no GeoJSON; "
                    "district_lookup = boundary from OSM district fallback."
    )
    final_n_inliers: int = Field(
        default=0,
        description="n_inliers from the final position_boundary call (0 if none)."
    )
    verify_position_called: bool = Field(
        default=False,
        description="Did you call verify_position for this result? "
                    "MUST be true if final_n_inliers is in 25-100 band."
    )
    visual_check_notes: str = Field(
        default="",
        description="If you called verify_position, describe whether OS tile features "
                    "(roads, buildings, settlement shape) matched the planning map. "
                    "Required when final_n_inliers is 25-100 and status=accepted."
    )
    rotation_checked: bool = Field(
        default=False,
        description="(Auto-set by wrapper.) True when the reader detected and "
                    "applied a rotation. You don't manage this — leave default."
    )
    pages_accumulated: int = Field(
        default=0,
        description="Number of times accumulate_boundary() was called. "
                    "MUST equal len(map_pages)-1 when map_pages has >1 entry and "
                    "status=accepted (because the last page's geojson is still "
                    "in current_result, not yet accumulated)."
    )
    reasoning: str = Field(
        description="One-paragraph summary of what you did and why the result is correct."
    )
    reject_reason: Optional[str] = Field(
        default=None,
        description="If status starts with 'rejected_', explain why in 1-2 sentences."
    )


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


def _dedup_check(state: AgentState, tool_name: str, args: dict) -> None:
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

    for ring in coord_rings:
        pts = []
        for lon_c, lat_c in ring:
            lat_rad = math.radians(lat_c)
            n = 2 ** zoom
            px = ((lon_c + 180) / 360 * n - tx_min) * tile_size
            py = ((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad))
                   / math.pi) / 2 * n - ty_min) * tile_size
            pts.append([int(px), int(py)])
        if len(pts) >= 3:
            cv2.polylines(tile_bgr, [np.array(pts, dtype=np.int32)],
                          True, (0, 0, 255), 2)
    return tile_bgr


def _get_instance_masks(map_crop_path, processor, model, device,
                        query="planning boundary", top_k=5):
    """Extract individual instance masks from SAM3."""
    from tools.sam3_boundary import extract_candidates
    candidates = extract_candidates(
        map_crop_path, processor, model, device, query=query, top_k=top_k)
    return [c["mask"] for c in candidates]


# ── Agent State ─────────────────────────────────────────────────────────────

class AgentState:
    """Mutable state shared across all tool calls."""

    def __init__(self, pdf_path, sam3_processor, sam3_model, device,
                 minima_matcher, dpi=200):
        self.pdf_path = pdf_path
        self.sam3_processor = sam3_processor
        self.sam3_model = sam3_model
        self.device = device
        self.minima_matcher = minima_matcher
        self.dpi = dpi

        # Set by render_page
        self.map_img: Optional[np.ndarray] = None
        self.map_crop_path: Optional[str] = None

        # Set by extract_boundary
        self.current_mask: Optional[np.ndarray] = None
        self.instance_masks: List[np.ndarray] = []

        # Set by position_boundary
        self.current_result: dict = {}
        self.centers: List[tuple] = []
        self.scale_ratio: Optional[float] = None

        # Multi-page: accumulated GeoJSON from previous pages
        self.accumulated_geojson: List[dict] = []

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
        self.pages_accumulated: int = 0
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


# ── Phase 1: PDF Reader Agent ──────────────────────────────────────────────

_reader_agent = Agent(
    "test",  # placeholder, overridden at runtime
    output_type=PDFInfo,
    retries=2,
    output_retries=2,
    instructions="""You are a UK planning document reader. Read every page of the PDF
carefully and populate the PDFInfo schema.

FIELD GUIDANCE (field descriptions in the schema are authoritative; these are
additional rules):

- map_pages: list ALL pages that contain a site/location map (1-based).
  Maps are usually near the end. Include every map page, even if there are multiple.

- postcodes: extract ALL UK postcodes. Look in site address, map title blocks,
  form fields, tables, and application metadata. Postcodes are the strongest
  geocoding signal — be thorough.

- grid_refs: OS grid references on map edges (e.g. "TG 210 080", "TR 34 SE").

- is_district_wide: true if the planning boundary covers an entire
  administrative district, false otherwise.
- district_name: if is_district_wide, the OSM-format name with "UK" suffix.
  Provide "|"-separated alternates if ambiguous.

- site_address: the SITE address (location of the boundary). Prefer
  "Site Address", "Location", or "Land at..." fields. IGNORE council/agent/
  architect office addresses. For multi-property documents, use the overall
  area name.

- multiple_map_areas: TRUE whenever map_pages has >1 entry unless the pages
  are zoomed views of the same exact site.

- map_rotation: 0 / 90 / 180 / 270, the clockwise rotation needed to make
  north point UP on the map. Check (a) the north arrow if drawn, (b) the
  orientation of place-name labels (should read left-to-right when correct),
  (c) the scale bar (usually horizontal at the bottom). Old planning maps
  often have rotated layouts to fit the page. Default 0; only set non-zero
  if you can clearly see the map needs rotating.
""",
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
    # the map / accumulated pages. Override the schema fields with real state.
    if out.verify_position_called != state.verify_position_called:
        out.verify_position_called = state.verify_position_called
    if out.rotation_checked != state.rotation_checked:
        out.rotation_checked = state.rotation_checked
    if out.pages_accumulated != state.pages_accumulated:
        out.pages_accumulated = state.pages_accumulated
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
            "Cannot accept: no successful position_boundary call has produced "
            "a result. Either run positioning to completion, or set status to "
            "'rejected_no_match' with a reject_reason."
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

    # Multi-page docs require accumulate_boundary for every page except the last.
    expected_pages = len(state.pdf_info.get("map_pages") or [])
    if expected_pages > 1:
        required_accumulations = expected_pages - 1
        if state.pages_accumulated < required_accumulations:
            raise ModelRetry(
                f"This document has {expected_pages} map pages but you have only "
                f"called accumulate_boundary {state.pages_accumulated} times "
                f"(need {required_accumulations}). You must process EACH map page: "
                f"render_page → geocode → position → extract → project → "
                f"accumulate_boundary, then render_page on the next map page. "
                f"Only submit status='accepted' after the LAST page's pipeline."
            )

    return out


# ── System Prompt ───────────────────────────────────────────────────────────

@_agent.system_prompt
def build_system_prompt(ctx: RunContext[AgentState]) -> str:
    return """Geographic boundary extractor for UK planning documents.

INPUT: PDFInfo summary + rendered first map page.
OUTPUT: a BoundaryOutcome (the output_validator enforces preconditions).

HARD RULES (validator will reject your submission if violated):
• 25 ≤ final_n_inliers ≤ 100 AND status="accepted" → you MUST call verify_position()
  and fill visual_check_notes with ≥20 chars describing feature comparison.
• len(map_pages) > 1 AND status="accepted" → you MUST call accumulate_boundary()
  (N-1) times — once per page except the last.
• status starts with "rejected_" → reject_reason must be non-empty.
• status="district_lookup" → lookup_district() must have succeeded.
The validator reads real tool-call state, so don't misreport flags.

WORKFLOW:
1. If PDFInfo.is_district_wide: call lookup_district(district_name). On
   success → submit status="district_lookup" and done.

2. If len(map_pages) > 1: process each page (steps 3-7), then accumulate_boundary,
   then render_page(next), repeat. Only submit after the LAST page is complete.

3. Geocoding is mostly AUTOMATIC. position_boundary internally geocodes
   PDFInfo.postcodes, grid_refs, place_names, and site_address (via
   postcodes.io, OS Open Names, Wikidata, Nominatim) and adds them as
   centers. You only need to call geocode() YOURSELF if you spot a postcode
   or grid reference on the rendered map IMAGE that PDFInfo doesn't already
   contain. Pass the lat/lon back via extra_centers.

4. position_boundary with optional extra_centers, scale_ratio (parse from
   PDFInfo.scale: "1:2500" → 2500), and road_names. Thresholds:
     ≥100 inliers: good.
     25-100: borderline — verify_position is MANDATORY.
     <25: retry with NEW geocoding, or submit status="rejected_no_match".
   Max 3 position_boundary calls.

5. verify_position (when borderline): inspect the OS tile. If roads/settlement/
   shape match the planning map → proceed. If clearly different →
   submit status="rejected_visual_mismatch" with reject_reason.

6. extract_boundary(mode="instance"): first without select_indices (see candidates),
   then with select_indices=[...] to combine. Use PDFInfo.boundary_color in query.

7. project_boundary to produce GeoJSON.

8. Submit BoundaryOutcome. Fields verify_position_called / rotation_checked /
   pages_accumulated are auto-overwritten from state — leave them at defaults.

NOTE: rotation is handled upstream. The reader detected map_rotation and
the wrapper pre-rotated the map BEFORE you saw it. You see a north-up map.
Do not try to rotate it.

RULES:
• No duplicate tool calls with same args.
• Geocoding doesn't position — you must call position_boundary afterwards.
• If stuck, reject cleanly rather than looping."""


# ── Tool 1: render_page ────────────────────────────────────────────────────

@_agent.tool
def render_page(ctx: RunContext[AgentState], page: int) -> ToolReturn:
    """Render a page from the planning PDF as an image.

    Use this after reading the PDF to render the page containing the site/location
    map. The map is usually on the LAST page or near the end. Use 1-based page
    numbering (first page = 1).

    The rendered image becomes the working map for all subsequent tools
    (extract_boundary, position_boundary, rotate_map, visualize).

    Args:
        page: Page number (1-based) to render.

    Returns:
        Image of the rendered page (shown to you), plus:
        {"success": true, "width": int, "height": int, "page": int}
    """
    state = ctx.deps
    _dedup_check(state, "render_page", {"page": page})

    page_idx = max(0, page - 1)  # convert 1-based to 0-based

    doc = fitz.open(state.pdf_path)
    n_pages = len(doc)
    if page_idx >= n_pages:
        doc.close()
        raise ModelRetry(
            f"Page {page} does not exist. Document has {n_pages} pages."
        )

    pix = doc[page_idx].get_pixmap(dpi=state.dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    doc.close()

    # Convert RGB to BGR for OpenCV
    if img.shape[2] == 4:  # RGBA
        map_img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        map_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    state.map_img = map_img

    # Save to temp file for SAM3
    if state.map_crop_path and os.path.exists(state.map_crop_path):
        os.unlink(state.map_crop_path)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        state.map_crop_path = tmp.name
        cv2.imwrite(tmp.name, map_img)

    h, w = map_img.shape[:2]

    return ToolReturn(
        return_value={"success": True, "width": w, "height": h, "page": page},
        content=[
            f"Rendered page {page} ({w}x{h} pixels):",
            _img_to_binary(map_img),
        ],
    )


# ── Tool 2: geocode ────────────────────────────────────────────────────────

@_agent.tool_plain
def geocode(
    type: str,
    postcode: Optional[str] = None,
    grid_ref: Optional[str] = None,
) -> dict:
    """Geocode a UK postcode or OS grid reference.

    USE THIS ONLY for postcodes or grid references YOU SEE on the map image
    that PDFInfo did NOT already extract. Place-name geocoding (villages,
    farms, conservation areas, named buildings, addresses) is fully
    AUTOMATIC — position_boundary internally queries OS Open Names,
    Wikidata, and Nominatim from PDFInfo.place_names and PDFInfo.site_address.

    You don't need to call geocode at all in most cases. Only use it when:
      - You spot a postcode on the map (small text near a building or
        the title block) that PDFInfo.postcodes is missing.
      - You see a grid reference at a map corner (e.g. "TG 21" or "TR 2638")
        that PDFInfo.grid_refs doesn't include.

    Types:
      - "postcode": UK postcode (e.g. "AL1 1BY").
      - "grid_ref": OS grid reference (e.g. "TL 1507 0672" or "TL 15 07")
        or full easting/northing like "528942 E 184544 N".

    Args:
        type: "postcode" or "grid_ref"
        postcode: For type="postcode" — UK postcode
        grid_ref: For type="grid_ref" — OS grid reference or easting/northing

    Returns:
        {"success": true, "lat": float, "lon": float, ...} or
        {"success": false, "error": str}.

        Pass the lat/lon to position_boundary(extra_centers=[
            {"name": "<your label>", "lat": ..., "lon": ...}
        ]).
    """
    if type == "postcode":
        import requests as req
        if not postcode:
            raise ModelRetry("postcode is required for type='postcode'")
        pc = postcode.strip()
        try:
            r = req.get(f"https://api.postcodes.io/postcodes/{pc}", timeout=10)
            data = r.json()
            if data.get("status") == 200 and data.get("result"):
                res = data["result"]
                return {"success": True, "lat": res["latitude"],
                        "lon": res["longitude"], "type": "postcode",
                        "admin_district": res.get("admin_district", "")}
        except Exception as e:
            return {"success": False, "error": str(e)}
        return {"success": False, "error": f"Postcode '{pc}' not found"}

    elif type == "grid_ref":
        from tools.geo_tools import os_grid_ref_to_latlon
        if not grid_ref:
            raise ModelRetry("grid_ref is required for type='grid_ref'")
        result = os_grid_ref_to_latlon(grid_ref)
        if result:
            return {"success": True, "lat": result[0], "lon": result[1],
                    "type": "grid_ref", "grid_ref": grid_ref}
        return {"success": False,
                "error": f"Could not parse grid reference '{grid_ref}'"}

    raise ModelRetry(
        f"Invalid type '{type}'. Use 'postcode' or 'grid_ref' only — "
        f"place names are auto-geocoded by position_boundary."
    )


# ── Tool 3: position_boundary ──────────────────────────────────────────────

@_agent.tool
def position_boundary(
    ctx: RunContext[AgentState],
    scale_ratio: Optional[float] = None,
    road_names: Optional[List[str]] = None,
    extra_centers: Optional[List[CenterInput]] = None,
    use_grayscale: bool = False,
    skip_sources: Optional[List[str]] = None,
) -> dict:
    """Geolocate the map by matching it against Ordnance Survey tiles.

    This runs MINIMA feature matching: it slides the map image across OS tiles at
    the geocoded locations and finds where the map features (roads, buildings) align
    with real-world map data. The output is an affine transform mapping pixels to
    lat/lon coordinates.

    IMPORTANT: You must geocode locations FIRST to give this tool approximate centers.
    Without centers, it has nowhere to search.

    Automatic geocoding sources (can be opted out via skip_sources):
      - "grid_refs_centroid": centroid of parsed OS grid refs
      - "gpkg": OS Zoomstack gazetteer on place_names
      - "wikidata": Wikidata fallback on place_names (conservation areas etc.)
      - "nominatim:addr": street-address lookup from site_address
      - "nominatim:road": each road name + inferred city

    The return value includes `centers_summary` showing which sources produced
    centers and which one won. Retry with `skip_sources=["wikidata"]` (etc.) if
    you think a particular source picked the wrong place.

    The quality metric is "n_inliers" — the number of matched feature points:
      - n_inliers >= 100: excellent positioning, very likely correct
      - n_inliers 50-100: decent positioning, probably correct
      - n_inliers < 50: positioning likely FAILED — find more geocoding clues and retry

    Args:
        scale_ratio: Map scale denominator (e.g. 2500 for "1:2500"). Look for this
            on the map — printed near the scale bar or in the title block. Common
            values: 1250, 2500, 5000, 10000, 10560, 25000. If not explicitly printed,
            you can estimate it yourself based on the map. If unsure, leave as None
            and the tool tries common scales automatically (but this is slower).
        road_names: List of road/street names visible on the map (e.g. ["Elm Road",
            "High Street"]). The tool uses these to verify the matched position:
            it queries OS GeoPackage data for roads within 1.5km and fuzzy-matches
            against your list. If the top MINIMA candidate has zero road matches but
            a lower candidate does, it will override the pick. Always provide road
            names when you can read them — it significantly improves accuracy.
        extra_centers: Additional search centers from geocoding. A list of
            CenterInput objects, each with name (str), lat (float), lon (float).
            Call geocode first to get (lat, lon) values, then pass them through
            here. Example: [{"name": "Elm St", "lat": 51.5074, "lon": -0.1278}].
        use_grayscale: If true, convert both the map and OS tiles to grayscale before
            matching. Use this when the map is black & white, sepia-tinted, or has
            an unusual colour scheme that differs from modern OS tiles.
        skip_sources: Disable specific automatic geocoding sources for this call.
            Values: "grid_refs_centroid", "gpkg", "wikidata", "nominatim:addr",
            "nominatim:road". Does NOT skip extra_centers (those are explicit).
            Useful to rerun after the centers_summary reveals a bad winner.

    Returns:
        {"success": true, "n_inliers": int, "score": float, "aspect": float,
         "center_latlon": [lat, lon], "zoom": int, "has_geojson": bool,
         "road_matches": "2/4", "road_confidence": "high",
         "centers_summary": {"n_centers": int, "sources": [...], "winning_source": str}}

        n_inliers is the key quality metric. score is a combined quality measure.
        aspect is how square the match is (1.0 = perfect, <0.5 = distorted).
    """
    state = ctx.deps
    skip = set(skip_sources or [])

    MAX_POSITION_CALLS = 3
    if state.position_calls >= MAX_POSITION_CALLS:
        mi = state.current_result.get("match_info", {})
        raise ModelRetry(
            f"You have already called position_boundary {MAX_POSITION_CALLS} times "
            f"(best: {mi.get('n_inliers', 0)} inliers). Accept the current positioning "
            f"and proceed to extract_boundary and project_boundary."
        )
    state.position_calls += 1

    _dedup_check(state, "position_boundary", {
        "scale_ratio": scale_ratio,
        "road_names": road_names,
        "extra_centers": extra_centers,
        "use_grayscale": use_grayscale,
        "skip_sources": sorted(skip),
    })

    if state.map_img is None:
        raise ModelRetry("No map image available. Call render_page first.")

    # Reset transparency log per call (but keep across centers list which is accumulated)
    state.centers_tried = []

    from tools.positioning import sliding_window_position

    # Update scale ratio if provided
    if scale_ratio is not None:
        state.scale_ratio = scale_ratio

    # Build centers list. extra_centers is List[CenterInput] — pydantic-ai
    # enforces the schema at tool-call validation, so every c has name/lat/lon.
    # Sigma=500 here is a placeholder; sliding_window_position overrides it
    # with a scale-aware value via sigma_from_scale(scale_ratio).
    centers = list(state.centers)
    # Seed centers_tried from any centers carried over from prior calls
    for c in centers:
        state.centers_tried.append({
            "source": "carryover", "name": c[0],
            "lat": c[1], "lon": c[2], "was_picked": False,
        })
    if extra_centers:
        for c in extra_centers:
            center_tuple = (c.name, c.lat, c.lon, 500)
            centers.append(center_tuple)
            if center_tuple not in state.centers:
                state.centers.append(center_tuple)
            state.centers_tried.append({
                "source": "extra_centers", "name": c.name,
                "lat": c.lat, "lon": c.lon, "was_picked": False,
            })

    # ── B: grid-ref centroid as an additional center ──────────────────────
    # If the reader extracted ≥2 parseable OS grid refs, their centroid is a
    # strong anchor. Added as one more center in the list; cross-validation
    # in sliding_window_position drops it if it's wildly inconsistent with
    # the geocoded centers (outlier rejection with max_outlier_km=5).
    pdf_grid_refs = (state.pdf_info or {}).get("grid_refs") or []
    all_parsed_refs = []  # shared with grid-ref sanity filter below
    if pdf_grid_refs:
        from tools.geo_tools import os_grid_ref_to_latlon
        for r in pdf_grid_refs:
            try:
                ll = os_grid_ref_to_latlon(r)
            except Exception:
                ll = None
            if ll is not None:
                all_parsed_refs.append(ll)
        if len(all_parsed_refs) >= 2 and "grid_refs_centroid" not in skip:
            lat_avg = sum(p[0] for p in all_parsed_refs) / len(all_parsed_refs)
            lon_avg = sum(p[1] for p in all_parsed_refs) / len(all_parsed_refs)
            gc_tuple = ("grid_refs_centroid", lat_avg, lon_avg, 500)
            if gc_tuple not in centers:
                centers.append(gc_tuple)
                if gc_tuple not in state.centers:
                    state.centers.append(gc_tuple)
                state.centers_tried.append({
                    "source": "grid_refs_centroid", "name": "grid_refs_centroid",
                    "lat": lat_avg, "lon": lon_avg, "was_picked": False,
                })
                print(f"    grid_refs_centroid added "
                      f"(from {len(all_parsed_refs)} refs): "
                      f"({lat_avg:.5f}, {lon_avg:.5f})")

    # C: OS Zoomstack gazetteer (offline, OGL v3) — place_names.
    # Disambiguates with a parent anchor so "Waterfoot" resolves to the
    # correct one. Grid-ref sanity filter rejects any hit >10km from the
    # grid-ref centroid (catches geocoder picking wrong part of UK).
    pdf_places = (state.pdf_info or {}).get("place_names") or []
    pdf_district = (state.pdf_info or {}).get("district_name") or ""
    if pdf_places and "gpkg" not in skip:
        from tools.geocoding import gpkg_place_search, _distance_m

        def _current_parent():
            if all_parsed_refs:
                plat = sum(p[0] for p in all_parsed_refs) / len(all_parsed_refs)
                plon = sum(p[1] for p in all_parsed_refs) / len(all_parsed_refs)
                return plat, plon
            if len(centers) >= 1:
                lats = sorted(c[1] for c in centers)
                lons = sorted(c[2] for c in centers)
                return lats[len(lats) // 2], lons[len(lons) // 2]
            return None, None

        ref_sanity_lat = ref_sanity_lon = None
        if len(all_parsed_refs) >= 1:
            ref_sanity_lat = sum(p[0] for p in all_parsed_refs) / len(all_parsed_refs)
            ref_sanity_lon = sum(p[1] for p in all_parsed_refs) / len(all_parsed_refs)

        added = 0
        skipped = 0
        for place in pdf_places[:6]:
            if not isinstance(place, str) or len(place.strip()) < 3:
                continue
            lower = place.lower().strip()
            if lower in ("none", "null", "n/a", "unknown", "various"):
                continue
            parent_lat, parent_lon = _current_parent()
            try:
                hits = gpkg_place_search(
                    place, parent_lat=parent_lat, parent_lon=parent_lon,
                    max_parent_distance_km=30.0, limit=1,
                )
            except Exception:
                hits = []
            if not hits:
                continue
            h = hits[0]
            if ref_sanity_lat is not None:
                d_km = _distance_m(h["lat"], h["lon"],
                                   ref_sanity_lat, ref_sanity_lon) / 1000.0
                if d_km > 10.0:
                    print(f"    gpkg_place: REJECT {place!r} -> {h['name']!r} "
                          f"({d_km:.1f}km from grid-ref centroid, >10km)")
                    skipped += 1
                    continue
            # The {type} suffix lets positioning's specificity ranker
            # deprioritise broad-area hits (District/County) when street-level
            # anchors exist.
            gc_tuple = (f"gpkg:{h['name']}({h['type']})",
                        h["lat"], h["lon"], 500)
            if gc_tuple not in centers:
                centers.append(gc_tuple)
                if gc_tuple not in state.centers:
                    state.centers.append(gc_tuple)
                state.centers_tried.append({
                    "source": "gpkg", "name": gc_tuple[0],
                    "lat": h["lat"], "lon": h["lon"], "was_picked": False,
                })
                added += 1
                print(f"    gpkg_place: added {place!r} → {h['name']!r} "
                      f"({h['type']}, {h['lat']:.5f}, {h['lon']:.5f})")
        if added or skipped:
            print(f"    gpkg_place summary: {added} added, {skipped} rejected "
                  f"(sanity)")

    # ── D: Wikidata fallback for named features Zoomstack didn't have ─────
    # Conservation areas, named buildings, historic landmarks like
    # "Belsize Park", "Colney Hall", etc. that aren't in OS Open Names
    # often live in Wikidata. Try ONCE per place_name that didn't already
    # produce a center via the Zoomstack pass above. Same parent-anchor
    # disambiguation, same grid-ref sanity filter.
    if pdf_places and "wikidata" not in skip:
        from tools.geocoding import wikidata_place_search, _distance_m

        # Re-collect existing center names so we don't double-add what
        # Zoomstack already found
        existing_names_lower = {c[0].split(":", 1)[-1].lower() for c in centers}

        wd_added = wd_skipped = 0
        for place in pdf_places[:5]:
            if not isinstance(place, str) or len(place.strip()) < 3:
                continue
            lower = place.lower().strip()
            if lower in ("none", "null", "n/a", "unknown", "various"):
                continue
            # Skip if Zoomstack already found this place
            if lower in existing_names_lower:
                continue
            # Use updated parent (gpkg hits may have refined it)
            p_lat = p_lon = None
            if all_parsed_refs:
                p_lat = sum(p[0] for p in all_parsed_refs) / len(all_parsed_refs)
                p_lon = sum(p[1] for p in all_parsed_refs) / len(all_parsed_refs)
            elif centers:
                lats = sorted(c[1] for c in centers)
                lons = sorted(c[2] for c in centers)
                p_lat = lats[len(lats) // 2]
                p_lon = lons[len(lons) // 2]
            try:
                hits = wikidata_place_search(
                    place, parent_lat=p_lat, parent_lon=p_lon,
                    max_parent_distance_km=30.0, limit=1)
            except Exception:
                hits = []
            if not hits:
                continue
            h = hits[0]
            # Grid-ref sanity (same threshold as Zoomstack)
            if all_parsed_refs:
                rsl = sum(p[0] for p in all_parsed_refs) / len(all_parsed_refs)
                rsn = sum(p[1] for p in all_parsed_refs) / len(all_parsed_refs)
                d_km = _distance_m(h["lat"], h["lon"], rsl, rsn) / 1000.0
                if d_km > 10.0:
                    print(f"    wikidata: REJECT {place!r} → {h['name']!r} "
                          f"({d_km:.1f}km from grid-ref centroid)")
                    wd_skipped += 1
                    continue
            gc_tuple = (f"wikidata:{h['name']}", h["lat"], h["lon"], 500)
            if gc_tuple not in centers:
                centers.append(gc_tuple)
                if gc_tuple not in state.centers:
                    state.centers.append(gc_tuple)
                state.centers_tried.append({
                    "source": "wikidata", "name": gc_tuple[0],
                    "lat": h["lat"], "lon": h["lon"], "was_picked": False,
                })
                wd_added += 1
                print(f"    wikidata: added {place!r} → {h['name']!r} "
                      f"({h['qid']}, {h['lat']:.5f}, {h['lon']:.5f})")
        if wd_added or wd_skipped:
            print(f"    wikidata summary: {wd_added} added, {wd_skipped} rejected")

    # E: Nominatim structured — two paths:
    #   (1) site_address starting with a house number -> street + city lookup.
    #   (2) each road_name + inferred city context -> street-level hit.
    # Both respect the grid-ref sanity filter.
    from tools.geocoding import nominatim_structured, _distance_m
    import re as _re

    # City context priority: reader's district_name, then the first Zoomstack
    # hit of type Town/City/Village/Hamlet, then the trailing comma-token of
    # site_address (skipping postcodes, countries, and county suffixes).
    city_ctx = ""
    if pdf_district:
        city_ctx = pdf_district.split(",")[0].strip()
    if not city_ctx:
        _settlement_types = ("(Town)", "(City)", "(Village)", "(Hamlet)",
                             "(Suburb)", "(Suburban Area)")
        for c in centers:
            cname = c[0]
            if any(t in cname for t in _settlement_types):
                # Format is "gpkg:NAME(TYPE)"
                name_part = cname.split(":", 1)[-1]
                bare = name_part.rsplit("(", 1)[0].strip()
                if bare:
                    city_ctx = bare
                    break
    if not city_ctx:
        _site_addr_for_city = (state.pdf_info or {}).get("site_address") or ""
        if _site_addr_for_city:
            # Countries, administrative counties, and metropolitan counties
            # that Nominatim does not resolve as city= parameter.
            _BAD_CITY = {
                "uk", "u.k.", "united kingdom", "england", "scotland", "wales",
                "northern ireland", "n ireland",
                "merseyside", "tyne and wear", "west midlands",
                "greater london", "greater manchester", "west yorkshire",
                "south yorkshire", "north yorkshire", "east sussex",
                "west sussex", "norfolk", "suffolk", "cambridgeshire",
                "hertfordshire", "staffordshire", "leicestershire",
                "warwickshire", "buckinghamshire", "bedfordshire", "kent",
                "essex", "surrey", "hampshire",
            }
            _PC_RE = _re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?\s*\d?[A-Z]{0,2}$")
            parts = [p.strip() for p in _site_addr_for_city.split(",")
                     if p.strip()]
            for p in reversed(parts[1:]):  # skip the street (first) token
                low = p.lower()
                if low in _BAD_CITY or _PC_RE.match(p) or low.endswith("shire"):
                    continue
                city_ctx = p
                break

    def _nominatim_sanity_ok(lat, lon):
        if not all_parsed_refs:
            return True
        rsl = sum(p[0] for p in all_parsed_refs) / len(all_parsed_refs)
        rsn = sum(p[1] for p in all_parsed_refs) / len(all_parsed_refs)
        return _distance_m(lat, lon, rsl, rsn) <= 10_000

    # Path 1: house-numbered site_address
    pdf_site_addr = (state.pdf_info or {}).get("site_address") or ""
    if pdf_site_addr and _re.match(r"^\d+\s+[A-Za-z]", pdf_site_addr.strip()) \
            and "nominatim:addr" not in skip:
        parts = [p.strip() for p in pdf_site_addr.split(",") if p.strip()]
        if parts:
            street = parts[0]
            city = city_ctx or (parts[1] if len(parts) > 1 else "")
            try:
                res = nominatim_structured(street=street, city=city, country="UK")
            except Exception:
                res = None
            if res and _nominatim_sanity_ok(res["lat"], res["lon"]):
                # addr: prefix is rank 0 in the specificity ranker.
                nm_tuple = (f"nominatim:addr:{street}", res["lat"], res["lon"], 500)
                if nm_tuple not in centers:
                    centers.append(nm_tuple)
                    if nm_tuple not in state.centers:
                        state.centers.append(nm_tuple)
                    state.centers_tried.append({
                        "source": "nominatim:addr", "name": nm_tuple[0],
                        "lat": res["lat"], "lon": res["lon"], "was_picked": False,
                    })
                    print(f"    nominatim(addr): added {street!r} → "
                          f"({res['lat']:.5f}, {res['lon']:.5f})")

    # Path 2: road_name + city context. Nominatim is picky about which admin
    # level matches city= — e.g. "Pipers Lane" + city="Heswall" fails, but
    # "Pipers Lane" + city="Wirral" resolves. Try multiple city candidates
    # until one returns a valid result.
    pdf_roads = (state.pdf_info or {}).get("road_names") or []
    if pdf_roads and "nominatim:road" not in skip:
        city_candidates = []
        def _add_city(c):
            if c and len(c) >= 3 and c not in city_candidates:
                city_candidates.append(c)
        _add_city(city_ctx)
        for pn in pdf_places[:4]:
            if isinstance(pn, str):
                _add_city(pn.strip())
        _site_all_tokens = [p.strip() for p in pdf_site_addr.split(",")
                            if p.strip()] if pdf_site_addr else []
        for t in reversed(_site_all_tokens[1:]):
            _add_city(t)

        nm_added = 0
        for road in pdf_roads[:4]:
            if not isinstance(road, str) or len(road.strip()) < 3:
                continue
            res = None
            city_used = None
            for cand in city_candidates[:4]:  # try up to 4 city variants
                try:
                    r = nominatim_structured(street=road.strip(), city=cand,
                                             country="UK")
                except Exception:
                    r = None
                if r and _nominatim_sanity_ok(r["lat"], r["lon"]):
                    res = r
                    city_used = cand
                    break
            if not res:
                continue
            nm_tuple = (f"nominatim:{road}", res["lat"], res["lon"], 500)
            if nm_tuple not in centers:
                centers.append(nm_tuple)
                if nm_tuple not in state.centers:
                    state.centers.append(nm_tuple)
                state.centers_tried.append({
                    "source": "nominatim:road", "name": nm_tuple[0],
                    "lat": res["lat"], "lon": res["lon"], "was_picked": False,
                })
                nm_added += 1
                print(f"    nominatim(roads): {road!r} → city={city_used!r}")
                if nm_added >= 3:  # plenty of street-level centers, stop
                    break
        if nm_added:
            print(f"    nominatim(roads) total: {nm_added} added")

    if not centers:
        raise ModelRetry(
            "No geocoded centers available. Call geocode first to get "
            "coordinates, then pass them as extra_centers."
        )

    result = sliding_window_position(
        matcher=state.minima_matcher,
        map_img=state.map_img,
        sam3_mask=state.current_mask,
        centers=centers,
        scale_ratio=state.scale_ratio,
        dpi=state.dpi,
        road_names=road_names or [],
        tile_fetcher=None,
        grayscale=use_grayscale,
    )
    mi = result.get("match_info", {})
    n_inliers = mi.get("n_inliers", 0)
    tile_source = "grayscale" if use_grayscale else "modern"

    state.current_result = result

    # Mark which center won (by matching its encoded name in match_info.center)
    winning_name = (mi.get("center") or "")
    winning_source = None
    if winning_name:
        for c in state.centers_tried:
            if c["name"] == winning_name or c["name"].split(":", 1)[-1] == winning_name:
                c["was_picked"] = True
                winning_source = c["source"]
                break
        if winning_source is None:
            # Fallback: match on (lat, lon) rounded to 5dp — labels can diverge
            wlat, wlon = mi.get("center_latlon") or (None, None)
            if wlat is not None:
                for c in state.centers_tried:
                    if abs(c["lat"] - wlat) < 1e-5 and abs(c["lon"] - wlon) < 1e-5:
                        c["was_picked"] = True
                        winning_source = c["source"]
                        break

    centers_summary = {
        "n_centers": len(state.centers_tried),
        "sources": sorted({c["source"] for c in state.centers_tried}),
        "winning_source": winning_source,
        "skipped": sorted(skip),
    }

    response = {
        "success": True,
        "n_inliers": n_inliers,
        "score": round(mi.get("score", 0), 1),
        "aspect": round(mi.get("aspect", 0), 3),
        "center_latlon": mi.get("center_latlon"),
        "zoom": mi.get("zoom"),
        "has_geojson": result.get("geojson") is not None,
        "tile_source": tile_source,
        "centers_summary": centers_summary,
    }

    if n_inliers < 50:
        hints = []
        if not use_grayscale:
            hints.append("try use_grayscale=true if the map is B&W or sepia-tinted")
        hints.append("find NEW geocoding clues you haven't tried yet (postcodes, grid refs, road names)")

        if state.position_calls >= 2:
            # On the last allowed attempt, tell the agent to just accept and move on
            response["WARNING"] = (
                f"POSITIONING POOR — only {n_inliers} inliers after {state.position_calls} attempts. "
                "Accept this result and proceed to extract_boundary and project_boundary. "
                "Do NOT call position_boundary again."
            )
        else:
            response["WARNING"] = (
                f"POSITIONING LIKELY FAILED — only {n_inliers} inliers (need >= 50). "
                "Do NOT just retry with the same centers — you must geocode NEW locations "
                "first (different postcodes, road names, grid refs you haven't tried). "
                "Retrying with the same inputs will give the same result. "
                "Suggestions: " + "; ".join(hints) + "."
            )
    elif n_inliers >= 100:
        response["status"] = f"excellent positioning ({tile_source} tiles)"
    else:
        response["status"] = f"decent positioning, probably correct ({tile_source} tiles)"

    return response


# ── Tool 4: extract_boundary ──────────────────────────────────────────────

@_agent.tool
def extract_boundary(
    ctx: RunContext[AgentState],
    mode: str,
    query: str = "planning boundary",
    select_indices: Optional[List[int]] = None,
) -> Union[dict, ToolReturn]:
    """Extract the planning boundary from the map image using SAM3 segmentation.

    SAM3 is a vision model that finds regions in the image matching a text query.
    It highlights the planning boundary (colored outline, shading, or tickmarks).

    Two modes:
      - "instance" (RECOMMENDED): Returns up to 5 candidate masks with individual
        visualizations. Inspect them, then call again with select_indices to combine
        your chosen ones. This gives you control over which regions to include.
      - "semantic": Returns one best mask for the query. Simpler but less control.

    Query tips:
      - Default: "planning boundary"
      - If boundary is red: "red planning boundary" or "land edged red"
      - If boundary is blue: "blue planning boundary"
      - If boundary is pink/shaded: "pink shaded area"
      - If there are tick marks: "red tick marks along buildings"

    After extracting a boundary, call project_boundary to convert it to GeoJSON.

    Args:
        mode: "instance" (recommended, multiple candidates) or "semantic" (single mask)
        query: Text prompt describing what to find (default: "planning boundary")
        select_indices: For instance mode only. 0-based indices of candidates to
            combine into one mask. Call without this first to see candidates.

    Returns:
        For mode="semantic":
          {"success": true, "mode": "semantic", "query": str, "mask_area_pct": float}
          The mask is applied automatically. Call project_boundary to get the GeoJSON.

        For mode="instance" without select_indices:
          Images of each candidate overlaid on the map (shown to you), plus:
          {"success": true, "n_candidates": int, "candidates": [{"index": int, "area_pct": float}, ...]}
          Inspect the candidate images, then call again with select_indices.

        For mode="instance" with select_indices:
          {"success": true, "mode": "instance_combine", "combined_indices": [int, ...], "mask_area_pct": float}
          The combined mask is applied. Call project_boundary to get the GeoJSON.
    """
    state = ctx.deps
    _dedup_check(state, "extract_boundary", {
        "mode": mode, "query": query, "select_indices": select_indices,
    })

    if state.map_img is None or state.map_crop_path is None:
        raise ModelRetry("No map image available. Call render_page first.")

    if mode == "semantic":
        from tools.sam3_boundary import extract_boundary_sam3_semantic
        mask = extract_boundary_sam3_semantic(
            state.map_crop_path, state.sam3_processor,
            state.sam3_model, state.device, query=query,
        )
        if mask is not None:
            area_pct = np.sum(mask > 0) / mask.size * 100
            # Auto-fallback: if semantic selected >60% of the image, it probably
            # grabbed the whole map instead of just the boundary. Switch to instance.
            if area_pct > 60:
                print(f"  Semantic mask covers {area_pct:.0f}% — too large, "
                      f"auto-switching to instance mode")
                raise ModelRetry(
                    f"Semantic segmentation selected {area_pct:.0f}% of the image, "
                    f"which is likely the entire map rather than just the boundary. "
                    f"Use mode='instance' instead to see individual candidates and "
                    f"select the correct ones."
                )
            state.current_mask = mask
            # Save selected overlay for caching
            if state.map_img is not None:
                sel_overlay = state.map_img.copy()
                sel_overlay[mask > 0] = [0, 255, 0]
                state.selected_overlay = cv2.addWeighted(
                    state.map_img, 0.5, sel_overlay, 0.5, 0)
            return {"success": True, "mode": "semantic", "query": query,
                    "mask_area_pct": round(area_pct, 2)}
        return {"success": False, "error": "SAM3 semantic returned no mask"}

    elif mode == "instance":
        if select_indices is not None:
            if not state.instance_masks:
                raise ModelRetry(
                    "No instance masks available. Call extract_boundary with "
                    "mode='instance' without select_indices first."
                )
            valid = [i for i in select_indices
                     if 0 <= i < len(state.instance_masks)]
            if not valid:
                raise ModelRetry(
                    f"Invalid indices. Available: 0-{len(state.instance_masks) - 1}"
                )
            combined = np.zeros_like(state.instance_masks[0])
            for i in valid:
                combined = np.maximum(combined, state.instance_masks[i])
            state.current_mask = combined
            state.selected_indices = valid
            # Save selected overlay for caching
            if state.map_img is not None:
                sel_overlay = state.map_img.copy()
                sel_overlay[combined > 0] = [0, 255, 0]
                state.selected_overlay = cv2.addWeighted(
                    state.map_img, 0.5, sel_overlay, 0.5, 0)
            area_pct = np.sum(combined > 0) / combined.size * 100
            return {"success": True, "mode": "instance_combine",
                    "combined_indices": valid,
                    "mask_area_pct": round(area_pct, 2)}
        else:
            instances = _get_instance_masks(
                state.map_crop_path, state.sam3_processor,
                state.sam3_model, state.device, query=query, top_k=5,
            )
            state.instance_masks = instances
            if instances:
                state.current_mask = instances[0]

            content_parts = []
            summaries = []
            state.candidate_overlays = []  # reset for this extraction
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
                      (255, 255, 0), (0, 255, 255)]
            for i, inst in enumerate(instances[:5]):
                area_pct = np.sum(inst > 0) / inst.size * 100
                summaries.append({"index": i, "area_pct": round(area_pct, 2)})
                overlay = state.map_img.copy()
                overlay[inst > 0] = colors[i % len(colors)]
                blended = cv2.addWeighted(state.map_img, 0.5, overlay, 0.5, 0)
                cv2.putText(blended, f"Candidate {i}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                state.candidate_overlays.append(blended.copy())
                content_parts.append(f"Candidate {i} (area={area_pct:.1f}%):")
                content_parts.append(_img_to_binary(blended))

            return ToolReturn(
                return_value={
                    "success": True, "mode": "instance",
                    "n_candidates": len(instances),
                    "candidates": summaries,
                    "instruction": "Call extract_boundary again with "
                                   "mode='instance' and select_indices=[...] "
                                   "to combine your chosen candidates.",
                },
                content=content_parts if content_parts else None,
            )

    raise ModelRetry(f"Invalid mode '{mode}'. Use 'semantic' or 'instance'.")


# ── Tool 5: project_boundary ──────────────────────────────────────────────

@_agent.tool
def project_boundary(ctx: RunContext[AgentState]) -> dict:
    """Project the current boundary mask to real-world coordinates (GeoJSON).

    Uses the affine transform from position_boundary to convert the pixel mask
    into a GeoJSON polygon with lat/lon coordinates. You must have called both
    position_boundary (to get the affine) and extract_boundary (to get the mask)
    before calling this.

    Call this whenever you want to produce or update the GeoJSON — after
    extract_boundary, or after re-running position_boundary or extract_boundary
    during refinement.

    Returns:
        {"success": true, "n_polygons": int, "total_area_m2": float}
        The GeoJSON is stored internally and used by verify_position and visualize.
    """
    state = ctx.deps

    if state.current_mask is None:
        raise ModelRetry("No boundary mask available. Call extract_boundary first.")

    affine_H = state.current_result.get("affine_H")
    tile_info = state.current_result.get("tile_info")
    if affine_H is None or tile_info is None:
        raise ModelRetry("No positioning result available. Call position_boundary first.")

    from tools.positioning import mask_to_geojson_affine
    geojson = mask_to_geojson_affine(state.current_mask, affine_H, tile_info)

    if geojson is None:
        return {"success": False, "error": "Mask projection produced no polygons"}

    state.current_result["geojson"] = geojson

    # Count polygons and estimate area
    geom = geojson.get("geometry", {})
    if geom.get("type") == "MultiPolygon":
        n_polys = len(geom.get("coordinates", []))
    elif geom.get("type") == "Polygon":
        n_polys = 1
    else:
        n_polys = 0

    return {"success": True, "n_polygons": n_polys}


# ── Tool 6: accumulate_boundary ─────────────────────────────────────────────

@_agent.tool
def accumulate_boundary(ctx: RunContext[AgentState]) -> dict:
    """Save the current page's boundary and reset state for the next map page.

    Use this when the planning document has MULTIPLE map pages covering different
    parts of the planning area. After completing the full pipeline for one page
    (render → geocode → position → extract → project), call this to save the
    result, then repeat for the next map page.

    The final GeoJSON (returned when the agent finishes) will automatically merge
    all accumulated boundaries into a single MultiPolygon.

    You do NOT need to call this for single-page maps — just call project_boundary
    and finish normally.

    Returns:
        {"success": true, "pages_accumulated": int, "n_polygons_this_page": int}
    """
    state = ctx.deps

    geojson = state.current_result.get("geojson")
    if geojson is None:
        raise ModelRetry(
            "No projected boundary to accumulate. Call project_boundary first."
        )

    state.accumulated_geojson.append(geojson)
    state.pages_accumulated = len(state.accumulated_geojson)

    # Count polygons in this page's result
    geom = geojson.get("geometry", {})
    if geom.get("type") == "MultiPolygon":
        n_polys = len(geom.get("coordinates", []))
    elif geom.get("type") == "Polygon":
        n_polys = 1
    else:
        n_polys = 0

    # Reset per-page state for the next map page (but KEEP validator flags since
    # they describe the whole document, not just the current page).
    state.map_img = None
    state.map_crop_path = None
    state.current_mask = None
    state.instance_masks = []
    state.current_result = {}
    state.centers = []
    state.scale_ratio = None
    state.position_calls = 0
    state.verify_position_called = False  # next page must re-verify if borderline
    # Keep recent_calls to avoid re-geocoding the same locations
    # Keep pages_accumulated, pdf_info, rotation_checked across pages

    return {
        "success": True,
        "pages_accumulated": state.pages_accumulated,
        "n_polygons_this_page": n_polys,
    }


def _fallback_geojson_at_anchor(mask, centers, scale_ratio, dpi) -> Optional[dict]:
    """Project SAM mask at the best geocode anchor without running MINIMA.

    Used when MINIMA failed entirely (0 inliers, or no matches kept after
    filter). Produces a low-accuracy GeoJSON that will at least have partial
    overlap with the true boundary for cases in the right city. Better than
    returning None.

    Picks the highest-specificity center and computes an affine from (i) the
    given scale_ratio and map DPI and (ii) the assumption rotation=0,
    mask-centroid lands at the anchor. Worst case IoU ≈ 0 if anchor is
    wildly wrong, best case ≈ 0.1-0.5 for in-neighborhood anchors.
    """
    if mask is None or not centers:
        return None
    try:
        import numpy as _np
        from tools.positioning import (
            _center_specificity, compute_map_mpp, best_zoom_for_scale,
            mask_to_geojson_affine,
        )
        from tools.os_opendata_tiles import fetch_os_opendata_grid

        # Pick the most specific center
        ranked = sorted(centers, key=lambda c: _center_specificity(c[0]))
        name, lat, lon, _ = ranked[0]

        # Build a rough affine: mask pixel (px, py) → tile canvas pixel.
        # scale_ratio → map_mpp (meters per map pixel)
        # zoom → tile_mpp (meters per tile pixel)
        # scale_factor = map_mpp / tile_mpp  (how much to resize map into canvas)
        # translate so mask centroid lands at anchor (canvas center)
        map_mpp = compute_map_mpp(scale_ratio, dpi)
        if map_mpp is None:
            return None
        zoom = best_zoom_for_scale(map_mpp, lat)
        import math as _math
        tile_mpp = 156543.03 * _math.cos(_math.radians(lat)) / (2 ** zoom)
        sf = map_mpp / tile_mpp

        # Mask centroid in mask-pixel coords
        ys, xs = _np.where(mask > 0)
        if len(xs) == 0:
            return None
        mcx, mcy = float(xs.mean()), float(ys.mean())

        # Fetch a tile grid around the anchor, sized to cover the
        # mask bbox (after scale) + small margin.
        h, w = mask.shape[:2]
        canvas_w = int(w * sf)
        canvas_h = int(h * sf)
        nx = max(5, (canvas_w // 256) + 4)
        ny = max(5, (canvas_h // 256) + 4)
        if nx % 2 == 0:
            nx += 1
        if ny % 2 == 0:
            ny += 1
        nx, ny = min(35, nx), min(35, ny)
        try:
            tile_info = fetch_os_opendata_grid(lat, lon, zoom, nx, ny)
        except Exception:
            return None

        # Affine: map-pixel (px, py) → canvas pixel
        # scaled_px = px * sf, then translate so mask centroid lands at
        # canvas center.
        canvas_cx = (nx * 256) / 2
        canvas_cy = (ny * 256) / 2
        tx = canvas_cx - mcx * sf
        ty = canvas_cy - mcy * sf
        affine_H = _np.array([[sf, 0, tx], [0, sf, ty]], dtype=_np.float64)

        return mask_to_geojson_affine(mask, affine_H, tile_info)
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def _merge_geojson(features: List[dict]) -> Optional[dict]:
    """Merge multiple GeoJSON Features into a single MultiPolygon Feature."""
    all_polys = []
    for feat in features:
        geom = feat.get("geometry", {})
        if geom.get("type") == "Polygon":
            all_polys.append(geom["coordinates"])
        elif geom.get("type") == "MultiPolygon":
            all_polys.extend(geom["coordinates"])

    if not all_polys:
        return None

    return {
        "type": "Feature",
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": all_polys,
        },
        "properties": {},
    }


# ── Tool 7: verify_position ─────────────────────────────────────────────────

@_agent.tool
def verify_position(
    ctx: RunContext[AgentState],
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> ToolReturn:
    """Visually inspect the positioned result on Ordnance Survey tiles.

    Renders the OS OpenData map at the current (or given) location with the
    predicted boundary drawn on it in red. Use this to visually confirm the
    positioning looks correct — check that roads, buildings, and features on
    the OS tiles match what you see on the planning map.

    Note: road name verification is already done automatically inside
    position_boundary (see its road_matches in the return value). This tool
    is for visual inspection.

    If no lat/lon provided, uses the current positioned center.

    Args:
        lat: Latitude to inspect (default: current positioned center)
        lon: Longitude to inspect (default: current positioned center)

    Returns:
        Image of OS tiles with boundary overlay (shown to you), plus:
        {"success": true, "lat": float, "lon": float}
    """
    state = ctx.deps
    from tools.os_opendata_tiles import fetch_os_opendata_grid

    if lat is None or lon is None:
        center_ll = state.current_result.get("match_info", {}).get("center_latlon")
        if center_ll:
            lat, lon = center_ll
        else:
            raise ModelRetry("No position available. Run position_boundary first "
                             "or provide lat/lon.")

    tile_info = fetch_os_opendata_grid(lat, lon, 17, 5, 5)
    tile_bgr = cv2.cvtColor(tile_info["image"], cv2.COLOR_RGB2BGR)

    geojson = state.current_result.get("geojson")
    if geojson:
        tile_bgr = _draw_geojson_on_tiles(tile_bgr, geojson, tile_info)

    # Mark that verify_position ran — the output_validator checks this flag.
    state.verify_position_called = True

    return ToolReturn(
        return_value={"success": True, "lat": lat, "lon": lon},
        content=[
            f"OS tiles at ({lat:.4f}, {lon:.4f}):",
            _img_to_binary(tile_bgr),
            "Visual verification complete. Compare road patterns, settlement "
            "shape, and named roads against the planning map. If they MATCH, "
            "proceed to submit with status='accepted' and fill visual_check_notes. "
            "If they DO NOT MATCH, submit with status='rejected_visual_mismatch'.",
        ],
    )


# ── Tool 8: lookup_district ──────────────────────────────────────────────

@_agent.tool
def lookup_district(
    ctx: RunContext[AgentState],
    district_name: str,
) -> dict:
    """Look up the full boundary of an administrative district from OpenStreetMap.

    Use this when the planning document covers an ENTIRE district, borough, ward,
    or parish — not a specific site within one.

    This returns the official boundary polygon directly from OSM, no positioning
    or SAM3 extraction needed. If this succeeds, you're done — respond with DONE.

    Naming conventions (be specific to avoid ambiguous matches):
      - "London Borough of Barnet, London, UK"
      - "Royal Borough of Kensington and Chelsea, London, UK"
      - "City of Westminster, London, UK"
      - "Rowley Green, London Borough of Barnet, London, UK" (for wards)

    Args:
        district_name: Full name of the district/borough/ward, including parent
            areas for disambiguation and "UK" suffix.

    Returns:
        {"success": true, "geojson": <GeoJSON Feature>} — the complete boundary
        {"success": false, "error": str} — if the district wasn't found in OSM
    """
    state = ctx.deps
    _dedup_check(state, "lookup_district", {"district_name": district_name})

    from tools.geo_tools import lookup_district_boundary

    # Support '|' alternates: try each variant in order until one works.
    variants = [v.strip() for v in district_name.split("|") if v.strip()]
    for variant in variants:
        result = lookup_district_boundary(variant)
        if result.get("success"):
            geojson = result["geojson"]
            # Normalize to MultiPolygon
            geom = geojson.get("geometry", {})
            if geom.get("type") == "Polygon":
                geojson["geometry"] = {
                    "type": "MultiPolygon",
                    "coordinates": [geom["coordinates"]],
                }
            geojson["properties"]["source"] = "osm_district_lookup"
            state.current_result = {"geojson": geojson, "match_info": {}}
            return {
                "success": True,
                "matched_variant": variant,
                "instruction": "District lookup succeeded. Submit your final "
                               "result with status='district_lookup' and a brief "
                               "reasoning. No positioning or verify_position needed.",
            }
    return {"success": False,
            "error": f"None of the variants {variants} matched in OSM"}


# ── Tool 9: visualize ────────────────────────────────────────────────────

@_agent.tool
def visualize(ctx: RunContext[AgentState]) -> ToolReturn:
    """Show the current state: boundary mask overlay on the map image, instance mask
    candidates if available, and the positioned boundary on OS OpenData tiles.

    Call this to inspect your work before finishing. Check that:
      - The boundary mask correctly outlines the planning site (not too much, not too little)
      - The positioned boundary on OS tiles is in the right real-world location
      - Road names on the OS tiles match roads visible on the planning map

    Returns:
        Images of current state (shown to you), plus:
        {"success": true, "images_available": ["boundary_overlay", "instance_overlay", "positioned_on_os_tiles"]}
    """
    state = ctx.deps
    content_parts: list = []
    images_available = []

    # 1. Boundary overlay
    if state.current_mask is not None and state.map_img is not None:
        overlay = _create_boundary_overlay(state.map_img, state.current_mask)
        content_parts.append("Boundary mask overlay (red):")
        content_parts.append(_img_to_binary(overlay))
        images_available.append("boundary_overlay")

    # 2. Instance masks
    if state.instance_masks and state.map_img is not None:
        inst_viz = state.map_img.copy()
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
                  (255, 255, 0), (0, 255, 255)]
        for i, inst in enumerate(state.instance_masks[:5]):
            color = colors[i % len(colors)]
            inst_viz[inst > 0] = color
        inst_overlay = cv2.addWeighted(state.map_img, 0.5, inst_viz, 0.5, 0)
        content_parts.append("Instance masks (red=0, green=1, blue=2, yellow=3, cyan=4):")
        content_parts.append(_img_to_binary(inst_overlay))
        images_available.append("instance_overlay")

    # 3. Positioned GeoJSON on OS tiles
    geojson = state.current_result.get("geojson")
    tile_info = state.current_result.get("tile_info")
    if geojson and tile_info and tile_info.get("image") is not None:
        tile_bgr = cv2.cvtColor(tile_info["image"], cv2.COLOR_RGB2BGR)
        tile_bgr = _draw_geojson_on_tiles(tile_bgr, geojson, tile_info)
        content_parts.append("Positioned boundary on OS OpenData tiles:")
        content_parts.append(_img_to_binary(tile_bgr))
        images_available.append("positioned_on_os_tiles")

    return ToolReturn(
        return_value={"success": True, "images_available": images_available},
        content=content_parts if content_parts else None,
    )


# ── Main Entry Point ────────────────────────────────────────────────────────

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


def _read_pdf_phase(pdf_path: str, model_name: str, verbose: bool = True) -> dict:
    """Phase 1: Send the PDF to the reader agent, get structured extraction.

    The reader agent's output_type=PDFInfo, so pydantic-ai enforces the schema.
    No JSON parsing or markdown fence stripping needed.

    Returns:
        Dict of extracted info (from PDFInfo.model_dump()), plus "_reader_tokens".
        On failure: {"error": ..., defaulted fields...}.
    """
    pdf_bytes = Path(pdf_path).read_bytes()

    if verbose:
        print(f"  Phase 1: reading PDF ({len(pdf_bytes) // 1024} KB)...")

    model = OpenRouterModel(model_name)

    from pydantic_ai.exceptions import UnexpectedModelBehavior

    try:
        result = _reader_agent.run_sync(
            [
                BinaryContent(data=pdf_bytes, media_type='application/pdf'),
                "Read this UK planning PDF and populate the PDFInfo schema with "
                "all geographic information you can find.",
            ],
            model=model,
            usage_limits=UsageLimits(request_limit=5),  # allow a validator retry
        )
        info_model: PDFInfo = result.output
        info = info_model.model_dump()

        if verbose:
            print(f"  Phase 1: map_pages={info['map_pages']}, "
                  f"postcodes={info['postcodes']}, "
                  f"roads={len(info['road_names'])}, "
                  f"scale={info['scale']}, "
                  f"district={info['is_district_wide']}")

        # Extract usage from reader phase
        usage = result.usage()
        info["_reader_tokens"] = {
            "request": usage.request_tokens,
            "response": usage.response_tokens,
        }

        return info

    except UnexpectedModelBehavior as e:
        if verbose:
            print(f"  Phase 1 failed: {e}")
        # Fallback: return an empty-ish PDFInfo so downstream doesn't crash.
        empty = PDFInfo().model_dump()
        empty["error"] = str(e)
        return empty


def run_agent(
    pdf_path: str,
    models_state: dict,
    model_name: str = "google/gemini-3.1-pro-preview",
    max_iterations: int = 6,
    dpi: int = 200,
    verbose: bool = True,
    enable_critic: bool = True,
) -> Dict[str, Any]:
    """Run the two-phase agent on a single planning document.

    Phase 1: Reader agent reads the full PDF once, extracts structured info (JSON).
    Phase 2: Worker agent receives only the JSON summary + rendered map image.
             The full PDF is never in the worker's context, saving ~90% of tokens
             on multi-turn conversations.
    Phase 3 (if enable_critic=True): Commenter VLM critic reviews the worker's
             output, auto-fixes simple issues in code, or re-enters the worker
             with feedback. Never nullifies the GeoJSON — partial IoU > 0.
             Skipped for multi-page and district_lookup cases.

    Args:
        pdf_path: Path to the planning PDF.
        models_state: Dict with sam3_ft/sam3_base and minima models.
        model_name: OpenRouter model identifier.
        max_iterations: Maximum number of agent turns.
        dpi: PDF rendering DPI.
        verbose: Print progress.
        enable_critic: Run Phase 3 critic loop after worker finishes.

    Returns:
        Dict with: geojson, match_info, mask, agent_accepted, agent_reason, etc.
    """
    # Resolve model alias
    model_name = MODEL_ALIASES.get(model_name, model_name)

    # Get SAM3 model (prefer fine-tuned)
    if "sam3_ft" in models_state:
        sam3 = models_state["sam3_ft"]
    elif "sam3_base" in models_state:
        sam3 = models_state["sam3_base"]
    else:
        return {"success": False, "error": "No SAM3 model loaded"}

    # ── Phase 1: Read the PDF ──────────────────────────────────────────────
    pdf_info = _read_pdf_phase(pdf_path, model_name, verbose=verbose)

    # ── Phase 2: Work from the summary ─────────────────────────────────────
    state = AgentState(
        pdf_path=str(pdf_path),
        sam3_processor=sam3["processor"],
        sam3_model=sam3["model"],
        device=sam3["device"],
        minima_matcher=models_state["minima"],
        dpi=dpi,
    )
    # Give the output_validator access to the reader's extraction (for
    # multi-page counting and district-wide checks).
    state.pdf_info = {k: v for k, v in pdf_info.items() if not k.startswith("_")}

    # Render the first map page so the worker agent has an image immediately
    map_pages = pdf_info.get("map_pages", [])
    map_page_imgs = []
    if map_pages:
        first_map_page = map_pages[0]
        doc = fitz.open(str(pdf_path))
        page_idx = max(0, first_map_page - 1)
        if page_idx < len(doc):
            pix = doc[page_idx].get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n)
            if img.shape[2] == 4:
                map_img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                map_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Apply reader-detected rotation so the map is north-up before
            # SAM3, MINIMA, and verify_position all see it.
            map_rotation = int(pdf_info.get("map_rotation", 0) or 0)
            if map_rotation in (90, 180, 270):
                rot_code = {
                    90: cv2.ROTATE_90_CLOCKWISE,
                    180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                }[map_rotation]
                map_img = cv2.rotate(map_img, rot_code)
                state.rotation_checked = True
                if verbose:
                    print(f"  Pre-rotated map by {map_rotation}° (reader detected rotation)")

            state.map_img = map_img
            # Save to temp file for SAM3
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                state.map_crop_path = tmp.name
                cv2.imwrite(tmp.name, map_img)
            map_page_imgs.append((first_map_page, map_img))
        doc.close()

    # Fast path: district-wide boundary. Skips the worker agent entirely.
    if pdf_info.get("is_district_wide") and pdf_info.get("district_name"):
        district_name = pdf_info["district_name"]
        if verbose:
            print(f"  District-wide detected: {district_name}")
        from tools.geo_tools import lookup_district_boundary
        result = lookup_district_boundary(district_name)
        if result.get("success"):
            geojson = result["geojson"]
            # Normalize to MultiPolygon
            geom = geojson.get("geometry", {})
            if geom.get("type") == "Polygon":
                geojson["geometry"] = {
                    "type": "MultiPolygon",
                    "coordinates": [geom["coordinates"]],
                }
            geojson["properties"]["source"] = "osm_district_lookup"
            if verbose:
                print("  District lookup succeeded — skipping agent")
            return {
                "success": True,
                "geojson": geojson,
                "match_info": {},
                "mask": None,
                "agent_accepted": True,
                "agent_reason": f"District lookup: {district_name}",
                "agent_stats": {"pdf_info": pdf_info, "method": "district_lookup"},
            }
        elif verbose:
            print("  District lookup failed, falling through to agent")

    # Build the worker's user prompt: JSON summary + pre-rendered map image
    # (no PDF binary — that's the whole point)
    summary_text = json.dumps({k: v for k, v in pdf_info.items()
                                if not k.startswith("_")}, indent=2)
    user_parts: list = [
        f"PDF EXTRACTION SUMMARY:\n{summary_text}\n\n"
        f"Use this information to geolocate and extract the planning boundary. "
        f"The first map page (page {map_pages[0] if map_pages else '?'}) has been "
        f"pre-rendered as your working map."
    ]

    # Attach the pre-rendered map image
    if map_page_imgs:
        page_num, img = map_page_imgs[0]
        user_parts.append(f"Map page {page_num}:")
        user_parts.append(_img_to_binary(img))

    if verbose:
        print(f"  Running agent ({model_name}, max {max_iterations} turns)")

    model = OpenRouterModel(model_name)

    from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded

    result = None
    agent_rejected = False  # set True if the agent's DONE message starts with REJECTED
    try:
        if verbose:
            print("  Phase 2: sending summary + map image to worker agent...")
            print(f"  Agent tools: {[t.name for t in _agent._toolset.tools()]}"
                  if hasattr(_agent, '_toolset') else "")
        # Floor of 25 model requests covers healthy hard cases (15-22 requests
        # typically), above that caps bad runs.
        result = _agent.run_sync(
            user_parts,
            deps=state,
            model=model,
            usage_limits=UsageLimits(request_limit=max(max_iterations * 4, 25)),
        )
        if verbose:
            print("  Agent completed normally")
            # Debug: show message history
            for msg in result.all_messages():
                role = getattr(msg, 'kind', type(msg).__name__)
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        kind = getattr(part, 'kind', type(part).__name__)
                        if kind == 'tool-call':
                            print(f"    [{role}] tool-call: {part.tool_name}({str(part.args)[:80]})")
                        elif kind == 'tool-return':
                            print(f"    [{role}] tool-return: {str(part.content)[:80]}")
                        elif kind == 'text':
                            print(f"    [{role}] text: {str(part.content)[:100]}")
                        else:
                            print(f"    [{role}] {kind}")
                else:
                    print(f"    [{role}] {str(msg)[:100]}")
    except (UnexpectedModelBehavior, UsageLimitExceeded) as e:
        if verbose:
            print(f"  Agent loop ended: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
        if not state.accepted:
            state.accepted = True
            state.accept_reason = f"Loop ended: {type(e).__name__}"
    except Exception as e:
        if verbose:
            print(f"  Agent error: {e}")
            traceback.print_exc()
        # Even on error, return any accumulated multi-page results
        partial_geojson = state.current_result.get("geojson")
        if state.accumulated_geojson:
            all_features = list(state.accumulated_geojson)
            if partial_geojson:
                all_features.append(partial_geojson)
            partial_geojson = _merge_geojson(all_features)
        return {
            "success": False,
            "error": str(e),
            "geojson": partial_geojson,
            "match_info": state.current_result.get("match_info", {}),
            "mask": state.current_mask,
        }
    else:
        # Structured output: result.output is a BoundaryOutcome (validated).
        outcome: BoundaryOutcome = result.output
        state.last_output = outcome
        state.accepted = (outcome.status in ("accepted", "district_lookup"))
        state.accept_reason = (
            f"[{outcome.status}] {outcome.reasoning[:160]}"
            + (f" | reject: {outcome.reject_reason[:120]}"
               if outcome.reject_reason else "")
        )
        if verbose:
            print(f"  Worker outcome: status={outcome.status} "
                  f"inliers={outcome.final_n_inliers} "
                  f"verify={outcome.verify_position_called} "
                  f"rotation_checked={outcome.rotation_checked} "
                  f"pages_acc={outcome.pages_accumulated}")

    # ── Phase 3: Commenter VLM critic loop ─────────────────────────────────
    # Runs only when the worker produced an "accepted" single-page result with
    # a geojson + mask. Multi-page and district_lookup are skipped for MVP.
    critic_result = None
    if enable_critic and result is not None \
            and state.last_output is not None \
            and state.last_output.status == "accepted" \
            and state.current_result.get("geojson") is not None \
            and state.current_mask is not None \
            and len(pdf_info.get("map_pages") or []) <= 1:
        try:
            from tools.critic import run_critic_loop
            critic_result = run_critic_loop(
                state=state,
                worker_agent=_agent,
                worker_result=result,
                model=model,
                sam3=sam3,
                minima_matcher=models_state["minima"],
                verbose=verbose,
            )
            state.critic_iterations = critic_result["iterations"]
            state.critic_final_decision = critic_result["final_decision"]
            state.critic_changed_mask = critic_result["changed_mask"]
            state.critic_suspected_wrong_location = critic_result.get(
                "suspected_wrong_location", False)
            state.critic_worker_reentered = critic_result.get(
                "worker_reentered", False)
            if critic_result["final_decision"] == "flag_low_confidence":
                # Label only — do not null the geojson.
                state.accepted = False
                last_reason = (
                    critic_result["iterations"][-1]["reasoning"][:160]
                    if critic_result["iterations"] else ""
                )
                state.accept_reason = (
                    f"CRITIC_LOW_CONFIDENCE: {last_reason} | prior: "
                    f"{(state.accept_reason or '')[:100]}"
                )
            if verbose:
                print(f"  Critic final: {critic_result['final_decision']} "
                      f"(changed_mask={state.critic_changed_mask}, "
                      f"worker_reentered={state.critic_worker_reentered})")
        except Exception as _critic_err:
            if verbose:
                print(f"  Critic loop failed (continuing): {_critic_err}")
                traceback.print_exc()

    # Agent-initiated rejection: either the structured status says rejected_*,
    # or (legacy/exception path) the accept_reason starts with REJECTED.
    if state.last_output is not None:
        agent_rejected = state.last_output.status.startswith("rejected_")
    else:
        agent_rejected = (state.accept_reason or "").upper().lstrip().startswith("REJECTED")

    # Clean up temp files
    if state.map_crop_path:
        try:
            os.unlink(state.map_crop_path)
        except OSError:
            pass
        # Clean up rotated variants
        for rot_path in Path(state.map_crop_path).parent.glob(
            Path(state.map_crop_path).stem + "_rot*.png"
        ):
            try:
                rot_path.unlink()
            except OSError:
                pass

    if verbose:
        mi = state.current_result.get("match_info", {})
        print(f"  Agent done: accepted={state.accepted}, "
              f"inliers={mi.get('n_inliers', 0)}, "
              f"reason={state.accept_reason[:100]}")

    # Extract comprehensive stats from message history
    agent_stats = {
        "position_calls": state.position_calls,
        "pdf_info": {k: v for k, v in pdf_info.items() if not k.startswith("_")},
    }
    # Include reader phase token usage
    reader_tokens = pdf_info.get("_reader_tokens", {})
    if reader_tokens:
        agent_stats["reader_request_tokens"] = reader_tokens.get("request", 0)
        agent_stats["reader_response_tokens"] = reader_tokens.get("response", 0)

    # Surface BoundaryOutcome fields directly so we don't have to rummage
    # through message_log.json to see WHY the agent accepted/rejected.
    if state.last_output is not None:
        out = state.last_output
        agent_stats["outcome_status"] = out.status
        agent_stats["outcome_reasoning"] = out.reasoning
        agent_stats["visual_check_notes"] = out.visual_check_notes
        agent_stats["reject_reason"] = out.reject_reason
        agent_stats["verify_position_called"] = out.verify_position_called
        agent_stats["rotation_checked"] = out.rotation_checked
        agent_stats["pages_accumulated"] = out.pages_accumulated

    # Extract full message history for offline analysis.
    # pydantic-ai 1.81 part kinds: SystemPromptPart, UserPromptPart, ToolCallPart,
    # ToolReturnPart, RetryPromptPart, TextPart, ThinkingPart.
    message_log = []
    try:
        if result is not None:
            tool_calls = {}
            turn_idx = 0
            for msg in result.all_messages():
                role = getattr(msg, 'kind', type(msg).__name__)
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        kind = getattr(part, 'kind', type(part).__name__)
                        # Normalize to lowercase for easier matching
                        kind_lower = kind.lower()
                        entry = {"turn": turn_idx, "role": role, "kind": kind}

                        if 'toolcall' in kind_lower:
                            name = getattr(part, 'tool_name', '?')
                            tool_calls[name] = tool_calls.get(name, 0) + 1
                            entry["tool"] = name
                            try:
                                args = getattr(part, 'args', None)
                                if args is None:
                                    entry["args"] = {}
                                elif isinstance(args, dict):
                                    entry["args"] = {
                                        k: (v if not isinstance(v, (bytes, bytearray))
                                            else f"<bytes:{len(v)}>")
                                        for k, v in args.items()
                                    }
                                elif isinstance(args, str):
                                    # Args can be a JSON string
                                    try:
                                        import json as _j
                                        parsed = _j.loads(args)
                                        entry["args"] = parsed if isinstance(parsed, dict) else str(args)[:500]
                                    except Exception:
                                        entry["args"] = args[:500]
                                else:
                                    entry["args"] = str(args)[:500]
                            except Exception as _e:
                                entry["args"] = f"<err:{_e}>"

                        elif 'toolreturn' in kind_lower:
                            entry["tool"] = getattr(part, 'tool_name', '?')
                            content = getattr(part, 'content', None)
                            if isinstance(content, dict):
                                entry["return"] = {
                                    k: (v if not isinstance(v, (bytes, bytearray))
                                        else f"<bytes:{len(v)}>")
                                    for k, v in content.items()
                                }
                            elif isinstance(content, str):
                                entry["return"] = content[:1000]
                            else:
                                entry["return"] = str(content)[:1000]

                        elif 'retry' in kind_lower:
                            # RetryPromptPart — the validator fired
                            rc = getattr(part, 'content', None)
                            entry["retry_content"] = str(rc)[:1000] if rc else ""

                        elif 'userprompt' in kind_lower:
                            # User prompts sometimes contain images — note their presence
                            c = getattr(part, 'content', None)
                            if isinstance(c, list):
                                n_images = sum(1 for x in c
                                               if hasattr(x, 'media_type'))
                                n_text = sum(1 for x in c if isinstance(x, str))
                                entry["user_summary"] = f"{n_text} text + {n_images} images"
                            elif isinstance(c, str):
                                entry["text"] = c[:500]

                        elif 'text' in kind_lower or 'thinking' in kind_lower:
                            entry["text"] = str(getattr(part, 'content', ''))[:2000]

                        message_log.append(entry)
                turn_idx += 1

            agent_stats["tool_calls"] = tool_calls
            agent_stats["total_tool_calls"] = sum(tool_calls.values())
            agent_stats["n_turns"] = turn_idx

            # Per-case geocode-type breakdown: which `type=` did the agent
            # actually pass to geocode()? Tells us whether agent prefers
            # gpkg_place over place, used wikidata, etc.
            geocode_types = {}
            validator_retries = 0
            for e in message_log:
                if e.get("tool") == "geocode":
                    args = e.get("args", {})
                    if isinstance(args, dict):
                        t = args.get("type", "?")
                        geocode_types[t] = geocode_types.get(t, 0) + 1
                if e.get("kind", "").lower().startswith("retryprompt"):
                    validator_retries += 1
            agent_stats["geocode_types"] = geocode_types
            agent_stats["validator_retries"] = validator_retries
            # Extract worker phase usage
            usage = result.usage()
            agent_stats["worker_request_tokens"] = usage.request_tokens
            agent_stats["worker_response_tokens"] = usage.response_tokens
            # Total = reader + worker
            reader_total = sum(reader_tokens.values()) if reader_tokens else 0
            worker_total = (usage.request_tokens or 0) + (usage.response_tokens or 0)
            agent_stats["request_tokens"] = (reader_tokens.get("request", 0) or 0) + (usage.request_tokens or 0)
            agent_stats["response_tokens"] = (reader_tokens.get("response", 0) or 0) + (usage.response_tokens or 0)
            agent_stats["total_tokens"] = reader_total + worker_total
    except Exception:
        pass

    # Merge accumulated multi-page boundaries into a single MultiPolygon.
    final_geojson = state.current_result.get("geojson")
    if state.accumulated_geojson:
        all_features = list(state.accumulated_geojson)
        if final_geojson and final_geojson not in all_features:
            all_features.append(final_geojson)
        final_geojson = _merge_geojson(all_features)
        if verbose:
            print(f"  Merged {len(all_features)} page boundaries into final GeoJSON")

    # Soft quality gate: flag LOW_QUALITY but never null the geojson —
    # partial IoU always beats no prediction.
    final_mi = state.current_result.get("match_info") or {}
    if (final_mi or agent_rejected) and not state.accumulated_geojson:
        _inl = final_mi.get("n_inliers", 0) or 0
        _score = final_mi.get("score", 0) or 0
        quant_reject = (_inl < 25 and _score < 15)
        if quant_reject or agent_rejected:
            prev_reason = state.accept_reason[:160] if state.accept_reason else ""
            if agent_rejected and not quant_reject:
                gate_reason = (
                    f"LOW_QUALITY (agent visual check flagged) "
                    f"(inliers={_inl}, score={_score:.1f}): {prev_reason}"
                )
            else:
                gate_reason = (
                    f"LOW_QUALITY (inliers={_inl} < 25, score={_score:.1f} "
                    f"< 15). Agent said: {prev_reason}"
                )
            state.accepted = False
            state.accept_reason = gate_reason
            if verbose:
                src = "agent visual" if agent_rejected and not quant_reject else "quality gate"
                print(f"  {src.upper()}: flagging low-quality (inliers={_inl}, "
                      f"score={_score:.1f}) - keeping geojson for partial IoU")

    # Fallback: if MINIMA never produced a geojson, project the SAM mask at
    # the best geocode anchor using the known scale. Low accuracy but
    # non-zero overlap on cases where MINIMA failed entirely.
    if final_geojson is None and state.current_mask is not None \
            and state.centers:
        final_geojson = _fallback_geojson_at_anchor(
            state.current_mask, state.centers, state.scale_ratio or 5000,
            state.dpi or 200)
        if final_geojson is not None:
            if verbose:
                print("  FALLBACK: no MINIMA geojson, synthesised from "
                      "best anchor — partial IoU only")
            if state.accept_reason:
                state.accept_reason = f"FALLBACK_ANCHOR | {state.accept_reason}"
            else:
                state.accept_reason = "FALLBACK_ANCHOR: no positioning, placed at best geocode anchor"

    return {
        "success": True,
        "geojson": final_geojson,
        "match_info": state.current_result.get("match_info", {}),
        "mask": state.current_mask,
        "affine_H": state.current_result.get("affine_H"),
        "tile_info_meta": {
            k: v for k, v in (state.current_result.get("tile_info") or {}).items()
            if k != "image"  # don't return the tile image array
        },
        "agent_accepted": state.accepted,
        "agent_reason": state.accept_reason,
        "agent_stats": agent_stats,
        "message_log": message_log,
        "candidate_overlays": state.candidate_overlays,
        "selected_overlay": state.selected_overlay,
        "selected_indices": state.selected_indices,
        # Phase 3 (Commenter critic) artifacts
        "critic_iterations": state.critic_iterations,
        "critic_final_decision": state.critic_final_decision,
        "critic_changed_mask": state.critic_changed_mask,
        "critic_applied_rotation_deg": state.critic_applied_rotation_deg,
        "critic_suspected_wrong_location": state.critic_suspected_wrong_location,
        "critic_worker_reentered": state.critic_worker_reentered,
        "critic_panel_img": (critic_result.get("panel_img_iter0")
                              if critic_result else None),
        "critic_tokens": (critic_result.get("tokens")
                           if critic_result else None),
        # Geocoding transparency
        "centers_tried": state.centers_tried,
    }
