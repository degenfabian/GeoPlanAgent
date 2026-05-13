"""Extract-stage worker tools: extract_boundary + project_boundary.

Extracted from ``tools/agent.py`` (stage-2 split, 2026-05-11). Registers
``extract_boundary`` and ``project_boundary`` against the shared
``_agent`` instance at import time. Also defines the private
``_get_instance_masks_rich`` helper used only by these tools.
"""

from __future__ import annotations

import os
from typing import List, Optional, Union

import cv2
import numpy as np
from pydantic_ai import ModelRetry, RunContext, ToolReturn

from tools.agent_core import (
    _agent,
    AgentState,
    _dedup_check,
    _img_to_binary,
)


def _get_instance_masks_rich(map_crop_path, processor, model, device,
                              top_k=5, bbox=None, plan_img_bgr=None):
    """Multi-prompt SAM3 + colour-line fallback. Returns (masks, labels).

    Adds candidates from alternative prompts ('site outline', 'red line
    boundary') and from HSV colour-line extraction. The trained 'planning
    boundary' prompt is always tried first, so its candidates have priority
    when ranked by SAM3 score. Colour candidates only appear when a closed
    red/blue/magenta region is detectable on the page (self-gating).
    """
    from tools.sam3_boundary import extract_candidates_multi_prompt
    cands = extract_candidates_multi_prompt(
        map_crop_path, processor, model, device,
        bbox=bbox, top_k_per_query=top_k, total_top_k=top_k + 3,
    )
    masks = [c["mask"] for c in cands]
    labels = [c.get("query", "?") for c in cands]
    if plan_img_bgr is not None and bbox is None:
        from tools.boundary_color import extract_color_boundary
        col = extract_color_boundary(plan_img_bgr)
        if col is not None and col.shape == plan_img_bgr.shape[:2]:
            masks.append(col)
            labels.append("color:red/blue")
    return masks, labels


# ── Tool 4: extract_boundary ──────────────────────────────────────────────

_FIXED_QUERY = "planning boundary"


@_agent.tool
def extract_boundary(
    ctx: RunContext[AgentState],
    mode: str = "semantic",
    select_indices: Optional[List[int]] = None,
    bbox: Optional[List[float]] = None,
) -> Union[dict, ToolReturn]:
    """Extract the planning boundary from the rendered map using SAM3.

    Text prompt is locked to "planning boundary" (the only prompt the LoRA
    was trained on). See the system prompt (step 5) for mode selection and
    bbox fallback rules.

    Args:
        mode: "semantic" (default, one merged mask) or "instance" (5
            candidates, pick with select_indices).
        select_indices: instance mode only — 0-based indices of candidates
            to combine. Call without this first to see the candidates.
        bbox: Optional [x1, y1, x2, y2] in pixels. FALLBACK only — don't
            pass on the first call. Cannot be combined with select_indices
            (bbox triggers a fresh extraction).

    Returns:
        semantic:                {"success": True, "mode": "semantic",
                                  "mask_area_pct": float}
        instance (initial):      {"success": True, "mode": "instance",
                                  "n_candidates": int, "candidates": [...]}
                                  Plus per-candidate overlay images.
        instance (with indices): {"success": True, "mode": "instance_combine",
                                  "combined_indices": [...],
                                  "mask_area_pct": float}
    """
    state = ctx.deps
    _dedup_check(state, "extract_boundary", {
        "mode": mode, "select_indices": select_indices, "bbox": bbox,
    })

    if state.map_img is None or state.map_crop_path is None:
        raise ModelRetry("No map image available. Call render_page first.")

    # Validate bbox if provided (applies to both instance and semantic modes)
    if bbox is not None:
        if select_indices is not None:
            raise ModelRetry(
                "bbox and select_indices cannot be used together. bbox "
                "triggers a FRESH extraction (gives you new candidates); "
                "select_indices picks from already-extracted candidates. "
                "Call extract_boundary with bbox alone (no select_indices) "
                "to get fresh candidates, then call again with "
                "select_indices on those.")
        if len(bbox) != 4:
            raise ModelRetry(
                f"bbox must be [x1, y1, x2, y2] (4 numbers), got {bbox}")
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            raise ModelRetry(
                f"bbox is degenerate (x1={x1} y1={y1} x2={x2} y2={y2}); "
                f"need x2 > x1 and y2 > y1")

    # Instance mode is disabled — semantic-only. Empirical evidence on the
    # v18 partial 111-case subset: 10 of 21 instance-escalation cases were
    # actually RESCUED by staying with semantic (+0.193 mean IoU). The agent
    # was escalating on "suspiciously tiny" masks which are in fact correct
    # (0.05-0.12% of image is normal for small sites). Redirect to bbox
    # refinement of semantic instead.
    if mode == "instance":
        raise ModelRetry(
            "Instance mode is disabled. The LoRA-fine-tuned SAM3 semantic "
            "head is calibrated for planning-boundary segmentation — trust "
            "its output even when the mask area is small (0.05-1% is normal "
            "for single-building sites). If the semantic mask is genuinely "
            "in the WRONG REGION or is a whole-map blob, retry with a "
            "tighter bbox: extract_boundary(bbox=[x1,y1,x2,y2]). Do NOT "
            "escalate to instance mode."
        )

    if mode == "semantic":
        if select_indices is not None:
            raise ModelRetry(
                "select_indices is for instance mode only — semantic "
                "produces one mask. Drop select_indices, or switch to "
                "mode='instance'.")
        from tools.sam3_boundary import (extract_boundary_sam3_semantic,
                                            set_fold_for_case)
        set_fold_for_case(state.sam3_state, state.case_name)
        mask = extract_boundary_sam3_semantic(
            state.map_crop_path, state.sam3_processor,
            state.sam3_model, state.device, query=_FIXED_QUERY, bbox=bbox,
        )
        if mask is None:
            return {"success": False, "error": "SAM3 semantic returned no mask"}
        area_pct = float(np.sum(mask > 0)) / mask.size * 100
        # Auto-fallback: if semantic grabbed >60% of the image, that's
        # almost always wrong (whole-map blob, like the failure case
        # 2ACB6DFF). Tell the agent to switch to instance mode instead
        # of accepting the trash mask.
        if area_pct > 60:
            raise ModelRetry(
                f"Semantic mode produced a mask covering {area_pct:.0f}% "
                f"of the image — almost certainly the whole map / legend, "
                f"not the planning boundary. Switch to "
                f"mode='instance'{' with bbox' if bbox else ''} to get "
                f"per-slot candidates you can select from.")
        state.current_mask = mask
        state.selected_indices = None
        if state.map_img is not None:
            sel_overlay = state.map_img.copy()
            sel_overlay[mask > 0] = [0, 255, 0]
            state.selected_overlay = cv2.addWeighted(
                state.map_img, 0.5, sel_overlay, 0.5, 0)
        return {"success": True, "mode": "semantic",
                "mask_area_pct": round(area_pct, 2)}

    if mode == "instance":
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
            from tools.sam3_boundary import set_fold_for_case
            set_fold_for_case(state.sam3_state, state.case_name)
            instances, labels = _get_instance_masks_rich(
                state.map_crop_path, state.sam3_processor,
                state.sam3_model, state.device,
                top_k=5, bbox=bbox, plan_img_bgr=state.map_img,
            )

            # Callout-aware reordering: when pdf_info indicates a small site
            # with "THE SITE" callout, SAM3 often picks the rectangular
            # callout BOX over the actual red site polygon. Reorder so the
            # candidate nearest the red callout target appears at index 0.
            # Statistical miner identified 4 stuck cases (A002S, A097S,
            # A100S, case 11) with this exact failure mode.
            if instances and state.map_img is not None:
                try:
                    pi = state.pdf_info or {}
                    labels_upper = [str(s).upper() for s in (pi.get("visible_map_labels") or [])]
                    has_callout = any("THE SITE" in s or s == "SITE" for s in labels_upper)
                    is_red = (pi.get("boundary_color") or "").lower() == "red"
                    if has_callout and is_red:
                        from tools.sam3_boundary import (
                            find_callout_target_centroid, rank_by_callout_proximity,
                        )
                        target = find_callout_target_centroid(state.map_img, instances)
                        best = rank_by_callout_proximity(instances, target)
                        if best is not None and best != 0:
                            instances[0], instances[best] = instances[best], instances[0]
                            labels[0], labels[best] = labels[best], labels[0]
                            print(f"  callout-aware: reordered candidate {best} → 0")
                except Exception as _e:
                    print(f"  callout-aware: skipped ({_e!s:.80})")

            state.instance_masks = instances
            if instances:
                state.current_mask = instances[0]

            content_parts = []
            summaries = []
            state.candidate_overlays = []  # reset for this extraction
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
                      (255, 255, 0), (0, 255, 255), (255, 0, 255),
                      (255, 128, 0), (128, 0, 255)]
            for i, inst in enumerate(instances[:8]):
                area_pct = np.sum(inst > 0) / inst.size * 100
                src = labels[i] if i < len(labels) else "?"
                summaries.append({"index": i, "area_pct": round(area_pct, 2),
                                    "source": src})
                overlay = state.map_img.copy()
                overlay[inst > 0] = colors[i % len(colors)]
                blended = cv2.addWeighted(state.map_img, 0.5, overlay, 0.5, 0)
                cv2.putText(blended, f"Cand {i} [{src}]", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                state.candidate_overlays.append(blended.copy())
                content_parts.append(f"Candidate {i} src={src} (area={area_pct:.1f}%):")
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

    Uses the committed match's affine transform to convert the pixel mask into
    a GeoJSON polygon with lat/lon coordinates. Requires a prior commit_match
    (for the affine) and extract_boundary (for the mask).

    Returns:
        {"success": true, "n_polygons": int}
        The GeoJSON is stored internally and used by verify_position / visualize.
    """
    state = ctx.deps

    if state.current_mask is None:
        raise ModelRetry("No boundary mask available. Call extract_boundary first.")

    affine_H = state.current_result.get("affine_H")
    tile_info = state.current_result.get("tile_info")
    if affine_H is None or tile_info is None:
        raise ModelRetry(
            "No positioning result available. Run match_at → commit_match first."
        )

    from tools.matching import mask_to_geojson_affine
    geojson = mask_to_geojson_affine(state.current_mask, affine_H, tile_info)

    if geojson is None:
        return {"success": False, "error": "Mask projection produced no polygons"}

    state.current_result["geojson"] = geojson

    # INSPIRE freehold-snap post-processing. Validated 2026-05-06 (Phase ZP
    # @5-8m tolerance): +2 cases past 0.8 IoU, 0 falls below 0.8.
    try:
        from tools.snap.inspire import InspireSnap, la_for_admin_region
        from shapely.geometry import shape as _shape, mapping as _mapping
        pi = state.pdf_info or {}
        admin = (pi.get("admin_region") or "").strip()
        la = la_for_admin_region(admin)
        if la:
            pred = _shape(geojson["geometry"])
            if pred.is_valid and not pred.is_empty:
                snap_obj = InspireSnap([la])
                snapped = snap_obj.snap_polygon(pred, max_dist_m=8.0)
                if snapped is not None and not snapped.is_empty:
                    # Update geojson if snap actually changed something
                    if not snapped.equals(pred):
                        geojson = {"type": "Feature",
                                   "properties": geojson.get("properties") or {},
                                   "geometry": _mapping(snapped)}
                        state.current_result["geojson"] = geojson
                        print(f"  INSPIRE snap: applied (LA={la})")
    except Exception as e:
        print(f"  INSPIRE snap skipped: {e!s:.80}")

    # Count polygons and estimate area
    geom = geojson.get("geometry", {})
    if geom.get("type") == "MultiPolygon":
        n_polys = len(geom.get("coordinates", []))
    elif geom.get("type") == "Polygon":
        n_polys = 1
    else:
        n_polys = 0

    return {"success": True, "n_polygons": n_polys}


# Tool 6 (accumulate_boundary) removed: multi-page handling caused
# more regressions than it solved. The pipeline now uses a single map
# page; if a document has multiple, the agent picks the best one with
# render_page.
