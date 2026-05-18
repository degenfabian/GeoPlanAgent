"""Match-stage worker tools: match_at + commit_match (multi-group capable).

Each match_at call:
  - takes an explicit `page` argument (the worker chooses which page to
    use for the area_group it wants to override)
  - resolves the full set of area_groups in the document and the page
    each should be matched on (worker's choice for its group, primaries
    for the others)
  - per group: lazily renders the page, lazily segments with SAM3
    (caches both per page across calls), runs MINIMA at the supplied
    (lat, lon) centre, projects the mask through the resulting affine
  - drops groups whose match fails the strict commit gate
  - unions remaining per-group polygons into a single GeoJSON
  - stores ONE candidate; commit_match commits it as-is.

SAM3 masks are cached on state.sam_masks_by_page keyed by 1-based page
number, so re-calling match_at on a page that's been segmented is fast
(MINIMA only).
"""

from __future__ import annotations

import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic_ai import ModelRetry, RunContext, ToolReturn

from tools.agent.state import (
    _agent,
    AgentState,
    _dedup_check,
    _img_to_binary,
)


# No inlier-count gate. A group "passes" iff MINIMA produced a valid
# affine_H (and therefore a geojson) for it. The mathematical floor is
# 3 inlier point pairs (6 equations, 6 unknowns), but MINIMA's internal
# RANSAC already enforces that — if we got a geojson back, we trust it.

# Fixed query for SAM3 semantic segmentation. The LoRA was trained against
# this literal phrase.
_SAM3_QUERY = "planning boundary"


# ── Per-page render + segmentation helpers ──────────────────────────────

def _get_or_render_page(state: AgentState, page: int) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Return (map_img, map_crop_path) for `page`. Cache on first need."""
    cached = state.rendered_pages.get(page)
    cached_path = state.rendered_page_paths.get(page)
    if cached is not None and cached_path is not None:
        return cached, cached_path

    from tools.io.map_page import render_map_page
    rendered = render_map_page(state.pdf_path, page, dpi=state.dpi,
                                  verbose=False, case_name=state.case_name)
    if rendered is None:
        return None, None
    map_img, rot_info = rendered
    if rot_info.get("applied"):
        state.rotation_checked = True
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    cv2.imwrite(path, map_img)
    state.rendered_pages[page] = map_img
    state.rendered_page_paths[page] = path
    return map_img, path


def _get_or_compute_mask(state: AgentState, page: int,
                          map_crop_path: str) -> Optional[np.ndarray]:
    """Return SAM3 mask for `page`. Compute + cache on first need."""
    cached = state.sam_masks_by_page.get(page)
    if cached is not None:
        return cached
    from tools.extraction.sam3 import (extract_boundary_sam3_semantic,
                                        set_fold_for_case)
    set_fold_for_case(state.sam3_state, state.case_name)
    mask = extract_boundary_sam3_semantic(
        map_crop_path, state.sam3_processor, state.sam3_model,
        state.device, query=_SAM3_QUERY,
    )
    if mask is not None:
        state.sam_masks_by_page[page] = mask
    return mask


def _groups_to_match(state: AgentState,
                       requested_page: int) -> List[Tuple[int, int]]:
    """Return [(area_group_id, page_to_match), ...] across all match groups.

    For the area_group of `requested_page`, use that page.
    For all other area_groups, use the primary (first map_pages entry of
    that group).
    """
    details = (state.pdf_info or {}).get("map_page_details") or []
    map_pages = (state.pdf_info or {}).get("map_pages") or []
    if not details or not map_pages:
        return [(0, requested_page)]

    by_page = {int(d["page"]): d for d in details
               if d.get("category") == "match"}
    req_meta = by_page.get(int(requested_page))
    if req_meta is None:
        raise ModelRetry(
            f"page={requested_page} is not a category='match' page. "
            f"Valid match pages: {sorted(by_page.keys())}. "
            f"Pick one from pdf_info.map_pages."
        )
    req_group = int(req_meta.get("area_group", 0))

    # Walk map_pages in order; pick first match page per area_group.
    seen: set = set()
    out: List[Tuple[int, int]] = []
    for page in map_pages:
        page = int(page)
        meta = by_page.get(page)
        if meta is None:
            continue
        g = int(meta.get("area_group", 0))
        if g in seen:
            continue
        seen.add(g)
        if g == req_group:
            out.append((g, int(requested_page)))
        else:
            out.append((g, page))
    return out or [(req_group, requested_page)]


# ── Visual helpers ───────────────────────────────────────────────────────

def _build_match_at_panel(map_img: np.ndarray,
                            tile_info: Dict[str, Any],
                            mi: Dict[str, Any],
                            label: str = "PLANNING MAP") -> Optional[np.ndarray]:
    """Visual panel for one group's match attempt."""
    if map_img is None or not isinstance(tile_info, dict) or "image" not in tile_info:
        return None
    target_h = 500

    def _label(img, text):
        bar = np.full((28, img.shape[1], 3), 30, dtype=np.uint8)
        cv2.putText(bar, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([bar, img])

    h, w = map_img.shape[:2]
    map_resized = cv2.resize(map_img, (max(1, int(w * target_h / h)), target_h))
    tile_img = tile_info["image"]
    if tile_img.shape[2] == 3 and tile_info.get("_was_rgb", True):
        tile_bgr = cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR)
    else:
        tile_bgr = tile_img.copy()
    th, tw = tile_bgr.shape[:2]
    wx, wy = mi.get("window") or (0, 0)
    sf = mi.get("scale_factor") or 1.0
    win_w = max(1, int(map_img.shape[1] * sf))
    win_h = max(1, int(map_img.shape[0] * sf))
    rect_img = tile_bgr.copy()
    cv2.rectangle(rect_img, (int(wx), int(wy)),
                  (int(wx + win_w), int(wy + win_h)),
                  (0, 0, 255), max(2, th // 200))
    tile_resized = cv2.resize(rect_img,
                              (max(1, int(tw * target_h / th)), target_h))
    left = _label(map_resized, label)
    right = _label(tile_resized,
                   f"OS TILES z={mi.get('zoom')} "
                   f"({(mi.get('center_latlon') or ['?','?'])[0]:.4f},"
                   f"{(mi.get('center_latlon') or ['?','?'])[1]:.4f}) "
                   f"— red = matched window")
    panel = np.hstack([left, right])
    if panel.shape[1] > 1800:
        s = 1800 / panel.shape[1]
        panel = cv2.resize(panel, (1800, int(panel.shape[0] * s)))
    return panel


def _stack_panels(panels: List[np.ndarray]) -> np.ndarray:
    """Vertically stack per-group panels, padding to a common width."""
    if not panels:
        return None
    max_w = max(p.shape[1] for p in panels)
    out = []
    for p in panels:
        if p.shape[1] < max_w:
            pad = np.full((p.shape[0], max_w - p.shape[1], 3), 200, dtype=np.uint8)
            p = np.hstack([p, pad])
        out.append(p)
        out.append(np.full((6, max_w, 3), 200, dtype=np.uint8))
    return np.vstack(out[:-1])


# ── match_at ─────────────────────────────────────────────────────────────

@_agent.tool
def match_at(
    ctx: RunContext[AgentState],
    page: int,
    name: str,
    lat: float,
    lon: float,
    sigma_m: Optional[float] = None,
    scale_ratio: Optional[float] = None,
) -> ToolReturn:
    """Run MINIMA at (lat, lon); auto-matches every area_group in the doc.

    The `page` argument selects which page to use FOR ITS area_group.
    Other area_groups in the document use their primaries automatically.
    The returned candidate's polygon is the UNION of per-group projections.

    Args:
        page: 1-based page number. Must be a category='match' page from
            the reader's map_pages list.
        name: Short label, e.g. "gpkg:Hampstead Heath".
        lat / lon: Centre latitude / longitude (must come from
            propose_centers — fabricated coordinates are rejected).
        sigma_m: Search radius in metres (default: scale-aware).
        scale_ratio: Map scale denominator (default: parsed from PDFInfo.scale).

    Returns:
        {"success": True, "candidate_id": int, "overall_score": float,
         "n_groups_committed": int, "per_group": [...]} + multi-panel viz.
    """
    state = ctx.deps
    if state.match_at_budget <= 0:
        raise ModelRetry(
            "match_at budget exhausted. Pick the best stored candidate via "
            "commit_match and proceed — the pipeline always produces a "
            "polygon, even if the best score is low."
        )

    _dedup_check(state, "match_at", {
        "page": int(page), "name": name,
        "lat": round(float(lat), 5), "lon": round(float(lon), 5),
        "sigma_m": sigma_m, "scale_ratio": scale_ratio,
    })

    # Reject invented coordinates (same logic as before).
    matched_candidate = None
    if state.proposed_centers:
        from tools.geo.coords import haversine_m as _distance_m
        nearest = min(
            (_distance_m(lat, lon, c["lat"], c["lon"]), c)
            for c in state.proposed_centers
        )
        if nearest[0] > 100.0:
            avail = ", ".join(
                f"id={c['id']} ({c['source'][:30]})"
                for c in state.proposed_centers[:8]
            )
            raise ModelRetry(
                f"match_at refuses fabricated coordinates "
                f"({lat:.5f}, {lon:.5f}) — nearest propose_centers entry "
                f"is {nearest[0]:.0f}m away ({nearest[1]['source']}). "
                f"Use a (lat, lon) from a propose_centers candidate "
                f"directly. Available: {avail}. If none look right, call "
                f"propose_centers(extra_terms=[...]) — do NOT invent "
                f"coordinates."
            )
        matched_candidate = nearest[1]

    state.match_at_budget -= 1

    # σ resolution.
    from tools.matching import sigma_from_scale

    def _parse_scale(s: Any) -> Optional[int]:
        if not s:
            return None
        import re
        m = re.search(r"1\s*[:/]\s*([\d,]+)", str(s))
        if not m:
            return None
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            return None

    if sigma_m is None and matched_candidate is not None:
        cand_sigma = matched_candidate.get("sigma_m")
        if cand_sigma is not None and float(cand_sigma) > 0:
            sigma_m = float(cand_sigma)
    if sigma_m is None:
        sr = scale_ratio
        if sr is None and state.pdf_info:
            sr = _parse_scale(state.pdf_info.get("scale"))
        sigma_m = sigma_from_scale(sr)
    if scale_ratio is None and state.pdf_info:
        scale_ratio = _parse_scale(state.pdf_info.get("scale"))

    # Resolve the per-group page list.
    groups_pages = _groups_to_match(state, int(page))

    per_group: List[Dict[str, Any]] = []
    panels: List[np.ndarray] = []
    requested_group = None
    for d in (state.pdf_info or {}).get("map_page_details") or []:
        if int(d.get("page", -1)) == int(page) and d.get("category") == "match":
            requested_group = int(d.get("area_group", 0))
            break

    for group_id, group_page in groups_pages:
        single = _match_single_page(state, group_page, name, float(lat), float(lon),
                                       float(sigma_m), scale_ratio, matched_candidate)
        single["area_group"] = int(group_id)
        single["page"] = int(group_page)
        per_group.append(single)
        if single.get("panel") is not None:
            label = (f"PAGE {group_page} (grp {group_id}"
                     f"{', requested' if group_id == requested_group else ''})")
            panel = _build_match_at_panel(
                state.rendered_pages.get(group_page),
                single.get("tile_info"), single.get("match_info") or {},
                label=label,
            )
            if panel is not None:
                panels.append(panel)
        # Safe even when `_match_single_page` took an error path (no 'panel' key).
        single.pop("panel", None)

    # Aggregate metrics across groups that produced a valid match.
    valid = [g for g in per_group
             if g.get("affine_H") is not None
             and not g.get("error")]
    total_inliers = sum(int((g.get("match_info") or {}).get("n_inliers") or 0)
                        for g in valid)
    # Weighted-mean overall_score, weighted by per-group n_inliers.
    overall_score = 0.0
    if valid:
        weights = [max(1, int((g.get("match_info") or {}).get("n_inliers") or 1))
                   for g in valid]
        scores = [float(g.get("overall_score") or 0.0) for g in valid]
        overall_score = float(np.average(scores, weights=weights))

    # Union per-group GeoJSONs that passed the per-group commit gate.
    committed_groups = []
    for g in valid:
        if g.get("geojson") is not None:
            committed_groups.append(g)

    unioned_geojson = _union_geojsons([g["geojson"] for g in committed_groups])

    cid = state._match_attempt_counter
    state._match_attempt_counter += 1
    state.match_attempts[cid] = {
        "candidate_id": cid,
        "name": name, "lat": float(lat), "lon": float(lon),
        "sigma_m": float(sigma_m), "scale_ratio": scale_ratio,
        "per_group": per_group,
        "committed_groups_idx": [i for i, g in enumerate(per_group)
                                  if g in committed_groups],
        "geojson": unioned_geojson,
        "overall_score": float(overall_score),
        "total_inliers": int(total_inliers),
        "n_groups": len(per_group),
        "n_groups_committed": len(committed_groups),
        "requested_page": int(page),
        "requested_group": requested_group,
    }

    summary = {
        "success": True,
        "candidate_id": cid,
        "overall_score": float(overall_score),
        "total_inliers": int(total_inliers),
        "n_groups": len(per_group),
        "n_groups_committed": len(committed_groups),
        "per_group": [
            {
                "page": g["page"], "area_group": g["area_group"],
                "n_inliers": int((g.get("match_info") or {}).get("n_inliers") or 0),
                "score": float((g.get("match_info") or {}).get("score") or 0.0),
                "overall_score": float(g.get("overall_score") or 0.0),
                "passed_gate": g in committed_groups,
                "weak_retry": g.get("weak_retry"),
            }
            for g in per_group
        ],
        "budget_remaining": state.match_at_budget,
    }

    if not panels:
        return summary
    big = _stack_panels(panels)
    text = (f"match_at id={cid}, overall_score={overall_score:.2f}, "
            f"{len(committed_groups)}/{len(per_group)} groups passed gate. "
            f"Per-group panels stacked vertically — each shows that group's "
            f"page (left) vs OS tiles at the matched window (right). "
            f"For multi-group docs the final polygon is the UNION of all "
            f"groups that passed.")
    return ToolReturn(return_value=summary,
                      content=[text, _img_to_binary(big)])


# ── Per-page MINIMA driver (called once per group inside match_at) ──────

def _match_single_page(state: AgentState, page: int, name: str,
                        lat: float, lon: float, sigma_m: float,
                        scale_ratio: Optional[int],
                        matched_candidate: Optional[dict]) -> Dict[str, Any]:
    """Render+segment+MINIMA on a single page at (lat, lon). Returns a dict
    with affine_H / tile_info / match_info / geojson / mask_frac /
    overall_score / panel-data; or error."""
    map_img, map_crop_path = _get_or_render_page(state, page)
    if map_img is None or map_crop_path is None:
        return {"error": f"render failed for page {page}"}
    mask = _get_or_compute_mask(state, page, map_crop_path)
    if mask is None:
        return {"error": f"SAM3 returned no mask for page {page}"}
    mask_frac = float(np.sum(mask > 0)) / float(mask.size)

    # Analytical short-circuit on this page.
    analytical = _try_analytical_match_at(
        state, page, map_img, mask, name, lat, lon, scale_ratio,
    )
    if analytical is not None:
        return analytical

    from tools.matching import sliding_window_position, mask_to_geojson_affine
    from tools.metrics.reward import compute_match_reward

    def _run_minima(sigma_used):
        return sliding_window_position(
            matcher=state.minima_matcher, map_img=map_img,
            sam3_mask=mask, centers=[(name, lat, lon, sigma_used)],
            scale_ratio=scale_ratio, dpi=state.dpi,
            rotations=None,
            road_names=state.pdf_info.get("road_names") or [],
            grayscale=False, return_candidates=False,
            directional_modifier=state.pdf_info.get("directional_modifier"),
        )

    def _evaluate(res):
        if not res or res.get("affine_H") is None:
            return None, None
        mi_local = res.get("match_info") or {}
        rw = compute_match_reward(
            match_info=mi_local, pdf_info=state.pdf_info,
            inlier_pts_in_map=None, map_shape_hw=tuple(map_img.shape[:2]),
        )
        return mi_local, rw

    try:
        result = _run_minima(sigma_m)
    except Exception as e:
        return {"error": f"sliding_window_position: {e!s:.140}"}
    if not result or result.get("affine_H") is None:
        return {"error": "MINIMA returned no usable match",
                "mask_frac": mask_frac}

    mi, reward = _evaluate(result)

    # 2× σ retry on weak first attempt. Telemetry on whether the retry
    # fired + whether it strictly improved overall_score is persisted in
    # the return dict so post-benchmark analysis can compute rescue rate.
    weak_retry = {
        "fired": False,
        "kept": False,
        "original_n_inliers": int((mi or {}).get("n_inliers") or 0),
        "original_overall_score": float(reward.overall_score) if reward is not None else 0.0,
        "retry_n_inliers": None,
        "retry_overall_score": None,
        "retry_sigma_m": None,
    }
    weak = (int((mi or {}).get("n_inliers") or 0) < 25
            or float(reward.overall_score) < 0.4)
    if weak:
        weak_retry["fired"] = True
        weak_retry["retry_sigma_m"] = float(sigma_m * 2.0)
        try:
            retry_result = _run_minima(sigma_m * 2.0)
        except Exception:
            retry_result = None
        if retry_result and retry_result.get("affine_H") is not None:
            retry_mi, retry_reward = _evaluate(retry_result)
            weak_retry["retry_n_inliers"] = int((retry_mi or {}).get("n_inliers") or 0)
            weak_retry["retry_overall_score"] = (
                float(retry_reward.overall_score) if retry_reward is not None else 0.0)
            if (retry_reward is not None
                    and retry_reward.overall_score > reward.overall_score):
                weak_retry["kept"] = True
                print(f"    [weak-retry] page {page}: "
                      f"score {weak_retry['original_overall_score']:.2f} → "
                      f"{weak_retry['retry_overall_score']:.2f}, "
                      f"n_inliers {weak_retry['original_n_inliers']} → "
                      f"{weak_retry['retry_n_inliers']}, σ "
                      f"{int(sigma_m)} → {int(sigma_m * 2.0)}m")
                result = retry_result
                mi = retry_mi
                reward = retry_reward
            else:
                print(f"    [weak-retry] page {page}: not kept "
                      f"(score {weak_retry['original_overall_score']:.2f} ≥ "
                      f"{weak_retry['retry_overall_score']:.2f})")
        else:
            print(f"    [weak-retry] page {page}: retry returned no usable match")

    affine_H = result.get("affine_H")
    tile_info = result.get("tile_info")
    geojson = result.get("geojson")
    if geojson is None and affine_H is not None and tile_info is not None:
        geojson = mask_to_geojson_affine(mask, affine_H, tile_info)

    return {
        "affine_H": affine_H, "tile_info": tile_info,
        "match_info": mi, "geojson": geojson,
        "reward": reward.to_dict() if reward is not None else None,
        "overall_score": float(reward.overall_score) if reward is not None else 0.0,
        "mask_frac": mask_frac,
        "weak_retry": weak_retry,
        "panel": True,
    }


# ── Polygon union helper ────────────────────────────────────────────────

def _union_geojsons(geojsons: List[dict]) -> Optional[dict]:
    """shapely-union per-group GeoJSON Features → one combined Feature.

    Single input → return as-is. Empty → None. Multiple → MultiPolygon.
    """
    if not geojsons:
        return None
    if len(geojsons) == 1:
        return geojsons[0]

    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union

    geoms = []
    properties: Dict[str, Any] = {}
    for g in geojsons:
        if not isinstance(g, dict):
            continue
        geom = g.get("geometry") or g
        try:
            geoms.append(shape(geom))
        except Exception:
            continue
        if not properties:
            properties = dict(g.get("properties") or {})
    if not geoms:
        return None
    union = unary_union(geoms)
    if union.is_empty or union.geom_type not in ("Polygon", "MultiPolygon"):
        return None
    out = {
        "type": "Feature",
        "geometry": mapping(union),
        "properties": {**properties, "source": "match_at_union"},
    }
    # Normalise to MultiPolygon for downstream.
    if out["geometry"].get("type") == "Polygon":
        out["geometry"] = {
            "type": "MultiPolygon",
            "coordinates": [out["geometry"]["coordinates"]],
        }
    return out


# ── commit_match ─────────────────────────────────────────────────────────

@_agent.tool
def commit_match(ctx: RunContext[AgentState], candidate_id: int) -> dict:
    """Mark a stored match_at candidate as the active result.

    For multi-group docs the candidate's geojson is already the union
    across area_groups for which MINIMA produced a valid affine. The
    smart-commit gate below redirects to a better candidate if the
    worker has tried multiple match_at calls and picked a worse one;
    the strict gate rejects commits where NO group produced an affine
    (and no analytical short-circuit fired) — try a different page or
    centre via match_at, or call propose_centers(extra_terms=[…]).

    Args:
        candidate_id: ID returned from a prior match_at call.
    """
    state = ctx.deps
    cand = state.match_attempts.get(int(candidate_id))
    if cand is None:
        raise ModelRetry(
            f"candidate_id={candidate_id} not found. Available IDs: "
            f"{sorted(state.match_attempts.keys())}"
        )

    # Smart-commit: prefer the candidate with the most total inliers
    # across groups, weighted by inside-LA filter on the requested page's
    # match. Skips when the worker has tried <2 candidates.
    if len(state.match_attempts) >= 2:
        from tools.matching import candidate_passes_la_filter
        from tools.scoring import commit_attempt_score
        admin_region = (state.pdf_info or {}).get("admin_region") if state.pdf_info else None

        def _attempt_score(c):
            # Use total inliers across all groups for multi-group candidates.
            try:
                n = int(c.get("total_inliers") or 0)
            except (TypeError, ValueError):
                return -1
            ll = None
            for g in c.get("per_group") or []:
                mi = g.get("match_info") or {}
                if mi.get("method") in ("analytical", "analytical_affine"):
                    return float("inf")
                if ll is None:
                    ll = mi.get("center_latlon") or mi.get("chosen_center_latlon")
            inside_la = True
            if admin_region and ll:
                try:
                    inside_la = candidate_passes_la_filter(
                        "feature_cluster", ll[0], ll[1], admin_region
                    )
                except Exception:
                    inside_la = True
            return commit_attempt_score(n, inside_la)

        best_id = None
        best_score = _attempt_score(cand)
        for cid, c in state.match_attempts.items():
            if cid == int(candidate_id):
                continue
            cscore = _attempt_score(c)
            if cscore > best_score:
                best_score = cscore
                best_id = cid

        if best_id is not None:
            best_cand = state.match_attempts[best_id]
            raise ModelRetry(
                f"commit_match REJECTED candidate_id={candidate_id}. "
                f"Candidate_id={best_id} has a better commit-score "
                f"(total_inliers={best_cand.get('total_inliers', '?')}, "
                f"inside-LA-weighted). Commit candidate_id={best_id} "
                f"instead."
            )

    # Strict gate: at least one group must have passed, OR an analytical
    # short-circuit fired in some group.
    n_committed = int(cand.get("n_groups_committed") or 0)
    has_analytical = any(
        str((g.get("match_info") or {}).get("method", ""))
        in ("analytical", "analytical_affine")
        for g in cand.get("per_group") or []
    )
    if n_committed == 0 and not has_analytical:
        avail_ids = sorted(state.match_attempts.keys())
        raise ModelRetry(
            f"commit_match REJECTED candidate_id={candidate_id}: MINIMA "
            f"produced no usable affine for any group (every group is "
            f"missing affine_H/geojson). Try a different page or a "
            f"different centre via match_at; or call propose_centers"
            f"(extra_terms=[...]) to add more candidates. "
            f"Available IDs: {avail_ids}."
        )

    geojson = cand.get("geojson")
    # The committed primary is the worker's requested area_group; other
    # groups in per_group represent the auto-matched alternates that were
    # unioned in. Downstream consumers (benchmark output, verify_position)
    # use committed_primary_page(state) to derive the relevant page/mask.
    primary_group = next(
        (g for g in cand.get("per_group") or []
         if g.get("area_group") == cand.get("requested_group")),
        (cand.get("per_group") or [{}])[0],
    )
    state.current_result = {
        "affine_H": primary_group.get("affine_H"),
        "tile_info": primary_group.get("tile_info"),
        "match_info": primary_group.get("match_info"),
        "geojson": geojson,
        "candidate_id": int(candidate_id),
        "reward": primary_group.get("reward"),
        "per_group": cand.get("per_group"),
        "requested_group": cand.get("requested_group"),
        "requested_page": cand.get("requested_page"),
        "n_groups_committed": n_committed,
    }
    state.position_calls += 1

    n_polys = 0
    if isinstance(geojson, dict):
        geom = geojson.get("geometry") or {}
        if geom.get("type") == "MultiPolygon":
            n_polys = len(geom.get("coordinates") or [])
        elif geom.get("type") == "Polygon":
            n_polys = 1

    return {
        "success": True,
        "committed": {
            "candidate_id": candidate_id,
            "name": cand["name"],
            "total_inliers": int(cand.get("total_inliers") or 0),
            "overall_score": cand["overall_score"],
            "n_groups_committed": n_committed,
            "n_polygons": n_polys,
        }
    }


# ── Analytical short-circuit (per-page) ──────────────────────────────────

def _try_analytical_match_at(state: AgentState, page: int, map_img: np.ndarray,
                                mask: np.ndarray, name: str, lat: float,
                                lon: float, scale_ratio: Optional[int],
                                tolerance_m: float = 50.0):
    """Per-page analytical-affine. None if no E/N anchor near (lat, lon)
    or scale_ratio missing."""
    from tools.geo.grid_ref import parse_easting_northing
    from tools.geo.coords import haversine_m as _distance_m
    from tools.matching import (analytical_affine_from_anchor,
                                       mask_to_geojson_affine)
    from tools.metrics.reward import RewardResult, AxisResult

    if mask is None or map_img is None or scale_ratio is None:
        return None

    en_anchor = None
    for gr in (state.pdf_info or {}).get("grid_refs") or []:
        ll = parse_easting_northing(gr)
        if ll is None:
            continue
        if _distance_m(lat, lon, ll[0], ll[1]) <= tolerance_m:
            en_anchor = (ll[0], ll[1], gr)
            break
    if en_anchor is None:
        return None

    bin_m = (mask > 0).astype(np.uint8)
    M = cv2.moments(bin_m)
    if M["m00"] == 0:
        return None
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    a_lat, a_lon, gr_text = en_anchor
    affine_H, tile_info = analytical_affine_from_anchor(
        plan_shape=map_img.shape[:2],
        mask_centroid_xy=(cx, cy),
        anchor_lat=a_lat, anchor_lon=a_lon,
        scale_ratio=int(scale_ratio), dpi=int(state.dpi),
    )
    geojson = mask_to_geojson_affine(mask, affine_H, tile_info)
    match_info = {
        "center": name, "center_latlon": [a_lat, a_lon],
        "zoom": tile_info["zoom"],
        "method": "analytical", "anchor_grid_ref": gr_text,
    }
    reward = RewardResult(
        axes={"analytical": AxisResult(
            score=1.0,
            verdict=f"affine constructed from {gr_text} + scale 1:{int(scale_ratio)}",
        )},
        overall_score=0.95,
        summary=(f"Analytical affine from {gr_text} + scale 1:{int(scale_ratio)} "
                 f"@ {state.dpi}dpi (no MINIMA, page {page})"),
    )
    mask_frac = float(np.sum(mask > 0)) / float(mask.size)
    return {
        "affine_H": affine_H, "tile_info": tile_info,
        "match_info": match_info, "geojson": geojson,
        "reward": reward.to_dict(),
        "overall_score": float(reward.overall_score),
        "mask_frac": mask_frac,
        "panel": True,
    }
