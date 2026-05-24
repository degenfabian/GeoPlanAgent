#!/usr/bin/env python3
"""
VLM Segmentation Ablation
=========================

Asks a vision-LLM (Gemini Flash/Pro/etc. via OpenRouter) to segment the
planning boundary directly from the rendered map page. Reports pixel IoU
against the human-traced ground-truth masks.

Usage
-----
Quick prompt iteration on 3 cases:
    uv run python ablations/vlm_segmentation.py --model gemini-flash --max-cases 3

Full run (all annotated_pages cases, all folds):
    uv run python ablations/vlm_segmentation.py --model gemini-flash
"""

import argparse
import csv
import io
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, NativeOutput
from pydantic_ai.usage import UsageLimits
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from geoplanagent.paths import ABL_VLM_SEG, TRAINING_DATASET_DIR
from geoplanagent.utils import resolve_model, normalise_case_name
from ablations.utils import iou_score, print_summary, summarise  # noqa: E402

load_dotenv()

# Output schema (enforced by pydantic-ai)


class Polygon(BaseModel):
    """One boundary polygon: ordered (y, x) integer vertices in [0, 1000] —
    Gemini-native bounding-box / segmentation coordinate convention."""

    vertices: List[List[int]] = Field(
        description=(
            "Ordered list of (y, x) integer vertices defining the polygon. "
            "Each value is an integer in [0, 1000]. y is the row index "
            "(top→bottom); x is the column index (left→right). Origin "
            "(0, 0) is the TOP-LEFT corner. Minimum 3 vertices."
        )
    )


class VlmSegmentation(BaseModel):
    """Boundary segmentation output: 0+ polygons (MultiPolygon allowed)."""

    polygons: List[Polygon] = Field(
        description=(
            "One or more boundary polygons. Output one polygon per "
            "disjoint boundary region on the page. If the page shows "
            "no drawn boundary, return an empty list."
        )
    )
    notes: str = Field(
        default="",
        description=(
            "One-sentence description of what you traced and the style "
            "of the boundary (e.g. 'red solid outline around the field "
            "north of the farmhouse')."
        ),
    )


DEFAULT_PROMPT = """You are a UK planning-permission boundary segmentation model.

Your input is a single map page from a UK planning document. Somewhere on the
page there is a DRAWN BOUNDARY marking the area of an application, an
Article 4 direction, a conservation area, or another planning designation.
The boundary may be:
  - A solid line in any colour (red, black, blue, green, pink, …)
  - A dashed or dotted line
  - A hatched region (parallel-line shading or cross-hatching)
  - A solid colour fill / wash
  - An outline overlaid on an OS-style or aerial map background

Your task: locate the boundary and trace it as a polygon.

If the page includes a LEGEND or KEY, read it first to identify which
colour, line style, or symbol denotes the application / site / Article 4
boundary — this disambiguates it from other markings. Trace only what
the legend identifies as the relevant planning boundary; do NOT trace
the legend swatches themselves.

OUTPUT FORMAT (enforced by the schema)
- One or more polygons. If the page shows multiple disjoint boundary
  regions, output one polygon per region.
- Each vertex is an integer (y, x) pair in [0, 1000] — y first (row,
  top→bottom), x second (column, left→right). Origin (0, 0) is the
  top-left corner. This is your native bounding-box / segmentation
  coordinate convention.

WHAT TO IGNORE
- Page frame / border / fold marks
- Title block, scale bar, north arrow
- Text annotations and labels
- The base map cartography (roads, buildings, etc.) UNLESS the boundary
  specifically follows those features"""


# Pydantic-ai agent


def build_agent(instructions: str, temperature: float = 0.0) -> Agent:
    return Agent(
        "test",  # model is overridden per-call
        # NativeOutput → pydantic-ai uses Gemini's native response_format /
        # json_schema mode instead of the default tool-call mechanism.
        # Avoids the tool-call framing overhead.
        output_type=NativeOutput(VlmSegmentation),
        retries=3,
        output_retries=0,
        model_settings={
            "temperature": temperature,
            # 100-vertex multi-polygon JSON serialises to ~3-4K tokens.
            # 8K cap is comfortable for any reasonable boundary.
            "max_tokens": 8192,
        },
        instructions=instructions,
    )


# Rasterization + IoU


def rasterize_polygons(polys: List[Polygon], width: int, height: int) -> np.ndarray:
    """Render (y, x) [0, 1000] polygons to a binary HxW mask (uint8 0/1).

    Vertices use Gemini's native convention: each pair is (y, x) integers
    in [0, 1000], applied to a 1000x1000 normalised view. We scale x by
    W/1000 and y by H/1000 to map onto the actual image dimensions
    (anisotropic for non-square pages). PIL.draw.polygon expects (x, y),
    so we flip the y-first model output to x-first pixel coords here."""

    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    scale_x = width / 1000.0
    scale_y = height / 1000.0
    for poly in polys:
        if len(poly.vertices) < 3:
            continue
        try:
            pixels = [(float(vertex[1]) * scale_x, float(vertex[0]) * scale_y) for vertex in poly.vertices]
            draw.polygon(pixels, fill=255)
        except Exception:
            continue
    # Convert to 0/255 uint8.
    return (np.asarray(canvas) > 127).astype(np.uint8)


def _row(case, fold, fname, iou=None, n_polygons=0, secs="", error="", notes=""):
    """One per-case result row; failures pass iou=None plus an error string."""
    return {
        "case": case,
        "fold": fold,
        "filename": fname,
        "iou": iou,
        "n_polygons": n_polygons,
        "call_seconds": secs,
        "error": error,
        "notes": notes,
    }


def _write_outputs(rows: List[dict], out_dir, note: str = "") -> None:
    """Idempotent rewrite of results.csv from in-memory rows.

    Safe to call repeatedly mid-loop: rewrites the file in full each time
    so the on-disk state always matches `rows` exactly. Used both for the
    periodic checkpoint inside the main loop (so a Ctrl-C never loses
    work) and for the final end-of-run write.
    """
    valid = [row["iou"] for row in rows if row["iou"] is not None]
    fails = sum(1 for row in rows if row["iou"] is None)
    summary_all = summarise("VLM-direct pixel IoU (all)", valid)

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["case", "fold", "filename", "iou", "n_polygons", "call_seconds", "error", "notes"])
        for row in rows:
            writer.writerow(
                [
                    row["case"],
                    row.get("fold"),
                    row["filename"],
                    row["iou"] if row.get("iou") is not None else "",
                    row.get("n_polygons", ""),
                    row.get("call_seconds", ""),
                    (row.get("error") or "")[:100],
                    (row.get("notes") or "")[:120],
                ]
            )
    if note and note != "final":
        print(
            f"  [save] {note}: {len(rows)} rows, {fails} failures, "
            f"mean IoU {summary_all.get('mean', 0):.4f} → {csv_path.name}"
        )
    else:
        print(f"\nWrote:\n  {csv_path}")


# Main eval loop


def _run_case(i, entry, n, args, agent, model, dataset_dir, preds_dir) -> dict:
    """Run (or resume) one case and return its single result row.

    Covers the missing-file skip, the --resume cached-mask shortcut, the
    JPEG/PNG encode, the VLM call, rasterisation and IoU. Sleeps
    ``args.throttle_s`` after any real API call.

    Args:
        i: zero-based index of this case (used for the ``[i+1/n]`` log prefix).
        entry: the case dict, with "case" (case name), "filename" (the shared
            map/mask image name) and "fold".
        n: total case count (the denominator in the log prefix).
        args: parsed CLI args; reads .resume, .jpeg_quality and .throttle_s.
        agent: the pydantic-ai Agent from build_agent (carries the prompt).
        model: the resolved model to call, from resolve_model(args.model).
        dataset_dir: root holding maps/<filename> and boundary_masks/<filename>.
        preds_dir: where pred masks are written (and read back on --resume).

    Returns:
        One _row() dict {case, fold, filename, iou, n_polygons, call_seconds,
        error, notes}. ``iou`` is None on a skip (missing files) or a failure
        (API error); ``error`` then carries the reason.
    """
    case = entry["case"]
    fname = entry["filename"]
    fold = entry.get("fold")
    img_path = dataset_dir / "maps" / fname
    mask_path = dataset_dir / "boundary_masks" / fname

    if not img_path.exists() or not mask_path.exists():
        print(f"  [{i + 1:>3}/{n}] SKIP {case}: missing files")
        return _row(case, fold, fname, error="missing files")

    # --resume: if a pred_mask already exists on disk, skip the API
    # call and score from the cached mask. Lets a partial run be
    # finished without re-paying for completed cases.
    if args.resume:
        cached_pred = preds_dir / fname
        if cached_pred.exists():
            try:
                pred_arr = np.asarray(Image.open(cached_pred).convert("L"))
                gt_arr = np.asarray(Image.open(mask_path).convert("L"))
                if pred_arr.shape == gt_arr.shape:
                    cached_iou = iou_score(pred_arr, gt_arr)
                    print(f"  [{i + 1:>3}/{n}] CACHED {case[:30]:<30}  IoU={cached_iou:.4f}")
                    return _row(case, fold, fname, iou=cached_iou, n_polygons=1, notes="resumed from existing pred_mask")
                else:
                    print(
                        f"  [{i + 1:>3}/{n}] cached mask shape "
                        f"{pred_arr.shape} != gt {gt_arr.shape}; re-running"
                    )
            except Exception as error:
                print(
                    f"  [{i + 1:>3}/{n}] failed to read cached "
                    f"mask ({error!s:.60}); re-running"
                )

    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.width, img.height

    # Encode as JPEG (smaller payload, dodges 413 on large maps) or send
    # the original PNG (lossless, default).
    if args.jpeg_quality is not None:
        jpeg_buffer = io.BytesIO()
        img.save(jpeg_buffer, format="JPEG", quality=int(args.jpeg_quality))
        png_bytes = jpeg_buffer.getvalue()
        media_type = "image/jpeg"
    else:
        png_bytes = img_path.read_bytes()
        media_type = "image/png"
    gt = np.asarray(Image.open(mask_path).convert("L"))  # pure 0/255; iou_score binarises

    if gt.shape != (orig_h, orig_w):
        # GT and original map should be at identical resolution per
        # the fine-tune training contract; flag loudly if they aren't.
        print(
            f"  [{i + 1:>3}] WARN {case}: gt shape {gt.shape} "
            f"!= original image (h,w)=({orig_h},{orig_w})"
        )

    # Call the VLM
    t_call = time.time()
    try:
        result = agent.run_sync(
            [
                BinaryContent(data=png_bytes, media_type=media_type),
                "Locate the drawn site boundary and output it per the schema.",
            ],
            model=model,
            usage_limits=UsageLimits(request_limit=4),
        )
        dt_call = time.time() - t_call
        polys = result.output.polygons
        notes = result.output.notes
        # Range sanity check: warn loudly if any vertex falls outside
        # [0, 1000] — the model may have ignored the convention and
        # emitted pixel coords or normalised floats instead.
        n_oor = sum(
            1
            for poly in polys
            for vertex in poly.vertices
            if not (0 <= vertex[0] <= 1000 and 0 <= vertex[1] <= 1000)
        )
        if n_oor > 0:
            example = next(
                (
                    vertex
                    for poly in polys
                    for vertex in poly.vertices
                    if not (0 <= vertex[0] <= 1000 and 0 <= vertex[1] <= 1000)
                ),
                None,
            )
            print(
                f"  [{i + 1:>3}] WARN {case}: {n_oor} vertices out of "
                f"[0, 1000] (e.g. {example}). Model may have ignored "
                f"the (y, x) ∈ [0, 1000] convention."
            )
    except Exception as error:
        print(f"  [{i + 1:>3}/{n}] FAIL {case[:30]}  {type(error).__name__}: {str(error)[:80]}")
        time.sleep(args.throttle_s)
        return _row(case, fold, fname, error=f"{type(error).__name__}: {str(error)[:200]}")

    # Rasterize at the ORIGINAL map resolution — polygons come back in
    # normalised [0,1000] coords, so the mask lines up with the GT.
    pred = rasterize_polygons(polys, orig_w, orig_h)
    score = iou_score(pred, gt)

    # Save pred mask for inspection
    Image.fromarray((pred * 255).astype(np.uint8)).save(preds_dir / f"{fname}")

    mark = "PASS" if score >= 0.8 else "OK  " if score >= 0.5 else "WEAK"
    print(
        f"  [{i + 1:>3}/{n}] {mark} {case[:30]:30s}  "
        f"IoU={score:.4f}  polys={len(polys)}  "
        f"({dt_call:.1f}s)"
    )
    time.sleep(args.throttle_s)
    return _row(case, fold, fname, iou=score, n_polygons=len(polys), secs=round(dt_call, 2), notes=notes[:160])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="gemini-flash",
        help="OpenRouter alias (gemini-flash, gemini-pro, …) or full ID",
    )
    parser.add_argument(
        "--max-cases", type=int, default=None, help="Cap on number of cases (for quick iteration)"
    )
    parser.add_argument("--out-dir", default=str(ABL_VLM_SEG))
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Only run these specific case identifiers. Useful for re-running failed cases.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        help="If set, re-encode the input image as JPEG at this "
        "quality (1-100) before sending. Cuts request payload "
        "without resizing — useful for cases that hit 413 "
        "errors on large PNGs. Recommended: 95.",
    )
    parser.add_argument(
        "--throttle-s",
        type=float,
        default=1.0,
        help="Seconds to sleep between API calls (rate-limit safety)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0). The paper runs "
        "used 1.0: temperature 0 caused frequent API errors..",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Rewrite results.csv every N cases (in "
        "addition to the end-of-run write). Ensures Ctrl-C "
        "never loses already-paid-for work. Default: 5.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases whose pred_mask already exists in "
        "out_dir/pred_masks; compute IoU from disk and "
        "include in results.csv. Use to continue an "
        "aborted run without re-paying for completed cases.",
    )
    args = parser.parse_args()

    from ablations.utils import load_annotated_pages

    dataset_dir = TRAINING_DATASET_DIR
    annotated_pages = load_annotated_pages(REPO)

    if args.cases:
        wanted = set(args.cases)
        annotated_pages = [page for page in annotated_pages if page.get("case") in wanted]
        missing = wanted - {page.get("case") for page in annotated_pages}
        if missing:
            print(f"WARNING: {len(missing)} requested cases not in annotated_pages: {sorted(missing)}")
    if args.max_cases:
        annotated_pages = annotated_pages[: args.max_cases]

    print(f"annotated_pages: {len(annotated_pages)} cases")

    instructions = DEFAULT_PROMPT

    # Output paths
    out_dir = Path(args.out_dir) / normalise_case_name(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = out_dir / "pred_masks"
    preds_dir.mkdir(exist_ok=True)
    print(f"output: {out_dir}")

    # Build agent
    agent = build_agent(instructions, temperature=args.temperature)
    model = resolve_model(args.model)
    print(
        f"agent: model={args.model}  temperature={args.temperature}  save_every={args.save_every}"
    )

    rows = []
    t0 = time.time()
    n = len(annotated_pages)

    for i, entry in enumerate(annotated_pages):
        rows.append(_run_case(i, entry, n, args, agent, model, dataset_dir, preds_dir))

        # Periodic save so a Ctrl-C never loses already-paid-for work.
        if args.save_every > 0 and (i + 1) % args.save_every == 0:
            _write_outputs(rows, out_dir, note=f"checkpoint after case {i + 1}/{n}")

    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed:.0f}s")

    # Aggregate (final)
    valid = [row["iou"] for row in rows if row["iou"] is not None]
    fails = sum(1 for row in rows if row["iou"] is None)
    print("\n" + "=" * 60)
    print(f"AGGREGATE — model={args.model}  temperature={args.temperature}")
    print(f"failures: {fails}/{len(rows)}")
    print("=" * 60)
    summary_all = summarise("VLM-direct pixel IoU (all)", valid)
    print_summary(summary_all)

    _write_outputs(rows, out_dir, note="final")
    print(f"Pred masks in: {preds_dir}")


if __name__ == "__main__":
    main()
