#!/usr/bin/env python3
"""
VLM Segmentation Ablation
=========================

Asks a vision-LLM (Gemini Flash/Pro/etc. via OpenRouter) to segment the
planning boundary directly from the rendered map page. Reports pixel IoU
against the human-traced ground-truth masks at
``training/dataset/boundary_masks/`` — the same masks used to fine-tune
SAM3 — so the numbers slot directly into the paper table next to the
SAM3-fine-tune row produced by ``scripts/eval_sam_kfold_v2.py``.

No pipeline involvement: pure single-shot VLM inference, no SAM3, no
MINIMA, no agent loop. Safe to run in parallel with the main benchmark.

Usage
-----
Quick prompt iteration on 3 cases:
    uv run python ablations/vlm_segmentation.py --model gemini-flash --max-cases 3

Full run (all manifest cases, all folds):
    uv run python ablations/vlm_segmentation.py --model gemini-flash

Held-out-fold-only (true zero-shot comparison vs SAM3-fine-tune):
    uv run python ablations/vlm_segmentation.py --model gemini-flash --held-out-only
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, NativeOutput
from pydantic_ai.usage import UsageLimits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.agent._model import resolve_model


# ── Output schema (enforced by pydantic-ai) ────────────────────────────────

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
        )
    )


# ── Default prompt (iterate via --prompt-file) ─────────────────────────────

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


# ── Pydantic-ai agent ──────────────────────────────────────────────────────

def build_agent(instructions: str) -> Agent:
    return Agent(
        "test",  # model is overridden per-call
        # NativeOutput → pydantic-ai uses Gemini's native response_format /
        # json_schema mode instead of the default tool-call mechanism.
        # Avoids the tool-call framing overhead that was causing complex
        # boundary outputs to hit the provider's default max_output_tokens
        # mid-response and fail with UnexpectedModelBehavior.
        output_type=NativeOutput(VlmSegmentation),
        retries=3,
        output_retries=2,
        model_settings={
            "temperature": 0,
            # Cap well above any plausible polygon output. 100-vertex
            # multi-polygon JSON serialises to ~3-4K tokens; 32K is
            # comfortable headroom and well within Gemini 3 Flash's
            # output budget.
            "max_tokens": 32768,
        },
        instructions=instructions,
    )


# ── Rasterization + IoU ────────────────────────────────────────────────────

def rasterize_polygons(polys: List[Polygon], width: int, height: int) -> np.ndarray:
    """Render (y, x) [0, 1000] polygons to a binary HxW mask (uint8 0/1).

    Vertices use Gemini's native convention: each pair is (y, x) integers
    in [0, 1000], applied to a 1000x1000 normalised view. We scale x by
    W/1000 and y by H/1000 to map onto the actual image dimensions
    (anisotropic for non-square pages). PIL.draw.polygon expects (x, y),
    so we flip the y-first model output to x-first pixel coords here."""
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    sx = width / 1000.0
    sy = height / 1000.0
    for poly in polys:
        if len(poly.vertices) < 3:
            continue
        try:
            pix = [(float(v[1]) * sx, float(v[0]) * sy)
                   for v in poly.vertices]
            draw.polygon(pix, fill=255)
        except Exception:
            continue
    return (np.asarray(canvas) > 127).astype(np.uint8)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Binary pixel IoU. pred, gt are HxW uint8 (0/1)."""
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = int((p & g).sum())
    union = int((p | g).sum())
    return float(inter / union) if union > 0 else 0.0


# ── Aggregation (mirrors scripts/eval_sam_kfold_v2.py:summarise) ───────────

def summarise(name: str, xs: List[float]) -> dict:
    n = len(xs)
    if n == 0:
        return {"name": name, "n": 0}
    s = sorted(xs)
    return {
        "name": name,
        "n": n,
        "mean": sum(xs) / n,
        "median": s[n // 2],
        "ge_0.50": sum(1 for x in xs if x >= 0.50) / n,
        "ge_0.70": sum(1 for x in xs if x >= 0.70) / n,
        "ge_0.80": sum(1 for x in xs if x >= 0.80) / n,
        "ge_0.90": sum(1 for x in xs if x >= 0.90) / n,
    }


def print_summary(s: dict) -> None:
    print(f"\n{s['name']} (N={s['n']})")
    if s['n'] == 0:
        print("  (no cases)")
        return
    print(f"  mean   = {s['mean']:.4f}")
    print(f"  median = {s['median']:.4f}")
    print(f"  >=0.50 = {s['ge_0.50']*100:.1f}%")
    print(f"  >=0.70 = {s['ge_0.70']*100:.1f}%")
    print(f"  >=0.80 = {s['ge_0.80']*100:.1f}%   <-- vs MHCLG 90%")
    print(f"  >=0.90 = {s['ge_0.90']*100:.1f}%")


# ── Main eval loop ─────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-flash",
                    help="OpenRouter alias (gemini-flash, gemini-pro, …) or full ID")
    ap.add_argument("--max-cases", type=int, default=None,
                    help="Cap on number of cases (for quick iteration)")
    ap.add_argument("--held-out-only", action="store_true",
                    help="Only evaluate cases that were in the held-out fold for "
                         "their fine-tune run (no fold-by-fold loop here — VLM "
                         "is fold-agnostic, but this filters to cases SAM3 "
                         "didn't train on, giving a like-for-like comparison)")
    ap.add_argument("--fold", type=int, default=None,
                    help="Only evaluate cases assigned to this fold (per "
                         "manifest). Useful to A/B against eval_sam_kfold_v2's "
                         "per-fold output.")
    ap.add_argument("--out-dir", default="results/ablation_vlm_seg")
    ap.add_argument("--throttle-s", type=float, default=1.0,
                    help="Seconds to sleep between API calls (rate-limit safety)")
    ap.add_argument("--prompt-file", type=str, default=None,
                    help="Path to a text file with a custom prompt (override "
                         "the built-in default — useful for prompt A/B).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be sent to the model for the first "
                         "case; do not call the API.")
    args = ap.parse_args()

    # ── Load manifest + filter ──────────────────────────────────────────────
    dataset_dir = REPO / "training" / "dataset"
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    if args.fold is not None:
        manifest = [r for r in manifest if r.get("fold") == args.fold]
    if args.held_out_only:
        # No-op: VLM has no training, so every case is "held out". Flag is
        # accepted for API parity with future ablations that do care.
        pass
    if args.max_cases:
        manifest = manifest[:args.max_cases]

    print(f"manifest: {len(manifest)} cases  "
          f"(fold={args.fold}, held_out_only={args.held_out_only})")

    # ── Load prompt ─────────────────────────────────────────────────────────
    if args.prompt_file:
        instructions = Path(args.prompt_file).read_text()
        print(f"prompt: loaded from {args.prompt_file} ({len(instructions)} chars)")
    else:
        instructions = DEFAULT_PROMPT
        print(f"prompt: built-in default ({len(instructions)} chars)")

    # ── Output paths ────────────────────────────────────────────────────────
    out_dir = REPO / args.out_dir / args.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = out_dir / "pred_masks"
    preds_dir.mkdir(exist_ok=True)
    print(f"output: {out_dir}")

    # ── Dry-run preview ─────────────────────────────────────────────────────
    if args.dry_run:
        e = manifest[0]
        img_path = dataset_dir / "maps" / e["filename"]
        img = Image.open(img_path)
        print(f"\nDRY RUN: would call {args.model}")
        print(f"  case:   {e['case']}")
        print(f"  image:  {img_path}  ({img.width}x{img.height})")
        print(f"  prompt (first 600 chars):\n{instructions[:600]}")
        print(f"\nUser message: 'Locate the drawn site boundary and "
              f"output it per the schema.'")
        return

    # ── Build agent ─────────────────────────────────────────────────────────
    agent = build_agent(instructions)
    model = resolve_model(args.model)

    rows = []
    t0 = time.time()

    for i, entry in enumerate(manifest):
        case = entry["case"]
        fname = entry["filename"]
        fold = entry.get("fold")
        img_path = dataset_dir / "maps" / fname
        mask_path = dataset_dir / "boundary_masks" / fname

        if not img_path.exists() or not mask_path.exists():
            print(f"  [{i+1:>3}/{len(manifest)}] SKIP {case}: missing files")
            rows.append({"case": case, "fold": fold, "filename": fname,
                         "iou": None, "n_polygons": 0,
                         "error": "missing files", "notes": ""})
            continue

        img = Image.open(img_path).convert("RGB")
        png_bytes = img_path.read_bytes()
        gt = np.asarray(Image.open(mask_path).convert("L"))
        gt_bin = (gt > 127).astype(np.uint8)

        if gt_bin.shape != (img.height, img.width):
            # GT and map should be at identical resolution per the fine-tune
            # training contract; flag loudly if they aren't.
            print(f"  [{i+1:>3}] WARN {case}: gt shape {gt_bin.shape} "
                  f"!= image (h,w)=({img.height},{img.width})")

        # ── Call the VLM ────────────────────────────────────────────────────
        t_call = time.time()
        try:
            result = agent.run_sync(
                [
                    BinaryContent(data=png_bytes, media_type="image/png"),
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
                1 for poly in polys for v in poly.vertices
                if not (0 <= v[0] <= 1000 and 0 <= v[1] <= 1000)
            )
            if n_oor > 0:
                example = next(
                    (v for poly in polys for v in poly.vertices
                     if not (0 <= v[0] <= 1000 and 0 <= v[1] <= 1000)),
                    None,
                )
                print(f"  [{i+1:>3}] WARN {case}: {n_oor} vertices out of "
                      f"[0, 1000] (e.g. {example}). Model may have ignored "
                      f"the (y, x) ∈ [0, 1000] convention.")
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            print(f"  [{i+1:>3}/{len(manifest)}] FAIL {case[:30]}  "
                  f"{type(e).__name__}: {str(e)[:80]}")
            rows.append({"case": case, "fold": fold, "filename": fname,
                         "iou": None, "n_polygons": 0,
                         "error": f"{type(e).__name__}: {str(e)[:200]}",
                         "notes": ""})
            time.sleep(args.throttle_s)
            continue

        # ── Rasterize + IoU ─────────────────────────────────────────────────
        pred = rasterize_polygons(polys, img.width, img.height)
        score = iou_score(pred, gt_bin)

        # Save pred mask for inspection
        Image.fromarray((pred * 255).astype(np.uint8)).save(
            preds_dir / f"{fname}")

        rows.append({"case": case, "fold": fold, "filename": fname,
                     "iou": score, "n_polygons": len(polys),
                     "error": None, "notes": notes[:160],
                     "call_seconds": round(dt_call, 2)})

        mark = "PASS" if score >= 0.8 else "OK  " if score >= 0.5 else "WEAK"
        print(f"  [{i+1:>3}/{len(manifest)}] {mark} {case[:30]:30s}  "
              f"IoU={score:.4f}  polys={len(polys)}  "
              f"({dt_call:.1f}s)")

        time.sleep(args.throttle_s)

    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed:.0f}s")

    # ── Aggregate ───────────────────────────────────────────────────────────
    valid = [r["iou"] for r in rows if r["iou"] is not None]
    fails = sum(1 for r in rows if r["iou"] is None)

    print("\n" + "=" * 60)
    print(f"AGGREGATE — model={args.model}")
    if args.prompt_file:
        print(f"prompt-file: {args.prompt_file}")
    print(f"failures: {fails}/{len(rows)}")
    print("=" * 60)
    summary_all = summarise("VLM-direct pixel IoU (all)", valid)
    print_summary(summary_all)

    # Per-fold breakdown (matches eval_sam_kfold_v2 fold-by-fold output)
    if any(r.get("fold") is not None for r in rows):
        print("\nPer-fold breakdown:")
        for f_id in sorted({r["fold"] for r in rows if r.get("fold") is not None}):
            fxs = [r["iou"] for r in rows if r.get("fold") == f_id and r["iou"] is not None]
            if fxs:
                s = summarise(f"fold {f_id}", fxs)
                print_summary(s)

    # ── Save CSV + JSON ─────────────────────────────────────────────────────
    csv_path = out_dir / "results.csv"
    with csv_path.open("w") as f:
        f.write("case,fold,filename,iou,n_polygons,call_seconds,error,notes\n")
        for r in rows:
            f.write(
                f"{r['case']},{r.get('fold')},{r['filename']},"
                f"{r.get('iou','')},{r.get('n_polygons','')},"
                f"{r.get('call_seconds','')},"
                f"\"{(r.get('error') or '').replace(chr(34),chr(39))[:100]}\","
                f"\"{(r.get('notes') or '').replace(chr(34),chr(39))[:120]}\"\n"
            )
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps({
        "model": args.model,
        "prompt_file": args.prompt_file,
        "n_cases": len(rows),
        "n_failures": fails,
        "elapsed_seconds": round(elapsed, 1),
        "summary": summary_all,
    }, indent=2))
    print(f"\nWrote:\n  {csv_path}\n  {json_path}")
    print(f"Pred masks in: {preds_dir}")


if __name__ == "__main__":
    main()
