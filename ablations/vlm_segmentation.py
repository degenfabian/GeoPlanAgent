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
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, NativeOutput
from pydantic_ai.usage import UsageLimits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from geoplanagent.agent._model import resolve_model


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
        )
    )


# Default prompt (iterate via --prompt-file)

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
        # Avoids the tool-call framing overhead that was causing complex
        # boundary outputs to hit the provider's default max_output_tokens
        # mid-response and fail with UnexpectedModelBehavior.
        output_type=NativeOutput(VlmSegmentation),
        retries=3,
        # output_retries=0 — validation failures get reported as iou=None
        # immediately rather than triggering up-to-2 paid retries. On
        # Gemini 3.1 Pro this drops per-failure cost from ~$0.30 (3 API
        # calls) to ~$0.10 (1 API call); the original Flash run showed
        # the retry-rescue rate is only ~3%, so the cost saving dominates.
        output_retries=0,
        model_settings={
            "temperature": temperature,
            # 100-vertex multi-polygon JSON serialises to ~3-4K tokens.
            # 8K cap is comfortable for any reasonable boundary on Flash.
            # NB: Gemini 3.1 Pro is a reasoning model and counts thinking
            # tokens against this cap; we briefly tried 32K to rescue 30
            # Pro schema-validation failures, but the "recovered" cases
            # all turned out to emit out-of-range pixel coords (>1000),
            # not the in-spec [0,1000] convention, and rasterized to
            # near-zero IoU. We reverted to 8K and report Pro on the
            # parseable subset (see paper §abl-vlm-direct-protocol).
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


# Aggregation (mirrors scripts/eval_sam_kfold_v2.py:summarise)

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


def _write_outputs(rows: List[dict], out_dir, args, elapsed: float,
                   note: str = "") -> None:
    """Idempotent rewrite of results.csv + summary.json from in-memory rows.

    Safe to call repeatedly mid-loop: rewrites both files in full each time
    so the on-disk state always matches `rows` exactly. Used both for the
    periodic checkpoint inside the main loop (so a Ctrl-C never loses
    work) and for the final end-of-run write.
    """
    valid = [r["iou"] for r in rows if r["iou"] is not None]
    fails = sum(1 for r in rows if r["iou"] is None)
    summary_all = summarise("VLM-direct pixel IoU (all)", valid)

    csv_path = out_dir / "results.csv"
    with csv_path.open("w") as f:
        f.write("case,fold,filename,iou,n_polygons,call_seconds,error,notes\n")
        for r in rows:
            f.write(
                f"{r['case']},{r.get('fold')},{r['filename']},"
                f"{r.get('iou','') if r.get('iou') is not None else ''},"
                f"{r.get('n_polygons','')},"
                f"{r.get('call_seconds','')},"
                f"\"{(r.get('error') or '').replace(chr(34),chr(39))[:100]}\","
                f"\"{(r.get('notes') or '').replace(chr(34),chr(39))[:120]}\"\n"
            )
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps({
        "model": args.model,
        "prompt_file": args.prompt_file,
        "temperature": getattr(args, "temperature", 0.0),
        "n_cases": len(rows),
        "n_failures": fails,
        "elapsed_seconds": round(elapsed, 1),
        "summary": summary_all,
        "checkpoint_note": note,
    }, indent=2))
    if note and note != "final":
        print(f"  [save] {note}: {len(rows)} rows, {fails} failures, "
              f"mean IoU {summary_all.get('mean', 0):.4f} → {csv_path.name}")
    else:
        print(f"\nWrote:\n  {csv_path}\n  {json_path}")


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


# Main eval loop

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
    ap.add_argument("--cases", nargs="+", default=None,
                    help="Only run these specific case identifiers (matches "
                         "manifest['case']). Useful for re-running failed cases.")
    ap.add_argument("--max-image-dim", type=int, default=None,
                    help="If set, resize input map so its longest side is at "
                         "most this many pixels (preserves aspect). Helps with "
                         "413 errors on very large maps. Recommended: 2048.")
    ap.add_argument("--jpeg-quality", type=int, default=None,
                    help="If set, re-encode the input image as JPEG at this "
                         "quality (1-100) before sending. Cuts request payload "
                         "without resizing — useful for cases that hit 413 "
                         "errors on large PNGs. Recommended: 95.")
    ap.add_argument("--throttle-s", type=float, default=1.0,
                    help="Seconds to sleep between API calls (rate-limit safety)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (default: 0.0). Google's Gemini "
                         "3 docs warn thinking models (e.g. gemini-pro) may "
                         "loop at temperature <1.0; pass --temperature 1.0 to "
                         "test that mitigation. Non-deterministic at >0.")
    ap.add_argument("--save-every", type=int, default=5,
                    help="Rewrite results.csv + summary.json every N cases (in "
                         "addition to the end-of-run write). Ensures Ctrl-C "
                         "never loses already-paid-for work. Default: 5.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip cases whose pred_mask already exists in "
                         "out_dir/pred_masks; compute IoU from disk and "
                         "include in results.csv. Use to continue an "
                         "aborted run without re-paying for completed cases.")
    ap.add_argument("--prompt-file", type=str, default=None,
                    help="Path to a text file with a custom prompt (override "
                         "the built-in default — useful for prompt A/B).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be sent to the model for the first "
                         "case; do not call the API.")
    args = ap.parse_args()

    # Load manifest + filter
    # Build the manifest in-memory from `maps/*.png` + `fold_assignment.json`
    # using the same helper the training scripts use. We deliberately do
    # NOT persist a `manifest.json` on disk — it would just duplicate
    # information that's already derivable from the case filenames and
    # the fold assignment, and could drift out of sync with them.
    dataset_dir = REPO / "training" / "dataset"
    fold_assignment_path = dataset_dir / "fold_assignment.json"
    if not fold_assignment_path.exists():
        sys.exit(f"fold_assignment.json not found: {fold_assignment_path}. "
                 f"Run training/build_sam3_training_set.py first.")
    from training.train_sam3_kfold import _build_manifest_from_disk
    fold_map = json.loads(fold_assignment_path.read_text())
    manifest = _build_manifest_from_disk(dataset_dir, fold_map)
    if not manifest:
        sys.exit(f"manifest is empty — no .png files found in "
                 f"{dataset_dir / 'maps'} matching fold_assignment.json")

    if args.cases:
        wanted = set(args.cases)
        manifest = [r for r in manifest if r.get("case") in wanted]
        missing = wanted - {r.get("case") for r in manifest}
        if missing:
            print(f"WARNING: {len(missing)} requested cases not in manifest: {sorted(missing)}")
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

    # Load prompt
    if args.prompt_file:
        instructions = Path(args.prompt_file).read_text()
        print(f"prompt: loaded from {args.prompt_file} ({len(instructions)} chars)")
    else:
        instructions = DEFAULT_PROMPT
        print(f"prompt: built-in default ({len(instructions)} chars)")

    # Output paths
    out_dir = REPO / args.out_dir / args.model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = out_dir / "pred_masks"
    preds_dir.mkdir(exist_ok=True)
    print(f"output: {out_dir}")

    # Dry-run preview
    if args.dry_run:
        e = manifest[0]
        img_path = dataset_dir / "maps" / e["filename"]
        img = Image.open(img_path)
        print(f"\nDRY RUN: would call {args.model}")
        print(f"  case:   {e['case']}")
        print(f"  image:  {img_path}  ({img.width}x{img.height})")
        print(f"  prompt (first 600 chars):\n{instructions[:600]}")
        print("\nUser message: 'Locate the drawn site boundary and "
              "output it per the schema.'")
        return

    # Build agent
    agent = build_agent(instructions, temperature=args.temperature)
    model = resolve_model(args.model)
    print(f"agent: model={args.model}  temperature={args.temperature}  "
          f"save_every={args.save_every}")

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

        # --resume: if a pred_mask already exists on disk, skip the API
        # call and score from the cached mask. Lets a partial run be
        # finished without re-paying for completed cases.
        if args.resume:
            cached_pred = preds_dir / fname
            if cached_pred.exists():
                try:
                    pred_arr = (np.asarray(Image.open(cached_pred).convert("L")) > 127).astype(np.uint8)
                    gt_arr = (np.asarray(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)
                    if pred_arr.shape == gt_arr.shape:
                        cached_iou = iou_score(pred_arr, gt_arr)
                        print(f"  [{i+1:>3}/{len(manifest)}] CACHED {case[:30]:<30}  IoU={cached_iou:.4f}")
                        rows.append({"case": case, "fold": fold, "filename": fname,
                                     "iou": cached_iou, "n_polygons": 1,
                                     "call_seconds": "", "error": "",
                                     "notes": "resumed from existing pred_mask"})
                        continue
                    else:
                        print(f"  [{i+1:>3}/{len(manifest)}] cached mask shape "
                              f"{pred_arr.shape} != gt {gt_arr.shape}; re-running")
                except Exception as e:
                    print(f"  [{i+1:>3}/{len(manifest)}] failed to read cached "
                          f"mask ({e!s:.60}); re-running")

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.width, img.height
        # Optionally downscale very large maps to avoid 413 errors at the
        # provider. Aspect is preserved. The polygon output is in
        # normalised [0,1000] coords, so we can rasterize at orig_w/orig_h
        # below regardless of the resize — keeping IoU at the same
        # resolution as the GT (and as the non-resized 196 cases).
        if args.max_image_dim is not None:
            longest = max(img.width, img.height)
            if longest > args.max_image_dim:
                scale = args.max_image_dim / longest
                new_w = int(img.width * scale)
                new_h = int(img.height * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)

        # Encode as JPEG (smaller payload, dodges 413 on large maps) or PNG
        # (lossless, default).
        if args.jpeg_quality is not None:
            import io as _io
            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=int(args.jpeg_quality))
            png_bytes = buf.getvalue()
            media_type = "image/jpeg"
        elif args.max_image_dim is not None and max(img.width, img.height) <= args.max_image_dim:
            # Resized → re-encode PNG from the resized image
            import io as _io
            buf = _io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            media_type = "image/png"
        else:
            png_bytes = img_path.read_bytes()
            media_type = "image/png"
        gt = np.asarray(Image.open(mask_path).convert("L"))
        gt_bin = (gt > 127).astype(np.uint8)

        if gt_bin.shape != (orig_h, orig_w):
            # GT and original map should be at identical resolution per
            # the fine-tune training contract; flag loudly if they aren't.
            print(f"  [{i+1:>3}] WARN {case}: gt shape {gt_bin.shape} "
                  f"!= original image (h,w)=({orig_h},{orig_w})")

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
            print(f"  [{i+1:>3}/{len(manifest)}] FAIL {case[:30]}  "
                  f"{type(e).__name__}: {str(e)[:80]}")
            rows.append({"case": case, "fold": fold, "filename": fname,
                         "iou": None, "n_polygons": 0,
                         "error": f"{type(e).__name__}: {str(e)[:200]}",
                         "notes": ""})
            time.sleep(args.throttle_s)
            continue

        # Rasterize + IoU
        # Rasterize at the ORIGINAL map resolution — polygons are in
        # normalised [0,1000] coords, so this is independent of any
        # --max-image-dim resize applied above. Keeps IoU comparable
        # across resized and non-resized cases.
        pred = rasterize_polygons(polys, orig_w, orig_h)
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

        # Periodic save so a Ctrl-C never loses already-paid-for work.
        if args.save_every > 0 and (i + 1) % args.save_every == 0:
            _write_outputs(rows, out_dir, args, time.time() - t0,
                           note=f"checkpoint after case {i + 1}/{len(manifest)}")

        time.sleep(args.throttle_s)

    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed:.0f}s")

    # Aggregate (final)
    valid = [r["iou"] for r in rows if r["iou"] is not None]
    fails = sum(1 for r in rows if r["iou"] is None)
    print("\n" + "=" * 60)
    print(f"AGGREGATE — model={args.model}  temperature={args.temperature}")
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

    _write_outputs(rows, out_dir, args, elapsed, note="final")
    print(f"Pred masks in: {preds_dir}")


if __name__ == "__main__":
    main()
