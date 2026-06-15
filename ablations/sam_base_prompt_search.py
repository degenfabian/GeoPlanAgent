"""SAM3 base (no LoRA) text-prompt search on the UK planning-boundary task.

Runs each candidate prompt across all 211 cases and reports per-prompt
aggregate IoU plus a cross-prompt comparison. The "winner" goes into
ablation B4 as the no-LoRA baseline.

Methodology: each prompt is a discrete, pre-registered hyperparameter (no
learned parameter), so a "subset search + held-out eval" split adds no
statistical protection — we report all five prompts on the full 211 cases,
following Perez et al. (EMNLP-Findings 2022) on full prompt-distribution
reporting. Prompt sensitivity is a documented open problem for promptable
segmentation (Liu et al., 2025; SAM3 paper §prompt-consistency).

Local SAM3 inference on MPS/CUDA/CPU. No API cost.

Usage:
    uv run python ablations/sam_base_prompt_search.py
    uv run python ablations/sam_base_prompt_search.py --resume
    uv run python ablations/sam_base_prompt_search.py --max-cases 10  # smoke
    uv run python ablations/sam_base_prompt_search.py \\
        --prompts "planning boundary" "site boundary"
"""

import argparse
import contextlib
import csv
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Production SAM3 semantic-segmentation call — same code path the LoRA
# pipeline uses. We just pass a base model loaded without an adapter.
from geoplanagent.tools.segment import extract_boundary_sam3_semantic  # noqa: E402

# IoU + per-prompt summary helpers from the sibling VLM ablation.
from ablations.vlm_segmentation import (  # noqa: E402
    iou_score,
    print_summary,
    summarise,
)


# Pre-registered candidate prompts (5). See module docstring.
DEFAULT_PROMPTS: List[str] = [
    "planning boundary",  # LoRA-trained anchor
    "article 4 site boundary",  # UK-specific phrasing
    "highlighted marked area",
    "site boundary",
    "application site",
]


def _slug(p: str) -> str:
    """Filesystem-safe prompt key for per-prompt output directories."""
    return p.replace(" ", "_").replace("/", "_")


def _load_sam3_base() -> Tuple[object, object, object]:
    """Load SAM3 base from HuggingFace (no LoRA adapter)."""
    import torch
    from transformers import Sam3Model, Sam3Processor

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("  WARNING: HF_TOKEN not set; download may fail if model is gated.")

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)
    model = Sam3Model.from_pretrained("facebook/sam3", token=hf_token)
    model = model.to(device).eval()
    print(f"SAM3 (base, no LoRA) loaded on {device}")
    return processor, model, device


def _iou_from_mask(pred_mask: np.ndarray, gt_bin: np.ndarray) -> float | None:
    """Binarise pred_mask (0/255 uint8) and IoU against gt_bin (0/1 uint8)."""
    pred_bin = (pred_mask > 127).astype(np.uint8)
    if pred_bin.shape != gt_bin.shape:
        return None
    return iou_score(pred_bin, gt_bin)


def run_prompt(
    prompt: str,
    manifest: List[Dict],
    dataset_dir: Path,
    prompt_dir: Path,
    processor,
    model,
    device,
    resume: bool,
) -> List[Dict]:
    """Run one prompt across all cases. Returns per-case rows."""
    preds_dir = prompt_dir / "pred_masks"
    preds_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []

    for i, entry in enumerate(manifest):
        case = entry["case"]
        fname = entry["filename"]
        fold = entry.get("fold")
        img_path = dataset_dir / "maps" / fname
        mask_path = dataset_dir / "boundary_masks" / fname

        if not img_path.exists() or not mask_path.exists():
            print(f"  [{i + 1:>3}/{len(manifest)}] SKIP {case[:30]:<30}  missing files")
            rows.append(
                {
                    "case": case,
                    "fold": fold,
                    "filename": fname,
                    "iou": None,
                    "call_seconds": "",
                    "error": "missing files",
                }
            )
            continue

        gt_bin = (np.asarray(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)

        cached_path = preds_dir / fname
        if resume and cached_path.exists():
            try:
                cached = np.asarray(Image.open(cached_path).convert("L"))
                iou = _iou_from_mask(cached, gt_bin)
                if iou is not None:
                    print(f"  [{i + 1:>3}/{len(manifest)}] CACHED {case[:30]:<30}  IoU={iou:.4f}")
                    rows.append(
                        {
                            "case": case,
                            "fold": fold,
                            "filename": fname,
                            "iou": iou,
                            "call_seconds": "",
                            "error": "",
                        }
                    )
                    continue
            except Exception:
                pass  # fall through to fresh inference

        # Silence extract_boundary_sam3_semantic's own debug prints to keep
        # our per-case line tidy. (It prints "SAM3 semantic: mask X% …".)
        t0 = time.time()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mask = extract_boundary_sam3_semantic(
                    cv2.imread(str(img_path)),
                    processor,
                    model,
                    device,
                    query=prompt,
                )
        except Exception as e:
            print(
                f"  [{i + 1:>3}/{len(manifest)}] FAIL  {case[:30]:<30}  "
                f"{type(e).__name__}: {str(e)[:60]}"
            )
            rows.append(
                {
                    "case": case,
                    "fold": fold,
                    "filename": fname,
                    "iou": None,
                    "call_seconds": "",
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                }
            )
            continue
        dt = time.time() - t0

        if mask is None:
            print(f"  [{i + 1:>3}/{len(manifest)}] FAIL  {case[:30]:<30}  no mask returned")
            rows.append(
                {
                    "case": case,
                    "fold": fold,
                    "filename": fname,
                    "iou": None,
                    "call_seconds": round(dt, 2),
                    "error": "no mask returned",
                }
            )
            continue

        iou = _iou_from_mask(mask, gt_bin)
        if iou is None:
            pred_shape = (mask > 127).astype(np.uint8).shape
            print(
                f"  [{i + 1:>3}/{len(manifest)}] WARN  {case[:30]:<30}  "
                f"shape pred {pred_shape} != gt {gt_bin.shape}"
            )
            rows.append(
                {
                    "case": case,
                    "fold": fold,
                    "filename": fname,
                    "iou": None,
                    "call_seconds": round(dt, 2),
                    "error": f"shape mismatch pred {pred_shape}",
                }
            )
            continue

        Image.fromarray(mask).save(cached_path)
        mark = "PASS" if iou >= 0.8 else "OK  " if iou >= 0.5 else "WEAK"
        print(f"  [{i + 1:>3}/{len(manifest)}] {mark}  {case[:30]:<30}  IoU={iou:.4f}  ({dt:.1f}s)")
        rows.append(
            {
                "case": case,
                "fold": fold,
                "filename": fname,
                "iou": iou,
                "call_seconds": round(dt, 2),
                "error": "",
            }
        )
    return rows


def _write_per_prompt(prompt: str, rows: List[Dict], prompt_dir: Path) -> Tuple[dict, int]:
    csv_path = prompt_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["case", "fold", "filename", "iou", "call_seconds", "error"])
        for r in rows:
            w.writerow(
                [
                    r.get("case"),
                    r.get("fold"),
                    r.get("filename"),
                    r.get("iou", ""),
                    r.get("call_seconds", ""),
                    (r.get("error") or "")[:200],
                ]
            )

    valid = [r["iou"] for r in rows if r.get("iou") is not None]
    fails = sum(1 for r in rows if r.get("iou") is None)
    s = summarise(prompt, valid)
    (prompt_dir / "summary.json").write_text(
        json.dumps(
            {
                "prompt": prompt,
                "n_cases": len(rows),
                "n_failures": fails,
                "summary": s,
            },
            indent=2,
        )
    )
    return s, fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--prompts", nargs="+", default=DEFAULT_PROMPTS, help="Override the candidate prompt list"
    )
    ap.add_argument(
        "--out-dir",
        default="results/ablation_sam_base",
        help="Per-prompt outputs land in <out>/<prompt_slug>/",
    )
    ap.add_argument("--max-cases", type=int, default=None, help="Cap on cases (smoke testing)")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases whose pred_mask already exists; compute IoU from disk",
    )
    args = ap.parse_args()

    from ablations._shared import load_annotation_manifest

    dataset_dir = REPO / "training" / "dataset"
    manifest = load_annotation_manifest(REPO)
    if args.max_cases is not None:
        manifest = manifest[: args.max_cases]
    print(f"manifest: {len(manifest)} cases")

    print(f"prompts ({len(args.prompts)}):")
    for p in args.prompts:
        print(f"  - {p!r}")

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    processor, model, device = _load_sam3_base()

    all_summaries: Dict[str, dict] = {}
    all_fails: Dict[str, int] = {}
    t_start = time.time()

    for pi, prompt in enumerate(args.prompts, 1):
        print(f"\n{'=' * 70}")
        print(f"[Prompt {pi}/{len(args.prompts)}]  {prompt!r}")
        print("=" * 70)
        prompt_dir = out_dir / _slug(prompt)
        rows = run_prompt(
            prompt, manifest, dataset_dir, prompt_dir, processor, model, device, resume=args.resume
        )
        s, fails = _write_per_prompt(prompt, rows, prompt_dir)
        print_summary(s)
        print(f"failures: {fails}/{len(rows)}")
        all_summaries[prompt] = s
        all_fails[prompt] = fails

    elapsed = time.time() - t_start

    # Cross-prompt comparison (sorted by mean IoU descending)
    print(f"\n{'=' * 70}")
    print(f"COMPARISON  (elapsed {elapsed / 60:.1f} min)")
    print("=" * 70)
    ranked = sorted(all_summaries.items(), key=lambda kv: -(kv[1].get("mean", -1)))
    for i, (p, s) in enumerate(ranked):
        marker = "  ← winner" if i == 0 else ""
        if s.get("n", 0) == 0:
            print(f"  {p!r:<40}  (no valid cases)")
            continue
        print(
            f"  {p!r:<40}  mean={s['mean']:.4f}  "
            f"median={s['median']:.4f}  "
            f">=0.5={s['ge_0.50'] * 100:5.1f}%  "
            f">=0.8={s['ge_0.80'] * 100:5.1f}%  "
            f"fails={all_fails[p]}{marker}"
        )

    (out_dir / "_compare.json").write_text(
        json.dumps(
            {
                "prompts_ranked": [
                    {"prompt": p, "summary": s, "n_failures": all_fails[p]} for p, s in ranked
                ],
                "winner": ranked[0][0] if ranked else None,
                "n_cases": len(manifest),
                "elapsed_seconds": round(elapsed, 1),
            },
            indent=2,
        )
    )

    print(f"\nWinner: {ranked[0][0]!r}" if ranked else "No prompts run")
    print(f"All outputs under: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
