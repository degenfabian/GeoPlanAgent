"""SAM3 base (no LoRA) text-prompt search on the UK planning-boundary task.

Runs each candidate prompt across all 211 annotated map pages and reports
per-prompt aggregate IoU plus a cross-prompt comparison (the paper's Table 12;
the winner is Figure 3's vanilla-SAM3 baseline).

Local SAM3 inference on MPS/CUDA/CPU. No API cost.
"""

import argparse
import contextlib
import csv
import io
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from geoplanagent.paths import ABL_SAM_BASE, TRAINING_DATASET_DIR  # noqa: E402
from geoplanagent.utils import device as get_device  # noqa: E402

# Production SAM3 semantic-segmentation call — same code path the LoRA
# pipeline uses. We just pass a base model loaded without an adapter.
from geoplanagent.tools.segment import extract_boundary_sam3_semantic  # noqa: E402

from ablations.utils import iou_score, print_summary, summarise  # noqa: E402


# Pre-registered candidate prompts (5). See module docstring.
DEFAULT_PROMPTS: List[str] = [
    "planning boundary",  # LoRA-trained anchor
    "article 4 site boundary",  # UK-specific phrasing
    "highlighted marked area",
    "site boundary",
    "application site",
]

load_dotenv()

def _prompt_dirname(prompt: str) -> str:
    """Filesystem-safe directory name for a prompt (spaces/slashes -> '_')."""
    return prompt.replace(" ", "_").replace("/", "_")


def _load_sam3_base() -> Tuple[object, object, object]:
    """Load SAM3 base from HuggingFace (no LoRA adapter)."""
    from transformers import Sam3Model, Sam3Processor

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not set. Cannot download SAM3 base model.")

    device = get_device()
    processor = Sam3Processor.from_pretrained("facebook/sam3", token=hf_token)
    model = Sam3Model.from_pretrained("facebook/sam3", token=hf_token)
    model = model.to(device).eval()
    print(f"SAM3 (base, no LoRA) loaded on {device}")
    return processor, model, device


def _row(case, fold, fname, iou=None, secs="", error=""):
    """One per-case result row; failures pass iou=None plus an error string."""
    return {
        "case": case,
        "fold": fold,
        "filename": fname,
        "iou": iou,
        "call_seconds": secs,
        "error": error,
    }


def try_fill_boundary_outline(mask):
    """Fill a thin base-SAM3 boundary outline into a solid region.

    Base SAM3 traces the boundary *line*, not the area, so its raw mask scores
    poorly vs the filled GT — we flood-fill the interior. Lives in the ablation
    (not segment.py) because the production LoRA already outputs filled regions.
    """
    if mask is None:
        return None
    h, w = mask.shape[:2]
    total_pixels = h * w
    fill_ratio = np.sum(mask > 0) / total_pixels
    # skip non-outlines: <0.1% is ~empty, >40% is already a solid blob
    if fill_ratio > 0.4 or fill_ratio < 0.001:
        return mask

    # close small gaps so the outline is one unbroken wall (else the flood leaks)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Trick: flood the OUTSIDE from the corners; what it can't reach is the inside.
    flood = closed.copy()
    flood_h, flood_w = flood.shape[:2]
    # cv2.floodFill demands a scratch buffer 2 px bigger in each dimension
    ff_mask = np.zeros((flood_h + 2, flood_w + 2), dtype=np.uint8)
    border_seeds = []
    if flood[0, 0] == 0:
        border_seeds.append((0, 0))
    if flood[0, flood_w - 1] == 0:
        border_seeds.append((flood_w - 1, 0))
    if flood[flood_h - 1, 0] == 0:
        border_seeds.append((0, flood_h - 1))
    if flood[flood_h - 1, flood_w - 1] == 0:
        border_seeds.append((flood_w - 1, flood_h - 1))
    # paint everything the flood reaches with 128 (any marker that isn't 0 or 255)
    for seed in border_seeds:
        cv2.floodFill(flood, ff_mask, seed, 128)

    # pixels still 0 = what the flood never reached = the interior; fill it
    filled = (flood == 0).astype(np.uint8) * 255
    filled = np.maximum(filled, closed)
    # accept only if it became a real region and didn't leak over the whole page
    filled_after = np.sum(filled > 0) / total_pixels
    if filled_after > fill_ratio * 1.2 and filled_after < 0.85:
        return filled
    return mask


def run_prompt(
    prompt: str,
    annotated_pages: List[Dict],
    dataset_dir: Path,
    processor,
    model,
    device,
) -> List[Dict]:
    """Run one text prompt through SAM3-base over every case and score it.

    Only the IoU scores are kept (returned as rows); predicted masks are not
    written to disk — this ablation only needs the numbers.

    Args:
        prompt: the text query handed to SAM3 (e.g. "site boundary").
        annotated_pages: one dict per case to evaluate, each with "case" (case name),
            "filename" (the shared map/mask image filename), and "fold".
        dataset_dir: root holding maps/<filename> (the input map) and
            boundary_masks/<filename> (the ground-truth boundary mask).
        processor: the SAM3 processor (from _load_sam3_base).
        model: the SAM3 base model (no LoRA adapter).
        device: torch device to run inference on.

    Returns:
        One row dict per case: {case, fold, filename, iou, call_seconds, error}.
        ``iou`` is None on failure (missing files, no mask returned, or a
        pred/GT shape mismatch); ``error`` carries the reason.
    """
    rows: List[Dict] = []

    for i, entry in enumerate(annotated_pages):
        case = entry["case"]
        filename = entry["filename"]
        fold = entry.get("fold")
        img_path = dataset_dir / "maps" / filename
        mask_path = dataset_dir / "boundary_masks" / filename

        if not img_path.exists() or not mask_path.exists():
            print(f"  [{i + 1:>3}/{len(annotated_pages)}] SKIP {case[:30]:<30}  missing files")
            rows.append(_row(case, fold, filename, error="missing files"))
            continue

        gt = np.asarray(Image.open(mask_path).convert("L"))

        # Silence extract_boundary_sam3_semantic's own debug prints to keep
        # our per-case line tidy. (It prints "SAM3 semantic: mask X% …".)
        start_time = time.time()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mask = extract_boundary_sam3_semantic(
                    cv2.imread(str(img_path)),
                    processor,
                    model,
                    device,
                    query=prompt,
                )
        except Exception as error:
            print(
                f"  [{i + 1:>3}/{len(annotated_pages)}] FAIL  {case[:30]:<30}  "
                f"{type(error).__name__}: {str(error)[:60]}"
            )
            rows.append(_row(case, fold, filename, error=f"{type(error).__name__}: {str(error)[:200]}"))
            continue
        elapsed_seconds = time.time() - start_time

        if mask is None:
            print(f"  [{i + 1:>3}/{len(annotated_pages)}] FAIL  {case[:30]:<30}  no mask returned")
            rows.append(_row(case, fold, filename, secs=round(elapsed_seconds, 2), error="no mask returned"))
            continue

        mask = try_fill_boundary_outline(mask)
        iou = iou_score(mask, gt)
        if iou is None:
            print(
                f"  [{i + 1:>3}/{len(annotated_pages)}] WARN  {case[:30]:<30}  "
                f"shape pred {mask.shape} != gt {gt.shape}"
            )
            rows.append(_row(case, fold, filename, secs=round(elapsed_seconds, 2), error=f"shape mismatch pred {mask.shape}"))
            continue

        mark = "PASS" if iou >= 0.8 else "OK  " if iou >= 0.5 else "WEAK"
        print(f"  [{i + 1:>3}/{len(annotated_pages)}] {mark}  {case[:30]:<30}  IoU={iou:.4f}  ({elapsed_seconds:.1f}s)")
        rows.append(_row(case, fold, filename, iou=iou, secs=round(elapsed_seconds, 2)))
    return rows


def _write_per_prompt(prompt: str, rows: List[Dict], prompt_dir: Path) -> Tuple[dict, int]:
    """Write one prompt's per-case results.csv and return its aggregate.

    Args:
        prompt: the prompt these rows are for (used as the summary's name).
        rows: the per-case row dicts from run_prompt (case, fold, filename, iou,
            call_seconds, error); a row with iou=None is a failure.
        prompt_dir: directory to write results.csv into.

    Returns:
        (summary, n_failures) — the summarise() stats over the valid IoUs, and the
        count of failed (iou=None) cases.
    """
    prompt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = prompt_dir / "results.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["case", "fold", "filename", "iou", "call_seconds", "error"])
        for row in rows:
            writer.writerow(
                [
                    row.get("case"),
                    row.get("fold"),
                    row.get("filename"),
                    row.get("iou", ""),
                    row.get("call_seconds", ""),
                    (row.get("error") or "")[:200],
                ]
            )

    valid = [row["iou"] for row in rows if row.get("iou") is not None]
    fails = sum(1 for row in rows if row.get("iou") is None)
    return summarise(prompt, valid), fails


def main() -> int:
    # No flags — but parse argv so `--help` prints the docstring instead of
    # silently launching the ~25-minute sweep.
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()

    from ablations.utils import load_annotated_pages

    dataset_dir = TRAINING_DATASET_DIR
    annotated_pages = load_annotated_pages(REPO)
    print(f"annotated_pages: {len(annotated_pages)} cases")

    print(f"prompts ({len(DEFAULT_PROMPTS)}):")
    for prompt in DEFAULT_PROMPTS:
        print(f"  - {prompt!r}")

    out_dir = ABL_SAM_BASE
    out_dir.mkdir(parents=True, exist_ok=True)

    processor, model, device = _load_sam3_base()

    all_summaries: Dict[str, dict] = {}
    all_fails: Dict[str, int] = {}
    t_start = time.time()

    for prompt_idx, prompt in enumerate(DEFAULT_PROMPTS, 1):
        print(f"\n{'=' * 70}")
        print(f"[Prompt {prompt_idx}/{len(DEFAULT_PROMPTS)}]  {prompt!r}")
        print("=" * 70)
        prompt_dir = out_dir / _prompt_dirname(prompt)
        rows = run_prompt(prompt, annotated_pages, dataset_dir, processor, model, device)
        summary, fails = _write_per_prompt(prompt, rows, prompt_dir)
        print_summary(summary)
        print(f"failures: {fails}/{len(rows)}")
        all_summaries[prompt] = summary
        all_fails[prompt] = fails

    elapsed = time.time() - t_start

    # Cross-prompt comparison (sorted by mean IoU descending)
    print(f"\n{'=' * 70}")
    print(f"COMPARISON  (elapsed {elapsed / 60:.1f} min)")
    print("=" * 70)
    ranked = sorted(all_summaries.items(), key=lambda item: -(item[1].get("mean", -1)))
    for i, (prompt, summary) in enumerate(ranked):
        marker = "  ← winner" if i == 0 else ""
        if summary.get("n", 0) == 0:
            print(f"  {prompt!r:<40}  (no valid cases)")
            continue
        print(
            f"  {prompt!r:<40}  mean={summary['mean']:.4f}  "
            f"median={summary['median']:.4f}  "
            f">=0.5={summary['ge_0.50'] * 100:5.1f}%  "
            f">=0.8={summary['ge_0.80'] * 100:5.1f}%  "
            f"fails={all_fails[prompt]}{marker}"
        )

    print(f"\nWinner: {ranked[0][0]!r}" if ranked else "No prompts run")
    print(f"All outputs under: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
