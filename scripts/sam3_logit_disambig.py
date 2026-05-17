"""SAM3 logit-statistics disambiguation test on 22 multi-page cases.

For each candidate map_page returned by the reader, run SAM3 (k-fold
adapter routed by case name) twice:
  1) query = "planning boundary"
  2) query = "legend"          (differential / negative control)

For each (page, query) collect:
  - presence_logit            scalar — does SAM3 think the queried thing
                                       exists in this image at all?
  - max(pred_logits)          scalar — best candidate-mask score
  - top5_mean(pred_logits)    scalar — robust to lucky max
  - sem_mean, sem_max, sem_p95 — raw semantic_seg logit stats
  - mask_area_pct, compactness — for back-compat with the prior table

Cross-reference each case against results/benchmark_v_postrot to mark
the page the worker actually committed to.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.extraction.sam3 import load_sam3_ft, set_fold_for_case
from tools.io.map_page import render_map_page


# The 22 cases identified earlier (in benchmark v_postrot folder-name form).
CASES_22 = [
    "A4D4A1",
    "2CCA14C4-32BC-443D-9148-22DFE30A3DAC",
    "1D1A9561-7534-409B-9937-29C887F66069",
    "Ar4.9",
    "Ar4.6",
    "12:00126:ART4",
    "A4D-04",
    "Ar4.30",
    "12:00121:ART4",
    "CPA4(2a)",
    "A108P",
    "A4D8A_merged",
    "A4aSHP1",
    "Ar4.4",
    "23:53149:ART4",
    "A4D-24",
    "Ar4.17",
    "Ar4.2",
    "A4D9",
    "85",
    "A4D6A_merged",
    "25",
]


BENCH = Path("results/benchmark_v_postrot/gemini-flash")
EVAL = Path("evaluation_data")


def _mine_used_page(case_dir: Path, reader_first: int) -> int:
    """Walk message_log.json for the last render_page; else reader's #1."""
    log = json.loads((case_dir / "message_log.json").read_text())
    used = reader_first
    for msg in log:
        if msg.get("kind") == "ToolCallPart" and msg.get("tool") == "render_page":
            args = msg.get("args", {})
            if isinstance(args, dict) and "page" in args:
                used = int(args["page"])
    return used


def _iou(case_dir: Path) -> float | None:
    m = json.loads((case_dir / "metrics.json").read_text())
    return m.get("iou") or m.get("iou_polygon")


def _find_pdf(case_name: str) -> Path | None:
    """Return the PDF path for a case (handles : ↔ _ form)."""
    safe = case_name.replace(":", "_").replace("/", "_")
    for variant in (case_name, safe):
        cdir = EVAL / variant
        if cdir.exists():
            for f in cdir.iterdir():
                if f.suffix.lower() == ".pdf":
                    return f
    return None


def _mask_geom(mask: np.ndarray) -> tuple[float, float, int]:
    """area_pct, compactness, n_components for a 0/255 mask."""
    h, w = mask.shape[:2]
    binary = (mask > 0).astype(np.uint8)
    area_pct = float(binary.sum()) / float(h * w) * 100.0
    import cv2 as _cv2
    n, _, _, _ = _cv2.connectedComponentsWithStats(binary, connectivity=8)
    n_components = max(0, n - 1)
    contours, _ = _cv2.findContours(binary, _cv2.RETR_EXTERNAL,
                                       _cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return area_pct, 0.0, n_components
    areas = [_cv2.contourArea(c) for c in contours]
    perims = [_cv2.arcLength(c, True) for c in contours]
    total_a = sum(areas)
    total_p = sum(perims)
    compactness = (4.0 * np.pi * total_a / (total_p * total_p)) if total_p > 0 else 0.0
    return area_pct, float(compactness), n_components


def _run_sam3_with_logits(map_img, processor, model, device, query):
    """Run SAM3 and return (mask 0/255 uint8, signal_dict).

    Mirrors extract_boundary_sam3_semantic but also exposes raw
    presence_logit / pred_logits / semantic_seg statistics.
    """
    from PIL import Image
    import cv2 as _cv2
    # map_img is BGR np array from render_map_page
    pil = Image.fromarray(_cv2.cvtColor(map_img, _cv2.COLOR_BGR2RGB))
    w, h = pil.size
    inputs = processor(images=pil, text=query, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)

    presence = float(out.presence_logits.flatten()[0].cpu())
    pred_logits = out.pred_logits.flatten().cpu().numpy()          # (200,)
    pred_max = float(pred_logits.max())
    pred_top5 = float(np.sort(pred_logits)[-5:].mean())
    sem = out.semantic_seg.flatten().cpu().numpy()                  # (288*288,)
    sem_mean = float(sem.mean())
    sem_max = float(sem.max())
    sem_p95 = float(np.percentile(sem, 95))

    masks = processor.post_process_semantic_segmentation(out, target_sizes=[(h, w)])
    if len(masks) == 0:
        mask = np.zeros((h, w), dtype=np.uint8)
    else:
        mask = (masks[0].cpu().numpy() > 0).astype(np.uint8) * 255

    return mask, {
        "presence": presence,
        "pred_max": pred_max,
        "pred_top5": pred_top5,
        "sem_mean": sem_mean,
        "sem_max": sem_max,
        "sem_p95": sem_p95,
    }


def main():
    models_state = {"sam3_ft": load_sam3_ft()}
    sam = models_state["sam3_ft"]
    processor = sam["processor"]
    model = sam["model"]
    device = sam["device"]

    rows = []
    for case in CASES_22:
        folder = BENCH / case
        if not folder.exists():
            print(f"  skipped {case}: no benchmark dir")
            continue
        pdf_info = json.loads((folder / "pdf_info.json").read_text())
        map_pages = pdf_info.get("map_pages") or []
        details = {d["page"]: d for d in (pdf_info.get("map_page_details") or [])}
        reader_first = map_pages[0] if map_pages else None
        used = _mine_used_page(folder, reader_first)
        iou = _iou(folder)

        pdf = _find_pdf(case)
        if pdf is None:
            print(f"  skipped {case}: no PDF")
            continue

        set_fold_for_case(sam, case)

        print(f"\n=== {case}  reader={map_pages}  used={used}  iou={iou} ===")
        for rank, p in enumerate(map_pages, 1):
            rendered = render_map_page(str(pdf), p, dpi=200, verbose=False,
                                          case_name=case)
            if rendered is None:
                print(f"   p{p}: render failed")
                continue
            map_img, _rot = rendered

            mask_pb, sig_pb = _run_sam3_with_logits(
                map_img, processor, model, device, "planning boundary")
            mask_lg, sig_lg = _run_sam3_with_logits(
                map_img, processor, model, device, "legend")

            area, comp, ncomp = _mask_geom(mask_pb)
            area_lg, _, _ = _mask_geom(mask_lg)

            rows.append({
                "case": case, "page": p, "rank": rank,
                "used": p == used,
                "iou": iou if p == used else None,
                "role": (details.get(p) or {}).get("role", ""),
                **{f"pb_{k}": v for k, v in sig_pb.items()},
                **{f"lg_{k}": v for k, v in sig_lg.items()},
                "area_pct": area, "compactness": comp, "n_comp": ncomp,
                "lg_area_pct": area_lg,
            })

            mark = " USED 🏆" if p == used else ""
            print(f"   p{p:>3d} ({rank}, {(details.get(p) or {}).get('role',''):8s}) "
                  f"pb_presence={sig_pb['presence']:+6.2f}  "
                  f"pb_max={sig_pb['pred_max']:+6.2f}  "
                  f"pb_sem_max={sig_pb['sem_max']:+6.2f}  "
                  f"pb_area={area:5.2f}%  "
                  f"lg_presence={sig_lg['presence']:+6.2f}  "
                  f"lg_area={area_lg:5.2f}%"
                  f"{mark}")

    out = Path("analysis/multi_map_pages/sam3_logit_disambig.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
