"""Generate §5 figures from cached data.

Outputs:
  figures/abl_cdfs.{pdf,png}       — Figure A (2-panel CDF)
  figures/iou_histogram.{pdf,png}  — Figure D (bimodal IoU histogram)

No API calls — reads only cached JSON / CSV files.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ── Load per-case data ────────────────────────────────────────────────────


def _read_iou_csv(path: Path) -> list[float]:
    rows = list(csv.DictReader(open(path)))
    return [float(r["iou"]) for r in rows if r.get("iou") not in (None, "")]


def _read_locate_errs(path: Path) -> list[float]:
    """Return centroid errors in metres from a locate-only locate_picks.csv."""
    rows = list(csv.DictReader(open(path)))
    return [float(r["err_km"]) * 1000.0
            for r in rows if r.get("err_km") not in (None, "")]


# Boundary segmentation, per-case pixel IoU (n=211 map masks)
vlm_flash_seg = _read_iou_csv(REPO / "results/ablation_vlm_seg/gemini-flash/results.csv")
vlm_pro_seg = _read_iou_csv(REPO / "results/ablation_vlm_seg/gemini-pro/results.csv")
vanilla_sam = _read_iou_csv(
    REPO / "results/ablation_sam_base/highlighted_marked_area/results.csv")
sam_lora_kfold = json.loads(
    (REPO / "training/eval/predictions/sam_kfold.json").read_text())
sam_lora = [v["sem_iou"] for v in sam_lora_kfold.values()]

# Locate-stage centroid error (n=208 documents)
locate_production = _read_locate_errs(
    REPO / "ablations/locate_only_eval/min_1_tool/locate_picks.csv")
locate_full_kit = _read_locate_errs(
    REPO / "ablations/locate_only_eval/full/locate_picks.csv")
locate_vlm = _read_locate_errs(
    REPO / "ablations/locate_only_eval/vlm_direct_gemini-flash/locate_picks.csv")

# Full-pipeline centroid error + GeoJSON IoU (n=208)
benchmark = json.loads(
    (REPO / "results/benchmark_v_post_refactor/gemini-flash/summary.json").read_text())
pipeline_errs = [c["positioning_error_m"] for c in benchmark["per_case"]
                 if c.get("positioning_error_m") is not None]
pipeline_ious = [c.get("iou", 0.0) for c in benchmark["per_case"]]


# ── Plot helpers ──────────────────────────────────────────────────────────


def _cdf(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Empirical CDF (x, y): at each sorted x, fraction of values $\\le$ x.
    Use when lower x = better (e.g. centroid error)."""
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def _survival(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Empirical survival (x, y): at each sorted x, fraction of values $\\ge$ x.
    Use when higher x = better (e.g. IoU)."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    y = (n - np.arange(n)) / n  # [1, (n-1)/n, ..., 1/n]
    return x, y


# Colour palette — colour-blind safe; SAM3-LoRA and Full pipeline get the
# eye-catching dark accents, alternatives use muted hues.
C_VLM_FLASH = "#e07b91"
C_VLM_PRO = "#c0142a"
C_VANILLA = "#7eb6d9"
C_SAMLORA = "#1f4e79"
C_LOCATE_PROD = "#7eb6d9"
C_LOCATE_KIT = "#3a7fb5"
C_LOCATE_VLM = "#c0142a"
C_PIPELINE = "#1a7a3e"


# ── Figure A: two-panel CDF ────────────────────────────────────────────────

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 7})

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.4))

# Panel (a): pixel-IoU survival curve — y(x) = fraction of cases scoring
# at IoU >= x. Higher curve = more cases pass that quality bar.
for label, vals, color in [
    ("VLM-direct (Gemini-3-Flash)", vlm_flash_seg, C_VLM_FLASH),
    ("VLM-direct (Gemini-3.1-Pro)", vlm_pro_seg, C_VLM_PRO),
    ("Vanilla SAM-3 (best prompt)", vanilla_sam, C_VANILLA),
    ("SAM3-LoRA (ours)", sam_lora, C_SAMLORA),
]:
    x, y = _survival(vals)
    ax_a.plot(x, y, label=label, color=color, linewidth=1.6)
ax_a.set_xlim(0, 1)
ax_a.set_ylim(0, 1.02)
ax_a.set_xlabel("Pixel IoU threshold $t$")
ax_a.set_ylabel("Fraction of cases with IoU $\\geq t$")
ax_a.set_title("(a) Boundary segmentation", fontsize=10)
ax_a.grid(True, alpha=0.3)
ax_a.legend(loc="lower left", frameon=False)

# Panel (b): centroid-error success rate — y(x) = fraction of cases whose
# centroid lands within x metres of ground truth. Higher curve = better.
for label, vals, color in [
    ("VLM-direct (Gemini-3-Flash)", locate_vlm, C_LOCATE_VLM),
    ("Locate: place only (prod.)", locate_production, C_LOCATE_PROD),
    ("Locate: all 6 tools", locate_full_kit, C_LOCATE_KIT),
    ("Full pipeline (+ match_at)", pipeline_errs, C_PIPELINE),
]:
    x, y = _cdf(vals)
    # Replace zeros (impossible on the log axis) with a tiny floor so the
    # CDF doesn't snap to log(0) = -inf.
    x = np.clip(x, 0.1, None)
    ax_b.plot(x, y, label=label, color=color, linewidth=1.6)
ax_b.set_xscale("log")
ax_b.set_xlim(1, 1e5)
ax_b.set_ylim(0, 1.02)
ax_b.set_xlabel("Centroid error threshold $t$ (m, log scale)")
ax_b.set_ylabel("Fraction of cases within $t$ metres")
ax_b.set_title("(b) Localization", fontsize=10)
ax_b.grid(True, alpha=0.3, which="both")
ax_b.legend(loc="upper left", frameon=False)

plt.tight_layout()
plt.savefig(FIG_DIR / "abl_cdfs.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "abl_cdfs.png", bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"Wrote {FIG_DIR / 'abl_cdfs.pdf'} (and .png)")


# ── Figure D: IoU histogram (bimodal distribution) ─────────────────────────

fig, ax = plt.subplots(figsize=(3.4, 2.2))
bins = np.linspace(0, 1, 21)  # 20 equal bins of width 0.05
ax.hist(pipeline_ious, bins=bins, color=C_SAMLORA,
        edgecolor="white", linewidth=0.5)
ax.set_xlim(0, 1)
ax.set_xlabel("Per-case GeoJSON IoU")
ax.set_ylabel("Number of cases")
ax.grid(True, alpha=0.3, axis="y")

# Annotate the three regions of the bimodal distribution.
n_ge09 = sum(1 for x in pipeline_ious if x >= 0.9)
n_lt005 = sum(1 for x in pipeline_ious if x < 0.05)
n_mid = sum(1 for x in pipeline_ious if 0.3 <= x <= 0.7)
total = len(pipeline_ious)

# Pad the y-axis so annotations sit above the bars rather than clipping.
ymax = max(ax.get_ylim()[1], 70)
ax.set_ylim(0, ymax)

ax.annotate(
    f"{n_ge09/total*100:.0f}% $\\geq$ 0.9",
    xy=(0.93, 52), xytext=(0.50, 64),
    fontsize=8, ha="left",
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.6))
ax.annotate(
    f"{n_lt005/total*100:.0f}% $<$ 0.05",
    xy=(0.04, n_lt005 + 1), xytext=(0.13, 50),
    fontsize=8, ha="left",
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.6))
ax.annotate(
    f"only {n_mid/total*100:.0f}% in [0.3, 0.7]",
    xy=(0.5, 5), xytext=(0.30, 30),
    fontsize=8, ha="left",
    arrowprops=dict(arrowstyle="->", color="gray", lw=0.6))

plt.tight_layout()
plt.savefig(FIG_DIR / "iou_histogram.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "iou_histogram.png", bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"Wrote {FIG_DIR / 'iou_histogram.pdf'} (and .png)")


# ── Sanity checks ─────────────────────────────────────────────────────────


def _summary(name: str, values: list[float], unit: str = "") -> None:
    arr = np.asarray(values)
    print(f"  {name:<38} n={len(arr):3d}  "
          f"mean={arr.mean():.3f}{unit}  median={np.median(arr):.3f}{unit}")


print("\nData summary:")
print("Boundary segmentation (n cases, mean IoU, median IoU):")
_summary("VLM-direct (Flash)", vlm_flash_seg)
_summary("VLM-direct (Pro)", vlm_pro_seg)
_summary("Vanilla SAM-3 (highlighted marked area)", vanilla_sam)
_summary("SAM3-LoRA (out-of-fold)", sam_lora)
print("\nLocate centroid error (n cases, mean m, median m):")
_summary("VLM-direct geocode (Flash)", locate_vlm, " m")
_summary("Locate place only (production)", locate_production, " m")
_summary("Locate all 6 tools", locate_full_kit, " m")
_summary("Full pipeline (+ match_at)", pipeline_errs, " m")
print(f"\nGeoJSON IoU histogram: n={len(pipeline_ious)} "
      f"≥0.9: {n_ge09} ({n_ge09/total*100:.1f}%)  "
      f"<0.05: {n_lt005} ({n_lt005/total*100:.1f}%)  "
      f"[0.3, 0.7]: {n_mid} ({n_mid/total*100:.1f}%)")
