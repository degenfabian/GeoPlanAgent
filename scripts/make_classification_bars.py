"""Bar charts of IoU broken down by document classification.

Two metric flavours are rendered with the same layout/style:
  end-to-end GeoJSON IoU (std run, n=208 cases)
  SAM3-LoRA pixel IoU    (cached k-fold predictions, n=208 cases after
                          averaging the per-page sem_iou for the 2
                          multi-page cases A108P (3 pages) and
                          A4D6A_merged (2 pages))

Reads:
  results/benchmark_std_post_fix/gemini-flash/summary.json
  training/eval/predictions/sam_kfold.json
  evaluation_data/new_updated.xlsx

Writes (PDF + PNG):
  figures/cls_bars_combined.{pdf,png}      — end-to-end, three-panel
  figures/cls_bars_colour.{pdf,png}        — end-to-end, standalone
  figures/cls_bars_quality.{pdf,png}
  figures/cls_bars_complexity.{pdf,png}
  figures/cls_bars_seg_combined.{pdf,png}  — SAM, three-panel
  figures/cls_bars_seg_colour.{pdf,png}    — SAM, standalone
  figures/cls_bars_seg_quality.{pdf,png}
  figures/cls_bars_seg_complexity.{pdf,png}

Style: solid bars = mean IoU, hatched bars = fraction of cases at
IoU >= 0.8. Each panel starts with a Total bar followed by the
per-bucket bars; each panel uses a distinct hue.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)

SUMMARY = REPO / "results/benchmark_std_post_fix/gemini-flash/summary.json"
SAM_KFOLD = REPO / "training/eval/predictions/sam_kfold.json"
XLSX = REPO / "evaluation_data/new_updated.xlsx"


# Load per-case IoU and join to classifications


def _load_iou_by_folder() -> dict[str, float]:
    """End-to-end GeoJSON IoU per case folder from the std run."""
    data = json.loads(SUMMARY.read_text())
    return {c["folder"]: c["iou"] for c in data["per_case"]}


def _load_sam_iou_by_case() -> dict[str, float]:
    """SAM3-LoRA pixel sem_iou per case from cached k-fold predictions.

    sam_kfold.json has 211 raw entries; multi-page cases use a `_pN`
    suffix on the key (A108P_p4, A108P_p5, ... and A4D6A_merged_p1,
    A4D6A_merged_p2). We collapse to 208 case-level scores by averaging
    sem_iou across the pages of each multi-page case — one contribution
    per case, matching the end-to-end aggregation.
    """
    data = json.loads(SAM_KFOLD.read_text())
    by_case: dict[str, list[float]] = defaultdict(list)
    for key, val in data.items():
        case = re.sub(r"_p\d+$", "", key)
        by_case[case].append(val["sem_iou"])
    return {case: float(np.mean(ious)) for case, ious in by_case.items()}


def _load_labels() -> pd.DataFrame:
    df = pd.read_excel(XLSX, sheet_name="Cleaned_up_208_planning_dataset")
    mrg = pd.read_excel(XLSX, sheet_name="Merged cases")
    # 5 merged cases have different folder names in xlsx vs. on disk
    # (e.g. xlsx "12_A_B_C_merged" -> run folder "12_merged").
    bridge = dict(zip(mrg["Unnamed: 5"].astype(str),
                      mrg["Merged folder"].astype(str)))
    df["run_folder"] = df["Unique ID (Folder_Name)"].astype(str).map(
        lambda x: bridge.get(x, x))
    return df


def _norm(s: object) -> str | None:
    return str(s).strip().lower() if pd.notna(s) else None


def _bucket_with_other(series: pd.Series, keep: list[str],
                       fallback: str) -> pd.Series:
    return series.map(lambda x: x if x in keep else fallback)


# Plot helpers


# One distinct hue per classification panel; Total bar always grey.
# Chosen colour-blind-safe and visually distinct from the seg-bars navy.
C_TOTAL = "#6c757d"        # neutral grey baseline
C_COLOUR = "#1f4e79"       # navy
C_QUALITY = "#2a7f6b"      # teal/green
C_COMPLEXITY = "#c46d3a"   # warm amber/terracotta


def _bars_on_axes(ax, rows: list[tuple], hue: str,
                  show_ylabel: bool, show_legend: bool,
                  bar_width: float = 0.38, label_fontsize: int = 8) -> None:
    """Draw the solid/hatched bar pair for one classification on ax."""
    labels = [r[0] for r in rows]
    n_per = [r[1] for r in rows]
    means = [r[2] for r in rows]
    fracs = [r[3] for r in rows]

    x = np.arange(len(labels))
    w = bar_width

    bar_colors = [C_TOTAL if lab == "Total" else hue for lab in labels]
    bars_mean = ax.bar(x - w / 2, means, w, color=bar_colors,
                       edgecolor="white", linewidth=0.8,
                       label="Mean IoU")
    bars_frac = ax.bar(x + w / 2, fracs, w, facecolor="white",
                       edgecolor=bar_colors, linewidth=1.0,
                       hatch="///",
                       label=r"Fraction of cases at IoU $\geq 0.8$")

    # Nudge label anchors slightly outward (left label leftward, right label
    # rightward) so paired values never touch — matters when both bars are
    # tall and their text widths exceed half a bar-pair (e.g. "0.85" / "84%"
    # on Medium in shape complexity).
    for b, v in zip(bars_mean, means):
        ax.annotate(f"{v:.2f}",
                    xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(-2, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=label_fontsize)
    for b, v in zip(bars_frac, fracs):
        ax.annotate(f"{v*100:.0f}%",
                    xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(2, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=label_fontsize)

    tick_labels = [f"{lab}\n(n={n})" for lab, n in zip(labels, n_per)]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    if show_ylabel:
        ax.set_ylabel("Score")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    if show_legend:
        ax.legend(loc="upper right", frameon=False, fontsize=8)


def _draw_single(rows: list[tuple], title: str, hue: str,
                 out_stem: str) -> None:
    """Standalone single-panel chart (one classification)."""
    fig, ax = plt.subplots(figsize=(0.95 * len(rows) + 1.6, 3.0))
    _bars_on_axes(ax, rows, hue, show_ylabel=True, show_legend=True)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / f"{out_stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {FIG_DIR / out_stem}.pdf (and .png)")


def _draw_combined(panels: list[tuple], out_stem: str) -> None:
    """Three-panel two-column figure.

    panels: [(rows, sub_title, hue)] in display order."""
    widths = [len(p[0]) for p in panels]  # column count per panel
    # Figure width tuned to fit a two-column LaTeX figure* slot.
    # 7.8" gives Medium-bar value labels room to breathe in panel (c).
    fig, axes = plt.subplots(
        1, len(panels), figsize=(7.8, 2.8),
        gridspec_kw={"width_ratios": widths})

    for i, (ax, (rows, sub_title, hue)) in enumerate(zip(axes, panels)):
        # Compact panels share the tighter bar/label settings.
        _bars_on_axes(ax, rows, hue,
                      show_ylabel=(i == 0),
                      show_legend=False,
                      bar_width=0.34, label_fontsize=7)
        ax.set_title(sub_title, fontsize=10)

    # Single shared legend at the top so each panel keeps its data area.
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#444444",
                      edgecolor="white", linewidth=0.8),
        plt.Rectangle((0, 0), 1, 1, facecolor="white",
                      edgecolor="#444444", linewidth=1.0, hatch="///"),
    ]
    fig.legend(
        handles,
        ["Mean IoU", r"Fraction of cases at IoU $\geq 0.8$"],
        loc="upper center", ncol=2, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(FIG_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / f"{out_stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote {FIG_DIR / out_stem}.pdf (and .png)")


# Main


plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8})

base = _load_labels()
base["col_norm"] = base["Document Colour"].apply(_norm)
base["qual_norm"] = base["Document Quality"].apply(_norm)
base["comp_norm"] = base["Shape Complexity"].apply(_norm)

# Per PI: fold the 2 "Other" cases (1 Green, 1 White-with-Black-line) into
# White, leaving a clean two-bucket split (White vs Yellow).
base["colour_bucket"] = _bucket_with_other(
    base["col_norm"], ["yellow"], fallback="white")
base["quality_bucket"] = base["qual_norm"]
base["complexity_bucket"] = base["comp_norm"]


def total_row(df_: pd.DataFrame) -> tuple:
    return ("Total", len(df_), df_["iou"].mean(),
            (df_["iou"] >= 0.8).mean())


def rows_for(df_: pd.DataFrame, col: str, order: list[str],
             pretty: dict[str, str]) -> list[tuple]:
    out = [total_row(df_)]
    for b in order:
        sub = df_[df_[col] == b]
        out.append((pretty.get(b, b), len(sub), sub["iou"].mean(),
                    (sub["iou"] >= 0.8).mean()))
    return out


def _print_table(name: str, rows: list[tuple]) -> None:
    print(f"\n{name}")
    print(f"  {'bucket':<10} {'n':>4}  {'mean IoU':>9}  {'%>=0.8':>7}")
    for label, n, mean, frac in rows:
        print(f"  {label:<10} {n:>4}  {mean:>9.3f}  {frac*100:>6.1f}%")


def render_variant(iou_by_case: dict[str, float], prefix: str,
                   variant_label: str) -> None:
    """Build all 4 chart files (3 standalone + 1 combined) for one IoU source."""
    df = base.copy()
    df["iou"] = df["run_folder"].map(iou_by_case)
    missing = df[df["iou"].isna()]
    assert missing.empty, (
        f"[{variant_label}] Unmapped cases: {missing['run_folder'].tolist()}")

    colour_rows = rows_for(df, "colour_bucket",
                           ["white", "yellow"],
                           {"white": "White", "yellow": "Yellow"})
    quality_rows = rows_for(df, "quality_bucket",
                            ["good", "bad"],
                            {"good": "Good", "bad": "Bad"})
    complexity_rows = rows_for(df, "complexity_bucket",
                               ["easy", "medium", "hard"],
                               {"easy": "Easy", "medium": "Medium",
                                "hard": "Hard"})

    _draw_single(colour_rows,
                 "(a) Document colour", C_COLOUR, f"{prefix}_colour")
    _draw_single(quality_rows,
                 "(b) Document quality", C_QUALITY, f"{prefix}_quality")
    _draw_single(complexity_rows,
                 "(c) Shape complexity", C_COMPLEXITY, f"{prefix}_complexity")
    _draw_combined(
        [(colour_rows, "(a) Document colour", C_COLOUR),
         (quality_rows, "(b) Document quality", C_QUALITY),
         (complexity_rows, "(c) Shape complexity", C_COMPLEXITY)],
        f"{prefix}_combined")

    print(f"\n=== {variant_label} ===")
    _print_table("Document colour", colour_rows)
    _print_table("Document quality", quality_rows)
    _print_table("Shape complexity", complexity_rows)


render_variant(_load_iou_by_folder(), "cls_bars",
               "End-to-end GeoJSON IoU (std run)")
render_variant(_load_sam_iou_by_case(), "cls_bars_seg",
               "SAM3-LoRA pixel IoU (cached k-fold)")
