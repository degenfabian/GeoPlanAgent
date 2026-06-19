"""Render the paper's bar-chart figures into figures/ (as .pdf). The numbers
behind them are verified by scripts/compute_tables.py; this script only draws
the images.

Figures and their number in the paper:
  - abl_seg_bars      -> Figure 3  segmentation-method comparison on pixel IoU:
                        VLM-direct (Flash/Pro), vanilla SAM 3 (best prompt),
                        and SAM 3-LoRA.
  - cls_bars_combined -> Figure 4  end-to-end GeoJSON IoU by document colour,
                        quality, and shape complexity.

In both figures, solid bars are mean IoU and hatched bars are the fraction of
cases at IoU >= 0.8. A figure whose input data isn't on disk is skipped.

Reads: the main run's per-case metrics (--run-dir), the cached SAM k-fold
predictions, the vanilla-SAM / VLM-seg ablation CSVs, and the dataset spreadsheet.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from geoplanagent.metrics import (  # noqa: E402
    load_sam_iou_by_case,
    pre_critic_iou_by_case,
    seg_iou_by_case,
)
from geoplanagent.paths import ABL_SAM_BASE, ABL_VLM_SEG, MAIN_RUN_DIR  # noqa: E402
from geoplanagent.utils import load_case_labels, CASE_LABEL_BUCKETS  # noqa: E402

FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)

N_CASES = 208


# Plot helpers


# Bar hues (the Total bar is always grey; each classification panel gets its own)
C_TOTAL = "#6c757d"  # neutral grey baseline
C_COLOUR = "#1f4e79"  # navy
C_QUALITY = "#2a7f6b"  # teal/green
C_COMPLEXITY = "#c46d3a"  # warm amber/terracotta
C_SEG = "#33475b"  # dark slate — segmentation method-comparison bars


def _bars_on_axes(
    ax,
    rows: list[tuple],
    hue: str,
    show_ylabel: bool,
    show_legend: bool,
    bar_width: float = 0.38,
    label_fontsize: int = 8,
) -> None:
    """Draw the solid/hatched bar pair for each row on ax."""
    labels = [row[0] for row in rows]
    n_per = [row[1] for row in rows]
    means = [row[2] for row in rows]
    fracs = [row[3] for row in rows]

    x = np.arange(len(labels))
    w = bar_width

    bar_colors = [C_TOTAL if label == "Total" else hue for label in labels]
    bars_mean = ax.bar(
        x - w / 2, means, w, color=bar_colors, edgecolor="white", linewidth=0.8, label="Mean IoU"
    )
    bars_frac = ax.bar(
        x + w / 2,
        fracs,
        w,
        facecolor="white",
        edgecolor=bar_colors,
        linewidth=1.0,
        hatch="///",
        label=r"Fraction of cases at IoU $\geq 0.8$",
    )

    # Nudge label anchors slightly outward (left label leftward, right label
    # rightward) so paired values never touch — matters when both bars are
    # tall and their text widths exceed half a bar-pair (e.g. "0.85" / "84%"
    # on Medium in shape complexity).
    for bar, value in zip(bars_mean, means):
        ax.annotate(
            f"{value:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(-2, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=label_fontsize,
        )
    for bar, value in zip(bars_frac, fracs):
        ax.annotate(
            f"{value * 100:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(2, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=label_fontsize,
        )

    tick_labels = [f"{label}\n(n={n})" for label, n in zip(labels, n_per)]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    if show_ylabel:
        ax.set_ylabel("Score")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    if show_legend:
        # Upper LEFT: the tallest bars (SAM3-LoRA) sit on the right edge, so the
        # top-right corner would collide with their value labels.
        ax.legend(loc="upper left", frameon=False, fontsize=8)


def _draw_single(rows: list[tuple], title: str, hue: str, out_stem: str) -> None:
    """Standalone single-panel bar chart (Figure 3's segmentation-method comparison)."""
    fig, ax = plt.subplots(figsize=(0.95 * len(rows) + 1.6, 3.0))
    _bars_on_axes(ax, rows, hue, show_ylabel=True, show_legend=True)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / out_stem}.pdf")


def _draw_combined(panels: list[tuple], out_stem: str) -> None:
    """Three-panel two-column figure.

    panels: [(rows, sub_title, hue)] in display order."""
    widths = [len(panel[0]) for panel in panels]  # column count per panel
    # Figure width tuned to fit a two-column LaTeX figure* slot.
    # 7.8" gives Medium-bar value labels room to breathe in panel (c).
    fig, axes = plt.subplots(
        1, len(panels), figsize=(7.8, 2.8), gridspec_kw={"width_ratios": widths}
    )

    for i, (ax, (rows, sub_title, hue)) in enumerate(zip(axes, panels)):
        # Compact panels share the tighter bar/label settings.
        _bars_on_axes(
            ax, rows, hue, show_ylabel=(i == 0), show_legend=False, bar_width=0.34, label_fontsize=7
        )
        ax.set_title(sub_title, fontsize=10)

    # Single shared legend at the top so each panel keeps its data area.
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#444444", edgecolor="white", linewidth=0.8),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="white", edgecolor="#444444", linewidth=1.0, hatch="///"
        ),
    ]
    fig.legend(
        handles,
        ["Mean IoU", r"Fraction of cases at IoU $\geq 0.8$"],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(FIG_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / out_stem}.pdf")


# Figure builders


def total_row(df: pd.DataFrame) -> tuple:
    return ("Total", len(df), df["iou"].mean(), (df["iou"] >= 0.8).mean())


def rows_for(df: pd.DataFrame, col: str, order: list[str]) -> list[tuple]:
    # Every scored case must land in exactly one bucket (normalise_label folds
    # the "?" labels in) — a case outside the buckets would silently shrink the
    # per-bucket bars while staying in Total, so fail loudly instead.
    dropped = df[~df[col].isin(order)]
    assert dropped.empty, (
        f"{len(dropped)} case(s) outside {col} buckets {order}: "
        f"{sorted(str(value) for value in dropped[col].dropna().unique())}"
    )
    out = [total_row(df)]
    for bucket in order:
        subset = df[df[col] == bucket]
        out.append((bucket.title(), len(subset), subset["iou"].mean(), (subset["iou"] >= 0.8).mean()))
    return out


def draw_classification_bars(run_dir: Path) -> None:
    """Figure 4 (cls_bars_combined): the main run's pre-critic end-to-end IoU
    broken down by document colour, quality, and shape complexity."""
    df = load_case_labels()
    iou_by_case = pre_critic_iou_by_case(run_dir)
    df["iou"] = df["folder"].map(iou_by_case)

    # Loud failure in both directions: every dataset case must be scored, and
    # every scored case must be in the dataset.
    missing = df[df["iou"].isna()]
    assert missing.empty, f"Dataset cases missing from the run: {missing['folder'].tolist()}"
    extra = set(iou_by_case) - set(df["folder"])
    assert not extra, f"Run cases not in the dataset sheet: {sorted(extra)}"

    _draw_combined(
        [
            (rows_for(df, "colour", CASE_LABEL_BUCKETS["colour"]), "(a) Document colour", C_COLOUR),
            (rows_for(df, "quality", CASE_LABEL_BUCKETS["quality"]), "(b) Document quality", C_QUALITY),
            (
                rows_for(df, "complexity", CASE_LABEL_BUCKETS["complexity"]),
                "(c) Shape complexity",
                C_COMPLEXITY,
            ),
        ],
        "cls_bars_combined",
    )


def draw_seg_method_bars() -> None:
    """Figure 3 (abl_seg_bars): segmentation-method comparison on all 208 cases.
    Solid bars = mean pixel IoU, hatched = fraction at IoU >= 0.8. Four regimes:
    VLM-direct (Flash/Pro), vanilla SAM3 (best prompt), SAM3-LoRA."""
    methods = [
        ("VLM (Flash)", seg_iou_by_case(ABL_VLM_SEG / "gemini-flash" / "results.csv")),
        ("VLM (Pro)", seg_iou_by_case(ABL_VLM_SEG / "gemini-pro" / "results.csv")),
        ("Vanilla SAM3", seg_iou_by_case(ABL_SAM_BASE / "highlighted_marked_area" / "results.csv")),
        ("SAM3-LoRA", np.asarray(list(load_sam_iou_by_case().values()))),
    ]
    for label, ious in methods:
        # seg_iou_by_case skips rows with blank iou cells — catch a silently
        # shrunken input here rather than shipping a figure with a wrong n.
        assert len(ious) == N_CASES, f"{label}: n={len(ious)}, expected {N_CASES}"
    rows = [
        (label, len(ious), float(ious.mean()), float(np.mean(ious >= 0.8)))
        for label, ious in methods
    ]
    _draw_single(rows, "Segmentation method (pixel IoU)", C_SEG, "abl_seg_bars")


# Main


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=MAIN_RUN_DIR,
        help="benchmark run to read end-to-end IoU from (default: %(default)s)",
    )
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )

    for figure in (draw_seg_method_bars, lambda: draw_classification_bars(args.run_dir)):
        try:
            figure()
        except FileNotFoundError as error:
            print(f"skipped — input data not on disk ({error})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
