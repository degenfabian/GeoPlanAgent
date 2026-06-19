"""Render a per-case predicted-vs-ground-truth boundary on an OSM basemap.

The benchmark no longer caches per-case visualisations (it saves only the
scores and the predicted boundary). Run this on demand for any case — it
reads predicted.geojson from the run dir and the ground-truth GeoJSON from
data/<case>/, so nothing heavy needs to be kept on disk:

    uv run scripts/visualize_case.py --run-dir results/benchmark_v1/gemini-flash --case <folder>
    uv run scripts/visualize_case.py --run-dir <dir> --case <folder> -o out.pdf
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import contextily as ctx
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from shapely.geometry import shape
from shapely.ops import unary_union

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from geoplanagent.paths import DATA_DIR  # noqa: E402
from geoplanagent.metrics import load_case_ground_truth  # noqa: E402

# Fraction of the combined bounding box added on each side of the plot.
_VIZ_PADDING = 1.5


def visualize_comparison(
    predicted_geojson: Dict[str, Any],
    ground_truth_geojson: Optional[Dict[str, Any]],
    output_path: str,
) -> None:
    """Render predicted (green) and optional GT (blue) on an OSM basemap; save as PDF."""
    plt.close("all")

    pred_geom = shape(predicted_geojson["geometry"])
    pred_gdf = gpd.GeoDataFrame({"geometry": [pred_geom]}, crs="EPSG:4326")

    gt_gdf = None
    if ground_truth_geojson:
        gt_geom = shape(ground_truth_geojson["geometry"])
        gt_gdf = gpd.GeoDataFrame({"geometry": [gt_geom]}, crs="EPSG:4326")

    all_shapes = [pred_geom]
    if gt_gdf is not None:
        all_shapes.append(gt_geom)
    combined = unary_union(all_shapes)
    combined_gdf = gpd.GeoDataFrame({"geometry": [combined]}, crs="EPSG:4326")

    pred_merc = pred_gdf.to_crs(epsg=3857)
    combined_merc = combined_gdf.to_crs(epsg=3857)
    gt_merc = gt_gdf.to_crs(epsg=3857) if gt_gdf is not None else None

    fig, ax = plt.subplots(figsize=(14, 12))

    if gt_merc is not None:
        gt_merc.plot(ax=ax, facecolor="blue", edgecolor="blue", alpha=0.15, linewidth=2)
        gt_merc.boundary.plot(ax=ax, color="blue", linewidth=2.5)

    pred_merc.plot(ax=ax, facecolor="green", edgecolor="green", alpha=0.15, linewidth=2)
    pred_merc.boundary.plot(ax=ax, color="green", linewidth=2.5)

    minx, miny, maxx, maxy = combined_merc.total_bounds
    x_pad = (maxx - minx) * _VIZ_PADDING
    y_pad = (maxy - miny) * _VIZ_PADDING
    ax.set_xlim(minx - x_pad, maxx + x_pad)
    ax.set_ylim(miny - y_pad, maxy + y_pad)

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    legend_handles = [
        mpatches.Patch(facecolor="green", edgecolor="green", alpha=0.4, label="Extracted"),
    ]
    if gt_merc is not None:
        legend_handles.insert(
            0, mpatches.Patch(facecolor="blue", edgecolor="blue", alpha=0.4, label="Ground Truth")
        )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=12)

    ax.set_title("Extracted vs Ground Truth" if gt_merc is not None else "Extracted Boundary",
                 fontsize=14, pad=10)
    ax.set_axis_off()
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Visualization saved: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="benchmark run dir, e.g. results/benchmark_v1/gemini-flash")
    parser.add_argument("--case", required=True, help="case folder name")
    parser.add_argument("--eval-dir", default=str(DATA_DIR), help="dataset root holding <case>/*.geojson ground truth (default: %(default)s)")
    parser.add_argument("-o", "--output", default=None, help="output PDF (default: <run-dir>/<case>/viz_comparison.pdf)")
    args = parser.parse_args()

    predicted_path = Path(args.run_dir) / args.case / "predicted.geojson"
    if not predicted_path.exists():
        print(f"No predicted.geojson at {predicted_path}")
        return 1
    predicted = json.loads(predicted_path.read_text())
    ground_truth = load_case_ground_truth(Path(args.eval_dir) / args.case)
    output = args.output or str(Path(args.run_dir) / args.case / "viz_comparison.pdf")
    visualize_comparison(predicted, ground_truth, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
