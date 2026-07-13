#!/usr/bin/env python3
"""
Generate the sentence-level CHAIR_S bar chart (Figure in the paper).

Three panels side by side, one per complexity measure (Handcrafted,
Visual-statistics, SAM-based). Each panel has three x-groups (Low / Medium /
High complexity) with four bars per group (Baseline, OPERA, REVERSE,
Attention Lens). Shared y-axis and legend.

Outputs:
    sentence_level_chair_bars.pdf   (vector, for LaTeX)
    sentence_level_chair_bars.png   (raster preview)

Usage:
    python plot_table2.py
    python plot_table2.py --output figures/   # save to a specific directory
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data from the paper's sentence-level CHAIR evaluation (Table 2).
# Source: sentence_level_per_image_chair_all_metrics.csv, grouped by
# complexity terciles (low / medium / high) per measure.
#
# Rows: [Baseline, OPERA, REVERSE, Attention Lens]
# Cols: [Low, Medium, High]
# Values: sentence-level CHAIR_S (%)
# ---------------------------------------------------------------------------

handcrafted = np.array([
    [19.4, 19.3, 17.4],   # Baseline
    [14.9, 18.5, 15.7],   # OPERA
    [12.7, 14.5, 12.5],   # REVERSE
    [ 6.9, 10.9, 10.1],   # Attention Lens
])

visual_statistics = np.array([
    [20.7, 19.1, 16.2],
    [16.3, 17.6, 15.4],
    [13.8, 12.3, 13.6],
    [ 9.1, 10.0,  9.1],
])

sam_based = np.array([
    [13.1, 21.8, 20.5],
    [ 9.4, 19.0, 19.6],
    [10.4, 13.1, 15.9],
    [ 4.1,  9.9, 13.3],
])

PANELS = [
    ("Handcrafted",       handcrafted),
    ("Visual-statistics", visual_statistics),
    ("SAM-based",         sam_based),
]

METHOD_NAMES = ["Baseline", "OPERA", "REVERSE", "Attention Lens"]
GROUP_NAMES  = ["Low", "Medium", "High"]


def main():
    ap = argparse.ArgumentParser(description="Plot sentence-level CHAIR_S bar chart.")
    ap.add_argument("--output", default=".", help="Output directory (default: current).")
    args = ap.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), sharey=True)

    n_methods = len(METHOD_NAMES)
    n_groups  = len(GROUP_NAMES)
    bar_w     = 0.20
    x         = np.arange(n_groups)

    for ax, (title, data) in zip(axes, PANELS):
        for m in range(n_methods):
            offset = (m - (n_methods - 1) / 2) * bar_w
            bars = ax.bar(x + offset, data[m], bar_w,
                          color=f"C{m}", label=METHOD_NAMES[m],
                          edgecolor="black", linewidth=0.4)
            for b, v in zip(bars, data[m]):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.3,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(GROUP_NAMES)
        ax.set_xlabel("Complexity group")
        ax.set_ylim(0, 26)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"Sentence-level CHAIR$_{S}$ (\%, $\downarrow$)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_methods,
               frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=10)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    pdf_path = out_dir / "sentence_level_chair_bars.pdf"
    png_path = out_dir / "sentence_level_chair_bars.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight", dpi=200)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
