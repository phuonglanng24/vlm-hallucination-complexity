#!/usr/bin/env python3
"""
Plot per-image regression of hallucination vs. continuous complexity score.

Produces three figures (one per complexity measure), each showing a scatter
plot with linear regression lines for every method. Used for the H1 regression
figure in the paper.

Input: a merged CSV with at least these columns:
    image_id, method, CHAIRs_image (or target metric),
    handcrafted_score, vis_score, sam_based_score

Usage:
    python plot_regression.py --input sentence_level_all_metrics.csv --output figures/

    # Plot a different target metric
    python plot_regression.py --input sentence_level_all_metrics.csv --target CHAIRi_image --output figures/

How to create the input CSV:
    For each method, run sentence-level CHAIR to get per-image metrics, then merge
    with the three complexity scores (from compute_handcrafted.py, compute_vis_metrics.py,
    compute_sam_counts.py). The merged CSV should have one row per (image_id, method)
    combination, with the continuous complexity scores attached. See README for details.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Display names for methods and measures
METHOD_ORDER = ["baseline", "opera", "reverse", "attention_lens"]
METHOD_NAMES = {
    "baseline": "Baseline",
    "opera": "OPERA",
    "reverse": "REVERSE",
    "attention_lens": "Attention Lens",
}

MEASURES = [
    ("handcrafted", "handcrafted_score", "Handcrafted complexity"),
    ("vis", "vis_score", "Visual-statistics complexity"),
    ("sam_based", "sam_based_score", "SAM-based complexity"),
]


def plot_regression(df: pd.DataFrame, target_col: str, output_dir: Path):
    """Plot one figure per complexity measure: scatter + regression per method."""
    slope_rows = []

    for metric_key, score_col, metric_title in MEASURES:
        if score_col not in df.columns:
            print(f"WARNING: column '{score_col}' not found, skipping {metric_key}")
            continue

        fig, ax = plt.subplots(figsize=(6, 4.5))

        for method in METHOD_ORDER:
            sub = df[df["method"] == method].copy()
            sub = sub.dropna(subset=[score_col, target_col])
            if sub.empty:
                continue

            x = sub[score_col].values.reshape(-1, 1)
            y = sub[target_col].values.astype(float)

            # Fit linear regression
            model = LinearRegression()
            model.fit(x, y)
            y_pred = model.predict(x)
            r2 = r2_score(y, y_pred)
            slope = model.coef_[0]
            intercept = model.intercept_

            # Scatter with small jitter for discrete-ish targets
            x_scatter = sub[score_col].values
            if target_col == "CHAIRs_image":
                y_scatter = y + np.random.normal(0, 0.006, size=len(y))
            else:
                y_scatter = y

            ax.scatter(x_scatter, y_scatter, alpha=0.10, s=12)

            # Regression line
            x_line = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
            y_line = model.predict(x_line)
            label = f"{METHOD_NAMES[method]} (slope={slope:.3f}, R\u00b2={r2:.2f})"
            ax.plot(x_line.ravel(), y_line, linewidth=2, label=label)

            slope_rows.append({
                "measure": metric_key,
                "method": method,
                "target": target_col,
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
                "n_images": len(sub),
            })

        ax.set_xlabel("Continuous complexity score")
        ax.set_ylabel(target_col.replace("_", " "))
        ax.set_title(metric_title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        # Save
        stem = f"regression_{metric_key}_{target_col}"
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_dir / stem}.pdf / .png")

    # Save slopes table
    if slope_rows:
        slope_df = pd.DataFrame(slope_rows)
        slope_path = output_dir / f"regression_slopes_{target_col}.csv"
        slope_df.to_csv(slope_path, index=False)
        print(f"Saved slope table: {slope_path}")
        print(slope_df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(
        description="Plot hallucination vs. complexity regression (H1 figure)."
    )
    ap.add_argument("--input", required=True,
                    help="Merged CSV with columns: image_id, method, target metric, "
                         "and three complexity score columns.")
    ap.add_argument("--target", default="CHAIRs_image",
                    help="Target column to regress (default: %(default)s).")
    ap.add_argument("--output", default="figures",
                    help="Output directory for figures (default: %(default)s).")
    args = ap.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    # Validate required columns
    required = ["image_id", "method", args.target]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV missing required columns: {missing}. "
            f"Available: {df.columns.tolist()}"
        )

    # Filter to known methods
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    if df.empty:
        raise ValueError(
            f"No rows with method in {METHOD_ORDER}. "
            f"Found methods: {df['method'].unique().tolist()}"
        )

    print(f"Loaded {len(df)} rows, {df['method'].nunique()} methods")
    plot_regression(df, args.target, output_dir)


if __name__ == "__main__":
    main()
