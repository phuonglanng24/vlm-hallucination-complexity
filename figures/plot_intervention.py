#!/usr/bin/env python3
"""
Plot the SAM region-removal intervention results (H2 figure).

Produces two bar charts:
    1. CHAIR scores (CHAIRs, CHAIRi) vs removal threshold tau
    2. Grounding scores (Recall, Precision, F1) vs removal threshold tau

The data is hardcoded from the paper's intervention experiment (30-image subset).
Source: CHAIR evaluation of LLaVA-1.5-7B captions on original and modified images
at three removal thresholds (tau = 0.0005, 0.001, 1.0).

Usage:
    python plot_intervention.py
    python plot_intervention.py --output figures/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Data from the paper's SAM region-removal intervention (Table / Figure).
# Source: CHAIR evaluation on original vs modified images at three thresholds.
# ---------------------------------------------------------------------------
INTERVENTION_DATA = pd.DataFrame([
    ["Original", 63.3, 13.3, 79.6, 75.2, 77.4],
    [r"$\tau$=0.0005", 50.0, 10.2, 79.6, 79.6, 79.6],
    [r"$\tau$=0.001", 33.3, 8.4, 81.6, 83.2, 82.4],
    [r"$\tau$=1.0", 73.3, 26.6, 53.4, 61.1, 57.0],
], columns=["Condition", "CHAIRs", "CHAIRi", "Recall", "Precision", "F1"])


def main():
    ap = argparse.ArgumentParser(description="Plot SAM intervention results (H2).")
    ap.add_argument("--output", default=".", help="Output directory (default: current).")
    args = ap.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = INTERVENTION_DATA.copy()

    # Save the data as CSV for reference
    df.to_csv(out_dir / "sam_intervention_results.csv", index=False)

    # Plot 1: CHAIR scores (lower is better)
    ax = df.set_index("Condition")[["CHAIRs", "CHAIRi"]].plot(
        kind="bar", figsize=(6, 4.5), rot=0
    )
    ax.set_xlabel(r"Removal threshold $\tau$")
    ax.set_ylabel(r"Score ($\downarrow$)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "sam_intervention_chair.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "sam_intervention_chair.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir / 'sam_intervention_chair.pdf'}")

    # Plot 2: Grounding scores (higher is better)
    ax = df.set_index("Condition")[["Recall", "Precision", "F1"]].plot(
        kind="bar", figsize=(6, 4.5), rot=0
    )
    ax.set_xlabel(r"Removal threshold $\tau$")
    ax.set_ylabel(r"Score ($\uparrow$)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "sam_intervention_grounding.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "sam_intervention_grounding.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_dir / 'sam_intervention_grounding.pdf'}")


if __name__ == "__main__":
    main()
