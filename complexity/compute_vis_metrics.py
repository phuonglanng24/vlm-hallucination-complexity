#!/usr/bin/env python3
"""
Compute visual-statistics complexity features for MS COCO images.

Extracts 11 low-level image features adapted from Chu et al. ("What Makes
Visualization Images Perceptually Demanding?"), normalises them (MinMax within
the processed set), and averages into a single visual-statistics complexity
score. Optionally partitions images into low/medium/high groups via tercile
split.

Features computed per image:
    edge_density, corner_density, distinct_color_count, color_entropy,
    grayscale_entropy, compression_ratio, colorfulness, contrast_std,
    texture_entropy, contour_density, contour_length_density

Usage:
    # All images in a directory
    python compute_vis_metrics.py --images val2014 --output vis_metrics.csv

    # Only images listed in a subset CSV (must have an image_id column)
    python compute_vis_metrics.py --images val2014 --ids subsets/eval_900_ids.csv --output vis_metrics_900.csv
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm


def _safe_entropy(hist: np.ndarray) -> float:
    """Shannon entropy (nats) of a histogram, with smoothing."""
    hist = hist.astype(np.float64)
    prob = hist / (hist.sum() + 1e-12)
    return float(scipy_entropy(prob + 1e-12))


def compute_features(img_bgr: np.ndarray) -> dict:
    """Compute all 11 visual-statistics features from a BGR image."""
    h, w = img_bgr.shape[:2]
    area = h * w
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Edge density (Canny)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float((edges > 0).sum() / area)

    # Corner density (Shi-Tomasi)
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=5000, qualityLevel=0.01, minDistance=3
    )
    corner_density = (0 if corners is None else len(corners)) / area

    # Quantised colour statistics (8 levels per channel = 512 bins)
    quant = (img_rgb // 32).astype(np.uint8)
    flat = quant.reshape(-1, 3)
    distinct_color_count = len(np.unique(flat, axis=0))
    bin_ids = (
        flat[:, 0].astype(np.int32) * 64
        + flat[:, 1].astype(np.int32) * 8
        + flat[:, 2].astype(np.int32)
    )
    color_entropy = _safe_entropy(np.bincount(bin_ids, minlength=512))

    # Grayscale entropy (256 bins)
    grayscale_entropy = _safe_entropy(
        cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    )

    # Compression ratio (PNG-encoded size / raw size)
    ok, enc = cv2.imencode(".png", img_bgr)
    compression_ratio = (len(enc.tobytes()) / img_bgr.size) if ok else np.nan

    # Colorfulness (Hasler & Süsstrunk)
    R, G, B = (img_rgb[:, :, i].astype(np.float32) for i in range(3))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    colorfulness = float(
        np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
        + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )

    # Contrast (standard deviation of grayscale, normalised to [0, 1])
    contrast_std = float(np.std(gray) / 255.0)

    # Texture entropy (Laplacian → histogram entropy)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    texture_entropy = _safe_entropy(
        cv2.calcHist([lap_abs], [0], None, [256], [0, 256]).flatten()
    )

    # Contour statistics
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_density = len(contours) / area
    contour_length_density = float(
        sum(cv2.arcLength(c, False) for c in contours) / area
    )

    return {
        "edge_density": edge_density,
        "corner_density": corner_density,
        "distinct_color_count": distinct_color_count,
        "color_entropy": color_entropy,
        "grayscale_entropy": grayscale_entropy,
        "compression_ratio": compression_ratio,
        "colorfulness": colorfulness,
        "contrast_std": contrast_std,
        "texture_entropy": texture_entropy,
        "contour_density": contour_density,
        "contour_length_density": contour_length_density,
    }


FEATURE_COLS = [
    "edge_density", "corner_density", "distinct_color_count", "color_entropy",
    "grayscale_entropy", "compression_ratio", "colorfulness", "contrast_std",
    "texture_entropy", "contour_density", "contour_length_density",
]


def main():
    ap = argparse.ArgumentParser(
        description="Compute visual-statistics complexity features for COCO images."
    )
    ap.add_argument("--images", required=True,
                    help="Path to image directory (e.g. val2014/).")
    ap.add_argument("--ids", default=None,
                    help="Optional CSV with an image_id column to filter to a subset.")
    ap.add_argument("--output", default="vis_metrics.csv",
                    help="Output CSV path (default: %(default)s).")
    ap.add_argument("--score", action="store_true",
                    help="If set, also compute the normalised composite score and "
                         "low/medium/high groups.")
    args = ap.parse_args()

    img_dir = Path(args.images)

    # Determine which images to process
    if args.ids:
        ids_df = pd.read_csv(args.ids)
        if "image_id" not in ids_df.columns:
            raise ValueError(f"{args.ids} must contain an 'image_id' column.")
        image_ids = ids_df["image_id"].tolist()
        files = [f"COCO_val2014_{iid:012d}.jpg" for iid in image_ids]
    else:
        files = sorted(f.name for f in img_dir.glob("COCO_val2014_*.jpg"))
        image_ids = [int(f.replace("COCO_val2014_", "").replace(".jpg", "")) for f in files]

    # Extract features
    rows = []
    for iid, fname in tqdm(zip(image_ids, files), total=len(files),
                           desc="Computing visual-statistics features"):
        img = cv2.imread(str(img_dir / fname))
        if img is None:
            print(f"  WARNING: could not read {fname}")
            continue
        feats = compute_features(img)
        feats["image_id"] = iid
        rows.append(feats)

    df = pd.DataFrame(rows)

    # Optionally compute normalised composite score and groups
    if args.score:
        for col in FEATURE_COLS:
            lo, hi = df[col].min(), df[col].max()
            df[col + "_norm"] = (df[col] - lo) / (hi - lo) if hi != lo else 0.0
        norm_cols = [col + "_norm" for col in FEATURE_COLS]
        df["vis_score"] = df[norm_cols].mean(axis=1)
        df["vis_group"] = pd.qcut(
            df["vis_score"], q=3, labels=["low", "medium", "high"]
        )

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    if args.score:
        print(df["vis_group"].value_counts().to_string())


if __name__ == "__main__":
    main()
