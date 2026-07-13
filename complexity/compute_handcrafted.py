#!/usr/bin/env python3
"""
Compute handcrafted complexity features for MS COCO images.

Extracts four per-image features:
    object_count     — total annotated object instances (from COCO annotations)
    entropy          — grayscale Shannon entropy (bits, 256 bins)
    color_diversity  — Shannon entropy (bits) over quantized RGB (512 bins)
    edge_density     — fraction of Canny edge pixels

Optionally normalises (MinMax within the processed set), averages into a single
handcrafted complexity score, and partitions into low/medium/high groups.

Usage:
    # All images in a directory
    python compute_handcrafted.py --images val2014 --instances instances_val2014.json --output hand_features.csv

    # Only images listed in a subset CSV
    python compute_handcrafted.py --images val2014 --instances instances_val2014.json --ids subsets/eval_900_ids.csv --output hand_features_900.csv

    # With normalised score and groups
    python compute_handcrafted.py --images val2014 --instances instances_val2014.json --ids subsets/eval_900_ids.csv --output hand_features_900.csv --score
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_object_counts(instances_path: str) -> dict:
    """Load COCO annotations and return {image_id: total_instance_count}."""
    with open(instances_path, "r") as f:
        data = json.load(f)
    counts = {}
    for ann in data["annotations"]:
        counts[ann["image_id"]] = counts.get(ann["image_id"], 0) + 1
    return counts


def compute_pixel_features(img_bgr: np.ndarray) -> dict:
    """Compute entropy, color_diversity, and edge_density from a BGR image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Grayscale Shannon entropy (bits, 256 bins)
    gh = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().astype(np.float64)
    pg = gh / (gh.sum() + 1e-12)
    pg = pg[pg > 0]
    entropy = float(-(pg * np.log2(pg)).sum())

    # Edge density (Canny 100/200)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float((edges > 0).sum() / edges.size)

    # Color diversity = Shannon entropy (bits) over quantized RGB
    # 8 levels per channel → 512 bins
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    q = (img_rgb // 32).astype(np.int32)
    bin_ids = (q[:, :, 0] * 64 + q[:, :, 1] * 8 + q[:, :, 2]).ravel()
    cc = np.bincount(bin_ids, minlength=512).astype(np.float64)
    pc = cc / (cc.sum() + 1e-12)
    pc = pc[pc > 0]
    color_diversity = float(-(pc * np.log2(pc)).sum())

    return {
        "entropy": entropy,
        "color_diversity": color_diversity,
        "edge_density": edge_density,
    }


FEATURE_COLS = ["object_count", "entropy", "color_diversity", "edge_density"]


def main():
    ap = argparse.ArgumentParser(
        description="Compute handcrafted complexity features for COCO images."
    )
    ap.add_argument("--images", required=True,
                    help="Path to image directory (e.g. val2014/).")
    ap.add_argument("--instances", required=True,
                    help="Path to COCO instances JSON (e.g. instances_val2014.json).")
    ap.add_argument("--ids", default=None,
                    help="Optional CSV with an image_id column to filter to a subset.")
    ap.add_argument("--output", default="hand_features.csv",
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
        image_ids = [
            int(f.replace("COCO_val2014_", "").replace(".jpg", "")) for f in files
        ]

    # Load COCO object counts
    obj_counts = load_object_counts(args.instances)

    # Extract features
    rows = []
    for iid, fname in tqdm(zip(image_ids, files), total=len(files),
                           desc="Computing handcrafted features"):
        img = cv2.imread(str(img_dir / fname))
        if img is None:
            print(f"  WARNING: could not read {fname}")
            continue
        feats = compute_pixel_features(img)
        feats["image_id"] = iid
        feats["object_count"] = obj_counts.get(iid, 0)
        rows.append(feats)

    df = pd.DataFrame(rows)

    # Optionally compute normalised composite score and groups
    if args.score:
        for col in FEATURE_COLS:
            lo, hi = df[col].min(), df[col].max()
            df[col + "_norm"] = (df[col] - lo) / (hi - lo) if hi != lo else 0.0
        norm_cols = [col + "_norm" for col in FEATURE_COLS]
        df["handcrafted_score"] = df[norm_cols].mean(axis=1)
        df["handcrafted_group"] = pd.qcut(
            df["handcrafted_score"], q=3, labels=["low", "medium", "high"]
        )

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    if args.score:
        print(df["handcrafted_group"].value_counts().to_string())


if __name__ == "__main__":
    main()
