#!/usr/bin/env python3
"""
Compute SAM-based complexity features (segment counts) for MS COCO images.

For each image, runs SAM's automatic mask generator at three granularities
(points_per_side = 16, 32, 64) and records the number of segments at each.
These counts are the raw features for the SAM-based complexity score.

The script is crash-safe: results are appended per image, and on restart it
skips images already in the output CSV.

Requirements:
    pip install torch torchvision opencv-python numpy pandas tqdm
    pip install git+https://github.com/facebookresearch/segment-anything.git

    Download the SAM ViT-H checkpoint:
    https://github.com/facebookresearch/segment-anything#model-checkpoints

Usage:
    # All images in a directory
    python compute_sam_counts.py --images val2014 --checkpoint sam_vit_h_4b8939.pth --output sam_counts.csv

    # Only images listed in a subset CSV
    python compute_sam_counts.py --images val2014 --ids subsets/eval_900_ids.csv --checkpoint sam_vit_h_4b8939.pth --output sam_counts_900.csv

    # Optionally add --score to normalise and compute groups
    python compute_sam_counts.py --images val2014 --ids subsets/eval_900_ids.csv --checkpoint sam_vit_h_4b8939.pth --output sam_counts_900.csv --score
"""

import argparse
import gc
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# SAM automatic mask generator settings (matching the paper)
SAM_MODEL_TYPE   = "vit_h"
POINTS_LIST      = [16, 32, 64]
SAM_PRED_IOU     = 0.88
SAM_STAB_THRESH  = 0.95
SAM_CROP_LAYERS  = 0
SAM_MIN_AREA     = 100


def main():
    ap = argparse.ArgumentParser(
        description="Compute SAM segment counts at multiple granularities."
    )
    ap.add_argument("--images", required=True,
                    help="Path to image directory (e.g. val2014/).")
    ap.add_argument("--ids", default=None,
                    help="Optional CSV with an image_id column to filter to a subset.")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to SAM ViT-H checkpoint (sam_vit_h_4b8939.pth).")
    ap.add_argument("--output", default="sam_counts.csv",
                    help="Output CSV path (default: %(default)s).")
    ap.add_argument("--score", action="store_true",
                    help="If set, also compute the normalised composite score and "
                         "low/medium/high groups.")
    args = ap.parse_args()

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    img_dir = Path(args.images)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Resume: skip images already in the output CSV
    done = set()
    if out_path.exists():
        done = set(pd.read_csv(out_path)["image_id"].tolist())
        print(f"Resuming — {len(done)} images already done.")

    # Load SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=args.checkpoint)
    sam.to(device=device)
    sam.eval()

    # Process images
    header_written = out_path.exists()
    todo = [(iid, fn) for iid, fn in zip(image_ids, files) if iid not in done]

    for image_id, fname in tqdm(todo, desc="SAM segment counts"):
        img = cv2.imread(str(img_dir / fname))
        if img is None:
            print(f"  WARNING: could not read {fname}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        row = {"image_id": image_id}
        for pps in POINTS_LIST:
            mg = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=pps,
                pred_iou_thresh=SAM_PRED_IOU,
                stability_score_thresh=SAM_STAB_THRESH,
                crop_n_layers=SAM_CROP_LAYERS,
                min_mask_region_area=SAM_MIN_AREA,
            )
            masks = mg.generate(img_rgb)
            row[f"sam_segments_{pps}"] = len(masks)
            del mg
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        # Append immediately (crash-safe)
        pd.DataFrame([row]).to_csv(
            out_path, mode="a", header=not header_written, index=False
        )
        header_written = True

    print(f"Done -> {out_path}")

    # Optionally compute normalised composite score and groups
    if args.score and out_path.exists():
        df = pd.read_csv(out_path).drop_duplicates("image_id")
        seg_cols = [f"sam_segments_{p}" for p in POINTS_LIST]
        for col in seg_cols:
            lo, hi = df[col].min(), df[col].max()
            df[col + "_norm"] = (df[col] - lo) / (hi - lo) if hi != lo else 0.0
        norm_cols = [col + "_norm" for col in seg_cols]
        df["sam_score"] = df[norm_cols].mean(axis=1)
        df["sam_group"] = pd.qcut(
            df["sam_score"], q=3, labels=["low", "medium", "high"]
        )
        df.to_csv(out_path, index=False)
        print(f"Added sam_score and sam_group columns.")
        print(df["sam_group"].value_counts().to_string())


if __name__ == "__main__":
    main()
