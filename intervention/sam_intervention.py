#!/usr/bin/env python3
"""
SAM region-removal intervention (Hypothesis 2).

For each image, runs SAM automatic mask generation, selects all masks whose
area is at most tau * image_area, combines them into a single binary mask,
dilates it slightly, and inpaints the masked regions using OpenCV Telea
inpainting. The output is a set of modified images that can be re-captioned
with LLaVA and re-evaluated with CHAIR to measure how removing visual content
affects hallucination.

The paper uses three thresholds:
    tau = 0.0005  (remove only very small regions)
    tau = 0.001   (remove small regions)
    tau = 1.0     (remove all detected regions)

Requirements:
    pip install torch opencv-python numpy tqdm
    pip install git+https://github.com/facebookresearch/segment-anything.git

Usage:
    # Remove regions below 0.05% of image area
    python sam_intervention.py --images val2014 --checkpoint sam_vit_b_01ec64.pth \
        --threshold 0.0005 --output intervened_0005/

    # Remove regions below 0.1% of image area
    python sam_intervention.py --images val2014 --checkpoint sam_vit_b_01ec64.pth \
        --threshold 0.001 --output intervened_001/

    # Remove ALL detected regions
    python sam_intervention.py --images val2014 --checkpoint sam_vit_b_01ec64.pth \
        --threshold 1.0 --output intervened_all/

    # Process only a subset of images
    python sam_intervention.py --images val2014 --ids subsets/eval_900_ids.csv \
        --checkpoint sam_vit_b_01ec64.pth --threshold 0.0005 --output intervened_0005/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def create_removal_mask(masks, image_shape, area_threshold_ratio):
    """Select masks whose area <= threshold * image_area, combine into one mask."""
    h, w = image_shape[:2]
    image_area = h * w
    area_threshold = area_threshold_ratio * image_area

    combined_mask = np.zeros((h, w), dtype=np.uint8)
    selected = 0

    for m in masks:
        if int(m["area"]) <= area_threshold:
            combined_mask[m["segmentation"].astype(bool)] = 255
            selected += 1

    return combined_mask, selected, len(masks)


def dilate_mask(mask, kernel_size=5, iterations=1):
    """Slightly dilate the mask to ensure clean inpainting boundaries."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=iterations)


def inpaint_image(image_bgr, mask, radius=3):
    """Inpaint masked regions using OpenCV Telea algorithm."""
    return cv2.inpaint(image_bgr, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


def main():
    ap = argparse.ArgumentParser(
        description="SAM region-removal intervention: remove small segments and inpaint."
    )
    ap.add_argument("--images", required=True,
                    help="Path to image directory (e.g. val2014/).")
    ap.add_argument("--ids", default=None,
                    help="Optional CSV with image_id column to filter to a subset.")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to SAM checkpoint (e.g. sam_vit_b_01ec64.pth).")
    ap.add_argument("--model-type", default="vit_b",
                    choices=["vit_b", "vit_l", "vit_h"],
                    help="SAM model type (default: %(default)s).")
    ap.add_argument("--threshold", type=float, required=True,
                    help="Area threshold ratio tau: remove masks with area <= tau * image_area. "
                         "Paper uses 0.0005, 0.001, and 1.0.")
    ap.add_argument("--output", required=True,
                    help="Output directory for inpainted images.")
    ap.add_argument("--save-masks", action="store_true",
                    help="If set, also save the binary masks and dilated masks.")
    ap.add_argument("--points-per-side", type=int, default=64,
                    help="SAM points_per_side (default: %(default)s).")
    args = ap.parse_args()

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    import pandas as pd

    img_dir = Path(args.images)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_masks:
        mask_dir = out_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

    # Determine which images to process
    if args.ids:
        ids_df = pd.read_csv(args.ids)
        if "image_id" not in ids_df.columns:
            raise ValueError(f"{args.ids} must contain an 'image_id' column.")
        files = [f"COCO_val2014_{iid:012d}.jpg" for iid in ids_df["image_id"]]
    else:
        files = sorted(
            f.name for f in img_dir.glob("*.jpg")
        ) + sorted(
            f.name for f in img_dir.glob("*.png")
        )

    print(f"Images: {len(files)}")
    print(f"Threshold tau: {args.threshold}")
    print(f"Output: {out_dir}")

    # Load SAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    sam.eval()

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=0,
        min_mask_region_area=0,
    )

    # Process images
    for fname in tqdm(files, desc=f"Intervention (tau={args.threshold})"):
        img_path = img_dir / fname
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  WARNING: could not read {fname}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Generate masks and select small ones
        masks = mask_generator.generate(image_rgb)
        combined_mask, selected, total = create_removal_mask(
            masks, image_rgb.shape, args.threshold
        )

        # Dilate and inpaint
        dilated = dilate_mask(combined_mask, kernel_size=5, iterations=1)
        result = inpaint_image(image_bgr, dilated)

        # Save
        cv2.imwrite(str(out_dir / fname), result)

        if args.save_masks:
            stem = Path(fname).stem
            cv2.imwrite(str(mask_dir / f"{stem}_mask.png"), combined_mask)
            cv2.imwrite(str(mask_dir / f"{stem}_mask_dilated.png"), dilated)

    print(f"Done. Saved inpainted images to {out_dir}")


if __name__ == "__main__":
    main()
