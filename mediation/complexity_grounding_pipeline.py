#!/usr/bin/env python3
"""
Mediation pipeline: complexity -> grounding -> hallucination.

Runs on a fixed random subset of MS COCO val2014 images:
  1. Draws and freezes the subset (--stage freeze)
  2. Computes three visual-complexity scores (--stage hand / vis / sam)
  3. Computes CLIP object-grounding scores (--stage ground)
  4. Assembles a master table (--stage assemble)
  5. Runs diagnostics, H1 regressions, and mediation (--stage diag / h1 / mediation)

Each stage writes a CSV that the next stage reads, so the pipeline is resumable.

Requirements:
    pip install torch torchvision opencv-python numpy scipy pandas scikit-learn
                statsmodels tqdm transformers pillow
    pip install git+https://github.com/facebookresearch/segment-anything.git

Usage:
    python complexity_grounding_pipeline.py --stage freeze --labels labels.csv --images val2014 --instances instances_val2014.json
    python complexity_grounding_pipeline.py --stage hand
    python complexity_grounding_pipeline.py --stage vis
    python complexity_grounding_pipeline.py --stage sam --sam-checkpoint sam_vit_h_4b8939.pth
    python complexity_grounding_pipeline.py --stage ground
    python complexity_grounding_pipeline.py --stage assemble
    python complexity_grounding_pipeline.py --stage all_analysis
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# =============================== DEFAULTS ====================================
# All overridable via command-line arguments (see main()).
DEFAULT_LABELS    = "labels_40k.csv"
DEFAULT_IMAGES    = "val2014"
DEFAULT_INSTANCES = "instances_val2014.json"
DEFAULT_SAM_CKPT  = "sam_vit_h_4b8939.pth"
DEFAULT_OUT_DIR   = "pipeline_out"
DEFAULT_SUBSET_N  = 5000
DEFAULT_SEED      = 42

# SAM automatic mask generator settings (ViT-H)
SAM_MODEL_TYPE   = "vit_h"
POINTS_LIST      = [16, 32, 64]
SAM_PRED_IOU     = 0.88
SAM_STAB_THRESH  = 0.95
SAM_CROP_LAYERS  = 0
SAM_MIN_AREA     = 100

# CLIP grounding settings
CLIP_MODEL       = "openai/clip-vit-base-patch32"
WEAK_THRESH      = 0.24   # scores below this = "weakly grounded"

# Analysis settings
OUTCOME_COL       = "CHAIRi_image"
GROUNDING_PRIMARY = "min_object_grounding"
GROUNDING_ALT     = "mean_object_grounding"

# Global path variables — set by _init_paths() after argument parsing.
LABELS_CSV = IMG_DIR = INSTANCES = SAM_CKPT = OUT_DIR = None
SUBSET_N = SEED = None
IDS_CSV = HAND_CSV = VIS_CSV = SAM_CSV = GROUND_CSV = MASTER_CSV = None


def _init_paths(args):
    """Set global path variables from parsed command-line arguments."""
    global LABELS_CSV, IMG_DIR, INSTANCES, SAM_CKPT, OUT_DIR
    global SUBSET_N, SEED
    global IDS_CSV, HAND_CSV, VIS_CSV, SAM_CSV, GROUND_CSV, MASTER_CSV

    LABELS_CSV = Path(args.labels)
    IMG_DIR    = Path(args.images)
    INSTANCES  = Path(args.instances)
    SAM_CKPT   = Path(args.sam_checkpoint)
    OUT_DIR    = Path(args.output)
    SUBSET_N   = args.subset_n
    SEED       = args.seed

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    IDS_CSV    = OUT_DIR / f"subset_{SUBSET_N}_ids.csv"
    HAND_CSV   = OUT_DIR / f"hand_features_{SUBSET_N}.csv"
    VIS_CSV    = OUT_DIR / f"vis_metrics_{SUBSET_N}.csv"
    SAM_CSV    = OUT_DIR / f"sam_counts_{SUBSET_N}.csv"
    GROUND_CSV = OUT_DIR / f"grounding_{SUBSET_N}.csv"
    MASTER_CSV = OUT_DIR / f"master_{SUBSET_N}.csv"


# =============================== HELPERS =====================================
def _id_from_name(stem: str) -> int:
    return int(stem.replace("COCO_val2014_", ""))

def _name_from_id(image_id: int) -> str:
    return f"COCO_val2014_{image_id:012d}.jpg"

def load_subset_ids() -> pd.DataFrame:
    if not IDS_CSV.exists():
        raise FileNotFoundError(f"{IDS_CSV} not found - run --stage freeze first.")
    return pd.read_csv(IDS_CSV)

def minmax(col: pd.Series) -> pd.Series:
    lo, hi = col.min(), col.max()
    if hi == lo:
        return pd.Series(0.0, index=col.index)
    return (col - lo) / (hi - lo)

def zscore(col: pd.Series) -> pd.Series:
    sd = col.std(ddof=0)
    if sd == 0:
        return pd.Series(0.0, index=col.index)
    return (col - col.mean()) / sd


# =============================== STAGE: freeze ===============================
def stage_freeze():
    """Draw a fixed SUBSET_N random image_ids from labels_40k.csv and save them.
    Optionally copies the images into a subset folder (not required downstream)."""
    labels = pd.read_csv(LABELS_CSV)
    if "image_id" not in labels.columns:
        raise ValueError("labels_40k.csv must contain an 'image_id' column.")
    labels = labels.drop_duplicates("image_id")
    n = min(SUBSET_N, len(labels))
    ids = labels.sample(n=n, random_state=SEED)["image_id"].sort_values().reset_index(drop=True)
    out = pd.DataFrame({"image_id": ids})
    out["file_name"] = out["image_id"].map(_name_from_id)
    out.to_csv(IDS_CSV, index=False)
    print(f"[freeze] wrote {len(out)} ids to {IDS_CSV} (seed={SEED})")

    # optional copy into a subset folder for convenience
    subset_dir = OUT_DIR / f"eval_subset_{SUBSET_N}"
    subset_dir.mkdir(exist_ok=True)
    missing = 0
    for fn in tqdm(out["file_name"], desc="[freeze] copying images"):
        src = IMG_DIR / fn
        dst = subset_dir / fn
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
        elif not src.exists():
            missing += 1
    if missing:
        print(f"[freeze] WARNING: {missing} images not found in {IMG_DIR}")
    print(f"[freeze] images available under {subset_dir}")


# =============================== STAGE: hand =================================
def _object_counts_from_coco() -> dict:
    """image_id -> total number of annotated object instances (matches your
    handcrafted 'object_count', e.g. 391895 -> 4)."""
    with open(INSTANCES, "r") as f:
        data = json.load(f)
    counts = {}
    for ann in data["annotations"]:
        counts[ann["image_id"]] = counts.get(ann["image_id"], 0) + 1
    return counts

def _handcrafted_pixel_feats(img_bgr):
    import cv2
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # grayscale Shannon entropy (bits, 256 bins)
    gh = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().astype(np.float64)
    pg = gh / (gh.sum() + 1e-12)
    pg = pg[pg > 0]
    entropy = float(-(pg * np.log2(pg)).sum())

    # edge density (Canny 100/200)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float((edges > 0).sum() / edges.size)

    # color_diversity = Shannon entropy (bits) over quantized RGB (512 bins)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    q = (img_rgb // 32).astype(np.int32)  # 8 levels/channel
    bin_ids = (q[:, :, 0] * 64 + q[:, :, 1] * 8 + q[:, :, 2]).ravel()
    cc = np.bincount(bin_ids, minlength=512).astype(np.float64)
    pc = cc / (cc.sum() + 1e-12)
    pc = pc[pc > 0]
    color_diversity = float(-(pc * np.log2(pc)).sum())

    return entropy, color_diversity, edge_density

def stage_hand():
    import cv2
    ids = load_subset_ids()
    obj_counts = _object_counts_from_coco()
    rows = []
    for _, r in tqdm(list(ids.iterrows()), desc="[hand] pixel features"):
        image_id = int(r["image_id"])
        img = cv2.imread(str(IMG_DIR / r["file_name"]))
        if img is None:
            print("  could not read", r["file_name"]); continue
        entropy, color_div, edge_den = _handcrafted_pixel_feats(img)
        rows.append({
            "image_id": image_id,
            "object_count": int(obj_counts.get(image_id, 0)),
            "entropy": entropy,
            "color_diversity": color_div,
            "edge_density": edge_den,
        })
    pd.DataFrame(rows).to_csv(HAND_CSV, index=False)
    print(f"[hand] wrote {len(rows)} rows to {HAND_CSV}")


# =============================== STAGE: vis ==================================
# Reproduces the visual-statistics feature definitions from Chu et al.
def _vis_feats(img_bgr):
    import cv2
    from scipy.stats import entropy as scipy_entropy
    h, w = img_bgr.shape[:2]
    area = h * w
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    def safe_entropy(hist):
        hist = hist.astype(np.float64)
        prob = hist / (hist.sum() + 1e-12)
        return float(scipy_entropy(prob + 1e-12))

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float((edges > 0).sum() / area)

    corners = cv2.goodFeaturesToTrack(gray, maxCorners=5000, qualityLevel=0.01, minDistance=3)
    corner_density = (0 if corners is None else len(corners)) / area

    quant = (img_rgb // 32).astype(np.uint8)
    flat = quant.reshape(-1, 3)
    distinct_color_count = len(np.unique(flat, axis=0))
    bin_ids = flat[:, 0].astype(np.int32) * 64 + flat[:, 1].astype(np.int32) * 8 + flat[:, 2].astype(np.int32)
    color_entropy = safe_entropy(np.bincount(bin_ids, minlength=512))

    grayscale_entropy = safe_entropy(cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten())

    ok, enc = cv2.imencode(".png", img_bgr)
    compression_ratio = (len(enc.tobytes()) / img_bgr.size) if ok else np.nan

    R, G, B = (img_rgb[:, :, i].astype(np.float32) for i in range(3))
    rg = np.abs(R - G); yb = np.abs(0.5 * (R + G) - B)
    colorfulness = float(np.sqrt(np.std(rg)**2 + np.std(yb)**2)
                         + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2))
    contrast_std = float(np.std(gray) / 255.0)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    texture_entropy = safe_entropy(cv2.calcHist([lap_abs], [0], None, [256], [0, 256]).flatten())

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_density = len(contours) / area
    contour_length_density = float(sum(cv2.arcLength(c, False) for c in contours) / area)

    return {
        "edge_density": edge_density, "corner_density": corner_density,
        "distinct_color_count": distinct_color_count, "color_entropy": color_entropy,
        "grayscale_entropy": grayscale_entropy, "compression_ratio": compression_ratio,
        "colorfulness": colorfulness, "contrast_std": contrast_std,
        "texture_entropy": texture_entropy, "contour_density": contour_density,
        "contour_length_density": contour_length_density,
    }

def stage_vis():
    import cv2
    ids = load_subset_ids()
    rows = []
    for _, r in tqdm(list(ids.iterrows()), desc="[vis] visual-stats features"):
        img = cv2.imread(str(IMG_DIR / r["file_name"]))
        if img is None:
            print("  could not read", r["file_name"]); continue
        rows.append({"image_id": int(r["image_id"]), **_vis_feats(img)})
    pd.DataFrame(rows).to_csv(VIS_CSV, index=False)
    print(f"[vis] wrote {len(rows)} rows to {VIS_CSV}")


# =============================== STAGE: sam ==================================
# SAM segment counts (ViT-H, points_per_side 16/32/64). Resumable and chunkable.
def stage_sam(chunk_index: int = 0, chunk_total: int = 1):
    import cv2, torch, gc
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    ids = load_subset_ids()
    # Deterministic split: sort by image_id (already sorted from freeze) and take
    # every chunk_total-th row starting at chunk_index. This gives every machine
    # a stable, non-overlapping slice regardless of how many machines you use.
    if chunk_total > 1:
        ids = ids.iloc[chunk_index::chunk_total].reset_index(drop=True)
        print(f"[sam] chunk {chunk_index+1}/{chunk_total} -> {len(ids)} ids")

    # Chunked runs write to their own CSV so machines don't collide.
    if chunk_total > 1:
        sam_csv = OUT_DIR / f"sam_counts_{SUBSET_N}_chunk{chunk_index}of{chunk_total}.csv"
    else:
        sam_csv = SAM_CSV

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[sam] device:", device, "| checkpoint:", SAM_CKPT, "exists:", SAM_CKPT.exists())
    print("[sam] output:", sam_csv)

    done = set()
    if sam_csv.exists():
        done = set(pd.read_csv(sam_csv)["image_id"].tolist())
        print(f"[sam] resuming - {len(done)} already done in this chunk")

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CKPT))
    sam.to(device=device); sam.eval()

    header_written = sam_csv.exists()
    todo = [r for _, r in ids.iterrows() if int(r["image_id"]) not in done]
    for r in tqdm(todo, desc=f"[sam] chunk {chunk_index+1}/{chunk_total}"):
        image_id = int(r["image_id"])
        img = cv2.imread(str(IMG_DIR / r["file_name"]))
        if img is None:
            print("  could not read", r["file_name"]); continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        row = {"image_id": image_id}
        for pps in POINTS_LIST:
            mg = SamAutomaticMaskGenerator(
                model=sam, points_per_side=pps,
                pred_iou_thresh=SAM_PRED_IOU, stability_score_thresh=SAM_STAB_THRESH,
                crop_n_layers=SAM_CROP_LAYERS, min_mask_region_area=SAM_MIN_AREA,
            )
            row[f"sam_segments_{pps}"] = len(mg.generate(img))
            del mg; gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        pd.DataFrame([row]).to_csv(sam_csv, mode="a", header=not header_written, index=False)
        header_written = True
    print(f"[sam] done -> {sam_csv}")


def stage_sam_merge():
    """After all chunks finish, concatenate them into the single sam_counts_{N}.csv
    that --stage assemble expects. Safe to run repeatedly."""
    chunks = sorted(OUT_DIR.glob(f"sam_counts_{SUBSET_N}_chunk*of*.csv"))
    if not chunks:
        print(f"[sam_merge] no chunk files found under {OUT_DIR}")
        return
    print(f"[sam_merge] found {len(chunks)} chunk files:")
    for c in chunks:
        print(f"  {c.name}")
    df = pd.concat([pd.read_csv(c) for c in chunks], ignore_index=True)
    df = df.drop_duplicates("image_id").sort_values("image_id").reset_index(drop=True)
    df.to_csv(SAM_CSV, index=False)
    print(f"[sam_merge] wrote {len(df)} rows to {SAM_CSV}")
    expected = SUBSET_N
    if len(df) != expected:
        print(f"[sam_merge] WARNING: expected {expected} rows, got {len(df)} -"
              f" some ids may not have been covered. Check chunk assignments.")





# =============================== STAGE: ground ===============================
def _coco_category_names():
    with open(INSTANCES, "r") as f:
        data = json.load(f)
    return [c["name"] for c in data["categories"]]

def _mentioned(caption: str, cat_names) -> list:
    cap = caption.lower()
    hits = []
    for name in cat_names:
        # word-boundary match, allow a trailing plural 's'
        if re.search(r"\b" + re.escape(name) + r"s?\b", cap):
            hits.append(name)
    return hits

def _clip_features(out):
    # Handle both older (returns Tensor) and newer (returns
    # BaseModelOutputWithPooling with .pooler_output) transformers versions.
    import torch as _torch
    if isinstance(out, _torch.Tensor):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0]  # CLIP-style CLS-equivalent
    raise TypeError(f"unexpected CLIP output type: {type(out)}")

def stage_ground():
    """CLIP object-grounding features (the mediator). For each caption we find the
    COCO categories it mentions, score each against the image with CLIP cosine, and
    summarise. NOTE: object extraction here is approximate (string match on the 80
    COCO names); swap in CHAIR's synonym mapping if you want exact parity."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    ids = load_subset_ids()
    labels = pd.read_csv(LABELS_CSV)[["image_id", "caption"]].drop_duplicates("image_id")
    ids = ids.merge(labels, on="image_id", how="left")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # use_safetensors avoids the torch.load CVE block on torch<2.6; safetensors is
    # always downloaded alongside the .bin for openai/clip-vit-base-patch32.
    model = CLIPModel.from_pretrained(CLIP_MODEL, use_safetensors=True).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    cat_names = _coco_category_names()

    # precompute text embeddings for all 80 categories
    with torch.no_grad():
        tin = proc(text=[f"a photo of a {c}" for c in cat_names],
                   return_tensors="pt", padding=True).to(device)
        temb = _clip_features(model.get_text_features(**tin))
        temb = temb / temb.norm(dim=-1, keepdim=True)
    cat_to_emb = {c: temb[i] for i, c in enumerate(cat_names)}

    rows = []
    for _, r in tqdm(list(ids.iterrows()), desc="[ground] CLIP grounding"):
        image_id = int(r["image_id"])
        caption = str(r.get("caption", "") or "")
        cats = _mentioned(caption, cat_names)
        row = {"image_id": image_id, "n_mentioned": len(cats)}
        if not cats:
            row.update({"min_object_grounding": np.nan, "mean_object_grounding": np.nan,
                        "num_weak_objects": 0, "frac_weak_objects": np.nan,
                        "grounding_gap": np.nan})
            rows.append(row); continue
        try:
            pil = Image.open(IMG_DIR / r["file_name"]).convert("RGB")
        except Exception:
            continue
        with torch.no_grad():
            iin = proc(images=pil, return_tensors="pt").to(device)
            iemb = _clip_features(model.get_image_features(**iin))
            iemb = (iemb / iemb.norm(dim=-1, keepdim=True)).squeeze(0)
            sims = np.array([float(torch.dot(cat_to_emb[c], iemb)) for c in cats])
        weak = sims < WEAK_THRESH
        row.update({
            "min_object_grounding": float(sims.min()),
            "mean_object_grounding": float(sims.mean()),
            "num_weak_objects": int(weak.sum()),
            "frac_weak_objects": float(weak.mean()),
            "grounding_gap": float(sims.max() - sims.min()),
        })
        rows.append(row)
    pd.DataFrame(rows).to_csv(GROUND_CSV, index=False)
    print(f"[ground] wrote {len(rows)} rows to {GROUND_CSV}")


# =============================== STAGE: assemble =============================
def stage_assemble():
    """Merge all per-image CSVs + labels, normalise WITHIN this subset, build the
    three composite complexity scores and low/med/high groups."""
    ids   = load_subset_ids()[["image_id"]]
    hand  = pd.read_csv(HAND_CSV)
    vis   = pd.read_csv(VIS_CSV)
    sam   = pd.read_csv(SAM_CSV).drop_duplicates("image_id")
    grnd  = pd.read_csv(GROUND_CSV).drop_duplicates("image_id")
    lab   = pd.read_csv(LABELS_CSV)[["image_id", "CHAIRs_image", "CHAIRi_image"]].drop_duplicates("image_id")

    # Rename handcrafted columns so they don't collide with visual-statistics'
    # (both produce 'edge_density'; visual-statistics also has grayscale entropy
    # under a different name, so only 'edge_density' actually collides, but we
    # rename all four for clarity and future-proofing).
    hand = hand.rename(columns={
        "object_count":    "hand_object_count",
        "entropy":         "hand_entropy",
        "color_diversity": "hand_color_diversity",
        "edge_density":    "hand_edge_density",
    })

    df = ids.merge(hand, on="image_id").merge(vis, on="image_id") \
            .merge(sam, on="image_id").merge(grnd, on="image_id").merge(lab, on="image_id")
    print(f"[assemble] merged rows: {len(df)}")

    # ---- handcrafted score (4 features, minmax within subset, mean) ----
    hf = ["hand_object_count", "hand_entropy", "hand_color_diversity", "hand_edge_density"]
    for c in hf:
        df[c + "_hn"] = minmax(df[c])
    df["handcrafted_score"] = df[[c + "_hn" for c in hf]].mean(axis=1)

    # ---- visual-statistics score (11 features, minmax within subset, mean) ----
    vf = ["edge_density", "corner_density", "distinct_color_count", "color_entropy",
          "grayscale_entropy", "compression_ratio", "colorfulness", "contrast_std",
          "texture_entropy", "contour_density", "contour_length_density"]
    for c in vf:
        df[c + "_vn"] = minmax(df[c])
    df["vis_score"] = df[[c + "_vn" for c in vf]].mean(axis=1)

    # ---- SAM score (3 granularities, minmax within subset, mean) ----
    sf = [f"sam_segments_{p}" for p in POINTS_LIST]
    for c in sf:
        df[c + "_sn"] = minmax(df[c])
    df["sam_score"] = df[[c + "_sn" for c in sf]].mean(axis=1)

    # ---- low/med/high groups per measure ----
    for name, col in [("handcrafted", "handcrafted_score"),
                      ("vis", "vis_score"), ("sam", "sam_score")]:
        df[name + "_group"] = pd.qcut(df[col], q=3, labels=["low", "medium", "high"])

    df.to_csv(MASTER_CSV, index=False)
    print(f"[assemble] wrote {MASTER_CSV} ({len(df)} rows)")


# =============================== STAGE: diag =================================
def stage_diag():
    """Sanity check before mediation. Prints how correlated each complexity
    score is with each grounding variable (the mediator) and with hallucination.
    Use to decide whether SAM/grounding overlap enough to warrant a caveat."""
    df = pd.read_csv(MASTER_CSV)
    comp_cols = [("handcrafted", "handcrafted_score"),
                 ("vis",         "vis_score"),
                 ("sam",         "sam_score")]
    grnd_cols = [GROUNDING_PRIMARY, GROUNDING_ALT]
    outcome  = OUTCOME_COL

    print(f"\n==== DIAGNOSTICS: pairwise correlations (n={len(df)}) ====\n")

    print("Complexity <-> Grounding (Pearson r):")
    header = "  {:<12s}  " + "  ".join([f"{g:>22s}" for g in grnd_cols])
    print(header.format(""))
    for name, col in comp_cols:
        cells = []
        for g in grnd_cols:
            r = df[[col, g]].dropna().corr().iloc[0, 1]
            cells.append(f"r={r:+.3f}")
        print(f"  {name:<12s}  " + "  ".join([f"{c:>22s}" for c in cells]))

    print("\nComplexity <-> Hallucination (Pearson r):")
    for name, col in comp_cols:
        r = df[[col, outcome]].dropna().corr().iloc[0, 1]
        print(f"  {name:<12s}  vs {outcome}: r={r:+.3f}")

    print("\nGrounding <-> Hallucination (Pearson r):")
    for g in grnd_cols:
        r = df[[g, outcome]].dropna().corr().iloc[0, 1]
        print(f"  {g:<24s} vs {outcome}: r={r:+.3f}")

    print("\nRead per measure (complexity vs primary grounding):")
    for name, col in comp_cols:
        r = df[[col, GROUNDING_PRIMARY]].dropna().corr().iloc[0, 1]
        a = abs(r)
        if a < 0.30:
            verdict = "SAFE - measures are largely distinct; mediation is a clean test."
        elif a < 0.60:
            verdict = "MODERATE overlap - report mediation, flag shared variance."
        else:
            verdict = "HIGH overlap - flag as partly circular; interpret cautiously."
        print(f"  {name:<12s}  |r|={a:.3f}  =>  {verdict}")
    print()



MEASURES = [("handcrafted", "handcrafted_score"),
            ("vis", "vis_score"),
            ("sam", "sam_score")]

def stage_h1():
    import statsmodels.formula.api as smf
    df = pd.read_csv(MASTER_CSV)
    y = OUTCOME_COL
    print(f"\n==== H1: {y} vs complexity (n={len(df)}) ====")
    for name, col in MEASURES:
        sub = df[[col, y, name + "_group"]].dropna()
        grp = sub.groupby(name + "_group", observed=True)[y].mean()
        d = sub.rename(columns={col: "X", y: "Y"})
        fit = smf.ols("Y ~ X", d).fit()
        slope = fit.params["X"]; r2 = fit.rsquared; p = fit.pvalues["X"]
        print(f"\n-- {name} --")
        print("  group means (low/med/high):",
              [round(float(grp.get(g, np.nan)), 3) for g in ["low", "medium", "high"]])
        print(f"  regression slope={slope:+.4f}  R2={r2:.4f}  p={p:.3g}")


def _run_mediation(df, xcol, mcol, ycol, n_boot=1000, seed=SEED):
    import statsmodels.formula.api as smf
    d = df[[xcol, mcol, ycol]].dropna().copy()
    d["X"] = zscore(d[xcol]); d["M"] = zscore(d[mcol]); d["Y"] = d[ycol].astype(float)
    c   = smf.ols("Y ~ X", d).fit().params["X"]                    # total
    a   = smf.ols("M ~ X", d).fit().params["X"]                    # X -> M
    fit = smf.ols("Y ~ X + M", d).fit()
    cp  = fit.params["X"]                                          # direct (X | M)
    b   = fit.params["M"]                                          # M -> Y | X
    indirect = a * b
    prop = indirect / c if abs(c) > 1e-9 else np.nan
    # nonparametric bootstrap CI for the indirect effect
    rng = np.random.default_rng(seed)
    boots = []
    idx = np.arange(len(d))
    for _ in range(n_boot):
        s = d.iloc[rng.choice(idx, len(idx), replace=True)]
        try:
            a_b = smf.ols("M ~ X", s).fit().params["X"]
            b_b = smf.ols("Y ~ X + M", s).fit().params["M"]
            boots.append(a_b * b_b)
        except Exception:
            pass
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return dict(n=len(d), c=c, a=a, b=b, c_prime=cp,
                indirect=indirect, prop_mediated=prop, ci_lo=lo, ci_hi=hi)

def stage_mediation():
    df = pd.read_csv(MASTER_CSV)
    print(f"\n==== MEDIATION: complexity -> {GROUNDING_PRIMARY} -> {OUTCOME_COL} ====")
    print("(X and M z-scored; observational - report as 'consistent with mediation')\n")
    for name, col in MEASURES:
        for mcol in [GROUNDING_PRIMARY, GROUNDING_ALT]:
            res = _run_mediation(df, col, mcol, OUTCOME_COL)
            print(f"-- {name}  (M = {mcol}) --  n={res['n']}")
            print(f"   total c      = {res['c']:+.4f}")
            print(f"   a (X->M)     = {res['a']:+.4f}")
            print(f"   b (M->Y|X)   = {res['b']:+.4f}")
            print(f"   direct c'    = {res['c_prime']:+.4f}")
            print(f"   indirect a*b = {res['indirect']:+.4f}  95% CI [{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}]")
            print(f"   prop mediated= {res['prop_mediated']:.3f}\n")


# =============================== main ========================================
STAGES = {
    "freeze": stage_freeze, "hand": stage_hand, "vis": stage_vis, "sam": stage_sam,
    "sam_merge": stage_sam_merge,
    "ground": stage_ground, "assemble": stage_assemble, "diag": stage_diag,
    "h1": stage_h1, "mediation": stage_mediation,
}

def main():
    ap = argparse.ArgumentParser(
        description="Mediation pipeline: complexity -> grounding -> hallucination"
    )
    ap.add_argument("--stage", required=True,
                    choices=list(STAGES.keys()) + ["all_analysis"],
                    help="Pipeline stage to run.")
    ap.add_argument("--labels", default=DEFAULT_LABELS,
                    help="Path to labels CSV with columns: image_id, CHAIRs_image, "
                         "CHAIRi_image, caption (default: %(default)s).")
    ap.add_argument("--images", default=DEFAULT_IMAGES,
                    help="Path to image directory, e.g. val2014/ (default: %(default)s).")
    ap.add_argument("--instances", default=DEFAULT_INSTANCES,
                    help="Path to COCO instances JSON (default: %(default)s).")
    ap.add_argument("--sam-checkpoint", default=DEFAULT_SAM_CKPT,
                    help="Path to SAM ViT-H checkpoint (default: %(default)s).")
    ap.add_argument("--output", default=DEFAULT_OUT_DIR,
                    help="Output directory for all pipeline CSVs (default: %(default)s).")
    ap.add_argument("--subset-n", type=int, default=DEFAULT_SUBSET_N,
                    help="Number of images in the subset (default: %(default)s).")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="Random seed for subset selection (default: %(default)s).")
    ap.add_argument("--chunk-index", type=int, default=0,
                    help="For --stage sam: this machine's chunk index, 0-based.")
    ap.add_argument("--chunk-total", type=int, default=1,
                    help="For --stage sam: total number of chunks (machines).")
    args = ap.parse_args()

    _init_paths(args)

    if args.stage == "all_analysis":
        stage_assemble(); stage_diag(); stage_h1(); stage_mediation()
    elif args.stage == "sam":
        stage_sam(chunk_index=args.chunk_index, chunk_total=args.chunk_total)
    else:
        STAGES[args.stage]()

if __name__ == "__main__":
    main()
