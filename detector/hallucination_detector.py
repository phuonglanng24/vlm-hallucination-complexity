"""
Interpretable hallucination detector (XGBoost).

Extracts features from (image, caption) pairs and trains a gradient-boosted
classifier to predict whether a caption hallucinates (CHAIR label). Reports
test metrics (ROC-AUC, accuracy, precision, recall, F1, confusion matrix),
feature importances, and per-feature correlations with the label.

Features are organised in families:
    Object-level grounding (CLIP): min/mean object grounding, num/frac weak
        objects, grounding gap
    Caption structure (text only): word count, num objects, num count words
    Pixel statistics (optional): edge density, texture entropy, colour
        diversity, brightness mean/std
    Count detection (optional, DETR): count error sum/max, num overclaimed,
        num missing objects

Select active features by editing USE_FEATURES below.

Requirements:
    pip install xgboost scikit-learn pandas numpy pillow
    pip install torch transformers     # for CLIP and DETR features

Usage:
    python hallucination_detector.py --images val2014 --json captions.jsonl --csv labels.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ======================= CHOOSE YOUR FEATURES HERE =======================
USE_FEATURES = [
    # --- Object-level grounding (CLIP) ---
    "min_object_grounding",
    "mean_object_grounding",
    "num_weak_objects",
    "frac_weak_objects",
    "grounding_gap",
    # --- Caption structure (text only) ---
    "num_objects",
    "word_count",
    "num_count_words",
    # --- Count detection (DETR, optional) ---
    # "count_error_sum",
    # "count_error_max",
    # "num_overclaimed",
    # "num_missing_objects",
    # --- Pixel statistics (optional) ---
    # "edge_density",
    # "texture_entropy",
    # "color_diversity",
    # "brightness_mean",
    # "brightness_std",
]
LABEL_COLUMN = "CHAIRs_image"
WEAK_OBJECT_THRESHOLD = 0.22   # CLIP grounding below this = "weakly grounded" object
SEED = 42
# =========================================================================

COCO_OBJECTS = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
}
SPATIAL_WORDS = {"left", "right", "background", "foreground", "center", "centre",
                 "behind", "front", "top", "bottom", "side", "middle", "above",
                 "below", "near", "beside", "corner"}
COUNT_WORDS = {"two", "three", "four", "five", "six", "seven", "eight", "nine",
               "ten", "several", "few", "many", "group", "herd", "pair", "couple",
               "multiple", "numerous", "some"}
HEDGE_WORDS = {"appears", "appear", "seems", "seem", "possibly", "likely", "might",
               "perhaps", "probably", "maybe", "could", "suggesting", "seemingly"}


# Map count words -> a number, for parsing how many of an object a caption claims.
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "a": 1, "an": 1, "single": 1,
    "couple": 2, "pair": 2, "both": 2, "few": 3, "several": 3, "some": 3,
    "many": 4, "multiple": 4, "numerous": 4, "group": 3, "herd": 4, "flock": 4,
    "bunch": 3,
}
_NUM_ALT = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True)) + r"|\d+"


def claimed_count(text_low: str, obj: str) -> int:
    """Best guess of how many of `obj` the caption claims.
    'two dogs' -> 2, 'a cat' -> 1, plural with no number -> 2, else 1."""
    pat = r"\b(" + _NUM_ALT + r")\b(?=(?:\s+\w+){0,2}\s+" + re.escape(obj) + r"s?\b)"
    counts = []
    for m in re.findall(pat, text_low):
        counts.append(int(m) if m.isdigit() else NUMBER_WORDS.get(m, 1))
    if counts:
        return min(max(counts), 20)          # cap to avoid weird captions
    if re.search(r"\b" + re.escape(obj) + r"s\b", text_low):
        return 2                             # plural, no explicit number
    return 1


# ----------------------------- text features -----------------------------
def mentioned_objects(text_low: str):
    """COCO object words present in the caption (singular or simple plural)."""
    return [o for o in COCO_OBJECTS
            if re.search(r"\b" + re.escape(o) + r"s?\b", text_low)]


def text_features(caption: str, objs):
    low = str(caption).lower()
    words = re.findall(r"[a-zA-Z']+", low)
    return {
        "word_count": len(words),
        "num_objects": len(objs),
        "num_count_words": sum(w in COUNT_WORDS for w in words),
        "num_spatial_words": sum(w in SPATIAL_WORDS for w in words),
        "num_hedge_words": sum(w in HEDGE_WORDS for w in words),
    }


# ----------------------------- image pixel features -----------------------------
def image_pixel_features(paths, size=224):
    """5 transparent pixel statistics per image (no pretrained model)."""
    from PIL import Image
    out = []
    for k, p in enumerate(paths):
        arr = np.asarray(Image.open(p).convert("RGB").resize((size, size)),
                         dtype=np.float32) / 255.0
        gray = arr.mean(axis=2)
        gy, gx = np.gradient(gray)
        edge_density = float((np.sqrt(gx ** 2 + gy ** 2) > 0.08).mean())
        hist, _ = np.histogram(gray, bins=32, range=(0, 1))
        pr = hist / (hist.sum() + 1e-9)
        texture_entropy = float(-(pr[pr > 0] * np.log2(pr[pr > 0])).sum())
        q = np.clip((arr * 4).astype(int), 0, 3)
        codes = q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]
        color_diversity = float(len(np.unique(codes)) / 64.0)
        out.append([edge_density, texture_entropy, color_diversity,
                    float(gray.mean()), float(gray.std())])
        print(f"    pixels {k + 1}/{len(paths)}", end="\r")
    print()
    cols = ["edge_density", "texture_entropy", "color_diversity",
            "brightness_mean", "brightness_std"]
    return pd.DataFrame(out, columns=cols)


# ----------------------------- CLIP features -----------------------------
def clip_features(image_paths, texts, objs_per_ex, batch_size=32,
                  weak_threshold=WEAK_OBJECT_THRESHOLD):
    """clip_similarity + min_object_grounding + num_weak_objects."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if getattr(torch.backends, "mps", None) is not None
              and torch.backends.mps.is_available() else "cpu")
    print(f"  CLIP device: {device}")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def norm(t):
        return t / t.norm(dim=1, keepdim=True)

    # Project into CLIP's shared space directly from the submodules. This is
    # exactly what get_text_features / get_image_features do, but it is robust
    # across transformers versions (some return a wrapper object instead of a
    # tensor, which broke the convenience methods).
    def embed_text(inputs):
        return model.text_projection(model.text_model(**inputs).pooler_output)

    def embed_image(inputs):
        return model.visual_projection(model.vision_model(**inputs).pooler_output)

    obj_list = sorted(COCO_OBJECTS)
    obj_index = {o: i for i, o in enumerate(obj_list)}
    with torch.no_grad():
        oi = proc(text=[f"a photo of a {o}" for o in obj_list],
                  return_tensors="pt", padding=True).to(device)
        obj_emb = norm(embed_text(oi))                     # (80, 512)

    sims, min_g, mean_g, n_weak, frac_weak, gap = [], [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            imgs = [Image.open(p).convert("RGB") for p in image_paths[i:i + batch_size]]
            txts = texts[i:i + batch_size]
            img_emb = norm(embed_image(
                proc(images=imgs, return_tensors="pt").to(device)))
            txt_emb = norm(embed_text(
                proc(text=txts, return_tensors="pt", padding=True, truncation=True).to(device)))
            batch_sim = (img_emb * txt_emb).sum(1).cpu().numpy()
            grounding = (img_emb @ obj_emb.T).cpu().numpy()   # (b, 80)
            for j, objs in enumerate(objs_per_ex[i:i + batch_size]):
                s = float(batch_sim[j])
                if objs:
                    scores = [grounding[j, obj_index[o]] for o in objs]
                    mn = float(min(scores))
                    weak = int(sum(x < weak_threshold for x in scores))
                    mean_g.append(float(np.mean(scores)))
                    frac_weak.append(weak / len(objs))
                else:
                    mn = s            # no objects named -> fall back to whole-caption sim
                    weak = 0
                    mean_g.append(s)
                    frac_weak.append(0.0)
                sims.append(s)
                min_g.append(mn)
                n_weak.append(weak)
                gap.append(s - mn)    # how far the worst object falls below overall match
            print(f"    clip {min(i + batch_size, len(image_paths))}/{len(image_paths)}", end="\r")
    print()
    return pd.DataFrame({
        "clip_similarity": sims,
        "min_object_grounding": min_g,
        "mean_object_grounding": mean_g,
        "num_weak_objects": np.array(n_weak, dtype=float),
        "frac_weak_objects": frac_weak,
        "grounding_gap": gap,
    })


# ----------------------------- object detector (count errors) -----------------------------
def detector_count_features(image_paths, texts, objs_per_ex, score_threshold=0.7, batch_size=8):
    """Run an object DETECTOR (DETR) that actually counts objects in each image,
    then compare those counts to what the caption claims. Catches count errors
    ('two dogs' when there's one) that CLIP grounding cannot.

    Features:
      count_error_sum     - total |claimed - detected| over mentioned objects
      count_error_max     - worst single count discrepancy
      num_overclaimed     - objects the caption claims MORE of than were detected
      num_missing_objects - mentioned objects the detector found ZERO of (invention)
    """
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if getattr(torch.backends, "mps", None) is not None
              and torch.backends.mps.is_available() else "cpu")
    print(f"  detector device: {device}")
    proc = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device).eval()
    id2label = model.config.id2label

    err_sum, err_max, overclaim, missing = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            imgs = [Image.open(p).convert("RGB") for p in image_paths[i:i + batch_size]]
            inputs = proc(images=imgs, return_tensors="pt").to(device)
            outputs = model(**inputs)
            sizes = torch.tensor([im.size[::-1] for im in imgs]).to(device)   # (h, w)
            results = proc.post_process_object_detection(
                outputs, target_sizes=sizes, threshold=score_threshold)
            for j, res in enumerate(results):
                detected = {}
                for lab in res["labels"].cpu().numpy():
                    name = str(id2label[int(lab)]).lower()
                    detected[name] = detected.get(name, 0) + 1
                errs, ov, ms = [], 0, 0
                for obj in objs_per_ex[i + j]:
                    c = claimed_count(str(texts[i + j]).lower(), obj)
                    d = detected.get(obj, 0)
                    errs.append(abs(c - d))
                    if c > d:
                        ov += 1
                    if d == 0:
                        ms += 1
                err_sum.append(float(sum(errs)))
                err_max.append(float(max(errs)) if errs else 0.0)
                overclaim.append(float(ov))
                missing.append(float(ms))
            print(f"    detector {min(i + batch_size, len(image_paths))}/{len(image_paths)}", end="\r")
    print()
    return pd.DataFrame({
        "count_error_sum": err_sum,
        "count_error_max": err_max,
        "num_overclaimed": overclaim,
        "num_missing_objects": missing,
    })


# ----------------------------- data loading -----------------------------
def coco_num_id(s):
    d = re.findall(r"\d+", os.path.basename(str(s)))
    return int(max(d, key=len)) if d else None


def to_binary(v):
    return 1 if str(v).strip().lower() in {"1", "1.0", "yes", "y", "true", "t"} else 0


def load_joined_data(images_dir, json_path, csv_path, label_col=LABEL_COLUMN):
    df = pd.read_csv(csv_path, sep=None, engine="python")
    label_map = {coco_num_id(r["image_id"]): to_binary(r[label_col])
                 for _, r in df.iterrows() if not pd.isna(r.get(label_col))}
    json_map = {}
    with open(json_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = coco_num_id(obj.get("image_id", ""))
            if iid is not None and "text" in obj:
                json_map[iid] = (obj["image_id"], obj["text"])

    paths, texts, labels, missing = [], [], [], 0
    for iid, (fname, text) in json_map.items():
        if iid not in label_map:
            continue
        p = os.path.join(images_dir, os.path.basename(str(fname)))
        if not os.path.exists(p):
            missing += 1
            continue
        paths.append(p); texts.append(text); labels.append(label_map[iid])
    if missing:
        print(f"  warning: {missing} captioned images not found on disk (skipped).")
    if not paths:
        raise SystemExit("No examples after joining. Check paths / ids / columns.")
    return paths, texts, np.array(labels, int)


def split_5_2_2(labels, seed=SEED):
    idx = np.arange(len(labels))
    idx_tv, idx_te = train_test_split(idx, test_size=2/9, stratify=labels, random_state=seed)
    idx_tr, idx_val = train_test_split(idx_tv, test_size=2/7, stratify=labels[idx_tv], random_state=seed)
    return idx_tr, idx_val, idx_te


# ----------------------------- feature assembly -----------------------------
def build_features(paths, texts, use_features, batch_size=32):
    objs_per_ex = [mentioned_objects(str(t).lower()) for t in texts]
    parts = [pd.DataFrame([text_features(t, o) for t, o in zip(texts, objs_per_ex)])]

    if any(f in use_features for f in ["edge_density", "texture_entropy",
                                       "color_diversity", "brightness_mean", "brightness_std"]):
        print("  computing pixel features...")
        parts.append(image_pixel_features(paths))

    clip_feats = ["clip_similarity", "min_object_grounding", "mean_object_grounding",
                  "num_weak_objects", "frac_weak_objects", "grounding_gap"]
    if any(f in use_features for f in clip_feats):
        try:
            print("  computing CLIP features...")
            parts.append(clip_features(paths, texts, objs_per_ex, batch_size))
        except ImportError:
            print("  note: torch/transformers not installed -> skipping CLIP features.")

    det_feats = ["count_error_sum", "count_error_max", "num_overclaimed", "num_missing_objects"]
    if any(f in use_features for f in det_feats):
        try:
            print("  computing object-detector count features (DETR)...")
            parts.append(detector_count_features(paths, texts, objs_per_ex))
        except ImportError:
            print("  note: torch/transformers not installed -> skipping detector features.")

    df = pd.concat(parts, axis=1)
    cols = [f for f in use_features if f in df.columns]
    return df[cols]


# ----------------------------- train + evaluate -----------------------------
def train_and_evaluate(X, labels, idx_tr, idx_val, idx_te, feat_df):
    y_tr, y_val, y_te = labels[idx_tr], labels[idx_val], labels[idx_te]
    n_pos = max(int((y_tr == 1).sum()), 1)
    n_neg = max(int((y_tr == 0).sum()), 1)
    clf = XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="auc",
        early_stopping_rounds=50, scale_pos_weight=n_neg / n_pos,
        tree_method="hist", importance_type="gain", n_jobs=-1, random_state=SEED,
    )
    clf.fit(X[idx_tr], y_tr, eval_set=[(X[idx_val], y_val)], verbose=False)
    print(f"  best iteration (early stopping): {clf.best_iteration}")

    prob = clf.predict_proba(X[idx_te])[:, 1]
    pred = (prob >= 0.5).astype(int)

    print("\n================ TEST METRICS ================")
    if len(np.unique(y_te)) > 1:
        print(f"  ROC-AUC  : {roc_auc_score(y_te, prob):.4f}")
    print(f"  Accuracy : {accuracy_score(y_te, pred):.4f}")
    print(f"  Precision: {precision_score(y_te, pred, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(y_te, pred, zero_division=0):.4f}")
    print(f"  F1       : {f1_score(y_te, pred, zero_division=0):.4f}")
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    print("\n  Confusion matrix (rows=true, cols=pred):")
    print("                 pred faithful   pred hallucinated")
    print(f"    faithful          {tn:>6d}            {fp:>6d}")
    print(f"    hallucinated      {fn:>6d}            {tp:>6d}")

    names = list(feat_df.columns)

    # SUMMARY 1: which features the model used most
    print("\n  [1] FEATURE IMPORTANCE (which features the model relied on):")
    for name, val in sorted(zip(names, clf.feature_importances_),
                            key=lambda t: t[1], reverse=True):
        print(f"      {name:<22s} {val:.4f}")

    # SUMMARY 2: which way each feature relates to hallucination
    print("\n  [2] RELATIONSHIP WITH HALLUCINATION (correlation with the label):")
    rels = []
    for col in names:
        x = feat_df[col].values.astype(float)
        r = 0.0 if np.std(x) == 0 else float(np.corrcoef(x, labels)[0, 1])
        rels.append((col, r))
    for col, r in sorted(rels, key=lambda t: abs(t[1]), reverse=True):
        direction = "MORE" if r > 0 else "LESS"
        strength = ("strong" if abs(r) >= 0.30 else "moderate" if abs(r) >= 0.15
                    else "weak" if abs(r) >= 0.05 else "negligible")
        print(f"      {col:<22s} corr={r:+.3f}  (higher -> {direction} hallucination, {strength})")
    return clf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    print("[1/4] Joining captions + labels + images...")
    paths, texts, labels = load_joined_data(args.images, args.json, args.csv)
    print(f"  {len(paths)} examples ({int(labels.sum())} hallucinated / {int((labels==0).sum())} faithful)")

    print("[2/4] Splitting 5:2:2 (stratified)...")
    idx_tr, idx_val, idx_te = split_5_2_2(labels)
    print(f"  train={len(idx_tr)} val={len(idx_val)} test={len(idx_te)}")

    print("[3/4] Building 13 meaningful features...")
    feat_df = build_features(paths, texts, USE_FEATURES, args.batch_size)
    print(f"  features used ({feat_df.shape[1]}): {list(feat_df.columns)}")
    X = feat_df.values.astype(np.float32)

    print("[4/4] Training XGBoost...")
    train_and_evaluate(X, labels, idx_tr, idx_val, idx_te, feat_df)


if __name__ == "__main__":
    main()
