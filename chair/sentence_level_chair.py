#!/usr/bin/env python3
"""
Sentence-level CHAIR evaluation.

Standard CHAIR scores each caption as a unit. This script enables a finer-grained
evaluation by splitting each caption into individual sentences, evaluating each
sentence separately with CHAIR, and then aggregating back to per-image metrics.

Two modes:

  1. --mode split
     Takes a captions JSONL (one caption per image) and splits each caption into
     sentences using NLTK. Produces a new JSONL where each line is one sentence,
     tagged with the original image_id and a sentence index.

  2. --mode aggregate
     Takes the CHAIR output JSON produced by running chair.py on the split file,
     and aggregates per-sentence results back to per-image sentence-level metrics:
       CHAIRs_image  = fraction of sentences containing any hallucinated object
       CHAIRi_image  = fraction of all mentioned objects (across sentences) that
                       are hallucinated
       Recall_image, Precision_image, F1_image = object-level grounding metrics
       num_sentences = number of sentences in the original caption

Workflow:
    # 1. Split captions into sentences
    python sentence_level_chair.py --mode split \\
        --input captions_for_chair.jsonl \\
        --output sentence_captions.jsonl

    # 2. Run standard CHAIR on the sentence-level file
    python chair.py --cap_file sentence_captions.jsonl --caption_key text \\
        --coco_path ./annotations --save_path sentence_chair_output.json

    # 3. Aggregate back to per-image metrics
    python sentence_level_chair.py --mode aggregate \\
        --input sentence_chair_output.json \\
        --output sentence_level_metrics.csv

Requirements:
    pip install nltk pandas
    # On first run, NLTK will download 'punkt' and 'punkt_tab' tokenizer data.
"""

import argparse
import json
from pathlib import Path

import nltk
import pandas as pd

# Ensure NLTK sentence tokenizer is available
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


def split_captions(input_path: str, output_path: str, caption_key: str = "text"):
    """Split each caption into sentences and write a new JSONL file.

    Each output line has:
        image_id       — same as the original (integer)
        text           — one sentence
        sentence_idx   — 0-based index within the original caption
        original_text  — the full original caption (for reference)
    """
    out_lines = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            image_id = entry["image_id"]
            caption = entry.get(caption_key, "")
            sentences = nltk.sent_tokenize(caption)
            if not sentences:
                sentences = [caption]
            for idx, sent in enumerate(sentences):
                sent = sent.strip()
                if not sent:
                    continue
                out_lines.append({
                    "image_id": image_id,
                    "text": sent,
                    "sentence_idx": idx,
                    "original_text": caption,
                })

    with open(output_path, "w") as f:
        for entry in out_lines:
            f.write(json.dumps(entry) + "\n")

    n_images = len(set(e["image_id"] for e in out_lines))
    print(f"[split] {n_images} images -> {len(out_lines)} sentences")
    print(f"[split] wrote {output_path}")


def aggregate_results(input_path: str, output_path: str):
    """Aggregate per-sentence CHAIR results back to per-image metrics.

    Reads the CHAIR output JSON (which was run on the sentence-level JSONL)
    and computes per-image:
        CHAIRs_image  — fraction of sentences with any hallucination
        CHAIRi_image  — fraction of mentioned objects that are hallucinated
        Recall_image  — fraction of GT objects correctly mentioned
        Precision_image — fraction of mentioned objects that are correct
        F1_image      — harmonic mean of recall and precision
        num_sentences — number of sentences in the original caption
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    sentences = data.get("sentences", [])
    if not sentences:
        print(f"[aggregate] WARNING: no 'sentences' key in {input_path}")
        return

    # Group sentences by image_id
    from collections import defaultdict
    by_image = defaultdict(list)
    for sent in sentences:
        image_id = sent["image_id"]
        by_image[image_id].append(sent)

    rows = []
    for image_id, sents in by_image.items():
        num_sentences = len(sents)

        # CHAIRs: fraction of sentences containing any hallucinated object
        n_hall_sentences = sum(
            1 for s in sents if s.get("metrics", {}).get("CHAIRs", 0) > 0
        )
        chairs_image = n_hall_sentences / num_sentences if num_sentences > 0 else 0.0

        # CHAIRi: fraction of all mentioned objects that are hallucinated
        total_objects = 0
        total_hallucinated = 0
        for s in sents:
            m = s.get("metrics", {})
            mentioned = m.get("num_mentioned_objects", 0)
            hallucinated = m.get("num_hallucinated_objects", 0)
            total_objects += mentioned
            total_hallucinated += hallucinated
        chairi_image = (
            total_hallucinated / total_objects if total_objects > 0 else 0.0
        )

        # Object-level grounding: recall, precision, F1
        total_gt = 0
        total_correct = 0
        for s in sents:
            m = s.get("metrics", {})
            gt = m.get("num_gt_objects", 0)
            correct = m.get("num_correct_objects", 0)
            total_gt += gt
            total_correct += correct

        recall = total_correct / total_gt if total_gt > 0 else 0.0
        precision = total_correct / total_objects if total_objects > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        rows.append({
            "image_id": image_id,
            "num_sentences": num_sentences,
            "CHAIRs_image": round(chairs_image, 4),
            "CHAIRi_image": round(chairi_image, 4),
            "Recall_image": round(recall, 4),
            "Precision_image": round(precision, 4),
            "F1_image": round(f1, 4),
        })

    df = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[aggregate] {len(df)} images -> {output_path}")
    print(f"  Mean CHAIRs_image: {df['CHAIRs_image'].mean():.3f}")
    print(f"  Mean CHAIRi_image: {df['CHAIRi_image'].mean():.3f}")
    print(f"  Mean F1_image:     {df['F1_image'].mean():.3f}")
    print(f"  Mean sentences:    {df['num_sentences'].mean():.1f}")


def main():
    ap = argparse.ArgumentParser(
        description="Sentence-level CHAIR evaluation: split captions into sentences "
                    "and aggregate per-sentence CHAIR results back to per-image metrics."
    )
    ap.add_argument("--mode", required=True, choices=["split", "aggregate"],
                    help="'split' to break captions into sentences; "
                         "'aggregate' to combine per-sentence CHAIR results.")
    ap.add_argument("--input", required=True,
                    help="Input file: captions JSONL (for split) or CHAIR output JSON "
                         "(for aggregate).")
    ap.add_argument("--output", required=True,
                    help="Output file: sentence JSONL (for split) or per-image CSV "
                         "(for aggregate).")
    ap.add_argument("--caption-key", default="text",
                    help="Key for the caption text in the input JSONL (default: text).")
    args = ap.parse_args()

    if args.mode == "split":
        split_captions(args.input, args.output, caption_key=args.caption_key)
    elif args.mode == "aggregate":
        aggregate_results(args.input, args.output)


if __name__ == "__main__":
    main()
