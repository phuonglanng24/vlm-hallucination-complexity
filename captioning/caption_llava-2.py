"""
Generate LLaVA-1.5 captions for a folder of images.

Model:  llava-hf/llava-1.5-7b-hf (bf16, Flash-Attention-2 if available)
Output: one JSON object per line (JSONL):
        {"image_id": "COCO_val2014_000000086848.jpg",
         "text": "<the caption>",
         "prompt": "Describe this image in detail."}

Crash-safe: each caption is appended and flushed immediately. On restart the
script skips images already present in the output file.

Requirements:
    pip install torch transformers accelerate pillow
    # Optional speedup (needs CUDA toolchain; falls back to SDPA if absent):
    pip install flash-attn --no-build-isolation

Usage:
    # Test on 20 images first
    python caption_llava.py --images ./val2014 --out captions.jsonl --limit 20
    # Full run
    python caption_llava.py --images ./val2014 --out captions.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png")
MODEL_ID = "llava-hf/llava-1.5-7b-hf"


def load_model():
    """Load LLaVA in bf16, preferring Flash-Attention-2, falling back to SDPA."""
    for attn in ("flash_attention_2", "sdpa"):
        try:
            model = LlavaForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn,
                low_cpu_mem_usage=True,
            ).to("cuda").eval()
            print(f"  loaded {MODEL_ID} with attn_implementation={attn}")
            return model
        except (ImportError, ValueError, RuntimeError) as e:
            print(f"  {attn} unavailable ({type(e).__name__}: {e}); trying next...")
    raise RuntimeError("Could not load the model with any attention implementation.")


def already_done(out_path):
    """Image ids already captioned (so we can resume)."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["image_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder of images")
    ap.add_argument("--out", default="captions.jsonl")
    ap.add_argument("--prompt", default="Describe this image in detail.",
                    help="captioning prompt (default: %(default)s)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="cap #images (0 = all); use for a test run")
    args = ap.parse_args()

    # gather images
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(args.images, ext)))
    files = sorted(files)
    if not files:
        raise SystemExit(f"No images found in {args.images}")

    done = already_done(args.out)
    todo = [f for f in files if os.path.basename(f) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"Found {len(files)} images; {len(done)} already done; {len(todo)} to caption.")
    if not todo:
        print("Nothing to do."); return

    print("Loading model...")
    model = load_model()
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
    processor.tokenizer.padding_side = "left"   # required for correct batched generation

    # build the prompt once via the chat template
    conversation = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": args.prompt}]}]
    prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)

    start = time.time()
    n_done = 0
    with open(args.out, "a", encoding="utf-8") as out_f:
        for i in range(0, len(todo), args.batch_size):
            batch_paths = todo[i:i + args.batch_size]
            try:
                results = caption_batch(model, processor, prompt_text, batch_paths,
                                        args.max_new_tokens)
            except Exception as e:
                # a bad image shouldn't kill the run: fall back to one-by-one
                print(f"  batch failed ({type(e).__name__}: {e}); retrying individually...")
                results = []
                for p in batch_paths:
                    try:
                        results += caption_batch(model, processor, prompt_text, [p],
                                                 args.max_new_tokens)
                    except Exception as e2:
                        print(f"    skipping {os.path.basename(p)} ({type(e2).__name__})")

            for path, text in results:
                out_f.write(json.dumps({
                    "image_id": os.path.basename(path),
                    "text": text.strip(),
                    "prompt": args.prompt,
                }, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())   # force to disk so a crash loses nothing

            n_done += len(results)
            rate = n_done / max(time.time() - start, 1e-6)
            remaining = (len(todo) - n_done) / max(rate, 1e-6)
            print(f"  {n_done}/{len(todo)}  |  {rate:.2f} img/s  |  "
                  f"ETA {remaining/60:.1f} min", end="\r")
    print(f"\nDone. Wrote captions for {n_done} images to {args.out}")


def caption_batch(model, processor, prompt_text, paths, max_new_tokens):
    """Caption a list of image paths; returns list of (path, caption_text)."""
    images = [Image.open(p).convert("RGB") for p in paths]
    prompts = [prompt_text] * len(images)
    inputs = processor(images=images, text=prompts, return_tensors="pt",
                       padding=True).to(model.device, torch.bfloat16)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # keep only the newly generated tokens (strip the prompt)
    gen = out[:, inputs["input_ids"].shape[1]:]
    texts = processor.batch_decode(gen, skip_special_tokens=True)
    return list(zip(paths, texts))


if __name__ == "__main__":
    main()
