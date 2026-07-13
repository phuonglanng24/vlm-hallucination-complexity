"""Convert image_id from filename to integer so CHAIR can match annotations.
Usage:  python prep_for_chair.py captions_in_progress.jsonl captions_for_chair.jsonl
"""
import json
import re
import sys

inp, outp = sys.argv[1], sys.argv[2]
n = 0
with open(inp, encoding="utf-8") as f, open(outp, "w", encoding="utf-8") as g:
    for line in f:
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        m = re.search(r"(\d+)\.jpg", str(o["image_id"]))
        if m:
            o["image_id"] = int(m.group(1))   # 'COCO_val2014_000000000042.jpg' -> 42
        g.write(json.dumps(o, ensure_ascii=False) + "\n")
        n += 1
print(f"wrote {n} lines to {outp} with integer image_id")
