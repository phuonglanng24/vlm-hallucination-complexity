"""Convert CHAIR's output JSON into labels.csv for the feature pipeline.
Usage:  python chair_to_labels.py chair_output.json labels.csv
"""
import csv
import json
import sys

inp, outp = sys.argv[1], sys.argv[2]
data = json.load(open(inp, encoding="utf-8"))
sentences = data["sentences"]

with open(outp, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["image_id", "CHAIRs_image", "CHAIRi_image", "caption"])
    pos = 0
    for s in sentences:
        chairs = int(s["metrics"]["CHAIRs"])
        pos += chairs
        w.writerow([s["image_id"], chairs,
                    round(float(s["metrics"].get("CHAIRi", 0.0)), 4),
                    s.get("caption", "")])

print(f"wrote {len(sentences)} rows to {outp}  "
      f"({pos} hallucinated / {len(sentences) - pos} faithful)")
