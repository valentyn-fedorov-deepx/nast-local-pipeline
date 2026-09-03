"""Extract the RAW12 polarization frames for a scene out of the recording
tars into <scene>/raw/<stem>.raw12 — the store the live normals decoder
(server /frames/live/...) reads. Ship rgb/ + raw/ and every normals product
decodes on demand; no offline layer pass needed.

Usage: python extract_raw.py <scene_dir> <raw_index.json>
"""
import json
import sys
from pathlib import Path

SCENE = Path(sys.argv[1]); INDEX = Path(sys.argv[2])
OUT = SCENE / "raw"
OUT.mkdir(exist_ok=True)
idx = json.loads(INDEX.read_text())
stems = sorted(p.stem for p in (SCENE / "rgb").glob("*.jpg"))
done = 0
for s in stems:
    if s not in idx:
        continue
    dst = OUT / f"{s}.raw12"
    if dst.exists():
        done += 1
        continue
    tar, pre, off, size = idx[s]
    with open(tar, "rb") as f:
        f.seek(pre + off)
        dst.write_bytes(f.read(size))
    done += 1
    if done % 200 == 0:
        print(f"{done}/{len(stems)}", flush=True)
print(f"EXTRACT_RAW_DONE {done} frames -> {OUT}", flush=True)
