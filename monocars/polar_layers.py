"""Batch: every polarization product for the dataset frames -> recorder layers.

Reads the raw frames straight out of the recording tars (raw_index.json maps
"<cam>_<stem>" -> [tar, tar_prefix_offset, data_offset, size]), computes the
product catalog (polar_normals.products) at half resolution in sensor
orientation (same as rgb/), and writes
    <scene>/layers/<product>/<cam>_<stem>.jpg
Resumable: frames that already have every product are skipped.

Usage: python polar_layers.py <scene_dir> <raw_index.json> [workers=6] [products=all|a,b,c] [limit=0]
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SCENE = Path(sys.argv[1]); INDEX = Path(sys.argv[2])
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
PRODS = sys.argv[4] if len(sys.argv) > 4 else "all"
LIMIT = int(sys.argv[5]) if len(sys.argv) > 5 else 0

# LOCKED catalog (2026-08-27 client naming): nxyz, n_xy, n_xz, phys, diffuse,
# specv2, edge + rgb_deglare. No scalar polarization views in the UI.
LAYER_SET = ["nxyz", "n_xy", "n_xz", "nxyz_phys", "nxyz_diffuse", "nxyz_specv2",
             "edge", "rgb_deglare"]
RENAME = {}


def read_raw(entry):
    tar, pre, off, size = entry
    with open(tar, "rb") as f:
        f.seek(pre + off)
        return f.read(size)


def work(job):
    name, entry, wanted = job
    try:
        from polar_normals import PolarFrame, products
        fr = PolarFrame(read_raw(entry))
        P = products(fr)
        written = 0
        for k, img in P.items():
            lk = RENAME.get(k, k)
            if lk not in wanted:
                continue
            d = SCENE / "layers" / lk
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            written += 1
        return name, written, None
    except Exception as e:
        return name, 0, repr(e)


if __name__ == "__main__":
    idx = json.loads(INDEX.read_text())
    frames = sorted(p.stem for p in (SCENE / "rgb").glob("*.jpg"))
    wanted = LAYER_SET if PRODS == "all" else PRODS.split(",")
    jobs = []
    for name in frames:
        if name not in idx:
            continue
        done = all((SCENE / "layers" / k / f"{name}.jpg").exists() for k in wanted)
        if done:
            continue
        jobs.append((name, idx[name], wanted))
    if LIMIT:
        jobs = jobs[:LIMIT]
    print(f"{len(frames)} frames, {len(jobs)} to do, {len(wanted)} products, {WORKERS} workers", flush=True)
    t0 = time.time(); n_ok = 0; n_err = 0
    with Pool(WORKERS) as pool:
        for i, (name, w, err) in enumerate(pool.imap_unordered(work, jobs, chunksize=2), 1):
            if err:
                n_err += 1; print("ERR", name, err, flush=True)
            else:
                n_ok += 1
            if i % 25 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"{i}/{len(jobs)} ok={n_ok} err={n_err} {el:.0f}s ({el / max(i, 1):.1f}s/frame)", flush=True)
    print("POLAR_LAYERS_DONE", n_ok, n_err, flush=True)
