"""Regenerate the working colour stream: recorder rgb x glare attenuation.

street_video/rgb = rgb_orig (the recorder's own look, untouched tone/detail)
multiplied by PolarFrame.deglare_atten(): the Stokes-minimum luma ratio,
floored at 0.35 and lightly smoothed. Specular sheen (glass, roofs, wet road)
is dimmed physically; everything else is byte-identical to the recorder.

Usage: python gen_rgb_soft.py <scene_dir> <raw_index.json> [workers=6]
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SCENE = Path(sys.argv[1]); INDEX = Path(sys.argv[2])
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
OUT = SCENE / "rgb"


def work(job):
    name, entry = job
    try:
        from polar_normals import PolarFrame
        tar, pre, off, size = entry
        with open(tar, "rb") as f:
            f.seek(pre + off)
            raw = f.read(size)
        att = PolarFrame(raw).deglare_atten()
        orig = cv2.imread(str(SCENE / "rgb_orig" / f"{name}.jpg")).astype(np.float32)
        out = np.clip(orig * att[..., None], 0, 255).astype(np.uint8)
        ok = cv2.imwrite(str(OUT / f"{name}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return name, None if ok else "imwrite failed"
    except Exception as e:
        return name, repr(e)


if __name__ == "__main__":
    idx = json.loads(INDEX.read_text())
    frames = sorted(p.stem for p in (SCENE / "rgb_orig").glob("*.jpg"))
    jobs = [(n, idx[n]) for n in frames if n in idx]
    print(f"{len(jobs)} frames -> {OUT} (recorder x atten), {WORKERS} workers", flush=True)
    t0 = time.time(); ok = 0; err = 0
    with Pool(WORKERS) as pool:
        for i, (name, e) in enumerate(pool.imap_unordered(work, jobs, chunksize=4), 1):
            if e:
                err += 1; print("ERR", name, e, flush=True)
            else:
                ok += 1
            if i % 200 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"{i}/{len(jobs)} ok={ok} err={err} {el:.0f}s", flush=True)
    print("RGB_SOFT_DONE", ok, err, flush=True)
