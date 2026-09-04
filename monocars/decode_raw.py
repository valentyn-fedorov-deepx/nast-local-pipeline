"""Batch-decode a scene's RAW12 frames into everything the viewer needs.

Input is raw ONLY: either <scene>/raw/*.raw12 or a folder holding the .raw12
files directly. Per frame it writes rgb/<stem>.jpg (the working colour
stream: the recorder look x glare attenuation), rgb_orig/<stem>.jpg (plain S0)
and layers/<product>/<stem>.jpg for the locked normals catalog. Resumable:
frames that already have every output are skipped (existing rgb/ from a
recorder is kept as is), so it doubles as the "finish the rest" pass. The
GUI runs this right after a raw import and shows its progress.

Usage: python decode_raw.py <scene_dir> [workers=6]
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

SCENE = Path(sys.argv[1])
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
RAW = SCENE / "raw" if (SCENE / "raw").is_dir() else SCENE
PRODUCTS = ("nxyz", "n_xy", "n_xz", "nxyz_phys", "nxyz_diffuse",
            "nxyz_specv2", "edge", "rgb_deglare")
JPG = [int(cv2.IMWRITE_JPEG_QUALITY), 92]


def raw_files():
    return sorted(set(RAW.glob("*.raw12")) | set(RAW.glob("*.raw")), key=lambda q: q.name)


def complete(stem):
    return ((SCENE / "rgb" / f"{stem}.jpg").exists() and (SCENE / "rgb_orig" / f"{stem}.jpg").exists()
            and all((SCENE / "layers" / k / f"{stem}.jpg").exists() for k in PRODUCTS))


def work(path):
    stem = Path(path).stem
    try:
        from polar_normals import PolarFrame, products
        if complete(stem):
            return stem, None
        fr = PolarFrame(Path(path).read_bytes())
        # colour first: rgb = working stream (soft deglare), rgb_orig = plain S0.
        # a recorder-made rgb/rgb_orig (the shipped dataset) is never overwritten
        for sub, fn in (("rgb", fr.color_work), ("rgb_orig", fr.color_recorder)):
            out = SCENE / sub / f"{stem}.jpg"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), fn(), JPG)
        P = products(fr)
        for k in PRODUCTS:
            d = SCENE / "layers" / k
            d.mkdir(parents=True, exist_ok=True)
            if not (d / f"{stem}.jpg").exists():
                cv2.imwrite(str(d / f"{stem}.jpg"), P[k], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return stem, None
    except Exception as e:
        return stem, repr(e)


if __name__ == "__main__":
    files = raw_files()
    todo = [str(q) for q in files if not complete(q.stem)]
    print(f"DECODE_START total={len(files)} todo={len(todo)} workers={WORKERS} raw={RAW}", flush=True)
    t0 = time.time(); ok = 0; err = 0
    with Pool(WORKERS) as pool:
        for i, (stem, e) in enumerate(pool.imap_unordered(work, todo, chunksize=2), 1):
            if e:
                err += 1; print("ERR", stem, e, flush=True)
            else:
                ok += 1
            if i % 20 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"DECODE_PROGRESS {i}/{len(todo)} ok={ok} err={err} "
                      f"{el / max(i, 1):.2f}s/frame eta={el / max(i, 1) * (len(todo) - i):.0f}s", flush=True)
    print(f"DECODE_DONE {ok} ok {err} err in {time.time() - t0:.0f}s", flush=True)
