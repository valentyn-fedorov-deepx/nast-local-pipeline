"""Batch-decode a scene's RAW12 frames into the locked normals catalog.

This is the up-front decode step: point it at a scene that has rgb/ + raw/
and it fills layers/<product>/ for every frame, in parallel. Resumable —
frames already decoded are skipped, so it doubles as the "finish the rest"
pass. The GUI runs this right after raw import and shows its progress.

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
RAW = SCENE / "raw"
PRODUCTS = ("nxyz", "n_xy", "n_xz", "nxyz_phys", "nxyz_diffuse",
            "nxyz_specv2", "edge", "rgb_deglare")


def work(stem):
    try:
        from polar_normals import PolarFrame, products
        if all((SCENE / "layers" / k / f"{stem}.jpg").exists() for k in PRODUCTS):
            return stem, None
        fr = PolarFrame((RAW / f"{stem}.raw12").read_bytes())
        P = products(fr)
        for k in PRODUCTS:
            d = SCENE / "layers" / k
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{stem}.jpg"), P[k], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return stem, None
    except Exception as e:
        return stem, repr(e)


if __name__ == "__main__":
    stems = sorted(p.stem for p in RAW.glob("*.raw12"))
    todo = [s for s in stems
            if not all((SCENE / "layers" / k / f"{s}.jpg").exists() for k in PRODUCTS)]
    print(f"DECODE_START total={len(stems)} todo={len(todo)} workers={WORKERS}", flush=True)
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
