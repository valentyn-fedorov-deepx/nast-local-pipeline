"""raw12 frames -> RGB + normal layers (Orthovector IMX264MYR polarization camera).

Reads every *.raw12 in <in_dir> and writes, per frame, into <out_dir>:
    rgb/<stem>.jpg          working colour: the recorder look with the glare attenuation
    rgb_orig/<stem>.jpg     the recorder look, plain (linear S0/2, fixed white balance)
    <product>/<stem>.jpg    normal layers, default nxyz + n_xy (see --products)

Products (all half resolution 1224x1024, sensor orientation, canon paint
R=X G=Y B=Z; --roll 0 = upright camera, no vector rotation):
    nxyz          pseudo normals (CameraController init_normals), |components|
    n_xy, n_xz    the same field with one component zeroed
    nxyz_phys     physical (VyzLut Fresnel LUT), signed x,y + |z|
    nxyz_diffuse  PxDiffuse (Atkinson)
    nxyz_specv2   PxSpecularV2 (Kadambi)
    edge          edge map from the pseudo normals
    rgb_deglare   full Stokes-minimum deglare colour

Usage:
    python raw_to_layers.py <in_dir> <out_dir> [--products nxyz,n_xy] [--workers 6] [--png] [--roll 0]
Resumable: frames whose outputs already exist are skipped.
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

ALL = ("nxyz", "n_xy", "n_xz", "nxyz_phys", "nxyz_diffuse", "nxyz_specv2", "edge", "rgb_deglare")


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("in_dir"); ap.add_argument("out_dir")
    ap.add_argument("--products", default="nxyz,n_xy", help="comma list, or 'all'")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--png", action="store_true", help="lossless PNG instead of JPEG q92")
    ap.add_argument("--no-rgb", action="store_true", help="skip rgb/ and rgb_orig/")
    ap.add_argument("--roll", type=float, default=0.0,
                    help="camera roll in degrees: 0 = upright camera (default), -90 = the Orthovector rig mount")
    return ap.parse_args()


ARGS = None


def outputs(stem):
    subs = [] if ARGS.no_rgb else ["rgb", "rgb_orig"]
    subs += ARGS.prods
    ext = ".png" if ARGS.png else ".jpg"
    return [Path(ARGS.out_dir) / s / f"{stem}{ext}" for s in subs]


def work(path):
    path = Path(path); stem = path.stem
    try:
        from polar_normals import PolarFrame, products
        outs = outputs(stem)
        if all(o.exists() for o in outs):
            return stem, None
        fr = PolarFrame(path.read_bytes())
        P = None
        enc = [] if ARGS.png else [int(cv2.IMWRITE_JPEG_QUALITY), 92]
        for o in outs:
            if o.exists():
                continue
            o.parent.mkdir(parents=True, exist_ok=True)
            kind = o.parent.name
            if kind == "rgb":
                img = fr.color_work()
            elif kind == "rgb_orig":
                img = fr.color_recorder()
            else:
                if P is None:
                    P = products(fr, roll_deg=ARGS.roll)
                img = P[kind]
            cv2.imwrite(str(o), img, enc)
        return stem, None
    except Exception as e:
        return stem, repr(e)


def main():
    global ARGS
    ARGS = parse()
    ARGS.prods = list(ALL) if ARGS.products.strip() == "all" else [p.strip() for p in ARGS.products.split(",") if p.strip()]
    bad = [p for p in ARGS.prods if p not in ALL]
    if bad:
        sys.exit(f"unknown products {bad}; choose from {ALL}")
    files = sorted(set(Path(ARGS.in_dir).glob("*.raw12")) | set(Path(ARGS.in_dir).glob("*.raw")), key=lambda p: p.name)
    if not files:
        sys.exit(f"no .raw12 files in {ARGS.in_dir}")
    todo = [str(p) for p in files if not all(o.exists() for o in outputs(p.stem))]
    print(f"{len(files)} frames, {len(todo)} to do, products {ARGS.prods}, workers {ARGS.workers}", flush=True)
    t0 = time.time(); ok = err = 0
    with Pool(ARGS.workers, initializer=_init, initargs=(ARGS,)) as pool:
        for i, (stem, e) in enumerate(pool.imap_unordered(work, todo, chunksize=2), 1):
            if e:
                err += 1; print("ERR", stem, e, flush=True)
            else:
                ok += 1
            if i % 20 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"{i}/{len(todo)}  {el / i:.2f} s/frame  eta {el / i * (len(todo) - i):.0f} s", flush=True)
    print(f"done: {ok} ok, {err} errors, {time.time() - t0:.0f} s -> {ARGS.out_dir}", flush=True)


def _init(args):
    global ARGS
    ARGS = args


if __name__ == "__main__":
    main()
