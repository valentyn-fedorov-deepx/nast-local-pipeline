"""Pack a bare point cloud (no COLMAP) into the viewer format for a quick look.

Synthesises a modest camera track along the cloud's long axis so the viewer
has an up vector and a non-empty timeline; free-orbit is what matters here.

Usage: python pack_bare.py <cloud.ply> <out_dir> [max_points]
"""
import json
import sys
import numpy as np
from pathlib import Path

INP = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
_num = [a for a in sys.argv[3:] if not a.startswith('--')]
MAXP = int(_num[0]) if _num else 8_000_000


def read_ply(p):
    with open(p, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"end_header\n"):
            hdr += f.readline()
        n = int([l for l in hdr.decode().splitlines() if l.startswith("element vertex")][0].split()[-1])
        d = np.fromfile(f, dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                           ("red", "u1"), ("green", "u1"), ("blue", "u1")]), count=n)
    return (np.stack([d["x"], d["y"], d["z"]], -1).astype(np.float32),
            np.stack([d["red"], d["green"], d["blue"]], -1).astype(np.uint8))


xyz, rgb = read_ply(INP)
med = np.median(xyz, 0); r = np.linalg.norm(xyz - med, axis=1)
keep = r < np.percentile(r, 99.5); xyz, rgb = xyz[keep], rgb[keep]
rs = np.random.RandomState(0)
if len(xyz) > MAXP:
    s = rs.permutation(len(xyz))[:MAXP]; xyz, rgb = xyz[s], rgb[s]
else:
    s = rs.permutation(len(xyz)); xyz, rgb = xyz[s], rgb[s]

lo, hi = xyz.min(0).astype(float), xyz.max(0).astype(float)
ctr = (lo + hi) / 2; ext = float(np.linalg.norm(hi - lo) / 3)
# PCA: long axis = travel, smallest = up
X = xyz - ctr
_, _, Vt = np.linalg.svd(X[rs.choice(len(X), min(50000, len(X)), replace=False)],
                         full_matrices=False)
axis = Vt[0]; up = Vt[2]
# object close-ups (asset2ply already re-axed them y-up): use +y, and orbit
# around a horizontal axis instead of the PCA "travel" one
if "--yup" in sys.argv:                          # y-up assets
    up = np.array([0.0, 1.0, 0.0]); axis = np.array([1.0, 0.0, 0.0])
if "--xup" in sys.argv:                          # TRELLIS gaussian assets are x-up
    up = np.array([1.0, 0.0, 0.0]); axis = np.array([0.0, 0.0, 1.0])
_upf = [a for a in sys.argv if a.startswith("--up=")]
front = None
if _upf:                                          # explicit up (+ optional front) from asset2ply
    _uj = json.load(open(_upf[0][5:]))
    up = np.array(_uj["up"], dtype=float); up /= np.linalg.norm(up)
    ref = np.array([0.0, 1.0, 0.0]) if abs(up[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    axis = np.cross(up, ref); axis /= np.linalg.norm(axis)
    if _uj.get("front"):
        front = np.array(_uj["front"], dtype=float); front -= up * (front @ up)
        front /= max(np.linalg.norm(front), 1e-9)
frames = []
for i, t in enumerate(np.linspace(-0.4, 0.4, 40)):
    c = ctr + axis * ext * t
    z = -(ctr - c); z = z / (np.linalg.norm(z) + 1e-9)
    x = np.cross(up, z); x /= np.linalg.norm(x) + 1e-9
    y = np.cross(z, x)
    R = np.stack([x, y, z])                      # world->cam rows
    t3 = -R @ c
    # quat from R
    q = np.empty(4); tr = np.trace(R)
    if tr > 0:
        s2 = 0.5 / np.sqrt(tr + 1); q = np.array([0.25 / s2, (R[2,1]-R[1,2])*s2, (R[0,2]-R[2,0])*s2, (R[1,0]-R[0,1])*s2])
    else:
        q = np.array([1.0, 0, 0, 0])
    frames.append({"i": i, "name": f"v_{i:04d}", "p": [float(v) for v in c],
                   "q": [float(v) for v in q]})

xyz.astype(np.float32).tofile(OUT / "pos.f32")
rgb.astype(np.uint8).tofile(OUT / "rgb.u8")
spacing = ext * 0.002
(OUT / "meta.json").write_text(json.dumps({
    "name": INP.stem, "count": int(len(xyz)),
    "bbox": {"min": lo.tolist(), "max": hi.tolist()}, "extent": ext,
    "spacing": spacing, "cell": ext, "has_nrm": False,
    "intrinsics": {"fx": 900.0, "fy": 900.0, "cx": 512, "cy": 384, "w": 1024, "h": 768},
    "up": [float(v) for v in up],
    "front": [float(v) for v in front] if front is not None else None}, indent=1))
(OUT / "poses.json").write_text(json.dumps({"up": [float(v) for v in up], "frames": frames}))
(OUT / "cells.json").write_text(json.dumps(
    [[float(ctr[0]), float(ctr[1]), float(ctr[2]), ext * 10, 0, int(len(xyz))]]))
print("PACK_BARE_DONE", OUT, len(xyz))
