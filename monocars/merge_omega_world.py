"""Merge anchored Omega clouds into one COLMAP-world point pack.

Each cloud carries a sim3 (from vggto_anchor) that drops it into the world
where the poses and object boxes live. Transform, concat, voxel-dedup,
re-pack for the WebGL point viewer (shuffle-LOD + cells, as everywhere).

Usage: python merge_omega_world.py <street_pack_dir> <out_dir> \
           <cloudA.ply> <anchorA.json> [<cloudB.ply> <anchorB.json> ...]
"""
import json
import sys
import numpy as np
from pathlib import Path

PACK = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
pairs = [(Path(sys.argv[i]), Path(sys.argv[i + 1])) for i in range(3, len(sys.argv), 2)]

DT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")])


def read_ply(p):
    raw = open(p, "rb").read()
    i0 = raw.find(b"end_header\n") + len(b"end_header\n")
    return np.frombuffer(raw[i0:], dtype=DT)


acc_p, acc_c = [], []
for ply, anc in pairs:
    d = read_ply(ply)
    X = np.stack([d["x"], d["y"], d["z"]], -1).astype(np.float64)
    if str(anc) != "none" and Path(anc).exists():
        a = json.loads(Path(anc).read_text())
        s, R, t = a["s"], np.array(a["R"]), np.array(a["t"])
        X = (s * (R @ X.T)).T + t
        print(f"{ply.name}: {len(X):,} pts, sim3 scale {s:.3f}", flush=True)
    else:
        print(f"{ply.name}: {len(X):,} pts (вже у світі)", flush=True)
    acc_p.append(X.astype(np.float32))
    acc_c.append(np.stack([d["red"], d["green"], d["blue"]], -1))

xyz = np.concatenate(acc_p, 0); col = np.concatenate(acc_c, 0)
print(f"merged raw {len(xyz):,}", flush=True)

VOX = 0.0035
OFF = 1 << 20
q = np.floor(xyz / VOX).astype(np.int64) + OFF
key = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
_, first = np.unique(key, return_index=True)
xyz, col = xyz[first], col[first]
print(f"after voxel {VOX}: {len(xyz):,}", flush=True)

med = np.median(xyz, axis=0)
r = np.linalg.norm(xyz - med, axis=1)
keep = r < np.percentile(r, 99.5)
xyz, col = xyz[keep], col[keep]

rs = np.random.RandomState(0)
order = rs.permutation(len(xyz))
if len(order) > 24_000_000:
    order = order[:24_000_000]
xyz, col = xyz[order], col[order]

ext_all = xyz.max(0) - xyz.min(0)
cell = float(max(ext_all.max() / 22.0, 1e-6))
q = np.floor((xyz - xyz.min(0)) / cell).astype(np.int64)
key = (q[:, 0] << 42) ^ (q[:, 1] << 21) ^ q[:, 2]
order2 = np.argsort(key, kind="stable")
xyz, col, key = xyz[order2], col[order2], key[order2]
uniq, starts, counts = np.unique(key, return_index=True, return_counts=True)
cells = []
for stt, ct in zip(starts, counts):
    block = xyz[stt:stt + ct]
    lo_b, hi_b = block.min(0), block.max(0)
    c = (lo_b + hi_b) / 2
    rad = float(np.linalg.norm(hi_b - c)) + 1e-6
    cells.append([float(c[0]), float(c[1]), float(c[2]), rad, int(stt), int(ct)])

(OUT / "pos.f32").write_bytes(np.ascontiguousarray(xyz, np.float32).tobytes())
(OUT / "rgb.u8").write_bytes(np.ascontiguousarray(col, np.uint8).tobytes())
(OUT / "cells.json").write_text(json.dumps(cells))
meta = json.loads((PACK / "meta.json").read_text())
meta["count"] = int(len(xyz))
mn, mx = xyz.min(0), xyz.max(0)
meta["bbox"] = {"min": [float(v) for v in mn], "max": [float(v) for v in mx]}
(OUT / "meta.json").write_text(json.dumps(meta))
(OUT / "poses.json").write_bytes((PACK / "poses.json").read_bytes())
print(f"MERGE_OMEGA_WORLD_DONE {OUT} pts={len(xyz)} cells={len(cells)}", flush=True)
