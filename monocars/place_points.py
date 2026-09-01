"""Drop a TRELLIS asset into the street POINT CLOUD as dense colored points.

No splat training anywhere: the asset's gaussians become points (f_dc -> RGB,
opacity-filtered), get scaled/rotated into the object's oriented pose, replace
the original points inside the box, and the result is re-packed for the WebGL
point viewer (shuffle-LOD + coarse cells, same as pack_cloud).

Usage: python place_points.py <street_pack_dir> <asset.ply> <box.json> <out_dir> [up_axis=0] [flip=1] [--up=<asset.up.json>]
"""
import json
import sys
import numpy as np
from pathlib import Path

PACK = Path(sys.argv[1]); ASSET = Path(sys.argv[2])
BOX = json.loads(Path(sys.argv[3]).read_text())
OUT = Path(sys.argv[4]); OUT.mkdir(parents=True, exist_ok=True)
A_UP_I = int(sys.argv[5]) if len(sys.argv) > 5 else 0     # TRELLIS is x-up
FLIP = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0

PLY_T = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}


def read_ply(p):
    with open(p, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"end_header\n"):
            hdr += f.readline()
        fields, n = [], 0
        for l in hdr.decode().splitlines():
            if l.startswith("element vertex"):
                n = int(l.split()[-1])
            elif l.startswith("property "):
                _, t, name = l.split()[:3]
                fields.append((name, PLY_T[t]))
        return np.fromfile(f, dtype=np.dtype(fields), count=n)


st_pos = np.frombuffer((PACK / "pos.f32").read_bytes(), np.float32).reshape(-1, 3).copy()
st_rgb = np.frombuffer((PACK / "rgb.u8").read_bytes(), np.uint8).reshape(-1, 3).copy()
print(f"street {len(st_pos):,} pts", flush=True)

g = read_ply(ASSET)
ax = np.stack([g["x"], g["y"], g["z"]], -1).astype(np.float64)
C0 = 0.28209479177387814
rgb = np.clip(np.stack([g["f_dc_0"], g["f_dc_1"], g["f_dc_2"]], -1) * C0 + 0.5, 0, 1)
rgb = (rgb * 255).astype(np.uint8)
if "opacity" in g.dtype.names:
    op = 1 / (1 + np.exp(-g["opacity"].astype(np.float64)))
    keep = op > 0.35
    ax, rgb = ax[keep], rgb[keep]
print(f"asset {len(ax):,} pts after opacity cut", flush=True)

ps = BOX["pose"]
qw, qx, qy, qz = ps["q"]
Rp = np.array([
    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
fw, up, lat = Rp[:, 0], Rp[:, 1], Rp[:, 2]
t = np.array(ps["t"]); L, H, W = ps["size"]

lo, hi = np.percentile(ax, 2, axis=0), np.percentile(ax, 98, axis=0)
a_ctr = (hi + lo) / 2
lat = np.cross(fw, up)                                     # world frame (fw, up, lat) right-handed

# ---- asset frame: up from asset2ply (--up=<json>: mask-matched, with an optional
# "front" = object->camera direction of the operator's first crop), else axis index
_upj = [a for a in sys.argv if a.startswith("--up=")]
a_front = None
if _upj:
    _uj = json.loads(Path(_upj[0][5:]).read_text())
    a_up = np.array(_uj["up"], float); a_up /= np.linalg.norm(a_up)
    if _uj.get("front"):
        a_front = np.array(_uj["front"], float); a_front -= a_up * (a_front @ a_up)
        a_front /= max(np.linalg.norm(a_front), 1e-9)
else:
    a_up = np.zeros(3); a_up[A_UP_I] = 1.0
Xc = ax - a_ctr
Hh = Xc - np.outer(Xc @ a_up, a_up)                       # horizontal part
_, _, Vt = np.linalg.svd(Hh[np.random.RandomState(0).choice(len(Hh), min(40000, len(Hh)), replace=False)],
                         full_matrices=False)
e_long = Vt[0] - a_up * (Vt[0] @ a_up); e_long /= np.linalg.norm(e_long)
e_side = np.cross(e_long, a_up)                            # (e_long, a_up, e_side) right-handed

def frame_map(src, dst):
    """rotation taking orthonormal triple src[i] -> dst[i]"""
    return sum(np.outer(d, s_) for s_, d in zip(src, dst))

# yaw: where was the camera of crop 0, seen from the object (world, horizontal)?
g = None
rf = BOX.get("roi_frames") or []
if a_front is not None and rf and rf[0].get("pos"):
    g = np.array(rf[0]["pos"], float) - t; g -= up * (g @ up)
    g = g / np.linalg.norm(g) if np.linalg.norm(g) > 1e-6 else None
elong = L / max(W, 1e-6)
if g is not None and elong > 1.3:
    # elongated: keep long-to-long, the crop's front only decides the 180-deg flip
    cands = [frame_map((sg * e_long, a_up, sg * e_side), (fw, up, lat)) for sg in (1.0, -1.0)]
    R = max(cands, key=lambda Rm: float((Rm @ a_front) @ g))
    how = "long-axis + front flip"
elif g is not None:
    R = frame_map((a_up, a_front, np.cross(a_up, a_front)), (up, g, np.cross(up, g)))
    how = "front-aligned yaw"
else:
    R = frame_map((FLIP * e_long, a_up, FLIP * e_side), (fw, up, lat))
    how = "long-axis (no front)"
P0 = Xc @ R.T
e_fw = float(np.percentile(P0 @ fw, 98) - np.percentile(P0 @ fw, 2))
e_lat = float(np.percentile(P0 @ lat, 98) - np.percentile(P0 @ lat, 2))
if elong > 1.3:
    s = L / max(e_fw, 1e-6)
else:
    s = float(np.sqrt((L / max(e_fw, 1e-6)) * (W / max(e_lat, 1e-6))))
P = P0 * s + t
base_off = (float(t @ up) - H / 2) - float(np.percentile(P @ up, 2))
P += up * base_off
print(f"asset placed: scale {s:.3f}, up {np.round(a_up, 2)}, {how}", flush=True)
# asset -> world:  X_w = s * R @ (X_a - a_ctr) + t + up * base_off
json.dump({"R": R.tolist(), "s": float(s), "t": t.tolist(), "a_ctr": a_ctr.tolist(),
           "up": up.tolist(), "base_off": float(base_off), "how": how},
          open(ASSET.parent / "placement.json", "w"))

# cut the original object out of the street (oriented box, generous margin)
Lc = (st_pos.astype(np.float64) - t) @ Rp
half = np.array([L, H, W]) * 1.15 / 2
inside = np.all(np.abs(Lc) < half + 0.02, axis=1)
xyz = np.concatenate([st_pos[~inside], P.astype(np.float32)], 0)
cols = np.concatenate([st_rgb[~inside], rgb], 0)
print(f"cut {inside.sum():,} street pts, merged total {len(xyz):,}", flush=True)

# re-pack: shuffle (prefix = LOD) then coarse-cell sort
rs = np.random.RandomState(0)
order = rs.permutation(len(xyz))
xyz, cols = xyz[order], cols[order]
ext_all = xyz.max(0) - xyz.min(0)
cell = float(max(ext_all.max() / 22.0, 1e-6))
q = np.floor((xyz - xyz.min(0)) / cell).astype(np.int64)
key = (q[:, 0] << 42) ^ (q[:, 1] << 21) ^ q[:, 2]
order2 = np.argsort(key, kind="stable")
xyz, cols, key = xyz[order2], cols[order2], key[order2]
uniq, starts, counts = np.unique(key, return_index=True, return_counts=True)
cells = []
for stt, ct in zip(starts, counts):
    block = xyz[stt:stt + ct]
    lo_b, hi_b = block.min(0), block.max(0)
    c = (lo_b + hi_b) / 2
    rad = float(np.linalg.norm(hi_b - c)) + 1e-6
    cells.append([float(c[0]), float(c[1]), float(c[2]), rad, int(stt), int(ct)])

(OUT / "pos.f32").write_bytes(np.ascontiguousarray(xyz, np.float32).tobytes())
(OUT / "rgb.u8").write_bytes(np.ascontiguousarray(cols, np.uint8).tobytes())
(OUT / "cells.json").write_text(json.dumps(cells))
meta = json.loads((PACK / "meta.json").read_text())
meta["count"] = int(len(xyz))
mn, mx = xyz.min(0), xyz.max(0)
meta["bbox"] = {"min": [float(v) for v in mn], "max": [float(v) for v in mx]}
(OUT / "meta.json").write_text(json.dumps(meta))
(OUT / "poses.json").write_bytes((PACK / "poses.json").read_bytes())
print(f"PLACE_POINTS_DONE {OUT} pts={len(xyz)} cells={len(cells)}", flush=True)
