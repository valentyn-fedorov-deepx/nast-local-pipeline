"""World point cloud from the MoGe depth dump — the "fine detail" map layer.

Omega drops low-confidence pixels (thin poles, edges, far objects). MoGe
depth is per-pixel and already world-scaled, so every lamp post that shows in
a frame is in the depth map. Unproject a pixel grid of every dumped frame,
voxel-dedup globally, write ply. Frame stride and grid step trade density
for size; defaults give ~10-20M points for the street.

Usage: python moge_layer.py <pack_dir> <images_dir> <depth_dir> <out.ply>
                            [stride=2] [step=3] [voxel=0.012] [zmax=45]
"""
import json
import sys
import numpy as np
from pathlib import Path
import cv2

PACK = Path(sys.argv[1]); IMAGES = Path(sys.argv[2]); DEPTHS = Path(sys.argv[3])
OUT = Path(sys.argv[4])
STRIDE = int(sys.argv[5]) if len(sys.argv) > 5 else 2
STEP = int(sys.argv[6]) if len(sys.argv) > 6 else 3
VOX = float(sys.argv[7]) if len(sys.argv) > 7 else 0.012
ZMAX = float(sys.argv[8]) if len(sys.argv) > 8 else 45.0

meta = json.loads((PACK / "meta.json").read_text())
poses = json.loads((PACK / "poses.json").read_text())
I = meta["intrinsics"]; W, H = I["w"], I["h"]
fx, fy, cx, cy = I["fx"], I["fy"], I["cx"], I["cy"]


def Rof(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


OFF = 1 << 20
seen = None
acc_p, acc_c = [], []
n_done = 0
frames = poses["frames"][::STRIDE]
for fi, f in enumerate(frames):
    dpath = DEPTHS / (Path(f["name"]).stem + ".png")
    if not dpath.exists():
        continue
    d16 = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
    img = cv2.imread(str(IMAGES / f["name"]))
    if d16 is None or img is None:
        continue
    dh, dw = d16.shape
    gy, gx = np.mgrid[0:H:STEP, 0:W:STEP]
    gy = gy.ravel().astype(np.float64); gx = gx.ravel().astype(np.float64)
    z = d16[(gy * dh / H).astype(np.int32).clip(0, dh - 1),
            (gx * dw / W).astype(np.int32).clip(0, dw - 1)].astype(np.float64) / 1000.0
    ok = (z > 0.05) & (z < ZMAX)
    gx, gy, z = gx[ok], gy[ok], z[ok]
    if len(z) < 100:
        continue
    R = Rof(f["q"]); C = np.array(f["p"])
    xc = (gx - cx) / fx * z
    yc = (gy - cy) / fy * z
    P = np.stack([xc, yc, z], -1) @ R + C
    col = img[gy.astype(np.int32), gx.astype(np.int32)][:, ::-1]      # BGR->RGB

    # per-frame unique then vs seen (same trick as moge_street2)
    q = np.floor(P / VOX).astype(np.int64) + OFF
    key = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
    key, first = np.unique(key, return_index=True)
    P, col = P[first], col[first]
    if seen is not None:
        pos = np.searchsorted(seen, key)
        pos = np.clip(pos, 0, len(seen) - 1)
        new = seen[pos] != key
        P, col, key = P[new], col[new], key[new]
    acc_p.append(P.astype(np.float32)); acc_c.append(col.astype(np.uint8))
    seen = key if seen is None else np.union1d(seen, key)
    n_done += 1
    if n_done % 50 == 0:
        print(f"moge layer: {n_done}/{len(frames)} frames, {sum(len(a) for a in acc_p):,} pts", flush=True)

xyz = np.concatenate(acc_p, 0); rgb = np.concatenate(acc_c, 0)
rec = np.zeros(len(xyz), dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                         ("red", "u1"), ("green", "u1"), ("blue", "u1")]))
rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
with open(OUT, "wb") as fo:
    fo.write(("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(rec)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n").encode())
    rec.tofile(fo)
print(f"MOGE_LAYER_DONE {OUT} pts={len(rec)} frames={n_done}", flush=True)
