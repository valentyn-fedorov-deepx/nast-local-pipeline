"""World point cloud of a new recording from its pose-consistent depth (depth_geo of local_gpu/vggto_poses.py).

Every frame is unprojected with its pose. Two things keep one object in one place:
  * the depth is VGGT's, in the same geometry as the poses (not monocular depth re-scaled per frame);
  * a surface seen from 5 m and from 40 m is drawn from the near views only: the error grows with the distance, so in every
    coarse voxel the points observed from much farther than the nearest observation of that voxel are dropped.
Low-confidence pixels, depth edges (flying pixels between an object and what is behind it) and the vignetted corners of the
frame (dark paint on the road under the rig) are dropped.

Usage: python geo_layer.py <pack_dir> <images_dir> <depth_geo_dir> <out.ply>
                           [stride=1] [conf_min=40] [voxel=0.0035] [zmax=15] [edge=0.05] [cap=16000000] [prefix=ALL]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PACK = Path(sys.argv[1]); IMAGES = Path(sys.argv[2]); GEO = Path(sys.argv[3]); OUT = Path(sys.argv[4])
STRIDE = int(sys.argv[5]) if len(sys.argv) > 5 else 1
CONF_MIN = int(sys.argv[6]) if len(sys.argv) > 6 else 40
VOX = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0035
ZMAX = float(sys.argv[8]) if len(sys.argv) > 8 else 15.0
EDGE = float(sys.argv[9]) if len(sys.argv) > 9 else 0.05
CAP = int(float(sys.argv[10])) if len(sys.argv) > 10 else 16_000_000
PREFIX = sys.argv[11] if len(sys.argv) > 11 else "ALL"
CONF = GEO.parent / (GEO.name + "_conf")
COARSE = 0.03                                       # "same place" for the near-view rule (0.03 world units ~ 10 cm)
FAR = 1.6                                           # keep observations up to 1.6 x the nearest one of their voxel
RIM = float(__import__("os").environ.get("GEO_RIM", "0.85"))      # pixels farther than this share of the half-diagonal from the centre: lens vignette

poses = json.loads((PACK / "poses.json").read_text())
I = json.loads((PACK / "meta.json").read_text())["intrinsics"]; W, H = I["w"], I["h"]
fx, fy, cx, cy = I["fx"], I["fy"], I["cx"], I["cy"]
OFF = 1 << 20


def Rof(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def keys(P, vox):
    q = np.floor(P / vox).astype(np.int64) + OFF
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


def points_of(f, step):
    """(world points, depth, pixel index arrays) of a frame; None when it has no depth_geo"""
    dpath = GEO / (Path(f["name"]).stem + ".png")
    if not dpath.exists():
        return None
    d16 = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
    if d16 is None:
        return None
    z = d16.astype(np.float32) / 1000.0; h, w = z.shape
    ok = (z > 0.02) & (z < ZMAX)
    cpath = CONF / dpath.name
    if cpath.exists():
        ok &= cv2.imread(str(cpath), cv2.IMREAD_UNCHANGED) >= CONF_MIN
    zz = np.where(z > 0, z, np.nan)
    gr = np.zeros_like(z)
    gr[:, 1:] = np.maximum(gr[:, 1:], np.abs(zz[:, 1:] - zz[:, :-1])); gr[:, :-1] = np.maximum(gr[:, :-1], np.abs(zz[:, 1:] - zz[:, :-1]))
    gr[1:, :] = np.maximum(gr[1:, :], np.abs(zz[1:, :] - zz[:-1, :])); gr[:-1, :] = np.maximum(gr[:-1, :], np.abs(zz[1:, :] - zz[:-1, :]))
    ok &= np.nan_to_num(gr, nan=1e9) < EDGE * z                     # a neighbour without depth (sky, cut) counts as an edge
    yy, xx = np.mgrid[0:h, 0:w]
    ok &= np.hypot((xx + 0.5) / w - 0.5, ((yy + 0.5) / h - 0.5) * H / W) < RIM * 0.5 * np.hypot(1.0, H / W)
    if step > 1:
        m = np.zeros_like(ok); m[::step, ::step] = True; ok &= m
    vy, vx = np.nonzero(ok)
    if len(vx) < 100:
        return None
    zv = z[vy, vx].astype(np.float64)
    u = (vx + 0.5) / w * W; v = (vy + 0.5) / h * H
    P = np.stack([(u - cx) / fx * zv, (v - cy) / fy * zv, zv], -1) @ Rof(f["q"]) + np.array(f["p"])
    return P, zv, vy, vx, (h, w)


frames = [f for f in poses["frames"] if PREFIX == "ALL" or f["name"].startswith(PREFIX)][::STRIDE]
# pass 1: the nearest observation of every coarse voxel
ck = np.zeros(0, np.int64); cz = np.zeros(0)
bk, bz = [], []
for fi, f in enumerate(frames):
    got = points_of(f, 2)
    if got is None:
        continue
    bk.append(keys(got[0], COARSE)); bz.append(got[1])
    if len(bk) >= 40 or fi == len(frames) - 1:
        k = np.concatenate([ck] + bk); zc = np.concatenate([cz] + bz); o = np.lexsort((zc, k)); k, zc = k[o], zc[o]
        first = np.ones(len(k), bool); first[1:] = k[1:] != k[:-1]
        ck, cz = k[first], zc[first]; bk, bz = [], []
        print(f"geo layer 1/2: {fi + 1}/{len(frames)} frames, {len(ck):,} places", flush=True)
if bk:
    k = np.concatenate([ck] + bk); zc = np.concatenate([cz] + bz); o = np.lexsort((zc, k)); k, zc = k[o], zc[o]
    first = np.ones(len(k), bool); first[1:] = k[1:] != k[:-1]; ck, cz = k[first], zc[first]
if not len(ck):
    sys.exit("geo layer: no depth_geo for the frames of this pack - run the poses step first")

# pass 2: points of the near views, fine voxel dedup (in batches of frames: one merge of the voxel set per batch, not per frame)
seen = np.zeros(0, np.int64); acc_p, acc_c = [], []; n_done = 0
bp, bc, bk = [], [], []


def flush():
    global seen, bp, bc, bk
    if not bp:
        return
    Pb = np.concatenate(bp); Cb = np.concatenate(bc); key, first = np.unique(np.concatenate(bk), return_index=True)
    Pb, Cb = Pb[first], Cb[first]
    if len(seen):
        pos = np.searchsorted(seen, key).clip(0, len(seen) - 1); new = seen[pos] != key
        Pb, Cb, key = Pb[new], Cb[new], key[new]
    acc_p.append(Pb.astype(np.float32)); acc_c.append(Cb.astype(np.uint8))
    seen = np.insert(seen, np.searchsorted(seen, key), key)          # both sorted: a linear merge
    bp, bc, bk = [], [], []


for fi, f in enumerate(frames):
    got = points_of(f, 1)
    if got is None:
        continue
    P, zv, vy, vx, (h, w) = got
    kc = keys(P, COARSE); pos = np.searchsorted(ck, kc).clip(0, len(ck) - 1)
    zmin = np.where(ck[pos] == kc, cz[pos], zv)
    near = zv <= np.maximum(FAR * zmin, zmin + 0.15)
    P, vy, vx = P[near], vy[near], vx[near]
    if len(P) < 50:
        continue
    img = cv2.imread(str(IMAGES / f["name"]))
    if img is None:
        continue
    bp.append(P); bc.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)[vy, vx][:, ::-1]); bk.append(keys(P, VOX))
    n_done += 1
    if len(bp) >= 24:
        flush()
        print(f"geo layer 2/2: {n_done}/{len(frames)} frames, {sum(len(a) for a in acc_p):,} pts", flush=True)
flush()

xyz = np.concatenate(acc_p, 0); rgb = np.concatenate(acc_c, 0)
if len(xyz) > CAP:
    ii = np.random.RandomState(0).choice(len(xyz), CAP, replace=False); xyz, rgb = xyz[ii], rgb[ii]
rec = np.zeros(len(xyz), dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]))
rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
with open(OUT, "wb") as fo:
    fo.write(("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(rec)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n").encode())
    rec.tofile(fo)
print(f"GEO_LAYER_DONE {OUT} pts={len(rec)} frames={n_done}", flush=True)
