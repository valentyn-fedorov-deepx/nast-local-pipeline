"""Chunked VGGT-Omega street reconstruction — LOCAL single-GPU edition.

Same algorithm as the tex1 vggto_chunks.py (every chunk anchored to the
COLMAP world independently via MoGe depth + Umeyama, no drift chaining),
adapted to a 12-16 GB consumer GPU:
  * default chunk 24 frames @ 512 px (fits a 16 GB card with headroom)
  * on CUDA OOM the chunk size halves (down to 6) and the same span retries
  * torch.cuda.empty_cache() between chunks
  * poses/intrinsics come from the local scene pack, images/depth from the
    local street_video tree — no remote anything

Usage:
  python vggto_local.py <ckpt> <images_dir> <depth_dir> <prefix|ALL> <out.ply>
                        [chunk=24] [overlap=2] [res=512] [conf_pct=15]
                        [stride=1] [vox=-0.00015] [cap=8000000]
  env PACK_DIR = scene pack with poses.json + meta.json (default ../viewer/scenes/street)
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "dx_wrap"))

CKPT = Path(sys.argv[1]); IMAGES = Path(sys.argv[2]); DEPTHS = Path(sys.argv[3])
PREFIX = sys.argv[4]; OUT = Path(sys.argv[5])
CHUNK = int(sys.argv[6]) if len(sys.argv) > 6 else 24
OVER = int(sys.argv[7]) if len(sys.argv) > 7 else 2
RES = int(sys.argv[8]) if len(sys.argv) > 8 else 512
CONF_PCT = float(sys.argv[9]) if len(sys.argv) > 9 else 15.0
STRIDE = int(sys.argv[10]) if len(sys.argv) > 10 else 1
VOX = float(sys.argv[11]) if len(sys.argv) > 11 else -0.00015
CAP = int(float(sys.argv[12])) if len(sys.argv) > 12 else 8_000_000

from core.vggt_omega_processor import VGGTOmegaProcessor

PACK = Path(os.environ.get("PACK_DIR", str(HERE.parent / "viewer" / "scenes" / "street")))
_meta = json.loads((PACK / "meta.json").read_text())
_poses = json.loads((PACK / "poses.json").read_text())
BY = {f["name"]: f for f in _poses["frames"]}
I_ = _meta["intrinsics"]
W_, H_ = I_["w"], I_["h"]


def Rof(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def moge_world_pts(name, gy, gx, h, w):
    f = BY.get(name)
    p = DEPTHS / (Path(name).stem + ".png")
    if f is None or not p.exists():
        return None
    d16 = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if d16 is None:
        return None
    dh, dw = d16.shape
    sx = (gx + 0.5) * W_ / w
    sy = (gy + 0.5) * H_ / h
    z = d16[(sy * dh / H_).astype(np.int32).clip(0, dh - 1),
            (sx * dw / W_).astype(np.int32).clip(0, dw - 1)].astype(np.float64) / 1000.0
    ok = z > 1e-3
    R = Rof(f["q"]); C = np.array(f["p"])
    xc = (sx - I_["cx"]) / I_["fx"] * z
    yc = (sy - I_["cy"]) / I_["fy"] * z
    Pw = np.stack([xc, yc, z], -1) @ R + C
    return Pw, ok


def umeyama(A, B):
    muA, muB = A.mean(0), B.mean(0)
    Ac, Bc = A - muA, B - muB
    Hm = Ac.T @ Bc / len(A)
    U, S, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    s = np.trace(np.diag(S) @ D) / (Ac ** 2).sum() * len(A)
    t = muB - s * R @ muA
    return s, R, t


roster = {p.stem for p in DEPTHS.glob("*.png")}
if PREFIX == "ALL":
    names = sorted((p.name for p in IMAGES.glob("*.jpg") if p.stem in roster),
                   key=lambda n: n.split("_", 1)[1])[::STRIDE]
else:
    names = sorted(p.name for p in IMAGES.glob(PREFIX + "*") if p.stem in roster)[::STRIDE]
names = [n for n in names if n in BY]
print(f"frames: {len(names)} (prefix {PREFIX}, stride {STRIDE}), chunk {CHUNK} overlap {OVER}, "
      f"res {RES}", flush=True)

proc = VGGTOmegaProcessor(model_path=CKPT, device="cuda", image_resolution=RES)
free, total = torch.cuda.mem_get_info()
print(f"model loaded — VRAM {total / 2**30:.1f} GB total, {free / 2**30:.1f} GB free", flush=True)


def npy(x):
    return x.detach().cpu().to(torch.float32).numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def moge_valid(name, h, w):
    p = DEPTHS / (Path(name).stem + ".png")
    if not p.exists():
        return np.ones((h, w), bool)
    d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if d is None:
        return np.ones((h, w), bool)
    return cv2.resize((d > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


acc_p, acc_c = [], []
t_all = time.time()
chunk = CHUNK
s0 = 0
ci = 0
while s0 < max(1, len(names) - OVER):
    sub = names[s0:s0 + chunk]
    if len(sub) <= OVER and ci > 0:
        break
    t0 = time.time()
    try:
        img_t = proc.preprocess([IMAGES / n for n in sub])
        with torch.inference_mode():
            preds = proc.run_model(img_t)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if chunk <= 6:
            raise
        chunk = max(6, chunk // 2)
        print(f"OOM at {len(sub)} frames -> retry span with chunk {chunk}", flush=True)
        continue
    wp = npy(preds["world_points"])[0]
    conf = npy(preds.get("world_points_conf", preds.get("depth_conf")))
    conf = conf[0] if conf.ndim == 4 else conf.reshape(wp.shape[:3])
    imgs = npy(preds["images"])[0].transpose(0, 2, 3, 1)
    del preds, img_t
    torch.cuda.empty_cache()
    N, h, w = wp.shape[:3]
    valid = np.stack([moge_valid(n, h, w) for n in sub])
    valid &= np.isfinite(wp).all(-1)
    thr = np.percentile(conf[valid], CONF_PCT) if valid.any() else 0
    keep = valid & (conf > thr)

    A, Bm = [], []
    gy4, gx4 = np.mgrid[0:h:4, 0:w:4]
    gy4 = gy4.ravel(); gx4 = gx4.ravel()
    for k in range(0, N, max(1, N // 6)):
        got = moge_world_pts(sub[k], gy4, gx4, h, w)
        if got is None:
            continue
        Pw, okd = got
        km = keep[k][gy4, gx4] & okd
        if km.sum() < 400:
            continue
        A.append(wp[k][gy4[km], gx4[km]])
        Bm.append(Pw[km])
    if not A:
        print(f"chunk {ci}: NO world anchor — chunk dropped", flush=True)
        s0 += chunk - OVER; ci += 1
        continue
    A = np.concatenate(A, 0); Bm = np.concatenate(Bm, 0)
    if len(A) > 40_000:
        rs = np.random.RandomState(0)
        ii = rs.choice(len(A), 40_000, replace=False)
        A, Bm = A[ii], Bm[ii]
    med = 0.0
    for it in range(2):
        sc, R, t = umeyama(A, Bm)
        res = np.linalg.norm(sc * (R @ A.T).T + t - Bm, axis=1)
        med = np.median(res)
        good = res < max(3 * med, 1e-6)
        A, Bm = A[good], Bm[good]
    wp = (sc * (R @ wp.reshape(-1, 3).T)).T.reshape(N, h, w, 3) + t
    print(f"chunk {ci}: anchor on {len(A)} px, scale {sc:.4f}, resid med {med:.3f}", flush=True)

    body = slice(OVER, None) if ci > 0 else slice(None)
    P = wp[body][keep[body]]
    Cc = (imgs[body][keep[body]] * 255).astype(np.uint8)
    per_cap = max(200_000, CAP // max(1, (len(names) // max(1, chunk - OVER)) + 1))
    if len(P) > per_cap:
        rs = np.random.RandomState(ci)
        ii = rs.choice(len(P), per_cap, replace=False)
        P, Cc = P[ii], Cc[ii]
    acc_p.append(P.astype(np.float32)); acc_c.append(Cc)
    print(f"chunk {ci + 1}: {len(sub)} frames -> {len(P):,} pts ({time.time() - t0:.1f}s)", flush=True)
    s0 += chunk - OVER; ci += 1

xyz = np.concatenate(acc_p, 0); col = np.concatenate(acc_c, 0)
print(f"raw total {len(xyz):,}", flush=True)
ext_scene = float((xyz.max(0) - xyz.min(0)).max())
vox = -VOX if VOX < 0 else VOX * ext_scene
print(f"voxel: {vox:.5f} ({ext_scene:.1f} extent)", flush=True)
OFF = 1 << 20
q = np.floor(xyz / vox).astype(np.int64) + OFF
key = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
_, first = np.unique(key, return_index=True)
xyz, col = xyz[first], col[first]
if len(xyz) > CAP:
    rs = np.random.RandomState(1)
    ii = rs.choice(len(xyz), CAP, replace=False)
    xyz, col = xyz[ii], col[ii]
print(f"after voxel {vox:.5f}: {len(xyz):,}", flush=True)

hdr = ("ply\nformat binary_little_endian 1.0\n"
       f"element vertex {len(xyz)}\n"
       "property float x\nproperty float y\nproperty float z\n"
       "property uchar red\nproperty uchar green\nproperty uchar blue\n"
       "end_header\n").encode()
rec = np.zeros(len(xyz), dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                         ("red", "u1"), ("green", "u1"), ("blue", "u1")]))
rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
rec["red"], rec["green"], rec["blue"] = col[:, 0], col[:, 1], col[:, 2]
with open(OUT, "wb") as f:
    f.write(hdr); rec.tofile(f)
print(f"VGGTO_LOCAL_DONE {OUT} pts={len(xyz)} chunks={ci} total {time.time() - t_all:.0f}s", flush=True)
