"""Camera poses AND pose-consistent depth for a NEW recording — no COLMAP, no GNSS.

VGGT-Omega on a chunk of frames gives cameras and depth maps that agree with each other: the same point seen from two frames
lands within 1-2 % of its depth. Monocular MoGe depth on the same cameras is off by 6-19 % (and differently near and far), which
smears one parked car into several. So the geometry comes from VGGT alone, MoGe sets one number — the metric size of the world.
  1. GPU: every camera (name prefix A_/B_/...) on its own, chunks of CHUNK frames with OVERLAP shared frames. Rotations, centres
     and depth maps of every chunk are kept (depth on disk, <out>/_geo).
  2. Scale of a chunk: the height of its cameras above the road plane fitted to its own points — a constant of the rig, so
     lam_k = H / h_k. Neighbouring chunks are tied by the depth ratio of the same pixels in the shared frames. Both go into one
     small least-squares problem over log lam. H, the camera height in world units, is the only number MoGe gives: baselines
     triangulated from SIFT matches against the MoGe depth, median over the recording.
  3. The reference camera (longest track): chunks chained by the shared cameras (rotation average + centres).
  4. Every other camera of the rig is NOT chained on its own. The rig is rigid: its pose is the reference camera's pose at the same
     timestamp times a constant transform. Rotation of that transform: the road normal and the driving direction as both cameras
     see them at the same moments. Lever arm: zero (a back-to-back unit; NAST_RIG_LEVER for another rig). Every chunk of that
     camera is fitted onto the predicted track with its own scale and shift, so its depth stays consistent with its poses.
  5. Output: <out>/poses.json + meta.json (format of inspector/scene_base) + poses_report.json, and per frame
     <recording>/depth_geo/<stem>.png — VGGT depth in world units, same 16-bit format as the MoGe dump (sky = 0) — with
     <recording>/depth_geo_conf/<stem>.png (confidence rank 0..255 inside the chunk). The map and the ROI solve read these.

Usage:
  python vggto_poses.py <ckpt> <images_dir> <depth_dir|none> <out_dir> [chunk=24] [overlap=6] [res=512] [max_frames=0]
  env NAST_GEO_DIR    where depth_geo / depth_geo_conf go (default: next to <images_dir>)
      NAST_RIG_LEVER  where the other camera sits in the reference camera's frame, in camera heights (default 0,0,0: a back-to-back
                      unit. Checked on the shipped recording: the clouds of the two cameras agree best at zero)
      NAST_POSES_KEEP=1   keep the GPU pass (<out>/_chunks_*.json + <out>/_geo, 1.3 MB per frame); NAST_POSES_REUSE=1 runs from it
"""
import datetime
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "dx_wrap"))

CKPT = Path(sys.argv[1]); IMAGES = Path(sys.argv[2])
DEPTHS = None if sys.argv[3].lower() in ("none", "", "-") else Path(sys.argv[3])
OUT = Path(sys.argv[4]); OUT.mkdir(parents=True, exist_ok=True)
CHUNK = int(sys.argv[5]) if len(sys.argv) > 5 else 24
OVER = int(sys.argv[6]) if len(sys.argv) > 6 else 6
RES = int(sys.argv[7]) if len(sys.argv) > 7 else 512
MAXF = int(sys.argv[8]) if len(sys.argv) > 8 else 0
GEO = Path(os.environ.get("NAST_GEO_DIR") or IMAGES.parent)
TMP = OUT / "_geo"; TMP.mkdir(exist_ok=True)
FX_RATIO = 641.601591290442 / 1224.0              # calibrated lens of the rig cameras: fx / image width
RIG_LEVER = [float(v) for v in os.environ.get("NAST_RIG_LEVER", "0,0,0").split(",")]      # back-to-back unit: both cameras in one place
GAPS = (4, 8)                                      # frame gaps of the triangulation pairs
PAIR_STEP = 5                                      # every 5th frame starts a pair: ~8 pairs per chunk, thousands of matches
SIG_H, SIG_R, SIG_M = 0.05, 0.02, 0.25             # log-scale sigmas: height anchor, seam depth ratio (two chunks agree on a shared frame to 1-2 %), MoGe triangulation of one chunk


# ------------------------------------------------------------------------------------------------ small geometry helpers
def so3_mean(Rs):
    U, _, Vt = np.linalg.svd(np.sum(Rs, 0))
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def rot_interp(R0, R1, a):
    """rotation a of the way from R0 to R1 (both world->camera)"""
    rv = cv2.Rodrigues(R1 @ R0.T)[0]
    return cv2.Rodrigues(rv * a)[0] @ R0


def unit(v):
    return v / (np.linalg.norm(v) + 1e-18)


def quat_wxyz(R):
    """rotation matrix -> unit quaternion (w, x, y, z), w >= 0 (poses.json: q = world->camera)"""
    K = np.array([[R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
                  [R[0, 1] + R[1, 0], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
                  [R[0, 2] + R[2, 0], R[1, 2] + R[2, 1], R[2, 2] - R[0, 0] - R[1, 1], 0],
                  [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R[0, 0] + R[1, 1] + R[2, 2]]]) / 3.0
    w, V = np.linalg.eigh(K)
    x, y, z, qw = V[:, np.argmax(w)]
    q = np.array([qw, x, y, z])
    return (q if q[0] >= 0 else -q).tolist()


def ts_of(name):
    m = re.search(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{6})", name)
    if not m:
        return None
    y, mo, d, h, mi, s, us = (int(g) for g in m.groups())
    try:
        return datetime.datetime(y, mo, d, h, mi, s, us).timestamp()
    except ValueError:
        return None


def frame_id(name):
    return int("".join(re.findall(r"\d+", Path(name).stem)) or "0")


def moge_full(name):
    """MoGe depth at full resolution (dump units / 1000), 0 = invalid/sky; None when there is no dump for this frame"""
    if DEPTHS is None:
        return None
    p = DEPTHS / (Path(name).stem + ".png")
    if not p.exists():
        return None
    d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return None if d is None else d.astype(np.float32) / 1000.0


# ------------------------------------------------------------------------------------------------ frames and cameras
names_all = sorted(p.name for p in IMAGES.glob("*.jpg")) or sorted(p.name for p in IMAGES.glob("*.png"))
if not names_all:
    sys.exit(f"no frames in {IMAGES}")
groups = {}
for n in names_all:
    pre = n.split("_", 1)[0]
    groups.setdefault(pre + "_" if (len(pre) == 1 and pre.isalpha()) else "", []).append(n)
if MAXF:
    groups = {k: v[:MAXF] for k, v in groups.items()}
im0 = cv2.imread(str(IMAGES / names_all[0])); H_, W_ = im0.shape[:2]
FX = FX_RATIO * W_; CX = W_ / 2.0; CY = H_ / 2.0
TOTAL = sum(len(v) for v in groups.values())
print(f"POSES_START frames={TOTAL} cameras={list(groups)} size={W_}x{H_} chunk={CHUNK} overlap={OVER} res={RES} "
      f"depth={'yes' if DEPTHS else 'NO (the world will not be metric: one unit = the camera height)'}", flush=True)

proc = None
DONE = [0]
RAYS = {}


def rays_of(h, w):
    """calibrated ray of every pixel of an (h, w) VGGT map: x/z, y/z"""
    if (h, w) not in RAYS:
        u = (np.arange(w) + 0.5) / w * W_; v = (np.arange(h) + 0.5) / h * H_
        RAYS[(h, w)] = (np.tile(((u - CX) / FX).astype(np.float32), (h, 1)), np.tile(((v - CY) / FX).astype(np.float32)[:, None], (1, w)))
    return RAYS[(h, w)]


def model():
    global proc
    if proc is None:
        from core.vggt_omega_processor import VGGTOmegaProcessor
        proc = VGGTOmegaProcessor(model_path=CKPT, device="cuda", image_resolution=RES)
        free, total = torch.cuda.mem_get_info()
        print(f"model loaded - VRAM {total / 2**30:.1f} GB total, {free / 2**30:.1f} GB free", flush=True)
    return proc


def seam_ratio(da, ca, db, cb):
    """log(lam_b / lam_a) from the same pixels of the same frames seen by two chunks: median log(depth_a / depth_b)"""
    vals = []
    for i in range(len(da)):
        a = da[i].astype(np.float32); b = db[i].astype(np.float32); qa = ca[i].astype(np.float32); qb = cb[i].astype(np.float32)
        m = (qa > np.median(qa)) & (qb > np.median(qb)) & (a > 0) & (b > 0)
        if m.sum() > 500:
            vals.append(float(np.median(np.log(a[m] / b[m]))))
    return float(np.median(vals)) if vals else float("nan")


def time_segments(names):
    """frame lists without holes in time (dropped frames, a cut): a chunk across a hole mixes two places"""
    tt = [ts_of(n) for n in names]
    if len(names) < 3 or any(v is None for v in tt):
        return [names]
    dd = np.diff(np.array(tt, float)); cut = np.where(dd > max(1.0, 6.0 * float(np.median(dd))))[0] + 1
    segs = [list(sg) for sg in np.split(np.array(names, dtype=object), cut)]
    out = []
    for sg in segs:                                                 # a few stray frames join their neighbour rather than make a chunk of their own
        if out and (len(sg) < 6 or len(out[-1]) < 6):
            out[-1] += sg
        else:
            out.append(sg)
    return out


def run_chunks(names, pre, split):
    """GPU pass of one camera: chunks {names, R (N,3,3) world->camera, c (N,3) centres, over = frames shared with the previous chunk,
    rel = log scale ratio to it}, each in its own frame; depth + confidence of the chunk go to <out>/_geo/<pre><k>.npz.
    split: chunk every time segment on its own (cameras that follow the rig); the reference camera is chunked straight through."""
    chunks = []
    for si, seg in enumerate(time_segments(names) if split else [names]):
        run_segment(seg, pre, si, chunks)
    return chunks


def run_segment(names, pre, si, chunks):
    F = len(names); chunk = CHUNK; s0 = 0; prev = None; first = True
    while s0 < F:
        sub = names[s0:s0 + chunk]
        over = min(OVER, chunk // 3)
        if not first and len(sub) <= over:
            break
        t0 = time.time()
        try:
            img_t = model().preprocess([IMAGES / n for n in sub])
            with torch.inference_mode():
                preds = model().run_model(img_t)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk <= 6:
                raise
            chunk = max(6, chunk // 2)
            print(f"OOM at {len(sub)} frames -> chunk {chunk}", flush=True)
            continue
        E = preds["extrinsic"].detach().cpu().to(torch.float32).numpy().reshape(-1, 3, 4)
        Kp = preds["intrinsic"].detach().cpu().to(torch.float32).numpy().reshape(-1, 3, 3)
        dp = preds["depth"].detach().cpu().to(torch.float32).numpy()[0]; dp = dp[..., 0] if dp.ndim == 4 else dp
        cf = preds["depth_conf"].detach().cpu().to(torch.float32).numpy(); cf = cf[0] if cf.ndim == 4 else cf.reshape(dp.shape)
        del preds, img_t
        torch.cuda.empty_cache()
        dp = np.where(np.isfinite(dp) & (dp > 0), dp, 0).astype(np.float16); cf = cf.astype(np.float16)
        R = E[:, :, :3].astype(np.float64); c = np.array([-e[:, :3].T @ e[:, 3] for e in E], np.float64)
        rel = float("nan")
        if prev is not None:
            sh = [n for n in sub if n in prev[0]]
            if sh:
                ia = [prev[0].index(n) for n in sh]; ib = [sub.index(n) for n in sh]
                rel = seam_ratio(prev[1][ia], prev[2][ia], dp[ib], cf[ib])
        k = len(chunks)
        np.savez(TMP / f"{pre or 'single_'}{k:04d}.npz", depth=dp, conf=cf)
        chunks.append({"seg": si, "names": sub, "R": R, "c": c, "over": 0 if first else over, "rel": rel, "fx": float(np.median(Kp[:, 0, 0]) * W_ / dp.shape[2])})
        prev = (sub, dp, cf)
        DONE[0] += len(sub) if first else len(sub) - over
        first = False
        print(f"POSES_PROGRESS {min(DONE[0], TOTAL)}/{TOTAL}  cameras + depth: chunk {k}, {len(sub)} frames ({time.time() - t0:.1f}s)", flush=True)
        if s0 + chunk >= F:
            break
        s0 += chunk - over


def geo_of(pre, k):
    z = np.load(TMP / f"{pre or 'single_'}{k:04d}.npz")
    return z["depth"], z["conf"]


# ------------------------------------------------------------------------------------------------ the road under the cameras
def up_axis_of(names):
    """which camera axis points up: where the sky is (MoGe invalid pixels). Returns (axis vector in the camera frame, label, how)"""
    cand = {"+x": np.array([1.0, 0, 0]), "-x": np.array([-1.0, 0, 0]), "+y": np.array([0, 1.0, 0]), "-y": np.array([0, -1.0, 0])}
    sky, val = [], []
    if DEPTHS is not None:
        yy, xx = np.mgrid[0:H_:16, 0:W_:16]
        d = np.stack([(xx - CX) / FX, (yy - CY) / FX, np.ones_like(xx, float)], -1); d /= np.linalg.norm(d, axis=-1, keepdims=True)
        for n in names[:: max(1, len(names) // 60)]:
            md = moge_full(n)
            if md is None:
                continue
            sk = md[0:H_:16, 0:W_:16] <= 0
            if sk.mean() > 0.02 and (~sk).mean() > 0.2:
                sky.append(d[sk].mean(0)); val.append(d[~sk].mean(0))
    if len(sky) >= 5:
        v = np.mean(sky, 0) - np.mean(val, 0)
        best = max(cand, key=lambda k: float(v @ cand[k]))
        return cand[best], best, "sky"
    return None, None, "no sky seen"


def ground_of(ch, pre, k, upax, rs):
    """road plane of a chunk in its own coordinates. Adds to the chunk: gn (unit normal, pointing up), h (median camera height), gq (quality 0..1)"""
    depth, conf = geo_of(pre, k)
    N, h, w = depth.shape; st = 4
    rx, ry = rays_of(h, w); rx = rx[::st, ::st]; ry = ry[::st, ::st]
    R, c = ch["R"], ch["c"]
    up = unit(np.einsum("nji,j->ni", R, upax).mean(0))
    cfs = conf[:, ::st, ::st].astype(np.float32); thr = np.percentile(cfs, 40)
    P = []
    for f in range(N):
        z = depth[f, ::st, ::st].astype(np.float32); ok = (cfs[f] > thr) & (z > 0)
        Xw = np.stack([rx * z, ry * z, z], -1)[ok].astype(np.float64) @ R[f] + c[f]
        rel = Xw - c[f]; down = -(rel @ up); hor = np.linalg.norm(rel + np.outer(down, up), axis=1)
        if (down > 0).sum() < 50:
            continue
        P.append(Xw[(down > 0.15 * hor) & (hor < 6 * np.median(down[down > 0]))])
    ch["gn"], ch["h"], ch["gq"] = up, float("nan"), 0.0
    P = np.concatenate(P) if P else np.zeros((0, 3))
    if len(P) < 500:
        return
    if len(P) > 40000:
        P = P[rs.choice(len(P), 40000, replace=False)]
    tol = 0.04 * float(np.median(-((P - c.mean(0)) @ up)))
    best = (0, None)
    for _ in range(300):
        a, b, d3 = P[rs.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, d3 - a); nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn * np.sign((n @ up) or 1.0)
        if n @ up < 0.94:                                           # the road, not a wall: within 20 degrees of "up"
            continue
        cnt = int((np.abs((P - a) @ n) < tol).sum())
        if cnt > best[0]:
            best = (cnt, (n, a))
    if best[1] is None:
        return
    n, a = best[1]
    for _ in range(3):
        inl = np.abs((P - a) @ n) < tol
        if inl.sum() < 200:
            return
        Q = P[inl]; a = Q.mean(0); _, _, Vt = np.linalg.svd(Q - a, full_matrices=False); n = Vt[2] * np.sign((Vt[2] @ up) or 1.0)
    hc = (c - a) @ n; hm = float(np.median(hc))
    if hm <= 0:
        return
    flat = float(np.std(hc)) / hm                                   # all cameras of a rig ride at one height
    ch["gn"], ch["h"] = n, hm
    ch["gq"] = float(np.clip(inl.mean() / 0.8, 0, 1) * np.clip(1.5 - flat / 0.06, 0, 1)) if inl.mean() > 0.4 else 0.0


# ------------------------------------------------------------------------------------------------ MoGe: the size of the world
def sift_of(name):
    g = cv2.imread(str(IMAGES / name), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))                                # own object per call: CLAHE is not thread-safe
    g = clahe.apply(cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))         # dark covert footage: lift the local contrast
    kp, de = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02).detectAndCompute(g, None)
    if de is None or len(kp) < 50:
        return None
    return np.array([k.pt for k in kp], np.float64) * 2.0, de


def baseline_in_moge_units(pi, pj, Rrel, bhat, Di, Dj):
    """matched pixels of frames i, j with X_j = Rrel X_i + L * bhat -> per-point estimates of L"""
    ri = np.column_stack([(pi[:, 0] - CX) / FX, (pi[:, 1] - CY) / FX, np.ones(len(pi))])
    rj = np.column_stack([(pj[:, 0] - CX) / FX, (pj[:, 1] - CY) / FX, np.ones(len(pj))])
    a = ri @ Rrel.T
    n = np.cross(np.tile(bhat, (len(a), 1)), a)
    ep = np.abs((rj * n).sum(1)) / (np.linalg.norm(n, axis=1) * np.linalg.norm(rj, axis=1) + 1e-12)
    par = np.degrees(np.arccos(np.clip((a * rj).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(rj, axis=1), -1, 1)))
    ok = (ep < 0.005) & (par > 1.0)
    if ok.sum() < 12:
        return np.zeros(0)
    a, rj, pi, pj = a[ok], rj[ok], pi[ok], pj[ok]
    A11 = (rj * rj).sum(1); A12 = -(rj * a).sum(1); A22 = (a * a).sum(1); b1 = rj @ bhat; b2 = -(a @ bhat)
    det = A11 * A22 - A12 * A12
    zj = (b1 * A22 - A12 * b2) / det; zi = (A11 * b2 - A12 * b1) / det
    good = (zi > 0) & (zj > 0) & np.isfinite(zi) & np.isfinite(zj)
    di = Di[pi[:, 1].astype(int).clip(0, Di.shape[0] - 1), pi[:, 0].astype(int).clip(0, Di.shape[1] - 1)]
    dj = Dj[pj[:, 1].astype(int).clip(0, Dj.shape[0] - 1), pj[:, 0].astype(int).clip(0, Dj.shape[1] - 1)]
    L = np.concatenate([(di / zi)[good & (di > 0)], (dj / zj)[good & (dj > 0)]])
    return L[np.isfinite(L) & (L > 0)]


def moge_scales(chunks, tag):
    """log of (MoGe units per chunk unit) for every chunk with enough triangulated matches, else nan"""
    K = len(chunks); meas = np.full(K, np.nan); npts = np.zeros(K, int)
    if DEPTHS is None:
        return meas, npts
    names = sorted({n for ch in chunks for n in ch["names"]})
    t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        feats = dict(zip(names, ex.map(sift_of, names)))
    print(f"{tag}: SIFT on {len(names)} frames ({time.time() - t0:.0f}s)", flush=True)
    bf = cv2.BFMatcher(cv2.NORM_L2); dcache = {}
    for k, ch in enumerate(chunks):
        est = []
        nm, R, c = ch["names"], ch["R"], ch["c"]
        for gap in GAPS:
            for i in range(0, len(nm) - gap, PAIR_STEP):
                j = i + gap
                fi, fj = feats.get(nm[i]), feats.get(nm[j])
                bv = R[j] @ (c[i] - c[j]); bl = float(np.linalg.norm(bv))
                if fi is None or fj is None or bl < 1e-5:
                    continue
                mm = bf.knnMatch(fi[1], fj[1], k=2)
                mm = [m for m, n2 in mm if m.distance < 0.8 * n2.distance]
                if len(mm) < 30:
                    continue
                for n in (nm[i], nm[j]):
                    if n not in dcache:
                        dcache[n] = moge_full(n)
                if dcache[nm[i]] is None or dcache[nm[j]] is None:
                    continue
                L = baseline_in_moge_units(fi[0][[m.queryIdx for m in mm]], fj[0][[m.trainIdx for m in mm]], R[j] @ R[i].T, bv / bl,
                                           dcache[nm[i]], dcache[nm[j]])
                est.extend((L / bl).tolist())
        for n in nm[: len(nm) - ch["over"]]:
            dcache.pop(n, None)
        if len(est) >= 40:
            meas[k] = float(np.log(np.median(est))); npts[k] = len(est)
        if k % 5 == 0 or k == K - 1:
            print(f"POSES_PROGRESS {tag} world size: chunk {k + 1}/{K}" + (f", {npts[k]} matches" if np.isfinite(meas[k]) else ", no usable matches"), flush=True)
    return meas, npts


def solve_scales(chunks, meas):
    """x_k = log lam_k: height anchor (x_k = log H - log h_k) + seam depth ratios (x_{k+1} - x_k = rel) + a weak MoGe term per chunk"""
    K = len(chunks)
    lh = np.array([np.log(ch["h"]) if ch["gq"] > 0 else np.nan for ch in chunks]); gq = np.array([ch["gq"] for ch in chunks])
    both = np.isfinite(lh) & np.isfinite(meas)
    if both.sum() >= 1:
        logH = float(np.median((meas + lh)[both])); spread = float(np.median(np.abs((meas + lh)[both] - logH)))
    else:
        logH, spread = 0.0, float("nan")                          # no MoGe (or no road anywhere): one unit = the camera height
    A = np.zeros((K, K)); y = np.zeros(K)
    for k in range(K):
        if np.isfinite(lh[k]):
            w = gq[k] / SIG_H ** 2; A[k, k] += w; y[k] += w * (logH - lh[k])
        if np.isfinite(meas[k]):
            w = (1.0 if np.isfinite(lh).any() else 25.0) / SIG_M ** 2; A[k, k] += w; y[k] += w * meas[k]
    for k in range(K - 1):
        r = chunks[k + 1]["rel"]
        if np.isfinite(r):
            w = 1.0 / SIG_R ** 2
            A[k, k] += w; A[k + 1, k + 1] += w; A[k, k + 1] -= w; A[k + 1, k] -= w
            y[k] -= w * r; y[k + 1] += w * r
    x = np.linalg.solve(A + 1e-9 * np.eye(K), y)
    return np.exp(x), float(np.exp(logH)), spread


def chain(chunks, lam):
    """metric chunks -> one track. Seam: rotation = mean over the shared cameras, translation = their centres.
    Returns per-frame names, centres, rotations, owner (chunk, index in chunk) and the chunk->world rotation of every chunk."""
    names = []; C = []; R = []; own = []; Gs = []
    G = (np.eye(3), np.zeros(3)); seams = []
    for k, ch in enumerate(chunks):
        ck = lam[k] * ch["c"]
        if k > 0:
            prev = chunks[k - 1]; pc = lam[k - 1] * prev["c"]
            shared = [n for n in ch["names"] if n in prev["names"]]
            i1 = [prev["names"].index(n) for n in shared]; i2 = [ch["names"].index(n) for n in shared]
            Rs = so3_mean([prev["R"][a].T @ ch["R"][b] for a, b in zip(i1, i2)])        # chunk k frame -> chunk k-1 frame
            ts = np.mean([pc[a] - Rs @ ck[b] for a, b in zip(i1, i2)], 0)
            Gr, Gt = G
            G = (Gr @ Rs, Gr @ ts + Gt)
            gap = [np.linalg.norm((Rs @ ck[b] + ts) - pc[a]) for a, b in zip(i1, i2)]
            ang = [rot_angle((prev["R"][a].T @ ch["R"][b]) @ Rs.T) for a, b in zip(i1, i2)]
            seams.append({"chunk": k, "shared": len(shared), "centre_gap_med": round(float(np.median(gap)), 4), "rot_spread_deg": round(float(np.median(ang)), 3)})
        Gr, Gt = G; Gs.append(Gr)
        skip = (ch["over"] + 1) // 2 if k > 0 else 0              # shared frames: the first half stays with the previous chunk
        cut = len(ch["names"]) - (chunks[k + 1]["over"] // 2 if k + 1 < len(chunks) else 0)
        for i in range(skip, cut):
            names.append(ch["names"][i]); C.append(Gr @ ck[i] + Gt); R.append(ch["R"][i] @ Gr.T); own.append((k, i))
    return names, np.array(C), np.array(R), own, Gs, seams


def owners(chunks):
    out = []
    for k, ch in enumerate(chunks):
        skip = (ch["over"] + 1) // 2 if k > 0 else 0
        cut = len(ch["names"]) - (chunks[k + 1]["over"] // 2 if k + 1 < len(chunks) else 0)
        out += [(k, i) for i in range(skip, cut)]
    return out


def camera_vectors(chunks):
    """what a car-mounted camera sees of the car, per frame and in its own frame: the road normal and the driving direction.
    Returns {"n": (times, vectors), "v": (times, vectors)} sorted by time."""
    out = {"n": ([], []), "v": ([], [])}
    for ch in chunks:
        if ch["gq"] <= 0:
            continue
        tt = [ts_of(n) for n in ch["names"]]
        c = ch["c"] / ch["h"]                                      # in camera heights
        for f in range(len(c)):
            out["n"][0].append(tt[f]); out["n"][1].append(ch["R"][f] @ ch["gn"])
            if 0 < f < len(c) - 1:
                d = c[f + 1] - c[f - 1]; step = float(np.linalg.norm(d))
                if step > 0.25 and tt[f + 1] - tt[f - 1] < 1.0:    # the rig really moves here
                    out["v"][0].append(tt[f]); out["v"][1].append(ch["R"][f] @ d / step)
    res = {}
    for key, (t, v) in out.items():
        o = np.argsort(t); res[key] = (np.array(t, float)[o], np.array(v, float).reshape(-1, 3)[o])
    return res


def rig_rotation(va, vb):
    """rotation taking camera-a-frame vectors to camera-b-frame vectors: the road normal and the driving direction of the SAME
    moments (so turns, where a camera ahead of the rear axle drifts sideways, do not bias it). Wahba's problem, outliers dropped."""
    A, B, Wt = [], [], []
    for key, wgt in (("n", 1.0), ("v", 1.0)):
        ta_, a = va[key]; tb_, b = vb[key]
        if len(ta_) < 2 or not len(tb_):
            continue
        j = np.searchsorted(ta_, tb_).clip(1, len(ta_) - 1)
        ok = (tb_ >= ta_[0]) & (tb_ <= ta_[-1]) & (ta_[j] - ta_[j - 1] < 0.6)
        al = ((tb_ - ta_[j - 1]) / np.maximum(ta_[j] - ta_[j - 1], 1e-9))[:, None]
        ai = (1 - al) * a[j - 1] + al * a[j]; ai /= np.linalg.norm(ai, axis=1, keepdims=True) + 1e-18
        A.append(ai[ok]); B.append(b[ok]); Wt.append(np.full(int(ok.sum()), wgt));
    if not A or min(len(x) for x in A) < 12 or len(A) < 2:
        return None
    nn, nv = len(A[0]), len(A[1]); A = np.concatenate(A); B = np.concatenate(B); Wt = np.concatenate(Wt)
    keep = np.ones(len(A), bool)
    for _ in range(3):
        U, _, Vt = np.linalg.svd((B[keep] * Wt[keep, None]).T @ A[keep])
        RX = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        ang = np.degrees(np.arccos(np.clip(((A @ RX.T) * B).sum(1), -1, 1)))
        keep = ang < max(6.0, 2.5 * float(np.median(ang)))
    return RX, nn, nv, float(np.median(ang))


# ------------------------------------------------------------------------------------------------ per camera
t_all = time.time()
cams = {}
rs = np.random.RandomState(0)
ref = max(groups, key=lambda k: len(groups[k]))                    # the longest track carries the world frame
for pre, names in groups.items():
    tag = f"camera '{pre or 'single'}'"
    print(f"--- {tag}: {len(names)} frames", flush=True)
    cache = OUT / f"_chunks_{pre or 'single'}.json"
    chunks = None
    if os.environ.get("NAST_POSES_REUSE") == "1" and cache.exists():
        chunks = [dict(c, R=np.array(c["R"]), c=np.array(c["c"])) for c in json.loads(cache.read_text())]
        if all((TMP / f"{pre or 'single_'}{k:04d}.npz").exists() for k in range(len(chunks))) and "seg" in chunks[0]:
            DONE[0] += len(names)
            print(f"{tag}: GPU pass reused from {cache.name} ({len(chunks)} chunks)", flush=True)
        else:
            chunks = None
    if chunks is None:
        chunks = run_chunks(names, pre, split=pre != ref)
        cache.write_text(json.dumps([dict(c, R=c["R"].tolist(), c=c["c"].tolist()) for c in chunks]))
    cams[pre] = {"chunks": chunks, "all": names}
proc = None
torch.cuda.empty_cache()

for pre, cam in cams.items():
    tag = f"camera '{pre or 'single'}'"
    upax, axis, how = up_axis_of(cam["all"])
    if upax is None:                                               # no sky: the axis that stays most constant along the track (sign = a guess)
        cand = {"-x": np.array([-1.0, 0, 0]), "-y": np.array([0, -1.0, 0])}
        axis = max(cand, key=lambda a: float(np.mean([np.linalg.norm(np.mean([Rk.T @ cand[a] for Rk in ch["R"]], 0)) for ch in cam["chunks"]])))
        upax, how = cand[axis], "axis constancy (no sky seen: the sign is a guess)"
    t0 = time.time()
    for k, ch in enumerate(cam["chunks"]):
        ground_of(ch, pre, k, upax, rs)
    good = [ch for ch in cam["chunks"] if ch["gq"] > 0]
    print(f"{tag}: road plane in {len(good)}/{len(cam['chunks'])} chunks ({time.time() - t0:.0f}s), up = camera {axis} by {how}", flush=True)
    cam.update(up_axis=axis, up_how=how, upax=upax)

cam = cams[ref]; chunks = cam["chunks"]; tag = f"camera '{ref or 'single'}'"
meas, npts = moge_scales(chunks, tag)
lam, Hw, spread = solve_scales(chunks, meas)
nm, C, R, own, Gs, seams = chain(chunks, lam)
gw = [Gs[k] @ ch["gn"] * ch["gq"] for k, ch in enumerate(chunks) if ch["gq"] > 0]
up_w = unit(np.sum(gw, 0)) if gw else unit(np.mean([Rk.T @ cam["upax"] for Rk in R], 0))
cam.update(names=nm, C=C, R=R, own=own, sig=np.array([lam[k] for k, _ in own]), lam=lam, seams=seams, H=Hw,
           path_len=float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum()))
hs = np.array([lam[k] * ch["h"] for k, ch in enumerate(chunks) if ch["gq"] > 0])
print(f"{tag}: path {cam['path_len']:.1f}, chunks {len(chunks)}, camera height {Hw:.4f} world units (MoGe spread {spread * 100:.0f} %, {int(np.isfinite(meas).sum())} chunks measured; "
      f"after the solve the height varies by {np.std(np.log(hs)) * 100 if len(hs) else 0:.1f} %), VGGT fx {np.median([c['fx'] for c in chunks]):.0f} px vs calibrated {FX:.0f}", flush=True)

# ------------------------------------------------------------------------------------------------ the other cameras follow the rig
reg = {}
ta = [ts_of(n) for n in nm]
vec_ref = camera_vectors(chunks)
for pre, cb in cams.items():
    if pre == ref:
        continue
    tagb = f"camera '{pre}'"; chb = cb["chunks"]; ownb = owners(chb)
    tb_all = {n: ts_of(n) for ch in chb for n in ch["names"]}
    if any(v is None for v in ta) or any(v is None for v in tb_all.values()):
        print(f"{tagb}: no timestamps in the names - cannot be tied to the rig, left out", flush=True); continue
    rr = rig_rotation(vec_ref, camera_vectors(chb))
    if rr is None:
        print(f"{tagb}: the rig never moved over a visible road while both cameras ran - cannot be tied to '{ref}', left out", flush=True); continue
    RX = rr[0]                                                      # reference camera frame -> this camera frame
    tarr = np.array(ta, float); dta = float(np.median(np.diff(tarr)))

    def rig_at(t):
        """reference camera pose at time t, or None when t is not bracketed by two close reference frames"""
        j = int(np.searchsorted(tarr, t))
        if j <= 0 or j >= len(tarr) or tarr[j] - tarr[j - 1] > max(0.6, 4 * dta):
            return None
        a = (t - tarr[j - 1]) / max(tarr[j] - tarr[j - 1], 1e-9)
        return rot_interp(R[j - 1], R[j], a), (1 - a) * C[j - 1] + a * C[j]

    ell = np.array(RIG_LEVER) * Hw                                  # where this camera sits, in the reference camera's frame
    Kb = len(chb); Q = [None] * Kb; eqs = []                        # eqs: (chunk, Q c, predicted centre)
    tied = [[(f, g) for f, g in ((f, rig_at(tb_all[n])) for f, n in enumerate(ch["names"])) if g is not None] for ch in chb]
    Qrig = [so3_mean([(RX @ g[0]).T @ ch["R"][f] for f, g in got]) if len(got) >= 4 else None for ch, got in zip(chb, tied)]     # chunk -> world
    # the rig gives the rotation of a chunk absolutely but to a degree or two; the cameras shared by neighbouring chunks give their
    # relative rotation to 0.1 degree. So: chain by the seams inside a run of chunks, then pull the run to the rig over a window.
    Qc = [None] * Kb; run = [0] * Kb
    for k in range(Kb):
        link = None
        if k > 0 and Qc[k - 1] is not None and chb[k]["over"] > 0:
            sh = [n for n in chb[k]["names"] if n in chb[k - 1]["names"]]
            if len(sh) >= 2:
                link = so3_mean([chb[k - 1]["R"][chb[k - 1]["names"].index(n)].T @ chb[k]["R"][chb[k]["names"].index(n)] for n in sh])
        if link is not None:
            Qc[k] = Qc[k - 1] @ link; run[k] = run[k - 1]
        else:
            Qc[k] = Qrig[k]; run[k] = (run[k - 1] + 1) if k else 0
    for k in range(Kb):
        if Qc[k] is None:
            continue
        nb = [j for j in range(max(0, k - 4), min(Kb, k + 5)) if run[j] == run[k] and Qrig[j] is not None and Qc[j] is not None]
        Q[k] = so3_mean([len(tied[j]) * Qrig[j] @ Qc[j].T for j in nb]) @ Qc[k] if nb else None
    for k, (ch, got) in enumerate(zip(chb, tied)):
        if Q[k] is not None and len(got) >= 4:
            eqs += [(k, Q[k] @ ch["c"][f], g[1] + g[0].T @ ell) for f, g in got]
        else:
            Q[k] = None
    solved = [k for k in range(Kb) if Q[k] is not None]
    if not solved:
        print(f"{tagb}: no frame of it falls between frames of '{ref}' - left out", flush=True); continue
    # every chunk: its own scale and shift onto the predicted track. The scale is solved in the log domain, like the reference
    # camera's: the chunk's own fit where the rig moved + the seam depth ratios + the height of THIS camera above the road.
    s_b = np.ones(Kb); tau = np.zeros((Kb, 3)); ls0 = np.full(Kb, np.nan); w0 = np.zeros(Kb)
    for k in solved:
        q = np.array([e[1] for e in eqs if e[0] == k]); p = np.array([e[2] for e in eqs if e[0] == k]); wq = np.ones(len(q)); sk = den = 0.0
        for _ in range(3):
            qm = (q * wq[:, None]).sum(0) / wq.sum(); pm = (p * wq[:, None]).sum(0) / wq.sum()
            den = float((wq * ((q - qm) ** 2).sum(1)).sum())
            if den < 1e-12:
                break
            sk = float((wq * ((q - qm) * (p - pm)).sum(1)).sum() / den)
            r_ = np.linalg.norm(sk * (q - qm) + pm - p, axis=1); wq = np.minimum(1.0, 2.5 * max(float(np.median(r_)), 1e-4) / np.maximum(r_, 1e-9))
        ext = sk * np.sqrt(den) if den > 1e-12 else 0.0              # how far the rig went inside this chunk, world units
        if sk > 0 and ext > 0.15:
            ls0[k] = np.log(sk); w0[k] = 1.0 / max(0.02 / ext, 0.05) ** 2                   # not better than 5 %: a chunk's translations and depth differ by that
    lhb = np.array([np.log(ch["h"]) if ch["gq"] > 0 else np.nan for ch in chb])
    both = np.isfinite(ls0) & np.isfinite(lhb)
    HB = float(np.exp(np.median((ls0 + lhb)[both]))) if both.sum() >= 2 else None
    A_ = np.zeros((Kb, Kb)); y_ = np.zeros(Kb)
    for k in range(Kb):
        if np.isfinite(ls0[k]):
            A_[k, k] += w0[k]; y_[k] += w0[k] * ls0[k]
        if HB is not None and np.isfinite(lhb[k]):
            w = chb[k]["gq"] / SIG_H ** 2; A_[k, k] += w; y_[k] += w * (np.log(HB) - lhb[k])
        if k + 1 < Kb and np.isfinite(chb[k + 1]["rel"]):
            w = 1.0 / SIG_R ** 2; r = chb[k + 1]["rel"]
            A_[k, k] += w; A_[k + 1, k + 1] += w; A_[k, k + 1] -= w; A_[k + 1, k] -= w; y_[k] -= w * r; y_[k + 1] += w * r
    if not np.isfinite(ls0).any():
        print(f"{tagb}: the rig never moved while both cameras ran - its scale is unknown, left out", flush=True); continue
    s_b = np.exp(np.linalg.solve(A_ + 1e-9 * np.eye(Kb), y_))
    res = []
    for k in solved:
        q = np.array([e[1] for e in eqs if e[0] == k]); p = np.array([e[2] for e in eqs if e[0] == k])
        tau[k] = np.median(p - s_b[k] * q, 0); res += np.linalg.norm(s_b[k] * q + tau[k] - p, axis=1).tolist()
    res = np.array(res)
    if os.environ.get("NAST_POSES_DEBUG") == "1":
        for k in range(Kb):
            rk = [np.linalg.norm(s_b[k] * e[1] + tau[k] - e[2]) for e in eqs if e[0] == k]
            print(f"  {tagb} chunk {k}: track-fit scale {np.exp(ls0[k]) if np.isfinite(ls0[k]) else float('nan'):.3f} (weight {w0[k]:.0f}), final {s_b[k]:.3f}, h {chb[k]['h']:.4f} q {chb[k]['gq']:.2f}, "
                  f"s*h {s_b[k] * chb[k]['h']:.4f}, rel {chb[k]['rel']:.3f}, frames tied {len(rk)}, resid med {np.median(rk) if rk else float('nan'):.3f} max {np.max(rk) if rk else float('nan'):.3f}", flush=True)
    for step in (1, -1):                                            # chunks with no time overlap with the reference camera: chained to a solved neighbour
        for k in (range(Kb) if step == 1 else range(Kb - 1, -1, -1)):
            j = k - step
            if Q[k] is not None or not (0 <= j < Kb) or Q[j] is None:
                continue
            a, b = chb[j], chb[k]
            sh = [n for n in b["names"] if n in a["names"]]
            rel = chb[max(j, k)]["rel"]
            if len(sh) < 2 or not np.isfinite(rel):
                continue
            ia = [a["names"].index(n) for n in sh]; ib = [b["names"].index(n) for n in sh]
            s_b[k] = s_b[j] * np.exp(rel if k > j else -rel)
            Q[k] = Q[j] @ so3_mean([a["R"][x_].T @ b["R"][y_] for x_, y_ in zip(ia, ib)])
            tau[k] = np.mean([s_b[j] * Q[j] @ a["c"][x_] + tau[j] - s_b[k] * Q[k] @ b["c"][y_] for x_, y_ in zip(ia, ib)], 0)
    keep = [(k, i) for k, i in ownb if Q[k] is not None and s_b[k] > 0]
    cb.update(names=[chb[k]["names"][i] for k, i in keep], C=np.array([s_b[k] * Q[k] @ chb[k]["c"][i] + tau[k] for k, i in keep]),
              R=np.array([chb[k]["R"][i] @ Q[k].T for k, i in keep]), own=keep, sig=np.array([s_b[k] for k, _ in keep]), lam=s_b, H=HB)
    cb["path_len"] = float(np.linalg.norm(np.diff(cb["C"], axis=0), axis=1).sum()) if len(keep) > 1 else 0.0
    lost = len(ownb) - len(keep)
    reg[pre] = {"to": ref, "rig_rotation_deg": [round(float(v), 2) for v in np.degrees(cv2.Rodrigues(RX)[0].ravel())], "rig_rotation_pairs": {"road_normal": rr[1], "driving_direction": rr[2]},
                "rig_rotation_resid_deg": round(rr[3], 2), "lever_arm_world": [round(float(v), 4) for v in ell],
                "chunks": Kb, "chunks_tied_by_time": len(solved), "frames_without_pose": lost, "fit_resid_med": round(float(np.median(res)), 4), "fit_resid_p90": round(float(np.percentile(res, 90)), 4),
                "camera_height": None if HB is None else round(HB, 4)}
    print(f"{tagb} -> '{ref}': rig rotation {reg[pre]['rig_rotation_deg']} deg from {rr[1]} road-normal and {rr[2]} driving-direction pairs (residual {rr[3]:.1f} deg), "
          f"{len(solved)}/{Kb} chunks tied by time, fit residual median {np.median(res):.3f} (p90 {np.percentile(res, 90):.3f}), {lost} frames without a pose", flush=True)

# ------------------------------------------------------------------------------------------------ depth in world units, per frame
(GEO / "depth_geo").mkdir(parents=True, exist_ok=True); (GEO / "depth_geo_conf").mkdir(parents=True, exist_ok=True)
t0 = time.time(); n_png = 0
for pre, cam in cams.items():
    if "own" not in cam:
        continue
    by_chunk = {}
    for (k, i), s in zip(cam["own"], cam["sig"]):
        by_chunk.setdefault(k, []).append((i, float(s)))
    for k, items in by_chunk.items():
        depth, conf = geo_of(pre, k)
        cfl = conf.astype(np.float32); grid = cfl[:, ::4, ::4].ravel()
        grid = np.sort(grid[grid > 1.02]) if (grid > 1.02).sum() > 1000 else np.sort(grid)     # rank among the pixels VGGT says anything about (sky sits at 1.0)
        for i, s in items:
            name = cam["chunks"][k]["names"][i]
            z = depth[i].astype(np.float32) * s
            md = moge_full(name)
            if md is not None:                                     # MoGe knows the sky; VGGT invents a depth there
                z[cv2.resize((md > 0).astype(np.uint8), (z.shape[1], z.shape[0]), interpolation=cv2.INTER_NEAREST) == 0] = 0
            rank = (np.searchsorted(grid, cfl[i].ravel()) / max(len(grid), 1) * 255).clip(0, 255).astype(np.uint8).reshape(z.shape)
            cv2.imwrite(str(GEO / "depth_geo" / (Path(name).stem + ".png")), np.clip(z * 1000.0, 0, 65535).astype(np.uint16))
            cv2.imwrite(str(GEO / "depth_geo_conf" / (Path(name).stem + ".png")), rank)
            n_png += 1
        if n_png % 96 < len(items):
            print(f"POSES_PROGRESS depth in world units: {n_png}/{sum(len(c.get('own', [])) for c in cams.values())} frames", flush=True)
print(f"depth_geo: {n_png} frames ({time.time() - t0:.0f}s)", flush=True)

# ------------------------------------------------------------------------------------------------ the pack
frames = []
for pre, cam in cams.items():
    if "own" not in cam:
        continue
    for n, c, Rk in zip(cam["names"], cam["C"], cam["R"]):
        frames.append({"i": frame_id(n), "name": n, "p": [float(v) for v in c], "q": quat_wxyz(Rk)})
frames.sort(key=lambda f: (ts_of(f["name"]) or 0, f["name"]))
allC = np.array([f["p"] for f in frames]); lo, hi = allC.min(0), allC.max(0)
ext_m = float((hi - lo).max()) or 1.0
intr = {"fx": FX, "fy": FX, "cx": CX, "cy": CY, "w": W_, "h": H_}
poses = dict(intr, up=[float(v) for v in up_w], frames=frames)
meta = {"name": f"poses by the VGGT-Omega chain ({IMAGES.parent.name})", "count": 0, "bbox": {"min": (lo - 1.5).tolist(), "max": (hi + 1.5).tolist()},
        "extent": ext_m + 3.0, "spacing": 0.01, "cell": 1.0, "has_nrm": False, "intrinsics": intr, "up": [float(v) for v in up_w],
        "units": "MoGe depth units (metres for a MoGe-2 dump)" if DEPTHS else "camera heights (no depth dump)", "pose_source": "vggto_poses", "depth_source": "depth_geo"}
(OUT / "poses.json").write_text(json.dumps(poses))
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
holes = {}
for pre, cam in cams.items():
    tt = [ts_of(n) for n in cam.get("names", [])]
    if len(tt) > 2 and all(v is not None for v in tt):
        dd = np.diff(np.array(tt, float)); holes[pre] = int((dd > max(1.0, 6.0 * float(np.median(dd)))).sum())
report = {"frames": len(frames), "seconds": round(time.time() - t_all), "chunk": CHUNK, "overlap": OVER, "res": RES, "units": meta["units"], "reference_camera": ref or "single",
          "camera_height_world": round(Hw, 4), "camera_height_moge_spread": None if not np.isfinite(spread) else round(spread, 3),
          "cameras": {pre or "single": {"frames": len(c.get("names", [])), "path_len": round(c.get("path_len", 0.0), 2), "up_axis": c["up_axis"], "up_by": c["up_how"],
                                        "time_holes": holes.get(pre, 0), "chunks": len(c["chunks"]), "chunks_with_road_plane": int(sum(ch["gq"] > 0 for ch in c["chunks"])),
                                        "scale_min_max": [round(float(np.min(c["lam"])), 4), round(float(np.max(c["lam"])), 4)] if "lam" in c else None,
                                        "vggt_fx_median": round(float(np.median([ch["fx"] for ch in c["chunks"]])), 1),
                                        "seam_centre_gap_med": round(float(np.median([s["centre_gap_med"] for s in c["seams"]])), 4) if c.get("seams") else None,
                                        "seam_rot_spread_deg_med": round(float(np.median([s["rot_spread_deg"] for s in c["seams"]])), 3) if c.get("seams") else None}
                      for pre, c in cams.items()},
          "world_size_matches_per_chunk_median": int(np.median(npts[np.isfinite(meas)])) if np.isfinite(meas).any() else 0,
          "registration": reg, "calibrated_fx": round(FX, 1), "up": [round(float(v), 4) for v in up_w], "depth_geo_frames": n_png}
(OUT / "poses_report.json").write_text(json.dumps(report, indent=1))
if os.environ.get("NAST_POSES_KEEP") != "1" and os.environ.get("NAST_POSES_REUSE") != "1":
    shutil.rmtree(TMP, ignore_errors=True)
    for q in OUT.glob("_chunks_*.json"):
        q.unlink()
print(f"POSES_DONE {OUT} frames={len(frames)} extent={ext_m:.1f} total {time.time() - t_all:.0f}s", flush=True)
