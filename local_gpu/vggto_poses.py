"""Camera poses for a NEW recording — no COLMAP, no GNSS: VGGT-Omega cameras on overlapping chunks, chained into one track
whose scale agrees with the MoGe depth dump.

The rest of the pipeline (ROI solve, object views, vggto_local map build, moge_layer) needs a pose per frame. A fresh
take has none, so this pass makes them on the local GPU:
  1. GPU: every camera (name prefix A_/B_/...) on its own, chunks of CHUNK frames with OVERLAP shared frames. Only the camera
     head is used (rotations + centres of a chunk are consistent; its depth maps are NOT scale-consistent with them).
  2. CPU: the length of the VGGT baselines in MoGe units. SIFT matches between frames a few steps apart, filtered with the
     VGGT relative pose (epipolar), triangulated with a unit baseline -> depth in baseline units; the MoGe depth of the same
     pixel divided by that = the baseline in MoGe units (metres for a MoGe-2 dump). Per chunk: median over all matches.
     Neighbouring chunks also share camera centres -> a precise RELATIVE scale; both go into one small least-squares problem,
     so a chunk with few matches (or a standing rig) inherits its scale from the neighbours.
  3. Seams: rotation = average of the shared cameras' rotations, translation = their centres. No depth involved.
  4. World "up": the camera axis that points at the sky (MoGe invalid pixels); the second camera of a back-to-back rig is
     registered onto the first by the shared trajectory (same timestamps -> same place) plus the up vector (fixes the roll on
     a straight drive).
Output = the pack the pipeline already understands: <out>/poses.json + <out>/meta.json (format of inspector/scene_base)
and <out>/poses_report.json with the scale / seam / registration numbers.

Usage:
  python vggto_poses.py <ckpt> <images_dir> <depth_dir|none> <out_dir> [chunk=24] [overlap=6] [res=512] [max_frames=0]
"""
import datetime
import json
import os
import re
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
FX_RATIO = 641.601591290442 / 1224.0              # calibrated lens of the rig cameras: fx / image width
GAPS = (4, 8)                                      # frame gaps of the triangulation pairs
PAIR_STEP = 5                                      # every 5th frame starts a pair: ~8 pairs per chunk, thousands of matches


# ------------------------------------------------------------------------------------------------ small geometry helpers
def umeyama(A, B, with_scale=True):
    """sim3 (s, R, t) with  B ~ s * R @ A + t  (rows are points)"""
    muA, muB = A.mean(0), B.mean(0)
    Ac, Bc = A - muA, B - muB
    U, S, Vt = np.linalg.svd(Ac.T @ Bc / len(A))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    s = float(np.trace(np.diag(S) @ D) / max((Ac ** 2).sum(), 1e-18) * len(A)) if with_scale else 1.0
    return s, R, muB - s * R @ muA


def robust_sim3(A, B, with_scale=True, iters=3):
    keep = np.ones(len(A), bool); med = 0.0
    for _ in range(iters):
        s, R, t = umeyama(A[keep], B[keep], with_scale)
        res = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)
        med = float(np.median(res[keep]))
        keep = res < max(3.0 * med, 1e-9)
        if keep.sum() < 20:
            break
    return s, R, t, med, int(keep.sum())


def so3_mean(Rs):
    U, _, Vt = np.linalg.svd(np.sum(Rs, 0))
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


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
      f"depth={'yes' if DEPTHS else 'NO (the track will not be metric)'}", flush=True)

proc = None
DONE = [0]


def model():
    global proc
    if proc is None:
        from core.vggt_omega_processor import VGGTOmegaProcessor
        proc = VGGTOmegaProcessor(model_path=CKPT, device="cuda", image_resolution=RES)
        free, total = torch.cuda.mem_get_info()
        print(f"model loaded - VRAM {total / 2**30:.1f} GB total, {free / 2**30:.1f} GB free", flush=True)
    return proc


def run_chunks(names):
    """GPU pass of one camera: list of chunks {s0, names, R (N,3,3) world->camera, c (N,3) centres} in each chunk's own frame"""
    F = len(names); chunks = []; chunk = CHUNK; s0 = 0
    while s0 < F:
        sub = names[s0:s0 + chunk]
        over = min(OVER, chunk // 3)
        if chunks and len(sub) <= over:
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
        wres = int(preds["depth"].shape[3])
        del preds, img_t
        torch.cuda.empty_cache()
        R = E[:, :, :3].astype(np.float64); c = np.array([-e[:, :3].T @ e[:, 3] for e in E], np.float64)
        chunks.append({"s0": s0, "names": sub, "R": R, "c": c, "over": over, "fx": float(np.median(Kp[:, 0, 0]) * W_ / wres)})
        DONE[0] += len(sub) if len(chunks) == 1 else len(sub) - over
        print(f"POSES_PROGRESS {min(DONE[0], TOTAL)}/{TOTAL}  cameras: chunk {len(chunks) - 1}, {len(sub)} frames ({time.time() - t0:.1f}s)", flush=True)
        if s0 + chunk >= F:
            break
        s0 += chunk - over
    return chunks


# ------------------------------------------------------------------------------------------------ scale of the baselines
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


def chunk_scales(chunks, tag):
    """log-scale per chunk: triangulation against MoGe (absolute, noisy) + shared camera centres (relative, precise)"""
    K = len(chunks)
    meas = np.full(K, np.nan); wts = np.zeros(K); npts = np.zeros(K, int)
    if DEPTHS is not None:
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
                e = np.array(est); meas[k] = np.log(np.median(e)); npts[k] = len(e)
                wts[k] = min(len(e), 800) / max(np.subtract(*np.percentile(np.log(e), [75, 25])), 0.05) ** 2
            if k % 5 == 0 or k == K - 1:
                print(f"POSES_PROGRESS {tag} scale: chunk {k + 1}/{K}" + (f", {npts[k]} matches, scale {np.exp(meas[k]):.3f}" if np.isfinite(meas[k]) else ", no usable matches"), flush=True)
    # relative scale of neighbouring chunks from the shared camera centres (only when the rig really moved there)
    rel = np.full(K - 1, np.nan); rw = np.zeros(K - 1)
    for k in range(K - 1):
        a, b = chunks[k], chunks[k + 1]
        shared = [n for n in b["names"] if n in a["names"]]
        if len(shared) < 3:
            continue
        ca = np.array([a["c"][a["names"].index(n)] for n in shared]); cb = np.array([b["c"][b["names"].index(n)] for n in shared])
        la = np.linalg.norm(np.diff(ca, axis=0), axis=1).sum(); lb = np.linalg.norm(np.diff(cb, axis=0), axis=1).sum()
        ext_a = np.linalg.norm(ca.max(0) - ca.min(0)); path_a = np.linalg.norm(np.diff(a["c"], axis=0), axis=1).sum()
        if lb > 1e-6 and la > 1e-6 and ext_a > 0.15 * path_a * len(shared) / len(a["names"]):
            rel[k] = np.log(la / lb)                              # lam_{k+1} = lam_k * la/lb
            rw[k] = 400.0 * min(1.0, ext_a / max(path_a, 1e-9) * len(a["names"]) / len(shared))
    # tridiagonal least squares over x_k = log lam_k
    A = np.zeros((K, K)); y = np.zeros(K)
    for k in range(K):
        if np.isfinite(meas[k]):
            A[k, k] += wts[k]; y[k] += wts[k] * meas[k]
    for k in range(K - 1):
        w = rw[k] if np.isfinite(rel[k]) else 2.0                  # no motion in the seam: weak "same scale as the neighbour" prior
        r = rel[k] if np.isfinite(rel[k]) else 0.0
        A[k, k] += w; A[k + 1, k + 1] += w; A[k, k + 1] -= w; A[k + 1, k] -= w
        y[k] -= w * r; y[k + 1] += w * r
    if not np.isfinite(meas).any():
        A[0, 0] += 1.0                                             # no MoGe at all: chunk 0 keeps its own scale
    x = np.linalg.solve(A + 1e-9 * np.eye(K), y)
    return np.exp(x), meas, npts, rel


def chain(chunks, lam):
    """metric chunks -> one track. Seam: rotation = mean over the shared cameras, translation = their centres."""
    names = []; C = []; R = []
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
            ang = [np.degrees(np.arccos(np.clip((np.trace((prev["R"][a].T @ ch["R"][b]) @ Rs.T) - 1) / 2, -1, 1))) for a, b in zip(i1, i2)]
            seams.append({"chunk": k, "shared": len(shared), "centre_gap_med": round(float(np.median(gap)), 4), "rot_spread_deg": round(float(np.median(ang)), 3)})
        Gr, Gt = G
        skip = (ch["over"] + 1) // 2 if k > 0 else 0              # shared frames: the first half stays with the previous chunk
        cut = len(ch["names"]) - (chunks[k + 1]["over"] // 2 if k + 1 < len(chunks) else 0)
        for i in range(skip, cut):
            names.append(ch["names"][i]); C.append(Gr @ ck[i] + Gt); R.append(ch["R"][i] @ Gr.T)
    return names, np.array(C), np.array(R), seams


def up_of(names, R):
    """the camera axis that points up: where the sky (MoGe invalid pixels) is; without sky the axis most constant along the track"""
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
        v = np.mean(sky, 0) - np.mean(val, 0); how = "sky"
        best = max(cand, key=lambda k: float(v @ cand[k]))
    else:
        how = "axis constancy (no sky seen: the sign is a guess)"
        best = max(("-x", "-y"), key=lambda k: float(np.linalg.norm(np.mean([Rk.T @ cand[k] for Rk in R], 0))))
    up = np.mean([Rk.T @ cand[best] for Rk in R], 0)
    return up / np.linalg.norm(up), best, how


# ------------------------------------------------------------------------------------------------ per camera, then the rig
t_all = time.time()
cams = {}
for pre, names in groups.items():
    tag = f"camera '{pre or 'single'}'"
    print(f"--- {tag}: {len(names)} frames", flush=True)
    cache = OUT / f"_chunks_{pre or 'single'}.json"
    if os.environ.get("NAST_POSES_REUSE") == "1" and cache.exists():
        chunks = [dict(c, R=np.array(c["R"]), c=np.array(c["c"])) for c in json.loads(cache.read_text())]
        print(f"{tag}: GPU pass reused from {cache.name} ({len(chunks)} chunks)", flush=True)
    else:
        chunks = run_chunks(names)
        cache.write_text(json.dumps([dict(c, R=c["R"].tolist(), c=c["c"].tolist()) for c in chunks]))
    lam, meas, npts, rel = chunk_scales(chunks, tag)
    nm, C, R, seams = chain(chunks, lam)
    up, axis, how = up_of(nm, R)
    have = np.isfinite(meas)
    tt = [ts_of(n) for n in nm]
    holes = 0
    if all(v is not None for v in tt) and len(tt) > 2:
        dd = np.diff(np.array(tt, float)); holes = int((dd > max(1.0, 6.0 * float(np.median(dd)))).sum())
        if holes:
            print(f"{tag}: {holes} hole(s) in time (dropped frames or a cut) - the track is only loosely tied across them", flush=True)
    cams[pre] = {"names": nm, "C": C, "R": R, "up": up, "up_axis": axis, "up_how": how, "seams": seams, "lam": lam, "meas": meas, "npts": npts, "holes": holes,
                 "fx": float(np.median([c["fx"] for c in chunks])), "path_len": float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())}
    print(f"{tag}: path {cams[pre]['path_len']:.1f}, chunks {len(chunks)} ({int(have.sum())} with a measured scale, {int(np.isfinite(rel).sum())} seams with a relative scale), "
          f"matches/chunk median {int(np.median(npts[have])) if have.any() else 0}, up = camera {axis} by {how}, VGGT fx {cams[pre]['fx']:.0f} px vs calibrated {FX:.0f}", flush=True)
proc = None
torch.cuda.empty_cache()

ref = max(cams, key=lambda k: len(cams[k]["names"]))              # the longest track carries the world frame
reg = {}
WIN, WSTEP = 48, 8                                                # registration windows along the second camera's track (frames)
for pre, cam in cams.items():
    if pre == ref:
        continue
    ta = [ts_of(n) for n in cams[ref]["names"]]; tb = [ts_of(n) for n in cam["names"]]
    if any(v is None for v in ta) or any(v is None for v in tb):
        print(f"camera '{pre}': no timestamps in the names - left in its own frame", flush=True); continue
    ta = np.array(ta, float); tb = np.array(tb, float)
    Ca_at = np.stack([np.interp(tb, ta, cams[ref]["C"][:, j]) for j in range(3)], 1)     # where the rig was when this camera fired
    inside = (tb >= ta.min()) & (tb <= ta.max())
    Cb, Rb, upb, upa = cam["C"], cam["R"], cam["up"], cams[ref]["up"]
    F = len(Cb)
    dtb = np.diff(tb); cut = np.where(dtb > max(1.0, 6.0 * float(np.median(dtb))))[0] + 1      # holes in time split the track into rigid pieces
    pieces = np.split(np.arange(F), cut)
    Cn = np.zeros_like(Cb); Rn = np.zeros_like(Rb)
    n_all = n_kept = borrowed = 0; all_wins = []
    Q_last = None
    for piece in pieces:
        wins = []
        starts = range(int(piece[0]), max(int(piece[0]) + 1, int(piece[-1]) - WIN + 2), WSTEP) if len(piece) > WIN else [int(piece[0])]
        for s0 in starts:
            idx = np.arange(s0, min(int(piece[-1]) + 1, s0 + WIN)); idx = idx[inside[idx]]
            if len(idx) < 8:
                continue
            ext = float(np.linalg.norm(Ca_at[idx].max(0) - Ca_at[idx].min(0)))
            if ext < 0.3:                                          # the rig hardly moved (0.3 world units ~ 1 m): centres say nothing about the heading
                continue
            L = 0.5 * ext                                          # the up vector as a lever: fixes the roll about a straight path
            A = np.concatenate([Cb[idx], Cb[idx] + L * upb]); B = np.concatenate([Ca_at[idx], Ca_at[idx] + L * upa])
            sc, Rq, tq, med, used = robust_sim3(A, B, with_scale=True)
            if not (0.6 < sc < 1.6):
                continue
            wins.append({"mid": float(idx.mean()), "s": sc, "R": Rq, "t": tq, "res": med, "w": ext / (med + 0.02 * ext + 1e-6)})
        n_all += len(wins)
        if wins:                                                   # consensus rotation of this piece: drop the windows that disagree
            keep = list(range(len(wins)))
            for _ in range(4):
                Q0 = so3_mean([wins[i]["w"] * wins[i]["R"] for i in keep])
                dev = [np.degrees(np.arccos(np.clip((np.trace(w["R"] @ Q0.T) - 1) / 2, -1, 1))) for w in wins]
                keep = [i for i, d in enumerate(dev) if d < 25.0] or keep
            wins = [wins[i] for i in keep]
        n_kept += len(wins); all_wins += wins
        mids = np.array([w["mid"] for w in wins])
        for f in piece:
            if len(wins):
                d = np.abs(mids - f); near = np.argsort(d)[:3]
                wgt = 1.0 / (d[near] + WSTEP); wgt /= wgt.sum()
                Rq = so3_mean([wgt[k] * wins[i]["R"] for k, i in enumerate(near)]); Q_last = Rq
                Cn[f] = np.sum([wgt[k] * (wins[i]["s"] * (wins[i]["R"] @ Cb[f]) + wins[i]["t"]) for k, i in enumerate(near)], 0)
            else:                                                  # a short or standing piece: the rig was where the reference camera was
                Rq = Q_last if Q_last is not None else np.eye(3)
                Cn[f] = Ca_at[f]; borrowed += 1
            Rn[f] = Rb[f] @ Rq.T
    wins = all_wins
    if not wins:
        print(f"camera '{pre}': the rig never moved enough to register it onto '{ref}' - left in its own frame", flush=True); continue
    gap = np.linalg.norm(Cn[inside] - Ca_at[inside], axis=1)
    cam["C"], cam["R"] = Cn, Rn
    cam["up"] = so3_mean([w["R"] for w in wins]) @ upb
    reg[pre] = {"to": ref, "pieces": len(pieces), "windows": len(wins), "windows_rejected": n_all - n_kept, "positions_from_reference_track": borrowed, "window_scale_med": round(float(np.median([w["s"] for w in wins])), 4),
                "window_resid_med": round(float(np.median([w["res"] for w in wins])), 4), "pairs": int(inside.sum()),
                "centre_gap_med": round(float(np.median(gap)), 3), "centre_gap_p90": round(float(np.percentile(gap, 90)), 3)}
    print(f"camera '{pre}' -> '{ref}': {len(pieces)} piece(s) in time, {len(wins)} windows ({n_all - n_kept} rejected, {borrowed} frames positioned by the rig track), scale median {reg[pre]['window_scale_med']:.3f}, centre gap to the rig track median "
          f"{np.median(gap):.3f} (p90 {np.percentile(gap, 90):.3f}) on {inside.sum()} time pairs", flush=True)

up = np.sum([c["up"] * len(c["names"]) for c in cams.values()], 0); up /= np.linalg.norm(up)
frames = []
for pre, cam in cams.items():
    for n, c, Rk in zip(cam["names"], cam["C"], cam["R"]):
        frames.append({"i": frame_id(n), "name": n, "p": [float(v) for v in c], "q": quat_wxyz(Rk)})
frames.sort(key=lambda f: (ts_of(f["name"]) or 0, f["name"]))
allC = np.array([f["p"] for f in frames]); lo, hi = allC.min(0), allC.max(0)
ext_m = float((hi - lo).max()) or 1.0
intr = {"fx": FX, "fy": FX, "cx": CX, "cy": CY, "w": W_, "h": H_}
poses = dict(intr, up=[float(v) for v in up], frames=frames)
meta = {"name": f"poses by the VGGT-Omega chain ({IMAGES.parent.name})", "count": 0, "bbox": {"min": (lo - 1.5).tolist(), "max": (hi + 1.5).tolist()},
        "extent": ext_m + 3.0, "spacing": 0.01, "cell": 1.0, "has_nrm": False, "intrinsics": intr, "up": [float(v) for v in up],
        "units": "MoGe depth units (metres for a MoGe-2 dump)" if DEPTHS else "arbitrary (no depth dump)", "pose_source": "vggto_poses"}
(OUT / "poses.json").write_text(json.dumps(poses))
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
report = {"frames": len(frames), "seconds": round(time.time() - t_all), "chunk": CHUNK, "overlap": OVER, "res": RES, "units": meta["units"],
          "cameras": {pre or "single": {"frames": len(c["names"]), "path_len": round(c["path_len"], 2), "up_axis": c["up_axis"], "up_by": c["up_how"],
                                        "time_holes": c["holes"], "chunks": len(c["lam"]), "chunks_with_measured_scale": int(np.isfinite(c["meas"]).sum()),
                                        "matches_per_chunk_median": int(np.median(c["npts"][np.isfinite(c["meas"])])) if np.isfinite(c["meas"]).any() else 0,
                                        "scale_min_max": [round(float(c["lam"].min()), 4), round(float(c["lam"].max()), 4)],
                                        "vggt_fx_median": round(c["fx"], 1),
                                        "seam_centre_gap_med": round(float(np.median([s["centre_gap_med"] for s in c["seams"]])), 4) if c["seams"] else None,
                                        "seam_rot_spread_deg_med": round(float(np.median([s["rot_spread_deg"] for s in c["seams"]])), 3) if c["seams"] else None}
                      for pre, c in cams.items()},
          "registration": reg, "calibrated_fx": round(FX, 1), "up": [round(float(v), 4) for v in up]}
(OUT / "poses_report.json").write_text(json.dumps(report, indent=1))
print(f"POSES_DONE {OUT} frames={len(frames)} extent={ext_m:.1f} total {time.time() - t_all:.0f}s", flush=True)
