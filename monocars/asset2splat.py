"""TRELLIS gaussian ply -> .splat files for the WebGL splat viewer (objects tab).

Keeps the gaussians AS gaussians (position, scale, rotation, colour, opacity)
instead of collapsing them to points, so the close-up is alpha-blended and
photoreal like TRELLIS' own turntable render.  The asset is re-oriented with
the up/front estimated from the crop silhouettes (asset_up.py via asset2ply):
    y = up, z = front (object -> camera of the operator's crop), x = y x z
centred and scaled to a unit box, so the viewer needs no per-object camera.

Writes <out_dir>/splat.splat      (32 B / gaussian: pos f32x3, scale f32x3,
                                    rgba u8x4, rot u8x4 (w,x,y,z)*128+128)
       <out_dir>/splat_n.splat    same, colours = kNN-PCA normals (nxyz look)
       <out_dir>/splat.json       {count, bbox, up_method}

Usage: python asset2splat.py <asset.ply> <up.json> <out_dir> [min_opacity=0.03] [--crops=<job dir>]
"""
import json
import sys
import numpy as np
from pathlib import Path

SRC = Path(sys.argv[1]); UPJ = Path(sys.argv[2]); OUT = Path(sys.argv[3])
MIN_OP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.03
OUT.mkdir(parents=True, exist_ok=True)

PLY_T = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4"}
with open(SRC, "rb") as f:
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
    g = np.fromfile(f, dtype=np.dtype(fields), count=n)

X = np.stack([g["x"], g["y"], g["z"]], -1).astype(np.float64)
C0 = 0.28209479177387814
rgb = np.clip(np.stack([g["f_dc_0"], g["f_dc_1"], g["f_dc_2"]], -1) * C0 + 0.5, 0, 1)
op = 1 / (1 + np.exp(-g["opacity"].astype(np.float64)))
sc = np.exp(np.stack([g["scale_0"], g["scale_1"], g["scale_2"]], -1).astype(np.float64))
q = np.stack([g["rot_0"], g["rot_1"], g["rot_2"], g["rot_3"]], -1).astype(np.float64)   # w,x,y,z
q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
keep = op > MIN_OP
X, rgb, op, sc, q = X[keep], rgb[keep], op[keep], sc[keep], q[keep]
print(f"{len(X):,} gaussians after opacity > {MIN_OP}", flush=True)

# ---- orientation: y = up, z = front, x = up x front -------------------------
uj = json.loads(UPJ.read_text()) if UPJ.exists() else {"up": [0, 1, 0], "front": None}
up = np.array(uj["up"], float); up /= np.linalg.norm(up)
front = uj.get("front")
if front is not None:
    front = np.array(front, float); front -= up * (front @ up)
    front = front / np.linalg.norm(front) if np.linalg.norm(front) > 1e-6 else None
if front is None:
    # no crop front: put the long horizontal axis left-right (side view)
    Xc0 = X - X.mean(0); Hh = Xc0 - np.outer(Xc0 @ up, up)
    _, _, Vt = np.linalg.svd(Hh[np.random.RandomState(0).choice(len(Hh), min(40000, len(Hh)), replace=False)],
                             full_matrices=False)
    xax = Vt[0] - up * (Vt[0] @ up); xax /= np.linalg.norm(xax)
    front = np.cross(xax, up)
xax = np.cross(up, front); xax /= np.linalg.norm(xax)
R = np.stack([xax, up, front])                    # rows: new axes in old coords -> p' = R p
X = X @ R.T
ctr = (np.percentile(X, 1, axis=0) + np.percentile(X, 99, axis=0)) / 2
X -= ctr
ext = np.percentile(X, 99.5, axis=0) - np.percentile(X, 0.5, axis=0)
s = 1.0 / max(float(ext.max()), 1e-6)
X *= s; sc *= s
# the close-up frame for anyone who needs to go back to asset coords:
#   X_cu = s * R @ (X_asset - ctr)   <=>   X_asset = R.T @ X_cu / s + ctr
json.dump({"R": R.tolist(), "ctr": ctr.tolist(), "scale": float(s), "up": up.tolist(), "front": front.tolist()},
          open(OUT / "splat_frame.json", "w"))

# rotate the gaussians too: q' = q_R (x) q
def mat2quat(M):
    tr = np.trace(M)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2; w = 0.25 * S
        x = (M[2, 1] - M[1, 2]) / S; y = (M[0, 2] - M[2, 0]) / S; z = (M[1, 0] - M[0, 1]) / S
    elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
        S = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
        w = (M[2, 1] - M[1, 2]) / S; x = 0.25 * S; y = (M[0, 1] + M[1, 0]) / S; z = (M[0, 2] + M[2, 0]) / S
    elif M[1, 1] > M[2, 2]:
        S = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
        w = (M[0, 2] - M[2, 0]) / S; x = (M[0, 1] + M[1, 0]) / S; y = 0.25 * S; z = (M[1, 2] + M[2, 1]) / S
    else:
        S = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
        w = (M[1, 0] - M[0, 1]) / S; x = (M[0, 2] + M[2, 0]) / S; y = (M[1, 2] + M[2, 1]) / S; z = 0.25 * S
    return np.array([w, x, y, z])

qR = mat2quat(R)
w1, x1, y1, z1 = qR
w2, x2, y2, z2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
q = np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
              w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
              w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
              w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)
q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)

# ---- colour fidelity: match the gaussians' colour statistics to the crop ---
# TRELLIS drifts in tone (a dark-red car for an orange one); the operator's
# crops are the reference, so pull the asset's per-channel mean/std onto the
# masked crop pixels (SAM alpha).  Only ROI crops (roi_*.png) -- projection
# crops are loosely framed.
_cd = [a[8:] for a in sys.argv if a.startswith("--crops=")]
if _cd:
    try:
        from PIL import Image
        d = Path(_cd[0]); enh = d / "crops_enh"
        px = []
        for cp in sorted(enh.glob("roi_*.png")) if enh.exists() else []:
            im = np.array(Image.open(cp).convert("RGBA")).astype(np.float64) / 255.0
            m = im[..., 3] > 0.5
            if m.sum() > 200:
                px.append(im[..., :3][m])
        if px:
            px = np.concatenate(px, 0)
            w = op / op.sum()
            mean_a = (rgb * w[:, None]).sum(0)
            std_a = np.sqrt(((rgb - mean_a) ** 2 * w[:, None]).sum(0)) + 1e-6
            mean_c, std_c = px.mean(0), px.std(0) + 1e-6
            gain = np.clip(std_c / std_a, 0.5, 2.0)
            rgb = np.clip((rgb - mean_a) * gain + mean_c, 0, 1)
            print(f"colour match: asset {np.round(mean_a,3)} -> crop {np.round(mean_c,3)} gain {np.round(gain,2)} "
                  f"({len(px):,} crop px)", flush=True)
            pass
        else:
            print("colour match skipped: no masked ROI crops", flush=True)
    except Exception as e:
        print("colour match skipped:", e, flush=True)

# ---- thin objects (poles, posts, signs): TRELLIS paints invented blotches on
# what the photos show as a uniform shaft. Colour varies little AROUND a thin
# object, so per-slice (along the up axis) robust medians are the truth: keep
# each gaussian close to its slice's median colour, a little shading survives
ext_y = float(np.percentile(X[:, 1], 99) - np.percentile(X[:, 1], 1))
ext_h = float(max(np.percentile(X[:, 0], 99) - np.percentile(X[:, 0], 1),
                  np.percentile(X[:, 2], 99) - np.percentile(X[:, 2], 1)))
if ext_y > 3.0 * ext_h and len(X) > 500:
    nb = 96
    b = np.clip(((X[:, 1] - X[:, 1].min()) / max(ext_y, 1e-6) * nb).astype(int), 0, nb - 1)
    med = np.zeros((nb, 3)); have = np.zeros(nb, bool)
    for k in range(nb):
        m_ = b == k
        if m_.sum() >= 5:
            med[k] = np.median(rgb[m_], axis=0); have[k] = True
    if not have.all():
        idx = np.arange(nb)
        for ch in range(3):
            med[:, ch] = np.interp(idx, idx[have], med[have, ch]) if have.any() else rgb.mean(0)[ch]
    # a running Gaussian window along the axis: no slice-to-slice stripes,
    # only genuine changes (the head vs the shaft) survive
    ker = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2); ker /= ker.sum()
    pad = np.pad(med, ((12, 12), (0, 0)), mode="edge")
    med = np.stack([np.convolve(pad[:, ch], ker, mode="valid") for ch in range(3)], -1)
    rgb = np.clip(med[b] + (rgb - med[b]) * 0.3, 0, 1)
    print(f"thin object (h/w {ext_y / max(ext_h, 1e-6):.1f}): colours smoothed per slice along the axis", flush=True)

# ---- normals for the nxyz look: kNN-PCA on gaussian centres, outward -------
try:
    from scipy.spatial import cKDTree
    K = 32
    Ps = X
    tree = cKDTree(Ps)
    _, nn = tree.query(Ps, k=min(K, len(Ps)))
    nb = Ps[nn] - Ps[:, None, :]
    cov = np.einsum("nki,nkj->nij", nb, nb) / max(nn.shape[1], 1)
    w_, v_ = np.linalg.eigh(cov)
    N = v_[:, :, 0]
    out = Ps - Ps.mean(0)
    flip = np.einsum("ij,ij->i", N, out) < 0
    N[flip] *= -1
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    nrgb = np.clip(N * 0.5 + 0.5, 0, 1)
except Exception as e:
    print("normals skipped:", e, flush=True)
    nrgb = np.full_like(rgb, 0.5)

# ---- pack (antimatter15 .splat layout), big + opaque first ------------------
order = np.argsort(-(op * np.prod(sc, axis=1)))
def pack(colors):
    buf = np.zeros(len(X), dtype=np.dtype([("pos", "<f4", 3), ("scale", "<f4", 3),
                                            ("rgba", "u1", 4), ("rot", "u1", 4)]))
    buf["pos"] = X[order].astype(np.float32)
    buf["scale"] = sc[order].astype(np.float32)
    buf["rgba"][:, :3] = (colors[order] * 255).astype(np.uint8)
    buf["rgba"][:, 3] = (op[order] * 255).astype(np.uint8)
    buf["rot"] = np.clip(q[order] * 128 + 128, 0, 255).astype(np.uint8)
    return buf.tobytes()

(OUT / "splat.splat").write_bytes(pack(rgb))
(OUT / "splat_n.splat").write_bytes(pack(nrgb))
lo, hi = X.min(0), X.max(0)
(OUT / "splat.json").write_text(json.dumps({
    "count": int(len(X)), "bbox": {"min": lo.tolist(), "max": hi.tolist()},
    "up_method": uj.get("method", "?"), "front_known": bool(uj.get("front"))}))
print(f"ASSET2SPLAT_DONE {OUT} n={len(X)} scale={s:.3f}", flush=True)
