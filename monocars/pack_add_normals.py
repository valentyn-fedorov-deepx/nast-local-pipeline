"""Add an nxyz color set to an existing point pack (asset or full scene).

Normals come from the geometry itself: k-NN PCA per point (smallest
eigenvector of the local covariance). Orientation: toward the nearest camera
of the pack's track when poses exist, else outward from the centroid (asset
close-ups). Colors follow the dataset convention rgb = n*0.5+0.5.

Writes nrm.u8 next to pos.f32 and flips has_nrm in meta.json — the viewer's
RGB/nxyz toggle lights up by itself.

Usage: python pack_add_normals.py <pack_dir> [k=20]
"""
import json
import sys
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

PACK = Path(sys.argv[1])
K = int(sys.argv[2]) if len(sys.argv) > 2 else 20

P = np.frombuffer((PACK / "pos.f32").read_bytes(), np.float32).reshape(-1, 3).astype(np.float64)
n_pts = len(P)
print(f"{PACK.name}: {n_pts:,} pts, k={K}", flush=True)

tree = cKDTree(P)
N = np.zeros((n_pts, 3), np.float64)
BATCH = 200_000
for s in range(0, n_pts, BATCH):
    e = min(s + BATCH, n_pts)
    _, idx = tree.query(P[s:e], k=K, workers=-1)
    nb = P[idx]                                   # (b, K, 3)
    nb = nb - nb.mean(1, keepdims=True)
    cov = np.einsum("bki,bkj->bij", nb, nb) / K
    w, v = np.linalg.eigh(cov)                    # ascending eigenvalues
    N[s:e] = v[:, :, 0]
    print(f"  normals {e:,}/{n_pts:,}", flush=True)

# one smoothing pass kills the salt-and-pepper that gaussian centres produce:
# average each normal with its neighbours' (sign-aligned), renormalize
Ns = np.zeros_like(N)
for s in range(0, n_pts, BATCH):
    e = min(s + BATCH, n_pts)
    _, idx = tree.query(P[s:e], k=K, workers=-1)
    nn = N[idx]                                   # (b, K, 3)
    sign = np.sign(np.einsum("bki,bi->bk", nn, N[s:e]))[..., None]
    m = (nn * sign).mean(1)
    Ns[s:e] = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
N = Ns
print("smoothed", flush=True)

# ---- orientation ----
poses_p = PACK / "poses.json"
oriented = False
if poses_p.exists():
    try:
        poses = json.loads(poses_p.read_text())
        F = poses.get("frames") or []
        C = np.array([f["p"] for f in F], np.float64)
        if len(C) > 4:
            ct = cKDTree(C)
            _, ci = ct.query(P, k=1, workers=-1)
            to_cam = C[ci] - P
            flip = np.einsum("ij,ij->i", N, to_cam) < 0
            N[flip] = -N[flip]
            oriented = True
            print("oriented toward nearest camera", flush=True)
    except Exception as e:
        print("pose orientation failed:", e, flush=True)
if not oriented:
    ctr = P.mean(0)
    out = P - ctr
    flip = np.einsum("ij,ij->i", N, out) < 0
    N[flip] = -N[flip]
    print("oriented outward from centroid", flush=True)

col = np.clip((N * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
(PACK / "nrm.u8").write_bytes(np.ascontiguousarray(col).tobytes())
meta = json.loads((PACK / "meta.json").read_text())
meta["has_nrm"] = True
(PACK / "meta.json").write_text(json.dumps(meta))
print(f"PACK_ADD_NORMALS_DONE {PACK}", flush=True)
