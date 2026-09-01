"""TRELLIS gaussian ply -> centered colored point ply for the close-up viewer.

Usage: python asset2ply.py <asset.ply> <out.ply> [min_opacity=0.35]
"""
import sys
import numpy as np
from pathlib import Path

SRC = Path(sys.argv[1]); OUT = Path(sys.argv[2])
MIN_OP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35

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
rgb = (rgb * 255).astype(np.uint8)
if "opacity" in g.dtype.names:
    op = 1 / (1 + np.exp(-g["opacity"].astype(np.float64)))
    keep = op > MIN_OP
    X, rgb = X[keep], rgb[keep]
X -= (np.percentile(X, 2, axis=0) + np.percentile(X, 98, axis=0)) / 2
# orientation for the close-up: TRELLIS gives no guaranteed up axis. Best
# evidence is the photo the asset came from -- with --masks=<job dir> the SAM
# silhouettes pick the up axis (and a "front" view) by projection matching;
# without masks fall back to a shape heuristic (thin -> long axis, else the
# smallest PCA axis)
import json as _json
_mask_dir = [a[8:] for a in sys.argv if a.startswith("--masks=")]
_est = None
if _mask_dir:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import asset_up
        _masks = asset_up.load_masks(_mask_dir[0])
        if _masks:
            _est = asset_up.estimate_up(X, _masks)
            _est["method"] = "mask"; _est["n_masks"] = len(_masks)
    except Exception as e:                       # never let orientation kill the job
        print("asset_up failed:", repr(e), flush=True); _est = None
if _est is not None and _est.get("margin", 0.0) < 0.03:
    print(f"asset_up inconclusive (margin {_est['margin']:.3f}) -> shape heuristic", flush=True)
    _est = None
if _est is None:
    _, sv, Vt = np.linalg.svd(X[np.random.RandomState(0).choice(len(X), min(60000, len(X)), replace=False)],
                              full_matrices=False)
    # thin = pole-like: most points hug the principal axis (median radial
    # distance << length); a car is elongated too but its points sit far
    # from the axis (body), so the ratio separates them where sv ratios don't
    _t = X @ Vt[0]
    _rad = np.linalg.norm(X - np.outer(_t, Vt[0]), axis=1)
    _len = float(np.percentile(_t, 98) - np.percentile(_t, 2))
    thin = (float(np.median(_rad)) / max(_len, 1e-6)) < 0.07 and sv[0] > 2.0 * sv[1]
    up_vec = Vt[0] if thin else Vt[2]
    if thin:
        # which end is the top? for poles/lamps/signs the head end is wider:
        # compare the lateral spread of the top 15% vs bottom 15% along the axis
        t = X @ up_vec
        lo_m, hi_m = t < np.percentile(t, 15), t > np.percentile(t, 85)
        lat = X - np.outer(t, up_vec)
        w_lo = np.linalg.norm(lat[lo_m], axis=1).mean() if lo_m.any() else 0
        w_hi = np.linalg.norm(lat[hi_m], axis=1).mean() if hi_m.any() else 0
        if w_lo > w_hi * 1.15:                       # wide end is at "low" -> flip
            up_vec = -up_vec
    else:
        # compact objects (cars, bins, boxes) stand on their WIDE end: the
        # footprint slab spreads further from the axis than the roof slab
        t = X @ up_vec
        lo_m, hi_m = t < np.percentile(t, 12), t > np.percentile(t, 88)
        lat = X - np.outer(t, up_vec)
        w_lo = np.linalg.norm(lat[lo_m], axis=1).mean() if lo_m.any() else 0
        w_hi = np.linalg.norm(lat[hi_m], axis=1).mean() if hi_m.any() else 0
        if w_hi > w_lo * 1.10:                       # wide end is at "high" -> flip
            up_vec = -up_vec
    _est = {"up": [float(v) for v in up_vec], "front": None, "thin": bool(thin), "method": "pca"}
_json.dump(_est, open(str(OUT) + ".up.json", "w"))
print("close-up up axis:", np.round(_est["up"], 3), _est["method"],
      ("front " + str(np.round(_est["front"], 2))) if _est.get("front") else "",
      (f"score {_est['score']:.3f} margin {_est['margin']:.3f}") if "score" in _est else "")

rec = np.zeros(len(X), dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                       ("red", "u1"), ("green", "u1"), ("blue", "u1")]))
rec["x"], rec["y"], rec["z"] = X[:, 0], X[:, 1], X[:, 2]
rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
with open(OUT, "wb") as f:
    f.write(("ply\nformat binary_little_endian 1.0\n"
             f"element vertex {len(rec)}\n"
             "property float x\nproperty float y\nproperty float z\n"
             "property uchar red\nproperty uchar green\nproperty uchar blue\n"
             "end_header\n").encode())
    rec.tofile(f)
print(f"ASSET2PLY_DONE {OUT} pts={len(rec)}", flush=True)
