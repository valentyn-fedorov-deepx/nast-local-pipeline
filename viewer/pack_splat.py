"""Pack a gaussian-splat .ply into the splat viewer's binary format.

Output (into <out_dir>):
  smeta.json  count + bbox (poses/meta come from the point-cloud pack, shared)
  center.f32  xyz
  cov.f32     per-splat 3D covariance, upper triangle (xx,xy,xz,yy,yz,zz) --
              rotation+scale are baked here once so the shader never touches
              quaternions
  color.u8    rgba: SH0 -> rgb, sigmoid(opacity) -> a

Usage: python pack_splat.py --splat <splat.ply> --out <dir> [--min-alpha 0.02]
"""
import argparse
import json
import numpy as np
from pathlib import Path

C0 = 0.28209479177387814


def read_ply(path):
    with open(path, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"end_header\n"):
            hdr += f.readline()
        fields, n = [], 0
        for l in hdr.decode().splitlines():
            if l.startswith("element vertex"):
                n = int(l.split()[-1])
            elif l.startswith("property "):
                fields.append((l.split()[2], "<f4"))
        return np.fromfile(f, dtype=np.dtype(fields), count=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-alpha", type=float, default=0.02)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    g = read_ply(Path(a.splat))
    print("splats in:", len(g))

    alpha = 1.0 / (1.0 + np.exp(-g["opacity"].astype(np.float64)))
    keep = alpha > a.min_alpha
    g, alpha = g[keep], alpha[keep]
    n = len(g)
    print(f"kept over alpha {a.min_alpha}: {n}")

    xyz = np.stack([g["x"], g["y"], g["z"]], -1).astype(np.float32)

    # covariance = R S S^T R^T, computed once here instead of per-frame in GLSL
    q = np.stack([g["rot_0"], g["rot_1"], g["rot_2"], g["rot_3"]], -1).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    s = np.exp(np.stack([g["scale_0"], g["scale_1"], g["scale_2"]], -1).astype(np.float64))
    M = R * s[:, None, :]                     # R @ diag(s)
    cov = M @ M.transpose(0, 2, 1)
    cov6 = np.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
                     cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], -1).astype(np.float32)

    rgb = np.stack([g["f_dc_0"], g["f_dc_1"], g["f_dc_2"]], -1).astype(np.float64)
    rgb = np.clip(0.5 + C0 * rgb, 0, 1)
    rgba = np.concatenate([rgb, alpha[:, None]], -1)
    rgba = (rgba * 255).astype(np.uint8)

    xyz.tofile(out / "center.f32")
    cov6.tofile(out / "cov.f32")
    rgba.tofile(out / "color.u8")
    lo, hi = xyz.min(0), xyz.max(0)
    (out / "smeta.json").write_text(json.dumps({
        "count": int(n),
        "bbox": {"min": lo.tolist(), "max": hi.tolist()},
        "extent": float(np.linalg.norm(hi - lo) / 3),
    }))
    print("PACK_SPLAT_DONE", out, n)


main()
