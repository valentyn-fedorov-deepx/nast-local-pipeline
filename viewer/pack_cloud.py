"""Pack a point cloud + COLMAP camera track into the viewer's binary format.

Output (into <out_dir>):
  meta.json   counts, bbox, intrinsics, which color sets exist
  pos.f32     float32 xyz, points pre-SHUFFLED (prefix = uniform random subset,
              which is what makes level-of-detail a single drawArrays count)
  rgb.u8      uint8 rgb
  nrm.u8      uint8 nxyz colors (optional)
  poses.json  per-frame camera centre + quaternion, ordered by frame

Usage:
  python pack_cloud.py --cloud <file.ply|file.npz> --colmap <sparse_dir>
                       --out <dir> [--max-points N] [--nrm-cloud <file>]
"""
import argparse
import json
import re
import struct
import numpy as np
from pathlib import Path


def read_ply(path: Path):
    with open(path, "rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise RuntimeError("bad ply header")
            header += line
        txt = header.decode()
        n = int([l for l in txt.splitlines() if l.startswith("element vertex")][0].split()[-1])
        props = [l.split()[1:] for l in txt.splitlines() if l.startswith("property")]
        np_t = {"float": np.float32, "float32": np.float32, "double": np.float64,
                "uchar": np.uint8, "uint8": np.uint8, "int": np.int32, "uint": np.uint32,
                "short": np.int16, "ushort": np.uint16}
        dt = np.dtype([(name, np_t[typ]) for typ, name in props])
        data = np.fromfile(f, dtype=dt, count=n)
    xyz = np.stack([data["x"], data["y"], data["z"]], -1).astype(np.float32)
    names = data.dtype.names
    if "red" in names:
        rgb = np.stack([data["red"], data["green"], data["blue"]], -1).astype(np.uint8)
    elif "f_dc_0" in names:                      # gaussian-splat ply
        C0 = 0.28209479177387814
        fdc = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], -1)
        rgb = np.clip(fdc * C0 + 0.5, 0, 1)
        rgb = (rgb * 255).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 200, np.uint8)
    return xyz, rgb


def read_cloud(path: Path):
    if path.suffix == ".npz":
        z = np.load(path)
        return z["xyz"].astype(np.float32), z["rgb"].astype(np.uint8)
    return read_ply(path)


def is_outdoor(name: str) -> bool:
    """Daylight frame? The exposure baked into the filename is the giveaway.

    Indoors the sensor opens up to ~33 ms at ISO 1000+; on the street it sits
    under 20 ms at ISO<=200. Frames without that metadata (the mono clip) are
    treated as outdoor so nothing gets silently dropped.
    """
    m = re.search(r"_et_(\d+)_iso_(\d+)", name)
    if not m:
        return True
    return int(m.group(2)) <= 200 and int(m.group(1)) / 1e6 < 20


def read_colmap_poses(sparse: Path, outdoor_only: bool = False):
    cam_txt = sparse / "cameras.txt"
    img_txt = sparse / "images.txt"
    fx = fy = cx = cy = None
    W = H = None
    for ln in cam_txt.read_text().splitlines():
        if ln.startswith("#"):
            continue
        p = ln.split()
        if len(p) < 5:
            continue
        W, H = int(p[2]), int(p[3])
        vals = list(map(float, p[4:]))
        if p[1] in ("PINHOLE", "OPENCV"):
            fx, fy, cx, cy = vals[0], vals[1], vals[2], vals[3]
        else:                                     # SIMPLE_* : one focal
            fx = fy = vals[0]
            cx, cy = vals[1], vals[2]
        break

    frames = []
    for ln in img_txt.read_text().splitlines():
        if ln.startswith("#"):
            continue
        p = ln.split()
        if len(p) >= 10 and not p[9][0].isdigit():
            if outdoor_only and not is_outdoor(p[9]):
                continue                          # keep the walk on the street
            qw, qx, qy, qz = map(float, p[1:5])
            t = np.array(list(map(float, p[5:8])))
            name = p[9]
            # world->cam rotation from quaternion
            R = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ])
            C = -R.T @ t
            digits = "".join(ch for ch in name if ch.isdigit())
            idx = int(digits) if digits else len(frames)
            frames.append({"i": idx, "name": name,
                           "p": [float(v) for v in C],
                           "q": [qw, qx, qy, qz]})
    frames.sort(key=lambda f: f["i"])
    # world up from the cameras: whichever image axis (x or y) stays constant
    # over the whole drive is the vertical -- for upright frames that is -y
    # (colmap camera Y points down), for the sideways-mounted Orthovector
    # sensor it is x. Mean of -y over a drive that turns around cancels to a
    # meaningless residual, which is exactly the bug this replaces. Sign: the
    # cloud lies below the cameras.
    xs, ys = [], []
    for f in frames:
        qw, qx, qy, qz = f["q"]
        R0 = np.array([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)])
        R1 = np.array([2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)])
        xs.append(R0); ys.append(-R1)
    mx = np.mean(xs, axis=0) if xs else np.zeros(3); my = np.mean(ys, axis=0) if ys else np.array([0.0, 1.0, 0.0])
    up = mx if np.linalg.norm(mx) > np.linalg.norm(my) else my
    up = up / (np.linalg.norm(up) + 1e-9)           # sign fixed against the cloud in main()
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "w": W, "h": H,
            "up": [float(v) for v in up], "frames": frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nrm-cloud", default=None,
                    help="second cloud, SAME point order, used as nxyz colours")
    ap.add_argument("--max-points", type=int, default=14_000_000)
    ap.add_argument("--name", default="scene")
    ap.add_argument("--outdoor-poses", action="store_true",
                    help="drop indoor frames from the camera track")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    xyz, rgb = read_cloud(Path(a.cloud))
    print("loaded", len(xyz), "points")
    nrm = None
    if a.nrm_cloud:
        nx, nc = read_cloud(Path(a.nrm_cloud))
        if len(nc) == len(rgb):
            nrm = nc
        else:
            print(f"WARNING: nrm cloud has {len(nc)} pts vs {len(rgb)} — ignored")

    # drop far outliers so the scene bbox (and thus the camera speed) stays sane
    med = np.median(xyz, axis=0)
    r = np.linalg.norm(xyz - med, axis=1)
    keep = r < np.percentile(r, 99.5)
    xyz, rgb = xyz[keep], rgb[keep]
    if nrm is not None:
        nrm = nrm[keep]

    # shuffle -> any prefix is a uniform random subset (this is the LOD trick)
    rs = np.random.RandomState(0)
    order = rs.permutation(len(xyz))
    if len(order) > a.max_points:
        order = order[:a.max_points]
        print("subsampled to", len(order))
    xyz, rgb = xyz[order], rgb[order]
    if nrm is not None:
        nrm = nrm[order]

    # ---- spatial cells -----------------------------------------------------
    # Sort points into a coarse grid so the viewer can throw away everything
    # outside the frustum and thin only the far cells. Within a cell the order
    # stays shuffled, so any prefix of a cell is still a uniform subset of it.
    ext_all = xyz.max(0) - xyz.min(0)
    cell = float(max(ext_all.max() / 22.0, 1e-6))
    q = np.floor((xyz - xyz.min(0)) / cell).astype(np.int64)
    key = (q[:, 0] << 42) ^ (q[:, 1] << 21) ^ q[:, 2]
    order2 = np.argsort(key, kind="stable")
    xyz, rgb = xyz[order2], rgb[order2]
    if nrm is not None:
        nrm = nrm[order2]
    key = key[order2]
    uniq, starts, counts = np.unique(key, return_index=True, return_counts=True)
    cells = []
    for st, ct in zip(starts, counts):
        block = xyz[st:st + ct]
        lo_b, hi_b = block.min(0), block.max(0)
        c = (lo_b + hi_b) / 2
        rad = float(np.linalg.norm(hi_b - c)) + 1e-6
        cells.append([float(c[0]), float(c[1]), float(c[2]), rad, int(st), int(ct)])
    print(f"cells: {len(cells)} (cell size {cell:.3f}, mean {len(xyz)/max(len(cells),1):.0f} pts)")

    poses = read_colmap_poses(Path(a.colmap), outdoor_only=a.outdoor_poses)
    # up sign: the cloud (ground, mostly) lies BELOW the cameras
    try:
        upv = np.array(poses["up"]); cams = np.array([f["p"] for f in poses["frames"]])
        if len(cams) and np.median((xyz[:: max(1, len(xyz) // 200000)] - cams.mean(0)) @ upv) > 0:
            poses["up"] = [float(-v) for v in upv]
            print("up flipped: points were above the cameras")
    except Exception as e:
        print("up sign check skipped:", e)
    print("poses:", len(poses["frames"]), "intrinsics:", poses["fx"], poses["cx"])

    (out / "pos.f32").write_bytes(np.ascontiguousarray(xyz, np.float32).tobytes())
    (out / "rgb.u8").write_bytes(np.ascontiguousarray(rgb, np.uint8).tobytes())
    if nrm is not None:
        (out / "nrm.u8").write_bytes(np.ascontiguousarray(nrm, np.uint8).tobytes())

    lo = xyz.min(0).tolist()
    hi = xyz.max(0).tolist()
    ext = float(np.percentile(np.linalg.norm(xyz - np.median(xyz, 0), axis=1), 95))
    # point spacing: median nearest-neighbour distance on a sample, corrected
    # for the sampling fraction (surface-distributed => spacing ~ 1/sqrt(density))
    try:
        from scipy.spatial import cKDTree
        rs2 = np.random.RandomState(1)
        sub = xyz[rs2.choice(len(xyz), min(300000, len(xyz)), replace=False)]
        qs = sub[rs2.choice(len(sub), min(4000, len(sub)), replace=False)]
        dd, _ = cKDTree(sub).query(qs, k=2)
        spacing = float(np.median(dd[:, 1]) * (len(sub) / len(xyz)) ** 0.5)
    except Exception as e:
        print("spacing estimate failed:", e)
        spacing = ext * 0.002

    meta = {
        "name": a.name,
        "count": int(len(xyz)),
        "bbox": {"min": lo, "max": hi},
        "extent": ext,
        "spacing": spacing,
        "cell": cell,
        "has_nrm": nrm is not None,
        "intrinsics": {k: poses[k] for k in ("fx", "fy", "cx", "cy", "w", "h")},
    }
    (out / "cells.json").write_text(json.dumps(cells))
    print("spacing:", round(spacing, 4))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "poses.json").write_text(json.dumps(poses))
    print("packed ->", out, meta["count"], "points, nrm:", meta["has_nrm"])


if __name__ == "__main__":
    main()
