"""Streaming pack: the 100M-point format.

One buffer for 100M points is 1.5 GB -- no browser holds that. This format
cuts the cloud into spatial cells and quantises positions inside each cell to
uint16 (6 B instead of 12), so the viewer streams cells on demand and keeps
only what the camera can see.

Layout (into <out_dir>):
  meta.json    format:2, counts, intrinsics, spacing (same fields as v1)
  poses.json   camera track (unchanged)
  cells2.json  [cx,cy,cz, r, count, minx,miny,minz, sx,sy,sz] per cell
  cell_<i>.bin u16 xyz * count, then u8 rgb * count; order shuffled inside the
               cell so any prefix is a uniform subsample (the LOD trick again)

Cells come from a regular grid, but any grid cell holding more than MAX_PER
points is split recursively (8 children) until every chunk fits -- the road
next to the walk path would otherwise be a single 20M-point monster.

Usage: python pack_stream.py --cloud x.ply --colmap sparse_dir --out dir
                             [--cell 1.2] [--max-per 1500000] [--outdoor-poses]
"""
import argparse
import json
import numpy as np
from pathlib import Path

from pack_cloud import read_cloud, read_colmap_poses

MAX_LEVEL = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", required=True)
    ap.add_argument("--colmap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=float, default=1.2)
    ap.add_argument("--max-per", type=int, default=1_500_000)
    ap.add_argument("--outdoor-poses", action="store_true")
    ap.add_argument("--name", default="scene-stream")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    xyz, rgb = read_cloud(Path(a.cloud))
    print("points:", len(xyz), flush=True)

    # same far-outlier trim as v1, for a sane bbox
    med = np.median(xyz, axis=0)
    r = np.linalg.norm(xyz - med, axis=1)
    keep = r < np.percentile(r, 99.5)
    xyz, rgb = xyz[keep], rgb[keep]

    lo = xyz.min(0)
    cell = a.cell

    # ---- assign to grid cells (keys packed into int64) ----
    ijk = np.floor((xyz - lo) / cell).astype(np.int64)
    dims = ijk.max(0) + 1
    key = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    order = np.argsort(key, kind="stable")
    xyz, rgb, key = xyz[order], rgb[order], key[order]
    uniq, starts, counts = np.unique(key, return_index=True, return_counts=True)
    print(f"grid cells: {len(uniq)}, biggest {counts.max()}", flush=True)

    rs = np.random.RandomState(0)
    cells_meta = []
    n_file = 0

    def emit(sub_xyz, sub_rgb, level):
        nonlocal n_file
        if len(sub_xyz) > a.max_per and level < MAX_LEVEL:
            mid = np.median(sub_xyz, axis=0)
            oct_idx = ((sub_xyz[:, 0] > mid[0]).astype(np.int8) * 4 +
                       (sub_xyz[:, 1] > mid[1]).astype(np.int8) * 2 +
                       (sub_xyz[:, 2] > mid[2]).astype(np.int8))
            for o in range(8):
                m = oct_idx == o
                if m.any():
                    emit(sub_xyz[m], sub_rgb[m], level + 1)
            return
        n = len(sub_xyz)
        perm = rs.permutation(n)
        sub_xyz, sub_rgb = sub_xyz[perm], sub_rgb[perm]
        mn = sub_xyz.min(0)
        size = np.maximum(sub_xyz.max(0) - mn, 1e-6)
        q = np.clip(np.round((sub_xyz - mn) / size * 65535), 0, 65535).astype("<u2")
        with open(out / f"cell_{n_file}.bin", "wb") as f:
            q.tofile(f)
            np.ascontiguousarray(sub_rgb).tofile(f)
        ctr = mn + size / 2
        rad = float(np.linalg.norm(size) / 2)
        cells_meta.append([float(ctr[0]), float(ctr[1]), float(ctr[2]), rad, n,
                           float(mn[0]), float(mn[1]), float(mn[2]),
                           float(size[0]), float(size[1]), float(size[2])])
        n_file += 1

    for u, s, c in zip(uniq, starts, counts):
        emit(xyz[s:s+c], rgb[s:s+c], 0)
    print(f"emitted {n_file} cell files", flush=True)

    poses = read_colmap_poses(Path(a.colmap), outdoor_only=a.outdoor_poses)

    # nearest-neighbour spacing off a sample, like v1 (drives point sizing)
    from scipy.spatial import cKDTree
    samp = xyz[rs.choice(len(xyz), min(200000, len(xyz)), replace=False)]
    dd, _ = cKDTree(samp).query(samp[rs.choice(len(samp), 4000, replace=False)], k=2, workers=-1)
    spacing = float(np.median(dd[:, 1])) * (len(samp) / len(xyz)) ** (1 / 3)

    hi = xyz.max(0)
    extent = float(np.linalg.norm(hi - lo) / 3)
    (out / "cells2.json").write_text(json.dumps(cells_meta))
    (out / "meta.json").write_text(json.dumps({
        "format": 2, "name": a.name, "count": int(len(xyz)),
        "bbox": {"min": lo.tolist(), "max": hi.tolist()},
        "extent": extent, "spacing": spacing, "cell": cell,
        "has_nrm": False,
        "intrinsics": {"fx": poses["fx"], "fy": poses["fy"], "cx": poses["cx"],
                       "cy": poses["cy"], "w": poses["w"], "h": poses["h"]},
        "up": poses["up"],
    }, indent=1))
    (out / "poses.json").write_text(json.dumps(
        {"up": poses["up"], "frames": poses["frames"]}))
    print("PACK_STREAM_DONE", out, len(xyz), n_file, flush=True)


main()
