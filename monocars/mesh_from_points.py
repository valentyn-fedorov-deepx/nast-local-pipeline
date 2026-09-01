"""Mesh an object scene straight from its packed POINTS (for assets without a
TRELLIS mesh, e.g. old jobs): occupancy -> 6-ray solidify -> marching cubes,
colors by nearest point. Writes mesh_pos/nrm/idx/rgb + mesh.json (+mesh.html
copy) so mesh_viewer and object_struct work as if TRELLIS exported it.

Usage: python mesh_from_points.py <obj_scene_dir> [pitch_frac=1/240]
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure

D = Path(sys.argv[1])
PF = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0 / 240
P = np.frombuffer((D / "pos.f32").read_bytes(), np.float32).reshape(-1, 3).astype(np.float64)
C = np.frombuffer((D / "rgb.u8").read_bytes(), np.uint8).reshape(-1, 3)
size = float((P.max(0) - P.min(0)).max())
pitch = size * PF
lo = P.min(0) - 6 * pitch
ijk = np.floor((P - lo) / pitch).astype(np.int64)
dims = ijk.max(0) + 7
occ = np.zeros(dims, bool)
occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
occ = ndimage.binary_dilation(occ, iterations=1)
occ = ndimage.binary_closing(occ, iterations=2)
print(f"{D.name}: {len(P):,} pts -> grid {tuple(dims)} occ {int(occ.sum())}", flush=True)


def blocked(o, axis):
    p = np.maximum.accumulate(o, axis=axis)
    n = np.flip(np.maximum.accumulate(np.flip(o, axis), axis=axis), axis)
    return p.astype(np.int8) + n.astype(np.int8)


solid = ndimage.binary_fill_holes((blocked(occ, 0) + blocked(occ, 1) + blocked(occ, 2) >= 5) | occ)
edt = ndimage.distance_transform_edt(solid)
vv, ff, _, _ = measure.marching_cubes(edt, level=0.8)
mesh = trimesh.Trimesh(vv * pitch + lo, ff, process=True)
trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=8)
comps = sorted(mesh.split(only_watertight=False), key=lambda c: -len(c.faces))
keep = [c for c in comps if len(c.faces) >= 0.03 * len(mesh.faces)]
mesh = trimesh.util.concatenate(keep) if keep else mesh
V = np.asarray(mesh.vertices); F = np.asarray(mesh.faces)
print(f"mesh: {len(V):,} verts {len(F):,} faces ({len(comps)} comps -> {len(keep)})", flush=True)

_, jj = cKDTree(P).query(V, k=1)
col = C[jj]
(D / "mesh_pos.f32").write_bytes(V.astype(np.float32).tobytes())
(D / "mesh_nrm.f32").write_bytes(np.asarray(mesh.vertex_normals, np.float32).tobytes())
(D / "mesh_idx.u32").write_bytes(F.astype(np.uint32).tobytes())
(D / "mesh_rgb.u8").write_bytes(col.astype(np.uint8).tobytes())
(D / "mesh.json").write_text(json.dumps({
    "n_vertices": int(len(V)), "n_faces": int(len(F)),
    "sets": [{"name": "RGB (points)", "file": "mesh_rgb.u8"}]}, indent=1))
src = Path(__file__).parent.parent / "viewer" / "mesh_viewer.html"
shutil.copy2(src, D / "mesh.html")
print("MESH_FROM_POINTS_DONE", flush=True)
