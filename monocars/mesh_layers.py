"""Mesh close-up of a reconstructed object, painted with the polarization layers.

The gaussian close-up is a cloud of overlapping semi-transparent primitives --
painting per gaussian and alpha-blending them back gives a speckled look.  The
mesh decoded from the same TRELLIS latent is a clean surface: one colour per
vertex, proper visibility from a depth buffer, and lighting-free display of
the layer colours.  This script:

  1. loads asset_mesh.ply (gaussian-ply frame), asset_mesh_uv.npy + the baked
     texture (RGB set) and the close-up frame (splat_frame.json) so the mesh
     shares orientation/scale with the splat close-up;
  2. projects the mesh into every view the object was seen in (placement.json
     + poses), rasterises a per-view depth buffer (painter's algorithm on the
     triangles), and samples each polarization layer at every VISIBLE, FACING
     vertex; normal layers are decoded camera -> world -> close-up frame;
  3. aggregates over views, fills unseen vertices from neighbours, smooths a
     little over the surface, and writes a compact mesh pack:
        mesh.json   {n_vertices, n_faces, sets:[{name,file}], bbox}
        mesh_pos.f32, mesh_nrm.f32 (close-up frame), mesh_idx.u32
        mesh_<set>.u8  (per-vertex RGB)
     sets: RGB (baked texture), nxyz geometric (vertex normals), + layers.

Usage: python mesh_layers.py <job_dir> <obj_scene_dir> <street_video_dir> <map_scene_dir> [products]
"""
import json
import os
import sys
import numpy as np
import cv2
import trimesh
from pathlib import Path
from scipy.spatial import cKDTree

JOB = Path(sys.argv[1]); OBJ = Path(sys.argv[2]); VID = Path(sys.argv[3]); MAP = Path(sys.argv[4])
PRODS = sys.argv[5].split(",") if len(sys.argv) > 5 else \
    ["nxyz", "n_xy", "n_xz", "nxyz_phys", "nxyz_diffuse", "nxyz_specv2", "edge", "rgb_deglare"]
NORMAL_PRODS = {"nxyz_phys", "nxyz_diffuse", "nxyz_specv2"}
LABELS = {"nxyz": "Nxyz", "n_xy": "N xy", "n_xz": "N xz", "nxyz_phys": "phys",
          "nxyz_diffuse": "diffuse", "nxyz_specv2": "specv2",
          "edge": "Edge", "rgb_deglare": "RGB deglare"}


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ---- mesh in the gaussian-ply (asset) frame --------------------------------------
sf = json.loads((OBJ / "splat_frame.json").read_text())
R_cu = np.array(sf["R"]); ctr_cu = np.array(sf["ctr"]); s_cu = float(sf["scale"])
if (JOB / "asset_mesh.ply").exists():
    m = trimesh.load(str(JOB / "asset_mesh.ply"), process=False, force="mesh")
    uv = np.load(JOB / "asset_mesh_uv.npy") if (JOB / "asset_mesh_uv.npy").exists() else None
else:
    # points-route fallback (no TRELLIS mesh): mesh_from_points already wrote
    # the mesh into the scene in the CLOSE-UP frame; lift it back to the
    # asset frame so the one transform chain below serves both routes
    V_cu0 = np.frombuffer((OBJ / "mesh_pos.f32").read_bytes(), np.float32).reshape(-1, 3).astype(np.float64)
    F_cu0 = np.frombuffer((OBJ / "mesh_idx.u32").read_bytes(), np.uint32).reshape(-1, 3).astype(np.int64)
    m = trimesh.Trimesh((V_cu0 / s_cu) @ R_cu + ctr_cu, F_cu0, process=False)
    uv = None
    print(f"no asset_mesh.ply — using the scene mesh ({len(V_cu0):,} verts, points route)", flush=True)
# drop floating crumbs (detached mirror bits, noise): keep components holding >= 0.4% of the faces
try:
    comps = trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(m.faces)))
    big = [c for c in comps if len(c) >= 0.004 * len(m.faces)]
    keep_f = np.sort(np.concatenate(big)) if big else np.arange(len(m.faces))
    if len(keep_f) < len(m.faces):
        F0 = np.asarray(m.faces)[keep_f]
        used = np.unique(F0); remap = -np.ones(len(m.vertices), np.int64); remap[used] = np.arange(len(used))
        m = trimesh.Trimesh(np.asarray(m.vertices)[used], remap[F0], process=False)
        if uv is not None: uv = uv[used]
        print(f"dropped {len(comps) - len(big)} small components", flush=True)
except Exception as e:
    print("component filter skipped:", e, flush=True)
V_a = np.asarray(m.vertices, dtype=np.float64); F = np.asarray(m.faces, dtype=np.int64)
tex = cv2.imread(str(JOB / "asset_mesh_tex.png"), cv2.IMREAD_COLOR) if (JOB / "asset_mesh_tex.png").exists() else None
print(f"mesh {len(V_a):,} vertices, {len(F):,} faces", flush=True)
# per-vertex painting wants a dense surface: linear subdivision (same geometry,
# 4x faces per pass) until ~60k faces; uv interpolates along
MIN_FACES = 60000
if len(m.faces) < MIN_FACES:
    # manual linear subdivision that carries uv (trimesh versions differ in their attribute support)
    V_a = np.asarray(m.vertices, dtype=np.float64); F = np.asarray(m.faces, dtype=np.int64)
    while len(F) < MIN_FACES:
        e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0); e.sort(1)
        uniq, inv = np.unique(e, axis=0, return_inverse=True)
        mid = (V_a[uniq[:, 0]] + V_a[uniq[:, 1]]) * 0.5
        base = len(V_a); V_a = np.vstack([V_a, mid])
        if uv is not None:
            uv = np.vstack([uv, (uv[uniq[:, 0]] + uv[uniq[:, 1]]) * 0.5]).astype(np.float32)
        inv = inv.reshape(3, -1)           # edge ids per (01, 12, 20)
        m01, m12, m20 = base + inv[0], base + inv[1], base + inv[2]
        F = np.vstack([np.stack([F[:, 0], m01, m20], 1), np.stack([m01, F[:, 1], m12], 1),
                       np.stack([m20, m12, F[:, 2]], 1), np.stack([m01, m12, m20], 1)])
    print(f"subdivided -> {len(V_a):,} vertices, {len(F):,} faces", flush=True)

V_cu = s_cu * (V_a - ctr_cu) @ R_cu.T                       # close-up frame (same as the splat)
# vertex normals in the close-up frame (geometry truth for comparison)
mc = trimesh.Trimesh(V_cu, F, process=False)
N_cu = np.asarray(mc.vertex_normals, dtype=np.float64)
# orient outward (TRELLIS meshes are closed-ish; outward = away from the centroid on average)
outw = V_cu - V_cu.mean(0)
if (N_cu * outw).sum() < 0:
    N_cu = -N_cu; F = F[:, ::-1]

pl = json.loads((JOB / "placement.json").read_text())
R_pp = np.array(pl["R"]); s_pp = float(pl["s"]); t_pp = np.array(pl["t"]); a_ctr = np.array(pl["a_ctr"])
up_w = np.array(pl["up"]); base_off = float(pl["base_off"])
V_w = s_pp * (V_a - a_ctr) @ R_pp.T + t_pp + up_w * base_off
N_w = (N_cu @ R_cu) @ R_pp.T
R_w2cu = R_cu @ R_pp.T

# ---- views -----------------------------------------------------------------------
poses = json.loads((MAP / "poses.json").read_text()); meta = json.loads((MAP / "meta.json").read_text())
I = meta["intrinsics"]; fx, fy, cx, cy, Ws, Hs = I["fx"], I["fy"], I["cx"], I["cy"], I["w"], I["h"]
by_name = {f["name"]: f for f in poses["frames"]}
box = json.loads((JOB / "box.json").read_text())
views = []
for rf in box.get("roi_frames", []):
    if rf.get("name"): views.append(rf["name"])
views += list(box.get("obs_frames", []))
views = [v for v in dict.fromkeys(views) if v in by_name]
print(f"{len(views)} views", flush=True)

NV = len(V_cu)
samples = {p: [] for p in PRODS}
for vn in views:
    f = by_name[vn]; R = quat_to_R(f["q"]); C = np.array(f["p"])
    cam = (V_w - C) @ R.T; z = cam[:, 2]
    okz = z > 0.2
    sx = fx * cam[:, 0] / np.where(okz, z, 1) + cx; sy = fy * cam[:, 1] / np.where(okz, z, 1) + cy
    inside = okz & (sx >= 0) & (sx < Ws - 1) & (sy >= 0) & (sy < Hs - 1)
    # depth buffer: painter's algorithm over triangles (far -> near), constant depth per triangle
    S = 1                                               # full-res buffer
    zbuf = np.full((Hs, Ws), 1e9, np.float32)
    fz = z[F].mean(1)
    fok = okz[F].all(1)
    order = np.argsort(-fz)                             # far -> near: near triangles overwrite
    pts = np.stack([sx, sy], 1)
    for fi in order:
        if not fok[fi]: continue
        tri = pts[F[fi]]
        if (tri < -50).any() or (tri[:, 0] > Ws + 50).any() or (tri[:, 1] > Hs + 50).any(): continue
        cv2.fillConvexPoly(zbuf, np.round(tri).astype(np.int32), float(fz[fi]))
    vd = V_w - C; vd /= np.linalg.norm(vd, axis=1, keepdims=True) + 1e-9
    facing = (N_w * vd).sum(1) < 0.0
    xi = np.clip(sx.astype(int), 0, Ws - 1); yi = np.clip(sy.astype(int), 0, Hs - 1)
    vis = inside & facing & (z <= zbuf[yi, xi] * 1.02 + 0.02)
    nv = int(vis.sum())
    print(f"  {vn[:26]}: {nv:,} visible vertices", flush=True)
    if nv < 30: continue
    xs, ys = sx[vis], sy[vis]
    for p in PRODS:
        lp = VID / "layers" / p / (vn.rsplit(".", 1)[0] + ".jpg")
        img = cv2.imread(str(lp), cv2.IMREAD_COLOR)
        if img is None: continue
        img = img.astype(np.float32) / 255.0
        x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int); ax = xs - x0; ay = ys - y0
        x1 = np.clip(x0 + 1, 0, img.shape[1] - 1); y1 = np.clip(y0 + 1, 0, img.shape[0] - 1)
        v = (img[y0, x0] * ((1 - ax) * (1 - ay))[:, None] + img[y0, x1] * (ax * (1 - ay))[:, None]
             + img[y1, x0] * ((1 - ax) * ay)[:, None] + img[y1, x1] * (ax * ay)[:, None])
        out = np.full((NV, 3), np.nan)
        if p in NORMAL_PRODS:
            # canon encoding (2026-08-26): B=|nz|, G=(ny+1)/2, R=(nx+1)/2 (display R=X G=Y B=Z)
            nx = v[:, 2] * 2 - 1; ny = v[:, 1] * 2 - 1; nz = v[:, 0]
            # decode convention (camera frame, OpenCV axes): env NCONV = "x,y,z" signs/swap, default "nx,ny,-nz"
            conv = os.environ.get("NCONV", "ny,-nx,-nz").split(",")   # layers are display-frame; back to sensor axes
            comp = {"nx": nx, "ny": ny, "nz": nz, "-nx": -nx, "-ny": -ny, "-nz": -nz}
            n_cam = np.stack([comp[conv[0]], comp[conv[1]], comp[conv[2]]], 1)
            n_cam /= np.linalg.norm(n_cam, axis=1, keepdims=True) + 1e-9
            if os.environ.get("DISAMB", "1") == "1":
                # polarization normals carry the AoLP pi-ambiguity (n vs its
                # in-plane flip) and the specular/diffuse 90-degree azimuth
                # branch; the mesh is the geometric prior the client's own
                # pipeline uses ("disambiguate_to_prior"): keep, per sample, the
                # candidate closest to the mesh normal -- zenith stays measured
                prior = (N_w[vis]) @ R.T                      # mesh normals in this camera frame
                cands = [n_cam,
                         np.stack([-n_cam[:, 0], -n_cam[:, 1], n_cam[:, 2]], 1),
                         np.stack([-n_cam[:, 1], n_cam[:, 0], n_cam[:, 2]], 1),
                         np.stack([n_cam[:, 1], -n_cam[:, 0], n_cam[:, 2]], 1)]
                dots = np.stack([(c * prior).sum(1) for c in cands], 1)
                best = np.argmax(dots, 1)
                n_cam = np.stack(cands, 1)[np.arange(len(best)), best]
            out[vis] = (n_cam @ R) @ R_w2cu.T
        else:
            out[vis] = v[:, ::-1]
        samples[p].append(out)

# ---- colour sets -----------------------------------------------------------------
tree = cKDTree(V_cu)
_, nn = tree.query(V_cu, k=min(16, NV))
sets = []
def write_set(name, fname, col):
    (OBJ / fname).write_bytes(np.clip(col * 255, 0, 255).astype(np.uint8).tobytes())
    sets.append({"name": name, "file": fname})

# RGB from the baked texture (per vertex via uv)
if uv is not None and tex is not None:
    th, tw = tex.shape[:2]
    u = np.clip((uv[:, 0] * (tw - 1)).astype(int), 0, tw - 1); vv = np.clip(((1 - uv[:, 1]) * (th - 1)).astype(int), 0, th - 1)
    rgb = tex[vv, u][:, ::-1].astype(np.float64) / 255.0
    write_set("RGB (baked)", "mesh_rgb.u8", rgb)
# geometric normals of the mesh itself
write_set("nxyz (mesh)", "mesh_nxyz.u8", N_cu * 0.5 + 0.5)
for p in PRODS:
    if not samples[p]: continue
    S = np.stack(samples[p], 0); cnt = np.isfinite(S[..., 0]).sum(0); seen = cnt > 0
    with np.errstate(all="ignore"):
        if p in NORMAL_PRODS:
            mm = np.nanmean(S, 0); mm /= np.linalg.norm(mm, axis=1, keepdims=True) + 1e-9
            flip = (mm * N_cu).sum(1) < 0; mm[flip] *= -1
            col = mm * 0.5 + 0.5
        else:
            col = np.nanmedian(S, 0)
    if seen.any() and (~seen).any():
        tr = cKDTree(V_cu[seen]); _, j = tr.query(V_cu[~seen], k=1); col[~seen] = col[seen][j]
    col = np.nan_to_num(col, nan=0.5)
    if p in NORMAL_PRODS:
        vv_ = (col * 2 - 1)[nn].mean(1); vv_ /= np.linalg.norm(vv_, axis=1, keepdims=True) + 1e-9
        col = np.clip(vv_ * 0.5 + 0.5, 0, 1)
    else:
        col = np.median(col[nn], axis=1)
        if p not in ("rgb_deglare",):    # true-colour layers keep photographic values
            lum = col.mean(1); lo, hi = np.percentile(lum, 2), np.percentile(lum, 98)
            if hi - lo > 1e-3: col = np.clip((col - lo) / (hi - lo), 0, 1)
    write_set(LABELS.get(p, p), f"mesh_{p}.u8", col)
    extra = ""
    if p in NORMAL_PRODS:
        d = np.degrees(np.arccos(np.clip(((col * 2 - 1) * N_cu).sum(1) / (np.linalg.norm(col * 2 - 1, axis=1) + 1e-9), -1, 1)))
        extra = f" | angle to mesh normals: median {np.median(d[seen]):.1f} deg, <30deg {np.mean(d[seen] < 30)*100:.0f}%"
    print(f"{p}: seen {seen.mean()*100:.0f}% of vertices{extra}", flush=True)

(OBJ / "mesh_pos.f32").write_bytes(V_cu.astype(np.float32).tobytes())
(OBJ / "mesh_nrm.f32").write_bytes(N_cu.astype(np.float32).tobytes())
(OBJ / "mesh_idx.u32").write_bytes(F.astype(np.uint32).tobytes())
lo, hi = V_cu.min(0), V_cu.max(0)
(OBJ / "mesh.json").write_text(json.dumps({"n_vertices": int(NV), "n_faces": int(len(F)), "sets": sets,
                                           "bbox": {"min": lo.tolist(), "max": hi.tolist()}}))
print("MESH_LAYERS_DONE", NV, len(F), len(sets), flush=True)
