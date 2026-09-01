"""Class-agnostic structural layers for an object scene (generic: car, lamp, any).

Writes into <obj_scene>:
  mesh_seg.u8            per-vertex segment colors (+ registers set "Segments" in mesh.json)
  struct_explode.f32     per-vertex displacement at explode=1 (same length as mesh_pos)
  struct_creases.f32     crease line segments, N*(2*3) float32
  struct_skel.f32        3D skeleton points
  layer_shell{1,2}.pos.f32/.nrm.f32/.idx.u32   SDF inner shells
  struct.json            manifest

Usage: python object_struct.py <obj_scene_dir>
"""
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure
from skimage.morphology import skeletonize

D = Path(sys.argv[1])
V0 = np.frombuffer((D / "mesh_pos.f32").read_bytes(), np.float32).reshape(-1, 3).astype(np.float64)
F0 = np.frombuffer((D / "mesh_idx.u32").read_bytes(), np.uint32).reshape(-1, 3).astype(np.int64)
size = float((V0.max(0) - V0.min(0)).max())

# ---- nameless segments on the FULL mesh (indexing preserved) ---------------------
smf = trimesh.Trimesh(V0, F0, process=False)
trimesh.smoothing.filter_taubin(smf, lamb=0.5, nu=-0.53, iterations=10)
fn = smf.face_normals
adjf = defaultdict(list)
emap = defaultdict(list)
for fi, (a, b, c) in enumerate(F0):
    for e in ((a, b), (b, c), (c, a)):
        emap[tuple(sorted(e))].append(fi)
for e, fs in emap.items():
    if len(fs) == 2:
        adjf[fs[0]].append(fs[1]); adjf[fs[1]].append(fs[0])
DIH = np.cos(np.radians(30))
seg = -np.ones(len(F0), np.int64)
sid = 0
for f0 in np.argsort(-np.abs(fn[:, 1])):
    if seg[f0] >= 0:
        continue
    q = deque([f0]); seg[f0] = sid
    while q:
        f = q.popleft()
        for g in adjf[f]:
            if seg[g] < 0 and float(np.dot(fn[f], fn[g])) > DIH:
                seg[g] = sid; q.append(g)
    sid += 1
for _ in range(6):
    sizes = np.bincount(seg, minlength=sid)
    small = set(np.where(sizes < 0.012 * len(F0))[0])
    if not small:
        break
    bnd = defaultdict(lambda: defaultdict(int))
    for f in range(len(F0)):
        for g in adjf[f]:
            if seg[f] != seg[g]:
                bnd[seg[f]][seg[g]] += 1
    remap = np.arange(sid); moved = 0
    for s in small:
        nbs = {k: v for k, v in bnd[s].items() if k not in small}
        tgt = max(nbs, key=nbs.get) if nbs else (max(bnd[s], key=bnd[s].get) if bnd[s] else s)
        if tgt != s:
            remap[s] = tgt; moved += 1
    seg = remap[seg]
    if not moved:
        break
u = np.unique(seg); seg = np.searchsorted(u, seg); nseg = len(u)
print(f"segments (dihedral): {nseg}", flush=True)

rs = np.random.RandomState(7)
pal = ((rs.rand(nseg, 3) * 0.55 + 0.40) * 255).astype(np.uint8)
vcol = np.full((len(V0), 3), 90, np.uint8)
vseg = -np.ones(len(V0), np.int64)
for fi, s in enumerate(seg):
    for vv in F0[fi]:
        vcol[vv] = pal[s]; vseg[vv] = s
(D / "mesh_seg.u8").write_bytes(vcol.tobytes())

segctr = np.zeros((nseg, 3))
for s in range(nseg):
    vs = np.unique(F0[seg == s])
    segctr[s] = V0[vs].mean(0)
off = segctr - V0.mean(0)
off /= (np.linalg.norm(off, axis=1, keepdims=True) + 1e-9)
disp = np.zeros_like(V0)
m = vseg >= 0
disp[m] = off[vseg[m]] * 0.16 * size
(D / "struct_explode.f32").write_bytes(disp.astype(np.float32).tobytes())

# ---- creases on the big-components body ------------------------------------------
angs = {}
for e, fs in emap.items():
    if len(fs) == 2:
        d = np.clip(np.dot(fn[fs[0]], fn[fs[1]]), -1, 1)
        angs[e] = float(np.degrees(np.arccos(d)))
thr_c = 35.0
if angs:
    a_arr = np.array(list(angs.values()))
    if (a_arr > thr_c).sum() < 300:
        thr_c = max(12.0, float(np.percentile(a_arr, 97)))
        print(f"crease threshold adapted -> {thr_c:.1f} deg", flush=True)
E = [e for e, fs in emap.items() if len(fs) == 1] + [e for e, a in angs.items() if a > thr_c]
g2 = defaultdict(list)
for a, b in E:
    g2[a].append((a, b)); g2[b].append((a, b))
seen = set(); chains = []
for e0 in E:
    if tuple(e0) in seen:
        continue
    ch = [e0]; seen.add(tuple(e0))
    for dr in (0, 1):
        cur = e0[dr]
        while True:
            nx = [e for e in g2[cur] if tuple(e) not in seen]
            if not nx:
                break
            e = nx[0]; seen.add(tuple(e)); ch.append(e)
            cur = e[1] if e[0] == cur else e[0]
    chains.append(ch)
Vs = np.asarray(smf.vertices)
chains = [c for c in chains if sum(np.linalg.norm(Vs[a] - Vs[b]) for a, b in c) > 0.03 * size]
segs = np.array([[V0[a], V0[b]] for c in chains for a, b in c], np.float32)
(D / "struct_creases.f32").write_bytes(segs.tobytes())
print(f"creases: {len(segs)} segments in {len(chains)} chains", flush=True)

# ---- shells + skeleton on big components -----------------------------------------
mesh0 = trimesh.Trimesh(V0, F0, process=False)
comps = sorted(mesh0.split(only_watertight=False), key=lambda c: -len(c.faces))
big = [c for c in comps if len(c.faces) >= 0.02 * len(F0)]
body = trimesh.util.concatenate(big) if big else mesh0
smb = body.copy(); trimesh.smoothing.filter_taubin(smb, lamb=0.5, nu=-0.53, iterations=12)


def blocked(o, axis):
    p = np.maximum.accumulate(o, axis=axis)
    n = np.flip(np.maximum.accumulate(np.flip(o, axis), axis=axis), axis)
    return p.astype(np.int8) + n.astype(np.int8)


# shells (-3% / -6% SDF layers) dropped from the locked GUI (2026-08-27):
# struct modes are Mesh / Sketch / Skeleton / Exploded only
shells_meta = []

vg2 = smb.voxelized(size / 160)
occ2 = np.pad(vg2.matrix.astype(bool), 4)
occ2 = ndimage.binary_closing(occ2, iterations=2)
solid2 = ndimage.binary_fill_holes((blocked(occ2, 0) + blocked(occ2, 1) + blocked(occ2, 2) >= 5) | occ2)
skel = skeletonize(solid2)
sk_ijk = np.argwhere(skel)
sk = (sk_ijk * (size / 160) + (np.asarray(vg2.translation) - 4 * (size / 160))).astype(np.float32)
(D / "struct_skel.f32").write_bytes(sk.tobytes())
print("skeleton pts:", len(sk), flush=True)

# ---- smooth-shape fallback: segment by skeleton BRANCHES -------------------------
if nseg < 3 and len(sk) > 20:
    from scipy.spatial import cKDTree as _KD
    vox = {tuple(v): i for i, v in enumerate(sk_ijk)}
    nbrs = [[] for _ in sk_ijk]
    for i, v in enumerate(sk_ijk):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    j = vox.get((v[0]+dx, v[1]+dy, v[2]+dz))
                    if j is not None:
                        nbrs[i].append(j)
    deg = np.array([len(n) for n in nbrs])
    junction = set(np.where(deg >= 3)[0])
    branch = -np.ones(len(sk_ijk), np.int64)
    bid = 0
    for i in range(len(sk_ijk)):
        if branch[i] >= 0 or i in junction:
            continue
        q = deque([i]); branch[i] = bid
        while q:
            a = q.popleft()
            for b in nbrs[a]:
                if branch[b] < 0 and b not in junction:
                    branch[b] = bid; q.append(b)
        bid += 1
    for i in junction:
        bs = [branch[b] for b in nbrs[i] if branch[b] >= 0]
        branch[i] = bs[0] if bs else 0
    sizes_b = np.bincount(branch[branch >= 0], minlength=bid)
    keepb = np.where(sizes_b >= max(4, 0.03 * len(sk_ijk)))[0]
    lut_b = -np.ones(bid, np.int64); lut_b[keepb] = np.arange(len(keepb))
    okb = (branch >= 0) & (lut_b[np.maximum(branch, 0)] >= 0)
    if len(keepb) < 2 and len(sk) > 30:
        # one long polyline: split by local tangent direction (pole vs arm etc.)
        kd_sk = _KD(sk)
        tang = np.zeros((len(sk), 3))
        for i in range(len(sk)):
            nb = kd_sk.query_ball_point(sk[i], r=4 * size / 160)
            Q = sk[nb] - sk[nb].mean(0)
            if len(nb) >= 3:
                _, _, Vt_ = np.linalg.svd(Q)
                tang[i] = Vt_[0]
        main = np.linalg.svd(sk - sk.mean(0))[2][0]
        grp = (np.abs(tang @ main) < 0.7).astype(np.int64)
        if 0.05 < grp.mean() < 0.95:
            branch = grp
            keepb = np.array([0, 1])
            lut_b = np.arange(2)
            okb = np.ones(len(sk), bool)
            print("skeleton split by tangent direction", flush=True)
    if okb.sum() > 10 and len(keepb) >= 2:
        # crisp boundaries: seed vertices near each branch, then multi-source
        # geodesic growth over the mesh graph (kNN-to-skeleton smears joints)
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        dd_s, jn_s = _KD(sk[okb]).query(V0, k=1)
        br_all = lut_b[branch[okb][jn_s]]
        near = dd_s < np.percentile(dd_s, 25)
        seeds = {}
        for b in range(len(keepb)):
            cand = np.where(near & (br_all == b))[0]
            if len(cand) == 0:
                cand = np.where(br_all == b)[0][np.argsort(dd_s[br_all == b])[:50]]
            seeds[b] = cand
        rows = np.concatenate([F0[:, 0], F0[:, 1], F0[:, 2]])
        cols = np.concatenate([F0[:, 1], F0[:, 2], F0[:, 0]])
        wts = np.linalg.norm(V0[rows] - V0[cols], axis=1)
        Gm = coo_matrix((np.concatenate([wts, wts]),
                         (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
                        shape=(len(V0), len(V0))).tocsr()
        dists = np.full((len(keepb), len(V0)), np.inf)
        for b, sd in seeds.items():
            dists[b] = dijkstra(Gm, directed=False, indices=sd, min_only=True)
        vb = np.argmin(dists, axis=0)
        nseg = len(keepb)
        pal = ((rs.rand(nseg, 3) * 0.55 + 0.40) * 255).astype(np.uint8)
        vcol = pal[vb]
        (D / "mesh_seg.u8").write_bytes(vcol.astype(np.uint8).tobytes())
        segctr = np.zeros((nseg, 3))
        for s in range(nseg):
            segctr[s] = V0[vb == s].mean(0)
        off = segctr - V0.mean(0)
        off /= (np.linalg.norm(off, axis=1, keepdims=True) + 1e-9)
        disp = off[vb] * 0.16 * size
        (D / "struct_explode.f32").write_bytes(disp.astype(np.float32).tobytes())
        print(f"segments (skeleton branches): {nseg}", flush=True)

# ---- exploded geometry with CUT seams: faces duplicated per segment ------------
# (uses whichever segmentation is final: vseg from dihedral path or vb fallback)
try:
    final_vseg = vb if 'vb' in dir() and isinstance(vb, np.ndarray) and len(vb) == len(V0) else vseg
except NameError:
    final_vseg = vseg
fseg = np.zeros(len(F0), np.int64)
for fi in range(len(F0)):
    ss = final_vseg[F0[fi]]
    ss = ss[ss >= 0]
    fseg[fi] = np.bincount(ss).argmax() if len(ss) else 0
nseg_f = int(fseg.max()) + 1
pal_f = ((np.random.RandomState(7).rand(max(nseg_f, 1), 3) * 0.55 + 0.40) * 255).astype(np.uint8)
segctr_f = np.zeros((nseg_f, 3))
for s in range(nseg_f):
    m_ = F0[fseg == s]
    segctr_f[s] = V0[np.unique(m_)].mean(0) if len(m_) else V0.mean(0)
off_f = segctr_f - V0.mean(0)
off_f /= (np.linalg.norm(off_f, axis=1, keepdims=True) + 1e-9)
pos2 = V0[F0.reshape(-1)].astype(np.float32)                      # 3 verts per face, cut
fn_flat = np.cross(V0[F0[:, 1]] - V0[F0[:, 0]], V0[F0[:, 2]] - V0[F0[:, 0]])
fn_flat /= (np.linalg.norm(fn_flat, axis=1, keepdims=True) + 1e-12)
nrm2 = np.repeat(fn_flat, 3, axis=0).astype(np.float32)
col2 = np.repeat(pal_f[fseg], 3, axis=0).astype(np.uint8)
disp2 = np.repeat((off_f[fseg] * 0.16 * size), 3, axis=0).astype(np.float32)
(D / "explode_pos.f32").write_bytes(pos2.tobytes())
(D / "explode_nrm.f32").write_bytes(nrm2.tobytes())
(D / "explode_col.u8").write_bytes(col2.tobytes())
(D / "explode_disp.f32").write_bytes(disp2.tobytes())
print(f"exploded cut geometry: {len(pos2):,} verts, {nseg_f} segments", flush=True)

# ---- splines from crease chains, coloured by gravity-frame normals ---------------
# virtual floor: y = min; palette = Dave-style nxyz with vertical as the "toward" axis:
#   R = |ny|, G = (nz+1)/2, B = (nx+1)/2   (horizontal azimuth spins G/B, verticality -> R)
from scipy import interpolate as _si
mesh_for_n = trimesh.Trimesh(V0, F0, process=False)
VN = np.asarray(mesh_for_n.vertex_normals, dtype=np.float64)

# ---- CRISP sketch: per-view silhouette LOOPS (long, continuous by construction) --
smf2 = trimesh.Trimesh(V0, F0, process=False)
trimesh.smoothing.filter_taubin(smf2, lamb=0.5, nu=-0.53, iterations=25)
fn2 = smf2.face_normals
V0s = np.asarray(smf2.vertices, dtype=np.float64)
VN = np.asarray(smf2.vertex_normals, dtype=np.float64)
from scipy.spatial import cKDTree as _KDs
kd_v = _KDs(V0s)
# technical wireframe on the SOLID body (voxel-filled from points -> no holes,
# every section is a complete closed ring; the raw TRELLIS mesh is hole-y)
try:
    vv_s, ff_s, _, _ = measure.marching_cubes(edt, level=0.75)
    body_solid = trimesh.Trimesh(vv_s * pitch + origin, ff_s, process=True)
    trimesh.smoothing.filter_taubin(body_solid, lamb=0.5, nu=-0.53, iterations=14)
    cs_b = sorted(body_solid.split(only_watertight=False), key=lambda c: -len(c.faces))
    keep_b = [c for c in cs_b if len(c.faces) >= 0.05 * len(body_solid.faces)]
    body_solid = trimesh.util.concatenate(keep_b) if keep_b else body_solid
except Exception as e:
    print("solid body failed, falling back to mesh:", e, flush=True)
    body_solid = smf2
# uniform surface sampling of the TRUE mesh = snap target without density bias
samp, _fi = trimesh.sample.sample_surface(mesh_for_n, 500_000)
PP = np.asarray(samp, dtype=np.float64)
# snap tree: pull section samples onto the REAL outer skin
kd_skin = None
try:
    from scipy.spatial import cKDTree as _KDsnap
    kd_skin = _KDsnap(PP)
except Exception:
    pass
chains_sketch = []
lo_s = V0s.min(0); hi_s = V0s.max(0); ext_s = hi_s - lo_s
ctr_s = V0s.mean(0)
# the object may sit diagonally in the frame: slice along ITS principal axes
H2 = V0s[:, [0, 2]] - ctr_s[[0, 2]]
_, _, Vt_h = np.linalg.svd(H2, full_matrices=False)
u_long = np.array([Vt_h[0, 0], 0.0, Vt_h[0, 1]]); u_long /= np.linalg.norm(u_long)
u_wide = np.array([-u_long[2], 0.0, u_long[0]])
half_L = float(np.abs(H2 @ Vt_h[0]).max()); half_W = float(np.abs(H2 @ Vt_h[1]).max())
axes_plan = []
horiz_dominant = max(ext_s[0], ext_s[2]) >= ext_s[1]
n_tr = 9 if horiz_dominant else 0
for i in range(n_tr):
    off = -half_L * 0.92 + 2 * 0.92 * half_L * i / (n_tr - 1)
    axes_plan.append(((ctr_s + u_long * off).tolist(), u_long.tolist()))
for fy in ((0.24, 0.55) if horiz_dominant else ()):                          # horizontal: bumper/waist/glass
    axes_plan.append(([float(ctr_s[0]), float(lo_s[1] + ext_s[1] * fy), float(ctr_s[2])], [0, 1, 0]))
for fx in ((0.0,) if horiz_dominant else ()):                          # longitudinal profiles
    axes_plan.append(((ctr_s + u_wide * (half_W * fx)).tolist(), u_wide.tolist()))
for org, nrm_p in axes_plan:
    try:
        sec = body_solid.section(plane_origin=org, plane_normal=nrm_p)
        if sec is None:
            continue
        for Pcur in sec.discrete:
            Pcur = np.asarray(Pcur, dtype=np.float64)
            L_c = np.linalg.norm(np.diff(Pcur, axis=0), axis=1).sum()
            if L_c > 0.25 * size:
                chains_sketch.append(Pcur)
    except Exception:
        continue
# plan outline from the top-view silhouette of the SOLID body
fn_t = body_solid.face_normals
Vb = np.asarray(body_solid.vertices, dtype=np.float64)
emap_b = defaultdict(list)
for fi_b, (a_b, b_b, c_b) in enumerate(np.asarray(body_solid.faces)):
    for e_b in ((a_b, b_b), (b_b, c_b), (c_b, a_b)):
        emap_b[tuple(sorted(e_b))].append(fi_b)
vis_t = fn_t @ np.array([0.02, 1.0, 0.03]) > 0
E_t = [e for e, fs in emap_b.items() if len(fs) == 2 and vis_t[fs[0]] != vis_t[fs[1]]]
g_t = defaultdict(list)
for a, b in E_t:
    g_t[a].append((a, b)); g_t[b].append((a, b))
seen_t = set()
for e0 in E_t:
    if tuple(e0) in seen_t:
        continue
    ch = [e0]; seen_t.add(tuple(e0))
    for dr in (0, 1):
        cur = e0[dr]
        while True:
            nx = [e for e in g_t[cur] if tuple(e) not in seen_t]
            if not nx:
                break
            e = nx[0]; seen_t.add(tuple(e)); ch.append(e)
            cur = e[1] if e[0] == cur else e[0]
    if sum(np.linalg.norm(Vb[a] - Vb[b]) for a, b in ch) > 0.5 * size:
        chains_sketch.append(Vb[[e[0] for e in ch] + [ch[-1][1]]])
print(f"sketch chains: {len(chains_sketch)} (sections + plan outline)", flush=True)

sp_pts = []; sp_cols = []
for ch in chains_sketch:
    if isinstance(ch, np.ndarray):
        P = ch
        if len(P) < 4:
            continue
        closed = np.linalg.norm(P[0] - P[-1]) < 0.03 * size
        try:
            tck, _u = _si.splprep(P.T, s=len(P) * (0.006 * size) ** 2, k=3, per=1 if closed else 0)
            L = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
            uu = np.linspace(0, 1, max(12, int(L / (0.008 * size))))
            Q = np.stack(_si.splev(uu, tck), -1)
        except Exception:
            Q = P
        if kd_skin is not None:
            dq, jq = kd_skin.query(Q, k=1)
            snap = PP[jq]
            dvec = snap - Q
            dn = np.linalg.norm(dvec, axis=1, keepdims=True)
            clamp = 0.025 * size
            dvec = dvec * np.minimum(1.0, clamp / np.maximum(dn, 1e-9))
            near = (dn[:, 0] < 0.05 * size)
            Q = Q + 0.6 * dvec * near[:, None]
            try:
                tck2, _u2 = _si.splprep(Q.T, s=len(Q) * (0.004 * size) ** 2, k=3, per=1 if closed else 0)
                Q = np.stack(_si.splev(np.linspace(0, 1, len(Q)), tck2), -1)
            except Exception:
                pass
        _, jv = kd_v.query(Q, k=6)
        n_avg = VN[jv].mean(1)
        n_avg /= (np.linalg.norm(n_avg, axis=1, keepdims=True) + 1e-12)
        col = np.stack([np.abs(n_avg[:, 1]), (n_avg[:, 2] + 1) * 0.5, (n_avg[:, 0] + 1) * 0.5], -1)
        for i in range(len(Q) - 1):
            sp_pts.append([Q[i], Q[i+1]]); sp_cols.append([col[i], col[i+1]])
        continue
    # ordered vertex path of the chain
    order_v = [ch[0][0], ch[0][1]]
    for a, b in ch[1:]:
        if a == order_v[-1]:
            order_v.append(b)
        elif b == order_v[-1]:
            order_v.append(a)
        elif a == order_v[0]:
            order_v.insert(0, b)
        elif b == order_v[0]:
            order_v.insert(0, a)
    P = V0s[order_v] if 'V0s' in dir() else V0[order_v]
    if len(P) < 4:
        continue
    closed = np.linalg.norm(P[0] - P[-1]) < 0.03 * size
    try:
        tck, _u = _si.splprep(P.T, s=len(P) * (0.010 * size) ** 2, k=3, per=1 if closed else 0)
        L = sum(np.linalg.norm(P[i+1] - P[i]) for i in range(len(P) - 1))
        nres = max(8, int(L / (0.008 * size)))
        uu = np.linspace(0, 1, nres)
        Q = np.stack(_si.splev(uu, tck), -1)
    except Exception:
        Q = P
    _, jv = kd_v.query(Q, k=6)
    n_avg = VN[jv].mean(1)
    n_avg /= (np.linalg.norm(n_avg, axis=1, keepdims=True) + 1e-12)
    col = np.stack([np.abs(n_avg[:, 1]),
                    (n_avg[:, 2] + 1) * 0.5,
                    (n_avg[:, 0] + 1) * 0.5], -1)
    for i in range(len(Q) - 1):
        sp_pts.append([Q[i], Q[i+1]])
        sp_cols.append([col[i], col[i+1]])
# smooth objects have few creases -> add skeleton splines + section rings
if len(sp_pts) < 800:
    def add_polyline(P):
        if len(P) < 3:
            return
        try:
            tck, _u = _si.splprep(P.T, s=len(P) * (0.006 * size) ** 2, k=min(3, len(P) - 1))
            L = sum(np.linalg.norm(P[i+1] - P[i]) for i in range(len(P) - 1))
            uu = np.linspace(0, 1, max(8, int(L / (0.01 * size))))
            Q = np.stack(_si.splev(uu, tck), -1)
        except Exception:
            Q = P
        _, jv2 = kd_v.query(Q, k=6)
        n2 = VN[jv2].mean(1); n2 /= (np.linalg.norm(n2, axis=1, keepdims=True) + 1e-12)
        c2 = np.stack([np.abs(n2[:, 1]), (n2[:, 2] + 1) * 0.5, (n2[:, 0] + 1) * 0.5], -1)
        for i in range(len(Q) - 1):
            sp_pts.append([Q[i], Q[i+1]]); sp_cols.append([c2[i], c2[i+1]])
    # skeleton as ordered-ish polylines (greedy nearest walk per branch group)
    try:
        grp_ids = branch if 'branch' in dir() and isinstance(branch, np.ndarray) and len(branch) == len(sk) else np.zeros(len(sk), np.int64)
        for g in np.unique(grp_ids):
            pts_g = sk[grp_ids == g].astype(np.float64)
            if len(pts_g) < 3:
                continue
            used_ = np.zeros(len(pts_g), bool)
            cur = int(np.argmin(pts_g[:, 1]))
            path = [cur]; used_[cur] = True
            for _ in range(len(pts_g) - 1):
                d2 = np.linalg.norm(pts_g - pts_g[path[-1]], axis=1)
                d2[used_] = np.inf
                nx2 = int(np.argmin(d2))
                if not np.isfinite(d2[nx2]) or d2[nx2] > 0.08 * size:
                    break
                path.append(nx2); used_[nx2] = True
            add_polyline(pts_g[path])
    except Exception as e:
        print("skeleton splines skipped:", e, flush=True)
    # cross-section rings every ~5% of height
    try:
        body_m = trimesh.Trimesh(V0, F0, process=False)
        y_lo, y_hi = V0[:, 1].min(), V0[:, 1].max()
        for frac in np.linspace(0.06, 0.94, 16):
            yl = y_lo + (y_hi - y_lo) * frac
            segs3 = trimesh.intersections.mesh_plane(body_m, plane_normal=[0, 1, 0],
                                                     plane_origin=[0, yl, 0])
            if segs3 is None or len(segs3) == 0:
                continue
            for a3, b3 in segs3:
                mid = (a3 + b3) / 2
                _, jv3 = kd_v.query(mid[None], k=6)
                n3 = VN[jv3[0]].mean(0); n3 /= (np.linalg.norm(n3) + 1e-12)
                c3 = np.array([abs(n3[1]), (n3[2] + 1) * 0.5, (n3[0] + 1) * 0.5])
                sp_pts.append([a3, b3]); sp_cols.append([c3, c3])
    except Exception as e:
        print("rings skipped:", e, flush=True)
    print(f"smooth-object fallback added -> total {len(sp_pts):,} segments", flush=True)

sp_pts = np.array(sp_pts, np.float32); sp_cols = np.array(sp_cols, np.float32)
(D / "spline_seg.f32").write_bytes(sp_pts.tobytes())
(D / "spline_col.f32").write_bytes(sp_cols.tobytes())
print(f"splines: {len(sp_pts):,} segments (normal-coloured)", flush=True)

# ---- virtual floor grid ----------------------------------------------------------
lo0 = V0.min(0); hi0 = V0.max(0)
y0 = float(lo0[1]) - 0.01 * size
cx0, cz0 = (lo0[0] + hi0[0]) / 2, (lo0[2] + hi0[2]) / 2
R_f = 0.75 * size
gl_ = []
nline = 9
for i in range(nline):
    a = -R_f + 2 * R_f * i / (nline - 1)
    gl_.append([[cx0 + a, y0, cz0 - R_f], [cx0 + a, y0, cz0 + R_f]])
    gl_.append([[cx0 - R_f, y0, cz0 + a], [cx0 + R_f, y0, cz0 + a]])
(D / "floor_seg.f32").write_bytes(np.array(gl_, np.float32).tobytes())

# ---- runtime-contour data: edges of a smooth CLOSED body + both face normals ----
# viewer draws, per frame, only edges where the surface folds away from the eye
# (sign(n1·v) != sign(n2·v)) -> clean outline from ANY angle, no interior lines
try:
    cpitch = size / 150
    vg_c = smb.voxelized(cpitch)
    occ_c = np.pad(vg_c.matrix.astype(bool), 4)
    occ_c = ndimage.binary_closing(occ_c, iterations=2)
    solid_c = ndimage.binary_fill_holes((blocked(occ_c, 0) + blocked(occ_c, 1) + blocked(occ_c, 2) >= 5) | occ_c)
    edt_c = ndimage.distance_transform_edt(solid_c)
    vv_c, ff_c, _, _ = measure.marching_cubes(edt_c, level=0.7)
    cm = trimesh.Trimesh(vv_c * cpitch + (np.asarray(vg_c.translation) - 4 * cpitch), ff_c, process=True)
    trimesh.smoothing.filter_taubin(cm, lamb=0.5, nu=-0.53, iterations=24)
    cs_c = sorted(cm.split(only_watertight=False), key=lambda c: -len(c.faces))
    keep_c = [c for c in cs_c if len(c.faces) >= 0.05 * len(cm.faces)]
    cm = trimesh.util.concatenate(keep_c) if keep_c else cm
    Vc_ = np.asarray(cm.vertices); Fc_ = np.asarray(cm.faces)
    # snap contour-mesh vertices gently to the true skin for accuracy
    if kd_skin is not None:
        dqc, jqc = kd_skin.query(Vc_, k=1)
        dvc = PP[jqc] - Vc_
        dnc = np.linalg.norm(dvc, axis=1, keepdims=True)
        clampc = 0.02 * size
        Vc_ = Vc_ + dvc * np.minimum(1.0, clampc / np.maximum(dnc, 1e-9)) * 0.35
        cm = trimesh.Trimesh(Vc_, Fc_, process=False)
        trimesh.smoothing.filter_taubin(cm, lamb=0.5, nu=-0.53, iterations=8)
        Vc_ = np.asarray(cm.vertices)
    fn_c = cm.face_normals
    ae = cm.face_adjacency_edges          # (E,2) vertex ids
    af = cm.face_adjacency                # (E,2) face ids
    e_pos = Vc_[ae]                        # (E,2,3)
    e_na = fn_c[af[:, 0]]; e_nb = fn_c[af[:, 1]]
    VNc = np.asarray(cm.vertex_normals)
    n_e = VNc[ae]                          # (E,2,3)
    e_col = np.stack([np.abs(n_e[..., 1]), (n_e[..., 2] + 1) * 0.5, (n_e[..., 0] + 1) * 0.5], -1)
    (D / "cedge_pos.f32").write_bytes(e_pos.astype(np.float32).tobytes())
    (D / "cedge_na.f32").write_bytes(e_na.astype(np.float32).tobytes())
    (D / "cedge_nb.f32").write_bytes(e_nb.astype(np.float32).tobytes())
    (D / "cedge_col.f32").write_bytes(e_col.astype(np.float32).tobytes())
    n_cedge = int(len(e_pos))
    print(f"contour edges: {n_cedge:,} ({len(Fc_):,} faces)", flush=True)
except Exception as e:
    n_cedge = 0
    print("contour data failed:", e, flush=True)

meta = json.loads((D / "mesh.json").read_text())
sets = meta.get("sets", [])
if not any(s.get("file") == "mesh_seg.u8" for s in sets):
    sets.insert(1, {"name": "Segments", "file": "mesh_seg.u8"})
    meta["sets"] = sets
    (D / "mesh.json").write_text(json.dumps(meta, indent=1))
(D / "struct.json").write_text(json.dumps({
    "n_segments": int(nseg), "explode": "struct_explode.f32",
    "creases": {"file": "struct_creases.f32", "n": int(len(segs))},
    "skeleton": {"file": "struct_skel.f32", "n": int(len(sk))},
    "splines": {"seg": "spline_seg.f32", "col": "spline_col.f32", "n": int(len(sp_pts))},
    "contour": {"pos": "cedge_pos.f32", "na": "cedge_na.f32", "nb": "cedge_nb.f32",
                "col": "cedge_col.f32", "n": int(n_cedge)},
    "floor": "floor_seg.f32",
    "explode_cut": {"pos": "explode_pos.f32", "nrm": "explode_nrm.f32",
                    "col": "explode_col.u8", "disp": "explode_disp.f32",
                    "n_verts": int(len(pos2))},
    "shells": shells_meta}, indent=1))
print("OBJECT_STRUCT_DONE", flush=True)
