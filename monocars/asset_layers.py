"""Paint an object's gaussians with the polarization products of its own views.

For a reconstructed object (TRELLIS gaussians placed in the world) we know
every frame the operator / auto-views saw it in, the camera poses and the
per-frame polarization layers (polar_layers.py).  Project each gaussian into
each view, keep the ones that are visible (z-buffer + front-facing), sample
the layer image there and take the median over views -> one colour per
gaussian per product -> extra colour sets for the close-up splat viewer
("look at the car by layers").

Normal products (nxyz_*) are DECODED into camera-frame normals, rotated into
the world and then into the close-up frame before being re-encoded, so the
colours mean the same thing from every side of the object (view-independent,
same frame as the kNN nxyz set).  Scalar / colour products (dolp, aolp_color,
deglare, polar_hsv) are sampled as-is.

Inputs (job dir): asset_pts.ply.up.json, placement.json (place_points),
box.json (roi_frames / obs frames), the scene's splat.splat + splat_frame.json
(asset2splat), poses.json + meta.json of the map, <scene>/layers/<product>/.

Usage: python asset_layers.py <job_dir> <obj_scene_dir> <street_video_dir> <map_scene_dir> [products=...]
Writes <obj_scene_dir>/splat_<product>.splat and updates splat.json["sets"].
"""
import json
import sys
import numpy as np
import cv2
from pathlib import Path

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


# ---- gaussians of the close-up (positions in the close-up frame) -------------
buf = np.frombuffer((OBJ / "splat.splat").read_bytes(), np.uint8).reshape(-1, 32)
X_cu = buf[:, 0:12].view("<f4").reshape(-1, 3).astype(np.float64)
sc_cu = buf[:, 12:24].view("<f4").reshape(-1, 3).astype(np.float64)
N = len(X_cu)
sf = json.loads((OBJ / "splat_frame.json").read_text())
R_cu = np.array(sf["R"]); ctr_cu = np.array(sf["ctr"]); s_cu = float(sf["scale"])
pl = json.loads((JOB / "placement.json").read_text())
R_pp = np.array(pl["R"]); s_pp = float(pl["s"]); t_pp = np.array(pl["t"]); a_ctr = np.array(pl["a_ctr"])
up_w = np.array(pl["up"]); base_off = float(pl["base_off"])
# close-up -> asset -> world
X_a = (X_cu / s_cu) @ R_cu + ctr_cu                       # R_cu.T @ x == x @ R_cu row-wise
X_w = s_pp * (X_a - a_ctr) @ R_pp.T + t_pp + up_w * base_off
# close-up normals (kNN) for the facing test: from splat_n.splat colours (n*0.5+0.5)
nbuf = np.frombuffer((OBJ / "splat_n.splat").read_bytes(), np.uint8).reshape(-1, 32)
n_cu = nbuf[:, 24:27].astype(np.float64) / 255.0 * 2 - 1
n_cu /= np.linalg.norm(n_cu, axis=1, keepdims=True) + 1e-9
n_w = (n_cu @ R_cu) @ R_pp.T                                # rotate like positions (no scale/shift)
# combined rotation world -> close-up for the normal products
R_w2cu = R_cu @ R_pp.T                                       # x_cu = R_cu R_pp^T (x_w - ...) up to scale

# ---- views ----------------------------------------------------------------------
poses = json.loads((MAP / "poses.json").read_text())
meta = json.loads((MAP / "meta.json").read_text())
I = meta["intrinsics"]; fx, fy, cx, cy, Ws, Hs = I["fx"], I["fy"], I["cx"], I["cy"], I["w"], I["h"]
by_name = {f["name"]: f for f in poses["frames"]}
box = json.loads((JOB / "box.json").read_text())
view_names = []
for rf in box.get("roi_frames", []):
    if rf.get("name"): view_names.append(rf["name"])
# every observation frame (manual + auto) from the object's obs, if recorded in box.json
for ob in box.get("obs_frames", []):
    view_names.append(ob)
view_names = [v for v in dict.fromkeys(view_names) if v in by_name]
if not view_names:
    print("no views with poses"); sys.exit(1)
print(f"{N:,} gaussians, {len(view_names)} views: {[v[:26] for v in view_names]}", flush=True)

# ---- per view: project, z-buffer visibility, sample -------------------------------
samples = {p: [] for p in PRODS}          # product -> list of (N,3) float arrays with NaN where unseen
for vn in view_names:
    f = by_name[vn]; R = quat_to_R(f["q"]); C = np.array(f["p"])
    cam = (X_w - C) @ R.T
    z = cam[:, 2]
    ok = z > 0.2
    sx = fx * cam[:, 0] / np.where(ok, z, 1) + cx
    sy = fy * cam[:, 1] / np.where(ok, z, 1) + cy
    inside = ok & (sx >= 0) & (sx < Ws - 1) & (sy >= 0) & (sy < Hs - 1)
    # facing: gaussian normal (world) against the viewing ray
    view_dir = (X_w - C); view_dir /= np.linalg.norm(view_dir, axis=1, keepdims=True) + 1e-9
    facing = (n_w * view_dir).sum(1) < 0.15                  # normal points back toward the camera (roughly)
    # z-buffer at 1/2 px cells: nearest depth per cell, tolerance = 4% of depth
    cell = 2
    gx = (sx / cell).astype(int); gy = (sy / cell).astype(int)
    gw, gh = Ws // cell + 1, Hs // cell + 1
    zb = np.full(gh * gw, np.inf)
    sel = inside
    np.minimum.at(zb, gy[sel] * gw + gx[sel], z[sel])
    # dilate the z-buffer a little so thin gaps don't leak background samples
    zb2 = zb.reshape(gh, gw); zb2 = np.where(np.isinf(zb2), np.nan, zb2)
    zmin = cv2.erode(np.nan_to_num(zb2, nan=1e9).astype(np.float32), np.ones((3, 3), np.uint8))
    vis = inside & facing & (z <= zmin[gy.clip(0, gh - 1), gx.clip(0, gw - 1)] * 1.04 + 1e-3)
    n_vis = int(vis.sum())
    if n_vis < 50:
        print(f"  {vn[:26]}: {n_vis} visible, skipped", flush=True); continue
    xs, ys = sx[vis], sy[vis]
    for p in PRODS:
        lp = VID / "layers" / p / (vn.rsplit(".", 1)[0] + ".jpg")
        if not lp.exists():
            lp = VID / "layers" / p / vn
        img = cv2.imread(str(lp), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = img.astype(np.float32) / 255.0
        # bilinear sample
        x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int); ax = xs - x0; ay = ys - y0
        x1 = np.clip(x0 + 1, 0, img.shape[1] - 1); y1 = np.clip(y0 + 1, 0, img.shape[0] - 1)
        v = (img[y0, x0] * ((1 - ax) * (1 - ay))[:, None] + img[y0, x1] * (ax * (1 - ay))[:, None]
             + img[y1, x0] * ((1 - ax) * ay)[:, None] + img[y1, x1] * (ax * ay)[:, None])      # BGR
        out = np.full((N, 3), np.nan)
        if p in NORMAL_PRODS:
            # canon encoding (2026-08-26): B=|nz|, G=(ny+1)/2, R=(nx+1)/2 (display R=X G=Y B=Z)
            nx = v[:, 2] * 2 - 1; ny = v[:, 1] * 2 - 1; nz = v[:, 0]
            # layers store DISPLAY-frame components (image rotated 90 CW for viewing);
            # back to sensor/COLMAP camera axes: nx_s = ny_d, ny_s = -nx_d
            n_cam = np.stack([ny, -nx, -nz], 1)
            n_cam /= np.linalg.norm(n_cam, axis=1, keepdims=True) + 1e-9
            n_world = n_cam @ R                                  # R^T n (row-wise)
            n_cu_v = n_world @ R_w2cu.T
            out[vis] = n_cu_v
        else:
            out[vis] = v[:, ::-1]                                # RGB
        samples[p].append(out)
    print(f"  {vn[:26]}: {n_vis:,} visible", flush=True)

# ---- aggregate + write ------------------------------------------------------------
sets = json.loads((OBJ / "splat.json").read_text()).get("sets") or [
    {"name": "RGB", "file": "splat.splat"}, {"name": "nxyz (kNN)", "file": "splat_n.splat"}]
sets = [s_ for s_ in sets if s_["file"] in ("splat.splat", "splat_n.splat")]
from scipy.spatial import cKDTree
tree = cKDTree(X_cu)
for p in PRODS:
    if not samples[p]:
        print(p, "no samples"); continue
    S = np.stack(samples[p], 0)                                 # (V, N, 3)
    cnt = np.isfinite(S[..., 0]).sum(0)
    with np.errstate(all="ignore"):
        if p in NORMAL_PRODS:
            # per-gaussian mean direction over views (already in the close-up frame)
            m = np.nanmean(S, 0)
            m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
            # face outward like the kNN set (flip against it where opposite)
            flip = (m * n_cu).sum(1) < 0
            m[flip] *= -1
            col = np.clip(m * 0.5 + 0.5, 0, 1)
        else:
            col = np.clip(np.nanmedian(S, 0), 0, 1)
    seen = cnt > 0
    # unseen gaussians borrow the nearest seen one's colour (the far side of the object)
    if seen.any() and (~seen).any():
        tr = cKDTree(X_cu[seen]); _, nn = tr.query(X_cu[~seen], k=1)
        col[~seen] = col[seen][nn]
    col = np.nan_to_num(col, nan=0.5)
    # surface smoothing: the raw polarization planes carry per-pixel dither and a
    # little view misalignment; averaging over each gaussian's neighbourhood on
    # the object gives the per-surface reading the layer is about
    K = 24
    _, nn = tree.query(X_cu, k=min(K, N))
    if p in NORMAL_PRODS:
        v = col * 2 - 1
        v = v[nn].mean(1); v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        col = np.clip(v * 0.5 + 0.5, 0, 1)
    else:
        col = np.median(col[nn], axis=1)
        # scalar layers: stretch to the object's own range so a dark car still
        # shows its DoLP structure. TRUE-COLOUR layers (rgb_deglare) keep their
        # photographic values -- the stretch hue-shifts and clips them.
        if p not in ("rgb_deglare",):
            lum = col.mean(1)
            lo, hi = np.percentile(lum, 2), np.percentile(lum, 98)
            if hi - lo > 1e-3:
                col = np.clip((col - lo) / (hi - lo), 0, 1)
    outbuf = buf.copy()
    outbuf[:, 24:27] = (col * 255).astype(np.uint8)
    fn = f"splat_{p}.splat"
    (OBJ / fn).write_bytes(outbuf.tobytes())
    sets.append({"name": LABELS.get(p, p), "file": fn, "seen_frac": round(float(seen.mean()), 3)})
    print(f"{p}: seen {seen.mean()*100:.0f}% of gaussians (views/gaussian median {int(np.median(cnt[seen])) if seen.any() else 0}) -> {fn}", flush=True)
sj = json.loads((OBJ / "splat.json").read_text()); sj["sets"] = sets
(OBJ / "splat.json").write_text(json.dumps(sj))
print("ASSET_LAYERS_DONE", len(sets), "sets", flush=True)
