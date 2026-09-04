"""NAST Inspector — one service that ties recorder + map + 3D + jobs together.

No manual scripts: the browser draws an ROI, POSTs it here, this process
solves the 3D box against the dense cloud, stores it, and every view reads
back from the same store. Heavy GPU reconstruction is enqueued to tex1.

Run:  python server.py [port]      then open http://localhost:8130/
Deps: stdlib + numpy (+ scipy optional). No web framework, so it just runs.
"""
import json
import os
import sqlite3
import sys
import threading
import time
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, unquote

import numpy as np

HERE = Path(__file__).parent
VIEWER = HERE.parent / "viewer"
CLOUD_DIR = VIEWER / "scenes" / "street"           # dense cloud + poses (packed)
VIDEO_DIR = VIEWER / "scenes" / "street_video"     # rgb/ + nxyz/ frames
# reconstruction crops come from the DEGLARED frames (Stokes-minimum colour):
# specular sheen on glass/paint never reaches TRELLIS. NAST_DEGLARE_CROPS=0
# reverts to the plain rgb frames.
DEGLARE_CROPS = os.environ.get("NAST_DEGLARE_CROPS", "1") != "0"


# fully-local pipeline (2026-08-27): no tex1 anywhere. Objects come from the
# dense map points inside the solved box (no generative model), the street map
# is rebuilt by the local VGGT-Omega runner (12-16 GB VRAM). NAST_LOCAL=0
# restores the tex1 TRELLIS/Omega route when the GPU server is reachable.
LOCAL_ONLY = os.environ.get("NAST_LOCAL", "1") != "0"
TRELLIS_LOCAL = os.environ.get("NAST_TRELLIS", "1") != "0"   # 0 = never use the local TRELLIS env
LOCAL_GPU = HERE.parent / "local_gpu"

# ---- live normals decode (2026-09-03): ship rgb/ + raw/ only; every locked
# polarization product is decoded ON DEMAND from the RAW12 frame the first
# time a viewer asks for it, then written through into layers/ as a cache —
# no offline layer pass, and reconstruction finds the files it samples.
LIVE_PRODUCTS = ("nxyz", "n_xy", "n_xz", "nxyz_phys", "nxyz_diffuse",
                 "nxyz_specv2", "edge", "rgb_deglare")
_live_locks = {}
_live_guard = threading.Lock()


def raw_dir(scene=None):
    """where a scene's raw frames live: <scene>/raw/ or the scene folder itself
    (the operator drops a folder of .raw12 files); None when there is no raw"""
    scene = Path(scene) if scene else VIDEO_DIR
    for d in (scene / "raw", scene):
        if d.is_dir() and (next(d.glob("*.raw12"), None) or next(d.glob("*.raw"), None)):
            return d
    return None


def raw_frames(scene=None):
    d = raw_dir(scene)
    return sorted(set(d.glob("*.raw12")) | set(d.glob("*.raw")), key=lambda q: q.name) if d else []


def _frame_complete(stem):
    return ((VIDEO_DIR / "rgb" / f"{stem}.jpg").exists() and
            (VIDEO_DIR / "layers" / LIVE_PRODUCTS[0] / f"{stem}.jpg").exists())


def live_decode(stem):
    """decode one frame from raw: rgb/ + rgb_orig/ (when missing) and every
    locked product into layers/ -- the on-demand twin of decode_raw.py"""
    d = raw_dir()
    raw = None
    if d:
        for ext in (".raw12", ".raw"):
            if (d / (stem + ext)).exists():
                raw = d / (stem + ext); break
    if raw is None:
        return False
    with _live_guard:
        lk = _live_locks.setdefault(stem, threading.Lock())
    with lk:
        if _frame_complete(stem):
            return True                        # another thread already did it
        import cv2
        import importlib
        sys.path.insert(0, str(HERE.parent / "monocars"))
        pn = importlib.import_module("polar_normals")
        fr = pn.PolarFrame(raw.read_bytes())
        for sub, fn in (("rgb", fr.color_work), ("rgb_orig", fr.color_recorder)):
            out = VIDEO_DIR / sub / f"{stem}.jpg"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), fn(), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        P = pn.products(fr)
        for k in LIVE_PRODUCTS:
            if k not in P:
                continue
            d2 = VIDEO_DIR / "layers" / k
            d2.mkdir(parents=True, exist_ok=True)
            if not (d2 / f"{stem}.jpg").exists():
                cv2.imwrite(str(d2 / f"{stem}.jpg"), P[k], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return True


def ensure_layer_frames(names):
    """make sure the bake steps find every product for these frames"""
    if raw_dir() is None:
        return
    for n in names:
        stem = Path(n).stem
        if not _frame_complete(stem):
            try:
                live_decode(stem)
            except Exception as e:
                print("live decode failed:", stem, e, flush=True)


# ---- up-front batch decode (raw import -> locked catalog, with progress) ----------
DECODE = {"running": False, "total": 0, "done": 0, "err": 0, "started": 0}


def decode_counts():
    """(raw frames, frames with rgb AND the catalog) -- rgb counts too now that
    a raw-only folder is a valid input"""
    total = len(raw_frames())
    if not total:
        return 0, 0
    nx = VIDEO_DIR / "layers" / LIVE_PRODUCTS[0]
    have_nx = {q.stem for q in nx.glob("*.jpg")} if nx.exists() else set()
    rgb = VIDEO_DIR / "rgb"
    have_rgb = {q.stem for q in rgb.glob("*.jpg")} if rgb.exists() else set()
    return total, len(have_nx & have_rgb)


def run_decode():
    try:
        total, _ = decode_counts()
        DECODE.update(running=True, total=total, err=0, started=time.time())
        proc = subprocess.Popen([sys.executable, "-u", str(MONO / "decode_raw.py"),
                                 str(VIDEO_DIR), "6"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("DECODE_PROGRESS") or line.startswith("DECODE_DONE"):
                _, done = decode_counts()
                DECODE["done"] = done
            if line.startswith("ERR"):
                DECODE["err"] += 1
        proc.wait()
        _, done = decode_counts()
        DECODE["done"] = done
    except Exception as e:
        print("decode run failed:", e, flush=True)
    finally:
        DECODE["running"] = False


def start_decode_if_needed():
    """kick the batch decode when raw/ exists but the catalog is incomplete"""
    if DECODE["running"] or raw_dir() is None:
        return
    total, done = decode_counts()
    if total and done < total:
        threading.Thread(target=run_decode, daemon=True).start()


def open_dataset(path):
    """point the service at another scene folder and start its decode when
    needed. Raw is enough: a folder of .raw12 files (or <scene>/raw/) yields
    rgb/, rgb_orig/ and the normals catalog by itself; a folder that already
    has rgb/ works as before. The map/poses stay; frames, raw, layers,
    reconstruction crops all follow the new folder."""
    global VIDEO_DIR, DEPTH_DIR
    p = Path(path).expanduser().resolve()
    if raw_dir(p) is None and not (p / "rgb").exists():
        raise ValueError(f"no .raw12 frames and no rgb/ inside {p}")
    (p / "rgb").mkdir(exist_ok=True)
    VIDEO_DIR = p
    DEPTH_DIR = p / "depth"
    DECODE.update(running=False, done=0, err=0, started=0)
    start_decode_if_needed()
    total, done = decode_counts()
    return {"path": str(p), "frames": len(list((p / "rgb").glob("*.jpg"))),
            "raw": total, "decoded": done, "decoding": DECODE["running"]}


def build_point_asset(box, out_ply, cap=1_500_000, margin=1.25):
    """Local (no-TRELLIS) asset: dense-cloud points inside the solved box,
    written as a TRELLIS-style gaussian PLY so the whole downstream chain
    (asset2ply, place_points, asset2splat, layer baking) runs unchanged."""
    pos = np.fromfile(str(CLOUD_DIR / "pos.f32"), dtype=np.float32).reshape(-1, 3)
    rgb = np.fromfile(str(CLOUD_DIR / "rgb.u8"), dtype=np.uint8).reshape(-1, 3)
    ctr = np.asarray(box["center"], np.float64)
    half = np.asarray(box["size"], np.float64) * 0.5 * margin
    pose = box.get("pose") or {}
    if pose.get("R"):
        # OBB -> conservative AABB half-extents (|R| bounds either convention)
        half = np.abs(np.asarray(pose["R"], np.float64)) @ half
    m = (np.abs(pos - ctr) <= half).all(1)
    idx = np.where(m)[0]
    if len(idx) < 500:
        raise RuntimeError(f"only {len(idx)} map points inside the box")
    if len(idx) > cap:
        idx = np.random.RandomState(0).choice(idx, cap, replace=False)
    X = pos[idx].astype(np.float64)
    C = rgb[idx].astype(np.float64) / 255.0
    from scipy.spatial import cKDTree
    probe = X[np.random.RandomState(1).choice(len(X), min(20000, len(X)), replace=False)]
    d, _ = cKDTree(X).query(probe, k=2)
    r = float(np.median(d[:, 1])) * 1.6 + 1e-6
    C0 = 0.28209479177387814
    n = len(X)
    names = (["x", "y", "z"] + [f"f_dc_{i}" for i in range(3)] + ["opacity"]
             + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {n}\n"
           + "".join(f"property float {nm}\n" for nm in names)
           + "end_header\n").encode()
    rec = np.zeros(n, dtype=np.dtype([(nm, "<f4") for nm in names]))
    rec["x"], rec["y"], rec["z"] = X[:, 0], X[:, 1], X[:, 2]
    for i in range(3):
        rec[f"f_dc_{i}"] = (C[:, i] - 0.5) / C0
    rec["opacity"] = 3.0                       # sigmoid -> 0.95
    for i in range(3):
        rec[f"scale_{i}"] = np.log(r)
    rec["rot_0"] = 1.0                         # identity quaternion (w,x,y,z)
    with open(out_ply, "wb") as f:
        f.write(hdr); rec.tofile(f)
    print(f"point asset: {n:,} pts, radius {r:.4f} -> {out_ply}", flush=True)
    return n


def crop_frame_src(frame_name):
    """frame path for reconstruction crops: the working rgb stream (SOFT
    deglare -- blowout removed, brightness floor keeps grazing-surface
    shading so TRELLIS doesn't punch holes in roofs/glass).
    NAST_DEGLARE_CROPS=0 -> the original S0 frames (rgb_orig)."""
    if not DEGLARE_CROPS:
        og = VIDEO_DIR / "rgb_orig" / frame_name
        if og.exists():
            return og
    return VIDEO_DIR / "rgb" / frame_name
DB = HERE / "inspector.db"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8130

# ------------------------------------------------------------------ state
STATE = {"pos": None, "poses": None, "meta": None, "by_name": {}}
LOCK = threading.Lock()


def load_scene():
    # poses/meta live in scene_base so a clean re-process (no clouds yet)
    # still boots; the cloud itself is optional — depth solving needs none
    base = HERE / "scene_base"
    src = CLOUD_DIR if (CLOUD_DIR / "meta.json").exists() else base
    meta = json.loads((src / "meta.json").read_text())
    poses = json.loads((src / "poses.json").read_text())
    if (CLOUD_DIR / "pos.f32").exists():
        pos = np.fromfile(CLOUD_DIR / "pos.f32", dtype=np.float32).reshape(-1, 3)
    else:
        pos = np.zeros((0, 3), np.float32)
    STATE["meta"] = meta
    STATE["poses"] = poses
    STATE["pos"] = pos
    STATE["by_name"] = {f["name"]: f for f in poses["frames"]}
    print(f"[scene] cloud {len(pos):,} pts, {len(poses['frames'])} poses", flush=True)
    # raw import present but catalog not fully decoded -> kick the batch decode
    try:
        start_decode_if_needed()
    except Exception as e:
        print("decode autostart skipped:", e, flush=True)


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    c = db()
    c.executescript("""
      CREATE TABLE IF NOT EXISTS objects(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT, frame TEXT, cam TEXT, kind TEXT, pts TEXT,
        cx REAL, cy REAL, cz REAL, sx REAL, sy REAL, sz REAL,
        npts INTEGER, created REAL);
      CREATE TABLE IF NOT EXISTS jobs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        object_id INTEGER, kind TEXT, status TEXT, detail TEXT, created REAL);
    """)
    for ddl in ("ALTER TABLE objects ADD COLUMN pose TEXT",
                "ALTER TABLE objects ADD COLUMN obs TEXT"):
        try:
            c.execute(ddl)
        except sqlite3.OperationalError:
            pass                                          # column already there
    c.commit(); c.close()


# ------------------------------------------------------------ ROI -> 3D box
def point_in_poly(px, py, poly):
    """Vectorised even-odd test: px,py are (N,), poly is (K,2)."""
    n = len(poly)
    inside = np.zeros(len(px), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        cond = ((yi > py) != (yj > py)) & \
               (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi)
        inside ^= cond
        j = i
    return inside


def roi_to_polygon(kind, pts):
    pts = np.array(pts, dtype=np.float64)
    if kind == "rect":
        (x0, y0), (x1, y1) = pts[0], pts[1]
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    if kind == "circle":
        c, e = pts[0], pts[1]
        r = np.hypot(*(e - c))
        t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], -1)
    return pts                                          # polygon


DEPTH_DIR = VIDEO_DIR / "depth"
_DCACHE = {}


def roi_points_depth(kind, pts, frame_name):
    """MoGe per-pixel depth: every ROI pixel becomes a world point, so the
    object boundary is sharp where the global cloud is sparse. Returns (N,3)
    or None when this frame's depth map isn't dumped yet."""
    dpath = DEPTH_DIR / (frame_name.rsplit(".", 1)[0] + ".png")
    f = STATE["by_name"].get(frame_name)
    if f is None or not dpath.exists():
        return None
    import cv2
    if frame_name in _DCACHE:
        d16 = _DCACHE[frame_name]
    else:
        d16 = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
        if d16 is None:
            return None
        if len(_DCACHE) > 24:
            _DCACHE.clear()
        _DCACHE[frame_name] = d16
    dh, dw = d16.shape
    I = STATE["meta"]["intrinsics"]
    Hs, Ws = I["h"], I["w"]
    R = quat_to_R(f["q"]); C = np.array(f["p"])

    poly = np.array(roi_to_polygon(kind, pts), dtype=np.float64)
    # upright poly -> sensor window (sx=dy, sy=(H-1)-dx)
    sx0 = max(0, int(poly[:, 1].min())); sx1 = min(Ws, int(poly[:, 1].max()) + 1)
    sy0 = max(0, int((Hs - 1) - poly[:, 0].max())); sy1 = min(Hs, int((Hs - 1) - poly[:, 0].min()) + 1)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    step = max(1, int(np.sqrt(max(1, (sx1 - sx0) * (sy1 - sy0)) / 150_000)))
    gy, gx = np.mgrid[sy0:sy1:step, sx0:sx1:step]
    gx = gx.ravel().astype(np.float64); gy = gy.ravel().astype(np.float64)
    ux = (Hs - 1) - gy; uy = gx
    inside = point_in_poly(ux, uy, poly)
    gx, gy = gx[inside], gy[inside]
    if len(gx) < 40:
        return None
    zi = d16[(gy * dh / Hs).astype(np.int32).clip(0, dh - 1),
             (gx * dw / Ws).astype(np.int32).clip(0, dw - 1)].astype(np.float64) / 1000.0
    ok = zi > 1e-3
    gx, gy, zi = gx[ok], gy[ok], zi[ok]
    if len(gx) < 40:
        return None
    # nearest depth cluster, same policy as the cloud path
    z_lo = np.percentile(zi, 15)
    band = (zi >= z_lo - 0.2) & (zi <= z_lo + 0.6)
    if band.sum() >= 40:
        gx, gy, zi = gx[band], gy[band], zi[band]
    xc = (gx - I["cx"]) / I["fx"] * zi
    yc = (gy - I["cy"]) / I["fy"] * zi
    P = np.stack([xc, yc, zi], -1) @ R + C          # R.T applied row-wise
    if len(P) > 60_000:
        rs = np.random.RandomState(0)
        P = P[rs.choice(len(P), 60_000, replace=False)]
    return P


def roi_ray_points(kind, pts, frame_name, max_ang=0.012):
    """Thin-object path: cast the ray through the ROI centre and gather the
    map points that lie inside a narrow cone around it (angular radius
    max_ang rad ≈ 0.7°). Depth maps miss lamp heads against bright sky and
    the ROI footprint holds a handful of cloud points -- but the map does
    contain the pole, and the cone finds it regardless of the ROI's size."""
    f = STATE["by_name"].get(frame_name)
    X = STATE["pos"]
    if f is None or len(X) == 0:
        return None
    I = STATE["meta"]["intrinsics"]
    Hs = I["h"]
    poly = np.array(roi_to_polygon(kind, pts), dtype=np.float64)
    ux, uy = poly[:, 0].mean(), poly[:, 1].mean()      # upright centre
    sx, sy = uy, (Hs - 1) - ux                          # -> sensor px
    R = quat_to_R(f["q"]); C = np.array(f["p"])
    dcam = np.array([(sx - I["cx"]) / I["fx"], (sy - I["cy"]) / I["fy"], 1.0])
    dcam /= np.linalg.norm(dcam)
    dw = R.T @ dcam                                     # world ray
    V = X - C
    t = V @ dw
    front = t > 0.3
    if not front.any():
        return None
    perp = np.linalg.norm(V[front] - np.outer(t[front], dw), axis=1)
    ang = perp / t[front]
    hit = np.where(front)[0][ang < max_ang]
    if len(hit) < 20:
        return None
    # nearest depth cluster along the ray, tight band (thin objects)
    tt = t[hit]
    t_lo = np.percentile(tt, 10)
    band = (tt >= t_lo - 0.15) & (tt <= t_lo + 0.9)
    return X[hit[band] if band.sum() >= 20 else hit]


def roi_points(kind, pts, frame_name):
    """ROI -> world points. THE MAP ITSELF comes first: solving against the
    loaded cloud guarantees the marker lands exactly on the cloud's geometry.
    Small ROIs (thin objects) use the ray-cone against the map; MoGe depth
    is the fallback when there is no map yet."""
    Pc = roi_points_cloud(kind, pts, frame_name)
    if Pc is not None and len(Pc) >= 300:
        return Pc
    poly = np.array(roi_to_polygon(kind, pts), dtype=np.float64)
    span = max(np.ptp(poly[:, 0]), np.ptp(poly[:, 1]))
    if span < 90:                                       # tiny ROI -> ray-cone
        Pr = roi_ray_points(kind, pts, frame_name)
        if Pr is not None:
            return Pr
    Pd = roi_points_depth(kind, pts, frame_name)
    if Pd is not None:
        return Pd
    return Pc


def roi_points_cloud(kind, pts, frame_name):
    """Dense-cloud points whose UPRIGHT projection lands inside the ROI,
    cut to the nearest depth cluster. Returns (N,3) world points or None."""
    f = STATE["by_name"].get(frame_name)
    if f is None:
        return None
    I = STATE["meta"]["intrinsics"]
    Hs, Ws = I["h"], I["w"]                              # sensor dims (sideways)
    R = quat_to_R(f["q"]); C = np.array(f["p"])
    X = STATE["pos"]

    cam = (X - C) @ R.T                                  # world->camera
    z = cam[:, 2]
    front = z > 1e-3
    sx = np.empty(len(X)); sy = np.empty(len(X))
    sx[front] = I["fx"] * cam[front, 0] / z[front] + I["cx"]
    sy[front] = I["fy"] * cam[front, 1] / z[front] + I["cy"]
    inframe = front & (sx >= 0) & (sx < Ws) & (sy >= 0) & (sy < Hs)

    # sensor -> upright (image is drawn rotated 90 CW in the recorder)
    dx = (Hs - 1) - sy
    dy = sx
    poly = roi_to_polygon(kind, pts)
    idx = np.where(inframe)[0]
    hit = idx[point_in_poly(dx[idx], dy[idx], poly)]
    if len(hit) < 12 or len(X) == 0:
        return None

    zc = z[hit]
    # nearest depth cluster: densest 0.4 m band around the closest quartile
    z_lo = np.percentile(zc, 15)
    band = (zc >= z_lo - 0.2) & (zc <= z_lo + 0.6)
    obj = hit[band] if band.sum() >= 12 else hit
    return X[obj]


def depth_map(frame_name):
    """MoGe depth (mm u16, sensor orientation) for a frame, cached; None if absent."""
    dpath = DEPTH_DIR / (frame_name.rsplit(".", 1)[0] + ".png")
    if not dpath.exists():
        return None
    if frame_name in _DCACHE:
        return _DCACHE[frame_name]
    import cv2
    d16 = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
    if d16 is None:
        return None
    if len(_DCACHE) > 24:
        _DCACHE.clear()
    _DCACHE[frame_name] = d16
    return d16


def depth_at(frame_name, sx, sy):
    """MoGe depth (m) at sensor pixels; None when the frame has no depth dump."""
    d16 = depth_map(frame_name)
    if d16 is None:
        return None
    I = STATE["meta"]["intrinsics"]; Hs, Ws = I["h"], I["w"]
    dh, dw = d16.shape
    z = d16[(np.asarray(sy) * dh / Hs).astype(np.int32).clip(0, dh - 1),
            (np.asarray(sx) * dw / Ws).astype(np.int32).clip(0, dw - 1)].astype(np.float64) / 1000.0
    return z


def world_up():
    up = np.array(STATE["poses"].get("up") or STATE["meta"]["up"], dtype=np.float64)
    return up / np.linalg.norm(up)


def object_world_points(obs, cap=3000):
    """Union of the OPERATOR's ROI points (auto views excluded), trimmed."""
    parts = []
    for ob in obs:
        if ob.get("auto"):
            continue
        P = roi_points(ob["kind"], ob["pts"], ob["frame"])
        if P is not None:
            parts.append(P)
    if not parts:
        return None
    P = np.concatenate(parts, 0)
    med = np.median(P, axis=0)
    d = np.linalg.norm(P - med, axis=1)
    P = P[d < np.percentile(d, 92) + 1e-6]
    if len(P) > cap:
        P = P[np.random.RandomState(0).choice(len(P), cap, replace=False)]
    return P


def project_upright(P, f):
    """World points -> upright pixels (ux, uy), sensor pixels, camera depth z, in-frame mask."""
    I = STATE["meta"]["intrinsics"]; Hs, Ws = I["h"], I["w"]
    R = quat_to_R(f["q"]); C = np.array(f["p"])
    cam = (P - C) @ R.T
    z = cam[:, 2]
    front = z > 0.3
    sx = np.full(len(P), -1.0); sy = np.full(len(P), -1.0)
    sx[front] = I["fx"] * cam[front, 0] / z[front] + I["cx"]
    sy[front] = I["fy"] * cam[front, 1] / z[front] + I["cy"]
    inside = front & (sx >= 0) & (sx < Ws) & (sy >= 0) & (sy < Hs)
    ux = (Hs - 1) - sy; uy = sx
    return ux, uy, sx, sy, z, inside


def sam_hints(frame_name, P, rect, n_pos=8, n_neg=8):
    """Prompt points for SAM in UPRIGHT frame coords: positives = the object's
    own world points seen in this frame (depth-consistent when MoGe agrees),
    negatives = pixels inside the ROI whose MoGe depth is clearly not the
    object (sky behind a lamp, wall behind a car). Class-agnostic."""
    f = STATE["by_name"].get(frame_name)
    if f is None or P is None or len(P) < 5:
        return [], []
    ux, uy, sx, sy, z, inside = project_upright(P, f)
    if inside.sum() < 5:
        return [], []
    x0, y0, x1, y1 = rect
    inr = inside & (ux >= x0) & (ux <= x1) & (uy >= y0) & (uy <= y1)
    if inr.sum() < 5:
        inr = inside
    zd = depth_at(frame_name, sx[inr], sy[inr])
    idx = np.where(inr)[0]
    if zd is not None:
        agree = np.abs(zd - z[inr]) / np.maximum(z[inr], 1e-3) < 0.25
        if agree.sum() >= 5:
            idx = idx[agree]
    # positives from the object's CORE: the ROI's cloud points include ground
    # under a car and stray edge points -- drop the lowest 12% (world up) and
    # the outer 30% by lateral radius before spreading the prompts
    try:
        upw = world_up()
        h_ = P[idx] @ upw
        core = h_ > np.percentile(h_, 12)
        ctr_ = P[idx].mean(0)
        lat_ = P[idx] - ctr_; lat_ -= np.outer(lat_ @ upw, upw)
        r_ = np.linalg.norm(lat_, axis=1)
        core &= r_ <= np.percentile(r_, 70)
        if core.sum() >= 5:
            idx = idx[core]
    except Exception:
        pass
    # spread positives over the object's vertical extent
    order = idx[np.argsort(uy[idx])]
    pick = order[np.linspace(0, len(order) - 1, min(n_pos, len(order))).astype(int)]
    pos = [[float(ux[i]), float(uy[i])] for i in pick]
    # negatives: grid over the ROI, keep pixels whose depth disagrees strongly
    neg = []
    zmed = float(np.median(z[inr]))
    # the object's own depth span in this view: a car seen from close by spans
    # metres, so "in front" / "behind" are judged against its near/far ends
    z_near = float(np.percentile(z[inr], 5)); z_far = float(np.percentile(z[inr], 95))
    gx, gy = np.meshgrid(np.linspace(x0, x1, 9)[1:-1], np.linspace(y0, y1, 9)[1:-1])
    gx = gx.ravel(); gy = gy.ravel()
    gsx = gy; gsy = (STATE["meta"]["intrinsics"]["h"] - 1) - gx          # upright -> sensor
    zg = depth_at(frame_name, gsx, gsy)
    if zg is not None:
        far = np.where((zg > z_far * 1.35) | (zg < z_near * 0.7) | (zg <= 1e-3))[0]
        span = max(x1 - x0, y1 - y0, 1.0)
        # luminance gate: MoGe reports sky depth on thin parts too (a lamp's arm),
        # so a negative must also LOOK unlike the object -- differ from the
        # positives' brightness by a clear margin
        lum = None
        try:
            import cv2
            im = cv2.imread(str(VIDEO_DIR / "rgb" / frame_name))
            if im is not None:
                lum = cv2.cvtColor(np.rot90(im, k=3).copy(), cv2.COLOR_BGR2GRAY)
        except Exception:
            lum = None
        lpos = None
        if lum is not None and pos:
            hh, ww = lum.shape
            lpos = float(np.median([lum[int(min(hh - 1, max(0, q[1]))), int(min(ww - 1, max(0, q[0])))] for q in pos]))
        for i in far:                                # keep negatives clear of the positives
            if not all(np.hypot(gx[i] - q[0], gy[i] - q[1]) > 0.08 * span for q in pos):
                continue
            # farther-than-object pixels (sky-like) must also look unlike the
            # object; nearer ones are occluders (a lamp in front, a bush) and
            # are wanted as negatives whatever their brightness
            nearer = zg[i] > 1e-3 and zg[i] < z_near * 0.7
            if lpos is not None and not nearer:
                lv = float(lum[int(min(hh - 1, max(0, gy[i]))), int(min(ww - 1, max(0, gx[i])))])
                if abs(lv - lpos) < 28:
                    continue
            neg.append([float(gx[i]), float(gy[i])])
        if len(neg) > n_neg:
            neg = [neg[i] for i in np.linspace(0, len(neg) - 1, n_neg).astype(int)]
    return pos, neg


def auto_views(obj, n=5, exclude_near=4, min_span=36):
    """Propagate the operator's ROI to other frames by GEOMETRY: project the
    object's world points into every camera pose, keep frames where it is
    fully in view, big enough and not occluded (MoGe depth agrees with the
    projected depth), then pick n views spread in bearing around the object.
    Returns obs entries {frame, cam, kind: rect, pts, auto: True}."""
    obs = json.loads(obj["obs"] or "[]") if isinstance(obj.get("obs"), str) else (obj.get("obs") or [])
    P = object_world_points(obs)
    if P is None:
        return []
    # the ROI points cover only what the map has (a dark roof is missing); the
    # operator's ROI polygons say what the object IS. Lift each manual ROI to a
    # billboard at the object's depth (rays through its corners cut at the
    # median object depth) and project those quads too -- consistent extents in
    # every view (TRELLIS multi-image needs the same object framed the same way)
    corners = None
    try:
        quads = []
        for ob in obs:
            if ob.get("auto"):
                continue
            fm = STATE["by_name"].get(ob["frame"])
            if fm is None:
                continue
            Rm = quat_to_R(fm["q"]); Cm = np.array(fm["p"])
            zc = ((P - Cm) @ Rm.T)[:, 2]
            zc = zc[zc > 0.3]
            if len(zc) < 5:
                continue
            zq = np.percentile(zc, [15, 85])
            I_ = STATE["meta"]["intrinsics"]
            poly = np.array(roi_to_polygon(ob["kind"], ob["pts"]), dtype=np.float64)
            for zz in zq:
                for ux_, uy_ in poly:
                    sx_, sy_ = uy_, (I_["h"] - 1) - ux_               # upright -> sensor
                    d = np.array([(sx_ - I_["cx"]) / I_["fx"], (sy_ - I_["cy"]) / I_["fy"], 1.0]) * zz
                    quads.append(Rm.T @ d + Cm)
        if quads:
            corners = np.array(quads)
    except Exception:
        corners = None
    up = world_up()
    ref = np.array([0, 1, 0]) if abs(up[1]) < 0.9 else np.array([1, 0, 0])
    wx = np.cross(ref, up); wx /= np.linalg.norm(wx); wz = np.cross(up, wx)
    ctr = P.mean(0)

    def bearing(C):
        v = C - ctr; v = v - up * (v @ up)
        return float(np.arctan2(v @ wz, v @ wx))

    frames = STATE["poses"]["frames"]
    idx_of = {f["name"]: i for i, f in enumerate(frames)}
    used = [(idx_of.get(ob["frame"], -99), ob["frame"][:1]) for ob in obs]
    have = [bearing(np.array(frames[i]["p"])) for i, _ in used if i >= 0]
    I = STATE["meta"]["intrinsics"]; Hs, Ws = I["h"], I["w"]

    def depth_vis(f):
        """fraction of the object's points whose MoGe depth agrees with the projection"""
        ux, uy, sx, sy, z, inside = project_upright(P, f)
        if inside.sum() < 5:
            return None
        zd = depth_at(f["name"], sx[inside], sy[inside])
        if zd is None:
            return None
        zi = z[inside]
        return float(np.mean(np.abs(zd - zi) / np.maximum(zi, 1e-3) < 0.25))

    # self-calibration: if MoGe cannot even confirm the object in the frames the
    # operator drew it in (thin poles against sky, glass), the occlusion test is
    # meaningless for this object -> switch it off instead of rejecting everything
    own = [depth_vis(frames[i]) for i, _ in used if i >= 0]
    own = [v for v in own if v is not None]
    use_depth = bool(own) and float(np.mean(own)) >= 0.5
    cands = []
    for i, f in enumerate(frames):
        cam = f["name"][:1]
        if any(c == cam and abs(i - j) <= exclude_near for j, c in used):
            continue
        ux, uy, sx, sy, z, inside = project_upright(P, f)
        if inside.mean() < 0.97:
            continue
        x0, x1 = np.percentile(ux[inside], [1, 99]); y0, y1 = np.percentile(uy[inside], [1, 99])
        if corners is not None:
            cx_, cy_, _, _, cz_, cin = project_upright(corners, f)
            if cin.all():
                x0, x1 = min(x0, cx_.min()), max(x1, cx_.max())
                y0, y1 = min(y0, cy_.min()), max(y1, cy_.max())
        span = float(max(x1 - x0, y1 - y0))
        if span < min_span:
            continue
        pad = max(12.0, 0.12 * span)                              # room for the unseen side of the object
        if x0 < pad or y0 < pad or x1 > Hs - 1 - pad or y1 > Ws - 1 - pad:   # upright frame: Hs wide, Ws tall
            continue
        vis = 0.5
        if use_depth:
            zd = depth_at(f["name"], sx[inside], sy[inside])
            if zd is not None:
                zi = z[inside]
                vis = float(np.mean(np.abs(zd - zi) / np.maximum(zi, 1e-3) < 0.25))
                if vis < 0.45:
                    continue
        cands.append({"i": i, "name": f["name"], "cam": cam,
                      "rect": [float(x0), float(y0), float(x1), float(y1)],
                      "span": span, "vis": vis, "bearing": bearing(np.array(f["p"])),
                      "dist": float(np.median(z[inside]))})
    # photometric check for small/far objects, where a pose error of a few
    # pixels throws the rect off the target: template-match the operator's crop
    # (rescaled) around the projected rect; drop misses, re-centre hits. Big,
    # near objects skip it -- their appearance changes too much between views.
    manual = [ob for ob in obs if not ob.get("auto")]
    if manual and cands:
        try:
            import cv2
            mo = manual[0]
            mpoly = np.array(roi_to_polygon(mo["kind"], mo["pts"]), dtype=np.float64)
            mx0, my0 = int(mpoly[:, 0].min()), int(mpoly[:, 1].min())
            mx1, my1 = int(mpoly[:, 0].max()), int(mpoly[:, 1].max())
            mspan = float(max(mx1 - mx0, my1 - my0, 1))
            mimg = cv2.imread(str(VIDEO_DIR / "rgb" / mo["frame"]))
            mup = cv2.cvtColor(np.rot90(mimg, k=3).copy(), cv2.COLOR_BGR2GRAY)
            mm = int(0.08 * mspan)
            templ0 = mup[max(0, my0 - mm):my1 + mm, max(0, mx0 - mm):mx1 + mm]
            kept = []
            for c in cands:
                if c["span"] >= 140:
                    kept.append(c); continue
                sc_ = c["span"] / mspan
                th, tw = max(8, int(templ0.shape[0] * sc_)), max(8, int(templ0.shape[1] * sc_))
                templ = cv2.resize(templ0, (tw, th), interpolation=cv2.INTER_AREA)
                x0, y0, x1, y1 = c["rect"]
                # small search window: verify the object is where the geometry
                # says (+-25% of its size), do not go hunting across the frame
                sp_ = 0.25 * max(x1 - x0, y1 - y0)
                X0, Y0 = int(max(0, x0 - sp_)), int(max(0, y0 - sp_))
                X1, Y1 = int(min(Hs, x1 + sp_)), int(min(Ws, y1 + sp_))
                cimg = cv2.imread(str(VIDEO_DIR / "rgb" / c["name"]))
                if cimg is None:
                    continue
                cup = cv2.cvtColor(np.rot90(cimg, k=3).copy(), cv2.COLOR_BGR2GRAY)
                win = cup[Y0:Y1, X0:X1]
                if win.shape[0] <= th or win.shape[1] <= tw:
                    continue
                res = cv2.matchTemplate(win, templ, cv2.TM_CCOEFF_NORMED)
                _, mv, _, ml = cv2.minMaxLoc(res)
                if mv < 0.45:
                    continue
                # accept/reject only -- the projected rect stays where geometry
                # put it (NCC on sky/pole texture is too weak to re-centre)
                c["ncc"] = round(float(mv), 2)
                kept.append(c)
            cands = kept
        except Exception as e:
            print("view verify skipped:", e, flush=True)
    chosen = []

    def angdiff(a, b):
        d = abs(a - b) % (2 * np.pi)
        return min(d, 2 * np.pi - d)

    while cands and len(chosen) < n:
        def score(c):
            ref_b = have + [k["bearing"] for k in chosen]
            div = min([angdiff(c["bearing"], b) for b in ref_b]) if ref_b else np.pi
            return (min(np.degrees(div) / 30.0, 1.5)
                    + 0.25 * np.log2(max(c["span"] / min_span, 1.0)) + 0.5 * c["vis"])
        best = max(cands, key=score)
        chosen.append(best)
        cands = [c for c in cands if not (c["cam"] == best["cam"] and abs(c["i"] - best["i"]) <= exclude_near)]
    out = []
    for c in chosen:
        x0, y0, x1, y1 = c["rect"]
        m = 0.15 * max(x1 - x0, y1 - y0)                     # the ROI points cover one side only
        out.append({"frame": c["name"], "cam": c["cam"], "kind": "rect",
                    "pts": [[int(max(0, x0 - m)), int(max(0, y0 - m))],
                            [int(min(Hs - 1, x1 + m)), int(min(Ws - 1, y1 + m))]],
                    "auto": True, "span": round(c["span"], 1), "vis": round(c["vis"], 2),
                    "ncc": c.get("ncc")})
    return out


def fit_points(P):
    """Robust trim + axis box + oriented pose from world points."""
    med = np.median(P, axis=0)
    d = np.linalg.norm(P - med, axis=1)
    P = P[d < np.percentile(d, 92) + 1e-6]
    lo = P.min(0); hi = P.max(0); ctr = (lo + hi) / 2; size = np.maximum(hi - lo, 1e-3)
    pose = fit_oriented_box(P)
    return {"cx": float(ctr[0]), "cy": float(ctr[1]), "cz": float(ctr[2]),
            "sx": float(size[0]), "sy": float(size[1]), "sz": float(size[2]),
            "npts": int(len(P)), "pose": pose}


def solve_multi(observations):
    """Union of per-view ROI point sets -> one fit. The object is static and
    the world frame is shared, so every extra view genuinely adds coverage
    (the unseen end of the object) while per-view depth bands keep the
    background out."""
    parts = []
    for ob in observations:
        if ob.get("auto"):                          # auto views are for crops, not for the fit
            continue
        P = roi_points(ob["kind"], ob["pts"], ob["frame"])
        if P is not None:
            parts.append(P)
    if not parts:
        return {"error": "no points under any ROI"}
    return fit_points(np.concatenate(parts, 0))


def solve_roi(kind, pts, frame_name):
    P = roi_points(kind, pts, frame_name)
    if P is None:
        return {"error": "few points under ROI", "npts": 0}
    return fit_points(P)


def fit_oriented_box(P):
    """Class-agnostic oriented box, the lidar way: footprint on the local
    ground plane -> min-area rectangle -> yaw; height along the plane normal;
    bottom locked to the ground. No semantics, works for any object."""
    import cv2
    up = np.array(STATE["poses"].get("up") or STATE["meta"]["up"], dtype=np.float64)
    up /= np.linalg.norm(up)
    ref = np.array([0, 1, 0]) if abs(up[1]) < 0.9 else np.array([1, 0, 0])
    wx = np.cross(ref, up); wx /= np.linalg.norm(wx); wz = np.cross(up, wx)

    # local ground: lowest band of the object's own neighbourhood
    h = P @ up
    g0 = np.percentile(h, 2)
    # footprint of everything above the ground band
    body = P[h > g0 + 0.02]
    if len(body) < 12:
        body = P
    f2 = np.stack([body @ wx, body @ wz], -1).astype(np.float32)
    (rc_x, rc_y), (rw, rh), ang = cv2.minAreaRect(f2)
    yaw = np.deg2rad(ang)
    if rw < rh:                                   # long side defines forward
        rw, rh = rh, rw
        yaw += np.pi / 2
    fw = np.cos(yaw) * wx + np.sin(yaw) * wz      # forward on the plane
    lat = np.cross(fw, up)                        # right-handed [fw, up, lat]
    # height from footprint members only -- background behind the object
    # (hedges, walls) sneaks through the ROI and would inflate it
    ctr2 = rc_x * wx + rc_y * wz
    lf = (P - ctr2) @ fw; ll = (P - ctr2) @ lat
    inb = (np.abs(lf) < rw / 2 * 1.02) & (np.abs(ll) < rh / 2 * 1.02)
    hin = h[inb] if inb.sum() >= 12 else h
    hgt = float(np.percentile(hin, 97) - g0)
    c = rc_x * wx + rc_y * wz + (g0 + hgt / 2) * up
    R = np.stack([fw, up, lat], 1)                # columns: fw, up, lat
    q = SRq_from_matrix(R)
    return {"t": [float(v) for v in c], "q": q,
            "size": [float(rw), float(hgt), float(rh)],
            "up": up.tolist(), "g0": float(g0)}


def SRq_from_matrix(R):
    from scipy.spatial.transform import Rotation as _SR
    x, y, z, w = _SR.from_matrix(R).as_quat()
    return [float(w), float(x), float(y), float(z)]


# ------------------------------------------------------------------ map proj
def map_basis():
    C = np.array([f["p"] for f in STATE["poses"]["frames"]])
    mu = C.mean(0)
    _, _, Vt = np.linalg.svd(C - mu)
    return mu, Vt[0], Vt[1]


def build_map():
    mu, a1, a2 = map_basis()
    C = np.array([f["p"] for f in STATE["poses"]["frames"]])
    tx = (C - mu) @ a1; ty = (C - mu) @ a2
    traj = np.stack([tx, ty], -1).tolist()
    objs = []
    c = db()
    for o in c.execute("SELECT * FROM objects").fetchall():
        p = np.array([o["cx"], o["cy"], o["cz"]]) - mu
        objs.append({"id": o["id"], "label": o["label"],
                     "x": float(p @ a1), "y": float(p @ a2),
                     "r": float(0.5 * np.hypot(o["sx"], np.hypot(o["sy"], o["sz"])))})
    c.close()
    allx = tx.tolist() + [o["x"] for o in objs]
    ally = ty.tolist() + [o["y"] for o in objs]
    return {"traj": traj, "objects": objs,
            "bounds": [min(allx), min(ally), max(allx), max(ally)]}


# ------------------------------------------------------------------ jobs
def enqueue_map(cams):
    c = db()
    jid = c.execute("INSERT INTO jobs(object_id,kind,status,detail,created) "
                    "VALUES(?,?,?,?,?)",
                    (0, "map", "queued", f"map {cams}", time.time())).lastrowid
    c.commit(); c.close()
    threading.Thread(target=run_map, args=(jid, cams), daemon=True).start()
    return jid


def run_map(jid, cams):
    """Full map build: Omega chunks per camera on tex1 -> anchors into the
    COLMAP world -> local merge into scenes/street (the base map) + nxyz."""
    def upd(st, detail):
        c = db(); c.execute("UPDATE jobs SET status=?,detail=? WHERE id=?",
                            (st, detail, jid)); c.commit(); c.close()
    try:
        cam_list = ["A", "B"] if cams == "AB" else [cams]
        CK = "/nvme0n1-disk/valentyn.fedorov/vggt-omega/checkpoints/vggt_omega_1b_512.pt"
        env = ("source /nvme0n1-disk/valentyn.fedorov/miniconda3/etc/profile.d/conda.sh && "
               "conda activate vggto && cd /nvme0n1-disk/valentyn.fedorov/dualcam && ")
        total = len(cam_list) + 3
        with GPU_LOCK:
            for i, cam in enumerate(cam_list):
                ply = VIEWER / "scenes" / f"map_{cam}.ply"
                if LOCAL_ONLY:
                    upd("running", f"{i + 1}/{total} Ω chunks, camera {cam} "
                                   f"(local GPU, chunk auto-fits 12-16 GB, ~20-30 min)")
                    env2 = dict(os.environ); env2["PACK_DIR"] = str(CLOUD_DIR)
                    ck = LOCAL_GPU / "models" / "vggt_omega_1b_512.pt"
                    if not ck.exists():
                        raise RuntimeError("VGGT weights missing: local_gpu/models/vggt_omega_1b_512.pt "
                                           "— extract nast_v2_gpu_extras.tar into the repo root")
                    if not (VIDEO_DIR / "depth").exists():
                        raise RuntimeError(f"{VIDEO_DIR.name}/depth missing (MoGe depth ships in "
                                           "nast_v2_gpu_extras.tar)")
                    chk = subprocess.run([sys.executable, "-c", "import torch,sys; "
                                          "sys.exit(0 if torch.cuda.is_available() else 3)"],
                                         capture_output=True, text=True)
                    if chk.returncode != 0:
                        raise RuntimeError("torch in venv has no CUDA — run install.sh on the GPU box "
                                           "(needs nvidia-smi) or: venv/bin/pip install torch "
                                           "--index-url https://download.pytorch.org/whl/cu128")
                    sh([sys.executable, str(LOCAL_GPU / "vggto_local.py"),
                        str(LOCAL_GPU / "models" / "vggt_omega_1b_512.pt"),
                        str(VIDEO_DIR / "rgb"), str(VIDEO_DIR / "depth"), f"{cam}_",
                        str(ply), "24", "2", "512", "15", "1", "0.00015", "8000000"],
                       timeout=7200, env=env2)
                else:
                    upd("running", f"{i + 1}/{total} Ω chunks, camera {cam} (HQ: conf 15%, "
                                   f"every chunk anchored to the world, ~15 min)")
                    sh(["ssh", TEX, f"bash -lc '{env}python vggto_chunks.py {CK} sfm/images "
                        f"depth_dump {cam}_ map_{cam}.ply 64 2 640 15 1 0.00015 8000000'"],
                       timeout=3000)
        upd("running", f"{len(cam_list) + 1}/{total} collecting clouds")
        pairs = []
        for cam in cam_list:
            ply = VIEWER / "scenes" / f"map_{cam}.ply"
            if not LOCAL_ONLY:
                sh(["scp", "-q", f"{TEX}:{TEXD}/map_{cam}.ply", str(ply)], timeout=1800)
            pairs += [str(ply), "none"]
        # fine-detail layer: per-pixel MoGe depth of every 2nd frame -- thin
        # poles, edges and far objects that Omega's confidence cut drops
        upd("running", f"{len(cam_list) + 2}/{total} MoGe layer (small objects, edges)")
        moge_ply = VIEWER / "scenes" / "map_moge.ply"
        try:
            sh([sys.executable, str(MONO / "moge_layer.py"), str(HERE / "scene_base"),
                str(VIDEO_DIR / "rgb"), str(VIDEO_DIR / "depth"), str(moge_ply),
                "2", "3", "0.012", "45"], timeout=3600)
            pairs += [str(moge_ply), "none"]
        except Exception as e:
            print("moge layer skipped:", e, flush=True)
        upd("running", f"{total}/{total} merge + pack + nxyz")
        street = VIEWER / "scenes" / "street"
        sh([sys.executable, str(MONO / "merge_omega_world.py"),
            str(HERE / "scene_base"), str(street), *pairs], timeout=2400)
        (street / "index.html").write_bytes((VIEWER / "point_viewer.html").read_bytes())
        try:
            sh([sys.executable, str(MONO / "pack_add_normals.py"), str(street), "32"],
               timeout=3600)
        except Exception:
            pass
        with LOCK:
            load_scene()                          # база оновилась — перечитати
        upd("done", "/scene/street/index.html")
    except Exception as e:
        upd("error", repr(e)[:300])


def enqueue_reconstruct(object_id):
    c = db()
    o = c.execute("SELECT * FROM objects WHERE id=?", (object_id,)).fetchone()
    if o is None:
        c.close()
        raise ValueError(f"no such object {object_id}")
    jid = c.execute("INSERT INTO jobs(object_id,kind,status,detail,created) "
                    "VALUES(?,?,?,?,?)",
                    (object_id, "reconstruct", "queued",
                     "observer selection", time.time())).lastrowid
    c.commit(); c.close()
    threading.Thread(target=run_reconstruct, args=(jid, dict(o)), daemon=True).start()
    return jid


MONO = HERE.parent / "monocars"
TEX = "tex1"
TEXD = "/nvme0n1-disk/valentyn.fedorov/dualcam"
GPU_LOCK = threading.Lock()                     # one remote GPU job at a time


def sh(args, timeout=300, env=None):
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env)
    if r.returncode != 0:
        what = Path(args[1]).name if len(args) > 1 and str(args[1]).endswith(".py") else args[0]
        raise RuntimeError(f"{what} rc={r.returncode}: {(r.stderr or r.stdout)[-600:]}")
    return r.stdout


def roi_crop(ob, out_path, margin=0.10, min_side=384, force_lean=None):
    """Crop the operator's ROI from its frame (upright), with a small margin,
    exposure lift, upscale to a workable size, and -- for silhouettes against
    bright sky -- a soft sky knock-out so rembg keeps the object, not the
    cloud behind it. Returns the path or None."""
    import cv2
    src = crop_frame_src(ob["frame"])
    img = cv2.imread(str(src))
    if img is None:
        return None
    up = np.rot90(img, k=3).copy()                     # sensor -> upright
    poly = np.array(roi_to_polygon(ob["kind"], ob["pts"]), dtype=np.float64)
    x0, y0 = poly[:, 0].min(), poly[:, 1].min()
    x1, y1 = poly[:, 0].max(), poly[:, 1].max()
    mw, mh = max((x1 - x0) * margin, 8), max((y1 - y0) * margin, 8)
    H, W = up.shape[:2]
    x0, x1 = int(max(0, x0 - mw)), int(min(W, x1 + mw))
    y0, y1 = int(max(0, y0 - mh)), int(min(H, y1 + mh))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    # gravity-level the crop: a rolled/tilted rig makes a vertical pole lean
    # in the frame, and TRELLIS faithfully generates a leaning pole. Measure
    # the dominant axis of the dark (object) pixels inside the ROI and rotate
    # the frame so it stands vertical -- works for any tall object.
    lean_used = 0.0
    if force_lean is not None:
        lean_used = float(force_lean)
    else:
        win = cv2.cvtColor(up[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        thr = min(80, int(np.percentile(win, 12)) + 15)
        ys, xs = np.where(win < thr)
        if len(xs) > 200:
            pts = np.stack([xs, ys], 1).astype(np.float64); pts -= pts.mean(0)
            _, sv, Vt = np.linalg.svd(pts, full_matrices=False)
            d = Vt[0]
            elongated = sv[0] > 2.5 * (sv[1] + 1e-6)          # tall thin thing
            lean = np.degrees(np.arctan2(d[0], d[1]))         # from vertical, deg
            if lean > 90: lean -= 180
            if lean < -90: lean += 180
            if elongated and 1.5 < abs(lean) < 45:
                lean_used = lean
    if abs(lean_used) > 1.5:
        cx_, cy_ = (x0 + x1) / 2, (y0 + y1) / 2
        M = cv2.getRotationMatrix2D((cx_, cy_), -lean_used, 1.0)
        up = cv2.warpAffine(up, M, (W, H), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    crop = up[y0:y1, x0:x1]
    native = max(crop.shape[0], crop.shape[1])
    # exposure: bring dusk footage toward what TRELLIS was trained on
    luma = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())
    if luma < 110:
        gain = min(110 / max(luma, 1.0), 3.0)
        crop = np.clip(crop.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    # RAW crop out (leveled, exposure-lifted, upscaled). Segmentation and
    # super-resolution happen on tex1 (SAM + Real-ESRGAN in crop_enhance.py);
    # a sidecar box tells SAM where the operator's ROI sits inside the crop.
    short = min(crop.shape[0], crop.shape[1])
    kk = 1
    if short < min_side:
        kk = int(np.ceil(min_side / max(short, 1)))
        crop = cv2.resize(crop, (crop.shape[1] * kk, crop.shape[0] * kk),
                          interpolation=cv2.INTER_LANCZOS4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)
    # ROI polygon in crop coords (after margin & upscale)
    pc = poly.copy(); pc[:, 0] -= x0; pc[:, 1] -= y0; pc *= kk
    side = {"x0": float(max(0, pc[:, 0].min())), "y0": float(max(0, pc[:, 1].min())),
            "x1": float(min(crop.shape[1], pc[:, 0].max())),
            "y1": float(min(crop.shape[0], pc[:, 1].max())), "scale": float(kk)}
    # SAM prompt hints (upright frame px) -> crop px: same rotation, offset, upscale
    def to_crop(pts):
        out = []
        for x, y in pts or []:
            if abs(lean_used) > 1.5:
                x, y = M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2]
            cx2, cy2 = (x - x0) * kk, (y - y0) * kk
            if 0 <= cx2 < crop.shape[1] and 0 <= cy2 < crop.shape[0]:
                out.append([float(cx2), float(cy2)])
        return out
    if ob.get("pos"):
        side["pos"] = to_crop(ob["pos"]); side["neg"] = to_crop(ob.get("neg"))
    out_path.with_suffix(".box.json").write_text(json.dumps(side))
    return out_path, native, lean_used, False


def run_reconstruct(jid, obj):
    """Frame crops locally -> TRELLIS + placement on tex1 -> splat pack back,
    served at /job/<jid>/index.html. Every stage stamps the job row."""
    def upd(st, detail):
        c = db(); c.execute("UPDATE jobs SET status=?,detail=? WHERE id=?",
                            (st, detail, jid)); c.commit(); c.close()
    try:
        wd = HERE / "jobs" / f"job_{jid}"
        (wd / "crops").mkdir(parents=True, exist_ok=True)
        upd("running", "1/6 observers + box")
        ctr = np.array([obj["cx"], obj["cy"], obj["cz"]])
        obs = []
        for f in STATE["poses"]["frames"]:
            R = quat_to_R(f["q"]); C = np.array(f["p"])
            d = ctr - C; dist = np.linalg.norm(d)
            if dist < 1e-6:
                continue
            fwd = R[2]
            if (d @ fwd) / dist > 0.85 and dist < STATE["meta"]["extent"] * 1.5:
                obs.append((f["name"], float(dist)))
        obs.sort(key=lambda t: t[1])
        if not obs:
            raise RuntimeError("no camera sees this box")
        box = {"center": ctr.tolist(), "size": [obj["sx"], obj["sy"], obj["sz"]],
               "observers": [{"name": n, "dist": d} for n, d in obs]}
        # camera positions of the operator's own ROI frames: place_points uses
        # the first one to turn the asset the way the crop saw it (yaw)
        obs_list = json.loads(obj.get("obs") or "[]") if isinstance(obj.get("obs"), str) else (obj.get("obs") or [])
        by_name = {f["name"]: f for f in STATE["poses"]["frames"]}
        by_stem = {Path(f["name"]).stem: f for f in STATE["poses"]["frames"]}
        rf = []
        for ob in obs_list[:6]:
            fr = by_name.get(ob.get("frame")) or by_stem.get(Path(ob.get("frame") or "").stem)
            rf.append({"name": ob.get("frame"), "pos": (fr["p"] if fr else None)})
        box["roi_frames"] = rf
        box["obs_frames"] = [ob.get("frame") for ob in obs_list if ob.get("frame")]   # every view (manual + auto)
        ps = obj.get("pose")
        if ps:
            box["pose"] = json.loads(ps) if isinstance(ps, str) else ps
        if not box.get("pose"):
            raise RuntimeError("object has no 3D pose — draw the ROI again")
        (wd / "box.json").write_text(json.dumps(box, indent=1))

        # ---- crops: the operator's own ROIs first (best framing there is),
        # then box-projected views from other observers for extra angles
        upd("running", "2/6 cropping frames")
        max_side = 0
        n_roi = 0
        thin_any = False
        # operator views first, auto-collected views after; SAM prompt hints
        # (object points + depth-disagreeing background) ride along per view
        obs_list = [ob for ob in obs_list if not ob.get("auto")] + [ob for ob in obs_list if ob.get("auto")]
        obs_list = obs_list[:8]
        P_obj = object_world_points(obs_list)
        for ob in obs_list:
            try:
                poly0 = np.array(roi_to_polygon(ob["kind"], ob["pts"]), dtype=np.float64)
                rect = [float(poly0[:, 0].min()), float(poly0[:, 1].min()),
                        float(poly0[:, 0].max()), float(poly0[:, 1].max())]
                ob["pos"], ob["neg"] = sam_hints(ob["frame"], P_obj, rect)
            except Exception as e:
                print("hints skipped:", e, flush=True)
        # pass 1: measure lean per ROI; the rig tilt is a property of the
        # frame, so the strongest confident measurement is applied to ALL
        leans = []
        for k, ob in enumerate(obs_list):
            got = roi_crop(ob, wd / "crops" / f"roi_{k:02d}.png")
            if got is not None:
                max_side = max(max_side, got[1]); n_roi += 1
                thin_any = thin_any or got[3]
                ob["_lean"] = got[2]
                if abs(got[2]) > 1.5:
                    leans.append(got[2])
        if leans:
            # views come from different frames/positions: each keeps its own
            # confident lean, the median only fills in where nothing was measured
            lean_all = float(np.median(leans))
            for k, ob in enumerate(obs_list):
                own = ob.get("_lean")
                roi_crop(ob, wd / "crops" / f"roi_{k:02d}.png",
                         force_lean=(own if own is not None and abs(own) > 1.5 else lean_all))
        # box-projected views only when the operator gave a single ROI: with
        # neighbours in the frame (two lamps) a loose projection crop can
        # frame the WRONG object and the model averages them
        if n_roi < 2:
            try:
                crop_dir = (VIDEO_DIR / "rgb_orig") if (
                    not DEGLARE_CROPS and (VIDEO_DIR / "rgb_orig").exists()) else (VIDEO_DIR / "rgb")
                sh([sys.executable, str(MONO / "project_box_crop.py"), str(CLOUD_DIR),
                    str(crop_dir), str(wd / "box.json"), str(wd / "crops"),
                    "4", "0.12"])
            except Exception as e:
                print("box crops skipped:", e, flush=True)
        crops = sorted((wd / "crops").glob("*.png"))

        # route: photo when the operator's ROI is at least ~60 px on its long
        # side (silhouette + edges are enough for TRELLIS), text otherwise
        label = (obj.get("label") or "").strip()
        # thin silhouettes (poles, lamps, signs) are where image-TRELLIS
        # hallucinates or collapses -- five rounds proved it; route by class
        tiny = False                       # photo path only (user decision)
        prompt = None
        LUT = {"камера": "security CCTV camera mounted on a bracket",
               "camera": "security CCTV camera mounted on a bracket",
               "смітник": "street dumpster garbage container",
               "знак": "road sign on a pole",
               "стовп": "utility pole",
               "pole": "utility pole",
               "ліхтар": "tall straight street light pole, single short horizontal arm at the top, flat rectangular LED luminaire head, dark grey metal",
               "лампа": "tall straight street light pole, single short horizontal arm at the top, flat rectangular LED luminaire head, dark grey metal",
               "lamp": "tall straight street light pole, single short horizontal arm at the top, flat rectangular LED luminaire head, dark grey metal",
               "гідрант": "red fire hydrant",
               "конус": "orange traffic cone",
               "cone": "orange traffic cone",
               "лавка": "park bench",
               "прапор": "flag on a flagpole",
               "дерево": "small street tree",
               "машина": "sedan car",
               "авто": "sedan car",
               "вен": "cargo van",
               "van": "cargo van"}
        generic = (not label) or label.lower() in ("?",) or \
                  (label[0] in "Rr" and label[1:].isdigit())
        if tiny and not generic:
            prompt = next((v for k, v in LUT.items() if k in label.lower()), label)
        elif tiny and generic:
            raise RuntimeError("object too small for photo generation — give it a class name "
                               "(lamp, camera, sign…) and run again")

        if not prompt and not crops:
            raise RuntimeError("no frame produced a crop")

        if LOCAL_ONLY:
            # TRELLIS on this machine's GPU when local_gpu/trellis/install_trellis.sh
            # has finished (its env_ok marker); otherwise the points-only asset
            worker = LOCAL_GPU / "trellis" / "trellis_local.sh"
            root_file = LOCAL_GPU / "trellis" / "ROOT"          # written by install_trellis.sh
            troot = Path(os.environ.get("NAST_TRELLIS_ROOT") or
                         (root_file.read_text().strip() if root_file.exists() else "") or
                         str(Path.home() / "nast_trellis"))
            if TRELLIS_LOCAL and crops and worker.exists() and (troot / "env_ok").exists():
                with GPU_LOCK:
                    upd("running", f"4/6 SAM + Real-ESRGAN + TRELLIS on the local GPU "
                                   f"({len(crops)} views, ~3-5 min)")
                    env2 = dict(os.environ); env2["NAST_TRELLIS_ROOT"] = str(troot)
                    sh(["bash", str(worker), str(wd)], timeout=3600, env=env2)
                if not (wd / "asset.ply").exists():
                    raise RuntimeError("local TRELLIS produced no asset.ply — see the job log")
            else:
                upd("running", "3/6 local asset: map points inside the box "
                               "(TRELLIS not installed: run local_gpu/trellis/install_trellis.sh)")
                build_point_asset(box, wd / "asset.ply")
        else:
            with GPU_LOCK:
                if prompt:
                    upd("running", f"4/6 TRELLIS text: \"{prompt}\" (~3 min)")
                    sh(["ssh", TEX, f"mkdir -p {TEXD}/insp/job_{jid} && cd {TEXD} && "
                        f"bash -lc 'source /nvme0n1-disk/valentyn.fedorov/miniconda3/etc/profile.d/conda.sh && "
                        f"conda activate trellis && python trellis_text_gen.py insp/job_{jid}/asset {prompt}'"],
                       timeout=2400)
                else:
                    upd("running", f"3/6 uploading {len(crops)} crops to tex1")
                    sh(["ssh", TEX, f"mkdir -p {TEXD}/insp/job_{jid}/crops"])
                    sides = [str(p.with_suffix(".box.json")) for p in crops if p.with_suffix(".box.json").exists()]
                    sh(["scp", "-q", *[str(p) for p in crops], *sides,
                        f"{TEX}:{TEXD}/insp/job_{jid}/crops/"])
                    upd("running", "4/6 SAM + Real-ESRGAN + TRELLIS on tex1 (~4 min)")
                    sh(["ssh", TEX, f"bash {TEXD}/inspector_trellis.sh {jid}"], timeout=1200)
                upd("running", "5/6 downloading asset")
                sh(["scp", "-q", f"{TEX}:{TEXD}/insp/job_{jid}/asset.ply", str(wd)])
                try:                                  # turntable preview for the renders tab
                    sh(["scp", "-q", f"{TEX}:{TEXD}/insp/job_{jid}/asset_turn.mp4", str(wd)])
                except Exception:
                    pass
                try:                                  # SAM silhouettes: they decide the asset's up axis
                    sh(["scp", "-q", "-r", f"{TEX}:{TEXD}/insp/job_{jid}/crops_enh", str(wd)], timeout=600)
                except Exception:
                    pass
                try:                                  # the mesh decoded from the same latent (+ uv + baked texture)
                    sh(["scp", "-q", f"{TEX}:{TEXD}/insp/job_{jid}/asset_mesh.ply",
                        f"{TEX}:{TEXD}/insp/job_{jid}/asset_mesh_uv.npy",
                        f"{TEX}:{TEXD}/insp/job_{jid}/asset_mesh_tex.png",
                        f"{TEX}:{TEXD}/insp/job_{jid}/asset_mesh.glb", str(wd)], timeout=600)
                except Exception as e:
                    print("mesh download skipped:", e, flush=True)

        if not (CLOUD_DIR / "pos.f32").exists():
            raise RuntimeError("no base map yet — build the point cloud first (Ω/MVS)")
        upd("running", "6/6 placing points + packing")
        scene = VIEWER / "scenes" / f"job_{jid}"
        # asset -> centered points + up/front from the crop silhouettes (asset_pts.ply.up.json)
        objply = wd / "asset_pts.ply"
        objscene = VIEWER / "scenes" / f"obj_job_{jid}"
        sh([sys.executable, str(MONO / "asset2ply.py"), str(wd / "asset.ply"), str(objply),
            "0.12", f"--masks={wd}"], timeout=600)
        sh([sys.executable, str(MONO / "place_points.py"), str(CLOUD_DIR),
            str(wd / "asset.ply"), str(wd / "box.json"), str(scene), "0", "1",
            f"--up={objply}.up.json"], timeout=900)
        (scene / "index.html").write_bytes((VIEWER / "point_viewer.html").read_bytes())
        # close-up scene for the objects tab: the asset alone, centered, upright
        sh([sys.executable, str(VIEWER / "pack_bare.py"), str(objply), str(objscene),
            f"--up={objply}.up.json"], timeout=600)
        (objscene / "index.html").write_bytes((VIEWER / "point_viewer.html").read_bytes())
        # gaussian-splat close-up (the photoreal one): asset kept as gaussians, upright + front
        try:                                      # live world: decode the view frames first
            ensure_layer_frames(box.get("obs_frames", []) +
                                [rf.get("name") for rf in box.get("roi_frames", []) if rf.get("name")])
        except Exception as e:
            print("ensure layers skipped:", e, flush=True)
        try:
            sh([sys.executable, str(MONO / "asset2splat.py"), str(wd / "asset.ply"),
                f"{objply}.up.json", str(objscene), "0.03", f"--crops={wd}"], timeout=600)
            (objscene / "splat.html").write_bytes((VIEWER / "splat_viewer.html").read_bytes())
            # polarization layers painted onto the gaussians (needs street_video/layers)
            if (VIDEO_DIR / "layers").exists():
                sh([sys.executable, str(MONO / "asset_layers.py"), str(wd), str(objscene),
                    str(VIDEO_DIR), str(CLOUD_DIR)], timeout=900)
        except Exception as e:
            print("splat pack skipped:", e, flush=True)
        # ---- mesh for EVERY object (2026-08-27): TRELLIS mesh when it exists,
        # otherwise meshed locally from the close-up points; then the layer
        # sets, the structural modes (Sketch/Skeleton/Exploded + Segments)
        # and the trimmed mesh viewer page
        try:
            upd("running", "6/6 meshing + structural modes")
            if not (wd / "asset_mesh.ply").exists():
                sh([sys.executable, str(MONO / "mesh_from_points.py"), str(objscene)], timeout=1200)
            sh([sys.executable, str(MONO / "mesh_layers.py"), str(wd), str(objscene),
                str(VIDEO_DIR), str(CLOUD_DIR)], timeout=1800)
            sh([sys.executable, str(MONO / "object_struct.py"), str(objscene)], timeout=1200)
            (objscene / "mesh.html").write_bytes((VIEWER / "mesh_viewer.html").read_bytes())
        except Exception as e:
            print("mesh flow skipped:", e, flush=True)
        try:
            # real views indexed by direction (split view: model + the photo from that side)
            sh([sys.executable, str(MONO / "object_views.py"), str(wd), str(objscene),
                str(VIDEO_DIR), str(CLOUD_DIR), str(DB)], timeout=600)
        except Exception as e:
            print("views skipped:", e, flush=True)
        try:                                      # nxyz color set for both scenes
            sh([sys.executable, str(MONO / "pack_add_normals.py"), str(objscene), "48"], timeout=1200)
            sh([sys.executable, str(MONO / "pack_add_normals.py"), str(scene), "32"], timeout=1800)
        except Exception:
            pass
        upd("done", f"/job/{jid}/index.html")
    except Exception as e:
        upd("error", repr(e)[:300])


# ------------------------------------------------------------------ http
CT = {".html": "text/html", ".js": "application/javascript", ".css": "text/css",
      ".json": "application/json", ".jpg": "image/jpeg", ".png": "image/png",
      ".f32": "application/octet-stream", ".u8": "application/octet-stream"}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path):
        if not path.exists() or not path.is_file():
            return self._send(404, {"error": "not found"})
        data = path.read_bytes()
        self._send(200, data, CT.get(path.suffix, "application/octet-stream"))

    def do_GET(self):
        u = urlparse(self.path); p = unquote(u.path)
        if p == "/" or p == "/index.html":
            return self._file(HERE / "static" / "app.html")
        if p.startswith("/static/"):
            return self._file(HERE / "static" / p[len("/static/"):])
        if p.startswith("/frames/live/"):
            # on-demand normals: layers/<variant>/<name>.jpg, decoded from raw/
            rel = unquote(p[len("/frames/live/"):])
            parts = rel.split("/", 1)
            if len(parts) == 2:
                variant, fname = parts
                tgt = (VIDEO_DIR / variant / fname) if variant in ("rgb", "rgb_orig") \
                      else (VIDEO_DIR / "layers" / variant / fname)
                if not tgt.exists():
                    try:
                        live_decode(Path(fname).stem.replace(".jpg", ""))
                    except Exception as e:
                        return self._send(500, {"error": f"live decode: {e}"})
                if tgt.exists():
                    return self._file(tgt)
            return self._send(404, {"error": "no frame"})
        if p.startswith("/frames/"):
            return self._file(VIDEO_DIR / p[len("/frames/"):])
        if p.startswith("/cloud/"):
            return self._file(CLOUD_DIR / p[len("/cloud/"):])
        if p.startswith("/job/"):
            rel = unquote(p[len("/job/"):])
            tgt = (VIEWER / "scenes" / ("job_" + rel.split("/")[0]) /
                   "/".join(rel.split("/")[1:] or ["index.html"])).resolve()
            if not str(tgt).startswith(str((VIEWER / "scenes").resolve())):
                return self._send(403, {"error": "path"})
            return self._file(tgt)
        if p.startswith("/scene/"):
            rel = unquote(p[len("/scene/"):])
            parts = rel.split("/")
            tgt = (VIEWER / "scenes" / parts[0] /
                   "/".join(parts[1:] or ["index.html"])).resolve()
            if not str(tgt).startswith(str((VIEWER / "scenes").resolve())):
                return self._send(403, {"error": "path"})
            return self._file(tgt)
        if p == "/api/objects3d":
            c = db(); rows = c.execute("SELECT * FROM objects").fetchall(); c.close()
            out = []
            for o in rows:
                out.append({"id": o["id"], "label": o["label"],
                            "c": [o["cx"], o["cy"], o["cz"]],
                            "r": float(0.6 * max(o["sx"], o["sy"], o["sz"]))})
            return self._send(200, out)
        if p == "/api/tracks":
            tr = {"A": [], "B": []}
            for f in STATE["poses"]["frames"]:
                cam = "A" if f["name"].startswith("A_") else "B"
                tr[cam].append([round(v, 4) for v in f["p"]])
            tr["up"] = STATE["poses"].get("up") or STATE["meta"].get("up")
            return self._send(200, tr)
        if p == "/api/scenes":
            out = []
            for d in sorted((VIEWER / "scenes").iterdir()):
                if d.is_dir() and (d / "pos.f32").exists() and (d / "index.html").exists():
                    out.append({"name": d.name, "url": f"/scene/{d.name}/index.html"})
            return self._send(200, out)
        if p == "/api/meta":
            m = STATE["meta"]; ps = STATE["poses"]["frames"]
            return self._send(200, {"intrinsics": m["intrinsics"], "extent": m["extent"],
                                    "count": m["count"], "nframes": len(ps),
                                    "up": STATE["poses"].get("up") or m.get("up")})
        if p == "/api/frames":
            ps = STATE["poses"]["frames"]
            A = [f["name"] for f in ps if f["name"].startswith("A_")]
            B = [f["name"] for f in ps if f["name"].startswith("B_")]
            return self._send(200, {"A": A, "B": B})
        if p == "/api/poses":
            return self._send(200, {f["name"]: {"q": f["q"], "p": f["p"]}
                                    for f in STATE["poses"]["frames"]})
        if p == "/api/objects":
            c = db(); rows = [dict(r) for r in c.execute("SELECT * FROM objects").fetchall()]; c.close()
            return self._send(200, rows)
        if p == "/api/map":
            return self._send(200, build_map())
        if p == "/api/jobs":
            c = db(); rows = [dict(r) for r in c.execute("SELECT * FROM jobs ORDER BY id DESC").fetchall()]; c.close()
            return self._send(200, rows)
        if p == "/api/dataset":
            total, done = decode_counts()
            return self._send(200, {"path": str(VIDEO_DIR), "raw": total, "decoded": done,
                                    "frames": len(list((VIDEO_DIR / "rgb").glob("*.jpg")))
                                    if (VIDEO_DIR / "rgb").exists() else 0})
        if p == "/api/decode_status":
            total, done = decode_counts()
            DECODE["done"] = done; DECODE["total"] = total
            eta = 0
            if DECODE["running"] and done and DECODE["started"]:
                rate = (time.time() - DECODE["started"]) / max(done, 1)
                eta = int(rate * (total - done))
            return self._send(200, {"running": DECODE["running"], "total": total,
                                    "done": done, "err": DECODE["err"],
                                    "complete": total > 0 and done >= total, "eta": eta})
        return self._send(404, {"error": "no route"})

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) or b"{}"
        try:
            return json.loads(raw)
        except UnicodeDecodeError:                # curl з cp1251-консолі
            return json.loads(raw.decode("cp1251"))

    def do_POST(self):
        u = urlparse(self.path); p = u.path
        if p == "/api/open_dataset":
            try:
                return self._send(200, open_dataset(self._body().get("path", "")))
            except Exception as e:
                return self._send(422, {"error": str(e)})
        if p == "/api/decode":
            start_decode_if_needed()
            total, done = decode_counts()
            return self._send(200, {"running": DECODE["running"], "total": total, "done": done})
        if p == "/api/roi":
            b = self._body()
            with LOCK:
                box = solve_roi(b["kind"], b["pts"], b["frame"])
            if "error" in box:
                return self._send(422, box)
            c = db()
            first_obs = [{"frame": b["frame"], "kind": b["kind"], "pts": b["pts"]}]
            oid = c.execute(
                "INSERT INTO objects(label,frame,cam,kind,pts,cx,cy,cz,sx,sy,sz,npts,created,pose,obs)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (b.get("label", "?"), b["frame"], b.get("cam", "A"), b["kind"],
                 json.dumps(b["pts"]), box["cx"], box["cy"], box["cz"],
                 box["sx"], box["sy"], box["sz"], box["npts"], time.time(),
                 json.dumps(box.get("pose")), json.dumps(first_obs))).lastrowid
            c.commit()
            row = dict(c.execute("SELECT * FROM objects WHERE id=?", (oid,)).fetchone()); c.close()
            return self._send(200, row)
        if p == "/api/dbg_png":
            b = self._body()
            import base64 as _b64
            data = b.get("dataURL", "")
            i = data.find(",")
            name = "".join(c for c in b.get("name", "dbg") if c.isalnum() or c in "_-")
            out = VIEWER.parent / "parts48" / (name + ".png")
            out.parent.mkdir(exist_ok=True)
            out.write_bytes(_b64.b64decode(data[i + 1:]))
            return self._send(200, {"saved": str(out)})
        if p == "/api/reconstruct":
            b = self._body()
            try:
                jid = enqueue_reconstruct(int(b["object_id"]))
            except ValueError as ex:
                return self._send(404, {"error": str(ex)})
            return self._send(200, {"job_id": jid})
        if p == "/api/build_map":
            b = self._body()
            cams = b.get("cams", "AB")
            if cams not in ("A", "B", "AB"):
                return self._send(422, {"error": "cams: A|B|AB"})
            jid = enqueue_map(cams)
            return self._send(200, {"job_id": jid})
        if p.startswith("/api/objects/") and p.endswith("/pose"):
            oid = int(p.split("/")[3])
            b = self._body()                      # {t,q,size,...} - operator edit
            c = db()
            c.execute("UPDATE objects SET pose=? WHERE id=?", (json.dumps(b), oid))
            c.commit(); c.close()
            return self._send(200, {"ok": True, "id": oid})
        if p.startswith("/api/objects/") and p.endswith("/resolve"):
            oid = int(p.split("/")[3])
            c = db()
            o = c.execute("SELECT obs FROM objects WHERE id=?", (oid,)).fetchone()
            if o is None or not o["obs"]:
                c.close(); return self._send(404, {"error": "no object/obs"})
            with LOCK:
                box = solve_multi(json.loads(o["obs"]))
            if "error" in box:
                c.close(); return self._send(422, box)
            c.execute("UPDATE objects SET pose=?,cx=?,cy=?,cz=?,sx=?,sy=?,sz=?,npts=? WHERE id=?",
                      (json.dumps(box["pose"]), box["cx"], box["cy"], box["cz"],
                       box["sx"], box["sy"], box["sz"], box["npts"], oid))
            c.commit()
            row = dict(c.execute("SELECT * FROM objects WHERE id=?", (oid,)).fetchone())
            c.close()
            return self._send(200, row)
        if p.startswith("/api/objects/") and p.endswith("/augment"):
            oid = int(p.split("/")[3])
            b = self._body() or {}
            c = db()
            j = c.execute("SELECT id FROM jobs WHERE object_id=? AND status='done' AND kind!='map' ORDER BY id DESC", (oid,)).fetchone()
            c.close()
            if j is None:
                return self._send(404, {"error": "no finished reconstruction for this object"})
            jid = j["id"]
            objscene = VIEWER / "scenes" / f"obj_job_{jid}"
            out = HERE / "jobs" / f"job_{jid}" / "augment"
            try:
                sh([sys.executable, str(MONO / "augment_set.py"), str(objscene), str(out),
                    str(b.get("yaw_step", 15)), ",".join(str(v) for v in b.get("pitches", [5, 20])),
                    str(b.get("size", 512))], timeout=1800)
                man = json.loads((out / "manifest.json").read_text())
                return self._send(200, {"job_id": jid, "path": str(out), "real_views": len(man["real"]),
                                        "synthetic": len(man["synthetic"]), "sets": man["sets"]})
            except Exception as e:
                return self._send(500, {"error": repr(e)[:300]})
        if p.startswith("/api/objects/") and p.endswith("/autoviews"):
            oid = int(p.split("/")[3])
            b = self._body() or {}
            n = int(b.get("n", 5))
            c = db()
            o = c.execute("SELECT * FROM objects WHERE id=?", (oid,)).fetchone()
            if o is None:
                c.close(); return self._send(404, {"error": "no object"})
            o = dict(o)
            obs = [ob for ob in json.loads(o["obs"] or "[]") if not ob.get("auto")]   # replace old autos
            o["obs"] = json.dumps(obs)
            with LOCK:
                added = auto_views(o, n=n)
            obs = obs + added
            c.execute("UPDATE objects SET obs=? WHERE id=?", (json.dumps(obs), oid))
            c.commit()
            row = dict(c.execute("SELECT * FROM objects WHERE id=?", (oid,)).fetchone())
            c.close()
            row["added"] = added
            return self._send(200, row)
        if p.startswith("/api/objects/") and p.endswith("/roi"):
            oid = int(p.split("/")[3])
            b = self._body()                      # {frame, kind, pts} - extra view
            c = db()
            o = c.execute("SELECT obs FROM objects WHERE id=?", (oid,)).fetchone()
            if o is None:
                c.close(); return self._send(404, {"error": "no object"})
            obs = json.loads(o["obs"] or "[]")
            obs.append({"frame": b["frame"], "kind": b["kind"], "pts": b["pts"]})
            with LOCK:
                box = solve_multi(obs)
            if "error" in box:
                c.close(); return self._send(422, box)
            c.execute("UPDATE objects SET obs=?,pose=?,cx=?,cy=?,cz=?,sx=?,sy=?,sz=?,npts=? WHERE id=?",
                      (json.dumps(obs), json.dumps(box["pose"]), box["cx"], box["cy"],
                       box["cz"], box["sx"], box["sy"], box["sz"], box["npts"], oid))
            c.commit()
            row = dict(c.execute("SELECT * FROM objects WHERE id=?", (oid,)).fetchone())
            c.close()
            return self._send(200, row)
        return self._send(404, {"error": "no route"})

    def do_DELETE(self):
        p = urlparse(self.path).path
        if p.startswith("/api/objects/"):
            oid = int(p.rsplit("/", 1)[1])
            c = db(); c.execute("DELETE FROM objects WHERE id=?", (oid,)); c.commit(); c.close()
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "no route"})


if __name__ == "__main__":
    init_db()
    # jobs left "running"/"queued" by a previous server process are dead now
    _c = db()
    _c.execute("UPDATE jobs SET status='error', detail='interrupted by server restart' "
               "WHERE status IN ('running', 'queued')")
    _c.commit(); _c.close()
    load_scene()
    ThreadingHTTPServer.daemon_threads = True
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"[nast-inspector] http://localhost:{PORT}/  (Ctrl+C to stop)", flush=True)
    srv.serve_forever()
