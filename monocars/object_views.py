"""Real views of a reconstructed object, indexed by viewing direction, in every layer.

For the split close-up ("rotate the model, see the real photo from that side"):
  * every frame the object was observed in (operator ROI + auto views) gets its
    camera direction expressed in the object's close-up frame (the frame the
    splat/mesh viewers orbit in) -> yaw/pitch the viewer can match against;
  * the object's rect in that frame is cropped (upright, +12% margin) from the
    RGB frame, the recorder nxyz and every polarization layer in
    <street_video>/layers/<product>/ -> <obj_scene>/views/v<i>_<layer>.jpg
  * <obj_scene>/views.json  {views:[{i,name,cam,yaw,pitch,dir,rect,auto}], layers:[...]}

Usage: python object_views.py <job_dir> <obj_scene_dir> <street_video_dir> <map_scene_dir> <inspector.db>
"""
import json
import sqlite3
import sys
import numpy as np
import cv2
from pathlib import Path

JOB = Path(sys.argv[1]); OBJ = Path(sys.argv[2]); VID = Path(sys.argv[3]); MAP = Path(sys.argv[4]); DB = Path(sys.argv[5])
MAXS = 560


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


jid = int(JOB.name.split("_")[1])
c = sqlite3.connect(str(DB)); c.row_factory = sqlite3.Row
job = c.execute("SELECT object_id FROM jobs WHERE id=?", (jid,)).fetchone()
obj = c.execute("SELECT * FROM objects WHERE id=?", (job["object_id"],)).fetchone()
obs = json.loads(obj["obs"] or "[]")
poses = json.loads((MAP / "poses.json").read_text()); by_name = {f["name"]: f for f in poses["frames"]}
I = json.loads((MAP / "meta.json").read_text())["intrinsics"]; Hs, Ws = I["h"], I["w"]

sf = json.loads((OBJ / "splat_frame.json").read_text())
R_cu = np.array(sf["R"]); ctr_cu = np.array(sf["ctr"]); s_cu = float(sf["scale"])
pl = json.loads((JOB / "placement.json").read_text())
R_pp = np.array(pl["R"]); s_pp = float(pl["s"]); t_pp = np.array(pl["t"]); a_ctr = np.array(pl["a_ctr"])
up_w = np.array(pl["up"]); base_off = float(pl["base_off"])
def world_to_cu(Xw):
    Xa = ((Xw - t_pp - up_w * base_off) / s_pp) @ R_pp + a_ctr       # R_pp^T (..) row-wise
    return s_cu * (Xa - ctr_cu) @ R_cu.T

# rgb = the DEGLARED working stream (2026-08-27 swap); rgb_orig = raw S0 colour
layers = [("rgb", VID / "rgb")]
if (VID / "rgb_orig").exists():
    layers.append(("rgb_orig", VID / "rgb_orig"))
if (VID / "layers").exists():
    layers += [(d.name, d) for d in sorted((VID / "layers").iterdir()) if d.is_dir()
               if d.name != "rgb_deglare"]
out_dir = OBJ / "views"; out_dir.mkdir(exist_ok=True)
views = []
for ob in obs:
    name = ob.get("frame")
    f = by_name.get(name)
    if f is None: continue
    C = np.array(f["p"]); d = world_to_cu(C); d /= np.linalg.norm(d) + 1e-9
    yaw = float(np.arctan2(d[0], d[2])); pitch = float(np.arcsin(np.clip(d[1], -1, 1)))
    pts = np.array(ob["pts"], float)
    if ob.get("kind") == "rect" or len(pts) == 2:
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    else:
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    m = 0.12 * max(x1 - x0, y1 - y0)
    # upright frame: Hs wide (1024), Ws tall (1224) -> clamp
    X0, Y0 = int(max(0, x0 - m)), int(max(0, y0 - m)); X1, Y1 = int(min(Hs, x1 + m)), int(min(Ws, y1 + m))
    i = len(views)
    have = []
    for lname, ldir in layers:
        p = ldir / (name if lname in ("rgb", "nxyz") else name.rsplit(".", 1)[0] + ".jpg")
        if not p.exists():
            p = ldir / name
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None: continue
        up = np.rot90(img, k=3)
        crop = up[Y0:Y1, X0:X1]
        if crop.size == 0: continue
        sc = min(1.0, MAXS / max(crop.shape[:2]))
        if sc < 1: crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"v{i}_{lname}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
        have.append(lname)
    views.append({"i": i, "name": name, "cam": name[:1], "yaw": yaw, "pitch": pitch, "dir": d.tolist(),
                  "rect": [X0, Y0, X1, Y1], "auto": bool(ob.get("auto")), "layers": have})
(OBJ / "views.json").write_text(json.dumps({"views": views, "layers": [l for l, _ in layers]}))
print(f"OBJECT_VIEWS_DONE {len(views)} views x {len(layers)} layers -> {out_dir}", flush=True)
