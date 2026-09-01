"""Crop an object out of several observer frames by projecting its 3D box.

Picks azimuth-diverse observers (one per sector around the object), projects
the eight box corners into each, crops the 2D bbox with margin, rotates to
upright (the rig is rolled ~90 deg; cw restores it) and writes crop_*.png --
ready to feed TRELLIS multi-image.

Usage: python project_box_crop.py <pack_dir> <images_dir> <box.json> <out_dir> [n=4]
"""
import json
import sys
import numpy as np
from pathlib import Path
import cv2

PACK = Path(sys.argv[1]); IMAGES = Path(sys.argv[2])
BOX = json.loads(Path(sys.argv[3]).read_text())
OUT = Path(sys.argv[4]); OUT.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[5]) if len(sys.argv) > 5 else 4
MARGIN = float(sys.argv[6]) if len(sys.argv) > 6 else 0.12
TARGET_LUMA = 110.0            # dusk footage is far darker than TRELLIS's diet

meta = json.loads((PACK / "meta.json").read_text())
poses = json.loads((PACK / "poses.json").read_text())
by = {f["name"]: f for f in poses["frames"]}
I = meta["intrinsics"]; W, H = I["w"], I["h"]
up = np.array(poses["up"]); up /= np.linalg.norm(up)
ctr = np.array(BOX["center"]); size = np.array(BOX["size"])
corners = np.array([ctr + size / 2 * np.array(s)
                    for s in [(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]])


def Rof(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


# azimuth basis around up
ref = np.array([0, 1, 0]) if abs(up[1]) < 0.9 else np.array([1, 0, 0])
ax = np.cross(ref, up); ax /= np.linalg.norm(ax); az = np.cross(up, ax)

cands = []
for o in BOX["observers"]:
    f = by.get(o["name"])
    if f is None:
        continue
    C = np.array(f["p"]); d = C - ctr
    theta = np.arctan2(float(d @ az), float(d @ ax))
    cands.append((o["name"], theta, o["dist"]))

# bucket candidates per azimuth sector, nearest first; the closest passes may
# already be PAST the object (box beside/behind the forward-looking camera),
# so each sector walks its list until a candidate actually frames the box
sectors = {}
for name, theta, dist in cands:
    s = int((theta + np.pi) / (2 * np.pi) * N) % N
    sectors.setdefault(s, []).append((dist, name))
for s in sectors:
    sectors[s].sort()
print(f"observers {len(cands)}, sectors filled {len(sectors)}", flush=True)


def try_crop(name, made):
    f = by[name]
    R = Rof(f["q"]); C = np.array(f["p"])
    Xc = (corners - C) @ R.T
    if (Xc[:, 2] <= 0.05).any():
        return None
    u = I["fx"] * Xc[:, 0] / Xc[:, 2] + I["cx"]
    v = I["fy"] * Xc[:, 1] / Xc[:, 2] + I["cy"]
    x0, x1 = float(u.min()), float(u.max()); y0, y1 = float(v.min()), float(v.max())
    # most of the box must land inside the frame
    ix = max(0.0, min(x1, W) - max(x0, 0)); iy = max(0.0, min(y1, H) - max(y0, 0))
    if ix * iy < 0.6 * (x1 - x0) * (y1 - y0):
        return None
    # absolute floor on the margin: a 20-px wall camera still needs a bit of
    # context for rembg, and the upscale below makes it a real subject
    mw = max((x1 - x0) * MARGIN, 14)
    mh = max((y1 - y0) * MARGIN, 14)
    x0, x1 = max(0, int(x0 - mw)), min(W, int(x1 + mw))
    y0, y1 = max(0, int(y0 - mh)), min(H, int(y1 + mh))
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    img = cv2.imread(str(IMAGES / name))
    if img is None:
        return None
    crop = np.rot90(img[y0:y1, x0:x1], k=3)              # cw -> upright
    luma = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())
    if luma < TARGET_LUMA:
        gain = min(TARGET_LUMA / max(luma, 1.0), 3.0)
        crop = np.clip(crop.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    # tiny objects (a wall camera at 15 px) must LOOK like a subject:
    # upscale until the short side is respectable, or rembg sees only wall
    short = min(crop.shape[0], crop.shape[1])
    if short < 380:
        k = int(np.ceil(380 / max(short, 1)))
        crop = cv2.resize(crop, (crop.shape[1] * k, crop.shape[0] * k),
                          interpolation=cv2.INTER_LANCZOS4)
    p = OUT / f"crop_{made:02d}_{Path(name).stem[:20]}.png"
    cv2.imwrite(str(p), np.ascontiguousarray(crop))
    print(f"crop {p.name}: {crop.shape[1]}x{crop.shape[0]} from {name}", flush=True)
    return p


made = 0
for s in sorted(sectors):
    for dist, name in sectors[s]:
        if try_crop(name, made) is not None:
            made += 1
            break
print(f"PROJECT_BOX_CROP_DONE {made} crops -> {OUT}", flush=True)
