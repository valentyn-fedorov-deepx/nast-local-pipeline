"""Per-frame MoGe-2 metric depth for a scene, as u16 PNG in millimetres.

Phase 2 of the raw import (after decode_raw.py): every frame gets a dense
depth map so ROI -> 3D solves see sharp object boundaries and the local map
build (VGGT-Omega) has its metric anchor + sky mask. Frames come from
rgb_orig/ (the recorder look; rgb/ when there is no rgb_orig), the output
is depth/<stem>.png with 0 = invalid (sky, unknown) and clip(depth*1000,
1, 65535) elsewhere -- the same encoding the shipped street_video/depth
uses. Resumable: frames that already have a depth map are skipped.

Usage: python moge_depth.py <scene_dir> [fov_x_deg|auto] [device]
  fov_x_deg: horizontal FOV of the camera (the service passes it from the
             scene intrinsics); "auto" lets MoGe estimate it per frame
Prints DEPTH_START / DEPTH_PROGRESS i/n ... / DEPTH_DONE for the service.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

SCENE = Path(sys.argv[1])
FOV = sys.argv[2] if len(sys.argv) > 2 else "auto"
DEV = sys.argv[3] if len(sys.argv) > 3 else ("cuda" if torch.cuda.is_available() else "cpu")
SRC = SCENE / "rgb_orig" if (SCENE / "rgb_orig").is_dir() else SCENE / "rgb"
OUT = SCENE / "depth"
OUT.mkdir(parents=True, exist_ok=True)
fov_x = None if FOV in ("auto", "", "0") else float(FOV)

frames = sorted(SRC.glob("*.jpg"))
todo = [p for p in frames if not (OUT / f"{p.stem}.png").exists()]
print(f"DEPTH_START total={len(frames)} todo={len(todo)} src={SRC.name} fov_x={fov_x} dev={DEV}", flush=True)
if not todo:
    print(f"DEPTH_DONE 0 new in 0s", flush=True)
    sys.exit(0)

from moge.model.v2 import MoGeModel                                  # noqa: E402
model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(DEV).eval()
print("moge loaded", flush=True)
(OUT / "meta_depth.json").write_text(json.dumps(
    {"unit": "mm", "invalid": 0, "model": "Ruicheng/moge-2-vitl-normal", "fov_x": fov_x,
     "source": SRC.name}, indent=1))

t0 = time.time(); ok = 0; err = 0
for i, p in enumerate(todo, 1):
    try:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError("unreadable")
        t = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0,
                         dtype=torch.float32, device=DEV).permute(2, 0, 1)
        with torch.inference_mode():
            out = model.infer(t, fov_x=fov_x, use_fp16=(DEV == "cuda"))
        depth = out["depth"].float().cpu().numpy()
        mask = out["mask"].cpu().numpy() > 0.5
        H, W = img.shape[:2]
        if depth.shape != (H, W):
            depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        u16 = np.where(mask & np.isfinite(depth), np.clip(depth * 1000.0, 1, 65535), 0).astype(np.uint16)
        cv2.imwrite(str(OUT / f"{p.stem}.png"), u16)
        ok += 1
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); err += 1
        print("ERR", p.stem, "cuda OOM", flush=True)
    except Exception as e:
        err += 1
        print("ERR", p.stem, repr(e), flush=True)
    if i % 10 == 0 or i == len(todo):
        el = time.time() - t0
        print(f"DEPTH_PROGRESS {i}/{len(todo)} ok={ok} err={err} {el / i:.2f}s/frame "
              f"eta={el / i * (len(todo) - i):.0f}s", flush=True)
print(f"DEPTH_DONE {ok} ok {err} err in {time.time() - t0:.0f}s", flush=True)
