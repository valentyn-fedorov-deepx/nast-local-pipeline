"""Official client n_xyz for the street dataset via gen_frames_from_pxi.

RAW12 tar frames -> temp uint16 PXI batches -> official dx_vyzai_python renderer
(--element n_xyz) -> axis fix for the -90 deg rig rotation (theta+90 => x<->y
swap, i.e. B<->G channel swap of the official output) -> resized layer jpgs.
"""
import json
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import polar_normals as pn

ROOT = Path(r"G:/nast_mode3")
REPO = Path(r"G:/dx_vyzai_python")
OUT = ROOT / "viewer/scenes/street_video/layers/nxyz_official"
OUT.mkdir(parents=True, exist_ok=True)

idx = json.loads((ROOT / "newshoot_2cam/raw_index.json").read_text())
scene = {p.stem for p in (ROOT / "viewer/scenes/street_video/rgb").glob("*.jpg")}
names = sorted(k for k in idx if k in scene)
todo = [k for k in names if not (OUT / f"{k}.jpg").exists()]
print(f"{len(names)} frames, {len(todo)} to do", flush=True)

B = 400
for bs in range(0, len(todo), B):
    batch = todo[bs:bs + B]
    tmp_in = ROOT / "tmp_pxi_batch"
    tmp_out = ROOT / "tmp_pxi_batch_out"
    for d in (tmp_in, tmp_out):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
    mapping = {}
    for i, k in enumerate(batch):
        e = idx[k]
        with open(e[0], "rb") as f:
            f.seek(e[1] + e[2]); raw = f.read(e[3])
        m = pn.unpack_raw12(raw)
        hdr = struct.pack("<" + "i" * 9 + "q" + "i" * 5,
                          1, 4, m.shape[1], m.shape[0], 1, 12, 0, 0, 1,
                          int(time.time()), 0, 0, 0, 0, 0)
        (tmp_in / f"Drive-{i:05d}.pxi").write_bytes(hdr + m.astype("<u2").tobytes())
        mapping[i] = k
    r = subprocess.run([sys.executable, str(REPO / "run_scripts/gen_frames_from_pxi.py"),
                        "--data-path", str(tmp_in), "--save-path", str(tmp_out),
                        "--element", "n_xyz", "--format", "png", "--workers", "6"],
                       cwd=str(REPO), env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("RENDER FAIL", r.stderr[-400:], flush=True)
        break
    sub = tmp_out / "tmp_pxi_batch"
    n_ok = 0
    for i, k in mapping.items():
        p = sub / f"tmp_pxi_batch-{i:03d}.png"
        if not p.exists():
            p = sub / f"tmp_pxi_batch-{i:05d}.png"
        if not p.exists():
            cand = list(sub.glob(f"*-{i:0d}.png")) or list(sub.glob(f"*{i:05d}*"))
            p = cand[0] if cand else None
        if p is None or not p.exists():
            continue
        im = cv2.imread(str(p))
        im = im[:, :, [0, 2, 1]]                     # x<->y swap (-90 fix): DPT stacks (z,y,x), swap G<->R
        im = cv2.resize(im, (1224, 1024), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT / f"{k}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        n_ok += 1
    print(f"batch {bs//B}: {n_ok}/{len(batch)}", flush=True)
    shutil.rmtree(tmp_in, ignore_errors=True)
    shutil.rmtree(tmp_out, ignore_errors=True)
print("OFFICIAL_NXYZ_DONE", len(list(OUT.glob('*.jpg'))), flush=True)
