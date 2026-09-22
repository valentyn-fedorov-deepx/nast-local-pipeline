"""Make one recording folder out of the per-camera frame folders of a rig, so the service knows which frame is which camera.

The pipeline tells the cameras apart by the first letter of the frame name (A_..., B_...): poses, the map, the ROI
solve and the app all read it. The Orthovector units write bare timestamps (2026_09_18_19_22_30_148427_et_..._iso_100.raw)
into a folder per unit, so this step links every frame of the first folder as A_<name> and of the second as B_<name>
into <out_dir>. Hard links when the disk allows it (no copy, no extra space), symlinks otherwise, --copy to copy.
Which unit is the front one does not matter to the pipeline: it works that out from the drive itself.

    python import_rig.py <out_dir> <frames of camera A> [<frames of camera B>] [--copy] [--upright-k=N]

Each camera folder is searched recursively for *.raw / *.raw12 (the layered archives extract into
<unit>/<session>/sequences/<session>/*.raw). Frames of the wrong size are left out and reported, session.json of every
unit is kept next to the frames as <letter>_session.json.

It also writes <out_dir>/rig.json, which the decoders read (monocars/polar_normals.py, RIG_DEFAULT):
  upright_k  how to turn the sensor image upright: from session.json (device.pipeline_rotation_deg 180 -> 2,
             0 -> 0); without a session.json the 07.08 rig (3). --upright-k=N sets it for every camera.
"""
import datetime
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FRAME_BYTES = 2048 * 3680                            # IMX264 raw12 with the 3680-byte row stride
TS = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{6})")

args = [a for a in sys.argv[1:] if not a.startswith("--")]
COPY = "--copy" in sys.argv
UPK = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--upright-k=")), None)
if len(args) < 2:
    sys.exit(__doc__)
OUT = Path(args[0]); cams = list(zip("ABCD", args[1:]))
OUT.mkdir(parents=True, exist_ok=True)


def ts_of(name):
    m = TS.search(name)
    if not m:
        return None
    y, mo, d, h, mi, s, us = (int(g) for g in m.groups())
    return datetime.datetime(y, mo, d, h, mi, s, us, tzinfo=datetime.timezone.utc).timestamp()


def place(src, dst):
    if dst.exists() or dst.is_symlink():
        return "kept"
    if COPY:
        shutil.copy2(src, dst); return "copied"
    try:
        os.link(src, dst); return "linked"
    except OSError:
        try:
            os.symlink(src.resolve(), dst); return "symlinked"
        except OSError:
            shutil.copy2(src, dst); return "copied"


def mount_of(root):
    """(upright_k, how) from the unit's session.json"""
    if UPK is not None:
        return UPK % 4, "--upright-k"
    for sj in root.rglob("session.json"):
        try:
            dev = json.loads(sj.read_text()).get("device", {})
        except Exception:
            continue
        rot = dev.get("pipeline_rotation_deg")
        if rot is None:
            continue
        rot = int(rot) % 360
        if rot in (0, 180):
            return rot // 90, f"session.json: pipeline_rotation_deg {rot}, sensor {dev.get('sensor_orientation', '?')}"
        sys.exit(f"{sj}: pipeline_rotation_deg {rot} - a quarter turn, its direction is not in the file: rerun with --upright-k=1 or 3 "
                 "(np.rot90 turns that make the raw image upright) after looking at one frame")
    return 3, "no session.json: the 07.08 rig (camera rolled -90)"


rig = {}
spans = {}
for letter, folder in cams:
    root = Path(folder)
    if not root.is_dir():
        sys.exit(f"camera {letter}: {root} is not a folder")
    frames = sorted(set(root.rglob("*.raw")) | set(root.rglob("*.raw12")), key=lambda p: p.name)
    if not frames:
        sys.exit(f"camera {letter}: no *.raw / *.raw12 under {root}")
    bad = [p for p in frames if p.stat().st_size != FRAME_BYTES]
    nots = [p for p in frames if ts_of(p.name) is None]
    good = [p for p in frames if p.stat().st_size == FRAME_BYTES and ts_of(p.name) is not None]
    how = {}
    for p in good:
        r = place(p, OUT / f"{letter}_{p.name}"); how[r] = how.get(r, 0) + 1
    for sj in root.rglob("session.json"):
        shutil.copy2(sj, OUT / f"{letter}_session.json"); break
    tt = sorted(ts_of(p.name) for p in good)
    dts = [b - a for a, b in zip(tt, tt[1:])]
    med = sorted(dts)[len(dts) // 2] if dts else 0.0
    spans[letter] = (tt[0], tt[-1]) if tt else (0, 0)
    print(f"camera {letter} <- {root}")
    print(f"  {len(good)} frames placed ({', '.join(f'{v} {k}' for k, v in how.items())}), {len(bad)} of the wrong size left out, {len(nots)} without a timestamp left out")
    if tt:
        print(f"  {datetime.datetime.fromtimestamp(tt[0], datetime.timezone.utc):%Y-%m-%d %H:%M:%S} .. {datetime.datetime.fromtimestamp(tt[-1], datetime.timezone.utc):%H:%M:%S} UTC, "
              f"{tt[-1] - tt[0]:.1f} s, {1 / med if med else 0:.2f} fps, {sum(1 for d in dts if d > max(1.0, 6 * med))} hole(s) in time")
    for p in bad[:3]:
        print(f"  wrong size: {p.name} ({p.stat().st_size} bytes)")
    if good:
        upk, how = mount_of(root)
        rig[letter] = {"upright_k": upk, "unit": root.name, "mount": how}
        turn = {0: "as recorded", 1: "turned 90 deg", 2: "turned 180 deg", 3: "turned 90 deg the other way"}[upk]
        print(f"  upright = sensor image {turn} ({how})")
(OUT / "rig.json").write_text(json.dumps(rig, indent=1))
if len(spans) > 1:
    (a0, a1), (b0, b1) = spans["A"], spans["B"]
    lo, hi = max(a0, b0), min(a1, b1)
    print(f"cameras A and B overlap in time for {max(0.0, hi - lo):.1f} s (A starts {b0 - a0:+.1f} s before B, ends {a1 - b1:+.1f} s after); "
          f"frames outside the overlap get poses only through their own chunks")
n = len([p for p in OUT.iterdir() if p.suffix in (".raw", ".raw12")])
print(f"IMPORT_RIG_DONE {OUT}: {n} frames, {n * FRAME_BYTES / 1e9:.1f} GB of raw, open this folder in the app (Open data folder)")
