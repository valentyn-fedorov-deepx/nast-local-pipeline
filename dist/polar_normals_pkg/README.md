# Polarization normals from Orthovector raw frames

`polar_normals.py` — the math (a Python re-implementation of the VyzAI
CameraController normal stack, verified against the controller's own renders),
`raw_to_layers.py` — the batch tool: a folder of `.raw12` → RGB + normal layers.

## Install

```
pip install -r requirements.txt        # numpy, opencv-python
```

## Run

```
python raw_to_layers.py <in_dir> <out_dir>                       # rgb, rgb_orig, nxyz, n_xy
python raw_to_layers.py <in_dir> <out_dir> --products all         # + n_xz, nxyz_phys, nxyz_diffuse, nxyz_specv2, edge, rgb_deglare
python raw_to_layers.py <in_dir> <out_dir> --png --workers 8      # lossless, more processes
python raw_to_layers.py <in_dir> <out_dir> --roll -90             # camera rolled -90 deg (the Orthovector rig mount)
```

Output layout: `<out_dir>/rgb/<stem>.jpg`, `<out_dir>/rgb_orig/<stem>.jpg`,
`<out_dir>/nxyz/<stem>.jpg`, `<out_dir>/n_xy/<stem>.jpg`, … — one file per
raw frame, same stem. Re-running skips frames that are already done.
Throughput: ~0.8 s per frame per process (all products), ~0.3 s for nxyz+n_xy.

From Python:

```python
from polar_normals import PolarFrame, products
fr = PolarFrame(open("frame.raw12", "rb").read())
rgb      = fr.color_work()          # recorder look x glare attenuation  (H, W, 3) uint8 BGR
rgb_orig = fr.color_recorder()      # recorder look, plain
P = products(fr, roll_deg=0.0)      # dict: nxyz, n_xy, n_xz, nxyz_phys, nxyz_diffuse, nxyz_specv2, edge, rgb_deglare
```

## Raw format (what the code assumes)

* Sensor IMX264MYR, colour polarization filter array, 2448×2048, MIPI RAW12
  (3 bytes → 2 pixels, low nibbles in byte 2), rows padded to 3680 bytes
  → one frame = 2048 × 3680 = 7 536 640 bytes.
* 4×4 super-pixel = 2×2 Bayer (BG on the polarizer sub-images) × 2×2
  polarizers in Sony's native order: (0,0)=90°, (0,1)=45°, (1,0)=135°, (1,1)=0°.
* Every product is half resolution, 1224×1024, in the sensor orientation
  (the same frame as the recorder's `rgb_half`).
* **Camera roll.** `--roll` (CLI) / `roll_deg` (`products()`) is the roll of the
  camera on its mount. This package defaults to **0°: an upright camera, the
  normal vectors are painted exactly as the sensor sees them, no rotation.**
  Our Orthovector rig has the camera rolled −90° (the image is displayed
  with `rot90(k=3)`), and for that mount the in-plane components are turned
  to match the display frame (`--roll -90`: nx' = −ny, ny' = nx). Any other
  angle is a plain in-plane rotation of (nx, ny) by −roll; nz is untouched.
  Only the *vectors* are affected — the pixel grid stays in sensor
  orientation either way.

## Conventions (locked to the CameraController)

* Stokes: S0 = I0+I90 (+I45+I135)/2 as in the controller, S1 = I0−I90,
  S2 = I45−I135; θ = ½·atan2(S1, S2) (the `cv::phase(S2, S1)` convention,
  equals textbook AoLP − 45°); DoLP = √(S1²+S2²)/S0.
* `nxyz` — CC pseudo normals (`init_normals` verbatim): nx = sin d·cos θ,
  ny = −sin d·sin θ, nz = |cos d|·0.05, "enhanced" and normalized; painted as
  absolute components. `n_xy` / `n_xz` — the same field with z / y zeroed.
* `nxyz_phys` — VyzLut Fresnel lookup (n = 1.5); `nxyz_diffuse` — PxDiffuse
  (Atkinson); `nxyz_specv2` — PxSpecularV2 (Kadambi, n + ik root 1). These
  are painted signed (x, y) + |z|, like the controller's physnorm dumps.
* Paint order is "canon": R = X, G = Y, B = Z. Decode a layer back with
  nx = R/255·2−1 (signed products) or R/255 (absolute products), etc.
* Colour: `rgb_orig` reproduces the recorder's own rgb — linear scaling of
  S0/2 with fixed per-channel gains (no gamma, no auto white balance; error
  ≈ 1/255 vs the recorder's jpgs). `rgb` multiplies it by the polarization
  glare attenuation (Stokes-minimum luma ratio floored at 0.35, 5-px blur) —
  windshields and wet paint lose their sheen, everything else is untouched.

`sample/` holds one raw frame to test on: `python raw_to_layers.py sample out_test`.
