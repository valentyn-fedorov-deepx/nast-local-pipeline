# Linux install & run guide

Everything runs on one Linux machine with an NVIDIA GPU (12–16 GB VRAM).
Verified boot path: Ubuntu-family, python 3.10/3.11.

## 0. Prerequisites

```
sudo apt update && sudo apt install -y git python3 python3-venv curl
nvidia-smi        # driver must be present; CUDA toolkit NOT required (torch wheels bundle it)
```

## 1. Clone (private repo — authenticate as a collaborator)

```
git clone https://github.com/valentyn-fedorov-deepx/nast-local-pipeline.git
cd nast-local-pipeline
```

## 2. Install — one command

```
bash install.sh
```

What it does: creates `venv`, installs python deps, installs torch matched
to the machine (cu128 wheel when `nvidia-smi` exists, CPU wheel otherwise),
pulls the VGGT weights from tex1 when reachable, and creates the
"NAST Deskview" desktop shortcut. Variants:

* `SKIP_TORCH=1 bash install.sh` — quick CPU-only install (no map rebuild).
* No tex1 access → copy `vggt_omega_1b_512.pt` (4.6 GB) into
  `local_gpu/models/` by hand.

## 3. Data (gitignored — copied separately)

The service boots with zero data, but the real pipeline needs the data
trees NEXT to the code, exactly these paths:

| path                                   | size   | what                              |
|----------------------------------------|--------|-----------------------------------|
| `viewer/scenes/street/`                | ~0.8G  | base map pack: pos.f32, rgb.u8, poses.json, meta.json, cells.json |
| `viewer/scenes/street_video/rgb/`      | ~0.3G  | working colour stream (deglare-attenuated) |
| `viewer/scenes/street_video/rgb_orig/` | ~0.3G  | original recorder frames          |
| `viewer/scenes/street_video/depth/`    | ~0.9G  | MoGe depth PNGs (map anchoring)   |
| `viewer/scenes/street_video/layers/`   | ~2.5G  | locked polarization products      |
| `viewer/scenes/obj_job_*`, `job_*`     | opt    | existing object scenes            |
| `inspector/inspector.db`, `inspector/jobs/` | opt | objects/jobs state           |
| `local_gpu/models/vggt_omega_1b_512.pt`| 4.6G   | VGGT-Omega checkpoint             |

Packing them on the source machine (Windows `tar` works since Win10):

```
cd <source nast root>
tar -czf nast_data.tgz viewer/scenes/street viewer/scenes/street_video/rgb ^
    viewer/scenes/street_video/rgb_orig viewer/scenes/street_video/depth ^
    viewer/scenes/street_video/layers inspector/inspector.db inspector/jobs
scp nast_data.tgz user@linuxbox:~/nast-local-pipeline/
```

On the Linux box: `tar -xzf nast_data.tgz` inside the repo root.

## 4. Run

```
./run.sh          # or click the "NAST Deskview" desktop shortcut
```

Starts the service on :8130 (once) and opens the GUI as a chromium/chrome
app window (falls back to the default browser). Health check by hand:

```
curl http://127.0.0.1:8130/api/meta     # expect intrinsics + point count
```

The GUI status bar must say "Backend online · N pts indexed".

## 5. Using the pipeline

* **Objects**: recorder tab → draw an ROI → Reconstruct. Fully local
  (`NAST_LOCAL=1` is the default): box solve → point asset from the dense
  map → placement → splat + point close-ups → mesh (`mesh_from_points`
  when there is no TRELLIS mesh) → layer sets → structural modes
  (Mesh / Sketch / Skeleton / Exploded + the "Segments" colour set) →
  split real-view panel. ~2–4 min per object, CPU.
* **Map rebuild**: map tab → rebuild. Runs `local_gpu/vggto_local.py` per
  camera on the local GPU: chunks start at 24 frames @ 512 px and halve
  automatically on OOM (12 GB cards degrade gracefully). ~10 min/camera.
* **New raw take** (fresh tars from the rig): regenerate the streams —

```
venv/bin/python monocars/polar_layers.py viewer/scenes/street_video <raw_index.json> 6 all 0
venv/bin/python monocars/gen_rgb_soft.py viewer/scenes/street_video <raw_index.json> 6
```

## 6. Environment switches

| var                     | default | meaning                                   |
|-------------------------|---------|-------------------------------------------|
| `NAST_LOCAL`            | 1       | 0 = use the remote tex1 TRELLIS/Omega route |
| `NAST_DEGLARE_CROPS`    | 1       | 0 = reconstruction crops from rgb_orig    |

Set them before starting the service, e.g. `NAST_LOCAL=0 ./run.sh`.

## 7. Troubleshooting

* Service didn't start → `cat inspector/srv.err`.
* Port 8130 busy → `fuser -k 8130/tcp` (or find the stale python) and rerun.
* GPU sanity → `venv/bin/python -c "import torch; print(torch.cuda.is_available())"`.
* Map rebuild OOM loops down to 6-frame chunks; if it still OOMs the card
  is below 12 GB — run with `NAST_LOCAL=0` against a GPU server instead.
* GUI opens but scenes are empty → the data trees from step 3 are missing
  or placed outside the repo root.
