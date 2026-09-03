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

## 3. Data

Two Drive bundles carry code + data, same tree layout (extract both into
the same place):

| bundle                    | size   | what                                   |
|---------------------------|--------|----------------------------------------|
| `nast_v2_core.tar`        | ~14 GB | code, base map, `street_video/{rgb,raw}`, the shipped objects, the objects db |
| `nast_v2_gpu_extras.tar`  | ~6 GB  | VGGT weights, `street_video/{depth,rgb_orig}`, the `job_54` map scene — only for the local map rebuild and the rgb_orig comparison layer |

Inputs of a scene are **`rgb/` + `raw/` only**. The locked normals catalog
(`layers/`) is decoded on the machine: the service starts the batch decode
the moment it sees `raw/` without a complete catalog (or when you open a
folder in the app), and the app shows a progress gate until it is done.

## 4. Run

```
./run.sh          # or click the "NAST Deskview" desktop shortcut
```

Starts the service on :8130 (once) and opens the desktop app (the Qt port of
the WPF Deskview, `linux_app/deskview_qt.py`); `NAST_WEB=1 ./run.sh` opens
the browser GUI instead. Health check by hand:

```
curl http://127.0.0.1:8130/api/meta     # expect intrinsics + point count
```

The GUI status bar must say "Backend online · N pts indexed".

## 5. Using the pipeline

* **Open data folder** (header button): pick any folder with `rgb/` + `raw/`
  — the service switches to it, decodes it (progress gate), and the recorder
  reloads from it. The map/poses stay.
* **Objects**: recorder tab → draw an ROI → Reconstruct. Fully local
  (`NAST_LOCAL=1` is the default): box solve → point asset from the dense
  map → placement → splat + point close-ups → mesh (`mesh_from_points`
  when there is no TRELLIS mesh) → layer sets → structural modes
  (Mesh / Sketch / Exploded + the "Segments" colour set) →
  split real-view panel. ~2–4 min per object, CPU.
* **Map rebuild**: map tab → rebuild. Runs `local_gpu/vggto_local.py` per
  camera on the local GPU: chunks start at 24 frames @ 512 px and halve
  automatically on OOM (12 GB cards degrade gracefully). ~10 min/camera.
* **New raw take**: put the frames as `<scene>/rgb/*.jpg` + `<scene>/raw/*.raw12`
  (`monocars/extract_raw.py` pulls raw/ out of the rig tars, `gen_rgb_soft.py`
  builds the working rgb) and open that folder in the app — the decode runs
  by itself. `monocars/decode_raw.py <scene>` is the same batch from a shell.

## 5a. TRELLIS on this machine (full object quality)

Without TRELLIS the local route meshes the map points inside the box — a
one-sided crust. The generative objects (closed textured mesh + gaussians,
as on tex1) need TRELLIS on the local GPU. One command, ~40–60 min, ~15 GB:

```
NAST_TRELLIS_ROOT=/big/disk/nast_trellis bash local_gpu/trellis/install_trellis.sh
```

It creates a conda env `trellis` (installs miniconda under the root when
there is none), clones microsoft/TRELLIS, builds its CUDA extensions with a
conda-provided nvcc (no system CUDA toolkit needed), downloads the weights
(TRELLIS-image-large, SAM ViT-H, Real-ESRGAN x4, DINOv2) into its own caches,
and ends with a real generation on a sample crop. Pins are chosen by the GPU:
Ampere/Ada (RTX 30xx/40xx, A-series) → torch 2.4.0+cu121; Blackwell
(RTX 50xx) → torch 2.7.0+cu128. Success writes `<root>/env_ok`; from then on
stage 4/6 of every reconstruction runs SAM + Real-ESRGAN + TRELLIS locally
(`local_gpu/trellis/trellis_local.sh`). 8 GB cards work: the models visit the
GPU one stage at a time and the fusion drops to 3 views on out-of-memory.
Set the same `NAST_TRELLIS_ROOT` before starting the service (`run.sh`), or
leave the default `$HOME/nast_trellis`. `NAST_TRELLIS=0` forces the points
route even when TRELLIS is installed.

## 6. Environment switches

| var                     | default | meaning                                   |
|-------------------------|---------|-------------------------------------------|
| `NAST_LOCAL`            | 1       | 0 = use the remote tex1 TRELLIS/Omega route |
| `NAST_DEGLARE_CROPS`    | 1       | 0 = reconstruction crops from rgb_orig    |
| `NAST_TRELLIS`          | 1       | 0 = never use the local TRELLIS env       |
| `NAST_TRELLIS_ROOT`     | ~/nast_trellis | where install_trellis.sh put the env + weights |

Set them before starting the service, e.g. `NAST_LOCAL=0 ./run.sh`.

## 7. Troubleshooting

* Service didn't start → `cat inspector/srv.err`.
* Port 8130 busy → `fuser -k 8130/tcp` (or find the stale python) and rerun.
* GPU sanity → `venv/bin/python -c "import torch; print(torch.cuda.is_available())"`.
* Map rebuild OOM loops down to 6-frame chunks; if it still OOMs the card
  is below 12 GB — run with `NAST_LOCAL=0` against a GPU server instead.
* GUI opens but scenes are empty → the data trees from step 3 are missing
  or placed outside the repo root.
