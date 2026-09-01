# NAST Mode 3 — local pipeline

The full covert-capture → map → objects → mesh pipeline, running on **one
machine** (no GPU server anywhere): a 12-16 GB consumer GPU handles the
VGGT-Omega street reconstruction, everything else is CPU.

```
raw polarization drive (RAW12 / tars)
   │  polar_layers.py / gen_rgb_soft.py         (CPU)
   ▼
street_video/: rgb (deglare-attenuated working stream), rgb_orig,
               depth (MoGe), layers/ (nxyz, n_xy, n_xz, phys, diffuse,
               specv2, edge, rgb_deglare)
   │  local_gpu/vggto_local.py                  (GPU, 12-16 GB safe)
   ▼
scenes/street: dense world-anchored point map (per-chunk MoGe+Umeyama anchor)
   │  inspector/server.py  (ROI → box solve → local point asset →
   │                        place → pack → mesh → structural modes)
   ▼
scenes/obj_job_N: point + splat + MESH close-ups, layer sets baked,
                  struct modes: Mesh / Sketch / Skeleton / Exploded
                  (+ "Segments" colour set), split real-view panel
```

## Layout

| dir            | what                                                       |
|----------------|------------------------------------------------------------|
| `inspector/`   | the one local service (stdlib + numpy): scenes, ROI, box solve, reconstruction chain, http on :8130 |
| `viewer/`      | WebGL viewers (point / splat / mesh) + scene packers       |
| `monocars/`    | pipeline scripts: polarization products, deglare, layer baking, meshing, structural generator, views |
| `local_gpu/`   | VGGT-Omega local runner + model wrapper (`dx_wrap`)        |
| `deskview/`    | WPF desktop app (attaches to the service)                  |

## Setup

```
python -m venv venv
venv/Scripts/pip install -r requirements.txt
# torch for YOUR cuda, e.g.:
venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Model weights (not in git): put `vggt_omega_1b_512.pt` (~4.6 GB) into
`local_gpu/models/` — see `local_gpu/models/README.md`.

Data lives NEXT to the repo dirs (gitignored): `viewer/scenes/...` — the
street pack (`pos.f32`, `rgb.u8`, `poses.json`, `meta.json`),
`street_video/{rgb,rgb_orig,depth,layers}`.

## Run

```
venv/Scripts/python -u inspector/server.py 8130     # the service
dotnet run --project deskview                       # or the published exe
```

* Draw an ROI in the recorder → **Reconstruct**: the object is cut out of
  the dense map (no generative model, ~1-2 min CPU), placed back into the
  world, meshed (`mesh_from_points`), painted with every polarization layer,
  and gets the structural modes. Every new object flows into the mesh view
  automatically.
* **Rebuild map**: runs `vggto_local.py` per camera on the local GPU. Chunks
  start at 24 frames @ 512 px and halve automatically on OOM, so a 12 GB
  card degrades gracefully instead of crashing.
* `NAST_LOCAL=0` before starting the service restores the remote-GPU route
  (TRELLIS close-ups + tex1 Omega) where that infrastructure exists.
* `NAST_DEGLARE_CROPS=0` feeds reconstruction from the original (non-deglared)
  frames.

## Locked layer catalog

`nxyz`, `n_xy`, `n_xz`, `phys`, `diffuse`, `specv2`, `edge`, `rgb_deglare`
— everywhere: the recorder stream, the split panel, splat and mesh colour
sets. Mesh struct modes: **Mesh / Sketch / Skeleton / Exploded** with the
explode slider (`pos = base + disp·t`, cut-seam geometry per segment).
