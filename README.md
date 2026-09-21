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

## Quick start — Linux

```
git clone <this repo> && cd nast-local-pipeline
bash install.sh          # venv + deps + torch (GPU-matched) + weights + desktop shortcut
./run.sh                 # or click the "NAST Deskview" shortcut
```

The shortcut starts the local service and opens the GUI as an app window
(chromium/chrome `--app`, falls back to the default browser). On Linux the
GUI is the web app — same screens and API as the Windows desktop app.
`SKIP_TORCH=1 bash install.sh` for a CPU-only install (no map rebuild).

## Quick start — Windows

```
git clone <this repo> ; cd nast-local-pipeline
powershell -ExecutionPolicy Bypass -File install.ps1   # + builds the WPF app if .NET 8 SDK exists
```

Click the "NAST Deskview" desktop shortcut (`run.bat`): service + WPF app,
or the browser GUI when the app is not built.

Model weights (not in git): `install.sh` pulls `vggt_omega_1b_512.pt`
(~4.6 GB) from tex1 automatically when it is reachable; otherwise put it
into `local_gpu/models/` by hand — see `local_gpu/models/README.md`.

Data lives NEXT to the repo dirs (gitignored): `viewer/scenes/...` — the
street pack (`pos.f32`, `rgb.u8`, `poses.json`, `meta.json`),
`street_video/{rgb,rgb_orig,depth,layers}`. The service boots without any
data (poses from `inspector/scene_base`) so a fresh clone opens fine.

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

## A new recording

The shipped street scene carries COLMAP poses. A fresh take has none, and the ROI
solve, the object views and the map build all need a pose per frame. The route:

1. **Open data folder**: the folder with the `.raw12` frames. Decode (rgb, rgb_orig,
   the normals catalog) and the MoGe-2 depth pass start by themselves.
2. MAP tab, **poses**: `local_gpu/vggto_poses.py` on the local GPU, about one minute
   per 300 frames. VGGT-Omega cameras on overlapping chunks are chained into one
   track; the scale of every chunk comes from triangulated SIFT matches against the
   MoGe depth (the depth maps of VGGT are not scale-consistent with its cameras, so
   they are not used); neighbouring chunks share camera centres, which ties the
   scales together; the world "up" is the camera axis that points at the sky; the
   second camera of the rig follows the first one by the shared trajectory, piece by
   piece when the recording has holes in time. `build map` runs this step by itself
   when the recording has no poses.
3. **build map** and objects as before.

Every recording opened from its own folder keeps its pack in `<recording>/map/`
(`poses.json`, `meta.json`, `poses_report.json`, then the point cloud); the shipped
`viewer/scenes/street` is never touched by it, and objects are listed per recording.
Recomputing the poses moves the map built on the old ones to `map/stale_<time>/`.

Checked against the COLMAP poses of the shipped recording (1580 frames, two cameras,
180 m): position error median 1.0 m (max 3.0 m) after one similarity alignment,
0.1 m inside 40-frame windows, relative rotation over 20 frames 1.1 deg, up vector
0.2 deg; 5 to 6 minutes on a 16 GB card.

World unit: one unit is 3.41 m (`NAST_WORLD_UNIT_M`), the scale of the shipped COLMAP
world in which every distance constant of the solver was tuned. The depth pass of a
new recording writes MoGe-2 depth in that unit, so its poses and everything
downstream keep the convention.

## Locked layer catalog

`nxyz`, `n_xy`, `n_xz`, `phys`, `diffuse`, `specv2`, `edge`, `rgb_deglare`
— everywhere: the recorder stream, the split panel, splat and mesh colour
sets. Mesh struct modes: **Mesh / Sketch / Skeleton / Exploded** with the
explode slider (`pos = base + disp·t`, cut-seam geometry per segment).
