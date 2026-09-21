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
2. MAP tab, **poses**: `local_gpu/vggto_poses.py` on the local GPU, about 25 s per
   100 frames. It makes the camera poses AND the depth the rest of the pipeline
   unprojects (`<recording>/depth_geo/`, same 16-bit format as the MoGe dump, with
   `depth_geo_conf/`). `build map` runs this step by itself when the recording has
   no poses.
3. **build map** and objects as before. For such a recording the map is its
   `depth_geo` unprojected with its poses (`monocars/geo_layer.py`): no second GPU
   pass and no MoGe points; about 30 s per 100 frames.

Why the geometry is VGGT's alone. Monocular MoGe depth placed on any poses puts the
same physical point 6 to 20 % of its depth apart when it is seen from two frames
(1 to 3 m for a parked car: one car drawn several times along the road). Cameras and
depth of one VGGT-Omega chunk agree to 1 to 2 %, so:

* every camera runs in chunks of 24 frames with 6 shared ones, rotations, centres and
  depth maps are kept;
* the scale of a chunk is the height of its cameras above the road plane fitted to its
  own points, a constant of the rig; neighbouring chunks are tied by the depth ratio
  of the same pixels in the shared frames; one least-squares problem over the log
  scales. MoGe gives one number for the whole recording: that height in world units
  (SIFT matches triangulated against the MoGe depth, median over the chunks);
* the camera with the most frames carries the world: its chunks are chained by the
  shared cameras. Every other camera is not chained at all: the rig is rigid, so its
  pose is the reference camera's pose at the same timestamp times one rotation (road
  normal and driving direction as both cameras see them at the same moments). Each
  of its chunks is fitted onto that predicted track with its own scale and shift.
  Holes in time split such a camera into segments, no chunk spans a hole;
* the map draws every place from its near views only (the depth error grows with
  the distance) and drops low-confidence pixels, depth edges and the vignetted
  corners of the frames.

Every recording opened from its own folder keeps its pack in `<recording>/map/`
(`poses.json`, `meta.json`, `poses_report.json`, then the point cloud); the shipped
`viewer/scenes/street` is never touched by it, and objects are listed per recording.
Recomputing the poses is a new world: the map built on the old ones moves to
`map/stale_<time>/` and objects solved before have to be solved again.

Measured on the shipped recording (1580 frames, two cameras, 180 m) and on a
450-frame stretch of it run as a new take:

* the same point from two frames of camera A, near objects (closer than 15 m):
  0.19 m apart 4 frames later, 0.45 m after 16, 0.87 m after 32; with MoGe depth on
  chained poses it was 0.69, 1.31 and 2.95 m. Across the viewing ray the error is
  under 1 % of the depth, what is left is depth noise along the ray;
* against the COLMAP poses, camera A: position error median 0.8 m (max 1.8 m) after
  one similarity alignment, 0.09 m inside 40-frame windows, relative rotation over
  20 frames 1.1 deg, up vector 0.6 deg. The COLMAP track of camera B sits 1.1 m ahead
  of camera A; the clouds of the two cameras agree best with both in one place (a
  back-to-back unit), so the lever arm is zero (`NAST_RIG_LEVER` for another rig);
* 6 minutes for the 1580 frames on a 16 GB card, 2 minutes for 450.

World unit: one unit is 3.41 m (`NAST_WORLD_UNIT_M`), the scale of the shipped COLMAP
world in which every distance constant of the solver was tuned. The depth pass of a
new recording writes MoGe-2 depth in that unit, so its poses and everything
downstream keep the convention.

## Locked layer catalog

`nxyz`, `n_xy`, `n_xz`, `phys`, `diffuse`, `specv2`, `edge`, `rgb_deglare`
— everywhere: the recorder stream, the split panel, splat and mesh colour
sets. Mesh struct modes: **Mesh / Sketch / Skeleton / Exploded** with the
explode slider (`pos = base + disp·t`, cut-seam geometry per segment).
