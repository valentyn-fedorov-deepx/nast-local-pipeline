"""Image crop(s) -> 3D asset via TRELLIS, plus a turntable preview and the
mesh decoded from the same latent.
The crop can be blurry and tiny -- that is the whole point. TRELLIS carries the
object prior; the photo only has to pin down class, colour and orientation.
RGBA inputs keep their alpha (our SAM silhouettes), RGB inputs go through rembg.
Usage: python trellis_gen.py <out_stem> <crop1.png> [crop2.png ...]
Env:   TRELLIS_DIR (the microsoft/TRELLIS checkout), TRELLIS_STEPS (default 20)
On CUDA out-of-memory the fusion retries with fewer views (5 -> 3 -> 1), so a
12 GB card still finishes instead of failing the job.
"""
import os
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")
import sys
import gc
import imageio
import torch
from PIL import Image
sys.path.insert(0, os.environ.get("TRELLIS_DIR", os.path.expanduser("~/nast_trellis/TRELLIS")))
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils
stem = sys.argv[1]
def _load(p):
    im = Image.open(p)
    # keep our own alpha (silhouette masks) -- TRELLIS then skips rembg
    return im.convert("RGBA") if im.mode in ("RGBA", "LA") else im.convert("RGB")
imgs = [_load(p) for p in sys.argv[2:]]
print("inputs:", len(imgs), [im.size for im in imgs], flush=True)
pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipe.cuda()
steps = int(os.environ.get("TRELLIS_STEPS", "20"))
SS = {"steps": steps, "cfg_strength": 7.5}
SL = {"steps": steps, "cfg_strength": 3.0}
def generate(views):
    if len(views) == 1:
        return pipe.run(views[0], seed=1, sparse_structure_sampler_params=SS, slat_sampler_params=SL)
    # multidiffusion fuses the views jointly at every step (stochastic picks a
    # view per step and averages -- softer, distorted bodies on inconsistent sets)
    return pipe.run_multi_image(views, seed=1, mode="multidiffusion",
                                sparse_structure_sampler_params=SS, slat_sampler_params=SL)
out = None
n = len(imgs)
while out is None:
    try:
        out = generate(imgs[:n])
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        if n <= 1:
            raise
        n = 3 if n > 3 else 1
        print(f"CUDA out of memory -> retrying with {n} view(s)", flush=True)
g = out["gaussian"][0]
g.save_ply(f"{stem}.ply")
video = render_utils.render_video(g, num_frames=120)["color"]
imageio.mimsave(f"{stem}_turn.mp4", video, fps=30)
# the mesh decoded from the SAME latent: a clean surface to paint layers on.
# to_glb simplifies, fills holes, parametrizes and bakes the gaussian look into
# a texture; its vertices are y-up (v @ T) while save_ply is (v @ T.T) -- undo
# the difference so the mesh sits exactly on the gaussians of <stem>.ply
try:
    import numpy as np, trimesh
    from trellis.utils import postprocessing_utils
    glb = postprocessing_utils.to_glb(g, out["mesh"][0], simplify=0.95, texture_size=1024, verbose=False)
    glb.export(f"{stem}_mesh.glb")
    v = np.asarray(glb.vertices); v = np.stack([v[:, 0], -v[:, 1], -v[:, 2]], 1)
    trimesh.Trimesh(v, np.asarray(glb.faces), process=False).export(f"{stem}_mesh.ply")
    np.save(f"{stem}_mesh_uv.npy", np.asarray(glb.visual.uv, dtype=np.float32))
    glb.visual.material.baseColorTexture.save(f"{stem}_mesh_tex.png")
    print(f"mesh: {len(v)} verts, {len(glb.faces)} faces", flush=True)
except Exception as e:
    print("mesh export failed:", repr(e), flush=True)
print("TRELLIS_GEN_DONE", stem, f"views={n}", flush=True)
