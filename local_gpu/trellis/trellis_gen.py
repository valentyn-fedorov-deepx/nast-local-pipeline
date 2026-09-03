"""Image crop(s) -> 3D asset via TRELLIS, plus a turntable preview and the
mesh decoded from the same latent.
The crop can be blurry and tiny -- that is the whole point. TRELLIS carries the
object prior; the photo only has to pin down class, colour and orientation.
RGBA inputs keep their alpha (our SAM silhouettes), RGB inputs go through rembg.

Low-VRAM by construction (8 GB laptop cards): the models stay on the CPU and
each one visits the GPU only for its own stage -- DINOv2 for conditioning,
the sparse-structure flow + decoder, the SLAT flow, then the gaussian and
mesh decoders one after the other; the radiance-field decoder is never run.
Peak memory is one model plus its activations instead of the whole set.
On a CUDA out-of-memory the fusion retries with fewer views (5 -> 3 -> 1).

Usage: python trellis_gen.py <out_stem> <crop1.png> [crop2.png ...]
Env:   TRELLIS_DIR (the microsoft/TRELLIS checkout), TRELLIS_STEPS (default 20),
       TRELLIS_VRAM_CAP_GB (testing: cap this process at N GB to emulate a
       smaller card), TRELLIS_KEEP_ON_GPU=1 (big cards: no offloading)
"""
import os
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")
import sys
import gc
import imageio
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blackwell_shim  # noqa: F401  (RTX 50xx: route xformers to its CUTLASS kernels)
sys.path.insert(0, os.environ.get("TRELLIS_DIR", os.path.expanduser("~/nast_trellis/TRELLIS")))
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils

stem = sys.argv[1]
CAP = float(os.environ.get("TRELLIS_VRAM_CAP_GB", "0") or 0)
if CAP > 0:
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, CAP * 1e9 / total))
    print(f"VRAM capped at {CAP} GB of {total / 1e9:.1f}", flush=True)
OFFLOAD = os.environ.get("TRELLIS_KEEP_ON_GPU", "0") != "1"


def _load(p):
    im = Image.open(p)
    # keep our own alpha (silhouette masks) -- TRELLIS then skips rembg
    return im.convert("RGBA") if im.mode in ("RGBA", "LA") else im.convert("RGB")


imgs = [_load(p) for p in sys.argv[2:]]
print("inputs:", len(imgs), [im.size for im in imgs], flush=True)
pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
if OFFLOAD:
    pipe.cpu()
    # the pipeline asks self.device for where to create noise; with the models
    # parked on the CPU that must still answer "cuda"
    type(pipe).device = property(lambda self: torch.device("cuda"))
else:
    pipe.cuda()
steps = int(os.environ.get("TRELLIS_STEPS", "20"))
SS = {"steps": steps, "cfg_strength": 7.5}
SL = {"steps": steps, "cfg_strength": 3.0}
FORMATS = ["gaussian", "mesh"]


class stage:
    """Bring the named models to the GPU for one stage, park them after."""
    def __init__(self, *names):
        self.names = names

    def __enter__(self):
        if OFFLOAD:
            for n in self.names:
                pipe.models[n].cuda()

    def __exit__(self, *a):
        if OFFLOAD:
            for n in self.names:
                pipe.models[n].cpu()
            gc.collect(); torch.cuda.empty_cache()


def mem(tag):
    print(f"  [{tag}] peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)


@torch.no_grad()
def generate(views):
    views = [pipe.preprocess_image(v) for v in views]
    with stage("image_cond_model"):
        cond = pipe.get_cond(views)
    cond["neg_cond"] = cond["neg_cond"][:1]
    mem("cond")
    torch.manual_seed(1)
    multi = len(views) > 1
    # multidiffusion fuses the views jointly at every step (stochastic picks a
    # view per step and averages -- softer, distorted bodies on inconsistent sets)
    with stage("sparse_structure_flow_model", "sparse_structure_decoder"):
        if multi:
            with pipe.inject_sampler_multi_image("sparse_structure_sampler", len(views), SS["steps"],
                                                 mode="multidiffusion"):
                coords = pipe.sample_sparse_structure(cond, 1, SS)
        else:
            coords = pipe.sample_sparse_structure(cond, 1, SS)
    mem("sparse structure")
    with stage("slat_flow_model"):
        if multi:
            with pipe.inject_sampler_multi_image("slat_sampler", len(views), SL["steps"],
                                                 mode="multidiffusion"):
                slat = pipe.sample_slat(cond, coords, SL)
        else:
            slat = pipe.sample_slat(cond, coords, SL)
    mem("slat")
    out = {}
    with stage("slat_decoder_gs"):
        out["gaussian"] = pipe.decode_slat(slat, ["gaussian"])["gaussian"]
    mem("decode gaussians")
    # the mesh decoder's FlexiCubes grid is the one stage that needs ~6.5 GB;
    # on a card that cannot hold it the asset still ships as gaussians and the
    # service meshes those points downstream (mesh_from_points)
    try:
        with stage("slat_decoder_mesh"):
            out["mesh"] = pipe.decode_slat(slat, ["mesh"])["mesh"]
        mem("decode mesh")
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        out["mesh"] = None
        print("TRELLIS mesh decoder out of memory -> no asset_mesh, "
              "the mesh is built from the gaussians downstream", flush=True)
    return out


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
if OFFLOAD:
    pipe.cpu(); gc.collect(); torch.cuda.empty_cache()
g = out["gaussian"][0]
g.save_ply(f"{stem}.ply")
video = render_utils.render_video(g, num_frames=120)["color"]
imageio.mimsave(f"{stem}_turn.mp4", video, fps=30)
# the mesh decoded from the SAME latent: a clean surface to paint layers on.
# to_glb simplifies, fills holes, parametrizes and bakes the gaussian look into
# a texture; its vertices are y-up (v @ T) while save_ply is (v @ T.T) -- undo
# the difference so the mesh sits exactly on the gaussians of <stem>.ply
try:
    if out.get("mesh") is None:
        raise RuntimeError("no TRELLIS mesh (decoder skipped)")
    import trimesh
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
mem("total")
print("TRELLIS_GEN_DONE", stem, f"views={n}", flush=True)
