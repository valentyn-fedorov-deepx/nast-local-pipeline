#!/usr/bin/env bash
# Install a pack_trellis.sh archive on this machine — no compiler, no conda.
# Downloads the model weights (~7 GB) from HuggingFace / GitHub into the
# install root, then runs a real generation on the sample crop.
#   bash unpack_trellis.sh <nast_trellis_pack.tar> <ROOT>
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARC="$1"; ROOT="$2"
[ -f "$ARC" ] || { echo "no archive $ARC"; exit 1; }
mkdir -p "$ROOT/env" "$ROOT/cache/hf" "$ROOT/cache/torch" "$ROOT/tmp"
echo "--- extracting $(du -h "$ARC" | cut -f1) into $ROOT"
tar -xf "$ARC" -C "$ROOT"
tar -xzf "$ROOT/env.tar.gz" -C "$ROOT/env" && rm -f "$ROOT/env.tar.gz"
"$ROOT/env/bin/python" "$ROOT/env/bin/conda-unpack"   # rewrite the prefixes for this path (its shebang wants a bare "python")
# basicsr 1.4.2 imports a torchvision module removed in 0.17: without this line Real-ESRGAN silently falls back to Lanczos
DEG=$("$ROOT/env/bin/python" -c "import importlib.util, os; print(os.path.join(os.path.dirname(importlib.util.find_spec('basicsr').origin), 'data', 'degradations.py'))" 2>/dev/null)
[ -f "$DEG" ] && sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$DEG"
echo "$ROOT" > "$HERE/ROOT"
echo "--- MoGe-2 (metric depth for raw imports) into the env"
NAST_TRELLIS_ROOT="$ROOT" bash "$HERE/add_moge.sh"
echo "--- weights (HuggingFace + torch hub), into $ROOT/cache"
export HF_HOME="$ROOT/cache/hf" TORCH_HOME="$ROOT/cache/torch" XDG_CACHE_HOME="$ROOT/cache" TMPDIR="$ROOT/tmp" PYTHONNOUSERSITE=1
"$ROOT/env/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
for r in ("microsoft/TRELLIS-image-large", "facebook/sam-vit-huge"):
    snapshot_download(r); print("weights ok:", r, flush=True)
import torch
torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", pretrained=True)
print("weights ok: dinov2_vitl14_reg", flush=True)
PY
[ -s "$ROOT/weights/RealESRGAN_x4plus.pth" ] || curl -fL --retry 3 \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -o "$ROOT/weights/RealESRGAN_x4plus.pth"
echo "--- smoke: SAM + Real-ESRGAN + TRELLIS on this GPU (nvdiffrast compiles its plugin once, ~3 min)"
rm -rf "$ROOT/smoke"; mkdir -p "$ROOT/smoke/crops"; cp "$HERE/sample_crop.png" "$ROOT/smoke/crops/roi_00.png"
NAST_TRELLIS_ROOT="$ROOT" bash "$HERE/trellis_local.sh" "$ROOT/smoke"
[ -s "$ROOT/smoke/asset.ply" ] || { echo "smoke produced no asset.ply"; exit 1; }
echo "UNPACK_DONE -> $ROOT/env_ok"
