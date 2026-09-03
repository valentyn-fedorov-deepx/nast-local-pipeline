#!/usr/bin/env bash
# TRELLIS for the local NAST pipeline: a conda env "trellis", the
# microsoft/TRELLIS repo with its CUDA extensions, and every weight the
# object stage needs (TRELLIS-image-large, SAM ViT-H, Real-ESRGAN x4, DINOv2).
#
# Two proven pin sets, picked by the GPU generation:
#   * Ampere / Ada  (RTX 30xx / 40xx, A-series)  -> torch 2.4.0+cu121 stack
#   * Blackwell     (RTX 50xx, sm_120)           -> torch 2.8.0+cu128 stack
# The flags that matter: ATTN_BACKEND=xformers (no flash-attn source build),
# SPCONV_ALGO=native (skips the auto-tuner that hangs on first run), and
# --no-build-isolation for the three extensions that import torch in setup.py.
# nvcc comes from conda's cuda-toolkit, so the machine needs no CUDA toolkit.
#
# Usage:  bash install_trellis.sh            (everything under $HOME/nast_trellis)
#         NAST_TRELLIS_ROOT=/big/disk/nast_trellis bash install_trellis.sh
# Re-runnable: finished steps are skipped. Ends with a real generation on a
# sample crop and writes $ROOT/env_ok — the service uses TRELLIS only when
# that marker exists.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NAST_TRELLIS_ROOT:-$HOME/nast_trellis}"
ENV_NAME="${NAST_TRELLIS_ENV:-trellis}"
mkdir -p "$ROOT/weights" "$ROOT/smoke" "$ROOT/cache/hf" "$ROOT/cache/torch" "$ROOT/cache/pip" "$ROOT/cache/conda_pkgs" "$ROOT/tmp"
cd "$ROOT"
# every cache lives under $ROOT: the root disk of a demo laptop is often full
export HF_HOME="$ROOT/cache/hf" TORCH_HOME="$ROOT/cache/torch" PIP_CACHE_DIR="$ROOT/cache/pip"
export CONDA_PKGS_DIRS="$ROOT/cache/conda_pkgs" TMPDIR="$ROOT/tmp"
export XDG_CACHE_HOME="$ROOT/cache" TRITON_CACHE_DIR="$ROOT/cache/triton"   # catch-all: nothing lands in ~/.cache
exec > >(tee -a "$ROOT/install.log") 2>&1
echo "=== TRELLIS install $(date) root=$ROOT ==="

# ---------------------------------------------------------------- conda
# a PRIVATE conda under $ROOT — never the user's own: that one usually lives
# on the small root disk, and an 8 GB env there fills it (seen on the demo box)
if [ ! -f "$ROOT/miniconda3/etc/profile.d/conda.sh" ]; then
  echo "--- installing miniconda into $ROOT/miniconda3"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$ROOT/miniconda.sh" || exit 1
  bash "$ROOT/miniconda.sh" -b -p "$ROOT/miniconda3" || exit 1
fi
source "$ROOT/miniconda3/etc/profile.d/conda.sh"
# private condarc (ignores ~/.condarc): conda-forge + nvidia only — the Anaconda
# "defaults" channels need an interactive Terms-of-Service click on conda >= 25
cat > "$ROOT/condarc" <<'CRC'
channels:
  - conda-forge
  - nvidia
channel_priority: flexible
CRC
export CONDARC="$ROOT/condarc"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main     --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
export CONDA_ENVS_PATH="$ROOT/miniconda3/envs"
conda --version || { echo "conda unavailable"; exit 1; }

# ---------------------------------------------------------------- GPU generation
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
[ -n "$CC" ] || { echo "nvidia-smi gave no compute capability — driver missing?"; exit 1; }
if [ "${CC%%.*}" -ge 10 ]; then GEN=blackwell; else GEN=ampere; fi
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) cc=$CC -> $GEN stack"

# ---------------------------------------------------------------- env + torch
PY="$ROOT/miniconda3/envs/$ENV_NAME/bin/python"
if [ ! -x "$PY" ]; then                       # absent or a half-made env from a killed run
  conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1; rm -rf "$ROOT/miniconda3/envs/$ENV_NAME"
  conda create -y --override-channels -c conda-forge -n "$ENV_NAME" python=3.10 || exit 1
fi
conda activate "$ENV_NAME"
[ -x "$PY" ] || { echo "env has no python: $PY"; exit 1; }
PIP="$PY -m pip"                              # never the system pip (it would install into ~/.local on the root disk)
export PIP_USER=0 PYTHONNOUSERSITE=1          # and the env never sees ~/.local packages (stale nvidia-* there broke torch)
$PIP install -q --upgrade pip
if [ "$GEN" = ampere ]; then
  TORCH="torch==2.4.0 torchvision==0.19.0"; TIDX=https://download.pytorch.org/whl/cu121
  CUDA_TK=12.1
  XF="xformers==0.0.27.post2"
  KAO="kaolin==0.17.0"; KIDX=https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
  SPC="spconv-cu120==2.3.6"
  ARCH="8.6;8.9"
else
  TORCH="torch==2.8.0 torchvision==0.23.0"; TIDX=https://download.pytorch.org/whl/cu128
  CUDA_TK=12.8
  XF="xformers==0.0.32.post2"          # its flash ops are wrong for sm_120 -> blackwell_shim routes to CUTLASS
  KAO="kaolin==0.18.0"; KIDX=https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
  SPC="spconv-cu126==2.3.8"
  ARCH="8.6;8.9;12.0"
fi
# --extra-index-url: with the torch index alone pip resolves EVERY dependency
# there, and PyTorch prunes old nvidia-* wheels from it (cudnn 9.1.0.70 is gone)
$PIP install $TORCH --index-url $TIDX --extra-index-url https://pypi.org/simple || exit 1
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1
$PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "torch sees no GPU"; exit 1; }
$PIP cache purge >/dev/null 2>&1; rm -rf "$ROOT/tmp"/*     # the torch wheels are installed: drop the 5 GB of downloads

# nvcc for the extensions, exactly torch's CUDA line
if ! "$CONDA_PREFIX/bin/nvcc" --version >/dev/null 2>&1; then
  conda install -y --override-channels -c nvidia -c conda-forge "cuda-toolkit=$CUDA_TK" ||   conda install -y --override-channels -c "nvidia/label/cuda-$CUDA_TK.0" -c conda-forge cuda-toolkit || exit 1
fi
conda clean -a -y >/dev/null 2>&1                          # toolkit tarballs + extracted copies: ~4 GB of cache gone
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="$ARCH"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
nvcc --version | tail -1

# ---------------------------------------------------------------- python deps
$PIP install wheel setuptools ninja pillow imageio imageio-ffmpeg tqdm easydict \
    "opencv-python-headless==4.10.0.84" scipy rembg onnxruntime trimesh open3d xatlas \
    pyvista pymeshfix igraph "transformers==4.46.3" safetensors huggingface_hub plyfile \
    timm realesrgan basicsr --extra-index-url https://pypi.org/simple || exit 1
$PIP install $XF --index-url $TIDX --extra-index-url https://pypi.org/simple || exit 1
$PIP install --force-reinstall --no-deps $KAO -f $KIDX || exit 1   # same version, different torch build
$PIP install $SPC || exit 1
$PIP install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 || exit 1
$PIP cache purge >/dev/null 2>&1
# basicsr 1.4.2 imports a torchvision module that was removed in 0.17 — one-line fix
DEG=$($PY -c "import importlib.util, os; print(os.path.join(os.path.dirname(importlib.util.find_spec('basicsr').origin), 'data', 'degradations.py'))")
[ -f "$DEG" ] && sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$DEG"

# ---------------------------------------------------------------- TRELLIS + extensions
if [ ! -d "$ROOT/TRELLIS/trellis" ]; then
  git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git "$ROOT/TRELLIS" || exit 1
  (cd "$ROOT/TRELLIS" && git checkout -q 442aa1e && git submodule update --init --recursive)
fi
ext_ok() { $PY -c "import $1" >/dev/null 2>&1; }
ext_ok nvdiffrast || $PIP install --no-build-isolation \
    git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae || exit 1
if ! ext_ok diffoctreerast; then
  [ -d "$ROOT/src/diffoctreerast" ] || git clone --recursive https://github.com/JeffreyXiang/diffoctreerast.git "$ROOT/src/diffoctreerast"
  $PIP install --no-build-isolation "$ROOT/src/diffoctreerast" || exit 1
fi
if ! ext_ok diff_gaussian_rasterization; then
  [ -d "$ROOT/src/mip-splatting" ] || git clone https://github.com/autonomousvision/mip-splatting.git "$ROOT/src/mip-splatting"
  $PIP install --no-build-isolation "$ROOT/src/mip-splatting/submodules/diff-gaussian-rasterization" || exit 1
fi

# kaolin's wheel is built against ONE numpy major; flip until it imports
if ! $PY -c "import kaolin" >/dev/null 2>&1; then
  $PIP install "numpy==1.26.4" "opencv-python-headless==4.10.0.84"
  $PY -c "import kaolin" >/dev/null 2>&1 || $PIP install "numpy>=2.0,<2.3"
fi
$PY -c "import kaolin; print('kaolin', kaolin.__version__)" || { echo "kaolin import broken"; exit 1; }

# self-healing import: any module the pins missed gets installed by name
cd "$ROOT/TRELLIS"
ok=0
for i in 1 2 3 4 5 6; do
  out=$(ATTN_BACKEND=xformers SPCONV_ALGO=native $PY -c "import sys; sys.path.insert(0,'.'); from trellis.pipelines import TrellisImageTo3DPipeline; print('PIPELINE_IMPORT_OK')" 2>&1)
  echo "$out" | tail -3
  if echo "$out" | grep -q PIPELINE_IMPORT_OK; then ok=1; break; fi
  mod=$(echo "$out" | grep -oP "No module named .\K[a-zA-Z0-9_]+" | head -1)
  [ -z "$mod" ] && break
  $PIP install "$mod" || break
done
[ $ok -eq 1 ] || { echo "TRELLIS import failed — see $ROOT/install.log"; exit 1; }

# ---------------------------------------------------------------- weights
$PY - <<'EOF' || exit 1
from huggingface_hub import snapshot_download
for r in ("microsoft/TRELLIS-image-large", "facebook/sam-vit-huge"):
    snapshot_download(r); print("weights ok:", r, flush=True)
import torch
torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", pretrained=True)   # TRELLIS image encoder
print("weights ok: dinov2_vitl14_reg", flush=True)
EOF
[ -s "$ROOT/weights/RealESRGAN_x4plus.pth" ] || curl -fL --retry 3 \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
    -o "$ROOT/weights/RealESRGAN_x4plus.pth"

# ---------------------------------------------------------------- smoke: a real generation
export TRELLIS_DIR="$ROOT/TRELLIS" REALESRGAN_WEIGHTS="$ROOT/weights/RealESRGAN_x4plus.pth"
rm -rf "$ROOT/smoke"; mkdir -p "$ROOT/smoke/crops"
cp "$HERE/sample_crop.png" "$ROOT/smoke/crops/roi_00.png"
bash "$HERE/trellis_local.sh" "$ROOT/smoke" || { echo "smoke generation failed"; exit 1; }
[ -s "$ROOT/smoke/asset.ply" ] || { echo "smoke produced no asset.ply"; exit 1; }
echo "gen=$GEN env=$ENV_NAME date=$(date -Is)" > "$ROOT/env_ok"
echo "$ROOT" > "$HERE/ROOT"                       # the service and the worker find the install here
conda clean -a -y >/dev/null 2>&1; pip cache purge >/dev/null 2>&1; rm -rf "$ROOT/tmp"/* "$ROOT/miniconda.sh"
du -sh "$ROOT" | sed 's/^/footprint: /'
echo "TRELLIS_INSTALL_DONE ($GEN) -> $ROOT/env_ok"
