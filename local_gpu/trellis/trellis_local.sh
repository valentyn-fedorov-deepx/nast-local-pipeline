#!/usr/bin/env bash
# Stage 4/6 of the local object route, on this machine's GPU (the same three
# steps tex1's inspector_trellis.sh ran): SAM mask + Real-ESRGAN x4 on the
# operator crops, then TRELLIS multi-image on the enhanced RGBA set.
# Usage: trellis_local.sh <job_dir>     (job_dir/crops/*.png [+ .box.json] in;
#        job_dir/asset.ply, asset_turn.mp4, asset_mesh.{ply,glb} + uv/tex, crops_enh/ out)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NAST_TRELLIS_ROOT:-$( [ -f "$HERE/ROOT" ] && cat "$HERE/ROOT" || echo "$HOME/nast_trellis")}"
ENV_NAME="${NAST_TRELLIS_ENV:-trellis}"
J="$1"
[ -d "$J/crops" ] || { echo "no crops in $J"; exit 2; }
[ -f "$ROOT/miniconda3/etc/profile.d/conda.sh" ] && source "$ROOT/miniconda3/etc/profile.d/conda.sh"
export CONDARC="$ROOT/condarc" CONDA_ENVS_PATH="$ROOT/miniconda3/envs"
command -v conda >/dev/null 2>&1 || { echo "conda not found (run install_trellis.sh)"; exit 2; }
conda activate "$ENV_NAME"
PY="$ROOT/miniconda3/envs/$ENV_NAME/bin/python"
[ -x "$PY" ] || { echo "no python in env $ENV_NAME under $ROOT (run install_trellis.sh)"; exit 2; }
export TRELLIS_DIR="${TRELLIS_DIR:-$ROOT/TRELLIS}"
export REALESRGAN_WEIGHTS="${REALESRGAN_WEIGHTS:-$ROOT/weights/RealESRGAN_x4plus.pth}"
# xformers everywhere (TRELLIS' sparse attention knows only xformers/flash_attn);
# on Blackwell trellis_gen.py's blackwell_shim steers it to the CUTLASS kernels
export ATTN_BACKEND="${ATTN_BACKEND:-xformers}" SPCONV_ALGO=native PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"   # less fragmentation on small cards
# the weights live in the install's own caches (see install_trellis.sh)
export HF_HOME="${HF_HOME:-$ROOT/cache/hf}" TORCH_HOME="${TORCH_HOME:-$ROOT/cache/torch}"
export XDG_CACHE_HOME="$ROOT/cache" TRITON_CACHE_DIR="$ROOT/cache/triton"   # catch-all: nothing lands in ~/.cache
"$PY" "$HERE/crop_enhance.py" "$J/crops" "$J/crops_enh"
n=$(ls "$J"/crops_enh/*.png 2>/dev/null | wc -l)
if [ "$n" -gt 0 ]; then
  "$PY" "$HERE/trellis_gen.py" "$J/asset" "$J"/crops_enh/*.png
else
  echo "ENHANCE_EMPTY: falling back to raw crops"
  "$PY" "$HERE/trellis_gen.py" "$J/asset" "$J"/crops/*.png
fi
echo "LOCAL_TRELLIS_DONE $J"
