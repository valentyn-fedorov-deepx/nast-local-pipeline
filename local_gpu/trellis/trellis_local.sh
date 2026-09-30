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
if [ -x "$ROOT/env/bin/python" ]; then
  # relocatable conda-pack env (unpack_trellis.sh): activation runs the packages' hooks too
  source "$ROOT/env/bin/activate"
  PY="$ROOT/env/bin/python"
else
  [ -f "$ROOT/miniconda3/etc/profile.d/conda.sh" ] && source "$ROOT/miniconda3/etc/profile.d/conda.sh"
  export CONDARC="$ROOT/condarc" CONDA_ENVS_PATH="$ROOT/miniconda3/envs"
  command -v conda >/dev/null 2>&1 || { echo "conda not found (run install_trellis.sh)"; exit 2; }
  conda activate "$ENV_NAME"
  PY="$ROOT/miniconda3/envs/$ENV_NAME/bin/python"
fi
[ -x "$PY" ] || { echo "no python in the TRELLIS env under $ROOT (run install_trellis.sh or unpack_trellis.sh)"; exit 2; }
export CONDA_PREFIX="${CONDA_PREFIX:-$(dirname "$(dirname "$PY")")}"
# nvdiffrast JIT-compiles its torch plugin on first use: same toolchain as the install
export CUDA_HOME="$CONDA_PREFIX" PATH="$CONDA_PREFIX/bin:$PATH"
TGT="$CONDA_PREFIX/targets/x86_64-linux"
export CPATH="$TGT/include:$CONDA_PREFIX/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$TGT/lib:$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$TGT/lib:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[ -x "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" ] &&   export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions"     # JIT builds off the root disk
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
# The mesh export of the generator has a time budget and is ended from inside when it runs over (see trellis_gen.py):
# asset_mesh.pending is then left behind. The gaussians are on disk by that point, so the object goes on without the
# generated mesh instead of failing. Anything else that ends the generator is an error, as before.
gen() {
  local rc=0
  rm -f "$J/asset_mesh.pending"
  "$PY" "$HERE/trellis_gen.py" "$J/asset" "$@" || rc=$?
  [ "$rc" = 0 ] && return 0
  if [ -f "$J/asset_mesh.pending" ] && [ -s "$J/asset.ply" ]; then
    rm -f "$J/asset_mesh.pending" "$J/asset_mesh.glb" "$J/asset_mesh.ply" "$J/asset_mesh_uv.npy" "$J/asset_mesh_tex.png"
    echo "mesh export skipped: it did not finish within its time budget (generator rc=$rc); the object keeps its gaussians"
    return 0
  fi
  return "$rc"
}
if [ "$n" -gt 0 ]; then
  gen "$J"/crops_enh/*.png
else
  echo "ENHANCE_EMPTY: falling back to raw crops"
  gen "$J"/crops/*.png
fi
echo "LOCAL_TRELLIS_DONE $J"
