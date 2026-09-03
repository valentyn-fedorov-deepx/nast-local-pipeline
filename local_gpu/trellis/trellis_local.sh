#!/usr/bin/env bash
# Stage 4/6 of the local object route, on this machine's GPU (the same three
# steps tex1's inspector_trellis.sh ran): SAM mask + Real-ESRGAN x4 on the
# operator crops, then TRELLIS multi-image on the enhanced RGBA set.
# Usage: trellis_local.sh <job_dir>     (job_dir/crops/*.png [+ .box.json] in;
#        job_dir/asset.ply, asset_turn.mp4, asset_mesh.{ply,glb} + uv/tex, crops_enh/ out)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NAST_TRELLIS_ROOT:-$HOME/nast_trellis}"
ENV_NAME="${NAST_TRELLIS_ENV:-trellis}"
J="$1"
[ -d "$J/crops" ] || { echo "no crops in $J"; exit 2; }
for c in "$ROOT/miniconda3" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
  [ -f "$c/etc/profile.d/conda.sh" ] && source "$c/etc/profile.d/conda.sh" && break
done
command -v conda >/dev/null 2>&1 || { echo "conda not found (run install_trellis.sh)"; exit 2; }
conda activate "$ENV_NAME"
export TRELLIS_DIR="${TRELLIS_DIR:-$ROOT/TRELLIS}"
export REALESRGAN_WEIGHTS="${REALESRGAN_WEIGHTS:-$ROOT/weights/RealESRGAN_x4plus.pth}"
export ATTN_BACKEND=xformers SPCONV_ALGO=native
python "$HERE/crop_enhance.py" "$J/crops" "$J/crops_enh"
n=$(ls "$J"/crops_enh/*.png 2>/dev/null | wc -l)
if [ "$n" -gt 0 ]; then
  python "$HERE/trellis_gen.py" "$J/asset" "$J"/crops_enh/*.png
else
  echo "ENHANCE_EMPTY: falling back to raw crops"
  python "$HERE/trellis_gen.py" "$J/asset" "$J"/crops/*.png
fi
echo "LOCAL_TRELLIS_DONE $J"
