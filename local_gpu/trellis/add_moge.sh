#!/usr/bin/env bash
# MoGe-2 (metric depth for raw imports) into the TRELLIS env — pure python on
# top of the env's torch, so no compiler. Pinned commits, fetched as GitHub
# archives (no git needed on the machine). Its optional deps (flex-gemm,
# pipeline, gradio) serve MoGe-3 / training / the demo app, not v2 inference.
#   bash add_moge.sh            (env found via ROOT file / NAST_TRELLIS_ROOT)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NAST_TRELLIS_ROOT:-$( [ -f "$HERE/ROOT" ] && cat "$HERE/ROOT" || echo "$HOME/nast_trellis")}"
ENV_NAME="${NAST_TRELLIS_ENV:-trellis}"
PY="$ROOT/env/bin/python"
[ -x "$PY" ] || PY="$ROOT/miniconda3/envs/$ENV_NAME/bin/python"
[ -x "$PY" ] || { echo "no TRELLIS env under $ROOT"; exit 2; }
export PYTHONNOUSERSITE=1 PIP_CACHE_DIR="$ROOT/cache/pip" TMPDIR="$ROOT/tmp"
mkdir -p "$ROOT/cache/pip" "$ROOT/tmp"
MOGE=https://github.com/microsoft/MoGe/archive/74fbce054ebed49800de42d0ad0e83495065719a.zip
U3D=https://github.com/EasternJournalist/utils3d-moge/archive/62f09d58509485564e24d5d9f6aac9ee9ebc0c37.zip
if ! "$PY" -c "import moge.model.v2, utils3d_moge" >/dev/null 2>&1; then
  "$PY" -m pip install -q --no-deps "$U3D" "$MOGE"
fi
"$PY" -c "from moge.model.v2 import MoGeModel; import utils3d_moge; print('moge ok')"
"$PY" -m pip cache purge >/dev/null 2>&1 || true
