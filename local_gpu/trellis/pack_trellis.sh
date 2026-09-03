#!/usr/bin/env bash
# Pack a finished install (env_ok present) into ONE relocatable archive for a
# machine of the same GPU generation: conda-pack'ed env (no conda needed on the
# target) + TRELLIS checkout + weights caches + sample. Run on the build machine:
#   bash pack_trellis.sh <ROOT> <out.tar>
set -e
ROOT="$1"; OUT="$2"
[ -f "$ROOT/env_ok" ] || { echo "no env_ok in $ROOT — finish install_trellis.sh first"; exit 1; }
source "$ROOT/miniconda3/etc/profile.d/conda.sh"
export CONDARC="$ROOT/condarc" CONDA_ENVS_PATH="$ROOT/miniconda3/envs"
ENV="$ROOT/miniconda3/envs/trellis"
# the linker/JIT need a REAL libcudart.so inside the env (symlinks may point outside)
for n in libcudart.so libcudart.so.12; do
  t=$(readlink -e "$ENV/lib/$n" 2>/dev/null || true)
  if [ -n "$t" ] && [ "$t" != "$ENV/lib/$n" ]; then rm -f "$ENV/lib/$n"; cp "$t" "$ENV/lib/$n"; fi
done
conda activate trellis
"$ENV/bin/python" -m pip install -q conda-pack
mkdir -p "$ROOT/pack"; rm -f "$ROOT/pack/env.tar.gz"
"$ENV/bin/conda-pack" -p "$ENV" -o "$ROOT/pack/env.tar.gz" --ignore-missing-files --ignore-editable-packages
cp "$ROOT/env_ok" "$ROOT/pack/env_ok"
cd "$ROOT"
tar -cf "$OUT" -C "$ROOT/pack" env.tar.gz env_ok \
    -C "$ROOT" TRELLIS weights cache/hf cache/torch
ls -lh "$OUT"; echo PACK_DONE
