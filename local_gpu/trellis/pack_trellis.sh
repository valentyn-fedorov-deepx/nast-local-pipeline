#!/usr/bin/env bash
# Pack a finished install (env_ok present) into ONE relocatable archive for a
# machine of the same GPU generation: the conda-pack'ed env (no conda needed on
# the target) + the TRELLIS checkout + the Real-ESRGAN weight + sample. The
# big HF/torch-hub weights are NOT included — unpack_trellis.sh downloads them
# on the target (a few GB from HuggingFace is faster than hauling them around).
# Runtime-unneeded CUDA libraries (torch brings its own via pip) are excluded.
#   bash pack_trellis.sh <ROOT> <out.tar>
set -e
ROOT="$1"; OUT="$2"
[ -f "$ROOT/env_ok" ] || { echo "no env_ok in $ROOT — finish install_trellis.sh first"; exit 1; }
ENV="$ROOT/miniconda3/envs/trellis"
# the linker/JIT need a REAL libcudart.so inside the env (a symlink may point outside)
for n in libcudart.so libcudart.so.12; do
  t=$(readlink -e "$ENV/lib/$n" 2>/dev/null || true)
  if [ -n "$t" ] && [ "$t" != "$ENV/lib/$n" ]; then rm -f "$ENV/lib/$n"; cp "$t" "$ENV/lib/$n"; fi
done
"$ENV/bin/python" -m pip install -q conda-pack
mkdir -p "$ROOT/pack"; rm -f "$ROOT/pack/env.tar.gz"
EXCL=""
for pat in libcublas libcublasLt libcusparse libcufft libcusolver libcurand libnpp libnvjpeg libcufile; do
  EXCL="$EXCL --exclude targets/x86_64-linux/lib/${pat}*.so* --exclude lib/${pat}*.so*"
done
EXCL="$EXCL --exclude 'targets/x86_64-linux/lib/*_static.a' --exclude 'lib/*_static.a' --exclude 'targets/x86_64-linux/lib/stubs/*'"
# profilers/debuggers/docs of the toolkit are dead weight at runtime (~1.7 GB)
for d in nsight-compute nsightee_plugins libnvvp compute-sanitizer share/doc share/man; do EXCL="$EXCL --exclude '$d/*'"; done
EXCL="$EXCL --exclude 'bin/cuda-gdb*' --exclude 'bin/nsys*' --exclude 'bin/ncu*' --exclude 'bin/nvvp'"
eval "$ENV/bin/conda-pack" -p "$ENV" -o "$ROOT/pack/env.tar.gz" --ignore-missing-files --ignore-editable-packages $EXCL
cp "$ROOT/env_ok" "$ROOT/pack/env_ok"
tar -cf "$OUT" -C "$ROOT/pack" env.tar.gz env_ok -C "$ROOT" TRELLIS weights
ls -lh "$OUT"; echo PACK_DONE
