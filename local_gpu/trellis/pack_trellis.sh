#!/usr/bin/env bash
# Pack a finished install (env_ok present) into ONE relocatable archive for a
# machine of the same GPU generation: the conda-pack'ed env (no conda needed on
# the target) + the TRELLIS checkout + the Real-ESRGAN weight + sample. The
# big HF/torch-hub weights are NOT included: pack_weights.sh bundles them
# separately (nast_weights.tar next to the pack = no download on the target),
# else unpack_trellis.sh downloads them. Runtime-unneeded CUDA libraries
# (torch brings its own via pip) and the .git of TRELLIS are excluded.
#   bash pack_trellis.sh <ROOT> <out.tar>
#   env NAST_PACK_TMP: scratch for env.tar.gz (~7 GB; default <ROOT>/pack - on a
#   WSL box point it at /mnt/<disk>/... so the vhdx on C: does not grow)
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
PACKDIR="${NAST_PACK_TMP:-$ROOT/pack}"
mkdir -p "$PACKDIR"; rm -f "$PACKDIR/env.tar.gz"
EXCL=""
for pat in libcublas libcublasLt libcusparse libcufft libcusolver libcurand libnpp libnvjpeg libcufile; do
  EXCL="$EXCL --exclude targets/x86_64-linux/lib/${pat}*.so* --exclude lib/${pat}*.so*"
done
EXCL="$EXCL --exclude 'targets/x86_64-linux/lib/*_static.a' --exclude 'lib/*_static.a' --exclude 'targets/x86_64-linux/lib/stubs/*'"
# profilers/debuggers/docs of the toolkit are dead weight at runtime (~1.7 GB)
for d in nsight-compute nsightee_plugins libnvvp compute-sanitizer share/doc share/man; do EXCL="$EXCL --exclude '$d/*'"; done
EXCL="$EXCL --exclude 'bin/cuda-gdb*' --exclude 'bin/nsys*' --exclude 'bin/ncu*' --exclude 'bin/nvvp'"
eval "$ENV/bin/conda-pack" -p "$ENV" -o "$PACKDIR/env.tar.gz" --ignore-missing-files --ignore-editable-packages $EXCL
cp "$ROOT/env_ok" "$PACKDIR/env_ok"
tar -cf "$OUT" --exclude='TRELLIS/.git' -C "$PACKDIR" env.tar.gz env_ok -C "$ROOT" TRELLIS weights
rm -f "$PACKDIR/env.tar.gz" "$PACKDIR/env_ok"
ls -lh "$OUT"; echo PACK_DONE
