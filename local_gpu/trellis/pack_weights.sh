#!/usr/bin/env bash
# Bundle the model weights a finished TRELLIS install uses, so that unpack_trellis.sh on the target machine has nothing to
# download: TRELLIS-image-large, SAM ViT-H (safetensors only, not the .bin/.h5 twins), MoGe-2, DINOv2 (torch-hub repo +
# checkpoint). Symlinks of the HuggingFace cache are dereferenced and its blobs/ left out, so the tar holds every file once.
# Layout inside the tar = <ROOT>/cache of the install (hf/hub/..., torch/hub/...).
#   bash pack_weights.sh <out.tar> [hf_home] [torch_home] [second hf_home for models missing in the first]
#   defaults: $HF_HOME or ~/.cache/huggingface, $TORCH_HOME or ~/.cache/torch
set -e
OUT="$1"; HF="${2:-${HF_HOME:-$HOME/.cache/huggingface}}"; TH="${3:-${TORCH_HOME:-$HOME/.cache/torch}}"; HF2="${4:-}"
[ -n "$OUT" ] || { echo "usage: pack_weights.sh <out.tar> [hf_home] [torch_home] [hf_home_2]"; exit 1; }
rm -f "$OUT"; tar -cf "$OUT" -T /dev/null
add_hf() {                                    # <hf_home> <models--org--name>: refs/main + the files of the current snapshot
  local home="$1" m="$2" rev files=()
  [ -f "$home/hub/$m/refs/main" ] || return 1
  rev=$(cat "$home/hub/$m/refs/main")
  while IFS= read -r x; do files+=("${x#$home/}"); done < <(find -L "$home/hub/$m/snapshots/$rev" -type f ! -name "*.bin" ! -name "*.h5" ! -name "*.msgpack" ! -name "*.ot")
  [ "${#files[@]}" -gt 0 ] || return 1
  tar -rhf "$OUT" --transform 's,^hub/,hf/hub/,' -C "$home" "hub/$m/refs/main" "${files[@]}"
  echo "  $m @ ${rev:0:10}: ${#files[@]} files from $home"
}
for m in models--microsoft--TRELLIS-image-large models--facebook--sam-vit-huge models--Ruicheng--moge-2-vitl-normal; do
  add_hf "$HF" "$m" || { [ -n "$HF2" ] && add_hf "$HF2" "$m"; } || { echo "MISSING: $m (not in $HF${HF2:+ nor $HF2})"; exit 1; }
done
[ -d "$TH/hub/facebookresearch_dinov2_main" ] && [ -f "$TH/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth" ] || { echo "MISSING: dinov2 in $TH/hub"; exit 1; }
tar -rhf "$OUT" --transform 's,^hub/,torch/hub/,' -C "$TH" hub/facebookresearch_dinov2_main hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth
echo "  dinov2_vitl14_reg4 (torch hub) from $TH"
ls -lh "$OUT"; echo PACK_WEIGHTS_DONE
