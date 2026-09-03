#!/usr/bin/env bash
# Install a pack_trellis.sh archive on this machine — no compiler, no conda,
# no downloads. Usage:  bash unpack_trellis.sh <nast_trellis_pack.tar> <ROOT>
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARC="$1"; ROOT="$2"
[ -f "$ARC" ] || { echo "no archive $ARC"; exit 1; }
mkdir -p "$ROOT/env" "$ROOT/cache"
echo "--- extracting $(du -h "$ARC" | cut -f1) into $ROOT"
tar -xf "$ARC" -C "$ROOT"
tar -xzf "$ROOT/env.tar.gz" -C "$ROOT/env" && rm -f "$ROOT/env.tar.gz"
"$ROOT/env/bin/conda-unpack"                      # rewrite the prefixes for this path
echo "$ROOT" > "$HERE/ROOT"
echo "--- smoke: SAM + Real-ESRGAN + TRELLIS on this GPU"
rm -rf "$ROOT/smoke"; mkdir -p "$ROOT/smoke/crops"; cp "$HERE/sample_crop.png" "$ROOT/smoke/crops/roi_00.png"
NAST_TRELLIS_ROOT="$ROOT" bash "$HERE/trellis_local.sh" "$ROOT/smoke"
[ -s "$ROOT/smoke/asset.ply" ] || { echo "smoke produced no asset.ply"; exit 1; }
echo "UNPACK_DONE -> $ROOT/env_ok"
