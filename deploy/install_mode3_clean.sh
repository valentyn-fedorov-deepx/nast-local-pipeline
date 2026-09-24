#!/usr/bin/env bash
# NAST Mode 3 — clean install on a Linux machine with an RTX 50xx (Blackwell) GPU. No old data: code + weights + envs only.
# Put these files into one folder and run this script from it (as the user who will use the app, not root):
#     nast_v3_clean_code.tar            code snapshot (git archive of nast-local-pipeline HEAD)
#     vggt_omega_1b_512.pt              VGGT-Omega weights, 4.4 GB (map build)
#     nast_trellis_pack_blackwell.tar   relocatable TRELLIS env for sm_120 (objects), optional: without it objects = map points only
#     nast_weights.tar                  TRELLIS / SAM / MoGe-2 / DINOv2 weights (optional: without it they are downloaded, ~8 GB)
#
#     bash install_mode3_clean.sh                 # everything under ~/nast
#     NAST_ROOT=/data/nast bash install_mode3_clean.sh   # another disk
# Needs: the NVIDIA driver (nvidia-smi works; RTX 50xx = open kernel modules, 570+), internet (pip; the model weights too unless nast_weights.tar is here).
set -e
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAST_ROOT="${NAST_ROOT:-$HOME/nast}"
APP="$NAST_ROOT/nast-local-pipeline"
TROOT="$NAST_ROOT/nast_trellis"
LOG="$SRC/install_mode3_clean.log"
exec > >(tee -a "$LOG") 2>&1
say() { printf '\n==== %s  [%s]\n' "$1" "$(date +%T)"; }

say "0. checks"
[ "$(id -u)" != "0" ] || { echo "run as the normal user, not root"; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi not found: install the NVIDIA driver first (nvidia-driver-570-open or newer), reboot, rerun"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
[ -f "$SRC/nast_v3_clean_code.tar" ] || { echo "nast_v3_clean_code.tar is not next to this script"; exit 1; }
[ -f "$SRC/vggt_omega_1b_512.pt" ] || echo "NOTE: vggt_omega_1b_512.pt is not here -> the map build will not work until it is copied to $APP/local_gpu/models/"
mkdir -p "$NAST_ROOT"
need=70; avail=$(df --output=avail -BG "$NAST_ROOT" | tail -1 | tr -dc '0-9')
echo "free under $NAST_ROOT: ${avail} GB (need about ${need} GB for the install itself)"
[ "${avail:-0}" -ge "$need" ] || { echo "not enough space: rerun with NAST_ROOT=<a bigger mount>"; exit 1; }

say "1. system packages (python venv, curl, Qt WebEngine libraries)"
PKGS="python3 python3-venv python3-pip curl tar libxcb-cursor0 libnss3 libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 libxtst6 libegl1 libopengl0"
if command -v apt-get >/dev/null 2>&1; then
  if sudo -n true 2>/dev/null || [ -t 0 ]; then
    sudo apt-get update -y && sudo apt-get install -y $PKGS && (sudo apt-get install -y libasound2t64 2>/dev/null || sudo apt-get install -y libasound2 || true)
  else
    echo "no sudo in this session -> run by hand if the app window fails to open:  sudo apt-get install -y $PKGS"
  fi
fi

say "2. code -> $APP"
mkdir -p "$APP"
tar -xf "$SRC/nast_v3_clean_code.tar" -C "$APP"
mkdir -p "$APP/local_gpu/models"
if [ -f "$SRC/vggt_omega_1b_512.pt" ] && [ ! -f "$APP/local_gpu/models/vggt_omega_1b_512.pt" ]; then
  cp "$SRC/vggt_omega_1b_512.pt" "$APP/local_gpu/models/"
fi
ls -la --block-size=M "$APP/local_gpu/models" | awk '{print $5, $9}'

say "3. python venv + torch (cu128 wheels carry sm_120) + desktop shortcut"
( cd "$APP" && bash install.sh )
"$APP/venv/bin/python" - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "| cc", torch.cuda.get_device_capability(0), "| archs", torch.cuda.get_arch_list()[-2:])
    x = torch.randn(2048, 2048, device="cuda"); print("matmul ok:", float((x @ x).abs().mean()) > 0)
EOF

say "4. TRELLIS env (objects: SAM + Real-ESRGAN + TRELLIS) + MoGe-2 (depth of new recordings) -> $TROOT"
if [ -f "$SRC/nast_trellis_pack_blackwell.tar" ]; then
  if [ -f "$SRC/nast_trellis_pack_blackwell.tar.md5" ]; then
    echo "checking the pack against its md5 (about a minute)…"
    ( cd "$SRC" && md5sum -c nast_trellis_pack_blackwell.tar.md5 ) || { echo "the pack is damaged: download it again"; exit 1; }
  fi
  if [ -f "$TROOT/env_ok" ] && [ -x "$TROOT/env/bin/python" ]; then
    echo "already unpacked: $TROOT (delete it to redo)"; echo "$TROOT" > "$APP/local_gpu/trellis/ROOT"
  else
    bash "$APP/local_gpu/trellis/unpack_trellis.sh" "$SRC/nast_trellis_pack_blackwell.tar" "$TROOT"
  fi
else
  echo "no nast_trellis_pack_blackwell.tar here -> skipped. Objects will be cut from the map points (no generative mesh)."
  echo "Alternative without the pack (40-60 min, compiles on this machine):  NAST_TRELLIS_ROOT=$TROOT bash $APP/local_gpu/trellis/install_trellis.sh"
fi

say "5. service boot check (no data yet: an empty scene is expected)"
cd "$APP"
( venv/bin/python -u inspector/server.py 8130 > inspector/srv.log 2> inspector/srv.err & echo $! > /tmp/nast_srv.pid )
ok=0; for i in $(seq 1 60); do curl -s -o /dev/null --max-time 1 http://127.0.0.1:8130/api/meta && { ok=1; break; }; sleep 0.5; done
if [ "$ok" = "1" ]; then
  echo "service: online"; curl -s --max-time 3 http://127.0.0.1:8130/api/dataset | cut -c1-300; echo
  grep -h -E "^\[scene\]|moge python" inspector/srv.log | head -4
else
  echo "service did NOT start:"; tail -n 20 inspector/srv.err
fi
kill "$(cat /tmp/nast_srv.pid)" 2>/dev/null || true

say "done"
du -sh "$APP" "$TROOT" 2>/dev/null
echo "start: $APP/run.sh   (or the 'NAST Deskview' desktop shortcut)"
echo "new recording: header button 'Open data folder' -> the folder with the .raw12 frames of BOTH cameras (A_*, B_*; decode + depth start by themselves),"
echo "               then MAP tab -> 'poses' (poses + depth_geo, about 25 s per 100 frames), then 'build map' (A+B) and objects as usual;"
echo "               the job text says which camera looks forward / backward. Objects: SAM + Real-ESRGAN + TRELLIS, log in inspector/jobs/job_<id>/trellis.log"
echo "log: $LOG"
