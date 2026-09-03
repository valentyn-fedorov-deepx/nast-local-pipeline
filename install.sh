#!/usr/bin/env bash
# NAST Mode 3 local pipeline — one-shot Linux installer.
#   git clone ... && cd nast-local-pipeline && bash install.sh
# Creates the venv, installs every dependency (torch matched to your GPU),
# pulls the VGGT weights when tex1 is reachable, and drops a desktop
# shortcut that starts the service and opens the GUI.
#   SKIP_TORCH=1 bash install.sh    # CPU-only install (no map rebuild)
set -e
cd "$(dirname "$0")"

echo "== python venv =="
PYBIN=$(command -v python3 || command -v python)
[ -n "$PYBIN" ] || { echo "python3 not found — install python 3.10+ first"; exit 1; }
"$PYBIN" -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
venv/bin/pip install PySide6

if [ "${SKIP_TORCH:-0}" != "1" ]; then
  echo "== torch =="
  if command -v nvidia-smi >/dev/null 2>&1; then
    venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
  else
    echo "no NVIDIA GPU detected — installing CPU torch (the VGGT map rebuild needs a GPU)"
    venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
fi

if [ ! -f local_gpu/models/vggt_omega_1b_512.pt ]; then
  echo "== VGGT weights =="
  if ssh -o BatchMode=yes -o ConnectTimeout=5 tex1 true 2>/dev/null; then
    echo "fetching vggt_omega_1b_512.pt from tex1 (4.6 GB)…"
    scp tex1:/nvme0n1-disk/valentyn.fedorov/vggt-omega/checkpoints/vggt_omega_1b_512.pt local_gpu/models/
  else
    echo "NOTE: put vggt_omega_1b_512.pt into local_gpu/models/ (see local_gpu/models/README.md)"
  fi
fi

chmod +x run.sh

echo "== desktop shortcut =="
APPDIR="$HOME/.local/share/applications"
mkdir -p "$APPDIR"
cat > "$APPDIR/nast-deskview.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=NAST Deskview
Comment=NAST Mode 3 — local polarization pipeline
Exec=$(pwd)/run.sh
Icon=$(pwd)/assets/icon.png
Terminal=false
Categories=Graphics;Science;
EOF
chmod +x "$APPDIR/nast-deskview.desktop"
if [ -d "$HOME/Desktop" ]; then
  cp "$APPDIR/nast-deskview.desktop" "$HOME/Desktop/" && chmod +x "$HOME/Desktop/nast-deskview.desktop"
  command -v gio >/dev/null 2>&1 && \
    gio set "$HOME/Desktop/nast-deskview.desktop" metadata::trusted true 2>/dev/null || true
fi

echo
echo "install done — launch with ./run.sh or the 'NAST Deskview' shortcut."
echo "data trees (viewer/scenes, street_video) are copied separately — see README."
