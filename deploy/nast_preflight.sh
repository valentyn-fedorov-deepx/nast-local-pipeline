#!/usr/bin/env bash
# NAST Mode 3 — preflight for a clean Linux machine. Changes nothing; prints an inventory and writes it to
# ./nast_preflight_report.txt. Run as the user who will own the install:
#     bash nast_preflight.sh                      # machine only
#     bash nast_preflight.sh /path/to/recording   # + inventory of the new recording (raw12 frames or rig tars)
# Send the report file back.
REC="${1:-}"
OUT="$(pwd)/nast_preflight_report.txt"
exec > >(tee "$OUT") 2>&1
hr() { printf '\n==== %s\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

hr "system"
date
( . /etc/os-release 2>/dev/null && echo "distro: $PRETTY_NAME" )
echo "kernel: $(uname -r)   arch: $(uname -m)   glibc: $(ldd --version 2>/dev/null | head -1 | awk '{print $NF}')"
echo "user: $(whoami)   sudo without password: $(sudo -n true 2>/dev/null && echo yes || echo no)   home: $HOME"
echo "session: ${XDG_SESSION_TYPE:-unknown}   desktop: ${XDG_CURRENT_DESKTOP:-unknown}"
echo "secure boot: $(mokutil --sb-state 2>/dev/null || echo unknown)"

hr "cpu / memory"
lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)|Thread|Core" | sed 's/  */ /g'
free -h | sed -n 1,3p

hr "gpu"
if have nvidia-smi; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,compute_cap,power.limit --format=csv
  echo "cuda runtime reported by the driver: $(nvidia-smi | grep -i -o -m1 'CUDA[^|]*Version: *[0-9.]*')"
  echo "kernel module flavour (RTX 50xx needs the OPEN modules): $(modinfo nvidia 2>/dev/null | grep -i -m1 '^license' | sed 's/  */ /g')"
  echo "module version file: $(cat /proc/driver/nvidia/version 2>/dev/null | head -1)"
  have nvcc && nvcc --version | tail -2 | head -1 || echo "nvcc: none (fine: the installers bring their own)"
else
  echo "nvidia-smi NOT FOUND -> the NVIDIA driver is not installed"
  lspci 2>/dev/null | grep -i -E "vga|3d|nvidia"
  echo "recommended by ubuntu-drivers:"; ubuntu-drivers devices 2>/dev/null | grep -i -E "driver|model" | head -8
fi

hr "disks"
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL 2>/dev/null | grep -v loop
echo
df -hT -x tmpfs -x devtmpfs -x squashfs -x efivarfs 2>/dev/null
echo
echo "free where the install would go:"
for d in "$HOME" /data /mnt /media /opt; do [ -d "$d" ] && echo "  $d: $(df -h --output=avail "$d" | tail -1 | tr -d ' ') free"; done

hr "tools"
for t in python3 pip3 git curl wget tar ffmpeg conda rclone gcc make chromium chromium-browser google-chrome fuser; do
  if have "$t"; then printf '  %-18s %s\n' "$t" "$($t --version 2>&1 | head -1 | cut -c1-70)"; else printf '  %-18s -\n' "$t"; fi
done
python3 -c "import venv, ensurepip; print('  python venv module: ok')" 2>/dev/null || echo "  python venv module: MISSING (sudo apt install python3-venv)"
echo "  Qt WebEngine system libraries:"
for l in libxcb-cursor.so.0 libnss3.so libxkbcommon.so.0 libasound.so.2 libxcomposite.so.1 libxdamage.so.1 libxrandr.so.2 libxtst.so.6; do
  printf '    %-24s %s\n' "$l" "$(ldconfig -p 2>/dev/null | grep -q "$l" && echo ok || echo MISSING)"
done

hr "network (the installers download about 25 GB)"
for h in pypi.org download.pytorch.org huggingface.co github.com drive.google.com repo.anaconda.com nvidia-kaolin.s3.us-east-2.amazonaws.com; do
  printf '  %-52s %s\n' "$h" "$(curl -s -o /dev/null -m 8 -w 'http %{http_code} in %{time_total}s (any code = reachable)' "https://$h" || echo unreachable)"
done
echo "  download speed probe (50 MB from pytorch CDN):"
curl -s -o /dev/null -m 40 -r 0-52428799 -w '    %{speed_download} bytes/s\n' https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl || echo "    probe failed"

hr "port / leftovers"
echo "port 8130: $( (have ss && ss -ltn 2>/dev/null | grep -q ':8130 ') && echo BUSY || echo free)"
ls -d "$HOME"/nast* "$HOME"/nast-local-pipeline /opt/nast* /data/nast* 2>/dev/null | sed 's/^/  existing: /' || true

if [ -n "$REC" ]; then
  hr "recording: $REC"
  if [ ! -e "$REC" ]; then echo "path not found"; else
    echo "total size: $(du -sh "$REC" 2>/dev/null | cut -f1)"
    echo "top level (first entries):"; ls -la "$REC" | head -9
    echo "files by type:"
    find "$REC" -type f 2>/dev/null | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -12
    n=$(find "$REC" -type f -name '*.raw12' 2>/dev/null | wc -l)
    echo "raw12 frames: $n"
    if [ "$n" -gt 0 ]; then
      f=$(find "$REC" -type f -name '*.raw12' | sort | head -1); l=$(find "$REC" -type f -name '*.raw12' | sort | tail -1)
      echo "  first: $(basename "$f")  ($(stat -c %s "$f") bytes)"; echo "  last:  $(basename "$l")"
      echo "  per camera prefix:"; find "$REC" -type f -name '*.raw12' -printf '%f\n' | cut -c1-2 | sort | uniq -c
      echo "  per folder:"; find "$REC" -type f -name '*.raw12' -printf '%h\n' | sort | uniq -c | head -12
    fi
    echo "rig tars: $(find "$REC" -type f -name '*.tar' 2>/dev/null | wc -l)"; find "$REC" -type f -name '*.tar' -printf '  %s bytes  %p\n' 2>/dev/null | head -12
    echo "metadata / imu:"; find "$REC" -maxdepth 3 -type f \( -name 'session*.json' -o -name 'raw_index*.json' -o -name 'xsens*' -o -name '*.bin' -o -name 'README*' -o -name '*.csv' \) -printf '  %s bytes  %p\n' 2>/dev/null | head -15
  fi
fi

hr "verdict"
need=110
avail=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc '0-9')
echo "free under \$HOME: ${avail} GB; Mode 3 without old data needs about 45 GB installed + the same again while unpacking."
echo "plan for >= ${need} GB free, plus 10 MB per frame of every recording (7.4 MB raw + 2.7 MB decoded)."
[ "${avail:-0}" -ge "$need" ] && echo "DISK: ok" || echo "DISK: tight -> pick another mount from the table above and tell the installer NAST_ROOT=<that path>"
have nvidia-smi && echo "GPU driver: present" || echo "GPU driver: MISSING -> install nvidia-driver-570-open or newer first (RTX 5090 = Blackwell, open kernel modules only)"
echo
echo "report saved: $OUT"
