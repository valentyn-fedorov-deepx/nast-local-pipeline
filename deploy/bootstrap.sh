#!/usr/bin/env bash
# One command on the target laptop: pull the kit from Google Drive, verify every file, run the installer.
#   bash bootstrap.sh [remote:path] [local dir]        defaults: gdrive:nast_demo/mode3_5090   ~/mode3_5090
# Needs rclone with a configured Drive remote (rclone config: name gdrive, type drive, log in once in the browser).
# Safe to run again: rclone skips what is already downloaded and complete, md5 catches a broken file.
set -e
REMOTE="${1:-gdrive:nast_demo/mode3_5090}"; DST="${2:-$HOME/mode3_5090}"
RNAME="${REMOTE%%:*}"
command -v rclone >/dev/null 2>&1 || { echo "rclone is missing. Install it:  curl -fsSL https://rclone.org/install.sh | sudo bash   then:  rclone config"; exit 1; }
rclone listremotes 2>/dev/null | grep -qx "$RNAME:" || { echo "rclone has no remote '$RNAME'. Run:  rclone config create $RNAME drive scope=drive.readonly root_folder_id=1hjRnP0IO_CxT26AnNiadzR_yGDpOj23u   (log in in the browser, wait for the prompt)"; exit 1; }
rclone lsf "$REMOTE" >/dev/null 2>&1 || { echo "cannot list $REMOTE: log in with the account that has nast_demo (rclone config reconnect $RNAME:)"; exit 1; }
mkdir -p "$DST"; cd "$DST"
echo "== 1/3 download $REMOTE -> $DST  (19 GB)"
rclone copy "$REMOTE" "$DST" -P --transfers 4 --checkers 8 --drive-chunk-size 64M
echo "== 2/3 verify"
md5sum -c manifest.md5 || { echo "a file is damaged: run this script again, rclone re-downloads only what is broken after you delete that file"; exit 1; }
echo "== 3/3 install (15-30 min, asks the sudo password for apt)"
bash install_mode3_clean.sh
