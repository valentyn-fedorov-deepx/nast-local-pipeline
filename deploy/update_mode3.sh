#!/usr/bin/env bash
# NAST Mode 3: update the code of a machine that install_mode3_clean.sh already set up. Code only: the venv, the
# generator environment, the weights, recordings, maps, jobs and accounts stay as they are.
# Put these next to this script (the kit folder on Drive has them):
#     nast_v3_clean_code.tar        code snapshot (git archive of nast-local-pipeline)
#     nast_v3_clean_code.tar.md5    its checksum
#     COMMIT                        which commit the snapshot is
#
#     bash update_mode3.sh                       # the install under ~/nast
#     NAST_ROOT=/data/nast bash update_mode3.sh  # an install on another disk
#     FORCE=1 bash update_mode3.sh               # also when a job is still running (it is stopped and lost)
#
# What it does: checks the archive, stops the app and the service (a service left running would keep serving the old
# code: run.sh reuses it), keeps a copy of the code it replaces, unpacks the new one, starts the service and checks
# that it is the new one, and makes sure there is an account to log in with (opening data needs a login).
set -e
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAST_ROOT="${NAST_ROOT:-$HOME/nast}"
APP="$NAST_ROOT/nast-local-pipeline"
TAR="$SRC/nast_v3_clean_code.tar"
PY="$APP/venv/bin/python"
URL="http://127.0.0.1:8130"
say() { printf '\n==== %s  [%s]\n' "$1" "$(date +%T)"; }
up() { curl -s -o /dev/null --max-time 2 "$URL/api/meta"; }

say "0. checks"
[ "$(id -u)" != "0" ] || { echo "run as the user who uses the app, not root"; exit 1; }
[ -d "$APP/inspector" ] && [ -x "$PY" ] || { echo "no install under $APP: run install_mode3_clean.sh first, or set NAST_ROOT"; exit 1; }
[ -f "$TAR" ] && [ -f "$TAR.md5" ] || { echo "nast_v3_clean_code.tar and its .md5 have to be next to this script"; exit 1; }
( cd "$SRC" && md5sum -c nast_v3_clean_code.tar.md5 ) || { echo "the code archive is damaged: download it again"; exit 1; }
echo "installed: $(head -1 "$APP/VERSION" 2>/dev/null || echo "no VERSION file (the code the installer put there)")"
echo "this kit:  $(head -1 "$SRC/COMMIT" 2>/dev/null || echo "no COMMIT file next to the script")"

say "1. stop the app and the service"
# anything the service started and that still runs is a job in progress (poses, map, depth, an object)
SPID="$(pgrep -f "server.py 8130" | head -1 || true)"
if [ -n "$SPID" ]; then
  KIDS="$(pgrep -P "$SPID" || true)"
  if [ -n "$KIDS" ]; then
    echo "the service has work in progress:"; ps -o pid=,etime=,args= -p $(echo $KIDS | tr ' ' ',') 2>/dev/null | cut -c1-150
    if [ "${FORCE:-0}" != "1" ]; then
      echo "wait until it is done (Jobs list in the app) and run this again, or FORCE=1 bash update_mode3.sh to stop it"
      exit 1
    fi
  fi
fi
pkill -f "linux_app/deskview_qt.py" 2>/dev/null && echo "the app window is closed" || true
fuser -k 8130/tcp >/dev/null 2>&1 || pkill -f "server.py 8130" 2>/dev/null || true
if [ "${FORCE:-0}" = "1" ]; then                     # what the service had started lives on without it
  pkill -KILL -f "$APP/local_gpu/" 2>/dev/null || true
  pkill -KILL -f "$APP/monocars/" 2>/dev/null || true
fi
for i in $(seq 1 20); do up || break; sleep 0.5; done
up && { echo "something still answers on port 8130: stop it and run this again"; exit 1; }
echo "stopped"

say "2. keep the code that is there"
BK="$NAST_ROOT/code_before_$(date +%Y%m%d_%H%M%S).tar"
LIST="$(mktemp)"
cd "$APP"
tar -tf "$TAR" | grep -v '/$' | while IFS= read -r f; do if [ -f "$f" ]; then printf '%s\n' "$f"; fi; done > "$LIST"
if [ -f VERSION ]; then echo VERSION >> "$LIST"; fi
tar -cf "$BK" -T "$LIST"
rm -f "$LIST"
echo "$BK  ($(du -h "$BK" | cut -f1), $(tar -tf "$BK" | wc -l) files)"
echo "to go back: tar -xf $BK -C $APP, then start the app again"

say "3. new code -> $APP"
tar -xf "$TAR" -C "$APP"
[ -f "$SRC/COMMIT" ] && cp "$SRC/COMMIT" "$APP/VERSION"
echo "$(tar -tf "$TAR" | grep -vc '/$') files; version: $(head -1 "$APP/VERSION" 2>/dev/null)"

say "4. start the service and check that it is the new one"
cd "$APP"
nohup "$PY" -u inspector/server.py 8130 > inspector/srv.log 2> inspector/srv.err &
ok=0; for i in $(seq 1 60); do up && { ok=1; break; }; sleep 0.5; done
[ "$ok" = "1" ] || { echo "the service did NOT start:"; tail -n 20 inspector/srv.err; echo "to go back: tar -xf $BK -C $APP"; exit 1; }
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 -X POST -H 'Content-Type: application/json' -d '{}' "$URL/api/auth/login" || true)"
case "$code" in
  400|401|503) echo "service: online, the login is there (answer $code to an empty login, as it should be)";;
  404) echo "the service that answers does not know the login: the old code is still running. Reboot and run this again."; exit 1;;
  *) echo "service: online, but the login answered $code: see inspector/srv.err";;
esac

say "5. an account to log in with"
NAMES="$("$PY" -c "import sys; sys.path.insert(0, 'inspector'); import auth; print(' '.join(sorted(auth.load_users())))" 2>/dev/null || true)"
if [ -n "$NAMES" ]; then
  echo "accounts on this machine: $NAMES"
elif [ -t 0 ]; then
  echo "There is no account yet, and opening data needs one. It is created here, on this machine;"
  echo "the password is stored as a salted hash only (inspector/users.json)."
  read -r -p "username [$USER]: " NAME; NAME="${NAME:-$USER}"
  "$PY" inspector/auth.py add-user "$NAME" || echo "no account was made. Make one before opening data:  cd $APP && venv/bin/python inspector/auth.py add-user <name>"
else
  echo "There is no account yet, and opening data needs one. Create it:"
  echo "    cd $APP && venv/bin/python inspector/auth.py add-user <name>"
fi

say "done"
echo "start: $APP/run.sh   (or the 'NAST Deskview' desktop shortcut). 'Open data folder' now asks for the login."
echo "another account, or a new password for one:  cd $APP && venv/bin/python inspector/auth.py add-user <name>"
