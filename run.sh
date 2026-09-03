#!/usr/bin/env bash
# NAST Deskview launcher (Linux): (re)starts the local service FROM THIS TREE
# and opens the desktop app. Safe to click any time — a stale service from an
# older install is replaced, a running one of this tree is reused.
cd "$(dirname "$0")"
PY=venv/bin/python
[ -x "$PY" ] || PY=$(command -v python3 || command -v python)

up()   { curl -s -o /dev/null --max-time 1 http://127.0.0.1:8130/api/meta; }
ours() { curl -s --max-time 1 http://127.0.0.1:8130/api/decode_status | grep -q '"total"'; }

if up && ! ours; then
  # something else answers on 8130 (an older install) -> replace it
  fuser -k 8130/tcp >/dev/null 2>&1 || pkill -f "server.py 8130" 2>/dev/null
  sleep 2
fi
if ! up; then
  nohup "$PY" -u inspector/server.py 8130 > inspector/srv.log 2> inspector/srv.err &
  for i in $(seq 1 60); do up && break; sleep 0.5; done
fi
up || { echo "service did not start — see inspector/srv.err"; exit 1; }

URL="http://localhost:8130/"
# the desktop app (Qt port of the WPF Deskview); NAST_WEB=1 forces the browser GUI
if [ "${NAST_WEB:-0}" != "1" ] && "$PY" -c "import PySide6" 2>/dev/null; then
  exec "$PY" linux_app/deskview_qt.py
fi
for B in chromium chromium-browser google-chrome google-chrome-stable; do
  if command -v "$B" >/dev/null 2>&1; then
    exec "$B" --app="$URL" --window-size=1500,950
  fi
done
xdg-open "$URL" 2>/dev/null || echo "open $URL in a browser"
