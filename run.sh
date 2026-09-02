#!/usr/bin/env bash
# NAST Deskview launcher (Linux): starts the local service if it is not up
# and opens the GUI as an app window (chromium/chrome) or the default browser.
cd "$(dirname "$0")"
PY=venv/bin/python
[ -x "$PY" ] || PY=$(command -v python3 || command -v python)

up() { curl -s -o /dev/null --max-time 1 http://127.0.0.1:8130/api/meta; }

if ! up; then
  nohup "$PY" -u inspector/server.py 8130 > inspector/srv.log 2> inspector/srv.err &
  for i in $(seq 1 60); do up && break; sleep 0.5; done
fi
up || { echo "service did not start — see inspector/srv.err"; exit 1; }

URL="http://localhost:8130/"
for B in chromium chromium-browser google-chrome google-chrome-stable; do
  if command -v "$B" >/dev/null 2>&1; then
    exec "$B" --app="$URL" --window-size=1500,950
  fi
done
xdg-open "$URL" 2>/dev/null || echo "open $URL in a browser"
