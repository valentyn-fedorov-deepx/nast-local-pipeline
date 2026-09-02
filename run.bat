@echo off
rem NAST Deskview launcher (Windows): service + app (or browser fallback)
cd /d "%~dp0"
set PY=venv\Scripts\python.exe
if not exist %PY% set PY=python

powershell -NoProfile -Command "try { (Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 http://127.0.0.1:8130/api/meta) | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
  start "" /min %PY% -u inspector\server.py 8130
  powershell -NoProfile -Command "foreach ($i in 1..60) { try { (Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 http://127.0.0.1:8130/api/meta) | Out-Null; exit 0 } catch { Start-Sleep -m 500 } }; exit 1"
)

if exist app_build\NastDeskview.exe (
  start "" app_build\NastDeskview.exe
) else (
  start "" http://localhost:8130/
)
