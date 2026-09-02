# NAST Mode 3 local pipeline — one-shot Windows installer.
#   git clone ... ; cd nast-local-pipeline ; powershell -ExecutionPolicy Bypass -File install.ps1
# Creates the venv, installs deps (+ torch for the GPU), builds the WPF app
# when the .NET 8 SDK is present, and drops a desktop shortcut.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host "== python venv ==" -ForegroundColor Cyan
python -m venv venv
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\pip install -r requirements.txt

if ($env:SKIP_TORCH -ne "1") {
    Write-Host "== torch ==" -ForegroundColor Cyan
    $gpu = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($gpu) { venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128 }
    else      { venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu }
}

if (-not (Test-Path "local_gpu\models\vggt_omega_1b_512.pt")) {
    Write-Host "NOTE: put vggt_omega_1b_512.pt into local_gpu\models\ (see its README)" -ForegroundColor Yellow
}

$dotnet = Get-Command dotnet -ErrorAction SilentlyContinue
if ($dotnet) {
    Write-Host "== WPF app ==" -ForegroundColor Cyan
    dotnet publish deskview -c Release -r win-x64 --self-contained true -o app_build
}

Write-Host "== desktop shortcut ==" -ForegroundColor Cyan
$ws = New-Object -ComObject WScript.Shell
$lnk = $ws.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Desktop')) "NAST Deskview.lnk"))
$lnk.TargetPath = (Join-Path (Get-Location) "run.bat")
$lnk.WorkingDirectory = (Get-Location).Path
$lnk.IconLocation = (Join-Path (Get-Location) "assets\icon.ico")
$lnk.Save()

Write-Host "install done — use the 'NAST Deskview' desktop shortcut or run.bat" -ForegroundColor Green
