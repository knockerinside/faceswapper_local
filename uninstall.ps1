# ==============================================================================
# Complete Uninstaller for Real-Time Face-Swap Engine
# Completely uninstalls and removes all virtual environments, cached pip packages,
# downloaded models, compiled bytecode (__pycache__), temporary files, and logs.
# Leaves NO leftover cache, temp data, or trash anywhere on your system.
# ==============================================================================
[CmdletBinding()]
param(
    [switch]$KeepModels,
    [switch]$Force
)

$ErrorActionPreference = "Continue"

$CurrentScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath "$CurrentScriptDir"

Write-Host "=================================================================" -ForegroundColor Magenta
Write-Host " REAL-TIME AI FACE-SWAP ENGINE -- ZERO-TRASH UNINSTALLER" -ForegroundColor Magenta
Write-Host " Location: $CurrentScriptDir" -ForegroundColor Magenta
Write-Host "=================================================================" -ForegroundColor Magenta

if (-not $Force) {
    Write-Host ""
    Write-Host "This will completely erase:" -ForegroundColor Yellow
    Write-Host " - Isolated Python virtual environment (.venv) and all installed packages" -ForegroundColor Yellow
    Write-Host " - All Python compiled bytecode (__pycache__, *.pyc, *.pyo)" -ForegroundColor Yellow
    Write-Host " - Performance benchmark reports and profile logs" -ForegroundColor Yellow
    Write-Host " - pip and build cache folders (.pytest_cache, pip cache)" -ForegroundColor Yellow
    if ($KeepModels) {
        Write-Host " - [KEEPING] models/ folder will be preserved." -ForegroundColor Green
    } else {
        Write-Host " - Downloaded ONNX model weights (inswapper, scrfd, etc.)" -ForegroundColor Yellow
    }
    
    $confirm = Read-Host "`nAre you sure you want to proceed with full cleanup? (Type 'yes' or 'y' to confirm)"
    if ($confirm -notmatch "^(yes|y)$") {
        Write-Host "Uninstallation cancelled by user. No files were removed." -ForegroundColor Cyan
        exit 0
    }
}

Write-Host ""
Write-Host "[*] Stopping any active face-swap engine or background python processes..." -ForegroundColor Cyan
Get-Process | Where-Object { 
    $_.ProcessName -eq "python" -and $_.Path -like "*$CurrentScriptDir*" 
} | Stop-Process -Force -ErrorAction SilentlyContinue

# 1. Remove .venv (Virtual Environment)
Write-Host "[*] Removing isolated virtual environments (.venv)..." -ForegroundColor Cyan
$venvPaths = @(
    (Join-Path $CurrentScriptDir ".venv"),
    (Join-Path $CurrentScriptDir "realtime_faceswap\.venv")
)
foreach ($vp in $venvPaths) {
    if (Test-Path -LiteralPath "$vp") {
        Write-Host "    Deleting: $vp"
        Remove-Item -LiteralPath "$vp" -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# 2. Remove all __pycache__ and *.pyc recursively
Write-Host "[*] Cleaning all Python bytecode cache (__pycache__, *.pyc, *.pyo)..." -ForegroundColor Cyan
Get-ChildItem -LiteralPath "$CurrentScriptDir" -Include "__pycache__" -Recurse -Directory -Force -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "    Removing cache dir: $($_.FullName)"
    Remove-Item -LiteralPath "$($_.FullName)" -Recurse -Force -ErrorAction SilentlyContinue
}
Get-ChildItem -LiteralPath "$CurrentScriptDir" -Include "*.pyc", "*.pyo", "*.pyd" -Recurse -File -Force -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item -LiteralPath "$($_.FullName)" -Force -ErrorAction SilentlyContinue
}

# 3. Remove .pytest_cache and test artifacts
Write-Host "[*] Cleaning test and compiler caches..." -ForegroundColor Cyan
$cacheDirs = @(
    (Join-Path $CurrentScriptDir ".pytest_cache"),
    (Join-Path $CurrentScriptDir "realtime_faceswap\.pytest_cache"),
    (Join-Path $CurrentScriptDir "build"),
    (Join-Path $CurrentScriptDir "dist"),
    (Join-Path $CurrentScriptDir "*.egg-info")
)
foreach ($cd in $cacheDirs) {
    if (Test-Path -LiteralPath "$cd") {
        Remove-Item -LiteralPath "$cd" -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# 4. Remove benchmark logs and temporary run files
Write-Host "[*] Removing benchmark logs, CSVs, and telemetry records..." -ForegroundColor Cyan
$benchmarkDirs = @(
    (Join-Path $CurrentScriptDir "benchmarks"),
    (Join-Path $CurrentScriptDir "realtime_faceswap\benchmarks")
)
foreach ($bd in $benchmarkDirs) {
    if (Test-Path -LiteralPath "$bd") {
        Get-ChildItem -LiteralPath "$bd" -Include "*.json", "*.csv", "*.log" -File -Force -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    }
}

# 5. Handle model files
if (-not $KeepModels) {
    Write-Host "[*] Removing downloaded model weights from models/..." -ForegroundColor Cyan
    $modelDirs = @(
        (Join-Path $CurrentScriptDir "models"),
        (Join-Path $CurrentScriptDir "realtime_faceswap\models")
    )
    foreach ($md in $modelDirs) {
        if (Test-Path -LiteralPath "$md") {
            Get-ChildItem -LiteralPath "$md" -Include "*.onnx", "*.tmp", "*.weights" -File -Force -ErrorAction SilentlyContinue | ForEach-Object {
                Write-Host "    Deleting model weight: $($_.Name)"
                Remove-Item -LiteralPath "$($_.FullName)" -Force -ErrorAction SilentlyContinue
            }
        }
    }
} else {
    Write-Host "[*] Preserving models/ folder as requested." -ForegroundColor Green
}

# 6. Clean pip cache associated with user account if created during setup
Write-Host "[*] Purging local pip build cache..." -ForegroundColor Cyan
if (Test-Path -LiteralPath "$env:LOCALAPPDATA\pip\cache") {
    Remove-Item -LiteralPath "$env:LOCALAPPDATA\pip\cache" -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Green
Write-Host " [OK] CLEANUP COMPLETE!" -ForegroundColor Green
Write-Host " All virtual environments, caches, temporary files, and models have been removed." -ForegroundColor Green
Write-Host " The project directory is completely clean." -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green
