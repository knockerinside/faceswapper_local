# ==============================================================================
# Real-Time Face-Swap Application Launcher for Windows 11
# ==============================================================================
[CmdletBinding()]
param(
    [string]$Profile = "balanced",
    [switch]$Headless
)

$ErrorActionPreference = "Stop"

$CurrentScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath "$CurrentScriptDir"

$venvPython = Join-Path $CurrentScriptDir ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath "$venvPython")) {
    Write-Warning "Virtual environment not found at .venv. Running .\setup.ps1 first..."
    & ".\setup.ps1"
}

# Check model directory
$hasDetector = (Test-Path -LiteralPath "$CurrentScriptDir\models\scrfd_10g_bnkps.onnx") -or `
               (Test-Path -LiteralPath "$CurrentScriptDir\models\det_10g.onnx")

$hasSwapper = (Test-Path -LiteralPath "$CurrentScriptDir\models\inswapper_128_fp16.onnx") -or `
              (Test-Path -LiteralPath "$CurrentScriptDir\models\inswapper_128.onnx")

if (-not $hasDetector -or -not $hasSwapper) {
    Write-Host ""
    Write-Host "[!] Notice: Model files missing in models/ directory." -ForegroundColor Yellow
    Write-Host "    Expected: scrfd_10g_bnkps.onnx and inswapper_128.onnx (or FP16 variant)" -ForegroundColor Yellow
    $reply = Read-Host "    Would you like to automatically download required models now? (Y/n)"
    if ($reply -eq "" -or $reply -match "^[yY]") {
        & "$venvPython" tools\download_models.py
    }
}

Write-Host "Launching Real-Time Face-Swap with Profile: $Profile..." -ForegroundColor Cyan

$launchArgs = @("app.py", "--profile", $Profile)
if ($Headless) {
    $launchArgs += "--headless"
}

& "$venvPython" @launchArgs
