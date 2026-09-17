# ==============================================================================
# Real-Time Face-Swap Application Launcher for Windows 11
# Works from root repository folder or realtime_faceswap subfolder
# ==============================================================================
[CmdletBinding()]
param(
    [string]$Profile = "balanced",
    [string]$VenvPath = "",
    [switch]$Headless
)

$ErrorActionPreference = "Stop"

$CurrentScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location -LiteralPath "$CurrentScriptDir"

# Determine environment paths
$venvPython = $null

if ($VenvPath -and (Test-Path -LiteralPath (Join-Path $VenvPath "Scripts\python.exe"))) {
    $venvPython = Join-Path $VenvPath "Scripts\python.exe"
} elseif ($VenvPath -and (Test-Path -LiteralPath (Join-Path $VenvPath "python.exe"))) {
    $venvPython = Join-Path $VenvPath "python.exe"
} elseif ($env:VIRTUAL_ENV -and (Test-Path -LiteralPath (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))) {
    $venvPython = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
} elseif (Test-Path -LiteralPath (Join-Path $CurrentScriptDir ".venv\Scripts\python.exe")) {
    $venvPython = Join-Path $CurrentScriptDir ".venv\Scripts\python.exe"
} elseif (Test-Path -LiteralPath (Join-Path $CurrentScriptDir "venv\Scripts\python.exe")) {
    $venvPython = Join-Path $CurrentScriptDir "venv\Scripts\python.exe"
} elseif (Test-Path -LiteralPath (Join-Path $CurrentScriptDir "env\Scripts\python.exe")) {
    $venvPython = Join-Path $CurrentScriptDir "env\Scripts\python.exe"
}

if (-not $venvPython -or -not (Test-Path -LiteralPath "$venvPython")) {
    Write-Warning "Virtual environment not found. Running .\setup.ps1 first..."
    & ".\setup.ps1"
    $venvPython = Join-Path $CurrentScriptDir ".venv\Scripts\python.exe"
}

# Verify that dependencies (numpy, cv2, onnxruntime) are actually installed
$hasNumpy = $false
try {
    & "$venvPython" -c "import numpy, cv2, onnxruntime" 2>$null
    if ($LASTEXITCODE -eq 0) { $hasNumpy = $true }
} catch {}

if (-not $hasNumpy) {
    Write-Host "[!] Core dependencies (numpy, cv2, onnxruntime) are missing or incomplete in environment." -ForegroundColor Yellow
    Write-Host "[*] Installing required packages into $venvPython..." -ForegroundColor Cyan
    & "$venvPython" -m pip install -r "requirements.txt"
}


# Identify model files (allow both FP32 and FP16, SCRFD and det_10g)
$hasDetector = (Test-Path -LiteralPath "$CurrentScriptDir\models\scrfd_10g_bnkps.onnx") -or `
               (Test-Path -LiteralPath "$CurrentScriptDir\models\det_10g.onnx") -or `
               (Test-Path -LiteralPath "$CurrentScriptDir\realtime_faceswap\models\scrfd_10g_bnkps.onnx") -or `
               (Test-Path -LiteralPath "$CurrentScriptDir\realtime_faceswap\models\det_10g.onnx")

$hasSwapper = (Test-Path -LiteralPath "$CurrentScriptDir\models\inswapper_128_fp16.onnx") -or `
              (Test-Path -LiteralPath "$CurrentScriptDir\models\inswapper_128.onnx") -or `
              (Test-Path -LiteralPath "$CurrentScriptDir\realtime_faceswap\models\inswapper_128_fp16.onnx") -or `
              (Test-Path -LiteralPath "$CurrentScriptDir\realtime_faceswap\models\inswapper_128.onnx")

if (-not $hasDetector -or -not $hasSwapper) {
    Write-Host ""
    Write-Host "[!] Notice: Model files missing in models/ directory." -ForegroundColor Yellow
    Write-Host "    Expected: scrfd_10g_bnkps.onnx / det_10g.onnx and inswapper_128_fp16.onnx / inswapper_128.onnx" -ForegroundColor Yellow
    $reply = Read-Host "    Would you like to automatically download required models now? (Y/n)"
    if ($reply -eq "" -or $reply -match "^[yY]") {
        $dlScript = Join-Path $CurrentScriptDir "tools\download_models.py"
        if (-not (Test-Path -LiteralPath "$dlScript")) {
            $dlScript = Join-Path $CurrentScriptDir "realtime_faceswap\tools\download_models.py"
        }
        & "$venvPython" "$dlScript"
    }
}

# Determine entry point app.py
$appPy = Join-Path $CurrentScriptDir "app.py"
if (-not (Test-Path -LiteralPath "$appPy")) {
    $appPy = Join-Path $CurrentScriptDir "realtime_faceswap\app.py"
}

Write-Host "Launching Real-Time Face-Swap with Profile: $Profile..." -ForegroundColor Cyan

$env:PYTHONPATH = "$CurrentScriptDir;$CurrentScriptDir\realtime_faceswap;$env:PYTHONPATH"

$launchArgs = @("$appPy", "--profile", $Profile)
if ($Headless) {
    $launchArgs += "--headless"
}

try {
    & "$venvPython" @launchArgs
} catch {
    Write-Host ""
    Write-Host "[ERROR] Execution failed: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
}

