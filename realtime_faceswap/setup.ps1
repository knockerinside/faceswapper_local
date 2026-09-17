# ==============================================================================
# Real-Time Face-Swap All-in-One Automated Setup Script for Windows 11
# ==============================================================================
[CmdletBinding()]
param(
    [switch]$ForceRedownloadModels,
    [switch]$SkipDiagnostics
)

$ErrorActionPreference = "Stop"

$CurrentScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath "$CurrentScriptDir"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " REAL-TIME AI FACE-SWAP ENGINE -- AUTOMATED SETUP" -ForegroundColor Cyan
Write-Host " Hardware: Windows 11, NVIDIA RTX 4050 Laptop GPU (6GB VRAM)" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Visual C++ check
Write-Host ""
Write-Host "[1/7] Checking Visual C++ 2015-2022 Redistributable..." -ForegroundColor Yellow
$vcInstalled = (Get-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" -ErrorAction SilentlyContinue) -or `
               (Get-ItemProperty -Path "HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" -ErrorAction SilentlyContinue)

if ($vcInstalled) {
    Write-Host " [OK] Visual C++ x64 runtime is installed." -ForegroundColor Green
} else {
    Write-Host " [!] Downloading official Visual C++ x64 installer..." -ForegroundColor Cyan
    $vcInstallerPath = Join-Path $env:TEMP "vc_redist.x64.exe"
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile $vcInstallerPath -UseBasicParsing
        Start-Process -FilePath $vcInstallerPath -ArgumentList "/install", "/quiet", "/norestart" -Wait
        Remove-Item -LiteralPath $vcInstallerPath -Force -ErrorAction SilentlyContinue
        Write-Host " [OK] Visual C++ installed successfully." -ForegroundColor Green
    } catch {
        Write-Warning " Automatic VC++ installation was skipped."
    }
}

# 2. Python validation
Write-Host ""
Write-Host "[2/7] Validating Python environment..." -ForegroundColor Yellow
try {
    $pythonVer = python --version 2>&1
    Write-Host " [OK] Found: $pythonVer" -ForegroundColor Green
} catch {
    Write-Host " [ERROR] Python 3.10 or 3.11 is not found in PATH." -ForegroundColor Red
    Write-Host "         Please install Python 3.10/3.11 from python.org with 'Add to PATH' checked." -ForegroundColor Red
    exit 1
}

# 3. GPU check
Write-Host ""
Write-Host "[3/7] Checking NVIDIA GPU..." -ForegroundColor Yellow
try {
    $smiOut = nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1
    Write-Host " [OK] Detected GPU: $smiOut" -ForegroundColor Green
} catch {
    Write-Warning " nvidia-smi not found. Ensure NVIDIA drivers >= 535.xx are installed."
}

# 4. Virtual Environment
Write-Host ""
Write-Host "[4/7] Setting up isolated Python virtual environment (.venv)..." -ForegroundColor Yellow
$venvDir = Join-Path $CurrentScriptDir ".venv"
if (-not (Test-Path -LiteralPath $venvDir)) {
    python -m venv "$venvDir"
    Write-Host " [OK] Virtual environment created at $venvDir" -ForegroundColor Green
} else {
    Write-Host " [OK] Existing virtual environment found." -ForegroundColor Green
}

$venvPython = Join-Path $venvDir "Scripts\python.exe"
$venvPip = Join-Path $venvDir "Scripts\pip.exe"

# 5. Dependencies
Write-Host ""
Write-Host "[5/7] Installing dependencies and CUDA ONNX Runtime..." -ForegroundColor Yellow
& "$venvPython" -m pip install --upgrade pip setuptools wheel --quiet
& "$venvPip" install -r requirements.txt

# 6. Models
Write-Host ""
Write-Host "[6/7] Checking and downloading ONNX models..." -ForegroundColor Yellow
$modelsDir = Join-Path $CurrentScriptDir "models"
if (-not (Test-Path -LiteralPath $modelsDir)) { New-Item -ItemType Directory -Path "$modelsDir" -Force | Out-Null }
$sourceDir = Join-Path $modelsDir "source"
if (-not (Test-Path -LiteralPath $sourceDir)) { New-Item -ItemType Directory -Path "$sourceDir" -Force | Out-Null }

$dlScript = Join-Path $CurrentScriptDir "tools\download_models.py"
& "$venvPython" "$dlScript"

# 7. Diagnostics
if (-not $SkipDiagnostics) {
    Write-Host ""
    Write-Host "[7/7] Verifying installation with diagnostics..." -ForegroundColor Yellow
    $diagScript = Join-Path $CurrentScriptDir "tools\diagnostics.py"
    & "$venvPython" "$diagScript"
}

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Green
Write-Host " [OK] SETUP COMPLETE!" -ForegroundColor Green
Write-Host " To start the engine: .\run.ps1" -ForegroundColor Cyan
Write-Host " To uninstall cleanly: .\uninstall.ps1" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Green
