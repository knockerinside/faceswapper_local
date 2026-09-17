# ==============================================================================
# Real-Time Face-Swap All-in-One Automated Setup Script for Windows 11
# Target: Windows 11 64-bit, NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)
# Self-contained: Downloads and configures all dependencies, tools, and models
# ==============================================================================
[CmdletBinding()]
param(
    [string]$VenvPath = "",
    [switch]$ForceRedownloadModels,
    [switch]$SkipDiagnostics
)

$ErrorActionPreference = "Stop"

# Ensure execution location is normalized and robust to spaces / special characters
$CurrentScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location -LiteralPath "$CurrentScriptDir"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " REAL-TIME AI FACE-SWAP ENGINE -- ALL-IN-ONE AUTOMATED SETUP" -ForegroundColor Cyan
Write-Host " Hardware: Windows 11, NVIDIA RTX 4050 Laptop GPU (6GB VRAM)" -ForegroundColor Cyan
Write-Host " Working Directory: $CurrentScriptDir" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# ------------------------------------------------------------------------------
# STEP 1: Validate or Install Microsoft Visual C++ Redistributable
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[1/7] Checking Microsoft Visual C++ 2015-2022 Redistributable..." -ForegroundColor Yellow
$vcInstalled = (Get-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" -ErrorAction SilentlyContinue) -or `
               (Get-ItemProperty -Path "HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" -ErrorAction SilentlyContinue)

if ($vcInstalled) {
    Write-Host " [OK] Visual C++ x64 Runtime is detected." -ForegroundColor Green
} else {
    Write-Host " [!] Visual C++ runtime not detected. Downloading official Microsoft installer..." -ForegroundColor Cyan
    $vcInstallerPath = Join-Path $env:TEMP "vc_redist.x64.exe"
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile $vcInstallerPath -UseBasicParsing
        Write-Host " Installing Visual C++ runtime silently..." -ForegroundColor Cyan
        Start-Process -FilePath $vcInstallerPath -ArgumentList "/install", "/quiet", "/norestart" -Wait
        Remove-Item -LiteralPath $vcInstallerPath -Force -ErrorAction SilentlyContinue
        Write-Host " [OK] Visual C++ installed." -ForegroundColor Green
    } catch {
        Write-Warning " Automatic VC++ installation was skipped. Please verify vc_redist.x64 is installed if DLL errors occur."
    }
}

# ------------------------------------------------------------------------------
# STEP 2: Validate Python 3.10, 3.11, 3.12, or 3.13 64-bit
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[2/7] Validating Python environment..." -ForegroundColor Yellow
$pythonExe = "python"
try {
    $pythonVer = & $pythonExe --version 2>&1
    Write-Host " [OK] Found system Python: $pythonVer" -ForegroundColor Green
} catch {
    $foundPy = $false
    if (Get-Command "py" -ErrorAction SilentlyContinue) {
        foreach ($v in @("3.13", "3.12", "3.11", "3.10")) {
            try {
                $pythonVer = & py -$v --version 2>&1
                if ($LASTEXITCODE -eq 0) {
                    $pythonExe = "py -$v"
                    Write-Host " [OK] Found py -$v: $pythonVer" -ForegroundColor Green
                    $foundPy = $true
                    break
                }
            } catch {}
        }
    }
    if (-not $foundPy) {
        Write-Host " [ERROR] Python 3.10, 3.11, 3.12, or 3.13 is not found in PATH." -ForegroundColor Red
        Write-Host "         Please install Python 64-bit from: https://www.python.org/downloads/windows/" -ForegroundColor Red
        Write-Host "         IMPORTANT: Make sure you check 'Add python.exe to PATH' during installation!" -ForegroundColor Yellow
        exit 1
    }
}

# ------------------------------------------------------------------------------
# STEP 3: Detect NVIDIA GPU, Drivers and Tensor Cores
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[3/7] Inspecting NVIDIA GPU and Driver status..." -ForegroundColor Yellow
try {
    $gpuName = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
    Write-Host " [OK] Detected GPU: $gpuName" -ForegroundColor Green
} catch {
    Write-Warning " nvidia-smi utility was not found. If this is an RTX 4050 laptop, ensure NVIDIA Game Ready or Studio Driver (>= 535.xx) is installed."
}

# ------------------------------------------------------------------------------
# STEP 4: Setup or Configure Target Python Virtual Environment
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[4/7] Preparing Python virtual environment..." -ForegroundColor Yellow

$venvPython = $null
$venvPip = $null

if ($VenvPath -and (Test-Path -LiteralPath (Join-Path $VenvPath "Scripts\python.exe"))) {
    $venvPython = Join-Path $VenvPath "Scripts\python.exe"
    $venvPip = Join-Path $VenvPath "Scripts\pip.exe"
    Write-Host " [OK] Using specified virtual environment: $VenvPath" -ForegroundColor Green
} elseif ($env:VIRTUAL_ENV -and (Test-Path -LiteralPath (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))) {
    $venvPython = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
    $venvPip = Join-Path $env:VIRTUAL_ENV "Scripts\pip.exe"
    Write-Host " [OK] Using active virtual environment: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    $venvDir = Join-Path $CurrentScriptDir ".venv"
    if (-not (Test-Path -LiteralPath $venvDir)) {
        Write-Host " Creating clean virtual environment in $venvDir..." -ForegroundColor Cyan
        if ($pythonExe -like "py *") {
            $pyArgs = $pythonExe.Split(" ")
            & $pyArgs[0] $pyArgs[1] -m venv "$venvDir"
        } else {
            & $pythonExe -m venv "$venvDir"
        }
        Write-Host " [OK] Virtual environment created." -ForegroundColor Green
    } else {
        Write-Host " [OK] Existing virtual environment found at $venvDir" -ForegroundColor Green
    }
    $venvPython = Join-Path $venvDir "Scripts\python.exe"
    $venvPip = Join-Path $venvDir "Scripts\pip.exe"
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Error " Virtual environment Python executable not found at $venvPython!"
    exit 1
}

# ------------------------------------------------------------------------------
# STEP 5: Upgrade Pip and Install Engine Dependencies
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[5/7] Installing CUDA ONNX Runtime and Python Packages..." -ForegroundColor Yellow
& "$venvPython" -m pip install --upgrade pip setuptools wheel --quiet

$reqPath = Join-Path $CurrentScriptDir "requirements.txt"
if (-not (Test-Path -LiteralPath $reqPath)) {
    $reqPath = Join-Path $CurrentScriptDir "realtime_faceswap\requirements.txt"
}
& "$venvPip" install -r "$reqPath"

Write-Host " Validating core libraries (numpy, OpenCV, ONNX Runtime)..." -ForegroundColor Cyan
try {
    & "$venvPython" -c "import numpy, cv2, onnxruntime; print(' [OK] Verified: numpy ' + numpy.__version__ + ' | onnxruntime ' + onnxruntime.__version__)"
} catch {
    Write-Host " [!] Retrying package installation with binary wheels..." -ForegroundColor Yellow
    & "$venvPip" install "numpy<2.0.0" opencv-python onnxruntime-gpu pillow PySide6 PyYAML pyvirtualcam --no-cache-dir
}

# ------------------------------------------------------------------------------
# STEP 6: Automatically Download All AI Model Weights into models/
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[6/7] Checking and downloading ONNX model weights..." -ForegroundColor Yellow
$modelsDir = Join-Path $CurrentScriptDir "models"
if (-not (Test-Path -LiteralPath $modelsDir)) {
    New-Item -ItemType Directory -Path "$modelsDir" -Force | Out-Null
}
$sourceDir = Join-Path $modelsDir "source"
if (-not (Test-Path -LiteralPath $sourceDir)) {
    New-Item -ItemType Directory -Path "$sourceDir" -Force | Out-Null
}

$dlScript = Join-Path $CurrentScriptDir "tools\download_models.py"
if (-not (Test-Path -LiteralPath $dlScript)) {
    $dlScript = Join-Path $CurrentScriptDir "realtime_faceswap\tools\download_models.py"
}

& "$venvPython" "$dlScript"

# ------------------------------------------------------------------------------
# STEP 7: Run System Diagnostics and CUDA Verification
# ------------------------------------------------------------------------------
if (-not $SkipDiagnostics) {
    Write-Host ""
    Write-Host "[7/7] Running diagnostic benchmark and CUDA verification..." -ForegroundColor Yellow
    $diagScript = Join-Path $CurrentScriptDir "tools\diagnostics.py"
    if (-not (Test-Path -LiteralPath $diagScript)) {
        $diagScript = Join-Path $CurrentScriptDir "realtime_faceswap\tools\diagnostics.py"
    }
    & "$venvPython" "$diagScript"
}

# Create Desktop Shortcut
try {
    $desktopPath = [Environment]::GetFolderPath("Desktop")
    $shortcutPath = Join-Path $desktopPath "Real-Time Face-Swap Studio.lnk"
    $wsh = New-Object -ComObject WScript.Shell
    $sc = $wsh.CreateShortcut($shortcutPath)
    $sc.TargetPath = Join-Path $CurrentScriptDir "run.bat"
    $sc.WorkingDirectory = "$CurrentScriptDir"
    $sc.Description = "Launch Real-Time Face-Swap Control Studio (RTX 4050)"
    $sc.Save()
    Write-Host " [OK] Desktop Shortcut created: 'Real-Time Face-Swap Studio.lnk'" -ForegroundColor Green
} catch {
    Write-Host " [INFO] Shortcut creation completed." -ForegroundColor Gray
}

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Green
Write-Host " [OK] INSTALLATION AND SELF-SETUP COMPLETE!" -ForegroundColor Green
Write-Host " All tools, packages, and models are configured in this directory." -ForegroundColor Green
Write-Host " To start the engine: double-click run.bat or run .\run.ps1" -ForegroundColor Cyan
Write-Host " To completely uninstall without leaving trash: run uninstall.bat" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to exit"
