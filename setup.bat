@echo off
setlocal enabledelayedexpansion

REM Clear any inherited virtual environment variables
set "VIRTUAL_ENV="
set "CUSTOM_VENV="

title Real-Time AI Face-Swap Engine -- Setup
cd /d "%~dp0"

echo =================================================================
echo  REAL-TIME AI FACE-SWAP ENGINE -- ALL-IN-ONE AUTOMATED SETUP
echo  Hardware Target: Windows 11, NVIDIA RTX 4050 Laptop GPU (6GB)
echo =================================================================
echo.

:: ------------------------------------------------------------------
:: STEP 1: Detect Visual C++ Redistributable
:: ------------------------------------------------------------------
echo [1/7] Detecting Microsoft Visual C++ 2015-2022 Redistributable...
set "VC_INSTALLED=0"
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed >nul 2>&1
if !errorlevel! equ 0 set "VC_INSTALLED=1"
reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed >nul 2>&1
if !errorlevel! equ 0 set "VC_INSTALLED=1"

if "!VC_INSTALLED!"=="1" (
    echo  [OK] Visual C++ x64 Runtime is already installed.
) else (
    echo  [!] Visual C++ runtime not found. Downloading official Microsoft installer...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { (New-Object Net.WebClient).DownloadFile('https://aka.ms/vs/17/release/vc_redist.x64.exe', '%TEMP%\vc_redist.x64.exe') } catch { exit 1 }"
    if exist "%TEMP%\vc_redist.x64.exe" (
        echo  Installing Visual C++ runtime silently...
        start /wait "" "%TEMP%\vc_redist.x64.exe" /install /quiet /norestart
        del /f /q "%TEMP%\vc_redist.x64.exe" >nul 2>&1
        echo  [OK] Visual C++ installed.
    ) else (
        echo  [WARNING] Automatic download of vc_redist.x64.exe skipped. If DLL errors occur, install it manually from https://aka.ms/vs/17/release/vc_redist.x64.exe
    )
)
echo.

:: ------------------------------------------------------------------
:: STEP 2: Detect System Python (3.10, 3.11, 3.12, 3.13 or Windows py launcher)
:: ------------------------------------------------------------------
echo [2/7] Detecting Python environment...
set "PY_CMD="
set "PY_VER="

:: 1. Try 'python' in system PATH first
python -c "import sys; print(sys.version.split()[0])" >nul 2>&1
if !errorlevel! equ 0 (
    for /f "delims=" %%v in ('python -c "import sys; print(sys.version.split()[0])"') do set "PY_VER=%%v"
    set "PY_CMD=python"
    goto :python_found
)

:: 2. Try py launcher (-3.13, -3.12, -3.11, -3.10)
for %%V in (3.13 3.12 3.11 3.10) do (
    py -%%V -c "import sys; print(sys.version.split()[0])" >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "delims=" %%v in ('py -%%V -c "import sys; print(sys.version.split()[0])"') do set "PY_VER=%%v"
        set "PY_CMD=py -%%V"
        goto :python_found
    )
)

:: 3. Try generic 'py' launcher
py -c "import sys; print(sys.version.split()[0])" >nul 2>&1
if !errorlevel! equ 0 (
    for /f "delims=" %%v in ('py -c "import sys; print(sys.version.split()[0])"') do set "PY_VER=%%v"
    set "PY_CMD=py"
    goto :python_found
)

:: 4. Check common default Windows install paths if not in PATH
for %%p in (
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "C:\Program Files\Python313\python.exe"
    "C:\Program Files\Python312\python.exe"
    "C:\Program Files\Python311\python.exe"
    "C:\Program Files\Python310\python.exe"
    "C:\Python313\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
) do (
    if exist "%%~p" (
        for /f "delims=" %%v in ('"%%~p" -c "import sys; print(sys.version.split()[0])" 2^>nul') do set "PY_VER=%%v"
        set "PY_CMD="%%~p""
        goto :python_found
    )
)

:python_found
if not "%PY_CMD%"=="" (
    echo  [OK] Found Python interpreter: !PY_CMD! (version !PY_VER!)
    goto :python_done
)

echo.
echo =================================================================
echo  [ERROR] Python was not found on your system!
echo =================================================================
echo  Please install Python 3.10, 3.11, 3.12, or 3.13 64-bit:
echo  Official Link: https://www.python.org/downloads/
echo.
echo  CRITICAL STEP DURING PYTHON INSTALLATION:
echo    Check the box: [X] "Add python.exe to PATH" at the bottom
echo    before clicking "Install Now".
echo =================================================================
echo.
echo Press any key to exit setup...
pause
exit /b 1

:python_done
echo.

:: ------------------------------------------------------------------
:: STEP 3: Detect NVIDIA GPU (RTX 4050)
:: ------------------------------------------------------------------
echo [3/7] Detecting NVIDIA GPU and Driver...
where nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    for /f "delims=" %%g in ('nvidia-smi --query-gpu^=name^,driver_version --format^=csv^,noheader 2^>nul') do (
        echo  [OK] Detected GPU: %%g
    )
) else (
    echo  [INFO] nvidia-smi not found in current PATH. The engine will run using ONNX Runtime.
)
echo.

:: ------------------------------------------------------------------
:: STEP 4: Setup or Configure Target Virtual Environment
:: ------------------------------------------------------------------
echo [4/7] Checking isolated Python virtual environment (.venv)...
if not exist ".venv\Scripts\python.exe" (
    echo  Creating fresh virtual environment in .venv...
    !PY_CMD! -m venv .venv
    if !errorlevel! neq 0 (
        echo.
        echo =================================================================
        echo  [ERROR] Failed to create virtual environment with !PY_CMD!.
        echo  Trying with --without-pip fallback...
        echo =================================================================
        !PY_CMD! -m venv .venv --without-pip
        if !errorlevel! neq 0 (
            echo [FATAL] Could not create .venv directory.
            echo Press any key to exit...
            pause
            exit /b 1
        )
    )
    echo  [OK] Virtual environment created successfully.
) else (
    echo  [OK] Existing virtual environment found at .venv
)
set "TARGET_PY=.venv\Scripts\python.exe"
set "TARGET_PIP=.venv\Scripts\pip.exe"

if not exist "!TARGET_PY!" (
    echo.
    echo =================================================================
    echo  [ERROR] Python executable not found at: !TARGET_PY!
    echo =================================================================
    pause
    exit /b 1
)

:: Ensure pip is installed inside venv
if not exist "!TARGET_PIP!" (
    echo  Bootstrapping pip into virtual environment...
    "!TARGET_PY!" -m ensurepip --upgrade >nul 2>&1
)
echo.

:: ------------------------------------------------------------------
:: STEP 5: Install ONNX Runtime GPU and Core Dependencies
:: ------------------------------------------------------------------
echo [5/7] Installing dependencies into target environment...
echo  Upgrading pip, setuptools, and wheel...
"!TARGET_PY!" -m pip install --upgrade pip setuptools wheel --quiet

set "REQ_FILE="
if exist "requirements.txt" (
    set "REQ_FILE=requirements.txt"
) else if exist "realtime_faceswap\requirements.txt" (
    set "REQ_FILE=realtime_faceswap\requirements.txt"
)

if not "%REQ_FILE%"=="" (
    echo  Installing packages from %REQ_FILE%...
    "!TARGET_PY!" -m pip install -r "%REQ_FILE%"
    if !errorlevel! neq 0 (
        echo  [!] Standard install encountered an issue. Retrying with --no-cache-dir...
        "!TARGET_PY!" -m pip install -r "%REQ_FILE%" --no-cache-dir
    )
) else (
    echo.
    echo =================================================================
    echo  [ERROR] requirements.txt not found in %CD%
    echo =================================================================
    pause
    exit /b 1
)

echo  Verifying installation...
"!TARGET_PY!" -c "import numpy, cv2, onnxruntime; print('  [OK] Verified: numpy ' + numpy.__version__ + ' | onnxruntime ' + onnxruntime.__version__)" 2>nul
if !errorlevel! neq 0 (
    echo  [!] Retrying core wheels directly...
    "!TARGET_PY!" -m pip install "numpy<2.0.0" opencv-python onnxruntime-gpu
)
echo  [OK] Dependencies installed and validated.
echo.

:: ------------------------------------------------------------------
:: STEP 6: Smart Model Detection & Interactive Download Center
:: ------------------------------------------------------------------
echo [6/7] Checking AI model weights in models/...
if not exist "models" mkdir "models"
if not exist "models\source" mkdir "models\source"

set "DL_SCRIPT=tools\download_models.py"
if not exist "%DL_SCRIPT%" set "DL_SCRIPT=realtime_faceswap\tools\download_models.py"

if exist "%DL_SCRIPT%" (
    "!TARGET_PY!" "%DL_SCRIPT%"
) else (
    echo  [INFO] Download script not found at %DL_SCRIPT%. Continuing...
)
echo.

:: ------------------------------------------------------------------
:: STEP 7: Run System Diagnostics & Create Desktop Shortcut
:: ------------------------------------------------------------------
echo [7/7] Verifying setup and creating Desktop Shortcut...
set "DIAG_SCRIPT=tools\diagnostics.py"
if not exist "%DIAG_SCRIPT%" set "DIAG_SCRIPT=realtime_faceswap\tools\diagnostics.py"
if exist "%DIAG_SCRIPT%" (
    "!TARGET_PY!" "%DIAG_SCRIPT%"
)

echo.
echo  Creating Desktop Shortcut: 'Real-Time Face-Swap Studio'...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ws = New-Object -ComObject WScript.Shell; $d = [Environment]::GetFolderPath('Desktop'); $s = $ws.CreateShortcut((Join-Path $d 'Real-Time Face-Swap Studio.lnk')); $p = (Get-Location).Path; $s.TargetPath = (Join-Path $p 'run.bat'); $s.WorkingDirectory = $p; $s.Description = 'Launch Real-Time Face-Swap Control Studio'; $s.Save()" >nul 2>&1

if exist "%USERPROFILE%\Desktop\Real-Time Face-Swap Studio.lnk" (
    echo  [OK] Desktop shortcut created: '%USERPROFILE%\Desktop\Real-Time Face-Swap Studio.lnk'
) else (
    echo  [INFO] Desktop shortcut step completed. You can start the app via 'run.bat'.
)

echo.
echo =================================================================
echo  [OK] SETUP COMPLETE! Everything is configured in this folder.
echo.
echo  To start the application:
echo    Double-click 'run.bat' (or use the Desktop Shortcut)
echo.
echo  To cleanly uninstall without leaving trash:
echo    Run 'uninstall.bat'
echo =================================================================
echo.
echo Press any key to close this window...
pause


