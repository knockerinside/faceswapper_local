@echo off
setlocal enabledelayedexpansion

title Real-Time Face-Swap Studio (RTX 4050)
cd /d "%~dp0"

set "PYTHONPATH=.;realtime_faceswap;!PYTHONPATH!"

:: Parse potential venv argument or profile
set "CUSTOM_VENV="
set "PROFILE=balanced"
set "EXTRA_ARGS="

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--venv" (
    set "CUSTOM_VENV=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="-v" (
    set "CUSTOM_VENV=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="performance" (
    set "PROFILE=performance"
    shift
    goto parse_args
)
if /i "%~1"=="balanced" (
    set "PROFILE=balanced"
    shift
    goto parse_args
)
if /i "%~1"=="quality" (
    set "PROFILE=quality"
    shift
    goto parse_args
)
if exist "%~1\Scripts\python.exe" (
    set "CUSTOM_VENV=%~1"
    shift
    goto parse_args
)
:: Otherwise collect remaining argument
set "EXTRA_ARGS=!EXTRA_ARGS! %1"
shift
goto parse_args

:args_done

:: Determine Python executable
set "PYTHON_EXE="
set "ENV_LABEL="

:: 1. If explicit custom venv path was passed
if not "!CUSTOM_VENV!"=="" (
    if exist "!CUSTOM_VENV!\Scripts\python.exe" (
        set "PYTHON_EXE=!CUSTOM_VENV!\Scripts\python.exe"
        set "ENV_LABEL=Custom venv (!CUSTOM_VENV!)"
    ) else if exist "!CUSTOM_VENV!\python.exe" (
        set "PYTHON_EXE=!CUSTOM_VENV!\python.exe"
        set "ENV_LABEL=Custom Python (!CUSTOM_VENV!)"
    )
)

:: 2. Check if user already activated a previous venv in this terminal session
if "!PYTHON_EXE!"=="" if not "%VIRTUAL_ENV%"=="" (
    if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
        set "PYTHON_EXE=%VIRTUAL_ENV%\Scripts\python.exe"
        set "ENV_LABEL=Active Terminal Venv (%VIRTUAL_ENV%)"
    )
)

:: 3. Check standard local project environments
if "!PYTHON_EXE!"=="" if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
    set "ENV_LABEL=Isolated .venv"
)
if "!PYTHON_EXE!"=="" if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
    set "ENV_LABEL=Existing venv"
)
if "!PYTHON_EXE!"=="" if exist "env\Scripts\python.exe" (
    set "PYTHON_EXE=env\Scripts\python.exe"
    set "ENV_LABEL=Existing env"
)

:: 4. If no virtual environment exists, run setup.bat to initialize one
if "!PYTHON_EXE!"=="" (
    echo [*] No virtual environment detected. Running setup.bat...
    call setup.bat
    if exist ".venv\Scripts\python.exe" (
        set "PYTHON_EXE=.venv\Scripts\python.exe"
        set "ENV_LABEL=Isolated .venv"
    ) else (
        echo.
        echo =================================================================
        echo  [ERROR] Virtual environment was not found after setup.
        echo  Please run setup.bat directly to view the setup log.
        echo =================================================================
        echo.
        pause
        exit /b 1
    )
)

:: Verify essential dependencies in selected Python environment
"!PYTHON_EXE!" -c "import numpy, cv2, onnxruntime" >nul 2>&1
if !errorlevel! neq 0 (
    echo =================================================================
    echo  [!] Missing core libraries (numpy, cv2, onnxruntime) in:
    echo      !ENV_LABEL!
    echo  [*] Installing required dependencies from requirements.txt...
    echo =================================================================
    echo.
    "!PYTHON_EXE!" -m pip install --upgrade pip setuptools wheel --quiet
    set "REQ_FILE=requirements.txt"
    if not exist "!REQ_FILE!" set "REQ_FILE=realtime_faceswap\requirements.txt"
    "!PYTHON_EXE!" -m pip install -r "!REQ_FILE!"
    if !errorlevel! neq 0 (
        echo [!] Retrying with binary wheels...
        "!PYTHON_EXE!" -m pip install "numpy<2.0.0" opencv-python onnxruntime-gpu pillow PySide6 PyYAML pyvirtualcam --no-cache-dir
    )
    "!PYTHON_EXE!" -c "import numpy, cv2, onnxruntime" >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to install required dependencies into !ENV_LABEL!.
        pause
        exit /b 1
    )
    echo  [OK] Dependencies installed and verified!
    echo.
)

:: Check if core models are present in models/
set "MODELS_DIR=models"
if not exist "!MODELS_DIR!" set "MODELS_DIR=realtime_faceswap\models"
set "HAS_DET="
if exist "!MODELS_DIR!\scrfd_10g_bnkps.onnx" set "HAS_DET=1"
if exist "!MODELS_DIR!\det_10g.onnx" set "HAS_DET=1"
set "HAS_SWP="
if exist "!MODELS_DIR!\inswapper_128_fp16.onnx" set "HAS_SWP=1"
if exist "!MODELS_DIR!\inswapper_128.onnx" set "HAS_SWP=1"

if "!HAS_DET!"=="" (
    echo =================================================================
    echo  [!] Required AI models are not yet installed in !MODELS_DIR!.
    echo  [*] Opening Model Download ^& Selection Center...
    echo =================================================================
    echo.
    set "DL_SCRIPT=tools\download_models.py"
    if not exist "!DL_SCRIPT!" set "DL_SCRIPT=realtime_faceswap\tools\download_models.py"
    "!PYTHON_EXE!" "!DL_SCRIPT!"
) else if "!HAS_SWP!"=="" (
    echo =================================================================
    echo  [!] Face swapper model is missing from !MODELS_DIR!.
    echo  [*] Opening Model Download ^& Selection Center...
    echo =================================================================
    echo.
    set "DL_SCRIPT=tools\download_models.py"
    if not exist "!DL_SCRIPT!" set "DL_SCRIPT=realtime_faceswap\tools\download_models.py"
    "!PYTHON_EXE!" "!DL_SCRIPT!"
)

:: Check if entry point script exists
set "APP_SCRIPT=app.py"
if not exist "%APP_SCRIPT%" set "APP_SCRIPT=realtime_faceswap\app.py"

echo =================================================================
echo  Launching Real-Time Face-Swap Studio
echo  Profile:     !PROFILE!
echo  Environment: !ENV_LABEL!
echo =================================================================
echo.

"!PYTHON_EXE!" "%APP_SCRIPT%" --profile !PROFILE! !EXTRA_ARGS!

if !errorlevel! neq 0 (
    echo.
    echo =================================================================
    echo  [!] Application exited with code: !errorlevel!
    echo  Check logs in logs\realtime_faceswap.log for details.
    echo =================================================================
    pause
)

