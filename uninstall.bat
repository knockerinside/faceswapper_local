@echo off
setlocal enabledelayedexpansion

title Real-Time AI Face-Swap Engine -- Zero-Trash Uninstaller
cd /d "%~dp0"

echo =================================================================
echo  REAL-TIME AI FACE-SWAP ENGINE -- ZERO-TRASH UNINSTALLER
echo =================================================================
echo.

if "%~1"=="/y" goto :do_uninstall
if "%~1"=="-y" goto :do_uninstall
if "%~1"=="/f" goto :do_uninstall
if "%~1"=="-f" goto :do_uninstall

echo This will completely and permanently remove:
echo   1. Isolated Python virtual environment (.venv) and all installed libraries
echo   2. All Python bytecodes (__pycache__, *.pyc, *.pyo)
echo   3. Test and build caches (.pytest_cache, dist, build)
echo   4. Benchmark logs, diagnostic reports, and telemetry logs
echo   5. Downloaded AI model weights (unless /keep-models is specified)
echo   6. Local pip build cache in %LOCALAPPDATA%\pip\cache
echo   7. Desktop shortcut 'Real-Time Face-Swap Studio.lnk'
echo.
set /p "CONFIRM=Are you sure you want to completely uninstall? (Type YES to confirm): "
if /i not "!CONFIRM!"=="YES" if /i not "!CONFIRM!"=="Y" (
    echo.
    echo Uninstallation cancelled. No files were removed.
    pause
    exit /b 0
)

:do_uninstall
echo.
echo [*] Terminating any running background face-swap processes...
taskkill /f /fi "IMAGENAME eq python.exe" /fi "WINDOWTITLE eq Real-Time Face-Swap*" >nul 2>&1

echo [*] Removing virtual environments (.venv)...
if exist ".venv" rd /s /q ".venv"
if exist "realtime_faceswap\.venv" rd /s /q "realtime_faceswap\.venv"

echo [*] Removing Python bytecode caches (__pycache__)...
for /d /r . %%d in (__pycache__) do (
    if exist "%%d" rd /s /q "%%d" >nul 2>&1
)
del /s /f /q *.pyc >nul 2>&1
del /s /f /q *.pyo >nul 2>&1

echo [*] Removing build, test, and compiler artifacts...
if exist ".pytest_cache" rd /s /q ".pytest_cache"
if exist "realtime_faceswap\.pytest_cache" rd /s /q "realtime_faceswap\.pytest_cache"
if exist "build" rd /s /q "build"
if exist "dist" rd /s /q "dist"

echo [*] Removing benchmark logs, CSVs, and telemetry...
if exist "benchmarks" (
    del /f /q "benchmarks\*.json" >nul 2>&1
    del /f /q "benchmarks\*.csv" >nul 2>&1
    del /f /q "benchmarks\*.log" >nul 2>&1
)
if exist "logs" rd /s /q "logs"

if /i "%~1"=="/keep-models" goto :skip_models
if /i "%~2"=="/keep-models" goto :skip_models

echo [*] Removing downloaded ONNX model weights from models/...
if exist "models" (
    del /f /q "models\*.onnx" >nul 2>&1
    del /f /q "models\*.tmp" >nul 2>&1
)
if exist "realtime_faceswap\models" (
    del /f /q "realtime_faceswap\models\*.onnx" >nul 2>&1
    del /f /q "realtime_faceswap\models\*.tmp" >nul 2>&1
)
goto :after_models

:skip_models
echo [*] Preserving models/ folder as requested.

:after_models
echo [*] Purging local pip build cache...
if exist "%LOCALAPPDATA%\pip\cache" (
    rd /s /q "%LOCALAPPDATA%\pip\cache" >nul 2>&1
)

echo [*] Removing Desktop shortcut...
if exist "%USERPROFILE%\Desktop\Real-Time Face-Swap Studio.lnk" (
    del /f /q "%USERPROFILE%\Desktop\Real-Time Face-Swap Studio.lnk" >nul 2>&1
)

echo.
echo =================================================================
echo  [OK] CLEANUP COMPLETE!
echo  All virtual environments, bytecode, logs, and caches purged.
echo  Zero residual trash left on your system.
echo =================================================================
echo.
pause
