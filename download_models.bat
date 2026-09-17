@echo off
setlocal enabledelayedexpansion

title Real-Time AI Face-Swap Engine -- Model Downloader
cd /d "%~dp0"

set "PYTHONPATH=.;realtime_faceswap;!PYTHONPATH!"

set "PYTHON_EXE="
if not "%VIRTUAL_ENV%"=="" if exist "%VIRTUAL_ENV%\Scripts\python.exe" set "PYTHON_EXE=%VIRTUAL_ENV%\Scripts\python.exe"
if "!PYTHON_EXE!"=="" if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if "!PYTHON_EXE!"=="" if exist "venv\Scripts\python.exe" set "PYTHON_EXE=venv\Scripts\python.exe"
if "!PYTHON_EXE!"=="" if exist "realtime_faceswap\.venv\Scripts\python.exe" set "PYTHON_EXE=realtime_faceswap\.venv\Scripts\python.exe"
if "!PYTHON_EXE!"=="" set "PYTHON_EXE=python"

set "SCRIPT=tools\download_models.py"
if not exist "%SCRIPT%" set "SCRIPT=realtime_faceswap\tools\download_models.py"

"!PYTHON_EXE!" "%SCRIPT%" %*

pause

