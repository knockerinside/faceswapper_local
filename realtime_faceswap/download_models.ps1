# ==============================================================================
# Real-Time Face-Swap Automated Model Downloader
# ==============================================================================
[CmdletBinding()]
param()

$venvPython = ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $venvPython = "python.exe"
}

& $venvPython tools\download_models.py
