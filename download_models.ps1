# ==============================================================================
# Real-Time Face-Swap Automated Model Downloader
# ==============================================================================
[CmdletBinding()]
param()

$CurrentScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath "$CurrentScriptDir"

$venvPython = ""
if ($env:VIRTUAL_ENV -and (Test-Path -LiteralPath (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))) {
    $venvPython = Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
}
if (-not $venvPython -and (Test-Path -LiteralPath (Join-Path $CurrentScriptDir ".venv\Scripts\python.exe"))) {
    $venvPython = Join-Path $CurrentScriptDir ".venv\Scripts\python.exe"
}
if (-not $venvPython -and (Test-Path -LiteralPath (Join-Path $CurrentScriptDir "venv\Scripts\python.exe"))) {
    $venvPython = Join-Path $CurrentScriptDir "venv\Scripts\python.exe"
}
if (-not $venvPython -and (Test-Path -LiteralPath (Join-Path $CurrentScriptDir "realtime_faceswap\.venv\Scripts\python.exe"))) {
    $venvPython = Join-Path $CurrentScriptDir "realtime_faceswap\.venv\Scripts\python.exe"
}
if (-not $venvPython) {
    $venvPython = "python"
}

$scriptPath = Join-Path $CurrentScriptDir "tools\download_models.py"
if (-not (Test-Path -LiteralPath "$scriptPath")) {
    $scriptPath = Join-Path $CurrentScriptDir "realtime_faceswap\tools\download_models.py"
}

& "$venvPython" "$scriptPath" @args

