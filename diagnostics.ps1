# ==============================================================================
# Diagnostics Launcher for Windows 11
# ==============================================================================
$CurrentScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath "$CurrentScriptDir"

$venvPython = Join-Path $CurrentScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath "$venvPython")) {
    $venvPython = Join-Path $CurrentScriptDir "realtime_faceswap\.venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath "$venvPython")) {
    $venvPython = "python"
}

$diagScript = Join-Path $CurrentScriptDir "tools\diagnostics.py"
if (-not (Test-Path -LiteralPath "$diagScript")) {
    $diagScript = Join-Path $CurrentScriptDir "realtime_faceswap\tools\diagnostics.py"
}

& "$venvPython" "$diagScript"
