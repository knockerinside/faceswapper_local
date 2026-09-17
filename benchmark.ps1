# ==============================================================================
# Benchmark Launcher for Windows 11
# ==============================================================================
param(
    [int]$Frames = 200,
    [int]$StressDuration = 0
)

$CurrentScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath "$CurrentScriptDir"

$venvPython = Join-Path $CurrentScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath "$venvPython")) {
    $venvPython = Join-Path $CurrentScriptDir "realtime_faceswap\.venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath "$venvPython")) {
    $venvPython = "python"
}

$benchScript = Join-Path $CurrentScriptDir "tools\benchmark.py"
if (-not (Test-Path -LiteralPath "$benchScript")) {
    $benchScript = Join-Path $CurrentScriptDir "realtime_faceswap\tools\benchmark.py"
}

$argsList = @("$benchScript", "--frames", $Frames)
if ($StressDuration -gt 0) {
    $argsList += @("--stress-duration", $StressDuration)
}

& "$venvPython" @argsList
