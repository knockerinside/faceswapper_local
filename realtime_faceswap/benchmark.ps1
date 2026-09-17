# Launch benchmark suite
param(
    [int]$Frames = 200,
    [int]$StressDuration = 0
)

$venvPython = ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    & $venvPython tools/benchmark.py --frames $Frames --stress-duration $StressDuration
} else {
    python tools/benchmark.py --frames $Frames --stress-duration $StressDuration
}
