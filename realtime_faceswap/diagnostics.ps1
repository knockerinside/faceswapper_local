# Launch diagnostics tool
$venvPython = ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    & $venvPython tools/diagnostics.py
} else {
    python tools/diagnostics.py
}
