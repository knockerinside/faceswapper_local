# ==============================================================================
# Complete Uninstaller for Real-Time Face-Swap Engine
# ==============================================================================
[CmdletBinding()]
param(
    [switch]$KeepModels,
    [switch]$Force
)

$CurrentScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$parentUninstall = Join-Path (Split-Path -Parent $CurrentScriptDir) "uninstall.ps1"

if (Test-Path $parentUninstall) {
    $args = @()
    if ($KeepModels) { $args += "-KeepModels" }
    if ($Force) { $args += "-Force" }
    & $parentUninstall @args
} else {
    Write-Host "Uninstaller running in $CurrentScriptDir"
}
