# THROWAWAY SPIKE launcher: DLSS-NR viability probe. Loads the snippet named by
# -Dll directly in the layer at device creation and traces what it answers.
param([string]$Dll = "$env:USERPROFILE\Downloads\1-Click-DLSS5-v2.6.0\payload\nvngx_dlssnr.dll")
Set-Location $PSScriptRoot
$env:TAA_NR_PROBE = "1"
$env:TAA_NR_DLL   = $Dll
Write-Host "NR PROBE armed: $Dll" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "launch_xp.ps1")
