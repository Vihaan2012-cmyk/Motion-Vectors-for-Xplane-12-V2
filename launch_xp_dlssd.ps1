# launch_xp_dlssd.ps1 - enable DLSS-D Ray Reconstruction (Streamline feature
# 1001) and launch X-Plane through the normal layer launcher.
#
# Needs streamline/ with: sl.interposer.dll + sl.common.dll + sl.dlss_d.dll +
# nvngx_dlssd.dll (all present after build 41 setup; interposer/common from the
# public Streamline SDK 2.12.0, sl.dlss_d + nvngx_dlssd shipped/extracted).
param([string]$SlDir = "")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if ([string]::IsNullOrEmpty($SlDir)) { $SlDir = Join-Path $PSScriptRoot "streamline" }

$env:TAA_SL_DIR = $SlDir
$env:TAA_DLSSD  = "1"    # turn RR on (also live via taa.dlssd in the ini)

Write-Host "DLSS-D Ray Reconstruction armed." -ForegroundColor Cyan
Write-Host "  TAA_SL_DIR = $SlDir"
foreach ($f in @("sl.interposer.dll","sl.common.dll","sl.dlss_d.dll","nvngx_dlssd.dll")) {
    $p = Join-Path $SlDir $f
    if (Test-Path $p) { Write-Host ("  ok   {0}" -f $f) }
    else { Write-Host ("  MISS {0}" -f $f) -ForegroundColor Yellow }
}
Write-Host ""
& (Join-Path $PSScriptRoot "launch_xp.ps1")
