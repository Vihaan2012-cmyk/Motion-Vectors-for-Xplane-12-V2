# metrics_dlssd.ps1 - one measured run with DLSS-D Ray Reconstruction on.
#
# The sweep deliberately keeps RR out: it needs the vulkan-1.dll shim installed
# and TAA_SL_DIR pointing at a user-supplied Streamline runtime, which is a
# different launch environment from every other row and would confound the
# table. This script sets that environment up the way shim_test.ps1 does, then
# hands off to metrics_run.ps1 with the RR guard lifted, so the RR row is
# measured by the same code path as everything else.
#
#   .\metrics_dlssd.ps1                 # runtime = .\streamline
#   .\metrics_dlssd.ps1 -SlDir <dir>    # another runtime
param(
    [string]$SlDir = "",
    [int]$Seconds = 200,
    [int]$Report = 200,
    [string]$Label = "dlssd"
)
$ErrorActionPreference = "Stop"
$MV = $PSScriptRoot
$XP = Split-Path $MV -Parent
$BK = Join-Path $MV "v1_sl_leftovers_backup"

Get-Process "X-Plane*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 1500

# Same install shim_test.ps1 does: our forwarding vulkan-1.dll beside the exe,
# the real loader renamed beside it, stale SL plugin DLLs parked. The nvngx_*
# models stay - NGX resolves them from the exe's own folder.
New-Item -ItemType Directory -Force -Path $BK | Out-Null
foreach ($f in @("sl.dlss_g.dll", "sl.reflex.dll", "sl.pcl.dll", "sl.common.dll")) {
    $p = Join-Path $XP $f
    if (Test-Path $p) { Move-Item $p (Join-Path $BK $f) -Force }
}
Copy-Item (Join-Path $MV "build\vulkan-1.dll") (Join-Path $XP "vulkan-1.dll") -Force
Copy-Item "C:\Windows\System32\vulkan-1.dll"   (Join-Path $XP "vk_real_mv.dll") -Force

$env:TAA_SL_DIR = if ($SlDir) { $SlDir } else { Join-Path $MV "streamline" }
Remove-Item Env:\TAA_DLSS_NR -ErrorAction SilentlyContinue
Write-Host ("  runtime: {0}" -f $env:TAA_SL_DIR) -ForegroundColor Cyan

# The shim arms Streamline from taa.dlssd=1 in the live ini (no env var
# needed); -AllowDlssd stops metrics_run forcing it back to 0, -KeepShim stops
# it removing the files just installed. The mask matches the sweep baseline so
# this row is comparable to the table.
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $MV "metrics_run.ps1") `
    -Label $Label -Seconds $Seconds -Report $Report -AllowDlssd -KeepShim `
    -Set "taa.dlssd=1,taa.metrics_mask=2,taa.metrics_far=0.0001"
$rc = $LASTEXITCODE

# Put X-Plane back to stock afterwards so the next plain run is plain.
foreach ($f in @("vulkan-1.dll", "vk_real_mv.dll")) {
    $p = Join-Path $XP $f
    if (Test-Path $p) { Remove-Item $p -Force -ErrorAction SilentlyContinue }
}
if (Test-Path $BK) {
    Get-ChildItem $BK -File | ForEach-Object { Move-Item $_.FullName (Join-Path $XP $_.Name) -Force }
}
exit $rc
