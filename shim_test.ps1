# shim_test.ps1 - install the MotionVectors vulkan-1.dll shim, launch X-Plane
# with Streamline armed, let it reach device creation, then kill it and dump
# the shim trace. One decisive run per code change.
#
#   .\shim_test.ps1                 # DLSS-D Ray Reconstruction (feature 1001)
#   .\shim_test.ps1 -NR             # DLSS-NR (feature 1004), if the driver has it
#   .\shim_test.ps1 -Uninstall      # put X-Plane back the way it was
#   .\shim_test.ps1 -Seconds 90     # longer soak
param(
    [switch]$NR,
    [switch]$Uninstall,
    [switch]$Soak,
    [string]$SlDir = "",
    [int]$Seconds = 60
)
$ErrorActionPreference = "Stop"
$XP  = Split-Path $PSScriptRoot -Parent
$MV  = $PSScriptRoot
$BK  = Join-Path $MV "v1_sl_leftovers_backup"
# v1's Streamline drop still sits in the X-Plane root. sl.interposer probes the
# exe directory, so an older sl.common/sl.dlss_g there can be picked up ahead of
# our 2.12 set. Move them aside for the duration of the test.
# Park only stale SL PLUGIN dlls. The nvngx_* neural models must STAY beside the
# exe: NGX resolves those from the executable's own folder, not from the
# Streamline plugin path - which is exactly how 1-Click installs them.
$Leftovers = @("sl.dlss_g.dll","sl.reflex.dll","sl.pcl.dll","sl.common.dll")

function Stop-XP {
    Get-Process "X-Plane*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 1500
}

if ($Uninstall) {
    Stop-XP
    foreach ($f in @("vulkan-1.dll","vk_real_mv.dll")) {
        $p = Join-Path $XP $f
        if (Test-Path $p) { Remove-Item $p -Force; Write-Host "removed $f" }
    }
    if (Test-Path $BK) {
        Get-ChildItem $BK -File | ForEach-Object {
            Move-Item $_.FullName (Join-Path $XP $_.Name) -Force
            Write-Host ("restored {0}" -f $_.Name)
        }
    }
    Write-Host "X-Plane is back to stock." -ForegroundColor Green
    exit 0
}

Stop-XP

# --- install
New-Item -ItemType Directory -Force -Path $BK | Out-Null
foreach ($f in $Leftovers) {
    $p = Join-Path $XP $f
    if (Test-Path $p) { Move-Item $p (Join-Path $BK $f) -Force; Write-Host "  parked $f" }
}
Copy-Item (Join-Path $MV "build\vulkan-1.dll") (Join-Path $XP "vulkan-1.dll") -Force
Copy-Item "C:\Windows\System32\vulkan-1.dll"   (Join-Path $XP "vk_real_mv.dll") -Force
Write-Host "  installed shim + private loader" -ForegroundColor Cyan

# --- arm
$env:TAA_SL_DIR = if ($SlDir) { $SlDir } else { Join-Path $MV "streamline" }
Write-Host ("  runtime: {0}" -f $env:TAA_SL_DIR) -ForegroundColor Cyan
if ($NR) { $env:TAA_DLSS_NR = "1"; Remove-Item Env:\TAA_DLSSD -ErrorAction SilentlyContinue; $feat = "DLSS-NR (1004)" }
else     { $env:TAA_DLSSD  = "1"; Remove-Item Env:\TAA_DLSS_NR -ErrorAction SilentlyContinue; $feat = "DLSS-D RR (1001)" }

$ini = Join-Path $env:TEMP "taa_live.ini"
if (Test-Path $ini) {
    (Get-Content $ini) -replace '^taa\.dlssd=.*$', 'taa.dlssd=1' | Set-Content $ini -Encoding utf8
}

$trace = Join-Path $env:TEMP "mv_vkshim.txt"
if (Test-Path $trace) { Remove-Item $trace -Force }
$slLog = Join-Path $env:TEMP "sl.log"
if (Test-Path $slLog) { Remove-Item $slLog -Force }

Write-Host ("  feature: {0}" -f $feat) -ForegroundColor Cyan
Write-Host ("  launching X-Plane, {0}s soak..." -f $Seconds) -ForegroundColor Cyan

$exe = Join-Path $XP "X-Plane.exe"
$p = Start-Process -FilePath $exe -WorkingDirectory $XP -PassThru
$deadline = (Get-Date).AddSeconds($Seconds)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 3
    if ($p.HasExited) { Write-Host ("  X-Plane exited early, code {0}" -f $p.ExitCode) -ForegroundColor Yellow; break }
    # Stop as soon as the shim has answered the question we launched to ask.
    if (Test-Path $trace) {
        $t = Get-Content $trace -Raw
        if (-not $Soak -and $t -match "VIABILITY|fresh thread|TIMED OUT|CreateThread failed|SL HAND-OFF DONE") { Write-Host "  hand-off resolved - stopping early" -ForegroundColor Green; break }
    }
}
Stop-XP

Write-Host ""
Write-Host "================= SHIM TRACE =================" -ForegroundColor Yellow
if (Test-Path $trace) { Get-Content $trace } else { Write-Host "(no trace - shim never loaded)" -ForegroundColor Red }
Write-Host "=============================================" -ForegroundColor Yellow
