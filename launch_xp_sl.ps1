# launch_xp_sl.ps1 - arm the Streamline DLSS-NR viability probe (mv_sl_probe.h)
# and launch X-Plane through the normal layer launcher. THROWAWAY spike.
#
# The probe needs the Streamline runtime in one folder. The 1-Click package
# ships only the feature plugin + snippet; sl.interposer.dll + sl.common.dll
# come from the public NVIDIA Streamline SDK release (v2.12.0, bin/x64). All
# four sit side by side in <SlDir>:
#   sl.interposer.dll  sl.common.dll  sl.dlss_nr.dll  nvngx_dlssnr.dll
#
# Default SlDir: <this folder>\streamline
param(
    [string]$SlDir = ""
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if ([string]::IsNullOrEmpty($SlDir)) { $SlDir = Join-Path $PSScriptRoot "streamline" }

$env:TAA_SL_PROBE = "1"
$env:TAA_SL_DIR   = $SlDir

Write-Host "SL PROBE armed." -ForegroundColor Cyan
Write-Host "  TAA_SL_DIR = $SlDir"
$need = @("sl.interposer.dll","sl.common.dll","sl.dlss_nr.dll","nvngx_dlssnr.dll")
foreach ($f in $need) {
    $p = Join-Path $SlDir $f
    if (Test-Path $p) {
        $h = (Get-FileHash -Algorithm SHA256 $p).Hash.ToLower()
        Write-Host ("  ok   {0,-20} {1}" -f $f, $h)
    } else {
        Write-Host ("  MISS {0}" -f $f) -ForegroundColor Yellow
    }
}
Write-Host ""

# Chain the base launcher, which sets VK_LOADER_LAYERS_ENABLE / VK_LAYER_PATH /
# TAA_LAYER_TRACE and starts X-Plane. Without it the layer never attaches.
& (Join-Path $PSScriptRoot "launch_xp.ps1")
