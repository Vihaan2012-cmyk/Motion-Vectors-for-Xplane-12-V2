# Plain launch, no validation layer. Mirrors exactly what launcher.cpp sets:
# VK_LAYER_PATH -> the built manifest dir, VK_LOADER_LAYERS_ENABLE -> our layer,
# TAA_VELOCITY=1 -> the master switch for SPIR-V injection. The shipped
# MotionVectorsLauncher.exe cannot be used from build\ because it looks for
# X-Plane.exe beside itself or one level up, and build\ is two levels down.
$root = (Get-Location).Path
$xp   = Split-Path $root -Parent
$env:VK_LAYER_PATH           = Join-Path $root "build\vklayer"
$env:VK_LOADER_LAYERS_ENABLE = "VK_LAYER_mv"
$env:TAA_VELOCITY            = "1"
# Trace ON: the last crash reported "Location undetermined" with no dump, so
# the layer's own log is the only record of what it did on the resize path.
$env:TAA_LAYER_TRACE         = "1"
Remove-Item Env:VK_LAYER_SETTINGS_PATH -ErrorAction SilentlyContinue

# ---- A CLEAN TRACE PER RUN, VERIFIED.
#
# Remove-Item on the trace fails silently while a dying X-Plane still holds the
# handle, and the layer opens it in APPEND mode - so the next run's lines land
# underneath the previous run's. That is not a cosmetic problem: two runs in one
# file read as one process doing something impossible, and it cost several wrong
# diagnoses here (a "second VkDevice", a "duplicate swapchain activation") that
# were simply the next launch.
#
# Wait for the file to actually be gone before starting.
$trace = Join-Path $env:TEMP "taa_layer.txt"
for ($i = 0; $i -lt 40 -and (Test-Path $trace); $i++) {
    Remove-Item $trace -Force -ErrorAction SilentlyContinue
    if (Test-Path $trace) { Start-Sleep -Milliseconds 250 }
}
if (Test-Path $trace) {
    Write-Host "WARNING: could not clear $trace - it is still locked. The trace" -ForegroundColor Yellow
    Write-Host "         will contain more than one run and must not be read as one." -ForegroundColor Yellow
}

Start-Process -FilePath (Join-Path $xp "X-Plane.exe") -WorkingDirectory $xp
Write-Host "launched"
