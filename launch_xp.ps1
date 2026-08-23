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
Start-Process -FilePath (Join-Path $xp "X-Plane.exe") -WorkingDirectory $xp
Write-Host "launched"
