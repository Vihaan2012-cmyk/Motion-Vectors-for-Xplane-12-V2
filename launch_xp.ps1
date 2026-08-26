$root = "d:\Steam Games\steamapps\common\X-Plane 12\MotionVectors"
$xp   = "d:\Steam Games\steamapps\common\X-Plane 12"
$env:VK_LAYER_PATH           = Join-Path $root "build\vklayer"
$env:VK_LOADER_LAYERS_ENABLE = "VK_LAYER_mv"
$env:TAA_VELOCITY            = "1"
$env:TAA_LAYER_TRACE         = "1"
Start-Process -FilePath (Join-Path $xp "X-Plane.exe") -WorkingDirectory $xp
Write-Host "launched"
