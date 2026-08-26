# X-Plane under validation WITH frame generation, plus loader diagnostics.
#
# VK_LOADER_DEBUG=layer makes the loader print the layer chain it actually
# builds, in order. That settles whether our layer sits above or below
# validation instead of it being inferred - and layer order is the whole
# question here, because FFX hands X-Plane a proxy swapchain handle that only
# works if WE absorb it before validation ever sees it.
$root = (Get-Location).Path
$xp   = Split-Path $root -Parent
$sdk  = (Get-ChildItem "C:\VulkanSDK\*" -Directory | Sort-Object Name | Select-Object -Last 1).FullName + "\Bin"

$env:VK_LAYER_PATH           = (Join-Path $root "build\vklayer") + ";" + $sdk
$env:VK_INSTANCE_LAYERS      = "VK_LAYER_KHRONOS_validation;VK_LAYER_mv"
Remove-Item Env:VK_LOADER_LAYERS_ENABLE -ErrorAction SilentlyContinue
$env:VK_LAYER_SETTINGS_PATH  = Join-Path $root "gpuav_lite"
$env:VK_LOADER_DEBUG         = "layer"
$env:TAA_VELOCITY            = "1"
$env:TAA_LAYER_TRACE         = "1"

$loaderLog = Join-Path $root "gpuav_lite\loader.txt"
Remove-Item $loaderLog -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $root "gpuav_lite\val_log.txt") -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $env:TEMP "taa_layer.txt") -Force -ErrorAction SilentlyContinue

Start-Process -FilePath (Join-Path $xp "X-Plane.exe") -WorkingDirectory $xp `
              -RedirectStandardError $loaderLog -RedirectStandardOutput "$loaderLog.out"
Write-Host "launched with VK_LOADER_DEBUG=layer -> gpuav_lite\loader.txt"
