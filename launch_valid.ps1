$root = (Get-Location).Path
$xp   = Split-Path $root -Parent
$env:VK_LAYER_PATH           = (Join-Path $root "build\vklayer") + ";C:\VulkanSDK\1.4.357.0\Bin"
$env:VK_LOADER_LAYERS_ENABLE = "VK_LAYER_mv,VK_LAYER_KHRONOS_validation"
$env:VK_LAYER_SETTINGS_PATH  = "C:\Users\bansa\AppData\Local\Temp\vk_layer_settings.txt"
$env:TAA_LAYER_TRACE         = "1"
$env:TAA_VELOCITY            = "1"
Start-Process -FilePath (Join-Path $xp "X-Plane.exe") -WorkingDirectory $xp
Write-Host "launched with validation"
