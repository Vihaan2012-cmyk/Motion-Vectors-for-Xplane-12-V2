# Take Steam's two Vulkan layers out of the chain.
#
# SteamOverlayVulkanLayer64 hooks vkQueuePresentKHR. Steam ships an FPS limiter
# and continuous Game Recording, both of which pace frames from inside present -
# which is where our 17 ms per-frame block is measured, and would explain a
# ceiling that is smooth (not vblank-quantised), survives IMMEDIATE, and does
# not move with resolution or GPU load.
#
# Both layers honour the loader's per-layer disable variable.
$root = (Get-Location).Path
$xp   = Split-Path $root -Parent
$env:VK_LAYER_PATH           = Join-Path $root "build\vklayer"
$env:VK_LOADER_LAYERS_ENABLE = "VK_LAYER_mv"
$env:TAA_VELOCITY            = "1"
$env:TAA_LAYER_TRACE         = "1"
$env:DISABLE_VK_LAYER_VALVE_steam_overlay_1   = "1"
$env:DISABLE_VK_LAYER_VALVE_steam_fossilize_1 = "1"
# Also ask the loader directly, which covers naming differences between builds.
$env:VK_LOADER_LAYERS_DISABLE = "VK_LAYER_VALVE_steam_overlay,VK_LAYER_VALVE_steam_fossilize"
Remove-Item Env:TAA_PRESENT_MODE  -ErrorAction SilentlyContinue
Remove-Item Env:TAA_SWAP_IMAGES   -ErrorAction SilentlyContinue
Remove-Item Env:TAA_MAX_OVERDRIVE -ErrorAction SilentlyContinue
Start-Process -FilePath (Join-Path $xp "X-Plane.exe") -WorkingDirectory $xp
Write-Host "launched with Steam's Vulkan layers disabled"
