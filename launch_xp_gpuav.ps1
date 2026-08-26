# GPU-Assisted Validation launch - name the MSAA null-descriptor crash.
#
# Chain: app -> VK_LAYER_mv -> VK_LAYER_KHRONOS_validation -> driver, so the
# validation layer sees everything OUR layer injects. GPU-AV instruments the
# shaders and reports the exact failed view / null descriptor instead of a
# silent device loss. Expect a heavy FPS hit - this is a diagnosis run.
#
# Repro conditions armed on purpose:
#   - renopt_MSAA 1 in prefs (set before this launch)
#   - TAA_KEEP_ON_777=1 so the 777 bypass does NOT stand the mod down
$root = "d:\Steam Games\steamapps\common\X-Plane 12\MotionVectors"
$xp   = "d:\Steam Games\steamapps\common\X-Plane 12"
$sdk  = "C:\VulkanSDK\1.4.357.0\Bin"

$env:VK_LAYER_PATH           = (Join-Path $root "build\vklayer") + ";" + $sdk
$env:VK_LOADER_LAYERS_ENABLE = "VK_LAYER_mv,VK_LAYER_KHRONOS_validation"
$env:VK_LAYER_SETTINGS_PATH  = Join-Path $root "gpuav"
$env:TAA_VELOCITY            = "1"
$env:TAA_LAYER_TRACE         = "1"
$env:TAA_KEEP_ON_777         = "1"

Remove-Item (Join-Path $root "gpuav\gpuav_log.txt") -ErrorAction SilentlyContinue
Start-Process -FilePath (Join-Path $xp "X-Plane.exe") -WorkingDirectory $xp
Write-Host "launched with GPU-AV (log: MotionVectors\gpuav\gpuav_log.txt)"
