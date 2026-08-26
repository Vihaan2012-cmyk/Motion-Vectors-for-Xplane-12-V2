# A/B the texture pager's per-frame budget.
#
# Our own VRAM system holds tex/paging/max_overdrive at 64 against X-Plane's
# stock 16. That control governs how far past its budget the pager may run in a
# frame, i.e. how much uploading it does - and the layer measures 64 MB/frame of
# uploads, which is resolution-independent and matches the resolution-
# independent 38 fps ceiling. 16 restores stock.
param([int]$Overdrive = 16)
$root = (Get-Location).Path
$xp   = Split-Path $root -Parent
$env:VK_LAYER_PATH           = Join-Path $root "build\vklayer"
$env:VK_LOADER_LAYERS_ENABLE = "VK_LAYER_mv"
$env:TAA_VELOCITY            = "1"
$env:TAA_LAYER_TRACE         = "1"
$env:TAA_MAX_OVERDRIVE       = "$Overdrive"
Remove-Item Env:TAA_PRESENT_MODE -ErrorAction SilentlyContinue
Remove-Item Env:TAA_SWAP_IMAGES  -ErrorAction SilentlyContinue
Start-Process -FilePath (Join-Path $xp "X-Plane.exe") -WorkingDirectory $xp
Write-Host "launched: max_overdrive = $Overdrive (stock is 16, we ship 64)"
