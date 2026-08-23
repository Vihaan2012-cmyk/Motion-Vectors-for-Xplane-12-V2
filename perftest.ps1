# Present-path A/B. TAA_PRESENT_MODE: 0 IMMEDIATE, 1 MAILBOX, 2 FIFO(vsync).
# IMMEDIATE never blocks on the display, so if the 15 ms present block is the
# display path this removes it; if the block survives, the CPU is really
# waiting for GPU work to drain and the display is innocent.
param([int]$Mode = 0, [int]$Images = 3)
$root = (Get-Location).Path
$xp   = Split-Path $root -Parent
$env:VK_LAYER_PATH           = Join-Path $root "build\vklayer"
$env:VK_LOADER_LAYERS_ENABLE = "VK_LAYER_mv"
$env:TAA_VELOCITY            = "1"
$env:TAA_LAYER_TRACE         = "1"
$env:TAA_PRESENT_MODE        = "$Mode"
$env:TAA_SWAP_IMAGES         = "$Images"
Start-Process -FilePath (Join-Path $xp "X-Plane.exe") -WorkingDirectory $xp
Write-Host "launched: present mode $Mode, swap images $Images"
