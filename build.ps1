# Build: XPLM plugin + Vulkan layer.
#
# No upscaler feature flags. The predecessor gated FSR2/FSR3/DLSS/Streamline
# behind defines and still carried their code; this project does not contain
# them at all, so there is nothing to gate.

$ErrorActionPreference = "Stop"
$root  = $PSScriptRoot
$src   = Join-Path $root "src"
$out   = Join-Path $root "build"
$vksdk = (Get-ChildItem "C:\VulkanSDK\*" -Directory | Sort-Object Name | Select-Object -Last 1).FullName

New-Item -ItemType Directory -Force $out | Out-Null
New-Item -ItemType Directory -Force (Join-Path $out "vklayer") | Out-Null

Write-Host "Building plugin..."
& g++ -shared -o "$out\MotionVectors.xpl" "$src\plugin.cpp" `
  -I"$root\SDK\CHeaders\XPLM" -DIBM=1 -m64 -O2 -std=c++17 `
  -static -static-libgcc -static-libstdc++ `
  -L"$root\SDK\Libraries\Win" -lXPLM_64
if ($LASTEXITCODE -ne 0) { throw "plugin build failed" }

Write-Host "Building Vulkan layer..."
& g++ -shared -o "$out\vklayer\VkLayer_mv.dll" "$src\vklayer\layer.cpp" `
  -I"$vksdk\Include" -m64 -O2 -std=c++17 `
  -static -static-libgcc -static-libstdc++
if ($LASTEXITCODE -ne 0) { throw "layer build failed" }
Copy-Item "$src\vklayer\VkLayer_mv.json" "$out\vklayer" -Force

Write-Host "Building launcher..."
# -mwindows: no console window. The launcher sets the two loader variables and
# starts the sim, so the layer is EXPLICIT and is never loaded into any other
# Vulkan application.
& g++ -o "$out\MotionVectorsLauncher.exe" "$src\launcher.cpp" `
  -m64 -O2 -std=c++17 -mwindows -static -static-libgcc -static-libstdc++
if ($LASTEXITCODE -ne 0) { throw "launcher build failed" }

Write-Host "Installing..."
$xp  = Split-Path $root -Parent
$dst = Join-Path $xp "Resources\plugins\MotionVectors\64"
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item "$out\MotionVectors.xpl" (Join-Path $dst "win.xpl") -Force

Write-Host "Done." -ForegroundColor Green
