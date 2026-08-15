param([switch]$Installer)

# Build: XPLM plugin + Vulkan layer.
#
# No upscaler feature flags. The predecessor gated FSR2/FSR3/DLSS/Streamline
# behind defines and still carried their code; this project does not contain
# them at all, so there is nothing to gate.

$ErrorActionPreference = "Stop"
$root  = $PSScriptRoot

# ---- ONE SOURCE OF TRUTH FOR THE VERSION.
#
# It used to live in installer.iss alone, so the installer said 0.0.08 while
# the tag said 0.0.11 and the plugin said nothing at all. A version that
# disagrees with itself across three files is how a user ends up reporting a
# bug against a build nobody can identify.
#
# VERSION is the file. Everything else reads it: the compiler takes it as a
# define, the installer takes it on the command line, and the Lua panel reads
# it back out of the plugin over a dataref.
$mvVersion = (Get-Content (Join-Path $root "VERSION") -Raw).Trim()
Write-Host "Motion Vectors $mvVersion"

$src   = Join-Path $root "src"
$out   = Join-Path $root "build"
$vksdk = (Get-ChildItem "C:\VulkanSDK\*" -Directory | Sort-Object Name | Select-Object -Last 1).FullName

New-Item -ItemType Directory -Force $out | Out-Null
New-Item -ItemType Directory -Force (Join-Path $out "vklayer") | Out-Null


Write-Host "Building plugin..."
& g++ -shared -o "$out\MotionVectors.xpl" "$src\plugin.cpp" `
  -I"$root\SDK\CHeaders\XPLM" -DIBM=1 "-DMV_VERSION=\`"$mvVersion\`"" -m64 -O2 -std=c++17 `
  -static -static-libgcc -static-libstdc++ `
  -L"$root\SDK\Libraries\Win" -lXPLM_64
if ($LASTEXITCODE -ne 0) { throw "plugin build failed" }

# ---- THE TAA SHADER, COMPILED FROM ITS SOURCE EVERY BUILD.
#
# src/vklayer/taa_spv.h opens with "Generated from src/shaders/taa.comp by
# build.ps1 - do not edit", and until now this script did no such thing. The
# header was produced by hand once and every later edit to taa.comp changed
# nothing at all, because the layer compiles the HEADER. A source file that is
# not built is worse than one that does not exist: it reads as the truth.
#
# This is the same failure as the Qt launcher two blocks down - shipped from a
# directory nothing regenerated - and the same fix. Regenerating costs a
# fraction of a second.
Write-Host "Compiling TAA shader..."
$glslang = Join-Path $vksdk "Bin\glslangValidator.exe"
if (-not (Test-Path $glslang)) { throw "glslangValidator not found at $glslang" }
$spvTmp = Join-Path $env:TEMP "taa_resolve.spv"
& $glslang -V --target-env vulkan1.2 -S comp "$src\shaders\taa.comp" -o $spvTmp
if ($LASTEXITCODE -ne 0) { throw "TAA shader compile failed" }

# spirv-val separately: glslangValidator accepts things the validator rejects,
# and a module that fails validation fails at pipeline creation instead - inside
# the sim, as a crash with no shader named.
$spvval = Join-Path $vksdk "Bin\spirv-val.exe"
if (Test-Path $spvval) {
    & $spvval $spvTmp
    if ($LASTEXITCODE -ne 0) { throw "TAA shader failed spirv-val" }
}

$bytes = [System.IO.File]::ReadAllBytes($spvTmp)
if ($bytes.Length % 4) { throw "TAA SPIR-V is not a whole number of words" }
$words = New-Object System.Collections.Generic.List[string]
for ($i = 0; $i -lt $bytes.Length; $i += 4) {
    $w = [System.BitConverter]::ToUInt32($bytes, $i)
    $words.Add(("0x{0:x8}u" -f $w))
}
$sb = New-Object System.Text.StringBuilder
[void]$sb.AppendLine("// Generated from src/shaders/taa.comp by build.ps1 - do not edit.")
[void]$sb.AppendLine("#pragma once")
[void]$sb.AppendLine("#include <stdint.h>")
[void]$sb.AppendLine("")
[void]$sb.AppendLine("static const uint32_t kTaaResolveSpv[] = {")
for ($i = 0; $i -lt $words.Count; $i += 8) {
    $n = [Math]::Min(8, $words.Count - $i)
    [void]$sb.AppendLine("    " + (($words.GetRange($i, $n)) -join ",") + ",")
}
[void]$sb.AppendLine("};")
[void]$sb.AppendLine("")
[void]$sb.AppendLine("static const size_t kTaaResolveSpvWords = $($words.Count);")
Set-Content -Path "$src\vklayer\taa_spv.h" -Value $sb.ToString() -Encoding utf8 -NoNewline
Write-Host "  taa_spv.h: $($words.Count) words"

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


# ---- THE Qt LAUNCHER.
#
# This was never built by this script. src\qtlauncher\main.cpp existed, the
# installer shipped build\qtlauncher\* unconditionally, and nothing in
# between regenerated it - so every installer since that folder was first
# populated by hand has carried a STALE launcher, built from a source nobody
# had compiled in the meantime. It is exactly the failure the VERSION file was
# just introduced to prevent, one directory over.
#
# windeployqt is what makes the output runnable: Qt will not start without
# platforms\qwindows.dll, and a missing plugin folder fails at run time on the
# user's machine rather than here.
$qtRoot = "C:\Qt\6.8.3\mingw_64"
if (Test-Path (Join-Path $qtRoot "bin\windeployqt.exe")) {
    Write-Host "Building Qt launcher..."
    $qtOut = Join-Path $out "qtlauncher"
    # Cleared first. windeployqt only ADDS files, so a runtime left behind by a
    # previous Qt version would be shipped alongside the current one.
    if (Test-Path $qtOut) { Remove-Item -Recurse -Force $qtOut }
    New-Item -ItemType Directory -Force $qtOut | Out-Null
    & g++ -o (Join-Path $qtOut "MotionVectors.exe") "$src\qtlauncher\main.cpp" `
      "-I$qtRoot\include" "-I$qtRoot\include\QtCore" "-I$qtRoot\include\QtGui" `
      "-I$qtRoot\include\QtWidgets" "-I$qtRoot\include\QtNetwork" `
      -m64 -O2 -std=c++17 -mwindows "-DMV_VERSION=\`"$mvVersion\`"" `
      "-L$qtRoot\lib" -lQt6Widgets -lQt6Gui -lQt6Core -lQt6Network -lQt6Network
    if ($LASTEXITCODE -ne 0) { throw "Qt launcher build failed" }
    & (Join-Path $qtRoot "bin\windeployqt.exe") --release --no-translations `
      (Join-Path $qtOut "MotionVectors.exe") | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "windeployqt failed" }
} else {
    Write-Host "Qt not found at $qtRoot - skipping the Qt launcher" -ForegroundColor Yellow
}

Write-Host "Installing..."
$xp  = Split-Path $root -Parent
$dst = Join-Path $xp "Resources\plugins\MotionVectors\64"
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item "$out\MotionVectors.xpl" (Join-Path $dst "win.xpl") -Force


# ---- THE INSTALLER, BUILT FROM THE SAME TREE AS THE BINARIES.
#
# On request only: it takes several seconds and most iterations do not need
# one. Every release does.
if ($Installer) {
    $iscc = Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"
    if (-not (Test-Path $iscc)) { throw "Inno Setup not found at $iscc" }
    Write-Host "Building installer..."
    & $iscc "/DAppVersion=$mvVersion" (Join-Path $root "installer.iss") | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "installer build failed" }
    Write-Host "  dist\MotionVectors-$mvVersion-setup.exe" -ForegroundColor Green
}

Write-Host "Done." -ForegroundColor Green
