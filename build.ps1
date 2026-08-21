param([switch]$Installer, [switch]$Dev)

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

# ---- -Dev PRODUCES A DEVELOPER BUILD, AND SAYS SO IN THE VERSION.
#
# The launcher opens its developer surfaces - the Advanced tab, the debug
# console, the layer log, and the unfinished backends as SELECTABLE rather than
# merely visible - when its version string contains "dev".
#
# Deciding that from the version rather than a separate define is deliberate. A
# binary that reports 0.0.16 and behaves like a development build is the same
# class of problem as a launcher that reported 0.0.08 while the build was 0.0.16:
# two facts about one binary that disagree. With one string deciding both, a
# build that claims to be a release IS one, and there is nothing to keep in step.
if ($Dev) { $mvVersion = "$mvVersion-dev" }

Write-Host "Motion Vectors $mvVersion"

# ---- WHICH VENDOR UPSCALER SDKs THIS BUILD CAN SEE.
#
# Detected from the filesystem, never hardcoded. upscaler_policy.h deliberately
# leaves MV_HAVE_* to the build system precisely so a backend cannot CLAIM to
# be present without the code actually being there - editing a constant in a
# header would be a one-line lie that survives all the way to the UI.
#
# Each probe looks for the specific file the compile would fail without, not
# merely for the directory. The FSR 2 and DLSS archives that landed here first
# were prebuilt samples: they had the directory, the DLLs and even the Vulkan
# import library, and zero API headers. A directory-existence check would have
# reported both as available and then failed at the first #include.
$sdkRoot   = Join-Path $root "third_party"
$haveFsr2  = Test-Path (Join-Path $sdkRoot "FSR2\vk\ffx_fsr2_vk.h")
$haveDlss  = Test-Path (Join-Path $sdkRoot "DLSS\nvsdk_ngx_vk.h")
$haveXess  = Test-Path (Join-Path $sdkRoot "XeSS\xess\xess_vk.h")

$sdkDefines = @()
if ($haveFsr2) { $sdkDefines += "-DMV_HAVE_FSR2_SDK=1" }
if ($haveDlss) { $sdkDefines += "-DMV_HAVE_DLSS_SDK=1" }
if ($haveXess) { $sdkDefines += "-DMV_HAVE_XESS_SDK=1" }

$sdkIncludes = @()
if ($haveFsr2) { $sdkIncludes += "-I`"$sdkRoot\FSR2`"" }
if ($haveDlss) { $sdkIncludes += "-I`"$sdkRoot\DLSS`"" }
if ($haveXess) { $sdkIncludes += "-I`"$sdkRoot\XeSS`"" }

Write-Host ("  SDKs: FSR2={0} DLSS={1} XeSS={2}" -f `
            $(if ($haveFsr2) { "yes" } else { "no" }),
            $(if ($haveDlss) { "yes" } else { "no" }),
            $(if ($haveXess) { "yes" } else { "no" }))
if ($Dev) { Write-Host "  DEVELOPER BUILD - launcher shows all dev surfaces" -ForegroundColor Cyan }

$src   = Join-Path $root "src"
$out   = Join-Path $root "build"
$vksdk = (Get-ChildItem "C:\VulkanSDK\*" -Directory | Sort-Object Name | Select-Object -Last 1).FullName

New-Item -ItemType Directory -Force $out | Out-Null
New-Item -ItemType Directory -Force (Join-Path $out "vklayer") | Out-Null


Write-Host "Building plugin..."
& g++ -shared -o "$out\MotionVectors.xpl" "$src\plugin.cpp" `
  -I"$root\SDK\CHeaders\XPLM" -DIBM=1 "-DMV_VERSION=\`"$mvVersion\`"" -m64 -O2 -std=c++17 `
  @sdkDefines @sdkIncludes `
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
  @sdkDefines @sdkIncludes `
  -static -static-libgcc -static-libstdc++
if ($LASTEXITCODE -ne 0) { throw "layer build failed" }
Copy-Item "$src\vklayer\VkLayer_mv.json" "$out\vklayer" -Force

Write-Host "Building launcher..."
# -mwindows: no console window. The launcher sets the two loader variables and
# starts the sim, so the layer is EXPLICIT and is never loaded into any other
# Vulkan application.
& g++ -o "$out\MotionVectorsLauncher.exe" "$src\launcher.cpp" `
  -m64 -O2 -std=c++17 -mwindows -static -static-libgcc -static-libstdc++ -s
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

# ---- THE Qt APPS NEED AN MSVCRT COMPILER. THIS IS NOT A PREFERENCE.
#
# Qt6Core.dll imports msvcrt.dll. The default g++ here is the UCRT variant
# (x86_64-ucrt-posix-seh), so it imports ucrtbase.dll. Those are two different C
# runtimes with two different HEAPS - so a QString allocated inside Qt and
# destroyed in our code calls free() on a pointer that heap never issued, and
# Windows answers "Invalid address specified to RtlFreeHeap".
#
# The symptom is a segfault before the window ever appears, in the constructor,
# on a line as innocent as setWindowTitle. Nothing in the application code is
# wrong and no amount of reading it helps - which is why BOTH Qt apps had it, and
# why the launcher this installer has been shipping had never once run.
#
# Confirmed directly:  objdump -p Qt6Core.dll | grep "DLL Name"  ->  msvcrt.dll
#
# The layer and the plugin stay on the UCRT compiler. They link no Qt, they are
# statically linked, and changing a working toolchain to match an unrelated
# dependency would be the wrong trade.
$qtcxx = @(
    "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.MSVCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin\g++.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if ((Test-Path (Join-Path $qtRoot "bin\windeployqt.exe")) -and -not $qtcxx) {
    Write-Host "Qt found, but no MSVCRT g++ - skipping the Qt apps." -ForegroundColor Yellow
    Write-Host "  Qt6Core.dll imports msvcrt.dll; building against it with the UCRT" -ForegroundColor Yellow
    Write-Host "  compiler produces binaries that crash on the first QString." -ForegroundColor Yellow
    Write-Host "  Install with: winget install BrechtSanders.WinLibs.POSIX.MSVCRT" -ForegroundColor Yellow
} elseif (Test-Path (Join-Path $qtRoot "bin\windeployqt.exe")) {
    Write-Host "Building Qt launcher..."
    $qtOut = Join-Path $out "qtlauncher"
    # Cleared first. windeployqt only ADDS files, so a runtime left behind by a
    # previous Qt version would be shipped alongside the current one.
    if (Test-Path $qtOut) { Remove-Item -Recurse -Force $qtOut }
    New-Item -ItemType Directory -Force $qtOut | Out-Null
    & $qtcxx -o (Join-Path $qtOut "MotionVectors.exe") "$src\qtlauncher\main.cpp" `
      "-I$qtRoot\include" "-I$qtRoot\include\QtCore" "-I$qtRoot\include\QtGui" `
      "-I$qtRoot\include\QtWidgets" "-I$qtRoot\include\QtNetwork" `
      -m64 -O2 -std=c++17 -mwindows -DQT_NO_DEBUG -DNDEBUG "-DMV_VERSION=\`"$mvVersion\`"" `
      "-L$qtRoot\lib" -lQt6Widgets -lQt6Gui -lQt6Core -lQt6Network
    if ($LASTEXITCODE -ne 0) { throw "Qt launcher build failed" }
    # ---- THE DEBUG CONSOLE.
    #
    # Built into the SAME directory as the public app, deliberately: they share
    # a Qt runtime, and windeployqt only has to run once for both. Two
    # directories would mean two ~40 MB copies of Qt for two executables that
    # differ by a few hundred kilobytes.
    #
    # It is a separate EXECUTABLE rather than a tab in the launcher because it
    # runs while the sim is flying, and the launcher's job ends the moment the
    # sim starts.
    Write-Host "Building debug console..."
    & $qtcxx -o (Join-Path $qtOut "MotionVectorsDebug.exe") "$src\qtdebug\main.cpp" `
      "-I$qtRoot\include" "-I$qtRoot\include\QtCore" "-I$qtRoot\include\QtGui" `
      "-I$qtRoot\include\QtWidgets" `
      -m64 -O2 -std=c++17 -mwindows -DQT_NO_DEBUG -DNDEBUG "-DMV_VERSION=\`"$mvVersion\`"" `
      "-L$qtRoot\lib" -lQt6Widgets -lQt6Gui -lQt6Core
    if ($LASTEXITCODE -ne 0) { throw "debug console build failed" }

    & (Join-Path $qtRoot "bin\windeployqt.exe") --release --no-translations `
      (Join-Path $qtOut "MotionVectors.exe") | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "windeployqt failed" }
    # Again for the console: it links QtWidgets/Gui/Core like the launcher, so
    # this adds nothing new today - but running it means a future dependency in
    # only one of the two cannot be missed, which is precisely the failure the
    # stale-launcher comment above records.
    & (Join-Path $qtRoot "bin\windeployqt.exe") --release --no-translations `
      (Join-Path $qtOut "MotionVectorsDebug.exe") | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "windeployqt failed for the debug console" }
} else {
    Write-Host "Qt not found at $qtRoot - skipping the Qt launcher" -ForegroundColor Yellow
}

Write-Host "Installing..."
$xp  = Split-Path $root -Parent
$dst = Join-Path $xp "Resources\plugins\MotionVectors\64"
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item "$out\MotionVectors.xpl" (Join-Path $dst "win.xpl") -Force

# ---- THE LUA PANEL, INSTALLED BY THE BUILD RATHER THAN BY HAND.
#
# This step did not exist, so the panel was only ever updated by remembering to
# copy it - and the installed copy had drifted a full day behind the tree. A
# stale panel is worse than an absent one: it shows settings that have moved and
# reports a version that is not what is running.
#
# MOD_VERSION is STAMPED here from VERSION rather than edited in the Lua, so
# VERSION stays the single source it was made into. A constant kept by hand in a
# second file is the same trap as a hand-kept settings table.
$luaSrc = Join-Path $root "lua\MotionVectors.lua"
$luaDst = Join-Path $xp "Resources\plugins\FlyWithLua\Scripts\MotionVectors.lua"
if (Test-Path $luaSrc) {
    $luaDir = Split-Path $luaDst -Parent
    if (Test-Path $luaDir) {
        $luaText = Get-Content $luaSrc -Raw
        $stamped = [regex]::Replace(
            $luaText,
            'local MOD_VERSION\s*=\s*"[^"]*"',
            ('local MOD_VERSION      = "' + $mvVersion + '"'))
        if ($stamped -eq $luaText -and $luaText -notmatch [regex]::Escape($mvVersion)) {
            Write-Host "  WARNING: MOD_VERSION not found in the panel - version not stamped" -ForegroundColor Yellow
        }
        # NO BOM. Set-Content -Encoding utf8 writes one on PowerShell 5.1,
        # and FlyWithLua's Lua 5.1 does not skip a BOM - it fails to parse the
        # first line and quarantines the script, which looks from the outside
        # like the panel having vanished.
        [System.IO.File]::WriteAllText(
            $luaDst, $stamped, (New-Object System.Text.UTF8Encoding($false)))
        Write-Host "  panel -> $luaDst ($mvVersion)"
    } else {
        Write-Host "  FlyWithLua not installed - skipping the panel" -ForegroundColor Yellow
    }
}


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
