param([switch]$Installer, [switch]$Dev,
      [ValidateSet('MotionVectors','Crash')] [string]$Target = 'MotionVectors')

# ---- WHICH PRODUCT THIS BUILD IS.
#
# One source tree, two mods: Motion Vectors and Realistic Crash Physics. They
# share the Vulkan layer because crash destruction IS the motion-vector
# machinery - the same SPIR-V injector, the same pipeline and descriptor
# interception, the same live config. Forking the tree would mean fixing the
# injector twice.
#
# src/product.h holds the identity; this block holds the file names, and the
# two must agree. Everything below is DERIVED from $Target rather than typed,
# because a build that writes MotionVectors.xpl while compiling the crash
# product surfaces later as "why is the wrong mod loading".
$isCrash = ($Target -eq 'Crash')
if ($isCrash) {
    $prodId     = 'RealisticCrashPhysics'
    $prodTitle  = 'Realistic Crash Physics'
    $xplName    = 'RealisticCrashPhysics.xpl'
    $layerDll   = 'VkLayer_rcp.dll'
    $layerJson  = 'VkLayer_rcp.json'
    $launcher   = 'RealisticCrashPhysicsLauncher.exe'
    $panelLua   = 'RealisticCrashPhysics.lua'
    $prodDefine = '-DMV_PRODUCT_CRASH=1'
} else {
    $prodId     = 'MotionVectors'
    $prodTitle  = 'Motion Vectors'
    $xplName    = 'MotionVectors.xpl'
    $layerDll   = 'VkLayer_mv.dll'
    $layerJson  = 'VkLayer_mv.json'
    $launcher   = 'MotionVectorsLauncher.exe'
    $panelLua   = 'MotionVectors.lua'
    $prodDefine = '-DMV_PRODUCT_CRASH=0'
}

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

Write-Host "$prodTitle $mvVersion"

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
# Per product, so switching targets cannot relink against the other
# one's objects or leave its DLL sitting where the installer looks.
$out   = Join-Path $root (Join-Path "build" $prodId)
$vksdk = (Get-ChildItem "C:\VulkanSDK\*" -Directory | Sort-Object Name | Select-Object -Last 1).FullName

New-Item -ItemType Directory -Force $out | Out-Null
New-Item -ItemType Directory -Force (Join-Path $out "vklayer") | Out-Null


Write-Host "Building plugin..."
& g++ -shared -o "$out\$xplName" "$src\plugin.cpp" $prodDefine `
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

# ---- OUR REPLACEMENT FOR X-PLANE'S FSR UPSCALE.
#
# Compiled the same way and for the same reason as the resolve above: this
# header was written by hand once, and a hand-written generated header is one
# that silently stops tracking its source.
#
# The interface is not free to change. It must match X-Plane's EASU module
# exactly - set 0 bindings 0-3, both images ARRAYED, Rgba16f output,
# LocalSize 64 1 1 - because X-Plane binds its own resources against it. Verify
# with spirv-dis after any edit; a mismatch binds real resources to the wrong
# slots, which shows up as a rendering artefact rather than an error.
Write-Host "Compiling X-Plane FSR replacement..."
$fsrTmp = Join-Path $env:TEMP "xpfsr_replace.spv"
& $glslang -V --target-env vulkan1.2 -S comp "$src\shaders\xpfsr_replace.comp" -o $fsrTmp | Out-Null
if ($LASTEXITCODE -ne 0) { throw "xpfsr_replace.comp failed to compile" }
$fbytes = [System.IO.File]::ReadAllBytes($fsrTmp)
if ($fbytes.Length % 4) { throw "FSR replacement SPIR-V is not a whole number of words" }
$fwords = New-Object System.Collections.Generic.List[string]
for ($i = 0; $i -lt $fbytes.Length; $i += 4) {
    $fwords.Add(("0x{0:x8}u" -f [System.BitConverter]::ToUInt32($fbytes, $i)))
}
$fsb = New-Object System.Text.StringBuilder
[void]$fsb.AppendLine("// Generated from src/shaders/xpfsr_replace.comp by build.ps1 - do not edit.")
[void]$fsb.AppendLine("#pragma once")
[void]$fsb.AppendLine("#include <stdint.h>")
[void]$fsb.AppendLine("")
[void]$fsb.AppendLine("static const uint32_t kXpFsrReplaceSpv[] = {")
for ($i = 0; $i -lt $fwords.Count; $i += 8) {
    $n = [Math]::Min(8, $fwords.Count - $i)
    [void]$fsb.AppendLine("    " + (($fwords.GetRange($i, $n)) -join ",") + ",")
}
[void]$fsb.AppendLine("};")
[void]$fsb.AppendLine("")
[void]$fsb.AppendLine("static const size_t kXpFsrReplaceSpvWords = $($fwords.Count);")
Set-Content -Path "$src\vklayer\xpfsr_spv.h" -Value $fsb.ToString() -Encoding utf8 -NoNewline
Write-Host "  xpfsr_spv.h: $($fwords.Count) words"

# ---- THE FIDELITYFX OBJECTS.
#
# Compiled once and reused. ffx_fsr3upscaler_shaderblobs.cpp alone is 1.5 MB of
# generated SPIR-V and is slow; rebuilding it on every layer edit would make the
# normal loop slower for nothing. Delete build\ffx_obj to force a rebuild after
# regenerating shaders.
#
# THREE FORCED INCLUDES. MSVC pulls <new>, <cmath> and <cstring> in
# transitively and the SDK relies on that, so placement new, log2/floor and
# memset/memcmp are otherwise undeclared under GCC. swprintf_s is mapped to
# _snwprintf for the same reason.
#
# -DFFX_FSR3UPSCALER selects the upscaler alone: ffx_shader_blobs.cpp is guarded
# per component, so without it every effect in the SDK would be pulled in and
# none of their shader headers exist here.
$ffxSdk = Join-Path $root "third_party\FidelityFX-SDK"
$ffxObj = Join-Path $root "build\ffx_obj"
$haveFfx = Test-Path (Join-Path $ffxSdk "sdk\src\backends\vk\ffx_vk.cpp")
if ($haveFfx -and -not (Test-Path (Join-Path $ffxObj "ffx_vk.o"))) {
    Write-Host "Building FidelityFX (once)..."
    New-Item -ItemType Directory -Force $ffxObj | Out-Null
    $ffxInc = @(
        "-I$ffxSdk\sdk\include", "-I$ffxSdk\sdk\src\backends\vk",
        "-I$ffxSdk\sdk\src\backends\shared",
        "-I$ffxSdk\sdk\src\backends\shared\blob_accessors",
        "-I$ffxSdk\sdk\src\components", "-I$ffxSdk\sdk\src",
        "-I$ffxSdk\sdk\src\shared", "-I$root\build\ffx_shaders",
        "-I$vksdk\Include")
    $ffxSrc = @(
        "$ffxSdk\sdk\src\backends\vk\ffx_vk.cpp",
        "$ffxSdk\sdk\src\components\fsr3upscaler\ffx_fsr3upscaler.cpp",
        "$ffxSdk\sdk\src\backends\shared\ffx_shader_blobs.cpp",
        "$ffxSdk\sdk\src\backends\shared\blob_accessors\ffx_fsr3upscaler_shaderblobs.cpp",
        "$ffxSdk\sdk\src\shared\ffx_assert.cpp",
        "$ffxSdk\sdk\src\shared\ffx_breadcrumbs_list.cpp",
        "$ffxSdk\sdk\src\shared\ffx_message.cpp",
        "$ffxSdk\sdk\src\shared\ffx_object_management.cpp",
        "$src\vklayer\ffx_fg_stub.cpp",
        "$src\vklayer\ffx_vk_shim.cpp")
    foreach ($f in $ffxSrc) {
        if (-not (Test-Path $f)) { continue }
        $o = Join-Path $ffxObj ((Split-Path $f -Leaf) -replace '\.cpp$', '.o')
        & g++ -c -O2 -std=c++17 -include new -include cmath -include cstring `
            -include cwchar -include cstdio $ffxInc `
            -DFFX_VK=1 -DFFX_FSR3UPSCALER -DWIN32 "-Dswprintf_s=_snwprintf" `
            $f -o $o
        if ($LASTEXITCODE -ne 0) { throw "FidelityFX: $(Split-Path $f -Leaf) failed to compile" }
    }
    Write-Host "  $((Get-ChildItem $ffxObj -Filter *.o).Count) FidelityFX objects"
}
$ffxObjs = @()
$ffxDefine = ""
if ($haveFfx -and (Test-Path (Join-Path $ffxObj "ffx_vk.o"))) {
    $ffxObjs = (Get-ChildItem $ffxObj -Filter *.o | ForEach-Object { $_.FullName })
    $ffxDefine = "-DMV_HAVE_FSR3=1"
    # ---- HIDE THE SHIM SYMBOLS FROM THE DLL EXPORT TABLE.
    #
    # ffx_vk_shim.cpp defines the Vulkan entry points FidelityFX calls by name,
    # and MinGW exports every symbol from a DLL by default. Several of those
    # names ARE the Vulkan layer interface - the loader calls
    # vkEnumerateDeviceExtensionProperties and vkGetDeviceProcAddr ON A LAYER -
    # so exporting them handed the loader our FFX forwarders as though they were
    # this layer's own entry points. It took X-Plane down inside vulkan-1.dll
    # with 0xc0000409, the loader __fastfail this project has met before.
    #
    # They must still exist for the internal link, so they are excluded from the
    # EXPORT TABLE rather than removed. MV_GetInstanceProcAddr and
    # MV_GetDeviceProcAddr, which the manifest names, are untouched.
    $ffxHide = @(
        "-Wl,--exclude-symbols=vkGetDeviceProcAddr",
        "-Wl,--exclude-symbols=vkEnumerateDeviceExtensionProperties",
        "-Wl,--exclude-symbols=vkGetPhysicalDeviceProperties",
        "-Wl,--exclude-symbols=vkGetPhysicalDeviceProperties2",
        "-Wl,--exclude-symbols=vkGetPhysicalDeviceFeatures",
        "-Wl,--exclude-symbols=vkGetPhysicalDeviceFeatures2",
        "-Wl,--exclude-symbols=vkGetPhysicalDeviceMemoryProperties",
        "-Wl,--exclude-symbols=vkCreateBuffer")
    # ---- THE LOADER IMPORT LIBRARY, FOR A FEW INSTANCE CALLS ONLY.
    #
    # ffx_vk.cpp calls vkGetPhysicalDeviceProperties2 and friends directly while
    # building its interface. Every PER-FRAME call it makes goes through the
    # vkGetDeviceProcAddr we hand it, which is the NEXT LAYER's - so the frame
    # path never re-enters the loader. These few are capability queries made
    # once at context creation, well away from any nested loader call, which is
    # the situation the XeSS probe crash warned about.
    $ffxVkLib = Join-Path $vksdk ("Lib" + [char]92 + "vulkan-1.lib")
}

# ---- NO vulkan-1 IMPORT LIBRARY ON THIS LINK.
#
# ffx_vk_shim.cpp defines the three Vulkan entry points FidelityFX calls by
# name. Linking the loader's exports alongside them would let link order decide
# which wins, and the loader's version means a call made from inside this layer
# re-enters the dispatch chain AT THE TOP - straight back into our own hook.
# That recursion killed the sim inside ffxGetScratchMemorySizeVK, FSR3's very
# first call, with no output at all.
Write-Host "Building Vulkan layer..."
& g++ -shared -o "$out\vklayer\$layerDll" "$src\vklayer\layer.cpp" $prodDefine `
  -I"$vksdk\Include" -m64 -O2 -std=c++17 `
  @sdkDefines @sdkIncludes $ffxDefine `
  "-I$ffxSdk\sdk\include" @ffxObjs $ffxHide `
  -static -static-libgcc -static-libstdc++
if ($LASTEXITCODE -ne 0) { throw "layer build failed" }
# GENERATED, not copied. The manifest names both the layer and its
# library, and a crash build shipping a manifest that says VK_LAYER_mv would
# have the loader hand our layer the other mod's name - or refuse both.
$mfSrc = Get-Content "$src\vklayer\VkLayer_mv.json" -Raw
if ($isCrash) {
    $mfSrc = $mfSrc -replace '"VK_LAYER_mv"', '"VK_LAYER_rcp"'
    $mfSrc = $mfSrc -replace 'VkLayer_mv\.dll', 'VkLayer_rcp.dll'
    $mfSrc = $mfSrc -replace '"description"\s*:\s*"[^"]*"',
                             '"description": "Realistic Crash Physics"'
}
[System.IO.File]::WriteAllText((Join-Path "$out\vklayer" $layerJson), $mfSrc,
                               (New-Object System.Text.UTF8Encoding($false)))

Write-Host "Building launcher..."
# -mwindows: no console window. The launcher sets the two loader variables and
# starts the sim, so the layer is EXPLICIT and is never loaded into any other
# Vulkan application.
& g++ -o "$out\$launcher" "$src\launcher.cpp" $prodDefine `
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
# Set by the catch around the Qt build; the installer gate near the end of the
# script reads it, so a build that could not produce the launcher does not go
# on to package an installer that silently lacks one.
$qtFailed = $false
$qtOut    = $null
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
  # ---- A COSMETIC APP MUST NOT BLOCK DEPLOYING THE MOD.
  #
  # These four steps used to throw straight out of the script, and everything
  # below - including "Installing..." - is downstream of them. So a Qt
  # toolchain problem left the freshly built .xpl and layer DLL sitting in
  # build\ while the sim went on loading the previous ones, and the build
  # reported a failure that looked like it was about the launcher.
  #
  # That cost a full test cycle: the plugin was rebuilt with a corrected
  # airframe box, the sim was restarted, and the log still printed the OLD
  # box - which reads as "the fix did not work" rather than "the fix was never
  # installed". The two are indistinguishable from the log alone.
  #
  # So Qt failures degrade to a warning here. The launcher is a convenience
  # around a mod that runs perfectly well without it; the layer and the plugin
  # are the product.
  try {
    # ---- A WARNING ON STDERR IS NOT A FAILED BUILD.
    #
    # $ErrorActionPreference is "Stop" for this whole script, and in PowerShell
    # 5.1 that turns ANY stderr output from a native executable into a
    # terminating NativeCommandError - regardless of the exit code. GCC 16
    # added -Wsfinae-incomplete, Qt 6.8.3's headers trip it, and from that day
    # every build reported
    #
    #   Qt launcher FAILED to build - continuing without it.
    #   In file included from .../QtCore/qstring.h:23,
    #
    # while g++ was returning 0 and producing a working 177 KB binary. The
    # message even quoted the first WARNING line as though it were the error.
    #
    # The cost was not cosmetic: the installer is deliberately skipped when the
    # launcher is missing, so this silently blocked every release.
    #
    # Exit codes are the truth here, and they are already checked after each
    # step below. Restored in the catch and after the block so nothing else
    # loses "Stop".
    $qtEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
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
      -Wno-sfinae-incomplete `
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
      -Wno-sfinae-incomplete `
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
    $ErrorActionPreference = $qtEap
  } catch {
    if ($qtEap) { $ErrorActionPreference = $qtEap }
    # The output directory is REMOVED rather than left half-built. The comment
    # above records that this folder was once shipped stale for months because
    # nothing regenerated it; a partial build left here would be shipped the
    # same way, and a launcher that starts and then fails is worse than one
    # that is plainly absent.
    $qtFailed = $true
    if ($qtOut -and (Test-Path $qtOut)) { Remove-Item -Recurse -Force $qtOut }
    Write-Host "Qt launcher FAILED to build - continuing without it." -ForegroundColor Yellow
    Write-Host "  $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "  The layer and the plugin are unaffected and are still being installed." -ForegroundColor Yellow
    Write-Host "  The installer will NOT be built, because shipping one without" -ForegroundColor Yellow
    Write-Host "  the launcher is a release defect rather than a local inconvenience." -ForegroundColor Yellow
  }
} else {
    Write-Host "Qt not found at $qtRoot - skipping the Qt launcher" -ForegroundColor Yellow
}

Write-Host "Installing..."
$xp  = Split-Path $root -Parent
# PER PRODUCT. This was hardcoded to MotionVectors, so building the crash
# target installed a crash plugin over the motion-vector one and left the
# sim loading the wrong mod under the right name.
$dst = Join-Path $xp "Resources\plugins\$prodId\64"
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item "$out\$xplName" (Join-Path $dst "win.xpl") -Force

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
$luaSrc = Join-Path $root "lua\$panelLua"
$luaDst = Join-Path $xp "Resources\plugins\FlyWithLua\Scripts\$panelLua"
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
    # A local build tolerates a missing launcher; a RELEASE does not. Refusing
    # here is the whole reason the Qt failure above is allowed to be a warning:
    # the inconvenience is absorbed during development and stopped at the point
    # where it would reach someone else.
    if ($qtFailed) {
        throw ("the Qt launcher failed to build, so this tree cannot produce a " +
               "complete installer. Fix the Qt build, or run without -Installer " +
               "to install locally.")
    }
    $iscc = Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"
    if (-not (Test-Path $iscc)) { throw "Inno Setup not found at $iscc" }
    Write-Host "Building installer..."
    & $iscc "/DAppVersion=$mvVersion" (Join-Path $root "installer.iss") | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "installer build failed" }
    Write-Host "  dist\MotionVectors-$mvVersion-setup.exe" -ForegroundColor Green
}

Write-Host "Done." -ForegroundColor Green
