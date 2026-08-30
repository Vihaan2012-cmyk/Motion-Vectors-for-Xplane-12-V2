param([switch]$Installer, [switch]$Dev)

# Build: XPLM plugin + Vulkan layer.
#
# No upscaler feature flags. The predecessor gated FSR2/FSR3/DLSS/Streamline
# behind defines and still carried their code; this project does not contain
# them at all, so there is nothing to gate.

$ErrorActionPreference = "Stop"
# Set if the optional Qt launcher fails; reported at the end so it stays visible
# without aborting the build before the plugin is installed.
$script:qtLauncherFailed = $false
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
if ($Dev) { Write-Host "  DEVELOPER BUILD - launcher shows all dev surfaces" -ForegroundColor Cyan }

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

# ---- THE FRAME-INTERPOLATION PREPARE SHADER.
#
# Produces the three textures FFX frame interpolation needs - dilatedDepth,
# dilatedMotionVectors and reconstructedPrevNearestDepth - which were previously
# harvested as a by-product of running the entire FSR3 upscaler and discarding
# its output. Generated from source for the same reason as the two above.
Write-Host "Compiling frame-gen prepare shader..."
$fgTmp = Join-Path $env:TEMP "fg_prepare.spv"
& $glslang -V --target-env vulkan1.2 -S comp "$src/shaders/fg_prepare.comp" -o $fgTmp | Out-Null
if ($LASTEXITCODE -ne 0) { throw "fg_prepare.comp failed to compile" }
if (Test-Path $spvval) {
    & $spvval $fgTmp | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "fg_prepare.comp failed spirv-val" }
}
$gbytes = [System.IO.File]::ReadAllBytes($fgTmp)
if ($gbytes.Length % 4) { throw "fg_prepare SPIR-V is not a whole number of words" }
$gwords = New-Object System.Collections.Generic.List[string]
for ($i = 0; $i -lt $gbytes.Length; $i += 4) {
    $gwords.Add(("0x{0:x8}u" -f [System.BitConverter]::ToUInt32($gbytes, $i)))
}
$gsb = New-Object System.Text.StringBuilder
[void]$gsb.AppendLine("// Generated from src/shaders/fg_prepare.comp by build.ps1 - do not edit.")
[void]$gsb.AppendLine("#pragma once")
[void]$gsb.AppendLine("#include <stdint.h>")
[void]$gsb.AppendLine("")
[void]$gsb.AppendLine("static const uint32_t kFgPrepareSpv[] = {")
for ($i = 0; $i -lt $gwords.Count; $i += 8) {
    $n = [Math]::Min(8, $gwords.Count - $i)
    [void]$gsb.AppendLine("    " + (($gwords.GetRange($i, $n)) -join ",") + ",")
}
[void]$gsb.AppendLine("};")
[void]$gsb.AppendLine("")
[void]$gsb.AppendLine("static const size_t kFgPrepareSpvWords = $($gwords.Count);")
[IO.File]::WriteAllText("$src/vklayer/fg_prepare_spv.h", $gsb.ToString())
Write-Host "  fg_prepare_spv.h: $($gwords.Count) words"

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
# ---- REBUILD WHEN THE SOURCE LIST GROWS, NOT ONLY WHEN ffx_vk.o IS ABSENT.
#
# This used to gate on ffx_vk.o alone, so the block ran exactly once ever.
# Adding a source then did nothing at all: the new .cpp was never compiled,
# the link failed on a symbol from a file that had never been built, and the
# build still printed "Done" for everything up to the link. That cost two
# rebuild cycles to spot, because a stale object directory looks identical to
# a correct one.
#
# The block now always runs, and the per-file check in the loop below
# skips anything whose object is already newer than its source - so adding
# a source compiles it, and an unchanged tree costs nothing.
if ($haveFfx) {
    Write-Host "Compiling GI gather shader..."
$giTmp = Join-Path $env:TEMP "gi_gather.spv"
& $glslang -V --target-env vulkan1.2 -S comp "$src/shaders/gi_gather.comp" -o $giTmp | Out-Null
if ($LASTEXITCODE -ne 0) { throw "gi_gather.comp failed to compile" }
if (Test-Path $spvval) {
    & $spvval $giTmp | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "gi_gather.comp failed spirv-val" }
}
$gibytes = [System.IO.File]::ReadAllBytes($giTmp)
$giwords = New-Object System.Collections.Generic.List[string]
for ($i = 0; $i -lt $gibytes.Length; $i += 4) {
    $giwords.Add(("0x{0:x8}u" -f [System.BitConverter]::ToUInt32($gibytes, $i)))
}
$gisb = New-Object System.Text.StringBuilder
[void]$gisb.AppendLine("// Generated from src/shaders/gi_gather.comp by build.ps1 - do not edit.")
[void]$gisb.AppendLine("#pragma once")
[void]$gisb.AppendLine("#include <stdint.h>")
[void]$gisb.AppendLine("")
[void]$gisb.AppendLine("static const uint32_t kGiGatherSpv[] = {")
for ($i = 0; $i -lt $giwords.Count; $i += 8) {
    $n = [Math]::Min(8, $giwords.Count - $i)
    [void]$gisb.AppendLine("    " + (($giwords.GetRange($i, $n)) -join ",") + ",")
}
[void]$gisb.AppendLine("};")
[IO.File]::WriteAllText("$src/vklayer/gi_gather_spv.h", $gisb.ToString())
Write-Host ("  gi_gather_spv.h: {0} words" -f $giwords.Count)

Write-Host "Compiling TAAU shader..."
$tuTmp = Join-Path $env:TEMP "taau.spv"
& $glslang -V --target-env vulkan1.2 -S comp "$src/shaders/taau.comp" -o $tuTmp | Out-Null
if ($LASTEXITCODE -ne 0) { throw "taau.comp failed to compile" }
if (Test-Path $spvval) {
    & $spvval $tuTmp | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "taau.comp failed spirv-val" }
}
$tubytes = [System.IO.File]::ReadAllBytes($tuTmp)
$tuwords = New-Object System.Collections.Generic.List[string]
for ($i = 0; $i -lt $tubytes.Length; $i += 4) {
    $tuwords.Add(("0x{0:x8}u" -f [System.BitConverter]::ToUInt32($tubytes, $i)))
}
$tusb = New-Object System.Text.StringBuilder
[void]$tusb.AppendLine("// Generated from src/shaders/taau.comp by build.ps1 - do not edit.")
[void]$tusb.AppendLine("#pragma once")
[void]$tusb.AppendLine("#include <stdint.h>")
[void]$tusb.AppendLine("")
[void]$tusb.AppendLine("static const uint32_t kTaauSpv[] = {")
for ($i = 0; $i -lt $tuwords.Count; $i += 8) {
    $n = [Math]::Min(8, $tuwords.Count - $i)
    [void]$tusb.AppendLine("    " + (($tuwords.GetRange($i, $n)) -join ",") + ",")
}
[void]$tusb.AppendLine("};")
[IO.File]::WriteAllText("$src/vklayer/taau_spv.h", $tusb.ToString())
Write-Host ("  taau_spv.h: {0} words" -f $tuwords.Count)

Write-Host "Compiling oracle sun-dump shader..."
$sdTmp = Join-Path $env:TEMP "oracle_sundump.spv"
& $glslang -V --target-env vulkan1.2 -S comp "$src/shaders/oracle_sundump.comp" -o $sdTmp | Out-Null
if ($LASTEXITCODE -ne 0) { throw "oracle_sundump.comp failed to compile" }
if (Test-Path $spvval) {
    & $spvval $sdTmp | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "oracle_sundump.comp failed spirv-val" }
}
$sdbytes = [System.IO.File]::ReadAllBytes($sdTmp)
$sdwords = New-Object System.Collections.Generic.List[string]
for ($i = 0; $i -lt $sdbytes.Length; $i += 4) {
    $sdwords.Add(("0x{0:x8}u" -f [System.BitConverter]::ToUInt32($sdbytes, $i)))
}
$sdsb = New-Object System.Text.StringBuilder
[void]$sdsb.AppendLine("// Generated from src/shaders/oracle_sundump.comp by build.ps1 - do not edit.")
[void]$sdsb.AppendLine("#pragma once")
[void]$sdsb.AppendLine("#include <stdint.h>")
[void]$sdsb.AppendLine("")
[void]$sdsb.AppendLine("static const uint32_t kOracleSunDumpSpv[] = {")
for ($i = 0; $i -lt $sdwords.Count; $i += 8) {
    $n = [Math]::Min(8, $sdwords.Count - $i)
    [void]$sdsb.AppendLine("    " + (($sdwords.GetRange($i, $n)) -join ",") + ",")
}
[void]$sdsb.AppendLine("};")
[IO.File]::WriteAllText("$src/vklayer/oracle_sundump_spv.h", $sdsb.ToString())
Write-Host ("  oracle_sundump_spv.h: {0} words" -f $sdwords.Count)

Write-Host "Compiling oracle probe shader..."
$opTmp = Join-Path $env:TEMP "oracle_probe.spv"
& $glslang -V --target-env vulkan1.2 -S comp "$src/shaders/oracle_probe.comp" -o $opTmp | Out-Null
if ($LASTEXITCODE -ne 0) { throw "oracle_probe.comp failed to compile" }
if (Test-Path $spvval) {
    & $spvval $opTmp | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "oracle_probe.comp failed spirv-val" }
}
$obytes = [System.IO.File]::ReadAllBytes($opTmp)
if ($obytes.Length % 4) { throw "oracle_probe SPIR-V is not a whole number of words" }
$owords = New-Object System.Collections.Generic.List[string]
for ($i = 0; $i -lt $obytes.Length; $i += 4) {
    $owords.Add(("0x{0:x8}u" -f [System.BitConverter]::ToUInt32($obytes, $i)))
}
$osb = New-Object System.Text.StringBuilder
[void]$osb.AppendLine("// Generated from src/shaders/oracle_probe.comp by build.ps1 - do not edit.")
[void]$osb.AppendLine("#pragma once")
[void]$osb.AppendLine("#include <stdint.h>")
[void]$osb.AppendLine("")
[void]$osb.AppendLine("static const uint32_t kOracleProbeSpv[] = {")
for ($i = 0; $i -lt $owords.Count; $i += 8) {
    $n = [Math]::Min(8, $owords.Count - $i)
    [void]$osb.AppendLine("    " + (($owords.GetRange($i, $n)) -join ",") + ",")
}
[void]$osb.AppendLine("};")
[IO.File]::WriteAllText("$src/vklayer/oracle_probe_spv.h", $osb.ToString())
Write-Host ("  oracle_probe_spv.h: {0} words" -f $owords.Count)

Write-Host "Building FidelityFX..."
    New-Item -ItemType Directory -Force $ffxObj | Out-Null
    # ---- FFX'S DIRECT VULKAN CALLS, REDIRECTED BY NAME.
    #
    # ffx_vk.cpp calls eight Vulkan entry points BY NAME rather than through the
    # vkGetDeviceProcAddr it is handed. Bound to the loader's exports, a call
    # from inside a layer re-enters the dispatch chain at the top and comes back
    # into our own hook - which killed FSR3 in its first call with no output.
    #
    # Defining the real vk* names inside this DLL fixed that and introduced a
    # worse problem: those names ARE the layer interface, so anything else here
    # that calls one binds to ours instead of the loader, and the export table
    # becomes somewhere a mistake can hide. It crashed the sim during load.
    #
    # Redefining the names for THIS COMPILE ONLY avoids both. FFX's calls become
    # calls to mvFfx* forwarders, which go down the chain; no Vulkan symbol is
    # defined in the DLL at all, so there is nothing to collide with or export.
    $ffxRename = @(
        "-DvkEnumerateDeviceExtensionProperties=mvFfxEnumerateDeviceExtensionProperties",
        "-DvkGetPhysicalDeviceProperties2=mvFfxGetPhysicalDeviceProperties2",
        "-DvkGetPhysicalDeviceFeatures2=mvFfxGetPhysicalDeviceFeatures2",
        "-DvkGetPhysicalDeviceProperties=mvFfxGetPhysicalDeviceProperties",
        "-DvkGetPhysicalDeviceMemoryProperties=mvFfxGetPhysicalDeviceMemoryProperties",
        "-DvkGetPhysicalDeviceFeatures=mvFfxGetPhysicalDeviceFeatures",
        "-DvkGetDeviceProcAddr=mvFfxGetDeviceProcAddr",
        "-DvkCreateBuffer=mvFfxCreateBuffer",
        # ---- THE FRAME-INTERPOLATION SWAPCHAIN'S DIRECT CALLS.
        #
        # Redirected to ffx_fg_shim.cpp, which forwards DOWN the dispatch
        # chain. Several of these are entry points this layer HOOKS, so
        # resolving them from the loader would recurse into us - see the
        # note at the top of that file.
        "-DvkAcquireNextImageKHR=mvFfx_vkAcquireNextImageKHR",
        "-DvkAllocateCommandBuffers=mvFfx_vkAllocateCommandBuffers",
        "-DvkAllocateDescriptorSets=mvFfx_vkAllocateDescriptorSets",
        "-DvkAllocateMemory=mvFfx_vkAllocateMemory",
        "-DvkBeginCommandBuffer=mvFfx_vkBeginCommandBuffer",
        "-DvkBindImageMemory=mvFfx_vkBindImageMemory",
        "-DvkCmdBeginRenderPass=mvFfx_vkCmdBeginRenderPass",
        "-DvkCmdBindDescriptorSets=mvFfx_vkCmdBindDescriptorSets",
        "-DvkCmdBindPipeline=mvFfx_vkCmdBindPipeline",
        "-DvkCmdCopyImage=mvFfx_vkCmdCopyImage",
        "-DvkCmdDraw=mvFfx_vkCmdDraw",
        "-DvkCmdEndRenderPass=mvFfx_vkCmdEndRenderPass",
        "-DvkCmdPipelineBarrier=mvFfx_vkCmdPipelineBarrier",
        "-DvkCmdPushConstants=mvFfx_vkCmdPushConstants",
        "-DvkCmdSetScissor=mvFfx_vkCmdSetScissor",
        "-DvkCmdSetViewport=mvFfx_vkCmdSetViewport",
        "-DvkCreateCommandPool=mvFfx_vkCreateCommandPool",
        "-DvkCreateDescriptorPool=mvFfx_vkCreateDescriptorPool",
        "-DvkCreateDescriptorSetLayout=mvFfx_vkCreateDescriptorSetLayout",
        "-DvkCreateFramebuffer=mvFfx_vkCreateFramebuffer",
        "-DvkCreateGraphicsPipelines=mvFfx_vkCreateGraphicsPipelines",
        "-DvkCreateImage=mvFfx_vkCreateImage",
        "-DvkCreateImageView=mvFfx_vkCreateImageView",
        "-DvkCreatePipelineLayout=mvFfx_vkCreatePipelineLayout",
        "-DvkCreateRenderPass=mvFfx_vkCreateRenderPass",
        "-DvkCreateSemaphore=mvFfx_vkCreateSemaphore",
        "-DvkCreateShaderModule=mvFfx_vkCreateShaderModule",
        "-DvkCreateSwapchainKHR=mvFfx_vkCreateSwapchainKHR",
        "-DvkDestroyCommandPool=mvFfx_vkDestroyCommandPool",
        "-DvkDestroyDescriptorPool=mvFfx_vkDestroyDescriptorPool",
        "-DvkDestroyDescriptorSetLayout=mvFfx_vkDestroyDescriptorSetLayout",
        "-DvkDestroyFramebuffer=mvFfx_vkDestroyFramebuffer",
        "-DvkDestroyImage=mvFfx_vkDestroyImage",
        "-DvkDestroyImageView=mvFfx_vkDestroyImageView",
        "-DvkDestroyPipeline=mvFfx_vkDestroyPipeline",
        "-DvkDestroyPipelineLayout=mvFfx_vkDestroyPipelineLayout",
        "-DvkDestroyRenderPass=mvFfx_vkDestroyRenderPass",
        "-DvkDestroySemaphore=mvFfx_vkDestroySemaphore",
        "-DvkDestroyShaderModule=mvFfx_vkDestroyShaderModule",
        "-DvkDestroySwapchainKHR=mvFfx_vkDestroySwapchainKHR",
        "-DvkDeviceWaitIdle=mvFfx_vkDeviceWaitIdle",
        "-DvkEndCommandBuffer=mvFfx_vkEndCommandBuffer",
        "-DvkFreeCommandBuffers=mvFfx_vkFreeCommandBuffers",
        "-DvkFreeDescriptorSets=mvFfx_vkFreeDescriptorSets",
        "-DvkFreeMemory=mvFfx_vkFreeMemory",
        "-DvkGetImageMemoryRequirements=mvFfx_vkGetImageMemoryRequirements",
        "-DvkGetPhysicalDeviceQueueFamilyProperties=mvFfx_vkGetPhysicalDeviceQueueFamilyProperties",
        "-DvkGetPhysicalDeviceSurfaceSupportKHR=mvFfx_vkGetPhysicalDeviceSurfaceSupportKHR",
        "-DvkGetSemaphoreCounterValue=mvFfx_vkGetSemaphoreCounterValue",
        "-DvkGetSwapchainImagesKHR=mvFfx_vkGetSwapchainImagesKHR",
        "-DvkQueuePresentKHR=mvFfx_vkQueuePresentKHR",
        "-DvkQueueSubmit=mvFfx_vkQueueSubmit",
        "-DvkQueueWaitIdle=mvFfx_vkQueueWaitIdle",
        "-DvkResetCommandBuffer=mvFfx_vkResetCommandBuffer",
        "-DvkResetCommandPool=mvFfx_vkResetCommandPool",
        "-DvkUpdateDescriptorSets=mvFfx_vkUpdateDescriptorSets",
        "-DvkWaitSemaphores=mvFfx_vkWaitSemaphores")
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
        # ---- FRAME GENERATION. The stub these replace is gone.
        #
        # ffx_fg_stub.cpp existed because FrameInterpolationSwapchainVK.cpp did
        # not compile under GCC - it relied on MSVC accepting two things ISO C++
        # forbids: a static_cast from void* to a function pointer, and binding a
        # temporary to a non-const lvalue reference. Both are fixed in the
        # vendored source now, so the real implementation builds and the stub
        # would be a duplicate symbol.
        #
        # Those SDK edits are NOT in git - third_party is ignored - so they
        # travel with the tree and must be re-applied to a fresh checkout.
        "$ffxSdk\sdk\src\backends\vk\FrameInterpolationSwapchain\FrameInterpolationSwapchainVK.cpp",
        "$ffxSdk\sdk\src\backends\vk\FrameInterpolationSwapchain\FrameInterpolationSwapchainVK_Helpers.cpp",
        "$ffxSdk\sdk\src\backends\vk\FrameInterpolationSwapchain\FrameInterpolationSwapchainVK_UiComposition.cpp",
        "$ffxSdk\sdk\src\backends\vk\FrameInterpolationSwapchain\FrameInterpolationSwapchainVK_DebugPacing.cpp",
        "$ffxSdk\sdk\src\components\frameinterpolation\ffx_frameinterpolation.cpp",
        "$ffxSdk\sdk\src\components\opticalflow\ffx_opticalflow.cpp",
        "$ffxSdk\sdk\src\backends\shared\blob_accessors\ffx_frameinterpolation_shaderblobs.cpp",
        "$ffxSdk\sdk\src\backends\shared\blob_accessors\ffx_opticalflow_shaderblobs.cpp",
        "$src\vklayer\ffx_vk_shim.cpp",
        "$src\vklayer\ffx_fg_shim.cpp")
    foreach ($f in $ffxSrc) {
        if (-not (Test-Path $f)) { continue }
        $o = Join-Path $ffxObj ((Split-Path $f -Leaf) -replace '\.cpp$', '.o')
        # ---- WARNINGS ARE SILENCED FOR VENDORED FFX SOURCES (-w ABOVE).
        #
        # Not cosmetic: PowerShell 5.1 turns anything a native tool writes to
        # stderr into a NativeCommandError and aborts the script, so a plain
        # warning stops the build midway - leaving a half-populated object
        # directory that then fails at link with "undefined reference" to a
        # file that was never compiled. Two separate warnings in AMD's source
        # did exactly that here ('#pragma once' in a main file, and a string
        # constant to wchar_t*). We do not own this code and are not going to
        # fix its warnings, so they are turned off rather than enumerated.
        # Our OWN sources are compiled elsewhere and keep their warnings.
        # Up to date? Skip it. ffx_fsr3upscaler_shaderblobs.cpp alone is 3.4 MB
        # of object and dominates this stage, so rebuilding it every time would
        # make the correct gate above cost more than the bug it fixes.
        if ((Test-Path $o) -and ((Get-Item $o).LastWriteTime -gt (Get-Item $f).LastWriteTime)) { continue }
        & g++ -c -O2 -std=c++17 -include new -include cmath -include cstring `
            -include cwchar -include cstdio $ffxInc `
            -w -DFFX_VK=1 -DFFX_FSR3UPSCALER -DFFX_FRAMEINTERPOLATION -DFFX_OPTICALFLOW -DFFX_FI -DFFX_OF -DWIN32 "-Dswprintf_s=_snwprintf" `
            @ffxRename `
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
& g++ -shared -o "$out\vklayer\VkLayer_mv.dll" "$src\vklayer\layer.cpp" `
  -I"$vksdk\Include" -m64 -O2 -std=c++17 -D__USE_MINGW_ANSI_STDIO=1 `
  @sdkDefines @sdkIncludes $ffxDefine `
  "-I$ffxSdk\sdk\include" @ffxObjs `
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
    # ---- AN OPTIONAL COMPONENT MUST NOT ABORT THE BUILD.
    #
    # This used to throw. With $ErrorActionPreference = "Stop" that ends the
    # script - and the plugin INSTALL step is further down, so a failed Qt
    # launcher silently left Resources\plugins\...\win.xpl at whatever version
    # it happened to be. The layer still updated (it is loaded in place from
    # VK_LAYER_PATH), so builds looked fine and only the plugin was stale.
    #
    # That cost real time: a TaaShare field added to the layer made structSize
    # disagree with the fourteen-hour-old plugin, which then reported "VULKAN
    # LAYER NOT ATTACHED" - a handshake failure presented as a missing layer.
    #
    # The launcher is a convenience UI. It must be able to fail without taking
    # the binaries anyone actually flies with down alongside it. Recorded and
    # reported loudly at the end instead.
    if ($LASTEXITCODE -ne 0) {
        $script:qtLauncherFailed = $true
        Write-Host "  Qt launcher build FAILED - continuing so the plugin and layer still install." -ForegroundColor Yellow
    }
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
    if ($LASTEXITCODE -ne 0) { $script:qtLauncherFailed = $true; Write-Host "  debug console build FAILED - continuing." -ForegroundColor Yellow }

    & (Join-Path $qtRoot "bin\windeployqt.exe") --release --no-translations `
      (Join-Path $qtOut "MotionVectors.exe") | Out-Null
    if ($LASTEXITCODE -ne 0) { $script:qtLauncherFailed = $true; Write-Host "  windeployqt FAILED - continuing." -ForegroundColor Yellow }
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
$dst = Join-Path $xp "Resources\plugins\MotionVectors\64"
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item "$out\MotionVectors.xpl" (Join-Path $dst "win.xpl") -Force

# ---- THE PANEL IS PART OF THE BUILD. IT WAS NOT, AND THAT COST A DEBUG CYCLE.
#
# build.ps1 installed the .xpl and nothing else, so every edit to the FlyWithLua
# panel sat in the repo while the sim kept loading whatever copy happened to be
# in Scripts\. That failed silently in the worst way: the panel LOOKED current,
# the build said "Done", and a freshly added button did nothing at all - which
# reads as a broken feature rather than as a stale file.
#
# Same failure mode as the .xpl lock this script already guards: the build
# succeeds, the sim runs something else. Copying it here means "built" and
# "installed" cannot drift apart again.
$luaDst = Join-Path $xp "Resources\plugins\FlyWithLua\Scripts"
if (Test-Path $luaDst) {
    # The panel shows its version in the UI and in bug-report dumps; it was a
    # hardcoded literal and read "1.1.5" for the whole of 1.2.0-dev. Stamp it
    # from the same VERSION file as every other component.
    (Get-Content "$root\lua\MotionVectors.lua" -Raw).Replace("@MV_VERSION@", $mvVersion) |
        Set-Content (Join-Path $luaDst "MotionVectors.lua") -Encoding utf8 -NoNewline
    Write-Host "  panel -> FlyWithLua\Scripts\MotionVectors.lua ($mvVersion)"
} else {
    Write-Host "  FlyWithLua Scripts folder not found - panel NOT installed" -ForegroundColor Yellow
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

if ($script:qtLauncherFailed) {
    Write-Host ""
    Write-Host "BUILD COMPLETED, BUT THE QT LAUNCHER DID NOT." -ForegroundColor Yellow
    Write-Host "  Plugin and layer installed correctly; only the launcher UI is stale." -ForegroundColor Yellow
}
Write-Host "Done." -ForegroundColor Green
