# FG trusted vectors + static snap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** stop FFX frame interpolation from warping cockpit text by making it trust our motion vectors where they exist and copy the real frame where nothing moves.

**Architecture:** the prep pass widens its dilated-vector image to RGBA16F and writes a trust flag in `.z`; the FFX interpolation pass binds that image, skips optical flow where trust is set, and snaps screen-static pixels to the current frame. Switches ride in the FFX dispatch flags word, so only the interpolation pass's permutation headers are regenerated (with our own `ffx_permute`).

**Tech Stack:** C++17 / MinGW g++ (build.ps1), GLSL compute via glslangValidator (Vulkan SDK 1.4.357.0), FidelityFX SDK 1.1.4 (vendored, patched), PowerShell 5.1.

**Spec:** `docs/superpowers/specs/2026-09-10-fg-trusted-vectors-design.md`

## Global Constraints

- No git in this tree: "commit" = keep a dated copy of the artefact you are about to overwrite (`build\vklayer\VkLayer_mv.<tag>.dll`, `build\ffx_shaders\*.h.bak`).
- Never edit an FFX shader before Task 1's gate passes (base digest `59247a0deabd3e29`, 16-bit `d2e13637e46a9843`).
- X-Plane running ⇒ the live DLL is locked: build with `$env:MV_LAYER_OUT = "<root>\build\vklayer\VkLayer_mv.staged.dll"` and swap it in after the sim exits. Never relaunch the sim yourself; the user flies.
- Flag bits are fixed by the spec: bit 16 trust, bit 17 snap, bits 24–31 eps in 1/32 px (0 ⇒ 0.25), bit 2 = FFX debug view. Same values in three places: host enum, GLSL callbacks, `src/vklayer/fg_flags.h`.
- Shader reflection name must stay `r_dilated_motion_vectors`; any other name makes `patchResourceBindings` return `FFX_ERROR_INVALID_ARGUMENT` and the FI context fails to build.
- The `_16bit` header is the one that runs on the 4060 (fp16 supported, wave64 never forced). Test that variant, not only the base.
- Bash tool collapses `\\` and chokes on backticks: write multi-line patches with the Write tool or a Python script file; run PowerShell scripts through `powershell -NoProfile -ExecutionPolicy Bypass -File`.
- Paths below are relative to `D:\Steam Games\steamapps\common\X-Plane 12\MotionVectors` (the "root").

---

### Task 1: Toolchain gate — reproduce today's interpolation-pass headers

**Files:**
- Create: `tools/ffx_regen_fi.ps1`
- Create: `tools/ffx_blob_dis.py` (fallback only)
- Read: `build/ffx_shaders/ffx_frameinterpolation_pass_permutations.h`, `build/ffx_shaders/ffx_frameinterpolation_pass_16bit_permutations.h`

**Interfaces:**
- Produces: `tools/ffx_regen_fi.ps1 [-OutDir <dir>]` — regenerates the four `ffx_frameinterpolation_pass*_permutations.h` headers into `<dir>` (default `build\ffx_shaders`) and prints each header's blob digest and byte count. Exit code non-zero on any failure.
- Produces: `tools/ffx_blob_dis.py <header.h> <out.spv>` — extracts the first `g_<name>_<digest>_data` byte array from a generated header into a `.spv` file.

- [ ] **Step 1: Write the regeneration script**

Write `tools/ffx_regen_fi.ps1`:

```powershell
param([string]$OutDir = "")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if ($OutDir -eq "") { $OutDir = Join-Path $root "build\ffx_shaders" }
$vksdk = (Get-ChildItem "C:\VulkanSDK\*" -Directory | Sort-Object Name | Select-Object -Last 1).FullName
$G = Join-Path $vksdk "Bin\glslangValidator.exe"
$permute = Join-Path $root "build\ffx_permute.exe"
$src = Join-Path $root "third_party\FidelityFX-SDK\sdk\src\backends\vk\shaders\frameinterpolation\ffx_frameinterpolation_pass.glsl"
$inc = Join-Path $root "third_party\FidelityFX-SDK\sdk\include\FidelityFX\gpu"
foreach ($p in @($G, $permute, $src, $inc)) { if (-not (Test-Path $p)) { throw "missing: $p" } }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$opts = @("--permute", "FFX_FRAMEINTERPOLATION_OPTION_LOW_RES_MOTION_VECTORS",
          "--permute", "FFX_FRAMEINTERPOLATION_OPTION_JITTER_MOTION_VECTORS",
          "--permute", "FFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH")
$variants = @(
    @{ name = "ffx_frameinterpolation_pass";              defs = @() },
    @{ name = "ffx_frameinterpolation_pass_16bit";        defs = @("-D", "FFX_HALF=1") },
    @{ name = "ffx_frameinterpolation_pass_wave64";       defs = @("-D", "FFX_PREFER_WAVE64=1") },
    @{ name = "ffx_frameinterpolation_pass_wave64_16bit"; defs = @("-D", "FFX_HALF=1", "-D", "FFX_PREFER_WAVE64=1") })
foreach ($v in $variants) {
    $out = Join-Path $OutDir ($v.name + "_permutations.h")
    $args = @("--glslang", $G, "--src", $src, "--name", $v.name, "--out", $out, "-I", $inc, "-D", "FFX_GPU=1", "-D", "FFX_GLSL=1") + $v.defs + $opts
    & $permute @args
    if ($LASTEXITCODE -ne 0) { throw ("ffx_permute failed for " + $v.name) }
    $m = (Select-String -Path $out -Pattern ("g_" + $v.name + "_([0-9a-f]{16})_size = (\d+)") | Select-Object -First 1).Matches[0]
    Write-Host ("  {0,-45} digest={1} bytes={2}" -f $v.name, $m.Groups[1].Value, $m.Groups[2].Value)
}
```

- [ ] **Step 2: Run the gate into a temp folder**

Run (PowerShell):
```
powershell -NoProfile -ExecutionPolicy Bypass -File tools\ffx_regen_fi.ps1 -OutDir $env:TEMP\ffx_gate
```
Expected: four lines. `ffx_frameinterpolation_pass` → `digest=59247a0deabd3e29 bytes=19108`; `..._16bit` → `digest=d2e13637e46a9843 bytes=19140`; the two wave64 lines equal to their non-wave64 twins.

- [ ] **Step 3: If a digest differs, prove it is only the SPIR-V generator word before going on**

Write `tools/ffx_blob_dis.py`:

```python
import re, sys
hdr, out = sys.argv[1], sys.argv[2]
s = open(hdr, encoding="utf-8").read()
m = re.search(r"static const unsigned char g_\w+_data\[\] = \{(.*?)\};", s, re.S)
data = bytes(int(x, 0) for x in re.findall(r"0x[0-9a-fA-F]{2}|\d+", m.group(1)))
open(out, "wb").write(data); print(out, len(data), "bytes")
```

Run:
```
python tools\ffx_blob_dis.py build\ffx_shaders\ffx_frameinterpolation_pass_permutations.h %TEMP%\old.spv
python tools\ffx_blob_dis.py %TEMP%\ffx_gate\ffx_frameinterpolation_pass_permutations.h %TEMP%\new.spv
"C:\VulkanSDK\1.4.357.0\Bin\spirv-dis.exe" %TEMP%\old.spv > %TEMP%\old.dis
"C:\VulkanSDK\1.4.357.0\Bin\spirv-dis.exe" %TEMP%\new.spv > %TEMP%\new.dis
fc %TEMP%\old.dis %TEMP%\new.dis
```
Expected: either `FC: no differences encountered`, or differences confined to the `; Generator:` / `; Version:` header lines. Any difference in instructions means wrong defines or include path: fix `tools/ffx_regen_fi.ps1` and repeat Step 2. Do not proceed to Task 4 until this holds.

- [ ] **Step 4: Snapshot the headers you will overwrite later**

Run:
```
copy build\ffx_shaders\ffx_frameinterpolation_pass_permutations.h build\ffx_shaders\ffx_frameinterpolation_pass_permutations.h.bak
copy build\ffx_shaders\ffx_frameinterpolation_pass_16bit_permutations.h build\ffx_shaders\ffx_frameinterpolation_pass_16bit_permutations.h.bak
copy build\ffx_shaders\ffx_frameinterpolation_pass_wave64_permutations.h build\ffx_shaders\ffx_frameinterpolation_pass_wave64_permutations.h.bak
copy build\ffx_shaders\ffx_frameinterpolation_pass_wave64_16bit_permutations.h build\ffx_shaders\ffx_frameinterpolation_pass_wave64_16bit_permutations.h.bak
```
Expected: four `.bak` files beside the originals.

---

### Task 2: Flag packing helper (host side) with a CPU test

**Files:**
- Create: `src/vklayer/fg_flags.h`
- Create: `src/test_fg_flags.cpp`

**Interfaces:**
- Produces: `uint32_t fgTrustFlags(bool trust, bool snap, float epsPx, bool debug)` and `float fgSnapEpsPx(uint32_t flags)`, plus constants `kFgFlagDebugView (1u<<2)`, `kFgFlagMvTrust (1u<<16)`, `kFgFlagMvSnap (1u<<17)`, `kFgFlagEpsShift (24)`. Task 5 consumes these.

- [ ] **Step 1: Write the failing test**

Write `src/test_fg_flags.cpp`:

```cpp
// Build:  g++ -O2 -std=c++17 -I src -o build/test_fg_flags.exe src/test_fg_flags.cpp
// Run:    build/test_fg_flags.exe      (exit 0 = all pass)
#include "vklayer/fg_flags.h"
#include <cstdio>
static int fails = 0;
static void check(bool ok, const char *what) { std::printf("%s %s\n", ok ? "ok  " : "FAIL", what); if (!ok) ++fails; }
int main()
{
    check(fgTrustFlags(false, false, 0.25f, false) == 0x08000000u, "eps 0.25 alone -> byte 8 in bits 24..31");
    check(fgTrustFlags(true,  true,  0.25f, false) == 0x08030000u, "trust + snap + 0.25 -> 0x08030000");
    check(fgTrustFlags(true,  false, 0.0f,  false) == 0x00010000u, "eps 0 -> byte 0 (shader default), trust only");
    check(fgTrustFlags(true,  true,  10.0f, true)  == 0xFF030004u, "eps clamps to 255, debug view = bit 2");
    check(fgTrustFlags(false, false, 0.03f, false) == 0x01000000u, "0.03 px rounds to 1/32");
    check(fgSnapEpsPx(0x08030000u) == 0.25f, "decode byte 8 -> 0.25");
    check(fgSnapEpsPx(0x00030000u) == 0.25f, "decode byte 0 -> default 0.25");
    check(fgSnapEpsPx(0x20000000u) == 1.0f,  "decode byte 32 -> 1.0");
    std::printf("%d failure(s)\n", fails);
    return fails;
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `g++ -O2 -std=c++17 -I src -o build/test_fg_flags.exe src/test_fg_flags.cpp`
Expected: compile error `fg_flags.h: No such file or directory`.

- [ ] **Step 3: Write the header**

Write `src/vklayer/fg_flags.h`:

```cpp
// fg_flags.h - the private bits MotionVectors adds to FfxFrameInterpolationDispatchDescription::flags.
//
// FFX copies that word verbatim into the interpolation constant buffer (cbFI.dispatchFlags)
// and only uses bits 0..3 itself, so it is the one channel into the interpolation shader
// that needs no constant-buffer layout change. Same values live in
//   third_party/.../host/ffx_frameinterpolation.h   (FFX_FRAMEINTERPOLATION_DISPATCH_MV_*)
//   third_party/.../gpu/frameinterpolation/ffx_frameinterpolation_callbacks_glsl.h (FFX_FI_FLAG_*)
// Change one, change all three. Spec: docs/superpowers/specs/2026-09-10-fg-trusted-vectors-design.md
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <stdint.h>

enum : uint32_t {
    kFgFlagDebugView = 1u << 2,   // FFX_FRAMEINTERPOLATION_DISPATCH_DRAW_DEBUG_VIEW (stock)
    kFgFlagMvTrust   = 1u << 16,  // game vector is final where our dilated .z is set (no optical flow)
    kFgFlagMvSnap    = 1u << 17,  // trusted + still (< eps) => output the current frame's texel
    kFgFlagEpsShift  = 24,        // bits 24..31: snap eps in 1/32 display px, 0 => shader default 0.25
};

inline uint32_t fgTrustFlags(bool trust, bool snap, float epsPx, bool debug)
{
    int e = (int)(epsPx * 32.0f + 0.5f);
    if (e < 0) e = 0;
    if (e > 255) e = 255;
    uint32_t f = ((uint32_t)e) << kFgFlagEpsShift;
    if (trust) f |= kFgFlagMvTrust;
    if (snap)  f |= kFgFlagMvSnap;
    if (debug) f |= kFgFlagDebugView;
    return f;
}

inline float fgSnapEpsPx(uint32_t flags)
{
    const uint32_t e = (flags >> kFgFlagEpsShift) & 0xFFu;
    return e ? (float)e / 32.0f : 0.25f;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `g++ -O2 -std=c++17 -I src -o build/test_fg_flags.exe src/test_fg_flags.cpp && build/test_fg_flags.exe`
Expected: eight `ok` lines, `0 failure(s)`, exit code 0.

---

### Task 3: Prep pass writes the trust flag (RGBA16F dilated vectors)

**Files:**
- Modify: `src/shaders/fg_prepare.comp:34` and `:83`
- Modify: `src/vklayer/fg_prepare.h:258`
- Modify: `src/vklayer/fg_backend.h:730-732`

**Interfaces:**
- Produces: dilated-vector image `fgprep::State::dilMv[k]` is `VK_FORMAT_R16G16B16A16_SFLOAT`, texel = `(mv.x px, mv.y px, trust 0|1, 0)`; registered with FFX as `FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT`. Task 4's shader reads `.z`.

- [ ] **Step 1: Shader — storage format and the stored value**

In `src/shaders/fg_prepare.comp` replace line 34
```glsl
layout(binding = 3, rg16f) uniform writeonly image2D uDilatedMv;
```
with
```glsl
// RGBA16F, not RG16F: .z carries the TRUST flag (1 = a real draw wrote this velocity texel,
// 0 = the target still held its "unwritten" sentinel). FFX's interpolation pass reads it to
// decide whether our vector is final. The qualifier MUST match the view format (VUID) - see
// fg_prepare.h. docs/superpowers/specs/2026-09-10-fg-trusted-vectors-design.md
layout(binding = 3, rgba16f) uniform writeonly image2D uDilatedMv;
```
and replace line 83 (after the edit above it is line 87)
```glsl
    imageStore(uDilatedMv,    px, vec4(mvPix, 0.0, 0.0));
```
with
```glsl
    imageStore(uDilatedMv,    px, vec4(mvPix, unwritten ? 0.0 : 1.0, 0.0));   // .z = trust
```

- [ ] **Step 2: Image format**

In `src/vklayer/fg_prepare.h` line 258 replace
```cpp
            !makeImage(device, gdpa, mp, w, h, VK_FORMAT_R16G16_SFLOAT,
```
with
```cpp
            !makeImage(device, gdpa, mp, w, h, VK_FORMAT_R16G16B16A16_SFLOAT,   // .z = trust, see fg_prepare.comp
```
(`makeImage` builds the view with the same `fmt`, line 161, so the storage view follows.)

- [ ] **Step 3: Registration with FFX**

In `src/vklayer/fg_backend.h` replace lines 730–732
```cpp
            fd.dilatedMotionVectors =
                ffxGetResourceVK(g.dilMv[r], sh.dilatedMotionVectors.resourceDescription,
                                 nullptr, FFX_RESOURCE_STATE_UNORDERED_ACCESS);
```
with
```cpp
            // Ours is RGBA16F (.z = trust); FFX's own description says RG16F and the view it
            // builds from the description must match the image, so override the format only.
            FfxResourceDescription dmvDesc = sh.dilatedMotionVectors.resourceDescription;
            dmvDesc.format = FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT;
            fd.dilatedMotionVectors =
                ffxGetResourceVK(g.dilMv[r], dmvDesc,
                                 nullptr, FFX_RESOURCE_STATE_UNORDERED_ACCESS);
```

- [ ] **Step 4: Compile the shader on its own (fast syntax check)**

Run: `"C:\VulkanSDK\1.4.357.0\Bin\glslangValidator.exe" -V --target-env vulkan1.2 -S comp src/shaders/fg_prepare.comp -o %TEMP%\fg_prepare_test.spv`
Expected: no output, exit 0.

- [ ] **Step 5: Build the layer (staged if the sim is running)**

Run:
```
copy build\vklayer\VkLayer_mv.dll build\vklayer\VkLayer_mv.pre-trust.dll
powershell -NoProfile -ExecutionPolicy Bypass -File build.ps1
```
(if X-Plane is running: `set MV_LAYER_OUT=<root>\build\vklayer\VkLayer_mv.staged.dll` first, then swap after exit.)
Expected: no `error`/`failed` lines; `objdump -p build\vklayer\VkLayer_mv.dll | findstr Time/Date` shows today's time.

- [ ] **Step 6: Behavioural check (user flies)**

In the new trace after a flight with `taa.fg=1`: `FG PREPARE: ready`, `FG: interpolation is reading OUR dilated depth/motion`, `FPS: ... (2.00x)`, no `device lost`. Picture unchanged versus the 20:21 build (FFX still reads only `.xy` at this point). If the FI context fails to build or the image is black, the format override in Step 3 did not reach the view: check the trace for `FG DISPATCH:` and the validation layer via `launch_xp_gpuav.ps1`.

---

### Task 4: FFX interpolation pass — trust, no optical flow, static snap

**Files:**
- Modify: `third_party/FidelityFX-SDK/sdk/include/FidelityFX/host/ffx_frameinterpolation.h:238-246`
- Modify: `third_party/FidelityFX-SDK/sdk/include/FidelityFX/gpu/frameinterpolation/ffx_frameinterpolation_callbacks_glsl.h:146-149` and `:250-258`
- Modify: `third_party/FidelityFX-SDK/sdk/include/FidelityFX/gpu/frameinterpolation/ffx_frameinterpolation.h:93-166`
- Modify: `third_party/FidelityFX-SDK/sdk/src/backends/vk/shaders/frameinterpolation/ffx_frameinterpolation_pass.glsl:42-44`
- Modify (mirror, not compiled here): `third_party/FidelityFX-SDK/sdk/include/FidelityFX/gpu/frameinterpolation/ffx_frameinterpolation_callbacks_hlsl.h` — same two functions

**Interfaces:**
- Consumes: dilated `.z` trust from Task 3; flag bits from Task 2's header (values only).
- Produces: shader reads `FFX_FI_FLAG_MV_TRUST`, `FFX_FI_FLAG_MV_SNAP`, `SnapEpsPx()`; host enum `FFX_FRAMEINTERPOLATION_DISPATCH_MV_TRUST`, `..._MV_SNAP`, `..._SNAP_EPS_SHIFT`.

- [ ] **Step 1: Host enum**

In the host `ffx_frameinterpolation.h`, after the line
```cpp
    FFX_FRAMEINTERPOLATION_DISPATCH_RESERVED_2 = (1 << 5), 
```
insert
```cpp
    // ---- MotionVectors private bits. Mirrored in gpu/frameinterpolation/ffx_frameinterpolation_callbacks_glsl.h
    // (FFX_FI_FLAG_*) and in src/vklayer/fg_flags.h. Bits 24..31 carry the static-snap threshold in 1/32 px.
    FFX_FRAMEINTERPOLATION_DISPATCH_MV_TRUST       = (1 << 16), ///< game vector is final where our dilated .z is set (no optical flow)
    FFX_FRAMEINTERPOLATION_DISPATCH_MV_SNAP        = (1 << 17), ///< trusted and still (< eps): output the current frame's texel, no blend
    FFX_FRAMEINTERPOLATION_DISPATCH_SNAP_EPS_SHIFT = 24,        ///< not a bit: shift of the 8-bit eps field (0 => 0.25 px)
```

- [ ] **Step 2: GLSL callbacks — flags and trust accessor**

In `ffx_frameinterpolation_callbacks_glsl.h`, after
```glsl
    FfxUInt32 GetDispatchFlags()
    {
        return cbFI.dispatchFlags;
    }
```
insert
```glsl
    // ---- MotionVectors private dispatch flags. Same values as the host enum
    // FFX_FRAMEINTERPOLATION_DISPATCH_MV_* and src/vklayer/fg_flags.h.
    #define FFX_FI_FLAG_MV_TRUST (1u << 16)
    #define FFX_FI_FLAG_MV_SNAP  (1u << 17)

    // Static-snap threshold on the HALF-frame displacement, display pixels. Bits 24..31,
    // 1/32 px units; a zero byte means the 0.25 px default.
    FfxFloat32 SnapEpsPx()
    {
        const FfxUInt32 e = (GetDispatchFlags() >> 24u) & 0xFFu;
        return (e == 0u) ? FfxFloat32(0.25) : FfxFloat32(e) / FfxFloat32(32.0);
    }
```
and inside the existing block
```glsl
#ifdef FFX_FRAMEINTERPOLATION_BIND_SRV_DILATED_MOTION_VECTORS
    ...
    FfxFloat32x2 LoadDilatedMotionVector(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return texelFetch(r_dilated_motion_vectors, iPxPos, 0).xy;
    }
```
add, before its `#endif`:
```glsl
    // MotionVectors: .z of our dilated vectors is 1 where a real draw wrote the velocity
    // texel and 0 where the target still held its "unwritten" sentinel (src/shaders/fg_prepare.comp).
    FfxFloat32 LoadDilatedTrust(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
    {
        return texelFetch(r_dilated_motion_vectors, iPxPos, 0).z;
    }
```

- [ ] **Step 3: HLSL callbacks mirror (symmetry only)**

In `ffx_frameinterpolation_callbacks_hlsl.h`, find the `GetDispatchFlags()` function and the `LoadDilatedMotionVector` function and add the HLSL equivalents right after each:
```hlsl
    #define FFX_FI_FLAG_MV_TRUST (1u << 16)
    #define FFX_FI_FLAG_MV_SNAP  (1u << 17)
    FfxFloat32 SnapEpsPx()
    {
        const FfxUInt32 e = (GetDispatchFlags() >> 24u) & 0xFFu;
        return (e == 0u) ? FfxFloat32(0.25) : FfxFloat32(e) / FfxFloat32(32.0);
    }
```
```hlsl
    FfxFloat32 LoadDilatedTrust(FfxInt32x2 iPxPos)
    {
        return r_dilated_motion_vectors[iPxPos].z;
    }
```
(Not compiled by this project; keeps the two callback headers in step for anyone diffing against upstream.)

- [ ] **Step 4: Main pass rules**

In the gpu `ffx_frameinterpolation.h`, `computeInterpolatedColor`, replace
```glsl
    // OF is done on the back buffers which already have black bars
    VectorFieldEntry ofMv;
    SampleOpticalFlowMotionVectorField(fUvInScreenSpace, ofMv);
```
with
```glsl
    // ---- MotionVectors: is OUR vector final here? (.z of the dilated vectors, same render-res
    // texel the game field was built from). docs/superpowers/specs/2026-09-10-fg-trusted-vectors-design.md
    FfxBoolean bTrust = false;
#ifdef FFX_FRAMEINTERPOLATION_BIND_SRV_DILATED_MOTION_VECTORS
    if ((GetDispatchFlags() & FFX_FI_FLAG_MV_TRUST) != 0u)
    {
        const FfxInt32x2 iTrustPx = FfxInt32x2(fUvInInterpolationRect * FfxFloat32x2(RenderSize()));
        bTrust = LoadDilatedTrust(iTrustPx) > 0.5;
    }
#endif

    // OF is done on the back buffers which already have black bars
    VectorFieldEntry ofMv;
    SampleOpticalFlowMotionVectorField(fUvInScreenSpace, ofMv);
```
Then replace
```glsl
    InterpolationSourceColor fPrevColorOF = SampleTextureBilinear(false, fUvInScreenSpace, +ofMv.fMotionVector * fUvLetterBoxScale, DisplaySize());
    InterpolationSourceColor fCurrColorOF = SampleTextureBilinear(true, fUvInScreenSpace, -ofMv.fMotionVector * fUvLetterBoxScale, DisplaySize());
```
with
```glsl
    // Optical-flow candidates are only needed where our vector is NOT final.
    InterpolationSourceColor fPrevColorOF = NewInterpolationSourceColor();
    InterpolationSourceColor fCurrColorOF = NewInterpolationSourceColor();
    if (!bTrust)
    {
        fPrevColorOF = SampleTextureBilinear(false, fUvInScreenSpace, +ofMv.fMotionVector * fUvLetterBoxScale, DisplaySize());
        fCurrColorOF = SampleTextureBilinear(true, fUvInScreenSpace, -ofMv.fMotionVector * fUvLetterBoxScale, DisplaySize());
    }
```
Then replace the opening of the optical-flow block
```glsl
    {
        FfxFloat32 ofT = 0.5f;
```
(the `{` on the line right after the disocclusion block's closing `}`) with
```glsl
    // ---- MotionVectors: static snap. Trusted, not moving, nothing disoccluded, both warps in
    // bounds => the pixel IS the current frame's texel. No blend, so screen text on generated
    // flips is pixel-identical to the real frame instead of a crossfade of two states.
#ifdef FFX_FRAMEINTERPOLATION_BIND_SRV_DILATED_MOTION_VECTORS
    if (bTrust && (GetDispatchFlags() & FFX_FI_FLAG_MV_SNAP) != 0u)
    {
        const FfxFloat32 fHalfStepPx = length(gameMv.fMotionVector * FfxFloat32x2(InterpolationRectSize()));
        const FfxBoolean bStill = (fHalfStepPx < SnapEpsPx())
                               && (fDisocclusionFactor.x == 1.0) && (fDisocclusionFactor.y == 1.0)
                               && !gameMv.bPosOutside && !gameMv.bNegOutside
                               && (fPrevColorGame.fBilinearWeightSum > 0.0) && (fCurrColorGame.fBilinearWeightSum > 0.0);
        if (bStill)
        {
            fInterpolatedColor = LoadCurrentBackbuffer(FfxInt32x2(iPxPos));
        }
    }
#endif

    // ---- Stock optical-flow blend, only where our vector is not final. Trusted pixels keep the
    // game result: fGameMvBias would be 1 by construction, so the block is skipped outright.
    if (!bTrust)
    {
        FfxFloat32 ofT = 0.5f;
```
The rest of that block (down to `fInterpolatedColor = ffxLerp(ofColor, fInterpolatedColor, ffxSaturate(fGameMvBias));` and its closing `}`) stays byte-for-byte as it is; the closing `}` now closes the `if (!bTrust)`.

- [ ] **Step 5: Pass entry — bind the dilated vectors**

In `ffx_frameinterpolation_pass.glsl`, after
```glsl
#define FFX_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION                       10
```
insert
```glsl
// MotionVectors: the main pass reads our dilated vectors' .z (trust). Binding 11 is unused here;
// the component binds it by the reflected name r_dilated_motion_vectors (srvResourceBindingTable).
#define FFX_FRAMEINTERPOLATION_BIND_SRV_DILATED_MOTION_VECTORS                 11
```

- [ ] **Step 6: Standalone compile of the pass, both variants (syntax + binding sanity)**

Run (PowerShell, from root):
```
$G="C:\VulkanSDK\1.4.357.0\Bin\glslangValidator.exe"; $I="third_party\FidelityFX-SDK\sdk\include\FidelityFX\gpu"; $S="third_party\FidelityFX-SDK\sdk\src\backends\vk\shaders\frameinterpolation\ffx_frameinterpolation_pass.glsl"
& $G -V --target-env vulkan1.2 -S comp -I"$I" -DFFX_GPU=1 -DFFX_GLSL=1 -DFFX_FRAMEINTERPOLATION_OPTION_LOW_RES_MOTION_VECTORS=0 -DFFX_FRAMEINTERPOLATION_OPTION_JITTER_MOTION_VECTORS=0 -DFFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH=1 $S -o $env:TEMP\fi_pass_base.spv
& $G -V --target-env vulkan1.2 -S comp -I"$I" -DFFX_GPU=1 -DFFX_GLSL=1 -DFFX_HALF=1 -DFFX_FRAMEINTERPOLATION_OPTION_LOW_RES_MOTION_VECTORS=0 -DFFX_FRAMEINTERPOLATION_OPTION_JITTER_MOTION_VECTORS=0 -DFFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH=1 $S -o $env:TEMP\fi_pass_16.spv
& "C:\VulkanSDK\1.4.357.0\Bin\spirv-dis.exe" $env:TEMP\fi_pass_16.spv | Select-String "r_dilated_motion_vectors|Binding 11"
```
Expected: both compiles exit 0 with no errors; the disassembly shows `OpName ... "r_dilated_motion_vectors"` and a `Binding 11` decoration.

---

### Task 5: Layer side — flags from live keys, trace, ini template

**Files:**
- Modify: `src/vklayer/fg_backend.h` (include near line 40; after `fd.frameID = p->frameID;` at line 759)
- Modify: `config/taa_live.ini` (append after the `taa.conf_exp=0` block)

**Interfaces:**
- Consumes: `fgTrustFlags`, `fgSnapEpsPx`, `kFgFlag*` from `src/vklayer/fg_flags.h` (Task 2); `live::onoff(key, env, default)` and `live::f(key, env, default)` as already used in this file.
- Produces: live keys `taa.fg_trust`, `taa.fg_snap`, `taa.fg_snap_eps`, `taa.fg_debug`; trace line `FG TRUST: flags=0x%08x ...` whenever the word changes.

- [ ] **Step 1: Include the helper**

In `src/vklayer/fg_backend.h`, after the line `#include <FidelityFX/host/ffx_opticalflow.h>` add
```cpp
#include "fg_flags.h"   // private bits in FfxFrameInterpolationDispatchDescription::flags
```

- [ ] **Step 2: Build the flags word and trace changes**

After
```cpp
    fd.frameID = p->frameID;
```
insert
```cpp
    // ---- MotionVectors private flags: trust our vectors, static snap, its threshold, FFX debug view.
    // Read per dispatch so the live ini toggles them without a relaunch (that is the A/B).
    fd.flags = fgTrustFlags(live::onoff("taa.fg_trust",  "TAA_FG_TRUST",  true),
                            live::onoff("taa.fg_snap",   "TAA_FG_SNAP",   true),
                            live::f    ("taa.fg_snap_eps", "TAA_FG_SNAP_EPS", 0.25f),
                            live::onoff("taa.fg_debug",  "TAA_FG_DEBUG",  false));
    {
        static uint32_t lastFlags = 0xFFFFFFFFu;
        if (fd.flags != lastFlags) {
            lastFlags = fd.flags;
            trace("FG TRUST: flags=0x%08x trust=%d snap=%d eps=%.2fpx debug=%d - where our dilated .z is set the "
                  "game vector is final (no optical flow); trusted+still pixels copy the current frame.",
                  fd.flags, (fd.flags & kFgFlagMvTrust) ? 1 : 0, (fd.flags & kFgFlagMvSnap) ? 1 : 0,
                  fgSnapEpsPx(fd.flags), (fd.flags & kFgFlagDebugView) ? 1 : 0);
        }
    }
```

- [ ] **Step 3: Ini template**

Append to `config/taa_live.ini` after the line `taa.conf_exp=0`:
```ini

# ---- frame generation (needs taa.fg=1 at launch). Inside FFX's interpolator: trust OUR vectors where a
# draw wrote them (no optical flow there) and snap screen-static pixels to the real frame. This is what
# stops cockpit screen text smearing on generated frames. fg_snap_eps = half-frame motion in display px
# below which a trusted pixel counts as still (0.25 => ~0.5 px/frame). fg_debug=1 draws FFX's vector view.
taa.fg_trust=1
taa.fg_snap=1
taa.fg_snap_eps=0.25
taa.fg_debug=0
```
Also add the same four keys to the live copy `%TEMP%\taa_live.ini` (the sim reads that one) — same values.

- [ ] **Step 4: Compile check of the layer source only (no link yet)**

Run:
```
g++ -fsyntax-only -std=c++17 -I "C:\VulkanSDK\1.4.357.0\Include" -I third_party/FidelityFX-SDK/sdk/include -I third_party/DLSS -I third_party/Streamline/include -D__USE_MINGW_ANSI_STDIO=1 src/vklayer/layer.cpp 2>&1 | grep -E "error" | head
```
Expected: no `error` lines. (Warnings about deprecated Streamline fields are pre-existing.)

---

### Task 6: Regenerate the four headers, rebuild FFX objects and the layer

**Files:**
- Regenerate: `build/ffx_shaders/ffx_frameinterpolation_pass{,_16bit,_wave64,_wave64_16bit}_permutations.h`
- Delete: `build/ffx_obj/ffx_frameinterpolation_shaderblobs.o`, `build/ffx_obj/ffx_frameinterpolation.o`
- Build: `build/vklayer/VkLayer_mv.dll` (or `VkLayer_mv.staged.dll`)

**Interfaces:**
- Consumes: `tools/ffx_regen_fi.ps1` (Task 1), shader edits (Task 4).
- Produces: a layer DLL whose interpolation pass carries the new logic.

- [ ] **Step 1: Regenerate**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File tools\ffx_regen_fi.ps1`
Expected: four digest lines; base and 16-bit digests DIFFER from the gate values (the shader changed); wave64 twins still equal their non-wave64 counterparts; byte counts larger than 19108 / 19140.

- [ ] **Step 2: Force the two FFX objects to recompile**

Run:
```
del build\ffx_obj\ffx_frameinterpolation_shaderblobs.o
del build\ffx_obj\ffx_frameinterpolation.o
```
Expected: both gone (`dir build\ffx_obj\ffx_frameinterpolation*` lists nothing).

- [ ] **Step 3: Build**

Run:
```
copy build\vklayer\VkLayer_mv.dll build\vklayer\VkLayer_mv.pre-trust-2.dll
powershell -NoProfile -ExecutionPolicy Bypass -File build.ps1
```
(sim running ⇒ `$env:MV_LAYER_OUT = "<root>\build\vklayer\VkLayer_mv.staged.dll"` before build.ps1; after the sim exits: `copy /Y build\vklayer\VkLayer_mv.staged.dll build\vklayer\VkLayer_mv.dll`.)
Expected: `... FidelityFX objects` count unchanged, no `error`/`failed`; `objdump -p build\vklayer\VkLayer_mv.dll | findstr Time/Date` shows now; `strings build\vklayer\VkLayer_mv.dll | findstr "FG TRUST"` finds the trace text; `dir build\ffx_obj\ffx_frameinterpolation*.o` shows both with today's time.

- [ ] **Step 4: Run the CPU test once more against the final tree**

Run: `build\test_fg_flags.exe`
Expected: `0 failure(s)`.

---

### Task 7: In-sim verification (user flies) and record the outcome

**Files:**
- Read: newest `%TEMP%\taa_layer_<pid>.txt`
- Modify: `%TEMP%\taa_live.ini` (A/B keys)
- Modify: `C:\Users\bansa\.claude\projects\d--Steam-Games-steamapps-common-X-Plane-12\memory\` (one new note)

**Interfaces:**
- Consumes: the DLL from Task 6; keys from Task 5.

- [ ] **Step 1: Bring-up lines**

After the user launches with `taa.fg=1` and loads a flight, run:
```
T=$(ls -t /c/Users/bansa/AppData/Local/Temp/taa_layer_*.txt | head -1); grep -a -n -E "FG: contexts created|FG TRUST:|generation is now|^FPS:.*x\)|device lost|INVALID_ARGUMENT|FG PREPARE: ready|FG: interpolation is reading OUR" "$T" | tail -12
```
Expected: `FG: contexts created`, `FG TRUST: flags=0x08030000 trust=1 snap=1 eps=0.25px debug=0`, `generation is now ENABLED`, `FPS: ... (2.00x)`, no `device lost`, no `INVALID_ARGUMENT`.

- [ ] **Step 2: Visual A/B, head still, cockpit screen in view**

User flips keys in `%TEMP%\taa_live.ini` (no relaunch), waits ~2 s each:
- `taa.fg_trust=0` → smear on screen text returns; `=1` → gone.
- `taa.fg_snap=0` (trust=1) → ticking numbers crossfade (double image) on generated flips; `=1` → crisp.
- `taa.fg_debug=1` → FFX vector view: cockpit flat, world moving; `=0` afterwards.
Expected: each toggle appears in the trace as a new `FG TRUST: flags=...` line within a couple of frames.

- [ ] **Step 3: Motion cases**

User checks: head movement in the 3D cockpit (no warp, TAA-level softening at most, no tearing at bezel edges); chase view livery text crisp on generated flips; a parked still frame shows no shimmer (generated == real); a slow pan of distant terrain does not judder (if it does, lower `taa.fg_snap_eps` to `0.12`).

- [ ] **Step 4: Record**

Write memory note `fg-trusted-vectors.md` (type: project) with: the mechanism (OF wins on content change), the fix (trust in dilated `.z`, OF skipped, static snap to current), the gate digests, the live keys, the "16-bit header is the live one on NVIDIA" fact, and the user's verdict; add one line to `MEMORY.md`.

---

## Self-review

- **Spec coverage:** §3 rules → Task 4 Step 4; §4 data path → Task 3 (+ Task 4 Step 5 binding); §5 switches → Task 2 + Task 5; §6 files → Tasks 3–5; §7 build/regen + gate → Tasks 1 and 6; §8 verification → Task 7 (+ Task 3 Step 6, Task 4 Step 6); §9 risks → constraints and Task 7 Step 3 (eps tuning); §10 out of scope → nothing planned for it. HUD-less interaction (§3 note) needs no code: `LoadCurrentBackbuffer` already resolves to the HUD-less image when attached.
- **Placeholders:** none; every edit shows the exact text and every command its expected output.
- **Type consistency:** `fgTrustFlags(bool,bool,float,bool)` / `fgSnapEpsPx(uint32_t)` used identically in Tasks 2 and 5; `LoadDilatedTrust(FfxInt32x2)` returns `FfxFloat32` in Task 4 Steps 2 and 4; `SnapEpsPx()` returns `FfxFloat32`; flag values 1<<16 / 1<<17 / shift 24 identical across Task 2, Task 4 Steps 1–2, and Task 5's trace decode.
