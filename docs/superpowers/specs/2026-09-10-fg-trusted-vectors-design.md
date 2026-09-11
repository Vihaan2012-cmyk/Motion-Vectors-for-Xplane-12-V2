# FG: trust our vectors + static snap 🎯

**Goal:** kill the cockpit text smear on generated frames. 2D overlays are the HUD-less path (already built, separate). This spec is the 3D side: screens, gauges, labels, anything with real motion vectors.

**Decision (user, 2026-09-10):** option 1 🟢 cockpit interpolated with OUR vectors, optical flow OFF where we have vectors, screen-static pixels snapped to the real frame. No per-pixel cockpit flag, no snap-under-head-movement (that's option 2, later, if ever).

---

## 1. What's actually wrong 🔍

`ffx_frameinterpolation.h` → `computeInterpolatedColor()`:

- two candidate warps per pixel: game vector (ours, dilated) and optical flow (FFX's own, 8×8 blocks)
- `fGameMvBias = sim(game warp) / sim(OF warp)` → final = lerp(OF result, game result, bias)
- screen content changes (PFD numbers tick, ND redraws) → game warp = identity → prev≠cur → low sim
- OF finds a block vector that "explains" the change → high sim → **OF wins → text warped by a blocky flow vector** 💥
- even when game wins: static screen = `lerp(prev, cur, 0.5)` = crossfade = double image 👻

Trace proof: in 1026 view body frame valid, camera delta 0, cockpit reprojection 0.000 px. Our vectors are right; FFX ignores them.

## 2. The fix in one line ✂️

> Where our dilated vector exists, the game result is final (no OF). Where it also says "not moving", output the current real frame's texel. Everything else = stock FFX.

## 3. Per-pixel rules 🧮

Inside `computeInterpolatedColor`, after the game samples + disocclusion block, before the OF block:

```glsl
// trust flag lives in the dilated MV texture, .z, render-res, same texel the field came from
FfxInt32x2 iTrustPx = FfxInt32x2(fUvInInterpolationRect * RenderSize());
const FfxBoolean trust = ((GetDispatchFlags() & FFX_FI_FLAG_MV_TRUST) != 0) && (LoadDilatedTrust(iTrustPx) > 0.5);

if (trust && (GetDispatchFlags() & FFX_FI_FLAG_MV_SNAP) != 0) {
    // half-frame displacement in DISPLAY pixels (field stores half vectors, UV units)
    const FfxFloat32 fPx = length(gameMv.fMotionVector * FfxFloat32x2(InterpolationRectSize()));
    const FfxFloat32 fEps = SnapEpsPx();   // bits 24..31 of dispatchFlags, 1/32 px units; 0 => 0.25
    const FfxBoolean still = fPx < fEps
                          && fDisocclusionFactor.x == 1.0 && fDisocclusionFactor.y == 1.0
                          && !gameMv.bPosOutside && !gameMv.bNegOutside
                          && fPrevColorGame.fBilinearWeightSum > 0.0 && fCurrColorGame.fBilinearWeightSum > 0.0;
    if (still) fInterpolatedColor = LoadCurrentBackbuffer(FfxInt32x2(iPxPos));   // exact texel, no blend
}

if (!trust) { /* stock OF block, unchanged */ } // trusted => fGameMvBias == 1, OF never touches the pixel
```

Notes 📝
- GLSL can't see the host enum: `ffx_frameinterpolation_callbacks_glsl.h` defines `FFX_FI_FLAG_MV_TRUST = (1u << 16)`, `FFX_FI_FLAG_MV_SNAP = (1u << 17)`, `SnapEpsPx()` = `(bits 24..31) / 32.0`, or `0.25` when the byte is 0. Same values as §5, both sides. One place each, comment cross-references the other.
- `fInPaintingWeight` untouched → disocclusion/inpainting logic identical to stock.
- snap picks **current**, not previous: screen updates land on the generated flip and repeat on the real one → no flicker, lowest latency.
- parked + still camera ⇒ whole frame snaps ⇒ generated frame == real frame. Correct, not a bug.
- with HUD-less attached, "current" IS the HUD-less image → snap copies HUD-less; inpainting pass re-stamps overlays afterwards (stock behaviour). Consistent ✅
- reset frame (`FrameIndexSinceLastReset()==0`) already copies current; untouched.

## 4. Data path 🚚

```
velocity target (RGBA16F)         xy = NDC disp, z = clip-w, w = coverage, cleared to kMvUnwritten = -60000
        │  fg_prepare.comp  (3x3 nearest-depth dilate, already there)
        ▼
dilated MV image  RG16F ──▶ RGBA16F   store (mvPix.x, mvPix.y, unwritten ? 0 : 1, 0)
        │  ffxGetResourceVK(g.dilMv[r], desc{format = R16G16B16A16_FLOAT})
        ▼
FFX FI main pass  binds r_dilated_motion_vectors @ binding 11   LoadDilatedTrust() = .z
```

- other FFX passes (`game_motion_vector_field`, `disocclusion_mask`) read `.xy` via untyped `texture2D` → unaffected ✅
- FFX's own dilate (`ffxFrameInterpolationPrepare`, rg16f UAV) is **never called** by the layer → no clash ✅
- component maps reflected SRV names → ids via `srvResourceBindingTable`; `r_dilated_motion_vectors` is already in it → one `#define` in the pass GLSL is all the binding work ✅
- VRAM: +14.7 MB (2 slots × 2560×1440 × 4 B extra) 🧾

## 5. Switches 🎛️

Flags word = `FfxFrameInterpolationDispatchDescription::flags` (uint32, copied verbatim into `cbFI.dispatchFlags`; FFX uses bits 0–3 only).

| bit(s) | name | default | live key |
|---|---|---|---|
| 16 | `FFX_FRAMEINTERPOLATION_DISPATCH_MV_TRUST` | on | `taa.fg_trust=1` |
| 17 | `FFX_FRAMEINTERPOLATION_DISPATCH_MV_SNAP` | on | `taa.fg_snap=1` |
| 24–31 | snap eps, 1/32 px, 0 ⇒ 0.25 px | 8 | `taa.fg_snap_eps=0.25` |
| 2 | FFX debug view (stock) | off | `taa.fg_debug=0` |

- keys read per dispatch in `fg::dispatchCallback` → A/B in flight, no relaunch 🔁
- one trace line on first dispatch: `FG TRUST: flags=0x%08x trust=%d snap=%d eps=%.2fpx`
- no constant-buffer layout change ⇒ only the **interpolation pass** needs regenerating

## 6. Files 📁

FFX (vendored, we own the fork):
- `sdk/include/FidelityFX/host/ffx_frameinterpolation.h` → 3 enum values in `FfxFrameInterpolationDispatchFlags`
- `sdk/include/FidelityFX/gpu/frameinterpolation/ffx_frameinterpolation_callbacks_glsl.h` → `LoadDilatedTrust()` inside the existing `#ifdef FFX_FRAMEINTERPOLATION_BIND_SRV_DILATED_MOTION_VECTORS` block; `SnapEpsPx()` next to `GetDispatchFlags()`
- `sdk/include/FidelityFX/gpu/frameinterpolation/ffx_frameinterpolation.h` → rules from §3
- `sdk/src/backends/vk/shaders/frameinterpolation/ffx_frameinterpolation_pass.glsl` → `#define FFX_FRAMEINTERPOLATION_BIND_SRV_DILATED_MOTION_VECTORS 11` before the callbacks include
- (hlsl callbacks header: mirror `LoadDilatedTrust` for symmetry, not compiled here)

Ours:
- `src/shaders/fg_prepare.comp` → binding 3 qualifier `rg16f` → `rgba16f`; store trust in `.z`
- `src/vklayer/fg_prepare.h` → `VK_FORMAT_R16G16_SFLOAT` → `VK_FORMAT_R16G16B16A16_SFLOAT` for `dilMv`
- `src/vklayer/fg_backend.h` → copy `sh.dilatedMotionVectors.resourceDescription`, set `.format = FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT`; build `fd.flags` from the live keys; trace line
- `build/ffx_shaders/ffx_frameinterpolation_pass{,_16bit,_wave64,_wave64_16bit}_permutations.h` → regenerated
- `config/taa_live.ini` → the 4 keys with comments
- `docs/` → this file

## 7. Build & regen 🔧

Tool: `build/ffx_permute.exe` (ours, `tools/ffx_permute.cpp`; AMD's needs MSVC). One run = one header.

```powershell
$G   = "third_party\FidelityFX-SDK\sdk\tools\binary_store\glslangValidator.exe"   # AMD's bundled 11.12.0, NOT the Vulkan SDK's 16.4.0 (see gate note)
$SRC = "third_party\FidelityFX-SDK\sdk\src\backends\vk\shaders\frameinterpolation\ffx_frameinterpolation_pass.glsl"
$INC = "third_party\FidelityFX-SDK\sdk\include\FidelityFX\gpu"
$OPT = "--permute","FFX_FRAMEINTERPOLATION_OPTION_LOW_RES_MOTION_VECTORS",
       "--permute","FFX_FRAMEINTERPOLATION_OPTION_JITTER_MOTION_VECTORS",
       "--permute","FFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH"       # same order as the struct in the existing headers

build\ffx_permute --glslang $G --src $SRC -I $INC -D FFX_GPU=1 -D FFX_GLSL=1 @OPT `
    --name ffx_frameinterpolation_pass               --out build\ffx_shaders\ffx_frameinterpolation_pass_permutations.h
build\ffx_permute ... -D FFX_HALF=1                   --name ffx_frameinterpolation_pass_16bit        --out ..._16bit_permutations.h
build\ffx_permute ... -D FFX_PREFER_WAVE64=1          --name ffx_frameinterpolation_pass_wave64       --out ..._wave64_permutations.h
build\ffx_permute ... -D FFX_HALF=1 -D FFX_PREFER_WAVE64=1 --name ffx_frameinterpolation_pass_wave64_16bit --out ..._wave64_16bit_permutations.h
```

🚧 **Gate — do this BEFORE touching any shader:** regenerate base + 16bit from the *unmodified* sources into a temp dir. Digests must be `59247a0deabd3e29` (base, 19108 B) and `d2e13637e46a9843` (16bit, 19140 B). Mismatch ⇒ wrong defines / include / glslang ⇒ fix the invocation first. Never debug shader edits and toolchain at the same time.

✅ Gate result 2026-09-10: with the Vulkan SDK's glslang 16.4.0 the blobs came out 232 B smaller with a different instruction mix (different `-Os` optimizer, same sources). With AMD's bundled glslang 11.12.0 (`sdk/tools/binary_store/glslangValidator.exe`) both digests reproduced exactly. `tools/ffx_regen_fi.ps1` uses the bundled one.

Facts that make the gate honest:
- existing wave64 headers are byte-identical to base/16bit (same digests) → the wave64 define is a no-op for this pass; any harmless define reproduces them
- FI main pass has **1 distinct blob** for all 8 keys → option order can't bite here
- on the 4060: `waveLaneCount 32/32` ⇒ wave64 never selected; `fp16Supported` ⇒ **the `_16bit` header is the one that runs** 🏃

Then:
1. `del build\ffx_obj\ffx_frameinterpolation_shaderblobs.o` and `ffx_frameinterpolation.o` (build.ps1 only recompiles when the .o is older than the .cpp; headers don't count)
2. `build.ps1` (sim running ⇒ `$env:MV_LAYER_OUT` staged DLL, swap at next launch)
3. pipeline creation failure shows as `FFX_ERROR_INVALID_ARGUMENT` from `patchResourceBindings` → visible in the trace as the FI context failing; treat as a naming bug (the SRV must be spelled `r_dilated_motion_vectors`)

## 8. Verification ✅

- [ ] gate digests match (§7) before any edit
- [ ] build: both .o recompiled, layer links, trace shows `FG: contexts created` and `FG TRUST: flags=0x08030000 ...` (bit 16 trust, bit 17 snap, byte 24–31 = 8 ⇒ 0.25 px)
- [ ] `FPS: ... (2.00x)` still holds, no `device lost`, no black frames
- [ ] visual, head still, FG on: PFD/ND/FMS text on generated flips == real flips (screenshot both; text edges identical)
- [ ] `taa.fg_trust=0` live ⇒ smear returns; `=1` ⇒ gone (proves it's this change)
- [ ] `taa.fg_snap=0` live ⇒ crossfade/double image on ticking numbers; `=1` ⇒ crisp
- [ ] head movement in 1026: no warp; softening ≤ TAA-level; no tearing at bezel edges (disocclusion path untouched)
- [ ] `taa.fg_debug=1` screenshot: vector field over the cockpit is flat/zero, world has motion
- [ ] chase view 1018: livery text on the airframe also crisp (same rule, body frame)
- [ ] a parked, still frame: generated == real (snap everywhere) — no shimmer
- [ ] perf: FI cost same or lower (one texelFetch added, 4 OF samples skipped where trusted)

## 9. Risks & trades ⚠️

- 🔸 anything without vectors (particles, smoke, some animated parts) rides the geometry behind it instead of OF → may **step** instead of smear. Same as TAA today. Accepted.
- 🔸 game-field inpainted holes (`bValid=false`) still take trust from our texel → they get the pyramid vector, no OF. Fine; edges are handled by disocclusion.
- 🔸 eps too high ⇒ slow pans snap (judder on distant terrain at <0.5 px/frame). 0.25 px half-step is ~1 px/frame at 2×; tune live if seen.
- 🔸 `rgba16f` qualifier MUST match the new view format for `imageStore` (VUID). Forgetting it = validation error / garbage vectors.
- 🔸 stale wave64 headers would still link, but regenerate all four anyway; mixed generations are how "works on mine" starts.
- 🔸 fp16 variant is the live one → test that one, not just the base.

## 10. Not in this spec 🚫

- option 2 (cockpit snapped under head movement, needs a per-pixel cockpit flag from the injection)
- HUD-less verification + the "how many backbuffer passes per frame" question (separate, already built 20:21)
- 3× FG, pacer rewrite, async compute

## 11. Numbers 📏

| thing | value |
|---|---|
| snap eps default | 0.25 px half-step (≈0.5 px/frame) |
| eps encoding | bits 24–31, 1/32 px, 0 ⇒ default |
| dilated MV format | RG16F → RGBA16F |
| extra VRAM @1440p | +14.7 MB |
| headers regenerated | 4 (base, 16bit, wave64, wave64_16bit) |
| .o to delete | 2 |
| live keys | `taa.fg_trust`, `taa.fg_snap`, `taa.fg_snap_eps`, `taa.fg_debug` |
