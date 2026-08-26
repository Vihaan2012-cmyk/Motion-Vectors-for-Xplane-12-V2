# MotionVectors - Deep Audit: what it does wrong & how to make it look better

Synthesis of eight adversarially-verified subsystem audits. Refuted findings dropped; overlapping findings merged to strongest evidence. Confidence labelled per line.

---

## VERIFICATION PASS (hand-checked against the source, 2026-08-26)

Every load-bearing citation below was re-read in the actual code. Corrections:

**Confirmed as written:** #8 (blendEnable-only test, layer.cpp:10556), #10 (resolve-miss
shake - the mod's own comment admits it, layer.cpp:3671-3678), #13 (frame-gen never
dispatched - g_fsr2LastDispatchCb declared at 704, never assigned), #16 (g_mv.failed
permanently sticky - mvDestroy's ONLY caller is inside mvCreate at mv_target.h:230,
BEHIND the failed early-out at 228), #20 (sharpen ungated but min/max-clamped, so
impact is modest), #22 (taa.mode defaults to 0=PASSTHROUGH), #26 (zero classic
render-pass handling - 0 grep hits), #32 (covMax loop runs with no live consumer).

**Downgraded:**
- **#7 (dstAlphaBlendFactor=ZERO): real mechanism, ZERO shipping impact.** The format
  IS RGBA16F (kMvFormat, mv_target.h:157) so alpha is stored and does get zeroed - but
  `unwritten` is classified by the -60000 sentinel (taa.comp:462), NOT coverage, and
  coverage's only two consumers both ship disabled (taa.reactive=0, taa.novec_cov=-1.0).
  Latent bug; fix before ever enabling the reactive mask (#21/#11-Part-2).
- **#12 (submission race): dormant.** The race detection keys on g_fsr2LastDispatchCb,
  which is never assigned (#13). Unreachable until FSR2 delivery is wired up.
- **#15 (jitter phases): mostly moot.** The "scale by upscale ratio" advice applies to
  FSR2-consumer mode, which never runs. For the same-res resolve, 8 phases is standard.
- **#2 (view-template crash origin): hazard real, crash link unproven.** The one
  2D_ARRAY/g_taa.layers/fmt template IS reused for history+sharpen+X-Plane's scene
  image (taa.h:516-542) - but every failure there `return false`s cleanly, so it does
  not by itself produce the null-descriptor crash. Still needs GPU-AV to name the crash.
- **#14 (jitter default 0.0): jitter is ON in practice** - live::f reads the ini first
  and the shipped ini has jitter_scale=1.0, which the launcher seeds. Residual risk is
  the empty-regeneration hole below.

**NEW findings from this pass:**
- **N1 (high, robustness): a deleted taa_live.ini permanently disables TAA with no
  recovery path.** The layer regenerates a deleted ini EMPTY; the launcher's seed copy
  uses CopyFileA(..., TRUE) = never-overwrite (launcher.cpp:119), so the empty file
  blocks re-seeding forever. All tuned values fall to compiled defaults - and the
  compiled defaults are taa.mode=0 (PASSTHROUGH) and jitter_scale=0.0 - so TAA is
  silently dead until the user hand-deletes the ini and relaunches via the launcher
  in the right order. Fix: ship sane compiled defaults (mode=2, jitter 1.0) OR make
  the launcher seed when the existing ini is empty/missing keys.
- **N2 (medium): release upgrades never update existing users' tuning.** Same
  never-overwrite copy: a user who ran any older version keeps their old %TEMP% ini
  forever, so shipped tuning improvements (e.g. 1.0.3's) never reach them. Needs a
  version key + merge.
- **N3 (low): stale log string** - mv_target.h:257 reports "R16G16_SFLOAT" but
  kMvFormat is R16G16B16A16_SFLOAT.

## GPU-AV RUN (2026-08-26, MSAA on + TAA_KEEP_ON_777=1, crashed as expected)

Captured live before the crash (the validation log itself was truncated on
reopen - only the semaphore VUID survived on disk; these were caught by the
monitor as they streamed):

- **VUID-VkRenderingInfo-multisampledRenderToSingleSampled-06857** - attachments
  in one VkRenderingInfo created with DIFFERENT sampleCounts. Driver-confirmed
  finding #1: the 1-sample velocity target (mv_target.h:245) is appended beside
  X-Plane's multisampled colour/depth.
- **VUID-vkCmdDrawIndexed-multisampledRenderToSingleSampled-07285** - the same
  mismatch at draw time (pipeline vs attachments).
- **VUID-vkCmdBeginRendering-pRenderingInfo-09592** - **N5 (new, confirmed)**:
  a colour attachment is accessed in a layout it is not actually in - the layer
  does not transition the velocity image to the layout it declares.
- **VUID-vkQueueSubmit-pSignalSemaphores-00067** (20x, hit the dup limit) -
  **N4 (new, confirmed)**: a binary semaphore is signalled while already
  signalled. Prime suspect: the VRAM upload governor holding/reordering whole
  submissions (vram::onSubmit) around X-Plane's semaphore chains.
- Warnings: patched fragments write the injected velocity output (Location 1)
  in single-attachment passes where the slot's imageView is NULL - the known
  writeMask-0 compromise, benign by design.
- The compute null-descriptor error never appeared: the crash comes from the
  render-pass UB upstream of the resolve. Fix ladder: read the pass's sample
  count and DECLINE to append velocity when samples > 1 (replaces the
  by-aircraft 777 bypass); transition the velocity image to the declared
  layout; audit vram::onSubmit's held-submission semaphore ordering.

---

## Part 1 - What the mod does wrong

### A. MSAA path is comprehensively broken (this is why the FF777 needs the bypass)
Several independent confirmed defects converge here. Treat MSAA as unsupported until the whole cluster is fixed.

1. **Velocity attachment is hard-coded single-sample and is injected into passes without matching the pass's sample count.** `mv_target.h:245` (`ici.samples = VK_SAMPLE_COUNT_1_BIT`, no MSAA branch); injection at `layer.cpp:4215` never reads `rasterizationSamples`/pass sample count. Binding a 1x image beside N-sample attachments violates the Vulkan same-sample-count rule and reaches the driver unchecked. **[confirmed, critical]**

2. **taa.h reuses one `VK_IMAGE_VIEW_TYPE_2D_ARRAY` view template with a caller-supplied `layerCount` (`g_taa.layers`) for history, sharpen, AND X-Plane's own `scene` image.** `taa.h:516-542` (layerCount from `g_taa.layers` set once at `taa.h:434`). If scene's real shape differs, `vkCreateImageView` fails → null descriptor → the established compute-dispatch crash chain. This is the most likely concrete origin of the FF777 null-descriptor crash. Fix: read scene's real `arrayLayers` from `g_colorImages` (already tracked) at this call site. **[plausible, high]**

3. **X-Plane's own `vkCmdResolveImage` is intercepted as pure pass-through, so under MSAA it overwrites the mod's TAA/sharpen output every frame.** `layer.cpp:6455-6515` (log-once, then `next(...)` unmodified). Note the tension: the mod's own trace found this call site inactive under the tested config (`renopt_MSAA=0`, `layer.cpp:6331-6332`), so it only bites when MSAA is actually enabled — but then it silently discards all the mod's work. **[confirmed as code path; active only when MSAA on]**

4. **Depth selection accepts multisampled depth with only a log warning, unlike colour which hard-excludes it.** `layer.cpp:1858-1859` (warn only, `g_sceneDepth` already assigned at 1834) vs colour's `usable = samples==1` at `layer.cpp:4461`. No `g_sceneDepthResolveImage` counterpart exists. Reprojection/disocclusion then reads raw MSAA depth through a non-MS view. **[confirmed, high]**

5. **G-buffer targets are frequently BOTH array-layered (stereo) AND multisampled**, and X-Plane runs native MS-storage-image consumers (`fix_hdr_1`, `msaa_categorize` with a spec-constant sample count defaulting to 4 that calls `OpTerminateInvocation`). q5:192-194, q5:114-127, 03_fsr:92-97. The mod's fixed-shape image creation cannot satisfy these. **[plausible/confirmed, high]**

6. **The documented TAA insertion point ("last draw before fix_hdr/tonemap") does not exist in every config: MSAA-on + bloom/FXAA-off routes HDR straight to a fused resolve+tonemap (`resolve_3/_7`) and skips the entire `hdr` family.** q5_hdr_chain.md:167-170, 187-190. Pass-detection assuming a distinct pre-tonemap HDR pass silently finds nothing. **[confirmed, critical (X-Plane side); plausible the mod's runtime selector is actually fooled]**

### B. Velocity correctness / coverage invariant

7. **The velocity attachment's alpha blend uses `dstAlphaBlendFactor = ZERO`, so a transparent overlay texel force-resets stored coverage to 0 even though the RG SELECT beside it correctly preserved the world's velocity.** `layer.cpp:10765-10774` + `spirv_inject.h:1779-1786`. This desyncs coverage from velocity and fires the resolve's `covMax<0.5` reactive mask (`taa.comp:966`) on pixels whose velocity was in fact correct → flicker/ghosting at glass edges. Fix: `dstAlphaBlendFactor = ONE_MINUS_SRC_ALPHA` (or `BLEND_OP_MAX` on alpha). **[confirmed, critical]**

8. **`alphaBlended` is set from `blendEnable==VK_TRUE` alone, with no check that the blend factors actually mean SRC_ALPHA=opacity.** `layer.cpp:10556-10558`, consumed `spirv_inject.h:1779`. Additive glow/fire/smoke pipelines (ONE/ONE) would still be routed through the coverage-gate/SELECT and have an unrelated alpha thresholded at 0.5. Not yet proven X-Plane ships such pipelines through this path. **[plausible, high]**

9. **Forward-composited effects — clouds, volumetric fog, rain/particles, light sprites/billboards, ocean shading — never pass through `deferred_gbuf`, so the mod's TAA reprojects them with the underlying world (or `kMvUnwritten`) velocity, not their own.** `layer.cpp:5426-5434`; corroborated by q6 (clouds write dedicated targets, never the g-buffer). This is the single largest source of visible ghosting/smearing on the most eye-catching moving elements. **[confirmed, critical]**

10. **Resolve-miss frames present the raw jittered picture → visible shake.** `layer.cpp:3660-3690` (the mod's own comment + `g_resolveMissFrames`/`g_resolveOkFrames` counters exist precisely because this was observed). Jitter-push and resolve-dispatch are decoupled; any per-frame desync shakes the image. **[confirmed, high]**

11. **`deferred_gbuf` is the deferred LIGHTING composite (writes one `data_out0`), not the 5-attachment G-buffer writer — that's the terrain/planet family.** q5:16-20,132-142; q3 max fragment-output Location=4 only in terrain/planet. The mod's `colorAttachmentCount==5` heuristic therefore targets terrain/planet, and the "attachments 1-4 aren't written" concern is false for terrain (it writes all 5). Scene-detection and "does X-Plane already emit velocity" checks were aimed at the wrong pass. **[confirmed, critical]**

### C. Submission ordering / frame-gen

12. **The delivery-before-dispatch submission-order race is detected but never corrected** — after one log line it goes silent every frame. `layer.cpp:1084-1107` (`noteSubmitOrder`, one-shot `told` guard), `1109-1137` (forwards only; `g_seqOfDispatchCb`/`g_seqOfDeliveryCb` used nowhere else). If it fires, the compute delivery reads `outImg` before the write completes → sporadic corruption. **[confirmed, critical]**

13. **Frame generation is vendored and compiled but never dispatched.** `third_party/FidelityFX-SDK/.../FrameInterpolationSwapchain/*` + built `build/ffx_obj/FrameInterpolationSwapchainVK*.o`; zero `FrameInterpolation`/`FfxFsr3` references in `layer.cpp`/`plugin.cpp`. The advertised frame-gen feature does not run. **[confirmed, high]**

### D. Jitter / temporal setup

14. **Jitter scale defaults to 0.0 in code, and three files disagree** (`layer.cpp:3648`/`7989` = 0.0f; `config/taa_live.ini:41` = 1.0; `lua/MotionVectors.lua:508` def = 0.0). If the ini is missing/regenerated, jitter silently collapses to 0 → zero sub-pixel diversity → image more aliased than stock, no error shown. `layer.cpp:11560-11561` multiplies the offset by `g_jitterScale`. **[confirmed, high]**

15. **FSR/TAA jitter phase count is a flat 8, never scaled by the render/display upscale ratio.** `plugin.cpp:3838` (`phases = g_jitterPhases`), `renderScale` stored (`3862`) but never fed back. At any upscaled `render_scale`, the accumulator gets too few sub-pixel samples → persistent softness/aliasing that never fully converges. (Note: AMD's FSR2 SDK is not actually linked; this is the mod's own resolve — the image-quality point stands.) **[confirmed, high]**

16. **`g_mv.failed` is permanently sticky for the process life** — one transient allocation failure disables velocity/TAA forever with no retry. `mv_target.h`: the `if (m.failed) return false;` early-out precedes the only reset in `mvDestroy`. Matches the "restart to fix" symptom. **[confirmed, high]**

17. **Scene-pass colour-attachment-count detection is a one-way ratchet (only increases).** `layer.cpp:3594-3596`. If the real scene pass's colour count legitimately drops, `isSceneSized` permanently rejects it, freezing render size and defeating stale-size auto-recovery. Same class already fixed for width/height, left unfixed here. **[confirmed, high]**

### E. Medium

18. **X-Plane's own per-pixel moving-geometry flag `gbuffer_vel` (uint, bit 2) is located but never consumed by the resolve** — the mod relies solely on the coarse `covMax<0.5` alpha heuristic and a distance proxy (`g_nearFieldM = 2.0m`, on by default, `layer.cpp:3719`). `layer.cpp:6186-6204` records the candidate for logging only. Binding bit 2 would give ground-truth cockpit-vs-world / moving-geometry classification. **[confirmed candidate; plausible it helps FF777]**

19. **~4-second (240-frame) velocity/resolve blackout on every resolution / render-scale / aircraft change before auto-recovery fires.** `layer.cpp:7647-7649`. Reacting to the change event directly (the plugin already knows) would eliminate the multi-second dropout. **[confirmed, medium]**

20. **`MODE_SHARPEN` sharpens every pixel uniformly with no confidence/convergence gating** — a fresh/disoccluded/reactive-masked pixel is sharpened identically to a converged one. `taa.comp:370-396` (separate early-return branch, 5-tap cross, no access to `a`/`moved`/`covMax`). **[confirmed, medium]**

21. **The reactive/transparency mask (prop-disc anti-ghost) ships off** (`taa.h:334`, `taa_live.ini:37`), so animated-transparency ghosting is unmitigated. Two prior default-on bugs mean flipping it needs a fresh A/B pass, not a one-line flip. **[plausible, medium]**

22. **`TAA_MODE` defaults to 0 (MODE_PASSTHROUGH, a no-op) independently of `taa.enable`.** `taa.h:247-248`, `taa.comp:406-409`. Latent footgun: enabling TAA without also setting mode does nothing. Shipping ini pairs them correctly, so it only bites a UI/tool that decouples them. **[plausible, medium]**

23. **A full-resolution compute delivery pass runs every frame to route around an un-root-caused 16F→8-bit blit corruption.** `layer.cpp:4937-4964` (every 16F→8-bit blit is "GARBAGE", same-format blits clean). Paid as a per-frame dispatch instead of fixed at source (likely a missing `MUTABLE_FORMAT` + matching view, or usage flag). **[confirmed, medium/perf]**

24. **`bodyReproj` treats the whole cockpit as one rigid body**, so animated manipulators (yoke/throttle/trim) moving relative to the airframe would be reprojected with airframe motion → local ghosting/flicker during animation. `plugin.cpp:3739`. Shader-side selection not fully traced. **[plausible, medium]**

25. **A distinct near-field / body-reprojection velocity substitution near the cockpit/world boundary is an untested FF777-shimmer candidate.** Real code at `layer.cpp:7957-7995` (near-field threshold) and `11350-11401` (per-vertex bodyReproj with identity fallback) — separate from the already-disproven glass-stamping theory. Worth instrumenting the identity-vs-real-bodyReproj branch on the FF777 boundary. **[plausible, medium]**

26. **The mod has zero classic `VkRenderPass` handling (dynamic-rendering only).** Grep of `layer.cpp` for `vkCreateRenderPass` = 0 hits; X-Plane's binary resolves both `vkCreateRenderPass` and `vkCmdBeginRendering`. Any velocity-relevant pipeline created with a non-null renderPass would silently get no velocity. Needs one check: does X-Plane use classic render passes for anything depth-testing/vertex-attributed. **[plausible, medium]**

27. **`fix_hdr`'s NaN/Inf scrub runs in-place on the HDR image; if history is fed from before the scrub, a single NaN/Inf draw can be accumulated and persist across frames.** Mod has no `fix_hdr` awareness (grep=0); its raster-only pass selector structurally cannot land after that compute dispatch. Cheap mitigation: isnan/isinf clamp before writing history. **[plausible, medium]**

28. **Stale `see fsr2_pass.h` comment pointers** at `layer.cpp:11520` and `plugin.cpp:2604` — the file does not exist; the load-bearing jitter-amplitude/phase-count docs it cites are unfindable. **[confirmed, medium/robustness]**

### F. Native-engine interactions (X-Plane ground truth)

29. **X-Plane runs three-to-four independent, mod-blind per-effect temporal accumulators** (cloud_upscale, volumetric_fog with froxel history and *no clamp*, sky-LUT EMA, cloud_shadows), each with its own matrix reprojection and no motion-vector awareness. q6:293-302, 03_fsr:70-71. The mod's TAA re-blends already-blended, already-lagging signals → compound lag on clouds/god-rays. Whether the mod's injected jitter actually leaks into the shared uniforms these read is unverified. **[confirmed conflict; jitter-leak plausible]**

30. **SSR is already handled** — the mod disables it during TAA via `sim/private/controls/debug/kill_ssr` (`plugin.cpp:1082-1103`, opt-out `TAA_KEEP_SSR`), having diagnosed the exact reflection feedback loop. Only residual concern: verify `kill_ssr` restores even after a crash so SSR isn't left permanently off. **[confirmed handled; minor]**

### G. Low-severity / robustness / cleanup

31. **Injected varying Location pair is chosen once from a coarse device-wide component budget; the precise per-module census only runs reactively.** `spirv_inject.h:405-472` (the code's own comment admits "a ceiling enforced only there is a ceiling not enforced at all"; documents two prior shipped silent-zero-velocity bugs). A collision produces zero velocity indistinguishable from correct output. Run the census unconditionally and mask-to-0 on overflow. **[confirmed, medium→low]** *(Note: device-limit query and loud-warning path DO exist at `layer.cpp:12483-12526` — the earlier "never validated" finding was refuted; the residual gap is that a min-spec device warns but still proceeds.)*

32. **Reactive-mask `covMax` loop (9 texture fetches/pixel) runs unconditionally even though `FLAG_REACTIVE` ships off.** `taa.comp:498-503` vs consumer at `966`. Pure wasted work; wrap in the flag check. **[confirmed, low/perf]**

33. **Variance clamp collapses to the hard 3×3 box at the shipping `varClip=8.0`** (`taa.comp:812-815`, `taa.h:257`). Important corrective: the file's own recorded experiment (`taa.comp:889-914`) shows tightening it to 1.25 "changed nothing" for the smear/ghost bug, because reprojection drift is a locally-plausible value, not a statistical outlier. **Do not chase ghosting by tightening the clamp.** The darkness/luma-based narrowing (`taa.comp:812-813`) is likely also inert for the same reason. **[plausible, low]**

34. **`mvFragHash` is a raw FNV-1a over all SPIR-V words with no debug-info stripping**, so cosmetically-different-but-identical modules silently fall out of the `TAA_MASK_FRAG` allowlist. `layer.cpp:10113-10131`. **[plausible, low]**

35. **In-file docs contradict themselves on the velocity attachment's channel count** (2-channel legacy vs 4-channel RGBA16F-with-coverage, 9 lines apart, `layer.cpp:10720-10735`, stale echo `~10753`) — this stale framing is what obscures finding #7. **[confirmed, low]**

36. Other confirmed dead/inert code, low impact: **`TAA_MASK_SEETHROUGH` heuristic is coarse** (matches opaque HUD/decals, off by default, `layer.cpp:10537-10550`); **coverage gate has no partial-coverage representation** (deliberate, documented `spirv_inject.h:1699-1723`); **legacy viewport-jitter path ignores `g_jitterScale`** (off by default, `layer.cpp:3832`); **camera cross-check is log-only** (`plugin.cpp:3060-3093`); **dead `px/py/pz` locals** (`plugin.cpp:3155-3160`); **dead `g_fsr2ShimsBound`** (`layer.cpp:1055`); **jitter/reproj state is a single global with no per-view slot** (VR-limiting, `plugin.cpp`/`share.h`); **VR provisional-arming of body-reproj has no head-tracking awareness** (`layer.cpp:11583-11598`); **Catmull-Rom history resample bottoms out on bilinear sub-taps** (accepted approximation). **[confirmed/plausible, low]**

*(X-Plane has zero native motion vectors and zero whole-scene TAA — confirmed corpus-wide, q6. This is the mod's foundational premise, not a bug: any main-scene instability is 100% the mod's own history/clamp logic, with no native safety net.)*

---

## Part 2 - How to make it LOOK BETTER

Ordered highest impact-to-effort first. Several interact — noted inline.

**1. Turn jitter back on with a validated default (attack: aliasing/shimmer on distant geometry). Effort S+M.**
Set the canonical `taa.jitter_scale` default to 1.0 in all three places (`layer.cpp:7989`, `lua:508`, ini already 1.0). Today it defaults to 0.0 → the accumulator samples the same sub-pixel position every frame → zero super-sampling benefit while paying full TAA cost. *Expected:* markedly less edge aliasing and distant-building crawl. *Risk:* uncancelled jitter = whole-frame shake (the stated reason it was disabled) — you MUST first validate unjitter cancellation on a static scene. *Interacts with #2:* jitter is only safe once resolve reliably runs every frame.

**2. Couple jitter emission to the resolve gate (attack: intermittent shake/shimmer). Effort M.**
Key jitter off the same latch that decides whether resolve will run, so a non-resolving frame also doesn't jitter (`layer.cpp:3660-3690` counters already detect the miss). Surface `g_resolveMissFrames` where a user can see it in normal flight. *Expected:* eliminates the shake on camera cuts / heavy frames. *Risk:* low. *Prerequisite for #1 to be safe.*

**3. Fix the velocity alpha blend factor (attack: flicker/ghosting at glass edges). Effort S.**
`layer.cpp:10774`: `dstAlphaBlendFactor = ONE_MINUS_SRC_ALPHA` (or MAX on alpha). Stops the reactive mask firing on pixels whose velocity was correctly preserved behind transparent overlays. *Expected:* less edge flicker around canopy glass and transparent overlays. *Risk:* low — no other channel depends on the ZERO factor.

**4. Give forward-composited effects correct velocity OR a history-reset sentinel (attack: cloud/rain/sprite ghosting — the most visible ghosting). Effort M-L.**
Clouds, fog, rain, light sprites, ocean shading never pass `deferred_gbuf` (`layer.cpp:5426-5434`) and inherit background velocity. Either write a reserved sentinel at these pixels and force full history reset in `taa.comp`, or inject coarse camera-relative velocity into `volumetric_apply`/rain/light shaders using their existing view/proj matrices. *Expected:* large reduction in smear/ghost trails on clouds, rain, and light points. *Risk:* medium (getting the sentinel path wrong reintroduces flicker). *Interacts with #5.*

**5. Reduce mod history weight / force fast rejection over already-temporally-resolved regions (attack: cloud/god-ray lag & double-blending). Effort M.**
Use the injector's coverage `.a` channel as a proxy for "forward-composited / native-accumulator region" and lower the mod's blend weight there, deferring to X-Plane's own converged cloud_upscale/volumetric_fog accumulation (q6:293-302). *Expected:* crisper, less-laggy clouds and volumetric light. *Risk:* low-medium. *Builds on #4.*

**6. Wire `gbuffer_vel` bit 2 into the reactive mask (attack: cockpit-vs-world ghosting, candidate FF777 fix). Effort M.**
Bind X-Plane's own uint flags image (located at `layer.cpp:6186-6204`), test bit 2, OR it into the existing `covMax<0.5` mask. Requires a one-frame capture to fix the view type/sample index. *Expected:* precise per-pixel moving-geometry masking — the first real shot at the cockpit-vs-world classification the distance heuristic only approximates, and a genuine FF777-shimmer candidate. *Risk:* medium (wrong bind = crash/no-op). *This is the most promising principled fix for the unresolved 777 bug.*

**7. Confidence-gate the sharpen pass (attack: crawling on disocclusion trails, over-sharpen). Effort S-M.**
Stash `1 - a` (convergence) in the resolve output alpha and scale `pc.sharpen` by it (`taa.comp:370-396`). *Expected:* sharp where converged, calm on fresh/disoccluded pixels — less shimmer along moving silhouettes. *Risk:* low.

**8. Clamp NaN/Inf before writing history (attack: stuck bright/dark specks). Effort S.**
One isnan/isinf clamp in the resolve before the accumulate write. *Expected:* removes rare persistent hot/dead texels. *Risk:* none.

**9. Scale jitter phase count by upscale ratio (attack: residual softness/aliasing when upscaling). Effort S.**
When `render_scale < 1`, scale `phases` toward `~8/renderScale²` clamped to the existing 1..64 (`plugin.cpp:3838`). Re-validate the multiplier against the mod's own convergence. *Expected:* sharper, better-converged upscaled image. *Risk:* low.

**10. Kill the resolution/aircraft-change blackout (attack: multi-second TAA dropout that reads as a pop/shimmer). Effort S-M.**
React to the change event directly instead of waiting 240 stale frames (`layer.cpp:7647`); drop the permanent `g_mv.failed` sticky in favour of a cooldown retry. *Expected:* no 4-second velocity/TAA dropout after resolution or aircraft swaps. *Risk:* low.

**11. Re-evaluate defaulting the reactive mask on (attack: prop-disc / animated-transparency ghosting). Effort M (validation-heavy).**
`taa.h:334` / `taa_live.ini:37`. *Expected:* less propeller-disc and animated-glass ghosting. *Risk:* medium — this exact mechanism shipped two default-on regressions before; requires an A/B pass, not a flip.

**12. MSAA: keep the clean 777 bypass short-term; treat full MSAA support as a separate L project.** The Part 1-A cluster (sample mismatch, view crash, native-resolve clobber, MSAA depth) means MSAA cannot currently produce a correct image. *Expected:* no crash / no garbage for MSAA users. *Risk/effort:* high/L to actually support; the bypass is the right near-term call.

---

## Part 3 - What needs a GPU capture, not more inference

A RenderDoc + GPU-AV frame capture in the FF777 cockpit (and one MSAA-on capture) can settle what static analysis cannot:

1. **What actually writes the velocity texel under a shimmering FF777 building** — glass draw, a forward-composited effect (finding #9), or the near-field/body-reproj SELECT (`layer.cpp:7957-7995`, `11350-11401`)? Inspect the velocity target where the shimmer appears.
2. **The exact failed `vkCreateImageView` under MSAA** — run GPU-Assisted Validation to name it (per the memory note, don't infer): is it the mv target (`mv_target.h:245`), the `taa.h:516-542` 2D_ARRAY view with wrong `layerCount`, or a native `fix_hdr_1`/`msaa_categorize` view?
3. **Does X-Plane's own `vkCmdResolveImage` actually run and land on `g_sceneColor` when MSAA is enabled** (`renopt_MSAA=1`)? The trace check at `layer.cpp:6500-6511` already exists — confirm `g_msaaResolveDst == g_sceneColor.image`.
4. **Which `VkImage` is bound to X-Plane's native FSR `u_input_texture` at EASU dispatch** — pre- or post-mod-resolve? Decides whether injected jitter reaches the jitter-blind FSR1 EASU edge estimator.
5. **Does the mod's jitter leak into the shared view/proj uniforms** that cloud_upscale / volumetric_fog / cloud_shadows read (finding #29)?
6. **The view type + sample index needed to bind `gbuffer_vel`** (blocks Part 2 item #6).
7. **Which buffer feeds next frame's `tex_ssr`** — confirm `kill_ssr` fully closes the reflection feedback loop and that its restore path survives a crash.
8. **Is the `scene` image actually arrayed and/or multisampled at the `taa.h` view-creation call site** — confirms finding #2's root cause and the exact `layerCount` to use.