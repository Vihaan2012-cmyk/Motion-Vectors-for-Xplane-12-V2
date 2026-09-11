# MotionVectors "Lifelike" — Neural Rendering Suite Design

**Date:** 2026-09-08
**Status:** design locked, pending build
**Owner:** Vihaan2012

## Goal

Make X-Plane 12 as lifelike as possible via a **suite of additive neural /
enhancement passes** inside the MotionVectors Vulkan layer. Every pass augments —
never replaces — the existing pipeline, reads a shared G-buffer feed, and is
gated behind its own live flag. Entirely our own code and models: no NVIDIA
gate, no signed binary, no driver patching, zero circumvention.

## The suite

| # | Module | What it does | Maturity |
|---|--------|--------------|----------|
| 1 | **Neural SR/AA** | Temporal super-resolution + anti-aliasing; arbitrary input→output ratio incl. non-integer (1440p→2160p); residual on top of TAA/TAAU | core — build first |
| 2 | **Adaptive light** | Neural bounce/GI + adaptive exposure / local tone-mapping, from the shared G-buffer feed | new — the ray-march SSGI (`gi_gather`/`gi_denoise`) **stays in-tree, off by default** (`taa.gi=0`) until this module replaces it; it was a no-op through 1.2.0 and is not a dependency of anything here |
| 3 | **Photoreal enhancement** | G-buffer-conditioned net (Intel "Photorealism Enhancement" lineage) pushing frames toward photographic reference, kept stable by our motion vectors | moonshot — horizon, not week one |

## Non-negotiable constraints (from the user)

1. **Additive, not replacing.** Every pass outputs a *residual correction* added
   on top of the existing result. Physically cannot blow up the image — worst
   case it adds ~zero and we fall back to the stock frame. The stock pipeline
   always runs underneath.
2. **Arbitrary-ratio scaling.** Continuous input→output resampling; non-integer
   ratios; target exact display resolution (4K native, etc.).
3. **Generalize across X-Plane.** Robust across the full content distribution —
   every biome, weather, time of day, altitude, aircraft, third-party scenery —
   independent of which addons a given user has installed.
4. **Lifelike is the goal**, not just sharper — lighting, exposure, and
   photoreal enhancement are in scope, not only resolution.

## Efficiency thesis (honest framing)

We will not out-muscle NVIDIA on raw fidelity — theirs generalizes over *every*
game. Our edge is the opposite: the nets only have to be excellent at *one*
title, so they can be far smaller/faster for the same quality. That
specialization is the real, defensible "more efficient than DLSS" claim — not a
promise to beat it on absolute quality.

Note: X-Plane 12 is rasterized (deferred G-buffer, shadow maps, SSAO, PBR), not
path-traced. DLSS Ray Reconstruction's "denoise ray-traced lighting" benefit has
no noisy-RT input here — so our lighting work is *added* GI/exposure/photoreal,
computed from the G-buffer we already have, not RT denoising. DLSS-D Ray
Reconstruction already runs on the 4060 for the plain SR/AA piece; this suite is
our own, tunable, additive layer.

## Shared architecture — why the big vision is buildable

All three modules read the **same G-buffer feed** and write **additive residuals**
through the **same descriptor/history rig**, so we build ONE framework and each
capability plugs in:

- **G-buffer feed (status as of the 2026-09-08 tree, Sept-5-b base):**

| Input | Source in layer | Status |
|-------|-----------------|--------|
| HDR scene color (low res) | `g_sceneColor` / `g_taa` resolved view | identified |
| albedo | the scene pass's other colour attachments (`passInfo.color*`); engine name not yet known — only `gbuf-normal` and `gbuf-depth` are named in the trace | **to identify (Stage 0 task)** |
| normal | `g_gbufNormalImage` ("gbuf-normal") | identified |
| material | as albedo | **to identify (Stage 0 task)** |
| depth | `depthcopy::state().image` (R32_SFLOAT, barriered copy — the resolve reads this since 2026-09-08; the direct `g_engineDepthImage` read was measured noisy) | identified |
| velocity | `g_mv`, R16G16_SFLOAT |
| jitter offset | snapshot `jitterX/Y`; taau.comp push-constant `shift` |
| env probes | engine cubemaps (already bound by `gi_gather`) |
| output-res history | taau/gi ping-pong pairs |

- **Additive residual discipline:** each pass adds a correction to the current
  frame; a bug degrades to +0, never to a broken image.
- **Live flags:** `taa.nn` (SR), `taa.light` (neural bounce/exposure/tonemap),
  `taa.photoreal` — each independent, off by default.
- The feed is proven: `taau.comp` (SR host) already binds
  color+velocity+depth+jitter+history correctly. The SSGI pass (`gi.h`, still
  in-tree, off) demonstrates that normal + env-probe + the `u_gbuffer_data`
  block are bindable from the same feed; the resolve's own taps (bindings
  5/7/8/9/10/12) are the reference implementation for each.
- **Capture path to extend:** `shotMaybe`/`ShotCap` in layer.cpp (host-visible
  readback buffer + `cmdCopyImageToBuffer` at present time + BMP writer, 12-slot
  ring under `D:\TAA Dumps`). The harness generalises it to N images per frame
  and a lossless format; BMP stays for the preview only.
- **Pose/jitter source:** the plugin share block (share.h): `proj[16]`,
  `world[16]`, `currViewProj`/`prevViewProj`/`invCurrViewProj`, `camX/Y/Z`,
  `fovDeg`, `nearClip/farClip`, `viewportW/H`, `renderScale`, `jitterX/Y`,
  `jitterIndex/Phases`, `simTime` — all already snapshotted per frame by the
  layer.

## Where the SR pass plugs in

`taau::record()` (src/vklayer/taau.h:406) runs right after X-Plane's EASU and
before RCAS, binding srcView (low-res color), velView, jitter, and the
output-res history ping-pong. The neural SR pass sits immediately after taau's
history integration: read {current low-res color, warped output-res history,
velocity, depth} → predict an output-res residual → add to taau's output before
RCAS. Reuses taau's descriptor set / ring / sampler rig (taau.h).

Runtime inference: small CNN → SPIR-V compute, or cooperative-matrix
(`VK_NV_cooperative_matrix`) on the 4060's tensor cores. Tiny by design, sized to
the frame budget.

## Training data — the addon-independence insight

Scraping frames from a live sim bakes in the capturing install's addons
(AutoOrtho tiles, custom scenery/aircraft). A net trained on that learns *that
install*, not X-Plane. Because we know X-Plane's exact shaders, we know the exact
forward/degradation model:

> high-res clean frame --(jitter + downsample, exactly as taau/taa.comp do it)--> low-res input + velocity + depth + jitter

That degradation is identical on every install; only content varies. So:

- **Primary — shader-exact synthetic pairs:** ground-truth high-res
  (supersampled / accumulated) degraded through the true jitter+downsample path.
  Universal across installs; input↔target mapping matches the runtime exactly.
- **Supplement — real in-sim captures** across varied conditions, for content
  realism; kept supplementary so no install's addons dominate.
- **Coverage:** broad sampling over biome/weather/time/altitude/aircraft +
  augmentation + multiple scale ratios (continuous-ratio upscaling, not one factor).

## Pipeline stages

0. **Capture harness** *(first deliverable — NOT started as of 2026-09-08 22:00; plan next)* — extend the
   swapchain dump path (`shotMaybe`/`ShotCap`) into aligned **full-G-buffer**
   training tuples: color + albedo + normal + material + depth + velocity +
   jitter + high-res reference, tagged with scale ratio + camera pose, to disk
   with a per-frame manifest. Reuses the cmdCopyImageToBuffer + host-mapped
   readback + BMP writer that already exist.
1. **Forward-model data generation** — shader-exact synthetic pairs; supersampled
   ground truth; broad sampling; real-capture supplement.
2. **Model + training** (PyTorch, 4060) — start with SR residual net; loss =
   perceptual + metrics-suite stability terms (S_FLICKER, S_STATIC_DELTA,
   S_GHOST, S_LAPLACIAN, S_QUADVAR). The metrics suite is already our yardstick,
   so it becomes the objective directly.
3. **Export + in-plugin inference** — weights → SPIR-V / coopmat; additive pass
   wired into taau; live flag off by default.
4. **Validate** — metrics suite A/B over a recorded flight (stock vs. + residual).
5. **Extend to light + photoreal** — reuse framework; build the neural
   adaptive-light module (bounce/GI + exposure/tonemap) from scratch on the
   G-buffer feed, then the photoreal capstone.

## Open decisions (settle during build, not blocking)

- Model family/size (start U-Net-lite residual; shrink to budget); coopmat vs. plain compute after a perf probe.
- Ground truth: supersampled single-frame vs. accumulated static multi-frame (likely both).
- Export format: raw SPIR-V weave vs. weight-blob + generic inference shader.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
