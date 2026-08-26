# MotionVectors — Audit: what the mod could be doing wrong

**Scope of this pass:** the velocity-injection path (`spirv_inject.h`), the resolve
shader (`shaders/taa.comp`), the pass-identification / attachment logic
(`layer.cpp`), and the plugin gates (`plugin.cpp`) — cross-checked against
**X-Plane's real shaders**, disassembled from `Resources/shaders/bin/spv/*.xsa`
(they are ZIP archives of raw SPIR-V; `spirv-dis` reads them).

This is FIND-only. Nothing here is fixed. Confidence is labelled per finding.

---

## Ground truth established from X-Plane's own shaders

These are facts read directly out of the disassembled SPIR-V, not inference:

- **`deferred_gbuf` (the G-buffer geometry shader, 986 permutations)** — checked
  the simplest, a mid, and the *heaviest* permutation (`_1`, `_100`, `_739`).
  Every one writes a **single** fragment output `data_out0` at **Location 0**,
  and uses exactly **one input varying, `v_eye` at Location 5**. No permutation
  goes anywhere near locations 16–31. **X-Plane emits no native motion/velocity
  output of any kind.**
- **`resolve` (9 permutations)** — the fragment variants read a single texture
  `u_tex` + `v_texcoord0`. No history sampler, no velocity sampler, no depth
  sampler. X-Plane's "resolve" is a plain copy/MSAA-resolve, **not** a temporal
  pass. There is no native TAA to cooperate or conflict with.
- The world/scene render pass the layer tags `colour=5` therefore has **5 colour
  attachments while the shader writing into it declares only one output
  (attachment 0)**. Attachments 1–4 are not written by `deferred_gbuf`.

---

## Findings

### 1. The FF777 "glass stamps velocity" model is empirically disproven — CONFIRMED
This session we forced **every** see-through cockpit surface to write zero
velocity (`TAA_MASK_SEETHROUGH`, all 259 candidates masked) and the buildings
seen through the windshield **still shimmered**. The mod's entire
`TAA_MASK_FRAG` / `TAA_MASK_SEETHROUGH` apparatus (`layer.cpp` ~10099–10113,
the whole "canopy stamps its velocity on the world behind it" comment block) is
built around a cause that a direct test contradicts. **The real shimmer cause is
still unidentified.** Everything the code currently says about it is a guess that
testing has now falsified.

### 2. The transparency/coverage channel only exists for *forward* alpha-blended draws — CONFIRMED
- The injected fragment writes `.a` (the coverage channel the resolve keys its
  reactive mask on) as a **constant 1.0** for normal geometry
  (`spirv_inject.h:1723`, `idCh3 = … idConstOneF`).
- The real coverage value (colour-alpha thresholded at 0.5) is only substituted
  **when `alphaBlended` is true** (`spirv_inject.h:1786`), and `alphaBlended`
  is read from `pColorBlendState->pAttachments[0].blendEnable == VK_TRUE`
  (`layer.cpp` ~10614).
- The resolve then does `if (FLAG_REACTIVE && !unwritten && covMax < 0.5) a=1.0`
  (`shaders/taa.comp:966`).

**Consequence:** a surface that is visually transparent but drawn with
`blendEnable = FALSE` (alpha-test / dithered / deferred glass) is classified
fully opaque, coverage 1.0, and can **never** be handed to the current frame by
the reactive mask. Its velocity fully stamps the world behind it and the world
is never treated as disoccluded. This is exactly the class the FF777 canopy
would fall in — **SUSPECTED** to be central to the shimmer, and it is invisible
to every masking knob the mod currently exposes.

### 3. The coverage/opacity read is a forward-rendering assumption on a deferred G-buffer — CONFIRMED (documented, still latent)
`spirv_inject.h:1773–1788` documents it directly: reading colour-attachment-0's
`.a` as opacity is a forward assumption, and X-Plane 12 is deferred — attachment
0 is packed G-buffer data, **not** a colour with opacity in alpha. The mod's
mitigation is to trust `blendEnable` instead of the channel. That is correct
*only as far as X-Plane's `blendEnable` correctly tracks visual transparency* —
and per finding #2 it does not for deferred/alpha-test glass. The safeguard has
a hole exactly where the FF777 lives.

### 4. Silent zero-velocity failure "looks like working plumbing" — CONFIRMED
When a fragment can't be patched (`INJ_LOCATION_TAKEN`, or "malformed"), the
both-or-neither rule discards the vertex patch too. The code's own comment
(`layer.cpp` ~10632) says the result is a field of zeros that "looks like
working plumbing." **There is no runtime signal that a chunk of geometry silently
lost its vectors** — it just quietly stops anti-aliasing there. For an aircraft
that trips many refusals, large parts of the frame can be running with no
velocity and nothing reports it.

### 5. The varying-location census is calibrated against a false premise — CONFIRMED premise, SUSPECTED impact
The census picks the top free adjacent pair (28/29) on the theory that X-Plane
uses varyings up through ~16. Ground truth: **X-Plane's G-buffer uses exactly one
varying, Location 5.** Locations 6–31 are all free. So:
- 28/29 is safe against X-Plane's own shaders — the "FF777 squats on 28/29"
  collision story cannot come from `deferred_gbuf`.
- Any real `LOCATION_TAKEN` therefore comes from **non-gbuffer** shaders (SASL
  avionics `displays.*` / `wxr.*`, or other archives), which the single global
  census cannot accommodate. The mechanism is real but the diagnosis of *which*
  shaders and *why* was aimed at the wrong shader set.

### 6. Pass identity rests on attachment *count*, which doesn't match the shader — OBSERVATION → verify
The layer treats the 5-colour-attachment pass as "the world" and appends
velocity as attachment index 5. But `deferred_gbuf` writes only attachment 0;
attachments 1–4 aren't written by the geometry shader at all. The
count-based identity (`colour=5` world vs `colour=1` overlay) is a heuristic on a
structure that doesn't behave like a classic MRT G-buffer. Worth confirming the
scene-pass classifier isn't tagging the wrong pass on aircraft that build passes
differently (the FF777 draws far more `colour=1` see-through passes than a
default aircraft — 259 distinct — which is itself a sign its cockpit pass
structure is unusual).

### 7. The mod is not robust to MSAA — it crashes instead of declining — CONFIRMED
With MSAA on, the resolve can't consume a multisampled target, and on the FF777
it reaches a failed `vkCreateImageView` → null descriptor → compute crash
(per the standing memory note). The mod *overrides* the user's AA anyway
(`plugin.cpp` `g_postAA` forces MSAA/FXAA/SSR off while it runs). The just-added
777 bypass is a **workaround** (stand fully down on `777*.acf`), not a fix — the
failed image-view is still un-named. The memory note is right: this needs a
GPU-AV capture to name the descriptor, not more inference.

### 8. Meta-finding: the mod reasons about X-Plane by inference where it should be capturing — the throughline
Findings #1, #3, #7 are all the same shape: a plausible model of what X-Plane is
doing, encoded in the shader/layer, that a direct measurement then contradicts.
The velocity-stamp model, the alpha-as-opacity model, and the MSAA path were all
reasoned, not observed. The single highest-value next step for *any* of the
remaining FF777 problems is a **RenderDoc / GPU-AV capture of one frame in the
777 cockpit** — what actually writes the velocity texel under a shimmering
building, and what the failed image view is — rather than another inference layer.

---

## Not yet deeply audited (honest coverage note)
- The reprojection math in `plugin.cpp` (the camera-relative `prevViewProj` /
  `bodyReproj` derivation, the 52 km float32 precision note).
- The neighbourhood clamp / variance / gain path in `taa.comp` beyond the
  reactive mask.
- The FSR/FSR3/frame-gen backends and their interaction with the velocity target.
- The VRAM pager / art-control holds in `plugin.cpp`.
Say the word and I'll take any of these to the same depth.
