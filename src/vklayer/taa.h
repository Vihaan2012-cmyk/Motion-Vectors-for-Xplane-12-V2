// MotionVectors for X-Plane 12 - temporal anti-aliasing from injected motion
// vectors, plus a VRAM manager.
//
// Copyright (C) 2026 Vihaan2012
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
// more details.
//
// You should have received a copy of the GNU General Public License along
// with this program. If not, see <https://www.gnu.org/licenses/>.

#pragma once

// TEMPORAL ANTIALIASING - THE VULKAN SIDE.
//
// Built to be falsifiable in stages. The predecessor was removed because a
// corrupt frame could not be attributed: plumbing and algorithm failed together
// and neither could be cleared. Here TAA_MODE selects how far the shader runs -
// 0 passthrough, 1 reproject, 2 full - so each stage is a separate experiment
// against the same wiring:
//
//   mode 0  every binding, barrier and dispatch exercised, output = input.
//           Any visible change is a plumbing fault, full stop.
//   mode 1  history fetched and blended. Ghosting along motion is EXPECTED and
//           means reprojection works.
//   mode 2  neighbourhood clamp on top. Ghosting should collapse.
//
// The history image is ours: created here, owned here, and explicitly cleared
// on first use. Uninitialised history is the likeliest source of the magenta
// frame the old resolve produced, and a clear costs one command.

#include <stdint.h>
#include <string.h>
#include <map>
#include "taa_spv.h"
#include "temporal.h"

struct TaaState {
    VkDevice        device      = VK_NULL_HANDLE;
    // ---- PING-PONG. ONE HISTORY IMAGE IS A DATA RACE.
    //
    // The first version bound a single history image as a storage image
    // (written) and a sampler (read) in the SAME dispatch. Workgroups read
    // while others wrote, so every frame sampled partially-written data and the
    // error compounded - the picture filled with black over about five seconds
    // of camera movement, slow enough to look like a blend-weight problem and
    // fast enough to destroy the image.
    //
    // Read last frame's, write this frame's, swap. This is not an optimisation;
    // a single buffer has no defined contents to read.
    VkImage         history[2]     = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDeviceMemory  historyMem[2]  = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkImageView     historyView[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    uint32_t        historyWrite   = 0;   // index written this frame
    // ---- THE SHARPEN TARGET. A THIRD BUFFER THAT NEVER FEEDS BACK.
    //
    // MODE_SHARPEN reads the resolved history and writes here; the copy-to-screen
    // then reads HERE instead of the history. Because nothing ever samples this
    // as history, the sharpen cannot compound into the accumulation - which is
    // the whole reason the sharpen is a separate buffer and not an in-resolve
    // step. Optional: if creation fails the resolve runs exactly as before and
    // the sharpen is simply unavailable.
    VkImage         sharpImage  = VK_NULL_HANDLE;
    VkDeviceMemory  sharpMem    = VK_NULL_HANDLE;
    VkImageView     sharpView   = VK_NULL_HANDLE;
    // Tracked so the pre-dispatch barrier names the right source: UNDEFINED on
    // first use, GENERAL every frame after (the copy-back returns it to GENERAL).
    VkImageLayout   sharpLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // ---- ONE VIEW PER SCENE IMAGE, CACHED, NEVER TORN DOWN MID-SESSION.
    //
    // X-Plane alternates between two HDR scene targets. Keying re-initialisation
    // on the image handle therefore rebuilt the pipeline, pool and views EVERY
    // frame, destroying objects that in-flight command buffers still referenced
    // - which crashed the sim, and showed up as a dispatch counter that never
    // got past 1 because each rebuild reset it.
    //
    // The history is ours and is unaffected by which target the frame drew
    // into, so only the scene view has to follow. Cached here and destroyed
    // with everything else.
    std::map<VkImage, VkImageView> sceneViews;
    VkImageView     sceneView   = VK_NULL_HANDLE;
    VkImage         sceneImage  = VK_NULL_HANDLE;   // the view above belongs to this
    VkImageView     velView     = VK_NULL_HANDLE;
    // Generation of the velocity target the descriptor was written against.
    // Handles are reused; this is not. See MvTarget::gen.
    uint64_t        velGen      = 0;
    VkSampler       sampler     = VK_NULL_HANDLE;
    // NEAREST, for the integer flags image - see the note at its creation.
    VkSampler       samplerNearest = VK_NULL_HANDLE;
    // Comparison samplers for the engine shadow cascades: the engine samples
    // tex_smap0 through sampler2DArrayShadow, so the compare runs in hardware
    // and the only open question is the compare OP - both are created and the
    // live knob picks per frame, because one screenshot settles it faster
    // than one relaunch per guess.
    VkSampler       samplerShadowLE = VK_NULL_HANDLE;
    VkSampler       samplerShadowGE = VK_NULL_HANDLE;
    // ---- X-PLANE'S gbuffer_vel, AND A FALLBACK FOR WHEN IT IS UNKNOWN.
    //
    // The flags view is over an image X-Plane owns, identified by shape by the
    // layer's census; views over it are cached like the scene views. The
    // fallback is a 1x1 zero uint image of our own: Vulkan requires every
    // statically-used binding to be valid even behind a branch, and a zero
    // flag word reads as bit-2-clear, which makes the fallback a no-op.
    std::map<VkImage, VkImageView> flagsViews;
    // Binding 5: the engine's own depth. Views cached per image exactly like
    // the flags views; edValid says binding 5 holds the real thing.
    std::map<VkImage, VkImageView> edViews;
    VkImageView     edView  = VK_NULL_HANDLE;
    bool            edValid = false;
    // Bindings 7/8: the engine's sun cascades and environment probes.
    std::map<VkImage, VkImageView> sunViews, probeViews, normViews;
    VkImageView     normView  = VK_NULL_HANDLE;
    bool            normValid = false;
    // 1x1 D16 dummy for binding 7's fallback. A COMPARISON sampler on the
    // RGBA16F velocity view was invalid Vulkan for every frame the cascades
    // were unidentified - i.e. all menus. Comparison sampling demands a depth
    // format, so the dummy is one.
    VkImage         sunDummy     = VK_NULL_HANDLE;
    VkDeviceMemory  sunDummyMem  = VK_NULL_HANDLE;
    VkImageView     sunDummyView = VK_NULL_HANDLE;
    bool            sunDummyReady = false;   // one-time layout transition done
    VkImageView     sunView   = VK_NULL_HANDLE;
    bool            sunValid  = false;
    VkImageView     probeView = VK_NULL_HANDLE;
    bool            probeValid = false;
    VkImageView     flagsView   = VK_NULL_HANDLE;   // what binding 4 gets
    bool            flagsValid  = false;            // true only for the real image
    VkImage         flagsFallback     = VK_NULL_HANDLE;
    VkDeviceMemory  flagsFallbackMem  = VK_NULL_HANDLE;
    VkImageView     flagsFallbackView = VK_NULL_HANDLE;

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout      pipeLayout = VK_NULL_HANDLE;
    VkPipeline            pipeline   = VK_NULL_HANDLE;
    VkDescriptorPool      pool       = VK_NULL_HANDLE;

    // A ring, not one set. The predecessor exhausted a single-set pool after
    // eight frames and stopped silently - the resolve simply stopped running
    // and nothing said so.
    // ---- 16, NOT 8: THE SHARPEN PASS TAKES A SECOND SET PER FRAME.
    //
    // With MODE_SHARPEN on, a resolve frame consumes two sets from this ring
    // (resolve + sharpen) instead of one. Eight left barely two frames of
    // in-flight headroom before a set still referenced by a submitted command
    // buffer could be overwritten; doubling the ring restores the margin.
    static const uint32_t kSets = 16;
    VkDescriptorSet sets[kSets] = { VK_NULL_HANDLE };
    uint32_t        nextSet = 0;
    // Per-dispatch uniform data (the reprojection matrix for depth-
    // reconstructed velocity). A ring of kSets slots aligned to 256 so the
    // slot written for this dispatch is never one the GPU may still be
    // reading - the same rotation discipline as the descriptor sets.
    VkBuffer        uboBuf = VK_NULL_HANDLE;
    VkDeviceMemory  uboMem = VK_NULL_HANDLE;
    void           *uboMap = nullptr;

    uint32_t w = 0, h = 0;
    // Array layers of the scene target, and therefore of our history. One for a
    // plain target, two for stereo. Every view is created as 2D_ARRAY over this
    // many layers, so the single-layer case needs no separate path.
    uint32_t layers = 1;
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool     ready = false;
    bool     historyCleared = false;
    uint64_t dispatches = 0;

    // ---- DIRECT HISTORY READBACK.
    //
    // Every claim about whether history accumulates has so far been inferred
    // from the composited screen, and viz=4 cannot settle it because the
    // visualisation is written INTO the history image - the measurement
    // contaminates the thing measured. This copies a strip of the history
    // image itself into host memory, so the bytes can be compared across
    // frames without passing through the resolve's own output path.
    //
    // A strip, not the frame: 512 texels of RGBA16F is 4 KB, enough for a
    // stable statistic and small enough that the copy cannot meaningfully
    // perturb the timing it is measuring.
    VkBuffer        readBuf  = VK_NULL_HANDLE;
    VkDeviceMemory  readMem  = VK_NULL_HANDLE;
    void           *readPtr  = nullptr;
    uint64_t        readFrame = 0;
};

static TaaState g_taa;

// Armed once per PRESENT by the present hook, consumed by the first resolve of
// that frame. The history ping-pong has to follow displayed frames, not the
// number of times the resolve happens to record - see the note at the flip.
static bool g_taaFlipArmed = true;

struct TaaPush {
    float invSizeX, invSizeY;
    float jitterX, jitterY;
    float alpha;
    int32_t mode;
    // ---- reset and cameraMoved live in `flags` now (kTaaFlagReset,
    // kTaaFlagCameraMoved). The block is at the 128-byte ceiling and each of
    // those int32s carried one bit; their bytes now hold the engine-depth
    // linearization:  view = -edB / (ndc_depth + edA), edA = proj[10],
    // edB = proj[14] from the live projection. See g_taaEdAB.
    float   edA;
    float   edB;
    // Debug visualisation, and the switches that remove one input at a time.
    // Packed as a bitmask rather than four ints because the block is pushed
    // every frame and 16 bytes of padding is 16 bytes of nothing.
    int32_t viz;
    float   vizScale;
    float   gain;
    float   varClip;
    int32_t flags;      // see kTaaFlag* below
    // Live A/B knobs for the reprojection convention - see the shader's note.
    float   velScale;
    float   velYSign;
    // 1 when binding 4 is X-Plane's real gbuffer_vel rather than the fallback.
    int32_t flagsValid;
    // ---- THE UNJITTER SHIFT, AS TWO NUMBERS RATHER THAN A CONVICTION.
    //
    // S = (sMulX * jitter.x, sMulY * jitter.y). The hardcoded (-0.5, +0.5)
    // encodes two conventions at once - NDC-to-UV is a half, and the viewport
    // height is negative so Y flips - and getting either wrong turns the
    // cancellation into a doubling. That is indistinguishable from "no
    // cancellation" by eye, and it scales with jitter amplitude, which is
    // exactly the symptom: shake at jitter_scale=1, none at 0. Both have been
    // wrong in this file before, so they are swept, not argued.
    //
    // MEASURED, parked aircraft, still camera, temporal mean absolute
    // deviation over 8 frames of the same ground:
    //     jitter off .................. 0.22   (the floor)
    //     (+0.5, -0.5) ................ 2.94   <- default
    //     no cancellation at all ...... 4.34
    //     (-0.5, +0.5), as shipped .... 5.93
    // The shipped pair was WORSE than doing nothing: it added the
    // displacement instead of removing it.
    float   sMulX;
    float   sMulY;
    // Largest |velocity|, in UV, that is believed. Live so the magnitude of
    // whatever is poisoning history can be found by bisection rather than
    // guessed: accumulation returns at whatever threshold excludes it.
    float   velMax;
    // Coverage below this counts as "nobody wrote here". Negative disables the
    // unwritten-pixel rejection entirely, which is how its cost is measured.
    float   novecCov;
    // Blend weight for a pixel that never received a vector. 1.0 is the old
    // hard rejection (shake); pc.alpha is keeping it forever (crawl); between
    // the two the stale history decays instead of being kept or thrown away.
    float   novecAlpha;
    float   movedDead;
    float   alphaMoving;
    float   alphaMovingPx;
    // Strength of the post-resolve MODE_SHARPEN pass. 0 disables it and the
    // record path skips the second dispatch entirely.
    float   sharpen;
    // Weight forced on transparent-covered pixels. 1.0 = the original
    // all-or-nothing mask; lower lets them still accumulate. See taa.comp.
    float   reactiveAlpha;
    // ---- AMBIENT OCCLUSION, APPENDED AFTER sharpen.
    //
    // The comment above says sharpen must be LAST so the block mirrors the
    // shader byte for byte. That constraint is about the two structs AGREEING,
    // not about which field ends them - so these go after it, and the shader's
    // Params block gets the same two in the same order.
    //
    // aoStrength 0 disables the whole thing at zero cost: the shader branches
    // out before taking a single tap.
    float   aoStrength;   // 0 = off, ~0.5 typical
    float   aoRadius;     // sampling radius in PIXELS at the render resolution

    // ---- CONTACT SHADOWS. The last four floats that fit.
    //
    // sunView is a unit vector toward the sun in view space, computed by the
    // plugin because that is where the camera matrices live. All zero means the
    // sun is below the horizon and the shader skips the march entirely.
    //
    // The direction arrives ready to use rather than being derived here from
    // pitch and heading, because deriving it would need the projection matrix
    // as well and there is no room: see the assert below.
    float   sunViewX, sunViewY, sunViewZ;
    float   csStrength;   // 0 = off, ~0.6 typical
};

enum {
    kTaaFlagFreezeHistory = 1 << 0,
    kTaaFlagNoMotion      = 1 << 1,
    kTaaFlagNoAccum       = 1 << 2,
    kTaaFlagReactive      = 1 << 3,
    kTaaFlagNoUnjitter    = 1 << 4,
    kTaaFlagCatmull       = 1 << 5,
    kTaaFlagNoVecByVel    = 1 << 6,
    kTaaFlagCrUnjitter    = 1 << 7,
    // Velocity dilation: take the closest neighbour's vector at silhouettes.
    // Default ON - a correctness fix for thin geometry, not an effect. Bit 8
    // because 7 was already taken on the frame-gen branch.
    kTaaFlagDilate        = 1 << 8,
    kTaaFlagReset         = 1 << 9,   // was the int32 TaaPush::reset
    kTaaFlagCameraMoved   = 1 << 10,  // was the int32 TaaPush::cameraMoved
    kTaaFlagEngineDepth   = 1 << 11,  // binding 5 holds X-Plane's gbuf-depth
    kTaaFlagDepthReproject = 1 << 12, // reconstruct velocity at unwritten px
    kTaaFlagSunTap        = 1 << 13,  // bindings 7+9 carry live cascade data
    kTaaFlagProbeTap      = 1 << 14,  // binding 8 carries the env cubemaps
    kTaaFlagGbufTap       = 1 << 15,  // binding 10 carries u_gbuffer_data
    kTaaFlagGi            = 1 << 16,  // binding 11 carries gathered bounce
    kTaaFlagNormalTap     = 1 << 17,  // binding 12 carries gbuf-normal
};

// ---- EVERY KNOB IS LIVE. NONE OF THESE ARE CACHED.
//
// They used to be `static const` initialised from getenv, which is why changing
// the alpha meant restarting the sim. Reading them per frame costs a map lookup
// against a table that is only rebuilt when the control file's timestamp moves,
// and buys the ability to answer a question in the ten seconds it takes to save
// a file instead of the four minutes it takes to relaunch.
static bool  taaEnabled()  { return live::onoff("taa.enable", "TAA_RESOLVE", false); }
// 2 (MODE_FULL), not 0 (MODE_PASSTHROUGH). The mode defaulted to a no-op
// INDEPENDENTLY of taa.enable, so "TAA on, mode unset" ran the whole pipeline -
// velocity, jitter, history, dispatch - and then copied the frame through
// untouched. Nothing reports that: the panel says On, the duty counter says
// 100%, and the picture is stock.
//
// It only ever bit when the two came apart, but the shipping ini is exactly
// where they come apart: %TEMP%\taa_live.ini does not exist on a new install
// (see the varclip note below), so on a fresh machine the compiled default IS
// the configuration, and a user who enables TAA gets passthrough. A default
// that does nothing is the wrong answer to "is it on?".
static int   taaMode()     { return live::i("taa.mode",  "TAA_MODE",  2); }
static float taaAlpha()    { return live::f("taa.alpha", "TAA_ALPHA", 0.05f); }
static float taaGain()     { return live::f("taa.gain",  "TAA_GAIN",  4.0f); }
// 8.0, not 1.25. A tight clamp rejects history wherever it differs from the
// current 3x3, and jitter guarantees it differs on a thin edge - so flap track
// fairings, gear struts and pylon edges took the raw frame every frame, never
// accumulated, and shimmered. Reported in flight and confirmed fixed at 8.0.
// This is a DEFAULT, not just a live value: %TEMP%	aa_live.ini does not exist
// on a new install, so whatever is compiled here IS the shipping configuration.
static float taaVarClip()  { return live::f("taa.varclip", "TAA_VARCLIP", 8.0f); }
// Deadband on the clamp correction, in units of the noise floor. 1.0 makes a
// correction at or below the floor read as zero, which is what the shader's
// floorS note asks for; 0.0 restores the old behaviour that pinned a at 1.0.
static float taaMovedDead(){ return live::f("taa.moved_dead", "TAA_MOVED_DEAD", 0.0f); }
// Sharp history resampling. Bilinear history fetch is the motion-only,
// surface-locked blur; see the note at sampleHistory in the shader.
static bool taaCatmull()   { return live::onoff("taa.hist_catmull", "TAA_HIST_CATMULL", true); }
// Blend weight while the camera moves. Long history is what anti-aliases a
// parked frame; it is also what lets a small reprojection error compound into
// a trail once things move. See the note in the shader.
static float taaAlphaMoving(){ return live::f("taa.alpha_moving", "TAA_ALPHA_MOVING", 0.35f); }
// Speed, in px/frame, at which alpha_moving is fully applied. The ramp runs
// from 0 to this, so a parked airframe's tremble reads as stationary.
static float taaAlphaMovingPx(){ return live::f("taa.alpha_moving_px", "TAA_ALPHA_MOVING_PX", 3.0f); }
static int   taaViz()      { return live::i("taa.viz",   "TAA_VIZ",   0); }
static float taaVizScale() { return live::f("taa.viz_scale", nullptr, 1.0f); }
// ---- POST-RESOLVE SHARPEN. THE ANSWER TO "TAA IS TOO SOFT".
//
// TAA resamples history every frame, so a low alpha (long, well-antialiased
// history) is also a heavily blurred one - fine detail is averaged out. Raising
// alpha trades that blur straight back for aliasing; a sharpen pass recovers the
// detail WITHOUT shortening the history, which is why every shipping TAA pairs
// the two. Runs as MODE_SHARPEN on a dedicated buffer so it never feeds back.
// 0 = off (the record path then skips the second dispatch and copies the
// resolved image straight to screen, i.e. exactly the pre-sharpen behaviour).

// ---- REMOVE ONE INPUT AT A TIME.
//
// The generalisation of MODE_PASSTHROUGH to every stage, and the reason it is
// worth having as separate switches rather than one debug mode: each one makes a
// DIFFERENT prediction, so the observation attributes the fault rather than
// merely changing the picture.
//
//   freeze_history   the image should freeze and smear along motion. If it does
//                    not, what is on screen is not the history.
//   no_motion        every vector reads zero, so reprojection is a same-pixel
//                    fetch. Ghosting that SURVIVES this is not the vectors.
//   no_accum         current frame out, every binding and barrier still live.
//   force_reset      same output as no_accum but reached through the reset path,
//                    so the pair separates "reset is broken" from "accumulation
//                    is broken" - which looked identical for two builds.
static float taaVelScale() { return live::f("taa.vel_scale", nullptr, 1.0f); }
// -1.0 is the shipping belief (negative-height viewport => d(uv_y) = -vel_y).
static float taaVelYSign() { return live::onoff("taa.vel_ypos", nullptr, false) ? 1.0f : -1.0f; }
// The gbuffer_vel weight override - on by default because everything it
// addresses (prop halo, airframe streaks, cockpit shake) is worse than the
// aliasing it reintroduces on the airframe. taa.objflags=0 turns it off live
// so its exact visual contribution can be isolated in one edit.
static bool taaObjFlags() { return live::onoff("taa.objflags", nullptr, true); }
// The C14 reactive mask - on by default for the same reason as the flag
// override: flicker parked in history is worse than aliasing on the flickering
// content. taa.reactive=0 isolates its contribution live.
// ---- DEFAULT OFF, AND HONESTLY UNDIAGNOSED.
//
// The reactive mask forces the blend weight to 1.0 where coverage reads below
// a half, so those pixels take the raw current frame every frame and never
// accumulate. It exists for the propeller disc, which flickers by
// construction. Measured parked, five interleaved rounds, scene verified
// stable, same view:
//
//     reactive ON   4.34x the temporal flicker of TAA-off
//     reactive OFF  1.65x
//     detail        identical either way (93%)
//
// So it costs well over half the temporal stability and buys nothing
// measurable. Four separate faults in the coverage path were found and fixed -
// a zero default where the fragment had no Location 0 output, attachment-0
// alpha read as opacity in a DEFERRED g-buffer, a variant cache that did not
// distinguish opaque from alpha-blended, and cleared pixels read as
// transparent - and none of them moved the number: 4.52 -> 4.34 -> 4.42.
//
// Why coverage still reads low across large regions is NOT understood. What is
// measured is that the mask is not earning its cost, so it is off by default
// and the knob remains for propeller aircraft, where the artefact it targets
// is real and this test aircraft has none.
static bool taaReactive() { return live::onoff("taa.reactive", "TAA_REACTIVE", false); }
// Velocity dilation. ON by default: a silhouette pixel that takes the texel
// centre's vector is reprojected as the BACKGROUND it partially covers, which
// is why thin geometry - struts, antennas, wires, blade edges - ghosts. Set 0
// to A/B it; the difference shows on edges under camera motion, nowhere else.
static bool taaDilate() { return live::onoff("taa.dilate", "TAA_DILATE", true); }
// Read AO/contact view depth from X-Plane's own gbuf-depth (binding 5) instead
// of the injected clip-w. Off by default until it has been A/B'd - and it can
// only engage once the name listener has identified the image anyway.
static bool taaPosHarvest() {
    // taa.engine_depth is the honest name (the harvest source is the engine's
    // gbuf-depth; "pos" predates learning that gbuffer_pos IS depth). The old
    // key stays as an alias so existing inis keep working.
    return live::onoff("taa.engine_depth", "TAA_ENGINE_DEPTH", false) ||
           live::onoff("taa.pos_harvest",  "TAA_POS_HARVEST",  false);
}
// Depth-reconstructed velocity for pixels the injection never wrote (sky,
// clouds, any unpatched shader). Needs pos_harvest's engine depth AND a valid
// reprojection matrix; degrades to the old vel=0 per-pixel otherwise. The
// reconstruction uses the SAME matrix and the SAME (curr - prev) * 0.5 formula
// as the injected vertex shaders, so both vector populations agree in units,
// sign and convention by construction.
static bool taaNovecReproject() { return live::onoff("taa.novec_reproject", "TAA_NOVEC_REPROJECT", true); }
// ---- AMBIENT OCCLUSION.
//
// Computed inside the resolve from the clip w the injector writes into the
// velocity target, so it costs eight texture taps and nothing else - no image,
// no pass, no VRAM. X-Plane's own SSAO is coarse and spatial; this one is
// accumulated by the history blend, which is where the quality comes from.
//
// 0 disables it before a single tap is taken. Default 0.5 - visible in cockpit
// creases without looking like a filter.
static float taaAoStrength() { return live::f("taa.ao",        "TAA_AO",        0.5f); }
// ---- DEFAULT OFF, DELIBERATELY.
//
// This shipped at 0.6 and therefore turned itself on for everyone the moment
// the build landed, before it had been looked at once. Whatever its final
// quality, a new screen-space effect that alters every frame is opt-in until it
// has been A/B'd - the panel has a CONTACT button for exactly that.
static float taaCsStrength() { return live::f("taa.contact",   "TAA_CONTACT",   0.0f); }
// Radius in PIXELS at render resolution. Small on purpose: this is contact
// darkening at the scale of a switch base, not a large-scale ambient term.
static float taaAoRadius()   { return live::f("taa.ao_radius", "TAA_AO_RADIUS", 12.0f); }
// The unjitter alignment - isolation knob for the aligned sampling, so its
// contribution can be removed live without touching the jitter itself.
static bool taaUnjitter() { return live::onoff("taa.unjitter", nullptr, true); }
// ---- THE TWO KNOBS THAT ANSWER "TAA ON IS SOFTER THAN TAA OFF".
//
// taa.cr_unjitter resamples the current frame's unjitter fetch with
// Catmull-Rom instead of bilinear. This is not a preference: the bilinear
// fetch at uv + S is a low-pass filter that runs on EVERY frame, parked
// included, and history accumulates its output - so it is baked into the
// converged image. Default ON for the same reason taa.hist_catmull is.
//
// taa.sharpen puts back what the resample cannot recover. Default 0.35: enough
// to read as sharper than TAA-off on text and panel edges, well below where
// the limiter in sharpenCurrent starts clipping on ordinary content. 0 turns
// the pass off entirely rather than sharpening by zero.
static bool  taaCrUnjitter() { return live::onoff("taa.cr_unjitter", nullptr, true); }
static float taaSharpen()    { return live::f("taa.sharpen", "TAA_SHARPEN", 0.35f); }
static float taaReactiveAlpha() { return live::f("taa.reactive_alpha", nullptr, 1.0f); }
static bool taaFreezeHistory() { return live::onoff("taa.freeze_history", nullptr, false); }
static bool taaNoMotion()      { return live::onoff("taa.no_motion",      nullptr, false); }
static bool taaNoAccum()       { return live::onoff("taa.no_accum",       nullptr, false); }
static bool taaForceReset()    { return live::onoff("taa.force_reset",    nullptr, false); }

static uint32_t taaFindMemory(DeviceData &dd, uint32_t typeBits, VkMemoryPropertyFlags want)
{
    VkPhysicalDeviceMemoryProperties mp;
    memset(&mp, 0, sizeof(mp));
    if (!g_getPhysMemProps || dd.phys == VK_NULL_HANDLE) return UINT32_MAX;
    g_getPhysMemProps(dd.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

// Destroy a parked state's objects for real. Only ever called on states that
// left service N presents ago - nothing in flight can still reference them.
static void taaDestroyState(DeviceData &dd, TaaState &g_taa)
{
    if (g_taa.pipeline)    dd.destroyPipeline(g_taa.device, g_taa.pipeline, nullptr);
    if (g_taa.pipeLayout)  dd.destroyPipelineLayout(g_taa.device, g_taa.pipeLayout, nullptr);
    if (g_taa.setLayout)   dd.destroyDescriptorSetLayout(g_taa.device, g_taa.setLayout, nullptr);
    if (g_taa.pool)        dd.destroyDescriptorPool(g_taa.device, g_taa.pool, nullptr);
    if (g_taa.uboBuf) {
        PFN_vkDestroyBuffer pfnDb =
            (PFN_vkDestroyBuffer)dd.gdpa(g_taa.device, "vkDestroyBuffer");
        if (pfnDb) pfnDb(g_taa.device, g_taa.uboBuf, nullptr);
    }
    if (g_taa.uboMem)      dd.freeMemory(g_taa.device, g_taa.uboMem, nullptr);
    if (g_taa.sampler)     dd.destroySampler(g_taa.device, g_taa.sampler, nullptr);
    if (g_taa.samplerNearest) dd.destroySampler(g_taa.device, g_taa.samplerNearest, nullptr);
    if (g_taa.samplerShadowLE) dd.destroySampler(g_taa.device, g_taa.samplerShadowLE, nullptr);
    if (g_taa.samplerShadowGE) dd.destroySampler(g_taa.device, g_taa.samplerShadowGE, nullptr);
    for (int i = 0; i < 2; ++i)
        if (g_taa.historyView[i]) dd.destroyImageView(g_taa.device, g_taa.historyView[i], nullptr);
    for (std::map<VkImage, VkImageView>::iterator it = g_taa.sceneViews.begin();
         it != g_taa.sceneViews.end(); ++it)
        if (it->second) dd.destroyImageView(g_taa.device, it->second, nullptr);
    g_taa.sceneViews.clear();
    // The engine-image view caches added for the taps: leaked on every
    // re-init (resolution change) until this loop existed.
    std::map<VkImage, VkImageView> *caches[5] = {
        &g_taa.flagsViews, &g_taa.edViews, &g_taa.sunViews, &g_taa.probeViews,
        &g_taa.normViews };
    for (int ci = 0; ci < 5; ++ci) {
        for (std::map<VkImage, VkImageView>::iterator it = caches[ci]->begin();
             it != caches[ci]->end(); ++it)
            if (it->second) dd.destroyImageView(g_taa.device, it->second, nullptr);
        caches[ci]->clear();
    }
    g_taa.flagsView = VK_NULL_HANDLE; g_taa.flagsValid = false;
    g_taa.edView    = VK_NULL_HANDLE; g_taa.edValid    = false;
    g_taa.sunView   = VK_NULL_HANDLE; g_taa.sunValid   = false;
    g_taa.probeView = VK_NULL_HANDLE; g_taa.probeValid = false;
    g_taa.normView  = VK_NULL_HANDLE; g_taa.normValid  = false;
    if (g_taa.sunDummyView) dd.destroyImageView(g_taa.device, g_taa.sunDummyView, nullptr);
    if (g_taa.sunDummy)     dd.destroyImage(g_taa.device, g_taa.sunDummy, nullptr);
    if (g_taa.sunDummyMem)  dd.freeMemory(g_taa.device, g_taa.sunDummyMem, nullptr);
    g_taa.sunDummyView = VK_NULL_HANDLE;
    g_taa.sunDummy = VK_NULL_HANDLE;
    g_taa.sunDummyMem = VK_NULL_HANDLE;
    g_taa.sunDummyReady = false;
    // velView is g_mv.viewArray - g_mv owns and destroys it; destroying it
    // here too was a latent double-destroy.
    for (std::map<VkImage, VkImageView>::iterator it = g_taa.flagsViews.begin();
         it != g_taa.flagsViews.end(); ++it)
        if (it->second) dd.destroyImageView(g_taa.device, it->second, nullptr);
    g_taa.flagsViews.clear();
    if (g_taa.flagsFallbackView) dd.destroyImageView(g_taa.device, g_taa.flagsFallbackView, nullptr);
    if (g_taa.flagsFallback)     dd.destroyImage(g_taa.device, g_taa.flagsFallback, nullptr);
    if (g_taa.flagsFallbackMem)  dd.freeMemory(g_taa.device, g_taa.flagsFallbackMem, nullptr);
    for (int i = 0; i < 2; ++i) {
        if (g_taa.history[i])    dd.destroyImage(g_taa.device, g_taa.history[i], nullptr);
        if (g_taa.historyMem[i]) dd.freeMemory(g_taa.device, g_taa.historyMem[i], nullptr);
    }
    if (g_taa.sharpView)  dd.destroyImageView(g_taa.device, g_taa.sharpView, nullptr);
    if (g_taa.sharpImage) dd.destroyImage(g_taa.device, g_taa.sharpImage, nullptr);
    if (g_taa.sharpMem)   dd.freeMemory(g_taa.device, g_taa.sharpMem, nullptr);
    TaaState fresh;
    fresh.device = g_taa.device;
    g_taa = fresh;
}

// Teardown = park, not destroy. Resolves recorded 1-2 frames ago still
// reference these objects; destroying them under the GPU was the teardown
// DEVICE_LOST, and deviceWaitIdle here races X-Plane's submit threads.
// The graveyard holds each retired state until 8 presents have passed.
struct TaaGrave { TaaState s; uint64_t frame; };
static std::vector<TaaGrave> g_taaGraves;
static uint64_t g_taaGraveNow = 0;

static void taaDestroy(DeviceData &dd)
{
    (void)dd;
    if (g_taa.ready || g_taa.pool) {
        TaaGrave gr; gr.s = g_taa; gr.frame = g_taaGraveNow;
        g_taaGraves.push_back(gr);
    }
    TaaState fresh;
    fresh.device = g_taa.device;
    g_taa = fresh;
}

// Called once per present with the current frame counter.
static void taaGraveFlush(DeviceData &dd, uint64_t frame)
{
    g_taaGraveNow = frame;
    for (size_t i = 0; i < g_taaGraves.size();) {
        if (frame > g_taaGraves[i].frame + 8) {
            taaDestroyState(dd, g_taaGraves[i].s);
            g_taaGraves.erase(g_taaGraves.begin() + (long)i);
        } else ++i;
    }
}

// Build everything that depends on the scene target's size and format. Called
// again whenever either changes, which is why teardown comes first.
static bool taaInit(DeviceData &dd, VkDevice dev, VkImage scene, VkFormat fmt,
                    uint32_t w, uint32_t h, uint32_t layers, VkImageView velView)
{
    g_taa.device = dev;
    taaDestroy(dd);
    g_taa.device = dev;
    g_taa.w = w; g_taa.h = h; g_taa.format = fmt;
    g_taa.layers = layers ? layers : 1;
    g_taa.sceneImage = scene;
    g_taa.velView = VK_NULL_HANDLE;

    // ---- History image, same format as the scene so no conversion is implied.
    VkImageCreateInfo ici;
    memset(&ici, 0, sizeof(ici));
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent.width = w; ici.extent.height = h; ici.extent.depth = 1;
    ici.mipLevels = 1; ici.arrayLayers = g_taa.layers;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    // TRANSFER_SRC because the resolve's OUTPUT PATH reads from here: history
    // is copied into the scene target every frame. Without it that copy is
    // illegal, and validation says so twenty times a run:
    //
    //   VUID-vkCmdCopyImage-aspect-06662
    //   srcImage was created with TRANSFER_DST|SAMPLED|STORAGE but requires
    //   VK_IMAGE_USAGE_2_TRANSFER_SRC_BIT
    //
    // TRANSFER_DST was here for the initial clear and nobody added the SRC
    // side when the copy-back was introduced. The image is written, the
    // accumulation happens - viz=4 shows a real accumulated picture - and the
    // step that carries it to the screen is undefined behaviour.
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    for (int hi = 0; hi < 2; ++hi) {
        if (dd.createImage(dev, &ici, nullptr, &g_taa.history[hi]) != VK_SUCCESS) {
            trace("TAA: history image %d creation failed (%ux%u fmt=%d)", hi, w, h, (int)fmt);
            return false;
        }
        VkMemoryRequirements mr;
        dd.getImageMemReq(dev, g_taa.history[hi], &mr);
        VkMemoryAllocateInfo mai;
        memset(&mai, 0, sizeof(mai));
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = taaFindMemory(dd, mr.memoryTypeBits,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX ||
            dd.allocateMemory(dev, &mai, nullptr, &g_taa.historyMem[hi]) != VK_SUCCESS) {
            trace("TAA: history memory %d allocation failed (%llu bytes)", hi,
                  (unsigned long long)mr.size);
            return false;
        }
        dd.bindImageMemory(dev, g_taa.history[hi], g_taa.historyMem[hi], 0);
    }

    // ---- THE SHARPEN TARGET. Same shape and usage as a history image (it is
    // written by a compute dispatch and copied to the scene), but it is NEVER
    // sampled as history, so the sharpen it holds cannot compound. Optional: on
    // any failure it is left null and taaRecordResolve falls back to copying the
    // resolved history straight to screen, exactly as before the sharpen existed.
    g_taa.sharpLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (dd.createImage(dev, &ici, nullptr, &g_taa.sharpImage) == VK_SUCCESS) {
        VkMemoryRequirements mr;
        dd.getImageMemReq(dev, g_taa.sharpImage, &mr);
        VkMemoryAllocateInfo mai;
        memset(&mai, 0, sizeof(mai));
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = taaFindMemory(dd, mr.memoryTypeBits,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex != UINT32_MAX &&
            dd.allocateMemory(dev, &mai, nullptr, &g_taa.sharpMem) == VK_SUCCESS) {
            dd.bindImageMemory(dev, g_taa.sharpImage, g_taa.sharpMem, 0);
        } else {
            trace("TAA: sharpen image memory allocation failed - sharpen disabled");
            if (g_taa.sharpImage) dd.destroyImage(dev, g_taa.sharpImage, nullptr);
            g_taa.sharpImage = VK_NULL_HANDLE;
            g_taa.sharpMem   = VK_NULL_HANDLE;
        }
    } else {
        trace("TAA: sharpen image creation failed - sharpen disabled");
        g_taa.sharpImage = VK_NULL_HANDLE;
    }

    VkImageViewCreateInfo ivci;
    memset(&ivci, 0, sizeof(ivci));
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    ivci.format = fmt;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.layerCount = g_taa.layers;
    for (int hi = 0; hi < 2; ++hi) {
        ivci.image = g_taa.history[hi];
        if (dd.createImageView(dev, &ivci, nullptr, &g_taa.historyView[hi]) != VK_SUCCESS) return false;
    }
    // The sharpen target's view. If it fails, disable the sharpen rather than
    // the whole resolve - the copy-back falls back to the history image.
    if (g_taa.sharpImage != VK_NULL_HANDLE) {
        ivci.image = g_taa.sharpImage;
        if (dd.createImageView(dev, &ivci, nullptr, &g_taa.sharpView) != VK_SUCCESS) {
            trace("TAA: sharpen image view creation failed - sharpen disabled");
            dd.destroyImage(dev, g_taa.sharpImage, nullptr);
            if (g_taa.sharpMem) dd.freeMemory(dev, g_taa.sharpMem, nullptr);
            g_taa.sharpImage = VK_NULL_HANDLE;
            g_taa.sharpMem   = VK_NULL_HANDLE;
            g_taa.sharpView  = VK_NULL_HANDLE;
        }
    }
    ivci.image = scene;
    if (dd.createImageView(dev, &ivci, nullptr, &g_taa.sceneView) != VK_SUCCESS) return false;
    g_taa.sceneViews[scene] = g_taa.sceneView;

    VkSamplerCreateInfo sci;
    memset(&sci, 0, sizeof(sci));
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    // Clamp to edge: a history fetch that lands outside is already rejected by
    // the shader's bounds test, so the address mode only has to be defined.
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = 0.0f;
    if (dd.createSampler(dev, &sci, nullptr, &g_taa.sampler) != VK_SUCCESS) return false;

    // ---- A SECOND, NEAREST SAMPLER, FOR THE INTEGER FLAGS IMAGE.
    //
    // uFlags is X-Plane's gbuffer_vel, VK_FORMAT_R32_UINT. Integer formats do
    // not advertise SAMPLED_IMAGE_FILTER_LINEAR, so sampling one through the
    // linear sampler above is undefined - and validation says so on every
    // dispatch:
    //
    //   VUID-vkCmdDispatch-magFilter-04553
    //   binding 4 "uFlags" ... VK_FILTER_LINEAR ... format VK_FORMAT_R32_UINT
    //   does not contain VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT
    //
    // Nothing about a bitfield wants interpolating anyway: the shader tests
    // bit 2, and a blended fraction of two flag words is meaningless. NEAREST
    // is both legal and the only correct filter for this input.
    sci.magFilter = VK_FILTER_NEAREST;
    sci.minFilter = VK_FILTER_NEAREST;
    if (dd.createSampler(dev, &sci, nullptr, &g_taa.samplerNearest) != VK_SUCCESS)
        return false;
    // The shadow-compare pair. LINEAR + compare = free 2x2 PCF on depth.
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.compareEnable = VK_TRUE;
    sci.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    if (dd.createSampler(dev, &sci, nullptr, &g_taa.samplerShadowLE) != VK_SUCCESS)
        return false;
    sci.compareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
    if (dd.createSampler(dev, &sci, nullptr, &g_taa.samplerShadowGE) != VK_SUCCESS)
        return false;
    sci.compareEnable = VK_FALSE;
    sci.compareOp = VK_COMPARE_OP_NEVER;
    sci.magFilter = VK_FILTER_NEAREST;
    sci.minFilter = VK_FILTER_NEAREST;

    // ---- THE READBACK BUFFER. See TaaState::readBuf.
    //
    // Host-visible and COHERENT so no explicit flush or invalidate is needed;
    // persistently mapped, because mapping per frame would serialise against
    // the very timing this exists to observe. Failure here is not fatal - the
    // resolve runs exactly as before and only the diagnostic is absent.
    {
        VkBufferCreateInfo bci;
        memset(&bci, 0, sizeof(bci));
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = 2ull * 512ull * 8ull;          // history strip + scene strip
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (dd.createBuffer && dd.createBuffer(dev, &bci, nullptr, &g_taa.readBuf) == VK_SUCCESS) {
            VkMemoryRequirements mr;
            memset(&mr, 0, sizeof(mr));
            if (dd.getBufferMemReq) dd.getBufferMemReq(dev, g_taa.readBuf, &mr);
            const uint32_t mt = taaFindMemory(dd, mr.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            VkMemoryAllocateInfo mai;
            memset(&mai, 0, sizeof(mai));
            mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            mai.allocationSize  = mr.size ? mr.size : bci.size;
            mai.memoryTypeIndex = mt;
            if (mt != UINT32_MAX && dd.allocateMemory &&
                dd.allocateMemory(dev, &mai, nullptr, &g_taa.readMem) == VK_SUCCESS &&
                dd.bindBufferMemory &&
                dd.bindBufferMemory(dev, g_taa.readBuf, g_taa.readMem, 0) == VK_SUCCESS &&
                dd.mapMemory &&
                dd.mapMemory(dev, g_taa.readMem, 0, VK_WHOLE_SIZE, 0, &g_taa.readPtr) == VK_SUCCESS) {
                memset(g_taa.readPtr, 0, (size_t)bci.size);
                trace("TAA READBACK: history strip buffer mapped - the history "
                      "image can now be compared across frames directly, "
                      "instead of through the resolve's own output path");
            } else {
                g_taa.readPtr = nullptr;
            }
        }
    }

    // ---- THE FALLBACK FLAG IMAGE. 1x1, uint, zero.
    //
    // Created unconditionally so binding 4 is always valid; cleared alongside
    // the history on first use.
    {
        VkImageCreateInfo fci;
        memset(&fci, 0, sizeof(fci));
        fci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        fci.imageType = VK_IMAGE_TYPE_2D;
        fci.format = VK_FORMAT_R32_UINT;
        fci.extent.width = 1; fci.extent.height = 1; fci.extent.depth = 1;
        fci.mipLevels = 1; fci.arrayLayers = 1;
        fci.samples = VK_SAMPLE_COUNT_1_BIT;
        fci.tiling = VK_IMAGE_TILING_OPTIMAL;
        fci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        fci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        fci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (dd.createImage(dev, &fci, nullptr, &g_taa.flagsFallback) == VK_SUCCESS) {
            VkMemoryRequirements fmr;
            dd.getImageMemReq(dev, g_taa.flagsFallback, &fmr);
            VkMemoryAllocateInfo fai;
            memset(&fai, 0, sizeof(fai));
            fai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            fai.allocationSize = fmr.size;
            fai.memoryTypeIndex = taaFindMemory(dd, fmr.memoryTypeBits,
                                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (fai.memoryTypeIndex != UINT32_MAX &&
                dd.allocateMemory(dev, &fai, nullptr, &g_taa.flagsFallbackMem) == VK_SUCCESS) {
                dd.bindImageMemory(dev, g_taa.flagsFallback, g_taa.flagsFallbackMem, 0);
                VkImageViewCreateInfo fvci;
                memset(&fvci, 0, sizeof(fvci));
                fvci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                fvci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
                fvci.format = VK_FORMAT_R32_UINT;
                fvci.image = g_taa.flagsFallback;
                fvci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                fvci.subresourceRange.levelCount = 1;
                fvci.subresourceRange.layerCount = 1;
                dd.createImageView(dev, &fvci, nullptr, &g_taa.flagsFallbackView);
            }
        }
        g_taa.flagsView  = g_taa.flagsFallbackView;
        g_taa.flagsValid = false;
        if (!g_taa.flagsFallbackView)
            trace("TAA: flags fallback unavailable - the gbuffer_vel weight "
                  "override stays off");
    }

    VkDescriptorSetLayoutBinding b[13];
    memset(b, 0, sizeof(b));
    // Binding 0 is a SAMPLER now, not a storage image: the dispatch only reads
    // the scene. That is what lets the scene target keep X-Plane's own usage
    // flags untouched.
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[3].binding = 3; b[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[4].binding = 4; b[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    // Binding 5: the engine's depth, sampled. Dummy-bound to the velocity
    // target whenever the real image is unidentified or the harvest is off.
    b[5].binding = 5; b[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    // Binding 6: the per-dispatch uniform slot (reprojection matrix). Always
    // bound; the shader only reads it under kTaaFlagDepthReproject.
    b[6].binding = 6; b[6].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    // Bindings 7/8: the engine's sun cascades and environment probes; dummies
    // (the velocity view) when unidentified, gated by flags like binding 5.
    b[7].binding = 7; b[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[8].binding = 8; b[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    // Binding 9: the engine's u_shadow_data - pointed at the ENGINE'S OWN
    // buffer region when the descriptor capture has seen it, at our zeroed
    // ring slot otherwise.
    b[9].binding = 9; b[9].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    // Binding 10: the engine's u_gbuffer_data - its own depth-linearization
    // and screen-to-eye coefficients, so eye reconstruction is a
    // transcription too, not a reinvention.
    b[10].binding = 10; b[10].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    // Binding 11: the GI gather's half-res result. Dummy-bound to the velocity
    // view when the gather is off, gated by kTaaFlagGi like every other tap.
    b[11].binding = 11; b[11].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    // Binding 12: the engine's gbuf-normal, spheremap-encoded in two channels.
    // AO and contact shadows both had to work from depth alone until now - AO
    // could not tell a sloped surface from an occluder, and contact shadows
    // could not tell a surface the sun never reached from one it did. Dummy-
    // bound to the velocity view when unidentified, gated by kTaaFlagNormalTap
    // exactly like every other engine tap.
    b[12].binding = 12; b[12].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    for (int i = 0; i < 13; ++i) {
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci;
    memset(&dlci, 0, sizeof(dlci));
    dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 13; dlci.pBindings = b;
    if (dd.createDescriptorSetLayout(dev, &dlci, nullptr, &g_taa.setLayout) != VK_SUCCESS) return false;

    VkPushConstantRange pcr;
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    // ---- THE PUSH BLOCK HAS A CEILING, AND IT SHOULD SAY SO OUT LOUD.
    //
    // 128 bytes is the Vulkan guaranteed minimum for maxPushConstantsSize.
    // Desktop drivers give 256, so overrunning it would work here and fail
    // silently somewhere else - which is the worst failure shape there is.
    //
    // Contact shadows brought this to exactly 128. Anything added after this
    // point does not fit, and this assert is what says so at compile time
    // rather than as corrupted uniforms on someone else's GPU.
    static_assert(sizeof(TaaPush) <= 128,
                  "TaaPush exceeds the 128-byte guaranteed push constant limit; "
                  "move a field into the descriptor set instead of growing it.");
    pcr.size = sizeof(TaaPush);
    VkPipelineLayoutCreateInfo plci;
    memset(&plci, 0, sizeof(plci));
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &g_taa.setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (dd.createPipelineLayout(dev, &plci, nullptr, &g_taa.pipeLayout) != VK_SUCCESS) return false;

    VkShaderModuleCreateInfo smci;
    memset(&smci, 0, sizeof(smci));
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = kTaaResolveSpvWords * 4;
    smci.pCode = kTaaResolveSpv;
    VkShaderModule sm = VK_NULL_HANDLE;
    if (dd.createShaderModule(dev, &smci, nullptr, &sm) != VK_SUCCESS) {
        trace("TAA: shader module creation failed");
        return false;
    }
    VkComputePipelineCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = g_taa.pipeLayout;
    VkResult pr = dd.createComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &g_taa.pipeline);
    dd.destroyShaderModule(dev, sm, nullptr);
    if (pr != VK_SUCCESS) {
        trace("TAA: compute pipeline creation failed (%d)", (int)pr);
        return false;
    }

    VkDescriptorPoolSize ps[3];
    ps[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ps[0].descriptorCount = 1 * TaaState::kSets;   // history write only
    ps[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ps[1].descriptorCount = 9 * TaaState::kSets;   // + sun cascades, env probes, GI, normals
    ps[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ps[2].descriptorCount = 3 * TaaState::kSets;   // reproj + u_shadow_data + u_gbuffer_data
    VkDescriptorPoolCreateInfo dpci;
    memset(&dpci, 0, sizeof(dpci));
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = TaaState::kSets;
    dpci.poolSizeCount = 3; dpci.pPoolSizes = ps;
    if (dd.createDescriptorPool(dev, &dpci, nullptr, &g_taa.pool) != VK_SUCCESS) return false;

    VkDescriptorSetLayout layouts[TaaState::kSets];
    for (uint32_t i = 0; i < TaaState::kSets; ++i) layouts[i] = g_taa.setLayout;
    VkDescriptorSetAllocateInfo dsai;
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = g_taa.pool;
    dsai.descriptorSetCount = TaaState::kSets;
    dsai.pSetLayouts = layouts;
    if (dd.allocateDescriptorSets(dev, &dsai, g_taa.sets) != VK_SUCCESS) return false;

    // ---- THE REPROJECTION SLOT RING.
    //
    // 256 bytes per slot (the worst-case minUniformBufferOffsetAlignment),
    // one slot per descriptor set, HOST_VISIBLE|COHERENT and mapped for the
    // lifetime of the state. The record path writes the slot belonging to the
    // set it is about to dispatch with; the ring depth is what makes that
    // write safe while earlier dispatches may still be reading their slots.
    //
    // Failure here IS fatal to init, deliberately. Binding 6 is statically
    // used by the shader, so a descriptor set with no buffer written there is
    // invalid Vulkan whatever the runtime flag says - the graceful path would
    // be undefined behaviour wearing a seatbelt. A 4 KB host-visible buffer
    // failing to allocate means the device is in far worse trouble than a
    // missing resolve.
    {
        PFN_vkCreateBuffer  pfnCreateBuffer =
            (PFN_vkCreateBuffer)dd.gdpa(dev, "vkCreateBuffer");
        PFN_vkGetBufferMemoryRequirements pfnBufReq =
            (PFN_vkGetBufferMemoryRequirements)dd.gdpa(dev, "vkGetBufferMemoryRequirements");
        PFN_vkBindBufferMemory pfnBindBuf =
            (PFN_vkBindBufferMemory)dd.gdpa(dev, "vkBindBufferMemory");
        PFN_vkMapMemory pfnMap = (PFN_vkMapMemory)dd.gdpa(dev, "vkMapMemory");
        if (pfnCreateBuffer && pfnBufReq && pfnBindBuf && pfnMap) {
            VkBufferCreateInfo bci;
            memset(&bci, 0, sizeof(bci));
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            // 2048 per slot: the first 96 bytes are the reproj block, the
            // rest stays zero and doubles as the u_shadow_data fallback
            // region (the shader reads offsets up to ~1548 when the tap flag
            // is off the values must still be READABLE, just zero).
            bci.size  = 2048ull * TaaState::kSets;
            bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (pfnCreateBuffer(dev, &bci, nullptr, &g_taa.uboBuf) == VK_SUCCESS) {
                VkMemoryRequirements mr;
                pfnBufReq(dev, g_taa.uboBuf, &mr);
                VkMemoryAllocateInfo mai;
                memset(&mai, 0, sizeof(mai));
                mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                mai.allocationSize = mr.size;
                mai.memoryTypeIndex = taaFindMemory(dd, mr.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                if (mai.memoryTypeIndex != UINT32_MAX &&
                    dd.allocateMemory(dev, &mai, nullptr, &g_taa.uboMem) == VK_SUCCESS &&
                    pfnBindBuf(dev, g_taa.uboBuf, g_taa.uboMem, 0) == VK_SUCCESS &&
                    pfnMap(dev, g_taa.uboMem, 0, VK_WHOLE_SIZE, 0,
                           &g_taa.uboMap) == VK_SUCCESS) {
                    // Zeroed ONCE here, which is what makes the fallback
                    // regions' "readable zeros" true rather than a comment's
                    // wish over undefined allocation contents.
                    memset(g_taa.uboMap, 0, (size_t)bci.size);
                } else {
                    g_taa.uboMap = nullptr;
                    trace("TAA: reproj UBO memory setup failed - resolve "
                          "cannot initialise (binding 6 must hold a buffer)");
                    return false;
                }
            } else {
                trace("TAA: reproj UBO creation failed - resolve cannot "
                      "initialise");
                return false;
            }
        } else {
            trace("TAA: buffer entry points unavailable - resolve cannot "
                  "initialise");
            return false;
        }
    }

    {
        VkImageCreateInfo ic;
        memset(&ic, 0, sizeof(ic));
        ic.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ic.imageType = VK_IMAGE_TYPE_2D;
        ic.format = VK_FORMAT_D16_UNORM;
        ic.extent.width = 1; ic.extent.height = 1; ic.extent.depth = 1;
        ic.mipLevels = 1; ic.arrayLayers = 1;
        ic.samples = VK_SAMPLE_COUNT_1_BIT;
        ic.tiling = VK_IMAGE_TILING_OPTIMAL;
        ic.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                   VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        ic.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (dd.createImage && dd.createImage(dev, &ic, nullptr,
                                               &g_taa.sunDummy) == VK_SUCCESS) {
            VkMemoryRequirements mr;
            dd.getImageMemReq(dev, g_taa.sunDummy, &mr);
            VkMemoryAllocateInfo ma;
            memset(&ma, 0, sizeof(ma));
            ma.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ma.allocationSize = mr.size;
            ma.memoryTypeIndex = taaFindMemory(dd, mr.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (ma.memoryTypeIndex != UINT32_MAX &&
                dd.allocateMemory(dev, &ma, nullptr, &g_taa.sunDummyMem) == VK_SUCCESS &&
                dd.bindImageMemory(dev, g_taa.sunDummy, g_taa.sunDummyMem, 0) == VK_SUCCESS) {
                VkImageViewCreateInfo vv;
                memset(&vv, 0, sizeof(vv));
                vv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                vv.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
                vv.format = VK_FORMAT_D16_UNORM;
                vv.image = g_taa.sunDummy;
                vv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
                vv.subresourceRange.levelCount = 1;
                vv.subresourceRange.layerCount = 1;
                dd.createImageView(dev, &vv, nullptr, &g_taa.sunDummyView);
            }
        }
        if (g_taa.sunDummyView == VK_NULL_HANDLE)
            trace("TAA: sun dummy depth creation failed - binding 7 fallback "
                  "will violate spec until cascades identify");
    }

    g_taa.velView = velView;
    g_taa.ready = true;
    g_taa.historyCleared = false;
    trace("TAA: ready - %ux%u x%u layer(s) fmt=%d, mode %d, alpha %.3f, "
          "%u descriptor sets",
          w, h, g_taa.layers, (int)fmt, taaMode(), taaAlpha(), TaaState::kSets);
    return true;
}

// Point the resolve at whichever target this frame drew into, creating that
// view once. Returns false only if the view cannot be made.
static bool taaBindScene(DeviceData &dd, VkImage scene)
{
    if (scene == VK_NULL_HANDLE) return false;
    std::map<VkImage, VkImageView>::iterator it = g_taa.sceneViews.find(scene);
    if (it != g_taa.sceneViews.end()) {
        g_taa.sceneView  = it->second;
        g_taa.sceneImage = scene;
        return true;
    }
    VkImageViewCreateInfo ivci;
    memset(&ivci, 0, sizeof(ivci));
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    ivci.format = g_taa.format;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.layerCount = g_taa.layers;
    ivci.image = scene;
    VkImageView v = VK_NULL_HANDLE;
    if (dd.createImageView(g_taa.device, &ivci, nullptr, &v) != VK_SUCCESS) {
        trace("TAA: scene view creation failed for a second target");
        return false;
    }
    g_taa.sceneViews[scene] = v;
    g_taa.sceneView  = v;
    g_taa.sceneImage = scene;
    trace("TAA: now %llu distinct scene targets - X-Plane alternates between "
          "them, so views are cached rather than rebuilt",
          (unsigned long long)g_taa.sceneViews.size());
    return true;
}

// ---- THE TAA BACKEND.
//
// The first implementation of temporal::IBackend, and for now the whole of it.
// It is written against the shared contract rather than against layer.cpp so
// that FSR, DLSS, XeSS and frame generation can be added beside it without any
// of them reaching into the interception code - and so that the questions each
// of them will ask (what shape is the target, which convention are the vectors
// in, why was history reset) are already answered in one place.
//
// The conversion table for this backend, stated rather than assumed:
//
//   coordinateSpace       UV        the stored 0.5 IS the NDC->UV conversion
//   direction             prev-curr we store prev minus curr
//   jitterIncluded        false     jitter is applied AFTER the varyings
//   cameraMotionIncluded  true
//   objectMotionIncluded  true      via the injected per-draw matrices
//
// This backend consumes those directly, so nothing converts. That is exactly
// what makes it the wrong place to discover a convention bug, and exactly why
// the declaration is written down for the backends that WILL convert.
class TaaBackend : public temporal::IBackend {
public:
    temporal::BackendInfo info() const override
    {
        temporal::BackendInfo i;
        i.name = "TAA";
        i.vendor = temporal::VENDOR_ANY;      // ours; always available
        i.supportsUpscaling = false;          // TAAU is a separate backend
        i.supportsNativeResolution = true;
        i.supportsArrayLayers = true;         // 2D_ARRAY views, dispatch z=layers
        i.supportsMultisample = false;        // see accepts()
        return i;
    }

    // Report WHY, not just no. A backend silently absent is indistinguishable
    // from a backend silently broken, which is how a dead code path once hid
    // through two crashes and a wrong diagnosis.
    bool accepts(const temporal::TemporalFrame &f, const char **why) const override
    {
        if (f.color.image == VK_NULL_HANDLE) {
            if (why) *why = "no colour target";
            return false;
        }
        if (f.color.samples != VK_SAMPLE_COUNT_1_BIT) {
            // 576 of the 1152 gbuffer_lit permutations are multisampled, and
            // fix_hdr_1 binds the HDR image as a multisampled storage image, so
            // this is a real configuration rather than a theoretical one. It
            // needs a per-sample resolve, which this shader does not do.
            // Declining leaves the frame untouched; pretending would corrupt it.
            if (why) *why = "multisampled target - needs a per-sample resolve";
            return false;
        }
        if (f.motion.image == VK_NULL_HANDLE || f.motion.view == VK_NULL_HANDLE) {
            if (why) *why = "no motion vectors this frame";
            return false;
        }
        if (f.motion.coordinateSpace != temporal::COORD_UV ||
            f.motion.direction != temporal::DIR_PREVIOUS_TO_CURRENT) {
            // The shader hard-codes this convention. If the core ever produces
            // vectors in another one, this must fail rather than reinterpret
            // them - a silently misread field is the failure mode the
            // MotionVectors metadata exists to prevent.
            if (why) *why = "motion vectors are not UV / previous-minus-current";
            return false;
        }
        if (f.motion.jitterIncluded) {
            if (why) *why = "motion vectors have jitter baked in";
            return false;
        }
        return true;
    }

    bool record(VkCommandBuffer cb, const temporal::TemporalFrame &f) override;
};

static TaaBackend g_taaBackend;

// The device dispatch the backend records through. It lives here rather than in
// TemporalFrame because it is a property of the LAYER's Vulkan plumbing, not of
// the frame - and TemporalFrame is meant to describe a frame to a consumer that
// knows nothing about how we intercepted it.
static DeviceData *g_taaDevice = nullptr;

// Point binding 4 at X-Plane's gbuffer_vel, creating that view once per image.
// Refuses multisampled or non-uint candidates: the shader declares a
// single-sample uint array sampler, and a mismatched view is exactly the class
// of silent wrongness the shape checks exist to prevent.
// Point binding 5 at X-Plane's own depth image (the one the engine names
// "gbuf-depth"), creating the view once per image. DEPTH aspect: the format is
// a depth format and a colour-aspect view of it is invalid. Multisampled
// candidates are refused for the same reason as the flags binding.
static void taaBindEngineDepth(DeviceData &dd, VkImage image, VkFormat fmt,
                               VkSampleCountFlagBits samples)
{
    if (image == VK_NULL_HANDLE || samples != VK_SAMPLE_COUNT_1_BIT) {
        g_taa.edView  = VK_NULL_HANDLE;
        g_taa.edValid = false;
        return;
    }
    std::map<VkImage, VkImageView>::iterator it = g_taa.edViews.find(image);
    if (it != g_taa.edViews.end()) {
        g_taa.edView  = it->second;
        g_taa.edValid = (it->second != VK_NULL_HANDLE);
        return;
    }
    VkImageViewCreateInfo v;
    memset(&v, 0, sizeof(v));
    v.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    v.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    v.format = fmt;
    v.image = image;
    v.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    v.subresourceRange.levelCount = 1;
    v.subresourceRange.layerCount = 1;
    VkImageView view = VK_NULL_HANDLE;
    if (dd.createImageView(g_taa.device, &v, nullptr, &view) != VK_SUCCESS) {
        trace("TAA: gbuf-depth view creation failed (fmt=%d) - pos harvest "
              "stays on the injected clip-w", (int)fmt);
        view = VK_NULL_HANDLE;
    } else {
        trace("TAA: engine gbuf-depth bound (fmt=%d) - taa.pos_harvest=1 now "
              "reads the engine's own depth for AO and contact shadows.",
              (int)fmt);
    }
    g_taa.edViews[image] = view;
    g_taa.edView  = view;
    g_taa.edValid = (view != VK_NULL_HANDLE);
}

// ---- A CACHED VIEW MUST NOT OUTLIVE THE IMAGE IT VIEWS.
//
// The engine-image binders cache one VkImageView per VkImage so a view is
// created once rather than every frame. Nothing purged those caches when the
// image itself was destroyed - only the full re-init teardown did - and
// X-Plane destroys these images on every "Rebuilding offscreens", which is
// part of loading an aircraft.
//
// Three separate faults, from one omission:
//
//   - The view was never destroyed. One leaked VkImageView per engine image
//     per rebuild, for the whole session.
//   - Destroying a VkImage while views of it still exist is itself invalid
//     (VUID-vkDestroyImage-image-04882), so we were making X-Plane's own
//     destroy call illegal.
//   - Worst: the map stayed keyed by a DEAD VkImage handle, and Vulkan reuses
//     handle values. A new image landing on the same handle hit the cache and
//     the binder handed a descriptor set a view of an image that no longer
//     exists - a use-after-free the GPU discovers, which is exactly the
//     "virtual address 0x0, Type: Compute" signature.
//
// Called immediately before the real vkDestroyImage on BOTH paths - the direct
// one and the four-present deferred one - so the view dies with its image and
// never before it, which is the ordering the in-flight protection exists for.
// ---- DETACH ONLY. THE DESTROY HAPPENS ELSEWHERE, LATER, AND UNLOCKED.
//
// The first version of this destroyed each view the moment its image was
// destroyed, inline, while holding g_lock. Two faults, and a resolution change
// hits both at once because it destroys many images in one burst:
//
//   - A view still referenced by an in-flight descriptor set was freed out
//     from under the GPU. The four-present deferral protects the IMAGE for
//     exactly this reason; the view had no such guard on the immediate path.
//   - It called down the dispatch chain (dd.destroyImageView) with g_lock
//     held, which the stability sweep established this lock must never do.
//
// So this now only unhooks the bookkeeping and HANDS BACK the views. The
// caller is responsible for destroying them at a safe point - which, because
// Vulkan requires a view to die BEFORE its image, means the image has to be
// deferred alongside them rather than the view deferred alone.
static void taaDetachImageViews(VkImage image, std::vector<VkImageView> &out)
{
    if (image == VK_NULL_HANDLE || g_taa.device == VK_NULL_HANDLE) return;
    std::map<VkImage, VkImageView> *caches[6] = {
        &g_taa.sceneViews, &g_taa.flagsViews, &g_taa.edViews,
        &g_taa.sunViews,   &g_taa.probeViews, &g_taa.normViews };
    for (int ci = 0; ci < 6; ++ci) {
        std::map<VkImage, VkImageView>::iterator it = caches[ci]->find(image);
        if (it == caches[ci]->end()) continue;
        const VkImageView v = it->second;
        caches[ci]->erase(it);
        if (v == VK_NULL_HANDLE) continue;
        out.push_back(v);
        // Clear the LIVE selection too, or the next resolve binds the handle we
        // are about to destroy.
        if (g_taa.edView    == v) { g_taa.edView    = VK_NULL_HANDLE; g_taa.edValid    = false; }
        if (g_taa.sunView   == v) { g_taa.sunView   = VK_NULL_HANDLE; g_taa.sunValid   = false; }
        if (g_taa.probeView == v) { g_taa.probeView = VK_NULL_HANDLE; g_taa.probeValid = false; }
        if (g_taa.normView  == v) { g_taa.normView  = VK_NULL_HANDLE; g_taa.normValid  = false; }
        if (g_taa.flagsView == v) { g_taa.flagsView = VK_NULL_HANDLE; g_taa.flagsValid = false; }
        if (g_taa.sceneView == v)   g_taa.sceneView = VK_NULL_HANDLE;
    }
}

// Bindings 7 and 8: the engine's sun cascades (depth aspect) and environment
// probe cubemaps (colour), cached-view binders in the gbuf-depth mould.
// Binding for the engine's own eye-space normals (gbuf-normal, R16G16_SFLOAT,
// spheremap-encoded). Colour aspect, single sample - the same discipline as
// the other engine-image binders.
static void taaBindNormals(DeviceData &dd, VkImage image, VkFormat fmt,
                           uint32_t layers, VkSampleCountFlagBits samples)
{
    if (image == VK_NULL_HANDLE || samples != VK_SAMPLE_COUNT_1_BIT) {
        g_taa.normView = VK_NULL_HANDLE; g_taa.normValid = false; return;
    }
    std::map<VkImage, VkImageView>::iterator it = g_taa.normViews.find(image);
    if (it != g_taa.normViews.end()) {
        g_taa.normView = it->second;
        g_taa.normValid = (it->second != VK_NULL_HANDLE);
        return;
    }
    VkImageViewCreateInfo v;
    memset(&v, 0, sizeof(v));
    v.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    v.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    v.format = fmt;
    v.image = image;
    v.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    v.subresourceRange.levelCount = 1;
    v.subresourceRange.layerCount = layers ? layers : 1;
    VkImageView view = VK_NULL_HANDLE;
    if (dd.createImageView(g_taa.device, &v, nullptr, &view) != VK_SUCCESS) {
        trace("TAA: gbuf-normal view failed (fmt=%d)", (int)fmt);
        view = VK_NULL_HANDLE;
    } else {
        trace("TAA: engine gbuf-normal bound (fmt=%d) - the GI gather uses "
              "the engine's own normals instead of depth derivatives.",
              (int)fmt);
    }
    g_taa.normViews[image] = view;
    g_taa.normView  = view;
    g_taa.normValid = (view != VK_NULL_HANDLE);
}

static void taaBindSunShadow(DeviceData &dd, VkImage image, VkFormat fmt,
                             uint32_t layers, VkSampleCountFlagBits samples)
{
    if (image == VK_NULL_HANDLE || samples != VK_SAMPLE_COUNT_1_BIT) {
        g_taa.sunView = VK_NULL_HANDLE; g_taa.sunValid = false; return;
    }
    std::map<VkImage, VkImageView>::iterator it = g_taa.sunViews.find(image);
    if (it != g_taa.sunViews.end()) {
        g_taa.sunView = it->second;
        g_taa.sunValid = (it->second != VK_NULL_HANDLE);
        return;
    }
    VkImageViewCreateInfo v;
    memset(&v, 0, sizeof(v));
    v.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    v.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    v.format = fmt;
    v.image = image;
    v.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    v.subresourceRange.levelCount = 1;
    v.subresourceRange.layerCount = layers ? layers : 1;
    VkImageView view = VK_NULL_HANDLE;
    if (dd.createImageView(g_taa.device, &v, nullptr, &view) != VK_SUCCESS) {
        trace("TAA: csm_shadow_maps view failed (fmt=%d)", (int)fmt);
        view = VK_NULL_HANDLE;
    } else {
        trace("TAA: engine sun cascades bound (fmt=%d, %u layers) - "
              "viz 9 shows per-pixel sun visibility.", (int)fmt, layers);
    }
    g_taa.sunViews[image] = view;
    g_taa.sunView  = view;
    g_taa.sunValid = (view != VK_NULL_HANDLE);
}

static void taaBindEnvProbes(DeviceData &dd, VkImage image, VkFormat fmt,
                             uint32_t layers, VkSampleCountFlagBits samples)
{
    if (image == VK_NULL_HANDLE || samples != VK_SAMPLE_COUNT_1_BIT) {
        g_taa.probeView = VK_NULL_HANDLE; g_taa.probeValid = false;
    g_taa.normView  = VK_NULL_HANDLE; g_taa.normValid  = false; return;
    }
    std::map<VkImage, VkImageView>::iterator it = g_taa.probeViews.find(image);
    if (it != g_taa.probeViews.end()) {
        g_taa.probeView = it->second;
        g_taa.probeValid = (it->second != VK_NULL_HANDLE);
        return;
    }
    VkImageViewCreateInfo v;
    memset(&v, 0, sizeof(v));
    v.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    v.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    v.format = fmt;
    v.image = image;
    v.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    v.subresourceRange.levelCount = 1;
    v.subresourceRange.layerCount = layers ? layers : 1;
    VkImageView view = VK_NULL_HANDLE;
    if (dd.createImageView(g_taa.device, &v, nullptr, &view) != VK_SUCCESS) {
        trace("TAA: environment_probes view failed (fmt=%d)", (int)fmt);
        view = VK_NULL_HANDLE;
    } else {
        trace("TAA: engine environment probes bound (fmt=%d, %u faces) - "
              "viz 10 shows probe radiance along the view ray.", (int)fmt, layers);
    }
    g_taa.probeViews[image] = view;
    g_taa.probeView  = view;
    g_taa.probeValid = (view != VK_NULL_HANDLE);
}

static void taaBindFlags(DeviceData &dd, VkImage image, VkFormat fmt,
                         uint32_t layers, VkSampleCountFlagBits samples)
{
    if (image == VK_NULL_HANDLE || samples != VK_SAMPLE_COUNT_1_BIT) {
        g_taa.flagsView  = g_taa.flagsFallbackView;
        g_taa.flagsValid = false;
        return;
    }
    std::map<VkImage, VkImageView>::iterator it = g_taa.flagsViews.find(image);
    if (it != g_taa.flagsViews.end()) {
        g_taa.flagsView  = it->second;
        g_taa.flagsValid = (it->second != VK_NULL_HANDLE);
        return;
    }
    VkImageViewCreateInfo v;
    memset(&v, 0, sizeof(v));
    v.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    v.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    v.format = fmt;
    v.image = image;
    v.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    v.subresourceRange.levelCount = 1;
    v.subresourceRange.layerCount = layers ? layers : 1;
    VkImageView view = VK_NULL_HANDLE;
    if (dd.createImageView(g_taa.device, &v, nullptr, &view) != VK_SUCCESS) {
        trace("TAA: gbuffer_vel view creation failed - weight override off");
        view = VK_NULL_HANDLE;
    } else {
        trace("TAA: gbuffer_vel bound - moving-geometry pixels (bit 2) now "
              "take the current frame outright. taa.objflags=0 disables live.");
    }
    g_taa.flagsViews[image] = view;
    g_taa.flagsView  = view ? view : g_taa.flagsFallbackView;
    g_taa.flagsValid = (view != VK_NULL_HANDLE);
}

// ---- RECORD THE RESOLVE.
//
// Called from vkCmdEndRendering once the scene pass has finished, which is the
// only point where the colour target holds a complete frame and the velocity
// target beside it describes that same frame.
// ---- DELIVER HISTORY WITHOUT RESOLVING. (the alternation fix)
//
// The copy that carries the accumulated image into the scene target lives at
// the end of taaRecordResolve, so a frame that does not dispatch does not
// deliver either - it ships the raw scene. Measured, static camera:
//
//     history buffer      0.447   <- stable, accumulating correctly
//     composited output   3.78    <- unstable
//     TAA off baseline    0.806
//
// History is MORE stable than the raw scene. The resolve works. What alternates
// is whether its result reaches the screen, and alternating between accumulated
// and raw output is the shake - which is also why jitter made it visible, since
// without jitter the two look nearly identical.
//
// So when a frame has a validated target and a ready history but skips the
// dispatch - the per-present resolve cap, a quiesce, a stale velocity field -
// it still delivers the last good accumulation instead of nothing. That is
// strictly better than raw: the image is at most one frame old, where before it
// was a different image entirely.
//
// Only called with a target this frame's pass actually rendered into; there is
// deliberately no path for frames with no candidate pass at all, because the
// destination would be a guess and copying into the wrong half is the smearing
// the target-latch experiment produced.
static void taaRecordDeliverOnly(DeviceData &dd, VkCommandBuffer cb, VkImage scene)
{
    if (!g_taa.ready || !taaEnabled() || !g_taa.historyCleared) return;
    if (scene == VK_NULL_HANDLE || !dd.cmdCopyImage) return;
    // The most recently written buffer. The flip happens at the end of a
    // resolve, so the freshest history is the one the index no longer points at.
    const uint32_t src = g_taa.historyWrite ^ 1u;
    if (!g_taa.history[src]) return;

    VkImageMemoryBarrier pre[2];
    memset(pre, 0, sizeof(pre));
    for (int i = 0; i < 2; ++i) {
        pre[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        pre[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pre[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pre[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        pre[i].subresourceRange.levelCount = 1;
        pre[i].subresourceRange.layerCount = g_taa.layers;
    }
    pre[0].image = scene;
    pre[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    pre[0].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    pre[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    pre[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    pre[1].image = g_taa.history[src];
    pre[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    pre[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    pre[1].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    pre[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    // Each half named with a stage that can actually perform its access - the
    // mistake VUID-02819 caught in the resolve's own barrier.
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                          0, nullptr, 0, nullptr, 1, &pre[0]);
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                          0, nullptr, 0, nullptr, 1, &pre[1]);

    VkImageCopy cp;
    memset(&cp, 0, sizeof(cp));
    cp.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    cp.srcSubresource.layerCount = g_taa.layers;
    cp.dstSubresource = cp.srcSubresource;
    cp.extent.width = g_taa.w; cp.extent.height = g_taa.h; cp.extent.depth = 1;
    dd.cmdCopyImage(cb, g_taa.history[src], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    scene, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);

    VkImageMemoryBarrier post[2] = { pre[0], pre[1] };
    post[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    post[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    post[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    post[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    post[1].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    post[1].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    post[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    post[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                          0, nullptr, 0, nullptr, 1, &post[0]);
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                          0, nullptr, 0, nullptr, 1, &post[1]);
}

static void taaRecordResolve(DeviceData &dd, VkCommandBuffer cb,
                             float jitterX, float jitterY, bool reset,
                             bool cameraMoved)
{
    if (!g_taa.ready || !taaEnabled()) return;

    // One-time: the 1x1 comparison-sampler dummy leaves UNDEFINED layout the
    // first time any resolve records - sampling an UNDEFINED image is itself
    // the class of invalid use the dummy exists to prevent.
    if (g_taa.sunDummy != VK_NULL_HANDLE && !g_taa.sunDummyReady) {
        VkImageMemoryBarrier db;
        memset(&db, 0, sizeof(db));
        db.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        db.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        db.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        db.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        db.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        db.image = g_taa.sunDummy;
        db.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        db.subresourceRange.levelCount = 1;
        db.subresourceRange.layerCount = 1;
        db.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0, 0, nullptr, 0, nullptr, 1, &db);
        g_taa.sunDummyReady = true;
    }

    // The engine's u_gbuffer_data region for the gather: the same frame-
    // matched lookup the resolve's own binding 10 uses, hoisted here because
    // the gather is recorded before that code runs.
    VkBuffer     giGbufBuf = VK_NULL_HANDLE;
    VkDeviceSize giGbufOff = 0, giGbufRange = 0;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<VkCommandBuffer, MvShadowRegion>::iterator it =
            g_cbGbufRegion.find(cb);
        if (it != g_cbGbufRegion.end()) {
            giGbufBuf = it->second.buf; giGbufOff = it->second.off;
            giGbufRange = it->second.range;
        } else if (g_gbufDataBuf != VK_NULL_HANDLE) {
            giGbufBuf = g_gbufDataBuf; giGbufOff = g_gbufDataOff;
            giGbufRange = g_gbufDataRange;
        }
    }

    VkImageMemoryBarrier bar[3];
    memset(bar, 0, sizeof(bar));
    for (int i = 0; i < 3; ++i) {
        bar[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        bar[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        bar[i].subresourceRange.levelCount = 1;
        bar[i].subresourceRange.layerCount = g_taa.layers;
    }
    bar[0].image = g_taa.sceneImage;
    bar[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    bar[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bar[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    bar[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    const uint32_t hw_ = g_taa.historyWrite;         // written this frame
    const uint32_t hr_ = hw_ ^ 1u;                    // read this frame
    bar[1].image = g_taa.history[hw_];
    bar[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bar[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    bar[1].oldLayout = g_taa.historyCleared ? VK_IMAGE_LAYOUT_GENERAL
                                            : VK_IMAGE_LAYOUT_UNDEFINED;
    bar[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;

    bar[2].image = g_mv.image;
    bar[2].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    bar[2].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bar[2].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    bar[2].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkImageMemoryBarrier barRead = bar[1];
    barRead.image = g_taa.history[hr_];
    barRead.oldLayout = g_taa.historyCleared ? VK_IMAGE_LAYOUT_GENERAL
                                             : VK_IMAGE_LAYOUT_UNDEFINED;
    barRead.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    // ---- TWO BARRIERS, BECAUSE THE FOUR IMAGES COME FROM DIFFERENT STAGES.
    //
    // All four used to go through one barrier whose source stage was chosen for
    // the colour attachments. But bar[1] and barRead describe the HISTORY
    // images, whose previous write was last frame's COMPUTE dispatch, and
    // SHADER_WRITE is not an access that can occur in COLOR_ATTACHMENT_OUTPUT.
    // The source half of those two was therefore empty: no execution dependency
    // on last frame's dispatch, and no availability operation for its writes.
    // Validation names them individually:
    //
    //   VUID-vkCmdPipelineBarrier-pImageMemoryBarriers-02819
    //   pImageMemoryBarriers[1].srcAccessMask (VK_ACCESS_2_SHADER_WRITE_BIT)
    //   is not supported by stage mask (COLOR_ATTACHMENT_OUTPUT)   [and [3]]
    //
    // Indices 1 and 3 are exactly bar[1] and barRead in the old all4 array.
    //
    // The copy-back does not rescue it: that path ends on a barrier whose
    // srcAccessMask is TRANSFER_READ, and a read performs no availability
    // operation, so the compute write is never made available to the next
    // frame's sampled read. Splitting gives each pair a source scope that
    // actually contains the write it is describing - and the history pair also
    // names TRANSFER, since the copy-back reads these images too.
    {
        VkImageMemoryBarrier att[2] = { bar[0], bar[2] };
        dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                              0, nullptr, 0, nullptr, 2, att);
        VkImageMemoryBarrier hist[2] = { bar[1], barRead };
        dd.cmdPipelineBarrier(cb,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                              0, nullptr, 0, nullptr, 2, hist);
    }

    // ---- THE GATHER, AFTER THE TRANSITIONS AND BEFORE THE RESOLVE.
    //
    // It must come after the barriers above, not before them: bar[0] moves the
    // scene image from COLOR_ATTACHMENT_OPTIMAL to SHADER_READ_ONLY_OPTIMAL,
    // and the gather samples that image while declaring the latter. Recorded
    // ahead of the barrier it was originally placed before, it would have been
    // sampling an image in a layout it does not have - invalid, and the kind
    // of invalid a driver may honour on one machine and not another.
    //
    // Still ahead of the resolve's own dispatch, so the result is composited
    // in the same frame, with the gather's trailing barrier ordering it.
    gi::record(dd, g_taa.device, cb, g_taa.sceneView, g_taa.velView,
               g_taa.edValid ? g_taa.edView : VK_NULL_HANDLE,
               g_taa.probeValid ? g_taa.probeView : VK_NULL_HANDLE,
               g_taa.w, g_taa.h, g_taaEdAB[0], g_taaEdAB[1],
               g_taaInvProj[0], g_taaInvProj[1],
               // Read directly rather than through the seqlock snapshot below:
               // this is a single float that changes only when the viewport
               // orientation does, and the snapshot is taken further down than
               // the gather now sits.
               g_taaVpYSign,
               g_taa.normValid ? g_taa.normView : VK_NULL_HANDLE,
               giGbufBuf, giGbufOff, giGbufRange);

    // Clear the history once, explicitly. Reading an UNDEFINED image gives
    // undefined contents and on the first frame every output pixel is a blend
    // with it - the shape of the whole-frame magenta the old resolve produced.
    if (!g_taa.historyCleared) {
        VkClearColorValue cc;
        memset(&cc, 0, sizeof(cc));
        VkImageSubresourceRange r;
        memset(&r, 0, sizeof(r));
        r.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        r.levelCount = 1; r.layerCount = g_taa.layers;
        VkImageMemoryBarrier tb[2];
        for (int hi = 0; hi < 2; ++hi) {
            tb[hi] = bar[1];
            tb[hi].image = g_taa.history[hi];
            tb[hi].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            tb[hi].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            tb[hi].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        }
        dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                              0, nullptr, 0, nullptr, 2, tb);
        // Both buffers: whichever is read first must be defined too.
        for (int hi = 0; hi < 2; ++hi)
            dd.cmdClearColorImage(cb, g_taa.history[hi],
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cc, 1, &r);
        // The flags fallback too: UNDEFINED to zero to SHADER_READ_ONLY, once.
        if (g_taa.flagsFallback != VK_NULL_HANDLE) {
            VkImageMemoryBarrier fb = tb[0];
            fb.image = g_taa.flagsFallback;
            fb.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            fb.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            fb.srcAccessMask = 0;
            fb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            fb.subresourceRange.layerCount = 1;
            dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                  0, nullptr, 0, nullptr, 1, &fb);
            VkClearColorValue fz;
            memset(&fz, 0, sizeof(fz));
            dd.cmdClearColorImage(cb, g_taa.flagsFallback,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &fz, 1,
                                  &fb.subresourceRange);
            fb.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            fb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            fb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            fb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                  0, nullptr, 0, nullptr, 1, &fb);
        }
        for (int hi = 0; hi < 2; ++hi) {
            tb[hi].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            tb[hi].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            tb[hi].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            tb[hi].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        }
        dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                              0, nullptr, 0, nullptr, 2, tb);
        g_taa.historyCleared = true;
        reset = true;
    }

    // Coherent copy of the frame-thread globals: retry while the writer is
    // mid-update (odd) or moved between our reads.
    float snapSun[3], snapEd[2], snapIp[2], snapReproj[16], snapYSign;
    bool  snapReprojValid;
    for (;;) {
        const uint32_t s1 = g_taaShareSeq.load(std::memory_order_acquire);
        if (s1 & 1u) continue;
        memcpy(snapSun,    g_taaSunView, sizeof(snapSun));
        memcpy(snapEd,     g_taaEdAB,    sizeof(snapEd));
        memcpy(snapIp,     g_taaInvProj, sizeof(snapIp));
        memcpy(snapReproj, g_taaReproj,  sizeof(snapReproj));
        snapYSign = g_taaVpYSign;
        snapReprojValid = g_taaReprojValid;
        if (g_taaShareSeq.load(std::memory_order_acquire) == s1) break;
    }

    const uint32_t setIdx = g_taa.nextSet;
    VkDescriptorSet set = g_taa.sets[setIdx];
    g_taa.nextSet = (setIdx + 1) % TaaState::kSets;

    // ---- THIS DISPATCH'S REPROJECTION SLOT.
    //
    // Written before the descriptor that points at it, same slot index as the
    // descriptor set so ring depth covers in-flight reads. Layout mirrors the
    // shader's std140 block: mat4 (column-major, as published), then a vec4 of
    // (ySign, valid, 0, 0).
    const bool reprojOn = g_taa.uboMap && snapReprojValid &&
                          taaPosHarvest() && g_taa.edValid &&
                          taaNovecReproject();
    if (g_taa.uboMap) {
        float *slot = (float *)((char *)g_taa.uboMap + 2048u * setIdx);
        memcpy(slot, snapReproj, 16 * sizeof(float));
        slot[16] = snapYSign;
        slot[17] = reprojOn ? 1.0f : 0.0f;
        slot[18] = 0.0f;
        slot[19] = 0.0f;
        // Second param row: eye-position reconstruction for the cascade tap.
        slot[20] = snapIp[0];
        slot[21] = snapIp[1];
        // GI strength: see the note on uReprojParams2 in the shader. Zeroed
        // when the gather has produced nothing, so the composite cannot read
        // a strength for a result that does not exist.
        slot[22] = (gi::resultView() != VK_NULL_HANDLE)
                 ? live::f("taa.gi_strength", "TAA_GI_STRENGTH", 0.5f) : 0.0f;
        slot[22] = 0.0f;
        slot[23] = 0.0f;
    }

    VkDescriptorImageInfo ii[8];
    memset(ii, 0, sizeof(ii));
    ii[0].imageView = g_taa.sceneView;
    ii[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ii[0].sampler   = g_taa.sampler;
    ii[1].imageView = g_taa.historyView[hw_]; ii[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ii[2].imageView = g_taa.velView;     ii[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    // NEAREST. A velocity field is piecewise per surface, and now it also
    // carries a large negative sentinel in unwritten pixels - bilinear would
    // blend that sentinel into neighbouring real vectors and mark them
    // unwritten too, eating history in a halo around every sky edge.
    ii[2].sampler   = g_taa.samplerNearest;
    ii[3].imageView = g_taa.historyView[hr_]; ii[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ii[3].sampler   = g_taa.sampler;
    // Binding 4: X-Plane's gbuffer_vel when identified, our zero fallback
    // otherwise. NO BARRIER is recorded for the real image: by the time the
    // resolve runs, ssr_deferred and the lighting pass have both consumed it,
    // so X-Plane has already transitioned it to a shader-readable layout and
    // will not write it again until next frame's G-buffer pass.
    ii[4].imageView = (g_taa.flagsValid && g_taa.flagsView) ? g_taa.flagsView
                                                            : g_taa.flagsFallbackView;
    ii[4].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ii[4].sampler   = g_taa.samplerNearest;   // integer format: NEAREST only
    // Binding 5: the engine's depth when identified, the velocity target as a
    // dummy otherwise (any float array view satisfies the layout; the shader
    // never samples it unless kTaaFlagEngineDepth is set, and that flag is only
    // set when edValid). Same no-barrier reasoning as binding 4: by resolve
    // time the deferred shading, clouds and fog have all sampled this image,
    // so the engine has already moved it to a shader-readable layout.
    ii[5].imageView = g_taa.edValid ? g_taa.edView : g_taa.velView;
    ii[5].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ii[5].sampler   = g_taa.samplerNearest;   // depth: no filtering across edges
    // Bindings 7/8: engine cascades and probes, dummy-bound like binding 5.
    // The cascades were consumed by the deferred lighting long before the
    // resolve runs; the probes are sampled by it too - same layout reasoning.
    ii[6].imageView = g_taa.sunValid ? g_taa.sunView
                     : (g_taa.sunDummyView ? g_taa.sunDummyView : g_taa.velView);
    ii[6].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    // Comparison sampler, op picked live: taa.smap_ge=1 flips the compare if
    // the viz shows lit/shadowed inverted. Binding 7 is sampler2DArrayShadow
    // in the shader, so a comparison sampler here is REQUIRED, not chosen.
    ii[6].sampler   = live::onoff("taa.smap_ge", "TAA_SMAP_GE", false)
                        ? g_taa.samplerShadowGE : g_taa.samplerShadowLE;
    ii[7].imageView = g_taa.probeValid ? g_taa.probeView : g_taa.velView;
    ii[7].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ii[7].sampler   = g_taa.sampler;          // radiance: bilinear is right

    VkWriteDescriptorSet wr[8];
    memset(wr, 0, sizeof(wr));
    for (int i = 0; i < 8; ++i) {
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = set;
        wr[i].dstBinding = (uint32_t)i;
        wr[i].descriptorCount = 1;
        wr[i].pImageInfo = &ii[i];
    }
    wr[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wr[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[6].dstBinding = 7;
    wr[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[7].dstBinding = 8;
    // Binding 6: this dispatch's reproj slot. Written even when the feature is
    // off (the layout demands a valid buffer); the shader gates on the flag.
    VkDescriptorBufferInfo bi;
    bi.buffer = g_taa.uboBuf;
    bi.offset = 2048ull * setIdx;
    bi.range  = 256;
    VkWriteDescriptorSet wru;
    memset(&wru, 0, sizeof(wru));
    wru.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wru.dstSet = set;
    wru.dstBinding = 6;
    wru.descriptorCount = 1;
    wru.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    wru.pBufferInfo = &bi;
    // Binding 9: the engine's u_shadow_data region when captured, our zeroed
    // slot otherwise. Pointing at THEIR buffer is what makes this a tap
    // rather than a copy: both shaders read the same allocation, so the
    // values cannot be stale and no TRANSFER usage is ever needed.
    // THIS command buffer's region first - the frame-matched one. The
    // "latest capture" globals are the fallback, and they are the ones that
    // strobed, so they only serve when the cb has no association at all.
    VkBuffer     sdBuf = VK_NULL_HANDLE;
    VkDeviceSize sdOff = 0, sdRange = 0;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<VkCommandBuffer, MvShadowRegion>::iterator it =
            g_cbShadowRegion.find(cb);
        if (it != g_cbShadowRegion.end()) {
            sdBuf = it->second.buf; sdOff = it->second.off;
            sdRange = it->second.range;
        } else if (g_shadowDataBuf != VK_NULL_HANDLE) {
            sdBuf = g_shadowDataBuf; sdOff = g_shadowDataOff;
            sdRange = g_shadowDataRange;
        }
    }
    VkBuffer     gbBuf = VK_NULL_HANDLE;
    VkDeviceSize gbOff = 0, gbRange = 0;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<VkCommandBuffer, MvShadowRegion>::iterator it =
            g_cbGbufRegion.find(cb);
        if (it != g_cbGbufRegion.end()) {
            gbBuf = it->second.buf; gbOff = it->second.off;
            gbRange = it->second.range;
        } else if (g_gbufDataBuf != VK_NULL_HANDLE) {
            gbBuf = g_gbufDataBuf; gbOff = g_gbufDataOff;
            gbRange = g_gbufDataRange;
        }
    }
    const bool gbufTap = (gbBuf != VK_NULL_HANDLE);
    const bool sunTap = (sdBuf != VK_NULL_HANDLE) && g_taa.sunValid;
    // What did the resolve's OWN command buffer actually carry? The taps only
    // help if the engine bound these sets into the same cb the resolve records
    // into - this says so, once every ~600 dispatches.
    {
        static uint64_t rt = 0;
        if ((rt++ % 600) == 0)
            trace("RESOLVE TAP: sun=%d gbuf=%d (this cb's regions)",
                  sunTap ? 1 : 0, gbufTap ? 1 : 0);
    }
    VkDescriptorBufferInfo bs;
    bs.buffer = sunTap ? sdBuf : g_taa.uboBuf;
    bs.offset = sunTap ? sdOff : 2048ull * setIdx;
    bs.range  = sunTap ? sdRange : 2048;
    VkWriteDescriptorSet wrs = wru;
    wrs.dstBinding = 9;
    wrs.pBufferInfo = &bs;
    // Binding 10: u_gbuffer_data by reference; zeroed ring slot otherwise
    // (the shader guards on clip.w == 0 and falls back to the legacy path).
    VkDescriptorBufferInfo bg;
    bg.buffer = gbufTap ? gbBuf : g_taa.uboBuf;
    bg.offset = gbufTap ? gbOff : 2048ull * setIdx;
    bg.range  = gbufTap ? gbRange : 2048;
    VkWriteDescriptorSet wrg = wru;
    wrg.dstBinding = 10;
    wrg.pBufferInfo = &bg;
    VkDescriptorImageInfo gii;
    memset(&gii, 0, sizeof(gii));
    const VkImageView giView = gi::resultView();
    gii.sampler = g_taa.sampler;
    gii.imageView = (giView != VK_NULL_HANDLE) ? giView : g_taa.velView;
    gii.imageLayout = (giView != VK_NULL_HANDLE)
        ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet wrgi;
    memset(&wrgi, 0, sizeof(wrgi));
    wrgi.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wrgi.dstSet = set;
    wrgi.dstBinding = 11;
    wrgi.descriptorCount = 1;
    wrgi.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wrgi.pImageInfo = &gii;

    // Binding 12: gbuf-normal, or the velocity view as a legal dummy. The
    // shader only reads it under FLAG_NORMAL_TAP, so the dummy is never
    // sampled - it exists because a descriptor set must be complete.
    VkDescriptorImageInfo nii;
    memset(&nii, 0, sizeof(nii));
    const bool normTap = g_taa.normValid && g_taa.normView != VK_NULL_HANDLE;
    nii.sampler = g_taa.sampler;
    nii.imageView = normTap ? g_taa.normView : g_taa.velView;
    nii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet wrn;
    memset(&wrn, 0, sizeof(wrn));
    wrn.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wrn.dstSet = set;
    wrn.dstBinding = 12;
    wrn.descriptorCount = 1;
    wrn.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wrn.pImageInfo = &nii;

    // Ledger: these are the views a submitted command buffer will reference
    // until it drains. Recorded here, at the only place they enter a
    // descriptor set, so "last bound" means what it says.
    viewLedgerNoteBind(ii[0].imageView,  g_taa.sceneImage);
    viewLedgerNoteBind(ii[5].imageView,  VK_NULL_HANDLE);
    viewLedgerNoteBind(ii[7].imageView,  VK_NULL_HANDLE);
    viewLedgerNoteBind(nii.imageView,    VK_NULL_HANDLE);

    VkWriteDescriptorSet wrAll[13];
    for (int k = 0; k < 8; ++k) wrAll[k] = wr[k];
    wrAll[8] = wru; wrAll[9] = wrs; wrAll[10] = wrg; wrAll[11] = wrgi;
    wrAll[12] = wrn;
    dd.updateDescriptorSets(g_taa.device, 13, wrAll, 0, nullptr);

    dd.cmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_taa.pipeline);
    dd.cmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_taa.pipeLayout,
                             0, 1, &set, 0, nullptr);

    TaaPush pcv;
    pcv.invSizeX = 1.0f / (float)g_taa.w;
    pcv.invSizeY = 1.0f / (float)g_taa.h;
    pcv.jitterX = jitterX;
    pcv.sMulX = live::f("taa.smul_x", "TAA_SMUL_X",  0.5f);
    pcv.sMulY = live::f("taa.smul_y", "TAA_SMUL_Y", -0.5f);
    pcv.velMax = live::f("taa.vel_max", "TAA_VEL_MAX", 1.0f);
    // ---- THE UNWRITTEN-PIXEL REJECTION IS OFF BY DEFAULT.
    //
    // Its purpose is sound - sky and cloud pixels must not reproject - but no
    // signal available here identifies them reliably. vel==0 is ambiguous the
    // moment the camera moves by less than a pixel, coverage reads zero
    // everywhere, and the sentinel marks whole passes that a racing clear
    // erased. Every version of the test rejected most of the frame, and
    // rejecting history IS the shake:
    //     rejection on ....... 1.409
    //     rejection off ...... 0.172
    //     jitter off ......... 0.128
    //     TAA off ............ 0.107
    // Negative disables it. A real fix needs a per-pixel written flag the
    // fragment patcher owns; until then, keeping history everywhere is the
    // measurably better picture.
    pcv.novecCov = live::f("taa.novec_cov", "TAA_NOVEC_COV", -1.0f);
    // 0.5 was chosen to stop the ground crawling, but it also refuses to
    // accumulate on every pixel the sentinel calls unwritten - which is most of
    // an external frame - and that is shimmer and aliasing. Measured with the
    // detection off entirely and this at the normal alpha:
    //     TAA off ............. shimmer 0.737   jaggedness 48.8
    //     detection ON ........ shimmer 1.986   jaggedness 25.6
    //     detection OFF ....... shimmer 0.358   jaggedness 24.6
    // TAA is then steadier than no TAA and halves the aliasing. The crawl is
    // the price, and it is the lower priority of the two.
    pcv.novecAlpha = live::f("taa.novec_alpha", "TAA_NOVEC_ALPHA", 0.05f);
    pcv.jitterY = jitterY;
    pcv.alpha = taaAlpha();
    pcv.mode = taaMode();
    const bool pcReset = (reset || taaForceReset());
    pcv.edA = snapEd[0];
    pcv.edB = snapEd[1];
    pcv.viz      = taaViz();
    pcv.vizScale = taaVizScale();
    pcv.gain     = taaGain();
    pcv.varClip  = taaVarClip();
    pcv.movedDead = taaMovedDead();
    pcv.alphaMoving = taaAlphaMoving();
    pcv.alphaMovingPx = taaAlphaMovingPx();
    // Read once; the main dispatch ignores it (mode != SHARPEN) but it must be
    // defined, and the sharpen pass below reuses this same block.
    const float sharpAmt  = taaSharpen();
    const bool  doSharpen = sharpAmt > 0.0f && g_taa.sharpImage != VK_NULL_HANDLE
                            && g_taa.sharpView != VK_NULL_HANDLE;
    pcv.sharpen       = sharpAmt;
    pcv.reactiveAlpha = taaReactiveAlpha();
    pcv.aoStrength    = taaAoStrength();
    // ---- THE RADIUS IS IN PIXELS, SO IT HAS TO BE SCALED TO THE FRAME.
    //
    // taa.ao_radius is a pixel count, fed straight through until now. That
    // makes the shipped default mean a DIFFERENT amount of occlusion on every
    // display: 12 px is 1.11% of screen height at 1080p, 0.83% at 1440p and
    // 0.56% at 2160p - so a 4K user got half the AO a 1080p user got from the
    // identical setting, and contact shadows (csLen = aoRadius * 0.5) shrank
    // with it until a 6 px march at 4K reached almost nothing.
    //
    // Normalised against 1080p: the setting keeps its existing meaning there
    // and now means the same thing everywhere else. Costs nothing - the tap
    // count is fixed, only their spread changes.
    pcv.aoRadius      = taaAoRadius() *
                        (g_taa.h > 0 ? (float)g_taa.h / 1080.0f : 1.0f);
    // The sun, straight through from the plugin. Zeroed when it is below the
    // horizon, which the shader reads as "do not march".
    pcv.sunViewX      = snapSun[0];
    pcv.sunViewY      = snapSun[1];
    pcv.sunViewZ      = snapSun[2];
    pcv.csStrength    = taaCsStrength();
    // ---- ORACLE MEASUREMENT MODE: our own effects OFF while measuring.
    //
    // AO, contact shadows and sharpen all modify the output the probes and
    // any visual comparison would read; a clean measurement sees the pipeline,
    // not our decoration of it. TAA itself stays on - it IS the pipeline.
    if (oracle::armed()) {
        pcv.aoStrength = 0.0f;
        pcv.csStrength = 0.0f;
        pcv.sharpen    = 0.0f;
        pcv.flags     &= ~kTaaFlagGi;   // GI strength lives in the UBO slot
    }
    pcv.flags    = (pcReset             ? kTaaFlagReset         : 0)
                 | (cameraMoved        ? kTaaFlagCameraMoved   : 0)
                 | ((taaPosHarvest() && g_taa.edValid) ? kTaaFlagEngineDepth : 0)
                 | (reprojOn            ? kTaaFlagDepthReproject : 0)
                 | (sunTap              ? kTaaFlagSunTap        : 0)
                 | (gbufTap             ? kTaaFlagGbufTap       : 0)
                 | ((gi::resultView() != VK_NULL_HANDLE)
                                        ? kTaaFlagGi            : 0)
                 | (g_taa.probeValid    ? kTaaFlagProbeTap      : 0)
                 | (normTap             ? kTaaFlagNormalTap     : 0)
                 | (taaFreezeHistory() ? kTaaFlagFreezeHistory : 0)
                 | (taaNoMotion()      ? kTaaFlagNoMotion      : 0)
                 | (taaNoAccum()       ? kTaaFlagNoAccum       : 0)
                 | (taaReactive()      ? kTaaFlagReactive      : 0)
                 | (taaUnjitter()      ? 0 : kTaaFlagNoUnjitter)
                 | (taaCatmull()       ? kTaaFlagCatmull       : 0)
                 | (taaCrUnjitter()    ? kTaaFlagCrUnjitter    : 0)
                 | (live::onoff("taa.novec_by_vel", nullptr, false)
                        ? kTaaFlagNoVecByVel : 0)
                 | (taaDilate()        ? kTaaFlagDilate        : 0);
    pcv.velScale = taaVelScale();
    pcv.velYSign = taaVelYSign();
    pcv.flagsValid = (g_taa.flagsValid && taaObjFlags()) ? 1 : 0;
    // ---- A CHANGED KNOB IS A CHANGED HISTORY.
    //
    // Every one of these redefines what the accumulated image MEANS: a
    // different viz draws a different picture into the history buffer, a
    // different sign or scale reprojects it differently, and blending across
    // the change drags the old regime's pixels into the new one. That is not
    // hypothetical - leaving the heatmap view without a reset visibly dissolved
    // the heatmap into the scene, and it read as scene corruption for half a
    // session. Any change forces one clean frame.
    {
        static int   lastViz   = -1;
        static float lastScale = 0.0f, lastSign = 0.0f;
        if (pcv.viz != lastViz || pcv.velScale != lastScale ||
            pcv.velYSign != lastSign) {
            if (lastViz != -1) pcv.flags |= kTaaFlagReset;
            lastViz = pcv.viz; lastScale = pcv.velScale; lastSign = pcv.velYSign;
        }
    }
    dd.cmdPushConstants(cb, g_taa.pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                        0, sizeof(pcv), &pcv);

    // z = layers, matching gl_GlobalInvocationID.z in the shader.
    dd.cmdDispatch(cb, (g_taa.w + 7) / 8, (g_taa.h + 7) / 8, g_taa.layers);

    // ---- OPTIONAL SHARPEN PASS (MODE_SHARPEN) ON A DEDICATED BUFFER.
    //
    // A second dispatch of the SAME pipeline, reading the resolved history just
    // written and writing g_taa.sharpImage. The accumulation history is never
    // touched, so the sharpen cannot compound - the mistake that turned an
    // in-resolve sharpen into runaway grain. Skipped entirely when taa.sharpen
    // is 0, in which case the copy below is byte-for-byte the old path.
    if (doSharpen) {
        VkImageMemoryBarrier sp[2];
        memset(sp, 0, sizeof(sp));
        for (int i = 0; i < 2; ++i) {
            sp[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            sp[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            sp[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            sp[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            sp[i].subresourceRange.levelCount = 1;
            sp[i].subresourceRange.layerCount = g_taa.layers;
        }
        // The resolve's write to history[hw_] -> readable by this dispatch.
        sp[0].image = g_taa.history[hw_];
        sp[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        sp[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        sp[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        sp[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        // The sharpen target -> writable. UNDEFINED on first use (contents are
        // fully overwritten, so discarding them is correct); GENERAL and last
        // read by the previous frame's copy thereafter.
        sp[1].image = g_taa.sharpImage;
        sp[1].oldLayout = g_taa.sharpLayout;
        sp[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        sp[1].srcAccessMask = (g_taa.sharpLayout == VK_IMAGE_LAYOUT_UNDEFINED)
                                ? 0 : VK_ACCESS_TRANSFER_READ_BIT;
        sp[1].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                              0, nullptr, 0, nullptr, 2, sp);

        VkDescriptorSet sset = g_taa.sets[g_taa.nextSet];
        g_taa.nextSet = (g_taa.nextSet + 1) % TaaState::kSets;

        // Bindings 0/2/4 are reused from the resolve set purely to keep every
        // statically-used descriptor valid; MODE_SHARPEN reads neither. Binding
        // 3 is the resolved image (read), binding 1 the sharpen target (write).
        VkDescriptorImageInfo si[5];
        memset(si, 0, sizeof(si));
        si[0] = ii[0];
        si[1].imageView = g_taa.sharpView; si[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        si[2] = ii[2];
        si[3].imageView = g_taa.historyView[hw_];
        si[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        si[3].sampler   = g_taa.sampler;
        si[4] = ii[4];

        VkWriteDescriptorSet sw[5];
        memset(sw, 0, sizeof(sw));
        for (int i = 0; i < 5; ++i) {
            sw[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            sw[i].dstSet = sset;
            sw[i].dstBinding = (uint32_t)i;
            sw[i].descriptorCount = 1;
            sw[i].pImageInfo = &si[i];
        }
        sw[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sw[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sw[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sw[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sw[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        dd.updateDescriptorSets(g_taa.device, 5, sw, 0, nullptr);

        // Pipeline is still bound from the resolve; only the set and the push
        // constant's mode change.
        dd.cmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 g_taa.pipeLayout, 0, 1, &sset, 0, nullptr);
        TaaPush spc = pcv;
        spc.mode = 4;              // MODE_SHARPEN
        spc.sharpen = sharpAmt;
        dd.cmdPushConstants(cb, g_taa.pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                            0, sizeof(spc), &spc);
        dd.cmdDispatch(cb, (g_taa.w + 7) / 8, (g_taa.h + 7) / 8, g_taa.layers);
    }

    // ---- COPY THE RESULT INTO THE SCENE TARGET.
    //
    // Separate command, explicit ordering: everything the dispatch reads is
    // finished before anything is written back. In place would reintroduce the
    // neighbourhood race the read-only binding just removed.
    {
        // The sharpen wrote g_taa.sharpImage; copy THAT to screen and leave the
        // resolved history[hw_] untouched (it stays pristine as next frame's
        // sample source). Without the sharpen, copy history[hw_] as before.
        VkImage copySrc = doSharpen ? g_taa.sharpImage : g_taa.history[hw_];

        VkImageMemoryBarrier pre[2];
        pre[0] = bar[0];
        pre[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        pre[0].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        pre[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        pre[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        pre[1] = bar[1];
        pre[1].image = copySrc;
        pre[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        pre[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        pre[1].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        pre[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                              0, nullptr, 0, nullptr, 2, pre);

        VkImageCopy cp;
        memset(&cp, 0, sizeof(cp));
        cp.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        cp.srcSubresource.layerCount = g_taa.layers;
        cp.dstSubresource = cp.srcSubresource;
        cp.extent.width = g_taa.w; cp.extent.height = g_taa.h; cp.extent.depth = 1;
        dd.cmdCopyImage(cb, copySrc, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        g_taa.sceneImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);

        VkImageMemoryBarrier post = pre[1];
        post.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        post.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        post.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        post.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                              0, nullptr, 0, nullptr, 1, &post);

        // copySrc is GENERAL again. For the sharpen target that is the state the
        // next frame's pre-dispatch barrier expects; record it.
        if (doSharpen) g_taa.sharpLayout = VK_IMAGE_LAYOUT_GENERAL;
    }

    // The destination stage for this transition back is COLOR_ATTACHMENT_OUTPUT
    // (see the call below), and SHADER_READ is not an access that occurs there
    // - VUID-vkCmdPipelineBarrier-pImageMemoryBarriers-02820, twenty a run. The
    // shader-read half was silently dropped exactly as the source half was at
    // the barrier above. Naming FRAGMENT_SHADER alongside the attachment stage
    // in the call makes both halves real; the mask itself is right.
    bar[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    bar[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    bar[2].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bar[2].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    bar[2].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bar[2].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                          0, nullptr, 0, nullptr, 1, &bar[0]);
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                          0, nullptr, 0, nullptr, 1, &bar[2]);

    // ---- COPY A STRIP OF HISTORY OUT, FOR THE CPU TO COMPARE.
    //
    // Recorded after the dispatch and after the copy-back, so it captures the
    // history this frame actually produced. No fence: the read happens at
    // present, at least one frame later, by which point the submission that
    // recorded this has long retired. A stale read would understate the
    // difference, never invent one, so the metric is conservative in the
    // direction that matters.
    if (g_taa.readPtr && g_taa.readBuf && dd.cmdCopyImageToBuffer) {
        VkImageMemoryBarrier rb;
        memset(&rb, 0, sizeof(rb));
        rb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        rb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        rb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        rb.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        rb.subresourceRange.levelCount = 1;
        rb.subresourceRange.layerCount = g_taa.layers;
        // ---- THE IMAGE THAT WAS ACTUALLY DELIVERED, NOT THE HISTORY.
        //
        // This probe compares "what we sent" against "what is in the scene
        // target" and reports a mismatch as delivery failing. It read
        // history[hw_] - but with sharpening on, the copy-back sends
        // g_taa.sharpImage instead, exactly as the note at copySrc says it
        // does. So the comparison was unsharpened-history against a
        // sharpened-scene, and the two differ BY THE SHARPENING.
        //
        // Sharpening is on by default, so this instrument has been reporting a
        // delivery failure that never happened - measured at 1839/2048 halves
        // differing, mean delta 1113. It cost real debugging time: the message
        // states outright that a difference means the copy is not reaching the
        // display, and that conclusion was false every time it was read.
        //
        // Same expression as copySrc so the two cannot drift apart again.
        const VkImage probeSrc = doSharpen ? g_taa.sharpImage
                                           : g_taa.history[hw_];
        rb.image = probeSrc;
        rb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        rb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        rb.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        rb.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                              0, nullptr, 0, nullptr, 1, &rb);
        VkBufferImageCopy bic;
        memset(&bic, 0, sizeof(bic));
        bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        bic.imageSubresource.layerCount = 1;
        // A horizontal strip across the middle of the frame: guaranteed to
        // cross real scene content at any camera angle.
        bic.imageOffset.x = 0;
        bic.imageOffset.y = (int32_t)(g_taa.h / 2);
        bic.imageExtent.width  = g_taa.w < 512 ? g_taa.w : 512;
        bic.imageExtent.height = 1;
        bic.imageExtent.depth  = 1;
        dd.cmdCopyImageToBuffer(cb, probeSrc,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                g_taa.readBuf, 1, &bic);
        // ---- AND THE SAME STRIP OF THE SCENE TARGET, RIGHT AFTER THE COPY.
        //
        // History accumulates correctly (delta 6.3 at 98% history weight) while
        // the composited image stays ~6x less stable than TAA off. Everything
        // between those two facts is delivery, and this settles it in one run:
        // if the scene strip matches the history strip, the copy landed and
        // something downstream repaints; if it differs, the copy is not
        // reaching the image the display reads from - which is what a
        // double-buffered scene target would do.
        //
        // Into the second half of the same buffer, so one map serves both.
        if (g_taa.sceneImage) {
            VkImageMemoryBarrier sb = rb;
            sb.image = g_taa.sceneImage;
            sb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            sb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            sb.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            sb.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT |
                                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                  0, nullptr, 0, nullptr, 1, &sb);
            VkBufferImageCopy sic = bic;
            sic.bufferOffset = 2048ull * sizeof(uint16_t);
            dd.cmdCopyImageToBuffer(cb, g_taa.sceneImage,
                                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    g_taa.readBuf, 1, &sic);
            VkImageMemoryBarrier sbk = sb;
            sbk.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            sbk.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            sbk.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            sbk.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                                  0, nullptr, 0, nullptr, 1, &sbk);
        }
        VkImageMemoryBarrier rbk = rb;
        rbk.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        rbk.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        rbk.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        rbk.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                              0, nullptr, 0, nullptr, 1, &rbk);
    }

    // ---- ONE FLIP PER PRESENTED FRAME, NOT ONE PER RESOLVE.
    //
    // This flipped unconditionally, and the resolve can run several times in a
    // single present (X-Plane records ahead; taa.max_resolves bounds it). With
    // an EVEN number of resolves in a frame the index returns to where it
    // started, so the same buffer is written every time and the other - the one
    // every resolve READS - is never updated again. History freezes, and with
    // it every property that depends on accumulation.
    //
    // Measured, with a static camera and 98% history weight, which cannot
    // change the image at all if history is real:
    //     TAA off                  frame-to-frame diff 0.149
    //     TAA on                                       4.35
    //     TAA on, alpha 0.02                           3.29
    //     TAA on, alpha 0.02, jitter off               0.229
    // TAA made a still image 29x LESS stable than no TAA. That is this.
    //
    // It also explains the shape of the whole hunt: without jitter consecutive
    // frames are nearly identical, so a frozen history is invisible; with
    // jitter every frame differs by a sub-pixel offset, the output becomes the
    // raw jittered frame, and that is the shake. Gain, varclip, the reactive
    // mask, both sign flips and the target latch all tune how history is
    // BLENDED, which is why none of them moved a symptom caused by there being
    // no history to blend.
    //
    // The arming flag is owned by the present hook, so the pairing follows
    // DISPLAYED frames however many times the resolve records.
    // ---- FLIP PER RECORDED FRAME, WHICH IS PER RESOLVE.
    //
    // This was briefly tied to PRESENT, on the reasoning that the ping-pong
    // should follow displayed frames. That is wrong here: X-Plane records
    // ahead, so several frames' command buffers are built before any present.
    // Arming at present meant the first recorded frame flipped and every
    // further frame recorded before that present did not - consecutive frames
    // wrote the SAME buffer, so the buffer being read never advanced and the
    // blend had nothing to accumulate against.
    //
    // Measured with the direct history readback, at alpha 0.02 (98% history),
    // vel_scale 0 (lookup pinned to the same pixel) and reset 0 - conditions
    // under which the output mathematically cannot change if history is real:
    //     1532 of 2048 halves differed every frame, mean |delta| 193.
    // The blend cannot produce that; only a history read that is not the
    // previous write can.
    //
    // One resolve per recorded frame (taa.max_resolves=1) makes per-resolve and
    // per-frame the same thing, which is the pairing the algorithm needs.
    g_taa.historyWrite ^= 1u;

    // The oracle's content probe rides the same command buffer, after the
    // resolve - scene complete, engine targets in their read layouts.
    oracle::record(dd, g_taa.device, cb);
    // Numeric dump of the two captured engine blocks - the instrument that
    // replaced testing sun-tap plumbing on the user's eyes.
    if (sunTap && gbufTap)
        oracle::sunDump(dd, g_taa.device, cb, sdBuf, sdOff, sdRange,
                        gbBuf, gbOff, gbRange);

    if ((++g_taa.dispatches % 600) == 1)
        trace("TAA: dispatch %llu - mode %d alpha %.3f reset %d (%ux%u x%u)",
              (unsigned long long)g_taa.dispatches, taaMode(), taaAlpha(),
              (pcv.flags & kTaaFlagReset) ? 1 : 0,
              g_taa.w, g_taa.h, g_taa.layers);
}

// The IBackend entry point. Thin on purpose: everything above it is the
// contract, everything below is this backend's own business, and the only job
// here is to translate one into the other.
inline bool TaaBackend::record(VkCommandBuffer cb, const temporal::TemporalFrame &f)
{
    if (!g_taaDevice) return false;
    taaRecordResolve(*g_taaDevice, cb, f.jitter.x, f.jitter.y,
                     f.reset != temporal::RESET_NONE, f.camera.moved);
    return true;
}
