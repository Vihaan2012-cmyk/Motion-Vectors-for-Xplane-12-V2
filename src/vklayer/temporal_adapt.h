// ============================================================================
//  temporal_adapt.h - THE ONLY PLACE A BACKEND'S CONVENTIONS ARE APPLIED.
// ============================================================================
//
// tcore::Frame carries raw engine values and converts for nobody. This file is
// the layer in between: one function per backend, each stating what that
// backend wants and why, so a convention lives in exactly one named place.
//
// The rule the rest of the layer follows: a backend never reads a layer global
// and never re-derives a unit. It receives one of these structs and uses the
// numbers in it. If a backend needs something that is not here, the fix is to
// add it to tcore::Frame and convert it here - not to reach around.
//
// Why that rule is worth the indirection: the two backends below want motion
// vectors in the SAME unit (render pixels) and the SAME direction (current ->
// previous), and still disagree about the jitter sign. That disagreement is
// invisible at a call site and obvious here.
// ============================================================================
#pragma once

#include "temporal_core.h"

namespace tadapt {

// ---------------------------------------------------------------- FSR 3 -----
//
// From ffx_fsr3upscaler.h, FfxFsr3UpscalerDispatchDescription:
//
//   motionVectors      sampled, then multiplied by motionVectorScale to give
//                      RENDER PIXELS, pointing current -> previous.
//   motionVectorScale  therefore carries BOTH the unit conversion and the sign.
//   jitterOffset       in RENDER PIXELS, the offset applied to the projection.
//   frameTimeDelta     MILLISECONDS. Not seconds - the field name does not say
//                      so and passing seconds yields a plausible, wrong result.
//   cameraNear/Far     with the infinite-far convention signalled by passing
//                      FLT_MAX for far, which is why depthInfiniteFar is a flag
//                      on the frame rather than an inference from the value.
//   reset              discard history entirely this frame.
struct Fsr3Inputs {
    float mvScaleX = 0.0f, mvScaleY = 0.0f;   // multiply sampled MV -> pixels
    float jitterX  = 0.0f, jitterY  = 0.0f;   // pixels
    float nearPlane = 0.0f, farPlane = 0.0f;
    float fovYRad   = 0.0f;
    float frameTimeMs = 16.6f;
    float preExposure = 1.0f;
    bool  reset = false;
    bool  depthInverted = false;
    bool  depthInfinite = false;
    uint32_t renderW = 0, renderH = 0, outW = 0, outH = 0;
};

inline bool toFsr3(const tcore::Frame &f, Fsr3Inputs &o, const char **whyNot = 0)
{
    if (!tcore::usable(f, whyNot)) return false;

    // UV -> pixels is a multiply by the render size. The Y sign rides along in
    // the scale because FSR3 offers no separate hook for it; that is the whole
    // reason mvYSign is a field on the frame instead of being folded into the
    // stored vectors, where it could never be corrected without a re-render.
    const float unit = (f.mvUnit == tcore::kMvUv) ? 1.0f
                     : (f.mvUnit == tcore::kMvNdc) ? 0.5f : (1.0f / (float)(f.renderW ? f.renderW : 1));
    o.mvScaleX = f.mvScale * unit * (float)f.renderW;
    o.mvScaleY = f.mvScale * unit * (float)f.renderH * f.mvYSign;

    // FSR3 wants current -> previous. If a frame ever stores the other
    // direction, negate rather than silently hand it a mirrored field.
    if (f.mvDir == tcore::kMvToCurrent) { o.mvScaleX = -o.mvScaleX;
                                          o.mvScaleY = -o.mvScaleY; }

    // NDC spans 2 across the render extent, so NDC -> pixels is size/2.
    o.jitterX = f.jitterNdcX * 0.5f * (float)f.renderW;
    o.jitterY = f.jitterNdcY * 0.5f * (float)f.renderH;

    o.nearPlane     = f.nearPlane;
    o.farPlane      = f.farPlane;
    o.fovYRad       = f.fovYRad;
    o.frameTimeMs   = f.frameTimeMs;
    o.preExposure   = f.exposure;
    o.reset         = f.cameraCut;
    o.depthInverted = f.depthReversed;
    o.depthInfinite = f.depthInfiniteFar;
    o.renderW = f.renderW; o.renderH = f.renderH;
    o.outW    = f.outW;    o.outH    = f.outH;
    return true;
}

// ----------------------------------------------------------------- DLSS -----
//
// From nvsdk_ngx_helpers_vk.h, NVSDK_NGX_VK_DLSS_Eval_Params:
//
//   pInMotionVectors   render-resolution pixels, current -> previous, unless
//                      NVSDK_NGX_DLSS_Feature_Flags_MVLowRes is cleared.
//   InMVScaleX/Y       multiplies the sampled vector, exactly like FSR3's
//                      motionVectorScale. Same unit, same direction.
//   InJitterOffsetX/Y  pixels - and this is where the two disagree.
//   InExposureScale    or a 1x1 exposure texture; DLSS is more sensitive to
//                      this than FSR3 and a wrong value reads as the image
//                      breathing rather than as an obvious fault.
//   InReset            history discard.
//
// ---- THE JITTER SIGN. READ THIS BEFORE CHANGING IT.
//
// FSR3 wants the offset that WAS APPLIED to the projection. NVIDIA's own
// samples pass the NEGATED offset, and integrations disagree in public about
// which is correct because both produce a stable image while only one produces
// a sharp one - a wrong sign here costs sub-pixel accuracy, not visible
// breakage, so it survives casual testing.
//
// This cannot be settled by reading a header, and it has never been run here.
// So it is a FLAG with a default and a stated reason, not a silent constant:
// jitterNegated defaults true to match NVIDIA's samples, and dlss.jitter_sign
// flips it live. Whoever first runs this should A/B it on a high-contrast
// static edge - the sharper one is right.
struct DlssInputs {
    float mvScaleX = 0.0f, mvScaleY = 0.0f;
    float jitterX  = 0.0f, jitterY  = 0.0f;
    float exposureScale = 1.0f;
    float sharpness = 0.0f;
    bool  reset = false;
    bool  depthInverted = false;
    bool  autoExposure = true;
    uint32_t renderW = 0, renderH = 0, outW = 0, outH = 0;
};

inline bool toDlss(const tcore::Frame &f, DlssInputs &o,
                   bool jitterNegated = true, const char **whyNot = 0)
{
    if (!tcore::usable(f, whyNot)) return false;

    // Identical to the FSR3 path on purpose: same unit, same direction, so if
    // one is right and the other is not, the difference is visible as a diff
    // of these two functions rather than buried in two call sites.
    const float unit = (f.mvUnit == tcore::kMvUv) ? 1.0f
                     : (f.mvUnit == tcore::kMvNdc) ? 0.5f : (1.0f / (float)(f.renderW ? f.renderW : 1));
    o.mvScaleX = f.mvScale * unit * (float)f.renderW;
    o.mvScaleY = f.mvScale * unit * (float)f.renderH * f.mvYSign;
    if (f.mvDir == tcore::kMvToCurrent) { o.mvScaleX = -o.mvScaleX;
                                          o.mvScaleY = -o.mvScaleY; }

    const float s = jitterNegated ? -1.0f : 1.0f;
    o.jitterX = s * f.jitterNdcX * 0.5f * (float)f.renderW;
    o.jitterY = s * f.jitterNdcY * 0.5f * (float)f.renderH;

    o.exposureScale = f.exposure;
    o.reset         = f.cameraCut;
    o.depthInverted = f.depthReversed;
    o.renderW = f.renderW; o.renderH = f.renderH;
    o.outW    = f.outW;    o.outH    = f.outH;
    return true;
}

// ---- A DIFF, BECAUSE THE TWO SHOULD AGREE EXCEPT WHERE THEY MUST NOT -------
//
// Both adapters derive motion vector scale identically. If they ever diverge
// the cause is an edit to one and not the other, which is precisely the class
// of bug this file exists to prevent - so it can be asserted rather than hoped
// for. The jitter is expected to differ in SIGN and nothing else.
inline bool adaptersAgree(const tcore::Frame &f, const char **whyNot = 0)
{
    Fsr3Inputs a; DlssInputs b;
    if (!toFsr3(f, a, whyNot)) return false;
    if (!toDlss(f, b, false, whyNot)) return false;   // sign off, compare rest
    const char *why = 0;
    if (a.mvScaleX != b.mvScaleX || a.mvScaleY != b.mvScaleY)
        why = "motion vector scale differs between the FSR3 and DLSS adapters";
    else if (a.jitterX != b.jitterX || a.jitterY != b.jitterY)
        why = "jitter magnitude differs with the DLSS sign flag disabled";
    if (whyNot) *whyNot = why;
    return why == 0;
}

} // namespace tadapt
