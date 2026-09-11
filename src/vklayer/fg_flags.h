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
