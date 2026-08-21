// Which upscaler backend runs, and - when it is not the one that was asked
// for - why not.
//
// WHY THIS IS A SEPARATE HEADER, AND WHY IT TOUCHES NO VULKAN
//
// The decision "can this machine run FSR 4" is arithmetic over a handful of
// facts: a PCI vendor id, an extension being present, whether a runtime was
// compiled in. None of it needs a device, a command buffer, or a running
// X-Plane - so none of it should need one to TEST. Everything here compiles
// into tests.ps1 and runs at a command line in milliseconds.
//
// The backends themselves live in backends.h, which does include Vulkan. The
// split is deliberate: policy is the part that is easy to get quietly wrong,
// so policy is the part that gets tests.
//
// THE RULE THIS ENCODES
//
// Report the HARDWARE verdict before the LIBRARY verdict, and never collapse
// the two. "FSR 4 needs an RDNA 4 card" and "FSR 4 needs an SDK this build was
// not compiled against" are different problems with different fixes - one is
// buy-different-hardware, the other is a build flag. A UI that shows a single
// "unavailable" for both sends the user to solve the wrong one.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../share.h"

namespace upscaler {

// PCI vendor ids. These are the values Vulkan reports in
// VkPhysicalDeviceProperties::vendorID, not something we choose.
enum {
    VENDOR_ID_AMD    = 0x1002,
    VENDOR_ID_NVIDIA = 0x10DE,
    VENDOR_ID_INTEL  = 0x8086
};

// The facts a backend decision is allowed to depend on.
//
// Every field is QUERIED, never inferred from another. The temptation is to
// derive "is RDNA 4" from a device name string; device names are marketing
// text, they change, and they are localised. Cooperative matrix support is a
// capability the driver answers for directly, which is why it is the field
// here rather than a parsed name.
struct DeviceCaps {
    uint32_t vendorId  = 0;
    bool     coopMatrix = false;   // VK_KHR_cooperative_matrix
    bool     valid      = false;   // false until a device has actually answered
};

// ---- TWO SEPARATE FACTS, AND THEY ARE NOT THE SAME FACT.
//
// MV_HAVE_*_SDK   the build found the vendor's headers and import library
// MV_BACKEND_*    this layer contains an implementation that uses them
//
// Collapsing these into one flag is a mistake this file made for exactly one
// build. The moment SDK detection was wired up, three SDKs were found, the
// single flag went to 1, and the availability report would have told the panel
// that FSR 2, DLSS and XeSS were all ready - while no record() existed for any
// of them. The user would have selected XeSS and got silence.
//
// Having the headers is not having the code. The SDK half is answered by the
// build system, because a header either is on disk or is not. The backend half
// is answered here, because it is a statement about this source tree.
#ifndef MV_HAVE_FSR2_SDK
#define MV_HAVE_FSR2_SDK 0
#endif
#ifndef MV_HAVE_FSR4_SDK
#define MV_HAVE_FSR4_SDK 0
#endif
#ifndef MV_HAVE_DLSS_SDK
#define MV_HAVE_DLSS_SDK 0
#endif
#ifndef MV_HAVE_XESS_SDK
#define MV_HAVE_XESS_SDK 0
#endif

// Flip one of these to 1 in the same commit that adds its record(). Not before.
#define MV_BACKEND_FSR2 0
#define MV_BACKEND_FSR4 0
#define MV_BACKEND_DLSS 0
#define MV_BACKEND_XESS 0

// ---- HARDWARE VERDICT.
//
// Answers only "could this silicon run it", ignoring whether we have the code.
// Split out so the two reasons stay distinguishable all the way to the UI.
inline bool hardwareCanRun(int upscaler, const DeviceCaps &c)
{
    switch (upscaler) {
    case TAA_UPSCALER_OFF:
    case TAA_UPSCALER_TAA:
        return true;                 // ours, plain compute, runs anywhere

    case TAA_UPSCALER_FSR2:
        // FSR 2 is vendor-agnostic by design - it is a compute shader, and it
        // runs on NVIDIA and Intel parts as happily as on AMD. Gating it on
        // vendor would be inventing a restriction AMD does not impose.
        return true;

    case TAA_UPSCALER_FSR4:
        // RDNA 4 only, because the model runs on the matrix cores. Expressed
        // as AMD + cooperative matrix rather than a device-name match: it is
        // the capability the shader actually needs, and it is a query.
        return c.vendorId == VENDOR_ID_AMD && c.coopMatrix;

    case TAA_UPSCALER_DLSS:
        // NGX runs on NVIDIA only. Turing-or-later is a further restriction
        // that NGX itself answers for at init; we do not guess it here.
        return c.vendorId == VENDOR_ID_NVIDIA;

    case TAA_UPSCALER_XESS:
        // Cross-vendor, and deliberately not gated on Intel. XeSS takes the
        // XMX path on Arc and a DP4a path everywhere else, so gating it on
        // vendorId would refuse it on the majority of cards that can run it.
        //
        // The real floor is DP4a, which is a shader capability rather than a
        // vendor - and the runtime answers for it at init through its own
        // query. Guessing it here from a vendor id is exactly the inference
        // this file exists to avoid.
        return true;
    }
    return false;
}

// ---- LIBRARY VERDICT: is the vendor SDK on disk for this build.
inline bool sdkPresent(int upscaler)
{
    switch (upscaler) {
    case TAA_UPSCALER_OFF:
    case TAA_UPSCALER_TAA:  return true;
    case TAA_UPSCALER_FSR2: return MV_HAVE_FSR2_SDK != 0;
    case TAA_UPSCALER_FSR4: return MV_HAVE_FSR4_SDK != 0;
    case TAA_UPSCALER_DLSS: return MV_HAVE_DLSS_SDK != 0;
    case TAA_UPSCALER_XESS: return MV_HAVE_XESS_SDK != 0;
    }
    return false;
}

// ---- IMPLEMENTATION VERDICT: does a record() exist here at all.
inline bool backendImplemented(int upscaler)
{
    switch (upscaler) {
    case TAA_UPSCALER_OFF:
    case TAA_UPSCALER_TAA:  return true;
    case TAA_UPSCALER_FSR2: return MV_BACKEND_FSR2 != 0;
    case TAA_UPSCALER_FSR4: return MV_BACKEND_FSR4 != 0;
    case TAA_UPSCALER_DLSS: return MV_BACKEND_DLSS != 0;
    case TAA_UPSCALER_XESS: return MV_BACKEND_XESS != 0;
    }
    return false;
}

// Both halves. Kept for callers that only care whether it can run.
inline bool libraryPresent(int upscaler)
{
    return sdkPresent(upscaler) && backendImplemented(upscaler);
}

// ---- THE REPORTED ANSWER.
//
// UNKNOWN is reserved for "no device has answered yet" and is never a verdict.
// Returning OK for a backend on a device we have not seen would put an option
// in the UI on the strength of a zeroed struct.
inline int availability(int upscaler, const DeviceCaps &c)
{
    if (upscaler < 0 || upscaler >= TAA_UPSCALER_COUNT)
        return TAA_AVAIL_NO_SUPPORT;

    // Ours needs no device query, so it can answer before one arrives.
    if (upscaler == TAA_UPSCALER_OFF || upscaler == TAA_UPSCALER_TAA)
        return TAA_AVAIL_OK;

    if (!c.valid)                     return TAA_AVAIL_UNKNOWN;
    if (!hardwareCanRun(upscaler, c)) return TAA_AVAIL_NO_GPU;
    // Order matters. NO_SUPPORT means "this layer has no code for it", which is
    // a developer's problem; NO_LIBRARY means "install the runtime", which is
    // the user's. Reporting the missing SDK first on a build that also has no
    // backend would send the user to download something that still would not
    // work.
    if (!backendImplemented(upscaler)) return TAA_AVAIL_NO_SUPPORT;
    if (!sdkPresent(upscaler))         return TAA_AVAIL_NO_LIBRARY;
    return TAA_AVAIL_OK;
}

// Fills the whole array the plugin reads. One call, so a backend cannot be
// added to the enum and forgotten here.
inline void availabilityAll(const DeviceCaps &c, int32_t *out)
{
    for (int i = 0; i < TAA_UPSCALER_COUNT; ++i)
        out[i] = (int32_t)availability(i, c);
}

// ---- RESOLUTION.
//
// Maps a REQUEST to what will actually run. The fallback is TAA rather than
// OFF: a user who asked for upscaling wants temporal accumulation more than
// they want nothing, and dropping to OFF would look like the layer had
// detached.
//
// why is always set, including on success, so a caller cannot log an
// uninitialised pointer - which is the shape of bug that made an earlier
// decline message print whatever was last on the stack.
inline int resolve(int requested, const DeviceCaps &c, const char **why)
{
    static const char *kOk = "";

    if (requested < 0 || requested >= TAA_UPSCALER_COUNT) {
        if (why) *why = "not a backend this build knows about";
        return TAA_UPSCALER_TAA;
    }

    const int a = availability(requested, c);
    if (a == TAA_AVAIL_OK) {
        if (why) *why = kOk;
        return requested;
    }

    if (why) {
        switch (a) {
        case TAA_AVAIL_UNKNOWN:
            *why = "no device has answered yet"; break;
        case TAA_AVAIL_NO_GPU:
            *why = "this GPU cannot run it"; break;
        case TAA_AVAIL_NO_LIBRARY:
            *why = "this build carries no runtime for it"; break;
        default:
            *why = "not built into this layer yet"; break;
        }
    }
    return TAA_UPSCALER_TAA;
}

// Human-readable, for traces and the UI. Kept beside the enum so the two
// cannot drift.
inline const char *name(int upscaler)
{
    switch (upscaler) {
    case TAA_UPSCALER_OFF:  return "Off";
    case TAA_UPSCALER_TAA:  return "TAA";
    case TAA_UPSCALER_FSR2: return "FSR 2";
    case TAA_UPSCALER_FSR4: return "FSR 4";
    case TAA_UPSCALER_DLSS: return "DLSS";
    case TAA_UPSCALER_XESS: return "XeSS";
    }
    return "?";
}

inline const char *availabilityName(int a)
{
    switch (a) {
    case TAA_AVAIL_UNKNOWN:    return "unknown";
    case TAA_AVAIL_OK:         return "ok";
    case TAA_AVAIL_NO_GPU:     return "no-gpu";
    case TAA_AVAIL_NO_LIBRARY: return "no-library";
    case TAA_AVAIL_NO_SUPPORT: return "no-support";
    }
    return "?";
}

} // namespace upscaler
