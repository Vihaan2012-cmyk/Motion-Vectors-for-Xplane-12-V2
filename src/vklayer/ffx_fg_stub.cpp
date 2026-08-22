// The one frame-generation symbol the FSR3 UPSCALER drags in, stubbed.
//
// ---- WHY A STUB AND NOT THE REAL FILE.
//
// ffx_vk.cpp references ffxSetFrameGenerationConfigToSwapchainVK, which lives
// in FrameInterpolationSwapchainVK.cpp. That file does not compile under GCC -
// not for want of an include, but because it relies on MSVC extensions:
//
//   error: cannot bind non-const lvalue reference of type 'SubmissionSemaphores&'
//          to an rvalue of type 'SubmissionSemaphores'
//   error: invalid 'static_cast' from type 'void*' to type 'FfxWaitCallbackFunc'
//
// MSVC accepts both; ISO C++ does not, and GCC is right. Making that file build
// means editing vendored AMD source, which is a decision to take deliberately
// when frame generation is actually being implemented - not incidentally, while
// getting the upscaler to link.
//
// The upscaler never calls this. It is referenced from a dispatch table in
// ffx_vk.cpp, so the linker demands the symbol exist whether or not any code
// path reaches it.
//
// ---- IT RETURNS AN ERROR RATHER THAN PRETENDING TO SUCCEED.
//
// If frame generation is ever wired up before this stub is replaced, a silent
// success would configure nothing and produce no generated frames, with
// everything reporting healthy. FFX_ERROR_BACKEND_API_ERROR says plainly that
// the call went nowhere.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <FidelityFX/host/ffx_types.h>
#include <FidelityFX/host/ffx_error.h>

struct FfxFrameGenerationConfig;

extern "C" FfxErrorCode ffxSetFrameGenerationConfigToSwapchainVK(
    FfxFrameGenerationConfig const *config)
{
    (void)config;
    // Frame generation is not implemented. See the note above: the real
    // implementation is FrameInterpolationSwapchainVK.cpp, which needs source
    // fixes to build outside MSVC.
    return FFX_ERROR_BACKEND_API_ERROR;
}
