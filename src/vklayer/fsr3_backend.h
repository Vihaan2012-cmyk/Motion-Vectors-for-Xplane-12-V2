// FSR 3 upscaling, driven from this layer into X-Plane's own upscale slot.
//
// ---- WHERE THIS SITS.
//
// X-Plane renders the 3-D scene below display resolution when its own FSR is
// enabled - measured at 2953x1661 against a 3840x2160 display - and then
// dispatches a spatial upscale. This layer substitutes that shader, so the
// upscale step is ours. FSR 3 replaces what runs in it.
//
// The chain is then:
//
//     X-Plane renders      2953x1661   its own supported sub-native path
//     our TAA resolves     2953x1661   temporal, on the SMALL image
//     FSR 3 upscales    -> 3840x2160   into X-Plane's own output image
//
// Nothing about X-Plane's rendering is redirected. It renders where it always
// did and reads the image it always read; only the code between changed.
//
// ---- WHY THE OUTPUT IMAGE HANDLE WAS THE HARD PART.
//
// FSR 3 is ten CPU-driven passes and must be HANDED the image to write. X-Plane
// exposes no descriptor contents to a layer - one descriptor set recorded per
// frame, and every push-descriptor entry point called zero times - so the
// handle cannot be obtained by inspection. It comes from fsr_probe.h, which
// stamps a sentinel from the substituted shader and finds which image carries
// it.
//
// ---- THE LOADER IS NOT RE-ENTERED.
//
// ffx_vk.cpp calls Vulkan entry points itself. Inside a layer that would mean
// re-entering the loader from within one of its own calls - one of the two
// suspects never separated in the XeSS probe crash. VkDeviceContext takes a
// PFN_vkGetDeviceProcAddr, so FFX is given the NEXT LAYER's address table and
// resolves everything below us instead of through the loader's trampoline.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <string.h>
#include <mutex>

#include <FidelityFX/host/ffx_fsr3upscaler.h>
#include <FidelityFX/host/backends/vk/ffx_vk.h>

namespace fsr3 {

struct State {
    bool                        ready   = false;
    bool                        failed  = false;
    FfxInterface                iface;
    FfxFsr3UpscalerContext      ctx;
    // ---- THIS MUST OUTLIVE ensure(). IT IS HELD BY POINTER.
    //
    // ffxGetDeviceVK does not copy the VkDeviceContext - it casts the pointer,
    // and CreateBackendContextVK reads it straight back out of
    // backendInterface->device. A local would be a dangling stack pointer for
    // every later call, the dispatch included.
    VkDeviceContext             vkCtx;
    void                       *scratch = nullptr;
    size_t                      scratchSize = 0;
    uint32_t                    renderW = 0, renderH = 0;
    uint32_t                    outW = 0,   outH = 0;
    uint64_t                    dispatches = 0;
    // The shared resources start UNDEFINED and are transitioned once.
    bool                        sharedReady = false;

    // ---- A SCRATCH OUTPUT, FOR ISOLATION.
    //
    // FSR3 dispatches six times, every call FFX_OK, validation clean, and then
    // the GPU dies - deterministically, and independently of FFX's queue depth.
    // Two possibilities remain: writing X-PLANE'S output image is what kills
    // it, or FSR3's own internal work does.
    //
    // Pointing the output at an image of ours separates them. Nothing reaches
    // the screen in this mode; that is the point.
    VkImage                     ownOut = VK_NULL_HANDLE;
    VkDeviceMemory              ownOutMem = VK_NULL_HANDLE;

    // ---- FSR3 IS NOT THREAD SAFE, AND THIS LAYER IS MULTI-THREADED.
    //
    // ensure() runs from the present path; dispatch() runs on whichever thread
    // records X-Plane's upscale. An FFX context used from two threads at once
    // is undefined, and the symptom fits exactly: it crashes normally, and
    // SURVIVES under the validation layer, which serialises enough to hide it.
    //
    // Its own mutex, never g_lock: g_lock is held across large parts of this
    // layer and taking it here is how the earlier deadlock happened.
    std::mutex                  lock;
    // ---- FSR3'S OWN DEBUG VIEW.
    //
    // FSR3 ships a pass that draws its internal view of its inputs into the
    // output. With the temporal half already ruled out by reset=1, this says
    // WHICH current-frame input is wrong instead of guessing at one at a time.
    bool                        debugView = false;

    // ---- RCAS SHARPNESS, 0 = OFF.
    //
    // Temporal accumulation resamples history every frame and upscaling
    // resamples space; both cost high-frequency detail. FSR3 ships RCAS to put
    // it back, and we were dispatching with enableSharpening=false - carrying
    // the softness while the correction sat unused in the same pipeline. Live,
    // so it can be dialled against a real scene instead of guessed at across
    // rebuilds.
    float                       sharpness = 0.0f;



    // The three resources FSR3 emits for downstream effects. Required by the
    // dispatch even when nothing downstream consumes them.
    VkImage                     shared[3] = { VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDeviceMemory              sharedMem[3] = { VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE };
    FfxResourceDescription      sharedDesc[3];
};

inline State &state()
{
    static State s;
    return s;
}

// FFX talks about resource state; we know the Vulkan layout our own images are
// in, so the mapping is stated once here rather than at each call site.
inline FfxResourceDescription describeTex2D(uint32_t w, uint32_t h,
                                            FfxSurfaceFormat fmt,
                                            FfxResourceUsage usage)
{
    FfxResourceDescription d;
    memset(&d, 0, sizeof(d));
    d.type     = FFX_RESOURCE_TYPE_TEXTURE2D;
    d.format   = fmt;
    d.width    = w;
    d.height   = h;
    d.depth    = 1;
    d.mipCount = 1;
    d.flags    = FFX_RESOURCE_FLAGS_NONE;
    d.usage    = usage;
    return d;
}

} // namespace fsr3
