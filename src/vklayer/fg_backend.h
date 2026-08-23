// Frame generation: optical flow + frame interpolation, driven from the
// FidelityFX frame-interpolation swapchain.
//
// ---- WHY THIS IS NOT THE COMBINED FfxFsr3Context.
//
// The obvious route is ffxFsr3ContextCreate, which builds upscaler, optical
// flow and interpolation together and exposes ffxFsr3ConfigureFrameGeneration.
// It is also a rewrite: this layer already runs ffxFsr3UpscalerContextCreate,
// and that upscaler is measured, tuned and working - bind ordering, fp16 off,
// the luma-history format fix, three layout barriers. Replacing it to gain
// frame generation would put all of that back in question at the same time as
// adding a feature.
//
// FfxFrameGenerationConfig takes a frameGenerationCallback: the swapchain asks
// US to produce the interpolated frame. So the interpolation contexts can live
// here, beside the upscaler rather than instead of it, and the upscaler is not
// touched at all.
//
// ---- THE HANDOFF ALREADY EXISTS.
//
// FfxFrameInterpolationDispatchDescription wants dilatedDepth,
// dilatedMotionVectors and reconstructedPrevDepth. Those are exactly the three
// shared resources the upscaler already emits every frame - fsr3_backend_impl.h
// creates them and its own comment calls them "the three resources FSR3 emits
// for downstream effects. Required by the dispatch even when nothing downstream
// consumes them." This is that downstream consumer. Nothing new has to be
// produced; the existing outputs are simply read.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <string.h>
#include <mutex>

#include <FidelityFX/host/ffx_frameinterpolation.h>
#include <FidelityFX/host/ffx_opticalflow.h>
#include <FidelityFX/host/backends/vk/ffx_vk.h>

namespace fg {

struct State {
    bool ready  = false;
    bool failed = false;

    FfxInterface                iface;
    FfxOpticalflowContext       ofCtx;
    FfxFrameInterpolationContext fiCtx;
    VkDeviceContext             vkCtx;
    void                       *scratch     = nullptr;
    size_t                      scratchSize = 0;

    uint32_t renderW = 0, renderH = 0;
    uint32_t dispW   = 0, dispH   = 0;

    // Optical flow's own outputs, allocated from its shared-resource
    // descriptions and handed straight back to the interpolation dispatch.
    VkImage        ofVector = VK_NULL_HANDLE, ofScd = VK_NULL_HANDLE;
    VkDeviceMemory ofVectorMem = VK_NULL_HANDLE, ofScdMem = VK_NULL_HANDLE;
    FfxResourceDescription ofVectorDesc, ofScdDesc;

    uint64_t frameID    = 0;
    uint64_t dispatches = 0;

    std::mutex lock;
};

inline State &state()
{
    static State s;
    return s;
}

// trace() is layer.cpp's file-scope static, already declared above the point
// this header is included. Redeclaring it here created fg::trace, which
// nothing defines - a link error rather than a compile one, because the
// call sites were perfectly valid against the wrong symbol.

// One image from an FFX resource description, device-local. Mirrors
// fsr3_backend_impl.h's createShared - deliberately, so the two allocate the
// same way and a fault in one is diagnosable against the other.
inline bool fgCreateImage(VkDevice device, VkPhysicalDevice phys,
                          const FfxResourceDescription &desc,
                          VkImage *outImg, VkDeviceMemory *outMem,
                          PFN_vkGetDeviceProcAddr gdpa,
                          PFN_vkGetPhysicalDeviceMemoryProperties getMemProps)
{
    PFN_vkCreateImage createImage = (PFN_vkCreateImage)gdpa(device, "vkCreateImage");
    PFN_vkGetImageMemoryRequirements getReq =
        (PFN_vkGetImageMemoryRequirements)gdpa(device, "vkGetImageMemoryRequirements");
    PFN_vkAllocateMemory allocMem = (PFN_vkAllocateMemory)gdpa(device, "vkAllocateMemory");
    PFN_vkBindImageMemory bindMem = (PFN_vkBindImageMemory)gdpa(device, "vkBindImageMemory");
    if (!createImage || !getReq || !allocMem || !bindMem) return false;

    VkImageCreateInfo ici;
    memset(&ici, 0, sizeof(ici));
    ici.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType   = VK_IMAGE_TYPE_2D;
    // fsr3::vkFormatOf, not an SDK call: the SDK only ships the VkFormat ->
    // FfxSurfaceFormat direction, so the reverse is this project's own and
    // already exists. It REFUSES an unknown format rather than substituting
    // a plausible one, which is what we want here too - a wrong format binds
    // real memory and misreads it, surfacing as corruption rather than error.
    ici.format      = fsr3::vkFormatOf(desc.format);
    ici.extent.width  = desc.width;
    ici.extent.height = desc.height;
    ici.extent.depth  = 1;
    ici.mipLevels   = desc.mipCount ? desc.mipCount : 1;
    ici.arrayLayers = 1;
    ici.samples     = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling      = VK_IMAGE_TILING_OPTIMAL;
    ici.usage       = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (createImage(device, &ici, nullptr, outImg) != VK_SUCCESS) return false;

    VkMemoryRequirements mr;
    memset(&mr, 0, sizeof(mr));
    getReq(device, *outImg, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    memset(&mp, 0, sizeof(mp));
    getMemProps(phys, &mp);
    uint32_t ti = UINT32_MAX;
    for (uint32_t k = 0; k < mp.memoryTypeCount; ++k)
        if ((mr.memoryTypeBits & (1u << k)) &&
            (mp.memoryTypes[k].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            ti = k; break;
        }
    if (ti == UINT32_MAX) return false;

    VkMemoryAllocateInfo mai;
    memset(&mai, 0, sizeof(mai));
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = mr.size;
    mai.memoryTypeIndex = ti;
    if (allocMem(device, &mai, nullptr, outMem) != VK_SUCCESS) return false;
    return bindMem(device, *outImg, *outMem, 0) == VK_SUCCESS;
}

// Built once the render and display sizes are both known. Its own scratch
// buffer and interface: sharing the upscaler's would tie two contexts'
// lifetimes together, and the upscaler is rebuilt on resize.
inline bool ensure(VkDevice device, VkPhysicalDevice phys,
                   PFN_vkGetDeviceProcAddr gdpa,
                   PFN_vkGetPhysicalDeviceMemoryProperties getMemProps,
                   uint32_t renderW, uint32_t renderH,
                   uint32_t dispW, uint32_t dispH,
                   VkFormat backbufferFmt)
{
    State &s = state();
    std::lock_guard<std::mutex> guard(s.lock);
    if (s.failed) return false;
    if (s.ready && s.renderW == renderW && s.renderH == renderH &&
        s.dispW == dispW && s.dispH == dispH) return true;
    if (s.ready) return true;   // resize: rebuilt by the caller's teardown
    if (!renderW || !renderH || !dispW || !dispH) return false;

    memset(&s.vkCtx, 0, sizeof(s.vkCtx));
    s.vkCtx.vkDevice         = device;
    s.vkCtx.vkPhysicalDevice = phys;
    s.vkCtx.vkDeviceProcAddr = gdpa;

    // Two contexts share this backend, so the scratch buffer must be sized for
    // two. Undersizing it is not a failure at create time - it is a memset off
    // the end of the buffer later, which is the heap corruption ffx_vk_shim.cpp
    // documents having already cost this project a day.
    const size_t maxContexts = 4;
    s.scratchSize = ffxGetScratchMemorySizeVK(phys, maxContexts);
    if (!s.scratchSize) { s.failed = true; return false; }
    s.scratch = calloc(1, s.scratchSize);   // calloc, not malloc - see ffx_vk.cpp
    if (!s.scratch) { s.failed = true; return false; }

    if (ffxGetInterfaceVK(&s.iface, ffxGetDeviceVK(&s.vkCtx),
                          s.scratch, s.scratchSize, maxContexts) != FFX_OK) {
        trace("FG: ffxGetInterfaceVK failed");
        s.failed = true; return false;
    }

    // ---- OPTICAL FLOW RUNS AT DISPLAY RESOLUTION.
    //
    // It reads the BACKBUFFER, which is the presented image, not the render
    // target - so its resolution is the display's. Sizing it to renderSize
    // would sample the wrong extent and the flow field would be wrong
    // everywhere rather than obviously broken.
    FfxOpticalflowContextDescription ofd;
    memset(&ofd, 0, sizeof(ofd));
    ofd.backendInterface = s.iface;
    ofd.flags            = 0;
    ofd.resolution.width  = dispW;
    ofd.resolution.height = dispH;
    if (ffxOpticalflowContextCreate(&s.ofCtx, &ofd) != FFX_OK) {
        trace("FG: ffxOpticalflowContextCreate failed");
        s.failed = true; return false;
    }

    FfxOpticalflowSharedResourceDescriptions ofShared;
    memset(&ofShared, 0, sizeof(ofShared));
    if (ffxOpticalflowGetSharedResourceDescriptions(&s.ofCtx, &ofShared) != FFX_OK) {
        trace("FG: optical flow shared resource descriptions unavailable");
        s.failed = true; return false;
    }
    if (!fgCreateImage(device, phys, ofShared.opticalFlowVector.resourceDescription,
                       &s.ofVector, &s.ofVectorMem, gdpa, getMemProps) ||
        !fgCreateImage(device, phys, ofShared.opticalFlowSCD.resourceDescription,
                       &s.ofScd, &s.ofScdMem, gdpa, getMemProps)) {
        trace("FG: could not allocate the optical flow outputs");
        s.failed = true; return false;
    }
    s.ofVectorDesc = ofShared.opticalFlowVector.resourceDescription;
    s.ofScdDesc    = ofShared.opticalFlowSCD.resourceDescription;

    // ---- HDR, AND DEPTH INVERTED, TO MATCH THE UPSCALER.
    //
    // These must agree with the upscaler's context flags. The interpolation
    // consumes ITS dilated depth and motion vectors, so a disagreement about
    // reversed-Z or HDR here is a disagreement about data already produced -
    // which does not fail, it just reprojects to the wrong place.
    FfxFrameInterpolationContextDescription fid;
    memset(&fid, 0, sizeof(fid));
    fid.backendInterface = s.iface;
    fid.flags = FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INVERTED |
                FFX_FRAMEINTERPOLATION_ENABLE_DEPTH_INFINITE |
                FFX_FRAMEINTERPOLATION_ENABLE_HDR_COLOR_INPUT;
    fid.maxRenderSize.width  = renderW;
    fid.maxRenderSize.height = renderH;
    fid.displaySize.width    = dispW;
    fid.displaySize.height   = dispH;
    fid.backBufferFormat = ffxGetSurfaceFormatVK(backbufferFmt);
    fid.previousInterpolationSourceFormat = fid.backBufferFormat;
    if (ffxFrameInterpolationContextCreate(&s.fiCtx, &fid) != FFX_OK) {
        trace("FG: ffxFrameInterpolationContextCreate failed");
        s.failed = true; return false;
    }

    s.renderW = renderW; s.renderH = renderH;
    s.dispW   = dispW;   s.dispH   = dispH;
    s.ready   = true;
    trace("FG: contexts created - optical flow %ux%u, interpolation %ux%u -> "
          "%ux%u. The swapchain can now ask for an interpolated frame.",
          dispW, dispH, renderW, renderH, dispW, dispH);
    return true;
}


// ---- THE SWAPCHAIN ASKS US FOR THE INTERPOLATED FRAME.
//
// Registered as FfxFrameGenerationConfig::frameGenerationCallback and called
// from inside the swapchain's present, on ITS command list - so this records
// commands and must not submit, wait, or take a lock anything in the present
// path already holds.
//
// Optical flow first, then interpolation, because interpolation consumes the
// flow field this produces. The three dilated resources come from the UPSCALER,
// which has already run for this frame: it is dispatched during the sim's own
// render, long before present. Reading them here is reading finished work, not
// racing it.
//
// Returns FFX_OK on the paths where there is nothing to do. An error tells the
// swapchain the interpolated frame is unavailable and it presents the real one
// instead, which is the correct degradation; returning an error for "not ready
// yet" would make a startup frame look like a failure.
inline FfxErrorCode dispatchCallback(const FfxFrameGenerationDispatchDescription *p,
                                     void *userCtx)
{
    (void)userCtx;
    State &s = state();
    if (!p || !s.ready || s.failed) return FFX_OK;

    FfxOpticalflowDispatchDescription od;
    memset(&od, 0, sizeof(od));
    od.commandList       = p->commandList;
    od.color             = p->presentColor;
    od.opticalFlowVector = ffxGetResourceVK(s.ofVector, s.ofVectorDesc, nullptr,
                                            FFX_RESOURCE_STATE_UNORDERED_ACCESS);
    od.opticalFlowSCD    = ffxGetResourceVK(s.ofScd, s.ofScdDesc, nullptr,
                                            FFX_RESOURCE_STATE_UNORDERED_ACCESS);
    od.reset             = p->reset;
    od.backbufferTransferFunction = p->backBufferTransferFunction;
    od.minMaxLuminance.x = p->minMaxLuminance[0];
    od.minMaxLuminance.y = p->minMaxLuminance[1];
    FfxErrorCode rc = ffxOpticalflowContextDispatch(&s.ofCtx, &od);
    if (rc != FFX_OK) {
        static bool said = false;
        if (!said) { said = true; trace("FG: optical flow dispatch failed (%d)", (int)rc); }
        return rc;
    }

    // The upscaler's three shared outputs. Held by fsr3::state(), produced this
    // frame - see the note at the top of this file.
    fsr3::State &u = fsr3::state();

    FfxFrameInterpolationDispatchDescription fd;
    memset(&fd, 0, sizeof(fd));
    fd.commandList        = p->commandList;
    fd.currentBackBuffer  = p->presentColor;
    fd.output             = p->outputs[0];
    fd.displaySize.width  = s.dispW;
    fd.displaySize.height = s.dispH;
    fd.renderSize.width   = s.renderW;
    fd.renderSize.height  = s.renderH;
    fd.interpolationRect  = p->interpolationRect;

    fd.opticalFlowVector  = od.opticalFlowVector;
    fd.opticalFlowSceneChangeDetection = od.opticalFlowSCD;
    fd.opticalFlowBufferSize.width  = s.ofVectorDesc.width;
    fd.opticalFlowBufferSize.height = s.ofVectorDesc.height;
    fd.opticalFlowScale.x = 1.0f / (float)s.dispW;
    fd.opticalFlowScale.y = 1.0f / (float)s.dispH;
    fd.opticalFlowBlockSize = 8;

    fd.reconstructedPrevDepth = ffxGetResourceVK(u.shared[0], u.sharedDesc[0], nullptr,
                                                 FFX_RESOURCE_STATE_UNORDERED_ACCESS);
    fd.dilatedDepth           = ffxGetResourceVK(u.shared[1], u.sharedDesc[1], nullptr,
                                                 FFX_RESOURCE_STATE_UNORDERED_ACCESS);
    fd.dilatedMotionVectors   = ffxGetResourceVK(u.shared[2], u.sharedDesc[2], nullptr,
                                                 FFX_RESOURCE_STATE_UNORDERED_ACCESS);

    // Same camera constants the upscaler is dispatched with. They describe the
    // same frame; two different answers here would reproject the interpolated
    // frame differently from the upscaled one.
    fd.cameraNear = 0.1f;
    fd.cameraFar  = 100000.0f;
    fd.cameraFovAngleVertical = 1.0472f;
    fd.viewSpaceToMetersFactor = 1.0f;   // X-Plane's world units ARE metres
    fd.frameTimeDelta = 16.6f;
    fd.reset   = p->reset;
    fd.frameID = p->frameID;
    fd.backBufferTransferFunction = p->backBufferTransferFunction;
    fd.minMaxLuminance[0] = p->minMaxLuminance[0];
    fd.minMaxLuminance[1] = p->minMaxLuminance[1];

    rc = ffxFrameInterpolationDispatch(&s.fiCtx, &fd);
    if (rc != FFX_OK) {
        static bool said = false;
        if (!said) { said = true; trace("FG: interpolation dispatch failed (%d)", (int)rc); }
        return rc;
    }
    if ((s.dispatches++ % 300) == 0)
        trace("FG: %llu interpolated frames dispatched (%ux%u)",
              (unsigned long long)s.dispatches, s.dispW, s.dispH);
    return FFX_OK;
}

} // namespace fg
