// FSR 3 upscaling: context creation and dispatch.
//
// Header-only and included once by layer.cpp, matching how the rest of this
// layer is built. See fsr3_backend.h for where this sits in the frame and why
// the output image handle had to be discovered rather than looked up.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "fsr3_backend.h"

#include <stdlib.h>   // malloc for the backend scratch

namespace fsr3 {

// One allocation for the three resources FSR 3 emits for downstream effects.
// They are required by the dispatch even though nothing here consumes them:
// frame interpolation would, and the API does not make them optional.
// ---- FfxSurfaceFormat BACK TO VkFormat.
//
// The SDK ships ffxGetSurfaceFormatVK for the other direction only, and the
// shared resources are described in FFX terms, so this half is ours. Covers
// what FSR 3 asks for and REFUSES anything else rather than substituting a
// plausible format - a wrong format here binds real memory and misreads it,
// which surfaces as corruption rather than an error.
inline VkFormat vkFormatOf(FfxSurfaceFormat f)
{
    switch (f) {
    case FFX_SURFACE_FORMAT_R32G32B32A32_FLOAT:  return VK_FORMAT_R32G32B32A32_SFLOAT;
    case FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT:  return VK_FORMAT_R16G16B16A16_SFLOAT;
    case FFX_SURFACE_FORMAT_R32G32_FLOAT:        return VK_FORMAT_R32G32_SFLOAT;
    case FFX_SURFACE_FORMAT_R8_UINT:             return VK_FORMAT_R8_UINT;
    case FFX_SURFACE_FORMAT_R32_UINT:            return VK_FORMAT_R32_UINT;
    case FFX_SURFACE_FORMAT_R8G8B8A8_UNORM:      return VK_FORMAT_R8G8B8A8_UNORM;
    case FFX_SURFACE_FORMAT_R11G11B10_FLOAT:     return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case FFX_SURFACE_FORMAT_R16G16_FLOAT:        return VK_FORMAT_R16G16_SFLOAT;
    case FFX_SURFACE_FORMAT_R16G16_UINT:         return VK_FORMAT_R16G16_UINT;
    case FFX_SURFACE_FORMAT_R16_FLOAT:           return VK_FORMAT_R16_SFLOAT;
    case FFX_SURFACE_FORMAT_R16_UINT:            return VK_FORMAT_R16_UINT;
    case FFX_SURFACE_FORMAT_R16_UNORM:           return VK_FORMAT_R16_UNORM;
    case FFX_SURFACE_FORMAT_R16_SNORM:           return VK_FORMAT_R16_SNORM;
    case FFX_SURFACE_FORMAT_R8_UNORM:            return VK_FORMAT_R8_UNORM;
    case FFX_SURFACE_FORMAT_R8G8_UNORM:          return VK_FORMAT_R8G8_UNORM;
    case FFX_SURFACE_FORMAT_R32_FLOAT:           return VK_FORMAT_R32_SFLOAT;
    default:                                     return VK_FORMAT_UNDEFINED;
    }
}

inline bool createShared(VkDevice device, VkPhysicalDevice phys,
                         const FfxCreateResourceDescription &desc, int slot,
                         PFN_vkGetDeviceProcAddr gdpa,
                         PFN_vkGetPhysicalDeviceMemoryProperties getMemProps)
{
    State &s = state();

    PFN_vkCreateImage       createImage = (PFN_vkCreateImage)gdpa(device, "vkCreateImage");
    PFN_vkGetImageMemoryRequirements getReq =
        (PFN_vkGetImageMemoryRequirements)gdpa(device, "vkGetImageMemoryRequirements");
    PFN_vkAllocateMemory    allocMem    = (PFN_vkAllocateMemory)gdpa(device, "vkAllocateMemory");
    PFN_vkBindImageMemory   bindMem     = (PFN_vkBindImageMemory)gdpa(device, "vkBindImageMemory");
    if (!createImage || !getReq || !allocMem || !bindMem) return false;

    VkImageCreateInfo ici;
    memset(&ici, 0, sizeof(ici));
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.extent.width  = desc.resourceDescription.width;
    ici.extent.height = desc.resourceDescription.height;
    ici.extent.depth  = 1;
    ici.mipLevels     = desc.resourceDescription.mipCount ? desc.resourceDescription.mipCount : 1;
    ici.arrayLayers   = 1;
    ici.format        = vkFormatOf(desc.resourceDescription.format);
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // STORAGE because FSR3 writes them from compute; SAMPLED and the transfer
    // bits because later passes read and clear them.
    ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

    if (ici.format == VK_FORMAT_UNDEFINED) {
        trace("FSR3: shared resource %d wants FfxSurfaceFormat %d, which has no "
              "mapping here - refusing rather than guessing a format.",
              slot, (int)desc.resourceDescription.format);
        return false;
    }
    if (createImage(device, &ici, nullptr, &s.shared[slot]) != VK_SUCCESS) return false;

    VkMemoryRequirements mr;
    getReq(device, s.shared[slot], &mr);
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
    if (allocMem(device, &mai, nullptr, &s.sharedMem[slot]) != VK_SUCCESS) return false;
    if (bindMem(device, s.shared[slot], s.sharedMem[slot], 0) != VK_SUCCESS) return false;

    s.sharedDesc[slot] = desc.resourceDescription;
    return true;
}

// ---- BUILD THE CONTEXT.
//
// Called once the render and display sizes are both known - which is only true
// after X-Plane has settled into sub-native rendering, so this cannot be done
// at device creation.
inline bool ensure(VkDevice device, VkPhysicalDevice phys,
                   PFN_vkGetDeviceProcAddr gdpa,
                   PFN_vkGetPhysicalDeviceMemoryProperties getMemProps,
                   uint32_t renderW, uint32_t renderH,
                   uint32_t outW, uint32_t outH)
{
    State &s = state();
    if (s.failed) return false;
    if (s.ready && s.renderW == renderW && s.renderH == renderH &&
        s.outW == outW && s.outH == outH) return true;
    if (s.ready) return true;          // size changed; recreation is a later concern
    if (!renderW || !renderH || !outW || !outH) return false;

    // ---- FFX RESOLVES THROUGH THE NEXT LAYER, NOT THE LOADER.
    //
    // ffx_vk.cpp calls Vulkan itself. Handing it the loader's trampoline would
    // mean re-entering the loader from inside one of its own calls, which is
    // one of the two suspects never separated in the XeSS probe crash. Given
    // our next-layer address table it resolves everything below us instead.
    VkDeviceContext vkCtx;
    memset(&vkCtx, 0, sizeof(vkCtx));
    vkCtx.vkDevice         = device;
    vkCtx.vkPhysicalDevice = phys;
    vkCtx.vkDeviceProcAddr = gdpa;

    const size_t maxContexts = 2;
    s.scratchSize = ffxGetScratchMemorySizeVK(phys, maxContexts);
    if (!s.scratchSize) {
        trace("FSR3: scratch size query returned 0 - backend unavailable");
        s.failed = true; return false;
    }
    s.scratch = malloc(s.scratchSize);
    if (!s.scratch) { s.failed = true; return false; }

    FfxErrorCode rc = ffxGetInterfaceVK(&s.iface, ffxGetDeviceVK(&vkCtx),
                                        s.scratch, s.scratchSize, maxContexts);
    if (rc != FFX_OK) {
        trace("FSR3: ffxGetInterfaceVK failed (%d)", (int)rc);
        s.failed = true; return false;
    }

    FfxFsr3UpscalerContextDescription cd;
    memset(&cd, 0, sizeof(cd));
    // HDR because X-Plane's scene target is R16G16B16A16_SFLOAT, pre-tonemap.
    // DEPTH_INVERTED because the sim uses a reversed-Z projection - the same
    // fact this layer's own reprojection already depends on.
    // AUTO_EXPOSURE because no exposure value is handed over; FSR3 deriving it
    // from the image is better than passing a wrong constant.
    cd.flags = FFX_FSR3UPSCALER_ENABLE_HIGH_DYNAMIC_RANGE |
               FFX_FSR3UPSCALER_ENABLE_DEPTH_INVERTED |
               FFX_FSR3UPSCALER_ENABLE_DEPTH_INFINITE |
               FFX_FSR3UPSCALER_ENABLE_AUTO_EXPOSURE;
    cd.maxRenderSize.width   = renderW;
    cd.maxRenderSize.height  = renderH;
    cd.maxUpscaleSize.width  = outW;
    cd.maxUpscaleSize.height = outH;
    cd.backendInterface      = s.iface;

    rc = ffxFsr3UpscalerContextCreate(&s.ctx, &cd);
    if (rc != FFX_OK) {
        trace("FSR3: ffxFsr3UpscalerContextCreate failed (%d)", (int)rc);
        s.failed = true; return false;
    }

    FfxFsr3UpscalerSharedResourceDescriptions sh;
    memset(&sh, 0, sizeof(sh));
    if (ffxFsr3UpscalerGetSharedResourceDescriptions(&s.ctx, &sh) != FFX_OK) {
        trace("FSR3: shared resource descriptions unavailable");
        s.failed = true; return false;
    }
    if (!createShared(device, phys, sh.reconstructedPrevNearestDepth, 0, gdpa, getMemProps) ||
        !createShared(device, phys, sh.dilatedDepth,                  1, gdpa, getMemProps) ||
        !createShared(device, phys, sh.dilatedMotionVectors,          2, gdpa, getMemProps)) {
        trace("FSR3: could not allocate the three shared resources");
        s.failed = true; return false;
    }

    s.renderW = renderW; s.renderH = renderH;
    s.outW = outW;       s.outH = outH;
    s.ready = true;
    trace("FSR3: context created - %ux%u -> %ux%u. This layer is now the "
          "upscaler in X-Plane's own slot.", renderW, renderH, outW, outH);
    return true;
}

// ---- ONE FRAME.
//
// Recorded into X-Plane's own command buffer, at the point its upscale would
// have run, so ordering against the scene render and everything downstream is
// the sim's own and needs no synchronisation of ours.
inline bool dispatch(VkCommandBuffer cb,
                     VkImage colorImg, VkFormat colorFmt,
                     VkImage depthImg, VkFormat depthFmt,
                     VkImage mvImg,    VkFormat mvFmt,
                     VkImage outImg,   VkFormat outFmt,
                     float jitterX, float jitterY,
                     float deltaMs, bool reset,
                     float camNear, float camFar, float fovY)
{
    State &s = state();
    if (!s.ready || s.failed) return false;
    if (colorImg == VK_NULL_HANDLE || depthImg == VK_NULL_HANDLE ||
        mvImg == VK_NULL_HANDLE || outImg == VK_NULL_HANDLE) return false;

    FfxFsr3UpscalerDispatchDescription d;
    memset(&d, 0, sizeof(d));
    d.commandList = ffxGetCommandListVK(cb);

    d.color = ffxGetResourceVK(colorImg,
        describeTex2D(s.renderW, s.renderH, ffxGetSurfaceFormatVK(colorFmt),
                      FFX_RESOURCE_USAGE_READ_ONLY),
        nullptr, FFX_RESOURCE_STATE_COMPUTE_READ);
    d.depth = ffxGetResourceVK(depthImg,
        describeTex2D(s.renderW, s.renderH, ffxGetSurfaceFormatVK(depthFmt),
                      FFX_RESOURCE_USAGE_DEPTHTARGET),
        nullptr, FFX_RESOURCE_STATE_COMPUTE_READ);
    d.motionVectors = ffxGetResourceVK(mvImg,
        describeTex2D(s.renderW, s.renderH, ffxGetSurfaceFormatVK(mvFmt),
                      FFX_RESOURCE_USAGE_READ_ONLY),
        nullptr, FFX_RESOURCE_STATE_COMPUTE_READ);
    d.output = ffxGetResourceVK(outImg,
        describeTex2D(s.outW, s.outH, ffxGetSurfaceFormatVK(outFmt),
                      FFX_RESOURCE_USAGE_UAV),
        nullptr, FFX_RESOURCE_STATE_UNORDERED_ACCESS);

    d.reconstructedPrevNearestDepth = ffxGetResourceVK(s.shared[0], s.sharedDesc[0],
        nullptr, FFX_RESOURCE_STATE_UNORDERED_ACCESS);
    d.dilatedDepth = ffxGetResourceVK(s.shared[1], s.sharedDesc[1],
        nullptr, FFX_RESOURCE_STATE_UNORDERED_ACCESS);
    d.dilatedMotionVectors = ffxGetResourceVK(s.shared[2], s.sharedDesc[2],
        nullptr, FFX_RESOURCE_STATE_UNORDERED_ACCESS);

    d.jitterOffset.x = jitterX;
    d.jitterOffset.y = jitterY;
    // ---- MOTION VECTOR SCALE: OURS ARE ALREADY IN NDC.
    //
    // FSR3 multiplies the sampled vector by this to reach pixels. This layer's
    // injected shaders emit clip-space displacement, so the conversion is the
    // render size - and the sign follows the same Y convention the resolve
    // uses. A disagreement here does not fail; it produces smearing that looks
    // like a temporal bug, which is the expensive kind to chase.
    d.motionVectorScale.x = (float)s.renderW;
    d.motionVectorScale.y = (float)s.renderH;

    d.renderSize.width   = s.renderW;
    d.renderSize.height  = s.renderH;
    d.upscaleSize.width  = s.outW;
    d.upscaleSize.height = s.outH;

    d.enableSharpening = false;
    d.sharpness        = 0.0f;
    d.frameTimeDelta   = deltaMs > 0.0f ? deltaMs : 16.6f;
    d.preExposure      = 1.0f;
    d.reset            = reset;
    d.cameraNear       = camNear;
    d.cameraFar        = camFar;
    d.cameraFovAngleVertical = fovY;
    d.viewSpaceToMetersFactor = 1.0f;   // X-Plane's world units ARE metres

    const FfxErrorCode rc = ffxFsr3UpscalerContextDispatch(&s.ctx, &d);
    if (rc != FFX_OK) {
        // Loud once, then silent: a dispatch failing every frame would bury
        // everything else in the trace.
        static bool said = false;
        if (!said) {
            said = true;
            trace("FSR3: dispatch failed (%d) - falling back to the built-in "
                  "upscaler for the rest of this run.", (int)rc);
        }
        s.failed = true;
        return false;
    }

    ++s.dispatches;
    if (s.dispatches == 1 || (s.dispatches % 600) == 0)
        trace("FSR3: %llu dispatches, %ux%u -> %ux%u",
              (unsigned long long)s.dispatches, s.renderW, s.renderH, s.outW, s.outH);
    return true;
}

} // namespace fsr3
