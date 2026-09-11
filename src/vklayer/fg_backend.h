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

// Live isolation control, implemented in layer.cpp where the live config lives.
// Returns non-zero when frame interpolation should run (taa.fg_fi, default on).
// Lets optical flow and interpolation be dispatched independently to bisect a
// fault between them without a rebuild.
extern "C" int mvFgWantFI();
// Returns non-zero when the callback should record ANY work (taa.fg_of, default
// on). fg_of=0 makes the callback a no-op, to tell an optical-flow/interpolation
// fault apart from the upscaler or the swapchain present path running alongside.
extern "C" int mvFgWantOF();

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
    // Wall-clock start of the current quiet window: the first unheld frame
    // after startup or after an aircraft-swap hold. 0 = re-arm on next frame.
    uint64_t quietStartMs = 0;
    // ---- HUD-LESS SOURCE. The backbuffer copied at the end of the FIRST pass that
    // targets it each frame, i.e. after the tonemapped scene lands on it and before
    // the 2-D panel, popups and menus are drawn. Handed to interpolation as
    // currentBackBuffer_HUDLess: FFX interpolates from it and re-stamps every pixel
    // where the presented frame differs (the overlays), so overlay text stays sharp
    // on generated frames instead of being warped like scenery.
    VkImage        hudless      = VK_NULL_HANDLE;
    VkDeviceMemory hudlessMem   = VK_NULL_HANDLE;
    uint32_t       hudlessW = 0, hudlessH = 0;
    VkFormat       hudlessFmt   = VK_FORMAT_UNDEFINED;
    bool           hudlessInit  = false;      // false until the first copy laid a layout down
    uint64_t       hudlessFrame = 0;          // present index the last copy belongs to
    uint64_t       presentFrame = 0;          // set by QueuePresent before the proxy presents
    uint32_t       swapPassesThisFrame = 0;   // census: how many passes target the backbuffer
    uint64_t       hudlessCopies = 0;
    uint64_t       hudlessUsed   = 0;
    uint32_t       swapPassesLastFrame = 0;   // census value captured before the present-time reset
    // ---- FG SNAP AUDIT (every 300 dispatches): row strips of our dilated vectors, the generated
    // frame, the current frame and the HUD-less copy, compared on the CPU three dispatches later.
    VkBuffer       audBuf[4]  = { VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDeviceMemory audMem[4]  = { VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE };
    void          *audPtr[4]  = { nullptr, nullptr, nullptr, nullptr };
    VkDeviceSize   audSize[4] = { 0, 0, 0, 0 };
    bool           audFailed  = false;
    bool           audPending = false;
    bool           audHadHudless = false;
    uint64_t       audRecordedAt = 0;
    uint32_t       audRows = 0, audMvW = 0, audMvH = 0, audColW = 0, audColH = 0;
    uint64_t       audCount = 0;

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

    trace("FG ENSURE: ffxGetInterfaceVK (own scratch %zu bytes)", (size_t)s.scratchSize);
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
    trace("FG ENSURE: creating optical flow context %ux%u", dispW, dispH);
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
    trace("FG ENSURE: creating frame interpolation context");
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
// ---- RECORD THE HUD-LESS COPY. Called from Layer_CmdEndRendering, on X-Plane's
// own command buffer, right after the first backbuffer-targeting pass of the frame
// ended. The backbuffer is still in the layout the pass declared; it goes back to
// exactly that. The copy image is ours, in the swapchain's (post-substitution,
// UNORM) format, which is what the FI context was created with, so FFX accepts it
// as currentBackBuffer_HUDLess without a format-group mismatch.
inline bool recordHudless(VkDevice device, VkPhysicalDevice phys,
                          PFN_vkGetDeviceProcAddr gdpa,
                          PFN_vkGetPhysicalDeviceMemoryProperties getMemProps,
                          PFN_vkCmdPipelineBarrier barrierFn, PFN_vkCmdCopyImage copyFn,
                          VkCommandBuffer cb, VkImage swap, VkImageLayout swapLayout,
                          uint32_t w, uint32_t h, VkFormat fmt, uint64_t frame)
{
    State &s = state();
    if (!device || !gdpa || !barrierFn || !copyFn || !cb || swap == VK_NULL_HANDLE || !w || !h) return false;
    if (s.hudless != VK_NULL_HANDLE && (s.hudlessW != w || s.hudlessH != h || s.hudlessFmt != fmt)) {
        PFN_vkDeviceWaitIdle waitIdle = (PFN_vkDeviceWaitIdle)gdpa(device, "vkDeviceWaitIdle");
        PFN_vkDestroyImage destroyImage = (PFN_vkDestroyImage)gdpa(device, "vkDestroyImage");
        PFN_vkFreeMemory freeMem = (PFN_vkFreeMemory)gdpa(device, "vkFreeMemory");
        if (waitIdle) waitIdle(device);
        if (destroyImage) destroyImage(device, s.hudless, nullptr);
        if (freeMem && s.hudlessMem) freeMem(device, s.hudlessMem, nullptr);
        s.hudless = VK_NULL_HANDLE; s.hudlessMem = VK_NULL_HANDLE; s.hudlessInit = false;
    }
    if (s.hudless == VK_NULL_HANDLE) {
        PFN_vkCreateImage createImage = (PFN_vkCreateImage)gdpa(device, "vkCreateImage");
        PFN_vkGetImageMemoryRequirements getReq = (PFN_vkGetImageMemoryRequirements)gdpa(device, "vkGetImageMemoryRequirements");
        PFN_vkAllocateMemory allocMem = (PFN_vkAllocateMemory)gdpa(device, "vkAllocateMemory");
        PFN_vkBindImageMemory bindMem = (PFN_vkBindImageMemory)gdpa(device, "vkBindImageMemory");
        PFN_vkDestroyImage destroyImage = (PFN_vkDestroyImage)gdpa(device, "vkDestroyImage");
        if (!createImage || !getReq || !allocMem || !bindMem || !getMemProps) return false;
        VkImageCreateInfo ici;
        memset(&ici, 0, sizeof(ici));
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = fmt;
        ici.extent.width = w; ici.extent.height = h; ici.extent.depth = 1;
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImage img = VK_NULL_HANDLE;
        if (createImage(device, &ici, nullptr, &img) != VK_SUCCESS) {
            static bool said = false;
            if (!said) { said = true; trace("FG HUDLESS: image create failed (%ux%u fmt %d)", w, h, (int)fmt); }
            return false;
        }
        VkMemoryRequirements mr; memset(&mr, 0, sizeof(mr)); getReq(device, img, &mr);
        VkPhysicalDeviceMemoryProperties mp; memset(&mp, 0, sizeof(mp)); getMemProps(phys, &mp);
        uint32_t ti = UINT32_MAX;
        for (uint32_t k = 0; k < mp.memoryTypeCount; ++k)
            if ((mr.memoryTypeBits & (1u << k)) && (mp.memoryTypes[k].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { ti = k; break; }
        VkMemoryAllocateInfo mai; memset(&mai, 0, sizeof(mai));
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; mai.allocationSize = mr.size; mai.memoryTypeIndex = ti;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        if (ti == UINT32_MAX || allocMem(device, &mai, nullptr, &mem) != VK_SUCCESS ||
            bindMem(device, img, mem, 0) != VK_SUCCESS) {
            if (destroyImage) destroyImage(device, img, nullptr);
            static bool said = false;
            if (!said) { said = true; trace("FG HUDLESS: memory for the %ux%u copy failed", w, h); }
            return false;
        }
        s.hudless = img; s.hudlessMem = mem; s.hudlessW = w; s.hudlessH = h; s.hudlessFmt = fmt;
        s.hudlessInit = false;
        trace("FG HUDLESS: %ux%u copy image created (fmt %d) - the backbuffer before the overlay passes "
              "now feeds interpolation as the HUD-less source.", w, h, (int)fmt);
    }
    VkImageMemoryBarrier b[2];
    memset(b, 0, sizeof(b));
    for (int i = 0; i < 2; ++i) {
        b[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b[i].srcQueueFamilyIndex = b[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b[i].subresourceRange.levelCount = 1;
        b[i].subresourceRange.layerCount = 1;
    }
    b[0].image = swap;
    b[0].oldLayout = swapLayout; b[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    b[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b[1].image = s.hudless;
    b[1].oldLayout = s.hudlessInit ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    b[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrierFn(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
              0, 0, nullptr, 0, nullptr, 2, b);
    VkImageCopy rgn; memset(&rgn, 0, sizeof(rgn));
    rgn.srcSubresource.aspectMask = rgn.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    rgn.srcSubresource.layerCount = rgn.dstSubresource.layerCount = 1;
    rgn.extent.width = w; rgn.extent.height = h; rgn.extent.depth = 1;
    copyFn(cb, swap, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.hudless, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rgn);
    b[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; b[0].newLayout = swapLayout;
    b[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    b[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrierFn(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
              0, 0, nullptr, 0, nullptr, 2, b);
    s.hudlessInit  = true;
    s.hudlessFrame = frame;
    ++s.hudlessCopies;
    return true;
}

// ---- FG SNAP AUDIT. The only number that settles "is the snap firing on the screens": read
// back row strips and count. Trust and stillness come from OUR dilated vectors (.z = trust,
// .xy = full-frame displacement in render px; the interpolator halves them, so still means
// |mv| < 2*eps). Snap firing shows as generated pixels byte-identical to the current frame.
// Overlays show as present != HUD-less. All copies ride the FI command list after the dispatch.
static const uint32_t kAudRowStep = 32;
inline bool fgAuditBuffer(State &s, int k, VkDeviceSize bytes)
{
    if (s.audBuf[k] != VK_NULL_HANDLE && s.audSize[k] >= bytes) return true;
    VkDevice dev = s.vkCtx.vkDevice; PFN_vkGetDeviceProcAddr gdpa = s.vkCtx.vkDeviceProcAddr;
    if (!dev || !gdpa || !g_getPhysMemProps) return false;
    PFN_vkCreateBuffer createBuf = (PFN_vkCreateBuffer)gdpa(dev, "vkCreateBuffer");
    PFN_vkGetBufferMemoryRequirements getReq = (PFN_vkGetBufferMemoryRequirements)gdpa(dev, "vkGetBufferMemoryRequirements");
    PFN_vkAllocateMemory allocMem = (PFN_vkAllocateMemory)gdpa(dev, "vkAllocateMemory");
    PFN_vkBindBufferMemory bindMem = (PFN_vkBindBufferMemory)gdpa(dev, "vkBindBufferMemory");
    PFN_vkMapMemory mapMem = (PFN_vkMapMemory)gdpa(dev, "vkMapMemory");
    if (!createBuf || !getReq || !allocMem || !bindMem || !mapMem) return false;
    VkBufferCreateInfo bci; memset(&bci, 0, sizeof(bci));
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer b = VK_NULL_HANDLE;
    if (createBuf(dev, &bci, nullptr, &b) != VK_SUCCESS) return false;
    VkMemoryRequirements mr; memset(&mr, 0, sizeof(mr)); getReq(dev, b, &mr);
    VkPhysicalDeviceMemoryProperties mp; memset(&mp, 0, sizeof(mp)); g_getPhysMemProps(s.vkCtx.vkPhysicalDevice, &mp);
    uint32_t ti = UINT32_MAX;
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) { ti = i; break; }
    if (ti == UINT32_MAX) return false;
    VkMemoryAllocateInfo mai; memset(&mai, 0, sizeof(mai));
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; mai.allocationSize = mr.size; mai.memoryTypeIndex = ti;
    VkDeviceMemory m = VK_NULL_HANDLE; void *ptr = nullptr;
    if (allocMem(dev, &mai, nullptr, &m) != VK_SUCCESS || bindMem(dev, b, m, 0) != VK_SUCCESS ||
        mapMem(dev, m, 0, VK_WHOLE_SIZE, 0, &ptr) != VK_SUCCESS) return false;
    s.audBuf[k] = b; s.audMem[k] = m; s.audPtr[k] = ptr; s.audSize[k] = bytes;
    return true;
}

// Copy every kAudRowStep-th row of `img` into audit buffer k. Layout in/out are what the image
// is in right now; it is returned to exactly that.
inline void fgAuditCopyRows(State &s, VkCommandBuffer cb, PFN_vkCmdPipelineBarrier barrierFn,
                            PFN_vkCmdCopyImageToBuffer copyFn, int k, VkImage img, VkImageLayout layout,
                            VkAccessFlags access, uint32_t w, uint32_t h, uint32_t bpp, uint32_t rows)
{
    VkImageMemoryBarrier b; memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; b.subresourceRange.levelCount = 1; b.subresourceRange.layerCount = 1;
    b.image = img; b.oldLayout = layout; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.srcAccessMask = access; b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrierFn(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    std::vector<VkBufferImageCopy> rg(rows);
    for (uint32_t r = 0; r < rows; ++r) {
        memset(&rg[r], 0, sizeof(rg[r]));
        rg[r].bufferOffset = (VkDeviceSize)r * w * bpp;
        rg[r].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; rg[r].imageSubresource.layerCount = 1;
        rg[r].imageOffset.y = (int32_t)(r * kAudRowStep + kAudRowStep / 2);
        rg[r].imageExtent.width = w; rg[r].imageExtent.height = 1; rg[r].imageExtent.depth = 1;
    }
    copyFn(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.audBuf[k], rows, rg.data());
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; b.newLayout = layout;
    b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT; b.dstAccessMask = access;
    barrierFn(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}

inline void fgAuditRecord(State &s, const FfxFrameGenerationDispatchDescription *p,
                          const FfxFrameInterpolationDispatchDescription &fd, bool ownInputs)
{
    if (s.audFailed || s.audPending || !ownInputs) return;
    if ((s.dispatches % 300) != 5) return;
    fgprep::State &g = fgprep::state();
    if (g.readySlot < 0 || !g.dilMv[g.readySlot]) return;
    VkDevice dev = s.vkCtx.vkDevice; PFN_vkGetDeviceProcAddr gdpa = s.vkCtx.vkDeviceProcAddr;
    static PFN_vkCmdPipelineBarrier barrierFn = nullptr; static PFN_vkCmdCopyImageToBuffer copyFn = nullptr;
    if (!barrierFn) barrierFn = (PFN_vkCmdPipelineBarrier)gdpa(dev, "vkCmdPipelineBarrier");
    if (!copyFn)    copyFn    = (PFN_vkCmdCopyImageToBuffer)gdpa(dev, "vkCmdCopyImageToBuffer");
    if (!barrierFn || !copyFn) { s.audFailed = true; return; }
    const uint32_t mvW = g.w, mvH = g.h, colW = s.dispW, colH = s.dispH;
    const uint32_t rows = (colH < mvH ? colH : mvH) / kAudRowStep;
    if (!rows || !mvW || !colW) return;
    if (!fgAuditBuffer(s, 0, (VkDeviceSize)rows * mvW * 8) || !fgAuditBuffer(s, 1, (VkDeviceSize)rows * colW * 4) ||
        !fgAuditBuffer(s, 2, (VkDeviceSize)rows * colW * 4) || !fgAuditBuffer(s, 3, (VkDeviceSize)rows * colW * 4)) {
        s.audFailed = true; trace("FG SNAP AUDIT: readback buffers failed - audit off"); return;
    }
    VkCommandBuffer cb = (VkCommandBuffer)p->commandList;
    // dilated vectors: GENERAL (prep pass leaves them there, FFX registered them UNORDERED_ACCESS)
    fgAuditCopyRows(s, cb, barrierFn, copyFn, 0, g.dilMv[g.readySlot], VK_IMAGE_LAYOUT_GENERAL,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, mvW, mvH, 8, rows);
    // generated frame: GENERAL after the dispatch (FFX returns it to its registered state)
    fgAuditCopyRows(s, cb, barrierFn, copyFn, 1, (VkImage)fd.output.resource, VK_IMAGE_LAYOUT_GENERAL,
                    VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT, colW, colH, 4, rows);
    // current (interpolation source): PIXEL_COMPUTE_READ = shader-read layout
    fgAuditCopyRows(s, cb, barrierFn, copyFn, 2, (VkImage)fd.currentBackBuffer.resource, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_SHADER_READ_BIT, colW, colH, 4, rows);
    s.audHadHudless = fd.currentBackBuffer_HUDLess.resource != nullptr;
    if (s.audHadHudless)
        fgAuditCopyRows(s, cb, barrierFn, copyFn, 3, (VkImage)fd.currentBackBuffer_HUDLess.resource, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_SHADER_READ_BIT, colW, colH, 4, rows);
    s.audRows = rows; s.audMvW = mvW; s.audMvH = mvH; s.audColW = colW; s.audColH = colH;
    s.audRecordedAt = s.dispatches; s.audPending = true;
}

static inline float fgHalfToFloat(uint16_t h)
{
    const uint32_t sgn = (h >> 15) & 1u, ex = (h >> 10) & 0x1Fu, mn = h & 0x3FFu;
    float v;
    if (ex == 0) v = (float)mn / 1024.0f / 16384.0f;
    else if (ex == 31) v = mn ? 0.0f : 65504.0f;
    else v = (1.0f + (float)mn / 1024.0f) * (float)pow(2.0, (int)ex - 15);
    return sgn ? -v : v;
}

inline void fgAuditCollect(State &s)
{
    if (!s.audPending || s.dispatches < s.audRecordedAt + 3) return;
    s.audPending = false;
    const uint32_t rows = s.audRows, mvW = s.audMvW, colW = s.audColW;
    const float epsFull = 2.0f * fgSnapEpsPx(fgTrustFlags(true, true, live::f("taa.fg_snap_eps", "TAA_FG_SNAP_EPS", 0.25f), false));
    uint64_t n = 0, nTrust = 0, nStill = 0, nB = 0, nBTrust = 0, nBStill = 0; double sumB = 0.0;
    const uint16_t *mv = (const uint16_t*)s.audPtr[0];
    for (uint32_t r = 0; r < rows; ++r) {
        const bool bottom = (r * kAudRowStep + kAudRowStep / 2) >= (s.audMvH * 2) / 3;
        for (uint32_t x = 0; x < mvW; x += 4) {
            const uint16_t *t = mv + ((size_t)r * mvW + x) * 4;
            const float vx = fgHalfToFloat(t[0]), vy = fgHalfToFloat(t[1]), tz = fgHalfToFloat(t[2]);
            const float len = sqrtf(vx * vx + vy * vy);
            const bool trusted = tz > 0.5f, still = trusted && len < epsFull;
            ++n; nTrust += trusted; nStill += still;
            if (bottom) { ++nB; nBTrust += trusted; nBStill += still; if (trusted) sumB += len; }
        }
    }
    uint64_t m = 0, same = 0, mB = 0, sameB = 0, ov = 0;
    const uint32_t *out = (const uint32_t*)s.audPtr[1], *cur = (const uint32_t*)s.audPtr[2], *hud = (const uint32_t*)s.audPtr[3];
    for (uint32_t r = 0; r < rows; ++r) {
        const bool bottom = (r * kAudRowStep + kAudRowStep / 2) >= (s.audColH * 2) / 3;
        for (uint32_t x = 0; x < colW; x += 2) {
            const size_t i = (size_t)r * colW + x;
            ++m; const bool eq = (out[i] & 0x00FFFFFFu) == (cur[i] & 0x00FFFFFFu); same += eq;   // RGB only: FFX keeps its inpainting weight in .a
            if (bottom) { ++mB; sameB += eq; }
            if (s.audHadHudless && (hud[i] & 0x00FFFFFFu) != (cur[i] & 0x00FFFFFFu)) ++ov;
        }
    }
    ++s.audCount;
    trace("FG SNAP AUDIT #%llu: dilated vectors trusted=%.1f%% still(|mv|<%.2fpx)=%.1f%% | bottom third: trusted=%.1f%% "
          "still=%.1f%% mean|mv|=%.2fpx | generated==current %.1f%% of pixels (bottom third %.1f%%) | overlays (present!=hudless) %.1f%%%s",
          (unsigned long long)s.audCount, 100.0 * nTrust / (n ? n : 1), epsFull, 100.0 * nStill / (n ? n : 1),
          100.0 * nBTrust / (nB ? nB : 1), 100.0 * nBStill / (nB ? nB : 1), nBTrust ? sumB / nBTrust : 0.0,
          100.0 * same / (m ? m : 1), 100.0 * sameB / (mB ? mB : 1), 100.0 * ov / (m ? m : 1),
          s.audHadHudless ? "" : " (no hudless this frame)");
}

// ---- NEVER LEAVE THE INTERPOLATED SLOT UNWRITTEN.
//
// Once the swapchain config has generation enabled it presents outputs[0] on
// every other flip no matter what this callback did. Every early return below
// (aircraft hold, quiet window, inputs not ready, dispatch failure) used to
// return without touching it, so the flip showed the buffer's stale or
// uninitialised contents - the black frames seen 2026-09-10 the moment
// generation first enabled. Copying the real frame in makes those flips a
// repeat of the real frame instead, which is what 'presenting real frames
// only' was always meant to be.
inline void presentRealFrame(const FfxFrameGenerationDispatchDescription *p, const char *why)
{
    State &s = state();
    if (!p || !p->commandList || !p->presentColor.resource || !p->outputs[0].resource) return;
    PFN_vkGetDeviceProcAddr gdpa = s.vkCtx.vkDeviceProcAddr;
    VkDevice device = s.vkCtx.vkDevice;
    if (!gdpa || !device) return;
    static PFN_vkCmdPipelineBarrier barrierFn = nullptr;
    static PFN_vkCmdCopyImage copyFn = nullptr;
    if (!barrierFn) barrierFn = (PFN_vkCmdPipelineBarrier)gdpa(device, "vkCmdPipelineBarrier");
    if (!copyFn)    copyFn    = (PFN_vkCmdCopyImage)gdpa(device, "vkCmdCopyImage");
    if (!barrierFn || !copyFn) return;
    VkCommandBuffer cb = (VkCommandBuffer)p->commandList;
    VkImage src = (VkImage)p->presentColor.resource;
    VkImage dst = (VkImage)p->outputs[0].resource;
    const uint32_t w = p->presentColor.description.width  < p->outputs[0].description.width
                     ? p->presentColor.description.width  : p->outputs[0].description.width;
    const uint32_t h = p->presentColor.description.height < p->outputs[0].description.height
                     ? p->presentColor.description.height : p->outputs[0].description.height;
    if (!w || !h) return;
    VkImageMemoryBarrier b[2];
    memset(b, 0, sizeof(b));
    for (int i = 0; i < 2; ++i) {
        b[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b[i].srcQueueFamilyIndex = b[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b[i].subresourceRange.levelCount = 1;
        b[i].subresourceRange.layerCount = 1;
    }
    // The swapchain hands presentColor in PIXEL_COMPUTE_READ (shader-read layout)
    // and outputs[0] in GENERAL; both go back to exactly that afterwards.
    b[0].image = src; b[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT; b[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b[1].image = dst; b[1].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    b[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    b[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrierFn(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
              0, 0, nullptr, 0, nullptr, 2, b);
    VkImageCopy rgn;
    memset(&rgn, 0, sizeof(rgn));
    rgn.srcSubresource.aspectMask = rgn.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    rgn.srcSubresource.layerCount = rgn.dstSubresource.layerCount = 1;
    rgn.extent.width = w; rgn.extent.height = h; rgn.extent.depth = 1;
    copyFn(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rgn);
    b[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; b[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT; b[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b[1].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    barrierFn(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
              0, 0, nullptr, 0, nullptr, 2, b);
    static uint64_t filled = 0;
    if ((filled++ % 300) == 0)
        trace("FG: interpolated slot filled with the REAL frame (%s) - %llu so far. Each one "
              "is a repeated frame, not a black one.", why, (unsigned long long)filled);
}
inline FfxErrorCode dispatchCallback(const FfxFrameGenerationDispatchDescription *p,
                                     void *userCtx)
{
    (void)userCtx;
    State &s = state();
    if (!p || !s.ready || s.failed) return FFX_OK;

    {
        static bool said = false;
        if (!said) {
            said = true;
            trace("FG CB ENTRY: present=%p cmdList=%p ofVec=%p ofScd=%p "
                  "reset=%d - optical flow runs on presentColor every frame, "
                  "before the interpolation skip.",
                  p->presentColor.resource, (void*)p->commandList,
                  (void*)s.ofVector, (void*)s.ofScd, (int)p->reset);
        }
    }

    // ---- ISOLATION CONTROL: dispatch NOTHING. taa.fg_of=0 returns before
    // optical flow (and so before interpolation). If the "0x0 Compute" fault
    // survives with the callback recording no work at all, it is not in our
    // optical-flow or interpolation dispatch - it is the upscaler running
    // alongside, or the swapchain's own present path. Default on.
    if (!mvFgWantOF()) {
        static bool said = false;
        if (!said) { said = true;
            trace("FG: taa.fg_of=0 - callback records nothing (no optical flow, "
                  "no interpolation)."); }
        return FFX_OK;
    }

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

    // ---- ISOLATION CONTROL: skip frame interpolation, run optical flow only.
    //
    // taa.fg_fi=0 stops here after optical flow, presenting the real frame. Used
    // to bisect the interpolation crash: if the "0x0 Compute" fault survives
    // with interpolation OFF, the faulting shader is in optical flow, not the
    // interpolation dispatch. Default on.
    if (!mvFgWantFI()) {
        static bool said = false;
        if (!said) { said = true;
            trace("FG: taa.fg_fi=0 - optical flow only, interpolation skipped."); }
        return FFX_OK;
    }

    // The upscaler's three shared outputs. Held by fsr3::state(), produced this
    // frame - see the note at the top of this file.
    fsr3::State &u = fsr3::state();

    // ---- NO DILATED RESOURCES, NO INTERPOLATION. THIS FRAME PRESENTS REAL.
    //
    // Frame interpolation reads dilatedDepth, dilatedMotionVectors and
    // reconstructedPrevDepth. Those are the UPSCALER's, and the upscaler is a
    // separate context that comes up on its own schedule - later than this
    // callback can first fire, because the swapchain (and so this callback) is
    // live from the loading screen while the upscaler waits for a real scene.
    //
    // Handing the interpolation a VK_NULL_HANDLE for any of them is a compute
    // shader dereferencing address 0: "Encountered Unknown at virtual address
    // 0x0, Type: Compute", one frame after the first dispatch. Measured exactly
    // that. Returning FFX_OK here tells the swapchain there is no interpolated
    // frame this time, so it presents the real one - the correct behaviour
    // until the upscaler is producing, and the natural place a future
    // self-contained depth/MV path would plug in.
    // ---- OUR OWN INPUTS COUNT AS INPUTS.
    //
    // The note above called this "the natural place a future self-contained
    // depth/MV path would plug in". This is that path: fgprep produces the same
    // three textures directly, so interpolation no longer needs the upscaler to
    // have run - which is what frees the gigabyte the upscaler was costing.
    // An aircraft swap holds generation off entirely - see fgHeldForAircraftSwap.
    if (fgHeldForAircraftSwap()) {
        // ---- AND THROW THE INPUTS AWAY, NOT JUST SKIP A FRAME.
        //
        // The prepare pass stops running while held, so its last results sit
        // there looking valid. Leaving readySlot alone means the moment the
        // hold lifts, interpolation blends against textures describing a scene
        // that no longer exists.
        //
        // That is precisely what was seen: real frames showing X-Plane's
        // loading screen while the interpolated ones still showed the previous
        // aircraft in flight. Marking the inputs stale forces interpolation to
        // wait for a genuinely fresh pair.
        fgprep::state().readySlot = -1;
        // Re-arm the post-hold quiet window: the 12 seconds start counting
        // from the frame the hold RELEASES, not from when it began.
        s.quietStartMs = 0;
        static uint64_t held = 0;
        if ((held++ % 300) == 0)
            trace("FG: holding - aircraft swap in progress, presenting real "
                  "frames only and discarding stale interpolation inputs "
                  "(%llu frames held).", (unsigned long long)held);
        presentRealFrame(p, "aircraft hold");
        return FFX_OK;
    }
    // ---- THE QUIET WINDOW: NO INTERPOLATED FRAMES FOR THE FIRST 12 SECONDS.
    //
    // The black flicker at the start of a flight is interpolation running
    // against inputs that are technically written but describe a scene still
    // assembling - texture paging, the loading blackout, the first unsettled
    // seconds. Every targeted gate tried so far (readySlot, runs > 0, the
    // swap hold) closed one hole and left the next; twelve seconds of real
    // frames closes the class. The cost is invisible: nobody needs generated
    // frames during a loading transition.
    //
    // Wall clock, not frames, deliberately - the whole point is that frame
    // cadence is chaotic in exactly this window. Counts from startup and
    // re-arms whenever the aircraft-swap hold releases.
    {
        const uint64_t nowMs = (uint64_t)GetTickCount64();
        if (s.quietStartMs == 0) s.quietStartMs = nowMs;
        const uint64_t quietMs = (uint64_t)live::f("taa.fg_quiet_ms",
                                                   "TAA_FG_QUIET_MS", 12000.0f);
        if (nowMs - s.quietStartMs < quietMs) {
            // Inputs stay stale too, so the first interpolated frame after the
            // window blends a genuinely fresh pair rather than one banked
            // twelve seconds ago.
            fgprep::state().readySlot = -1;
            static uint64_t quiet = 0;
            if ((quiet++ % 300) == 0)
                trace("FG: quiet window - %llu ms of %llu elapsed, presenting "
                      "real frames only.",
                      (unsigned long long)(nowMs - s.quietStartMs),
                      (unsigned long long)quietMs);
            presentRealFrame(p, "quiet window");
            return FFX_OK;
        }
    }
    // ---- READY IS NOT THE SAME AS WRITTEN.
    //
    // This asked only whether the prepare pass had been BUILT. Its images exist
    // from that moment but contain nothing until it has actually dispatched, so
    // interpolation spent the first frames reading undefined memory and the
    // swapchain presented the result: black frames, flickering, until the pass
    // caught up.
    //
    // That is the same failure 62d744e fixed for the upscaler path - enabling
    // generation before its inputs exist - reintroduced here because "ready"
    // reads like it means the data is there. runs > 0 is the honest test: the
    // three textures have been written at least once.
    const bool ownInputs = fgprep::state().ready && !fgprep::state().failed &&
                           fgprep::state().readySlot >= 0 &&
                           live::onoff("taa.fg_own_prepare", "TAA_FG_OWN_PREPARE", true);
    if (!ownInputs &&
        (!u.ready || u.failed ||
         u.shared[0] == VK_NULL_HANDLE ||
         u.shared[1] == VK_NULL_HANDLE ||
         u.shared[2] == VK_NULL_HANDLE)) {
        static uint64_t skipped = 0;
        if ((skipped++ % 300) == 0)
            trace("FG: upscaler resources not ready (ready=%d) - presenting the "
                  "real frame, no interpolation yet (%llu skipped).",
                  u.ready ? 1 : 0, (unsigned long long)skipped);
        presentRealFrame(p, "inputs not ready");
        return FFX_OK;
    }

    fgAuditCollect(s);
    FfxFrameInterpolationDispatchDescription fd;
    memset(&fd, 0, sizeof(fd));
    fd.commandList        = p->commandList;
    fd.currentBackBuffer  = p->presentColor;
    fd.output             = p->outputs[0];
    fd.displaySize.width  = s.dispW;
    fd.displaySize.height = s.dispH;
    // HUD-less source: only when this present's copy exists (same frame index) and
    // the live switch is on. taa.fg_hudless=0 is the A/B: overlays interpolated as
    // scenery again.
    if (s.hudless != VK_NULL_HANDLE && s.hudlessInit && s.hudlessFrame == s.presentFrame &&
        s.hudlessW == s.dispW && s.hudlessH == s.dispH &&
        live::onoff("taa.fg_hudless", "TAA_FG_HUDLESS", true)) {
        FfxResourceDescription hd;
        memset(&hd, 0, sizeof(hd));
        hd.type     = FFX_RESOURCE_TYPE_TEXTURE2D;
        hd.format   = ffxGetSurfaceFormatVK(s.hudlessFmt);
        hd.width    = s.hudlessW;
        hd.height   = s.hudlessH;
        hd.depth    = 1;
        hd.mipCount = 1;
        hd.flags    = FFX_RESOURCE_FLAGS_NONE;
        hd.usage    = FFX_RESOURCE_USAGE_READ_ONLY;
        fd.currentBackBuffer_HUDLess = ffxGetResourceVK(s.hudless, hd, nullptr, FFX_RESOURCE_STATE_PIXEL_COMPUTE_READ);
        if ((s.hudlessUsed++ % 600) == 0)
            trace("FG HUDLESS: interpolating from the pre-overlay backbuffer (%llu dispatches so far, "
                  "%u backbuffer passes in the last frame - the copy is taken after the first).",
                  (unsigned long long)s.hudlessUsed, s.swapPassesLastFrame);
    }
    // ---- RENDER SIZE MUST MATCH THE DILATED RESOURCES, NOT THE CONTEXT.
    //
    // The context was created with the DISPLAY extent as its max render size (a
    // safe upper bound that needs neither FSR3 nor the real render size). But
    // the dilated depth and motion vectors handed to this dispatch are the
    // UPSCALER's, produced at its RENDER resolution - 2953x1661 here, not the
    // 3840x2160 display. Passing the display size as renderSize told the
    // interpolation to read a 3840-wide field out of a 2953-wide resource, and
    // the out-of-bounds sample is a compute shader reading unmapped memory:
    // "DEVICE_LOST, Type: Compute" one frame after the first dispatch.
    //
    // The dilated depth's own description carries the right extent, so it is the
    // authority - the resources and the size cannot disagree if the size comes
    // from the resource.
    // The dilated inputs have their own size; the legacy upscaler keeps its old
    // render size across an FSR toggle while the prep pass follows the scene,
    // and a mismatch here warps every vector.
    if (ownInputs) {
        fd.renderSize.width   = fgprep::state().w ? fgprep::state().w : s.renderW;
        fd.renderSize.height  = fgprep::state().h ? fgprep::state().h : s.renderH;
    } else {
        fd.renderSize.width   = u.sharedDesc[1].width  ? u.sharedDesc[1].width  : s.renderW;
        fd.renderSize.height  = u.sharedDesc[1].height ? u.sharedDesc[1].height : s.renderH;
    }
    fd.interpolationRect  = p->interpolationRect;

    fd.opticalFlowVector  = od.opticalFlowVector;
    fd.opticalFlowSceneChangeDetection = od.opticalFlowSCD;
    fd.opticalFlowBufferSize.width  = s.ofVectorDesc.width;
    fd.opticalFlowBufferSize.height = s.ofVectorDesc.height;
    fd.opticalFlowScale.x = 1.0f / (float)s.dispW;
    fd.opticalFlowScale.y = 1.0f / (float)s.dispH;
    fd.opticalFlowBlockSize = 8;

    // The descriptions are asked of the INTERPOLATION context rather than
    // restated here. It is the authority on what it consumes, it answers
    // without an upscaler existing, and a description written out by hand is
    // one that silently stops matching after an SDK update.
    if (ownInputs) {
        FfxFrameInterpolationSharedResourceDescriptions sh;
        memset(&sh, 0, sizeof(sh));
        if (ffxFrameInterpolationGetSharedResourceDescriptions(&s.fiCtx, &sh) == FFX_OK) {
            fgprep::State &g = fgprep::state();
            // The COMPLETED slot, never the one in flight. readySlot is set only
            // after a submit finishes, so this can never be a texture the GPU is
            // still writing.
            const uint32_t r = (uint32_t)g.readySlot;
            fd.reconstructedPrevDepth =
                ffxGetResourceVK(g.prevDepth[r], sh.reconstructedPrevNearestDepth.resourceDescription,
                                 nullptr, FFX_RESOURCE_STATE_UNORDERED_ACCESS);
            fd.dilatedDepth =
                ffxGetResourceVK(g.dilDepth[r], sh.dilatedDepth.resourceDescription,
                                 nullptr, FFX_RESOURCE_STATE_UNORDERED_ACCESS);
            // Ours is RGBA16F (.z = trust); FFX's own description says RG16F and the view it
            // builds from the description must match the image, so override the format only.
            FfxResourceDescription dmvDesc = sh.dilatedMotionVectors.resourceDescription;
            dmvDesc.format = FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT;
            fd.dilatedMotionVectors =
                ffxGetResourceVK(g.dilMv[r], dmvDesc,
                                 nullptr, FFX_RESOURCE_STATE_UNORDERED_ACCESS);
            static bool told = false;
            if (!told) {
                told = true;
                trace("FG: interpolation is reading OUR dilated depth/motion and "
                      "reconstructed previous depth - the FSR3 upscaler is no "
                      "longer in the frame-generation path.");
            }
        }
    } else {
        fd.reconstructedPrevDepth = ffxGetResourceVK(u.shared[0], u.sharedDesc[0], nullptr,
                                                     FFX_RESOURCE_STATE_UNORDERED_ACCESS);
        fd.dilatedDepth           = ffxGetResourceVK(u.shared[1], u.sharedDesc[1], nullptr,
                                                     FFX_RESOURCE_STATE_UNORDERED_ACCESS);
        fd.dilatedMotionVectors   = ffxGetResourceVK(u.shared[2], u.sharedDesc[2], nullptr,
                                                     FFX_RESOURCE_STATE_UNORDERED_ACCESS);
    }

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

    {
        static bool said = false;
        if (!said) {
            said = true;
            trace("FG DISPATCH: present=%p output=%p ofVec=%p ofScd=%p "
                  "recPrev=%p dilDepth=%p dilMV=%p hudless=%p distort=%p "
                  "render=%ux%u disp=%ux%u",
                  fd.currentBackBuffer.resource, fd.output.resource,
                  fd.opticalFlowVector.resource, fd.opticalFlowSceneChangeDetection.resource,
                  fd.reconstructedPrevDepth.resource, fd.dilatedDepth.resource,
                  fd.dilatedMotionVectors.resource, fd.currentBackBuffer_HUDLess.resource,
                  fd.distortionField.resource,
                  fd.renderSize.width, fd.renderSize.height,
                  fd.displaySize.width, fd.displaySize.height);
        }
    }
    rc = ffxFrameInterpolationDispatch(&s.fiCtx, &fd);
    if (rc != FFX_OK) {
        static bool said = false;
        if (!said) { said = true; trace("FG: interpolation dispatch failed (%d)", (int)rc); }
        presentRealFrame(p, "dispatch failed");
        return rc;
    }
    fgAuditRecord(s, p, fd, ownInputs);
    if ((s.dispatches++ % 300) == 0)
        trace("FG: %llu interpolated frames dispatched (%ux%u)",
              (unsigned long long)s.dispatches, s.dispW, s.dispH);
    return FFX_OK;
}

} // namespace fg
