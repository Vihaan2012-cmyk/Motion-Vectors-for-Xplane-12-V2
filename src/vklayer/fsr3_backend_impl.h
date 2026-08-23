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

// ---- NAME EACH PIPELINE BEFORE IT IS BUILT.
//
// Context creation does not return, and it builds ten compute pipelines from
// the blobs tools/ffx_permute.cpp generated. The interface is a table of
// function pointers we own, so substituting fpCreatePipeline turns "it dies in
// there somewhere" into "it dies on pass N".
//
// Written to the marker file rather than through trace(): whatever goes wrong
// takes the thread with it, and anything buffered dies with it.
static FfxCreatePipelineFunc g_realCreatePipeline = nullptr;
static FfxCreateBackendContextFunc  g_realCreateBackendContext = nullptr;
static FfxGetDeviceCapabilitiesFunc g_realGetDeviceCapabilities = nullptr;

// These two run before any pipeline is built, and fpGetDeviceCapabilities is
// the first code to reach our renamed Vulkan forwarders - the newest thing in
// the path, and so the first suspect.
static FfxErrorCode tracedCreateBackendContext(FfxInterface *bi, FfxEffect effect,
                                               FfxEffectBindlessConfig *cfg,
                                               FfxUInt32 *effectContextId)
{
    {
        FILE *m = fopen("C:/Users/bansa/AppData/Local/Temp/fsr3_mark.txt", "a");
        if (m) { fprintf(m, "  fpCreateBackendContext enter%c", 10);
                 fflush(m); fclose(m); }
    }
    const FfxErrorCode rc = g_realCreateBackendContext
        ? g_realCreateBackendContext(bi, effect, cfg, effectContextId)
        : FFX_ERROR_BACKEND_API_ERROR;
    {
        FILE *m = fopen("C:/Users/bansa/AppData/Local/Temp/fsr3_mark.txt", "a");
        if (m) { fprintf(m, "  fpCreateBackendContext returned %d%c", (int)rc, 10);
                 fflush(m); fclose(m); }
    }
    return rc;
}

static FfxErrorCode tracedGetDeviceCapabilities(FfxInterface *bi,
                                                FfxDeviceCapabilities *caps)
{
    {
        FILE *m = fopen("C:/Users/bansa/AppData/Local/Temp/fsr3_mark.txt", "a");
        if (m) { fprintf(m, "  fpGetDeviceCapabilities enter%c", 10);
                 fflush(m); fclose(m); }
    }
    const FfxErrorCode rc = g_realGetDeviceCapabilities
        ? g_realGetDeviceCapabilities(bi, caps)
        : FFX_ERROR_BACKEND_API_ERROR;
    {
        FILE *m = fopen("C:/Users/bansa/AppData/Local/Temp/fsr3_mark.txt", "a");
        if (m) { fprintf(m, "  fpGetDeviceCapabilities returned %d%c", (int)rc, 10);
                 fflush(m); fclose(m); }
    }
    return rc;
}


static FfxErrorCode tracedCreatePipeline(
    FfxInterface *backendInterface, FfxEffect effect, FfxPass pass,
    uint32_t permutationOptions, const FfxPipelineDescription *pipelineDescription,
    FfxUInt32 effectContextId, FfxPipelineState *outPipeline)
{
    {
        FILE *m = fopen("C:/Users/bansa/AppData/Local/Temp/fsr3_mark.txt", "a");
        if (m) { fprintf(m, "  pipeline: effect=%d pass=%d perm=0x%x%c",
                         (int)effect, (int)pass, permutationOptions, 10);
                 fflush(m); fclose(m); }
    }
    if (!g_realCreatePipeline) return FFX_ERROR_BACKEND_API_ERROR;
    const FfxErrorCode rc = g_realCreatePipeline(backendInterface, effect, pass,
                                                permutationOptions,
                                                pipelineDescription,
                                                effectContextId, outPipeline);
    {
        FILE *m = fopen("C:/Users/bansa/AppData/Local/Temp/fsr3_mark.txt", "a");
        if (m) { fprintf(m, "  pipeline: pass=%d returned %d%c",
                         (int)pass, (int)rc, 10);
                 fflush(m); fclose(m); }
    }
    return rc;
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
    std::lock_guard<std::mutex> guard(s.lock);
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
    // Stored in State, not on the stack: the interface keeps the POINTER.
    memset(&s.vkCtx, 0, sizeof(s.vkCtx));
    s.vkCtx.vkDevice         = device;
    s.vkCtx.vkPhysicalDevice = phys;
    s.vkCtx.vkDeviceProcAddr = gdpa;

    // ---- EVERY STEP TRACES BEFORE IT RUNS.
    //
    // The first attempt died with NO FSR3 output whatever, which says only that
    // it failed somewhere before the first trace - and there was no trace until
    // after the scratch query, so that is most of the function. Naming the step
    // before doing it turns "it crashed" into "it crashed HERE", which is the
    // difference between one run and five.
    const size_t maxContexts = 2;
    trace("FSR3: step 1 - ffxGetScratchMemorySizeVK(phys=%p)", (void*)phys);
    s.scratchSize = ffxGetScratchMemorySizeVK(phys, maxContexts);
    trace("FSR3: step 1 OK - scratch %u bytes", (unsigned)s.scratchSize);
    if (!s.scratchSize) {
        trace("FSR3: scratch size query returned 0 - backend unavailable");
        s.failed = true; return false;
    }
    // ---- ZEROED. malloc ALONE IS THE BUG.
    //
    // CreateBackendContextVK reads its own state straight out of this buffer
    // before initialising anything:
    //
    //     BackendContext_VK* backendContext =
    //         (BackendContext_VK*)backendInterface->scratchBuffer;
    //     if (!backendContext->refCount) { ...set everything up... }
    //
    // With malloc'd memory refCount is whatever was on the heap. Non-zero, and
    // the entire initialisation block is SKIPPED - the function then runs on
    // uninitialised pointers. maxEffectContexts is garbage too, so the array
    // sizes derived from it are garbage, and the memsets that follow write
    // wherever those sizes reach.
    //
    // That is why it was entered and never returned, and why it presented first
    // as a hang and then as a crash once a lock stopped serialising it.
    s.scratch = calloc(1, s.scratchSize);
    if (!s.scratch) { s.failed = true; return false; }

    trace("FSR3: step 2 - ffxGetInterfaceVK(device=%p gdpa=%p)",
          (void*)device, (void*)gdpa);
    FfxErrorCode rc = ffxGetInterfaceVK(&s.iface, ffxGetDeviceVK(&s.vkCtx),
                                        s.scratch, s.scratchSize, maxContexts);
    trace("FSR3: step 2 returned %d", (int)rc);
    // Substituted before the context is created, which is what builds the
    // pipelines. The original is kept and called through, so behaviour is
    // unchanged apart from the two lines each pass now writes.
    g_realCreatePipeline = s.iface.fpCreatePipeline;
    s.iface.fpCreatePipeline = tracedCreatePipeline;
    g_realCreateBackendContext  = s.iface.fpCreateBackendContext;
    g_realGetDeviceCapabilities = s.iface.fpGetDeviceCapabilities;
    s.iface.fpCreateBackendContext  = tracedCreateBackendContext;
    s.iface.fpGetDeviceCapabilities = tracedGetDeviceCapabilities;
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

    // ---- MARKERS STRAIGHT TO DISK, FLUSHED, AROUND THE CALL.
    //
    // The contradiction this settles: step 3 is entered once, its return is
    // never traced, ensure() is never called again - which should be impossible
    // unless it returned - and the render thread that calls it keeps presenting
    // at 37 fps. All three cannot hold, and trace() is the one component
    // involved in all three, so it is the one to take out of the question.
    //
    // A separate file, opened and closed per write, with fflush: if "after"
    // appears the call returns and the fault is in the reporting; if it does
    // not, the call really does not come back and the render thread is somehow
    // not the caller.
    {
        FILE *m = fopen("C:/Users/bansa/AppData/Local/Temp/fsr3_mark.txt", "a");
        if (m) { fprintf(m, "before ContextCreate %ux%u -> %ux%u%c",
                         renderW, renderH, outW, outH, 10);
                 fflush(m); fclose(m); }
    }
    trace("FSR3: step 3 - ffxFsr3UpscalerContextCreate %ux%u -> %ux%u",
          renderW, renderH, outW, outH);
    rc = ffxFsr3UpscalerContextCreate(&s.ctx, &cd);
    {
        FILE *m = fopen("C:/Users/bansa/AppData/Local/Temp/fsr3_mark.txt", "a");
        if (m) { fprintf(m, "after ContextCreate rc=%d%c", (int)rc, 10);
                 fflush(m); fclose(m); }
    }
    trace("FSR3: step 3 returned %d", (int)rc);
    if (rc != FFX_OK) {
        trace("FSR3: ffxFsr3UpscalerContextCreate failed (%d)", (int)rc);
        s.failed = true; return false;
    }

    FfxFsr3UpscalerSharedResourceDescriptions sh;
    memset(&sh, 0, sizeof(sh));
    trace("FSR3: step 4 - shared resource descriptions");
    if (ffxFsr3UpscalerGetSharedResourceDescriptions(&s.ctx, &sh) != FFX_OK) {
        trace("FSR3: shared resource descriptions unavailable");
        s.failed = true; return false;
    }
    // ---- THE ISOLATION OUTPUT.
    //
    // Same extent and format as X-Plane's, so FSR3 cannot tell the difference.
    // Enabled with TAA_FSR3_OWN_OUTPUT=1; off, the real image is used.
    if (getenv("TAA_FSR3_OWN_OUTPUT")) {
        FfxCreateResourceDescription od;
        memset(&od, 0, sizeof(od));
        od.resourceDescription = describeTex2D(outW, outH,
            FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT, FFX_RESOURCE_USAGE_UAV);
        if (createShared(device, phys, od, 0, gdpa, getMemProps)) {
            s.ownOut = s.shared[0];
            s.shared[0] = VK_NULL_HANDLE;   // slot reused below for the real one
            trace("FSR3: ISOLATION MODE - writing our own %ux%u image, not "
                  "X-Plane's. Nothing will reach the screen; this only asks "
                  "whether writing the sim's image is what kills it.",
                  outW, outH);
        }
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

// A colour or depth layout transition, recorded into X-Plane's command buffer.
//
// FFX barriers FROM the state it is told a resource is in. Telling it
// COMPUTE_READ while the image is still a colour attachment is not a hint that
// gets corrected - it is a wrong barrier, and the first dispatch pays for it.
inline void barrier(PFN_vkCmdPipelineBarrier cmdBarrier, VkCommandBuffer cb,
                    VkImage img, VkImageAspectFlags aspect,
                    VkImageLayout from, VkImageLayout to,
                    VkAccessFlags srcAccess, VkAccessFlags dstAccess)
{
    if (!cmdBarrier || img == VK_NULL_HANDLE) return;
    VkImageMemoryBarrier b;
    memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    b.subresourceRange.aspectMask = aspect;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    cmdBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// ---- ONE FRAME.
//
// Recorded into X-Plane's own command buffer, at the point its upscale would
// have run, so ordering against the scene render and everything downstream is
// the sim's own and needs no synchronisation of ours.
inline bool dispatch(PFN_vkCmdPipelineBarrier cmdBarrier,
                     VkCommandBuffer cb,
                     VkImageLayout colorLayout, VkImageLayout depthLayout,
                     VkImage colorImg, VkFormat colorFmt,
                     VkImage depthImg, VkFormat depthFmt,
                     VkImage mvImg,    VkFormat mvFmt,
                     VkImage outImg,   VkFormat outFmt,
                     float jitterX, float jitterY,
                     float mvScaleX, float mvScaleY,
                     float deltaMs, bool reset,
                     float camNear, float camFar, float fovY)
{
    State &s = state();
    // Same lock as ensure(). A dispatch that begins while the context is still
    // being built - or two dispatches from different recording threads - is the
    // race this fixes.
    std::lock_guard<std::mutex> guard(s.lock);
    if (!s.ready || s.failed) return false;
    if (colorImg == VK_NULL_HANDLE || depthImg == VK_NULL_HANDLE ||
        mvImg == VK_NULL_HANDLE || outImg == VK_NULL_HANDLE) return false;

    // ---- MAKE THE DECLARED STATES TRUE BEFORE DECLARING THEM.
    //
    // The three shared resources have never been touched, so they are still
    // UNDEFINED on the first frame; FFX expects GENERAL. Transitioning from
    // UNDEFINED discards contents, which is correct exactly once - after that
    // they hold data FSR3 wrote and must be left alone.
    if (!s.sharedReady) {
        s.sharedReady = true;
        for (int i = 0; i < 3; ++i)
            barrier(cmdBarrier, cb, s.shared[i], VK_IMAGE_ASPECT_COLOR_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                    0, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }

    // ---- ORDER X-PLANE'S WRITES BEFORE FSR3'S READS.
    //
    // FSR3 reads the scene colour, the depth and our motion vectors, all of
    // which X-Plane (and this layer) wrote earlier in the SAME frame. Recording
    // FFX's passes into X-Plane's command buffer does not order them: without a
    // barrier the reads may run before the writes land, which validation does
    // not catch and the GPU does.
    //
    // A GLOBAL MEMORY BARRIER, not per-image transitions. The earlier version
    // transitioned images from layouts they were not in - undefined behaviour,
    // and destructive - because the true layout at this point is not something
    // this layer reliably knows. A memory barrier needs no layout at all: it
    // orders access, which is the part that was actually missing.
    if (cmdBarrier) {
        VkMemoryBarrier mb;
        memset(&mb, 0, sizeof(mb));
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                           VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                           VK_ACCESS_SHADER_WRITE_BIT |
                           VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        cmdBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                   1, &mb, 0, nullptr, 0, nullptr);
    }

    // ---- THE INPUTS ARE NOT TRANSITIONED. THEY ARE ALREADY RIGHT.
    //
    // This runs at X-Plane's own upscale dispatch, and X-Plane has just bound
    // the scene colour as a sampled texture for that upscale - so it is already
    // in SHADER_READ_ONLY_OPTIMAL, which is exactly what
    // FFX_RESOURCE_STATE_COMPUTE_READ means to the VK backend.
    //
    // The previous version transitioned it FROM COLOR_ATTACHMENT_OPTIMAL. An
    // oldLayout that is not the image's actual layout is undefined behaviour -
    // the driver is entitled to discard the contents - and that is destructive
    // rather than merely wrong. It dispatched once and the sim died shortly
    // after.
    //
    // Declaring the state and leaving the image alone is both correct and less
    // work. If a layout ever genuinely disagrees, the answer is to learn what
    // it is, not to assert one.

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
    if (s.ownOut != VK_NULL_HANDLE) outImg = s.ownOut;
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
    // ---- MOTION VECTOR SCALE. THE UNITS, STATED ONCE.
    //
    // This layer's velocity is in UV, not clip space - taa.comp fetches history
    // at literally
    //
    //     hUv = uv + vec2(vel.x, pc.velYSign * vel.y)
    //
    // so the vector already points from the current pixel to where that surface
    // was, which is the direction FSR3 wants. Converting UV to pixels is
    // therefore just the render size.
    //
    // The Y SIGN is the part that would otherwise be wrong. X-Plane draws with
    // a negative-height viewport, so velYSign is -1 and the resolve flips Y on
    // every fetch. FSR3 has no such hook, so the flip has to live in the scale.
    //
    // Both numbers are computed by the CALLER from taaVelScale() and
    // taaVelYSign() - the same accessors the resolve uses - so the two cannot
    // drift apart. Passing them in rather than recomputing here is the whole
    // point: a units disagreement does not fail, it smears, and smearing looks
    // like a temporal bug rather than a units bug.
    d.motionVectorScale.x = mvScaleX;
    d.motionVectorScale.y = mvScaleY;

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

    // ---- EVERY ONE OF THE FIRST FEW, BEFORE AND AFTER.
    //
    // The count stops at exactly 1 on every run, which is deterministic rather
    // than a race. Two possibilities remain and they need separating: the
    // SECOND CPU call dies, or the GPU dies executing the first. A marker
    // before and after each call says which - "before 2" with no "after 2"
    // means the CPU call; "after 1" with no "before 2" means the frame never
    // came back.
    if (s.dispatches < 400) {
        FILE *m = fopen("C:/Users/bansa/AppData/Local/Temp/fsr3_mark.txt", "a");
        if (m) { fprintf(m, "before dispatch %llu%c",
                         (unsigned long long)(s.dispatches + 1), 10);
                 fflush(m); fclose(m); }
    }
    const FfxErrorCode rc = ffxFsr3UpscalerContextDispatch(&s.ctx, &d);
    if (s.dispatches < 400) {
        FILE *m = fopen("C:/Users/bansa/AppData/Local/Temp/fsr3_mark.txt", "a");
        if (m) { fprintf(m, "after dispatch %llu rc=%d%c",
                         (unsigned long long)(s.dispatches + 1), (int)rc, 10);
                 fflush(m); fclose(m); }
    }
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

    // And order FSR3's writes before whatever X-Plane does next with the
    // output image - the same argument in the other direction.
    if (cmdBarrier) {
        VkMemoryBarrier mb;
        memset(&mb, 0, sizeof(mb));
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                           VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                           VK_ACCESS_TRANSFER_READ_BIT;
        cmdBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                   1, &mb, 0, nullptr, 0, nullptr);
    }

    ++s.dispatches;
    if (s.dispatches == 1 || (s.dispatches % 30) == 0)
        trace("FSR3: %llu dispatches, %ux%u -> %ux%u",
              (unsigned long long)s.dispatches, s.renderW, s.renderH, s.outW, s.outH);
    return true;
}

} // namespace fsr3
