#pragma once
// ---- SIDE-CAR FSR3: THE DILATION HELPER FOR FRAME GENERATION.
//
// Frame interpolation needs dilatedDepth, dilatedMotionVectors and
// reconstructedPrevDepth. Those are a by-product of the FSR3 upscaler's
// reconstruct-and-dilate pass - there is no "dilate only" entry point, so the
// full upscaler runs, its colour output is thrown away, and only the three
// shared resources it leaves in fsr3::state().shared[] are harvested.
//
// WHY A SEPARATE COMMAND BUFFER. Recording that FSR3 dispatch into X-Plane's own
// render command buffer while the frame-generation proxy swapchain is live is a
// measured device-lost: "virtual address 0x0, Type: Compute", isolated exactly
// by taa.fsr_run=0 (skip the in-render dispatch -> stable for minutes; run it ->
// crash at ~30 s). This records FSR3 into a command buffer this layer owns and
// submits it itself, so the upscaler never touches X-Plane's command stream.
//
// WHY AT PRESENT, ON THE GAME QUEUE. Present runs once per frame with no command
// buffer recording, and X-Plane's render for the frame has already been
// submitted to the game queue by then. Submitting the side-car to that SAME
// queue orders it after the render, so the scene colour, depth and motion
// vectors it reads are the finished frame - no cross-queue race, no borrowed
// fence. The submit is waited on, so the shared resources are complete before
// the interpolation callback reads them later in the same present.
#include <vulkan/vulkan.h>
#include <string.h>
#include <mutex>
#include "fsr3_backend_impl.h"
#include "depth_copy.h"
#include "fg_backend.h"

namespace fgdilate {

struct State {
    bool ready  = false;
    bool failed = false;
    VkDevice        device = VK_NULL_HANDLE;
    VkCommandPool   pool   = VK_NULL_HANDLE;
    VkCommandBuffer cb     = VK_NULL_HANDLE;
    VkFence         fence  = VK_NULL_HANDLE;
    // Per-slot buffers and fences for the double-buffered prepare path. The
    // single cb/fence above still serve the legacy upscaler-harvest route.
    VkCommandBuffer cb2[2]    = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkFence         fence2[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    bool            submitted[2] = { false, false };
    VkQueue         queue  = VK_NULL_HANDLE;
    uint32_t        queueFamily = 0;

    // Throwaway upscale output. FSR3 must be given somewhere to write its
    // display-resolution result; nothing reads it - only the shared dilated
    // resources matter - so it is one image reused every frame.
    VkImage         outImg = VK_NULL_HANDLE;
    VkDeviceMemory  outMem = VK_NULL_HANDLE;
    uint32_t        outW = 0, outH = 0;
    // Set when a fence wait timed out: the submit is still running and the
    // buffer must not be touched until it completes.
    bool            inFlight = false;
    uint64_t        timeouts = 0;
    VkFormat        outFmt = VK_FORMAT_R16G16B16A16_SFLOAT;

    // Resolved once from the device. Only the handful the side-car records with.
    PFN_vkBeginCommandBuffer  begin  = nullptr;
    PFN_vkEndCommandBuffer    end    = nullptr;
    PFN_vkResetCommandBuffer  reset  = nullptr;
    PFN_vkQueueSubmit         submit = nullptr;
    PFN_vkWaitForFences       waitFences = nullptr;
    PFN_vkResetFences         resetFences = nullptr;
    PFN_vkCmdPipelineBarrier  barrier = nullptr;
    PFN_vkCmdBindPipeline     bindPipe = nullptr;
    PFN_vkCmdBindDescriptorSets bindSets = nullptr;
    // Needed by the self-contained prepare pass: it pushes its own constants
    // and clears the scatter target every frame.
    PFN_vkCmdPushConstants    pushConst = nullptr;
    PFN_vkCmdClearColorImage  clearImg  = nullptr;
    PFN_vkCmdDispatch         dispatch = nullptr;

    uint64_t runs = 0;
    std::mutex lock;
};

inline State &state() { static State s; return s; }

// Build the command pool, buffer, fence, queue handle and throwaway output.
// gameFamily is the queue family X-Plane renders on (see fgPickQueues); the
// side-car submits to the same family so its work orders after the render.
inline bool ensure(VkDevice device, VkPhysicalDevice phys,
                   PFN_vkGetDeviceProcAddr gdpa,
                   PFN_vkGetPhysicalDeviceMemoryProperties getMemProps,
                   uint32_t gameFamily, VkQueue gameQueue,
                   uint32_t outW, uint32_t outH)
{
    State &s = state();
    std::lock_guard<std::mutex> guard(s.lock);
    if (s.failed) return false;
    if (s.ready && s.outW == outW && s.outH == outH) return true;
    if (s.ready) return true;   // extent change is handled by teardown elsewhere
    if (!device || !gdpa || !gameQueue || !outW || !outH) return false;

    s.device      = device;
    s.queue       = gameQueue;
    s.queueFamily = gameFamily;

    PFN_vkCreateCommandPool      createPool =
        (PFN_vkCreateCommandPool)gdpa(device, "vkCreateCommandPool");
    PFN_vkAllocateCommandBuffers allocCb =
        (PFN_vkAllocateCommandBuffers)gdpa(device, "vkAllocateCommandBuffers");
    PFN_vkCreateFence            createFence =
        (PFN_vkCreateFence)gdpa(device, "vkCreateFence");
    s.begin       = (PFN_vkBeginCommandBuffer)gdpa(device, "vkBeginCommandBuffer");
    s.end         = (PFN_vkEndCommandBuffer)gdpa(device, "vkEndCommandBuffer");
    s.reset       = (PFN_vkResetCommandBuffer)gdpa(device, "vkResetCommandBuffer");
    s.submit      = (PFN_vkQueueSubmit)gdpa(device, "vkQueueSubmit");
    s.waitFences  = (PFN_vkWaitForFences)gdpa(device, "vkWaitForFences");
    s.resetFences = (PFN_vkResetFences)gdpa(device, "vkResetFences");
    s.barrier     = (PFN_vkCmdPipelineBarrier)gdpa(device, "vkCmdPipelineBarrier");
    s.bindPipe    = (PFN_vkCmdBindPipeline)gdpa(device, "vkCmdBindPipeline");
    s.bindSets    = (PFN_vkCmdBindDescriptorSets)gdpa(device, "vkCmdBindDescriptorSets");
    s.pushConst   = (PFN_vkCmdPushConstants)gdpa(device, "vkCmdPushConstants");
    s.clearImg    = (PFN_vkCmdClearColorImage)gdpa(device, "vkCmdClearColorImage");
    s.dispatch    = (PFN_vkCmdDispatch)gdpa(device, "vkCmdDispatch");

    if (!createPool || !allocCb || !createFence || !s.begin || !s.end || !s.reset ||
        !s.submit || !s.waitFences || !s.resetFences || !s.barrier || !s.bindPipe ||
        !s.bindSets || !s.dispatch || !s.pushConst || !s.clearImg) {
        trace("FG DILATE: a required device entry point is missing - side-car off.");
        s.failed = true; return false;
    }

    VkCommandPoolCreateInfo pci;
    memset(&pci, 0, sizeof(pci));
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = gameFamily;
    if (createPool(device, &pci, nullptr, &s.pool) != VK_SUCCESS) {
        trace("FG DILATE: vkCreateCommandPool failed - side-car off.");
        s.failed = true; return false;
    }

    VkCommandBufferAllocateInfo ai;
    memset(&ai, 0, sizeof(ai));
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = s.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (allocCb(device, &ai, &s.cb) != VK_SUCCESS) {
        trace("FG DILATE: vkAllocateCommandBuffers failed - side-car off.");
        s.failed = true; return false;
    }

    VkFenceCreateInfo fci;
    memset(&fci, 0, sizeof(fci));
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (createFence(device, &fci, nullptr, &s.fence) != VK_SUCCESS) {
        trace("FG DILATE: vkCreateFence failed - side-car off.");
        s.failed = true; return false;
    }

    // A buffer and a fence PER SLOT for the double-buffered prepare path. Two
    // of each is what lets a frame be submitted without waiting for it: the
    // only thing ever waited on is the submit into this slot from two frames
    // ago, which has long since finished.
    ai.commandBufferCount = 2;
    if (allocCb(device, &ai, s.cb2) != VK_SUCCESS) {
        trace("FG DILATE: per-slot command buffers failed - side-car off.");
        s.failed = true; return false;
    }
    for (int k = 0; k < 2; ++k)
        if (createFence(device, &fci, nullptr, &s.fence2[k]) != VK_SUCCESS) {
            trace("FG DILATE: per-slot fence %d failed - side-car off.", k);
            s.failed = true; return false;
        }

    // Throwaway output at display extent, R16F, storage-capable. Reuses the
    // frame-gen backend's image helper so the format and memory choice match.
    FfxResourceDescription od;
    memset(&od, 0, sizeof(od));
    od.type   = FFX_RESOURCE_TYPE_TEXTURE2D;
    od.format = FFX_SURFACE_FORMAT_R16G16B16A16_FLOAT;
    od.width  = outW;
    od.height = outH;
    od.depth  = 1;
    od.mipCount = 1;
    od.usage  = FFX_RESOURCE_USAGE_UAV;
    if (!fg::fgCreateImage(device, phys, od, &s.outImg, &s.outMem, gdpa, getMemProps)) {
        trace("FG DILATE: could not allocate the throwaway output image - side-car off.");
        s.failed = true; return false;
    }
    s.outW = outW; s.outH = outH;

    s.ready = true;
    trace("FG DILATE: side-car ready - FSR3 will run on its own command buffer "
          "(queue family %u), output discarded, only the dilated depth/motion "
          "vectors kept for interpolation.", gameFamily);
    return true;
}

// Record depth-copy + FSR3 into the side-car's own command buffer, submit it to
// the game queue, and wait. On return, fsr3::state().shared[0..2] hold this
// frame's reconstructedPrevDepth / dilatedDepth / dilatedMotionVectors.
//
// colour/depth/mv are X-Plane's scene images; their CURRENT layouts are passed
// so FSR3's own barriers transition from the truth rather than a guess.
inline bool run(VkImage colour, VkFormat colourFmt, VkImageLayout colourLayout,
                VkImage depthRaw, VkFormat depthRawFmt, VkImageLayout depthRawLayout,
                VkImage mv, VkFormat mvFmt,
                uint32_t renderW, uint32_t renderH,
                float jitterX, float jitterY,
                float mvScaleX, float mvScaleY,
                bool reset, bool reverseZ)
{
    State &s = state();
    std::lock_guard<std::mutex> guard(s.lock);
    if (!s.ready || s.failed) return false;
    // A previous submit may not have finished. Never reset or re-record a
    // command buffer while the GPU still owns it - poll once, and skip this
    // frame if it is not done.
    if (s.inFlight) {
        if (s.waitFences(s.device, 1, &s.fence, VK_TRUE, 0) != VK_SUCCESS)
            return false;
        s.inFlight = false;
    }
    if (colour == VK_NULL_HANDLE || depthRaw == VK_NULL_HANDLE || mv == VK_NULL_HANDLE)
        return false;
    if (!fsr3::state().ready || fsr3::state().failed) return false;
    if (!depthcopy::state().ready) return false;

    if (s.reset(s.cb, 0) != VK_SUCCESS) return false;

    VkCommandBufferBeginInfo bi;
    memset(&bi, 0, sizeof(bi));
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (s.begin(s.cb, &bi) != VK_SUCCESS) return false;

    // depth-copy first: it turns X-Plane's depth-format image into the plain
    // R32_SFLOAT the upscaler reads, and leaves it GENERAL.
    depthcopy::record(s.bindPipe, s.bindSets, s.dispatch, s.barrier, s.cb, depthRawLayout);

    // ---- OUR OWN PREPARE PASS, IF IT IS UP.
    //
    // This is the whole point of fg_prepare: the three textures interpolation
    // needs, computed directly, instead of running an entire FSR3 upscaler and
    // discarding its output to harvest them. When it is available the upscaler
    // is skipped completely here - which is what removes it, and its ~1 GB,
    // from the frame-generation path.
    if (fgprep::state().ready && !fgprep::state().failed) {
        // ---- DO NOT WAIT ON WORK WE JUST SUBMITTED.
        //
        // The wait below used to be immediate: submit the pass, then block the
        // CPU until the GPU finished it. That is a full sync point every frame -
        // nothing overlaps across it, so CPU time and GPU time add instead of
        // running together. It was tolerable when this ran a whole FSR3
        // upscaler and the work dwarfed the stall; with the work down to one
        // compute dispatch, the stall IS the cost.
        //
        // So the pass writes into the slot interpolation is not reading, and
        // the only thing waited on is the submit from the PREVIOUS frame -
        // which has had an entire frame to finish and is normally already
        // signalled, so the wait returns immediately.
        fgprep::State &g = fgprep::state();
        const uint32_t slot = g.writeSlot;

        // The previous submit into THIS slot (two frames ago) must be done
        // before the buffer is re-recorded. Long finished in practice; the wait
        // is a correctness guard, not a stall.
        if (s.submitted[slot]) {
            const VkResult fw2 = s.waitFences(s.device, 1, &s.fence2[slot], VK_TRUE,
                                              1000ull * 1000ull * 1000ull);
            if (fw2 != VK_SUCCESS) { s.inFlight = true; return false; }
            s.submitted[slot] = false;
        }
        s.inFlight = false;

        // The prologue above began s.cb for the legacy upscaler route. This
        // path uses its own per-slot buffer, so it records the depth copy and
        // the prepare pass into THAT one instead - a buffer must be begun
        // before anything is recorded into it, and s.cb is not the buffer being
        // submitted here.
        if (s.reset(s.cb2[slot], 0) != VK_SUCCESS) return false;
        VkCommandBufferBeginInfo bi2;
        memset(&bi2, 0, sizeof(bi2));
        bi2.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi2.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (s.begin(s.cb2[slot], &bi2) != VK_SUCCESS) return false;
        depthcopy::record(s.bindPipe, s.bindSets, s.dispatch, s.barrier,
                          s.cb2[slot], depthRawLayout);

        fgprep::record(s.bindPipe, s.bindSets, s.dispatch, s.barrier,
                       s.pushConst, s.clearImg, s.cb2[slot],
                       mvScaleX, mvScaleY, reverseZ, slot);
        if (s.end(s.cb2[slot]) != VK_SUCCESS) return false;
        s.resetFences(s.device, 1, &s.fence2[slot]);
        VkSubmitInfo si2;
        memset(&si2, 0, sizeof(si2));
        si2.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si2.commandBufferCount = 1;
        si2.pCommandBuffers = &s.cb2[slot];
        if (s.submit(s.queue, 1, &si2, s.fence2[slot]) != VK_SUCCESS) return false;
        s.submitted[slot] = true;

        // What interpolation may read is the slot completed LAST frame, not the
        // one now in flight. Publishing it here is what keeps FFX off a texture
        // the GPU is still writing.
        g.readySlot = (int32_t)(1u - slot);
        g.writeSlot = 1u - slot;
        ++g.runs;
        ++s.runs;
        if (s.runs == 1 || (s.runs % 600) == 0)
            trace("FG DILATE: %llu side-car dispatches (own prepare pass, "
                  "double-buffered - the CPU no longer waits on the GPU).",
                  (unsigned long long)s.runs);
        return true;
    }

    // Then the upscaler. Output is the throwaway image; the shared dilated
    // resources are the point. Camera constants match the in-render path.
    fsr3::dispatch(s.barrier, s.cb,
                   colourLayout, VK_IMAGE_LAYOUT_GENERAL,
                   colour, colourFmt,
                   depthcopy::state().image, VK_FORMAT_R32_SFLOAT,
                   mv, mvFmt,
                   s.outImg, s.outFmt,
                   jitterX, jitterY,
                   mvScaleX, mvScaleY,
                   16.6f, reset,
                   0.1f, 100000.0f, 1.0472f);

    if (s.end(s.cb) != VK_SUCCESS) return false;

    s.resetFences(s.device, 1, &s.fence);
    VkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &s.cb;
    if (s.submit(s.queue, 1, &si, s.fence) != VK_SUCCESS) return false;
    // Synchronous: the interpolation callback reads shared[] later this present,
    // so the dilation must be finished on the GPU before we return.
    //
    // ---- AND THE TIMEOUT IS NOT ADVISORY.
    //
    // This ignored the result. A timeout then meant the command buffer was
    // STILL EXECUTING while the next present called reset() on it, which is
    // undefined behaviour and reaches the driver as a device-lost.
    //
    // It is not a theoretical window. Loading an aircraft drops the sim to a
    // few frames a second while textures stream, and a one-second wait at 3 fps
    // is a wait that can genuinely expire - which is exactly when the sim died.
    //
    // On timeout the buffer is left alone and the side-car stands down until a
    // later wait succeeds. Skipping a frame of dilation costs one uninterpolated
    // frame; resetting a live command buffer costs the session.
    const VkResult fw = s.waitFences(s.device, 1, &s.fence, VK_TRUE,
                                     1000ull * 1000ull * 1000ull);
    if (fw != VK_SUCCESS) {
        s.inFlight = true;
        if (++s.timeouts <= 3 || (s.timeouts % 100) == 0)
            trace("FG DILATE: fence wait returned %d (%llu so far) - the dilation "
                  "is still on the GPU. Standing down rather than resetting a "
                  "command buffer that is executing.",
                  (int)fw, (unsigned long long)s.timeouts);
        return false;
    }
    s.inFlight = false;

    ++s.runs;
    if (s.runs == 1 || (s.runs % 300) == 0)
        trace("FG DILATE: %llu side-car dispatches - dilated depth/MV produced "
              "off X-Plane's command buffer.", (unsigned long long)s.runs);
    return true;
}

} // namespace fgdilate
