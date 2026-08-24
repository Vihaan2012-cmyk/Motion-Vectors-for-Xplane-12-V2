// The VRAM management system, removed.
//
// WHAT THIS FILE USED TO BE
//
// 2236 lines implementing the architecture in "Engineering Plan for Vram
// System.txt": residency zones (GREEN through CRITICAL), a filtered budget
// estimator, a migration tick that moved allocations between heaps, a recycle
// pool that held freed device memory for reuse, per-resource churn tracking,
// memory-priority tagging, upload budgeting and an upload content cache.
//
// WHY IT IS GONE
//
// It was switched off in practice long before it was removed, and it never
// paid for itself while on. The recycle pool ran for a whole session reporting
// "0 hits 0 misses" - it was never consulted at all, because poolTake()
// refused any allocation carrying a pNext, and 77 of 84 refusals were a benign
// VkMemoryAllocateFlagsInfo. That is representative rather than unusual: the
// system had a large surface, a lot of state, and very little demonstrated
// effect, and every measurement taken through it had to first establish
// whether the system itself was interfering.
//
// The thing that actually works is not here and never was. Capping X-Plane's
// texture downscaling is a BINARY PATCH in plugin.cpp -
// patchTextureScaleFloor() raises the pager's 1/16 floor, and
// patchTextureBudgetReserve() adjusts its reserve. Those are standalone, they
// are staying, and they do not depend on a line of this file.
//
// WHY A SHIM RATHER THAN DELETION
//
// layer.cpp calls into this namespace from 79 sites. Tearing all of those out
// is a large, error-prone edit across a 12000-line file whose interception
// paths are the most delicate part of the mod, in exchange for nothing a
// no-op cannot deliver. So the API survives as inert stubs: the system is
// genuinely gone - no state, no threads, no allocation interception, no
// per-frame work - while the call sites keep compiling.
//
// The counters that remain read zero on purpose. A reader who finds
// vram_used_mb sitting at 0 should conclude the system is absent, which is
// true, rather than that it is running and finding nothing.
//
// The history is in git. If any part of this is wanted back, it is one file at
// one revision, not a reconstruction.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace vram {

enum Zone { GREEN = 0, YELLOW, ORANGE, RED, CRITICAL };

inline const char *zoneName(int) { return "OFF"; }

// Kept as a type because layer.cpp declares one. Nothing fills it.
struct PrioChain { VkMemoryPriorityAllocateInfoEXT info; VkMemoryAllocateInfo ai; };

struct Config {
    bool  enable;
    bool  shape;
    bool  recycle;
    bool  priority;
    bool  governor;
    float budgetAlpha;
    int   reserveMB[5];
    int   deflateMB;
    int   deflateFrames;
    int   recycleMaxMB;
    int   recycleHoldFrames;
    int   uploadBudgetMB[5];
    int   uploadMaxHold;
    float teleportM;
    int   teleportFrames;
    int   traceEvery;
    float speedReserve;
    bool  adaptive;
    bool  uploadCache;
    int   warmupFrames;
    int   warmupMB;
    int   holdMaxMB;
    int   lookaheadFrames;
    int   ageFrames;
};

// enable=false is the whole removal, expressed once. Any call site that still
// asks is told the system is off rather than being lied to.
static Config cfg = { false, false, false, false, false,
                      0.0f, {0,0,0,0,0}, 0, 0, 0, 0, {0,0,0,0,0}, 0,
                      0.0f, 0, 0, 0.0f, false, false, 0, 0, 0, 0, 0 };

// ---- Reported state. All zero, all of the time.
static int      zone         = GREEN;
static uint64_t rawBudget    = 0, rawUsage = 0;
static uint64_t lastReported = 0;
static uint64_t upBytesLast  = 0;
static uint64_t sparseBinds  = 0;
static uint64_t heldNow      = 0;
static uint64_t g_poolBytes  = 0;
static uint64_t flushOnWait  = 0;
static std::atomic<uint64_t> allocLatWorstUs(0);

static std::atomic<uint64_t> allocFails(0);

// ---- Lifecycle.
inline void bindDevice(VkDevice, VkPhysicalDevice,
                       PFN_vkGetPhysicalDeviceMemoryProperties2,
                       PFN_vkFreeMemory, PFN_vkUnmapMemory, PFN_vkQueueSubmit,
                       PFN_vkSetDeviceMemoryPriorityEXT, bool, bool,
                       const VkPhysicalDeviceMemoryProperties *, uint32_t) {}
inline void shutdown() {}

// ---- Allocation and binding. The interception is what is being removed, so
// these observe nothing and change nothing.
inline void noteAlloc(VkDeviceMemory, const VkMemoryAllocateInfo *) {}
inline void onBind(VkDeviceMemory, int, bool, bool = false) {}
inline void noteFreeGone(VkDeviceMemory) {}
inline bool prioTag(const VkMemoryAllocateInfo *, PrioChain *) { return false; }

// ---- The recycle pool. Never taking and never holding is exactly what it
// achieved in practice while it existed.
inline bool poolTake(const VkMemoryAllocateInfo *, VkDeviceMemory *) { return false; }
inline bool poolHold(VkDeviceMemory) { return false; }

// ---- Images and buffers.
inline void noteImageCreate(VkImage, uint32_t, uint32_t, uint32_t, uint32_t,
                            uint64_t, int, VkImageUsageFlags, uint32_t, bool,
                            uint32_t, bool *hotOut)
{
    if (hotOut) *hotOut = false;
}
inline void noteImageDestroy(VkImage) {}
inline void noteImageMem(VkImage, VkDeviceMemory) {}
inline void noteImageUse(VkImage, bool = false) {}
inline void noteBufBind(VkBuffer, VkDeviceMemory, VkDeviceSize) {}
inline void noteBufferGone(VkBuffer) {}
inline bool churnHot(VkImage) { return false; }
inline uint8_t protectionOf(VkImage) { return 0; }

// baseDrop passes straight through. Refusing to refine is the correct
// behaviour for an absent system: whatever the caller decided stands.
inline uint32_t refineDrop(uint32_t baseDrop, uint32_t, uint32_t,
                           uint32_t, uint32_t, bool)
{
    return baseDrop;
}

// ---- Uploads and content caching.
inline const uint8_t *bufferBytes(VkBuffer, VkDeviceSize, VkDeviceSize) { return nullptr; }
inline uint64_t contentHash(const uint8_t *, uint64_t) { return 0; }
inline bool cacheUpload(VkImage, uint32_t, uint64_t, uint64_t) { return false; }
inline void contentInvalidate(VkImage) {}
inline void retainPayload(uint32_t, uint32_t, uint32_t, uint32_t,
                          const uint8_t *, uint64_t) {}
inline void noteUploadRegion(uint64_t, uint32_t) {}
inline void chargeCopy(VkCommandBuffer, uint64_t, uint8_t = 0) {}

// ---- Mapping.
inline void noteMap(VkDeviceMemory, VkDeviceSize, VkDeviceSize, void *) {}
inline void noteUnmap(VkDeviceMemory) {}

// ---- Frame and queue accounting.
inline void noteQueue(uint32_t, VkQueue) {}
inline void notePipelines(uint32_t, uint64_t) {}
inline void notePipelineBind() {}
inline void noteDescriptorUpdates(uint32_t) {}
inline void noteDescriptorAllocs(uint32_t) {}
inline void noteCamera(float, float, float) {}
inline void ledgerTotal(uint64_t) {}
inline void ledgerRt(uint64_t) {}

// Returning false means "not handled" - the caller submits normally, which is
// the path that runs when the governor is absent.
inline bool onSubmit(VkQueue, uint32_t, const VkSubmitInfo *, VkFence, VkResult *)
{
    return false;
}
inline void onPresent(float, float = -1.0f) {}
inline void touchFences(uint32_t, const VkFence *) {}
inline void touchSemaphores(uint32_t, const VkSemaphore *, const uint64_t *) {}
inline void flushAll(uint64_t *) {}

// ---- Reporting.
inline bool emergency() { return false; }
inline const char *typeFlagsText(uint32_t) { return ""; }
inline void shapeReport(const VkPhysicalDeviceMemoryProperties *,
                        VkPhysicalDeviceMemoryBudgetPropertiesEXT *, float) {}

} // namespace vram
