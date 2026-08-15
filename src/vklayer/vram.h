#pragma once

// ============================================================ THE VRAM SYSTEM
//
// The Engineering Plan (VRAM_STUDY.md maps it against the studied engine),
// implemented in the layer. X-Plane HAS the mechanisms - VMA, a budget query,
// a defragmenter, a global-scale texture pager - but it does not DO the plan:
// no zones, no hysteresis worth the name (measured: scale 1.0 -> 2.0 -> 1.0 in
// one second with headroom collapsing to zero), no memory priorities (the
// extension is enabled and never used), no deferred-free recycling, no upload
// pacing, no prediction. This file is those pieces, built where the layer sits:
// between every call X-Plane makes and the driver that answers it.
//
// The actuators, and why each lives where it does:
//
//   BUDGET SHAPER   - vkGetPhysicalDeviceMemoryProperties2 is the ONE number
//                     X-Plane's whole memory stack reads (VMA's heapBudget
//                     feeds the memory controller feeds the texture pager's
//                     evaluate()). Shaping it steers the engine without
//                     touching a byte of engine state.
//   ZONES           - GREEN/YELLOW/ORANGE/RED/CRITICAL from projected usage,
//                     with split enter/exit thresholds and dwell, so every
//                     other actuator has one word to key from instead of five
//                     raw numbers.
//   RECYCLE POOL    - deferred free + reuse of device-local allocations. The
//                     plan's SS19 retire queue, at the only level the layer
//                     truly owns.
//   PRIORITY ENGINE - VkMemoryPriorityAllocateInfoEXT at allocation and
//                     vkSetDeviceMemoryPriorityEXT at first bind. Measured:
//                     the engine enables VK_EXT_memory_priority and never sets
//                     one, so under WDDM pressure the driver demotes render
//                     targets as blindly as stale autogen. This fixes that.
//   UPLOAD GOVERNOR - caps bytes submitted per frame on the transfer-only
//                     queue under pressure, holding whole submissions in FIFO
//                     and releasing them on a budget - with flush triggers on
//                     every wait path so nothing can deadlock (SS below).
//   PREDICTOR      -  camera speed and teleport detection from the camera
//                     delta the layer already receives per frame. No datarefs,
//                     no art controls - this system runs entirely from what
//                     passes through the layer.
//   EMERGENCY       - on VK_ERROR_OUT_OF_DEVICE_MEMORY: flush the recycle
//                     pool (real bytes back to the driver), deflate the shaped
//                     budget so the engine's own pager cuts on its next
//                     evaluate, then let the caller retry.
//
// Everything is live-controllable (vram.* keys in taa_live.ini), defaults ON -
// the env-gated-feature trap is documented three times in learnings.md and is
// not being walked into a fourth.

#include <vulkan/vulkan.h>
#include <windows.h>
#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace vram {

// ------------------------------------------------------------------ config
// Read through live:: (file > env > default). Cached per frame in onPresent so
// hot paths never touch the live mutex.
struct Config {
    bool  enable;            // vram.enable          master switch
    bool  shape;             // vram.shape           budget shaping on/off
    bool  recycle;           // vram.recycle         deferred-free pool
    bool  priority;          // vram.priority        memory priorities
    bool  governor;          // vram.governor        upload pacing
    float budgetAlpha;       // vram.budget_alpha    low-pass per query
    int   reserveMB[5];      // vram.reserve_g/y/o/r/c
    int   deflateMB;         // vram.deflate_mb      emergency budget cut
    int   deflateFrames;     // vram.deflate_frames  how long the cut lasts
    int   recycleMaxMB;      // vram.recycle_max_mb
    int   recycleHoldFrames; // vram.recycle_hold_frames
    int   uploadBudgetMB[5]; // vram.upload_g/y/o/r/c   per-frame transfer cap
    int   uploadMaxHold;     // vram.upload_max_hold    frames a submit may wait
    float teleportM;         // vram.teleport_m         camera jump threshold
    int   teleportFrames;    // vram.teleport_frames    zone bias duration
    int   traceEvery;        // vram.trace_every        heartbeat cadence
};
static Config cfg = { true, true, true, true, true,
                      0.02f, {128,256,384,512,768}, 512, 600,
                      256, 180, {0,0,64,24,8}, 2,
                      2000.0f, 900, 600 };

// ------------------------------------------------------------------ zones
enum Zone { GREEN = 0, YELLOW, ORANGE, RED, CRITICAL };
static const char *zoneName(int z)
{
    switch (z) {
        case GREEN:  return "GREEN";  case YELLOW:   return "YELLOW";
        case ORANGE: return "ORANGE"; case RED:      return "RED";
        default:     return "CRITICAL";
    }
}

// ------------------------------------------------------------------ state
static std::mutex m;                 // guards everything below except atomics

struct Dev {
    VkDevice         dev  = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    PFN_vkGetPhysicalDeviceMemoryProperties2 memProps2 = nullptr;  // down-chain
    PFN_vkFreeMemory                 freeMemory  = nullptr;        // down-chain
    PFN_vkQueueSubmit                queueSubmit = nullptr;        // down-chain
    PFN_vkSetDeviceMemoryPriorityEXT setPriority = nullptr;        // down-chain
    bool priorityExt = false;        // VK_EXT_memory_priority enabled on device
    bool pageableExt = false;        // VK_EXT_pageable_device_local_memory
    VkPhysicalDeviceMemoryProperties memProps;   // cached, for type properties
    uint32_t transferFamily = ~0u;   // the transfer-ONLY family, if any
} dev;

// Queues, recorded from vkGetDeviceQueue. Only transfer-only-family queues are
// governed; everything else passes through untouched.
static std::set<VkQueue> g_transferQueues;
// Per-governed-queue submit serialisation. VkQueue is externally synchronised;
// once the layer can submit to a queue from two threads (the app's and a flush
// trigger's), the layer must provide the exclusion the spec assumes.
static std::map<VkQueue, std::mutex*> g_queueLock;

// ---- telemetry (atomics where hot paths touch them)
static uint64_t heapSize = 0;        // device-local heap, bytes
static uint64_t rawBudget = 0, rawUsage = 0;      // driver's latest
static double   filteredBudget = 0.0;             // low-passed
static uint64_t lastReported = 0;                 // monotone-under-free clamp
static uint64_t lastLedgerBytes = 0;              // app trend detection
static std::atomic<uint64_t> ledgerBytesNow(0);   // fed from layer ledger
static int      zone = GREEN;
static int      zoneDwell = 0;                    // frames since last change
static int      deflateLeft = 0;                  // frames of emergency cut
static int      teleportLeft = 0;                 // frames of teleport bias
static uint64_t frameIndex = 0;

static std::atomic<uint64_t> upBytesFrame(0);     // charged at record time
static uint64_t upBytesLast = 0, upBytesPeak = 0;
static std::atomic<uint64_t> allocsTotal(0), freesTotal(0), allocFails(0);
static std::atomic<uint64_t> allocLatWorstUs(0);
static uint64_t recycleHits = 0, recycleMisses = 0, recycleFlushes = 0;
static uint64_t prioAllocTagged = 0, prioBindSet = 0, prioZoneMoves = 0;
static uint64_t heldNow = 0, heldBytesNow = 0;
static uint64_t flushOnWait = 0, flushOnDep = 0, flushOnAge = 0, flushOnPresent = 0;
static uint64_t sparseBinds = 0;
static double   camSpeedEma = 0.0;
static uint64_t teleports = 0;

// ---- recycle pool
struct Held { VkDeviceMemory mem; uint64_t size; uint32_t type; uint64_t frame; };
static std::vector<Held> g_pool;                  // FIFO by frame
static uint64_t g_poolBytes = 0;
// Every allocation we saw, so free knows whether recycling is legal.
struct AllocRec { uint64_t size; uint32_t type; bool plain; };  // plain: pNext==NULL
static std::map<VkDeviceMemory, AllocRec> g_allocs;

// ---- priority engine
struct BlockPrio { float best; bool streamedTexOnly; bool demoted; };
static std::map<VkDeviceMemory, BlockPrio> g_blockPrio;

// ---- upload governor
struct HeldOne {
    std::vector<VkSemaphore>          waits;
    std::vector<VkPipelineStageFlags> waitStages;
    std::vector<VkCommandBuffer>      cbs;
    std::vector<VkSemaphore>          sigs;
    bool hasTimeline = false;
    std::vector<uint64_t> waitVals, sigVals;
};
struct HeldSubmit {
    VkQueue q; VkFence fence; uint64_t bytes; uint64_t frame;
    std::vector<HeldOne> subs;
};
static std::vector<HeldSubmit> g_heldSubmits;                 // FIFO
static std::set<VkFence>     g_heldFences;                    // fast trigger check
static std::map<VkSemaphore, uint64_t> g_heldSignals;         // sem -> max value (0=binary)
static std::map<VkCommandBuffer, uint64_t> g_cbBytes;         // recorded copy bytes
static uint64_t g_frameUploadSpent = 0;                       // released this frame

// ------------------------------------------------------------------ helpers
static bool typeRecyclable(uint32_t typeIndex)
{
    // Device-local and NOT host-visible: nothing can hold a persistent map on
    // it, so handing the same VkDeviceMemory to a new owner is legal (contents
    // of a fresh allocation are undefined anyway). VMA persistently maps every
    // host-visible block it owns, and a second vkMapMemory on a recycled one
    // would be a spec violation - so host-visible never enters the pool.
    if (typeIndex >= dev.memProps.memoryTypeCount) return false;
    VkMemoryPropertyFlags f = dev.memProps.memoryTypes[typeIndex].propertyFlags;
    return (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
          !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
}

static float prioOfCat(int cat, bool streamed)
{
    // Categories are layer.cpp's VRAM_* enum, passed as int so this header
    // stays self-contained. Order of importance is the plan's SS8: the frame's
    // own attachments above everything, streamed scenery below everything that
    // is expensive to lose.
    switch (cat) {
        case 1: /*RT*/      case 2: /*DEPTH*/   return 1.0f;
        case 3: /*STORAGE*/                     return 0.9f;
        case 6: /*BUF_UNIFORM*/                 return 0.8f;
        case 5: /*BUF_GEOM*/                    return 0.7f;
        case 0: /*TEX*/     return streamed ? 0.35f : 0.6f;
        case 7: /*BUF_STAGING*/                 return 0.2f;
        default:                                return 0.5f;
    }
}

static uint64_t zoneUploadBudget()
{
    int mb = cfg.uploadBudgetMB[zone];
    return mb <= 0 ? ~0ull : (uint64_t)mb * 1048576ull;
}

// ============================================================ device binding
// Called from TAA_CreateDevice once the down-chain pointers exist.
static void bindDevice(VkDevice d, VkPhysicalDevice p,
                       PFN_vkGetPhysicalDeviceMemoryProperties2 mp2,
                       PFN_vkFreeMemory freeMem, PFN_vkQueueSubmit qsub,
                       PFN_vkSetDeviceMemoryPriorityEXT setPrio,
                       bool prioExt, bool pageExt,
                       const VkPhysicalDeviceMemoryProperties *props,
                       uint32_t transferOnlyFamily)
{
    std::lock_guard<std::mutex> g(m);
    dev.dev = d; dev.phys = p; dev.memProps2 = mp2;
    dev.freeMemory = freeMem; dev.queueSubmit = qsub;
    dev.setPriority = setPrio;
    dev.priorityExt = prioExt; dev.pageableExt = pageExt && setPrio != nullptr;
    if (props) dev.memProps = *props;
    dev.transferFamily = transferOnlyFamily;
    trace("VRAMSYS: bound - priority ext %s, pageable ext %s, transfer-only "
          "family %s (%u)",
          prioExt ? "ON" : "off",
          dev.pageableExt ? "ON" : "off",
          transferOnlyFamily == ~0u ? "NONE" : "found", transferOnlyFamily);
}

static void noteQueue(uint32_t family, VkQueue q)
{
    if (family != dev.transferFamily || dev.transferFamily == ~0u) return;
    std::lock_guard<std::mutex> g(m);
    if (g_transferQueues.insert(q).second) {
        g_queueLock[q] = new std::mutex;
        trace("VRAMSYS: transfer queue %p (family %u) is under the governor",
              (void*)q, family);
    }
}

// ============================================================ ledger feed
// layer.cpp's ledger calls these so the shaper can see the app's trend without
// this header reaching into layer globals.
static void ledgerTotal(uint64_t bytes) { ledgerBytesNow.store(bytes); }

// ============================================================ upload charge
static void chargeCopy(VkCommandBuffer cb, uint64_t bytes)
{
    if (!cfg.enable) return;
    upBytesFrame.fetch_add(bytes);
    std::lock_guard<std::mutex> g(m);
    g_cbBytes[cb] += bytes;
}

// ============================================================ recycle pool
// Try to satisfy an allocation from the pool. Returns true and fills *out on a
// hit; the caller skips the driver entirely.
static bool poolTake(const VkMemoryAllocateInfo *ai, VkDeviceMemory *out)
{
    if (!cfg.enable || !cfg.recycle || !ai || ai->pNext) return false;
    std::lock_guard<std::mutex> g(m);
    for (size_t i = 0; i < g_pool.size(); ++i) {
        if (g_pool[i].type != ai->memoryTypeIndex) continue;
        if (g_pool[i].size != ai->allocationSize)  continue;   // exact only
        *out = g_pool[i].mem;
        g_poolBytes -= g_pool[i].size;
        g_pool.erase(g_pool.begin() + (long)i);
        ++recycleHits;
        // The block re-enters service; its record stays in g_allocs (same
        // size/type, still plain), so a later free can hold it again.
        return true;
    }
    ++recycleMisses;
    return false;
}

// Free path. Returns true if the block was pooled (caller must NOT free it).
static bool poolHold(VkDeviceMemory mem)
{
    if (!cfg.enable || !cfg.recycle) return false;
    std::lock_guard<std::mutex> g(m);
    std::map<VkDeviceMemory, AllocRec>::iterator it = g_allocs.find(mem);
    if (it == g_allocs.end() || !it->second.plain) return false;
    if (!typeRecyclable(it->second.type)) return false;
    if (zone >= RED) return false;                     // pressure: give it back
    uint64_t cap = (uint64_t)cfg.recycleMaxMB * 1048576ull;
    if (g_poolBytes + it->second.size > cap) return false;
    Held h; h.mem = mem; h.size = it->second.size;
    h.type = it->second.type; h.frame = frameIndex;
    g_pool.push_back(h);
    g_poolBytes += h.size;
    g_blockPrio.erase(mem);          // ownership identity ends here
    return true;
}

// Drop pooled blocks - aged ones each frame, everything under pressure.
// Called with m NOT held; takes it, collects, frees outside.
static void poolTrim(bool everything)
{
    std::vector<VkDeviceMemory> toFree;
    {
        std::lock_guard<std::mutex> g(m);
        for (size_t i = 0; i < g_pool.size();) {
            bool aged = frameIndex - g_pool[i].frame >
                        (uint64_t)cfg.recycleHoldFrames;
            if (everything || aged || zone >= RED) {
                toFree.push_back(g_pool[i].mem);
                g_poolBytes -= g_pool[i].size;
                g_allocs.erase(g_pool[i].mem);
                g_pool.erase(g_pool.begin() + (long)i);
                ++recycleFlushes;
            } else ++i;
        }
    }
    if (!toFree.empty() && dev.freeMemory && dev.dev)
        for (size_t i = 0; i < toFree.size(); ++i)
            dev.freeMemory(dev.dev, toFree[i], nullptr);
}

// ============================================================ priority engine
// Allocation-time tag. Builds a shallow copy of the app's info with our
// priority struct spliced at the head of the chain; the app's own chain is
// untouched (it is const and stays const).
struct PrioChain { VkMemoryPriorityAllocateInfoEXT info; VkMemoryAllocateInfo ai; };
static bool prioTag(const VkMemoryAllocateInfo *ai, PrioChain *pc)
{
    if (!cfg.enable || !cfg.priority || !dev.priorityExt) return false;
    pc->info.sType = VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT;
    pc->info.pNext = (void*)ai->pNext;
    pc->info.priority = 0.5f;        // neutral until a bind says what it holds
    pc->ai = *ai;
    pc->ai.pNext = &pc->info;
    ++prioAllocTagged;
    return true;
}

static void noteAlloc(VkDeviceMemory mem, const VkMemoryAllocateInfo *ai)
{
    allocsTotal.fetch_add(1);
    if (!ai) return;
    std::lock_guard<std::mutex> g(m);
    AllocRec r; r.size = ai->allocationSize; r.type = ai->memoryTypeIndex;
    r.plain = (ai->pNext == nullptr);
    g_allocs[mem] = r;
}

static void noteFreeGone(VkDeviceMemory mem)
{
    freesTotal.fetch_add(1);
    std::lock_guard<std::mutex> g(m);
    g_allocs.erase(mem);
    g_blockPrio.erase(mem);
}

// First-bind classification: the resource's category names what the block
// holds, and the block takes the highest priority of anything bound into it.
static void onBind(VkDeviceMemory mem, int cat, bool streamed)
{
    if (!cfg.enable || !cfg.priority || !dev.pageableExt || !dev.dev) return;
    float p = prioOfCat(cat, streamed);
    bool  set = false;
    {
        std::lock_guard<std::mutex> g(m);
        BlockPrio &bp = g_blockPrio[mem];
        if (bp.best == 0.0f) {       // fresh entry
            bp.best = p; bp.streamedTexOnly = (cat == 0 && streamed);
            bp.demoted = false; set = true;
        } else {
            if (!(cat == 0 && streamed)) bp.streamedTexOnly = false;
            if (p > bp.best) { bp.best = p; set = true; }
        }
    }
    if (set) { dev.setPriority(dev.dev, mem, p); ++prioBindSet; }
}

// Zone modulation: in RED and worse, push streamed-texture-only blocks further
// down so the driver demotes THOSE first; lift them again in GREEN. Bounded
// per call so a zone flip never stalls a present.
static void prioZoneWalk()
{
    if (!cfg.enable || !cfg.priority || !dev.pageableExt || !dev.dev) return;
    std::vector<std::pair<VkDeviceMemory, float> > moves;
    {
        std::lock_guard<std::mutex> g(m);
        int budget = 64;
        for (std::map<VkDeviceMemory, BlockPrio>::iterator it = g_blockPrio.begin();
             it != g_blockPrio.end() && budget > 0; ++it) {
            if (!it->second.streamedTexOnly) continue;
            if (zone >= RED && !it->second.demoted) {
                it->second.demoted = true;
                moves.push_back(std::make_pair(it->first, 0.2f));
                --budget;
            } else if (zone == GREEN && it->second.demoted) {
                it->second.demoted = false;
                moves.push_back(std::make_pair(it->first, it->second.best));
                --budget;
            }
        }
    }
    for (size_t i = 0; i < moves.size(); ++i)
        dev.setPriority(dev.dev, moves[i].first, moves[i].second);
    prioZoneMoves += moves.size();
}

// ============================================================ upload governor
//
// DEADLOCK-PROOFING, stated once and honoured everywhere: a held submission is
// released BEFORE any code path that could wait on its results -
//   - vkWaitForFences / vkGetFenceStatus / vkResetFences on its fence
//   - vkWaitSemaphores on any semaphore it signals (timeline value compared)
//   - any OTHER submission (any queue) or sparse bind that waits on a
//     semaphore it signals
//   - vkQueueWaitIdle / vkDeviceWaitIdle
//   - age: nothing is held longer than cfg.uploadMaxHold presents
// A submission with a pNext we do not fully understand is never held at all.

static void flushLocked(std::vector<HeldSubmit> &out, VkQueue only /*or null*/)
{
    // Collect FIFO holds (all for a queue, or all queues) under m; the caller
    // submits them outside the lock. FIFO per queue is preserved because the
    // vector is FIFO and we take every match.
    for (size_t i = 0; i < g_heldSubmits.size();) {
        if (only == VK_NULL_HANDLE || g_heldSubmits[i].q == only) {
            out.push_back(g_heldSubmits[i]);
            g_heldSubmits.erase(g_heldSubmits.begin() + (long)i);
        } else ++i;
    }
    if (only == VK_NULL_HANDLE || g_heldSubmits.empty()) {
        if (only == VK_NULL_HANDLE) { g_heldFences.clear(); g_heldSignals.clear(); }
    }
    // Rebuild trigger indices for what remains.
    g_heldFences.clear(); g_heldSignals.clear();
    heldNow = g_heldSubmits.size(); heldBytesNow = 0;
    for (size_t i = 0; i < g_heldSubmits.size(); ++i) {
        heldBytesNow += g_heldSubmits[i].bytes;
        if (g_heldSubmits[i].fence) g_heldFences.insert(g_heldSubmits[i].fence);
        for (size_t s = 0; s < g_heldSubmits[i].subs.size(); ++s) {
            const HeldOne &o = g_heldSubmits[i].subs[s];
            for (size_t k = 0; k < o.sigs.size(); ++k) {
                uint64_t v = o.hasTimeline && k < o.sigVals.size() ? o.sigVals[k] : 0;
                uint64_t &cur = g_heldSignals[o.sigs[k]];
                if (v > cur) cur = v;
            }
        }
    }
}

static void submitHeld(std::vector<HeldSubmit> &list)
{
    for (size_t i = 0; i < list.size(); ++i) {
        HeldSubmit &h = list[i];
        std::vector<VkSubmitInfo> infos(h.subs.size());
        std::vector<VkTimelineSemaphoreSubmitInfo> tls(h.subs.size());
        for (size_t s = 0; s < h.subs.size(); ++s) {
            HeldOne &o = h.subs[s];
            VkSubmitInfo &si = infos[s];
            memset(&si, 0, sizeof(si));
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.waitSemaphoreCount = (uint32_t)o.waits.size();
            si.pWaitSemaphores    = o.waits.empty() ? nullptr : o.waits.data();
            si.pWaitDstStageMask  = o.waitStages.empty() ? nullptr : o.waitStages.data();
            si.commandBufferCount = (uint32_t)o.cbs.size();
            si.pCommandBuffers    = o.cbs.empty() ? nullptr : o.cbs.data();
            si.signalSemaphoreCount = (uint32_t)o.sigs.size();
            si.pSignalSemaphores  = o.sigs.empty() ? nullptr : o.sigs.data();
            if (o.hasTimeline) {
                VkTimelineSemaphoreSubmitInfo &tl = tls[s];
                memset(&tl, 0, sizeof(tl));
                tl.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
                tl.waitSemaphoreValueCount   = (uint32_t)o.waitVals.size();
                tl.pWaitSemaphoreValues      = o.waitVals.empty() ? nullptr : o.waitVals.data();
                tl.signalSemaphoreValueCount = (uint32_t)o.sigVals.size();
                tl.pSignalSemaphoreValues    = o.sigVals.empty() ? nullptr : o.sigVals.data();
                si.pNext = &tl;
            }
        }
        std::mutex *ql = nullptr;
        {
            std::lock_guard<std::mutex> g(m);
            std::map<VkQueue, std::mutex*>::iterator it = g_queueLock.find(h.q);
            if (it != g_queueLock.end()) ql = it->second;
        }
        if (ql) ql->lock();
        if (dev.queueSubmit)
            dev.queueSubmit(h.q, (uint32_t)infos.size(),
                            infos.empty() ? nullptr : infos.data(), h.fence);
        if (ql) ql->unlock();
    }
    list.clear();
}

static void flushAll(uint64_t *counter)
{
    std::vector<HeldSubmit> out;
    {
        std::lock_guard<std::mutex> g(m);
        if (g_heldSubmits.empty()) return;
        flushLocked(out, VK_NULL_HANDLE);
        if (counter) *counter += out.size();
    }
    submitHeld(out);
}

// Trigger: a fence the app is about to wait on / query / reset.
static void touchFences(uint32_t count, const VkFence *fences)
{
    if (!count || !fences) return;
    bool hit = false;
    {
        std::lock_guard<std::mutex> g(m);
        if (g_heldFences.empty()) return;
        for (uint32_t i = 0; i < count && !hit; ++i)
            if (g_heldFences.count(fences[i])) hit = true;
    }
    if (hit) flushAll(&flushOnWait);
}

// Trigger: semaphores the app is about to wait on (vkWaitSemaphores, or the
// wait list of another submission / sparse bind).
static bool anyHeldSignalLocked(uint32_t count, const VkSemaphore *sems,
                                const uint64_t *values)
{
    for (uint32_t i = 0; i < count; ++i) {
        std::map<VkSemaphore, uint64_t>::iterator it = g_heldSignals.find(sems[i]);
        if (it == g_heldSignals.end()) continue;
        if (!values || !it->second) return true;       // binary, or no values
        if (values[i] <= it->second) return true;      // wait <= held signal
        // waiting for a value beyond what we hold still needs our signal to
        // make progress toward it on the same timeline:
        return true;
    }
    return false;
}

static void touchSemaphores(uint32_t count, const VkSemaphore *sems,
                            const uint64_t *values)
{
    if (!count || !sems) return;
    bool hit = false;
    {
        std::lock_guard<std::mutex> g(m);
        if (g_heldSignals.empty()) return;
        hit = anyHeldSignalLocked(count, sems, values);
    }
    if (hit) flushAll(&flushOnDep);
}

// The submit path. Returns true when the layer has fully handled the call and
// *result carries the answer; false means the caller must pass it down.
static bool onSubmit(VkQueue q, uint32_t count, const VkSubmitInfo *submits,
                     VkFence fence, VkResult *result)
{
    if (!cfg.enable) return false;

    // ANY submission's wait list can depend on a held signal - graphics
    // waiting on the transfer timeline is the normal case. Check first,
    // whatever the queue.
    if (submits) {
        bool dep = false;
        {
            std::lock_guard<std::mutex> g(m);
            if (!g_heldSignals.empty())
                for (uint32_t s = 0; s < count && !dep; ++s) {
                    const VkTimelineSemaphoreSubmitInfo *tl = nullptr;
                    for (const VkBaseInStructure *p =
                             (const VkBaseInStructure*)submits[s].pNext;
                         p; p = p->pNext)
                        if (p->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO)
                            tl = (const VkTimelineSemaphoreSubmitInfo*)p;
                    dep = anyHeldSignalLocked(submits[s].waitSemaphoreCount,
                                              submits[s].pWaitSemaphores,
                                              tl ? tl->pWaitSemaphoreValues : nullptr);
                }
        }
        if (dep) flushAll(&flushOnDep);
    }

    // Bytes this submission carries, from the copies recorded into its
    // command buffers. Collected - and ERASED - for every queue, or the map
    // would grow forever on graphics-queue uploads and command-buffer reuse
    // would accumulate stale charges. Unknown buffers count zero - pacing
    // needs an estimate, not an audit.
    uint64_t bytes = 0;
    bool holdable = true;
    for (uint32_t s = 0; s < count && submits; ++s) {
        for (uint32_t c = 0; c < submits[s].commandBufferCount; ++c) {
            std::lock_guard<std::mutex> g(m);
            std::map<VkCommandBuffer, uint64_t>::iterator it =
                g_cbBytes.find(submits[s].pCommandBuffers[c]);
            if (it != g_cbBytes.end()) { bytes += it->second; g_cbBytes.erase(it); }
        }
        // Only a bare chain or a single timeline struct is understood well
        // enough to reconstruct. Anything else must not be held.
        for (const VkBaseInStructure *p =
                 (const VkBaseInStructure*)submits[s].pNext; p; p = p->pNext)
            if (p->sType != VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO)
                holdable = false;
    }

    bool governed;
    {
        std::lock_guard<std::mutex> g(m);
        governed = cfg.governor && g_transferQueues.count(q) != 0;
    }
    if (!governed) return false;

    bool mustQueueBehind;
    uint64_t budget = zoneUploadBudget();
    {
        std::lock_guard<std::mutex> g(m);
        mustQueueBehind = false;
        for (size_t i = 0; i < g_heldSubmits.size(); ++i)
            if (g_heldSubmits[i].q == q) { mustQueueBehind = true; break; }
        bool over = (g_frameUploadSpent + bytes) > budget;
        if ((!over && !mustQueueBehind) || !holdable || zone < ORANGE) {
            // Passes now. If order requires it, everything held for this queue
            // goes first - budget or not, order beats pacing.
            g_frameUploadSpent += bytes;
        } else {
            // Hold it.
            HeldSubmit h; h.q = q; h.fence = fence; h.bytes = bytes;
            h.frame = frameIndex;
            for (uint32_t s = 0; s < count; ++s) {
                HeldOne o;
                const VkSubmitInfo &si = submits[s];
                o.waits.assign(si.pWaitSemaphores,
                               si.pWaitSemaphores + si.waitSemaphoreCount);
                if (si.pWaitDstStageMask)
                    o.waitStages.assign(si.pWaitDstStageMask,
                                        si.pWaitDstStageMask + si.waitSemaphoreCount);
                o.cbs.assign(si.pCommandBuffers,
                             si.pCommandBuffers + si.commandBufferCount);
                o.sigs.assign(si.pSignalSemaphores,
                              si.pSignalSemaphores + si.signalSemaphoreCount);
                for (const VkBaseInStructure *p =
                         (const VkBaseInStructure*)si.pNext; p; p = p->pNext)
                    if (p->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO) {
                        const VkTimelineSemaphoreSubmitInfo *tl =
                            (const VkTimelineSemaphoreSubmitInfo*)p;
                        o.hasTimeline = true;
                        if (tl->pWaitSemaphoreValues)
                            o.waitVals.assign(tl->pWaitSemaphoreValues,
                                              tl->pWaitSemaphoreValues +
                                              tl->waitSemaphoreValueCount);
                        if (tl->pSignalSemaphoreValues)
                            o.sigVals.assign(tl->pSignalSemaphoreValues,
                                             tl->pSignalSemaphoreValues +
                                             tl->signalSemaphoreValueCount);
                    }
                h.subs.push_back(o);
            }
            g_heldSubmits.push_back(h);
            if (fence) g_heldFences.insert(fence);
            for (size_t s = 0; s < h.subs.size(); ++s)
                for (size_t k = 0; k < h.subs[s].sigs.size(); ++k) {
                    uint64_t v = h.subs[s].hasTimeline &&
                                 k < h.subs[s].sigVals.size()
                               ? h.subs[s].sigVals[k] : 0;
                    uint64_t &cur = g_heldSignals[h.subs[s].sigs[k]];
                    if (v > cur) cur = v;
                }
            heldNow = g_heldSubmits.size();
            heldBytesNow += bytes;
            *result = VK_SUCCESS;
            return true;                       // handled: held
        }
    }

    // Passing through - but order first if anything is still held for q.
    if (mustQueueBehind) {
        std::vector<HeldSubmit> out;
        {
            std::lock_guard<std::mutex> g(m);
            flushLocked(out, q);
            flushOnDep += out.size();
        }
        submitHeld(out);
    }

    std::mutex *ql = nullptr;
    {
        std::lock_guard<std::mutex> g(m);
        std::map<VkQueue, std::mutex*>::iterator it = g_queueLock.find(q);
        if (it != g_queueLock.end()) ql = it->second;
    }
    if (ql) ql->lock();
    *result = dev.queueSubmit ? dev.queueSubmit(q, count, submits, fence)
                              : VK_ERROR_INITIALIZATION_FAILED;
    if (ql) ql->unlock();
    return true;
}

// ============================================================ budget shaping
// Applied to every VkPhysicalDeviceMemoryBudgetPropertiesEXT the app receives.
static void shapeReport(const VkPhysicalDeviceMemoryProperties *mp,
                        VkPhysicalDeviceMemoryBudgetPropertiesEXT *b,
                        float legacyScale)
{
    if (!b || !mp) return;
    std::lock_guard<std::mutex> g(m);
    for (uint32_t i = 0; i < mp->memoryHeapCount; ++i) {
        if (!(mp->memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) continue;

        heapSize  = mp->memoryHeaps[i].size;
        rawBudget = b->heapBudget[i];
        rawUsage  = b->heapUsage[i];

        if (!cfg.enable || !cfg.shape) {
            // Legacy scale still honoured with shaping off.
            if (legacyScale > 1.0f) {
                VkDeviceSize want = (VkDeviceSize)(b->heapBudget[i] * legacyScale);
                if (want > heapSize) want = heapSize;
                if (want > b->heapBudget[i]) b->heapBudget[i] = want;
            }
            break;
        }

        // Low-pass. The driver's figure can step hundreds of MB in one query
        // under external pressure; the engine then feeds the step straight
        // into evaluate() and rescales the world. Smooth it at the source.
        if (filteredBudget <= 0.0) filteredBudget = (double)rawBudget;
        filteredBudget += ((double)rawBudget - filteredBudget) * cfg.budgetAlpha;

        uint64_t report = (uint64_t)filteredBudget;

        // Monotone under free: while the app's own footprint is falling, the
        // budget it sees must not fall with it. "Available fell as it freed"
        // is the measured death spiral (V1 log: 2.45 gb of 2.44 gb available,
        // shrinking as textures were released).
        uint64_t led = ledgerBytesNow.load();
        bool appFreeing = led + (16ull << 20) < lastLedgerBytes;   // 16 MB slack
        lastLedgerBytes = led;
        if (appFreeing && lastReported && report < lastReported)
            report = lastReported - lastReported / 1000;           // decay 0.1%

        // Zone reserve - the split thresholds the engine's pager lacks. In
        // GREEN almost everything is offered; each zone up withholds more, so
        // the engine starts adapting BEFORE the wall, not at it.
        uint64_t reserve = (uint64_t)cfg.reserveMB[zone] * 1048576ull;
        report = report > reserve ? report - reserve : 0;

        // Emergency deflate after an allocation failure: make the engine's
        // next evaluate() see genuine scarcity.
        if (deflateLeft > 0) {
            uint64_t cut = (uint64_t)cfg.deflateMB * 1048576ull;
            report = report > cut ? report - cut : 0;
        }

        if (legacyScale > 1.0f) {
            VkDeviceSize want = (VkDeviceSize)(report * legacyScale);
            report = want;
        }

        // Floors and ceilings: never below current usage plus one working set
        // (reporting less than usage trips the engine's overstep alarm and
        // the user-facing "not enough VRAM" dialog), never above the heap.
        uint64_t floorB = rawUsage + (64ull << 20);
        if (report < floorB)  report = floorB;
        if (report > heapSize) report = heapSize;

        lastReported = report;
        b->heapBudget[i] = report;
        break;                        // first device-local heap only
    }
}

// ============================================================ emergency
// Called on VK_ERROR_OUT_OF_DEVICE_MEMORY before any retry. Returns true if
// anything was actually reclaimed (a retry is then worth making).
static bool emergency()
{
    allocFails.fetch_add(1);
    uint64_t before;
    {
        std::lock_guard<std::mutex> g(m);
        before = g_poolBytes;
        deflateLeft = cfg.deflateFrames;
        zone = CRITICAL; zoneDwell = 0;
    }
    flushAll(&flushOnDep);            // held uploads complete, then retire
    poolTrim(true);                   // pooled blocks back to the driver
    trace("VRAMSYS: EMERGENCY - allocation failed; recycle pool flushed "
          "(%.1f MB reclaimed), budget deflated %d MB for %d frames, zone "
          "CRITICAL", before / 1048576.0, cfg.deflateMB, cfg.deflateFrames);
    return before > 0;
}

// ============================================================ per-frame tick
static void refreshConfig()
{
    cfg.enable   = live::onoff("vram.enable",   "TAA_VRAMSYS",         true);
    cfg.shape    = live::onoff("vram.shape",    "TAA_VRAM_SHAPE",      true);
    cfg.recycle  = live::onoff("vram.recycle",  "TAA_VRAM_RECYCLE",    true);
    cfg.priority = live::onoff("vram.priority", "TAA_VRAM_PRIORITY",   true);
    cfg.governor = live::onoff("vram.governor", "TAA_VRAM_GOVERNOR",   true);
    cfg.budgetAlpha       = live::f("vram.budget_alpha", nullptr, 0.02f);
    cfg.reserveMB[GREEN]  = live::i("vram.reserve_g", nullptr, 128);
    cfg.reserveMB[YELLOW] = live::i("vram.reserve_y", nullptr, 256);
    cfg.reserveMB[ORANGE] = live::i("vram.reserve_o", nullptr, 384);
    cfg.reserveMB[RED]    = live::i("vram.reserve_r", nullptr, 512);
    cfg.reserveMB[CRITICAL] = live::i("vram.reserve_c", nullptr, 768);
    cfg.deflateMB      = live::i("vram.deflate_mb",      nullptr, 512);
    cfg.deflateFrames  = live::i("vram.deflate_frames",  nullptr, 600);
    cfg.recycleMaxMB   = live::i("vram.recycle_max_mb",  nullptr, 256);
    cfg.recycleHoldFrames = live::i("vram.recycle_hold_frames", nullptr, 180);
    cfg.uploadBudgetMB[GREEN]  = live::i("vram.upload_g", nullptr, 0);   // 0 = off
    cfg.uploadBudgetMB[YELLOW] = live::i("vram.upload_y", nullptr, 0);
    cfg.uploadBudgetMB[ORANGE] = live::i("vram.upload_o", nullptr, 64);
    cfg.uploadBudgetMB[RED]    = live::i("vram.upload_r", nullptr, 24);
    cfg.uploadBudgetMB[CRITICAL] = live::i("vram.upload_c", nullptr, 8);
    cfg.uploadMaxHold  = live::i("vram.upload_max_hold", nullptr, 2);
    cfg.teleportM      = live::f("vram.teleport_m",      nullptr, 2000.0f);
    cfg.teleportFrames = live::i("vram.teleport_frames", nullptr, 900);
    cfg.traceEvery     = live::i("vram.trace_every",     nullptr, 600);
}

static void zoneUpdate()
{
    // Projected usage: what is resident now plus what the last frame's uploads
    // suggest is arriving. Cheap, monotone with real pressure, and computable
    // without a single readback.
    double basis = filteredBudget > 0 ? filteredBudget : (double)heapSize;
    if (basis <= 0) return;
    double projected = (double)rawUsage + (double)upBytesLast * 2.0;
    double f = projected / basis;

    static const double enter[5] = { 0.0, 0.80, 0.88, 0.94, 0.98 };
    static const double exitT[5] = { 0.0, 0.75, 0.83, 0.90, 0.95 };

    int want = zone;
    while (want < CRITICAL && f >= enter[want + 1]) ++want;    // up: immediate
    while (want > GREEN && f < exitT[want]) --want;            // down: candidate

    if (teleportLeft > 0 && want < YELLOW) want = YELLOW;      // predictive bias

    if (want > zone) { zone = want; zoneDwell = 0;
        trace("VRAMSYS: zone %s (%.0f%% of %.2f GB shaped basis, usage %.2f GB "
              "+ uploads %.1f MB/f)", zoneName(zone), f * 100.0,
              basis / 1073741824.0, rawUsage / 1073741824.0,
              upBytesLast / 1048576.0);
    } else if (want < zone) {
        if (++zoneDwell >= 60) {                                // 60-frame dwell
            zone = want; zoneDwell = 0;
            trace("VRAMSYS: zone %s (%.0f%%)", zoneName(zone), f * 100.0);
        }
    } else zoneDwell = 0;
}

static void onPresent(float camDeltaMeters)
{
    refreshConfig();
    if (!cfg.enable) { flushAll(&flushOnPresent); return; }

    uint64_t fi;
    {
        std::lock_guard<std::mutex> g(m);
        fi = ++frameIndex;
        if (deflateLeft > 0) --deflateLeft;
        if (teleportLeft > 0) --teleportLeft;
        g_frameUploadSpent = 0;
        upBytesLast = upBytesFrame.exchange(0);
        if (upBytesLast > upBytesPeak) upBytesPeak = upBytesLast;
    }

    // Predictor: speed for telemetry, teleports for policy. The camera delta
    // arrives from the layer's own per-frame snapshot - nothing here reads the
    // sim.
    if (camDeltaMeters >= 0.0f) {
        camSpeedEma += (camDeltaMeters - camSpeedEma) * 0.05;
        if (camDeltaMeters > cfg.teleportM) {
            std::lock_guard<std::mutex> g(m);
            teleportLeft = cfg.teleportFrames;
            ++teleports;
            trace("VRAMSYS: teleport (%.0f m in one frame) - zone floor YELLOW "
                  "for %d frames, recycle pool flushed", camDeltaMeters,
                  cfg.teleportFrames);
        }
    }
    if (teleportLeft == cfg.teleportFrames) poolTrim(true);

    // A fresh heap sample, on our own cadence, through the down-chain query -
    // shaping happens in the report hook; this keeps zones honest even when
    // the app is not asking.
    if (dev.memProps2 && dev.phys) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT bud;
        memset(&bud, 0, sizeof(bud));
        bud.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        VkPhysicalDeviceMemoryProperties2 mp2;
        memset(&mp2, 0, sizeof(mp2));
        mp2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        mp2.pNext = &bud;
        dev.memProps2(dev.phys, &mp2);
        std::lock_guard<std::mutex> g(m);
        for (uint32_t i = 0; i < mp2.memoryProperties.memoryHeapCount; ++i) {
            if (!(mp2.memoryProperties.memoryHeaps[i].flags &
                  VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) continue;
            heapSize  = mp2.memoryProperties.memoryHeaps[i].size;
            rawBudget = bud.heapBudget[i];
            rawUsage  = bud.heapUsage[i];
            if (filteredBudget <= 0.0) filteredBudget = (double)rawBudget;
            filteredBudget += ((double)rawBudget - filteredBudget) * cfg.budgetAlpha;
            break;
        }
    }

    { std::lock_guard<std::mutex> g(m); zoneUpdate(); }

    // Governor: age out anything held too long, then release within budget.
    std::vector<HeldSubmit> release;
    {
        std::lock_guard<std::mutex> g(m);
        uint64_t budget = zoneUploadBudget();
        for (size_t i = 0; i < g_heldSubmits.size();) {
            bool aged = fi - g_heldSubmits[i].frame >= (uint64_t)cfg.uploadMaxHold;
            bool fits = g_frameUploadSpent + g_heldSubmits[i].bytes <= budget;
            // FIFO per queue: once one stays, everything younger for that
            // queue stays behind it.
            bool blocked = false;
            for (size_t k = 0; k < i; ++k)
                if (g_heldSubmits[k].q == g_heldSubmits[i].q) { blocked = true; break; }
            if (!blocked && (aged || fits)) {
                g_frameUploadSpent += g_heldSubmits[i].bytes;
                if (aged) ++flushOnAge; else ++flushOnPresent;
                release.push_back(g_heldSubmits[i]);
                g_heldSubmits.erase(g_heldSubmits.begin() + (long)i);
            } else ++i;
        }
        // Rebuild trigger indices.
        std::vector<HeldSubmit> tmp;
        tmp.swap(g_heldSubmits);
        g_heldSubmits.swap(tmp);
        g_heldFences.clear(); g_heldSignals.clear();
        heldNow = g_heldSubmits.size(); heldBytesNow = 0;
        for (size_t i = 0; i < g_heldSubmits.size(); ++i) {
            heldBytesNow += g_heldSubmits[i].bytes;
            if (g_heldSubmits[i].fence) g_heldFences.insert(g_heldSubmits[i].fence);
            for (size_t s = 0; s < g_heldSubmits[i].subs.size(); ++s)
                for (size_t k = 0; k < g_heldSubmits[i].subs[s].sigs.size(); ++k) {
                    const HeldOne &o = g_heldSubmits[i].subs[s];
                    uint64_t v = o.hasTimeline && k < o.sigVals.size()
                               ? o.sigVals[k] : 0;
                    uint64_t &cur = g_heldSignals[o.sigs[k]];
                    if (v > cur) cur = v;
                }
        }
    }
    submitHeld(release);

    poolTrim(false);
    prioZoneWalk();

    if (live::i("vram.report", nullptr, 0)) {
        live::clearOneShot("vram.report");
        std::lock_guard<std::mutex> g(m);
        trace("VRAMSYS REPORT ------------------------------------------------");
        trace("  zone %s  heap %.2f GB  raw budget %.2f GB  usage %.2f GB  "
              "shaped %.2f GB", zoneName(zone), heapSize / 1073741824.0,
              rawBudget / 1073741824.0, rawUsage / 1073741824.0,
              lastReported / 1073741824.0);
        trace("  uploads: %.1f MB last frame, peak %.1f MB;  governor held now "
              "%llu (%.1f MB), flushes wait/dep/age/present %llu/%llu/%llu/%llu",
              upBytesLast / 1048576.0, upBytesPeak / 1048576.0,
              (unsigned long long)heldNow, heldBytesNow / 1048576.0,
              (unsigned long long)flushOnWait, (unsigned long long)flushOnDep,
              (unsigned long long)flushOnAge, (unsigned long long)flushOnPresent);
        trace("  allocs %llu  frees %llu  failures %llu  worst latency %llu us",
              (unsigned long long)allocsTotal.load(),
              (unsigned long long)freesTotal.load(),
              (unsigned long long)allocFails.load(),
              (unsigned long long)allocLatWorstUs.load());
        trace("  recycle: %llu hits  %llu misses  pool %.1f MB in %llu blocks  "
              "%llu flushed",
              (unsigned long long)recycleHits, (unsigned long long)recycleMisses,
              g_poolBytes / 1048576.0, (unsigned long long)g_pool.size(),
              (unsigned long long)recycleFlushes);
        trace("  priority: %llu tagged at alloc  %llu set at bind  %llu zone "
              "moves  (ext %s, pageable %s)",
              (unsigned long long)prioAllocTagged,
              (unsigned long long)prioBindSet,
              (unsigned long long)prioZoneMoves,
              dev.priorityExt ? "on" : "OFF", dev.pageableExt ? "on" : "OFF");
        trace("  camera: %.2f m/frame EMA, %llu teleports;  sparse binds %llu",
              camSpeedEma, (unsigned long long)teleports,
              (unsigned long long)sparseBinds);
        trace("-----------------------------------------------------------------");
    }

    if (cfg.traceEvery > 0 && fi % (uint64_t)cfg.traceEvery == 0)
        trace("VRAMSYS: zone %s  usage %.2f / shaped %.2f GB  up %.1f MB/f  "
              "held %llu  pool %.1f MB  fails %llu",
              zoneName(zone), rawUsage / 1073741824.0,
              lastReported / 1073741824.0, upBytesLast / 1048576.0,
              (unsigned long long)heldNow, g_poolBytes / 1048576.0,
              (unsigned long long)allocFails.load());
}

// Device teardown: everything held goes back before the device dies.
static void shutdown()
{
    flushAll(&flushOnDep);
    poolTrim(true);
}

} // namespace vram
