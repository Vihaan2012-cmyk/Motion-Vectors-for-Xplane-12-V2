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
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
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
    float speedReserve;      // vram.speed_reserve      reserve growth per m/frame
    bool  adaptive;          // vram.adaptive           frame-time feedback
    bool  uploadCache;       // vram.upload_cache       content elision (SS23)
    int   warmupFrames;      // vram.warmup_frames      SS73 progressive fill
    int   warmupMB;          // vram.warmup_mb          extra reserve at frame 0
    int   holdMaxMB;         // vram.hold_max_mb        governor backpressure
    int   lookaheadFrames;   // vram.lookahead          SS85 trend horizon
    int   ageFrames;         // vram.age_frames         SS86 staleness window
};
static Config cfg = { true, true, true, true, true,
                      0.02f, {128,256,384,512,768}, 512, 600,
                      256, 180, {0,0,64,24,8}, 2,
                      2000.0f, 900, 600, 0.01f, true,
                      true, 900, 512, 512, 300, 1800 };

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
    PFN_vkUnmapMemory                unmapMemory = nullptr;        // down-chain
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

// ---- frame-time statistics (plan SS29/30): the primary metric is
// consistency, not average. A ring of present-to-present times feeds an EMA
// for the adaptive upload notch and percentiles for the report.
static double        g_ftRing[512];
static int           g_ftAt = 0, g_ftN = 0;
static LARGE_INTEGER g_lastPresentQpc;
static bool          g_havePresentQpc = false;
static double        frameAvgMs = 0.0;     // EMA, alpha 0.05
static double        frameBestMs = 1e9;    // best recent EMA, decays upward
static int           uploadNotch = 0;      // halves the upload budget per step
static uint64_t      notchCalm = 0;        // clean frames since last notch

// ---- churn (SS50/51): a texture that cycles resident->evicted->resident is
// the classic streaming failure. Identity is the creation shape - the reload
// creates a NEW VkImage, so the handle cannot carry it, but dims+format+mips
// can. A key seen destroyed and re-created within the window is a cycle;
// three cycles make the shape HOT, and hot images take a higher memory
// priority at bind so the driver stops demoting exactly them.
struct ChurnKey {
    uint32_t w, h, fmt, mips;
    bool operator<(const ChurnKey &o) const {
        if (w != o.w) return w < o.w;
        if (h != o.h) return h < o.h;
        if (fmt != o.fmt) return fmt < o.fmt;
        return mips < o.mips;
    }
};
struct ChurnRec {
    uint32_t creates = 0, destroys = 0, cycles = 0;
    uint64_t lastDestroyFrame = 0, bytes = 0;
};
static std::map<ChurnKey, ChurnRec> g_churn;
static std::map<VkImage, ChurnKey>  g_imgKey;
static std::set<VkImage>            g_hotImgs;
static uint64_t churnCycles = 0;

// ---- pipeline creation (SS59): JIT pipeline compiles are a classic stutter
// source. Counted and timed; creations after the first present are "live"
// and the per-frame peak is the number that indicts a hitch.
static std::atomic<uint64_t> pipesTotal(0), pipesLive(0), pipesFrame(0);
static std::atomic<uint64_t> pipeUsTotal(0);
static uint64_t pipesFramePeak = 0, pipesFramePeakAt = 0;

// ---- descriptor churn (SS35): counters only - the engine owns its
// descriptors; the layer's job is to know whether they are a problem.
static std::atomic<uint64_t> descUpdFrame(0), descAllocFrame(0);
static uint64_t descUpdPeak = 0, descAllocPeak = 0;

// ---- duplicate uploads (SS52): the same image mip uploaded twice in one
// frame is wasted PCIe bandwidth. Detected per frame, cleared at present.
static std::set<std::pair<uint64_t, uint32_t> > g_upSeen;
static uint64_t dupUploads = 0;

// ---- render-target bytes (SS34): fed from the ledger each frame; the sum of
// RT + depth + storage is the theoretical upper bound transient aliasing
// could reclaim, reported so the investigation the plan asks for has its
// number.
static uint64_t rtBytesNow = 0;

// ---- persistence (SS77): a small state file carries what a session learned
// - allocation failures ever seen become a standing reserve bias, so a
// machine that has hit the wall keeps more margin from the next launch on.
static int      reserveBiasMB = 0;
static uint64_t failsEver = 0;

// ---- host-pointer and binding tracking, the legal route into upload
// contents. VMA maps its staging blocks through vkMapMemory, which passes
// through the layer - so the pointer the APP holds is recorded here, and
// reading through it at record time is reading our own process memory.
// A second vkMapMemory would be a spec violation; remembering the first is
// not.
struct MapRec  { uint8_t *ptr; VkDeviceSize offset, size; };
static std::map<VkDeviceMemory, MapRec> g_mapped;
struct BufBind { VkDeviceMemory mem; VkDeviceSize offset; };
static std::map<VkBuffer, BufBind> g_bufBind;

// ---- the upload content cache (SS23/24). Two levels:
//   LIVE:  (image, mip) -> content hash. A re-upload of identical bytes into
//          the same subresource is ELIDED - the texels are already there and
//          the PCIe transfer buys nothing.
//   DEAD:  (shape, mip) -> content hash of destroyed images. When the engine
//          reloads a texture from disk after eviction (new VkImage, same
//          shape) and uploads the same bytes, the match is counted: that is
//          the measured cost of the engine's missing CPU cache, stated in
//          bytes rather than argued.
struct MipKey {
    uint64_t img; uint32_t mip;
    bool operator<(const MipKey &o) const {
        if (img != o.img) return img < o.img;
        return mip < o.mip;
    }
};
struct ShapeMip {
    uint32_t w, h, fmt, mip;
    bool operator<(const ShapeMip &o) const {
        if (w != o.w) return w < o.w;
        if (h != o.h) return h < o.h;
        if (fmt != o.fmt) return fmt < o.fmt;
        return mip < o.mip;
    }
};
static std::map<MipKey, uint64_t>   g_liveContent;
static std::map<ShapeMip, uint64_t> g_deadContent;
static uint64_t elidedUploads = 0, elidedBytes = 0;
static uint64_t reloadIdenticalCount = 0, reloadIdenticalBytes = 0;
static uint64_t hostRecycleUnmaps = 0;

// ---- usage-trend prediction (SS85): the direction usage is moving, so
// degradation can begin BEFORE the peak instead of at it.
static double   usageTrendBytes = 0.0;     // EMA of per-frame usage delta
static uint64_t prevRawUsage = 0;

// ---- warmup (SS73): an extra reserve that decays to zero over the first
// frames, so startup fills VRAM progressively instead of spiking into it.
static uint64_t warmupStartFrame = 0;

// ---- aircraft change (SS76): a burst of large preloaded-texture
// destructions is a livery or aircraft swap. Response mirrors a teleport.
static double   bigDestroyScore = 0.0;
static uint64_t aircraftChanges = 0, aircraftChangeCooldown = 0;

// ---- pipeline bind census (SS60): binds per frame, for the usage-frequency
// picture the plan wants specialization decisions fed by.
static std::atomic<uint64_t> pipeBindsFrame(0);
static uint64_t pipeBindsPeak = 0;

// ---- benchmark harness (SS99): vram.bench=1 opens a measurement window,
// vram.bench=0 closes it and dumps every metric the plan lists.
static bool     benchOn = false;
static uint64_t benchStartFrame = 0;
static double   benchFtSum = 0, benchFtWorst = 0;
static std::vector<double> benchFt;
static uint64_t benchVramPeak = 0, benchUpPeak = 0, benchUpTotal = 0;
static uint64_t benchAllocs0 = 0, benchFails0 = 0, benchPipes0 = 0;
static uint64_t benchElide0 = 0, benchReload0 = 0;
static uint64_t benchZoneFrames[5] = {0,0,0,0,0};

// ---- recycle pool
struct Held { VkDeviceMemory mem; uint64_t size; uint32_t type; uint64_t frame; };
static std::vector<Held> g_pool;                  // FIFO by frame
static uint64_t g_poolBytes = 0;
// Every allocation we saw, so free knows whether recycling is legal.
struct AllocRec { uint64_t size; uint32_t type; bool plain; };  // plain: pNext==NULL
static std::map<VkDeviceMemory, AllocRec> g_allocs;

// ---- priority engine
struct BlockPrio { float best; bool streamedTexOnly; bool demoted; bool aged; };
static std::map<VkDeviceMemory, BlockPrio> g_blockPrio;

// ---- per-resource aging (SS21's age weight, SS86/87 recency AND frequency).
// Usage is sampled at descriptor-bind time (1 in 64 binds - aging needs "used
// this minute", not "used this draw"), and blocks whose images have gone
// unused decay to a lower priority so the driver demotes exactly the stale
// ones. Frequently-used resources age at half speed - a texture touched every
// two seconds outranks one touched once a second ago, which is SS87 verbatim.
struct UseRec { uint64_t last; uint32_t count; };
static std::map<VkImage, UseRec>         g_imgLastUse;
static std::map<VkImage, VkDeviceMemory> g_imgMem;
static uint64_t agedDemotions = 0, agedRestores = 0;

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
        // IMG_OTHER is arrays and 3D volumes. Preloaded ones are engine
        // infrastructure (cloud volumes carry STORAGE usage and classify
        // higher already); streamed sampled arrays are scenery mega-texture
        // pages and rank with streamed scenery.
        case 4: /*IMG_OTHER*/ return streamed ? 0.4f : 0.6f;
        default:                                return 0.5f;
    }
}

static uint64_t zoneUploadBudget()
{
    int mb = cfg.uploadBudgetMB[zone];
    if (mb <= 0) {
        // Unlimited by zone - but the adaptive notch still applies: frame-time
        // pressure caps uploads even in GREEN (plan SS29: a theoretically
        // perfect streaming system can still cause stutters).
        if (uploadNotch <= 0) return ~0ull;
        return (uint64_t)(256 >> uploadNotch) * 1048576ull;
    }
    uint64_t b = (uint64_t)mb * 1048576ull;
    return b >> (uploadNotch > 3 ? 3 : uploadNotch);
}

// ============================================================ churn tracking
static void noteImageCreate(VkImage img, uint32_t w, uint32_t h, uint32_t fmt,
                            uint32_t mips, uint64_t bytes, bool *hotOut)
{
    if (hotOut) *hotOut = false;
    if (!cfg.enable || !img) return;
    std::lock_guard<std::mutex> g(m);
    ChurnKey k; k.w = w; k.h = h; k.fmt = fmt; k.mips = mips;
    ChurnRec &r = g_churn[k];
    ++r.creates;
    r.bytes = bytes;
    // A re-create within ten seconds of a destroy of the same shape is the
    // cycle the plan's SS50 describes.
    if (r.destroys && frameIndex - r.lastDestroyFrame < 600) {
        ++r.cycles;
        ++churnCycles;
    }
    g_imgKey[img] = k;
    if (r.cycles >= 3) {
        g_hotImgs.insert(img);
        if (hotOut) *hotOut = true;
    }
}

static void noteImageDestroy(VkImage img)
{
    if (!img) return;
    std::lock_guard<std::mutex> g(m);
    std::map<VkImage, ChurnKey>::iterator it = g_imgKey.find(img);
    if (it != g_imgKey.end()) {
        ChurnRec &r = g_churn[it->second];
        ++r.destroys;
        r.lastDestroyFrame = frameIndex;
        // Content identity outlives the handle (SS24's L2): the destroyed
        // image's mip hashes move to the dead-shape memory so a disk reload
        // of the same bytes can be recognised - and costed - when it arrives
        // in a fresh VkImage.
        uint64_t base = (uint64_t)(uintptr_t)img;
        MipKey from; from.img = base; from.mip = 0;
        for (std::map<MipKey, uint64_t>::iterator ci =
                 g_liveContent.lower_bound(from);
             ci != g_liveContent.end() && ci->first.img == base;) {
            if (g_deadContent.size() < 65536) {
                ShapeMip s; s.w = it->second.w; s.h = it->second.h;
                s.fmt = it->second.fmt; s.mip = ci->first.mip;
                g_deadContent[s] = ci->second;
            }
            g_liveContent.erase(ci++);
        }
        // A burst of large preload-class destructions is an aircraft or
        // livery change (SS76). Score decays at present; the threshold fires
        // the same protective response as a teleport.
        if (it->second.w >= 2048 && frameIndex > 900)
            bigDestroyScore += 1.0;
        g_imgKey.erase(it);
    }
    g_hotImgs.erase(img);
    g_imgLastUse.erase(img);
    g_imgMem.erase(img);
}

static bool churnHot(VkImage img)
{
    std::lock_guard<std::mutex> g(m);
    return g_hotImgs.count(img) != 0;
}

// ============================================================ misc telemetry
static void notePipelines(uint32_t count, uint64_t us)
{
    pipesTotal.fetch_add(count);
    pipeUsTotal.fetch_add(us);
    if (frameIndex > 0) {                 // after the first present = in flight
        pipesLive.fetch_add(count);
        pipesFrame.fetch_add(count);
    }
}

static void noteDescriptorUpdates(uint32_t n) { descUpdFrame.fetch_add(n); }
static void noteDescriptorAllocs(uint32_t n)  { descAllocFrame.fetch_add(n); }
static void notePipelineBind()                { pipeBindsFrame.fetch_add(1); }

static void noteImageUse(VkImage img)
{
    std::lock_guard<std::mutex> g(m);
    UseRec &r = g_imgLastUse[img];
    r.last = frameIndex;
    if (r.count < 0xFFFFFFFFu) ++r.count;
}

static void noteImageMem(VkImage img, VkDeviceMemory mem)
{
    std::lock_guard<std::mutex> g(m);
    g_imgMem[img] = mem;
}

// ============================================================ map + bind maps
static void noteMap(VkDeviceMemory mem, VkDeviceSize offset, VkDeviceSize size,
                    void *ptr)
{
    if (!ptr) return;
    std::lock_guard<std::mutex> g(m);
    MapRec r; r.ptr = (uint8_t*)ptr; r.offset = offset;
    if (size == VK_WHOLE_SIZE) {
        std::map<VkDeviceMemory, AllocRec>::iterator it = g_allocs.find(mem);
        size = (it != g_allocs.end() && it->second.size > offset)
             ? it->second.size - offset : 0;
    }
    r.size = size;
    g_mapped[mem] = r;
}

static void noteUnmap(VkDeviceMemory mem)
{
    std::lock_guard<std::mutex> g(m);
    g_mapped.erase(mem);
}

static void noteBufBind(VkBuffer buf, VkDeviceMemory mem, VkDeviceSize offset)
{
    std::lock_guard<std::mutex> g(m);
    BufBind b; b.mem = mem; b.offset = offset;
    g_bufBind[buf] = b;
}

static void noteBufferGone(VkBuffer buf)
{
    std::lock_guard<std::mutex> g(m);
    g_bufBind.erase(buf);
}

// The host pointer a buffer's byte range lives at, if its memory is mapped.
// nullptr when unmapped, unknown, or out of the mapped range.
static const uint8_t *bufferBytes(VkBuffer buf, VkDeviceSize bufOffset,
                                  VkDeviceSize len)
{
    std::lock_guard<std::mutex> g(m);
    std::map<VkBuffer, BufBind>::iterator b = g_bufBind.find(buf);
    if (b == g_bufBind.end()) return nullptr;
    std::map<VkDeviceMemory, MapRec>::iterator mr = g_mapped.find(b->second.mem);
    if (mr == g_mapped.end()) return nullptr;
    VkDeviceSize memOff = b->second.offset + bufOffset;
    if (memOff < mr->second.offset) return nullptr;
    VkDeviceSize rel = memOff - mr->second.offset;
    if (rel + len > mr->second.size) return nullptr;
    return mr->second.ptr + rel;
}

// ============================================================ content cache
// FNV-1a over strided samples plus the length: 4 KB of reads bounds the cost
// on a 30 MB upload while still touching every region of it.
static uint64_t contentHash(const uint8_t *p, uint64_t len)
{
    uint64_t h = 1469598103934665603ull ^ len;
    uint64_t stride = len > 4096 ? (len / 4096) & ~7ull : 1;
    if (!stride) stride = 1;
    for (uint64_t i = 0; i < len; i += stride)
        h = (h ^ p[i]) * 1099511628211ull;
    // The tail, always - truncation artefacts live there.
    for (uint64_t i = len > 64 ? len - 64 : 0; i < len; ++i)
        h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

// Verdict on one full-subresource upload. Returns true when the copy can be
// ELIDED - the identical bytes are already in that image mip.
static bool cacheUpload(VkImage img, uint32_t mip, uint64_t hash,
                        uint64_t bytes)
{
    std::lock_guard<std::mutex> g(m);
    MipKey k; k.img = (uint64_t)(uintptr_t)img; k.mip = mip;
    std::map<MipKey, uint64_t>::iterator it = g_liveContent.find(k);
    if (it != g_liveContent.end() && it->second == hash) {
        ++elidedUploads;
        elidedBytes += bytes;
        return true;
    }
    g_liveContent[k] = hash;
    // Did the engine just reload from disk what a destroyed image held?
    std::map<VkImage, ChurnKey>::iterator ck = g_imgKey.find(img);
    if (ck != g_imgKey.end()) {
        ShapeMip s; s.w = ck->second.w; s.h = ck->second.h;
        s.fmt = ck->second.fmt; s.mip = mip;
        std::map<ShapeMip, uint64_t>::iterator dc = g_deadContent.find(s);
        if (dc != g_deadContent.end() && dc->second == hash) {
            ++reloadIdenticalCount;
            reloadIdenticalBytes += bytes;
        }
    }
    return false;
}

// (Image teardown's content transfer lives inside noteImageDestroy - one
// mutex acquisition, one entry point.)

// A transition FROM UNDEFINED legally discards an image's contents. Any
// cached content identity for that image is dead the moment such a barrier
// is recorded - eliding an upload after it would leave garbage texels. The
// barrier hooks call this for every oldLayout==UNDEFINED image barrier.
static void contentInvalidate(VkImage img)
{
    if (!img) return;
    std::lock_guard<std::mutex> g(m);
    if (g_liveContent.empty()) return;
    uint64_t base = (uint64_t)(uintptr_t)img;
    MipKey from; from.img = base; from.mip = 0;
    for (std::map<MipKey, uint64_t>::iterator it = g_liveContent.lower_bound(from);
         it != g_liveContent.end() && it->first.img == base;)
        g_liveContent.erase(it++);
}

static void noteUploadRegion(uint64_t imgHandle, uint32_t mip)
{
    std::lock_guard<std::mutex> g(m);
    if (!g_upSeen.insert(std::make_pair(imgHandle, mip)).second) ++dupUploads;
}

static void ledgerRt(uint64_t bytes) { rtBytesNow = bytes; }

// ============================================================ persistence
static const char *statePath()
{
    static char p[MAX_PATH] = {0};
    if (!p[0]) {
        const char *t = getenv("TEMP");
        _snprintf(p, sizeof(p) - 1, "%s\\taa_vram_state.txt", t ? t : ".");
    }
    return p;
}

static void stateLoad()
{
    FILE *f = fopen(statePath(), "r");
    if (!f) return;
    char line[128];
    while (fgets(line, sizeof(line), f)) {
        unsigned long long v = 0;
        if (sscanf(line, "fails_ever=%llu", &v) == 1) failsEver = v;
        else if (sscanf(line, "reserve_bias_mb=%llu", &v) == 1)
            reserveBiasMB = (int)(v > 256 ? 256 : v);
        else if (sscanf(line, "upload_peak_mb=%llu", &v) == 1)
            upBytesPeak = v * 1048576ull;
    }
    fclose(f);
    if (reserveBiasMB || failsEver)
        trace("VRAMSYS: state loaded - %llu historical allocation failures, "
              "standing reserve bias +%d MB",
              (unsigned long long)failsEver, reserveBiasMB);
}

static void stateSave()
{
    FILE *f = fopen(statePath(), "w");
    if (!f) return;
    fprintf(f, "# VRAM system state - carried across sessions (plan SS77).\n"
               "fails_ever=%llu\n"
               "reserve_bias_mb=%d\n"
               "upload_peak_mb=%llu\n",
            (unsigned long long)failsEver, reserveBiasMB,
            (unsigned long long)(upBytesPeak / 1048576ull));
    fclose(f);
}

// A memory type's property flags, for the enriched failure log (SS69).
static const char *typeFlagsText(uint32_t typeIndex)
{
    static char buf[96];
    buf[0] = 0;
    if (typeIndex >= dev.memProps.memoryTypeCount) return "unknown-type";
    VkMemoryPropertyFlags f = dev.memProps.memoryTypes[typeIndex].propertyFlags;
    if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)  strcat(buf, "DEVICE ");
    if (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)  strcat(buf, "HOSTVIS ");
    if (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) strcat(buf, "COHERENT ");
    if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)   strcat(buf, "CACHED ");
    if (!buf[0]) strcat(buf, "none");
    return buf;
}

// ============================================================ device binding
// Called from TAA_CreateDevice once the down-chain pointers exist.
static void bindDevice(VkDevice d, VkPhysicalDevice p,
                       PFN_vkGetPhysicalDeviceMemoryProperties2 mp2,
                       PFN_vkFreeMemory freeMem, PFN_vkUnmapMemory unmapMem,
                       PFN_vkQueueSubmit qsub,
                       PFN_vkSetDeviceMemoryPriorityEXT setPrio,
                       bool prioExt, bool pageExt,
                       const VkPhysicalDeviceMemoryProperties *props,
                       uint32_t transferOnlyFamily)
{
    std::lock_guard<std::mutex> g(m);
    dev.dev = d; dev.phys = p; dev.memProps2 = mp2;
    dev.freeMemory = freeMem; dev.unmapMemory = unmapMem;
    dev.queueSubmit = qsub;
    dev.setPriority = setPrio;
    warmupStartFrame = frameIndex;
    dev.priorityExt = prioExt; dev.pageableExt = pageExt && setPrio != nullptr;
    if (props) dev.memProps = *props;
    dev.transferFamily = transferOnlyFamily;
    trace("VRAMSYS: bound - priority ext %s, pageable ext %s, transfer-only "
          "family %s (%u)",
          prioExt ? "ON" : "off",
          dev.pageableExt ? "ON" : "off",
          transferOnlyFamily == ~0u ? "NONE" : "found", transferOnlyFamily);
    stateLoad();
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
// Host-visible blocks are recyclable too (SS15's staging pool, at our level)
// with one legality step: if the app still holds a mapping, the block is
// explicitly unmapped BEFORE it enters the pool - vkFreeMemory would have
// unmapped implicitly, and a recycled block carrying a live mapping would
// make the next owner's vkMapMemory a spec violation. The unmap happens
// under our mutex, before the block becomes takeable, so no thread can
// receive a still-mapped block. (Calling the down-chain unmap under m is
// safe: m is ours alone; the driver cannot re-enter the layer.)
static bool poolHold(VkDeviceMemory mem)
{
    if (!cfg.enable || !cfg.recycle) return false;
    std::lock_guard<std::mutex> g(m);
    std::map<VkDeviceMemory, AllocRec>::iterator it = g_allocs.find(mem);
    if (it == g_allocs.end() || !it->second.plain) return false;
    bool devLocal = typeRecyclable(it->second.type);
    bool hostOk   = false;
    if (!devLocal && it->second.type < dev.memProps.memoryTypeCount) {
        VkMemoryPropertyFlags f =
            dev.memProps.memoryTypes[it->second.type].propertyFlags;
        // Host-visible, host-coherent staging - the class VMA churns.
        hostOk = (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                 (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    if (!devLocal && !hostOk) return false;
    if (zone >= RED && devLocal) return false;         // pressure: give it back
    uint64_t cap = (uint64_t)cfg.recycleMaxMB * 1048576ull;
    if (g_poolBytes + it->second.size > cap) return false;
    if (hostOk) {
        std::map<VkDeviceMemory, MapRec>::iterator mr = g_mapped.find(mem);
        if (mr != g_mapped.end()) {
            if (!dev.unmapMemory || !dev.dev) return false;
            dev.unmapMemory(dev.dev, mem);
            ++hostRecycleUnmaps;
            g_mapped.erase(mr);
        }
    }
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
// A churn-hot resource (SS50: cycling resident->evicted->resident) takes at
// least 0.65 whatever its class - "increase its retention priority", made
// literal at the driver level.
static void onBind(VkDeviceMemory mem, int cat, bool streamed, bool hot = false)
{
    if (!cfg.enable || !cfg.priority || !dev.pageableExt || !dev.dev) return;
    float p = prioOfCat(cat, streamed);
    if (hot && p < 0.65f) p = 0.65f;
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
// down so the driver demotes THOSE first - LARGEST blocks first (SS21's size
// weight, SS22: the most memory back per priority move); lift them again in
// GREEN. Bounded per call so a zone flip never stalls a present.
static void prioZoneWalk()
{
    if (!cfg.enable || !cfg.priority || !dev.pageableExt || !dev.dev) return;
    std::vector<std::pair<VkDeviceMemory, float> > moves;
    {
        std::lock_guard<std::mutex> g(m);
        // Candidates first, size-ordered, THEN the per-frame budget - so the
        // budget is spent on the blocks whose demotion frees the most.
        std::vector<std::pair<uint64_t, VkDeviceMemory> > cand;
        for (std::map<VkDeviceMemory, BlockPrio>::iterator it = g_blockPrio.begin();
             it != g_blockPrio.end(); ++it) {
            if (!it->second.streamedTexOnly) continue;
            bool wantDemote  = zone >= RED  && !it->second.demoted;
            bool wantRestore = zone == GREEN && it->second.demoted;
            if (!wantDemote && !wantRestore) continue;
            uint64_t sz = 0;
            std::map<VkDeviceMemory, AllocRec>::iterator ar = g_allocs.find(it->first);
            if (ar != g_allocs.end()) sz = ar->second.size;
            cand.push_back(std::make_pair(sz, it->first));
        }
        std::sort(cand.begin(), cand.end());
        int budget = 64;
        for (size_t i = cand.size(); i > 0 && budget > 0; --i, --budget) {
            std::map<VkDeviceMemory, BlockPrio>::iterator it =
                g_blockPrio.find(cand[i - 1].second);
            if (it == g_blockPrio.end()) continue;
            if (zone >= RED && !it->second.demoted) {
                it->second.demoted = true;
                moves.push_back(std::make_pair(it->first, 0.2f));
            } else if (zone == GREEN && it->second.demoted) {
                it->second.demoted = false;
                moves.push_back(std::make_pair(it->first,
                    it->second.aged ? 0.25f : it->second.best));
            }
        }
    }
    for (size_t i = 0; i < moves.size(); ++i)
        dev.setPriority(dev.dev, moves[i].first, moves[i].second);
    prioZoneMoves += moves.size();
}

// The aging walk (SS86-88): a rotating cursor visits sampled-use records, 64
// a frame. A streamed-texture block whose images have gone unused for the age
// window decays to 0.25; the window DOUBLES for frequently-used resources
// (SS87 - recency alone is the wrong statistic). First use after decay
// restores the class priority. Hysteresis is inherent: the window is minutes,
// the restore is immediate on use.
static void agingWalk()
{
    if (!cfg.enable || !cfg.priority || !dev.pageableExt || !dev.dev) return;
    static VkImage cursor = VK_NULL_HANDLE;
    std::vector<std::pair<VkDeviceMemory, float> > moves;
    {
        std::lock_guard<std::mutex> g(m);
        std::map<VkImage, UseRec>::iterator it = g_imgLastUse.upper_bound(cursor);
        int budget = 64;
        for (; it != g_imgLastUse.end() && budget > 0; ++it, --budget) {
            cursor = it->first;
            std::map<VkImage, VkDeviceMemory>::iterator im = g_imgMem.find(it->first);
            if (im == g_imgMem.end()) continue;
            std::map<VkDeviceMemory, BlockPrio>::iterator bp =
                g_blockPrio.find(im->second);
            if (bp == g_blockPrio.end() || !bp->second.streamedTexOnly) continue;
            uint64_t window = (uint64_t)cfg.ageFrames;
            if (it->second.count > 100) window *= 2;       // frequency (SS87)
            bool stale = frameIndex > it->second.last &&
                         frameIndex - it->second.last > window;
            if (stale && !bp->second.aged && !bp->second.demoted) {
                bp->second.aged = true;
                moves.push_back(std::make_pair(im->second, 0.25f));
                ++agedDemotions;
            } else if (!stale && bp->second.aged && !bp->second.demoted) {
                bp->second.aged = false;
                moves.push_back(std::make_pair(im->second, bp->second.best));
                ++agedRestores;
            }
        }
        if (it == g_imgLastUse.end()) cursor = VK_NULL_HANDLE;
    }
    for (size_t i = 0; i < moves.size(); ++i)
        dev.setPriority(dev.dev, moves[i].first, moves[i].second);
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
        // Backpressure (SS96): every queue needs a maximum size. Past the
        // cap the governor stops holding and passes work through - a bounded
        // delay line, never an unbounded backlog that bursts.
        bool overCap = heldBytesNow + bytes >
                       (uint64_t)cfg.holdMaxMB * 1048576ull;
        bool over = (g_frameUploadSpent + bytes) > budget;
        if ((!over && !mustQueueBehind) || !holdable || overCap ||
            zone < ORANGE) {
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
        // the engine starts adapting BEFORE the wall, not at it. Two
        // modulations on top: the persistent bias a session that ever hit an
        // allocation failure carries forward (SS77), and camera speed - a fast
        // camera is about to demand scenery, so the reserve grows with it
        // (SS11/49, prediction without a single dataref).
        double speedFactor = camSpeedEma * cfg.speedReserve;
        if (speedFactor > 1.0) speedFactor = 1.0;
        uint64_t reserve = (uint64_t)(((double)cfg.reserveMB[zone] +
                                       (double)reserveBiasMB) *
                                      (1.0 + speedFactor)) * 1048576ull;

        // Warmup (SS73): an extra reserve, largest at device creation and
        // decaying linearly to nothing, so startup fills VRAM progressively -
        // "do not attempt to fully populate VRAM immediately."
        uint64_t warmFrames = (uint64_t)(cfg.warmupFrames > 0
                                         ? cfg.warmupFrames : 0);
        if (warmFrames && frameIndex - warmupStartFrame < warmFrames) {
            double t = 1.0 - (double)(frameIndex - warmupStartFrame) /
                             (double)warmFrames;
            reserve += (uint64_t)((double)cfg.warmupMB * t) * 1048576ull;
        }
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
        // Learn from it: every failure raises the standing reserve carried
        // into future sessions, capped so one bad day cannot eat the card.
        ++failsEver;
        if (reserveBiasMB < 256) reserveBiasMB += 64;
    }
    stateSave();
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
    cfg.speedReserve   = live::f("vram.speed_reserve",   nullptr, 0.01f);
    cfg.adaptive       = live::onoff("vram.adaptive",    nullptr, true);
    cfg.uploadCache    = live::onoff("vram.upload_cache", "TAA_VRAM_UPLOAD_CACHE", true);
    cfg.warmupFrames   = live::i("vram.warmup_frames",   nullptr, 900);
    cfg.warmupMB       = live::i("vram.warmup_mb",       nullptr, 512);
    cfg.holdMaxMB      = live::i("vram.hold_max_mb",     nullptr, 512);
    cfg.lookaheadFrames = live::i("vram.lookahead",      nullptr, 300);
    cfg.ageFrames      = live::i("vram.age_frames",      nullptr, 1800);
}

// Frame-time sample and the adaptive upload notch (SS29/58). If the rolling
// average degrades more than 30% past the best recent average while uploads
// are flowing, the upload budget halves; after 600 clean frames it steps back.
// Smoothing both ways so it cannot oscillate.
static void frameTimeTick()
{
    LARGE_INTEGER now, fq;
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&fq);
    if (g_havePresentQpc && fq.QuadPart > 0) {
        double ms = (double)(now.QuadPart - g_lastPresentQpc.QuadPart) *
                    1000.0 / (double)fq.QuadPart;
        if (ms > 0.01 && ms < 10000.0) {
            g_ftRing[g_ftAt] = ms;
            g_ftAt = (g_ftAt + 1) & 511;
            if (g_ftN < 512) ++g_ftN;
            if (frameAvgMs <= 0.0) frameAvgMs = ms;
            frameAvgMs += (ms - frameAvgMs) * 0.05;
            if (frameAvgMs < frameBestMs) frameBestMs = frameAvgMs;
            frameBestMs *= 1.0001;         // decays upward: "recent" best
            if (cfg.adaptive) {
                bool uploadsFlowing = upBytesLast > (4ull << 20);
                if (frameAvgMs > frameBestMs * 1.30 && uploadsFlowing) {
                    if (uploadNotch < 3) {
                        ++uploadNotch;
                        notchCalm = 0;
                        trace("VRAMSYS: frame time %.1f ms against a best of "
                              "%.1f ms with uploads flowing - upload budget "
                              "notched down to 1/%d",
                              frameAvgMs, frameBestMs, 1 << uploadNotch);
                    }
                } else if (uploadNotch > 0) {
                    if (frameAvgMs < frameBestMs * 1.10) {
                        if (++notchCalm >= 600) {
                            --uploadNotch;
                            notchCalm = 0;
                            trace("VRAMSYS: frame time recovered - upload "
                                  "budget notch released to 1/%d",
                                  1 << uploadNotch);
                        }
                    } else notchCalm = 0;
                }
            }
        }
    }
    g_lastPresentQpc = now;
    g_havePresentQpc = true;
}

// Percentiles over the frame-time ring, for the report (SS30: average FPS,
// 1% low, 0.1% low, variance, maximum).
static void frameTimeStats(double *avg, double *p99, double *p999,
                           double *worst, double *var)
{
    *avg = *p99 = *p999 = *worst = *var = 0.0;
    if (!g_ftN) return;
    std::vector<double> v(g_ftRing, g_ftRing + g_ftN);
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (int i = 0; i < g_ftN; ++i) sum += v[i];
    *avg = sum / g_ftN;
    *p99   = v[(int)((g_ftN - 1) * 0.99)];
    *p999  = v[(int)((g_ftN - 1) * 0.999)];
    *worst = v[g_ftN - 1];
    double s2 = 0;
    for (int i = 0; i < g_ftN; ++i) s2 += (v[i] - *avg) * (v[i] - *avg);
    *var = s2 / g_ftN;
}

static void zoneUpdate()
{
    // Projected usage (SS27): what is resident now, plus what the last frame's
    // uploads suggest is arriving, plus what the governor is holding and will
    // release - minus the recycle pool, which is reclaimable the instant it is
    // needed. Cheap, monotone with real pressure, no readbacks.
    double basis = filteredBudget > 0 ? filteredBudget : (double)heapSize;
    if (basis <= 0) return;
    // SS85: usage HISTORY predicts the peak. The per-frame delta's EMA,
    // projected over the lookahead horizon, joins the projection - so a
    // climb toward a dense region begins degradation BEFORE the peak.
    double delta = (double)rawUsage - (double)prevRawUsage;
    prevRawUsage = rawUsage;
    if (delta > -1e9 && delta < 1e9)
        usageTrendBytes += (delta - usageTrendBytes) * 0.05;
    double trendTerm = usageTrendBytes > 0
                     ? usageTrendBytes * (double)cfg.lookaheadFrames : 0.0;
    double projected = (double)rawUsage + (double)upBytesLast * 2.0 +
                       (double)heldBytesNow + trendTerm - (double)g_poolBytes;
    if (projected < 0) projected = 0;
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
        g_upSeen.clear();                       // per-frame dup detection
    }

    frameTimeTick();

    // Per-frame peaks for the stutter telemetry.
    {
        uint64_t pf = pipesFrame.exchange(0);
        if (pf > pipesFramePeak) { pipesFramePeak = pf; pipesFramePeakAt = fi; }
        uint64_t du = descUpdFrame.exchange(0);
        if (du > descUpdPeak) descUpdPeak = du;
        uint64_t da = descAllocFrame.exchange(0);
        if (da > descAllocPeak) descAllocPeak = da;
        uint64_t pb = pipeBindsFrame.exchange(0);
        if (pb > pipeBindsPeak) pipeBindsPeak = pb;
    }

    // Aircraft / livery change (SS76): the burst score decays; over the
    // threshold, the teleport response fires - pool flushed, zone floored -
    // because the working set is about to be replaced wholesale.
    {
        std::lock_guard<std::mutex> g(m);
        bigDestroyScore *= 0.98;
        if (aircraftChangeCooldown > 0) --aircraftChangeCooldown;
        if (bigDestroyScore > 40.0 && !aircraftChangeCooldown) {
            bigDestroyScore = 0.0;
            aircraftChangeCooldown = 1800;
            teleportLeft = cfg.teleportFrames;
            ++aircraftChanges;
            trace("VRAMSYS: aircraft/livery change detected (large-texture "
                  "destruction burst) - zone floor YELLOW, recycle pool "
                  "flushed, old working set will demote naturally");
        }
    }
    if (teleportLeft == cfg.teleportFrames && aircraftChangeCooldown == 1800)
        poolTrim(true);

    // The benchmark window (SS99): vram.bench=1 opens it, 0 closes and dumps.
    {
        bool want = live::i("vram.bench", nullptr, 0) != 0;
        if (want && !benchOn) {
            benchOn = true;
            benchStartFrame = fi;
            benchFt.clear(); benchFtSum = 0; benchFtWorst = 0;
            benchVramPeak = 0; benchUpPeak = 0; benchUpTotal = 0;
            benchAllocs0 = allocsTotal.load(); benchFails0 = allocFails.load();
            benchPipes0 = pipesLive.load();
            benchElide0 = elidedUploads; benchReload0 = reloadIdenticalCount;
            for (int z = 0; z < 5; ++z) benchZoneFrames[z] = 0;
            trace("VRAMSYS BENCH: window open at frame %llu",
                  (unsigned long long)fi);
        } else if (!want && benchOn) {
            benchOn = false;
            uint64_t n = fi - benchStartFrame;
            double avg = benchFt.empty() ? 0 : benchFtSum / benchFt.size();
            double p99 = 0, p999 = 0;
            if (!benchFt.empty()) {
                std::vector<double> v(benchFt);
                std::sort(v.begin(), v.end());
                p99  = v[(size_t)((v.size() - 1) * 0.99)];
                p999 = v[(size_t)((v.size() - 1) * 0.999)];
            }
            trace("VRAMSYS BENCH: window closed - %llu frames", (unsigned long long)n);
            trace("  frame ms: avg %.2f  1%%low %.2f  0.1%%low %.2f  worst %.2f",
                  avg, p99, p999, benchFtWorst);
            trace("  vram peak %.2f GB  upload peak %.1f MB/f  upload total %.1f MB",
                  benchVramPeak / 1073741824.0, benchUpPeak / 1048576.0,
                  benchUpTotal / 1048576.0);
            trace("  allocs %llu  failures %llu  JIT pipelines %llu  "
                  "elided uploads %llu  identical reloads %llu",
                  (unsigned long long)(allocsTotal.load() - benchAllocs0),
                  (unsigned long long)(allocFails.load() - benchFails0),
                  (unsigned long long)(pipesLive.load() - benchPipes0),
                  (unsigned long long)(elidedUploads - benchElide0),
                  (unsigned long long)(reloadIdenticalCount - benchReload0));
            trace("  zone frames: G %llu  Y %llu  O %llu  R %llu  C %llu",
                  (unsigned long long)benchZoneFrames[0],
                  (unsigned long long)benchZoneFrames[1],
                  (unsigned long long)benchZoneFrames[2],
                  (unsigned long long)benchZoneFrames[3],
                  (unsigned long long)benchZoneFrames[4]);
        }
        if (benchOn) {
            if (g_ftN > 0) {
                double lastMs = g_ftRing[(g_ftAt + 511) & 511];
                benchFt.push_back(lastMs);
                benchFtSum += lastMs;
                if (lastMs > benchFtWorst) benchFtWorst = lastMs;
                if (benchFt.size() > 200000) { benchFt.clear(); benchFtSum = 0; }
            }
            if (rawUsage > benchVramPeak) benchVramPeak = rawUsage;
            if (upBytesLast > benchUpPeak) benchUpPeak = upBytesLast;
            benchUpTotal += upBytesLast;
            benchZoneFrames[zone < 0 ? 0 : (zone > 4 ? 4 : zone)]++;
        }
    }

    if (fi % 3600 == 0) stateSave();            // SS77: survive the session

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
    agingWalk();

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
        {
            double avg, p99, p999, worst, var;
            frameTimeStats(&avg, &p99, &p999, &worst, &var);
            trace("  frame time: avg %.2f ms  1%%low %.2f ms  0.1%%low %.2f ms "
                  " worst %.2f ms  var %.2f  (upload notch 1/%d)",
                  avg, p99, p999, worst, var, 1 << uploadNotch);
        }
        trace("  pipelines: %llu total (%.0f ms), %llu created IN FLIGHT, "
              "worst frame %llu at frame %llu",
              (unsigned long long)pipesTotal.load(),
              pipeUsTotal.load() / 1000.0,
              (unsigned long long)pipesLive.load(),
              (unsigned long long)pipesFramePeak,
              (unsigned long long)pipesFramePeakAt);
        trace("  descriptors: peak %llu updates / %llu allocs in one frame;  "
              "duplicate uploads %llu;  churn cycles %llu (%llu hot shapes)",
              (unsigned long long)descUpdPeak,
              (unsigned long long)descAllocPeak,
              (unsigned long long)dupUploads,
              (unsigned long long)churnCycles,
              (unsigned long long)g_hotImgs.size());
        trace("  render targets: %.1f MB RT+depth+storage - the upper bound "
              "transient aliasing could ever reclaim (SS34 investigation)",
              rtBytesNow / 1048576.0);
        trace("  upload cache: %llu elided (%.1f MB saved), %llu identical "
              "disk reloads (%.1f MB) - the measured cost of the engine's "
              "missing CPU cache;  host staging recycles unmapped %llu",
              (unsigned long long)elidedUploads, elidedBytes / 1048576.0,
              (unsigned long long)reloadIdenticalCount,
              reloadIdenticalBytes / 1048576.0,
              (unsigned long long)hostRecycleUnmaps);
        trace("  binds: pipeline bind peak %llu/frame;  usage trend %+.1f "
              "MB/s;  aircraft changes %llu",
              (unsigned long long)pipeBindsPeak,
              usageTrendBytes * 60.0 / 1048576.0,
              (unsigned long long)aircraftChanges);
        trace("  aging: %llu tracked images, %llu decayed, %llu restored "
              "(window %d frames, doubled for frequent users)",
              (unsigned long long)g_imgLastUse.size(),
              (unsigned long long)agedDemotions,
              (unsigned long long)agedRestores, cfg.ageFrames);
        {
            // Top churners - the shapes cycling through residency (SS51).
            int printed = 0;
            for (std::map<ChurnKey, ChurnRec>::iterator it = g_churn.begin();
                 it != g_churn.end() && printed < 5; ++it) {
                if (it->second.cycles < 2) continue;
                trace("  churner: %ux%u fmt=%u mips=%u - %u cycles, %u creates, "
                      "%.1f MB each",
                      it->first.w, it->first.h, it->first.fmt, it->first.mips,
                      it->second.cycles, it->second.creates,
                      it->second.bytes / 1048576.0);
                ++printed;
            }
        }
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

// Device teardown: everything held goes back before the device dies, the
// session's learning is saved, and what remains allocated is reported - the
// plan's SS67 leak check, stated rather than assumed.
static void shutdown()
{
    flushAll(&flushOnDep);
    poolTrim(true);
    stateSave();
    std::lock_guard<std::mutex> g(m);
    uint64_t remBytes = 0;
    for (std::map<VkDeviceMemory, AllocRec>::iterator it = g_allocs.begin();
         it != g_allocs.end(); ++it)
        remBytes += it->second.size;
    trace("VRAMSYS: shutdown - %llu allocations still live (%.1f MB). "
          "Driver-owned and swapchain blocks are expected here; a large or "
          "growing figure across device recreations is a leak.",
          (unsigned long long)g_allocs.size(), remBytes / 1048576.0);
}

} // namespace vram
