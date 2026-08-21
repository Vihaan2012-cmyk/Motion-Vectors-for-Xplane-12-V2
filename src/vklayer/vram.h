// MotionVectors for X-Plane 12 - temporal anti-aliasing from injected motion
// vectors, plus a VRAM manager.
//
// Copyright (C) 2026 Vihaan2012
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
// more details.
//
// You should have received a copy of the GNU General Public License along
// with this program. If not, see <https://www.gnu.org/licenses/>.

#pragma once

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

static Config cfg = { true, false, false, false, false,
                      0.02f, {128,256,384,512,768}, 512, 600,
                      256, 180, {0,0,64,24,8}, 2,
                      2000.0f, 900, 600, 0.01f, true,
                      false, 900, 512, 512, 300, 1800 };

static bool alive()
{
    static int a = -1;
    if (a < 0) {
        const char *vh = getenv("TAA_VRAM_HOOKS");
        a = (!vh || atoi(vh) != 0) ? 1 : 0;
    }
    return a != 0;
}

enum Zone { GREEN = 0, YELLOW, ORANGE, RED, CRITICAL };
static const char *zoneName(int z)
{
    switch (z) {
        case GREEN:  return "GREEN";  case YELLOW:   return "YELLOW";
        case ORANGE: return "ORANGE"; case RED:      return "RED";
        default:     return "CRITICAL";
    }
}

static std::mutex m;

struct Dev {
    VkDevice         dev  = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    PFN_vkGetPhysicalDeviceMemoryProperties2 memProps2 = nullptr;
    PFN_vkFreeMemory                 freeMemory  = nullptr;
    PFN_vkUnmapMemory                unmapMemory = nullptr;
    PFN_vkQueueSubmit                queueSubmit = nullptr;
    PFN_vkSetDeviceMemoryPriorityEXT setPriority = nullptr;
    bool priorityExt = false;
    bool pageableExt = false;
    VkPhysicalDeviceMemoryProperties memProps;
    uint32_t transferFamily = ~0u;
} dev;

static std::set<VkQueue> g_transferQueues;

static std::map<VkQueue, std::mutex*> g_queueLock;

static uint64_t heapSize = 0;
static uint64_t rawBudget = 0, rawUsage = 0;
static double   filteredBudget = 0.0;
static uint64_t lastReported = 0;
static uint64_t lastLedgerBytes = 0;
static std::atomic<uint64_t> ledgerBytesNow(0);
static int      zone = GREEN;
static int      zoneDwell = 0;
static int      deflateLeft = 0;
static int      teleportLeft = 0;
static uint64_t frameIndex = 0;

static std::atomic<uint64_t> upBytesFrame(0);
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

static double        g_ftRing[512];
static int           g_ftAt = 0, g_ftN = 0;
static LARGE_INTEGER g_lastPresentQpc;
static bool          g_havePresentQpc = false;
static double        frameAvgMs = 0.0;
static double        frameBestMs = 1e9;
static int           uploadNotch = 0;
static uint64_t      notchCalm = 0;

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
    uint8_t  lastDrop = 0;
};
static uint64_t mipRestorations = 0;
static uint64_t perResProtected = 0, perResExtraCut = 0;
static uint64_t zoneFlips = 0;
static std::map<ChurnKey, ChurnRec> g_churn;
static uint64_t churnCycles = 0;

enum ResClass {
    RC_UNKNOWN = 0, RC_RENDER_TARGET, RC_DEPTH, RC_SHADOW, RC_STORAGE,
    RC_CLOUD_VOLUME, RC_AIRCRAFT_COCKPIT, RC_SCENERY_STREAMED,
    RC_TERRAIN_ARRAY, RC_UI_SMALL
};
static const char *resClassName(int c)
{
    switch (c) {
        case RC_RENDER_TARGET:    return "render-target";
        case RC_DEPTH:            return "depth";
        case RC_SHADOW:           return "shadow";
        case RC_STORAGE:          return "storage";
        case RC_CLOUD_VOLUME:     return "cloud-volume";
        case RC_AIRCRAFT_COCKPIT: return "aircraft/cockpit";
        case RC_SCENERY_STREAMED: return "scenery-streamed";
        case RC_TERRAIN_ARRAY:    return "terrain-array";
        case RC_UI_SMALL:         return "ui-small";
        default:                  return "unknown";
    }
}
enum ResState {
    RS_FULL = 0,
    RS_REDUCED_AT_CREATE,
    RS_DEMOTED,
    RS_AGED,
    RS_DESTROYED
};
static const char *resStateName(int s)
{
    switch (s) {
        case RS_REDUCED_AT_CREATE: return "REDUCED";
        case RS_DEMOTED:           return "DEMOTED";
        case RS_AGED:              return "AGED";
        default:                   return "FULL";
    }
}
struct ResRec {
    uint64_t id;
    ChurnKey shape;
    uint64_t bytes;
    uint64_t createdFrame;
    uint8_t  cls;
    uint8_t  state;
    uint8_t  protection;
    bool     streamed;
    float    score;

    float    birthX, birthY, birthZ;
    uint32_t burst;
};

static float  g_camX = 0, g_camY = 0, g_camZ = 0;
static float  g_camVX = 0, g_camVY = 0, g_camVZ = 0;
static bool   g_camValid = false;
static uint32_t g_burstId = 0;
static uint64_t g_lastCreateFrame = 0;
static std::map<VkImage, ResRec> g_registry;
static uint64_t g_nextResId = 1;
static std::set<VkImage> g_hotImgs;

static std::atomic<uint64_t> pipesTotal(0), pipesLive(0), pipesFrame(0);
static std::atomic<uint64_t> pipeUsTotal(0);
static uint64_t pipesFramePeak = 0, pipesFramePeakAt = 0;

static std::atomic<uint64_t> descUpdFrame(0), descAllocFrame(0);
static uint64_t descUpdPeak = 0, descAllocPeak = 0;

static std::set<std::pair<uint64_t, uint32_t> > g_upSeen;
static uint64_t dupUploads = 0;

static uint64_t rtBytesNow = 0;

static int      reserveBiasMB = 0;
static uint64_t failsEver = 0;
static int      ageBiasFrames = 0;
static double   camRotEma = 0.0;

struct MapRec  { uint8_t *ptr; VkDeviceSize offset, size; };
static std::map<VkDeviceMemory, MapRec> g_mapped;
struct BufBind { VkDeviceMemory mem; VkDeviceSize offset; };
static std::map<VkBuffer, BufBind> g_bufBind;

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
static std::atomic<uint32_t>        g_liveContentCount(0);
static std::map<ShapeMip, uint64_t> g_deadContent;

static std::map<ShapeMip, std::vector<uint8_t> > g_retained;
static uint64_t retainedBytes = 0, retainStores = 0, retainRefused = 0;

static void retainPayload(uint32_t w, uint32_t h, uint32_t fmt, uint32_t mip,
                          const uint8_t *p, uint64_t len)
{
    if (!p || !len || len > (64ull << 20)) return;
    std::lock_guard<std::mutex> g(m);
    uint64_t cap = (uint64_t)live::i("vram.retain_max_mb", nullptr, 256)
                 * 1048576ull;
    ShapeMip k; k.w = w; k.h = h; k.fmt = fmt; k.mip = mip;
    std::map<ShapeMip, std::vector<uint8_t> >::iterator it = g_retained.find(k);
    if (it != g_retained.end()) {
        retainedBytes -= it->second.size();
        it->second.assign(p, p + len);
        retainedBytes += len;
        return;
    }
    if (retainedBytes + len > cap) { ++retainRefused; return; }
    g_retained[k].assign(p, p + len);
    retainedBytes += len;
    ++retainStores;
}

static bool restoreReady(uint32_t w, uint32_t h, uint32_t fmt)
{
    ShapeMip k; k.w = w; k.h = h; k.fmt = fmt; k.mip = 0;
    return g_retained.count(k) != 0;
}

static uint64_t elidedUploads = 0, elidedBytes = 0;
static uint64_t reloadIdenticalCount = 0, reloadIdenticalBytes = 0;
static uint64_t hostRecycleUnmaps = 0;

static double   usageTrendBytes = 0.0;
static uint64_t prevRawUsage = 0;

static uint64_t warmupStartFrame = 0;

static double   bigDestroyScore = 0.0;
static uint64_t aircraftChanges = 0, aircraftChangeCooldown = 0;

static std::atomic<uint64_t> pipeBindsFrame(0);
static uint64_t pipeBindsPeak = 0;

static bool     benchOn = false;
static uint64_t benchStartFrame = 0;
static double   benchFtSum = 0, benchFtWorst = 0;
static std::vector<double> benchFt;
static uint64_t benchVramPeak = 0, benchUpPeak = 0, benchUpTotal = 0;
static uint64_t benchAllocs0 = 0, benchFails0 = 0, benchPipes0 = 0;
static uint64_t benchElide0 = 0, benchReload0 = 0;
static uint64_t benchZoneFrames[5] = {0,0,0,0,0};

struct Held { VkDeviceMemory mem; uint64_t size; uint32_t type; uint64_t frame; };
static std::vector<Held> g_pool;
static uint64_t g_poolBytes = 0;

struct AllocRec { uint64_t size; uint32_t type; bool plain; };
static std::map<VkDeviceMemory, AllocRec> g_allocs;

struct BlockPrio { float best; bool streamedTexOnly; bool demoted; bool aged; };
static std::map<VkDeviceMemory, BlockPrio> g_blockPrio;

struct UseRec { uint64_t last; uint32_t count; uint32_t sceneCount; };
static std::map<VkImage, UseRec>         g_imgLastUse;
static std::map<VkImage, VkDeviceMemory> g_imgMem;
static uint64_t agedDemotions = 0, agedRestores = 0;

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
static std::vector<HeldSubmit> g_heldSubmits;
static std::set<VkFence>     g_heldFences;
static std::map<VkSemaphore, uint64_t> g_heldSignals;
static std::set<VkSemaphore> g_everWaited;

static std::map<VkCommandBuffer, uint64_t> g_cbBytes;
static std::map<VkCommandBuffer, uint8_t>  g_cbProt;
static uint64_t governorBypassed = 0;
static uint64_t g_frameUploadSpent = 0;

static bool typeRecyclable(uint32_t typeIndex)
{

    if (typeIndex >= dev.memProps.memoryTypeCount) return false;
    VkMemoryPropertyFlags f = dev.memProps.memoryTypes[typeIndex].propertyFlags;
    return (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
          !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
}

static float prioOfCat(int cat, bool streamed)
{

    switch (cat) {
        case 1:       case 2:    return 1.0f;
        case 3:                      return 0.9f;
        case 6:                  return 0.8f;
        case 5:                     return 0.7f;
        case 0:      return streamed ? 0.35f : 0.6f;
        case 7:                  return 0.2f;

        case 4:  return streamed ? 0.4f : 0.6f;
        default:                                return 0.5f;
    }
}

static uint64_t zoneUploadBudget()
{
    int mb = cfg.uploadBudgetMB[zone];
    if (mb <= 0) {

        if (uploadNotch <= 0) return ~0ull;
        return (uint64_t)(256 >> uploadNotch) * 1048576ull;
    }
    uint64_t b = (uint64_t)mb * 1048576ull;
    return b >> (uploadNotch > 3 ? 3 : uploadNotch);
}

static bool fmtIsCompressed(uint32_t f)
{
    return f >= 131 && f <= 146;
}

static uint8_t classify(int cat, VkImageUsageFlags usage, uint32_t layers,
                        bool is3D, bool streamed, uint32_t w, uint32_t h,
                        uint32_t fmt)
{
    switch (cat) {
        case 1:       return RC_RENDER_TARGET;
        case 2:

            return (usage & VK_IMAGE_USAGE_SAMPLED_BIT) && w == h && w >= 1024
                 ? RC_SHADOW : RC_DEPTH;
        case 3:  return is3D ? RC_CLOUD_VOLUME : RC_STORAGE;
        case 0:

            if (w <= 256 && h <= 256) return RC_UI_SMALL;
            if (!fmtIsCompressed(fmt) && w <= 512 && h <= 512)
                return RC_UI_SMALL;
            return streamed ? RC_SCENERY_STREAMED : RC_AIRCRAFT_COCKPIT;
        case 4:
            if (is3D) return RC_CLOUD_VOLUME;
            if (layers > 1 && (usage & VK_IMAGE_USAGE_SAMPLED_BIT))
                return RC_TERRAIN_ARRAY;
            return RC_UNKNOWN;
        default:            return RC_UNKNOWN;
    }
}

static uint8_t protectionOfClass(uint8_t cls)
{
    switch (cls) {
        case RC_RENDER_TARGET: case RC_DEPTH: case RC_STORAGE:
        case RC_CLOUD_VOLUME:                       return 3;
        case RC_AIRCRAFT_COCKPIT: case RC_SHADOW:
        case RC_UI_SMALL:                           return 2;
        case RC_TERRAIN_ARRAY:                      return 1;
        case RC_SCENERY_STREAMED:                   return 0;
        default:                                    return 2;
    }
}

static float scoreOf(const ResRec &r, uint64_t lastUse, uint32_t useCount,
                     uint32_t churnCyclesOfShape, uint32_t sceneCount = 0)
{
    float wCat  = live::f("vram.w_category",  nullptr, 0.40f);
    float wRec  = live::f("vram.w_recency",   nullptr, 0.25f);
    float wFreq = live::f("vram.w_frequency", nullptr, 0.15f);
    float wCost = live::f("vram.w_recreate",  nullptr, 0.10f);
    float wSize = live::f("vram.w_size",      nullptr, 0.10f);

    float wSpat = live::f("vram.w_spatial", nullptr, 0.20f);

    float cat = (float)r.protection / 3.0f;
    float rec = 0.0f;
    if (lastUse) {
        uint64_t age = frameIndex > lastUse ? frameIndex - lastUse : 0;
        rec = age < 60 ? 1.0f : age < 600 ? 0.6f : age < 3600 ? 0.25f : 0.0f;
    }
    float freq = useCount > 200 ? 1.0f : (float)useCount / 200.0f;

    if (useCount)
        freq = 0.5f * freq + 0.5f * ((float)sceneCount / (float)useCount);

    float spat = 0.5f;
    if (g_camValid && r.streamed) {
        float px = g_camX + g_camVX * 180.0f;
        float py = g_camY + g_camVY * 180.0f;
        float pz = g_camZ + g_camVZ * 180.0f;
        float dx = r.birthX - px, dy = r.birthY - py, dz = r.birthZ - pz;
        float d = sqrtf(dx*dx + dy*dy + dz*dz);
        spat = d < 5000.0f ? 1.0f : d < 20000.0f ? 0.6f
             : d < 60000.0f ? 0.3f : 0.0f;
    }
    float cost = churnCyclesOfShape >= 3 ? 1.0f
               : (float)churnCyclesOfShape / 3.0f;
    float sizePen = r.bytes > (64ull << 20) ? 1.0f
                  : (float)r.bytes / (float)(64ull << 20);
    return wCat * cat + wRec * rec + wFreq * freq + wCost * cost
         + wSpat * spat - wSize * sizePen;
}

static std::map<VkImage, VkImage> g_physOf;
static VkImage phys(VkImage app)
{
    std::lock_guard<std::mutex> g(m);
    std::map<VkImage, VkImage>::iterator it = g_physOf.find(app);
    return it == g_physOf.end() ? app : it->second;
}

static uint64_t migCandidates = 0, migProjSaveBytes = 0, migRestoreProj = 0;
static void decide(const ResRec &r, const char *action, const char *reason);
static void migrationTick()
{
    int mode = live::i("vram.migrate", nullptr, 1);
    if (!cfg.enable || mode < 1) return;
    int every = live::i("vram.migrate_every", nullptr, 300);
    if (every < 60) every = 60;
    if (frameIndex % (uint64_t)every) return;

    std::lock_guard<std::mutex> g(m);

    const ResRec *demote = nullptr, *restore = nullptr;
    for (std::map<VkImage, ResRec>::const_iterator it = g_registry.begin();
         it != g_registry.end(); ++it) {
        const ResRec &r = it->second;
        if (!r.streamed || r.protection >= 2 || r.shape.mips <= 1) continue;
        if (r.cls != RC_SCENERY_STREAMED) continue;
        if (zone >= ORANGE && r.state == RS_FULL) {
            if (!demote || r.score < demote->score) demote = &r;
        } else if (zone == GREEN && r.state == RS_REDUCED_AT_CREATE &&
                   restoreReady(r.shape.w, r.shape.h, r.shape.fmt)) {
            if (!restore || r.score > restore->score) restore = &r;
        }
    }
    if (demote) {
        ++migCandidates;
        migProjSaveBytes += demote->bytes - demote->bytes / 4;
        decide(*demote, "MIGRATE-DRY demote",
               "worst score at pressure; top mip cut would free 75% "
               "(execution stage pending dry-run flight data)");
    }
    if (restore) {
        ++migCandidates;
        ++migRestoreProj;
        decide(*restore, "MIGRATE-DRY restore",
               "best-scoring reduced resource, payloads retained - full "
               "quality rebuildable with zero disk I/O");
    }
}

static void noteCamera(float x, float y, float z)
{
    if (!alive()) return;
    if (g_camValid) {
        float vx = x - g_camX, vy = y - g_camY, vz = z - g_camZ;
        if (vx*vx + vy*vy + vz*vz < 1e8f) {
            g_camVX += (vx - g_camVX) * 0.1f;
            g_camVY += (vy - g_camVY) * 0.1f;
            g_camVZ += (vz - g_camVZ) * 0.1f;
        }
    }
    g_camX = x; g_camY = y; g_camZ = z;
    g_camValid = true;
}

static uint64_t decisionsDemote = 0, decisionsRestore = 0;
static void decide(const ResRec &r, const char *action, const char *reason)
{
    int verb = live::i("vram.explain", nullptr, 1);
    bool isRestore = action[0] == 'R' || action[0] == 'r';
    if (isRestore) ++decisionsRestore; else ++decisionsDemote;
    if (verb < 1 || (isRestore && verb < 2)) return;
    static uint64_t said = 0;
    if (++said > 2000 && verb < 2) return;
    trace("VRAMSYS DECIDE #%llu %s: %s %ux%u fmt=%u %.1f MB score=%.2f "
          "state=%s prot=%u zone=%s - %s",
          (unsigned long long)r.id, action, resClassName(r.cls),
          r.shape.w, r.shape.h, r.shape.fmt, r.bytes / 1048576.0, r.score,
          resStateName(r.state), r.protection, zoneName(zone), reason);
}

static void noteImageCreate(VkImage img, uint32_t w, uint32_t h, uint32_t fmt,
                            uint32_t mips, uint64_t bytes,
                            int cat, VkImageUsageFlags usage, uint32_t layers,
                            bool is3D, uint32_t droppedMips, bool *hotOut)
{
    if (hotOut) *hotOut = false;
    if (!alive() || !cfg.enable || !img) return;
    std::lock_guard<std::mutex> g(m);
    ChurnKey k; k.w = w; k.h = h; k.fmt = fmt; k.mips = mips;
    ChurnRec &r = g_churn[k];
    ++r.creates;
    r.bytes = bytes;

    if (r.destroys && frameIndex - r.lastDestroyFrame < 600) {
        ++r.cycles;
        ++churnCycles;
    }

    if (r.lastDrop > 0 && droppedMips == 0) ++mipRestorations;
    r.lastDrop = (uint8_t)droppedMips;

    bool streamed = frameIndex > 900;
    ResRec rec;
    rec.id = g_nextResId++;
    rec.shape = k;
    rec.bytes = bytes;
    rec.createdFrame = frameIndex;
    rec.cls = classify(cat, usage, layers, is3D, streamed, w, h, fmt);
    rec.protection = protectionOfClass(rec.cls);
    rec.state = droppedMips ? RS_REDUCED_AT_CREATE : RS_FULL;
    rec.streamed = streamed;
    rec.score = 0.0f;

    if (frameIndex - g_lastCreateFrame > 120) ++g_burstId;
    g_lastCreateFrame = frameIndex;
    rec.birthX = g_camX; rec.birthY = g_camY; rec.birthZ = g_camZ;
    rec.burst = g_burstId;
    g_registry[img] = rec;
    if (r.cycles >= 3) {
        g_hotImgs.insert(img);
        if (hotOut) *hotOut = true;
    }
}

static void noteImageDestroy(VkImage img)
{
    if (!alive() || !img) return;
    std::lock_guard<std::mutex> g(m);
    std::map<VkImage, ResRec>::iterator it = g_registry.find(img);
    if (it != g_registry.end()) {
        ChurnRec &r = g_churn[it->second.shape];
        ++r.destroys;
        r.lastDestroyFrame = frameIndex;

        uint64_t base = (uint64_t)(uintptr_t)img;
        MipKey from; from.img = base; from.mip = 0;
        for (std::map<MipKey, uint64_t>::iterator ci =
                 g_liveContent.lower_bound(from);
             ci != g_liveContent.end() && ci->first.img == base;) {
            if (g_deadContent.size() < 65536) {
                ShapeMip s; s.w = it->second.shape.w;
                s.h = it->second.shape.h;
                s.fmt = it->second.shape.fmt; s.mip = ci->first.mip;
                g_deadContent[s] = ci->second;
            }
            g_liveContent.erase(ci++);
        }
        g_liveContentCount.store((uint32_t)g_liveContent.size(),
                                 std::memory_order_relaxed);

        if (it->second.shape.w >= 2048 && frameIndex > 900)
            bigDestroyScore += 1.0;
        g_registry.erase(it);
    }
    g_hotImgs.erase(img);
    g_imgLastUse.erase(img);
    g_imgMem.erase(img);
}

static bool churnHot(VkImage img)
{
    if (!alive()) return false;
    std::lock_guard<std::mutex> g(m);
    return g_hotImgs.count(img) != 0;
}

static uint32_t refineDrop(uint32_t baseDrop, uint32_t w, uint32_t h,
                           uint32_t fmt, uint32_t mips, bool streamed)
{
    if (!alive() || !cfg.enable) return baseDrop;
    std::lock_guard<std::mutex> g(m);
    ChurnKey k; k.w = w; k.h = h; k.fmt = fmt; k.mips = mips;
    std::map<ChurnKey, ChurnRec>::iterator it = g_churn.find(k);
    if (it != g_churn.end() && it->second.cycles >= 3) {
        if (baseDrop) ++perResProtected;
        return 0;
    }
    if (!streamed || zone < ORANGE) return baseDrop;

    if ((w & (w - 1)) != 0 || (h & (h - 1)) != 0) return baseDrop;
    uint64_t bytes = it != g_churn.end() && it->second.bytes
                   ? it->second.bytes : (uint64_t)w * h * 4;
    uint32_t extra = 0;
    if (zone == ORANGE)      extra = bytes > (32ull << 20) ? 1 : 0;
    else if (zone == RED)    extra = bytes > (16ull << 20) ? 1 : 0;
    else       extra = bytes > (8ull << 20) ? 2 : 1;
    uint32_t total = baseDrop + extra;
    if (total >= mips) total = mips > 1 ? mips - 1 : 0;
    if (total > baseDrop) ++perResExtraCut;
    return total;
}

static void notePipelines(uint32_t count, uint64_t us)
{
    if (!alive()) return;
    pipesTotal.fetch_add(count);
    pipeUsTotal.fetch_add(us);
    if (frameIndex > 0) {
        pipesLive.fetch_add(count);
        pipesFrame.fetch_add(count);
    }
}

static void noteDescriptorUpdates(uint32_t n) { descUpdFrame.fetch_add(n); }
static void noteDescriptorAllocs(uint32_t n)  { descAllocFrame.fetch_add(n); }
static void notePipelineBind()                { if (alive()) pipeBindsFrame.fetch_add(1); }

static void noteImageUse(VkImage img, bool inScenePass = false)
{
    if (!alive()) return;
    std::lock_guard<std::mutex> g(m);
    UseRec &r = g_imgLastUse[img];
    r.last = frameIndex;
    if (r.count < 0xFFFFFFFFu) ++r.count;

    if (inScenePass && r.sceneCount < 0xFFFFFFFFu) ++r.sceneCount;
}

static void noteImageMem(VkImage img, VkDeviceMemory mem)
{
    if (!alive()) return;
    std::lock_guard<std::mutex> g(m);
    g_imgMem[img] = mem;
}

static void noteMap(VkDeviceMemory mem, VkDeviceSize offset, VkDeviceSize size,
                    void *ptr)
{
    if (!alive() || !ptr) return;
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
    if (!alive()) return;
    std::lock_guard<std::mutex> g(m);
    BufBind b; b.mem = mem; b.offset = offset;
    g_bufBind[buf] = b;
}

static void noteBufferGone(VkBuffer buf)
{
    std::lock_guard<std::mutex> g(m);
    g_bufBind.erase(buf);
}

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

static uint64_t contentHash(const uint8_t *p, uint64_t len)
{
    uint64_t h = 1469598103934665603ull ^ len;
    uint64_t stride = len > 4096 ? (len / 4096) & ~7ull : 1;
    if (!stride) stride = 1;
    for (uint64_t i = 0; i < len; i += stride)
        h = (h ^ p[i]) * 1099511628211ull;

    for (uint64_t i = len > 64 ? len - 64 : 0; i < len; ++i)
        h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

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
    g_liveContentCount.store((uint32_t)g_liveContent.size(),
                             std::memory_order_relaxed);

    std::map<VkImage, ResRec>::iterator ck = g_registry.find(img);
    if (ck != g_registry.end()) {
        ShapeMip s; s.w = ck->second.shape.w; s.h = ck->second.shape.h;
        s.fmt = ck->second.shape.fmt; s.mip = mip;
        std::map<ShapeMip, uint64_t>::iterator dc = g_deadContent.find(s);
        if (dc != g_deadContent.end() && dc->second == hash) {
            ++reloadIdenticalCount;
            reloadIdenticalBytes += bytes;
        }
    }
    return false;
}

static void contentInvalidate(VkImage img)
{
    if (!alive() || !img) return;

    if (g_liveContentCount.load(std::memory_order_relaxed) == 0) return;
    std::lock_guard<std::mutex> g(m);
    if (g_liveContent.empty()) return;
    uint64_t base = (uint64_t)(uintptr_t)img;
    MipKey from; from.img = base; from.mip = 0;
    for (std::map<MipKey, uint64_t>::iterator it = g_liveContent.lower_bound(from);
         it != g_liveContent.end() && it->first.img == base;)
        g_liveContent.erase(it++);
    g_liveContentCount.store((uint32_t)g_liveContent.size(),
                             std::memory_order_relaxed);
}

static void noteUploadRegion(uint64_t imgHandle, uint32_t mip)
{
    if (!alive()) return;
    std::lock_guard<std::mutex> g(m);
    if (!g_upSeen.insert(std::make_pair(imgHandle, mip)).second) ++dupUploads;
}

static void ledgerRt(uint64_t bytes) { rtBytesNow = bytes; }

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
        else if (sscanf(line, "age_bias_frames=%llu", &v) == 1)
            ageBiasFrames = (int)(v > 3600 ? 3600 : v);
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
               "upload_peak_mb=%llu\n"
               "age_bias_frames=%d\n",
            (unsigned long long)failsEver, reserveBiasMB,
            (unsigned long long)(upBytesPeak / 1048576ull), ageBiasFrames);
    fclose(f);
}

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
    if (!alive()) return;
    if (family != dev.transferFamily || dev.transferFamily == ~0u) return;
    std::lock_guard<std::mutex> g(m);
    if (g_transferQueues.insert(q).second) {
        g_queueLock[q] = new std::mutex;
        trace("VRAMSYS: transfer queue %p (family %u) is under the governor",
              (void*)q, family);
    }
}

static void ledgerTotal(uint64_t bytes) { ledgerBytesNow.store(bytes); }

static void chargeCopy(VkCommandBuffer cb, uint64_t bytes, uint8_t prot = 0)
{
    if (!alive() || !cfg.enable) return;
    upBytesFrame.fetch_add(bytes);
    std::lock_guard<std::mutex> g(m);
    g_cbBytes[cb] += bytes;

    uint8_t &p = g_cbProt[cb];
    if (prot > p) p = prot;
}

static uint8_t protectionOf(VkImage img)
{
    if (!alive()) return 0;
    std::lock_guard<std::mutex> g(m);
    std::map<VkImage, ResRec>::iterator it = g_registry.find(img);
    return it == g_registry.end() ? 0 : it->second.protection;
}

static bool poolTake(const VkMemoryAllocateInfo *ai, VkDeviceMemory *out)
{
    if (!alive() || !cfg.enable || !cfg.recycle || !ai || ai->pNext) return false;
    std::lock_guard<std::mutex> g(m);
    for (size_t i = 0; i < g_pool.size(); ++i) {
        if (g_pool[i].type != ai->memoryTypeIndex) continue;
        if (g_pool[i].size != ai->allocationSize)  continue;
        *out = g_pool[i].mem;
        g_poolBytes -= g_pool[i].size;
        g_pool.erase(g_pool.begin() + (long)i);
        ++recycleHits;

        return true;
    }
    ++recycleMisses;
    return false;
}

static bool poolHold(VkDeviceMemory mem)
{
    if (!alive() || !cfg.enable || !cfg.recycle) return false;
    std::lock_guard<std::mutex> g(m);
    std::map<VkDeviceMemory, AllocRec>::iterator it = g_allocs.find(mem);
    if (it == g_allocs.end() || !it->second.plain) return false;
    bool devLocal = typeRecyclable(it->second.type);
    bool hostOk   = false;
    if (!devLocal && it->second.type < dev.memProps.memoryTypeCount) {
        VkMemoryPropertyFlags f =
            dev.memProps.memoryTypes[it->second.type].propertyFlags;

        hostOk = (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                 (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    if (!devLocal && !hostOk) return false;
    if (zone >= RED && devLocal) return false;
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
    g_blockPrio.erase(mem);
    return true;
}

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

struct PrioChain { VkMemoryPriorityAllocateInfoEXT info; VkMemoryAllocateInfo ai; };
static bool prioTag(const VkMemoryAllocateInfo *ai, PrioChain *pc)
{
    if (!alive() || !cfg.enable || !cfg.priority || !dev.priorityExt) return false;
    pc->info.sType = VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT;
    pc->info.pNext = (void*)ai->pNext;
    pc->info.priority = 0.5f;
    pc->ai = *ai;
    pc->ai.pNext = &pc->info;
    ++prioAllocTagged;
    return true;
}

static void noteAlloc(VkDeviceMemory mem, const VkMemoryAllocateInfo *ai)
{
    if (!alive()) return;
    allocsTotal.fetch_add(1);
    if (!ai) return;
    std::lock_guard<std::mutex> g(m);
    AllocRec r; r.size = ai->allocationSize; r.type = ai->memoryTypeIndex;
    r.plain = (ai->pNext == nullptr);
    g_allocs[mem] = r;
}

static void noteFreeGone(VkDeviceMemory mem)
{
    if (!alive()) return;
    freesTotal.fetch_add(1);
    std::lock_guard<std::mutex> g(m);
    g_allocs.erase(mem);
    g_blockPrio.erase(mem);
}

static void onBind(VkDeviceMemory mem, int cat, bool streamed, bool hot = false)
{
    if (!cfg.enable || !cfg.priority || !dev.pageableExt || !dev.dev) return;
    float p = prioOfCat(cat, streamed);
    if (hot && p < 0.65f) p = 0.65f;
    bool  set = false;
    {
        std::lock_guard<std::mutex> g(m);
        BlockPrio &bp = g_blockPrio[mem];
        if (bp.best == 0.0f) {
            bp.best = p; bp.streamedTexOnly = (cat == 0 && streamed);
            bp.demoted = false; set = true;
        } else {
            if (!(cat == 0 && streamed)) bp.streamedTexOnly = false;
            if (p > bp.best) { bp.best = p; set = true; }
        }
    }
    if (set) { dev.setPriority(dev.dev, mem, p); ++prioBindSet; }
}

static void prioZoneWalk()
{
    if (!cfg.enable || !cfg.priority || !dev.pageableExt || !dev.dev) return;
    std::vector<std::pair<VkDeviceMemory, float> > moves;
    {
        std::lock_guard<std::mutex> g(m);

        std::map<VkDeviceMemory, float> blockScore;
        for (std::map<VkImage, VkDeviceMemory>::iterator im = g_imgMem.begin();
             im != g_imgMem.end(); ++im) {
            std::map<VkImage, ResRec>::iterator rr = g_registry.find(im->first);
            if (rr == g_registry.end()) continue;
            uint64_t lastUse = 0; uint32_t useCount = 0;
            std::map<VkImage, UseRec>::iterator ur = g_imgLastUse.find(im->first);
            if (ur != g_imgLastUse.end()) { lastUse = ur->second.last;
                                            useCount = ur->second.count; }
            uint32_t cyc = 0;
            std::map<ChurnKey, ChurnRec>::iterator cr =
                g_churn.find(rr->second.shape);
            if (cr != g_churn.end()) cyc = cr->second.cycles;
            uint32_t sceneN = 0;
            if (ur != g_imgLastUse.end()) sceneN = ur->second.sceneCount;
            float s = scoreOf(rr->second, lastUse, useCount, cyc, sceneN);
            rr->second.score = s;
            std::map<VkDeviceMemory, float>::iterator bs =
                blockScore.find(im->second);
            if (bs == blockScore.end() || s > bs->second)
                blockScore[im->second] = s;
        }
        std::vector<std::pair<float, VkDeviceMemory> > cand;
        for (std::map<VkDeviceMemory, BlockPrio>::iterator it = g_blockPrio.begin();
             it != g_blockPrio.end(); ++it) {
            if (!it->second.streamedTexOnly) continue;
            bool wantDemote  = zone >= RED  && !it->second.demoted;
            bool wantRestore = zone == GREEN && it->second.demoted;
            if (!wantDemote && !wantRestore) continue;
            std::map<VkDeviceMemory, float>::iterator bs = blockScore.find(it->first);
            float s = bs != blockScore.end() ? bs->second : 0.0f;
            cand.push_back(std::make_pair(wantDemote ? s : -s, it->first));
        }

        std::sort(cand.begin(), cand.end());
        int budget = 64;
        for (size_t i = 0; i < cand.size() && budget > 0; ++i, --budget) {
            std::map<VkDeviceMemory, BlockPrio>::iterator it =
                g_blockPrio.find(cand[i].second);
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

        for (size_t i = 0; i < moves.size(); ++i) {
            bool demote = moves[i].second <= 0.21f;
            for (std::map<VkImage, VkDeviceMemory>::iterator im = g_imgMem.begin();
                 im != g_imgMem.end(); ++im) {
                if (im->second != moves[i].first) continue;
                std::map<VkImage, ResRec>::iterator rr = g_registry.find(im->first);
                if (rr == g_registry.end()) continue;
                rr->second.state = demote ? RS_DEMOTED : RS_FULL;
                decide(rr->second, demote ? "DEMOTE" : "RESTORE",
                       demote ? "zone pressure, lowest score in walk budget"
                              : "pressure cleared");
            }
        }
    }
    for (size_t i = 0; i < moves.size(); ++i)
        dev.setPriority(dev.dev, moves[i].first, moves[i].second);
    prioZoneMoves += moves.size();
}

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

            {
                std::map<VkImage, ResRec>::iterator rg = g_registry.find(cursor);
                if (rg != g_registry.end()) {
                    if (frameIndex - rg->second.createdFrame < 600) continue;

                    if (rg->second.score > 0.6f) continue;
                }
            }
            uint64_t window = (uint64_t)cfg.ageFrames + (uint64_t)ageBiasFrames;
            if (it->second.count > 100) window *= 2;
            bool stale = frameIndex > it->second.last &&
                         frameIndex - it->second.last > window;
            if (stale && !bp->second.aged && !bp->second.demoted) {
                bp->second.aged = true;
                moves.push_back(std::make_pair(im->second, 0.25f));
                ++agedDemotions;
                std::map<VkImage, ResRec>::iterator rr = g_registry.find(cursor);
                if (rr != g_registry.end()) {
                    rr->second.state = RS_AGED;
                    decide(rr->second, "DEMOTE",
                           "unused past the age window (frequency-weighted)");
                }
            } else if (!stale && bp->second.aged && !bp->second.demoted) {
                bp->second.aged = false;
                moves.push_back(std::make_pair(im->second, bp->second.best));
                ++agedRestores;
                std::map<VkImage, ResRec>::iterator rr = g_registry.find(cursor);
                if (rr != g_registry.end()) {
                    rr->second.state = RS_FULL;
                    decide(rr->second, "RESTORE", "used again after aging");
                }
            }
        }
        if (it == g_imgLastUse.end()) cursor = VK_NULL_HANDLE;
    }
    for (size_t i = 0; i < moves.size(); ++i)
        dev.setPriority(dev.dev, moves[i].first, moves[i].second);
}

static void flushLocked(std::vector<HeldSubmit> &out, VkQueue only )
{

    for (size_t i = 0; i < g_heldSubmits.size();) {
        if (only == VK_NULL_HANDLE || g_heldSubmits[i].q == only) {
            out.push_back(g_heldSubmits[i]);
            g_heldSubmits.erase(g_heldSubmits.begin() + (long)i);
        } else ++i;
    }
    if (only == VK_NULL_HANDLE || g_heldSubmits.empty()) {
        if (only == VK_NULL_HANDLE) { g_heldFences.clear(); g_heldSignals.clear(); }
    }

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

static void touchFences(uint32_t count, const VkFence *fences)
{
    if (!alive() || !count || !fences) return;
    bool hit = false;
    {
        std::lock_guard<std::mutex> g(m);
        if (g_heldFences.empty()) return;
        for (uint32_t i = 0; i < count && !hit; ++i)
            if (g_heldFences.count(fences[i])) hit = true;
    }
    if (hit) flushAll(&flushOnWait);
}

static bool anyHeldSignalLocked(uint32_t count, const VkSemaphore *sems,
                                const uint64_t *values)
{
    for (uint32_t i = 0; i < count; ++i) {
        std::map<VkSemaphore, uint64_t>::iterator it = g_heldSignals.find(sems[i]);
        if (it == g_heldSignals.end()) continue;
        if (!values || !it->second) return true;
        if (values[i] <= it->second) return true;

        return true;
    }
    return false;
}

static void touchSemaphores(uint32_t count, const VkSemaphore *sems,
                            const uint64_t *values)
{
    if (!alive() || !count || !sems) return;
    bool hit = false;
    {
        std::lock_guard<std::mutex> g(m);
        for (uint32_t i = 0; i < count; ++i)
            if (g_everWaited.size() < 4096) g_everWaited.insert(sems[i]);
        if (g_heldSignals.empty()) return;
        hit = anyHeldSignalLocked(count, sems, values);
    }
    if (hit) flushAll(&flushOnDep);
}

static bool onSubmit(VkQueue q, uint32_t count, const VkSubmitInfo *submits,
                     VkFence fence, VkResult *result)
{
    if (!alive() || !cfg.enable) return false;

    if (submits) {
        bool dep = false;
        {
            std::lock_guard<std::mutex> g(m);

            for (uint32_t s = 0; s < count; ++s)
                for (uint32_t w = 0; w < submits[s].waitSemaphoreCount; ++w)
                    if (g_everWaited.size() < 4096)
                        g_everWaited.insert(submits[s].pWaitSemaphores[w]);
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

    uint64_t bytes = 0;
    uint8_t  maxProt = 0;
    bool holdable = true;
    for (uint32_t s = 0; s < count && submits; ++s) {
        for (uint32_t c = 0; c < submits[s].commandBufferCount; ++c) {
            std::lock_guard<std::mutex> g(m);
            std::map<VkCommandBuffer, uint64_t>::iterator it =
                g_cbBytes.find(submits[s].pCommandBuffers[c]);
            if (it != g_cbBytes.end()) { bytes += it->second; g_cbBytes.erase(it); }
            std::map<VkCommandBuffer, uint8_t>::iterator pt =
                g_cbProt.find(submits[s].pCommandBuffers[c]);
            if (pt != g_cbProt.end()) {
                if (pt->second > maxProt) maxProt = pt->second;
                g_cbProt.erase(pt);
            }
        }

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

        bool overCap = heldBytesNow + bytes >
                       (uint64_t)cfg.holdMaxMB * 1048576ull;
        bool over = (g_frameUploadSpent + bytes) > budget;

        if (holdable)
            for (uint32_t s = 0; s < count && holdable; ++s)
                for (uint32_t k = 0; k < submits[s].signalSemaphoreCount; ++k)
                    if (g_everWaited.count(submits[s].pSignalSemaphores[k]))
                        { holdable = false; break; }

        if (maxProt >= 2) { over = false; ++governorBypassed; }
        if ((!over && !mustQueueBehind) || !holdable || overCap ||
            zone < ORANGE) {

            g_frameUploadSpent += bytes;
        } else {

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
            return true;
        }
    }

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

            if (legacyScale > 1.0f) {
                VkDeviceSize want = (VkDeviceSize)(b->heapBudget[i] * legacyScale);
                if (want > heapSize) want = heapSize;
                if (want > b->heapBudget[i]) b->heapBudget[i] = want;
            }
            break;
        }

        if (filteredBudget <= 0.0) filteredBudget = (double)rawBudget;
        filteredBudget += ((double)rawBudget - filteredBudget) * cfg.budgetAlpha;

        uint64_t report = (uint64_t)filteredBudget;

        uint64_t led = ledgerBytesNow.load();
        bool appFreeing = led + (16ull << 20) < lastLedgerBytes;
        lastLedgerBytes = led;
        if (appFreeing && lastReported && report < lastReported)
            report = lastReported - lastReported / 1000;

        double speedFactor = camSpeedEma * cfg.speedReserve;
        if (speedFactor > 1.0) speedFactor = 1.0;
        uint64_t reserve = (uint64_t)(((double)cfg.reserveMB[zone] +
                                       (double)reserveBiasMB) *
                                      (1.0 + speedFactor)) * 1048576ull;

        uint64_t warmFrames = (uint64_t)(cfg.warmupFrames > 0
                                         ? cfg.warmupFrames : 0);
        if (warmFrames && frameIndex - warmupStartFrame < warmFrames) {
            double t = 1.0 - (double)(frameIndex - warmupStartFrame) /
                             (double)warmFrames;
            reserve += (uint64_t)((double)cfg.warmupMB * t) * 1048576ull;
        }
        report = report > reserve ? report - reserve : 0;

        if (deflateLeft > 0) {
            uint64_t cut = (uint64_t)cfg.deflateMB * 1048576ull;
            report = report > cut ? report - cut : 0;
        }

        if (legacyScale > 1.0f) {
            VkDeviceSize want = (VkDeviceSize)(report * legacyScale);
            report = want;
        }

        uint64_t floorB = rawUsage + (64ull << 20);
        if (report < floorB)  report = floorB;
        if (report > heapSize) report = heapSize;

        lastReported = report;
        b->heapBudget[i] = report;
        break;
    }
}

static bool emergency()
{
    allocFails.fetch_add(1);
    uint64_t before;
    {
        std::lock_guard<std::mutex> g(m);
        before = g_poolBytes;
        deflateLeft = cfg.deflateFrames;
        zone = CRITICAL; zoneDwell = 0;

        ++failsEver;
        if (reserveBiasMB < 256) reserveBiasMB += 64;
    }
    stateSave();
    flushAll(&flushOnDep);
    poolTrim(true);
    trace("VRAMSYS: EMERGENCY - allocation failed; recycle pool flushed "
          "(%.1f MB reclaimed), budget deflated %d MB for %d frames, zone "
          "CRITICAL", before / 1048576.0, cfg.deflateMB, cfg.deflateFrames);
    return before > 0;
}

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
    cfg.uploadBudgetMB[GREEN]  = live::i("vram.upload_g", nullptr, 0);
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
            frameBestMs *= 1.0001;

            if (frameBestMs < 4.0) frameBestMs = 4.0;
            if (cfg.adaptive && g_camValid) {
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

    double basis = filteredBudget > 0 ? filteredBudget : (double)heapSize;
    if (basis <= 0) return;

    double delta = (double)rawUsage - (double)prevRawUsage;
    prevRawUsage = rawUsage;
    if (delta > -1e9 && delta < 1e9)
        usageTrendBytes += (delta - usageTrendBytes) * 0.05;
    double trendTerm = usageTrendBytes > 0
                     ? usageTrendBytes * (double)cfg.lookaheadFrames : 0.0;

    if (trendTerm > basis * 0.25) trendTerm = basis * 0.25;
    if ((double)rawUsage < basis * 0.5) trendTerm = 0.0;
    double projected = (double)rawUsage + (double)upBytesLast * 2.0 +
                       (double)heldBytesNow + trendTerm - (double)g_poolBytes;
    if (projected < 0) projected = 0;
    double f = projected / basis;

    static const double enter[5] = { 0.0, 0.80, 0.88, 0.94, 0.98 };
    static const double exitT[5] = { 0.0, 0.75, 0.83, 0.90, 0.95 };

    int want = zone;
    while (want < CRITICAL && f >= enter[want + 1]) ++want;
    while (want > GREEN && f < exitT[want]) --want;

    if (teleportLeft > 0 && want < YELLOW) want = YELLOW;

    if (want > zone) { zone = want; zoneDwell = 0; ++zoneFlips;
        trace("VRAMSYS: zone %s (%.0f%% of %.2f GB shaped basis, usage %.2f GB "
              "+ uploads %.1f MB/f)", zoneName(zone), f * 100.0,
              basis / 1073741824.0, rawUsage / 1073741824.0,
              upBytesLast / 1048576.0);
    } else if (want < zone) {
        if (++zoneDwell >= 60) {
            zone = want; zoneDwell = 0; ++zoneFlips;
            trace("VRAMSYS: zone %s (%.0f%%)", zoneName(zone), f * 100.0);
        }
    } else zoneDwell = 0;
}

static void onPresent(float camDeltaMeters, float camRotDeg = -1.0f)
{
    if (!alive()) return;
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
        g_upSeen.clear();
    }

    frameTimeTick();

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

    if (fi % 3600 == 0) stateSave();

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

    if (camRotDeg >= 0.0f && camRotDeg < 45.0f) {
        camRotEma += (camRotDeg - camRotEma) * 0.1;
        if (camRotEma > live::f("vram.rot_floor_deg", nullptr, 1.0f)) {
            std::lock_guard<std::mutex> g(m);
            if (teleportLeft < 120) teleportLeft = 120;
        }
    }

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

    std::vector<HeldSubmit> release;
    {
        std::lock_guard<std::mutex> g(m);
        uint64_t budget = zoneUploadBudget();
        for (size_t i = 0; i < g_heldSubmits.size();) {
            bool aged = fi - g_heldSubmits[i].frame >= (uint64_t)cfg.uploadMaxHold;
            bool fits = g_frameUploadSpent + g_heldSubmits[i].bytes <= budget;

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
    migrationTick();

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
        trace("  camera: %.2f m/frame EMA, %.2f deg/frame rotation EMA, "
              "%llu teleports;  sparse binds %llu",
              camSpeedEma, camRotEma, (unsigned long long)teleports,
              (unsigned long long)sparseBinds);
        trace("  migration (stage 1, %s): %llu candidates selected, %.1f MB "
              "projected demote savings, %llu restores rebuildable from "
              "retention - execution lands on this data",
              live::i("vram.migrate", nullptr, 1) ? "DRY RUN" : "off",
              (unsigned long long)migCandidates,
              migProjSaveBytes / 1048576.0,
              (unsigned long long)migRestoreProj);
        trace("  spatial: %u load bursts, camera velocity (%.1f, %.1f, %.1f) "
              "m/frame - streamed resources score against the position 3 s "
              "ahead;  indirection table %llu entries (identity until the "
              "migration executor writes it)",
              g_burstId, g_camVX, g_camVY, g_camVZ,
              (unsigned long long)g_physOf.size());
        {

            uint64_t largest = 0;
            for (size_t i = 0; i < g_pool.size(); ++i)
                if (g_pool[i].size > largest) largest = g_pool[i].size;
            trace("  pool composition: %llu blocks, largest %.1f MB of %.1f "
                  "MB held;  age bias %d frames (learned)",
                  (unsigned long long)g_pool.size(), largest / 1048576.0,
                  g_poolBytes / 1048576.0, ageBiasFrames);
        }
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
        {
            uint64_t ready = 0, reduced = 0;
            for (std::map<VkImage, ResRec>::iterator it = g_registry.begin();
                 it != g_registry.end(); ++it) {
                if (it->second.state != RS_REDUCED_AT_CREATE) continue;
                ++reduced;
                if (restoreReady(it->second.shape.w, it->second.shape.h,
                                 it->second.shape.fmt)) ++ready;
            }
            trace("  retention: %.1f MB in %llu payloads (%llu refused at "
                  "cap); %llu of %llu REDUCED resources are RESTORE-READY - "
                  "full quality is rebuildable without the disk",
                  retainedBytes / 1048576.0, (unsigned long long)retainStores,
                  (unsigned long long)retainRefused,
                  (unsigned long long)ready, (unsigned long long)reduced);
        }
        trace("  per-resource quality: %llu shapes protected from cuts "
              "(churners), %llu extra cuts (large streamed under pressure), "
              "%llu full-quality restorations observed; %llu zone flips",
              (unsigned long long)perResProtected,
              (unsigned long long)perResExtraCut,
              (unsigned long long)mipRestorations,
              (unsigned long long)zoneFlips);
        trace("  aging: %llu tracked images, %llu decayed, %llu restored "
              "(window %d frames, doubled for frequent users)",
              (unsigned long long)g_imgLastUse.size(),
              (unsigned long long)agedDemotions,
              (unsigned long long)agedRestores, cfg.ageFrames);
        {

            uint64_t byClass[10] = {0}, clsBytes[10] = {0};
            uint64_t st[5] = {0};
            for (std::map<VkImage, ResRec>::iterator it = g_registry.begin();
                 it != g_registry.end(); ++it) {
                if (it->second.cls < 10) {
                    ++byClass[it->second.cls];
                    clsBytes[it->second.cls] += it->second.bytes;
                }
                if (it->second.state < 5) ++st[it->second.state];
            }
            trace("  registry: %llu resources (next id %llu) - states: "
                  "%llu FULL, %llu REDUCED, %llu DEMOTED, %llu AGED",
                  (unsigned long long)g_registry.size(),
                  (unsigned long long)g_nextResId,
                  (unsigned long long)st[RS_FULL],
                  (unsigned long long)st[RS_REDUCED_AT_CREATE],
                  (unsigned long long)st[RS_DEMOTED],
                  (unsigned long long)st[RS_AGED]);
            for (int c = 0; c < 10; ++c)
                if (byClass[c])
                    trace("    class %-18s %6llu  %8.1f MB", resClassName(c),
                          (unsigned long long)byClass[c],
                          clsBytes[c] / 1048576.0);
            trace("  decisions: %llu demote, %llu restore (vram.explain=%d); "
                  "governor bypassed %llu protected uploads",
                  (unsigned long long)decisionsDemote,
                  (unsigned long long)decisionsRestore,
                  live::i("vram.explain", nullptr, 1),
                  (unsigned long long)governorBypassed);
        }
        {

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

static void shutdown()
{
    flushAll(&flushOnDep);
    poolTrim(true);

    if (zoneFlips > 200 && reserveBiasMB < 256) {
        reserveBiasMB += 32;
        trace("VRAMSYS: %llu zone transitions this session - reserve bias "
              "raised to +%d MB for future sessions",
              (unsigned long long)zoneFlips, reserveBiasMB);
    }

    if (agedDemotions > 50 && agedRestores * 2 > agedDemotions &&
        ageBiasFrames < 3600) {
        ageBiasFrames += 600;
        trace("VRAMSYS: %llu of %llu aged resources came back - age window "
              "extended by %d frames for future sessions",
              (unsigned long long)agedRestores,
              (unsigned long long)agedDemotions, ageBiasFrames);
    }
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

}
