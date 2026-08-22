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

#include <direct.h>   // _mkdir for the screenshot folder
// VK_LAYER_taa_impl
//
// The GPU half of TAAImplementation. Stage 1: READ-ONLY.
//
// Every call is forwarded unmodified, so this cannot change rendering or break
// the sim. That is deliberate. Before writing any GPU code there are facts about
// X-Plane's frame that have to be established rather than guessed:
//
//   1. Which image is the scene depth buffer, and what are its format, sample
//      count and extent?
//   2. Does that image already carry VK_IMAGE_USAGE_SAMPLED_BIT? A compute
//      shader cannot read it otherwise, and if it is missing we have to add it
//      at creation time - a modification, and modifying resources the app owns
//      is where this kind of layer gets unstable.
//   3. Is it multisampled? MSAA depth needs a resolve before it can be sampled.
//   4. Where does the 3D scene end and the UI begin, so the resolve lands
//      before instrument text and ATC boxes get temporally smeared?
//
// Answering these read-only first is the pattern that works. Modifying the
// frame before understanding it is where things go wrong.
//
// Unlike that project this one builds against the REAL Vulkan headers. The
// hand-written ABI there had a wrong constant (LOADER_INSTANCE_CREATE_INFO is
// 47, not 1000009000) that surfaced only as a mystery runtime failure. With the
// genuine declarations that entire class of bug becomes a compile error.
//
// Output: %TEMP%\taa_layer.txt

#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR

#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>

// Not defined by the SDK headers under MinGW. The loader resolves the two entry
// points named in the manifest by symbol name, so they must actually be
// exported from the DLL - without this the layer loads and then silently does
// nothing, which is hard to diagnose.
#ifndef VK_LAYER_EXPORT
  #define VK_LAYER_EXPORT __declspec(dllexport)
#endif

#include <windows.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdarg>
#include <string>
#include <map>
#include <set>
#include <mutex>
#include <atomic>
#include <vector>
#include <cmath>

#include "../share.h"

// RenderDoc's in-application API (header shipped with the install, MIT). Only
// live when the sim was launched under RenderDoc - GetModuleHandle finds the
// injected dll - and gives us the one thing the CLI cannot: multi-frame
// captures, triggered from inside via the live file instead of F12.
#include "renderdoc_app.h"

// ---------------------------------------------------------------- tracing

static std::mutex g_traceLock;

static void trace(const char *fmt, ...)
{
    static const bool on = getenv("TAA_LAYER_TRACE") != nullptr;
    if (!on) return;

    static std::string path;
    if (path.empty()) {
        const char *t = getenv("TEMP");
        path = std::string(t ? t : ".") + "\\" MV_TRACE_FILE;
    }

    std::lock_guard<std::mutex> g(g_traceLock);
    static FILE *f = nullptr;
    if (!f) f = fopen(path.c_str(), "a");
    if (!f) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
    fputc('\n', f);
    fflush(f);   // flushed, not closed: durability without an open and close
                 // per line, which cost 63 MB of file I/O and most of the fps.
}

// Live controls. Included here rather than with the other headers because it
// calls trace(), which is defined directly above.
#include "mv_live.h"

// The VRAM system - zones, budget shaping, recycle pool, memory priorities,
// upload governor, predictor, emergency ladder. Same placement logic: it
// calls trace() and live::, both defined above.
#include "vram.h"
#include "upscaler_policy.h"
#include "xess_probe.h"
#include "../destruct/bounds.h"
#include "../destruct/voxelise.h"

// ------------------------------------------------------- shared memory

static HANDLE    g_shareHandle = nullptr;
static TaaShare *g_share       = nullptr;
static bool      g_shareOpen   = false;

// MUST retry. Vulkan initialises long before X-Plane loads plugins, so the
// first attempt always fails - the plugin has not created the mapping yet.
// Trying once and giving up meant the sibling project's layer never saw the
// camera at all, which took a while to spot because "attached" was only logged
// on success.
static void openShare()
{
    if (g_shareOpen) return;
    static int attempts = 0;
    if (++attempts % 256 != 1) return;   // throttle, but keep trying

    // READ/WRITE, not read-only.
    //
    // The layer is mostly a consumer of this block, and it was opened read-only
    // to make that structurally true. But the contract has always had a status
    // section the LAYER is supposed to fill in - layerAttached, the GPU name,
    // per-backend availability - and a read-only mapping cannot write it. The
    // in-sim UI therefore showed every backend as "not attached" no matter what
    // was actually running, which is worse than useless: it is a display that
    // reports the same thing whether the layer is working or absent.
    g_shareHandle = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, TAA_SHARE_NAME);
    if (!g_shareHandle) {
        if (attempts == 1) trace("SHARE: not published yet, will retry");
        return;
    }

    TaaShare *s = (TaaShare*)MapViewOfFile(g_shareHandle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(TaaShare));
    if (!s) { trace("SHARE: MapViewOfFile failed %lu", GetLastError()); return; }
    if (s->magic != TAA_MAGIC) { trace("SHARE: bad magic 0x%08X", s->magic); return; }

    // Version AND size must both match. A mismatch means plugin and layer were
    // built from different sources; reading on would produce a plausible-looking
    // but wrong velocity field, which is far worse than refusing.
    if (s->version != TAA_VERSION || s->structSize != sizeof(TaaShare)) {
        trace("SHARE: MISMATCH - plugin v%u/%uB, layer v%u/%uB. Rebuild both. Disabled.",
              s->version, s->structSize, (unsigned)TAA_VERSION, (unsigned)sizeof(TaaShare));
        UnmapViewOfFile(s);
        g_shareOpen = true;   // latch: do not spam, do not use
        return;
    }

    g_share     = s;
    g_shareOpen = true;
    trace("SHARE: attached (v%u, %u bytes)", s->version, s->structSize);
}

// What the velocity pass needs, snapshotted coherently. The plugin writes
// without a lock, so reading fields inline at dispatch time could tear across a
// frame boundary - matrix from frame N, reset flag from N+1. The plugin bumps
// `frame` only after everything else is written, so a matching counter either
// side of the copy proves we did not tear. Cheaper and safer than a
// cross-process mutex.
struct Snapshot {
    bool     valid;
    uint64_t frame;
    float    reproj[16], invCurrViewProj[16], prevViewProj[16];
    // The view and projection matrices as published, so the layer can pair its
    // OWN consecutive frames rather than trusting the plugin's flight-loop
    // pairing. Kept separate rather than pre-multiplied: the origin shift has
    // to happen before the projection, or the 52 km of world translation is
    // cancelled after it has been scaled and four significant digits go with it.
    float    world[16], proj[16], prevProj[16];
    float    bodyReproj[16];
    int32_t  bodyReprojValid;
    float    camBodyDrift, camGap;
    int32_t  selfTestPhase;
    // Which view the frame was drawn from. The residual is measured per frame
    // and reported as one number; without this there is no way to tell a
    // cockpit frame from an external one after the fact, and the two behave
    // completely differently - a capture showed 0.006 px with the aeroplane
    // small in frame and 300.020 px with it filling the frame.
    int32_t  viewType;
    float    selfTestExpectedPx;
    int32_t  reverseZ, historyReset, resetReason;
    int32_t  viewportW, viewportH;
    float    nearClip, farClip;
    float    fovDeg;

    // Milliseconds, not seconds.
    float    frameTimeMs;
    double   simTime;
    float    jitterX, jitterY;
    int32_t  jitterIndex, jitterPhases;
    float    lodBias;
    float    camX, camY, camZ, camDelta;
    int32_t  objectCount;
    TaaMovingObject    objects[TAA_MAX_OBJECTS];
};

static bool snapshot(Snapshot *o)
{
    o->valid = false;
    if (!g_share || !g_share->valid) return false;

    for (int attempt = 0; attempt < 4; ++attempt) {
        uint64_t before = g_share->frame;

        memcpy(o->reproj,          g_share->reproj,          sizeof(o->reproj));
        memcpy(o->invCurrViewProj, g_share->invCurrViewProj, sizeof(o->invCurrViewProj));
        memcpy(o->prevViewProj,    g_share->prevViewProj,    sizeof(o->prevViewProj));
        memcpy(o->world,           g_share->world,           sizeof(o->world));
        memcpy(o->proj,            g_share->proj,            sizeof(o->proj));
        memcpy(o->prevProj,        g_share->prevProj,        sizeof(o->prevProj));
        memcpy(o->bodyReproj,      g_share->bodyReproj,      sizeof(o->bodyReproj));
        o->bodyReprojValid = g_share->bodyReprojValid;
        o->camBodyDrift    = g_share->camBodyDrift;
        o->camGap          = g_share->camGap;
        o->selfTestPhase      = g_share->selfTestPhase;
        o->viewType           = g_share->viewType;
        o->selfTestExpectedPx = g_share->selfTestExpectedPx;
        o->reverseZ      = g_share->reverseZ;
        o->historyReset  = g_share->historyReset;
        o->resetReason   = g_share->resetReason;
        o->viewportW     = g_share->viewportW;
        o->viewportH     = g_share->viewportH;
        o->nearClip      = g_share->nearClip;
        o->farClip       = g_share->farClip;
        o->fovDeg        = g_share->fovDeg;
        o->simTime       = g_share->simTime;

        // Derived here rather than published, because the plugin's frame and
        // the layer's frame are not the same thing - the plugin publishes from
        // a flight loop callback, the layer consumes at present, and the two
        // can differ. Measuring the interval between the values we actually
        // consume keeps it honest.
        {
            static double prevSim = -1.0;
            double dt = (prevSim >= 0.0) ? (g_share->simTime - prevSim) : 0.0;
            prevSim = g_share->simTime;
            if (dt > 0.0 && dt < 1.0) o->frameTimeMs = (float)(dt * 1000.0);
            else                      o->frameTimeMs = 16.6f;
        }
        o->jitterX       = g_share->jitterX;
        o->jitterY       = g_share->jitterY;
        o->jitterIndex   = g_share->jitterIndex;
        o->jitterPhases  = g_share->jitterPhases;
        o->lodBias       = g_share->lodBias;
        o->camX          = g_share->camX;
        o->camY          = g_share->camY;
        o->camZ          = g_share->camZ;
        o->camDelta      = g_share->camDelta;
        o->objectCount   = g_share->objectCount;
        if (o->objectCount   < 0) o->objectCount   = 0;
        if (o->objectCount   > TAA_MAX_OBJECTS)  o->objectCount  = TAA_MAX_OBJECTS;
        memcpy(o->objects,  g_share->objects,  sizeof(TaaMovingObject) * o->objectCount);

        MemoryBarrier();
        if (g_share->frame == before) {
            o->frame = before;
            o->valid = true;
            return true;
        }
    }
    // Four consecutive tears cannot happen at one write per frame; treat as no
    // data rather than using a half-torn matrix.
    return false;
}

// ------------------------------------------------------------- dispatch
//
// Layers key their tables on the loader dispatch pointer, which is the first
// word of any dispatchable handle - not on the handle value itself, because the
// loader hands different layers different handle values for the same object.
static void *dispatchKey(void *handle) { return *(void**)handle; }

struct InstanceData {
    PFN_vkGetInstanceProcAddr gipa;
    PFN_vkDestroyInstance     destroyInstance;
    PFN_vkGetPhysicalDeviceFormatProperties getFormatProps;
    // The handle itself, which nothing needed until NGX. NVSDK_NGX_VULKAN_Init
    // takes the instance, and the map is keyed by DISPATCH POINTER rather than
    // by handle - so without this there is no way back to the VkInstance.
    VkInstance instance;
};

struct DeviceData {
    VkDevice         device;
    VkPhysicalDevice phys;
    PFN_vkGetDeviceProcAddr gdpa;
    PFN_vkDestroyDevice     destroyDevice;
    PFN_vkCreateImage       createImage;
    PFN_vkDestroyImage      destroyImage;
    PFN_vkQueuePresentKHR   queuePresent;
    PFN_vkCreateSampler     createSampler;
    PFN_vkCmdBeginRenderPass  cmdBeginRenderPass;
    PFN_vkCmdBeginRendering   cmdBeginRendering;
    PFN_vkCmdEndRendering     cmdEndRendering;
    PFN_vkCmdSetViewport      cmdSetViewport;

    // Everything below exists only for the velocity compute pass. All of it
    // operates on resources we create ourselves; the sole exception is the
    // depth image, which is read and restored to the layout it was found in.
    PFN_vkGetImageMemoryRequirements  getImageMemReq;
    PFN_vkGetBufferMemoryRequirements getBufferMemReq;
    PFN_vkAllocateMemory      allocateMemory;
    PFN_vkFreeMemory          freeMemory;
    PFN_vkBindImageMemory     bindImageMemory;
    PFN_vkBindBufferMemory    bindBufferMemory;
    PFN_vkMapMemory           mapMemory;
    PFN_vkCreateImageView     createImageView;
    PFN_vkDestroyImageView    destroyImageView;
    PFN_vkCreateBuffer        createBuffer;
    PFN_vkDestroyBuffer       destroyBuffer;
    PFN_vkCreateDescriptorSetLayout createDescriptorSetLayout;
    PFN_vkCreateDescriptorPool      createDescriptorPool;
    PFN_vkAllocateDescriptorSets    allocateDescriptorSets;
    PFN_vkUpdateDescriptorSets      updateDescriptorSets;
    PFN_vkCreateShaderModule    createShaderModule;
    PFN_vkCreatePipelineLayout  createPipelineLayout;
    PFN_vkCreateComputePipelines createComputePipelines;
    PFN_vkCreateCommandPool     createCommandPool;
    PFN_vkAllocateCommandBuffers allocateCommandBuffers;
    PFN_vkBeginCommandBuffer    beginCommandBuffer;
    PFN_vkEndCommandBuffer      endCommandBuffer;
    PFN_vkCmdBindPipeline       cmdBindPipeline;
    PFN_vkCmdBindDescriptorSets cmdBindDescriptorSets;
    PFN_vkCmdPushConstants      cmdPushConstants;
    PFN_vkCmdDispatch           cmdDispatch;
    PFN_vkCmdPushDescriptorSetKHR cmdPushDescriptorSet;  // X-Plane binds FSR's resources this way
    PFN_vkCmdPushDescriptorSet2 cmdPushDescriptorSet2;   // the Vulkan 1.4 form, which is the one it actually uses
    PFN_vkCmdPipelineBarrier    cmdPipelineBarrier;
    PFN_vkCmdCopyImageToBuffer  cmdCopyImageToBuffer;
    PFN_vkCmdFillBuffer         cmdFillBuffer;
    PFN_vkCmdCopyImage          cmdCopyImage;   // resolve result -> scene colour
    PFN_vkCmdClearColorImage    cmdClearColorImage;  // TAA_PROVE_OUTPUT probe
    PFN_vkCmdBlitImage          cmdBlitImage;        // FSR2 output -> swapchain
    PFN_vkCmdResolveImage       cmdResolveImage;     // X-Plane's MSAA resolve
    PFN_vkGetSwapchainImagesKHR getSwapchainImagesKHR;
    // GPU timing. Nothing else in the layer measures where the frame goes, and
    // CPU-side timing cannot: the work is recorded now and executed later.
    PFN_vkCreateQueryPool       createQueryPool;
    PFN_vkDestroyQueryPool      destroyQueryPool;
    PFN_vkCmdResetQueryPool     cmdResetQueryPool;
    PFN_vkCmdWriteTimestamp     cmdWriteTimestamp;
    PFN_vkGetQueryPoolResults   getQueryPoolResults;
    // Owned only when frame generation replaces the swapchain. Resolved
    // unconditionally so the pointers exist for the fall-through path too.
    PFN_vkCreateSwapchainKHR    createSwapchainKHR;
    PFN_vkDestroySwapchainKHR   destroySwapchainKHR;
    PFN_vkAcquireNextImageKHR   acquireNextImageKHR;

    // The resolve records into X-PLANE's command buffers, so unlike the
    // velocity pass there is no fence of ours to wait on before destroying its
    // resources. vkDeviceWaitIdle is the blunt but correct answer: teardown
    // happens on resize and shutdown, where a stall costs nothing, and freeing
    // an image the GPU is still reading is not a recoverable mistake.
    PFN_vkDeviceWaitIdle        deviceWaitIdle;
    PFN_vkCreateFence           createFence;
    PFN_vkResetFences           resetFences;
    PFN_vkWaitForFences         waitForFences;
    PFN_vkQueueSubmit           queueSubmit;
    PFN_vkGetDeviceQueue        getDeviceQueue;
    PFN_vkDestroyPipeline       destroyPipeline;
    PFN_vkDestroyPipelineLayout destroyPipelineLayout;
    PFN_vkDestroyShaderModule   destroyShaderModule;
    PFN_vkDestroyDescriptorPool destroyDescriptorPool;
    PFN_vkDestroyDescriptorSetLayout destroyDescriptorSetLayout;
    PFN_vkDestroySampler        destroySampler;
    PFN_vkDestroyCommandPool    destroyCommandPool;
    PFN_vkDestroyFence          destroyFence;
    PFN_vkUnmapMemory           unmapMemory;
};

static PFN_vkGetPhysicalDeviceMemoryProperties g_getPhysMemProps = nullptr;
// Needed by the Streamline path to clamp DLSS-G's extra queue requests against
// the number of queues each family actually has. Asking for more than exist is
// a validation error and, on some drivers, a device that fails to create.
static PFN_vkGetPhysicalDeviceQueueFamilyProperties g_getPhysQueueFamProps = nullptr;
// Needed for minUniformBufferOffsetAlignment, which sets the UBO ring stride.
static PFN_vkGetPhysicalDeviceProperties       g_getPhysProps    = nullptr;
static bool g_availReported = false;

// ---------------------------------------------------------------- VRAM survey
//
// X-Plane's texture pager collapsed to 1/8 resolution on an 8 GB card, its log
// reporting "2.45 gb out of 2.44 gb available" and the available figure FALLING
// as it freed textures. A budget that moves away from you as you release memory
// is not a budget you can satisfy, so the pager kept cutting.
//
// Two very different causes, needing opposite fixes:
//
//   * X-Plane is being TOLD it has little memory - by heap sizes, or by
//     VK_EXT_memory_budget's heapBudget, which reports what the driver is
//     willing to give right now rather than what the card has. Then the fix is
//     at the reporting layer and no texture needs touching.
//
//   * The card really is full - other processes, or our own allocations, which
//     currently come to about 112 MB before FSR2 asks for more. Then the fix is
//     to use less, and compression is the lever.
//
// Guessing between them wastes the same way three wrong theories about the
// resolve did. So: record what is asked, and what is answered.
static PFN_vkGetPhysicalDeviceMemoryProperties2 g_nextMemProps2 = nullptr;
static PFN_vkEnumerateDeviceExtensionProperties g_nextEnumDeviceExt = nullptr;
// Needed by ffx_vk_shim.cpp, which answers the Vulkan entry points FidelityFX
// calls by name so they go DOWN the chain instead of re-entering the loader.
static PFN_vkGetPhysicalDeviceProperties2 g_getPhysProps2 = nullptr;
static PFN_vkGetPhysicalDeviceFeatures2   g_getPhysFeat2  = nullptr;
static PFN_vkGetPhysicalDeviceFeatures    g_getPhysFeat   = nullptr;

extern "C" PFN_vkEnumerateDeviceExtensionProperties mvNextEnumDeviceExtensionProperties()
{ return g_nextEnumDeviceExt; }
extern "C" PFN_vkGetPhysicalDeviceProperties2 mvNextGetPhysicalDeviceProperties2()
{ return g_getPhysProps2; }
extern "C" PFN_vkGetPhysicalDeviceFeatures2 mvNextGetPhysicalDeviceFeatures2()
{ return g_getPhysFeat2; }
extern "C" PFN_vkGetPhysicalDeviceProperties mvNextGetPhysicalDeviceProperties()
{ return g_getPhysProps; }
extern "C" PFN_vkGetPhysicalDeviceMemoryProperties mvNextGetPhysicalDeviceMemoryProperties()
{ return g_getPhysMemProps; }
extern "C" PFN_vkGetPhysicalDeviceFeatures mvNextGetPhysicalDeviceFeatures()
{ return g_getPhysFeat; }

// Device functions, resolved through the NEXT layer for the device in question.
// Defined after g_devices below; declared here so the shim can bind to it.
extern "C" PFN_vkVoidFunction mvNextDeviceProcAddr(VkDevice device, const char *name);
static uint64_t g_memQueryCount = 0;
static float    g_vramBudgetScale = 1.0f;   // >1 inflates the reported budget
static PFN_vkAllocateMemory g_nextAllocateMemory = nullptr;
// Motion vector injection accounting.
//
// COVERAGE is the number that matters. A vertex shader we cannot patch draws
// geometry carrying no velocity, and that shows up as smearing confined to one
// part of the scene with nothing in the logs to explain it - the same class of
// silent gap that cost a day when NaN depth was quietly routing scenery into
// the near-field branch. Counting the failures is the difference between
// "injection works" and "injection works on the shaders I happened to look at".
static bool     g_spirvInject  = false;
// LIVE is a separate switch from INJECT on purpose. Injection alone patches and
// counts; LIVE hands the patched modules to the driver and adds the attachment.
// Keeping them apart is what allowed every earlier step to be measured against
// a running sim without any chance of breaking it.
static bool     g_spirvLive    = false;
// Which fragment modules were patched. A pipeline whose fragment shader was NOT
// patched still gains the extra attachment - it has to, or its format list
// disagrees with the pass - but with colorWriteMask 0, so it writes nothing
// there and leaves the cleared zero in place.
static std::set<VkShaderModule> g_patchedFrag;
// Carries the patched words from the injection block to the substitution below,
// which sit in the same function but either side of the census and dump code.
static std::vector<uint32_t> g_patchedCode;
// Reset at present. The FIRST scene pass of a frame clears the velocity image;
// later ones load it, so a scene split across several passes accumulates
// instead of each pass wiping what the previous one wrote.
// ATOMIC, because X-Plane records command buffers on several threads at once.
//
// This was a plain bool doing read-then-write:
//
//     bool first = !g_mvClearedThisFrame;
//     g_mvClearedThisFrame = true;
//
// Two threads reaching that together both see first == true and both emit
// LOAD_OP_CLEAR, so one scene pass's velocities are wiped by the other's clear.
// It depends on thread timing, so it comes and goes - a whole-scene flicker
// rather than a steady artefact. The same hazard is documented a few hundred
// lines above for the in-scene-pass flag; this one was missed.
//
// A single atomic exchange means exactly one pass clears. It does NOT fix
// recording order versus execution order - the pass that records first is not
// necessarily the one that runs first - but that is a separate and much rarer
// problem than two clears in one frame.
static std::atomic<bool> g_mvClearedThisFrame(false);

// Set once the present-time clear has run at least once. Until then the target
// has never been zeroed and the recording-time clear is still needed for the
// very first frames; after it, every scene pass LOADs and nothing has to decide.
static std::atomic<bool> g_mvClearedAtPresent(false);

// Declared here, ahead of vkCreateShaderModule, because that is where the
// original words are stored - and ahead of vkCreateGraphicsPipelines, which is
// where they are finally patched.
// Descriptor-set state for crash destruction.
//
// g_layoutOurSet records, per pipeline layout, WHICH set index ours ended up
// at - it is not a constant, because we append at the layout's own
// setLayoutCount and different layouts declare different counts. The shader
// patch and the bind both need this number, and getting it wrong means reading
// a buffer that belongs to X-Plane.
// What this GPU can actually run, as QUERIED facts rather than a device-name
// guess. Filled once at device creation, where the extension list is already
// in hand, and read by the capability report the plugin's UI depends on.
static upscaler::DeviceCaps g_upscalerCaps;

static uint32_t g_maxBoundSets = 4;

// ---- THE SINGLE CRASH-DESTRUCTION GATE.
//
// One function rather than a live::onoff() at each site, because the gate has
// to cover THREE things that are easy to gate separately and wrong to: whether
// the descriptor resources exist, whether every pipeline layout is extended to
// carry them, and whether the set is bound per draw.
//
// Gating only the bind - which is what this did first - still appended a
// descriptor set to every layout X-Plane creates. That is a permanent change
// to the layout of every pipeline in the sim in exchange for a feature that is
// switched off, and it is invisible in a trace because nothing reports it.
// With this, crash.enable=0 means destructgpu::state().ready stays false and
// all three fall away together.
// ---- WHETHER THE PER-VERTEX OCCUPANCY CODE IS EMITTED AT ALL.
//
// Separate from crash.enable, and off by default, because the two costs are
// nothing alike.
//
// crash.enable buys the descriptor resources and the per-pipeline-bind, which
// is a bind per pipeline change - about a thousand a frame, and cheap.
//
// This buys a matrix multiply, three divides, six comparisons and a store ON
// EVERY VERTEX IN THE SIM, on every frame, forever. Measured at 4K that is
// 38 fps down to 27.5 - and it is spent on a feature that matters for exactly
// two frames of a discovery run, or during a crash that has not happened yet.
//
// Read once at first use, like crash.enable, because pipelines are patched at
// startup and a value that changed later could not reach them anyway.
// ---- THE STATIC DISPLACEMENT, FOR TASK 10'S ACCEPTANCE TEST.
//
// A constant offset in AIRCRAFT-LOCAL metres, applied to every vertex the
// classification accepts. The plan's test is "set crash.test_offset=5 and
// confirm the whole airframe moves five metres while the world stays put",
// which is the cheapest possible proof that the displacement path reaches real
// geometry with the right transform.
//
// Read fresh every frame rather than latched, unlike crash.enable and
// crash.occupancy. Those two decide what gets COMPILED INTO a shader and so
// cannot change without new modules; this is data in a buffer, and being able
// to drag it while watching the aeroplane is the entire point of the test.
//
// Along +y, so the airframe rises out of the scenery rather than sliding into
// it - a vertical move is unambiguous from any camera angle, and a horizontal
// one at an airport is easy to mistake for the world moving instead.
static const float *crashTestOffset()
{
    static float off[3] = { 0.0f, 0.0f, 0.0f };
    off[0] = 0.0f;
    off[1] = live::f("crash.test_offset", "TAA_CRASH_TEST_OFFSET", 0.0f);
    off[2] = 0.0f;
    return off;
}

// ---- A LIVE NUDGE ON THE GRID ORIGIN.
//
// The grid is built by the plugin in ".acf minus the reference point". The
// shader classifies with crashAircraftInv, which comes from Mc. Nothing has
// ever confirmed those two frames share an origin, and the evidence says they
// do not: the GPU discovery measured the airframe at x -27.43 to +33.16 - a
// width of 60.6 m, which is the aeroplane, centred on +2.87 rather than on 0.
//
// That offset pushes one wing toward the +x face of the box, and a vertex
// outside the box keeps its own position. It is precisely why the right
// engines, the gear and the tail lag while the middle of the aeroplane moves.
//
// Tunable live so the offset can be MEASURED by dialling it until the
// aeroplane moves as one piece, rather than derived from another theory about
// which frame X-Plane draws in. Once it is known, it belongs in the plugin
// next to referencePointOffset().
static void crashGridNudge(float out[3])
{
    out[0] = live::f("crash.offset_x", "TAA_CRASH_OFF_X", 0.0f);
    out[1] = live::f("crash.offset_y", "TAA_CRASH_OFF_Y", 0.0f);
    out[2] = live::f("crash.offset_z", "TAA_CRASH_OFF_Z", 0.0f);
}

static bool crashOccupancy()
{
    static int on = -1;
    if (on < 0) {
        live::loadNow();
        on = live::onoff("crash.occupancy", "TAA_CRASH_OCCUPANCY", false) ? 1 : 0;
        trace("DESTRUCT: crash.occupancy=%d - the per-vertex classification is "
              "%s. It costs a matrix multiply and a store on every vertex in "
              "the sim, so it is off unless a discovery run needs it.",
              on, on ? "COMPILED INTO EVERY VERTEX SHADER" : "not emitted");
    }
    return on != 0;
}

static bool crashEnabled()
{
    static int on = -1;
    if (on < 0) {
        // FORCE A READ FIRST.
        //
        // The first caller of this is vkCreateDevice, and live::poll() only
        // runs on the frame path - so the key table was empty here, the lookup
        // missed, the built-in default of false won, and that false was then
        // cached for the whole process. crash.enable=1 in the control file
        // could not switch the system on no matter what it said.
        //
        // Caching is still right: this is read per pipeline bind. What was
        // wrong was caching a value taken before the file had been read.
        live::loadNow();
        on = live::onoff("crash.enable", "TAA_CRASH", false) ? 1 : 0;
        trace("DESTRUCT: crash.enable=%d (from %s)", on, live::path());
    }
    return on != 0;
}
static std::map<VkPipelineLayout, uint32_t> g_layoutOurSet;

static std::map<VkShaderModule, std::vector<uint32_t> > g_moduleCode;

// ---- DOES THIS VERTEX SHADER PULL ITS VERTICES FROM MEMORY?
//
// "No vertex attributes" collapses two opposite things into one test, and
// both halves of this project's terrain work have now been wrong because of
// it. X-Plane 12 pulls TERRAIN vertices from storage buffers, so terrain
// declares no attributes; a post-process full-screen triangle builds its
// position from gl_VertexIndex and also declares none.
//
// Depth was tried as the separator and is not sufficient. X-Plane sets depth
// state dynamically, so create-info depth reads false for terrain; widening
// the rule to "declares depth dynamic state AND draws into a depth pass"
// rescued terrain and swept in the post-process quads with it. Measured, at
// viz=2 during a pan: the whole frame saturates - 100% of pixels nonzero,
// ~100% at the top of the magnitude ramp - because a full-screen triangle
// covers every pixel and reprojecting a screen corner is meaningless. With
// the strict rule the same scene measures 0.0% saturated. That is the
// "translucent boxes" failure this file already warned the loose rule would
// cause, and it is what a full-screen stamp looks like from the outside.
//
// The honest separator is the mechanism itself. Pulling vertices means
// READING A BUFFER: an SSBO in StorageBuffer storage, or the pre-1.3 spelling,
// a Uniform-storage struct decorated BufferBlock. A full-screen triangle reads
// no buffer at all to place itself - that is the whole point of the trick.
//
// Note this is deliberately a property of the SHADER, not of the pass, the
// draw or the pipeline state. Passes and dynamic state are X-Plane's to change
// between versions; "does the vertex stage read a storage buffer" is a fact
// about the module and survives a recompile.
enum { kSC_Uniform = 2, kSC_StorageBuffer = 12 };
enum { kDeco_BufferBlock = 3 };

// ---- WHAT IS ACTUALLY IN THE ZERO-ATTRIBUTE PILE?
//
// Two guesses have now been made about this pile and both were wrong. "No
// vertex attributes means full-screen quad" blacked the ground; "depth state
// separates them" swept in 1379 post-process quads that stamp the frame; and
// "the terrain reads an SSBO" rescued exactly ONE pipeline, which is not a
// terrain system. Guess three would be worth less than one measurement.
//
// So this prints a FINGERPRINT of every distinct zero-attribute vertex module:
// how it is built and where it reads from. The terrain must place its vertices
// from somewhere, and whatever that source is will show up here next to the
// post-process quads that read nothing to place themselves. The discriminator
// then gets chosen from the data instead of from an assumption about how
// X-Plane is written.
struct VsFingerprint {
    uint32_t words;
    bool vertexIndex, instanceIndex;
    bool ssbo, uniform, bufferBlock, runtimeArray;
    bool imageFetchOrRead, sampledImage, texelBuffer;
    uint32_t nUniformLoads;
};

static VsFingerprint spirvVsFingerprint(const std::vector<uint32_t> &w)
{
    VsFingerprint f;
    memset(&f, 0, sizeof(f));
    f.words = (uint32_t)w.size();
    if (w.size() < 5 || w[0] != 0x07230203u) return f;

    std::set<uint32_t> bufferBlockTypes, ptrsToWatch;
    std::map<uint32_t, uint32_t> ptrPointee, ptrClass;
    for (size_t k = 5; k < w.size(); ) {
        const uint32_t op = w[k] & 0xFFFFu, len = w[k] >> 16;
        if (len == 0 || k + len > w.size()) break;
        switch (op) {
        case 71: /* OpDecorate */
            if (len >= 3 && w[k+2] == 3 /* BufferBlock */) bufferBlockTypes.insert(w[k+1]);
            if (len >= 4 && w[k+2] == 11 /* BuiltIn */) {
                if (w[k+3] == 42) f.vertexIndex   = true;
                if (w[k+3] == 43) f.instanceIndex = true;
            }
            break;
        case 32: /* OpTypePointer */
            if (len >= 4) { ptrPointee[w[k+1]] = w[k+3]; ptrClass[w[k+1]] = w[k+2]; }
            break;
        case 29: /* OpTypeRuntimeArray */ f.runtimeArray = true; break;
        case 25: /* OpTypeImage */
            // Dim 5 = Buffer: a texel buffer, the other way to pull vertices.
            if (len >= 4 && w[k+3] == 5) f.texelBuffer = true;
            break;
        case 27: /* OpTypeSampledImage */ f.sampledImage = true; break;
        case 59: /* OpVariable */
            if (len >= 4) {
                const uint32_t sc = w[k+3];
                if (sc == 12) f.ssbo = true;
                if (sc == 2) {
                    f.uniform = true;
                    std::map<uint32_t,uint32_t>::iterator it = ptrPointee.find(w[k+1]);
                    if (it != ptrPointee.end() && bufferBlockTypes.count(it->second))
                        f.bufferBlock = true;
                }
                if (sc == 2 || sc == 12) ptrsToWatch.insert(w[k+2]);
            }
            break;
        case 95: /* OpImageFetch  */
        case 98: /* OpImageRead   */ f.imageFetchOrRead = true; break;
        case 61: /* OpLoad */
            if (len >= 4 && ptrsToWatch.count(w[k+3])) ++f.nUniformLoads;
            break;
        default: break;
        }
        k += len;
    }
    return f;
}

static bool spirvPullsVertices(const std::vector<uint32_t> &w)
{
    if (w.size() < 5 || w[0] != 0x07230203u) return false;
    std::set<uint32_t> bufferBlockTypes;
    // Pass 1: type ids decorated BufferBlock (the Uniform-storage SSBO spelling).
    for (size_t k = 5; k < w.size(); ) {
        const uint32_t op = w[k] & 0xFFFFu, len = w[k] >> 16;
        if (len == 0 || k + len > w.size()) break;
        if (op == 71 /* OpDecorate */ && len >= 3 && w[k+2] == kDeco_BufferBlock)
            bufferBlockTypes.insert(w[k+1]);
        k += len;
    }
    // Pass 2: any OpVariable in StorageBuffer storage, or in Uniform storage
    // whose pointee type was decorated BufferBlock.
    std::map<uint32_t, uint32_t> ptrPointee;   // pointer type id -> pointee id
    for (size_t k = 5; k < w.size(); ) {
        const uint32_t op = w[k] & 0xFFFFu, len = w[k] >> 16;
        if (len == 0 || k + len > w.size()) break;
        if (op == 32 /* OpTypePointer */ && len >= 4)
            ptrPointee[w[k+1]] = w[k+3];
        else if (op == 59 /* OpVariable */ && len >= 4) {
            const uint32_t sc = w[k+3];
            if (sc == kSC_StorageBuffer) return true;
            if (sc == kSC_Uniform) {
                std::map<uint32_t, uint32_t>::iterator it = ptrPointee.find(w[k+1]);
                if (it != ptrPointee.end() && bufferBlockTypes.count(it->second))
                    return true;
            }
        }
        k += len;
    }
    return false;
}
// The second half of the key packs the attachment index and the quantised
// Keyed on the blend mode too: an opaque and an alpha-blended pipeline
// sharing a module need DIFFERENT variants, because only the blended one
// gets a real coverage gate. Without this the first one to arrive would
// hand its variant to the other.
typedef std::pair<VkShaderModule, uint32_t> FragModKey;
typedef std::pair<FragModKey, bool> FragKey;
static std::map<FragKey, VkShaderModule> g_fragVariant;
static uint64_t g_fragVariants = 0, g_fragPatchFail = 0;
static bool g_patchedWasFrag = false;
static uint64_t g_injOk        = 0;
static uint64_t g_injFailed    = 0;
static uint64_t g_injNotVertex = 0;
static uint64_t g_injFrag      = 0;   // of g_injOk, how many were fragment

// ARMED FROM WHICHEVER ENTRY POINT COMES FIRST.
//
// This is the third appearance of the same ordering mistake, so it is a
// function rather than a line copied into one place. Features that act on
// FRAMES can read their environment at the first vkQueuePresentKHR; features
// that act during LOAD cannot. Pipeline layouts and shader modules are both
// created while the scenery loads, and neither has any fixed order with respect
// to the other - so arming in only one of them leaves the other reading a flag
// that is still false. The failure is silent: no patches, no errors, and
// counters that look exactly like a clean run.
static void armSpirvInject()
{
    static bool armed = false;
    if (armed) return;
    armed = true;
    // ---- ON BY DEFAULT. THIS IS THE PROJECT.
    //
    // Both of these were armed only when an environment variable was set, which
    // the old development launcher did and nothing else does. Measured: the
    // creation gate reported spirvLive=0 with every other condition satisfied -
    // depth found, scene image valid, stable for 1768 frames against a
    // threshold of 120 - so no velocity target was ever built and the project
    // produced nothing at all when installed.
    //
    // This is the same trap as the texture pager holds, in the one place where
    // it costs everything: behaviour that exists only under the development
    // launcher looks like working code right up until someone installs it.
    //
    // Set either variable to 0 to switch them off; anything else, or absence,
    // leaves them on.
    auto envOn = [](const char *name) {
        const char *v = getenv(name);
        return !v || atoi(v) != 0;      // absent means ON
    };
    // Gated on the velocity master switch: with TAA_VELOCITY off the probe
    // parsed EVERY shader module anyway - thousands per complex aircraft -
    // for statistics nobody would read. Dormant now means dormant: the
    // Felis-load crash lived somewhere in that pointless work.
    const char *velEnv = getenv("TAA_VELOCITY");
    const bool velArmed = velEnv && velEnv[0] == '1' && velEnv[1] == '\0';
    g_spirvInject = velArmed && envOn("TAA_SPIRV_INJECT");
    g_spirvLive   = envOn("TAA_SPIRV_LIVE") && g_spirvInject;
    if (g_spirvInject)
        trace("SPIRV INJECT: armed - %s",
              g_spirvLive
                ? "LIVE: patched shaders go to the driver and the velocity "
                  "attachment is bound"
                : "DRY RUN: patched and counted, originals still used");
}

// Frames presented. Incremented in vkQueuePresentKHR and read wherever "how
// long has this been true" is the question - currently the FSR2 idle timeout.
static uint64_t g_frameCount = 0;
// Which frame FSR2 last wrote its output on. The present blit needs to know
// the image it is about to copy was produced now, not several seconds ago.
static uint64_t g_fsr2LastDispatchFrame = 0;
// ---- WHICH COMMAND BUFFER WROTE outImg.
//
// FSR2's own copy-back reads the FULL 3840x2160 of outImg and produces a
// perfect picture. The delivery reads the same image, same extent, and gets
// garbage - through a blit AND through our own compute shader, so neither
// the conversion nor the blit engine is responsible. The remaining variable
// is which command buffer each read lands in: submission order only orders
// work within one queue's submit sequence, and if these are separate command
// buffers the sim is free to submit them in an order that puts our read
// before FSR2's write. No barrier spans that, which is exactly why every
// barrier experiment changed nothing.
static VkCommandBuffer g_fsr2LastDispatchCb = VK_NULL_HANDLE;
// The delivery command buffer, and a per-frame record of which of the two
// reached the queue first.
static VkCommandBuffer g_deliveryCb = VK_NULL_HANDLE;
static uint32_t g_submitSeq = 0;
static uint32_t g_seqOfDispatchCb = 0xFFFFFFFF;
static uint32_t g_seqOfDeliveryCb = 0xFFFFFFFF;
// Every queue family the sim creates a queue on. Needed because an image
// written on one family and read on another must either be handed over
// explicitly or declared shared across them; anything else leaves its
// contents undefined for the reader, with no barrier able to help.
static std::vector<uint32_t> g_deviceFamilies;

// The driver's own figure for device-local heap usage, from the last
// VK_EXT_memory_budget query. The ledger compares itself against this.
static uint64_t g_lastHeapUsage = 0;
static uint64_t g_lastHeapBudget = 0;

// ---- DEGRADE ONLY WHEN IT IS ACTUALLY NEEDED.
//
// The pager used to reduce every texture over its threshold, always, whether
// the card was full or empty. That is a policy from when VRAM was the binding
// constraint - it no longer is, and the cost was visible: cockpit panel
// surfaces read from 0.7 m away were arriving mushy while 2.5 GB sat unused.
//
// So paging is now conditional on headroom. Above the reserve there is no
// reason to touch anything; below it, the pager does exactly what it did
// before. This is the "exclusion zone" idea generalised - rather than trying to
// name which textures matter, which vkCreateImage gives us no way to know, it
// excludes ALL of them until the memory is genuinely wanted.
//
// Hysteresis matters here. A pager that switches on and off around a single
// threshold would drop mips on some textures and not others depending on the
// instant they happened to be created, which is a worse artefact than either
// policy - the same surface would be sharp or soft by luck of load order. So
// once it starts paging it keeps paging until there is comfortably more room
// than the point it started at.
static uint64_t g_pagerHeadroomMB = 1024;   // start paging below this
static bool     g_pagerEngaged    = false;

static bool pagerShouldEngage()
{
    if (!g_pagerHeadroomMB) return true;          // 0 = always page, old behaviour
    if (!g_lastHeapBudget || !g_lastHeapUsage) return false;  // no data yet: leave textures alone
    uint64_t freeMB = (g_lastHeapBudget > g_lastHeapUsage)
                    ? (g_lastHeapBudget - g_lastHeapUsage) / 1048576ull : 0;
    // Engage below the threshold, disengage only once there is 50% more room
    // than that - see the note on hysteresis.
    if (!g_pagerEngaged && freeMB < g_pagerHeadroomMB) {
        g_pagerEngaged = true;
        trace("PAGER: ENGAGING - %llu MB free is below the %llu MB reserve. "
              "Textures above the threshold will now lose mip levels.",
              (unsigned long long)freeMB, (unsigned long long)g_pagerHeadroomMB);
    } else if (g_pagerEngaged && freeMB > g_pagerHeadroomMB + g_pagerHeadroomMB / 2) {
        g_pagerEngaged = false;
        trace("PAGER: DISENGAGING - %llu MB free. New textures are created at "
              "full resolution again.", (unsigned long long)freeMB);
    }
    return g_pagerEngaged;
}

static uint64_t g_allocCount = 0, g_allocBytes = 0;
static uint64_t g_allocFailed = 0, g_allocRescued = 0;
static bool     g_memoryPriority = false;   // VK_EXT_memory_priority usable
static bool     g_pageableMemory = false;   // pageable_device_local feature on
static bool     g_overcommit = false;

// ======================================================== THE VRAM LEDGER
//
// Every image and every buffer, by ACTUAL memory requirement, split into the
// categories the pager treats differently - and carrying the reason each one is
// or is not pageable.
//
// WHY THIS EXISTS, given there is already a texture census.
//
// The census answers a different question and answers it deliberately wrong for
// this purpose. It buckets textures by the size the APPLICATION ASKED FOR, not
// the size we gave it, because its job is to choose the drop threshold and the
// pager's own effect would hide the thing being measured. So it reads 4.20 GB
// while the pager has already taken 3.04 GB of that back out - the real
// resident figure is about 1.16 GB, and the two numbers are three-fold apart
// for a completely legitimate reason.
//
// It also excludes, by construction:
//
//   - colour attachments   (filtered at the census, and skipped by the pager)
//   - depth buffers        (same)
//   - storage images       (same)
//   - EVERY BUFFER         (never counted anywhere at all)
//
// Which is how a card reporting 6.28 GB in use could be "explained" by a 4.20
// GB texture figure that was neither resident nor complete. The gap was about
// five gigabytes and nothing in this layer could see any of it.
//
// So: actual requirements from vkGetImageMemoryRequirements rather than
// width x height x bpp, because that includes alignment, mip tails and whatever
// compression metadata the driver attaches - all of which is real memory that
// the arithmetic version silently omits.
enum VramCat {
    VRAM_TEX = 0,      // sampled, not an attachment - the pager's territory
    VRAM_RT,           // colour attachment
    VRAM_DEPTH,        // depth/stencil
    VRAM_STORAGE,      // written by shaders
    VRAM_IMG_OTHER,    // transient, 3D, arrays - anything else
    VRAM_BUF_GEOM,     // vertex + index
    VRAM_BUF_UNIFORM,  // uniform + storage buffers
    VRAM_BUF_STAGING,  // transfer only, host-visible staging
    VRAM_BUF_OTHER,
    VRAM_CAT_COUNT
};

static const char *vramCatName(int c)
{
    switch (c) {
        case VRAM_TEX:         return "textures (sampled)";
        case VRAM_RT:          return "render targets";
        case VRAM_DEPTH:       return "depth buffers";
        case VRAM_STORAGE:     return "storage images";
        case VRAM_IMG_OTHER:   return "other images";
        case VRAM_BUF_GEOM:    return "geometry buffers";
        case VRAM_BUF_UNIFORM: return "uniform/storage buffers";
        case VRAM_BUF_STAGING: return "staging buffers";
        default:               return "other buffers";
    }
}

// Why the pager leaves this category alone. Printed next to the size, because a
// list of things we are not paging is only useful with the reason attached -
// otherwise every line reads as an oversight and the ones that genuinely are
// get lost among the ones that cannot be helped.
static const char *vramCatWhy(int c)
{
    switch (c) {
    case VRAM_TEX:
        return "PAGED - mip levels dropped at creation";
    case VRAM_RT:
        return "not pageable: sized by the render resolution, not by content. "
               "Lowering these means lowering the render scale";
    case VRAM_DEPTH:
        return "not pageable: one mip, exact precision required";
    case VRAM_STORAGE:
        return "not pageable: written by shaders, so a smaller image would be "
               "written out of bounds";
    case VRAM_IMG_OTHER:
        return "not pageable: cube maps, arrays and transient targets - the "
               "pager takes only single-layer 2D";
    case VRAM_BUF_GEOM:
        return "not pageable by this mechanism: mesh data has no mip chain. "
               "X-Plane's own object LOD is what controls this";
    case VRAM_BUF_UNIFORM:
        return "not pageable: per-draw constants, tiny individually";
    case VRAM_BUF_STAGING:
        return "not pageable: upload scratch. Should be host-visible, so it "
               "costs system RAM rather than VRAM - worth checking if large";
    default:
        return "not pageable";
    }
}

struct VramCatStat { uint64_t count = 0; uint64_t bytes = 0; uint64_t peak = 0; };
static VramCatStat g_vram[VRAM_CAT_COUNT];

// ---------------------------------------------- geometry, broken down by size
//
// 16638 buffers holding 2423 MB is an average of 149 KB, and an average is the
// wrong statistic for deciding what to do about it. Two very different worlds
// produce that number:
//
//   - a few hundred large vertex buffers, in which case the memory is real mesh
//     data and only X-Plane's object density controls it;
//   - tens of thousands of small ones, in which case a large part of the cost
//     is per-allocation ALIGNMENT PADDING rather than vertices, and that is
//     recoverable without touching what is drawn.
//
// The second is worth checking because bufferImageGranularity on this class of
// hardware is commonly 1 KB and some drivers round small buffers up hard. A
// 3 KB buffer occupying 64 KB is a twentyfold overhead, invisible in every
// figure anyone normally looks at.
//
// So: a histogram by power-of-two REQUESTED size, carrying both what was asked
// for and what the driver actually reserved. The difference between those two
// columns IS the padding, stated rather than inferred.
static uint64_t g_geomCount[24];      // buffers in each bucket
static uint64_t g_geomAsked[24];      // ci->size, summed
static uint64_t g_geomGot[24];        // req.size, summed

static int geomBucketOf(uint64_t bytes)
{
    int b = 0;
    while (b < 23 && ((uint64_t)1 << (b + 10)) < bytes) ++b;   // 1 KB .. 8 GB
    return b;
}

// What each live handle contributed, so destroy subtracts exactly what create
// added. The texture census learned this the hard way: it only ever added, and
// after twelve hours read 8.07 GB on a 7.77 GB card while the driver reported
// 4.29 GB - a number that cannot be true, quoted as a measurement.
// ---- THE LEDGER HAS A COST, AND IT IS ON THE LOADING PATH.
//
// Every vkCreateImage and vkCreateBuffer takes the layer's GLOBAL mutex to
// record an entry, and calls vkGetImageMemoryRequirements first. X-Plane
// created 26,331 geometry buffers and thousands of images in one session, from
// several loader threads at once - so a single global lock on that path
// serialises texture loading across all of them.
//
// That matters because the profiler shows TEX_obj::do_load at 40 ms per call
// on a 24-core CPU with an NVMe SSD, where neither compute nor I/O explains it.
// Microprofile timers are WALL CLOCK: they measure waiting just as happily as
// working, and a contended mutex looks exactly like slow code.
//
// So the accounting is switchable. TAA_LEDGER=0 removes every map insert and
// requirements query from resource creation, which turns "is our instrumen-
// tation the stall" from a suspicion into a measurement. It is on by default
// because the numbers it produces are the only reason we understand the memory
// picture at all - but it should never have been assumed free.
static bool g_ledgerOn = true;

struct VramEntry { int cat; uint64_t bytes;
                   uint32_t w, h, fmt, mips; };   // dims: images only (SS66)
static std::map<VkImage,  VramEntry> g_vramImg;
static std::map<VkBuffer, VramEntry> g_vramBuf;

// Ledger total, maintained incrementally and fed to the VRAM system so its
// budget shaper can see the app's own trend (the monotone-under-free clamp
// needs to know whether the app is currently releasing memory).
static uint64_t g_vramTotalBytes = 0;

static void vramAdd(int cat, uint64_t bytes)
{
    g_vram[cat].count++;
    g_vram[cat].bytes += bytes;
    if (g_vram[cat].bytes > g_vram[cat].peak) g_vram[cat].peak = g_vram[cat].bytes;
    g_vramTotalBytes += bytes;
    vram::ledgerTotal(g_vramTotalBytes);
}

static void vramRemove(int cat, uint64_t bytes)
{
    if (g_vram[cat].count) g_vram[cat].count--;
    g_vram[cat].bytes = (g_vram[cat].bytes > bytes) ? (g_vram[cat].bytes - bytes) : 0;
    g_vramTotalBytes = (g_vramTotalBytes > bytes) ? (g_vramTotalBytes - bytes) : 0;
    vram::ledgerTotal(g_vramTotalBytes);
}

// The bind hooks' category lookups are defined with the VRAM system hooks,
// below the g_lock they need.

static int vramCatOfImage(const VkImageCreateInfo *ci, bool depthFmt)
{
    // Order matters. An image can carry several usage bits and the category has
    // to be the one that decides whether the pager may touch it - so the
    // disqualifying bits are tested first, and SAMPLED last.
    if (ci->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) return VRAM_DEPTH;
    if (depthFmt)                                                return VRAM_DEPTH;
    if (ci->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)         return VRAM_RT;
    if (ci->usage & VK_IMAGE_USAGE_STORAGE_BIT)                  return VRAM_STORAGE;
    if ((ci->usage & VK_IMAGE_USAGE_SAMPLED_BIT) &&
        ci->imageType == VK_IMAGE_TYPE_2D && ci->arrayLayers == 1)
        return VRAM_TEX;
    return VRAM_IMG_OTHER;
}

static int vramCatOfBuffer(VkBufferUsageFlags u)
{
    if (u & (VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT))
        return VRAM_BUF_GEOM;
    if (u & (VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT))
        return VRAM_BUF_UNIFORM;
    // Transfer-only: nothing reads it as a resource, so it is scratch.
    if ((u & (VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) &&
        !(u & ~(VkBufferUsageFlags)(VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT)))
        return VRAM_BUF_STAGING;
    return VRAM_BUF_OTHER;
}

// Texture census, by format. Answers whether compression is even the
// opportunity: if X-Plane's textures are already BC-compressed there is nothing
// to win, and the whole idea is dead before a line of transcoder is written.
struct FmtStat { uint64_t count = 0; uint64_t bytes = 0; };
static std::map<int, FmtStat> g_texCensus;
static uint64_t g_texBytesTotal = 0;

// THE CENSUS MUST SUBTRACT ON DESTROY, or it is not a census.
//
// It only ever added, so the printed figure was total-ever-created, not what is
// resident - and it kept climbing for as long as the sim ran. After twelve
// hours it read 8.07 GB on a 7.77 GB card while the driver reported 4.29 GB in
// use, which is the contradiction that gave it away. A number that cannot be
// true was being quoted as a measurement, including by me.
//
// So each censused image records what it contributed, and destroy takes exactly
// that back out.
struct CensusEntry { int format; uint64_t bytes; int sizeBucket; };
static std::map<VkImage, CensusEntry> g_texCensusOf;

// Resident texture memory by LONGEST SIDE, bucketed by power of two.
//
// The format census answered "is there anything to compress" - no, it is
// already BC. This answers the question the custom pager actually turns on:
// where does the memory live by SIZE, and therefore what does a drop threshold
// have to be set to before it takes a meaningful bite.
//
// Without it the threshold is a guess, and the first guess - 2048 - caught
// about twenty images and saved 150 MB against a 2.80 GB ceiling. That is not
// a tuning error so much as a measurement that was never taken.
static uint64_t g_sizeCount[17];
static uint64_t g_sizeBytes[17];

static int sizeBucketOf(uint32_t w, uint32_t h)
{
    uint32_t big = (w > h) ? w : h;
    int b = 0;
    while ((1u << b) < big && b < 16) ++b;
    return b;
}

// Bytes per pixel-block, enough for a size estimate rather than an exact one.
static double formatBytesPerPixel(VkFormat f)
{
    switch (f) {
        case VK_FORMAT_R8_UNORM:                     return 1.0;
        case VK_FORMAT_R8G8_UNORM:                   return 2.0;
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:                return 4.0;
        case VK_FORMAT_R16G16B16A16_SFLOAT:          return 8.0;
        case VK_FORMAT_R32G32B32A32_SFLOAT:          return 16.0;
        // BC formats: 4x4 blocks. BC1 is 8 bytes per block, the rest 16.
        case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
        case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
        case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
        case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
        case VK_FORMAT_BC4_UNORM_BLOCK:              return 0.5;
        case VK_FORMAT_BC2_UNORM_BLOCK:
        case VK_FORMAT_BC3_UNORM_BLOCK:
        case VK_FORMAT_BC3_SRGB_BLOCK:
        case VK_FORMAT_BC5_UNORM_BLOCK:
        case VK_FORMAT_BC6H_UFLOAT_BLOCK:
        case VK_FORMAT_BC7_UNORM_BLOCK:
        case VK_FORMAT_BC7_SRGB_BLOCK:               return 1.0;
        default:                                     return 4.0;
    }
}

static std::mutex                      g_lock;
static std::map<void*, InstanceData>   g_instances;
static VkInstance                      g_firstInstance = VK_NULL_HANDLE;
static bool                            g_instanceGettersBound = false;
static bool                            g_queueFamGetterBound  = false;
static bool                            g_fsr2ShimsBound       = false;
static std::map<void*, DeviceData>     g_devices;

// Command-buffer and present functions carry no device handle we can key on
// cheaply per call, so cache the ones we forward on a hot path. Looking these
// up through a map on every call - and worse, default-constructing a null entry
// on a miss - is exactly how the sibling project lost a device.
static PFN_vkQueuePresentKHR    g_nextQueuePresent = nullptr;

// Which queues X-Plane actually submits rendering to, versus the queue it
// presents on.
//
// The velocity pass submits its work to the PRESENT queue and relies on
// submission order to be sequenced after the frame's rendering. That is only
// true if they are the same queue. If X-Plane renders on one queue and presents
// on another, our depth barrier races against their rendering with no
// synchronisation at all - which would produce a corrupted frame with no API
// error, most likely during a load when queue usage is at its busiest.
//
// This records the truth rather than continuing to assume it.
static std::set<VkQueue> g_submitQueues;
static PFN_vkQueueSubmit g_nextQueueSubmit = nullptr;


// Which of the two command buffers reaches the queue first. Both are family 0
// and family 0 has exactly one queue, so execution is strictly serialised: the
// ONLY way the delivery can read outImg before FSR2 writes it is if the sim
// submits the delivery buffer first. Engines do exactly that when they
// composite the previous frame while recording the next.
static void noteSubmitOrder(uint32_t count, const VkCommandBuffer *cbs)
{
    if (!cbs) return;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t seq = ++g_submitSeq;
        if (cbs[i] == g_fsr2LastDispatchCb) g_seqOfDispatchCb = seq;
        if (cbs[i] == g_deliveryCb)         g_seqOfDeliveryCb = seq;
    }
    if (g_seqOfDispatchCb != 0xFFFFFFFF && g_seqOfDeliveryCb != 0xFFFFFFFF) {
        static bool told = false;
        if (!told) {
            told = true;
            trace("SUBMIT ORDER: FSR2 dispatch cb submitted #%u, delivery cb "
                  "submitted #%u  -> %s",
                  g_seqOfDispatchCb, g_seqOfDeliveryCb,
                  (g_seqOfDeliveryCb < g_seqOfDispatchCb)
                      ? "DELIVERY RUNS FIRST. Our read executes before the write "
                        "that fills the image, on the same queue, and no barrier "
                        "can reorder that."
                      : "dispatch runs first, so the write is ordered before our "
                        "read and submission order is NOT the fault.");
        }
    }
}

static VKAPI_ATTR VkResult VKAPI_CALL Layer_QueueSubmit(
    VkQueue queue, uint32_t count, const VkSubmitInfo *submits, VkFence fence)
{
    for (uint32_t si = 0; si < count && submits; ++si)
        noteSubmitOrder(submits[si].commandBufferCount,
                        submits[si].pCommandBuffers);

    {
        std::lock_guard<std::mutex> g(g_lock);
        if (g_submitQueues.insert(queue).second)
            trace("QUEUE: app submits on %p (%u distinct so far)",
                  (void*)queue, (unsigned)g_submitQueues.size());
    }
    // The upload governor. It handles the call entirely when the queue is the
    // transfer-only family - pacing under pressure, holding whole submissions
    // FIFO - and it also flushes anything held whose signals this submission
    // waits on, whatever the queue. Deadlock-proofing is documented in vram.h.
    {
        VkResult vr = VK_SUCCESS;
        if (vram::onSubmit(queue, count, submits, fence, &vr)) return vr;
    }
    // Serialised against the frame generation present worker. VkQueue is
    // externally synchronised and that worker uses one from another thread, so
    // without this the two race - which is what took the sim down the first
    // time present was moved off the render thread. The layer sees both sides,
    // so the layer is what provides the guarantee.
    return g_nextQueueSubmit ? g_nextQueueSubmit(queue, count, submits, fence)
                             : VK_SUCCESS;
}

// ============================================================ VRAM SYSTEM HOOKS
//
// The interception surface the VRAM system runs on. Each hook is a thin
// passthrough that feeds vram:: and forwards; every decision lives in vram.h.

static PFN_vkFreeMemory        g_nextFreeMemory        = nullptr;
static PFN_vkBindImageMemory   g_nextBindImageMemory   = nullptr;
static PFN_vkBindImageMemory2  g_nextBindImageMemory2  = nullptr;
static PFN_vkBindBufferMemory  g_nextBindBufferMemory  = nullptr;
static PFN_vkBindBufferMemory2 g_nextBindBufferMemory2 = nullptr;
static PFN_vkGetDeviceQueue    g_nextGetDeviceQueue    = nullptr;
static PFN_vkWaitForFences     g_nextWaitForFences     = nullptr;
static PFN_vkGetFenceStatus    g_nextGetFenceStatus    = nullptr;
static PFN_vkResetFences       g_nextResetFences       = nullptr;
static PFN_vkWaitSemaphores    g_nextWaitSemaphores    = nullptr;
static PFN_vkQueueWaitIdle     g_nextQueueWaitIdle     = nullptr;
static PFN_vkDeviceWaitIdle    g_nextDeviceWaitIdle    = nullptr;
static PFN_vkQueueBindSparse   g_nextQueueBindSparse   = nullptr;
static PFN_vkCmdCopyBuffer     g_nextCmdCopyBuffer     = nullptr;

// Category of a live handle, for the bind hooks. -1 when the handle is not in
// the ledger (created before the layer attached, or the ledger is off).
static int vramCatOfImageHandle(VkImage img)
{
    std::lock_guard<std::mutex> g(g_lock);
    std::map<VkImage, VramEntry>::iterator it = g_vramImg.find(img);
    return it == g_vramImg.end() ? -1 : it->second.cat;
}

static int vramCatOfBufferHandle(VkBuffer buf)
{
    std::lock_guard<std::mutex> g(g_lock);
    std::map<VkBuffer, VramEntry>::iterator it = g_vramBuf.find(buf);
    return it == g_vramBuf.end() ? -1 : it->second.cat;
}

static VKAPI_ATTR void VKAPI_CALL Vram_FreeMemory(
    VkDevice device, VkDeviceMemory mem, const VkAllocationCallbacks *alloc)
{
    if (!g_nextFreeMemory) return;
    if (mem == VK_NULL_HANDLE) { g_nextFreeMemory(device, mem, alloc); return; }
    // The pool may keep the block; if it does, the driver never sees this free
    // and a later identical allocation is answered without a driver call.
    if (vram::poolHold(mem)) return;
    vram::noteFreeGone(mem);
    g_nextFreeMemory(device, mem, alloc);
}

static VKAPI_ATTR VkResult VKAPI_CALL Vram_BindImageMemory(
    VkDevice device, VkImage image, VkDeviceMemory mem, VkDeviceSize offset)
{
    int cat = vramCatOfImageHandle(image);
    if (cat >= 0)
        vram::onBind(mem, cat, g_share && g_share->valid,
                     vram::churnHot(image));
    vram::noteImageMem(image, mem);
    return g_nextBindImageMemory
         ? g_nextBindImageMemory(device, image, mem, offset)
         : VK_ERROR_INITIALIZATION_FAILED;
}

static VKAPI_ATTR VkResult VKAPI_CALL Vram_BindImageMemory2(
    VkDevice device, uint32_t count, const VkBindImageMemoryInfo *infos)
{
    for (uint32_t i = 0; i < count && infos; ++i) {
        int cat = vramCatOfImageHandle(infos[i].image);
        if (cat >= 0)
            vram::onBind(infos[i].memory, cat, g_share && g_share->valid,
                         vram::churnHot(infos[i].image));
        vram::noteImageMem(infos[i].image, infos[i].memory);
    }
    return g_nextBindImageMemory2
         ? g_nextBindImageMemory2(device, count, infos)
         : VK_ERROR_INITIALIZATION_FAILED;
}

static VKAPI_ATTR VkResult VKAPI_CALL Vram_BindBufferMemory(
    VkDevice device, VkBuffer buffer, VkDeviceMemory mem, VkDeviceSize offset)
{
    int cat = vramCatOfBufferHandle(buffer);
    if (cat >= 0)
        vram::onBind(mem, cat, g_share && g_share->valid);
    vram::noteBufBind(buffer, mem, offset);
    return g_nextBindBufferMemory
         ? g_nextBindBufferMemory(device, buffer, mem, offset)
         : VK_ERROR_INITIALIZATION_FAILED;
}

static VKAPI_ATTR VkResult VKAPI_CALL Vram_BindBufferMemory2(
    VkDevice device, uint32_t count, const VkBindBufferMemoryInfo *infos)
{
    for (uint32_t i = 0; i < count && infos; ++i) {
        int cat = vramCatOfBufferHandle(infos[i].buffer);
        if (cat >= 0)
            vram::onBind(infos[i].memory, cat, g_share && g_share->valid);
        vram::noteBufBind(infos[i].buffer, infos[i].memory,
                          infos[i].memoryOffset);
    }
    return g_nextBindBufferMemory2
         ? g_nextBindBufferMemory2(device, count, infos)
         : VK_ERROR_INITIALIZATION_FAILED;
}

// The app's own mappings, recorded so upload payloads can be read through the
// pointer the APP holds - the legal route into staging contents. Recording a
// pointer is not mapping; the spec's one-map rule is untouched.
static PFN_vkMapMemory   g_nextMapMemory   = nullptr;
static PFN_vkUnmapMemory g_nextUnmapMemory = nullptr;

static VKAPI_ATTR VkResult VKAPI_CALL Vram_MapMemory(
    VkDevice device, VkDeviceMemory mem, VkDeviceSize offset,
    VkDeviceSize size, VkMemoryMapFlags flags, void **ppData)
{
    VkResult r = g_nextMapMemory
               ? g_nextMapMemory(device, mem, offset, size, flags, ppData)
               : VK_ERROR_INITIALIZATION_FAILED;
    if (r == VK_SUCCESS && ppData)
        vram::noteMap(mem, offset, size, *ppData);
    return r;
}

static VKAPI_ATTR void VKAPI_CALL Vram_UnmapMemory(
    VkDevice device, VkDeviceMemory mem)
{
    vram::noteUnmap(mem);
    if (g_nextUnmapMemory) g_nextUnmapMemory(device, mem);
}

static VKAPI_ATTR void VKAPI_CALL Vram_GetDeviceQueue(
    VkDevice device, uint32_t family, uint32_t index, VkQueue *out)
{
    if (!g_nextGetDeviceQueue) return;
    g_nextGetDeviceQueue(device, family, index, out);
    if (out && *out) vram::noteQueue(family, *out);
}

// Every wait path releases held submissions first - the deadlock-proofing.
// ---- WHERE THE FRAME ACTUALLY GOES. (the 38 fps serialisation)
//
// The sim's own readout says frame time = CPU time + GPU time EXACTLY
// (26.3 = 16.5 + 9.9 ms), and that sum is the signature of no overlap: a
// pipelined renderer's frame is the LARGER of the two, not the total. Something
// makes the CPU wait for the GPU inside the frame instead of a frame or two
// behind it, and these counters name it without guessing from a decompile.
//
// Blocking time is attributed to the call that blocks - fence waits, semaphore
// waits, acquire, present - and reported per second alongside the frame count.
// A fence figure close to the GPU time is the classic single-frame-in-flight
// stall; an acquire figure that size is a swapchain with too few images or a
// present mode that blocks; a present figure is the display path.
static std::atomic<uint64_t> g_blkFenceUs(0), g_blkSemUs(0),
                             g_blkAcquireUs(0), g_blkPresentUs(0);
static std::atomic<uint64_t> g_blkFenceN(0), g_blkAcquireN(0);

static inline uint64_t nowUs()
{
    LARGE_INTEGER c, f;
    QueryPerformanceCounter(&c);
    QueryPerformanceFrequency(&f);
    return f.QuadPart ? (uint64_t)(c.QuadPart * 1000000ll / f.QuadPart) : 0;
}

static VKAPI_ATTR VkResult VKAPI_CALL Vram_WaitForFences(
    VkDevice device, uint32_t count, const VkFence *fences,
    VkBool32 waitAll, uint64_t timeout)
{
    vram::touchFences(count, fences);
    if (!g_nextWaitForFences) return VK_ERROR_INITIALIZATION_FAILED;
    const uint64_t t0 = nowUs();
    VkResult r = g_nextWaitForFences(device, count, fences, waitAll, timeout);
    g_blkFenceUs.fetch_add(nowUs() - t0, std::memory_order_relaxed);
    g_blkFenceN.fetch_add(1, std::memory_order_relaxed);
    return r;
}

// vkWaitForFences is never called by this engine (measured: 0 waits per frame),
// so if the CPU is waiting for the GPU it is doing it by POLLING - and a spin
// on vkGetFenceStatus is charged to the frame as CPU time, not as blocked time.
// That would report as "CPU bound" on a machine doing no work, and would
// produce frame time = CPU + GPU exactly. Count the calls and how many came
// back NOT_READY: a large ready:notready ratio per frame is a spin loop.
static std::atomic<uint64_t> g_fenceStatusN(0), g_fenceStatusNotReady(0);

static VKAPI_ATTR VkResult VKAPI_CALL Vram_GetFenceStatus(
    VkDevice device, VkFence fence)
{
    vram::touchFences(1, &fence);
    if (g_nextGetFenceStatus) {
        VkResult r = g_nextGetFenceStatus(device, fence);
        g_fenceStatusN.fetch_add(1, std::memory_order_relaxed);
        if (r == VK_NOT_READY)
            g_fenceStatusNotReady.fetch_add(1, std::memory_order_relaxed);
        return r;
    }
    return g_nextGetFenceStatus ? g_nextGetFenceStatus(device, fence)
                                : VK_ERROR_INITIALIZATION_FAILED;
}

static VKAPI_ATTR VkResult VKAPI_CALL Vram_ResetFences(
    VkDevice device, uint32_t count, const VkFence *fences)
{
    vram::touchFences(count, fences);
    return g_nextResetFences ? g_nextResetFences(device, count, fences)
                             : VK_ERROR_INITIALIZATION_FAILED;
}

static VKAPI_ATTR VkResult VKAPI_CALL Vram_WaitSemaphores(
    VkDevice device, const VkSemaphoreWaitInfo *info, uint64_t timeout)
{
    if (info)
        vram::touchSemaphores(info->semaphoreCount, info->pSemaphores,
                              info->pValues);
    if (!g_nextWaitSemaphores) return VK_ERROR_INITIALIZATION_FAILED;
    const uint64_t t0 = nowUs();
    VkResult r = g_nextWaitSemaphores(device, info, timeout);
    g_blkSemUs.fetch_add(nowUs() - t0, std::memory_order_relaxed);
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL Vram_QueueWaitIdle(VkQueue queue)
{
    vram::flushAll(&vram::flushOnWait);
    return g_nextQueueWaitIdle ? g_nextQueueWaitIdle(queue)
                               : VK_ERROR_INITIALIZATION_FAILED;
}

static VKAPI_ATTR VkResult VKAPI_CALL Vram_DeviceWaitIdle(VkDevice device)
{
    vram::flushAll(&vram::flushOnWait);
    return g_nextDeviceWaitIdle ? g_nextDeviceWaitIdle(device)
                                : VK_ERROR_INITIALIZATION_FAILED;
}

static VKAPI_ATTR VkResult VKAPI_CALL Vram_QueueBindSparse(
    VkQueue queue, uint32_t count, const VkBindSparseInfo *infos, VkFence fence)
{
    ++vram::sparseBinds;
    // A sparse bind can wait on semaphores a held submission signals.
    for (uint32_t i = 0; i < count && infos; ++i) {
        const VkTimelineSemaphoreSubmitInfo *tl = nullptr;
        for (const VkBaseInStructure *p =
                 (const VkBaseInStructure*)infos[i].pNext; p; p = p->pNext)
            if (p->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO)
                tl = (const VkTimelineSemaphoreSubmitInfo*)p;
        vram::touchSemaphores(infos[i].waitSemaphoreCount,
                              infos[i].pWaitSemaphores,
                              tl ? tl->pWaitSemaphoreValues : nullptr);
    }
    return g_nextQueueBindSparse
         ? g_nextQueueBindSparse(queue, count, infos, fence)
         : VK_ERROR_INITIALIZATION_FAILED;
}

static VKAPI_ATTR void VKAPI_CALL Vram_CmdCopyBuffer(
    VkCommandBuffer cb, VkBuffer src, VkBuffer dst,
    uint32_t count, const VkBufferCopy *regions)
{
    if (!g_nextCmdCopyBuffer) return;
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < count && regions; ++i) bytes += regions[i].size;
    if (bytes) vram::chargeCopy(cb, bytes);
    g_nextCmdCopyBuffer(cb, src, dst, count, regions);
}

// Descriptor-set allocation counter (SS35) - measurement only; the engine
// owns its descriptors and the layer's job is to know whether they churn.
static PFN_vkAllocateDescriptorSets g_nextAllocDescSets = nullptr;
static VKAPI_ATTR VkResult VKAPI_CALL Vram_AllocateDescriptorSets(
    VkDevice device, const VkDescriptorSetAllocateInfo *ai,
    VkDescriptorSet *out)
{
    if (ai) vram::noteDescriptorAllocs(ai->descriptorSetCount);
    return g_nextAllocDescSets
         ? g_nextAllocDescSets(device, ai, out)
         : VK_ERROR_INITIALIZATION_FAILED;
}
static PFN_vkCreateSampler      g_nextCreateSampler = nullptr;
static PFN_vkCmdSetViewport     g_nextCmdSetViewport = nullptr;
static PFN_vkCmdBeginRenderPass g_nextCmdBeginRenderPass = nullptr;
static PFN_vkCmdBeginRendering  g_nextCmdBeginRendering  = nullptr;

// Defined below, but needed by the passes: they report the formats they bind,
// and a bare enum value is the kind of detail that gets skimmed past.
static const char *formatName(VkFormat f);

// ---- THE DEPTH-DERIVED VELOCITY PASS IS GONE.
//
// It reconstructed per-pixel velocity from depth plus the two most recent camera
// matrices. That was the original approach and it is obsolete twice over: the
// injected shaders now emit TRUE motion vectors from the actual clip positions,
// which is exact where reprojection is an approximation, and the pass itself was
// the confirmed cause of the stutter - a full-resolution dispatch plus barriers
// transitioning X-Plane's depth image every frame, forcing a pipeline flush that
// queued texture uploads behind it.
//
// It has been switched off since then and only the wiring remained. All that was
// still load-bearing is the depth LAYOUT it tracked, which FSR2 needs to know so
// it can borrow and return X-Plane's depth image in the state it was left in.
// That is one variable, not a compute pass.
#include "vk_util.h"

static VkImageLayout g_sceneDepthLayout = VK_IMAGE_LAYOUT_UNDEFINED;
static VkImageView   g_sceneDepthView   = VK_NULL_HANDLE;
#include "mv_target.h"
#include "spirv_inject.h"
#include "xpfsr_spv.h"
#include "fsr_probe.h"
#include "fsr3_backend_impl.h"

// Defined with the probe below; called from the present path, which comes
// first in this file.
static void fsrProbeResolve();

// ---- DEVICE FUNCTIONS, DOWN THE CHAIN.
//
// ffx_vk_shim.cpp answers vkGetDeviceProcAddr and vkCreateBuffer for the
// FidelityFX objects; both come here. Resolving through the DeviceData we
// already hold means the call continues below this layer instead of restarting
// at the loader - which is the recursion that killed the sim inside
// ffxGetScratchMemorySizeVK.
extern "C" PFN_vkVoidFunction mvNextDeviceProcAddr(VkDevice device, const char *name)
{
    std::lock_guard<std::mutex> g(g_lock);
    std::map<void*, DeviceData>::iterator it = g_devices.find(dispatchKey(device));
    if (it == g_devices.end()) it = g_devices.begin();
    if (it == g_devices.end() || !it->second.gdpa) return nullptr;
    return it->second.gdpa(device, name);
}
#include "taa.h"
#include "destruct_gpu.h"


// FSR2 is optional at BUILD time as well as run time. Its static library takes
// several minutes to produce, so requiring it in order to compile the layer
// would make every unrelated change slow. Build it with build-fsr2.ps1 and the
// layer picks it up; without it everything else still works and the upscaler
// selection falls back to the built-in resolve.

// DLSS needs no static library and no link step - see dlss_loader.h for why
// the SDK's own archive is unusable here and why that turns out not to matter.
// The flag exists only so the NGX headers are optional at build time.

// Streamline, driven from here because this is where the device, the swapchain
// and every resource it needs to be told about actually live. The shim next to
// X-Plane.exe only starts it.

// ------------------------------------------------- scene depth discovery

static bool isDepthFormat(VkFormat f)
{
    switch (f) {
        case VK_FORMAT_D16_UNORM:
        case VK_FORMAT_X8_D24_UNORM_PACK32:
        case VK_FORMAT_D32_SFLOAT:
        case VK_FORMAT_D16_UNORM_S8_UINT:
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return true;
        default:
            return false;
    }
}

static const char *formatName(VkFormat f)
{
    switch (f) {
        case VK_FORMAT_D16_UNORM:           return "D16_UNORM";
        case VK_FORMAT_X8_D24_UNORM_PACK32: return "X8_D24_UNORM_PACK32";
        case VK_FORMAT_D32_SFLOAT:          return "D32_SFLOAT";
        case VK_FORMAT_D16_UNORM_S8_UINT:   return "D16_UNORM_S8_UINT";
        case VK_FORMAT_D24_UNORM_S8_UINT:   return "D24_UNORM_S8_UINT";
        case VK_FORMAT_D32_SFLOAT_S8_UINT:  return "D32_SFLOAT_S8_UINT";

        // Colour formats. The table used to hold depth only, so the scene
        // colour target printed as "?" - which hid the single most important
        // fact about it, namely whether it is an HDR float format or an 8-bit
        // one that has already been tonemapped.
        case VK_FORMAT_R16G16B16A16_SFLOAT: return "R16G16B16A16_SFLOAT";
        case VK_FORMAT_R32G32B32A32_SFLOAT: return "R32G32B32A32_SFLOAT";
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return "A2B10G10R10_UNORM";
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32:  return "B10G11R11_UFLOAT";
        case VK_FORMAT_R8G8B8A8_UNORM:      return "R8G8B8A8_UNORM";
        case VK_FORMAT_R8G8B8A8_SRGB:       return "R8G8B8A8_SRGB";
        case VK_FORMAT_B8G8R8A8_UNORM:      return "B8G8R8A8_UNORM";
        case VK_FORMAT_B8G8R8A8_SRGB:       return "B8G8R8A8_SRGB";
        default:                            return "?";
    }
}

// The 3D scene's COLOUR target - what the resolve reads and writes.
//
// Found the same way the depth image is: from the last full-viewport pass that
// carries depth. The 3D scene has a depth attachment, the 2D overlays do not,
// so the last depth-bearing pass is the last one drawing world geometry, and
// its colour attachment is the image the resolve has to operate on.
//
// The usage flags are the thing to watch. Reading it needs SAMPLED_BIT and
// writing needs STORAGE_BIT, and an application has no reason to set either on
// an image it only ever renders into. Depth had exactly this problem, and it
// was only caught because the flags were being recorded. They can only be read
// at creation time, so every colour target is captured and matched to the frame
// later.
struct ColorTarget {
    VkImage        image  = VK_NULL_HANDLE;
    VkFormat       format = VK_FORMAT_UNDEFINED;
    uint32_t       w = 0, h = 0;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    // ---- ARRAY LAYERS WERE NEVER RECORDED, AND THAT IS THE TAA BUG.
    //
    // The shader corpus says gbuffer_lit exists in three shapes: plain 2D (288
    // permutations), 2D ARRAY (288), and 2D array MULTISAMPLED (576). The array
    // layer is the stereo eye. Our resolve builds a plain VK_IMAGE_VIEW_TYPE_2D
    // single-sample view and copies with layerCount 1 no matter which shape the
    // target really is - so in exactly the configurations that use an arrayed or
    // multisampled target, the view does not describe the image. That is the
    // black-in-some-camera-views symptom, and no amount of refining WHICH pass
    // we pick could ever have fixed it, because it is not a pass-choice fault.
    uint32_t       arrayLayers = 1;
    VkImageUsageFlags usage = 0;
    VkImageLayout  layout = VK_IMAGE_LAYOUT_UNDEFINED;
};
static std::map<VkImage, ColorTarget> g_colorImages;   // every colour image made

// Defined further down, next to the transfer census where the rest of the
// image-identification work lives. Declared here because the colour-image census
// runs long before it.
static void noteGbufferVelCandidate(const ColorTarget &c);
// The identified gbuffer_vel image. Lives up here because both the census
// (early) and the resolve (middle) touch it, while the identification logic
// stays with the transfer census below.
static VkImage g_gbufferVelCandidate = VK_NULL_HANDLE;

// Defined with the injector plumbing much further down; the full-state report
// calls them and sits above it.
static void mvLogInjectReasons();
static uint64_t g_layoutPatched, g_layoutSkipped;
static ColorTarget g_sceneColor;                       // the 3D one, this frame
static bool        g_sceneColorReported = false;
static uint32_t    g_sceneResolveMode   = 0;
static VkImage     g_sceneResolveImage  = VK_NULL_HANDLE;
static std::vector<VkImage> g_hdrTargets;    // distinct HDR scene targets seen
static VkImage     g_sceneColorLast     = VK_NULL_HANDLE;
static uint32_t    g_sceneColorStable   = 0;
static bool        g_resInitTried       = false;

struct DepthCandidate {
    VkImage           image;
    VkFormat          format;
    uint32_t          w, h;
    VkSampleCountFlagBits samples;
    // Depth is array-layered for the same reason colour is - one layer per eye.
    // Tracked here as well as on ColorTarget because a consumer needs an
    // image's SHAPE, not just its handle: binding a plain 2D view of a
    // two-layer target silently reaches one eye, which is how the black
    // camera views went unexplained for so long.
    uint32_t          arrayLayers;
    VkImageUsageFlags usage;
    bool              sampled;    // usage has SAMPLED_BIT
};

static std::vector<DepthCandidate> g_depthCandidates;

// imageView -> image. Render passes reference depth by VIEW, but everything we
// need (format, extent, sample count) is recorded per IMAGE, so the two have to
// be tied together to answer "which image does the scene actually render into".
static std::map<VkImageView, VkImage> g_viewToImage;

// commandBuffer -> device. Command recording functions carry no device handle,
// but the dispatch table is per-device, so the two have to be tied together to
// record anything from inside a Cmd* hook.
static std::map<VkCommandBuffer, VkDevice> g_cbToDevice;

// Index of the last full-viewport depth pass seen in the PREVIOUS frame.
//
// The dispatch has to go after the final depth write, but a frame's structure
// is only known once it has finished. X-Plane's frame is stable (31-33 passes,
// depth ending at 29), so the previous frame's index is a sound predictor, and
// being wrong for one frame after a structural change costs one stale velocity
// buffer rather than anything visible.
static int g_lastDepthPassIdx     = -1;
static int g_prevLastDepthPassIdx = -1;

// Set by the present hook when the next recorded frame should also copy the
// velocity image back for a disk dump.
static bool g_velWantDump = false;

// Snapshot and gating flags, published by the present hook for the NEXT frame's
// recording. Recording happens mid-frame, before present, so it has to work
// from the state the previous present established.
static Snapshot g_velSnap;
// TAA_MV_PLUGIN_REPROJ restores the plugin's own flight-loop pairing, so the
// two can be compared rather than argued about.
static const bool g_usePluginReproj = (getenv("TAA_MV_PLUGIN_REPROJ") != nullptr);
static bool     g_velArmed  = false;
static bool     g_velInjectedThisFrame = false;

// ---- ONCE A FRAME, ON THE LAST SCENE PASS.
//
// Two fixes that were each made once and lost when the resolve block was moved
// out of the render pass. Kept together so they travel together.
//
// ONCE A FRAME: X-Plane records 27 passes and several end with the scene flag
// set. Resolving at each applied mix(history, current, 0.1) repeatedly inside
// one frame - 0.1^k - crushing terrain to black and leaving only the very
// bright HDR sky. Accumulation is a temporal operator; applying it more than
// once per frame is a different operator, not a mis-tuning.
//
// LAST, NOT FIRST: the first flagged pass catches the frame half drawn, so the
// aircraft, cockpit and overlays are written into history as black. History
// feeds itself, so that black then bleeds outward a pixel a frame through the
// bilinear fetch - it climbs the gear strut and eventually takes the screen.
//
// Which pass is last is unknowable while recording it, so the previous frame's
// count is used. A wrong guess costs one frame of history, not a corrupt image.
static bool     g_taaResolvedThisFrame = false;
static uint32_t g_sceneEndsThisFrame   = 0;
static uint32_t g_sceneEndsLastFrame   = 0;
// Candidate HDR passes seen so far this frame, and how many the LAST frame had.
// The resolve has to run on the final one - see the block that increments this
// - and the final one is only identifiable in arrears.
static uint32_t g_hdrPassesThisFrame   = 0;
static uint32_t g_hdrPassesLastFrame   = 0;
// How far down the resolve gate chain the frame got, as a high-water mark:
//   0 no candidate lit pass matched at all (pass identification is the fault)
//   1 candidate lit pass seen        2 entered resolve body (fresh bind, no dump)
//   3 device + velocity target ready 4 HDR float format confirmed
//   5 chosen as last HDR pass        6 backend accepted the frame
//   7 quiesce clear                  8 scene target bound
//   9 RESOLVED
// resolved=0 on the GATE line says THAT it never ran; this says WHERE it died.
static std::atomic<uint32_t> g_gateDepthThisFrame{0};
static std::atomic<uint32_t> g_gateDepthLastFrame{0};
// HDR candidate passes that END after the resolve already recorded this
// frame - the overwrite census behind the "output only survives in flashes"
// symptom. Reset at present.
static std::atomic<uint32_t> g_hdrAfterResolveSame{0};
static std::atomic<uint32_t> g_hdrAfterResolveOther{0};
static inline void gateReach(uint32_t d) {
    uint32_t cur = g_gateDepthThisFrame.load(std::memory_order_relaxed);
    while (cur < d && !g_gateDepthThisFrame.compare_exchange_weak(cur, d)) {}
}
// The image our resolve wrote into this frame, watched for the rest of it by
// noteSsrFeedbackCheck - if it becomes the source of a half-resolution transfer,
// our output is feeding SSR's reflection pyramid and the loop is closed.
static VkImage  g_taaWroteImageThisFrame = VK_NULL_HANDLE;
// Set while resolves are being skipped because the velocity field is stale;
// consumed by the first live resolve, which must reset history rather than
// blend pre-freeze accumulation into the resumed picture.
static bool     g_taaStaleResume = false;
static int      dumpEvery = 0;   // frames between dumps; 0 = off, set at runtime

// Per-command-buffer record of whether it rendered into OUR depth image, and in
// what layout.
//
// This replaces the pass-index approach, which could not work: g_passesThisFrame
// is a plain global incremented from vkCmdBeginRendering, and X-Plane records
// command buffers on several threads at once. The counter was racy, so "is this
// the last depth pass" was never a meaningful question to ask of it. The
// diagnostic showed it stuck at 24 while waiting for 29.
//
// Keying on the command buffer sidesteps ordering entirely: each buffer knows
// what it contains, regardless of which thread built it.
struct CbDepthUse {
    bool          used = false;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    int           depthPasses = 0;   // non-clearing passes seen in this buffer
};
static std::map<VkCommandBuffer, CbDepthUse> g_cbDepthUse;
static bool     g_velStable = false;
static PFN_vkCmdEndRendering g_nextCmdEndRendering = nullptr;

// The depth image bound by the LAST depth-bearing pass of the previous frame.
//
// This is the authoritative answer to which buffer holds scene depth, and it
// replaces scoring candidates by format and sample count. That heuristic picked
// a plausible-looking 2560x1440 D32_SFLOAT image out of four at that size and
// got it wrong: the sampled depth came back constant, every reconstructed point
// landed at the 16.5 mm near plane, and the velocity field came out ~1000 px
// per frame. The frame tells us directly - there is no need to guess.
static VkImage g_lastDepthPassImage = VK_NULL_HANDLE;
static VkImage g_frameDepthImage    = VK_NULL_HANDLE;

// Every full-viewport depth image the frame binds, in binding order, plus the
// ones a content check has already disproved.
//
// The selection heuristic has now been wrong twice - once scoring by format,
// once taking the last depth pass - and each wrong answer cost a rebuild and
// another flight, because the disproof only arrived after the session ended.
// The list plus the reject set let the layer try the next candidate a few
// hundred frames later instead, so one flight converges on the right image
// rather than eliminating a single hypothesis.
// g_frameDepthList is the frame being recorded and is cleared every present.
// g_frameDepthListDone is the last COMPLETED frame, which is what selection
// reads.
//
// Two lists, because recording runs ahead of and across presents: choosing from
// the live list would mean choosing from a half-built one, whose last entry is
// wherever recording happens to have reached rather than the frame's final
// depth pass. The first version had one list and never cleared it, so it
// accumulated every distinct depth image of the session and selection settled
// on an image no pass was still binding - the marking test never matched it,
// nothing was ever injected, and the log showed a ready pipeline doing nothing.
static std::vector<VkImage> g_frameDepthList;
static std::vector<VkImage> g_frameDepthListDone;
static std::vector<VkImage> g_depthRejected;
static bool     g_depthProven   = false;   // a content check said yes
static uint64_t g_depthProbeAt  = 0;       // frame to run the next probe on

// Did a full-viewport depth pass actually run THIS frame, and in what layout?
//
// This gates the whole velocity dispatch, and it exists because of a real
// symptom: a white flash on aircraft load. During a load X-Plane stops
// rendering the 3D scene, so no depth pass runs - but the captured layout from
// before the load was still sitting there, and the pass went on barriering the
// depth image with an oldLayout it was no longer in. Declaring the wrong
// oldLayout is undefined behaviour, and a driver may legitimately decompress or
// clear the image, which is exactly what a flash looks like.
//
// So: never transition an image using a layout we did not observe this frame.
static bool          g_depthFreshThisFrame = false;
static VkImageLayout g_depthLayoutThisFrame = VK_IMAGE_LAYOUT_UNDEFINED;
static VkImage  g_sceneDepth  = VK_NULL_HANDLE;
static uint32_t g_sceneDepthW = 0, g_sceneDepthH = 0;
static bool     g_depthReported = false;

// The whole point of stage 1. Record every depth image and, critically, whether
// it can be sampled - because if X-Plane does not already set SAMPLED_BIT we
// cannot read depth from a compute shader without modifying image creation, and
// that changes the risk profile of the entire project.
static void noteDepthImage(const VkImageCreateInfo *ci, VkImage img)
{
    DepthCandidate c;
    c.image   = img;
    c.format  = ci->format;
    c.w       = ci->extent.width;
    c.h       = ci->extent.height;
    c.samples = ci->samples;
    c.arrayLayers = ci->arrayLayers;
    c.usage   = ci->usage;
    c.sampled = (ci->usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0;

    std::lock_guard<std::mutex> g(g_lock);
    if (g_depthCandidates.size() < 64)
        g_depthCandidates.push_back(c);

    trace("DEPTH image %ux%u fmt=%s samples=%u usage=0x%x sampled=%s",
          c.w, c.h, formatName(c.format), (unsigned)c.samples, c.usage,
          c.sampled ? "YES" : "NO <-- cannot be read by compute as-is");
}

// Selection is deliberately NOT done at creation time.
//
// The first version was, and it never fired: X-Plane creates its depth images
// during startup, long before the plugin has published the shared block, so the
// viewport to match against was still unknown and every candidate was skipped.
// The trace showed "depth=MISSING" for 2760 frames with the right image sitting
// in the candidate list the whole time.
//
// So candidates are recorded as they appear and the choice is made later, from
// present, once the viewport is actually known.
static void selectSceneDepth()
{
    if (g_sceneDepth != VK_NULL_HANDLE) return;

    // Take the image the frame ACTUALLY rendered depth into, captured from the
    // last full-viewport depth pass. Not a scored guess among candidates that
    // happen to share a resolution.
    //
    // The scoring version picked a 2560x1440 D32_SFLOAT image that looked ideal
    // on paper - sampleable, single-sampled, no stencil - out of four at that
    // size. The depth it returned was constant, so every pixel reconstructed to
    // the 16.5 mm near plane and the velocity field came out around a thousand
    // pixels per frame. Nothing about the format told us it was the wrong one;
    // only the frame could.
    // Walk backwards through the depth images this frame bound, skipping any
    // that a content check has already disproved. With nothing rejected this is
    // exactly the old behaviour - the last full-viewport depth pass.
    VkImage chosen = VK_NULL_HANDLE;
    {
        std::lock_guard<std::mutex> g(g_lock);
        for (size_t i = g_frameDepthListDone.size(); i-- > 0; ) {
            bool bad = false;
            for (size_t k = 0; k < g_depthRejected.size(); ++k)
                if (g_depthRejected[k] == g_frameDepthListDone[i]) { bad = true; break; }
            if (!bad) { chosen = g_frameDepthListDone[i]; break; }
        }
    }
    if (chosen == VK_NULL_HANDLE) chosen = g_frameDepthImage;
    if (chosen == VK_NULL_HANDLE) return;

    std::lock_guard<std::mutex> g(g_lock);
    const DepthCandidate *c = nullptr;
    for (size_t i = 0; i < g_depthCandidates.size(); ++i)
        if (g_depthCandidates[i].image == chosen) { c = &g_depthCandidates[i]; break; }

    if (!c) {
        static bool warned = false;
        if (!warned) { warned = true; trace("DEPTH: bound image is not a tracked candidate"); }
        return;
    }
    if (!c->sampled) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            trace("DEPTH: the scene depth image (%ux%u fmt=%s usage=0x%x) has no SAMPLED_BIT. "
                  "Compute cannot read it as-is.", c->w, c->h, formatName(c->format), c->usage);
        }
        return;
    }

    g_sceneDepth  = c->image;
    g_sceneDepthW = c->w;
    g_sceneDepthH = c->h;
    trace("DEPTH: selected FROM THE FRAME - %ux%u fmt=%s samples=%u usage=0x%x",
          c->w, c->h, formatName(c->format), (unsigned)c->samples, c->usage);

    // Print what the choice was made FROM, not only what it landed on. Without
    // this the log says a depth image was selected and gives no way to tell
    // whether there was one candidate or six, which is the difference between
    // "the pick is wrong" and "there is nothing else to pick".
    {
        char line[512];
        int len = snprintf(line, sizeof(line), "DEPTH: %zu candidate(s) this frame, last bound last:",
                           g_frameDepthListDone.size());
        for (size_t i = 0; i < g_frameDepthListDone.size() && len < (int)sizeof(line) - 40; ++i) {
            const char *mark = (g_frameDepthListDone[i] == c->image) ? "<==" : "";
            bool rej = false;
            for (size_t k = 0; k < g_depthRejected.size(); ++k)
                if (g_depthRejected[k] == g_frameDepthListDone[i]) { rej = true; break; }
            len += snprintf(line + len, sizeof(line) - len, " [%zu]%p%s%s",
                            i, (void*)g_frameDepthListDone[i], rej ? "REJ" : "", mark);
        }
        trace("%s", line);
    }
    if (c->samples != VK_SAMPLE_COUNT_1_BIT)
        trace("DEPTH: WARNING multisampled - needs a resolve before sampling");
}

// ------------------------------------------------------------- hooks

// =====================================================================
// CUSTOM TEXTURE PAGER
//
// X-Plane's pager decides a global "target scale" and applies it to
// everything. It cannot be reasoned with from outside: three separate budget
// levers were raised with no effect, and the one control that did work
// (max_overdrive) improved residency without stopping the resolution
// oscillation.
//
// This replaces the mechanism rather than arguing with it. The only lever a
// layer genuinely owns is IMAGE CREATION: vkCreateImage passes through us, so
// we can decide each texture's resolution before the application ever uploads
// to it. Dropping the top mip of a 2048x2048 texture makes it 1024x1024 and
// saves three quarters of its memory, permanently and per-texture, rather than
// as a global slider.
//
// WHY THIS WORKS WHERE EVICTION CANNOT. A layer cannot free X-Plane's images -
// it does not own them and the application still holds the handles. It cannot
// know which ortho tile is far away either. But it does not need to know any of
// that to decide that a 4096x4096 texture can be 2048x2048.
//
// THE UPLOAD MUST BE REMAPPED TO MATCH. The application still has full-size
// pixel data and will copy it in per-mip regions. Those regions have to be
// filtered - the dropped levels skipped - and the survivors renumbered, or the
// copy writes mip 0's data into a smaller mip 0 and the driver either faults or
// produces garbage. That remapping is the part that makes this honest rather
// than a trick.
// THE POLICY MAP IS KEYED ON A HANDLE, AND HANDLES ARE REUSED.
//
// This is what corrupted the ground textures after a long session, and it is
// worth stating plainly because nothing about it looks like a lifetime bug.
//
// The map recorded "this image lost a mip" and was never erased on destroy.
// Vulkan is free to hand back a previously freed VkImage value for a new image,
// and X-Plane streams scenery tiles in and out continuously, so eventually a
// brand new full-size texture is created at the address of a dead shrunken one.
// It inherits dropMips from a texture that no longer exists.
//
// The consequence is not a crash, which is why it survived twelve hours and
// then showed up as scenery. The new image is full size with a full mip chain -
// only the UPLOAD is remapped, so the region targeting mip 0 is discarded as
// "a level that no longer exists". Mip 0 is then never written, and sampling it
// returns whatever was in that memory: hard-edged blotches of unrelated
// content across runway and apron surfaces, worst where streaming churn is
// highest, and absent on the aircraft, which is loaded once and stays.
//
// So the entry must die with the image. Every handle-keyed map in a layer has
// this hazard; this one had teeth because a stale hit silently deletes data.
struct TexPolicy {
    uint32_t dropMips = 0;      // levels removed from the top
    uint32_t origW = 0, origH = 0;
};
static std::map<VkImage, TexPolicy> g_texPolicy;
static uint32_t g_pagerDropAbove = 0;     // drop a mip above this size; 0 = off
// The environment pins the pager thresholds; the VRAM system's zone policy
// drives them only when nothing was pinned. Set where the env is parsed.
static bool     g_pagerEnvLocked = false;

// Textures left alone because X-Plane had already scaled them to a
// non-power-of-two size. Counted rather than silent: if this is large, the two
// pagers are fighting over the same textures and ours is doing nothing.
static uint64_t g_pagerSkippedScaled = 0;
static uint32_t g_pagerMaxDrop = 1;       // levels a single texture may lose

// ---- AUTOGEN, TAKEN FURTHER DOWN THAN THE AIRCRAFT.
//
// The pager has one threshold for everything, and that is why it has always
// been a compromise: the setting that would shrink scenery usefully is the same
// setting that turns instrument faces to mush. A layer sees VkImage handles and
// dimensions, never filenames, so "only the scenery" cannot be asked directly.
//
// TIMING answers it well enough, and the pager already relies on this to bucket
// its own statistics: the aircraft is loaded during the loading screen, while
// scenery streams for as long as the flight runs. An image created while the
// share block is valid - that is, mid-flight - is streaming scenery, which on
// this sim means autogen buildings, trees and ortho tiles. The cockpit is not
// in that set, because it was resident before the flight started.
//
// So flight-time images get their own, harder target and their own drop limit,
// and load-time images keep the conservative behaviour that protects the
// aircraft. 0 disables it and restores a single policy for everything.
static uint32_t g_pagerAutogenTo = 0;     // target longest side for streamed textures

static uint64_t g_pagerSaved = 0;         // bytes not allocated
static uint64_t g_pagerImages = 0;

// WHOSE TEXTURES ARE WE SHRINKING - the aircraft's, or the scenery's?
//
// A layer sees VkImage handles and dimensions. It has no filenames and no
// object names, so a 4096 livery sheet and a 4096 ortho tile are the same
// thing to it, and "how much of the saving came out of the aircraft" cannot be
// answered from what is logged today.
//
// The one discriminator available is TIMING. The aircraft is loaded during the
// loading screen, while scenery streams continuously for as long as the flight
// runs - and the plugin only marks the shared block valid once a flight is
// actually running. So an image created while the share is invalid came from
// load (aircraft plus the initial scenery), and one created after came from
// streaming (scenery only).
//
// That is not a clean aircraft/scenery split and it is not presented as one:
// the load bucket contains initial scenery too. What it does answer is the
// question that decides policy - whether excluding load-time textures would
// cost a couple of hundred megabytes or most of the three gigabytes. Measuring
// that is cheaper than another round of arguing about it.
struct PagerBucket { uint64_t images = 0; uint64_t saved = 0; uint64_t at4096 = 0; };
static PagerBucket g_pagerLoad, g_pagerFlight;

// How many top levels to remove for an image of this size.
//
// Deliberately conservative and size-based: the biggest textures cost the most
// and lose the least perceptually, while small UI and instrument textures are
// left completely alone - those are the ones a global scale slider ruins first,
// and they are cheap to keep.
static uint32_t pagerDropLevelsRaw(const VkImageCreateInfo *ci)
{
    if (!g_pagerDropAbove) return 0;
    if (!pagerShouldEngage()) return 0;
    if (!(ci->usage & VK_IMAGE_USAGE_SAMPLED_BIT)) return 0;
    if (ci->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) return 0;
    if (ci->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) return 0;
    if (ci->usage & VK_IMAGE_USAGE_STORAGE_BIT) return 0;   // written by shaders
    if (ci->imageType != VK_IMAGE_TYPE_2D) return 0;
    if (ci->mipLevels < 2) return 0;                        // nothing to drop
    if (ci->arrayLayers != 1) return 0;                     // keep it simple

    // ---- HANDS OFF ANYTHING X-PLANE HAS ALREADY SCALED.
    //
    // Two pagers, one texture, and they cannot both have it.
    //
    // Ours drops top mip levels and renumbers the uploads, which is only valid
    // when every level is exactly half the last - true of a texture straight
    // off disk. X-Plane's own pager instead multiplies the base size by a
    // continuous ratio: released from its floor it chose 0.855262, so a 4096
    // texture arrives here at 3502.
    //
    // 3502 is not a power of two and, worse for a block-compressed format, not
    // a multiple of 4. BC stores 4x4 blocks, so the level sizes stop being
    // clean halves and the copy regions X-Plane submits no longer describe the
    // smaller image we made. That is a buffer overrun dressed as a texture
    // setting, and it took the sim down on the first view change after the
    // floor was released.
    //
    // So: if the dimensions are not both powers of two, X-Plane has already
    // taken its share and we take none. It keeps whatever ratio it computed and
    // our mip drop simply does not apply to that texture - which is the correct
    // answer rather than a compromise, because two independent reductions
    // multiplied together were never the intent.
    if ((ci->extent.width  & (ci->extent.width  - 1)) != 0 ||
        (ci->extent.height & (ci->extent.height - 1)) != 0) {
        ++g_pagerSkippedScaled;
        return 0;
    }

    uint32_t big = ci->extent.width > ci->extent.height
                 ? ci->extent.width : ci->extent.height;

    if (big <= g_pagerDropAbove) return 0;

    // ONE level by default, not two, and the difference is the aircraft.
    //
    // At two levels a 4096 texture goes to 1024, and 4096 is where livery
    // sheets live - 90 of them were shrunk in a two-minute session against 83
    // resident, so it is essentially all of them. A fuselage at 1024 is
    // visibly soft; at 2048 it is not.
    //
    // The measured cost of the cap is about 360 MB - 2438 MB saved at two
    // levels against 2077 MB at one. That was worth paying once the pager
    // freeze started holding scale 2.0 with 1.12 GB of headroom: the job
    // stopped being "save as much as possible" and became "save enough", and
    // the second level was buying margin we no longer need at a price paid
    // entirely in the things you look at closest.
    // Streamed scenery gets the harder target; anything loaded before the
    // flight began - the aircraft, the cockpit - keeps the gentle one.
    bool streamed = (g_share && g_share->valid);
    if (streamed && g_pagerAutogenTo && big > g_pagerAutogenTo) {
        uint32_t drop = 0;
        while (big > g_pagerAutogenTo && (ci->mipLevels - drop) > 1) {
            big >>= 1;
            ++drop;
        }
        return drop;
    }

    uint32_t drop = 0;
    while (big > g_pagerDropAbove && drop < g_pagerMaxDrop &&
           (ci->mipLevels - drop) > 1) {
        big >>= 1;
        ++drop;
    }
    return drop;
}

// ---- ONE ANSWER PER SHAPE, FOR THE LIFE OF THE PROCESS.
//
// X-Plane's VMA DEFRAGMENTER relocates an image by creating a new one from the
// same create info and copying the old into it, with copy regions sized for the
// ORIGINAL. That is only sound if both images came out the same size - and ours
// did not, because the answer above moves with the pressure zone: an image
// created in GREEN keeps every level, its defragment replacement created in
// CRITICAL loses two, and the copy then describes a source twice the size of
// the destination.
//
// Validation caught it exactly, in a command buffer named [Defragment]:
//   VUID-vkCmdCopyImage-dstOffset-00150
//     "extent.width (256) exceeds miplevel 0 which has a width of 128"
//     src Resources/bitmaps/runways/taxi_LIT.dds  256x128, 9 mips
//     dst                                         128x64,  8 mips
//   VUID-vkCmdCopyImage-dstSubresource-07967
//     "dstSubresource.mipLevel is 8, but has only 8 mip levels"
// Every region overruns and the last names a level that does not exist, which
// is a GPU fault, and the defragmenter runs during load - the crash window.
// A baseline run under validation shows none of these, so this is ours.
//
// Fixing the copy hooks cannot solve it: by the time the defragmenter records,
// the two images already have different shapes and the data genuinely does not
// fit. The decision itself has to be stable, so it is cached per SHAPE - the
// same key the engine's own create info produces - and reused unchanged
// afterwards. The cost is that a texture class settles on the first answer it
// is given; the benefit is that source and destination agree by construction,
// for the defragmenter and for anything else that assumes a recreated image
// matches the one it replaced.
//
// The zero answers are cached too, and must be: a shape that kept all its
// levels in GREEN must keep them later, or the mismatch simply runs the other
// way.
struct ShapeKey {
    uint32_t w, h, mips, layers, fmt;
    bool operator<(const ShapeKey &o) const {
        if (w != o.w) return w < o.w;
        if (h != o.h) return h < o.h;
        if (mips != o.mips) return mips < o.mips;
        if (layers != o.layers) return layers < o.layers;
        return fmt < o.fmt;
    }
};
static std::map<ShapeKey, uint32_t> g_shapeDrop;
// Shapes this pager has PRODUCED - see the double-drop note below.
static std::set<ShapeKey> g_shapeMade;

static uint32_t pagerDropLevels(const VkImageCreateInfo *ci)
{
    if (!ci) return 0;
    ShapeKey k;
    k.w      = ci->extent.width;
    k.h      = ci->extent.height;
    k.mips   = ci->mipLevels;
    k.layers = ci->arrayLayers;
    k.fmt    = (uint32_t)ci->format;
    {
        std::lock_guard<std::mutex> g(g_lock);
        // ---- NEVER SHRINK OUR OWN OUTPUT.
        //
        // Caching one answer per shape was not enough, and the way it failed
        // named the rest of the mechanism: the violation flipped from dstOffset
        // to srcOffset, so the SOURCE had become the smaller of the pair.
        //
        // The defragmenter does not replay the texture's original create info -
        // it recreates the image as it exists NOW, which is the shape WE already
        // reduced. That request is a different shape key, so it was evaluated
        // fresh and reduced a second time, and the new destination came out half
        // the source instead of matching it.
        //
        // A shape we produced is therefore already paged and must be reproduced
        // verbatim. The cost is that a texture whose natural size coincides with
        // one of our outputs is never paged; the benefit is that recreating any
        // image we touched returns exactly what it replaced.
        if (g_shapeMade.count(k)) return 0;
        std::map<ShapeKey, uint32_t>::iterator it = g_shapeDrop.find(k);
        if (it != g_shapeDrop.end()) return it->second;
    }
    uint32_t drop = pagerDropLevelsRaw(ci);
    // ---- THE REFINEMENT HAS TO BE INSIDE THE CACHE, NOT AFTER IT.
    //
    // refineDrop consults live churn and zone state, so it answers differently
    // for the same shape at different moments - which is precisely the property
    // the cache exists to remove. Applied at the call site it silently reopened
    // the defragmenter mismatch: two images of one shape, two sizes, and a copy
    // between them sized for neither. It runs here, once, and its answer is
    // what gets remembered. The eligibility guard is the one from the call site
    // (see the note there): refinement may only touch images that pass exactly
    // the tests pagerDropLevelsRaw enforces, or it can shrink a render target.
    // ---- REFINEMENT REFINES; IT DOES NOT DECIDE.
    //
    // refineDrop can RAISE a drop, including raising the 0 that means "the
    // pager is switched off" - so with g_pagerDropAbove at 0 and the VRAM
    // actuators live, images were still being shrunk by a path nobody thought
    // of as the pager. That is why the survivals and the crashes sorted by
    // whether the actuators were on rather than by the pager flag, and why
    // switching the pager off never stopped the ~2-minute death: the run that
    // survived eight minutes had the actuators off, which left refineDrop's
    // inputs inert, not the pager disabled.
    //
    // Requiring a pager that is genuinely enabled AND a base cut it already
    // chose makes the refinement what its name claims - an adjustment to a
    // decision, never the origin of one.
    if (g_pagerDropAbove && drop &&
        (ci->usage & VK_IMAGE_USAGE_SAMPLED_BIT) &&
        !(ci->usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                       VK_IMAGE_USAGE_STORAGE_BIT)) &&
        ci->imageType == VK_IMAGE_TYPE_2D &&
        ci->arrayLayers == 1 && ci->mipLevels >= 2 &&
        (ci->extent.width  & (ci->extent.width  - 1)) == 0 &&
        (ci->extent.height & (ci->extent.height - 1)) == 0)
        drop = vram::refineDrop(drop, ci->extent.width, ci->extent.height,
                                (uint32_t)ci->format, ci->mipLevels,
                                g_share && g_share->valid);
    // Cache what will ACTUALLY be applied, including this clamp, or the
    // remembered answer and the created image disagree.
    if (drop && (ci->mipLevels - drop) < 1) drop = 0;
    {
        std::lock_guard<std::mutex> g(g_lock);
        // Bounded: the engine's distinct texture shapes are a small set, and a
        // stale entry costs one texture's worth of quality, never correctness.
        if (g_shapeDrop.size() < 4096) g_shapeDrop[k] = drop;
        else if (drop) return drop;
        // Record what this drop will actually create, so a later recreation of
        // that image - by the defragmenter or anyone else - is left alone.
        if (drop && g_shapeMade.size() < 4096) {
            ShapeKey out;
            out.w      = k.w >> drop; if (!out.w) out.w = 1;
            out.h      = k.h >> drop; if (!out.h) out.h = 1;
            out.mips   = k.mips - drop;
            out.layers = k.layers;
            out.fmt    = k.fmt;
            g_shapeMade.insert(out);
        }
    }
    return drop;
}

// The other half of the pager, and the half that makes it correct.
//
// The application still holds full-size pixel data and copies it in per-mip
// regions. For a shrunk image those regions no longer describe it: region 0
// targets a mip that no longer exists, and every other region is numbered one
// or two levels too high.
//
// So regions for dropped levels are discarded and the rest renumbered. The
// buffer offsets are untouched - the data for mip 1 is still at mip 1's offset
// in the staging buffer, it simply becomes the new mip 0.
//
// Getting this wrong does not produce a subtle artefact. Copying mip 0's data
// into a quarter-sized mip 0 overruns the destination, and the driver either
// faults or writes over unrelated memory.
// Shrinking an image is not a local change.
//
// Dropping mip levels breaks every later call that names one: a view asking for
// levelCount = 12 on an image that now has 11, a barrier spanning levels that
// no longer exist, a blit generating mips into nothing. Each is undefined
// behaviour and in practice an immediate crash - which is what happened,
// several seconds after the shrink itself worked perfectly.
//
// So the pager has to fix up every entry point that references a level, not
// just the upload. This is the real cost of the approach, and it is the reason
// doing the shrink later rather than at load would not have helped: the same
// calls happen whenever a tile streams in.
static uint32_t pagerDropFor(VkImage img)
{
    std::lock_guard<std::mutex> g(g_lock);
    std::map<VkImage, TexPolicy>::iterator it = g_texPolicy.find(img);
    return (it == g_texPolicy.end()) ? 0 : it->second.dropMips;
}

// Clamp a subresource range to the levels the image actually has now.
static void pagerClampRange(VkImage img, VkImageSubresourceRange *r)
{
    uint32_t drop = pagerDropFor(img);
    if (!drop || !r) return;

    // baseMipLevel counts from the ORIGINAL top, so it shifts down by the
    // number of levels removed - and anything that pointed at a removed level
    // now points at the new level 0.
    const uint32_t oldBase = r->baseMipLevel;
    r->baseMipLevel = (oldBase > drop) ? (oldBase - drop) : 0;

    if (r->levelCount != VK_REMAINING_MIP_LEVELS) {
        // Map the range by its END, not by subtracting drop from the count.
        // Subtracting from both base and count shortens the range twice
        // whenever base <= drop: a barrier for original levels [1, 12) became
        // new levels [0, 11) instead of [0, 11]... one level short, so the
        // tail level never received its layout transition and was read in
        // whatever layout it happened to hold. Clamping by the end is exact
        // for every base, and can only shrink a range to fit the image.
        const uint32_t oldEnd = oldBase + r->levelCount;
        const uint32_t newEnd = (oldEnd > drop) ? (oldEnd - drop) : 1;
        r->levelCount = (newEnd > r->baseMipLevel)
                      ? (newEnd - r->baseMipLevel) : 1;
    }
}

// ---- WHAT LAYOUT EACH IMAGE IS ACTUALLY IN, RECORDED FROM THE SIM'S BARRIERS.
//
// Reading an image in a layout it is not in does not fail and does not warn.
// It tells the driver the contents need no decompression, so the compressed
// representation is read as if it were colour - which is the striping. It is
// the same wrong answer whether the guess is COLOR_ATTACHMENT, TRANSFER_SRC or
// anything else, so no better constant exists to pick.
//
// Every transition in the sim passes through the two hooks below. Recording
// them makes the layout a fact rather than a guess. Record order is execute
// order within a queue, so the value read while recording our delivery is the
// layout the image will be in when the delivery runs.
static std::map<VkImage, VkImageLayout> g_imgLayout;
static std::mutex                       g_imgLayoutLock;

// Only the images anyone asks about. This fired on EVERY image barrier the sim
// issues - a lock and a map insert per barrier, thousands per frame - and cost
// more than twenty frames a second. The table is a diagnostic for two specific
// images; watching the other thousands was pure overhead.
//
// The two globals are read without the lock on purpose: a stale handle here
// costs one missed sample in a debug table, and taking g_lock on this path is
// exactly what made it expensive.
static VkImage g_layoutWatchScene = VK_NULL_HANDLE;
static VkImage g_layoutWatchOut   = VK_NULL_HANDLE;

static void noteLayout(VkImage img, VkImageLayout l)
{
    if (img == VK_NULL_HANDLE || l == VK_IMAGE_LAYOUT_UNDEFINED) return;
    // Compared against the live globals directly. The previous version needed
    // "watch" handles armed first, and they were armed inside the block that
    // logs once - so the table was always empty at the only moment anything
    // read it, and printed UNKNOWN regardless of what the sim had recorded.
    // Nothing to arm, nothing to get out of order.
    if (img != g_sceneColor.image) return;
    std::lock_guard<std::mutex> g(g_imgLayoutLock);
    g_imgLayout[img] = l;
}

static bool knownLayout(VkImage img, VkImageLayout &out)
{
    std::lock_guard<std::mutex> g(g_imgLayoutLock);
    std::map<VkImage, VkImageLayout>::iterator it = g_imgLayout.find(img);
    if (it == g_imgLayout.end()) return false;
    out = it->second;
    return true;
}

static PFN_vkCmdPipelineBarrier g_nextCmdPipelineBarrier2 = nullptr;

static VKAPI_ATTR void VKAPI_CALL TAA_CmdPipelineBarrier(
    VkCommandBuffer cb, VkPipelineStageFlags src, VkPipelineStageFlags dst,
    VkDependencyFlags flags,
    uint32_t mCount, const VkMemoryBarrier *mem,
    uint32_t bCount, const VkBufferMemoryBarrier *buf,
    uint32_t iCount, const VkImageMemoryBarrier *img)
{
    if (!g_nextCmdPipelineBarrier2) return;

    // Recorded before any early return - the tracker must see every barrier,
    // not only the ones the mip clamp happens to rewrite.
    for (uint32_t i = 0; i < iCount && img; ++i) {
        noteLayout(img[i].image, img[i].newLayout);
        // UNDEFINED discards contents: the upload cache's identity for this
        // image dies here, or an elision after it would leave garbage.
        if (img[i].oldLayout == VK_IMAGE_LAYOUT_UNDEFINED)
            vram::contentInvalidate(img[i].image);
    }

    if (!g_pagerDropAbove || iCount == 0 || !img) {
        g_nextCmdPipelineBarrier2(cb, src, dst, flags, mCount, mem, bCount, buf,
                                  iCount, img);
        return;
    }

    std::vector<VkImageMemoryBarrier> fixed(img, img + iCount);
    for (uint32_t i = 0; i < iCount; ++i)
        pagerClampRange(fixed[i].image, &fixed[i].subresourceRange);

    g_nextCmdPipelineBarrier2(cb, src, dst, flags, mCount, mem, bCount, buf,
                              iCount, fixed.data());
}

// EVERY entry point that names a mip level, handled together.
//
// Shrinking an image invalidates all of them at once, and each is a crash
// rather than an artefact - so discovering the list one launch at a time is the
// wrong way to find it. X-Plane's extension list includes
// VK_KHR_synchronization2 and VK_KHR_copy_commands2, so the "2" variants are
// not hypothetical: they are most likely the ones actually in use, and hooking
// only the originals is why an earlier attempt still crashed with the barrier
// clamp supposedly in place.
static PFN_vkCmdPipelineBarrier2 g_nextCmdPipelineBarrier2KHR = nullptr;
static PFN_vkCmdBlitImage        g_nextCmdBlitImage = nullptr;
static PFN_vkCmdCopyImage        g_nextCmdCopyImage = nullptr;
static PFN_vkCmdClearColorImage  g_nextCmdClearColorImage = nullptr;

static VKAPI_ATTR void VKAPI_CALL TAA_CmdPipelineBarrier2(
    VkCommandBuffer cb, const VkDependencyInfo *info)
{
    if (!g_nextCmdPipelineBarrier2KHR) return;

    // X-Plane enables VK_KHR_synchronization2, so this is the variant actually
    // carrying most transitions. Tracking only the original would have left the
    // table describing an image the sim had since moved.
    if (info)
        for (uint32_t i = 0; i < info->imageMemoryBarrierCount; ++i) {
            noteLayout(info->pImageMemoryBarriers[i].image,
                       info->pImageMemoryBarriers[i].newLayout);
            if (info->pImageMemoryBarriers[i].oldLayout ==
                VK_IMAGE_LAYOUT_UNDEFINED)
                vram::contentInvalidate(info->pImageMemoryBarriers[i].image);
        }

    if (!g_pagerDropAbove || !info || !info->imageMemoryBarrierCount) {
        g_nextCmdPipelineBarrier2KHR(cb, info);
        return;
    }
    std::vector<VkImageMemoryBarrier2> fixed(
        info->pImageMemoryBarriers,
        info->pImageMemoryBarriers + info->imageMemoryBarrierCount);
    for (size_t i = 0; i < fixed.size(); ++i)
        pagerClampRange(fixed[i].image, &fixed[i].subresourceRange);

    VkDependencyInfo info2 = *info;
    info2.pImageMemoryBarriers = fixed.data();
    g_nextCmdPipelineBarrier2KHR(cb, &info2);
}

static VKAPI_ATTR void VKAPI_CALL TAA_CmdBlitImage(
    VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst,
    VkImageLayout dl, uint32_t count, const VkImageBlit *regions, VkFilter filter)
{
    if (!g_nextCmdBlitImage) return;
    uint32_t ds = pagerDropFor(src), dd = pagerDropFor(dst);
    if (!ds && !dd) {
        g_nextCmdBlitImage(cb, src, sl, dst, dl, count, regions, filter);
        return;
    }

    // Mip generation blits level N into level N+1. Where the source level has
    // been removed there is nothing to rebase onto, and the destination it
    // would have filled is a level we removed too, so the blit is dropped.
    std::vector<VkImageBlit> kept;
    for (uint32_t i = 0; i < count; ++i) {
        VkImageBlit b = regions[i];
        if (b.srcSubresource.mipLevel < ds) continue;
        if (b.dstSubresource.mipLevel < dd) continue;
        b.srcSubresource.mipLevel -= ds;
        b.dstSubresource.mipLevel -= dd;
        kept.push_back(b);
    }
    if (!kept.empty())
        g_nextCmdBlitImage(cb, src, sl, dst, dl, (uint32_t)kept.size(),
                           kept.data(), filter);
}

static VKAPI_ATTR void VKAPI_CALL TAA_CmdCopyImage(
    VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst,
    VkImageLayout dl, uint32_t count, const VkImageCopy *regions)
{
    if (!g_nextCmdCopyImage) return;
    uint32_t ds = pagerDropFor(src), dd = pagerDropFor(dst);
    if (!ds && !dd) {
        g_nextCmdCopyImage(cb, src, sl, dst, dl, count, regions);
        return;
    }
    std::vector<VkImageCopy> kept;
    for (uint32_t i = 0; i < count; ++i) {
        VkImageCopy c = regions[i];
        if (c.srcSubresource.mipLevel < ds) continue;
        if (c.dstSubresource.mipLevel < dd) continue;
        c.srcSubresource.mipLevel -= ds;
        c.dstSubresource.mipLevel -= dd;
        kept.push_back(c);
    }
    if (!kept.empty())
        g_nextCmdCopyImage(cb, src, sl, dst, dl, (uint32_t)kept.size(), kept.data());
}

static VKAPI_ATTR void VKAPI_CALL TAA_CmdClearColorImage(
    VkCommandBuffer cb, VkImage img, VkImageLayout layout,
    const VkClearColorValue *colour, uint32_t count,
    const VkImageSubresourceRange *ranges)
{
    if (!g_nextCmdClearColorImage) return;
    if (!pagerDropFor(img) || !ranges) {
        g_nextCmdClearColorImage(cb, img, layout, colour, count, ranges);
        return;
    }
    std::vector<VkImageSubresourceRange> fixed(ranges, ranges + count);
    for (size_t i = 0; i < fixed.size(); ++i) pagerClampRange(img, &fixed[i]);
    g_nextCmdClearColorImage(cb, img, layout, colour, count, fixed.data());
}

static PFN_vkCmdCopyBufferToImage g_nextCmdCopyBufferToImage = nullptr;

static VKAPI_ATTR void VKAPI_CALL TAA_CmdCopyBufferToImage(
    VkCommandBuffer cb, VkBuffer src, VkImage dst, VkImageLayout layout,
    uint32_t count, const VkBufferImageCopy *regions)
{
    if (!g_nextCmdCopyBufferToImage) return;

    // Upload accounting for the VRAM system. 4 bytes per texel is an estimate
    // - BC-compressed textures are 8x smaller - but pacing needs a consistent
    // figure, not an audit, and the governor's budgets are tuned against this
    // same estimate.
    {
        uint64_t bytes = 0;
        for (uint32_t i = 0; i < count && regions; ++i) {
            bytes += (uint64_t)regions[i].imageExtent.width *
                     regions[i].imageExtent.height *
                     (regions[i].imageExtent.depth ? regions[i].imageExtent.depth : 1) * 4ull;
            // Duplicate-upload detection (SS52): same image mip twice in one
            // frame is bandwidth spent on nothing.
            vram::noteUploadRegion((uint64_t)(uintptr_t)dst,
                                   regions[i].imageSubresource.mipLevel);
        }
        // The charge carries the destination's protection class, so the
        // governor can let cockpit/aircraft/infrastructure uploads bypass
        // pacing (upload priority classes, task SS9).
        if (bytes) vram::chargeCopy(cb, bytes, vram::protectionOf(dst));
    }

    uint32_t drop = 0, origW = 0, origH = 0;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<VkImage, TexPolicy>::iterator it = g_texPolicy.find(dst);
        if (it != g_texPolicy.end()) {
            drop  = it->second.dropMips;
            origW = it->second.origW;
            origH = it->second.origH;
        }
    }
    // ---- THE UPLOAD CONTENT CACHE (SS23/24), on the un-remapped path only.
    //
    // Every full-subresource upload to a sampled texture is hashed through
    // the pointer the app's own vkMapMemory returned. Identical bytes going
    // into the identical mip of the SAME image are elided - the texels are
    // already resident and the PCIe transfer buys nothing. New content into a
    // fresh image is compared against the dead-shape memory, which prices the
    // engine's eviction-means-disk-reload behaviour in bytes.
    if (!drop && regions && count && vram::cfg.enable && vram::cfg.uploadCache) {
        uint32_t imgW = 0, imgH = 0, imgFmt = 0; int imgCat = -1;
        {
            std::lock_guard<std::mutex> g(g_lock);
            std::map<VkImage, VramEntry>::iterator it = g_vramImg.find(dst);
            if (it != g_vramImg.end()) {
                imgCat = it->second.cat;
                imgW = it->second.w; imgH = it->second.h;
                imgFmt = it->second.fmt;
            }
        }
        if (imgCat == VRAM_TEX && imgW) {
            std::vector<VkBufferImageCopy> kept;
            bool any = false;
            for (uint32_t i = 0; i < count; ++i) {
                const VkBufferImageCopy &rg = regions[i];
                uint32_t mip = rg.imageSubresource.mipLevel;
                uint32_t mw = imgW >> mip; if (!mw) mw = 1;
                uint32_t mh = imgH >> mip; if (!mh) mh = 1;
                bool whole = rg.imageOffset.x == 0 && rg.imageOffset.y == 0 &&
                             rg.imageOffset.z == 0 &&
                             rg.bufferRowLength == 0 &&
                             rg.bufferImageHeight == 0 &&
                             rg.imageSubresource.baseArrayLayer == 0 &&
                             rg.imageSubresource.layerCount == 1 &&
                             rg.imageExtent.width == mw &&
                             rg.imageExtent.height == mh;
                bool elide = false;
                if (whole) {
                    uint64_t len = (uint64_t)((double)mw * mh *
                                   formatBytesPerPixel((VkFormat)imgFmt));
                    if (len) {
                        const uint8_t *p =
                            vram::bufferBytes(src, rg.bufferOffset, len);
                        if (p)
                            elide = vram::cacheUpload(dst, mip,
                                        vram::contentHash(p, len), len);
                    }
                }
                if (elide) { any = true; continue; }
                kept.push_back(rg);
            }
            if (any) {
                if (!kept.empty())
                    g_nextCmdCopyBufferToImage(cb, src, dst, layout,
                                               (uint32_t)kept.size(),
                                               kept.data());
                return;
            }
        }
    }

    if (!drop || !regions || !count) {
        g_nextCmdCopyBufferToImage(cb, src, dst, layout, count, regions);
        return;
    }

    // TRIPWIRE for a stale policy.
    //
    // The application still believes this texture is its original size, so its
    // region for mip 0 must describe exactly that. If it describes something
    // else, the policy belongs to a different image that happened to occupy
    // this handle earlier, and remapping would silently discard a real mip 0.
    //
    // That is precisely the failure that corrupted scenery, and the reason it
    // took twelve hours and a screenshot to notice is that nothing said a word.
    // Erasing the policy on destroy should make this unreachable; if it ever
    // fires, that assumption is wrong and this is how we find out.
    for (uint32_t i = 0; i < count; ++i) {
        if (regions[i].imageSubresource.mipLevel != 0) continue;
        if (regions[i].imageExtent.width == origW &&
            regions[i].imageExtent.height == origH) break;
        static uint64_t warned = 0;
        if (++warned <= 10)
            trace("PAGER: WARNING stale policy on image %p - upload mip 0 is "
                  "%ux%u but policy recorded %ux%u. Handle reuse; not remapping.",
                  (void*)dst, regions[i].imageExtent.width,
                  regions[i].imageExtent.height, origW, origH);
        g_nextCmdCopyBufferToImage(cb, src, dst, layout, count, regions);
        return;
    }

    // Format for the retention path, once.
    uint32_t dstFmt = 0;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<VkImage, VramEntry>::iterator ve = g_vramImg.find(dst);
        if (ve != g_vramImg.end()) dstFmt = ve->second.fmt;
    }

    std::vector<VkBufferImageCopy> kept;
    kept.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t lvl = regions[i].imageSubresource.mipLevel;
        if (lvl < drop) {
            // The level the pager is discarding: retain its payload. This is
            // the last moment these bytes exist outside the disk, and they
            // are exactly what a future runtime restoration needs.
            const VkBufferImageCopy &rg = regions[i];
            uint32_t mw = origW >> lvl; if (!mw) mw = 1;
            uint32_t mh = origH >> lvl; if (!mh) mh = 1;
            if (dstFmt && rg.bufferRowLength == 0 && rg.bufferImageHeight == 0 &&
                rg.imageSubresource.baseArrayLayer == 0 &&
                rg.imageSubresource.layerCount == 1) {
                uint64_t len = (uint64_t)((double)mw * mh *
                               formatBytesPerPixel((VkFormat)dstFmt));
                const uint8_t *p = len ? vram::bufferBytes(src, rg.bufferOffset,
                                                           len) : nullptr;
                if (p) vram::retainPayload(origW, origH, dstFmt, lvl, p, len);
            }
            continue;                             // this level no longer exists
        }
        VkBufferImageCopy r = regions[i];
        r.imageSubresource.mipLevel = lvl - drop; // renumber what survives
        kept.push_back(r);
    }

    static uint64_t remapped = 0;
    if (++remapped <= 3)
        trace("PAGER: upload remapped - %u regions in, %zu kept (dropped %u levels)",
              count, kept.size(), drop);

    if (!kept.empty())
        g_nextCmdCopyBufferToImage(cb, src, dst, layout,
                                   (uint32_t)kept.size(), kept.data());
}

static VKAPI_ATTR VkResult VKAPI_CALL Layer_CreateImage(
    VkDevice device, const VkImageCreateInfo *ci,
    const VkAllocationCallbacks *alloc, VkImage *out)
{
    PFN_vkCreateImage next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, DeviceData>::iterator it = g_devices.find(dispatchKey(device));
        if (it != g_devices.end()) next = it->second.createImage;
    }
    if (!next) return VK_ERROR_INITIALIZATION_FAILED;

    // Opened BEFORE the pager decision, not after, because the decision is now
    // recorded against whether a flight is running - see the load/flight split
    // below. It is idempotent and latched, so calling it here costs nothing.
    openShare();

    // ---- custom pager: shrink the texture before it is ever created. The
    // global zone policy chooses a base drop; the registry refines it PER
    // RESOURCE - churn-hot shapes are protected from any cut (cutting them is
    // what caused their reload cycling), and large streamed shapes take an
    // extra level under pressure before small ones are touched.
    VkImageCreateInfo ci2 = *ci;
    // One call, one answer. The refinement that used to run here now lives
    // inside pagerDropLevels so the cached decision covers it - applying it
    // afterwards made the same shape resolve to different sizes at different
    // moments, which is exactly what the defragmenter cannot survive. The
    // eligibility guard moved with it: refinement may only touch images that
    // pass the same tests, or it can shrink a render target under load.
    uint32_t drop = pagerDropLevels(ci);
    if (drop) {
        ci2.extent.width  = ci->extent.width  >> drop;
        ci2.extent.height = ci->extent.height >> drop;
        if (ci2.extent.width  < 1) ci2.extent.width  = 1;
        if (ci2.extent.height < 1) ci2.extent.height = 1;
        ci2.mipLevels = ci->mipLevels - drop;
        if (ci2.mipLevels < 1) { ci2.mipLevels = 1; drop = 0; ci2 = *ci; }
    }

    // ---- MAKE X-PLANE'S UPSCALE OUTPUT BLITTABLE.
    //
    // Its FSR writes a storage image, and a storage image need not be a
    // transfer destination - so even once identified, there would be no legal
    // way to put our result into it. Usage can only be chosen at creation, so
    // it is added here.
    //
    // Narrow on purpose: 2D, non-depth, storage, and at least 720p, which is
    // every plausible upscale target and no thumbnail or lookup table. Adding
    // TRANSFER_DST cannot change how the image behaves for its owner; it only
    // widens what is permitted.
    bool addedDst = false;
    if (!isDepthFormat(ci->format) && ci->imageType == VK_IMAGE_TYPE_2D &&
        ci->extent.depth == 1 &&
        (ci->usage & VK_IMAGE_USAGE_STORAGE_BIT) &&
        !(ci->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) &&
        ci->extent.width >= 1280 && ci->extent.height >= 720) {
        if (!drop) ci2 = *ci;
        ci2.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        addedDst = true;
    }

    // ---- NO USAGE PATCH ANY MORE.
    //
    // This used to add STORAGE_BIT to every HDR colour target so the resolve
    // could write the scene in place. It crashed the sim the first time it ran,
    // and it was only needed because the shader wrote its output back into the
    // scene image.
    //
    // The resolve now writes only its own history and the layer copies that
    // into the scene afterwards. X-Plane already creates the target with
    // TRANSFER_DST and SAMPLED (usage 0x17), so nothing needs patching: we read
    // it through a sampler and write it through a copy, both of which its own
    // flags already permit. Removing an interception is worth more than making
    // one work.

    VkResult r = next(device, (drop || addedDst) ? &ci2 : ci, alloc, out);
    if (r != VK_SUCCESS) return r;

    if (addedDst) {
        static uint64_t n = 0;
        if (++n <= 6)
            trace("IMAGE: added TRANSFER_DST to a %ux%u storage image (fmt=%d, "
                  "usage 0x%x -> 0x%x) so our upscaled result can be written "
                  "into it if this turns out to be X-Plane's FSR output",
                  ci->extent.width, ci->extent.height, (int)ci->format,
                  (unsigned)ci->usage, (unsigned)ci2.usage);
    }

    // ---- WRITE THE POLICY FOR EVERY IMAGE, INCLUDING THE UNTOUCHED ONES.
    //
    // This used to be written only when the image was shrunk, which left the
    // undropped ones relying on the destroy hook having erased whatever the
    // last owner of their handle recorded. Vulkan handles are recycled freely
    // and the two events happen on different threads, so a create can land
    // before the matching destroy's erase - and then an untouched full-size
    // image carries a previous texture's drop. Every upload region for it is
    // renumbered and its top level discarded, and a full-size copy lands in a
    // level that is not the one the data describes. Silent, timing-dependent,
    // and invisible to validation because each individual call is legal.
    //
    // An unconditional write makes the handle's policy always the current
    // image's own, so nothing can be inherited and the erase on destroy
    // becomes an optimisation rather than a correctness requirement.
    {
        TexPolicy p;
        p.dropMips = drop;                  // 0 for anything we left alone
        p.origW = ci->extent.width;
        p.origH = ci->extent.height;
        std::lock_guard<std::mutex> g(g_lock);
        g_texPolicy[*out] = p;
    }

    if (drop) {
        std::lock_guard<std::mutex> g(g_lock);
        double before = (double)ci->extent.width * ci->extent.height;
        double after  = (double)ci2.extent.width * ci2.extent.height;
        uint64_t saved = (uint64_t)((before - after)
                       * formatBytesPerPixel(ci->format) * 4.0 / 3.0);
        g_pagerSaved += saved;
        ++g_pagerImages;

        // Attribute it to load or to streaming - see PagerBucket.
        bool inFlight = (g_share && g_share->valid);
        PagerBucket &bk = inFlight ? g_pagerFlight : g_pagerLoad;
        ++bk.images;
        bk.saved += saved;
        if (ci->extent.width >= 4096 || ci->extent.height >= 4096) ++bk.at4096;
        // Format and mip count are logged too. Without them a shrink line says
        // nothing about WHICH texture was affected, and "2048x4096" alone was
        // not enough to tell a runway atlas from a normal map when the ground
        // came out wrong.
        if (g_pagerImages <= 5 || g_pagerImages % 500 == 0)
            trace("PAGER: %ux%u -> %ux%u (dropped %u mip%s) fmt=%d mips %u->%u, "
                  "%llu images, %.1f MB saved",
                  ci->extent.width, ci->extent.height,
                  ci2.extent.width, ci2.extent.height, drop, drop == 1 ? "" : "s",
                  (int)ci->format, ci->mipLevels, ci2.mipLevels,
                  (unsigned long long)g_pagerImages, g_pagerSaved / 1048576.0);
    }

    openShare();

    // ---- THE LEDGER. Every image, by what the driver says it actually costs.
    //
    // Asked of the image we really created, so a paged texture is recorded at
    // its REDUCED size - the opposite of the census below, and deliberately so.
    // The census answers "where should the threshold go", this answers "what is
    // on the card right now", and those need the size before and the size after
    // respectively.
    {
        PFN_vkGetImageMemoryRequirements getReq = nullptr;
        if (g_ledgerOn) {
            std::lock_guard<std::mutex> g(g_lock);
            std::map<void*, DeviceData>::iterator it =
                g_devices.find(dispatchKey(device));
            if (it != g_devices.end()) getReq = it->second.getImageMemReq;
        }
        if (getReq && *out != VK_NULL_HANDLE) {
            VkMemoryRequirements req;
            memset(&req, 0, sizeof(req));
            getReq(device, *out, &req);
            int cat = vramCatOfImage(ci, isDepthFormat(ci->format));
            {
                std::lock_guard<std::mutex> g(g_lock);
                vramAdd(cat, req.size);
                VramEntry e; e.cat = cat; e.bytes = req.size;
                e.w = ci->extent.width; e.h = ci->extent.height;
                e.fmt = (uint32_t)ci->format; e.mips = ci->mipLevels;
                g_vramImg[*out] = e;
            }
            // Registry + churn (task SS2/3): the shape is the identity,
            // because a pager reload creates a NEW handle for the same
            // texture; usage/type/layers feed the classification.
            bool hot = false;
            vram::noteImageCreate(*out, ci->extent.width, ci->extent.height,
                                  (uint32_t)ci->format, ci->mipLevels,
                                  req.size, cat, ci->usage, ci->arrayLayers,
                                  ci->imageType == VK_IMAGE_TYPE_3D,
                                  drop, &hot);
            if (hot) {
                static uint64_t hotSaid = 0;
                if (++hotSaid <= 20)
                    trace("VRAMSYS: %ux%u fmt=%d is churning through residency "
                          "- its blocks now take retention priority 0.65",
                          ci->extent.width, ci->extent.height, (int)ci->format);
            }
        }
    }

    // Census every SAMPLED image - that is what "texture" means here, as
    // opposed to render targets, which the pager does not control.
    if ((ci->usage & VK_IMAGE_USAGE_SAMPLED_BIT) &&
        !(ci->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) &&
        !isDepthFormat(ci->format)) {
        // Mip chains converge on 4/3 of the base level.
        double px = (double)ci->extent.width * ci->extent.height *
                    ci->extent.depth * ci->arrayLayers;
        double bytes = px * formatBytesPerPixel(ci->format);
        if (ci->mipLevels > 1) bytes *= 4.0 / 3.0;

        std::lock_guard<std::mutex> g(g_lock);
        FmtStat &st = g_texCensus[(int)ci->format];
        ++st.count;
        st.bytes += (uint64_t)bytes;
        g_texBytesTotal += (uint64_t)bytes;

        // Remember the contribution so destroy can remove it exactly, rather
        // than recomputing from a VkImageCreateInfo nobody has any more.
        CensusEntry ce;
        ce.format = (int)ci->format;
        ce.bytes  = (uint64_t)bytes;
        // Bucket by the size the APPLICATION asked for, not the size we gave
        // it. The histogram is there to decide where to set the threshold, so
        // it has to describe the textures as X-Plane would have made them -
        // otherwise the pager's own effect hides the thing being measured.
        ce.sizeBucket = sizeBucketOf(ci->extent.width, ci->extent.height);
        g_sizeCount[ce.sizeBucket]++;
        g_sizeBytes[ce.sizeBucket] += ce.bytes;
        g_texCensusOf[*out] = ce;
    }

    if (isDepthFormat(ci->format) && ci->imageType == VK_IMAGE_TYPE_2D &&
        ci->extent.depth == 1)
        noteDepthImage(ci, *out);

    // Colour render targets, recorded with the usage flags they were actually
    // created with. Which one is the 3D scene cannot be known here - that comes
    // from the frame - but the flags can only be read at creation, so they are
    // captured for every candidate and matched up later.
    // STORAGE IMAGES COUNT TOO, and that omission hid the thing we were hunting.
    //
    // X-Plane's FSR writes i_output_texture, which is a storage image and need
    // never be a colour attachment - so it was invisible to this map, and the
    // display-sized search turned up only attachments (usage=0x17, storage=no).
    // We then dropped the real upscale and wrote our result into images that
    // were not it.
    if (!isDepthFormat(ci->format) && ci->imageType == VK_IMAGE_TYPE_2D &&
        ci->extent.depth == 1 &&
        ((ci->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ||
         (ci->usage & VK_IMAGE_USAGE_STORAGE_BIT))) {
        ColorTarget c;
        c.image   = *out;
        c.format  = ci->format;
        c.w       = ci->extent.width;
        c.h       = ci->extent.height;
        c.samples = ci->samples;
        c.arrayLayers = ci->arrayLayers;
        c.usage   = ci->usage;
        std::lock_guard<std::mutex> g(g_lock);
        if (g_colorImages.size() < 256) g_colorImages[*out] = c;
        noteGbufferVelCandidate(c);
    }

    return r;
}

// ---- WHICH QUEUE FAMILY A COMMAND BUFFER WILL RUN ON.
//
// X-Plane creates queues on families 0, 1 AND 2. Every image here is
// SHARING_MODE_EXCLUSIVE, and an exclusive image written on one family and read
// on another - without a release/acquire ownership transfer and a semaphore -
// has UNDEFINED contents for the reader. A pipeline barrier cannot substitute:
// barriers order work within a queue.
//
// That is the shape of this bug exactly. A clear into the presented image is
// clean because a clear does not read - undefined prior contents do not matter
// when you overwrite them. Every blit is corrupt because a blit reads. Every
// parameter is legal, so validation says nothing, and widening the barriers
// changed nothing because barriers were never the mechanism.
//
// A command buffer's family is fixed by the pool it came from, so this is known
// while recording, long before anyone submits it.
static std::map<VkCommandPool, uint32_t>   g_poolFamily;
static std::map<VkCommandBuffer, uint32_t> g_cbFamily;

static VKAPI_ATTR VkResult VKAPI_CALL Layer_CreateCommandPool(
    VkDevice device, const VkCommandPoolCreateInfo *ci,
    const VkAllocationCallbacks *alloc, VkCommandPool *out)
{
    PFN_vkCreateCommandPool next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, DeviceData>::iterator it = g_devices.find(dispatchKey(device));
        if (it != g_devices.end()) next = it->second.createCommandPool;
    }
    if (!next) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult r = next(device, ci, alloc, out);
    if (r == VK_SUCCESS && ci && out) {
        std::lock_guard<std::mutex> g(g_lock);
        g_poolFamily[*out] = ci->queueFamilyIndex;
    }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL Layer_AllocateCommandBuffers(
    VkDevice device, const VkCommandBufferAllocateInfo *ai, VkCommandBuffer *out)
{
    PFN_vkAllocateCommandBuffers next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, DeviceData>::iterator it = g_devices.find(dispatchKey(device));
        if (it != g_devices.end()) next = it->second.allocateCommandBuffers;
    }
    if (!next) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult r = next(device, ai, out);
    if (r == VK_SUCCESS && ai && out) {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<VkCommandPool, uint32_t>::iterator pf =
            g_poolFamily.find(ai->commandPool);
        for (uint32_t i = 0; i < ai->commandBufferCount; ++i) {
            g_cbToDevice[out[i]] = device;
            if (pf != g_poolFamily.end()) g_cbFamily[out[i]] = pf->second;
        }
    }
    return r;
}

// ---------------------------------------------------------------- buffers
//
// Hooked purely to be counted. Nothing here modifies anything - buffers have no
// mip chain, so there is no pager equivalent for them - but they were the one
// category with no accounting whatsoever, which meant the biggest single
// unknown in the VRAM picture was invisible by construction.
//
// X-Plane's scenery meshes live here. Whether that is two hundred megabytes or
// two gigabytes changes what is worth doing next, and nothing in this layer
// could previously tell the difference.
static VKAPI_ATTR VkResult VKAPI_CALL Layer_CreateBuffer(
    VkDevice device, const VkBufferCreateInfo *ci,
    const VkAllocationCallbacks *alloc, VkBuffer *out)
{
    PFN_vkCreateBuffer next = nullptr;
    PFN_vkGetBufferMemoryRequirements getReq = nullptr;
    {
        // Still needs the dispatch pointer even with the ledger off, but the
        // ledger's own work below is skipped.
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, DeviceData>::iterator it = g_devices.find(dispatchKey(device));
        if (it != g_devices.end()) {
            next   = it->second.createBuffer;
            getReq = it->second.getBufferMemReq;
        }
    }
    if (!next) return VK_ERROR_INITIALIZATION_FAILED;

    VkResult r = next(device, ci, alloc, out);
    if (r != VK_SUCCESS || !ci || !out || *out == VK_NULL_HANDLE) return r;

    if (getReq && g_ledgerOn) {
        VkMemoryRequirements req;
        memset(&req, 0, sizeof(req));
        getReq(device, *out, &req);
        int cat = vramCatOfBuffer(ci->usage);
        std::lock_guard<std::mutex> g(g_lock);
        vramAdd(cat, req.size);
        VramEntry e; e.cat = cat; e.bytes = req.size;
        e.w = e.h = e.fmt = e.mips = 0;
        g_vramBuf[*out] = e;

        if (cat == VRAM_BUF_GEOM) {
            int b = geomBucketOf(ci->size);
            ++g_geomCount[b];
            g_geomAsked[b] += ci->size;
            g_geomGot[b]   += req.size;
        }
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL Layer_DestroyBuffer(
    VkDevice device, VkBuffer buf, const VkAllocationCallbacks *alloc)
{
    vram::noteBufferGone(buf);
    PFN_vkDestroyBuffer next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, DeviceData>::iterator it = g_devices.find(dispatchKey(device));
        if (it != g_devices.end()) next = it->second.destroyBuffer;

        std::map<VkBuffer, VramEntry>::iterator ve = g_vramBuf.find(buf);
        if (ve != g_vramBuf.end()) {
            vramRemove(ve->second.cat, ve->second.bytes);
            g_vramBuf.erase(ve);
        }
    }
    if (next) next(device, buf, alloc);
}

static VKAPI_ATTR VkResult VKAPI_CALL Layer_CreateImageView(
    VkDevice device, const VkImageViewCreateInfo *ci,
    const VkAllocationCallbacks *alloc, VkImageView *out)
{
    PFN_vkCreateImageView next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, DeviceData>::iterator it = g_devices.find(dispatchKey(device));
        if (it != g_devices.end()) next = it->second.createImageView;
    }
    if (!next) return VK_ERROR_INITIALIZATION_FAILED;

    // The mip clamp belongs HERE, not in a second hook.
    //
    // vkCreateImageView was already hooked by this function, so the separate
    // clamping version added later was never dispatched - the loader returns
    // whichever function the table names, and it named this one. The clamp was
    // dead code, which is why the run still crashed with the fix "in place" and
    // why no "view clamped" line ever appeared. Duplicating a hook silently
    // disables the newer one.
    VkImageViewCreateInfo ci2;
    const VkImageViewCreateInfo *use = ci;
    if (ci && g_pagerDropAbove && pagerDropFor(ci->image)) {
        ci2 = *ci;
        pagerClampRange(ci->image, &ci2.subresourceRange);
        use = &ci2;
        static uint64_t n = 0;
        if (++n <= 3)
            trace("PAGER: view clamped - base %u->%u count %u->%u",
                  ci->subresourceRange.baseMipLevel, ci2.subresourceRange.baseMipLevel,
                  ci->subresourceRange.levelCount, ci2.subresourceRange.levelCount);
    }

    VkResult r = next(device, use, alloc, out);
    if (r == VK_SUCCESS && ci) {
        std::lock_guard<std::mutex> g(g_lock);
        g_viewToImage[*out] = ci->image;
    }
    return r;
}

// Measured, deliberately reproduced: TAA on + a view-snap hotkey = instant
// DEVICE_LOST. The view change destroys scene-sized images the resolve's
// descriptors still reference. Any scene-sized destruction quiesces the
// resolve for a few frames - a view change costs three TAA-less frames
// instead of the device.
static std::atomic<int> g_taaQuiesce(0);

// ---- THE RESOLVE LIFETIME FIX (the view-snap DEVICE_LOST, root-caused).
// Our resolve's descriptors reference engine images - the scene target and
// gbuffer_vel. The engine's deferred destruction waits for ITS OWN GPU work,
// not ours, so a view change could destroy an image while our dispatch,
// recorded a frame or two earlier, was still in flight. Any image the
// resolve bound within the last four frames has its down-chain destroy
// DEFERRED four presents; the app treats the handle as dead immediately,
// the driver sees the destroy only after our dispatch provably drained.
// X-Plane's allocation callbacks are a process-lifetime static (measured:
// g_vk_allocation_callbacks), so storing the pointer is sound.
static std::map<VkImage, uint64_t> g_taaBoundImgs;          // img -> frame
struct DeferredImgKill {
    VkDevice dev; VkImage img; const VkAllocationCallbacks *alloc;
    uint64_t due;
};
static std::vector<DeferredImgKill> g_deferredImgKills;

static VKAPI_ATTR void VKAPI_CALL Layer_DestroyImage(
    VkDevice device, VkImage img, const VkAllocationCallbacks *alloc)
{
    // MEASURED IN THE CAPTURE: the blanket scene-sized quiesce was firing on
    // every streamed-texture destruction, so the resolve NEVER RAN - frame
    // 3595 contains no uVelocity dispatch at all - and the frames where it
    // lapsed alternated blended output with raw passthrough, which IS the
    // cockpit shake and the motion crawl. The quiesce now fires only when a
    // destroyed image is one the resolve actually bound (the lifetime
    // ledger), which is the only destruction that ever threatened it.
    {
        std::lock_guard<std::mutex> g(g_lock);
        if (g_taaBoundImgs.count(img))
            g_taaQuiesce.store(3);
    }
    vram::noteImageDestroy(img);
    PFN_vkDestroyImage next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, DeviceData>::iterator it = g_devices.find(dispatchKey(device));
        if (it != g_devices.end()) next = it->second.destroyImage;

        // Take this image back out of the ledger.
        std::map<VkImage, VramEntry>::iterator ve = g_vramImg.find(img);
        if (ve != g_vramImg.end()) {
            vramRemove(ve->second.cat, ve->second.bytes);
            g_vramImg.erase(ve);
        }

        // Take this image back out of the census - see CensusEntry.
        std::map<VkImage, CensusEntry>::iterator ce = g_texCensusOf.find(img);
        if (ce != g_texCensusOf.end()) {
            FmtStat &st = g_texCensus[ce->second.format];
            if (st.count) --st.count;
            st.bytes = (st.bytes > ce->second.bytes) ? (st.bytes - ce->second.bytes) : 0;
            g_texBytesTotal = (g_texBytesTotal > ce->second.bytes)
                            ? (g_texBytesTotal - ce->second.bytes) : 0;
            int b = ce->second.sizeBucket;
            if (b >= 0 && b <= 16) {
                if (g_sizeCount[b]) --g_sizeCount[b];
                g_sizeBytes[b] = (g_sizeBytes[b] > ce->second.bytes)
                               ? (g_sizeBytes[b] - ce->second.bytes) : 0;
            }
            g_texCensusOf.erase(ce);
        }

        // The pager's per-image policy dies with the image too. Vulkan reuses
        // handles, so a stale entry would make the next image at this address
        // inherit a mip drop it never had - and that is a corruption bug that
        // would look exactly like a texture problem in the world.
        g_texPolicy.erase(img);

        // The scene depth buffer is recreated on resize. Holding a destroyed
        // handle and later building a view from it would be a use-after-free
        // on the GPU, which does not fail cleanly.
        if (img == g_sceneDepth) {
            g_sceneDepth = VK_NULL_HANDLE;
            g_depthReported = false;
            // The velocity pass holds this image and a view built from it.
            // Both must go before the app frees it, or the next dispatch
            // samples and barriers memory that is no longer there.
            trace("DEPTH: scene depth destroyed - velocity pass will rebuild");
        }
        if (img == g_frameDepthImage) g_frameDepthImage = VK_NULL_HANDLE;


        // COLOUR image lifetime. The resolve holds an image view built from the
        // scene colour target, and X-Plane destroys and recreates that target
        // whenever the render settings change - which is exactly what happened
        // when the antialiasing setting was toggled mid-session: the image went
        // away, our view did not, and the next dispatch read freed memory.
        //
        // The depth path already guarded against this. The colour path did not,
        // because until this stage nothing held a colour view - the velocity
        // pass never touched colour at all. Adding a resource means adding its
        // lifetime, and that is the part it is easy to forget.
        g_colorImages.erase(img);
        if (img == g_sceneColor.image) {
            g_sceneColor = ColorTarget();
            g_sceneColorLast = VK_NULL_HANDLE;
            g_sceneColorStable = 0;
            g_sceneColorReported = false;
        }
        for (size_t q = 0; q < g_hdrTargets.size(); ++q)
            if (g_hdrTargets[q] == img) {
                g_hdrTargets.erase(g_hdrTargets.begin() + q);
                break;
            }

        // Drop the handle from the depth lists and from the reject set.
        //
        // A VkImage handle is only unique while it is alive - the driver is free
        // to hand the same value back for the next image it creates. A stale
        // entry in the reject set would then condemn a brand-new image for the
        // sins of a destroyed one, and the rejection would look like a content
        // verdict rather than a recycled pointer.
        for (size_t i = 0; i < g_depthRejected.size(); ++i)
            if (g_depthRejected[i] == img) {
                g_depthRejected.erase(g_depthRejected.begin() + i);
                break;
            }
        for (size_t i = 0; i < g_frameDepthListDone.size(); ++i)
            if (g_frameDepthListDone[i] == img) {
                g_frameDepthListDone.erase(g_frameDepthListDone.begin() + i);
                break;
            }

        for (std::map<VkImageView, VkImage>::iterator vi = g_viewToImage.begin();
             vi != g_viewToImage.end(); ) {
            if (vi->second == img) vi = g_viewToImage.erase(vi); else ++vi;
        }
        for (size_t i = 0; i < g_depthCandidates.size(); ++i)
            if (g_depthCandidates[i].image == img) {
                g_depthCandidates.erase(g_depthCandidates.begin() + i);
                break;
            }
    }
    // The resolve lifetime fix: an image the resolve bound recently is
    // destroyed LATE, after our in-flight dispatches have provably drained.
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<VkImage, uint64_t>::iterator bi = g_taaBoundImgs.find(img);
        if (bi != g_taaBoundImgs.end()) {
            uint64_t boundAt = bi->second;
            bool recent = g_frameCount - boundAt <= 4;
            g_taaBoundImgs.erase(bi);
            if (recent && next) {
                DeferredImgKill k;
                k.dev = device; k.img = img; k.alloc = alloc;
                k.due = g_frameCount + 4;
                g_deferredImgKills.push_back(k);
                static uint64_t said = 0;
                if (++said <= 8)
                    trace("TAA LIFETIME: image %p destroy deferred 4 presents "
                          "- the resolve referenced it %llu frame(s) ago",
                          (void*)img,
                          (unsigned long long)(g_frameCount - boundAt));
                return;                       // driver sees it later
            }
        }
    }
    if (next) next(device, img, alloc);
}

// Where the 3D scene ends and the UI begins. Knowing this is what lets the
// resolve run before instrument text, ATC boxes and the map get temporally
// smeared. Stage 1 only counts passes per frame and records their attachment
// sizes so the boundary can be identified from data rather than guessed.
static uint32_t g_passesThisFrame = 0;

// HOW MANY PASSES BIND THE VELOCITY TARGET IN ONE FRAME.
//
// isScene is size-plus-depth, nothing more: any full-resolution pass carrying a
// depth attachment qualifies. X-Plane draws 28 passes a frame, and if more than
// one of them matches, the second writes ITS camera's motion over the first's -
// which would explain a field that is sometimes correct, sometimes uniformly
// wrong, and sometimes a mixture, while the matrix stays constant.
static uint32_t g_mvBindsThisFrame = 0;
// Set when the pass currently being recorded on this command buffer has the
// velocity target attached; cleared when that pass ends.
static std::map<VkCommandBuffer, bool> g_cbMvBoundPass;

// ---- THE LIT PASS, NOT THE G-BUFFER PASS.
//
// X-Plane is DEFERRED: the pass carrying the velocity target has colour=5 -
// five attachments of material data. That is the right place to PRODUCE motion
// vectors and the wrong place to consume them, because its colour attachment is
// albedo/normal, not a lit image. Resolving there blends G-buffer data across
// frames and feeds it into lighting, which is the corruption along the top of
// the frame.
//
// Lighting runs afterwards into a single full-size attachment: after shading,
// before tonemap, before the cockpit overlays (which reload depth). That is the
// HDR image TAA is for, and its colour attachment has to be looked up from the
// pass itself - g_sceneColor tracks something else.
struct CbPassInfo {
    uint32_t colorCount = 0;
    uint32_t w = 0, h = 0;
    VkImage  color0 = VK_NULL_HANDLE;
    bool     depthLoad = false;
};
static std::map<VkCommandBuffer, CbPassInfo> g_cbPassInfo;
// Which qualifying pass is currently open, counted from 0 in submission order,
// and how many patched GEOMETRY pipelines get bound inside each. The world pass
// is the one that draws the world; that is a thing to measure, not to guess at
// from attachment counts. -1 means no qualifying pass is open.
static int      g_mvPassOrdinal = -1;
static uint64_t g_mvPassDraws[16] = {0};
// Per-FRAME geometry binds per qualifying pass, and the ordinal chosen from the
// previous frame's counts. The lifetime totals above answer "which pass draws
// the world overall"; this answers it for the view being rendered right now,
// which is the question that matters when the view can change.
static uint64_t g_mvPassDrawsFrame[16] = {0};
static long     g_mvSceneOrdinal = 1;
static uint32_t g_mvQualifyThisFrame = 0;
// THE MATRIX ACTUALLY PUSHED, kept so the dump can compare against it.
//
// The dump had been comparing the field against g_velSnap.reproj - the WORLD
// matrix - while the push may hand a draw g_velSnap.bodyReproj instead. In a
// cockpit view the body-frame matrix goes to most of the screen, so the field
// and the yardstick describe different transforms over most of the frame, and
// p05 flips between them depending on which covers more pixels. Comparing
// against what was really pushed removes the last place the two can disagree
// without either being wrong.
static float g_lastPushed[16];
static uint64_t g_layoutOverlap = 0;
static uint64_t g_drawRepushMissed = 0;
static uint32_t g_pushDistinctThisFrame = 0;
static uint32_t g_pushDistinctMax       = 0;
static uint32_t g_mvBindsMax       = 0;
// The frame on which the velocity target was last BOUND - which, because the
// first bind of a frame carries LOAD_OP_CLEAR, is also the frame on which it
// was last cleared and written. This is the staleness authority for everything
// that consumes the field.
//
// It exists because the resolve's old gate, g_mvBindsThisFrame > 0, is a
// per-frame counter sampled DURING RECORDING - and X-Plane records command
// buffers on worker threads, so the lit pass can be recorded before or after
// the scene pass in wall-clock order regardless of submission order. A counter
// read mid-recording answers "has the scene pass been recorded YET", not "does
// this frame have a live field", and during the frozen episodes it let the
// resolve consume a texture nothing had written for three hundred frames. A
// frame stamp compared with one frame of tolerance answers the real question
// and survives the recording-order race.
static std::atomic<uint64_t> g_mvLastBindFrame(0);
// Frames in a row with no velocity bind, counted at the frame boundary. The
// transitions are the story: onset says the pass identification lost the scene
// pass, recovery says how long the episode ran - and the frozen-field episodes
// this exists to catch ran for ~370 frames while every per-frame log line
// looked individually normal.
static uint32_t g_mvNoBindStreak   = 0;
static uint32_t g_passSizes[32][2];
static uint32_t g_passHasDepth[32];
static uint32_t g_passColorCount[32];

static VKAPI_ATTR void VKAPI_CALL Layer_CmdBeginRenderPass(
    VkCommandBuffer cb, const VkRenderPassBeginInfo *info, VkSubpassContents contents)
{
    if (info && g_passesThisFrame < 32) {
        g_passSizes[g_passesThisFrame][0] = info->renderArea.extent.width;
        g_passSizes[g_passesThisFrame][1] = info->renderArea.extent.height;
        g_passHasDepth[g_passesThisFrame] = 0;   // unknown without the VkRenderPass
    }
    ++g_passesThisFrame;
    if (g_nextCmdBeginRenderPass) g_nextCmdBeginRenderPass(cb, info, contents);
}

// X-Plane 12 uses DYNAMIC RENDERING, not render pass objects.
//
// Measured: vkCmdBeginRenderPass fired exactly zero times across 2760 frames.
// Hooking it and reporting "passes=0" looked like the frame had no structure at
// all, when in fact the whole frame goes through vkCmdBeginRendering.
//
// This is the hook that finds the 3D/UI boundary, which is what decides where
// the resolve can be inserted so instrument text and ATC boxes are never
// temporally smeared. Recording whether each pass has a depth attachment is the
// discriminator: the 3D scene has depth, the 2D overlays do not.
// ---------------------------------------------------------------- jitter
//
// Sub-pixel camera jitter, applied at the VIEWPORT rather than in the
// projection matrix.
//
// This is the mechanism TAA and every upscaler run on: shifting the sample grid
// a fraction of a pixel each frame gives the accumulation new information to
// combine, and that is where the extra detail comes from. Without it the
// history is just the same samples over and over and the result is blur.
//
// Why the viewport and not the matrix. The matrices are published to the layer
// and consumed by the velocity pass, and jitter must NOT appear in them - a
// jittered view-projection produces motion vectors carrying the jitter, so
// every static pixel reads as moving by a fraction of a pixel and the whole
// image shimmers. Offsetting the viewport moves the sample grid while leaving
// the matrices exactly as X-Plane computed them, which makes that entire class
// of bug structurally impossible rather than merely avoided.
//
// Only full-viewport passes that carry depth are jittered. The 2D panel, ATC
// boxes and the map must not be: sub-pixel shift on glyphs reads as wobbling
// text, which is far more objectionable than any aliasing it would fix.
static std::map<VkCommandBuffer, bool> g_cbInScenePass;

// Is this command buffer inside the 3D COCKPIT pass?
//
// Tracked separately from the scene pass because the two want different
// reprojection matrices - see the note at the push site. X-Plane draws the
// cockpit in its own full-size depth pass after the world, so it looks
// identical to a scene pass by shape alone; what distinguishes it is that it
// comes after the world passes in the same frame.
static std::map<VkCommandBuffer, bool> g_cbInCockpitPass;
static uint64_t g_bodyReprojPushes = 0;

// WHICH scene pass is the cockpit, counted from 1 within a command buffer.
//
// Set from a MEASUREMENT, not a guess. X-Plane draws the 3D cockpit in its own
// full-size depth pass, which by shape alone is indistinguishable from a world
// pass - same extent, same depth attachment. The only thing separating them is
// their order in the frame, and that order is a property of this sim version
// and this aircraft, not something derivable from the Vulkan calls.
//
// So it defaults to OFF - every draw keeps the world-frame matrix, exactly as
// before - and TAA_COCKPIT_PASS=<n> turns it on once the CB dump has said which
// index the cockpit actually is. Shipping a guess here would put body-frame
// motion vectors on the world, which is a far worse error than leaving the
// cockpit on the world frame where it already is.
static int g_cockpitPassIndex = -1;

// Has this command buffer recorded a 3D pass yet, and has the resolve already
// gone in? Both are per-command-buffer, and both are reset when the buffer is
// closed, so a reused buffer starts clean rather than inheriting a decision
// from whatever it was last used for.
static std::map<VkCommandBuffer, bool> g_cbSawScenePass;
static std::map<VkCommandBuffer, bool> g_cbResolvedThisCb;

// How many full-viewport depth passes this command buffer has recorded, and how
// many it recorded last time round.
//
// "The first depth-less full-viewport pass after ANY scene pass" is too weak a
// test for the end of the 3D scene. X-Plane runs full-viewport passes without
// depth in the MIDDLE of the scene as well as after it, and resolving at the
// first one copied a half-drawn frame: the result was roughly half the screen
// correct and the rest horizontal bands of stale content.
//
// Waiting until this buffer has recorded as many scene passes as it did last
// frame puts the resolve after the last one instead of the first. It is derived
// from the frame rather than from a constant, so it survives a settings change
// altering the pass count - which a hardcoded index would not.
static std::map<VkCommandBuffer, uint32_t> g_cbScenePassCount;
static std::map<VkCommandBuffer, uint32_t> g_cbScenePassPrev;

// The most scene passes this command buffer has ever recorded. Resolving before
// the last of them means the remainder overdraw the result - see the note at the
// relaxed gate.
static std::map<VkCommandBuffer, uint32_t> g_cbScenePassHigh;

// Per-command-buffer pass log, for working out WHERE the resolve can safely go.
//
// The question that has to be answered from data rather than reasoning: is the
// 3D scene rendered entirely within one command buffer, and is the UI recorded
// in that same buffer or a later one? Those two facts decide whether the
// resolve belongs at the 3D/UI boundary inside a buffer, at the end of the
// buffer that drew the scene, or somewhere else entirely - and guessing has
// already produced a half-drawn frame and then a flickering one.
//
// Each entry: 'S' full-viewport with depth (scene), 'P' full-viewport without
// depth (post-process or UI), 'o' anything else (shadows, reflections, small).
// ------------------------------------------------- THE RENDER RESOLUTION
//
// X-Plane does not necessarily render its 3D scene at the display resolution,
// and every part of this layer used to assume it did.
//
// With the sim's own FSR 1.0 enabled - renopt_FSR_04, which the settings UI
// calls "Rendering Resolution (FSR Supersampling)" - the scene is drawn at a
// fraction of the window and spatially upscaled at the end. Measured on a 4K
// display at quality 3: passes at 2953x1661, which is 0.769, exactly FSR's
// Ultra Quality 1.3x ratio.
//
// The consequence was total. The scene-pass test compared renderArea against
// the DISPLAY size, so with FSR on it never matched a single pass - and that
// test gates the depth image, the jitter, the resolve boundary and the
// velocity target. The whole pipeline was searching for a pass that no longer
// existed at that size, and reported nothing wrong because "no scene pass this
// frame" is indistinguishable from "not in the scene pass yet".
//
// So the render resolution is LEARNED rather than assumed: the largest
// depth-carrying pass that is a substantial fraction of the display. Shadow
// cascades and reflection probes also carry depth, at their own much smaller
// sizes, which is what the fraction excludes - and it has to be a fraction
// rather than an equality test precisely because the answer is not known in
// advance.
// ---- WHERE THE FRAME ACTUALLY GOES.
//
// The frame rate has sat at 38 through native 4K, a 1440p sub-native render,
// and frame generation on and off. Something fixed dominates, and until now
// nothing in this project could say what: CPU-side timing measures when work is
// RECORDED, not when the GPU runs it, so it would have reported our passes as
// free no matter what they cost.
//
// Vulkan timestamps are the only honest answer. A timestamp written into the
// command buffer is resolved by the GPU when it reaches that point, so the
// difference between two of them is real device time.
//
// Results are read one frame late and without waiting - blocking on them would
// change the thing being measured.
struct GpuSpan {
    const char *name;
    double      ms;
};

static VkQueryPool g_tsPool       = VK_NULL_HANDLE;
static uint32_t    g_tsCount      = 0;      // timestamps written this frame
static uint32_t    g_tsCapacity   = 0;
static bool        g_tsPending    = false;  // a frame is in flight to read back
static float       g_tsPeriodNs   = 1.0f;   // nanoseconds per tick, from limits
static const char *g_tsNames[16];
static bool        g_gpuTiming    = false;

static bool gpuTimingOn()
{
    static int on = -1;
    if (on < 0) { const char *e = getenv("TAA_GPU_TIMING"); on = (e && atoi(e)) ? 1 : 0; }
    return on != 0;
}

// Begin a frame's timing. Resets the pool and clears the labels.
static void gpuTimeBegin(DeviceData &dd, VkCommandBuffer cb)
{
    if (!g_gpuTiming || g_tsPool == VK_NULL_HANDLE || !dd.cmdResetQueryPool) return;
    dd.cmdResetQueryPool(cb, g_tsPool, 0, g_tsCapacity);
    g_tsCount = 0;
}

// Mark a point. Pairs are formed by consecutive marks, so a span is
// gpuTimeMark("upscale") ... gpuTimeMark("upscale end").
static void gpuTimeMark(DeviceData &dd, VkCommandBuffer cb, const char *label)
{
    if (!g_gpuTiming || g_tsPool == VK_NULL_HANDLE || !dd.cmdWriteTimestamp) return;
    if (g_tsCount >= g_tsCapacity) return;
    g_tsNames[g_tsCount] = label;
    dd.cmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         g_tsPool, g_tsCount);
    ++g_tsCount;
}

// Read back last frame's marks and report. Never waits.
static void gpuTimeReport(DeviceData &dd, uint64_t frames)
{
    if (!g_gpuTiming || g_tsPool == VK_NULL_HANDLE || !dd.getQueryPoolResults) return;
    if (!g_tsPending || g_tsCount < 2) return;
    if (frames % 300 != 0) return;

    uint64_t vals[16];
    VkResult r = dd.getQueryPoolResults(dd.device, g_tsPool, 0, g_tsCount,
                                        sizeof(vals), vals, sizeof(uint64_t),
                                        VK_QUERY_RESULT_64_BIT);
    if (r != VK_SUCCESS) return;   // not ready; try again in 300 frames

    char line[512];
    int  n = snprintf(line, sizeof(line), "GPU TIME:");
    double total = (double)(vals[g_tsCount - 1] - vals[0]) * g_tsPeriodNs / 1e6;
    for (uint32_t i = 0; i + 1 < g_tsCount && n < (int)sizeof(line) - 40; ++i) {
        double ms = (double)(vals[i + 1] - vals[i]) * g_tsPeriodNs / 1e6;
        n += snprintf(line + n, sizeof(line) - n, "  %s=%.2fms", g_tsNames[i], ms);
    }
    snprintf(line + n, sizeof(line) - n, "  | ours total %.2fms", total);
    trace("%s", line);
}

static uint32_t g_renderW = 0, g_renderH = 0;
static uint32_t g_sceneColourCount = 0;
// The latched shape of the velocity-bound pass (see MV STICKY below): the
// colour count the world pass had when first bound at the current target
// size. 0 = unlatched. Reset when the target resizes or the bind starves.
static uint32_t g_mvStickyColour = 0;
static uint32_t g_mvStickyW      = 0;

static bool isSceneSized(uint32_t w, uint32_t h, uint32_t colourCount = 1)
{
    if (!g_share) return false;
    uint32_t dw = (uint32_t)g_share->viewportW, dh = (uint32_t)g_share->viewportH;
    if (!dw || !dh) return false;

    // Half the display in each axis. FSR's most aggressive published ratio is
    // 2.0x - half linear - so anything smaller than that is not the scene.
    if (w * 2 < dw || h * 2 < dh) return false;
    if (w > dw || h > dh) return false;

    // ---- ASPECT RATIO, which is what the size test alone got wrong.
    //
    // A 2048x2048 shadow cascade passes every size test above: it is more than
    // half the display in both axes and smaller than the display in both. The
    // first version latched onto exactly that, reported "scene passes are
    // 2048x2048 (0.533x)", and the real colour target was then never matched -
    // so the resolve found no scene target at all and disabled itself. The
    // image came out untouched, which looks like a fixed shake and is actually
    // a dead pipeline.
    //
    // Scaling the render preserves the window's aspect: 2953x1661 is 1.778 and
    // so is 3840x2160. A shadow map is square. That is the discriminator, and
    // it is independent of WHAT ratio the sim chose, which is the whole point -
    // the size is unknown in advance but the shape is not.
    double aPass = (double)w / (double)h;
    double aDisp = (double)dw / (double)dh;
    if (aPass < aDisp * 0.97 || aPass > aDisp * 1.03) return false;

    // ---- THE G-BUFFER IS THE SCENE, AND SIZE ALONE DOES NOT FIND IT.
    //
    // Picking the LARGEST depth-carrying pass looked reasonable and was wrong.
    // A frame contains both of these:
    //
    //     pass shape 3840x2160 colour=1 depth=yes
    //     pass shape 2953x1661 colour=5 depth=yes
    //
    // The second is the 3D scene - five colour attachments plus depth is a
    // G-buffer, and nothing else in the frame looks like that. The first is
    // some full-window pass that merely happens to be bigger, and because it
    // was bigger it won, so the real scene pass never matched again. Every
    // consequence followed from that: no jitter applied, FSR2's context never
    // created, and the resolve latching onto a post-tonemap LDR target.
    //
    // Attachment count is the discriminator. A deferred renderer's geometry
    // pass writes several targets at once; shadow maps write depth only,
    // post-process passes write one, and the UI writes one. So the pass with
    // the MOST colour attachments wins, and size only breaks ties.
    bool better = (colourCount > g_sceneColourCount) ||
                  (colourCount == g_sceneColourCount &&
                   (w > g_renderW || h > g_renderH));
    if (better) {
        bool first = (g_renderW == 0);
        g_sceneColourCount = colourCount;
        g_renderW = w; g_renderH = h;

        // Hand the measured size back to the plugin. It sizes the jitter
        // sequence off this rather than off its own quality setting, which
        // does not know about X-Plane's FSR - see measRenderW in share.h.
        g_share->measRenderW  = w;  g_share->measRenderH  = h;
        // PUBLISH THE RATIO THE UPSCALER ACTUALLY RUNS AT, not the window's.
        //
        // The plugin sizes the jitter sequence from display/render, because
        // FSR2's accumulation assumes 8*(display/render)^2 phases. FSR2 is now
        // pinned 1:1 - its result is copied back into the render-sized scene
        // target, so there is nowhere for an upscaled image to go - and
        // publishing the 3840x2160 window here made the plugin ask for 14
        // phases against the 8 that FSR2 assumes at 1:1. A longer sequence than
        // the accumulator expects never converges: it is spreading samples over
        // phases FSR2 is not counting on.
        //
        // When there is a display-sized target to present from, this goes back
        // to dw/dh and the phase count follows automatically.
        g_share->measDisplayW = w;  g_share->measDisplayH = h;
        (void)dw; (void)dh;

        if (first || w != dw) {
            trace("RENDER RESOLUTION: scene passes are %ux%u (%u colour "
                  "attachments) against a %ux%u display (%.3fx). %s",
                  w, h, colourCount, dw, dh, (double)w / (double)dw,
                  (w == dw)
                      ? "Native - X-Plane's own FSR is off."
                      : "SUB-NATIVE: X-Plane's FSR is rendering smaller and "
                        "upscaling. Our velocity, depth and resolve must follow "
                        "THIS size, not the window.");
        }
    }
    return (w == g_renderW && h == g_renderH);
}

struct CbPassLog { std::string seq; uint64_t frame = 0; };
static std::map<VkCommandBuffer, CbPassLog> g_cbPassLog;
static bool     g_cbDumpOn = false;
static uint64_t g_cbDumpsLeft = 0;

static bool g_jitterArmed = false;
// Amplitude, in fractions of the full +/-0.5 px Halton offset. Computed once
// per frame at present time - the push path runs millions of times a frame and
// must never take the live-file mutex - and it FOLLOWS THE RESOLVE, because
// either alone is a downgrade: jitter with no consumer is deliberate edge
// crawl, a consumer with no jitter averages identical samples and cannot
// antialias anything.
static float g_jitterScale = 0.0f;
// The NDC offsets actually pushed this frame - what the resolve must be told,
// as opposed to g_velSnap.jitterX/Y, which is the plugin's REQUEST in units of
// pixels. The resolve was told the request for a while, believing it was NDC:
// three orders of magnitude and one axis convention wrong, which is why the
// value is stored at the single site that knows what was applied.
static float g_appliedJitX = 0.0f, g_appliedJitY = 0.0f;
// Reads the jitter out of THIS command buffer's pending-push slot - defined
// below the slot machinery it needs, used above it by the resolve.
static bool mvPendingJitter(VkCommandBuffer cb, float *jx, float *jy);
// Velocity-target clear value meaning "no patched shader wrote this pixel".
// RGBA16F maxes near 65504, so this saturates to -inf/-65504 and stays
// unmistakable; the shader tests with a threshold, never equality.
static const float kMvUnwritten = -60000.0f;
static std::atomic<uint64_t> g_jitSlotHit(0);
static std::atomic<uint64_t> g_jitSlotMiss(0);
static std::atomic<uint64_t> g_jitZero(0);
static std::atomic<uint64_t> g_jitNonZero(0);
static float g_jitLastX = 0.0f, g_jitLastY = 0.0f;
static uint64_t g_jitterApplied = 0;

// Did a resolve actually happen this frame, and how often does it not.
//
// A resolve that runs on only some frames is invisible in the logs and obvious
// on screen: the frames it skips are passed through with the jitter still in
// them, so the picture shakes. Counting the misses is the difference between
// diagnosing that in one run and guessing at it across six.
static bool     g_resolveRanThisFrame = false;
static uint64_t g_resolveMissFrames = 0;
static uint64_t g_resolveOkFrames   = 0;
static bool     g_resolveRelaxed    = false;

// Does clip Y point the opposite way to framebuffer Y.
//
// Read from the viewport X-Plane actually sets rather than assumed, and used by
// the jitter push to decide which way "down the screen" is in clip space.
static bool g_viewportYFlipped = false;

// LEGACY VIEWPORT JITTER, off by default.
//
// This is where jitter used to be applied, and TAA_JITTER_VIEWPORT=1 brings it
// back for a direct A/B against the vertex-shader version. Kept because the two
// should look IDENTICAL on geometry and differ only on the full-screen passes -
// so if the shake survives the move, that comparison says whether the diagnosis
// was wrong rather than leaving it to be re-argued.
static bool g_jitterViewport = false;

// Metres. Anything closer than this reprojects as body-fixed rather than
// world-fixed. A 777 panel sits about 0.6-1.2 m from the eye and the windscreen
// frame under 2 m, while the nearest outside geometry during a landing is the
// runway several metres away - so the threshold has real headroom on both
// sides. Overridable because that headroom is aircraft-dependent.
// OFF by default. Zero disables the select entirely - gl_Position.w is positive
// for anything in front of the camera, so `w < 0` is never true.
//
// The idea was sound and the discriminator is not: give anything nearer than
// this a velocity of zero because it is bolted to the airframe. Depth cannot
// tell that. Flying low over a city puts building ROOFS inside the window, and
// they are world geometry passing at approach speed - so they get history
// pinned at the moment they are moving fastest. That is the roof going black
// for a frame when passing close over it, and it is why the flicker gets worse
// the closer you are to landing.
//
// The premise it was built on still holds - cockpit geometry rigid in a rigid
// airframe genuinely has zero velocity, and the body frame is now measured to
// 0.0022 m/frame - but the cockpit has to be IDENTIFIED, not inferred from
// distance. TAA_NEARFIELD_M re-enables it for experiments.
// 2.0 m by default, not zero. Zero kept the cockpit fix disabled through
// every session in which the shake was reported; two metres covers the panel
// and the window frame while leaving the runway (below and ahead of the
// glareshield at more than that) on the world path. Live: taa.nearfield_m.
static float g_nearFieldM = 2.0f;

static VKAPI_ATTR void VKAPI_CALL Layer_CmdSetViewport(
    VkCommandBuffer cb, uint32_t first, uint32_t count, const VkViewport *vp)
{
    if (!g_nextCmdSetViewport) return;

    // LATCH THE Y FLIP FROM THE SCENE VIEWPORT ONLY.
    //
    // This used to take the flip from EVERY viewport X-Plane set - shadow
    // cascades, UI, offscreen composites, all of it, last write wins. The
    // clip-space jitter multiplies its Y by that flag, so the sign of the
    // jitter depended on whichever unrelated pass happened to set a viewport
    // most recently before the draw. A jitter whose sign changes between frames
    // is not jitter, it is noise: FSR2 is told one sign and shown another, and
    // the accumulation it builds is being fed contradictory samples. That is
    // shimmer on movement and an image that will not sit still, from a
    // one-line detail with no logging on it.
    //
    // Only the pass we actually jitter gets a vote, and it is stated once.
    // ---- EVERY DISTINCT VIEWPORT, NOT JUST THE ONE THAT MATCHES.
    //
    // The latch below only votes when the viewport equals the render target, so
    // a pass drawing at a different size into the SAME velocity attachment is
    // invisible to it - and that is now the leading suspect. With the identity
    // matrix pushed the field is exactly zero everywhere, so the band is
    // produced by the matrix; the matrix rebuilds view space with 1/sx and 1/sy
    // from the published projection; and the band has vx EXACTLY zero with vy
    // proportional to y, which is what a matching sx and a differing sy do.
    //
    // A different viewport height is one way to get a differing sy. This says
    // whether that is happening rather than leaving it inferred.
    if (vp && count > 0 && vp[0].height != 0.0f) {
        struct VpSeen { float w, h; uint64_t n; };
        static VpSeen seen[8];
        static int nSeen = 0;
        bool found = false;
        for (int i = 0; i < nSeen; ++i)
            if (seen[i].w == vp[0].width && seen[i].h == vp[0].height) {
                ++seen[i].n; found = true; break;
            }
        if (!found && nSeen < 8) {
            seen[nSeen].w = vp[0].width;
            seen[nSeen].h = vp[0].height;
            seen[nSeen].n = 1;
            ++nSeen;
            char buf[512]; buf[0] = 0;
            for (int i = 0; i < nSeen; ++i) {
                char one[64];
                snprintf(one, sizeof(one), "%.0fx%.0f ", seen[i].w, seen[i].h);
                if (strlen(buf) + strlen(one) < sizeof(buf) - 1) strcat(buf, one);
            }
            trace("VIEWPORT CENSUS: %d distinct so far - %s(render target is "
                  "%ux%u). A height of the opposite sign or a different "
                  "magnitude means a different sy for those draws.",
                  nSeen, buf, g_renderW, g_renderH);
        }
    }

    if (vp && count > 0 && vp[0].height != 0.0f) {
        uint32_t vw = (uint32_t)fabsf(vp[0].width);
        uint32_t vh = (uint32_t)fabsf(vp[0].height);
        if (g_renderW && vw == g_renderW && vh == g_renderH) {
            bool flip = (vp[0].height < 0.0f);
            static int said = -1;
            if (said != (int)flip) {
                said = (int)flip;
                trace("JITTER: scene viewport is %s (height %.1f) - clip-space "
                      "jitter Y sign is %s. This is latched from the SCENE pass "
                      "only; it used to follow whatever pass set a viewport last.",
                      flip ? "Y-FLIPPED" : "not Y-flipped", vp[0].height,
                      flip ? "-1" : "+1");
            }
            g_viewportYFlipped = flip;
        }
    }

    // Jitter only while something is consuming it.
    //
    // On its own it makes the image worse: it shifts the sample grid every
    // frame and, with nothing accumulating the result, high-contrast edges

    bool scene = false;
    if (g_jitterViewport && g_jitterArmed && vp && count > 0) {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<VkCommandBuffer, bool>::iterator it = g_cbInScenePass.find(cb);
        scene = (it != g_cbInScenePass.end() && it->second);
    }

    // VIEWPORT Y SIGN, reported once.
    //
    // A negative height is VK_KHR_maintenance1 being used to keep OpenGL's
    // orientation, which X-Plane has every reason to do given its history. It
    // decides whether NDC derived from gl_FragCoord matches what a vertex
    // shader emitted - and if it does not, motion vectors recovered in a
    // fragment shader come out Y-flipped. That is a silent sign error, so it is
    // worth knowing from the API rather than from the look of the result.
    if (vp && count > 0) {
        static bool reported = false;
        if (!reported) {
            reported = true;
            trace("VIEWPORT: x=%.1f y=%.1f w=%.1f h=%.1f -> %s",
                  vp[0].x, vp[0].y, vp[0].width, vp[0].height,
                  vp[0].height < 0.0f
                      ? "NEGATIVE height: GL orientation via maintenance1. "
                        "gl_FragCoord-derived NDC is FLIPPED relative to the "
                        "vertex shader's clip position."
                      : "positive height: Vulkan orientation, no flip.");
        }
    }

    if (!scene) { g_nextCmdSetViewport(cb, first, count, vp); return; }

    float jx = g_velSnap.jitterX, jy = g_velSnap.jitterY;
    if (jx == 0.0f && jy == 0.0f) { g_nextCmdSetViewport(cb, first, count, vp); return; }

    VkViewport tmp[8];
    if (count > 8) { g_nextCmdSetViewport(cb, first, count, vp); return; }

    for (uint32_t i = 0; i < count; ++i) {
        tmp[i] = vp[i];
        // Only the full-size viewport. A pass can set a smaller one for a
        // sub-region, and shifting that would move the region rather than
        // jitter the samples within it.
        // Compared against the RENDER resolution, which is not the window when
        // X-Plane's FSR is on. This is the legacy viewport-jitter path and is
        // off by default, but a size test that silently never matches is worse
        // than one that is simply disabled.
        uint32_t rw = g_renderW ? g_renderW : (uint32_t)g_velSnap.viewportW;
        uint32_t rh = g_renderH ? g_renderH : (uint32_t)g_velSnap.viewportH;
        if ((uint32_t)fabsf(tmp[i].width)  == rw &&
            (uint32_t)fabsf(tmp[i].height) == rh) {
            tmp[i].x += jx;
            tmp[i].y += jy;
            ++g_jitterApplied;
        }
    }
    g_nextCmdSetViewport(cb, first, count, tmp);
}

// Appends the velocity attachment slot to a rendering pass.
//
// EVERY pass gets the slot, not just the scene. Under dynamic rendering a
// pipeline's colour formats must agree with the pass, and X-Plane reuses
// pipelines across passes - so a slot that appeared only in the scene pass
// would make every one of those pipelines invalid everywhere else. With
// VK_EXT_dynamic_rendering_unused_attachments the slot may be null, so
// non-scene passes carry an empty one at no cost.
//
// The scene pass - full viewport, with depth - gets the real image. The first
// such pass of a frame CLEARS it; later ones LOAD, so a scene split across
// several passes accumulates instead of each pass wiping the last.
// Swapchain images, recorded from vkGetSwapchainImagesKHR. Declared up here
// because the scene-target selection needs to ask whether a pass is drawing
// straight into the presented image, and that runs long before present.
static std::map<VkSwapchainKHR, std::vector<VkImage> > g_swapImages;

// ---- THE SIZE AND FORMAT OF WHAT WE PRESENT INTO, WHICH WE NEVER READ.
//
// Tracking the image handles was never enough. Every other image the layer
// touches is recorded at vkCreateImage with its extent and format; swapchain
// images are handed back already built, so they carried neither, and the
// delivery blit filled the gap by assuming they matched the upscale output.
// That assumption printed as "dst fmt=-1 0x0" the first time anything asked.
//
// The create info has both. Keep them.
struct SwapInfo { uint32_t w, h; VkFormat format; };
static std::map<VkSwapchainKHR, SwapInfo> g_swapInfo;

// Whether the presented images were actually created with TRANSFER_DST usage.
// The delivery blit is only a legal operation when they were.
static bool g_swapTransferDst = false;

// The extent of the swapchain a given presented image belongs to.
static bool swapInfoFor(VkImage img, SwapInfo &out)
{
    for (std::map<VkSwapchainKHR, std::vector<VkImage> >::iterator
             si = g_swapImages.begin(); si != g_swapImages.end(); ++si) {
        for (size_t k = 0; k < si->second.size(); ++k) {
            if (si->second[k] != img) continue;
            std::map<VkSwapchainKHR, SwapInfo>::iterator ii =
                g_swapInfo.find(si->first);
            if (ii == g_swapInfo.end()) return false;
            out = ii->second;
            return true;
        }
    }
    return false;
}
static std::map<VkDescriptorSet, std::vector<VkImageView> > g_setViews;
// The same, for STORAGE images. g_setViews holds only sampled images because
// it answers "what does the composite read"; a compute shader's OUTPUT is a
// storage image and is filtered out of it. Separate rather than merged: the
// composite search depends on g_setViews meaning exactly what it means today.
static std::map<VkDescriptorSet, std::vector<VkImageView> > g_setStorageViews;
static std::map<VkCommandBuffer, bool> g_cbInSwapPass;

// Every image that has ever been a render-pass colour attachment.
static std::set<VkImage> g_seenAsAttachment;

// ---- THE COMPUTE COMPOSITE, WHICH NO ATTACHMENT SEARCH COULD EVER FIND.
//
// Bisecting the shotgun landed on a full-window RGBA16F image that appears in
// NO pass whatsoever. X-Plane composites its final frame with a COMPUTE shader
// writing a storage image, and the swapchain pass then samples that. Every
// selection rule this project has used - last full-viewport target, last HDR
// one, the sRGB one, prefer-not-multisampled - searched render-pass colour
// attachments, so the one image that matters was invisible to all of them.
//
// It also explains the two "HDR scene targets" we did find: those are real, and
// X-Plane ping-pongs them, but they feed the compute pass rather than the
// screen. Writing to both changed nothing for exactly that reason.
//
// The signature is distinctive: full window, HDR float, single sample, STORAGE
// usage (a compute shader writes it), and never once a colour attachment.
static VkImage g_computeComposite = VK_NULL_HANDLE;

// Display-sized images that can be a transfer destination - where X-Plane's
// dropped upscale dispatch would have written, and therefore where ours goes.
static std::vector<VkImage> g_upscaleTargets;
// Defined below, beside the barrier helper it needs. Declared here because the
// upscaled result is written from the FSR2 dispatch site, which comes first.


// Which swapchain image, if any, the pass currently recording into this command
// buffer is writing. Cleared when the pass ends.
struct CbSwapTarget { VkImage image; VkImageLayout layout; };
static std::map<VkCommandBuffer, CbSwapTarget> g_cbSwapTarget;

static bool isSwapImage(VkImage img)
{
    for (std::map<VkSwapchainKHR, std::vector<VkImage> >::iterator
             si = g_swapImages.begin(); si != g_swapImages.end(); ++si)
        for (size_t k = 0; k < si->second.size(); ++k)
            if (si->second[k] == img) return true;
    return false;
}


static bool mvAppendAttachment(const VkRenderingInfo *info,
                               std::vector<VkRenderingAttachmentInfo> &atts,
                               VkRenderingInfo &out)
{
    if (!g_spirvLive || !info) return false;

    // Matched against the VELOCITY IMAGE's own size, not the plugin's viewport.
    //
    // g_velSnap.viewportW is the window size the plugin reports. The scene is
    // rendered at the scene target's size, which is what the velocity image was
    // built from - and on this machine those differ, so the comparison never
    // succeeded and the image was never bound. Every pass took the null slot,
    // which is why the sim rendered perfectly and produced no velocity at all.
    //
    // Comparing against g_mv is self-consistent by construction: the image is
    // sized from the scene target, so a pass at that size IS the scene pass.
    bool fullViewport = g_mv.w > 0 &&
        info->renderArea.extent.width  == g_mv.w &&
        info->renderArea.extent.height == g_mv.h;
    bool hasDepth = info->pDepthAttachment &&
                    info->pDepthAttachment->imageView != VK_NULL_HANDLE;
    bool isScene  = fullViewport && hasDepth && g_mv.ready;

    // ---- ONLY THE FIRST QUALIFYING PASS.
    //
    // isScene is size-plus-depth, and THIRTEEN passes a frame satisfy it. They
    // are not all the 3D world: X-Plane composites panel and instrument content
    // at full resolution with depth too, and geometry drawn from a vertex
    // buffer in screen space reprojects to nonsense exactly as a full-screen
    // triangle does. Because the attachment is LOAD, whichever of the thirteen
    // draws last owns the pixel - which is why the same configuration produced
    // p05/far of 1.001 on one dump and 21x on the next.
    //
    // The arithmetic rules out parallax as the explanation: a median of 813 px
    // would need geometry 1.5 cm from the eye, and the matrix's own near-plane
    // figure implies a translation of about 4 mm, which puts the panel at ~17
    // px. Nothing in the cockpit can move that far.
    //
    // The main scene pass is the FIRST to qualify - the world is drawn before
    // anything is composited over it. Set TAA_MV_ALL_PASSES to go back to
    // binding all thirteen and watch the field come apart again.
    // WHICH of the thirteen is the world pass has to be measured, not guessed.
    // Binding only the first gave three dumps of exact zeros - the first to
    // qualify writes nothing, so it is a prepass. TAA_MV_PASS pins a specific
    // index once the census below says which one draws the world.
    // PASS 1, MEASURED. The census settles which of the thirteen draws the
    // world, and it is not close:
    //
    //   geometry binds per qualifying pass - [0]=108 [1]=57132 [2]=540
    //     [3]=540 [4]=216 [6]=432 [7]=540 [8]=108 [9]=216 [11]=540
    //
    // Pass 1 carries two orders of magnitude more geometry than any other. Pass
    // 0 is a prepass with 108 binds - pinning to it produced dumps of exact
    // zeros, which is how that guess was caught.
    //
    // DEFAULTED, not env-gated. An env-only switch is what left the SPIR-V
    // injection dead for a whole session in this project: the code was right
    // and nothing set the variable. TAA_MV_PASS overrides, and -1 restores
    // binding every qualifying pass so the census can be re-run.
    // ---- THE WORLD PASS IS FOUND, NOT PINNED.
    //
    // This was pinned to qualifying-pass ordinal 1, a number measured once by a
    // census taken in the COCKPIT. Pass ordinals are counted per frame in
    // submission order, and an external view composes its frame differently -
    // there is no 3D cockpit pass - so ordinal 1 there is a different pass
    // entirely. The world pass then never receives the velocity attachment, the
    // target keeps its cleared zeros, and the field reports no motion at all.
    //
    // Measured: in the cockpit the field is accurate to 0.00002 of each pixel's
    // own motion; in an external view the residual is 89 to 783 px with NO pixel
    // carrying even one pixel of measured flow. That is not a wrong
    // reprojection - it is an absent one.
    //
    // So identify the pass by what it is. The world pass is the one that draws
    // the overwhelming majority of the geometry - the census that produced the
    // "1" showed it carrying two orders of magnitude more binds than any other
    // ([1]=57132 against [2]=540). That property holds in every view; the
    // ordinal does not.
    //
    // The choice uses the PREVIOUS frame's counts, because the current frame's
    // are still being accumulated when the first pass has to be decided. Passes
    // are stable frame to frame, so a one-frame-old census is the right answer
    // for this frame.
    // LIVE, not env-only. Which passes carry velocity is the single control
    // that decides whether terrain smears, and an env variable can only be
    // judged by relaunching - which is how the wrong answer survived: the
    // "binding every pass is worse" verdict came from a measurement, never from
    // looking at the ground. -2 keeps the census winner, -1 binds every
    // qualifying pass, >=0 pins one ordinal.
    long onlyPass = g_mvSceneOrdinal;
    {
        // -1 (auto) is what the tuned config ships; -2 was the compiled
        // default and disagreed with it.
        const int sel = live::i("taa.mv_pass", "TAA_MV_PASS", -1);
        if (sel != -2) onlyPass = sel;
    }

    // The ordinal counts every QUALIFYING pass, whether or not it ends up
    // bound - otherwise pinning to one pass would renumber the very thing being
    // pinned, and index 3 would mean something different on the run that
    // selected it than on the run that measured it.
    if (isScene) {
        g_mvPassOrdinal = (int)g_mvQualifyThisFrame++;
        if (onlyPass >= 0 && g_mvPassOrdinal != (int)onlyPass) isScene = false;
    } else {
        g_mvPassOrdinal = -1;
    }

    // The ordinal renumbers when the frame's pass list shifts, which is how
    // the bind alternated between the G-buffer (colour=5) and lit (colour=1)
    // passes at 4K and left the field zero where the resolve reads: the clear
    // and the writes landed on a different pass every frame. A pass's SHAPE
    // does not renumber. Latch the colour count of the first pass bound at
    // this target size and refuse a differently-shaped pass - one frame
    // unbound self-corrects; a wrong bind poisons the whole field.
    // ---- THE LATCH WAS LOCKING OUT THE G-BUFFER, WHICH IS THE TERRAIN.
    //
    // X-Plane 12 is a DEFERRED renderer. The opaque world - terrain first of
    // all - is drawn into a five-attachment G-buffer, and only the lighting
    // result lands in a single-colour target. Both are 3840x2160 with depth, so
    // both satisfy isScene; the pass census recorded them as
    //
    //     [17] 3840x2160  depth=yes  color=5     <- G-buffer, the terrain
    //     [20] 3840x2160  depth=yes  color=3
    //     [16/19/21..] 3840x2160  depth=yes  color=1
    //
    // The latch below took the colour count of whichever qualifying pass
    // arrived FIRST and refused every differently-shaped pass afterwards. It
    // latched 1. From then on the G-buffer could never bind the velocity
    // target, so the terrain draws had no velocity attachment to write into -
    // and it did not matter in the slightest that they were being patched.
    //
    // That is the whole of the black ground. viz=8 showed the written map black
    // across the entire terrain while 14872 pipelines reported "both" stages
    // patched, which reads as a contradiction only until you notice that a
    // patched shader writing to an attachment that is not bound writes nowhere.
    // Pipeline counts are not draw counts: the 122 pipelines still declined
    // by design can be, and were, the whole ground.
    //
    // The alternation the latch was written to stop came from the ORDINAL
    // filter - g_mvPassOrdinal renumbers when the frame's pass list shifts, so
    // a pinned ordinal points at the G-buffer one frame and the lit pass the
    // next, and the clear and the writes land on different passes. That is a
    // real failure, but it only exists while an ordinal is pinned. With
    // taa.mv_pass = -1 every qualifying pass binds, the first one of the frame
    // clears and the rest LOAD, and passes accumulate instead of fighting.
    //
    // So the latch is now scoped to the case it was built for.
    //   taa.sticky_colour = -1  AUTO: latch only while an ordinal is pinned
    //                        0  never latch - every qualifying pass may bind
    //                        1  always latch (the old behaviour)
    {
        const int stickyMode = live::i("taa.sticky_colour",
                                       "TAA_STICKY_COLOUR", -1);
        const bool stickyOn = (stickyMode > 0) ||
                              (stickyMode < 0 && onlyPass >= 0);
        if (isScene && stickyOn) {
            const uint32_t cc = info->colorAttachmentCount;
            if (g_mvStickyColour == 0 || g_mvStickyW != g_mv.w) {
                g_mvStickyColour = cc;
                g_mvStickyW      = g_mv.w;
                trace("MV STICKY: latched pass shape colour=%u at %ux%u", cc,
                      g_mv.w, g_mv.h);
            } else if (cc != g_mvStickyColour) {
                // Never silent again. A pass refused here writes no velocity
                // for everything drawn in it, and the only symptom downstream
                // is an empty region of the field that looks exactly like a
                // shader that failed to patch.
                static std::set<uint32_t> refused;
                std::lock_guard<std::mutex> g(g_lock);
                if (!refused.count(cc)) {
                    refused.insert(cc);
                    trace("MV STICKY: REFUSING pass colour=%u (latched %u) - "
                          "every draw in it writes no velocity", cc,
                          g_mvStickyColour);
                }
                isScene = false;
            }
        }
    }

    if (isScene) {
        // ---- THIS is the pass to resolve after.
        //
        // Resolving on the FIRST scene pass caught the frame half drawn and
        // wrote black where the aircraft had not been rendered yet. Moving it
        // to the LAST scene pass overshot the other way: the flag marks any
        // full-viewport pass with depth, which includes the cockpit pass AFTER
        // tonemapping, so the result was written into an HDR target X-Plane had
        // already consumed - invisible, and paying a 63 MB copy per frame for
        // nothing.
        //
        // The pass that binds the velocity target is the one where the 3D scene
        // and its vectors are both complete and neither post-processing nor the
        // cockpit has run. That is the only correct boundary.
        uint32_t n = ++g_mvBindsThisFrame;
        // The staleness stamp. g_frameCount is the PRESENTED frame count, so
        // during recording of the next frame this stores the previous frame's
        // number - which is why every consumer compares with one frame of
        // tolerance rather than for equality.
        g_mvLastBindFrame.store(g_frameCount);
        if (n > g_mvBindsMax) {
            g_mvBindsMax = n;
            trace("MV BINDS: %u pass(es) bound the velocity target in one frame "
                  "- shape %ux%u colour=%u", n,
                  info->renderArea.extent.width, info->renderArea.extent.height,
                  info->colorAttachmentCount);
        }
    }

    // EVERY DISTINCT PASS SHAPE, once each.
    //
    // The pass log printed the first three and then every hundred-thousandth,
    // which is a sample chosen by arrival order rather than by what is
    // interesting - and it happened to contain no scene passes at all, so "no
    // BOUND lines" could not be told apart from "scene passes exist but were
    // never logged". Keying the log to distinct SHAPES instead means every kind
    // of pass appears exactly once, and the one that should have matched is
    // visible next to the reason it did not.
    {
        static std::set<uint64_t> seen;
        uint64_t key = ((uint64_t)info->renderArea.extent.width << 32)
                     | ((uint64_t)info->renderArea.extent.height << 8)
                     | ((uint64_t)info->colorAttachmentCount << 2)
                     | (hasDepth ? 2u : 0u) | (isScene ? 1u : 0u);
        std::lock_guard<std::mutex> g(g_lock);
        if (seen.size() < 40 && !seen.count(key)) {
            seen.insert(key);
            trace("MV: pass shape %ux%u colour=%u depth=%s -> %s "
                  "(velocity target is %ux%u)",
                  info->renderArea.extent.width, info->renderArea.extent.height,
                  info->colorAttachmentCount, hasDepth ? "yes" : "no",
                  isScene ? "SCENE - binding" : "not the scene pass",
                  g_mv.w, g_mv.h);
        }
    }

    // EXACTLY ONE EXTRA ATTACHMENT. No padding.
    //
    // The previous version padded every pass out to a fixed index of 7, so that
    // one fragment module could write the same Location everywhere. It was
    // valid and the driver hated it: depth-only shadow passes claiming eight
    // colour attachments, every pipeline declaring eight formats with seven
    // undefined, and vkCreateGraphicsPipelines returning VK_ERROR_UNKNOWN.
    //
    // Patching fragment shaders per PIPELINE instead removes the need for any
    // of it. A pipeline built against a pass with N attachments gets N+1, and
    // its fragment shader is patched to write Location N - so the index always
    // matches without a single wasted slot.
    if (info->colorAttachmentCount >= 8) return false;   // no room; leave alone

    // Depth-only passes (shadow cascades) get NO slot. Their pipelines carry
    // no colour blend state, so the pipeline hook skips them - and an extended
    // pass drawing unextended pipelines is an attachment-count mismatch, which
    // is undefined behaviour that intermittently surfaces as DEVICE_LOST at
    // submit. A velocity slot on a shadow pass bought nothing anyway.
    if (info->colorAttachmentCount == 0) return false;

    atts.assign(info->pColorAttachments,
                info->pColorAttachments + info->colorAttachmentCount);

    VkRenderingAttachmentInfo mv;
    memset(&mv, 0, sizeof(mv));
    mv.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    mv.imageView   = isScene ? g_mv.view : VK_NULL_HANDLE;
    mv.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    mv.loadOp      = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    mv.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    if (isScene) {
        // Once the present-time clear is running, no pass clears: the target
        // was zeroed after FSR2 consumed it last frame, so LOAD is correct and
        // no thread has to win a race to decide it.
        bool first = !g_mvClearedAtPresent.load() &&
                     !g_mvClearedThisFrame.exchange(true);
        mv.loadOp  = first ? VK_ATTACHMENT_LOAD_OP_CLEAR
                           : VK_ATTACHMENT_LOAD_OP_LOAD;
        mv.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        // ---- A SENTINEL, NOT ZERO.
        //
        // Zero conflates two opposite facts: "nothing drew here" and "this
        // surface is genuinely stationary". The resolve has to tell them apart -
        // sky and clouds must decline history, stationary ground must keep it -
        // and it was doing so with `cameraMoved && vel == 0`. That works only
        // while a still camera reads as still. It does not: the matrix is float32
        // and the airframe trembles on its gear with engines running, so
        // cameraMoved is true, a stationary world writes exactly zero
        // everywhere, and the ENTIRE FRAME is declared unwritten and loses its
        // history. Measured: the rejection viz is solid red across the ground,
        // alpha has no effect at any value, and disabling the test alone takes
        // temporal deviation from 0.89 to 0.154.
        //
        // Clearing to a value no real vector can hold removes the ambiguity at
        // the source. Screen-space motion is bounded by a couple of screens per
        // frame; -1e9 is not reachable, survives RGBA16F (finite, ~-1e9 rounds
        // to a large negative half-float... which is why the test below uses a
        // threshold rather than equality), and needs no extra channel.
        mv.clearValue.color.float32[0] = kMvUnwritten;
        mv.clearValue.color.float32[1] = kMvUnwritten;
    }

    atts.push_back(mv);
    out = *info;
    out.colorAttachmentCount = (uint32_t)atts.size();
    out.pColorAttachments    = atts.data();
    return true;
}

// ---- TAKING OVER X-PLANE'S OWN FSR, WHICH IS COMPUTE AND NOT A RENDER PASS.
//
// Read out of the sim's own shader archive rather than guessed at.
// Resources/shaders/bin/spv/fsr.xsa is a ZIP holding six SPIR-V modules, and
// fsr_mapping.xsv names their resources: u_input_texture with u_input_sampler
// in, i_output_texture out, u_fsr_data for constants. Every one of them is
// OpEntryPoint GLCompute.
//
// So there is no render pass to swallow - an earlier version of this waited on
// vkCmdBeginRendering and would have waited forever. It is a vkCmdDispatch with
// a compute pipeline bound, and the way to take it over is to recognise the
// shader module, recognise pipelines built from it, and drop the dispatch.
//
// The fingerprint is u_fsr_data: distinctive, present in all six variants, and
// carried in OpName so it survives an X-Plane update that recompiles them.
static std::set<VkShaderModule> g_xpFsrModules;    // modules that ARE X-Plane's FSR
static std::set<VkPipeline>     g_xpFsrPipelines;  // compute pipelines built from them
static std::map<void*, bool>    g_cbFsrBound;      // is an FSR pipeline bound on this cb
static std::map<void*, bool>    g_cbFsrOurs;       // is it specifically OUR substituted pipeline
static uint64_t g_xpFsrDropped   = 0;
// The module we actually substituted, and the pipelines built from it. There
// are TWO FSR modules - EASU, which we replace, and RCAS, which we leave alone
// - and only the first writes the sentinel. Probing after the wrong one reads
// a pixel X-Plane has since overwritten.
static VkShaderModule       g_xpFsrOurModule = VK_NULL_HANDLE;
static std::set<VkPipeline> g_xpFsrOurPipelines;

// Images seen in descriptor sets bound while one of X-Plane's FSR pipelines was
// active on this command buffer. The upscale's OUTPUT is in here; so is its
// input, and both are found the same way, so they are told apart by size at
// dispatch time - an upscaler's destination is the larger of the two, and that
// is true whatever X-Plane calls them.
static std::map<void*, std::vector<VkImage> > g_cbFsrImages;
// STORAGE images pushed on this command buffer, newest last. The upscale's
// destination is a storage image, and X-Plane pushes it immediately before the
// dispatch, so the most recent one of the right extent is the answer - no
// ranking, no size heuristic, no guessing between seven identical candidates.
static std::map<void*, std::vector<VkImage> > g_cbPushedStorage;
static uint64_t g_xpFsrBlits   = 0;
static uint64_t g_xpFsrNoTarget = 0;

// ---- fsr.replace: DO WE TAKE OVER X-PLANE'S UPSCALE?
//
// Read live so it can be flipped mid-flight against the sim's own FSR toggle,
// which is the only honest way to compare the two paths - a restart changes
// scenery, weather and thermal state along with the setting.
//
// Default OFF. A layer that silently replaces a shipped feature the moment it
// is installed is not something a user can reason about, and X-Plane's FSR
// working exactly as Laminar wrote it has to be the state you get by default.
// Which upscaler runs in the slot we took over. "shader" is the built-in
// Catmull-Rom replacement; "fsr3" is AMD's temporal upscaler.
//
// Read live rather than latched, unlike fsr.replace: the substitution has to be
// decided before shader modules are created, but WHICH upscaler dispatches is a
// per-frame decision and can be switched in flight to compare them on the same
// scene, weather and thermal state.
static bool fsr3Wanted()
{
    const char *e = getenv("TAA_FSR3");
    if (e && atoi(e)) return true;
    return live::i("fsr.backend_fsr3", nullptr, 0) != 0;
}

static bool fsrReplaceEnabled()
{
    // ---- LATCHED ONCE, AND ONLY AFTER THE FILE HAS BEEN READ.
    //
    // Read per call, this gave DIFFERENT answers for the same shader module.
    // Modules are created during startup, before the live config has been
    // polled, so an early call saw the built-in default while a later one saw
    // the file - and the trace showed exactly that: the 58436-byte module was
    // created without the "carries our code" marker while SUBSTITUTING printed
    // once. Substitution and tagging disagreed about the same module, so the
    // pipeline was never recognised and the probe never fired.
    //
    // loadNow() forces the read rather than waiting for the next poll, so the
    // first caller gets the file's answer instead of a default that would then
    // be latched for the life of the process.
    //
    // Latching also makes the switch honest about itself: module substitution
    // happens once at startup, so this cannot take effect mid-flight however
    // often the file is edited. A value that only applies at launch should be
    // read at launch.
    static int cached = -1;
    if (cached < 0) {
        live::loadNow();
        cached = (live::i("fsr.replace", "TAA_REPLACE_XPFSR", 0) != 0) ? 1 : 0;
        trace("XP FSR: fsr.replace latched %s for this process - module "
              "substitution is decided at startup and cannot change later.",
              cached ? "ON" : "off");
    }
    return cached != 0;
}

// Does this SPIR-V belong to X-Plane's FSR?
//
// u_fsr_data is the uniform block every one of the six variants declares, and
// the name survives in OpName, so this keeps working if the sim recompiles
// them. A raw scan of the word stream is enough - SPIR-V stores literal strings
// as packed words, so the ASCII appears contiguously.
static bool spirvIsXpFsr(const uint32_t *code, size_t words)
{
    if (!code || words < 8) return false;
    const char *p = (const char *)code;
    size_t bytes = words * 4;
    static const char kNeedle[] = "u_fsr_data";
    const size_t n = sizeof(kNeedle) - 1;
    if (bytes < n) return false;
    for (size_t i = 0; i + n <= bytes; ++i)
        if (!memcmp(p + i, kNeedle, n)) return true;
    return false;
}

static VKAPI_ATTR void VKAPI_CALL Layer_CmdBeginRendering(
    VkCommandBuffer cb, const VkRenderingInfo *info)
{
    if (info && g_passesThisFrame < 32) {
        g_passSizes[g_passesThisFrame][0] = info->renderArea.extent.width;
        g_passSizes[g_passesThisFrame][1] = info->renderArea.extent.height;
        g_passHasDepth[g_passesThisFrame] =
            (info->pDepthAttachment && info->pDepthAttachment->imageView != VK_NULL_HANDLE) ? 1 : 0;

        // Record the layout the app is using for depth. The velocity pass has
        // to barrier from the ACTUAL layout, not an assumed one - naming the
        // wrong oldLayout would discard the very contents we came to read.
        if (info->pDepthAttachment && info->pDepthAttachment->imageView != VK_NULL_HANDLE) {
            g_sceneDepthLayout = info->pDepthAttachment->imageLayout;

            // Track the depth image of the last full-viewport depth pass. Full
            // viewport only: shadow cascades and reflection passes also carry
            // depth, at their own resolutions.
            if (isSceneSized(info->renderArea.extent.width,
                             info->renderArea.extent.height,
                             info->colorAttachmentCount)) {
                std::lock_guard<std::mutex> g(g_lock);

                // Full viewport AND depth: this pass draws world geometry, so
                // any viewport set inside it is a jitter candidate. Tracked per
                // command buffer rather than globally because X-Plane records
                // on several threads at once - a global "are we in the scene
                // pass" flag would be set by one thread and read by another,
                // which is how the pass-index approach failed earlier.
                g_cbInScenePass[cb] = true;
                g_cbSawScenePass[cb] = true;   // the resolve boundary needs this
                ++g_cbScenePassCount[cb];

                // Body-frame reprojection applies only to the cockpit pass, and
                // only when a pass index has been measured and configured.
                g_cbInCockpitPass[cb] =
                    (g_cockpitPassIndex > 0 &&
                     (int)g_cbScenePassCount[cb] == g_cockpitPassIndex);

                // ---- WHAT EACH SCENE PASS ACTUALLY IS.
                //
                // The cockpit pass index was chosen from a census that showed
                // TWO scene passes and an assumption about draw order. The trace
                // then showed six per command buffer, so the index was picked
                // from a set that was never characterised - and index 2 put
                // body-frame reprojection on the world, which is how the
                // "intense vibration" happened.
                //
                // Rather than guess again, describe every scene pass by the
                // things that distinguish a G-buffer fill from a cockpit
                // overlay: attachment count, extent, depth load/store ops, and
                // whether depth is being cleared. A cockpit pass typically
                // CLEARS depth so the panel draws over the world regardless of
                // distance - that is a fingerprint, not a draw-order guess.
                {
                    int idx = (int)g_cbScenePassCount[cb];
                    static std::map<int, uint64_t> seen;
                    if (seen[idx]++ % 2000 == 0) {
                        const VkRenderingAttachmentInfo *da = info->pDepthAttachment;
                        trace("SCENE PASS %d: %ux%u colour=%u depth=%s "
                              "depthLoad=%d depthStore=%d clearDepth=%.3f%s",
                              idx, info->renderArea.extent.width,
                              info->renderArea.extent.height,
                              info->colorAttachmentCount,
                              da ? "yes" : "no",
                              da ? (int)da->loadOp : -1,
                              da ? (int)da->storeOp : -1,
                              da ? da->clearValue.depthStencil.depth : -1.0f,
                              (da && da->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR)
                                  ? "  <- CLEARS DEPTH (cockpit-overlay fingerprint)"
                                  : "");
                    }
                }
                std::map<VkImageView, VkImage>::iterator vi =
                    g_viewToImage.find(info->pDepthAttachment->imageView);
                if (vi != g_viewToImage.end()) {
                    g_frameDepthImage = vi->second;
                    g_lastDepthPassIdx = (int)g_passesThisFrame;

                    // Keep EVERY full-viewport depth image the frame binds, in
                    // the order it binds them, not just the last one.
                    //
                    // "Last full-viewport depth pass" is a good first guess but
                    // it is still a guess, and when it is wrong the only way to
                    // try the next candidate has been to rebuild and fly again.
                    // With the whole ordered list retained, a wrong pick can be
                    // rejected and replaced from inside the same session.
                    bool seen = false;
                    for (size_t k = 0; k < g_frameDepthList.size(); ++k)
                        if (g_frameDepthList[k] == vi->second) { seen = true; break; }
                    if (!seen && g_frameDepthList.size() < 32)
                        g_frameDepthList.push_back(vi->second);

                    // This pass is full-viewport AND carries depth, so it is
                    // drawing world geometry. Its colour attachment is the
                    // scene colour the resolve will have to work on. Taking the
                    // LAST such pass in the frame gives the fully composed 3D
                    // image, before any 2D overlay goes on top.
                    if (info->colorAttachmentCount > 0 && info->pColorAttachments
                        && info->pColorAttachments[0].imageView != VK_NULL_HANDLE) {
                        std::map<VkImageView, VkImage>::iterator ci2 =
                            g_viewToImage.find(info->pColorAttachments[0].imageView);
                        if (ci2 != g_viewToImage.end()) {
                            std::map<VkImage, ColorTarget>::iterator ct =
                                g_colorImages.find(ci2->second);
                            if (ct != g_colorImages.end()) {
                                // Prefer an HDR FLOAT target over merely the
                                // last one.
                                //
                                // "Last full-viewport depth pass" identifies
                                // the depth image correctly but not the colour
                                // image: X-Plane's final 3D composite still has
                                // depth bound while writing to an 8-bit sRGB
                                // target, so taking the last one latched
                                // R8G8B8A8_SRGB and resolved the tonemapped
                                // composite instead of the scene. It ran 600
                                // dispatches on the wrong image and looked
                                // completely fine, because with jitter off a
                                // correct resolve is invisible.
                                //
                                // The HDR target is also the RIGHT place to
                                // resolve: before tonemapping, which is where
                                // upscalers want to sit too.
                                // SINGLE-SAMPLE ONLY.
                                //
                                // X-Plane keeps two HDR targets of identical
                                // size and format, one single-sample and one
                                // 2x multisampled. The multisampled one cannot
                                // be used at either end: a plain sampler2D view
                                // over a multisampled image is undefined - the
                                // same trap that made depth read back as
                                // near-zero - and vkCmdCopyImage requires the
                                // sample counts to match, so copying our
                                // single-sample result into it produced the
                                // sheared bands.
                                //
                                // The single-sample target is also the right
                                // one on the merits: it is the post-resolve,
                                // pre-tonemap image, which is exactly what a
                                // temporal resolve and an upscaler both want.
                                bool continueSel = true;
                                VkFormat f = ct->second.format;
                                bool hdr = (f == VK_FORMAT_R16G16B16A16_SFLOAT ||
                                            f == VK_FORMAT_R32G32B32A32_SFLOAT ||
                                            f == VK_FORMAT_B10G11R11_UFLOAT_PACK32)
                                        && ct->second.samples == VK_SAMPLE_COUNT_1_BIT;
                                bool haveHdr =
                                    (g_sceneColor.format == VK_FORMAT_R16G16B16A16_SFLOAT ||
                                     g_sceneColor.format == VK_FORMAT_R32G32B32A32_SFLOAT ||
                                     g_sceneColor.format == VK_FORMAT_B10G11R11_UFLOAT_PACK32);
                                // Never select a multisampled target, even as a
                                // fallback: we can neither sample it through
                                // the view we build nor copy into it.
                                bool usable = ct->second.samples == VK_SAMPLE_COUNT_1_BIT;
                                // ---- TAA_SCENE_8BIT: target the composite instead.
                                //
                                // Preferring HDR is right on the merits -
                                // pre-tonemap is where a resolve and an upscaler
                                // both want to sit - and it is why the original
                                // code was changed, because latching
                                // R8G8B8A8_SRGB "resolved the tonemapped
                                // composite instead of the scene".
                                //
                                // Except every test since says nothing reads the
                                // HDR target after we write it: no seam from a
                                // half copy, no change from writing both HDR
                                // targets, no change from resolving ahead of the
                                // composite - and a direct swapchain blit was the
                                // only thing that ever reached the screen. The
                                // 8-bit composite is what is actually read and
                                // presented. Resolving post-tonemap is a real
                                // compromise, not a preference, but a compromise
                                // that is visible beats a purer one that is not.
                                // ONLY 8-BIT TARGETS, or this oscillates.
                                //
                                // "The last full-viewport target" is not stable:
                                // the log showed it latching fmt 37, then 97,
                                // then 83, then 97, then 43 within a single
                                // run, so the destination changed from pass to
                                // pass and the resolve chased a moving target.
                                // The composite is the 8-bit one; ignore the
                                // HDR intermediates entirely in this mode.
                                // ---- IS THIS PASS DRAWING STRAIGHT INTO THE SWAPCHAIN?
                                //
                                // Every internal target we have written to has
                                // been invisible, and a direct swapchain blit was
                                // the only thing that ever showed. The obvious
                                // question - does X-Plane composite into the
                                // swapchain image itself? - could not be asked
                                // until the layer started tracking those images
                                // an hour ago. If it does, that image is both the
                                // right destination AND the one place a write is
                                // known to survive, and the UI still goes on top
                                // afterwards because the UI passes come later.
                                {
                                    bool isSwap = false;
                                    for (std::map<VkSwapchainKHR, std::vector<VkImage> >::iterator
                                             si = g_swapImages.begin();
                                         si != g_swapImages.end() && !isSwap; ++si)
                                        for (size_t k = 0; k < si->second.size(); ++k)
                                            if (si->second[k] == ci2->second) { isSwap = true; break; }
                                    static std::set<VkImage> seenTargets;
                                    if (seenTargets.size() < 16 && !seenTargets.count(ci2->second)) {
                                        seenTargets.insert(ci2->second);
                                        trace("PASS TARGET %p fmt=%d %ux%u samples=%d%s",
                                              (void*)ci2->second, (int)ct->second.format,
                                              ct->second.w, ct->second.h,
                                              (int)ct->second.samples,
                                              isSwap ? "  [SWAPCHAIN]" : "");
                                    }
                                    static int saidSwap = -1;
                                    if (saidSwap != (int)isSwap) {
                                        saidSwap = (int)isSwap;
                                        trace("COLOR: full-viewport pass target %s a "
                                              "swapchain image (fmt=%d). %s",
                                              isSwap ? "IS" : "is NOT",
                                              (int)ct->second.format,
                                              isSwap
                                                ? "So X-Plane composites straight into "
                                                  "the presented image, and that is where "
                                                  "the resolve belongs."
                                                : "So the presented image is only ever "
                                                  "written by something we have not "
                                                  "intercepted.");
                                    }
                                    if (isSwap && getenv("TAA_SCENE_SWAPCHAIN")) {
                                        g_sceneColor = ct->second;
                                        g_sceneColor.layout =
                                            info->pColorAttachments[0].imageLayout;
                                        continueSel = false;
                                    }
                                }

                                // sRGB SPECIFICALLY, not merely "not HDR".
                                //
                                // The full pass census finally showed where the
                                // frame goes: a full-viewport DEPTH-LESS pass
                                // draws straight into the swapchain image, and
                                // the last full-viewport pass WITH depth writes
                                //
                                //   3840x2160 att0 -> ... fmt=43 depth=yes
                                //
                                // fmt 43 is R8G8B8A8_SRGB - the final 3D
                                // composite the swapchain pass samples. "Any
                                // 8-bit target" also matches fmt 37 at the same
                                // size, which is an intermediate, so the earlier
                                // version latched the wrong one and the resolve
                                // went somewhere nothing reads.
                                bool srgb = (f == VK_FORMAT_R8G8B8A8_SRGB ||
                                             f == VK_FORMAT_B8G8R8A8_SRGB);
                                bool haveSrgb = (g_sceneColor.format == VK_FORMAT_R8G8B8A8_SRGB ||
                                                 g_sceneColor.format == VK_FORMAT_B8G8R8A8_SRGB);
                                bool want8 = getenv("TAA_SCENE_8BIT") != nullptr;
                                if (continueSel && usable && want8 && (srgb || !haveSrgb) && !hdr) {
                                    // Last full-viewport target of the frame,
                                    // whatever its format - which is the
                                    // composite.
                                    g_sceneColor = ct->second;
                                    g_sceneColor.layout =
                                        info->pColorAttachments[0].imageLayout;
                                    static VkFormat saidFmt = VK_FORMAT_UNDEFINED;
                                    if (saidFmt != ct->second.format) {
                                        saidFmt = ct->second.format;
                                        trace("COLOR: TAA_SCENE_8BIT - scene target "
                                              "is now the LAST full-viewport target "
                                              "(fmt=%d), not the HDR one.",
                                              (int)ct->second.format);
                                    }
                                } else if (continueSel && usable && (hdr || !haveHdr)) {
                                    g_sceneColor = ct->second;
                                    g_sceneColor.layout =
                                        info->pColorAttachments[0].imageLayout;
                                }

                                // How many DISTINCT HDR scene targets does the
                                // application rotate through?
                                //
                                // The resolve latches one image and writes into
                                // it every frame. If X-Plane double-buffers its
                                // scene target - and its command buffers
                                // demonstrably alternate between two families -
                                // then half the time we would be resolving into
                                // an image the current frame is not drawing to,
                                // which would corrupt part of the picture and
                                // flicker as it alternates. That is exactly the
                                // symptom, so the count decides it.
                                if (hdr) {
                                    bool known = false;
                                    for (size_t q = 0; q < g_hdrTargets.size(); ++q)
                                        if (g_hdrTargets[q] == ct->second.image) { known = true; break; }
                                    if (!known && g_hdrTargets.size() < 16) {
                                        g_hdrTargets.push_back(ct->second.image);
                                        trace("COLOR: distinct HDR scene target #%zu = %p "
                                              "(%ux%u)", g_hdrTargets.size(),
                                              (void*)ct->second.image,
                                              ct->second.w, ct->second.h);
                                    }
                                }

                                // Does this pass resolve MSAA on the way out?
                                //
                                // If it does, the resolve target is the
                                // single-sample image an upscaler actually
                                // wants, and it is handed to us directly - no
                                // need to hunt for a later pass or resolve it
                                // ourselves. The multisampled attachment is the
                                // wrong input for FSR2 or DLSS either way.
                                g_sceneResolveMode =
                                    (uint32_t)info->pColorAttachments[0].resolveMode;
                                g_sceneResolveImage = VK_NULL_HANDLE;
                                if (info->pColorAttachments[0].resolveImageView
                                        != VK_NULL_HANDLE) {
                                    std::map<VkImageView, VkImage>::iterator ri =
                                        g_viewToImage.find(
                                            info->pColorAttachments[0].resolveImageView);
                                    if (ri != g_viewToImage.end())
                                        g_sceneResolveImage = ri->second;
                                }
                            }
                        }
                    }
                    // Only trust the layout for the exact image we sample.
                    if (g_sceneDepth == VK_NULL_HANDLE || vi->second == g_sceneDepth) {
                        g_depthFreshThisFrame  = true;
                        g_depthLayoutThisFrame = info->pDepthAttachment->imageLayout;
                        // Only mark a command buffer once depth is being READ
                        // rather than cleared.
                        //
                        // loadOp CLEAR means the pass is starting fresh - the
                        // depth buffer is wiped at that point and contains
                        // nothing to reproject from. Injecting after such a
                        // buffer sampled an all-zero depth image, which put
                        // every pixel on the far plane; the dump showed
                        // depth=0.000000 everywhere while the final velocity
                        // image (written by a later, correct dispatch) looked
                        // fine.
                        //
                        // loadOp LOAD means the pass is continuing into depth
                        // that already holds the scene, which is what we want
                        // to read.
                        if (info->pDepthAttachment->loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR) {
                            CbDepthUse &u = g_cbDepthUse[cb];
                            u.used   = true;
                            u.layout = info->pDepthAttachment->imageLayout;
                            ++u.depthPasses;
                        }
                    }
                }
            }
        }
        g_passColorCount[g_passesThisFrame] = info->colorAttachmentCount;

        // A pass that is not full-viewport-with-depth is not the 3D scene, so
        // nothing recorded inside it should be jittered. Setting this false
        // explicitly matters: shadow cascades, reflection passes and the UI all
        // reuse command buffers, and a flag left set by an earlier scene pass
        // would jitter whichever came next.
        // Scene-SIZED, not display-sized. With X-Plane's own FSR enabled the
        // 3D passes are a fraction of the window, and comparing against the
        // window made this false for every pass in the frame.
        bool fullViewport = isSceneSized(info->renderArea.extent.width,
                                         info->renderArea.extent.height,
                                         info->colorAttachmentCount);
        bool sceneNow = info->pDepthAttachment
                     && info->pDepthAttachment->imageView != VK_NULL_HANDLE
                     && fullViewport;

        // ---- EVERY PASS, INCLUDING DEPTH-LESS ONES.
        //
        // The earlier target census only ran inside the depth-carrying branch,
        // so a composite pass - full-viewport, no depth, draws the 3D result and
        // the UI into the presented image - could never appear in it. That is
        // precisely the shape of the pass we are missing: no render pass, blit,
        // copy or resolve that we intercept writes a swapchain handle, yet the
        // frame plainly gets there.
        {
            std::lock_guard<std::mutex> g(g_lock);
            // Flag the pass that draws into a swapchain image, so the descriptor
            // hook knows when to look.
            bool anySwap = false;
            for (uint32_t a = 0; a < info->colorAttachmentCount; ++a) {
                if (!info->pColorAttachments ||
                    info->pColorAttachments[a].imageView == VK_NULL_HANDLE) continue;
                std::map<VkImageView, VkImage>::iterator vs =
                    g_viewToImage.find(info->pColorAttachments[a].imageView);
                if (vs != g_viewToImage.end() && isSwapImage(vs->second)) anySwap = true;
            }
            g_cbInSwapPass[cb] = anySwap;

            // Remember every image ever bound as a colour attachment. The
            // composite source is identified by NOT being one - see
            // g_computeComposite below.
            for (uint32_t a = 0; a < info->colorAttachmentCount; ++a) {
                if (!info->pColorAttachments ||
                    info->pColorAttachments[a].imageView == VK_NULL_HANDLE) continue;
                std::map<VkImageView, VkImage>::iterator va =
                    g_viewToImage.find(info->pColorAttachments[a].imageView);
                if (va != g_viewToImage.end()) g_seenAsAttachment.insert(va->second);
            }

            static std::set<VkImage> seenAny;
            for (uint32_t a = 0; a < info->colorAttachmentCount; ++a) {
                if (!info->pColorAttachments ||
                    info->pColorAttachments[a].imageView == VK_NULL_HANDLE) continue;
                std::map<VkImageView, VkImage>::iterator vi =
                    g_viewToImage.find(info->pColorAttachments[a].imageView);
                if (vi == g_viewToImage.end()) {
                    static int unknown = 0;
                    if (++unknown <= 6)
                        trace("ANY PASS: attachment %u view %p has NO image mapping - "
                              "created before we hooked vkCreateImageView, so it "
                              "could never have matched a swapchain handle.",
                              a, (void*)info->pColorAttachments[a].imageView);
                    continue;
                }
                // Remember it: the pass that writes a swapchain image is where
                // our upscaled result has to land, and it is a DRAW pass rather
                // than a transfer - which is why substituting a blit source
                // never fired. Recorded before the trace throttle, or only the
                // first two dozen frames would ever deliver.
                // NO lock_guard HERE. This block already holds g_lock, and
                // std::mutex is not recursive - taking it twice on one thread
                // deadlocks the render thread outright. That is a black screen
                // and zero frames with every feature flag switched off, which
                // is exactly how it presented.
                // The layout comes from the attachment info, NOT from an
                // assumption. Transitioning from the wrong oldLayout is
                // undefined, and guessing COLOR_ATTACHMENT_OPTIMAL is what I
                // did the first time this was attempted.
                // AUTHORITATIVE LAYOUT, STRAIGHT FROM THE PASS.
            //
            // A render pass states the layout each attachment is in. That is
            // ground truth, and it is what made the swapchain self-blit work
            // while every guess at the scene target's layout failed. Barriers
            // alone could never complete the table: this layer's own
            // transitions go through dd.cmdPipelineBarrier - the next layer's
            // pointer - and so bypass our own hook entirely.
            noteLayout(vi->second, info->pColorAttachments[a].imageLayout);

            if (isSwapImage(vi->second)) {
                    CbSwapTarget t;
                    t.image  = vi->second;
                    t.layout = info->pColorAttachments[a].imageLayout;
                    g_cbSwapTarget[cb] = t;
                }
                if (seenAny.size() >= 24 || seenAny.count(vi->second)) continue;
                seenAny.insert(vi->second);
                std::map<VkImage, ColorTarget>::iterator ct = g_colorImages.find(vi->second);
                trace("ANY PASS: %ux%u att%u -> %p fmt=%d depth=%s%s",
                      info->renderArea.extent.width, info->renderArea.extent.height,
                      a, (void*)vi->second,
                      ct != g_colorImages.end() ? (int)ct->second.format : -1,
                      info->pDepthAttachment && info->pDepthAttachment->imageView ? "yes" : "no",
                      isSwapImage(vi->second) ? "  [SWAPCHAIN]" : "");
            }
        }

        if (g_cbDumpOn && g_cbDumpsLeft > 0) {
            std::lock_guard<std::mutex> g(g_lock);
            CbPassLog &pl = g_cbPassLog[cb];
            if (pl.seq.size() < 96)
                pl.seq += sceneNow ? 'S' : (fullViewport ? 'P' : 'o');
        }
        if (!sceneNow) {
            std::lock_guard<std::mutex> g(g_lock);
            g_cbInScenePass[cb] = false;
        }

        // ---- THE 3D/UI BOUNDARY, and where the resolve goes.
        //
        // A full-viewport pass with NO depth, in a command buffer that has
        // already recorded full-viewport passes WITH depth, is the first thing
        // after the 3D scene - post-processing or UI. Recording the resolve
        // here, before this pass is begun, means everything drawn from now on
        // lands on top of the resolved image and is never accumulated. That is
        // what keeps instrument text, ATC boxes and the map out of the history.
        //
        // Derived from the measured frame, not guessed: the pass dump showed
        // 2560x1440 depth=yes for passes 20-29 and depth=no for 30-31, so the
        // boundary is exactly the 29/30 transition.
        //
        // Tracked per command buffer for the same reason the jitter flag is -
        // X-Plane records on several threads, and a global "have we passed the
        // boundary yet" would be set by one thread and read by another.
        // ---- THE BOUNDARY WAS ONE PASS TOO LATE, AND THE CODE SAID SO.
        //
        // "A full-viewport pass with NO depth" assumes the first thing after
        // the 3D scene has no depth attachment. The comment on the scene-target
        // selection records the exception and nobody joined the two up:
        //
        //   "X-Plane's final 3D composite still has depth bound while writing
        //    to an 8-bit sRGB target"
        //
        // That composite is the pass that READS the HDR target. Because it
        // carries depth it counts as a scene pass here, so the depth-less
        // boundary fires only AFTER the HDR image has been consumed - and every
        // resolve we have ever recorded went into a buffer nothing reads again.
        // That is why the seam test produced no seam, why writing into both HDR
        // targets changed nothing, and why blitting onto the swapchain was the
        // only thing that ever appeared.
        //
        // The real boundary is the first full-viewport pass that stops writing
        // to the HDR scene target, whether or not it has depth. Catching it
        // means the resolve lands while the HDR image still has a reader.
        bool leavingSceneTarget = false;
        if (fullViewport && sceneNow && g_sceneColor.image != VK_NULL_HANDLE &&
            info && info->colorAttachmentCount > 0 && info->pColorAttachments &&
            info->pColorAttachments[0].imageView != VK_NULL_HANDLE) {
            std::lock_guard<std::mutex> g(g_lock);
            // TWO CONDITIONS, because the first version had neither tight
            // enough and fired on the second pass of the frame.
            //
            // "Writes somewhere other than the HDR target" also matches the
            // OTHER HDR target and every intermediate, so it triggered after a
            // single scene pass and resolved a half-drawn frame - the log read
            // "boundary after 1/1" and it cost 18 fps to do nothing.
            //
            // The composite is specifically the pass that writes an 8-BIT
            // target, and it only happens once the scene is complete. So
            // require both: an 8-bit destination, and as many scene passes as
            // this command buffer has ever recorded.
            std::map<VkImageView, VkImage>::iterator vi =
                g_viewToImage.find(info->pColorAttachments[0].imageView);
            if (vi != g_viewToImage.end() && vi->second != g_sceneColor.image) {
                std::map<VkImage, ColorTarget>::iterator ct = g_colorImages.find(vi->second);
                bool eightBit = ct != g_colorImages.end() &&
                                ct->second.format != VK_FORMAT_R16G16B16A16_SFLOAT &&
                                ct->second.format != VK_FORMAT_R32G32B32A32_SFLOAT &&
                                ct->second.format != VK_FORMAT_B10G11R11_UFLOAT_PACK32;

                uint32_t seenNow = g_cbScenePassCount.count(cb) ? g_cbScenePassCount[cb] : 0;
                uint32_t hi      = g_cbScenePassHigh.count(cb) ? g_cbScenePassHigh[cb] : 0;
                bool sceneDone   = hi > 0 && seenNow >= hi;

                std::map<VkCommandBuffer, bool>::iterator sw = g_cbSawScenePass.find(cb);
                leavingSceneTarget = eightBit && sceneDone &&
                                     (sw != g_cbSawScenePass.end() && sw->second);
            }
        }
        if (leavingSceneTarget) {
            static uint64_t nlog = 0;
            if (++nlog % 600 == 1)
                trace("RESOLVE: composite pass - full-viewport, depth bound, "
                      "writing an 8-bit target after all %u scene passes. This "
                      "is the pass that READS the HDR image, so the resolve goes "
                      "in ahead of it rather than at the depth-less pass after.",
                      g_cbScenePassHigh.count(cb) ? g_cbScenePassHigh[cb] : 0);
        }

        // WITH AN 8-BIT DESTINATION THE TIMING INVERTS.
        //
        // Resolving ahead of the composite is right when the destination is the
        // HDR image, because the composite READS it. When the destination IS
        // the composite's output, the composite WRITES it - so going in first
        // means X-Plane paints over us immediately. In that mode the correct
        // point is the original one: the depth-less pass after the composite
        // and before the UI.
        // The composite-pass trigger fires BEFORE X-Plane's compute writes the
        // composite. That is right when the destination is an attachment the
        // composite reads, and wrong when the destination IS the composite's
        // own output - we would be resolving an image that does not exist yet
        // for this frame. In that mode the correct point is the depth-less
        // swapchain pass, which is after the compute.

        // ---- MEASURE THE FIELD, WHATEVER ELSE IS RUNNING.
        //
        // In the predecessor this sat inside the resolve's dispatch, so it only
        // measured when an upscaler was active. The vectors are the product
        // here, so the measurement is unconditional.
        {
            std::map<VkCommandBuffer, VkDevice>::iterator rci = g_cbToDevice.find(cb);
            if (rci != g_cbToDevice.end()) {
                std::map<void*, DeviceData>::iterator rdi =
                    g_devices.find(dispatchKey(rci->second));
                if (rdi != g_devices.end() && g_mv.ready)
                    mvRecordReadback(rdi->second, cb, g_lastPushed,
                                     g_velSnap.selfTestExpectedPx,
                                     g_velSnap.selfTestPhase, g_velSnap.frame,
                                     g_velSnap.nearClip, g_velSnap.proj,
                                     g_velSnap.viewType);

            }
        }

    }
    ++g_passesThisFrame;

    // Append the velocity attachment slot. The vectors have to outlive the
    // call, so they are declared here rather than inside the helper.
    // Record what this pass looks like so vkCmdEndRendering can tell the
    // G-buffer pass from the lit pass without re-deriving it.
    if (info) {
        std::lock_guard<std::mutex> g(g_lock);
        CbPassInfo pi;
        pi.colorCount = info->colorAttachmentCount;
        pi.w = info->renderArea.extent.width;
        pi.h = info->renderArea.extent.height;
        if (info->colorAttachmentCount >= 1 && info->pColorAttachments) {
            std::map<VkImageView, VkImage>::iterator vi =
                g_viewToImage.find(info->pColorAttachments[0].imageView);
            if (vi != g_viewToImage.end()) pi.color0 = vi->second;
        }
        if (info->pDepthAttachment)
            pi.depthLoad = (info->pDepthAttachment->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
        g_cbPassInfo[cb] = pi;
    }

    std::vector<VkRenderingAttachmentInfo> mvAtts;
    VkRenderingInfo info2;
    if (mvAppendAttachment(info, mvAtts, info2)) {
        // The velocity target is attached to THIS pass on THIS command buffer.
        // Recorded here rather than inside the helper, which has no cb.
        { std::lock_guard<std::mutex> g(g_lock); g_cbMvBoundPass[cb] = true; }
        static uint64_t nlog = 0;
        if (++nlog <= 3 || (nlog % 100000) == 0)
            trace("MV: pass with %u colour attachments -> %u (velocity %s)",
                  info->colorAttachmentCount, info2.colorAttachmentCount,
                  mvAtts.back().imageView ? "BOUND" : "null slot");
        if (g_nextCmdBeginRendering) g_nextCmdBeginRendering(cb, &info2);
        return;
    }

    if (g_nextCmdBeginRendering) g_nextCmdBeginRendering(cb, info);
}

// Clearing the jitter flag is the whole job here. A viewport set between one
// pass ending and the next beginning belongs to neither, and must not be
// treated as scene geometry.

// ---- DELIVERY THROUGH A COMPUTE SHADER, BECAUSE THE BLIT ENGINE CANNOT DO IT.
//
// Measured on this machine, every combination tried:
//
//   outImg 16F  -> scene target 16F       clean   (FSR2's own copy-back)
//   swapchain 8 -> swapchain 8-bit sRGB   clean   (self-blit, 1:1 AND scaled)
//   outImg 16F  -> swapchain 8-bit sRGB   GARBAGE
//   scene  16F  -> swapchain 8-bit sRGB   GARBAGE
//
// Same-class blits work. Every 16F -> 8-bit conversion of real pixel data
// fails, whatever the source image, layout, extent, scaling or filter - and
// both BLIT_SRC and BLIT_DST are advertised, so the API never objects. The
// fast-cleared 16F source that appeared to work was metadata, not pixels: a
// fast clear writes compression state rather than texels, which is the same
// trap the green clear set earlier.
//
// The sim's own composite SAMPLES these 16F images every frame and looks
// perfect, so the data is intact and readable. What fails is the fixed-function
// conversion. This reads through the texture unit, like the sim does, and
// converts in code we control.
//
// It stores into an R8G8B8A8_UNORM image we own rather than the swapchain,
// because sRGB formats cannot be STORAGE images. That image reaches the
// swapchain as a raw 8-bit -> 8-bit COPY, which converts nothing, so the
// shader's sRGB encode lands exactly once and the channel swap in the shader
// accounts for the copy not reordering bytes.


struct DeliverPush { int32_t srcW, srcH, dstW, dstH; };



// ---- PERIODIC SCREENSHOTS, STRAIGHT OFF THE SWAPCHAIN.
//
// Read back the presented image every few seconds and write it to disk. This
// exists so the loop stops needing a human: build, launch, and look at the
// files afterwards, instead of asking someone to alt-tab and take a picture of
// every experiment.
//
// It captures the swapchain itself, so what lands on disk is exactly what was
// on screen - including anything our delivery wrote into it.
struct ShotCap {
    VkBuffer       buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void          *ptr = nullptr;
    uint32_t       w = 0, h = 0;
    bool           ready = false, armed = false, failed = false;
    int            wait = 0;
    uint32_t       index = 0;
    uint64_t       nextFrame = 0;
};
static ShotCap g_shot;

static bool shotWriteBmp(const char *path, const unsigned char *bgra,
                         uint32_t w, uint32_t h)
{
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    const uint32_t rowBytes = ((w * 3u) + 3u) & ~3u;
    const uint32_t imgBytes = rowBytes * h;
    unsigned char hdr[54];
    memset(hdr, 0, sizeof(hdr));
    hdr[0] = 'B'; hdr[1] = 'M';
    uint32_t total = 54 + imgBytes;  memcpy(hdr + 2, &total, 4);
    uint32_t off = 54;               memcpy(hdr + 10, &off, 4);
    uint32_t hs = 40;                memcpy(hdr + 14, &hs, 4);
    memcpy(hdr + 18, &w, 4);
    memcpy(hdr + 22, &h, 4);
    uint16_t planes = 1, bpp = 24;
    memcpy(hdr + 26, &planes, 2);
    memcpy(hdr + 28, &bpp, 2);
    memcpy(hdr + 34, &imgBytes, 4);
    fwrite(hdr, 1, sizeof(hdr), f);

    unsigned char *row = (unsigned char*)calloc(rowBytes, 1);
    if (!row) { fclose(f); return false; }
    for (int y = (int)h - 1; y >= 0; --y) {          // BMP is bottom-up
        const unsigned char *src = bgra + (size_t)y * w * 4;
        for (uint32_t x = 0; x < w; ++x) {
            row[x * 3 + 0] = src[x * 4 + 0];         // B
            row[x * 3 + 1] = src[x * 4 + 1];         // G
            row[x * 3 + 2] = src[x * 4 + 2];         // R
        }
        fwrite(row, 1, rowBytes, f);
    }
    free(row);
    fclose(f);
    return true;
}

// Called with the presented image already in TRANSFER_DST_OPTIMAL, at the end
// of the delivery, so the capture includes whatever we just wrote.
static void shotMaybe(DeviceData &dd, VkCommandBuffer cb, VkImage swapImg,
                      uint32_t w, uint32_t h, VkImageLayout layout)
{
    static int every = -1;
    if (every < 0) {
        const char *e = getenv("TAA_SHOT_SECONDS");
        every = e ? atoi(e) : 0;
    }
    if (every <= 0 || g_shot.failed) return;

    if (g_shot.armed) {
        if (--g_shot.wait > 0) return;
        g_shot.armed = false;
        _mkdir("D:\\TAA Dumps");
        char path[512];
        // ---- BOUNDED. THESE ARE 25 MB EACH.
        //
        // Unbounded, this wrote about three gigabytes in one session, filled the
        // drive, and truncated a file that happened to be mid-write at the time.
        // Twelve slots that wrap keep the recent history and cap the cost at
        // roughly 300 MB - enough to see what a change did, and it can never
        // grow past that.
        snprintf(path, sizeof(path), "D:\\TAA Dumps\\shot_%02u.bmp",
                 g_shot.index % 12u);
        g_shot.index++;
        bool ok = shotWriteBmp(path, (const unsigned char*)g_shot.ptr,
                               g_shot.w, g_shot.h);
        trace("SHOT: %s %s", ok ? "wrote" : "FAILED", path);
        return;
    }

    // Frames rather than a clock: no wall-clock call on the record path, and
    // the cadence only needs to be roughly every few seconds.
    if (g_frameCount < g_shot.nextFrame) return;
    g_shot.nextFrame = g_frameCount + (uint64_t)(every * 20);   // ~20 fps assumed

    if (!g_shot.ready) {
        if (!g_getPhysMemProps || !dd.createBuffer) { g_shot.failed = true; return; }
        VkDeviceSize bytes = (VkDeviceSize)w * h * 4;
        VkBufferCreateInfo bci;
        memset(&bci, 0, sizeof(bci));
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = bytes;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (dd.createBuffer(dd.device, &bci, nullptr, &g_shot.buf) != VK_SUCCESS) {
            g_shot.failed = true; trace("SHOT: buffer failed"); return;
        }
        VkMemoryRequirements mr;
        dd.getBufferMemReq(dd.device, g_shot.buf, &mr);
        VkPhysicalDeviceMemoryProperties mp;
        memset(&mp, 0, sizeof(mp));
        g_getPhysMemProps(dd.phys, &mp);
        uint32_t ti = UINT32_MAX;
        for (uint32_t k = 0; k < mp.memoryTypeCount; ++k)
            if ((mr.memoryTypeBits & (1u << k)) &&
                (mp.memoryTypes[k].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (mp.memoryTypes[k].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                ti = k; break;
            }
        VkMemoryAllocateInfo mai;
        memset(&mai, 0, sizeof(mai));
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mr.size; mai.memoryTypeIndex = ti;
        if (ti == UINT32_MAX ||
            dd.allocateMemory(dd.device, &mai, nullptr, &g_shot.mem) != VK_SUCCESS ||
            dd.bindBufferMemory(dd.device, g_shot.buf, g_shot.mem, 0) != VK_SUCCESS ||
            dd.mapMemory(dd.device, g_shot.mem, 0, bytes, 0, &g_shot.ptr) != VK_SUCCESS) {
            g_shot.failed = true; trace("SHOT: memory failed"); return;
        }
        g_shot.w = w; g_shot.h = h; g_shot.ready = true;
        trace("SHOT: capturing %ux%u to D:\\TAA Dumps every ~%d s", w, h, every);
    }
    if (g_shot.w != w || g_shot.h != h) return;   // size changed; skip this one

    // Own the transitions. This used to sit inside the delivery block and so
    // only fired when delivery was active - useless for a diagnostic, which is
    // needed most when the thing being diagnosed is switched off.
    VkImageMemoryBarrier sb;
    memset(&sb, 0, sizeof(sb));
    sb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    sb.srcQueueFamilyIndex = sb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    sb.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    sb.subresourceRange.levelCount = 1;
    sb.subresourceRange.layerCount = 1;
    sb.image         = swapImg;
    sb.oldLayout     = layout;
    sb.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    sb.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    sb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                          0, nullptr, 1, &sb);

    VkBufferImageCopy bic;
    memset(&bic, 0, sizeof(bic));
    bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bic.imageSubresource.layerCount = 1;
    bic.imageExtent.width = w; bic.imageExtent.height = h; bic.imageExtent.depth = 1;
    dd.cmdCopyImageToBuffer(cb, swapImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            g_shot.buf, 1, &bic);

    sb.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    sb.newLayout     = layout;                 // hand it back as we found it
    sb.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    sb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                          0, nullptr, 1, &sb);

    g_shot.armed = true;
    g_shot.wait  = 4;    // let it actually execute before reading
}

static VKAPI_ATTR void VKAPI_CALL Layer_CmdEndRendering(VkCommandBuffer cb)
{
    VkImage       swapTarget = VK_NULL_HANDLE;
    VkImageLayout swapLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    bool          wasScenePass = false;
    bool          mvBoundPass  = false;
    CbPassInfo    passInfo;
    {
        std::lock_guard<std::mutex> g(g_lock);
        wasScenePass = g_cbInScenePass[cb];
        g_cbInScenePass[cb] = false;
        std::map<VkCommandBuffer, bool>::iterator mb = g_cbMvBoundPass.find(cb);
        if (mb != g_cbMvBoundPass.end()) { mvBoundPass = mb->second; mb->second = false; }
        std::map<VkCommandBuffer, CbPassInfo>::iterator pit = g_cbPassInfo.find(cb);
        if (pit != g_cbPassInfo.end()) { passInfo = pit->second; g_cbPassInfo.erase(pit); }
        std::map<VkCommandBuffer, CbSwapTarget>::iterator st = g_cbSwapTarget.find(cb);
        if (st != g_cbSwapTarget.end()) {
            swapTarget = st->second.image;
            swapLayout = st->second.layout;
            g_cbSwapTarget.erase(st);
        }
    }
    if (g_nextCmdEndRendering) g_nextCmdEndRendering(cb);

    // ---- TAA RESOLVE, AFTER THE PASS HAS ACTUALLY ENDED.
    //
    // This was recorded BEFORE vkCmdEndRendering, which put a vkCmdDispatch and
    // several layout transitions inside an active render pass. Both are illegal
    // - compute cannot be dispatched inside a render pass and images cannot
    // change layout while attached - so the result was undefined and varied
    // from frame to frame, which is what "the blackness changes per frame"
    // describes.
    //
    // This is the only point where the colour target holds a finished frame and
    // the velocity target beside it describes that same frame. Running it at
    // present time instead would resolve a composited image whose HUD and panel
    // have no vectors, which is what makes UI ghost.
    if (wasScenePass) ++g_sceneEndsThisFrame;
    // Why the resolve is or is not running. Two crashes have now been blamed on
    // TAA while the log showed it never initialised, which is not a diagnosis.
    if (taaEnabled()) {
        static uint64_t gateLog = 0;
        if ((gateLog++ % 600) == 0)
            // bindAge is the only number here that is NOT mid-frame-racy:
            // mvBinds and ends are per-frame counters sampled during recording,
            // so "0" from them can mean "not yet this frame" as easily as
            // "never" - which is exactly the ambiguity that made the frozen
            // episodes hard to see. An age over 1 is stale regardless of where
            // in the frame it is read.
            trace("TAA GATE: scenePass=%d mvBinds=%d bindAge=%llu resolved=%d "
                  "gateDepth=%u ends=%u/%u ready=%d mvReady=%d mvView=%p "
                  "sceneImg=%p %ux%u",
                  wasScenePass ? 1 : 0, (int)g_mvBindsThisFrame,
                  (unsigned long long)(g_frameCount - g_mvLastBindFrame.load()),
                  g_taaResolvedThisFrame ? 1 : 0,
                  g_gateDepthLastFrame.load(),
                  g_sceneEndsThisFrame, g_sceneEndsLastFrame,
                  g_taa.ready ? 1 : 0, g_mv.ready ? 1 : 0,
                  (void*)g_mv.view, (void*)g_sceneColor.image,
                  g_sceneColor.w, g_sceneColor.h);
    }
    const bool lastScenePass =
        g_sceneEndsLastFrame == 0 || g_sceneEndsThisFrame >= g_sceneEndsLastFrame;
    // Resolve on the LIT pass: one full-size colour attachment, no depth
    // reload, and only after the velocity target has been bound this frame.
    // That excludes the G-buffer pass (colour=5, material data) and the cockpit
    // overlays (depthLoad=1, they run after tonemap).
    const bool litPass = passInfo.colorCount == 1 && !passInfo.depthLoad &&
                         passInfo.color0 != VK_NULL_HANDLE &&
                         passInfo.w == g_mv.w && passInfo.h == g_mv.h;
    if (taaEnabled()) {
        if (litPass) gateReach(1);
        else if (passInfo.colorCount == 1 && !passInfo.depthLoad &&
                 passInfo.color0 != VK_NULL_HANDLE &&
                 (passInfo.w != g_mv.w || passInfo.h != g_mv.h)) {
            // A pass that fails ONLY on dimensions is the silent way the whole
            // chain dies with every telemetry flag still reading healthy.
            static uint32_t lastW = 0, lastH = 0;
            if (passInfo.w != lastW || passInfo.h != lastH) {
                lastW = passInfo.w; lastH = passInfo.h;
                trace("TAA GATE: candidate rejected on SIZE alone - pass %ux%u "
                      "vs velocity target %ux%u",
                      passInfo.w, passInfo.h, g_mv.w, g_mv.h);
            }
        }
    }
    // ---- STAND ASIDE ON DUMP FRAMES.
    //
    // The resolve samples the velocity target, so it transitions g_mv.image to
    // SHADER_READ_ONLY and back. mvRecordReadback records its own barrier and
    // copy on the same image in the same command buffer with no coordination
    // between them - which is why the accuracy panel began reading BROKEN the
    // moment TAA was switched on while the picture itself stayed correct. The
    // vectors were never affected; the measurement of them was.
    //
    // The dump runs one frame in twenty (TAA_VELOCITY_DUMP), so yielding those
    // frames costs nothing visible and leaves one owner of the image per frame.
    // ---- BAIL OUT OF THE RESOLVE, NOT OUT OF THE FUNCTION.
    //
    // These guards used plain `return`, which returns from vkCmdEndRendering
    // itself and skips everything after it, including the swapchain capture.
    // Once the HDR-format check was added it fired on most passes, so the
    // function bailed nearly every frame and the screen went fully black. A
    // guard that rejects a pass must skip the RESOLVE only.
    // ---- IS THE FIELD ALIVE, OR A FOSSIL?
    //
    // The old gate here was g_mvBindsThisFrame > 0, and it failed in both
    // directions for the same reason: it is a per-frame counter sampled during
    // RECORDING, and X-Plane records command buffers on worker threads, so the
    // lit pass can be recorded before the scene pass regardless of submission
    // order. It answers "has the scene pass been recorded yet", not "is the
    // field current".
    //
    // The consequence was the frozen-field episodes: when the pass
    // identification loses the scene pass, nothing binds the velocity target,
    // and because the per-frame clear rides on the FIRST BIND, nothing clears
    // it either - the texture holds the last written motion indefinitely. The
    // resolve kept running through ~370-frame dropouts, reprojecting every
    // frame through motion from seconds ago. That stale field, consumed as
    // current, was the mirror, the mode-2 streaks, and "red ground with a
    // still camera".
    //
    // The bind site stamps the frame it bound on; one frame of tolerance
    // absorbs the recording-order race. Anything older is a fossil and the
    // only correct resolve is no resolve.
    const uint64_t mvBindAge = g_frameCount - g_mvLastBindFrame.load();
    // A stale velocity field means the resolve cannot run, but history is still
    // the best image we have - deliver it rather than shipping the raw frame.
    // See taaRecordDeliverOnly: the measured fault was delivery, not accumulation.
    if (litPass && taaEnabled() && !g_taaResolvedThisFrame && mvBindAge > 1 &&
        g_taa.ready && passInfo.color0 == g_taa.sceneImage) {
        std::map<VkCommandBuffer, VkDevice>::iterator dci = g_cbToDevice.find(cb);
        if (dci != g_cbToDevice.end()) {
            std::map<void*, DeviceData>::iterator ddi =
                g_devices.find(dispatchKey(dci->second));
            if (ddi != g_devices.end())
                taaRecordDeliverOnly(ddi->second, cb, passInfo.color0);
        }
    }
    if (litPass && taaEnabled() && !g_taaResolvedThisFrame && mvBindAge > 1) {
        static uint64_t nStaleSkip = 0;
        if ((++nStaleSkip % 300) == 1)
            trace("TAA: velocity field is STALE (last bound %llu frame(s) ago) "
                  "- resolve skipped, history will reset on resume. The field "
                  "froze because no pass bound the velocity target, which is "
                  "the pass-identification dropout, not a TAA fault.",
                  (unsigned long long)mvBindAge);
        g_taaStaleResume = true;
    }
    // NOT gated on g_taaResolvedThisFrame: one present interval often carries
    // two recorded frames (the MISS census), and a once-per-interval flag
    // silently dropped the second frame's resolve. The modulo boundary below
    // is the per-frame selector now; the flag stays as telemetry only.
    // ---- AT MOST ONE RESOLVE PER PRESENTED FRAME.
    //
    // The modulo gate below fires at EVERY multiple of the per-frame pass
    // count, which was right for catching frames X-Plane records ahead - but
    // nothing capped how many times that could happen inside one present, and
    // the once-per-frame guard had been dropped from this condition. Each extra
    // pass is a full-resolution compute dispatch over 8.3 M pixels at 4K, and
    // the sim measured 38 -> 19 fps when the clamp stage was switched on: the
    // resolve was costing ~26 ms, which is several dispatches, not one.
    //
    // Only one frame is ever displayed per present, so only one resolve can
    // reach the screen; the rest are work nobody sees. The counter says how
    // many were being run, so the saving is visible rather than assumed, and
    // taa.max_resolves raises the cap if a frame ever genuinely needs more.
    static uint32_t resolvesThisPresent = 0;
    static uint64_t resolvePresentTag = ~0ull;
    if (resolvePresentTag != g_frameCount) {
        if (resolvesThisPresent > 1) {
            static uint32_t worst = 0;
            if (resolvesThisPresent > worst) {
                worst = resolvesThisPresent;
                trace("TAA COST: %u resolves ran in one presented frame - each "
                      "is a full-resolution dispatch and only one can be seen. "
                      "Capping at taa.max_resolves.", resolvesThisPresent);
            }
        }
        resolvesThisPresent = 0;
        resolvePresentTag = g_frameCount;
    }
    const uint32_t maxResolves =
        (uint32_t)live::i("taa.max_resolves", "TAA_MAX_RESOLVES", 1);
    // ---- A FRAME WITH NO VALID REPROJECTION MUST NOT BE RESOLVED.
    //
    // The plugin sets reprojValid = 0 when it cannot invert the current
    // view-projection, and leaves reproj holding whatever it last had - the
    // identity, if the inverse has never succeeded. Nothing here ever read the
    // flag, so those frames were resolved against a matrix asserting that
    // nothing moved: every vector zero, history fetched from the wrong place
    // for the entire screen, the clamp dragging it back a huge distance, and
    // the weight map going fully red the instant the camera moved. Measured
    // directly - "MV REPROJ INVALID" beside a camera that moved 60 m.
    //
    // Skipping is the honest answer: one frame without temporal accumulation
    // is invisible, whereas one frame reprojected through a lie poisons the
    // history that every later frame builds on. The stale-resume path already
    // exists for exactly this, so history resets cleanly when vectors return.
    const bool reprojUsable = !g_share || g_share->reprojValid != 0;
    if (!reprojUsable) g_taaStaleResume = true;
    do if (litPass && taaEnabled() && !g_mv.wantDump && reprojUsable &&
        resolvesThisPresent < maxResolves &&
        mvBindAge <= 1) {
        gateReach(2);
        std::map<VkCommandBuffer, VkDevice>::iterator tci = g_cbToDevice.find(cb);
        if (tci != g_cbToDevice.end()) {
            std::map<void*, DeviceData>::iterator tdi = g_devices.find(dispatchKey(tci->second));
            if (tdi != g_devices.end() && g_mv.ready && g_mv.view != VK_NULL_HANDLE) {
                gateReach(3);
                DeviceData &tdd = tdi->second;
                // ---- CRASH DESTRUCTION RESOURCES.
                //
                // Created here because this is the one place that reliably has
                // both a DeviceData and the device handle, on a path that runs
                // every frame. ensure() is a no-op after the first success and
                // latches on failure, so a device that cannot support it is
                // asked once and never again.
                //
                // Nothing binds this yet - the buffer exists and is described,
                // which is deliberately the whole of this step. Adding a
                // descriptor set to every pipeline is the invasive part and is
                // easier to judge when creation is already known good.
                if (crashEnabled()) destructgpu::ensure(tdd, tci->second, g_maxBoundSets);
                // Re-init only on a real change of shape. The scene IMAGE
                // alternates every frame between two targets, and keying on it
                // rebuilt everything each frame and destroyed objects still in
                // use - only the view needs to follow.
                // This pass's OWN attachment, resolved through the view it
                // was bound with. g_sceneColor points at a different image.
                VkFormat litFmt = VK_FORMAT_UNDEFINED;
                uint32_t litLayers = 1;
                VkSampleCountFlagBits litSamples = VK_SAMPLE_COUNT_1_BIT;
                {
                    std::lock_guard<std::mutex> g(g_lock);
                    std::map<VkImage, ColorTarget>::iterator lci =
                        g_colorImages.find(passInfo.color0);
                    if (lci != g_colorImages.end()) {
                        litFmt     = lci->second.format;
                        litLayers  = lci->second.arrayLayers;
                        litSamples = lci->second.samples;
                    }
                }
                // ---- THE TARGET MUST BE THE HDR SCENE IMAGE, NOT MERELY A
                //      FULL-SIZE SINGLE-ATTACHMENT PASS.
                //
                // "first full-size colour=1 pass after the G-buffer" matched a
                // different pass depending on the camera view, and in the views
                // where it matched something else - a reflection or an LDR
                // composite - the TAA result was written over that buffer and
                // whatever consumed it produced black. Hence black in some
                // views and not others.
                //
                // Requiring the HDR float format excludes the LDR targets, and
                // logging the choice makes a wrong pick visible instead of
                // silent.
                if (litFmt != VK_FORMAT_R16G16B16A16_SFLOAT &&
                    litFmt != VK_FORMAT_R32G32B32A32_SFLOAT) break;
                gateReach(4);
                // The flashes diagnosis: if HDR candidate passes keep ENDING
                // after the resolve already ran this frame, the engine paints
                // over the resolve's output - same-image counts are direct
                // overwrites, other-image counts mean the pick landed on the
                // wrong half of a double-buffered pair.
                if (g_taaResolvedThisFrame) {
                    if (passInfo.color0 == g_taaWroteImageThisFrame)
                        ++g_hdrAfterResolveSame;
                    else
                        ++g_hdrAfterResolveOther;
                }
                {
                    // CANDIDATE, not chosen: this fires BEFORE the last-pass
                    // filter below, so a frame with two qualifying passes
                    // prints both and reads as the target alternating every
                    // frame - which is exactly the false alarm it produced.
                    // The CHOSEN line after the filter is the one that means
                    // what this one used to claim.
                    static VkImage lastPick = VK_NULL_HANDLE;
                    if (passInfo.color0 != lastPick) {
                        lastPick = passInfo.color0;
                        trace("TAA CANDIDATE: %p fmt=%d %ux%u "
                              "(colour=%u depthLoad=%d)",
                              (void*)passInfo.color0, (int)litFmt,
                              passInfo.w, passInfo.h, passInfo.colorCount,
                              passInfo.depthLoad ? 1 : 0);
                    }
                }
                if (litFmt == VK_FORMAT_UNDEFINED) break;

                // ---- THE LAST HDR PASS OF THE FRAME, NOT THE FIRST.
                //
                // Everything above only establishes that this pass is a
                // CANDIDATE. Several passes qualify, and the shader corpus says
                // exactly which:
                //
                //   deferred_gbuf   the deferred LIGHTING pass, writes the lit
                //                   HDR attachment
                //   light           sprite/billboard lights, additive, SAME
                //                   attachment
                //   volumetric_apply  fog + cloud composite, same attachment
                //   rain, particle, dome, atmosphere, ocean_shading
                //                   all forward-composited into it too
                //
                // Latching on the first match resolved straight after lighting,
                // so every one of those was painted ON TOP of the TAA output,
                // un-antialiased, and never entered the history. The
                // accumulation was built from a picture that is not the one on
                // screen.
                //
                // The last candidate cannot be known until it has happened, so
                // count them and use the previous frame's total: the pass order
                // does not change between frames of the same configuration, and
                // when it does change - a camera cut, a settings change - the
                // count moves by one frame and self-corrects. The identical
                // trick already drives lastScenePass above.
                //
                // Frame 1 has no previous count, so nothing resolves; frame 2
                // onward does. That is one frame without TAA at startup, which
                // is invisible and strictly better than a frame resolved into
                // the wrong pass.
                const uint32_t hdrIdx = ++g_hdrPassesThisFrame;
                // MODULO, not equality: X-Plane records ahead, so one present
                // interval often carries TWO frames' passes (hdr=6 then hdr=0
                // the next interval - the TAA MISS census measured it). With
                // equality, the second frame's last pass (idx 6 != 3) never
                // resolved and every other displayed frame shipped raw - the
                // 54% duty and the flashes. Every multiple of the per-frame
                // count is a frame boundary; resolve at each of them.
                // ---- LATCH THE TARGET; THE ORDINAL ONLY ACQUIRES IT.
                //
                // The modulo above keeps the CADENCE right but says nothing
                // about WHICH image, and the count is not stable: the trace
                // shows "hdr 3 of 3" and "hdr 4 of 4" on consecutive frames,
                // resolving into two DIFFERENT images. Accumulating into two
                // images in turn means every other frame reprojects history
                // belonging to the other one - that is the whole-scene wobble,
                // still present with jitter at zero, and the crawl on
                // high-frequency surfaces for the same reason.
                //
                // So the ordinal ACQUIRES a target and the image is then HELD.
                // A latched target is re-accepted on sight whatever ordinal it
                // carries, and released only after it has not appeared for a
                // while - destroyed, or the view changed shape - at which point
                // the ordinal picks the next one. Same shape as the pager's
                // sticky decision: derive once, then hold, so frames agree.
                // ---- TRIED, AND IT WAS THE WRONG READING. (off by default)
                //
                // Holding one image made the picture SMEAR heavily, which
                // settles the question the comment below poses: the alternation
                // is X-Plane double-buffering its lit target, not the pick
                // drifting between passes. Both images are legitimate scene
                // targets, so pinning to one resolves a buffer holding the
                // PREVIOUS frame every other frame and blends that forward -
                // textbook smear. The wobble has another cause and the ordinal
                // path is correct after all. Kept behind TAA_TARGET_LATCH
                // because the experiment is worth being able to repeat.
                static const bool latchOn = getenv("TAA_TARGET_LATCH") != nullptr;
                static VkImage  latchTarget = VK_NULL_HANDLE;
                static uint64_t latchSeen   = 0;
                const bool latchFresh = latchOn && latchTarget != VK_NULL_HANDLE &&
                                        (g_frameCount - latchSeen) < 120;
                if (latchFresh) {
                    if (passInfo.color0 != latchTarget) break;
                    latchSeen = g_frameCount;
                } else {
                    if (g_hdrPassesLastFrame == 0 ||
                        (hdrIdx % g_hdrPassesLastFrame) != 0)
                        break;
                    if (latchTarget != passInfo.color0)
                        trace("TAA LATCH: holding target %p (hdr %u of %u). The "
                              "pass count alternates, so re-deriving the target "
                              "every frame split history between two images - "
                              "the whole-scene wobble.",
                              (void*)passInfo.color0, hdrIdx,
                              g_hdrPassesLastFrame);
                    latchTarget = passInfo.color0;
                    latchSeen   = g_frameCount;
                }
                gateReach(5);

                {
                    static VkImage lastChosen = VK_NULL_HANDLE;
                    if (passInfo.color0 != lastChosen) {
                        lastChosen = passInfo.color0;
                        trace("TAA CHOSEN: resolving into %p (hdr %u of %u, "
                              "sceneEnds=%u mvBinds=%d) - alternation HERE is "
                              "real: either X-Plane double-buffers the lit "
                              "target (benign, history still carries) or the "
                              "pick is drifting between two different passes",
                              (void*)passInfo.color0, hdrIdx,
                              g_hdrPassesLastFrame, g_sceneEndsThisFrame,
                              (int)g_mvBindsThisFrame);
                    }
                }

                // ---- DESCRIBE THE FRAME, THEN ASK THE BACKEND.
                //
                // Everything the resolve needs is assembled into one
                // temporal::TemporalFrame and the backend decides for itself
                // whether it can act. That is the whole point of the split: the
                // layer knows HOW TO OBTAIN these resources and nothing about
                // what any consumer does with them, and the backend knows what
                // they mean and nothing about X-Plane.
                //
                // It also puts the shape check where it belongs. The target's
                // layer count and sample count travel WITH the image rather than
                // being looked up beside it, which is precisely the information
                // that was missing when a plain 2D view of an arrayed target
                // silently reached one eye.
                temporal::TemporalFrame tf;
                tf.color.image   = passInfo.color0;
                tf.color.format  = litFmt;
                tf.color.w       = passInfo.w;
                tf.color.h       = passInfo.h;
                tf.color.layers  = litLayers;
                tf.color.samples = litSamples;
                tf.motion.image  = g_mv.image;
                tf.motion.view   = g_mv.viewArray;   // array-typed; see MvTarget
                tf.motion.w      = g_mv.w;
                tf.motion.h      = g_mv.h;
                tf.motion.layers = 1;
                // Our convention, declared rather than assumed. The 0.5 the
                // vertex shader applies IS the NDC-to-UV conversion, the stored
                // value is prev minus curr, and jitter is applied after the
                // varyings are written so it is not baked in.
                tf.motion.coordinateSpace      = temporal::COORD_UV;
                tf.motion.direction            = temporal::DIR_PREVIOUS_TO_CURRENT;
                tf.motion.jitterIncluded       = false;
                tf.motion.cameraMotionIncluded = true;
                tf.motion.objectMotionIncluded = true;
                tf.renderW = tf.outputW = passInfo.w;
                tf.renderH = tf.outputH = passInfo.h;
                // The APPLIED NDC values, not g_velSnap's pixel-unit request -
                // the shader adds pc.jitter as an NDC offset, so it must be
                // told the same number the vertex splice consumed. And from
                // THIS command buffer's pending-push slot, not the globals:
                // X-Plane records several frames on several threads at once,
                // so by the time this resolve records, the globals can already
                // hold the NEXT frame's offset - the same straddle MV PUSH
                // RACE counts. The slot is thread-local and keyed by this cb,
                // which is the recording the raster was actually displaced in.
                // ---- DERIVE IT, DO NOT FETCH IT.
                //
                // The per-cb slot lookup SUCCEEDS and hands back the zeroed
                // block: measured 450 zero jitters out of 450 resolves. The
                // resolve is not recorded in the command buffer that carried
                // the geometry pushes, so its slot never held the offset. S was
                // therefore always zero and the unjitter has never once run -
                // which is why no sign of it changed anything, why toggling
                // taa.unjitter did nothing, and why only jitter_scale=0 ever
                // stopped the shake.
                //
                // The applied offset is a pure function of the frame snapshot
                // and the target size - the same expression the vertex push
                // uses - so compute it from g_velSnap, which this resolve is
                // already using for the reprojection and is therefore
                // consistent with by construction.
                {
                    const float ySignR = g_viewportYFlipped ? -1.0f : 1.0f;
                    const float wR = (float)g_taa.w, hR = (float)g_taa.h;
                    if (wR > 0.0f && hR > 0.0f) {
                        tf.jitter.x = 2.0f * g_velSnap.jitterX * g_jitterScale / wR;
                        tf.jitter.y = ySignR * 2.0f * g_velSnap.jitterY * g_jitterScale / hR;
                    } else {
                        tf.jitter.x = tf.jitter.y = 0.0f;
                    }
                }
                // The per-cb slot is left in place as INSTRUMENTATION only:
                // it proved the zero (450 of 450) and stays useful if the
                // recording order ever changes. It no longer feeds the shader.
                { float px_, py_;
                  if (mvPendingJitter(cb, &px_, &py_)) ++g_jitSlotHit;
                  else                                 ++g_jitSlotMiss; }
                if (tf.jitter.x == 0.0f && tf.jitter.y == 0.0f) ++g_jitZero;
                else                                            ++g_jitNonZero;
                g_jitLastX = tf.jitter.x; g_jitLastY = tf.jitter.y;
                memcpy(tf.camera.reproj, g_velSnap.reproj, sizeof(tf.camera.reproj));
                // Not camDelta: that is translation only, and a camera rotating
                // in place moves every pixel while translating zero. The
                // reprojection matrix is the identity exactly when nothing
                // moved, whatever the motion was made of.
                // ---- 1e-6 CALLS A PARKED CAMERA "MOVING".
                //
                // The matrix is built in float32 from datarefs that are
                // themselves float32, so its identity case carries noise around
                // 1e-7..1e-5 - permanently above a 1e-6 threshold. cameraMoved
                // therefore reads TRUE with the aircraft parked and the view
                // still, and downstream that is not a cosmetic error: the
                // unwritten-pixel test is `cameraMoved && vel == 0`, and a
                // stationary world writes exactly zero everywhere, so the whole
                // frame is declared unwritten and its history is discarded
                // every frame. That is the missing accumulation, the missing
                // antialiasing, and the shake.
                //
                // The threshold that matters is one PIXEL, not one epsilon. The
                // reprojection maps NDC to NDC, so a term of e displaces a
                // pixel by e * width / 2; at 3840 one pixel is 5.2e-4. A
                // default of 1e-4 is a fifth of a pixel - far below anything
                // visible, far above the float noise floor.
                const float movedEps = live::f("taa.moved_eps", "TAA_MOVED_EPS", 1e-4f);
                for (int mi = 0; mi < 16 && !tf.camera.moved; ++mi) {
                    const float ident = (mi % 5) == 0 ? 1.0f : 0.0f;
                    if (fabsf(g_velSnap.reproj[mi] - ident) > movedEps)
                        tf.camera.moved = true;
                }

                const char *why = "";
                if (!g_taaBackend.accepts(tf, &why)) {
                    // Say WHY, once per distinct reason. A backend silently
                    // absent is indistinguishable from a backend silently
                    // broken, and this project has already paid for that once.
                    static const char *lastWhy = nullptr;
                    if (why != lastWhy) {
                        lastWhy = why;
                        trace("TAA: DECLINED %p (%ux%u, %u layer(s), %u sample(s)) "
                              "- %s. Leaving the frame untouched is correct; "
                              "writing through a descriptor that does not match "
                              "the target is what made this black in some camera "
                              "views.",
                              (void*)passInfo.color0, passInfo.w, passInfo.h,
                              litLayers, (unsigned)litSamples, why);
                    }
                    break;
                }
                gateReach(6);

                // Quiesce after any scene-sized destruction (the view-snap
                // crash): skip the resolve entirely and resume with a reset.
                if (g_taaQuiesce.load() > 0) {
                    g_taaQuiesce.fetch_sub(1);
                    g_taaStaleResume = true;
                    // Deliver the accumulated image anyway. Skipping the
                    // DISPATCH is the point of a quiesce; skipping the COPY
                    // just ships a raw frame, and a raw frame between resolved
                    // ones is the alternation the history/composited split
                    // measured (0.447 against 3.78).
                    taaRecordDeliverOnly(tdd, cb, passInfo.color0);
                    break;
                }
                gateReach(7);
                // A regenerated velocity target needs NO rebuild: descriptors
                // are rewritten every dispatch, so repointing the view in
                // place is the whole fix - zero hitch. (The earlier full
                // re-init here was the 19fps stutter.) The reset flag matters:
                // a new target usually means a scene change, and blending
                // history across one drags the old scene forward.
                if (g_taa.ready && g_taa.velGen != g_mv.gen) {
                    g_taa.velView = g_mv.viewArray;
                    g_taa.velGen  = g_mv.gen;
                    tf.reset |= temporal::RESET_SCENE_LOAD;
                    trace("TAA: velocity view repointed in place (gen %llu)",
                          (unsigned long long)g_mv.gen);
                }
                const bool needInit = !g_taa.ready ||
                                      g_taa.w != passInfo.w ||
                                      g_taa.h != passInfo.h ||
                                      g_taa.layers != litLayers ||
                                      g_taa.format != litFmt;
                if (needInit) {
                    // Re-init storms are the 19fps dips: each init rebuilds the
                    // pipeline and views (a hitch) and parks a state in the
                    // graveyard. One init per 60 frames is plenty - if the
                    // trigger is real it still happens, just once; while
                    // throttled the resolve skips rather than thrashes.
                    static uint64_t lastInitFrame = 0;
                    if (g_taa.ready && g_frameCount - lastInitFrame < 600) {
                        static uint64_t thrLog = 0;
                        if ((thrLog++ % 120) == 0)
                            trace("TAA: re-init THROTTLED (last init %llu "
                                  "frames ago) - resolve skipped this frame",
                                  (unsigned long long)(g_frameCount - lastInitFrame));
                        break;
                    }
                    lastInitFrame = g_frameCount;
                    // A shape change is a history discontinuity in its own right.
                    tf.reset |= temporal::RESET_RESOLUTION;
                    taaInit(tdd, tci->second, passInfo.color0, litFmt,
                            passInfo.w, passInfo.h, litLayers, g_mv.viewArray);
                    // Stamp which velocity-target generation the descriptors
                    // were just written against - the needInit clause above
                    // compares this, not the reusable handle value.
                    g_taa.velGen = g_mv.gen;
                    trace("TAA: descriptors bound to velocity gen %llu (view %p)",
                          (unsigned long long)g_mv.gen, (void*)g_mv.viewArray);
                } else if (!taaBindScene(tdd, passInfo.color0)) {
                    break;
                }
                gateReach(8);
                // A target swap is a history discontinuity: the accumulated
                // image belongs to the old one and reprojecting into it would
                // drag a whole frame of the wrong picture forward.
                // A target swap is a history discontinuity: the accumulated
                // image belongs to the old one, and reprojecting into it would
                // drag a whole frame of the wrong picture forward.
                if (needInit) tf.reset |= temporal::RESET_TARGET_SWAP;
                // Resuming after a stale-field skip is a discontinuity too:
                // whatever is in history was accumulated before the freeze,
                // and blending it into the first live frame drags the fossil
                // forward one last time.
                if (g_taaStaleResume) {
                    tf.reset |= temporal::RESET_SCENE_LOAD;
                    g_taaStaleResume = false;
                    trace("TAA: field is live again - history reset (%s)",
                          temporal::resetReasonName(tf.reset));
                }
                g_taaDevice = &tdd;
                // Hand the resolve X-Plane's own moving-geometry flags if the
                // census has identified them. Shape-checked in taaBindFlags.
                {
                    // The census identifies candidates at image CREATION, and
                    // X-Plane creates its G-buffer at startup - before our
                    // velocity target exists, when the size comparison inside
                    // noteGbufferVelCandidate cannot succeed. Sweep the images
                    // already recorded once the target is real; the sweep is at
                    // most 256 map entries and stops mattering the moment a
                    // candidate is found.
                    if (g_gbufferVelCandidate == VK_NULL_HANDLE && g_mv.ready) {
                        std::lock_guard<std::mutex> g(g_lock);
                        for (std::map<VkImage, ColorTarget>::iterator ci2 =
                                 g_colorImages.begin();
                             ci2 != g_colorImages.end() &&
                                 g_gbufferVelCandidate == VK_NULL_HANDLE; ++ci2)
                            noteGbufferVelCandidate(ci2->second);
                    }
                    VkImage fi = g_gbufferVelCandidate;
                    VkFormat ff = VK_FORMAT_UNDEFINED;
                    uint32_t fl = 1;
                    VkSampleCountFlagBits fsm = VK_SAMPLE_COUNT_1_BIT;
                    if (fi != VK_NULL_HANDLE) {
                        std::lock_guard<std::mutex> g(g_lock);
                        std::map<VkImage, ColorTarget>::iterator fc =
                            g_colorImages.find(fi);
                        if (fc != g_colorImages.end()) {
                            ff  = fc->second.format;
                            fl  = fc->second.arrayLayers;
                            fsm = fc->second.samples;
                        } else {
                            fi = VK_NULL_HANDLE;
                        }
                    }
                    taaBindFlags(tdd, fi, ff, fl, fsm);
                    // Lifetime ledger for the deferred-destroy protection:
                    // this engine image is now referenced by a dispatch that
                    // may execute up to a few frames from now.
                    if (fi != VK_NULL_HANDLE) {
                        std::lock_guard<std::mutex> g(g_lock);
                        g_taaBoundImgs[fi] = g_frameCount;
                    }
                }
                {
                    std::lock_guard<std::mutex> g(g_lock);
                    g_taaBoundImgs[passInfo.color0] = g_frameCount;
                }
                g_taaBackend.record(cb, tf);
                g_taaResolvedThisFrame = true;

                // ---- CLEAR THE VELOCITY TARGET HERE, NOT IN A RENDER PASS.
                //
                // The in-pass clear picks whichever scene pass is RECORDED
                // first and makes it LOAD_OP_CLEAR. X-Plane records passes on
                // several threads, so the pass that wins the flag is not
                // necessarily the pass that EXECUTES first - and when it is
                // not, its clear wipes velocities another pass already wrote.
                // That is why the terrain reads as unwritten while trees and
                // buildings, drawn after the clear, survive: the sentinel viz
                // is red across the ground and yellow on exactly the geometry
                // that happens to be drawn later.
                //
                // Clearing immediately after the resolve has consumed the
                // target removes the ordering question entirely: there is one
                // clear, at one point, and every pass then LOADs. The flag
                // below is what the pass hook already reads to stop clearing -
                // it was declared and read but never set, so the racy path was
                // the only one that ever ran.
                // Live, because the two strategies win in different views: the
                // post-resolve clear fixed the cockpit and emptied the external
                // view, which means the resolve does not sit at the same point
                // in the frame in both.
                if (g_mv.image && tdd.cmdClearColorImage &&
                    live::onoff("taa.clear_after_resolve", "TAA_CLEAR_AFTER_RESOLVE", false)) {
                    VkImageMemoryBarrier mb;
                    memset(&mb, 0, sizeof(mb));
                    mb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    mb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    mb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    mb.image = g_mv.image;
                    mb.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                    mb.subresourceRange.levelCount = 1;
                    mb.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
                    mb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    mb.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    mb.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    tdd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                          0, nullptr, 0, nullptr, 1, &mb);

                    VkClearColorValue cv;
                    memset(&cv, 0, sizeof(cv));
                    cv.float32[0] = kMvUnwritten;
                    cv.float32[1] = kMvUnwritten;
                    VkImageSubresourceRange rng = mb.subresourceRange;
                    tdd.cmdClearColorImage(cb, g_mv.image,
                                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                          &cv, 1, &rng);

                    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    mb.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                    mb.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    mb.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    tdd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                                          0, nullptr, 0, nullptr, 1, &mb);
                    g_mvClearedAtPresent.store(true);
                } else if (g_mv.image) {
                    // Hand the clear back to the render passes.
                    g_mvClearedAtPresent.store(false);
                }
                ++resolvesThisPresent;
                gateReach(9);
                // Watched by noteSsrFeedbackCheck for the rest of the frame.
                g_taaWroteImageThisFrame = passInfo.color0;
            }
        }
    } while (0);


    // Capture whatever is on screen, whether or not any of our delivery ran.
    if (swapTarget != VK_NULL_HANDLE) {
        DeviceData *sdd = nullptr;
        SwapInfo    sinf;
        bool        haveInfo = false;
        {
            std::lock_guard<std::mutex> g(g_lock);
            std::map<VkCommandBuffer, VkDevice>::iterator ci = g_cbToDevice.find(cb);
            if (ci != g_cbToDevice.end()) {
                std::map<void*, DeviceData>::iterator di = g_devices.find(dispatchKey(ci->second));
                if (di != g_devices.end()) sdd = &di->second;
            }
            haveInfo = swapInfoFor(swapTarget, sinf);
        }
        if (sdd && haveInfo && sdd->cmdCopyImageToBuffer)
            shotMaybe(*sdd, cb, swapTarget, sinf.w, sinf.h, swapLayout);
    }

}

// THE INJECTION POINT.
//
// vkEndCommandBuffer, not vkCmdEndRendering. At this moment the application has
// finished recording the buffer but has not closed it, so commands appended
// here are guaranteed to run after everything else in it - including the final
// depth write - without needing to know which pass was last.
//
// That question turned out to be unanswerable the way it was being asked:
// X-Plane records on multiple threads, so a global pass counter is racy, and
// the previous version silently never fired.
//
// Recording into the app's own buffer also fixes what the present-time submit
// got wrong: the queue is whichever they submit this buffer to, ordering
// follows from position in the buffer, and the layout is the one observed when
// the pass began.
static VKAPI_ATTR VkResult VKAPI_CALL Layer_EndCommandBuffer(VkCommandBuffer cb)
{
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    bool wants = false;
    DeviceData dd;
    bool haveDd = false;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<VkCommandBuffer, CbDepthUse>::iterator ui = g_cbDepthUse.find(cb);
        if (ui != g_cbDepthUse.end() && ui->second.used) {
            wants  = true;
            layout = ui->second.layout;
            ui->second.used = false;          // consumed; re-proven each recording
            ui->second.depthPasses = 0;
        }
        // A command buffer is about to be closed and may be re-recorded next
        // frame. Both resolve decisions must be re-earned rather than carried
        // over, or a reused buffer would skip the resolve because a previous
        // recording had already done it.
        {
            // Carry this recording's scene-pass count forward as next
            // recording's target, then start counting again. This is what makes
            // the boundary land after the LAST scene pass rather than the first
            // depth-less one.
            uint32_t n = g_cbScenePassCount.count(cb) ? g_cbScenePassCount[cb] : 0;
            if (n > 0) g_cbScenePassPrev[cb] = n;
            g_cbScenePassCount[cb] = 0;

            // Report this buffer's shape as it closes. S=scene (full viewport
            // with depth), P=full viewport without depth, o=everything else.
            // What matters is whether S and P appear in the same buffer, and
            // whether S ever appears in more than one.
            if (g_cbDumpOn && g_cbDumpsLeft > 0) {
                std::map<VkCommandBuffer, CbPassLog>::iterator pi = g_cbPassLog.find(cb);
                if (pi != g_cbPassLog.end() && !pi->second.seq.empty()) {
                    --g_cbDumpsLeft;
                    trace("CBDUMP cb=%p  %s", (void*)cb, pi->second.seq.c_str());
                    pi->second.seq.clear();
                }
            }

            g_cbSawScenePass[cb]   = false;
            g_cbResolvedThisCb[cb] = false;
            g_cbInScenePass[cb]    = false;
        }
        if (wants) {
            std::map<VkCommandBuffer, VkDevice>::iterator ci = g_cbToDevice.find(cb);
            if (ci != g_cbToDevice.end()) {
                std::map<void*, DeviceData>::iterator di = g_devices.find(dispatchKey(ci->second));
                if (di != g_devices.end()) { dd = di->second; haveDd = true; }
            }
        }
    }

    // The depth-derived pass used to be recorded into X-Plane's own command
    // buffer here. It is gone; the injected shaders write velocity as part of
    // the scene draw itself, so there is nothing to inject and nothing to hold
    // stable before injecting it.
    (void)wants;

    PFN_vkEndCommandBuffer next = haveDd ? dd.endCommandBuffer : nullptr;
    if (!next) {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<VkCommandBuffer, VkDevice>::iterator ci = g_cbToDevice.find(cb);
        if (ci != g_cbToDevice.end()) {
            std::map<void*, DeviceData>::iterator di = g_devices.find(dispatchKey(ci->second));
            if (di != g_devices.end()) next = di->second.endCommandBuffer;
        }
    }
    return next ? next(cb) : VK_SUCCESS;
}


// ---------------------------------------------------------------- arming
//
// Every one-time switch the layer reads from the environment, in ONE place
// that runs regardless of what else is enabled.
//
// This used to live inside the velocity pass's lazy initialiser, purely
// because that was a convenient spot when it was written. That created a
// dependency nobody would ever guess at: turning off TAA_VELOCITY - the
// depth-derived compute pass, which the SPIR-V injection made redundant -
// silently disabled the JITTER, the RESOLVE, FSR2, the cockpit reprojection
// and the pass dump as well. The layer attached, logged, and did nothing, and
// the image was X-Plane's own output with no sign anything was wrong.
//
// It also poisoned a measurement: disabling the velocity pass appeared to be
// worth +15 fps and every stutter, and that was the whole layer switching off
// rather than one dispatch.
static void armLayerOnce()
{
    static bool done = false;
    if (done) return;
    done = true;

            // 120 frames is ~2-4 s: frequent enough to spot-check, rare
            // enough that the readback stall is not felt. Override with
            // TAA_VELOCITY_DUMP=<frames>; 10 is usable for a short capture
            // but costs several stalls per second and ~4 MB/s of disk.
            // TAA_VELOCITY_DUMP=0 disables dumping entirely, including the
            // startup burst. Necessary for a clean flash test: the readback
            // copies 28 MB per dump and three of those firing right after a
            // load would add stutter that has nothing to do with what is
            // being measured.
            dumpEvery = 0;
            if (const char *d = getenv("TAA_VELOCITY_DUMP")) dumpEvery = atoi(d);

            // ---- "WILL LATER BE GATED ON A RESOLVE BEING PRESENT." IT IS NOW.
            //
            // On its own jitter makes the image worse: it shifts the sample
            // grid every frame with nothing accumulating the result, so
            // high-contrast edges crawl. With the resolve accumulating, that
            // same shifted grid is the entire mechanism by which temporal
            // anti-aliasing gets samples a single frame never had.
            //
            // So the two are tied. TAA_JITTER still forces it on for
            // measurement, but the resolve is what arms it in normal use, and
            // neither can be left on without the other by accident.
            // TAA is gone, so the condition this was tied to is gone with it.
            // Jitter alone shifts the sample grid every frame with nothing
            // accumulating the result, which makes high-contrast edges crawl -
            // strictly worse than none. It stays available for measurement and
            // off otherwise.
            g_jitterArmed = (getenv("TAA_JITTER") != nullptr);
            g_jitterViewport = (getenv("TAA_JITTER_VIEWPORT") != nullptr);
            if (const char *nf = getenv("TAA_NEARFIELD_M")) {
                g_nearFieldM = (float)atof(nf);
                if (g_nearFieldM < 0.0f)  g_nearFieldM = 0.0f;
                if (g_nearFieldM > 50.0f) g_nearFieldM = 50.0f;
            }
            trace("NEAR FIELD: geometry closer than %.2f m reprojects as "
                  "body-fixed (velocity zero) instead of world-fixed. %s "
                  "Set TAA_NEARFIELD_M to change it, 0 to disable.",
                  g_nearFieldM,
                  g_nearFieldM > 0.0f
                      ? "Active only once the plugin reports the camera rigid in "
                        "the body frame; off in external views."
                      : "DISABLED.");
            if (const char *lg = getenv("TAA_LEDGER"))
                g_ledgerOn = (lg[0] != '0');
            trace("LEDGER: per-resource accounting is %s. %s", 
                  g_ledgerOn ? "ON" : "OFF",
                  g_ledgerOn
                      ? "Every image and buffer creation takes the global "
                        "lock and queries memory requirements - set "
                        "TAA_LEDGER=0 to measure what that costs."
                      : "Resource creation is untouched; the VRAM ledger "
                        "and geometry histogram will report nothing.");
            if (const char *cp = getenv("TAA_COCKPIT_PASS")) {
                g_cockpitPassIndex = atoi(cp);
                trace("COCKPIT: scene pass %d will be reprojected in the "
                      "AIRCRAFT BODY frame instead of the world frame. "
                      "Cockpit surfaces travel with the camera, so the "
                      "world matrix claims most of a screen of motion where "
                      "the true value is near zero.", g_cockpitPassIndex);
            } else {
                trace("COCKPIT: body-frame reprojection OFF. Set "
                      "TAA_COCKPIT_PASS=<n> once TAA_CB_DUMP has shown "
                      "which scene pass draws the cockpit.");
            }
            if (g_jitterArmed)
                trace("JITTER: ARMED - %s. Expect crawling edges until the "
                      "resolve consumes it.",
                      g_jitterViewport
                          ? "LEGACY viewport offset, every full-size draw in a "
                            "3D pass including full-screen ones"
                          : "clip-space offset in the patched vertex shaders, "
                            "geometry pipelines inside a 3D pass only");

            if (getenv("TAA_CB_DUMP")) {
                g_cbDumpOn = true;
                g_cbDumpsLeft = 40;   // a couple of frames' worth, then quiet
                trace("CBDUMP: logging per-command-buffer pass structure. "
                      "S=full-viewport+depth (scene), P=full-viewport no depth, "
                      "o=other. Question: do S and P share a buffer, and does "
                      "S span more than one?");
            }


            // SPIR-V injection is NOT armed here - see TAA_CreateShaderModule.
            // Shader modules are created during load, before any present,
            // so anything armed in this function arrives too late for them.

            // ---- HEADROOM WAS LOST IN THE PORT FROM V1.
            //
            // V1 ran with TAA_PAGER_HEADROOM_MB=200 but the port never parsed
            // it, leaving the 1024 MB default. That is a five-fold difference
            // in when the pager engages: at 1024 it trips almost at once, at
            // 200 only under real pressure. The rest of V1's settings were
            // carried over as code but never set in V2's run script, so the
            // whole mip-drop path has been dormant here since the port.
            if (const char *hm = getenv("TAA_PAGER_HEADROOM_MB")) {
                g_pagerHeadroomMB = (uint64_t)atoll(hm);
                trace("PAGER: headroom reserve set to %llu MB - paging engages "
                      "below this and releases above 1.5x it",
                      (unsigned long long)g_pagerHeadroomMB);
            }

            if (const char *md = getenv("TAA_PAGER_MAX_DROP")) {
                int v = atoi(md);
                if (v >= 1 && v <= 4) g_pagerMaxDrop = (uint32_t)v;
                g_pagerEnvLocked = true;
            }
            if (const char *dp = getenv("TAA_PAGER_DROP_ABOVE")) {
                g_pagerEnvLocked = true;
                g_pagerDropAbove = (uint32_t)atoi(dp);
                trace("PAGER: custom texture pager armed - textures larger "
                      "than %u px lose up to %u mip level%s at creation, and "
                      "their uploads are remapped to match",
                      g_pagerDropAbove, g_pagerMaxDrop,
                      g_pagerMaxDrop == 1 ? "" : "s");
            }

            if (const char *ag = getenv("TAA_PAGER_AUTOGEN_TO")) {
                g_pagerEnvLocked = true;
                g_pagerAutogenTo = (uint32_t)atoi(ag);
                if (g_pagerAutogenTo)
                    trace("PAGER: streamed scenery capped at %u px - textures "
                          "created mid-flight are autogen and ortho, and lose as "
                          "many levels as it takes. The aircraft loads before the "
                          "flight starts and is not in that set.",
                          g_pagerAutogenTo);
            }
                trace("XP FSR: TAA_REPLACE_XPFSR - X-Plane's own upscale "
                      "dispatches will be dropped once its shader modules are "
                      "recognised, and FSR2's display-sized result written in "
                      "their place.");
            g_overcommit = (getenv("TAA_OVERCOMMIT") != nullptr);
            if (g_overcommit)
                trace("ALLOC: overcommit armed - device-local failures will retry "
                      "from host-visible memory");

            if (const char *vb = getenv("TAA_VRAM_BUDGET")) {
                g_vramBudgetScale = (float)atof(vb);
                trace("VRAM: budget override armed, scale x%.2f", g_vramBudgetScale);
            }

}

// ---- THE SWAPCHAIN, WHICH NOTHING HAS EVER TRACKED.
//
// The seam test settled it: copying only the left half of FSR2's output into
// g_sceneColor.image produced no seam at all, so nothing written there reaches
// the display. The scene target we have been resolving into is an intermediate
// X-Plane has already finished with by the time the 3D/UI boundary fires.
//
// The swapchain image is the one thing that is definitionally on screen. To
// compare against it - or eventually to present our own upscaled result - the
// layer has to know which images belong to the swapchain, and it never asked.

// ---- X-PLANE'S OWN gbuffer_vel: FIND IT, NAME IT, DO NOT YET READ IT.
//
// The corpus is unambiguous about what it is. In ssr_deferred, and nowhere else
// in 6855 modules:
//
//     %244 = OpTypeImage %uint 2D 0 1 1 1 Unknown     ; set 0, binding 4
//     %13174 = OpImageFetch %v4uint %21460 %12832 Sample|ZeroExtend %int_0
//     %12255 = OpBitwiseAnd %uint %22474 %uint_4
//     %21876 = OpIEqual %bool %12255 %uint_4
//     ... OpPhi selects u_local_reproj when set, u_reproj when clear
//
// So it is a MULTISAMPLED UNSIGNED-INTEGER 2D ARRAY image storing FLAGS, and bit
// 2 means "this pixel is local/ownship geometry rather than world-static". That
// makes bit 2 the only per-pixel motion classification X-Plane provides, and it
// is exactly the signal our epipolar residual needs - that residual is valid
// only for world-static geometry, so a per-pixel "do not trust this here" flag
// is precisely the missing input.
//
// WHAT THIS DOES AND DOES NOT DO. It identifies the image by shape and reports
// it. It does not bind it, and deliberately: consuming it needs the right view
// type for an arrayed multisampled uint image, a sample index, and a decision
// about what to do at pixels where the flag disagrees with our vectors - and
// none of that can be checked without watching a frame. Guessing all three at
// once and shipping it is how the velocity field acquired six weeks of wrong
// explanations.
//
// Naming the image is the part that needed the corpus. The rest needs a sim.
// (Defined early, beside the census forward declaration.)

static bool velFormatIsUint(VkFormat f)
{
    switch (f) {
        case VK_FORMAT_R8_UINT:        case VK_FORMAT_R8G8_UINT:
        case VK_FORMAT_R8G8B8A8_UINT:  case VK_FORMAT_R16_UINT:
        case VK_FORMAT_R16G16_UINT:    case VK_FORMAT_R16G16B16A16_UINT:
        case VK_FORMAT_R32_UINT:       case VK_FORMAT_R32G32_UINT:
        case VK_FORMAT_R32G32B32A32_UINT:
            return true;
        default:
            return false;
    }
}

// Called from the colour-image census, under g_lock.
static void noteGbufferVelCandidate(const ColorTarget &c)
{
    if (g_gbufferVelCandidate != VK_NULL_HANDLE) return;
    if (!velFormatIsUint(c.format)) return;
    // Scene-sized. The G-buffer attachments share the scene's dimensions, and a
    // uint image of some other size is a histogram or a tile buffer.
    if (!g_mv.w || !g_mv.h || c.w != g_mv.w || c.h != g_mv.h) return;
    g_gbufferVelCandidate = c.image;
    trace("GBUFFER_VEL: candidate %p fmt=%s %ux%u layers=%u samples=%u. "
          "ssr_deferred reads a uint flags image at set 0 binding 4 and tests "
          "bit 2 to choose u_local_reproj over u_reproj, so bit 2 is X-Plane's "
          "own per-pixel moving-geometry flag - the only one it has. NOT bound "
          "yet: reading it needs a view type for an arrayed multisampled uint "
          "image and a sample index, neither of which can be verified without "
          "watching a frame.",
          (void*)c.image, formatName(c.format), c.w, c.h,
          c.arrayLayers, (unsigned)c.samples);
}

// ---- ONE COMMAND THAT ANSWERS EVERYTHING.
//
// Set `report=1` in the live file and the next frame writes the complete state
// of the layer to the log, then clears the key back to 0 so it fires once.
//
// The point is not that any single line here is new. It is that they are all in
// ONE place at ONE instant, so a question like "did it pick the right pass, and
// was that target arrayed, and were the vectors bound, and how many shaders got
// patched" is answered by one keystroke rather than by four launches each
// looking for one number. Almost every dead end in this project's history was a
// fact that was already knowable and simply had not been printed next to the
// fact it contradicted.
static void mvFullReport(const char *why, uint64_t frames)
{
    trace("");
    trace("================ MOTION VECTORS - FULL STATE (%s, frame %llu) ============",
          why, (unsigned long long)frames);

    // ---- what the live file is currently forcing
    trace("-- LIVE CONTROLS (%s, %llu reload(s))",
          live::path(), (unsigned long long)live::reloads());
    trace("   enable=%d mode=%d alpha=%.4f gain=%.2f varclip=%.2f viz=%d scale=%.2f",
          taaEnabled() ? 1 : 0, taaMode(), taaAlpha(), taaGain(),
          taaVarClip(), taaViz(), taaVizScale());
    trace("   freeze_history=%d no_motion=%d no_accum=%d force_reset=%d",
          taaFreezeHistory() ? 1 : 0, taaNoMotion() ? 1 : 0,
          taaNoAccum() ? 1 : 0, taaForceReset() ? 1 : 0);

    // ---- the injector: how much of the scene actually carries vectors
    trace("-- SPIR-V INJECTION");
    trace("   varyings at Location %u/%u, velocity at attachment %u, push offset %u",
          spvinj::currClipLocation(), spvinj::prevClipLocation(),
          spvinj::mvAttachmentIndex(), spvinj::pushConstantOffset());
    trace("   device locations=%u  safe=%s  multi-return modules=%llu",
          spvinj::deviceLocationCount(),
          spvinj::locationsAreSafe() ? "yes" : "NO",
          (unsigned long long)spvinj::multiReturnModules());
    mvLogInjectReasons();
    trace("   layouts: %llu extended, %llu skipped",
          (unsigned long long)g_layoutPatched,
          (unsigned long long)g_layoutSkipped);

    // ---- the velocity target
    trace("-- VELOCITY TARGET");
    trace("   ready=%d %ux%u image=%p view=%p viewArray=%p binds last frame=%u",
          g_mv.ready ? 1 : 0, g_mv.w, g_mv.h, (void*)g_mv.image,
          (void*)g_mv.view, (void*)g_mv.viewArray, g_diagBoundPasses);
    trace("   gbuffer_vel candidate (X-Plane's own flags image): %p",
          (void*)g_gbufferVelCandidate);

    // ---- the resolve
    trace("-- TAA RESOLVE");
    trace("   ready=%d %ux%u x%u layer(s) fmt=%d dispatches=%llu scene targets=%llu",
          g_taa.ready ? 1 : 0, g_taa.w, g_taa.h, g_taa.layers, (int)g_taa.format,
          (unsigned long long)g_taa.dispatches,
          (unsigned long long)g_taa.sceneViews.size());
    trace("   HDR candidate passes: %u this frame, %u last frame (the resolve "
          "runs on the LAST one)", g_hdrPassesThisFrame, g_hdrPassesLastFrame);
    trace("   wrote into %p this frame", (void*)g_taaWroteImageThisFrame);
    {
        temporal::BackendInfo bi = g_taaBackend.info();
        trace("   backend '%s': upscaling=%d framegen=%d native=%d arrays=%d msaa=%d",
              bi.name, bi.supportsUpscaling ? 1 : 0,
              bi.supportsFrameGeneration ? 1 : 0,
              bi.supportsNativeResolution ? 1 : 0,
              bi.supportsArrayLayers ? 1 : 0, bi.supportsMultisample ? 1 : 0);
    }

    // ---- every colour image we know about, with its SHAPE.
    //
    // The shape is the column that matters and the one that was missing for
    // weeks: layers and samples are what decide whether the resolve can touch a
    // target at all, and a census that printed only format and size could not
    // have shown why the picture depended on the camera view.
    {
        std::lock_guard<std::mutex> g(g_lock);
        trace("-- COLOUR IMAGE CENSUS (%llu tracked)",
              (unsigned long long)g_colorImages.size());
        int n = 0;
        for (std::map<VkImage, ColorTarget>::iterator it = g_colorImages.begin();
             it != g_colorImages.end() && n < 64; ++it, ++n)
            trace("   %p %-22s %5ux%-5u layers=%u samples=%u usage=0x%x%s",
                  (void*)it->first, formatName(it->second.format),
                  it->second.w, it->second.h, it->second.arrayLayers,
                  (unsigned)it->second.samples, it->second.usage,
                  it->first == g_taaWroteImageThisFrame ? "   <== TAA WROTE HERE" :
                  it->first == g_gbufferVelCandidate    ? "   <== gbuffer_vel?" : "");
    }

    trace("-- CAMERA");
    trace("   viewType=%d jitter=(%.5f %.5f) camDelta=%.5f",
          g_velSnap.viewType, g_velSnap.jitterX, g_velSnap.jitterY,
          g_velSnap.camDelta);

    trace("================ END FULL STATE ==========================================");
    trace("");
}

static void mvMaybeReport(uint64_t frames)
{
    // One-shot: fires once and clears its own key, so leaving it set in the file
    // by accident costs one report rather than one per frame forever.
    if (live::i("report", nullptr, 0)) {
        mvFullReport("requested", frames);
        live::clearOneShot("report");
    }
    int every = live::i("report.every", nullptr, 0);
    if (every > 0 && (frames % (uint64_t)every) == 0)
        mvFullReport("periodic", frames);
}

// ---- WHO WRITES THE PRESENTED IMAGE?
//
// The scene-target census answered a question that had been unanswerable: no
// full-viewport render pass ever draws into a swapchain image. So X-Plane gets
// its final frame there some other way, and the only candidates are a blit or a
// copy - neither of which this layer has ever intercepted.
//
// Whatever the SOURCE of that transfer is, it is by definition the last image
// that matters, and it is the one place a resolve would survive. Every target we
// have written to today has been invisible because none of them was it.
static VkImage g_presentSource = VK_NULL_HANDLE;

// Every DISTINCT transfer, once. Not just the ones into a swapchain image.
//
// Three targeted guesses have now missed - a blit into the swapchain, a copy
// into it, and an MSAA resolve (that call site never runs; renopt_MSAA is 0).
// Guessing one candidate per launch is the expensive way to search a space this
// small, so this prints the whole transfer graph in a single run and the
// swapchain handles beside it.
// ---- DOES OUR OUTPUT BECOME THE SSR REFLECTION SOURCE?
//
// ssr_deferred reads `tex_ssr` at set 0 binding 8 - the PREVIOUS frame's
// rendered colour, sampled as a 2D array with an explicit LOD, and
// `u_half_ssr_history_res` is consumed only to turn a screen-space ray footprint
// into a mip level. So tex_ssr is a HALF-RESOLUTION, MIPPED pyramid of last
// frame's picture.
//
// That pyramid has to be built from something, and if it is built from the HDR
// target AFTER our resolve has written into it, reflections inherit our history
// and close a multi-frame temporal feedback loop through SSR. It is a plausible
// contributor to the "giant reflection glare" symptom, and it would not look
// like a feedback loop - it would look like a quality problem in the reflections.
//
// This cannot be settled by reading the shaders: they say what ssr_deferred
// consumes, not which Vulkan image X-Plane fills it from. But the transfer graph
// says it exactly. If the image we wrote this frame becomes the SOURCE of a
// transfer into a smaller destination, that is the pyramid being built from our
// output. Detect it and say so, rather than guess.
static void noteSsrFeedbackCheck(const char *how, VkImage src, VkImage dst,
                                 const ColorTarget *cs, const ColorTarget *cd)
{
    if (src == VK_NULL_HANDLE || src != g_taaWroteImageThisFrame) return;
    if (!cs || !cd) return;
    // Half resolution or smaller, in both axes: the shape of a pyramid base, not
    // of a present blit.
    if (cd->w > cs->w / 2 || cd->h > cs->h / 2) return;
    static bool said = false;
    if (said) return;
    said = true;
    trace("TAA: *** FEEDBACK RISK - the image our resolve wrote (%p %ux%u) is the "
          "SOURCE of a %s into %p (%ux%u), which is half resolution or smaller. "
          "That is the shape of the tex_ssr pyramid, and ssr_deferred samples "
          "tex_ssr as the previous frame's colour. If so, reflections inherit our "
          "history and close a temporal feedback loop through SSR. Feed SSR the "
          "PRE-TAA colour instead. ***",
          (void*)src, cs->w, cs->h, how, (void*)dst, cd->w, cd->h);
}

static void noteTransfer(const char *how, VkImage src, VkImage dst)
{
    std::lock_guard<std::mutex> g(g_lock);
    {
        std::map<VkImage, ColorTarget>::iterator fs = g_colorImages.find(src);
        std::map<VkImage, ColorTarget>::iterator fd = g_colorImages.find(dst);
        noteSsrFeedbackCheck(how, src, dst,
                             fs != g_colorImages.end() ? &fs->second : nullptr,
                             fd != g_colorImages.end() ? &fd->second : nullptr);
    }
    static std::set<std::pair<VkImage,VkImage> > seen;
    std::pair<VkImage,VkImage> k(src,dst);
    if (seen.size() < 24 && !seen.count(k)) {
        seen.insert(k);
        std::map<VkImage, ColorTarget>::iterator cs = g_colorImages.find(src);
        std::map<VkImage, ColorTarget>::iterator cd = g_colorImages.find(dst);
        trace("XFER %-4s %p (fmt=%d %ux%u)%s -> %p (fmt=%d %ux%u)%s", how,
              (void*)src, cs != g_colorImages.end() ? (int)cs->second.format : -1,
              cs != g_colorImages.end() ? cs->second.w : 0,
              cs != g_colorImages.end() ? cs->second.h : 0,
              isSwapImage(src) ? " [SWAPCHAIN]" : "",
              (void*)dst, cd != g_colorImages.end() ? (int)cd->second.format : -1,
              cd != g_colorImages.end() ? cd->second.w : 0,
              cd != g_colorImages.end() ? cd->second.h : 0,
              isSwapImage(dst) ? " [SWAPCHAIN]" : "");
    }
}

static void notePresentSource(const char *how, VkImage src, VkImage dst)
{
    noteTransfer(how, src, dst);
    std::lock_guard<std::mutex> g(g_lock);
    if (!isSwapImage(dst)) return;
    if (g_presentSource == src) return;
    g_presentSource = src;
    std::map<VkImage, ColorTarget>::iterator ct = g_colorImages.find(src);
    trace("PRESENT SOURCE: %s into a swapchain image from %p (fmt=%d %ux%u). "
          "THIS is the image that reaches the screen - the resolve belongs here, "
          "and it is not the one we have been writing to (%p).",
          how, (void*)src,
          ct != g_colorImages.end() ? (int)ct->second.format : -1,
          ct != g_colorImages.end() ? ct->second.w : 0,
          ct != g_colorImages.end() ? ct->second.h : 0,
          (void*)g_sceneColor.image);
}

static VKAPI_ATTR void VKAPI_CALL Layer_CmdBlitImage(
    VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst,
    VkImageLayout dl, uint32_t n, const VkImageBlit *regions, VkFilter filter)
{
    notePresentSource("BLIT", src, dst);
    PFN_vkCmdBlitImage next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<VkCommandBuffer, VkDevice>::iterator ci = g_cbToDevice.find(cb);
        if (ci != g_cbToDevice.end()) {
            std::map<void*, DeviceData>::iterator di = g_devices.find(dispatchKey(ci->second));
            if (di != g_devices.end()) next = di->second.cmdBlitImage;
        }
    }
    if (next) next(cb, src, sl, dst, dl, n, regions, filter);
}

static VKAPI_ATTR void VKAPI_CALL Layer_CmdCopyImage(
    VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst,
    VkImageLayout dl, uint32_t n, const VkImageCopy *regions)
{
    notePresentSource("COPY", src, dst);
    PFN_vkCmdCopyImage next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<VkCommandBuffer, VkDevice>::iterator ci = g_cbToDevice.find(cb);
        if (ci != g_cbToDevice.end()) {
            std::map<void*, DeviceData>::iterator di = g_devices.find(dispatchKey(ci->second));
            if (di != g_devices.end()) next = di->second.cmdCopyImage;
        }
    }
    if (next) next(cb, src, sl, dst, dl, n, regions);
}

// ---- X-PLANE'S MSAA RESOLVE, WHICH HAS BEEN ERASING OUR WORK.
//
// Found by disassembling the renderer rather than guessing at pass shapes.
// There is exactly one vkCmdResolveImage call site, at 0x140638eb8:
//
//     mov r8d, 6                ; srcImageLayout = TRANSFER_SRC_OPTIMAL
//     mov dword [rsp+0x20], 7   ; dstImageLayout = TRANSFER_DST_OPTIMAL
//     mov rdx, [rdx+0x78]       ; srcImage   (the 2x MSAA target)
//     mov r9,  [r9+0x78]        ; dstImage   (the single-sample target)
//     call qword ptr [rip+...]  ; vkCmdResolveImage
//
// X-Plane renders the scene MULTISAMPLED and resolves it down into the
// single-sample HDR image - the very image every resolve we have recorded has
// been written into. Our copy goes in at the pass boundary; this resolve then
// lands on top and overwrites it. That is why the accumulation measured
// correct, the copy measured correct, and the screen never changed.
//
// The existing code knew both targets existed - "one single-sample and one 2x
// multisampled" - and deliberately avoided the multisampled one because
// copying into it "produced the sheared bands". It never followed that through
// to the consequence: the MSAA image is the source of truth, and the
// single-sample one is downstream of it.
//
// Nothing is changed here yet. The resolve is intercepted, reported once, and
// passed straight through, so the ordering can be confirmed before anything
// depends on it.
static VkImage g_msaaResolveSrc = VK_NULL_HANDLE;
static VkImage g_msaaResolveDst = VK_NULL_HANDLE;

static VKAPI_ATTR void VKAPI_CALL Layer_CmdResolveImage(
    VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst,
    VkImageLayout dl, uint32_t n, const VkImageResolve *regions)
{
    PFN_vkCmdResolveImage next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<VkCommandBuffer, VkDevice>::iterator ci = g_cbToDevice.find(cb);
        if (ci != g_cbToDevice.end()) {
            std::map<void*, DeviceData>::iterator di = g_devices.find(dispatchKey(ci->second));
            if (di != g_devices.end()) next = di->second.cmdResolveImage;
        }
        if (g_msaaResolveDst != dst || g_msaaResolveSrc != src) {
            g_msaaResolveSrc = src; g_msaaResolveDst = dst;
            std::map<VkImage, ColorTarget>::iterator cs = g_colorImages.find(src);
            std::map<VkImage, ColorTarget>::iterator cd = g_colorImages.find(dst);
            trace("MSAA RESOLVE: %p (fmt=%d samples=%d) -> %p (fmt=%d samples=%d). "
                  "Our scene target is %p, and we are %s. If they match, X-Plane "
                  "overwrites our resolve here every frame.",
                  (void*)src,
                  cs != g_colorImages.end() ? (int)cs->second.format : -1,
                  cs != g_colorImages.end() ? (int)cs->second.samples : -1,
                  (void*)dst,
                  cd != g_colorImages.end() ? (int)cd->second.format : -1,
                  cd != g_colorImages.end() ? (int)cd->second.samples : -1,
                  (void*)g_sceneColor.image,
                  dst == g_sceneColor.image ? "WRITING INTO ITS DESTINATION"
                                            : "writing somewhere else");
        }
    }
    if (next) next(cb, src, sl, dst, dl, n, regions);
}

// ---- WHAT DOES THE SWAPCHAIN PASS ACTUALLY SAMPLE?
//
// Every destination so far has been inferred from render order - the last
// full-viewport target, the last HDR one, the last 8-bit one, the sRGB one -
// and every one of them was invisible. Inference has now been wrong five times
// in a row, so read the binding instead of guessing it.
//
// vkUpdateDescriptorSets records which image views each descriptor set holds.
// vkCmdBindDescriptorSets, while the pass writing a swapchain image is open,
// says which of those sets that pass is about to sample. The intersection is
// the image the final composite reads, which is the only correct destination.

static VKAPI_ATTR void VKAPI_CALL Layer_UpdateDescriptorSets(
    VkDevice device, uint32_t nw, const VkWriteDescriptorSet *w,
    uint32_t nc, const VkCopyDescriptorSet *c)
{
    vram::noteDescriptorUpdates(nw + nc);
    PFN_vkUpdateDescriptorSets next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, DeviceData>::iterator it = g_devices.find(dispatchKey(device));
        if (it != g_devices.end()) next = it->second.updateDescriptorSets;
        for (uint32_t i = 0; i < nw && w; ++i) {
            if (!w[i].pImageInfo) continue;
            // STORAGE images go in their own map. X-Plane's FSR writes
            // i_output_texture, a storage image, and this filter is why it was
            // invisible: four rebuilds were spent looking for it in descriptor
            // sets that had been recorded with it stripped out.
            if (w[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
                std::vector<VkImageView> &sv = g_setStorageViews[w[i].dstSet];
                for (uint32_t k = 0; k < w[i].descriptorCount; ++k)
                    if (w[i].pImageInfo[k].imageView != VK_NULL_HANDLE)
                        sv.push_back(w[i].pImageInfo[k].imageView);
                if (sv.size() > 64) sv.erase(sv.begin(), sv.begin() + (sv.size() - 64));
                continue;
            }
            if (w[i].descriptorType != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
                w[i].descriptorType != VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) continue;
            std::vector<VkImageView> &v = g_setViews[w[i].dstSet];
            for (uint32_t k = 0; k < w[i].descriptorCount; ++k)
                if (w[i].pImageInfo[k].imageView != VK_NULL_HANDLE)
                    v.push_back(w[i].pImageInfo[k].imageView);
            if (v.size() > 64) v.erase(v.begin(), v.begin() + (v.size() - 64));
        }
    }
    if (next) next(device, nw, w, nc, c);
}

// ---- IS X-PLANE REBINDING STATE IT HAS ALREADY BOUND?
//
// The frame budget says the render thread is busy ~97% of wall clock and blocks
// on nothing, so the ceiling is CPU work inside the driver. The cheapest thing a
// layer can do about that is stop forwarding calls that change nothing: engines
// routinely rebind the same pipeline or the same descriptor sets between draws,
// and the driver validates every one.
//
// Counted before anything is filtered, because the ratio decides whether
// filtering is worth the risk at all. Recording is per-thread, so the cache is
// thread_local and needs no lock; it is keyed by command buffer and reset when
// the thread starts recording a different one, which is conservative - a stale
// entry can only make us MISS a redundancy, never invent one.
struct BindCache {
    VkCommandBuffer  cb = VK_NULL_HANDLE;
    VkPipeline       pipe[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };  // gfx, compute
    VkPipelineLayout dsLayout = VK_NULL_HANDLE;
    uint32_t         dsFirst = 0, dsCount = 0, dsDyn = 0;
    VkDescriptorSet  dsSets[8] = {};
};
static thread_local BindCache g_bindCache;
static std::atomic<uint64_t> g_pipeBinds(0), g_pipeBindsRedundant(0);
static std::atomic<uint64_t> g_dsBinds(0), g_dsBindsRedundant(0);

static VKAPI_ATTR void VKAPI_CALL Layer_CmdBindDescriptorSets(
    VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout layout,
    uint32_t first, uint32_t n, const VkDescriptorSet *sets,
    uint32_t nd, const uint32_t *dyn)
{
    // Sampled per-resource usage for the aging walk (SS86): 1 in 64 binds
    // resolves this set's sampled images and stamps their last-use frame.
    // Aging needs "used this minute", not "used this draw", so sampling costs
    // nothing it needs.
    static std::atomic<uint32_t> useSample(0);
    bool sampleUse = (useSample.fetch_add(1) & 63) == 0;

    PFN_vkCmdBindDescriptorSets next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        if (sampleUse && sets) {
            std::map<VkCommandBuffer, bool>::iterator sc = g_cbInScenePass.find(cb);
            bool inScene = sc != g_cbInScenePass.end() && sc->second;
            for (uint32_t s = 0; s < n; ++s) {
                std::map<VkDescriptorSet, std::vector<VkImageView> >::iterator
                    sv = g_setViews.find(sets[s]);
                if (sv == g_setViews.end()) continue;
                for (size_t v = 0; v < sv->second.size(); ++v) {
                    std::map<VkImageView, VkImage>::iterator im =
                        g_viewToImage.find(sv->second[v]);
                    if (im != g_viewToImage.end())
                        vram::noteImageUse(im->second, inScene);
                }
            }
        }
        std::map<VkCommandBuffer, VkDevice>::iterator ci = g_cbToDevice.find(cb);
        if (ci != g_cbToDevice.end()) {
            std::map<void*, DeviceData>::iterator di = g_devices.find(dispatchKey(ci->second));
            if (di != g_devices.end()) next = di->second.cmdBindDescriptorSets;
        }
        // ---- REMEMBER WHAT AN FSR DISPATCH IS ABOUT TO READ AND WRITE.
        //
        // vkCmdDispatch names nothing - no pipeline, no resources - so by the
        // time we decide to drop X-Plane's upscale there is no way left to ask
        // what it would have written. The descriptor sets bound beforehand are
        // the only place that answer exists, and only while the FSR pipeline is
        // the one bound on this command buffer.
        //
        // Recorded unconditionally of fsr.replace: the switch can be flipped
        // mid-flight, and a dispatch that arrives in the same frame as the flip
        // would otherwise find nothing recorded and be dropped with no target.
        {
            // NOT gated on the FSR pipeline already being bound. X-Plane
            // binds descriptor sets BEFORE the pipeline, so gating on
            // g_cbFsrBound recorded nothing and the dispatch found a null
            // destination. Recording is cheap and the dispatch identifies its
            // own output precisely, so a few extra candidates cost nothing.
            if (sets) {
                std::vector<VkImage> &imgs = g_cbFsrImages[(void*)cb];
                if (imgs.size() > 64) imgs.clear();   // bounded, per buffer
                for (uint32_t s2 = 0; s2 < n; ++s2) {
                    // Storage first: the upscale's DESTINATION is here, and
                    // it is the only one of these images we can legally write.
                    std::map<VkDescriptorSet, std::vector<VkImageView> >::iterator
                        stv = g_setStorageViews.find(sets[s2]);
                    if (stv != g_setStorageViews.end())
                        for (size_t v = 0; v < stv->second.size(); ++v) {
                            std::map<VkImageView, VkImage>::iterator im2 =
                                g_viewToImage.find(stv->second[v]);
                            if (im2 == g_viewToImage.end()) continue;
                            bool have2 = false;
                            for (size_t q = 0; q < imgs.size(); ++q)
                                if (imgs[q] == im2->second) { have2 = true; break; }
                            if (!have2) imgs.push_back(im2->second);
                        }

                    std::map<VkDescriptorSet, std::vector<VkImageView> >::iterator
                        sv = g_setViews.find(sets[s2]);
                    if (sv == g_setViews.end()) continue;
                    for (size_t v = 0; v < sv->second.size(); ++v) {
                        std::map<VkImageView, VkImage>::iterator im =
                            g_viewToImage.find(sv->second[v]);
                        if (im == g_viewToImage.end()) continue;
                        bool have = false;
                        for (size_t q = 0; q < imgs.size(); ++q)
                            if (imgs[q] == im->second) { have = true; break; }
                        if (!have) imgs.push_back(im->second);
                    }
                }
            }
        }
        std::map<VkCommandBuffer, bool>::iterator sp = g_cbInSwapPass.find(cb);
        if (sp != g_cbInSwapPass.end() && sp->second && sets) {
            static std::set<VkImage> reported;
            for (uint32_t i = 0; i < n; ++i) {
                std::map<VkDescriptorSet, std::vector<VkImageView> >::iterator sv =
                    g_setViews.find(sets[i]);
                if (sv == g_setViews.end()) continue;
                for (size_t k = 0; k < sv->second.size(); ++k) {
                    std::map<VkImageView, VkImage>::iterator vi =
                        g_viewToImage.find(sv->second[k]);
                    if (vi == g_viewToImage.end()) continue;
                    std::map<VkImage, ColorTarget>::iterator ct =
                        g_colorImages.find(vi->second);
                    if (ct == g_colorImages.end()) continue;          // not a render target
                    // ---- NOT A FIXED 1920, WHICH ASSUMED 1440p OR BETTER.
                    //
                    // This filters which images the diagnostic reports, and a
                    // hard 1920 floor silently drops the whole thing at 1080p
                    // with any render scale below 1.0 - the same run logs
                    // 960x540 passes, so that is not hypothetical. A diagnostic
                    // that goes quiet at exactly the resolutions you are trying
                    // to debug is worse than no diagnostic.
                    //
                    // 1280 matches the floor the FSR blit path already uses and
                    // still excludes thumbnails and lookup tables, which is all
                    // this was ever meant to do.
                    if (ct->second.w < 1280) continue;                // not full-window
                    if (reported.size() >= 12 || reported.count(vi->second)) continue;
                    reported.insert(vi->second);
                    trace("SWAP PASS SAMPLES: %p fmt=%d %ux%u  (our scene target is "
                          "%p fmt=%d) -> %s",
                          (void*)vi->second, (int)ct->second.format,
                          ct->second.w, ct->second.h,
                          (void*)g_sceneColor.image, (int)g_sceneColor.format,
                          vi->second == g_sceneColor.image
                              ? "MATCH - we are writing the right image"
                              : "MISMATCH - THIS is what reaches the screen");
                }
            }
        }
    }
    // Redundancy census: identical layout, range, set handles and no dynamic
    // offsets means this bind changes nothing the driver has not already been
    // told. Counted only; nothing is filtered yet - see BindCache.
    if (sets && n <= 8) {
        if (g_bindCache.cb != cb) {
            g_bindCache = BindCache();
            g_bindCache.cb = cb;
        }
        bool same = (g_bindCache.dsLayout == layout &&
                     g_bindCache.dsFirst == first &&
                     g_bindCache.dsCount == n &&
                     g_bindCache.dsDyn == 0 && nd == 0);
        for (uint32_t s = 0; same && s < n; ++s)
            if (g_bindCache.dsSets[s] != sets[s]) same = false;
        g_dsBinds.fetch_add(1, std::memory_order_relaxed);
        if (same && n) g_dsBindsRedundant.fetch_add(1, std::memory_order_relaxed);
        g_bindCache.dsLayout = layout;
        g_bindCache.dsFirst  = first;
        g_bindCache.dsCount  = n;
        g_bindCache.dsDyn    = nd;
        for (uint32_t s = 0; s < n && s < 8; ++s) g_bindCache.dsSets[s] = sets[s];
    }
    if (next) next(cb, bp, layout, first, n, sets, nd, dyn);
}

static VKAPI_ATTR VkResult VKAPI_CALL Layer_GetSwapchainImagesKHR(
    VkDevice device, VkSwapchainKHR sc, uint32_t *count, VkImage *images)
{
    PFN_vkGetSwapchainImagesKHR next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, DeviceData>::iterator it = g_devices.find(dispatchKey(device));
        if (it != g_devices.end()) next = it->second.getSwapchainImagesKHR;
    }
    if (!next) return VK_ERROR_INITIALIZATION_FAILED;


    VkResult r = next ? next(device, sc, count, images)
                      : VK_ERROR_INITIALIZATION_FAILED;
    if (r == VK_SUCCESS && images && count && *count) {
        std::lock_guard<std::mutex> g(g_lock);
        std::vector<VkImage> &v = g_swapImages[sc];
        v.assign(images, images + *count);
        static bool said = false;
        if (!said) {
            said = true;
            trace("SWAPCHAIN: %u images tracked - the layer can now see what is "
                  "actually presented, which it never could before.", *count);
            for (uint32_t k = 0; k < *count; ++k)
                trace("SWAPCHAIN:   image[%u] = %p", k, (void*)images[k]);
        }
    }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL Layer_QueuePresentKHR(
    VkQueue queue, const VkPresentInfoKHR *info)
{
    // Global rather than a static local, because the FSR2 idle timeout needs a
    // frame count too and it runs in vkEndCommandBuffer, nowhere near here.
    // One counter with one definition of "a frame" - a second one incremented
    // somewhere else would drift from this and the two would disagree in logs
    // for reasons nobody could reconstruct later.
    uint64_t frames = ++g_frameCount;

    // The probe's answer becomes readable a few frames after its copies were
    // recorded. Checked here because this is the one place that runs exactly
    // once per frame and is already past the submit that carried them.
    fsrProbeResolve();

    // ---- BUILD THE FSR3 CONTEXT HERE, NOT IN THE DISPATCH.
    //
    // Present is the one point per frame where no command buffer is being
    // recorded, so pipeline and memory creation is safe. Attempted from inside
    // vkCmdDispatch it killed the sim before any FSR3 trace appeared.
    //
    // Everything it needs settles at different times - the sub-native render
    // size, the output image the probe identifies - so this simply retries each
    // frame until they are all present, and does nothing once built.
    if (fsrReplaceEnabled() && fsr3Wanted() && !fsr3::state().ready &&
        !fsr3::state().failed) {
        VkDevice dev = VK_NULL_HANDLE; VkPhysicalDevice ph = VK_NULL_HANDLE;
        PFN_vkGetDeviceProcAddr gd = nullptr;
        uint32_t rw = 0, rh = 0, ow = 0, oh = 0;
        {
            std::lock_guard<std::mutex> g(g_lock);
            std::map<void*, DeviceData>::iterator di = g_devices.begin();
            if (di != g_devices.end()) {
                dev = di->second.device; ph = di->second.phys; gd = di->second.gdpa;
            }
            rw = g_sceneColor.w; rh = g_sceneColor.h;
            std::map<VkImage, ColorTarget>::iterator oc =
                g_colorImages.find(fsrprobe::state().output);
            if (oc != g_colorImages.end()) { ow = oc->second.w; oh = oc->second.h; }
        }
        if (dev && ph && gd && rw && rh && ow && oh && rw < ow)
            fsr3::ensure(dev, ph, gd, g_getPhysMemProps, rw, rh, ow, oh);
    }

    // Last frame's timings, read without blocking - waiting on them would
    // change what is being measured.
    {
        std::map<void*, DeviceData>::iterator dit;
        { std::lock_guard<std::mutex> g(g_lock); dit = g_devices.begin(); }
        if (dit != g_devices.end()) { g_tsPending = true; gpuTimeReport(dit->second, frames); }
    }

    // Arm everything on the first present, unconditionally. Present runs no
    // matter which subsystems are enabled, which is the whole point - see the
    // note on armLayerOnce.
    armLayerOnce();

    // ---- FRAME RATE, MEASURED HERE AND NOWHERE ELSE.
    //
    // The sim's own readout counts frames the SIM rendered. With generation on
    // that stops being the number on the monitor, and no plugin callback can
    // ever see the difference - a flight loop runs once per sim frame by
    // definition. This is the one place both are visible.
    //
    // Averaged over half a second: per-frame reciprocals jitter far too much to
    // read, and a full second is slow to respond when something starts costing.
    if (g_share && g_share->magic == TAA_MAGIC) {
        LARGE_INTEGER now, freq;
        QueryPerformanceCounter(&now);
        QueryPerformanceFrequency(&freq);

        static LARGE_INTEGER windowStart = { 0 };
        static uint64_t      framesAtStart = 0;
        static uint64_t      displayedAtStart = 0;

        g_share->framesPresented = frames;
        // Zero while frame generation is off, and the sim's own count is then
        // the honest answer for both rows rather than a zero that reads as a
        // stall.
        g_share->framesDisplayed = frames;

        if (windowStart.QuadPart == 0) {
            windowStart = now;
            framesAtStart = frames;
            displayedAtStart = g_share->framesDisplayed;
        } else {
            double elapsed = (double)(now.QuadPart - windowStart.QuadPart) /
                             (double)freq.QuadPart;
            if (elapsed >= 0.5) {
                g_share->fpsPresented =
                    (float)((double)(frames - framesAtStart) / elapsed);
                g_share->fpsDisplayed =
                    (float)((double)(g_share->framesDisplayed - displayedAtStart) / elapsed);

                // ---- SAY IT IN THE LOG TOO, NOT ONLY IN THE PANEL.
                //
                // The panel row is for the person flying; this is for reading
                // back afterwards. Without it "is frame generation actually
                // producing frames" can only be answered by someone looking at
                // the screen and reporting, which is a slow way to learn that a
                // ratio is 1.00 and nothing is being generated at all.
                static uint64_t lastSaid = 0;
                if (frames - lastSaid >= 300) {
                    lastSaid = frames;
                    float ratio = g_share->fpsPresented > 0.0f
                                ? g_share->fpsDisplayed / g_share->fpsPresented : 0.0f;
                    trace("FPS: %.1f rendered by the sim -> %.1f presented "
                          "(%.2fx)%s", g_share->fpsPresented, g_share->fpsDisplayed,
                          ratio,
                          ratio < 1.5f ? "  <- frame generation is NOT doubling"
                                       : "");
                }
                windowStart = now;
                framesAtStart = frames;
                displayedAtStart = g_share->framesDisplayed;
            }
        }
    }

    // Tuning file, once a second, HERE rather than inside the FSR2 dispatch.
    // Present runs whatever the upscaler is set to - including Off, which is
    // exactly where someone investigating a bad image ends up, and exactly
    // where the old placement made every diagnostic control silently inert.

    // ---- PUBLISH THE SNAPSHOT, UNCONDITIONALLY.
    //
    // g_velSnap is what the recording hooks read: vkCmdBindPipeline needs the
    // selected upscaler to decide whether to jitter, and the resolve and FSR2
    // paths read the same struct.
    //
    // It used to be published only inside the velocity pass's block, so with
    // TAA_VELOCITY off it kept its zero-initialised value forever. upscaler 0
    // is "Off", so the jitter's consumer test was false on every one of 400,000
    // pipeline binds while the log cheerfully printed "upscaler=FSR2" from a
    // different snapshot. Measured: inScene=326723 isGeometry=385927 and
    // consumer=0.
    //
    // Same shape of bug as the arming block - per-frame state written inside a
    // subsystem that turned out to be optional. Published here instead, where
    // it cannot be gated on anything.
    // ---- FLIGHT LOOPS PER PRESENT.
    //
    // This is the whole of item 2. The plugin's expectedPx is the angle between
    // world and prevWorld, which are consecutive FLIGHT LOOP samples. The
    // layer's reprojection is built across consecutive PRESENTS. If the two
    // rates are equal the pair is the same and both describe one rendered
    // frame; if the flight loop runs more often, the plugin describes a
    // fraction of what the renderer drew and reports 13.15 px on a frame whose
    // reprojection legitimately encodes 842.
    //
    // Neither number is wrong in that case - they answer different questions -
    // and the SHADER must use the present-paired one, which it now does. This
    // says which case is actually occurring instead of leaving it inferred.
    if (g_share && g_share->magic == TAA_MAGIC) {
        static uint64_t lastShareFrame = 0;
        static uint64_t nOne = 0, nMore = 0, nZero = 0, maxGap = 0;
        const uint64_t f = g_share->frame;
        if (lastShareFrame) {
            const uint64_t gap = f - lastShareFrame;
            if (gap == 1) ++nOne; else if (gap == 0) ++nZero; else ++nMore;
            if (gap > maxGap) maxGap = gap;
            if (((nOne + nMore + nZero) % 600) == 0)
                trace("MV RATE: flight loops per present - exactly one %llu, "
                      "more than one %llu, none %llu, worst gap %llu",
                      (unsigned long long)nOne, (unsigned long long)nMore,
                      (unsigned long long)nZero, (unsigned long long)maxGap);
        }
        lastShareFrame = f;
    }

    // EVERY SWITCH, ONCE, AS THE LAYER SEES IT.
    //
    // An experiment run through a switch that never arrived is worse than no
    // experiment: TAA_MV_IDENTITY was set for a whole run and the output came
    // back byte-identical to the run without it, which reads as "identity makes
    // no difference" rather than as "the variable never reached the process".
    // The self-test is deterministic - paused sim, scripted camera - so
    // identical output is exactly what an ineffective switch produces.
    {
        static bool said = false;
        if (!said) {
            said = true;
            trace("MV SWITCHES: identity=%d pass=%ld noBody=%d pluginReproj=%d",
                  getenv("TAA_MV_IDENTITY") ? 1 : 0,
                  getenv("TAA_MV_PASS") ? atol(getenv("TAA_MV_PASS")) : 1L,
                  getenv("TAA_MV_NO_BODY") ? 1 : 0,
                  g_usePluginReproj ? 1 : 0);
        }
    }

    if (g_share && g_share->magic == TAA_MAGIC) {
        Snapshot fresh;
        if (snapshot(&fresh)) {
            g_velSnap = fresh;

            // ---- REPROJECTION BUILT ACROSS CONSECUTIVE PRESENTS.
            //
            // The plugin pairs world with prevWorld inside a flight loop. The
            // velocity field is produced per RENDER frame, and at 4K with a
            // 31.9 MB readback those two are not one to one - so between two
            // rendered frames the camera can advance several self-test steps
            // while the published matrix still describes one.
            //
            // That is what the surviving bad samples look like: fields uniform
            // across the frame at 5.5x, 12.2x and 28.1x the prediction. A
            // uniform field IS a rigid rotation - it is the field for a
            // rotation of several steps, measured against a matrix describing
            // one. Both halves were internally correct and simply described
            // different pairs of frames.
            //
            // Presents are the frames the renderer actually drew, and this hook
            // is the one place that knows where they end. Pairing here makes
            // the two frames the matrix describes the two the renderer drew, by
            // construction rather than by hoping the rates match.
            //
            // The origin shift is done in the plugin's order - world * Tc
            // first, projection after - because doing it the other way cancels
            // 52 km of world translation only after it has been scaled by the
            // projection, which measured a 10-18% residual when it was tried.
            // DOES THE PROJECTION CHANGE BETWEEN FRAMES?
            //
            // The failing matrices carry terms of +-0.17 where a 0.25 degree
            // rotation should give 0.004, arranged so they cancel at ndcZ=1 and
            // not elsewhere - 1.12097 against 0.12097, 0.87902 against -0.12098.
            // That is a depth REMAPPING of about 12%%, which is what appears in
            // prevProj * R * proj^-1 when the two projections differ. It is not
            // camera motion, and the field carries it because it is genuinely
            // part of the reprojection.
            //
            // If this fires, the vectors are right and the yardstick - one
            // predicted displacement along the centre ray - is what cannot
            // describe the frame.
            {
                static uint64_t nDiff = 0, nSame = 0;
                bool same = memcmp(fresh.proj, fresh.prevProj, 64) == 0;
                if (same) ++nSame; else ++nDiff;
                if (((nDiff + nSame) % 600) == 0)
                    trace("MV PROJ: prevProj differs from proj on %llu of %llu "
                          "frames - near=%.4f, [10]=%.5f vs %.5f, [14]=%.5f vs %.5f",
                          (unsigned long long)nDiff,
                          (unsigned long long)(nDiff + nSame), fresh.nearClip,
                          fresh.proj[10], fresh.prevProj[10],
                          fresh.proj[14], fresh.prevProj[14]);
            }

            // ---- THE SAVED FRAME MUST BE THE IMMEDIATELY PRECEDING ONE.
            //
            // snapshot() can fail: a torn read that does not settle in four
            // attempts, or a share block not yet valid. The save below sits
            // inside that success branch, so a failure left prevWorldSaved
            // holding an OLDER frame and the next success built a reprojection
            // spanning several frames instead of one.
            //
            // That is what the cross-check exposed. The plugin reported 13.13 px
            // on every line - correct, the camera is driven at a constant
            // 0.25 deg/frame - while this matrix read up to 742 px, and the
            // FIELD matched the matrix because the shader faithfully rendered
            // whatever it was handed. A motion vector spanning five frames is
            // wrong for a consumer that reprojects one.
            //
            // The share frame number travels with the saved matrix now, and the
            // reprojection is only built when the gap is exactly one. Anything
            // else falls back to the plugin's own pairing, which is correct by
            // construction because the plugin rolls prev into curr every
            // flight loop unconditionally.
            // ---- ONE-FRAME LAG TEST.
            //
            // The residual image says the error is depth-dependent: distant
            // geometry and the aeroplane are correct to under a pixel, the near
            // runway is 20 px or worse. That is the signature of a wrong
            // TRANSLATION - rotation dominates far geometry, translation
            // dominates near geometry as 1/depth.
            //
            // The translation is not missing: t matches dC component by
            // component. So it may describe the wrong PAIR of frames. X-Plane
            // runs camera callbacks after flight loops, so world_matrix read in
            // a flight loop can be one frame behind what is actually rendered -
            // invisible with a still camera, worth a whole frame of parallax
            // during an orbit, which is exactly the view that fails.
            //
            // TAA_MV_LAG pairs the two OLDER matrices instead. If the residual
            // collapses, the pairing is the bug and this is the fix.
            static const int lagFrames = getenv("TAA_MV_LAG")
                                       ? atoi(getenv("TAA_MV_LAG")) : 0;
            static float    prevPrevWorldSaved[16], prevPrevProjSaved[16];
            static bool     havePrevPrev = false;
            static float    prevWorldSaved[16], prevProjSaved[16];
            static uint64_t prevSavedFrame = 0;
            static bool  havePrevFrame = false;
            const bool   adjacent = havePrevFrame
                                 && fresh.frame == prevSavedFrame + 1;

            // Same closed form the plugin uses, and for the same reason: this
            // built Tc from fresh.camX, a float about 33,870 m from the origin,
            // then cancelled two huge products to leave a small one. Seven
            // significant digits at that distance is a 3.4 mm grid, and the
            // residue lands in the matrix the shader is pushed.
            //
            // Fixing only the plugin left this copy intact, which is why `far`
            // kept disagreeing with the plugin's estimate after that fix - the
            // two were computing the same quantity by different arithmetic, and
            // this one is the arithmetic that loses.
            //
            //     world * Tc     = [R | 0]                      exactly
            //     prevWorld * Tc = [R_prev | R_prev * (C - C_prev)]
            float worldRel[16], currVPrel[16], invCurr[16];
            memcpy(worldRel, fresh.world, sizeof(worldRel));
            worldRel[12] = worldRel[13] = worldRel[14] = 0.0f;
            taaMul(currVPrel, fresh.proj, worldRel);

            // ---- THE PROJECTION IS INVERTED IN CLOSED FORM, NOT NUMERICALLY.
            //
            // reproj = P_prev * W_rel * P_curr^-1. Taking that last inverse with
            // a general 4x4 cofactor expansion in float32 is what broke the
            // vectors: this projection has an INFINITE far plane and a 1.6 cm
            // near plane, so the matrix spans an enormous dynamic range and is
            // severely ill conditioned. The rotation part survives - which is
            // why the field always had the right SHAPE - while the w row
            // degenerates.
            //
            // Measured consequence: prevClip.w arrived at the fragment shader as
            // ZERO, and the shader divides by it. mv = curr.xy/curr.w -
            // prev.xy/prev.w then explodes by however close to zero w landed,
            // which is exactly the 3x to 21x seen, varying per frame, while
            // staying uniform and coherent because the rotation itself was fine.
            //
            // A perspective matrix inverts exactly, in four terms. For the form
            // X-Plane uses - proj[0], proj[5] the x and y scales, proj[10] = -1,
            // proj[11] = -1, proj[14] = -near:
            //
            //     x_view = x_clip / proj[0]
            //     y_view = y_clip / proj[5]
            //     z_view = -w_clip
            //     w_view = (z_clip - proj[10] * (-w_clip)) / proj[14]
            //
            // Every term is a reciprocal of a well-scaled number. Nothing
            // cancels, nothing is conditioned on the far plane.
            float invProj[16];
            {
                const float sx = fresh.proj[0]  != 0.0f ? fresh.proj[0]  : 1.0f;
                const float sy = fresh.proj[5]  != 0.0f ? fresh.proj[5]  : 1.0f;
                const float m10 = fresh.proj[10], m11 = fresh.proj[11], m14 = fresh.proj[14];
                memset(invProj, 0, sizeof(invProj));
                invProj[0]  = 1.0f / sx;                       // x_view from x_clip
                invProj[5]  = 1.0f / sy;                       // y_view from y_clip
                invProj[11] = (m14 != 0.0f) ? 1.0f / m14 : 0.0f;   // w_view from z_clip
                invProj[14] = (m11 != 0.0f) ? 1.0f / m11 : 0.0f;   // z_view from w_clip
                invProj[15] = (m14 != 0.0f) ? -m10 / (m11 * m14) : 0.0f;
            }

            // W_rel = W_prev * W_curr^-1 for two RIGID matrices, which is also
            // closed form: [R_prev * R_curr^T | R_prev * (C_curr - C_prev)].
            // The rotation transpose is exact and the translation is the
            // millimetre-scale delta already differenced in double below.
            if (adjacent && taaInverse(invCurr, currVPrel)) {
                // The camera positions are recovered from each rigid matrix in
                // DOUBLE and differenced there, so the millimetre that survives
                // is exact rather than the remains of a cancellation.
                float prevWorldRel[16], prevVPrel[16], r[16];
                // Under the lag test BOTH ends move back a frame, so the pair
                // stays adjacent. Lagging only the rotation would compare a
                // rotation from one pair against a translation from another and
                // measure nothing.
                const float *curW = (lagFrames && havePrevPrev) ? prevWorldSaved
                                                                : fresh.world;
                const float *preW = (lagFrames && havePrevPrev) ? prevPrevWorldSaved
                                                                : prevWorldSaved;
                const double ct0 = curW[12], ct1 = curW[13], ct2 = curW[14];
                const double pt0 = preW[12], pt1 = preW[13], pt2 = preW[14];
                const double ccx = -((double)curW[0] * ct0 + (double)curW[1] * ct1 + (double)curW[2]  * ct2);
                const double ccy = -((double)curW[4] * ct0 + (double)curW[5] * ct1 + (double)curW[6]  * ct2);
                const double ccz = -((double)curW[8] * ct0 + (double)curW[9] * ct1 + (double)curW[10] * ct2);
                const double ppx = -((double)preW[0] * pt0 + (double)preW[1] * pt1 + (double)preW[2]  * pt2);
                const double ppy = -((double)preW[4] * pt0 + (double)preW[5] * pt1 + (double)preW[6]  * pt2);
                const double ppz = -((double)preW[8] * pt0 + (double)preW[9] * pt1 + (double)preW[10] * pt2);
                const double dx = ccx - ppx, dy = ccy - ppy, dz = ccz - ppz;

                memcpy(prevWorldRel, preW, sizeof(prevWorldRel));
                for (int i = 0; i < 3; ++i)
                    prevWorldRel[12 + i] = (float)((double)prevWorldSaved[0 + i] * dx
                                                 + (double)prevWorldSaved[4 + i] * dy
                                                 + (double)prevWorldSaved[8 + i] * dz);
                taaMul(prevVPrel, prevProjSaved, prevWorldRel);

                // reproj = (P_prev * W_prev_rel) * (W_curr_rel^-1 * P_curr^-1).
                // W_curr_rel is [R_curr | 0], so its inverse is [R_curr^T | 0] -
                // exact, no division at all.
                float invWorldRel[16];
                memset(invWorldRel, 0, sizeof(invWorldRel));
                for (int c = 0; c < 3; ++c)
                    for (int rr = 0; rr < 3; ++rr)
                        invWorldRel[c*4 + rr] = worldRel[rr*4 + c];   // transpose
                invWorldRel[15] = 1.0f;

                float invCurrExact[16];
                taaMul(invCurrExact, invWorldRel, invProj);

                // ---- THE MATRIX THE SHADER GETS IS VIEW-TO-CLIP, NOT CLIP-TO-CLIP.
                //
                // A clip-to-clip reprojection cannot be applied in float32 with
                // this projection. Its w row comes out at +-1/near = +-61.9, and
                // the shader then evaluates 61.9 * (w_clip - z_clip) - two
                // nearly equal large numbers subtracted. At 8 km that needs six
                // significant digits and float32 has seven; at sky distances it
                // has none, prev.w collapses to noise, and the divide by it
                // inflated the vectors by the measured 3x to 21x.
                //
                // With m10 = m11 = -1 the projection gives z_clip = w_clip +
                // m14, so z_clip carries NO information beyond w_clip - depth is
                // entirely in w. View space therefore rebuilds with no
                // subtraction at all:
                //
                //     view = (x_clip/sx, y_clip/sy, -w_clip, 1)
                //
                // The shader builds (pos.x, pos.y, pos.w, 1) and this matrix
                // absorbs 1/sx, 1/sy and the -1. Nothing cancels anywhere.
                //
                //     M = P_prev * [R_prev*R_curr^T | R_prev*dC] * diag(1/sx, 1/sy, -1, 1)
                //
                // Every factor is exact: a transpose for the rotation, the
                // millimetre-scale camera delta already differenced in double,
                // and three reciprocals of well-scaled numbers.
                float relRot[16];
                memset(relRot, 0, sizeof(relRot));
                for (int c = 0; c < 3; ++c)
                    for (int rr = 0; rr < 3; ++rr) {
                        double s = 0.0;
                        for (int k = 0; k < 3; ++k)
                            s += (double)preW[k*4 + rr] * (double)curW[k*4 + c];
                        relRot[c*4 + rr] = (float)s;      // R_prev * R_curr^T
                    }
                for (int i = 0; i < 3; ++i)
                    relRot[12 + i] = prevWorldRel[12 + i];   // R_prev * dC
                relRot[15] = 1.0f;

                float clipToView[16];
                memset(clipToView, 0, sizeof(clipToView));
                clipToView[0]  = (fresh.proj[0] != 0.0f) ? 1.0f / fresh.proj[0] : 1.0f;
                clipToView[5]  = (fresh.proj[5] != 0.0f) ? 1.0f / fresh.proj[5] : 1.0f;
                clipToView[10] = -1.0f;   // view z from the w the shader passes in slot 2
                clipToView[15] = 1.0f;

                float viewToPrevClip[16], m2[16];
                // ---- WITH TAA_MV_EYE THE SHADER SUPPLIES VIEW SPACE ITSELF.
                //
                // clipToView divides one globally sampled projection back out of
                // clip space. The offending shader selects mvp_matrix[] and
                // modelview_matrix[] per INSTANCE and writes gl_ViewportIndex,
                // so a single draw can emit through several projections and no
                // one inverse fits them all. The leftover is the parallax term
                // M[12]/d, which is why the error tracks 1/depth and why near
                // ground is worst while distant geometry looks clean.
                //
                // When the vertex shader reads its own eye-space varying there
                // is nothing to invert, so the matrix is just prevProj*relRot.
                static const bool useEye = (getenv("TAA_MV_EYE") != nullptr);
                if (useEye) {
                    taaMul(viewToPrevClip, prevProjSaved, relRot);
                } else {
                    taaMul(m2, relRot, clipToView);
                    taaMul(viewToPrevClip, prevProjSaved, m2);
                }

                static const bool useClipToClip = (getenv("TAA_MV_CLIP2CLIP") != nullptr);
                if (useClipToClip) {
                    static const bool useNumeric = (getenv("TAA_MV_NUMERIC_INV") != nullptr);
                    taaMul(r, prevVPrel, useNumeric ? invCurr : invCurrExact);
                } else {
                    memcpy(r, viewToPrevClip, sizeof(r));
                }

                // ---- THE ANGLE, FROM THE SAME TWO MATRICES.
                //
                // far and the plugin's estimate disagree while both claim to
                // describe the pair (world(N-1), world(N)). Exactly one of three
                // things is wrong: the matrices are not that pair, the plugin's
                // trace formula, or the extraction of far from the reprojection.
                //
                // Computing the angle HERE, from fresh.world and prevWorldSaved,
                // separates them. If it agrees with the plugin, the matrices are
                // one step apart and far is being extracted wrongly. If it
                // agrees with far, the plugin's formula is wrong. It is the same
                // trace identity: tr(R_curr^T R_prev) = 1 + 2cos(a).
                // Phases 3, 4 and 5 are YAW, YAW-LEFT and PITCH - the only
                // ones with a rotation to measure. Firing every 600 frames
                // instead landed almost every sample outside them: four
                // consecutive MV ANGLE lines read "trace says 0.000 px" while
                // the disagreement they exist to resolve happens only while the
                // camera is turning. Matched to the dump cadence so each line
                // pairs with a verdict line.
                const int stPhase = fresh.selfTestPhase;
                // EVERY phase, not just the rotating ones. The epipolar
                // probe says the reprojection carries almost no translation -
                // the line it traces from one metre to infinity is 0.00 px
                // through the hold phases and 0.2 to 3 px through the rotations,
                // while the aircraft is flying at cruise. If that is real then
                // near-field geometry cannot reproject correctly no matter how
                // good the rotation is, which is exactly the shape of the
                // residual: median 0.000 px, p95 15 to 330 px.
                //
                // dC is printed in metres so the question stops being an
                // inference. Either the camera delta is genuinely tiny - which
                // would mean the plugin's world matrices are expressed in a
                // frame that travels with the aircraft - or it is being lost
                // between the recovery and the matrix.
                // ---- FIRE IN EVERY VIEW, NOT ONLY DURING THE SELF-TEST.
                //
                // These were gated on a self-test phase, so the one case that is
                // now known to be broken - an EXTERNAL view, which happens with
                // the test finished and the phase back at 0 - produced no
                // diagnostics at all. The residual reads 333 px there against
                // 0.003 px in the cockpit, on a PARKED aeroplane, so the fault
                // is in the camera path and these are the numbers that describe
                // it.
                const bool rotating = (stPhase >= 3 && stPhase <= 5);
                if ((frames % 20) == 0 && g_mv.w) {
                    double tr = 0.0;
                    for (int c = 0; c < 3; ++c)
                        for (int rr = 0; rr < 3; ++rr)
                            tr += (double)fresh.world[c*4+rr] * (double)prevWorldSaved[c*4+rr];
                    double ca = (tr - 1.0) * 0.5;
                    if (ca >  1.0) ca =  1.0;
                    if (ca < -1.0) ca = -1.0;
                    const double ang = acos(ca);
                    const double angPx = ang * (double)fresh.proj[0] * (double)g_mv.w * 0.5;
                    // THE POINT AT INFINITY IS COLUMN 2, NOT COLUMN 3.
                    //
                    // This read r[12]/r[15] - column 3 - which is the image of
                    // the point (0,0,0,1): the camera's own origin. Through a
                    // view-space matrix whose w row is -1/near = -61.9 that is
                    // a division by something near zero, and it duly reported
                    // 4113, 3869 and 20349 px against a trace angle of 0.7 px.
                    // Numbers like that read as a broken reprojection; they
                    // were a broken probe.
                    //
                    // A point at infinity along the centre ray is (0,0,+-1,0),
                    // so its image is +-column 2 and the sign cancels in the
                    // magnitude.
                    double fx = (double)r[8], fw = (double)r[11];
                    const double farPx = fabs(fw) < 1e-12 ? 0.0
                                       : fabs(0.5 * fx / fw) * g_mv.w;
                    // THE DEPTH CONVENTION, TAKEN FROM THE PROJECTION.
                    //
                    // As d -> infinity, z_clip/w_clip -> proj[10]/proj[11].
                    // That is exact and needs no assumption about reverse-Z,
                    // GL-versus-Vulkan ranges, or the sign of view-space z -
                    // all of which have now been guessed at and got wrong.
                    const double m10 = fresh.proj[10], m11 = fresh.proj[11];
                    const double zInf = (fabs(m11) > 1e-12) ? m10 / m11 : 0.0;
                    const double dcLen = sqrt(dx*dx + dy*dy + dz*dz);
                    g_diagDcMetres = (float)dcLen;
                    const double tcam = sqrt((double)relRot[12]*relRot[12]
                                           + (double)relRot[13]*relRot[13]
                                           + (double)relRot[14]*relRot[14]);
                    // COMPONENTS, not just lengths.
                    //
                    // The flow in phase 6 sits 46 to 54 degrees off the epipolar
                    // line it must lie along, while phase 7 - the same geometry
                    // with 58x less translation - is exact to 0.001 degrees. A
                    // wrong axis would be a CONSTANT angle. This one grows
                    // through the phase, so something is accumulating, and the
                    // lengths agreeing tells us nothing about direction: a
                    // rotation preserves length, so |t| = |dC| holds for every
                    // wrong rotation as well as the right one.
                    // ---- THE TWO PROJECTIONS, SIDE BY SIDE.
                    //
                    // M = prevProj * relRot * clipToView, and clipToView[5] is
                    // 1/proj[5] from THIS frame while prevProj is LAST frame's.
                    // So M[5] = prevProj[5]/proj[5], and if those disagree the
                    // result is prev.y = (that ratio) * curr.y - a Y-only scale
                    // with vx exactly zero, which is the measured signature of
                    // the band down to the sign.
                    trace("MV PROJ: view=%d | proj[0]=%.5f proj[5]=%.5f | "
                          "prevProj[0]=%.5f prevProj[5]=%.5f | M[5] would be "
                          "%.5f (1.0 means they agree)",
                          fresh.viewType, (double)fresh.proj[0], (double)fresh.proj[5],
                          (double)prevProjSaved[0], (double)prevProjSaved[5],
                          fresh.proj[5] != 0.0f
                              ? (double)prevProjSaved[5] / (double)fresh.proj[5]
                              : 0.0);
                    trace("MV DELTA: view=%d phase=%d camera moved %.4f m between these "
                          "two frames; the same delta in previous-camera axes is "
                          "%.4f m | dC=(%+.4f, %+.4f, %+.4f) t=(%+.4f, %+.4f, "
                          "%+.4f) | cam=(%.2f, %.2f, %.2f)",
                          fresh.viewType, stPhase, dcLen, tcam, dx, dy, dz,
                          (double)relRot[12], (double)relRot[13], (double)relRot[14],
                          ccx, ccy, ccz);
                    trace("MV ANGLE: same two matrices - trace says %.3f px, "
                          "reprojection says %.3f px, plugin says %.3f px | "
                          "proj[10]=%.5f proj[11]=%.5f proj[14]=%.5f -> "
                          "infinity is ndcZ=%.5f",
                          angPx, farPx, (double)fresh.selfTestExpectedPx,
                          m10, m11, (double)fresh.proj[14], zInf);
                }
                if (!g_usePluginReproj) memcpy(g_velSnap.reproj, r, sizeof(r));

                // Both, once in a while, so the difference is a measurement
                // rather than a claim. Column 3 plus column 2 is the centre ray
                // at infinity; half of it, in pixels, is the displacement.
                if (rotating && (frames % 20) == 0 && g_mv.w) {
                    // Column 2 alone. col3 + col2 is the image of (0,0,1,1) -
                    // a point one metre along the centre ray - not a point at
                    // infinity, and at one metre the translation term still
                    // dominates. Same correction as MV ANGLE above.
                    auto farPx = [&](const float *m) {
                        double x = (double)m[8];
                        double w = (double)m[11];
                        return fabs(w) < 1e-12 ? 0.0 : fabs(0.5 * x / w) * g_mv.w;
                    };
                    trace("MV REPROJ: layer-paired %.3f px vs plugin-paired %.3f px "
                          "at infinity - a ratio far from 1 means the flight loop "
                          "and the renderer are not stepping together",
                          farPx(r), farPx(fresh.reproj));
                }
            }

            memcpy(prevPrevWorldSaved, prevWorldSaved, sizeof(prevPrevWorldSaved));
            memcpy(prevPrevProjSaved,  prevProjSaved,  sizeof(prevPrevProjSaved));
            havePrevPrev = havePrevFrame;
            memcpy(prevWorldSaved, fresh.world, sizeof(prevWorldSaved));
            memcpy(prevProjSaved,  fresh.proj,  sizeof(prevProjSaved));
            prevSavedFrame = fresh.frame;
            havePrevFrame = true;

            // Counted, because "the pairing is fine" is exactly the kind of
            // thing that has been assumed here before and was not.
            {
                static uint64_t nAdj = 0, nGap = 0;
                if (adjacent) ++nAdj; else ++nGap;
                if (((nAdj + nGap) % 600) == 0)
                    trace("MV PAIRING: adjacent %llu, non-adjacent %llu - "
                          "non-adjacent frames fall back to the plugin's matrix",
                          (unsigned long long)nAdj, (unsigned long long)nGap);
            }
        }
    }

    // Report the very first call unconditionally, so "hook never invoked" is
    // distinguishable from "invoked but my logging condition never fired". That
    // ambiguity cost real time on the sibling project.
    if (frames == 1) trace("PRESENT: first call (share=%s)", g_share ? "attached" : "null");

    // Report once whether the present queue is one the app renders on. If it is
    // not, submission order guarantees us nothing and the whole approach of
    // submitting at present time is unsound.
    static bool queueChecked = false;
    if (!queueChecked && !g_submitQueues.empty()) {
        queueChecked = true;
        std::lock_guard<std::mutex> g(g_lock);
        bool same = g_submitQueues.count(queue) > 0;
        trace("QUEUE: present queue %p %s one the app submits rendering to "
              "(%u distinct app queues) -- %s",
              (void*)queue, same ? "IS" : "is NOT",
              (unsigned)g_submitQueues.size(),
              same ? "submission order sequences us after the frame"
                   : "NO ORDERING GUARANTEE - our depth read races their writes");
    }

    openShare();
    selectSceneDepth();

    // Report what this layer can actually do, once per attach.
    //
    // Written from the layer rather than guessed by the plugin, because the
    // plugin has no way to know whether the layer is loaded at all, let alone
    // which backends were compiled into it. Availability is stated per backend
    // with a reason, so "FSR 2 - not built into this layer" and "FSR 2 -
    // unsupported GPU" are distinguishable in the UI instead of both reading as
    // a blank entry.
    // RE-ASSERTED EVERY FRAME, not written once.
    //
    // The plugin memsets the whole shared block when it maps it, and that
    // happens on entering flight - AFTER the layer has already reported. With a
    // one-shot write the flag was simply erased and never restored, so the
    // panel read "Vulkan layer NOT attached" while the layer was demonstrably
    // running and had logged its own attach a moment earlier.
    //
    // Worse than cosmetic: the plugin refuses every upscaler when the layer
    // looks absent, so the whole path switched itself off. Intermittent,
    // because it depended purely on whether the plugin's map landed before or
    // after the layer's report.
    //
    // One store per frame is free next to everything else here, and it cannot
    // be lost by anyone else clearing the block.
    if (g_share) g_share->layerAttached = 1;

    // ================= CRASH DESTRUCTION: DISCOVERY =======================
    //
    // Two frames, driven from the control file. Frame one publishes the grid
    // and switches the vertex patch on with the occupancy region cleared;
    // frame two reads back what the airframe's own vertices marked.
    //
    // Two frames and not one because the write happens on the GPU during the
    // frame we arm it. Reading in the same frame would report whatever was
    // there before the draws ran, which is zero - and zero is
    // indistinguishable from "the transform is wrong and nothing landed in the
    // grid", which is the single most important failure this is looking for.
    // ---- STAMP THE AIRFRAME INTO OCCUPANCY, ONCE PER AEROPLANE.
    //
    // The displacement gate asks whether a cell is airframe. Nothing filled
    // that, so it refused everything - which is why a 5 m test offset produced
    // a completely normal sim rather than a moved aeroplane.
    //
    // From the .acf rather than from the GPU discovery pass. Discovery
    // classifies EVERY vertex the sim draws, so a parked aeroplane marks the
    // terminal building beside it; the measured box came out asymmetric,
    // -27.4 to +33.2, which a symmetric aeroplane cannot do. It is not fixable
    // by tightening the classification, because a vertex carries no flag
    // saying which model it came from.
    //
    // Keyed on the PATH: parsing 49105 properties costs about a tenth of a
    // second, which is fine once per aircraft and not fine per frame.
    // Say which gate is shut, once. Four conditions and a silent skip is the
    // same shape of problem as the discard word: no output is not a diagnosis.
    if (g_share && crashEnabled()) {
        static bool saidGate = false;
        if (!saidGate && destructgpu::state().ready) {
            saidGate = true;
            trace("DESTRUCT: voxelise gate - ready=%d path='%s' nx=%d cell=%.2f",
                  destructgpu::state().ready ? 1 : 0,
                  g_share->crashAcfPath[0] ? g_share->crashAcfPath : "(empty)",
                  g_share->crashNx, (double)g_share->crashCell);
        }
    }

    if (g_share && crashEnabled() && destructgpu::state().ready &&
        g_share->crashAcfPath[0] && g_share->crashNx > 0) {
        static std::string voxPath;
        static float       voxCell = 0.0f;
        // The GRID matters as much as the path: the same aeroplane on a
        // different grid needs restamping, or the occupancy describes cells
        // that have moved.
        if (voxPath != g_share->crashAcfPath || voxCell != g_share->crashCell) {
            voxPath = g_share->crashAcfPath;
            voxCell = g_share->crashCell;

            destruct::Airframe frame;
            if (destruct::parseAcf(voxPath.c_str(), frame)) {
                destruct::Grid g;
                g.min[0] = g_share->crashGridMin[0];
                g.min[1] = g_share->crashGridMin[1];
                g.min[2] = g_share->crashGridMin[2];
                g.cell   = g_share->crashCell;
                g.nx = g_share->crashNx; g.ny = g_share->crashNy; g.nz = g_share->crashNz;

                const uint32_t cells = destruct::gpuCellCount(g);
                std::vector<unsigned char> occ(cells ? cells : 1, 0);
                const uint32_t hull = destruct::voxeliseAirframe(
                    frame, g_share->crashRefOffset, g, occ.data(), cells);
                const uint32_t gear = destruct::voxeliseGear(
                    frame, g_share->crashRefOffset, g, occ.data(), cells);

                uint32_t total = 0;
                for (uint32_t i = 0; i < cells; ++i) if (occ[i]) ++total;

                destructgpu::writeOccupancy(occ.data(), cells);

                const float frac = cells ? (float)total / (float)cells : 0.0f;
                trace("DESTRUCT: airframe voxelised from %s - %u of %u cells "
                      "(%.1f%%), hull %u gear %u. This is the aeroplane's own "
                      "geometry, so it contains no scenery however close the "
                      "sim is parked to it.",
                      voxPath.c_str(), total, cells, (double)(frac * 100.0f),
                      hull, gear);
            } else {
                trace("DESTRUCT: could not read the airframe from '%s' - "
                      "occupancy stays empty, so displacement will move "
                      "nothing rather than move the wrong thing.",
                      voxPath.c_str());
            }
        }
    }

    if (g_share && crashEnabled() && destructgpu::state().ready) {
        static int discoverPhase = 0;   // 0 idle, 1 armed, 2 read next frame
        static uint32_t discoverCells = 0;

        const bool want = live::onoff("crash.discover", "TAA_CRASH_DISCOVER", false);

        // The matrix is refused rather than defaulted. A zeroed
        // crashAircraftInv means the plugin could not invert the body-to-view
        // matrix, and classifying against identity would fill the grid with
        // clip-space nonsense that looks like a working discovery.
        bool haveMatrix = false;
        for (int mi = 0; mi < 16; ++mi)
            if (g_share->crashAircraftInv[mi] != 0.0f) { haveMatrix = true; break; }

        if (want && discoverPhase == 0) {
            if (!haveMatrix) {
                static bool said = false;
                if (!said) {
                    said = true;
                    trace("DESTRUCT: discovery asked for but the clip-to-aircraft "
                          "matrix is not available - the plugin could not invert "
                          "the body-to-view transform. Refusing rather than "
                          "classifying against identity.");
                }
            } else if (g_share->crashNx > 0 && g_share->crashCell > 0.0f) {
                destruct::Grid g;
                g.min[0] = g_share->crashGridMin[0];
                g.min[1] = g_share->crashGridMin[1];
                g.min[2] = g_share->crashGridMin[2];
                g.cell   = g_share->crashCell;
                g.nx = g_share->crashNx; g.ny = g_share->crashNy; g.nz = g_share->crashNz;
                discoverCells = destruct::gpuCellCount(g);

                destructgpu::clearOccupancy(discoverCells);
                destructgpu::clearDiscard();
                destructgpu::uploadHeader(g_share->crashAircraftInv,
                                          g_share->crashAircraftFwd,
                                          g.min, g.cell,
                                          g.nx, g.ny, g.nz, 1,
                                          crashTestOffset());
                discoverPhase = 1;
                trace("DESTRUCT: discovery armed - %u cells of %.2f m, grid "
                      "%dx%dx%d from (%.1f %.1f %.1f)",
                      discoverCells, (double)g.cell, g.nx, g.ny, g.nz,
                      (double)g.min[0], (double)g.min[1], (double)g.min[2]);
            }
        } else if (discoverPhase == 1) {
            // Let the armed frame actually render before believing the buffer.
            discoverPhase = 2;
        } else if (discoverPhase == 2) {
            std::vector<unsigned char> occ(discoverCells ? discoverCells : 1, 0);
            const uint32_t hit = destructgpu::readOccupancy(occ.data(), discoverCells);
            const float frac = discoverCells ? (float)hit / (float)discoverCells : 0.0f;

            destruct::Grid g;
            g.min[0] = g_share->crashGridMin[0];
            g.min[1] = g_share->crashGridMin[1];
            g.min[2] = g_share->crashGridMin[2];
            g.cell   = g_share->crashCell;
            g.nx = g_share->crashNx; g.ny = g_share->crashNy; g.nz = g_share->crashNz;

            float rMin[3], rMax[3];
            const bool refined = destruct::refineBounds(g, occ.data(), rMin, rMax);

            // The plan's gate, stated in its own terms, because a number with
            // no verdict beside it gets read as whatever the reader hoped.
            const char *verdict =
                (frac < 0.02f) ? "TOO LOW - classification is missing the aircraft" :
                (frac > 0.40f) ? "TOO HIGH - the box or the transform is catching the world" :
                                 "plausible";

            // The discard word separates the two ways this can read zero.
            // Without it, "no patched shader ran" and "every vertex was
            // rejected by the transform" are the same number with completely
            // different causes.
            const uint32_t disc = destructgpu::readDiscard();
            trace("DESTRUCT: discovery %u of %u cells occupied (%.1f%%) - %s",
                  hit, discoverCells, (double)(frac * 100.0f), verdict);
            trace("DESTRUCT: %llu vertex module(s) carry the occupancy write",
                  (unsigned long long)spvinj::occupancyVsCount());

            // ---- THE COVERAGE NUMBERS, IN FULL, ONCE.
            //
            // These counters are otherwise only ever seen through a
            // "% 500 == 1" trace, so the log said "1 layout(s) carry the
            // fragment set" whether the true figure was 1 or 499. That is a
            // LOWER BOUND being read as a count, and it is the reason it was
            // not possible to tell whether layout coverage matched the 307
            // patched modules - the single most useful comparison there is.
            //
            // A patched shader whose pipeline layout does not declare the set
            // cannot write, however correct its arithmetic, so these two
            // numbers failing to correspond is a diagnosis on its own.
            trace("DESTRUCT: coverage - %llu layout(s) extended, %llu refused "
                  "(already past index %u or at the device limit), %llu "
                  "pipeline bind(s) carried the set, %llu draw-time rebind(s)",
                  (unsigned long long)destructgpu::layoutsExtended(),
                  (unsigned long long)destructgpu::layoutsTooMany(),
                  destructgpu::state().setIndex,
                  (unsigned long long)destructgpu::bindsIssued(),
                  (unsigned long long)destructgpu::drawRebinds());
            trace("DESTRUCT: discard word = %u. %s", disc,
                  disc ? "The shader DID run, so the occupancy result is a "
                         "verdict on the transform."
                       : "The shader did NOT run - no patched vertex shader "
                         "reached the store, so this says nothing about the "
                         "transform. Look at emission and binding first.");

            if (refined)
                trace("DESTRUCT: measured airframe box (%.2f %.2f %.2f) to "
                      "(%.2f %.2f %.2f), %.1f x %.1f x %.1f m",
                      (double)rMin[0], (double)rMin[1], (double)rMin[2],
                      (double)rMax[0], (double)rMax[1], (double)rMax[2],
                      (double)(rMax[0] - rMin[0]), (double)(rMax[1] - rMin[1]),
                      (double)(rMax[2] - rMin[2]));
            else
                trace("DESTRUCT: nothing occupied. The transform put the "
                      "aeroplane somewhere other than the grid - this is the "
                      "answer, not an absence of one.");

            // Switch the vertex writes back off and clear the one-shot, so a
            // key left set in the file costs one discovery rather than one per
            // frame forever.
            destructgpu::uploadHeader(g_share->crashAircraftInv,
                                      g_share->crashAircraftFwd,
                                      g.min, g.cell,
                                      g.nx, g.ny, g.nz, 0,
                                      crashTestOffset());
            live::clearOneShot("crash.discover");
            discoverPhase = 0;
        }

        if (!want && discoverPhase == 0) {
            // Keep the header current even when idle, so the first armed frame
            // is not classifying against a matrix from several seconds ago.
            // ---- ACTIVE IS NOT THE SAME QUESTION AS DISCOVERING.
            //
            // The shader's `active` flag drives ONE select that both the
            // occupancy write and the displacement read, so uploading 0 here
            // told every vertex to skip - and the 5 m test offset moved
            // precisely nothing. Measured rather than assumed: two captures
            // either side of the key differed by 801 pixels out of 950300,
            // which is instrument animation.
            //
            // Discovery is one reason to be active. A displacement is another,
            // and it outlives the two frames discovery runs for. So the flag is
            // the union, and the fragment transforms will join it at Task 11
            // rather than adding a third switch.
            // gridDim.w is DISCOVER and nothing else. It briefly also meant
            // "displacing", back when one flag drove both; testOffset.w carries
            // that now. Leaving it joined meant setting a test offset switched
            // the occupancy WRITE on, which filled the very cells the
            // displacement gate then read - and moved the entire screen.
            const float *tOff = crashTestOffset();
            const int    live = 0;
            float nudge[3];
            crashGridNudge(nudge);
            const float gmin[3] = { g_share->crashGridMin[0] + nudge[0],
                                    g_share->crashGridMin[1] + nudge[1],
                                    g_share->crashGridMin[2] + nudge[2] };
            destructgpu::uploadHeader(g_share->crashAircraftInv,
                                      g_share->crashAircraftFwd,
                                      gmin, g_share->crashCell,
                                      g_share->crashNx, g_share->crashNy,
                                      g_share->crashNz, live,
                                      tOff);
        }
    }


    if (g_share && !g_availReported) {
        g_availReported = true;

        VkPhysicalDeviceProperties props;
        memset(&props, 0, sizeof(props));
        {
            std::lock_guard<std::mutex> g(g_lock);
            std::map<void*, DeviceData>::iterator it = g_devices.begin();
            if (it != g_devices.end() && g_getPhysProps && it->second.phys)
                g_getPhysProps(it->second.phys, &props);
        }
        if (props.deviceName[0]) {
            strncpy(g_share->gpuName, props.deviceName, sizeof(g_share->gpuName) - 1);
            g_share->gpuName[sizeof(g_share->gpuName) - 1] = 0;
        }
        g_share->gpuVendorId = props.vendorID;

        // ---- THE AVAILABILITY THIS TRACE HAS ALWAYS CLAIMED TO REPORT.
        //
        // It did not. The array was left at zero, which is TAA_AVAIL_UNKNOWN,
        // while the line below said "reported attach + availability" - so the
        // only symptom of the gap was a UI that offered nothing and a log that
        // said it had been told everything.
        //
        // Every slot is written from one call, so a backend added to the enum
        // cannot be silently left out of the report.
        upscaler::availabilityAll(g_upscalerCaps, g_share->availability);

        MemoryBarrier();
        trace("SHARE: reported attach + availability (gpu=%s, vendor=0x%04X)",
              g_share->gpuName, (unsigned)props.vendorID);
        for (int ui = 0; ui < TAA_UPSCALER_COUNT; ++ui)
            trace("UPSCALER: %-5s %s", upscaler::name(ui),
                  upscaler::availabilityName(g_share->availability[ui]));
    }

    // Report the 3D colour target once, with the verdict on whether a resolve
    // can touch it as it stands.
    //
    // This is the same question that cost time on depth: an application sets
    // only the usage flags it needs, and an image it merely renders into needs
    // neither SAMPLED nor STORAGE. Without SAMPLED we cannot read the frame;
    // without STORAGE we cannot write the result back in place. Both are fixable
    // by adding flags in vkCreateImage - the layer already sees every creation -
    // but that is a real change to something X-Plane owns, so it is worth
    // knowing before designing around it rather than after.
    if (!g_sceneColorReported && g_sceneColor.image != VK_NULL_HANDLE) {
        g_sceneColorReported = true;
        bool sampled = (g_sceneColor.usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0;
        bool storage = (g_sceneColor.usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0;
        bool xfer    = (g_sceneColor.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;
        // The numeric format goes in alongside the name: the name table only
        // covers depth formats, so a colour format prints as "?" and the one
        // piece of information that identifies it is lost.
        trace("COLOR: scene target %ux%u fmt=%s(%d) samples=%u usage=0x%x layout=%d",
              g_sceneColor.w, g_sceneColor.h, formatName(g_sceneColor.format),
              (int)g_sceneColor.format,
              (unsigned)g_sceneColor.samples, g_sceneColor.usage,
              (int)g_sceneColor.layout);
        trace("COLOR: readable(SAMPLED)=%s  writable(STORAGE)=%s  copyable(TRANSFER_SRC)=%s",
              sampled ? "yes" : "NO", storage ? "yes" : "NO", xfer ? "yes" : "NO");
        if (!sampled || !storage) {
            trace("COLOR: the resolve cannot work in place on this image as created. "
                  "Options: add usage flags in vkCreateImage, or resolve into our own "
                  "image and blit back.");
        }
        if (g_sceneColor.samples != VK_SAMPLE_COUNT_1_BIT) {
            if (g_sceneResolveImage != VK_NULL_HANDLE) {
                std::map<VkImage, ColorTarget>::iterator rt =
                    g_colorImages.find(g_sceneResolveImage);
                trace("COLOR: multisampled, but the pass RESOLVES (mode=%u) into "
                      "%ux%u fmt=%d usage=0x%x - that single-sample image is the "
                      "upscaler input, handed to us directly.",
                      g_sceneResolveMode,
                      rt != g_colorImages.end() ? rt->second.w : 0,
                      rt != g_colorImages.end() ? rt->second.h : 0,
                      rt != g_colorImages.end() ? (int)rt->second.format : -1,
                      rt != g_colorImages.end() ? rt->second.usage : 0u);
            } else {
                trace("COLOR: WARNING multisampled with no resolve attachment - an "
                      "upscaler wants a single-sample input, so the insertion point "
                      "has to move to wherever X-Plane resolves this.");
            }
        }
    }

    Snapshot snap = {};
    bool have = snapshot(&snap);

    if ((frames % 120) == 0) {
        trace("FRAME %llu  shareframe=%llu valid=%d  cam=(%.3f %.3f %.3f) moved=%.5fm "
              "revZ=%d near=%.4f objs=%d reset=%d passes=%u depth=%s",
              (unsigned long long)frames, (unsigned long long)snap.frame,
              have ? 1 : 0, snap.camX, snap.camY, snap.camZ, snap.camDelta,
              snap.reverseZ, snap.nearClip, snap.objectCount,
              snap.historyReset, g_passesThisFrame,
              g_sceneDepth != VK_NULL_HANDLE ? "found" : "MISSING");

        // Render-pass sizes, once, so the 3D/UI boundary can be read off rather
        // than guessed at.
        if (frames == 120 || frames == 600) {
            uint32_t n = g_passesThisFrame < 32 ? g_passesThisFrame : 32;
            trace("PASSES this frame (%u total, first %u shown):", g_passesThisFrame, n);
            // The 3D/UI boundary is the last pass that still has a depth
            // attachment: the scene renders with depth, the 2D overlays do not.
            // The resolve has to land there, before instrument text and ATC
            // boxes can be temporally smeared.
            int lastDepth = -1;
            for (uint32_t i = 0; i < n; ++i) {
                trace("   [%2u] %ux%u  depth=%s  color=%u",
                      i, g_passSizes[i][0], g_passSizes[i][1],
                      g_passHasDepth[i] ? "yes" : "no ", g_passColorCount[i]);
                if (g_passHasDepth[i]) lastDepth = (int)i;
            }
            trace("   last depth pass = %d  -> resolve insertion point is after it",
                  lastDepth);
        }
    }

    // Dump interval, re-read from a FILE at runtime rather than fixed at
    // startup from an environment variable.
    //
    // An env var is latched when the process starts, so turning dumps on meant
    // restarting the sim, re-loading an aircraft and re-flying - and twice that
    // was discovered only after the flight was already over and nothing had
    // been captured. A file can be changed while the sim is running.
    //
    // %TEMP%	aa_dump_every.txt containing an integer: frames between dumps,
    // 0 to disable. Checked once a second, which costs nothing.
    {
        static uint64_t lastCheck = 0;
        if (frames - lastCheck >= 60) {
            lastCheck = frames;
            const char *t = getenv("TEMP");
            // "\\taa_..." - the backslash MUST be escaped. This read
            // "\taa_dump_every.txt", where \t is a TAB, so it looked for a file
            // called <TAB>aa_dump_every.txt and silently found nothing. The
            // runtime dump control advertised in the comments had never worked.
            std::string path = std::string(t ? t : ".") + "\\taa_dump_every.txt";
            FILE *f = fopen(path.c_str(), "r");
            if (f) {
                int v = 0;
                if (fscanf(f, "%d", &v) == 1 && v >= 0 && v != dumpEvery) {
                    trace("VEL: dump interval -> %d frames (was %d)", v, dumpEvery);
                    dumpEvery = v;
                }
                fclose(f);
            }
        }
    }

    // RESOLVE MODE, switchable while the sim is running - same reasoning as the
    // dump interval above, for the same reason it was written.
    //
    // The outline has now survived four explanations: the alpha channel, the
    // min/max clamp, the jitter, and the jitter delta. Two of those were real
    // bugs and got fixed; none of them was this. That record is the argument
    // against producing a fifth theory and spending a flight on it.
    //
    // These two modes do not test an idea, they SPLIT THE PROBLEM. Pass-through
    // writes the current frame straight out - no history sample, no
    // reprojection, no blend - so an outline that survives it is in the read,
    // the copy, the barriers or the target selection, and every theory about
    // temporal logic is irrelevant. Debug 2 paints red where history was
    // rejected and green where it was kept, which images the mechanism
    // directly rather than inferring it from how the result looks.
    //
    // Being a file rather than an env var means the modes can be cycled from
    // outside while a flight continues, so one launch answers all of it. Fly
    // per hypothesis was the thing this project kept doing wrong.
    //
    // %TEMP%\taa_resolve_mode.txt: 0 normal, 1 pass-through, 2 show history
    // rejection, 3 show velocity.
    {
        static uint64_t lastCheck = 0;
        static int lastMode = -1;
        if (frames - lastCheck >= 60) {
            lastCheck = frames;
            const char *t = getenv("TEMP");
            std::string path = std::string(t ? t : ".") + "\\taa_resolve_mode.txt";
            FILE *f = fopen(path.c_str(), "r");
            if (f) {
                int v = 0;
                if (fscanf(f, "%d", &v) == 1 && v >= 0 && v <= 5 && v != lastMode) {
                    lastMode = v;
                    static const char *kName[6] = {
                        "normal",
                        "PASS-THROUGH (no history, no blend)",
                        "DEBUG: red=history rejected, green=kept",
                        "DEBUG: velocity field",
                        "NO JITTER DELTA (history sampled at uv - velocity only)",
                        "JITTER OFF (accumulation still running)" };
                    trace("RESOLVE: mode -> %d, %s", v, kName[v]);
                }
                fclose(f);
            }
        }
    }


    // ---- velocity pass.
    //
    // Opt-in via TAA_VELOCITY=1. Unset, the layer stays purely observational
    // and cannot influence rendering at all - which is the state anyone should
    // be able to fall back to instantly if something looks wrong.
    // DISARMED unless TAA_VELOCITY is explicitly "1".
    //
    // Presence of the variable used to be enough, which meant a leftover
    // TAA_VELOCITY= from an earlier shell still armed it. Requiring an exact
    // value makes "off" the state you get from anything other than a
    // deliberate opt-in.
    //
    // Off, this layer forwards every call unmodified: no GPU work, no
    // allocations, no barriers on X-Plane's resources.
    static const bool velWanted = []{
        const char *e = getenv("TAA_VELOCITY");
        bool on = (e && e[0] == '1' && e[1] == '\0');
        trace("VEL: velocity pass %s", on ? "ARMED (TAA_VELOCITY=1)"
                                          : "DISARMED - observation only");
        return on;
    }();
    g_velArmed = velWanted;
    static bool velInitTried = false;

    // Tear down BEFORE any other use, and outside the selection path, so a
    // recreated depth buffer can never be sampled through a stale handle.
    // ---- resolve creation. Needs the velocity pass up first: it samples the
    // velocity image, so there is nothing to build a descriptor against until
    // that exists. It also needs the scene colour target, which only the frame
    // can identify.
    // Wait for the scene colour target to STOP CHANGING before building
    // against it.
    //
    // During load, X-Plane draws full-viewport depth-bearing passes into
    // targets that are not the 3D scene, and the first version latched one of
    // those: it built against an R8G8B8A8_SRGB image, ran for exactly one
    // frame, then tore itself down when the real R16G16B16A16_SFLOAT target
    // appeared. Requiring the choice to hold still for a couple of seconds
    // costs nothing and skips the whole unsettled period.
    // Stability is measured on the SET of HDR targets, not on which one is
    // current. The current one alternates every frame by design, so a counter
    // keyed on it resets forever and the resolve is never created at all.
    {
        static size_t lastCount = 0;
        if (g_hdrTargets.size() != lastCount) {
            lastCount = g_hdrTargets.size();
            g_sceneColorStable = 0;
        } else if (g_sceneColorStable < 100000) {
            ++g_sceneColorStable;
        }
    }

    // NOT gated on the compute velocity pass.
    //
    // g_vel.ready and g_velStable both describe the DEPTH-DERIVED pass, which
    // the SPIR-V injection replaced and which is now off by default. This block
    // creates the resolve AND the injected motion-vector target, so requiring
    // that pass meant: pass off -> no resolve, no MV target -> FSR2's gate
    // never opened -> the jitter ran with nothing consuming it. That is the
    // shake, and it was five layers of accidental dependency deep.
    //
    // What this actually needs is a settled scene target and a depth image -
    // which is what it uses. `stable` is now derived from the scene target
    // holding still rather than from a subsystem that may not be running.
    bool velOrInjected = g_spirvLive;
    bool stableEnough  = (g_sceneDepth != VK_NULL_HANDLE);
    // ---- THE VELOCITY TARGET, SIZED TO THE SCENE.
    //
    // Built when the scene's real size is finally known, not at device creation:
    // the render size moves, and a velocity image of different dimensions than
    // the pass it is bound to is a silent mismatch rather than an error.
    //
    // This used to live inside the resolve's setup, so it only happened when an
    // upscaler was being built.
    // Say WHICH condition is holding it up. Five conditions gate this and a
    // silent no-op looks identical whichever one fails.
    if (!g_mv.ready && !g_mv.failed) {
        static uint64_t n = 0;
        if ((n++ % 600) == 0)
            trace("MV GATE: spirvLive=%d depth=%d sceneImg=%p %ux%u stable=%u/120",
                  velOrInjected ? 1 : 0, stableEnough ? 1 : 0,
                  (void*)g_sceneColor.image, g_sceneColor.w, g_sceneColor.h,
                  g_sceneColorStable);
    }
    // ---- AND REBUILT WHEN THE SCENE RESOLUTION MOVES.
    //
    // This was gated on !g_mv.ready alone, so the velocity target was sized
    // ONCE from the first scene target that held still and then never again.
    // Change the render resolution afterwards and the target keeps its old
    // dimensions forever, while the gate below demands
    //
    //     passInfo.w == g_mv.w && passInfo.h == g_mv.h
    //
    // so every real scene pass is refused on size and NOTHING resolves:
    //
    //     TAA GATE: candidate rejected on SIZE alone -
    //               pass 2560x1440 vs velocity target 3840x2160
    //
    // measured at 1440p on a 4K swapchain. Every health flag still reads
    // green - spirvLive, depth, jitter, the injected shaders all fine - which
    // is why this presents as "TAA just does not work at that resolution"
    // rather than as an error anybody could act on.
    //
    // mvCreate ALREADY handles the resize: it returns early when the size
    // matches and tears down first when it does not. Only the caller never
    // asked a second time. Rebuilding is safe because it bumps g_mv.gen, and
    // the resolve's needInit clause compares g_taa.velGen against that
    // generation - so the descriptors are rewritten against the new view
    // instead of keeping a handle to a destroyed one.
    //
    // Still behind g_sceneColorStable >= 120: a resolution change churns the
    // HDR target set, which resets that counter, so the rebuild waits for the
    // new size to settle rather than chasing every intermediate one.
    const bool mvSizeStale = g_mv.ready &&
                             (g_mv.w != g_sceneColor.w || g_mv.h != g_sceneColor.h);
    if (velOrInjected && stableEnough && (!g_mv.ready || mvSizeStale) &&
        !g_mv.failed &&
        g_sceneColor.image != VK_NULL_HANDLE && g_sceneColor.w && g_sceneColor.h &&
        g_sceneColorStable >= 120) {
        if (mvSizeStale)
            trace("MV TARGET: scene is now %ux%u but the velocity target is "
                  "%ux%u - rebuilding. Left stale, every scene pass is "
                  "rejected on size and the resolve never runs.",
                  g_sceneColor.w, g_sceneColor.h, g_mv.w, g_mv.h);
        std::map<void*, DeviceData>::iterator mvi = g_devices.begin();
        if (mvi != g_devices.end())
            mvCreate(mvi->second, mvi->second.device, mvi->second.phys,
                     g_sceneColor.w, g_sceneColor.h);
    }


    // Rebuild if the frame starts using a target we have no view for.
    //
    // This used to compare against a single latched image, which meant it fired
    // every other frame once X-Plane's two targets started alternating - a
    // continuous teardown/rebuild that also permanently disabled the pass,
    // because the retry flag was never cleared. Now it only fires for a target
    // genuinely outside the set we built.





    // A short settling period before dispatching.
    //
    // This is now a convenience, NOT a correctness requirement. It was
    // introduced to mask a load-time flash, and it did - but the flash was
    // caused by submitting our own work to the queue passed to
    // vkQueuePresentKHR and assuming submission order sequenced it after the
    // frame's rendering. X-Plane submits on THREE queues, so that ordering
    // never existed and the depth barrier raced their writes.
    //
    // Recording into X-Plane's own command buffer fixed it properly. Tested
    // with this gate at 5 frames and dumping disabled: injection active from
    // the moment of load, no flash.
    //
    // A small value is still worth keeping: the pass has nothing to contribute
    // during a load - no temporal history to build, nothing on screen to
    // anti-alias - so there is no reason to spend GPU time there.
    static int stableFrames = 0;
    if (have && !snap.historyReset && g_depthFreshThisFrame) ++stableFrames;
    else stableFrames = 0;

    // Overridable so the gate can be tested independently of the injection fix.
    // Both changes landed together, so "no flash" could be either one; lowering
    // this is the only way to tell them apart.
    static const int kStableRequired = []{
        const char *e = getenv("TAA_VELOCITY_STABLE");
        int v = e ? atoi(e) : 10;
        return v < 0 ? 0 : v;
    }();

    // The depth-derived pass is no longer created, dispatched, settled or read
    // back. Everything between here and the jitter accounting used to do that:
    // pick a depth image, build a compute pipeline against it, wait for the
    // scene to hold still, dispatch it every frame and periodically copy the
    // result back for verification.
    //
    // All of it is obsolete. The injected shaders emit velocity from the real
    // clip positions during the scene draw, which is exact rather than
    // reconstructed, costs no extra dispatch, and needs no depth barrier - and
    // that barrier was the confirmed cause of the stutter.
    (void)velWanted; (void)have;

    // Jitter accounting. A count of zero with jitter armed means the viewport
    // hook never fired inside a scene pass, which is a silent failure - the
    // image would look completely normal and the resolve would have nothing to
    // accumulate. Counting it is how that gets noticed.
    //
    // But zero is only a FAULT when something was asking to be jittered. With
    // Jitter is phase two and its amplitude is zero, so this reports the
    // count and the phase and claims nothing else.
    // Distinct phases seen since the last report, because a sample every 600
    // frames against an 8-phase sequence lands on the SAME index every time -
    // 600 is divisible by 8 - and reads as a jitter frozen at phase 2/8 when it
    // is advancing perfectly. Aliasing the instrument against the thing being
    // measured; counting distinct values cannot alias.
    {
        static uint32_t seenMask = 0;
        if (snap.jitterIndex >= 0 && snap.jitterIndex < 32)
            seenMask |= (1u << snap.jitterIndex);
        if (g_jitterArmed && (frames % 601) == 0) {
            int distinct = 0;
            for (int i = 0; i < 32; ++i) if (seenMask & (1u << i)) ++distinct;
            trace("JITTER SEQUENCE: %d distinct phases seen of %d expected",
                  distinct, snap.jitterPhases);
            seenMask = 0;
        }
    }

    if (g_jitterArmed && (frames % 600) == 0) {
        // Pixels, not NDC. The push is in clip units and reads as a meaningless
        // fraction; what matters is whether the offset is the sub-pixel value
        // the sequence asked for, so it is reported in the units it was
        // specified in.
        trace("JITTER: %llu draws offset so far, current=(%.3f %.3f) px "
              "phase %d/%d, amplitude %.2f",
              (unsigned long long)g_jitterApplied,
              snap.jitterX, snap.jitterY, snap.jitterIndex, snap.jitterPhases,
              getenv("TAA_JITTER_SCALE") ? atof(getenv("TAA_JITTER_SCALE")) : 0.0);
        if (g_jitterApplied == 0)
            trace("JITTER: nothing offset - armed=%d but no draw met "
                  "inScene && isGeometry", g_jitterArmed ? 1 : 0);
    }

    // ---- did the resolve actually run this frame?
    //
    // Self-correcting rather than something to be discovered by staring at the
    // picture. If a frame presents with TAA selected and no resolve recorded,
    // that frame reached the screen with the jitter still in it - and a stream
    // of those alternating with resolved frames is precisely what "the screen
    // is shaky" means.
    g_resolveRanThisFrame = false;

    // Measure the injected velocity field. Read one frame after the copy was
    // recorded, so the GPU has certainly finished with it - reading the same
    // frame would race the submission and produce numbers describing nothing.

    if (g_spirvLive && g_mv.ready) {
        mvReport(snap.camDelta, snap.frame);
        if (dumpEvery > 0 && (frames % (uint64_t)dumpEvery) == 0)
            g_mv.wantDump = true;
    }

    // Re-read the live control file. One GetFileAttributesEx and a 64-bit
    // compare in the common case; a reparse only when the timestamp moves.
    live::poll();

    // ---- RENDERDOC MULTI-FRAME CAPTURE, FROM THE INSIDE.
    //
    // renderdoccmd has no frame-count option and F12 takes exactly one frame;
    // the in-app API is the only scriptable path to an N-frame capture. Edge-
    // triggered on the live key changing to a positive value, so the file can
    // sit at rd.capture=3 without re-firing every present; captures land on
    // D: beside the project instead of filling the C: scratchpad.
    {
        static RENDERDOC_API_1_1_2 *rdApi = nullptr;
        static bool rdTried = false;
        static int rdLast = 0;
        const int rdN = live::i("rd.capture", "TAA_RD_CAPTURE", 0);
        if (rdN != rdLast) {
            rdLast = rdN;
            if (rdN > 0) {
                if (!rdTried) {
                    rdTried = true;
                    if (HMODULE m = GetModuleHandleA("renderdoc.dll")) {
                        pRENDERDOC_GetAPI get = (pRENDERDOC_GetAPI)
                            GetProcAddress(m, "RENDERDOC_GetAPI");
                        if (get && get(eRENDERDOC_API_Version_1_1_2,
                                       (void **)&rdApi) == 1 && rdApi) {
                            rdApi->SetCaptureFilePathTemplate(
                                "d:\\Steam Games\\steamapps\\common\\"
                                "X-Plane 12\\MotionVectors\\captures\\xpcap");
                            trace("RD: in-app API live, captures go to "
                                  "MotionVectors\\captures");
                        } else rdApi = nullptr;
                    }
                    if (!rdApi)
                        trace("RD: rd.capture set but renderdoc.dll is not in "
                              "this process - launch the sim under RenderDoc");
                }
                if (rdApi) {
                    rdApi->TriggerMultiFrameCapture((uint32_t)rdN);
                    trace("RD: triggered %d-frame capture", rdN);
                }
            }
        }
    }

    // The VRAM system's frame tick: fresh heap sample, zone update, governor
    // release, recycle trim, priority walk, frame-time feedback. The camera
    // delta feeds the teleport detector; garbage (pre-flight, unset snapshot)
    // is passed as -1 so it can never fake a teleport.
    {
        std::lock_guard<std::mutex> g(g_lock);
        vram::ledgerRt(g_vram[VRAM_RT].bytes + g_vram[VRAM_DEPTH].bytes +
                       g_vram[VRAM_STORAGE].bytes);
    }
    // Camera rotation rate from the view matrix the plugin already
    // publishes: the forward axis's frame-to-frame angle. A fast sustained
    // turn is the "new scenery incoming" signal the predictor floors the
    // zone on.
    float rotDeg = -1.0f;
    if (g_share && g_share->valid) {
        // Camera position feeds the spatial mapping: resources are stamped
        // with where the camera was at their creation, and scored against
        // where it will be.
        vram::noteCamera(g_share->camX, g_share->camY, g_share->camZ);
        static float pf[3] = {0, 0, 0};
        static bool havePf = false;
        const float *mv = g_share->modelview;
        float f0 = mv[8], f1 = mv[9], f2 = mv[10];
        float len = sqrtf(f0*f0 + f1*f1 + f2*f2);
        if (len > 1e-6f) {
            f0 /= len; f1 /= len; f2 /= len;
            if (havePf) {
                float d = f0*pf[0] + f1*pf[1] + f2*pf[2];
                if (d > 1.0f) d = 1.0f; if (d < -1.0f) d = -1.0f;
                rotDeg = acosf(d) * 57.29578f;
            }
            pf[0] = f0; pf[1] = f1; pf[2] = f2; havePf = true;
        }
    }
    vram::onPresent((snap.camDelta >= 0.0f && snap.camDelta < 100000.0f)
                        ? snap.camDelta : -1.0f, rotDeg);

    // Graves are NOT flushed mid-flight. The 8-present window assumed the
    // engine submits recorded work within 8 frames; during scenery churn it
    // holds command buffers far longer, and destroying a parked target under
    // one of them was the deterministic 1:28 DEVICE_LOST. Retired states now
    // stay allocated until DestroyDevice drains them - a bounded VRAM cost
    // (recreations are throttled) bought for absolute lifetime correctness.

    // Flush deferred image destroys whose safety window has passed, and
    // prune stale lifetime-ledger entries so the map stays bounded.
    {
        std::vector<DeferredImgKill> due;
        {
            std::lock_guard<std::mutex> g(g_lock);
            for (size_t i = 0; i < g_deferredImgKills.size();) {
                if (g_frameCount >= g_deferredImgKills[i].due) {
                    due.push_back(g_deferredImgKills[i]);
                    g_deferredImgKills.erase(g_deferredImgKills.begin() + (long)i);
                } else ++i;
            }
            for (std::map<VkImage, uint64_t>::iterator it = g_taaBoundImgs.begin();
                 it != g_taaBoundImgs.end();) {
                if (g_frameCount - it->second > 120) g_taaBoundImgs.erase(it++);
                else ++it;
            }
        }
        for (size_t i = 0; i < due.size(); ++i) {
            PFN_vkDestroyImage nd = nullptr;
            {
                std::lock_guard<std::mutex> g(g_lock);
                std::map<void*, DeviceData>::iterator it =
                    g_devices.find(dispatchKey(due[i].dev));
                if (it != g_devices.end()) nd = it->second.destroyImage;
            }
            if (nd) nd(due[i].dev, due[i].img, due[i].alloc);
        }
    }

    // Zone-driven texture quality caps (SS25/43/47): the create-time pager's
    // thresholds follow the zone unless the environment pinned them - an env
    // variable was always the explicit override here and stays the strongest
    // word. GREEN and YELLOW leave textures entirely alone; each zone above
    // caps streamed scenery harder while the preload set (aircraft, cockpit)
    // keeps a gentler ceiling, which is SS47's "autogen degrades first" made
    // concrete.
    // ---- THE ZONE NO LONGER ARMS THE CREATE-TIME PAGER. (2026-08-16)
    //
    // Shrinking an image at creation is a lie told to the application, and
    // X-Plane checks. Four separate faults were found and fixed here in one
    // night - an unstable per-shape answer, a double drop when the defragmenter
    // recreates an already-reduced image, a refinement that escaped the cached
    // decision, and a policy inherited through a recycled handle - and the load
    // still dies about two minutes in with the pager active, while it is stable
    // for eight-plus minutes with the pager off. Validation reports nothing
    // illegal and the fault vanishes under instrumentation, so what remains is
    // a race we have not yet named.
    //
    // A texture-memory optimisation that ends the session is worth less than
    // the memory it saves, so pressure alone must not turn it on. It stays
    // available for the hunt through TAA_PAGER_DROP_ABOVE / vram.tex_drop_above,
    // which is how every run above armed it; what changes is that a CRITICAL
    // zone can no longer arm it behind the user's back. The rest of the VRAM
    // system - shaping, recycling, priority, the governor, the upload cache -
    // is untouched and still zone-driven, and none of it rewrites a resource.
    static const bool pagerOptIn =
        getenv("TAA_PAGER_DROP_ABOVE") || getenv("TAA_PAGER_AUTOGEN_TO") ||
        live::i("vram.tex_drop_above", nullptr, 0) > 0 ||
        live::i("vram.tex_streamed_to", nullptr, 0) > 0;
    if (!g_pagerEnvLocked && pagerOptIn) {
        struct { uint32_t dropAbove, maxDrop, autogenTo; } zp;
        switch (vram::zone) {
            case vram::GREEN: case vram::YELLOW: zp.dropAbove = 0; zp.maxDrop = 1; zp.autogenTo = 0;    break;
            case vram::ORANGE:   zp.dropAbove = 0;    zp.maxDrop = 1; zp.autogenTo = 4096; break;
            case vram::RED:      zp.dropAbove = 4096; zp.maxDrop = 1; zp.autogenTo = 2048; break;
            default:             zp.dropAbove = 2048; zp.maxDrop = 2; zp.autogenTo = 1024; break;
        }
        zp.dropAbove = (uint32_t)live::i("vram.tex_drop_above", nullptr, (int)zp.dropAbove);
        zp.autogenTo = (uint32_t)live::i("vram.tex_streamed_to", nullptr, (int)zp.autogenTo);
        if (zp.dropAbove != g_pagerDropAbove || zp.autogenTo != g_pagerAutogenTo) {
            trace("VRAMSYS: zone %s texture policy - preload cap %u px "
                  "(drop %u), streamed cap %u px",
                  vram::zoneName(vram::zone), zp.dropAbove, zp.maxDrop,
                  zp.autogenTo);
            g_pagerDropAbove = zp.dropAbove;
            g_pagerMaxDrop   = zp.maxDrop;
            g_pagerAutogenTo = zp.autogenTo;
        }
    }

    // Publish the VRAM system's state to the shared block (SS65): the plugin
    // and the debug window get the zone and the live counters without reading
    // a log.
    if (g_share) {
        g_share->vramZone          = (uint32_t)vram::zone;
        g_share->vramShapedMB      = (uint32_t)(vram::lastReported / 1048576ull);
        g_share->vramUploadKBFrame = (uint32_t)(vram::upBytesLast / 1024ull);
        g_share->vramHeldSubmits   = (uint32_t)vram::heldNow;
        g_share->vramPoolMB        = (uint32_t)(vram::g_poolBytes / 1048576ull);
        g_share->vramAllocFails    = (uint32_t)vram::allocFails.load();
    }
    mvMaybeReport(frames);
    // The near-field threshold, live. Read ONCE per frame here rather than in
    // the push path - the push runs millions of times a frame and a mutexed
    // map lookup there would be the layer becoming the bottleneck it wraps.
    //
    // This is the cockpit-shake fix that has been BUILT AND OFF all along: the
    // injected select replaces prevClip with currClip (velocity zero) for
    // geometry closer than this many metres, which is CORRECT for the panel -
    // it rides the camera, so its true screen motion is zero - and the pushed
    // threshold has been 0 the whole time, so the select never fired and the
    // panel got the scenery's world reprojection instead. That is the shake.
    // ---- JITTER RIDES THE RESOLVE. The old design's invariant, restored.
    //
    // "The resolve is what arms it in normal use, and neither can be left on
    // without the other by accident" - written when the coupling existed,
    // still true, and broken when the resolve was rebuilt: arming became
    // env-only, the sequence generated its 8 phases, and the offsets stayed
    // (0, 0) through every session. Without sub-pixel jitter each frame
    // samples the identical grid, accumulation averages identical samples,
    // and TAA can remove shimmer but never aliasing - which, with X-Plane's
    // FXAA also held off, leaves the picture RAWER than stock. That is
    // "no artifacts but still aliased" in one line.
    //
    // Tied to the live enable per frame, so switching the resolve off also
    // stops the jitter - jitter with nothing accumulating it is deliberate
    // edge crawl. TAA_JITTER still forces it for measurement.
    g_jitterArmed = taaEnabled() || (getenv("TAA_JITTER") != nullptr);
    // Jitter defaults OFF even with the resolve on: the unjitter
    // cancellation has never been verified on screen, and the first flight
    // that ran it showed exactly what uncancelled jitter looks like - a
    // trembling panel and crawling smear in motion. Sub-pixel AA returns
    // via taa.jitter_scale=1 only after a run proves the cancellation
    // (static scene, jitter on: the image must not move AT ALL).
    // ---- DEFAULT 1.0, NOT 0.0. ZERO JITTER IS NO ANTI-ALIASING.
    //
    // The sub-pixel offset is what gives TAA more samples than the raster has
    // pixels; at scale 0 every frame samples the same points and there is
    // nothing for the resolve to average. Independently fatal alongside
    // taa.mode=0, so a fresh install had two separate reasons to look untouched.
    g_jitterScale = live::f("taa.jitter_scale", "TAA_JITTER_SCALE", 1.0f);
    {
        float nf = live::f("taa.nearfield_m", "TAA_NEARFIELD_M", g_nearFieldM);
        if (nf < 0.0f)  nf = 0.0f;
        if (nf > 50.0f) nf = 50.0f;
        if (nf != g_nearFieldM) {
            trace("NEAR FIELD: %.2f -> %.2f m (live)", g_nearFieldM, nf);
            g_nearFieldM = nf;
        }
    }

    // Resolve duty, counted per frame at the boundary. The GATE lines are
    // sampled mid-recording and cannot distinguish "not yet" from "never";
    // this can. Every frame the resolve skips while jitter is armed shows the
    // raw offset raster on screen, so duty below ~99% IS a visible twitch and
    // the number says whether the pass-identification tail is worth more work.
    if (taaEnabled()) {
        static uint64_t dutyFrames = 0, dutyResolved = 0;
        ++dutyFrames;
        if (g_taaResolvedThisFrame) ++dutyResolved;
        if ((dutyFrames % 600) == 0)
            trace("RESOLVE DUTY: %llu of %llu frames resolved (%.1f%%)",
                  (unsigned long long)dutyResolved,
                  (unsigned long long)dutyFrames,
                  100.0 * (double)dutyResolved / (double)dutyFrames);
    }
    g_velInjectedThisFrame = false;
    g_taaResolvedThisFrame = false;
    g_sceneEndsLastFrame   = g_sceneEndsThisFrame;
    g_sceneEndsThisFrame   = 0;
    // Report a change rather than only carrying it: the count moving means the
    // frame's pass structure moved, and one frame's TAA landed on the pass that
    // used to be last. Rare and self-correcting, but it should be visible.
    // The per-frame HDR pass count survives intervals that carry zero or two
    // frames' worth of recording: an empty interval keeps the known count, a
    // clean multiple of it (two frames batched) keeps the PER-FRAME value,
    // and only a genuinely new structure replaces it.
    if (g_hdrPassesThisFrame) {
        if (g_hdrPassesLastFrame == 0 ||
            (g_hdrPassesThisFrame % g_hdrPassesLastFrame) != 0) {
            if (g_hdrPassesLastFrame)
                trace("TAA: HDR candidate passes %u -> %u per interval - pass "
                      "structure changed; the modulo boundary follows it.",
                      g_hdrPassesLastFrame, g_hdrPassesThisFrame);
            g_hdrPassesLastFrame = g_hdrPassesThisFrame;
        }
    }
    g_hdrPassesThisFrame   = 0;
    // Name the stage every MISSED frame died at - the 54% duty question.
    if (taaEnabled() && g_mv.ready && !g_taaResolvedThisFrame) {
        static uint64_t missLog = 0;
        if ((missLog++ % 60) == 0)
            trace("TAA MISS: frame ended unresolved at gateDepth=%u "
                  "(hdr=%u/%u ends=%u/%u)",
                  g_gateDepthThisFrame.load(),
                  g_hdrPassesThisFrame, g_hdrPassesLastFrame,
                  g_sceneEndsThisFrame, g_sceneEndsLastFrame);
    }
    g_gateDepthLastFrame.store(g_gateDepthThisFrame.load());
    g_gateDepthThisFrame.store(0);
    {
        const uint32_t same  = g_hdrAfterResolveSame.exchange(0);
        const uint32_t other = g_hdrAfterResolveOther.exchange(0);
        static uint64_t ovLog = 0;
        if ((same || other) && (ovLog++ % 120) == 0)
            trace("TAA OVERWRITE: %u same-image / %u other-image HDR passes "
                  "ended AFTER the resolve this frame - the engine is painting "
                  "over the output (this is the flashes mechanism)",
                  same, other);
    }
    g_prevLastDepthPassIdx = g_lastDepthPassIdx;   // predicts where to inject next frame
    g_lastDepthPassIdx     = -1;
    g_passesThisFrame      = 0;
    if ((frames % 600) == 0) {
        char line[512]; int off = 0;
        for (int i = 0; i < 16 && off < 460; ++i)
            if (g_mvPassDraws[i])
                off += snprintf(line + off, sizeof(line) - off, " [%d]=%llu", i,
                                (unsigned long long)g_mvPassDraws[i]);
        if (off) trace("MV PASS CENSUS: geometry binds per qualifying pass -%s", line);
    }

    // ---- CHOOSE THE WORLD PASS FROM THE FRAME JUST FINISHED.
    //
    // The world pass is whichever qualifying pass drew the most geometry. That
    // property is what makes it the world pass, and it holds in the cockpit, in
    // an external view, and in whatever X-Plane does next - unlike the fixed
    // ordinal it replaces, which was measured once in the cockpit and silently
    // selected the wrong pass everywhere else.
    //
    // Only switch on a clear majority. A pass that merely edges ahead on one
    // frame would make the choice oscillate, and alternating which pass owns
    // the velocity target is worse than picking the wrong one consistently.
    {
        int best = -1; uint64_t bestN = 0, total = 0;
        for (int i = 0; i < 16; ++i) {
            total += g_mvPassDrawsFrame[i];
            if (g_mvPassDrawsFrame[i] > bestN) { bestN = g_mvPassDrawsFrame[i]; best = i; }
        }
        // Print the distribution periodically whether or not it changes. The
        // silent version could not distinguish "the chosen pass is already
        // right" from "the census never ran", and those need opposite fixes.
        static uint64_t nCensus = 0;
        if ((++nCensus % 300) == 1) {
            char b[256]; b[0] = 0;
            for (int i = 0; i < 16; ++i) {
                if (!g_mvPassDrawsFrame[i]) continue;
                char one[48];
                snprintf(one, sizeof(one), "[%d]=%llu ", i,
                         (unsigned long long)g_mvPassDrawsFrame[i]);
                if (strlen(b) + strlen(one) < sizeof(b) - 1) strcat(b, one);
            }
            trace("MV PASS SHARE: view=%d | geometry binds this frame per "
                  "qualifying pass: %s| currently bound: ordinal %ld",
                  g_velSnap.viewType, b, g_mvSceneOrdinal);
        }
        if (best >= 0 && total > 0 && bestN * 2 > total && best != g_mvSceneOrdinal) {
            trace("MV SCENE PASS: now ordinal %d (%llu of %llu geometry binds "
                  "this frame). Chosen by geometry share rather than pinned - a "
                  "fixed ordinal measured in the cockpit selects the wrong pass "
                  "in other views, and the velocity target then keeps its "
                  "cleared zeros.",
                  best, (unsigned long long)bestN, (unsigned long long)total);
            g_mvSceneOrdinal = best;
        }
        // ---- A DEAD CHOICE LOSES TO ANY LIVE ONE. NO MAJORITY REQUIRED.
        //
        // The majority rule above exists to stop the choice oscillating between
        // two passes that both draw real geometry. It also meant that when the
        // frame's structure shifted - a scenery-streaming burst inserting or
        // removing passes - and the chosen ordinal stopped receiving ANY draws,
        // nothing switched away from it: the best live pass rarely clears 50%
        // of a frame in flux, so the census sat on a ghost for entire episodes
        // while the velocity target went unbound and unclear-ed, frame after
        // frame. Ordinal-points-at-nothing is not an oscillation risk; there is
        // nothing to oscillate WITH. Any pass that actually drew beats it.
        else if (best >= 0 && bestN > 0 && g_mvSceneOrdinal >= 0 &&
                 g_mvSceneOrdinal < 16 &&
                 g_mvPassDrawsFrame[g_mvSceneOrdinal] == 0) {
            trace("MV SCENE PASS: ordinal %ld received ZERO geometry this frame "
                  "- the frame's pass structure moved under it. Switching to "
                  "ordinal %d (%llu of %llu binds) immediately rather than "
                  "waiting for a majority a frame in flux may never produce.",
                  g_mvSceneOrdinal, best,
                  (unsigned long long)bestN, (unsigned long long)total);
            g_mvSceneOrdinal = best;
        }
        memset(g_mvPassDrawsFrame, 0, sizeof(g_mvPassDrawsFrame));
    }

    g_diagQualifyingPasses = g_mvQualifyThisFrame;
    g_diagBoundPasses      = g_mvBindsThisFrame;
    // ---- EPISODE BOUNDARIES, NOT PER-FRAME NOISE.
    //
    // Counted here, at the frame boundary, where the per-frame totals are
    // final - the mid-recording samples that made the GATE line ambiguous
    // cannot happen at present time. The two transitions carry the story:
    // onset distinguishes "nothing qualified" (the world rendered at a size or
    // shape the heuristics do not recognise) from "qualified but the ordinal
    // matched nothing" (the census pointing at a ghost - the case the
    // dead-choice rule above now heals), and recovery records how long the
    // field was frozen. The 2026-08-15 episodes ran ~370 frames and every
    // individual log line looked normal; only the boundary view showed them.
    if (g_mv.ready) {
        if (g_mvBindsThisFrame == 0) {
            // If the latched pass shape stops matching anything for a sustained
            // stretch, the latch is pinning a ghost - the frame's structure
            // genuinely changed. Release it and let the next bind re-latch.
            if (g_mvNoBindStreak == 30 && g_mvStickyColour != 0) {
                g_mvStickyColour = 0;
                trace("MV STICKY: released after 30 unbound frames - pass "
                      "shape changed for real, re-latching on next bind");
            }
            if (++g_mvNoBindStreak == 1)
                trace("MV FREEZE: no pass bound the velocity target this frame "
                      "(%u qualified). The target is neither written nor "
                      "cleared while this lasts - it holds STALE MOTION, and "
                      "the resolve refuses it from the second frame on.",
                      g_mvQualifyThisFrame);
        } else if (g_mvNoBindStreak) {
            trace("MV FREEZE: over after %u frame(s) - the field is being "
                  "written again.", g_mvNoBindStreak);
            g_mvNoBindStreak = 0;
        }
    }
    g_mvBindsThisFrame     = 0;
    g_mvQualifyThisFrame   = 0;
    g_pushDistinctThisFrame = 0;
    g_mvPassOrdinal        = -1;

    g_mvClearedThisFrame.store(false);
    g_depthFreshThisFrame  = false;   // must be re-proven every frame

    // Retire this frame's depth bindings and start the next frame's list empty,
    // so selection always reads one whole frame in binding order.
    {
        std::lock_guard<std::mutex> g(g_lock);
        if (!g_frameDepthList.empty()) {
            g_frameDepthListDone = g_frameDepthList;
            g_frameDepthList.clear();
        }
    }


    // ---- TAA_PRESENT_BLIT: put FSR2's result on the swapchain directly.
    //
    // The seam test proved that writing into g_sceneColor.image reaches nothing.
    // The swapchain image is the one surface that is definitionally displayed,
    // so this blits FSR2's output straight onto it, immediately before the
    // present goes down.
    //
    // It uses our OWN command pool, buffer and fence rather than borrowing
    // X-Plane's, because there is no application command buffer open at this
    // point - present is past the end of the frame's recording.
    //
    // THIS OVERWRITES THE WHOLE IMAGE, INCLUDING THE UI. That is deliberate for
    // a diagnostic: if the picture becomes the accumulated 3D scene with the
    // panel and windows gone, the path from FSR2 to the display works and the
    // only remaining question is where to composite. If the screen is unchanged,
    // even the swapchain write does nothing and the problem is not where any of
    // us have been looking.
    //
    // A blit rather than a copy: the output is R16G16B16A16_SFLOAT and the
    // swapchain is an 8-bit format, and vkCmdCopyImage requires matching size
    // classes while vkCmdBlitImage converts.
    // ONLY IF SOMETHING WAS ACTUALLY DISPATCHED INTO IT THIS FRAME.
    //
    // g_fsr2.ready means the context exists, not that this frame produced a
    // picture. Selecting Off in the menu stops the dispatch while the context
    // stays alive, so the blit went on painting the last thing FSR2 wrote - or
    // an untouched image - over every frame. That is a black screen the moment
    // the upscaler is switched off, and it looks exactly like the layer
    // breaking rather than the layer being asked to do nothing.


    // ---- THE FRAME BUDGET, BROKEN DOWN BY WHAT BLOCKED THE CPU.
    //
    // Reported once a second so the numbers can be read against the sim's own
    // CPU/GPU row. Frame time equal to the SUM of CPU and GPU time means the
    // two never overlap, and whichever line below carries the missing
    // milliseconds is the call doing the waiting:
    //   fence   - the CPU is waiting on work it just submitted (one frame in
    //             flight); the fix is to wait on a fence a frame or two old
    //   present - the display path, vsync or the compositor
    //   sem     - a timeline wait inside the frame
    // A total far below the gap means the CPU is genuinely busy and the sim is
    // CPU-bound, which is a different problem with a different fix.
    if (!g_nextQueuePresent) return VK_SUCCESS;
    const uint64_t pt0 = nowUs();
    VkResult pr = g_nextQueuePresent(queue, info);
    g_blkPresentUs.fetch_add(nowUs() - pt0, std::memory_order_relaxed);
    {
        static uint64_t lastReport = 0, lastFrames = 0;
        const uint64_t now = nowUs();
        if (!lastReport) { lastReport = now; lastFrames = frames; }
        else if (now - lastReport >= 1000000ull) {
            const double secs = (now - lastReport) / 1000000.0;
            const uint64_t df = frames - lastFrames;
            const double f  = g_blkFenceUs.exchange(0)   / 1000.0;
            const double s  = g_blkSemUs.exchange(0)     / 1000.0;
            const double p  = g_blkPresentUs.exchange(0) / 1000.0;
            const uint64_t fn = g_blkFenceN.exchange(0);
            const uint64_t fs  = g_fenceStatusN.exchange(0);
            const uint64_t fsn = g_fenceStatusNotReady.exchange(0);
            {
                const uint64_t pb = g_pipeBinds.exchange(0);
                const uint64_t pr2 = g_pipeBindsRedundant.exchange(0);
                const uint64_t db = g_dsBinds.exchange(0);
                const uint64_t dr = g_dsBindsRedundant.exchange(0);
                if (df && (pb || db))
                    trace("BIND CENSUS: per frame pipeline %llu binds (%llu "
                          "redundant, %.0f%%), descriptor-set %llu binds (%llu "
                          "redundant, %.0f%%). Redundant means the driver was "
                          "told something it already knew - the only CPU work a "
                          "layer can remove without changing behaviour.",
                          (unsigned long long)(pb / df),
                          (unsigned long long)(pr2 / df),
                          pb ? 100.0 * pr2 / pb : 0.0,
                          (unsigned long long)(db / df),
                          (unsigned long long)(dr / df),
                          db ? 100.0 * dr / db : 0.0);
            }
            if (df)
                trace("FRAME BUDGET: %.1f fps | per frame: fence %.2f ms (%llu "
                      "waits), present %.2f ms, timeline %.2f ms, fenceStatus "
                      "%llu polls (%llu NOT_READY) | blocked %.1f%% of wall "
                      "clock. Frame time = CPU + GPU means no overlap: a large "
                      "NOT_READY count is a spin loop charged to CPU time; a "
                      "large present figure is the display path.",
                      df / secs, f / df, (unsigned long long)(fn / df),
                      p / df, s / df,
                      (unsigned long long)(fs / df),
                      (unsigned long long)(fsn / df),
                      100.0 * (f + s + p) / (secs * 1000.0));
            lastReport = now; lastFrames = frames;
        }
    }

    // Re-arm the history ping-pong for the frame about to be recorded. One
    // flip per DISPLAYED frame; see the note at the flip in taa.h.
    g_taaFlipArmed = true;

    // ---- THE HISTORY IMAGE, COMPARED AGAINST ITSELF, ON THE CPU.
    //
    // Reads the strip copied out during the resolve (see taa.h) and diffs it
    // against the previous frame's. This is the one measurement of history that
    // does not pass through the resolve's output path, which is what made every
    // previous attempt unusable: viz=4 writes the visualisation INTO the
    // history image, so it reports on itself.
    //
    // Halves are 16-bit floats; comparing raw bit patterns is enough for a
    // stability metric and avoids a half-to-float conversion per texel. Two
    // reads a frame apart that differ mean history is genuinely changing;
    // reads that match while the composited image moves put the fault after
    // the accumulation rather than in it.
    if (g_taa.readPtr && taaEnabled()) {
        static uint16_t prev[2048];
        static bool havePrev = false;
        static uint64_t nDiff = 0, nSame = 0;
        const uint16_t *cur = (const uint16_t *)g_taa.readPtr;
        const size_t nHalf = 512 * 4;              // 512 texels, RGBA
        if (havePrev) {
            uint64_t differing = 0;
            double   sumAbs = 0.0;
            for (size_t i = 0; i < nHalf; ++i) {
                if (cur[i] != prev[i]) {
                    ++differing;
                    sumAbs += fabs((double)cur[i] - (double)prev[i]);
                }
            }
            if (differing) ++nDiff; else ++nSame;
            static uint64_t log = 0;
            if ((log++ % 300) == 0)
                trace("TAA HISTORY READBACK: %llu/%zu halves differ from last "
                      "frame (mean |delta| %.1f raw) | frames identical %llu, "
                      "changed %llu. History that never changes cannot be "
                      "accumulating; history that changes while the composited "
                      "image is stable puts the fault after the blend.",
                      (unsigned long long)differing, nHalf,
                      differing ? sumAbs / (double)differing : 0.0,
                      (unsigned long long)nSame, (unsigned long long)nDiff);
        }
        // ---- DID THE COPY ACTUALLY LAND? history strip vs scene strip, same
        //      frame, same pixels.
        //
        // The copy-back writes history into the scene target, so immediately
        // after it the two strips must be IDENTICAL. They are read from the two
        // halves of one buffer, captured in the same command buffer, so there
        // is no timing skew between them.
        //
        //   match  -> delivery works; the composited instability is something
        //             downstream repainting over our output
        //   differ -> the copy is not reaching the image the display reads
        //             from, which is what a double-buffered scene target does
        {
            const uint16_t *scn = cur + nHalf;
            uint64_t bad = 0;
            double   sum = 0.0;
            for (size_t i = 0; i < nHalf; ++i)
                if (cur[i] != scn[i]) { ++bad; sum += fabs((double)cur[i] - (double)scn[i]); }
            static uint64_t l2 = 0;
            if ((l2++ % 300) == 0)
                trace("TAA DELIVERY: history vs scene, SAME FRAME: %llu/%zu "
                      "halves differ (mean |delta| %.1f). The copy-back writes "
                      "history into the scene, so identical means it landed and "
                      "the instability is downstream; differing means it is not "
                      "reaching the image the display reads.",
                      (unsigned long long)bad, nHalf,
                      bad ? sum / (double)bad : 0.0);
        }
        memcpy(prev, cur, nHalf * sizeof(uint16_t));
        havePrev = true;
    }

    // ================================================================ SUITE
    //
    // One periodic dump of EVERY quantity this project has had to guess at,
    // with stable "SUITE <topic>:" prefixes so a single grep answers a
    // question without a rebuild, a relaunch, or someone sitting at the sim.
    //
    // Written because the debugging pattern that kept failing was: form a
    // theory, change a knob, ask the user what changed. Every real find came
    // instead from a number that was already being measured - the pushed
    // matrix being identity, the pass count being stable, the weight map going
    // red - and the cost was always in getting that number out. This puts them
    // all in one place, every 600 frames, whatever else is switched on.
    {
        // Per-frame accumulation, so the dump reports RATES rather than a
        // single instant. Every question this project has got wrong was got
        // wrong by reading one sample: the pass count "alternating" (it was
        // stable), the reprojection "failing 23% of the time" (a one-shot
        // warning), the duty gap "being failures" (it was cadence).
        static uint64_t sFrames = 0, sNoReproj = 0, sResolved = 0, sNoCand = 0;
        static uint64_t sGate[10] = {0,0,0,0,0,0,0,0,0,0};
        static uint64_t sBodyInvalid = 0, sIdentityReproj = 0;
        ++sFrames;
        if (g_share && !g_share->reprojValid) ++sNoReproj;
        if (!g_velSnap.bodyReprojValid)       ++sBodyInvalid;
        {
            const uint32_t gd = g_gateDepthLastFrame.load();
            sGate[gd < 10 ? gd : 9]++;
            if (gd == 0) ++sNoCand;
            if (gd == 9) ++sResolved;
        }
        // Is the pushed world matrix the identity? That single fact explains
        // total history rejection under motion, and it is cheap to test: the
        // off-diagonal translation terms are what a moving camera fills in.
        {
            const float *r = g_velSnap.reproj;
            const bool ident = fabsf(r[0]-1.0f) < 1e-6f && fabsf(r[5]-1.0f) < 1e-6f &&
                               fabsf(r[12]) < 1e-6f && fabsf(r[13]) < 1e-6f &&
                               fabsf(r[8])  < 1e-6f && fabsf(r[9])  < 1e-6f;
            if (ident) ++sIdentityReproj;
        }

        static uint64_t suiteTick = 0;
        if ((++suiteTick % 600) == 0) {
            trace("SUITE RATES over %llu frames: noReproj=%llu (%.1f%%) "
                  "identityReproj=%llu (%.1f%%) bodyInvalid=%llu (%.1f%%) "
                  "noCandidatePass=%llu (%.1f%%) resolved=%llu (%.1f%%)",
                  (unsigned long long)sFrames,
                  (unsigned long long)sNoReproj, 100.0*sNoReproj/sFrames,
                  (unsigned long long)sIdentityReproj, 100.0*sIdentityReproj/sFrames,
                  (unsigned long long)sBodyInvalid, 100.0*sBodyInvalid/sFrames,
                  (unsigned long long)sNoCand, 100.0*sNoCand/sFrames,
                  (unsigned long long)sResolved, 100.0*sResolved/sFrames);
            trace("SUITE GATE HISTOGRAM: 0=%llu 1=%llu 2=%llu 3=%llu 4=%llu "
                  "5=%llu 6=%llu 7=%llu 8=%llu 9=%llu  (0=no candidate pass, "
                  "4=format ok, 5=chosen, 6=backend accepted, 9=resolved)",
                  (unsigned long long)sGate[0], (unsigned long long)sGate[1],
                  (unsigned long long)sGate[2], (unsigned long long)sGate[3],
                  (unsigned long long)sGate[4], (unsigned long long)sGate[5],
                  (unsigned long long)sGate[6], (unsigned long long)sGate[7],
                  (unsigned long long)sGate[8], (unsigned long long)sGate[9]);
            // --- inputs: is the camera data even usable this frame
            trace("SUITE INPUT: shareValid=%d reprojValid=%d bodyValid=%d "
                  "viewType=%d frame=%llu camMoved=%.4f rotDeg=%.3f",
                  g_share ? (int)g_share->valid : -1,
                  g_share ? (int)g_share->reprojValid : -1,
                  (int)g_velSnap.bodyReprojValid, (int)g_velSnap.viewType,
                  (unsigned long long)g_frameCount, snap.camDelta, rotDeg);
            // --- the reprojection matrix as actually pushed. Identity here
            //     means zero velocity everywhere, which is the failure that
            //     survived a whole evening of downstream tuning.
            trace("SUITE MATRIX: reproj row0=(%.5f %.5f %.5f %.5f) "
                  "row1=(%.5f %.5f %.5f %.5f) wrow=(%.5f %.5f %.5f %.5f) "
                  "proj[0]=%.5f proj[5]=%.5f proj[14]=%.5f",
                  (double)g_velSnap.reproj[0], (double)g_velSnap.reproj[4],
                  (double)g_velSnap.reproj[8], (double)g_velSnap.reproj[12],
                  (double)g_velSnap.reproj[1], (double)g_velSnap.reproj[5],
                  (double)g_velSnap.reproj[9], (double)g_velSnap.reproj[13],
                  (double)g_velSnap.reproj[3], (double)g_velSnap.reproj[7],
                  (double)g_velSnap.reproj[11], (double)g_velSnap.reproj[15],
                  (double)g_velSnap.proj[0], (double)g_velSnap.proj[5],
                  (double)g_velSnap.proj[14]);
            // --- targets: a velocity target of 0x0 explains everything
            //     downstream of it, and the panel has been reporting 0 MB.
            trace("SUITE TARGET: mvReady=%d mv=%ux%u mvImage=%p mvView=%p | "
                  "taaReady=%d taa=%ux%u layers=%u",
                  g_mv.ready ? 1 : 0, g_mv.w, g_mv.h,
                  (void*)g_mv.image, (void*)g_mv.view,
                  g_taa.ready ? 1 : 0, g_taa.w, g_taa.h, g_taa.layers);
            // --- jitter: applied amplitude, not the requested one
            trace("SUITE JITVALUE: resolve saw a ZERO jitter %llu times, non-zero "
                  "%llu times, last=(%.7f %.7f) - a zero here means S is zero and "
                  "no sign of the unjitter can matter",
                  (unsigned long long)g_jitZero.load(),
                  (unsigned long long)g_jitNonZero.load(),
                  (double)g_jitLastX, (double)g_jitLastY);
            trace("SUITE JITSLOT: resolve got the raster's own jitter %llu times, "
                  "fell back to the racy globals %llu times - a fallback cancels "
                  "this frame's displacement with another frame's number",
                  (unsigned long long)g_jitSlotHit.load(),
                  (unsigned long long)g_jitSlotMiss.load());
            trace("SUITE JITTER: scale=%.3f snapJit=(%.5f %.5f) px "
                  "appliedNDC=(%.6f %.6f) armed=%d",
                  (double)g_jitterScale, (double)g_velSnap.jitterX,
                  (double)g_velSnap.jitterY, (double)g_appliedJitX,
                  (double)g_appliedJitY, g_jitterArmed ? 1 : 0);
            // --- resolve: how far the gate got, and how many ran
            trace("SUITE RESOLVE: gateDepthLast=%u hdrPassesLast=%u "
                  "mvBindsMax=%u resolvedThisFrame=%d",
                  g_gateDepthLastFrame.load(), g_hdrPassesLastFrame,
                  g_mvBindsMax, g_taaResolvedThisFrame ? 1 : 0);
            // --- injection: what fraction of geometry carries vectors
            trace("SUITE INJECT: pipeBinds=%llu redundant=%llu dsBinds=%llu "
                  "dsRedundant=%llu",
                  (unsigned long long)g_pipeBinds.load(),
                  (unsigned long long)g_pipeBindsRedundant.load(),
                  (unsigned long long)g_dsBinds.load(),
                  (unsigned long long)g_dsBindsRedundant.load());
            // --- pass census: which qualifying pass draws the world
            {
                char b[256]; int o = 0;
                for (int i = 0; i < 16 && o < 200; ++i)
                    if (g_mvPassDraws[i])
                        o += snprintf(b + o, sizeof(b) - o, " [%d]=%llu", i,
                                      (unsigned long long)g_mvPassDraws[i]);
                trace("SUITE PASSES:%s", o ? b : " none");
            }
            // --- pager / vram, since these have crashed the sim before
            trace("SUITE VRAM: zone=%s pagerImages=%llu pagerSavedMB=%.1f "
                  "skippedScaled=%llu",
                  vram::zoneName(vram::zone),
                  (unsigned long long)g_pagerImages,
                  g_pagerSaved / 1048576.0,
                  (unsigned long long)g_pagerSkippedScaled);
            // --- the resolve's own tuning, echoed so a report is
            //     self-contained: a trace without the settings that produced
            //     it has cost this project several wrong conclusions, because
            //     the file on disk had moved on by the time it was read.
            trace("SUITE TUNING: enable=%d mode=%d alpha=%.3f gain=%.2f "
                  "varclip=%.2f jitterScale=%.2f nearfield=%.2f unjitter=%d "
                  "reactive=%d velScale=%.2f velYSign=%.1f maxResolves=%d | "
                  "smulX=%.3f smulY=%.3f velMax=%.6f",
                  // ---- ASK THE ACCESSORS, NEVER RE-DERIVE THE DEFAULTS.
                  //
                  // These arguments used to repeat the live:: calls with their
                  // own default arguments, and the copies had drifted: this
                  // line claimed mode=2, alpha=0.15, varclip=1.25 and
                  // reactive=1 while the shader actually ran mode=0,
                  // alpha=0.05, varclip=8.0 and reactive=0.
                  //
                  // It only lied when the control file did NOT set a key -
                  // which is every fresh install, since taa_live.ini does not
                  // exist until something writes it. So the one configuration
                  // most likely to be reported by a user was the one
                  // configuration this line described wrongly, and varclip is
                  // the value whose 1.25-vs-8.0 confusion caused the edge
                  // shimmer that was already diagnosed and fixed once.
                  //
                  // A diagnostic that exists to prevent wrong conclusions must
                  // not hold a second opinion about the settings.
                  taaEnabled() ? 1 : 0,
                  taaMode(),
                  (double)taaAlpha(),
                  (double)taaGain(),
                  (double)taaVarClip(),
                  (double)g_jitterScale,
                  (double)g_nearFieldM,
                  taaUnjitter() ? 1 : 0,
                  taaReactive() ? 1 : 0,
                  (double)taaVelScale(),
                  (double)taaVelYSign(),
                  live::i("taa.max_resolves", "TAA_MAX_RESOLVES", 1),
                  (double)live::f("taa.smul_x", "TAA_SMUL_X", 0.5f),
                  (double)live::f("taa.smul_y", "TAA_SMUL_Y", -0.5f),
                  (double)live::f("taa.vel_max", "TAA_VEL_MAX", 1.0f));
        }
    }
    return pr;
}


// Report sampler configuration. Stage 2 applies the LOD bias here; stage 1 only
// establishes how many samplers exist and what bias they already carry, so a
// change can be attributed rather than assumed.
static uint64_t g_samplerCount = 0;
static bool     g_samplerReported = false;

static VKAPI_ATTR VkResult VKAPI_CALL Layer_CreateSampler(
    VkDevice device, const VkSamplerCreateInfo *ci,
    const VkAllocationCallbacks *alloc, VkSampler *out)
{
    if (!g_samplerReported && ci) {
        g_samplerReported = true;
        trace("SAMPLER first: minLod=%.2f maxLod=%.2f mipLodBias=%.2f aniso=%.1f mipmapMode=%d",
              ci->minLod, ci->maxLod, ci->mipLodBias,
              ci->anisotropyEnable ? ci->maxAnisotropy : 0.0f, (int)ci->mipmapMode);
    }
    ++g_samplerCount;
    return g_nextCreateSampler ? g_nextCreateSampler(device, ci, alloc, out)
                               : VK_ERROR_INITIALIZATION_FAILED;
}

static VKAPI_ATTR void VKAPI_CALL Layer_DestroyDevice(
    VkDevice device, const VkAllocationCallbacks *alloc)
{
    // The VRAM system first: every held upload submission goes down the chain
    // and every pooled block is genuinely freed, while the device and its
    // queues still exist to accept them.
    vram::shutdown();

    // ---- RELEASE THE VELOCITY TARGET.
    //
    // mvDestroy had no caller at all, so the target and its readback buffer -
    // 31.9 MB plus a mapped host allocation at 4K - were leaked every time the
    // device went away. X-Plane recreates its device on some settings changes,
    // so this is not only a shutdown concern.
    //
    // Before the dispatch table entry is erased, because destroying images
    // needs the functions it holds.
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, DeviceData>::iterator di = g_devices.find(dispatchKey(device));
        if (di != g_devices.end()) {
            // The device is dying and the app guarantees its queues are idle -
            // the ONE moment immediate destruction is safe and deferral is
            // fatal: any grave flushed after this device is gone is a call on
            // dead handles (the post-dormancy DEVICE_LOST). Drain everything
            // belonging to this device NOW.
            for (size_t i = 0; i < g_mvGraves.size();) {
                if (g_mvGraves[i].t.device == device) {
                    mvDestroyState(di->second, g_mvGraves[i].t);
                    g_mvGraves.erase(g_mvGraves.begin() + (long)i);
                } else ++i;
            }
            for (size_t i = 0; i < g_taaGraves.size();) {
                if (g_taaGraves[i].s.device == device) {
                    taaDestroyState(di->second, g_taaGraves[i].s);
                    g_taaGraves.erase(g_taaGraves.begin() + (long)i);
                } else ++i;
            }
            if (g_mv.device == device) {
                mvDestroyState(di->second, g_mv);
                g_mv = MvTarget();
            }
            if (g_taa.device == device) {
                taaDestroyState(di->second, g_taa);
                TaaState fresh;
                g_taa = fresh;
            }
        }
    }

    PFN_vkDestroyDevice next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, DeviceData>::iterator it = g_devices.find(dispatchKey(device));
        if (it != g_devices.end()) { next = it->second.destroyDevice; g_devices.erase(it); }
    }
    trace("DestroyDevice: %llu samplers seen, %u depth images tracked",
          (unsigned long long)g_samplerCount, (unsigned)g_depthCandidates.size());
    if (next) next(device, alloc);
}

// ------------------------------------------------------- chain construction

static VkLayerInstanceCreateInfo *findInstanceLink(const VkInstanceCreateInfo *ci)
{
    VkLayerInstanceCreateInfo *p = (VkLayerInstanceCreateInfo*)ci->pNext;
    while (p && !(p->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO &&
                  p->function == VK_LAYER_LINK_INFO))
        p = (VkLayerInstanceCreateInfo*)p->pNext;
    return p;
}

static VkLayerDeviceCreateInfo *findDeviceLink(const VkDeviceCreateInfo *ci)
{
    VkLayerDeviceCreateInfo *p = (VkLayerDeviceCreateInfo*)ci->pNext;
    while (p && !(p->sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO &&
                  p->function == VK_LAYER_LINK_INFO))
        p = (VkLayerDeviceCreateInfo*)p->pNext;
    return p;
}

extern "C" VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
MV_GetDeviceProcAddr(VkDevice device, const char *name);

extern "C" VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL TAA_CreateInstance(
    const VkInstanceCreateInfo *ci, const VkAllocationCallbacks *alloc, VkInstance *out)
{
    // ---- ONLY X-PLANE. THIS IS AN IMPLICIT LAYER.
    //
    // Shipped through the loader's ImplicitLayers key, this is offered to every
    // Vulkan application on the machine, not just the sim. Nothing here is
    // useful to anything else and the shader patching would be actively
    // unwelcome, so any other process gets a straight pass-through: chain to
    // the next layer and touch nothing.
    //
    // Checked by executable name rather than by a window or a dataref, because
    // this runs before either exists.
    {
        static int forUs = -1;
        if (forUs < 0) {
            char exe[MAX_PATH] = {0};
            GetModuleFileNameA(nullptr, exe, sizeof(exe) - 1);
            const char *base = strrchr(exe, 0x5C);   // 0x5C is a backslash: written
            // as a code rather than a char literal so no layer of escaping can eat it
            base = base ? base + 1 : exe;
            forUs = (_stricmp(base, "X-Plane.exe") == 0) ? 1 : 0;
            if (!forUs)
                trace("not X-Plane (%s) - passing through untouched", base);
        }
        if (!forUs) {
            VkLayerInstanceCreateInfo *lnk = findInstanceLink(ci);
            if (!lnk || !lnk->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;
            PFN_vkGetInstanceProcAddr gipa =
                lnk->u.pLayerInfo->pfnNextGetInstanceProcAddr;
            lnk->u.pLayerInfo = lnk->u.pLayerInfo->pNext;
            PFN_vkCreateInstance nextCI =
                (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
            return nextCI ? nextCI(ci, alloc, out) : VK_ERROR_INITIALIZATION_FAILED;
        }
    }

    trace("=== TAA layer CreateInstance ===");

    // Read here rather than in armLayerOnce: the instance is created long before
    // that runs, and the apiVersion decision below cannot wait.

    VkLayerInstanceCreateInfo *link = findInstanceLink(ci);
    if (!link || !link->u.pLayerInfo) {
        trace("  NO LINK FOUND - layer cannot chain");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    PFN_vkGetInstanceProcAddr nextGIPA = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    // Advance the chain for the next layer down BEFORE calling through.
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;

    PFN_vkCreateInstance nextCreate = (PFN_vkCreateInstance)nextGIPA(nullptr, "vkCreateInstance");
    if (!nextCreate) return VK_ERROR_INITIALIZATION_FAILED;

    VkInstanceCreateInfo ci2 = *ci;

    // ---- RAISE THE INSTANCE API VERSION TO 1.3 FOR STREAMLINE.
    //
    // The shim log says "vkCreatePrivateDataSlot not available (driver pre-1.3)"
    // and that is wrong about the driver - 610.74 supports 1.3 comfortably. It
    // is right about the INSTANCE: entry points that became core in 1.3 only
    // resolve if the instance was created asking for 1.3, and X-Plane asks for
    // less. Streamline needs private data slots to attach its own state to the
    // swap chain, and losing them is the last thing before it throws.
    //
    // Raising it is safe: apiVersion is a maximum the application promises not
    // to exceed, not a demand for behaviour changes. X-Plane keeps using exactly
    // the calls it always did.
    // pApplicationInfo may be NULL, and that is the case here.
    //
    // The first version guarded on it being present, so the fix never ran and
    // no log line appeared. A null pApplicationInfo means apiVersion defaults to
    // 1.0, which is the lowest possible - so it is exactly the case that most
    // needs raising, and the guard skipped it.
    VkApplicationInfo appInfo;
    memset(&appInfo, 0, sizeof(appInfo));
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    // Log it whether or not it needs raising.
    //
    // Absence of the "raising" line below was read as "X-Plane already asks for
    // 1.3", which is one of two readings - the other is that the whole block was
    // skipped. Streamline's "vkCreatePrivateDataSlot not available (driver
    // pre-1.3)" turns on exactly this number, so it gets stated rather than
    // inferred from a line that did not appear.
    trace("INSTANCE: application requests API %u.%u (appInfo %s)",
          ci->pApplicationInfo ? VK_API_VERSION_MAJOR(ci->pApplicationInfo->apiVersion) : 1u,
          ci->pApplicationInfo ? VK_API_VERSION_MINOR(ci->pApplicationInfo->apiVersion) : 0u,
          ci->pApplicationInfo ? "present" : "absent, so 1.0 by default");

    std::vector<const char*> instExts;



    VkResult r = nextCreate(&ci2, alloc, out);

    // Fall back to X-Plane's original request if our additions were refused.
    // An instance extension the loader rejects would otherwise stop the sim
    // starting, which is an absurd price for an optional upscaler.
    if (r != VK_SUCCESS && !instExts.empty()) {
        trace("INSTANCE: creation failed (%d) with the DLSS/Streamline extensions "
              "added - retrying with X-Plane's original list", (int)r);
        r = nextCreate(ci, alloc, out);
    }
    if (r != VK_SUCCESS) return r;

    InstanceData id;
    id.instance        = *out;
    id.gipa            = nextGIPA;
    id.destroyInstance = (PFN_vkDestroyInstance)nextGIPA(*out, "vkDestroyInstance");
    id.getFormatProps  = (PFN_vkGetPhysicalDeviceFormatProperties)
                             nextGIPA(*out, "vkGetPhysicalDeviceFormatProperties");
    // ---- BIND THESE ONCE, FROM THE APPLICATION'S INSTANCE ONLY.
    //
    // These are GLOBALS, and they were rebound by every vkCreateInstance in the
    // process. Streamline creates its own instances during slInit - in
    // commonEntry.cpp getNGXFeatureRequirements - and destroys them again, so
    // the last rebind before the sim starts rendering pointed into an instance
    // that no longer exists.
    //
    // A physical device function resolved from instance B, called with a
    // physical device belonging to instance A, makes the loader look the handle
    // up in the wrong instance's list. It does not find it, and it does not
    // return an error - it __fastfails:
    //
    //   vkGetPhysicalDeviceMemoryProperties: Invalid physicalDevice
    //
    // which is the 0xc0000409 the Windows event log blames on vulkan-1.dll, and
    // it kills X-Plane before the window opens. g_getPhysMemProps is named in
    // that message, which is what gave it away.
    //
    // The application's instance is the first one created and outlives every
    // probe, so binding once is both correct and the fix.
    if (!g_instanceGettersBound) {
        g_instanceGettersBound = true;
        g_getPhysMemProps  = (PFN_vkGetPhysicalDeviceMemoryProperties)
                                 nextGIPA(*out, "vkGetPhysicalDeviceMemoryProperties");
        g_getPhysProps     = (PFN_vkGetPhysicalDeviceProperties)
                                 nextGIPA(*out, "vkGetPhysicalDeviceProperties");
        g_nextEnumDeviceExt = (PFN_vkEnumerateDeviceExtensionProperties)
                                 nextGIPA(*out, "vkEnumerateDeviceExtensionProperties");
        // Bound here for the same reason as the others: from the APPLICATION's
        // instance, once. A physical-device function resolved from some other
        // instance and called with this one's device is answered by __fastfail,
        // not by an error - which this file already records having cost a day.
        g_getPhysProps2 = (PFN_vkGetPhysicalDeviceProperties2)
                              nextGIPA(*out, "vkGetPhysicalDeviceProperties2");
        g_getPhysFeat2  = (PFN_vkGetPhysicalDeviceFeatures2)
                              nextGIPA(*out, "vkGetPhysicalDeviceFeatures2");
        g_getPhysFeat   = (PFN_vkGetPhysicalDeviceFeatures)
                              nextGIPA(*out, "vkGetPhysicalDeviceFeatures");
        g_getPhysQueueFamProps = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)
                                 nextGIPA(*out, "vkGetPhysicalDeviceQueueFamilyProperties");
        g_nextMemProps2    = (PFN_vkGetPhysicalDeviceMemoryProperties2)
                                 nextGIPA(*out, "vkGetPhysicalDeviceMemoryProperties2");
        // The KHR fallback belongs INSIDE the guard. Left outside it, a probe
        // instance would rebind it whenever the core name happened to be null,
        // which is the same dangling pointer by a quieter route.
        if (!g_nextMemProps2)
            g_nextMemProps2 = (PFN_vkGetPhysicalDeviceMemoryProperties2)
                                 nextGIPA(*out, "vkGetPhysicalDeviceMemoryProperties2KHR");
    } else {
        trace("INSTANCE: %p is not the application's - leaving the global "
              "physical-device getters bound to the first instance", (void*)*out);
    }



    std::lock_guard<std::mutex> g(g_lock);
    // THE APPLICATION'S OWN INSTANCE, remembered by being first.
    //
    // Streamline creates further instances of its own during slInit and
    // destroys them again; telling them apart matters at destroy time, and
    // creation order is the only thing that distinguishes them here.
    if (g_firstInstance == VK_NULL_HANDLE) g_firstInstance = *out;
    g_instances[dispatchKey(*out)] = id;
    trace("  instance created ok");
    return r;
}

extern "C" VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL TAA_DestroyInstance(
    VkInstance inst, const VkAllocationCallbacks *alloc)
{
    // LOG EVERY INSTANCE DEATH, with the handle.
    //
    // The loader aborts with "vkGetPhysicalDeviceMemoryProperties: Invalid
    // physicalDevice" - a __fastfail, which is the 0xc0000409 in the Windows
    // event log - and immediately before it unloads the layer libraries. That
    // ordering says an instance was destroyed and something then used a
    // physical device belonging to it.
    //
    // Three instances exist in this process and only one is X-Plane's. If the
    // one handed to Streamline dies while Streamline is still using its
    // physical device, this line and the publish line together will say so.
    PFN_vkDestroyInstance next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, InstanceData>::iterator it = g_instances.find(dispatchKey(inst));
        trace("INSTANCE: destroying %p (known=%s, %zu tracked before this)",
              (void*)inst, it != g_instances.end() ? "yes" : "NO",
              g_instances.size());
        if (it != g_instances.end()) { next = it->second.destroyInstance; g_instances.erase(it); }
    }

    // ---- KEEP STREAMLINE'S PROBE INSTANCES ALIVE, DELIBERATELY.
    //
    // sl.common creates a VkInstance in getNGXFeatureRequirements, enumerates a
    // physical device from it, destroys the instance - and then keeps using
    // that physical device. The loader catches it and kills the process:
    //
    //   vkGetPhysicalDeviceMemoryProperties: Invalid physicalDevice
    //
    // a __fastfail, which is the 0xc0000409 the event log blames on
    // vulkan-1.dll, and it lands inside slSetVulkanInfo so that call never
    // returns. It is Streamline's bug and it is not reachable from here.
    //
    // What IS reachable is the destroy. A physical device stays valid as long
    // as its instance lives, so not destroying these leaves the handles
    // Streamline kept using still legal. The cost is one leaked instance per
    // probe, for the life of the process, which is a few hundred kilobytes -
    // against the sim being killed outright.
    //
    // X-Plane's own instance is destroyed normally: it is the first one
    // created, and leaking it would keep the whole device alive at shutdown.
    if (next) next(inst, alloc);
}

// What is X-Plane actually told about memory?
//
// vkGetPhysicalDeviceMemoryProperties2 is where VK_EXT_memory_budget answers
// arrive, and heapBudget is the number a pager will believe. It is NOT the size
// of the card: it is what the driver is currently willing to hand out, and it
// moves with what every other process on the machine is doing. A pager reading
// it during a period of pressure sees a small number, frees textures, and can
// see an even smaller one if something else grabbed the memory meanwhile -
// which is exactly the falling "available" figure in X-Plane's log.
//
// This only observes. Changing the answer is a decision to take after seeing
// the numbers, not before.
static VKAPI_ATTR void VKAPI_CALL TAA_GetPhysicalDeviceMemoryProperties2(
    VkPhysicalDevice phys, VkPhysicalDeviceMemoryProperties2 *props)
{
    if (g_nextMemProps2) g_nextMemProps2(phys, props);
    if (!props) return;

    // ---- BUDGET OVERRIDE. An experiment, not a feature.
    //
    // The question it answers: does X-Plane's texture pager read this at all?
    //
    // What we know is that the driver reports budget=7.02 GB of a 7.77 GB heap
    // with 2.96 GB in use - four gigabytes free - and the pager still cut
    // textures to an eighth, reporting "2.13 gb available" in its own log. So
    // either it derives its budget from this number in a way that leaves a huge
    // reserve, or it never reads this number and the figure comes from
    // somewhere else entirely.
    //
    // Raising what we report distinguishes those in one run: if X-Plane's
    // "available" moves, it is reading this and the reserve is the lever; if it
    // does not move at all, the pager is working from its own arithmetic and no
    // amount of reporting will change it. Either answer redirects the work.
    //
    // Capped at the true heap size. Reporting more memory than physically
    // exists invites an allocation failure at a moment of the driver's
    // choosing, and an out-of-memory crash mid-flight is a far worse outcome
    // than low-resolution ground textures.
    // PUBLISHED TO THE PLUGIN FIRST, unconditionally.
    //
    // This runs whatever g_vramBudgetScale is, because reporting the numbers and
    // altering them are different jobs and only the second one is optional. The
    // driver's own figures are the only authoritative VRAM data in the process,
    // and every VRAM argument in this project so far has been settled - and
    // usually overturned - by finally looking at them.
    {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT *rb = nullptr;
        for (VkBaseOutStructure *p = (VkBaseOutStructure*)props->pNext; p; p = p->pNext)
            if (p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT)
                rb = (VkPhysicalDeviceMemoryBudgetPropertiesEXT*)p;
        if (rb && g_share) {
            const VkPhysicalDeviceMemoryProperties &mp0 = props->memoryProperties;
            for (uint32_t i = 0; i < mp0.memoryHeapCount; ++i) {
                if (!(mp0.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) continue;
                g_share->vramTotalMB  = (uint32_t)(mp0.memoryHeaps[i].size / 1048576ull);
                g_share->vramBudgetMB = (uint32_t)(rb->heapBudget[i] / 1048576ull);
                g_share->vramUsageMB  = (uint32_t)(rb->heapUsage[i]  / 1048576ull);
                break;   // the first device-local heap is the one that runs out
            }
        }
    }

    // ---- THE BUDGET SHAPER. This one number is X-Plane's whole view of VRAM:
    // VMA's vmaGetHeapBudgets reads it, the memory controller averages it, and
    // the texture pager's evaluate() spends it. The shaper low-passes the
    // driver's figure, holds it monotone while the app is freeing (the
    // measured "available fell as it freed" spiral), withholds a zone-scaled
    // reserve so the engine adapts BEFORE the wall, and deflates hard after an
    // allocation failure so the engine's own pager performs the eviction the
    // memory controller cannot. The legacy TAA_VRAM_BUDGET scale still applies
    // inside it.
    {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT *b = nullptr;
        for (VkBaseOutStructure *p = (VkBaseOutStructure*)props->pNext; p; p = p->pNext)
            if (p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT)
                b = (VkPhysicalDeviceMemoryBudgetPropertiesEXT*)p;
        if (b)
            vram::shapeReport(&props->memoryProperties, b, g_vramBudgetScale);
    }

    // Report the first few, then every few hundred: a pager queries this every
    // frame and the log would be nothing else.
    ++g_memQueryCount;
    bool report = (g_memQueryCount <= 3) || (g_memQueryCount % 600 == 0);
    if (!report) return;

    const VkPhysicalDeviceMemoryProperties &mp = props->memoryProperties;

    VkPhysicalDeviceMemoryBudgetPropertiesEXT *budget = nullptr;
    for (VkBaseOutStructure *p = (VkBaseOutStructure*)props->pNext; p; p = p->pNext)
        if (p->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT)
            budget = (VkPhysicalDeviceMemoryBudgetPropertiesEXT*)p;

    for (uint32_t i = 0; i < mp.memoryHeapCount; ++i) {
        bool devLocal = (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
        if (!devLocal) continue;
        if (budget) {
            // Kept so the ledger can compare itself against the driver rather
            // than being read as VRAM when part of it is in host memory.
            g_lastHeapUsage  = budget->heapUsage[i];
            g_lastHeapBudget = budget->heapBudget[i];
            trace("VRAM[q%llu] heap%u: size=%.2f GB  budget=%.2f GB  usage=%.2f GB",
                  (unsigned long long)g_memQueryCount, i,
                  mp.memoryHeaps[i].size / 1073741824.0,
                  budget->heapBudget[i] / 1073741824.0,
                  budget->heapUsage[i]  / 1073741824.0);
        } else {
            trace("VRAM[q%llu] heap%u: size=%.2f GB  (no memory_budget in the chain - "
                  "X-Plane is reading heap SIZE, not a live budget)",
                  (unsigned long long)g_memQueryCount, i,
                  mp.memoryHeaps[i].size / 1073741824.0);
        }
    }

    // The texture census alongside it, so the two can be compared directly.
    if (g_memQueryCount <= 3 || g_memQueryCount % 600 == 0) {
        std::lock_guard<std::mutex> g(g_lock);
        if (!g_texCensus.empty()) {
            // ---- THE LEDGER, first, because it is the one that adds up.
            //
            // Printed ahead of the census deliberately. The census figure looks
            // like a total and is not one: it is what X-Plane ASKED for, before
            // the pager, and counts only sampled images. Reading it as "VRAM in
            // use" is what made a 6.28 GB card appear to be 4.20 GB of
            // textures, and left five gigabytes unexplained.
            {
                uint64_t total = 0, pageable = 0;
                for (int c = 0; c < VRAM_CAT_COUNT; ++c) {
                    total += g_vram[c].bytes;
                    if (c == VRAM_TEX) pageable += g_vram[c].bytes;
                }
                trace("VRAM LEDGER: %.2f GB resident across images and buffers "
                      "(actual driver requirements, after paging)",
                      total / 1073741824.0);
                for (int c = 0; c < VRAM_CAT_COUNT; ++c) {
                    if (!g_vram[c].count) continue;
                    trace("  %-24s %6llu objs  %9.1f MB  peak %8.1f MB   %s",
                          vramCatName(c),
                          (unsigned long long)g_vram[c].count,
                          g_vram[c].bytes / 1048576.0,
                          g_vram[c].peak  / 1048576.0,
                          vramCatWhy(c));
                }
                trace("  ---- %.1f MB of that is pageable (%.0f%%). The rest is "
                      "listed above with the reason it is not.",
                      pageable / 1048576.0,
                      total ? (100.0 * pageable / total) : 0.0);

                // The largest individual residents (SS66) - when one number in
                // the categories looks wrong, these are the names behind it.
                {
                    std::vector<std::pair<uint64_t, const VramEntry*> > top;
                    for (std::map<VkImage, VramEntry>::const_iterator ti =
                             g_vramImg.begin(); ti != g_vramImg.end(); ++ti)
                        top.push_back(std::make_pair(ti->second.bytes,
                                                     &ti->second));
                    std::sort(top.begin(), top.end());
                    int shown = 0;
                    for (size_t ti = top.size(); ti > 0 && shown < 8; --ti) {
                        const VramEntry *e = top[ti - 1].second;
                        trace("  biggest: %7.1f MB  %ux%u fmt=%u mips=%u  %s",
                              e->bytes / 1048576.0, e->w, e->h, e->fmt,
                              e->mips, vramCatName(e->cat));
                        ++shown;
                    }
                }

                // THIS TOTAL IS NOT VRAM, and the gap says by how much.
                //
                // vkGetImageMemoryRequirements reports what a resource NEEDS,
                // not which heap it was given. With overcommit armed, allocation
                // failures retry from host-visible memory - system RAM, which
                // never appears in heap0's usage. So the ledger can and does
                // exceed the driver's figure, and the difference is roughly what
                // has spilled off the card.
                //
                // Printed as a signed comparison rather than left for someone to
                // spot, because the first populated ledger read 6.25 GB against
                // a 4.23 GB heap and a table that contradicts the driver by two
                // gigabytes should say so on its own line.
                if (g_lastHeapUsage) {
                    double led = total / 1073741824.0;
                    double drv = g_lastHeapUsage / 1073741824.0;
                    trace("  ---- driver reports %.2f GB in heap0; ledger is "
                          "%+.2f GB against that. %s",
                          drv, led - drv,
                          led > drv * 1.05
                              ? "The excess is memory that is NOT on the card - "
                                "host-visible allocations, via the overcommit "
                                "path or by the application's own choice."
                              : "Close enough that everything counted is "
                                "resident; the shortfall is pipelines, "
                                "descriptor pools and driver overhead.");
                }
            }

            // ---- geometry, the largest unpaged category, by size.
            {
                uint64_t n = 0, asked = 0, got = 0;
                for (int b = 0; b < 24; ++b) {
                    n += g_geomCount[b]; asked += g_geomAsked[b]; got += g_geomGot[b];
                }
                if (n) {
                    trace("GEOMETRY BUFFERS: %llu created, %.1f MB asked for, "
                          "%.1f MB reserved - %.1f MB (%.1f%%) is alignment "
                          "padding",
                          (unsigned long long)n, asked / 1048576.0,
                          got / 1048576.0, (got - asked) / 1048576.0,
                          asked ? (100.0 * (got - asked) / asked) : 0.0);
                    for (int b = 0; b < 24; ++b) {
                        if (!g_geomCount[b]) continue;
                        double overhead = g_geomAsked[b]
                            ? (100.0 * (g_geomGot[b] - g_geomAsked[b]) / g_geomAsked[b])
                            : 0.0;
                        trace("  <=%6llu KB  %7llu bufs  %8.1f MB asked  "
                              "%8.1f MB reserved  (+%.1f%%)",
                              (unsigned long long)(1ull << b),
                              (unsigned long long)g_geomCount[b],
                              g_geomAsked[b] / 1048576.0,
                              g_geomGot[b] / 1048576.0, overhead);
                    }
                }
            }

            trace("TEXTURES REQUESTED (pre-pager, sampled only - NOT resident): "
                  "%.2f GB across %zu formats",
                  g_texBytesTotal / 1073741824.0, g_texCensus.size());
            for (std::map<int, FmtStat>::iterator it = g_texCensus.begin();
                 it != g_texCensus.end(); ++it) {
                if (it->second.bytes < 16u * 1024 * 1024) continue;   // noise
                // The numeric format matters more than the name here: the name
                // table does not cover the block-compressed formats, and "?"
                // for the two largest consumers is precisely the fact the whole
                // question turns on. Already-compressed textures mean there is
                // nothing for a transcoder to win.
                trace("  %-22s(%3d) %6llu images  %8.2f MB",
                      formatName((VkFormat)it->first), it->first,
                      (unsigned long long)it->second.count,
                      it->second.bytes / 1048576.0);
            }

            // EVERY COLOUR ATTACHMENT, with format - looking for a velocity
            // buffer X-Plane might already be rendering.
            //
            // Vulkan has no API that hands you an application's motion vectors;
            // the application has to produce them. But if X-Plane produces them
            // for its own purposes, they are sitting in a render target we can
            // see, and taking an existing correct buffer beats deriving one
            // from depth. A velocity target is recognisable: two-channel float
            // (R16G16_SFLOAT is 83, R32G32_SFLOAT is 103) at scene resolution.
            //
            // This prints the whole set once so the question is answered from
            // the frame rather than from an assumption about what X-Plane does.
            {
                static bool dumped = false;
                if (!dumped) {
                    dumped = true;
                    trace("COLOUR ATTACHMENTS (%zu): looking for a 2-channel "
                          "float target at scene resolution", g_colorImages.size());
                    std::map<int, int> seen;
                    for (std::map<VkImage, ColorTarget>::iterator ci = g_colorImages.begin();
                         ci != g_colorImages.end(); ++ci) {
                        int key = (int)ci->second.format * 1000000
                                + (int)ci->second.w * 10 + (int)ci->second.h % 10;
                        if (seen.count(key)) continue;
                        seen[key] = 1;
                        trace("  fmt=%-3d %-22s %ux%u samples=%d usage=0x%x",
                              (int)ci->second.format,
                              formatName(ci->second.format),
                              ci->second.w, ci->second.h,
                              (int)ci->second.samples, ci->second.usage);
                    }
                }
            }

            // By size, with the saving each threshold would produce.
            //
            // Dropping one mip quarters a texture, so a threshold's saving is
            // three quarters of everything ABOVE it. Printed directly rather
            // than left to be worked out from the histogram, because that
            // arithmetic is the entire decision and doing it by eye is how the
            // threshold ended up at 2048 in the first place.
            trace("TEXTURES by longest side:");
            for (int b = 0; b <= 16; ++b) {
                if (!g_sizeCount[b]) continue;
                uint64_t above = 0;
                for (int k = b + 1; k <= 16; ++k) above += g_sizeBytes[k];
                trace("  %5u px  %6llu images  %8.2f MB   "
                      "(threshold here would save %.2f MB)",
                      1u << b, (unsigned long long)g_sizeCount[b],
                      g_sizeBytes[b] / 1048576.0, above * 0.75 / 1048576.0);
            }

            // What excluding the aircraft would cost. The load bucket is an
            // UPPER BOUND on the aircraft's share, not the aircraft's share:
            // the initial scenery loads in the same window. If that bound is
            // small the exclusion is free and worth doing outright; if it is
            // most of the saving, the aircraft has to be separated properly
            // rather than by timing.
            trace("PAGER split: load %llu images %.1f MB saved (%llu at 4096) | "
                  "flight %llu images %.1f MB saved (%llu at 4096)",
                  (unsigned long long)g_pagerLoad.images,
                  g_pagerLoad.saved / 1048576.0,
                  (unsigned long long)g_pagerLoad.at4096,
                  (unsigned long long)g_pagerFlight.images,
                  g_pagerFlight.saved / 1048576.0,
                  (unsigned long long)g_pagerFlight.at4096);
            if (g_pagerSkippedScaled)
                trace("PAGER: %llu textures left alone because X-Plane had "
                      "already scaled them to a non-power-of-two size. Both "
                      "pagers cannot reduce the same texture - see the note in "
                      "pagerDropLevels. A large number here means X-Plane's "
                      "scale is doing the work and ours is not.",
                      (unsigned long long)g_pagerSkippedScaled);
        }
    }
}

// ---------------------------------------------- SPIR-V patching reconnaissance
//
// Before any of that is worth attempting, three facts decide whether it is even
// possible on THIS application, and all three are cheap to measure:
//
//   1. Does X-Plane call vkCreateShaderModule at all, or does it chain
//      VkShaderModuleCreateInfo into VkPipelineShaderStageCreateInfo under
//      VK_KHR_maintenance5? Hooking only the former would silently miss every
//      shader while looking like it worked - the worst possible failure for
//      something this invasive.
//   2. How many modules and pipelines are there? This sets the cost of
//      analysing and caching, and whether a per-launch cache is mandatory or
//      merely nice.
//   3. How big are the modules? A backward slice from Position is meant to be
//      20-60 instructions after normalisation; megabyte modules with dozens of
//      entry points would change that estimate.
static uint64_t g_shaderModules = 0;
static uint64_t g_shaderBytes   = 0;
static uint64_t g_gfxPipelines  = 0;

static uint64_t g_inlineModules = 0;   // maintenance5 path - no module object
static PFN_vkCreateShaderModule     g_nextCreateShaderModule = nullptr;
static PFN_vkCreateGraphicsPipelines g_nextCreateGfxPipelines = nullptr;

// Defined with the rest of the injection plumbing below. Declared here because
// the refusals they count happen at module creation, which comes first in the
// file - the reason tally was originally attached only to the pipeline-time
// path, where almost nothing fails, so it printed nothing while 79 shaders were
// being refused up here.
static void mvNoteInjectReason(spvinj::Result r);
static void mvLogInjectReasons();

static VKAPI_ATTR VkResult VKAPI_CALL TAA_CreateShaderModule(
    VkDevice device, const VkShaderModuleCreateInfo *ci,
    const VkAllocationCallbacks *alloc, VkShaderModule *out)
{
    // Note X-Plane's own FSR shaders as they are created, so the compute
    // pipelines built from them can be recognised later. Six variants ship in
    // Resources/shaders/bin/spv/fsr.xsa and every one declares u_fsr_data.
    // ---- NOT EVERY SHADER THAT MENTIONS u_fsr_data IS THE UPSCALER.
    //
    // fsr.xsa holds six variants and three of them turn up in a session:
    //
    //   58436 bytes  EASU - the actual spatial upscale
    //   15280 bytes  RCAS - the sharpen that follows it
    //    3876 bytes  a copy/passthrough, far too small to be either
    //
    // The small one is used elsewhere in the frame, so dropping it did not skip
    // an upscale - it removed work the rest of the frame depended on. That is
    // why the scene came back white on one half and black on the other whether
    // we sampled FSR2's output or the composite directly, and why it only went
    // wrong once the takeover engaged a few seconds after load.
    //
    // Size is the discriminator because it is a property of the shader rather
    // than of any binding, and the gap here is fifteenfold rather than marginal.
    bool isXpFsr = ci && spirvIsXpFsr(ci->pCode, ci->codeSize / 4) &&
                   ci->codeSize >= 10000;

    // ---- HAND THE DRIVER OUR UPSCALER INSTEAD OF X-PLANE'S.
    //
    // Substituted at MODULE CREATION, not at the dispatch. X-Plane then builds
    // its pipeline, allocates its descriptors and binds its resources exactly
    // as it always does - it cannot observe that the code inside is not its
    // own - and our shader runs against those bindings.
    //
    // This is why the approach changed. Dropping the dispatch required naming
    // the image the upscale writes, and that is unanswerable from outside:
    // thirteen storage images share the 3840x2160 output extent, and every way
    // a descriptor could be observed came up empty - vkUpdateDescriptorSets
    // records one set per frame, and vkCmdPushDescriptorSetKHR, the 1.4 core
    // name and the "2" form are each called zero times. Replacing the module
    // makes the question unnecessary instead of answering it.
    //
    // The interface was compared against the real module with spirv-dis before
    // this was wired: set 0 bindings 0-3, both images ARRAYED, Rgba16f output,
    // LocalSize 64 1 1 - identical on every point.
    //
    // EASU only. RCAS is a sharpening pass with its own uniform layout, and
    // substituting an upscaler for it would be replacing a shader that does a
    // different job. The two are told apart by size: 14609 words against 3820.
    VkShaderModuleCreateInfo mvCi2;
    const VkShaderModuleCreateInfo *ciUse = ci;
    if (isXpFsr && ci && fsrReplaceEnabled() && (ci->codeSize / 4) > 8000) {
        mvCi2 = *ci;
        mvCi2.pCode    = kXpFsrReplaceSpv;
        mvCi2.codeSize = kXpFsrReplaceSpvWords * 4;
        ciUse = &mvCi2;
        static bool mvSaidSub = false;
        if (!mvSaidSub) {
            mvSaidSub = true;
            trace("XP FSR: SUBSTITUTING our upscaler for X-Plane's EASU module "
                  "(%u words -> %u). X-Plane binds its own resources; only the "
                  "code that runs is ours.",
                  (unsigned)(ci->codeSize / 4), (unsigned)kXpFsrReplaceSpvWords);
        }
    }
    if (ci && ci->codeSize < 10000 && spirvIsXpFsr(ci->pCode, ci->codeSize / 4))
        trace("XP FSR: shader module (%zu bytes) mentions u_fsr_data but is too "
              "small to be EASU or RCAS - left alone, it is used elsewhere in "
              "the frame", ci->codeSize);

    if (ci) {
        ++g_shaderModules;
        g_shaderBytes += ci->codeSize;
        if (g_shaderModules <= 3 || g_shaderModules % 500 == 0)
            trace("SPIRV: module %llu, %zu bytes (%.1f MB total)",
                  (unsigned long long)g_shaderModules, ci->codeSize,
                  g_shaderBytes / 1048576.0);

        // DUMP THE SPIR-V, so the injection question can be answered by reading
        // X-Plane's actual shaders instead of assuming what they contain.
        //
        // The whole approach hinges on ONE property. A true motion vector needs
        //     prevClip = prevViewProj * model * vertex
        // and we already have prevViewProj from the plugin. So if these shaders
        // take view-projection and model as SEPARATE matrices, we substitute
        // ours, keep theirs, and get exact vectors for every static object -
        // with no per-object identity, which is the thing a layer cannot get.
        //
        // If instead they receive one pre-combined MVP per draw, there is
        // nothing to substitute: we would have to snapshot per-draw uniforms
        // and match objects frame to frame, and Vulkan gives a layer no stable
        // identity to match on. Same effort, completely different odds.
        //
        // ARMED HERE, LAZILY, not in the present path.
        //
        // Every other feature reads its environment variable at the first
        // vkQueuePresentKHR, which is fine for things that act on frames. It is
        // useless for this one: shader modules are compiled while the scenery
        // loads, long before a single frame is presented, so arming at first
        // present meant 1503 modules went past before injection was switched
        // on and the coverage count came back empty. The counters were right -
        // there was simply nothing left to count by the time they existed.
        armSpirvInject();

        // ---- motion vector injection.
        //
        // DRY RUN BY DEFAULT. The patched module is produced and counted but
        // NOT handed to the driver, because a patched vertex shader declares a
        // push constant that the pipeline layout does not yet contain and
        // writes to an attachment that the render pass does not yet have.
        // Substituting it before that plumbing exists would not degrade - it
        // would fail pipeline creation outright and take the sim with it.
        //
        // So this measures coverage first: how many vertex shaders can be
        // patched, and how many cannot. That number decides whether the rest of
        // the work is worth doing, and it costs nothing to learn.
        if (g_spirvInject) {
            std::vector<uint32_t> patched;
            uint32_t loc = 0;
            bool isFrag = false;

            // Vertex first; if it is not a vertex shader, try the fragment
            // transform on the same module. One of the two applies, or neither
            // does and it is a compute shader we have no business touching.
            // The set index is passed rather than assumed. -1 means emit nothing at
            // all, so with crash.enable off the occupancy instructions do not
            // exist in the module and cannot affect normal flight.
            const int dsSet = (crashEnabled() && crashOccupancy() && destructgpu::state().ready)
                                ? (int)destructgpu::state().setIndex : -1;
            // WHY, not just what. A single "emitted 0 modules" cannot say which
            // of the three conditions was false, and the whole discovery result
            // is meaningless if this is -1 - which is exactly how a 0% reading
            // was nearly blamed on the transform.
            {
                static bool saidDs = false;
                if (!saidDs) {
                    saidDs = true;
                    trace("DESTRUCT: shader patch decision - crash.enable=%d "
                          "crash.occupancy=%d resources_ready=%d -> set %d%s",
                          crashEnabled() ? 1 : 0, crashOccupancy() ? 1 : 0,
                          destructgpu::state().ready ? 1 : 0, dsSet,
                          dsSet < 0 ? " (NO occupancy code will be emitted)" : "");
                }
            }
            spvinj::Result r = spvinj::inject(ci->pCode, ci->codeSize, patched, &loc, dsSet);
            if (r == spvinj::INJ_NOT_VERTEX) {
                // USE THE REAL ATTACHMENT INDEX, NOT A NOMINAL 1.
                //
                // This call is only a coverage probe - the patch that gets used
                // is built at pipeline creation - but the index still decides
                // the answer. injectFragment refuses with LOCATION_TAKEN when
                // the shader already writes the attachment it was asked for, and
                // asking for 1 means every ordinary shader that writes colour
                // attachment 1 was counted as a refusal. That is where "79
                // shaders collide with our varying pair" came from: a probe
                // against the wrong attachment, reported as a coverage hole and
                // chased as one, while the location census says 16..31 are
                // entirely free and 30/31 was never contended at all.
                // alphaBlended=false: this is only a probe of whether the
                // patch SUCCEEDS. There is no pipeline here and therefore no
                // blend state to read; the variant that gets used is built at
                // pipeline creation, where the real answer is known.
                r = spvinj::injectFragment(ci->pCode, ci->codeSize, patched,
                                           spvinj::mvAttachmentIndex(), false);
                isFrag = true;
            }

            switch (r) {
            case spvinj::INJ_OK:
                ++g_injOk;
                if (isFrag) ++g_injFrag;
                g_patchedCode = patched;
                g_patchedWasFrag = isFrag;
                if (g_injOk <= 6)
                    trace("SPIRV INJECT: module %llu patched (%s) - %zu -> %zu "
                          "words, %s %u",
                          (unsigned long long)g_shaderModules,
                          isFrag ? "FRAGMENT" : "vertex",
                          ci->codeSize / 4, patched.size(),
                          isFrag ? "writes attachment" : "prevClip at Location",
                          loc);
                break;
            case spvinj::INJ_NOT_VERTEX: ++g_injNotVertex; break;
            default:
                ++g_injFailed;
                // Tally the REASON, not just the count. The refusals live on
                // this path, not on the pipeline-time one that was instrumented
                // first - which is why that tally never printed anything while
                // 79 shaders were being refused here.
                mvNoteInjectReason(r);
                if (g_injFailed <= 5 || (g_injFailed % 25) == 0) mvLogInjectReasons();
                trace("SPIRV INJECT: module %llu FAILED (reason %d, %zu bytes) - "
                      "this shader would leave holes in the velocity buffer",
                      (unsigned long long)g_shaderModules, (int)r, ci->codeSize);
                break;
            }
            // Reported against the MODULE count, not against the patch count.
            //
            // The first version fired when (patched + failed) hit a multiple of
            // 200, which meant that if there were fewer vertex shaders than
            // that it never printed at all - and it did not, so a run with zero
            // failures was indistinguishable from a run where nothing ever
            // executed. Keying it to a counter that is guaranteed to advance
            // makes the number appear whether it is good news or bad.
            if ((g_shaderModules % 250) == 0)
                trace("SPIRV INJECT: %llu patched (%llu frag), %llu failed, %llu other (of %llu modules)",
                      (unsigned long long)g_injOk,
                      (unsigned long long)g_injFrag,
                      (unsigned long long)g_injFailed,
                      (unsigned long long)g_injNotVertex,
                      (unsigned long long)g_shaderModules);
        }

        // Cheap to answer and expensive to guess at, so: TAA_DUMP_SHADERS=1
        // writes the first modules to disk for spirv-dis.
        if (getenv("TAA_DUMP_SHADERS") && g_shaderModules <= 40) {
            const char *tmp = getenv("TEMP");
            char dir[512], path[640];
            snprintf(dir, sizeof(dir), "%s\\taa_shaders", tmp ? tmp : ".");
            CreateDirectoryA(dir, nullptr);
            snprintf(path, sizeof(path), "%s\\mod_%03llu.spv", dir,
                     (unsigned long long)g_shaderModules);
            FILE *f = fopen(path, "wb");
            if (f) {
                fwrite(ci->pCode, 1, ci->codeSize, f);
                fclose(f);
                if (g_shaderModules <= 3)
                    trace("SPIRV: dumped %s (%zu bytes)", path, ci->codeSize);
            }
        }
    }
    // Keep the original words so the fragment patch can be built later, when
    // the attachment index is known. Only under LIVE - in dry run nothing will
    // ever ask for them, and this is several megabytes.
    if (g_spirvLive) {
        VkResult rr = g_nextCreateShaderModule
            ? g_nextCreateShaderModule(device, ciUse, alloc, out)
            : VK_ERROR_INITIALIZATION_FAILED;
        if (rr == VK_SUCCESS && out) {
            std::lock_guard<std::mutex> g(g_lock);
            g_moduleCode[*out].assign(ci->pCode, ci->pCode + ci->codeSize / 4);
            if (isXpFsr) g_xpFsrModules.insert(*out);
            // Remember WHICH module got our code, so the probe can wait
            // for the dispatch that actually writes the sentinel.
            if (isXpFsr && ciUse != ci) g_xpFsrOurModule = *out;
            // ---- DUMP X-PLANE'S FSR SPIR-V.
            //
            // A replacement shader has to declare the SAME descriptor layout as
            // the module it stands in for - same sets, same bindings, same
            // types - or X-Plane's own bindings land in the wrong places. That
            // layout is not documented anywhere; it is in this SPIR-V, so it
            // gets written out and read rather than guessed at.
            //
            // This is also the answer to the output-image problem: replacing
            // the shader means X-Plane binds its own resources exactly as it
            // always does, so there is no longer any need to work out WHICH of
            // thirteen same-sized storage images the upscale writes.
            if (isXpFsr) {
                static int nDump = 0;
                if (nDump < 4) {
                    char path[512];
                    // Forward slash on purpose: Windows accepts it in fopen,
                    // and a backslash here has been eaten by the shell twice
                    // already, producing "\x used with no following hex digits".
                    snprintf(path, sizeof(path), "%s/xpfsr_%d_%u.spv",
                             getenv("TEMP") ? getenv("TEMP") : ".",
                             nDump, (unsigned)(ci->codeSize / 4));
                    FILE *f = fopen(path, "wb");
                    if (f) {
                        fwrite(ci->pCode, 1, ci->codeSize, f);
                        fclose(f);
                        trace("XP FSR: dumped module to %s (%u words)",
                              path, (unsigned)(ci->codeSize / 4));
                    }
                    ++nDump;
                }
            }
        }
        if (isXpFsr)
            trace("XP FSR: shader module %p is one of X-Plane's FSR variants "
                  "(u_fsr_data present, %zu bytes)", (void*)(out ? *out : 0),
                  ci->codeSize);
        g_patchedCode.clear();
        return rr;
    }

    // The FRAGMENT patch happens at pipeline creation, not here - see
    // mvPatchFragment. The vertex patch could happen here, but keeping both in
    // one place makes the pairing obvious: a pipeline gets a patched vertex
    // shader only when its fragment shader could also be patched.
    g_patchedCode.clear();
    {
        VkResult rr = g_nextCreateShaderModule
            ? g_nextCreateShaderModule(device, ciUse, alloc, out)
            : VK_ERROR_INITIALIZATION_FAILED;
        if (rr == VK_SUCCESS && out && isXpFsr) {
            std::lock_guard<std::mutex> g(g_lock);
            g_xpFsrModules.insert(*out);
            // ---- TAG IT HERE TOO.
            //
            // There are two module-creation paths and only the other one
            // recorded which module got our code, so g_xpFsrOurPipelines stayed
            // empty and the probe never fired. Substitution is decided above,
            // before the branch; the tagging has to follow it on BOTH sides.
            if (ciUse != ci) g_xpFsrOurModule = *out;
            trace("XP FSR: shader module %p is one of X-Plane's FSR variants "
                  "(u_fsr_data present, %zu bytes)%s", (void*)*out, ci->codeSize,
                  ciUse != ci ? " - THIS ONE CARRIES OUR CODE" : "");
        }
        return rr;
    }
}

// Compute pipelines built from those modules ARE the upscaler. Hooked purely to
// learn which VkPipeline handles to watch for at bind time - the pipeline is
// created exactly as the application asked.
static VKAPI_ATTR VkResult VKAPI_CALL TAA_CreateComputePipelines(
    VkDevice device, VkPipelineCache cache, uint32_t count,
    const VkComputePipelineCreateInfo *ci, const VkAllocationCallbacks *alloc,
    VkPipeline *out)
{
    PFN_vkCreateComputePipelines next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, DeviceData>::iterator it = g_devices.find(dispatchKey(device));
        if (it != g_devices.end()) next = it->second.createComputePipelines;
    }
    if (!next) return VK_ERROR_INITIALIZATION_FAILED;

    LARGE_INTEGER cpc0, cpc1, cpcf;
    QueryPerformanceCounter(&cpc0);
    VkResult r = next(device, cache, count, ci, alloc, out);
    QueryPerformanceCounter(&cpc1);
    QueryPerformanceFrequency(&cpcf);
    if (cpcf.QuadPart > 0)
        vram::notePipelines(count,
            (uint64_t)((cpc1.QuadPart - cpc0.QuadPart) * 1000000ll /
                       cpcf.QuadPart));
    if (r == VK_SUCCESS && ci && out) {
        std::lock_guard<std::mutex> g(g_lock);
        for (uint32_t i = 0; i < count; ++i) {
            if (!g_xpFsrModules.count(ci[i].stage.module)) continue;
            g_xpFsrPipelines.insert(out[i]);
            if (g_xpFsrOurModule != VK_NULL_HANDLE &&
                ci[i].stage.module == g_xpFsrOurModule)
                g_xpFsrOurPipelines.insert(out[i]);
            // ---- IS THE PIPELINE BUILT FROM THE MODULE WE SUBSTITUTED?
            //
            // The screen stayed normal with the shader painting every pixel
            // magenta, so our code is not executing even though substitution
            // logs. The tagging above never matched either. Say the two handles
            // out loud rather than infer why.
            trace("XP FSR: compute pipeline from module %p; our substituted "
                  "module is %p -> %s",
                  (void*)ci[i].stage.module, (void*)g_xpFsrOurModule,
                  (g_xpFsrOurModule != VK_NULL_HANDLE &&
                   ci[i].stage.module == g_xpFsrOurModule)
                      ? "MATCH - this pipeline runs our code"
                      : "DIFFERENT MODULE - our code is not in this pipeline");
            trace("XP FSR: compute pipeline %p is X-Plane's upscaler - its "
                  "dispatches will be dropped and replaced by FSR2's result",
                  (void*)out[i]);
        }
    }
    return r;
}

// THE PUSH CONSTANT THE INJECTED SHADERS DECLARE HAS TO EXIST IN THE LAYOUT.
//
// A patched vertex shader reads a mat4 from push constant offset 0. If the
// pipeline layout does not declare a range covering that, pipeline creation
// fails outright - so this hook adds one.
//
// WHY OFFSET 0 IS SAFE HERE, AND WHY IT IS CHECKED ANYWAY. Vulkan forbids two
// push constant ranges in one layout from naming the same stage
// (VUID-VkPipelineLayoutCreateInfo-pPushConstantRanges-00292), but ranges for
// DIFFERENT stages may overlap freely. X-Plane's push constants live in its
// fragment shaders - the fifteen vertex shaders use none, which was measured
// from the dumps rather than assumed - so a VERTEX range at offset 0 collides
// with nothing.
//
// "Measured rather than assumed" is doing real work in that sentence, so it is
// verified per layout instead of trusted: if a layout already has a range
// naming the vertex stage, we cannot add a second one and cannot move the
// shader's offset after the fact, so that layout is left alone and recorded.
// Pipelines built from it keep their original shaders - a hole in the velocity
// buffer rather than a dead sim.
static std::map<VkPipelineLayout, bool> g_layoutHasOurPC;



static VKAPI_ATTR VkResult VKAPI_CALL TAA_CreatePipelineLayout(
    VkDevice device, const VkPipelineLayoutCreateInfo *ci,
    const VkAllocationCallbacks *alloc, VkPipelineLayout *out)
{
    PFN_vkCreatePipelineLayout next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, DeviceData>::iterator it = g_devices.find(dispatchKey(device));
        if (it != g_devices.end()) next = it->second.createPipelineLayout;
    }
    if (!next) return VK_ERROR_INITIALIZATION_FAILED;

    armSpirvInject();
    if (!g_spirvInject || !ci)
        return next(device, ci, alloc, out);

    bool vertexRangeExists = false;
    for (uint32_t i = 0; i < ci->pushConstantRangeCount; ++i)
        if (ci->pPushConstantRanges[i].stageFlags & VK_SHADER_STAGE_VERTEX_BIT)
            vertexRangeExists = true;

    if (vertexRangeExists) {
        VkResult r = next(device, ci, alloc, out);
        if (r == VK_SUCCESS) {
            std::lock_guard<std::mutex> g(g_lock);
            g_layoutHasOurPC[*out] = false;
            if (++g_layoutSkipped <= 5)
                trace("SPIRV INJECT: layout already has a VERTEX push constant "
                      "range - left unpatched (%llu so far). Draws using it will "
                      "carry no velocity.",
                      (unsigned long long)g_layoutSkipped);
        }
        return r;
    }

    std::vector<VkPushConstantRange> ranges(
        ci->pPushConstantRanges,
        ci->pPushConstantRanges + ci->pushConstantRangeCount);

    // ---- WHAT X-PLANE ALREADY DECLARES, AND WHETHER WE COLLIDE WITH IT.
    //
    // The matrix loads as ZERO in the vertex shader - m33 reads 0.000, so
    // prevClip is 0 and every motion vector produced has been currNDC * 0.5, a
    // difference against nothing. The offset is not the problem: 176 is chosen
    // before the first shader is patched and both the declaration and the push
    // use it.
    //
    // What has never been looked at is the ranges X-Plane itself declares. If
    // one of them already covers our bytes FOR THE VERTEX STAGE, the layout is
    // invalid by overlap; if X-Plane pushes a range spanning them, its own
    // writes land on our matrix. Either way the shader reads something nobody
    // intended, and zero is the commonest something.
    // ---- ANY STAGE'S RANGE CAN CLOBBER OURS, NOT JUST A VERTEX ONE.
    //
    // Push constant memory is ONE block shared by every stage. The check above
    // only skips layouts that declare a VERTEX range, so a FRAGMENT range
    // covering our bytes is merely logged and then patched anyway - and when
    // X-Plane pushes to it, its writes land on our matrix and the vertex shader
    // reads whatever X-Plane put there.
    //
    // That is what the raw clip values show. currClip.y is verified correct and
    // M[9] is ~0, so yc and d are right, yet the shader's prevY moves OPPOSITE
    // to the prediction - d(prevY)/d(yc) = -0.79 where M[5] = 1.0000 was pushed.
    // Every input checks out, so the matrix those draws actually saw was not
    // ours. It is not a flip either: substituting -yc predicts +7.013 against a
    // measured -6.957.
    //
    // A wrong vector is worse than none, so overlapping layouts go unpatched.
    {
        bool anyOverlap = false;
        for (uint32_t i = 0; i < ci->pushConstantRangeCount; ++i) {
            const VkPushConstantRange &e = ci->pPushConstantRanges[i];
            if ((e.offset < spvinj::pushConstantOffset() + spvinj::kPushConstantBytes) &&
                (spvinj::pushConstantOffset() < e.offset + e.size))
                anyOverlap = true;
        }
        if (anyOverlap) {
            VkResult r = next(device, ci, alloc, out);
            if (r == VK_SUCCESS) {
                std::lock_guard<std::mutex> g(g_lock);
                g_layoutHasOurPC[*out] = false;
                if (++g_layoutOverlap <= 5)
                    trace("SPIRV INJECT: a declared push range OVERLAPS ours at "
                          "%u..%u in another stage - left unpatched (%llu so far). "
                          "X-Plane's pushes would land on our matrix.",
                          spvinj::pushConstantOffset(),
                          spvinj::pushConstantOffset() + spvinj::kPushConstantBytes,
                          (unsigned long long)g_layoutOverlap);
            }
            return r;
        }
    }

    {
        static bool said = false;
        if (!said) {
            said = true;
            for (uint32_t i = 0; i < ci->pushConstantRangeCount; ++i) {
                const VkPushConstantRange &e = ci->pPushConstantRanges[i];
                const bool overlaps =
                    (e.offset < spvinj::pushConstantOffset() + spvinj::kPushConstantBytes) &&
                    (spvinj::pushConstantOffset() < e.offset + e.size);
                trace("SPIRV INJECT: X-Plane push range %u/%u - stages 0x%x, "
                      "offset %u, size %u (ours is vertex, %u..%u)%s",
                      i + 1, ci->pushConstantRangeCount, e.stageFlags,
                      e.offset, e.size, spvinj::pushConstantOffset(),
                      spvinj::pushConstantOffset() + spvinj::kPushConstantBytes,
                      overlaps ? "  *** OVERLAPS OURS ***" : "");
            }
            if (ci->pushConstantRangeCount == 0)
                trace("SPIRV INJECT: X-Plane declares NO push constant ranges "
                      "on this layout - ours is the only one");
        }
    }

    VkPushConstantRange ours;
    ours.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    ours.offset     = spvinj::pushConstantOffset();
    ours.size       = spvinj::kPushConstantBytes;   // one mat4
    ranges.push_back(ours);

    VkPipelineLayoutCreateInfo ci2 = *ci;
    ci2.pushConstantRangeCount = (uint32_t)ranges.size();
    ci2.pPushConstantRanges    = ranges.data();

    // ---- APPEND THE DESTRUCTION SET.
    //
    // At the layout's OWN setLayoutCount, so sets 0..N-1 keep the exact
    // layouts X-Plane declared. That matters for more than tidiness: Vulkan
    // pipeline layout compatibility is defined per set index, and leaving the
    // lower indices untouched is what stops X-Plane's own bound sets being
    // disturbed by our layout existing.
    //
    // The consequence is that our index VARIES per layout, so it is recorded
    // rather than assumed. A hardcoded index would read whichever buffer
    // X-Plane happened to bind there.
    std::vector<VkDescriptorSetLayout> sets(
        ci->pSetLayouts, ci->pSetLayouts + ci->setLayoutCount);
    uint32_t ourSet = UINT32_MAX;
    if (destructgpu::state().ready) {
        // ---- A FIXED INDEX, PADDED UP TO.
        //
        // This used to append at the layout's own setLayoutCount, which put our
        // set at index 1 on one layout and 4 on another. That works while only
        // C++ touches it, because the bind looks the index up - and it becomes
        // impossible the moment a SHADER references the set, because
        // OpDecorate DescriptorSet is a literal baked into a module that is
        // patched once and used with many layouts.
        //
        // So every extended layout puts our set at the SAME index, padding the
        // gap with an empty layout that declares no bindings. Indices below
        // X-Plane's own count are untouched, which is what keeps its bound sets
        // undisturbed; the padding only occupies indices nobody was using.
        const uint32_t want = destructgpu::state().setIndex;
        if (ci->setLayoutCount <= want && want + 1u <= g_maxBoundSets) {
            ourSet = want;
            while (sets.size() < want)
                sets.push_back(destructgpu::state().emptyLayout);
            sets.push_back(destructgpu::state().setLayout);
            ci2.setLayoutCount = (uint32_t)sets.size();
            ci2.pSetLayouts    = sets.data();
            if (++destructgpu::layoutsExtended() % 500 == 1)
                trace("DESTRUCT: %llu layout(s) carry the fragment set at the "
                      "fixed index %u (%u of X-Plane's own, %u empty pad, %u max)",
                      (unsigned long long)destructgpu::layoutsExtended(),
                      ourSet, ci->setLayoutCount,
                      want - ci->setLayoutCount, g_maxBoundSets);
        } else if (ci->setLayoutCount > want) {
            // X-Plane already uses the index we picked. Counted, never silent:
            // overwriting one of its sets would be corruption, so this layout
            // simply does not carry ours and its draws cannot displace.
            if (++destructgpu::layoutsTooMany() % 100 == 1)
                trace("DESTRUCT: %llu layout(s) already declare %u sets, past our "
                      "fixed index %u - not extended, so their draws cannot "
                      "displace. Raise the index if this is common.",
                      (unsigned long long)destructgpu::layoutsTooMany(),
                      ci->setLayoutCount, want);
        } else {
            // Counted, never silent. A layout at the device limit cannot take
            // our set, and every draw using it will be unable to displace -
            // which downstream looks exactly like a shader that failed.
            if (++destructgpu::layoutsTooMany() % 100 == 1)
                trace("DESTRUCT: %llu layout(s) already at the %u-set device "
                      "limit - no fragment set, so their draws cannot displace",
                      (unsigned long long)destructgpu::layoutsTooMany(),
                      g_maxBoundSets);
        }
    }

    VkResult r = next(device, &ci2, alloc, out);
    if (r == VK_SUCCESS && ourSet != UINT32_MAX) {
        std::lock_guard<std::mutex> g(g_lock);
        g_layoutOurSet[*out] = ourSet;
    }
    if (r != VK_SUCCESS) {
        // Fall back rather than fail the application's call. A layout we cannot
        // extend is a velocity hole; a layout we fail to create is a crash.
        trace("SPIRV INJECT: layout with our push constant range was REJECTED "
              "(%d) - retrying unmodified", (int)r);
        r = next(device, ci, alloc, out);
        if (r == VK_SUCCESS) {
            std::lock_guard<std::mutex> g(g_lock);
            g_layoutHasOurPC[*out] = false;
        }
        return r;
    }

    std::lock_guard<std::mutex> g(g_lock);
    g_layoutHasOurPC[*out] = true;
    if (++g_layoutPatched <= 3 || (g_layoutPatched % 250) == 0)
        trace("SPIRV INJECT: pipeline layout extended with a %u-byte VERTEX "
              "push constant range (%llu patched, %llu skipped)",
              spvinj::kPushConstantBytes,
              (unsigned long long)g_layoutPatched,
              (unsigned long long)g_layoutSkipped);
    return r;
}

// PUSHING uReproj: which pipelines can receive it, and when.
//
// A push constant is not stored in the pipeline - it is command buffer state,
// bound to a layout, and it is invalidated whenever a pipeline with an
// incompatible layout is bound. So it cannot be written once per frame; it has
// to be written after each bind of a pipeline whose layout carries our range.
//
// Recording pipeline -> layout at creation is what makes that possible.
// vkCmdBindPipeline names only the pipeline, and there is no Vulkan query that
// takes a VkPipeline and returns its layout - the association exists only in
// the create info, which is gone by the time the draw happens.
static std::map<VkPipeline, VkPipelineLayout> g_pipelineLayoutOf;

// ---- WHAT TO PUSH, PER COMMAND BUFFER, PUSHED AGAIN AT DRAW TIME.
//
// The matrix loads as ZERO in the vertex shader while everything upstream
// checks out: spirv-val passes, the block is declared at offset 176 with the
// entry point listing it, the layout carries our range, X-Plane declares none
// of its own, no layout is skipped, and 2.2 million pushes happen.
//
// The one mechanism left is Vulkan's own: push constant values become UNDEFINED
// when a pipeline layout that is not push-constant-compatible is bound. We push
// at vkCmdBindPipeline, and X-Plane then binds descriptor sets and other state
// before the draw. Any of those with a different layout discards our write, and
// discards it silently - no error, no validation message, just zeros at the
// shader.
//
// Pushing again immediately before each draw closes that window: nothing can be
// bound between the push and the draw that consumes it.
struct PendingPush { VkCommandBuffer cb; VkPipelineLayout layout;
                     float block[36]; bool valid;
                     // UINT32_MAX when this pipeline's layout does not
                     // carry the destruction set - which is the common
                     // case, and must bind NOTHING rather than bind at 0.
                     uint32_t destructSet; };

// THREAD-LOCAL, NOT A MAP UNDER A MUTEX.
//
// The first version of this kept a std::map keyed by command buffer and took
// the global lock on every draw. X-Plane records on several threads, so every
// draw on every thread serialised on one mutex and the sim ran at 9 fps.
//
// A recording thread has exactly one command buffer open and one pipeline last
// bound, so the state is naturally per-thread. No sharing, no lock, no map
// lookup - a pointer compare and a memcpy.
// ---- ONE SLOT PER THREAD WAS THE BUG.
//
// The single-slot version assumed a recording thread has exactly one command
// buffer open. X-Plane interleaves several, so any draw into a command buffer
// other than the last one bound on that thread failed the
// `g_tlPush.cb != cb` guard and skipped its re-push - leaving whatever was last
// written in that buffer, which is X-Plane's own data.
//
// Measured: the shader loads M[12] = 110.94 / 161.00 / 179.88 on the bad pixels
// while the layer pushes -0.024233. A few distinct values, per draw, not noise.
// prevNDC.x - u is about M[12]/d, so 179/180 is 0.6 NDC or 1150 px - exactly
// the flows exact mode reports on pixels whose real depth cannot produce them.
//
// A mutex-guarded map was tried before and ran the sim at 9 fps. This keeps the
// state thread-local so there is still no lock, but holds a few slots so
// interleaved command buffers each keep their own matrix. Linear scan over 8
// entries is cheaper than the map lookup and allocates nothing.
struct PushSlots {
    static const int kN = 8;
    PendingPush slot[kN];
    int next;
    PushSlots() : next(0) {
        for (int i = 0; i < kN; ++i) {
            slot[i].cb = VK_NULL_HANDLE;
            slot[i].layout = VK_NULL_HANDLE;
            slot[i].valid = false;
        }
    }
    // ---- OCCUPANCY IS NOT VALIDITY. THEY WERE THE SAME FLAG AND STOPPED BEING.
    //
    // This matched on `valid`, which means "a matrix has been pushed into this
    // slot". That was the same thing as "this slot belongs to this command
    // buffer" for as long as a push was the only thing that ever created one.
    //
    // It stopped being the same thing when the pipeline bind began creating
    // slots to carry destructSet - and those start with valid = false on
    // purpose, so the push path cannot fire on a block nobody wrote. So find()
    // skipped exactly the slots that carried the descriptor set index, the
    // draw took the "no slot" early return, and set 7 was never bound.
    //
    // The shader statically uses set 7, so a draw with it unbound is
    // VUID-vkCmdDrawIndexed-None-08600 and then a lost device. Validation
    // named it in one run: "The set (7) is out of bounds for the number of
    // sets bound (3)".
    //
    // Occupancy is the command buffer matching. Whether the MATRIX is usable is
    // a separate question, asked separately by every caller that reads it.
    PendingPush *find(VkCommandBuffer cb) {
        if (cb == VK_NULL_HANDLE) return nullptr;
        for (int i = 0; i < kN; ++i)
            if (slot[i].cb == cb) return &slot[i];
        return nullptr;
    }
    PendingPush *obtain(VkCommandBuffer cb) {
        if (PendingPush *p = find(cb)) return p;
        PendingPush *p = &slot[next];
        next = (next + 1) % kN;
        // ---- A RECYCLED SLOT CARRIES THE PREVIOUS BUFFER'S STATE.
        //
        // It was left uninitialised while the only field that mattered was the
        // matrix, which every caller wrote before use. destructSet is not like
        // that: the push path never sets it, so a slot recycled from a command
        // buffer whose pipeline DID carry the set arrived at one whose pipeline
        // does not, still saying 7.
        //
        // The result is vkCmdBindDescriptorSets(firstSet = 7) against a layout
        // that declares one set. X-Plane did not survive the first frame.
        //
        // Cleared here, once, rather than at each call site - the previous
        // version guarded the site that was written last week and not the one
        // written a month ago, which is the failure mode this avoids.
        p->cb          = cb;
        p->layout      = VK_NULL_HANDLE;
        p->valid       = false;
        p->destructSet = UINT32_MAX;
        memset(p->block, 0, sizeof(p->block));
        return p;
    }
};
static thread_local PushSlots g_tlPushSlots;
static thread_local PendingPush g_tlPush = { VK_NULL_HANDLE, VK_NULL_HANDLE, {0}, false, UINT32_MAX };

static bool mvPendingJitter(VkCommandBuffer cb, float *jx, float *jy)
{
    PendingPush *pp = g_tlPushSlots.find(cb);
    // find() now returns a slot the PIPELINE BIND may have created, whose block
    // nobody has written. It used to filter those out itself, and the jitter
    // read would otherwise be two floats of zeroed memory presented as a real
    // sub-pixel offset.
    if (!pp || !pp->valid) return false;
    *jx = pp->block[16];
    *jy = pp->block[17];
    return true;
}

// Does this pipeline draw geometry from vertex buffers, and how many of each
// kind exist. Filled at pipeline creation; read at bind time to decide whether
// this draw is jittered. See the note where it is written.
static std::map<VkPipeline, bool> g_pipelineIsGeometry;
static uint64_t g_pipeGeometry = 0, g_pipeFullscreen = 0;
// Pipelines the driver refused with the injection and which fell back to
// X-Plane's original. Every one is a hole in the velocity field, and 14,835 of
// them is what the Location 31 fault looked like from the outside - so this is
// the single number that says whether that class of failure has returned.
static uint64_t g_pipeRejected = 0;

// ---- DRAW-WEIGHTED MOTION VECTOR COVERAGE.
//
// "79 of 1250 modules refused" was the only coverage figure this project had,
// and it was both wrong (a probe artefact) and the wrong unit. What decides
// whether the image shimmers is how much of the SCREEN is drawn by pipelines
// that write no motion vectors, and a module census cannot see that. A single
// unpatched pipeline drawing terrain covers more pixels than fifty patched ones
// drawing cockpit switches.
//
// Counted per bind inside a scene pass, which is the closest cheap proxy for
// draw calls and needs no vkCmdDraw interception.
static std::map<VkPipeline, bool> g_pipelineMvPatched;
static uint64_t g_bindScenePatched = 0, g_bindSceneUnpatched = 0;
static uint64_t g_bindCockpitPatched = 0, g_bindCockpitUnpatched = 0;
static uint64_t g_pushCount = 0;

// Original SPIR-V, kept so fragment shaders can be patched LATER.
//
// A fragment shader's velocity output Location must equal the render pass's
// colour attachment count, and that is not known at vkCreateShaderModule - only
// at vkCreateGraphicsPipelines. So the words are stored here and the patch
// happens when the answer exists.
//
// Costs a few megabytes for X-Plane's ~1500 modules, which is the price of not
// guessing an index. Both earlier guesses failed: deriving it from the shader's
// own outputs wrote velocity over a live G-buffer target, and pinning it to a
// fixed index needed every pass padded to eight attachments, which the driver
// rejected outright.

// The VERTEX half. Without it the fragment shaders read varyings nobody writes.
//
// This was briefly lost: moving the fragment patch to pipeline creation meant
// vkCreateShaderModule handed back every module unpatched, vertex included. The
// fragment shaders still read currClip and prevClip at their fixed Locations,
// nothing wrote them, undefined inputs read as zero, and (0 - 0) * 0.5 gave a
// velocity field that was uniformly zero while the camera moved six metres.
//
// Keyed on the module alone, unlike the fragment variants: the vertex locations
// are fixed device-wide, so one patched copy serves every pipeline.
static std::map<VkShaderModule, VkShaderModule> g_vertVariant;
// The patched handles themselves, so a pipeline stage can be recognised as
// carrying injected code whether it names the original (about to be
// substituted) or the substituted module. Used only by the layout-mismatch
// check in TAA_CreateGraphicsPipelines.
static std::set<VkShaderModule> g_patchedVertModules;
static uint64_t g_vertVariants = 0, g_vertPatchFail = 0;

// Choose the varying pair from what X-Plane actually uses, once.
//
// chooseLocations() picked the top of the device limit - 31/30 on a card
// reporting 128 output components. That is within the limit and says nothing
// about whether anything already lives there, and 79 of 1250 modules did:
// refused with LOCATION_TAKEN, patched into nothing, writing no motion vectors
// at all. Every pixel those draws produced had FSR2 reject history and crawl.
//
// Done here rather than at device creation because this is the first moment a
// representative set of modules exists - vertex variants are built lazily at
// pipeline creation, by which point X-Plane has handed us most of its shaders.
// Set the moment the first module is patched. After that the varying pair is
// frozen, because it is a CONTRACT BETWEEN TWO SHADERS: the vertex stage writes
// Location N and the fragment stage of the same pipeline reads it. Both are
// patched with whatever the global pair is at the time. Move the pair midway
// and a vertex module cached under the old pair can be linked against a
// fragment module patched under the new one - the fragment then reads a
// location nothing wrote, which is undefined data feeding the motion vectors.
// That is worse than the refusals this whole census exists to remove, and it
// would appear as corruption in some pipelines and not others.
static bool g_locLatched = false;

// WHY THE REFUSAL REASON IS COUNTED AND NOT JUST THE REFUSAL.
//
// The old counter said "105 refused" and stopped there, so the only way to
// learn WHY was to add logging and fly again. The reasons are not equivalent:
// NOT_VERTEX is expected and harmless, NO_POSITION means the shader does not
// draw geometry, and LOCATION_TAKEN is the one that is our fault and fixable.
// A single number cannot distinguish "everything that could be patched was"
// from "a fifth of the scene silently writes no motion vectors", and those two
// look identical on screen except that the second one shimmers.
static uint64_t g_injReason[spvinj::INJ_MALFORMED + 1] = {0};

static void mvNoteInjectReason(spvinj::Result r)
{
    if ((int)r >= 0 && (int)r <= (int)spvinj::INJ_MALFORMED) ++g_injReason[(int)r];
}

static void mvLogInjectReasons()
{
    static const char *kName[] = {
        "patched", "not-a-vertex-shader", "no-gl_Position",
        "never-writes-gl_Position", "LOCATION-TAKEN", "malformed"
    };
    std::string s;
    char buf[96];
    for (int i = 0; i <= (int)spvinj::INJ_MALFORMED; ++i) {
        if (!g_injReason[i]) continue;
        snprintf(buf, sizeof(buf), "%s%s=%llu", s.empty() ? "" : "  ",
                 kName[i], (unsigned long long)g_injReason[i]);
        s += buf;
    }
    trace("SPIRV INJECT: outcomes at locations %u/%u - %s",
          spvinj::currClipLocation(), spvinj::prevClipLocation(),
          s.empty() ? "(nothing attempted)" : s.c_str());
    if (g_injReason[spvinj::INJ_LOCATION_TAKEN])
        trace("SPIRV INJECT: %llu shaders still collide with our varying pair "
              "and write NO motion vectors. Those draws reproject from depth "
              "only, which is what shimmers on moving geometry.",
              (unsigned long long)g_injReason[spvinj::INJ_LOCATION_TAKEN]);
    // The injected body is spliced at the module's single OpReturn, which is
    // what makes it correct for shaders that write gl_Position from several
    // mutually exclusive branches. A module with more than one return has no
    // single exit to splice at, so it falls back to the first-store placement -
    // the assumption light_vis disproved. Zero is the expected value; anything
    // else names a real gap rather than a theoretical one.
    if (spvinj::multiReturnModules())
        trace("SPIRV INJECT: %llu vertex modules have MORE THAN ONE OpReturn and "
              "fell back to splicing at the first gl_Position store. That is the "
              "assumption light_vis breaks - if any of these write gl_Position "
              "from several branches, their vectors are wrong.",
              (unsigned long long)spvinj::multiReturnModules());
}

// Re-run until it has a big enough sample, then commit once.
//
// WHY THIS IS ALLOWED TO MOVE THE PAIR AFTER PATCHING HAS STARTED.
//
// The first version refused to: it latched on the first patch, X-Plane creates
// its first pipeline when only 16 modules exist, and the census therefore never
// ran at all - locations stayed at 30/31 and the shaders that collide with them
// kept writing no motion vectors. The latch was protecting against a real
// hazard but with the wrong instrument.
//
// The hazard is a CACHED vertex variant patched under the old pair being linked
// against a fragment patched under the new one. It is not that two pipelines
// disagree - each pipeline patches its vertex and fragment together at creation,
// so a pipeline built at pair A and another built at pair B are both internally
// consistent and both correct. Only the caches can straddle the change.
//
// So the fix is to flush the caches at the instant the pair moves, not to
// forbid the move. Pipelines already built keep the modules they were built
// with, which are still valid; everything created afterwards is re-patched at
// the new pair.
//
// The old patched modules are dropped rather than destroyed. They may still be
// referenced by pipelines already created, and a VkShaderModule is a few KB of
// driver-side object - a few hundred of them is not a leak worth risking a
// use-after-free to reclaim.
static void mvPickLocationsOnce()
{
    static bool done = false;
    if (done) return;

    std::vector<std::vector<uint32_t> > mods;
    {
        std::lock_guard<std::mutex> g(g_lock);
        // A census of 16 modules is barely better than none - it can move the
        // pair onto locations the other 1200 use heavily. Wait for a real
        // sample; patching in the meantime is fine because the pair can still
        // move afterwards.
        if (g_moduleCode.size() < 512) return;
        for (std::map<VkShaderModule, std::vector<uint32_t> >::iterator it =
                 g_moduleCode.begin(); it != g_moduleCode.end(); ++it)
            mods.push_back(it->second);
    }
    done = true;

    const uint32_t kMax = 32;
    bool used[kMax];
    memset(used, 0, sizeof(used));
    for (size_t i = 0; i < mods.size(); ++i)
        spvinj::scanUsedLocations(mods[i].data(), mods[i].size() * 4, used, kMax);

    uint32_t before[2] = { spvinj::currClipLocation(), spvinj::prevClipLocation() };
    if (spvinj::placeLocations(used, kMax)) {
        std::string map;
        for (uint32_t L = 0; L < kMax; ++L) map += used[L] ? '#' : '.';
        bool moved = (before[0] != spvinj::currClipLocation() ||
                      before[1] != spvinj::prevClipLocation());
        size_t flushedV = 0, flushedF = 0;
        if (moved) {
            std::lock_guard<std::mutex> g(g_lock);
            flushedV = g_vertVariant.size();
            flushedF = g_fragVariant.size();
            g_vertVariant.clear();
            g_patchedVertModules.clear();
            g_fragVariant.clear();
            // The refusal tally described the OLD pair. Keeping it would blend
            // two different experiments into one number.
            memset(g_injReason, 0, sizeof(g_injReason));
        }
        trace("SPIRV INJECT: varyings %s %u/%u -> %u/%u after scanning %zu "
              "modules (flushed %zu vertex / %zu fragment variants so later "
              "pipelines re-patch at the new pair). Location map "
              "(# = used by X-Plane): %s",
              moved ? "moved" : "kept", before[0], before[1],
              spvinj::currClipLocation(), spvinj::prevClipLocation(),
              mods.size(), flushedV, flushedF, map.c_str());
    } else {
        std::string map;
        for (uint32_t L = 0; L < kMax; ++L) map += used[L] ? '#' : '.';
        trace("SPIRV INJECT: no free adjacent Location pair in %zu modules - "
              "keeping %u/%u. Shaders using those will still be refused. "
              "Location map (# = used by X-Plane): %s",
              mods.size(), before[0], before[1], map.c_str());
    }
}

// TAA_MV_DUMP_SPIRV writes the first patched vertex modules to disk.
//
// Every link has now been checked by reasoning and each looked correct: the
// offset is chosen before the first patch, the layout carries our range and
// X-Plane declares none of its own, the push uses that layout, and the entry
// point lists the push constant variable for SPIR-V 1.4 and later. The matrix
// still loads as zero, so the reasoning is wrong somewhere and the bytes are
// the only thing left that can say where.
static void mvMaybeDumpSpirv(const std::vector<uint32_t> &code, const char *what)
{
    static const char *dir = getenv("TAA_MV_DUMP_SPIRV");
    if (!dir) return;
    // A counter PER KIND. One shared counter meant the two vertex dumps used the
    // whole budget and the fragment module - the one that has never been
    // inspected, and the side the dead varying is read on - was never written.
    static int nVert = 0, nFrag = 0;
    const bool isVert = (what[0] == 'v');
    int &n = isVert ? nVert : nFrag;
    if (n >= 2) return;
    char path[512];
    // "%s\mv_..." - `\m` is not an escape sequence, so the compiler dropped the
    // backslash and every dump landed as <dir>mv_vert_0.spv: a sibling of the
    // directory rather than a file inside it, or nothing at all if <dir> had no
    // trailing separator. The dumps this writes are the only way to read back
    // what was actually injected, so a silent misplacement is expensive.
    snprintf(path, sizeof(path), "%s\\mv_%s_%d.spv", dir, what, n++);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(code.data(), 4, code.size(), f);
    fclose(f);
    trace("SPIRV INJECT: wrote patched %s module to %s", what, path);
}

static VkShaderModule mvPatchVertex(VkDevice device, VkShaderModule orig)
{
    mvPickLocationsOnce();
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<VkShaderModule, VkShaderModule>::iterator it = g_vertVariant.find(orig);
        if (it != g_vertVariant.end()) return it->second;
    }

    std::vector<uint32_t> src;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<VkShaderModule, std::vector<uint32_t> >::iterator it =
            g_moduleCode.find(orig);
        if (it == g_moduleCode.end()) return VK_NULL_HANDLE;
        src = it->second;
    }

    std::vector<uint32_t> patched;
    uint32_t loc = 0;
    VkShaderModule out = VK_NULL_HANDLE;
    const int dsSet2 = (crashEnabled() && crashOccupancy() && destructgpu::state().ready)
                         ? (int)destructgpu::state().setIndex : -1;
    spvinj::Result ir = spvinj::inject(src.data(), src.size() * 4, patched, &loc, dsSet2);
    mvNoteInjectReason(ir);
    if (ir == spvinj::INJ_OK) mvMaybeDumpSpirv(patched, "vert");
    if (ir == spvinj::INJ_OK) {
        VkShaderModuleCreateInfo smci;
        memset(&smci, 0, sizeof(smci));
        smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = patched.size() * 4;
        smci.pCode    = patched.data();
        if (!g_nextCreateShaderModule ||
            g_nextCreateShaderModule(device, &smci, nullptr, &out) != VK_SUCCESS)
            out = VK_NULL_HANDLE;
    }

    std::lock_guard<std::mutex> g(g_lock);
    g_vertVariant[orig] = out;
    g_patchedVertModules.insert(out);
    if (out == VK_NULL_HANDLE) {
        ++g_vertPatchFail;
        // Report the breakdown on a schedule tied to failures, so a run that
        // refuses a lot says so early rather than only in a summary at exit -
        // the sim is often killed before any exit path runs.
        if (g_vertPatchFail <= 3 || (g_vertPatchFail % 50) == 0)
            mvLogInjectReasons();
    } else if (++g_vertVariants <= 5 || (g_vertVariants % 100) == 0) {
        trace("SPIRV INJECT: vertex variant %llu (%llu refused)",
              (unsigned long long)g_vertVariants,
              (unsigned long long)g_vertPatchFail);
        if ((g_vertVariants % 500) == 0) mvLogInjectReasons();
    }
    return out;
}

// Cached per (module, attachment index). X-Plane builds ~16000 pipelines from
// ~1500 modules, so without the cache this would re-patch and re-create the
// same module thousands of times.
static VkShaderModule mvPatchFragment(VkDevice device, VkShaderModule orig,
                                      uint32_t attachmentIndex, bool alphaBlended)
{
    FragKey key(FragModKey(orig, attachmentIndex), alphaBlended);
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<FragKey, VkShaderModule>::iterator it = g_fragVariant.find(key);
        if (it != g_fragVariant.end()) return it->second;
    }

    std::vector<uint32_t> src;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<VkShaderModule, std::vector<uint32_t> >::iterator it =
            g_moduleCode.find(orig);
        if (it == g_moduleCode.end()) return VK_NULL_HANDLE;
        src = it->second;
    }

    std::vector<uint32_t> patched;
    spvinj::Result fr = spvinj::injectFragment(src.data(), src.size() * 4, patched,
                                               attachmentIndex, alphaBlended);
    // The VERTEX module's Location decorations were verified from a dump - 30
    // and 31, both stored. The FRAGMENT module's INPUT locations never were,
    // and a mismatch there would deliver zeros through a varying that looks
    // perfectly correct on the writing side.
    if (fr == spvinj::INJ_OK) mvMaybeDumpSpirv(patched, "frag");
    if (fr != spvinj::INJ_OK) {
        std::lock_guard<std::mutex> g(g_lock);
        g_fragVariant[key] = VK_NULL_HANDLE;   // remember the refusal too
        ++g_fragPatchFail;
        // EVERY unpatched pipeline this session refused here, with the vertex
        // stage patched and the fragment stage declining - 200 of 200, one
        // cause, not a mixture. The counter said "how many" and never "why",
        // so the reason had to be inferred from reading the injector, twice,
        // wrongly. Print it instead. attachmentIndex is included because it
        // varies per pipeline - it is the pass's colorAttachmentCount - and a
        // refusal that depends on it means the slot is contested, while one
        // that does not means the varyings are.
        static uint64_t nSaid = 0;
        static const char *kWhy[] = {
            "OK", "not-a-fragment-shader", "no-gl_Position",
            "never-writes-gl_Position", "LOCATION-TAKEN", "malformed"
        };
        if (++nSaid <= 10 || (nSaid % 250) == 0)
            trace("MV FRAGMENT REFUSED #%llu: reason=%s at attachmentIndex=%u "
                  "(varyings %u/%u). The vertex stage patched; both-or-neither "
                  "then discards that too, so these draws write no velocity.",
                  (unsigned long long)nSaid,
                  ((int)fr >= 0 && (int)fr <= 5) ? kWhy[(int)fr] : "?",
                  attachmentIndex, spvinj::currClipLocation(),
                  spvinj::prevClipLocation());
        return VK_NULL_HANDLE;
    }

    VkShaderModuleCreateInfo smci;
    memset(&smci, 0, sizeof(smci));
    smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = patched.size() * 4;
    smci.pCode    = patched.data();

    VkShaderModule out = VK_NULL_HANDLE;
    if (!g_nextCreateShaderModule ||
        g_nextCreateShaderModule(device, &smci, nullptr, &out) != VK_SUCCESS) {
        std::lock_guard<std::mutex> g(g_lock);
        g_fragVariant[key] = VK_NULL_HANDLE;
        ++g_fragPatchFail;
        return VK_NULL_HANDLE;
    }

    std::lock_guard<std::mutex> g(g_lock);
    g_fragVariant[key] = out;
    if (++g_fragVariants <= 5 || (g_fragVariants % 200) == 0)
        trace("SPIRV INJECT: fragment variant %llu - velocity at attachment %u "
              "(%llu refused)",
              (unsigned long long)g_fragVariants, attachmentIndex,
              (unsigned long long)g_fragPatchFail);
    return out;
}

static VKAPI_ATTR VkResult VKAPI_CALL TAA_CreateGraphicsPipelines(
    VkDevice device, VkPipelineCache cache, uint32_t count,
    const VkGraphicsPipelineCreateInfo *ci, const VkAllocationCallbacks *alloc,
    VkPipeline *out)
{
    if (ci) {
        for (uint32_t i = 0; i < count; ++i) {
            ++g_gfxPipelines;
            for (uint32_t s = 0; s < ci[i].stageCount; ++s) {
                // A stage with no module handle must be carrying its SPIR-V
                // inline via maintenance5. That is the case that would be
                // missed by hooking vkCreateShaderModule alone.
                if (ci[i].pStages[s].module == VK_NULL_HANDLE) {
                    ++g_inlineModules;
                    if (g_inlineModules <= 3)
                        trace("SPIRV: INLINE module in pipeline stage (maintenance5) "
                              "- vkCreateShaderModule alone would miss this");
                }
            }
        }
        if (g_gfxPipelines <= 3 || g_gfxPipelines % 500 == 0)
            trace("SPIRV: %llu graphics pipelines, %llu modules, %llu inline",
                  (unsigned long long)g_gfxPipelines,
                  (unsigned long long)g_shaderModules,
                  (unsigned long long)g_inlineModules);

        // ---- A PATCHED SHADER ON A LAYOUT WITHOUT OUR RANGE READS GARBAGE.
        //
        // This is the one remaining explanation for a measurement that has never
        // been accounted for: a per-pixel dump of matrix element 12 returns
        // 110.9375, 161.0 and 179.875 where the layer pushed -0.024233. A
        // constant attribute interpolates to itself, so those are per-draw
        // constants, not noise. prevNDC.x - u is about M[12]/d, and 179/180 is
        // 0.6 NDC or roughly 1150 px - exactly the flow the exact-mode
        // diagnostic reports on pixels whose measured depth cannot produce it.
        //
        // Foreign pushes were ruled out by watch counts, offset ordering by
        // construction, and per-thread push state by test. The shader corpus
        // then removed the last external candidate: the string "PushConstant"
        // appears in ZERO of 6855 X-Plane modules, so nothing of X-Plane's is
        // writing that memory. Whatever is wrong is ours.
        //
        // And this is the shape it would take. TAA_CreatePipelineLayout can
        // decline to extend a layout - it falls back deliberately, because a
        // layout we cannot extend is a velocity hole while a layout we fail to
        // create is a crash. But the SHADER is patched independently, so a
        // patched vertex module bound to an unextended layout declares a push
        // constant block that the layout does not provide, and reads whatever
        // is there. That produces exactly this: stable per-draw values,
        // unrelated to anything we pushed.
        //
        // Detect the combination rather than continue to reason about it.
        for (uint32_t i = 0; i < count; ++i) {
            if (ci[i].layout == VK_NULL_HANDLE) continue;
            bool patchedVert = false;
            {
                std::lock_guard<std::mutex> g(g_lock);
                for (uint32_t s = 0; s < ci[i].stageCount && !patchedVert; ++s) {
                    if (!(ci[i].pStages[s].stage & VK_SHADER_STAGE_VERTEX_BIT))
                        continue;
                    VkShaderModule m = ci[i].pStages[s].module;
                    // Either the original that is about to be substituted, or
                    // the substituted handle itself.
                    if (g_vertVariant.count(m) || g_patchedVertModules.count(m))
                        patchedVert = true;
                }
                if (!patchedVert) continue;

                // ---- THE SET MISMATCH, WHICH IS FATAL RATHER THAN WRONG.
                //
                // A patched module declares the storage buffer at descriptor
                // set 7 STATICALLY. Using it with a pipeline layout that does
                // not declare that set is invalid, and now that the occupancy
                // store actually executes it is a GPU fault rather than a bad
                // number: VK_ERROR_DEVICE_LOST on vkQueueSubmit, a few frames
                // in.
                //
                // It could not bite before, because the store was being built
                // into a vector that had already been emitted and so never
                // reached the module at all. Fixing the emission is what made
                // this reachable.
                if (destructgpu::state().ready &&
                    g_layoutOurSet.find(ci[i].layout) == g_layoutOurSet.end()) {
                    static uint64_t nSetMismatch = 0;
                    if (++nSetMismatch <= 3 || (nSetMismatch % 500) == 0)
                        trace("DESTRUCT: *** SET MISMATCH - a PATCHED vertex "
                              "module is bound to layout %p, which does NOT "
                              "carry our descriptor set (%llu so far). The "
                              "shader writes through a set the layout never "
                              "declared, which is a device loss rather than a "
                              "wrong answer. ***",
                              (void*)ci[i].layout,
                              (unsigned long long)nSetMismatch);
                }

                std::map<VkPipelineLayout, bool>::iterator lt =
                    g_layoutHasOurPC.find(ci[i].layout);
                if (lt != g_layoutHasOurPC.end() && lt->second) continue;
            }
            static uint64_t nMismatch = 0;
            if (++nMismatch <= 3 || (nMismatch % 500) == 0)
                trace("SPIRV INJECT: *** MISMATCH - a PATCHED vertex module is "
                      "bound to layout %p, which does NOT carry our push "
                      "constant range (%llu so far). The shader declares the "
                      "block and reads memory the layout never provided, so "
                      "uReproj is whatever happens to be there. X-Plane uses no "
                      "push constants anywhere in 6855 modules, so this is the "
                      "only way a foreign matrix can reach that shader - and it "
                      "is the shape of the M[12]=111/161/180 readings. ***",
                      (void*)ci[i].layout, (unsigned long long)nMismatch);
        }
    }
    // EVERY graphics pipeline gains the velocity attachment format, not just
    // the ones we patched.
    //
    // Under dynamic rendering a pipeline's colour formats must agree with the
    // pass it runs in. We add the attachment slot to every pass, so every
    // pipeline has to declare it or become invalid everywhere. Pipelines whose
    // fragment shader could not be patched get colorWriteMask 0 for that slot:
    // present, correctly formatted, writing nothing, leaving the cleared zero -
    // which reads as "did not move" and is the honest answer for geometry we
    // have no vectors for.
    std::vector<VkGraphicsPipelineCreateInfo> ci2;
    std::vector<VkPipelineRenderingCreateInfo> rinfo;
    std::vector<std::vector<VkFormat> > fmts;
    std::vector<VkPipelineColorBlendStateCreateInfo> blends;
    std::vector<std::vector<VkPipelineColorBlendAttachmentState> > blendAtt;
    std::vector<std::vector<VkPipelineShaderStageCreateInfo> > stages;
    std::vector<char> mvPatchedThisCall;

    if (ci && count) mvPatchedThisCall.assign(count, 0);

    if (g_spirvLive && ci && count) {
        ci2.assign(ci, ci + count);
        rinfo.resize(count);
        fmts.resize(count);
        blends.resize(count);
        blendAtt.resize(count);
        stages.resize(count);

        for (uint32_t i = 0; i < count; ++i) {
            // Find the dynamic rendering info in the pNext chain. A pipeline
            // without one is using a render pass object, which X-Plane does not
            // - measured, 0 vkCmdBeginRenderPass calls in 2760 frames - so
            // leaving those alone costs nothing.
            const VkPipelineRenderingCreateInfo *src = nullptr;
            for (const VkBaseInStructure *p = (const VkBaseInStructure*)ci[i].pNext;
                 p; p = p->pNext)
                if (p->sType == VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO)
                    src = (const VkPipelineRenderingCreateInfo*)p;
            if (!src || !ci[i].pColorBlendState) continue;
            // Mirror of the pass hook's depth-only rule: passes with zero
            // colour attachments are no longer extended, so a zero-attachment
            // pipeline must keep zero formats or it mismatches the very pass
            // it runs in.
            if (src->colorAttachmentCount == 0) continue;

            // ---- FULL-SCREEN PIPELINES MUST NOT WRITE VELOCITY.
            //
            // A pipeline with no vertex attributes draws the modern full-screen
            // triangle from gl_VertexIndex. Its gl_Position is a screen corner,
            // not a point in the world, so reprojecting it through uReproj
            // produces a number that means nothing - and because the triangle
            // covers the frame, it writes that number over EVERY pixel.
            //
            // This is what the dumps were showing. Thirteen passes a frame bind
            // the velocity target (they are all full-resolution with depth,
            // which is the whole of the isScene test), and a full-screen pass
            // landing among them replaces a correct field with a screen-uniform
            // one. It matches the signature exactly: p25 through p95 all within
            // a few tenths of each other at ~350 px, on a frame whose matrix
            // predicts 13.15 - and it explains why the same frame could read
            // correct or uniformly wrong depending on which pass wrote last.
            //
            // The blend state below already sets the velocity write mask to 0
            // for anything left unpatched, so declining here is sufficient: the
            // attachment stays bound and keeps whatever the geometry wrote.
            // Full-screen pipelines (no vertex attributes) must not WRITE
            // velocity - reprojecting a screen corner means nothing and the
            // triangle covers the frame - but they MUST still declare the
            // attachment format. This used to `continue`, leaving them N
            // formats inside N+1-attachment passes: an attachment-count
            // mismatch on every post-process draw, which is the undefined
            // behaviour behind the intermittent DEVICE_LOST at flight load
            // and view changes. Extend the formats, skip the shader patch;
            // the unpatched path below already sets colorWriteMask 0.
            // ---- "NO VERTEX ATTRIBUTES" IS NOT "FULL-SCREEN QUAD".
            //
            // It was, once. X-Plane 12 draws its TERRAIN by pulling vertices
            // from storage buffers, so the terrain pipelines declare no vertex
            // attributes either - and were therefore classified as post-process
            // quads and left unpatched. The ground then writes no velocity at
            // all, which the sentinel makes visible: the rejection viz is red
            // across the runway and grass and yellow on the trees and buildings,
            // because those are ordinary attribute meshes and did get patched.
            //
            // Depth separates them. A full-screen pass neither tests nor writes
            // depth - it covers the frame unconditionally - while any pipeline
            // drawing world geometry does at least one of the two.
            const VkPipelineDepthStencilStateCreateInfo *dss = ci[i].pDepthStencilState;

            // ---- X-PLANE SETS DEPTH STATE DYNAMICALLY, SO THE CREATE INFO
            //      SAYS NOTHING.
            //
            // The rule above is right and its test was not. Reading
            // depthTestEnable out of pDepthStencilState answers "was depth
            // baked on at pipeline creation", not "does this pipeline use
            // depth" - and X-Plane sets depth test, depth write and the
            // compare op through DYNAMIC STATE, so the create info reads
            // VK_FALSE for terrain and terrain stayed indistinguishable from a
            // post-process quad. Measured with viz=8: the written map is BLACK
            // across the entire ground, white only on the wings, which are
            // ordinary attribute meshes that were patched.
            //
            // A pipeline that declares any of the depth dynamic states intends
            // to use depth. On its own that is too loose - it would sweep in
            // genuine post-process passes and stamp a screen-space vector
            // across them - so it must ALSO be drawing into a pass that
            // actually has a depth attachment. Both together is the terrain
            // signature and nothing else's.
            bool dynamicDepth = false;
            if (ci[i].pDynamicState) {
                for (uint32_t d = 0; d < ci[i].pDynamicState->dynamicStateCount; ++d) {
                    const VkDynamicState ds = ci[i].pDynamicState->pDynamicStates[d];
                    if (ds == VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE ||
                        ds == VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE ||
                        ds == VK_DYNAMIC_STATE_DEPTH_COMPARE_OP) {
                        dynamicDepth = true;
                        break;
                    }
                }
            }
            const bool intoDepth =
                src && src->depthAttachmentFormat != VK_FORMAT_UNDEFINED;
            const bool touchesDepth =
                (dss && (dss->depthTestEnable == VK_TRUE ||
                         dss->depthWriteEnable == VK_TRUE))
                || (dynamicDepth && intoDepth);
            const bool quadRuleDepthAware =
                live::onoff("taa.quad_needs_depth", "TAA_QUAD_NEEDS_DEPTH", true);

            // Depth alone is too loose - it rescued the post-process quads and
            // they stamped the frame. Require the vertex stage to actually PULL
            // its vertices from a buffer as well; see spirvPullsVertices.
            //   taa.quad_needs_pull = 1  require the buffer read too (default)
            //                         0  depth alone (the loose rule; stamps)
            const bool quadRulePull =
                live::onoff("taa.quad_needs_pull", "TAA_QUAD_NEEDS_PULL", true);
            const bool zeroAttribs =
                !ci[i].pVertexInputState ||
                ci[i].pVertexInputState->vertexAttributeDescriptionCount == 0;

            bool pullsVertices = false;
            if (quadRulePull || zeroAttribs) {
                for (uint32_t st = 0; st < ci[i].stageCount; ++st) {
                    if (!(ci[i].pStages[st].stage & VK_SHADER_STAGE_VERTEX_BIT))
                        continue;
                    std::map<VkShaderModule, std::vector<uint32_t> >::iterator mit =
                        g_moduleCode.find(ci[i].pStages[st].module);
                    if (mit != g_moduleCode.end()) {
                        pullsVertices = spirvPullsVertices(mit->second);

                        // ---- REPORT EVERY DISTINCT ZERO-ATTRIBUTE MODULE ONCE.
                        //
                        // Keyed by word count plus the fingerprint bits, so one
                        // line appears per KIND of shader rather than per
                        // pipeline. The terrain and the post-process quads are
                        // both in here; this is what tells them apart, and it
                        // is the step that should have come before either of
                        // the two discriminators that were guessed.
                        if (zeroAttribs) {
                            const VsFingerprint f = spirvVsFingerprint(mit->second);
                            const uint32_t key =
                                (f.words << 10) ^
                                ((uint32_t)f.vertexIndex      << 0) ^
                                ((uint32_t)f.instanceIndex    << 1) ^
                                ((uint32_t)f.ssbo             << 2) ^
                                ((uint32_t)f.uniform          << 3) ^
                                ((uint32_t)f.bufferBlock      << 4) ^
                                ((uint32_t)f.runtimeArray     << 5) ^
                                ((uint32_t)f.imageFetchOrRead << 6) ^
                                ((uint32_t)f.sampledImage     << 7) ^
                                ((uint32_t)f.texelBuffer      << 8);
                            static std::set<uint32_t> seenVs;
                            std::lock_guard<std::mutex> g(g_lock);
                            if (seenVs.size() < 60 && !seenVs.count(key)) {
                                seenVs.insert(key);
                                trace("VS ZEROATTR: %5u words | vtxIdx=%d instIdx=%d "
                                      "| ssbo=%d ubo=%d bufblock=%d rtarray=%d "
                                      "| imgfetch=%d sampled=%d texelbuf=%d "
                                      "| bufLoads=%u | depth=%d",
                                      f.words, (int)f.vertexIndex, (int)f.instanceIndex,
                                      (int)f.ssbo, (int)f.uniform, (int)f.bufferBlock,
                                      (int)f.runtimeArray, (int)f.imageFetchOrRead,
                                      (int)f.sampledImage, (int)f.texelBuffer,
                                      f.nUniformLoads, (int)touchesDepth);
                            }
                        }
                    }
                    break;
                }
            }
            const bool rescued = quadRuleDepthAware && touchesDepth &&
                                 (!quadRulePull || pullsVertices);

            if (rescued && (!ci[i].pVertexInputState ||
                 ci[i].pVertexInputState->vertexAttributeDescriptionCount == 0)) {
                static uint64_t nPulled = 0;
                if (++nPulled % 200 == 1)
                    trace("MV: %llu vertex-PULLED pipeline(s) rescued - zero "
                          "attributes but the vertex stage reads a storage "
                          "buffer, which is the terrain", 
                          (unsigned long long)nPulled);
            }
            bool noPatch = false;
            if ((!ci[i].pVertexInputState ||
                 ci[i].pVertexInputState->vertexAttributeDescriptionCount == 0)
                && !rescued) {
                static uint64_t nFullScreen = 0, nStamps = 0;
                // Pipelines the LOOSE rule would have rescued and this one
                // declines: zero attributes, touches depth, but reads no
                // buffer to place itself. Those are the frame-stamping
                // post-process quads, and their count is the whole difference
                // between a saturated field and a correct one.
                if (quadRuleDepthAware && touchesDepth && !pullsVertices)
                    ++nStamps;
                if (++nFullScreen % 500 == 1)
                    trace("MV: %llu full-screen pipeline(s) - formats extended, "
                          "shaders left alone, write mask 0 (%llu of them would "
                          "have been rescued by depth alone and would stamp the "
                          "frame)",
                          (unsigned long long)nFullScreen,
                          (unsigned long long)nStamps);
                noPatch = true;
            }

            // One extra format, at index colorAttachmentCount - matching the
            // single extra slot the pass hook appends, and matching the
            // Location the fragment shader is patched to write.
            if (src->colorAttachmentCount >= 8) continue;
            uint32_t mvIndex = src->colorAttachmentCount;
            fmts[i].assign(src->pColorAttachmentFormats,
                           src->pColorAttachmentFormats + src->colorAttachmentCount);
            fmts[i].push_back(kMvFormat);

            rinfo[i] = *src;
            rinfo[i].colorAttachmentCount    = (uint32_t)fmts[i].size();
            rinfo[i].pColorAttachmentFormats = fmts[i].data();


            // ---- WHY DEPTH STATE IS ONLY MEASURED HERE, NOT ACTED ON.
            //
            // The velocity attachment is written with blending OFF, so the last
            // draw covering a pixel wins outright. That is right for geometry -
            // the frontmost surface owns the pixel's motion - but it means a
            // single full-screen composite could stamp its own vectors over the
            //
            // Acting on that reading was wrong, twice over, and both mistakes
            // are worth leaving written down:
            //
            //   The premise did not survive arithmetic. "13 px everywhere with
            //   the camera still" was called implausible before it was checked.
            //   The dump reported 0.002 m of camera movement, and 2 mm against
            //   cockpit geometry half a metre away is ~0.23 deg, which over a
            //   60 deg field at 3840 px is ~15 px. The measurement was fine.
            //
            //   The discriminator did not survive contact. Excluding pipelines
            //   whose depthTestEnable is false removed 79% of them and zeroed
            //   the velocity field. Nothing renders 79% of its pipelines without
            //   depth testing - so that field is not saying what it appears to.
            //   The likely reason is dynamic state: if a pipeline lists
            //   VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, the driver ignores the value
            //   in pDepthStencilState entirely and takes it from the command
            //   buffer, leaving whatever the application happened to put in the
            //   struct - frequently zero.
            //
            // So this counts the three cases apart and reports them. No pipeline
            // is excluded until the numbers say which signal is real.
            {
                const VkPipelineDepthStencilStateCreateInfo *ds = ci[i].pDepthStencilState;
                bool dynDepth = false;
                if (ci[i].pDynamicState && ci[i].pDynamicState->pDynamicStates) {
                    for (uint32_t d = 0; d < ci[i].pDynamicState->dynamicStateCount; ++d) {
                        VkDynamicState s = ci[i].pDynamicState->pDynamicStates[d];
                        if (s == VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE ||
                            s == VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE_EXT) { dynDepth = true; break; }
                    }
                }
                // AND THE SAME QUESTION ASKED OF BLEND STATE, which is the far
                // more important one.
                //

                static uint64_t nNull = 0, nDyn = 0, nOff = 0, nOn = 0;
                if (!ds)                          ++nNull;
                else if (dynDepth)                ++nDyn;
                else if (!ds->depthTestEnable)    ++nOff;
                else                              ++nOn;
                uint64_t tot = nNull + nDyn + nOff + nOn;
                if (tot == 1000 || tot % 5000 == 0)
                    trace("MV DEPTH STATE: no-ds %llu, dynamic %llu, static-off %llu, "
                          "static-on %llu (of %llu)",
                          (unsigned long long)nNull, (unsigned long long)nDyn,
                          (unsigned long long)nOff,  (unsigned long long)nOn,
                          (unsigned long long)tot);
            }

            // PATCH THE FRAGMENT SHADER HERE, now that the attachment index and
            bool fragPatched = false;
            stages[i].assign(ci[i].pStages, ci[i].pStages + ci[i].stageCount);
            bool vertPatched = false;
            for (uint32_t s = 0; !noPatch && s < ci[i].stageCount; ++s) {
                if (ci[i].pStages[s].stage & VK_SHADER_STAGE_FRAGMENT_BIT) {
                    // Genuine transparency, read from the pipeline's own blend
                    // state rather than guessed from a G-buffer channel.
                    const VkPipelineColorBlendStateCreateInfo *cb =
                        ci[i].pColorBlendState;
                    const bool alphaBlended =
                        cb && cb->attachmentCount > 0 &&
                        cb->pAttachments[0].blendEnable == VK_TRUE;
                    VkShaderModule use = mvPatchFragment(
                        device, ci[i].pStages[s].module, mvIndex, alphaBlended);
                    if (use != VK_NULL_HANDLE) {
                        stages[i][s].module = use;
                        fragPatched = true;
                    }
                } else if (ci[i].pStages[s].stage & VK_SHADER_STAGE_VERTEX_BIT) {
                    VkShaderModule use = mvPatchVertex(device, ci[i].pStages[s].module);
                    if (use != VK_NULL_HANDLE) {
                        stages[i][s].module = use;
                        vertPatched = true;
                    }
                }
            }

            // BOTH OR NEITHER. A patched fragment shader reads varyings only a
            // patched vertex shader writes; pairing one with an unpatched
            // partner gives undefined inputs, which read as zero and produce a
            // velocity field of zeros that looks like working plumbing.
            mvPatchedThisCall[i] = (fragPatched && vertPatched);

            // ---- WHICH PIPELINES FAIL, AND ON WHICH STAGE.
            //
            // The MV target is cleared to zero every frame, so a pipeline that
            // is not patched does not degrade gracefully - it writes nothing,
            // the pixels keep zero, and zero means "this pixel did not move".
            // FSR2 then fetches history from the same screen position while the
            // camera moves, which pins those pixels and trails them. That is a
            // patchy smear on exactly the geometry that failed to patch, which
            // is what the terrain is doing while the cockpit stays clean.
            //
            // Bind-weighted coverage said 98.6% and that was reassuring for the
            // wrong reason: it counts DRAWS, not AREA. A terrain tile and a
            // cockpit switch are one bind each, and 1.4% of binds can be most of
            // the screen. So the useful number is not the percentage, it is
            // WHICH pipelines and WHY.
            if (!mvPatchedThisCall[i] && !noPatch) {
                // Name the shaders themselves, once each: a pipeline that fails
                // both stages draws geometry with no jitter and no velocity,
                // and knowing WHICH module defeats the injector is the whole
                // difference between "895 failures" and a fixable list.
                if (!vertPatched && !fragPatched) {
                    static std::set<VkShaderModule> named;
                    for (uint32_t s = 0; s < ci[i].stageCount; ++s) {
                        VkShaderModule m = ci[i].pStages[s].module;
                        if (m == VK_NULL_HANDLE || named.count(m)) continue;
                        if (named.size() >= 64) break;
                        named.insert(m);
                        trace("MV UNPATCHABLE MODULE %p stage 0x%x - this "
                              "pipeline draws geometry with neither jitter nor "
                              "velocity; the shader idiom here is what the "
                              "injector cannot handle",
                              (void*)m, (unsigned)ci[i].pStages[s].stage);
                    }
                }
                static uint64_t nFail = 0, nVertOnly = 0, nFragOnly = 0, nNeither = 0;
                ++nFail;
                if (vertPatched && !fragPatched) ++nVertOnly;
                else if (!vertPatched && fragPatched) ++nFragOnly;
                else ++nNeither;
                if (nFail <= 8 || (nFail % 200) == 0) {
                    trace("MV PIPELINE UNPATCHED #%llu: vert=%d frag=%d "
                          "(running: vertex-only %llu, fragment-only %llu, "
                          "neither %llu). These draws write no velocity and the "
                          "target is cleared to zero, so their pixels read as "
                          "stationary and trail.",
                          (unsigned long long)nFail, vertPatched ? 1 : 0,
                          fragPatched ? 1 : 0,
                          (unsigned long long)nVertOnly,
                          (unsigned long long)nFragOnly,
                          (unsigned long long)nNeither);
                    mvLogInjectReasons();
                }
            }
            // ---- HOW MANY PIPELINES ACTUALLY GET BOTH STAGES PATCHED.
            //
            // currClip has been the control for every conclusion tonight: it
            // arrives correctly, so the varying mechanism was assumed sound and
            // prevClip's zero was blamed on the matrix. That control is only
            // valid if the VERTEX module is really substituted. If it is not,
            // Location 30 is carrying some pre-existing X-Plane varying whose .w
            // merely looks like plausible depth, Location 31 carries nothing,
            // and the whole reading is an illusion.
            //
            // A pipeline needs BOTH stages patched or it gets neither, so this
            // counts the outcome per pipeline instead of per module.
            {
                // ---- DELIBERATE SKIPS ARE NOT FAILURES, AND POOLING THEM HID
                //      THE ONLY NUMBER THAT MATTERS.
                //
                // Full-screen pipelines are skipped on purpose - they have no
                // vertex attributes, so there is no geometry to displace and
                // nothing to reproject - but they were counted in "neither"
                // alongside genuine patch failures. That made ~900 look like
                // un-jittered geometry when most of it is post-process quads
                // that correctly want none. Split them: "declined" is by
                // design, "neither" is geometry we FAILED to patch, and only
                // the latter renders unjittered while the resolve un-jitters
                // the whole image.
                static uint64_t nBoth = 0, nVertOnly = 0, nFragOnly = 0,
                                nNeither = 0, nDeclined = 0;
                if (noPatch)                         ++nDeclined;
                else if (fragPatched && vertPatched) ++nBoth;
                else if (vertPatched)                ++nVertOnly;
                else if (fragPatched)                ++nFragOnly;
                else                                 ++nNeither;
                const uint64_t tot = nBoth + nVertOnly + nFragOnly + nNeither
                                   + nDeclined;
                if (tot == 500 || (tot % 5000) == 0)
                    trace("CRASH PROBE: %llu vertex modules carry view depth, %llu fragment modules read it back - zero on either side means the probe never emitted and a black screen says nothing about the geometry",
                          (unsigned long long)spvinj::probeVsCount(),
                          (unsigned long long)spvinj::probeFsCount());
                    trace("SPIRV INJECT: pipelines by patch outcome - both %llu, "
                          "vertex only %llu, fragment only %llu, NEITHER %llu, "
                          "declined-by-design %llu (of %llu). Only NEITHER is "
                          "geometry drawn without jitter while the resolve "
                          "un-jitters the frame.",
                          (unsigned long long)nBoth, (unsigned long long)nVertOnly,
                          (unsigned long long)nFragOnly, (unsigned long long)nNeither,
                          (unsigned long long)nDeclined,
                          (unsigned long long)tot);
            }

            if (fragPatched && vertPatched) {
                ci2[i].pStages = stages[i].data();
            } else {
                fragPatched = false;   // no velocity from this pipeline
                stages[i].clear();
            }

            // Non-null: the loop skipped this pipeline otherwise.
            const VkPipelineColorBlendStateCreateInfo *sb = ci[i].pColorBlendState;
            blendAtt[i].assign(sb->pAttachments,
                               sb->pAttachments + sb->attachmentCount);

            VkPipelineColorBlendAttachmentState mvBlend;
            memset(&mvBlend, 0, sizeof(mvBlend));
            mvBlend.blendEnable    = VK_FALSE;   // a velocity is replaced, never blended
            // R and G only. The target is two channels, so enabling B would
            // name a component the attachment does not have.
            //
            // Under TAA_MV_RGBA the attachment IS four channels and the shader
            // writes depths into zw. The mask has to open or they are dropped
            // silently - which is exactly what happened on the first exact-mode
            // run: coverage 99.7%, no image failures, the report printed its
            // header, and every depth read back as zero because B and A were
            // masked off downstream of a shader that wrote them correctly.
            // All four channels, always: the target is RGBA16F now, and the
            // alpha channel CARRIES the C13/C14 coverage gate - masking it off
            // would silently drop the reactive mask the resolve depends on,
            // the same silent-channel failure the RGBA note below records.
            const VkColorComponentFlags mvMaskOn =
                  VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            mvBlend.colorWriteMask = fragPatched ? mvMaskOn : 0;
            // ---- ALPHA-BLENDED PIPELINES GET A PER-PIXEL SELECT, NOT A STAMP.
            //
            // With blending off, the propeller disc and the canopy glass write
            // their velocity across their entire quad footprint - including
            // texels that are visually transparent - overwriting the correct
            // vector of whatever is behind them (C13).
            //
            // Velocities must never be MIXED, but they can be SELECTED: the
            // patched fragment puts a hard 0-or-1 in its output alpha (its own
            // colour alpha thresholded at 0.5), and SRC_ALPHA blending with a
            // binary source is a select - opaque texels replace the velocity,
            // transparent texels keep the one underneath. The blend stage reads
            // source alpha from the shader output, so this works on the
            // two-channel RG16F attachment; the format merely has nowhere to
            // STORE alpha, which is fine because nothing reads it back.
            //
            // Only for pipelines whose OWN attachment 0 blends - opaque draws
            // keep the plain replace - and only outside the debug channel
            // modes, where .w carries diagnostics rather than a 0-or-1 gate and
            // "blending" it would corrupt both.
            {
                static const bool debugChannels =
                    getenv("TAA_MV_WRITE_DEPTH") || getenv("TAA_MV_RGBA") ||
                    getenv("TAA_MV_FIELDCHK")   || getenv("TAA_MV_MATDUMP") ||
                    getenv("TAA_MV_RAWCLIP")    || getenv("TAA_MV_PID");
                if (fragPatched && !debugChannels &&
                    sb->attachmentCount > 0 && sb->pAttachments[0].blendEnable) {
                    mvBlend.blendEnable         = VK_TRUE;
                    mvBlend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
                    mvBlend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                    mvBlend.colorBlendOp        = VK_BLEND_OP_ADD;
                    mvBlend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                    mvBlend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
                    mvBlend.alphaBlendOp        = VK_BLEND_OP_ADD;
                }
            }
            blendAtt[i].push_back(mvBlend);

            blends[i] = *sb;
            blends[i].attachmentCount = (uint32_t)blendAtt[i].size();
            blends[i].pAttachments    = blendAtt[i].data();

            // Rebuild the pNext chain with our rendering info in place of the
            // original. The struct is copied, so the application's own remains
            // untouched - it may well reuse it for the next pipeline.
            rinfo[i].pNext = src->pNext;
            ci2[i].pNext = &rinfo[i];
            ci2[i].pColorBlendState = &blends[i];
        }
    }

    // Pipeline-compile stutter telemetry (SS59): count and time every driver
    // compile. Creations after the first present are JIT - the classic
    // mid-flight hitch - and the per-frame peak names the guilty frame.
    LARGE_INTEGER vpc0, vpc1, vpcf;
    QueryPerformanceCounter(&vpc0);
    VkResult r = g_nextCreateGfxPipelines
        ? g_nextCreateGfxPipelines(device, cache, count,
                                   ci2.empty() ? ci : ci2.data(), alloc, out)
        : VK_ERROR_INITIALIZATION_FAILED;
    QueryPerformanceCounter(&vpc1);
    QueryPerformanceFrequency(&vpcf);
    if (vpcf.QuadPart > 0)
        vram::notePipelines(count,
            (uint64_t)((vpc1.QuadPart - vpc0.QuadPart) * 1000000ll /
                       vpcf.QuadPart));

    // If the extended pipelines were rejected, retry unmodified. A pipeline we
    // cannot extend is a hole in the velocity buffer; a pipeline that fails to
    // create is a dead sim.
    if (r != VK_SUCCESS && !ci2.empty()) {
        // ---- RETRY ONE AT A TIME, NOT THE WHOLE BATCH.
        //
        // vkCreateGraphicsPipelines takes an ARRAY. One bad pipeline makes the
        // whole call fail, and dropping back to the originals for the entire
        // batch threw away every good pipeline alongside it. Measured: 14,835
        // rejections out of ~16,000 pipelines, all VK_ERROR_UNKNOWN, and the
        // survivors were too few to write a velocity field - which is why
        // prevClip read zero and why that zero was blamed on the push constant,
        // the offset, the varying location and the matrix in turn.
        //
        // Rebuilding individually keeps every pipeline the driver will accept
        // and falls back only for the ones it will not. It also names them: the
        // attachment count of a pipeline that fails alone is the evidence for
        // what the driver objects to.
        uint32_t okExtended = 0, fellBack = 0;
        for (uint32_t i = 0; i < count; ++i) {
            out[i] = VK_NULL_HANDLE;

            const VkGraphicsPipelineCreateInfo *one = ci2.empty() ? &ci[i] : &ci2[i];
            VkResult r1 = g_nextCreateGfxPipelines(device, cache, 1, one, alloc, &out[i]);
            if (r1 == VK_SUCCESS) { ++okExtended; continue; }

            r1 = g_nextCreateGfxPipelines(device, cache, 1, &ci[i], alloc, &out[i]);
            ++fellBack;
            ++g_pipeRejected;
            if (fellBack <= 4) {
                const VkPipelineRenderingCreateInfo *src = nullptr;
                for (const VkBaseInStructure *pn = (const VkBaseInStructure*)ci[i].pNext;
                     pn; pn = pn->pNext)
                    if (pn->sType == VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO)
                        src = (const VkPipelineRenderingCreateInfo*)pn;
                trace("SPIRV INJECT: pipeline %u of %u refused extended (%d) - "
                      "colour attachments %u, stages %u", i, count, (int)r1,
                      src ? src->colorAttachmentCount : 0u, ci[i].stageCount);
            }
            if (r1 != VK_SUCCESS) r = r1;
        }
        if (r != VK_SUCCESS && fellBack == 0) r = VK_SUCCESS;
        else if (fellBack) {
            static uint64_t nBatches = 0;
            if (++nBatches <= 8 || (nBatches % 200) == 0)
                trace("SPIRV INJECT: batch of %u - %u extended, %u fell back "
                      "(a whole batch used to be discarded for one failure)",
                      count, okExtended, fellBack);
            r = VK_SUCCESS;
        }
    }

    // Remember each pipeline's layout. vkCmdBindPipeline names only the
    // pipeline, and no Vulkan query maps a VkPipeline back to its layout - the
    // association exists only here, in the create info, which is gone by the
    // time a draw is recorded.
    if (r == VK_SUCCESS && ci && out && g_spirvInject) {
        std::lock_guard<std::mutex> g(g_lock);
        for (uint32_t i = 0; i < count; ++i) {
            if (out[i] == VK_NULL_HANDLE) continue;
            g_pipelineLayoutOf[out[i]] = ci[i].layout;

            // IS THIS PIPELINE GEOMETRY, OR A FULL-SCREEN QUAD?
            //
            // The jitter must reach the scene's triangles and nothing else. A
            // full-screen composite or post-process pass that is ALSO jittered
            // displaces the image a second time on top of the geometry that
            // already moved - which is precisely what viewport jitter did, and
            // what moving jitter into the vertex shader is meant to stop. Doing
            // it per-draw only helps if there is something to decide on.
            //
            // Vertex attributes are that something, and unlike depthTestEnable -
            // which the census found to be dynamic state on 100% of X-Plane's
            // pipelines, and therefore worth nothing at creation time - this is
            // fixed at creation and cannot be overridden later. Scene geometry
            // is drawn from vertex buffers; the modern full-screen triangle is
            // drawn from gl_VertexIndex with no vertex input at all.
            //
            // Wrong in the safe direction if X-Plane turns out to draw a quad
            // from a buffer: that quad gets jittered, which is the behaviour we
            // already have. The census below says how often it fires, so this is
            // a measurement rather than a belief.
            bool hasAttribs = ci[i].pVertexInputState &&
                              ci[i].pVertexInputState->vertexAttributeDescriptionCount > 0;
            g_pipelineIsGeometry[out[i]] = hasAttribs;
            // Whether THIS pipeline ended up writing motion vectors. Module
            // counts cannot answer the question that matters - a module is not
            // a pixel. One unpatched pipeline drawing the terrain matters more
            // than fifty patched ones drawing cockpit switches.
            g_pipelineMvPatched[out[i]] =
                (i < mvPatchedThisCall.size()) && mvPatchedThisCall[i] != 0;
            if (hasAttribs) ++g_pipeGeometry; else ++g_pipeFullscreen;
        }
    }
    return r;
}

// Push uReproj after any bind of a pipeline whose layout carries our range.
//
// Once per bind rather than once per frame, because push constants are command
// buffer state tied to a layout and are invalidated when a pipeline with an
// incompatible layout is bound. Pushing once at the start of a frame would
// survive exactly until X-Plane's next bind.
//
// The value is g_velSnap.reproj - prevViewProj * inverse(currViewProj) - which
// the plugin already computes and the scripted self-test already verified at
// 8.78 px against 8.77 predicted. This is the same matrix the depth path uses;
// the only difference is that the shader applies it to a position it knows
// exactly rather than one recovered from a quantised depth value.
static PFN_vkCmdBindPipeline  g_nextCmdBindPipeline = nullptr;
static PFN_vkCmdPushConstants g_nextCmdPushConstants = nullptr;
// Resolved once beside the push entry point, for the same reason: the
// draw path must not take the global lock to find it.
static PFN_vkCmdBindDescriptorSets g_nextCmdBindSets = nullptr;
static thread_local bool g_inOurPush = false;
static uint64_t g_foreignPushes = 0;
static uint32_t g_foreignLo = 0xFFFFFFFFu, g_foreignHi = 0;
static PFN_vkCmdDraw            g_nextCmdDraw           = nullptr;
static PFN_vkCmdDrawIndexed     g_nextCmdDrawIndexed    = nullptr;
static PFN_vkCmdDrawIndirect            g_nextCmdDrawIndirect            = nullptr;
static PFN_vkCmdDrawIndexedIndirect     g_nextCmdDrawIndexedIndirect     = nullptr;
static PFN_vkCmdDrawIndirectCount       g_nextCmdDrawIndirectCount       = nullptr;
static PFN_vkCmdDrawIndexedIndirectCount g_nextCmdDrawIndexedIndirectCount = nullptr;
static uint64_t                 g_drawRepushes          = 0;

// A colour image layout transition. Small, explicit, and local: the blit that
// replaces X-Plane's upscale needs both its source and destination in transfer
// layouts and back again, and the surrounding code has no idea what layout
// either was in.
static void velImageBarrier(DeviceData &dd, VkCommandBuffer cb, VkImage img,
                            VkImageLayout from, VkImageLayout to,
                            VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                            VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage)
{
    if (img == VK_NULL_HANDLE || !dd.cmdPipelineBarrier) return;
    VkImageMemoryBarrier bar;
    memset(&bar, 0, sizeof(bar));
    bar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    bar.oldLayout = from;
    bar.newLayout = to;
    bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.image = img;
    bar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bar.subresourceRange.levelCount = 1;
    bar.subresourceRange.layerCount = 1;
    bar.srcAccessMask = srcAccess;
    bar.dstAccessMask = dstAccess;
    dd.cmdPipelineBarrier(cb, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &bar);
}


// Put the upscaled result into the images X-Plane's own FSR would have written.
//
// Called from the FSR2 dispatch site, in the SAME command buffer as the work
// that produced it. Doing this at X-Plane's dispatch instead put the read in a
// different buffer from the write, with three queues in play and nothing
// ordering them - a race, and the tiled corruption that comes with one.

// X-Plane's FSR is a compute dispatch, so this is the interception point: the
// dispatch is simply never forwarded. Nothing else in the frame changes, and
// the sim has no way to observe that its upscale did not happen.
// ---- WHERE X-PLANE ACTUALLY NAMES ITS FSR RESOURCES.
//
// This layer tracked descriptor sets only through vkUpdateDescriptorSets, and
// X-Plane does not bind the upscale's resources that way - so the set was
// invisible and the dispatch had nothing to write into. Seven storage images
// share the output's extent, which is why extent alone could not settle it.
//
// A push descriptor states the type of every binding. The FSR output is
// declared STORAGE_IMAGE, so recording the storage-image writes on this command
// buffer names the destination outright.
//
// Forwarded unconditionally: this is observation, and the sim's own upscale
// must keep working exactly as before whenever fsr.replace is off.
// The Vulkan 1.4 form. Same job, arguments wrapped in a struct.
//
// The classic entry point counted ZERO calls while 98 descriptor-set binds went
// past per frame, which is what sent the search here rather than to another
// guess about naming.
// ---- RECORD ONE PIXEL FROM EVERY CANDIDATE.
//
// Called immediately after X-Plane's FSR dispatch has been forwarded, in the
// SAME command buffer, so the copies are ordered after the writes that produced
// them. Done once: the answer does not change for the life of the process.
static void fsrProbeRecord(DeviceData &dd, VkCommandBuffer cb,
                           uint32_t wantW, uint32_t wantH)
{
    fsrprobe::State &ps = fsrprobe::state();
    if (ps.resolved || ps.failed || ps.copiesRecorded) return;
    // ---- NOT DURING THE LOAD.
    //
    // The first FSR dispatch happens while the sim is still loading, and the
    // command buffer carrying it is not necessarily submitted. The copies were
    // recorded into one that was discarded: every candidate read back as
    // 0 0 0 0, alpha included, which is the memset and not image content - a
    // real frame has alpha 1. Recording is not execution.
    if (g_frameCount < 240) return;
    if (!dd.createBuffer || !dd.cmdCopyImageToBuffer || !dd.cmdPipelineBarrier ||
        !g_getPhysMemProps) { ps.failed = true; return; }

    // Candidates: every storage image of the output's exact extent.
    ps.candidates.clear();
    for (std::map<VkImage, ColorTarget>::iterator it = g_colorImages.begin();
         it != g_colorImages.end(); ++it) {
        if (!(it->second.usage & VK_IMAGE_USAGE_STORAGE_BIT)) continue;
        // ---- EVERY PLAUSIBLE OUTPUT EXTENT, NOT ONE DERIVED ONE.
        //
        // wantW/wantH assume a 16x16 tile per 64-thread group. If the tile is
        // really 8x8 the output is half that in each axis - and the rendered
        // image cannot tell the two apart, because a 16x16 mapping over a
        // 1920x1080 target still fills it completely: the groups that would
        // run past the edge are discarded by the shader's own bounds check.
        // So a correct-looking frame is consistent with BOTH, and the probe
        // must not exclude one on the strength of it.
        if (it->second.w < 1920 || it->second.h < 1080) continue;
        if (ps.candidates.size() >= fsrprobe::kMaxCandidates) break;
        ps.candidates.push_back(it->second.image);
    }
    // ---- SWAPCHAIN IMAGES ARE CANDIDATES TOO.
    //
    // They can never appear in g_colorImages: they come from
    // vkGetSwapchainImagesKHR, not vkCreateImage, so the creation hook that
    // fills that map never sees them. If X-Plane's upscale writes straight into
    // the image it is about to present - which it may, when the swapchain was
    // created with STORAGE usage - then the destination was never in the
    // candidate list at all, and the probe would report "not found" forever
    // while everything else about it worked.
    //
    // That is exactly the state this reached: the sentinel is provably in the
    // shader, the shader provably runs (the upscaled image is correct), the
    // copies provably execute, and no candidate carried the stamp.
    for (std::map<VkSwapchainKHR, std::vector<VkImage> >::iterator si =
             g_swapImages.begin(); si != g_swapImages.end(); ++si) {
        // ---- UNCONDITIONALLY. NO EXTENT FILTER.
        //
        // The extent recorded for a swapchain is 0x0 - g_swapInfo has no entry
        // for it - so filtering on a match excluded every presented image, and
        // those are precisely the ones that can never appear in g_colorImages:
        // they come from vkGetSwapchainImagesKHR, not vkCreateImage.
        //
        // There are three of them. Copying a pixel from three extra images
        // costs nothing next to excluding the one category the answer might be
        // hiding in.
        for (size_t k = 0; k < si->second.size(); ++k) {
            if (ps.candidates.size() >= fsrprobe::kMaxCandidates) break;
            bool have = false;
            for (size_t q = 0; q < ps.candidates.size(); ++q)
                if (ps.candidates[q] == si->second[k]) { have = true; break; }
            if (!have) ps.candidates.push_back(si->second[k]);
        }
    }

    if (ps.candidates.empty()) return;      // nothing to ask about yet

    if (ps.buf == VK_NULL_HANDLE) {
        const VkDeviceSize bytes =
            (VkDeviceSize)fsrprobe::kMaxCandidates * fsrprobe::kPixelBytes;
        VkBufferCreateInfo bci;
        memset(&bci, 0, sizeof(bci));
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = bytes;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (dd.createBuffer(dd.device, &bci, nullptr, &ps.buf) != VK_SUCCESS) {
            ps.failed = true; trace("FSR PROBE: buffer creation failed"); return;
        }
        VkMemoryRequirements mr;
        dd.getBufferMemReq(dd.device, ps.buf, &mr);
        VkPhysicalDeviceMemoryProperties mp;
        memset(&mp, 0, sizeof(mp));
        g_getPhysMemProps(dd.phys, &mp);
        uint32_t ti = UINT32_MAX;
        for (uint32_t k = 0; k < mp.memoryTypeCount; ++k)
            if ((mr.memoryTypeBits & (1u << k)) &&
                (mp.memoryTypes[k].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (mp.memoryTypes[k].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                ti = k; break;
            }
        VkMemoryAllocateInfo mai;
        memset(&mai, 0, sizeof(mai));
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = mr.size;
        mai.memoryTypeIndex = ti;
        if (ti == UINT32_MAX ||
            dd.allocateMemory(dd.device, &mai, nullptr, &ps.mem) != VK_SUCCESS ||
            dd.bindBufferMemory(dd.device, ps.buf, ps.mem, 0) != VK_SUCCESS ||
            dd.mapMemory(dd.device, ps.mem, 0, bytes, 0, &ps.ptr) != VK_SUCCESS) {
            ps.failed = true; trace("FSR PROBE: memory failed"); return;
        }
        memset(ps.ptr, 0, (size_t)bytes);
        ps.device = dd.device;
    }

    for (size_t i = 0; i < ps.candidates.size(); ++i) {
        VkImageMemoryBarrier b;
        memset(&b, 0, sizeof(b));
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        b.image = ps.candidates[i];
        // GENERAL is where a storage image lives; ALL_COMMANDS because a
        // candidate that is NOT the output may have been touched by anything.
        b.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                              0, nullptr, 0, nullptr, 1, &b);

        VkBufferImageCopy bic;
        memset(&bic, 0, sizeof(bic));
        bic.bufferOffset = (VkDeviceSize)i * fsrprobe::kPixelBytes;
        bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        bic.imageSubresource.layerCount = 1;
        bic.imageExtent.width = 1; bic.imageExtent.height = 1; bic.imageExtent.depth = 1;
        dd.cmdCopyImageToBuffer(cb, ps.candidates[i],
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                ps.buf, 1, &bic);

        b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                              0, nullptr, 0, nullptr, 1, &b);
    }

    ps.copiesRecorded = true;
    ps.copiedOnFrame  = g_frameCount;
    trace("FSR PROBE: copied pixel (0,0) from %u candidate(s), any storage image at least 1920x1080 (the %ux%u guess is no longer trusted). The one "
          "carrying the sentinel our shader stamps is X-Plane's real upscale "
          "output.", (unsigned)ps.candidates.size(), wantW, wantH);
}

// Read the answer, a few frames after the copies were recorded. No fence: see
// the note in fsr_probe.h.
static void fsrProbeResolve()
{
    fsrprobe::State &ps = fsrprobe::state();
    if (ps.resolved || ps.failed || !ps.copiesRecorded || !ps.ptr) return;
    if (g_frameCount < ps.copiedOnFrame + 3) return;

    const uint8_t *base = (const uint8_t *)ps.ptr;
    for (size_t i = 0; i < ps.candidates.size(); ++i) {
        if (!fsrprobe::looksLikeSentinel(base + i * fsrprobe::kPixelBytes)) continue;
        ps.output   = ps.candidates[i];
        ps.resolved = true;
        trace("FSR PROBE: X-Plane's upscale output is %p - candidate %u of %u "
              "carried the sentinel. This handle is what a CPU-driven upscaler "
              "must be given to write.",
              (void*)ps.output, (unsigned)i, (unsigned)ps.candidates.size());
        return;
    }
    // ---- SAY WHAT IS ACTUALLY THERE.
    //
    // "Does this pixel match" cannot tell you where the value went, and three
    // attempts were spent theorising about that instead of looking. The
    // sentinel is provably written by a shader that provably runs, so it is
    // SOMEWHERE - print every candidate's contents and let the numbers say
    // which, or say plainly that none of them was ever written at all.
    {
        static bool dumped = false;
        if (!dumped) {
            dumped = true;
            const uint8_t *b0 = (const uint8_t *)ps.ptr;
            trace("FSR PROBE: no match. Contents of pixel (0,0) in each "
                  "candidate - the sentinel is (%.4f %.4f %.4f):",
                  fsrprobe::kSentinel[0], fsrprobe::kSentinel[1],
                  fsrprobe::kSentinel[2]);
            for (size_t i = 0; i < ps.candidates.size(); ++i) {
                uint16_t h[4];
                memcpy(h, b0 + i * fsrprobe::kPixelBytes, sizeof(h));
                std::map<VkImage, ColorTarget>::iterator ct =
                    g_colorImages.find(ps.candidates[i]);
                trace("FSR PROBE:   [%2u] %p  %.4f %.4f %.4f %.4f   %ux%u usage=0x%x",
                      (unsigned)i, (void*)ps.candidates[i],
                      fsrprobe::halfToFloat(h[0]), fsrprobe::halfToFloat(h[1]),
                      fsrprobe::halfToFloat(h[2]), fsrprobe::halfToFloat(h[3]),
                      ct != g_colorImages.end() ? ct->second.w : 0,
                      ct != g_colorImages.end() ? ct->second.h : 0,
                      ct != g_colorImages.end() ? (unsigned)ct->second.usage : 0u);
            }
            // Everything the layer knows that is big enough to be an upscale
            // target, whether or not it was a candidate - so an output that was
            // excluded by the filters shows up here rather than being invisible.
            trace("FSR PROBE: every tracked image at least 1920 wide:");
            unsigned n = 0;
            for (std::map<VkImage, ColorTarget>::iterator it = g_colorImages.begin();
                 it != g_colorImages.end() && n < 40; ++it) {
                if (it->second.w < 1920) continue;
                ++n;
                trace("FSR PROBE:   %p %ux%u fmt=%d usage=0x%x storage=%s",
                      (void*)it->second.image, it->second.w, it->second.h,
                      (int)it->second.format, (unsigned)it->second.usage,
                      (it->second.usage & VK_IMAGE_USAGE_STORAGE_BIT) ? "yes" : "no");
            }
            trace("FSR PROBE: swapchain images tracked: %u chain(s)",
                  (unsigned)g_swapImages.size());
            for (std::map<VkSwapchainKHR, std::vector<VkImage> >::iterator si =
                     g_swapImages.begin(); si != g_swapImages.end(); ++si) {
                std::map<VkSwapchainKHR, SwapInfo>::iterator ii =
                    g_swapInfo.find(si->first);
                trace("FSR PROBE:   chain %p: %u image(s) %ux%u",
                      (void*)si->first, (unsigned)si->second.size(),
                      ii != g_swapInfo.end() ? ii->second.w : 0,
                      ii != g_swapInfo.end() ? ii->second.h : 0);
            }
        }
    }

    // Not found: say so once and allow one retry, rather than silently
    // reporting nothing forever.
    static int retries = 0;
    if (retries++ < 20) {
        // Retried properly rather than twice. The failure mode this is built
        // for - copies recorded into a command buffer that is never submitted -
        // is transient, so giving up after two attempts turns a timing problem
        // into a permanent "not found".
        if (retries <= 3 || (retries % 10) == 0)
            trace("FSR PROBE: sentinel not found in %u candidate(s), attempt %d "
                  "- retrying on a later frame.",
                  (unsigned)ps.candidates.size(), retries);
        ps.copiesRecorded = false;
        ps.copiedOnFrame  = g_frameCount;
    } else {
        ps.failed = true;
        trace("FSR PROBE: giving up after %d attempts.", retries);
    }
}

static VKAPI_ATTR void VKAPI_CALL TAA_CmdPushDescriptorSet2(
    VkCommandBuffer cb, const VkPushDescriptorSetInfo *info)
{
    PFN_vkCmdPushDescriptorSet2 next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, DeviceData>::iterator it = g_devices.find(dispatchKey(cb));
        if (it == g_devices.end()) it = g_devices.begin();
        if (it != g_devices.end()) next = it->second.cmdPushDescriptorSet2;

        if (info && info->pDescriptorWrites) {
            static uint64_t nPush2 = 0;
            if (nPush2++ < 4)
                trace("PUSH DESC 2: call %llu stages=0x%x writes=%u",
                      (unsigned long long)nPush2, (unsigned)info->stageFlags,
                      info->descriptorWriteCount);
            std::vector<VkImage> &v = g_cbPushedStorage[(void*)cb];
            for (uint32_t i = 0; i < info->descriptorWriteCount; ++i) {
                const VkWriteDescriptorSet &w = info->pDescriptorWrites[i];
                if (w.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) continue;
                if (!w.pImageInfo) continue;
                for (uint32_t d = 0; d < w.descriptorCount; ++d) {
                    std::map<VkImageView, VkImage>::iterator im =
                        g_viewToImage.find(w.pImageInfo[d].imageView);
                    if (im == g_viewToImage.end()) continue;
                    if (v.size() > 32) v.erase(v.begin(), v.begin() + 16);
                    v.push_back(im->second);
                }
            }
        }
    }
    if (next) next(cb, info);
}

static VKAPI_ATTR void VKAPI_CALL TAA_CmdPushDescriptorSetKHR(
    VkCommandBuffer cb, VkPipelineBindPoint bindPoint, VkPipelineLayout layout,
    uint32_t set, uint32_t writeCount, const VkWriteDescriptorSet *writes)
{
    PFN_vkCmdPushDescriptorSetKHR next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, DeviceData>::iterator it = g_devices.find(dispatchKey(cb));
        if (it == g_devices.end()) it = g_devices.begin();
        if (it != g_devices.end()) next = it->second.cmdPushDescriptorSet;

        // ---- IS THIS HOOK CALLED AT ALL, AND WITH WHAT?
        //
        // Four rebuilds were spent inferring why the FSR output stayed unnamed:
        // KHR name, core 1.4 name, descriptor sets, image-map fallback. Every
        // one of those was a guess about a call nobody had confirmed happens.
        // Count it and say what it carries.
        {
            static uint64_t nPush = 0;
            if (nPush++ < 6)
                trace("PUSH DESC: call %llu bindPoint=%d writes=%u",
                      (unsigned long long)nPush, (int)bindPoint, writeCount);
            if (writes && nPush <= 6)
                for (uint32_t i = 0; i < writeCount && i < 8; ++i)
                    trace("PUSH DESC:   write %u type=%d count=%u",
                          i, (int)writes[i].descriptorType, writes[i].descriptorCount);
        }
        if (writes && bindPoint == VK_PIPELINE_BIND_POINT_COMPUTE) {
            std::vector<VkImage> &v = g_cbPushedStorage[(void*)cb];
            for (uint32_t i = 0; i < writeCount; ++i) {
                if (writes[i].descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                    continue;
                if (!writes[i].pImageInfo) continue;
                for (uint32_t d = 0; d < writes[i].descriptorCount; ++d) {
                    std::map<VkImageView, VkImage>::iterator im =
                        g_viewToImage.find(writes[i].pImageInfo[d].imageView);
                    if (im == g_viewToImage.end()) continue;
                    // Newest last, and bounded: a command buffer that pushes
                    // all frame would otherwise grow without limit.
                    if (v.size() > 32) v.erase(v.begin(), v.begin() + 16);
                    v.push_back(im->second);
                }
            }
        }
    }
    if (next) next(cb, bindPoint, layout, set, writeCount, writes);
}

static VKAPI_ATTR void VKAPI_CALL TAA_CmdDispatch(
    VkCommandBuffer cb, uint32_t gx, uint32_t gy, uint32_t gz)
{
    DeviceData *dd = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, DeviceData>::iterator it = g_devices.find(dispatchKey(cb));
        if (it == g_devices.end()) it = g_devices.begin();
        if (it != g_devices.end()) dd = &it->second;
    }
    // ---- THE TAKEOVER SWITCH.
    //
    // OFF (default): X-Plane's FSR is forwarded untouched and behaves exactly
    // as Laminar intended. This layer must never change what the sim does with
    // a feature the user did not point at us.
    //
    // ON: X-Plane's own FSR toggle becomes the control for OUR upscaler. The
    // sim renders the 3-D scene BELOW display resolution through its own
    // supported path - which is the part a layer cannot do safely on its own -
    // and then dispatches its spatial upscale. We drop that dispatch and write
    // our own result into the image it would have produced.
    //
    // In through X-Plane's path, out through X-Plane's supported output. No
    // viewport games, no second swap chain, and the reduced render is a
    // documented sim setting rather than something we forced.
    //
    // g_cbFsrBound was recorded at vkCmdBindPipeline: the pipeline was built
    // from a module carrying u_fsr_data, which is X-Plane's FSR uniform block
    // and survives a recompile because it lives in OpName.
    if (dd && fsrReplaceEnabled()) {
        bool fsrBound = false;
        {
            std::lock_guard<std::mutex> g(g_lock);
            std::map<void*, bool>::iterator fb = g_cbFsrBound.find((void*)cb);
            fsrBound = (fb != g_cbFsrBound.end() && fb->second);
        }
        // ---- THE DISPATCH IS NO LONGER DROPPED.
        //
        // It runs, and what runs is our module. Dropping it was the earlier
        // route and it could not be completed: it needed the output image named
        // from outside, which X-Plane does not expose.
        //
        // Kept, disabled, because this counter and its trace are how anyone
        // checks the interception point is still being reached.
        if (false && fsrBound) {
            ++g_xpFsrDropped;

            // ---- PUT OUR RESULT WHERE X-PLANE'S UPSCALE WOULD HAVE PUT ITS OWN.
            //
            // The destination is the LARGEST image the FSR descriptor sets
            // named: an upscaler writes bigger than it reads, so size separates
            // output from input without needing to know X-Plane's binding
            // numbers - which are not documented and would not survive a
            // recompile if they were.
            //
            // The source is the scene target, which at this point in the frame
            // holds the resolved low-resolution image: X-Plane rendered it
            // small because ITS OWN FSR setting is on, and our resolve has
            // already run over it. So this blit is the upscale, and the rest of
            // the frame - tonemap, cockpit, UI - happens downstream exactly as
            // the sim intended, because we are handing back the same image it
            // was going to read.
            VkImage dst = VK_NULL_HANDLE;
            uint32_t dw = 0, dh = 0;
            VkImage src = VK_NULL_HANDLE;
            uint32_t sw = 0, sh = 0;
            {
                std::lock_guard<std::mutex> g(g_lock);
                std::map<void*, std::vector<VkImage> >::iterator fi =
                    g_cbFsrImages.find((void*)cb);
                // ---- THE DISPATCH STATES ITS OWN OUTPUT SIZE.
                //
                // A compute upscale covers its destination exactly once, so the
                // grid times the workgroup size IS the output extent - measured
                // at 240x135 groups for a 1920x1080 output, which fixes the
                // workgroup at 8x8. Matching on that plus the STORAGE bit
                // identifies X-Plane's i_output_texture outright, where
                // "largest image in the set" was only ever a ranking of
                // guesses.
                const uint32_t wantW = gx * 8, wantH = gy * 8;
                if (fi != g_cbFsrImages.end()) {
                    for (size_t i = 0; i < fi->second.size(); ++i) {
                        std::map<VkImage, ColorTarget>::iterator ct =
                            g_colorImages.find(fi->second[i]);
                        if (ct == g_colorImages.end()) continue;
                        if (!(ct->second.usage & VK_IMAGE_USAGE_STORAGE_BIT)) continue;
                        if (ct->second.w != wantW || ct->second.h != wantH) continue;
                        dst = ct->second.image;
                        dw  = ct->second.w;
                        dh  = ct->second.h;
                        break;
                    }
                }

                // ---- NO DESCRIPTOR SET? FIND IT IN THE IMAGE MAP INSTEAD.
                //
                // g_setViews is fed only by vkUpdateDescriptorSets. X-Plane
                // binds the FSR resources by push descriptor or update
                // template, and this layer hooks neither, so the descriptor
                // route reports zero candidates and always will.
                //
                // It was only ever a way of naming one image, and the dispatch
                // names it already: an upscale covers its destination exactly
                // once, so the grid times the workgroup gives the extent. Every
                // created image is recorded with its usage, so the output is
                // findable without knowing which set held it.
                //
                // AMBIGUITY IS NOT RESOLVED BY PICKING ONE. Writing into the
                // wrong same-sized target produces a frame that looks plausible
                // and is wrong, which costs far more to attribute later than a
                // refusal costs now.
                // The push descriptor named it. Search newest-first: the
                // upscale's output is pushed immediately before its dispatch.
                if (dst == VK_NULL_HANDLE) {
                    std::map<void*, std::vector<VkImage> >::iterator ps =
                        g_cbPushedStorage.find((void*)cb);
                    if (ps != g_cbPushedStorage.end()) {
                        for (size_t i = ps->second.size(); i-- > 0; ) {
                            std::map<VkImage, ColorTarget>::iterator ct =
                                g_colorImages.find(ps->second[i]);
                            if (ct == g_colorImages.end()) continue;
                            if (ct->second.w != wantW || ct->second.h != wantH) continue;
                            dst = ct->second.image;
                            dw = ct->second.w; dh = ct->second.h;
                            static bool said = false;
                            if (!said) {
                                said = true;
                                trace("XP FSR: output named by its PUSH DESCRIPTOR "
                                      "- storage image %p at %ux%u. No guessing "
                                      "between same-sized candidates.",
                                      (void*)dst, dw, dh);
                            }
                            break;
                        }
                    }
                }

                // ---- THE WORKGROUP SIZE IS NOT ASSUMED.
                //
                // Reading the 240x135 grid as 8x8 threads gave 1920x1080, and
                // five rebuilds went looking for that image. FSR's EASU pass
                // uses 16x16, which makes the same grid 3840x2160 - this
                // display, and the extent 11 of this frame's passes run at.
                // The target was never 1920x1080.
                //
                // So every plausible group size is tried, and the one yielding
                // EXACTLY ONE storage image of that extent wins. Ambiguity is
                // still refused: writing into the wrong same-sized target gives
                // a frame that looks plausible and is not.
                if (dst == VK_NULL_HANDLE) {
                    static const uint32_t kGroups[] = { 16, 8, 32, 64 };
                    for (size_t gi = 0; gi < sizeof(kGroups)/sizeof(kGroups[0]); ++gi) {
                        const uint32_t tw = gx * kGroups[gi], th = gy * kGroups[gi];
                        VkImage only = VK_NULL_HANDLE;
                        uint32_t matches = 0;
                        for (std::map<VkImage, ColorTarget>::iterator it2 = g_colorImages.begin();
                             it2 != g_colorImages.end(); ++it2) {
                            if (!(it2->second.usage & VK_IMAGE_USAGE_STORAGE_BIT)) continue;
                            if (it2->second.w != tw || it2->second.h != th) continue;
                            ++matches;
                            only = it2->second.image;
                        }
                        static uint32_t saidN = 0;
                        if (saidN < 8) {
                            ++saidN;
                            trace("XP FSR: group %ux%u -> output would be %ux%u; "
                                  "%u storage image(s) match",
                                  kGroups[gi], kGroups[gi], tw, th, matches);
                        }
                        if (matches == 1) {
                            dst = only; dw = tw; dh = th;
                            static bool said = false;
                            if (!said) {
                                said = true;
                                trace("XP FSR: output is %p at %ux%u - the only "
                                      "storage image matching a %ux%u dispatch "
                                      "at %ux%u threads per group.",
                                      (void*)dst, dw, dh, gx, gy,
                                      kGroups[gi], kGroups[gi]);
                            }
                            break;
                        }
                    }
                }
                src = g_sceneColor.image;
                sw  = g_sceneColor.w;
                sh  = g_sceneColor.h;
            }

            if (dst != VK_NULL_HANDLE && src != VK_NULL_HANDLE &&
                dst != src && sw && sh && dw && dh && dd->cmdBlitImage) {
                // Scene target out of colour-attachment layout, FSR output out
                // of GENERAL - a storage image is always in GENERAL, which is
                // what makes it legal for the shader that was going to write it.
                velImageBarrier(*dd, cb, src,
                                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                VK_ACCESS_TRANSFER_READ_BIT,
                                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                VK_PIPELINE_STAGE_TRANSFER_BIT);
                velImageBarrier(*dd, cb, dst,
                                VK_IMAGE_LAYOUT_GENERAL,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                VK_ACCESS_SHADER_WRITE_BIT,
                                VK_ACCESS_TRANSFER_WRITE_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_TRANSFER_BIT);

                VkImageBlit blit;
                memset(&blit, 0, sizeof(blit));
                blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                blit.srcSubresource.layerCount = 1;
                blit.dstSubresource = blit.srcSubresource;
                blit.srcOffsets[1].x = (int32_t)sw;
                blit.srcOffsets[1].y = (int32_t)sh;
                blit.srcOffsets[1].z = 1;
                blit.dstOffsets[1].x = (int32_t)dw;
                blit.dstOffsets[1].y = (int32_t)dh;
                blit.dstOffsets[1].z = 1;
                // LINEAR because this IS the upscale. A nearest blit here would
                // be a resolution downgrade dressed up as a feature.
                dd->cmdBlitImage(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                 dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 1, &blit, VK_FILTER_LINEAR);

                velImageBarrier(*dd, cb, dst,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                VK_IMAGE_LAYOUT_GENERAL,
                                VK_ACCESS_TRANSFER_WRITE_BIT,
                                VK_ACCESS_SHADER_READ_BIT,
                                VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
                velImageBarrier(*dd, cb, src,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_ACCESS_TRANSFER_READ_BIT,
                                VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                VK_ACCESS_SHADER_READ_BIT,
                                VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

                ++g_xpFsrBlits;
                if (g_xpFsrBlits <= 3 || (g_xpFsrBlits % 600) == 0)
                    trace("XP FSR: upscaled %ux%u -> %ux%u into X-Plane's own "
                          "output image %p (%llu blits). We are the upscaler "
                          "now.", sw, sh, dw, dh, (void*)dst,
                          (unsigned long long)g_xpFsrBlits);
            } else {
                // Named, because a dropped dispatch with nothing written in its
                // place is exactly the corrupted frame this feature is meant to
                // stop producing.
                ++g_xpFsrNoTarget;
                if (g_xpFsrNoTarget <= 3 || (g_xpFsrNoTarget % 600) == 0)
                {
                    std::lock_guard<std::mutex> g(g_lock);
                    std::map<void*, std::vector<VkImage> >::iterator fi2 =
                        g_cbFsrImages.find((void*)cb);
                    const size_t n2 = (fi2 == g_cbFsrImages.end()) ? 0 : fi2->second.size();
                    // Split the remaining possibilities with one number
                    // each, instead of another attempt:
                    //   storageSets == 0  -> no STORAGE write was ever recorded,
                    //                        so the update path is still wrong.
                    //   storageSets  > 0  -> writes ARE recorded, but the sets
                    //                        bound on this buffer are not among
                    //                        them, so bind-sets is the problem.
                    //   binds == 0        -> vkCmdBindDescriptorSets never ran
                    //                        on this command buffer at all.
                    trace("XP FSR: DIAG storageSets=%u sampledSets=%u binds=%u "
                          "viewToImage=%u colourImages=%u",
                          (unsigned)g_setStorageViews.size(),
                          (unsigned)g_setViews.size(),
                          (unsigned)g_cbFsrImages.size(),
                          (unsigned)g_viewToImage.size(),
                          (unsigned)g_colorImages.size());
                    trace("XP FSR: dispatch dropped but NO target written. "
                          "Wanted a STORAGE image of %ux%u (from a %ux%u grid); "
                          "%u candidate(s) were bound on this buffer. "
                          "src=%p scene %ux%u",
                          gx * 8, gy * 8, gx, gy, (unsigned)n2,
                          (void*)src, sw, sh);
                    if (fi2 != g_cbFsrImages.end())
                        for (size_t i = 0; i < fi2->second.size() && i < 12; ++i) {
                            std::map<VkImage, ColorTarget>::iterator ct =
                                g_colorImages.find(fi2->second[i]);
                            if (ct == g_colorImages.end())
                                trace("XP FSR:   candidate %p - not in the colour map",
                                      (void*)fi2->second[i]);
                            else
                                trace("XP FSR:   candidate %p %ux%u usage=0x%x storage=%s",
                                      (void*)ct->second.image, ct->second.w, ct->second.h,
                                      (unsigned)ct->second.usage,
                                      (ct->second.usage & VK_IMAGE_USAGE_STORAGE_BIT)
                                          ? "yes" : "no");
                        }
                }
            }
            // Loud for the first few, then rare: a dispatch dropped every
            // frame is thousands of lines an hour, and a count nobody can see
            // is how the occupancy write went missing for a week.
            if (g_xpFsrDropped <= 4 || (g_xpFsrDropped % 600) == 0)
                trace("XP FSR: dropped X-Plane's upscale dispatch %ux%ux%u "
                      "(%llu so far). fsr.replace=1, so this layer owns the "
                      "upscale now. Until a backend writes into its output "
                      "image the frame WILL look wrong - that is the "
                      "interception working, not failing.",
                      gx, gy, gz, (unsigned long long)g_xpFsrDropped);
            return;                      // never forwarded
        }
    }
    // ---- FSR 3 RUNS HERE, IN PLACE OF THE DISPATCH.
    //
    // Not instead of X-Plane's upscale - that slot is already ours, because the
    // shader in it was substituted at module creation. This chooses which
    // upscaler fills it: our Catmull-Rom shader (forward the dispatch) or FSR3
    // (record its ten passes into this same command buffer and skip it).
    //
    // Recording into X-Plane's own buffer means ordering against the scene
    // render and everything downstream is the sim's, and needs no
    // synchronisation of ours.
    bool fsr3Ran = false;
    if (dd && fsrReplaceEnabled() && fsr3Wanted()) {
        VkImage  colour = VK_NULL_HANDLE, depth = VK_NULL_HANDLE;
        VkImage  mv = VK_NULL_HANDLE,     out = VK_NULL_HANDLE;
        VkFormat colourFmt = VK_FORMAT_UNDEFINED, mvFmt = VK_FORMAT_UNDEFINED;
        VkFormat outFmt = VK_FORMAT_UNDEFINED;
        uint32_t rw = 0, rh = 0, ow = 0, oh = 0;
        bool ours = false;
        {
            std::lock_guard<std::mutex> g(g_lock);
            std::map<void*, bool>::iterator fb = g_cbFsrBound.find((void*)cb);
            ours   = (fb != g_cbFsrBound.end() && fb->second);
            colour = g_sceneColor.image;  colourFmt = g_sceneColor.format;
            rw     = g_sceneColor.w;      rh = g_sceneColor.h;
            depth  = g_sceneDepth;
            mv     = g_mv.image;          mvFmt = VK_FORMAT_R16G16B16A16_SFLOAT;
            out    = fsrprobe::state().output;
            std::map<VkImage, ColorTarget>::iterator oc = g_colorImages.find(out);
            if (oc != g_colorImages.end()) {
                ow = oc->second.w; oh = oc->second.h; outFmt = oc->second.format;
            }
        }
        // ---- ensure() IS NOT CALLED HERE, AND THAT IS THE POINT.
        //
        // Creating the context builds pipelines, allocates memory and creates
        // descriptor pools. Doing that from inside vkCmdDispatch means doing it
        // while a command buffer is being recorded, from within a Vulkan call -
        // which took the sim down with no trace at all, before a single FSR3
        // line printed. It is the same hazard this file already documents for
        // the XeSS probe.
        //
        // The context is built from the present path instead, where nothing is
        // being recorded. Here we only dispatch, and only once it is ready.
        if (ours && out != VK_NULL_HANDLE && colour != VK_NULL_HANDLE &&
            depth != VK_NULL_HANDLE && mv != VK_NULL_HANDLE && rw && rh && ow && oh &&
            fsr3::state().ready) {
            // ---- THE UNITS, FROM THE RESOLVE'S OWN ACCESSORS.
            //
            // Velocity is stored in UV. taa.comp fetches history at
            // uv + (vel.x, velYSign * vel.y), so UV to pixels is the render
            // size and the Y sign is the resolve's - X-Plane draws with a
            // negative-height viewport, so it is -1. Taking both from
            // taaVelScale()/taaVelYSign() rather than restating them is what
            // stops FSR3 and the resolve drifting apart.
            const float vs = taaVelScale();
            const float ys = taaVelYSign();
            fsr3Ran = fsr3::dispatch(
                cb, colour, colourFmt, depth, VK_FORMAT_D32_SFLOAT, mv, mvFmt,
                out, outFmt,
                // Jitter in PIXELS, which is what FSR3 wants. g_velSnap holds
                // the plugin's request and g_jitterScale is the amplitude the
                // layer actually applied - the resolve converts the same pair
                // to NDC with 2*j*scale/width, so taking it before that
                // conversion is the pixel value.
                g_velSnap.jitterX * g_jitterScale,
                g_velSnap.jitterY * g_jitterScale,
                vs * (float)rw, vs * ys * (float)rh,
                16.6f, false,
                0.1f, 100000.0f, 1.0472f);
        }
    }

    if (!fsr3Ran && dd && dd->cmdDispatch) dd->cmdDispatch(cb, gx, gy, gz);

    // After the dispatch, in the same command buffer: our substituted shader
    // has just written the output, sentinel and all.
    if (dd && fsrReplaceEnabled()) {
        bool fsrBound2 = false;
        {
            std::lock_guard<std::mutex> g(g_lock);
            std::map<void*, bool>::iterator fb = g_cbFsrBound.find((void*)cb);
            fsrBound2 = (fb != g_cbFsrBound.end() && fb->second);
            // OURS specifically. X-Plane's RCAS pass also dispatches at this
            // grid and runs after ours, so probing on any FSR dispatch reads a
            // pixel that has already been overwritten - which is exactly what
            // "sentinel not found in 13 candidates" was reporting.
            // ---- THE FIRST FSR DISPATCH OF A FRAME IS EASU, WHICH IS OURS.
            //
            // Two FSR pipelines dispatch per frame at this same grid: EASU,
            // which we substituted, and RCAS, which sharpens the result
            // afterwards. Probing after RCAS reads a pixel X-Plane has already
            // overwritten, which is what "sentinel not found in 13 candidates"
            // was reporting.
            //
            // Matching the pipeline handle would be more direct and did not
            // work - the tag never reached g_cbFsrOurs - so this uses ordering
            // instead, which is a property of the algorithm rather than of our
            // bookkeeping: a spatial upscale must produce the image before a
            // sharpener can sharpen it.
            static uint64_t lastProbeFrame = ~0ull;
            const bool firstThisFrame = (lastProbeFrame != g_frameCount);
            if (fsrBound2 && firstThisFrame && (gx * 16) > 64) {
                lastProbeFrame = g_frameCount;
                fsrProbeRecord(*dd, cb, gx * 16, gy * 16);
            }
        }
    }
}

// Re-push immediately before the draw. Cheap - a push constant write is a few
// dwords into the command stream - and it is the only point at which nothing
// can intervene between the value and its use.
// ---- OFF BY DEFAULT. IT COST 9 FPS.
//
// This re-pushed the matrix before every draw to test whether an incompatible
// layout bind was invalidating it. The test ran - 4.3 million re-pushes - and
// the answer was no, the matrix still arrived as zeros. So the experiment is
// finished, and what remained was a GLOBAL MUTEX taken on every draw call while
// X-Plane records command buffers on several threads. Every draw on every
// thread serialising on one lock took the sim to 9 fps.
//
// Kept behind TAA_MV_REPUSH because the mechanism is worth being able to retest,
// but never on by default. A diagnostic that halves the frame rate is a
// diagnostic that changes what it measures.
// ---- RE-PUSH BEFORE EVERY DRAW. ON BY DEFAULT NOW.
//
// The field is a uniform rigid rotation of the right shape but 8 to 21 times
// the magnitude the recorded matrix encodes, and the factor varies per frame.
// A uniform field IS a rotation, so the shader is applying SOME matrix - just
// not the one that was current when the draw ran. Pushing at bind time leaves a
// window: anything that re-records or replays between the bind and the draw
// consumes whatever was last written into that command buffer.
//
// Writing it immediately before each draw closes the window by construction.
// TAA_MV_NO_REPUSH turns it off to compare.
// ---- THE DESTRUCTION SET, BOUND WHERE IT SURVIVES.
//
// Immediately before the draw is the only point at which nothing can intervene
// between the binding and its use - which is the same sentence the push
// constant re-push above is justified by, because it is the same problem.
//
// pp->layout is the PIPELINE'S layout, and therefore the extended one: sets
// 0..N-1 are described exactly as X-Plane declared them, so binding at index 7
// leaves its own bound sets undisturbed.
static void mvRebindDestructSet(VkCommandBuffer cb, const PendingPush *pp)
{
    if (!pp || pp->destructSet == UINT32_MAX) return;
    // A set index without the layout it was measured against is not a binding,
    // it is a guess at one. Both come from the same pipeline bind, so either
    // both are present or neither is usable.
    if (pp->layout == VK_NULL_HANDLE) return;
    if (!g_nextCmdBindSets || !destructgpu::state().ready) return;
    VkDescriptorSet set = destructgpu::state().set;
    g_nextCmdBindSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pp->layout,
                      pp->destructSet, 1, &set, 0, nullptr);
    ++destructgpu::drawRebinds();
}

static void mvRepushBeforeDraw(VkCommandBuffer cb)
{
    static const bool off = (getenv("TAA_MV_NO_REPUSH") != nullptr);
    PendingPush *pp = g_tlPushSlots.find(cb);
    if (!pp) { ++g_drawRepushMissed; return; }

    // Before the early return, and deliberately. The descriptor set and the
    // matrix are independent: TAA_MV_NO_REPUSH is a switch for testing the
    // matrix, and having it silently also disable crash destruction would make
    // one diagnostic quietly disable an unrelated feature.
    mvRebindDestructSet(cb, pp);

    // pp->valid, not merely pp. A slot is now created by the pipeline bind as
    // well as by a push, so its existence no longer implies a matrix has been
    // written into it.
    if (off || !g_nextCmdPushConstants || !pp->valid) return;
    g_inOurPush = true;
    g_nextCmdPushConstants(cb, pp->layout, VK_SHADER_STAGE_VERTEX_BIT,
                           spvinj::pushConstantOffset(),
                           spvinj::kPushConstantBytes, pp->block);
    g_inOurPush = false;
    ++g_drawRepushes;
}

// X-PLANE DRAWS INDIRECTLY. Measured: 20243 pushes at bind time against 9
// re-pushes at draw time, so vkCmdDraw and vkCmdDrawIndexed are almost never
// called and hooking only those tested nothing at all.
// ---- IS ANYTHING ELSE WRITING PUSH CONSTANTS?
//
// The matrix arrives as zeros with everything else verified: the SPIR-V loads
// and multiplies correctly, the block is declared at the offset we push to, the
// layout carries our range, and we push immediately before every draw - 4.3
// million times. The remaining possibility is that someone else writes over it
// between our push and the shader's read.
//
// X-Plane declares no push constant ranges on the layouts we have inspected, so
// in principle it never pushes. "In principle" is what has been wrong all night,
// so this counts and reports every call that is not ours, with its range.

static VKAPI_ATTR void VKAPI_CALL TAA_CmdPushConstants(
    VkCommandBuffer cb, VkPipelineLayout layout, VkShaderStageFlags stages,
    uint32_t offset, uint32_t size, const void *values)
{
    // Counting is cheap but this hook is on X-Plane's own hot path, and its
    // question is already answered: zero foreign pushes, nothing overwrites the
    // matrix. Opt-in from here.
    static const bool watch = (getenv("TAA_MV_WATCH_PUSH") != nullptr);
    if (watch && !g_inOurPush) {
        uint64_t n = ++g_foreignPushes;
        if (offset < g_foreignLo) g_foreignLo = offset;
        if (offset + size > g_foreignHi) g_foreignHi = offset + size;
        const bool hitsUs =
            (offset < spvinj::pushConstantOffset() + spvinj::kPushConstantBytes) &&
            (spvinj::pushConstantOffset() < offset + size);
        if (n <= 4 || (n % 200000) == 0)
            trace("PUSH FOREIGN: call %llu - stages 0x%x, offset %u, size %u "
                  "(ours %u..%u)%s", (unsigned long long)n, stages, offset, size,
                  spvinj::pushConstantOffset(),
                  spvinj::pushConstantOffset() + spvinj::kPushConstantBytes,
                  hitsUs ? "  *** OVERWRITES OUR MATRIX ***" : "");
    }
    g_nextCmdPushConstants(cb, layout, stages, offset, size, values);
}

static VKAPI_ATTR void VKAPI_CALL TAA_CmdDrawIndirect(
    VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off, uint32_t cnt, uint32_t stride)
{
    mvRepushBeforeDraw(cb);
    g_nextCmdDrawIndirect(cb, buf, off, cnt, stride);
}

static VKAPI_ATTR void VKAPI_CALL TAA_CmdDrawIndexedIndirect(
    VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off, uint32_t cnt, uint32_t stride)
{
    mvRepushBeforeDraw(cb);
    g_nextCmdDrawIndexedIndirect(cb, buf, off, cnt, stride);
}

static VKAPI_ATTR void VKAPI_CALL TAA_CmdDrawIndirectCount(
    VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off,
    VkBuffer cntBuf, VkDeviceSize cntOff, uint32_t maxCnt, uint32_t stride)
{
    mvRepushBeforeDraw(cb);
    g_nextCmdDrawIndirectCount(cb, buf, off, cntBuf, cntOff, maxCnt, stride);
}

static VKAPI_ATTR void VKAPI_CALL TAA_CmdDrawIndexedIndirectCount(
    VkCommandBuffer cb, VkBuffer buf, VkDeviceSize off,
    VkBuffer cntBuf, VkDeviceSize cntOff, uint32_t maxCnt, uint32_t stride)
{
    mvRepushBeforeDraw(cb);
    g_nextCmdDrawIndexedIndirectCount(cb, buf, off, cntBuf, cntOff, maxCnt, stride);
}

static VKAPI_ATTR void VKAPI_CALL TAA_CmdDraw(
    VkCommandBuffer cb, uint32_t vc, uint32_t ic, uint32_t fv, uint32_t fi)
{
    mvRepushBeforeDraw(cb);
    g_nextCmdDraw(cb, vc, ic, fv, fi);
}

static VKAPI_ATTR void VKAPI_CALL TAA_CmdDrawIndexed(
    VkCommandBuffer cb, uint32_t ic, uint32_t inst, uint32_t fi, int32_t vo, uint32_t firstInst)
{
    mvRepushBeforeDraw(cb);
    g_nextCmdDrawIndexed(cb, ic, inst, fi, vo, firstInst);
}

static VKAPI_ATTR void VKAPI_CALL TAA_CmdBindPipeline(
    VkCommandBuffer cb, VkPipelineBindPoint bind, VkPipeline pipeline)
{
    if (!g_nextCmdBindPipeline) return;
    vram::notePipelineBind();
    // Redundancy census (see BindCache). Graphics and compute keep separate
    // slots because binding one does not disturb the other.
    if (bind == VK_PIPELINE_BIND_POINT_GRAPHICS ||
        bind == VK_PIPELINE_BIND_POINT_COMPUTE) {
        const int slot = (bind == VK_PIPELINE_BIND_POINT_COMPUTE) ? 1 : 0;
        if (g_bindCache.cb != cb) {
            g_bindCache = BindCache();
            g_bindCache.cb = cb;
        }
        g_pipeBinds.fetch_add(1, std::memory_order_relaxed);
        if (g_bindCache.pipe[slot] == pipeline && pipeline != VK_NULL_HANDLE)
            g_pipeBindsRedundant.fetch_add(1, std::memory_order_relaxed);
        g_bindCache.pipe[slot] = pipeline;
    }
    g_nextCmdBindPipeline(cb, bind, pipeline);

    // ---- BIND THE FRAGMENT TRANSFORM SET.
    //
    // ON EVERY PIPELINE BIND, not once per command buffer. Vulkan only
    // guarantees a bound descriptor set survives if subsequent binds use a
    // layout compatible at that index, and X-Plane binds with ITS layout,
    // which does not contain our set at all. Binding once would work until the
    // first time X-Plane rebound anything, which is every frame.
    //
    // The set index is looked up rather than assumed. We appended at each
    // layout's OWN setLayoutCount, so ours is index 1 on one layout and 4 on
    // another; a hardcoded index would bind our buffer over one of X-Plane's,
    // which is a corruption rather than a missing feature.
    //
    // Binding with the PIPELINE'S layout - the extended one, since we extended
    // it at creation - means sets 0..N-1 are described identically to what
    // X-Plane declared, so their bindings are not disturbed.
    // ---- NOT WHILE THE FEATURE IS DORMANT.
    //
    // MEASURED: rebinding on every graphics pipeline bind costs 3-4 fps of
    // 35-38, about 9%. 1085 pipeline binds a frame each gain a descriptor-set
    // bind, taking that traffic from ~4400 to 5498 per frame.
    //
    // That is the correct thing to do WHEN the set is being read, and pure
    // waste when it is not - and right now no shader reads it and crashActive
    // is never true. A 9% tax on a dormant feature is not defensible, so the
    // bind is gated on the feature being switched on at all.
    //
    // It cannot be gated on crashActive alone: once a patched shader
    // statically references the binding, the descriptor has to be valid at
    // draw time whether or not the branch is taken. crash.enable is the gate
    // because it also decides whether displacement is patched in.
    // ---- RECORD WHERE OUR SET LIVES. DO NOT BIND IT HERE.
    //
    // This used to bind the set on every graphics pipeline bind, and that is
    // too early to be worth anything. Vulkan disturbs the bindings for every
    // set index >= N the moment a later vkCmdBindDescriptorSets uses a layout
    // that is not compatible for set N - and X-Plane binds its own sets AFTER
    // the pipeline, with a layout declaring nothing like eight of them.
    //
    // So the set was bound 11.5 million times and unbound again before every
    // draw. The readback said exactly that and it was hard to hear: 307 patched
    // vertex modules, 11.5M binds, discard word still zero. The store is
    // branch-free, so a shader that RAN had to write somewhere; zero everywhere
    // means the writes had no descriptor to land in.
    //
    // The index is recorded on the thread-local slot instead, and the bind
    // happens immediately before the draw - the same place, and for the same
    // reason, as the push constant re-push.
    if (bind == VK_PIPELINE_BIND_POINT_GRAPHICS) {
        uint32_t ourSet = UINT32_MAX;
        VkPipelineLayout lay = VK_NULL_HANDLE;
        if (crashEnabled() && destructgpu::state().ready) {
            std::lock_guard<std::mutex> g(g_lock);
            std::map<VkPipeline, VkPipelineLayout>::iterator pi =
                g_pipelineLayoutOf.find(pipeline);
            if (pi != g_pipelineLayoutOf.end()) {
                lay = pi->second;
                std::map<VkPipelineLayout, uint32_t>::iterator li =
                    g_layoutOurSet.find(lay);
                if (li != g_layoutOurSet.end()) ourSet = li->second;
            }
        }
        // Written on EVERY graphics bind, including as UINT32_MAX. A slot is
        // recycled across command buffers, so leaving it alone when the feature
        // is off would let a stale index bind our buffer over one of X-Plane's
        // on an unrelated pipeline - corruption rather than a missing feature.
        PendingPush *dp = g_tlPushSlots.find(cb);
        if (!dp) {
            // A slot now exists for this command buffer whether or not a matrix
            // was ever pushed into it, so it must start INVALID. Without this,
            // mvRepushBeforeDraw's "no slot means nothing to push" test starts
            // succeeding on a slot recycled from another command buffer, and
            // pushes that buffer's matrix into this one.
            dp = g_tlPushSlots.obtain(cb);
            dp->cb     = cb;
            dp->layout = VK_NULL_HANDLE;
            dp->valid  = false;
            memset(dp->block, 0, sizeof(dp->block));
        }
        dp->destructSet = ourSet;
        if (ourSet != UINT32_MAX) {
            dp->layout = lay;
            if (++destructgpu::bindsIssued() % 100000 == 1)
                trace("DESTRUCT: %llu pipeline(s) bound whose layout carries the "
                      "fragment set at index %u; the set itself is bound before "
                      "each draw, because binding it here is undone by X-Plane's "
                      "own descriptor binds before the draw ever runs",
                      (unsigned long long)destructgpu::bindsIssued(), ourSet);
        }
    }

    // Remember whether the compute pipeline now bound is X-Plane's upscaler.
    if (bind == VK_PIPELINE_BIND_POINT_COMPUTE) {
        std::lock_guard<std::mutex> g(g_lock);
        g_cbFsrBound[(void*)cb] = (g_xpFsrPipelines.count(pipeline) != 0);
        // Separately: is it OUR pipeline, the one whose shader stamps the
        // sentinel? Only that dispatch is worth probing after.
        g_cbFsrOurs[(void*)cb] = (g_xpFsrOurPipelines.count(pipeline) != 0);
    }

    if (!g_spirvInject || bind != VK_PIPELINE_BIND_POINT_GRAPHICS) return;
    if (!g_nextCmdPushConstants) return;

    VkPipelineLayout layout = VK_NULL_HANDLE;
    bool isGeometry = false, inScene = false, inCockpit = false;
    {
        std::lock_guard<std::mutex> g(g_lock);

        // ---- COVERAGE ACCOUNTING FIRST, BEFORE ANY EARLY RETURN.
        //
        // This function bails out below for pipelines whose layout does not
        // carry our push constant range - i.e. exactly the pipelines that write
        // no motion vectors. Counting after those returns would tally only the
        // pipelines that ARE working and report 100% coverage forever, which is
        // the same shape of mistake as the module census: measuring the thing
        // that succeeded and inferring the thing that failed.
        {
            std::map<VkCommandBuffer, bool>::iterator st0 = g_cbInScenePass.find(cb);
            bool inScene0 = (st0 != g_cbInScenePass.end() && st0->second);
            std::map<VkPipeline, bool>::iterator gt0 = g_pipelineIsGeometry.find(pipeline);
            bool isGeom0 = (gt0 != g_pipelineIsGeometry.end() && gt0->second);
            if (inScene0 && isGeom0) {
                std::map<VkPipeline, bool>::iterator pt = g_pipelineMvPatched.find(pipeline);
                bool patched = (pt != g_pipelineMvPatched.end() && pt->second);
                std::map<VkCommandBuffer, bool>::iterator ck0 = g_cbInCockpitPass.find(cb);
                bool inCk = (ck0 != g_cbInCockpitPass.end() && ck0->second);
                if (patched) { ++g_bindScenePatched;   if (inCk) ++g_bindCockpitPatched; }
                else         { ++g_bindSceneUnpatched; if (inCk) ++g_bindCockpitUnpatched; }

                uint64_t tot = g_bindScenePatched + g_bindSceneUnpatched;
                if (tot == 20000 || (tot % 200000) == 0) {
                    uint64_t ctot = g_bindCockpitPatched + g_bindCockpitUnpatched;
                    trace("MV COVERAGE (draw-weighted): %llu of %llu geometry "
                          "binds in scene passes write motion vectors (%.1f%%); "
                          "%llu do NOT. Cockpit passes: %llu of %llu (%.1f%%). "
                          "Unpatched geometry reprojects from depth only, which "
                          "is what shimmers while the camera moves.",
                          (unsigned long long)g_bindScenePatched,
                          (unsigned long long)tot,
                          tot ? 100.0 * (double)g_bindScenePatched / (double)tot : 0.0,
                          (unsigned long long)g_bindSceneUnpatched,
                          (unsigned long long)g_bindCockpitPatched,
                          (unsigned long long)ctot,
                          ctot ? 100.0 * (double)g_bindCockpitPatched / (double)ctot : 0.0);
                }
            }
        }

        std::map<VkPipeline, VkPipelineLayout>::iterator it =
            g_pipelineLayoutOf.find(pipeline);
        if (it == g_pipelineLayoutOf.end()) return;
        layout = it->second;
        std::map<VkPipelineLayout, bool>::iterator lt = g_layoutHasOurPC.find(layout);
        if (lt == g_layoutHasOurPC.end() || !lt->second) return;

        std::map<VkPipeline, bool>::iterator gt = g_pipelineIsGeometry.find(pipeline);
        isGeometry = (gt != g_pipelineIsGeometry.end() && gt->second);
        std::map<VkCommandBuffer, bool>::iterator st = g_cbInScenePass.find(cb);
        inScene = (st != g_cbInScenePass.end() && st->second);
        std::map<VkCommandBuffer, bool>::iterator ck = g_cbInCockpitPass.find(cb);
        inCockpit = (ck != g_cbInCockpitPass.end() && ck->second);
    }

    // Column-major, 16 floats, exactly the mat4 the injected shader declares
    // with ColMajor and MatrixStride 16.
    // TAA_MV_IDENTITY: push the IDENTITY instead of the reprojection.
    //
    // A test with no prediction in it. prevClip = I * gl_Position = gl_Position,
    // so the motion vector is exactly (p/w - p/w) = 0 for every fragment, at
    // every camera speed, with no arithmetic to get wrong. Anything other than
    // a field of zeros means the shader is not reading what we push - which is
    // a completely different problem from the reprojection being wrong, and the
    // two are indistinguishable from the magnitudes alone.
    //
    // The matrix itself has already been cleared of suspicion: printed in full
    // it is a well-formed clip-to-clip reprojection, and at screen centre it
    // predicts about 10 px where 730 was measured.
    static const float kIdentity[16] = {
        1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1
    };
    // The environment variable still works, but the tuning file can flip it
    // mid-flight - which is when the question actually comes up.
    static const bool envIdentity = (getenv("TAA_MV_IDENTITY") != nullptr);
    bool useIdentity = envIdentity;

    // 16 floats of reprojection, then 4 of jitter. One push rather than two:
    // the block is contiguous and a second call would cost a command per draw
    // for eight bytes of payload.
    // ---- WHICH REPROJECTION THIS DRAW GETS.
    //
    // `reproj` is a WORLD-frame matrix: it answers "where was this point last
    // frame, given it did not move in the world". For the cockpit that answer
    // is wrong by most of a screen. Panel surfaces sit about 0.7 m from the eye
    // and travel WITH the camera, so a few metres of aircraft motion implies
    // enormous parallax while their true screen motion is nearly zero. Measured
    // previously at 451 px/frame peak, with the panel the largest contributor.
    //
    // `bodyReproj` is the same matrix in the aircraft's body frame, where a
    // bolted-down panel has constant coordinates and therefore no motion. The
    // plugin already computes and publishes it, and the depth-derived compute
    // path already uses it - the injected shaders never did, so every cockpit
    // pixel drawn by X-Plane's own geometry carried world-frame motion.
    //
    // Chosen per BIND rather than per pixel, because a push constant is one
    // matrix and the shader has no depth to branch on at vertex time. The
    // discriminator is the pass: X-Plane draws the 3D cockpit in its own
    // render pass, and g_cbInCockpitPass tracks it the same way the scene pass
    // is tracked.
    //
    // Gated on bodyReprojValid, which the plugin clears whenever the camera is
    // NOT rigid with the airframe - an external view, or a head that has moved.
    // Using the body frame then would be as wrong as the world frame is now.
    // TAA_MV_NO_BODY forces the world matrix everywhere.
    //
    // The body-frame path rests on a premise the SELF-TEST deliberately breaks:
    // that the camera is rigid with the airframe. The test rotates the camera
    // INSIDE the cockpit, so while it runs the panel is not stationary relative
    // to the eye and a body-frame reprojection is the wrong answer for it - by
    // most of a screen. The measured field is bimodal, clustering near the
    // correct value and near twenty-odd times it, which is what a frame split
    // between two populations does to a median. This switch decides whether the
    // cockpit is that second population.
    static const bool noBody = (getenv("TAA_MV_NO_BODY") != nullptr);
    // ---- THE WHOLESALE BODY SWAP IS GONE. IT ZEROED THE WORLD.
    //
    // This replaced member 0 - the matrix EVERY vertex reprojects through -
    // with the body matrix whenever the view was the 3-D cockpit. The body
    // matrix is identity while the camera is rigid to the airframe, which is
    // right for the panel and catastrophic for everything else: the trace
    // caught it exactly, "camera moved 60.4628 m between these two frames"
    // beside "MV PUSHED: [0]=1.00000 [5]=1.00000 (a still camera wants
    // 1,0,0,0 / 0,1,0,0)". An identity reprojection means zero velocity, so
    // the world got no vectors at all while it swept past the window.
    //
    // Downstream that is total history rejection under motion - the weight map
    // goes fully red the moment the camera moves, flat interiors worst because
    // their sigma is smallest - and it is why no amount of gain, clamp width,
    // reactive-mask or velocity-sign tuning moved the shake: all of them sit
    // downstream of a field that was zero.
    //
    // The per-vertex select added with member 2 is what this was reaching for:
    // world reprojection for the world, body reprojection for near geometry,
    // chosen per vertex on depth rather than per frame on view type. So member
    // 0 is now always the world matrix. TAA_MV_BODY_WHOLESALE re-arms the old
    // behaviour for comparison.
    static const bool bodyWholesale = getenv("TAA_MV_BODY_WHOLESALE") != nullptr;
    bool useBody = bodyWholesale && inCockpit && g_velSnap.bodyReprojValid &&
                   !useIdentity && !noBody;

    // PER-PASS CENSUS. Which qualifying pass actually draws the world is a
    // measurement; attachment counts and submission order have both already
    // been guessed at and both were wrong.
    // ---- A MISSING BRACE WAS WRITING OUT OF BOUNDS ON EVERY NON-SCENE DRAW.
    //
    // Only the first increment was guarded. The second ran unconditionally,
    // and g_mvPassOrdinal is -1 for every pass that is not a qualifying scene
    // pass - so this was ++g_mvPassDrawsFrame[-1], a write to whatever global
    // precedes that array, thousands of times a frame. Undefined behaviour in
    // the recording hot path, and the kind of corruption that shows up as
    // neighbouring state going wrong intermittently rather than as a crash.
    if (g_mvPassOrdinal >= 0 && g_mvPassOrdinal < 16 && isGeometry) {
        ++g_mvPassDraws[g_mvPassOrdinal];
        ++g_mvPassDrawsFrame[g_mvPassOrdinal];
    }

    // TAA_MV_TESTYAW is gone. It pushed a synthetic clip-to-clip rotation to
    // separate "the matrix is wrong" from "the shader is wrong", and both of
    // those are now answered by the epipolar residual - 0.000 to 0.003 px per
    // frame against the real matrix. It also had its own sx inverted, so it
    // would have mislabelled the very fault it existed to find.
    //
    // Removed rather than defaulted off: it REPLACED the pushed matrix, so a
    // stray environment variable would have made every motion vector describe
    // a rotation the camera never performed, and no magnitude test would have
    // noticed.
    float block[36];
    // ---- THE BODY MATRIX IS STILL CLIP-TO-CLIP. CONVERT IT.
    //
    // When the main reprojection moved to view space the shader changed what it
    // feeds the matrix: it now builds (pos.x, pos.y, pos.w, 1) and the matrix
    // absorbs 1/sx, 1/sy and the -1. bodyReproj was not converted with it. It
    // is still (prevProj * prevWorldRel * Bp) * (proj * worldRel * Bc)^-1, a
    // clip-to-clip matrix, and it was being handed a vector in the other
    // convention.
    //
    // That is the band across the bottom of the screen: a full-width region
    // with vx exactly zero and vy growing linearly from the horizon, present
    // while the aircraft is frozen - local_x/y/z move 0.0000 m per frame at a
    // constant 337.69 m. Its ratio prevNDC.y / currNDC.y measured 0.6549, dead
    // constant over 28,536 pixels and seven consecutive frames. A constant
    // scale, on one axis, over one coherent group of draws - which is what
    // feeding one convention's vector to the other convention's matrix does.
    // Cockpit geometry sits below the horizon, so only the lower half showed
    // it; the sky above writes no velocity at all.
    //
    // The conversion is exact. With m10 = m11 = -1 the projection gives
    // z_clip = w_clip + m14, so the vector the shader passes expands to full
    // clip space through
    //
    //     K: (x, y, w, 1) -> (x, y, w + m14, w)
    //
    // and pushing bodyReproj * K puts the body path back on the same footing.
    // K's columns are (1,0,0,0), (0,1,0,0), (0,0,1,1), (0,0,m14,0) - the same
    // matrix that proj * clipToView reduces to, which is the cross-check that
    // the two paths now agree.
    if (useBody) {
        float K[16];
        memset(K, 0, sizeof(K));
        K[0] = 1.0f; K[5] = 1.0f;
        K[10] = 1.0f; K[11] = 1.0f;
        K[14] = g_velSnap.proj[14];
        float bodyView[16];
        taaMul(bodyView, g_velSnap.bodyReproj, K);
        memcpy(block, bodyView, 64);
    } else
    memcpy(block, useIdentity ? kIdentity : g_velSnap.reproj, 64);
    if (useIdentity) memcpy(block, kIdentity, 64);
    if (useBody) ++g_bodyReprojPushes;
    block[16] = block[17] = block[18] = block[19] = 0.0f;

    // ---- MEMBER 2: THE BODY MATRIX, FOR THE NEAR-FIELD SELECT.
    //
    // The block already carried the body reprojection, but only as a per-draw
    // REPLACEMENT for member 0 (useBody above). That cannot serve the
    // near-field select, which is per-VERTEX: one draw can contain both panel
    // and windscreen, and the choice has to be made per vertex on depth.
    //
    // So it is sent alongside as well, always, and the shader picks between
    // world and body per vertex. Same K conversion as the useBody path - the
    // shader feeds every matrix (x, y, w, 1), so an unconverted clip-to-clip
    // matrix produces the constant-scale band that note describes, and both
    // paths must agree or the two halves of the frame disagree.
    {
        float K[16];
        memset(K, 0, sizeof(K));
        K[0] = 1.0f; K[5] = 1.0f;
        K[10] = 1.0f; K[11] = 1.0f;
        K[14] = g_velSnap.proj[14];
        float bodyView[16];
        if (g_velSnap.bodyReprojValid)
            taaMul(bodyView, g_velSnap.bodyReproj, K);
        else
        {
            // ---- IDENTITY, NOT THE WORLD MATRIX. THIS IS THE COCKPIT SHAKE.
            //
            // "Fall back to the world matrix" sounds conservative and is the
            // bug. The near-field select exists to give cockpit geometry -
            // which is rigidly attached to the camera and therefore does NOT
            // move on screen - a reprojection that says "stationary". Filling
            // the body slot with the WORLD matrix makes the select a no-op:
            // both branches carry world motion, so the panel and yoke are told
            // they moved by the full camera delta, the resolve drags history
            // across them, and the cockpit shakes.
            //
            // bodyReprojValid is 0 for a long time by design - the plugin needs
            // 120 frames of the aircraft ROTATING before it will vouch for a
            // body frame - so this fallback is not a rare edge case. It is the
            // normal state for most of a session, which is why the shake read
            // as constant rather than intermittent.
            //
            // Identity is the correct answer: no valid body frame means the
            // best available statement about near-field geometry is that it did
            // not move relative to the camera, which is exactly what identity
            // pushes. It is also what the world matrix degenerates to when the
            // camera is still, so this changes nothing in the case the old code
            // was right about, and fixes every case it was wrong about.
            memset(bodyView, 0, sizeof(bodyView));
            bodyView[0] = bodyView[5] = bodyView[10] = bodyView[15] = 1.0f;
        }
        memcpy(block + 20, bodyView, 64);
    }
    // Recorded for the diagnostic. block[18] is the near-field threshold the
    // shader compares gl_Position.w against, and it is the one push value never
    // actually observed at runtime - the code path says zero, which is not the
    // same as having seen zero.
    g_diagLastBlock18 = block[18];
    g_diagBodyValid   = g_velSnap.bodyReprojValid;

    // ---- WHAT WE ACTUALLY PUSH.
    //
    // Every input is now verified: the projection is centred ([8] and [9]
    // exactly zero), the camera position matches XPLMReadCameraPosition to
    // sub-millimetre in both views, the pairing is right (a one-frame lag made
    // the good case worse and the bad case identical), prevProj equals proj,
    // and both patched shaders are correct in their disassembly. An identity
    // matrix produces an exactly zero field.
    //
    // Yet a STATIC camera in an external view produces 339 to 681 px. With no
    // camera motion this matrix must leave x and y alone - its top-left 2x2
    // should be the identity and its last column zero in x and y. If it is not,
    // the matrix is wrong despite correct inputs; if it is, the matrix is right
    // and something between here and the shader changes it.
    {
        // Every 600 PUSHES was the wrong gate: there are about 14,000 geometry
        // binds per frame, so all of it landed in the first two frames - on the
        // loading screen, before the plugin had published anything - and read as
        // an all-zero matrix. That is startup state, not a defect.
        //
        // Gated on the snapshot being live, and spaced far enough apart to be
        // sampling steady flight.
        static uint64_t nlog = 0;
        // During the failing case specifically: an external view with the
        // camera actually moving. A static camera in the same view measures
        // 0.000 px, so a matrix sampled there confirms only the case that
        // already worked - which is what the first correct reading did.
        // ---- IS THE MATRIX EVEN VALID? NOTHING HERE EVER ASKED.
        //
        // The plugin sets reprojValid = 0 when it cannot invert the current
        // view-projection, and in that case it LEAVES s->reproj untouched - so
        // the field keeps whatever it last held, which is the identity it was
        // initialised with if the inverse has never succeeded. This layer never
        // read the flag (zero references before this), so an identity was
        // pushed as though it were a real reprojection: velocity zero for the
        // whole world while the camera moves, total history rejection, and the
        // fully red weight map. Say it out loud, once, when it happens.
        {
            // COUNT, do not merely warn. The first version of this was a
            // one-shot bool, which cannot tell "failed once while loading,
            // when the projection is legitimately still zero" from "failing
            // every frame" - and those want opposite fixes. A rate is the only
            // form of this measurement worth having.
            static uint64_t nSeen = 0, nBad = 0;
            if (g_share) {
                ++nSeen;
                if (!g_share->reprojValid) ++nBad;
                if ((nSeen % 600) == 0)
                    trace("MV REPROJ VALIDITY: %llu of %llu pushes had NO valid "
                          "reprojection (%.1f%%). Those frames carry a stale or "
                          "identity matrix, so every vector in them is zero. "
                          "A rate near 0 means the inverse is healthy and the "
                          "resolve duty gap is cadence, not failure.",
                          (unsigned long long)nBad, (unsigned long long)nSeen,
                          100.0 * (double)nBad / (double)nSeen);
            }
        }
        const bool failing = (g_velSnap.viewType != 0 && g_velSnap.viewType != 1026);
        if (failing && (++nlog % 20000) == 1)
            trace("MV PUSHED: view=%d | rows 0/1 of the pushed matrix: "
                  "[0]=%.5f [4]=%.5f [8]=%.5f [12]=%.5f | [1]=%.5f [5]=%.5f "
                  "[9]=%.5f [13]=%.5f | w row [3]=%.5f [7]=%.5f [11]=%.5f "
                  "[15]=%.5f (a still camera wants 1,0,0,0 / 0,1,0,0)",
                  g_velSnap.viewType,
                  block[0], block[4], block[8],  block[12],
                  block[1], block[5], block[9],  block[13],
                  block[3], block[7], block[11], block[15]);
    }

    // ---- JITTER, converted from framebuffer pixels to a clip-space offset.
    //
    // The shader adds jitter.xy * w to gl_Position, so this has to be in NDC:
    // NDC spans 2 units across the screen while the viewport spans W pixels,
    // hence the factor of two.
    //
    // THE Y SIGN IS TAKEN FROM THE VIEWPORT, not assumed. X-Plane sets a
    // negative-height viewport (GL orientation via maintenance1), which makes
    // clip Y point UP while framebuffer Y points DOWN - so moving the image
    // down by jy pixels means DECREASING clip y. Reading the sign from the
    // viewport we actually saw rather than hardcoding it means a future build
    // that switches to Vulkan orientation flips with it instead of silently
    // jittering the wrong way, which is a bug that looks like shimmer rather
    // than like an error.
    //
    // Gated twice. Only inside a scene pass, and only on pipelines that draw
    // from vertex buffers - everything else gets zero, which is the point of
    // pushing this per draw instead of setting a viewport once.

    // Which of the four conditions is failing, counted rather than guessed.
    // "0 draws offset" says the result; it does not say which gate closed, and
    // there are four of them.
    static uint64_t nBind = 0, nScene = 0, nGeom = 0, nBoth = 0;
    ++nBind;
    if (inScene) ++nScene;
    if (isGeometry) ++nGeom;
    if (inScene && isGeometry) ++nBoth;
    if ((nBind % 200000) == 0)
        trace("JITTER GATE: %llu binds, inScene=%llu isGeometry=%llu both=%llu, "
              "armed=%d",
              (unsigned long long)nBind, (unsigned long long)nScene,
              (unsigned long long)nGeom, (unsigned long long)nBoth,
              g_jitterArmed ? 1 : 0);

    if (g_jitterArmed && inScene && isGeometry) {
        // The RENDER resolution, not the window. Converting a pixel offset to
        // NDC against the display size would scale the jitter by 0.769 when
        // X-Plane's FSR is on - a wrong jitter rather than none, which is
        // harder to notice.
        float w = (float)(g_renderW ? g_renderW : (uint32_t)g_velSnap.viewportW);
        float h = (float)(g_renderH ? g_renderH : (uint32_t)g_velSnap.viewportH);
        if (w > 0.0f && h > 0.0f) {
            float ySign = g_viewportYFlipped ? -1.0f : 1.0f;
            // g_jitterScale is the live amplitude knob - see fsr2_pass.h. FSR2
            // is told the same scaled value, so the two cannot drift apart.
            // ---- JITTER IS PHASE TWO. AMPLITUDE IS ZERO.
            //
            // The vectors are the product right now, and they are measured
            // against an unjittered render - which removes a whole class of
            // sign and amplitude bugs from the calibration. The injection path
            // stays so it can be switched on once the vectors read a ratio of
            // one, but it contributes nothing until then.
            // ---- AMPLITUDE. Phase two starts here.
            //
            // Held at zero while the vectors were being calibrated, so the
            // field was measured against an unjittered render and a whole class
            // of sign and amplitude bugs could not contaminate it. The vectors
            // now pass acceptance 30 of 32, so the reason for holding it is
            // gone.
            //
            // TAA_JITTER_SCALE sets it. 1.0 is a full Halton(2,3) offset of
            // +-0.5 px; the plugin already centres the sequence on the pixel.
            // The default stays 0 because jitter with NOTHING CONSUMING IT is
            // strictly worse than none - it shifts the sample grid every frame
            // and, with no accumulation, high-contrast edges crawl. It is armed
            // for measurement, not for use, until a resolve exists to cancel it.
            // ---- THE CONDITION THIS COMMENT NAMES IS NOW MET.
            //
            // "It is armed for measurement, not for use, until a resolve exists
            // to cancel it." A resolve exists. So the default follows the
            // resolve rather than staying at zero: jitter with nothing
            // consuming it is strictly worse than none, and jitter WITH a
            // consumer is the entire mechanism by which temporal
            // anti-aliasing gets samples the single frame did not have.
            //
            // Still overridable, and still zero when the resolve is off - the
            // two are tied together because either alone is a downgrade.
            // The comment above said "the default follows the resolve" and
            // the code said 0.0f - the prose was updated when the resolve was
            // built and the constant was not, so the amplitude stayed zero
            // through every session and the resolve averaged identical
            // samples. The value now lives in g_jitterScale, computed once per
            // frame from the live enable, default 1.0 while resolving.
            block[16] =  2.0f * g_velSnap.jitterX * g_jitterScale / w;
            block[17] = ySign * 2.0f * g_velSnap.jitterY * g_jitterScale / h;
            g_appliedJitX = block[16];
            g_appliedJitY = block[17];
            if (block[16] != 0.0f || block[17] != 0.0f) ++g_jitterApplied;
        }
    }

    // ---- NEAR-FIELD DISTANCE, in metres, into the spare .z of the jitter vec4.
    //
    // The patched vertex shader uses this to decide, per vertex, whether to
    // reproject through uReproj (the world) or to reuse the current clip
    // position (velocity zero). See the long note in spirv_inject.h: for
    // geometry bolted to the airframe, viewed from a camera bolted to the
    // airframe, the previous clip position IS the current one.
    //
    // GATED ON bodyReprojValid, which is the plugin's statement that it has
    // measured the camera to be rigid in the body frame - 0.0022 m/frame over
    // 120 rotating cockpit frames. In an external view, or before that has
    // resolved, the premise does not hold: the camera moves independently, the
    // panel is not stationary relative to it, and zeroing its velocity would be
    // a new bug rather than a fix. Zero disables the select entirely, because
    // gl_Position.w is positive for anything in front of the camera.
    // ---- PROVISIONAL ARMING: THE CALIBRATION CANNOT RESOLVE WHILE PARKED.
    //
    // bodyReprojValid requires 120 frames of the aircraft ROTATING in a
    // cockpit view before the quaternion-mapping census is decisive - read the
    // gate: `rigid && rotating && !external`, then `samples >= 120`. A parked
    // aircraft never rotates, so every parked session ever flown had
    // bodyValid=0 and the near-field select disarmed, which is the cockpit
    // shake in one sentence.
    //
    // viewType 1018 is X-Plane's 3-D cockpit: the camera rides the airframe by
    // construction there, which is the exact premise the calibration exists to
    // verify for exotic cameras. So the select arms provisionally in 1018, and
    // the calibrated path simply confirms it once the aircraft has flown. The
    // failure mode of a wrong provisional arming is zeroed velocity on
    // geometry within two metres of the eye in a cockpit view - which is the
    // panel, whose correct velocity in that view IS zero.
    // ---- 1018 IS NOT THE 3-D COCKPIT. 1026 IS.
    //
    // The provisional arming above was written against the wrong constant, and
    // this same file contradicts it eleven lines of code earlier: the MV PUSHED
    // diagnostic treats viewType 1026 as the normal, non-failing view, because
    // 1026 is where the sim actually sits in a cockpit. X-Plane's view_type
    // numbering is
    //
    //     1000  forward with 2-D panel      1018  forward with HUD
    //     1017  forward with nothing        1026  3-D COCKPIT
    //
    // so arming on 1018 alone armed the select in a view nobody flies in, and
    // left block[18] at zero in the one view the select exists for. The
    // threshold being zero means no vertex ever passes `w < block[18]`, so the
    // body matrix is never selected no matter what it contains - which is why
    // restoring the identity fallback changed nothing on its own. Two separate
    // faults were sitting in series on the same feature.
    //
    // All four of those views are mounted on the airframe: the camera rides the
    // aeroplane by construction, which is the premise the body calibration
    // exists to verify for exotic cameras. The external views - 1003 tower,
    // 1004 ride-along, 1005 track, 1006 free, 1007 chase - are not, and must
    // keep waiting for the calibrated path.
    //
    //   taa.nearfield_view = -1  AUTO: any airframe-mounted view (default)
    //                         0  never arm provisionally (calibration only)
    //                     >1000  arm only in exactly this view id
    const int nfView = live::i("taa.nearfield_view", "TAA_NEARFIELD_VIEW", -1);
    const int vt     = g_velSnap.viewType;
    // ---- 1018 IS EXTERNAL. THE SET ABOVE WAS GUESSED.
    //
    // This armed on {1000, 1017, 1018, 1026} as "airframe-mounted views". Only
    // 1026 was ever evidenced. The rest were inferred from X-Plane's view_type
    // numbering as I believed it to be, and 1018 in particular was called
    // "forward with HUD" - it is EXTERNAL. learnings.md recorded that months
    // ago, from the residual measurement that separated the two:
    //
    //     view 1026 (cockpit)   median residual 0.000 - 0.005 px
    //     view 1018 (external)  median residual 56 - 681 px
    //
    // and a screenshot of the bench captures settles it visually: the aircraft
    // is seen from outside while the trace reports 1018.
    //
    // Arming provisionally in an EXTERNAL view is wrong on its own premise.
    // The provisional arming exists because in a cockpit view the camera rides
    // the airframe BY CONSTRUCTION, which is the thing the body calibration
    // would otherwise have to prove. In an external view the camera does not
    // ride anything, so near geometry - a hangar the camera drifts past, the
    // ground on a low fly-by - would be handed the identity matrix and told it
    // did not move. Rare, because little comes within nearfield_m of an
    // external camera, but wrong whenever it happens.
    //
    // So the set is now only what is evidenced. 1026. Anything else waits for
    // bodyReprojValid, which is the calibrated path and needs no guessing.
    const bool ridesAirframe = (vt == 1026);
    const bool provisional = (nfView < 0)      ? ridesAirframe
                           : (nfView == 0)     ? false
                                               : (vt == nfView);
    if ((g_velSnap.bodyReprojValid || provisional) && g_nearFieldM > 0.0f)
        block[18] = g_nearFieldM;

    // ---- block[19]: MAY THIS PASS BE DISPLACED?
    //
    // The crash displacement is clip' = clip + M*d, and M is the MAIN CAMERA's
    // aircraft-local-to-clip. That equation is only true in a pass built from
    // the same projection. The layer patches EVERY vertex shader - shadow,
    // reflection, cockpit near-field - so applying one camera's delta in
    // another's clip space displaced the same vertex by different amounts in
    // different passes: bent wingtips, displaced nacelles, an aeroplane that
    // appeared to shrink because M*d carries a w component and a changed w
    // changes the perspective divide.
    //
    // inScene && isGeometry is the same test the jitter already uses, for the
    // same reason: it is the set of passes the main camera's matrices describe.
    // Everything else keeps the position its own pass computed.
    // ---- WHICH PASSES MAY BE DISPLACED, CHOSEN AT RUN TIME.
    //
    // inScene && isGeometry matches the jitter exactly - measured, 184908
    // against 184907 - and the gear, engine pylons and tail still do not move
    // while the fuselage and wings do. So the lagging parts are among the
    // 15093 draws this excludes, and the question is WHICH exclusion is
    // wrong: 11864 draws are isGeometry without inScene, 2022 the reverse.
    //
    // Switchable rather than guessed, because guessing at this has cost five
    // rebuilds tonight:
    //
    //   0  inScene && isGeometry   the jitter's test, today's behaviour
    //   1  isGeometry              drops the scene requirement
    //   2  inScene                 drops the geometry requirement
    //   3  everything              proves the gate is the cause, or is not
    //
    // 3 will displace full-screen quads and shadow passes - it is a diagnosis,
    // not a setting.
    {
        const int mode = live::i("crash.gate", "TAA_CRASH_GATE", 0);
        const bool pass = (mode == 1) ? isGeometry
                        : (mode == 2) ? inScene
                        : (mode == 3) ? true
                                      : (inScene && isGeometry);
        block[19] = pass ? 1.0f : 0.0f;
    }

    // ---- HOW MANY DRAWS ACTUALLY CARRY THE GATE.
    //
    // The tail lags behind the rest of the aeroplane, and there are only two
    // ways that happens: its vertices are outside the grid, or its draws have
    // this flag clear. Counting both sides distinguishes them without another
    // guess - a tail drawn in a pass that is inScene but not isGeometry, or
    // neither, shows up here as a large "off" count.
    {
        static uint64_t nOn = 0, nOff = 0;
        if (block[19] != 0.0f) ++nOn; else ++nOff;
        if (((nOn + nOff) % 200000) == 1)
            trace("DESTRUCT: displace gate - %llu draws ON, %llu OFF "
                  "(inScene && isGeometry). Geometry drawn with it OFF keeps "
                  "the position its own pass computed.",
                  (unsigned long long)nOn, (unsigned long long)nOff);
    }

    // Never silent again. The select being disarmed is invisible from every
    // downstream measurement - the field simply carries world motion on the
    // panel, which looks like a correct field of a moving world.
    {
        static int lastVt = -1;
        static float lastThr = -1.0f;
        if (vt != lastVt || block[18] != lastThr) {
            lastVt = vt; lastThr = block[18];
            trace("NEAR FIELD SELECT: view=%d bodyValid=%d -> threshold %.2f m "
                  "(0 means DISARMED: every vertex takes the world matrix and "
                  "cockpit geometry is told it moved with the camera)",
                  vt, (int)g_velSnap.bodyReprojValid, block[18]);
        }
    }
    g_diagLastBlock18 = block[18];


    // ---- HOW MANY DIFFERENT MATRICES ONE FRAME PUSHES.
    //
    // The identity test proved the field is built from the pushed matrix for
    // 100%% of pixels, and the stashed matrix always encodes exactly one
    // self-test step - yet some frames measure twenty-six. The remaining way
    // both can be true is that the matrix CHANGED DURING THE FRAME, so the
    // draws and the stash saw different ones.
    //
    // g_velSnap is rewritten in QueuePresentKHR while X-Plane records command
    // buffers on several threads, so a recording thread can straddle a present
    // and pick up the next frame's matrix. Counting distinct pushes per frame
    // says whether that actually happens rather than assuming it does.
    {
        static bool  haveLast = false;
        if (!haveLast || memcmp(g_lastPushed, block, 64) != 0) {
            memcpy(g_lastPushed, block, 64);
            haveLast = true;
            uint32_t n = ++g_pushDistinctThisFrame;
            if (n > g_pushDistinctMax) {
                g_pushDistinctMax = n;
                trace("MV PUSH RACE: %u distinct matrices pushed within one frame "
                      "- more than 1 means the draws in a frame did not all see "
                      "the same camera", n);
            }
        }
    }

    g_inOurPush = true;
    g_nextCmdPushConstants(cb, layout, VK_SHADER_STAGE_VERTEX_BIT,
                           spvinj::pushConstantOffset(),
                           spvinj::kPushConstantBytes,
                           block);
    g_inOurPush = false;
    PendingPush *pp = g_tlPushSlots.obtain(cb);
    pp->cb     = cb;
    pp->layout = layout;
    memcpy(pp->block, block, sizeof(pp->block));
    pp->valid  = true;
    g_tlPush.cb     = cb;
    g_tlPush.layout = layout;
    memcpy(g_tlPush.block, block, sizeof(g_tlPush.block));
    g_tlPush.valid  = true;

    // ---- BRACES. THE SECOND TRACE WAS NOT INSIDE THE IF.
    //
    // Both lines were meant to fire on the first three pushes and then every
    // hundred thousandth. Only the first did. The second ran on EVERY push, and
    // the last acceptance run left a 3.4 GB trace containing 19,099,210 copies
    // of it - a formatted write per draw, per frame, on the render path.
    //
    // The indentation said what was intended and the compiler did something
    // else, which is the same shape as every other unbraced body that has cost
    // this project time.
    if (++g_pushCount <= 3 || (g_pushCount % 100000) == 0) {
        trace("SPIRV INJECT: draw-time re-pushes %llu", (unsigned long long)g_drawRepushes);
        trace("SPIRV INJECT: pushed uReproj + jitter (%llu times), body-frame "
              "%llu, bodyValid=%d cockpitPass=%d | jitter now (%.5f %.5f) ndc, "
              "pipelines: %llu geometry / %llu full-screen",
              (unsigned long long)g_pushCount,
              (unsigned long long)g_bodyReprojPushes,
              g_velSnap.bodyReprojValid, g_cockpitPassIndex,
              block[16], block[17],
              (unsigned long long)g_pipeGeometry,
              (unsigned long long)g_pipeFullscreen);
    }

    // ---- PUBLISH THE QUALITY FIGURES WHERE A USER CAN SEE THEM.
    //
    // These live in the shared block so the plugin can turn them into datarefs
    // and the panel can show them. Cheap: two integer stores against a counter
    // that already exists.
    if (g_share && g_share->magic == TAA_MAGIC) {
        g_share->mvPipelinesPatched  = (uint32_t)g_pipeGeometry;
        g_share->mvPipelinesRejected = (uint32_t)g_pipeRejected;
    }
}

// ------------------------------------------------------- VRAM overcommit
//
// Recreates what an OpenGL driver used to do for you: when a device-local
// allocation cannot be satisfied, put the resource in system RAM and let it
// stream over PCIe instead of failing.
//
// This is why the texture pager exists at all. Under OpenGL the driver
// virtualised VRAM, so an application could allocate past the card's capacity
// and merely get slower. Vulkan removed that - allocations are explicit and a
// failure is a failure - so X-Plane had to write a pager, and its conservative
// budget is the consequence we are looking at.
//
// Failing SOFT is the whole point. Raising X-Plane's budget makes it believe it
// has room it may not have, and the allocation failure then lands somewhere it
// cannot recover from - which is the overrun that -gfx-no-pager already caused.
// A fallback means the worst case is a slow texture.
//
// HONEST CAVEAT, and the reason this logs before it acts: on Windows, WDDM
// already demotes device-local allocations to system memory under pressure, so
// vkAllocateMemory may simply never fail. If the fallback counter stays at zero
// while the pager is still cutting textures, then overcommit is not the lever
// and the stored budget is - and knowing that is worth more than the hook.
//
// State for this lives with the other VRAM globals near the top, because the
// present hook reads g_overcommit long before this point in the file.

static VKAPI_ATTR VkResult VKAPI_CALL TAA_AllocateMemory(
    VkDevice device, const VkMemoryAllocateInfo *ai,
    const VkAllocationCallbacks *alloc, VkDeviceMemory *out)
{
    if (!g_nextAllocateMemory) return VK_ERROR_INITIALIZATION_FAILED;

    // ---- RECYCLE FIRST. A pooled identical block answers without a driver
    // call at all - the plan's deferred-free retire queue paying out. Contents
    // of a fresh allocation are undefined by spec, so handing back a used
    // block is legal; only never-mapped device-local blocks enter the pool.
    if (ai && out && vram::poolTake(ai, out)) {
        vram::noteAlloc(*out, ai);
        ++g_allocCount;
        g_allocBytes += ai->allocationSize;
        return VK_SUCCESS;
    }

    // ---- PRIORITY TAG. Neutral 0.5 at allocation; the first bind names what
    // the block holds and raises or lowers it. Ignored by the driver unless
    // the memoryPriority feature was enabled at device creation, which the
    // CreateDevice hook takes care of.
    vram::PrioChain pc;
    const VkMemoryAllocateInfo *use = ai;
    if (ai && vram::prioTag(ai, &pc)) use = &pc.ai;

    LARGE_INTEGER t0, t1, fq;
    QueryPerformanceCounter(&t0);
    VkResult r = g_nextAllocateMemory(device, use, alloc, out);
    QueryPerformanceCounter(&t1);
    QueryPerformanceFrequency(&fq);
    if (fq.QuadPart > 0) {
        uint64_t us = (uint64_t)((t1.QuadPart - t0.QuadPart) * 1000000ll /
                                 fq.QuadPart);
        uint64_t worst = vram::allocLatWorstUs.load();
        while (us > worst &&
               !vram::allocLatWorstUs.compare_exchange_weak(worst, us)) {}
    }

    if (r == VK_SUCCESS && ai && out) vram::noteAlloc(*out, ai);

    if (ai) {
        ++g_allocCount;
        if (r == VK_SUCCESS) g_allocBytes += ai->allocationSize;
        if (g_allocCount % 2000 == 0)
            trace("ALLOC: %llu allocations, %.2f GB live-ish, %llu failures, "
                  "%llu rescued to host memory",
                  (unsigned long long)g_allocCount, g_allocBytes / 1073741824.0,
                  (unsigned long long)g_allocFailed,
                  (unsigned long long)g_allocRescued);
    }

    if (r == VK_SUCCESS || !ai) return r;
    if (r != VK_ERROR_OUT_OF_DEVICE_MEMORY) return r;

    ++g_allocFailed;
    // Never hide an allocation failure (SS69): size, type, its property
    // flags, and the budget picture at the moment it happened.
    trace("ALLOC: OUT OF DEVICE MEMORY for %.1f MB (type %u: %s) - usage "
          "%.2f GB of %.2f GB raw budget, %.2f GB shaped, ledger %.2f GB. "
          "This is the moment X-Plane's pager is trying to avoid.",
          ai->allocationSize / 1048576.0, ai->memoryTypeIndex,
          vram::typeFlagsText(ai->memoryTypeIndex),
          vram::rawUsage / 1073741824.0, vram::rawBudget / 1073741824.0,
          vram::lastReported / 1073741824.0,
          g_vramTotalBytes / 1073741824.0);

    // ---- EMERGENCY LADDER, before overcommit. Flush the recycle pool (real
    // bytes returned to the driver), release every held upload, deflate the
    // shaped budget so the engine's own pager evicts on its next evaluate -
    // then retry the allocation once.
    if (vram::emergency()) {
        r = g_nextAllocateMemory(device, use, alloc, out);
        if (r == VK_SUCCESS) {
            vram::noteAlloc(*out, ai);
            trace("ALLOC: emergency reclaim rescued the %.1f MB allocation on "
                  "retry", ai->allocationSize / 1048576.0);
            return r;
        }
    }

    if (!g_overcommit) return r;

    // Retry from a host-visible heap.
    //
    // The catch, and it may make this impossible for images: the allowed
    // memory types come from the RESOURCE, via memoryTypeBits, and
    // vkAllocateMemory is not told which resource this is for. Drivers commonly
    // permit only device-local types for optimally-tiled images, in which case
    // no host-visible type is legal and this cannot work for textures - only
    // for buffers. The attempt is logged either way so the answer is recorded
    // rather than assumed.
    VkPhysicalDeviceMemoryProperties mp;
    memset(&mp, 0, sizeof(mp));
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, DeviceData>::iterator it = g_devices.find(dispatchKey(device));
        if (it == g_devices.end() || !g_getPhysMemProps) return r;
        g_getPhysMemProps(it->second.phys, &mp);
    }

    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if (!(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
        if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)    continue;   // same heap
        if (i == ai->memoryTypeIndex) continue;

        VkMemoryAllocateInfo ai2 = *ai;
        ai2.memoryTypeIndex = i;
        VkResult r2 = g_nextAllocateMemory(device, &ai2, alloc, out);
        if (r2 == VK_SUCCESS) {
            vram::noteAlloc(*out, &ai2);
            ++g_allocRescued;
            trace("ALLOC: rescued %.1f MB into host memory type %u - it will "
                  "stream over PCIe rather than fail",
                  ai->allocationSize / 1048576.0, i);
            return VK_SUCCESS;
        }
    }

    trace("ALLOC: no host-visible type accepted the allocation. For optimally "
          "tiled images the driver usually permits device-local types only, so "
          "overcommit cannot help textures on this driver.");
    return r;
}

extern "C" VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL TAA_CreateDevice(
    VkPhysicalDevice phys, const VkDeviceCreateInfo *ci,
    const VkAllocationCallbacks *alloc, VkDevice *out)
{
    VkLayerDeviceCreateInfo *link = findDeviceLink(ci);
    if (!link || !link->u.pLayerInfo) return VK_ERROR_INITIALIZATION_FAILED;

    PFN_vkGetInstanceProcAddr nextGIPA = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr   nextGDPA = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;

    PFN_vkCreateDevice nextCreate = (PFN_vkCreateDevice)nextGIPA(nullptr, "vkCreateDevice");
    if (!nextCreate) return VK_ERROR_INITIALIZATION_FAILED;

    // ---- ADD DEVICE EXTENSIONS. This is the only moment it is possible.
    //
    // A device's extension set is fixed at creation. Anything needed later -
    // NVIDIA's optical flow engine, and DLSS when it arrives - has to be asked
    // for HERE, before the device exists, or it is simply unavailable for the
    // lifetime of the process and there is no second chance.
    //
    // Adding an extension the driver does not support makes vkCreateDevice fail
    // outright, which would stop X-Plane starting at all. So each one is
    // checked against the driver's list first, and anything missing is silently
    // left out rather than risking the sim over a feature we can do without.
    // ---- X-PLANE'S OWN REQUEST, DUMPED VERBATIM.
    //
    // NvLL_VK_InitLowLatencyDevice accepts a bare device on this machine and
    // refuses X-Plane's, and every property varied so far - the low latency
    // extensions, our layer, Steam's overlay and fossilize layers, being the
    // second caller in a process - has left the -229 exactly where it was. What
    // has never been varied is what X-PLANE asks for, because it has never been
    // written down.
    //
    // Dumped once, in a form that can be pasted straight into a standalone
    // harness, so the bisection runs in a second per attempt instead of a sim
    // launch per attempt.
    std::vector<const char*> exts;
    for (uint32_t i = 0; i < ci->enabledExtensionCount; ++i)
        exts.push_back(ci->ppEnabledExtensionNames[i]);

    {
        trace("XP DEVICE REQUEST: %u extensions, %u queue create infos",
              ci->enabledExtensionCount, ci->queueCreateInfoCount);
        for (uint32_t i = 0; i < ci->enabledExtensionCount; ++i)
            trace("XP DEVICE EXT: %s", ci->ppEnabledExtensionNames[i]);
        g_deviceFamilies.clear();
        for (uint32_t i = 0; i < ci->queueCreateInfoCount; ++i) {
            uint32_t fam = ci->pQueueCreateInfos[i].queueFamilyIndex;
            bool dup = false;
            for (size_t k = 0; k < g_deviceFamilies.size(); ++k)
                if (g_deviceFamilies[k] == fam) dup = true;
            if (!dup) g_deviceFamilies.push_back(fam);
        }
        for (uint32_t i = 0; i < ci->queueCreateInfoCount; ++i)
            trace("XP DEVICE QUEUE: family %u count %u flags 0x%x",
                  ci->pQueueCreateInfos[i].queueFamilyIndex,
                  ci->pQueueCreateInfos[i].queueCount,
                  ci->pQueueCreateInfos[i].flags);
        // The pNext chain is where feature structs live, and a feature is at
        // least as likely a culprit as an extension name.
        for (const VkBaseInStructure *p = (const VkBaseInStructure*)ci->pNext;
             p; p = p->pNext)
            trace("XP DEVICE pNext: sType %d", (int)p->sType);
    }

    VkDeviceCreateInfo ci2 = *ci;
    {
        // What the driver actually offers.
        // USE THE POINTER RESOLVED WITH A REAL INSTANCE.
        //
        // This used to call nextGIPA(nullptr, "vkEnumerateDeviceExtensionProperties").
        // vkGetInstanceProcAddr with a NULL instance resolves only the handful
        // of global entry points - vkCreateInstance and the instance-level
        // enumerations - and returns NULL for everything else. So the pointer
        // was null, `have` stayed empty, and EVERY extension was reported "not
        // supported by this driver".
        //
        // That message was believed. Optical flow, format_feature_flags2,
        // pageable_device_local_memory and memory_priority were all recorded as
        // unavailable and quietly dropped, while X-Plane's own log listed every
        // one of them as present. The extension injection has never once
        // worked, and it announced its failure in a form indistinguishable from
        // a hardware limitation.
        //
        // g_nextEnumDeviceExt is resolved at instance creation with the real
        // VkInstance and has been correct all along.
        uint32_t n = 0;
        std::vector<VkExtensionProperties> have;
        if (g_nextEnumDeviceExt &&
            g_nextEnumDeviceExt(phys, nullptr, &n, nullptr) == VK_SUCCESS && n) {
            have.resize(n);
            g_nextEnumDeviceExt(phys, nullptr, &n, have.data());
        }
        trace("DEVICE: driver offers %zu device extensions", have.size());

        // ---- ASK XeSS WHAT IT WOULD NEED, WHILE A DEVICE CAN STILL BE GIVEN IT.
        //
        // xessVKCreateContext takes a device that was ALREADY created with the
        // extensions and features XeSS requires, so this is the only moment the
        // answer can be acted on - afterwards X-Plane's device exists and a
        // layer cannot go back and add to it.
        //
        // Reported whether or not it succeeds. "XeSS wants three extensions
        // this driver does not have" and "libxess.dll is missing" are different
        // problems, and the availability report has different names for them.
        // ---- OFF BY DEFAULT, BECAUSE THIS KILLED THE SIM.
        //
        // The first version of this ran unconditionally and X-Plane died
        // inside vkCreateDevice - the trace stops on the line above and the
        // log never reaches a frame. Two plausible causes, neither yet
        // separated: LoadLibrary of a 77 MB DLL from inside a layer's device
        // creation, and XeSS re-entering the Vulkan loader while the loader is
        // already inside our interception of it.
        //
        // Whatever the cause, an optional capability probe must not be able to
        // take the whole mod down, so it is behind a key that defaults to off
        // and the default path is exactly what shipped before it existed.
        if (live::onoff("xess.probe", "TAA_XESS_PROBE", false)) {
            xessprobe::query(g_firstInstance, phys);
            const xessprobe::Requirements &xr = xessprobe::state();
            if (xr.queried) {
                trace("XESS: runtime %u.%u.%u wants %zu device extension(s)",
                      xr.major, xr.minor, xr.patch, xr.deviceExts.size());
                for (size_t xi = 0; xi < xr.deviceExts.size(); ++xi) {
                    bool present = false;
                    for (size_t k = 0; k < have.size(); ++k)
                        if (strcmp(have[k].extensionName,
                                   xr.deviceExts[xi].c_str()) == 0) { present = true; break; }
                    trace("XESS:   %-52s %s", xr.deviceExts[xi].c_str(),
                          present ? "offered by the driver" : "NOT OFFERED - XeSS cannot run here");
                }
            } else {
                trace("XESS: not usable - %s", xr.why);
            }
        }

        // ---- UPSCALER CAPABILITY, RECORDED WHILE THE LIST IS IN SCOPE.
        //
        // Cooperative matrix is the capability FSR 4's model actually needs, so
        // it is what gets asked about - rather than matching "RDNA 4" against
        // props.deviceName, which is marketing text that changes between driver
        // releases and is localised in some of them.
        for (size_t ei = 0; ei < have.size(); ++ei) {
            if (strcmp(have[ei].extensionName, "VK_KHR_cooperative_matrix") == 0) {
                g_upscalerCaps.coopMatrix = true;
                break;
            }
        }
        if (g_getPhysProps) {
            VkPhysicalDeviceProperties up;
            memset(&up, 0, sizeof(up));
            g_getPhysProps(phys, &up);
            g_upscalerCaps.vendorId = up.vendorID;
        }
        // Last, so a half-filled struct can never read as authoritative.
        g_upscalerCaps.valid = true;

        // ---- STREAMLINE'S OWN REQUIREMENTS, ADDED FROM ITS OWN LIST.
        //
        // The static list below covers what this layer needs. Streamline needs
        // more, and it tells us exactly what - we were printing that list and
        // then ignoring it. VK_KHR_push_descriptor, VK_KHR_maintenance4 and the
        // external_*_win32 pair are not core, so a device created without them
        // cannot host DLSS-G, and Reflex says so with -229.
        //
        // Taking the names from Streamline rather than hardcoding them means the
        // list cannot drift when the SDK changes what it wants.
        std::vector<const char*> slWanted;

        static const char *kWanted[] = {
            // Reflex needs this and Streamline does NOT list it.
            //
            // slSetVulkanInfo returns 24 (eErrorExceptionHandler) - Streamline
            // throwing internally - and immediately before it the shim log has
            // "Low latency API for VK failed to initialize device -229". Reflex
            // is the upstream failure and everything after it is fallout.
            //
            // Reflex on Vulkan is implemented over VK_NV_low_latency2, which is
            // absent from the eleven extensions slGetFeatureRequirements
            // reports. DLSS-G requires Reflex, so a device without it cannot
            // host frame generation no matter how many of the listed
            // requirements are met - which is why adding all eleven and growing
            // every queue changed nothing.
            "VK_NV_low_latency2",

            // The ORIGINAL low latency extension, asked for alongside the "2".
            //
            // Reflex is the first failure in the chain - "Low latency API for
            // VK failed to initialize device -229" arrives before the private
            // data warning and before the exception - and it comes from
            // NvLowLatencyVk.dll, which predates VK_NV_low_latency2. Only the
            // newer extension has ever been requested here. Adding the older
            // one costs nothing if the driver does not offer it: the list below
            // is filtered against what the driver advertises, and anything
            // missing is skipped and logged rather than failing the device.
            "VK_NV_low_latency",
            "VK_NV_optical_flow",          // hardware motion estimation
            "VK_KHR_format_feature_flags2",// required by the optical flow spec

            // Streamline attaches its own state to the swap chain through a
            // private data slot, and reports losing that immediately before it
            // throws. The entry point is core in 1.3, and X-Plane's instance
            // already asks for 1.3 - but a core promotion only guarantees the
            // command exists, not that vkGetDeviceProcAddr will hand it over.
            // Enabling the extension by name removes the question.
            "VK_EXT_private_data",

            // The two that make a custom pager possible.
            //
            // A layer cannot evict X-Plane's textures - it does not own them
            // and has no idea which ortho tile is forty kilometres away. But it
            // does not need to: pageable_device_local_memory lets the DRIVER
            // demote allocations to system RAM under pressure instead of
            // failing, and memory_priority decides what it demotes first.
            //
            // So the division of labour is: X-Plane's pager off, we classify
            // what matters, the driver does the eviction. That is the OpenGL
            // behaviour whose loss forced X-Plane to write a pager in the first
            // place, with the residency decisions made explicitly rather than
            // guessed at by the driver alone.
            "VK_EXT_pageable_device_local_memory",
            "VK_EXT_memory_priority",

            // WHAT MAKES THE VELOCITY ATTACHMENT POSSIBLE WITHOUT DUPLICATING
            // EVERY PIPELINE.
            //
            // Under dynamic rendering a pipeline's colour attachment formats
            // must match the render pass it is used in. X-Plane reuses the same
            // pipelines across several passes, so adding an attachment to the
            // scene pass alone would make those pipelines invalid in every
            // other pass they appear in - and the only way out would be to
            // build a second variant of each, which is a lot of pipelines and a
            // lot of memory.
            //
            // This extension relaxes the match: a pipeline may declare a format
            // for an attachment the pass leaves undefined, and vice versa. So
            // EVERY pass gains one attachment slot and every pipeline declares
            // one extra format, uniformly - and only the scene pass actually
            // binds an image to it. Everywhere else the slot is null and costs
            // nothing.
            "VK_EXT_dynamic_rendering_unused_attachments"
        };

        // ---- TAA_SL_NO_LL: leave the low latency extensions OFF the device.
        //
        // NvLL_VK_InitLowLatencyDevice rejects our device with -229, measured
        // directly rather than inferred - our own call to it fails exactly as
        // Reflex's does, while NvLL_VK_Initialize returns 0. So the API and the
        // driver are fine and the DEVICE is what is refused.
        //
        // The suspect is an addition of ours. VK_NV_low_latency2 is NOT in the
        // eleven extensions slGetFeatureRequirements reports; it was added here
        // on the reasoning that "Reflex is implemented over low_latency2, so it
        // must need it". That reasoning has no evidence behind it, and the
        // opposite reading fits better: NvLL_VK is NVIDIA's PRIVATE low latency
        // path and VK_NV_low_latency2 is the PUBLIC one, two implementations of
        // the same thing, and a driver refusing the private path on a device
        // that enabled the public one would produce exactly this. Streamline
        // omitting it from its own requirements is then deliberate, not an
        // oversight to be corrected.

        // ---- THE `false &&` HERE CREATED AN INVALID DEVICE. (root cause of
        //      the 4K Felis-load DEVICE_LOST, found by the validation layer)
        //
        // VK_NV_low_latency2 REQUIRES VK_KHR_present_id. We enabled the former
        // and never the latter, so every device this layer created was invalid:
        //   VUID-vkCreateDevice-ppEnabledExtensionNames-01387
        //   "Missing extension required to enable device extension
        //    VK_NV_low_latency2: VK_KHR_present_id"
        // An invalid device is undefined behaviour, not a guaranteed failure -
        // which is why it ran at all, why it was intermittent, and why it died
        // under load rather than at creation. The whole block is gated on
        // TAA_VRAM_HOOKS, which is exactly why hooks-off runs survived and
        // hooks-on runs crashed: nothing to do with VRAM policy.
        //
        // These two exist for Streamline/DLSS-G, which this layer does not
        // run. Not enabling them is strictly better than enabling them plus a
        // dependency chain we have no use for. TAA_SL_LOW_LATENCY=1 re-arms
        // both for anyone who takes the Streamline path up again - and that
        // path must add VK_KHR_present_id with them.
        static const bool wantLowLatency = getenv("TAA_SL_LOW_LATENCY") &&
                                           atoi(getenv("TAA_SL_LOW_LATENCY")) != 0;

        // ---- TAA_DLSS_EXT: the DLSS-G / Streamline extensions, OFF by default.
        //
        // These sit on the device create for frame generation, which this layer
        // does not run. VK_NV_optical_flow, VK_KHR_format_feature_flags2 (which
        // the optical flow spec requires) and VK_EXT_private_data (Streamline's
        // swap chain state) buy nothing while DLSS-G is absent, and every
        // extension enabled is driver code the sim would otherwise not run at
        // all - on a device that has already been made invalid once by an
        // addition of ours.
        //
        // VK_EXT_dynamic_rendering_unused_attachments is deliberately NOT in
        // this set: the velocity attachment depends on it, so motion vectors
        // and TAA go with it. Only the frame generation names are gated.
        //
        // TAA_DLSS_EXT=1 re-arms them for whoever takes the DLSS-G path up.
        static const bool wantDlssExt = getenv("TAA_DLSS_EXT") &&
                                        atoi(getenv("TAA_DLSS_EXT")) != 0;
        for (size_t k = 0; k < sizeof(kWanted)/sizeof(kWanted[0]); ++k) {
            if (!wantLowLatency &&
                (!strcmp(kWanted[k], "VK_NV_low_latency2") ||
                 !strcmp(kWanted[k], "VK_NV_low_latency"))) {
                trace("DEVICE: NOT enabling %s - it requires VK_KHR_present_id, "
                      "which nothing here provides, and an extension enabled "
                      "without its dependency makes the DEVICE invalid",
                      kWanted[k]);
                continue;
            }
            if (!wantDlssExt &&
                (!strcmp(kWanted[k], "VK_NV_optical_flow") ||
                 !strcmp(kWanted[k], "VK_KHR_format_feature_flags2") ||
                 !strcmp(kWanted[k], "VK_EXT_private_data"))) {
                trace("DEVICE: NOT enabling %s - DLSS-G/Streamline only, and this "
                      "layer runs neither. TAA_DLSS_EXT=1 re-arms it.",
                      kWanted[k]);
                continue;
            }
            bool supported = false;
            for (size_t i = 0; i < have.size(); ++i)
                if (!strcmp(have[i].extensionName, kWanted[k])) { supported = true; break; }

            bool already = false;
            for (size_t i = 0; i < exts.size(); ++i)
                if (!strcmp(exts[i], kWanted[k])) { already = true; break; }

            if (supported && !already) {
                exts.push_back(kWanted[k]);
                trace("DEVICE: adding %s", kWanted[k]);
            } else if (!supported) {
                trace("DEVICE: %s not supported by this driver - skipping", kWanted[k]);
            }
        }

        // The same treatment for Streamline's list.
        if (!wantDlssExt && !slWanted.empty())
            trace("DEVICE: skipping Streamline's %u requirement(s) - DLSS-G is "
                  "not running, so its device extensions are not added. TAA_DLSS_EXT=1.",
                  (unsigned)slWanted.size());
        for (size_t k = 0; wantDlssExt && k < slWanted.size(); ++k) {
            // The same exclusion as above, and it has to be here too:
            // kSlDeviceExt carries VK_NV_low_latency2 as well, so filtering
            // only kWanted would leave the extension enabled by this loop and
            // the test would silently measure nothing.
            if (!wantLowLatency &&
                (!strcmp(slWanted[k], "VK_NV_low_latency2") ||
                 !strcmp(slWanted[k], "VK_NV_low_latency"))) {
                trace("DEVICE: NOT enabling %s (Streamline list) - same missing "
                      "VK_KHR_present_id dependency as above",
                      slWanted[k]);
                continue;
            }
            bool supported = false;
            for (size_t i = 0; i < have.size(); ++i)
                if (!strcmp(have[i].extensionName, slWanted[k])) { supported = true; break; }
            bool already = false;
            for (size_t i = 0; i < exts.size(); ++i)
                if (!strcmp(exts[i], slWanted[k])) { already = true; break; }
            if (supported && !already) {
                exts.push_back(slWanted[k]);
                trace("DEVICE: adding %s (Streamline)", slWanted[k]);
            } else if (!supported) {
                // Worth saying loudly: a Streamline requirement the driver does
                // not offer means DLSS-G cannot run on this machine, and that is
                // a different answer from "we forgot to ask for it".
                trace("DEVICE: %s REQUIRED BY STREAMLINE but not offered by this "
                      "driver - DLSS-G cannot run", slWanted[k]);
            }
        }


        // ENABLING THE EXTENSION IS NOT ENABLING THE FEATURE.
        //
        // Adding the extension name only makes the entry points legal. The
        // relaxed format matching is a FEATURE bit and stays off until it is
        // requested through the pNext chain - and with it off, the driver
        // silently keeps enforcing exact format matching, so the first pipeline
        // built against a pass with a null velocity slot is rejected. That
        // failure would arrive far from its cause.
        //
        // The struct is static because the chain must outlive this call: the
        // driver reads pNext during vkCreateDevice, and a stack local would be
        // gone. Chained ahead of whatever X-Plane already had, which stays
        // intact behind it.
        static VkPhysicalDeviceDynamicRenderingUnusedAttachmentsFeaturesEXT unusedFeat;
        bool haveUnused = false;
        for (size_t i = 0; i < exts.size(); ++i)
            if (!strcmp(exts[i], "VK_EXT_dynamic_rendering_unused_attachments"))
                haveUnused = true;
        if (haveUnused) {
            memset(&unusedFeat, 0, sizeof(unusedFeat));
            unusedFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_UNUSED_ATTACHMENTS_FEATURES_EXT;
            unusedFeat.dynamicRenderingUnusedAttachments = VK_TRUE;
            unusedFeat.pNext = (void*)ci2.pNext;
            ci2.pNext = &unusedFeat;
            trace("DEVICE: dynamicRenderingUnusedAttachments feature requested "
                  "- lets one pipeline serve passes with and without the "
                  "velocity attachment");
        }

        // ---- MEMORY PRIORITY, which we have been requesting and not using.
        //
        // VK_EXT_memory_priority and VK_EXT_pageable_device_local_memory are
        // both on the device already - X-Plane asks for them itself - and the
        // layer has never set a priority on anything, so the driver had no way
        // to know which allocations it should spill first. Under pressure it
        // guesses, and what it guesses wrong costs texture resolution: the log
        // has X-Plane's pager going 1.0 -> 0.5 -> 0.25 -> 0.0625 in one second.
        //
        // VkMemoryPriorityAllocateInfoEXT is IGNORED unless this feature is
        // enabled, so asking for the extension without it did nothing at all.
        // With it on, our own buffers can be marked low and demoted to system
        // RAM ahead of the sim's textures - which is the right order, because
        // an upscaler's history streaming over PCIe costs frame time while a
        // texture at a sixteenth costs the picture.
        static VkPhysicalDeviceMemoryPriorityFeaturesEXT prioFeat;
        bool havePrio = false;
        for (size_t i = 0; i < exts.size(); ++i)
            if (!strcmp(exts[i], "VK_EXT_memory_priority")) havePrio = true;
        if (havePrio) {
            memset(&prioFeat, 0, sizeof(prioFeat));
            prioFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT;
            prioFeat.memoryPriority = VK_TRUE;
            prioFeat.pNext = (void*)ci2.pNext;
            ci2.pNext = &prioFeat;
            g_memoryPriority = true;
            trace("DEVICE: memoryPriority feature requested - our own "
                  "allocations will be marked low so the driver demotes them "
                  "before X-Plane's textures");
        }

        // ---- THE VRAM SYSTEM'S EXTENSIONS, added defensively if the driver
        // offers them and nobody asked. X-Plane requests both itself when it
        // sees them, so these adds are normally no-ops - but the priority
        // engine must not depend on the engine's mood.
        {
            const char *vramWanted[] = {
                "VK_EXT_memory_priority",
                "VK_EXT_pageable_device_local_memory",
            };
            // Total-kill also skips the defensive extension adds, so device
            // creation matches the pre-VRAM build exactly.
            const char *vhk = getenv("TAA_VRAM_HOOKS");
            bool vramDead = vhk && atoi(vhk) == 0;
            for (int k = vramDead ? 2 : 0; k < 2; ++k) {
                bool supported = false;
                for (size_t i = 0; i < have.size(); ++i)
                    if (!strcmp(have[i].extensionName, vramWanted[k])) { supported = true; break; }
                bool already = false;
                for (size_t i = 0; i < exts.size(); ++i)
                    if (!strcmp(exts[i], vramWanted[k])) { already = true; break; }
                if (supported && !already) {
                    exts.push_back(vramWanted[k]);
                    trace("DEVICE: adding %s (VRAM system)", vramWanted[k]);
                }
            }
        }

        // The pageable feature is what makes vkSetDeviceMemoryPriorityEXT
        // legal - the priority engine's live re-prioritisation runs on it.
        // Feature struct static for the same lifetime reason as the others.
        static VkPhysicalDevicePageableDeviceLocalMemoryFeaturesEXT pageFeat;
        bool havePageable = false;
        for (size_t i = 0; i < exts.size(); ++i)
            if (!strcmp(exts[i], "VK_EXT_pageable_device_local_memory"))
                havePageable = true;
        // ---- OFF BY DEFAULT: prime suspect in three identical DEVICE_LOST
        // crashes at flight start (0:01:26-0:02:15), the third with every
        // VRAM actuator live-disabled - which leaves device creation as the
        // delta, and this feature is the one bit never flown before. It
        // changes the driver's residency machinery globally. TAA_VRAM_PAGEABLE=1
        // re-arms it for an A/B once the system is otherwise proven.
        {
            const char *pg = getenv("TAA_VRAM_PAGEABLE");
            if (!pg || atoi(pg) == 0) havePageable = false;
        }
        if (havePageable) {
            memset(&pageFeat, 0, sizeof(pageFeat));
            pageFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PAGEABLE_DEVICE_LOCAL_MEMORY_FEATURES_EXT;
            pageFeat.pageableDeviceLocalMemory = VK_TRUE;
            pageFeat.pNext = (void*)ci2.pNext;
            ci2.pNext = &pageFeat;
            g_pageableMemory = true;
            trace("DEVICE: pageableDeviceLocalMemory feature requested - "
                  "vkSetDeviceMemoryPriorityEXT becomes legal and the priority "
                  "engine can re-rank live blocks");
        }

        ci2.enabledExtensionCount   = (uint32_t)exts.size();
        ci2.ppEnabledExtensionNames = exts.data();
    }


    // ---- FRAME GENERATION NEEDS QUEUES THE SIM NEVER TOUCHES.
    //
    // Deliberately after the Streamline block and reading ci2 rather than ci:
    // both augmenters grow the same queue list, and one that started again from
    // X-Plane's original request would drop the other's queues on the floor. The
    // device would still be created, and the lost index would surface much later
    // as a submit to a queue that does not exist.
    //
    // Queues cannot be added once the device exists, so if this does not happen
    // here there is no frame generation for the rest of the session.
    {
        VkDeviceCreateInfo fgCi;
    }

    // ---- independentBlend. THIS IS WHY 14,835 PIPELINES WERE REJECTED.
    //
    // We append one colour blend attachment for the velocity target, and it is
    // deliberately unlike X-Plane's own: blendEnable FALSE and a write mask of
    // R|G, because a velocity is replaced rather than blended and the target
    // has two channels. The spec permits that only with independentBlend:
    //
    //   "If the independentBlend feature is not enabled, all elements of
    //    pAttachments must be identical."
    //
    // The layer never asked for it. X-Plane does not enable it, and 74% of its
    // pipelines blend - so almost every extended pipeline was invalid and the
    // driver refused it as VK_ERROR_UNKNOWN, which is not a validation message
    // and says nothing about what was wrong. The batch then fell back to the
    // application's originals, leaving pipelines with no varyings and no
    // velocity output. That is the whole reason prevClip read zero, and why
    // the push constant, the offset, the varying location and the matrix were
    // all suspected in turn while every one of them was innocent.
    //
    // Enabled only if the driver reports it, and only added to a features
    // struct we own, so X-Plane's request is never altered in a way that could
    // stop the sim starting.
    VkPhysicalDeviceFeatures wantFeatures;
    memset(&wantFeatures, 0, sizeof(wantFeatures));
    if (ci->pEnabledFeatures) wantFeatures = *ci->pEnabledFeatures;

    // ---- RESOLVED FROM THE APPLICATION'S INSTANCE, NOT FROM A NULL ONE.
    //
    // This passed nullptr as the instance. vkGetInstanceProcAddr accepts a null
    // instance for exactly four global-level entry points -
    // vkEnumerateInstanceVersion, vkEnumerateInstanceExtensionProperties,
    // vkEnumerateInstanceLayerProperties and vkCreateInstance - and
    // vkGetPhysicalDeviceFeatures is not among them. So it returned NULL, the
    // whole query was skipped, haveIndependentBlend stayed false, and the
    // feature was never requested.
    //
    // That reading was then quoted as evidence the device did not support
    // independentBlend, which is the feature the velocity attachment needs and
    // the one named in the comment above as the cause of 14,835 rejected
    // pipelines. A query that never ran is not a negative result.
    //
    // The instance map is keyed by dispatch pointer and a physical device
    // shares its instance's dispatch key, so phys finds its own instance. Same
    // discipline as the memory-properties getter above: resolve physical-device
    // functions from the instance the physical device actually belongs to, or
    // the loader looks the handle up in the wrong list.
    bool haveIndependentBlend = false;
    {
        PFN_vkGetInstanceProcAddr gipa = nextGIPA;
        VkInstance owner = VK_NULL_HANDLE;
        std::map<void*, InstanceData>::iterator ii = g_instances.find(dispatchKey(phys));
        if (ii != g_instances.end()) { gipa = ii->second.gipa; owner = ii->second.instance; }
        else if (g_firstInstance != VK_NULL_HANDLE) { owner = g_firstInstance; }

        PFN_vkGetPhysicalDeviceFeatures getFeatures = owner != VK_NULL_HANDLE
            ? (PFN_vkGetPhysicalDeviceFeatures)gipa(owner, "vkGetPhysicalDeviceFeatures")
            : nullptr;
        if (getFeatures) {
            VkPhysicalDeviceFeatures supported;
            memset(&supported, 0, sizeof(supported));
            getFeatures(phys, &supported);
            haveIndependentBlend = (supported.independentBlend == VK_TRUE);
        }
        trace("SPIRV INJECT: independentBlend query - instance %s, getter %s, "
              "supported=%d (a null instance here returned no getter at all and "
              "the answer was read as unsupported)",
              owner != VK_NULL_HANDLE ? "found" : "NOT FOUND",
              getFeatures ? "resolved" : "NULL", haveIndependentBlend ? 1 : 0);
    }

    // Only touch pEnabledFeatures when X-Plane used that form. When it passes
    // features through pNext instead, the chain is edited there.
    bool patchedFeatures = false;
    if (haveIndependentBlend && !wantFeatures.independentBlend) {
        wantFeatures.independentBlend = VK_TRUE;
        if (ci->pEnabledFeatures) { ci2.pEnabledFeatures = &wantFeatures; patchedFeatures = true; }
        else {
            for (VkBaseOutStructure *pn = (VkBaseOutStructure*)ci2.pNext; pn; pn = pn->pNext)
                if (pn->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2) {
                    VkPhysicalDeviceFeatures2 *f2 = (VkPhysicalDeviceFeatures2*)pn;
                    f2->features.independentBlend = VK_TRUE;
                    patchedFeatures = true;
                    break;
                }
        }
    }
    trace("SPIRV INJECT: independentBlend supported=%d, enabled by us=%d - without "
          "it every appended blend state must match X-Plane's exactly, and ours "
          "deliberately does not",
          haveIndependentBlend ? 1 : 0, patchedFeatures ? 1 : 0);

    VkResult r = nextCreate(phys, &ci2, alloc, out);

    // If the modified create fails, fall back to X-Plane's original request.
    // Failing to start the sim because a diagnostic extension was refused would
    // be an absurd trade, and a driver can reject a combination even when each
    // extension is individually advertised.
    if (r != VK_SUCCESS) {
        trace("DEVICE: creation with added extensions failed (%d) - retrying "
              "with the application's original list", (int)r);
        r = nextCreate(phys, ci, alloc, out);
    }
    if (r != VK_SUCCESS) return r;


    DeviceData dd;
    memset(&dd, 0, sizeof(dd));
    dd.device        = *out;
    dd.phys          = phys;
    dd.gdpa          = nextGDPA;
    dd.destroyDevice = (PFN_vkDestroyDevice)nextGDPA(*out, "vkDestroyDevice");
    dd.createImage   = (PFN_vkCreateImage)nextGDPA(*out, "vkCreateImage");
    dd.destroyImage  = (PFN_vkDestroyImage)nextGDPA(*out, "vkDestroyImage");
    dd.queuePresent  = (PFN_vkQueuePresentKHR)nextGDPA(*out, "vkQueuePresentKHR");
    dd.createSampler = (PFN_vkCreateSampler)nextGDPA(*out, "vkCreateSampler");
    dd.cmdBeginRenderPass = (PFN_vkCmdBeginRenderPass)nextGDPA(*out, "vkCmdBeginRenderPass");
    // Core in 1.3, KHR extension before that. X-Plane uses dynamic rendering
    // exclusively - vkCmdBeginRenderPass fired zero times in 2760 frames - so
    // failing to resolve this means seeing no frame structure at all.
    dd.cmdBeginRendering = (PFN_vkCmdBeginRendering)nextGDPA(*out, "vkCmdBeginRendering");
    if (!dd.cmdBeginRendering)
        dd.cmdBeginRendering = (PFN_vkCmdBeginRendering)nextGDPA(*out, "vkCmdBeginRenderingKHR");
    dd.cmdSetViewport     = (PFN_vkCmdSetViewport)nextGDPA(*out, "vkCmdSetViewport");

#define GD(m, N) dd.m = (PFN_vk##N)nextGDPA(*out, "vk" #N)
    GD(getImageMemReq, GetImageMemoryRequirements);  GD(getBufferMemReq, GetBufferMemoryRequirements);
    GD(allocateMemory, AllocateMemory);              GD(freeMemory, FreeMemory);
    GD(bindImageMemory, BindImageMemory);             GD(bindBufferMemory, BindBufferMemory);
    GD(mapMemory, MapMemory);
    GD(createImageView, CreateImageView);             GD(destroyImageView, DestroyImageView);
    GD(createBuffer, CreateBuffer);                GD(destroyBuffer, DestroyBuffer);
    GD(createDescriptorSetLayout, CreateDescriptorSetLayout);   GD(createDescriptorPool, CreateDescriptorPool);
    GD(allocateDescriptorSets, AllocateDescriptorSets);      GD(updateDescriptorSets, UpdateDescriptorSets);
    GD(createShaderModule, CreateShaderModule);          GD(createPipelineLayout, CreatePipelineLayout);
    GD(createComputePipelines, CreateComputePipelines);
    GD(createCommandPool, CreateCommandPool);           GD(allocateCommandBuffers, AllocateCommandBuffers);
    GD(beginCommandBuffer, BeginCommandBuffer);
    // Resolved into plain globals, not DeviceData: the re-push happens on a
    // command buffer whose device is not looked up on that path.
    if (!g_nextCmdDraw)
        g_nextCmdDraw = (PFN_vkCmdDraw)nextGDPA(*out, "vkCmdDraw");
    if (!g_nextCmdDrawIndexed)
        g_nextCmdDrawIndexed = (PFN_vkCmdDrawIndexed)nextGDPA(*out, "vkCmdDrawIndexed");
    if (!g_nextCmdDrawIndirect)
        g_nextCmdDrawIndirect = (PFN_vkCmdDrawIndirect)nextGDPA(*out, "vkCmdDrawIndirect");
    if (!g_nextCmdDrawIndexedIndirect)
        g_nextCmdDrawIndexedIndirect = (PFN_vkCmdDrawIndexedIndirect)nextGDPA(*out, "vkCmdDrawIndexedIndirect");
    if (!g_nextCmdDrawIndirectCount)
        g_nextCmdDrawIndirectCount = (PFN_vkCmdDrawIndirectCount)nextGDPA(*out, "vkCmdDrawIndirectCount");
    if (!g_nextCmdDrawIndexedIndirectCount)
        g_nextCmdDrawIndexedIndirectCount = (PFN_vkCmdDrawIndexedIndirectCount)nextGDPA(*out, "vkCmdDrawIndexedIndirectCount");          GD(endCommandBuffer, EndCommandBuffer);
    GD(cmdBindPipeline, CmdBindPipeline);             GD(cmdBindDescriptorSets, CmdBindDescriptorSets);
    GD(cmdPushConstants, CmdPushConstants);            GD(cmdDispatch, CmdDispatch);
    dd.cmdPushDescriptorSet = (PFN_vkCmdPushDescriptorSetKHR)nextGDPA(*out, "vkCmdPushDescriptorSetKHR");
    if (!dd.cmdPushDescriptorSet)
        dd.cmdPushDescriptorSet = (PFN_vkCmdPushDescriptorSetKHR)
            nextGDPA(*out, "vkCmdPushDescriptorSet");   // core in 1.4
    dd.cmdPushDescriptorSet2 = (PFN_vkCmdPushDescriptorSet2)
        nextGDPA(*out, "vkCmdPushDescriptorSet2");
    if (!dd.cmdPushDescriptorSet2)
        dd.cmdPushDescriptorSet2 = (PFN_vkCmdPushDescriptorSet2)
            nextGDPA(*out, "vkCmdPushDescriptorSet2KHR");
    GD(cmdPipelineBarrier, CmdPipelineBarrier);          GD(cmdCopyImageToBuffer, CmdCopyImageToBuffer);
    GD(cmdFillBuffer, CmdFillBuffer);
    GD(cmdCopyImage, CmdCopyImage);                   GD(deviceWaitIdle, DeviceWaitIdle);
    GD(cmdClearColorImage, CmdClearColorImage);
    GD(cmdBlitImage, CmdBlitImage);
    GD(cmdResolveImage, CmdResolveImage);
    GD(getSwapchainImagesKHR, GetSwapchainImagesKHR);
    GD(createQueryPool, CreateQueryPool);       GD(destroyQueryPool, DestroyQueryPool);
    GD(cmdResetQueryPool, CmdResetQueryPool);   GD(cmdWriteTimestamp, CmdWriteTimestamp);
    GD(getQueryPoolResults, GetQueryPoolResults);
    GD(createSwapchainKHR, CreateSwapchainKHR);   GD(destroySwapchainKHR, DestroySwapchainKHR);
    GD(acquireNextImageKHR, AcquireNextImageKHR);
    g_nextAllocateMemory = (PFN_vkAllocateMemory)nextGDPA(*out, "vkAllocateMemory");
    g_nextCmdPipelineBarrier2KHR = (PFN_vkCmdPipelineBarrier2)
                                       nextGDPA(*out, "vkCmdPipelineBarrier2");
    if (!g_nextCmdPipelineBarrier2KHR)
        g_nextCmdPipelineBarrier2KHR = (PFN_vkCmdPipelineBarrier2)
                                       nextGDPA(*out, "vkCmdPipelineBarrier2KHR");
        // Choose the motion vector varying locations from the DEVICE, not from a
    // sample of shaders. Fixing them at 15/16 - reasoned from a 40-module dump
    // where the highest was 7 - was refused by 97 of X-Plane's 1500 modules,
    // because its large shaders already use both.
    // USE THE PHYSICAL DEVICE THIS FUNCTION WAS HANDED.
    //
    // The first version looked the device up in g_devices to find its
    // VkPhysicalDevice, and found nothing: the DeviceData entry is not inserted
    // until later in this function, so the lookup failed and the whole block
    // was skipped in silence - no trace line, locations left at their defaults,
    // and 105 shaders refused for exactly the reason this was meant to fix.
    //
    // TAA_CreateDevice receives the physical device as its first argument.
    // There was never anything to look up.
    if (g_getPhysProps && phys) {
        VkPhysicalDeviceProperties pp;
        memset(&pp, 0, sizeof(pp));
        g_getPhysProps(phys, &pp);
        spvinj::chooseLocations(pp.limits.maxVertexOutputComponents,
                                pp.limits.maxFragmentInputComponents);
        // How many descriptor sets a pipeline layout may declare. Appending
        // ours to a layout already at the limit would make the layout
        // creation FAIL, and a failed layout is not a velocity hole - it is a
        // pipeline X-Plane never gets. Captured here because this is the only
        // place the physical device is in scope.
        g_maxBoundSets = pp.limits.maxBoundDescriptorSets;
        spvinj::chooseAttachment(pp.limits.maxColorAttachments);
        uint32_t pcOff = spvinj::choosePushOffset(pp.limits.maxPushConstantsSize);
        trace("SPIRV INJECT: push block of %u bytes at offset %u of %u - leaves "
              "X-Plane's own push constants %u bytes before they share storage "
              "with ours%s",
              spvinj::kPushConstantBytes, pcOff, pp.limits.maxPushConstantsSize,
              pcOff,
              pcOff < 64 ? " *** TIGHT: a fragment block reaching past this "
                           "would silently overwrite the matrix ***" : "");
        trace("SPIRV INJECT: varyings at Location %u/%u, velocity at colour "
              "attachment %u (maxVertexOutputComponents=%u maxColorAttachments=%u)",
              spvinj::currClipLocation(), spvinj::prevClipLocation(),
              spvinj::mvAttachmentIndex(),
              pp.limits.maxVertexOutputComponents,
              pp.limits.maxColorAttachments);
        // ---- FAIL LOUDLY, NOT SILENTLY, ON A DEVICE WITH NO ROOM.
        //
        // X-Plane's highest varying Location is 16, measured across all 6855
        // shader modules. Vulkan's guaranteed minimum is also 16. So a
        // minimum-specification implementation has no free pair above X-Plane
        // at all, and the fallback arithmetic would return 13/14 - Locations
        // read by 2352 and 2784 fragment shaders respectively.
        //
        // Nothing rejects that. The pipeline builds, the module validates, and
        // the varyings quietly carry X-Plane's own interpolants into our
        // velocity computation. Say so at full volume; the module census runs
        // later and may still rescue it, which is the only reason this is a
        // warning rather than a hard stop.
        if (!spvinj::locationsAreSafe())
            trace("SPIRV INJECT: *** THIS DEVICE REPORTS ONLY %u VARYING "
                  "LOCATIONS. X-Plane's own ceiling is %u, so there is no free "
                  "pair above it and Locations %u/%u may already be in use - "
                  "Location 13 is read by 2352 shaders and Location 14 by 2784. "
                  "Motion vectors from those draws would be X-Plane's own "
                  "interpolants. The module census may still find a free pair; "
                  "if it does not, this build cannot produce correct vectors on "
                  "this device. ***",
                  spvinj::deviceLocationCount(), spvinj::kXPlaneMaxLocation,
                  spvinj::currClipLocation(), spvinj::prevClipLocation());
    } else {
        trace("SPIRV INJECT: could not read device limits - varyings stay at "
              "Location %u/%u, which some shaders may already use",
              spvinj::currClipLocation(), spvinj::prevClipLocation());
    }
    g_nextCmdBindPipeline = (PFN_vkCmdBindPipeline)nextGDPA(*out, "vkCmdBindPipeline");
    g_nextCmdPushConstants = (PFN_vkCmdPushConstants)nextGDPA(*out, "vkCmdPushConstants");
    if (!g_nextCmdBindSets)
        g_nextCmdBindSets = (PFN_vkCmdBindDescriptorSets)
            nextGDPA(*out, "vkCmdBindDescriptorSets");
    g_nextCmdBlitImage = (PFN_vkCmdBlitImage)nextGDPA(*out, "vkCmdBlitImage");
    g_nextCmdCopyImage = (PFN_vkCmdCopyImage)nextGDPA(*out, "vkCmdCopyImage");
    g_nextCmdClearColorImage = (PFN_vkCmdClearColorImage)
                                   nextGDPA(*out, "vkCmdClearColorImage");
    g_nextCmdPipelineBarrier2 = (PFN_vkCmdPipelineBarrier)
                                    nextGDPA(*out, "vkCmdPipelineBarrier");
    g_nextCmdCopyBufferToImage = (PFN_vkCmdCopyBufferToImage)
                                     nextGDPA(*out, "vkCmdCopyBufferToImage");
    g_nextCreateShaderModule = (PFN_vkCreateShaderModule)
                                   nextGDPA(*out, "vkCreateShaderModule");
    g_nextCreateGfxPipelines = (PFN_vkCreateGraphicsPipelines)
                                   nextGDPA(*out, "vkCreateGraphicsPipelines");
    GD(createFence, CreateFence);                 GD(resetFences, ResetFences);  GD(waitForFences, WaitForFences);
    GD(queueSubmit, QueueSubmit);                 GD(getDeviceQueue, GetDeviceQueue);
    GD(destroyPipeline, DestroyPipeline);         GD(destroyPipelineLayout, DestroyPipelineLayout);
    GD(destroyShaderModule, DestroyShaderModule); GD(destroyDescriptorPool, DestroyDescriptorPool);
    GD(destroyDescriptorSetLayout, DestroyDescriptorSetLayout);
    GD(destroySampler, DestroySampler);           GD(destroyCommandPool, DestroyCommandPool);
    GD(destroyFence, DestroyFence);               GD(unmapMemory, UnmapMemory);
#undef GD


    g_nextQueueSubmit        = dd.queueSubmit;
    g_nextQueuePresent       = dd.queuePresent;

    // ---- THE VRAM SYSTEM'S DOWN-CHAIN POINTERS AND DEVICE BINDING.
    g_nextFreeMemory        = (PFN_vkFreeMemory)nextGDPA(*out, "vkFreeMemory");
    g_nextBindImageMemory   = (PFN_vkBindImageMemory)nextGDPA(*out, "vkBindImageMemory");
    g_nextBindImageMemory2  = (PFN_vkBindImageMemory2)nextGDPA(*out, "vkBindImageMemory2");
    if (!g_nextBindImageMemory2)
        g_nextBindImageMemory2 = (PFN_vkBindImageMemory2)nextGDPA(*out, "vkBindImageMemory2KHR");
    g_nextBindBufferMemory  = (PFN_vkBindBufferMemory)nextGDPA(*out, "vkBindBufferMemory");
    g_nextBindBufferMemory2 = (PFN_vkBindBufferMemory2)nextGDPA(*out, "vkBindBufferMemory2");
    if (!g_nextBindBufferMemory2)
        g_nextBindBufferMemory2 = (PFN_vkBindBufferMemory2)nextGDPA(*out, "vkBindBufferMemory2KHR");
    g_nextGetDeviceQueue    = (PFN_vkGetDeviceQueue)nextGDPA(*out, "vkGetDeviceQueue");
    g_nextWaitForFences     = (PFN_vkWaitForFences)nextGDPA(*out, "vkWaitForFences");
    g_nextGetFenceStatus    = (PFN_vkGetFenceStatus)nextGDPA(*out, "vkGetFenceStatus");
    g_nextResetFences       = (PFN_vkResetFences)nextGDPA(*out, "vkResetFences");
    g_nextWaitSemaphores    = (PFN_vkWaitSemaphores)nextGDPA(*out, "vkWaitSemaphores");
    if (!g_nextWaitSemaphores)
        g_nextWaitSemaphores = (PFN_vkWaitSemaphores)nextGDPA(*out, "vkWaitSemaphoresKHR");
    g_nextQueueWaitIdle     = (PFN_vkQueueWaitIdle)nextGDPA(*out, "vkQueueWaitIdle");
    g_nextDeviceWaitIdle    = (PFN_vkDeviceWaitIdle)nextGDPA(*out, "vkDeviceWaitIdle");
    g_nextQueueBindSparse   = (PFN_vkQueueBindSparse)nextGDPA(*out, "vkQueueBindSparse");
    g_nextCmdCopyBuffer     = (PFN_vkCmdCopyBuffer)nextGDPA(*out, "vkCmdCopyBuffer");
    g_nextAllocDescSets     = (PFN_vkAllocateDescriptorSets)nextGDPA(*out, "vkAllocateDescriptorSets");
    g_nextMapMemory         = (PFN_vkMapMemory)nextGDPA(*out, "vkMapMemory");
    g_nextUnmapMemory       = (PFN_vkUnmapMemory)nextGDPA(*out, "vkUnmapMemory");
    PFN_vkSetDeviceMemoryPriorityEXT setPrio =
        (PFN_vkSetDeviceMemoryPriorityEXT)nextGDPA(*out, "vkSetDeviceMemoryPriorityEXT");

    // The transfer-ONLY family, if the device exposes one among the families
    // X-Plane created queues on. Only queues from that family are governed -
    // pacing the graphics queue would pace the frame itself.
    uint32_t transferOnly = ~0u;
    if (g_getPhysQueueFamProps) {
        uint32_t n = 0;
        g_getPhysQueueFamProps(phys, &n, nullptr);
        std::vector<VkQueueFamilyProperties> qf(n);
        if (n) g_getPhysQueueFamProps(phys, &n, qf.data());
        for (size_t i = 0; i < g_deviceFamilies.size(); ++i) {
            uint32_t fam = g_deviceFamilies[i];
            if (fam < n && (qf[fam].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
                !(qf[fam].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) {
                transferOnly = fam;
                break;
            }
        }
    }

    {
        VkPhysicalDeviceMemoryProperties vmp;
        memset(&vmp, 0, sizeof(vmp));
        if (g_getPhysMemProps) g_getPhysMemProps(phys, &vmp);
        vram::bindDevice(*out, phys, g_nextMemProps2, g_nextFreeMemory,
                         g_nextUnmapMemory, g_nextQueueSubmit, setPrio,
                         g_memoryPriority, g_pageableMemory, &vmp,
                         transferOnly);
        // The SS36 bindless verdict, performed rather than skipped: the
        // corpus (6855 modules) uses fixed bindings and zero descriptor
        // indexing; a bindless retrofit means rewriting every shader and the
        // material system behind them - an engine change, not a layer one.
        // The compatible parts of SS35 (recycling pressure, churn) are
        // measured every frame instead.
        trace("VRAMSYS: descriptor model - engine uses fixed bindings "
              "corpus-wide; bindless (SS36) verdict: incompatible from a "
              "layer, descriptor churn measured instead (SS35)");
    }


    // ---- FINISH dd AND REGISTER THE DEVICE, BEFORE ANYTHING CAN ASK ABOUT IT.
    //
    // This used to sit at the very end of the function, below the Streamline
    // block - so the comment there described a fix the code did not make. The
    // measurement that caught it, logged three ways at the call site:
    //
    //   vkCreatePrivateDataSlot - loader export yes, vkGetDeviceProcAddr NO,
    //                             EXT spelling NO, vkGetInstanceProcAddr yes
    //
    // Instance-level lookups answered because they do not come through us.
    // Device-level lookups did not, because they do: MV_GetDeviceProcAddr ends
    // in a g_devices lookup and returns nullptr for a device that is not in the
    // map yet. Streamline was asking a layer that had not finished admitting the
    // device exists.
    g_nextCreateSampler      = dd.createSampler;
    g_nextCmdBeginRenderPass = dd.cmdBeginRenderPass;
    g_nextCmdBeginRendering  = dd.cmdBeginRendering;
    dd.cmdEndRendering = (PFN_vkCmdEndRendering)nextGDPA(*out, "vkCmdEndRendering");
    if (!dd.cmdEndRendering)
        dd.cmdEndRendering = (PFN_vkCmdEndRendering)nextGDPA(*out, "vkCmdEndRenderingKHR");
    g_nextCmdEndRendering = dd.cmdEndRendering;
    g_nextCmdSetViewport     = dd.cmdSetViewport;

    // ---- THE GPU TIMING POOL.
    //
    // timestampPeriod converts ticks to nanoseconds and is per-device, so it
    // has to come from the physical device rather than be assumed. A device
    // that reports timestampComputeAndGraphics false cannot do this at all, and
    // saying so beats silently reporting zeroes.
    g_gpuTiming = gpuTimingOn();
    if (g_gpuTiming && dd.createQueryPool) {
        VkPhysicalDeviceProperties pdp;
        memset(&pdp, 0, sizeof(pdp));
        if (g_getPhysProps) g_getPhysProps(phys, &pdp);
        g_tsPeriodNs = pdp.limits.timestampPeriod;

        if (g_tsPeriodNs <= 0.0f) {
            trace("GPU TIME: this device reports no timestamp period - cannot "
                  "measure where the frame goes");
            g_gpuTiming = false;
        } else {
            VkQueryPoolCreateInfo qpi;
            memset(&qpi, 0, sizeof(qpi));
            qpi.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            qpi.queryType  = VK_QUERY_TYPE_TIMESTAMP;
            qpi.queryCount = 16;
            if (dd.createQueryPool(*out, &qpi, nullptr, &g_tsPool) == VK_SUCCESS) {
                g_tsCapacity = 16;
                trace("GPU TIME: on - %.2f ns per tick. Spans are reported every "
                      "300 frames as real device time, which is the only kind "
                      "that means anything here.", (double)g_tsPeriodNs);
            } else {
                g_gpuTiming = false;
            }
        }
    }

    trace("CreateDevice ok: present=%p sampler=%p beginRP=%p beginRendering=%p setViewport=%p",
          (void*)dd.queuePresent, (void*)dd.createSampler,
          (void*)dd.cmdBeginRenderPass, (void*)dd.cmdBeginRendering,
          (void*)dd.cmdSetViewport);

    // The reserved queues, now that there is a device to fetch them from. This
    // also catches the case where creation fell back to X-Plane's original
    // request: on that device the reserved indices address queues that were
    // never created, and fgBindQueues turns that into a log line instead of a
    // driver fault on the first submit.

    // Scoped, NOT function-scoped. The Streamline block below takes g_lock for
    // its own instance lookup, and a second lock_guard on a non-recursive mutex
    // already held by this thread is a deadlock, not an error.
    {
        std::lock_guard<std::mutex> g(g_lock);
        g_devices[dispatchKey(*out)] = dd;
    }

    // ---- CRASH DESTRUCTION RESOURCES, HERE AND NOT AT FRAME TIME.
    //
    // Pipeline layouts are created during startup, long before the first frame
    // renders. Creating the descriptor set layout on the per-frame path meant
    // it did not exist when the layouts that need it were built, so every one
    // of them would have been silently left unextended - and the symptom would
    // have been "displacement does nothing", days later, with nothing to
    // suggest an ordering problem.
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, DeviceData>::iterator di = g_devices.find(dispatchKey(*out));
        if (crashEnabled() && di != g_devices.end())
            destructgpu::ensure(di->second, *out, g_maxBoundSets);
    }


    return r;
}

// ------------------------------------------------------------- proc addr

#define RETURN_IF(nm, fn) if (!strcmp(name, nm)) return (PFN_vkVoidFunction)(fn);

// The 38fps serialization instrument: X-Plane's frame time is CPU+GPU added,
// never overlapped, and the present mode is where that lives. Log what the
// engine asks for; TAA_PRESENT_MODE=<n> forces it (0 IMMEDIATE, 1 MAILBOX,
// 2 FIFO, 3 FIFO_RELAXED) - MAILBOX is the "uncap without tearing" test.
static VKAPI_ATTR VkResult VKAPI_CALL Layer_CreateSwapchainKHR(
    VkDevice device, const VkSwapchainCreateInfoKHR *ci,
    const VkAllocationCallbacks *alloc, VkSwapchainKHR *out)
{
    VkSwapchainCreateInfoKHR mod = *ci;
    const char *e = getenv("TAA_PRESENT_MODE");
    if (e && e[0]) {
        mod.presentMode = (VkPresentModeKHR)atoi(e);
        trace("SWAPCHAIN: present mode FORCED %d -> %d", (int)ci->presentMode,
              (int)mod.presentMode);
    }

    // ---- MAILBOX WITH TWO IMAGES IS A STALL, AND IT IS THE 38 FPS.
    //
    // Measured in flight: vkQueuePresentKHR blocks 8-14 ms of a 26 ms frame and
    // the CPU sits in it for 31-54% of wall clock, while nothing waits on a
    // fence or a semaphore. The swapchain explains it - X-Plane asks for
    // MAILBOX with minImageCount 2.
    //
    // MAILBOX only pipelines with THREE: one on screen, one queued for the next
    // scan-out, one being rendered. With two there is no spare, so the CPU
    // cannot hand over a frame until the display releases the image it is
    // showing, and the frame time becomes render time PLUS a wait for the
    // panel. That is also why the total looked like CPU + GPU added together
    // and why the ceiling never moved with resolution, textures, plugins or
    // this layer: none of them change how many images the swapchain has.
    //
    // Asking for one more is a request, not a demand - the driver clamps to
    // what the surface supports - and if the create fails for any reason the
    // original is retried unchanged below, so the worst case is today's
    // behaviour. TAA_SWAP_IMAGES pins it (0 leaves the engine's choice alone).
    uint32_t wantImages = 3;
    if (const char *si = getenv("TAA_SWAP_IMAGES")) wantImages = (uint32_t)atoi(si);
    if (wantImages && mod.minImageCount < wantImages &&
        (mod.presentMode == VK_PRESENT_MODE_MAILBOX_KHR ||
         mod.presentMode == VK_PRESENT_MODE_FIFO_KHR ||
         mod.presentMode == VK_PRESENT_MODE_FIFO_RELAXED_KHR)) {
        trace("SWAPCHAIN: raising minImageCount %u -> %u for present mode %d - "
              "two images cannot pipeline, so present blocks on the display "
              "(measured 8-14 ms per frame)",
              mod.minImageCount, wantImages, (int)mod.presentMode);
        mod.minImageCount = wantImages;
    }
    trace("SWAPCHAIN: %ux%u images=%u presentMode=%d (0=IMM 1=MBOX 2=FIFO "
          "3=RELAXED) fmt=%d",
          mod.imageExtent.width, mod.imageExtent.height, mod.minImageCount,
          (int)mod.presentMode, (int)mod.imageFormat);
    PFN_vkCreateSwapchainKHR next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, DeviceData>::iterator it =
            g_devices.find(dispatchKey(device));
        if (it != g_devices.end()) next = it->second.createSwapchainKHR;
    }
    if (!next) return VK_ERROR_INITIALIZATION_FAILED;
    VkResult scr = next(device, &mod, alloc, out);
    if (scr != VK_SUCCESS && mod.minImageCount != ci->minImageCount) {
        // Never trade a working swapchain for a faster one.
        trace("SWAPCHAIN: create failed (%d) with %u images - retrying with the "
              "engine's original %u", (int)scr, mod.minImageCount,
              ci->minImageCount);
        mod.minImageCount = ci->minImageCount;
        scr = next(device, &mod, alloc, out);
    }
    return scr;
}

extern "C" VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
MV_GetDeviceProcAddr(VkDevice device, const char *name)
{
    RETURN_IF("vkCreateSwapchainKHR",  Layer_CreateSwapchainKHR)
    RETURN_IF("vkGetDeviceProcAddr",   MV_GetDeviceProcAddr)
    RETURN_IF("vkDestroyDevice",       Layer_DestroyDevice)
    RETURN_IF("vkCreateImage",         Layer_CreateImage)
    RETURN_IF("vkCreateImageView",     Layer_CreateImageView)
    RETURN_IF("vkDestroyImage",        Layer_DestroyImage)
    RETURN_IF("vkCreateBuffer",        Layer_CreateBuffer)
    RETURN_IF("vkDestroyBuffer",       Layer_DestroyBuffer)
    RETURN_IF("vkQueuePresentKHR",     Layer_QueuePresentKHR)
    RETURN_IF("vkGetSwapchainImagesKHR", Layer_GetSwapchainImagesKHR)
    RETURN_IF("vkCmdBlitImage",        Layer_CmdBlitImage)
    RETURN_IF("vkCmdResolveImage",     Layer_CmdResolveImage)
    RETURN_IF("vkUpdateDescriptorSets", Layer_UpdateDescriptorSets)
    RETURN_IF("vkCmdBindDescriptorSets", Layer_CmdBindDescriptorSets)
    RETURN_IF("vkCmdCopyImage",        Layer_CmdCopyImage)
    RETURN_IF("vkQueueSubmit",         Layer_QueueSubmit)
    RETURN_IF("vkCreateSampler",       Layer_CreateSampler)
    RETURN_IF("vkCmdBeginRenderPass",  Layer_CmdBeginRenderPass)
    RETURN_IF("vkCmdBeginRendering",    Layer_CmdBeginRendering)
    RETURN_IF("vkEndCommandBuffer",     Layer_EndCommandBuffer)
    RETURN_IF("vkAllocateCommandBuffers", Layer_AllocateCommandBuffers)
    RETURN_IF("vkCreateCommandPool",      Layer_CreateCommandPool)
    RETURN_IF("vkCmdBeginRenderingKHR", Layer_CmdBeginRendering)
    RETURN_IF("vkCmdEndRendering",      Layer_CmdEndRendering)
    RETURN_IF("vkCmdEndRenderingKHR",   Layer_CmdEndRendering)
    RETURN_IF("vkCmdSetViewport",       Layer_CmdSetViewport)
    // ---- HOOK-LEVEL KILL SWITCH for the crash bisect: TAA_VRAM_HOOKS=0
    // removes every VRAM-system hook from dispatch entirely - the engine
    // talks straight to the driver, and only the pre-existing layer remains.
    // Fast default-aircraft loads crash where slow FF loads survive with all
    // LIVE switches off, so the suspect is an always-on hook path racing the
    // load; this cleaves the binary at that exact boundary.
    static int vramHooks = -1;
    if (vramHooks < 0) {
        const char *vh = getenv("TAA_VRAM_HOOKS");
        vramHooks = (!vh || atoi(vh) != 0) ? 1 : 0;
    }
    // TAA_VRAM_HOOK_MASK bisects within the set (default 15 = all):
    //   bit0 memory (free/bind/map)   bit1 sync (fences/semaphores/idles)
    //   bit2 sparse (QueueBindSparse) bit3 copy+descriptors
    // The Felis-load crash lives behind ONE of these bits.
    static int vramMask = -1;
    if (vramMask < 0) {
        const char *vm = getenv("TAA_VRAM_HOOK_MASK");
        vramMask = vm ? atoi(vm) : 15;
    }
    if (vramHooks) {
    if (vramMask & 1) {
    RETURN_IF("vkFreeMemory",          Vram_FreeMemory)
    RETURN_IF("vkBindImageMemory",     Vram_BindImageMemory)
    RETURN_IF("vkBindImageMemory2",    Vram_BindImageMemory2)
    RETURN_IF("vkBindImageMemory2KHR", Vram_BindImageMemory2)
    RETURN_IF("vkBindBufferMemory",    Vram_BindBufferMemory)
    RETURN_IF("vkBindBufferMemory2",   Vram_BindBufferMemory2)
    RETURN_IF("vkBindBufferMemory2KHR", Vram_BindBufferMemory2)
    RETURN_IF("vkMapMemory",           Vram_MapMemory)
    RETURN_IF("vkUnmapMemory",         Vram_UnmapMemory)
    }
    if (vramMask & 2) {
    RETURN_IF("vkGetDeviceQueue",      Vram_GetDeviceQueue)
    RETURN_IF("vkWaitForFences",       Vram_WaitForFences)
    RETURN_IF("vkGetFenceStatus",      Vram_GetFenceStatus)
    RETURN_IF("vkResetFences",         Vram_ResetFences)
    RETURN_IF("vkWaitSemaphores",      Vram_WaitSemaphores)
    RETURN_IF("vkWaitSemaphoresKHR",   Vram_WaitSemaphores)
    RETURN_IF("vkQueueWaitIdle",       Vram_QueueWaitIdle)
    RETURN_IF("vkDeviceWaitIdle",      Vram_DeviceWaitIdle)
    }
    if (vramMask & 4) {
    RETURN_IF("vkQueueBindSparse",     Vram_QueueBindSparse)
    }
    if (vramMask & 8) {
    RETURN_IF("vkCmdCopyBuffer",       Vram_CmdCopyBuffer)
    RETURN_IF("vkAllocateDescriptorSets", Vram_AllocateDescriptorSets)
    }
    }
    // Everything below is the PRE-EXISTING layer - never gated. AllocateMemory
    // stays hooked for the ledger and overcommit rescue; its new VRAM extras
    // are internally inert when the hooks above are stripped.
    RETURN_IF("vkAllocateMemory",      TAA_AllocateMemory)
    RETURN_IF("vkCmdCopyBufferToImage", TAA_CmdCopyBufferToImage)
    RETURN_IF("vkCmdPipelineBarrier",  TAA_CmdPipelineBarrier)
    RETURN_IF("vkCmdPipelineBarrier2", TAA_CmdPipelineBarrier2)
    RETURN_IF("vkCmdPipelineBarrier2KHR", TAA_CmdPipelineBarrier2)
    RETURN_IF("vkCmdBlitImage",        TAA_CmdBlitImage)
    RETURN_IF("vkCmdCopyImage",        TAA_CmdCopyImage)
    RETURN_IF("vkCmdClearColorImage",  TAA_CmdClearColorImage)
    RETURN_IF("vkCreateShaderModule",   TAA_CreateShaderModule)
    RETURN_IF("vkCreateComputePipelines", TAA_CreateComputePipelines)
    RETURN_IF("vkCmdDispatch",          TAA_CmdDispatch)
    RETURN_IF("vkCmdPushDescriptorSetKHR", TAA_CmdPushDescriptorSetKHR)
    // Vulkan 1.4 promoted push descriptors to core and dropped the suffix.
    // X-Plane reports a 1.4 device, so it resolves the CORE name and the KHR
    // hook above never fired - which is why the FSR output stayed unnamed.
    RETURN_IF("vkCmdPushDescriptorSet",    TAA_CmdPushDescriptorSetKHR)
    RETURN_IF("vkCmdPushDescriptorSet2",    TAA_CmdPushDescriptorSet2)
    RETURN_IF("vkCmdPushDescriptorSet2KHR", TAA_CmdPushDescriptorSet2)
    RETURN_IF("vkCreatePipelineLayout", TAA_CreatePipelineLayout)
    RETURN_IF("vkCmdBindPipeline",     TAA_CmdBindPipeline)
    RETURN_IF("vkCmdDraw",             TAA_CmdDraw)
    RETURN_IF("vkCmdDrawIndexed",      TAA_CmdDrawIndexed)
    RETURN_IF("vkCmdPushConstants",    TAA_CmdPushConstants)
    RETURN_IF("vkCmdDrawIndirect",            TAA_CmdDrawIndirect)
    RETURN_IF("vkCmdDrawIndexedIndirect",     TAA_CmdDrawIndexedIndirect)
    RETURN_IF("vkCmdDrawIndirectCount",       TAA_CmdDrawIndirectCount)
    RETURN_IF("vkCmdDrawIndexedIndirectCount", TAA_CmdDrawIndexedIndirectCount)
    RETURN_IF("vkCreateGraphicsPipelines", TAA_CreateGraphicsPipelines)

    PFN_vkGetDeviceProcAddr next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, DeviceData>::iterator it = g_devices.find(dispatchKey(device));
        if (it != g_devices.end()) next = it->second.gdpa;
    }
    return next ? next(device, name) : nullptr;
}

// ---- KEEP VALIDATION MESSAGES OUT OF THE SIM'S MODAL DIALOG.
//
// X-Plane registers its own debug messenger and raises a blocking X-System
// Message box for every error the validation layer reports. That is fine for
// one message and unusable for a per-frame synchronisation hazard, which is
// exactly what we turned validation on to find.
//
// Refusing to pass the messenger down leaves the sim without a callback, so
// validation reports go only to the log file the settings name. A cooked
// handle is returned so the sim believes it succeeded; the matching destroy
// swallows it rather than handing a fabricated pointer to the loader.
//
// Only while TAA_SILENCE_APP_VALIDATION is set - the sim's own error reporting
// is worth having the rest of the time.
static bool silenceAppValidation()
{
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("TAA_SILENCE_APP_VALIDATION");
        on = (e && atoi(e)) ? 1 : 0;
    }
    return on != 0;
}

// A recognisable, never-allocated value. Only ever compared, never dereferenced
// and never handed to the loader.
#define TAA_FAKE_MESSENGER ((VkDebugUtilsMessengerEXT)(uintptr_t)0xFA15EDBEEFULL)

static VKAPI_ATTR VkResult VKAPI_CALL TAA_CreateDebugUtilsMessengerEXT(
    VkInstance inst, const VkDebugUtilsMessengerCreateInfoEXT *ci,
    const VkAllocationCallbacks *alloc, VkDebugUtilsMessengerEXT *out)
{
    if (silenceAppValidation()) {
        if (out) *out = TAA_FAKE_MESSENGER;
        trace("VALIDATION: withheld the sim's debug messenger - reports go to "
              "the validation log only, not to a modal dialog per frame.");
        return VK_SUCCESS;
    }
    PFN_vkCreateDebugUtilsMessengerEXT next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, InstanceData>::iterator it = g_instances.find(dispatchKey(inst));
        if (it != g_instances.end() && it->second.gipa)
            next = (PFN_vkCreateDebugUtilsMessengerEXT)
                       it->second.gipa(inst, "vkCreateDebugUtilsMessengerEXT");
    }
    return next ? next(inst, ci, alloc, out) : VK_ERROR_EXTENSION_NOT_PRESENT;
}

static VKAPI_ATTR void VKAPI_CALL TAA_DestroyDebugUtilsMessengerEXT(
    VkInstance inst, VkDebugUtilsMessengerEXT m, const VkAllocationCallbacks *alloc)
{
    if (m == TAA_FAKE_MESSENGER) return;   // never existed below us
    PFN_vkDestroyDebugUtilsMessengerEXT next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, InstanceData>::iterator it = g_instances.find(dispatchKey(inst));
        if (it != g_instances.end() && it->second.gipa)
            next = (PFN_vkDestroyDebugUtilsMessengerEXT)
                       it->second.gipa(inst, "vkDestroyDebugUtilsMessengerEXT");
    }
    if (next) next(inst, m, alloc);
}

extern "C" VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
MV_GetInstanceProcAddr(VkInstance inst, const char *name)
{
    RETURN_IF("vkCreateDebugUtilsMessengerEXT",  TAA_CreateDebugUtilsMessengerEXT)
    RETURN_IF("vkDestroyDebugUtilsMessengerEXT", TAA_DestroyDebugUtilsMessengerEXT)
    RETURN_IF("vkGetInstanceProcAddr", MV_GetInstanceProcAddr)
    RETURN_IF("vkCreateInstance",      TAA_CreateInstance)
    RETURN_IF("vkDestroyInstance",     TAA_DestroyInstance)
    RETURN_IF("vkCreateDevice",        TAA_CreateDevice)
    RETURN_IF("vkGetDeviceProcAddr",   MV_GetDeviceProcAddr)

    // Both spellings. The KHR alias is what an application targeting Vulkan 1.0
    // with VK_KHR_get_physical_device_properties2 will ask for, and hooking only
    // the core name would miss it entirely - the layer would look installed and
    // simply never see the query.
    RETURN_IF("vkGetPhysicalDeviceMemoryProperties2",
              TAA_GetPhysicalDeviceMemoryProperties2)
    RETURN_IF("vkGetPhysicalDeviceMemoryProperties2KHR",
              TAA_GetPhysicalDeviceMemoryProperties2)

    PFN_vkGetInstanceProcAddr next = nullptr;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<void*, InstanceData>::iterator it = g_instances.find(dispatchKey(inst));
        if (it != g_instances.end()) next = it->second.gipa;
    }
    return next ? next(inst, name) : nullptr;
}
