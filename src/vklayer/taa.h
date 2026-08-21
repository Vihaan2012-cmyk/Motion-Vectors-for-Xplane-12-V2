#pragma once

// TEMPORAL ANTIALIASING - THE VULKAN SIDE.
//
// Built to be falsifiable in stages. The predecessor was removed because a
// corrupt frame could not be attributed: plumbing and algorithm failed together
// and neither could be cleared. Here TAA_MODE selects how far the shader runs -
// 0 passthrough, 1 reproject, 2 full - so each stage is a separate experiment
// against the same wiring:
//
//   mode 0  every binding, barrier and dispatch exercised, output = input.
//           Any visible change is a plumbing fault, full stop.
//   mode 1  history fetched and blended. Ghosting along motion is EXPECTED and
//           means reprojection works.
//   mode 2  neighbourhood clamp on top. Ghosting should collapse.
//
// The history image is ours: created here, owned here, and explicitly cleared
// on first use. Uninitialised history is the likeliest source of the magenta
// frame the old resolve produced, and a clear costs one command.

#include <stdint.h>
#include <string.h>
#include <map>
#include "taa_spv.h"
#include "temporal.h"

struct TaaState {
    VkDevice        device      = VK_NULL_HANDLE;
    // ---- PING-PONG. ONE HISTORY IMAGE IS A DATA RACE.
    //
    // The first version bound a single history image as a storage image
    // (written) and a sampler (read) in the SAME dispatch. Workgroups read
    // while others wrote, so every frame sampled partially-written data and the
    // error compounded - the picture filled with black over about five seconds
    // of camera movement, slow enough to look like a blend-weight problem and
    // fast enough to destroy the image.
    //
    // Read last frame's, write this frame's, swap. This is not an optimisation;
    // a single buffer has no defined contents to read.
    VkImage         history[2]     = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDeviceMemory  historyMem[2]  = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkImageView     historyView[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    uint32_t        historyWrite   = 0;   // index written this frame
    // ---- ONE VIEW PER SCENE IMAGE, CACHED, NEVER TORN DOWN MID-SESSION.
    //
    // X-Plane alternates between two HDR scene targets. Keying re-initialisation
    // on the image handle therefore rebuilt the pipeline, pool and views EVERY
    // frame, destroying objects that in-flight command buffers still referenced
    // - which crashed the sim, and showed up as a dispatch counter that never
    // got past 1 because each rebuild reset it.
    //
    // The history is ours and is unaffected by which target the frame drew
    // into, so only the scene view has to follow. Cached here and destroyed
    // with everything else.
    std::map<VkImage, VkImageView> sceneViews;
    VkImageView     sceneView   = VK_NULL_HANDLE;
    VkImage         sceneImage  = VK_NULL_HANDLE;   // the view above belongs to this
    VkImageView     velView     = VK_NULL_HANDLE;
    // Generation of the velocity target the descriptor was written against.
    // Handles are reused; this is not. See MvTarget::gen.
    uint64_t        velGen      = 0;
    VkSampler       sampler     = VK_NULL_HANDLE;
    // NEAREST, for the integer flags image - see the note at its creation.
    VkSampler       samplerNearest = VK_NULL_HANDLE;
    // ---- X-PLANE'S gbuffer_vel, AND A FALLBACK FOR WHEN IT IS UNKNOWN.
    //
    // The flags view is over an image X-Plane owns, identified by shape by the
    // layer's census; views over it are cached like the scene views. The
    // fallback is a 1x1 zero uint image of our own: Vulkan requires every
    // statically-used binding to be valid even behind a branch, and a zero
    // flag word reads as bit-2-clear, which makes the fallback a no-op.
    std::map<VkImage, VkImageView> flagsViews;
    VkImageView     flagsView   = VK_NULL_HANDLE;   // what binding 4 gets
    bool            flagsValid  = false;            // true only for the real image
    VkImage         flagsFallback     = VK_NULL_HANDLE;
    VkDeviceMemory  flagsFallbackMem  = VK_NULL_HANDLE;
    VkImageView     flagsFallbackView = VK_NULL_HANDLE;

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout      pipeLayout = VK_NULL_HANDLE;
    VkPipeline            pipeline   = VK_NULL_HANDLE;
    VkDescriptorPool      pool       = VK_NULL_HANDLE;

    // A ring, not one set. The predecessor exhausted a single-set pool after
    // eight frames and stopped silently - the resolve simply stopped running
    // and nothing said so.
    static const uint32_t kSets = 8;
    VkDescriptorSet sets[kSets] = { VK_NULL_HANDLE };
    uint32_t        nextSet = 0;

    uint32_t w = 0, h = 0;
    // Array layers of the scene target, and therefore of our history. One for a
    // plain target, two for stereo. Every view is created as 2D_ARRAY over this
    // many layers, so the single-layer case needs no separate path.
    uint32_t layers = 1;
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool     ready = false;
    bool     historyCleared = false;
    uint64_t dispatches = 0;

    // ---- DIRECT HISTORY READBACK.
    //
    // Every claim about whether history accumulates has so far been inferred
    // from the composited screen, and viz=4 cannot settle it because the
    // visualisation is written INTO the history image - the measurement
    // contaminates the thing measured. This copies a strip of the history
    // image itself into host memory, so the bytes can be compared across
    // frames without passing through the resolve's own output path.
    //
    // A strip, not the frame: 512 texels of RGBA16F is 4 KB, enough for a
    // stable statistic and small enough that the copy cannot meaningfully
    // perturb the timing it is measuring.
    VkBuffer        readBuf  = VK_NULL_HANDLE;
    VkDeviceMemory  readMem  = VK_NULL_HANDLE;
    void           *readPtr  = nullptr;
    uint64_t        readFrame = 0;
};

static TaaState g_taa;

// Armed once per PRESENT by the present hook, consumed by the first resolve of
// that frame. The history ping-pong has to follow displayed frames, not the
// number of times the resolve happens to record - see the note at the flip.
static bool g_taaFlipArmed = true;

struct TaaPush {
    float invSizeX, invSizeY;
    float jitterX, jitterY;
    float alpha;
    int32_t mode;
    int32_t reset;
    // Whether the camera moved this frame. The shader needs it to read a ZERO
    // velocity correctly: our attachment is cleared and written only by the
    // vertex shaders we patched, so at a sky or cloud pixel zero means "nothing
    // wrote here", not "this pixel is stationary". With the camera still, zero
    // means static and history is perfect; with it moving, zero means the pixel
    // cannot be reprojected at all. Same word, opposite treatment.
    int32_t cameraMoved;
    // Debug visualisation, and the switches that remove one input at a time.
    // Packed as a bitmask rather than four ints because the block is pushed
    // every frame and 16 bytes of padding is 16 bytes of nothing.
    int32_t viz;
    float   vizScale;
    float   gain;
    float   varClip;
    int32_t flags;      // see kTaaFlag* below
    // Live A/B knobs for the reprojection convention - see the shader's note.
    float   velScale;
    float   velYSign;
    // 1 when binding 4 is X-Plane's real gbuffer_vel rather than the fallback.
    int32_t flagsValid;
    // ---- THE UNJITTER SHIFT, AS TWO NUMBERS RATHER THAN A CONVICTION.
    //
    // S = (sMulX * jitter.x, sMulY * jitter.y). The hardcoded (-0.5, +0.5)
    // encodes two conventions at once - NDC-to-UV is a half, and the viewport
    // height is negative so Y flips - and getting either wrong turns the
    // cancellation into a doubling. That is indistinguishable from "no
    // cancellation" by eye, and it scales with jitter amplitude, which is
    // exactly the symptom: shake at jitter_scale=1, none at 0. Both have been
    // wrong in this file before, so they are swept, not argued.
    //
    // MEASURED, parked aircraft, still camera, temporal mean absolute
    // deviation over 8 frames of the same ground:
    //     jitter off .................. 0.22   (the floor)
    //     (+0.5, -0.5) ................ 2.94   <- default
    //     no cancellation at all ...... 4.34
    //     (-0.5, +0.5), as shipped .... 5.93
    // The shipped pair was WORSE than doing nothing: it added the
    // displacement instead of removing it.
    float   sMulX;
    float   sMulY;
    // Largest |velocity|, in UV, that is believed. Live so the magnitude of
    // whatever is poisoning history can be found by bisection rather than
    // guessed: accumulation returns at whatever threshold excludes it.
    float   velMax;
    // Coverage below this counts as "nobody wrote here". Negative disables the
    // unwritten-pixel rejection entirely, which is how its cost is measured.
    float   novecCov;
    // Blend weight for a pixel that never received a vector. 1.0 is the old
    // hard rejection (shake); pc.alpha is keeping it forever (crawl); between
    // the two the stale history decays instead of being kept or thrown away.
    float   novecAlpha;
    float   movedDead;
};

enum {
    kTaaFlagFreezeHistory = 1 << 0,
    kTaaFlagNoMotion      = 1 << 1,
    kTaaFlagNoAccum       = 1 << 2,
    kTaaFlagReactive      = 1 << 3,
    kTaaFlagNoUnjitter    = 1 << 4,
    kTaaFlagNoVecByVel    = 1 << 6,
};

// ---- EVERY KNOB IS LIVE. NONE OF THESE ARE CACHED.
//
// They used to be `static const` initialised from getenv, which is why changing
// the alpha meant restarting the sim. Reading them per frame costs a map lookup
// against a table that is only rebuilt when the control file's timestamp moves,
// and buys the ability to answer a question in the ten seconds it takes to save
// a file instead of the four minutes it takes to relaunch.
static bool  taaEnabled()  { return live::onoff("taa.enable", "TAA_RESOLVE", false); }
static int   taaMode()     { return live::i("taa.mode",  "TAA_MODE",  0); }
static float taaAlpha()    { return live::f("taa.alpha", "TAA_ALPHA", 0.1f); }
static float taaGain()     { return live::f("taa.gain",  "TAA_GAIN",  4.0f); }
static float taaVarClip()  { return live::f("taa.varclip", "TAA_VARCLIP", 1.25f); }
// Deadband on the clamp correction, in units of the noise floor. 1.0 makes a
// correction at or below the floor read as zero, which is what the shader's
// floorS note asks for; 0.0 restores the old behaviour that pinned a at 1.0.
static float taaMovedDead(){ return live::f("taa.moved_dead", "TAA_MOVED_DEAD", 0.0f); }
static int   taaViz()      { return live::i("taa.viz",   "TAA_VIZ",   0); }
static float taaVizScale() { return live::f("taa.viz_scale", nullptr, 1.0f); }

// ---- REMOVE ONE INPUT AT A TIME.
//
// The generalisation of MODE_PASSTHROUGH to every stage, and the reason it is
// worth having as separate switches rather than one debug mode: each one makes a
// DIFFERENT prediction, so the observation attributes the fault rather than
// merely changing the picture.
//
//   freeze_history   the image should freeze and smear along motion. If it does
//                    not, what is on screen is not the history.
//   no_motion        every vector reads zero, so reprojection is a same-pixel
//                    fetch. Ghosting that SURVIVES this is not the vectors.
//   no_accum         current frame out, every binding and barrier still live.
//   force_reset      same output as no_accum but reached through the reset path,
//                    so the pair separates "reset is broken" from "accumulation
//                    is broken" - which looked identical for two builds.
static float taaVelScale() { return live::f("taa.vel_scale", nullptr, 1.0f); }
// -1.0 is the shipping belief (negative-height viewport => d(uv_y) = -vel_y).
static float taaVelYSign() { return live::onoff("taa.vel_ypos", nullptr, false) ? 1.0f : -1.0f; }
// The gbuffer_vel weight override - on by default because everything it
// addresses (prop halo, airframe streaks, cockpit shake) is worse than the
// aliasing it reintroduces on the airframe. taa.objflags=0 turns it off live
// so its exact visual contribution can be isolated in one edit.
static bool taaObjFlags() { return live::onoff("taa.objflags", nullptr, true); }
// The C14 reactive mask - on by default for the same reason as the flag
// override: flicker parked in history is worse than aliasing on the flickering
// content. taa.reactive=0 isolates its contribution live.
static bool taaReactive() { return live::onoff("taa.reactive", nullptr, true); }
// The unjitter alignment - isolation knob for the aligned sampling, so its
// contribution can be removed live without touching the jitter itself.
static bool taaUnjitter() { return live::onoff("taa.unjitter", nullptr, true); }
static bool taaFreezeHistory() { return live::onoff("taa.freeze_history", nullptr, false); }
static bool taaNoMotion()      { return live::onoff("taa.no_motion",      nullptr, false); }
static bool taaNoAccum()       { return live::onoff("taa.no_accum",       nullptr, false); }
static bool taaForceReset()    { return live::onoff("taa.force_reset",    nullptr, false); }

static uint32_t taaFindMemory(DeviceData &dd, uint32_t typeBits, VkMemoryPropertyFlags want)
{
    VkPhysicalDeviceMemoryProperties mp;
    memset(&mp, 0, sizeof(mp));
    if (!g_getPhysMemProps || dd.phys == VK_NULL_HANDLE) return UINT32_MAX;
    g_getPhysMemProps(dd.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

// Destroy a parked state's objects for real. Only ever called on states that
// left service N presents ago - nothing in flight can still reference them.
static void taaDestroyState(DeviceData &dd, TaaState &g_taa)
{
    if (g_taa.pipeline)    dd.destroyPipeline(g_taa.device, g_taa.pipeline, nullptr);
    if (g_taa.pipeLayout)  dd.destroyPipelineLayout(g_taa.device, g_taa.pipeLayout, nullptr);
    if (g_taa.setLayout)   dd.destroyDescriptorSetLayout(g_taa.device, g_taa.setLayout, nullptr);
    if (g_taa.pool)        dd.destroyDescriptorPool(g_taa.device, g_taa.pool, nullptr);
    if (g_taa.sampler)     dd.destroySampler(g_taa.device, g_taa.sampler, nullptr);
    if (g_taa.samplerNearest) dd.destroySampler(g_taa.device, g_taa.samplerNearest, nullptr);
    for (int i = 0; i < 2; ++i)
        if (g_taa.historyView[i]) dd.destroyImageView(g_taa.device, g_taa.historyView[i], nullptr);
    for (std::map<VkImage, VkImageView>::iterator it = g_taa.sceneViews.begin();
         it != g_taa.sceneViews.end(); ++it)
        if (it->second) dd.destroyImageView(g_taa.device, it->second, nullptr);
    g_taa.sceneViews.clear();
    // velView is g_mv.viewArray - g_mv owns and destroys it; destroying it
    // here too was a latent double-destroy.
    for (std::map<VkImage, VkImageView>::iterator it = g_taa.flagsViews.begin();
         it != g_taa.flagsViews.end(); ++it)
        if (it->second) dd.destroyImageView(g_taa.device, it->second, nullptr);
    g_taa.flagsViews.clear();
    if (g_taa.flagsFallbackView) dd.destroyImageView(g_taa.device, g_taa.flagsFallbackView, nullptr);
    if (g_taa.flagsFallback)     dd.destroyImage(g_taa.device, g_taa.flagsFallback, nullptr);
    if (g_taa.flagsFallbackMem)  dd.freeMemory(g_taa.device, g_taa.flagsFallbackMem, nullptr);
    for (int i = 0; i < 2; ++i) {
        if (g_taa.history[i])    dd.destroyImage(g_taa.device, g_taa.history[i], nullptr);
        if (g_taa.historyMem[i]) dd.freeMemory(g_taa.device, g_taa.historyMem[i], nullptr);
    }
    TaaState fresh;
    fresh.device = g_taa.device;
    g_taa = fresh;
}

// Teardown = park, not destroy. Resolves recorded 1-2 frames ago still
// reference these objects; destroying them under the GPU was the teardown
// DEVICE_LOST, and deviceWaitIdle here races X-Plane's submit threads.
// The graveyard holds each retired state until 8 presents have passed.
struct TaaGrave { TaaState s; uint64_t frame; };
static std::vector<TaaGrave> g_taaGraves;
static uint64_t g_taaGraveNow = 0;

static void taaDestroy(DeviceData &dd)
{
    (void)dd;
    if (g_taa.ready || g_taa.pool) {
        TaaGrave gr; gr.s = g_taa; gr.frame = g_taaGraveNow;
        g_taaGraves.push_back(gr);
    }
    TaaState fresh;
    fresh.device = g_taa.device;
    g_taa = fresh;
}

// Called once per present with the current frame counter.
static void taaGraveFlush(DeviceData &dd, uint64_t frame)
{
    g_taaGraveNow = frame;
    for (size_t i = 0; i < g_taaGraves.size();) {
        if (frame > g_taaGraves[i].frame + 8) {
            taaDestroyState(dd, g_taaGraves[i].s);
            g_taaGraves.erase(g_taaGraves.begin() + (long)i);
        } else ++i;
    }
}

// Build everything that depends on the scene target's size and format. Called
// again whenever either changes, which is why teardown comes first.
static bool taaInit(DeviceData &dd, VkDevice dev, VkImage scene, VkFormat fmt,
                    uint32_t w, uint32_t h, uint32_t layers, VkImageView velView)
{
    g_taa.device = dev;
    taaDestroy(dd);
    g_taa.device = dev;
    g_taa.w = w; g_taa.h = h; g_taa.format = fmt;
    g_taa.layers = layers ? layers : 1;
    g_taa.sceneImage = scene;
    g_taa.velView = VK_NULL_HANDLE;

    // ---- History image, same format as the scene so no conversion is implied.
    VkImageCreateInfo ici;
    memset(&ici, 0, sizeof(ici));
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent.width = w; ici.extent.height = h; ici.extent.depth = 1;
    ici.mipLevels = 1; ici.arrayLayers = g_taa.layers;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    // TRANSFER_SRC because the resolve's OUTPUT PATH reads from here: history
    // is copied into the scene target every frame. Without it that copy is
    // illegal, and validation says so twenty times a run:
    //
    //   VUID-vkCmdCopyImage-aspect-06662
    //   srcImage was created with TRANSFER_DST|SAMPLED|STORAGE but requires
    //   VK_IMAGE_USAGE_2_TRANSFER_SRC_BIT
    //
    // TRANSFER_DST was here for the initial clear and nobody added the SRC
    // side when the copy-back was introduced. The image is written, the
    // accumulation happens - viz=4 shows a real accumulated picture - and the
    // step that carries it to the screen is undefined behaviour.
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    for (int hi = 0; hi < 2; ++hi) {
        if (dd.createImage(dev, &ici, nullptr, &g_taa.history[hi]) != VK_SUCCESS) {
            trace("TAA: history image %d creation failed (%ux%u fmt=%d)", hi, w, h, (int)fmt);
            return false;
        }
        VkMemoryRequirements mr;
        dd.getImageMemReq(dev, g_taa.history[hi], &mr);
        VkMemoryAllocateInfo mai;
        memset(&mai, 0, sizeof(mai));
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = taaFindMemory(dd, mr.memoryTypeBits,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX ||
            dd.allocateMemory(dev, &mai, nullptr, &g_taa.historyMem[hi]) != VK_SUCCESS) {
            trace("TAA: history memory %d allocation failed (%llu bytes)", hi,
                  (unsigned long long)mr.size);
            return false;
        }
        dd.bindImageMemory(dev, g_taa.history[hi], g_taa.historyMem[hi], 0);
    }

    VkImageViewCreateInfo ivci;
    memset(&ivci, 0, sizeof(ivci));
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    ivci.format = fmt;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.layerCount = g_taa.layers;
    for (int hi = 0; hi < 2; ++hi) {
        ivci.image = g_taa.history[hi];
        if (dd.createImageView(dev, &ivci, nullptr, &g_taa.historyView[hi]) != VK_SUCCESS) return false;
    }
    ivci.image = scene;
    if (dd.createImageView(dev, &ivci, nullptr, &g_taa.sceneView) != VK_SUCCESS) return false;
    g_taa.sceneViews[scene] = g_taa.sceneView;

    VkSamplerCreateInfo sci;
    memset(&sci, 0, sizeof(sci));
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    // Clamp to edge: a history fetch that lands outside is already rejected by
    // the shader's bounds test, so the address mode only has to be defined.
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = 0.0f;
    if (dd.createSampler(dev, &sci, nullptr, &g_taa.sampler) != VK_SUCCESS) return false;

    // ---- A SECOND, NEAREST SAMPLER, FOR THE INTEGER FLAGS IMAGE.
    //
    // uFlags is X-Plane's gbuffer_vel, VK_FORMAT_R32_UINT. Integer formats do
    // not advertise SAMPLED_IMAGE_FILTER_LINEAR, so sampling one through the
    // linear sampler above is undefined - and validation says so on every
    // dispatch:
    //
    //   VUID-vkCmdDispatch-magFilter-04553
    //   binding 4 "uFlags" ... VK_FILTER_LINEAR ... format VK_FORMAT_R32_UINT
    //   does not contain VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT
    //
    // Nothing about a bitfield wants interpolating anyway: the shader tests
    // bit 2, and a blended fraction of two flag words is meaningless. NEAREST
    // is both legal and the only correct filter for this input.
    sci.magFilter = VK_FILTER_NEAREST;
    sci.minFilter = VK_FILTER_NEAREST;
    if (dd.createSampler(dev, &sci, nullptr, &g_taa.samplerNearest) != VK_SUCCESS)
        return false;

    // ---- THE READBACK BUFFER. See TaaState::readBuf.
    //
    // Host-visible and COHERENT so no explicit flush or invalidate is needed;
    // persistently mapped, because mapping per frame would serialise against
    // the very timing this exists to observe. Failure here is not fatal - the
    // resolve runs exactly as before and only the diagnostic is absent.
    {
        VkBufferCreateInfo bci;
        memset(&bci, 0, sizeof(bci));
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = 2ull * 512ull * 8ull;          // history strip + scene strip
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (dd.createBuffer && dd.createBuffer(dev, &bci, nullptr, &g_taa.readBuf) == VK_SUCCESS) {
            VkMemoryRequirements mr;
            memset(&mr, 0, sizeof(mr));
            if (dd.getBufferMemReq) dd.getBufferMemReq(dev, g_taa.readBuf, &mr);
            const uint32_t mt = taaFindMemory(dd, mr.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            VkMemoryAllocateInfo mai;
            memset(&mai, 0, sizeof(mai));
            mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            mai.allocationSize  = mr.size ? mr.size : bci.size;
            mai.memoryTypeIndex = mt;
            if (mt != UINT32_MAX && dd.allocateMemory &&
                dd.allocateMemory(dev, &mai, nullptr, &g_taa.readMem) == VK_SUCCESS &&
                dd.bindBufferMemory &&
                dd.bindBufferMemory(dev, g_taa.readBuf, g_taa.readMem, 0) == VK_SUCCESS &&
                dd.mapMemory &&
                dd.mapMemory(dev, g_taa.readMem, 0, VK_WHOLE_SIZE, 0, &g_taa.readPtr) == VK_SUCCESS) {
                memset(g_taa.readPtr, 0, (size_t)bci.size);
                trace("TAA READBACK: history strip buffer mapped - the history "
                      "image can now be compared across frames directly, "
                      "instead of through the resolve's own output path");
            } else {
                g_taa.readPtr = nullptr;
            }
        }
    }

    // ---- THE FALLBACK FLAG IMAGE. 1x1, uint, zero.
    //
    // Created unconditionally so binding 4 is always valid; cleared alongside
    // the history on first use.
    {
        VkImageCreateInfo fci;
        memset(&fci, 0, sizeof(fci));
        fci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        fci.imageType = VK_IMAGE_TYPE_2D;
        fci.format = VK_FORMAT_R32_UINT;
        fci.extent.width = 1; fci.extent.height = 1; fci.extent.depth = 1;
        fci.mipLevels = 1; fci.arrayLayers = 1;
        fci.samples = VK_SAMPLE_COUNT_1_BIT;
        fci.tiling = VK_IMAGE_TILING_OPTIMAL;
        fci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        fci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        fci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (dd.createImage(dev, &fci, nullptr, &g_taa.flagsFallback) == VK_SUCCESS) {
            VkMemoryRequirements fmr;
            dd.getImageMemReq(dev, g_taa.flagsFallback, &fmr);
            VkMemoryAllocateInfo fai;
            memset(&fai, 0, sizeof(fai));
            fai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            fai.allocationSize = fmr.size;
            fai.memoryTypeIndex = taaFindMemory(dd, fmr.memoryTypeBits,
                                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (fai.memoryTypeIndex != UINT32_MAX &&
                dd.allocateMemory(dev, &fai, nullptr, &g_taa.flagsFallbackMem) == VK_SUCCESS) {
                dd.bindImageMemory(dev, g_taa.flagsFallback, g_taa.flagsFallbackMem, 0);
                VkImageViewCreateInfo fvci;
                memset(&fvci, 0, sizeof(fvci));
                fvci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                fvci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
                fvci.format = VK_FORMAT_R32_UINT;
                fvci.image = g_taa.flagsFallback;
                fvci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                fvci.subresourceRange.levelCount = 1;
                fvci.subresourceRange.layerCount = 1;
                dd.createImageView(dev, &fvci, nullptr, &g_taa.flagsFallbackView);
            }
        }
        g_taa.flagsView  = g_taa.flagsFallbackView;
        g_taa.flagsValid = false;
        if (!g_taa.flagsFallbackView)
            trace("TAA: flags fallback unavailable - the gbuffer_vel weight "
                  "override stays off");
    }

    VkDescriptorSetLayoutBinding b[5];
    memset(b, 0, sizeof(b));
    // Binding 0 is a SAMPLER now, not a storage image: the dispatch only reads
    // the scene. That is what lets the scene target keep X-Plane's own usage
    // flags untouched.
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[3].binding = 3; b[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[4].binding = 4; b[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    for (int i = 0; i < 5; ++i) {
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci;
    memset(&dlci, 0, sizeof(dlci));
    dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 5; dlci.pBindings = b;
    if (dd.createDescriptorSetLayout(dev, &dlci, nullptr, &g_taa.setLayout) != VK_SUCCESS) return false;

    VkPushConstantRange pcr;
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(TaaPush);
    VkPipelineLayoutCreateInfo plci;
    memset(&plci, 0, sizeof(plci));
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &g_taa.setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (dd.createPipelineLayout(dev, &plci, nullptr, &g_taa.pipeLayout) != VK_SUCCESS) return false;

    VkShaderModuleCreateInfo smci;
    memset(&smci, 0, sizeof(smci));
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = kTaaResolveSpvWords * 4;
    smci.pCode = kTaaResolveSpv;
    VkShaderModule sm = VK_NULL_HANDLE;
    if (dd.createShaderModule(dev, &smci, nullptr, &sm) != VK_SUCCESS) {
        trace("TAA: shader module creation failed");
        return false;
    }
    VkComputePipelineCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = g_taa.pipeLayout;
    VkResult pr = dd.createComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &g_taa.pipeline);
    dd.destroyShaderModule(dev, sm, nullptr);
    if (pr != VK_SUCCESS) {
        trace("TAA: compute pipeline creation failed (%d)", (int)pr);
        return false;
    }

    VkDescriptorPoolSize ps[2];
    ps[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ps[0].descriptorCount = 1 * TaaState::kSets;   // history write only
    ps[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ps[1].descriptorCount = 4 * TaaState::kSets;   // scene, velocity, history, flags
    VkDescriptorPoolCreateInfo dpci;
    memset(&dpci, 0, sizeof(dpci));
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = TaaState::kSets;
    dpci.poolSizeCount = 2; dpci.pPoolSizes = ps;
    if (dd.createDescriptorPool(dev, &dpci, nullptr, &g_taa.pool) != VK_SUCCESS) return false;

    VkDescriptorSetLayout layouts[TaaState::kSets];
    for (uint32_t i = 0; i < TaaState::kSets; ++i) layouts[i] = g_taa.setLayout;
    VkDescriptorSetAllocateInfo dsai;
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = g_taa.pool;
    dsai.descriptorSetCount = TaaState::kSets;
    dsai.pSetLayouts = layouts;
    if (dd.allocateDescriptorSets(dev, &dsai, g_taa.sets) != VK_SUCCESS) return false;

    g_taa.velView = velView;
    g_taa.ready = true;
    g_taa.historyCleared = false;
    trace("TAA: ready - %ux%u x%u layer(s) fmt=%d, mode %d, alpha %.3f, "
          "%u descriptor sets",
          w, h, g_taa.layers, (int)fmt, taaMode(), taaAlpha(), TaaState::kSets);
    return true;
}

// Point the resolve at whichever target this frame drew into, creating that
// view once. Returns false only if the view cannot be made.
static bool taaBindScene(DeviceData &dd, VkImage scene)
{
    if (scene == VK_NULL_HANDLE) return false;
    std::map<VkImage, VkImageView>::iterator it = g_taa.sceneViews.find(scene);
    if (it != g_taa.sceneViews.end()) {
        g_taa.sceneView  = it->second;
        g_taa.sceneImage = scene;
        return true;
    }
    VkImageViewCreateInfo ivci;
    memset(&ivci, 0, sizeof(ivci));
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    ivci.format = g_taa.format;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.layerCount = g_taa.layers;
    ivci.image = scene;
    VkImageView v = VK_NULL_HANDLE;
    if (dd.createImageView(g_taa.device, &ivci, nullptr, &v) != VK_SUCCESS) {
        trace("TAA: scene view creation failed for a second target");
        return false;
    }
    g_taa.sceneViews[scene] = v;
    g_taa.sceneView  = v;
    g_taa.sceneImage = scene;
    trace("TAA: now %llu distinct scene targets - X-Plane alternates between "
          "them, so views are cached rather than rebuilt",
          (unsigned long long)g_taa.sceneViews.size());
    return true;
}

// ---- THE TAA BACKEND.
//
// The first implementation of temporal::IBackend, and for now the whole of it.
// It is written against the shared contract rather than against layer.cpp so
// that FSR, DLSS, XeSS and frame generation can be added beside it without any
// of them reaching into the interception code - and so that the questions each
// of them will ask (what shape is the target, which convention are the vectors
// in, why was history reset) are already answered in one place.
//
// The conversion table for this backend, stated rather than assumed:
//
//   coordinateSpace       UV        the stored 0.5 IS the NDC->UV conversion
//   direction             prev-curr we store prev minus curr
//   jitterIncluded        false     jitter is applied AFTER the varyings
//   cameraMotionIncluded  true
//   objectMotionIncluded  true      via the injected per-draw matrices
//
// This backend consumes those directly, so nothing converts. That is exactly
// what makes it the wrong place to discover a convention bug, and exactly why
// the declaration is written down for the backends that WILL convert.
class TaaBackend : public temporal::IBackend {
public:
    temporal::BackendInfo info() const override
    {
        temporal::BackendInfo i;
        i.name = "TAA";
        i.vendor = temporal::VENDOR_ANY;      // ours; always available
        i.supportsUpscaling = false;          // TAAU is a separate backend
        i.supportsNativeResolution = true;
        i.supportsArrayLayers = true;         // 2D_ARRAY views, dispatch z=layers
        i.supportsMultisample = false;        // see accepts()
        return i;
    }

    // Report WHY, not just no. A backend silently absent is indistinguishable
    // from a backend silently broken, which is how a dead code path once hid
    // through two crashes and a wrong diagnosis.
    bool accepts(const temporal::TemporalFrame &f, const char **why) const override
    {
        if (f.color.image == VK_NULL_HANDLE) {
            if (why) *why = "no colour target";
            return false;
        }
        if (f.color.samples != VK_SAMPLE_COUNT_1_BIT) {
            // 576 of the 1152 gbuffer_lit permutations are multisampled, and
            // fix_hdr_1 binds the HDR image as a multisampled storage image, so
            // this is a real configuration rather than a theoretical one. It
            // needs a per-sample resolve, which this shader does not do.
            // Declining leaves the frame untouched; pretending would corrupt it.
            if (why) *why = "multisampled target - needs a per-sample resolve";
            return false;
        }
        if (f.motion.image == VK_NULL_HANDLE || f.motion.view == VK_NULL_HANDLE) {
            if (why) *why = "no motion vectors this frame";
            return false;
        }
        if (f.motion.coordinateSpace != temporal::COORD_UV ||
            f.motion.direction != temporal::DIR_PREVIOUS_TO_CURRENT) {
            // The shader hard-codes this convention. If the core ever produces
            // vectors in another one, this must fail rather than reinterpret
            // them - a silently misread field is the failure mode the
            // MotionVectors metadata exists to prevent.
            if (why) *why = "motion vectors are not UV / previous-minus-current";
            return false;
        }
        if (f.motion.jitterIncluded) {
            if (why) *why = "motion vectors have jitter baked in";
            return false;
        }
        return true;
    }

    bool record(VkCommandBuffer cb, const temporal::TemporalFrame &f) override;
};

static TaaBackend g_taaBackend;

// The device dispatch the backend records through. It lives here rather than in
// TemporalFrame because it is a property of the LAYER's Vulkan plumbing, not of
// the frame - and TemporalFrame is meant to describe a frame to a consumer that
// knows nothing about how we intercepted it.
static DeviceData *g_taaDevice = nullptr;

// Point binding 4 at X-Plane's gbuffer_vel, creating that view once per image.
// Refuses multisampled or non-uint candidates: the shader declares a
// single-sample uint array sampler, and a mismatched view is exactly the class
// of silent wrongness the shape checks exist to prevent.
static void taaBindFlags(DeviceData &dd, VkImage image, VkFormat fmt,
                         uint32_t layers, VkSampleCountFlagBits samples)
{
    if (image == VK_NULL_HANDLE || samples != VK_SAMPLE_COUNT_1_BIT) {
        g_taa.flagsView  = g_taa.flagsFallbackView;
        g_taa.flagsValid = false;
        return;
    }
    std::map<VkImage, VkImageView>::iterator it = g_taa.flagsViews.find(image);
    if (it != g_taa.flagsViews.end()) {
        g_taa.flagsView  = it->second;
        g_taa.flagsValid = (it->second != VK_NULL_HANDLE);
        return;
    }
    VkImageViewCreateInfo v;
    memset(&v, 0, sizeof(v));
    v.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    v.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    v.format = fmt;
    v.image = image;
    v.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    v.subresourceRange.levelCount = 1;
    v.subresourceRange.layerCount = layers ? layers : 1;
    VkImageView view = VK_NULL_HANDLE;
    if (dd.createImageView(g_taa.device, &v, nullptr, &view) != VK_SUCCESS) {
        trace("TAA: gbuffer_vel view creation failed - weight override off");
        view = VK_NULL_HANDLE;
    } else {
        trace("TAA: gbuffer_vel bound - moving-geometry pixels (bit 2) now "
              "take the current frame outright. taa.objflags=0 disables live.");
    }
    g_taa.flagsViews[image] = view;
    g_taa.flagsView  = view ? view : g_taa.flagsFallbackView;
    g_taa.flagsValid = (view != VK_NULL_HANDLE);
}

// ---- RECORD THE RESOLVE.
//
// Called from vkCmdEndRendering once the scene pass has finished, which is the
// only point where the colour target holds a complete frame and the velocity
// target beside it describes that same frame.
// ---- DELIVER HISTORY WITHOUT RESOLVING. (the alternation fix)
//
// The copy that carries the accumulated image into the scene target lives at
// the end of taaRecordResolve, so a frame that does not dispatch does not
// deliver either - it ships the raw scene. Measured, static camera:
//
//     history buffer      0.447   <- stable, accumulating correctly
//     composited output   3.78    <- unstable
//     TAA off baseline    0.806
//
// History is MORE stable than the raw scene. The resolve works. What alternates
// is whether its result reaches the screen, and alternating between accumulated
// and raw output is the shake - which is also why jitter made it visible, since
// without jitter the two look nearly identical.
//
// So when a frame has a validated target and a ready history but skips the
// dispatch - the per-present resolve cap, a quiesce, a stale velocity field -
// it still delivers the last good accumulation instead of nothing. That is
// strictly better than raw: the image is at most one frame old, where before it
// was a different image entirely.
//
// Only called with a target this frame's pass actually rendered into; there is
// deliberately no path for frames with no candidate pass at all, because the
// destination would be a guess and copying into the wrong half is the smearing
// the target-latch experiment produced.
static void taaRecordDeliverOnly(DeviceData &dd, VkCommandBuffer cb, VkImage scene)
{
    if (!g_taa.ready || !taaEnabled() || !g_taa.historyCleared) return;
    if (scene == VK_NULL_HANDLE || !dd.cmdCopyImage) return;
    // The most recently written buffer. The flip happens at the end of a
    // resolve, so the freshest history is the one the index no longer points at.
    const uint32_t src = g_taa.historyWrite ^ 1u;
    if (!g_taa.history[src]) return;

    VkImageMemoryBarrier pre[2];
    memset(pre, 0, sizeof(pre));
    for (int i = 0; i < 2; ++i) {
        pre[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        pre[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pre[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pre[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        pre[i].subresourceRange.levelCount = 1;
        pre[i].subresourceRange.layerCount = g_taa.layers;
    }
    pre[0].image = scene;
    pre[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    pre[0].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    pre[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    pre[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    pre[1].image = g_taa.history[src];
    pre[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    pre[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    pre[1].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    pre[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    // Each half named with a stage that can actually perform its access - the
    // mistake VUID-02819 caught in the resolve's own barrier.
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                          0, nullptr, 0, nullptr, 1, &pre[0]);
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                          0, nullptr, 0, nullptr, 1, &pre[1]);

    VkImageCopy cp;
    memset(&cp, 0, sizeof(cp));
    cp.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    cp.srcSubresource.layerCount = g_taa.layers;
    cp.dstSubresource = cp.srcSubresource;
    cp.extent.width = g_taa.w; cp.extent.height = g_taa.h; cp.extent.depth = 1;
    dd.cmdCopyImage(cb, g_taa.history[src], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    scene, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);

    VkImageMemoryBarrier post[2] = { pre[0], pre[1] };
    post[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    post[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    post[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    post[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    post[1].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    post[1].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    post[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    post[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                          0, nullptr, 0, nullptr, 1, &post[0]);
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                          0, nullptr, 0, nullptr, 1, &post[1]);
}

static void taaRecordResolve(DeviceData &dd, VkCommandBuffer cb,
                             float jitterX, float jitterY, bool reset,
                             bool cameraMoved)
{
    if (!g_taa.ready || !taaEnabled()) return;

    VkImageMemoryBarrier bar[3];
    memset(bar, 0, sizeof(bar));
    for (int i = 0; i < 3; ++i) {
        bar[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        bar[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        bar[i].subresourceRange.levelCount = 1;
        bar[i].subresourceRange.layerCount = g_taa.layers;
    }
    bar[0].image = g_taa.sceneImage;
    bar[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    bar[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bar[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    bar[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    const uint32_t hw_ = g_taa.historyWrite;         // written this frame
    const uint32_t hr_ = hw_ ^ 1u;                    // read this frame
    bar[1].image = g_taa.history[hw_];
    bar[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bar[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    bar[1].oldLayout = g_taa.historyCleared ? VK_IMAGE_LAYOUT_GENERAL
                                            : VK_IMAGE_LAYOUT_UNDEFINED;
    bar[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;

    bar[2].image = g_mv.image;
    bar[2].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    bar[2].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bar[2].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    bar[2].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkImageMemoryBarrier barRead = bar[1];
    barRead.image = g_taa.history[hr_];
    barRead.oldLayout = g_taa.historyCleared ? VK_IMAGE_LAYOUT_GENERAL
                                             : VK_IMAGE_LAYOUT_UNDEFINED;
    barRead.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    // ---- TWO BARRIERS, BECAUSE THE FOUR IMAGES COME FROM DIFFERENT STAGES.
    //
    // All four used to go through one barrier whose source stage was chosen for
    // the colour attachments. But bar[1] and barRead describe the HISTORY
    // images, whose previous write was last frame's COMPUTE dispatch, and
    // SHADER_WRITE is not an access that can occur in COLOR_ATTACHMENT_OUTPUT.
    // The source half of those two was therefore empty: no execution dependency
    // on last frame's dispatch, and no availability operation for its writes.
    // Validation names them individually:
    //
    //   VUID-vkCmdPipelineBarrier-pImageMemoryBarriers-02819
    //   pImageMemoryBarriers[1].srcAccessMask (VK_ACCESS_2_SHADER_WRITE_BIT)
    //   is not supported by stage mask (COLOR_ATTACHMENT_OUTPUT)   [and [3]]
    //
    // Indices 1 and 3 are exactly bar[1] and barRead in the old all4 array.
    //
    // The copy-back does not rescue it: that path ends on a barrier whose
    // srcAccessMask is TRANSFER_READ, and a read performs no availability
    // operation, so the compute write is never made available to the next
    // frame's sampled read. Splitting gives each pair a source scope that
    // actually contains the write it is describing - and the history pair also
    // names TRANSFER, since the copy-back reads these images too.
    {
        VkImageMemoryBarrier att[2] = { bar[0], bar[2] };
        dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                              0, nullptr, 0, nullptr, 2, att);
        VkImageMemoryBarrier hist[2] = { bar[1], barRead };
        dd.cmdPipelineBarrier(cb,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                              0, nullptr, 0, nullptr, 2, hist);
    }

    // Clear the history once, explicitly. Reading an UNDEFINED image gives
    // undefined contents and on the first frame every output pixel is a blend
    // with it - the shape of the whole-frame magenta the old resolve produced.
    if (!g_taa.historyCleared) {
        VkClearColorValue cc;
        memset(&cc, 0, sizeof(cc));
        VkImageSubresourceRange r;
        memset(&r, 0, sizeof(r));
        r.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        r.levelCount = 1; r.layerCount = g_taa.layers;
        VkImageMemoryBarrier tb[2];
        for (int hi = 0; hi < 2; ++hi) {
            tb[hi] = bar[1];
            tb[hi].image = g_taa.history[hi];
            tb[hi].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            tb[hi].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            tb[hi].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        }
        dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                              0, nullptr, 0, nullptr, 2, tb);
        // Both buffers: whichever is read first must be defined too.
        for (int hi = 0; hi < 2; ++hi)
            dd.cmdClearColorImage(cb, g_taa.history[hi],
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cc, 1, &r);
        // The flags fallback too: UNDEFINED to zero to SHADER_READ_ONLY, once.
        if (g_taa.flagsFallback != VK_NULL_HANDLE) {
            VkImageMemoryBarrier fb = tb[0];
            fb.image = g_taa.flagsFallback;
            fb.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            fb.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            fb.srcAccessMask = 0;
            fb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            fb.subresourceRange.layerCount = 1;
            dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                  0, nullptr, 0, nullptr, 1, &fb);
            VkClearColorValue fz;
            memset(&fz, 0, sizeof(fz));
            dd.cmdClearColorImage(cb, g_taa.flagsFallback,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &fz, 1,
                                  &fb.subresourceRange);
            fb.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            fb.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            fb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            fb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                  0, nullptr, 0, nullptr, 1, &fb);
        }
        for (int hi = 0; hi < 2; ++hi) {
            tb[hi].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            tb[hi].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            tb[hi].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            tb[hi].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        }
        dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                              0, nullptr, 0, nullptr, 2, tb);
        g_taa.historyCleared = true;
        reset = true;
    }

    VkDescriptorSet set = g_taa.sets[g_taa.nextSet];
    g_taa.nextSet = (g_taa.nextSet + 1) % TaaState::kSets;

    VkDescriptorImageInfo ii[5];
    memset(ii, 0, sizeof(ii));
    ii[0].imageView = g_taa.sceneView;
    ii[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ii[0].sampler   = g_taa.sampler;
    ii[1].imageView = g_taa.historyView[hw_]; ii[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ii[2].imageView = g_taa.velView;     ii[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    // NEAREST. A velocity field is piecewise per surface, and now it also
    // carries a large negative sentinel in unwritten pixels - bilinear would
    // blend that sentinel into neighbouring real vectors and mark them
    // unwritten too, eating history in a halo around every sky edge.
    ii[2].sampler   = g_taa.samplerNearest;
    ii[3].imageView = g_taa.historyView[hr_]; ii[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ii[3].sampler   = g_taa.sampler;
    // Binding 4: X-Plane's gbuffer_vel when identified, our zero fallback
    // otherwise. NO BARRIER is recorded for the real image: by the time the
    // resolve runs, ssr_deferred and the lighting pass have both consumed it,
    // so X-Plane has already transitioned it to a shader-readable layout and
    // will not write it again until next frame's G-buffer pass.
    ii[4].imageView = (g_taa.flagsValid && g_taa.flagsView) ? g_taa.flagsView
                                                            : g_taa.flagsFallbackView;
    ii[4].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ii[4].sampler   = g_taa.samplerNearest;   // integer format: NEAREST only

    VkWriteDescriptorSet wr[5];
    memset(wr, 0, sizeof(wr));
    for (int i = 0; i < 5; ++i) {
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].dstSet = set;
        wr[i].dstBinding = (uint32_t)i;
        wr[i].descriptorCount = 1;
        wr[i].pImageInfo = &ii[i];
    }
    wr[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wr[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    dd.updateDescriptorSets(g_taa.device, 5, wr, 0, nullptr);

    dd.cmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_taa.pipeline);
    dd.cmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_taa.pipeLayout,
                             0, 1, &set, 0, nullptr);

    TaaPush pcv;
    pcv.invSizeX = 1.0f / (float)g_taa.w;
    pcv.invSizeY = 1.0f / (float)g_taa.h;
    pcv.jitterX = jitterX;
    pcv.sMulX = live::f("taa.smul_x", "TAA_SMUL_X",  0.5f);
    pcv.sMulY = live::f("taa.smul_y", "TAA_SMUL_Y", -0.5f);
    pcv.velMax = live::f("taa.vel_max", "TAA_VEL_MAX", 1.0f);
    // ---- THE UNWRITTEN-PIXEL REJECTION IS OFF BY DEFAULT.
    //
    // Its purpose is sound - sky and cloud pixels must not reproject - but no
    // signal available here identifies them reliably. vel==0 is ambiguous the
    // moment the camera moves by less than a pixel, coverage reads zero
    // everywhere, and the sentinel marks whole passes that a racing clear
    // erased. Every version of the test rejected most of the frame, and
    // rejecting history IS the shake:
    //     rejection on ....... 1.409
    //     rejection off ...... 0.172
    //     jitter off ......... 0.128
    //     TAA off ............ 0.107
    // Negative disables it. A real fix needs a per-pixel written flag the
    // fragment patcher owns; until then, keeping history everywhere is the
    // measurably better picture.
    pcv.novecCov = live::f("taa.novec_cov", "TAA_NOVEC_COV", -1.0f);
    // 0.5 was chosen to stop the ground crawling, but it also refuses to
    // accumulate on every pixel the sentinel calls unwritten - which is most of
    // an external frame - and that is shimmer and aliasing. Measured with the
    // detection off entirely and this at the normal alpha:
    //     TAA off ............. shimmer 0.737   jaggedness 48.8
    //     detection ON ........ shimmer 1.986   jaggedness 25.6
    //     detection OFF ....... shimmer 0.358   jaggedness 24.6
    // TAA is then steadier than no TAA and halves the aliasing. The crawl is
    // the price, and it is the lower priority of the two.
    pcv.novecAlpha = live::f("taa.novec_alpha", "TAA_NOVEC_ALPHA", 0.05f);
    pcv.jitterY = jitterY;
    pcv.alpha = taaAlpha();
    pcv.mode = taaMode();
    pcv.reset = (reset || taaForceReset()) ? 1 : 0;
    pcv.cameraMoved = cameraMoved ? 1 : 0;
    pcv.viz      = taaViz();
    pcv.vizScale = taaVizScale();
    pcv.gain     = taaGain();
    pcv.varClip  = taaVarClip();
    pcv.movedDead = taaMovedDead();
    pcv.flags    = (taaFreezeHistory() ? kTaaFlagFreezeHistory : 0)
                 | (taaNoMotion()      ? kTaaFlagNoMotion      : 0)
                 | (taaNoAccum()       ? kTaaFlagNoAccum       : 0)
                 | (taaReactive()      ? kTaaFlagReactive      : 0)
                 | (taaUnjitter()      ? 0 : kTaaFlagNoUnjitter)
                 | (live::onoff("taa.novec_by_vel", nullptr, false)
                        ? kTaaFlagNoVecByVel : 0);
    pcv.velScale = taaVelScale();
    pcv.velYSign = taaVelYSign();
    pcv.flagsValid = (g_taa.flagsValid && taaObjFlags()) ? 1 : 0;
    // ---- A CHANGED KNOB IS A CHANGED HISTORY.
    //
    // Every one of these redefines what the accumulated image MEANS: a
    // different viz draws a different picture into the history buffer, a
    // different sign or scale reprojects it differently, and blending across
    // the change drags the old regime's pixels into the new one. That is not
    // hypothetical - leaving the heatmap view without a reset visibly dissolved
    // the heatmap into the scene, and it read as scene corruption for half a
    // session. Any change forces one clean frame.
    {
        static int   lastViz   = -1;
        static float lastScale = 0.0f, lastSign = 0.0f;
        if (pcv.viz != lastViz || pcv.velScale != lastScale ||
            pcv.velYSign != lastSign) {
            if (lastViz != -1) pcv.reset = 1;
            lastViz = pcv.viz; lastScale = pcv.velScale; lastSign = pcv.velYSign;
        }
    }
    dd.cmdPushConstants(cb, g_taa.pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                        0, sizeof(pcv), &pcv);

    // z = layers, matching gl_GlobalInvocationID.z in the shader.
    dd.cmdDispatch(cb, (g_taa.w + 7) / 8, (g_taa.h + 7) / 8, g_taa.layers);

    // ---- COPY THE RESULT INTO THE SCENE TARGET.
    //
    // Separate command, explicit ordering: everything the dispatch reads is
    // finished before anything is written back. In place would reintroduce the
    // neighbourhood race the read-only binding just removed.
    {
        VkImageMemoryBarrier pre[2];
        pre[0] = bar[0];
        pre[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        pre[0].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        pre[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        pre[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        pre[1] = bar[1];
        pre[1].image = g_taa.history[hw_];
        pre[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        pre[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        pre[1].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        pre[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                              0, nullptr, 0, nullptr, 2, pre);

        VkImageCopy cp;
        memset(&cp, 0, sizeof(cp));
        cp.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        cp.srcSubresource.layerCount = g_taa.layers;
        cp.dstSubresource = cp.srcSubresource;
        cp.extent.width = g_taa.w; cp.extent.height = g_taa.h; cp.extent.depth = 1;
        dd.cmdCopyImage(cb, g_taa.history[hw_], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        g_taa.sceneImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);

        VkImageMemoryBarrier post = pre[1];
        post.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        post.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        post.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        post.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                              0, nullptr, 0, nullptr, 1, &post);
    }

    // The destination stage for this transition back is COLOR_ATTACHMENT_OUTPUT
    // (see the call below), and SHADER_READ is not an access that occurs there
    // - VUID-vkCmdPipelineBarrier-pImageMemoryBarriers-02820, twenty a run. The
    // shader-read half was silently dropped exactly as the source half was at
    // the barrier above. Naming FRAGMENT_SHADER alongside the attachment stage
    // in the call makes both halves real; the mask itself is right.
    bar[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    bar[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    bar[2].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bar[2].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    bar[2].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bar[2].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                          0, nullptr, 0, nullptr, 1, &bar[0]);
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                          0, nullptr, 0, nullptr, 1, &bar[2]);

    // ---- COPY A STRIP OF HISTORY OUT, FOR THE CPU TO COMPARE.
    //
    // Recorded after the dispatch and after the copy-back, so it captures the
    // history this frame actually produced. No fence: the read happens at
    // present, at least one frame later, by which point the submission that
    // recorded this has long retired. A stale read would understate the
    // difference, never invent one, so the metric is conservative in the
    // direction that matters.
    if (g_taa.readPtr && g_taa.readBuf && dd.cmdCopyImageToBuffer) {
        VkImageMemoryBarrier rb;
        memset(&rb, 0, sizeof(rb));
        rb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        rb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        rb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        rb.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        rb.subresourceRange.levelCount = 1;
        rb.subresourceRange.layerCount = g_taa.layers;
        rb.image = g_taa.history[hw_];
        rb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        rb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        rb.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        rb.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                              0, nullptr, 0, nullptr, 1, &rb);
        VkBufferImageCopy bic;
        memset(&bic, 0, sizeof(bic));
        bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        bic.imageSubresource.layerCount = 1;
        // A horizontal strip across the middle of the frame: guaranteed to
        // cross real scene content at any camera angle.
        bic.imageOffset.x = 0;
        bic.imageOffset.y = (int32_t)(g_taa.h / 2);
        bic.imageExtent.width  = g_taa.w < 512 ? g_taa.w : 512;
        bic.imageExtent.height = 1;
        bic.imageExtent.depth  = 1;
        dd.cmdCopyImageToBuffer(cb, g_taa.history[hw_],
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                g_taa.readBuf, 1, &bic);
        // ---- AND THE SAME STRIP OF THE SCENE TARGET, RIGHT AFTER THE COPY.
        //
        // History accumulates correctly (delta 6.3 at 98% history weight) while
        // the composited image stays ~6x less stable than TAA off. Everything
        // between those two facts is delivery, and this settles it in one run:
        // if the scene strip matches the history strip, the copy landed and
        // something downstream repaints; if it differs, the copy is not
        // reaching the image the display reads from - which is what a
        // double-buffered scene target would do.
        //
        // Into the second half of the same buffer, so one map serves both.
        if (g_taa.sceneImage) {
            VkImageMemoryBarrier sb = rb;
            sb.image = g_taa.sceneImage;
            sb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            sb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            sb.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            sb.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT |
                                      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                  0, nullptr, 0, nullptr, 1, &sb);
            VkBufferImageCopy sic = bic;
            sic.bufferOffset = 2048ull * sizeof(uint16_t);
            dd.cmdCopyImageToBuffer(cb, g_taa.sceneImage,
                                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    g_taa.readBuf, 1, &sic);
            VkImageMemoryBarrier sbk = sb;
            sbk.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            sbk.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            sbk.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            sbk.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                                  0, nullptr, 0, nullptr, 1, &sbk);
        }
        VkImageMemoryBarrier rbk = rb;
        rbk.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        rbk.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        rbk.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        rbk.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                              0, nullptr, 0, nullptr, 1, &rbk);
    }

    // ---- ONE FLIP PER PRESENTED FRAME, NOT ONE PER RESOLVE.
    //
    // This flipped unconditionally, and the resolve can run several times in a
    // single present (X-Plane records ahead; taa.max_resolves bounds it). With
    // an EVEN number of resolves in a frame the index returns to where it
    // started, so the same buffer is written every time and the other - the one
    // every resolve READS - is never updated again. History freezes, and with
    // it every property that depends on accumulation.
    //
    // Measured, with a static camera and 98% history weight, which cannot
    // change the image at all if history is real:
    //     TAA off                  frame-to-frame diff 0.149
    //     TAA on                                       4.35
    //     TAA on, alpha 0.02                           3.29
    //     TAA on, alpha 0.02, jitter off               0.229
    // TAA made a still image 29x LESS stable than no TAA. That is this.
    //
    // It also explains the shape of the whole hunt: without jitter consecutive
    // frames are nearly identical, so a frozen history is invisible; with
    // jitter every frame differs by a sub-pixel offset, the output becomes the
    // raw jittered frame, and that is the shake. Gain, varclip, the reactive
    // mask, both sign flips and the target latch all tune how history is
    // BLENDED, which is why none of them moved a symptom caused by there being
    // no history to blend.
    //
    // The arming flag is owned by the present hook, so the pairing follows
    // DISPLAYED frames however many times the resolve records.
    // ---- FLIP PER RECORDED FRAME, WHICH IS PER RESOLVE.
    //
    // This was briefly tied to PRESENT, on the reasoning that the ping-pong
    // should follow displayed frames. That is wrong here: X-Plane records
    // ahead, so several frames' command buffers are built before any present.
    // Arming at present meant the first recorded frame flipped and every
    // further frame recorded before that present did not - consecutive frames
    // wrote the SAME buffer, so the buffer being read never advanced and the
    // blend had nothing to accumulate against.
    //
    // Measured with the direct history readback, at alpha 0.02 (98% history),
    // vel_scale 0 (lookup pinned to the same pixel) and reset 0 - conditions
    // under which the output mathematically cannot change if history is real:
    //     1532 of 2048 halves differed every frame, mean |delta| 193.
    // The blend cannot produce that; only a history read that is not the
    // previous write can.
    //
    // One resolve per recorded frame (taa.max_resolves=1) makes per-resolve and
    // per-frame the same thing, which is the pairing the algorithm needs.
    g_taa.historyWrite ^= 1u;

    if ((++g_taa.dispatches % 600) == 1)
        trace("TAA: dispatch %llu - mode %d alpha %.3f reset %d (%ux%u x%u)",
              (unsigned long long)g_taa.dispatches, taaMode(), taaAlpha(),
              pcv.reset, g_taa.w, g_taa.h, g_taa.layers);
}

// The IBackend entry point. Thin on purpose: everything above it is the
// contract, everything below is this backend's own business, and the only job
// here is to translate one into the other.
inline bool TaaBackend::record(VkCommandBuffer cb, const temporal::TemporalFrame &f)
{
    if (!g_taaDevice) return false;
    taaRecordResolve(*g_taaDevice, cb, f.jitter.x, f.jitter.y,
                     f.reset != temporal::RESET_NONE, f.camera.moved);
    return true;
}
