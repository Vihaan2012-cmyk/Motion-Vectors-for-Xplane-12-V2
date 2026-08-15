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
    VkSampler       sampler     = VK_NULL_HANDLE;

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
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool     ready = false;
    bool     historyCleared = false;
    uint64_t dispatches = 0;
};

static TaaState g_taa;

struct TaaPush {
    float invSizeX, invSizeY;
    float jitterX, jitterY;
    float alpha;
    int32_t mode;
    int32_t reset;
    int32_t pad;
};

static bool taaEnabled()
{
    static const bool on = (getenv("TAA_RESOLVE") != nullptr);
    return on;
}

static int taaMode()
{
    static const int m = getenv("TAA_MODE") ? atoi(getenv("TAA_MODE")) : 0;
    return m;
}

static float taaAlpha()
{
    static const float a = getenv("TAA_ALPHA") ? (float)atof(getenv("TAA_ALPHA")) : 0.1f;
    return a;
}

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

static void taaDestroy(DeviceData &dd)
{
    if (g_taa.pipeline)    dd.destroyPipeline(g_taa.device, g_taa.pipeline, nullptr);
    if (g_taa.pipeLayout)  dd.destroyPipelineLayout(g_taa.device, g_taa.pipeLayout, nullptr);
    if (g_taa.setLayout)   dd.destroyDescriptorSetLayout(g_taa.device, g_taa.setLayout, nullptr);
    if (g_taa.pool)        dd.destroyDescriptorPool(g_taa.device, g_taa.pool, nullptr);
    if (g_taa.sampler)     dd.destroySampler(g_taa.device, g_taa.sampler, nullptr);
    for (int i = 0; i < 2; ++i)
        if (g_taa.historyView[i]) dd.destroyImageView(g_taa.device, g_taa.historyView[i], nullptr);
    for (std::map<VkImage, VkImageView>::iterator it = g_taa.sceneViews.begin();
         it != g_taa.sceneViews.end(); ++it)
        if (it->second) dd.destroyImageView(g_taa.device, it->second, nullptr);
    g_taa.sceneViews.clear();
    if (g_taa.velView)     dd.destroyImageView(g_taa.device, g_taa.velView, nullptr);
    for (int i = 0; i < 2; ++i) {
        if (g_taa.history[i])    dd.destroyImage(g_taa.device, g_taa.history[i], nullptr);
        if (g_taa.historyMem[i]) dd.freeMemory(g_taa.device, g_taa.historyMem[i], nullptr);
    }
    TaaState fresh;
    fresh.device = g_taa.device;
    g_taa = fresh;
}

// Build everything that depends on the scene target's size and format. Called
// again whenever either changes, which is why teardown comes first.
static bool taaInit(DeviceData &dd, VkDevice dev, VkImage scene, VkFormat fmt,
                    uint32_t w, uint32_t h, VkImageView velView)
{
    g_taa.device = dev;
    taaDestroy(dd);
    g_taa.device = dev;
    g_taa.w = w; g_taa.h = h; g_taa.format = fmt;
    g_taa.sceneImage = scene;
    g_taa.velView = VK_NULL_HANDLE;

    // ---- History image, same format as the scene so no conversion is implied.
    VkImageCreateInfo ici;
    memset(&ici, 0, sizeof(ici));
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = fmt;
    ici.extent.width = w; ici.extent.height = h; ici.extent.depth = 1;
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT;
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
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = fmt;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.layerCount = 1;
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

    VkDescriptorSetLayoutBinding b[4];
    memset(b, 0, sizeof(b));
    // Binding 0 is a SAMPLER now, not a storage image: the dispatch only reads
    // the scene. That is what lets the scene target keep X-Plane's own usage
    // flags untouched.
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[3].binding = 3; b[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    for (int i = 0; i < 4; ++i) {
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci;
    memset(&dlci, 0, sizeof(dlci));
    dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 4; dlci.pBindings = b;
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
    ps[1].descriptorCount = 3 * TaaState::kSets;   // scene, velocity, history read
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
    trace("TAA: ready - %ux%u fmt=%d, mode %d, alpha %.3f, %u descriptor sets",
          w, h, (int)fmt, taaMode(), taaAlpha(), TaaState::kSets);
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
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = g_taa.format;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.layerCount = 1;
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

// ---- RECORD THE RESOLVE.
//
// Called from vkCmdEndRendering once the scene pass has finished, which is the
// only point where the colour target holds a complete frame and the velocity
// target beside it describes that same frame.
static void taaRecordResolve(DeviceData &dd, VkCommandBuffer cb,
                             float jitterX, float jitterY, bool reset)
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
        bar[i].subresourceRange.layerCount = 1;
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
    VkImageMemoryBarrier all4[4] = { bar[0], bar[1], bar[2], barRead };
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                          0, nullptr, 0, nullptr, 4, all4);

    // Clear the history once, explicitly. Reading an UNDEFINED image gives
    // undefined contents and on the first frame every output pixel is a blend
    // with it - the shape of the whole-frame magenta the old resolve produced.
    if (!g_taa.historyCleared) {
        VkClearColorValue cc;
        memset(&cc, 0, sizeof(cc));
        VkImageSubresourceRange r;
        memset(&r, 0, sizeof(r));
        r.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        r.levelCount = 1; r.layerCount = 1;
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

    VkDescriptorImageInfo ii[4];
    memset(ii, 0, sizeof(ii));
    ii[0].imageView = g_taa.sceneView;
    ii[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ii[0].sampler   = g_taa.sampler;
    ii[1].imageView = g_taa.historyView[hw_]; ii[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ii[2].imageView = g_taa.velView;     ii[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ii[2].sampler   = g_taa.sampler;
    ii[3].imageView = g_taa.historyView[hr_]; ii[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ii[3].sampler   = g_taa.sampler;

    VkWriteDescriptorSet wr[4];
    memset(wr, 0, sizeof(wr));
    for (int i = 0; i < 4; ++i) {
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
    dd.updateDescriptorSets(g_taa.device, 4, wr, 0, nullptr);

    dd.cmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_taa.pipeline);
    dd.cmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_taa.pipeLayout,
                             0, 1, &set, 0, nullptr);

    TaaPush pcv;
    pcv.invSizeX = 1.0f / (float)g_taa.w;
    pcv.invSizeY = 1.0f / (float)g_taa.h;
    pcv.jitterX = jitterX;
    pcv.jitterY = jitterY;
    pcv.alpha = taaAlpha();
    pcv.mode = taaMode();
    pcv.reset = reset ? 1 : 0;
    pcv.pad = 0;
    dd.cmdPushConstants(cb, g_taa.pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                        0, sizeof(pcv), &pcv);

    dd.cmdDispatch(cb, (g_taa.w + 7) / 8, (g_taa.h + 7) / 8, 1);

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
        cp.srcSubresource.layerCount = 1;
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

    bar[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    bar[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    bar[2].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bar[2].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    bar[2].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bar[2].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                          0, nullptr, 0, nullptr, 1, &bar[0]);
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                          0, nullptr, 0, nullptr, 1, &bar[2]);

    g_taa.historyWrite ^= 1u;

    if ((++g_taa.dispatches % 600) == 1)
        trace("TAA: dispatch %llu - mode %d alpha %.3f reset %d (%ux%u)",
              (unsigned long long)g_taa.dispatches, taaMode(), taaAlpha(),
              pcv.reset, g_taa.w, g_taa.h);
}
