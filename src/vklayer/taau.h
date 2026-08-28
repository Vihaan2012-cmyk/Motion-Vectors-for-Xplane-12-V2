#pragma once
// ================================================================ TAAU, STAGE 0
//
// Our own compute dispatch writing X-Plane's FINAL upscaled image.
//
// Everything this needs was already built for other reasons and is only now
// pointed at the same target:
//
//   fsrprobe   names the image X-Plane's upscale writes - the one question
//              that genuinely could not be answered by inspection, settled by
//              stamping a sentinel and reading pixel (0,0) out of every
//              candidate.
//   g_allImages  records that image's format, extent and layer count at
//              creation, so the view can be built without guessing.
//   g_taa      already owns a view of the low-res resolved scene.
//
// Recorded immediately after X-Plane's EASU dispatch, in the same command
// buffer: EASU produces the image, we overwrite our share of it, and RCAS -
// which runs afterwards and reads that same image - sharpens OUR pixels along
// with X-Plane's. Ordering does the synchronisation that a fence would
// otherwise have to.
//
// STAGE 0 IS A SPATIAL UPSCALE ON PURPOSE. The temporal machinery (history at
// output resolution, reprojection by our own velocity) is the point of the
// exercise, but it is worth nothing until writing this image is proven to
// reach the screen. So: prove the write, then write the shader.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

namespace taau {

inline bool enabled()
{
    return live::onoff("taa.taau", "TAA_TAAU", false);
}

struct State {
    bool tried = false, ready = false;
    VkDevice dev = VK_NULL_HANDLE;

    PFN_vkCreateShaderModule        createSm   = nullptr;
    PFN_vkCreateDescriptorSetLayout createDsl  = nullptr;
    PFN_vkCreatePipelineLayout      createPl   = nullptr;
    PFN_vkCreateComputePipelines    createPipe = nullptr;
    PFN_vkCreateDescriptorPool      createPool = nullptr;
    PFN_vkAllocateDescriptorSets    allocSets  = nullptr;
    PFN_vkUpdateDescriptorSets      updSets    = nullptr;
    PFN_vkCreateSampler             createSamp = nullptr;
    PFN_vkCreateImageView           createView = nullptr;
    PFN_vkDestroyImageView          destroyView = nullptr;
    PFN_vkCmdBindPipeline           bindPipe   = nullptr;
    PFN_vkCmdBindDescriptorSets     bindSets   = nullptr;
    PFN_vkCmdPushConstants          pushConst  = nullptr;
    PFN_vkCmdDispatch               dispatch   = nullptr;
    PFN_vkCmdPipelineBarrier        barrier    = nullptr;

    VkShaderModule        sm   = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl  = VK_NULL_HANDLE;
    VkPipelineLayout      pl   = VK_NULL_HANDLE;
    VkPipeline            pipe = VK_NULL_HANDLE;
    VkDescriptorPool      pool = VK_NULL_HANDLE;
    VkSampler             samp = VK_NULL_HANDLE;     // LINEAR, for resampling
    VkSampler             sampNear = VK_NULL_HANDLE; // NEAREST, for velocity

    // A ring, for the same reason the resolve keeps one: a set written this
    // frame must not be one the GPU is still reading from two frames ago.
    static const uint32_t kSets = 8;
    VkDescriptorSet sets[kSets] = { VK_NULL_HANDLE };
    uint32_t nextSet = 0;

    // Views of X-Plane's output image, cached per handle. Purged when the
    // image dies - see the DestroyImage sweep in layer.cpp.
    std::map<VkImage, VkImageView> dstViews;

    // ---- THE HISTORY, AT OUTPUT RESOLUTION.
    //
    // Two of them, alternating: this frame reads one and writes the other,
    // because a single buffer read and written by the same dispatch has no
    // defined contents. Output resolution is the entire point - it is where
    // the extra detail accumulates, and a render-resolution history could not
    // hold detail the render does not have.
    VkImage        hist[2]     = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDeviceMemory histMem[2]  = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkImageView    histView[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    uint32_t histW = 0, histH = 0, histLayers = 0;
    bool     histLaidOut = false;   // one-time UNDEFINED -> GENERAL
    bool     histPrimed  = false;   // has anything ever been written into it
    uint32_t frame = 0;

    PFN_vkCreateImage        createImage  = nullptr;
    PFN_vkDestroyImage       destroyImage = nullptr;
    PFN_vkGetImageMemoryRequirements imgReq = nullptr;
    PFN_vkAllocateMemory     allocMem     = nullptr;
    PFN_vkFreeMemory         freeMem      = nullptr;
    PFN_vkBindImageMemory    bindImage    = nullptr;

    uint64_t dispatches = 0;
    bool     announced = false;
};

inline State &state() { static State s; return s; }

struct Push {
    int32_t outW, outH;
    int32_t inW, inH;
    float   shiftX, shiftY;
    float   alpha;
    float   gain;
    float   velScale;
    float   velYSign;
    float   split;
    float   sharpness;
    int32_t layer;
    int32_t flags;
};
enum {
    kTaauReset     = 1,
    kTaauTint      = 2,
    kTaauShowAlpha = 4,
};

inline bool init(DeviceData &dd, VkDevice dev)
{
    State &s = state();
    if (s.tried) return s.ready;
    s.tried = true;
    s.dev = dev;

    #define TAAU_GET(field, name) \
        s.field = (decltype(s.field))dd.gdpa(dev, name); if (!s.field) return false
    TAAU_GET(createSm,    "vkCreateShaderModule");
    TAAU_GET(createDsl,   "vkCreateDescriptorSetLayout");
    TAAU_GET(createPl,    "vkCreatePipelineLayout");
    TAAU_GET(createPipe,  "vkCreateComputePipelines");
    TAAU_GET(createPool,  "vkCreateDescriptorPool");
    TAAU_GET(allocSets,   "vkAllocateDescriptorSets");
    TAAU_GET(updSets,     "vkUpdateDescriptorSets");
    TAAU_GET(createSamp,  "vkCreateSampler");
    TAAU_GET(createView,  "vkCreateImageView");
    TAAU_GET(destroyView, "vkDestroyImageView");
    TAAU_GET(bindPipe,    "vkCmdBindPipeline");
    TAAU_GET(bindSets,    "vkCmdBindDescriptorSets");
    TAAU_GET(pushConst,   "vkCmdPushConstants");
    TAAU_GET(dispatch,    "vkCmdDispatch");
    TAAU_GET(barrier,     "vkCmdPipelineBarrier");
    TAAU_GET(createImage,  "vkCreateImage");
    TAAU_GET(destroyImage, "vkDestroyImage");
    TAAU_GET(imgReq,       "vkGetImageMemoryRequirements");
    TAAU_GET(allocMem,     "vkAllocateMemory");
    TAAU_GET(freeMem,      "vkFreeMemory");
    TAAU_GET(bindImage,    "vkBindImageMemory");
    #undef TAAU_GET

    VkShaderModuleCreateInfo smci;
    memset(&smci, 0, sizeof(smci));
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sizeof(kTaauSpv);
    smci.pCode = kTaauSpv;
    if (s.createSm(dev, &smci, nullptr, &s.sm) != VK_SUCCESS) return false;

    VkDescriptorSetLayoutBinding b[5];
    memset(b, 0, sizeof(b));
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; // src
    b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          // dst
    b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; // velocity
    b[3].binding = 3; b[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; // history read
    b[4].binding = 4; b[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          // history write
    for (int i = 0; i < 5; ++i) {
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dl;
    memset(&dl, 0, sizeof(dl));
    dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl.bindingCount = 5; dl.pBindings = b;
    if (s.createDsl(dev, &dl, nullptr, &s.dsl) != VK_SUCCESS) return false;

    VkPushConstantRange pr;
    pr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pr.offset = 0;
    pr.size = sizeof(Push);
    VkPipelineLayoutCreateInfo pli;
    memset(&pli, 0, sizeof(pli));
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1; pli.pSetLayouts = &s.dsl;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pr;
    if (s.createPl(dev, &pli, nullptr, &s.pl) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo cp;
    memset(&cp, 0, sizeof(cp));
    cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = s.sm;
    cp.stage.pName = "main";
    cp.layout = s.pl;
    if (s.createPipe(dev, VK_NULL_HANDLE, 1, &cp, nullptr, &s.pipe) != VK_SUCCESS)
        return false;

    VkDescriptorPoolSize psz[2];
    psz[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    psz[0].descriptorCount = 3 * State::kSets;
    psz[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    psz[1].descriptorCount = 2 * State::kSets;
    VkDescriptorPoolCreateInfo dp;
    memset(&dp, 0, sizeof(dp));
    dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp.maxSets = State::kSets; dp.poolSizeCount = 2; dp.pPoolSizes = psz;
    if (s.createPool(dev, &dp, nullptr, &s.pool) != VK_SUCCESS) return false;
    VkDescriptorSetLayout lays[State::kSets];
    for (uint32_t i = 0; i < State::kSets; ++i) lays[i] = s.dsl;
    VkDescriptorSetAllocateInfo da;
    memset(&da, 0, sizeof(da));
    da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    da.descriptorPool = s.pool;
    da.descriptorSetCount = State::kSets;
    da.pSetLayouts = lays;
    if (s.allocSets(dev, &da, s.sets) != VK_SUCCESS) return false;

    // LINEAR, and clamped: this is a resampling filter, so the hardware's
    // bilinear is doing half the work in the Catmull-Rom taps.
    VkSamplerCreateInfo sc;
    memset(&sc, 0, sizeof(sc));
    sc.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sc.magFilter = VK_FILTER_LINEAR;
    sc.minFilter = VK_FILTER_LINEAR;
    sc.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sc.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sc.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sc.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sc.maxLod = 0.25f;
    if (s.createSamp(dev, &sc, nullptr, &s.samp) != VK_SUCCESS) return false;

    // ---- VELOCITY IS SAMPLED NEAREST, AND THAT IS NOT A DETAIL.
    //
    // The velocity target carries a large negative sentinel in pixels no
    // patched shader wrote. Bilinear filtering blends that sentinel into
    // neighbouring real vectors, which marks good pixels unwritten and eats
    // history in a halo around every sky edge - the resolve carries the same
    // note over its own NEAREST sampler.
    sc.magFilter = VK_FILTER_NEAREST;
    sc.minFilter = VK_FILTER_NEAREST;
    if (s.createSamp(dev, &sc, nullptr, &s.sampNear) != VK_SUCCESS) return false;

    s.ready = true;
    trace("TAAU: stage-0 pipeline ready (spatial upscale, split proof).");
    return true;
}

// The output-resolution history pair. Rebuilt when the output size changes -
// a resolution change must not leave a history describing the old geometry,
// which is the failure the resolve has on record for exactly this reason.
// ---- RETIRE, DO NOT DESTROY. A resolution change happens while frames are
// still in flight, and destroying an image a queued dispatch is about to
// sample is a device-lost with a settings-change trigger. Retired objects wait
// out a few frames in the graveyard first - the same discipline the resolve
// uses for its own re-inits.
struct Retired {
    VkImage        img = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkImageView    view = VK_NULL_HANDLE;
    uint64_t       frame = 0;
};

inline std::vector<Retired> &graveyard()
{
    static std::vector<Retired> g;
    return g;
}

inline void sweepGraveyard(State &s, uint64_t now, bool force)
{
    std::vector<Retired> &g = graveyard();
    for (size_t i = 0; i < g.size(); ) {
        if (!force && now - g[i].frame < 8) { ++i; continue; }
        if (g[i].view && s.destroyView)  s.destroyView(s.dev, g[i].view, nullptr);
        if (g[i].img && s.destroyImage)  s.destroyImage(s.dev, g[i].img, nullptr);
        if (g[i].mem && s.freeMem)       s.freeMem(s.dev, g[i].mem, nullptr);
        g.erase(g.begin() + i);
    }
}

inline void freeHistory(State &s)
{
    for (int i = 0; i < 2; ++i) {
        if (s.hist[i] != VK_NULL_HANDLE || s.histView[i] != VK_NULL_HANDLE) {
            Retired r;
            r.img = s.hist[i]; r.mem = s.histMem[i]; r.view = s.histView[i];
            r.frame = s.frame;
            graveyard().push_back(r);
        }
        s.histView[i] = VK_NULL_HANDLE;
        s.hist[i] = VK_NULL_HANDLE;
        s.histMem[i] = VK_NULL_HANDLE;
    }
    s.histW = s.histH = s.histLayers = 0;
    s.histLaidOut = false;
    s.histPrimed = false;
}

inline bool ensureHistory(DeviceData &dd, State &s, uint32_t w, uint32_t h,
                          uint32_t layers, VkFormat fmt)
{
    if (s.hist[0] != VK_NULL_HANDLE && s.histW == w && s.histH == h &&
        s.histLayers == layers)
        return true;
    freeHistory(s);
    for (int i = 0; i < 2; ++i) {
        VkImageCreateInfo ic;
        memset(&ic, 0, sizeof(ic));
        ic.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ic.imageType = VK_IMAGE_TYPE_2D;
        ic.format = fmt;
        ic.extent.width = w; ic.extent.height = h; ic.extent.depth = 1;
        ic.mipLevels = 1;
        ic.arrayLayers = layers ? layers : 1;
        ic.samples = VK_SAMPLE_COUNT_1_BIT;
        ic.tiling = VK_IMAGE_TILING_OPTIMAL;
        ic.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ic.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (s.createImage(s.dev, &ic, nullptr, &s.hist[i]) != VK_SUCCESS) {
            freeHistory(s); return false;
        }
        VkMemoryRequirements mr;
        s.imgReq(s.dev, s.hist[i], &mr);
        VkMemoryAllocateInfo ma;
        memset(&ma, 0, sizeof(ma));
        ma.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ma.allocationSize = mr.size;
        ma.memoryTypeIndex = taaFindMemory(dd, mr.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (ma.memoryTypeIndex == UINT32_MAX ||
            s.allocMem(s.dev, &ma, nullptr, &s.histMem[i]) != VK_SUCCESS ||
            s.bindImage(s.dev, s.hist[i], s.histMem[i], 0) != VK_SUCCESS) {
            freeHistory(s); return false;
        }
        VkImageViewCreateInfo v;
        memset(&v, 0, sizeof(v));
        v.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        v.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        v.format = fmt;
        v.image = s.hist[i];
        v.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        v.subresourceRange.levelCount = 1;
        v.subresourceRange.layerCount = layers ? layers : 1;
        if (s.createView(s.dev, &v, nullptr, &s.histView[i]) != VK_SUCCESS) {
            freeHistory(s); return false;
        }
    }
    s.histW = w; s.histH = h; s.histLayers = layers;
    trace("TAAU: history pair allocated %ux%ux%u (%.1f MB total)",
          w, h, layers ? layers : 1,
          2.0 * (double)w * (double)h * (double)(layers ? layers : 1) * 8.0 / 1048576.0);
    return true;
}

// Cached view of X-Plane's output image. Storage-image view, so the format
// must be one; the creation record supplies it rather than an assumption.
inline VkImageView dstView(State &s, VkImage img, VkFormat fmt, uint32_t layers)
{
    std::map<VkImage, VkImageView>::iterator it = s.dstViews.find(img);
    if (it != s.dstViews.end()) return it->second;
    VkImageViewCreateInfo v;
    memset(&v, 0, sizeof(v));
    v.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    v.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    v.format = fmt;
    v.image = img;
    v.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    v.subresourceRange.levelCount = 1;
    v.subresourceRange.layerCount = layers ? layers : 1;
    VkImageView view = VK_NULL_HANDLE;
    if (s.createView(s.dev, &v, nullptr, &view) != VK_SUCCESS) {
        trace("TAAU: could not create a view of X-Plane's output image "
              "(fmt=%d) - stage 0 cannot write", (int)fmt);
        view = VK_NULL_HANDLE;
    }
    s.dstViews[img] = view;
    return view;
}

// Record our upscale into X-Plane's output image. Called right after its EASU
// dispatch has been forwarded, in the same command buffer.
inline void record(DeviceData &dd, VkDevice dev, VkCommandBuffer cb,
                   VkImage dstImg, VkFormat dstFmt,
                   uint32_t dstW, uint32_t dstH, uint32_t dstLayers,
                   VkImageView srcView, uint32_t srcW, uint32_t srcH,
                   VkImageView velView, float jitterNdcX, float jitterNdcY)
{
    if (!enabled()) return;
    if (dstImg == VK_NULL_HANDLE || srcView == VK_NULL_HANDLE) return;
    if (velView == VK_NULL_HANDLE) return;
    if (!dstW || !dstH || !srcW || !srcH) return;
    if (!init(dd, dev)) return;
    State &s = state();

    VkImageView dv = dstView(s, dstImg, dstFmt, dstLayers);
    if (dv == VK_NULL_HANDLE) return;

    // ---- HISTORY IN THE SAME FORMAT AS THE TARGET.
    //
    // Matching the destination means no conversion on the way out and no
    // precision cliff on the way in; it is also HDR, which the accumulation
    // needs since the scene is pre-tonemap.
    if (!ensureHistory(dd, s, dstW, dstH, dstLayers ? dstLayers : 1, dstFmt)) {
        static bool said = false;
        if (!said) { said = true;
            trace("TAAU: history allocation failed - upsampling disabled"); }
        return;
    }

    // One-time UNDEFINED -> GENERAL for both history images. Sampling or
    // storing to an image still in UNDEFINED is the same class of invalid use
    // the sun dummy needed fixing for.
    if (!s.histLaidOut) {
        VkImageMemoryBarrier hb[2];
        memset(hb, 0, sizeof(hb));
        for (int i = 0; i < 2; ++i) {
            hb[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            hb[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            hb[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            hb[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            hb[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            hb[i].image = s.hist[i];
            hb[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            hb[i].subresourceRange.levelCount = 1;
            hb[i].subresourceRange.layerCount = s.histLayers ? s.histLayers : 1;
            hb[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                  VK_ACCESS_SHADER_WRITE_BIT;
        }
        s.barrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                  0, nullptr, 2, hb);
        s.histLaidOut = true;
    }

    if (!s.announced) {
        s.announced = true;
        trace("TAAU: temporal upsample %ux%u -> %ux%u (%.0f%% scale), writing "
              "X-Plane's output image %p (fmt=%d, %u layer(s)). "
              "taa.taau_split=%.2f of the width is ours; the rest stays "
              "X-Plane's own spatial upscale for comparison.",
              srcW, srcH, dstW, dstH,
              100.0 * (double)srcW / (double)(dstW ? dstW : 1),
              (void*)dstImg, (int)dstFmt, dstLayers,
              live::f("taa.taau_split", "TAA_TAAU_SPLIT", 0.5f));
    }

    // WAW against EASU's write, and RAW against our own source. EASU wrote the
    // destination as a storage image and the resolve wrote the source, so both
    // hazards are shader-write -> shader-read/write in COMPUTE.
    VkMemoryBarrier mb;
    memset(&mb, 0, sizeof(mb));
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    s.barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr,
              0, nullptr);

    sweepGraveyard(s, s.frame, false);

    // Alternate every frame: read what last frame wrote, write the other.
    const uint32_t rd = s.frame & 1u;
    const uint32_t wrIdx = 1u - rd;
    ++s.frame;

    // ---- THE UNJITTER SHIFT, IN THE RESOLVE'S OWN ARITHMETIC.
    //
    // g_jitLastX/Y hold the NDC offset actually applied to the viewport this
    // frame, and sMulX/sMulY are the swept conversion to a UV shift. Reusing
    // both rather than re-deriving them is deliberate: this pair has been
    // wrong in this project before, and a second derivation would be a second
    // chance to get it wrong independently.
    const float sMulX = live::f("taa.smul_x", "TAA_SMUL_X",  0.5f);
    const float sMulY = live::f("taa.smul_y", "TAA_SMUL_Y", -0.5f);
    const bool  unjit = live::onoff("taa.unjitter", "TAA_UNJITTER", true);

    const uint32_t layers = dstLayers ? dstLayers : 1;
    for (uint32_t l = 0; l < layers; ++l) {
        const uint32_t si = s.nextSet;
        s.nextSet = (si + 1) % State::kSets;
        VkDescriptorSet set = s.sets[si];

        VkDescriptorImageInfo ii[5];
        memset(ii, 0, sizeof(ii));
        ii[0].sampler = s.samp;
        ii[0].imageView = srcView;
        // SHADER_READ_ONLY_OPTIMAL, matching what the resolve binds this same
        // view with - X-Plane's EASU is sampling it at this very moment, so
        // that is provably its layout here. Claiming GENERAL would be a lie
        // the validator catches and the driver may not.
        ii[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ii[1].imageView = dv;
        ii[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ii[2].sampler = s.sampNear;
        ii[2].imageView = velView;
        ii[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ii[3].sampler = s.samp;
        ii[3].imageView = s.histView[rd];
        ii[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ii[4].imageView = s.histView[wrIdx];
        ii[4].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet wr[5];
        memset(wr, 0, sizeof(wr));
        for (int k = 0; k < 5; ++k) {
            wr[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[k].dstSet = set;
            wr[k].dstBinding = (uint32_t)k;
            wr[k].descriptorCount = 1;
            wr[k].pImageInfo = &ii[k];
            wr[k].descriptorType = (k == 1 || k == 4)
                ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        }
        s.updSets(dev, 5, wr, 0, nullptr);

        Push p;
        p.outW = (int32_t)dstW; p.outH = (int32_t)dstH;
        p.inW  = (int32_t)srcW; p.inH  = (int32_t)srcH;
        p.shiftX = unjit ? sMulX * jitterNdcX : 0.0f;
        p.shiftY = unjit ? sMulY * jitterNdcY : 0.0f;
        // Lower than the resolve's alpha by design: at this ratio one frame
        // carries only (in/out)^2 of the output's samples, so the detail comes
        // from many frames landing on different sub-pixel positions.
        p.alpha     = live::f("taa.taau_alpha", "TAA_TAAU_ALPHA", 0.12f);
        p.gain      = live::f("taa.taau_gain",  "TAA_TAAU_GAIN",  4.0f);
        p.velScale  = taaVelScale();
        p.velYSign  = taaVelYSign();
        p.split     = live::f("taa.taau_split", "TAA_TAAU_SPLIT", 0.5f);
        p.sharpness = live::f("taa.taau_sharp", "TAA_TAAU_SHARP", 1.0f);
        p.layer     = (int32_t)l;
        p.flags = 0;
        if (!s.histPrimed) p.flags |= kTaauReset;
        if (live::onoff("taa.taau_tint", "TAA_TAAU_TINT", false))
            p.flags |= kTaauTint;
        if (live::onoff("taa.taau_show_alpha", "TAA_TAAU_SHOW_ALPHA", false))
            p.flags |= kTaauShowAlpha;

        s.bindPipe(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipe);
        s.bindSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pl, 0, 1, &set,
                   0, nullptr);
        s.pushConst(cb, s.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        s.dispatch(cb, (dstW + 7) / 8, (dstH + 7) / 8, 1);
    }
    s.histPrimed = true;

    if ((s.dispatches++ % 600) == 0)
        trace("TAAU: %llu dispatches (%ux%u -> %ux%u, %u layer(s))",
              (unsigned long long)s.dispatches, srcW, srcH, dstW, dstH, layers);
}

inline void shutdown()
{
    State &s = state();
    if (s.dev == VK_NULL_HANDLE) return;
    freeHistory(s);
    // The device is going away, so nothing can still be reading: force it.
    sweepGraveyard(s, s.frame, true);
    for (std::map<VkImage, VkImageView>::iterator it = s.dstViews.begin();
         it != s.dstViews.end(); ++it)
        if (it->second && s.destroyView) s.destroyView(s.dev, it->second, nullptr);
    s.dstViews.clear();
}

inline void noteImageGone(VkImage img)
{
    State &s = state();
    std::map<VkImage, VkImageView>::iterator it = s.dstViews.find(img);
    if (it == s.dstViews.end()) return;
    if (it->second && s.destroyView && s.dev != VK_NULL_HANDLE)
        s.destroyView(s.dev, it->second, nullptr);
    s.dstViews.erase(it);
}

} // namespace taau
