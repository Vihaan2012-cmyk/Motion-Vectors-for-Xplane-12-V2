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
    VkSampler             samp = VK_NULL_HANDLE;

    // A ring, for the same reason the resolve keeps one: a set written this
    // frame must not be one the GPU is still reading from two frames ago.
    static const uint32_t kSets = 8;
    VkDescriptorSet sets[kSets] = { VK_NULL_HANDLE };
    uint32_t nextSet = 0;

    // Views of X-Plane's output image, cached per handle. Purged when the
    // image dies - see the DestroyImage sweep in layer.cpp.
    std::map<VkImage, VkImageView> dstViews;

    uint64_t dispatches = 0;
    bool     announced = false;
};

inline State &state() { static State s; return s; }

struct Push {
    int32_t outW, outH;
    int32_t inW, inH;
    float   split;
    float   sharpness;
    int32_t layer;
    int32_t debugTint;
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
    #undef TAAU_GET

    VkShaderModuleCreateInfo smci;
    memset(&smci, 0, sizeof(smci));
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sizeof(kTaauSpv);
    smci.pCode = kTaauSpv;
    if (s.createSm(dev, &smci, nullptr, &s.sm) != VK_SUCCESS) return false;

    VkDescriptorSetLayoutBinding b[2];
    memset(b, 0, sizeof(b));
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    for (int i = 0; i < 2; ++i) {
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dl;
    memset(&dl, 0, sizeof(dl));
    dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl.bindingCount = 2; dl.pBindings = b;
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
    psz[0].descriptorCount = State::kSets;
    psz[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    psz[1].descriptorCount = State::kSets;
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

    s.ready = true;
    trace("TAAU: stage-0 pipeline ready (spatial upscale, split proof).");
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
                   VkImageView srcView, uint32_t srcW, uint32_t srcH)
{
    if (!enabled()) return;
    if (dstImg == VK_NULL_HANDLE || srcView == VK_NULL_HANDLE) return;
    if (!dstW || !dstH || !srcW || !srcH) return;
    if (!init(dd, dev)) return;
    State &s = state();

    VkImageView dv = dstView(s, dstImg, dstFmt, dstLayers);
    if (dv == VK_NULL_HANDLE) return;

    if (!s.announced) {
        s.announced = true;
        trace("TAAU: writing X-Plane's output image %p (%ux%u, fmt=%d, %u "
              "layer(s)) from %ux%u. taa.taau_split=%.2f of the width is ours; "
              "the rest stays X-Plane's own upscale for comparison.",
              (void*)dstImg, dstW, dstH, (int)dstFmt, dstLayers, srcW, srcH,
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

    const uint32_t layers = dstLayers ? dstLayers : 1;
    for (uint32_t l = 0; l < layers; ++l) {
        const uint32_t si = s.nextSet;
        s.nextSet = (si + 1) % State::kSets;
        VkDescriptorSet set = s.sets[si];

        VkDescriptorImageInfo ii[2];
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

        VkWriteDescriptorSet wr[2];
        memset(wr, 0, sizeof(wr));
        for (int k = 0; k < 2; ++k) {
            wr[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[k].dstSet = set;
            wr[k].dstBinding = (uint32_t)k;
            wr[k].descriptorCount = 1;
            wr[k].pImageInfo = &ii[k];
        }
        wr[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        wr[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        s.updSets(dev, 2, wr, 0, nullptr);

        Push p;
        p.outW = (int32_t)dstW; p.outH = (int32_t)dstH;
        p.inW  = (int32_t)srcW; p.inH  = (int32_t)srcH;
        p.split     = live::f("taa.taau_split", "TAA_TAAU_SPLIT", 0.5f);
        p.sharpness = live::f("taa.taau_sharp", "TAA_TAAU_SHARP", 1.0f);
        p.layer     = (int32_t)l;
        p.debugTint = live::onoff("taa.taau_tint", "TAA_TAAU_TINT", false) ? 1 : 0;

        s.bindPipe(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipe);
        s.bindSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pl, 0, 1, &set,
                   0, nullptr);
        s.pushConst(cb, s.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        s.dispatch(cb, (dstW + 7) / 8, (dstH + 7) / 8, 1);
    }

    if ((s.dispatches++ % 600) == 0)
        trace("TAAU: %llu dispatches (%ux%u -> %ux%u, %u layer(s))",
              (unsigned long long)s.dispatches, srcW, srcH, dstW, dstH, layers);
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
