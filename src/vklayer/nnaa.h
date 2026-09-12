#pragma once
// ================================================================ NNAA
//
// The native-resolution anti-aliasing net, run inside the resolve.
//
// The resolve (taa.comp, FLAG_NN) writes the net's input tensor - the
// unjittered current-frame fetch, the real jitter, depth / normal / vector
// planes and the warped history - into a six-layer rgba16f image (binding 15).
// This module then runs the net's eleven 3x3 convolutions as compute
// dispatches over rgba16f feature maps (nn_conv.comp) and a post pass
// (nn_post.comp) that forms base + tanh(r) * (0.5 + |base|) and writes it into
// the history image for the left `taa.nn_split` of the frame. The right side
// stays the resolve's own output: the seam is the A/B.
//
// Everything about the net (channel widths, layer order, the concat order of
// the U-Net skips) comes from the weight blob written by tools/nn/export_aa.py,
// which mirrors quick_train.ResidualSR / train_aa.AANet. Weights are looked
// for at TAA_NN_WEIGHTS, then the live key taa.nn_weights, then nn_aa.bin
// beside this DLL, then %TEMP%\nn_aa.bin.
//
// taa.nn_dump=1 copies one frame's input map, the last conv output and the
// history image to %TEMP%\nn_aa_dump.bin so tools/nn/nn_check.py can run the
// same tensor through torch and diff the two - the numeric proof that the
// shader IS the model, which no screenshot can give.
//
// Feature maps live at full, half and quarter resolution as image arrays with
// four channels per layer; at 1440p the whole set is ~500 MB in fp16. That is
// the v1 cost of not touching device features; a coopmat / 16-bit-storage
// path can halve it later without changing this module's contract.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <stdio.h>
#include <string>
#include <vector>
#include "nn_conv_spv.h"
#include "nn_post_spv.h"

namespace nnaa {

inline bool enabled() { return live::onoff("taa.nn", "TAA_NN", false); }

struct Map {
    VkImage        img  = VK_NULL_HANDLE;
    VkDeviceMemory mem  = VK_NULL_HANDLE;
    VkImageView    view = VK_NULL_HANDLE;
    uint32_t w = 0, h = 0, layers = 0;
    bool laidOut = false;
};

struct Layer { int cout = 0, cin = 0, ksize = 3; uint32_t wOff = 0, bOff = 0; };

struct State {
    bool tried = false, ready = false;
    VkDevice dev = VK_NULL_HANDLE;
    VkShaderModule smConv = VK_NULL_HANDLE, smPost = VK_NULL_HANDLE;
    VkDescriptorSetLayout dslConv = VK_NULL_HANDLE, dslPost = VK_NULL_HANDLE;
    VkPipelineLayout plConv = VK_NULL_HANDLE, plPost = VK_NULL_HANDLE;
    VkPipeline pipeConv = VK_NULL_HANDLE, pipePost = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    static const uint32_t kSets = 96;    // 12 dispatches per frame, several frames in flight
    VkDescriptorSet sets[kSets];
    uint32_t nextSet = 0;

    VkBuffer       wbuf = VK_NULL_HANDLE;
    VkDeviceMemory wmem = VK_NULL_HANDLE;
    std::vector<Layer> layers;
    int inCh = 0, ch0 = 0, ch1 = 0, ch2 = 0;
    bool weightsTried = false, weightsOk = false;
    std::string weightsPath;

    Map in, a, b, g, out, c, d, h, e, f;
    uint32_t w = 0, hh = 0;
    uint64_t frame = 0, dispatches = 0;
    bool announced = false, sizeWarned = false;

    // one-shot dump
    VkBuffer       dbuf = VK_NULL_HANDLE;
    VkDeviceMemory dmem = VK_NULL_HANDLE;
    void          *dptr = nullptr;
    VkDeviceSize   dbytes = 0;
    uint64_t       dumpFrame = 0;
    int            dumpState = 0;   // 0 idle, 1 copied and waiting, 2 written
};

inline State &state() { static State s; return s; }

struct ConvPush {
    int32_t outW, outH, aW, aH, bW, bH;
    int32_t cinA, cinB, cout, wOff, bOff, modeA, modeB, act;
};
struct PostPush {
    int32_t w, h, layer, viz;
    float   split, blend;
};
static_assert(sizeof(ConvPush) == 56, "nn_conv.comp push block mismatch");
static_assert(sizeof(PostPush) == 24, "nn_post.comp push block mismatch");

// ---------------------------------------------------------------- resources
inline void freeMap(State &s, DeviceData &dd, Map &m)
{
    if (m.view) dd.destroyImageView(s.dev, m.view, nullptr);
    if (m.img)  dd.destroyImage(s.dev, m.img, nullptr);
    if (m.mem)  dd.freeMemory(s.dev, m.mem, nullptr);
    m = Map();
}

inline bool makeMap(State &s, DeviceData &dd, Map &m, uint32_t w, uint32_t h, uint32_t layers, VkImageUsageFlags extra = 0)
{
    if (m.img && m.w == w && m.h == h && m.layers == layers) return true;
    freeMap(s, dd, m);
    VkImageCreateInfo ic; memset(&ic, 0, sizeof(ic));
    ic.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ic.imageType = VK_IMAGE_TYPE_2D;
    ic.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    ic.extent.width = w; ic.extent.height = h; ic.extent.depth = 1;
    ic.mipLevels = 1; ic.arrayLayers = layers;
    ic.samples = VK_SAMPLE_COUNT_1_BIT;
    ic.tiling = VK_IMAGE_TILING_OPTIMAL;
    ic.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | extra;
    ic.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (dd.createImage(s.dev, &ic, nullptr, &m.img) != VK_SUCCESS) { m.img = VK_NULL_HANDLE; return false; }
    VkMemoryRequirements mr; dd.getImageMemReq(s.dev, m.img, &mr);
    VkMemoryAllocateInfo ma; memset(&ma, 0, sizeof(ma));
    ma.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; ma.allocationSize = mr.size;
    ma.memoryTypeIndex = taaFindMemory(dd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (ma.memoryTypeIndex == UINT32_MAX || dd.allocateMemory(s.dev, &ma, nullptr, &m.mem) != VK_SUCCESS ||
        dd.bindImageMemory(s.dev, m.img, m.mem, 0) != VK_SUCCESS) { freeMap(s, dd, m); return false; }
    VkImageViewCreateInfo v; memset(&v, 0, sizeof(v));
    v.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    v.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY; v.format = ic.format; v.image = m.img;
    v.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    v.subresourceRange.levelCount = 1; v.subresourceRange.layerCount = layers;
    if (dd.createImageView(s.dev, &v, nullptr, &m.view) != VK_SUCCESS) { freeMap(s, dd, m); return false; }
    m.w = w; m.h = h; m.layers = layers; m.laidOut = false;
    return true;
}

// UNDEFINED -> GENERAL once per map, recorded into the frame's command buffer.
inline void layoutMap(DeviceData &dd, VkCommandBuffer cb, Map &m)
{
    if (m.laidOut || !m.img) return;
    VkImageMemoryBarrier b; memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = m.img;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1; b.subresourceRange.layerCount = m.layers;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 0, nullptr, 0, nullptr, 1, &b);
    m.laidOut = true;
}

inline void computeBarrier(DeviceData &dd, VkCommandBuffer cb)
{
    VkMemoryBarrier mb; memset(&mb, 0, sizeof(mb));
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 1, &mb, 0, nullptr, 0, nullptr);
}

// ---------------------------------------------------------------- weights
inline std::string weightsPath()
{
    std::string p;
    if (const char *e = getenv("TAA_NN_WEIGHTS")) if (*e) return e;
    if (live::lookup("taa.nn_weights", p) && !p.empty()) return p;
    char mod[MAX_PATH]; mod[0] = 0;
    HMODULE self = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)&state, &self) && GetModuleFileNameA(self, mod, sizeof(mod) - 1)) {
        std::string d = mod; size_t k = d.find_last_of("\\/");
        if (k != std::string::npos) {
            std::string cand = d.substr(0, k + 1) + "nn_aa.bin";
            FILE *f = fopen(cand.c_str(), "rb"); if (f) { fclose(f); return cand; }
        }
    }
    const char *t = getenv("TEMP");
    return std::string(t ? t : ".") + "\\nn_aa.bin";
}

inline bool loadWeights(State &s, DeviceData &dd)
{
    if (s.weightsTried) return s.weightsOk;
    s.weightsTried = true;
    s.weightsPath = weightsPath();
    FILE *f = fopen(s.weightsPath.c_str(), "rb");
    if (!f) { trace("NNAA: no weight blob at %s - taa.nn stays off", s.weightsPath.c_str()); return false; }
    char magic[4]; uint32_t hdr[3];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "MVNN", 4) != 0 || fread(hdr, 4, 3, f) != 3 || hdr[0] != 1) {
        fclose(f); trace("NNAA: %s is not a version-1 MVNN blob", s.weightsPath.c_str()); return false;
    }
    const uint32_t nl = hdr[2];
    if (nl != 11 || hdr[1] != 17) {
        fclose(f); trace("NNAA: blob has %u layers / %u input channels; this module expects 11 / 17", nl, hdr[1]); return false;
    }
    s.inCh = (int)hdr[1];
    s.layers.assign(nl, Layer());
    uint32_t off = 0;
    for (uint32_t i = 0; i < nl; ++i) {
        uint32_t d[3];
        if (fread(d, 4, 3, f) != 3 || d[2] != 3 || d[0] > 24 || d[0] == 0 || d[1] == 0) { fclose(f); trace("NNAA: bad layer %u in blob", i); return false; }
        s.layers[i].cout = (int)d[0]; s.layers[i].cin = (int)d[1]; s.layers[i].ksize = 3;
        s.layers[i].wOff = off; off += d[0] * d[1] * 9;
        s.layers[i].bOff = off; off += d[0];
    }
    std::vector<float> w(off);
    if (fread(w.data(), 4, off, f) != off) { fclose(f); trace("NNAA: blob truncated"); return false; }
    fclose(f);
    // The U-Net's shape, read off the layers so the maps are sized from the file, not a constant.
    const Layer *L = s.layers.data();
    s.ch0 = L[0].cout; s.ch1 = L[2].cout; s.ch2 = L[4].cout;
    const bool shapeOk = L[0].cin == s.inCh && L[1].cin == s.ch0 && L[1].cout == s.ch0 && L[2].cin == s.ch0 &&
        L[3].cin == s.ch1 && L[3].cout == s.ch1 && L[4].cin == s.ch1 && L[5].cin == s.ch2 && L[5].cout == s.ch2 &&
        L[6].cin == s.ch2 + s.ch1 && L[6].cout == s.ch1 && L[7].cin == s.ch1 && L[7].cout == s.ch1 &&
        L[8].cin == s.ch1 + s.ch0 && L[8].cout == s.ch0 && L[9].cin == s.ch0 && L[9].cout == s.ch0 &&
        L[10].cin == s.ch0 && L[10].cout == 3;
    if (!shapeOk) { trace("NNAA: blob layer shapes do not form the expected U-Net (ch %d,%d,%d)", s.ch0, s.ch1, s.ch2); return false; }

    VkBufferCreateInfo bc; memset(&bc, 0, sizeof(bc));
    bc.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bc.size = (VkDeviceSize)off * 4; bc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bc.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (dd.createBuffer(s.dev, &bc, nullptr, &s.wbuf) != VK_SUCCESS) { trace("NNAA: weight buffer creation failed"); return false; }
    VkMemoryRequirements mr; dd.getBufferMemReq(s.dev, s.wbuf, &mr);
    VkMemoryAllocateInfo ma; memset(&ma, 0, sizeof(ma));
    ma.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; ma.allocationSize = mr.size;
    ma.memoryTypeIndex = taaFindMemory(dd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void *p = nullptr;
    if (ma.memoryTypeIndex == UINT32_MAX || dd.allocateMemory(s.dev, &ma, nullptr, &s.wmem) != VK_SUCCESS ||
        dd.bindBufferMemory(s.dev, s.wbuf, s.wmem, 0) != VK_SUCCESS || dd.mapMemory(s.dev, s.wmem, 0, bc.size, 0, &p) != VK_SUCCESS) {
        trace("NNAA: weight memory failed"); return false;
    }
    memcpy(p, w.data(), bc.size);
    dd.unmapMemory(s.dev, s.wmem);
    s.weightsOk = true;
    trace("NNAA: loaded %s - 11 layers, channels (%d,%d,%d), %u floats", s.weightsPath.c_str(), s.ch0, s.ch1, s.ch2, off);
    return true;
}

// ---------------------------------------------------------------- pipelines
inline bool init(State &s, DeviceData &dd)
{
    if (s.tried) return s.ready;
    s.tried = true;
    s.dev = dd.device;
    for (uint32_t i = 0; i < State::kSets; ++i) s.sets[i] = VK_NULL_HANDLE;

    VkShaderModuleCreateInfo sm; memset(&sm, 0, sizeof(sm));
    sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm.codeSize = sizeof(kNnConvSpv); sm.pCode = kNnConvSpv;
    if (dd.createShaderModule(s.dev, &sm, nullptr, &s.smConv) != VK_SUCCESS) return false;
    sm.codeSize = sizeof(kNnPostSpv); sm.pCode = kNnPostSpv;
    if (dd.createShaderModule(s.dev, &sm, nullptr, &s.smPost) != VK_SUCCESS) return false;

    // conv: 3 storage images + the weight buffer; post: 3 storage images
    VkDescriptorSetLayoutBinding b[4]; memset(b, 0, sizeof(b));
    for (int i = 0; i < 4; ++i) { b[i].binding = (uint32_t)i; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; }
    b[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    VkDescriptorSetLayoutCreateInfo dl; memset(&dl, 0, sizeof(dl));
    dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl.bindingCount = 4; dl.pBindings = b;
    if (dd.createDescriptorSetLayout(s.dev, &dl, nullptr, &s.dslConv) != VK_SUCCESS) return false;
    dl.bindingCount = 3;
    if (dd.createDescriptorSetLayout(s.dev, &dl, nullptr, &s.dslPost) != VK_SUCCESS) return false;

    VkPushConstantRange pr; pr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pr.offset = 0; pr.size = sizeof(ConvPush);
    VkPipelineLayoutCreateInfo pl; memset(&pl, 0, sizeof(pl));
    pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1; pl.pSetLayouts = &s.dslConv; pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &pr;
    if (dd.createPipelineLayout(s.dev, &pl, nullptr, &s.plConv) != VK_SUCCESS) return false;
    pr.size = sizeof(PostPush); pl.pSetLayouts = &s.dslPost;
    if (dd.createPipelineLayout(s.dev, &pl, nullptr, &s.plPost) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo cp; memset(&cp, 0, sizeof(cp));
    cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cp.stage.pName = "main";
    cp.stage.module = s.smConv; cp.layout = s.plConv;
    if (dd.createComputePipelines(s.dev, VK_NULL_HANDLE, 1, &cp, nullptr, &s.pipeConv) != VK_SUCCESS) return false;
    cp.stage.module = s.smPost; cp.layout = s.plPost;
    if (dd.createComputePipelines(s.dev, VK_NULL_HANDLE, 1, &cp, nullptr, &s.pipePost) != VK_SUCCESS) return false;

    VkDescriptorPoolSize ps[2];
    ps[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  ps[0].descriptorCount = 3 * State::kSets;
    ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; ps[1].descriptorCount = 1 * State::kSets;
    VkDescriptorPoolCreateInfo dp; memset(&dp, 0, sizeof(dp));
    dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp.maxSets = State::kSets; dp.poolSizeCount = 2; dp.pPoolSizes = ps;
    if (dd.createDescriptorPool(s.dev, &dp, nullptr, &s.pool) != VK_SUCCESS) return false;
    // Half the ring is conv sets, half post sets: alternate layouts by index parity.
    VkDescriptorSetLayout lays[State::kSets];
    for (uint32_t i = 0; i < State::kSets; ++i) lays[i] = (i % 2 == 0) ? s.dslConv : s.dslPost;
    VkDescriptorSetAllocateInfo da; memset(&da, 0, sizeof(da));
    da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    da.descriptorPool = s.pool; da.descriptorSetCount = State::kSets; da.pSetLayouts = lays;
    if (dd.allocateDescriptorSets(s.dev, &da, s.sets) != VK_SUCCESS) return false;
    s.ready = true;
    trace("NNAA: pipelines ready (conv %u words, post %u words)", (unsigned)(sizeof(kNnConvSpv) / 4), (unsigned)(sizeof(kNnPostSpv) / 4));
    return true;
}

inline VkDescriptorSet takeSet(State &s, bool conv)
{
    // even indices hold conv layouts, odd hold post layouts
    for (int tries = 0; tries < (int)State::kSets; ++tries) {
        uint32_t i = s.nextSet; s.nextSet = (s.nextSet + 1) % State::kSets;
        if ((i % 2 == 0) == conv) return s.sets[i];
    }
    return VK_NULL_HANDLE;
}

// The input map the resolve writes (binding 15). Created and laid out here so the
// resolve's descriptor write can name it before its own dispatch.
inline VkImageView prepareInput(DeviceData &dd, VkCommandBuffer cb, uint32_t w, uint32_t h, uint32_t layers)
{
    State &s = state();
    if (!enabled()) return VK_NULL_HANDLE;
    if (!init(s, dd)) return VK_NULL_HANDLE;
    if (!loadWeights(s, dd)) return VK_NULL_HANDLE;
    if (layers != 1 || (w & 3) || (h & 3)) {
        if (!s.sizeWarned) { s.sizeWarned = true;
            trace("NNAA: unsupported target %ux%u x%u (needs one layer and sizes divisible by 4) - taa.nn inactive", w, h, layers); }
        return VK_NULL_HANDLE;
    }
    if (!makeMap(s, dd, s.in, w, h, 6)) { trace("NNAA: input map allocation failed"); return VK_NULL_HANDLE; }
    layoutMap(dd, cb, s.in);
    s.w = w; s.hh = h;
    return s.in.view;
}

inline uint32_t groups(int ch) { return (uint32_t)((ch + 3) / 4); }

inline bool ensureMaps(State &s, DeviceData &dd, VkCommandBuffer cb)
{
    const uint32_t w = s.w, h = s.hh, w2 = w / 2, h2 = h / 2, w4 = w / 4, h4 = h / 4;
    bool ok = makeMap(s, dd, s.a, w, h, groups(s.ch0)) && makeMap(s, dd, s.b, w, h, groups(s.ch0)) &&
              makeMap(s, dd, s.g, w, h, groups(s.ch0)) && makeMap(s, dd, s.out, w, h, 1) &&
              makeMap(s, dd, s.c, w2, h2, groups(s.ch1)) && makeMap(s, dd, s.d, w2, h2, groups(s.ch1)) &&
              makeMap(s, dd, s.h, w2, h2, groups(s.ch1)) &&
              makeMap(s, dd, s.e, w4, h4, groups(s.ch2)) && makeMap(s, dd, s.f, w4, h4, groups(s.ch2));
    if (!ok) return false;
    Map *all[] = { &s.a, &s.b, &s.g, &s.out, &s.c, &s.d, &s.h, &s.e, &s.f };
    for (Map *m : all) layoutMap(dd, cb, *m);
    return true;
}

inline void conv(State &s, DeviceData &dd, VkCommandBuffer cb, const Layer &L,
                 Map &A, int modeA, int cinA, Map *B, int cinB, Map &out, int act)
{
    VkDescriptorSet set = takeSet(s, true);
    VkDescriptorImageInfo ii[3]; memset(ii, 0, sizeof(ii));
    ii[0].imageView = A.view; ii[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ii[1].imageView = B ? B->view : A.view; ii[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ii[2].imageView = out.view; ii[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorBufferInfo bi; bi.buffer = s.wbuf; bi.offset = 0; bi.range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet wr[4]; memset(wr, 0, sizeof(wr));
    for (int k = 0; k < 4; ++k) {
        wr[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr[k].dstSet = set; wr[k].dstBinding = (uint32_t)k;
        wr[k].descriptorCount = 1;
        wr[k].descriptorType = k == 3 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        if (k < 3) wr[k].pImageInfo = &ii[k]; else wr[k].pBufferInfo = &bi;
    }
    dd.updateDescriptorSets(s.dev, 4, wr, 0, nullptr);
    ConvPush p;
    p.outW = (int32_t)out.w; p.outH = (int32_t)out.h;
    p.aW = (int32_t)A.w; p.aH = (int32_t)A.h;
    p.bW = B ? (int32_t)B->w : (int32_t)A.w; p.bH = B ? (int32_t)B->h : (int32_t)A.h;
    p.cinA = cinA; p.cinB = B ? cinB : 0; p.cout = L.cout;
    p.wOff = (int32_t)L.wOff; p.bOff = (int32_t)L.bOff;
    p.modeA = modeA; p.modeB = 0; p.act = act;
    dd.cmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipeConv);
    dd.cmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.plConv, 0, 1, &set, 0, nullptr);
    dd.cmdPushConstants(cb, s.plConv, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
    dd.cmdDispatch(cb, (out.w + 7) / 8, (out.h + 7) / 8, 1);
    computeBarrier(dd, cb);
    ++s.dispatches;
}

// ---------------------------------------------------------------- dump
inline void dumpBegin(State &s, DeviceData &dd, VkCommandBuffer cb, VkImage histImg)
{
    const uint32_t planes = 8;                                // in[0..5], out, history
    const VkDeviceSize plane = (VkDeviceSize)s.w * s.hh * 8;  // rgba16f
    s.dbytes = plane * planes;
    VkBufferCreateInfo bc; memset(&bc, 0, sizeof(bc));
    bc.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bc.size = s.dbytes;
    bc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT; bc.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (dd.createBuffer(s.dev, &bc, nullptr, &s.dbuf) != VK_SUCCESS) { s.dumpState = 2; return; }
    VkMemoryRequirements mr; dd.getBufferMemReq(s.dev, s.dbuf, &mr);
    VkMemoryAllocateInfo ma; memset(&ma, 0, sizeof(ma));
    ma.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; ma.allocationSize = mr.size;
    ma.memoryTypeIndex = taaFindMemory(dd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (ma.memoryTypeIndex == UINT32_MAX || dd.allocateMemory(s.dev, &ma, nullptr, &s.dmem) != VK_SUCCESS ||
        dd.bindBufferMemory(s.dev, s.dbuf, s.dmem, 0) != VK_SUCCESS || dd.mapMemory(s.dev, s.dmem, 0, s.dbytes, 0, &s.dptr) != VK_SUCCESS) {
        trace("NNAA: dump buffer failed"); s.dumpState = 2; return;
    }
    // shader writes -> transfer reads on the three images (all GENERAL)
    VkMemoryBarrier mb; memset(&mb, 0, sizeof(mb));
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    VkBufferImageCopy r; memset(&r, 0, sizeof(r));
    r.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; r.imageSubresource.layerCount = 1;
    r.imageExtent.width = s.w; r.imageExtent.height = s.hh; r.imageExtent.depth = 1;
    for (uint32_t l = 0; l < 6; ++l) {
        r.bufferOffset = plane * l; r.imageSubresource.baseArrayLayer = l;
        dd.cmdCopyImageToBuffer(cb, s.in.img, VK_IMAGE_LAYOUT_GENERAL, s.dbuf, 1, &r);
    }
    r.bufferOffset = plane * 6; r.imageSubresource.baseArrayLayer = 0;
    dd.cmdCopyImageToBuffer(cb, s.out.img, VK_IMAGE_LAYOUT_GENERAL, s.dbuf, 1, &r);
    r.bufferOffset = plane * 7;
    dd.cmdCopyImageToBuffer(cb, histImg, VK_IMAGE_LAYOUT_GENERAL, s.dbuf, 1, &r);
    mb.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    s.dumpFrame = s.frame; s.dumpState = 1;
    trace("NNAA: dump copied at frame %llu (%ux%u, 8 planes)", (unsigned long long)s.frame, s.w, s.hh);
}

inline void dumpFinish(State &s, DeviceData &dd)
{
    const char *t = getenv("TEMP");
    std::string path = std::string(t ? t : ".") + "\\nn_aa_dump.bin";
    FILE *f = fopen(path.c_str(), "wb");
    if (f) {
        uint32_t hdr[4] = { 0x444E564Du /* 'MVND' */, s.w, s.hh, 8u };
        fwrite(hdr, 4, 4, f);
        fwrite(s.dptr, 1, (size_t)s.dbytes, f);
        fclose(f);
        trace("NNAA: dump written to %s (%llu bytes)", path.c_str(), (unsigned long long)s.dbytes);
    } else trace("NNAA: could not write %s", path.c_str());
    dd.unmapMemory(s.dev, s.dmem);
    dd.destroyBuffer(s.dev, s.dbuf, nullptr); dd.freeMemory(s.dev, s.dmem, nullptr);
    s.dbuf = VK_NULL_HANDLE; s.dmem = VK_NULL_HANDLE; s.dptr = nullptr;
    s.dumpState = 2;
}

// ---------------------------------------------------------------- the frame
// Recorded after the resolve's dispatch (which filled the input map and wrote
// its own output into histView) and before the sharpen / copy-out.
inline void record(DeviceData &dd, VkCommandBuffer cb, VkImageView histView, VkImage histImg, uint32_t histLayer)
{
    State &s = state();
    if (!s.ready || !s.weightsOk || !s.in.img) return;
    if (!ensureMaps(s, dd, cb)) { static bool said = false; if (!said) { said = true; trace("NNAA: feature map allocation failed"); } return; }
    ++s.frame;
    if (!s.announced) {
        s.announced = true;
        double mb = 0; Map *all[] = { &s.in, &s.a, &s.b, &s.g, &s.out, &s.c, &s.d, &s.h, &s.e, &s.f };
        for (Map *m : all) mb += (double)m->w * m->h * m->layers * 8.0 / 1048576.0;
        trace("NNAA: running the AA net at %ux%u - channels (%d,%d,%d), %.0f MB of feature maps, split %.2f",
              s.w, s.hh, s.ch0, s.ch1, s.ch2, mb, live::f("taa.nn_split", "TAA_NN_SPLIT", 0.5f));
    }
    computeBarrier(dd, cb);   // the resolve's writes to the input map
    const Layer *L = s.layers.data();
    conv(s, dd, cb, L[0],  s.in, 0, s.inCh, nullptr, 0,     s.a,   1);   // e0.c1
    conv(s, dd, cb, L[1],  s.a,  0, s.ch0,  nullptr, 0,     s.b,   1);   // e0.c2   (skip for d0)
    conv(s, dd, cb, L[2],  s.b,  1, s.ch0,  nullptr, 0,     s.c,   1);   // e1.c1 on pool(e0)
    conv(s, dd, cb, L[3],  s.c,  0, s.ch1,  nullptr, 0,     s.d,   1);   // e1.c2   (skip for d1)
    conv(s, dd, cb, L[4],  s.d,  1, s.ch1,  nullptr, 0,     s.e,   1);   // e2.c1 on pool(e1)
    conv(s, dd, cb, L[5],  s.e,  0, s.ch2,  nullptr, 0,     s.f,   1);   // e2.c2
    conv(s, dd, cb, L[6],  s.f,  2, s.ch2,  &s.d,    s.ch1, s.c,   1);   // d1.c1 on cat(up(e2), e1)
    conv(s, dd, cb, L[7],  s.c,  0, s.ch1,  nullptr, 0,     s.h,   1);   // d1.c2
    conv(s, dd, cb, L[8],  s.h,  2, s.ch1,  &s.b,    s.ch0, s.a,   1);   // d0.c1 on cat(up(d1), e0)
    conv(s, dd, cb, L[9],  s.a,  0, s.ch0,  nullptr, 0,     s.g,   1);   // d0.c2
    conv(s, dd, cb, L[10], s.g,  0, s.ch0,  nullptr, 0,     s.out, 0);   // out (no activation)

    // post: base + tanh(r) * (0.5 + |base|) into the history image
    VkDescriptorSet set = takeSet(s, false);
    VkDescriptorImageInfo ii[3]; memset(ii, 0, sizeof(ii));
    ii[0].imageView = s.out.view; ii[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ii[1].imageView = s.in.view;  ii[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ii[2].imageView = histView;   ii[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet wr[3]; memset(wr, 0, sizeof(wr));
    for (int k = 0; k < 3; ++k) {
        wr[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr[k].dstSet = set; wr[k].dstBinding = (uint32_t)k;
        wr[k].descriptorCount = 1; wr[k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; wr[k].pImageInfo = &ii[k];
    }
    dd.updateDescriptorSets(s.dev, 3, wr, 0, nullptr);
    PostPush p;
    p.w = (int32_t)s.w; p.h = (int32_t)s.hh; p.layer = (int32_t)histLayer;
    p.viz   = live::i("taa.nn_viz", "TAA_NN_VIZ", 0);
    p.split = live::f("taa.nn_split", "TAA_NN_SPLIT", 0.5f);
    p.blend = live::f("taa.nn_blend", "TAA_NN_BLEND", 1.0f);
    dd.cmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipePost);
    dd.cmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.plPost, 0, 1, &set, 0, nullptr);
    dd.cmdPushConstants(cb, s.plPost, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
    dd.cmdDispatch(cb, (s.w + 7) / 8, (s.hh + 7) / 8, 1);
    computeBarrier(dd, cb);
    ++s.dispatches;

    const bool wantDump = live::onoff("taa.nn_dump", "TAA_NN_DUMP", false);
    if (!wantDump && s.dumpState == 2) s.dumpState = 0;            // re-arm when the key is turned off
    if (wantDump && s.dumpState == 0) dumpBegin(s, dd, cb, histImg);
    else if (s.dumpState == 1 && s.frame >= s.dumpFrame + 6) dumpFinish(s, dd);

    if ((s.frame % 600) == 1)
        trace("NNAA: frame %llu, %llu dispatches so far", (unsigned long long)s.frame, (unsigned long long)s.dispatches);
}

inline void shutdown(DeviceData &dd)
{
    State &s = state();
    if (s.dev == VK_NULL_HANDLE) return;
    if (s.dptr) { dd.unmapMemory(s.dev, s.dmem); s.dptr = nullptr; }
    if (s.dbuf) dd.destroyBuffer(s.dev, s.dbuf, nullptr);
    if (s.dmem) dd.freeMemory(s.dev, s.dmem, nullptr);
    Map *all[] = { &s.in, &s.a, &s.b, &s.g, &s.out, &s.c, &s.d, &s.h, &s.e, &s.f };
    for (Map *m : all) freeMap(s, dd, *m);
    if (s.wbuf) dd.destroyBuffer(s.dev, s.wbuf, nullptr);
    if (s.wmem) dd.freeMemory(s.dev, s.wmem, nullptr);
    if (s.pipeConv) dd.destroyPipeline(s.dev, s.pipeConv, nullptr);
    if (s.pipePost) dd.destroyPipeline(s.dev, s.pipePost, nullptr);
    if (s.plConv) dd.destroyPipelineLayout(s.dev, s.plConv, nullptr);
    if (s.plPost) dd.destroyPipelineLayout(s.dev, s.plPost, nullptr);
    if (s.pool) dd.destroyDescriptorPool(s.dev, s.pool, nullptr);
    if (s.dslConv) dd.destroyDescriptorSetLayout(s.dev, s.dslConv, nullptr);
    if (s.dslPost) dd.destroyDescriptorSetLayout(s.dev, s.dslPost, nullptr);
    if (s.smConv) dd.destroyShaderModule(s.dev, s.smConv, nullptr);
    if (s.smPost) dd.destroyShaderModule(s.dev, s.smPost, nullptr);
    s = State();
}

} // namespace nnaa
