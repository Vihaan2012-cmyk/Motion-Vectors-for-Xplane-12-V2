#pragma once
// ==================================================================== SSGI
//
// The gather pass's host side: a half-resolution dispatch recorded immediately
// BEFORE the resolve, in the resolve's own command buffer, so its result is
// ready for the resolve to composite in the same frame.
//
// Every input it needs was tapped for something else first:
//
//   the engine's depth   (gbuf-depth, identified by the name listener)
//   the engine's probes  (environment_probes, likewise)
//   our velocity         (injected into X-Plane's vertex shaders)
//   the HDR scene        (pre-tonemap, which is what makes bounce light carry
//                         energy instead of reading as a grey wash)
//
// It owns only two things: the half-resolution history pair, and the noise
// that turns four rays a frame into an integral.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// ---- BORROWED FROM taa.h, WHICH IS INCLUDED AFTER THIS HEADER.
//
// The resolve calls into us (it records the gather and composites the result),
// so we cannot include it back. These three are forward-declared rather than
// duplicated: a second memory-type search or a second velocity convention
// would be a second chance to disagree with the resolve, and the velocity
// convention in particular has been wrong in this project before.
static uint32_t taaFindMemory(DeviceData &dd, uint32_t typeBits,
                              VkMemoryPropertyFlags want);
static float taaVelScale();
static float taaVelYSign();

namespace gi {

inline bool enabled() { return live::onoff("taa.gi", "TAA_GI", false); }

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
    // ---- WITHOUT THESE, shutdown() COULD NOT FREE WHAT ensure() BUILT.
    //
    // Only the create entry points were ever resolved, so the pipeline, its
    // layouts, the descriptor pool, both samplers and the shader module had no
    // way to be destroyed at all - see the note over shutdown().
    PFN_vkDestroyPipeline           destroyPipe = nullptr;
    PFN_vkDestroyPipelineLayout     destroyPl   = nullptr;
    PFN_vkDestroyDescriptorSetLayout destroyDsl = nullptr;
    PFN_vkDestroyDescriptorPool     destroyPool = nullptr;
    PFN_vkDestroySampler            destroySamp = nullptr;
    PFN_vkDestroyShaderModule       destroySm   = nullptr;
    PFN_vkCreateImage               createImage = nullptr;
    PFN_vkDestroyImage              destroyImage = nullptr;
    PFN_vkGetImageMemoryRequirements imgReq    = nullptr;
    PFN_vkAllocateMemory            allocMem   = nullptr;
    PFN_vkFreeMemory                freeMem    = nullptr;
    PFN_vkBindImageMemory           bindImage  = nullptr;
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
    VkSampler             samp = VK_NULL_HANDLE;      // LINEAR
    VkSampler             sampNear = VK_NULL_HANDLE;  // velocity + depth

    static const uint32_t kSets = 8;
    VkDescriptorSet sets[kSets] = { VK_NULL_HANDLE };
    uint32_t nextSet = 0;

    // Half-resolution history pair. Ping-pong for the same reason everything
    // else here does: an image read and written by one dispatch has no
    // defined contents.
    VkImage        hist[2]     = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDeviceMemory histMem[2]  = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkImageView    histView[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    uint32_t w = 0, h = 0;
    bool     laidOut = false;
    bool     primed  = false;
    uint32_t frame   = 0;
    uint64_t dispatches = 0;
    bool     announced  = false;
    bool     announced2 = false;
};

inline State &state() { static State s; return s; }

// The result the resolve composites. Null until the first dispatch, which is
// what the resolve's flag gate keys on.
inline VkImageView resultView()
{
    State &s = state();
    if (!s.ready || !s.primed) return VK_NULL_HANDLE;
    return s.histView[1u - (s.frame & 1u)];   // what the last dispatch wrote
}

struct Push {
    int32_t halfW, halfH;
    int32_t fullW, fullH;
    float   edA, edB;
    float   invProjX, invProjY;
    float   radius, maxScreenPx, thickness;
    float   alpha;
    float   velScale, velYSign;
    float   ySign;
    float   intensity;
    int32_t steps, rays, frame, flags;
};
// The resolve has had this guard since a push block silently grew past the
// limit; the newer modules did not. A mismatch between this struct and the
// shader's block is not a compile error - it is garbage in every field after
// the first divergence, which is how a 132-vs-128 byte mismatch once shipped.
static_assert(sizeof(Push) <= 128,
              "gi::Push exceeds the guaranteed 128-byte push constant limit");
static_assert(sizeof(Push) == 80,
              "gi::Push changed size - update gi_gather.comp's block to match");
enum { kGiProbes = 1, kGiReset = 2, kGiEngine = 4 };

inline void freeHistory(State &s)
{
    for (int i = 0; i < 2; ++i) {
        if (s.histView[i] && s.destroyView) s.destroyView(s.dev, s.histView[i], nullptr);
        if (s.hist[i] && s.destroyImage)    s.destroyImage(s.dev, s.hist[i], nullptr);
        if (s.histMem[i] && s.freeMem)      s.freeMem(s.dev, s.histMem[i], nullptr);
        s.histView[i] = VK_NULL_HANDLE;
        s.hist[i] = VK_NULL_HANDLE;
        s.histMem[i] = VK_NULL_HANDLE;
    }
    s.w = s.h = 0;
    s.laidOut = false;
    s.primed = false;
}

inline bool ensureHistory(DeviceData &dd, State &s, uint32_t w, uint32_t h)
{
    if (s.hist[0] != VK_NULL_HANDLE && s.w == w && s.h == h) return true;
    freeHistory(s);
    for (int i = 0; i < 2; ++i) {
        VkImageCreateInfo ic;
        memset(&ic, 0, sizeof(ic));
        ic.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ic.imageType = VK_IMAGE_TYPE_2D;
        ic.format = VK_FORMAT_R16G16B16A16_SFLOAT;   // HDR radiance
        ic.extent.width = w; ic.extent.height = h; ic.extent.depth = 1;
        ic.mipLevels = 1; ic.arrayLayers = 1;
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
        v.format = ic.format;
        v.image = s.hist[i];
        v.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        v.subresourceRange.levelCount = 1;
        v.subresourceRange.layerCount = 1;
        if (s.createView(s.dev, &v, nullptr, &s.histView[i]) != VK_SUCCESS) {
            freeHistory(s); return false;
        }
    }
    s.w = w; s.h = h;
    trace("GI: half-res history pair allocated %ux%u (%.1f MB total)", w, h,
          2.0 * (double)w * (double)h * 8.0 / 1048576.0);
    return true;
}

inline bool init(DeviceData &dd, VkDevice dev)
{
    State &s = state();
    if (s.tried) return s.ready;
    s.tried = true;
    s.dev = dev;

    #define GI_GET(field, name) \
        s.field = (decltype(s.field))dd.gdpa(dev, name); if (!s.field) return false
    GI_GET(createSm,    "vkCreateShaderModule");
    GI_GET(createDsl,   "vkCreateDescriptorSetLayout");
    GI_GET(createPl,    "vkCreatePipelineLayout");
    GI_GET(createPipe,  "vkCreateComputePipelines");
    GI_GET(createPool,  "vkCreateDescriptorPool");
    GI_GET(allocSets,   "vkAllocateDescriptorSets");
    GI_GET(updSets,     "vkUpdateDescriptorSets");
    GI_GET(createSamp,  "vkCreateSampler");
    GI_GET(createView,  "vkCreateImageView");
    GI_GET(destroyView, "vkDestroyImageView");
    GI_GET(destroyPipe, "vkDestroyPipeline");
    GI_GET(destroyPl,   "vkDestroyPipelineLayout");
    GI_GET(destroyDsl,  "vkDestroyDescriptorSetLayout");
    GI_GET(destroyPool, "vkDestroyDescriptorPool");
    GI_GET(destroySamp, "vkDestroySampler");
    GI_GET(destroySm,   "vkDestroyShaderModule");
    GI_GET(createImage, "vkCreateImage");
    GI_GET(destroyImage,"vkDestroyImage");
    GI_GET(imgReq,      "vkGetImageMemoryRequirements");
    GI_GET(allocMem,    "vkAllocateMemory");
    GI_GET(freeMem,     "vkFreeMemory");
    GI_GET(bindImage,   "vkBindImageMemory");
    GI_GET(bindPipe,    "vkCmdBindPipeline");
    GI_GET(bindSets,    "vkCmdBindDescriptorSets");
    GI_GET(pushConst,   "vkCmdPushConstants");
    GI_GET(dispatch,    "vkCmdDispatch");
    GI_GET(barrier,     "vkCmdPipelineBarrier");
    #undef GI_GET

    VkShaderModuleCreateInfo smci;
    memset(&smci, 0, sizeof(smci));
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sizeof(kGiGatherSpv);
    smci.pCode = kGiGatherSpv;
    if (s.createSm(dev, &smci, nullptr, &s.sm) != VK_SUCCESS) return false;

    // 0 scene | 1 velocity | 2 depth | 3 hist read | 4 hist write |
    // 5 probes | 6 engine normals | 7 engine u_gbuffer_data (by reference)
    VkDescriptorSetLayoutBinding b[8];
    memset(b, 0, sizeof(b));
    for (int i = 0; i < 8; ++i) {
        b[i].binding = (uint32_t)i;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[i].descriptorType =
            (i == 4) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE :
            (i == 7) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                     : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }
    VkDescriptorSetLayoutCreateInfo dl;
    memset(&dl, 0, sizeof(dl));
    dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl.bindingCount = 8; dl.pBindings = b;
    if (s.createDsl(dev, &dl, nullptr, &s.dsl) != VK_SUCCESS) return false;

    VkPushConstantRange pr;
    pr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pr.offset = 0; pr.size = sizeof(Push);
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

    VkDescriptorPoolSize psz[3];
    psz[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    psz[0].descriptorCount = 6 * State::kSets;
    psz[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    psz[1].descriptorCount = 1 * State::kSets;
    psz[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    psz[2].descriptorCount = 1 * State::kSets;
    VkDescriptorPoolCreateInfo dp;
    memset(&dp, 0, sizeof(dp));
    dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp.maxSets = State::kSets; dp.poolSizeCount = 3; dp.pPoolSizes = psz;
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

    VkSamplerCreateInfo sc;
    memset(&sc, 0, sizeof(sc));
    sc.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sc.magFilter = VK_FILTER_LINEAR; sc.minFilter = VK_FILTER_LINEAR;
    sc.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sc.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sc.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sc.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sc.maxLod = 0.25f;
    if (s.createSamp(dev, &sc, nullptr, &s.samp) != VK_SUCCESS) return false;
    // Velocity carries a sentinel and depth is piecewise: both must be point
    // sampled, for the reason the resolve documents over its own pair.
    sc.magFilter = VK_FILTER_NEAREST; sc.minFilter = VK_FILTER_NEAREST;
    if (s.createSamp(dev, &sc, nullptr, &s.sampNear) != VK_SUCCESS) return false;

    s.ready = true;
    trace("GI: gather pipeline ready.");
    return true;
}

// Recorded at the TOP of the resolve's recording, so the result is one
// dispatch old at composite time - which is exactly what the resolve's own
// barrier already orders.
inline void record(DeviceData &dd, VkDevice dev, VkCommandBuffer cb,
                   VkImageView sceneView, VkImageView velView,
                   VkImageView depthView, VkImageView probeView,
                   uint32_t fullW, uint32_t fullH,
                   float edA, float edB, float invProjX, float invProjY,
                   float ySign, VkImageView normalView,
                   VkBuffer gbufBuf, VkDeviceSize gbufOff, VkDeviceSize gbufRange)
{
    if (!enabled()) return;
    if (sceneView == VK_NULL_HANDLE || velView == VK_NULL_HANDLE) return;
    if (depthView == VK_NULL_HANDLE) return;   // no depth, no geometry, no GI
    if (!fullW || !fullH) return;
    if (!init(dd, dev)) return;
    State &s = state();

    const uint32_t hw = (fullW + 1) / 2, hh = (fullH + 1) / 2;
    if (!ensureHistory(dd, s, hw, hh)) {
        static bool said = false;
        if (!said) { said = true; trace("GI: history allocation failed"); }
        return;
    }

    if (!s.laidOut) {
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
            hb[i].subresourceRange.layerCount = 1;
            hb[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                  VK_ACCESS_SHADER_WRITE_BIT;
        }
        s.barrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                  0, nullptr, 2, hb);
        s.laidOut = true;
    }

    const uint32_t rd = s.frame & 1u;
    const uint32_t wrIdx = 1u - rd;
    ++s.frame;

    const uint32_t si = s.nextSet;
    s.nextSet = (si + 1) % State::kSets;
    VkDescriptorSet set = s.sets[si];

    // ---- PROBES ARE OFF, AND BLACK IS THE CORRECT ANSWER.
    //
    // Not a limitation - a correction. The engine's own deferred shader
    // ALREADY applies environment irradiance to every surface:
    //
    //   for (i < u_ibl_probes_count_diffuse && accum < 1)
    //       dir = probe_data[i].x == 1 ? boxParallax(eye_to_probe[i], P, N,
    //                                                probe_data[i].yzw)
    //                                  : mat3(modelview_inverse_3d) * N;
    //       s = textureLod(cube, vec4(dir, probe_layer[i].x), 6.0)
    //           * exposure_scale;
    //       irradiance += s.rgb * (1-accum);  accum += s.a * (1-accum);
    //
    // Sampled along the NORMAL at mip 6 - an irradiance lookup - and alpha-
    // blended across up to eight probes. That light is therefore already in
    // the scene colour this pass reads and the resolve composites onto.
    //
    // So adding probe radiance for escaped rays would apply the ambient a
    // SECOND time. What this pass exists to contribute is the delta the
    // engine cannot produce: local coloured bounce between on-screen
    // surfaces, which an unoccluded irradiance lookup has no way to express.
    // A ray that escapes has left the region we can say anything new about,
    // and zero is the honest value for it.
    //
    // The knob remains for experiment. If it is ever turned on properly it
    // needs all four of the things above - cube ARRAY layer, the per-probe
    // eye_to_probe matrix, box parallax, and exposure_scale - plus a way to
    // avoid the double count.
    // ---- THE ENGINE'S OWN GEOMETRY, WHEN BOTH HALVES ARE PRESENT.
    //
    // Its normals AND its screen-to-eye coefficients, or neither: the
    // reconstruction and the normal have to describe the same space, and
    // mixing an engine normal with our hand-rolled position would put the
    // hemisphere and the ray in different frames.
    const bool haveEngine = (normalView != VK_NULL_HANDLE) &&
                            (gbufBuf != VK_NULL_HANDLE) && gbufRange >= 96;

    const bool haveProbes = (probeView != VK_NULL_HANDLE) &&
                            live::onoff("taa.gi_probes", "TAA_GI_PROBES", false);

    VkDescriptorImageInfo ii[8];
    memset(ii, 0, sizeof(ii));
    ii[0].sampler = s.samp;     ii[0].imageView = sceneView;
    ii[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ii[1].sampler = s.sampNear; ii[1].imageView = velView;
    ii[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ii[2].sampler = s.sampNear; ii[2].imageView = depthView;
    ii[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ii[3].sampler = s.samp;     ii[3].imageView = s.histView[rd];
    ii[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ii[4].imageView = s.histView[wrIdx];
    ii[4].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    // Probes are optional: without them escaped rays return black, which is
    // ordinary screen-space GI. The dummy keeps the binding legal.
    ii[5].sampler = s.samp;
    ii[5].imageView = haveProbes ? probeView : s.histView[rd];
    ii[5].imageLayout = haveProbes ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                   : VK_IMAGE_LAYOUT_GENERAL;

    // Binding 6: the engine's normals, or the depth image as a legal dummy
    // (the shader only reads it under GI_ENGINE).
    ii[6].sampler = s.sampNear;
    ii[6].imageView = haveEngine ? normalView : depthView;
    ii[6].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Binding 7: u_gbuffer_data BY REFERENCE - the very region the engine's
    // deferred shader reads, captured by the template scanner and frame-
    // matched at bind time. Our own zeroed ring is not an option here because
    // we have none; when the tap is absent the shader takes the fallback path
    // and never touches this binding, so a dummy of the right TYPE is all it
    // needs. There is no spare uniform buffer, so the engine's own region is
    // bound either way and the flag decides whether it is read.
    VkDescriptorBufferInfo gbi;
    gbi.buffer = gbufBuf;
    gbi.offset = gbufOff;
    gbi.range  = gbufRange;

    VkWriteDescriptorSet wr[8];
    memset(wr, 0, sizeof(wr));
    const int nWrites = haveEngine ? 8 : 7;
    for (int k = 0; k < nWrites; ++k) {
        wr[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[k].dstSet = set;
        wr[k].dstBinding = (uint32_t)k;
        wr[k].descriptorCount = 1;
        wr[k].descriptorType =
            (k == 4) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE :
            (k == 7) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                     : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        if (k == 7) wr[k].pBufferInfo = &gbi;
        else        wr[k].pImageInfo  = &ii[k];
    }
    s.updSets(dev, (uint32_t)nWrites, wr, 0, nullptr);

    Push p;
    p.halfW = (int32_t)hw;   p.halfH = (int32_t)hh;
    p.fullW = (int32_t)fullW; p.fullH = (int32_t)fullH;
    p.edA = edA; p.edB = edB;
    p.invProjX = invProjX; p.invProjY = invProjY;
    p.radius      = live::f("taa.gi_radius",  "TAA_GI_RADIUS",  6.0f);
    // 48 half-res pixels over 10 steps is ~4.8 px per step - the range a
    // screen-space march can actually resolve. The metre radius alone put
    // 1,739 px between steps in the cockpit.
    p.maxScreenPx = live::f("taa.gi_max_px",  "TAA_GI_MAX_PX", 48.0f);
    p.thickness = live::f("taa.gi_thickness", "TAA_GI_THICKNESS", 0.5f);
    p.alpha     = live::f("taa.gi_alpha",     "TAA_GI_ALPHA",     0.08f);
    p.velScale  = taaVelScale();
    p.velYSign  = taaVelYSign();
    // ---- THE VIEWPORT Y FLIP, WHICH THIS PASS WAS IGNORING.
    //
    // X-Plane sets a NEGATIVE-height viewport, so framebuffer Y and clip Y
    // point opposite ways. Every other consumer in this layer respects that -
    // it is the whole reason smul_y is negative and velYSign exists - but the
    // eye reconstruction here assumed the Vulkan default. Positions came out
    // mirrored in Y, and the depth-derived normal is a CROSS PRODUCT of two
    // such differences, so it did not merely flip: it came out as
    // (-n.x, n.y, -n.z), pointing somewhere the surface does not face. Every
    // ray would then be cast into the wrong hemisphere.
    p.ySign     = ySign;
    p.intensity = 1.0f;   // applied at composite; see the shader's note
    p.steps     = (int32_t)live::i("taa.gi_steps", "TAA_GI_STEPS", 10);
    p.rays      = (int32_t)live::i("taa.gi_rays",  "TAA_GI_RAYS",  4);
    p.frame     = (int32_t)(s.frame & 0x3fffffff);
    p.flags     = (haveProbes ? kGiProbes : 0) | (s.primed ? 0 : kGiReset)
                | (haveEngine ? kGiEngine : 0);

    s.bindPipe(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipe);
    s.bindSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pl, 0, 1, &set, 0, nullptr);
    s.pushConst(cb, s.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
    s.dispatch(cb, (hw + 7) / 8, (hh + 7) / 8, 1);

    // The resolve samples what we just wrote, in this same command buffer.
    VkMemoryBarrier mb;
    memset(&mb, 0, sizeof(mb));
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    s.barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr,
              0, nullptr);

    s.primed = true;
    if (!s.announced) {
        s.announced = true;
        trace("GI: gathering at %ux%u (half of %ux%u), %d rays x %d steps, "
              "ray <= %.0f m and <= %.0f px, probes %s. Escaped rays %s. "
              "Stored value is radiance RELATIVE to local surface luminance, "
              "so gi_strength is O(1) rather than O(1e-4).",
              hw, hh, fullW, fullH, p.rays, p.steps, p.radius, p.maxScreenPx,
              haveProbes ? "BOUND" : "absent",
              haveProbes ? "read the engine's environment capture"
                         : "return black - this is plain screen-space GI");
    if (!s.announced2) {
        s.announced2 = true;
        trace("GI: geometry from %s.", haveEngine
              ? "the ENGINE - gbuf-normal (spheremap) and u_gbuffer_data's own "
                "rational reconstruction, so asymmetric frusta, the viewport "
                "flip and jitter are all carried by its coefficients"
              : "our fallback - depth-derivative normals and a symmetric-"
                "frustum reconstruction; gbuf-normal or u_gbuffer_data absent");
    }
    }
    if ((s.dispatches++ % 600) == 0)
        trace("GI: %llu dispatches", (unsigned long long)s.dispatches);
}

inline void shutdown()
{
    State &s = state();
    if (s.dev == VK_NULL_HANDLE) return;
    freeHistory(s);

    // ---- EVERYTHING ensure() BUILT, AND THE LATCH THAT SAID IT WAS BUILT.
    //
    // This used to free the history images and stop. Two consequences, and the
    // second is fatal rather than untidy:
    //
    // 1. The pipeline, pipeline layout, descriptor set layout, descriptor pool,
    //    both samplers and the shader module leaked on every device teardown.
    //    X-Plane recreates its device on some settings changes, so that is not
    //    only a shutdown concern.
    //
    // 2. ensure() begins "if (s.tried) return s.ready;" - a one-shot latch -
    //    and neither tried, ready nor dev was reset here. So after the device
    //    went away this module still reported READY, still held s.dev pointing
    //    at the destroyed device, and still held a pipeline, layouts and a
    //    descriptor pool belonging to it. The next ensure() returned true
    //    without rebuilding anything and record() bound objects from a dead
    //    device into a live command buffer.
    //
    // Destroyed in dependency order: pipeline, then the layouts it was built
    // from, then the pool (which frees its sets), then the set layout, the
    // samplers and the module.
    if (s.pipe && s.destroyPipe) s.destroyPipe(s.dev, s.pipe, nullptr);
    if (s.pl   && s.destroyPl)   s.destroyPl(s.dev, s.pl, nullptr);
    if (s.pool && s.destroyPool) s.destroyPool(s.dev, s.pool, nullptr);
    if (s.dsl  && s.destroyDsl)  s.destroyDsl(s.dev, s.dsl, nullptr);
    if (s.samp     && s.destroySamp) s.destroySamp(s.dev, s.samp, nullptr);
    if (s.sampNear && s.destroySamp) s.destroySamp(s.dev, s.sampNear, nullptr);
    if (s.sm   && s.destroySm)   s.destroySm(s.dev, s.sm, nullptr);
    s.pipe = VK_NULL_HANDLE; s.pl = VK_NULL_HANDLE; s.pool = VK_NULL_HANDLE;
    s.dsl  = VK_NULL_HANDLE; s.samp = VK_NULL_HANDLE;
    s.sampNear = VK_NULL_HANDLE; s.sm = VK_NULL_HANDLE;
    for (int i = 0; i < State::kSets; ++i) s.sets[i] = VK_NULL_HANDLE;

    // The latch last, so a rebuild on a new device starts from nothing.
    s.dev = VK_NULL_HANDLE;
    s.ready = false;
    s.tried = false;
}

} // namespace gi
