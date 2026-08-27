#pragma once
// =================================================================== THE ORACLE
//
// One armed run answers every open question about what the engine gives us.
//
// Three organs, one report:
//
//   REFLECTION  - every shader module the engine creates is scanned for the
//                 uniform / storage blocks on the watch list (u_shadow_data,
//                 ssbo_light_list, ...). The shipped SPIR-V carries OpName,
//                 OpMemberName and Offset decorations, so the exact byte
//                 layout of the engine's own data - cascade matrices, light
//                 records - is READ, not guessed. Set and binding numbers
//                 come with it, which is where a future tap goes.
//
//   NAMES       - the debug-utils listener (in layer.cpp) already records
//                 what the engine calls every image; the oracle adds buffers
//                 and folds both into the report with formats, sizes, usage.
//
//   CONTENT     - a tiny compute probe cycles across every watch-list image
//                 and samples what is actually IN it: min / max / mean /
//                 centre. "The cascade exists" and "the cascade contains
//                 depth-shaped numbers" are different facts, and only the
//                 second one is worth building on. Results ride a 24-slot
//                 ring; each is read one full lap after its dispatch, by
//                 which time the frame that wrote it has long retired.
//
// Armed by taa.oracle=1 (live). While armed, the resolve suppresses AO,
// contact shadows and sharpen so measurements see the pipeline, not our own
// effects layered over it. The report lands in %TEMP%\mv_oracle.txt, rewritten
// every ~1800 frames, and ends with one ANSWER line per question - the point
// of the exercise is that a single launch settles all of them at once.

namespace oracle {

// ------------------------------------------------------------- the watch lists
static const char *kBlockWatch[] = {
    "u_shadow_data", "ssbo_light_list", "light_tile_data", "u_gbuffer_data",
    "ibl_probes_data", "u_environment_data", "u_deferred_shading_data",
    "u_new_sky_data", "deferred_metering_data", "u_immediate_data",
};
static const char *kShaderResWatch[] = {
    "tex_smap", "tex_hiZ", "tex_cubemap", "gbuffer_", "tex_ssr", "tex_brdf",
    "u_tex_cloud_shadow", "u_tex_in_scatter",
};
// Image-name fragments worth probing. Lower-case compared.
static const char *kImgWatch[] = {
    "smap", "shadow", "cascad", "cloud", "cube", "probe", "env", "hiz",
    "hi-z", "hi_z", "weather", "cacao", "ssao", "water depth", "gbuf-",
};

// ----------------------------------------------------------------- reflection
struct Member { std::string name; uint32_t offset = 0; };
struct Block  {
    uint32_t set = 0xffffffffu, binding = 0xffffffffu;
    std::vector<Member> members;
};
static std::map<std::string, Block> g_blocks;                       // by name
static std::map<std::string, std::pair<uint32_t, uint32_t>> g_shaderRes;
static std::mutex   g_omx;
static uint64_t     g_modScanned = 0, g_modMatched = 0;

inline bool bytesContain(const uint8_t *d, size_t n, const char *needle)
{
    const size_t m = strlen(needle);
    if (m == 0 || m > n) return false;
    for (size_t i = 0; i + m <= n; ++i)
        if (d[i] == (uint8_t)needle[0] && memcmp(d + i, needle, m) == 0)
            return true;
    return false;
}

inline void reflect(const uint32_t *w, size_t nWords)
{
    if (!w || nWords < 5 || w[0] != 0x07230203u) return;
    {
        std::lock_guard<std::mutex> g(g_omx);
        ++g_modScanned;
    }
    // Cheap gate first: no watch string in the raw bytes, no parse.
    const uint8_t *bytes = (const uint8_t *)w;
    const size_t   nb    = nWords * 4;
    bool any = false;
    for (size_t i = 0; i < sizeof(kBlockWatch) / sizeof(kBlockWatch[0]); ++i)
        if (bytesContain(bytes, nb, kBlockWatch[i])) { any = true; break; }
    if (!any)
        for (size_t i = 0; i < sizeof(kShaderResWatch) / sizeof(kShaderResWatch[0]); ++i)
            if (bytesContain(bytes, nb, kShaderResWatch[i])) { any = true; break; }
    if (!any) return;

    std::map<uint32_t, std::string> names;                    // id -> OpName
    std::map<uint32_t, std::map<uint32_t, std::string>> memberNames;
    std::map<uint32_t, std::map<uint32_t, uint32_t>>    memberOffsets;
    std::map<uint32_t, uint32_t> decoSet, decoBind;           // id -> value
    std::map<uint32_t, uint32_t> ptrPointee;                  // ptr type -> pointee
    std::map<uint32_t, uint32_t> varType;                     // var id -> ptr type

    size_t i = 5;
    while (i < nWords) {
        const uint32_t op = w[i] & 0xffffu, len = w[i] >> 16;
        if (len == 0 || i + len > nWords) break;
        switch (op) {
        case 5:   // OpName target, literal
            if (len >= 3) {
                const char *s = (const char *)&w[i + 2];
                names[w[i + 1]] = std::string(s, strnlen(s, (len - 2) * 4));
            }
            break;
        case 6:   // OpMemberName type, member, literal
            if (len >= 4) {
                const char *s = (const char *)&w[i + 3];
                memberNames[w[i + 1]][w[i + 2]] =
                    std::string(s, strnlen(s, (len - 3) * 4));
            }
            break;
        case 71:  // OpDecorate target, decoration, [operand]
            if (len >= 4) {
                if (w[i + 2] == 34) decoSet[w[i + 1]]  = w[i + 3];
                if (w[i + 2] == 33) decoBind[w[i + 1]] = w[i + 3];
            }
            break;
        case 72:  // OpMemberDecorate struct, member, decoration, [operand]
            if (len >= 5 && w[i + 3] == 35)   // Offset
                memberOffsets[w[i + 1]][w[i + 2]] = w[i + 4];
            break;
        case 32:  // OpTypePointer result, storage, pointee
            if (len >= 4) ptrPointee[w[i + 1]] = w[i + 3];
            break;
        case 59:  // OpVariable resultType, result, storage
            if (len >= 4) varType[w[i + 2]] = w[i + 1];
            break;
        default: break;
        }
        i += len;
    }

    std::lock_guard<std::mutex> g(g_omx);
    ++g_modMatched;
    for (std::map<uint32_t, uint32_t>::iterator it = varType.begin();
         it != varType.end(); ++it) {
        const uint32_t varId = it->first;
        std::map<uint32_t, uint32_t>::iterator pp = ptrPointee.find(it->second);
        const uint32_t pointee = (pp != ptrPointee.end()) ? pp->second : 0;

        // Blocks: GLSL puts the block NAME on the struct type.
        std::map<uint32_t, std::string>::iterator tn = names.find(pointee);
        if (tn != names.end()) {
            for (size_t k = 0; k < sizeof(kBlockWatch) / sizeof(kBlockWatch[0]); ++k) {
                if (tn->second != kBlockWatch[k]) continue;
                Block &b = g_blocks[tn->second];
                if (decoSet.count(varId))  b.set     = decoSet[varId];
                if (decoBind.count(varId)) b.binding = decoBind[varId];
                std::map<uint32_t, std::string> &mn = memberNames[pointee];
                std::map<uint32_t, uint32_t>    &mo = memberOffsets[pointee];
                if (b.members.size() < mn.size()) {
                    b.members.clear();
                    for (std::map<uint32_t, std::string>::iterator m = mn.begin();
                         m != mn.end(); ++m) {
                        Member mm;
                        mm.name   = m->second;
                        mm.offset = mo.count(m->first) ? mo[m->first] : 0;
                        b.members.push_back(mm);
                    }
                }
            }
        }
        // Named resources (samplers etc.): the name is on the VARIABLE.
        std::map<uint32_t, std::string>::iterator vn = names.find(varId);
        if (vn != names.end()) {
            for (size_t k = 0; k < sizeof(kShaderResWatch) / sizeof(kShaderResWatch[0]); ++k) {
                if (vn->second.compare(0, strlen(kShaderResWatch[k]),
                                       kShaderResWatch[k]) != 0) continue;
                if (decoSet.count(varId) || decoBind.count(varId))
                    g_shaderRes[vn->second] = std::make_pair(
                        decoSet.count(varId)  ? decoSet[varId]  : 0xffffffffu,
                        decoBind.count(varId) ? decoBind[varId] : 0xffffffffu);
            }
        }
    }
}

// -------------------------------------------------------------- the probe pass
inline bool armed()
{
    return live::onoff("taa.oracle", "TAA_ORACLE", false);
}

static const uint32_t kSlots = 24;

struct PState {
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
    PFN_vkCreateBuffer              createBuf  = nullptr;
    PFN_vkGetBufferMemoryRequirements bufReq   = nullptr;
    PFN_vkAllocateMemory            allocMem   = nullptr;
    PFN_vkBindBufferMemory          bindBuf    = nullptr;
    PFN_vkMapMemory                 mapMem     = nullptr;
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
    VkDescriptorSet       sets[kSlots];
    VkSampler             samp = VK_NULL_HANDLE;
    VkBuffer              ssbo = VK_NULL_HANDLE;
    VkDeviceMemory        mem  = VK_NULL_HANDLE;
    void                 *map  = nullptr;

    std::map<VkImage, VkImageView> views;
    struct Pending {
        bool        used = false;
        std::string name;
        VkFormat    fmt = VK_FORMAT_UNDEFINED;
        uint32_t    w = 0, h = 0, layers = 0, layer = 0;
    } pend[kSlots];
    uint32_t slot   = 0;   // next ring slot to use
    uint32_t cursor = 0;   // rotates through the watch images
    uint64_t probes = 0;
};
static PState P;

inline bool fmtIsDepth(VkFormat f)
{
    return f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_X8_D24_UNORM_PACK32 ||
           f == VK_FORMAT_D32_SFLOAT || f == VK_FORMAT_D16_UNORM_S8_UINT ||
           f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

inline bool nameOnImgWatch(const std::string &n)
{
    std::string lo(n);
    for (size_t i = 0; i < lo.size(); ++i)
        lo[i] = (char)tolower((unsigned char)lo[i]);
    // Streamed content (texture paths) matches watch fragments constantly;
    // the probe wants render targets, and those carry short engine names.
    if (lo.find('/') != std::string::npos || lo.find('\\') != std::string::npos)
        return false;
    for (size_t i = 0; i < sizeof(kImgWatch) / sizeof(kImgWatch[0]); ++i)
        if (lo.find(kImgWatch[i]) != std::string::npos) return true;
    return false;
}

inline bool initProbe(DeviceData &dd, VkDevice dev)
{
    if (P.tried) return P.ready;
    P.tried = true;
    P.dev = dev;
    #define ORC_GET(field, name) \
        P.field = (decltype(P.field))dd.gdpa(dev, name); if (!P.field) return false
    ORC_GET(createSm,   "vkCreateShaderModule");
    ORC_GET(createDsl,  "vkCreateDescriptorSetLayout");
    ORC_GET(createPl,   "vkCreatePipelineLayout");
    ORC_GET(createPipe, "vkCreateComputePipelines");
    ORC_GET(createPool, "vkCreateDescriptorPool");
    ORC_GET(allocSets,  "vkAllocateDescriptorSets");
    ORC_GET(updSets,    "vkUpdateDescriptorSets");
    ORC_GET(createSamp, "vkCreateSampler");
    ORC_GET(createView, "vkCreateImageView");
    ORC_GET(createBuf,  "vkCreateBuffer");
    ORC_GET(bufReq,     "vkGetBufferMemoryRequirements");
    ORC_GET(allocMem,   "vkAllocateMemory");
    ORC_GET(bindBuf,    "vkBindBufferMemory");
    ORC_GET(mapMem,     "vkMapMemory");
    ORC_GET(bindPipe,   "vkCmdBindPipeline");
    ORC_GET(bindSets,   "vkCmdBindDescriptorSets");
    ORC_GET(pushConst,  "vkCmdPushConstants");
    ORC_GET(dispatch,   "vkCmdDispatch");
    ORC_GET(barrier,    "vkCmdPipelineBarrier");
    #undef ORC_GET

    VkShaderModuleCreateInfo smci;
    memset(&smci, 0, sizeof(smci));
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sizeof(kOracleProbeSpv);
    smci.pCode = kOracleProbeSpv;
    if (P.createSm(dev, &smci, nullptr, &P.sm) != VK_SUCCESS) return false;

    VkDescriptorSetLayoutBinding b[2];
    memset(b, 0, sizeof(b));
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    for (int k = 0; k < 2; ++k) { b[k].descriptorCount = 1; b[k].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dl;
    memset(&dl, 0, sizeof(dl));
    dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl.bindingCount = 2; dl.pBindings = b;
    if (P.createDsl(dev, &dl, nullptr, &P.dsl) != VK_SUCCESS) return false;

    VkPushConstantRange pr;
    pr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pr.offset = 0; pr.size = 16;
    VkPipelineLayoutCreateInfo pl;
    memset(&pl, 0, sizeof(pl));
    pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1; pl.pSetLayouts = &P.dsl;
    pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &pr;
    if (P.createPl(dev, &pl, nullptr, &P.pl) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo cp;
    memset(&cp, 0, sizeof(cp));
    cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = P.sm;
    cp.stage.pName = "main";
    cp.layout = P.pl;
    if (P.createPipe(dev, VK_NULL_HANDLE, 1, &cp, nullptr, &P.pipe) != VK_SUCCESS)
        return false;

    VkDescriptorPoolSize psz[2];
    psz[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; psz[0].descriptorCount = kSlots;
    psz[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         psz[1].descriptorCount = kSlots;
    VkDescriptorPoolCreateInfo dp;
    memset(&dp, 0, sizeof(dp));
    dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp.maxSets = kSlots; dp.poolSizeCount = 2; dp.pPoolSizes = psz;
    if (P.createPool(dev, &dp, nullptr, &P.pool) != VK_SUCCESS) return false;
    VkDescriptorSetLayout lays[kSlots];
    for (uint32_t k = 0; k < kSlots; ++k) lays[k] = P.dsl;
    VkDescriptorSetAllocateInfo da;
    memset(&da, 0, sizeof(da));
    da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    da.descriptorPool = P.pool; da.descriptorSetCount = kSlots; da.pSetLayouts = lays;
    if (P.allocSets(dev, &da, P.sets) != VK_SUCCESS) return false;

    VkSamplerCreateInfo sc;
    memset(&sc, 0, sizeof(sc));
    sc.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sc.magFilter = VK_FILTER_NEAREST; sc.minFilter = VK_FILTER_NEAREST;
    sc.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sc.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sc.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (P.createSamp(dev, &sc, nullptr, &P.samp) != VK_SUCCESS) return false;

    VkBufferCreateInfo bc;
    memset(&bc, 0, sizeof(bc));
    bc.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bc.size  = 64ull * kSlots;
    bc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bc.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (P.createBuf(dev, &bc, nullptr, &P.ssbo) != VK_SUCCESS) return false;
    VkMemoryRequirements mr;
    P.bufReq(dev, P.ssbo, &mr);
    // Host-visible + coherent, found the same way taaFindMemory does.
    // Same memory-property source the whole layer uses (g_getPhysMemProps is
    // captured at instance creation and lives above this include).
    uint32_t idx = UINT32_MAX;
    if (g_getPhysMemProps && dd.phys != VK_NULL_HANDLE) {
        VkPhysicalDeviceMemoryProperties mp;
        memset(&mp, 0, sizeof(mp));
        g_getPhysMemProps(dd.phys, &mp);
        const VkMemoryPropertyFlags want =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t k = 0; k < mp.memoryTypeCount; ++k)
            if ((mr.memoryTypeBits & (1u << k)) &&
                (mp.memoryTypes[k].propertyFlags & want) == want) { idx = k; break; }
    }
    VkMemoryAllocateInfo ma;
    memset(&ma, 0, sizeof(ma));
    ma.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ma.allocationSize = mr.size;
    ma.memoryTypeIndex = idx;
    if (idx == UINT32_MAX ||
        P.allocMem(dev, &ma, nullptr, &P.mem) != VK_SUCCESS ||
        P.bindBuf(dev, P.ssbo, P.mem, 0) != VK_SUCCESS ||
        P.mapMem(dev, P.mem, 0, VK_WHOLE_SIZE, 0, &P.map) != VK_SUCCESS)
        return false;

    P.ready = true;
    trace("ORACLE: probe pass ready (%u-slot ring).", kSlots);
    return true;
}

// Record one probe into the frame's command buffer. Called where the resolve
// records - scene finished, engine targets in their read layouts.
inline void record(DeviceData &dd, VkDevice dev, VkCommandBuffer cb)
{
    if (!armed()) return;
    if (!initProbe(dd, dev)) return;

    // Report the result this slot carried from a full lap ago.
    PState::Pending &pd = P.pend[P.slot];
    if (pd.used && P.map) {
        const float *v = (const float *)P.map + P.slot * 16;
        trace("ORACLE CONTENT: %-28s fmt=%3d %ux%ux%u L%u | "
              "min(%.4g,%.4g,%.4g,%.4g) max(%.4g,%.4g,%.4g,%.4g) "
              "mean(%.4g,%.4g,%.4g,%.4g) centre(%.4g,%.4g,%.4g,%.4g)",
              pd.name.c_str(), (int)pd.fmt, pd.w, pd.h, pd.layers, pd.layer,
              v[0], v[1], v[2], v[3],  v[4], v[5], v[6], v[7],
              v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
        pd.used = false;
    }

    // Choose the next watch image, rotating. Snapshot under the lock.
    VkImage      img = VK_NULL_HANDLE;
    std::string  name;
    OracleImgInfo info;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::vector<std::pair<VkImage, std::string> > cand;
        for (std::map<VkImage, std::string>::iterator it = g_imageNames.begin();
             it != g_imageNames.end(); ++it)
            if (nameOnImgWatch(it->second) && g_allImages.count(it->first))
                cand.push_back(std::make_pair(it->first, it->second));
        if (cand.empty()) return;
        const uint32_t pick = P.cursor % (uint32_t)cand.size();
        ++P.cursor;
        img  = cand[pick].first;
        name = cand[pick].second;
        info = g_allImages[img];
    }
    if (info.samples != VK_SAMPLE_COUNT_1_BIT) return;   // cannot sample MS
    if (info.type != VK_IMAGE_TYPE_2D) return;           // probe is 2D-array only

    VkImageView view = VK_NULL_HANDLE;
    std::map<VkImage, VkImageView>::iterator vi = P.views.find(img);
    if (vi != P.views.end()) view = vi->second;
    else {
        VkImageViewCreateInfo v;
        memset(&v, 0, sizeof(v));
        v.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        v.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        v.format = info.format;
        v.image = img;
        v.subresourceRange.aspectMask =
            fmtIsDepth(info.format) ? VK_IMAGE_ASPECT_DEPTH_BIT
                                    : VK_IMAGE_ASPECT_COLOR_BIT;
        v.subresourceRange.levelCount = 1;
        v.subresourceRange.layerCount = info.layers ? info.layers : 1;
        if (P.createView(dev, &v, nullptr, &view) != VK_SUCCESS) {
            // Some formats (pure stencil, compressed) refuse; skip forever by
            // caching the null.
            P.views[img] = VK_NULL_HANDLE;
            return;
        }
        P.views[img] = view;
    }
    if (view == VK_NULL_HANDLE) return;

    // Cubemaps get their layers visited on successive laps.
    PState::Pending np;
    np.used = true; np.name = name; np.fmt = info.format;
    np.w = info.w; np.h = info.h; np.layers = info.layers;
    np.layer = (info.layers > 1) ? (P.cursor % info.layers) : 0;

    VkDescriptorImageInfo di;
    di.sampler = P.samp;
    di.imageView = view;
    di.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorBufferInfo db;
    db.buffer = P.ssbo; db.offset = 0; db.range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet wr[2];
    memset(wr, 0, sizeof(wr));
    for (int k = 0; k < 2; ++k) {
        wr[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[k].dstSet = P.sets[P.slot];
        wr[k].dstBinding = (uint32_t)k;
        wr[k].descriptorCount = 1;
    }
    wr[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    wr[0].pImageInfo = &di;
    wr[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr[1].pBufferInfo = &db;
    P.updSets(dev, 2, wr, 0, nullptr);

    struct { int32_t slot, layer, p0, p1; } pc;
    pc.slot = (int32_t)P.slot; pc.layer = (int32_t)np.layer; pc.p0 = pc.p1 = 0;
    P.bindPipe(cb, VK_PIPELINE_BIND_POINT_COMPUTE, P.pipe);
    P.bindSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, P.pl, 0, 1,
               &P.sets[P.slot], 0, nullptr);
    P.pushConst(cb, P.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, &pc);
    P.dispatch(cb, 1, 1, 1);
    VkMemoryBarrier mb;
    memset(&mb, 0, sizeof(mb));
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    P.barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

    P.pend[P.slot] = np;
    P.slot = (P.slot + 1) % kSlots;
    ++P.probes;
}

// ------------------------------------------------------------------ the report
inline void dump()
{
    const char *t = getenv("TEMP");
    std::string path = std::string(t ? t : ".") + "\\mv_oracle.txt";
    FILE *f = fopen(path.c_str(), "w");
    if (!f) return;

    fprintf(f, "==================== MV ORACLE REPORT ====================\n\n");

    fprintf(f, "---- REFLECTED BLOCKS (layouts read from the engine's own "
               "SPIR-V; offsets are bytes)\n");
    {
        std::lock_guard<std::mutex> g(g_omx);
        fprintf(f, "modules scanned: %llu, containing watch names: %llu\n\n",
                (unsigned long long)g_modScanned, (unsigned long long)g_modMatched);
        for (std::map<std::string, Block>::iterator it = g_blocks.begin();
             it != g_blocks.end(); ++it) {
            fprintf(f, "%s  (set=%u binding=%u)\n", it->first.c_str(),
                    it->second.set, it->second.binding);
            for (size_t k = 0; k < it->second.members.size(); ++k)
                fprintf(f, "    +%-6u %s\n", it->second.members[k].offset,
                        it->second.members[k].name.c_str());
        }
        fprintf(f, "\n---- SHADER RESOURCES (watch samplers and where they bind)\n");
        for (std::map<std::string, std::pair<uint32_t, uint32_t>>::iterator it =
                 g_shaderRes.begin(); it != g_shaderRes.end(); ++it)
            fprintf(f, "%-28s set=%u binding=%u\n", it->first.c_str(),
                    it->second.first, it->second.second);
    }

    fprintf(f, "\n---- NAMED IMAGES ON THE WATCH LIST (engine names)\n");
    size_t nImg = 0;
    bool hasSun = false, hasCloudShadow = false, hasCube = false, hasHiz = false,
         hasWeather = false, hasCacao = false;
    {
        std::lock_guard<std::mutex> g(g_lock);
        for (std::map<VkImage, std::string>::iterator it = g_imageNames.begin();
             it != g_imageNames.end(); ++it) {
            if (!nameOnImgWatch(it->second)) continue;
            ++nImg;
            std::string lo(it->second);
            for (size_t k = 0; k < lo.size(); ++k)
                lo[k] = (char)tolower((unsigned char)lo[k]);
            if ((lo.find("smap") != std::string::npos ||
                 lo.find("shadow") != std::string::npos ||
                 lo.find("cascad") != std::string::npos) &&
                lo.find("cloud") == std::string::npos) hasSun = true;
            if (lo.find("cloud") != std::string::npos &&
                lo.find("shadow") != std::string::npos) hasCloudShadow = true;
            if (lo.find("cube") != std::string::npos ||
                lo.find("env")  != std::string::npos ||
                lo.find("probe") != std::string::npos) hasCube = true;
            if (lo.find("hiz") != std::string::npos ||
                lo.find("hi-z") != std::string::npos ||
                lo.find("hi_z") != std::string::npos) hasHiz = true;
            if (lo.find("weather") != std::string::npos) hasWeather = true;
            if (lo.find("cacao") != std::string::npos ||
                lo.find("ssao") != std::string::npos) hasCacao = true;
            if (g_allImages.count(it->first)) {
                OracleImgInfo &ii = g_allImages[it->first];
                fprintf(f, "%-40s %p fmt=%3d %ux%u layers=%u samples=%u usage=0x%x\n",
                        it->second.c_str(), (void *)it->first, (int)ii.format,
                        ii.w, ii.h, ii.layers, (unsigned)ii.samples,
                        (unsigned)ii.usage);
            } else {
                fprintf(f, "%-40s %p (no creation record)\n",
                        it->second.c_str(), (void *)it->first);
            }
        }
        fprintf(f, "(%zu watch images of %zu named total)\n",
                nImg, g_imageNames.size());

        fprintf(f, "\n---- NAMED BUFFERS (engine names; the light list lives "
                   "here if anywhere)\n");
        for (std::map<VkBuffer, std::string>::iterator it = g_bufferNames.begin();
             it != g_bufferNames.end(); ++it) {
            uint64_t sz = g_allBuffers.count(it->first) ? g_allBuffers[it->first] : 0;
            fprintf(f, "%-48s %p  %llu bytes\n", it->second.c_str(),
                    (void *)it->first, (unsigned long long)sz);
        }
        fprintf(f, "(%zu named buffers)\n", g_bufferNames.size());
    }

    bool lightsReflected, shadowReflected;
    {
        std::lock_guard<std::mutex> g(g_omx);
        lightsReflected = g_blocks.count("ssbo_light_list") ||
                          g_blocks.count("light_tile_data");
        shadowReflected = g_blocks.count("u_shadow_data") != 0;
    }

    fprintf(f, "\n==================== ANSWERS ====================\n");
    fprintf(f, "sun shadow cascades found ........ %s\n", hasSun ? "YES" : "no");
    fprintf(f, "u_shadow_data layout read ........ %s\n", shadowReflected ? "YES" : "no");
    fprintf(f, "cloud shadow maps found .......... %s\n", hasCloudShadow ? "YES" : "no");
    fprintf(f, "environment cubemap probes found . %s\n", hasCube ? "YES" : "no");
    fprintf(f, "hi-z pyramid found ............... %s\n", hasHiz ? "YES" : "no");
    fprintf(f, "light list layout read ........... %s\n", lightsReflected ? "YES" : "no");
    fprintf(f, "gbuf-weather found ............... %s\n", hasWeather ? "YES" : "no");
    fprintf(f, "engine AO (CACAO) found .......... %s\n", hasCacao ? "YES" : "no");
    fprintf(f, "gbuf-depth identified & bound .... %s\n",
            g_engineDepthImage != VK_NULL_HANDLE ? "YES" : "no");
    fprintf(f, "content probes dispatched ........ %llu (stats in taa_layer trace, "
            "'ORACLE CONTENT')\n", (unsigned long long)P.probes);
    fclose(f);
    trace("ORACLE: report written to %s", path.c_str());
}

// Called from the present hook. Owns its own cadence.
inline void tick(uint64_t frame)
{
    if (!armed()) return;
    static bool announced = false;
    if (!announced) {
        announced = true;
        trace("ORACLE: armed - AO, contact shadows and sharpen are suppressed "
              "while measuring; report -> %%TEMP%%\\mv_oracle.txt");
    }
    if ((frame % 1800) == 600) dump();
}

} // namespace oracle
