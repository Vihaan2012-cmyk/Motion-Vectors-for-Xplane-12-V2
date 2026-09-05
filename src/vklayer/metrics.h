#pragma once
// =================================================================== METRICS
//
// Image-quality measurement. The point is to answer "which setting combination
// is least stable?" with numbers instead of opinions, so a sweep harness can
// rank configurations without a human staring at the screen.
//
// WHAT IT MEASURES. metrics.comp accumulates ~20 statistics per frame into a
// small SSBO - temporal residual under BOTH velocity sign conventions, flicker
// RATE (sign reversals, which is what shimmer actually is), contrast-weighted
// instability (what you SEE), Laplacian energy (aliasing), gradient energy
// (over-blur, so a config cannot win by turning the image to mush), quad-vs-
// neighbourhood variance (staircase), ghosting behind moving geometry, pure
// instability on pixels that did not move, disocclusion cost, motion-vector
// coverage, and three histograms.
//
// HOW THE NUMBERS GET OUT. Nothing per-pixel is ever read back. The shader
// reduces per workgroup in shared memory, then pushes one atomic per slot. The
// host reads 256 bytes per frame off a host-coherent mapping and accumulates
// in double, so the long-window average is exact while every GPU-side counter
// stays inside uint32 (see the PRECISION note in metrics.comp).
//
// THE RING. Section s = frame % 8. At frame f the host reads section s - which
// holds frame f-8, retired long ago - then zeroes it and dispatches frame f
// into it. Eight frames of slack is ~130ms at 60fps; the GPU is never that far
// behind. The descriptor set for a section is rewritten each frame, which is
// safe for the same reason and means a scene-image rotation or a resize is
// picked up automatically instead of silently measuring a dead view.
//
// THE DEPTH CUT. A cockpit-view frame is mostly cockpit shell - near, static,
// and enormous - and leaving it in swamps every scenery number with a
// constant. So it gets masked out. But X-Plane's depth encoding is not
// documented anywhere we trust, so the cut is NOT guessed: metrics runs
// unmasked by default and reports a 16-bin histogram of raw depth over every
// pixel. You read the shape off one run, set taa.metrics_cut, and only then do
// the masked numbers mean anything. Measure first, threshold second.
//
// Armed by taa.metrics=1 (live). Off by default and costs nothing when off.
//
// Copyright (C) 2026 MotionVectors contributors. SPDX: GPL-3.0-or-later

#include "metrics_spv.h"

namespace metrics {

// ---- accumulator slots. MUST mirror metrics.comp exactly. -------------------
enum {
    S_COUNT_MASKED = 0, S_TEMPORAL_PLUS, S_TEMPORAL_MINUS, S_TEMPORAL_SQ,
    S_CONTRAST_WEIGHT, S_CONTRAST, S_FLICKER, S_LAPLACIAN, S_QUADVAR,
    S_NBRVAR, S_GHOST, S_COUNT_GHOST, S_DISOCCL, S_COUNT_DISOCCL, S_VELMAG,
    S_ZEROVEL_MOVED, S_VALID_REPROJ, S_SHARPNESS, S_COUNT_STATIC,
    S_STATIC_DELTA,
    S_HIST_T = 20,   // 16 bins
    S_HIST_V = 36,   //  8 bins
    S_HIST_D = 44,   // 16 bins
    S_COUNT_ALL = 60,
    S_COUNT_NOVEC = 61,
    S_SLOTS = 64
};

static const float kScale    = 1024.0f;   // fixed-point divisor, mirrors the shader
static const uint32_t kSect  = 8;         // ring sections
// Slots whose per-pixel value was normalised into [0,1] by the shader so the
// uint32 accumulator could not overflow. The host puts the range back.
inline float slotScale(uint32_t s)
{
    if (s == S_VELMAG)    return 64.0f;
    if (s == S_LAPLACIAN) return 4.0f;
    if (s == S_SHARPNESS) return 2.0f;
    return 1.0f;
}

inline bool  armed()     { return live::onoff("taa.metrics", "TAA_METRICS", false); }
inline int   reportEvery(){ return live::i("taa.metrics_report", "TAA_METRICS_REPORT", 600); }
inline float depthCut()  { return live::f("taa.metrics_cut",  "TAA_METRICS_CUT",  0.0f); }
inline float depthFar()  { return live::f("taa.metrics_far",  "TAA_METRICS_FAR",  0.0f); }
inline int   maskMode()  { return live::i("taa.metrics_mask", "TAA_METRICS_MASK", 0); }
// ~2 LSB of 8-bit. Below 1 LSB the flicker rate measures dither, not shimmer.
inline float flickThr()  { return live::f("taa.metrics_flick", "TAA_METRICS_FLICK", 0.008f); }

struct MState {
    bool tried = false, ready = false, havePrev = false, signCleared = false;
    VkDevice dev = VK_NULL_HANDLE;

    VkShaderModule        sm   = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl  = VK_NULL_HANDLE;
    VkPipelineLayout      pl   = VK_NULL_HANDLE;
    VkPipeline            pipe = VK_NULL_HANDLE;
    VkDescriptorPool      pool = VK_NULL_HANDLE;
    VkDescriptorSet       sets[kSect];
    VkSampler             samp = VK_NULL_HANDLE;

    VkBuffer       ssbo = VK_NULL_HANDLE;
    VkDeviceMemory ssboMem = VK_NULL_HANDLE;
    void          *map = nullptr;

    // Our copy of last frame's resolved colour, and the per-pixel sign of the
    // last frame-to-frame delta (r32ui) that turns "it changed" into "it
    // reversed", which is the difference between motion and shimmer.
    VkImage        prevImg = VK_NULL_HANDLE, signImg = VK_NULL_HANDLE;
    VkDeviceMemory prevMem = VK_NULL_HANDLE, signMem = VK_NULL_HANDLE;
    VkImageView    prevView = VK_NULL_HANDLE, signView = VK_NULL_HANDLE;
    VkFormat       prevFmt = VK_FORMAT_UNDEFINED;
    uint32_t       w = 0, h = 0;

    uint64_t frame = 0;          // dispatches recorded
    uint64_t counted = 0;        // frames folded into the accumulator
    uint64_t saturated = 0;      // frames rejected as overflowed
    double   acc[S_SLOTS];       // long-window sums, already un-scaled
    uint64_t reports = 0;
};
static MState M;

inline void resetAcc()
{
    for (uint32_t k = 0; k < S_SLOTS; ++k) M.acc[k] = 0.0;
    M.counted = 0; M.saturated = 0;
}

// --------------------------------------------------------------------- setup
inline uint32_t findMem(DeviceData &dd, uint32_t bits, VkMemoryPropertyFlags want)
{
    if (!g_getPhysMemProps || dd.phys == VK_NULL_HANDLE) return UINT32_MAX;
    VkPhysicalDeviceMemoryProperties mp;
    memset(&mp, 0, sizeof(mp));
    g_getPhysMemProps(dd.phys, &mp);
    for (uint32_t k = 0; k < mp.memoryTypeCount; ++k)
        if ((bits & (1u << k)) && (mp.memoryTypes[k].propertyFlags & want) == want)
            return k;
    return UINT32_MAX;
}

inline bool makeImage(DeviceData &dd, VkDevice dev, uint32_t w, uint32_t h,
                      VkFormat fmt, VkImageUsageFlags usage,
                      VkImage *img, VkDeviceMemory *mem, VkImageView *view)
{
    VkImageCreateInfo ic;
    memset(&ic, 0, sizeof(ic));
    ic.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ic.imageType = VK_IMAGE_TYPE_2D;
    ic.format = fmt;
    ic.extent.width = w; ic.extent.height = h; ic.extent.depth = 1;
    ic.mipLevels = 1; ic.arrayLayers = 1;
    ic.samples = VK_SAMPLE_COUNT_1_BIT;
    ic.tiling = VK_IMAGE_TILING_OPTIMAL;
    ic.usage = usage;
    ic.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ic.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (dd.createImage(dev, &ic, nullptr, img) != VK_SUCCESS) return false;

    VkMemoryRequirements mr;
    dd.getImageMemReq(dev, *img, &mr);
    const uint32_t idx = findMem(dd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (idx == UINT32_MAX) return false;
    VkMemoryAllocateInfo ma;
    memset(&ma, 0, sizeof(ma));
    ma.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ma.allocationSize = mr.size; ma.memoryTypeIndex = idx;
    if (dd.allocateMemory(dev, &ma, nullptr, mem) != VK_SUCCESS) return false;
    if (dd.bindImageMemory(dev, *img, *mem, 0) != VK_SUCCESS) return false;

    VkImageViewCreateInfo vi;
    memset(&vi, 0, sizeof(vi));
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = *img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1; vi.subresourceRange.layerCount = 1;
    return dd.createImageView(dev, &vi, nullptr, view) == VK_SUCCESS;
}

inline void destroyTargets(DeviceData &dd, VkDevice dev)
{
    if (M.prevView) { dd.destroyImageView(dev, M.prevView, nullptr); M.prevView = VK_NULL_HANDLE; }
    if (M.signView) { dd.destroyImageView(dev, M.signView, nullptr); M.signView = VK_NULL_HANDLE; }
    if (M.prevImg)  { dd.destroyImage(dev, M.prevImg, nullptr); M.prevImg = VK_NULL_HANDLE; }
    if (M.signImg)  { dd.destroyImage(dev, M.signImg, nullptr); M.signImg = VK_NULL_HANDLE; }
    if (M.prevMem)  { dd.freeMemory(dev, M.prevMem, nullptr); M.prevMem = VK_NULL_HANDLE; }
    if (M.signMem)  { dd.freeMemory(dev, M.signMem, nullptr); M.signMem = VK_NULL_HANDLE; }
    M.havePrev = false; M.signCleared = false;
}

// Per-resolution resources. Called again whenever the resolve changes size or
// colour format - a stale prev image would otherwise make every temporal
// number a comparison against a differently-shaped frame, which is exactly the
// failure mode that already bit the deliver-copy path.
inline bool ensureTargets(DeviceData &dd, VkDevice dev, uint32_t w, uint32_t h, VkFormat fmt)
{
    if (M.prevImg && M.w == w && M.h == h && M.prevFmt == fmt) return true;
    if (M.prevImg) {
        dd.deviceWaitIdle(dev);
        destroyTargets(dd, dev);
    }
    M.w = w; M.h = h; M.prevFmt = fmt;
    if (!makeImage(dd, dev, w, h, fmt,
                   VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                   &M.prevImg, &M.prevMem, &M.prevView)) return false;
    if (!makeImage(dd, dev, w, h, VK_FORMAT_R32_UINT,
                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                   &M.signImg, &M.signMem, &M.signView)) return false;
    trace("METRICS: targets %ux%u fmt=%d", w, h, (int)fmt);
    return true;
}

inline bool init(DeviceData &dd, VkDevice dev)
{
    if (M.tried) return M.ready;
    M.tried = true;
    M.dev = dev;
    resetAcc();

    VkShaderModuleCreateInfo smci;
    memset(&smci, 0, sizeof(smci));
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sizeof(kMetricsSpv);
    smci.pCode = kMetricsSpv;
    if (dd.createShaderModule(dev, &smci, nullptr, &M.sm) != VK_SUCCESS) return false;

    VkDescriptorSetLayoutBinding b[6];
    memset(b, 0, sizeof(b));
    for (int k = 0; k < 6; ++k) {
        b[k].binding = (uint32_t)k; b[k].descriptorCount = 1;
        b[k].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[k].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }
    b[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    VkDescriptorSetLayoutCreateInfo dl;
    memset(&dl, 0, sizeof(dl));
    dl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dl.bindingCount = 6; dl.pBindings = b;
    if (dd.createDescriptorSetLayout(dev, &dl, nullptr, &M.dsl) != VK_SUCCESS) return false;

    VkPushConstantRange pr;
    pr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pr.offset = 0; pr.size = 28;
    VkPipelineLayoutCreateInfo plc;
    memset(&plc, 0, sizeof(plc));
    plc.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plc.setLayoutCount = 1; plc.pSetLayouts = &M.dsl;
    plc.pushConstantRangeCount = 1; plc.pPushConstantRanges = &pr;
    if (dd.createPipelineLayout(dev, &plc, nullptr, &M.pl) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo cp;
    memset(&cp, 0, sizeof(cp));
    cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = M.sm; cp.stage.pName = "main";
    cp.layout = M.pl;
    if (dd.createComputePipelines(dev, VK_NULL_HANDLE, 1, &cp, nullptr, &M.pipe) != VK_SUCCESS)
        return false;

    VkDescriptorPoolSize psz[3];
    psz[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; psz[0].descriptorCount = 4 * kSect;
    psz[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         psz[1].descriptorCount = kSect;
    psz[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          psz[2].descriptorCount = kSect;
    VkDescriptorPoolCreateInfo dp;
    memset(&dp, 0, sizeof(dp));
    dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp.maxSets = kSect; dp.poolSizeCount = 3; dp.pPoolSizes = psz;
    if (dd.createDescriptorPool(dev, &dp, nullptr, &M.pool) != VK_SUCCESS) return false;
    VkDescriptorSetLayout lays[kSect];
    for (uint32_t k = 0; k < kSect; ++k) lays[k] = M.dsl;
    VkDescriptorSetAllocateInfo da;
    memset(&da, 0, sizeof(da));
    da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    da.descriptorPool = M.pool; da.descriptorSetCount = kSect; da.pSetLayouts = lays;
    if (dd.allocateDescriptorSets(dev, &da, M.sets) != VK_SUCCESS) return false;

    VkSamplerCreateInfo sc;
    memset(&sc, 0, sizeof(sc));
    sc.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    // NEAREST throughout: every statistic here is defined on the pixel grid,
    // and a filtered fetch would blur the very high-frequency detail the
    // aliasing and flicker numbers exist to detect.
    sc.magFilter = VK_FILTER_NEAREST; sc.minFilter = VK_FILTER_NEAREST;
    sc.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sc.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sc.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (dd.createSampler(dev, &sc, nullptr, &M.samp) != VK_SUCCESS) return false;

    VkBufferCreateInfo bc;
    memset(&bc, 0, sizeof(bc));
    bc.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bc.size = (VkDeviceSize)(S_SLOTS * 4u * kSect);
    bc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bc.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (dd.createBuffer(dev, &bc, nullptr, &M.ssbo) != VK_SUCCESS) return false;
    VkMemoryRequirements mr;
    dd.getBufferMemReq(dev, M.ssbo, &mr);
    const uint32_t idx = findMem(dd, mr.memoryTypeBits,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (idx == UINT32_MAX) return false;
    VkMemoryAllocateInfo ma;
    memset(&ma, 0, sizeof(ma));
    ma.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ma.allocationSize = mr.size; ma.memoryTypeIndex = idx;
    if (dd.allocateMemory(dev, &ma, nullptr, &M.ssboMem) != VK_SUCCESS) return false;
    if (dd.bindBufferMemory(dev, M.ssbo, M.ssboMem, 0) != VK_SUCCESS) return false;
    if (dd.mapMemory(dev, M.ssboMem, 0, VK_WHOLE_SIZE, 0, &M.map) != VK_SUCCESS) return false;
    memset(M.map, 0, (size_t)(S_SLOTS * 4u * kSect));

    M.ready = true;
    trace("METRICS: ready (%u-section ring, %u slots).", kSect, (unsigned)S_SLOTS);
    return true;
}

// ---------------------------------------------------------------- the report
inline double mean(uint32_t sum, uint32_t cnt)
{
    return cnt ? (double)sum / (double)cnt : 0.0;
}

inline void dump()
{
    if (!M.counted) return;

    const double frames = (double)M.counted;
    // Everything below is a per-frame total; dividing by the matching count
    // turns it into a per-pixel mean. Counts are raw, sums are already
    // un-scaled by fold().
    const double nMask   = M.acc[S_COUNT_MASKED]  / frames;
    const double nAll    = M.acc[S_COUNT_ALL]     / frames;
    const double nValid  = M.acc[S_VALID_REPROJ];
    const double nGhost  = M.acc[S_COUNT_GHOST];
    const double nStatic = M.acc[S_COUNT_STATIC];
    const double nDis    = M.acc[S_COUNT_DISOCCL];
    const double mAll    = M.acc[S_COUNT_MASKED];

    #define MDIV(a, b) ((b) > 0.0 ? (a) / (b) : 0.0)
    const double tPlus  = MDIV(M.acc[S_TEMPORAL_PLUS],  nValid);
    const double tMinus = MDIV(M.acc[S_TEMPORAL_MINUS], nValid);
    const bool   plusWins = (tPlus <= tMinus);
    const double temporal = plusWins ? tPlus : tMinus;
    const double tsq      = MDIV(M.acc[S_TEMPORAL_SQ], nValid);
    const double trms     = tsq > 0.0 ? sqrt(tsq) : 0.0;
    const double edge     = MDIV(M.acc[S_CONTRAST_WEIGHT], mAll);
    const double contrast = MDIV(M.acc[S_CONTRAST],   mAll);
    const double flicker  = MDIV(M.acc[S_FLICKER],    mAll);
    const double lap      = MDIV(M.acc[S_LAPLACIAN],  mAll);
    const double sharp    = MDIV(M.acc[S_SHARPNESS],  mAll);
    const double quadv    = MDIV(M.acc[S_QUADVAR],    mAll);
    const double nbrv     = MDIV(M.acc[S_NBRVAR],     mAll);
    const double stair    = MDIV(quadv, nbrv);
    const double ghost    = MDIV(M.acc[S_GHOST],      nGhost);
    const double stat     = MDIV(M.acc[S_STATIC_DELTA], nStatic);
    const double disoccl  = MDIV(M.acc[S_DISOCCL],    nDis);
    const double vel      = MDIV(M.acc[S_VELMAG],     mAll);
    const double zvm      = MDIV(M.acc[S_ZEROVEL_MOVED], mAll);
    const double novec    = MDIV(M.acc[S_COUNT_NOVEC], mAll);

    // ONE ranking number, so a sweep can sort. The weights are a judgement and
    // are printed with the components so nobody has to take them on faith:
    // static delta is the purest instability (a pixel that did not move must
    // not change), edge is what the eye actually catches, flicker separates
    // buzzing from smooth drift. Rank on a component instead if you disagree.
    const double instab = 2000.0 * stat + 1000.0 * edge + 100.0 * flicker;
    #undef MDIV

    char path[MAX_PATH];
    const DWORD n = GetEnvironmentVariableA("TEMP", path, (DWORD)sizeof(path));
    if (!n || n >= sizeof(path)) return;
    std::string fp = std::string(path) + "\\mv_metrics.txt";
    FILE *f = fopen(fp.c_str(), "a");
    if (!f) return;

    ++M.reports;
    fprintf(f, "\n===== MOTIONVECTORS METRICS report %llu =====\n",
            (unsigned long long)M.reports);
    fprintf(f, "frames %llu  saturated %llu  %ux%u  masked %.0f of %.0f px "
               "(%.1f%%)  mask_mode %d cut %.4f far %.4f\n",
            (unsigned long long)M.counted, (unsigned long long)M.saturated,
            M.w, M.h, nMask, nAll, nAll > 0.0 ? 100.0 * nMask / nAll : 0.0,
            maskMode(), depthCut(), depthFar());
    fprintf(f, "flicker threshold %.4f luma\n", flickThr());

    // The machine-readable line. One per report, stable field order: this is
    // what sweep.ps1 parses to rank configurations.
    fprintf(f, "MET instab=%.4f stat=%.6f edge=%.6f flicker=%.6f temporal=%.6f "
               "trms=%.6f ghost=%.6f disoccl=%.6f lap=%.6f sharp=%.6f "
               "stair=%.4f contrast=%.6f vel=%.4f zvm=%.6f novec=%.6f sign=%s frames=%llu\n",
            instab, stat, edge, flicker, temporal, trms, ghost, disoccl,
            lap, sharp, stair, contrast, vel, zvm, novec,
            plusWins ? "plus" : "minus", (unsigned long long)M.counted);

    fprintf(f, "  INSTABILITY   %.4f   = 2000*static + 1000*edge + 100*flicker\n", instab);
    fprintf(f, "  static        %.6f   mean |delta| on pixels that did not move "
               "(pure instability; 0 is perfect)\n", stat);
    fprintf(f, "  edge          %.6f   |delta| weighted by local contrast "
               "(what the eye catches)\n", edge);
    fprintf(f, "  flicker       %.6f   fraction of pixels reversing delta sign "
               "(shimmer rate, not magnitude)\n", flicker);
    fprintf(f, "  temporal      %.6f   reprojected residual, '%s' convention "
               "(plus %.6f / minus %.6f)\n",
            temporal, plusWins ? "+vel" : "-vel", tPlus, tMinus);
    fprintf(f, "  trms          %.6f   RMS of the residual (tail, not just mean)\n", trms);
    fprintf(f, "  ghost         %.6f   residual behind geometry moving >2px\n", ghost);
    fprintf(f, "  disoccl       %.6f   cost at newly revealed pixels\n", disoccl);
    fprintf(f, "  lap           %.6f   Laplacian energy (aliasing proxy; HIGHER "
               "= more high-frequency detail AND more jaggies)\n", lap);
    fprintf(f, "  sharp         %.6f   gradient energy (over-blur detector; a "
               "config cannot win by going soft)\n", sharp);
    fprintf(f, "  stair         %.4f   intra-quad / neighbourhood variance "
               "(staircase; lower is smoother edges)\n", stair);
    fprintf(f, "  contrast      %.6f   mean local contrast (scene reference)\n", contrast);
    fprintf(f, "  vel           %.4f px mean motion-vector magnitude\n", vel);
    fprintf(f, "  zvm           %.6f   pixels with geometry depth but ~zero "
               "velocity (needs metrics_cut set)\n", zvm);
    fprintf(f, "  novec         %.6f   pixels whose vector hit the 64px clamp: "
               "NO vector written (coverage failure; sky is expected here, "
               "geometry is not)\n", novec);

    // The calibration histogram. Unmasked, every pixel, raw depth as sampled.
    fprintf(f, "  depth histogram (raw .r, 16 bins over [0,1], ALL pixels):\n   ");
    for (uint32_t k = 0; k < 16; ++k)
        fprintf(f, " %5.1f%%", nAll > 0.0 ? 100.0 * (M.acc[S_HIST_D + k] / frames) / nAll : 0.0);
    fprintf(f, "\n  temporal-delta histogram (16 bins over [0,0.25]):\n   ");
    for (uint32_t k = 0; k < 16; ++k)
        fprintf(f, " %5.1f%%", nValid > 0.0 ? 100.0 * M.acc[S_HIST_T + k] / nValid : 0.0);
    fprintf(f, "\n  velocity histogram (8 log bins):\n   ");
    for (uint32_t k = 0; k < 8; ++k)
        fprintf(f, " %5.1f%%", mAll > 0.0 ? 100.0 * M.acc[S_HIST_V + k] / mAll : 0.0);
    fprintf(f, "\n");
    fclose(f);

    trace("METRICS: instab %.4f (stat %.6f edge %.6f flick %.6f) over %llu frames "
          "-> mv_metrics.txt", instab, stat, edge, flicker,
          (unsigned long long)M.counted);
}

// Fold one retired section into the long-window accumulator.
inline void fold(const uint32_t *sec)
{
    // A wrapped counter would be indistinguishable from a real measurement and
    // would quietly poison a ranking, so a frame anywhere near the uint32
    // ceiling is discarded and counted rather than trusted.
    for (uint32_t k = 0; k < S_SLOTS; ++k) {
        if (sec[k] >= 0xF0000000u) { ++M.saturated; return; }
    }
    for (uint32_t k = 0; k < S_SLOTS; ++k) {
        const bool isCount =
            (k == S_COUNT_MASKED || k == S_COUNT_GHOST || k == S_COUNT_DISOCCL ||
             k == S_COUNT_STATIC || k == S_VALID_REPROJ || k == S_ZEROVEL_MOVED ||
             k == S_FLICKER || k == S_COUNT_ALL || k == S_COUNT_NOVEC ||
             (k >= S_HIST_T && k < S_COUNT_ALL));
        M.acc[k] += isCount ? (double)sec[k]
                            : (double)sec[k] * (double)slotScale(k) / (double)kScale;
    }
    ++M.counted;
}

// ---------------------------------------------------------------- the record
struct Inputs {
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkImage     scene = VK_NULL_HANDLE;
    VkImageView sceneView = VK_NULL_HANDLE;
    VkFormat    sceneFmt = VK_FORMAT_UNDEFINED;
    VkImageView velView = VK_NULL_HANDLE;
    VkImageView depthView = VK_NULL_HANDLE;
    uint32_t    w = 0, h = 0;
};

inline void record(DeviceData &dd, VkDevice dev, const Inputs &in)
{
    if (!armed()) {
        if (M.ready && M.counted) { dump(); resetAcc(); }
        return;
    }
    if (!in.cb || !in.scene || !in.sceneView || !in.velView || !in.depthView ||
        !in.w || !in.h)
        return;
    if (!init(dd, dev)) return;
    if (!ensureTargets(dd, dev, in.w, in.h, in.sceneFmt)) return;

    const uint32_t s   = (uint32_t)(M.frame % kSect);
    const VkDeviceSize secOff = (VkDeviceSize)s * S_SLOTS * 4u;

    // Read the section this slot carried eight frames ago, BEFORE zeroing it.
    if (M.frame >= kSect && M.map)
        fold((const uint32_t *)((const uint8_t *)M.map + secOff));

    dd.cmdFillBuffer(in.cb, M.ssbo, secOff, (VkDeviceSize)(S_SLOTS * 4u), 0u);

    // One-shot: the sign image starts as uninitialised device memory, and
    // garbage there would read as real sign reversals for the first frames.
    if (!M.signCleared) {
        VkImageMemoryBarrier sb;
        memset(&sb, 0, sizeof(sb));
        sb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        sb.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        sb.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        sb.srcQueueFamilyIndex = sb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        sb.image = M.signImg;
        sb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sb.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        sb.subresourceRange.levelCount = 1; sb.subresourceRange.layerCount = 1;
        dd.cmdPipelineBarrier(in.cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &sb);
        VkClearColorValue cv;
        memset(&cv, 0, sizeof(cv));
        dd.cmdClearColorImage(in.cb, M.signImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              &cv, 1, &sb.subresourceRange);
        sb.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        sb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        sb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        dd.cmdPipelineBarrier(in.cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &sb);
        M.signCleared = true;
    }

    // Nothing to compare against on the very first frame: fill prev, skip the
    // dispatch, and start measuring next frame. Reporting a temporal residual
    // against uninitialised memory would be worse than reporting nothing.
    const bool doDispatch = M.havePrev;

    // ---- scene COLOR_ATTACHMENT -> SHADER_READ (proven layout, taa.h:2466),
    // sign GENERAL->GENERAL to order this dispatch after the last one, and the
    // fill above before either.
    VkImageMemoryBarrier pre[2];
    memset(pre, 0, sizeof(pre));
    for (int k = 0; k < 2; ++k) {
        pre[k].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        pre[k].srcQueueFamilyIndex = pre[k].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pre[k].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        pre[k].subresourceRange.levelCount = 1;
        pre[k].subresourceRange.layerCount = 1;
    }
    pre[0].image = in.scene;
    pre[0].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    pre[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    pre[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    pre[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    pre[1].image = M.signImg;
    pre[1].oldLayout = pre[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    pre[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    pre[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    VkBufferMemoryBarrier bb;
    memset(&bb, 0, sizeof(bb));
    bb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bb.srcQueueFamilyIndex = bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.buffer = M.ssbo; bb.offset = secOff; bb.size = (VkDeviceSize)(S_SLOTS * 4u);
    bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    dd.cmdPipelineBarrier(in.cb,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &bb, 2, pre);

    if (doDispatch) {
        // The set for this section was last used kSect frames ago, so it is
        // safe to rewrite - and rewriting every frame means a rotated scene
        // view or a resize is picked up instead of silently measured wrong.
        VkDescriptorImageInfo ii[4];
        memset(ii, 0, sizeof(ii));
        ii[0].sampler = M.samp; ii[0].imageView = in.sceneView;
        ii[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ii[1].sampler = M.samp; ii[1].imageView = M.prevView;
        ii[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ii[2].sampler = M.samp; ii[2].imageView = in.velView;
        ii[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ii[3].sampler = M.samp; ii[3].imageView = in.depthView;
        ii[3].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkDescriptorBufferInfo bi;
        bi.buffer = M.ssbo; bi.offset = secOff; bi.range = (VkDeviceSize)(S_SLOTS * 4u);
        VkDescriptorImageInfo si;
        memset(&si, 0, sizeof(si));
        si.imageView = M.signView; si.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet wr[6];
        memset(wr, 0, sizeof(wr));
        for (int k = 0; k < 6; ++k) {
            wr[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[k].dstSet = M.sets[s]; wr[k].dstBinding = (uint32_t)k;
            wr[k].descriptorCount = 1;
            wr[k].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            wr[k].pImageInfo = &ii[k < 4 ? k : 0];
        }
        wr[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[4].pImageInfo = nullptr; wr[4].pBufferInfo = &bi;
        wr[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        wr[5].pImageInfo = &si;
        dd.updateDescriptorSets(dev, 6, wr, 0, nullptr);

        // NB: 'far' and 'near' are macros from WinDef.h - naming a member
        // 'far' silently deletes it. The shader's field is still depthFar.
        struct PC { uint32_t dim[2]; float cut, farCut; uint32_t flags, frame; float flick; } pc;
        pc.dim[0] = in.w; pc.dim[1] = in.h;
        pc.cut = depthCut(); pc.farCut = depthFar();
        pc.flags = (uint32_t)(maskMode() & 3);
        pc.frame = (uint32_t)M.frame;
        pc.flick = flickThr();

        dd.cmdBindPipeline(in.cb, VK_PIPELINE_BIND_POINT_COMPUTE, M.pipe);
        dd.cmdBindDescriptorSets(in.cb, VK_PIPELINE_BIND_POINT_COMPUTE, M.pl,
                                 0, 1, &M.sets[s], 0, nullptr);
        dd.cmdPushConstants(in.cb, M.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        dd.cmdDispatch(in.cb, (in.w + 7) / 8, (in.h + 7) / 8, 1);
    }

    // ---- scene -> TRANSFER_SRC, prev -> TRANSFER_DST, copy, restore.
    VkImageMemoryBarrier mid[2];
    memcpy(mid, pre, sizeof(mid));
    mid[0].image = in.scene;
    mid[0].oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    mid[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    mid[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    mid[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    mid[1].image = M.prevImg;
    mid[1].oldLayout = M.havePrev ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                  : VK_IMAGE_LAYOUT_UNDEFINED;
    mid[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    mid[1].srcAccessMask = M.havePrev ? VK_ACCESS_SHADER_READ_BIT : 0;
    mid[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    dd.cmdPipelineBarrier(in.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, mid);

    VkImageCopy cp;
    memset(&cp, 0, sizeof(cp));
    cp.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    cp.srcSubresource.layerCount = 1;
    cp.dstSubresource = cp.srcSubresource;
    cp.extent.width = in.w; cp.extent.height = in.h; cp.extent.depth = 1;
    dd.cmdCopyImage(in.cb, in.scene, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    M.prevImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);

    VkImageMemoryBarrier post[2];
    memcpy(post, mid, sizeof(post));
    post[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    post[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    post[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    post[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    post[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    post[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    post[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    post[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dd.cmdPipelineBarrier(in.cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 0, nullptr, 0, nullptr, 2, post);

    M.havePrev = true;
    ++M.frame;

    const int every = reportEvery();
    if (every > 0 && M.counted >= (uint64_t)every) { dump(); resetAcc(); }
}

inline void shutdown(DeviceData &dd, VkDevice dev)
{
    if (!M.tried) return;
    if (M.counted) dump();
    if (dev != VK_NULL_HANDLE && M.ready) {
        dd.deviceWaitIdle(dev);
        destroyTargets(dd, dev);
        if (M.map)     { dd.unmapMemory(dev, M.ssboMem); M.map = nullptr; }
        if (M.ssbo)    { dd.destroyBuffer(dev, M.ssbo, nullptr); M.ssbo = VK_NULL_HANDLE; }
        if (M.ssboMem) { dd.freeMemory(dev, M.ssboMem, nullptr); M.ssboMem = VK_NULL_HANDLE; }
        if (M.samp)    { dd.destroySampler(dev, M.samp, nullptr); M.samp = VK_NULL_HANDLE; }
        if (M.pool)    { dd.destroyDescriptorPool(dev, M.pool, nullptr); M.pool = VK_NULL_HANDLE; }
        if (M.pipe)    { dd.destroyPipeline(dev, M.pipe, nullptr); M.pipe = VK_NULL_HANDLE; }
        if (M.pl)      { dd.destroyPipelineLayout(dev, M.pl, nullptr); M.pl = VK_NULL_HANDLE; }
        if (M.dsl)     { dd.destroyDescriptorSetLayout(dev, M.dsl, nullptr); M.dsl = VK_NULL_HANDLE; }
        if (M.sm)      { dd.destroyShaderModule(dev, M.sm, nullptr); M.sm = VK_NULL_HANDLE; }
    }
    M.ready = false; M.tried = false;
}

}   // namespace metrics
