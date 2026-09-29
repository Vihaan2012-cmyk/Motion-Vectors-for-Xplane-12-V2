// dlaa.h - NVIDIA DLAA (DLSS Super Resolution at 1:1) as an anti-aliasing backend,
// selected with taa.backend = taa | dlaa | dlss (dlss runs as DLAA until render-scale
// control exists; X-Plane renders at output resolution).
//
// ---- WHY THIS SHAPE ------------------------------------------------------------
// The v1.3.0 fork (binaries only, decoded 2026-09-17) drove DLSS through raw NGX and fed
// it the velocity target unconverted, with sign/scale knobs tuned by eye. This backend
// uses the Streamline context the vulkan-1.dll shim already owns (the path DLSS-D proved
// on the RTX 4060, see dlssd.h) with NVIDIA's 2.13 runtime in MotionVectors\beta, and
// feeds it the SAME vectors our TAA resolve uses: the resolve runs in "prep mode"
// (uReprojParams4.x) and writes, per pixel, the finished history offset - dilated,
// depth-reconstructed where nothing wrote, near-field/body-frame rules applied - instead
// of a colour. So every fix the TAA path has (glass alpha select, cockpit screen class,
// pulled-geometry jitter, draw-time camera matrices) reaches DLSS for free.
//
//   motion vectors  history[hw].rg  = prevUV - currUV, UV units, Y already in UV
//                                     orientation -> mvecScale (1, 1)
//   bias mask       hdepth[hw].r    = 1 on the cockpit screen class (coverage 0.5),
//                                     and on light-class pixels when
//                                     taa.dlss_bias_lights=1 -> kBufferTypeBiasCurrentColorHint
//   depth           X-Plane's hardware depth, reverse-Z -> depthInverted
//   jitter          the resolve's un-jitter S (UV) in pixels, sign from the convention
//                   "rendered content shifted by S": taa.dlss_jitter_sx/sy (default -1)
//                   exist ONLY because the DLSS sign could not be measured offline -
//                   a wrong sign shows as a soft, crawling image; flip and compare.
//
// Copyright (C) 2026 MotionVectors contributors. SPDX: GPL-3.0-or-later
#pragma once
#include "sl_dlss.h"

namespace mvdlaa {

static const sl::Feature kDLSS = sl::kFeatureDLSS;   // 0

typedef sl::Result (*PFN_slDLSSSetOptions)(const sl::ViewportHandle &, const sl::DLSSOptions &);

// eWarnOutOfVRAM is a WARNING: Streamline still evaluated and the output is valid. The
// first DLAA flight (2026-09-17) counted 30 of them as failures and switched itself
// off at OMDB with the A380 loaded; every one matched an "Exceeded VRAM budget" line.
inline bool slOk(sl::Result r) { return r == sl::Result::eOk || r == sl::Result::eWarnOutOfVRAM; }
inline void noteVramWarn(sl::Result r, const char *where)
{
    if (r != sl::Result::eWarnOutOfVRAM) return;
    static uint64_t n = 0;
    if ((n++ % 600) == 0)
        trace("DLAA: %s -> eWarnOutOfVRAM (%llu so far) - over the VRAM budget, still evaluating",
              where, (unsigned long long)n);
}

// taa.backend, read live. Anything but "dlaa"/"dlss" is the built-in TAA.
inline bool wanted()
{
    std::string v;
    if (!live::lookup("taa.backend", v)) {
        const char *e = getenv("TAA_BACKEND");
        if (!e) return false;
        v = e;
    }
    return v == "dlaa" || v == "dlss";
}

struct State {
    bool adopted = false, supported = false, failed = false, announced = false;
    PFN_slDLSSSetOptions SetOptions = nullptr;
    uint32_t optW = 0, optH = 0; int optPreset = -1;
    VkImage outImg = VK_NULL_HANDLE; VkDeviceMemory outMem = VK_NULL_HANDLE; VkImageView outView = VK_NULL_HANDLE;
    uint32_t outW = 0, outH = 0; VkFormat outFmt = VK_FORMAT_UNDEFINED;
    VkImage outLayoutFor = VK_NULL_HANDLE;   // the output image last moved UNDEFINED -> GENERAL
    uint32_t frame = 0, fails = 0;
};
inline State &state() { static State s; return s; }

// Everything that can be decided before the resolve dispatch: Streamline adopted from the
// shim, the DLSS feature supported, its option setter resolved, the output image sized.
// False = run the normal TAA this frame (nothing was recorded).
static bool prepare(DeviceData &dd, VkDevice dev, VkPhysicalDevice phys,
                    uint32_t w, uint32_t h, VkFormat fmt, bool haveDepth, bool haveBiasTarget)
{
    State &s = state();
    if (!wanted() || s.failed) return false;
    if (!live::envset("TAA_SL_SHIM_OWNS")) {
        if (!s.announced) { s.announced = true;
            trace("DLAA: taa.backend=dlaa needs the vulkan-1.dll shim to own Streamline (it does not here) - using TAA"); }
        return false;
    }
    if (!haveDepth || !w || !h) return false;
    (void)haveBiasTarget;
    if (!s.adopted) {
        s.adopted = true;
        if (!mvdlssd::load()) { trace("DLAA: cannot bind the shim's sl.interposer exports - using TAA"); s.failed = true; return false; }
        mvdlssd::State &sl = mvdlssd::state();
        if (sl.IsFeatureSupported) {
            sl::AdapterInfo ai{}; ai.vkPhysicalDevice = phys;
            sl::Result r = sl.IsFeatureSupported(kDLSS, ai);
            s.supported = (r == sl::Result::eOk);
            trace("DLAA: slIsFeatureSupported(kFeatureDLSS) -> %s  <== %s", mvdlssd::rc(r),
                  s.supported ? "DLSS AVAILABLE" : "not available (is the shim loading the 2.13 runtime with taa.backend=dlaa at launch?)");
        }
        if (!s.supported) { s.failed = true; return false; }
    }
    if (!s.SetOptions) {
        void *p = nullptr;
        mvdlssd::State &sl = mvdlssd::state();
        if (sl.GetFeatureFunction && sl.GetFeatureFunction(kDLSS, "slDLSSSetOptions", p) == sl::Result::eOk)
            s.SetOptions = (PFN_slDLSSSetOptions)p;
        if (!s.SetOptions) return false;   // not up yet: the shim hands SL the device after vkCreateDevice
        trace("DLAA: slDLSSSetOptions resolved");
    }
    // Output image, same size and format as the scene target.
    if (!(s.outImg && s.outW == w && s.outH == h && s.outFmt == fmt)) {
        if (s.outView) { dd.destroyImageView(dev, s.outView, nullptr); s.outView = VK_NULL_HANDLE; }
        if (s.outImg)  { dd.destroyImage(dev, s.outImg, nullptr);      s.outImg  = VK_NULL_HANDLE; }
        if (s.outMem)  { dd.freeMemory(dev, s.outMem, nullptr);        s.outMem  = VK_NULL_HANDLE; }
        s.outW = s.outH = 0;
        s.outLayoutFor = VK_NULL_HANDLE;   // a recycled handle value must still get its first transition
        VkImageCreateInfo ici{}; ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D; ici.format = fmt; ici.extent = { w, h, 1 };
        ici.mipLevels = 1; ici.arrayLayers = 1; ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (dd.createImage(dev, &ici, nullptr, &s.outImg) != VK_SUCCESS) { s.outImg = VK_NULL_HANDLE; s.failed = true; return false; }
        VkMemoryRequirements mr{}; dd.getImageMemReq(dev, s.outImg, &mr);
        VkPhysicalDeviceMemoryProperties mp{}; g_getPhysMemProps(phys, &mp);
        uint32_t mt = UINT32_MAX;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((mr.memoryTypeBits & (1u << i)) &&
                (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { mt = i; break; }
        VkMemoryAllocateInfo mai{}; mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = mr.size; mai.memoryTypeIndex = mt;
        if (mt == UINT32_MAX || dd.allocateMemory(dev, &mai, nullptr, &s.outMem) != VK_SUCCESS) {
            dd.destroyImage(dev, s.outImg, nullptr); s.outImg = VK_NULL_HANDLE; s.outMem = VK_NULL_HANDLE;
            trace("DLAA: output image allocation failed (%ux%u) - using TAA", w, h); s.failed = true; return false;
        }
        dd.bindImageMemory(dev, s.outImg, s.outMem, 0);
        VkImageViewCreateInfo ivci{}; ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = s.outImg; ivci.viewType = VK_IMAGE_VIEW_TYPE_2D; ivci.format = fmt;
        ivci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        if (dd.createImageView(dev, &ivci, nullptr, &s.outView) != VK_SUCCESS) { s.outView = VK_NULL_HANDLE; s.failed = true; return false; }
        s.outW = w; s.outH = h; s.outFmt = fmt;
        s.optW = 0;   // re-send options for the new size
        s.frame = 0;
    }
    return true;
}

struct Inputs {
    VkCommandBuffer cb;
    VkImage sceneImg;  VkImageView sceneView;  VkFormat sceneFmt;  VkImageLayout sceneLayout;
    VkImage depthImg;  VkImageView depthView;  VkFormat depthFmt;  VkImageLayout depthLayout;
    VkImage mvImg;     VkImageView mvView;     // history[hw]: RG = prevUV - currUV, GENERAL
    VkImage biasImg;   VkImageView biasView;   // hdepth[hw]: R = bias, GENERAL (may be null)
    uint32_t w, h;
    const float *proj;     // cameraViewToClip, row major
    const float *reproj;   // clipToPrevClip, row major
    float sx, sy;          // the resolve's un-jitter S, UV units
    bool reset;
};

// After the prep dispatch: tag, set constants, evaluate. Returns the output image (in
// GENERAL layout, same size/format as the scene) or VK_NULL_HANDLE.
static VkImage evaluate(DeviceData &dd, const Inputs &in)
{
    State &s = state();
    mvdlssd::State &sl = mvdlssd::state();
    const sl::ViewportHandle vp(0u);

    // The output is tagged GENERAL; a fresh image is UNDEFINED until told otherwise.
    if (s.outLayoutFor != s.outImg) {
        VkImageMemoryBarrier b{}; b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = s.outImg; b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        dd.cmdPipelineBarrier(in.cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0, 0, nullptr, 0, nullptr, 1, &b);
        s.outLayoutFor = s.outImg;
    }

    const int presetKey = live::i("taa.dlss_preset", "TAA_DLSS_PRESET", 0);   // sl::DLSSPreset value, 0 = NVIDIA default
    if (s.optW != in.w || s.optH != in.h || s.optPreset != presetKey) {
        sl::DLSSOptions opt{};
        opt.mode = sl::DLSSMode::eDLAA;
        opt.outputWidth = in.w; opt.outputHeight = in.h;
        opt.colorBuffersHDR = sl::Boolean::eTrue;
        opt.useAutoExposure = sl::Boolean::eTrue;
        opt.dlaaPreset = (sl::DLSSPreset)presetKey;
        sl::Result r = s.SetOptions(vp, opt);
        trace("DLAA: slDLSSSetOptions(DLAA %ux%u, preset %d, HDR, auto exposure) -> %s",
              in.w, in.h, presetKey, mvdlssd::rc(r));
        if (!slOk(r)) { if (++s.fails >= 30) s.failed = true; return VK_NULL_HANDLE; }
        s.optW = in.w; s.optH = in.h; s.optPreset = presetKey;
    }

    sl::FrameToken *tok = nullptr;
    uint32_t fi = s.frame++;
    if (sl.GetNewFrameToken(tok, &fi) != sl::Result::eOk || !tok) return VK_NULL_HANDLE;

    sl::Constants c{};
    mvdlssd::row(c.cameraViewToClip, in.proj);
    mvdlssd::row(c.clipToPrevClip, in.reproj);
    const float jsx = live::f("taa.dlss_jitter_sx", nullptr, -1.0f);
    const float jsy = live::f("taa.dlss_jitter_sy", nullptr, -1.0f);
    c.jitterOffset = { jsx * in.sx * (float)in.w, jsy * in.sy * (float)in.h };
    c.mvecScale = { live::f("taa.dlss_mv_sx", nullptr, 1.0f), live::f("taa.dlss_mv_sy", nullptr, 1.0f) };
    c.depthInverted = live::onoff("taa.dlss_depth_inverted", nullptr, true) ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    c.cameraMotionIncluded = sl::Boolean::eTrue;
    c.motionVectors3D = sl::Boolean::eFalse;
    c.motionVectorsJittered = sl::Boolean::eFalse;
    c.motionVectorsDilated = sl::Boolean::eTrue;   // the prep pass already took the closest neighbour's vector
    c.orthographicProjection = sl::Boolean::eFalse;
    c.reset = in.reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    sl::Result cr = sl.SetConstants(c, *tok, vp);
    if (!slOk(cr)) { trace("DLAA: slSetConstants -> %s", mvdlssd::rc(cr)); return VK_NULL_HANDLE; }

    sl::Resource colIn = mvdlssd::mkRes(in.sceneImg, in.sceneView, in.sceneFmt, in.sceneLayout, in.w, in.h);
    sl::Resource depth = mvdlssd::mkRes(in.depthImg, in.depthView, in.depthFmt, in.depthLayout, in.w, in.h);
    sl::Resource mvec  = mvdlssd::mkRes(in.mvImg, in.mvView, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_GENERAL, in.w, in.h);
    sl::Resource out   = mvdlssd::mkRes(s.outImg, s.outView, s.outFmt, VK_IMAGE_LAYOUT_GENERAL, in.w, in.h);
    sl::Resource bias  = mvdlssd::mkRes(in.biasImg, in.biasView, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_GENERAL, in.w, in.h);
    std::vector<sl::ResourceTag> tags;
    tags.push_back(sl::ResourceTag(&colIn, sl::kBufferTypeScalingInputColor,  sl::ResourceLifecycle::eValidUntilEvaluate));
    tags.push_back(sl::ResourceTag(&depth, sl::kBufferTypeDepth,              sl::ResourceLifecycle::eValidUntilEvaluate));
    tags.push_back(sl::ResourceTag(&mvec,  sl::kBufferTypeMotionVectors,      sl::ResourceLifecycle::eValidUntilEvaluate));
    tags.push_back(sl::ResourceTag(&out,   sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eValidUntilEvaluate));
    if (in.biasImg != VK_NULL_HANDLE && live::onoff("taa.dlss_bias_mask", nullptr, true))
        tags.push_back(sl::ResourceTag(&bias, sl::kBufferTypeBiasCurrentColorHint, sl::ResourceLifecycle::eValidUntilEvaluate));
    sl::Result tr = sl.SetTag(vp, tags.data(), (uint32_t)tags.size(), (sl::CommandBuffer *)in.cb);
    if (!slOk(tr)) { trace("DLAA: slSetTag(%zu) -> %s", tags.size(), mvdlssd::rc(tr)); if (++s.fails >= 30) s.failed = true; return VK_NULL_HANDLE; }
    noteVramWarn(tr, "slSetTag");

    const sl::BaseStructure *inputs[1] = { &vp };
    sl::Result er = sl.EvaluateFeature(kDLSS, *tok, inputs, 1, (sl::CommandBuffer *)in.cb);
    if (!slOk(er)) {
        trace("DLAA: slEvaluateFeature(kFeatureDLSS) -> %s", mvdlssd::rc(er));
        if (++s.fails >= 30) { s.failed = true; trace("DLAA: 30 consecutive failures - falling back to TAA for this session"); }
        return VK_NULL_HANDLE;
    }
    noteVramWarn(er, "slEvaluateFeature");
    s.fails = 0;   // the limit is for a backend that never works, not for scattered hiccups
    if (!s.announced) {
        s.announced = true;
        trace("DLAA: RUNNING - %ux%u, tags colour/depth/mvec/out%s, mvecScale (%.2f, %.2f), jitter sign (%.0f, %.0f), "
              "depthInverted %d. Vectors are the TAA resolve's own (prep mode).",
              in.w, in.h, in.biasImg ? "/bias" : "", c.mvecScale.x, c.mvecScale.y, jsx, jsy,
              c.depthInverted == sl::Boolean::eTrue ? 1 : 0);
    }
    return s.outImg;
}

} // namespace mvdlaa
