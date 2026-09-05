// dlssd.h - DLSS-D Ray Reconstruction (Streamline feature kFeatureDLSS_RR = 1001)
// driven from the MotionVectors Vulkan layer, the HONEST way: NVIDIA's public
// Streamline runtime, a driver-supported feature, no patched binary, no
// signature strip, no caller spoof.
//
// WHY THIS FEATURE AND NOT 1004: measured, not assumed. sl.dlss_nr loads fine
// through this same shim ("Loaded plugin 'sl.dlss_nr' - version 2.13.0 PRODUCTION
// - id 1004"), but driver 610.88's NGX core answers 0xbad0000c to the feature-18
// requirements query, so SL drops it ("not supported on this platform") and
// slIsFeatureSupported(1004) returns eErrorFeatureNotSupported. Note the driver
// refuses to DESCRIBE the feature at all - for RR it reports "NGX feature 13
// requirements - minHW 0x160" and our Ada part is 0x190, but for NR there is no
// minHW to compare against. Whether that gate is a real silicon requirement or a
// product decision cannot be settled from here; either way nothing downstream can
// use 1004 until the driver enumerates it. Set TAA_DLSS_NR=1 to re-test on a
// newer driver - the shim already requests 1004 and the layer needs no change.
//
// ---- STATUS: RUNNING ON HARDWARE (RTX 4060, driver 610.88) --------------
//  Measured end to end: slSetVulkanInfo eOk, slIsFeatureSupported(1001) eOk,
//  slDLSSDSetOptions eOk, slEvaluateFeature clean every frame, and SL's
//  presentCommon() observed (so its bookkeeping and GC run).
//
//  It only got there once four things were true, all of them easy to get wrong:
//   1. The vulkan-1.dll shim merges SL's device requirements into
//      vkCreateDevice (NVX_binary_import, NVX_image_view_handle,
//      EXT_buffer_device_address, the 1.2 feature bits, privateData).
//   2. The shim owns slInit. A SECOND slInit from this layer throws inside
//      sl.interposer's mutex and SL swallows it WITHOUT unlocking, after which
//      every SL call returns 24 on that thread and blocks on every other.
//   3. That handshake is read with GetEnvironmentVariable, not getenv - the
//      shim publishes it with SetEnvironmentVariable and the CRT's snapshot
//      never sees it (see live::envset).
//   4. The app presents THROUGH sl.interposer's vkQueuePresentKHR, handed out
//      from our vkGetDeviceProcAddr.
//
// ---- HONEST LIMITATIONS (read before judging output) --------------------
//  * X-Plane is RASTERISED. Ray Reconstruction's denoiser is built for
//    path-traced input; here it runs as a high-quality neural AA on the guide
//    buffers we can provide (colour, depth, motion, normals).
//  * There is no albedo/roughness G-buffer to harvest, and RR refuses to
//    evaluate without them, so we feed CONSTANT guides: albedo 0.5, specular
//    0.04, roughness taa.dlssd_roughness (default 1.0 = treat everything as
//    diffuse). That is an honest description of an already-shaded frame, not a
//    real material buffer - image quality is for the screen to settle.
//  * Costs VRAM: three full-res guides plus RR's own output. Measured ~6.2 GB
//    total on an 8 GB card at 2953x1661. Default OFF (taa.dlssd).
//  * The Streamline runtime folder (streamline/) must hold sl.interposer.dll +
//    sl.common.dll + sl.dlss_d.dll + nvngx_dlssd.dll. TAA_SL_DIR overrides.
//
// Copyright (C) 2026 MotionVectors contributors. SPDX: GPL-3.0-or-later
#pragma once
#include <windows.h>
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>

#include "sl.h"
#include "sl_helpers_vk.h"
#include "sl_dlss_d.h"
#include "mv_live.h"    // live::onoff / live::f

namespace mvdlssd {

static const sl::Feature kRR = sl::kFeatureDLSS_RR;   // 1001

// ---- interposer C exports, resolved by name (we do not link the SDK libs).
typedef sl::Result (*PFN_slInit)(const sl::Preferences &, uint64_t);
typedef sl::Result (*PFN_slShutdown)();
typedef sl::Result (*PFN_slSetVulkanInfo)(const sl::VulkanInfo &);
typedef sl::Result (*PFN_slIsFeatureSupported)(sl::Feature, const sl::AdapterInfo &);
typedef sl::Result (*PFN_slIsFeatureLoaded)(sl::Feature, bool &);
typedef sl::Result (*PFN_slGetFeatureRequirements)(sl::Feature, sl::FeatureRequirements &);
typedef sl::Result (*PFN_slGetFeatureFunction)(sl::Feature, const char *, void *&);
typedef sl::Result (*PFN_slSetTag)(const sl::ViewportHandle &, const sl::ResourceTag *, uint32_t, sl::CommandBuffer *);
typedef sl::Result (*PFN_slSetConstants)(const sl::Constants &, const sl::FrameToken &, const sl::ViewportHandle &);
typedef sl::Result (*PFN_slEvaluateFeature)(sl::Feature, const sl::FrameToken &, const sl::BaseStructure **, uint32_t, sl::CommandBuffer *);
typedef sl::Result (*PFN_slAllocateResources)(sl::CommandBuffer *, sl::Feature, const sl::ViewportHandle &);
typedef sl::Result (*PFN_slGetNewFrameToken)(sl::FrameToken *&, const uint32_t *);
// resolved from the feature via slGetFeatureFunction
typedef sl::Result (*PFN_slDLSSDSetOptions)(const sl::ViewportHandle &, const sl::DLSSDOptions &);
typedef sl::Result (*PFN_slDLSSDGetOptimalSettings)(const sl::DLSSDOptions &, sl::DLSSDOptimalSettings &);

struct State {
    HMODULE lib = nullptr;
    bool    inited   = false;   // slInit + slSetVulkanInfo done
    bool    supported = false;  // slIsFeatureSupported(1001) == eOk
    bool    optionsSet = false;
    bool    announced = false;
    std::wstring dir;

    PFN_slInit                   Init = nullptr;
    PFN_slShutdown               Shutdown = nullptr;
    PFN_slSetVulkanInfo          SetVulkanInfo = nullptr;
    PFN_slIsFeatureSupported     IsFeatureSupported = nullptr;
    PFN_slIsFeatureLoaded        IsFeatureLoaded = nullptr;
    PFN_slGetFeatureRequirements GetFeatureRequirements = nullptr;
    PFN_slGetFeatureFunction     GetFeatureFunction = nullptr;
    PFN_slSetTag                 SetTag = nullptr;
    PFN_slSetConstants           SetConstants = nullptr;
    PFN_slEvaluateFeature        EvaluateFeature = nullptr;
    PFN_slAllocateResources      AllocateResources = nullptr;
    PFN_slGetNewFrameToken       GetNewFrameToken = nullptr;
    PFN_slDLSSDSetOptions        DLSSDSetOptions = nullptr;
    PFN_slDLSSDGetOptimalSettings DLSSDGetOptimalSettings = nullptr;

    // our own RR output image (RR must not write in-place over the input).
    VkImage        outImg  = VK_NULL_HANDLE;
    VkDeviceMemory outMem  = VK_NULL_HANDLE;
    VkImageView    outView = VK_NULL_HANDLE;
    uint32_t       outW = 0, outH = 0;
    VkFormat       outFmt = VK_FORMAT_UNDEFINED;

    // Neutral guide buffers. X-Plane is a raster engine and harvests no albedo
    // G-buffer, but RR refuses to evaluate without kBufferTypeAlbedo. A constant
    // mid-grey albedo (and a dark specular albedo) means "already demodulated",
    // which is the honest description of a lit raster frame. Cleared once.
    VkImage        gAlbImg = VK_NULL_HANDLE, gSpecImg = VK_NULL_HANDLE, gRghImg = VK_NULL_HANDLE;
    VkDeviceMemory gAlbMem = VK_NULL_HANDLE, gSpecMem = VK_NULL_HANDLE, gRghMem = VK_NULL_HANDLE;
    VkImageView    gAlbView = VK_NULL_HANDLE, gSpecView = VK_NULL_HANDLE, gRghView = VK_NULL_HANDLE;
    uint32_t       gW = 0, gH = 0;
    bool           gCleared = false;

    uint32_t frame = 0;
    std::vector<std::string> extStore;
    std::wstring logPath;
};
inline State &state() { static State s; return s; }

// Remembered so the lazy adoption can still answer slIsFeatureSupported.
static VkPhysicalDevice g_dlssdPhys = VK_NULL_HANDLE;

inline bool enabled() { return live::onoff("taa.dlssd", "TAA_DLSSD", false); }

static void NVSDK_log(sl::LogType t, const char *msg)
{
    std::string s = msg ? msg : "";
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    trace("DLSSD [sl %d] %s", (int)t, s.c_str());
}

static const char *rc(sl::Result r)
{
    switch (r) {
        case sl::Result::eOk: return "eOk";
        case sl::Result::eErrorFeatureNotSupported: return "eErrorFeatureNotSupported";
        case sl::Result::eErrorFeatureMissing: return "eErrorFeatureMissing";
        case sl::Result::eErrorNotInitialized: return "eErrorNotInitialized";
        case sl::Result::eErrorVulkanAPI: return "eErrorVulkanAPI";
        case sl::Result::eErrorMissingInputParameter: return "eErrorMissingInputParameter";
        case sl::Result::eErrorMissingConstants: return "eErrorMissingConstants";
        case sl::Result::eErrorMissingResourceState: return "eErrorMissingResourceState";
        default: return "eError(other)";
    }
}

// ---- load the interposer + resolve exports + slInit. Called once, before
// vkCreateDevice (so it can also report device-extension requirements).
static bool load()
{
    State &s = state();
    if (s.inited) return true;
    if (s.lib) return false;

    wchar_t base[MAX_PATH * 2];
    const char *env = getenv("TAA_SL_DIR");
    if (env && *env) MultiByteToWideChar(CP_UTF8, 0, env, -1, base, MAX_PATH * 2);
    else {
        GetModuleFileNameW(nullptr, base, MAX_PATH * 2);
        std::wstring p = base; p = p.substr(0, p.find_last_of(L"\\/")) + L"\\MotionVectors\\streamline";
        wcscpy(base, p.c_str());
    }
    s.dir = base;
    // SHIM MODE: sl.interposer is installed as X-Plane's vulkan-1.dll (the
    // standard OptiScaler/ReShade injection), so Streamline IS the app's Vulkan
    // module and holds the instance/device context at slInit - the thing the
    // layer path lacked, which made slInit null-deref. Resolve SL from that
    // already-loaded module; do NOT load a second copy (two SL contexts fight).
    // The vulkan-1.dll shim already called slInit and owns the one Streamline
    // context in this process. A SECOND slInit here throws inside
    // sl.interposer's global mutex; SL's catch-all swallows the exception
    // WITHOUT unlocking it, and from then on every SL call returns
    // eErrorExceptionHandler (24) on this thread and blocks forever on any
    // other. Bind to the module the shim already initialised instead.
    const bool shimOwns = live::envset("TAA_SL_SHIM_OWNS");
    if (shimOwns) {
        s.lib = GetModuleHandleW(L"sl.interposer.dll");
        if (!s.lib) {
            trace("DLSSD: shim owns Streamline but sl.interposer.dll is not loaded - aborting");
            return false;
        }
        trace("DLSSD: shim owns Streamline - binding to its already-initialised context");
    }
    const bool shim = live::envset("TAA_SL_SHIM");
    if (shimOwns) {
        /* s.lib already resolved above */
    } else if (shim) {
        s.lib = GetModuleHandleW(L"vulkan-1.dll");
        if (!s.lib || !GetProcAddress(s.lib, "slInit")) {
            trace("DLSSD: SHIM mode but vulkan-1.dll is not the SL interposer "
                  "(no slInit export) - install sl.interposer.dll as "
                  "<X-Plane>\\vulkan-1.dll. Aborting.");
            s.lib = nullptr; return false;
        }
        trace("DLSSD: SHIM mode - Streamline is the app's vulkan-1.dll, single context.");
    } else {
        std::wstring interposer = s.dir + L"\\sl.interposer.dll";
        SetDllDirectoryW(s.dir.c_str());
        s.lib = LoadLibraryW(interposer.c_str());
        SetDllDirectoryW(nullptr);
        if (!s.lib) {
            trace("DLSSD: LoadLibrary sl.interposer.dll failed (%lu) in %ls - need "
                  "sl.interposer.dll + sl.common.dll + sl.dlss_d.dll + nvngx_dlssd.dll there",
                  GetLastError(), s.dir.c_str());
            return false;
        }
    }
#define G(field, pfn, name) s.field = (pfn)(void*)GetProcAddress(s.lib, name); \
    if (!s.field) trace("DLSSD: export %s MISSING", name)
    G(Init, PFN_slInit, "slInit");
    G(Shutdown, PFN_slShutdown, "slShutdown");
    G(SetVulkanInfo, PFN_slSetVulkanInfo, "slSetVulkanInfo");
    G(IsFeatureSupported, PFN_slIsFeatureSupported, "slIsFeatureSupported");
    G(IsFeatureLoaded, PFN_slIsFeatureLoaded, "slIsFeatureLoaded");
    G(GetFeatureRequirements, PFN_slGetFeatureRequirements, "slGetFeatureRequirements");
    G(GetFeatureFunction, PFN_slGetFeatureFunction, "slGetFeatureFunction");
    G(SetTag, PFN_slSetTag, "slSetTag");
    G(SetConstants, PFN_slSetConstants, "slSetConstants");
    G(EvaluateFeature, PFN_slEvaluateFeature, "slEvaluateFeature");
    G(AllocateResources, PFN_slAllocateResources, "slAllocateResources");
    G(GetNewFrameToken, PFN_slGetNewFrameToken, "slGetNewFrameToken");
#undef G
    if (!s.Init || !s.SetVulkanInfo || !s.EvaluateFeature || !s.SetTag ||
        !s.SetConstants || !s.GetFeatureFunction || !s.GetNewFrameToken) {
        trace("DLSSD: interposer missing required exports - aborting"); return false;
    }

    // Bound to the shim's context: exports resolved, nothing left to initialise.
    if (shimOwns) return true;

    { wchar_t tmp[MAX_PATH]; GetTempPathW(MAX_PATH, tmp); s.logPath = std::wstring(tmp) + L"mv_dlssd"; CreateDirectoryW(s.logPath.c_str(), nullptr); }

    static const wchar_t *paths[1]; paths[0] = s.dir.c_str();
    static sl::Feature feats[1]; feats[0] = kRR;
    sl::Preferences pref{};
    pref.logLevel = sl::LogLevel::eDefault;
    pref.logMessageCallback = NVSDK_log;
    pref.pathsToPlugins = paths;
    pref.numPathsToPlugins = 1;
    pref.pathToLogsAndData = s.logPath.c_str();
    pref.flags = sl::PreferenceFlags::eUseManualHooking | sl::PreferenceFlags::eDisableCLStateTracking;
    pref.featuresToLoad = feats;
    pref.numFeaturesToLoad = 1;
    pref.applicationId = 0x5A1D0002u;
    pref.engine = sl::EngineType::eCustom;
    pref.renderAPI = sl::RenderAPI::eVulkan;

    sl::Result r = s.Init(pref, sl::kSDKVersion);
    trace("DLSSD: slInit(Vulkan, manual hooking) -> %s", rc(r));
    return r == sl::Result::eOk;
}

// Phase 1: before vkCreateDevice - add the device extensions RR wants that the
// driver offers, exactly as the layer does for its own needs.
static void deviceExtensions(VkInstance inst, VkPhysicalDevice phys,
                             const std::vector<VkExtensionProperties> &have,
                             std::vector<const char*> &exts)
{
    (void)inst;
    if (!load()) return;
    State &s = state();
    if (!s.GetFeatureRequirements) return;
    sl::FeatureRequirements req{};
    sl::Result r = s.GetFeatureRequirements(kRR, req);
    trace("DLSSD: slGetFeatureRequirements(1001) -> %s | devExt=%u instExt=%u vkCompute=%u vkGfx=%u",
          rc(r), req.vkNumDeviceExtensions, req.vkNumInstanceExtensions,
          req.vkNumComputeQueuesRequired, req.vkNumGraphicsQueuesRequired);
    if (r != sl::Result::eOk) return;
    for (uint32_t i = 0; i < req.vkNumDeviceExtensions && req.vkDeviceExtensions; ++i) {
        const char *name = req.vkDeviceExtensions[i];
        bool offered = false, already = false;
        for (const VkExtensionProperties &h : have) if (!strcmp(h.extensionName, name)) offered = true;
        for (const char *e : exts) if (!strcmp(e, name)) already = true;
        if (offered && !already) s.extStore.push_back(name);
        trace("DLSSD:   device ext %-44s %s", name, offered ? (already ? "(have)" : "(added)") : "NOT OFFERED");
    }
    for (const std::string &n : s.extStore) exts.push_back(n.c_str());
}

// Adopt the Streamline context the vulkan-1.dll shim already built: it did
// slInit + slSetVulkanInfo with the app-visible device and confirmed the
// feature. We only resolve exports and mark ourselves live, so the per-frame
// path (slSetTag / slEvaluateFeature) runs. Callable at device creation OR on
// the first frame after taa.dlssd is switched on in the live ini.
static bool adoptShimContext()
{
    State &s = state();
    if (s.inited) return true;
    if (!s.lib && !load()) return false;
    if (!s.lib) return false;
    s.inited = true;
    if (s.IsFeatureSupported) {
        sl::AdapterInfo ai{}; ai.vkPhysicalDevice = g_dlssdPhys;
        sl::Result sr = s.IsFeatureSupported(kRR, ai);
        s.supported = (sr == sl::Result::eOk);
        trace("DLSSD: adopted the shim's Streamline context; slIsFeatureSupported(1001) -> %s  <== %s",
              rc(sr), s.supported ? "RR AVAILABLE" : "not available on this adapter");
    }
    return s.supported;
}

// Phase 2: after vkCreateDevice - give SL our device and check support.
static void initAtDevice(VkInstance inst, VkPhysicalDevice phys, VkDevice dev, uint32_t family)
{
    g_dlssdPhys = phys;
    // When the vulkan-1.dll shim hosts Streamline, IT calls slInit +
    // slSetVulkanInfo with the app-visible device (the loader top-level handle).
    // The layer only has its chain-internal device handle, which the loader does
    // not recognise - passing it to slSetVulkanInfo builds an all-null dispatch
    // and crashes Streamline. So the layer must NOT init Streamline itself here.
    if (live::envset("TAA_SL_SHIM_OWNS")) {
        adoptShimContext();
        return;
    }
    State &s = state();
    if (!s.lib || !s.SetVulkanInfo) return;
    sl::VulkanInfo vi{};
    vi.device = dev; vi.instance = inst; vi.physicalDevice = phys;
    vi.computeQueueFamily = family; vi.computeQueueIndex = 0;
    vi.graphicsQueueFamily = family; vi.graphicsQueueIndex = 0;
    sl::Result r = s.SetVulkanInfo(vi);
    trace("DLSSD: slSetVulkanInfo(family=%u) -> %s", family, rc(r));
    if (r != sl::Result::eOk) return;
    s.inited = true;

    sl::AdapterInfo ai{}; ai.vkPhysicalDevice = phys;
    if (s.IsFeatureSupported) {
        sl::Result sr = s.IsFeatureSupported(kRR, ai);
        s.supported = (sr == sl::Result::eOk);
        trace("DLSSD: slIsFeatureSupported(1001) -> %s  <== %s", rc(sr),
              s.supported ? "RR AVAILABLE" : "not available on this adapter");
    }
    // resolve the RR option setters from the feature
    if (s.supported && s.GetFeatureFunction) {
        void *p = nullptr;
        if (s.GetFeatureFunction(kRR, "slDLSSDSetOptions", p) == sl::Result::eOk)
            s.DLSSDSetOptions = (PFN_slDLSSDSetOptions)p;
        p = nullptr;
        if (s.GetFeatureFunction(kRR, "slDLSSDGetOptimalSettings", p) == sl::Result::eOk)
            s.DLSSDGetOptimalSettings = (PFN_slDLSSDGetOptimalSettings)p;
        trace("DLSSD: option fns setOptions=%s optimal=%s",
              s.DLSSDSetOptions ? "ok" : "MISSING", s.DLSSDGetOptimalSettings ? "ok" : "MISSING");
    }
}

// ---- our output image, created lazily to match the frame size/format.
static bool ensureOutput(VkDevice dev, uint32_t w, uint32_t h, VkFormat fmt,
                         PFN_vkCreateImage crImg, PFN_vkAllocateMemory alloc,
                         PFN_vkBindImageMemory bind, PFN_vkGetImageMemoryRequirements memReq,
                         PFN_vkCreateImageView crView, PFN_vkDestroyImage dImg,
                         PFN_vkFreeMemory fMem, PFN_vkDestroyImageView dView,
                         VkPhysicalDevice phys)
{
    State &s = state();
    if (s.outImg && s.outW == w && s.outH == h && s.outFmt == fmt) return true;
    if (s.outView) { dView(dev, s.outView, nullptr); s.outView = VK_NULL_HANDLE; }
    if (s.outImg)  { dImg(dev, s.outImg, nullptr);   s.outImg  = VK_NULL_HANDLE; }
    if (s.outMem)  { fMem(dev, s.outMem, nullptr);   s.outMem  = VK_NULL_HANDLE; }

    VkImageCreateInfo ici{}; ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D; ici.format = fmt;
    ici.extent = { w, h, 1 }; ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT; ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (crImg(dev, &ici, nullptr, &s.outImg) != VK_SUCCESS) { s.outImg = VK_NULL_HANDLE; return false; }

    VkMemoryRequirements mr{}; memReq(dev, s.outImg, &mr);
    VkPhysicalDeviceMemoryProperties mp{}; g_getPhysMemProps(phys, &mp);
    uint32_t mt = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((mr.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { mt = i; break; }
    if (mt == UINT32_MAX) { dImg(dev, s.outImg, nullptr); s.outImg = VK_NULL_HANDLE; return false; }
    VkMemoryAllocateInfo mai{}; mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size; mai.memoryTypeIndex = mt;
    if (alloc(dev, &mai, nullptr, &s.outMem) != VK_SUCCESS) { dImg(dev, s.outImg, nullptr); s.outImg = VK_NULL_HANDLE; return false; }
    bind(dev, s.outImg, s.outMem, 0);

    VkImageViewCreateInfo ivci{}; ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = s.outImg; ivci.viewType = VK_IMAGE_VIEW_TYPE_2D; ivci.format = fmt;
    ivci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    if (crView(dev, &ivci, nullptr, &s.outView) != VK_SUCCESS) { s.outView = VK_NULL_HANDLE; return false; }
    s.outW = w; s.outH = h; s.outFmt = fmt;
    return true;
}

static inline sl::Resource mkRes(VkImage img, VkImageView view, VkFormat fmt,
                                 VkImageLayout layout, uint32_t w, uint32_t h)
{
    sl::Resource r(sl::ResourceType::eTex2d, (void*)img, nullptr, (void*)view, (uint32_t)layout);
    r.width = w; r.height = h; r.nativeFormat = (uint32_t)fmt;
    r.mipLevels = 1; r.arrayLayers = 1;
    return r;
}

static inline void row(sl::float4x4 &m, const float *p) {
    for (int r = 0; r < 4; ++r) m.setRow((uint32_t)r, sl::float4(p[r*4+0], p[r*4+1], p[r*4+2], p[r*4+3]));
}

// Per-frame: tag inputs, set constants, evaluate RR, copy result over the scene
// image so the existing present path shows it. Returns true if RR ran.
// Layouts are the CURRENT layout of each image at call time (resolve point).
struct RecordInputs {
    VkCommandBuffer cb;
    VkImage  sceneImg;  VkImageView sceneView;  VkFormat sceneFmt;  VkImageLayout sceneLayout;
    VkImage  velImg;    VkImageView velView;    VkFormat velFmt;    VkImageLayout velLayout;
    VkImage  depthImg;  VkImageView depthView;  VkFormat depthFmt;  VkImageLayout depthLayout;
    VkImage  normImg;   VkImageView normView;   VkFormat normFmt;   VkImageLayout normLayout;
    uint32_t w, h;
    const float *proj;      // cameraViewToClip (row major, 16)
    const float *reproj;    // clipToPrevClip   (row major, 16)
    float jitterX, jitterY; // pixel-space jitter
    bool depthInverted;
};

// One constant-colour full-res image, allocated once and cleared once.
static bool makeGuide(VkDevice dev, DeviceData &dd, VkPhysicalDevice phys,
                      uint32_t w, uint32_t h, VkFormat fmt,
                      VkImage &img, VkDeviceMemory &mem, VkImageView &view)
{
    VkImageCreateInfo ici{}; ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D; ici.format = fmt;
    ici.extent = { w, h, 1 }; ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT; ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (dd.createImage(dev, &ici, nullptr, &img) != VK_SUCCESS) { img = VK_NULL_HANDLE; return false; }

    VkMemoryRequirements mr{}; dd.getImageMemReq(dev, img, &mr);
    VkPhysicalDeviceMemoryProperties mp{}; g_getPhysMemProps(phys, &mp);
    uint32_t mt = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((mr.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { mt = i; break; }
    if (mt == UINT32_MAX) { dd.destroyImage(dev, img, nullptr); img = VK_NULL_HANDLE; return false; }
    VkMemoryAllocateInfo mai{}; mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size; mai.memoryTypeIndex = mt;
    if (dd.allocateMemory(dev, &mai, nullptr, &mem) != VK_SUCCESS) {
        dd.destroyImage(dev, img, nullptr); img = VK_NULL_HANDLE; return false;
    }
    dd.bindImageMemory(dev, img, mem, 0);

    VkImageViewCreateInfo ivci{}; ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = img; ivci.viewType = VK_IMAGE_VIEW_TYPE_2D; ivci.format = fmt;
    ivci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    if (dd.createImageView(dev, &ivci, nullptr, &view) != VK_SUCCESS) { view = VK_NULL_HANDLE; return false; }
    return true;
}

static bool ensureGuides(VkDevice dev, DeviceData &dd, VkPhysicalDevice phys,
                         VkCommandBuffer cb, uint32_t w, uint32_t h)
{
    State &s = state();
    if (s.gAlbImg && s.gW == w && s.gH == h) return true;
    if (s.gAlbView)  { dd.destroyImageView(dev, s.gAlbView, nullptr);  s.gAlbView = VK_NULL_HANDLE; }
    if (s.gAlbImg)   { dd.destroyImage(dev, s.gAlbImg, nullptr);       s.gAlbImg = VK_NULL_HANDLE; }
    if (s.gAlbMem)   { dd.freeMemory(dev, s.gAlbMem, nullptr);         s.gAlbMem = VK_NULL_HANDLE; }
    if (s.gSpecView) { dd.destroyImageView(dev, s.gSpecView, nullptr); s.gSpecView = VK_NULL_HANDLE; }
    if (s.gSpecImg)  { dd.destroyImage(dev, s.gSpecImg, nullptr);      s.gSpecImg = VK_NULL_HANDLE; }
    if (s.gSpecMem)  { dd.freeMemory(dev, s.gSpecMem, nullptr);        s.gSpecMem = VK_NULL_HANDLE; }
    if (s.gRghView)  { dd.destroyImageView(dev, s.gRghView, nullptr);  s.gRghView = VK_NULL_HANDLE; }
    if (s.gRghImg)   { dd.destroyImage(dev, s.gRghImg, nullptr);       s.gRghImg = VK_NULL_HANDLE; }
    if (s.gRghMem)   { dd.freeMemory(dev, s.gRghMem, nullptr);         s.gRghMem = VK_NULL_HANDLE; }
    s.gCleared = false;

    const VkFormat gf = VK_FORMAT_R8G8B8A8_UNORM;
    if (!makeGuide(dev, dd, phys, w, h, gf, s.gAlbImg,  s.gAlbMem,  s.gAlbView))  return false;
    if (!makeGuide(dev, dd, phys, w, h, gf, s.gSpecImg, s.gSpecMem, s.gSpecView)) return false;
    if (!makeGuide(dev, dd, phys, w, h, gf, s.gRghImg,  s.gRghMem,  s.gRghView))  return false;
    s.gW = w; s.gH = h;

    // UNDEFINED -> TRANSFER_DST, clear, -> GENERAL and leave them there.
    VkImageMemoryBarrier b[3]{};
    for (int i = 0; i < 3; ++i) {
        b[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b[i].srcQueueFamilyIndex = b[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[i].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b[i].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b[i].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    }
    b[0].image = s.gAlbImg; b[1].image = s.gSpecImg; b[2].image = s.gRghImg;
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          0, 0, nullptr, 0, nullptr, 3, b);

    VkImageSubresourceRange rng = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    VkClearColorValue alb{};  alb.float32[0] = alb.float32[1] = alb.float32[2] = 0.5f; alb.float32[3] = 1.0f;
    VkClearColorValue spec{}; spec.float32[0] = spec.float32[1] = spec.float32[2] = 0.04f; spec.float32[3] = 1.0f;
    dd.cmdClearColorImage(cb, s.gAlbImg,  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &alb,  1, &rng);
    dd.cmdClearColorImage(cb, s.gSpecImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &spec, 1, &rng);
    // Fully rough: tells RR to treat every pixel as diffuse, so it accumulates
    // temporally rather than trying to reconstruct sharp specular from a frame
    // that is already shaded. taa.dlssd_roughness overrides.
    float rv = live::f("taa.dlssd_roughness", nullptr, 1.0f);
    if (rv < 0.0f) rv = 0.0f; if (rv > 1.0f) rv = 1.0f;
    VkClearColorValue rgh{}; rgh.float32[0] = rgh.float32[1] = rgh.float32[2] = rv; rgh.float32[3] = 1.0f;
    dd.cmdClearColorImage(cb, s.gRghImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &rgh, 1, &rng);

    for (int i = 0; i < 3; ++i) {
        b[i].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b[i].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    }
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                          0, 0, nullptr, 0, nullptr, 3, b);
    s.gCleared = true;
    trace("DLSSD: neutral guide buffers ready %ux%u (albedo 0.5, specular 0.04) - "
          "X-Plane harvests no albedo G-buffer; RR treats the frame as pre-demodulated", w, h);
    trace("DLSSD: roughness guide = %.2f (taa.dlssd_roughness)", rv);
    return true;
}

// The shim hands Streamline the device AFTER vkCreateDevice returns - calling
// slSetVulkanInfo from inside it re-enters sl.interposer's lock and throws - so
// at layer device-init time the feature is not up and slGetFeatureFunction
// answers "kFeatureDLSS_RR has not been initialized yet". Resolve on first use.
static bool ensureFeatureFns()
{
    State &s = state();
    if (s.DLSSDSetOptions && s.DLSSDGetOptimalSettings) return true;
    if (!s.GetFeatureFunction) return false;
    void *p = nullptr;
    if (!s.DLSSDSetOptions &&
        s.GetFeatureFunction(kRR, "slDLSSDSetOptions", p) == sl::Result::eOk)
        s.DLSSDSetOptions = (PFN_slDLSSDSetOptions)p;
    p = nullptr;
    if (!s.DLSSDGetOptimalSettings &&
        s.GetFeatureFunction(kRR, "slDLSSDGetOptimalSettings", p) == sl::Result::eOk)
        s.DLSSDGetOptimalSettings = (PFN_slDLSSDGetOptimalSettings)p;

    static bool said = false;
    if (s.DLSSDSetOptions && !said) {
        said = true;
        trace("DLSSD: feature fns resolved on first use - setOptions=ok optimal=%s",
              s.DLSSDGetOptimalSettings ? "ok" : "MISSING");
    }
    return s.DLSSDSetOptions != nullptr;
}

static bool record(DeviceData &dd, VkDevice dev, VkPhysicalDevice phys, const RecordInputs &in)
{
    State &s = state();
    if (!enabled()) return false;
    g_dlssdPhys = phys;
    if (!s.inited && live::envset("TAA_SL_SHIM_OWNS") && !adoptShimContext()) return false;
    if (!s.inited || !s.supported) return false;
    if (in.sceneImg == VK_NULL_HANDLE || in.velImg == VK_NULL_HANDLE ||
        in.depthImg == VK_NULL_HANDLE || !in.w || !in.h) return false;
    if (!ensureFeatureFns()) return false;

    if (!ensureOutput(dev, in.w, in.h, in.sceneFmt,
                      dd.createImage, dd.allocateMemory, dd.bindImageMemory,
                      dd.getImageMemReq, dd.createImageView,
                      dd.destroyImage, dd.freeMemory, dd.destroyImageView,
                      phys)) {
        trace("DLSSD: output image alloc failed"); return false;
    }

    const sl::ViewportHandle vp(0u);

    // ---- options: DLAA (no upscale) as the honest raster mode.
    if (!s.optionsSet) {
        sl::DLSSDOptions opt{};
        opt.mode = sl::DLSSMode::eDLAA;
        opt.outputWidth = in.w; opt.outputHeight = in.h;
        opt.colorBuffersHDR = sl::Boolean::eTrue;
        opt.normalRoughnessMode = sl::DLSSDNormalRoughnessMode::eUnpacked;
        sl::Result orr = s.DLSSDSetOptions(vp, opt);
        trace("DLSSD: slDLSSDSetOptions(DLAA %ux%u) -> %s", in.w, in.h, rc(orr));
        if (orr != sl::Result::eOk) return false;
        s.optionsSet = true;
    }

    // ---- frame token
    sl::FrameToken *tok = nullptr;
    uint32_t fi = s.frame++;
    if (s.GetNewFrameToken(tok, &fi) != sl::Result::eOk || !tok) return false;

    // ---- constants
    sl::Constants c{};
    row(c.cameraViewToClip, in.proj);
    row(c.clipToPrevClip, in.reproj);
    c.jitterOffset = { in.jitterX, in.jitterY };
    c.mvecScale = { 1.0f, 1.0f };            // our vectors are in UV space already
    c.depthInverted = in.depthInverted ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    c.cameraMotionIncluded = sl::Boolean::eTrue;
    c.motionVectors3D = sl::Boolean::eFalse;
    c.reset = sl::Boolean::eFalse;
    c.orthographicProjection = sl::Boolean::eFalse;
    c.motionVectorsDilated = sl::Boolean::eFalse;
    c.motionVectorsJittered = sl::Boolean::eFalse;
    sl::Result cr = s.SetConstants(c, *tok, vp);
    if (cr != sl::Result::eOk) { trace("DLSSD: slSetConstants -> %s", rc(cr)); return false; }

    // ---- tags (what we can honestly provide on a raster pipeline)
    sl::Resource colIn  = mkRes(in.sceneImg, in.sceneView, in.sceneFmt, in.sceneLayout, in.w, in.h);
    sl::Resource depth  = mkRes(in.depthImg, in.depthView, in.depthFmt, in.depthLayout, in.w, in.h);
    sl::Resource mvec   = mkRes(in.velImg,   in.velView,   in.velFmt,   in.velLayout,   in.w, in.h);
    sl::Resource outCol = mkRes(s.outImg, s.outView, s.outFmt, VK_IMAGE_LAYOUT_GENERAL, in.w, in.h);

    std::vector<sl::ResourceTag> tags;
    tags.push_back(sl::ResourceTag(&colIn,  sl::kBufferTypeScalingInputColor,  sl::ResourceLifecycle::eValidUntilEvaluate));
    tags.push_back(sl::ResourceTag(&depth,  sl::kBufferTypeDepth,              sl::ResourceLifecycle::eValidUntilEvaluate));
    tags.push_back(sl::ResourceTag(&mvec,   sl::kBufferTypeMotionVectors,      sl::ResourceLifecycle::eValidUntilEvaluate));
    tags.push_back(sl::ResourceTag(&outCol, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eValidUntilEvaluate));
    if (ensureGuides(dev, dd, phys, in.cb, in.w, in.h)) {
        static sl::Resource alb, spec;
        alb  = mkRes(s.gAlbImg,  s.gAlbView,  VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_GENERAL, in.w, in.h);
        spec = mkRes(s.gSpecImg, s.gSpecView, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_GENERAL, in.w, in.h);
        tags.push_back(sl::ResourceTag(&alb,  sl::kBufferTypeAlbedo,         sl::ResourceLifecycle::eValidUntilEvaluate));
        tags.push_back(sl::ResourceTag(&spec, sl::kBufferTypeSpecularAlbedo, sl::ResourceLifecycle::eValidUntilEvaluate));
        static sl::Resource rgh;
        rgh = mkRes(s.gRghImg, s.gRghView, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_GENERAL, in.w, in.h);
        tags.push_back(sl::ResourceTag(&rgh, sl::kBufferTypeRoughness, sl::ResourceLifecycle::eValidUntilEvaluate));
    }
    if (in.normImg != VK_NULL_HANDLE) {
        static sl::Resource norm; norm = mkRes(in.normImg, in.normView, in.normFmt, in.normLayout, in.w, in.h);
        tags.push_back(sl::ResourceTag(&norm, sl::kBufferTypeNormals, sl::ResourceLifecycle::eValidUntilEvaluate));
    }
    sl::Result tr = s.SetTag(vp, tags.data(), (uint32_t)tags.size(), (sl::CommandBuffer*)in.cb);
    if (tr != sl::Result::eOk) { trace("DLSSD: slSetTag(%zu) -> %s", tags.size(), rc(tr)); return false; }

    // ---- evaluate
    const sl::BaseStructure *inputs[1] = { &vp };
    sl::Result er = s.EvaluateFeature(kRR, *tok, inputs, 1, (sl::CommandBuffer*)in.cb);
    if (er != sl::Result::eOk) { trace("DLSSD: slEvaluateFeature(1001) -> %s", rc(er)); return false; }

    // ---- copy RR output back over the scene image so present shows it.
    VkImageMemoryBarrier b[2]{};
    for (int i = 0; i < 2; ++i) {
        b[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b[i].srcQueueFamilyIndex = b[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[i].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    }
    b[0].image = s.outImg; b[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL; b[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; b[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b[1].image = in.sceneImg; b[1].oldLayout = in.sceneLayout; b[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT; b[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    dd.cmdPipelineBarrier(in.cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          0, 0, nullptr, 0, nullptr, 2, b);

    VkImageCopy cp{}; cp.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    cp.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }; cp.extent = { in.w, in.h, 1 };
    dd.cmdCopyImage(in.cb, s.outImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    in.sceneImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);

    // restore layouts
    b[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; b[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT; b[0].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b[1].newLayout = in.sceneLayout;
    b[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    dd.cmdPipelineBarrier(in.cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                          0, 0, nullptr, 0, nullptr, 2, b);

    if (!s.announced) {
        s.announced = true;
        trace("DLSSD: Ray Reconstruction RUNNING - DLAA %ux%u, tags=color/depth/mvec/"
              "normals/albedo/specular/roughness/out. Albedo, specular and roughness are "
              "CONSTANT guides (X-Plane harvests none); judge the image on screen.", in.w, in.h);
    }
    return true;
}

} // namespace mvdlssd
