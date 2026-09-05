// mv_sl_probe.h - THROWAWAY SPIKE. Not part of the product.
//
// Question: can this layer drive DLSS-NR the SUPPORTED way - through NVIDIA's
// Streamline runtime (sl.interposer.dll + sl.common.dll + the sl.dlss_nr.dll
// feature plugin, id 1004) - instead of loading the raw NGX snippet, which
// refuses every call that does not come from nvngx.dll ("Not called from NGX
// runtime", see mv_nr_probe.h)?
//
// Streamline's manual-hooking Vulkan path is built for exactly our situation:
// we already own the VkInstance / VkPhysicalDevice / VkDevice, so we tell
// Streamline about them with slSetVulkanInfo and it never creates or proxies a
// device of its own. The dlss_nr plugin declares no API hooks (its JSON:
// "hooks":[]), so nothing has to intercept the frame - it is a pure
// evaluate-driven feature.
//
// Armed by TAA_SL_PROBE=1. The Streamline binaries come from TAA_SL_DIR
// (default: <X-Plane>\MotionVectors\streamline). That directory must contain,
// side by side:
//     sl.interposer.dll   (the public Streamline SDK, github.com/NVIDIA-RTX)
//     sl.common.dll       (same)
//     sl.dlss_nr.dll      (the feature plugin, id 1004)
//     nvngx_dlssnr.dll    (the NGX snippet common loads for feature 18)
// The interposer and common plugin are NOT in the 1-Click package; the user
// supplies them. Everything learned goes to the layer trace under "SL PROBE".
// It never touches the frame.
//
// Two phases, matching the NGX probe:
//   1. slProbeDeviceExtensions() - after slInit, before vkCreateDevice: ask
//      slGetFeatureRequirements(1004) which device extensions and queues the
//      feature needs, add the device ones the driver offers to the create list.
//   2. slProbeAfterDevice()      - after vkCreateDevice: slSetVulkanInfo with
//      our device, then slIsFeatureSupported(1004) on our adapter. That is the
//      viability answer; CreateFeature/Evaluate need real tagged resources and
//      are the next spike, not this one.
#pragma once
#include <windows.h>
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

// The Streamline public SDK headers (MIT), vendored under third_party.
// vulkan.h must precede sl_helpers_vk.h, which it does above.
#include "sl.h"
#include "sl_helpers_vk.h"

// id 1004 = sl.dlss_nr. NOT in the public SDK (its features stop at 1003,
// DirectSR); the plugin's own embedded JSON manifest is the source.
static const sl::Feature kFeatureDLSS_NR = (sl::Feature)1004;

// ---- the interposer's C exports, resolved by name from sl.interposer.dll.
typedef sl::Result (*PFN_slInit)(const sl::Preferences &, uint64_t);
typedef sl::Result (*PFN_slShutdown)();
typedef sl::Result (*PFN_slSetVulkanInfo)(const sl::VulkanInfo &);
typedef sl::Result (*PFN_slIsFeatureSupported)(sl::Feature, const sl::AdapterInfo &);
typedef sl::Result (*PFN_slIsFeatureLoaded)(sl::Feature, bool &);
typedef sl::Result (*PFN_slSetFeatureLoaded)(sl::Feature, bool);
typedef sl::Result (*PFN_slGetFeatureRequirements)(sl::Feature, sl::FeatureRequirements &);
typedef sl::Result (*PFN_slGetFeatureVersion)(sl::Feature, sl::FeatureVersion &);

struct SlRuntime {
    HMODULE lib = nullptr;
    PFN_slInit                   Init = nullptr;
    PFN_slShutdown               Shutdown = nullptr;
    PFN_slSetVulkanInfo          SetVulkanInfo = nullptr;
    PFN_slIsFeatureSupported     IsFeatureSupported = nullptr;
    PFN_slIsFeatureLoaded        IsFeatureLoaded = nullptr;
    PFN_slSetFeatureLoaded       SetFeatureLoaded = nullptr;
    PFN_slGetFeatureRequirements GetFeatureRequirements = nullptr;
    PFN_slGetFeatureVersion      GetFeatureVersion = nullptr;
    std::wstring dir;
    bool inited = false;
};
static SlRuntime g_sl;
static std::vector<std::string> g_slExtStore;   // storage for names added to the device
static std::wstring g_slLogPath;

static bool slProbeEnabled() { static const bool on = getenv("TAA_SL_PROBE") != nullptr; return on; }

static const char *slResultName(sl::Result r)
{
    switch (r) {
        case sl::Result::eOk: return "eOk";
        case sl::Result::eErrorIO: return "eErrorIO";
        case sl::Result::eErrorDriverOutOfDate: return "eErrorDriverOutOfDate";
        case sl::Result::eErrorOSOutOfDate: return "eErrorOSOutOfDate";
        case sl::Result::eErrorDeviceNotCreated: return "eErrorDeviceNotCreated";
        case sl::Result::eErrorNoSupportedAdapterFound: return "eErrorNoSupportedAdapterFound";
        case sl::Result::eErrorAdapterNotSupported: return "eErrorAdapterNotSupported";
        case sl::Result::eErrorNoPlugins: return "eErrorNoPlugins";
        case sl::Result::eErrorVulkanAPI: return "eErrorVulkanAPI";
        case sl::Result::eErrorMissingOrInvalidAPI: return "eErrorMissingOrInvalidAPI";
        case sl::Result::eErrorFeatureMissing: return "eErrorFeatureMissing";
        case sl::Result::eErrorFeatureNotSupported: return "eErrorFeatureNotSupported";
        case sl::Result::eErrorNotInitialized: return "eErrorNotInitialized";
        default: return "eError(other)";
    }
}

static void NVSDK_slLog(sl::LogType type, const char *msg)
{
    std::string s = msg ? msg : "";
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    trace("SL PROBE [sl log %d] %s", (int)type, s.c_str());
}

static bool slLoad()
{
    if (g_sl.inited) return true;
    if (g_sl.lib) return false;   // tried and failed to init once

    wchar_t base[MAX_PATH * 2];
    const char *env = getenv("TAA_SL_DIR");
    if (env && *env) MultiByteToWideChar(CP_UTF8, 0, env, -1, base, MAX_PATH * 2);
    else {
        GetModuleFileNameW(nullptr, base, MAX_PATH * 2);
        std::wstring p = base; p = p.substr(0, p.find_last_of(L"\\/")) + L"\\MotionVectors\\streamline";
        wcscpy(base, p.c_str());
    }
    g_sl.dir = base;
    std::wstring interposer = g_sl.dir + L"\\sl.interposer.dll";
    trace("SL PROBE: interposer dir %ls", g_sl.dir.c_str());

    // AddDllDirectory so the interposer finds sl.common.dll and the plugins
    // beside it without polluting the process search path permanently.
    SetDllDirectoryW(g_sl.dir.c_str());
    g_sl.lib = LoadLibraryW(interposer.c_str());
    SetDllDirectoryW(nullptr);
    if (!g_sl.lib) {
        trace("SL PROBE: LoadLibrary sl.interposer.dll failed, error %lu. "
              "Place sl.interposer.dll + sl.common.dll + sl.dlss_nr.dll + "
              "nvngx_dlssnr.dll in %ls (interposer/common from the public "
              "NVIDIA Streamline SDK release).", GetLastError(), g_sl.dir.c_str());
        return false;
    }
#define SLGET(field, pfn, name) g_sl.field = (pfn)(void*)GetProcAddress(g_sl.lib, name); \
    trace("SL PROBE: export %-28s %s", name, g_sl.field ? "ok" : "MISSING")
    SLGET(Init, PFN_slInit, "slInit");
    SLGET(Shutdown, PFN_slShutdown, "slShutdown");
    SLGET(SetVulkanInfo, PFN_slSetVulkanInfo, "slSetVulkanInfo");
    SLGET(IsFeatureSupported, PFN_slIsFeatureSupported, "slIsFeatureSupported");
    SLGET(IsFeatureLoaded, PFN_slIsFeatureLoaded, "slIsFeatureLoaded");
    SLGET(SetFeatureLoaded, PFN_slSetFeatureLoaded, "slSetFeatureLoaded");
    SLGET(GetFeatureRequirements, PFN_slGetFeatureRequirements, "slGetFeatureRequirements");
    SLGET(GetFeatureVersion, PFN_slGetFeatureVersion, "slGetFeatureVersion");
#undef SLGET
    if (!g_sl.Init) { trace("SL PROBE: no slInit - wrong or corrupt interposer"); return false; }

    // logs + OTA data path
    { wchar_t tmp[MAX_PATH]; GetTempPathW(MAX_PATH, tmp); g_slLogPath = std::wstring(tmp) + L"mv_sl_probe"; CreateDirectoryW(g_slLogPath.c_str(), nullptr); }

    static const wchar_t *paths[1]; paths[0] = g_sl.dir.c_str();
    static sl::Feature feats[1]; feats[0] = kFeatureDLSS_NR;

    sl::Preferences pref{};
    pref.showConsole = false;
    pref.logLevel = sl::LogLevel::eVerbose;
    pref.logMessageCallback = NVSDK_slLog;
    pref.pathsToPlugins = paths;
    pref.numPathsToPlugins = 1;
    pref.pathToLogsAndData = g_slLogPath.c_str();
    // Manual hooking: WE own the Vulkan device; Streamline must not proxy
    // vkCreateDevice/Instance. Only load the plugin we shipped, no OTA.
    pref.flags = sl::PreferenceFlags::eUseManualHooking | sl::PreferenceFlags::eDisableCLStateTracking;
    pref.featuresToLoad = feats;
    pref.numFeaturesToLoad = 1;
    pref.applicationId = 0x5A1D0001u;
    pref.engine = sl::EngineType::eCustom;
    pref.renderAPI = sl::RenderAPI::eVulkan;

    sl::Result r = g_sl.Init(pref, sl::kSDKVersion);
    trace("SL PROBE: slInit(renderAPI=Vulkan, manual hooking, plugin=%ls) -> %s",
          g_sl.dir.c_str(), slResultName(r));
    if (r != sl::Result::eOk) return false;
    g_sl.inited = true;

    if (g_sl.IsFeatureLoaded) {
        bool loaded = false; sl::Result lr = g_sl.IsFeatureLoaded(kFeatureDLSS_NR, loaded);
        trace("SL PROBE: slIsFeatureLoaded(1004) -> %s, loaded=%d", slResultName(lr), (int)loaded);
    }
    if (g_sl.GetFeatureVersion) {
        sl::FeatureVersion v{}; sl::Result vr = g_sl.GetFeatureVersion(kFeatureDLSS_NR, v);
        trace("SL PROBE: slGetFeatureVersion(1004) -> %s, sl %u.%u.%u ngx %u.%u.%u",
              slResultName(vr), v.versionSL.major, v.versionSL.minor, v.versionSL.build,
              v.versionNGX.major, v.versionNGX.minor, v.versionNGX.build);
    }
    return true;
}

// Phase 1: after slInit, before vkCreateDevice.
static void slProbeDeviceExtensions(VkInstance inst, VkPhysicalDevice phys,
                                    const std::vector<VkExtensionProperties> &have,
                                    std::vector<const char*> &exts)
{
    (void)inst;
    if (!slLoad()) return;
    if (!g_sl.GetFeatureRequirements) { trace("SL PROBE: no slGetFeatureRequirements"); return; }

    sl::FeatureRequirements req{};
    sl::Result r = g_sl.GetFeatureRequirements(kFeatureDLSS_NR, req);
    trace("SL PROBE: slGetFeatureRequirements(1004) -> %s | flags 0x%x vkCompute=%u vkGfx=%u "
          "vkOpticalFlow=%u devExt=%u instExt=%u f12=%u f13=%u numTags=%u",
          slResultName(r), (unsigned)req.flags, req.vkNumComputeQueuesRequired,
          req.vkNumGraphicsQueuesRequired, req.vkNumOpticalFlowQueuesRequired,
          req.vkNumDeviceExtensions, req.vkNumInstanceExtensions,
          req.vkNumFeatures12, req.vkNumFeatures13, req.numRequiredTags);
    if (r != sl::Result::eOk) return;

    for (uint32_t i = 0; i < req.vkNumInstanceExtensions && req.vkInstanceExtensions; ++i)
        trace("SL PROBE:   instance ext %s", req.vkInstanceExtensions[i]);
    for (uint32_t i = 0; i < req.vkNumDeviceExtensions && req.vkDeviceExtensions; ++i) {
        const char *name = req.vkDeviceExtensions[i];
        bool offered = false, already = false;
        for (const VkExtensionProperties &h : have) if (!strcmp(h.extensionName, name)) offered = true;
        for (const char *e : exts) if (!strcmp(e, name)) already = true;
        if (offered && !already) g_slExtStore.push_back(name);
        trace("SL PROBE:   device ext %-44s driver offers: %s%s", name, offered ? "yes" : "NO",
              already ? " (already requested)" : (offered ? " (added)" : ""));
    }
    for (const std::string &s : g_slExtStore) exts.push_back(s.c_str());
    for (uint32_t i = 0; i < req.vkNumFeatures12 && req.vkFeatures12; ++i)
        trace("SL PROBE:   Vk1.2 feature %s", req.vkFeatures12[i]);
    for (uint32_t i = 0; i < req.vkNumFeatures13 && req.vkFeatures13; ++i)
        trace("SL PROBE:   Vk1.3 feature %s", req.vkFeatures13[i]);
    for (uint32_t i = 0; i < req.numRequiredTags && req.requiredTags; ++i)
        trace("SL PROBE:   required tag (BufferType) %u", (unsigned)req.requiredTags[i]);
}

// Phase 2: after vkCreateDevice.
static void slProbeAfterDevice(VkInstance inst, VkPhysicalDevice phys, VkDevice dev,
                               uint32_t queueFamily)
{
    if (!g_sl.inited) return;

    if (g_sl.SetVulkanInfo) {
        sl::VulkanInfo vi{};
        vi.device = dev;
        vi.instance = inst;
        vi.physicalDevice = phys;
        vi.computeQueueFamily = queueFamily;
        vi.computeQueueIndex = 0;
        vi.graphicsQueueFamily = queueFamily;
        vi.graphicsQueueIndex = 0;
        sl::Result r = g_sl.SetVulkanInfo(vi);
        trace("SL PROBE: slSetVulkanInfo(dev=%p inst=%p phys=%p family=%u) -> %s",
              (void*)dev, (void*)inst, (void*)phys, queueFamily, slResultName(r));
        if (r != sl::Result::eOk) return;
    }

    if (g_sl.IsFeatureSupported) {
        // AdapterInfo by Vulkan physical device (the LUID path is for D3D).
        sl::AdapterInfo ai{};
        ai.vkPhysicalDevice = phys;
        sl::Result r = g_sl.IsFeatureSupported(kFeatureDLSS_NR, ai);
        trace("SL PROBE: slIsFeatureSupported(1004, this adapter) -> %s  <== VIABILITY",
              slResultName(r));
    }
    trace("SL PROBE: done. Runtime stays initialised; nothing else touches it. "
          "Next spike: slSetTag (color/depth/mvec/output) + slSetConstants + "
          "slEvaluateFeature(1004) on a scratch frame.");
}
