// mv_nr_probe.h - THROWAWAY SPIKE. Not part of the product.
//
// Question: can this layer load NVIDIA's DLSS-NR snippet (nvngx_dlssnr.dll,
// NGX feature 18) directly on X-Plane's Vulkan device and create the feature
// inline - the way the RenoDX add-on does on D3D12 - with no host process and
// no dependency on the driver's NGX feature table (which, on driver 610.88,
// does not know feature 18)?
//
// Armed by TAA_NR_PROBE=1. The snippet path comes from TAA_NR_DLL (default:
// <X-Plane>\MotionVectors\nvngx_dlssnr.dll). Everything it learns goes to the
// layer trace under "NR PROBE". It never touches the frame.
//
// Two phases, because extensions must be enabled at device creation:
//   1. nrProbeDeviceExtensions()  - before vkCreateDevice: ask the snippet
//      which instance/device extensions it needs, add the device ones the
//      driver offers to the create list, log the rest.
//   2. nrProbeAfterDevice()       - after vkCreateDevice: Init_Ext2 with a
//      logging callback, a map-backed NVSDK_NGX_Parameter (the snippet exports
//      no Vulkan parameter allocator), PopulateParameters_Impl to read its
//      defaults, GetScratchBufferSize, then ONE CreateFeature1 on a scratch
//      command buffer, submitted and waited. Release, done.
#pragma once
#include <windows.h>
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "nvsdk_ngx_vk.h"

// ---- tiny SHA-256, so the trace names the exact binary that was loaded.
namespace nrsha {
static inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static std::string sha256(const std::vector<uint8_t> &in)
{
    static const uint32_t K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::vector<uint8_t> m = in; uint64_t bits = (uint64_t)in.size() * 8;
    m.push_back(0x80); while ((m.size() % 64) != 56) m.push_back(0);
    for (int i = 7; i >= 0; --i) m.push_back((uint8_t)(bits >> (i * 8)));
    for (size_t off = 0; off < m.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) w[i] = (m[off+i*4]<<24)|(m[off+i*4+1]<<16)|(m[off+i*4+2]<<8)|m[off+i*4+3];
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3), s1 = rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e,6)^rotr(e,11)^rotr(e,25), ch = (e&f)^(~e&g), t1 = hh+S1+ch+K[i]+w[i];
            uint32_t S0 = rotr(a,2)^rotr(a,13)^rotr(a,22), mj = (a&b)^(a&c)^(b&c), t2 = S0+mj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    char out[65]; for (int i = 0; i < 8; ++i) snprintf(out + i*8, 9, "%08x", h[i]); return out;
}
} // namespace nrsha

// ---- a map-backed NVSDK_NGX_Parameter. 17 virtuals, in header order.
struct NrParams : public NVSDK_NGX_Parameter {
    struct V { int kind = 0; unsigned long long u64 = 0; float f = 0; double d = 0; unsigned int u = 0; int i = 0; void *p = nullptr; };
    std::map<std::string, V> m;
    std::vector<std::string> setLog;
    template <class T> void put(const char *n, int kind, T v) {
        V &x = m[n]; x.kind = kind;
        if (kind == 1) x.u64 = (unsigned long long)v; else if (kind == 2) x.f = (float)v; else if (kind == 3) x.d = (double)v;
        else if (kind == 4) x.u = (unsigned int)v; else if (kind == 5) x.i = (int)v;
        char b[160]; snprintf(b, sizeof b, "%s=%s", n, describe(x).c_str()); setLog.push_back(b);
    }
    static std::string describe(const V &x) {
        char b[64];
        switch (x.kind) { case 1: snprintf(b, sizeof b, "%llu", x.u64); break; case 2: snprintf(b, sizeof b, "%g", x.f); break;
            case 3: snprintf(b, sizeof b, "%g", x.d); break; case 4: snprintf(b, sizeof b, "%u", x.u); break;
            case 5: snprintf(b, sizeof b, "%d", x.i); break; default: snprintf(b, sizeof b, "ptr %p", x.p); }
        return b;
    }
    void Set(const char *n, unsigned long long v) override { put(n, 1, v); }
    void Set(const char *n, float v) override { put(n, 2, v); }
    void Set(const char *n, double v) override { put(n, 3, v); }
    void Set(const char *n, unsigned int v) override { put(n, 4, v); }
    void Set(const char *n, int v) override { put(n, 5, v); }
    void Set(const char *n, ID3D11Resource *v) override { V &x = m[n]; x.kind = 6; x.p = v; }
    void Set(const char *n, ID3D12Resource *v) override { V &x = m[n]; x.kind = 7; x.p = v; }
    void Set(const char *n, void *v) override { V &x = m[n]; x.kind = 8; x.p = v; char b[160]; snprintf(b, sizeof b, "%s=ptr %p", n, v); setLog.push_back(b); }
    NVSDK_NGX_Result Get(const char *n, unsigned long long *o) const override { auto it = m.find(n); if (it == m.end()) return NVSDK_NGX_Result_FAIL_InvalidParameter; *o = it->second.kind == 1 ? it->second.u64 : (unsigned long long)it->second.u; return NVSDK_NGX_Result_Success; }
    NVSDK_NGX_Result Get(const char *n, float *o) const override { auto it = m.find(n); if (it == m.end()) return NVSDK_NGX_Result_FAIL_InvalidParameter; *o = it->second.kind == 2 ? it->second.f : (float)it->second.d; return NVSDK_NGX_Result_Success; }
    NVSDK_NGX_Result Get(const char *n, double *o) const override { auto it = m.find(n); if (it == m.end()) return NVSDK_NGX_Result_FAIL_InvalidParameter; *o = it->second.kind == 3 ? it->second.d : (double)it->second.f; return NVSDK_NGX_Result_Success; }
    NVSDK_NGX_Result Get(const char *n, unsigned int *o) const override { auto it = m.find(n); if (it == m.end()) return NVSDK_NGX_Result_FAIL_InvalidParameter; *o = it->second.kind == 4 ? it->second.u : (unsigned int)it->second.i; return NVSDK_NGX_Result_Success; }
    NVSDK_NGX_Result Get(const char *n, int *o) const override { auto it = m.find(n); if (it == m.end()) return NVSDK_NGX_Result_FAIL_InvalidParameter; *o = it->second.kind == 5 ? it->second.i : (int)it->second.u; return NVSDK_NGX_Result_Success; }
    NVSDK_NGX_Result Get(const char *n, ID3D11Resource **o) const override { auto it = m.find(n); if (it == m.end()) return NVSDK_NGX_Result_FAIL_InvalidParameter; *o = (ID3D11Resource*)it->second.p; return NVSDK_NGX_Result_Success; }
    NVSDK_NGX_Result Get(const char *n, ID3D12Resource **o) const override { auto it = m.find(n); if (it == m.end()) return NVSDK_NGX_Result_FAIL_InvalidParameter; *o = (ID3D12Resource*)it->second.p; return NVSDK_NGX_Result_Success; }
    NVSDK_NGX_Result Get(const char *n, void **o) const override { auto it = m.find(n); if (it == m.end()) return NVSDK_NGX_Result_FAIL_InvalidParameter; *o = it->second.p; return NVSDK_NGX_Result_Success; }
    void Reset() override { m.clear(); }
};

// ---- the snippet's own Vulkan exports, resolved by name.
// Explicit typedefs, not decltype of the SDK declarations: the header hides
// some of these behind version guards, and the snippet's exports follow the
// app-side prototypes read from nvsdk_ngx_vk.h.
typedef NVSDK_NGX_Result (NVSDK_CONV *PFN_NrInitExt2)(unsigned long long, const wchar_t *, VkInstance, VkPhysicalDevice, VkDevice,
                                                      PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr, const NVSDK_NGX_FeatureCommonInfo *, NVSDK_NGX_Version);
typedef NVSDK_NGX_Result (NVSDK_CONV *PFN_NrShutdown1)(VkDevice);
typedef NVSDK_NGX_Result (NVSDK_CONV *PFN_NrGetFeatureRequirements)(VkInstance, VkPhysicalDevice, const NVSDK_NGX_FeatureDiscoveryInfo *, NVSDK_NGX_FeatureRequirement *);
typedef NVSDK_NGX_Result (NVSDK_CONV *PFN_NrGetInstanceExt)(const NVSDK_NGX_FeatureDiscoveryInfo *, uint32_t *, VkExtensionProperties **);
typedef NVSDK_NGX_Result (NVSDK_CONV *PFN_NrGetDeviceExt)(VkInstance, VkPhysicalDevice, const NVSDK_NGX_FeatureDiscoveryInfo *, uint32_t *, VkExtensionProperties **);
typedef NVSDK_NGX_Result (NVSDK_CONV *PFN_NrGetScratchBufferSize)(NVSDK_NGX_Feature, const NVSDK_NGX_Parameter *, size_t *);
typedef NVSDK_NGX_Result (NVSDK_CONV *PFN_NrCreateFeature1)(VkDevice, VkCommandBuffer, NVSDK_NGX_Feature, NVSDK_NGX_Parameter *, NVSDK_NGX_Handle **);
typedef NVSDK_NGX_Result (NVSDK_CONV *PFN_NrReleaseFeature)(NVSDK_NGX_Handle *);
typedef NVSDK_NGX_Result (NVSDK_CONV *PFN_NrPopulateParameters)(NVSDK_NGX_Parameter *);
typedef unsigned long long (NVSDK_CONV *PFN_NrGetU64)();
struct NrSnippet {
    HMODULE lib = nullptr;
    PFN_NrInitExt2               Init_Ext2 = nullptr;
    PFN_NrShutdown1              Shutdown1 = nullptr;
    PFN_NrGetFeatureRequirements GetFeatureRequirements = nullptr;
    PFN_NrGetInstanceExt         GetInstanceExt = nullptr;
    PFN_NrGetDeviceExt           GetDeviceExt = nullptr;
    PFN_NrGetScratchBufferSize   GetScratchBufferSize = nullptr;
    PFN_NrCreateFeature1         CreateFeature1 = nullptr;
    PFN_NrReleaseFeature         ReleaseFeature = nullptr;
    PFN_NrPopulateParameters     PopulateParameters_Impl = nullptr;
    PFN_NrGetU64                 GetSnippetVersion = nullptr;
    PFN_NrGetU64                 GetAPIVersion = nullptr;
    std::wstring path;
    std::string sha;
};
static NrSnippet g_nr;
static std::vector<std::string> g_nrExtStore;   // storage for extension names we add
static std::wstring g_nrDataPath;

static bool nrProbeEnabled() { static const bool on = getenv("TAA_NR_PROBE") != nullptr; return on; }

static void NVSDK_CONV nrLogCb(const char *msg, NVSDK_NGX_Logging_Level lvl, NVSDK_NGX_Feature src)
{
    // strip the trailing newline NGX tends to send
    std::string s = msg ? msg : ""; while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    trace("NR PROBE [ngx log lvl %d feat %d] %s", (int)lvl, (int)src, s.c_str());
}

static const NVSDK_NGX_FeatureCommonInfo *nrCommonInfo()
{
    static NVSDK_NGX_FeatureCommonInfo ci;
    static const wchar_t *paths[1];
    static bool init = false;
    if (!init) {
        init = true;
        memset(&ci, 0, sizeof(ci));
        static std::wstring dir; dir = g_nr.path.substr(0, g_nr.path.find_last_of(L"\\/"));
        paths[0] = dir.c_str();
        ci.PathListInfo.Path = paths; ci.PathListInfo.Length = 1;
        ci.LoggingInfo.LoggingCallback = nrLogCb;
        ci.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_VERBOSE;
        ci.LoggingInfo.DisableOtherLoggingSinks = false;
    }
    return &ci;
}

static const unsigned long long kNrAppId = 0x5A1D0001ull;   // arbitrary non-zero
static const NVSDK_NGX_Feature   kNrFeature = (NVSDK_NGX_Feature)18;

static NVSDK_NGX_FeatureDiscoveryInfo nrDiscovery()
{
    NVSDK_NGX_FeatureDiscoveryInfo d; memset(&d, 0, sizeof(d));
    d.SDKVersion = NVSDK_NGX_Version_API;
    d.FeatureID  = kNrFeature;
    d.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Application_Id;
    d.Identifier.v.ApplicationId = kNrAppId;
    d.ApplicationDataPath = g_nrDataPath.c_str();
    d.FeatureInfo = nrCommonInfo();
    return d;
}

static bool nrLoad()
{
    if (g_nr.lib) return true;
    wchar_t buf[MAX_PATH * 2];
    const char *env = getenv("TAA_NR_DLL");
    if (env && *env) { MultiByteToWideChar(CP_UTF8, 0, env, -1, buf, MAX_PATH * 2); }
    else {
        // default: <exe dir>\MotionVectors\nvngx_dlssnr.dll
        GetModuleFileNameW(nullptr, buf, MAX_PATH * 2);
        std::wstring p = buf; p = p.substr(0, p.find_last_of(L"\\/")) + L"\\MotionVectors\\nvngx_dlssnr.dll";
        wcscpy(buf, p.c_str());
    }
    g_nr.path = buf;
    {
        wchar_t tmp[MAX_PATH]; GetTempPathW(MAX_PATH, tmp); g_nrDataPath = std::wstring(tmp) + L"mv_nr_probe";
        CreateDirectoryW(g_nrDataPath.c_str(), nullptr);
    }
    // hash it first, so the trace always names the exact binary
    {
        FILE *f = _wfopen(buf, L"rb");
        if (!f) { trace("NR PROBE: cannot open %ls", buf); return false; }
        std::vector<uint8_t> data; fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        data.resize(n > 0 ? (size_t)n : 0); if (n > 0) fread(data.data(), 1, (size_t)n, f); fclose(f);
        g_nr.sha = nrsha::sha256(data);
        trace("NR PROBE: %ls  %ld bytes  sha256 %s", buf, n, g_nr.sha.c_str());
    }
    g_nr.lib = LoadLibraryW(buf);
    if (!g_nr.lib) { trace("NR PROBE: LoadLibraryW failed, error %lu", GetLastError()); return false; }
#define NRGET(field, name) g_nr.field = (decltype(g_nr.field))(void*)GetProcAddress(g_nr.lib, name); \
    trace("NR PROBE: export %-52s %s", name, g_nr.field ? "ok" : "MISSING")
    NRGET(Init_Ext2, "NVSDK_NGX_VULKAN_Init_Ext2");
    NRGET(Shutdown1, "NVSDK_NGX_VULKAN_Shutdown1");
    NRGET(GetFeatureRequirements, "NVSDK_NGX_VULKAN_GetFeatureRequirements");
    NRGET(GetInstanceExt, "NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements");
    NRGET(GetDeviceExt, "NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements");
    NRGET(GetScratchBufferSize, "NVSDK_NGX_VULKAN_GetScratchBufferSize");
    NRGET(CreateFeature1, "NVSDK_NGX_VULKAN_CreateFeature1");
    NRGET(ReleaseFeature, "NVSDK_NGX_VULKAN_ReleaseFeature");
    NRGET(PopulateParameters_Impl, "NVSDK_NGX_VULKAN_PopulateParameters_Impl");
    NRGET(GetSnippetVersion, "NVSDK_NGX_GetSnippetVersion");
    NRGET(GetAPIVersion, "NVSDK_NGX_GetAPIVersion");
#undef NRGET
    if (g_nr.GetSnippetVersion) trace("NR PROBE: snippet version 0x%llx", g_nr.GetSnippetVersion());
    if (g_nr.GetAPIVersion)     trace("NR PROBE: snippet API version 0x%llx (our headers 0x%x)", g_nr.GetAPIVersion(), (unsigned)NVSDK_NGX_Version_API);
    return true;
}

// Phase 1: before vkCreateDevice.
static void nrProbeDeviceExtensions(VkInstance inst, VkPhysicalDevice phys,
                                    const std::vector<VkExtensionProperties> &have,
                                    std::vector<const char*> &exts)
{
    if (!nrLoad()) return;
    NVSDK_NGX_FeatureDiscoveryInfo disc = nrDiscovery();
    if (g_nr.GetInstanceExt) {
        uint32_t n = 0; VkExtensionProperties *p = nullptr;
        NVSDK_NGX_Result r = g_nr.GetInstanceExt(&disc, &n, &p);
        trace("NR PROBE: instance extension requirements -> 0x%x, %u extensions", (unsigned)r, n);
        for (uint32_t i = 0; p && i < n; ++i) trace("NR PROBE:   instance ext %s (v%u)", p[i].extensionName, p[i].specVersion);
    }
    if (g_nr.GetDeviceExt) {
        uint32_t n = 0; VkExtensionProperties *p = nullptr;
        NVSDK_NGX_Result r = g_nr.GetDeviceExt(inst, phys, &disc, &n, &p);
        trace("NR PROBE: device extension requirements -> 0x%x, %u extensions", (unsigned)r, n);
        for (uint32_t i = 0; p && i < n; ++i) {
            bool offered = false, already = false;
            for (const VkExtensionProperties &h : have) if (!strcmp(h.extensionName, p[i].extensionName)) offered = true;
            for (const char *e : exts) if (!strcmp(e, p[i].extensionName)) already = true;
            if (offered && !already) { g_nrExtStore.push_back(p[i].extensionName); }
            trace("NR PROBE:   device ext %-44s driver offers: %s%s", p[i].extensionName, offered ? "yes" : "NO",
                  already ? " (already requested)" : (offered ? " (added)" : ""));
        }
        for (const std::string &s : g_nrExtStore) exts.push_back(s.c_str());
    }
    if (g_nr.GetFeatureRequirements) {
        NVSDK_NGX_FeatureRequirement req; memset(&req, 0, sizeof(req));
        NVSDK_NGX_Result r = g_nr.GetFeatureRequirements(inst, phys, &disc, &req);
        trace("NR PROBE: GetFeatureRequirements(18) -> 0x%x | supported=%d minHWArch=0x%x minOS=%s",
              (unsigned)r, (int)req.FeatureSupported, req.MinHWArchitecture, req.MinOSVersion);
    }
}

// Phase 2: after vkCreateDevice.
static void nrProbeAfterDevice(VkInstance inst, VkPhysicalDevice phys, VkDevice dev,
                               PFN_vkGetInstanceProcAddr gipa, PFN_vkGetDeviceProcAddr gdpa,
                               uint32_t queueFamily)
{
    if (!nrLoad() || !g_nr.Init_Ext2) return;
    trace("NR PROBE: Init_Ext2 (appId 0x%llx, data path %ls, family %u) ...", kNrAppId, g_nrDataPath.c_str(), queueFamily);
    NVSDK_NGX_Result r = g_nr.Init_Ext2(kNrAppId, g_nrDataPath.c_str(), inst, phys, dev, gipa, gdpa, nrCommonInfo(), NVSDK_NGX_Version_API);
    trace("NR PROBE: Init_Ext2 -> 0x%x (%s)", (unsigned)r, r == NVSDK_NGX_Result_Success ? "SUCCESS" : "fail");
    if (r != NVSDK_NGX_Result_Success) return;

    NrParams *params = new NrParams();
    if (g_nr.PopulateParameters_Impl) {
        NVSDK_NGX_Result pr = g_nr.PopulateParameters_Impl(params);
        trace("NR PROBE: PopulateParameters_Impl -> 0x%x, %zu keys set by the snippet:", (unsigned)pr, params->m.size());
        for (auto &kv : params->m) trace("NR PROBE:   default %-48s = %s", kv.first.c_str(), NrParams::describe(kv.second).c_str());
    }
    // A DLAA-shaped request at the sim's render size.
    const unsigned W = 2953, H = 1661;
    params->Set("DLSSNR.Width", W);  params->Set("DLSSNR.Height", H);
    params->Set("DLSSNR.InputWidth", W); params->Set("DLSSNR.InputHeight", H);
    params->Set("DLSSNR.ScalingRatio", 1.0f);
    params->Set("DLSSNR.Enabled", 1);
    params->Set("DLSSNR.DepthInverted", 0);
    params->Set("DLSSNR.Hint.Render.Preset", 0);
    params->Set("DLSSNR.MVecScaleX", 1.0f); params->Set("DLSSNR.MVecScaleY", 1.0f);
    params->Set("DLSS.Feature.Create.Flags", 0);
    if (g_nr.GetScratchBufferSize) {
        size_t sz = 0; NVSDK_NGX_Result sr = g_nr.GetScratchBufferSize(kNrFeature, params, &sz);
        trace("NR PROBE: GetScratchBufferSize(18) -> 0x%x, %zu bytes", (unsigned)sr, sz);
    }
    // scratch command buffer on the device's first queue family
    auto vkCreateCommandPool_ = (PFN_vkCreateCommandPool)gdpa(dev, "vkCreateCommandPool");
    auto vkAllocateCommandBuffers_ = (PFN_vkAllocateCommandBuffers)gdpa(dev, "vkAllocateCommandBuffers");
    auto vkBeginCommandBuffer_ = (PFN_vkBeginCommandBuffer)gdpa(dev, "vkBeginCommandBuffer");
    auto vkEndCommandBuffer_ = (PFN_vkEndCommandBuffer)gdpa(dev, "vkEndCommandBuffer");
    auto vkGetDeviceQueue_ = (PFN_vkGetDeviceQueue)gdpa(dev, "vkGetDeviceQueue");
    auto vkQueueSubmit_ = (PFN_vkQueueSubmit)gdpa(dev, "vkQueueSubmit");
    auto vkQueueWaitIdle_ = (PFN_vkQueueWaitIdle)gdpa(dev, "vkQueueWaitIdle");
    auto vkDestroyCommandPool_ = (PFN_vkDestroyCommandPool)gdpa(dev, "vkDestroyCommandPool");
    if (!vkCreateCommandPool_ || !vkAllocateCommandBuffers_ || !vkBeginCommandBuffer_ || !vkEndCommandBuffer_ || !vkGetDeviceQueue_ || !vkQueueSubmit_ || !vkQueueWaitIdle_) {
        trace("NR PROBE: device entry points missing - stopping before CreateFeature"); return;
    }
    VkCommandPoolCreateInfo pci; memset(&pci, 0, sizeof(pci)); pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT; pci.queueFamilyIndex = queueFamily;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool_(dev, &pci, nullptr, &pool) != VK_SUCCESS) { trace("NR PROBE: command pool failed"); return; }
    VkCommandBufferAllocateInfo ai; memset(&ai, 0, sizeof(ai)); ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cb = VK_NULL_HANDLE; vkAllocateCommandBuffers_(dev, &ai, &cb);
    VkCommandBufferBeginInfo bi; memset(&bi, 0, sizeof(bi)); bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; vkBeginCommandBuffer_(cb, &bi);

    NVSDK_NGX_Handle *handle = nullptr;
    trace("NR PROBE: CreateFeature1(18, %ux%u -> %ux%u) ...", W, H, W, H);
    NVSDK_NGX_Result cr = g_nr.CreateFeature1(dev, cb, kNrFeature, params, &handle);
    trace("NR PROBE: CreateFeature1 -> 0x%x (%s), handle %p", (unsigned)cr, cr == NVSDK_NGX_Result_Success ? "SUCCESS" : "fail", (void*)handle);
    for (const std::string &s : params->setLog) trace("NR PROBE:   param during create: %s", s.c_str());
    vkEndCommandBuffer_(cb);
    VkQueue q = VK_NULL_HANDLE; vkGetDeviceQueue_(dev, queueFamily, 0, &q);
    VkSubmitInfo si; memset(&si, 0, sizeof(si)); si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VkResult qr = vkQueueSubmit_(q, 1, &si, VK_NULL_HANDLE); VkResult wr = vkQueueWaitIdle_(q);
    trace("NR PROBE: submit %d, wait %d", (int)qr, (int)wr);
    // read back anything the snippet reported through the parameters
    { int v = 0; unsigned u = 0; float f = 0;
      if (params->Get("DLSSNR.FeatureInitResult", &v) == NVSDK_NGX_Result_Success) trace("NR PROBE: FeatureInitResult %d", v);
      if (params->Get("DLSSNR.NeedsUpdatedDriver", &u) == NVSDK_NGX_Result_Success) trace("NR PROBE: NeedsUpdatedDriver %u", u);
      if (params->Get("DLSSNR.ScalingRatio", &f) == NVSDK_NGX_Result_Success) trace("NR PROBE: ScalingRatio after create %g", f); }
    for (auto &kv : params->m) if (kv.first.rfind("DLSSNR.", 0) == 0 || kv.first[0] == '#') trace("NR PROBE:   after create %-48s = %s", kv.first.c_str(), NrParams::describe(kv.second).c_str());
    if (handle && g_nr.ReleaseFeature) { NVSDK_NGX_Result rr = g_nr.ReleaseFeature(handle); trace("NR PROBE: ReleaseFeature -> 0x%x", (unsigned)rr); }
    if (vkDestroyCommandPool_) vkDestroyCommandPool_(dev, pool, nullptr);
    trace("NR PROBE: done. The snippet stays loaded and initialised; nothing else touches it.");
}
