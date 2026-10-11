// mv_vkshim.cpp - MotionVectors Vulkan shim (installed as <X-Plane>\vulkan-1.dll)
//
// The OptiScaler/ReShade injection pattern for our mod. It IS vulkan-1.dll and:
//
//   1. Forwards the entire Vulkan API to vk_real_mv.dll, a private copy of the
//      real loader (259 exports forwarded by mv_vkshim.def; the loader entry
//      points implemented below). sl.interposer resolves its "vulkan-1.dll" to
//      us and, through our forwards, to the REAL functions.
//
//   2. Raises every VkInstance to Vulkan 1.3 (X-Plane asks 1.1) so the device
//      exposes the 1.3 entry points Streamline needs.
//
//   3. MERGES STREAMLINE'S DEVICE REQUIREMENTS into vkCreateDevice. This is the
//      part an SL-integrated game does by hand and X-Plane obviously does not:
//      slGetFeatureRequirements names the instance/device extensions, the 1.2
//      and 1.3 feature bits (privateData among them) and the extra compute /
//      graphics queues SL needs. Without them SL's chi::Vulkan::init throws and
//      slSetVulkanInfo returns 24 (eErrorExceptionHandler). We union SL's list
//      with X-Plane's, AND it against what the physical device actually
//      supports, and reserve SL's queues past X-Plane's own.
//
//   4. HOSTS STREAMLINE at the top of the loader, where the APP-VISIBLE device
//      handle lives. A Vulkan LAYER only sees its chain-internal device handle,
//      which the top loader does not recognise - hand THAT to slSetVulkanInfo
//      and Streamline builds an all-null dispatch table and crashes on the
//      first call (measured: 1064 null resolutions, AV at 0x0 in CreateSampler).
//      The shim sits above the loader, so the device it gets back from
//      vkCreateDevice is the one Streamline can actually use.
//
//   5. Arms our existing Vulkan layer so the whole mod loads on startup.
//
// The layer does the per-frame RR work (slSetTag + slEvaluateFeature); it finds
// the same Streamline the shim initialised and skips its own init (env
// TAA_SL_SHIM_OWNS). If Streamline is off or fails, forwarding still holds and
// X-Plane runs unchanged.
//
// Copyright (C) 2026 MotionVectors contributors. SPDX: GPL-3.0-or-later
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <cassert>
#include <tlhelp32.h>
#include <cwctype>

#include "sl.h"
#include "sl_helpers_vk.h"

static void shimTrace(const char *fmt, ...)
{
    char path[MAX_PATH]; DWORD n = GetTempPathA(MAX_PATH, path);
    if (!n) return;
    snprintf(path + n, sizeof(path) - n, "mv_vkshim.txt");
    FILE *f = fopen(path, "a"); if (!f) return;
    va_list ap; va_start(ap, fmt); vfprintf(f, fmt, ap); va_end(ap);
    fputc('\n', f); fclose(f);
}

// Streamline's own log, verbatim, in our trace. When slSetVulkanInfo throws
// this is the only place that says why.
static void slLogCallback(sl::LogType type, const char *msg)
{
    if (!msg) return;
    const char *k = (type == sl::LogType::eError) ? "ERR " : (type == sl::LogType::eWarn) ? "WARN" : "info";
    std::string m(msg);
    while (!m.empty() && (m.back() == 10 || m.back() == 13)) m.pop_back();
    shimTrace("  SL[%s] %s", k, m.c_str());
}

// ---- exception spelunking -------------------------------------------------
// Streamline swallows whatever its init throws and returns eErrorExceptionHandler
// (24), which says nothing about the cause and logs nothing. A vectored handler
// runs BEFORE Streamline's frame-based one, so we get to name the C++ type, the
// message and the throwing module, then let SL catch it exactly as it would.
static volatile LONG g_watchThrows = 0;

static void logAddr(const char *tag, const void *p)
{
    HMODULE m = nullptr; char name[MAX_PATH] = {0};
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)p, &m) && m && GetModuleFileNameA(m, name, MAX_PATH)) {
        const char *b = strrchr(name, 92);   // last backslash
        shimTrace("    %s %p  %s+0x%llx", tag, p, b ? b + 1 : name,
                  (unsigned long long)((const char *)p - (const char *)m));
    } else {
        shimTrace("    %s %p  (no module)", tag, p);
    }
}

static bool readable(const void *p, size_t n)
{
    MEMORY_BASIC_INFORMATION mbi{};
    if (!p || !VirtualQuery(p, &mbi, sizeof(mbi))) return false;
    if (mbi.State != MEM_COMMIT) return false;
    const DWORD ok = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY
                   | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    if (!(mbi.Protect & ok)) return false;
    return ((const char *)p + n) <= ((const char *)mbi.BaseAddress + mbi.RegionSize);
}

static LONG CALLBACK shimVeh(EXCEPTION_POINTERS *ep)
{
    if (!g_watchThrows) return EXCEPTION_CONTINUE_SEARCH;
    static LONG logged = 0;
    if (InterlockedIncrement(&logged) > 14) return EXCEPTION_CONTINUE_SEARCH;
    const DWORD kMsvcCpp = 0xE06D7363;
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    if (code != kMsvcCpp && code != (DWORD)EXCEPTION_ACCESS_VIOLATION) return EXCEPTION_CONTINUE_SEARCH;

    shimTrace("  THROW: code 0x%08lX at %p", code, ep->ExceptionRecord->ExceptionAddress);
    if (code == (DWORD)EXCEPTION_ACCESS_VIOLATION && ep->ExceptionRecord->NumberParameters >= 2)
        shimTrace("    AV %s address %p",
                  ep->ExceptionRecord->ExceptionInformation[0] ? "writing" : "reading",
                  (void *)ep->ExceptionRecord->ExceptionInformation[1]);

    if (code == kMsvcCpp && ep->ExceptionRecord->NumberParameters >= 4) {
        // MSVC throw: [1]=thrown object, [2]=ThrowInfo, [3]=base for its RVAs
        const char *base = (const char *)ep->ExceptionRecord->ExceptionInformation[3];
        const uint32_t *ti = (const uint32_t *)ep->ExceptionRecord->ExceptionInformation[2];
        if (base && readable(ti, 16)) {
            const uint32_t *cta = (const uint32_t *)(base + ti[3]);     // CatchableTypeArray
            if (readable(cta, 8) && cta[0]) {
                const uint32_t *ct = (const uint32_t *)(base + cta[1]); // first CatchableType
                if (readable(ct, 8)) {
                    const char *td = base + ct[1];                      // TypeDescriptor
                    if (readable(td, 32)) shimTrace("    C++ type: %s", td + 16);
                }
            }
        }
        const char *obj = (const char *)ep->ExceptionRecord->ExceptionInformation[1];
        if (readable(obj, 16)) {
            const char *msg = *(const char *const *)(obj + 8);          // std::exception message
            if (readable(msg, 1)) shimTrace("    what(): %s", msg);
        }
    }

    void *frames[24] = {0};
    typedef USHORT (WINAPI *PFN_cap)(ULONG, ULONG, PVOID *, PULONG);
    static PFN_cap cap = (PFN_cap)(void *)GetProcAddress(GetModuleHandleA("ntdll.dll"), "RtlCaptureStackBackTrace");
    if (cap) {
        USHORT n = cap(0, 24, frames, nullptr);
        for (USHORT i = 0; i < n; i++) logAddr("frame", frames[i]);
    }
    return EXCEPTION_CONTINUE_SEARCH;   // Streamline still gets to catch it
}

static HMODULE g_real = nullptr;   // vk_real_mv.dll
static HMODULE g_sl   = nullptr;   // sl.interposer.dll
static bool    g_inited = false;
static bool    g_slOn = false;
static PFN_vkGetInstanceProcAddr realGIPA = nullptr;
static PFN_vkGetDeviceProcAddr   realGDPA = nullptr;
static VkInstance g_lastInstance = VK_NULL_HANDLE;

// The device waiting to be handed to Streamline, once we are clear of the
// vkCreateDevice call frame.
static VkDevice         g_pendDev  = VK_NULL_HANDLE;
static VkPhysicalDevice g_pendPhys = VK_NULL_HANDLE;
static uint32_t g_pendCFam = 0, g_pendCIdx = 0, g_pendGFam = 0, g_pendGIdx = 0;
static bool     g_slHanded = false, g_slHanding = false;

typedef sl::Result (*PFN_slInit)(const sl::Preferences &, uint64_t);
typedef sl::Result (*PFN_slSetVulkanInfo)(const sl::VulkanInfo &);
typedef sl::Result (*PFN_slIsFeatureSupported)(sl::Feature, const sl::AdapterInfo &);
typedef sl::Result (*PFN_slGetFeatureRequirements)(sl::Feature, sl::FeatureRequirements &);
static PFN_slInit                    slInitFn = nullptr;
static PFN_slSetVulkanInfo           slSetVkFn = nullptr;
static PFN_slIsFeatureSupported      slIsSupp = nullptr;
static PFN_slGetFeatureRequirements  slReqFn = nullptr;
// sl.interposer's own vkQueuePresentKHR. In manual-hooking mode the app must
// present THROUGH Streamline or its presentCommon() never runs - SL says so
// outright: "internal bookkeeping and garbage collection will not run".
static PFN_vkQueuePresentKHR slPresentFn = nullptr;
static thread_local bool t_inSlPresent = false;

// What Streamline asked us for, captured once after slInit.
static std::vector<std::string> g_slInstExt, g_slDevExt, g_slFeat12, g_slFeat13;
static uint32_t g_slNumComputeQ = 0, g_slNumGfxQ = 0;
static bool     g_haveReq = false;

// DLSS-NR is feature 1004, introduced in Streamline 2.13. Our vendored 2.12
// headers stop at kFeatureDirectSR (1003), so name it here.
static constexpr sl::Feature kMvFeatureDLSS_NR = 1004;

// Which feature to build the device for. RR by default; NR when asked.
static sl::Feature g_slFeature = sl::kFeatureDLSS_RR;

static std::wstring shimDir()
{
    wchar_t buf[MAX_PATH * 2]; GetModuleFileNameW(nullptr, buf, MAX_PATH * 2);
    std::wstring p = buf; return p.substr(0, p.find_last_of(L"\\/"));
}
// Chosen in ensureInit when the feature needs a different runtime: DLSS Super Resolution
// (taa.backend=dlaa) loads NVIDIA's 2.13 set in MotionVectors\beta, which carries sl.dlss.dll;
// the 2.12 set in MotionVectors\streamline carries only DLSS-D.
static std::wstring g_slDirPick;
static std::wstring slDir()
{
    const char *e = getenv("TAA_SL_DIR");
    if (e && *e) { wchar_t w[MAX_PATH * 2]; MultiByteToWideChar(CP_UTF8, 0, e, -1, w, MAX_PATH * 2); return w; }
    if (!g_slDirPick.empty()) return g_slDirPick;
    return shimDir() + L"\\MotionVectors\\streamline";
}

// The live ini, or - when Windows' temp cleanup has deleted it - the packaged copy
// the layer is about to seed it from (MotionVectors\taa_live.ini in the package,
// MotionVectors\config\taa_live.ini in the dev tree). The shim reads BEFORE the
// layer exists, so without this a wiped ini meant taa.backend/taa.dlssd were
// silently off for the first launch after the wipe.
static FILE *openLiveIni()
{
    char path[MAX_PATH]; DWORD n = GetTempPathA(MAX_PATH, path);
    if (n) {
        snprintf(path + n, sizeof(path) - n, "taa_live.ini");
        if (FILE *f = fopen(path, "rb")) return f;
    }
    const std::wstring cands[2] = { shimDir() + L"\\MotionVectors\\taa_live.ini",
                                    shimDir() + L"\\MotionVectors\\config\\taa_live.ini" };
    for (int i = 0; i < 2; ++i)
        if (FILE *f = _wfopen(cands[i].c_str(), L"rb")) return f;
    return nullptr;
}

// The launcher may not set anything - the user just starts X-Plane. Honour the
// same live ini the layer reads so taa.dlssd=1 is enough to arm Streamline.
static bool liveIniOn(const char *key)
{
    FILE *f = openLiveIni(); if (!f) return false;
    char line[512]; bool on = false;
    const size_t klen = strlen(key);
    while (fgets(line, sizeof(line), f)) {
        const char *p = line;
        while (*p == ' ' || *p == '	') p++;
        if (*p == '#' || *p == ';') continue;
        if (strncmp(p, key, klen) != 0) continue;
        const char *v = p + klen;
        while (*v == ' ' || *v == '	') v++;
        if (*v != '=') continue;
        v++;
        while (*v == ' ' || *v == '	') v++;
        on = !(*v == '0' || !strncmp(v, "off", 3) || !strncmp(v, "false", 5) || !strncmp(v, "no", 2));
    }
    fclose(f);
    return on;
}

// The last value of a key in %TEMP%\taa_live.ini, trimmed; empty when absent.
static std::string liveIniStr(const char *key)
{
    FILE *f = openLiveIni(); if (!f) return std::string();
    char line[512]; std::string val;
    const size_t klen = strlen(key);
    while (fgets(line, sizeof(line), f)) {
        const char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == ';') continue;
        if (strncmp(p, key, klen) != 0) continue;
        const char *v = p + klen;
        while (*v == ' ' || *v == '\t') v++;
        if (*v != '=') continue;
        v++;
        while (*v == ' ' || *v == '\t') v++;
        std::string s(v);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
        val = s;
    }
    fclose(f);
    return val;
}

// ---- WHICH LAYERS THIS SHIM ARMS (shim 2.1).
//
// The shim can be shipped by more than one product (MotionVectors, and the FBW
// A380X port's taxi-camera layer), so it arms only what is actually installed:
//
//   VK_LAYER_mv   when MotionVectors' manifest exists (dev or package layout);
//   extra layers  listed in <X-Plane>\vk_extra_layers.ini, one per line:
//                     LAYER_NAME|relative\dir\holding\the\manifest
//                 '#' or ';' starts a comment. dir is resolved against the X-Plane
//                 folder; absolute paths, drive letters and ".." are REJECTED -
//                 this file decides which DLLs are loaded into the sim, so it must
//                 not be able to point outside it. At most 8 entries.
//
// Extra layers go FIRST in VK_INSTANCE_LAYERS (closest to X-Plane, so they see its
// unmodified calls); VK_LAYER_mv goes last. Nothing installed = no variables set,
// a pure pass-through to vk_real_mv.dll. Runs from DllMain under the loader lock,
// so the ini is read with plain Win32 calls and the report is traced later, from
// ensureInit.
static const char *kShimVersion = "2.1.0";
static std::string g_armReport;   // traced on the first Vulkan call
static bool g_mvArmed = false;    // VK_LAYER_mv armed: Streamline and the TAA_* flags belong to it

static bool extraLayerNameOk(const std::string &n)
{
    if (n.empty() || n.size() > 128) return false;
    for (char c : n) if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) return false;
    return true;
}

static bool extraLayerDirOk(const std::string &rel)
{
    if (rel.empty() || rel.size() > 200) return false;
    if (rel[0] == '\\' || rel[0] == '/') return false;            // rooted
    if (rel.find(':') != std::string::npos) return false;          // drive letter, stream
    if (rel.find("..") != std::string::npos) return false;         // escaping the folder
    return true;
}

static void readExtraLayers(const std::wstring &dir, std::vector<std::wstring> &names,
                            std::vector<std::wstring> &paths)
{
    const std::wstring ini = dir + L"\\vk_extra_layers.ini";
    HANDLE h = CreateFileW(ini.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return;
    char buf[16384]; DWORD got = 0;
    BOOL ok = ReadFile(h, buf, sizeof(buf) - 1, &got, nullptr);
    CloseHandle(h);
    if (!ok) return;
    buf[got] = 0;
    const char *p = buf;
    if ((unsigned char)p[0] == 0xEF && (unsigned char)p[1] == 0xBB && (unsigned char)p[2] == 0xBF) p += 3;   // BOM
    std::string all(p);
    size_t pos = 0;
    while (pos < all.size() && names.size() < 8) {
        size_t e = all.find('\n', pos); if (e == std::string::npos) e = all.size();
        std::string line = all.substr(pos, e - pos); pos = e + 1;
        size_t c = line.find_first_of("#;"); if (c != std::string::npos) line.resize(c);
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) line.pop_back();
        size_t s = line.find_first_not_of(" \t"); if (s == std::string::npos) continue;
        line = line.substr(s);
        size_t bar = line.find('|');
        if (bar == std::string::npos) { g_armReport += "SHIM: vk_extra_layers.ini: skipped '" + line + "' (want NAME|dir)\n"; continue; }
        std::string name = line.substr(0, bar), rel = line.substr(bar + 1);
        while (!name.empty() && (name.back() == ' ' || name.back() == '\t')) name.pop_back();
        size_t rs = rel.find_first_not_of(" \t"); rel = (rs == std::string::npos) ? std::string() : rel.substr(rs);
        for (char &ch : rel) if (ch == '/') ch = '\\';
        while (!rel.empty() && rel.back() == '\\') rel.pop_back();
        if (!extraLayerNameOk(name) || name == "VK_LAYER_mv") { g_armReport += "SHIM: vk_extra_layers.ini: REJECTED name '" + name + "'\n"; continue; }
        if (!extraLayerDirOk(rel)) { g_armReport += "SHIM: vk_extra_layers.ini: REJECTED dir '" + rel + "' for " + name + " (must be relative, inside the X-Plane folder)\n"; continue; }
        std::wstring wrel(rel.begin(), rel.end());
        std::wstring abs = dir + L"\\" + wrel;
        DWORD a = GetFileAttributesW(abs.c_str());
        if (a == INVALID_FILE_ATTRIBUTES || !(a & FILE_ATTRIBUTE_DIRECTORY)) { g_armReport += "SHIM: vk_extra_layers.ini: " + name + " - dir '" + rel + "' not found, skipped\n"; continue; }
        bool dup = false; std::wstring wname(name.begin(), name.end());
        for (const std::wstring &n : names) if (n == wname) dup = true;
        if (dup) continue;
        names.push_back(wname); paths.push_back(abs);
        g_armReport += "SHIM: extra layer " + name + " from " + rel + "\n";
    }
}

static std::wstring joinW(const std::vector<std::wstring> &v, wchar_t sep)
{
    std::wstring s;
    for (size_t i = 0; i < v.size(); ++i) { if (i) s += sep; s += v[i]; }
    return s;
}

static void armLayer()
{
    std::wstring dir = shimDir();
    // Two layouts: the development tree keeps the layer in
    // MotionVectors\build\vklayer; the drag-and-drop package puts it directly
    // in MotionVectors\. Probe for the manifest rather than assume either - and
    // arm nothing of ours when neither exists (the shim can be installed without us).
    std::wstring mvPath;
    const wchar_t *cands[2] = { L"\\MotionVectors\\build\\vklayer", L"\\MotionVectors" };
    for (int i = 0; i < 2 && mvPath.empty(); ++i)
        if (GetFileAttributesW((dir + cands[i] + L"\\VkLayer_mv.json").c_str()) != INVALID_FILE_ATTRIBUTES)
            mvPath = dir + cands[i];

    std::vector<std::wstring> names, paths;
    readExtraLayers(dir, names, paths);
    if (!mvPath.empty()) { names.push_back(L"VK_LAYER_mv"); paths.push_back(mvPath); g_mvArmed = true; }
    else g_armReport += "SHIM: MotionVectors not installed (no VkLayer_mv.json) - VK_LAYER_mv not armed\n";

    if (names.empty()) { g_armReport += "SHIM: no layers to arm - pass-through to vk_real_mv.dll\n"; return; }
    SetEnvironmentVariableW(L"VK_LAYER_PATH", joinW(paths, L';').c_str());
    SetEnvironmentVariableW(L"VK_INSTANCE_LAYERS", joinW(names, L';').c_str());
    SetEnvironmentVariableW(L"VK_LOADER_LAYERS_ENABLE", joinW(names, L',').c_str());
    std::wstring order = joinW(names, L' ');
    g_armReport += "SHIM: armed (closest to X-Plane first): " + std::string(order.begin(), order.end()) + "\n";
    if (!g_mvArmed) return;
    SetEnvironmentVariableW(L"TAA_VELOCITY", L"1");
    SetEnvironmentVariableW(L"TAA_LAYER_TRACE", L"1");
    // Tell the layer that the shim owns Streamline init - the layer must NOT
    // call slInit/slSetVulkanInfo with its chain-internal device.
    SetEnvironmentVariableW(L"TAA_SL_SHIM_OWNS", L"1");
}

// Ask Streamline what the device must look like. Called once, after slInit and
// before the app's first vkCreateInstance.
static void captureRequirements()
{
    if (!slReqFn) { shimTrace("SHIM: no slGetFeatureRequirements export - cannot merge SL device needs"); return; }
    sl::FeatureRequirements req{};
    sl::Result r = slReqFn(g_slFeature, req);
    shimTrace("SHIM: slGetFeatureRequirements(feature %u) -> %d", (unsigned)g_slFeature, (int)r);
    if (r != sl::Result::eOk) {
        shimTrace("SHIM:   driver detected %u.%u.%u required %u.%u.%u",
                  req.driverVersionDetected.major, req.driverVersionDetected.minor, req.driverVersionDetected.build,
                  req.driverVersionRequired.major, req.driverVersionRequired.minor, req.driverVersionRequired.build);
        return;
    }
    for (uint32_t i = 0; i < req.vkNumInstanceExtensions; i++) if (req.vkInstanceExtensions[i]) g_slInstExt.push_back(req.vkInstanceExtensions[i]);
    for (uint32_t i = 0; i < req.vkNumDeviceExtensions;   i++) if (req.vkDeviceExtensions[i])   g_slDevExt.push_back(req.vkDeviceExtensions[i]);
    for (uint32_t i = 0; i < req.vkNumFeatures12;         i++) if (req.vkFeatures12[i])         g_slFeat12.push_back(req.vkFeatures12[i]);
    for (uint32_t i = 0; i < req.vkNumFeatures13;         i++) if (req.vkFeatures13[i])         g_slFeat13.push_back(req.vkFeatures13[i]);
    g_slNumComputeQ = req.vkNumComputeQueuesRequired;
    g_slNumGfxQ     = req.vkNumGraphicsQueuesRequired;
    g_haveReq = true;

    shimTrace("SHIM: === STREAMLINE REQUIREMENTS ===");
    shimTrace("SHIM:   driver detected %u.%u.%u  required %u.%u.%u",
              req.driverVersionDetected.major, req.driverVersionDetected.minor, req.driverVersionDetected.build,
              req.driverVersionRequired.major, req.driverVersionRequired.minor, req.driverVersionRequired.build);
    shimTrace("SHIM:   queues: compute=%u graphics=%u opticalflow=%u",
              g_slNumComputeQ, g_slNumGfxQ, req.vkNumOpticalFlowQueuesRequired);
    for (auto &s : g_slInstExt) shimTrace("SHIM:   instance ext: %s", s.c_str());
    for (auto &s : g_slDevExt)  shimTrace("SHIM:   device   ext: %s", s.c_str());
    for (auto &s : g_slFeat12)  shimTrace("SHIM:   feature 1.2 : %s", s.c_str());
    for (auto &s : g_slFeat13)  shimTrace("SHIM:   feature 1.3 : %s", s.c_str());
    shimTrace("SHIM: === END REQUIREMENTS ===");
}

static void lockProbe(const char *where);
static void dumpModules(const char *when);

static void ensureInit()
{
    if (g_inited) return;
    g_inited = true;
    std::wstring real = shimDir() + L"\\vk_real_mv.dll";
    g_real = LoadLibraryW(real.c_str());
    if (!g_real) { shimTrace("SHIM: FATAL cannot load %ls (%lu)", real.c_str(), GetLastError()); return; }
    realGIPA = (PFN_vkGetInstanceProcAddr)GetProcAddress(g_real, "vkGetInstanceProcAddr");
    realGDPA = (PFN_vkGetDeviceProcAddr)GetProcAddress(g_real, "vkGetDeviceProcAddr");
    shimTrace("SHIM: ---- session start ---- shim %s, real loader GIPA=%p GDPA=%p", kShimVersion, (void*)realGIPA, (void*)realGDPA);
    {   // what armLayer() decided under the loader lock, one line each
        size_t s = 0;
        while (s < g_armReport.size()) {
            size_t e = g_armReport.find('\n', s); if (e == std::string::npos) e = g_armReport.size();
            shimTrace("%s", g_armReport.substr(s, e - s).c_str());
            s = e + 1;
        }
    }

    const bool wantNR = getenv("TAA_DLSS_NR") != nullptr || liveIniOn("taa.dlss_nr");
    const bool wantRR = getenv("TAA_DLSSD")   != nullptr || liveIniOn("taa.dlssd");
    if (wantNR) g_slFeature = kMvFeatureDLSS_NR;

    // taa.backend=dlaa|dlss: DLSS Super Resolution (feature 0) for the layer's DLAA backend.
    // RR/NR, when also asked for, keep the single Streamline context.
    std::string backend = liveIniStr("taa.backend");
    if (const char *eb = getenv("TAA_BACKEND")) backend = eb;
    const bool wantDLSS = (backend == "dlaa" || backend == "dlss") && !wantRR && !wantNR;
    if (wantDLSS) {
        g_slFeature = sl::kFeatureDLSS;
        std::wstring beta = shimDir() + L"\\MotionVectors\\beta";
        if (GetFileAttributesW((beta + L"\\sl.dlss.dll").c_str()) != INVALID_FILE_ATTRIBUTES)
            g_slDirPick = beta;
        shimTrace("SHIM: taa.backend=%s - arming Streamline DLSS (feature 0) from %ls",
                  backend.c_str(), slDir().c_str());
    }

    // Streamline serves VK_LAYER_mv only; with MotionVectors absent a stale %TEMP% ini
    // must not make the shim load an interposer for a layer that is not there.
    if ((wantRR || wantNR || wantDLSS) && !g_mvArmed)
        shimTrace("SHIM: Streamline requested by the live ini but VK_LAYER_mv is not armed - not loading it");
    if ((wantRR || wantNR || wantDLSS) && g_mvArmed) {
        std::wstring sl = slDir() + L"\\sl.interposer.dll";
        SetDllDirectoryW(slDir().c_str());
        g_sl = LoadLibraryW(sl.c_str());
        SetDllDirectoryW(nullptr);
        if (g_sl) {
            slInitFn  = (PFN_slInit)(void*)GetProcAddress(g_sl, "slInit");
            slSetVkFn = (PFN_slSetVulkanInfo)(void*)GetProcAddress(g_sl, "slSetVulkanInfo");
            slIsSupp  = (PFN_slIsFeatureSupported)(void*)GetProcAddress(g_sl, "slIsFeatureSupported");
            slReqFn   = (PFN_slGetFeatureRequirements)(void*)GetProcAddress(g_sl, "slGetFeatureRequirements");
            slPresentFn = (PFN_vkQueuePresentKHR)(void*)GetProcAddress(g_sl, "vkQueuePresentKHR");
            g_slOn = (slInitFn && slSetVkFn);
            shimTrace("SHIM: sl.interposer %ls slInit=%p slSetVulkanInfo=%p slGetFeatureRequirements=%p",
                      sl.c_str(), (void*)slInitFn, (void*)slSetVkFn, (void*)slReqFn);
        } else shimTrace("SHIM: sl.interposer load failed (%lu)", GetLastError());
    }

    if (g_slOn) {
        static const wchar_t *paths[1]; static std::wstring d = slDir(); paths[0] = d.c_str();
        static sl::Feature feats[1]; feats[0] = g_slFeature;
        sl::Preferences pref{};
        pref.logLevel = getenv("TAA_SL_QUIET") ? sl::LogLevel::eDefault : sl::LogLevel::eVerbose;
        pref.logMessageCallback = slLogCallback;
        pref.pathsToPlugins = paths; pref.numPathsToPlugins = 1;
        pref.flags = sl::PreferenceFlags::eUseManualHooking | sl::PreferenceFlags::eDisableCLStateTracking;
        pref.featuresToLoad = feats; pref.numFeaturesToLoad = 1;
        pref.applicationId = 0x5A1D0003u;
        pref.engine = sl::EngineType::eCustom;
        pref.renderAPI = sl::RenderAPI::eVulkan;
        DWORD initTid = GetCurrentThreadId();
        sl::Result r = slInitFn(pref, sl::kSDKVersion);
        shimTrace("SHIM: slInit(Vulkan, manual hooking, feature %u) on tid %lu -> %d",
                  (unsigned)g_slFeature, initTid, (int)r);
        if (r != sl::Result::eOk) g_slOn = false;
        else { captureRequirements(); lockProbe("after slInit"); }
    }
}

// Is sl.interposer's global lock still takeable? slGetFeatureRequirements is
// read-only and uses it, so its result is a clean yes/no. Used to bracket the
// exact call that acquires the lock and never gives it back.
// Which Vulkan/Streamline modules are live right now. Tells us whether our own
// layer, or something else, joined the chain at device creation.
static void dumpModules(const char *when)
{
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, 0);
    if (snap == INVALID_HANDLE_VALUE) return;
    MODULEENTRY32W me{}; me.dwSize = sizeof(me);
    shimTrace("SHIM: === modules (%s) ===", when);
    if (Module32FirstW(snap, &me)) do {
        std::wstring n = me.szModule;
        for (auto &c : n) c = (wchar_t)towlower(c);
        if (n.find(L"vklayer") != std::wstring::npos || n.find(L"sl.") != std::wstring::npos ||
            n.find(L"nvngx")   != std::wstring::npos || n.find(L"vulkan") != std::wstring::npos ||
            n.find(L"lowlatency") != std::wstring::npos || n.find(L"nvoglv") != std::wstring::npos ||
            n.find(L"vk_real") != std::wstring::npos)
            shimTrace("SHIM:   %ls", me.szModule);
    } while (Module32NextW(snap, &me));
    CloseHandle(snap);
}

static void lockProbe(const char *where)
{
    // DEBUG ONLY (TAA_SL_LOCKPROBE=1). Asked before the device exists,
    // slGetFeatureRequirements throws inside sl.interposer, which catches it,
    // writes a minidump (211 of them in ProgramData by 2026-09-17) and answers 24
    // - the same swallowed-throw path that has leaked its lock before. A
    // diagnostic that can cause the fault it measures stays off by default.
    static const bool on = getenv("TAA_SL_LOCKPROBE") != nullptr;
    if (!on) return;
    if (!g_slOn || !slReqFn) return;
    sl::FeatureRequirements q{};
    sl::Result r = slReqFn(g_slFeature, q);
    shimTrace("SHIM: [lock @ %-22s] tid %lu -> %s (result %d)", where, GetCurrentThreadId(),
              (r == sl::Result::eOk) ? "FREE" : "HELD", (int)r);
}

extern "C" VkResult VKAPI_CALL vkCreateInstance(const VkInstanceCreateInfo *, const VkAllocationCallbacks *, VkInstance *);
extern "C" VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice, const VkDeviceCreateInfo *, const VkAllocationCallbacks *, VkDevice *);
extern "C" VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char *, uint32_t *, VkExtensionProperties *);
extern "C" VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t *, VkLayerProperties *);
extern "C" VkResult VKAPI_CALL vkEnumerateInstanceVersion(uint32_t *);
extern "C" PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice, const char *);
extern "C" VkResult VKAPI_CALL vkQueuePresentKHR(VkQueue, const VkPresentInfoKHR *);

extern "C" __declspec(dllexport)
PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *pName)
{
    ensureInit();
    if (pName) {
        if (!strcmp(pName, "vkGetInstanceProcAddr")) return (PFN_vkVoidFunction)&vkGetInstanceProcAddr;
        if (!strcmp(pName, "vkGetDeviceProcAddr"))   return (PFN_vkVoidFunction)&vkGetDeviceProcAddr;
        if (!strcmp(pName, "vkCreateInstance"))       return (PFN_vkVoidFunction)&vkCreateInstance;
        if (!strcmp(pName, "vkCreateDevice"))         return (PFN_vkVoidFunction)&vkCreateDevice;
        if (!strcmp(pName, "vkEnumerateInstanceExtensionProperties")) return (PFN_vkVoidFunction)&vkEnumerateInstanceExtensionProperties;
        if (!strcmp(pName, "vkEnumerateInstanceLayerProperties"))     return (PFN_vkVoidFunction)&vkEnumerateInstanceLayerProperties;
        if (!strcmp(pName, "vkEnumerateInstanceVersion"))             return (PFN_vkVoidFunction)&vkEnumerateInstanceVersion;
        if (!strcmp(pName, "vkQueuePresentKHR") && g_slOn && slPresentFn && !t_inSlPresent)
            return (PFN_vkVoidFunction)&vkQueuePresentKHR;
    }
    return realGIPA ? realGIPA(instance, pName) : nullptr;
}

static sl::VulkanInfo g_thrVi{};
static sl::Result     g_thrRes = (sl::Result)999;
static DWORD WINAPI handoffThreadProc(LPVOID)
{
    PVOID veh = AddVectoredExceptionHandler(1, shimVeh);
    InterlockedExchange(&g_watchThrows, 1);
    g_thrRes = slSetVkFn(g_thrVi);
    InterlockedExchange(&g_watchThrows, 0);
    if (veh) RemoveVectoredExceptionHandler(veh);
    return 0;
}

// Give Streamline the app-visible device, from OUTSIDE vkCreateDevice's frame.
// sl.interposer holds a non-recursive lock across device creation, so calling
// it there throws; the first device-level call after creation is both safe and
// still early enough for SL ("immediately after the base interface is created").
static void handOffToSL(const char *when)
{
    if (g_slHanded || g_slHanding || !g_pendDev || !g_slOn || !slSetVkFn) return;
    g_slHanding = true;
    sl::VulkanInfo vi{};
    vi.device = g_pendDev; vi.physicalDevice = g_pendPhys; vi.instance = g_lastInstance;
    vi.computeQueueFamily  = g_pendCFam; vi.computeQueueIndex  = g_pendCIdx;
    vi.graphicsQueueFamily = g_pendGFam; vi.graphicsQueueIndex = g_pendGIdx;

    PVOID veh = AddVectoredExceptionHandler(1, shimVeh);
    InterlockedExchange(&g_watchThrows, 1);
    lockProbe("hand-off");
    sl::Result sr = slSetVkFn(vi);
    InterlockedExchange(&g_watchThrows, 0);
    if (veh) RemoveVectoredExceptionHandler(veh);

    shimTrace("SHIM: slSetVulkanInfo(%s) dev %p compute %u:%u gfx %u:%u -> %d",
              when, (void*)g_pendDev, g_pendCFam, g_pendCIdx, g_pendGFam, g_pendGIdx, (int)sr);

    if (sr == sl::Result::eOk) {
        if (!slIsSupp) shimTrace("SHIM: slIsFeatureSupported export missing - cannot check viability");
        else {
            sl::AdapterInfo ai{}; ai.vkPhysicalDevice = g_pendPhys;
            shimTrace("SHIM: calling slIsFeatureSupported(%u)...", (unsigned)g_slFeature);
            sl::Result fr = slIsSupp(g_slFeature, ai);
            shimTrace("SHIM: slIsFeatureSupported(%u) -> %d  <== VIABILITY", (unsigned)g_slFeature, (int)fr);
        }
    }
    shimTrace("SHIM: SL HAND-OFF DONE (slSetVulkanInfo=%d)", (int)sr);

    g_slHanded = true;
    g_slHanding = false;
}

extern "C" __declspec(dllexport)
PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *pName)
{
    ensureInit();
    if (g_pendDev && !g_slHanded && !g_slHanding) handOffToSL("deferred/GDPA");
    // Hand the app OUR present so Streamline's presentCommon() runs each frame.
    // SL resolving its own downstream present lands here too - the thread guard
    // inside sends that one to the loader.
    if (pName && !strcmp(pName, "vkQueuePresentKHR") && g_slOn && slPresentFn && !t_inSlPresent)
        return (PFN_vkVoidFunction)&vkQueuePresentKHR;
    return realGDPA ? realGDPA(device, pName) : nullptr;
}

static bool hasName(const std::vector<const char*> &v, const char *n)
{
    for (auto *s : v) if (s && !strcmp(s, n)) return true;
    return false;
}

extern "C" __declspec(dllexport)
VkResult VKAPI_CALL vkCreateInstance(const VkInstanceCreateInfo *ci, const VkAllocationCallbacks *a, VkInstance *out)
{
    ensureInit();
    PFN_vkCreateInstance f = realGIPA ? (PFN_vkCreateInstance)realGIPA(nullptr, "vkCreateInstance") : nullptr;
    if (!f) return VK_ERROR_INITIALIZATION_FAILED;

    VkApplicationInfo app{}; app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    if (ci->pApplicationInfo) app = *ci->pApplicationInfo; else app.apiVersion = VK_API_VERSION_1_0;
    uint32_t asked = app.apiVersion;
    VkInstanceCreateInfo ci2 = *ci;
    if (app.apiVersion < VK_API_VERSION_1_3) { app.apiVersion = VK_API_VERSION_1_3; ci2.pApplicationInfo = &app; }

    // Union SL's instance extensions with the app's, keeping only ones the
    // loader actually reports.
    std::vector<const char*> exts(ci->ppEnabledExtensionNames,
                                  ci->ppEnabledExtensionNames + ci->enabledExtensionCount);
    if (!g_slInstExt.empty()) {
        PFN_vkEnumerateInstanceExtensionProperties eF = realGIPA
            ? (PFN_vkEnumerateInstanceExtensionProperties)realGIPA(nullptr, "vkEnumerateInstanceExtensionProperties") : nullptr;
        std::vector<VkExtensionProperties> avail;
        if (eF) { uint32_t n = 0; eF(nullptr, &n, nullptr); avail.resize(n); if (n) eF(nullptr, &n, avail.data()); }
        for (auto &want : g_slInstExt) {
            if (hasName(exts, want.c_str())) continue;
            bool ok = false;
            for (auto &p : avail) if (!strcmp(p.extensionName, want.c_str())) { ok = true; break; }
            if (ok) { exts.push_back(want.c_str()); shimTrace("SHIM:   +instance ext %s", want.c_str()); }
            else      shimTrace("SHIM:   !instance ext %s NOT AVAILABLE", want.c_str());
        }
        ci2.enabledExtensionCount = (uint32_t)exts.size();
        ci2.ppEnabledExtensionNames = exts.data();
    }

    lockProbe("vkCreateInstance entry");
    VkResult r = f(&ci2, a, out);
    lockProbe("vkCreateInstance exit");
    shimTrace("SHIM: vkCreateInstance app='%s' api %u.%u -> %u.%u  exts %u->%u  result=%d",
              (ci->pApplicationInfo && ci->pApplicationInfo->pApplicationName) ? ci->pApplicationInfo->pApplicationName : "(none)",
              VK_API_VERSION_MAJOR(asked), VK_API_VERSION_MINOR(asked),
              VK_API_VERSION_MAJOR(app.apiVersion), VK_API_VERSION_MINOR(app.apiVersion),
              ci->enabledExtensionCount, ci2.enabledExtensionCount, (int)r);
    if (r == VK_SUCCESS && out) g_lastInstance = *out;
    return r;
}

// Walk a pNext chain for a given sType. Returns the node, or null.
static void *findChained(const void *pNext, VkStructureType t)
{
    auto *p = (VkBaseOutStructure *)const_cast<void *>(pNext);
    while (p) { if (p->sType == t) return p; p = p->pNext; }
    return nullptr;
}

// The app-visible device. Hand THIS to Streamline, not the layer's handle -
// and build it with the features/extensions/queues Streamline needs.
extern "C" __declspec(dllexport)
VkResult VKAPI_CALL vkCreateDevice(VkPhysicalDevice phys, const VkDeviceCreateInfo *ci, const VkAllocationCallbacks *a, VkDevice *out)
{
    ensureInit();
    PFN_vkCreateDevice f = realGIPA ? (PFN_vkCreateDevice)realGIPA(g_lastInstance, "vkCreateDevice") : nullptr;
    if (!f) f = realGIPA ? (PFN_vkCreateDevice)realGIPA(nullptr, "vkCreateDevice") : nullptr;
    if (!f) return VK_ERROR_INITIALIZATION_FAILED;

    lockProbe("vkCreateDevice entry");
    VkDeviceCreateInfo ci2 = *ci;

    // Storage that must outlive the rewrite but not the call.
    std::vector<const char*>             exts(ci->ppEnabledExtensionNames,
                                              ci->ppEnabledExtensionNames + ci->enabledExtensionCount);
    std::vector<VkDeviceQueueCreateInfo> queues(ci->pQueueCreateInfos,
                                                ci->pQueueCreateInfos + ci->queueCreateInfoCount);
    std::vector<std::vector<float>>      prios;
    VkPhysicalDeviceVulkan12Features     f12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    VkPhysicalDeviceVulkan13Features     f13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };

    // SL queue placement, reported to slSetVulkanInfo below.
    uint32_t cFam = 0, cIdx = 0, gFam = 0, gIdx = 0;
    if (ci->queueCreateInfoCount && ci->pQueueCreateInfos) {
        cFam = gFam = ci->pQueueCreateInfos[0].queueFamilyIndex;
    }

    if (g_slOn && g_haveReq) {
        auto gipa = [&](const char *n) { return realGIPA ? realGIPA(g_lastInstance, n) : nullptr; };

        // ---- device extensions: SL's list, filtered by what the GPU reports
        auto devExtF = (PFN_vkEnumerateDeviceExtensionProperties)gipa("vkEnumerateDeviceExtensionProperties");
        std::vector<VkExtensionProperties> avail;
        if (devExtF) { uint32_t n = 0; devExtF(phys, nullptr, &n, nullptr); avail.resize(n); if (n) devExtF(phys, nullptr, &n, avail.data()); }
        for (auto &want : g_slDevExt) {
            if (hasName(exts, want.c_str())) { shimTrace("SHIM:   =device ext %s (app already had it)", want.c_str()); continue; }
            bool ok = false;
            for (auto &p : avail) if (!strcmp(p.extensionName, want.c_str())) { ok = true; break; }
            if (ok) { exts.push_back(want.c_str()); shimTrace("SHIM:   +device ext %s", want.c_str()); }
            else      shimTrace("SHIM:   !device ext %s NOT SUPPORTED BY GPU", want.c_str());
        }
        ci2.enabledExtensionCount   = (uint32_t)exts.size();
        ci2.ppEnabledExtensionNames = exts.data();

        // ---- 1.2 / 1.3 features: SL's wanted bits AND device support, ORed
        //      into whatever the app already chained (or a fresh struct).
        VkPhysicalDeviceVulkan12Features want12 = sl::getVkPhysicalDeviceVulkan12Features(0, nullptr);
        VkPhysicalDeviceVulkan13Features want13 = sl::getVkPhysicalDeviceVulkan13Features(0, nullptr);
        {   // build the "wanted" sets from SL's name lists
            std::vector<const char*> n12, n13;
            for (auto &s : g_slFeat12) n12.push_back(s.c_str());
            for (auto &s : g_slFeat13) n13.push_back(s.c_str());
            want12 = sl::getVkPhysicalDeviceVulkan12Features((uint32_t)n12.size(), n12.empty() ? nullptr : n12.data());
            want13 = sl::getVkPhysicalDeviceVulkan13Features((uint32_t)n13.size(), n13.empty() ? nullptr : n13.data());
        }

        VkPhysicalDeviceVulkan12Features sup12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
        VkPhysicalDeviceVulkan13Features sup13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
        sup12.pNext = &sup13;
        VkPhysicalDeviceFeatures2 sup2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 }; sup2.pNext = &sup12;
        auto featF = (PFN_vkGetPhysicalDeviceFeatures2)gipa("vkGetPhysicalDeviceFeatures2");
        if (!featF) featF = (PFN_vkGetPhysicalDeviceFeatures2)gipa("vkGetPhysicalDeviceFeatures2KHR");
        if (featF) featF(phys, &sup2);
        else shimTrace("SHIM:   ! vkGetPhysicalDeviceFeatures2 unavailable - enabling SL bits unchecked");

        // Merge into the app's chained structs if present, else use our own.
        auto *app12 = (VkPhysicalDeviceVulkan12Features *)findChained(ci->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
        auto *app13 = (VkPhysicalDeviceVulkan13Features *)findChained(ci->pNext, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
        VkPhysicalDeviceVulkan12Features *dst12 = app12 ? app12 : &f12;
        VkPhysicalDeviceVulkan13Features *dst13 = app13 ? app13 : &f13;
        if (!app12) { f12 = VkPhysicalDeviceVulkan12Features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES }; }
        if (!app13) { f13 = VkPhysicalDeviceVulkan13Features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES }; }

        // sl_helpers_vk.h does exactly this merge for us: dst = (dst || want) && supported
        sl::getMergedSupportedVkPhysicalDeviceVulkanFeatures(
            (VkBaseOutStructure *)dst12, (const VkBaseOutStructure *)&want12, (const VkBaseOutStructure *)&sup12);
        sl::getMergedSupportedVkPhysicalDeviceVulkanFeatures(
            (VkBaseOutStructure *)dst13, (const VkBaseOutStructure *)&want13, (const VkBaseOutStructure *)&sup13);
        // SL's requirement list does not name privateData, but sl.common's
        // chi::Vulkan::init calls vkCreatePrivateDataSlot regardless and logs
        // "not available" when the feature is off. Turn it on if the GPU has it.
        if (sup13.privateData) dst13->privateData = VK_TRUE;
        // Same story pre-1.3: hand it the EXT spelling if the GPU offers it.
        if (!hasName(exts, VK_EXT_PRIVATE_DATA_EXTENSION_NAME)) {
            for (auto &pp : avail) if (!strcmp(pp.extensionName, VK_EXT_PRIVATE_DATA_EXTENSION_NAME)) {
                exts.push_back(VK_EXT_PRIVATE_DATA_EXTENSION_NAME);
                ci2.enabledExtensionCount = (uint32_t)exts.size();
                ci2.ppEnabledExtensionNames = exts.data();
                shimTrace("SHIM:   +device ext %s", VK_EXT_PRIVATE_DATA_EXTENSION_NAME);
                break;
            }
        }

        shimTrace("SHIM:   features: app chained 1.2=%s 1.3=%s; privateData=%d sync2=%d bufferDeviceAddress=%d timelineSemaphore=%d",
                  app12 ? "yes" : "no", app13 ? "yes" : "no",
                  (int)dst13->privateData, (int)dst13->synchronization2,
                  (int)dst12->bufferDeviceAddress, (int)dst12->timelineSemaphore);

        // Chain ours only if the app had none of its own.
        if (!app13) { f13.pNext = (void *)ci2.pNext; ci2.pNext = &f13; }
        if (!app12) { f12.pNext = (void *)ci2.pNext; ci2.pNext = &f12; }

        // ---- queues: reserve SL's past the app's own in a compute family
        auto qpF = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)gipa("vkGetPhysicalDeviceQueueFamilyProperties");
        std::vector<VkQueueFamilyProperties> qprops;
        if (qpF) { uint32_t n = 0; qpF(phys, &n, nullptr); qprops.resize(n); if (n) qpF(phys, &n, qprops.data()); }
        for (uint32_t i = 0; i < qprops.size(); i++)
            shimTrace("SHIM:   family %u: count=%u flags=0x%x%s%s", i, qprops[i].queueCount, qprops[i].queueFlags,
                      (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) ? " GFX" : "",
                      (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) ? " COMP" : "");
        for (auto &q : queues)
            shimTrace("SHIM:   app asks family %u x%u", q.queueFamilyIndex, q.queueCount);

        // Grow an existing app queue-family request by the count SL wants, if
        // the family can take more. SL's queues then start at the old count.
        auto reserve = [&](VkQueueFlagBits need, uint32_t count, uint32_t &famOut, uint32_t &idxOut) {
            if (!count) return;
            for (size_t i = 0; i < queues.size(); i++) {
                uint32_t fam = queues[i].queueFamilyIndex;
                if (fam >= qprops.size()) continue;
                if (!(qprops[fam].queueFlags & need)) continue;
                uint32_t have = queues[i].queueCount;
                uint32_t room = qprops[fam].queueCount;
                famOut = fam;
                if (have + count <= room) {
                    prios.push_back(std::vector<float>(have + count, 1.0f));
                    // preserve the app's own priorities for its queues
                    if (queues[i].pQueuePriorities)
                        for (uint32_t k = 0; k < have; k++) prios.back()[k] = queues[i].pQueuePriorities[k];
                    queues[i].queueCount = have + count;
                    queues[i].pQueuePriorities = prios.back().data();
                    idxOut = have;
                    shimTrace("SHIM:   reserved %u queue(s) for SL in family %u at index %u (family cap %u)",
                              count, fam, have, room);
                } else {
                    idxOut = 0;
                    shimTrace("SHIM:   family %u FULL (%u/%u) - SL shares index 0", fam, have, room);
                }
                return;
            }
            shimTrace("SHIM:   no app queue family with flags 0x%x - SL falls back to family %u index 0", (unsigned)need, famOut);
        };
        reserve(VK_QUEUE_COMPUTE_BIT,  g_slNumComputeQ, cFam, cIdx);
        reserve(VK_QUEUE_GRAPHICS_BIT, g_slNumGfxQ,     gFam, gIdx);

        ci2.queueCreateInfoCount = (uint32_t)queues.size();
        ci2.pQueueCreateInfos    = queues.data();
    }

    dumpModules("before real vkCreateDevice"); lockProbe("before real create");
    VkResult r = f(phys, &ci2, a, out);
    lockProbe("after real create"); dumpModules("after real vkCreateDevice");
    shimTrace("SHIM: vkCreateDevice -> %d device=%p (exts %u->%u, queues %u->%u)",
              (int)r, (void*)(out ? *out : nullptr),
              ci->enabledExtensionCount, ci2.enabledExtensionCount,
              ci->queueCreateInfoCount, ci2.queueCreateInfoCount);

    if (r != VK_SUCCESS && ci2.enabledExtensionCount != ci->enabledExtensionCount) {
        // Our additions broke the device. Fall back to exactly what X-Plane
        // asked for so the sim still runs; SL just stays off.
        shimTrace("SHIM: RETRY with the app's original VkDeviceCreateInfo (SL merge rejected)");
        r = f(phys, ci, a, out);
        shimTrace("SHIM: vkCreateDevice(original) -> %d device=%p", (int)r, (void*)(out ? *out : nullptr));
        if (r == VK_SUCCESS) g_slOn = false;
    }

    if (r == VK_SUCCESS && g_slOn && slSetVkFn && out) {
        // Remember the device; the hand-off happens once we are OUT of this
        // call frame. Calling slSetVulkanInfo from inside vkCreateDevice makes
        // sl.interposer re-enter a std::mutex this thread already holds and it
        // throws std::system_error("device or resource busy") -> eResult 24.
        g_pendDev = *out; g_pendPhys = phys;
        g_pendCFam = cFam; g_pendCIdx = cIdx; g_pendGFam = gFam; g_pendGIdx = gIdx;
        if (getenv("TAA_SL_INLINE")) handOffToSL("inline");
        else shimTrace("SHIM: device handed to the deferred Streamline hand-off");
    }
    return r;
}

extern "C" __declspec(dllexport)
VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(const char *layer, uint32_t *n, VkExtensionProperties *p)
{
    ensureInit();
    PFN_vkEnumerateInstanceExtensionProperties f = realGIPA
        ? (PFN_vkEnumerateInstanceExtensionProperties)realGIPA(nullptr, "vkEnumerateInstanceExtensionProperties") : nullptr;
    return f ? f(layer, n, p) : VK_ERROR_INITIALIZATION_FAILED;
}
extern "C" __declspec(dllexport)
VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(uint32_t *n, VkLayerProperties *p)
{
    ensureInit();
    PFN_vkEnumerateInstanceLayerProperties f = realGIPA
        ? (PFN_vkEnumerateInstanceLayerProperties)realGIPA(nullptr, "vkEnumerateInstanceLayerProperties") : nullptr;
    return f ? f(n, p) : VK_ERROR_INITIALIZATION_FAILED;
}
extern "C" __declspec(dllexport)
VkResult VKAPI_CALL vkEnumerateInstanceVersion(uint32_t *v)
{
    ensureInit();
    PFN_vkEnumerateInstanceVersion f = realGIPA
        ? (PFN_vkEnumerateInstanceVersion)realGIPA(nullptr, "vkEnumerateInstanceVersion") : nullptr;
    if (!f) { if (v) *v = VK_API_VERSION_1_0; return VK_SUCCESS; }
    return f(v);
}

// Present through Streamline so sl.common's presentCommon() runs every frame.
// SL resolves the "real" Vulkan from vulkan-1.dll - which is us - so a thread
// guard sends its own downstream call straight to the loader instead of back
// into this hook.
extern "C" __declspec(dllexport)
VkResult VKAPI_CALL vkQueuePresentKHR(VkQueue queue, const VkPresentInfoKHR *pi)
{
    ensureInit();
    if (g_slOn && g_slHanded && slPresentFn && !t_inSlPresent) {
        t_inSlPresent = true;
        VkResult r = slPresentFn(queue, pi);
        t_inSlPresent = false;
        return r;
    }
    static PFN_vkQueuePresentKHR real = nullptr;
    if (!real && realGDPA && g_pendDev) real = (PFN_vkQueuePresentKHR)realGDPA(g_pendDev, "vkQueuePresentKHR");
    if (!real && realGIPA) real = (PFN_vkQueuePresentKHR)realGIPA(g_lastInstance, "vkQueuePresentKHR");
    return real ? real(queue, pi) : VK_ERROR_INITIALIZATION_FAILED;
}

BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH) armLayer();
    return TRUE;
}
