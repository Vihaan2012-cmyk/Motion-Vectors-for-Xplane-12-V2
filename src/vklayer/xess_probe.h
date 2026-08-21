// What XeSS needs from the Vulkan device, asked of XeSS itself.
//
// WHY A PROBE BEFORE A BACKEND
//
// xessVKCreateContext takes a VkDevice that was already created with the
// extensions and features XeSS requires. A layer does not get to go back and
// add them afterwards - by the time anything wants to upscale, vkCreateDevice
// has long returned. So the only place this decision can be made is inside our
// vkCreateDevice interception, before X-Plane's device exists, and the only
// honest way to make it is to ask XeSS what it wants rather than hardcode a
// list that goes stale when the SDK moves.
//
// That is the same lesson the Streamline block above it records: we were
// printing Streamline's own requirement list and then ignoring it, and Reflex
// failed with -229 for want of three extensions it had told us about.
//
// WHY THE DLL IS LOADED BY HAND
//
// libxess.lib is an MSVC import library and this layer is built with MinGW,
// but that is the smaller reason. The real one is that a Vulkan layer must
// survive its optional dependencies being absent: linking the import library
// makes libxess.dll a hard load-time requirement of the whole layer, so a user
// without it would lose motion vectors and TAA as well as XeSS. Resolved by
// hand, a missing DLL is exactly the no-library verdict the policy already has
// a name for.
//
// NOTHING HERE RUNS XeSS. This asks what it would need and reports the answer.
// Whether the answer is usable is a separate question, deliberately, because
// finding out that this GPU or this X-Plane cannot host XeSS is worth knowing
// on its own and is cheap to establish.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>
#include <vector>

#ifndef MV_HAVE_XESS_SDK
#define MV_HAVE_XESS_SDK 0
#endif

#if MV_HAVE_XESS_SDK
#include <xess/xess_vk.h>
#endif

namespace xessprobe {

struct Requirements {
    bool loaded  = false;   // the DLL is present and its symbols resolved
    bool queried = false;   // the query itself returned success
    std::vector<std::string> deviceExts;
    // XeSS owns this chain and patches it; it must be threaded into
    // VkDeviceCreateInfo::pNext, never freed by us.
    void *featureChain = nullptr;
    uint32_t major = 0, minor = 0, patch = 0;
    // Set on every failure path. A probe that fails silently is worse than no
    // probe, because the backend then reports "unavailable" with no reason.
    const char *why = "not attempted";
};

inline Requirements &state() { static Requirements r; return r; }

#if MV_HAVE_XESS_SDK

typedef xess_result_t (*PFN_xessGetVersion)(xess_version_t *);
typedef xess_result_t (*PFN_xessVKGetRequiredDeviceExtensions)(
    VkInstance, VkPhysicalDevice, uint32_t *, const char *const **);
typedef xess_result_t (*PFN_xessVKGetRequiredDeviceFeatures)(
    VkInstance, VkPhysicalDevice, void **);

struct Fns {
    HMODULE dll = nullptr;
    PFN_xessGetVersion getVersion = nullptr;
    PFN_xessVKGetRequiredDeviceExtensions getDeviceExts = nullptr;
    PFN_xessVKGetRequiredDeviceFeatures   getDeviceFeatures = nullptr;
};

inline Fns &fns() { static Fns f; return f; }

// Looked for beside the layer DLL first, then on the normal search path. Beside
// the layer is where the installer puts it, and relying on the search path
// alone would silently pick up whatever other copy a different application had
// already loaded.
inline bool load()
{
    Requirements &r = state();
    if (r.loaded) return true;
    Fns &f = fns();
    if (f.dll) return true;

    char self[MAX_PATH] = {0};
    HMODULE me = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)&load, &me) && me &&
        GetModuleFileNameA(me, self, MAX_PATH)) {
        std::string dir(self);
        const size_t cut = dir.find_last_of("\\/");
        if (cut != std::string::npos) {
            const std::string beside = dir.substr(0, cut + 1) + "libxess.dll";
            f.dll = LoadLibraryA(beside.c_str());
        }
    }
    if (!f.dll) f.dll = LoadLibraryA("libxess.dll");
    if (!f.dll) { r.why = "libxess.dll not found"; return false; }

    f.getVersion = (PFN_xessGetVersion)
        GetProcAddress(f.dll, "xessGetVersion");
    f.getDeviceExts = (PFN_xessVKGetRequiredDeviceExtensions)
        GetProcAddress(f.dll, "xessVKGetRequiredDeviceExtensions");
    f.getDeviceFeatures = (PFN_xessVKGetRequiredDeviceFeatures)
        GetProcAddress(f.dll, "xessVKGetRequiredDeviceFeatures");

    if (!f.getDeviceExts || !f.getDeviceFeatures) {
        // A DLL that loads but has no Vulkan entry points is a DX-only or
        // older build. Naming that is the difference between "install XeSS" and
        // "install a different XeSS".
        r.why = "libxess.dll has no Vulkan entry points";
        FreeLibrary(f.dll);
        f.dll = nullptr;
        return false;
    }

    if (f.getVersion) {
        xess_version_t v;
        memset(&v, 0, sizeof(v));
        if (f.getVersion(&v) == XESS_RESULT_SUCCESS) {
            r.major = v.major; r.minor = v.minor; r.patch = v.patch;
        }
    }
    r.loaded = true;
    r.why = "loaded, not yet queried";
    return true;
}

// Asks XeSS what a device must carry. Returns false and leaves `why` set on
// every failure path, including the ones that are not errors so much as
// answers - an older GPU that XeSS declines is a legitimate outcome.
inline bool query(VkInstance inst, VkPhysicalDevice phys)
{
    Requirements &r = state();
    if (r.queried) return true;
    if (!load()) return false;
    if (!inst || !phys) { r.why = "no instance or physical device"; return false; }

    Fns &f = fns();

    uint32_t n = 0;
    const char *const *names = nullptr;
    const xess_result_t re = f.getDeviceExts(inst, phys, &n, &names);
    if (re != XESS_RESULT_SUCCESS) {
        // Kept as a number rather than mapped to a string: the mapping would be
        // one more table to go stale, and the numeric code is what the XeSS
        // documentation is indexed by.
        static char buf[96];
        snprintf(buf, sizeof(buf),
                 "xessVKGetRequiredDeviceExtensions returned %d", (int)re);
        r.why = buf;
        return false;
    }
    for (uint32_t i = 0; i < n && names; ++i)
        if (names[i]) r.deviceExts.push_back(names[i]);

    // Null in, chain out: XeSS builds a fresh chain we then merge. Passing our
    // own chain would have it patch structures X-Plane owns.
    void *chain = nullptr;
    const xess_result_t rf = f.getDeviceFeatures(inst, phys, &chain);
    if (rf != XESS_RESULT_SUCCESS) {
        static char buf[96];
        snprintf(buf, sizeof(buf),
                 "xessVKGetRequiredDeviceFeatures returned %d", (int)rf);
        r.why = buf;
        return false;
    }
    r.featureChain = chain;

    r.queried = true;
    r.why = "";
    return true;
}

#else  // no SDK in this build

inline bool load()  { state().why = "built without the XeSS SDK"; return false; }
inline bool query(VkInstance, VkPhysicalDevice)
{ state().why = "built without the XeSS SDK"; return false; }

#endif

} // namespace xessprobe
