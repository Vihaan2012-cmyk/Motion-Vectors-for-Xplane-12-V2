// The DLSS probe, implemented.
//
// Header-only and included once by layer.cpp, matching how every other piece of
// this layer is built.
//
// ---- THE SIGNATURES ARE READ FROM THE SDK, NOT REMEMBERED.
//
//   NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
//       const NVSDK_NGX_FeatureDiscoveryInfo *, uint32_t *, VkExtensionProperties **)
//
//   NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
//       VkInstance, VkPhysicalDevice,
//       const NVSDK_NGX_FeatureDiscoveryInfo *, uint32_t *, VkExtensionProperties **)
//
// Both are documented as callable BEFORE the object they precede exists, which
// is the only reason a layer can enable DLSS on an application that never asked
// for it.
//
// ---- RESOLVED BY HAND, FOR THE REASON THE XeSS PROBE GIVES.
//
// Linking the NGX import library would make nvngx_dlss.dll a load-time
// dependency of the whole layer, so a user without it would lose motion vectors
// and TAA as well as DLSS. Resolved by hand, a missing DLL is exactly the
// "no-library" verdict the availability policy already has a name for.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "dlss_probe.h"

#include <windows.h>
#include <vulkan/vulkan.h>
#include <stdint.h>

namespace dlssprobe {

// ---- ONLY WHAT IS CALLED IS DECLARED.
//
// The NGX headers pull in a great deal and are built for MSVC; this layer is
// MinGW. Two function pointers and two small structs are the whole surface
// needed to ask the two questions, so they are declared here rather than
// dragging the SDK into the layer's translation unit.
//
// The values come from nvsdk_ngx_defs.h and are checked against it by
// tools/dlss_check.cpp rather than trusted.
enum { NGX_FEATURE_SUPERSAMPLING = 1 };
enum { NGX_RESULT_SUCCESS = 1 };

struct NgxVersionedStruct {          // NVSDK_NGX_Version is a plain enum value
    uint32_t sdkVersion;
    uint32_t featureId;
    // NVSDK_NGX_Application_Identifier: a union of an app id and a project
    // description, tagged by IdentifierType. Laid out here as bytes because
    // only the tag and the id are set, and mis-declaring the union would be a
    // silent ABI error rather than a compile one.
    uint32_t identifierType;         // 0 = application id
    uint32_t pad0;
    uint64_t applicationId;
    uint64_t pad1[6];                // the project-description arm, unused
    const wchar_t *applicationDataPath;
    const void *featureInfo;
};

typedef uint32_t (__cdecl *PFN_InstanceExtReq)(
    const void *discovery, uint32_t *count, VkExtensionProperties **props);
typedef uint32_t (__cdecl *PFN_DeviceExtReq)(
    VkInstance instance, VkPhysicalDevice pd,
    const void *discovery, uint32_t *count, VkExtensionProperties **props);

inline Requirements &state()
{
    static Requirements r;
    return r;
}

namespace detail {

inline HMODULE &lib()
{
    static HMODULE h = nullptr;
    return h;
}

// The discovery info both queries take. Filled once: the two calls must agree
// about which feature is being asked after, and building it twice is two places
// for them to disagree.
inline NgxVersionedStruct &discovery()
{
    static NgxVersionedStruct d;
    static bool init = false;
    if (!init) {
        init = true;
        memset(&d, 0, sizeof(d));
        d.sdkVersion    = 0x0000015; // NVSDK_NGX_Version_API
        d.featureId     = NGX_FEATURE_SUPERSAMPLING;
        d.identifierType = 0;
        d.applicationId = 0x4D56;    // 'MV' - this layer, not X-Plane
        d.applicationDataPath = L".";
        d.featureInfo   = nullptr;
    }
    return d;
}

inline void copyExts(VkExtensionProperties *props, uint32_t n,
                     std::vector<std::string> &out)
{
    out.clear();
    if (!props) return;
    for (uint32_t i = 0; i < n; ++i) {
        // extensionName is a fixed char array; it may be unterminated only if
        // the driver lied, so the length is bounded rather than trusted.
        char name[VK_MAX_EXTENSION_NAME_SIZE + 1];
        memcpy(name, props[i].extensionName, VK_MAX_EXTENSION_NAME_SIZE);
        name[VK_MAX_EXTENSION_NAME_SIZE] = 0;
        out.push_back(std::string(name));
    }
}

} // namespace detail

inline bool instanceStage(const char *sdkPath)
{
    Requirements &r = state();
    if (r.loaded) return r.queried;

    // ---- LOADED HERE, BEFORE THE CALL DESCENDS.
    //
    // The XeSS probe took X-Plane down inside vkCreateDevice, and one of the
    // two suspects was LoadLibrary of a very large DLL while the Vulkan loader
    // held its lock. nvngx_dlss.dll is 56 MB, so this happens at the TOP of
    // vkCreateInstance - inside our hook, but before anything nested.
    //
    // Traced before it runs, so if it dies here the log names the step rather
    // than leaving the same two suspects the XeSS crash left.
    trace("DLSS: loading NGX from '%s' (56 MB - if the sim dies on this line, "
          "it is the load and not the query)", sdkPath ? sdkPath : "(default)");

    std::string path = sdkPath ? sdkPath : "";
    if (!path.empty() && path[path.size() - 1] != '\\') path += "\\";
    path += "nvngx_dlss.dll";

    detail::lib() = LoadLibraryA(path.c_str());
    if (!detail::lib()) {
        r.why = "nvngx_dlss.dll did not load from '" + path + "'";
        trace("DLSS: %s - reporting no-library, which is a verdict and not a "
              "failure", r.why.c_str());
        return false;
    }
    r.loaded = true;

    PFN_InstanceExtReq q = (PFN_InstanceExtReq)GetProcAddress(
        detail::lib(), "NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements");
    if (!q) {
        r.why = "NGX loaded but GetFeatureInstanceExtensionRequirements is absent";
        trace("DLSS: %s", r.why.c_str());
        return false;
    }

    uint32_t n = 0;
    VkExtensionProperties *props = nullptr;
    trace("DLSS: asking NGX which INSTANCE extensions it needs");
    const uint32_t rc = q(&detail::discovery(), &n, &props);
    if (rc != NGX_RESULT_SUCCESS) {
        char buf[96];
        snprintf(buf, sizeof(buf), "instance extension query returned 0x%08x", rc);
        r.why = buf;
        trace("DLSS: %s", r.why.c_str());
        return false;
    }

    detail::copyExts(props, n, r.instanceExts);
    trace("DLSS: NGX wants %u instance extension(s)", n);
    for (size_t i = 0; i < r.instanceExts.size(); ++i)
        trace("DLSS:   %s", r.instanceExts[i].c_str());
    return true;
}

inline bool deviceStage(void *instance, void *physicalDevice)
{
    Requirements &r = state();
    if (!r.loaded) {
        if (r.why.empty()) r.why = "instanceStage never ran";
        return false;
    }

    PFN_DeviceExtReq q = (PFN_DeviceExtReq)GetProcAddress(
        detail::lib(), "NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements");
    if (!q) {
        r.why = "NGX loaded but GetFeatureDeviceExtensionRequirements is absent";
        trace("DLSS: %s", r.why.c_str());
        return false;
    }

    uint32_t n = 0;
    VkExtensionProperties *props = nullptr;
    trace("DLSS: asking NGX which DEVICE extensions it needs");
    const uint32_t rc = q((VkInstance)instance, (VkPhysicalDevice)physicalDevice,
                          &detail::discovery(), &n, &props);
    if (rc != NGX_RESULT_SUCCESS) {
        char buf[96];
        snprintf(buf, sizeof(buf), "device extension query returned 0x%08x", rc);
        r.why = buf;
        trace("DLSS: %s", r.why.c_str());
        return false;
    }

    detail::copyExts(props, n, r.deviceExts);
    r.queried = true;
    trace("DLSS: NGX wants %u device extension(s)", n);
    for (size_t i = 0; i < r.deviceExts.size(); ++i)
        trace("DLSS:   %s", r.deviceExts[i].c_str());

    // The verdict this fills in, stated in the availability report's own terms.
    trace("DLSS: requirements known. This says what DLSS NEEDS; whether the "
          "backend exists to use them is a separate question and still no.");
    return true;
}

} // namespace dlssprobe
