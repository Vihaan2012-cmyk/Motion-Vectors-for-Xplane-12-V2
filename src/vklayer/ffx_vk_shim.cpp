// The Vulkan entry points FidelityFX calls DIRECTLY, answered by this layer.
//
// ---- WHY THIS FILE EXISTS. IT IS NOT A CONVENIENCE.
//
// ffx_vk.cpp does not resolve everything through the vkGetDeviceProcAddr it is
// given. A few instance-level queries it calls by name:
//
//     vkEnumerateDeviceExtensionProperties   (sizing the scratch buffer)
//     vkGetPhysicalDeviceProperties2
//     vkGetPhysicalDeviceFeatures2
//
// Linked against the loader's import library those names resolve to the
// LOADER'S exported functions. Calling one of those from inside a layer
// re-enters the dispatch chain FROM THE TOP - and this layer hooks
// vkEnumerateDeviceExtensionProperties, so the call comes straight back into
// us and goes round again.
//
// That is what killed the sim: FSR3 died inside ffxGetScratchMemorySizeVK, its
// very first call, with no output at all. A stack overflow produces exactly
// that - no error, no return, nothing written.
//
// ---- HOW THIS FIXES IT.
//
// These objects are linked into the layer, so defining the symbols HERE means
// FFX's references bind to these rather than to the loader's exports. Each one
// forwards to the NEXT LAYER's pointer - the same table the rest of this layer
// uses - so the call continues DOWN the chain instead of restarting at the top.
//
// The loader import library must not be linked alongside this, or the duplicate
// definitions decide the winner by link order rather than by intent.
//
// ---- WHY NOT PATCH THE SDK INSTEAD.
//
// Editing vendored AMD source means re-editing it on every SDK update, and the
// edit would have to thread a proc-addr through functions whose signatures do
// not take one. Three forwarders that the linker prefers is smaller, needs no
// upstream change, and states the reason in one place.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <vulkan/vulkan.h>

// Defined in layer.cpp, bound once from the application's instance. The
// binding is deliberately done ONCE there: this project has already been taken
// down by a physical-device function resolved from a probe's instance and then
// called with the application's physical device, which the loader answers by
// __fastfail rather than by an error.
extern "C" {
PFN_vkEnumerateDeviceExtensionProperties mvNextEnumDeviceExtensionProperties();
PFN_vkGetPhysicalDeviceProperties2       mvNextGetPhysicalDeviceProperties2();
PFN_vkGetPhysicalDeviceFeatures2         mvNextGetPhysicalDeviceFeatures2();
PFN_vkGetPhysicalDeviceProperties        mvNextGetPhysicalDeviceProperties();
PFN_vkGetPhysicalDeviceMemoryProperties  mvNextGetPhysicalDeviceMemoryProperties();
PFN_vkGetPhysicalDeviceFeatures          mvNextGetPhysicalDeviceFeatures();
// Device-level: resolved per device through the NEXT layer's table, which is
// the whole point - a device function taken from the loader would restart the
// dispatch chain rather than continue it.
PFN_vkVoidFunction mvNextDeviceProcAddr(VkDevice device, const char *name);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(
    VkPhysicalDevice physicalDevice, const char *pLayerName,
    uint32_t *pPropertyCount, VkExtensionProperties *pProperties)
{
    PFN_vkEnumerateDeviceExtensionProperties next = mvNextEnumDeviceExtensionProperties();
    if (!next) {
        // Reporting zero extensions is safe here: the only caller that reaches
        // this is FFX sizing an array, and a zero-length array is merely a
        // smaller scratch buffer, not a wrong one.
        if (pPropertyCount) *pPropertyCount = 0;
        return VK_SUCCESS;
    }
    return next(physicalDevice, pLayerName, pPropertyCount, pProperties);
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties2(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2 *pProperties)
{
    PFN_vkGetPhysicalDeviceProperties2 next = mvNextGetPhysicalDeviceProperties2();
    if (next) next(physicalDevice, pProperties);
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2 *pFeatures)
{
    PFN_vkGetPhysicalDeviceFeatures2 next = mvNextGetPhysicalDeviceFeatures2();
    if (next) next(physicalDevice, pFeatures);
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties *pProperties)
{
    PFN_vkGetPhysicalDeviceProperties next = mvNextGetPhysicalDeviceProperties();
    if (next) next(physicalDevice, pProperties);
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties *pProperties)
{
    PFN_vkGetPhysicalDeviceMemoryProperties next = mvNextGetPhysicalDeviceMemoryProperties();
    if (next) next(physicalDevice, pProperties);
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures(
    VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures *pFeatures)
{
    PFN_vkGetPhysicalDeviceFeatures next = mvNextGetPhysicalDeviceFeatures();
    if (next) next(physicalDevice, pFeatures);
}

extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(
    VkDevice device, const char *pName)
{
    return mvNextDeviceProcAddr(device, pName);
}

// FFX creates one staging buffer directly rather than through the table it was
// given. Resolved down the chain like everything else here.
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(
    VkDevice device, const VkBufferCreateInfo *pCreateInfo,
    const VkAllocationCallbacks *pAllocator, VkBuffer *pBuffer)
{
    PFN_vkCreateBuffer next =
        (PFN_vkCreateBuffer)mvNextDeviceProcAddr(device, "vkCreateBuffer");
    if (!next) return VK_ERROR_INITIALIZATION_FAILED;
    return next(device, pCreateInfo, pAllocator, pBuffer);
}
