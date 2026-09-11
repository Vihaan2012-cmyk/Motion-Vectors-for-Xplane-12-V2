// nn_capture_fwd.h - the capture harness's interface, declared early so
// taa.h (included before most of the layer's globals) can call it. The
// definitions live in nn_capture.h, included further down layer.cpp once
// g_velSnap, g_colorImages, g_viewToImage and friends exist.
//
// Copyright (C) 2026 MotionVectors contributors. SPDX: GPL-3.0-or-later
#pragma once
#include <vulkan/vulkan.h>
#include <stdint.h>

struct DeviceData;

namespace nncap {

// What the resolve hands over each frame: the images it just read/wrote, in
// the layouts they are in at that point of the command buffer. VK_NULL_HANDLE
// = not available this frame (the plane is simply omitted).
struct ResolvePlanes {
    VkImage       scene = VK_NULL_HANDLE;     VkImageLayout sceneLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkImage       velocity = VK_NULL_HANDLE;  VkFormat velocityFormat = VK_FORMAT_UNDEFINED;
                                              VkImageLayout velocityLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkImage       depth = VK_NULL_HANDLE;     VkImageLayout depthLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkImage       normal = VK_NULL_HANDLE;    VkFormat normalFormat = VK_FORMAT_UNDEFINED;
                                              VkImageLayout normalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkImage       resolved = VK_NULL_HANDLE;  VkFormat resolvedFormat = VK_FORMAT_UNDEFINED;
                                              VkImageLayout resolvedLayout = VK_IMAGE_LAYOUT_GENERAL;
    uint32_t      w = 0, h = 0, layers = 1;
};

// Record the copies into X-Plane's command buffer (called from the resolve,
// right after its dispatch). Cheap no-op unless nn.capture=1 and a frame is due.
void recordFromResolve(DeviceData &dd, VkCommandBuffer cb, const ResolvePlanes &p);
// Once per present: hand finished readbacks to the writer thread.
void poll();
// nn.capture=1 - the resolve records the depth copy for the harness even when
// taa.pos_harvest is off (the depth plane must be in every training tuple).
bool captureArmed();
// From Layer_CmdBeginRendering, with g_lock HELD: remember the G-buffer pass's
// colour attachments (albedo / material / emissive live among them).
void noteScenePass(VkCommandBuffer cb, const VkRenderingInfo *info);
void noteScenePassEnd(DeviceData &dd, VkCommandBuffer cb);   // copies the G-buffer planes while they are still intact

} // namespace nncap
