// Find out WHICH image X-Plane's FSR writes, by asking the GPU.
//
// ---- WHY THIS EXISTS.
//
// Anything that wants to hand X-Plane a finished upscaled frame - FSR3 above
// all, which is ten CPU-driven passes and must be given a VkImage to write -
// needs the handle of the image X-Plane's own upscale would have produced.
//
// That handle cannot be obtained by inspection. Measured, not assumed:
//
//   * thirteen storage images share the output's 3840x2160 extent, so size
//     cannot discriminate;
//   * vkUpdateDescriptorSets records ONE set per frame;
//   * vkCmdPushDescriptorSetKHR, the Vulkan 1.4 core name and the "2" form are
//     each called EXACTLY ZERO times;
//   * neither descriptor update templates nor descriptor buffers are enabled.
//
// Four rebuilds went into inferring around that before it was measured. The
// conclusion is that X-Plane simply does not expose it.
//
// ---- HOW THIS ANSWERS IT ANYWAY.
//
// Our substituted upscaler is ALREADY writing the correct image - that is the
// whole point of the substitution. So it stamps a sentinel into pixel (0,0),
// and this copies that one pixel out of every candidate into a host-visible
// buffer. Whichever slot carries the stamp names the image.
//
// One frame of work, one answer, cached for the life of the process.
//
// ---- WHY THERE IS NO FENCE.
//
// The copies go into X-Plane's own command buffer, and the results are read
// several frames later rather than waited on. A fence would mean either
// blocking the render thread or owning a synchronisation object across a
// submit this layer does not control. Waiting three frames costs nothing, is
// impossible to get wrong, and the answer is needed once.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <string.h>
#include <vector>

namespace fsrprobe {

// The value the shader stamps. Three independent improbable components: real
// content cannot forge all three at exactly (0,0) by accident.
static const float kSentinel[3] = { 0.111111f, 0.222222f, 0.333333f };

// rgba16f, so four halves per pixel.
static const uint32_t kPixelBytes = 8;
// More than the thirteen seen, because the count depends on the sim's settings
// and a probe that silently truncates its candidate list would report "not
// found" for the one case it was built for.
static const uint32_t kMaxCandidates = 64;

struct State {
    VkBuffer       buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void          *ptr = nullptr;
    VkDevice       device = VK_NULL_HANDLE;

    std::vector<VkImage> candidates;
    uint64_t copiedOnFrame = 0;
    bool     copiesRecorded = false;
    bool     failed  = false;

    // The answer.
    VkImage  output  = VK_NULL_HANDLE;
    bool     resolved = false;
};

inline State &state()
{
    static State s;
    return s;
}

// ---- HALF TO FLOAT.
//
// Only enough of it to compare against a sentinel: the values involved are
// ordinary normals, so denormals and infinities are handled by returning
// something that simply will not match rather than by being correct.
inline float halfToFloat(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h >> 15) & 1u;
    const uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    const uint32_t man  = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign << 31;                 // zero
        else          bits = sign << 31;                 // denormal: not a match
    } else if (exp == 31) {
        bits = (sign << 31) | 0x7F800000u | (man << 13); // inf/nan: not a match
    } else {
        bits = (sign << 31) | ((exp + 112u) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// Tolerance, because the sentinel is written as float and stored as half. The
// gap between neighbouring halves near 0.1 is about 6e-5, so 2e-3 accepts the
// rounding by a wide margin while remaining far tighter than the distance
// between the three components.
inline bool looksLikeSentinel(const uint8_t *px)
{
    uint16_t h[4];
    memcpy(h, px, sizeof(h));
    for (int i = 0; i < 3; ++i) {
        const float v = halfToFloat(h[i]);
        const float d = v - kSentinel[i];
        if (d > 2e-3f || d < -2e-3f) return false;
    }
    return true;
}

} // namespace fsrprobe
