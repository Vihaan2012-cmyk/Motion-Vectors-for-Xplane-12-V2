// vk_util.h - small helpers shared by every pass in the layer.
//
// These lived in velocity_pass.h, which was the depth-derived velocity pass:
// it reconstructed per-pixel motion from depth plus the two most recent camera
// matrices. That pass is gone. The SPIR-V injection emits true motion vectors
// from the real clip positions during the scene draw, which is exact where
// reprojection is an approximation, and the compute dispatch it replaced was
// the confirmed cause of the stutter - a full-resolution pass plus barriers
// transitioning X-Plane's depth image every frame, forcing a pipeline flush
// that queued texture uploads behind it.
//
// Deleting the file meant rehoming three things it happened to own. Nothing
// here is velocity-specific; they are here because a memory-type lookup and a
// half-float decode are needed by anything that allocates an image or reads one
// back, which is all four remaining passes.

#pragma once

#include <cstdint>

// Depth convention, MEASURED rather than assumed.
//
// 1 = reverse-Z, which is what this install uses. Kept as a global because the
// resolve, FSR2 and DLSS-FG all have to agree with whatever the scene actually
// renders, and a disagreement here does not fail loudly - it produces an image
// that will not settle, which is indistinguishable from a badly tuned history
// clamp until someone thinks to check.
static int g_depthFlip = 1;

// First memory type satisfying `want`. UINT32_MAX when the device offers none,
// which every caller must treat as a failure rather than binding type 0.
static uint32_t velFindMemType(const VkPhysicalDeviceMemoryProperties *mp,
                               uint32_t typeBits, VkMemoryPropertyFlags want)
{
    for (uint32_t i = 0; i < mp->memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) &&
            (mp->memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

// IEEE half -> float. The velocity and history targets are RGBA16F, so any
// readback that gets inspected on the CPU has to come back through this.
static float velHalfToFloat(uint16_t h)
{
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) { f = sign << 31; }                 // +/- zero
        else {                                             // subnormal
            exp = 127 - 15 + 1;
            while (!(mant & 0x400u)) { mant <<= 1; --exp; }
            mant &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (mant << 13);     // inf / nan
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }

    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

// ---- MARK AN ALLOCATION AS OURS, AND THEREFORE EXPENDABLE.
//
// VK_EXT_memory_priority takes a float from 0 to 1, default 0.5. Everything the
// layer allocates for itself - the velocity target, the resolve history, an
// upscaler's output - is reconstructible from the next frame, while X-Plane's
// textures are not: losing one costs a frame, losing the other costs the
// picture until the pager claws it back.
//
// So ours go low. Under pressure the driver demotes these to system RAM first
// and the sim's textures stay resident, which is the ordering the pager
// collapse showed we were not getting.
//
// Silently ignored unless the memoryPriority FEATURE was enabled at device
// creation, which is why g_memoryPriority is checked rather than assumed.
extern bool g_memoryPriority;

static inline void velLowPriority(VkMemoryAllocateInfo &mai,
                                  VkMemoryPriorityAllocateInfoEXT &prio,
                                  float value = 0.25f)
{
    if (!g_memoryPriority) return;
    memset(&prio, 0, sizeof(prio));
    prio.sType = VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT;
    prio.priority = value;
    prio.pNext = mai.pNext;
    mai.pNext = &prio;
}

// One image barrier, spelled out once. Layouts are tracked rather than assumed:
// naming an oldLayout the image is not actually in is undefined behaviour, and
// a driver is entitled to discard the contents.
//
// Lifted here from the predecessor's resolve pass, which this project does not
// carry - the barrier itself has nothing to do with upscaling.
static void resBarrier(DeviceData &dd, VkCommandBuffer cb, VkImage img,
                       VkImageLayout from, VkImageLayout to,
                       VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                       VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage)
{
    VkImageMemoryBarrier bar;
    memset(&bar, 0, sizeof(bar));
    bar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    bar.oldLayout = from;
    bar.newLayout = to;
    bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.image = img;
    bar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bar.subresourceRange.levelCount = 1;
    bar.subresourceRange.layerCount = 1;
    bar.srcAccessMask = srcAccess;
    bar.dstAccessMask = dstAccess;
    dd.cmdPipelineBarrier(cb, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &bar);
}
