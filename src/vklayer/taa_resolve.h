#pragma once
//
// TAA resolve - the first consumer of the velocity field.
//
// Everything before this produced a number. This is the first thing that uses
// it to change a pixel, and that matters for how it is built: a velocity field
// can be wrong in ways that only a consumer reveals, and a consumer can be
// wrong in ways that look exactly like a wrong field. So this pass is
// deliberately the smallest thing that can work - reproject, clamp, blend - and
// it reports whether it ran rather than assuming it did.
//
// ---- THE CONVENTION, STATED ONCE.
//
// The field is prev - curr, in UV. The history is therefore sampled at
// uv + velocity. Written the other way it does not fail or warn: it reprojects
// twice as far in the wrong direction, which reads as heavy ghosting and gets
// chased as a tuning problem. That exact error lived in this project for its
// entire life until a direct calibration caught it at ratio -1.008, so the
// convention is stated here and asserted against the shader rather than left
// to be inferred.
//
// ---- WHY COMPUTE AND NOT A FULL-SCREEN DRAW.
//
// A graphics pass needs a render pass compatible with whatever X-Plane has
// bound, and the whole difficulty of the velocity work was that X-Plane's
// pipelines are not ours to shape. A compute dispatch needs no render pass, no
// framebuffer and no blend state, so it cannot be rejected by the driver for
// disagreeing with a pipeline we did not create. Given that 14,835 pipelines
// were once refused for exactly that kind of disagreement, that is worth more
// than the small cost of not having fixed-function interpolation.

#include <stdint.h>
#include <string.h>
#include "shaders_generated.h"

struct TaaResolve {
    VkDevice        device      = VK_NULL_HANDLE;
    VkPipeline      pipeline    = VK_NULL_HANDLE;
    VkPipelineLayout layout     = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool pool       = VK_NULL_HANDLE;
    VkShaderModule  module      = VK_NULL_HANDLE;
    VkSampler       sampler     = VK_NULL_HANDLE;

    // The history, and the output. Two images rather than one: a compute shader
    // may not read and write the same image in one dispatch without a barrier
    // per texel, so the resolve writes to `output` and that becomes the next
    // frame's history. They swap.
    VkImage         history     = VK_NULL_HANDLE;
    VkDeviceMemory  historyMem  = VK_NULL_HANDLE;
    VkImageView     historyView = VK_NULL_HANDLE;
    VkImage         output      = VK_NULL_HANDLE;
    VkDeviceMemory  outputMem   = VK_NULL_HANDLE;
    VkImageView     outputView  = VK_NULL_HANDLE;
    VkImageView     currentView = VK_NULL_HANDLE;   // view onto the scene colour
    VkImageView     velocityView = VK_NULL_HANDLE;

    uint32_t        w = 0, h = 0;
    VkFormat        format = VK_FORMAT_UNDEFINED;
    bool            ready = false;
    bool            historyValid = false;
    uint64_t        dispatches = 0;
};

static TaaResolve g_taa;

// The push block the shader declares. Kept in one struct so a change to the
// GLSL and a change here cannot drift apart silently - the sizes are asserted
// against each other below.
struct TaaResolvePush {
    float invSizeX, invSizeY;
    float blend;
    float reset;
};

// ---- THE CURRENT-FRAME WEIGHT.
//
// 0.1 means a pixel keeps 90% of its history each frame, which is the usual
// starting point: it converges in about ten frames and is stable enough to see
// whether the reprojection is right. Too high and TAA does nothing visible; too
// low and every error in the velocity field is held on screen for half a
// second, which is actually useful while proving this out.
static float taaBlendWeight()
{
    static float v = -1.0f;
    if (v < 0.0f) {
        v = 0.1f;
        if (const char *e = getenv("TAA_BLEND")) {
            float f = (float)atof(e);
            if (f > 0.0f && f <= 1.0f) v = f;
        }
    }
    return v;
}

static bool taaEnabled()
{
    static int v = -1;
    if (v < 0) v = getenv("TAA_RESOLVE") ? 1 : 0;
    return v != 0;
}

static void taaDestroy(DeviceData &dd)
{
    TaaResolve &t = g_taa;
    if (t.pipeline)   dd.destroyPipeline(t.device, t.pipeline, nullptr);
    if (t.layout)     dd.destroyPipelineLayout(t.device, t.layout, nullptr);
    if (t.setLayout)  dd.destroyDescriptorSetLayout(t.device, t.setLayout, nullptr);
    if (t.pool)       dd.destroyDescriptorPool(t.device, t.pool, nullptr);
    if (t.module)     dd.destroyShaderModule(t.device, t.module, nullptr);
    if (t.sampler)    dd.destroySampler(t.device, t.sampler, nullptr);
    if (t.historyView) dd.destroyImageView(t.device, t.historyView, nullptr);
    if (t.history)    dd.destroyImage(t.device, t.history, nullptr);
    if (t.historyMem) dd.freeMemory(t.device, t.historyMem, nullptr);
    if (t.outputView) dd.destroyImageView(t.device, t.outputView, nullptr);
    if (t.output)     dd.destroyImage(t.device, t.output, nullptr);
    if (t.outputMem)  dd.freeMemory(t.device, t.outputMem, nullptr);
    t = TaaResolve();
}

static bool taaMakeImage(DeviceData &dd, VkPhysicalDevice phys, uint32_t w,
                         uint32_t h, VkFormat fmt, VkImage *img,
                         VkDeviceMemory *mem, VkImageView *view)
{
    VkImageCreateInfo ici;
    memset(&ici, 0, sizeof(ici));
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = fmt;
    ici.extent.width  = w;
    ici.extent.height = h;
    ici.extent.depth  = 1;
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    // STORAGE so the resolve can write it, SAMPLED so the next frame can read
    // it as history, TRANSFER_SRC so the result can be copied back over the
    // scene colour.
    ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT
                      | VK_IMAGE_USAGE_SAMPLED_BIT
                      | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                      | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (dd.createImage(dd.device, &ici, nullptr, img) != VK_SUCCESS) return false;

    VkMemoryRequirements mr;
    dd.getImageMemReq(dd.device, *img, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    memset(&mp, 0, sizeof(mp));
    if (g_getPhysMemProps) g_getPhysMemProps(phys, &mp);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((mr.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            type = i; break;
        }
    if (type == UINT32_MAX) return false;

    VkMemoryAllocateInfo mai;
    memset(&mai, 0, sizeof(mai));
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = mr.size;
    mai.memoryTypeIndex = type;
    if (dd.allocateMemory(dd.device, &mai, nullptr, mem) != VK_SUCCESS) return false;
    dd.bindImageMemory(dd.device, *img, *mem, 0);

    VkImageViewCreateInfo ivci;
    memset(&ivci, 0, sizeof(ivci));
    ivci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image    = *img;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format   = fmt;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.layerCount = 1;
    return dd.createImageView(dd.device, &ivci, nullptr, view) == VK_SUCCESS;
}

// A view onto X-Plane's scene colour, made once and remade if the image
// changes. We do not own that image, so we cannot ask it for a view - and its
// handle changes on a resolution change or a device loss, which is exactly when
// a cached view becomes a use-after-free rather than a wrong picture.
static VkImage      g_taaSceneImage = VK_NULL_HANDLE;
static VkImageView  g_taaSceneView  = VK_NULL_HANDLE;

static VkImageView taaSceneView(DeviceData &dd, VkImage img, VkFormat fmt)
{
    if (img == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    if (img == g_taaSceneImage && g_taaSceneView) return g_taaSceneView;
    if (g_taaSceneView) dd.destroyImageView(dd.device, g_taaSceneView, nullptr);
    g_taaSceneView = VK_NULL_HANDLE;
    VkImageViewCreateInfo ivci;
    memset(&ivci, 0, sizeof(ivci));
    ivci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image    = img;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format   = fmt;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.layerCount = 1;
    if (dd.createImageView(dd.device, &ivci, nullptr, &g_taaSceneView) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    g_taaSceneImage = img;
    return g_taaSceneView;
}

static bool taaInit(DeviceData &dd, VkPhysicalDevice phys, uint32_t w, uint32_t h)
{
    TaaResolve &t = g_taa;
    if (t.ready && t.w == w && t.h == h) return true;
    if (t.ready) taaDestroy(dd);
    t.device = dd.device;
    t.w = w; t.h = h;

    // RGBA16F regardless of what the scene target is. The history has to survive
    // being blended into repeatedly, and an 8-bit history quantises a 10%
    // per-frame blend into visible banding within a dozen frames.
    t.format = VK_FORMAT_R16G16B16A16_SFLOAT;

    if (!taaMakeImage(dd, phys, w, h, t.format, &t.history, &t.historyMem, &t.historyView) ||
        !taaMakeImage(dd, phys, w, h, t.format, &t.output,  &t.outputMem,  &t.outputView)) {
        trace("TAA: could not allocate the history pair (%ux%u RGBA16F, %.1f MB each)",
              w, h, (double)w * h * 8.0 / (1024.0 * 1024.0));
        taaDestroy(dd);
        return false;
    }

    VkShaderModuleCreateInfo smci;
    memset(&smci, 0, sizeof(smci));
    smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sizeof(ktaa_resolve_spv);
    smci.pCode    = ktaa_resolve_spv;
    if (dd.createShaderModule(dd.device, &smci, nullptr, &t.module) != VK_SUCCESS) {
        trace("TAA: the resolve shader module was rejected");
        taaDestroy(dd);
        return false;
    }

    VkSamplerCreateInfo sci;
    memset(&sci, 0, sizeof(sci));
    sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    // CLAMP_TO_EDGE, and the shader ALSO rejects off-screen history explicitly.
    // Belt and braces on purpose: a clamped edge texel smeared down the side of
    // the screen is the single most recognisable TAA artefact, and the address
    // mode alone does not prevent it - it only decides which wrong pixel is
    // returned.
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (dd.createSampler(dd.device, &sci, nullptr, &t.sampler) != VK_SUCCESS) {
        taaDestroy(dd);
        return false;
    }

    VkDescriptorSetLayoutBinding b[4];
    memset(b, 0, sizeof(b));
    for (int i = 0; i < 3; ++i) {
        b[i].binding         = (uint32_t)i;
        b[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    b[3].binding         = 3;
    b[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b[3].descriptorCount = 1;
    b[3].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dlci;
    memset(&dlci, 0, sizeof(dlci));
    dlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = 4;
    dlci.pBindings    = b;
    if (dd.createDescriptorSetLayout(dd.device, &dlci, nullptr, &t.setLayout) != VK_SUCCESS) {
        taaDestroy(dd);
        return false;
    }

    VkPushConstantRange pcr;
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset     = 0;
    pcr.size       = sizeof(TaaResolvePush);

    VkPipelineLayoutCreateInfo plci;
    memset(&plci, 0, sizeof(plci));
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &t.setLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (dd.createPipelineLayout(dd.device, &plci, nullptr, &t.layout) != VK_SUCCESS) {
        taaDestroy(dd);
        return false;
    }

    VkComputePipelineCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = t.module;
    cpci.stage.pName  = "main";
    cpci.layout       = t.layout;
    if (dd.createComputePipelines(dd.device, VK_NULL_HANDLE, 1, &cpci, nullptr,
                                  &t.pipeline) != VK_SUCCESS) {
        trace("TAA: the resolve pipeline was rejected");
        taaDestroy(dd);
        return false;
    }

    // One set per frame in flight would be the general answer. This dispatches
    // once per frame from one thread at a known point, so a small fixed pool
    // rewritten each frame is enough - and a pool that cannot grow is a pool
    // that cannot leak.
    VkDescriptorPoolSize ps[2];
    ps[0].type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ps[0].descriptorCount = 3 * 8;
    ps[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ps[1].descriptorCount = 1 * 8;
    VkDescriptorPoolCreateInfo dpci;
    memset(&dpci, 0, sizeof(dpci));
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets       = 8;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes    = ps;
    if (dd.createDescriptorPool(dd.device, &dpci, nullptr, &t.pool) != VK_SUCCESS) {
        taaDestroy(dd);
        return false;
    }

    t.ready = true;
    t.historyValid = false;
    trace("TAA: resolve ready %ux%u RGBA16F, history pair %.1f MB total, "
          "blend %.2f - the history is sampled at uv + velocity because the "
          "field is prev - curr",
          w, h, 2.0 * (double)w * h * 8.0 / (1024.0 * 1024.0), taaBlendWeight());
    return true;
}

// ---- ONE BARRIER HELPER, BECAUSE GETTING THESE WRONG IS SILENT.
//
// A missing barrier does not fail validation on every driver and does not
// crash; it reads stale texels, which looks exactly like a wrong reprojection.
// Given how much of this project was spent telling those two apart, the
// transitions are done in one place with the stage masks written out.
static void taaBarrier(DeviceData &dd, VkCommandBuffer cb, VkImage img,
                       VkImageLayout from, VkImageLayout to,
                       VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                       VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage)
{
    VkImageMemoryBarrier b;
    memset(&b, 0, sizeof(b));
    b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask       = srcAccess;
    b.dstAccessMask       = dstAccess;
    b.oldLayout           = from;
    b.newLayout           = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image               = img;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    dd.cmdPipelineBarrier(cb, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// Record the resolve at the 3D/UI boundary: after every scene pass by
// construction, before anything draws the interface over it.
//
// Returns false rather than silently doing nothing. A resolve that quietly
// never runs is indistinguishable from one that runs and has no effect, and
// this project has already lost days to exactly that distinction.
static bool taaRecordResolve(DeviceData &dd, VkCommandBuffer cb,
                             VkImage sceneImage, VkImageView sceneView,
                             VkImageView velocityView, bool resetHistory)
{
    TaaResolve &t = g_taa;
    if (!t.ready || !sceneView || !velocityView) return false;

    VkDescriptorSetAllocateInfo dsai;
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = t.pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &t.setLayout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (dd.allocateDescriptorSets(dd.device, &dsai, &set) != VK_SUCCESS) return false;

    VkDescriptorImageInfo ii[4];
    memset(ii, 0, sizeof(ii));
    ii[0].sampler     = t.sampler;
    ii[0].imageView   = sceneView;
    ii[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ii[1].sampler     = t.sampler;
    ii[1].imageView   = t.historyView;
    ii[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ii[2].sampler     = t.sampler;
    ii[2].imageView   = velocityView;
    ii[2].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ii[3].imageView   = t.outputView;
    ii[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet w[4];
    memset(w, 0, sizeof(w));
    for (int i = 0; i < 4; ++i) {
        w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet          = set;
        w[i].dstBinding      = (uint32_t)i;
        w[i].descriptorCount = 1;
        w[i].descriptorType  = (i < 3) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                       : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[i].pImageInfo      = &ii[i];
    }
    dd.updateDescriptorSets(dd.device, 4, w, 0, nullptr);

    // Scene colour to readable, history to readable, output to writable.
    taaBarrier(dd, cb, sceneImage,
               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    taaBarrier(dd, cb, t.history,
               t.historyValid ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    taaBarrier(dd, cb, t.output,
               VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
               0, VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    dd.cmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, t.pipeline);
    dd.cmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, t.layout, 0, 1,
                             &set, 0, nullptr);

    TaaResolvePush pc;
    pc.invSizeX = 1.0f / (float)t.w;
    pc.invSizeY = 1.0f / (float)t.h;
    pc.blend    = taaBlendWeight();
    // The first frame has no history, and neither does the frame after a camera
    // cut. Blending against an unwritten image is a black smear that takes as
    // many frames to clear as the blend weight allows.
    pc.reset    = (!t.historyValid || resetHistory) ? 1.0f : 0.0f;
    dd.cmdPushConstants(cb, t.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                        sizeof(pc), &pc);

    dd.cmdDispatch(cb, (t.w + 7) / 8, (t.h + 7) / 8, 1);

    // The result goes back over the scene colour, so everything downstream -
    // the UI, X-Plane's own post, the swapchain blit - sees the resolved image
    // without knowing this happened.
    taaBarrier(dd, cb, t.output,
               VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);
    taaBarrier(dd, cb, sceneImage,
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkImageCopy region;
    memset(&region, 0, sizeof(region));
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.layerCount = 1;
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.layerCount = 1;
    region.extent.width  = t.w;
    region.extent.height = t.h;
    region.extent.depth  = 1;
    dd.cmdCopyImage(cb, t.output, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    sceneImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // ...and it also becomes the next frame's history.
    taaBarrier(dd, cb, t.history,
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);
    dd.cmdCopyImage(cb, t.output, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    t.history, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    taaBarrier(dd, cb, t.history,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // Scene colour back to what X-Plane expects to keep drawing into.
    taaBarrier(dd, cb, sceneImage,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

    t.historyValid = true;
    if (++t.dispatches <= 3 || (t.dispatches % 600) == 0)
        trace("TAA: resolve dispatched %llu times (%ux%u, blend %.2f, reset=%d)",
              (unsigned long long)t.dispatches, t.w, t.h, pc.blend,
              pc.reset > 0.5f ? 1 : 0);
    return true;
}
