#pragma once
// ---- FRAME INTERPOLATION'S THREE INPUTS, WITHOUT AN UPSCALER.
//
// FFX frame interpolation consumes exactly three resources it does not produce:
// reconstructedPrevNearestDepth, dilatedDepth and dilatedMotionVectors. They are
// normally a by-product of the FSR3 upscaler's reconstruct-and-dilate pass, and
// FSR3 has no dilate-only entry point - so the entire upscaler was stood up, run
// every frame, and its display-sized colour output discarded, purely to harvest
// them.
//
// Measured cost of that on this machine: 5.56-5.61 GB of heap without frame
// generation, 6.61 GB with it, on an 8 GB card. The ~1.4 GB left over is not
// enough for an aircraft reload, and the sim died on the next aircraft change
// every time - with X-Plane logging heavy_pressure to the last line and
// validation reporting nothing, because running out of memory is not an API
// misuse for validation to catch.
//
// None of it is necessary. ffxFrameInterpolationGetSharedResourceDescriptions
// answers all three descriptions from the INTERPOLATION context alone, the
// layer already allocates the images itself, and the contents are two things
// this file can compute directly:
//
//   dilated depth + motion  a 3x3 closest-depth search, the same one the TAA
//                           resolve uses for its own history fetch
//   reconstructed prev      a scatter along the motion vector keeping the
//                           nearest sample
//
// So FSR3 leaves the frame-generation path entirely, and the gigabyte goes with
// it. See src/shaders/fg_prepare.comp for the arithmetic.
#include <vulkan/vulkan.h>
#include <string.h>
#include <mutex>
#include "fg_prepare_spv.h"

namespace fgprep {

struct State {
    bool ready  = false;
    bool failed = false;

    VkDevice device = VK_NULL_HANDLE;

    // The three FFX consumes. Formats are dictated by
    // ffxFrameInterpolationGetSharedResourceDescriptions and are NOT free to
    // change: R32_FLOAT, R16G16_FLOAT, R32_UINT, all at render size.
    VkImage        dilDepth  = VK_NULL_HANDLE;
    VkImage        dilMv     = VK_NULL_HANDLE;
    VkImage        prevDepth = VK_NULL_HANDLE;
    VkDeviceMemory dilDepthMem = VK_NULL_HANDLE;
    VkDeviceMemory dilMvMem    = VK_NULL_HANDLE;
    VkDeviceMemory prevDepthMem = VK_NULL_HANDLE;
    VkImageView    dilDepthView  = VK_NULL_HANDLE;
    VkImageView    dilMvView     = VK_NULL_HANDLE;
    VkImageView    prevDepthView = VK_NULL_HANDLE;

    // Inputs. The depth view is over depthcopy's R32_SFLOAT image, not
    // X-Plane's combined depth-stencil - the same reason depthcopy exists at
    // all, since a compute shader cannot sample D32_SFLOAT_S8_UINT.
    VkImageView    depthView = VK_NULL_HANDLE;
    VkImageView    velView   = VK_NULL_HANDLE;
    VkImage        depthSrc  = VK_NULL_HANDLE;   // what depthView was built for
    VkImage        velSrc    = VK_NULL_HANDLE;
    VkSampler      sampler   = VK_NULL_HANDLE;

    VkDescriptorSetLayout setLayout  = VK_NULL_HANDLE;
    VkDescriptorPool      pool       = VK_NULL_HANDLE;
    VkDescriptorSet       set        = VK_NULL_HANDLE;
    VkPipelineLayout      pipeLayout = VK_NULL_HANDLE;
    VkPipeline            pipeline   = VK_NULL_HANDLE;

    uint32_t w = 0, h = 0;
    uint64_t runs = 0;
    std::mutex lock;
};

inline State &state() { static State s; return s; }

struct Push {
    int32_t sizeX, sizeY;
    float   mvScaleX, mvScaleY;
    int32_t reverseZ;
    int32_t pad0, pad1, pad2;
};

inline uint32_t memTypeFor(const VkPhysicalDeviceMemoryProperties &mp,
                           uint32_t bits, VkMemoryPropertyFlags want)
{
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

inline bool makeImage(VkDevice device, PFN_vkGetDeviceProcAddr gdpa,
                      const VkPhysicalDeviceMemoryProperties &mp,
                      uint32_t w, uint32_t h, VkFormat fmt,
                      VkImage *img, VkDeviceMemory *mem, VkImageView *view)
{
    PFN_vkCreateImage createImage = (PFN_vkCreateImage)gdpa(device, "vkCreateImage");
    PFN_vkCreateImageView createView = (PFN_vkCreateImageView)gdpa(device, "vkCreateImageView");
    PFN_vkGetImageMemoryRequirements getReq =
        (PFN_vkGetImageMemoryRequirements)gdpa(device, "vkGetImageMemoryRequirements");
    PFN_vkAllocateMemory allocMem = (PFN_vkAllocateMemory)gdpa(device, "vkAllocateMemory");
    PFN_vkBindImageMemory bindMem = (PFN_vkBindImageMemory)gdpa(device, "vkBindImageMemory");
    if (!createImage || !createView || !getReq || !allocMem || !bindMem) return false;

    VkImageCreateInfo ici;
    memset(&ici, 0, sizeof(ici));
    ici.sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType   = VK_IMAGE_TYPE_2D;
    ici.format      = fmt;
    ici.extent.width = w; ici.extent.height = h; ici.extent.depth = 1;
    ici.mipLevels   = 1;
    ici.arrayLayers = 1;
    ici.samples     = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling      = VK_IMAGE_TILING_OPTIMAL;
    // STORAGE because this shader writes them; SAMPLED and the transfer bits
    // because FFX reads them and the prev-depth image is cleared every frame.
    ici.usage       = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (createImage(device, &ici, nullptr, img) != VK_SUCCESS) return false;

    VkMemoryRequirements mr;
    getReq(device, *img, &mr);
    const uint32_t ti = memTypeFor(mp, mr.memoryTypeBits,
                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (ti == UINT32_MAX) return false;
    VkMemoryAllocateInfo mai;
    memset(&mai, 0, sizeof(mai));
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = mr.size;
    mai.memoryTypeIndex = ti;
    if (allocMem(device, &mai, nullptr, mem) != VK_SUCCESS) return false;
    if (bindMem(device, *img, *mem, 0) != VK_SUCCESS) return false;

    VkImageViewCreateInfo ivci;
    memset(&ivci, 0, sizeof(ivci));
    ivci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image    = *img;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format   = fmt;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.layerCount = 1;
    return createView(device, &ivci, nullptr, view) == VK_SUCCESS;
}

// depthImage is depthcopy's R32_SFLOAT copy; velImage is our velocity target,
// which is a 2D ARRAY (stereo), so its view must be an array view to match the
// shader's sampler2DArray.
inline bool ensure(VkDevice device, VkPhysicalDevice phys,
                   PFN_vkGetDeviceProcAddr gdpa,
                   PFN_vkGetPhysicalDeviceMemoryProperties getMemProps,
                   VkImage depthImage, VkImage velImage, VkFormat velFormat,
                   uint32_t w, uint32_t h)
{
    State &s = state();
    std::lock_guard<std::mutex> g(s.lock);
    if (s.failed) return false;
    if (s.ready && s.w == w && s.h == h &&
        s.depthSrc == depthImage && s.velSrc == velImage) return true;
    if (!device || !gdpa || !w || !h ||
        depthImage == VK_NULL_HANDLE || velImage == VK_NULL_HANDLE) return false;
    // A changed input image means the views are stale. Rebuilding is out of
    // scope here - report and refuse rather than sample a destroyed image,
    // which is the failure this whole session kept chasing.
    if (s.ready) {
        trace("FG PREPARE: inputs changed (depth %p->%p, velocity %p->%p) - "
              "standing down rather than sampling stale views.",
              (void*)s.depthSrc, (void*)depthImage, (void*)s.velSrc, (void*)velImage);
        s.failed = true;
        return false;
    }

    s.device = device;
    s.w = w; s.h = h;
    s.depthSrc = depthImage; s.velSrc = velImage;

    VkPhysicalDeviceMemoryProperties mp;
    memset(&mp, 0, sizeof(mp));
    if (getMemProps) getMemProps(phys, &mp);

    // The three, in FFX's formats. Not negotiable - see the header note.
    if (!makeImage(device, gdpa, mp, w, h, VK_FORMAT_R32_SFLOAT,
                   &s.dilDepth, &s.dilDepthMem, &s.dilDepthView) ||
        !makeImage(device, gdpa, mp, w, h, VK_FORMAT_R16G16_SFLOAT,
                   &s.dilMv, &s.dilMvMem, &s.dilMvView) ||
        !makeImage(device, gdpa, mp, w, h, VK_FORMAT_R32_UINT,
                   &s.prevDepth, &s.prevDepthMem, &s.prevDepthView)) {
        trace("FG PREPARE: could not allocate the three interpolation inputs");
        s.failed = true; return false;
    }

    PFN_vkCreateImageView createView = (PFN_vkCreateImageView)gdpa(device, "vkCreateImageView");
    PFN_vkCreateSampler   createSampler = (PFN_vkCreateSampler)gdpa(device, "vkCreateSampler");
    if (!createView || !createSampler) { s.failed = true; return false; }

    VkImageViewCreateInfo dv;
    memset(&dv, 0, sizeof(dv));
    dv.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    dv.image    = depthImage;
    dv.viewType = VK_IMAGE_VIEW_TYPE_2D;
    dv.format   = VK_FORMAT_R32_SFLOAT;
    dv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    dv.subresourceRange.levelCount = 1;
    dv.subresourceRange.layerCount = 1;
    if (createView(device, &dv, nullptr, &s.depthView) != VK_SUCCESS) {
        trace("FG PREPARE: depth view failed"); s.failed = true; return false;
    }

    VkImageViewCreateInfo vv = dv;
    vv.image    = velImage;
    vv.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    vv.format   = velFormat;
    if (createView(device, &vv, nullptr, &s.velView) != VK_SUCCESS) {
        trace("FG PREPARE: velocity view failed"); s.failed = true; return false;
    }

    VkSamplerCreateInfo sci;
    memset(&sci, 0, sizeof(sci));
    sci.sType     = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_NEAREST;
    sci.minFilter = VK_FILTER_NEAREST;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = sci.addressModeU;
    sci.addressModeW = sci.addressModeU;
    sci.maxLod = 0.25f;
    if (createSampler(device, &sci, nullptr, &s.sampler) != VK_SUCCESS) {
        trace("FG PREPARE: sampler failed"); s.failed = true; return false;
    }

    // 2 sampled inputs + 3 storage outputs, matching fg_prepare.comp.
    VkDescriptorSetLayoutBinding b[5];
    memset(b, 0, sizeof(b));
    for (int i = 0; i < 5; ++i) {
        b[i].binding = (uint32_t)i;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[i].descriptorType = (i < 2) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                      : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }
    VkDescriptorSetLayoutCreateInfo slci;
    memset(&slci, 0, sizeof(slci));
    slci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    slci.bindingCount = 5; slci.pBindings = b;
    PFN_vkCreateDescriptorSetLayout mkSetLayout =
        (PFN_vkCreateDescriptorSetLayout)gdpa(device, "vkCreateDescriptorSetLayout");
    if (!mkSetLayout || mkSetLayout(device, &slci, nullptr, &s.setLayout) != VK_SUCCESS) {
        trace("FG PREPARE: set layout failed"); s.failed = true; return false;
    }

    // The pool must carry a size for EVERY descriptor type the layout uses -
    // a pool missing one is the WrongType warning validation reports, and some
    // drivers do not return OUT_OF_POOL_MEMORY as they should.
    VkDescriptorPoolSize ps[2];
    memset(ps, 0, sizeof(ps));
    ps[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; ps[0].descriptorCount = 2;
    ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          ps[1].descriptorCount = 3;
    VkDescriptorPoolCreateInfo pci;
    memset(&pci, 0, sizeof(pci));
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    PFN_vkCreateDescriptorPool mkPool =
        (PFN_vkCreateDescriptorPool)gdpa(device, "vkCreateDescriptorPool");
    if (!mkPool || mkPool(device, &pci, nullptr, &s.pool) != VK_SUCCESS) {
        trace("FG PREPARE: descriptor pool failed"); s.failed = true; return false;
    }

    VkDescriptorSetAllocateInfo dsai;
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = s.pool;
    dsai.descriptorSetCount = 1; dsai.pSetLayouts = &s.setLayout;
    PFN_vkAllocateDescriptorSets allocSets =
        (PFN_vkAllocateDescriptorSets)gdpa(device, "vkAllocateDescriptorSets");
    if (!allocSets || allocSets(device, &dsai, &s.set) != VK_SUCCESS) {
        trace("FG PREPARE: descriptor set failed"); s.failed = true; return false;
    }

    VkDescriptorImageInfo ii[5];
    memset(ii, 0, sizeof(ii));
    ii[0].sampler = s.sampler; ii[0].imageView = s.depthView;
    ii[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ii[1].sampler = s.sampler; ii[1].imageView = s.velView;
    ii[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ii[2].imageView = s.dilDepthView;  ii[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ii[3].imageView = s.dilMvView;     ii[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    ii[4].imageView = s.prevDepthView; ii[4].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet w5[5];
    memset(w5, 0, sizeof(w5));
    for (int i = 0; i < 5; ++i) {
        w5[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w5[i].dstSet = s.set;
        w5[i].dstBinding = (uint32_t)i;
        w5[i].descriptorCount = 1;
        w5[i].descriptorType = (i < 2) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                       : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w5[i].pImageInfo = &ii[i];
    }
    PFN_vkUpdateDescriptorSets updSets =
        (PFN_vkUpdateDescriptorSets)gdpa(device, "vkUpdateDescriptorSets");
    if (!updSets) { s.failed = true; return false; }
    updSets(device, 5, w5, 0, nullptr);

    VkPushConstantRange pcr;
    memset(&pcr, 0, sizeof(pcr));
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size = sizeof(Push);
    VkPipelineLayoutCreateInfo plci;
    memset(&plci, 0, sizeof(plci));
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &s.setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    PFN_vkCreatePipelineLayout mkPipeLayout =
        (PFN_vkCreatePipelineLayout)gdpa(device, "vkCreatePipelineLayout");
    if (!mkPipeLayout || mkPipeLayout(device, &plci, nullptr, &s.pipeLayout) != VK_SUCCESS) {
        trace("FG PREPARE: pipeline layout failed"); s.failed = true; return false;
    }

    VkShaderModuleCreateInfo smci;
    memset(&smci, 0, sizeof(smci));
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = kFgPrepareSpvWords * 4;
    smci.pCode    = kFgPrepareSpv;
    VkShaderModule mod = VK_NULL_HANDLE;
    PFN_vkCreateShaderModule mkModule =
        (PFN_vkCreateShaderModule)gdpa(device, "vkCreateShaderModule");
    if (!mkModule || mkModule(device, &smci, nullptr, &mod) != VK_SUCCESS) {
        trace("FG PREPARE: shader module failed"); s.failed = true; return false;
    }

    VkComputePipelineCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName  = "main";
    cpci.layout = s.pipeLayout;
    PFN_vkCreateComputePipelines mkPipe =
        (PFN_vkCreateComputePipelines)gdpa(device, "vkCreateComputePipelines");
    const VkResult pr = mkPipe ? mkPipe(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &s.pipeline)
                               : VK_ERROR_INITIALIZATION_FAILED;
    PFN_vkDestroyShaderModule delModule =
        (PFN_vkDestroyShaderModule)gdpa(device, "vkDestroyShaderModule");
    if (delModule && mod) delModule(device, mod, nullptr);
    if (pr != VK_SUCCESS) {
        trace("FG PREPARE: compute pipeline failed (%d)", (int)pr);
        s.failed = true; return false;
    }

    s.ready = true;
    trace("FG PREPARE: ready - %ux%u, producing dilatedDepth (R32F), "
          "dilatedMotionVectors (RG16F) and reconstructedPrevNearestDepth "
          "(R32U) directly. The FSR3 upscaler is no longer needed to obtain "
          "them, and neither is the display-sized output image it wrote and "
          "nobody read.", w, h);
    return true;
}

// Records into a command buffer the CALLER owns and submits. Same discipline as
// the dilation side-car: never record into X-Plane's command stream while the
// interpolation proxy swapchain is live.
inline void record(PFN_vkCmdBindPipeline bindPipe,
                   PFN_vkCmdBindDescriptorSets bindSets,
                   PFN_vkCmdDispatch dispatch,
                   PFN_vkCmdPipelineBarrier barrierFn,
                   PFN_vkCmdPushConstants pushFn,
                   PFN_vkCmdClearColorImage clearFn,
                   VkCommandBuffer cb,
                   float mvScaleX, float mvScaleY, bool reverseZ)
{
    State &s = state();
    if (!s.ready || s.failed) return;

    VkImageSubresourceRange rng;
    memset(&rng, 0, sizeof(rng));
    rng.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    rng.levelCount = 1; rng.layerCount = 1;

    VkImageMemoryBarrier toGeneral[3];
    memset(toGeneral, 0, sizeof(toGeneral));
    VkImage imgs[3] = { s.dilDepth, s.dilMv, s.prevDepth };
    for (int i = 0; i < 3; ++i) {
        toGeneral[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toGeneral[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toGeneral[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        toGeneral[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral[i].image = imgs[i];
        toGeneral[i].subresourceRange = rng;
        toGeneral[i].srcAccessMask = 0;
        toGeneral[i].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                                     VK_ACCESS_SHADER_WRITE_BIT;
    }
    barrierFn(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
              VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              0, 0, nullptr, 0, nullptr, 3, toGeneral);

    // ---- THE SCATTER TARGET MUST BE CLEARED EVERY FRAME.
    //
    // reconstructedPrevNearestDepth is written with imageAtomicMax. Without a
    // clear, last frame's values survive and the max keeps whichever depth was
    // ever nearest at that texel - so the reconstruction becomes the union of
    // every frame so far and drifts further from the truth the longer the sim
    // runs. It would never error; it would just be quietly wrong, worse over
    // time, which is the hardest kind of bug to notice.
    //
    // Cleared to 0, which is "nothing here": the complement trick in the shader
    // means 0 is the FARTHEST value under both depth conventions, so any real
    // sample wins the max.
    VkClearColorValue zero;
    memset(&zero, 0, sizeof(zero));
    clearFn(cb, s.prevDepth, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &rng);

    VkImageMemoryBarrier afterClear;
    memset(&afterClear, 0, sizeof(afterClear));
    afterClear.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    afterClear.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    afterClear.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    afterClear.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    afterClear.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    afterClear.image = s.prevDepth;
    afterClear.subresourceRange = rng;
    afterClear.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    afterClear.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrierFn(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
              1, &afterClear);

    bindPipe(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipeline);
    bindSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipeLayout, 0, 1, &s.set, 0, nullptr);

    Push p;
    memset(&p, 0, sizeof(p));
    p.sizeX = (int32_t)s.w; p.sizeY = (int32_t)s.h;
    p.mvScaleX = mvScaleX;  p.mvScaleY = mvScaleY;
    p.reverseZ = reverseZ ? 1 : 0;
    pushFn(cb, s.pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);

    dispatch(cb, (s.w + 7) / 8, (s.h + 7) / 8, 1);

    // FFX reads all three next; make the writes visible to it.
    VkImageMemoryBarrier toRead[3];
    memcpy(toRead, toGeneral, sizeof(toRead));
    for (int i = 0; i < 3; ++i) {
        toRead[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        toRead[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        toRead[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        toRead[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    }
    barrierFn(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
              3, toRead);

    ++s.runs;
    if (s.runs == 1 || (s.runs % 600) == 0)
        trace("FG PREPARE: %llu dispatches - interpolation inputs produced "
              "without an upscaler.", (unsigned long long)s.runs);
}

} // namespace fgprep
