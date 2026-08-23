// Turn X-Plane's depth-stencil buffer into a plain depth image FSR 3 can read.
//
// ---- WHY THIS IS NEEDED.
//
// X-Plane's scene depth is D32_SFLOAT_S8_UINT. FidelityFX's Vulkan backend
// assumes plain depth: for a depth target it forces the view format to
// VK_FORMAT_D32_SFLOAT, and that view is not a legal pairing over a combined
// depth-stencil image. Every CPU call still returns FFX_OK and the validation
// layer says nothing about the dispatch - the GPU dies a few frames later, when
// the view is sampled. FSR 3 ran exactly six dispatches, every run, and the
// output image turned out to be irrelevant: pointing it at an image of ours
// changed nothing, which is what put the fault on the INPUT side.
//
// Vulkan offers no cheap conversion. Depth/stencil copies require identical
// formats, and a blit cannot cross depth to colour. So one small compute pass
// reads the depth aspect and writes R32_SFLOAT.
//
// ---- ITS OWN PIPELINE, ITS OWN DESCRIPTORS.
//
// Deliberately self-contained rather than sharing the TAA resolve's machinery:
// this runs at a different point in the frame, against different images, and
// the resolve's descriptors are already the most delicate part of this layer.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <string.h>

#include "depth_copy_spv.h"

namespace depthcopy {

struct State {
    bool             ready  = false;
    bool             failed = false;

    VkImage          image  = VK_NULL_HANDLE;   // R32_SFLOAT copy
    VkDeviceMemory   mem    = VK_NULL_HANDLE;
    VkImageView      dstView = VK_NULL_HANDLE;
    VkImageView      srcView = VK_NULL_HANDLE;  // depth aspect of X-Plane's image

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool      pool      = VK_NULL_HANDLE;
    VkDescriptorSet       set       = VK_NULL_HANDLE;
    VkPipelineLayout      pipeLayout = VK_NULL_HANDLE;
    VkPipeline            pipeline   = VK_NULL_HANDLE;

    VkImage          srcImage = VK_NULL_HANDLE;  // what srcView was built for
    uint32_t         w = 0, h = 0;
};

inline State &state()
{
    static State s;
    return s;
}

// Built once the depth image and render size are known. Returns false and
// latches `failed` on any error, so a broken setup is reported once rather than
// retried every frame.
inline bool ensure(VkDevice device, VkPhysicalDevice phys,
                   PFN_vkGetDeviceProcAddr gdpa,
                   PFN_vkGetPhysicalDeviceMemoryProperties getMemProps,
                   VkImage depthImage, VkFormat depthFormat,
                   uint32_t w, uint32_t h)
{
    State &s = state();
    if (s.failed) return false;
    if (s.ready && s.srcImage == depthImage && s.w == w && s.h == h) return true;
    if (s.ready) return true;          // the depth image moved; a later concern
    if (!device || !gdpa || depthImage == VK_NULL_HANDLE || !w || !h) return false;

    PFN_vkCreateImage      createImage = (PFN_vkCreateImage)gdpa(device, "vkCreateImage");
    PFN_vkCreateImageView  createView  = (PFN_vkCreateImageView)gdpa(device, "vkCreateImageView");
    PFN_vkGetImageMemoryRequirements getReq =
        (PFN_vkGetImageMemoryRequirements)gdpa(device, "vkGetImageMemoryRequirements");
    PFN_vkAllocateMemory   allocMem   = (PFN_vkAllocateMemory)gdpa(device, "vkAllocateMemory");
    PFN_vkBindImageMemory  bindMem    = (PFN_vkBindImageMemory)gdpa(device, "vkBindImageMemory");
    PFN_vkCreateDescriptorSetLayout createSetLayout =
        (PFN_vkCreateDescriptorSetLayout)gdpa(device, "vkCreateDescriptorSetLayout");
    PFN_vkCreateDescriptorPool createPool =
        (PFN_vkCreateDescriptorPool)gdpa(device, "vkCreateDescriptorPool");
    PFN_vkAllocateDescriptorSets allocSets =
        (PFN_vkAllocateDescriptorSets)gdpa(device, "vkAllocateDescriptorSets");
    PFN_vkUpdateDescriptorSets updateSets =
        (PFN_vkUpdateDescriptorSets)gdpa(device, "vkUpdateDescriptorSets");
    PFN_vkCreatePipelineLayout createPipeLayout =
        (PFN_vkCreatePipelineLayout)gdpa(device, "vkCreatePipelineLayout");
    PFN_vkCreateComputePipelines createComputePipelines =
        (PFN_vkCreateComputePipelines)gdpa(device, "vkCreateComputePipelines");
    PFN_vkCreateShaderModule createShader =
        (PFN_vkCreateShaderModule)gdpa(device, "vkCreateShaderModule");
    PFN_vkDestroyShaderModule destroyShader =
        (PFN_vkDestroyShaderModule)gdpa(device, "vkDestroyShaderModule");

    if (!createImage || !createView || !getReq || !allocMem || !bindMem ||
        !createSetLayout || !createPool || !allocSets || !updateSets ||
        !createPipeLayout || !createComputePipelines || !createShader) {
        s.failed = true;
        trace("DEPTH COPY: a required entry point is missing");
        return false;
    }

    // ---- the R32_SFLOAT destination
    VkImageCreateInfo ici;
    memset(&ici, 0, sizeof(ici));
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = VK_FORMAT_R32_SFLOAT;
    ici.extent.width  = w;
    ici.extent.height = h;
    ici.extent.depth  = 1;
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    // STORAGE so this pass can write it, SAMPLED so FSR3 can read it.
    ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    if (createImage(device, &ici, nullptr, &s.image) != VK_SUCCESS) {
        s.failed = true; trace("DEPTH COPY: image creation failed"); return false;
    }

    VkMemoryRequirements mr;
    getReq(device, s.image, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    memset(&mp, 0, sizeof(mp));
    getMemProps(phys, &mp);
    uint32_t ti = UINT32_MAX;
    for (uint32_t k = 0; k < mp.memoryTypeCount; ++k)
        if ((mr.memoryTypeBits & (1u << k)) &&
            (mp.memoryTypes[k].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            ti = k; break;
        }
    VkMemoryAllocateInfo mai;
    memset(&mai, 0, sizeof(mai));
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = mr.size;
    mai.memoryTypeIndex = ti;
    if (ti == UINT32_MAX ||
        allocMem(device, &mai, nullptr, &s.mem) != VK_SUCCESS ||
        bindMem(device, s.image, s.mem, 0) != VK_SUCCESS) {
        s.failed = true; trace("DEPTH COPY: memory failed"); return false;
    }

    // ---- views. The source view takes the DEPTH ASPECT ONLY, which is the
    // whole point: it is legal over a depth-stencil image where FFX's own view
    // would not be.
    VkImageViewCreateInfo ivci;
    memset(&ivci, 0, sizeof(ivci));
    ivci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.layerCount = 1;

    ivci.image  = depthImage;
    ivci.format = depthFormat;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (createView(device, &ivci, nullptr, &s.srcView) != VK_SUCCESS) {
        s.failed = true; trace("DEPTH COPY: source view failed"); return false;
    }

    ivci.image  = s.image;
    ivci.format = VK_FORMAT_R32_SFLOAT;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    if (createView(device, &ivci, nullptr, &s.dstView) != VK_SUCCESS) {
        s.failed = true; trace("DEPTH COPY: destination view failed"); return false;
    }

    // ---- descriptors
    VkDescriptorSetLayoutBinding binds[2];
    memset(binds, 0, sizeof(binds));
    binds[0].binding = 0;
    binds[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    binds[0].descriptorCount = 1;
    binds[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    binds[1].binding = 1;
    binds[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binds[1].descriptorCount = 1;
    binds[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo slci;
    memset(&slci, 0, sizeof(slci));
    slci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    slci.bindingCount = 2;
    slci.pBindings = binds;
    if (createSetLayout(device, &slci, nullptr, &s.setLayout) != VK_SUCCESS) {
        s.failed = true; trace("DEPTH COPY: set layout failed"); return false;
    }

    VkDescriptorPoolSize sizes[2];
    sizes[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; sizes[0].descriptorCount = 1;
    sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; sizes[1].descriptorCount = 1;
    VkDescriptorPoolCreateInfo pci;
    memset(&pci, 0, sizeof(pci));
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1;
    pci.poolSizeCount = 2;
    pci.pPoolSizes = sizes;
    if (createPool(device, &pci, nullptr, &s.pool) != VK_SUCCESS) {
        s.failed = true; trace("DEPTH COPY: descriptor pool failed"); return false;
    }

    VkDescriptorSetAllocateInfo dsai;
    memset(&dsai, 0, sizeof(dsai));
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = s.pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &s.setLayout;
    if (allocSets(device, &dsai, &s.set) != VK_SUCCESS) {
        s.failed = true; trace("DEPTH COPY: descriptor set failed"); return false;
    }

    VkDescriptorImageInfo dii[2];
    memset(dii, 0, sizeof(dii));
    dii[0].imageView   = s.srcView;
    dii[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    dii[1].imageView   = s.dstView;
    dii[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[2];
    memset(writes, 0, sizeof(writes));
    for (int i = 0; i < 2; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = s.set;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].pImageInfo = &dii[i];
    }
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    updateSets(device, 2, writes, 0, nullptr);

    // ---- pipeline
    VkPipelineLayoutCreateInfo plci;
    memset(&plci, 0, sizeof(plci));
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &s.setLayout;
    if (createPipeLayout(device, &plci, nullptr, &s.pipeLayout) != VK_SUCCESS) {
        s.failed = true; trace("DEPTH COPY: pipeline layout failed"); return false;
    }

    VkShaderModuleCreateInfo smci;
    memset(&smci, 0, sizeof(smci));
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = kDepthCopySpvWords * 4;
    smci.pCode    = kDepthCopySpv;
    VkShaderModule module = VK_NULL_HANDLE;
    if (createShader(device, &smci, nullptr, &module) != VK_SUCCESS) {
        s.failed = true; trace("DEPTH COPY: shader module failed"); return false;
    }

    VkComputePipelineCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = module;
    cpci.stage.pName  = "main";
    cpci.layout = s.pipeLayout;
    const VkResult pr = createComputePipelines(device, VK_NULL_HANDLE, 1, &cpci,
                                               nullptr, &s.pipeline);
    if (destroyShader) destroyShader(device, module, nullptr);
    if (pr != VK_SUCCESS) {
        s.failed = true; trace("DEPTH COPY: compute pipeline failed (%d)", (int)pr);
        return false;
    }

    s.srcImage = depthImage;
    s.w = w; s.h = h;
    s.ready = true;
    trace("DEPTH COPY: ready - %ux%u, %s -> R32_SFLOAT. FSR3 cannot read a "
          "combined depth-stencil image, so it reads this instead.",
          w, h, "D32_SFLOAT_S8_UINT");
    return true;
}

// Recorded into X-Plane's command buffer immediately before FSR 3's dispatch.
inline void record(PFN_vkCmdBindPipeline bindPipe,
                   PFN_vkCmdBindDescriptorSets bindSets,
                   PFN_vkCmdDispatch dispatch,
                   PFN_vkCmdPipelineBarrier barrierFn,
                   VkCommandBuffer cb, VkImageLayout depthLayout)
{
    State &s = state();
    if (!s.ready || s.failed) return;
    if (!bindPipe || !bindSets || !dispatch || !barrierFn) return;

    // The destination starts UNDEFINED and must be GENERAL to be written. Done
    // every frame from UNDEFINED on purpose: the previous contents are entirely
    // replaced, so discarding them is correct and avoids tracking a layout.
    VkImageMemoryBarrier b;
    memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = s.image;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcAccessMask = 0;
    b.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrierFn(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);

    bindPipe(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipeline);
    bindSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, s.pipeLayout, 0, 1, &s.set, 0, nullptr);
    dispatch(cb, (s.w + 7) / 8, (s.h + 7) / 8, 1);

    // Order this write before FSR3 reads it, and move it to the layout FFX
    // expects for a sampled input.
    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrierFn(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    (void)depthLayout;
}

} // namespace depthcopy
