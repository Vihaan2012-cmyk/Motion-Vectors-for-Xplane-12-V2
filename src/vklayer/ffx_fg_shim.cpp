// The 56 Vulkan entry points FrameInterpolationSwapchainVK.cpp calls BY NAME.
//
// GENERATED shape, hand-reviewed: every signature below was taken from
// vulkan_core.h rather than transcribed, because a single wrong parameter type
// here is an ABI mismatch that corrupts the stack at the call rather than
// failing to compile.
//
// ---- WHY THESE CANNOT SIMPLY LINK AGAINST THE LOADER.
//
// This is the same trap ffx_vk_shim.cpp documents, and worse. The FI swapchain
// calls vkCreateSwapchainKHR, vkQueuePresentKHR, vkDestroyImage,
// vkCmdPipelineBarrier and others that THIS LAYER HOOKS. Resolved from the
// loader's import library they would re-enter the dispatch chain at the TOP and
// land back in our own hooks - unbounded recursion, and for vkQueuePresentKHR
// that is once per frame.
//
// So build.ps1 compiles the FFX frame-generation objects with each name
// redefined to mvFfx_<name>, and these forward DOWN the chain through the
// next-layer table instead. No Vulkan symbol is defined in this DLL.
//
// ---- ONE DEVICE, SO ONE RESOLVER.
//
// FFX builds its context for a single device, so a device-level function
// resolved from it is valid for every queue and command buffer created from it.
// That is what lets calls whose first parameter is a VkDevice, a VkQueue or a
// VkCommandBuffer all resolve through mvFfxDeviceProc without a dispatch-key
// lookup. The two physical-device queries go through mvFfxInstanceProc, bound
// once from the application's instance.
//
// A null resolve returns zero rather than calling through null: the caller then
// fails its own way, which is recoverable and reportable. Jumping to address
// zero is neither.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <vulkan/vulkan.h>

extern "C" {
PFN_vkVoidFunction mvFfxDeviceProc(const char *name);
PFN_vkVoidFunction mvFfxInstanceProc(const char *name);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkAcquireNextImageKHR(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore, VkFence fence, uint32_t* pImageIndex)
{
    static PFN_vkAcquireNextImageKHR fn = nullptr;
    if (!fn) fn = (PFN_vkAcquireNextImageKHR)mvFfxDeviceProc("vkAcquireNextImageKHR");
    if (!fn) return (VkResult)0;
    return fn(device, swapchain, timeout, semaphore, fence, pImageIndex);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkAllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo* pAllocateInfo, VkCommandBuffer* pCommandBuffers)
{
    static PFN_vkAllocateCommandBuffers fn = nullptr;
    if (!fn) fn = (PFN_vkAllocateCommandBuffers)mvFfxDeviceProc("vkAllocateCommandBuffers");
    if (!fn) return (VkResult)0;
    return fn(device, pAllocateInfo, pCommandBuffers);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo, VkDescriptorSet* pDescriptorSets)
{
    static PFN_vkAllocateDescriptorSets fn = nullptr;
    if (!fn) fn = (PFN_vkAllocateDescriptorSets)mvFfxDeviceProc("vkAllocateDescriptorSets");
    if (!fn) return (VkResult)0;
    return fn(device, pAllocateInfo, pDescriptorSets);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkAllocateMemory(VkDevice device, const VkMemoryAllocateInfo* pAllocateInfo, const VkAllocationCallbacks* pAllocator, VkDeviceMemory* pMemory)
{
    static PFN_vkAllocateMemory fn = nullptr;
    if (!fn) fn = (PFN_vkAllocateMemory)mvFfxDeviceProc("vkAllocateMemory");
    if (!fn) return (VkResult)0;
    return fn(device, pAllocateInfo, pAllocator, pMemory);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkBeginCommandBuffer(VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo* pBeginInfo)
{
    static PFN_vkBeginCommandBuffer fn = nullptr;
    if (!fn) fn = (PFN_vkBeginCommandBuffer)mvFfxDeviceProc("vkBeginCommandBuffer");
    if (!fn) return (VkResult)0;
    return fn(commandBuffer, pBeginInfo);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkBindImageMemory(VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize memoryOffset)
{
    static PFN_vkBindImageMemory fn = nullptr;
    if (!fn) fn = (PFN_vkBindImageMemory)mvFfxDeviceProc("vkBindImageMemory");
    if (!fn) return (VkResult)0;
    return fn(device, image, memory, memoryOffset);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkCmdBeginRenderPass(VkCommandBuffer commandBuffer, const VkRenderPassBeginInfo* pRenderPassBegin, VkSubpassContents contents)
{
    static PFN_vkCmdBeginRenderPass fn = nullptr;
    if (!fn) fn = (PFN_vkCmdBeginRenderPass)mvFfxDeviceProc("vkCmdBeginRenderPass");
    if (fn) fn(commandBuffer, pRenderPassBegin, contents);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkCmdBindDescriptorSets(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipelineLayout layout, uint32_t firstSet, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets, uint32_t dynamicOffsetCount, const uint32_t* pDynamicOffsets)
{
    static PFN_vkCmdBindDescriptorSets fn = nullptr;
    if (!fn) fn = (PFN_vkCmdBindDescriptorSets)mvFfxDeviceProc("vkCmdBindDescriptorSets");
    if (fn) fn(commandBuffer, pipelineBindPoint, layout, firstSet, descriptorSetCount, pDescriptorSets, dynamicOffsetCount, pDynamicOffsets);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkCmdBindPipeline(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipeline pipeline)
{
    static PFN_vkCmdBindPipeline fn = nullptr;
    if (!fn) fn = (PFN_vkCmdBindPipeline)mvFfxDeviceProc("vkCmdBindPipeline");
    if (fn) fn(commandBuffer, pipelineBindPoint, pipeline);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkCmdCopyImage(VkCommandBuffer commandBuffer, VkImage srcImage, VkImageLayout srcImageLayout, VkImage dstImage, VkImageLayout dstImageLayout, uint32_t regionCount, const VkImageCopy* pRegions)
{
    static PFN_vkCmdCopyImage fn = nullptr;
    if (!fn) fn = (PFN_vkCmdCopyImage)mvFfxDeviceProc("vkCmdCopyImage");
    if (fn) fn(commandBuffer, srcImage, srcImageLayout, dstImage, dstImageLayout, regionCount, pRegions);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkCmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
{
    static PFN_vkCmdDraw fn = nullptr;
    if (!fn) fn = (PFN_vkCmdDraw)mvFfxDeviceProc("vkCmdDraw");
    if (fn) fn(commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkCmdEndRenderPass(VkCommandBuffer commandBuffer)
{
    static PFN_vkCmdEndRenderPass fn = nullptr;
    if (!fn) fn = (PFN_vkCmdEndRenderPass)mvFfxDeviceProc("vkCmdEndRenderPass");
    if (fn) fn(commandBuffer);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkCmdPipelineBarrier(VkCommandBuffer commandBuffer, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask, VkDependencyFlags dependencyFlags, uint32_t memoryBarrierCount, const VkMemoryBarrier* pMemoryBarriers, uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier* pBufferMemoryBarriers, uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier* pImageMemoryBarriers)
{
    static PFN_vkCmdPipelineBarrier fn = nullptr;
    if (!fn) fn = (PFN_vkCmdPipelineBarrier)mvFfxDeviceProc("vkCmdPipelineBarrier");
    if (fn) fn(commandBuffer, srcStageMask, dstStageMask, dependencyFlags, memoryBarrierCount, pMemoryBarriers, bufferMemoryBarrierCount, pBufferMemoryBarriers, imageMemoryBarrierCount, pImageMemoryBarriers);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkCmdPushConstants(VkCommandBuffer commandBuffer, VkPipelineLayout layout, VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size, const void* pValues)
{
    static PFN_vkCmdPushConstants fn = nullptr;
    if (!fn) fn = (PFN_vkCmdPushConstants)mvFfxDeviceProc("vkCmdPushConstants");
    if (fn) fn(commandBuffer, layout, stageFlags, offset, size, pValues);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkCmdSetScissor(VkCommandBuffer commandBuffer, uint32_t firstScissor, uint32_t scissorCount, const VkRect2D* pScissors)
{
    static PFN_vkCmdSetScissor fn = nullptr;
    if (!fn) fn = (PFN_vkCmdSetScissor)mvFfxDeviceProc("vkCmdSetScissor");
    if (fn) fn(commandBuffer, firstScissor, scissorCount, pScissors);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkCmdSetViewport(VkCommandBuffer commandBuffer, uint32_t firstViewport, uint32_t viewportCount, const VkViewport* pViewports)
{
    static PFN_vkCmdSetViewport fn = nullptr;
    if (!fn) fn = (PFN_vkCmdSetViewport)mvFfxDeviceProc("vkCmdSetViewport");
    if (fn) fn(commandBuffer, firstViewport, viewportCount, pViewports);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkCreateCommandPool(VkDevice device, const VkCommandPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkCommandPool* pCommandPool)
{
    static PFN_vkCreateCommandPool fn = nullptr;
    if (!fn) fn = (PFN_vkCreateCommandPool)mvFfxDeviceProc("vkCreateCommandPool");
    if (!fn) return (VkResult)0;
    return fn(device, pCreateInfo, pAllocator, pCommandPool);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkCreateDescriptorPool(VkDevice device, const VkDescriptorPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorPool* pDescriptorPool)
{
    static PFN_vkCreateDescriptorPool fn = nullptr;
    if (!fn) fn = (PFN_vkCreateDescriptorPool)mvFfxDeviceProc("vkCreateDescriptorPool");
    if (!fn) return (VkResult)0;
    return fn(device, pCreateInfo, pAllocator, pDescriptorPool);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkCreateDescriptorSetLayout(VkDevice device, const VkDescriptorSetLayoutCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorSetLayout* pSetLayout)
{
    static PFN_vkCreateDescriptorSetLayout fn = nullptr;
    if (!fn) fn = (PFN_vkCreateDescriptorSetLayout)mvFfxDeviceProc("vkCreateDescriptorSetLayout");
    if (!fn) return (VkResult)0;
    return fn(device, pCreateInfo, pAllocator, pSetLayout);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkCreateFramebuffer(VkDevice device, const VkFramebufferCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkFramebuffer* pFramebuffer)
{
    static PFN_vkCreateFramebuffer fn = nullptr;
    if (!fn) fn = (PFN_vkCreateFramebuffer)mvFfxDeviceProc("vkCreateFramebuffer");
    if (!fn) return (VkResult)0;
    return fn(device, pCreateInfo, pAllocator, pFramebuffer);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkCreateGraphicsPipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t createInfoCount, const VkGraphicsPipelineCreateInfo* pCreateInfos, const VkAllocationCallbacks* pAllocator, VkPipeline* pPipelines)
{
    static PFN_vkCreateGraphicsPipelines fn = nullptr;
    if (!fn) fn = (PFN_vkCreateGraphicsPipelines)mvFfxDeviceProc("vkCreateGraphicsPipelines");
    if (!fn) return (VkResult)0;
    return fn(device, pipelineCache, createInfoCount, pCreateInfos, pAllocator, pPipelines);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkCreateImage(VkDevice device, const VkImageCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkImage* pImage)
{
    static PFN_vkCreateImage fn = nullptr;
    if (!fn) fn = (PFN_vkCreateImage)mvFfxDeviceProc("vkCreateImage");
    if (!fn) return (VkResult)0;
    return fn(device, pCreateInfo, pAllocator, pImage);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkCreateImageView(VkDevice device, const VkImageViewCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkImageView* pView)
{
    static PFN_vkCreateImageView fn = nullptr;
    if (!fn) fn = (PFN_vkCreateImageView)mvFfxDeviceProc("vkCreateImageView");
    if (!fn) return (VkResult)0;
    return fn(device, pCreateInfo, pAllocator, pView);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkCreatePipelineLayout(VkDevice device, const VkPipelineLayoutCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkPipelineLayout* pPipelineLayout)
{
    static PFN_vkCreatePipelineLayout fn = nullptr;
    if (!fn) fn = (PFN_vkCreatePipelineLayout)mvFfxDeviceProc("vkCreatePipelineLayout");
    if (!fn) return (VkResult)0;
    return fn(device, pCreateInfo, pAllocator, pPipelineLayout);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkCreateRenderPass(VkDevice device, const VkRenderPassCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkRenderPass* pRenderPass)
{
    static PFN_vkCreateRenderPass fn = nullptr;
    if (!fn) fn = (PFN_vkCreateRenderPass)mvFfxDeviceProc("vkCreateRenderPass");
    if (!fn) return (VkResult)0;
    return fn(device, pCreateInfo, pAllocator, pRenderPass);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkCreateSemaphore(VkDevice device, const VkSemaphoreCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSemaphore* pSemaphore)
{
    static PFN_vkCreateSemaphore fn = nullptr;
    if (!fn) fn = (PFN_vkCreateSemaphore)mvFfxDeviceProc("vkCreateSemaphore");
    if (!fn) return (VkResult)0;
    return fn(device, pCreateInfo, pAllocator, pSemaphore);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkShaderModule* pShaderModule)
{
    static PFN_vkCreateShaderModule fn = nullptr;
    if (!fn) fn = (PFN_vkCreateShaderModule)mvFfxDeviceProc("vkCreateShaderModule");
    if (!fn) return (VkResult)0;
    return fn(device, pCreateInfo, pAllocator, pShaderModule);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkCreateSwapchainKHR(VkDevice device, const VkSwapchainCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain)
{
    static PFN_vkCreateSwapchainKHR fn = nullptr;
    if (!fn) fn = (PFN_vkCreateSwapchainKHR)mvFfxDeviceProc("vkCreateSwapchainKHR");
    if (!fn) return (VkResult)0;
    return fn(device, pCreateInfo, pAllocator, pSwapchain);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkDestroyCommandPool(VkDevice device, VkCommandPool commandPool, const VkAllocationCallbacks* pAllocator)
{
    static PFN_vkDestroyCommandPool fn = nullptr;
    if (!fn) fn = (PFN_vkDestroyCommandPool)mvFfxDeviceProc("vkDestroyCommandPool");
    if (fn) fn(device, commandPool, pAllocator);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkDestroyDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool, const VkAllocationCallbacks* pAllocator)
{
    static PFN_vkDestroyDescriptorPool fn = nullptr;
    if (!fn) fn = (PFN_vkDestroyDescriptorPool)mvFfxDeviceProc("vkDestroyDescriptorPool");
    if (fn) fn(device, descriptorPool, pAllocator);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkDestroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout, const VkAllocationCallbacks* pAllocator)
{
    static PFN_vkDestroyDescriptorSetLayout fn = nullptr;
    if (!fn) fn = (PFN_vkDestroyDescriptorSetLayout)mvFfxDeviceProc("vkDestroyDescriptorSetLayout");
    if (fn) fn(device, descriptorSetLayout, pAllocator);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkDestroyFramebuffer(VkDevice device, VkFramebuffer framebuffer, const VkAllocationCallbacks* pAllocator)
{
    static PFN_vkDestroyFramebuffer fn = nullptr;
    if (!fn) fn = (PFN_vkDestroyFramebuffer)mvFfxDeviceProc("vkDestroyFramebuffer");
    if (fn) fn(device, framebuffer, pAllocator);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkDestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks* pAllocator)
{
    static PFN_vkDestroyImage fn = nullptr;
    if (!fn) fn = (PFN_vkDestroyImage)mvFfxDeviceProc("vkDestroyImage");
    if (fn) fn(device, image, pAllocator);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkDestroyImageView(VkDevice device, VkImageView imageView, const VkAllocationCallbacks* pAllocator)
{
    static PFN_vkDestroyImageView fn = nullptr;
    if (!fn) fn = (PFN_vkDestroyImageView)mvFfxDeviceProc("vkDestroyImageView");
    if (fn) fn(device, imageView, pAllocator);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkDestroyPipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* pAllocator)
{
    static PFN_vkDestroyPipeline fn = nullptr;
    if (!fn) fn = (PFN_vkDestroyPipeline)mvFfxDeviceProc("vkDestroyPipeline");
    if (fn) fn(device, pipeline, pAllocator);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkDestroyPipelineLayout(VkDevice device, VkPipelineLayout pipelineLayout, const VkAllocationCallbacks* pAllocator)
{
    static PFN_vkDestroyPipelineLayout fn = nullptr;
    if (!fn) fn = (PFN_vkDestroyPipelineLayout)mvFfxDeviceProc("vkDestroyPipelineLayout");
    if (fn) fn(device, pipelineLayout, pAllocator);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkDestroyRenderPass(VkDevice device, VkRenderPass renderPass, const VkAllocationCallbacks* pAllocator)
{
    static PFN_vkDestroyRenderPass fn = nullptr;
    if (!fn) fn = (PFN_vkDestroyRenderPass)mvFfxDeviceProc("vkDestroyRenderPass");
    if (fn) fn(device, renderPass, pAllocator);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkDestroySemaphore(VkDevice device, VkSemaphore semaphore, const VkAllocationCallbacks* pAllocator)
{
    static PFN_vkDestroySemaphore fn = nullptr;
    if (!fn) fn = (PFN_vkDestroySemaphore)mvFfxDeviceProc("vkDestroySemaphore");
    if (fn) fn(device, semaphore, pAllocator);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkDestroyShaderModule(VkDevice device, VkShaderModule shaderModule, const VkAllocationCallbacks* pAllocator)
{
    static PFN_vkDestroyShaderModule fn = nullptr;
    if (!fn) fn = (PFN_vkDestroyShaderModule)mvFfxDeviceProc("vkDestroyShaderModule");
    if (fn) fn(device, shaderModule, pAllocator);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkDestroySwapchainKHR(VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks* pAllocator)
{
    static PFN_vkDestroySwapchainKHR fn = nullptr;
    if (!fn) fn = (PFN_vkDestroySwapchainKHR)mvFfxDeviceProc("vkDestroySwapchainKHR");
    if (fn) fn(device, swapchain, pAllocator);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkDeviceWaitIdle(VkDevice device)
{
    static PFN_vkDeviceWaitIdle fn = nullptr;
    if (!fn) fn = (PFN_vkDeviceWaitIdle)mvFfxDeviceProc("vkDeviceWaitIdle");
    if (!fn) return (VkResult)0;
    return fn(device);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkEndCommandBuffer(VkCommandBuffer commandBuffer)
{
    static PFN_vkEndCommandBuffer fn = nullptr;
    if (!fn) fn = (PFN_vkEndCommandBuffer)mvFfxDeviceProc("vkEndCommandBuffer");
    if (!fn) return (VkResult)0;
    return fn(commandBuffer);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkFreeCommandBuffers(VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount, const VkCommandBuffer* pCommandBuffers)
{
    static PFN_vkFreeCommandBuffers fn = nullptr;
    if (!fn) fn = (PFN_vkFreeCommandBuffers)mvFfxDeviceProc("vkFreeCommandBuffers");
    if (fn) fn(device, commandPool, commandBufferCount, pCommandBuffers);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkFreeDescriptorSets(VkDevice device, VkDescriptorPool descriptorPool, uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets)
{
    static PFN_vkFreeDescriptorSets fn = nullptr;
    if (!fn) fn = (PFN_vkFreeDescriptorSets)mvFfxDeviceProc("vkFreeDescriptorSets");
    if (!fn) return (VkResult)0;
    return fn(device, descriptorPool, descriptorSetCount, pDescriptorSets);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkFreeMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* pAllocator)
{
    static PFN_vkFreeMemory fn = nullptr;
    if (!fn) fn = (PFN_vkFreeMemory)mvFfxDeviceProc("vkFreeMemory");
    if (fn) fn(device, memory, pAllocator);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkGetImageMemoryRequirements(VkDevice device, VkImage image, VkMemoryRequirements* pMemoryRequirements)
{
    static PFN_vkGetImageMemoryRequirements fn = nullptr;
    if (!fn) fn = (PFN_vkGetImageMemoryRequirements)mvFfxDeviceProc("vkGetImageMemoryRequirements");
    if (fn) fn(device, image, pMemoryRequirements);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice physicalDevice, uint32_t* pQueueFamilyPropertyCount, VkQueueFamilyProperties* pQueueFamilyProperties)
{
    static PFN_vkGetPhysicalDeviceQueueFamilyProperties fn = nullptr;
    if (!fn) fn = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)mvFfxInstanceProc("vkGetPhysicalDeviceQueueFamilyProperties");
    if (fn) fn(physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkGetPhysicalDeviceSurfaceSupportKHR(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, VkSurfaceKHR surface, VkBool32* pSupported)
{
    static PFN_vkGetPhysicalDeviceSurfaceSupportKHR fn = nullptr;
    if (!fn) fn = (PFN_vkGetPhysicalDeviceSurfaceSupportKHR)mvFfxInstanceProc("vkGetPhysicalDeviceSurfaceSupportKHR");
    if (!fn) return (VkResult)0;
    return fn(physicalDevice, queueFamilyIndex, surface, pSupported);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkGetSemaphoreCounterValue(VkDevice device, VkSemaphore semaphore, uint64_t* pValue)
{
    static PFN_vkGetSemaphoreCounterValue fn = nullptr;
    if (!fn) fn = (PFN_vkGetSemaphoreCounterValue)mvFfxDeviceProc("vkGetSemaphoreCounterValue");
    if (!fn) return (VkResult)0;
    return fn(device, semaphore, pValue);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkGetSwapchainImagesKHR(VkDevice device, VkSwapchainKHR swapchain, uint32_t* pSwapchainImageCount, VkImage* pSwapchainImages)
{
    static PFN_vkGetSwapchainImagesKHR fn = nullptr;
    if (!fn) fn = (PFN_vkGetSwapchainImagesKHR)mvFfxDeviceProc("vkGetSwapchainImagesKHR");
    if (!fn) return (VkResult)0;
    return fn(device, swapchain, pSwapchainImageCount, pSwapchainImages);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkQueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pPresentInfo)
{
    static PFN_vkQueuePresentKHR fn = nullptr;
    if (!fn) fn = (PFN_vkQueuePresentKHR)mvFfxDeviceProc("vkQueuePresentKHR");
    if (!fn) return (VkResult)0;
    return fn(queue, pPresentInfo);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkQueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo* pSubmits, VkFence fence)
{
    static PFN_vkQueueSubmit fn = nullptr;
    if (!fn) fn = (PFN_vkQueueSubmit)mvFfxDeviceProc("vkQueueSubmit");
    if (!fn) return (VkResult)0;
    return fn(queue, submitCount, pSubmits, fence);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkQueueWaitIdle(VkQueue queue)
{
    static PFN_vkQueueWaitIdle fn = nullptr;
    if (!fn) fn = (PFN_vkQueueWaitIdle)mvFfxDeviceProc("vkQueueWaitIdle");
    if (!fn) return (VkResult)0;
    return fn(queue);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkResetCommandBuffer(VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags)
{
    static PFN_vkResetCommandBuffer fn = nullptr;
    if (!fn) fn = (PFN_vkResetCommandBuffer)mvFfxDeviceProc("vkResetCommandBuffer");
    if (!fn) return (VkResult)0;
    return fn(commandBuffer, flags);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkResetCommandPool(VkDevice device, VkCommandPool commandPool, VkCommandPoolResetFlags flags)
{
    static PFN_vkResetCommandPool fn = nullptr;
    if (!fn) fn = (PFN_vkResetCommandPool)mvFfxDeviceProc("vkResetCommandPool");
    if (!fn) return (VkResult)0;
    return fn(device, commandPool, flags);
}

extern "C" VKAPI_ATTR void VKAPI_CALL mvFfx_vkUpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount, const VkWriteDescriptorSet* pDescriptorWrites, uint32_t descriptorCopyCount, const VkCopyDescriptorSet* pDescriptorCopies)
{
    static PFN_vkUpdateDescriptorSets fn = nullptr;
    if (!fn) fn = (PFN_vkUpdateDescriptorSets)mvFfxDeviceProc("vkUpdateDescriptorSets");
    if (fn) fn(device, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount, pDescriptorCopies);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL mvFfx_vkWaitSemaphores(VkDevice device, const VkSemaphoreWaitInfo* pWaitInfo, uint64_t timeout)
{
    static PFN_vkWaitSemaphores fn = nullptr;
    if (!fn) fn = (PFN_vkWaitSemaphores)mvFfxDeviceProc("vkWaitSemaphores");
    if (!fn) return (VkResult)0;
    return fn(device, pWaitInfo, timeout);
}
