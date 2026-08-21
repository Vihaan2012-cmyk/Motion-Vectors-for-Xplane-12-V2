#pragma once

// GPU side of crash destruction: the fragment transform buffer and the
// descriptor set that carries it into every patched pipeline.
//
// WHY A DESCRIPTOR SET AND NOT PUSH CONSTANTS
//
// 2000 fragments at three floats is 24 KB. The push constant block is 144 of
// a 256-byte limit and the layer already reports it TIGHT, with X-Plane's own
// ranges overlapping ours on some layouts. There is no version of this that
// fits in push constants.
//
// WHY THIS IS THE RISKIEST CHANGE IN THE LAYER
//
// Everything else the layer does either APPENDS to a structure X-Plane already
// declared (a push constant range, a colour attachment format) or rewrites a
// shader in place. This adds a descriptor set that must then be BOUND before
// draws, which means participating in X-Plane's own descriptor state rather
// than sitting alongside it.
//
// Two specific hazards, both handled below:
//
//   THE SET INDEX VARIES. We append at ci->setLayoutCount, and different
//   pipelines declare different counts, so "our" set is index 2 on one
//   pipeline and 4 on another. The shader has to be patched with the right
//   index per pipeline, which is why the variant caches are keyed on it.
//
//   BINDING CAN BE DISTURBED. Vulkan only guarantees a bound set survives if
//   subsequent binds use a layout compatible for that index. X-Plane binds
//   with ITS layout, which does not contain our set at all, so ours must be
//   re-bound rather than bound once.
//
// Every step counts itself. A descriptor set that silently fails to bind
// produces geometry that silently fails to displace, and this project has lost
// entire evenings to exactly that shape of failure - the recycle pool that was
// never consulted, the reactive mask firing on coverage nobody wrote.

namespace destructgpu {

struct State {
    VkDevice              device   = VK_NULL_HANDLE;
    VkBuffer              buf      = VK_NULL_HANDLE;
    VkDeviceMemory        mem      = VK_NULL_HANDLE;
    void                 *mapped   = nullptr;
    VkDeviceSize          bytes    = 0;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool      pool     = VK_NULL_HANDLE;
    VkDescriptorSet       set      = VK_NULL_HANDLE;
    uint32_t              capacity = 0;     // fragments the buffer can hold
    bool                  ready    = false;
    bool                  failed   = false; // tried and could not; do not retry
};

inline State &state() { static State s; return s; }

// Counters. See the note at the top: a silent failure here is invisible
// downstream, so every outcome is countable.
inline uint64_t &layoutsExtended() { static uint64_t n = 0; return n; }
inline uint64_t &layoutsTooMany()  { static uint64_t n = 0; return n; }
inline uint64_t &bindsIssued()     { static uint64_t n = 0; return n; }

// Fragments the buffer is sized for. The plan targets 800-2000 for an
// airliner; 4096 leaves headroom without the allocation mattering (48 KB).
static const uint32_t kMaxFragments = 4096;

// One vec4 per fragment rather than vec3: std430 pads a vec3 in an array to 16
// bytes anyway, so the tighter packing would buy nothing and cost a class of
// alignment bug that is tedious to find from a picture.
static const uint32_t kFragmentStride = 16;

inline bool ensure(DeviceData &dd, VkDevice dev)
{
    State &s = state();
    if (s.ready)  return true;
    if (s.failed) return false;
    if (!dd.createBuffer || !dd.allocateMemory || !dd.mapMemory ||
        !dd.createDescriptorSetLayout || !dd.createDescriptorPool ||
        !dd.allocateDescriptorSets || !dd.updateDescriptorSets) {
        s.failed = true;
        trace("DESTRUCT: device is missing entry points this needs - disabled");
        return false;
    }

    s.device = dev;
    s.bytes  = (VkDeviceSize)kMaxFragments * kFragmentStride;

    VkBufferCreateInfo bci;
    memset(&bci, 0, sizeof(bci));
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = s.bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (dd.createBuffer(dev, &bci, nullptr, &s.buf) != VK_SUCCESS) {
        s.failed = true;
        trace("DESTRUCT: transform buffer creation failed (%llu bytes)",
              (unsigned long long)s.bytes);
        return false;
    }

    VkMemoryRequirements mr;
    memset(&mr, 0, sizeof(mr));
    if (dd.getBufferMemReq) dd.getBufferMemReq(dev, s.buf, &mr);
    else { s.failed = true; return false; }

    // HOST_VISIBLE and COHERENT: the plugin publishes new transforms every
    // frame, so this is written far more often than it is read and a staging
    // copy would cost a barrier per frame for no benefit.
    const uint32_t mt = taaFindMemory(dd, mr.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt == UINT32_MAX) {
        s.failed = true;
        trace("DESTRUCT: no host-visible coherent memory type - disabled");
        return false;
    }

    VkMemoryAllocateInfo mai;
    memset(&mai, 0, sizeof(mai));
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = mr.size;
    mai.memoryTypeIndex = mt;
    if (dd.allocateMemory(dev, &mai, nullptr, &s.mem) != VK_SUCCESS) {
        s.failed = true;
        trace("DESTRUCT: transform memory allocation failed");
        return false;
    }
    if (!dd.bindBufferMemory ||
        dd.bindBufferMemory(dev, s.buf, s.mem, 0) != VK_SUCCESS) {
        s.failed = true;
        trace("DESTRUCT: bindBufferMemory failed");
        return false;
    }
    // Mapped once and left mapped, like the VRAM readback buffer. Mapping per
    // frame would serialise against the driver for no reason.
    if (dd.mapMemory(dev, s.mem, 0, s.bytes, 0, &s.mapped) != VK_SUCCESS) {
        s.failed = true;
        trace("DESTRUCT: persistent map failed");
        return false;
    }
    memset(s.mapped, 0, (size_t)s.bytes);

    VkDescriptorSetLayoutBinding b;
    memset(&b, 0, sizeof(b));
    b.binding = 0;
    b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = 1;
    // VERTEX only. The displacement happens there; the fragment stage reads
    // nothing from this, and asking for a stage we do not use would widen the
    // blast radius of a mistake for nothing.
    b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo lci;
    memset(&lci, 0, sizeof(lci));
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 1;
    lci.pBindings    = &b;
    if (dd.createDescriptorSetLayout(dev, &lci, nullptr, &s.setLayout) != VK_SUCCESS) {
        s.failed = true;
        trace("DESTRUCT: descriptor set layout creation failed");
        return false;
    }

    VkDescriptorPoolSize ps;
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 1;

    VkDescriptorPoolCreateInfo pci;
    memset(&pci, 0, sizeof(pci));
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &ps;
    if (dd.createDescriptorPool(dev, &pci, nullptr, &s.pool) != VK_SUCCESS) {
        s.failed = true;
        trace("DESTRUCT: descriptor pool creation failed");
        return false;
    }

    VkDescriptorSetAllocateInfo sai;
    memset(&sai, 0, sizeof(sai));
    sai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    sai.descriptorPool = s.pool;
    sai.descriptorSetCount = 1;
    sai.pSetLayouts = &s.setLayout;
    if (dd.allocateDescriptorSets(dev, &sai, &s.set) != VK_SUCCESS) {
        s.failed = true;
        trace("DESTRUCT: descriptor set allocation failed");
        return false;
    }

    VkDescriptorBufferInfo dbi;
    dbi.buffer = s.buf;
    dbi.offset = 0;
    dbi.range  = s.bytes;

    VkWriteDescriptorSet wr;
    memset(&wr, 0, sizeof(wr));
    wr.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wr.dstSet = s.set;
    wr.dstBinding = 0;
    wr.descriptorCount = 1;
    wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr.pBufferInfo = &dbi;
    dd.updateDescriptorSets(dev, 1, &wr, 0, nullptr);

    s.capacity = kMaxFragments;
    s.ready = true;
    trace("DESTRUCT: transform buffer ready - %u fragments, %llu KB, "
          "host-visible and persistently mapped",
          s.capacity, (unsigned long long)(s.bytes / 1024));
    return true;
}

// Publish this frame's fragment positions. Coherent memory, so no flush.
inline void upload(const float *xyz, uint32_t count)
{
    State &s = state();
    if (!s.ready || !s.mapped || !xyz) return;
    if (count > s.capacity) count = s.capacity;
    float *dst = (float *)s.mapped;
    for (uint32_t i = 0; i < count; ++i) {
        dst[i * 4 + 0] = xyz[i * 3 + 0];
        dst[i * 4 + 1] = xyz[i * 3 + 1];
        dst[i * 4 + 2] = xyz[i * 3 + 2];
        dst[i * 4 + 3] = 0.0f;
    }
}

inline void destroy(DeviceData &dd, VkDevice dev)
{
    State &s = state();
    if (s.mapped && dd.unmapMemory) { dd.unmapMemory(dev, s.mem); s.mapped = nullptr; }
    if (s.pool && dd.destroyDescriptorPool) dd.destroyDescriptorPool(dev, s.pool, nullptr);
    if (s.setLayout && dd.destroyDescriptorSetLayout)
        dd.destroyDescriptorSetLayout(dev, s.setLayout, nullptr);
    if (s.buf && dd.destroyBuffer) dd.destroyBuffer(dev, s.buf, nullptr);
    if (s.mem && dd.freeMemory)    dd.freeMemory(dev, s.mem, nullptr);
    s = State();
}

}  // namespace destructgpu
