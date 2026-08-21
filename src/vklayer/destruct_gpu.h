#pragma once

#include "../destruct/gpu_layout.h"

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
    // An EMPTY set layout, used to pad the indices between what X-Plane
    // declares and where ours sits. See setIndex below for why it has to sit
    // at a fixed place rather than wherever the layout happens to end.
    VkDescriptorSetLayout emptyLayout = VK_NULL_HANDLE;
    uint32_t              setIndex   = 0;
    VkDescriptorPool      pool     = VK_NULL_HANDLE;
    VkDescriptorSet       set      = VK_NULL_HANDLE;
    uint32_t              capacity = 0;     // fragments the buffer can hold
    bool                  ready    = false;
    bool                  failed   = false; // tried and could not; do not retry
};

inline State &state() { static State s; return s; }

// ---- WHY OUR SET SITS AT A FIXED INDEX.
//
// The first version appended our set at each pipeline layout's OWN
// setLayoutCount, so it landed at index 1 on one layout and 4 on another. That
// is fine while only C++ touches it, because the bind looks the index up.
//
// It stops being fine the moment a SHADER references the set, because a
// shader's OpDecorate DescriptorSet is a literal baked into the module - and a
// module is patched once and then used with many different pipeline layouts.
// A varying index cannot be expressed in a constant. Hardcoding whichever
// index we happened to see first would make every other layout bind our buffer
// over one of X-Plane's, which is corruption rather than a missing feature.
//
// So the index is chosen once per device and every extended layout is padded
// with EMPTY set layouts up to it. Empty layouts cost nothing - they declare no
// bindings - and sets 0..N-1 keep exactly the layouts X-Plane declared, which
// is what keeps its own bound sets undisturbed.
//
// 7 is high enough to clear anything X-Plane uses (it binds a handful) and low
// enough to exist on any sane device; the min() keeps it legal on a driver
// with a small maxBoundDescriptorSets.
inline uint32_t chooseSetIndex(uint32_t maxBoundSets)
{
    const uint32_t want = 7;
    if (maxBoundSets == 0) return 0;
    return (want < maxBoundSets - 1) ? want : (maxBoundSets - 1);
}

// Counters. See the note at the top: a silent failure here is invisible
// downstream, so every outcome is countable.
inline uint64_t &layoutsExtended() { static uint64_t n = 0; return n; }
inline uint64_t &layoutsTooMany()  { static uint64_t n = 0; return n; }
inline uint64_t &bindsIssued()     { static uint64_t n = 0; return n; }
// Binds issued immediately before a draw, which are the only ones that
// survive to be read. See the comment on mvRebindDestructSet.
inline uint64_t &drawRebinds()     { static uint64_t n = 0; return n; }

// Fragments the buffer is sized for. The plan targets 800-2000 for an
// airliner; 4096 leaves headroom without the allocation mattering (48 KB).
static const uint32_t kMaxFragments = 4096;

// One vec4 per fragment rather than vec3: std430 pads a vec3 in an array to 16
// bytes anyway, so the tighter packing would buy nothing and cost a class of
// alignment bug that is tedious to find from a picture.
static const uint32_t kFragmentStride = 16;

// The layout header and this file must agree on how many fragments there are.
// A mismatch would size the buffer for one count and index it with another.
#if defined(__cplusplus) && __cplusplus >= 201103L
static_assert(kMaxFragments == destruct::kMaxGpuFragments,
              "fragment capacity disagrees with gpu_layout.h");
#endif

inline bool ensure(DeviceData &dd, VkDevice dev, uint32_t maxBoundSets)
{
    State &s = state();
    // Chosen before anything can read it. setIndex defaulting to 0 would
    // put our set exactly where X-Plane's first set lives.
    s.setIndex = chooseSetIndex(maxBoundSets);
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
    // The whole block, not just the transforms: header, occupancy and
    // transforms are one buffer because they are one descriptor. Sizes and
    // offsets come from gpu_layout.h, which is the file the SPIR-V generator
    // also reads them from - two copies of an offset is how a shader ends up
    // reading transforms out of the middle of the occupancy region.
    s.bytes  = (VkDeviceSize)destruct::kBufferBytes;

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
    // The empty padding layout, created alongside ours. One is enough: the same
    // handle can fill every gap index on every layout, because an empty layout
    // declares nothing and so is compatible with itself everywhere.
    {
        VkDescriptorSetLayoutCreateInfo eci;
        memset(&eci, 0, sizeof(eci));
        eci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        eci.bindingCount = 0;
        eci.pBindings    = nullptr;
        if (dd.createDescriptorSetLayout(dev, &eci, nullptr, &s.emptyLayout) != VK_SUCCESS) {
            s.failed = true;
            trace("DESTRUCT: empty padding set layout failed - disabled");
            return false;
        }
    }
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
    // Transforms live at their own offset now, not at the start of the buffer.
    float *dst = (float *)((unsigned char *)s.mapped + destruct::kOffXform);
    for (uint32_t i = 0; i < count; ++i) {
        dst[i * 4 + 0] = xyz[i * 3 + 0];
        dst[i * 4 + 1] = xyz[i * 3 + 1];
        dst[i * 4 + 2] = xyz[i * 3 + 2];
        dst[i * 4 + 3] = 0.0f;
    }
}

// ---- THE HEADER THE VERTEX SHADER READS.
//
// aircraftInv maps whatever space the shader reconstructs into aircraft-local
// space. Sent every frame the aircraft moves, because the whole classification
// is relative to the airframe, and a stale matrix classifies vertices against
// where the aeroplane WAS.
//
// active is the uniform branch the shader tests. It is in the buffer rather
// than a specialisation constant because it changes at crash time and
// recompiling every pipeline at the moment of a crash is not an option.
inline void uploadHeader(const float aircraftInv[16], const float gridMin[3],
                         float cell, int nx, int ny, int nz, int active)
{
    State &s = state();
    if (!s.ready || !s.mapped) return;
    unsigned char *base = (unsigned char *)s.mapped;

    if (aircraftInv)
        memcpy(base + destruct::kOffAircraftInv, aircraftInv, 16 * sizeof(float));

    float *mc = (float *)(base + destruct::kOffGridMinCell);
    mc[0] = gridMin ? gridMin[0] : 0.0f;
    mc[1] = gridMin ? gridMin[1] : 0.0f;
    mc[2] = gridMin ? gridMin[2] : 0.0f;
    mc[3] = cell;

    int32_t *gd = (int32_t *)(base + destruct::kOffGridDim);
    gd[0] = nx; gd[1] = ny; gd[2] = nz; gd[3] = active;
}

// ---- OCCUPANCY.
//
// Cleared before a discovery frame and read after it. Both operate on the
// CELLS THE CURRENT GRID USES, not the whole region: leftovers from a previous,
// larger grid would otherwise read as occupied and seed fragments in mid-air.
inline void clearOccupancy(uint32_t cells)
{
    State &s = state();
    if (!s.ready || !s.mapped) return;
    if (cells > destruct::kMaxCells) cells = destruct::kMaxCells;
    memset((unsigned char *)s.mapped + destruct::kOffOccupancy, 0,
           (size_t)cells * 4u);
}

// Returns how many cells came back set. The count is the number the trace
// reports and the number Task 9's gate is judged on, so it is produced here
// rather than recomputed by each caller.
// The discard word, read on its own.
//
// This is the difference between "the shader ran and rejected everything" and
// "the shader never ran at all", which an occupancy count of zero cannot
// distinguish and which have completely different causes. Every vertex that is
// out of the grid, and every vertex at all while discovery is off, writes 1
// here - so a zero means no patched vertex shader executed the code, and a one
// means the code ran and the transform is wrong.
inline uint32_t readDiscard()
{
    State &s = state();
    if (!s.ready || !s.mapped) return 0;
    const uint32_t *p =
        (const uint32_t *)((const unsigned char *)s.mapped + destruct::kOffDiscard);
    return *p;
}

inline void clearDiscard()
{
    State &s = state();
    if (!s.ready || !s.mapped) return;
    uint32_t *p = (uint32_t *)((unsigned char *)s.mapped + destruct::kOffDiscard);
    *p = 0;
}

inline uint32_t readOccupancy(unsigned char *out, uint32_t cells)
{
    State &s = state();
    if (!s.ready || !s.mapped) return 0;
    if (cells > destruct::kMaxCells) cells = destruct::kMaxCells;
    const uint32_t *src =
        (const uint32_t *)((const unsigned char *)s.mapped + destruct::kOffOccupancy);
    uint32_t n = 0;
    for (uint32_t i = 0; i < cells; ++i) {
        const unsigned char v = src[i] ? 1u : 0u;
        if (out) out[i] = v;
        n += v;
    }
    return n;
}

inline void destroy(DeviceData &dd, VkDevice dev)
{
    State &s = state();
    if (s.mapped && dd.unmapMemory) { dd.unmapMemory(dev, s.mem); s.mapped = nullptr; }
    if (s.pool && dd.destroyDescriptorPool) dd.destroyDescriptorPool(dev, s.pool, nullptr);
    if (s.setLayout && dd.destroyDescriptorSetLayout)
        dd.destroyDescriptorSetLayout(dev, s.setLayout, nullptr);
    if (s.emptyLayout && dd.destroyDescriptorSetLayout)
        dd.destroyDescriptorSetLayout(dev, s.emptyLayout, nullptr);
    if (s.buf && dd.destroyBuffer) dd.destroyBuffer(dev, s.buf, nullptr);
    if (s.mem && dd.freeMemory)    dd.freeMemory(dev, s.mem, nullptr);
    s = State();
}

}  // namespace destructgpu
