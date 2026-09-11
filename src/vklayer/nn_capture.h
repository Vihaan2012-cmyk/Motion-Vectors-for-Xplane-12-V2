// nn_capture.h - Stage 0 of the neural suite: full-G-buffer training tuples.
//
// Every nn.capture_every frames, at the resolve, the scene colour, velocity
// target, depth copy, engine normal, the resolve's own output and the G-buffer
// pass's colour attachments are copied into one host-visible buffer, read back
// a few presents later, and written by a worker thread as raw planes plus a
// JSON manifest - the format in docs/nn/capture-format.md. Modelled on the
// swapchain screenshot path (ShotCap): same readback buffer discipline, same
// "arm now, read N presents later" timing, but N images per frame, lossless,
// and never a file write on the render thread.
//
// Included by layer.cpp after the globals it reads (g_velSnap, g_colorImages,
// g_viewToImage, g_share, g_taaSunView, formatName, isSceneSized) and before
// Layer_CmdBeginRendering. Declarations: nn_capture_fwd.h.
//
// Copyright (C) 2026 MotionVectors contributors. SPDX: GPL-3.0-or-later
#pragma once
#include "nn_capture_fwd.h"
#include <vector>
#include <string>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <direct.h>

namespace nncap {

struct PlaneSrc {
    const char   *name;
    VkImage       image;
    VkFormat      format;
    uint32_t      w, h;
    VkImageLayout layout;
};

struct PlaneDesc {
    std::string  name, format, dtype;
    uint32_t     w = 0, h = 0, c = 0, bpp = 0;
    VkDeviceSize offset = 0, bytes = 0;
};

struct Slot {
    VkBuffer       buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void          *ptr = nullptr;
    VkDeviceSize   cap = 0, used = 0;
    bool           armed = false;
    int            wait = 0;
    std::vector<PlaneDesc> planes;
    std::string    manifest;
    std::string    dir;
    uint64_t       burstDone = 0;
    uint64_t       gbufFrame = 0;       // frame whose G-buffer planes already sit at [0, gbufEnd)
    VkDeviceSize   gbufEnd = 0;
    uint32_t       gbufCount = 0;       // attachments in the snapshot; a wider pass this frame replaces it
    std::vector<PlaneDesc> gbufPlanes;
};

struct Job {
    std::string dir, manifest;
    std::vector<PlaneDesc> planes;
    std::vector<unsigned char> data;
    uint64_t burstDone = 0;     // non-zero: this frame closes burst #burstDone
};

struct Gbuf {                       // the G-buffer pass's colour attachments, this frame
    uint64_t frame = 0;
    uint32_t count = 0;
    VkImage  image[8] = {};
    VkFormat format[8] = {};
    VkImageLayout layout[8] = {};
    VkCommandBuffer cb = VK_NULL_HANDLE;
    uint32_t w = 0, h = 0;
};

struct State {
    bool        failed = false, sessionWritten = false, workerUp = false;
    std::string sessionDir;
    uint64_t    nextFrame = 0, seq = 0, written = 0, dropped = 0;
    uint64_t    bytesWritten = 0;
    // Bursts: nn.capture_burst consecutive frames per capture, so a temporal
    // model gets the frame its velocity points back to. Eight slots hold a
    // burst of up to eight in flight (each slot waits four presents).
    Slot        slots[16];
    int         nextSlot = 0;
    int         burstLeft = 0;
    uint64_t    burstId = 0;
    int         burstIdx = 0;
    Gbuf        gbuf;
    VkDeviceSize lastResolveBytes = 0;
    // A slot buffer replaced while recorded-but-unexecuted copies still name it
    // must outlive those commands: destroying it there is a device-lost crash
    // (measured 2026-09-09, first capture frame). Retired buffers are freed by
    // pollImpl after eight presents.
    struct Retired { DeviceData *dd; VkBuffer buf; VkDeviceMemory mem; void *ptr; int wait; };
    std::vector<Retired> retired;
    std::mutex  sm;             // slots, nextSlot, retired: pass end and resolve may record on different threads
    std::mutex  qm;
    std::condition_variable qcv;
    std::deque<Job> q;
};

inline State &state() { static State s; return s; }

inline bool armed()        { return live::onoff("nn.capture", "TAA_NN_CAPTURE", false); }
inline int  everyFrames()  { int e = live::i("nn.capture_every", "TAA_NN_CAPTURE_EVERY", 120); return e < 8 ? 8 : e; }
inline bool gbufWanted()   { return live::onoff("nn.capture_gbuf", nullptr, true); }
inline int  burstLen()     { int k = live::i("nn.capture_burst", nullptr, 1); return k < 1 ? 1 : (k > 16 ? 16 : k); }
// Session cap, GB written to disk; the second guard beside free space.
inline double maxGb()      { return (double)live::f("nn.capture_max_gb", nullptr, 100.0f); }
// Stop before the drive is full: below this many GB free, capture stands down.
inline bool diskOk(const std::string &dir)
{
    ULARGE_INTEGER freeB; freeB.QuadPart = 0;
    std::string root = dir.size() >= 2 ? dir.substr(0, 3) : dir;      // "E:\"
    if (!GetDiskFreeSpaceExA(root.c_str(), &freeB, nullptr, nullptr)) return true;
    const double gb = (double)freeB.QuadPart / (1024.0 * 1024.0 * 1024.0);
    return gb > (double)live::f("nn.capture_min_free_gb", nullptr, 5.0f);
}
inline std::string root()  { std::string s; if (live::lookup("nn.capture_dir", s) && !s.empty()) return s; return "D:\\NNCap"; }
inline std::string tag()   { std::string s; if (live::lookup("nn.capture_tag", s) && !s.empty()) return s; return "user"; }

// Vulkan format -> (dtype, channels, bytes per pixel). Unknown = not captured.
inline bool describe(VkFormat f, const char **dtype, uint32_t *c, uint32_t *bpp)
{
    switch (f) {
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_A8B8G8R8_UNORM_PACK32: case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
        *dtype = "u8";  *c = 4; *bpp = 4;  return true;
    case VK_FORMAT_R8G8_UNORM:            *dtype = "u8";  *c = 2; *bpp = 2;  return true;
    case VK_FORMAT_R8_UNORM:              *dtype = "u8";  *c = 1; *bpp = 1;  return true;
    case VK_FORMAT_R16G16B16A16_SFLOAT:   *dtype = "f16"; *c = 4; *bpp = 8;  return true;
    case VK_FORMAT_R16G16_SFLOAT:         *dtype = "f16"; *c = 2; *bpp = 4;  return true;
    case VK_FORMAT_R16_SFLOAT:            *dtype = "f16"; *c = 1; *bpp = 2;  return true;
    case VK_FORMAT_R16G16B16A16_UNORM:    *dtype = "u16"; *c = 4; *bpp = 8;  return true;
    case VK_FORMAT_R16G16_UNORM:          *dtype = "u16"; *c = 2; *bpp = 4;  return true;
    case VK_FORMAT_R16_UNORM:             *dtype = "u16"; *c = 1; *bpp = 2;  return true;
    case VK_FORMAT_R32_SFLOAT:            *dtype = "f32"; *c = 1; *bpp = 4;  return true;
    case VK_FORMAT_R32G32_SFLOAT:         *dtype = "f32"; *c = 2; *bpp = 8;  return true;
    case VK_FORMAT_R32G32B32A32_SFLOAT:   *dtype = "f32"; *c = 4; *bpp = 16; return true;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
    case VK_FORMAT_R32_UINT:
        *dtype = "u32"; *c = 1; *bpp = 4;  return true;
    default: return false;
    }
}

// ---- the writer thread: files only, never the render thread.
inline void workerLoop()
{
    State &s = state();
    for (;;) {
        Job j;
        {
            std::unique_lock<std::mutex> lk(s.qm);
            s.qcv.wait(lk, [&] { return !s.q.empty(); });
            j = std::move(s.q.front()); s.q.pop_front();
        }
        _mkdir(j.dir.c_str());
        bool ok = true;
        for (size_t i = 0; i < j.planes.size(); ++i) {
            const PlaneDesc &p = j.planes[i];
            std::string path = j.dir + "\\" + p.name + ".bin";
            FILE *f = fopen(path.c_str(), "wb");
            if (!f) { ok = false; continue; }
            fwrite(j.data.data() + p.offset, 1, (size_t)p.bytes, f);
            fclose(f);
        }
        {
            std::string path = j.dir + "\\manifest.json";
            FILE *f = fopen(path.c_str(), "wb");
            if (f) { fwrite(j.manifest.data(), 1, j.manifest.size(), f); fclose(f); } else ok = false;
        }
        if (ok) { ++s.written; s.bytesWritten += j.data.size(); } else ++s.dropped;
        // The burst counter the panel script polls: it steps the sim clock once
        // per finished burst, so frames inside a burst share their lighting.
        if (j.burstDone) {
            std::string path = root() + "\\bursts.txt";
            FILE *f = fopen(path.c_str(), "wb");
            if (f) { fprintf(f, "%llu\n", (unsigned long long)j.burstDone); fclose(f); }
        }
        if ((s.written % 25) == 1 || !ok)
            trace("NN CAPTURE: %s %s (%llu written, %llu dropped)", ok ? "wrote" : "FAILED",
                  j.dir.c_str(), (unsigned long long)s.written, (unsigned long long)s.dropped);
    }
}

inline void ensureWorker()
{
    State &s = state();
    if (s.workerUp) return;
    s.workerUp = true;
    std::thread(workerLoop).detach();   // detached: dies with the process, no join at exit
}

// ---- JSON helpers (no library; the manifest is small and flat).
inline void jf(std::string &o, const char *key, double v)
{
    char b[96]; snprintf(b, sizeof(b), "\"%s\": %.9g", key, v); o += b;
}
inline void jarr(std::string &o, const char *key, const float *v, int n)
{
    o += "\""; o += key; o += "\": [";
    for (int i = 0; i < n; ++i) { char b[40]; snprintf(b, sizeof(b), "%s%.9g", i ? ", " : "", (double)v[i]); o += b; }
    o += "]";
}

inline std::string sessionName()
{
    time_t t = time(nullptr); struct tm lt; localtime_s(&lt, &t);
    char b[64]; strftime(b, sizeof(b), "%Y-%m-%d_%H%M%S", &lt);
    return std::string(b) + "_" + tag();
}

inline std::string liveSnapshot()
{
    static const char *keys[] = { "taa.alpha", "taa.contact", "taa.ao", "taa.sharpen", "taa.jitter_scale",
        "taa.pos_harvest", "taa.engine_depth", "taa.mode", "taa.gi", "taa.box_mod", "taa.unlock",
        "taa.kill_ssr", "taa.enable", "taa.accum_exp", "taa.conf_exp", "taa.chase_exp", "taa.chase_body",
        "taa.varclip", "taa.alpha_moving", "nn.capture_every", "nn.capture_tag", "nn.sweep_time", "nn.sweep_step" };
    std::string o = "{";
    bool first = true;
    for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); ++i) {
        std::string v; if (!live::lookup(keys[i], v)) continue;
        o += first ? "" : ", "; first = false;
        o += "\""; o += keys[i]; o += "\": \""; o += v; o += "\"";
    }
    return o + "}";
}

inline void writeSession(uint32_t w, uint32_t h)
{
    State &s = state();
    if (s.sessionWritten) return;
    s.sessionWritten = true;
    _mkdir(root().c_str());
    s.sessionDir = root() + "\\" + sessionName();
    _mkdir(s.sessionDir.c_str());
    std::string o = "{\n  \"schema\": 1,\n  \"layer_build\": \"" __DATE__ " " __TIME__ "\",\n";
    char b[160];
    snprintf(b, sizeof(b), "  \"render\": {\"w\": %u, \"h\": %u},\n  \"display\": {\"w\": %d, \"h\": %d},\n",
             w, h, g_share ? g_share->viewportW : 0, g_share ? g_share->viewportH : 0);
    o += b;
    o += "  \"tag\": \""; o += tag(); o += "\",\n  \"live\": "; o += liveSnapshot(); o += "\n}\n";
    FILE *f = fopen((s.sessionDir + "\\session.json").c_str(), "wb");
    if (f) { fwrite(o.data(), 1, o.size(), f); fclose(f); }
    trace("NN CAPTURE: session %s (every %d frames, gbuf=%d)", s.sessionDir.c_str(), everyFrames(), gbufWanted() ? 1 : 0);
}

inline bool ensureSlot(DeviceData &dd, Slot &sl, VkDeviceSize need)
{
    if (sl.buf != VK_NULL_HANDLE && sl.cap >= need) return true;
    if (sl.buf != VK_NULL_HANDLE) {
        State::Retired r; r.dd = &dd; r.buf = sl.buf; r.mem = sl.mem; r.ptr = sl.ptr; r.wait = 8;
        state().retired.push_back(r);
        sl.buf = VK_NULL_HANDLE; sl.mem = VK_NULL_HANDLE; sl.ptr = nullptr; sl.cap = 0;
    }
    VkBufferCreateInfo bci; memset(&bci, 0, sizeof(bci));
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = need; bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (dd.createBuffer(dd.device, &bci, nullptr, &sl.buf) != VK_SUCCESS) return false;
    VkMemoryRequirements mr; dd.getBufferMemReq(dd.device, sl.buf, &mr);
    VkPhysicalDeviceMemoryProperties mp; memset(&mp, 0, sizeof(mp)); g_getPhysMemProps(dd.phys, &mp);
    uint32_t ti = UINT32_MAX;
    for (uint32_t k = 0; k < mp.memoryTypeCount; ++k)
        if ((mr.memoryTypeBits & (1u << k)) &&
            (mp.memoryTypes[k].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (mp.memoryTypes[k].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { ti = k; break; }
    VkMemoryAllocateInfo mai; memset(&mai, 0, sizeof(mai));
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; mai.allocationSize = mr.size; mai.memoryTypeIndex = ti;
    if (ti == UINT32_MAX || dd.allocateMemory(dd.device, &mai, nullptr, &sl.mem) != VK_SUCCESS ||
        dd.bindBufferMemory(dd.device, sl.buf, sl.mem, 0) != VK_SUCCESS ||
        dd.mapMemory(dd.device, sl.mem, 0, need, 0, &sl.ptr) != VK_SUCCESS) {
        dd.destroyBuffer(dd.device, sl.buf, nullptr); sl.buf = VK_NULL_HANDLE;
        if (sl.mem) { dd.freeMemory(dd.device, sl.mem, nullptr); sl.mem = VK_NULL_HANDLE; }
        return false;
    }
    sl.cap = need;
    return true;
}

inline void copyPlane(DeviceData &dd, VkCommandBuffer cb, const PlaneSrc &p, VkBuffer buf, VkDeviceSize off)
{
    VkImageMemoryBarrier b; memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = p.image;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1; b.subresourceRange.layerCount = 1;
    b.oldLayout = p.layout; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                          0, nullptr, 0, nullptr, 1, &b);
    VkBufferImageCopy bic; memset(&bic, 0, sizeof(bic));
    bic.bufferOffset = off;
    bic.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bic.imageSubresource.layerCount = 1;
    bic.imageExtent.width = p.w; bic.imageExtent.height = p.h; bic.imageExtent.depth = 1;
    dd.cmdCopyImageToBuffer(cb, p.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &bic);
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; b.newLayout = p.layout;   // hand it back as found
    b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                          0, nullptr, 0, nullptr, 1, &b);
}

// The per-frame manifest, from the CPU-side snapshot the resolve already uses.
inline std::string manifest(const std::vector<PlaneDesc> &planes, uint32_t w, uint32_t h, uint32_t layers)
{
    const Snapshot &sn = g_velSnap;
    std::string o = "{\n  \"schema\": 1,\n";
    char b[256];
    snprintf(b, sizeof(b), "  \"frame\": %llu,\n  \"sim_time\": %.6f,\n  \"render\": {\"w\": %u, \"h\": %u, \"layers\": %u},\n"
             "  \"display\": {\"w\": %d, \"h\": %d},\n  \"render_scale\": %.6f,\n",
             (unsigned long long)g_frameCount, sn.simTime, w, h, layers, sn.viewportW, sn.viewportH,
             sn.viewportW > 0 ? (double)w / (double)sn.viewportW : 0.0);
    o += b;
    snprintf(b, sizeof(b), "  \"jitter\": {\"x\": %.9g, \"y\": %.9g, \"index\": %d, \"phases\": %d, \"scale\": %.9g},\n",
             (double)sn.jitterX, (double)sn.jitterY, sn.jitterIndex, sn.jitterPhases,
             (double)live::f("taa.jitter_scale", "TAA_JITTER_SCALE", 1.0f));
    o += b;
    o += "  "; jarr(o, "proj", sn.proj, 16); o += ",\n";
    o += "  "; jarr(o, "world", sn.world, 16); o += ",\n";
    o += "  "; jarr(o, "reproj", sn.reproj, 16); o += ",\n";
    snprintf(b, sizeof(b), "  \"cam\": {\"x\": %.9g, \"y\": %.9g, \"z\": %.9g},\n  \"fov_deg\": %.6g, \"near\": %.9g, \"far\": %.9g, "
             "\"reverse_z\": %d, \"view_type\": %d,\n", (double)sn.camX, (double)sn.camY, (double)sn.camZ,
             (double)sn.fovDeg, (double)sn.nearClip, (double)sn.farClip, sn.reverseZ, sn.viewType);
    o += b;
    o += "  \"clip_info\": null,\n  ";
    jarr(o, "sun_view", g_taaSunView, 3); o += ",\n  \"planes\": {\n";
    for (size_t i = 0; i < planes.size(); ++i) {
        const PlaneDesc &p = planes[i];
        snprintf(b, sizeof(b), "    \"%s\": {\"format\": \"%s\", \"w\": %u, \"h\": %u, \"c\": %u, \"dtype\": \"%s\"}%s\n",
                 p.name.c_str(), p.format.c_str(), p.w, p.h, p.c, p.dtype.c_str(), i + 1 < planes.size() ? "," : "");
        o += b;
    }
    o += "  },\n  \"live\": "; o += liveSnapshot(); o += "\n}\n";
    return o;
}

inline void noteScenePassImpl(VkCommandBuffer cb, const VkRenderingInfo *info)
{
    if (!info || info->colorAttachmentCount < 2 || !info->pColorAttachments) return;
    if (!isSceneSized(info->renderArea.extent.width, info->renderArea.extent.height, info->colorAttachmentCount)) return;
    Gbuf &g = state().gbuf;
    if (g.frame == g_frameCount && info->colorAttachmentCount <= g.count) return;   // keep the widest pass this frame
    g.frame = g_frameCount; g.count = 0; g.cb = cb;
    g.w = info->renderArea.extent.width; g.h = info->renderArea.extent.height;
    for (uint32_t i = 0; i < info->colorAttachmentCount && g.count < 8; ++i) {
        VkImageView v = info->pColorAttachments[i].imageView;
        if (v == VK_NULL_HANDLE) continue;
        std::map<VkImageView, VkImage>::iterator vi = g_viewToImage.find(v);
        if (vi == g_viewToImage.end()) continue;
        std::map<VkImage, ColorTarget>::iterator ci = g_colorImages.find(vi->second);
        if (ci == g_colorImages.end()) continue;
        g.image[g.count] = vi->second; g.format[g.count] = ci->second.format;
        g.layout[g.count] = info->pColorAttachments[i].imageLayout; ++g.count;
    }
}

// The G-buffer attachments are copied the moment their pass ends. By resolve
// time X-Plane has moved on - layouts changed, transient memory reused - and a
// copy there reads scrambled data (measured 2026-09-09: every gbuf plane and
// the engine normal came back as block-shuffled noise while colour, which the
// resolve barriers itself, was clean). Here the layout is the one the pass
// declared and the contents are what the pass wrote.
inline bool willCapture()
{
    State &s = state();
    return armed() && !s.failed && (s.burstLeft > 0 || g_frameCount >= s.nextFrame);
}

inline void noteScenePassEndImpl(DeviceData &dd, VkCommandBuffer cb)
{
    State &s = state();
    if (!gbufWanted() || !willCapture()) return;
    Gbuf gb;
    { std::lock_guard<std::mutex> g(g_lock); gb = s.gbuf; }
    if (gb.frame != g_frameCount || gb.cb != cb || !gb.count || !gb.w || !gb.h) return;
    if (!dd.cmdCopyImageToBuffer || !dd.createBuffer || !g_getPhysMemProps) return;
    std::lock_guard<std::mutex> lk(s.sm);
    Slot &sl = s.slots[s.nextSlot];
    if (sl.armed) return;                                   // the resolve skips this frame too
    if (sl.gbufFrame == g_frameCount && gb.count <= sl.gbufCount) return;   // already have this frame's widest pass
    static char gbName[8][8] = { "gbuf_0", "gbuf_1", "gbuf_2", "gbuf_3", "gbuf_4", "gbuf_5", "gbuf_6", "gbuf_7" };
    std::vector<PlaneSrc> src; std::vector<PlaneDesc> descs;
    VkDeviceSize off = 0;
    for (uint32_t i = 0; i < gb.count; ++i) {
        const char *dtype; uint32_t c, bpp;
        if (!describe(gb.format[i], &dtype, &c, &bpp)) continue;
        // The engine normal is one of these attachments: give it the name the
        // trainers already read, so the clean pass-end copy replaces the resolve-time one.
        const bool isNorm = (gb.image[i] == g_taa.normImage);
        PlaneDesc d; d.name = isNorm ? "normal" : gbName[i]; d.format = formatName(gb.format[i]); d.dtype = dtype;
        d.w = gb.w; d.h = gb.h; d.c = c; d.bpp = bpp;
        d.offset = off; d.bytes = (VkDeviceSize)bpp * gb.w * gb.h;
        off = (off + d.bytes + 255) & ~(VkDeviceSize)255;
        descs.push_back(d); src.push_back({ isNorm ? "normal" : gbName[i], gb.image[i], gb.format[i], gb.w, gb.h, gb.layout[i] });
    }
    if (src.empty()) return;
    // Reserve the resolve's planes now so the slot never regrows under this snapshot.
    const VkDeviceSize reserve = s.lastResolveBytes ? s.lastResolveBytes : ((VkDeviceSize)32 * gb.w * gb.h + 6 * 256);
    if (!ensureSlot(dd, sl, off + reserve)) { s.failed = true; trace("NN CAPTURE: readback buffer failed (%llu bytes) - capture off", (unsigned long long)(off + reserve)); return; }
    for (size_t i = 0; i < src.size(); ++i) copyPlane(dd, cb, src[i], sl.buf, descs[i].offset);
    sl.gbufFrame = g_frameCount; sl.gbufEnd = off; sl.gbufPlanes = descs; sl.gbufCount = gb.count;
    static int said = 0;
    if (said++ < 2) trace("NN CAPTURE: G-buffer snapshot at pass end: %d planes, %llu bytes, layout %d", (int)src.size(), (unsigned long long)off, (int)gb.layout[0]);
}

inline void recordImpl(DeviceData &dd, VkCommandBuffer cb, const ResolvePlanes &rp)
{
    State &s = state();
    if (s.failed || !armed()) return;
    if (rp.layers != 1 || !rp.w || !rp.h) return;
    if (s.burstLeft <= 0) {
        if (g_frameCount < s.nextFrame) return;
        s.nextFrame = g_frameCount + (uint64_t)everyFrames();
        s.burstLeft = burstLen(); ++s.burstId; s.burstIdx = 0;
        if (!diskOk(root()) || (double)s.bytesWritten / (1024.0 * 1024.0 * 1024.0) >= maxGb()) {
            static uint64_t said = 0;
            if ((said++ % 50) == 0) trace("NN CAPTURE: nn.capture_max_gb reached or drive below nn.capture_min_free_gb - standing down (%.1f GB written)", (double)s.bytesWritten / (1024.0 * 1024.0 * 1024.0));
            s.burstLeft = 0; return;
        }
    }
    --s.burstLeft;
    const int burstIdx = s.burstIdx++;
    if (!dd.cmdCopyImageToBuffer || !dd.createBuffer || !g_getPhysMemProps) { s.failed = true; return; }

    std::lock_guard<std::mutex> lk(s.sm);
    Slot &sl = s.slots[s.nextSlot];
    if (sl.armed) { ++s.dropped; return; }                 // still in flight: skip this frame, no stall
    ensureWorker();
    writeSession(rp.w, rp.h);

    // Assemble the plane list. Formats for X-Plane's images come from the
    // colour census; ours are known.
    std::vector<PlaneSrc> src;
    VkFormat sceneFmt = VK_FORMAT_UNDEFINED;
    {
        std::lock_guard<std::mutex> g(g_lock);
        std::map<VkImage, ColorTarget>::iterator ci = g_colorImages.find(rp.scene);
        if (ci != g_colorImages.end()) sceneFmt = ci->second.format;
    }
    if (rp.scene && sceneFmt != VK_FORMAT_UNDEFINED) src.push_back({ "color", rp.scene, sceneFmt, rp.w, rp.h, rp.sceneLayout });
    if (rp.velocity) src.push_back({ "velocity", rp.velocity, rp.velocityFormat, rp.w, rp.h, rp.velocityLayout });
    if (rp.depth)    src.push_back({ "depth", rp.depth, VK_FORMAT_R32_SFLOAT, rp.w, rp.h, rp.depthLayout });
    bool snapNormal = false;                                // pass-end snapshot already holds a clean "normal"
    if (sl.gbufFrame == g_frameCount)
        for (size_t i = 0; i < sl.gbufPlanes.size(); ++i) if (std::string(sl.gbufPlanes[i].name) == "normal") snapNormal = true;
    if (rp.normal && rp.normalFormat != VK_FORMAT_UNDEFINED && !snapNormal) src.push_back({ "normal", rp.normal, rp.normalFormat, rp.w, rp.h, rp.normalLayout });
    if (rp.resolved) src.push_back({ "resolved", rp.resolved, rp.resolvedFormat, rp.w, rp.h, rp.resolvedLayout });
    // Byte layout: 256-aligned offsets, one buffer.
    sl.planes.clear();
    VkDeviceSize off = 0;
    if (sl.gbufFrame == g_frameCount && !sl.gbufPlanes.empty()) { sl.planes = sl.gbufPlanes; off = sl.gbufEnd; }
    else if (gbufWanted()) {
        static int said = 0;
        if (said++ < 4) trace("NN CAPTURE: frame %llu has no G-buffer snapshot (pass end missed)", (unsigned long long)g_frameCount);
    }
    size_t nGb = sl.planes.size();
    const VkDeviceSize start = off;
    std::vector<PlaneSrc> keep;
    for (size_t i = 0; i < src.size(); ++i) {
        const char *dtype; uint32_t c, bpp;
        if (!describe(src[i].format, &dtype, &c, &bpp)) {
            static int said = 0;
            if (said++ < 8) trace("NN CAPTURE: plane %s has unsupported format %s - skipped", src[i].name, formatName(src[i].format));
            continue;
        }
        PlaneDesc d; d.name = src[i].name; d.format = formatName(src[i].format); d.dtype = dtype;
        d.w = src[i].w; d.h = src[i].h; d.c = c; d.bpp = bpp;
        d.offset = off; d.bytes = (VkDeviceSize)bpp * src[i].w * src[i].h;
        off = (off + d.bytes + 255) & ~(VkDeviceSize)255;
        sl.planes.push_back(d); keep.push_back(src[i]);
    }
    if (keep.empty()) return;
    s.lastResolveBytes = off - start;
    if (nGb && (sl.buf == VK_NULL_HANDLE || sl.cap < off)) {          // regrow would orphan the snapshot: drop it, keep the offsets
        sl.planes.erase(sl.planes.begin(), sl.planes.begin() + nGb); nGb = 0;
    }
    if (!ensureSlot(dd, sl, off)) { s.failed = true; trace("NN CAPTURE: readback buffer failed (%llu bytes) - capture off", (unsigned long long)off); return; }

    for (size_t i = 0; i < keep.size(); ++i) copyPlane(dd, cb, keep[i], sl.buf, sl.planes[nGb + i].offset);

    char b[64]; snprintf(b, sizeof(b), "\\frame_%06llu", (unsigned long long)s.seq++);
    sl.dir = s.sessionDir + b;
    sl.manifest = manifest(sl.planes, rp.w, rp.h, rp.layers);
    {   // burst tag, spliced in after the opening brace: id, index within, length
        char bb[96]; snprintf(bb, sizeof(bb), "  \"burst\": {\"id\": %llu, \"index\": %d, \"len\": %d},\n",
                              (unsigned long long)s.burstId, burstIdx, burstLen());
        sl.manifest.insert(sl.manifest.find('\n') + 1, bb);
    }
    sl.used = off; sl.armed = true; sl.wait = 4; sl.gbufFrame = 0; sl.gbufCount = 0;
    sl.burstDone = (s.burstLeft == 0) ? s.burstId : 0;     // last frame of the burst
    s.nextSlot = (s.nextSlot + 1) % 16;
}

inline void pollImpl()
{
    State &s = state();
    std::lock_guard<std::mutex> lk(s.sm);
    for (size_t i = 0; i < s.retired.size();) {
        State::Retired &r = s.retired[i];
        if (--r.wait > 0) { ++i; continue; }
        if (r.ptr) r.dd->unmapMemory(r.dd->device, r.mem);
        r.dd->destroyBuffer(r.dd->device, r.buf, nullptr);
        r.dd->freeMemory(r.dd->device, r.mem, nullptr);
        s.retired.erase(s.retired.begin() + i);
    }
    for (int i = 0; i < 16; ++i) {
        Slot &sl = s.slots[i];
        if (!sl.armed) continue;
        if (--sl.wait > 0) continue;
        Job j; j.dir = sl.dir; j.manifest = sl.manifest; j.planes = sl.planes; j.burstDone = sl.burstDone;
        j.data.resize((size_t)sl.used);
        memcpy(j.data.data(), sl.ptr, (size_t)sl.used);     // ~50 MB, a few ms, once per capture
        sl.armed = false;
        std::lock_guard<std::mutex> lk(s.qm);
        if (s.q.size() < 24) { s.q.push_back(std::move(j)); s.qcv.notify_one(); } else ++s.dropped;
    }
}

void recordFromResolve(DeviceData &dd, VkCommandBuffer cb, const ResolvePlanes &p) { recordImpl(dd, cb, p); }
void noteScenePassEnd(DeviceData &dd, VkCommandBuffer cb) { noteScenePassEndImpl(dd, cb); }
void poll() { pollImpl(); }
bool captureArmed() { return armed(); }
void noteScenePass(VkCommandBuffer cb, const VkRenderingInfo *info) { noteScenePassImpl(cb, info); }

} // namespace nncap
