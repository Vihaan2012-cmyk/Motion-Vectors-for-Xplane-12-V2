// control.h - the contract between the external control panel and the plugin.
//
// Included by BOTH src/plugin.cpp (reader) and src/panel/panel.cpp (writer).
//
// ============================================================== WHY SEPARATE
//
// The obvious place for this was TaaShare, which the plugin already publishes
// and which already carries every value the panel wants to change. Putting it
// there would have been one mapping instead of two.
//
// It is separate because TaaShare is a contract with the VULKAN LAYER, and that
// contract is checked by size: the layer reads structSize and bails if it
// disagrees. Adding fields for a control panel means a version bump, which
// means the layer and the plugin must be rebuilt and shipped together or the
// velocity path silently stops - a real risk taken on behalf of a window.
//
// A second mapping costs a page of address space and cannot break anything
// that was working. The layer never learns this exists.
//
// ========================================================== WHY NOT AN INI FILE
//
// taa.ini is read once, at plugin load, and the values it asks for are held
// until the layer reports what the GPU can do. That is right for a config file
// and useless for a panel: a control you have to restart the sim to see the
// effect of is not a control.
//
// ============================================================ WHY NOT DATAREFS
//
// The plugin already exposes everything here as a writable dataref, which is
// how the in-sim window and FlyWithLua drive it. But datarefs are reachable
// only from inside X-Plane's process - there is no way for a separate
// executable to write one without injecting into the sim, which is a far larger
// and far more fragile thing to do than sharing four hundred bytes.
//
// ================================================================= THE PROTOCOL
//
// The panel writes its desired values, then increments `seq`. The plugin reads
// `seq` once per flight loop and applies the block only when it has changed.
//
// A sequence number rather than a dirty flag the plugin clears. The panel and
// the sim are separate processes with no synchronisation between them, so a
// flag the reader writes back is a race: the panel can set the flag between the
// plugin's read and its clear, and that request is lost with nothing to
// indicate it. A counter only the writer touches cannot lose an edit - the
// reader either sees the new value or sees it next frame.
//
// `applied` runs the other way, plugin to panel, and is advisory: it is the
// last seq the plugin acted on, so the panel can show "connected" rather than
// leaving a dead sim looking like a working one.

#ifndef TAA_CONTROL_H
#define TAA_CONTROL_H

#include "product.h"

#include <stdint.h>

#define TAA_CONTROL_MAGIC   0x4C43414Du          // 'MACL'
#define TAA_CONTROL_VERSION 1
#define TAA_CONTROL_NAME    MV_CONTROL_NAME

// Which fields of a request the plugin should act on.
//
// A mask rather than "apply everything every time" because the panel is not the
// only writer: the in-sim window, the menu commands and any Lua script drive
// the same values. Without a mask, a panel that pushed its whole block on every
// edit would fight anything else touching the sim - move the sharpness slider
// and the upscaler you just picked from the in-sim menu snaps back to whatever
// the panel last displayed.
enum {
    TAA_CTL_ENABLED    = 1 << 0,
    TAA_CTL_UPSCALER   = 1 << 1,
    TAA_CTL_QUALITY    = 1 << 2,
    TAA_CTL_SHARPNESS  = 1 << 3,
    TAA_CTL_LOD_BIAS   = 1 << 4,
    TAA_CTL_OPT_FLOW   = 1 << 6,
    TAA_CTL_JITTER     = 1 << 7,   // jitter phase count
    TAA_CTL_OBJECTS    = 1 << 8,   // moving-object velocity
    TAA_CTL_TRAFFIC    = 1 << 10,  // AI traffic bounding radius
    TAA_CTL_XPFSR      = 1 << 12   // X-Plane's own render resolution
};

struct TaaControl {
    uint32_t magic;
    uint32_t version;
    uint32_t structSize;
    uint32_t pad0;

    uint32_t seq;         // panel increments after writing a request
    uint32_t applied;     // plugin echoes the last seq it acted on
    uint32_t mask;        // which fields below are meant
    uint32_t pad1;

    int32_t  enabled;
    float    lodBias;

    int32_t  jitterPhases;
    int32_t  movingObjects;
    float    trafficRadius;

    // X-PLANE'S OWN SUB-NATIVE RENDERING.
    //
    // -1 disables it and renders at native; 0..4 select the sim's own FSR
    // quality preset, which is the only way to make X-Plane render its 3D scene
    // smaller. Nothing in a Vulkan layer can do it: the size of every render
    // target is X-Plane's decision, and the passes writing them carry matching
    // viewports and shader constants.
    //
    // This is also the ONLY control here that changes performance rather than
    // memory. Render targets, depth buffers and the upscaler's storage images
    // are all sized by it, so it moves roughly 500 MB as well as the frame time.
    //
    // `bypass` asks X-Plane to skip its own spatial upscale so ours can do the
    // job with motion vectors and history instead. Whether that actually keeps
    // the reduced render or just turns the feature off is measured, not assumed.
    int32_t  xpFsrQuality;
    int32_t  xpFsrBypass;

    // Room to add controls without moving anything above, so an old panel and a
    // new plugin still agree on where `upscaler` lives. structSize catches the
    // case where they do not.
    //
    // Fields added since v1 came OUT OF HERE rather than being appended, so the
    // block stays the same size and an older panel talking to a newer plugin
    // still passes the size check instead of being rejected wholesale.
    uint32_t reserved[24];
};

#endif // TAA_CONTROL_H
