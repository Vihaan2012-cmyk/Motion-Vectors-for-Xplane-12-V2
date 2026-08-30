// ============================================================================
//  temporal_core.h - WHAT THE FRAME IS, IN THE ENGINE'S OWN TERMS.
// ============================================================================
//
// Every temporal technique - our resolve, FSR3, DLSS, frame interpolation -
// needs the same six or seven facts about a frame: the colour, the depth, the
// motion vectors, the jitter that was applied, the camera, and how long the
// frame took. They differ only in the UNITS and SIGNS they want those facts in.
//
// Until now each consumer reached into the layer's globals and applied its own
// conversion inline at the call site. The FSR3 dispatch does it here:
//
//     const float vs = taaVelScale() * live::f("fsr.mv_x", nullptr, 1.0f);
//     const float ys = taaVelYSign() * live::f("fsr.mv_y", nullptr, 1.0f);
//     ... vs * (float)w, vs * ys * (float)h ...
//
// That works, and it is exactly how a sign error becomes a week of work: the
// convention lives at the call site, so every new consumer re-derives it, and
// when one of them is wrong nothing says which. This file exists so the
// conversion happens ONCE PER BACKEND, in a named place, next to the reason.
//
// ---- THE RULE THIS FILE ENFORCES
//
// A Frame carries RAW values in the engine's own convention. It performs no
// conversion for anybody. Backends never read it directly: they go through an
// adapter in temporal_adapt.h that states, in one place, what that backend
// wants and why. Adding a third upscaler means writing a third adapter, not
// touching this struct and not touching any of the others.
//
// The units below are not a suggestion. They are what the layer actually
// produces, established by measurement and recorded here so the next consumer
// does not have to rediscover them:
//
//   motion vectors  UV of the RENDER target, pointing CURRENT -> PREVIOUS,
//                   with a separate Y sign because X-Plane draws with a
//                   negative-height viewport. Stored with a large negative
//                   sentinel where no shader wrote.
//   jitter          NDC, the offset that was ADDED to the projection.
//   depth           whatever the engine uses; reversed and infinite-far are
//                   flags here, never assumed.
//
// Nothing in this file calls Vulkan or touches layer state. It is a
// description, so it can be unit-reasoned about without a device.
// ============================================================================
#pragma once

#include <string.h>
#include <stdint.h>

namespace tcore {

// ---- MOTION VECTOR UNITS ---------------------------------------------------
enum MvUnit {
    kMvUv     = 0,   // fraction of the render target, 0..1
    kMvPixels = 1,   // render-resolution pixels
    kMvNdc    = 2    // -1..1
};

// Which way the vector points. Getting this backwards does not fail loudly -
// it reprojects to the mirror image of the right place, which reads as
// smearing in the wrong direction and is easy to mistake for bad tuning.
enum MvDirection {
    kMvToPrevious = 0,   // ours: where this pixel WAS. A history fetch adds it.
    kMvToCurrent  = 1
};

struct Image {
    void         *image   = 0;   // VkImage, opaque here so this header needs no vulkan.h
    void         *view    = 0;   // VkImageView
    uint32_t      format  = 0;   // VkFormat as an integer
    int32_t       layout  = 0;   // VkImageLayout as an integer
    uint32_t      w = 0, h = 0, layers = 1;
    bool valid() const { return image != 0 && w && h; }
};

// ---- THE FRAME -------------------------------------------------------------
struct Frame {
    uint64_t index = 0;          // monotonic present counter
    bool     valid = false;

    Image colour;                // pre-tonemap scene, render resolution
    Image depth;                 // the engine's depth
    Image motion;                // our velocity target
    Image output;                // where an upscaler should write, display res
    Image reactive;              // optional; absent is normal

    uint32_t renderW = 0, renderH = 0;
    uint32_t outW    = 0, outH    = 0;

    // ---- motion vector convention, AS STORED
    MvUnit      mvUnit  = kMvUv;
    MvDirection mvDir   = kMvToPrevious;
    float       mvYSign = -1.0f;      // X-Plane's negative-height viewport
    float       mvScale = 1.0f;       // taa.vel_scale; 1.0 is the shipped belief
    float       mvSentinel = -1000.0f; // below this means "nothing wrote here"

    // ---- jitter, AS APPLIED to the projection, in NDC
    float jitterNdcX = 0.0f, jitterNdcY = 0.0f;
    uint32_t jitterIndex = 0, jitterPhases = 8;

    // ---- depth convention. Flags, never assumptions.
    bool  depthReversed    = false;   // 1.0 at the near plane
    bool  depthInfiniteFar = false;
    float nearPlane = 0.0f, farPlane = 0.0f;
    // The engine's own rational linearisation, transcribed from its deferred
    // shader: z = (d*clip[0] + clip[1]) / (d*clip[2] + clip[3]). Exact where a
    // two-constant form is only a special case.
    float clipInfo[4] = { 0, 0, 0, 0 };

    // ---- camera
    float proj[16]     = { 0 };
    float prevProj[16] = { 0 };
    float reproj[16]   = { 0 };       // prevViewProj * inverse(currViewProj)
    float fovYRad      = 0.0f;
    bool  reprojValid  = false;

    // ---- frame
    float frameTimeMs  = 16.6f;
    float exposure     = 1.0f;        // 1.0 = the engine applies its own
    bool  cameraCut    = false;       // history must be discarded this frame

    void reset() { *this = Frame(); }
};

// ---- SANITY, SO A BAD FRAME IS REFUSED RATHER THAN CONVERTED ---------------
//
// An adapter that silently converts a zero-sized frame produces a plausible
// looking set of numbers that describe nothing, and the backend then fails
// somewhere far away. Cheap to check here, once.
inline bool usable(const Frame &f, const char **whyNot = 0)
{
    const char *why = 0;
    if (!f.valid)                     why = "frame not marked valid";
    else if (!f.colour.valid())       why = "no colour image";
    else if (!f.depth.valid())        why = "no depth image";
    else if (!f.motion.valid())       why = "no motion vector image";
    else if (!f.renderW || !f.renderH) why = "render size is zero";
    else if (!f.outW || !f.outH)      why = "output size is zero";
    else if (f.frameTimeMs <= 0.0f)   why = "frame time is not positive";
    if (whyNot) *whyNot = why;
    return why == 0;
}

} // namespace tcore
