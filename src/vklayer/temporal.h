#pragma once

// THE TEMPORAL CORE: one contract between the Vulkan layer and every consumer.
//
//                    X-PLANE 12
//                        |
//                 +-------------+
//                 | Vulkan Layer|      knows HOW TO OBTAIN resources
//                 +-------------+
//                        |
//          +-------------+-------------+
//    Resource Capture              GPU Timing
//          +-------------+-------------+
//                        v
//                 +-------------+
//                 |Temporal Core|      knows WHAT THEY MEAN   <- this file
//                 +-------------+
//                        |
//                 +-------------+
//                 | Backend API |      TemporalFrame
//                 +-------------+
//                        |
//      +-----------------+-----------------+
//      v                 v                 v
// Reconstruction        AA            Frame Gen
// FSR DLSS XeSS      TAA  DLAA      FSR DLSS XeSS FG
//      +-----------------+-----------------+
//                        v
//                  PRESENTATION
//
// The boundary is hard in both directions. No backend code reaches into the
// interception code, and no X-Plane knowledge appears below the split. The
// alternative is what this project would otherwise become:
//
//     layer.cpp
//       |-- FSR shit
//       |-- DLSS shit
//       |-- XeSS shit
//       |-- TAA shit
//       +-- Streamline shit
//
// which is unmaintainable, and - more to the point here - makes it impossible
// to tell a plumbing bug from an algorithm bug. That distinction is the single
// most expensive thing in this project's history. Nearly every defect in the TAA
// chain was found only once the two were separable, and MODE_PASSTHROUGH exists
// in taa.comp for exactly that reason. This file makes the separation
// structural instead of something each consumer reinvents.

#include <stdint.h>

namespace temporal {

// ---- MOTION VECTORS ARE NEVER A BARE TEXTURE.
//
// This is the most important type here, and the reason is a failure mode this
// project is unusually well placed to walk into:
//
//     "FSR looks okay but DLSS ghosts horribly and XeSS moves everything
//      backwards."
//
// Every one of those is the same bug - a convention mismatch - and none of them
// announces itself. They look like quality problems, so they get debugged as
// quality problems, for days. XeSS alone distinguishes NDC from pixel-space
// velocity, expects vectors that EXCLUDE camera jitter, and accepts low-res
// vectors as a performance option: three separate places our field could be
// handed over wrong while looking entirely plausible on screen.
//
// So the metadata travels with the image and is never implicit. An adapter
// converts from a DECLARATION, not from an assumption.
enum CoordSpace { COORD_PIXEL, COORD_UV, COORD_NDC };

// Which endpoint the vector points FROM. Naming it after the endpoints rather
// than "forward"/"backward" because those two words are used in opposite senses
// by different SDKs, which is precisely the confusion this type exists to end.
enum Direction { DIR_CURRENT_TO_PREVIOUS, DIR_PREVIOUS_TO_CURRENT };

struct MotionVectors {
    VkImage      image  = VK_NULL_HANDLE;
    VkImageView  view   = VK_NULL_HANDLE;       // array-typed; see Texture::view
    VkFormat     format = VK_FORMAT_UNDEFINED;
    uint32_t     w = 0, h = 0;
    uint32_t     layers = 1;

    CoordSpace   coordinateSpace     = COORD_UV;
    Direction    direction           = DIR_PREVIOUS_TO_CURRENT;
    bool         jitterIncluded      = false;
    bool         cameraMotionIncluded = true;
    bool         objectMotionIncluded = true;
};

// ---- A TEXTURE CARRIES ITS SHAPE, NOT JUST ITS HANDLE.
//
// Passing a bare VkImage is exactly how the array-layered and multisampled cases
// stayed invisible until the shader corpus was read. gbuffer_lit exists in three
// shapes - plain 2D (288 permutations), arrayed (288), and arrayed+multisampled
// (576) - and a resolve that assumed the first silently reached one eye in
// stereo and was outright invalid under MSAA. That was the black that depended
// on the camera view.
//
// A consumer needs samples and layers to decide whether it can act at all, so
// they are part of the type rather than something to look up.
struct Texture {
    VkImage               image   = VK_NULL_HANDLE;
    VkImageView           view    = VK_NULL_HANDLE;
    VkFormat              format  = VK_FORMAT_UNDEFINED;
    uint32_t              w = 0, h = 0;
    uint32_t              layers  = 1;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
};

struct Jitter { float x = 0.0f, y = 0.0f; };   // NDC, this frame's offset

struct CameraState {
    float view[16];
    float proj[16];
    float prevView[16];
    float prevProj[16];
    // curr clip -> prev clip. The identity exactly when nothing moved, which is
    // the only honest test of "did the camera move": translation alone misses a
    // camera that rotates in place while moving every pixel on screen.
    float reproj[16];
    bool  moved = false;
};

// ---- ONE DEFINITION OF WHAT INVALIDATES HISTORY.
//
// A missed reset is a whole frame of the wrong picture dragged forward, and it
// presents as a quality problem rather than a state problem - which is how it
// survives review. Several of these fire constantly in a flight simulator and
// are easy to forget: the external/internal view toggle is a camera cut, replay
// scrubbing is a teleport, a scenery load is a scene transition. All three
// produce a frame where reprojection is meaningless.
//
// No backend may invent its own.
enum ResetReason {
    RESET_NONE            = 0,
    RESET_FIRST_FRAME     = 1 << 0,
    RESET_CAMERA_CUT      = 1 << 1,
    RESET_TELEPORT        = 1 << 2,
    RESET_RESOLUTION      = 1 << 3,
    RESET_BACKEND_CHANGE  = 1 << 4,
    RESET_WINDOW_RESIZE   = 1 << 5,
    RESET_SWAPCHAIN       = 1 << 6,
    RESET_SCENE_LOAD      = 1 << 7,
    RESET_TARGET_SWAP     = 1 << 8,   // X-Plane alternates two HDR targets
};

inline const char *resetReasonName(uint32_t r)
{
    if (r & RESET_FIRST_FRAME)    return "first-frame";
    if (r & RESET_CAMERA_CUT)     return "camera-cut";
    if (r & RESET_TELEPORT)       return "teleport";
    if (r & RESET_RESOLUTION)     return "resolution-change";
    if (r & RESET_BACKEND_CHANGE) return "backend-change";
    if (r & RESET_WINDOW_RESIZE)  return "window-resize";
    if (r & RESET_SWAPCHAIN)      return "swapchain-recreated";
    if (r & RESET_SCENE_LOAD)     return "scene-load";
    if (r & RESET_TARGET_SWAP)    return "target-swap";
    return "none";
}

// ---- THE CONTRACT.
//
// One of these per frame. Each backend translates it into whatever its SDK
// expects; nothing above this line knows a vendor exists.
struct TemporalFrame {
    Texture       color;              // lit HDR, pre-tonemap
    Texture       depth;
    MotionVectors motion;
    Jitter        jitter;
    float         exposure = 1.0f;
    uint32_t      renderW = 0, renderH = 0;
    uint32_t      outputW = 0, outputH = 0;
    uint64_t      frameIndex = 0;
    float         deltaTime = 0.0f;
    CameraState   camera;
    uint32_t      reset = RESET_NONE;   // bitmask of ResetReason
};

// ---- WHAT A BACKEND IS, DECLARED RATHER THAN INFERRED.
//
// So the launcher enumerates what is ACTUALLY available instead of matching a
// GPU name. A vendor table gets this wrong in both directions: FSR and XeSS run
// on NVIDIA and AMD alike, and the hardware-generation gates (FSR 4 wants
// RDNA4-class matrix hardware, DLSS-G wants Ada or later, MFG wants Blackwell)
// belong to each SDK's own capability query.
//
// Query, do not infer. It is the same rule that makes our varying Locations work
// here and fail silently on a minimum-specification device.
enum Vendor { VENDOR_ANY, VENDOR_AMD, VENDOR_NVIDIA, VENDOR_INTEL };

struct BackendInfo {
    const char *name    = "";
    Vendor      vendor  = VENDOR_ANY;
    bool supportsUpscaling        = false;
    bool supportsFrameGeneration  = false;
    bool supportsNativeResolution = false;   // DLAA is this bit on DLSS
    bool supportsDynamicResolution = false;
    // Shapes it can actually consume. Declared so that refusing is a normal,
    // reportable outcome rather than a crash or a corrupted frame.
    bool supportsArrayLayers      = false;
    bool supportsMultisample      = false;
};

// ---- THE INTERFACE.
//
// accepts() is not decoration. It is the array/multisample lesson made
// mandatory: a backend that cannot handle a target must say so and be skipped,
// and the core must log the refusal. Every black frame in this project's history
// came from a component proceeding on a target it did not understand.
class IBackend {
public:
    virtual ~IBackend() {}
    virtual BackendInfo info() const = 0;
    virtual bool accepts(const TemporalFrame &f, const char **why) const = 0;
    virtual bool record(VkCommandBuffer cb, const TemporalFrame &f) = 0;
};

} // namespace temporal
