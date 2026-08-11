// share.h - the contract between the XPLM plugin and the Vulkan layer.
//
// Included by BOTH src/plugin.cpp (writer) and src/vklayer/layer.cpp (reader).
// One definition, deliberately: two hand-copied structs in two files is a
// silent-corruption bug waiting to happen - add a field on one side and the
// other reads garbage with nothing anywhere to indicate it.
//
// ============================================================ WHY THIS EXISTS
//
// X-Plane renders no velocity buffer. TAA cannot resolve without one: with no
// per-pixel motion it cannot know which texel of the history buffer belongs to
// this pixel, and the result is either a smeared mess or no accumulation at all.
//
// But a flight sim is the favourable case. Terrain, ortho, buildings and
// airports are STATIC; only the camera moves. For that geometry velocity is not
// something you must render - it is derivable from depth plus the two most
// recent camera matrices:
//
//     ndc      = (uv * 2 - 1, depth)
//     prevClip = prevViewProj * inverse(currViewProj) * ndc
//     velocity = uv - prevUv
//
// Depth the Vulkan layer can see. The matrices only X-Plane knows. So the
// plugin publishes them here every frame and the layer picks them up.
//
// The plugin collapses the two matrices into one `reproj` on the CPU. Not just
// tidiness: the two-matrix push block came to 144 bytes, over Vulkan's
// guaranteed 128-byte minimum, and would have failed on any device offering
// only the minimum. Combining them gives 80 bytes and drops a mat4 multiply
// from every pixel.
//
// ===================================================== WHAT CAMERA MOTION MISSES
//
// Camera-derived velocity is exactly right for static geometry and exactly
// wrong for everything else. The failure cases, and what this header carries to
// address each:
//
//   AI traffic / multiplayer  -> `objects[]`. Rigid bodies with known world
//        transforms, so they get real (if coarse) motion rather than masking.
//        Each entry is a bounding sphere plus its previous position; the shader
//        unprojects depth to world space and, inside a sphere, reprojects using
//        that object's motion instead of the static assumption.
//
//   Prop discs / rotor blades -> `reactive[]`. Deliberately NOT solved with
//        velocity. A spinning prop is unresolvable by TAA on principle: blade
//        tips move at hundreds of m/s, the disc is semi-transparent, and at 30-60
//        fps it aliases temporally no matter what (the wagon-wheel effect is not
//        a bug you can clamp away). Even perfect per-vertex rotational velocity
//        would not help, because alpha blending means the colour under a pixel
//        changes completely between frames. So the goal is not correctness, it
//        is invisibility: force the resolve toward the current frame and let the
//        disc alias the way it would with no AA at all. That reads as motion
//        blur on a spinning prop, which people accept instantly; ghosting reads
//        as a broken renderer, which they do not.
//
//   Sky and clouds -> handled in the shader, not here. See the note on
//        `reverseZ`: distant geometry must reproject by ROTATION, not be given
//        zero velocity, or the whole sky smears whenever you pan.
//
//   Volumetric clouds, water, SSR, windshield rain -> no velocity field can be
//        right for these. They are the job of neighbourhood clamping and the
//        variance-driven reactive term in the resolve, plus choosing an
//        insertion point before the windshield overlays where possible.
//
// ============================================================== THE JITTER RULE
//
// The matrices in this struct are ALWAYS UNJITTERED, and that is load-bearing.
//
// Jitter is applied by the layer through vkCmdSetViewport, which for a
// perspective projection is exactly equivalent to offsetting the projection
// matrix, and is far easier to intercept than the uniform buffer that carries
// the matrix. Because X-Plane computes and reports its projection before that
// offset exists, what arrives here is naturally jitter-free.
//
// This matters because the classic TAA bug is computing velocity from jittered
// matrices: the jitter delta leaks into the motion vectors and everything
// shimmers slightly even when completely static, which is very easy to
// misdiagnose as a bad neighbourhood clamp and burn a day on. Here the velocity
// path cannot see jitter at all, so the bug is structurally impossible rather
// than merely avoided. Do not "helpfully" pre-apply jitter to these matrices.
//
// The resolve still needs to know the jitter (to unjitter the current sample
// when comparing against history), which is what `jitterX/jitterY` carry.

#ifndef TAA_SHARE_H
#define TAA_SHARE_H

#include <stdint.h>
#include <string.h>
#include <math.h>

#define TAA_MAGIC       0x4D414154u            // 'TAAM'
#define TAA_VERSION     6      // 6: bodyReproj for own-aircraft/cockpit geometry

// Which temporal backend to run. All of them consume the SAME inputs - velocity
// buffer, jitter sequence, insertion point - which is the whole reason the
// project builds those first and picks a backend second. Nothing upstream of
// this enum is specific to any vendor.
enum TaaUpscaler {
    TAA_UPSCALER_OFF   = 0,
    TAA_UPSCALER_TAA   = 1,   // ours; no SDK, native resolution only
    // FSR 2, not 3. FSR2 is the version with a first-class NATIVE VULKAN
    // backend in AMD's standalone repo, which is what makes it usable from a
    // Vulkan layer at all; the newer FidelityFX SDK is DX12-only. This slot is
    // named for what is actually integrated rather than for the newest number.
    TAA_UPSCALER_FSR2  = 2,   // AMD FidelityFX Super Resolution 2, any GPU
    TAA_UPSCALER_FSR4  = 3,   // AMD FSR 4 - RDNA 4 ONLY, see caps below
    TAA_UPSCALER_DLSS  = 4,   // NVIDIA DLSS Super Resolution, RTX only
    TAA_UPSCALER_COUNT = 5
};

// Why a backend cannot be used. Reported by the LAYER (which is the only half
// that can see the GPU) and read back by the plugin so the UI can grey an
// option out with a reason, instead of letting it be selected and fail
// somewhere deep in a vendor SDK.
enum TaaAvailability {
    TAA_AVAIL_UNKNOWN     = 0,   // layer has not reported yet
    TAA_AVAIL_OK          = 1,
    TAA_AVAIL_NO_GPU      = 2,   // hardware cannot run it (e.g. FSR4 off RDNA4)
    TAA_AVAIL_NO_LIBRARY  = 3,   // SDK runtime not installed - see DOWNLOADS.md
    TAA_AVAIL_NO_SUPPORT  = 4    // not built into this layer yet
};

// Quality presets, mapped onto each backend's own preset enum. The scale each
// implies is the set the vendors publish, so switching backend at a given
// quality level keeps the same internal render resolution and the comparison
// stays honest.
enum TaaQuality {
    TAA_QUALITY_NATIVE      = 0,   // 1.00x - no upscaling; the only TAA mode
    TAA_QUALITY_ULTRA       = 1,   // 1.30x
    TAA_QUALITY_QUALITY     = 2,   // 1.50x
    TAA_QUALITY_BALANCED    = 3,   // 1.70x
    TAA_QUALITY_PERFORMANCE = 4,   // 2.00x
    TAA_QUALITY_COUNT       = 5
};

// Where image-space motion estimation comes from.
//
// Both do the same job - estimate motion from the IMAGE rather than from
// geometry - and both therefore cover the case depth reprojection provably
// cannot: content moving independently of the surface carrying it. A scrolling
// ND map on a static panel, CDU text, rain on glass, screen-space reflections.
// For all of those the geometric velocity is correct and the content still
// moved.
//
// They differ in cost and in how invasive they are, and the two point in
// opposite directions:
//
//   NVIDIA  - runs on the Optical Flow Accelerator, dedicated silicon on
//             Turing and later (much faster on Ada). Effectively free: it does
//             not compete with the shaders for SM time. But it needs the layer
//             to add a device extension AND an optical-flow queue at
//             vkCreateDevice, because X-Plane requests neither. Appending to
//             what the app asked for is a standard layer technique, but it is
//             the most invasive thing in this project.
//
//   FFX     - AMD FidelityFX Optical Flow, the one FSR 3 frame interpolation
//             uses. Pure compute shaders, so it needs NO device changes at all
//             - just a pipeline we dispatch, exactly like the velocity pass.
//             Runs on any vendor. Costs real GPU time, since it shares the SMs
//             with rendering.
//
// So the faster option is the riskier one to land, which is why both exist here
// rather than only the "best" one for this machine.
enum TaaOpticalFlow {
    TAA_OF_OFF    = 0,
    TAA_OF_NVIDIA = 1,   // VK_NV_optical_flow, hardware OFA
    TAA_OF_FFX    = 2,   // FidelityFX Optical Flow, compute
    TAA_OF_COUNT  = 3
};

static inline const char *taaOpticalFlowName(int o)
{
    switch (o) {
        case TAA_OF_OFF:    return "Off";
        case TAA_OF_NVIDIA: return "NVIDIA (hardware)";
        case TAA_OF_FFX:    return "FidelityFX (compute)";
        default:            return "?";
    }
}

static inline const char *taaUpscalerName(int u)
{
    switch (u) {
        case TAA_UPSCALER_OFF:  return "Off";
        case TAA_UPSCALER_TAA:  return "TAA";
        case TAA_UPSCALER_FSR2: return "FSR 2";
        case TAA_UPSCALER_FSR4: return "FSR 4";
        case TAA_UPSCALER_DLSS: return "DLSS";
        default:                return "?";
    }
}

static inline const char *taaQualityName(int q)
{
    switch (q) {
        case TAA_QUALITY_NATIVE:      return "Native";
        case TAA_QUALITY_ULTRA:       return "Ultra Quality";
        case TAA_QUALITY_QUALITY:     return "Quality";
        case TAA_QUALITY_BALANCED:    return "Balanced";
        case TAA_QUALITY_PERFORMANCE: return "Performance";
        default:                      return "?";
    }
}

static inline float taaQualityScale(int q)
{
    switch (q) {
        case TAA_QUALITY_NATIVE:      return 1.00f;
        case TAA_QUALITY_ULTRA:       return 1.30f;
        case TAA_QUALITY_QUALITY:     return 1.50f;
        case TAA_QUALITY_BALANCED:    return 1.70f;
        case TAA_QUALITY_PERFORMANCE: return 2.00f;
        default:                      return 1.00f;
    }
}

static inline const char *taaAvailabilityText(int a)
{
    switch (a) {
        case TAA_AVAIL_OK:         return "";
        case TAA_AVAIL_NO_GPU:     return "unsupported GPU";
        case TAA_AVAIL_NO_LIBRARY: return "runtime not installed";
        case TAA_AVAIL_NO_SUPPORT: return "not implemented yet";
        default:                   return "detecting...";
    }
}

#define TAA_SHARE_NAME  "Local\\TAAImpl_Matrices"

#define TAA_MAX_OBJECTS   32
#define TAA_MAX_REACTIVE  16

// Why temporal history has to be thrown away this frame. Reprojection is only
// valid when the previous frame shows the same world from a nearby viewpoint;
// after a teleport or a view cut the "previous" pixel is unrelated and blending
// it produces a smear that can take a second to wash out.
enum TaaResetReason {
    TAA_RESET_NONE      = 0,
    TAA_RESET_STARTUP   = 1,   // fewer than two frames of history
    TAA_RESET_CAMJUMP   = 2,   // teleport / reposition / replay scrub
    TAA_RESET_VIEWTYPE  = 3,   // cockpit -> external etc.
    TAA_RESET_FOV       = 4,   // zoom changed
    TAA_RESET_VIEWPORT  = 5,   // window resized
    TAA_RESET_PAUSE     = 6,   // paused or unpaused
    TAA_RESET_TIMEJUMP  = 7    // sim time discontinuity
};

// A rigid moving body, as a world-space bounding sphere plus where it was last
// frame. Coarse by construction: we over-mask around the silhouette rather than
// tracking geometry. That trade is deliberate - over-masking a small screen
// region costs a little aliasing on the outline, whereas ghosting drags a
// visible smear across the sky behind every aircraft.
// POSITIONS ARE CAMERA-RELATIVE (world minus the current camera position), not
// world. The shader works in that space because the world-space reprojection
// matrix cannot be inverted accurately in float32 at 52 km from the local
// origin - it left a 10-18% residual. Object motion is unchanged by the shift,
// since current and previous are offset by the same amount.
struct TaaMovingObject {
    float    x, y, z;         // camera-relative position this frame
    float    radius;          // bounding sphere, metres

    float    px, py, pz;      // camera-relative position last frame
    float    speed;           // metres/second, for logging and gating

    // Identity, and why it is versioned:
    //
    // X-Plane reuses AI aircraft slots. When traffic despawns and a new
    // aircraft appears in the same index, a naive cache hands you the OLD
    // aircraft's previous position - producing a velocity vector pointing
    // across half the world, which drags a smear over the screen for a frame.
    // `generation` increments whenever a slot's occupant changes, so a stale
    // entry is detectable rather than silently wrong.
    uint32_t id;
    uint32_t generation;

    // 0 => no trustworthy previous transform for this object (it just spawned,
    // or the slot was reused). The shader must REJECT HISTORY in this region,
    // NOT assume zero velocity. Those are very different: zero velocity claims
    // "this pixel did not move", which is a confident wrong answer and blends
    // in whatever was behind the aircraft last frame.
    int32_t  hasPrev;
    int32_t  pad0;
};

// A screen-space region that should ignore history and take the current frame.
// Used for prop/rotor discs, which are stamped on the CPU from known geometry
// rather than detected on the GPU.
struct TaaReactiveEllipse {
    float cx, cy;             // centre, in uv (0..1)
    float rx, ry;             // radii, in uv
    float strength;           // 0..1; 1 = take the current frame entirely
    float feather;            // uv units to ramp over, so there is no hard seam
    float pad0, pad1;
};

// Matrices are float[16], column-major, column-vector convention (v' = M * v)
// - the layout X-Plane's datarefs use and the one GLSL's mat4 expects, so they
// go straight into a push block with no transpose.
struct TaaShare {
    uint32_t magic;
    uint32_t version;
    uint32_t structSize;      // reader bails cleanly if this disagrees
    uint32_t pad0;

    uint64_t frame;           // increments once per rendered frame
    double   simTime;

    // Raw, exactly as X-Plane reports them - and UNJITTERED. See the jitter
    // rule above before touching these.
    float proj[16],     modelview[16],     world[16];
    float prevProj[16], prevModelview[16], prevWorld[16];

    // Precomputed: clip(now) -> clip(prev) in one step. The matrix the velocity
    // shader takes as a push constant.
    float   reproj[16];
    int32_t reprojValid;      // 0 if currViewProj was singular

    // Needed separately from `reproj` for the moving-object path, which has to
    // go position -> prev clip after applying the object's own motion, and so
    // cannot use the collapsed form.
    //
    // prevViewProj and invCurrViewProj are CAMERA-RELATIVE: they map to and
    // from (world - current camera position), NOT world. This is not a
    // convenience - inverting the world-space form in float32 at 52 km from the
    // local origin left a 10-18% residual, which is visibly wrong motion. See
    // the derivation in plugin.cpp.
    //
    // currViewProj is the WORLD-space form. It exists only for CPU-side
    // projection (reactive disc placement, semantic self-checks) and must never
    // be inverted.
    float   currViewProj[16];      // world -> clip        (CPU use only)
    float   prevViewProj[16];      // camera-rel -> prev clip
    float   invCurrViewProj[16];   // clip -> camera-rel

    // ---- reprojection for geometry rigidly attached to the OWN AIRCRAFT.
    //
    // `reproj` above is a world-frame reprojection: it answers "where was this
    // point last frame, given it did not move in the world". For the cockpit
    // that answer is wrong, and wrong by a lot. Cockpit surfaces sit around
    // 0.7 m from the eye, so a few metres of aircraft motion implies a parallax
    // of most of the screen - while their true screen motion is nearly zero,
    // because they travel with the camera.
    //
    // Measured: with the aircraft moving ~3 m per frame, the world-frame field
    // peaked at 451 px/frame, and the largest contributor was the panel.
    //
    // The fix is to reproject those pixels in the aircraft's BODY frame, where
    // a bolted-down panel has constant coordinates and therefore no motion.
    // Deriving it: a body point p has clip position P*V*B*p, so
    //
    //     bodyReproj = P_prev * (V_prev * B_prev) * (V_curr * B_curr)^-1 * P_curr^-1
    //
    // which is the same shape as `reproj` with the view matrix V replaced by
    // the body-to-eye matrix V*B. It falls out as identity when the camera is
    // fixed in the cockpit, which is the common case and a useful sanity check.
    //
    // This also stays correct when the head DOES move: the camera then moves
    // relative to the body, and the matrix reports exactly that motion.
    //
    // Body coordinates are metres from the aircraft datum, so the float32
    // conditioning problem that forced camera-relative space on the world path
    // does not arise here.
    float   bodyReproj[16];
    int32_t bodyReprojValid;   // 0 if the body-to-eye matrix was singular

    // Is the camera actually rigid with the airframe right now?
    //
    // Published because the cockpit path is only meaningful when it is, and
    // because it is measured rather than assumed: the plugin tracks the
    // camera's pose in body coordinates and reports how much it drifts. In a
    // fixed cockpit view this must be ~0 no matter how the aircraft manoeuvres,
    // so a large value means the body matrix convention is wrong - which is
    // exactly the class of error that has been expensive here.
    float   camBodyDrift;      // metres of camera movement in body frame

    // Distance from the camera to the aircraft datum. Small means a cockpit
    // view and a camera rigid with the airframe; large means an external view,
    // where the body path does not apply and bodyReprojValid goes to 0.
    float   camGap;

    // ---- scripted self-test. 0 when not running; see the phase enum in
    // plugin.cpp. selfTestExpectedPx is the motion the current phase PREDICTS
    // at the centre of the screen, so the layer can print measured against
    // expected instead of leaving a human to judge a picture.
    int32_t selfTestPhase;
    float   selfTestExpectedPx;

    int32_t viewportW, viewportH;
    float   fovDeg;

    // Depth convention. Read from sim/private/controls/hdr/use_reverse_z AND
    // independently derived from the projection matrix, then cross-checked:
    // getting this backwards silently inverts the shader's distance test and
    // the whole velocity field comes out wrong in a way that still looks
    // plausible - the worst kind of bug to chase from screenshots.
    int32_t reverseZ;
    int32_t reverseZFromMatrix;
    float   nearClip, farClip;
    int32_t infiniteFar;      // 1 => farClip is meaningless (infinite-far proj)

    // ---- jitter. In PIXELS, range about +/-0.5. Applied by the layer via
    // vkCmdSetViewport; published here so the resolve can unjitter. The
    // matrices above never include it - see the jitter rule.
    float   jitterX, jitterY;
    int32_t jitterIndex;      // position in the sequence
    int32_t jitterPhases;     // sequence length (8 native, 16+ upscaling)

    // ---- quality knobs the layer applies.
    //
    // Texture LOD bias, applied to every sampler the layer sees. This is the
    // single highest-impact knob for perceived sharpness: TAA and any upscaler
    // resolve from a lower effective sampling rate, and without a negative bias
    // runway markings, taxiway signage and distant terrain go mushy - which is
    // precisely the detail people notice first in a flight sim.
    //
    // -0.5 at native, log2(renderScale) when upscaling.
    //
    // CAUTION, specific to this install: a negative bias makes the sim sample
    // higher mips, which raises texture working set - and X-Plane's pager
    // already downscales under VRAM pressure. Pushing this too far can provoke
    // the very downscaling the other project exists to avoid. Measure both.
    float   lodBias;
    float   renderScale;      // 1.0 = native

    // ---- near-field (cockpit panel) reactive ramp.
    //
    // THE WORST CASE FOR THIS WHOLE APPROACH, and it is not an edge case - it
    // is the panel of every airliner.
    //
    // Glass cockpit displays are 2D content drawn onto static 3D surfaces. In a
    // cockpit view the panel barely moves relative to the camera, so
    // depth-reprojection gives it CORRECT geometric velocity: near zero. But
    // the content on that surface - a scrolling ND map, CDU text, a blinking
    // annunciator - moves independently of the surface carrying it. History
    // reprojects to exactly the right pixel and exactly the wrong content.
    //
    // The result is smeared map symbology and ghosting text, and no improvement
    // to the velocity field can fix it, because the velocity field is already
    // right. The fix has to be refusing history.
    //
    // The general mechanism for that is colour-difference rejection in the
    // resolve (large |current - reprojected history| with near-zero velocity is
    // the signature). This ramp is a deterministic backstop underneath it:
    // anything closer than `cockpitReactiveDist` gets its history weight pulled
    // down regardless of what the heuristic decides.
    //
    // The trade is explicit: panel edges get less temporal smoothing and alias
    // slightly more. Panel edges are low-contrast; display text is not. Losing
    // AA on the former to keep the latter readable is the right direction.
    float   cockpitReactiveDist;      // metres; 0 disables
    float   cockpitReactiveStrength;  // 0..1 at zero distance

    int32_t historyReset;     // 1 => discard history (see resetReason)
    int32_t resetReason;
    int32_t paused;
    int32_t viewType;

    float   camX, camY, camZ; // camera world position
    float   camDelta;         // metres travelled since the previous frame

    // ---- moving objects (AI traffic, multiplayer).
    int32_t objectCount;
    int32_t pad1;
    TaaMovingObject objects[TAA_MAX_OBJECTS];

    // ---- reactive regions (prop and rotor discs).
    int32_t reactiveCount;
    int32_t pad2;
    TaaReactiveEllipse reactive[TAA_MAX_REACTIVE];

    // ---- backend selection (plugin -> layer).
    //
    // Changed live from the in-sim window, from a Lua script, or from taa.ini.
    // The layer applies it on the next frame; it never has to be set at
    // startup, so backends can be compared back to back in one flight.
    int32_t upscaler;         // TaaUpscaler
    int32_t quality;          // TaaQuality
    float   sharpness;        // 0..1, backend's own sharpening pass
    int32_t pad3;

    // ---- capability report (layer -> plugin).
    //
    // The ONLY fields written by the layer. The plugin owns everything else in
    // this block, so this is a deliberate one-way exception: the layer is the
    // only half that can query the GPU, and the UI needs to know what is
    // actually usable before offering it.
    //
    // Without this the UI would happily offer FSR 4 on hardware that cannot run
    // it, and the failure would surface somewhere inside a vendor SDK as "it
    // didn't work" with nothing to act on.
    // ---- NVIDIA hardware optical flow (VK_NV_optical_flow).
    //
    // Worth having for a reason that is not obvious: it fixes the case
    // geometric reprojection provably cannot.
    //
    // Depth reprojection is EXACT for static geometry and structurally wrong
    // whenever content moves independently of the surface carrying it - a
    // scrolling ND map on a static panel, CDU text, rain on glass, screen-space
    // reflections. For all of those the geometric velocity is correct and the
    // content still moved.
    //
    // Optical flow estimates motion from the IMAGE, so it sees exactly that
    // motion. The two are complementary rather than competing: geometry is
    // exact where it applies and free of the aperture problem; optical flow
    // covers what geometry cannot express.
    //
    // The most valuable output is not the flow field itself but the
    // DISAGREEMENT between the two. Where they diverge, history is untrustworthy
    // - which is a principled reactive mask derived from measurement, rather
    // than the colour-difference heuristic standing in for one.
    //
    // Cost: the OFA is dedicated silicon (Turing+, much faster on Ada), so this
    // does not compete with the shaders for SM time.
    //
    // Requires the layer to add a device extension and an optical-flow queue at
    // vkCreateDevice, since X-Plane requests neither. Appending to what the app
    // asked for is a standard layer technique, but it is the most invasive
    // thing here - so it stays behind a flag and lands after the geometric path
    // is verified working on its own.
    int32_t opticalFlowWanted;    // TaaOpticalFlow - which source to use
    int32_t opticalFlowActive;    // TaaOpticalFlow - which is actually running

    int32_t  layerAttached;                    // 1 once the layer is running
    int32_t  availability[TAA_UPSCALER_COUNT]; // TaaAvailability per backend
    int32_t  ofAvailability[TAA_OF_COUNT];     // TaaAvailability per flow source
    char     gpuName[128];
    uint32_t gpuVendorId;
    int32_t  padCaps;

    // ---- VRAM, WRITTEN BY THE LAYER, READ BY THE PLUGIN.
    //
    // Every VRAM argument in this project has so far been conducted on guesses -
    // "Streamline is eating 2.5 GB", "our readbacks are the problem" - and each
    // time the actual numbers said something different. The layer already sees
    // the authoritative figures, because it intercepts
    // vkGetPhysicalDeviceMemoryProperties2 and reads the driver's own
    // VK_EXT_memory_budget heapBudget and heapUsage. There is no reason for that
    // to stay inside the layer's log.
    //
    // Device-local heap only - host-visible system memory is not what runs out.
    // Megabytes rather than bytes: a uint32 of bytes overflows at 4 GB, which is
    // half this card.
    uint32_t vramTotalMB;     // physical size of the device-local heap
    uint32_t vramBudgetMB;    // what the driver says this process may use
    uint32_t vramUsageMB;     // what this process currently has allocated

    // ---- FRAME RATE, MEASURED WHERE THE FRAMES ACTUALLY LEAVE.
    //
    // X-Plane has its own frame rate readout, and with frame generation on it
    // is no longer the number on the screen. The sim counts the frames IT
    // renders; the interpolation swapchain presents one more between every pair
    // of those, and knows nothing about the dataref. So both are published:
    // fpsPresented is what the sim produced, fpsDisplayed is what reached the
    // monitor, and the gap between them is the whole point of the feature.
    //
    // Measured in vkQueuePresentKHR rather than in a flight loop. A plugin
    // callback runs once per SIM frame by definition and could never see a
    // generated one; and it is skipped while the sim is paused or loading,
    // which is exactly when a frame-rate readout is most obviously wrong.
    float    fpsPresented;    // real frames per second, submitted by the sim
    float    fpsDisplayed;    // including generated frames; equals the above
                              // when frame generation is off
    uint64_t framesPresented; // running totals, so a reader can compute its own
    uint64_t framesDisplayed; // interval without trusting ours

    // ---- MEASURED render size, WRITTEN BY THE LAYER, READ BY THE PLUGIN.
    //
    // The plugin knows what IT asked for; it does not know what X-Plane did.
    // Those are different numbers whenever X-Plane's own FSR is on, and this
    // install runs it on: the sim renders 2953x1661 into a 3840x2160 window
    // while the plugin's quality setting says "Native, 1.00x".
    //
    // That gap is not cosmetic. FSR2's phase count is 8*(display/render)^2, so
    // the plugin computed 8*1*1 = 8 and fed FSR2 a sequence a third the length
    // of the one its accumulation assumes - the shortfall shows up as exactly
    // the shimmer-in-motion this project exists to remove, at the one setting
    // where nobody would think to look for a scaling bug.
    //
    // The layer already derives these from the G-buffer pass it latches onto
    // (see isSceneSized), so this is publishing a fact, not measuring a new one.
    // Zero means "not determined yet" and the plugin falls back to its own
    // quality setting.
    uint32_t measRenderW, measRenderH;
    uint32_t measDisplayW, measDisplayH;

    int32_t valid;            // 0 until two frames of history exist
};

// ------------------------------------------------------------ matrix helpers
//
// Column-major indexing: m[col * 4 + row].

// out = a * b, column-vector convention.
static inline void taaMul(float *out, const float *a, const float *b)
{
    float t[16];
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            t[c * 4 + r] = a[0 * 4 + r] * b[c * 4 + 0]
                         + a[1 * 4 + r] * b[c * 4 + 1]
                         + a[2 * 4 + r] * b[c * 4 + 2]
                         + a[3 * 4 + r] * b[c * 4 + 3];
    memcpy(out, t, sizeof(t));
}

// General 4x4 cofactor inverse. Deliberately not a fast affine shortcut: this
// runs on projection matrices, which are neither affine nor orthonormal, and it
// must stay correct for reverse-Z and infinite-far forms alike.
// Returns false (leaving out untouched) if the matrix is singular.
static inline bool taaInverse(float *out, const float *m)
{
    float inv[16];

    inv[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15]
             + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    inv[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15]
             - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    inv[8]  =  m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15]
             + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14]
             - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    inv[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15]
             - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    inv[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15]
             + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    inv[9]  = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15]
             - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    inv[13] =  m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14]
             + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    inv[2]  =  m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15]
             + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
    inv[6]  = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15]
             - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
    inv[10] =  m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15]
             + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14]
             - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
    inv[3]  = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11]
             - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
    inv[7]  =  m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11]
             + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11]
             - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
    inv[15] =  m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10]
             + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];

    float det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
    if (det > -1e-20f && det < 1e-20f) return false;

    det = 1.0f / det;
    for (int i = 0; i < 16; ++i) out[i] = inv[i] * det;
    return true;
}

// View-space distance to the plane a given clip-space z maps to. Recovers
// near/far without guessing at the projection's form: unproject the centre of
// the screen at that depth and read off the resulting z. Works for standard,
// reverse-Z, finite and infinite-far projections alike, which case analysis on
// proj[10]/proj[14] does not.
//
// Returns false when the plane is at infinity (w collapses to zero).
//
// Precision note, measured: recovering a FAR plane this way is ill-conditioned
// in float32 when far/near is large - a 5000 m far plane came back as 4993
// because the whole signal carrying it is ~2e-5 against 1.0. Near is accurate;
// far is a 3-digit number. Irrelevant for X-Plane's infinite-far projection.
static inline bool taaDepthPlaneDistance(const float *invProj, float clipZ, float *outDist)
{
    // (0, 0, clipZ, 1) through the inverse projection.
    float z = invProj[10] * clipZ + invProj[14];
    float w = invProj[11] * clipZ + invProj[15];
    if (w > -1e-9f && w < 1e-9f) return false;   // plane at infinity
    float vz = z / w;
    *outDist = vz < 0 ? -vz : vz;                 // magnitude; sign is handedness
    return true;
}

// Project a world point to uv (0..1). Returns false if it is behind the eye.
static inline bool taaProjectToUv(const float *viewProj, float x, float y, float z,
                                  float *outU, float *outV, float *outW)
{
    float cx = viewProj[0]*x + viewProj[4]*y + viewProj[8]*z  + viewProj[12];
    float cy = viewProj[1]*x + viewProj[5]*y + viewProj[9]*z  + viewProj[13];
    float cw = viewProj[3]*x + viewProj[7]*y + viewProj[11]*z + viewProj[15];
    if (outW) *outW = cw;
    if (cw <= 1e-6f) return false;
    *outU = (cx / cw) * 0.5f + 0.5f;
    *outV = (cy / cw) * 0.5f + 0.5f;
    return true;
}

// Halton radical inverse - the standard TAA jitter sequence.
//
// Halton(2,3) is used rather than a random or regular pattern because it is
// low-discrepancy: every prefix of the sequence covers the pixel area evenly,
// so the accumulated result is well distributed no matter where a history reset
// lands. A regular grid leaves structured gaps; random leaves clumps.
//
// Sequence length is a real trade, not a constant to copy: longer sequences
// converge to a cleaner image on static ground scenery, but during a fast pan
// on approach the older samples are stale and contribute smear. 8 at native,
// 16+ only when upscaling needs the extra coverage.
static inline float taaHalton(int index, int base)
{
    float f = 1.0f, r = 0.0f;
    int i = index;
    while (i > 0) {
        f /= (float)base;
        r += f * (float)(i % base);
        i /= base;
    }
    return r;
}

#endif // TAA_SHARE_H
