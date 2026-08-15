// TAAImplementation - X-Plane 12 half of a temporal anti-aliasing pipeline.
//
// This plugin does exactly one thing: publish the camera state the Vulkan layer
// needs, once per frame, into shared memory. It renders nothing, hooks nothing
// and changes no sim behaviour. All the GPU work lives in the layer.
//
// See src/share.h for why the layer cannot get this itself.
//
// SAFETY RULES. Reading datarefs at the wrong point in startup is unstable,
// so these are not optional:
//
//   * Never read dataref VALUES from XPluginEnable. At that point the sim has
//     not started and a good number of read accessors dereference state that
//     does not exist yet ("Sim is not yet started - Time is unset"). Finding
//     refs is fine; reading them is not.
//   * Only read values from a flight-loop callback, which cannot run until the
//     sim is actually up.
//   * Resolve datarefs lazily, inside that callback, never at load time.

#define XPLM200 1
#define XPLM210 1
#define XPLM300 1
#define XPLM301 1
#define XPLM400 1
#define IBM 1

#include "XPLMPlugin.h"
#include "XPLMDataAccess.h"
#include "XPLMProcessing.h"
#include "XPLMUtilities.h"
#include "XPLMDisplay.h"
#include "XPLMGraphics.h"
#include "XPLMMenus.h"
#include "XPLMCamera.h"
#include "XPLMPlanes.h"   // XPLMSetUsersAircraft / XPLMPlaceUserAtAirport

#include "share.h"
#include "control.h"

#include <windows.h>

#include <cstdio>
#include <cstring>
#include <cstdarg>
#include <string>
#include <vector>

#define TAA_PLUGIN_VERSION "0.2.0"

static std::string g_configPath;

static void xlog(const char *fmt, ...)
{
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    char line[1200];
    snprintf(line, sizeof(line), "TAAImpl: %s\n", buf);
    XPLMDebugString(line);
}

// ---------------------------------------------- dataref lookups, audited
//
// EVERY dataref in this plugin is fetched through here, and the misses are
// remembered.
//
// XPLMFindDataRef returns null for a name that does not exist and says nothing.
// Every call site then does `if (ref) read(...)` and carries on with whatever
// the variable was initialised to. That is indistinguishable from a successful
// read of that value, and it has already cost this project real time twice:
// layerAttached was written once and erased by a memset, and the SPIR-V
// coverage probe asked about the wrong attachment. The body-frame quaternion is
// the current suspect for exactly the same reason - if
// sim/flightmodel/position/q did not resolve, q stays {1,0,0,0}, R is the
// identity every frame, and the measured drift is |rel| * turn rate, which is
// precisely the number being measured.
//
// A miss is not necessarily a bug - several datarefs here are probes for things
// that legitimately may not exist. The point is that it should never be SILENT.
static std::vector<std::string> g_drefMissing;
static int g_drefTotal = 0, g_drefFound = 0;

static XPLMDataRef taaFind(const char *name)
{
    ++g_drefTotal;
    XPLMDataRef r = XPLMFindDataRef(name);
    if (r) ++g_drefFound;
    else if (g_drefMissing.size() < 64) g_drefMissing.push_back(name ? name : "(null)");
    return r;
}

static void taaLogDrefAudit()
{
    xlog("datarefs: %d of %d resolved, %d missing",
         g_drefFound, g_drefTotal, g_drefTotal - g_drefFound);
    for (size_t i = 0; i < g_drefMissing.size(); ++i)
        xlog("datarefs:   MISSING  %s", g_drefMissing[i].c_str());
    if (g_drefMissing.size() >= 64)
        xlog("datarefs:   (list truncated at 64)");
}

// ------------------------------------------------------------ shared memory

static HANDLE    g_shareHandle = nullptr;
static TaaShare *g_share       = nullptr;

// projection_matrix is whatever the LAST render pass set, which from a flight
// loop is the 2D UI pass - it comes back as an orthographic pixel-to-NDC matrix
// (row0[0] = 2/2560 = 0.0008) and is useless here. projection_matrix_3d is the
// one we want. This cost real debugging time once; do not "simplify" it back.
static XPLMDataRef g_drProj = nullptr, g_drProj3d = nullptr;
static XPLMDataRef g_drMv = nullptr, g_drWorld = nullptr;
static XPLMDataRef g_drVpW = nullptr, g_drVpH = nullptr, g_drFov = nullptr;
static XPLMDataRef g_drRevZ = nullptr, g_drViewType = nullptr, g_drPaused = nullptr;
static XPLMDataRef g_drViewExternal = nullptr;
static XPLMDataRef g_drSimTime = nullptr;
static XPLMDataRef g_drFlightTime = nullptr;
static bool g_refsResolved  = false;
static bool g_loggedFirst   = false;

// Frames actually spent in flight, as opposed to s->frame which the layer uses
// as a seqlock. Logging milestones off this means "frame 900" is 900 frames of
// flying, not 900 frames of sitting in a menu.
static uint64_t g_flightFrames = 0;
static bool     g_wasInFlight  = false;

// ---------------------------------------------------------------- self-test
//
// The plugin flies the camera itself, on a scripted path, so that validating
// the velocity field does not require a human to fly.
//
// This exists because the loop of "rebuild, launch, fly for a minute, read the
// log, find one thing" was answering roughly one question per flight, and most
// of those questions had exact answers that nobody needed to be airborne to
// produce. Two of the phases below have ground truth that can be stated in
// advance rather than eyeballed:
//
//   HOLD  - camera perfectly still. Every static pixel must read exactly zero.
//           Anything else is jitter leaking into the matrices, a stale previous
//           frame, or a reprojection that is not the identity when it must be.
//   YAW   - pure rotation about the eye, no translation. Every pixel moves by
//           the SAME amount regardless of depth, sky included: f * dYaw. That
//           is a single number to check against, and it is depth-independent,
//           so it isolates the matrices from the depth buffer completely.
//
// TRANSLATE and HEADMOVE have no single expected value but bracket the two
// cases that differ: world parallax, and motion of the camera relative to the
// airframe. The sim does not need to be flying - parked on a runway gives
// terrain, buildings and cockpit all in one frame.
// YAWL and PITCH exist because every check above is on MAGNITUDE, and a
// magnitude cannot see a sign error or a swapped axis. Yawing only ever to the
// right would pass a field whose x was negated, and never touching pitch would
// pass one whose y was flipped - which is a live risk here, since Vulkan's
// clip-space y points the opposite way to OpenGL's.
//
// Rather than assert a sign from first principles and risk baking in my own
// convention error, the checks are relative: left yaw must produce the
// OPPOSITE sign to right yaw at a similar magnitude, and yaw must move pixels
// horizontally while pitch moves them vertically. Both hold regardless of which
// convention is correct, and both fail loudly if the axes are crossed.
enum {
    TAA_ST_OFF = 0,
    TAA_ST_SETTLE,
    TAA_ST_HOLD,
    TAA_ST_YAW,
    TAA_ST_YAWL,
    TAA_ST_PITCH,
    TAA_ST_TRANSLATE,
    TAA_ST_HEADMOVE,
    // The only phase that moves the AEROPLANE. Everything above drives the
    // camera while the aircraft sits parked with the brake on, so the whole
    // suite verified rotation and a camera slide and nothing else.
    TAA_ST_FLY,
    // ---- EXTERNAL VIEW, AIRCRAFT PARKED.
    //
    // Isolates one variable. A capture showed 0.006 px with the aeroplane small
    // in frame and 300.020 px with it filling the frame, which says the
    // aircraft's own surfaces reproject wrongly - but "external camera" and
    // "aircraft is a moving object" were confounded in every frame that showed
    // it. Parked and external separates them: if the residual is clean here,
    // the external camera path is fine and the fault is the aircraft moving; if
    // it is broken here, the camera path itself is wrong.
    TAA_ST_EXTERNAL,
    // ---- SCRIPTED MOTION IN THE EXTERNAL VIEW.
    //
    // EXTERNAL above hands the camera to X-Plane's circle view, which then
    // sits perfectly still - so the one view where the wing-ghost artifact
    // forms was only ever measured motionless, where any reprojection error
    // reads as zero. These two phases drive the missing half deterministically:
    // ORBIT circles the parked aircraft at a fixed radius - the exact motion a
    // user's hand makes when the ghost appears - and HOLD then freezes at the
    // final orbit pose, which is the parked-residue check: whatever ORBIT
    // smeared into history either clears here or is convicted of parking.
    //
    // The camera is RE-ACQUIRED for these phases: the EXTERNAL case surrenders
    // it by returning 0, which uninstalls the callback, so the phase change
    // into ORBIT calls XPLMControlCamera again. The view TYPE stays circle
    // (1026) throughout - taking the camera positions it but does not change
    // the view - which is precisely the point: external view, our motion.
    TAA_ST_EXT_ORBIT,
    TAA_ST_EXT_HOLD,
    TAA_ST_DONE
};

static int      g_stPhase      = TAA_ST_OFF;
static bool     g_stPhaseWasExternal = false;
static int      g_stFrame      = 0;
static bool     g_stActive     = false;   // we hold the camera
static bool     g_stRequested  = false;
static bool     g_stHaveBase   = false;
static float    g_stLastHdg    = 0.0f;
static bool     g_stLastHdgValid = false;
static float    g_stAppliedYaw = 0.0f;   // degrees actually applied last render
static float    g_stLastPitch  = 0.0f;
static bool     g_stLastPitchValid = false;
static float    g_stAppliedPitch   = 0.0f;
static int      g_stWatchdog       = 0;
static int      g_stShakeSaved[2] = {0, 0};
static float    g_stBrakeSaved = 0.0f;
static float    g_stBaseX = 0, g_stBaseY = 0, g_stBaseZ = 0;
static float    g_stBaseHdg = 0, g_stBasePitch = 0;

// Degrees of yaw per frame during TAA_ST_YAW, and metres per frame during
// TAA_ST_TRANSLATE. Small enough to stay well inside one screen of motion so
// nothing clips against the frame edge and skews the statistics.
// TAA_ST_RATE scales the commanded rotation.
//
// A scaling test is the one thing that says whether the surviving uniform
// fields are a ROTATION at all. Everything else about them is consistent with a
// rigid rotation of several steps - but the flight loop and the renderer are
// 1:1 (shareframe advances 120 per 120 presents), the matrix describes one
// step, and the shader provably builds the field from that matrix. If the
// uniform fields scale with this, they are rotations and something is
// multiplying them; if they do not, they are not our rotation at all.
static float kStYawPerFrame = getenv("TAA_ST_RATE")
                            ? (float)atof(getenv("TAA_ST_RATE")) : 0.25f;
static const float kStMetresPerFrame = 0.35f;
static const int   kStPhaseFrames   = 150;

// Resets per reason. The headline number for whether the detector is sane: in
// normal flight this should be a handful of view changes, not hundreds. The
// first run logged 205 camera jumps in one short flight, every one of them a
// false positive from a bad camera-position extraction.
static uint64_t g_resetCounts[8] = {0};

// ---- moving objects (AI traffic / multiplayer)
//
// Slot 0 is reserved for the user's own aircraft, which matters in any external
// view. Slots 1..N are AI/multiplayer traffic.
#define TAA_TRAFFIC_SLOTS 19

static XPLMDataRef g_drTrafficX[TAA_TRAFFIC_SLOTS] = {0};
static XPLMDataRef g_drTrafficY[TAA_TRAFFIC_SLOTS] = {0};
static XPLMDataRef g_drTrafficZ[TAA_TRAFFIC_SLOTS] = {0};
static XPLMDataRef g_drOwnX = nullptr, g_drOwnY = nullptr, g_drOwnZ = nullptr;
static XPLMDataRef g_drOwnQ = nullptr;

// Pilot eye position in AIRCRAFT BODY COORDINATES, metres. This is the ground
// truth the body frame has been missing.
//
// The handedness test below proves the body transform is RIGID - the camera
// holds still in body coordinates - and rigidity was assumed to imply
// correctness. It does not. A rotation built with the wrong axis convention is
// still orthonormal, so the camera sits at a CONSTANT WRONG position, drift
// measures zero, and the test reports "body frame trusted" while every cockpit
// motion vector is wrong. That is how the cockpit kept shaking with a body
// frame that had passed its own check.
//
// acf_pe* breaks the tie because it is an independent statement of the same
// quantity: where the eye is on the airframe. If the computed camera position
// disagrees with it, the rotation is wrong no matter how steady it is.
static XPLMDataRef g_drPeX = nullptr, g_drPeY = nullptr, g_drPeZ = nullptr;

// ------------------------------------------------- texture budget patch
//
// X-Plane caps its own texture memory far below what the card offers, and then
// degrades textures to "Low" while nothing is actually failing.
//
// MEASURED, not inferred. The driver reports a 7.02 GB budget on a 7.77 GB
// heap; total usage peaks near 5 GB; X-Plane allocates 2.79 GB, announces that
// the card "doesn't have enough VRAM", and drops texture resolution. The layer
// hooks vkAllocateMemory and recorded ZERO failed allocations throughout, so
// there is no wall being avoided - the limit is entirely self-imposed.
//
// From the disassembly of the budget accessor at RVA 0x104F000:
//
//     mov  rcx,[rbx+0xc0]
//     call [rax+0x8]              ; bytes currently allocated
//     mov  rdx,rax
//     add  rdx,[rbx+0x108]        ; plus a second reserve
//     mov  rax,[rbx+0xe8]         ; TOTAL BUDGET  <-- the 7 bytes we replace
//     sub  rax,rdx                ; available = total - (allocated + reserve)
//
// Replacing the load of the stored total with an immediate raises the ceiling
// and leaves the accounting - and therefore the pager, and therefore eviction -
// completely intact. That is the important difference from -gfx-no-pager, which
// removed the eviction and produced an overrun.
//
//   48 8B 83 E8 00 00 00   mov rax,[rbx+0xe8]     (7 bytes)
//   B8 xx xx xx xx 90 90   mov eax,imm32 ; nop;nop
//
// mov eax zero-extends into rax, so the ceiling is 4 GB. That is short of the
// card but comfortably above the ~3 GB X-Plane chose, and staying under 4 GB
// leaves headroom for buffers, our own allocations and the desktop rather than
// pushing the card to its limit.
//
// THE BYTE CHECK IS THE SAFETY. If X-Plane updates and those seven bytes are
// anything else, the patch is refused and logged. Writing an immediate into the
// middle of whatever replaced it would corrupt code rather than fail.
static const unsigned char kBudgetOrig[7] =
    { 0x48, 0x8B, 0x83, 0xE8, 0x00, 0x00, 0x00 };
static const size_t kBudgetRva = 0x104F065;

// ---------------------------------------------- texture pager art controls
//
// The names come from the string table around TEX_paging.cpp in the binary, not
// from documentation - X-Plane's public DataRefs.txt lists no sim/private
// entries at all. XPVRAMUnlock already holds two of these, which is how they
// were found.
//
// Reading them first, before changing anything. The defaults are the thing
// worth knowing: a pager that reserves too much will say so in its fudge
// factor, and guessing at replacements without seeing the originals is how you
// end up unable to explain a change later.
static const char *kPagerControls[] = {
    // How far past budget the pager will run before it starts cutting. This is
    // the most direct expression of "too aggressive with 4 GB free".
    "sim/private/controls/tex/paging/max_overdrive",

    // Applied to the pager's ESTIMATE of what textures will cost. If it
    // over-estimates, it cuts while memory is still free - which is exactly the
    // symptom.
    "sim/private/controls/tex/paging/size_fudge_factor",

    "sim/private/controls/tex/paging/max_distance",
    "sim/private/controls/tex/paging/downscale_cooldown",
    "sim/private/controls/tex/ortho_boost_factor",

    // Seen as bare names in the same table; the full paths are a guess, so a
    // missing ref here is information rather than a failure.
    // ---- X-PLANE'S OWN SUB-NATIVE RENDERING.
    //
    // The sim already knows how to render its 3D scene smaller than the window.
    // The binary carries the strings "fsr/enable", "fsr/quality", "fsr/bypass"
    // and "fsr/clamp_size" next to a settings caption reading "Rendering
    // Resolution (FSR Supersampling)", and symbols for OGL_fsr_init,
    // OGL_fsr_build_upscale_graph and OGL_fsr_build_sharpen_graph - so this is a
    // complete FSR 1.0 path, not a leftover.
    //
    // Why it matters more than anything in the pager. Every category the ledger
    // says we cannot page - render targets, depth buffers, and the upscaler's
    // own storage images - is sized by the RENDER RESOLUTION. Nothing in a
    // Vulkan layer can shrink those, because their size is X-Plane's decision
    // and the passes writing them carry matching viewports and shader
    // constants. Asking X-Plane to render smaller is the only lever that moves
    // them, and it moves all of them at once.
    //
    // `bypass` is the interesting one. FSR 1.0 is spatial: render low, then
    // upscale and sharpen. If bypass skips the upscale while keeping the
    // reduced render, the scene arrives small and OUR resolve can do the
    // upscale with motion vectors and history - which is a temporal
    // reconstruction rather than a spatial guess, and the whole point of this
    // project. If it instead disables the feature outright, that reading is
    // wrong, which is why these are probed rather than assumed.
    //
    // `clamp_size` probably bounds the smallest render the sim will accept.
    // Worth knowing before asking for 0.75 and quietly getting something else.
    "sim/private/controls/fsr/enable",
    "sim/private/controls/fsr/quality",
    "sim/private/controls/fsr/bypass",
    "sim/private/controls/fsr/clamp_size",
};

// ------------------------------------------------ everything else that costs
//
// Found by pulling the whole art-control namespace out of the binary rather
// than by guessing names: X-Plane stores them as bare suffixes like
// "tex/paging/max_overdrive" and prepends sim/private/controls/ at
// registration, so the entire set is greppable.
//
// Grouped by which line of the VRAM ledger it moves, because that is the only
// thing that decides whether a control is worth touching. Probed rather than
// used: a control that is absent, read-only, or already at its cheapest setting
// is not an opportunity, and the three are indistinguishable without looking.
static const char *kVramControls[] = {
    // ---- GEOMETRY BUFFERS. The largest category in the ledger at 2423 MB and
    // the one our layer cannot page at all: mesh data has no mip chain.
    //
    // X-Plane HAS A BUFFER PAGER OF ITS OWN, which is the find that matters.
    // The counters name tiers - device, host, agp, hot, warm, cold, missed -
    // each with bytes and count, which is a residency system for exactly the
    // memory we had written off. `missed` is the interesting one: buffers that
    // were not resident when a draw needed them, which is what a stall looks
    // like from the inside, and a candidate for the frame drops on camera
    // movement that I have twice guessed wrong about.
    "sim/private/controls/gfx/managed_buffers/max_movement_bytes",
    "sim/private/controls/gfx/managed_buffers/max_movement_count",
    // Geometry the sim chooses to build. Fewer or coarser objects means fewer
    // vertex buffers, which is the only honest way to shrink that 2423 MB.
    "sim/private/controls/forest/density",
    "sim/private/controls/forest/lod",
    "sim/private/controls/forest/lod_multiplier",
    "sim/private/controls/forest/quad_count",
    "sim/private/controls/forest/quality_level",
    "sim/private/controls/cars/lod_min",
    "sim/private/controls/cube/lod_bias_objects",
    "sim/private/controls/cube/lod_bias_forest",
    "sim/private/controls/cube/max_dsf_dist",
    "sim/private/controls/instance/max_lod_ratio",
    "sim/private/controls/instance/tiny_obj_cutoff",
    "sim/private/controls/instance/merge_vbos",
    "sim/private/controls/ag/tile_lod_bias",

    // ---- TEXTURES. Ours already pages these, but these decide how many exist
    // and how big they are before our pager ever sees one.
    "sim/private/controls/reno/tex_res",
    "sim/private/controls/reno/comp_texes",
    "sim/private/controls/tex/preload_dist",
    "sim/private/controls/tex/distance_pad",
    "sim/private/controls/dsf/base_terrain_in_vram",

    // ---- RENDER TARGETS AND DEPTH BUFFERS, 495 + 173 MB and unpageable by any
    // mechanism we have. Shadow and reflection targets are sized by these
    // directly, so they are the only lever that exists for that memory short of
    // rendering the whole scene smaller.
    "sim/private/controls/fbo/shadow_cam_size",
    "sim/private/controls/clouds/shadow_size",
    "sim/private/controls/cloud/shadow_cascades",
    "sim/private/controls/cubemap/interior_proj_size",
    "sim/private/controls/perf/disable_reflection_cam",
    "sim/private/controls/perf/disable_shadow_prep",
};

// Probe one list and report what is really there.
//
// Absent, read-only, and already-at-its-cheapest look identical from outside
// and mean completely different things, so all three are printed rather than
// summarised. A name lifted from the binary's string table is a candidate, not
// a control - some are compiled out, some are registered under a different
// prefix, and some exist only in the debug build.
static int dumpControlList(const char *title, const char **names, size_t count)
{
    xlog("---- %s ----", title);
    int found = 0;
    for (size_t i = 0; i < count; ++i) {
        XPLMDataRef r = taaFind(names[i]);
        if (!r) { xlog("  (absent) %s", names[i]); continue; }
        ++found;

        XPLMDataTypeID t = XPLMGetDataRefTypes(r);
        int writable = XPLMCanWriteDataRef(r);
        if (t & xplmType_Float)
            xlog("  %s = %g  (float, %s)", names[i],
                 XPLMGetDataf(r), writable ? "writable" : "READ-ONLY");
        else if (t & xplmType_Double)
            xlog("  %s = %g  (double, %s)", names[i],
                 XPLMGetDatad(r), writable ? "writable" : "READ-ONLY");
        else if (t & xplmType_Int)
            xlog("  %s = %d  (int, %s)", names[i],
                 XPLMGetDatai(r), writable ? "writable" : "READ-ONLY");
        else
            xlog("  %s exists, type mask 0x%x, %s", names[i],
                 (unsigned)t, writable ? "writable" : "READ-ONLY");
    }
    xlog("---- %d of %zu present ----", found, count);
    return found;
}


// ------------------------------------------- give the texture budget back
//
// THE STUTTER IS EVICTION CHURN, and this is where the budget goes.
//
// Measured: 174 "compressed texture used as material source" warnings from ~25
// unique cockpit normal maps - each one reprocessed SEVEN OR EIGHT TIMES in a
// single session. They are not loaded once; they are loaded, evicted, and
// loaded again, and every reload costs a decompress plus CPU mipmap generation
// at 18.9 ms a call. The profiler shows the tex group averaging 11.85 ms a
// frame and spiking to 114.85 ms.
//
// They thrash because they do not fit, and they do not fit because X-Plane
// withholds 2.75 GB before textures are counted at all. Reconstructed from the
// disassembly of the pager's budget function:
//
//   available = (free - reserve + usage) * 0.975
//   free      = none_threshold - allocated - managed_buffer_budget
//
//   none_threshold : device budget minus a table of three reserves, chosen by
//                    card size. On an 8 GB card: 512 + 256 + 512 = 1280 MB.
//   reserve        : a flat 4 x 256 MB = 1024 MB, a literal in the code.
//
// Both are cut here. Not to zero: the reserves exist so the driver has headroom
// for render targets, command buffers and allocation spikes, and a card that is
// genuinely full needs that. Halving is defensible; removing it trades texture
// quality for allocation failures, which is a worse failure than a stutter.
static bool g_budgetPatched = false;

static void patchTextureBudgetReserve()
{
    if (g_budgetPatched) return;
    g_budgetPatched = true;
    if (!getenv("TAA_BUDGET_RESERVE")) {
        xlog("budget reserve: not armed - set TAA_BUDGET_RESERVE=<MB> to cut "
             "the flat 1024 MB reserve the texture pager holds back");
        return;
    }
    int wantMB = atoi(getenv("TAA_BUDGET_RESERVE"));
    if (wantMB < 256 || wantMB > 1024) {
        xlog("budget reserve: %d MB is outside 256..1024 - refusing. Below 256 "
             "leaves the driver no headroom for render targets and command "
             "buffers; above 1024 is worse than the default.", wantMB);
        return;
    }

    //   mov dword ptr [rsp+0x40], 4.0f
    // The 4 is multiplied by 256 MB. Verified UNIQUE in the image: this exact
    // eight-byte sequence occurs once in 200 MB, so a search cannot land on an
    // unrelated store of the same constant.
    static const unsigned char kOrig[8] =
        { 0xC7, 0x44, 0x24, 0x40, 0x00, 0x00, 0x80, 0x40 };

    HMODULE base = GetModuleHandleA(nullptr);
    if (!base) return;
    unsigned char *b = (unsigned char*)base;
    IMAGE_DOS_HEADER   *dos = (IMAGE_DOS_HEADER*)b;
    IMAGE_NT_HEADERS64 *nt  = (IMAGE_NT_HEADERS64*)(b + dos->e_lfanew);
    IMAGE_SECTION_HEADER *sec = IMAGE_FIRST_SECTION(nt);

    unsigned char *found = nullptr;
    int hits = 0;
    for (int i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        if (memcmp(sec[i].Name, ".text", 5) != 0) continue;
        unsigned char *p = b + sec[i].VirtualAddress;
        size_t n = sec[i].Misc.VirtualSize;
        for (size_t k = 0; k + sizeof(kOrig) <= n; ++k)
            if (memcmp(p + k, kOrig, sizeof(kOrig)) == 0) {
                if (!found) found = p + k;
                if (++hits > 1) break;
            }
    }
    if (!found || hits != 1) {
        xlog("budget reserve: the constant %s - refusing to patch",
             !found ? "was not found" : "is no longer unique");
        return;
    }

    float want = (float)wantMB / 256.0f;
    DWORD old = 0;
    if (!VirtualProtect(found + 4, 4, PAGE_EXECUTE_READWRITE, &old)) return;
    memcpy(found + 4, &want, 4);
    VirtualProtect(found + 4, 4, old, &old);
    FlushInstructionCache(GetCurrentProcess(), found, 8);

    xlog("budget reserve: the pager's flat reserve is now %d MB instead of 1024 "
         "- that is %d MB more texture budget, in memory only, X-Plane.exe on "
         "disk is untouched", wantMB, 1024 - wantMB);
    xlog("budget reserve: this is aimed at the eviction churn - 25 cockpit "
         "normal maps were each being decompressed and re-mipmapped 7-8 times "
         "because they did not fit. Watch the 'compressed texture used as "
         "material source' count in this log: fewer repeats means it worked.");
}

// ------------------------------------------------------------ VRAM trim
//
// Modest reductions to the controls that cannot make the CURRENT problems
// worse - see the reasoning per entry. Enabled with TAA_VRAM_TRIM=1.
//
// RELATIVE, NOT ABSOLUTE. Each value is a factor applied to whatever the
// control was already set to, captured the first time it resolves. Writing
// absolute numbers would mean knowing the units and the range of a dozen
// undocumented controls, and being wrong about one of them looks identical to
// being right about it until something renders strangely an hour later. A
// factor is correct whether shadow_cam_size turns out to be 2048 or 4096.
//
// The originals are logged next to the new values, so the change is a
// measurement rather than an assertion.
struct TrimControl {
    const char *path;
    float       factor;      // multiply the ORIGINAL by this
    bool        wholeNumber; // round the result - see below
    const char *why;
    XPLMDataRef ref;
    float       orig;
    float       want;
    bool        ready;
};

// COUNTS MUST STAY WHOLE, and the first version of this got it wrong.
//
// A factor is the right way to scale a size or a distance without knowing its
// units. It is the WRONG way to scale a count: forest/quad_count defaults to 2,
// and x0.75 asked for 1.5 quads per tree, which is not a thing. Whether that
// truncates, rounds or does something worse is X-Plane's business, and none of
// the three is what was intended.
//
// So controls whose value is a number OF something are marked and rounded, with
// a floor of 1 - scaling a count to zero disables the feature rather than
// reducing it, which is a different decision than the one being made here.

static TrimControl g_trim[] = {
    // ---- render targets and depth. Quadratic in the dimension, and shadow
    // maps are the one thing on the list that gets worse invisibly - you see
    // softer shadow edges long before you see anything wrong.
    // { "sim/private/controls/fbo/shadow_cam_size", 0.5f, false,
    // "shadow map dimension - quarter the memory, multiplied by cascade count",
    // nullptr, 0, 0, false },
    // { "sim/private/controls/clouds/shadow_size", 0.5f, false,
    // "cloud shadow map dimension", nullptr, 0, 0, false },
    // { "sim/private/controls/cubemap/interior_proj_size", 0.5f, false,
    // "interior cubemap face - six faces, so it multiplies", nullptr, 0, 0, false },

    // ---- geometry. Everything here is a distance, a density or a count, so
    // smaller unambiguously means less - which is why these are safe to scale
    // blind while the lod_bias controls below are not.
    { "sim/private/controls/cube/max_dsf_dist", 0.5f, false,
      "how far scenery is built for the reflection cubemap", nullptr, 0, 0, false },
    { "sim/private/controls/forest/density", 0.75f, false,
    "trees per unit area - the largest single geometry lever", nullptr, 0, 0, false },
    { "sim/private/controls/forest/lod_multiplier", 0.75f, false,
    "forest detail distance", nullptr, 0, 0, false },
    // { "sim/private/controls/forest/quad_count", 0.75f, true,
    // "quads per tree - multiplies with density", nullptr, 0, 0, false },

    // Bigger cutoff culls MORE, so this one goes up. The only entry whose
    // factor is above 1, and worth the comment for that reason alone.
    // ---- LOAD PACING. Do not do the whole aircraft in one frame.
    //
    // runloop/time_per_frame_usec is a TIME BUDGET: how many microseconds
    // X-Plane will spend on background work inside a single frame. Whatever it
    // is set to is, by definition, how long one frame can be stalled by
    // loading - so it is the direct control over stutter DEPTH.
    //
    // Quartered. The work does not go away, it is spread over four times as
    // many frames: the aircraft takes longer to finish loading and no single
    // frame pays as much for it. That is the trade being made deliberately -
    // a longer, smoother load instead of a shorter, spikier one.
    //
    // This matters because the churn is only half fixed. Cutting the pager's
    // flat reserve took the reload count from 174 to 101 across 25 files, so
    // each is still being decompressed and re-mipmapped about four times at
    // 18.9 ms a call. Pacing bounds what any one frame pays for that even
    // while the reloads continue.
    { "sim/private/controls/runloop/time_per_frame_usec", 0.25f, false,
      "microseconds of background work allowed per frame - the stutter depth",
      nullptr, 0, 0, false },
    { "sim/private/controls/runloop/tasks_per_frame", 0.5f, true,
      "background tasks per frame", nullptr, 0, 0, false },

    { "sim/private/controls/instance/tiny_obj_cutoff", 1.5f, false,
      "screen-size below which an object is skipped - RAISED to cull more",
      nullptr, 0, 0, false },

    // ---- THE BUFFER PAGER'S PER-FRAME MIGRATION LIMIT.
    //
    // These two are the only members of gfx/managed_buffers that are actually
    // registered as datarefs - the tier counters named in the binary
    // (device/host/agp/hot/warm/cold/missed) do not exist at runtime, so the
    // residency state cannot be read at all. What CAN be changed is how much
    // may move per frame, and the defaults are small: 40 buffers, 100 MB.
    //
    // The hypothesis this tests: swinging the camera makes far more than 40
    // buffers newly visible, they can only become resident 40 per frame, and
    // every draw waiting on the rest stalls. That is a much better fit for a
    // frame rate that collapses on camera movement and recovers when still
    // than anything else measured so far - and unlike the previous three
    // theories, it is one knob away from being tested.
    //
    // RAISED rather than lowered, which makes these the only entries here that
    // spend memory bandwidth instead of saving it. The limit exists to stop
    // migration monopolising a frame; quadrupling it trades a little of that
    // for not stalling. If frame times get WORSE, the limit was doing its job
    // and the answer is somewhere else entirely.
};

// Watched but NOT written, because the direction is genuinely unknown.
//
// A "bias" may be added or multiplied, and may point either way - a positive
// lod_bias could mean more detail or less depending on whose convention it
// follows. Guessing costs nothing to apply and a lot to diagnose: the result
// would look like a quality change either way, and attributing it correctly
// means already knowing the answer. So these are logged at their defaults and
// left alone until there is a value to compare against.
static const char *kTrimObserveOnly[] = {
    "sim/private/controls/cube/lod_bias_objects",
    "sim/private/controls/cube/lod_bias_forest",
    "sim/private/controls/ag/tile_lod_bias",
    "sim/private/controls/forest/lod",
    "sim/private/controls/forest/quality_level",
    "sim/private/controls/cars/lod_min",
    "sim/private/controls/reno/comp_texes",
};

static bool g_trimOn = false;

static void applyVramTrim()
{
    if (!g_trimOn) return;

    for (size_t i = 0; i < sizeof(g_trim)/sizeof(g_trim[0]); ++i) {
        TrimControl &t = g_trim[i];
        if (!t.ready) {
            t.ref = taaFind(t.path);
            if (!t.ref) { t.ready = true; xlog("trim: %s absent", t.path); continue; }
            if (!XPLMCanWriteDataRef(t.ref)) {
                t.ready = true;
                xlog("trim: %s is READ-ONLY - left alone", t.path);
                t.ref = nullptr;
                continue;
            }
            t.orig = XPLMGetDataf(t.ref);
            t.want = t.orig * t.factor;
            if (t.wholeNumber) {
                t.want = (float)(int)(t.want + 0.5f);
                if (t.want < 1.0f) t.want = 1.0f;
            }
            t.ready = true;
            // A control already at zero is either disabled or not what the name
            // suggests; scaling zero by anything is still zero, so say so
            // rather than silently doing nothing.
            if (t.orig == 0.0f) {
                xlog("trim: %s is 0 - nothing to scale, left alone (%s)",
                     t.path, t.why);
                t.ref = nullptr;
                continue;
            }
            xlog("trim: %s  %g -> %g  (x%.2f, %s)",
                 t.path, (double)t.orig, (double)t.want, (double)t.factor, t.why);
        }
        if (!t.ref) continue;

        // Re-asserted, like the pager controls: X-Plane rewrites these when
        // settings change or a scene loads, and a value set once quietly
        // reverts at the first rendering-settings touch.
        float cur = XPLMGetDataf(t.ref);
        if (cur != t.want) XPLMSetDataf(t.ref, t.want);
    }
}

// --------------------------------------------------- live budget watch
//
// The counters that answer the two open questions, sampled every few seconds
// while the sim runs rather than once at startup.
//
// WHY A WATCH AND NOT A DUMP. Everything else here reports a value at load,
// which is exactly when nothing interesting is happening. The questions are
// "what does the texture budget do when scenery streams" and "do geometry
// buffers fall out of residency when the camera moves" - both of which are
// about CHANGE over a flight, and a single reading at startup cannot answer
// either.
//
// managed_buffers/missed is the one to watch. It counts buffers that were not
// resident when a draw needed them. If it climbs while the camera moves, the
// frame collapse has an explanation that no amount of staring at the ledger
// would have produced - and if it stays flat, that whole theory is dead and I
// stop proposing it.
static const char *kWatch[] = {
    // ---- IS THE MAIN THREAD BLOCKED, OR IS THE WORK MERELY EXPENSIVE?
    //
    // The profiler says the tex group spikes to 234 ms, but a microprofile
    // group can span worker threads - expensive work on a worker costs nothing
    // if the main thread never waits for it. These two counters are the
    // difference between "texture loading is slow" and "texture loading STALLS
    // THE FRAME", and only the second one explains a frame rate collapse.
    //
    // If wait_workers climbs when the camera moves, the main thread is blocked
    // on texture jobs and more workers may genuinely help. If it stays flat,
    // the cost is somewhere the worker count cannot reach and raising it would
    // have been a wasted run.
    "sim/private/controls/time/wait_workers",
    "sim/private/controls/time/thread_sync_wait",
    "sim/private/controls/region/async_per_frame",
    "sim/private/controls/runloop/time_per_frame_usec",
    "sim/private/controls/runloop/tasks_per_frame",
};

static void budgetWatch()
{
    static XPLMDataRef refs[sizeof(kWatch)/sizeof(kWatch[0])];
    static bool  looked = false;
    static float last[sizeof(kWatch)/sizeof(kWatch[0])];
    static int   ticks = 0;

    if (!looked) {
        looked = true;
        int n = 0;
        for (size_t i = 0; i < sizeof(kWatch)/sizeof(kWatch[0]); ++i) {
            refs[i] = taaFind(kWatch[i]);
            last[i] = -1.0f;
            if (refs[i]) ++n;
        }
        xlog("budget watch: %d of %d counters resolved - sampling every 5 s, "
             "printing only what CHANGED", n,
             (int)(sizeof(kWatch)/sizeof(kWatch[0])));
    }

    // Every 5 seconds at a nominal 60 fps. Cheap, and the interesting motion
    // happens over seconds rather than frames.
    if (++ticks % 300) return;

    // Only what moved. A block of sixteen unchanged numbers every five seconds
    // buries the two that matter, and the whole point is to see the change.
    std::string line;
    for (size_t i = 0; i < sizeof(kWatch)/sizeof(kWatch[0]); ++i) {
        if (!refs[i]) continue;
        XPLMDataTypeID t = XPLMGetDataRefTypes(refs[i]);
        float v = (t & xplmType_Int) ? (float)XPLMGetDatai(refs[i])
                                     : XPLMGetDataf(refs[i]);
        if (v == last[i]) continue;
        last[i] = v;
        const char *shortName = strrchr(kWatch[i], '/');
        // Two components, so device/bytes and host/bytes stay distinguishable.
        const char *p = kWatch[i];
        for (const char *q = kWatch[i]; *q; ++q) if (*q == '/') { p = shortName; break; }
        char buf[96];
        if (v > 1048576.0f) snprintf(buf, sizeof(buf), "%s=%.0fMB ", p + 1, v / 1048576.0);
        else                snprintf(buf, sizeof(buf), "%s=%.0f ", p + 1, v);
        line += buf;
    }
    if (!line.empty()) xlog("budget watch: %s", line.c_str());
}

static void dumpPagerControls()
{
    static bool done = false;
    if (done) return;
    done = true;

    int found = dumpControlList("texture pager art controls", kPagerControls,
                                sizeof(kPagerControls)/sizeof(kPagerControls[0]));
    if (found == 0)
        xlog("none resolved - the paths are assembled differently at runtime "
             "than the string table suggested, so the prefixes need revisiting");

    dumpControlList("VRAM art controls, by ledger category", kVramControls,
                    sizeof(kVramControls)/sizeof(kVramControls[0]));

    // The observe-only set, at its defaults. These are the controls whose
    // direction is unknown, plus the buffer-pager counters - which are the most
    // interesting numbers here and cost nothing to read.
    dumpControlList("watched but not modified", kTrimObserveOnly,
                    sizeof(kTrimObserveOnly)/sizeof(kTrimObserveOnly[0]));

    // Ask X-Plane to keep its reduced render but skip its own spatial upscale,
    // leaving the reconstruction to ours. See the note in launch-xp.cmd.

    g_trimOn = (getenv("TAA_VRAM_TRIM") != nullptr);
    xlog("trim: VRAM trim is %s", g_trimOn ? "ON" : "off (set TAA_VRAM_TRIM=1)");
}

// max_overdrive, held at a value of our choosing.
//
// Chosen alone, out of the five available, because it is the only one that acts
// on how far past the line the pager will run rather than on what it thinks
// things cost. Every "budget" lever tried so far - the driver's heapBudget, the
// stored total at [rbx+0xe8], and --device_budget - applied cleanly and changed
// nothing, which says the constraint is not the size of the budget. Changing
// several controls at once now would make it impossible to say which one moved
// the result.
//
// HELD, not set once. X-Plane rewrites art controls when settings change or a
// new scene loads, and a value written at startup quietly reverts - which is
// why XPVRAMUnlock re-asserts its two. The write only happens when the value
// has actually drifted, so this is not fighting the sim every frame.
// Held art controls.
//
// These act on X-Plane's DECISION to cut textures, which is the only thing that
// matters. The sim budgets from its own metadata - it knows it asked for a
// 4096x4096 texture and reserves accordingly - so anything done behind its back
// to reduce real memory is invisible to it and changes nothing. That is why the
// custom pager was abandoned: it saved real VRAM the sim never learned about,
// and the degradation continued unchanged.
//
// max_overdrive is proven: 16 -> 64 took resident textures from 973 MB to
// 2.13 GB with no side effects.
//
// size_fudge_factor is the other half of the same comparison. The pager weighs
// an ESTIMATE of texture cost against its budget, and a factor above 1 inflates
// that estimate - so at 1.05 it believes every texture costs 5% more than it
// does, and cuts 5% early. Lowering it makes the estimate honest rather than
// pessimistic.
// DISABLING X-PLANE'S PAGER AT RUNTIME, NOT AT STARTUP.
//
// There is no on/off switch for it: the obvious controls - texture_paging,
// all_tex_paging, tex_paging_data and the rest - all report ABSENT in this
// build, so the dataref route to "turn it off" does not exist. And the command
// line route, --gfx_no_pager, disables it for the WHOLE session including load,
// which is where the memory surge actually happens. That is the overrun we
// already hit once.
//
// downscale_cooldown is the lever that works, because of what the pager does
// with it. It is a delay between scale decisions, and the log shows those
// decisions cascading: 1.0 -> 0.5 -> 0.25 -> 0.125 -> 0.0625 inside one second,
// each step making the reported headroom WORSE rather than better. Freezing the
// cooldown at an enormous value means the first decision is also the last: the
// pager is still there, still accounting, but it can no longer act.
//
// So the sequencing writes itself. Their pager runs normally through load,
// which is when it earns its place, and once the flight has settled we freeze
// it and ours is the only thing degrading textures from then on.
//
// Worth knowing: XPVRAMUnlock sets this to 2e6 against a default of 1e7 - five
// times SHORTER, so it downscales five times more eagerly. It re-asserts, so
// this fights it every frame and the last writer wins.
struct HeldControl {
    const char *path;
    const char *env;
    XPLMDataRef ref;
    float want;
    float orig;
    int   reasserts;
    int   afterFrames;   // 0 = hold from the first flight frame
};

static HeldControl g_held[] = {
    { "sim/private/controls/tex/paging/max_overdrive",
      "TAA_MAX_OVERDRIVE",     nullptr, 0.0f, 0.0f, 0, 0 },
    { "sim/private/controls/tex/paging/size_fudge_factor",
      "TAA_SIZE_FUDGE",        nullptr, 0.0f, 0.0f, 0, 0 },
    // Held only after the flight has settled - see above.
    { "sim/private/controls/tex/paging/downscale_cooldown",
      "TAA_DOWNSCALE_COOLDOWN",nullptr, 0.0f, 0.0f, 0, 900 },
};

static void patchTextureScaleFreeze();   // defined below, applied from here
static void patchTextureScaleFloor();    // ditto - raises the 1/16 floor to 1/2

// ---------------------------------------------------------------- scale freeze
//
// FREEZE X-PLANE'S TEXTURE SCALE BY PATCHING THE ONE INSTRUCTION THAT LOWERS IT.
//
// The art controls cannot do this. downscale_cooldown gates only VOLUNTARY
// rescaling; when headroom reaches zero there is an emergency path that ignores
// it completely, which is why textures collapse to a sixteenth the instant an
// upscaler takes its share of VRAM. And --gfx_no_pager removes the pager
// entirely, which killed the sim nine seconds into a scenery load: the pager
// limits how many textures are resident as well as how big they are, and only
// the second half is the problem.
//
// So: freeze the SIZE decision and leave residency alone.
//
// X-Plane computes a new target scale, logs "Target scale moved to %f", and
// writes it back with exactly one instruction:
//
//     movss DWORD PTR [rsi+0x50], xmm6      f3 0f 11 76 50
//
// Five bytes of NOP there and the scale stays wherever it started, which the log
// shows is 1.000000. Eviction, residency limiting and all the accounting carry
// on untouched - the pager simply loses its ability to change resolution.
//
// FOUND BY SIGNATURE, NOT BY ADDRESS. That byte sequence occurs exactly ONCE in
// the whole .text section, verified against this build, so a scan cannot hit the
// wrong site and an X-Plane update that moves the function still finds it. If a
// future build makes it ambiguous or absent, this refuses rather than guesses.
//
// APPLIED DURING FLIGHT, NOT AT LOAD. Load is the surge that overran when the
// pager was removed wholesale; downscaling during load is doing useful work.
// This waits for the flight to settle, exactly as the cooldown hold does.
static bool  g_scaleFrozen  = false;
static bool  g_floorRaised  = false;
static float g_scaleStep    = 0.5f;   // what the pager multiplies by, after patching

// Re-assert the art controls we hold, once per flight loop.
//
// Re-asserted rather than set once: X-Plane rewrites these when settings change
// or a scene loads, and another plugin may be holding the same value -
// XPVRAMUnlock already holds downscale_cooldown, so whichever writes last wins
// and it has to be re-checked every frame.
// ---- X-PLANE'S OWN ANTIALIASING, TURNED OFF WHILE OURS RUNS.
//
// The shader corpus settles what these actually do. The `hdr` family is a single
// pass doing TONEMAP + BLOOM COMPOSITE + FXAA together - `u_hdr_data` is
// {u_offset, u_y_offset, u_bloom_start_mip, u_bloom_end_mip, u_bloom_coef,
// u_sample_count, u_sample_count_inv, u_fxaa_stencil} with an optional
// `u_fxaa_stencil_mask` - and it runs AFTER our resolve. So FXAA blurs an image
// that is already antialiased, which costs time and destroys exactly the
// sub-pixel detail temporal accumulation exists to recover.
//
// MSAA is worse than merely redundant. `fix_hdr_1` binds the lit HDR image as a
// MULTISAMPLED storage image, and 576 of the 1152 gbuffer_lit permutations are
// `2D 0 1 1 1` - arrayed and multisampled. That shape is the one the resolve
// declines outright, so leaving MSAA on does not degrade TAA, it disables it.
//
// Both are saved and restored, and both are overridable: TAA_KEEP_FXAA=1 and
// TAA_KEEP_MSAA=1 leave X-Plane's setting alone. Nothing here fires unless the
// layer's resolve is actually enabled - turning off a user's antialiasing to
// replace it with nothing would be strictly worse than doing nothing.
struct PostAAControl {
    const char *path;
    const char *envKeep;
    XPLMDataRef ref;
    int         saved;
    bool        held;
};

static PostAAControl g_postAA[] = {
    { "sim/private/controls/hdr/use_post_aa", "TAA_KEEP_FXAA", nullptr, 0, false },
    { "sim/private/controls/hdr/msaa_hw",     "TAA_KEEP_MSAA", nullptr, 0, false },
};

static void suppressXPlanePostAA()
{
    // Only while our resolve is running. TAA_RESOLVE is what the layer gates on
    // (taaEnabled()), so the two cannot disagree.
    if (!getenv("TAA_RESOLVE")) return;

    for (size_t i = 0; i < sizeof(g_postAA)/sizeof(g_postAA[0]); ++i) {
        PostAAControl &c = g_postAA[i];
        if (c.held) {
            // Re-assert: X-Plane rewrites these from its own settings when the
            // rendering options change, and a value that silently comes back is
            // the same failure as one that never applied.
            if (c.ref && XPLMGetDatai(c.ref) != 0) XPLMSetDatai(c.ref, 0);
            continue;
        }
        if (getenv(c.envKeep)) continue;
        if (!c.ref) {
            c.ref = taaFind(c.path);
            if (!c.ref) continue;      // taaFind already recorded the miss
        }
        c.saved = XPLMGetDatai(c.ref);
        if (c.saved != 0) {
            XPLMSetDatai(c.ref, 0);
            xlog("post-aa: %s %d -> 0. X-Plane's own pass runs AFTER our resolve, "
                 "so it would blur a frame that is already antialiased. "
                 "%s=1 keeps it.", c.path, c.saved, c.envKeep);
        }
        c.held = true;
    }
}

static void restoreXPlanePostAA()
{
    for (size_t i = 0; i < sizeof(g_postAA)/sizeof(g_postAA[0]); ++i) {
        PostAAControl &c = g_postAA[i];
        if (!c.held || !c.ref) continue;
        XPLMSetDatai(c.ref, c.saved);
        xlog("post-aa: restored %s = %d", c.path, c.saved);
        c.held = false;
    }
}

static void holdArtControls()
{
    suppressXPlanePostAA();
    // Applied immediately. Raising the FLOOR only bounds how far the pager may
    // cut, so it is harmless early and useless late - the collapse happened 86
    // seconds in, and anything that waits that long arrives after the damage.
    patchTextureBudgetReserve();
    patchTextureScaleFloor();
    applyVramTrim();
    budgetWatch();

    // The freeze waits for the flight to settle, because pinning a scale is
    // only safe once the scale is good.
    // FREEZE EARLY, because the scale no longer starts low.
    //
    // 900 frames was chosen when the pager was dragging the scale down during
    // load - freezing before it recovered would have pinned a bad value. With
    // the budget reserve cut the very first line of the log is now "Target
    // scale moved to 1.000000", so there is nothing to wait for, and waiting
    // leaves fifteen seconds in which it can move.
    //
    // 120 frames is ~2-4 s: past the first scenery burst, long before anything
    // has a reason to rescale.
    if ((int)g_flightFrames >= 120) patchTextureScaleFreeze();

    for (size_t i = 0; i < sizeof(g_held)/sizeof(g_held[0]); ++i) {
        HeldControl &h = g_held[i];
        if (h.want <= 0.0f) continue;

        // Deferred controls wait for the load surge to be over. Applying the
        // cooldown freeze at frame zero would pin the pager at whatever scale
        // the loading screen happened to be at, which is the opposite of what
        // it is for.
        if (h.afterFrames && (int)g_flightFrames < h.afterFrames) continue;

        if (!h.ref) {
            h.ref = taaFind(h.path);
            if (!h.ref) { h.want = 0.0f; xlog("hold: %s absent", h.path); continue; }
            h.orig = XPLMGetDataf(h.ref);
            xlog("hold: %s -> %g (was %g)", h.path, h.want, h.orig);
            if (h.afterFrames)
                xlog("hold: X-Plane's pager is now FROZEN at flight frame %llu. "
                     "It keeps accounting but can no longer change scale, so "
                     "ours is the only thing degrading textures from here.",
                     (unsigned long long)g_flightFrames);
        }

        float cur = XPLMGetDataf(h.ref);
        if (cur != h.want) {
            XPLMSetDataf(h.ref, h.want);
            if (++h.reasserts <= 2 || h.reasserts % 500 == 0)
                xlog("hold: re-asserted %s = %g (was %g, %d times)",
                     h.path, h.want, cur, h.reasserts);
        }
    }
}


// RAISE THE PAGER'S FLOOR so it can cut one step, not four.
//
// X-Plane's scale search halves the texture scale until the estimated cost fits
// the budget, with a hard floor of 1/16. On this machine it walked the whole way
// down in 0.8 seconds - 1.0, 0.5, 0.25, 0.125, 0.0625 - and a sixteenth of the
// resolution is not a graceful degradation, it is a different sim.
//
// The loop, at 0x15bc820:
//
//     0x15bca4c  movss xmm9, [rip+0xf5176b]   ; floor = 0.0625
//     0x15bca55  comiss xmm7, xmm9
//     0x15bcb1e  cmp   rdi, r15               ; estimated cost vs budget
//     0x15bcb21  jbe   0x15bcb78              ; fits -> accept this scale
//     0x15bcb23  mulss xmm7, xmm11            ; else halve
//     0x15bcb28  comiss xmm7, xmm9            ; still >= floor?
//     0x15bcb31  jae   0x15bca70              ; yes -> try again
//     0x15bcb37  mov   [rsp+0x48], 0x3d800000 ; no  -> use 0.0625 anyway
//
// TWO SITES, and patching either alone accomplishes nothing. Raise only the
// floor and the loop stops earlier but the second site still writes 0.0625.
// Change only the second and the loop still walks all the way down first.
//
// The floor is moved by REPOINTING the RIP-relative displacement at the 0.5
// constant the halving factor already uses, rather than by editing 0.0625 where
// it sits - that lives in a shared read-only pool and an unknown number of
// unrelated functions read it.
//
// SITE 2 IS NOT UNIQUE: its eight bytes occur twice in .text. So it is not
// searched for. Site 1's nine bytes ARE unique, and site 2 sits at a fixed
// +0xEB from it, so the unique site anchors the ambiguous one and the bytes
// there are verified before anything is written.
//
// THE TRADE, stated plainly: if half resolution still does not fit, X-Plane will
// use it anyway and run over its budget. Eviction still works, so this is not
// unbounded, but it can be unstable - it is trading a guaranteed ugly image for
// a possible allocation failure. Ive seen this kind of thing misbehave, so it is
// opt-in and refuses rather than guesses whenever the bytes are not exactly
// what it expects.
static float g_floorValue = 0.0625f;   // what the floor ended up being

static void patchTextureScaleFloor()
{
    if (g_floorRaised) return;
    // ---- DEFAULTS TO 1.0: THE PAGER MAY NEVER CUT A TEXTURE.
    //
    // This used to do nothing unless a variable was set, which meant it only
    // ever ran under the development launcher. Installed, the sim paged
    // textures down as usual and the code that exists to stop it sat inert.
    //
    // 1.0 is the floor because the reachable scales are powers of two - 2, 1,
    // 0.5, 0.25, 0.125, 0.0625 - so a floor of 1 means full size and no cut.
    const char *env = getenv("TAA_SCALE_FLOOR");
    if (!env || !env[0]) env = "1";

    // A VALUE, not a switch.
    //
    // This took "1" to mean "enabled" and hardcoded a 0.5 floor, which conflated
    // two different decisions. The reachable scales are 2, 1, 0.5, 0.25, 0.125
    // and 0.0625 - powers of two, because dimensions come out as
    // (int)(width * scale) and block-compressed textures only shed whole mip
    // levels - so the only meaningful choice is WHERE to stop, and that is worth
    // being able to say directly.
    //
    // 1.0 means the pager may never cut at all.
    float want = (float)atof(env);
    if (!(want > 0.0f && want <= 1.0f)) {
        xlog("scale floor: %g is not in (0, 1] - refusing. The pager's own floor "
             "is 0.0625 and values above 1.0 are not scales it cuts to.",
             (double)want);
        g_floorRaised = true;
        return;
    }

    //  movss xmm9, dword ptr [rip + 0xf5176b]      -> 0.0625, the floor
    static const unsigned char kFloorOrig[9] =
        { 0xF3, 0x44, 0x0F, 0x10, 0x0D, 0x6B, 0x17, 0xF5, 0x00 };
    //  mov dword ptr [rsp+0x48], 0x3d800000        -> 0.0625, the fallback
    static const unsigned char kFallbackOrig[8] =
        { 0xC7, 0x44, 0x24, 0x48, 0x00, 0x00, 0x80, 0x3D };
    static const ptrdiff_t kFallbackDelta = 0xEB;   // site2 - site1

    HMODULE base = GetModuleHandleA(nullptr);
    if (!base) { xlog("scale floor: no module handle"); return; }
    unsigned char *b = (unsigned char*)base;
    IMAGE_DOS_HEADER   *dos = (IMAGE_DOS_HEADER*)b;
    IMAGE_NT_HEADERS64 *nt  = (IMAGE_NT_HEADERS64*)(b + dos->e_lfanew);
    IMAGE_SECTION_HEADER *sec = IMAGE_FIRST_SECTION(nt);

    unsigned char *found = nullptr;
    unsigned char *rdata = nullptr;
    size_t rdataSize = 0;
    int hits = 0;
    for (int i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        if (memcmp(sec[i].Name, ".text", 5) == 0) {
            unsigned char *p = b + sec[i].VirtualAddress;
            size_t n = sec[i].Misc.VirtualSize;
            for (size_t k = 0; k + sizeof(kFloorOrig) <= n; ++k) {
                if (memcmp(p + k, kFloorOrig, sizeof(kFloorOrig)) == 0) {
                    if (!found) found = p + k;
                    if (++hits > 1) break;
                }
            }
        } else if (memcmp(sec[i].Name, ".rdata", 6) == 0) {
            rdata     = b + sec[i].VirtualAddress;
            rdataSize = sec[i].Misc.VirtualSize;
        }
    }

    if (!found) {
        xlog("scale floor: the floor load was not found - X-Plane has changed. "
             "Refusing to patch.");
        g_floorRaised = true;              // do not rescan every frame
        return;
    }
    if (hits > 1) {
        xlog("scale floor: the floor load matches %d sites, not 1 - refusing.", hits);
        g_floorRaised = true;
        return;
    }

    unsigned char *fallback = found + kFallbackDelta;
    if (memcmp(fallback, kFallbackOrig, sizeof(kFallbackOrig)) != 0) {
        xlog("scale floor: the fallback store is not at +0x%X from the floor load "
             "- the function has been rearranged. Refusing to patch, because "
             "raising the floor alone would change nothing.", (unsigned)kFallbackDelta);
        g_floorRaised = true;
        return;
    }

    // The floor is moved by REPOINTING the RIP-relative displacement at a
    // constant already present in .rdata, never by editing 0.0625 where it sits:
    // that lives in a shared read-only pool and an unknown number of unrelated
    // functions read the same four bytes.
    if (!rdata) {
        xlog("scale floor: no .rdata section - cannot find a constant to point at.");
        g_floorRaised = true;
        return;
    }
    unsigned char *target = nullptr;
    for (size_t k = 0; k + 4 <= rdataSize; k += 4) {
        float v;
        memcpy(&v, rdata + k, 4);
        if (v == want) { target = rdata + k; break; }
    }
    if (!target) {
        xlog("scale floor: no aligned constant equal to %g exists in .rdata. The "
             "reachable scales are powers of two, so 1, 0.5, 0.25, 0.125 and "
             "0.0625 are the values worth asking for.", (double)want);
        g_floorRaised = true;
        return;
    }

    unsigned char *next = found + sizeof(kFloorOrig);
    long long disp = (long long)(target - next);
    if (disp > 0x7FFFFFFFLL || disp < -0x80000000LL) {
        xlog("scale floor: constant too far for a 32-bit displacement - refusing.");
        g_floorRaised = true;
        return;
    }
    int32_t d32 = (int32_t)disp;

    DWORD old = 0;
    if (!VirtualProtect(found, sizeof(kFloorOrig), PAGE_EXECUTE_READWRITE, &old)) {
        xlog("scale floor: VirtualProtect failed at the floor load (%lu)", GetLastError());
        g_floorRaised = true;
        return;
    }
    memcpy(found + 5, &d32, 4);
    VirtualProtect(found, sizeof(kFloorOrig), old, &old);

    // The fallback is an IMMEDIATE inside the instruction, so it is written
    // directly rather than repointed.
    unsigned char newFallback[8];
    memcpy(newFallback, kFallbackOrig, 4);      // C7 44 24 48
    memcpy(newFallback + 4, &want, 4);
    if (!VirtualProtect(fallback, sizeof(newFallback), PAGE_EXECUTE_READWRITE, &old)) {
        xlog("scale floor: VirtualProtect failed at the fallback store (%lu) - the "
             "floor is raised but the fallback still writes 0.0625, so the pager "
             "can still reach a sixteenth.", GetLastError());
        g_floorRaised = true;
        return;
    }
    memcpy(fallback, newFallback, sizeof(newFallback));
    VirtualProtect(fallback, sizeof(newFallback), old, &old);

    FlushInstructionCache(GetCurrentProcess(), found, kFallbackDelta + sizeof(newFallback));
    g_floorRaised = true;
    g_floorValue  = want;
    xlog("scale floor: texture scale floor raised from 0.0625 to %g (patched "
         "+0x%llX and +0x%llX in memory only - X-Plane.exe on disk is untouched)",
         (double)want,
         (unsigned long long)(found - b),
         (unsigned long long)(fallback - b));
    if (want >= 1.0f)
        xlog("scale floor: at 1.0 the pager can no longer reduce texture "
             "resolution AT ALL. It still evicts and still limits residency, but "
             "when it decides to cut it will run over budget instead. This is the "
             "most aggressive setting available - watch for allocation failures.");
    else
        xlog("scale floor: the pager may now cut no further than %g. If that still "
             "does not fit it will run over budget rather than cut again - watch "
             "for allocation failures if this is too aggressive for the VRAM "
             "available.", (double)want);
}

static void patchTextureScaleFreeze()
{
    if (g_scaleFrozen) return;

    const char *env = getenv("TAA_FREEZE_TEX_SCALE");
    if (!env || env[0] != '1') return;

    static const unsigned char kStoreOrig[5] = { 0xF3, 0x0F, 0x11, 0x76, 0x50 };

    HMODULE base = GetModuleHandleA(nullptr);
    if (!base) { xlog("scale freeze: no module handle"); return; }

    // Walk the PE headers to find .text rather than trusting a fixed offset.
    unsigned char *b = (unsigned char*)base;
    IMAGE_DOS_HEADER *dos = (IMAGE_DOS_HEADER*)b;
    IMAGE_NT_HEADERS64 *nt = (IMAGE_NT_HEADERS64*)(b + dos->e_lfanew);
    IMAGE_SECTION_HEADER *sec = IMAGE_FIRST_SECTION(nt);

    unsigned char *found = nullptr;
    int hits = 0;
    for (int i = 0; i < nt->FileHeader.NumberOfSections; ++i) {
        if (memcmp(sec[i].Name, ".text", 5) != 0) continue;
        unsigned char *p   = b + sec[i].VirtualAddress;
        size_t         n   = sec[i].Misc.VirtualSize;
        for (size_t k = 0; k + sizeof(kStoreOrig) <= n; ++k) {
            if (memcmp(p + k, kStoreOrig, sizeof(kStoreOrig)) == 0) {
                if (!found) found = p + k;
                if (++hits > 1) break;
            }
        }
        break;
    }

    if (!found) {
        xlog("scale freeze: the target scale store was not found - X-Plane has "
             "changed. Refusing to patch.");
        return;
    }
    if (hits > 1) {
        // Ambiguity means the signature no longer identifies one site, and
        // patching the wrong one would corrupt something unrelated.
        xlog("scale freeze: the byte pattern matches %d sites, not 1 - it no "
             "longer identifies the target uniquely. Refusing to patch.", hits);
        return;
    }

    unsigned char nops[5] = { 0x90, 0x90, 0x90, 0x90, 0x90 };
    DWORD old = 0;
    if (!VirtualProtect(found, sizeof(nops), PAGE_EXECUTE_READWRITE, &old)) {
        xlog("scale freeze: VirtualProtect failed (%lu)", GetLastError());
        return;
    }
    memcpy(found, nops, sizeof(nops));
    VirtualProtect(found, sizeof(nops), old, &old);
    FlushInstructionCache(GetCurrentProcess(), found, sizeof(nops));

    g_scaleFrozen = true;
    xlog("scale freeze: texture target scale is now PINNED at its current value "
         "(patched +0x%zX in memory only - X-Plane.exe on disk is untouched)",
         (size_t)(found - b));
    xlog("scale freeze: the pager still evicts and still limits residency; it "
         "has only lost the ability to change texture resolution. Watch for the "
         "ABSENCE of further 'Target scale moved to' lines in Log.txt.");
}

static void patchTextureBudget()
{
    const char *env = getenv("TAA_VRAM_PATCH_GB");
    if (!env) return;

    double gb = atof(env);
    if (gb < 1.0 || gb > 4.0) {
        xlog("vram patch: %.2f GB is outside the 1-4 GB the instruction can "
             "encode - refusing", gb);
        return;
    }
    uint64_t bytes = (uint64_t)(gb * 1073741824.0);
    if (bytes > 0xFFFFFFFFull) bytes = 0xFFFFFFF0ull;

    HMODULE base = GetModuleHandleA(nullptr);
    if (!base) { xlog("vram patch: no module handle"); return; }
    unsigned char *addr = (unsigned char*)base + kBudgetRva;

    if (memcmp(addr, kBudgetOrig, sizeof(kBudgetOrig)) != 0) {
        xlog("vram patch: the bytes at +0x%zX are not the expected budget load "
             "- X-Plane has changed. Refusing to patch.", kBudgetRva);
        xlog("vram patch: found %02X %02X %02X %02X %02X %02X %02X",
             addr[0], addr[1], addr[2], addr[3], addr[4], addr[5], addr[6]);
        return;
    }

    unsigned char patch[7];
    patch[0] = 0xB8;                                  // mov eax, imm32
    patch[1] = (unsigned char)( bytes        & 0xFF);
    patch[2] = (unsigned char)((bytes >>  8) & 0xFF);
    patch[3] = (unsigned char)((bytes >> 16) & 0xFF);
    patch[4] = (unsigned char)((bytes >> 24) & 0xFF);
    patch[5] = 0x90;                                  // nop
    patch[6] = 0x90;                                  // nop

    DWORD old = 0;
    if (!VirtualProtect(addr, sizeof(patch), PAGE_EXECUTE_READWRITE, &old)) {
        xlog("vram patch: VirtualProtect failed (%lu)", GetLastError());
        return;
    }
    memcpy(addr, patch, sizeof(patch));
    VirtualProtect(addr, sizeof(patch), old, &old);
    FlushInstructionCache(GetCurrentProcess(), addr, sizeof(patch));

    xlog("vram patch: texture budget ceiling set to %.2f GB (in memory only - "
         "X-Plane.exe on disk is untouched)", bytes / 1073741824.0);
    xlog("vram patch: eviction and accounting are unchanged, so the pager still "
         "works. This raises the ceiling, it does not remove it.");
}

// Rotation part of the aircraft body -> world transform, from the attitude
// quaternion, written column-major to match X-Plane's matrix datarefs.
//
// `conjugate` selects the opposite handedness. Which one X-Plane's q uses is
// settled by measurement below rather than by trusting a convention, because a
// backwards rotation produces a matrix that is orthonormal, invertible and
// entirely wrong - it fails silently, which is the expensive kind. The same
// mistake on modelview_matrix cost a flight and 205 false teleport resets.
static void taaBodyRotation(float *R, const float q[4], bool conjugate)
{
    float w = q[0], x = q[1], y = q[2], z = q[3];
    if (conjugate) { x = -x; y = -y; z = -z; }

    float n = w*w + x*x + y*y + z*z;
    if (n > 1e-12f) { float s = 1.0f / sqrtf(n); w*=s; x*=s; y*=s; z*=s; }
    else            { w = 1.0f; x = y = z = 0.0f; }

    R[0]  = 1-2*(y*y+z*z);  R[1]  = 2*(x*y + w*z);  R[2]  = 2*(x*z - w*y);  R[3]  = 0;
    R[4]  = 2*(x*y - w*z);  R[5]  = 1-2*(x*x+z*z);  R[6]  = 2*(y*z + w*x);  R[7]  = 0;
    R[8]  = 2*(x*z + w*y);  R[9]  = 2*(y*z - w*x);  R[10] = 1-2*(x*x+y*y);  R[11] = 0;
    R[12] = 0;              R[13] = 0;              R[14] = 0;              R[15] = 1;
}

// Per-slot tracking cache. See the comment on TaaMovingObject::generation for
// why this is versioned rather than a plain "last position" array.
struct SlotTrack {
    float    px, py, pz;
    bool     hadPrev;
    bool     occupied;
    uint32_t generation;
};
static SlotTrack g_slots[TAA_TRAFFIC_SLOTS + 1];   // +1 for the user aircraft

// ---- prop / rotor discs
static XPLMDataRef g_drNumEngines = nullptr;
static XPLMDataRef g_drPropRpm    = nullptr;   // float[8]
static XPLMDataRef g_drEngRotRate = nullptr;   // float[8], rad/s

// Tunables, overridable from taa.ini so this does not need a rebuild to retune.
static float g_lodBias      = -0.5f;
static float g_renderScale  = 1.0f;
static int   g_jitterPhases = 8;
static bool  g_objectsOn    = true;
static float g_trafficRadius = 35.0f;
static int   g_jitterIndex   = 0;
// 2.5 m covers the panel and glareshield of an airliner without reaching the
// windscreen pillars or anything outside.

// ---- backend selection, live-adjustable.
// Default OFF, so nothing touches the image until it is asked for.
//
// This is the first stage that changes what the user sees, and it now writes

static int   g_enabled     = 1;

// ============================================================ control surface
//
// Everything adjustable is exposed as a custom DATAREF rather than being
// reachable only from our own window.
//
// Datarefs are X-Plane's universal interface: the in-sim window drives them,
// FlyWithLua or any other scripting plugin drives the same ones, and they can
// be bound to hardware or shown on a StreamDeck. Building a Lua script instead
// would have given scripting access and nothing else; building only a window
// would have given the opposite. This gives both for the same work.
//
//   taaimpl/enabled            int    0/1 master switch
//   taaimpl/upscaler           int    0=Off 1=TAA 2=FSR3 3=FSR4 4=DLSS
//   taaimpl/quality            int    0=Native 1=Ultra 2=Quality 3=Balanced 4=Perf
//   taaimpl/sharpness          float  0..1
//   taaimpl/optical_flow       int    0/1 use NV optical flow if available
//   taaimpl/lod_bias           float  texture mip bias
//   taaimpl/render_scale       float  READ-ONLY, derived from quality
//   taaimpl/upscaler_available int[5] READ-ONLY availability per backend
//   taaimpl/layer_attached     int    READ-ONLY

static XPLMDataRef g_myEnabled = nullptr, g_myUpscaler = nullptr, g_myQuality = nullptr;
static XPLMDataRef g_myLodBias = nullptr;
static XPLMDataRef g_myAvail = nullptr, g_myAttached = nullptr;
static XPLMDataRef g_myVramTotal = nullptr, g_myVramBudget = nullptr,
                   g_myVramUsed  = nullptr;
static XPLMDataRef g_myTexFloor  = nullptr, g_myTexStep    = nullptr;
static XPLMDataRef g_myRevZMat   = nullptr, g_myViewportW  = nullptr,
                   g_myViewportH = nullptr, g_myJitPhases  = nullptr,
                   g_myReverseZ  = nullptr, g_myObjCount   = nullptr;

static int   getEnabled(void*)          { return g_enabled; }
static void  setEnabled(void*, int v)   { g_enabled = v ? 1 : 0; }


static float getLodBias(void*)          { return g_lodBias; }
static void  setLodBias(void*, float v) { g_lodBias = v < -3.0f ? -3.0f : (v > 1.0f ? 1.0f : v); }

// VRAM, as the DRIVER reports it - not as X-Plane's pager estimates it. The two
// have disagreed by gigabytes all afternoon, and the pager's own "available"
// figure is the one that has repeatedly turned out to be the fiction.
static int   getVramTotal(void*)        { return g_share ? (int)g_share->vramTotalMB  : 0; }
static int   getVramBudget(void*)       { return g_share ? (int)g_share->vramBudgetMB : 0; }
static int   getVramUsed(void*)         { return g_share ? (int)g_share->vramUsageMB  : 0; }

// The pager's real limits after patching. Stock values if a patch refused.
// g_floorValue, not a hardcoded 0.5 - the floor is a configurable value now, and
// a panel that reports the wrong one is worse than one that reports nothing.
static float getTexFloor(void*)         { return g_floorValue; }
static float getTexStep(void*)          { return g_scaleStep; }
static int   getAttached(void*)         { return (g_share && g_share->layerAttached) ? 1 : 0; }

// ---- diagnostics the panel had no way to show.
//
// reverse_z is the one that earns its place. The LAYER guesses the depth
// convention from a histogram of depth texels and hands the answer to FSR2 as
// DEPTH_INVERTED; the PLUGIN derives it from X-Plane's actual projection
// matrix. Those disagreed for a whole session - the histogram said reverse-Z
// while its own sanity check computed a mean scene distance of a billion metres
// - and nothing put the two numbers side by side. A wrong depth convention
// inverts FSR2's disocclusion test, which reads as shimmer rather than as an
// error, so this is worth a line on screen.
static int   getViewportW(void*)    { return g_share ? g_share->viewportW : 0; }
static int   getViewportH(void*)    { return g_share ? g_share->viewportH : 0; }
static int   getJitterPhases(void*) { return g_share ? g_share->jitterPhases : 0; }
static int   getObjectCount(void*)  { return g_share ? g_share->objectCount : 0; }
static float getRenderScale(void*)  { return g_share ? g_share->renderScale : 1.0f; }

// The residual, out of the shared block. It travels as milli-pixels because
// that block is written by the layer and read here, and a torn float reads as a
// denormal rather than as a stale number. It comes back out as a float because
// "0.002 px" is what a person reads, not "2".
static float getMvResidual(void*)
    { return g_share ? (float)g_share->mvResidualMilliPx * 0.001f : -1.0f; }
static float getMvResidualP95(void*)
    { return g_share ? (float)g_share->mvResidualP95MilliPx * 0.001f : -1.0f; }
static int   getMvSamples(void*)
    { return g_share ? (int)g_share->mvResidualSamples : 0; }
static int   getMvVelocityMB(void*)
    { return g_share ? (int)g_share->mvVelocityMB : 0; }
static int   getMvPatched(void*)
    { return g_share ? (int)g_share->mvPipelinesPatched : 0; }
static int   getMvRejected(void*)
    { return g_share ? (int)g_share->mvPipelinesRejected : 0; }

// The build's own version, so a bug report names a build. MV_VERSION comes from
// the VERSION file by way of build.ps1; the fallback is deliberately not a
// plausible number.
#ifndef MV_VERSION
#define MV_VERSION "0.0.0-unversioned"
#endif
static int getVersionString(void *, void *out, int off, int max)
{
    static const char *v = MV_VERSION;
    const int len = (int)strlen(v);
    if (!out) return len;
    if (off >= len) return 0;
    int n = len - off; if (n > max) n = max;
    memcpy(out, v + off, (size_t)n);
    return n;
}

// TWO reverse-Z answers, exposed separately ON PURPOSE.
//
// share.h already says why they exist: the convention is "independently derived
// from the projection matrix, then cross-checked", because getting it backwards
// "silently inverts the shader's distance test... in a way that still looks
// plausible - the worst kind of bug to chase from screenshots".
//
// The cross-check was built and then never surfaced anywhere a person would see
// it. Meanwhile the LAYER makes a third determination from a histogram of depth
// texels and hands THAT to FSR2 as DEPTH_INVERTED. This session it decided
// reverse-Z on 7% of texels while its own sanity line computed a mean scene
// distance of a billion metres, and nothing compared the two.
static int   getReverseZ(void*)     { return g_share ? g_share->reverseZ : 0; }
static int   getReverseZMat(void*)  { return g_share ? g_share->reverseZFromMatrix : 0; }






static void registerDatarefs()
{
    g_myEnabled  = XPLMRegisterDataAccessor("taaimpl/enabled", xplmType_Int, 1,
                     getEnabled, setEnabled, 0,0,0,0,0,0,0,0,0,0, nullptr, nullptr);
    g_myAttached = XPLMRegisterDataAccessor("taaimpl/layer_attached", xplmType_Int, 0,
                     getAttached, nullptr, 0,0,0,0,0,0,0,0,0,0, nullptr, nullptr);

    g_myLodBias  = XPLMRegisterDataAccessor("taaimpl/lod_bias", xplmType_Float, 1,
                     0,0, getLodBias, setLodBias, 0,0,0,0,0,0,0,0, nullptr, nullptr);

    // VRAM, straight from the driver via the layer. Read-only - these report,
    // they do not steer.
    g_myVramTotal  = XPLMRegisterDataAccessor("taaimpl/vram_total_mb", xplmType_Int, 0,
                       getVramTotal, nullptr, 0,0,0,0,0,0,0,0,0,0, nullptr, nullptr);
    g_myVramBudget = XPLMRegisterDataAccessor("taaimpl/vram_budget_mb", xplmType_Int, 0,
                       getVramBudget, nullptr, 0,0,0,0,0,0,0,0,0,0, nullptr, nullptr);
    g_myVramUsed   = XPLMRegisterDataAccessor("taaimpl/vram_used_mb", xplmType_Int, 0,
                       getVramUsed, nullptr, 0,0,0,0,0,0,0,0,0,0, nullptr, nullptr);

    // What the binary patches actually achieved, so the panel can show the
    // pager's real limits rather than what the launcher asked for. A patch that
    // refused leaves these at the stock values, which is exactly the case worth
    // being able to see without reading Log.txt.
    g_myTexFloor = XPLMRegisterDataAccessor("taaimpl/tex_scale_floor", xplmType_Float, 0,
                     0,0, getTexFloor, nullptr, 0,0,0,0,0,0,0,0, nullptr, nullptr);
    g_myTexStep  = XPLMRegisterDataAccessor("taaimpl/tex_scale_step", xplmType_Float, 0,
                     0,0, getTexStep, nullptr, 0,0,0,0,0,0,0,0, nullptr, nullptr);

    g_myRevZMat   = XPLMRegisterDataAccessor("taaimpl/reverse_z_matrix", xplmType_Int, 0,
                      getReverseZMat, nullptr, 0,0,0,0,0,0,0,0,0,0, nullptr, nullptr);
    g_myViewportW = XPLMRegisterDataAccessor("taaimpl/viewport_w", xplmType_Int, 0,
                      getViewportW, nullptr, 0,0,0,0,0,0,0,0,0,0, nullptr, nullptr);
    g_myViewportH = XPLMRegisterDataAccessor("taaimpl/viewport_h", xplmType_Int, 0,
                      getViewportH, nullptr, 0,0,0,0,0,0,0,0,0,0, nullptr, nullptr);
    g_myJitPhases = XPLMRegisterDataAccessor("taaimpl/jitter_phases", xplmType_Int, 0,
                      getJitterPhases, nullptr, 0,0,0,0,0,0,0,0,0,0, nullptr, nullptr);
    g_myReverseZ  = XPLMRegisterDataAccessor("taaimpl/reverse_z", xplmType_Int, 0,
                      getReverseZ, nullptr, 0,0,0,0,0,0,0,0,0,0, nullptr, nullptr);
    g_myObjCount  = XPLMRegisterDataAccessor("taaimpl/moving_objects", xplmType_Int, 0,
                      getObjectCount, nullptr, 0,0,0,0,0,0,0,0,0,0, nullptr, nullptr);


    // render_scale was DOCUMENTED at the top of this file and never registered.
    // The panel read it, FlyWithLua reported a missing name through its own
    // error path rather than through Lua, and it quarantined the script and
    // took the whole Lua engine down with it - every other script the user had
    // loaded, for one name in a comment that was never code.

    XPLMRegisterDataAccessor("taaimpl/render_scale", xplmType_Float, 0,
        nullptr, nullptr, getRenderScale, nullptr, 0,0,0,0,0,0,0,0, nullptr, nullptr);

    XPLMRegisterDataAccessor("taaimpl/mv_residual_px", xplmType_Float, 0,
        nullptr, nullptr, getMvResidual, nullptr, 0,0,0,0,0,0,0,0, nullptr, nullptr);
    XPLMRegisterDataAccessor("taaimpl/mv_residual_p95_px", xplmType_Float, 0,
        nullptr, nullptr, getMvResidualP95, nullptr, 0,0,0,0,0,0,0,0, nullptr, nullptr);
    XPLMRegisterDataAccessor("taaimpl/mv_samples", xplmType_Int, 0,
        getMvSamples, nullptr, 0,0,0,0,0,0,0,0,0,0, nullptr, nullptr);
    XPLMRegisterDataAccessor("taaimpl/velocity_mb", xplmType_Int, 0,
        getMvVelocityMB, nullptr, 0,0,0,0,0,0,0,0,0,0, nullptr, nullptr);
    XPLMRegisterDataAccessor("taaimpl/pipelines_patched", xplmType_Int, 0,
        getMvPatched, nullptr, 0,0,0,0,0,0,0,0,0,0, nullptr, nullptr);
    XPLMRegisterDataAccessor("taaimpl/pipelines_rejected", xplmType_Int, 0,
        getMvRejected, nullptr, 0,0,0,0,0,0,0,0,0,0, nullptr, nullptr);
    XPLMRegisterDataAccessor("taaimpl/version", xplmType_Data, 0,
        nullptr, nullptr, nullptr, nullptr, 0,0,0,0,0,0,
        getVersionString, nullptr, nullptr, nullptr);

    xlog("registered %d datarefs under taaimpl/ (usable from FlyWithLua or any "
         "script); build %s", 22, MV_VERSION);
}

static void unregisterDatarefs()
{
    // Every accessor this plugin registers, so none is left dangling when the
    // plugin unloads. The previous list named nine, four of which no longer
    // existed and eight of which were registered but never named here - an
    // accessor that outlives its plugin is a dataref X-Plane will call into
    // unloaded code for.
    XPLMDataRef refs[] = { g_myEnabled, g_myAttached, g_myLodBias,
                           g_myVramTotal, g_myVramBudget, g_myVramUsed,
                           g_myTexFloor, g_myTexStep,
                           g_myRevZMat, g_myViewportW, g_myViewportH,
                           g_myJitPhases, g_myReverseZ, g_myObjCount };
    for (size_t i = 0; i < sizeof(refs)/sizeof(refs[0]); ++i)
        if (refs[i]) XPLMUnregisterDataAccessor(refs[i]);
}

// Previous-frame scalars. Kept here rather than in the shared block because the
// layer has no use for them - they only feed history-reset detection.
static float  g_prevFov      = -1.0f;
static int    g_prevViewType = -1;
static int    g_prevPaused   = -1;
static int    g_prevVpW = 0, g_prevVpH = 0;
static double g_prevSimTime  = -1.0;
static float  g_prevSpeed    = 0.0f;

static bool openShare()
{
    if (g_share) return true;

    g_shareHandle = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                                       0, sizeof(TaaShare), TAA_SHARE_NAME);
    if (!g_shareHandle) {
        xlog("share: CreateFileMapping failed (%lu)", GetLastError());
        return false;
    }
    g_share = (TaaShare*)MapViewOfFile(g_shareHandle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(TaaShare));
    if (!g_share) {
        xlog("share: MapViewOfFile failed (%lu)", GetLastError());
        return false;
    }

    memset(g_share, 0, sizeof(*g_share));
    g_share->magic      = TAA_MAGIC;
    g_share->version    = TAA_VERSION;
    g_share->structSize = (uint32_t)sizeof(TaaShare);
    xlog("share: mapped %d bytes at %s (v%d)",
         (int)sizeof(TaaShare), TAA_SHARE_NAME, TAA_VERSION);
    return true;
}

// ------------------------------------------------- external control panel
//
// A second, much smaller mapping that the panel executable writes and this
// plugin reads. See control.h for why it is not part of TaaShare and not a
// dataref.
//
// Failure here is not an error. The panel is optional, and a sim that cannot
// create the mapping should carry on with its menu and its datarefs exactly as
// before rather than complain about a window nobody opened.
static HANDLE      g_ctlHandle = nullptr;
static TaaControl *g_ctl       = nullptr;
static uint32_t    g_ctlSeen   = 0;

static void openControl()
{
    if (g_ctl) return;

    g_ctlHandle = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                                     0, sizeof(TaaControl), TAA_CONTROL_NAME);
    if (!g_ctlHandle) {
        xlog("control: CreateFileMapping failed (%lu) - the external panel will "
             "not be able to reach this session", GetLastError());
        return;
    }

    // Whether we created it or opened one the panel made first decides whether
    // the header may be initialised: writing magic over a block the panel has
    // already put a request in would discard that request.
    bool existed = (GetLastError() == ERROR_ALREADY_EXISTS);

    g_ctl = (TaaControl*)MapViewOfFile(g_ctlHandle, FILE_MAP_ALL_ACCESS, 0, 0,
                                       sizeof(TaaControl));
    if (!g_ctl) {
        xlog("control: MapViewOfFile failed (%lu)", GetLastError());
        return;
    }

    if (!existed || g_ctl->magic != TAA_CONTROL_MAGIC) {
        memset(g_ctl, 0, sizeof(*g_ctl));
        g_ctl->magic      = TAA_CONTROL_MAGIC;
        g_ctl->version    = TAA_CONTROL_VERSION;
        g_ctl->structSize = (uint32_t)sizeof(TaaControl);
    }
    g_ctlSeen = g_ctl->seq;
    xlog("control: mapped %d bytes at %s (v%d)%s",
         (int)sizeof(TaaControl), TAA_CONTROL_NAME, TAA_CONTROL_VERSION,
         existed ? " - panel was already running" : "");
}

// ------------------------------------------- X-Plane's own render resolution
//
// The only way to make the sim render its 3D scene below native. Every render
// target, every depth buffer and the upscaler's storage images are sized by
// this, so it is the one control that moves both frame time and several hundred
// megabytes - and nothing in a Vulkan layer can reach it, because those sizes
// are X-Plane's decision and the passes writing them carry matching viewports
// and shader constants.
//
// quality < 0 turns it off and renders at native. 0..4 select the sim's own
// FSR presets; what scale each one produces is not documented, so the layer
// logs the resulting scene target size and we read the mapping off that rather
// than assuming it.
//
// All four controls are floats even though three of them are flags - that is
// how X-Plane's art controls are typed, and writing an int to one silently does
// nothing.

// Apply a pending request, if there is one. Called once per flight loop.
//
// Only the fields named in `mask` are touched. The panel is not the only thing
// driving these - the in-sim window, the menu commands and any Lua script write
// the same values - so applying the whole block every time would mean the panel
// silently reverted anything else that changed since it last refreshed.
static void pumpControl()
{
    if (!g_ctl || g_ctl->magic != TAA_CONTROL_MAGIC) return;
    if (g_ctl->structSize != sizeof(TaaControl)) {
        static bool said = false;
        if (!said) {
            said = true;
            xlog("control: panel reports a %u-byte block, this plugin has %u - "
                 "IGNORING it. Rebuild the panel from the same source as the "
                 "plugin.", g_ctl->structSize, (uint32_t)sizeof(TaaControl));
        }
        return;
    }

    uint32_t seq = g_ctl->seq;
    if (seq == g_ctlSeen) return;
    g_ctlSeen = seq;

    uint32_t m = g_ctl->mask;
    if (m & TAA_CTL_ENABLED)   g_enabled   = g_ctl->enabled ? 1 : 0;
    if (m & TAA_CTL_LOD_BIAS)  g_lodBias   = g_ctl->lodBias;
    if (m & TAA_CTL_OBJECTS)   g_objectsOn   = (g_ctl->movingObjects != 0);
    if (m & TAA_CTL_TRAFFIC)   g_trafficRadius   = g_ctl->trafficRadius;
    if (m & TAA_CTL_JITTER) {
        int p = g_ctl->jitterPhases;
        // Clamped here as well as in the panel. The panel is one writer of this
        // block and a hand-written one is possible; a phase count of zero would
        // divide by zero in the Halton index, which is a crash rather than a
        // bad-looking image.
        if (p < 1)  p = 1;
        if (p > 64) p = 64;
        g_jitterPhases = p;
    }

    g_ctl->applied = seq;
    xlog("control: applied request %u (mask 0x%02x)",
         seq, m);
}

// Derive near/far and the depth convention straight from the projection matrix,
// independently of the art control, so the two can be cross-checked. A flipped
// reverseZ inverts the shader's sky test and produces a velocity field that is
// wrong but still looks superficially reasonable - exactly the kind of bug that
// costs a day of staring at screenshots.
static void deriveDepthRange(TaaShare *s)
{
    float invProj[16];
    if (!taaInverse(invProj, s->proj)) {
        s->nearClip = s->farClip = 0.0f;
        s->infiniteFar = 0;
        s->reverseZFromMatrix = -1;
        return;
    }

    // Vulkan clip space is z in [0,1]. Whichever end unprojects CLOSER to the
    // eye is the near plane; if that is z=1 the projection is reverse-Z.
    float d0 = 0.0f, d1 = 0.0f;
    bool ok0 = taaDepthPlaneDistance(invProj, 0.0f, &d0);
    bool ok1 = taaDepthPlaneDistance(invProj, 1.0f, &d1);

    if (ok0 && ok1) {
        s->reverseZFromMatrix = (d1 < d0) ? 1 : 0;
        s->nearClip    = (d1 < d0) ? d1 : d0;
        s->farClip     = (d1 < d0) ? d0 : d1;
        s->infiniteFar = 0;
    } else if (ok0 && !ok1) {
        // z=1 at infinity => standard-Z with an infinite far plane.
        s->reverseZFromMatrix = 0;
        s->nearClip = d0; s->farClip = 0.0f; s->infiniteFar = 1;
    } else if (!ok0 && ok1) {
        // z=0 at infinity => reverse-Z with an infinite far plane, which is the
        // usual modern choice and what X-Plane 12 is expected to use.
        s->reverseZFromMatrix = 1;
        s->nearClip = d1; s->farClip = 0.0f; s->infiniteFar = 1;
    } else {
        s->reverseZFromMatrix = -1;
        s->nearClip = s->farClip = 0.0f; s->infiniteFar = 0;
    }
}

// Is a flight actually running, as opposed to the main menu or a loading
// screen? This matters more than it sounds. In the menu the matrix datarefs
// still hold whatever the last pass left in them, so everything downstream -
// the depth-convention detection, the reprojection self-check, the object
// tracking - would happily report confident numbers derived from a stale or
// orthographic projection. Worse, the first "camera position" on entering a
// flight would be measured against a menu camera and read as a 4000 km
// teleport.
//
// There is no single "in menu" dataref, so this tests the things that actually
// have to be true for the velocity pass to mean anything, rather than trying to
// name the sim's internal state:
//
//   * the 3D projection is a real perspective matrix, not zeros and not the
//     orthographic one the 2D pass leaves behind
//   * the viewport has a size
//   * flight time is running
//
// Any of those failing means the frame is not one we can derive motion for.
// Drives the camera through the scripted phases. Returns 1 while it wants the
// camera and 0 to hand it back.
//
// XPLMControlCamera is used with xplm_ControlCameraUntilViewChanges, so any
// deliberate view change by the user ends the test immediately rather than
// fighting them for the camera.
static int taaSelfTestCamera(XPLMCameraPosition_t *pos, int losingControl, void *)
{
    if (losingControl || g_stPhase == TAA_ST_DONE) {
        // The transition INTO the external phase is our own doing: the flight
        // loop issues sim/view/circle while this callback is still installed,
        // so the resulting losingControl is the command landing, not the user
        // taking over. Ending the test here is what made everything after
        // EXTERNAL unreachable. Hand the camera over and keep the phase clock
        // running; EXT_ORBIT re-acquires control explicitly.
        if (losingControl && g_stPhase == TAA_ST_EXTERNAL)
            return 0;
        g_stActive   = false;
        g_stHaveBase = false;
        g_stPhase    = TAA_ST_DONE;
        return 0;
    }
    if (!pos) return 1;

    // Refuse to drive the camera without a captured starting pose.
    //
    // The base used to be captured HERE, gated on being the first frame of the
    // first phase. That branch never ran: the flight loop increments the frame
    // counter and can do so before the first camera callback, so the condition
    // was already false by the time this was reached. The base stayed at
    // (0,0,0) and the camera was flown to the local origin - sea level at the
    // scenery reference point - which put the view underground with the terrain
    // smeared across the screen, rotating.
    //
    // The pose is now read before control is taken, which is the only ordering
    // that can work: once we hold the camera, what we would be reading back is
    // our own output.
    if (!g_stHaveBase) return 0;

    pos->x = g_stBaseX; pos->y = g_stBaseY; pos->z = g_stBaseZ;
    pos->pitch   = g_stBasePitch;
    pos->heading = g_stBaseHdg;
    pos->roll    = 0.0f;
    pos->zoom    = 1.0f;

    switch (g_stPhase) {
    case TAA_ST_SETTLE:
    case TAA_ST_HOLD:
        break;                                  // perfectly still

    case TAA_ST_YAW:
    case TAA_ST_YAWL:
        // Rotation about the eye point only. No translation, so parallax is
        // exactly zero and the angular shift is the same at every depth.
        pos->heading = g_stBaseHdg + (g_stPhase == TAA_ST_YAWL ? -1.0f : 1.0f)
                     * kStYawPerFrame * (float)g_stFrame;

        // Record the yaw actually APPLIED between renders, and predict from
        // that rather than from the nominal rate.
        //
        // The rate is per flight-loop frame, but a velocity field is produced
        // per RENDER frame, and the two are not one to one. Predicting from the
        // nominal rate made a correct field look 11-44% wrong, with the error
        // wandering between dumps - which reads exactly like a real bug.
        if (g_stLastHdgValid) {
            float dh = pos->heading - g_stLastHdg;
            while (dh >  180.0f) dh -= 360.0f;
            while (dh < -180.0f) dh += 360.0f;
            g_stAppliedYaw = dh;
        }
        g_stLastHdg      = pos->heading;
        g_stLastHdgValid = true;
        break;

    case TAA_ST_PITCH:
        // Pitch about the eye. Exercises the vertical axis, which nothing else
        // here touches, and which is the axis most likely to carry a sign error.
        pos->pitch = g_stBasePitch + kStYawPerFrame * (float)g_stFrame;
        if (g_stLastPitchValid)
            g_stAppliedPitch = pos->pitch - g_stLastPitch;
        g_stLastPitch      = pos->pitch;
        g_stLastPitchValid = true;
        break;

    case TAA_ST_TRANSLATE: {
        // Straight line along the starting heading, so the flow is radial from
        // a focus of expansion at the centre of the screen.
        float h = g_stBaseHdg * 3.14159265f / 180.0f;
        float d = kStMetresPerFrame * (float)g_stFrame;
        pos->x = g_stBaseX + sinf(h) * d;
        pos->z = g_stBaseZ - cosf(h) * d;

        // ---- LIFT THE CAMERA CLEAR OF THE GROUND FOR THIS PHASE.
        //
        // g_stBaseY is 338.31 m against a field elevation of 337.69 - the
        // camera sits 0.6 m off the dirt, and this phase then slides it forward
        // 0.35 m per frame. So the one phase that tests translation was doing
        // it while skimming the ground, where near-field geometry fills the
        // screen and a single frame moves surfaces most of the way across it.
        //
        // That is not a fair test of the reprojection and it is not a case the
        // field will ever be asked for. TAA_ST_TRANSLATE_AGL lifts the camera
        // so the same translation happens against ordinary scenery distances.
        // If the residual collapses, the failure was the geometry; if it
        // survives, the matrix is wrong and the altitude was never the point.
        static const float agl = getenv("TAA_ST_TRANSLATE_AGL")
                               ? (float)atof(getenv("TAA_ST_TRANSLATE_AGL"))
                               : 150.0f;
        pos->y = g_stBaseY + agl;
        break;
    }

    case TAA_ST_EXTERNAL:
        // Camera released to X-Plane, exactly as in FLY - the point is to
        // measure X-Plane's own external camera, not a scripted imitation of
        // one.
        return 0;

    case TAA_ST_EXT_ORBIT:
    case TAA_ST_EXT_HOLD: {
        // Circle the parked aircraft at a fixed radius, facing it - the
        // ghost-forming motion, made deterministic. HOLD freezes at the orbit's
        // final azimuth rather than snapping anywhere new, so the transition
        // itself adds no motion and anything visible during HOLD is history
        // that PARKED, not history being made.
        //
        // The camera base was captured in the cockpit, so it sits inside the
        // fuselage - as an orbit CENTRE that is exactly right.
        const float azd = kStYawPerFrame * 4.0f *
            (float)(g_stPhase == TAA_ST_EXT_ORBIT ? g_stFrame : kStPhaseFrames);
        const float az  = azd * 3.14159265f / 180.0f;
        const float R   = 12.0f;
        pos->x = g_stBaseX + sinf(az) * R;
        pos->z = g_stBaseZ + cosf(az) * R;
        pos->y = g_stBaseY + 3.0f;
        // Local axes are +X east, +Z south; heading 0 faces north (-Z). The
        // camera at azimuth a must face the centre, which solves to h = -a.
        pos->heading = -azd;
        pos->pitch   = -8.0f;
        break;
    }

    case TAA_ST_FLY:
        // ---- THE CAMERA IS LEFT ALONE HERE, ON PURPOSE.
        //
        // It rides the aircraft, so the reprojection is exercised through the
        // same path a user's frames take rather than through a synthetic
        // override - and any error the override was masking has nowhere to
        // hide. Returning 0 hands the camera back to X-Plane for this phase.
        return 0;

    case TAA_ST_HEADMOVE:
        // Sideways shuffle in the seat: small, and crucially motion of the
        // camera RELATIVE to the airframe, which is the only case where the
        // cockpit is not simply still.
        pos->x = g_stBaseX + 0.15f * sinf((float)g_stFrame * 0.06f);
        pos->y = g_stBaseY + 0.05f * sinf((float)g_stFrame * 0.09f);
        break;
    }
    return 1;
}

static const char *taaSelfTestName(int p)
{
    switch (p) {
    case TAA_ST_SETTLE:    return "SETTLE";
    case TAA_ST_HOLD:      return "HOLD";
    case TAA_ST_YAW:       return "YAW-RIGHT";
    case TAA_ST_YAWL:      return "YAW-LEFT";
    case TAA_ST_PITCH:     return "PITCH";
    case TAA_ST_TRANSLATE: return "TRANSLATE";
    case TAA_ST_HEADMOVE:  return "HEADMOVE";
    case TAA_ST_FLY:       return "FLY";
    case TAA_ST_EXTERNAL:  return "EXTERNAL";
    case TAA_ST_EXT_ORBIT: return "EXT-ORBIT";
    case TAA_ST_EXT_HOLD:  return "EXT-HOLD";
    case TAA_ST_DONE:      return "DONE";
    }
    return "OFF";
}

static bool isInFlight(const TaaShare *s)
{
    // proj[11] is -1 for a perspective projection and 0 for an orthographic
    // one, which is the cheapest way to tell the 3D pass from the UI pass.
    if (fabsf(s->proj[0]) < 1e-6f)  return false;
    if (fabsf(s->proj[11]) < 1e-6f) return false;
    if (s->viewportW <= 0 || s->viewportH <= 0) return false;
    if (g_drFlightTime && XPLMGetDataf(g_drFlightTime) <= 0.0f) return false;
    return true;
}

// ------------------------------------------------------- moving objects
//
// AI aircraft are rigid bodies with known world transforms, so they get real
// motion rather than a blanket "reject history here" mask. The approximation is
// the bounding sphere: we over-mask around the silhouette. That is the right
// way to be wrong - a little aliasing on an aircraft outline is far cheaper
// visually than a smear dragged across the sky behind it.
// NOTE ON SPACE: positions are published CAMERA-RELATIVE (world minus the
// current camera position), because that is the space the shader works in - the
// world-space reprojection matrix cannot be inverted accurately in float32 at
// 52 km from the origin. Object motion is unaffected by the shift, since both
// current and previous positions are offset by the same amount.
static void updateMovingObjects(TaaShare *s, double dt)
{
    s->objectCount = 0;
    if (!g_objectsOn) return;

    for (int slot = 0; slot <= TAA_TRAFFIC_SLOTS; ++slot)
    {
        if (s->objectCount >= TAA_MAX_OBJECTS) break;

        float x, y, z;
        float radius = g_trafficRadius;

        if (slot == 0) {
            if (!g_drOwnX) continue;
            x = (float)XPLMGetDatad(g_drOwnX);
            y = (float)XPLMGetDatad(g_drOwnY);
            z = (float)XPLMGetDatad(g_drOwnZ);
        } else {
            int i = slot - 1;
            if (!g_drTrafficX[i]) continue;
            x = (float)XPLMGetDataf(g_drTrafficX[i]);
            y = (float)XPLMGetDataf(g_drTrafficY[i]);
            z = (float)XPLMGetDataf(g_drTrafficZ[i]);
        }

        SlotTrack &t = g_slots[slot];

        // X-Plane parks unused traffic slots at the origin rather than removing
        // them. Treating that as a real aircraft would stamp a mask at world
        // (0,0,0) - which is a real place you can fly over.
        bool occupied = (slot == 0) ||
                        (fabsf(x) > 1e-3f || fabsf(y) > 1e-3f || fabsf(z) > 1e-3f);

        if (!occupied) {
            if (t.occupied) ++t.generation;   // slot emptied; invalidate history
            t.occupied = false;
            t.hadPrev  = false;
            continue;
        }

        // Detect slot reuse. A despawn/respawn into the same index would
        // otherwise hand us the previous aircraft's position and produce a
        // velocity vector pointing across the world, dragging a smear over the
        // screen for a frame. An implausible single-frame jump is the signal.
        bool reused = false;
        if (t.occupied && t.hadPrev && dt > 1e-4) {
            float dx = x - t.px, dy = y - t.py, dz = z - t.pz;
            float step = sqrtf(dx*dx + dy*dy + dz*dz);
            if (step > (float)(1500.0 * dt) + 500.0f) reused = true;
        }
        if (!t.occupied || reused) {
            ++t.generation;
            t.hadPrev = false;
        }

        // The slot cache keeps WORLD positions, so motion is unaffected by the
        // camera moving; only the published values are shifted.
        float wpx = t.hadPrev ? t.px : x;
        float wpy = t.hadPrev ? t.py : y;
        float wpz = t.hadPrev ? t.pz : z;

        TaaMovingObject &o = s->objects[s->objectCount];
        o.x = x   - s->camX;  o.y = y   - s->camY;  o.z = z   - s->camZ;
        o.px = wpx - s->camX; o.py = wpy - s->camY; o.pz = wpz - s->camZ;
        o.radius = radius;

        float dx = o.x - o.px, dy = o.y - o.py, dz = o.z - o.pz;
        o.speed = (dt > 1e-4) ? (float)(sqrtf(dx*dx + dy*dy + dz*dz) / dt) : 0.0f;

        o.id         = (uint32_t)slot;
        o.generation = t.generation;
        // A missing previous transform means REJECT history here, not "velocity
        // is zero" - the latter is a confident wrong answer that blends in
        // whatever was behind the aircraft last frame.
        o.hasPrev    = t.hadPrev ? 1 : 0;
        o.pad0       = 0;

        ++s->objectCount;

        t.px = x; t.py = y; t.pz = z;
        t.hadPrev  = true;
        t.occupied = true;
    }
}


// Every frame. A handful of dataref reads, one 4x4 inverse and two multiplies -
// nowhere near a measurable cost, and deliberately kept that way. Nothing in
// this plugin is allowed to be a reason the sim runs slower.

// ---- START A FLIGHT WITHOUT ANYONE CLICKING.
//
// Every change to the layer costs a launch, and every launch has needed a human
// to pick an aircraft, pick an airport and wait for the world to load. That is
// the slowest part of this loop by far.
//
// XPLMSetUsersAircraft reloads the aircraft and reinitialises the sim; asking
// for the airport afterwards puts us on the ground somewhere with scenery worth
// looking at. Both are no-ops until the sim is actually running, so this waits
// for a valid frame rather than firing during startup.
static int   g_autoStartDone  = 0;
static float g_autoStartDelay = 0.0f;

// ---- RUN THE SELF TEST ON ITS OWN.
//
// The test takes the camera, yaws it a known number of degrees per frame and
// publishes the pixel displacement that yaw SHOULD produce. Comparing that
// against what the motion vector buffer actually contains is the only way to
// settle the convention by measurement rather than by sweeping signs and
// scales and asking which looked better.
//
// Fires once, well after the flight is up, when TAA_SELFTEST_AUTO is set.
static float g_autoTestDelay = 0.0f;
static int   g_autoTestDone  = 0;

static void autoSelfTestTick(float dt)
{
    static int want = -1;
    if (want < 0) {
        const char *e = getenv("TAA_SELFTEST_AUTO");
        want = (e && atoi(e)) ? 1 : 0;
    }
    if (!want || g_autoTestDone) return;
    g_autoTestDelay += dt;
    if (g_autoTestDelay < 25.0f) return;      // let the world finish loading
    g_autoTestDone = 1;
    g_stRequested = true;
    g_stPhase     = TAA_ST_OFF;
    xlog("self-test: started automatically (TAA_SELFTEST_AUTO)");
}

static void autoStartTick(float dt)
{
    static int want = -1;
    if (want < 0) {
        const char *e = getenv("TAA_AUTOSTART");
        want = (e && atoi(e)) ? 1 : 0;
        if (want) XPLMDebugString("TAAImpl: autostart armed - PA-18 at LOWS\n");
    }
    if (!want || g_autoStartDone) return;

    // Let the sim settle first. Placing the user while the world is still
    // coming up gets ignored, and a silent no-op looks exactly like a bug.
    g_autoStartDelay += dt;
    if (g_autoStartDelay < 10.0f) return;

    g_autoStartDone = 1;
    const char *acf = "Aircraft/Laminar Research/Piper PA-18 Super Cub/PA-18-150.acf";
    XPLMDebugString("TAAImpl: autostart - loading PA-18 Super Cub\n");
    XPLMSetUsersAircraft(acf);
    XPLMDebugString("TAAImpl: autostart - placing at LOWS\n");
    XPLMPlaceUserAtAirport("LOWS");
}

// Registered on the window phase, which keeps ticking on the flight
// configuration screen. The flight-loop callback below does not run there at
// all - that is why the first attempt at this silently did nothing, and why
// --load_acf/--load_apt/--go_to_apt were not enough either: X-Plane parses all
// three (they appear in Log.txt) but they only pre-fill the screen.
static int autoStartDrawCb(XPLMDrawingPhase, int, void *)
{
    autoStartTick(0.02f);
    return 1;
}

static float matrixCallback(float sinceLast, float, int, void *)
{
    // Real elapsed time, not a guessed constant - the delay before placing the
    // aircraft has to be wall-clock or it drifts with frame rate.
    autoStartTick(sinceLast);
    autoSelfTestTick(sinceLast);

    if (!g_refsResolved) {
        g_refsResolved = true;
        g_drProj     = taaFind("sim/graphics/view/projection_matrix");
        g_drProj3d   = taaFind("sim/graphics/view/projection_matrix_3d");
        g_drMv       = taaFind("sim/graphics/view/modelview_matrix");
        g_drWorld    = taaFind("sim/graphics/view/world_matrix");
        g_drVpW      = taaFind("sim/graphics/view/window_width");
        g_drVpH      = taaFind("sim/graphics/view/window_height");
        g_drFov      = taaFind("sim/graphics/view/field_of_view_deg");
        g_drRevZ     = taaFind("sim/private/controls/hdr/use_reverse_z");
        g_drViewType = taaFind("sim/graphics/view/view_type");
        // Whether the camera is outside the aeroplane. The body-frame test is
        // only meaningful when it is not - see the sampling gate.
        g_drViewExternal = taaFind("sim/graphics/view/view_is_external");
        g_drPaused   = taaFind("sim/time/paused");
        g_drSimTime    = taaFind("sim/time/total_running_time_sec");
        g_drFlightTime = taaFind("sim/time/total_flight_time_sec");
        xlog("refs: proj3d=%p mv=%p world=%p revz=%p viewtype=%p",
             (void*)g_drProj3d, (void*)g_drMv, (void*)g_drWorld,
             (void*)g_drRevZ, (void*)g_drViewType);

        g_drOwnX = taaFind("sim/flightmodel/position/local_x");
        g_drOwnY = taaFind("sim/flightmodel/position/local_y");
        g_drOwnZ = taaFind("sim/flightmodel/position/local_z");

        // Aircraft attitude, for the body-frame reprojection that keeps the
        // cockpit still. The quaternion is used rather than the Euler angles
        // because it needs no assumption about rotation order.
        g_drOwnQ = taaFind("sim/flightmodel/position/q");

        g_drPeX = taaFind("sim/aircraft/view/acf_peX");
        g_drPeY = taaFind("sim/aircraft/view/acf_peY");
        g_drPeZ = taaFind("sim/aircraft/view/acf_peZ");
        xlog("body frame: eye-position ground truth %s",
             (g_drPeX && g_drPeY && g_drPeZ)
                 ? "found (acf_peX/Y/Z) - the axis convention will be solved "
                   "against it rather than assumed"
                 : "NOT found - falling back to the rigidity test alone, which "
                   "cannot tell a wrong convention from a right one");

        int trafficFound = 0;
        for (int i = 0; i < TAA_TRAFFIC_SLOTS; ++i) {
            char n[128];
            snprintf(n, sizeof(n), "sim/multiplayer/position/plane%d_x", i + 1);
            g_drTrafficX[i] = taaFind(n);
            snprintf(n, sizeof(n), "sim/multiplayer/position/plane%d_y", i + 1);
            g_drTrafficY[i] = taaFind(n);
            snprintf(n, sizeof(n), "sim/multiplayer/position/plane%d_z", i + 1);
            g_drTrafficZ[i] = taaFind(n);
            if (g_drTrafficX[i] && g_drTrafficY[i] && g_drTrafficZ[i]) ++trafficFound;
            else g_drTrafficX[i] = nullptr;   // require all three or use none
        }

        g_drNumEngines = taaFind("sim/aircraft/engine/acf_num_engines");
        g_drPropRpm    = taaFind("sim/cockpit2/engine/indicators/prop_speed_rpm");
        g_drEngRotRate = taaFind("sim/flightmodel2/engines/engine_rotation_speed_rad_sec");

        xlog("objects: own=%s traffic slots=%d/%d  props: engines=%s rpm=%s",
             g_drOwnX ? "yes" : "NO", trafficFound, TAA_TRAFFIC_SLOTS,
             g_drNumEngines ? "yes" : "NO", g_drPropRpm ? "yes" : "NO");

        memset(g_slots, 0, sizeof(g_slots));
        if (!openShare()) return -1.0f;
        openControl();
    }
    if (!g_share) return -1.0f;

    // Before anything is published, so a change the panel made this frame is
    // reflected in the snapshot the layer reads rather than one frame late.
    pumpControl();

    TaaShare *s = g_share;

    // Roll current into previous before overwriting.
    memcpy(s->prevProj,      s->proj,      sizeof(s->proj));
    memcpy(s->prevModelview, s->modelview, sizeof(s->modelview));
    memcpy(s->prevWorld,     s->world,     sizeof(s->world));

    if (g_drProj3d)    XPLMGetDatavf(g_drProj3d, s->proj, 0, 16);
    else if (g_drProj) XPLMGetDatavf(g_drProj,   s->proj, 0, 16);
    if (g_drMv)    XPLMGetDatavf(g_drMv,    s->modelview, 0, 16);
    if (g_drWorld) XPLMGetDatavf(g_drWorld, s->world,     0, 16);
    if (g_drVpW)   s->viewportW = XPLMGetDatai(g_drVpW);
    if (g_drVpH)   s->viewportH = XPLMGetDatai(g_drVpH);
    if (g_drFov)   s->fovDeg    = XPLMGetDataf(g_drFov);
    if (g_drRevZ)  s->reverseZ  = (XPLMGetDataf(g_drRevZ) != 0.0f) ? 1 : 0;
    if (g_drViewType) s->viewType = XPLMGetDatai(g_drViewType);
    if (g_drPaused)   s->paused   = XPLMGetDatai(g_drPaused);

    double simTime = g_drSimTime ? XPLMGetDataf(g_drSimTime) : 0.0;
    s->simTime = simTime;

    // ---- self-test watchdog. Runs BEFORE the in-flight gate, deliberately.
    //
    // The phase machine lives after that gate, so anything making isInFlight
    // false - a pause, a menu, the map - stops the phases advancing while the
    // camera callback carries on holding the camera. The test then never ends
    // and the view is stuck, with no way back except restarting the sim.
    //
    // A watchdog on total elapsed callbacks cannot be starved the same way,
    // because it is upstream of every early return. Holding a user's camera is
    // the single most intrusive thing this plugin does, so it gets a hard
    // ceiling independent of the state machine that is supposed to end it.
    if (g_stActive) {
        ++g_stWatchdog;
        int budget = kStPhaseFrames * (TAA_ST_DONE + 2);
        if (g_stWatchdog > budget) {
            xlog("self-test: watchdog fired after %d callbacks (phase %s) - "
                 "releasing the camera. The phase machine stalled, most likely "
                 "because the sim left flight while the test was running.",
                 g_stWatchdog, taaSelfTestName(g_stPhase));
            g_stPhase    = TAA_ST_DONE;
            g_stActive   = false;
            g_stHaveBase = false;
            XPLMDontControlCamera();
        }
    } else {
        g_stWatchdog = 0;
    }

    // ---- menu / loading gate.
    //
    // Bail BEFORE touching any history state. Letting a menu frame through
    // would seed the previous-camera, previous-object and previous-time caches
    // with nonsense, so the first real flight frame would measure itself
    // against a menu camera and read as a teleport.
    if (!isInFlight(s)) {
        if (g_wasInFlight) {
            xlog("left flight after %llu frames - suspending",
                 (unsigned long long)g_flightFrames);
            g_wasInFlight = false;
        }

        // Hand the camera straight back. The watchdog above would eventually
        // catch this, but "eventually" is twenty seconds of a view the user
        // cannot move, and leaving flight is a case we can recognise exactly
        // rather than time out on.
        if (g_stActive) {
            xlog("self-test: left flight during phase %s - releasing the camera.",
                 taaSelfTestName(g_stPhase));
            g_stPhase    = TAA_ST_DONE;
            g_stActive   = false;
            g_stHaveBase = false;
            XPLMDontControlCamera();
        }
        // Everything downstream must treat this as "no data", not "stale data".
        s->valid        = 0;
        s->historyReset = 1;
        s->resetReason  = TAA_RESET_STARTUP;
        s->objectCount   = 0;

        // Drop the caches so re-entering a flight starts clean rather than
        // differencing against wherever the menu camera happened to sit.
        g_prevSimTime  = -1.0;
        g_prevFov      = -1.0f;
        g_prevViewType = -1;
        g_prevPaused   = -1;
        g_prevVpW = g_prevVpH = 0;
        g_prevSpeed    = 0.0f;
        g_jitterIndex  = 0;
        memset(g_slots, 0, sizeof(g_slots));
        return -1.0f;
    }

    bool justEntered = false;
    if (!g_wasInFlight) {
        g_wasInFlight  = true;
        justEntered    = true;
        g_flightFrames = 0;
        xlog("entered flight - %dx%d fov=%.1f", s->viewportW, s->viewportH, s->fovDeg);

        // The previous-frame matrices still hold the menu's, because the roll
        // above runs before the gate. Differencing against those would produce
        // a garbage reproj on the very first flight frame. Seed prev from curr
        // instead: reproj comes out as the identity, velocity as zero, which is
        // the honest answer for a frame with no history.
        memcpy(s->prevProj,      s->proj,      sizeof(s->proj));
        memcpy(s->prevModelview, s->modelview, sizeof(s->modelview));
        memcpy(s->prevWorld,     s->world,     sizeof(s->world));

        // Same for the camera, or camDelta reads as a continent-sized teleport.
        s->camX = s->world[12];
        s->camY = s->world[13];
        s->camZ = s->world[14];
    }
    ++g_flightFrames;

    // Once, from a flight loop - never from XPluginEnable, where reading a
    // dataref value is unstable. Same rule as every other read here.
    dumpPagerControls();
    holdArtControls();

    // APPLY taa.ini's upscaler request, once the layer can answer for it.
    //
    // Retried every frame until it succeeds rather than attempted once, because
    // "available" is not a fixed property: DLSS publishes its answer from a
    // worker thread precisely so a stalled NGX cannot hang the sim, so the
    // availability table can still be empty several seconds in. A single early
    // attempt would read "not attached", refuse, and leave the file silently
    // ignored - the failure mode this whole block exists to avoid.
    // ---- X-PLANE'S OWN FSR, SETTLED AT STARTUP RATHER THAN LEFT RUNNING.
    //
    // applyXpFsr existed but was reachable only from the control panel, so in
    // every session so far fsr/enable stayed at 1 and X-Plane spatially upscaled
    // 2953x1661 to the window UNDERNEATH our temporal upscaler. Two upscalers on
    // one image, and the spatial one resamples away the sub-pixel detail the
    // temporal one is trying to accumulate - fsr2_pass.h says exactly this:
    // "a spatial upscale smears a sub-pixel offset across a resample kernel
    // before our resolve sees it, so no jitterOffset can recover it."
    //
    // TAA_XP_FSR: -1 disables it (render native, our FSR2 becomes pure TAA at
    // render==display), 0..4 sets its quality. TAA_XP_FSR_BYPASS=1 keeps the
    // sub-native render but skips X-Plane's own upscale pass, which is the
    // arrangement this project actually wants: X-Plane draws small, we upscale.
    {
        // AS EARLY AS THE DATAREF EXISTS, not at frame 60.
        //
        // Frame 60 was chosen to be "after startup settles" and that is exactly
        // wrong for this control: X-Plane allocates its render targets when the
        // renderer comes up, so writing fsr/enable sixty frames later reports
        // "disabled - rendering at native resolution" while the scene pass stays
        // 2953x1661. The log said the write happened and the layer said the size
        // had not changed, which is the pair of statements that matters.
    }


    // ---- self-test phase machine.
    //
    // Started either by the command or by TAA_SELFTEST=1, and only once the
    // flight has settled - grabbing the camera mid-load would be measuring the
    // loading screen.
    if (g_stRequested && !g_stActive && g_flightFrames > 120) {
        // Read the pose FIRST, then take control. Reversing these reads back
        // our own uninitialised output instead of the user's viewpoint.
        XPLMCameraPosition_t p;
        memset(&p, 0, sizeof(p));
        XPLMReadCameraPosition(&p);

        // A camera at the exact local origin is not a real viewpoint; it is
        // what an uninitialised read looks like. Refuse rather than fly there.
        if (fabsf(p.x) < 1e-3f && fabsf(p.y) < 1e-3f && fabsf(p.z) < 1e-3f) {
            xlog("self-test: camera read came back at the origin - not starting. "
                 "Will retry.");
        } else {
            g_stRequested = false;
            g_stActive    = true;
            g_stHaveBase  = true;
            g_stPhase     = TAA_ST_SETTLE;
            g_stFrame     = 0;
            g_stBaseX = p.x; g_stBaseY = p.y; g_stBaseZ = p.z;
            g_stBaseHdg = p.heading; g_stBasePitch = p.pitch;

            XPLMControlCamera(xplm_ControlCameraUntilViewChanges,
                              taaSelfTestCamera, nullptr);

            // NOT PAUSED, AND NOT BRAKED.
            //
            // Both were added to stop the aeroplane creeping under the camera,
            // on a diagnosis that was wrong twice: the parking brake was
            // already 1.00, and the "2 to 4 mm a frame of drift" turned out to
            // be float32 quantising a camera position recovered from a matrix
            // 33,870 m from the origin. That is fixed at the source now, so
            // neither is doing anything except sitting on the user's sim.
            //
            // Pausing is also actively suspect. This test drives the camera
            // through a CAMERA CALLBACK, which runs per rendered frame whether
            // or not the sim is paused, while the plugin reads world_matrix in
            // the FLIGHT LOOP. If a pause changes the rate or the timing of
            // those view-matrix updates, the field shows the real motion of the
            // scripted camera while the matrix describes something else - which
            // is the disagreement being chased.

            for (int i = 0; i < 2; ++i) {
                const char *name = i ? "sim/graphics/view/handheld_external_cam"
                                     : "sim/graphics/view/gloaded_internal_cam";
                if (XPLMDataRef d = XPLMFindDataRef(name)) {
                    g_stShakeSaved[i] = XPLMGetDatai(d);
                    if (g_stShakeSaved[i]) {
                        XPLMSetDatai(d, 0);
                        xlog("self-test: disabled %s (was %d) - it moves the eye "
                             "a couple of mm a frame, which is 431 px at a 1.6 cm "
                             "near plane", name, g_stShakeSaved[i]);
                    }
                } else {
                    g_stShakeSaved[i] = 0;
                }
            }

            xlog("self-test: taking the camera from (%.1f %.1f %.1f) hdg=%.1f. "
                 "Phases: SETTLE, HOLD, YAW, TRANSLATE, HEADMOVE - %d frames each.",
                 p.x, p.y, p.z, p.heading, kStPhaseFrames);
            xlog("self-test: do not touch the controls; changing view ends it.");
            // Say the rate OUT LOUD. A run made through a switch that never
            // arrived returns numbers identical to a run without it, and the
            // self-test is deterministic enough that identical output is
            // exactly what an ineffective switch looks like. This has now cost
            // two runs.
            xlog("self-test: rate %.4f deg/frame, %d frames per phase",
                 kStYawPerFrame, kStPhaseFrames);
        }
    }

    if (g_stActive && g_stPhase != TAA_ST_DONE) {
        ++g_stFrame;
        if (g_stFrame >= kStPhaseFrames) {
            g_stFrame = 0;
            ++g_stPhase;
            if (g_stPhase == TAA_ST_DONE) {
                xlog("self-test: complete - camera released.");
                // The FLY phase is the last one, so completion is where its
                // controls have to be put back. Doing it only on a phase change
                // would leave the throttle open forever.
                if (XPLMDataRef d = XPLMFindDataRef("sim/flightmodel/controls/parkbrake"))
                    XPLMSetDataf(d, 1.0f);
                if (XPLMDataRef d = XPLMFindDataRef("sim/cockpit2/engine/actuators/throttle_ratio_all"))
                    XPLMSetDataf(d, 0.0f);
                XPLMDontControlCamera();
                for (int i = 0; i < 2; ++i) {
                    if (!g_stShakeSaved[i]) continue;
                    const char *name = i ? "sim/graphics/view/handheld_external_cam"
                                         : "sim/graphics/view/gloaded_internal_cam";
                    if (XPLMDataRef d = XPLMFindDataRef(name))
                        XPLMSetDatai(d, g_stShakeSaved[i]);
                    g_stShakeSaved[i] = 0;
                }
                g_stActive   = false;
                g_stHaveBase = false;
            } else {
                // ---- DRIVE THE AEROPLANE FOR THE FLY PHASE.
                //
                // Release the parking brake and open the throttle, then put
                // both back exactly as they were. The suite has run its whole
                // life against paused=0, groundspeed=0.00, throttle=0.00,
                // parkbrake=1.00 - so every "verified" figure covered rotation
                // and a camera slide, and real flight, which translates metres
                // per frame, was never tested at all.
                //
                // Restoring is not politeness. Leaving a user's brake off and
                // the throttle open after a diagnostic is how a sim ends up
                // rolling into something while nobody is looking.
                static float savedBrake = -1.0f, savedThrottle = -1.0f;
                XPLMDataRef drBrake = XPLMFindDataRef("sim/flightmodel/controls/parkbrake");
                XPLMDataRef drThr   = XPLMFindDataRef("sim/cockpit2/engine/actuators/throttle_ratio_all");
                // The external phase asks the sim to circle the aircraft, and
                // hands the view back afterwards. Commands rather than datarefs:
                // the view type dataref is read-only in X-Plane 12.
                const bool nowExternal =
                    g_stPhase == TAA_ST_EXTERNAL ||
                    g_stPhase == TAA_ST_EXT_ORBIT ||
                    g_stPhase == TAA_ST_EXT_HOLD;
                if (g_stPhase == TAA_ST_EXTERNAL) {
                    if (XPLMCommandRef c = XPLMFindCommand("sim/view/circle"))
                        XPLMCommandOnce(c);
                    xlog("self-test: phase EXTERNAL - circling view, aircraft "
                         "parked. This separates the external camera path from "
                         "the aircraft being a moving object.");
                } else if (g_stPhase == TAA_ST_EXT_ORBIT) {
                    // EXTERNAL surrendered the camera by returning 0, which
                    // uninstalls the callback - so scripted motion in the
                    // external view needs it re-acquired. The view TYPE stays
                    // circle; control positions the camera without changing it.
                    XPLMControlCamera(xplm_ControlCameraUntilViewChanges,
                                      taaSelfTestCamera, nullptr);
                    xlog("self-test: phase EXT-ORBIT - scripted orbit of the "
                         "parked aircraft in the external view. This is the "
                         "ghost-forming motion, made deterministic.");
                } else if (g_stPhase == TAA_ST_EXT_HOLD) {
                    xlog("self-test: phase EXT-HOLD - frozen at the orbit's "
                         "final pose. Anything visible now is history that "
                         "PARKED, not history being made.");
                } else if (g_stPhaseWasExternal) {
                    if (XPLMCommandRef c = XPLMFindCommand("sim/view/3d_cockpit_cmnd_look"))
                        XPLMCommandOnce(c);
                }
                g_stPhaseWasExternal = nowExternal;

                if (g_stPhase == TAA_ST_FLY) {
                    if (drBrake && savedBrake < 0.0f) {
                        savedBrake = XPLMGetDataf(drBrake);
                        XPLMSetDataf(drBrake, 0.0f);
                    }
                    if (drThr && savedThrottle < 0.0f) {
                        savedThrottle = XPLMGetDataf(drThr);
                        XPLMSetDataf(drThr, 1.0f);
                    }
                    xlog("self-test: phase FLY - brake released, throttle open. "
                         "This is the only phase in which the AIRCRAFT moves.");
                } else {
                    if (drBrake && savedBrake >= 0.0f) {
                        XPLMSetDataf(drBrake, savedBrake);
                        savedBrake = -1.0f;
                    }
                    if (drThr && savedThrottle >= 0.0f) {
                        XPLMSetDataf(drThr, savedThrottle);
                        savedThrottle = -1.0f;
                    }
                    xlog("self-test: phase %s", taaSelfTestName(g_stPhase));
                }
            }
        }
    }

    s->selfTestPhase = g_stActive ? g_stPhase : 0;

    // What this phase predicts, at the centre of the screen.
    //
    // Under pure yaw the shift is depth-independent - the same for the panel,
    // the terrain and the sky alike - which is what makes it such a strong
    // check: it tests the matrices with the depth buffer factored out. proj[0]
    // maps view-space x/z to ndc, and uv is half of ndc.
    s->selfTestExpectedPx = 0.0f;
    if (s->selfTestPhase == TAA_ST_HOLD) {
        s->selfTestExpectedPx = 0.0f;
    } else if (s->selfTestPhase == TAA_ST_PITCH
            || s->selfTestPhase == TAA_ST_YAW
            || s->selfTestPhase == TAA_ST_YAWL) {
        // Take the rotation from the VIEW MATRICES the shader actually used,
        // not from the angle the script meant to apply.
        //
        // The script's rate is per flight-loop frame while velocity is produced
        // per render frame, and the two are not one to one - so the intended
        // angle and the realised one drift apart, and samples that were exact
        // to 0.1% sat beside one reading 32% low. That is the harness losing
        // sync with itself, and it is indistinguishable from a real defect by
        // anyone reading the log later.
        //
        // world and prevWorld ARE the two frames being reprojected between, so
        // the angle between them is the rotation the field is built on, by
        // construction. The relative rotation is R_curr * R_prev^T, whose trace
        // is the elementwise product of the two rotation blocks, and the angle
        // follows from trace = 1 + 2cos(a).
        double tr = 0.0;
        for (int c = 0; c < 3; ++c)
            for (int r = 0; r < 3; ++r)
                tr += (double)s->world[c*4+r] * (double)s->prevWorld[c*4+r];

        double ca = (tr - 1.0) * 0.5;
        if (ca >  1.0) ca =  1.0;
        if (ca < -1.0) ca = -1.0;
        float a = (float)acos(ca);

        // Stated for the CENTRE of the screen, which is where it is measured.
        // Comparing a centre prediction against a whole-screen mean was an
        // earlier version of this same mistake: the shift is uniform in ANGLE,
        // not in NDC, growing by 1 + (x/f)^2 towards the edges, so the screen
        // mean sits about 20% above centre at this field of view.
        if (s->selfTestPhase == TAA_ST_PITCH)
            s->selfTestExpectedPx = a * s->proj[5] * (float)s->viewportH * 0.5f;
        else
            s->selfTestExpectedPx = a * s->proj[0] * (float)s->viewportW * 0.5f;
    }

    // ---- IS THE AIRCRAFT ACTUALLY MOVING?
    //
    // The velocity field carries a full-width band across the bottom of the
    // screen with vx exactly zero at x = 700, 996 and 3808 and vy growing
    // linearly with y from zero at the horizon. Purely vertical flow whose
    // magnitude goes as 1/depth is what a camera translating VERTICALLY
    // produces; forward translation would be radial and would show vx at those
    // x positions, and it shows none. Meanwhile dC, differenced from the world
    // matrices, reads 0.0000 m.
    //
    // Exactly one of those is wrong, and the aircraft's own local_y settles it
    // without any matrix in the way. If local_y is moving while dC is zero, the
    // world matrices are in a frame that travels with the aircraft and the
    // reprojection is missing a real translation.
    {
        static double lastOwnY = 0.0, lastOwnX = 0.0, lastOwnZ = 0.0;
        static bool   haveOwn = false;
        const double oy = g_drOwnY ? XPLMGetDatad(g_drOwnY) : 0.0;
        const double ox = g_drOwnX ? XPLMGetDatad(g_drOwnX) : 0.0;
        const double oz = g_drOwnZ ? XPLMGetDatad(g_drOwnZ) : 0.0;
        static int every = 0;
        if (haveOwn && (++every % 20) == 0)
            xlog("MV OWN: aircraft moved (%+.4f, %+.4f, %+.4f) m this frame, "
                   "%.4f m total, altitude %.2f m - compare against the camera "
                   "delta the layer prints",
                   ox - lastOwnX, oy - lastOwnY, oz - lastOwnZ,
                   sqrt((ox-lastOwnX)*(ox-lastOwnX) + (oy-lastOwnY)*(oy-lastOwnY)
                      + (oz-lastOwnZ)*(oz-lastOwnZ)), oy);
        // WHY it is not moving, not just that it is not.
        //
        // A stationary aircraft at a constant 337.69 m reads the same whether
        // the sim is paused, the engine is idle with the brakes on, or a freeze
        // is in force - and those need completely different fixes. The whole
        // acceptance suite has been running in this state, so every number it
        // produced covers rotation and near-zero translation only.
        if (haveOwn && (every % 20) == 0) {
            static XPLMDataRef drGS  = taaFind("sim/flightmodel/position/groundspeed");
            static XPLMDataRef drThr = taaFind("sim/cockpit2/engine/actuators/throttle_ratio_all");
            static XPLMDataRef drPbk = taaFind("sim/flightmodel/controls/parkbrake");
            xlog("MV STATE: paused=%d groundspeed=%.2f m/s throttle=%.2f "
                 "parkbrake=%.2f - a frozen world tests no translation at all",
                 g_drPaused ? XPLMGetDatai(g_drPaused) : -1,
                 drGS  ? (double)XPLMGetDataf(drGS)  : -1.0,
                 drThr ? (double)XPLMGetDataf(drThr) : -1.0,
                 drPbk ? (double)XPLMGetDataf(drPbk) : -1.0);
        }
        lastOwnX = ox; lastOwnY = oy; lastOwnZ = oz; haveOwn = true;
    }

    // ---- GO STRAIGHT TO THE VIEW BEING INVESTIGATED.
    //
    // The external phase sits ninth in the self-test, about 1350 frames in, and
    // three runs in a row ended at FLY - one phase short - so the reading that
    // matters was never taken. TAA_FORCE_EXTERNAL asks the sim to circle the
    // aircraft a few seconds after flight starts and leaves it there, which
    // reaches the broken case in about a minute instead of five.
    // ---- A SUSTAINED EXTERNAL TEST, WITHOUT STEERING THE CAMERA.
    //
    // Every external claim made today rested on thin data. The self-test's
    // EXTERNAL phase is ninth and runs kept ending before reaching it, so tables
    // labelled "per view" contained cockpit rows only - and I read external
    // conclusions out of them anyway.
    //
    // This holds the sim in an external view for the whole run: the view is set
    // ONCE at flight start, the camera is never steered (the previous version
    // panned every frame and drove it under the runway), and the aeroplane is
    // put under power so the chase camera moves because the AIRCRAFT does. That
    // is the case a user actually flies, and it produces external frames
    // continuously instead of for a few seconds at the end.
    if (getenv("TAA_EXT_TEST")) {
        static int f = 0;
        ++f;
        if (f == 120) {
            if (XPLMCommandRef c = XPLMFindCommand("sim/view/circle"))
                XPLMCommandOnce(c);
            if (XPLMDataRef d = XPLMFindDataRef("sim/flightmodel/controls/parkbrake"))
                XPLMSetDataf(d, 0.0f);
            if (XPLMDataRef d = XPLMFindDataRef("sim/cockpit2/engine/actuators/throttle_ratio_all"))
                XPLMSetDataf(d, 1.0f);
            xlog("MV EXT TEST: external view set once, brake off, throttle open. "
                 "The camera is never steered - it moves because the aeroplane "
                 "does, which is the case a user actually flies.");
        }
    }

    // ---- THE RESIDUAL TEST IS ONLY VALID ON STATIC GEOMETRY.
    //
    // The epipolar residual is depth-free, which is why it was worth building:
    // it holds under rotation and translation alike, at any distance. But that
    // property comes from the geometry being FIXED IN THE WORLD while only the
    // camera moves. A pixel on something that moves independently has no reason
    // to lie on the epipolar line, and registers a large residual even when its
    // motion vector is perfectly correct.
    //
    // TAA_EXT_TEST points the camera at the aeroplane and opens the throttle.
    // So the pixels it measures are largely MOVING geometry - fuselage, gear,
    // prop disc - and those are near and fast, which is exactly the population
    // the residual tail has been blamed on all along. The cockpit case, which
    // reads 0.000-0.017 px, is geometry rigidly attached to the camera.
    //
    // So the tail may be the test misapplied rather than a defect in the field.
    // This separates the two: external view, aeroplane frozen, camera the only
    // thing that moves. Every pixel is then static world geometry and the
    // residual is valid everywhere. If it collapses, the field was right and
    // the tail was mine.
    //
    // The camera is oscillated rather than driven one way, and only every third
    // frame, because a previous version issued sim/general/right every frame
    // and drove the camera under the runway - measuring millimetre-range
    // geometry through the near plane and calling it a reprojection error.
    if (getenv("TAA_EXT_STATIC")) {
        static int f = 0;
        ++f;
        if (f == 120) {
            if (XPLMCommandRef c = XPLMFindCommand("sim/view/circle"))
                XPLMCommandOnce(c);
            if (XPLMDataRef d = XPLMFindDataRef("sim/flightmodel/controls/parkbrake"))
                XPLMSetDataf(d, 1.0f);
            if (XPLMDataRef d = XPLMFindDataRef("sim/cockpit2/engine/actuators/throttle_ratio_all"))
                XPLMSetDataf(d, 0.0f);
            // ---- PAUSE THE SIM SO ANIMATED GEOMETRY STOPS TOO.
            //
            // Freezing the aeroplane is not enough. prevClip = M*currClip
            // assumes each vertex held the same world position last frame, and
            // X-Plane's water and camera-relative meshes do not: the grid
            // follows the camera, so the same vertex index is a different world
            // point every frame. No camera-only reprojection can be right for
            // that geometry however correct the matrix is.
            //
            // The dominant shader is exactly the terrain/water one - it outputs
            // v_water_height - and the scenery is Seattle. Pausing stops the
            // animation while the camera can still be moved, so if the bad
            // pixels collapse the residual is moving geometry and not a bug.
            if (const char *p = getenv("TAA_MV_PAUSE")) {
                (void)p;
                if (XPLMDataRef d = XPLMFindDataRef("sim/time/paused"))
                    XPLMSetDatai(d, 1);
                xlog("MV EXT STATIC: sim PAUSED - animated geometry frozen too");
            }
            xlog("MV EXT STATIC: external view, parking brake ON, throttle shut. "
                 "The aeroplane is frozen, so every pixel is static world "
                 "geometry and the epipolar residual is valid on all of it.");
        }
        // ---- LIFT THE CAMERA OFF THE GROUND FIRST.
        //
        // In circle view the camera sits about a metre above the airport, so the
        // bottom of the screen is ground almost touching the lens. The frozen
        // run measured 790 px of flow from 0.056 m of translation, which puts
        // that geometry at sx*t/flow_ndc = 1.57*0.056/0.412 = 0.21 m. Every
        // pixel under 16 px/frame read 0.000-0.293 px residual; everything above
        // it blew up to 320 px. The tail is entirely geometry within tens of
        // centimetres.
        //
        // That is the degenerate near-field an earlier commit already flagged,
        // not a case a user flies. Lifting the camera so nothing is within
        // metres decides it: if the residual collapses, the field is correct and
        // the tail was the camera sitting on the runway.
        if (f > 130 && f <= 150 && (f % 2) == 0) {
            if (XPLMCommandRef c = XPLMFindCommand("sim/general/up"))
                XPLMCommandOnce(c);
            if (XPLMCommandRef c = XPLMFindCommand("sim/general/backward"))
                XPLMCommandOnce(c);
        }
        if (f > 150 && (f % 3) == 0) {
            const bool right = ((f / 45) % 2) == 0;
            if (XPLMCommandRef c = XPLMFindCommand(right ? "sim/general/right"
                                                         : "sim/general/left"))
                XPLMCommandOnce(c);
        }
    }

    // ---- REMOVED: the forced external view.
    //
    // It issued sim/general/right every frame to keep the camera orbiting, and
    // it drove the camera straight through the runway. Every measurement taken
    // with it - including the "external, camera moving, relative error 0.99"
    // figures just committed - was made with the camera buried in terrain,
    // where geometry is millimetres away and clipping through the near plane.
    // That is a degenerate case, not a test of the reprojection.
    //
    // It also confounded two earlier runs by switching view mid-session, so
    // measurements straddled a discontinuity.
    //
    // Camera motion in an external view is still the case worth measuring. The
    // self-test's own EXTERNAL phase does it under control, from a known pose,
    // without steering the camera into the ground.

    // ---- THE WHOLE PROJECTION, INCLUDING THE TERMS NEVER LOOKED AT.
    //
    // clipToView is built as diag(1/sx, 1/sy, -1, 1), which ASSUMES a centred
    // projection - proj[8] and proj[9] zero. Those two have never been logged
    // once in this project; only [0], [5], [10], [11] and [14] ever were.
    //
    // If either is non-zero the view-space reconstruction is wrong, because
    // x_clip = sx*x_view + proj[8]*z_view and the inverse used here drops the
    // second term. An off-centre projection is exactly what a view that is not
    // straight ahead would use.
    {
        static int said = 0;
        if ((++said % 60) == 1)
            xlog("MV PROJFULL: view=%d | [0]=%.5f [4]=%.5f [8]=%.5f [12]=%.5f | "
                 "[1]=%.5f [5]=%.5f [9]=%.5f [13]=%.5f | [2]=%.5f [6]=%.5f "
                 "[10]=%.5f [14]=%.5f | [3]=%.5f [7]=%.5f [11]=%.5f [15]=%.5f",
                 s->viewType,
                 s->proj[0], s->proj[4], s->proj[8],  s->proj[12],
                 s->proj[1], s->proj[5], s->proj[9],  s->proj[13],
                 s->proj[2], s->proj[6], s->proj[10], s->proj[14],
                 s->proj[3], s->proj[7], s->proj[11], s->proj[15]);
    }

    // ---- IS THE RECOVERED CAMERA THE CAMERA X-PLANE RENDERED FROM?
    //
    // The reprojection's translation is C_curr - C_prev, and C is recovered from
    // the world matrix as -R^T t. That recovery has never been checked against
    // anything; it has only ever been checked against ITSELF, which cannot fail.
    //
    // It matters now because the residual image shows the error is
    // depth-dependent - distant geometry and the aeroplane correct to under a
    // pixel, the near runway 20 px or worse - which is the signature of a wrong
    // translation, and a one-frame lag test just ruled out stale pairing
    // (external unchanged at 358-665 px, cockpit made worse).
    //
    // XPLMReadCameraPosition is an independent answer from the sim itself. If
    // the two disagree the recovery is the bug; if they agree the translation is
    // right and the fault is downstream of it.
    {
        XPLMCameraPosition_t cp;
        memset(&cp, 0, sizeof(cp));
        XPLMReadCameraPosition(&cp);
        const double t0 = (double)s->world[12], t1 = (double)s->world[13], t2 = (double)s->world[14];
        const double rx = -((double)s->world[0] * t0 + (double)s->world[1] * t1 + (double)s->world[2]  * t2);
        const double ry = -((double)s->world[4] * t0 + (double)s->world[5] * t1 + (double)s->world[6]  * t2);
        const double rz = -((double)s->world[8] * t0 + (double)s->world[9] * t1 + (double)s->world[10] * t2);
        static int every = 0;
        if ((++every % 20) == 0) {
            const double dx = rx - (double)cp.x, dy = ry - (double)cp.y, dz = rz - (double)cp.z;
            xlog("MV CAMCHECK: view=%d | from world_matrix (-R^T t) = (%.3f, "
                 "%.3f, %.3f) | XPLMReadCameraPosition = (%.3f, %.3f, %.3f) | "
                 "difference %.4f m (a non-zero difference means the "
                 "reprojection's translation is built from the wrong camera)",
                 s->viewType, rx, ry, rz, (double)cp.x, (double)cp.y, (double)cp.z,
                 sqrt(dx*dx + dy*dy + dz*dz));
        }
    }

    deriveDepthRange(s);

    // The matrix wins any disagreement with the art control: it is what the GPU
    // actually rasterised with.
    if (s->reverseZFromMatrix >= 0) s->reverseZ = s->reverseZFromMatrix;

    // Camera position, from the INVERSE of the WORLD matrix.
    //
    // Two things here that both had to be measured rather than assumed:
    //
    // 1. It is world_matrix, NOT modelview_matrix. X-Plane renders
    //    camera-relative to preserve float precision at 50 km from the local
    //    origin, so modelview_matrix is rotation-only - its translation column
    //    is exactly zero. world_matrix is the real world -> eye view matrix.
    //    Confirmed from live data: |world row3| = 53646 against an aircraft
    //    53650 m from the origin, which is the signature of t = -R * camPos.
    //
    // 2. It is the INVERSE, not the translation column. For V = [R | t],
    //    t = -R * camPos, so that column is the camera position rotated into
    //    eye space. Reading it directly makes any view rotation look like
    //    translation, scaled by distance from the origin: a 1.7 degree pan
    //    produced 1500 m of apparent movement and tripped the teleport detector
    //    205 times in one short flight.
    // ---- RECOVERED IN DOUBLE, AND WITHOUT A GENERAL INVERSE.
    //
    // The world matrix is RIGID, so its inverse is exactly [R^T | -R^T t] and
    // no 4x4 cofactor expansion is needed. That matters because of what the
    // numbers are: the translation here is about 33,870 m, and float32 carries
    // roughly seven significant digits, so a position recovered in float lands
    // on a grid about 3.4 mm wide.
    //
    // That was showing up as "the camera drifts 2 to 4 mm a frame with the sim
    // paused and the brakes on" - and it was never camera motion, it was the
    // recovery quantising. It did not stay a cosmetic reading either: Tc below
    // is built from this position, so the noise went into worldRel and from
    // there into the REPROJECTION MATRIX the shader is pushed. Three millimetres
    // is about 430 px at a 1.6 cm near plane and about 9 px at one metre, which
    // is the same size as the rotation being measured - and it is exactly the
    // at1m-versus-far disagreement that had been failing frames.
    //
    // Double gives about fifteen digits, so 33,870 m resolves to tens of
    // nanometres and the noise is gone rather than reduced.
    const double t0 = (double)s->world[12];
    const double t1 = (double)s->world[13];
    const double t2 = (double)s->world[14];
    const double cx = -((double)s->world[0]  * t0 + (double)s->world[1]  * t1 + (double)s->world[2]  * t2);
    const double cy = -((double)s->world[4]  * t0 + (double)s->world[5]  * t1 + (double)s->world[6]  * t2);
    const double cz = -((double)s->world[8]  * t0 + (double)s->world[9]  * t1 + (double)s->world[10] * t2);

    // The DELTA is differenced in double as well. Two positions 33 km from the
    // origin differing by a millimetre cannot be subtracted in float at all -
    // the difference is entirely below the last bit of either operand.
    static double prevCx = 0.0, prevCy = 0.0, prevCz = 0.0;
    static bool   havePrevCam = false;
    const double ddx = cx - prevCx, ddy = cy - prevCy, ddz = cz - prevCz;
    s->camDelta = havePrevCam
                ? (float)sqrt(ddx*ddx + ddy*ddy + ddz*ddz) : 0.0f;
    prevCx = cx; prevCy = cy; prevCz = cz;
    havePrevCam = true;

    s->camX = (float)cx;
    s->camY = (float)cy;
    s->camZ = (float)cz;
    float px = s->camX, py = s->camY, pz = s->camZ;
    if (justEntered) { px = s->camX; py = s->camY; pz = s->camZ; s->camDelta = 0.0f; }

    // ---- reprojection matrix: clip(now) -> clip(prev), collapsed CPU-side.
    //
    // The view matrix is world_matrix, NOT modelview_matrix.
    //
    // This is the single most important line in the file to get right, and the
    // obvious choice is wrong. modelview_matrix is rotation-only - X-Plane
    // renders camera-relative for float precision - so proj * modelview maps
    // CAMERA-RELATIVE coordinates to clip. Reprojecting through that captures
    // only rotation: the previous frame's camera-relative space has a different
    // origin, and treating the two as the same silently asserts the camera
    // never translated.
    //
    // The failure would have been subtle and very easy to misread: the sky and
    // distant terrain would reproject correctly (rotation dominates there),
    // while everything near the aircraft would show no parallax at all. That
    // reads as "TAA is smearing the ground" rather than "the view matrix is
    // wrong", and would have been chased in the resolve shader for a long time.
    //
    // Note the algebraic self-check below CANNOT catch this: reproj * currVP ==
    // prevVP holds no matter what those matrices mean. Only the camera/aircraft
    // gap check is sensitive to it.
    //
    // These matrices are UNJITTERED and must stay that way - see the jitter rule
    // in share.h. Jitter is applied later, by the layer, at the viewport.
    // World-space form. Semantically correct, and used ONLY for the CPU-side
    // projection for the semantic self-check - never inverted.
    float currVP[16], prevVP[16];
    taaMul(currVP, s->proj,     s->world);
    taaMul(prevVP, s->prevProj, s->prevWorld);
    memcpy(s->currViewProj, currVP, sizeof(currVP));

    // ---- everything the SHADER uses is CAMERA-RELATIVE, and it has to be.
    //
    // Inverting the world-space viewProj in float32 does not work here. The
    // world matrix carries a ~52 km translation, so the matrix spans entries
    // from 1 to 52000; inverting that and multiplying back lost four
    // significant digits and left a 10-18% residual - large enough to be
    // plainly wrong motion vectors. Measured, not theoretical:
    //
    //     world-space reproj residual:  1.04e-01, 1.77e-01
    //
    // Shifting the origin to the current camera fixes it at the source. Define
    // x' = x - C_curr. Then for any view matrix W = R * T(-C):
    //
    //     W_curr * (x' + C_curr) = R_curr * x'
    //     W_prev * (x' + C_curr) = R_prev * (x' + C_curr - C_prev)
    //
    // so W * T(C_curr) is the camera-relative view matrix for either frame.
    // Every quantity involved is then order-1 or metres-scale, and the inverse
    // is well conditioned.
    //
    // Deriving it as W * T(C_curr) rather than reusing modelview_matrix is
    // deliberate: modelview appears to be exactly the rotation, but "appears
    // to be" is what produced the last two bugs. This construction is exact
    // from a matrix we have already validated semantically.
    // ---- CAMERA-RELATIVE, WITHOUT EVER FORMING THE CAMERA POSITION.
    //
    // The old form built Tc = translate(camX, camY, camZ) and multiplied both
    // view matrices by it. That is correct algebra and numerically hopeless
    // here: camX is a float about 33,870 m from the origin, so it lands on a
    // 3.4 mm grid, and the product then cancels two huge numbers to leave a
    // small one. Storing the position through the float share block quantised
    // it a second time. The residue went straight into the reprojection the
    // shader is pushed - about 430 px at a 1.6 cm near plane - and is what made
    // at1m disagree with far on frames where the camera had not moved.
    //
    // Both products have closed forms for a RIGID matrix, so neither needs the
    // position at all:
    //
    //     world * Tc      = [R_curr | 0]                 - it cancels exactly
    //     prevWorld * Tc  = [R_prev | R_prev * (C_curr - C_prev)]
    //
    // Only the camera DELTA survives, it is millimetres, and a millimetre in
    // float is exact to a picometre. The subtraction that produces it is done
    // in double, from the two rigid inverses, so the cancellation happens once
    // at full precision instead of repeatedly at seven digits.
    float worldRel[16], prevWorldRel[16];
    memcpy(worldRel, s->world, sizeof(worldRel));
    worldRel[12] = worldRel[13] = worldRel[14] = 0.0f;

    const double pt0 = (double)s->prevWorld[12];
    const double pt1 = (double)s->prevWorld[13];
    const double pt2 = (double)s->prevWorld[14];
    const double pcx = -((double)s->prevWorld[0] * pt0 + (double)s->prevWorld[1] * pt1 + (double)s->prevWorld[2]  * pt2);
    const double pcy = -((double)s->prevWorld[4] * pt0 + (double)s->prevWorld[5] * pt1 + (double)s->prevWorld[6]  * pt2);
    const double pcz = -((double)s->prevWorld[8] * pt0 + (double)s->prevWorld[9] * pt1 + (double)s->prevWorld[10] * pt2);

    const double dCx = cx - pcx, dCy = cy - pcy, dCz = cz - pcz;

    memcpy(prevWorldRel, s->prevWorld, sizeof(prevWorldRel));
    for (int i = 0; i < 3; ++i)
        prevWorldRel[12 + i] = (float)((double)s->prevWorld[0 + i] * dCx
                                     + (double)s->prevWorld[4 + i] * dCy
                                     + (double)s->prevWorld[8 + i] * dCz);

    float currVPrel[16], prevVPrel[16], invCurrVPrel[16];
    taaMul(currVPrel, s->proj,     worldRel);
    taaMul(prevVPrel, s->prevProj, prevWorldRel);

    memcpy(s->prevViewProj, prevVPrel, sizeof(prevVPrel));
    if (taaInverse(invCurrVPrel, currVPrel)) {
        taaMul(s->reproj, prevVPrel, invCurrVPrel);
        memcpy(s->invCurrViewProj, invCurrVPrel, sizeof(invCurrVPrel));
        s->reprojValid = 1;
    } else {
        s->reprojValid = 0;
    }

    // ---- body-frame reprojection, for geometry bolted to the own aircraft.
    //
    // Built camera-relative for exactly the reason the world path is: the
    // aircraft's local position and the view matrix's translation are both tens
    // of kilometres, and their combination has to resolve a cockpit panel less
    // than a metre from the eye. Multiplying the two full matrices in float32
    // would subtract 53,650 from 53,646 and keep whatever survived. Taking the
    // difference in double FIRST keeps every quantity small.
    {
        double ownXd = g_drOwnX ? XPLMGetDatad(g_drOwnX) : 0.0;
        double ownYd = g_drOwnY ? XPLMGetDatad(g_drOwnY) : 0.0;
        double ownZd = g_drOwnZ ? XPLMGetDatad(g_drOwnZ) : 0.0;

        // ---- IS THE QUATERNION ACTUALLY BEING READ?
        //
        // This is the single most important thing to establish before any more
        // theories about axis conventions or component order, because a q that
        // never changes produces EXACTLY the symptom being measured: R is
        // constant, camB = -R^T*rel swings as rel rotates in world space, and
        // the drift comes out at |rel| * turn rate - zero when parked, growing
        // with turn rate, and identical for every interpretation. That is the
        // measurement, precisely.
        //
        // The old code was `if (g_drOwnQ) XPLMGetDatavf(...)` with q initialised
        // to identity and no else. A null dataref, a wrong name, or a read that
        // returns fewer than 4 values all leave identity behind and look exactly
        // like a successful read of a level aeroplane.
        float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        int   qGot = 0;
        if (g_drOwnQ) qGot = XPLMGetDatavf(g_drOwnQ, q, 0, 4);
        {
            static bool  said = false;
            static float q0[4] = {0,0,0,0};
            static float qMin[4], qMax[4];
            static int   qFrames = 0;
            if (!said) {
                said = true;
                xlog("body frame: q dataref %s, XPLMGetDatavf returned %d values, "
                     "first read (%.4f %.4f %.4f %.4f), |q|=%.4f",
                     g_drOwnQ ? "RESOLVED" : "*** NULL - q will stay identity and "
                                             "the body frame is meaningless ***",
                     qGot, q[0], q[1], q[2], q[3],
                     sqrtf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]));
                for (int i = 0; i < 4; ++i) { q0[i] = qMin[i] = qMax[i] = q[i]; }
            }
            for (int i = 0; i < 4; ++i) {
                if (q[i] < qMin[i]) qMin[i] = q[i];
                if (q[i] > qMax[i]) qMax[i] = q[i];
            }
            // Report the RANGE, not the value. A single sample cannot show
            // whether it is live; the spread over a few hundred frames can, and
            // a spread of zero across a manoeuvre is the smoking gun.
            if (++qFrames % 600 == 0) {
                float span = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    float d = qMax[i] - qMin[i];
                    if (d > span) span = d;
                }
                xlog("body frame: q over %d frames = (%.4f %.4f %.4f %.4f), "
                     "range per component (%.4f %.4f %.4f %.4f), widest %.5f - %s",
                     qFrames, q[0], q[1], q[2], q[3],
                     qMax[0]-qMin[0], qMax[1]-qMin[1],
                     qMax[2]-qMin[2], qMax[3]-qMin[3], span,
                     span < 1e-5f
                       ? "*** FROZEN: the attitude is not updating, so every body "
                         "rotation is the same matrix and all drift candidates are "
                         "measuring the aircraft turning under a fixed frame ***"
                       : "live");
            }
        }

        // Aircraft origin relative to the CURRENT camera - the same origin the
        // world path's Tc uses, so the two live in one space.
        float relX = (float)(ownXd - (double)s->camX);
        float relY = (float)(ownYd - (double)s->camY);
        float relZ = (float)(ownZd - (double)s->camZ);

        // Resolve the quaternion handedness by measuring both.
        //
        // The camera's position in body coordinates is a physical constant
        // while the view is fixed in the cockpit, no matter how the aeroplane
        // manoeuvres. Under the wrong handedness every roll and turn swings it
        // around. So compute it both ways, accumulate how much each moves, and
        // after a few hundred frames keep the one that stayed still. Head
        // movement perturbs both equally and does not bias the comparison.
        // ---- FOUR CANDIDATES, NOT TWO: ELEMENT ORDER AS WELL AS HANDEDNESS.
        //
        // Testing only direct-vs-conjugate assumed the array is [w x y z] and
        // asked only which way round the rotation goes. The measurements say
        // that assumption is wrong. Over 960 frames of real manoeuvring the two
        // candidates drifted 0.22 and 0.18 m PER FRAME - both enormous, and
        // only 1.2x apart. Parked, the winner drifted 0.0044 m/frame.
        //
        // A constant axis convention cannot produce that. If camB is constant
        // then C^T*camB is constant for any fixed C, so constancy survives any
        // convention error - which is also why the acf_pe search above was
        // pointless. Neither candidate holding still under rotation means the
        // rotation is wrong FRAME BY FRAME, and the usual cause is reading the
        // quaternion's components in the wrong order: X-Plane's array is taken
        // as [w x y z] here, and [x y z w] is the other common layout. A
        // misordered quaternion is still unit-length and still produces a valid
        // rotation matrix, so nothing downstream complains - it just rotates to
        // the wrong place, every frame, which is exactly a cockpit that shakes
        // only while the aeroplane manoeuvres.
        //
        // So enumerate both orders against both handednesses and let the drift
        // metric choose. It needs no ground truth: the right answer is the one
        // that holds the camera still, and that is a physical fact about a seat
        // bolted to an airframe rather than a convention anyone has to look up.
        // ---- SEARCH EVERY COMPONENT MAPPING, BECAUSE NOTHING ELSE CAN MATTER.
        //
        // Two candidate theories are now dead by measurement. The attitude is
        // not frozen - q resolves, returns 4 values, is unit length and ranges
        // 0.198 across a manoeuvre. And it is not a one-frame pairing lag: the
        // lagged candidates came back identical to the unlagged ones to four
        // decimals (0.1173 vs 0.1175), where a real lag would have collapsed one
        // of them to nearly zero.
        //
        // What remains is bounded by an invariance worth stating, because it
        // rules out most of what was being guessed at:
        //
        //     camB = -(R * P_in)^T * rel = -P_in^T * (R^T * rel)
        //
        // A constant signed permutation on either side merely rotates a constant
        // vector, so if R^T*rel is constant it stays constant. DRIFT IS BLIND TO
        // EVERY AXIS CONVENTION, incoming or outgoing. All the convention
        // hunting earlier in this session could never have moved this number.
        //
        // The only thing that can is how the four array elements map onto
        // (w,x,y,z) - that changes the rotation itself rather than the frame it
        // is expressed in. Exactly two of the twenty-four orderings had been
        // tried. So try all of them, with independent sign flips on x, y and z
        // (w's sign is free: q and -q are the same rotation).
        //
        // 24 * 8 = 192 candidates, a few dozen flops each. If none of them holds
        // the camera still, the fault is not in the quaternion at all - it means
        // the camera genuinely is not rigid in the airframe for this view, and
        // the body-frame approach does not apply here. That is a real answer too,
        // and cheaper to obtain than another round of guessing.
        enum { BODY_CANDS = 192 };
        static const int kQPerm[24][4] = {
            {0,1,2,3},{0,1,3,2},{0,2,1,3},{0,2,3,1},{0,3,1,2},{0,3,2,1},
            {1,0,2,3},{1,0,3,2},{1,2,0,3},{1,2,3,0},{1,3,0,2},{1,3,2,0},
            {2,0,1,3},{2,0,3,1},{2,1,0,3},{2,1,3,0},{2,3,0,1},{2,3,1,0},
            {3,0,1,2},{3,0,2,1},{3,1,0,2},{3,1,2,0},{3,2,0,1},{3,2,1,0}
        };
        static bool   resolved = false, bodyTrusted = false;
        static int    bodyMode = 0;
        static double drift[BODY_CANDS];
        static float  lastCam[BODY_CANDS][3];
        static bool   haveLast = false;
        static int    samples  = 0;
        static bool   candInit = false;
        if (!candInit) {
            candInit = true;
            memset(drift, 0, sizeof(drift));
            memset(lastCam, 0, sizeof(lastCam));
        }

        // Candidate c: ordering c/8, sign bits c%8 applied to x,y,z.
        struct QCand {
            static void build(const float qin[4], int c, float *R)
            {
                const int *pm = kQPerm[c / 8];
                int sb = c % 8;
                float qq[4] = { qin[pm[0]], qin[pm[1]], qin[pm[2]], qin[pm[3]] };
                if (sb & 1) qq[1] = -qq[1];
                if (sb & 2) qq[2] = -qq[2];
                if (sb & 4) qq[3] = -qq[3];
                taaBodyRotation(R, qq, false);
            }
            static void name(int c, char *out, size_t n)
            {
                static const char *kL = "wxyz";
                const int *pm = kQPerm[c / 8];
                int sb = c % 8;
                snprintf(out, n, "%c%c%c%c%s%s%s",
                         kL[pm[0]], kL[pm[1]], kL[pm[2]], kL[pm[3]],
                         (sb&1)?" -x":"", (sb&2)?" -y":"", (sb&4)?" -z":"");
            }
        };

        // ---- WHY THERE IS NO AXIS-CONVENTION SEARCH HERE.
        //
        // The measured body position is (-13.2 19.4 13.8) m, which is 27.2 m
        // from the datum - the right DISTANCE for a 777 flight deck, since
        // distance is rotation-invariant, but a direction no aeroplane has.
        // That looks like a broken axis convention, and a search was built to
        // solve it against acf_peX/Y/Z. Both halves of that were wrong.
        //
        // It could not have worked: acf_pe is measured from the Plane Maker
        // datum and local_x/y/z is the centre of gravity. On an airliner those
        // are tens of metres apart, so the two vectors differ by an unknown
        // constant and NO rotation reconciles them. The search duly reported a
        // 12.52 m best residual and refused - correctly, for the wrong reason.
        //
        // And it was not worth solving. If the true rotation is R*C for a
        // constant convention C, then Bc = [R*C | rel] = [R | rel] * [C|0], so
        // Ac_true = Ac*C_hat and Ap_true = Ap*C_hat, and
        //
        //     bodyReproj = Ap*C_hat * (Ac*C_hat)^-1 = Ap*C_hat*C_hat^-1*Ac^-1
        //                = Ap * Ac^-1
        //
        // THE CONSTANT CANCELS. bodyReproj is invariant to the axis convention,
        // the odd-looking position is cosmetic, and gating the cockpit on a
        // convention solve blocked the fix over a non-problem.
        //
        // Handedness is a different matter and does NOT cancel: conjugate is a
        // transpose, which is a per-frame function of q, not a constant right
        // multiply. So that one still has to be measured - see below.

        // ---- ONLY JUDGE HANDEDNESS WHILE THE AIRCRAFT IS ACTUALLY ROTATING.
        //
        // The drift test compares how much the camera moves in body coordinates
        // under each handedness and keeps the steadier. That is sound, and it
        // was being run on an aeroplane parked on a runway. With no rotation
        // both conventions give nearly the same answer, so it resolved on noise:
        // 1.051 m against 1.142 m, an 8% margin on a decision that is 50/50 and
        // silently wrong half the time.
        //
        // Wrong handedness is exactly a cockpit that shakes when the aircraft
        // manoeuvres and sits still when it does not - which is the report.
        //
        // So samples are only counted when the attitude actually changed, and
        // the winner has to win CLEARLY. Until then the body frame stays
        // unresolved and the cockpit uses the world frame, which is what it did
        // before any of this existed.
        float qAngle = 0.0f;
        {
            static float lastQ[4];
            static bool  haveQ = false;
            if (haveQ) {
                // |dot| of two unit quaternions is cos(half-angle). Fabs folds
                // the double cover, so q and -q read as the same attitude.
                float d = fabsf(q[0]*lastQ[0] + q[1]*lastQ[1] +
                                q[2]*lastQ[2] + q[3]*lastQ[3]);
                if (d > 1.0f) d = 1.0f;
                qAngle = 2.0f * acosf(d);          // radians this frame
            }
            memcpy(lastQ, q, sizeof(lastQ));
            haveQ = true;
        }
        // About 0.06 deg/frame. Below this the two handednesses are
        // indistinguishable and the sample carries no information.
        bool rotating = (qAngle > 0.001f);

        // camBody = R^T * (camPos - ownPos), and camPos - ownPos is -rel.
        float cam[BODY_CANDS][3];
        {
            float Rc[16];
            for (int c = 0; c < BODY_CANDS; ++c) {
                QCand::build(q, c, Rc);
                for (int i = 0; i < 3; ++i)
                    cam[c][i] = -(Rc[i*4+0]*relX + Rc[i*4+1]*relY + Rc[i*4+2]*relZ);
            }
        }

        // Is the camera actually part of the airframe right now?
        //
        // Everything below depends on it. The first version did not check, and
        // resolved the handedness over 240 frames of an external chase view 84 m
        // behind the aircraft - where the camera is not rigid with the body, so
        // neither convention holds still and the comparison is between two
        // meaningless numbers. It settled on one and reported the camera sitting
        // 29 m below the aircraft datum, which is not a seat position.
        //
        // Tested by whether the distance to the aircraft is CONSTANT, not by
        // whether it is small.
        //
        // "Small" was tried and was wrong. local_x/y/z is the centre of
        // gravity, and on an airliner the flight deck is tens of metres ahead
        // of it - a 777 cockpit measured a rock-steady 29.5 m out, so a 15 m
        // threshold rejected the exact case the body frame exists to serve.
        // The distance to the datum is worthless as an absolute; its constancy
        // is precisely the rigidity we need, and it is convention-free - it
        // does not depend on the quaternion being interpreted correctly, which
        // matters because it is what decides whether the quaternion gets
        // interpreted at all.
        float gap = sqrtf(relX*relX + relY*relY + relZ*relZ);
        s->camGap = gap;

        static float lastGap = -1.0f;
        bool rigid = (lastGap >= 0.0f) && (fabsf(gap - lastGap) < 0.05f);
        lastGap = gap;

        // s->historyReset is not written until later in this function, so it
        // cannot be used to gate this. Guard on the measurement itself instead:
        // a metres-per-frame jump in body coordinates means a teleport, a view
        // change or a reload, and no candidate should be charged for it.
        // ---- EXTERNAL VIEWS MUST BE EXCLUDED, NOT MERELY SURVIVED.
        //
        // The rigidity test checks that |rel| is CONSTANT, which an orbit or
        // chase camera passes trivially - it holds a fixed distance while
        // sweeping around the aircraft, so the camera is nowhere near rigid in
        // the airframe and every candidate drifts. A run was contaminated
        // exactly this way: |rel| jumped from 27.2 m in the cockpit to 93.7 m
        // in an orbit view, and although only a few seconds were external, an
        // orbiting camera moves so much per frame in body coordinates that it
        // can dominate a running average of a thousand cockpit frames.
        //
        // So ask the sim directly rather than inferring it from geometry, and
        // THROW AWAY the accumulation on any view change. Averaging across a
        // view switch mixes two different experiments and the result describes
        // neither.
        bool external = g_drViewExternal ? (XPLMGetDatai(g_drViewExternal) != 0) : false;
        {
            static int lastView = -1;
            if (!resolved && lastView >= 0 && s->viewType != lastView && samples > 0) {
                xlog("body frame: view changed (%d -> %d) after %d samples - "
                     "discarding them; a body-frame average across a view switch "
                     "describes neither view",
                     lastView, s->viewType, samples);
                memset(drift, 0, sizeof(drift));
                samples  = 0;
                haveLast = false;
            }
            lastView = s->viewType;
        }

        if (!resolved && haveLast && rigid && rotating && !external) {
            double d[BODY_CANDS];
            bool   sane = true;
            for (int c = 0; c < BODY_CANDS; ++c) {
                double q2 = 0.0;
                for (int i = 0; i < 3; ++i)
                    q2 += (cam[c][i]-lastCam[c][i]) * (double)(cam[c][i]-lastCam[c][i]);
                d[c] = sqrt(q2);
                if (d[c] >= 5.0) sane = false;
            }
            if (sane) {
                for (int c = 0; c < BODY_CANDS; ++c) drift[c] += d[c];
                ++samples;

                int best = 0;
                for (int c = 1; c < BODY_CANDS; ++c)
                    if (drift[c] < drift[best]) best = c;

                // ---- TIES ARE EXPECTED, AND THE OLD TEST CHOKED ON THEM.
                //
                // Several component mappings describe the SAME rotation - the
                // quaternion double cover plus sign flips that cancel - so the
                // correct answer arrives as a group of exactly equal scores. The
                // margin test compared best against second-best, which in that
                // case is an identical number: the margin is 1.0 and can never
                // reach 3x. The search found the answer at 0.0021 m/frame
                // against 0.0444 for the first real alternative, a 21x
                // separation, and then reported itself unresolved forever.
                //
                // Compare against the best DISTINCT score instead, and require
                // the winner to be rigid in absolute terms as well - a landslide
                // among bad options is still bad options.
                int second = -1;
                for (int c = 0; c < BODY_CANDS; ++c) {
                    if (drift[c] <= drift[best] * 1.05) continue;   // same answer
                    if (second < 0 || drift[c] < drift[second]) second = c;
                }

                double perFrameBest = drift[best] / (double)samples;
                bool separated = (second < 0) ||
                                 (drift[best] > 1e-9
                                    ? drift[second] > drift[best] * 3.0
                                    : true);
                bool decisive = separated && perFrameBest < 0.01;

                if (samples >= 120 && decisive) {
                    resolved = true;
                    bodyMode = best;
                    char nb[32], ns[32];
                    QCand::name(best, nb, sizeof(nb));
                    if (second >= 0) QCand::name(second, ns, sizeof(ns));
                    else             snprintf(ns, sizeof(ns), "(none distinct)");
                    xlog("body frame: component mapping resolved by measurement - "
                         "[%s] at %.4f m/frame vs the best DIFFERENT mapping "
                         "[%s] at %.4f, over %d rotating cockpit frames (%.1fx). "
                         "The code had assumed [wxyz]; if that is not this, the "
                         "quaternion was being read in the wrong order.",
                         nb, drift[best]/samples, ns,
                         second >= 0 ? drift[second]/samples : 0.0, samples,
                         (second >= 0 && drift[best] > 1e-9)
                             ? drift[second] / drift[best] : 999.0);

                    double perFrame = drift[best] / (double)samples;
                    bodyTrusted = (perFrame < 0.02);
                    xlog("body frame: camera at (%.2f %.2f %.2f) m in body "
                         "coords, residual %.4f m/frame - %s",
                         cam[best][0], cam[best][1], cam[best][2], perFrame,
                         bodyTrusted ? "rigid, body frame trusted"
                                     : "NOT rigid: body frame is suspect, cockpit "
                                       "will fall back to the world frame");
                }
            }
        }
        for (int c = 0; c < BODY_CANDS; ++c)
            for (int i = 0; i < 3; ++i) lastCam[c][i] = cam[c][i];
        haveLast = true;

        float Rsel[16];
        QCand::build(q, bodyMode, Rsel);
        const float *R = Rsel;
        const float *camB = cam[bodyMode];
        s->camBodyDrift = 0.0f;
        {
            static float prevCamB[3] = {0,0,0};
            static bool  havePrev = false;
            if (havePrev) {
                float dx = camB[0]-prevCamB[0], dy = camB[1]-prevCamB[1], dz = camB[2]-prevCamB[2];
                s->camBodyDrift = sqrtf(dx*dx + dy*dy + dz*dz);
            }
            for (int i = 0; i < 3; ++i) prevCamB[i] = camB[i];
            havePrev = true;
        }

        // body -> camera-relative world, this frame and last.
        float Bc[16], Bp[16];
        memcpy(Bc, R, sizeof(Bc));
        Bc[12] = relX; Bc[13] = relY; Bc[14] = relZ;

        // The PREVIOUS aircraft position is kept in absolute local coordinates
        // and differenced against the CURRENT camera here, not stored
        // pre-differenced. worldRel and prevWorldRel are both built around the
        // current camera position, so the body matrices must share that origin;
        // reusing last frame's offset would silently mix two origins one frame
        // of camera travel apart and add that travel to every cockpit pixel.
        static double prevOwn[3] = {0,0,0};
        static float  prevR[16];
        static bool   havePrevBody = false;

        memcpy(Bp, havePrevBody ? prevR : R, sizeof(Bp));
        Bp[12] = havePrevBody ? (float)(prevOwn[0] - (double)s->camX) : relX;
        Bp[13] = havePrevBody ? (float)(prevOwn[1] - (double)s->camY) : relY;
        Bp[14] = havePrevBody ? (float)(prevOwn[2] - (double)s->camZ) : relZ;

        // bodyReproj = (prevProj * prevWorldRel * Bp) * (proj * worldRel * Bc)^-1
        float Mc[16], Mp[16], Ac[16], Ap[16], invAc[16];
        taaMul(Mc, worldRel,     Bc);
        taaMul(Mp, prevWorldRel, Bp);
        taaMul(Ac, s->proj,     Mc);
        taaMul(Ap, s->prevProj, Mp);

        // The body path applies only while the camera really is bolted to the
        // airframe. In an external view the camera follows its own damped path,
        // so body-frame reprojection would be wrong there - and wrong in the
        // near field, where errors are largest. Better to fall back to the world
        // frame, which is at least correct for everything that is not the
        // aeroplane.
        if (resolved && bodyTrusted && rigid && taaInverse(invAc, Ac)) {
            taaMul(s->bodyReproj, Ap, invAc);
            s->bodyReprojValid = 1;
        } else {
            s->bodyReprojValid = 0;
        }

        // Report state changes, and report PROGRESS while unresolved. Without
        // the second half "off" is indistinguishable from "hung", and the
        // handedness test now needs the aeroplane to manoeuvre before it can
        // finish - so a run that never turns will sit at off forever and the
        // only honest thing to do is say how many rotating frames it has.
        {
            static int   said = -2;
            static int   lastSaid = -1;
            int now = s->bodyReprojValid ? 1 : 0;
            if (said != now) {
                said = now;
                xlog("body frame: cockpit reprojection %s (resolved=%d trusted=%d "
                     "rigid=%d)", s->bodyReprojValid ? "ACTIVE" : "off",
                     resolved ? 1 : 0, bodyTrusted ? 1 : 0, rigid ? 1 : 0);
            }
            if (!resolved && samples / 30 != lastSaid) {
                lastSaid = samples / 30;
                // Rank and print the best few. 192 lines would be unreadable
                // and the tail is uninformative - what matters is whether the
                // leader is separated from the pack, and by how much.
                int order[8];
                for (int k = 0; k < 8; ++k) {
                    int best = -1;
                    for (int c = 0; c < BODY_CANDS; ++c) {
                        bool taken = false;
                        for (int j = 0; j < k; ++j) if (order[j] == c) taken = true;
                        if (taken) continue;
                        if (best < 0 || drift[c] < drift[best]) best = c;
                    }
                    order[k] = best;
                }
                char nm[5][32];
                for (int k = 0; k < 5; ++k) QCand::name(order[k], nm[k], sizeof(nm[k]));
                xlog("body frame: %d rotating frames, best of %d component "
                     "mappings: [%s]=%.4f  [%s]=%.4f  [%s]=%.4f  [%s]=%.4f  "
                     "[%s]=%.4f m/frame (|rel|=%.1f m, turn=%.4f rad/frame -> "
                     "a frame of unremoved rotation would be %.4f m) "
                     "[view=%d external=%d]",
                     samples, (int)BODY_CANDS,
                     nm[0], drift[order[0]]/samples, nm[1], drift[order[1]]/samples,
                     nm[2], drift[order[2]]/samples, nm[3], drift[order[3]]/samples,
                     nm[4], drift[order[4]]/samples,
                     gap, qAngle, gap * qAngle, s->viewType, external ? 1 : 0);
                if (drift[order[0]]/samples > 0.02)
                    xlog("body frame: even the best mapping leaves %.4f m/frame. "
                         "Drift is invariant to every axis convention, so if no "
                         "component mapping holds the camera still the camera is "
                         "NOT RIGID in the airframe for this view and the body "
                         "frame does not apply - stop looking at the quaternion.",
                         drift[order[0]]/samples);
            }
        }

        memcpy(prevR, R, sizeof(prevR));
        prevOwn[0] = ownXd; prevOwn[1] = ownYd; prevOwn[2] = ownZd;
        havePrevBody = true;
    }

    // ---- history reset. Each test below is a case where the previous frame no
    // longer shows the same world from a nearby viewpoint, which is the only
    // condition under which reprojection means anything. A false positive costs
    // one aliased frame, a false negative one smeared frame - both cheap, so
    // these thresholds are deliberately loose rather than clever.
    int reason = TAA_RESET_NONE;

    double dt    = (g_prevSimTime >= 0.0) ? (simTime - g_prevSimTime) : 0.0;
    float  speed = (dt > 1e-4) ? (float)(s->camDelta / dt) : 0.0f;

    if (s->frame < 2 || justEntered)                               reason = TAA_RESET_STARTUP;
    else if (!s->reprojValid)                                      reason = TAA_RESET_STARTUP;
    else if (g_prevViewType >= 0 && s->viewType != g_prevViewType) reason = TAA_RESET_VIEWTYPE;
    else if (g_prevVpW && (s->viewportW != g_prevVpW || s->viewportH != g_prevVpH))
                                                                   reason = TAA_RESET_VIEWPORT;
    else if (g_prevFov > 0 && fabsf(s->fovDeg - g_prevFov) > 0.5f) reason = TAA_RESET_FOV;
    else if (g_prevPaused >= 0 && s->paused != g_prevPaused)       reason = TAA_RESET_PAUSE;
    else if (dt < 0.0 || dt > 1.0)                                 reason = TAA_RESET_TIMEJUMP;
    // Teleport: a jump no aircraft could fly. 1500 m/s is past anything in the
    // sim short of the space shuttle, and the second clause catches slower
    // repositions by looking for a step change rather than an absolute speed.
    else if (s->camDelta > 1000.0f ||
             (speed > 1500.0f && speed > 4.0f * g_prevSpeed + 500.0f))
                                                                   reason = TAA_RESET_CAMJUMP;

    s->resetReason  = reason;
    s->historyReset = (reason != TAA_RESET_NONE) ? 1 : 0;

    // ---- jitter. Computed here, applied by the layer at the viewport, and
    // deliberately NOT folded into the matrices above.
    //
    // Halton(2,3) centred on the pixel, so the offset is +/-0.5 px. On a reset
    // the sequence restarts: continuing mid-sequence after a cut would bias the
    // first few accumulated frames toward one corner of the pixel.
    if (s->historyReset) g_jitterIndex = 0;
    int phases = g_jitterPhases < 1 ? 1 : g_jitterPhases;

    // FSR2 DECIDES THE PHASE COUNT WHEN FSR2 IS THE CONSUMER.
    //
    // The jitter VALUES here already match FSR2's own generator exactly - both
    // are halton(index+1, 2/3) - 0.5, so there is nothing to gain by calling
    // ffxFsr2GetJitterOffset instead (and the plugin does not link FSR2 anyway).
    //
    // The COUNT is a different matter. FSR2 computes
    //
    //     jitterPhaseCount = 8 * (displayWidth / renderWidth)^2
    //
    // internally and its accumulation assumes the application used that many
    // phases. A fixed 8 is right only at native resolution, where the ratio is
    // 1 - which is why this has never shown up: the sim has been run at Native.
    // At Performance the ratio is 2, FSR2 expects 32, and feeding it 8 starves
    // its reconstruction of exactly the sample coverage upscaling needs. That
    s->jitterX = taaHalton(g_jitterIndex + 1, 2) - 0.5f;
    s->jitterY = taaHalton(g_jitterIndex + 1, 3) - 0.5f;
    s->jitterIndex  = g_jitterIndex;
    s->jitterPhases = phases;
    g_jitterIndex = (g_jitterIndex + 1) % phases;

    s->lodBias     = g_lodBias;
    s->renderScale = g_renderScale;

    // Only meaningful from inside the cockpit. In an external view the near
    // field is the airframe, which is ordinary rigid geometry that reprojects
    bool cockpitView = (s->viewType == 1026);

    // Backend selection, republished every frame so changes from the window or
    // from a script take effect on the next one.

    updateMovingObjects(s, dt);

    g_prevFov      = s->fovDeg;
    g_prevViewType = s->viewType;
    g_prevPaused   = s->paused;
    g_prevVpW      = s->viewportW;
    g_prevVpH      = s->viewportH;
    g_prevSimTime  = simTime;
    g_prevSpeed    = speed;

    // frame and valid go LAST. The layer reads this without a lock, using the
    // frame counter as a seqlock, so the counter must never advertise data that
    // is still half-written.
    MemoryBarrier();
    ++s->frame;
    s->valid = (s->frame > 1 && s->reprojValid) ? 1 : 0;

    // Dump the full state twice, counted in FLIGHT frames rather than total
    // frames: once as soon as the projection is real, and again a few seconds
    // in once the scenery has settled and the camera is genuinely moving. The
    // second one is the useful one - a static camera makes reproj the identity,
    // which proves nothing.
    if (g_flightFrames == 3 || g_flightFrames == 900) {
        g_loggedFirst = true;
        xlog("---- state dump at flight frame %llu ----", (unsigned long long)g_flightFrames);
        xlog("viewport %dx%d fov=%.1f viewtype=%d", s->viewportW, s->viewportH, s->fovDeg, s->viewType);
        xlog("proj3d row0: %.4f %.4f %.4f %.4f", s->proj[0], s->proj[1], s->proj[2], s->proj[3]);
        xlog("proj col2: %.6f %.6f %.6f %.6f", s->proj[8], s->proj[9], s->proj[10], s->proj[11]);
        xlog("proj col3: %.6f %.6f %.6f %.6f", s->proj[12], s->proj[13], s->proj[14], s->proj[15]);
        xlog("depth: reverseZ(dataref)=%d fromMatrix=%d near=%.4f far=%.1f infiniteFar=%d",
             g_drRevZ ? (XPLMGetDataf(g_drRevZ) != 0.0f ? 1 : 0) : -1,
             s->reverseZFromMatrix, s->nearClip, s->farClip, s->infiniteFar);
        // Cross-check the camera against the aircraft's own position. In a
        // cockpit view these must agree to within a few metres; a large gap
        // means the extraction is wrong again. The first version of this read
        // the modelview translation column directly and was out by tens of
        // kilometres, so this check earns its place.
        if (g_drOwnX) {
            float ax = (float)XPLMGetDatad(g_drOwnX);
            float ay = (float)XPLMGetDatad(g_drOwnY);
            float az = (float)XPLMGetDatad(g_drOwnZ);
            float ddx = s->camX - ax, ddy = s->camY - ay, ddz = s->camZ - az;
            float gap = sqrtf(ddx*ddx + ddy*ddy + ddz*ddz);
            xlog("camera: %.2f %.2f %.2f", s->camX, s->camY, s->camZ);
            xlog("aircraft: %.2f %.2f %.2f   gap=%.2fm %s",
                 ax, ay, az, gap,
                 (s->viewType == 1026 && gap > 100.0f)
                     ? "(SUSPECT - cockpit view should be metres, not this)" : "");
        } else {
            xlog("camera: %.2f %.2f %.2f", s->camX, s->camY, s->camZ);
        }
        xlog("world row3 (NOT the camera): %.2f %.2f %.2f",
             s->world[12], s->world[13], s->world[14]);
        xlog("reproj valid=%d row0: %.5f %.5f %.5f %.5f",
             s->reprojValid, s->reproj[0], s->reproj[4], s->reproj[8], s->reproj[12]);

        // End-to-end check of the chain: reproj * currVP should reproduce
        // prevVP exactly. A wildly wrong matrix shows up here immediately
        // rather than as a mysteriously smeared image later.
        // Algebraic check, run on the CAMERA-RELATIVE matrices - the ones the
        // shader actually receives. Running it on the world-space pair instead
        // measures a matrix nothing consumes, and would have hidden the
        // conditioning problem rather than exposing it.
        float check[16];
        taaMul(check, s->reproj, currVPrel);
        float worst = 0.0f;
        for (int i = 0; i < 16; ++i) {
            float d = fabsf(check[i] - prevVPrel[i]);
            float scale = fabsf(prevVPrel[i]);
            if (scale > 1.0f) d /= scale;
            if (d > worst) worst = d;
        }
        xlog("reproj self-check (camera-relative): worst relative residual %.2e %s",
             worst, worst < 1e-3f ? "(OK)" : "(SUSPECT - maths is wrong)");

        // And the same check on the world-space pair, purely to record how bad
        // it is. This is why the shader is not given these matrices.
        float invWorldVP[16];
        if (taaInverse(invWorldVP, currVP)) {
            float wreproj[16], wcheck[16];
            taaMul(wreproj, prevVP, invWorldVP);
            taaMul(wcheck, wreproj, currVP);
            float wworst = 0.0f;
            for (int i = 0; i < 16; ++i) {
                float d = fabsf(wcheck[i] - prevVP[i]);
                float scale = fabsf(prevVP[i]);
                if (scale > 1.0f) d /= scale;
                if (d > wworst) wworst = d;
            }
            xlog("  (world-space equivalent would be %.2e - %.0fx worse, hence camera-relative)",
                 wworst, worst > 0 ? wworst / worst : 0.0f);
        }

        // SEMANTIC self-check. The residual above is purely algebraic - it
        // confirms reproj * currVP == prevVP, which is true regardless of what
        // those matrices actually represent. It passed at 8.5e-08 while the
        // view matrix was wrong, so on its own it proves nothing about meaning.
        //
        // This one does: place a point 1 km straight ahead of the camera and
        // project it. If proj, the view matrix and the handedness are all
        // right, it lands dead centre. Anything else and the projection chain
        // is wrong in a way that would otherwise only show up as bad-looking
        // TAA much later.
        float invWorld2[16];
        if (taaInverse(invWorld2, s->world)) {
            // Columns 0,1,2 of the inverse view matrix are the camera's right,
            // up and BACK axes in world space (OpenGL convention looks down -Z),
            // so forward is the negated third column.
            float fx = -invWorld2[8], fy = -invWorld2[9], fz = -invWorld2[10];
            float rx =  invWorld2[0], ry =  invWorld2[1], rz =  invWorld2[2];

            float u, v, w;
            bool ok = taaProjectToUv(currVP,
                                     s->camX + fx * 1000.0f,
                                     s->camY + fy * 1000.0f,
                                     s->camZ + fz * 1000.0f, &u, &v, &w);
            float err = ok ? fmaxf(fabsf(u - 0.5f), fabsf(v - 0.5f)) : 999.0f;
            xlog("semantic check: 1km ahead -> uv=(%.4f %.4f) w=%.1f  %s",
                 ok ? u : -1.0f, ok ? v : -1.0f, ok ? w : 0.0f,
                 (ok && err < 0.01f) ? "(OK - centred)"
                                     : "(BROKEN - projection chain is wrong)");

            // And 200 m to the right at the same distance, which must land on
            // the right half. Catches a mirrored or transposed view matrix that
            // the centred test alone would happily pass.
            float u2, v2, w2;
            bool ok2 = taaProjectToUv(currVP,
                                      s->camX + fx * 1000.0f + rx * 200.0f,
                                      s->camY + fy * 1000.0f + ry * 200.0f,
                                      s->camZ + fz * 1000.0f + rz * 200.0f, &u2, &v2, &w2);
            xlog("semantic check: +200m right -> uv=(%.4f %.4f)  %s",
                 ok2 ? u2 : -1.0f, ok2 ? v2 : -1.0f,
                 (ok2 && u2 > 0.5f) ? "(OK - right of centre)"
                                    : "(BROKEN - handedness/mirror)");
        }
        xlog("jitter: Halton(2,3) %d phases, lodBias=%.2f renderScale=%.2f",
             s->jitterPhases, s->lodBias, s->renderScale);
    }

    if (reason != TAA_RESET_NONE) ++g_resetCounts[reason];

    // Object counts change slowly; sampling is enough and keeps
    // the log readable during a long flight.
    if (g_flightFrames % 600 == 0) {
        xlog("flight frame %llu: %d moving objects, "
             "cam=(%.0f %.0f %.0f) moved=%.2fm jitter=(%.3f %.3f)",
             (unsigned long long)g_flightFrames, s->objectCount,
             s->camX, s->camY, s->camZ, s->camDelta, s->jitterX, s->jitterY);

        // ---- FULL STATE DUMP.
        //
        // Everything the layer consumes, in one place, sampled rather than
        // streamed. Most of these have never been printed at all, which is how
        // several of this session's dead ends survived as long as they did: a
        // value nobody looks at is a value nobody can rule out.
        xlog("state: view=%d fov=%.2f near=%.1f far=%.1f infFar=%d revZ=%d/%d "
             "vp=%dx%d meas=%ux%u->%ux%u",
             s->viewType, s->fovDeg, s->nearClip, s->farClip, s->infiniteFar,
             s->reverseZ, s->reverseZFromMatrix, s->viewportW, s->viewportH,
             s->measRenderW, s->measRenderH, s->measDisplayW, s->measDisplayH);
        xlog("state: jitter=(%.4f %.4f) idx=%d/%d  lodBias=%.2f",
             s->jitterX, s->jitterY, s->jitterIndex, s->jitterPhases,
             s->lodBias);
        xlog("state: reprojValid=%d bodyValid=%d camGap=%.3f camBodyDrift=%.4f "
             "reset=%d(%s) paused=%d",
             s->reprojValid, s->bodyReprojValid, s->camGap, s->camBodyDrift,
             s->resetReason, s->historyReset ? "RESET" : "keep", s->paused);
        xlog("state: vram %u used / %u budget / %u total MB, layerAttached=%d",
             s->vramUsageMB, s->vramBudgetMB, s->vramTotalMB, s->layerAttached);

        // The dataref audit, once, after everything has had a chance to
        // resolve. Anything on this list that matters is a silent null.
        static bool auditDone = false;
        if (!auditDone) { auditDone = true; taaLogDrefAudit(); }

        // Reset rate is the headline health metric. Anything above a few per
        // thousand frames means the detector is firing on normal flying, and
        // TAA would be discarding history it should be keeping.
        double per1k = 1000.0 * (double)(g_resetCounts[TAA_RESET_CAMJUMP] +
                                         g_resetCounts[TAA_RESET_TIMEJUMP])
                     / (double)g_flightFrames;
        xlog("  resets: camjump=%llu viewtype=%llu fov=%llu pause=%llu "
             "timejump=%llu viewport=%llu  (%.2f spurious per 1k frames)",
             (unsigned long long)g_resetCounts[TAA_RESET_CAMJUMP],
             (unsigned long long)g_resetCounts[TAA_RESET_VIEWTYPE],
             (unsigned long long)g_resetCounts[TAA_RESET_FOV],
             (unsigned long long)g_resetCounts[TAA_RESET_PAUSE],
             (unsigned long long)g_resetCounts[TAA_RESET_TIMEJUMP],
             (unsigned long long)g_resetCounts[TAA_RESET_VIEWPORT],
             per1k);
    }

    // Reset events are rare and each one explains a visible artefact, so log
    // them all rather than sampling.
    if (reason != TAA_RESET_NONE && g_flightFrames > 3) {
        static const char *kReason[] = {
            "none", "startup", "camera jump", "view type", "fov", "viewport",
            "pause", "time jump"
        };
        xlog("history reset: %s (flight frame %llu, delta=%.2fm, speed=%.0fm/s)",
             kReason[reason], (unsigned long long)g_flightFrames, s->camDelta, speed);
    }

    return -1.0f;   // every frame
}

// ============================================================ in-sim window
//
// Deliberately plain: XPLM's own drawing, no ImGui dependency. Every row maps
// onto one of the datarefs above, so the window and any script are driving the
// same state and cannot disagree.

static XPLMMenuID   g_menu     = nullptr;
static int          g_menuItem = 0;

// Wider and taller than it was. 430x300 was sized for a window with no state
// block; the readouts pushed the controls off the bottom and the longer lines -
// "Depth: standard  (matrix: reverse-Z)", "VRAM: 1784 / 2752 MB of 7772" - ran
// past the right edge and were simply clipped.
#define TAA_WIN_W 620
#define TAA_WIN_H 460

struct Row { int y; int kind; };   // kind: 0 upscaler, 1 quality, 2 optflow, 3 enabled
static Row g_rows[8];
static int g_rowCount = 0;



// Plain functions rather than lambdas: the XPLM callback types are specific
// enough that a lambda's deduced return type does not always convert.
static void winKey(XPLMWindowID, char, XPLMKeyFlags, char, void*, int) {}
static XPLMCursorStatus winCursor(XPLMWindowID, int, int, void*) { return xplm_CursorDefault; }


// Commands, so this can be bound to a key or a hardware button.
static XPLMCommandRef g_cmdSelfTest = nullptr;

static void requestSelfTest(const char *why)
{
    g_stRequested = true;
    g_stPhase     = TAA_ST_OFF;
    XPLMSpeakString("Motion vector self test starting");
    xlog("self-test: requested by %s", why);
}

static int cmdSelfTest(XPLMCommandRef, XPLMCommandPhase ph, void*)
{
    if (ph == xplm_CommandBegin) requestSelfTest("command");
    return 0;
}

static void menuHandler(void*, void *item)
{
    if ((intptr_t)item == 0) requestSelfTest("menu");
}

static void createMenu()
{
    int idx = XPLMAppendMenuItem(XPLMFindPluginsMenu(), "Motion Vectors", nullptr, 0);
    g_menu = XPLMCreateMenu("Motion Vectors", XPLMFindPluginsMenu(), idx, menuHandler, nullptr);
    XPLMAppendMenuItem(g_menu, "Run the velocity self-test", (void*)0, 0);
    g_menuItem = idx;

    // Say so, because "the menu is not there" and "the menu is there and I did
    // not find it" look identical from the outside and need different answers.
    xlog("menu: Plugins > Motion Vectors created (item %d). The command "
         "motionvectors/self_test is bindable if you would rather have a key.", idx);

    g_cmdSelfTest = XPLMCreateCommand("motionvectors/self_test",
                                      "Motion Vectors: run the scripted velocity self-test");
    XPLMRegisterCommandHandler(g_cmdSelfTest, cmdSelfTest, 1, nullptr);

    // Auto-start, so a validation run needs no interaction at all: launch,
    // load any flight, and leave it. Parked on a runway is a perfectly good
    // test scene - terrain, buildings and cockpit are all in frame.
    if (getenv("TAA_SELFTEST")) {
        g_stRequested = true;
        xlog("self-test: armed by TAA_SELFTEST - will start once flight settles");
    }
}

// ------------------------------------------------------------ plugin entry

// Retuning LOD bias and jitter length is the bulk of the tuning work and both
// are pure numbers, so they live in a file rather than needing a rebuild and a
// sim restart for every experiment.
static void loadConfig(const std::string &path)
{
    FILE *f = fopen(path.c_str(), "r");
    if (!f) { xlog("no config at %s - using defaults", path.c_str()); return; }

    char line[512];
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') ++p;
        if (*p == '#' || *p == ';' || *p == '\n' || *p == '\r' || *p == 0) continue;

        char key[128] = {0};
        double val = 0;
        if (sscanf(p, "%127[^= \t] = %lf", key, &val) != 2 &&
            sscanf(p, "%127[^= \t]=%lf",   key, &val) != 2) continue;

        if      (!strcmp(key, "lod_bias"))       g_lodBias      = (float)val;
        else if (!strcmp(key, "render_scale"))   g_renderScale  = (float)val;
        else if (!strcmp(key, "jitter_phases"))  g_jitterPhases = (int)val;
        else if (!strcmp(key, "moving_objects")) g_objectsOn    = (val != 0);
        else if (!strcmp(key, "traffic_radius")) g_trafficRadius= (float)val;
        // The upscaler selection, which this file claimed to carry and did not.
        //
        // The default stays Off, deliberately - something that changes what the
        // sim looks like the moment it is installed should be opted into. But
    }
    fclose(f);

    if (g_jitterPhases < 1)   g_jitterPhases = 1;
    if (g_jitterPhases > 64)  g_jitterPhases = 64;
    // A bias past -1.5 is aliasing traded for sharpness at a bad rate, and on
    // this install it also drives texture working set up into the pager's
    // downscaling threshold. Clamp rather than trust the file.
    if (g_lodBias < -3.0f)    g_lodBias = -3.0f;
    if (g_lodBias >  1.0f)    g_lodBias =  1.0f;

    xlog("config: lodBias=%.2f renderScale=%.2f jitterPhases=%d objects=%d",
         g_lodBias, g_renderScale, g_jitterPhases, g_objectsOn ? 1 : 0);
}

PLUGIN_API int XPluginStart(char *outName, char *outSig, char *outDesc)
{
    strcpy(outName, "TAAImplementation");
    strcpy(outSig,  "com.nitin.taaimplementation");
    strcpy(outDesc, "Publishes camera matrices for the TAA Vulkan layer.");

    char root[1024] = {0};
    XPLMGetSystemPath(root);
    g_configPath = std::string(root) + "Resources/plugins/TAAImplementation/taa.ini";

    xlog("v%s starting", TAA_PLUGIN_VERSION);

    // As early as possible. The budget is read while textures are being loaded,
    // so patching later would leave the first decisions made against the old
    // ceiling - and those are the ones that set the texture scale.
    patchTextureBudget();

    // Read here so values can be changed without a rebuild.
    // ---- HELD BY DEFAULT, not only when a launcher sets the variables.
    //
    // These were read from the environment alone, so an installed build held
    // nothing and X-Plane's pager downscaled textures freely - measured:
    // max_overdrive sat at its default of 16 where the development launcher had
    // been setting 64. The environment still overrides, for sweeping values.
    static const float kDefaults[] = {
        64.0f,      // max_overdrive     - how far the pager may run ahead
        0.75f,      // size_fudge_factor - shrinks its size estimate
        0.0f,       // downscale_cooldown - 0 leaves X-Plane's own value alone
    };
    for (size_t i = 0; i < sizeof(g_held)/sizeof(g_held[0]); ++i) {
        const char *v = getenv(g_held[i].env);
        float want = v ? (float)atof(v)
                       : (i < sizeof(kDefaults)/sizeof(kDefaults[0]) ? kDefaults[i] : 0.0f);
        if (want == 0.0f) continue;
        g_held[i].want = want;
        xlog("hold: armed %s = %g%s", g_held[i].path, g_held[i].want,
             v ? " (from environment)" : " (default)");
    }
    return 1;
}

PLUGIN_API void XPluginStop(void) {}

PLUGIN_API int XPluginEnable(void)
{
    // Config parsing only - no dataref access of any kind here. See the header
    // comment: reading dataref values before the sim has started is unstable.
    loadConfig(g_configPath);

    registerDatarefs();
    createMenu();

    // ---- OPEN IT BY DEFAULT.
    //
    // Everything worth watching while flying is in this window - the upscaler,
    // the quality that now drives render size, and the sim -> presented frame
    // rate that X-Plane's own counter cannot show. Hiding it behind a menu
    // nobody can find makes all of that invisible. TAA_NO_WINDOW=1 suppresses
    // it for anyone who wants the screen clear.

    XPLMRegisterFlightLoopCallback(matrixCallback, -1.0f, nullptr);
    XPLMRegisterDrawCallback(autoStartDrawCb, xplm_Phase_Window, 0, nullptr);
    return 1;
}

PLUGIN_API void XPluginDisable(void)
{
    XPLMUnregisterFlightLoopCallback(matrixCallback, nullptr);
    XPLMUnregisterDrawCallback(autoStartDrawCb, xplm_Phase_Window, 0, nullptr);

    // Give the user their antialiasing settings back. Leaving FXAA and MSAA
    // switched off after the plugin unloads would look like the plugin broke
    // them, and the setting would survive into every later session.
    restoreXPlanePostAA();

    if (g_menu)   { XPLMDestroyMenu(g_menu); g_menu = nullptr; }
    unregisterDatarefs();

    // Mark the block invalid before tearing it down. The layer may still be
    // holding its mapping; letting it read a stale camera forever would be
    // worse than letting it see valid=0 and skip the pass.
    if (g_share) {
        g_share->valid = 0;
        g_share->magic = 0;
        UnmapViewOfFile(g_share);
        g_share = nullptr;
    }
    if (g_shareHandle) { CloseHandle(g_shareHandle); g_shareHandle = nullptr; }

    // The control block's magic is deliberately LEFT INTACT.
    //
    // TaaShare's is zeroed on the way out so the layer stops trusting stale
    // matrices. This one has no such reader: the panel's liveness test is the
    // share block, not this, and if the panel outlives the sim it keeps the
    // mapping alive - so clearing the magic would only mean a panel that
    // reconnects to a restarted sim finds a block it has to reinitialise.
    if (g_ctl)       { UnmapViewOfFile(g_ctl); g_ctl = nullptr; }
    if (g_ctlHandle) { CloseHandle(g_ctlHandle); g_ctlHandle = nullptr; }
}

PLUGIN_API void XPluginReceiveMessage(XPLMPluginID, int, void *) {}
