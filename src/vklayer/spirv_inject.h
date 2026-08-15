// SPIR-V injection: add a previous-frame clip position output to a vertex
// shader, so X-Plane's own shaders emit the motion vectors it never renders.
//
// The transform is the smallest one that can work:
//
//     prevClip = uReproj * gl_Position
//
// where uReproj = prevViewProj * inverse(currViewProj) - one matrix, the same
// for every object in the frame, supplied through a push constant. The plugin
// already computes it and the scripted self-test already verified it: 8.78 px
// measured against 8.77 predicted under pure yaw, 0.000 px with the camera
// still.
//
// WHY OPERATE ON gl_Position RATHER THAN REBUILD THE TRANSFORM. X-Plane's
// vertex shaders index an array of transforms by instance and pre-offset the
// vertex by a camera-relative origin:
//
//     i    = (gl_InstanceIndex - gl_BaseInstance) % u_transform_count
//     clip = mvp_matrix[i] * vec4(a_vertex - u_mv_offset, 1)
//
// Replicating that means replicating index arithmetic that differs between
// shaders, and getting it subtly wrong yields plausible-looking vectors that
// are quietly incorrect - the worst failure available, because nothing reports
// it and the image merely looks slightly off. Taking the value they already
// stored sidesteps all of it, and the same four instructions work on every one
// of the fifteen vertex shaders regardless of how each got there.
//
// It is also EXACT, which the depth path is not: depth reprojection recovers a
// position from a quantised depth value, this uses the position the shader
// itself emitted.
//
// Developed and validated OFFLINE first - all fifteen of X-Plane's vertex
// shaders were dumped, patched and run through spirv-val before a line of it
// ran inside the sim.

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace spvinj {

enum {
    OpName = 5, OpMemberName = 6,
    OpEntryPoint = 15,
    OpTypeBool = 20, OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23,
    OpTypeMatrix = 24, OpTypeStruct = 30, OpTypePointer = 32,
    OpConstant = 43, OpConstantComposite = 44,
    OpFunction = 54, OpVariable = 59, OpLoad = 61, OpStore = 62,
    OpAccessChain = 65,
    OpDecorate = 71, OpMemberDecorate = 72, OpDecorationGroup = 73,
    OpGroupDecorate = 74, OpGroupMemberDecorate = 75,
    OpMatrixTimesVector = 145,
    OpVectorShuffle = 79, OpCompositeConstruct = 80, OpCompositeExtract = 81,
    OpCompositeInsert = 82, OpConvertSToF = 111, OpISub = 130,
    OpFSub = 131, OpFDiv = 136, OpVectorTimesScalar = 142,
    OpFNegate = 127, OpFAdd = 129, OpFMul = 133,
    OpSelect = 169, OpFOrdLessThan = 184,
    OpReturn = 253
};
enum { SC_Input = 1, SC_Output = 3, SC_PushConstant = 9 };
enum { Deco_Block = 2, Deco_ColMajor = 5, Deco_MatrixStride = 7,
       Deco_BuiltIn = 11, Deco_Location = 30, Deco_Offset = 35 };
enum { BuiltIn_Position = 0 };
enum { ExecModel_Vertex = 0, ExecModel_Fragment = 4 };

// Size of the push constant block this adds: one mat4, then one vec4 of jitter.
//
// THE JITTER MOVED IN HERE FROM THE VIEWPORT, and that is the whole point of
// the second member.
//
// Jitter used to be applied by offsetting the viewport origin on any
// full-viewport draw inside a scene pass. For the GEOMETRY that is exactly
// equivalent to offsetting the projection - a viewport translation and a clip
// translation put the triangle on the same pixels. What is NOT equivalent is
// which draws it reaches: a viewport applies to every draw recorded under it,
// so X-Plane's full-screen composite and post-process quads were shifted too.
// A quad that samples its source through gl_FragCoord then reads the texel it
// is displaced onto, so its output moves by the jitter a SECOND time, on top of
// the geometry that already moved once. How many such passes run varies frame
// to frame, so the total displacement varies frame to frame - and a jitter
// whose magnitude the upscaler cannot know is a jitter it cannot cancel. That
// is shake, and no value of jitterOffset handed to FSR2 fixes it, which is
// exactly what the sign sweep found.
//
// Applied in the vertex shader it reaches only what we patched and only what we
// choose to push a non-zero value for, and the jitter FSR2 is told about is
// then the jitter that was actually applied.
//
// vec4 rather than vec2 for the 16-byte alignment; only .xy is read.
static const uint32_t kPushConstantBytes = 80;

// WHERE IT SITS, AND WHY NOT AT ZERO.
//
// Push constant memory is ONE BLOCK per command buffer. A VkPushConstantRange
// declares which STAGES may read which bytes - it does not give each stage its
// own storage. Our vertex range at offset 0 therefore shared bytes 0..63 with
// X-Plane's fragment push constants, which are also at offset 0, and every
// fragment push overwrote the matrix.
//
// Nothing failed. The layout was valid, the push succeeded, and the vertex
// shader multiplied gl_Position by whatever the fragment stage had last
// written. The plugin's matrix was measured as the identity at rest while the
// field reported a third of a screen of motion per frame - the two could only
// be reconciled by the shader reading something else entirely.
//
// Placed at the TOP of the available range for the same reason the varyings
// are: writers allocate upward from 0, so the far end is the least contended.
inline uint32_t &pushConstantOffset() { static uint32_t v = 64; return v; }

// Returns the number of bytes left below our block - how much room X-Plane's
// own push constants have before they start sharing storage with ours. The
// caller logs it, because the block grew when jitter moved in and the margin is
// no longer obviously comfortable on every device.
//
// Vulkan guarantees only 128 bytes. NVIDIA reports 256, which puts us at 176
// and leaves X-Plane the whole lower half. At the 128-byte minimum we would sit
// at 48, and a fragment block reaching past that would collide - silently, the
// same way the offset-0 version did before it was moved up here. Worth a line
// in the log rather than a surprise.
inline uint32_t choosePushOffset(uint32_t maxPushConstantsSize)
{
    // ---- TAA_MV_PCOFFSET overrides. Default is now 0.
    //
    // This used to sit at the TOP of the range - 176 of 256 - reasoning that
    // writers allocate upward from 0 so the far end is least contended. That
    // reasoning does not apply here, because X-Plane declares NO push constant
    // ranges at all on these layouts: measured, logged, one range on the layout
    // and it is ours. The whole space is free and offset 0 is the plainest
    // address there is.
    //
    // It matters because at 176 the matrix arrives at the vertex shader as
    // ZEROS, with everything else verified: spirv-val clean, one push constant
    // block, the entry point listing it, the layout carrying the range, and
    // 4.3 million re-pushes issued immediately before draws. The offset is the
    // last input to that read which has never been varied.
    if (const char *e = getenv("TAA_MV_PCOFFSET")) {
        pushConstantOffset() = (uint32_t)(atoi(e) / 16) * 16;
        return pushConstantOffset();
    }
    (void)maxPushConstantsSize;
    pushConstantOffset() = 0;
    return pushConstantOffset();
}

// THE VARYING LOCATION IS FIXED, NOT PER-SHADER.
//
// The first version picked the lowest free Location in each module, which gave
// 0, 1, 3 and 4 across X-Plane's vertex shaders. That is correct in isolation
// and useless in practice: a vertex output at Location N must be read by a
// fragment input at Location N, and one fragment shader is paired with several
// different vertex shaders. Per-shader locations mean those pairings disagree,
// and the mismatch surfaces at pipeline creation - by which point the two
// modules have long since been patched independently and cannot be reconciled.
//
// 15 is chosen because the highest Location observed in any X-Plane shader is
// 7, and the Vulkan minimum guarantees 16 vertex output locations - so it is
// free everywhere measured while staying inside the floor every device must
// support. A shader that already reaches 15 is refused rather than patched
// around, because a silent collision would corrupt whatever varying already
// lived there.
// TWO varyings, not one: the current clip position as well as the previous.
//
// The motion vector is (curr.xy/curr.w - prev.xy/prev.w), and that subtraction
// has to happen per FRAGMENT, after perspective-correct interpolation - doing
// it per vertex and interpolating the result is wrong on anything not parallel
// to the screen.
//
// So the fragment shader needs both, and it cannot read gl_Position: that is a
// vertex output, not a fragment input. Passing the same value again as an
// ordinary varying is the standard answer and is what every engine does here.
// CHOSEN FROM THE DEVICE LIMIT, NOT FROM A SAMPLE.
//
// These were fixed at 15 and 16, reasoned from the highest Location seen in the
// first forty modules, which was 7. Across a full session of 1500 modules that
// was wrong 97 times: X-Plane's large shaders - the 12 KB, 51 KB and 54 KB ones
// - already use both, and every one of them was refused. Six per cent of the
// scene would have drawn with no velocity.
//
// Two lessons, and the second is the one that generalises. A limit taken from a
// convenient sample is a guess wearing a measurement's clothes. And the right
// end of the range to claim is the TOP: shaders allocate varyings upward from
// 0, so the highest locations the device supports are the least contended.
//
// Set at device creation from maxVertexOutputComponents / 4 (components, not
// locations - four per vec4). The Vulkan minimum of 64 components gives 16
// locations, so the fallback below stays inside what every device guarantees.
// The diagnostic probes that lived here - TAA_MV_RAW, TAA_MV_DEBUG_DEPTH and
// TAA_MV_PROBE_CONST - are gone. They existed to answer whether the prevClip
// varying linked at all and whether uReproj arrived, and both questions are
// settled: the epipolar residual is 0.000 to 0.003 px per frame, which no
// shader with a dead varying or a zero matrix can produce.
//
// They are removed rather than left switched off. Each one REPLACED the real
// velocity write with something else, so a stray environment variable in a
// user's launcher would silently turn the field into a debug pattern, and the
// field would still pass every magnitude test. That is the same failure shape
// as the sign error these probes were built to chase.
inline uint32_t &currClipLocation() { static uint32_t v = 14; return v; }
inline uint32_t &prevClipLocation() { static uint32_t v = 15; return v; }

// THE VELOCITY ATTACHMENT INDEX IS FIXED TOO, AND FOR THE SAME REASON.
//
// In a fragment shader, Location on an Output IS the colour attachment index.
// The first version wrote to "one past the highest output this shader already
// uses", which is wrong in exactly the way the per-shader varying locations
// were wrong: it is a property of the SHADER, when it has to be a property of
// the PASS. A shader with one output, drawing into a pass with three
// attachments, wrote its velocity into attachment 1 - a live G-buffer target -
// instead of attachment 3. The screen went black, with zero rejections from the
// driver: nothing was invalid, it was all written to the wrong place.
//
// So the index is pinned, every pass is padded out to it with null slots, and
// every fragment shader writes the same Location meaning the same thing
// everywhere. Chosen from maxColorAttachments; the Vulkan minimum is 4, so the
// fallback stays inside what every device guarantees.
inline uint32_t &mvAttachmentIndex() { static uint32_t v = 3; return v; }

inline void chooseAttachment(uint32_t maxColorAttachments)
{
    uint32_t n = maxColorAttachments;
    if (n < 4) n = 4;
    if (n > 8) n = 8;          // padding every pass to more than this is waste
    mvAttachmentIndex() = n - 1;
}

// Called once the device's limits are known. `vertexComponents` and
// `fragmentComponents` are maxVertexOutputComponents and
// maxFragmentInputComponents; the pair has to fit inside BOTH, since the same
// two locations are written by one stage and read by the other.
// How many Locations this device actually offers, and whether that is enough to
// sit clear of X-Plane. Set by chooseLocations(); read by placeLocations() and
// by the layer, so that a device we have never run on says something specific
// instead of failing silently.
inline uint32_t &deviceLocationCount() { static uint32_t n = 0; return n; }
inline bool     &locationsAreSafe()    { static bool ok = true; return ok; }

// X-PLANE'S CEILING, MEASURED, NOT GUESSED.
//
// Every Location decoration in the shader corpus, resolved against its storage
// class: the highest is 16, in both vertex-output and fragment-input, and
// nothing reaches 17. Every Location-decorated variable is a scalar or a vector
// occupying exactly one slot - there are no mat4s silently consuming L..L+3 -
// so the highest CONSUMED slot is 16 as well. Locations 17..31 are free in all
// 6855 modules.
//
// A pair placed at 17 or above therefore cannot collide with anything X-Plane
// ships. Below it, collision is not a risk but a certainty in the busy range:
// Location 13 is read by 2352 fragment shaders and Location 14 by 2784.
static const uint32_t kXPlaneMaxLocation = 16;

// Every Location this module decorates, into a caller-supplied bitmap.
//
// Used to pick our varying pair from EVIDENCE rather than from the device
// limit. The limit says how many locations exist; it says nothing about which
// ones X-Plane already uses, and picking the top two because they are within
// the limit is how 79 of 1250 modules came back LOCATION_TAKEN - refused,
// writing no motion vectors at all, so every pixel they drew had FSR2 reject
// history and crawl.
inline void scanUsedLocations(const uint32_t *code, size_t sizeBytes,
                              bool *used, uint32_t nLoc)
{
    if (!code || sizeBytes < 20) return;
    size_t n = sizeBytes / 4;
    if (code[0] != 0x07230203u) return;
    size_t i = 5;
    while (i < n) {
        uint16_t op  = (uint16_t)(code[i] & 0xFFFF);
        uint16_t len = (uint16_t)(code[i] >> 16);
        if (len == 0 || i + len > n) break;
        // OpDecorate <id> Location <n>, and the member form for block I/O.
        if (op == OpDecorate && len >= 4 && code[i + 2] == Deco_Location) {
            uint32_t L = code[i + 3];
            if (L < nLoc) used[L] = true;
        } else if (op == OpMemberDecorate && len >= 5 && code[i + 3] == Deco_Location) {
            uint32_t L = code[i + 4];
            if (L < nLoc) used[L] = true;
        }
        i += len;
    }
}

// Place our pair in the highest ADJACENT gap that no module uses.
//
// Highest rather than lowest: shaders allocate upward from 0, so the top of the
// range is the least contended, and a pair that is free today at location 3 is
// far more likely to be claimed by some shader we have not seen yet.
inline bool placeLocations(const bool *used, uint32_t nLoc)
{
    // ---- THE CENSUS MUST NOT EXCEED WHAT THE DEVICE OFFERS.
    //
    // This scanned the full 32-slot bitmap regardless of the device, so on an
    // implementation reporting 64 components (16 Locations) it would happily
    // return 30/31 - free in every X-Plane module, and past the end of what the
    // stage can carry. Nothing rejects that: the pipeline builds, the module
    // validates, and the varying reads zero. Location 31 already taught us this
    // once, on a card with 32 slots, by delivering exactly 0.000 for a value
    // known to survive Location 30.
    //
    // deviceLocationCount() is 0 until chooseLocations() has run, in which case
    // the caller's bound is all we have.
    uint32_t devLocs = deviceLocationCount();
    if (devLocs && devLocs < nLoc) nLoc = devLocs;
    // One slot of headroom below the ceiling: built-ins consume output
    // components too, so the last slot the arithmetic allows is not necessarily
    // a slot that links.
    if (nLoc >= 3) nLoc -= 1;

    for (uint32_t L = nLoc; L >= 2; --L) {
        uint32_t a = L - 1, b = L - 2;
        if (!used[a] && !used[b]) {
            prevClipLocation() = a;
            currClipLocation() = b;
            // Evidence beats the corpus-wide constant: if the census found this
            // pair free across every module X-Plane has handed us, it is safe
            // wherever it landed.
            locationsAreSafe() = true;
            return true;
        }
    }
    return false;
}

inline void chooseLocations(uint32_t vertexComponents, uint32_t fragmentComponents)
{
    uint32_t comps = vertexComponents < fragmentComponents
                   ? vertexComponents : fragmentComponents;
    uint32_t locs = comps / 4;
    deviceLocationCount() = locs;
    // ---- IS THERE ROOM ABOVE X-PLANE AT ALL?
    //
    // Vulkan's guaranteed minimum is 64 components = 16 Locations. X-Plane's own
    // ceiling is also 16. So on a minimum-specification implementation there is
    // NO free pair above it, and the arithmetic below would happily hand back
    // 13/14 - two of the most heavily used slots in the corpus - with the
    // pipeline building, the module validating, and the varyings reading
    // whatever X-Plane wrote there.
    //
    // We need our pair plus one slot of headroom below the ceiling (see the
    // Location-31 note below), so 19 Locations is the floor.
    //
    // This does not fire on the hardware we develop on - real desktop GPUs
    // report 128 components = 32 Locations - which is exactly why it has to be
    // checked rather than assumed. It is the same failure shape as the varying
    // limit taken from a sample of forty modules: a limit that happens to hold
    // where you are looking is not a limit.
    locationsAreSafe() = (locs >= kXPlaneMaxLocation + 3);
    if (locs < 4) locs = 4;               // absurdly small device; stay in range

    // ---- HEADROOM BELOW THE CEILING, and why.
    //
    // This used the top two slots: 31 and 30 of 32. Location 30 delivers
    // correctly and Location 31 delivers ZERO - proven by sending
    // gl_Position.w, a value known to survive location 30, through 31 and
    // reading back 0.000 on every pixel.
    //
    // The component limit is not the whole story of what a stage can carry.
    // Built-ins consume output components too - this vertex shader's entry
    // point lists v_clip_distances alongside gl_Position - so the last slot the
    // arithmetic allows is not necessarily a slot that links. The failure is
    // silent: the pipeline builds, the module validates, and the varying simply
    // reads zero.
    //
    // TAA_MV_LOC pins the pair for testing; the default now leaves two slots of
    // headroom rather than sitting on the ceiling.
    uint32_t top = locs - 2;
    if (const char *e = getenv("TAA_MV_LOC")) {
        uint32_t v = (uint32_t)atoi(e);
        if (v >= 2 && v < locs) top = v;
    }
    prevClipLocation() = top;
    currClipLocation() = top - 1;
}

// Vertex modules with more than one OpReturn, which is the single case the
// exit-block splice cannot cover. Counted rather than silently tolerated: if
// this is ever non-zero the affected shaders fall back to the first-store
// placement, which is the assumption light_vis proved wrong, and the log has to
// say so rather than leave it to be rediscovered.
inline uint64_t &multiReturnModules() { static uint64_t n = 0; return n; }

struct Ins { uint16_t op, len; size_t at; };

inline uint32_t head(uint16_t op, uint16_t len)
{
    return ((uint32_t)len << 16) | (uint32_t)op;
}

enum Result {
    INJ_OK = 0,
    INJ_NOT_VERTEX,       // fragment/compute - nothing to do, not a failure
    INJ_NO_POSITION,      // no BuiltIn Position
    INJ_NO_STORE,         // never writes gl_Position
    INJ_LOCATION_TAKEN,   // shader already uses prevClipLocation()
    INJ_MALFORMED
};

// Returns INJ_OK and fills `out` on success. `location` receives the varying
// location chosen, which the paired fragment shader has to read from.
inline Result inject(const uint32_t *code, size_t sizeBytes,
                     std::vector<uint32_t> &out, uint32_t *location)
{
    if (!code || sizeBytes < 20) return INJ_MALFORMED;
    std::vector<uint32_t> w(code, code + sizeBytes / 4);
    if (w[0] != 0x07230203u) return INJ_MALFORMED;
    uint32_t version = w[1];

    std::vector<Ins> ins;
    {
        size_t i = 5;
        while (i < w.size()) {
            Ins d;
            d.op  = (uint16_t)(w[i] & 0xFFFF);
            d.len = (uint16_t)(w[i] >> 16);
            d.at  = i;
            if (d.len == 0 || i + d.len > w.size()) break;
            ins.push_back(d);
            i += d.len;
        }
    }

    uint32_t idFloat = 0, idV4 = 0, idMat4 = 0, idInt = 0, idConst0 = 0, idConst1 = 0;
    uint32_t idBool = 0;
    uint32_t idPtrOutV4 = 0, idPtrPCV4Existing = 0;
    uint32_t idPerVertexType = 0, idPerVertexVar = 0, idDirectPosVar = 0;
    size_t   entryAt = 0; uint16_t entryLen = 0;
    bool     isVertex = false;
    uint32_t maxLocation = 0;
    bool     locationTaken = false;
    size_t   annotationsEnd = 0, globalsEnd = 0;

    for (size_t k = 0; k < ins.size(); ++k) {
        const Ins &d = ins[k];
        const uint32_t *p = &w[d.at];
        switch (d.op) {
        case OpEntryPoint:
            if (d.len >= 2 && p[1] == ExecModel_Vertex) {
                isVertex = true; entryAt = d.at; entryLen = d.len;
            }
            break;
        case OpTypeBool:  if (d.len >= 2 && !idBool) idBool = p[1]; break;
        case OpTypeInt:   if (d.len >= 4 && p[2] == 32 && p[3] == 1 && !idInt) idInt = p[1]; break;
        case OpTypeFloat: if (d.len >= 3 && p[2] == 32) idFloat = p[1]; break;
        case OpTypeVector: if (d.len >= 4 && p[2] == idFloat && p[3] == 4) idV4 = p[1]; break;
        case OpTypeMatrix: if (d.len >= 4 && p[2] == idV4 && p[3] == 4) idMat4 = p[1]; break;
        case OpTypePointer:
            if (d.len >= 4 && p[2] == SC_Output && p[3] == idV4 && !idPtrOutV4) idPtrOutV4 = p[1];
            // A PushConstant-to-vec4 pointer type may already exist. Declaring a
            // second one with identical operands is not an error the way a
            // duplicate OpConstant is, but reusing it is free and keeps the
            // module closer to what the driver already compiled once.
            if (d.len >= 4 && p[2] == SC_PushConstant && p[3] == idV4 && !idPtrPCV4Existing)
                idPtrPCV4Existing = p[1];
            break;
        case OpConstant:
            // SPIR-V requires constants to be UNIQUE: two OpConstants of the
            // same type and value make the module invalid. So an existing 0 or 1
            // has to be found and reused rather than declared again - the same
            // rule that bit the duplicate-constant case in the fragment path.
            if (d.len >= 4 && p[1] == idInt && p[3] == 0 && !idConst0) idConst0 = p[2];
            if (d.len >= 4 && p[1] == idInt && p[3] == 1 && !idConst1) idConst1 = p[2];
            break;
        case OpDecorate:
            if (d.len >= 4 && p[2] == Deco_Location) {
                if (p[3] >= maxLocation) maxLocation = p[3] + 1;
                if (p[3] == currClipLocation() || p[3] == prevClipLocation()) locationTaken = true;
            }
            // gl_Position appears BOTH as a gl_PerVertex member and as a
            // standalone variable. Two of X-Plane's fifteen use the second
            // form, and handling only the first left them unpatched - which
            // would not fail, it would leave holes in the velocity buffer
            // wherever those shaders draw.
            if (d.len >= 4 && p[2] == Deco_BuiltIn && p[3] == BuiltIn_Position)
                idDirectPosVar = p[1];
            break;
        case OpMemberDecorate:
            if (d.len >= 5 && p[3] == Deco_BuiltIn && p[4] == BuiltIn_Position)
                idPerVertexType = p[1];
            break;
        default: break;
        }
    }

    if (!isVertex) return INJ_NOT_VERTEX;
    if (!idV4)     return INJ_NO_POSITION;
    if (!idPerVertexType && !idDirectPosVar) return INJ_NO_POSITION;
    if (locationTaken) return INJ_LOCATION_TAKEN;

    if (idPerVertexType) {
        for (size_t k = 0; k < ins.size() && !idPerVertexVar; ++k) {
            if (ins[k].op != OpVariable || ins[k].len < 3) continue;
            uint32_t ptrType = w[ins[k].at + 1];
            for (size_t j = 0; j < ins.size(); ++j) {
                if (ins[j].op != OpTypePointer || ins[j].len < 4) continue;
                const uint32_t *q = &w[ins[j].at];
                if (q[1] == ptrType && q[3] == idPerVertexType && q[2] == SC_Output) {
                    idPerVertexVar = w[ins[k].at + 2];
                    break;
                }
            }
        }
    }
    if (!idPerVertexVar && !idDirectPosVar) return INJ_NO_POSITION;

    // SPIR-V's logical layout is fixed: annotations, then types/constants/
    // global variables, then functions. A declaration in the wrong section is
    // invalid even when every individual instruction is well formed.
    for (size_t k = 0; k < ins.size(); ++k) {
        uint16_t op = ins[k].op;
        if (op == OpDecorate || op == OpMemberDecorate || op == OpDecorationGroup ||
            op == OpGroupDecorate || op == OpGroupMemberDecorate)
            annotationsEnd = ins[k].at + ins[k].len;
        if (op == OpFunction) break;
        if (op != OpName && op != OpMemberName)
            globalsEnd = ins[k].at + ins[k].len;
    }
    if (!annotationsEnd) annotationsEnd = globalsEnd;
    if (!globalsEnd) return INJ_MALFORMED;

    // ---- DOES THIS SHADER WRITE gl_Position AT ALL?
    //
    // That is the ONLY question asked here now. The value it stored and the
    // pointer it stored through used to be captured as well, and both are gone:
    // the injected code re-loads the position through its own access chain at
    // the exit block, so the shader's own store is evidence that a position
    // exists and nothing more. See the OpLoad block below.
    //
    // storeEnd doubles as the fallback splice point for the multiple-return
    // case, which is why the position of the store is still recorded.
    size_t storeEnd = 0;

    if (idDirectPosVar) {
        for (size_t k = 0; k < ins.size(); ++k) {
            if (ins[k].op != OpStore || ins[k].len < 3) continue;
            if (w[ins[k].at + 1] != idDirectPosVar) continue;
            storeEnd = ins[k].at + ins[k].len;
            break;
        }
    }
    for (size_t k = 0; k < ins.size() && !storeEnd; ++k) {
        if (ins[k].op != OpAccessChain || ins[k].len < 5) continue;
        const uint32_t *p = &w[ins[k].at];
        if (p[3] != idPerVertexVar) continue;
        if (p[4] != idConst0) continue;
        uint32_t chainId = p[2];
        for (size_t j = k + 1; j < ins.size(); ++j) {
            if (ins[j].op != OpStore || ins[j].len < 3) continue;
            if (w[ins[j].at + 1] != chainId) continue;
            storeEnd = ins[j].at + ins[j].len;
            break;
        }
    }
    if (!storeEnd) return INJ_NO_STORE;

    // ---- SPLICE AT THE EXIT, NOT AT THE STORE. See the OpLoad block for why
    //      the VALUE has to be re-loaded there; this is where it is loaded.
    //
    // Two independent reasons converge on the same placement:
    //
    //   1. The eye path. storeEnd sits immediately after the gl_Position write,
    //      but in the offending shader that is instruction 318 while
    //      v_position_eye_* is written at 328-336 - AFTER. Reading the varying
    //      at storeEnd reads an output nothing has written yet, and the first
    //      attempt duly produced a median of 978 px against 4.8 px on the old
    //      path. The idea was sound and the placement was not.
    //
    //   2. light_vis. gl_Position is written from 6-7 mutually exclusive
    //      branches with no unconditional write, so there is no single store
    //      whose operand is the final position. Only the exit sees it.
    //
    // Every id we use is still live at the return: each is defined before the
    // return it dominates, and the pointer we load through is module scope.
    //
    // MULTIPLE RETURNS ARE THE ONE CASE THIS CANNOT COVER. Patching the last
    // OpReturn in program order is only correct if control actually reaches it;
    // a shader with an early return would take a path with no varyings written,
    // which is undefined data feeding the motion vectors - worse than the old
    // behaviour rather than better. glslang emits a single return for main, so
    // this is expected to be empty, but "expected" is what the first-store
    // assumption was too. Fall back to the old placement and count it.
    size_t injectAt = storeEnd;
    size_t nReturns = 0, lastReturn = 0;
    for (size_t k = 0; k < ins.size(); ++k)
        if (ins[k].op == OpReturn) { ++nReturns; lastReturn = ins[k].at; }
    if (nReturns == 1) injectAt = lastReturn;
    else ++multiReturnModules();

    uint32_t bound = w[3];
    uint32_t idStructPC    = bound++;
    uint32_t idPtrPCStruct = bound++;
    uint32_t idPtrPCMat4   = bound++;
    uint32_t idPCVar       = bound++;
    // Per-vertex-module tag, mirroring the fragment one. The fragment tag named
    // the wrong stage: it only divides varyings, while prevClip is built here.
    static uint32_t s_vsPidCounter = 0;
    const uint32_t myVsPid = ++s_vsPidCounter;
    // Set when this module is the one being dumped, so the PATCHED words can be
    // written too. The original was dumped before; the injected code itself has
    // never been read back, and the identity it is supposed to implement fails
    // on a fifth of pixels.
    uint32_t g_dumpThisVsPid = 0;
    // Name the module so a tag in the report identifies a real shader, and dump
    // it on request so the thing can actually be read rather than theorised
    // about. VS pid 180 owns 164509 of ~200000 bad pixels.
    if (getenv("TAA_MV_PID")) {
        uint64_t vh = 1469598103934665603ull;
        for (size_t k = 0; k < w.size(); ++k) { vh ^= (uint64_t)w[k]; vh *= 1099511628211ull; }
        trace("MV VS PID %u -> module hash %016llx, %llu words",
              myVsPid, (unsigned long long)vh, (unsigned long long)w.size());
        // Match on the module HASH, not the pid. The pid counter follows module
        // creation order, which varies between runs - pid 180 identified the
        // terrain shader in one run and did not exist in the next. The hash is
        // a property of the module itself and is stable.
        // The hash used earlier (67f90e8ea1acad18) belongs to the FRAGMENT
        // module, not the vertex one - the two were conflated. Word count is a
        // stable property of the vertex module itself: the terrain shader is
        // 2077 words.
        if (const char *wantW = getenv("TAA_MV_DUMP_WORDS")) {
            if ((size_t)atoi(wantW) == w.size()) g_dumpThisVsPid = myVsPid;
        }
        const char *wantHash = getenv("TAA_MV_DUMP_HASH");
        char myHash[32];
        snprintf(myHash, sizeof(myHash), "%016llx", (unsigned long long)vh);
        const char *want = getenv("TAA_MV_DUMP_VS");
        if (wantHash || want) {
            if ((wantHash && strcmp(wantHash, myHash) == 0) ||
                (!wantHash && want && (uint32_t)atoi(want) == myVsPid)) {
                g_dumpThisVsPid = myVsPid;
                char pth[512];
                snprintf(pth, sizeof(pth), "%s/mv_vs_%u.spv",
                         getenv("TAA_MV_DIAG") ? getenv("TAA_MV_DIAG") : ".", myVsPid);
                if (FILE *vf = fopen(pth, "wb")) {
                    fwrite(&w[0], 4, w.size(), vf);
                    fclose(vf);
                    trace("MV VS DUMP: wrote %s (%llu words)", pth,
                          (unsigned long long)w.size());
                }
            }
        }
    }
    uint32_t idOutCurr     = bound++;
    uint32_t idOutPrev     = bound++;
    uint32_t newMat4 = 0, newPtrOutV4 = 0, newInt = 0, newConst0 = 0, newConst1 = 0;
    if (!idMat4)     { newMat4     = bound++; idMat4     = newMat4; }
    if (!idPtrOutV4) { newPtrOutV4 = bound++; idPtrOutV4 = newPtrOutV4; }
    if (!idInt)      { newInt      = bound++; idInt      = newInt; }
    if (!idConst0)   { newConst0   = bound++; idConst0   = newConst0; }
    if (!idConst1)   { newConst1   = bound++; idConst1   = newConst1; }
    // ---- NEAR-FIELD SELECT. See the block that emits these for why it is a
    // comparison and a select rather than a second matrix.
    uint32_t newBool = 0;
    if (!idBool) { newBool = bound++; idBool = newBool; }

    // A float 1.0, for the view vector's w. It is a literal so that nothing can
    // cancel: the clip-to-clip form this replaces recovered w as a difference
    // and lost all precision on distant geometry.
    uint32_t idConstOneV = 0, newOneV = 0;
    for (size_t k = 0; k < ins.size(); ++k) {
        const uint32_t *q = &w[ins[k].at];
        if (ins[k].op == OpConstant && ins[k].len >= 4 && q[1] == idFloat) {
            float v; memcpy(&v, &q[3], 4);
            if (v == 1.0f) { idConstOneV = q[2]; break; }
        }
    }
    if (!idConstOneV) { newOneV = bound++; idConstOneV = newOneV; }
    uint32_t idConst2NF = bound++;              // component index 2 of the vec4
    uint32_t idChainNF = bound++, idNFvec = bound++, idNFdist = bound++;
    uint32_t idNFcmp = bound++, idPrevSel = bound++;
    uint32_t idPrevSelTagged = 0;
    uint32_t idLoadedMat = bound++;
    uint32_t idChainPC   = bound++;
    uint32_t idPrevClip  = bound++;
    uint32_t idPosX = bound++, idPosY = bound++, idPosZ = bound++, idPosW = bound++;
    uint32_t idNegY = bound++, idFlipped = bound++;
    // Our own access chain to gl_Position, and the value LOADED through it at
    // the injection point. See the OpLoad block below for why the shader's own
    // stored value and its own pointer cannot be reused.
    uint32_t idPosChain  = bound++;
    uint32_t idLoadedPos = bound++;

    // Jitter: the pointer type, the access chain, the loaded vec4, its two
    // useful components, the two w-scaled offsets, the two summed coordinates,
    // and the vec4 that goes back to gl_Position.
    uint32_t idPtrPCV4 = idPtrPCV4Existing;
    uint32_t newPtrPCV4 = 0;
    if (!idPtrPCV4) { newPtrPCV4 = bound++; idPtrPCV4 = newPtrPCV4; }
    uint32_t idChainJit = bound++, idJit = bound++;
    uint32_t idJitX = bound++, idJitY = bound++;
    uint32_t idOffX = bound++, idOffY = bound++;
    uint32_t idJitPosX = bound++, idJitPosY = bound++, idJittered = bound++;

    std::vector<uint32_t> annos, globals, body;

    annos.push_back(head(OpDecorate, 3));       annos.push_back(idStructPC); annos.push_back(Deco_Block);
    annos.push_back(head(OpMemberDecorate, 5)); annos.push_back(idStructPC); annos.push_back(0); annos.push_back(Deco_Offset); annos.push_back(pushConstantOffset());
    annos.push_back(head(OpMemberDecorate, 4)); annos.push_back(idStructPC); annos.push_back(0); annos.push_back(Deco_ColMajor);
    annos.push_back(head(OpMemberDecorate, 5)); annos.push_back(idStructPC); annos.push_back(0); annos.push_back(Deco_MatrixStride); annos.push_back(16);
    // Member 1, the jitter, immediately after the matrix. std430 would place a
    // vec4 there anyway; it is stated explicitly because the layer pushes both
    // members in one call and the two descriptions have to agree exactly.
    annos.push_back(head(OpMemberDecorate, 5)); annos.push_back(idStructPC); annos.push_back(1); annos.push_back(Deco_Offset); annos.push_back(pushConstantOffset() + 64);
    annos.push_back(head(OpDecorate, 4)); annos.push_back(idOutCurr); annos.push_back(Deco_Location); annos.push_back(currClipLocation());
    annos.push_back(head(OpDecorate, 4)); annos.push_back(idOutPrev); annos.push_back(Deco_Location); annos.push_back(prevClipLocation());

    if (newInt)      { globals.push_back(head(OpTypeInt, 4));     globals.push_back(newInt); globals.push_back(32); globals.push_back(1); }
    if (newConst0)   { globals.push_back(head(OpConstant, 4));    globals.push_back(idInt);  globals.push_back(newConst0); globals.push_back(0); }
    if (newConst1)   { globals.push_back(head(OpConstant, 4));    globals.push_back(idInt);  globals.push_back(newConst1); globals.push_back(1); }
    if (newMat4)     { globals.push_back(head(OpTypeMatrix, 4));  globals.push_back(newMat4); globals.push_back(idV4); globals.push_back(4); }
    if (newPtrOutV4) { globals.push_back(head(OpTypePointer, 4)); globals.push_back(newPtrOutV4); globals.push_back(SC_Output); globals.push_back(idV4); }
    // OpTypeBool for the near-field comparison. Declared only if the module did
    // not already have one - duplicate type declarations are legal but a shader
    // the driver has already compiled is better left as close to itself as
    // possible.
    if (newBool)     { globals.push_back(head(OpTypeBool, 2));    globals.push_back(newBool); }
    if (newOneV)     { uint32_t bits; float o = 1.0f; memcpy(&bits, &o, 4);
                       globals.push_back(head(OpConstant, 4)); globals.push_back(idFloat); globals.push_back(newOneV); globals.push_back(bits); }
    (void)idConst2NF;

    globals.push_back(head(OpTypeStruct, 4));  globals.push_back(idStructPC);    globals.push_back(idMat4); globals.push_back(idV4);
    globals.push_back(head(OpTypePointer, 4)); globals.push_back(idPtrPCStruct); globals.push_back(SC_PushConstant); globals.push_back(idStructPC);
    globals.push_back(head(OpTypePointer, 4)); globals.push_back(idPtrPCMat4);   globals.push_back(SC_PushConstant); globals.push_back(idMat4);
    if (newPtrPCV4) { globals.push_back(head(OpTypePointer, 4)); globals.push_back(newPtrPCV4); globals.push_back(SC_PushConstant); globals.push_back(idV4); }
    globals.push_back(head(OpVariable, 4));    globals.push_back(idPtrPCStruct); globals.push_back(idPCVar);  globals.push_back(SC_PushConstant);
    globals.push_back(head(OpVariable, 4)); globals.push_back(idPtrOutV4); globals.push_back(idOutCurr); globals.push_back(SC_Output);
    globals.push_back(head(OpVariable, 4)); globals.push_back(idPtrOutV4); globals.push_back(idOutPrev); globals.push_back(SC_Output);

    // FLIP Y INTO THE CONVENTION reproj WAS BUILT FOR.
    //
    // X-Plane sets a NEGATIVE-height viewport (GL orientation via
    // maintenance1), so framebuffer y runs opposite to clip y. The depth-derived
    // shader builds its NDC from pixel coordinates - uv * 2 - 1, y downward -
    // and reproj was verified against THAT convention: 8.78 px measured, 8.77
    // predicted. gl_Position is y-up, so handing it over unflipped reprojects
    // along the wrong axis.
    //
    // Flipping here rather than in the fragment shader keeps the difference
    // symmetric: both varyings leave in the same space, and the fragment stays
    // a plain subtraction with nothing asymmetric to get backwards later.
    // ---- LOAD gl_Position. DO NOT REUSE THE SHADER'S STORED VALUE.
    //
    // This used `storedValue` - the operand of the first OpStore to gl_Position
    // - and that assumption holds for 488 of X-Plane's 494 vertex shaders and
    // semantically for 492. It breaks in one family, and the corpus names it:
    //
    //   light_vis_0.txt   6 stores to gl_Position, NONE unconditional
    //   light_vis_2.txt   7 stores to gl_Position, NONE unconditional
    //
    // Their whole body sits inside OpSelectionMerge / OpSwitch %uint_0, and
    // gl_Position is written from one branch per light shape. Taking the first
    // store picks an arbitrary shape; at runtime that branch is taken for one
    // value of the light-type uniform, so for every other value we would carry a
    // value that was never written - or an OpUndef.
    //
    // (The two-store cases in `particle` and `deferred_gbuf_3` are benign: the
    // second store is a conditional cull writing vec4(0), so the first store is
    // the real transform. Loading at the end is correct for those too - it just
    // carries the cull value for a vertex that rasterises nothing.)
    //
    // gl_Position is an Output variable, so it is loadable, and its value at the
    // return is by definition the final one however many branches wrote it.
    // Correct for all 494, and it deletes the store-scanning special cases
    // rather than adding another one.
    //
    // The chain is OURS rather than the shader's `storePtr`. In light_vis the
    // shader's OpAccessChain results are defined inside those same branches and
    // do not dominate the exit block, so reusing one would not even be valid
    // SPIR-V. gl_PerVertex is module scope and always available.
    uint32_t idPosPtr = idDirectPosVar;
    if (!idPosPtr) {
        body.push_back(head(OpAccessChain, 5)); body.push_back(idPtrOutV4); body.push_back(idPosChain); body.push_back(idPerVertexVar); body.push_back(idConst0);
        idPosPtr = idPosChain;
    }
    body.push_back(head(OpLoad, 4)); body.push_back(idV4); body.push_back(idLoadedPos); body.push_back(idPosPtr);

    body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idPosX); body.push_back(idLoadedPos); body.push_back(0);
    body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idPosY); body.push_back(idLoadedPos); body.push_back(1);
    body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idPosZ); body.push_back(idLoadedPos); body.push_back(2);
    body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idPosW); body.push_back(idLoadedPos); body.push_back(3);
    body.push_back(head(OpFNegate, 4)); body.push_back(idFloat); body.push_back(idNegY); body.push_back(idPosY);
    body.push_back(head(OpCompositeConstruct, 7)); body.push_back(idV4); body.push_back(idFlipped); body.push_back(idPosX); body.push_back(idNegY); body.push_back(idPosZ); body.push_back(idPosW);

    body.push_back(head(OpAccessChain, 5)); body.push_back(idPtrPCMat4); body.push_back(idChainPC); body.push_back(idPCVar); body.push_back(idConst0);
    body.push_back(head(OpLoad, 4));        body.push_back(idMat4);      body.push_back(idLoadedMat); body.push_back(idChainPC);

    // ---- ROW 1 IS THE ONLY ROW LEFT.
    //
    // prevClip.w is confirmed correct (relative error 0.0003), so row 3 is
    // right. dx is correct, so row 0 is right. M[5] came back 0.99951. Yet
    // prevY is wrong on 20.4% of pixels, so the fault is M[1], M[9] or M[13].
    // The divergence grows as depth falls, which points at M[9]*d: the observed
    // +0.43 shift over a depth change of -0.4 needs M[9] about -1.07, where the
    // layer reports 0.0000.
    //
    // Extract here, right after the load, so both the prevClip and currClip
    // stores can carry a value out. The previous attempt inserted into prevClip
    // AFTER its store had already been emitted, which is why M[0] came back as
    // the real z (about 10.3) instead of a matrix element.
    uint32_t idMatRow1a = 0, idMatRow1b = 0;
    if (getenv("TAA_MV_MATDUMP") || getenv("TAA_MV_RAWCLIP")) {
        const uint32_t idCol2 = bound++, idCol3 = bound++;
        idMatRow1a = bound++; idMatRow1b = bound++;
        // Carry M[5] out, the one element that determines prevY for a near
        // identity matrix, so the prediction can use the matrix THIS draw saw
        // rather than whatever the layer last pushed in some other frame.
        // Carry M[13] out now, not M[5]. M[5] histogrammed as a single value
        // so it is not varying; the least-squares model failing under two
        // different weightings means SOME element differs between draws, and
        // M[13] is the one still untested per-pixel.
        // ---- CARRY M[12], THE TERM THAT ACTUALLY MOVES.
        //
        // M[13] was verified equal between shader and diagnostic (0.001005 vs
        // 0.001000) but it is tiny and nearly static, so it could never expose a
        // frame skew. M[12] is the x translation - large and changing every
        // frame as the camera moves. prevNDC.x - u is about M[12]/d, so a stale
        // M[12] produces error scaling as 1/depth: worst on near geometry,
        // invisible far away, concentrated where flow is large. That is the
        // exact-mode tail.
        body.push_back(head(OpCompositeExtract, 5)); body.push_back(idV4); body.push_back(idCol2); body.push_back(idLoadedMat); body.push_back(3);
        body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idMatRow1a); body.push_back(idCol2); body.push_back(0);
        body.push_back(head(OpCompositeExtract, 5)); body.push_back(idV4); body.push_back(idCol3); body.push_back(idLoadedMat); body.push_back(3);
        body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idMatRow1b); body.push_back(idCol3); body.push_back(1);
    }
    // ---- THE MATRIX IS APPLIED TO THE RAW CLIP POSITION, NOT THE FLIPPED ONE.
    //
    // uReproj is built from X-Plane's world and projection matrices, which are
    // y-UP clip space. Feeding it a y-NEGATED vector mixes -y into every row,
    // the w row included, so prevClip.w comes out wrong wherever |y| is large -
    // and the fragment then divides by it.
    //
    // Measured, in metres, over the same frames:
    //
    //     prev.w   p05 0.017   med 0.383   p95 2210
    //     curr.w   p05 0.24    med 0.351   p95 8336
    //
    // The medians agree; the tails do not. prev.w reaches the 1.6 cm near plane
    // where curr.w never goes below 0.24 m, and those near-zero pixels are what
    // inflate the motion vectors by the 3x to 21x that varied with how many of
    // them were in frame.
    //
    // The flip was inherited from the depth-derived shader, which built its NDC
    // from PIXEL COORDINATES and so was genuinely y-down. That shader is gone.
    // A matrix in y-up clip space must be applied to a y-up clip position.
    //
    // TAA_MV_FLIP restores the old behaviour for comparison.
    static const bool flipForMatrix = (getenv("TAA_MV_FLIP") != nullptr);
    static const bool clipToClip     = (getenv("TAA_MV_CLIP2CLIP") != nullptr);

    // ---- THE VECTOR HANDED TO THE MATRIX IS (x, y, w, 1), NOT THE CLIP POSITION.
    //
    // uReproj is now a VIEW-to-previous-clip matrix. It expects the current view
    // position, and with this projection that rebuilds from gl_Position with no
    // subtraction: z_clip = w_clip + m14, so z_clip holds nothing w does not,
    // and view = (x/sx, y/sy, -w, 1) with 1/sx, 1/sy and the -1 folded into the
    // matrix on the host.
    //
    // The clip-to-clip form it replaces could not be evaluated in float32 here:
    // its w row is +-1/near = +-61.9 and the shader computed 61.9 * (w - z), a
    // difference of two nearly equal large numbers. prev.w collapsed to noise
    // for distant geometry and the divide by it inflated the field 3x to 21x.
    //
    // The literal 1.0 is what makes this exact - it is a constant, not a
    // difference, so nothing can cancel.
    // ---- USE THE EYE-SPACE POSITION THE SHADER ALREADY COMPUTED.
    //
    // (idPosX, idPosY, idPosW, 1) is clip xy with w in slot 2, which only
    // becomes view space after clipToView = diag(1/sx, 1/sy, -1, 1) divides the
    // projection back out. That step assumes ONE projection for the whole
    // frame, and the offending shader disproves the assumption:
    //
    //     transformIdx = (gl_InstanceIndex - gl_BaseInstance) % u_transform_count
    //     gl_Position  = mvp_matrix[transformIdx] * vec4(a_vertex - u_mv_offset, 1)
    //     eye          = modelview_matrix[transformIdx] * vec4(pos, 1)
    //
    // mvp_matrix and modelview_matrix are ARRAYS selected per instance, and the
    // shader writes gl_ViewportIndex and gl_Layer, so one draw can emit the same
    // geometry through several projections. A single sampled proj cannot invert
    // all of them, and the residual error is the parallax term M[12]/d - which
    // is why it tracks 1/depth and why the shaders drawing near ground are the
    // worst while distant ones look clean.
    //
    // But the shader hands us the answer: it already computed eye space and
    // wrote it to v_position_eye_*. Reading that varying skips the inversion
    // entirely, so it does not matter which projection the draw used. The layer
    // then pushes prevProj*relRot instead of prevProj*relRot*clipToView.
    uint32_t idEyeVar = 0;
    if (getenv("TAA_MV_EYE")) {
        // Walk OpName (opcode 5) looking for the eye-space varying. The literal
        // is a packed, null-terminated string starting at word 2.
        size_t sc = 5;
        while (sc < w.size()) {
            const uint16_t sop = (uint16_t)(w[sc] & 0xFFFF);
            const uint16_t slen = (uint16_t)(w[sc] >> 16);
            if (slen == 0) break;
            if (sop == 5 && slen >= 3) {
                std::string nm;
                for (size_t q = sc + 2; q < sc + slen && q < w.size(); ++q) {
                    const uint32_t packed = w[q];
                    bool done = false;
                    for (int b = 0; b < 4; ++b) {
                        const char ch = (char)((packed >> (8 * b)) & 0xFF);
                        if (ch == '\0') { done = true; break; }
                        nm.push_back(ch);
                    }
                    if (done) break;
                }
                if (nm.find("position_eye") != std::string::npos) {
                    idEyeVar = w[sc + 1];
                    break;
                }
            }
            sc += slen;
        }
    }
    uint32_t idViewVec = bound++;
    if (idEyeVar) {
        const uint32_t idEyeLoad = bound++;
        const uint32_t idEx = bound++, idEy = bound++, idEz = bound++;
        body.push_back(head(OpLoad, 4)); body.push_back(idV4);
        body.push_back(idEyeLoad); body.push_back(idEyeVar);
        body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idEx); body.push_back(idEyeLoad); body.push_back(0);
        body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idEy); body.push_back(idEyeLoad); body.push_back(1);
        body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idEz); body.push_back(idEyeLoad); body.push_back(2);
        body.push_back(head(OpCompositeConstruct, 7)); body.push_back(idV4); body.push_back(idViewVec);
        body.push_back(idEx); body.push_back(idEy); body.push_back(idEz); body.push_back(idConstOneV);
    } else {
        body.push_back(head(OpCompositeConstruct, 7)); body.push_back(idV4); body.push_back(idViewVec);
        body.push_back(idPosX); body.push_back(idPosY); body.push_back(idPosW); body.push_back(idConstOneV);
    }

    body.push_back(head(OpMatrixTimesVector, 5)); body.push_back(idV4);  body.push_back(idPrevClip); body.push_back(idLoadedMat);
    body.push_back(clipToClip ? (flipForMatrix ? idFlipped : idLoadedPos) : idViewVec);

    // ---- NEAR FIELD: THE COCKPIT'S CORRECT MOTION VECTOR IS ZERO.
    //
    // uReproj is a WORLD reprojection. It is exactly right for terrain and
    // buildings and exactly wrong for the instrument panel, which does not move
    // through the world at all - it moves WITH the camera. Feeding the panel a
    // world reprojection gives it the velocity of the scenery behind it, which
    // is the cockpit shake.
    //
    // The fix does not need a second matrix. For geometry rigidly attached to
    // the airframe, with a camera also rigidly attached - measured rigid to
    // 0.0022 m/frame - the previous-frame clip position is
    //
    //     prevClip = prevProj * view_body * p,   view_body constant
    //              = prevProj * proj^-1 * currClip
    //
    // which is currClip whenever the projection did not change. So the whole
    // correction is: near the camera, use the current position as the previous
    // one, and the velocity comes out zero.
    //
    // The threshold arrives in the spare .z of the jitter vec4 - the block
    // already carries a vec4 for a vec2 of jitter, so this costs no push
    // constant space and no extra range. Zero disables it for free: gl_Position.w
    // is the positive view depth for anything in front of the camera, so
    // `w < 0` is never true and the select always picks the world path.
    //
    // Attempted first as a per-PASS choice driven by TAA_COCKPIT_PASS, which
    // could not work: the pass ordinal is counted per command buffer and names
    // a different pass in each, so it put body-frame reprojection on the world
    // G-buffer twice. Depth is a property of the pixel and needs no census.
    body.push_back(head(OpAccessChain, 5)); body.push_back(idPtrPCV4); body.push_back(idChainNF); body.push_back(idPCVar); body.push_back(idConst1);
    body.push_back(head(OpLoad, 4));        body.push_back(idV4);      body.push_back(idNFvec);   body.push_back(idChainNF);
    body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idNFdist); body.push_back(idNFvec); body.push_back(2);
    body.push_back(head(OpFOrdLessThan, 5)); body.push_back(idBool); body.push_back(idNFcmp); body.push_back(idPosW); body.push_back(idNFdist);
    body.push_back(head(OpSelect, 6)); body.push_back(idV4); body.push_back(idPrevSel); body.push_back(idNFcmp); body.push_back(idFlipped); body.push_back(idPrevClip);

    // ---- DEBUG: REPORT THE MATRIX THE VERTEX SHADER ACTUALLY LOADED.
    //
    // prevClip arrives at the fragment shader as ZERO while currClip arrives
    // correct, and prevClip is the only one that passes through the loaded
    // push-constant matrix. prevClip = 0 is exactly what a matrix of zeros
    // produces, so the question is whether the load returns anything at all.
    //
    // This replaces prevClip with (m00, m33, 0, 1). The fragment already writes
    // prevClip.w into .y in this mode, so .y becomes m33: 1.0 if the matrix
    // arrived, 0.0 if the load returned zeros and every motion vector this
    // shader has ever written was a difference against nothing.
    if (idMatRow1b) {
        const uint32_t idPT = bound++;
        body.push_back(head(OpCompositeInsert, 6)); body.push_back(idV4);
        body.push_back(idPT); body.push_back(idMatRow1b);
        body.push_back(idPrevSel); body.push_back(2);
        idPrevSelTagged = idPT;
    }
    // ---- TAA_MV_PREV_EQ_CURR: STORE THE SAME VALUE TO BOTH VARYINGS.
    //
    // The probe the identity substitution should have been, and this time the
    // expected result is EXACT, not a judgement call: prevClip and currClip
    // carry the identical id, so the fragment's (prev/pw - curr/cw) * 0.5 is
    // zero at every pixel by construction - same interpolated value, same
    // divide, same rounding. Nothing about the matrix, the push constants or
    // the reconstruction is involved.
    //
    //   field reads zero everywhere  ->  varyings, locations, interpolation,
    //                                    the fragment divide and the write are
    //                                    ALL clean. The fault is upstream: the
    //                                    matrix the multiply consumed.
    //   field still shows motion     ->  the fault is downstream of the vertex
    //                                    maths, and the M[12] anomaly is a
    //                                    separate problem, not this one.
    //
    // The identity probe failed as a probe because the matrix it replaced also
    // carries clipToView, so "identity" was not a no-op and the output had no
    // exact expected value - median 450 px of meaningless signal. This one has
    // exactly one possible correct answer.
    static const bool prevEqCurr = (getenv("TAA_MV_PREV_EQ_CURR") != nullptr);
    body.push_back(head(OpStore, 3)); body.push_back(idOutPrev);
    body.push_back(prevEqCurr ? (flipForMatrix ? idFlipped : idLoadedPos)
                              : (idPrevSelTagged ? idPrevSelTagged : idPrevSel));
    // currClip goes out in the same space the matrix works in, so the fragment's
    // subtraction is between two comparable vectors. Both raw, or both flipped -
    // never one of each.
    // ---- STAMP THE VERTEX SHADER'S IDENTITY INTO currClip.z.
    //
    // The fragment tag found the whole defect in one shader: pid 1477 owns
    // 83165 of ~85500 bad pixels (97%), 28.74% of its own 289410 pixels wrong,
    // mean error 48.99 px, while ten other shaders sit at 0.00% bad and
    // 0.15-0.25 px. But that tag names the FRAGMENT stage, and the fragment
    // only divides two varyings - prevClip is computed in the VERTEX shader, so
    // that is the stage worth naming.
    //
    // currClip.z is dead: the fragment reads .xy and .w and never touches it.
    // Writing the vertex module's own tag there costs nothing and carries the
    // right stage's identity through to the readback.
    // Locate gl_InstanceIndex (BuiltIn 43), gl_BaseInstance (BuiltIn 4425) and
    // the signed 32-bit int type, so the instance tag can be emitted. Any of
    // them missing simply disables the tag for that module.
    uint32_t idInstIdxVar = 0, idBaseInstVar = 0, idIntType = 0;
    if (getenv("TAA_MV_INST")) {
        size_t q = 5;
        while (q < w.size()) {
            const uint16_t qop = (uint16_t)(w[q] & 0xFFFF);
            const uint16_t qln = (uint16_t)(w[q] >> 16);
            if (qln == 0) break;
            if (qop == OpDecorate && qln >= 4 && w[q + 2] == Deco_BuiltIn) {
                if (w[q + 3] == 43)   idInstIdxVar  = w[q + 1];
                if (w[q + 3] == 4425) idBaseInstVar = w[q + 1];
            }
            if (qop == OpTypeInt && qln >= 4 && w[q + 2] == 32 && w[q + 3] == 1)
                idIntType = w[q + 1];
            q += qln;
        }
        if (!idIntType) { idInstIdxVar = 0; idBaseInstVar = 0; }
    }

    // ---- IS THE BAD GEOMETRY DRAWN AT A NONZERO INSTANCE INDEX?
    //
    // The dominant shader selects mvp_matrix[] and modelview_matrix[] by
    // (gl_InstanceIndex - gl_BaseInstance) % u_transform_count and writes
    // gl_ViewportIndex and gl_Layer, so one draw can emit the same geometry
    // through several transforms. 26.75% of that shader's pixels are wrong
    // while the rest are exact, which is what a subset of instances looks like.
    //
    // u_transform_count cannot be read generically, but gl_InstanceIndex is a
    // built-in available in every vertex shader. Stamping the relative instance
    // index into currClip.z tests the whole family at once: if the bad pixels
    // carry nonzero indices and the good ones carry zero, the multi-view path
    // is the mechanism and our single pushed matrix is wrong for those draws.
    // ---- LET THE SHADER REPORT THE MATRIX IT ACTUALLY RECEIVED.
    //
    // Twice now "the matrix must be wrong" has been INFERRED from the shader's
    // outputs. currClip.y is verified equal to the pixel's v, M[9] is ~0, and
    // still d(prevY)/d(yc) comes out -0.79 where 1.0000 was pushed. The push
    // range overlap was a real hazard but only 3 layouts hit it and the tail did
    // not move, so that was not the mechanism either.
    //
    // currClip.z and prevClip.z are both dead - the fragment reads .xy and .w
    // from each. Carrying M[5] and M[0] out through them reports, per pixel,
    // exactly what the vertex shader loaded from the push constant, next to the
    // velocity it produced. No inference left.
    const uint32_t idMatM5 = idMatRow1a;

    uint32_t idCurrOut = flipForMatrix ? idFlipped : idLoadedPos;
    if (idMatM5) {
        const uint32_t idCurrTag = bound++;
        body.push_back(head(OpCompositeInsert, 6)); body.push_back(idV4);
        body.push_back(idCurrTag); body.push_back(idMatM5);
        body.push_back(idCurrOut); body.push_back(2);
        idCurrOut = idCurrTag;
    }
    if (getenv("TAA_MV_INST") && idInstIdxVar && idBaseInstVar) {
        const uint32_t idII = bound++, idBI = bound++, idRel = bound++, idRelF = bound++;
        const uint32_t idTagged2 = bound++;
        body.push_back(head(OpLoad, 4)); body.push_back(idIntType); body.push_back(idII); body.push_back(idInstIdxVar);
        body.push_back(head(OpLoad, 4)); body.push_back(idIntType); body.push_back(idBI); body.push_back(idBaseInstVar);
        body.push_back(head(OpISub, 5)); body.push_back(idIntType); body.push_back(idRel); body.push_back(idII); body.push_back(idBI);
        body.push_back(head(OpConvertSToF, 4)); body.push_back(idFloat); body.push_back(idRelF); body.push_back(idRel);
        body.push_back(head(OpCompositeInsert, 6)); body.push_back(idV4);
        body.push_back(idTagged2); body.push_back(idRelF);
        body.push_back(idCurrOut); body.push_back(2);
        idCurrOut = idTagged2;
    }
    else if (getenv("TAA_MV_PID")) {
        uint32_t bits; float pv = (float)myVsPid; memcpy(&bits, &pv, 4);
        const uint32_t idVsPidK = bound++;
        globals.push_back(head(OpConstant, 4)); globals.push_back(idFloat);
        globals.push_back(idVsPidK); globals.push_back(bits);
        const uint32_t idTagged = bound++;
        body.push_back(head(OpCompositeInsert, 6)); body.push_back(idV4);
        body.push_back(idTagged); body.push_back(idVsPidK);
        body.push_back(idCurrOut); body.push_back(2);
        idCurrOut = idTagged;
    }
    body.push_back(head(OpStore, 3)); body.push_back(idOutCurr); body.push_back(idCurrOut);

    // ---- SUB-PIXEL JITTER, in clip space.
    //
    //     gl_Position.xy += jitter.xy * gl_Position.w
    //
    // Multiplied by w because the perspective divide is still to come: adding a
    // constant to clip xy would displace near geometry far more than distant
    // geometry, which is a warp, not a jitter. Scaling by w makes the offset
    // constant in NDC after the divide, which is the same thing a translation
    // in the projection matrix does - and is why this is the honest way to say
    // "we jittered the projection" to an upscaler that assumes it.
    //
    // AFTER the two varyings are stored, and deliberately so. currClip and
    // prevClip must stay UNJITTERED: the fragment shader differences them into
    // a motion vector, and a jitter present in one frame and not the last would
    // appear as sub-pixel motion on every static surface. FSR2 is told the
    // jitter separately, once, and cancels it itself - which it can only do if
    // it is not already baked into the vectors.
    //
    // idPosX/Y/Z/W are the ORIGINAL components extracted above. idNegY is the
    // flipped copy that went to the varyings and is not touched here.
    body.push_back(head(OpAccessChain, 5)); body.push_back(idPtrPCV4); body.push_back(idChainJit); body.push_back(idPCVar); body.push_back(idConst1);
    body.push_back(head(OpLoad, 4));        body.push_back(idV4);      body.push_back(idJit);      body.push_back(idChainJit);
    body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idJitX); body.push_back(idJit); body.push_back(0);
    body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idJitY); body.push_back(idJit); body.push_back(1);
    body.push_back(head(OpFMul, 5)); body.push_back(idFloat); body.push_back(idOffX); body.push_back(idJitX); body.push_back(idPosW);
    body.push_back(head(OpFMul, 5)); body.push_back(idFloat); body.push_back(idOffY); body.push_back(idJitY); body.push_back(idPosW);
    body.push_back(head(OpFAdd, 5)); body.push_back(idFloat); body.push_back(idJitPosX); body.push_back(idPosX); body.push_back(idOffX);
    body.push_back(head(OpFAdd, 5)); body.push_back(idFloat); body.push_back(idJitPosY); body.push_back(idPosY); body.push_back(idOffY);
    body.push_back(head(OpCompositeConstruct, 7)); body.push_back(idV4); body.push_back(idJittered); body.push_back(idJitPosX); body.push_back(idJitPosY); body.push_back(idPosZ); body.push_back(idPosW);
    // Through OUR pointer, for the same dominance reason as the load above.
    body.push_back(head(OpStore, 3)); body.push_back(idPosPtr); body.push_back(idJittered);

    out.clear();
    out.reserve(w.size() + annos.size() + globals.size() + body.size() + 4);
    for (int i = 0; i < 5; ++i) out.push_back(w[i]);
    out[3] = bound;

    size_t i = 5;
    while (i < w.size()) {
        uint16_t op  = (uint16_t)(w[i] & 0xFFFF);
        uint16_t len = (uint16_t)(w[i] >> 16);
        if (len == 0) break;

        if (i == entryAt) {
            // From SPIR-V 1.4 EVERY global variable must appear in the entry
            // point's interface, push constants included; before that, only
            // Input and Output. Getting this wrong is a validation error rather
            // than a silent one, which is a mercy.
            bool needPC = (version >= 0x00010400u);
            uint16_t extra = needPC ? 3 : 2;
            out.push_back(head(OpEntryPoint, (uint16_t)(len + extra)));
            for (uint16_t k = 1; k < len; ++k) out.push_back(w[i + k]);
            out.push_back(idOutCurr);
            out.push_back(idOutPrev);
            if (needPC) out.push_back(idPCVar);
            i += len;
            continue;
        }

        for (uint16_t k = 0; k < len; ++k) out.push_back(w[i + k]);
        i += len;

        if (i == annotationsEnd) for (size_t k = 0; k < annos.size();   ++k) out.push_back(annos[k]);
        if (i == globalsEnd)     for (size_t k = 0; k < globals.size(); ++k) out.push_back(globals[k]);
        if (i == injectAt)       for (size_t k = 0; k < body.size();    ++k) out.push_back(body[k]);
    }

    // ---- DUMP THE PATCHED MODULE, NOT JUST THE ORIGINAL.
    //
    // prevY = M[1]*cx + M[5]*cy + M[9]*cw + M[13] is forced by the code as
    // written and fails on 21.95% of pixels with every term measured. The
    // injected instructions themselves have never been read back. This writes
    // them so the disassembly can be compared against what the source intends.
    if (g_dumpThisVsPid) {
        char pth2[512];
        snprintf(pth2, sizeof(pth2), "%s/mv_vs_%u_patched.spv",
                 getenv("TAA_MV_DIAG") ? getenv("TAA_MV_DIAG") : ".", g_dumpThisVsPid);
        if (FILE *pf = fopen(pth2, "wb")) {
            fwrite(&out[0], 4, out.size(), pf);
            fclose(pf);
            trace("MV VS DUMP: wrote %s (%llu words patched)", pth2,
                  (unsigned long long)out.size());
        }
    }

    if (location) *location = prevClipLocation();
    return INJ_OK;
}

// ---------------------------------------------------------------- fragment
//
// Reads the two clip positions the patched vertex shader emits and writes the
// screen-space motion vector to a new colour attachment:
//
//     mv = (curr.xy / curr.w - prev.xy / prev.w) * 0.5
//
// The 0.5 converts NDC to UV: NDC spans -1..1 across the screen while UV spans
// 0..1, so a displacement in NDC is twice the same displacement in UV. Getting
// that factor wrong produces vectors that are exactly double - which looks like
// a plausible over-shoot rather than an obvious error, and would be attributed
// to something else entirely.
//
// The divide happens HERE, per fragment, not in the vertex shader. Dividing per
// vertex and interpolating the quotient is wrong on any surface not parallel to
// the screen: interpolation is perspective-correct on the clip positions, and
// doing the division first throws that away. Ground and runway - shallow
// surfaces filling most of the screen - are exactly where it would show.
// THE ATTACHMENT INDEX IS PASSED IN, not decided here.
//
// It is the render pass's own colorAttachmentCount, which is knowable only at
// pipeline creation - so this is called from there, once per (module, count)
// pair, rather than once per module at vkCreateShaderModule.
//
// Two earlier versions got this wrong in opposite directions. Deriving it from
// the shader's own outputs put velocity on top of a live G-buffer target and
// rendered black. Pinning it to a fixed index and padding every pass out to
// eight attachments was valid but so heavy the driver returned
// VK_ERROR_UNKNOWN. Passing it in needs no padding at all: a pipeline built for
// a pass with N attachments gets N+1, and the shader writes Location N.
// THE REACTIVE VALUE IS BAKED IN, not pushed.
//
// It is a property of the PIPELINE - derived from its blend state, which is
// fixed at creation - so it is constant for every fragment the shader will ever
// produce. A push constant would re-send that constant on every draw, and would
// also mean extending each pipeline layout with a fragment-visible range, which
// is the kind of change that turns a rendering bug into a validation error.
//
inline Result injectFragment(const uint32_t *code, size_t sizeBytes,
                             std::vector<uint32_t> &out, uint32_t attachmentIndex)
{
    if (!code || sizeBytes < 20) return INJ_MALFORMED;
    std::vector<uint32_t> w(code, code + sizeBytes / 4);
    if (w[0] != 0x07230203u) return INJ_MALFORMED;
    uint32_t version = w[1];

    std::vector<Ins> ins;
    {
        size_t i = 5;
        while (i < w.size()) {
            Ins d;
            d.op  = (uint16_t)(w[i] & 0xFFFF);
            d.len = (uint16_t)(w[i] >> 16);
            d.at  = i;
            if (d.len == 0 || i + d.len > w.size()) break;
            ins.push_back(d);
            i += d.len;
        }
    }

    uint32_t idFloat = 0, idV4 = 0, idV2 = 0;
    uint32_t idPtrOutV4 = 0, idPtrInV4 = 0;
    uint32_t idConstHalf = 0, idConstZero = 0;
    size_t   entryAt = 0;
    bool     isFragment = false;
    uint32_t maxOutLocation = 0;
    bool     locationTaken = false;
    size_t   annotationsEnd = 0, globalsEnd = 0, entryReturnAt = 0;
    std::vector<uint32_t> outputVars;

    for (size_t k = 0; k < ins.size(); ++k) {
        const Ins &d = ins[k];
        const uint32_t *p = &w[d.at];
        switch (d.op) {
        case OpEntryPoint:
            if (d.len >= 2 && p[1] == ExecModel_Fragment) { isFragment = true; entryAt = d.at; }
            break;
        case OpTypeFloat: if (d.len >= 3 && p[2] == 32) idFloat = p[1]; break;
        case OpTypeVector:
            if (d.len >= 4 && p[2] == idFloat && p[3] == 4) idV4 = p[1];
            if (d.len >= 4 && p[2] == idFloat && p[3] == 2) idV2 = p[1];
            break;
        case OpTypePointer:
            if (d.len >= 4 && p[3] == idV4 && p[2] == SC_Output && !idPtrOutV4) idPtrOutV4 = p[1];
            if (d.len >= 4 && p[3] == idV4 && p[2] == SC_Input  && !idPtrInV4)  idPtrInV4  = p[1];
            break;
        case OpConstant:
            if (d.len >= 4 && p[1] == idFloat) {
                float v; memcpy(&v, &p[3], 4);
                if (v == 0.5f && !idConstHalf) idConstHalf = p[2];
                if (v == 0.0f && !idConstZero) idConstZero = p[2];
            }
            break;
        case OpDecorate:
            if (d.len >= 4 && p[2] == Deco_Location) {
                if (p[3] == currClipLocation() || p[3] == prevClipLocation())
                    locationTaken = true;
                // NOTE: the attachment-slot collision is handled after this
                // loop, not here. It cannot be checked here at all - outputVars
                // is filled by the OpVariable case below, and in SPIR-V the
                // annotations section precedes the variable declarations, so
                // this vector is always empty at this point. The version that
                // lived here also compared against the GLOBAL mvAttachmentIndex()
                // rather than the caller's, which is a different slot again.
            }
            break;
        case OpVariable:
            if (d.len >= 4 && p[3] == SC_Output) outputVars.push_back(p[2]);
            break;
        default: break;
        }
    }

    if (!isFragment) return INJ_NOT_VERTEX;   // "not our stage" - not a failure
    if (!idV4 || !idFloat) return INJ_MALFORMED;
    if (locationTaken) return INJ_LOCATION_TAKEN;

    // ---- MOVE A DEAD OUTPUT ASIDE RATHER THAN REFUSING THE WHOLE SHADER.
    //
    // The caller's index is the pass's colorAttachmentCount, so the pass owns
    // Locations 0..attachmentIndex-1 and we are adding attachmentIndex. A
    // fragment shader that ALREADY declares an output at attachmentIndex is
    // therefore writing past the end of the pass - that output is discarded by
    // the driver today and contributes nothing.
    //
    // The old code refused, to avoid "overwriting a real render target, which
    // is what turned the screen black". That fear is right in general and wrong
    // here: a real render target is one the pass declares, and this one is by
    // construction beyond them. Refusing cost far more than it saved - the
    // pipeline then failed the both-or-neither rule, threw away a perfectly
    // good vertex patch, and left those draws writing NO velocity into a target
    // cleared to zero. Zero means "did not move", so FSR2 pinned those pixels
    // and trailed them while the camera moved.
    //
    // So relocate the dead output to one past the highest location in use. It
    // was writing nowhere before and it writes nowhere after; the only change
    // is that it stops sitting in the slot we need.
    {
        uint32_t maxLoc = attachmentIndex;
        for (size_t k = 0; k < ins.size(); ++k) {
            if (ins[k].op != OpDecorate || ins[k].len < 4) continue;
            const uint32_t *p = &w[ins[k].at];
            if (p[2] != Deco_Location) continue;
            for (size_t v = 0; v < outputVars.size(); ++v)
                if (outputVars[v] == p[1] && p[3] > maxLoc) maxLoc = p[3];
        }
        for (size_t k = 0; k < ins.size(); ++k) {
            if (ins[k].op != OpDecorate || ins[k].len < 4) continue;
            uint32_t *p = &w[ins[k].at];
            if (p[2] != Deco_Location) continue;
            for (size_t v = 0; v < outputVars.size(); ++v)
                if (outputVars[v] == p[1] && p[3] == attachmentIndex)
                    p[3] = ++maxLoc;
        }
    }

    for (size_t k = 0; k < ins.size(); ++k) {
        uint16_t op = ins[k].op;
        if (op == OpDecorate || op == OpMemberDecorate || op == OpDecorationGroup ||
            op == OpGroupDecorate || op == OpGroupMemberDecorate)
            annotationsEnd = ins[k].at + ins[k].len;
        if (op == OpFunction) break;
        if (op != OpName && op != OpMemberName)
            globalsEnd = ins[k].at + ins[k].len;
    }
    if (!annotationsEnd) annotationsEnd = globalsEnd;
    if (!globalsEnd) return INJ_MALFORMED;

    // Insert before the LAST OpReturn, so the write happens whatever the shader
    // did on the way there - including early-out branches, which a discard-heavy
    // fragment shader is full of.
    for (size_t k = 0; k < ins.size(); ++k)
        if (ins[k].op == OpReturn) entryReturnAt = ins[k].at;
    if (!entryReturnAt) return INJ_NO_STORE;

    uint32_t bound = w[3];
    uint32_t newV2 = 0, newPtrOutV4 = 0, newPtrInV4 = 0, newHalf = 0, newZero = 0;
    // ---- A PER-PIPELINE TAG, SO THE BAD DRAWS CAN BE NAMED.
    //
    // The exact-depth prediction agrees with the epipolar metric (176.0 px
    // against 174 px on the worst band), so the metric was sound and the field
    // really is wrong. The error is bimodal - median 0.2955 px with 22.88% of
    // pixels beyond 1 px - which is a subset of DRAWS, not a wrong matrix.
    //
    // It cannot be the matrix or the inputs: rasterisation guarantees a
    // fragment's NDC equals currClip.xy/currClip.w, one matrix is pushed per
    // frame, and prev.w comes back correct to 0.1% on every band. So some draws
    // must be producing a prevClip that is not M*currClip.
    //
    // prevDepth is verified good and no longer worth a channel. Spending it on
    // a per-module tag turns "some subset" into a name: bin the error by tag,
    // then dump that module's SPIR-V and read what it actually does.
    static uint32_t s_pidCounter = 0;
    const uint32_t myPid = ++s_pidCounter;
    uint32_t newPid = 0;
    if (!idV2)        { newV2       = bound++; idV2        = newV2; }
    if (!idPtrOutV4)  { newPtrOutV4 = bound++; idPtrOutV4  = newPtrOutV4; }
    if (!idPtrInV4)   { newPtrInV4  = bound++; idPtrInV4   = newPtrInV4; }
    if (!idConstHalf) { newHalf     = bound++; idConstHalf = newHalf; }
    if (!idConstZero) { newZero     = bound++; idConstZero = newZero; }
    uint32_t idConstPid = 0;
    if (getenv("TAA_MV_PID")) { newPid = bound++; idConstPid = newPid; }

    uint32_t idInCurr  = bound++, idInPrev = bound++, idOutMV = bound++;
    uint32_t idLc = bound++, idLp = bound++;
    uint32_t idCw = bound++, idPw = bound++;
    uint32_t idCwv = bound++, idPwv = bound++;
    uint32_t idCxy = bound++, idPxy = bound++;
    uint32_t idCn = bound++, idPn = bound++;
    uint32_t idDiff = bound++, idScaled = bound++, idResult = bound++;

    std::vector<uint32_t> annos, globals, body;

    annos.push_back(head(OpDecorate, 4)); annos.push_back(idInCurr); annos.push_back(Deco_Location); annos.push_back(currClipLocation());
    annos.push_back(head(OpDecorate, 4)); annos.push_back(idInPrev); annos.push_back(Deco_Location); annos.push_back(prevClipLocation());
    annos.push_back(head(OpDecorate, 4)); annos.push_back(idOutMV);  annos.push_back(Deco_Location); annos.push_back(attachmentIndex);

    if (newV2)       { globals.push_back(head(OpTypeVector, 4)); globals.push_back(newV2); globals.push_back(idFloat); globals.push_back(2); }
    if (newPtrOutV4) { globals.push_back(head(OpTypePointer, 4)); globals.push_back(newPtrOutV4); globals.push_back(SC_Output); globals.push_back(idV4); }
    if (newPtrInV4)  { globals.push_back(head(OpTypePointer, 4)); globals.push_back(newPtrInV4);  globals.push_back(SC_Input);  globals.push_back(idV4); }
    if (newHalf)     { uint32_t bits; float h = 0.5f; memcpy(&bits, &h, 4);
                       globals.push_back(head(OpConstant, 4)); globals.push_back(idFloat); globals.push_back(newHalf); globals.push_back(bits); }
    if (newZero)     { uint32_t bits; float z = 0.0f; memcpy(&bits, &z, 4);
                       globals.push_back(head(OpConstant, 4)); globals.push_back(idFloat); globals.push_back(newZero); globals.push_back(bits); }
    if (newPid)      { uint32_t bits; float pv = (float)myPid; memcpy(&bits, &pv, 4);
                       globals.push_back(head(OpConstant, 4)); globals.push_back(idFloat); globals.push_back(newPid); globals.push_back(bits); }
    // A tag is only useful if it names something. Hash the incoming module so a
    // pid in the report can be matched back to the exact shader and dumped.
    if (newPid) {
        uint64_t h = 1469598103934665603ull;
        for (size_t k = 0; k < w.size(); ++k) {
            h ^= (uint64_t)w[k];
            h *= 1099511628211ull;
        }
        trace("MV FS PID %u -> module hash %016llx, %llu words",
              myPid, (unsigned long long)h, (unsigned long long)w.size());
    }

    globals.push_back(head(OpVariable, 4)); globals.push_back(idPtrInV4);  globals.push_back(idInCurr); globals.push_back(SC_Input);
    globals.push_back(head(OpVariable, 4)); globals.push_back(idPtrInV4);  globals.push_back(idInPrev); globals.push_back(SC_Input);
    globals.push_back(head(OpVariable, 4)); globals.push_back(idPtrOutV4); globals.push_back(idOutMV);  globals.push_back(SC_Output);

    body.push_back(head(OpLoad, 4)); body.push_back(idV4); body.push_back(idLc); body.push_back(idInCurr);
    body.push_back(head(OpLoad, 4)); body.push_back(idV4); body.push_back(idLp); body.push_back(idInPrev);
    body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idCw); body.push_back(idLc); body.push_back(3);
    body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idPw); body.push_back(idLp); body.push_back(3);
    body.push_back(head(OpCompositeConstruct, 5)); body.push_back(idV2); body.push_back(idCwv); body.push_back(idCw); body.push_back(idCw);
    body.push_back(head(OpCompositeConstruct, 5)); body.push_back(idV2); body.push_back(idPwv); body.push_back(idPw); body.push_back(idPw);
    body.push_back(head(OpVectorShuffle, 7)); body.push_back(idV2); body.push_back(idCxy); body.push_back(idLc); body.push_back(idLc); body.push_back(0); body.push_back(1);
    body.push_back(head(OpVectorShuffle, 7)); body.push_back(idV2); body.push_back(idPxy); body.push_back(idLp); body.push_back(idLp); body.push_back(0); body.push_back(1);
    body.push_back(head(OpFDiv, 5)); body.push_back(idV2); body.push_back(idCn); body.push_back(idCxy); body.push_back(idCwv);
    body.push_back(head(OpFDiv, 5)); body.push_back(idV2); body.push_back(idPn); body.push_back(idPxy); body.push_back(idPwv);
    // ---- prev - curr, NOT curr - prev.
    //
    // This subtracted the other way round for the whole life of the project,
    // and no test could see it: every statistic on the field was a magnitude,
    // and a negated vector has exactly the right magnitude. The direct
    // calibration finally named it, on the steady frames of a scripted yaw:
    //
    //     field=(-13.249, -0.041)  matrix=(+13.150, -0.000)  ratio=-1.008
    //     field=(-13.219, +0.038)  matrix=(+13.150, -0.000)  ratio=-1.005
    //     field=(-13.247, -0.042)  matrix=(+13.150, -0.000)  ratio=-1.007
    //
    // A clean -1, three frames running.
    //
    // Every consumer this field exists for - FSR2, DLSS, a TAA resolve - wants
    // the vector that carries the current pixel back to where it was, so the
    // history is sampled at uv + velocity. That is prev - curr. Written the
    // other way it does not fail or warn; it reprojects exactly twice as far in
    // exactly the wrong direction, which reads as heavy ghosting and gets
    // chased as a tuning problem.
    body.push_back(head(OpFSub, 5)); body.push_back(idV2); body.push_back(idDiff); body.push_back(idPn); body.push_back(idCn);
    body.push_back(head(OpVectorTimesScalar, 5)); body.push_back(idV2); body.push_back(idScaled); body.push_back(idDiff); body.push_back(idConstHalf);
    body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(bound); body.push_back(idScaled); body.push_back(0);
    uint32_t idMx = bound++;
    body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(bound); body.push_back(idScaled); body.push_back(1);
    uint32_t idMy = bound++;

    // (velocity.x, velocity.y, 0, 0). A colour attachment output is a
    // vec4; the target is two channels, so the last two are shape, not
    // information, and the format discards them.
    // ---- TRUE DEPTH, SO NEARNESS STOPS BEING INFERRED FROM THE FLOW.
    //
    // The residual tail was attributed to geometry about 0.2 m from the lens,
    // via d = sx*t/flow_ndc. That distance is computed FROM the flow, so it
    // assumes the flow is right - which is the thing under test. If the flow is
    // wrongly large the inferred distance comes out spuriously small, and the
    // conclusion "it is near geometry" follows no matter what the truth is. It
    // is the same circularity as solving depth from the X channel and then
    // finding X correct.
    //
    // currClip.w and prevClip.w ARE the view-space depths, already sitting in
    // this shader. Writing them in place of the velocity pair routes real depth
    // through the readback that already works, with no new buffer, no format
    // handling and no layout transitions on an image X-Plane owns.
    //
    // Set TAA_MV_WRITE_DEPTH=1 and the target carries (currDepth, prevDepth).
    // Then "are the huge-flow pixels actually near?" is answered by measurement
    // instead of by assuming the answer.
    static const bool writeDepth = getenv("TAA_MV_WRITE_DEPTH") != nullptr;
    static const bool wantRGBA   = getenv("TAA_MV_RGBA") != nullptr;
    const uint32_t idCh0 = writeDepth ? idCw : idMx;
    const uint32_t idCh1 = writeDepth ? idPw : idMy;
    // In RGBA mode both arrive together: velocity in xy, depths in zw, so the
    // flow can be predicted from measured depth instead of from an epipolar
    // line that degenerates near the focus of expansion.
    const uint32_t idCh2 = wantRGBA ? idCw : idConstZero;
    // Channel 3 carries the VERTEX shader's tag, which the vertex stage stamped
    // into currClip.z - the component this shader never reads. The fragment's
    // own tag named a stage that only performs a divide; prevClip is computed
    // in the vertex shader, so that is the identity worth carrying out.
    // ---- HAND BACK THE RAW CLIP VALUES.
    //
    // Every prediction in this project assumes currClip.y/currClip.w equals the
    // pixel's NDC v. Rasterisation guarantees that only if currClip really is
    // gl_Position, and that assumption has never been measured - it is the one
    // thing every elimination so far has rested on.
    //
    // TAA_MV_RAWCLIP writes (prevClip.y, prevClip.w, currClip.y, currClip.w) so
    // the shader's own numbers can be compared against both the pixel position
    // and the pushed matrix, ending the inference.
    // Field check: emit (vx, vy, prevY, prevW) so the velocity can be tested
    // against the clip values it is computed from - the last link in the chain
    // that has never been measured.
    static const bool fieldChk = getenv("TAA_MV_FIELDCHK") != nullptr;
    static const bool matDump = getenv("TAA_MV_MATDUMP") != nullptr;
    static const bool rawClip = getenv("TAA_MV_RAWCLIP") != nullptr;
    uint32_t idCh3 = wantRGBA ? idPw : idConstZero;
    uint32_t idMatCz = 0, idMatPz = 0;
    if (matDump || rawClip) {
        idMatCz = bound++; idMatPz = bound++;
        body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idMatCz); body.push_back(idLc); body.push_back(2);
        body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idMatPz); body.push_back(idLp); body.push_back(2);
    }
    // All four inputs measured: (prevY, currX, currY, currW). Nothing derived,
    // so prevY = M[1]*currX + M[5]*currY + M[9]*currW + M[13] can be checked
    // against the shader's own numbers.
    uint32_t idRawCy = 0, idRawPy = 0, idRawCx = 0;
    if (rawClip || fieldChk) {
        idRawCy = bound++; idRawPy = bound++; idRawCx = bound++;
        body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idRawCx); body.push_back(idLc); body.push_back(0);
        body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idRawCy); body.push_back(idLc); body.push_back(1);
        body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idRawPy); body.push_back(idLp); body.push_back(1);
    }
    if (idConstPid) {
        const uint32_t idVsTag = bound++;
        body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat);
        body.push_back(idVsTag); body.push_back(idLc); body.push_back(2);
        idCh3 = idVsTag;
    }
    body.push_back(head(OpCompositeConstruct, 7)); body.push_back(idV4); body.push_back(idResult);
    body.push_back(fieldChk ? idMx : (rawClip ? idRawPy : idCh0));
    body.push_back(fieldChk ? idMy : (rawClip ? idRawCx : idCh1));
    // ---- ONE FRAGMENT, ONE FRAME, EVERY QUANTITY.
    //
    // prevW measured right to 0.03% while prevY was wrong by 23% at the same
    // pixel, with every row-1 element verified - a contradiction only possible
    // because those two readings came from different RUNS. This puts
    // (prevY, prevW, currY, M[5]) in a single fragment so the shader's own
    // numbers can be checked against each other with nothing cross-run left.
    // currW is recoverable as currY / v, so no fifth channel is needed.
    body.push_back(fieldChk ? idRawPy : (rawClip ? idRawCy : (matDump ? idCw : idCh2)));
    body.push_back(fieldChk ? idPw   : (rawClip ? idCw   : (matDump ? idMatCz : idCh3)));
    body.push_back(head(OpStore, 3)); body.push_back(idOutMV); body.push_back(idResult);

    out.clear();
    out.reserve(w.size() + annos.size() + globals.size() + body.size() + 8);
    for (int i = 0; i < 5; ++i) out.push_back(w[i]);
    out[3] = bound;

    size_t i = 5;
    while (i < w.size()) {
        uint16_t op  = (uint16_t)(w[i] & 0xFFFF);
        uint16_t len = (uint16_t)(w[i] >> 16);
        if (len == 0) break;

        if (i == entryAt) {
            bool needAll = (version >= 0x00010400u);
            (void)needAll;
            out.push_back(head(OpEntryPoint, (uint16_t)(len + 3)));
            for (uint16_t k = 1; k < len; ++k) out.push_back(w[i + k]);
            out.push_back(idInCurr);
            out.push_back(idInPrev);
            out.push_back(idOutMV);
            i += len;
            continue;
        }

        if (i == entryReturnAt) {
            for (size_t k = 0; k < body.size(); ++k) out.push_back(body[k]);
        }

        for (uint16_t k = 0; k < len; ++k) out.push_back(w[i + k]);
        i += len;

        if (i == annotationsEnd) for (size_t k = 0; k < annos.size();   ++k) out.push_back(annos[k]);
        if (i == globalsEnd)     for (size_t k = 0; k < globals.size(); ++k) out.push_back(globals[k]);
    }

    return INJ_OK;
}

} // namespace spvinj
