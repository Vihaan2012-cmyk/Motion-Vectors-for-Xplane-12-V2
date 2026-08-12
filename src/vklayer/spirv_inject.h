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
    OpConstant = 43,
    OpFunction = 54, OpVariable = 59, OpLoad = 61, OpStore = 62,
    OpAccessChain = 65,
    OpDecorate = 71, OpMemberDecorate = 72, OpDecorationGroup = 73,
    OpGroupDecorate = 74, OpGroupMemberDecorate = 75,
    OpMatrixTimesVector = 145,
    OpVectorShuffle = 79, OpCompositeConstruct = 80, OpCompositeExtract = 81,
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
// TAA_MV_DEBUG_DEPTH: write the fragment.s NDC DEPTH into .y instead of the
// vertical velocity.
//
// Depth is the one quantity in this whole chain that has never been measured.
// The field disagrees with the matrix by factors that do NOT scale with the
// commanded rotation - cutting the rate fivefold left them where they were - so
// what remains is a component that depends on depth and on a camera drift that
// is also rate-independent. At 1.5 mm of drift, 350 px means geometry 1.3 cm
// from the eye. Either something really is that close, or the depth reaching
// this shader is not the depth of the surface being drawn. This says which.
// TAA_MV_RAW: read at CALL time, not from a global written by a static
// initialiser. The previous form was false when vertex modules were patched and
// true when fragment ones were, so the vertex probe silently never compiled in -
// and its apparent zeros were trusted for six turns.
inline bool debugRaw()
{
    static int v = -1;
    if (v < 0) v = getenv("TAA_MV_RAW") ? 1 : 0;
    return v != 0;
}
inline bool &debugDepthMode() { static bool v = false; return v; }
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
    for (uint32_t L = nLoc; L >= 2; --L) {
        uint32_t a = L - 1, b = L - 2;
        if (!used[a] && !used[b]) {
            prevClipLocation() = a;
            currClipLocation() = b;
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

    size_t   storeEnd = 0;
    uint32_t storedValue = 0;
    // The POINTER the shader stored gl_Position through, kept so the jittered
    // position can be written back to the same place. Whichever of the two
    // forms this shader uses - a gl_PerVertex member or a standalone variable -
    // the pointer is what OpStore took, so storing through it again needs no
    // knowledge of which form it was.
    uint32_t storePtr = 0;

    if (idDirectPosVar) {
        for (size_t k = 0; k < ins.size(); ++k) {
            if (ins[k].op != OpStore || ins[k].len < 3) continue;
            if (w[ins[k].at + 1] != idDirectPosVar) continue;
            storedValue = w[ins[k].at + 2];
            storePtr    = w[ins[k].at + 1];
            storeEnd    = ins[k].at + ins[k].len;
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
            storedValue = w[ins[j].at + 2];
            storePtr    = chainId;
            storeEnd    = ins[j].at + ins[j].len;
            break;
        }
    }
    if (!storeEnd) return INJ_NO_STORE;

    uint32_t bound = w[3];
    uint32_t idStructPC    = bound++;
    uint32_t idPtrPCStruct = bound++;
    uint32_t idPtrPCMat4   = bound++;
    uint32_t idPCVar       = bound++;
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
    uint32_t idConst2NF = bound++;              // component index 2 of the vec4
    uint32_t idChainNF = bound++, idNFvec = bound++, idNFdist = bound++;
    uint32_t idNFcmp = bound++, idPrevSel = bound++;
    uint32_t idLoadedMat = bound++;
    uint32_t idChainPC   = bound++;
    uint32_t idPrevClip  = bound++;
    uint32_t idPosX = bound++, idPosY = bound++, idPosZ = bound++, idPosW = bound++;
    uint32_t idNegY = bound++, idFlipped = bound++;

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
    body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idPosX); body.push_back(storedValue); body.push_back(0);
    body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idPosY); body.push_back(storedValue); body.push_back(1);
    body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idPosZ); body.push_back(storedValue); body.push_back(2);
    body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(idPosW); body.push_back(storedValue); body.push_back(3);
    body.push_back(head(OpFNegate, 4)); body.push_back(idFloat); body.push_back(idNegY); body.push_back(idPosY);
    body.push_back(head(OpCompositeConstruct, 7)); body.push_back(idV4); body.push_back(idFlipped); body.push_back(idPosX); body.push_back(idNegY); body.push_back(idPosZ); body.push_back(idPosW);

    body.push_back(head(OpAccessChain, 5)); body.push_back(idPtrPCMat4); body.push_back(idChainPC); body.push_back(idPCVar); body.push_back(idConst0);
    body.push_back(head(OpLoad, 4));        body.push_back(idMat4);      body.push_back(idLoadedMat); body.push_back(idChainPC);
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
    body.push_back(head(OpMatrixTimesVector, 5)); body.push_back(idV4);  body.push_back(idPrevClip); body.push_back(idLoadedMat); body.push_back(flipForMatrix ? idFlipped : storedValue);

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
    if (debugDepthMode()) {
        body.push_back(head(OpCompositeExtract, 5)); body.push_back(idV4); body.push_back(bound); body.push_back(idLoadedMat); body.push_back(0);
        uint32_t idDbgC0 = bound++;
        body.push_back(head(OpCompositeExtract, 5)); body.push_back(idV4); body.push_back(bound); body.push_back(idLoadedMat); body.push_back(3);
        uint32_t idDbgC3 = bound++;
        body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(bound); body.push_back(idDbgC0); body.push_back(0);
        uint32_t idDbgM00 = bound++;
        body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(bound); body.push_back(idDbgC3); body.push_back(3);
        uint32_t idDbgM33 = bound++;
        // ---- TAA_MV_PROBE_CONST: write a LITERAL 1.0 instead of the matrix.
        //
        // "m33 reads zero" has been the basis of six turns of reasoning and the
        // probe behind it has never itself been checked. If a constant 1.0 also
        // arrives as zero, the fault is in this probe or in the prevClip
        // varying - NOT in the matrix - and everything concluded from that
        // reading is void.
        //
        // The .w component is what the fragment forwards, so the constant goes
        // The .w component is what the fragment forwards, so gl_Position.w goes
// there - a value ALREADY PROVEN to arrive through this exact varying,
// since currClip.w reads 0.24 m to 8.3 km correctly. If it arrives via
// prevClip too, the varying and the probe are sound and the matrix is
// genuinely zero. If it does not, the prevClip path is broken and every
// zero read from it means nothing about the matrix at all.
        uint32_t idDbgOut = bound++;
        if (getenv("TAA_MV_PROBE_CONST")) {
            body.push_back(head(OpCompositeConstruct, 7)); body.push_back(idV4); body.push_back(idDbgOut); body.push_back(idDbgM00); body.push_back(idPosW); body.push_back(idDbgM00); body.push_back(idPosW);
        } else {
            body.push_back(head(OpCompositeConstruct, 7)); body.push_back(idV4); body.push_back(idDbgOut); body.push_back(idDbgM00); body.push_back(idDbgM33); body.push_back(idDbgM00); body.push_back(idDbgM33);
        }
        body.push_back(head(OpStore, 3)); body.push_back(idOutPrev); body.push_back(idDbgOut);
    } else
    body.push_back(head(OpStore, 3)); body.push_back(idOutPrev); body.push_back(idPrevSel);
    // currClip goes out in the same space the matrix works in, so the fragment's
    // subtraction is between two comparable vectors. Both raw, or both flipped -
    // never one of each.
    body.push_back(head(OpStore, 3)); body.push_back(idOutCurr); body.push_back(flipForMatrix ? idFlipped : storedValue);

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
    body.push_back(head(OpStore, 3)); body.push_back(storePtr); body.push_back(idJittered);

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
        if (i == storeEnd)       for (size_t k = 0; k < body.size();    ++k) out.push_back(body[k]);
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
    if (!idV2)        { newV2       = bound++; idV2        = newV2; }
    if (!idPtrOutV4)  { newPtrOutV4 = bound++; idPtrOutV4  = newPtrOutV4; }
    if (!idPtrInV4)   { newPtrInV4  = bound++; idPtrInV4   = newPtrInV4; }
    if (!idConstHalf) { newHalf     = bound++; idConstHalf = newHalf; }
    if (!idConstZero) { newZero     = bound++; idConstZero = newZero; }

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
    body.push_back(head(OpFSub, 5)); body.push_back(idV2); body.push_back(idDiff); body.push_back(idCn); body.push_back(idPn);
    body.push_back(head(OpVectorTimesScalar, 5)); body.push_back(idV2); body.push_back(idScaled); body.push_back(idDiff); body.push_back(idConstHalf);
    body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(bound); body.push_back(idScaled); body.push_back(0);
    uint32_t idMx = bound++;
    body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(bound); body.push_back(idScaled); body.push_back(1);
    uint32_t idMy = bound++;

    if (debugDepthMode()) {
        // ---- RAW currClip.w INTO .y. NO CONVENTION INVOLVED.
        //
        // The previous version of this wrote z_ndc = z/w, and its output
        // contradicts the projection matrix: it measured small values for
        // distant geometry, implying z_ndc = near/d, while proj[10]/proj[11]
        // says z_ndc = 1 - near/d which tends to 1. Both cannot be true.
        //
        // w is the view depth in metres and needs no interpretation, so it
        // settles which measurement to distrust. Mountains should read
        // thousands and the instrument panel a fraction of a metre. If instead
        // it reads centimetres, the w reaching the fragment is wrong - and
        // since the fragment shader divides by exactly this w to form the
        // motion vector, that would explain a field the size of a near-plane
        // displacement.
        // .x keeps the velocity, .y carries this fragment's own w. With both
        // in the same pixel the CPU can reconstruct exactly what the matrix
        // implies FOR THAT PIXEL - its screen position and its depth - and
        // compare against the velocity actually written. No centre region, no
        // assumed depth, no percentile standing in for a value.
        // ---- WRITE prevClip.w INTO .y.
        //
        // The fragment shader has NO push constant - it never sees uReproj at
        // all. It receives prevClip as a varying that the VERTEX shader is
        // supposed to fill with uReproj * gl_Position. So the question is
        // whether that varying arrives carrying a reprojected position.
        //
        // A synthetic 0.25 degree yaw was pushed and stashed, the dump read
        // 5.337 px out of it, and the field came back at 0.002 px - no motion
        // from a matrix demanding 5.3 px. If prevClip.w comes back equal to
        // currClip.w, the vertex shader wrote the position through unchanged
        // and the reprojection never happened. If it comes back zero, the
        // varying is not linking at all and every velocity ever written was a
        // difference against nothing.
        //
        // It also retires the identity test, which proved nothing: a shader
        // that ignores uReproj writes zero whether identity or a rotation is
        // pushed, and that zero was read all night as proof it worked.
        body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(bound); body.push_back(idLp); body.push_back(3);
        idMy = bound++;
    }
    if (debugRaw()) {
        // .x = prev.w, .y = curr.w. Two numbers that MUST be equal for any
        // rigid camera motion - the same point is the same distance in front of
        // the eye either frame. No convention enters: not y-up versus y-down,
        // not the depth range, not a scale. Their ratio is exactly the factor
        // the fragment divides by when it forms the motion vector.
        body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(bound); body.push_back(idLp); body.push_back(0);
        idMx = bound++;
        body.push_back(head(OpCompositeExtract, 5)); body.push_back(idFloat); body.push_back(bound); body.push_back(idLp); body.push_back(1);
        idMy = bound++;
    }
    // (velocity.x, velocity.y, 0, 0). A colour attachment output is a
    // vec4; the target is two channels, so the last two are shape, not
    // information, and the format discards them.
    body.push_back(head(OpCompositeConstruct, 7)); body.push_back(idV4); body.push_back(idResult); body.push_back(idMx); body.push_back(idMy); body.push_back(idConstZero); body.push_back(idConstZero);
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
