// The velocity image X-Plane's own shaders write into, once patched.
//
// This is now the ONLY velocity field. It is RENDERED, as an extra colour
// attachment, by the shaders that draw the geometry.
//
// It replaced a compute pass that reconstructed velocity from the depth buffer.
// That derivation was limited by depth precision, near/far classification and
// every other property of a buffer never meant to answer this question, and it
// cost a full-resolution dispatch plus a depth barrier every frame - which was
// the confirmed cause of the stutter. This one is exact and costs nothing extra,
// because the geometry is already being drawn.
//
// FORMAT IS R16G16_SFLOAT: two channels, for the two numbers a motion vector
// is. Half float because the values are small screen-space offsets - a full
// screen of motion is 1.0 in UV - where half's ~3 decimal digits near zero are
// far more precision than a sub-pixel displacement needs, at half the bandwidth
// of a 32-bit format, written by every fragment of every draw.
//
// It briefly widened to R16G16B16A16 to carry an FSR reactive mask in .z. That
// mask is gone, and so is the width. Anything reading .z now reads the NEXT
// PIXEL - which is exactly the bug the readback stride comment below describes.

#pragma once

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <vector>

struct MvTarget {
    bool ready  = false;
    bool failed = false;

    VkDevice         device = VK_NULL_HANDLE;
    VkPhysicalDevice phys   = VK_NULL_HANDLE;
    uint32_t w = 0, h = 0;

    VkImage        image  = VK_NULL_HANDLE;
    VkDeviceMemory mem    = VK_NULL_HANDLE;
    VkImageView    view   = VK_NULL_HANDLE;
    VkImageLayout  layout = VK_IMAGE_LAYOUT_UNDEFINED;


    // Readback, so the VALUES can be measured rather than assumed.
    //
    // Everything so far proves the plumbing: shaders patched, attachment bound,
    // sim rendering. None of it proves the numbers are right. The depth-derived
    // field looked equally healthy until its dump reported world velocity of
    // 0.00 px with the camera moving 3.6 m in the frame - so this exists
    // specifically to ask the new field the same question.
    VkBuffer       readback    = VK_NULL_HANDLE;
    VkDeviceMemory readbackMem = VK_NULL_HANDLE;
    void          *readbackPtr = nullptr;
    VkDeviceSize   readbackSize = 0;
    bool           wantDump = false;    // set by the interval; cleared on record
    // COUNTDOWN, not a flag. The copy is recorded during the frame; the read
    // happens in QueuePresentKHR - and the present that ENDS the recording
    // frame arrives before the GPU has necessarily executed the copy. Reading
    // there returns whatever the buffer held before, which is the PREVIOUS
    // dump, twenty frames stale. That is not a visible failure: it is a
    // plausible velocity field belonging to a different camera pose, and it
    // reads as the vectors being wrong by a wandering factor. Counting two
    // presents puts a full frame boundary between the copy and the read.
    int            dumpPending = 0;

    // WHAT THE SHADER WAS HANDED FOR *THIS* COPY.
    //
    // The copy is recorded on one frame and read on the next, so comparing it
    // against the CURRENT snapshot compares a field to a matrix that describes
    // a different frame. Most of the time the camera is turning steadily and
    // the two agree; on the frame a self-test phase changes, the camera jumps,
    // and the field then shows a rigid ~350 px while the snapshot still reads
    // the steady 13 px - which looked exactly like the vectors being 26x wrong.
    // Captured at record time, these always describe the frame in the buffer.
    float          dumpReproj[16] = {0};
    float          dumpExpectedPx = 0.0f;
    int            dumpPhase      = 0;
    // Which SHARE frame that matrix came from, and which one is current when it
    // is read. `far` has been constant at 13.150 px in every line ever printed,
    // including frames whose field is a uniform 378 px - a matrix that never
    // varies while the field does means the two still are not the same frame.
    // Printing both ids says so outright instead of leaving it to be inferred.
    uint64_t       dumpShareFrame = 0;
    float          dumpNearClip   = 0.0f;
    // The projection, so the depth convention can be READ rather than assumed.
    float          dumpProj[16] = {0};
};

static MvTarget g_mv;
// Set by the layer once spirv_inject.h has been included. mv_target.h is
// included first, so it cannot ask the injector directly.
static bool g_mvDebugDepth = false;
static bool g_mvRawMode    = false;
static const int g_dumpDelay = getenv("TAA_MV_DUMP_DELAY")
                             ? atoi(getenv("TAA_MV_DUMP_DELAY")) : 8;
// .xy velocity in UV units. Two channels is all a velocity buffer carries -
static const VkFormat kMvFormat = VK_FORMAT_R16G16_SFLOAT;

// DERIVED, never restated. The readback size and the index stride both have to
// track kMvFormat, and the comment below used to say so while the numbers said
// otherwise: the format went back to RG16F and these stayed at RGBA16F values.
// The result measured as exactly 50% zeros and hundreds of pixels of motion on
// a camera that had not moved - every read landing on the wrong pixel.
static const uint32_t kMvHalves = 2;                 // R16G16 = two halves
static const uint32_t kMvBytes  = kMvHalves * 2;     // ...four bytes

static void mvDestroy(DeviceData &dd)
{
    MvTarget &m = g_mv;
    if (m.device == VK_NULL_HANDLE) { m = MvTarget(); return; }
    if (m.readbackPtr) dd.unmapMemory(m.device, m.readbackMem);
    if (m.readback)    dd.destroyBuffer(m.device, m.readback, nullptr);
    if (m.readbackMem) dd.freeMemory(m.device, m.readbackMem, nullptr);
    if (m.view)  dd.destroyImageView(m.device, m.view, nullptr);
    if (m.image) dd.destroyImage(m.device, m.image, nullptr);
    if (m.mem)   dd.freeMemory(m.device, m.mem, nullptr);
    m = MvTarget();
}

// Built once the scene target's size is known, which is why this is not created
// at device creation: the swapchain and the render scale both move, and a
// velocity image of the wrong size would silently mismatch the pass it is bound
// to rather than fail loudly.
static bool mvCreate(DeviceData &dd, VkDevice device, VkPhysicalDevice phys,
                     uint32_t w, uint32_t h)
{
    MvTarget &m = g_mv;
    if (m.ready && m.w == w && m.h == h) return true;
    if (m.failed) return false;

    mvDestroy(dd);
    m.device = device;
    m.phys   = phys;
    m.w = w; m.h = h;

    VkImageCreateInfo ici;
    memset(&ici, 0, sizeof(ici));
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = kMvFormat;
    ici.extent.width  = w;
    ici.extent.height = h;
    ici.extent.depth  = 1;
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    // COLOR_ATTACHMENT because the geometry shaders render into it; SAMPLED so
    // the resolve can read it; TRANSFER_DST so it can be cleared outside a
    // render pass if a frame ever needs that.
    ici.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                      | VK_IMAGE_USAGE_SAMPLED_BIT
                      | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (dd.createImage(device, &ici, nullptr, &m.image) != VK_SUCCESS) {
        trace("MV: image creation failed (%ux%u R16G16_SFLOAT)", w, h);
        m.failed = true;
        return false;
    }

    VkMemoryRequirements mr;
    memset(&mr, 0, sizeof(mr));
    dd.getImageMemReq(device, m.image, &mr);

    VkPhysicalDeviceMemoryProperties mp;
    memset(&mp, 0, sizeof(mp));
    if (!g_getPhysMemProps || phys == VK_NULL_HANDLE) {
        trace("MV: no memory properties available");
        m.failed = true;
        return false;
    }
    g_getPhysMemProps(phys, &mp);

    VkMemoryAllocateInfo mai;
    memset(&mai, 0, sizeof(mai));
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = mr.size;
    mai.memoryTypeIndex = velFindMemType(&mp, mr.memoryTypeBits,
                                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkMemoryPriorityAllocateInfoEXT vkPrio;
    velLowPriority(mai, vkPrio);
    if (mai.memoryTypeIndex == UINT32_MAX ||
        dd.allocateMemory(device, &mai, nullptr, &m.mem) != VK_SUCCESS) {
        trace("MV: memory allocation failed (%.1f MB)", mr.size / 1048576.0);
        m.failed = true;
        return false;
    }
    dd.bindImageMemory(device, m.image, m.mem, 0);

    VkImageViewCreateInfo ivci;
    memset(&ivci, 0, sizeof(ivci));
    ivci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image    = m.image;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format   = kMvFormat;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.layerCount = 1;
    if (dd.createImageView(device, &ivci, nullptr, &m.view) != VK_SUCCESS) {
        trace("MV: image view creation failed");
        m.failed = true;
        return false;
    }


    // Host-visible readback buffer, allocated once alongside the image.
    //
    // Sized from kMvBytes rather than a literal, because a copy into a buffer
    // half the size it needs is not a validation error on every driver - it is
    // a truncated dump that reads as the bottom half of the screen being empty.
    m.readbackSize = (VkDeviceSize)w * h * kMvBytes;
    VkBufferCreateInfo bci;
    memset(&bci, 0, sizeof(bci));
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = m.readbackSize;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (dd.createBuffer(device, &bci, nullptr, &m.readback) == VK_SUCCESS) {
        VkMemoryRequirements bmr;
        memset(&bmr, 0, sizeof(bmr));
        dd.getBufferMemReq(device, m.readback, &bmr);
        VkMemoryAllocateInfo bai;
        memset(&bai, 0, sizeof(bai));
        bai.sType          = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        bai.allocationSize = bmr.size;
        bai.memoryTypeIndex = velFindMemType(&mp, bmr.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (bai.memoryTypeIndex != UINT32_MAX &&
            dd.allocateMemory(device, &bai, nullptr, &m.readbackMem) == VK_SUCCESS) {
            dd.bindBufferMemory(device, m.readback, m.readbackMem, 0);
            dd.mapMemory(device, m.readbackMem, 0, VK_WHOLE_SIZE, 0, &m.readbackPtr);
        }
    }
    if (!m.readbackPtr)
        trace("MV: readback buffer unavailable - the field can be bound but not "
              "measured");

    m.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    m.ready  = true;
    trace("MV: velocity target ready %ux%u RG16F vel.xy (%.1f MB) - this is "
          "the one X-Plane's own shaders render into",
          w, h, mr.size / 1048576.0);
    return true;
}

// Copy the velocity image into the readback buffer.
//
// Recorded at the 3D/UI boundary - the same point the resolve uses - because
// that is after every scene pass by construction. Copying earlier would capture
// a half-written field and produce numbers that look like a broken shader
// rather than a mistimed read, which is a distinction that has already cost
// this project several rounds elsewhere.
static void mvRecordReadback(DeviceData &dd, VkCommandBuffer cb,
                             const float *reproj, float expectedPx, int phase,
                             uint64_t shareFrame, float nearClip,
                             const float *proj)
{
    MvTarget &m = g_mv;
    if (!m.ready || !m.readbackPtr || !m.wantDump) return;
    m.wantDump = false;

    if (reproj) memcpy(m.dumpReproj, reproj, sizeof(m.dumpReproj));
    m.dumpExpectedPx = expectedPx;
    m.dumpPhase      = phase;
    m.dumpShareFrame = shareFrame;
    m.dumpNearClip   = nearClip;
    if (proj) memcpy(m.dumpProj, proj, sizeof(m.dumpProj));

    VkImageMemoryBarrier b;
    memset(&b, 0, sizeof(b));
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b.oldLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    b.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = m.image;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                          0, nullptr, 1, &b);

    VkBufferImageCopy r;
    memset(&r, 0, sizeof(r));
    r.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    r.imageSubresource.layerCount = 1;
    r.imageExtent.width  = m.w;
    r.imageExtent.height = m.h;
    r.imageExtent.depth  = 1;
    dd.cmdCopyImageToBuffer(cb, m.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            m.readback, 1, &r);

    b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.newLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    dd.cmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                          0, nullptr, 0, nullptr, 1, &b);

    // EIGHT presents, not two.
    //
    // Two was chosen to put one frame boundary between the copy and the read.
    // At 4K with 28 passes the GPU runs further behind than that, and a copy
    // that has not executed leaves the PREVIOUS dump in the buffer - twenty
    // frames of camera rotation earlier, which is a rigid field several times
    // the size of the current one. That is exactly the shape of the surviving
    // failures: uniform across the frame, at a multiple of the prediction, on
    // frames whose matrix is correct.
    //
    // TAA_MV_DUMP_DELAY overrides, so the number can be measured rather than
    // guessed: if the failures thin out as it rises, latency was the cause.
    m.dumpPending = g_dumpDelay;
}

// Measure it. Deliberately the SAME statistics the depth-derived dump prints,
// so the two fields can be compared line for line rather than by impression.
static void mvReport(double camMoved, uint64_t nowShareFrame)
{
    MvTarget &m = g_mv;
    if (!m.dumpPending || !m.readbackPtr) return;
    if (--m.dumpPending > 0) return;

    // Taken from the record, not from now. See dumpReproj.
    const float *reproj       = m.dumpReproj;
    const float  expectedPx   = m.dumpExpectedPx;
    const int    selfTestPhase = m.dumpPhase;

    const uint16_t *px = (const uint16_t*)m.readbackPtr;
    double sum = 0.0, maxMag = 0.0;
    uint64_t n = 0, zero = 0, nan = 0;
    // Kept so a MEDIAN can be reported, not only a mean.
    //
    // A mean cannot distinguish "a handful of fragments dividing by a near-zero
    // w" from "every value wrong by a factor", and those need opposite fixes.
    // The median is immune to the first and reads the factor straight off in
    // the second - so one measurement decides it instead of a flight per guess.
    std::vector<float> mags;
    // Per-axis samples. A MEDIAN over the whole frame is robust to near-field
    // geometry: with any camera translation at all, close surfaces move far
    // more than distant ones, and a mean over a centre region is dominated by
    // them. Under pure rotation every depth moves alike, so the median is the
    // honest estimate of the rotation the camera actually performed.
    std::vector<float> axX, axY;
    mags.reserve(600000);

    // A CENTRE REGION, reported separately.
    //
    // Whole-screen statistics have no reference to be checked against. On an
    // approach the ground directly below the aircraft genuinely moves hundreds
    // of pixels per frame, so a median over the whole screen is dominated by
    // near-field geometry and cannot distinguish "correct" from "ten times too
    // large".
    //
    // The centre has a reference: the scripted self-test measured 8.78 px there
    // against 8.77 predicted under pure yaw. It looks at distant geometry along
    // the view axis, where the motion is modest and dominated by camera
    // rotation rather than by proximity - which is exactly the case the
    // prediction covers.
    double cSum = 0.0; uint64_t cN = 0;
    uint32_t cx0 = m.w * 3 / 8, cx1 = m.w * 5 / 8;
    uint32_t cy0 = m.h * 3 / 8, cy1 = m.h * 5 / 8;

    // SIGNED means, per component, for both fields.
    //
    // Magnitudes agreeing to 5% proves the two derivations compute the same
    // SIZE of motion. It says nothing about DIRECTION, and direction is where
    // this can be wrong without looking wrong: the depth shader works in
    // texture space with Y down, while the injected vertex shader applies its
    // own Y flip to match X-Plane's inverted viewport. If those disagree, both
    // fields still report identical magnitudes and identical medians.
    //
    // A flipped Y handed to FSR2 does not fail. It ghosts vertically and
    // sharpens horizontally, which reads as a tuning problem and is exactly the
    // kind of thing that costs a week. One extra pair of numbers settles it:
    // matching signs mean the conventions agree, opposite signs on Y alone name
    // the flip outright.
    double cSx = 0.0, cSy = 0.0;
    std::vector<float> cAxX, cAxY;

    // ---- THE DEPTH-FREE CONSISTENCY TEST.
    //
    // Every verdict above compares the field against a prediction made at an
    // ASSUMED depth - one metre, or infinity. That only decides anything while
    // the camera is purely rotating, because only then does depth stop
    // mattering. The aircraft is flying throughout this test, so from phase 4
    // onward the matrix carries real translation (near swings 9.06 to 15.43 px
    // while far stays pinned at 13.150), the centre pixel sits at some finite
    // depth nobody knows, and a disagreement with either endpoint proves
    // nothing at all. Phases 4 to 7 could not have passed as written.
    //
    // This needs no depth. Each pixel has two measured components and one
    // unknown distance, so the distance is solved from x and then used to
    // PREDICT y. If the field is right, the predicted y lands on the measured y
    // for every pixel at once; if it is wrong, no single distance satisfies
    // both and the residual blows up. It works under translation, rotation and
    // both together, which is what the later phases actually contain.
    //
    // Writing P = (u*d/sx, v*d/sy, -d, 1) for the view-space point behind pixel
    // (u, v) at distance d, each clip component is linear in d:
    //
    //     clip.k = d * (M[0][k]*u/sx + M[1][k]*v/sy - M[2][k]) + M[3][k]
    //
    // so x/w = Xp is one linear equation in d, and y follows.
    //
    // Four sign conventions are evaluated rather than one. The image-space Y
    // flip has already been guessed at and got wrong once, and a flipped Y is
    // invisible in every magnitude statistic on this page. Printing all four
    // lets the log name the convention instead of me assuming it.
    const float *RJ = m.dumpReproj;
    const double sxP = (m.dumpProj[0] != 0.0f) ? (double)m.dumpProj[0] : 1.0;
    const double syP = (m.dumpProj[5] != 0.0f) ? (double)m.dumpProj[5] : 1.0;
    std::vector<float> resid[4];
    std::vector<float> solvedD;
    // WHERE the bad pixels are, not just how many.
    //
    // The residual tail can mean two completely different things and the
    // numbers alone cannot tell them apart. If it is spread over the whole
    // frame, the reprojection is wrong. If it is clustered, it is geometry that
    // does not move with the camera - the propeller disc, control surfaces,
    // instrument needles - which no camera-only reprojection can predict and
    // which a TAA resolve is expected to reject by colour clamping instead.
    //
    // A centroid and a spread settle it in one line. Accumulated for the first
    // sign convention; the four agree closely enough that the choice does not
    // move the answer.
    double badSx = 0.0, badSy = 0.0, badSxx = 0.0, badSyy = 0.0;
    uint64_t badN = 0, liveN = 0;


    // Strided. A full 8.3 M pixel scan per dump costs more than it tells us -
    // every 4th pixel in each direction is 500k samples, which settles any of
    // these statistics to more digits than the question needs.
    for (uint32_t y = 0; y < m.h; y += 4) {
        for (uint32_t x = 0; x < m.w; x += 4) {
            size_t i = ((size_t)y * m.w + x) * kMvHalves;
            float vx = velHalfToFloat(px[i]);
            float vy = velHalfToFloat(px[i + 1]);


            if (vx != vx || vy != vy) { ++nan; ++n; continue; }
            double mag = sqrt((double)vx * vx + (double)vy * vy);
            // Sky and anything else with no geometry behind it writes zero, and
            // at 4K that can be most of the frame - enough to drag a median to
            // zero regardless of how the rest of the field moved. A pixel with
            // no surface in it carries no motion, so it is not a sample.
            if (mag > 0.0) {
                axX.push_back(fabsf(vx));
                axY.push_back(fabsf(vy));

                // ---- EPIPOLAR RESIDUAL, not a solved depth.
                //
                // The first version of this solved d from the x channel and
                // predicted y from it. That is ill-posed exactly where it
                // matters: under a pure rotation the image of a ray does not
                // depend on d at all, the numerator M[12] - Xp*M[15] goes to
                // zero with the translation, and d comes out 0/0. It duly
                // rejected 518400 samples of 518400 on the rotation frames and
                // solved 0.7 m on the rest.
                //
                // What is always well posed is the CURVE. As d sweeps 0 to
                // infinity the previous position of this pixel traces the image
                // of a ray, which is a straight line - the epipolar line. So
                // the question is not "what depth" but "does the measured
                // previous position lie on the line at all", and that is a
                // perpendicular distance. Under translation the line is long
                // and the test is sharp; under pure rotation it collapses to a
                // single point and the test degrades gracefully into the
                // fixed-depth comparison, which is the right behaviour rather
                // than a division by zero.
                //
                // Two points fix the line: d -> infinity, where the constant
                // column drops out, and d = 1 m.
                const double u    = 2.0 * ((double)x + 0.5) / (double)m.w - 1.0;
                const double vTop = 1.0 - 2.0 * ((double)y + 0.5) / (double)m.h;
                const double hw = 0.5 * (double)m.w, hh = 0.5 * (double)m.h;
                (void)sxP; (void)syP;
                for (int c = 0; c < 4; ++c) {
                    const double v   = (c & 1) ? -vTop : vTop;
                    const double dyN = (c & 2) ?  2.0 * (double)vy
                                               : -2.0 * (double)vy;
                    // THE MATRIX TAKES CLIP, NOT VIEW.
                    //
                    // viewToPrevClip = prevProj * relRot * clipToView, and that
                    // trailing clipToView is what converts clip to view - so
                    // the matrix expects exactly what the shader hands it,
                    // (x_clip, y_clip, w_clip, 1). Building a view-space point
                    // here divided by the projection scales a second time and
                    // negated z on top, and the residual came out 500 to 2600 px
                    // with no sign convention winning - the signature of a
                    // structurally wrong probe rather than a wrong field.
                    //
                    // For the pixel at NDC (u, v) at clip depth d, x_clip = u*d
                    // and y_clip = v*d, so every clip component is d*A + M[3][k]
                    // with A = M[0][k]*u + M[1][k]*v + M[2][k].
                    //
                    // Checked against the identity case: with no camera motion
                    // M reduces to columns (1,0,0,0), (0,1,0,0), (0,0,1,1),
                    // (0,0,m14,0), giving Ax = u, Aw = 1 and x/w = u. No
                    // motion, which is the only answer that can be right.
                    const double Ax = RJ[0]*u + RJ[4]*v + RJ[8];
                    const double Ay = RJ[1]*u + RJ[5]*v + RJ[9];
                    const double Aw = RJ[3]*u + RJ[7]*v + RJ[11];
                    if (fabs(Aw) < 1e-12) continue;
                    const double ex = (Ax / Aw) * hw, ey = (Ay / Aw) * hh;
                    const double w1 = Aw + (double)RJ[15];
                    if (fabs(w1) < 1e-12) continue;
                    const double fx = ((Ax + (double)RJ[12]) / w1) * hw;
                    const double fy = ((Ay + (double)RJ[13]) / w1) * hh;
                    const double mx = (u + 2.0 * (double)vx) * hw;
                    const double my = (v + dyN) * hh;
                    double lx = fx - ex, ly = fy - ey;
                    const double len = sqrt(lx*lx + ly*ly);
                    double r;
                    if (len < 1e-4) {
                        // Pure rotation: the line is a point.
                        r = sqrt((mx-ex)*(mx-ex) + (my-ey)*(my-ey));
                    } else {
                        lx /= len; ly /= len;
                        r = fabs((mx - ex) * ly - (my - ey) * lx);
                    }
                    resid[c].push_back((float)r);
                    // ---- CONVENTION 2 IS v+,dy+, AND IT IS THE RIGHT ONE.
                    //
                    // Measured, on every phase of a full self-test run:
                    //
                    //   phase 7  v+,dy-=34.792 v-,dy-=30.831 v+,dy+=0.003 v-,dy+=7.183
                    //   phase 7  v+,dy-=19.360 v-,dy-=22.297 v+,dy+=0.002 v-,dy+=4.328
                    //   phase 0  v+,dy-=0.711  v-,dy-=0.057  v+,dy+=0.005 v-,dy+=0.748
                    //   phase 0  v+,dy-=1.897  v-,dy-=0.044  v+,dy+=0.008 v-,dy+=1.922
                    //
                    // v+,dy+ wins every line by two to four orders of
                    // magnitude, and it wins hardest on phase 7 where the
                    // epipolar line is 27 px long and the test has the most to
                    // discriminate with. The shader therefore writes velocity
                    // with Y already in the NDC-up sense and needs no flip.
                    //
                    // The tail statistics were being accumulated on convention
                    // 0 - v+,dy-, one of the wrong ones - which is why they
                    // reported 95% of moving pixels missing by more than a
                    // pixel while the median for the correct convention was
                    // 0.003 px. They read the wrong column, not a broken field.
                    if (c == 2) {
                        solvedD.push_back((float)len);
                        ++liveN;
                        if (r > 1.0) {
                            ++badN;
                            badSx += (double)x; badSy += (double)y;
                            badSxx += (double)x * x; badSyy += (double)y * y;
                        }
                    }
                }
            }
            if (mag == 0.0) ++zero;
            if (mag > maxMag) maxMag = mag;
            sum += mag;
            mags.push_back((float)mag);
            if (x >= cx0 && x < cx1 && y >= cy0 && y < cy1) {
                cSum += mag; cSx += vx; cSy += vy; ++cN;
                // Magnitudes as well as the signed sums. The signed mean is
                // what tells sign conventions apart, but it is the wrong thing
                // to hold against a predicted displacement: a minority of
                // pixels moving the other way - a prop disc, an animated panel
                // element - drags it toward zero, and the verdict then reads
                // systematically LOW while never reading high. That is exactly
                // the residue left after the depth convention was fixed:
                // 0.70, 0.84, 0.85, 0.88, 0.90, and nothing above 1.01.
                cAxX.push_back(fabsf(vx));
                cAxY.push_back(fabsf(vy));
            }
            ++n;
        }
    }
    if (!n) return;

    {
        auto med = [](std::vector<float> &v) -> double {
            if (v.empty()) return -1.0;
            size_t k = v.size() / 2;
            std::nth_element(v.begin(), v.begin() + k, v.end());
            return (double)v[k];
        };
        static const char *kConv[4] = { "v+,dy-", "v-,dy-", "v+,dy+", "v-,dy+" };
        double best = 1e30; int bestC = -1;
        double mr[4];
        for (int c = 0; c < 4; ++c) {
            mr[c] = med(resid[c]);
            if (mr[c] >= 0.0 && mr[c] < best) { best = mr[c]; bestC = c; }
        }
        const double medD = med(solvedD);
        // p95 as well as the median, and the PHASE on the line.
        //
        // Pairing this against the verdict line by position in the log does not
        // work - the two do not fire on the same set of frames - and reading a
        // translation phase as a rotation one has already cost an analysis
        // cycle once on this project. The line carries its own phase now.
        //
        // The median alone would hide the case that matters most: a field that
        // is right almost everywhere and wrong on the near-field geometry is
        // exactly what a consumer ghosts on.
        std::vector<float> &rb = resid[bestC >= 0 ? bestC : 0];
        double r95 = -1.0;
        if (!rb.empty()) {
            size_t k95 = (size_t)(rb.size() * 0.95);
            if (k95 >= rb.size()) k95 = rb.size() - 1;
            std::nth_element(rb.begin(), rb.begin() + k95, rb.end());
            r95 = (double)rb[k95];
        }
        trace("MV EPI: phase=%d | distance from the measured previous position "
              "to the epipolar line, median per sign convention: %s=%.3f "
              "%s=%.3f %s=%.3f %s=%.3f px | best=%s p95=%.3f px | median line "
              "length=%.2f px over %llu of %llu samples (a short line means the "
              "frame is nearly a pure rotation, where depth cannot be recovered "
              "and the residual reduces to the fixed-depth comparison)",
              m.dumpPhase,
              kConv[0], mr[0], kConv[1], mr[1], kConv[2], mr[2], kConv[3], mr[3],
              bestC >= 0 ? kConv[bestC] : "none", r95, medD,
              (unsigned long long)solvedD.size(), (unsigned long long)n);
        if (badN) {
            const double bx = badSx / (double)badN, by = badSy / (double)badN;
            double vxx = badSxx / (double)badN - bx * bx;
            double vyy = badSyy / (double)badN - by * by;
            if (vxx < 0.0) vxx = 0.0;
            if (vyy < 0.0) vyy = 0.0;
            trace("MV TAIL: %llu of %llu moving pixels miss the epipolar line by "
                  "more than 1 px (%.2f%%), centred at (%.0f, %.0f) of %ux%u "
                  "with a spread of %.0f x %.0f px - clustered low and central "
                  "is the propeller and the panel, which are not rigid with the "
                  "camera and cannot be reprojected from it; spread over the "
                  "whole frame would mean the matrix is wrong",
                  (unsigned long long)badN, (unsigned long long)liveN,
                  100.0 * (double)badN / (double)(liveN ? liveN : 1),
                  bx, by, m.w, m.h, sqrt(vxx), sqrt(vyy));
        }
    }

    // Reported in PIXELS. The shader writes UV, but a UV number is unreadable
    // without knowing the resolution, and every prediction worth checking - the
    // self-test's 8.77 px, "a pixel of jitter" - is stated in pixels.
    // COMPARE AGAINST THE DEPTH-DERIVED FIELD, which is independently verified.
    //
    // Every attempt to judge this field against my own arithmetic has been
    // wrong in both directions - first calling a correct magnitude broken by
    // predicting for the far plane, then nearly "fixing" a systematic error
    // with an outlier guard. The depth field needs no prediction: it was
    // measured at 8.78 px against 8.77 predicted under pure yaw, it is computed
    // from completely different inputs, and it is running right now.
    //
    // Two independent derivations of the same quantity agreeing is far stronger
    // evidence than either one matching a number I worked out by hand. Where
    // they disagree, the RATIO says what kind of error it is: a constant factor
    // is a scale or convention problem, a varying one is geometry-dependent and
    // points at the reprojection itself.
    // The depth-derived cross-check is gone with the pass that produced it.
    //
    // It compared the injected field against velocity reconstructed from depth,
    // on the sound principle that two independent derivations agreeing is
    // stronger evidence than either matching a hand-worked number. There is no
    // second derivation any more - the depth pass was removed - so this reports
    // nothing rather than pretending to a comparison it cannot make.
    double refSum = 0.0, refSx = 0.0, refSy = 0.0; uint64_t refN = 0;

    double med = 0.0, p95 = 0.0;
    if (!mags.empty()) {
        std::sort(mags.begin(), mags.end());
        med = mags[mags.size() / 2];
        p95 = mags[(size_t)(mags.size() * 0.95)];
    }

    // MEDIAN FIRST. It is the number that decides what is wrong:
    //   median sane, mean huge   -> a few fragments dividing by a near-zero w
    //   median == mean, both off -> systematic; the ratio IS the correction
    //   both near zero when still -> the field is behaving
    double centre = cN ? (cSum / (double)cN) * m.w : 0.0;
    double refCentre = refN ? (refSum / (double)refN) * m.w : 0.0;
    trace("MV MEASURED: CENTRE=%.3f px  DEPTH-FIELD=%.3f px  ratio=%.3f | "
          "median=%.3f mean=%.3f p95=%.3f max=%.3f px "
          "zero=%.1f%% NaN=%llu camera moved %.3f m",
          centre, refCentre,
          refCentre > 0.001 ? centre / refCentre : 0.0,
          med * m.w, (sum / (double)n) * m.w, p95 * m.w, maxMag * m.w,
          100.0 * (double)zero / (double)n,
          (unsigned long long)nan, camMoved);

    // ---- THE ACCEPTANCE GATE.
    //
    // The self test yaws and pitches the camera by a known amount and publishes
    // the pixel displacement that motion must produce. Comparing the measured
    // field against it turns "does this look right" into a number, which is the
    // whole basis on which this project calls the vectors correct.
    //
    // A ratio of 1 is the target. Sign is not meaningful here: the vectors point
    // backwards by convention - from a pixel to where it was - while the
    // expectation is a magnitude.
    // ---- PER-PIXEL VERIFICATION, when .y carries w instead of velocity.y.
    //
    // Every comparison so far has held a REGION statistic against a CENTRE-RAY
    // prediction at an ASSUMED depth, and each of those three has been wrong at
    // least once. This holds neither: for each sampled pixel it takes the
    // pixel's own screen position and its own w, forms the clip position the
    // vertex shader must have produced, pushes it through the very matrix that
    // was pushed, and compares the result against the velocity that pixel
    // actually contains.
    //
    // If these agree, the shader is faithfully applying the matrix and any
    // remaining disagreement is in how the CPU-side prediction was being
    // formed. If they do not, the field is not f(matrix, depth) and the
    // identity test - which showed 100% zeros - was answering a narrower
    // question than it appeared to.
    if (g_mvDebugDepth && m.dumpProj[11] != 0.0f) {
        const double m10 = m.dumpProj[10], m11 = m.dumpProj[11], m14 = m.dumpProj[14];
        // A MEAN OF RATIOS IS THE WRONG STATISTIC HERE, and the first version
        // of this used one: predVx goes to near zero wherever the matrix
        // predicts almost no motion, and dividing by 0.004 px produces
        // thousands without meaning anything. The first run duly reported a
        // mean of 954 and a worst of 20384, which says only that some
        // denominators were small.
        //
        // The median is immune to that, and the absolute pixel error says
        // whether any disagreement actually matters.
        std::vector<float> ratios;
        double sumAbsErr = 0.0;
        uint64_t nCmp = 0;
        for (uint32_t y = 0; y < m.h; y += 64) {
            for (uint32_t x = 0; x < m.w; x += 64) {
                const size_t i = ((size_t)y * m.w + x) * kMvHalves;
                const double vx = velHalfToFloat(px[i]);
                const double w  = velHalfToFloat(px[i + 1]);
                if (!(w > 0.05) || vx != vx) continue;      // no geometry here

                // Screen position -> NDC. The viewport is Y-flipped, but only
                // x is being compared so that does not enter.
                const double ndcX = ((double)x + 0.5) / (double)m.w * 2.0 - 1.0;
                const double ndcY = ((double)y + 0.5) / (double)m.h * 2.0 - 1.0;
                const double ndcZ = m10 / m11 - m14 / (m11 * w);

                // clip = w * (ndc, 1), then prev = reproj * clip.
                const double cx4 = ndcX * w, cy4 = ndcY * w, cz4 = ndcZ * w;
                const double pxv = reproj[0]*cx4 + reproj[4]*cy4 + reproj[8]*cz4  + reproj[12]*w;
                const double pwv = reproj[3]*cx4 + reproj[7]*cy4 + reproj[11]*cz4 + reproj[15]*w;
                if (fabs(pwv) < 1e-9) continue;

                const double predVx = (ndcX - pxv / pwv) * 0.5;
                sumAbsErr += fabs(vx - predVx) * (double)m.w;
                ++nCmp;
                // Only pixels the matrix says should MOVE can carry a ratio.
                if (fabs(predVx) * (double)m.w > 1.0)
                    ratios.push_back((float)(vx / predVx));
            }
        }
        if (nCmp) {
            double medRatio = 0.0;
            if (!ratios.empty()) {
                std::nth_element(ratios.begin(), ratios.begin() + ratios.size()/2, ratios.end());
                medRatio = ratios[ratios.size()/2];
            }
            trace("MV PERPIXEL: %llu pixels, %llu movers - median "
                  "measured/predicted %.4f, mean |error| %.3f px "
                  "(1.0 and 0.0 mean the shader applied the pushed matrix)",
                  (unsigned long long)nCmp, (unsigned long long)ratios.size(),
                  medRatio, sumAbsErr / (double)nCmp);
        }
    }

    // ---- RAW MODE REPORTS BOTH CHANNELS, EXPLICITLY.
    //
    // The verdict line picks its axis from the phase - axY for pitch, axX
    // otherwise - which is right for a velocity but wrong for the raw probe,
    // where .x is prev.w and .y is curr.w. Reading pitch rows as prev.w cost a
    // full analysis cycle. These two must be EQUAL for any rigid camera motion,
    // so they are printed side by side with nothing deciding which is which.
    if (g_mvRawMode && !axX.empty() && !axY.empty()) {
        auto q = [](std::vector<float> &v, double f) -> double {
            size_t k = (size_t)(f * (double)(v.size() - 1));
            std::nth_element(v.begin(), v.begin() + k, v.end());
            return v[k];
        };
        trace("MV RAW: prev.w  p05=%.3f med=%.3f p95=%.3f m   |   "
              "curr.w  p05=%.3f med=%.3f p95=%.3f m   (they must match)",
              q(axX, 0.05), q(axX, 0.50), q(axX, 0.95),
              q(axY, 0.05), q(axY, 0.50), q(axY, 0.95));
    }

    // Gated on the PHASE alone, not on the plugin's expectedPx.
    //
    // The plugin only fills expectedPx for the rotation phases - it is zero for
    // HOLD, TRANSLATE and HEADMOVE - so gating on it silently discarded every
    // sample from the three phases that test the things rotation cannot: that a
    // still camera produces a still field, and that translation produces
    // PARALLAX rather than a uniform shift. Those are the samples that prove
    // the field is depth-aware, and none of them were ever being printed.
    //
    // Nothing here needs expectedPx any more; the matrix is the yardstick.
    if (selfTestPhase != 0) {
        // ---- COMPARE THE MATCHING AXIS, NOT THE MAGNITUDE.
        //
        // `centre` above is sqrt(vx^2 + vy^2). The expectation for a yaw is a
        // purely HORIZONTAL displacement, and a magnitude can only ever be
        // greater than or equal to |vx| - so any vertical content inflates the
        // ratio and it can never read below 1. That alone would explain a
        // stable reading of about 1.25.
        //
        // Yaw phases (3, 4) are horizontal; pitch (5) is vertical.
        auto quantile = [](std::vector<float> &v, double q) -> double {
            if (v.empty()) return 0.0;
            size_t k = (size_t)(q * (double)(v.size() - 1));
            std::nth_element(v.begin(), v.begin() + k, v.end());
            return v[k];
        };
        const double cxPx = quantile(axX, 0.5) * m.w;
        const double cyPx = quantile(axY, 0.5) * m.h;

        // THE DISTANT TAIL IS THE TEST.
        //
        // A single number cannot separate "the vectors are wrong" from "the
        // vectors are right and the frame contains a range of depths". Under a
        // camera that both rotates and translates, near geometry genuinely
        // moves further than far geometry, so a spread is CORRECT behaviour and
        // a median lands wherever the depth histogram happens to put it.
        //
        // What is not free to vary is the far end. Salzburg has mountains at
        // effectively infinite distance, so the least-moving pixels in the
        // frame must agree with the matrix's own far prediction. If the low
        // quantile matches, the field is right and the spread is parallax; if
        // even the low quantile is a multiple of it, the field is wrong.
        const bool   vertical = (selfTestPhase == 5);   // ST_PITCH
        const double axisPx = vertical ? cyPx : cxPx;
        std::vector<float> &axis = vertical ? axY : axX;
        const double scale = vertical ? (double)m.h : (double)m.w;
        const double p05 = quantile(axis, 0.05) * scale;
        const double p25 = quantile(axis, 0.25) * scale;
        const double p75 = quantile(axis, 0.75) * scale;
        const double p95 = quantile(axis, 0.95) * scale;

        // Frames straight after a phase change are a camera JUMP, not the
        // steady motion being measured - they read tens of times too large and
        // say nothing about the convention.
        static int lastPhase = -1;
        static int settle    = 0;
        if (selfTestPhase != lastPhase) { lastPhase = selfTestPhase; settle = 3; }
        // WHAT THE MATRIX ITSELF PREDICTS.
        //
        // THE PLUGIN'S expectedPx IS NOT USED AND IS NOT PRINTED. It is the
        // angle between two view matrices sampled in the FLIGHT LOOP, turned
        // into pixels, and it disagrees with the matrix the shader was actually
        // handed - reading 13.15 px on frames where the reprojection encodes
        // 842. One of the two is describing a different pair of frames, and
        // until that is understood a number that looks authoritative and is
        // wrong is worse in a log than no number at all.
        //
        // `reproj` is what the SHADER was handed, so it is the yardstick. The
        // rest of this comment records why a second one existed: the cockpit
        // pass is pushed a body-frame matrix while this comparison is against
        // the world-frame
        // one - and when they do, "the vectors are wrong" and "the vectors are
        // right and the yardstick is wrong" look identical from one number.
        //
        // Down the centre ray, at BOTH ends of the depth range.
        //
        // Clip is w * (ndcX, ndcY, ndcZ, 1), and reproj is linear, so the w
        // cancels in the perspective divide: the centre ray at a given depth is
        // just (0, 0, ndcZ, 1). Column-major, as the shader consumes it, so
        // that product is column 3 plus ndcZ times column 2. The shader's own
        // subtraction against a centre NDC of zero is then -prevNDC * 0.5.
        //
        // Both ends, rather than picking one, because WHICH end is "infinity"
        // depends on the depth convention - and this projection is not
        // reverse-Z, so a first version that assumed z=0 was sampling the NEAR
        // plane and reading translation parallax instead of rotation.
        //
        // Reporting both is the better instrument anyway: under a pure rotation
        // the reprojection is depth-INDEPENDENT, so the two must agree. When
        // they diverge, the camera translated, and no single expected
        // displacement fits the frame.
        // ---- THE MATRIX IS VIEW-SPACE. FEED IT A VIEW-SPACE POINT.
        //
        // This took a CLIP-space depth and built col3 + ndcZ*col2, the image of
        // (0, 0, ndcZ, 1). That was right while the reprojection went clip to
        // clip. It has been wrong since the matrix became view-to-prev-clip,
        // and it is the whole of the far=9.5..17.3 scatter that the last two
        // rounds were spent chasing: the matrix never moved. MV ANGLE, reading
        // the same matrix correctly, gives 13.150 px on every line while the
        // trace angle and the plugin independently give 13.13-13.18.
        //
        // The shader hands the matrix (x_clip, y_clip, w_clip, 1) and clipToView
        // turns that into (x_view, y_view, z_view, 1) - it divides x and y by
        // the projection scales and sets clipToView[10] = -1, so z_view =
        // -w_clip, and w_clip is positive for anything in front of the eye.
        // Forward is therefore -z, and a point d metres down the centre ray is
        // (0, 0, -d, 1).
        //
        // At infinity the constant column drops out under the division, leaving
        // column 2 alone - which is exactly the correction just made to the
        // MV ANGLE probe, for exactly the same reason.
        auto predictAtDistance = [&](double d, double *px, double *py) {
            double x, y, w;
            if (d <= 0.0) {                       // d <= 0 means "at infinity"
                x = (double)reproj[8];
                y = (double)reproj[9];
                w = (double)reproj[11];
            } else {
                // +d, not -d. Same mistake as the epipolar probe had: the
                // matrix consumes (x_clip, y_clip, w_clip, 1) and w_clip is
                // positive in front of the eye, so the centre ray at distance d
                // is (0, 0, d, 1). Negating it put the sample behind the
                // camera, which is why near disagreed with far by up to 40%
                // while far - taken from column 2 alone, and so immune to the
                // sign - was steady and correct at 13.150 px.
                x = (double)reproj[12] + d * (double)reproj[8];
                y = (double)reproj[13] + d * (double)reproj[9];
                w = (double)reproj[15] + d * (double)reproj[11];
            }
            if (fabs(w) < 1e-12) { *px = *py = 0.0; return; }
            *px = fabs(0.5 * x / w) * m.w;
            *py = fabs(0.5 * y / w) * m.h;
        };
        // AT ONE METRE, NOT AT THE NEAR PLANE.
        //
        // The near plane here is 1.6 CM. A point there moves hundreds of pixels
        // for a camera wobble of well under a millimetre, so comparing it
        // against infinity flags translation that nothing visible cares about -
        // and the first version of this test duly rejected almost every frame
        // as "camera translated" while the sim was PAUSED.
        //
        // One metre is the real question: does something at arm.s length - the
        // instrument panel - move like the terrain does? If those two agree,
        // the frame is a pure rotation as far as anything being drawn is
        // concerned. With an infinite far plane the depth mapping is
        // ndcZ = 1 - near/d, so a metre is 1 - nearClip.
        // ---- THE DEPTH CONVENTION IS REVERSE-Z. MEASURED, NOT ASSUMED.
        //
        // Writing currClip.z/currClip.w into the field and reading its
        // distribution settles it. The fragments come back at
        //
        //     z_ndc = 0.0000009, 0.0223, 0.0262, 0.0433
        //
        // all clustered near ZERO. Under z_ndc = 1 - near/d that would put the
        // whole scene 1.6 cm from the eye. Under z_ndc = near/d it reads as
        // 17 km, 0.72 m, 0.62 m and 0.37 m - mountains, then the instrument
        // panel. That is the scene, so infinity is ndcZ = 0 and the near plane
        // is ndcZ = 1.
        //
        // Which means this predictor had it backwards from the day it was
        // written: "far" was being evaluated at the NEAR plane. The evidence was
        // already in the log - a pitch frame measured 190-197 px while
        // predictAt(0) said 195.4 - and it was read as a failure of the vectors
        // instead of as the convention being inverted.
        //
        // The plugin publishes reverseZ=0, so that flag is wrong too, and
        // nothing here trusts it.
        // ---- THE DEPTH CONVENTION, READ FROM THE PROJECTION.
        //
        // For a point at view depth d (forward negative, as X-Plane has it):
        //
        //     z_ndc(d) = m10/m11 - m14/(m11 * d)
        //
        // so infinity is m10/m11 and everything else follows. Measured here:
        // m10 = m11 = -1 and m14 = -0.01615, giving infinity at ndcZ = 1 and
        // the near plane at 0.
        //
        // THIS HAS NOW BEEN ASSUMED TWICE AND WRONG TWICE. First as
        // non-reverse-Z, then "corrected" to reverse-Z on the strength of a
        // depth histogram, which put infinity at 0 and had `far` sampling the
        // 1.6 cm near plane. The passing numbers survived only because
        // translation was negligible in those frames, so both ends gave the
        // rotation and the wrong end happened to agree. That is luck, not
        // correctness, and it is why this is derived rather than believed.
        const double m10 = m.dumpProj[10], m11 = m.dumpProj[11];
        const double m14 = m.dumpProj[14];
        const bool   haveProj  = fabs(m11) > 1e-12;
        const double infinityZ = haveProj ? m10 / m11 : 1.0;
        const double oneMetreZ = haveProj ? (m10 / m11 - m14 / m11) : 0.98;
        double nearX = 0.0, nearY = 0.0, farX = 0.0, farY = 0.0;
        if (reproj) {
            // One metre and infinity, both as DISTANCES now. The two clip
            // depths above are kept because the depth convention they encode is
            // still printed elsewhere, but they are no longer what the
            // predictor consumes.
            (void)oneMetreZ; (void)infinityZ;
            predictAtDistance(1.0, &nearX, &nearY);
            predictAtDistance(-1.0, &farX,  &farY);
        }
        const double predNear = vertical ? nearY : nearX;
        const double predFar  = vertical ? farY  : farX;

        if (settle > 0) { --settle; }
        else {
            // The verdict is read off the FAR end, not the median. p05 is the
            // least-moving twentieth of the frame, which is the most distant
            // geometry in it, and that is what the matrix's far prediction
            // describes.
            // COMPARE THE CENTRE AGAINST THE CENTRE-RAY PREDICTION.
            //
            // predFar is the displacement along the CENTRE ray. Over a 65 degree
            // field a rigid rotation does not move every pixel by that amount -
            // displacement grows off-axis - so a whole-frame quantile was never
            // the right thing to hold against it. p05 was chosen when the far
            // end was believed to be a lower bound; with the depth convention
            // corrected it is not.
            //
            // The centre region is what predFar describes, so that is what it
            // is compared with.
            std::vector<float> &cAxis = vertical ? cAxY : cAxX;
            const double centrePx = quantile(cAxis, 0.5)
                                  * (vertical ? (double)m.h : (double)m.w);
            // ---- REVERTED: judged against predFar again, and here is why.
            //
            // I switched this to the plugin's angle one run ago on the strength
            // of four consecutive yaw frames where the field and the plugin
            // agreed to better than 1% while predFar wandered. A fuller sample
            // says that reading was wrong, and the evidence is a correlation:
            //
            //     centre   far
            //      5.760  10.062
            //      7.766   7.062
            //     12.780  13.004
            //     17.996  14.770
            //     21.835  15.856
            //
            // centre and far rise and fall together; the plugin sits flat at
            // 13.15 through both the yaw and the pitch phase. Two numbers that
            // track each other are measuring one thing, and the flat one is
            // measuring another - so those four agreeing yaw frames were a
            // coincidence of the commanded rate matching the realised one, not
            // corroboration.
            //
            // The two describe different frame PAIRS. The plugin's angle comes
            // from world against prevWorld, consecutive FLIGHT LOOP samples
            // (plugin.cpp, selfTestExpectedPx). The layer's reprojection is
            // built across consecutive PRESENTS. Neither is wrong; they answer
            // different questions, and the one that governs whether the field
            // is correct is the present-paired one, because that is the pair
            // the shader reprojected between.
            //
            // So predFar is the judge. The plugin's figure stays in the log as
            // an independent witness, and its disagreement here is information
            // about loop-to-present pacing rather than a defect in the field.
            const double farRatio  = predFar > 1e-6 ? centrePx / predFar : 0.0;

            // ---- WHAT EACH PHASE PREDICTS. Magnitude at the centre is not a
            // test on its own: a field with the right size and the wrong sign is
            // useless to every consumer of it, and rotation phases cannot
            // exercise depth at all.
            //
            //   1 HOLD       camera still            -> the field must be ZERO
            //   2 YAW  3 YAWL 4 PITCH  rotation      -> centre == centre-ray
            //                                           prediction, and the SAME
            //                                           at every depth
            //   5 TRANSLATE  camera slides sideways  -> infinity must NOT move.
            //                                           Parallax only, so far
            //                                           goes to zero while near
            //                                           geometry moves a lot
            //
            // TRANSLATE is the one that proves the vectors are depth-aware
            // rather than a global offset that happens to fit a rotation.
            // The phase numbers come from the plugin's enum, which starts at
            // TAA_ST_OFF = 0. Naming them here rather than writing the integers
            // is not decoration: the first version of this switch had them off
            // by one and applied the TRANSLATION rule to the PITCH phase, so
            // every pitch row was failed for "moving infinity" while its ratio
            // read 1.002.
            enum { ST_OFF = 0, ST_SETTLE, ST_HOLD, ST_YAW, ST_YAWL,
                   ST_PITCH, ST_TRANSLATE, ST_HEADMOVE };

            // SIGN COHERENCE. Under any rigid camera motion over static scenery
            // the centre pixels all move the same way, so the SIGNED mean and
            // the median MAGNITUDE must agree. If they diverge, neighbouring
            // pixels disagree with each other - which is the defect the
            // predecessor spent weeks on and never measured properly.
            const double signedPx = (vertical ? fabs(cSy / (double)(cN ? cN : 1)) * m.h
                                              : fabs(cSx / (double)(cN ? cN : 1)) * m.w);
            const double coherence = centrePx > 1e-6 ? signedPx / centrePx : 1.0;

            // A VERDICT ONLY WHEN THE MATRIX SAYS PURE ROTATION.
            //
            // The expectation is depth-independent only if the camera did not
            // translate; then one number describes the whole frame and p05 must
            // equal it. Once the camera translates, near geometry genuinely
            // moves further than far geometry, the field spreads smoothly from
            // below the prediction to many times it, and NO single number is
            // right - so reporting a ratio there is reporting noise with three
            // decimal places on it.
            //
            // near and far are the same rotation evaluated at the two ends of
            // the depth range, so their agreement IS the purity test, and it
            // comes from the matrix the shader was handed rather than from an
            // assumption about what the aeroplane was doing.
            const double purity = predFar > 1e-6 ? predNear / predFar : 0.0;
            const bool   pure   = purity > 0.95 && purity < 1.05;
            const char *verdict = "";
            switch (selfTestPhase) {
            case ST_SETTLE:
            case ST_HOLD:
                verdict = (centrePx < 0.05) ? "  <- CORRECT (still)"
                                            : "  <- WRONG (should be zero)";
                break;
            case ST_TRANSLATE: {
                // The camera translates FORWARD here - measured, cam z goes
                // -33869.9 to -33864.1 while x goes -237.6 to -235.2 at
                // 0.348 m per frame.
                //
                // Forward motion makes the centre of the screen the FOCUS OF
                // EXPANSION: the one point that does not move, at any depth.
                // The matrix says so plainly, far=8.090 against near=8.084,
                // both near zero - so a ratio against the centre-ray prediction
                // is degenerate and cannot judge this phase. The first version
                // of this test did exactly that and failed a phase the vectors
                // were passing.
                //
                // The real signature of radial flow is a large magnitude with a
                // signed mean near zero, because pixels either side of centre
                // move in OPPOSITE directions. That is what coherence measures,
                // and it reads 0.02 to 0.09 here against 1.00 for every
                // rotation - so it is the assertion.
                const bool radial = coherence < 0.30 && centrePx > 1.0;
                verdict = radial ? "  <- CORRECT (radial parallax from forward motion)"
                                 : "  <- WRONG (forward translation should be radial)";
                break;
            }
            default:   // ST_YAW, ST_YAWL, ST_PITCH, ST_HEADMOVE
                // A rigid-camera prediction only describes pixels whose motion
                // comes from the camera. INDEPENDENTLY MOVING GEOMETRY does not
                // obey it, and a Cirrus has a large piece of it filling the
                // middle of the view: the propeller. Blades either side of the
                // hub move opposite ways at a rate set by engine RPM, so a
                // centre region they dominate reads as a large magnitude with a
                // signed mean near zero.
                //
                // That is precisely the failing signature, and it is the same
                // with jitter on and off: centre pinned near 76 px whatever the
                // matrix says, coherence 0.03 to 0.15. Every PASS sits at 0.76
                // to 1.01. The vectors are right in those frames - a spinning
                // prop must produce large mixed vectors - so the honest verdict
                // is that the frame does not test what this assertion asks.
                if (coherence < 0.60) {
                    verdict = "  (moving geometry in centre - not a rigid-camera frame)";
                } else {
                    verdict = (farRatio > 0.95 && farRatio < 1.05) ? "  <- CORRECT"
                                                                   : "  <- WRONG";
                }
            }

            trace("MV RATIO: phase=%d %s  centre=%.3f | p05=%.3f p25=%.3f med=%.3f p75=%.3f "
                  "p95=%.3f px | far=%.3f near=%.3f  ->  ratio=%.3f%s "
                  "coherence=%.2f plugin=%.3f (frame %llu recorded / %llu now)",
                  selfTestPhase, vertical ? "pitch" : "yaw  ",
                  centrePx, p05, p25, axisPx, p75, p95, predFar, predNear, farRatio,
                  verdict,
                  coherence, expectedPx,
                  (unsigned long long)m.dumpShareFrame,
                  (unsigned long long)nowShareFrame);
            // `plugin` is the plugin's own angle-based estimate, printed as a
            // CROSS-CHECK and never used for a verdict. The rates are measured
            // at exactly 1:1 - 3000 presents, 3000 flight loops, worst gap 1 -
            // so it and the matrix describe the SAME interval and any
            // disagreement between them is arithmetic, not sampling.
            (void)pure;
            // THE MATRIX ITSELF, on the frames that fail.
            //
            // Every derived number has now been checked and every one of them
            // says the same thing: one step. The field says twenty-six. One of
            // those two readings is of something other than what it claims, and
            // the sixteen floats are the only place left that cannot be
            // paraphrasing. A reprojection for a small rotation about Y is
            // near-identity with a small [8] and [2]; anything else here is the
            // answer.
            // During SETTLE and HOLD the camera is still, so the reprojection
            // must be the IDENTITY to within noise. That is the one frame where
            // the correct matrix is known a priori, which makes it the only
            // place the construction can be checked without trusting any other
            // measurement. If it is not identity here, nothing downstream can
            // be right - and the per-pixel test reporting 4.9 to 7.4 px of
            // error on exactly these frames, where the field is measurably
            // zero, says one of the two is wrong.
            if (selfTestPhase == ST_SETTLE || selfTestPhase == ST_HOLD ||
                farRatio > 3.0 || farRatio < 0.5) {
                trace("  matrix: [%.5f %.5f %.5f %.5f][%.5f %.5f %.5f %.5f]"
                      "[%.5f %.5f %.5f %.5f][%.5f %.5f %.5f %.5f]",
                      reproj[0], reproj[1], reproj[2], reproj[3],
                      reproj[4], reproj[5], reproj[6], reproj[7],
                      reproj[8], reproj[9], reproj[10], reproj[11],
                      reproj[12], reproj[13], reproj[14], reproj[15]);
            }
            (void)centre;
        }
    }


    // The direction check, stated so it cannot be misread. Signed means over
    // the same centre region, in pixels, for both fields side by side - and
    // then the verdict spelled out rather than left as four numbers to compare
    // by eye at two in the morning.
    if (refN && cN) {
        double ix = (cSx / (double)cN) * m.w, iy = (cSy / (double)cN) * m.h;
        double dx = (refSx / (double)refN) * m.w, dy = (refSy / (double)refN) * m.h;
        // A threshold, because near-zero motion has no meaningful sign. Below
        // a tenth of a pixel of mean displacement the comparison would report
        // a disagreement from noise alone.
        const double kMin = 0.1;
        const char *vx = (fabs(ix) < kMin || fabs(dx) < kMin) ? "too small to say"
                       : ((ix < 0) == (dx < 0) ? "AGREE" : "OPPOSITE");
        const char *vy = (fabs(iy) < kMin || fabs(dy) < kMin) ? "too small to say"
                       : ((iy < 0) == (dy < 0) ? "AGREE" : "OPPOSITE");
        trace("  direction: injected (%+.3f, %+.3f) px  depth-field "
              "(%+.3f, %+.3f) px  ->  X %s, Y %s",
              ix, iy, dx, dy, vx, vy);
    }

    // The matrix the shaders were handed, printed alongside what it produced.
    //
    // With the camera nearly still this must be close to the identity - if it
    // is not, no amount of shader arithmetic will give sensible vectors, and
    // that is checkable here rather than by another flight. Two rows are enough
    // to see identity, scale or transposition.
    if (!reproj) return;
    // ALL SIXTEEN, in storage order.
    //
    // Rows 0 and 1 alone were useless for the question that matters. A
    // near-identity matrix looks near-identity transposed, so they read as
    // healthy either way - while the measured centre motion was 130x the value
    // those same rows predict. That factor is what a TRANSPOSE does: the
    // perspective row becomes a column, prevClip.w comes out wrong, and the
    // divide inflates everything.
    //
    // Elements 12..15 settle it. Stored column-major, a projection-like matrix
    // has its perspective term at [11] and [15] near zero; stored row-major it
    // sits at [14] and [15]. Whichever of those is non-trivial names the layout,
    // and the shader's ColMajor decoration has to agree with it.
    const float *R = reproj;
    trace("  uReproj [%.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f | "
          "%.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f]",
          R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7],
          R[8], R[9], R[10], R[11], R[12], R[13], R[14], R[15]);
}
