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
};

static MvTarget g_mv;
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
                             const float *reproj, float expectedPx, int phase)
{
    MvTarget &m = g_mv;
    if (!m.ready || !m.readbackPtr || !m.wantDump) return;
    m.wantDump = false;

    if (reproj) memcpy(m.dumpReproj, reproj, sizeof(m.dumpReproj));
    m.dumpExpectedPx = expectedPx;
    m.dumpPhase      = phase;

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

    m.dumpPending = 2;
}

// Measure it. Deliberately the SAME statistics the depth-derived dump prints,
// so the two fields can be compared line for line rather than by impression.
static void mvReport(double camMoved)
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
            }
            if (mag == 0.0) ++zero;
            if (mag > maxMag) maxMag = mag;
            sum += mag;
            mags.push_back((float)mag);
            if (x >= cx0 && x < cx1 && y >= cy0 && y < cy1) {
                cSum += mag; cSx += vx; cSy += vy; ++cN;
            }
            ++n;
        }
    }
    if (!n) return;

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
    if (selfTestPhase != 0 && expectedPx > 0.01) {
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
        const bool   vertical = (selfTestPhase == 5);
        const double axisPx = vertical ? cyPx : cxPx;
        std::vector<float> &axis = vertical ? axY : axX;
        const double scale = vertical ? (double)m.h : (double)m.w;
        const double p05 = quantile(axis, 0.05) * scale;
        const double p25 = quantile(axis, 0.25) * scale;
        const double p75 = quantile(axis, 0.75) * scale;
        const double p95 = quantile(axis, 0.95) * scale;

        const double ratio  = axisPx / (double)expectedPx;

        // Frames straight after a phase change are a camera JUMP, not the
        // steady motion being measured - they read tens of times too large and
        // say nothing about the convention.
        static int lastPhase = -1;
        static int settle    = 0;
        if (selfTestPhase != lastPhase) { lastPhase = selfTestPhase; settle = 3; }
        // WHAT THE MATRIX ITSELF PREDICTS, so the column can be split.
        //
        // `expectedPx` comes from the PLUGIN: it is the angle between the two
        // view matrices, turned into pixels. `reproj` is what the SHADER was
        // actually handed. They can disagree - the cockpit pass is pushed a
        // body-frame matrix while this comparison is against the world-frame
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
        auto predictAt = [&](double ndcZ, double *px, double *py) {
            const double x = (double)reproj[12] + ndcZ * (double)reproj[8];
            const double y = (double)reproj[13] + ndcZ * (double)reproj[9];
            const double w = (double)reproj[15] + ndcZ * (double)reproj[11];
            if (fabs(w) < 1e-12) { *px = *py = 0.0; return; }
            *px = fabs(0.5 * x / w) * m.w;
            *py = fabs(0.5 * y / w) * m.h;
        };
        double nearX = 0.0, nearY = 0.0, farX = 0.0, farY = 0.0;
        if (reproj) {
            predictAt(0.0, &nearX, &nearY);
            predictAt(1.0, &farX,  &farY);
        }
        const double predNear = vertical ? nearY : nearX;
        const double predFar  = vertical ? farY  : farX;

        if (settle > 0) { --settle; }
        else {
            // The verdict is read off the FAR end, not the median. p05 is the
            // least-moving twentieth of the frame, which is the most distant
            // geometry in it, and that is what the matrix's far prediction
            // describes.
            const double farRatio = predFar > 1e-6 ? p05 / predFar : 0.0;
            trace("MV RATIO: phase=%d %s  p05=%.3f p25=%.3f med=%.3f p75=%.3f "
                  "p95=%.3f px | matrix far=%.3f px  ->  p05/far=%.3f%s "
                  "(expected=%.3f, near=%.3f)",
                  selfTestPhase, vertical ? "pitch" : "yaw  ",
                  p05, p25, axisPx, p75, p95, predFar, farRatio,
                  (farRatio > 0.95 && farRatio < 1.05) ? "  <- CORRECT" : "",
                  expectedPx, predNear);
            (void)ratio; (void)centre;
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
