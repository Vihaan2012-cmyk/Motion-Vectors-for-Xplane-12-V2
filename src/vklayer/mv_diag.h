#pragma once
//
// One report that tests every standing hypothesis at once.
//
// This exists because the session that produced it proposed and killed five
// explanations in a row, each of which cost a build, a launch and several
// minutes, and three of which were killed by a measurement that could have been
// taken at the same time as the first. Testing them one at a time is what made
// that expensive, not the difficulty of any individual test.
//
// So: every hypothesis, every run, written to a file as plain prose and numbers.
// No greping for a needle - the whole picture at once, so a reading that
// contradicts another is visible immediately rather than three runs later.
//
// The hypotheses, numbered as they were listed:
//
//   A. the near-field select fires when it should not
//      1 block[18] non-zero at push time      2 patched binds with no push
//      3 bodyReprojValid true in this view
//   B. the reprojection is genuinely invalid very close in
//      4 prev.w <= 0 (point behind the previous camera)
//      5 geometry nearer than the near plane's usable range
//      6 half-float quantisation at large UV velocities
//   C. the geometry itself moved, so a camera-only field cannot know
//      7 terrain LOD vertex morphing        8 animated vegetation
//      9 water                             10 decals   11 aircraft shadow
//   D. more than one projection or pass in play
//      12 per-draw projection differs      13 a second pass writes the target
//      14 a pass with different viewport/depth state
//   E. the measurement rather than the field
//      15 epipolar line extrapolated beyond its construction points
//      16 the mag > 0 filter skewing which pixels are sampled
//      17 disocclusion - the previous position off-screen

#include <stdio.h>
#include <math.h>
#include <vector>
#include <string>
#include <map>

// Layer-side counters the report needs. Defined here, incremented in layer.cpp.
static uint64_t g_diagPatchedBindsNoPush = 0;   // hypothesis 2
static float    g_diagLastBlock18        = -1.0f; // hypothesis 1
static int      g_diagBodyValid          = -1;    // hypothesis 3
static uint32_t g_diagQualifyingPasses   = 0;     // hypothesis 13
static uint32_t g_diagBoundPasses        = 0;     // hypothesis 13
static float    g_diagDcMetres           = -1.0f; // camera motion this frame

struct MvDiagInput {
    const uint16_t *px;
    uint32_t w, h, halves;
    const float *reproj;     // the matrix the draws were given
    const float *proj;       // the projection at dump time
    int viewType, phase;
    double medianAbsPx;      // what the epipolar test concluded
    double medianFrac;
};

static void mvWriteDiagnostic(const MvDiagInput &in)
{
    const char *dir = getenv("TAA_MV_DIAG");
    if (!dir) return;
    static int nWritten = 0;
    if (nWritten >= 4) return;

    char path[512];
    snprintf(path, sizeof(path), "%s/mv_diag_%d.txt", dir, nWritten);
    FILE *f = fopen(path, "w");
    if (!f) return;
    ++nWritten;

    const float *M = in.reproj;
    const double hw = 0.5 * (double)in.w, hh = 0.5 * (double)in.h;

    // ---- Accumulators, all gathered in ONE pass over the readback.
    uint64_t nTotal = 0, nZero = 0, nNaN = 0, nMoving = 0;
    uint64_t nFlipMatch = 0, nGoodPix = 0, nBadPix = 0;
    uint64_t nPrevOffScreen = 0, nPrevWNeg = 0;
    double   maxAbsUv = 0.0;
    // Residual binned by measured speed - a proxy for "how near", since near
    // geometry moves furthest for a given camera motion.
    const int kBins = 6;
    const double binEdge[kBins] = { 1.0, 4.0, 16.0, 64.0, 256.0, 1e30 };
    double   binResSum[kBins] = {0}; uint64_t binResN[kBins] = {0};
    // Residual binned by screen row, in eighths.
    double   rowResSum[8] = {0}; uint64_t rowResN[8] = {0};
    // Fit prev.y = k * curr.y over the bad pixels: the flip predicts k = -1,
    // a scale error predicts a constant k that is not 1, and a correct field
    // predicts k = 1.
    double   kSum = 0.0; uint64_t kN = 0;

    // ---- HYPOTHESIS 18: WHICH MATRIX DOES THE FIELD ACTUALLY MATCH?
    //
    // The surviving description is a direction error on near, fast pixels that
    // scales with camera translation. That is exactly what a field built from
    // ROTATION ALONE looks like: far geometry is unaffected because translation
    // contributes nothing there, and near geometry is wrong by precisely the
    // translation parallax - measured at about half the flow.
    //
    // So stop reading code and ask the pixels. Predict each one three ways and
    // count which prediction it lands nearest:
    //
    //   full        the matrix as pushed
    //   rotOnly     the same matrix with its translation column zeroed
    //   transOnly   identity rotation, translation kept
    //
    // If near pixels match rotOnly, the translation is not reaching the shader
    // whatever the matrix contains.
    float Mrot[16], Mtrans[16];
    for (int i = 0; i < 16; ++i) { Mrot[i] = M[i]; Mtrans[i] = 0.0f; }
    Mrot[12] = 0.0f; Mrot[13] = 0.0f; Mrot[14] = 0.0f; Mrot[15] = 0.0f;
    Mtrans[0] = 1.0f; Mtrans[5] = 1.0f; Mtrans[10] = 1.0f;
    Mtrans[11] = M[11];
    Mtrans[12] = M[12]; Mtrans[13] = M[13];
    Mtrans[14] = M[14]; Mtrans[15] = M[15];
    uint64_t winFull = 0, winRot = 0, winTrans = 0, winNone = 0;
    double angSum = 0.0, angAbsSum = 0.0; uint64_t angN = 0;
    struct Row { uint32_t x, y; double mvxPx, mvyPx, pfxPx, pfyPx, prxPx, pryPx, resid; };
    std::vector<Row> rows;

    for (uint32_t y = 0; y < in.h; y += 4) {
        for (uint32_t x = 0; x < in.w; x += 4) {
            const size_t i = ((size_t)y * in.w + x) * in.halves;
            const float vx = velHalfToFloat(in.px[i]);
            const float vy = velHalfToFloat(in.px[i + 1]);
            ++nTotal;
            if (vx != vx || vy != vy) { ++nNaN; continue; }
            if (vx == 0.0f && vy == 0.0f) { ++nZero; continue; }
            ++nMoving;

            const double u    = 2.0 * ((double)x + 0.5) / (double)in.w - 1.0;
            const double vTop = 1.0 - 2.0 * ((double)y + 0.5) / (double)in.h;
            const double mvx  = 2.0 * (double)vx;      // NDC
            const double mvy  = 2.0 * (double)vy;
            const double px_  = u + mvx;               // measured previous, NDC
            const double py_  = vTop + mvy;

            const double auv = sqrt((double)vx*vx + (double)vy*vy);
            if (auv > maxAbsUv) maxAbsUv = auv;
            const double speedPx = sqrt(mvx*hw*mvx*hw + mvy*hh*mvy*hh);

            // ---- Predicted previous position, from the matrix, over depth.
            // Two points on the epipolar line: at infinity and at one metre.
            const double Ax = M[0]*u + M[4]*vTop + M[8];
            const double Ay = M[1]*u + M[5]*vTop + M[9];
            const double Aw = M[3]*u + M[7]*vTop + M[11];
            double resid = -1.0;
            if (fabs(Aw) > 1e-12) {
                const double ex = Ax / Aw, ey = Ay / Aw;
                const double w1 = Aw + M[15];
                if (fabs(w1) > 1e-12) {
                    const double fx = (Ax + M[12]) / w1, fy = (Ay + M[13]) / w1;
                    double lx = fx - ex, ly = fy - ey;
                    const double len = sqrt(lx*lx + ly*ly);
                    if (len < 1e-6) {
                        resid = sqrt((px_-ex)*(px_-ex)*hw*hw + (py_-ey)*(py_-ey)*hh*hh);
                    } else {
                        lx /= len; ly /= len;
                        resid = fabs((px_ - ex) * ly - (py_ - ey) * lx) * hh;
                    }
                }
                // Hypothesis 4: was the point behind the previous camera? The
                // sign of the predicted w at the depth the motion implies.
                if (w1 < 0.0) ++nPrevWNeg;
            }

            // Hypothesis 17: is the predicted previous position off-screen?
            if (px_ < -1.0 || px_ > 1.0 || py_ < -1.0 || py_ > 1.0) ++nPrevOffScreen;

            // ---- THE DEPTH IS SOLVED FROM X, SO X CANNOT BE A TEST.
            //
            // dd below is chosen to make the FULL matrix reproduce the measured
            // X exactly. Everything predicted with it therefore agrees with the
            // measurement in X by construction, and the raw rows duly printed
            // meas dx and full dx equal to every digit - 402480.00 against
            // 402480.00. That was read as "the X component of the field is
            // provably correct" and committed as such. It is a tautology.
            //
            // The Y column IS a test, because nothing forced it to agree: given
            // the depth that explains X, does the same depth explain Y? When it
            // does not, the measured previous position is consistent with no
            // single depth - which is the same statement the epipolar residual
            // makes, arrived at from the other side.
            //
            // Predicted flow under each of the three matrices, at that depth.
            if (fabs(Aw) > 1e-12) {
                const double denom = Aw * px_ - Ax;
                const double dd = (fabs(denom) > 1e-9)
                                ? ((double)M[12] - px_ * (double)M[15]) / denom
                                : -1.0;
                if (dd > 0.01 && dd < 100000.0) {
                    double fx1, fy1, fx2, fy2, fx3, fy3;
                    const float *QS[3] = { M, Mrot, Mtrans };
                    double OX[3], OY[3];
                    for (int q = 0; q < 3; ++q) {
                        const float *Q = QS[q];
                        const double qx = Q[0]*u + Q[4]*vTop + Q[8];
                        const double qy = Q[1]*u + Q[5]*vTop + Q[9];
                        const double qw = Q[3]*u + Q[7]*vTop + Q[11];
                        const double W  = dd * qw + Q[15];
                        if (fabs(W) < 1e-12) { OX[q] = u; OY[q] = vTop; }
                        else { OX[q] = (dd * qx + Q[12]) / W;
                               OY[q] = (dd * qy + Q[13]) / W; }
                    }
                    fx1 = OX[0]; fy1 = OY[0];
                    fx2 = OX[1]; fy2 = OY[1];
                    fx3 = OX[2]; fy3 = OY[2];
                    const double e1x = (px_ - fx1) * hw, e1y = (py_ - fy1) * hh;
                    const double e2x = (px_ - fx2) * hw, e2y = (py_ - fy2) * hh;
                    const double e3x = (px_ - fx3) * hw, e3y = (py_ - fy3) * hh;
                    const double e1 = e1x*e1x + e1y*e1y;
                    const double e2 = e2x*e2x + e2y*e2y;
                    const double e3 = e3x*e3x + e3y*e3y;
                    double best3 = e1; if (e2 < best3) best3 = e2; if (e3 < best3) best3 = e3;
                    if (best3 > 256.0)       ++winNone;
                    else if (best3 == e1)    ++winFull;
                    else if (best3 == e2)    ++winRot;
                    else                     ++winTrans;

                    const double pfx = (fx1 - u) * hw, pfy = (fy1 - vTop) * hh;
                    const double mfx = mvx * hw,       mfy = mvy * hh;
                    const double lm = sqrt(mfx*mfx + mfy*mfy);
                    const double lp = sqrt(pfx*pfx + pfy*pfy);
                    if (lm > 4.0 && lp > 4.0) {
                        double cs = (mfx*pfx + mfy*pfy) / (lm * lp);
                        if (cs >  1.0) cs =  1.0;
                        if (cs < -1.0) cs = -1.0;
                        const double sn = (mfx*pfy - mfy*pfx) / (lm * lp);
                        const double ang = atan2(sn, cs) * 57.29577951308232;
                        angSum += ang; angAbsSum += fabs(ang); ++angN;
                    }
                    if (rows.size() < 24 && resid > 8.0 && (x % 512) < 4) {
                        Row rw;
                        rw.x = x; rw.y = y;
                        rw.mvxPx = mvx * hw;  rw.mvyPx = mvy * hh;
                        rw.pfxPx = pfx;       rw.pfyPx = pfy;
                        rw.prxPx = (fx2 - u) * hw; rw.pryPx = (fy2 - vTop) * hh;
                        rw.resid = resid;
                        rows.push_back(rw);
                    }
                }
            }

            if (resid >= 0.0) {
                if (resid <= 1.0) ++nGoodPix; else ++nBadPix;
                for (int b = 0; b < kBins; ++b)
                    if (speedPx < binEdge[b]) { binResSum[b] += resid; ++binResN[b]; break; }
                const int rb = (int)((double)y / (double)in.h * 8.0);
                if (rb >= 0 && rb < 8) { rowResSum[rb] += resid; ++rowResN[rb]; }

                // Hypothesis 1: does the measured previous position equal the
                // Y-FLIPPED current position? That is precisely what the
                // near-field select writes when it fires.
                if (fabs(px_ - u) < 0.002 && fabs(py_ + vTop) < 0.02) ++nFlipMatch;
                if (resid > 1.0 && fabs(vTop) > 0.05) { kSum += py_ / vTop; ++kN; }
            }
        }
    }

    // ---- EXACT MODE: predict the flow from MEASURED depth.
    //
    // Every residual number in this project comes from an epipolar construction:
    // the perpendicular distance from the measured previous position to the
    // epipolar line. That is depth-free, which is why it was built, but it is
    // ill-conditioned near the focus of expansion and degenerate under near-pure
    // rotation, so a handful of pixels can throw enormous values into a MEAN
    // while the median sits at 0.000 px. Every run has reported exactly that
    // shape, and the tail was read as a defect for many commits.
    //
    // With TAA_MV_RGBA the target carries (vx, vy, currDepth, prevDepth), so the
    // flow can be predicted outright. Given NDC (u,v) and view depth d, the clip
    // coordinates are (u*d, v*d, d, 1), and
    //
    //     prev = M * (u*d, v*d, d, 1),   prevNDC = prev.xy / prev.w
    //
    // is the whole answer with no line, no epipole and no depth solved from the
    // number under test. The error printed here is in pixels, directly
    // comparable with the flow beside it.
    // Raw-clip mode reuses the same four channels for different quantities, so
    // exact mode must stand aside or it reads clip values as velocities - which
    // it duly did, reporting a median of 125135 px.
    // ---- MATRIX DUMP: what each draw actually loaded from the push constant.
    //
    // Channels are (vx, vy, M[5], M[0]) as seen by the vertex shader itself.
    // If the bad pixels report M[5] != 1, their draws never received our push
    // and the fault is upstream of every shader-side theory. If they report
    // M[5] == 1, the matrix is fine and the error is in what we do with it.
    if (getenv("TAA_MV_MATDUMP") && in.halves >= 4) {
        std::map<int, uint64_t> m5hist;
        uint64_t nGood = 0, nBad = 0;
        double m5GoodSum = 0.0, m5BadSum = 0.0;
        int shown = 0;
        std::vector<std::string> rows;
        for (uint32_t y = 0; y < in.h; y += 4) {
            for (uint32_t x = 0; x < in.w; x += 4) {
                const size_t i2 = ((size_t)y * in.w + x) * in.halves;
                const double vx = velHalfToFloat(in.px[i2]);
                const double vy = velHalfToFloat(in.px[i2 + 1]);
                const double m5 = velHalfToFloat(in.px[i2 + 3]);
                const double m0 = velHalfToFloat(in.px[i2 + 2]);
                if (vx != vx || vy != vy || m5 != m5) continue;
                const double spd = sqrt(vx*vx + vy*vy) * 2.0 * (in.w * 0.5);
                // Finer bucket: M[13] is small, so 100x buckets would collapse
                // every distinct value into one and hide the very variation
                // being looked for.
                m5hist[(int)(m5 * 10000.0 + (m5 < 0 ? -0.5 : 0.5))] += 1;
                if (spd > 64.0) { ++nBad; m5BadSum += m5; }
                else            { ++nGood; m5GoodSum += m5; }
                if (shown < 12 && spd > 64.0 && (x % 384) < 4 && y > in.h * 3 / 4) {
                    char b[192];
                    snprintf(b, sizeof(b), "  %6u %6u | speed %8.1f px | M[5] %9.5f  M[0] %9.5f",
                             x, y, spd, m5, m0);
                    rows.push_back(std::string(b));
                    ++shown;
                }
            }
        }
        // ---- PREDICT WITH THE MATRIX THIS PIXEL'S DRAW ACTUALLY SAW.
        //
        // Every element of row 1 checked out individually and prevW is right to
        // 0.03%, yet prevY is wrong by 23% at the same pixel. With identical
        // inputs and a correct row 3, the only way row 1 can be wrong is if the
        // matrix the shader used is not the one the diagnostic is comparing
        // against - and those readings came from different runs and frames.
        //
        // Channels are now (vx, vy, currW, M[5]) from the same fragment, so the
        // prediction can substitute the per-pixel M[5] and see whether the error
        // collapses. If it does, the field is right and the diagnostic has been
        // comparing against a stale matrix all along.
        {
            uint64_t nBoth = 0, badGlobal = 0, badLocal = 0;
            for (uint32_t y = 0; y < in.h; y += 4) {
                for (uint32_t x = 0; x < in.w; x += 4) {
                    const size_t i2 = ((size_t)y * in.w + x) * in.halves;
                    const double vx = velHalfToFloat(in.px[i2]);
                    const double vy = velHalfToFloat(in.px[i2 + 1]);
                    const double d  = velHalfToFloat(in.px[i2 + 2]);
                    const double m5 = velHalfToFloat(in.px[i2 + 3]);
                    if (!(d > 0.0) || d != d || vx != vx || vy != vy) continue;
                    if (!(fabs(m5) > 1e-6)) continue;
                    static const double vS7 = getenv("TAA_MV_VNEG") ? -1.0 : 1.0;
                    const double u  = ((x + 0.5) / (double)in.w) * 2.0 - 1.0;
                    const double vT = (((y + 0.5) / (double)in.h) * 2.0 - 1.0) * vS7;
                    const double xc = u * d, yc = vT * d;
                    ++nBoth;
                    for (int s = 0; s < 2; ++s) {
                        const double m5u = s ? m5 : (double)M[5];
                        const double nx = M[0]*xc + M[4]*yc + M[8]*d + M[12];
                        const double ny = M[1]*xc + m5u*yc + M[9]*d + M[13];
                        const double nw = M[3]*xc + M[7]*yc + M[11]*d + M[15];
                        if (fabs(nw) < 1e-9) continue;
                        const double pX = (nx / nw - u)  * 0.5;
                        const double pY = (ny / nw - vT) * 0.5;
                        const double ex = (vx - pX) * 2.0 * (in.w * 0.5);
                        const double ey = (vy - pY) * 2.0 * (in.h * 0.5);
                        if (sqrt(ex*ex + ey*ey) > 1.0) { if (s) ++badLocal; else ++badGlobal; }
                    }
                }
            }
            fprintf(f, "PREDICTION WITH GLOBAL vs PER-PIXEL M[5]\n");
            fprintf(f, "  pixels compared           %llu\n", (unsigned long long)nBoth);
            fprintf(f, "  bad with global M[5]      %llu  (%.2f%%)\n",
                    (unsigned long long)badGlobal,
                    nBoth ? 100.0 * badGlobal / (double)nBoth : 0.0);
            fprintf(f, "  bad with per-pixel M[5]   %llu  (%.2f%%)\n\n",
                    (unsigned long long)badLocal,
                    nBoth ? 100.0 * badLocal / (double)nBoth : 0.0);
        }

        fprintf(f, "MATRIX AS THE SHADER RECEIVED IT\n");
        fprintf(f, "view=%d  %ux%u\n\n", in.viewType, in.w, in.h);
        fprintf(f, "  mean M[5] on slow pixels (<64 px)  %10.5f   n=%llu\n",
                nGood ? m5GoodSum / (double)nGood : -1.0, (unsigned long long)nGood);
        fprintf(f, "  mean M[5] on fast pixels (>64 px)  %10.5f   n=%llu\n\n",
                nBad ? m5BadSum / (double)nBad : -1.0, (unsigned long long)nBad);
        fprintf(f, "DISTINCT M[13] VALUES SEEN (count)\n");
        int emitted = 0;
        for (std::map<int, uint64_t>::const_iterator it = m5hist.begin();
             it != m5hist.end() && emitted < 20; ++it, ++emitted)
            fprintf(f, "  %10.5f   %llu\n", it->first / 10000.0,
                    (unsigned long long)it->second);
        fprintf(f, "\nFAST PIXELS IN THE BAD REGION\n");
        for (size_t k = 0; k < rows.size(); ++k) fprintf(f, "%s\n", rows[k].c_str());
        fclose(f);
        return;
    }

    if (getenv("TAA_MV_RGBA") && in.halves >= 4 && !getenv("TAA_MV_RAWCLIP")
        && !getenv("TAA_MV_MATDUMP")) {
        const double hw = in.w * 0.5, hh = in.h * 0.5;
        const int kEB = 6;
        const double eEdge[6] = { 1.0, 4.0, 16.0, 64.0, 256.0, 1e30 };
        double eSum[6]; uint64_t eN[6];
        for (int b = 0; b < 6; ++b) { eSum[b]=0.0; eN[b]=0; }
        double rowESum[8]; uint64_t rowEN[8]; double rowEMax[8];
        for (int b = 0; b < 8; ++b) { rowESum[b]=0.0; rowEN[b]=0; rowEMax[b]=0.0; }
        uint64_t nOver1 = 0, nTot = 0;
        std::vector<double> errAll;
        for (uint32_t y = 0; y < in.h; y += 4) {
            for (uint32_t x = 0; x < in.w; x += 4) {
                const size_t i2 = ((size_t)y * in.w + x) * in.halves;
                const double vx = velHalfToFloat(in.px[i2]);
                const double vy = velHalfToFloat(in.px[i2 + 1]);
                const double d  = velHalfToFloat(in.px[i2 + 2]);
                if (!(d > 0.0) || d != d || vx != vx || vy != vy) continue;
                // ---- THE VIEWPORT HEIGHT IS NEGATIVE.
                //
                // X-Plane renders with 3840x-2160. A negative-height viewport
                // flips the NDC-to-framebuffer mapping, so ndc_y = +1 is at the
                // TOP, while this line has always produced -1 there. Every error
                // figure in this project used that convention, and both the
                // epipolar metric and this exact prediction share it - which is
                // how they agreed at 174 and 176 px while both could be wrong.
                //
                // A sign error in v produces error proportional to v and exactly
                // zero at the centre row. That is the signature the whole hunt
                // started from: "slope 1.712 px/row, zero at y = 1080".
                static const double vSign = getenv("TAA_MV_VNEG") ? -1.0 : 1.0;
                const double u    = ((x + 0.5) / (double)in.w) * 2.0 - 1.0;
                const double vTop = (((y + 0.5) / (double)in.h) * 2.0 - 1.0) * vSign;
                const double xc = u * d, yc = vTop * d;
                const double nx = M[0]*xc + M[4]*yc + M[8]*d + M[12];
                const double ny = M[1]*xc + M[5]*yc + M[9]*d + M[13];
                const double nw = M[3]*xc + M[7]*yc + M[11]*d + M[15];
                if (fabs(nw) < 1e-9) continue;
                const double pu = nx / nw, pv = ny / nw;
                // The field stores (prev - curr) in UV, scaled by 0.5.
                const double predX = (pu - u)    * 0.5;
                const double predY = (pv - vTop) * 0.5;
                const double ex = (vx - predX) * 2.0 * hw;
                const double ey = (vy - predY) * 2.0 * hh;
                const double err = sqrt(ex*ex + ey*ey);
                const double spd = sqrt(vx*vx + vy*vy) * 2.0 * hw;
                ++nTot; if (err > 1.0) ++nOver1;
                errAll.push_back(err);
                for (int b = 0; b < kEB; ++b)
                    if (spd < eEdge[b]) { eSum[b] += err; ++eN[b]; break; }
                const int rb = (int)((double)y / (double)in.h * 8.0);
                if (rb >= 0 && rb < 8) {
                    rowESum[rb] += err; ++rowEN[rb];
                    if (err > rowEMax[rb]) rowEMax[rb] = err;
                }
            }
        }
        double med = -1.0, p95 = -1.0, p999 = -1.0;
        if (!errAll.empty()) {
            std::sort(errAll.begin(), errAll.end());
            med  = errAll[errAll.size() / 2];
            p95  = errAll[(size_t)(errAll.size() * 0.95)];
            p999 = errAll[(size_t)(errAll.size() * 0.999)];
        }
        fprintf(f, "EXACT MODE - flow predicted from MEASURED depth, no epipolar line\n");
        fprintf(f, "view=%d  %ux%u  samples %llu\n\n",
                in.viewType, in.w, in.h, (unsigned long long)nTot);
        fprintf(f, "ERROR AGAINST EXACT PREDICTION (pixels)\n");
        fprintf(f, "  median            %10.4f px\n", med);
        fprintf(f, "  95th percentile   %10.4f px\n", p95);
        fprintf(f, "  99.9th percentile %10.4f px\n", p999);
        fprintf(f, "  beyond 1 px       %llu of %llu  (%.2f%%)\n\n",
                (unsigned long long)nOver1, (unsigned long long)nTot,
                nTot ? 100.0 * nOver1 / (double)nTot : 0.0);
        fprintf(f, "ERROR BY MEASURED SPEED\n");
        const char *elbl[6] = { "under 1 px", "1-4 px", "4-16 px", "16-64 px",
                                "64-256 px", "over 256 px" };
        for (int b = 0; b < kEB; ++b)
            fprintf(f, "  %-12s mean error %10.4f px   n=%llu\n", elbl[b],
                    eN[b] ? eSum[b] / (double)eN[b] : -1.0,
                    (unsigned long long)eN[b]);
        // ---- PROFILE THE PIXELS THAT ARE STILL WRONG.
        //
        // With the v sign corrected everything under 16 px/frame is exact
        // (0.008-0.025 px) and then 64-256 px jumps to 86.7 px - an error the
        // same size as the flow itself. That shape is not a small inaccuracy,
        // it is the field being roughly zero, roughly doubled, or pointing the
        // wrong way. The ratio of measured to predicted magnitude and the angle
        // between them says which, so measure it instead of theorising again.
        {
            double rSum = 0.0, aSum = 0.0, dSum = 0.0, pwSum = 0.0;
            uint64_t rN = 0, offN = 0, zeroN = 0;
            for (uint32_t y = 0; y < in.h; y += 4) {
                for (uint32_t x = 0; x < in.w; x += 4) {
                    const size_t i2 = ((size_t)y * in.w + x) * in.halves;
                    const double vx = velHalfToFloat(in.px[i2]);
                    const double vy = velHalfToFloat(in.px[i2 + 1]);
                    const double d  = velHalfToFloat(in.px[i2 + 2]);
                    if (!(d > 0.0) || d != d || vx != vx || vy != vy) continue;
                    static const double vS = getenv("TAA_MV_VNEG") ? -1.0 : 1.0;
                    const double u    = ((x + 0.5) / (double)in.w) * 2.0 - 1.0;
                    const double vTop = (((y + 0.5) / (double)in.h) * 2.0 - 1.0) * vS;
                    const double xc = u * d, yc = vTop * d;
                    const double nx = M[0]*xc + M[4]*yc + M[8]*d + M[12];
                    const double ny = M[1]*xc + M[5]*yc + M[9]*d + M[13];
                    const double nw = M[3]*xc + M[7]*yc + M[11]*d + M[15];
                    if (fabs(nw) < 1e-9) continue;
                    const double pu = nx / nw, pv = ny / nw;
                    const double predX = (pu - u) * 0.5, predY = (pv - vTop) * 0.5;
                    const double ex = (vx - predX) * 2.0 * hw;
                    const double ey = (vy - predY) * 2.0 * hh;
                    if (sqrt(ex*ex + ey*ey) <= 1.0) continue;
                    const double mMag = sqrt(vx*vx + vy*vy);
                    const double pMag = sqrt(predX*predX + predY*predY);
                    ++rN;
                    dSum += d; pwSum += nw;
                    if (pMag > 1e-12) rSum += mMag / pMag;
                    if (mMag < 1e-9) ++zeroN;
                    if (fabs(pu) > 1.0 || fabs(pv) > 1.0) ++offN;
                    if (mMag > 1e-12 && pMag > 1e-12) {
                        double cs = (vx*predX + vy*predY) / (mMag * pMag);
                        if (cs > 1.0) cs = 1.0;
                        if (cs < -1.0) cs = -1.0;
                        aSum += acos(cs) * 57.29577951308232;
                    }
                }
            }
            // ---- WHERE ARE THE BAD PIXELS ON SCREEN?
        //
        // A contiguous blob is a moving object - water, most likely, since this
        // is the terrain/water shader and the scenery is Seattle. Scatter across
        // the frame is a systematic fault. The row means cannot tell these apart
        // but a coarse tile map can.
        {
            fprintf(f, "\nBAD-PIXEL MAP (16x8 tiles, %% of tile beyond 1 px)\n");
            for (int ty = 0; ty < 8; ++ty) {
                fprintf(f, "  ");
                for (int tx = 0; tx < 16; ++tx) {
                    uint64_t tn = 0, tb = 0;
                    const uint32_t y0 = in.h * ty / 8, y1 = in.h * (ty + 1) / 8;
                    const uint32_t x0 = in.w * tx / 16, x1 = in.w * (tx + 1) / 16;
                    for (uint32_t y = y0; y < y1; y += 8) {
                        for (uint32_t x = x0; x < x1; x += 8) {
                            const size_t i2 = ((size_t)y * in.w + x) * in.halves;
                            const double vx = velHalfToFloat(in.px[i2]);
                            const double vy = velHalfToFloat(in.px[i2 + 1]);
                            const double d  = velHalfToFloat(in.px[i2 + 2]);
                            if (!(d > 0.0) || d != d || vx != vx || vy != vy) continue;
                            static const double vS4 = getenv("TAA_MV_VNEG") ? -1.0 : 1.0;
                            const double u  = ((x + 0.5) / (double)in.w) * 2.0 - 1.0;
                            const double vT = (((y + 0.5) / (double)in.h) * 2.0 - 1.0) * vS4;
                            const double xc = u * d, yc = vT * d;
                            const double nx = M[0]*xc + M[4]*yc + M[8]*d + M[12];
                            const double ny = M[1]*xc + M[5]*yc + M[9]*d + M[13];
                            const double nw = M[3]*xc + M[7]*yc + M[11]*d + M[15];
                            if (fabs(nw) < 1e-9) continue;
                            const double pX = (nx / nw - u)  * 0.5;
                            const double pY = (ny / nw - vT) * 0.5;
                            const double ex = (vx - pX) * 2.0 * hw;
                            const double ey = (vy - pY) * 2.0 * hh;
                            ++tn;
                            if (sqrt(ex*ex + ey*ey) > 1.0) ++tb;
                        }
                    }
                    fprintf(f, "%4d", tn ? (int)(100.0 * tb / (double)tn) : -1);
                }
                fprintf(f, "\n");
            }
            fprintf(f, "      -1 = nothing drew there\n");
        }

        fprintf(f, "\nPROFILE OF THE PIXELS STILL WRONG (err > 1 px)\n");
            fprintf(f, "  count                       %llu\n", (unsigned long long)rN);
            fprintf(f, "  mean |measured|/|predicted| %10.4f   (1 = right size)\n",
                    rN ? rSum / (double)rN : -1.0);
            fprintf(f, "  mean angle between them     %10.4f deg\n",
                    rN ? aSum / (double)rN : -1.0);
            fprintf(f, "  mean depth                  %10.4f m\n",
                    rN ? dSum / (double)rN : -1.0);
            fprintf(f, "  mean predicted prev.w       %10.4f\n",
                    rN ? pwSum / (double)rN : -1.0);
            fprintf(f, "  measured exactly zero       %llu\n", (unsigned long long)zeroN);
            fprintf(f, "  predicted prev off-screen   %llu  (%.2f%%)\n",
                    (unsigned long long)offN, rN ? 100.0 * offN / (double)rN : 0.0);

            // ---- DO THE BAD PIXELS FIT THE OPPOSITE Y CONVENTION?
            //
            // The raw rows show dx correct to 2% while dy ramps linearly at
            // 1.64 px per row against a flat prediction. Depth cannot do that -
            // dy proportional to 1/d would need 0.85 m and would ruin dx too.
            // A linear ramp in v is what a Y-convention mismatch looks like, so
            // re-predict these pixels with v negated and count the matches. If
            // they fit, some draws use the opposite Y sign from the rest and the
            // fix belongs in the shader, per pipeline.
            {
                uint64_t altFit = 0, altN = 0;
                for (uint32_t y = 0; y < in.h; y += 4) {
                    for (uint32_t x = 0; x < in.w; x += 4) {
                        const size_t i2 = ((size_t)y * in.w + x) * in.halves;
                        const double vx = velHalfToFloat(in.px[i2]);
                        const double vy = velHalfToFloat(in.px[i2 + 1]);
                        const double d  = velHalfToFloat(in.px[i2 + 2]);
                        if (!(d > 0.0) || d != d || vx != vx || vy != vy) continue;
                        static const double vS3 = getenv("TAA_MV_VNEG") ? -1.0 : 1.0;
                        const double u  = ((x + 0.5) / (double)in.w) * 2.0 - 1.0;
                        const double v0 = (((y + 0.5) / (double)in.h) * 2.0 - 1.0) * vS3;
                        double bestOwn = -1.0, bestAlt = -1.0;
                        for (int s = 0; s < 2; ++s) {
                            const double vv = s ? -v0 : v0;
                            const double xc = u * d, yc = vv * d;
                            const double nx = M[0]*xc + M[4]*yc + M[8]*d + M[12];
                            const double ny = M[1]*xc + M[5]*yc + M[9]*d + M[13];
                            const double nw = M[3]*xc + M[7]*yc + M[11]*d + M[15];
                            if (fabs(nw) < 1e-9) continue;
                            const double pX = (nx / nw - u)  * 0.5;
                            const double pY = (ny / nw - vv) * 0.5;
                            const double ex = (vx - pX) * 2.0 * hw;
                            const double ey = (vy - pY) * 2.0 * hh;
                            const double e  = sqrt(ex*ex + ey*ey);
                            if (s) bestAlt = e; else bestOwn = e;
                        }
                        if (bestOwn <= 1.0 || bestOwn < 0.0) continue;
                        ++altN;
                        if (bestAlt >= 0.0 && bestAlt <= 1.0) ++altFit;
                    }
                }
            // ---- DO THE BAD PIXELS FIT THE TRANSPOSED MATRIX?
            //
            // The shader reports M[5] = 0.99951, so the matrix it receives is
            // right - the "effective M[5] = -0.79" was a bad inference. M[5]
            // and M[0] are the DIAGONAL, which a transpose leaves untouched
            // while swapping the translation column M[12..14] into the w row
            // M[3],M[7],M[11]. At the bad pixels that changes prevW from about
            // 10.82 to about 0.56, and the resulting error scales as 1/depth -
            // which is why it tracks speed, spares distant geometry and hugs
            // the bottom-right corner.
            {
                uint64_t trFit = 0, trN = 0;
                float T[16];
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c)
                        T[c * 4 + r] = M[r * 4 + c];
                for (uint32_t y = 0; y < in.h; y += 4) {
                    for (uint32_t x = 0; x < in.w; x += 4) {
                        const size_t i2 = ((size_t)y * in.w + x) * in.halves;
                        const double vx = velHalfToFloat(in.px[i2]);
                        const double vy = velHalfToFloat(in.px[i2 + 1]);
                        const double d  = velHalfToFloat(in.px[i2 + 2]);
                        if (!(d > 0.0) || d != d || vx != vx || vy != vy) continue;
                        static const double vS6 = getenv("TAA_MV_VNEG") ? -1.0 : 1.0;
                        const double u  = ((x + 0.5) / (double)in.w) * 2.0 - 1.0;
                        const double vT = (((y + 0.5) / (double)in.h) * 2.0 - 1.0) * vS6;
                        const double xc = u * d, yc = vT * d;
                        double e[2];
                        for (int s = 0; s < 2; ++s) {
                            const float *Q = s ? T : M;
                            const double nx = Q[0]*xc + Q[4]*yc + Q[8]*d + Q[12];
                            const double ny = Q[1]*xc + Q[5]*yc + Q[9]*d + Q[13];
                            const double nw = Q[3]*xc + Q[7]*yc + Q[11]*d + Q[15];
                            if (fabs(nw) < 1e-9) { e[s] = 1e30; continue; }
                            const double pX = (nx / nw - u)  * 0.5;
                            const double pY = (ny / nw - vT) * 0.5;
                            const double ex = (vx - pX) * 2.0 * hw;
                            const double ey = (vy - pY) * 2.0 * hh;
                            e[s] = sqrt(ex*ex + ey*ey);
                        }
                        if (e[0] <= 1.0) continue;
                        ++trN;
                        if (e[1] <= 1.0) ++trFit;
                    }
                }
                fprintf(f, "  bad pixels fitting TRANSPOSED M  %llu of %llu  (%.2f%%)\n",
                        (unsigned long long)trFit, (unsigned long long)trN,
                        trN ? 100.0 * trFit / (double)trN : 0.0);
            }

                fprintf(f, "  bad pixels fitting NEGATED v  %llu of %llu  (%.2f%%)\n",
                        (unsigned long long)altFit, (unsigned long long)altN,
                        altN ? 100.0 * altFit / (double)altN : 0.0);
                fprintf(f, "      a high share means those draws use the opposite\n");
                fprintf(f, "      Y convention and the fix is per pipeline\n");
            }
            fprintf(f, "      ratio 0 = field empty, 2 = doubled, 1 with a large\n");
            fprintf(f, "      angle = right size wrong direction\n");

            // Summary statistics have misled this investigation repeatedly.
            // Print the actual pixels: position, depth, measured and predicted
            // flow in pixels, and a clean neighbour for contrast.
            fprintf(f, "\nRAW BAD PIXELS\n");
            fprintf(f, "  %6s %6s %9s | %9s %9s | %9s %9s | %8s\n",
                    "x", "y", "depth", "meas dx", "meas dy", "pred dx", "pred dy", "ratio");
            int shown = 0;
            for (uint32_t y = 0; y < in.h && shown < 20; y += 4) {
                for (uint32_t x = 0; x < in.w && shown < 20; x += 4) {
                    const size_t i2 = ((size_t)y * in.w + x) * in.halves;
                    const double vx = velHalfToFloat(in.px[i2]);
                    const double vy = velHalfToFloat(in.px[i2 + 1]);
                    const double d  = velHalfToFloat(in.px[i2 + 2]);
                    if (!(d > 0.0) || d != d || vx != vx || vy != vy) continue;
                    static const double vS2 = getenv("TAA_MV_VNEG") ? -1.0 : 1.0;
                    const double u    = ((x + 0.5) / (double)in.w) * 2.0 - 1.0;
                    const double vTop = (((y + 0.5) / (double)in.h) * 2.0 - 1.0) * vS2;
                    const double xc = u * d, yc = vTop * d;
                    const double nx = M[0]*xc + M[4]*yc + M[8]*d + M[12];
                    const double ny = M[1]*xc + M[5]*yc + M[9]*d + M[13];
                    const double nw = M[3]*xc + M[7]*yc + M[11]*d + M[15];
                    if (fabs(nw) < 1e-9) continue;
                    const double predX = (nx / nw - u) * 0.5;
                    const double predY = (ny / nw - vTop) * 0.5;
                    const double ex = (vx - predX) * 2.0 * hw;
                    const double ey = (vy - predY) * 2.0 * hh;
                    if (sqrt(ex*ex + ey*ey) <= 8.0) continue;
                    if ((x % 384) > 4) continue;
                    const double mM = sqrt(vx*vx + vy*vy), pM = sqrt(predX*predX + predY*predY);
                    fprintf(f, "  %6u %6u %9.3f | %9.2f %9.2f | %9.2f %9.2f | %8.2f\n",
                            x, y, d, vx * 2.0 * hw, vy * 2.0 * hh,
                            predX * 2.0 * hw, predY * 2.0 * hh,
                            pM > 1e-12 ? mM / pM : -1.0);
                    ++shown;
                }
            }
        }

        fprintf(f, "\nERROR BY SCREEN ROW (top to bottom, eighths)\n");
        for (int b = 0; b < 8; ++b)
            fprintf(f, "  rows %4u-%4u  mean %10.4f px   worst %10.2f px   n=%llu\n",
                    (unsigned)(in.h * b / 8), (unsigned)(in.h * (b + 1) / 8 - 1),
                    rowEN[b] ? rowESum[b] / (double)rowEN[b] : -1.0,
                    rowEN[b] ? rowEMax[b] : -1.0,
                    (unsigned long long)rowEN[b]);
        // ---- WHICH FRAGMENT SHADER WROTE THE BAD PIXELS?
        //
        // Channel 3 carries the per-module tag baked in at injection time, so
        // every bad pixel traces to the exact shader that produced it. The
        // error is bimodal - most pixels perfect, about a quarter ruined - and
        // nothing shared (the matrix, the inputs, prev.w) can do that. This
        // names the subset instead of guessing at it.
        if (getenv("TAA_MV_PID")) {
            struct PidAcc { double sum; uint64_t n, bad; double worst; };
            std::map<int, PidAcc> byPid;
            for (uint32_t y = 0; y < in.h; y += 4) {
                for (uint32_t x = 0; x < in.w; x += 4) {
                    const size_t i2 = ((size_t)y * in.w + x) * in.halves;
                    const double vx = velHalfToFloat(in.px[i2]);
                    const double vy = velHalfToFloat(in.px[i2 + 1]);
                    const double d  = velHalfToFloat(in.px[i2 + 2]);
                    const double pf = velHalfToFloat(in.px[i2 + 3]);
                    if (!(d > 0.0) || d != d || vx != vx || vy != vy) continue;
                    // Instance index 0 is the ordinary path and most of the
                    // frame; excluding it hid exactly the pixels under test.
                    const int pid = (int)(pf + 0.5);
                    if (pid < 0) continue;
                    static const double vSign2 = getenv("TAA_MV_VNEG") ? -1.0 : 1.0;
                    const double u    = ((x + 0.5) / (double)in.w) * 2.0 - 1.0;
                    const double vTop = (((y + 0.5) / (double)in.h) * 2.0 - 1.0) * vSign2;
                    const double xc = u * d, yc = vTop * d;
                    const double nx = M[0]*xc + M[4]*yc + M[8]*d + M[12];
                    const double ny = M[1]*xc + M[5]*yc + M[9]*d + M[13];
                    const double nw = M[3]*xc + M[7]*yc + M[11]*d + M[15];
                    if (fabs(nw) < 1e-9) continue;
                    const double predX = (nx / nw - u)    * 0.5;
                    const double predY = (ny / nw - vTop) * 0.5;
                    const double ex = (vx - predX) * 2.0 * hw;
                    const double ey = (vy - predY) * 2.0 * hh;
                    const double err = sqrt(ex*ex + ey*ey);
                    PidAcc &a = byPid[pid];
                    a.sum += err; ++a.n;
                    if (err > 1.0) ++a.bad;
                    if (err > a.worst) a.worst = err;
                }
            }
            fprintf(f, "\nERROR BY %s (channel 3 tag)\n",
                    getenv("TAA_MV_INST") ? "INSTANCE INDEX" : "FRAGMENT SHADER");
            fprintf(f, "  %6s %10s %10s %10s %12s %10s\n",
                    "pid", "pixels", "bad>1px", "bad%", "mean err", "worst");
            std::vector<std::pair<uint64_t, int> > order;
            for (std::map<int, PidAcc>::const_iterator it = byPid.begin();
                 it != byPid.end(); ++it)
                order.push_back(std::make_pair(it->second.bad, it->first));
            std::sort(order.begin(), order.end());
            std::reverse(order.begin(), order.end());
            const size_t show = order.size() < 24 ? order.size() : 24;
            for (size_t k = 0; k < show; ++k) {
                const PidAcc &a = byPid[order[k].second];
                fprintf(f, "  %6d %10llu %10llu %9.2f%% %12.4f %10.2f\n",
                        order[k].second, (unsigned long long)a.n,
                        (unsigned long long)a.bad,
                        a.n ? 100.0 * a.bad / (double)a.n : 0.0,
                        a.n ? a.sum / (double)a.n : -1.0, a.worst);
            }
            fprintf(f, "  %llu distinct fragment shaders covered this frame\n",
                    (unsigned long long)byPid.size());
            fprintf(f, "      a few pids owning nearly all the bad pixels names\n");
            fprintf(f, "      the shader family to dump and read\n");
        }

        fprintf(f, "\n  The epipolar metric called rows 1890-2159 174 px wrong.\n");
        fprintf(f, "  If this says otherwise, the metric was the defect.\n");
        fclose(f);
        return;
    }

    // ---- RAW CLIP MODE: check the assumption everything else rests on.
    //
    // Channels are (prevClip.y, prevClip.w, currClip.y, currClip.w) straight
    // from the shader. Two things get tested that never have been:
    //
    //   currClip.y / currClip.w  ==  the pixel's NDC v ?
    //   prevClip.y  ==  M[1]*xc + M[5]*yc + M[9]*d + M[13] ?
    //
    // The first is the assumption behind every prediction and elimination in
    // this investigation. The second says whether the shader is applying the
    // matrix that was pushed.
    if (getenv("TAA_MV_RAWCLIP") && in.halves >= 4) {
        const double hw2 = in.w * 0.5, hh2 = in.h * 0.5;
        (void)hw2;
        double vErrSum = 0.0, pErrSum = 0.0; uint64_t vN = 0, vBad = 0, pBad = 0;
        double worstV = 0.0, worstP = 0.0;
        int shown = 0;
        std::vector<std::string> rows;
        for (uint32_t y = 0; y < in.h; y += 4) {
            for (uint32_t x = 0; x < in.w; x += 4) {
                const size_t i2 = ((size_t)y * in.w + x) * in.halves;
                // Channels are now (prevY, prevW, currY, M[5]) from ONE fragment.
                // currW is recovered as currY / v, which is exact because
                // currClip.y/currClip.w was already verified to equal the pixel
                // v to better than 0.001 on every sample.
                const double py  = velHalfToFloat(in.px[i2]);
                const double pw  = velHalfToFloat(in.px[i2 + 1]);
                const double cy  = velHalfToFloat(in.px[i2 + 2]);
                const double m5s = velHalfToFloat(in.px[i2 + 3]);
                if (cy != cy || pw != pw || py != py || !(fabs(m5s) > 1e-6)) continue;
                static const double vS5 = getenv("TAA_MV_VNEG") ? -1.0 : 1.0;
                const double vPix = (((y + 0.5) / (double)in.h) * 2.0 - 1.0) * vS5;
                if (fabs(vPix) < 1e-6) continue;
                const double cw   = cy / vPix;
                if (!(cw > 0.0) || cw != cw) continue;
                const double vSh  = vPix;
                const double u    = ((x + 0.5) / (double)in.w) * 2.0 - 1.0;
                const double d    = cw;
                const double xc = u * d, yc = vSh * d;
                // ---- CHANNEL 3 IS NOW M[13], THE SHADER'S OWN TRANSLATION.
                //
                // M[5] histogrammed as a single value so it does not vary.
                // M[12..14] are the terms that change every frame as the camera
                // moves, so they are what a frame skew between the readback and
                // the diagnostic's matrix would corrupt - and the error would
                // scale as translation/depth, which is the 1/d signature.
                //
                // Comparing the shader's M[13] against the diagnostic's in the
                // SAME run is the apples-to-apples test. The previous apparent
                // mismatch (0.00080 against 0.00521) came from two different
                // runs on different frames, which is the trap this
                // investigation has fallen into repeatedly.
                const double m13s = m5s;
                const double expPy = M[1]*xc + M[5]*yc + M[9]*d + m13s;
                // ---- RELATIVE, NOT ABSOLUTE.
                //
                // The first pass used an absolute threshold of 0.01 on prevY,
                // whose values reach 8000 at far geometry where half-float
                // steps are 4 and 8. It reported 31% disagreement and was
                // measuring the readback's quantisation. Relative error, and
                // only where the flow is big enough to matter, tests the shader
                // instead of the channel.
                const double dv = fabs(vSh - vPix);
                const double scale = fabs(expPy) > 1.0 ? fabs(expPy) : 1.0;
                const double dp = fabs(py - expPy) / scale;
                ++vN; vErrSum += dv; pErrSum += dp;
                if (dv > 0.001) ++vBad;
                if (dp > 0.001) ++pBad;
                if (dv > worstV) worstV = dv;
                if (dp > worstP) worstP = dp;
                // Aim at the bottom-right, where the bad pixels actually are.
                if (shown < 16 && dp > 0.001 && y > in.h * 3 / 4 && (x % 384) < 4) {
                    char b[256];
                    snprintf(b, sizeof(b),
                             "  %6u %6u | vPix %9.5f vShader %9.5f | prevY %10.4f expected %10.4f | w %8.3f",
                             x, y, vPix, vSh, py, expPy, cw);
                    rows.push_back(std::string(b));
                    ++shown;
                }
            }
        }
        fprintf(f, "RAW CLIP MODE - the shader's own clip values\n");
        fprintf(f, "view=%d  %ux%u  samples %llu\n\n",
                in.viewType, in.w, in.h, (unsigned long long)vN);
        fprintf(f, "IS currClip.y/currClip.w THE PIXEL'S v?\n");
        fprintf(f, "  mean |vShader - vPixel|   %12.8f\n", vN ? vErrSum / (double)vN : -1.0);
        fprintf(f, "  worst                     %12.8f\n", worstV);
        fprintf(f, "  beyond 0.001              %llu of %llu  (%.2f%%)\n\n",
                (unsigned long long)vBad, (unsigned long long)vN,
                vN ? 100.0 * vBad / (double)vN : 0.0);
        fprintf(f, "IS prevClip.y THE PUSHED MATRIX APPLIED?  (RELATIVE)\n");
        fprintf(f, "  mean relative error       %12.8f\n", vN ? pErrSum / (double)vN : -1.0);
        fprintf(f, "  worst                     %12.8f\n", worstP);
        fprintf(f, "  beyond 0.001 relative     %llu of %llu  (%.2f%%)\n\n",
                (unsigned long long)pBad, (unsigned long long)vN,
                vN ? 100.0 * pBad / (double)vN : 0.0);
        // ---- prevClip.w IS THE DENOMINATOR AND HAS NEVER BEEN COMPARED.
        //
        // Measured flow is 19.7x the prediction at 73 degrees. The matrix is
        // confirmed correct (M[5] = 0.99951), the inputs are confirmed correct
        // (currClip.y/w equals the pixel v), transpose fits 0 of 88660 and
        // negated v fits 33. A field that much too large is a divide that blew
        // up, and prevClip.w is the divisor. It has been sitting in channel 1
        // the whole time unexamined.
        {
            double wRelSum = 0.0, worstW = 0.0; uint64_t wN = 0, wBad = 0;
            int shownW = 0;
            std::vector<std::string> wrows;
            for (uint32_t y = 0; y < in.h; y += 4) {
                for (uint32_t x = 0; x < in.w; x += 4) {
                    const size_t i2 = ((size_t)y * in.w + x) * in.halves;
                    const double pw = velHalfToFloat(in.px[i2 + 1]);
                    const double cy = velHalfToFloat(in.px[i2 + 2]);
                    const double py = velHalfToFloat(in.px[i2]);
                    static const double vS8 = getenv("TAA_MV_VNEG") ? -1.0 : 1.0;
                    const double vP8 = (((y + 0.5) / (double)in.h) * 2.0 - 1.0) * vS8;
                    if (fabs(vP8) < 1e-6) continue;
                    const double cw = cy / vP8;
                    if (!(cw > 0.0) || cw != cw || pw != pw || cy != cy) continue;
                    const double u  = ((x + 0.5) / (double)in.w) * 2.0 - 1.0;
                    const double d  = cw;
                    const double xc = u * d, yc = cy;
                    const double expPw = M[3]*xc + M[7]*yc + M[11]*d + M[15];
                    (void)0;
                    const double sc = fabs(expPw) > 1.0 ? fabs(expPw) : 1.0;
                    const double rel = fabs(pw - expPw) / sc;
                    ++wN; wRelSum += rel;
                    if (rel > 0.001) ++wBad;
                    if (rel > worstW) worstW = rel;
                    if (shownW < 12 && rel > 0.001 && y > in.h * 3 / 4 && (x % 384) < 4) {
                        char b[224];
                        snprintf(b, sizeof(b),
                                 "  %6u %6u | prevW %10.4f expected %10.4f | currW %9.3f | prevY %9.3f",
                                 x, y, pw, expPw, cw, py);
                        wrows.push_back(std::string(b));
                        ++shownW;
                    }
                }
            }
            // ---- IS THE NEAR-FIELD SELECT FIRING?
            //
            // idPrevSel = select(posW < nearFieldDist, idFlipped, idPrevClip)
            // with idFlipped = (posX, -posY, posZ, posW). When it fires,
            // prevW = posW = d, which MATCHES the matrix prediction because
            // M[11] is ~1 and prevW is approximately d anyway - while prevY is
            // the negation of currClip.y. Row 3 looks perfect and row 1 is
            // wrong, which is precisely the contradiction measured.
            //
            // block[18] read 0.000000 and the old flip test read low, but both
            // predate the v-sign correction, so neither ruled this out.
            {
                uint64_t flipFit = 0, flipN = 0;
                for (uint32_t y = 0; y < in.h; y += 4) {
                    for (uint32_t x = 0; x < in.w; x += 4) {
                        const size_t i2 = ((size_t)y * in.w + x) * in.halves;
                        const double py2 = velHalfToFloat(in.px[i2]);
                        const double cy2 = velHalfToFloat(in.px[i2 + 2]);
                        if (py2 != py2 || cy2 != cy2) continue;
                        if (fabs(cy2) < 1e-6) continue;
                        ++flipN;
                        const double r = py2 / cy2;
                        if (r < -0.9 && r > -1.1) ++flipFit;
                    }
                }
                // ---- SOLVE FOR THE ROW 1 THE SHADER IS ACTUALLY USING.
            //
            // prevY = a*xc + b*yc + c*d + e is linear in four unknowns and
            // there are hundreds of thousands of pixels, so row 1 can be
            // recovered by least squares instead of guessed one channel at a
            // time. Comparing the fit against M[1], M[5], M[9], M[13] says
            // exactly which element is wrong - or, if the fit matches the
            // pushed row, that prevY is not a linear function of these inputs
            // at all and the input vector is not what the code says.
            {
                double A[4][5];
                for (int r = 0; r < 4; ++r)
                    for (int c2 = 0; c2 < 5; ++c2) A[r][c2] = 0.0;
                uint64_t fitN = 0;
                for (uint32_t y = 0; y < in.h; y += 4) {
                    for (uint32_t x = 0; x < in.w; x += 4) {
                        const size_t i2 = ((size_t)y * in.w + x) * in.halves;
                        const double py2 = velHalfToFloat(in.px[i2]);
                        const double cy2 = velHalfToFloat(in.px[i2 + 2]);
                        if (py2 != py2 || cy2 != cy2) continue;
                        static const double vS9 = getenv("TAA_MV_VNEG") ? -1.0 : 1.0;
                        const double vP9 = (((y + 0.5) / (double)in.h) * 2.0 - 1.0) * vS9;
                        if (fabs(vP9) < 1e-6) continue;
                        const double dd = cy2 / vP9;
                        if (!(dd > 0.0) || dd > 1e5) continue;
                        const double uu = ((x + 0.5) / (double)in.w) * 2.0 - 1.0;
                        // ---- WEIGHT THE FIT, OR THE FAR FIELD DECIDES IT.
                        //
                        // prevY reaches 8000 on distant geometry where the
                        // half-float step is 8, so those residuals swamp the
                        // near pixels the error actually lives in. The first
                        // fit returned a constant of 0.395 against a pushed
                        // M[13] of 0.00126 - but at a near pixel the PUSHED
                        // value predicted better than the fitted one, which is
                        // the signature of a fit decided by the far tail.
                        //
                        // Restricting to the near field and weighting by 1/|py|
                        // makes every pixel contribute its RELATIVE error, which
                        // is the quantity that matters after the divide.
                        if (dd > 200.0) continue;
                        const double wgt = 1.0 / (fabs(py2) > 1.0 ? fabs(py2) : 1.0);
                        const double b[4] = { uu * dd * wgt, cy2 * wgt, dd * wgt, wgt };
                        for (int r = 0; r < 4; ++r) {
                            for (int c2 = 0; c2 < 4; ++c2) A[r][c2] += b[r] * b[c2];
                            A[r][4] += b[r] * py2 * wgt;
                        }
                        ++fitN;
                    }
                }
                // Gaussian elimination with partial pivoting.
                double sol[4] = {0,0,0,0};
                bool ok = fitN > 100;
                if (ok) {
                    for (int i3 = 0; i3 < 4 && ok; ++i3) {
                        int piv = i3;
                        for (int r = i3 + 1; r < 4; ++r)
                            if (fabs(A[r][i3]) > fabs(A[piv][i3])) piv = r;
                        if (fabs(A[piv][i3]) < 1e-12) { ok = false; break; }
                        for (int c2 = 0; c2 < 5; ++c2) {
                            const double tmp = A[i3][c2]; A[i3][c2] = A[piv][c2]; A[piv][c2] = tmp;
                        }
                        for (int r = 0; r < 4; ++r) {
                            if (r == i3) continue;
                            const double f2 = A[r][i3] / A[i3][i3];
                            for (int c2 = i3; c2 < 5; ++c2) A[r][c2] -= f2 * A[i3][c2];
                        }
                    }
                    if (ok) for (int i3 = 0; i3 < 4; ++i3) sol[i3] = A[i3][4] / A[i3][i3];
                }
                {
                    // Per-pixel M[13] against the diagnostic's, same run.
                    std::map<int, uint64_t> m13h;
                    for (uint32_t y = 0; y < in.h; y += 16)
                        for (uint32_t x = 0; x < in.w; x += 16) {
                            const size_t i2 = ((size_t)y * in.w + x) * in.halves;
                            const double v3 = velHalfToFloat(in.px[i2 + 3]);
                            if (v3 != v3) continue;
                            m13h[(int)(v3 * 100000.0 + (v3 < 0 ? -0.5 : 0.5))] += 1;
                        }
                    fprintf(f, "SHADER M[13] vs DIAGNOSTIC M[13], SAME RUN\n");
                    fprintf(f, "  diagnostic M[13]  %12.6f\n", (double)M[13]);
                    int shownH = 0;
                    for (std::map<int, uint64_t>::const_iterator it = m13h.begin();
                         it != m13h.end() && shownH < 8; ++it, ++shownH)
                        fprintf(f, "  shader    M[13]  %12.6f   n=%llu\n",
                                it->first / 100000.0, (unsigned long long)it->second);
                    fprintf(f, "\n");
                }
                fprintf(f, "ROW 1 RECOVERED FROM THE PIXELS (weighted, d<200 m, n=%llu)\n",
                        (unsigned long long)fitN);
                if (ok) {
                    fprintf(f, "  coefficient of xc  fitted %10.5f   pushed M[1]  %10.5f\n", sol[0], (double)M[1]);
                    fprintf(f, "  coefficient of yc  fitted %10.5f   pushed M[5]  %10.5f\n", sol[1], (double)M[5]);
                    fprintf(f, "  coefficient of d   fitted %10.5f   pushed M[9]  %10.5f\n", sol[2], (double)M[9]);
                    fprintf(f, "  constant           fitted %10.5f   pushed M[13] %10.5f\n\n", sol[3], (double)M[13]);
                } else {
                    fprintf(f, "  fit failed\n\n");
                }
            }

            fprintf(f, "IS THE NEAR-FIELD FLIP FIRING?\n");
                fprintf(f, "  prevY / currY within 10%% of -1   %llu of %llu  (%.2f%%)\n",
                        (unsigned long long)flipFit, (unsigned long long)flipN,
                        flipN ? 100.0 * flipFit / (double)flipN : 0.0);
                fprintf(f, "      the flip preserves w, so prevW still matches the\n");
                fprintf(f, "      matrix while prevY is negated - row 3 right,\n");
                fprintf(f, "      row 1 wrong, which is the observed contradiction\n\n");
            }

            fprintf(f, "IS prevClip.w THE PUSHED MATRIX APPLIED?  (RELATIVE)\n");
            fprintf(f, "  mean relative error       %12.8f\n", wN ? wRelSum / (double)wN : -1.0);
            fprintf(f, "  worst                     %12.8f\n", worstW);
            fprintf(f, "  beyond 0.001 relative     %llu of %llu  (%.2f%%)\n\n",
                    (unsigned long long)wBad, (unsigned long long)wN,
                    wN ? 100.0 * wBad / (double)wN : 0.0);
            fprintf(f, "DISAGREEING prevW PIXELS (bottom quarter)\n");
            for (size_t k = 0; k < wrows.size(); ++k)
                fprintf(f, "%s\n", wrows[k].c_str());
            fprintf(f, "\n");
        }

        fprintf(f, "DISAGREEING PIXELS\n");
        for (size_t k = 0; k < rows.size(); ++k)
            fprintf(f, "%s\n", rows[k].c_str());
        fclose(f);
        return;
    }

    // ---- DEPTH MODE: measure nearness instead of inferring it from the flow.
    //
    // With TAA_MV_WRITE_DEPTH the shader writes (currClip.w, prevClip.w), the
    // view-space depths in metres, so this reports what is actually close to
    // the lens. The residual tail was blamed on geometry about 0.2 m away, but
    // that distance was solved from the very flow under test. If the bottom
    // band really is centimetres from the camera the tail is a degenerate near
    // field; if it is metres, the flow there is too large and the field is
    // wrong.
    if (getenv("TAA_MV_WRITE_DEPTH")) {
        double rowDepSum[8]; uint64_t rowDepN[8]; double rowDepMin[8];
        for (int i = 0; i < 8; ++i) { rowDepSum[i]=0.0; rowDepN[i]=0; rowDepMin[i]=1e30; }
        const double depEdge[8] = { 0.05, 0.2, 0.5, 1.0, 5.0, 20.0, 100.0, 1e30 };
        uint64_t depHist[8]; for (int i=0;i<8;++i) depHist[i]=0;
        uint64_t depN = 0;
        for (uint32_t y = 0; y < in.h; y += 4) {
            for (uint32_t x = 0; x < in.w; x += 4) {
                const size_t i2 = ((size_t)y * in.w + x) * in.halves;
                const double dCur = velHalfToFloat(in.px[i2]);
                if (!(dCur > 0.0) || dCur != dCur) continue;
                ++depN;
                const int rb = (int)((double)y / (double)in.h * 8.0);
                if (rb >= 0 && rb < 8) {
                    rowDepSum[rb] += dCur; ++rowDepN[rb];
                    if (dCur < rowDepMin[rb]) rowDepMin[rb] = dCur;
                }
                for (int b = 0; b < 8; ++b)
                    if (dCur < depEdge[b]) { ++depHist[b]; break; }
            }
        }
        fprintf(f, "DEPTH MODE - target carries (currDepth, prevDepth) in metres\n");
        fprintf(f, "view=%d  %ux%u  samples with depth %llu\n\n",
                in.viewType, in.w, in.h, (unsigned long long)depN);
        fprintf(f, "TRUE VIEW DEPTH BY SCREEN ROW (top to bottom, eighths)\n");
        for (int b = 0; b < 8; ++b)
            fprintf(f, "  rows %4u-%4u  mean %10.3f m   nearest %10.4f m   n=%llu\n",
                    (unsigned)(in.h * b / 8), (unsigned)(in.h * (b + 1) / 8 - 1),
                    rowDepN[b] ? rowDepSum[b] / (double)rowDepN[b] : -1.0,
                    rowDepN[b] ? rowDepMin[b] : -1.0,
                    (unsigned long long)rowDepN[b]);
        // ---- prevDepth IS ALREADY IN CHANNEL 1 AND HAS NEVER BEEN LOOKED AT.
        //
        // prev.w = M[3]*x_c + M[7]*y_c + M[11]*w_c + M[15] is the denominator of
        // both output components. Over a 0.06 m camera move it must stay within
        // a few centimetres of currDepth. Where it does not, the denominator is
        // wrong and the whole pixel follows; where it tracks but the flow is
        // still large, the fault is in the numerator rows instead.
        //
        // This costs nothing - the channel is already being read.
        {
            double rowPD[8]; uint64_t rowPDN[8]; double rowPDMax[8];
            for (int i = 0; i < 8; ++i) { rowPD[i]=0.0; rowPDN[i]=0; rowPDMax[i]=0.0; }
            for (uint32_t y = 0; y < in.h; y += 4) {
                for (uint32_t x = 0; x < in.w; x += 4) {
                    const size_t i2 = ((size_t)y * in.w + x) * in.halves;
                    const double dCur = velHalfToFloat(in.px[i2]);
                    const double dPrv = velHalfToFloat(in.px[i2 + 1]);
                    if (!(dCur > 0.0) || dCur != dCur || dPrv != dPrv) continue;
                    const double rel = fabs(dPrv - dCur) / dCur;
                    const int rb = (int)((double)y / (double)in.h * 8.0);
                    if (rb >= 0 && rb < 8) {
                        rowPD[rb] += rel; ++rowPDN[rb];
                        if (rel > rowPDMax[rb]) rowPDMax[rb] = rel;
                    }
                }
            }
            fprintf(f, "\nPREV DEPTH AGAINST CURR DEPTH (the shared denominator)\n");
            for (int b = 0; b < 8; ++b)
                fprintf(f, "  rows %4u-%4u  mean |dPrev-dCurr|/dCurr %10.5f   worst %10.5f   n=%llu\n",
                        (unsigned)(in.h * b / 8), (unsigned)(in.h * (b + 1) / 8 - 1),
                        rowPDN[b] ? rowPD[b] / (double)rowPDN[b] : -1.0,
                        rowPDN[b] ? rowPDMax[b] : -1.0,
                        (unsigned long long)rowPDN[b]);
            fprintf(f, "      a 0.06 m camera move over 7 m geometry is under 1%%;\n");
            fprintf(f, "      a large value here means the denominator is wrong\n");
        }

        fprintf(f, "\nDEPTH HISTOGRAM\n");
        const char *dlbl[8] = { "under 0.05 m", "0.05-0.2 m", "0.2-0.5 m",
                                "0.5-1 m", "1-5 m", "5-20 m", "20-100 m",
                                "over 100 m" };
        for (int b = 0; b < 8; ++b)
            fprintf(f, "  %-14s %8llu  (%.2f%%)\n", dlbl[b],
                    (unsigned long long)depHist[b],
                    depN ? 100.0 * depHist[b] / (double)depN : 0.0);
        fprintf(f, "\n  The velocity run put the tail in rows 1890-2159 at 174 px\n");
        fprintf(f, "  and implied 0.2 m. Compare the nearest depth in that band.\n");
        fclose(f);
        return;
    }

    fprintf(f, "MOTION VECTOR DIAGNOSTIC\n");
    fprintf(f, "========================\n\n");
    fprintf(f, "view=%d phase=%d  target %ux%u  epipolar median %.3f px  relative %.5f\n",
            in.viewType, in.phase, in.w, in.h, in.medianAbsPx, in.medianFrac);
    fprintf(f, "camera moved %.4f m between the two frames\n\n", (double)g_diagDcMetres);

    fprintf(f, "PIXEL CENSUS (every 4th pixel both ways)\n");
    fprintf(f, "  sampled            %llu\n", (unsigned long long)nTotal);
    fprintf(f, "  exactly zero       %llu   (%.1f%%)  <- nothing drew, or drew and did not move\n",
            (unsigned long long)nZero, 100.0 * nZero / (nTotal ? nTotal : 1));
    fprintf(f, "  NaN                %llu\n", (unsigned long long)nNaN);
    fprintf(f, "  carrying motion    %llu\n", (unsigned long long)nMoving);
    fprintf(f, "  of those: within 1 px %llu, beyond %llu\n\n",
            (unsigned long long)nGoodPix, (unsigned long long)nBadPix);

    fprintf(f, "A. THE NEAR-FIELD SELECT\n");
    fprintf(f, "  1 block[18] at last push        %.6f   (0 means the select cannot fire)\n",
            (double)g_diagLastBlock18);
    fprintf(f, "  1 pixels where prev == flip(curr) %llu of %llu moving  (%.2f%%)\n",
            (unsigned long long)nFlipMatch, (unsigned long long)nMoving,
            100.0 * nFlipMatch / (nMoving ? nMoving : 1));
    fprintf(f, "      a large share here IS the flip firing, whatever block[18] reads\n");
    fprintf(f, "  2 patched binds with no push    %llu\n",
            (unsigned long long)g_diagPatchedBindsNoPush);
    fprintf(f, "  3 bodyReprojValid in this view  %d\n\n", g_diagBodyValid);

    fprintf(f, "  fitted k in prev.y = k * curr.y over the WRONG pixels: %.5f (n=%llu)\n",
            kN ? kSum / (double)kN : 0.0, (unsigned long long)kN);
    fprintf(f, "      k = -1 is the flip; k = 1 is correct; any other constant is a scale\n\n");

    fprintf(f, "B. VALIDITY VERY CLOSE IN\n");
    fprintf(f, "  4 predicted prev.w <= 0        %llu  (point was behind the previous camera)\n",
            (unsigned long long)nPrevWNeg);
    fprintf(f, "  5 near plane                   %.5f m   (proj[14]=%.5f)\n",
            fabs((double)in.proj[14]), (double)in.proj[14]);
    fprintf(f, "  6 largest |velocity|           %.5f uv = %.1f px;  half-float step there %.4f px\n\n",
            maxAbsUv, maxAbsUv * in.w,
            // half has 10 mantissa bits: step is 2^-10 of the exponent's scale
            (maxAbsUv > 0.0 ? (double)in.w * ldexp(maxAbsUv, -10) : 0.0));

    fprintf(f, "RESIDUAL BY MEASURED SPEED  (speed is a proxy for nearness)\n");
    const char *binName[kBins] = { "under 1 px", "1-4 px", "4-16 px",
                                   "16-64 px", "64-256 px", "over 256 px" };
    for (int b = 0; b < kBins; ++b)
        fprintf(f, "  %-12s  mean residual %10.3f px   n=%llu\n", binName[b],
                binResN[b] ? binResSum[b] / (double)binResN[b] : -1.0,
                (unsigned long long)binResN[b]);
    fprintf(f, "      rising with speed means near geometry; flat means it is not depth\n\n");

    fprintf(f, "RESIDUAL BY SCREEN ROW (top to bottom, eighths)\n");
    for (int r = 0; r < 8; ++r)
        fprintf(f, "  rows %4u-%4u  mean residual %10.3f px   n=%llu\n",
                r * in.h / 8, (r + 1) * in.h / 8 - 1,
                rowResN[r] ? rowResSum[r] / (double)rowResN[r] : -1.0,
                (unsigned long long)rowResN[r]);
    fprintf(f, "      a sharp horizontal boundary is screen-space; a gradual one is depth\n\n");

    fprintf(f, "18. WHICH MATRIX DOES THE FIELD MATCH?\n");
    {
        const uint64_t tot = winFull + winRot + winTrans + winNone;
        const uint64_t den = tot ? tot : 1;
        fprintf(f, "  full matrix      %8llu  (%.1f%%)\n",
                (unsigned long long)winFull, 100.0 * winFull / den);
        fprintf(f, "  rotation only    %8llu  (%.1f%%)  <- translation not reaching the shader\n",
                (unsigned long long)winRot, 100.0 * winRot / den);
        fprintf(f, "  translation only %8llu  (%.1f%%)\n",
                (unsigned long long)winTrans, 100.0 * winTrans / den);
        fprintf(f, "  none within 16px %8llu  (%.1f%%)\n\n",
                (unsigned long long)winNone, 100.0 * winNone / den);
    }

    fprintf(f, "ANGLE BETWEEN MEASURED AND PREDICTED FLOW\n");
    fprintf(f, "  mean signed %+8.3f deg, mean absolute %8.3f deg, n=%llu\n",
            angN ? angSum / (double)angN : 0.0,
            angN ? angAbsSum / (double)angN : 0.0, (unsigned long long)angN);
    fprintf(f, "      a consistent signed angle is a rotation applied somewhere;\n");
    fprintf(f, "      large absolute with near-zero mean is scatter, not a transform\n\n");

    fprintf(f, "RAW PIXELS (residual over 8 px) - flow in pixels\n");
    fprintf(f, "  %6s %6s | %9s %9s | %9s %9s | %9s %9s | %8s\n",
            "x", "y", "meas dx", "meas dy", "full dx", "full dy",
            "rot dx", "rot dy", "resid");
    for (size_t i = 0; i < rows.size(); ++i)
        fprintf(f, "  %6u %6u | %9.2f %9.2f | %9.2f %9.2f | %9.2f %9.2f | %8.2f\n",
                rows[i].x, rows[i].y, rows[i].mvxPx, rows[i].mvyPx,
                rows[i].pfxPx, rows[i].pfyPx, rows[i].prxPx, rows[i].pryPx,
                rows[i].resid);
    fprintf(f, "\n");

    fprintf(f, "D. PASSES\n");
    fprintf(f, "  13 qualifying passes this frame %u, of which bound %u\n",
            g_diagQualifyingPasses, g_diagBoundPasses);
    fprintf(f, "      more than one bound means two passes wrote the same target\n\n");

    fprintf(f, "E. THE MEASUREMENT ITSELF\n");
    fprintf(f, "  16 skipped as exactly zero      %llu of %llu\n",
            (unsigned long long)nZero, (unsigned long long)nTotal);
    fprintf(f, "  17 predicted prev off-screen    %llu of %llu moving (%.1f%%)\n\n",
            (unsigned long long)nPrevOffScreen, (unsigned long long)nMoving,
            100.0 * nPrevOffScreen / (nMoving ? nMoving : 1));

    fprintf(f, "THE MATRIX THE DRAWS WERE GIVEN\n");
    for (int r = 0; r < 4; ++r)
        fprintf(f, "  [%d]=%10.5f [%d]=%10.5f [%d]=%10.5f [%d]=%10.5f\n",
                r, (double)M[r], r+4, (double)M[r+4],
                r+8, (double)M[r+8], r+12, (double)M[r+12]);
    fprintf(f, "\nTHE PROJECTION\n");
    for (int r = 0; r < 4; ++r)
        fprintf(f, "  [%d]=%10.5f [%d]=%10.5f [%d]=%10.5f [%d]=%10.5f\n",
                r, (double)in.proj[r], r+4, (double)in.proj[r+4],
                r+8, (double)in.proj[r+8], r+12, (double)in.proj[r+12]);

    fclose(f);
    trace("MV DIAG: wrote %s", path);
}
