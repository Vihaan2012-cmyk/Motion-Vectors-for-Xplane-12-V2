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
    if (getenv("TAA_MV_RGBA") && in.halves >= 4) {
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
                    const int pid = (int)(pf + 0.5);
                    if (pid <= 0) continue;
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
            fprintf(f, "\nERROR BY FRAGMENT SHADER (channel 3 tag)\n");
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
