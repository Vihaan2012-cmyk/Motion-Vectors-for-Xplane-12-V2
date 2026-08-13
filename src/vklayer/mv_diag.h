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

            // Predicted flow under each of the three matrices, at the depth
            // the measured motion implies for the FULL matrix. The same depth
            // is reused for all three so they are compared on one geometry.
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
