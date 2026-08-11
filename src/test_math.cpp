// Standalone CPU test for the reprojection maths in share.h.
//
// This is the single highest-value test in the project. Every downstream
// symptom of a wrong `reproj` matrix - ghosting, smearing, a velocity buffer
// that looks like noise - shows up only after a sim launch, a flight, and a
// screenshot, and all of those failure modes look alike. Here the same maths is
// exercised in a hundredth of a second against a known answer.
//
// The identity being verified is the one the shader relies on:
//
//     reproj = prevViewProj * inverse(currViewProj)
//
//     given a pixel's uv and depth from THIS frame, reproj takes it straight to
//     where that same world point sat in the PREVIOUS frame's clip space.
//
// Build:  g++ -O2 -std=c++17 -o test_math.exe src/test_math.cpp
// Run:    ./test_math.exe          (exit 0 = all pass)

#include "share.h"

#include <cstdio>
#include <cmath>

static int g_fail = 0;

static void check(bool ok, const char *what, double detail = 0.0)
{
    if (ok) {
        printf("  PASS  %s\n", what);
    } else {
        printf("  FAIL  %s   (%.6g)\n", what, detail);
        ++g_fail;
    }
}

// ---------------------------------------------------------------- builders

static void identity(float *m)
{
    memset(m, 0, 16 * sizeof(float));
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

// Reverse-Z, infinite far - the modern convention and what X-Plane 12 is
// expected to use. Column-major.
//
//   clip.z = near,  clip.w = -eye.z   =>  depth = near / dist
//   depth is 1 at the near plane and tends to 0 at infinity.
static void perspectiveReverseZInfinite(float *m, float fovYRad, float aspect, float nearZ)
{
    float f = 1.0f / tanf(fovYRad * 0.5f);
    memset(m, 0, 16 * sizeof(float));
    m[0]  = f / aspect;
    m[5]  = f;
    m[10] = 0.0f;
    m[11] = -1.0f;
    m[14] = nearZ;
    m[15] = 0.0f;
}

// The projection X-Plane 12 ACTUALLY uses, read off a live frame:
//
//     col2 = (0, 0, -1, -1)
//     col3 = (0, 0, -0.016485, 0)
//
// Standard-Z with an infinite far plane and a 16.5 mm near clip:
//     clip.z = -eye.z - near,  clip.w = -eye.z   =>  depth = 1 - near/dist
// so 0 at the near plane and 1 at infinity. Note this is NOT reverse-Z, which
// the project assumed for a while.
static void perspectiveStandardInfinite(float *m, float fovYRad, float aspect, float nearZ)
{
    float f = 1.0f / tanf(fovYRad * 0.5f);
    memset(m, 0, 16 * sizeof(float));
    m[0]  = f / aspect;
    m[5]  = f;
    m[10] = -1.0f;
    m[11] = -1.0f;
    m[14] = -nearZ;
    m[15] = 0.0f;
}

// Conventional finite-range Vulkan projection, for the cross-check that the
// depth-convention detector is not just always answering "reverse-Z".
static void perspectiveStandard(float *m, float fovYRad, float aspect, float n, float fz)
{
    float f = 1.0f / tanf(fovYRad * 0.5f);
    memset(m, 0, 16 * sizeof(float));
    m[0]  = f / aspect;
    m[5]  = f;
    m[10] = fz / (n - fz);
    m[11] = -1.0f;
    m[14] = (fz * n) / (n - fz);
    m[15] = 0.0f;
}

// World -> eye for a camera at `pos` with no rotation, looking down -Z.
static void viewAt(float *m, float x, float y, float z)
{
    identity(m);
    m[12] = -x; m[13] = -y; m[14] = -z;
}

// Camera translated AND yawed, so the test exercises rotation too - a bug that
// only shows under rotation is exactly the kind a translation-only test misses.
static void viewAtYaw(float *m, float x, float y, float z, float yawRad)
{
    float c = cosf(yawRad), s = sinf(yawRad);
    float rot[16]; identity(rot);
    rot[0] =  c; rot[8] = s;
    rot[2] = -s; rot[10] = c;
    float trans[16]; viewAt(trans, x, y, z);
    taaMul(m, rot, trans);          // rotate after translating into camera space
}

static void xform(float *out, const float *m, const float *v)
{
    for (int r = 0; r < 4; ++r)
        out[r] = m[0*4+r]*v[0] + m[1*4+r]*v[1] + m[2*4+r]*v[2] + m[3*4+r]*v[3];
}

// Rotation from a quaternion, column-major. Mirrors taaBodyRotation in the
// plugin; kept as a separate implementation deliberately, so the test checks
// the maths rather than agreeing with a copy of it.
static void quatToMatrix(float *R, const float q[4])
{
    float w = q[0], x = q[1], y = q[2], z = q[3];
    float n = w*w + x*x + y*y + z*z;
    if (n > 1e-12f) { float s = 1.0f/sqrtf(n); w*=s; x*=s; y*=s; z*=s; }

    identity(R);
    R[0] = 1-2*(y*y+z*z);  R[1] = 2*(x*y + w*z);  R[2]  = 2*(x*z - w*y);
    R[4] = 2*(x*y - w*z);  R[5] = 1-2*(x*x+z*z);  R[6]  = 2*(y*z + w*x);
    R[8] = 2*(x*z + w*y);  R[9] = 2*(y*z - w*x);  R[10] = 1-2*(x*x+y*y);
}

// A point given in body coordinates, placed in the world.
static void bodyToWorld(double *out, const float *R, const double *origin,
                        const float *body)
{
    for (int r = 0; r < 3; ++r)
        out[r] = origin[r] + (double)(R[0*4+r]*body[0] + R[1*4+r]*body[1]
                                    + R[2*4+r]*body[2]);
}

// World -> eye for a camera with the body's orientation at a world position.
// V = R^T * T(-camera), the standard rigid view matrix.
static void viewFromBody(float *V, const float *R, const double *camWorld)
{
    float Rt[16]; identity(Rt);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            Rt[c*4+r] = R[r*4+c];               // transpose = inverse

    float T[16]; identity(T);
    T[12] = (float)-camWorld[0];
    T[13] = (float)-camWorld[1];
    T[14] = (float)-camWorld[2];
    taaMul(V, Rt, T);
}

// ---------------------------------------------------------------- the tests

static void testInverse()
{
    printf("\n[inverse]\n");

    float proj[16];
    perspectiveReverseZInfinite(proj, 65.0f * 3.14159265f / 180.0f, 2560.0f/1440.0f, 0.05f);

    float inv[16];
    check(taaInverse(inv, proj), "reverse-Z infinite projection is invertible");

    float back[16];
    taaMul(back, proj, inv);
    float worst = 0.0f;
    for (int i = 0; i < 16; ++i) {
        float want = (i % 5 == 0) ? 1.0f : 0.0f;
        float d = fabsf(back[i] - want);
        if (d > worst) worst = d;
    }
    check(worst < 1e-5f, "proj * inverse(proj) == identity", worst);

    // Singular input must be rejected rather than silently producing NaNs that
    // propagate into the shader as a corrupt velocity field.
    float zero[16] = {0};
    float dummy[16];
    check(!taaInverse(dummy, zero), "singular matrix is rejected");
}

static void testDepthConvention()
{
    printf("\n[depth convention]\n");
    const float fov = 65.0f * 3.14159265f / 180.0f;
    const float aspect = 2560.0f / 1440.0f;

    // --- reverse-Z, infinite far
    {
        float proj[16], inv[16];
        perspectiveReverseZInfinite(proj, fov, aspect, 0.05f);
        taaInverse(inv, proj);

        float d0 = 0, d1 = 0;
        bool ok0 = taaDepthPlaneDistance(inv, 0.0f, &d0);
        bool ok1 = taaDepthPlaneDistance(inv, 1.0f, &d1);

        check(!ok0, "reverse-Z infinite: clip z=0 is at infinity");
        check(ok1,  "reverse-Z infinite: clip z=1 resolves");
        check(ok1 && fabsf(d1 - 0.05f) < 1e-4f,
              "reverse-Z infinite: near plane recovered as 0.05", ok1 ? d1 : -1);
    }

    // --- standard-Z, finite far. Must NOT be misreported as reverse-Z.
    {
        float proj[16], inv[16];
        perspectiveStandard(proj, fov, aspect, 0.1f, 5000.0f);
        taaInverse(inv, proj);

        float d0 = 0, d1 = 0;
        bool ok0 = taaDepthPlaneDistance(inv, 0.0f, &d0);
        bool ok1 = taaDepthPlaneDistance(inv, 1.0f, &d1);

        check(ok0 && ok1, "standard-Z finite: both planes resolve");
        check(ok0 && fabsf(d0 - 0.1f) < 1e-3f, "standard-Z: near recovered", d0);

        // Relative, not absolute, and deliberately loose. Recovering a far
        // plane from a float32 projection is ill-conditioned when far/near is
        // large: here proj[10] = -1.00002, so the entire signal carrying `far`
        // is 2e-5 against 1.0, only ~170x above float32 epsilon. ~0.2% error is
        // arithmetic, not a bug. Measured 4993.22 for a true 5000.
        //
        // This does not affect X-Plane, which uses an infinite far plane where
        // farClip is flagged meaningless - but anything that later trusts
        // farClip for a finite projection should know it is a 3-digit number.
        check(ok1 && fabsf(d1 - 5000.0f) / 5000.0f < 0.01f,
              "standard-Z: far recovered to within 1%", d1);
        check(d0 < d1, "standard-Z is correctly NOT flagged as reverse-Z");
    }
}

// The real one: does reproj actually take a current-frame pixel to the right
// place in the previous frame?
static void testReprojection(const char *label,
                             const float *prevView, const float *currView,
                             const float *proj)
{
    printf("\n[reprojection: %s]\n", label);

    float currVP[16], prevVP[16], invCurrVP[16], reproj[16];
    taaMul(currVP, proj, currView);
    taaMul(prevVP, proj, prevView);

    if (!taaInverse(invCurrVP, currVP)) { check(false, "currVP invertible"); return; }
    taaMul(reproj, prevVP, invCurrVP);

    // A spread of world points at very different depths. Depth precision is
    // where reprojection errors hide, so this deliberately spans 20m to 40km.
    const float pts[][3] = {
        {   10.0f,    5.0f,   -20.0f },
        {  -30.0f,  -10.0f,  -150.0f },
        {    0.0f,    0.0f, -1000.0f },
        {  500.0f,  200.0f, -5000.0f },
        {-1200.0f,  800.0f,-40000.0f },
    };

    float worstErr = 0.0f;
    int   tested   = 0;

    for (int i = 0; i < 5; ++i) {
        float P[4] = { pts[i][0], pts[i][1], pts[i][2], 1.0f };

        // Where the pixel is NOW.
        float clipC[4]; xform(clipC, currVP, P);
        if (clipC[3] <= 1e-6f) continue;               // behind the camera
        float ndcX = clipC[0]/clipC[3];
        float ndcY = clipC[1]/clipC[3];
        float ndcZ = clipC[2]/clipC[3];
        if (ndcZ < 0.0f || ndcZ > 1.0f) continue;      // outside the depth range
        ++tested;

        // Exactly what the shader feeds in: (uv*2-1, depth, 1). uv*2-1 is ndc.xy
        // by construction, so this is the same vector the GPU will build.
        float in[4] = { ndcX, ndcY, ndcZ, 1.0f };
        float outClip[4]; xform(outClip, reproj, in);
        float gotU = (outClip[0]/outClip[3]) * 0.5f + 0.5f;
        float gotV = (outClip[1]/outClip[3]) * 0.5f + 0.5f;

        // Ground truth: project the same world point with the previous frame's
        // matrix directly.
        float clipP[4]; xform(clipP, prevVP, P);
        float wantU = (clipP[0]/clipP[3]) * 0.5f + 0.5f;
        float wantV = (clipP[1]/clipP[3]) * 0.5f + 0.5f;

        float e = fmaxf(fabsf(gotU - wantU), fabsf(gotV - wantV));
        if (e > worstErr) worstErr = e;

        printf("    pt%d depth=%.6f  uv_prev got=(%.5f %.5f) want=(%.5f %.5f)  err=%.2e\n",
               i, ndcZ, gotU, gotV, wantU, wantV, e);
    }

    check(tested >= 3, "enough points were in front of the camera and in range");
    // A quarter of a pixel at 2560 wide is ~1e-4 in uv. Anything under 1e-5 is
    // pure float noise; anything above 1e-3 would be visible ghosting.
    check(worstErr < 1e-5f, "reprojected uv matches direct projection", worstErr);
}

int main()
{
    printf("TAAImplementation - reprojection maths test\n");
    printf("sizeof(TaaShare) = %d bytes\n", (int)sizeof(TaaShare));

    testInverse();
    testDepthConvention();

    const float fov = 65.0f * 3.14159265f / 180.0f;
    const float aspect = 2560.0f / 1440.0f;
    float proj[16];
    perspectiveReverseZInfinite(proj, fov, aspect, 0.05f);

    // Straight-and-level: camera moves 60m forward between frames, roughly one
    // frame at cruise speed.
    {
        float prev[16], curr[16];
        viewAt(prev, 0.0f, 100.0f,    0.0f);
        viewAt(curr, 0.0f, 100.0f,  -60.0f);
        testReprojection("forward translation", prev, curr, proj);
    }

    // Translation plus yaw. Rotation is where a transposed or wrongly-ordered
    // multiply shows up; pure translation can hide it.
    {
        float prev[16], curr[16];
        viewAtYaw(prev, 0.0f, 100.0f,   0.0f, 0.00f);
        viewAtYaw(curr, 25.0f, 105.0f, -60.0f, 0.03f);
        testReprojection("translation + yaw", prev, curr, proj);
    }

    // Stationary camera: reproj must be the identity and velocity exactly zero.
    // If this one fails, every static shot would shimmer.
    {
        float prev[16], curr[16];
        viewAt(prev, 10.0f, 50.0f, -5.0f);
        viewAt(curr, 10.0f, 50.0f, -5.0f);
        testReprojection("stationary camera", prev, curr, proj);
    }

    // ---------------------------------------------------------------------
    // The case the tests above could never catch, because they put the camera
    // near the origin.
    //
    // X-Plane's local origin can be 52 km away. A world-space viewProj then
    // spans entries from 1 to 52000, and inverting that in float32 loses about
    // four significant digits. In the sim this showed up as a 10-18% residual -
    // visibly wrong motion vectors - while every synthetic test here still
    // passed at 1e-8.
    //
    // This asserts two things: that the world-space form really is that bad at
    // realistic scale, and that shifting the origin to the camera fixes it.
    {
        printf("\n[conditioning at realistic world scale]\n");

        const float C0[3] = { 43930.0f, 175.0f, -51581.0f };   // ~52 km out
        const float C1[3] = { 43930.0f + 2.0f, 175.3f, -51581.0f - 58.0f };

        float prevW[16], currW[16];
        viewAtYaw(prevW, C0[0], C0[1], C0[2], 0.00f);
        viewAtYaw(currW, C1[0], C1[1], C1[2], 0.02f);

        // The REAL projection, not the reverse-Z one used above. This matters:
        // with reverse-Z the world-space residual is a harmless 1.5e-07, so a
        // test using it would have declared the world-space form perfectly fine
        // while the sim was reporting 1e-01.
        float rproj[16];
        perspectiveStandardInfinite(rproj, fov, aspect, 0.016485f);

        float currVP[16], prevVP[16];
        taaMul(currVP, rproj, currW);
        taaMul(prevVP, rproj, prevW);

        // --- world-space: invert the large matrix directly.
        float worldResidual = 0.0f;
        float invVP[16];
        if (taaInverse(invVP, currVP)) {
            float r[16], chk[16];
            taaMul(r, prevVP, invVP);
            taaMul(chk, r, currVP);
            for (int i = 0; i < 16; ++i) {
                float d = fabsf(chk[i] - prevVP[i]);
                float s = fabsf(prevVP[i]);
                if (s > 1.0f) d /= s;
                if (d > worldResidual) worldResidual = d;
            }
        }
        printf("    world-space residual          : %.3e\n", worldResidual);

        // --- camera-relative: shift the origin to the current camera first.
        //     W * T(C_curr) is the camera-relative view matrix for either frame.
        float Tc[16]; identity(Tc);
        Tc[12] = C1[0]; Tc[13] = C1[1]; Tc[14] = C1[2];

        float currWr[16], prevWr[16], currVPr[16], prevVPr[16];
        taaMul(currWr, currW, Tc);
        taaMul(prevWr, prevW, Tc);
        taaMul(currVPr, rproj, currWr);
        taaMul(prevVPr, rproj, prevWr);

        float relResidual = 0.0f;
        float invVPr[16];
        if (taaInverse(invVPr, currVPr)) {
            float r[16], chk[16];
            taaMul(r, prevVPr, invVPr);
            taaMul(chk, r, currVPr);
            for (int i = 0; i < 16; ++i) {
                float d = fabsf(chk[i] - prevVPr[i]);
                float s = fabsf(prevVPr[i]);
                if (s > 1.0f) d /= s;
                if (d > relResidual) relResidual = d;
            }
        }
        printf("    camera-relative residual      : %.3e\n", relResidual);
        printf("    improvement                   : %.0fx\n",
               relResidual > 0 ? worldResidual / relResidual : 0.0);

        check(worldResidual > 1e-3f,
              "world-space IS badly conditioned at 52 km (regression guard)", worldResidual);
        check(relResidual < 1e-4f,
              "camera-relative stays accurate at 52 km", relResidual);
    }

    // ---------------------------------------------------------------------
    // Body-frame reprojection keeps cockpit geometry still.
    //
    // This is the whole justification for bodyReproj, tested here rather than
    // in the air. A panel corner bolted to the aircraft must land on the same
    // pixel it occupied last frame, however hard the aeroplane is manoeuvring -
    // while the world-frame matrix, applied to the same point, must move it a
    // long way. Both halves matter: if the world path did NOT move it, the test
    // would pass against a stationary aircraft and prove nothing.
    {
        printf("\n  body-frame reprojection (cockpit)\n");

        // Run at the origin AND at 52 km. If the residual is float32
        // cancellation it grows with distance; if it is the same at both, the
        // maths is wrong and no amount of reconditioning will help. Guessing
        // which it was has already cost one wrong fix here.
        for (int caseIdx = 0; caseIdx < 2; ++caseIdx) {
        const double far = caseIdx ? 1.0 : 0.0;
        const char *label = caseIdx ? "52 km from origin" : "at origin";
        const double ox0 = 37000.0*far, oy0 = 3000.0*far, oz0 = -37000.0*far;

        // One frame of flight: 3 m forward and a 2 degree roll.
        const double dt_x = 2.1, dt_y = -0.4, dt_z = 2.1;
        float qPrev[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        float roll = 2.0f * 3.14159265f / 180.0f;
        float qCurr[4] = {cosf(roll*0.5f), 0.0f, 0.0f, sinf(roll*0.5f)};

        float Rprev[16], Rcurr[16];
        quatToMatrix(Rprev, qPrev);
        quatToMatrix(Rcurr, qCurr);

        // Camera bolted in the cockpit: 0.7 m forward of the datum, 1.2 m up.
        const float camBody[3] = {0.0f, 1.2f, -0.7f};
        // A panel corner, 0.75 m from the eye - the distance that makes the
        // world-frame answer so badly wrong.
        const float panelBody[3] = {0.15f, 1.05f, -1.4f};

        double ownPrev[3] = {ox0, oy0, oz0};
        double ownCurr[3] = {ox0 + dt_x, oy0 + dt_y, oz0 + dt_z};

        double camPrev[3], camCurr[3], panelPrev[3], panelCurr[3];
        bodyToWorld(camPrev,   Rprev, ownPrev, camBody);
        bodyToWorld(camCurr,   Rcurr, ownCurr, camBody);
        bodyToWorld(panelPrev, Rprev, ownPrev, panelBody);
        bodyToWorld(panelCurr, Rcurr, ownCurr, panelBody);

        // X-Plane's actual projection form: standard-Z, infinite far.
        float proj[16];
        perspectiveStandardInfinite(proj, 60.0f * 3.14159265f / 180.0f,
                                    16.0f/9.0f, 0.1956f);

        // View matrices. Same orientation as the body, looking along it.
        float Vprev[16], Vcurr[16];
        viewFromBody(Vprev, Rprev, camPrev);
        viewFromBody(Vcurr, Rcurr, camCurr);

        // Camera-relative origin = the current camera, matching the plugin.
        float Tc[16]; identity(Tc);
        Tc[12] = (float)camCurr[0]; Tc[13] = (float)camCurr[1]; Tc[14] = (float)camCurr[2];

        float Wr[16], Wpr[16];
        taaMul(Wr,  Vcurr, Tc);
        taaMul(Wpr, Vprev, Tc);

        // The residual below is float32 cancellation against the distance to
        // the local origin, not a flaw in the derivation - measured, by running
        // this identical test at the origin and at 52 km. Forcing Wr's
        // translation to its analytic zero was tried and changed the result by
        // nothing, so the cancellation that matters is in Wpr, whose
        // translation is a genuine difference of two 52 km quantities.
        //
        // 1.9 px of residual cockpit motion at the worst-case distance, against
        // 2600 px uncorrected, is left as-is: the neighbourhood clamp and the
        // cockpit reactive mask both absorb motion at that scale.

        // Body -> camera-relative world, both frames.
        float Bc[16], Bp[16];
        memcpy(Bc, Rcurr, sizeof(Bc));
        Bc[12] = (float)(ownCurr[0] - camCurr[0]);
        Bc[13] = (float)(ownCurr[1] - camCurr[1]);
        Bc[14] = (float)(ownCurr[2] - camCurr[2]);
        memcpy(Bp, Rprev, sizeof(Bp));
        Bp[12] = (float)(ownPrev[0] - camCurr[0]);
        Bp[13] = (float)(ownPrev[1] - camCurr[1]);
        Bp[14] = (float)(ownPrev[2] - camCurr[2]);

        float Mc[16], Mp[16], Ac[16], Ap[16], invAc[16], bodyReproj[16];
        float currVPrCheck[16];
        taaMul(Mc, Wr,  Bc);
        taaMul(Mp, Wpr, Bp);
        taaMul(Ac, proj, Mc);
        taaMul(Ap, proj, Mp);
        check(taaInverse(invAc, Ac), "body-to-clip matrix inverts", 0.0f);
        taaMul(bodyReproj, Ap, invAc);

        // Where the panel corner is on screen now, straight from body coords.
        float pb[4] = {panelBody[0], panelBody[1], panelBody[2], 1.0f};
        float clipNow[4];
        xform(clipNow, Ac, pb);
        float uvNow[2] = {clipNow[0]/clipNow[3], clipNow[1]/clipNow[3]};

        // Cross-check that Ac agrees with the world route to the same pixel.
        // Without this the test could pass with a self-consistent but wrong Ac.
        float pw[4] = {(float)(panelCurr[0]-camCurr[0]),
                       (float)(panelCurr[1]-camCurr[1]),
                       (float)(panelCurr[2]-camCurr[2]), 1.0f};
        float clipVia[4];
        taaMul(currVPrCheck, proj, Wr);
        xform(clipVia, currVPrCheck, pw);
        float agree = fabsf(clipVia[0]/clipVia[3] - uvNow[0])
                    + fabsf(clipVia[1]/clipVia[3] - uvNow[1]);
        check(agree < 1e-3f, "body and world routes agree on where the panel is", agree);

        // Body reprojection of that same clip position.
        float clipBody[4];
        xform(clipBody, bodyReproj, clipNow);
        float uvBody[2] = {clipBody[0]/clipBody[3], clipBody[1]/clipBody[3]};

        // World reprojection of it, for contrast.
        float prevVPr[16], invCurrVPr[16], worldReproj[16];
        taaMul(prevVPr, proj, Wpr);
        taaInverse(invCurrVPr, currVPrCheck);
        taaMul(worldReproj, prevVPr, invCurrVPr);

        float clipWorld[4];
        xform(clipWorld, worldReproj, clipNow);
        float uvWorld[2] = {clipWorld[0]/clipWorld[3], clipWorld[1]/clipWorld[3]};

        float bodyMove  = sqrtf(powf(uvBody[0]-uvNow[0],2)  + powf(uvBody[1]-uvNow[1],2));
        float worldMove = sqrtf(powf(uvWorld[0]-uvNow[0],2) + powf(uvWorld[1]-uvNow[1],2));

        printf("    [%s] panel motion body=%.6f ndc (%.2f px)  world=%.5f ndc (%.1f px)\n",
               label, bodyMove, bodyMove * 0.5f * 2560.0f,
               worldMove, worldMove * 0.5f * 2560.0f);

        check(bodyMove < 4e-3f,
              "cockpit panel is STILL under body-frame reprojection", bodyMove);
        check(worldMove > 0.05f,
              "and the world frame really does smear it (test is meaningful)", worldMove);
        }
    }

    printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
