// Offline tests for the crash destruction maths.
//
// None of this needs X-Plane. Detection, grid classification, integration and
// the constraint solver are arithmetic, and arithmetic that can be tested at a
// command line should never be tested by flying an aeroplane into the ground -
// which is slow, unrepeatable, and cannot be run before a commit.

#include "destruct/trigger.h"
#include "destruct/occupancy.h"
#include "destruct/solver.h"
#include "destruct/gpu_layout.h"
#include "destruct/bounds.h"
#include "destruct/acf_planform.h"
#include <cstdio>
#include <cmath>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char *what)
{
    if (ok) { printf("  PASS  %s\n", what); }
    else    { printf("  FAIL  %s\n", what); ++g_fail; }
}

static void checkNear(double got, double want, double tol, const char *what)
{
    bool ok = std::fabs(got - want) <= tol;
    if (ok) printf("  PASS  %s (%.3f)\n", what, got);
    else  { printf("  FAIL  %s: got %.3f want %.3f +/- %.3f\n",
                   what, got, want, tol); ++g_fail; }
}

// ---- A flight profile, fed to the detector one frame at a time.
//
// Runs the aeroplane airborne long enough to arm, then puts it on the ground
// with the given descent rate and G spike, then holds it there. Returns the
// event, if any.
static destruct::Event fly(destruct::Detector &d, float contactFpm,
                           float spikeG, int spikeAtFrame = 2,
                           int airborneFrames = 90)
{
    destruct::Event out;
    out.fired = 0;
    destruct::Sample s;
    s.gNrml = 1.0f; s.gAxil = 0.0f; s.gSide = 0.0f;

    // Airborne, descending.
    s.onGround = 0;
    s.vhFpm = contactFpm;
    for (int i = 0; i < airborneFrames; ++i) destruct::step(d, s);

    // On the ground. The spike lands a couple of frames after contact, which
    // is what a real impact looks like - the gear compresses first.
    s.onGround = 1;
    for (int i = 0; i < 40; ++i) {
        s.gNrml = (i == spikeAtFrame) ? spikeG : 1.0f;
        s.vhFpm = (i == 0) ? contactFpm : 0.0f;
        destruct::Event e = destruct::step(d, s);
        if (e.fired) out = e;
    }
    return out;
}

int main()
{
    printf("destruct tests\n");

    const destruct::Thresholds T = destruct::defaultThresholds();

    // ---------------------------------------------------------- classify
    printf("\nseverity classification\n");
    check(destruct::classify(T, 1.3f, -200.0f) == destruct::SEV_NONE,
          "gentle touchdown is not an event");
    check(destruct::classify(T, 2.5f, -300.0f) == destruct::SEV_HARD,
          "2.5 g is a hard landing");
    check(destruct::classify(T, 1.4f, -800.0f) == destruct::SEV_HARD,
          "800 fpm is a hard landing even at low g");
    check(destruct::classify(T, 8.0f, -400.0f) == destruct::SEV_STRUCT,
          "8 g is structural");
    check(destruct::classify(T, 1.5f, -1400.0f) == destruct::SEV_STRUCT,
          "1400 fpm is structural even at low g");
    check(destruct::classify(T, 22.0f, -3000.0f) == destruct::SEV_BREAKUP,
          "22 g is a breakup");

    // ---------------------------------------------------------- detector
    printf("\ndetector over a flight profile\n");
    {
        destruct::Detector d = destruct::makeDetector(T);
        destruct::Event e = fly(d, -150.0f, 1.3f);
        check(!e.fired, "a normal landing produces no event");
    }
    {
        destruct::Detector d = destruct::makeDetector(T);
        destruct::Event e = fly(d, -250.0f, 3.0f);
        check(e.fired && e.severity == destruct::SEV_HARD,
              "a 3 g arrival fires as a hard landing");
        checkNear(e.peakG, 3.0, 0.01, "peak g is the spike, not the average");
        checkNear(e.contactFpm, -250.0, 0.01,
                  "contact fpm is sampled AT touchdown, not after");
    }
    {
        destruct::Detector d = destruct::makeDetector(T);
        destruct::Event e = fly(d, -2500.0f, 18.0f);
        check(e.fired && e.severity == destruct::SEV_BREAKUP,
              "a terrain impact fires as a breakup");
        check(e.impulse[1] > 10.0f,
              "the impulse carries the normal axis for seeding fragments");
    }

    // ---- The failure this detector exists to avoid.
    //
    // g_nrml is noisy and a parked airframe trembles on its gear with the
    // engines running. A single-frame comparison against the threshold fires on
    // taxiing. The peak-over-window plus the airborne arming requirement is
    // what stops it.
    printf("\nnoise rejection\n");
    {
        destruct::Detector d = destruct::makeDetector(T);
        destruct::Sample s;
        s.onGround = 1; s.vhFpm = 0.0f;
        s.gAxil = 0.0f; s.gSide = 0.0f;
        bool fired = false;
        for (int i = 0; i < 600; ++i) {
            // Taxi rumble: 1 g with spikes to 2.5 that never last.
            s.gNrml = (i % 7 == 0) ? 2.5f : 1.0f;
            if (destruct::step(d, s).fired) fired = true;
        }
        check(!fired, "taxi rumble never fires - never airborne, never armed");
    }
    {
        // Starting a flight already on the ground must not fire either, even
        // with a jolt, because the detector has never seen the aeroplane fly.
        destruct::Detector d = destruct::makeDetector(T);
        destruct::Sample s;
        s.onGround = 1; s.vhFpm = -1500.0f;
        s.gNrml = 20.0f; s.gAxil = 0.0f; s.gSide = 0.0f;
        bool fired = false;
        for (int i = 0; i < 60; ++i) if (destruct::step(d, s).fired) fired = true;
        check(!fired, "a cold start on the ground cannot fire");
    }

    // ---- One impact, one event.
    printf("\nlatching\n");
    {
        destruct::Detector d = destruct::makeDetector(T);
        destruct::Event first = fly(d, -1200.0f, 9.0f);
        check(first.fired, "the first impact fires");

        // A bounce: briefly airborne, but not long enough to re-arm, then
        // another heavy contact. That is the SAME arrival and must not fire
        // again, or a bounce down the runway is four crashes.
        destruct::Sample s;
        s.gNrml = 1.0f; s.gAxil = 0.0f; s.gSide = 0.0f;
        s.onGround = 0; s.vhFpm = -900.0f;
        for (int i = 0; i < 20; ++i) destruct::step(d, s);   // < rearmFrames
        s.onGround = 1;
        bool again = false;
        for (int i = 0; i < 40; ++i) {
            s.gNrml = (i == 2) ? 9.0f : 1.0f;
            if (destruct::step(d, s).fired) again = true;
        }
        check(!again, "a bounce does not fire a second event");
    }
    {
        // A genuinely separate flight - properly airborne again - must be able
        // to fire, or the system only ever works once per session.
        destruct::Detector d = destruct::makeDetector(T);
        destruct::Event first = fly(d, -1200.0f, 9.0f);
        check(first.fired, "first flight fires");
        destruct::Event second = fly(d, -1300.0f, 10.0f);
        check(second.fired, "a later flight fires again after re-arming");
    }

    // ---------------------------------------------------------- grid
    printf("\ngrid geometry\n");
    {
        // An A320-sized box. The design target is 800-2000 fragments for an
        // airliner once occupancy is applied, so the raw cell count has to be
        // an order of magnitude above that.
        const float lo[3] = { -19.0f, -6.0f, -18.0f };
        const float hi[3] = {  19.0f,  6.0f,  18.0f };
        destruct::Grid g = destruct::gridForBounds(lo, hi);
        checkNear(g.cell, 0.94, 0.25, "airliner cell size lands in the debris range");
        check(destruct::gridCells(g) > 8000, "airliner grid is dense enough");

        const float mid[3]  = { 0.0f, 0.0f, 0.0f };
        const float away[3] = { 500.0f, 0.0f, 0.0f };
        check(destruct::gridClassify(g, mid)  >= 0, "centre classifies");
        check(destruct::gridClassify(g, away) == -1, "far point is outside");

        // Adjacent on x must mean adjacent in index: the constraint graph
        // walks neighbours by index arithmetic and depends on this exactly.
        const float a[3] = { -18.5f, 0.0f, 0.0f };
        const float b[3] = { -18.5f + 1.2f, 0.0f, 0.0f };
        check(destruct::gridClassify(g, b) - destruct::gridClassify(g, a) == 1,
              "adjacent x cells are adjacent indices");

        // A cell centre must classify back into its own cell. If it does not,
        // the rest positions the physics uses describe different fragments
        // from the ones the shader displaces, and the wreckage would be built
        // from one set of pieces and drawn from another.
        int idx = destruct::gridClassify(g, mid);
        float c[3];
        destruct::gridCellCentre(g, idx, c);
        check(destruct::gridClassify(g, c) == idx, "cell centre round-trips");
    }
    {
        const float lo[3] = { -5.5f, -1.5f, -4.0f };
        const float hi[3] = {  5.5f,  1.5f,  4.0f };
        destruct::Grid g = destruct::gridForBounds(lo, hi);
        check(destruct::gridCells(g) > 2000, "a light aircraft subdivides enough");
        check(g.cell >= 0.4f - 1e-4f, "and its cell is clamped, not confetti");
    }
    {
        // The axis cap has to hold or the transform buffer and the constraint
        // graph both blow up on a large aircraft with a small target.
        const float lo[3] = { -60.0f, -12.0f, -60.0f };
        const float hi[3] = {  60.0f,  12.0f,  60.0f };
        destruct::Grid g = destruct::gridForBounds(lo, hi, 500000.0f, 0.2f, 0.2f);
        check(g.nx <= 64 && g.ny <= 64 && g.nz <= 64, "axis cap holds");
        check(g.cell > 0.2f, "the cell grew to honour the cap");
    }

    // ---------------------------------------------------------- occupancy
    printf("\noccupancy and constraint graph\n");
    {
        const float lo[3] = { 0.0f, 0.0f, 0.0f };
        const float hi[3] = { 4.0f, 4.0f, 4.0f };
        destruct::Grid g = destruct::gridForBounds(lo, hi, 64.0f, 1.0f, 1.0f);
        check(g.nx == 4 && g.ny == 4 && g.nz == 4, "4x4x4 test grid");

        destruct::Occupancy o;
        destruct::occupancyInit(o, g);

        const float p0[3] = { 0.5f, 0.5f, 0.5f };
        const float p1[3] = { 1.5f, 0.5f, 0.5f };
        destruct::occupancySet(o, destruct::gridClassify(g, p0));
        destruct::occupancySet(o, destruct::gridClassify(g, p1));
        check(destruct::occupancyCount(o) == 2, "two cells occupied");

        std::vector<destruct::Link> links = destruct::buildLinks(o, g);
        check(links.size() == 1, "a domino yields exactly one link");
        checkNear(links[0].rest, g.cell, 1e-5, "rest length is the cell size");

        // The shader writes occupancy from EVERY vertex, so a cell is set
        // thousands of times per frame. Setting twice must not double count
        // and must not duplicate links.
        destruct::occupancySet(o, destruct::gridClassify(g, p0));
        check(destruct::occupancyCount(o) == 2, "repeat set does not double count");
        check(destruct::buildLinks(o, g).size() == 1, "repeat set does not duplicate");

        // Six-connected only: a diagonal neighbour adds the ONE face-adjacent
        // link it shares, not a diagonal one.
        const float pd[3] = { 1.5f, 1.5f, 0.5f };
        destruct::occupancySet(o, destruct::gridClassify(g, pd));
        check(destruct::buildLinks(o, g).size() == 2,
              "a diagonal adds one face-adjacent link, not two");

        std::vector<int> occ = destruct::occupiedCells(o);
        check(occ.size() == 3, "occupied list matches the count");

        // An isolated cell has no links. It is debris from the first frame,
        // which is correct rather than a special case.
        destruct::Occupancy lone;
        destruct::occupancyInit(lone, g);
        const float far[3] = { 3.5f, 3.5f, 3.5f };
        destruct::occupancySet(lone, destruct::gridClassify(g, far));
        check(destruct::buildLinks(lone, g).empty(), "an isolated cell has no links");
    }

    // ---------------------------------------------------------- integration
    printf("\nfragment integration\n");
    {
        // One second of free fall covers 4.905 m. Semi-implicit Euler at 60 Hz
        // should land within a few centimetres of that.
        destruct::Fragment f;
        f.p[0] = 0; f.p[1] = 100; f.p[2] = 0;
        f.v[0] = f.v[1] = f.v[2] = 0;
        f.restP[0] = f.restP[1] = f.restP[2] = 0;
        f.alive = 1;
        for (int i = 0; i < 60; ++i)
            destruct::integrate(&f, 1, 1.0f / 60.0f, 9.81f, 0.0f);
        checkNear(f.p[1], 100.0 - 4.905, 0.15, "one second of free fall");
    }
    {
        // Damping removes energy and must never reverse or amplify it. A sign
        // flip here would send wreckage upward, which is the sort of thing
        // that looks like a physics bug and is actually an arithmetic one.
        destruct::Fragment d;
        d.p[0] = d.p[1] = d.p[2] = 0;
        d.v[0] = 10.0f; d.v[1] = d.v[2] = 0;
        d.restP[0] = d.restP[1] = d.restP[2] = 0;
        d.alive = 1;
        destruct::integrate(&d, 1, 1.0f / 60.0f, 0.0f, 5.0f);
        check(d.v[0] < 10.0f, "damping reduces speed");
        check(d.v[0] > 0.0f,  "damping does not reverse it");

        // An absurd damping value must clamp, not invert.
        destruct::Fragment e = d;
        e.v[0] = 10.0f;
        destruct::integrate(&e, 1, 1.0f / 60.0f, 0.0f, 1000.0f);
        check(e.v[0] >= 0.0f, "extreme damping clamps rather than inverting");
    }
    {
        // A dead fragment is inert.
        destruct::Fragment f;
        f.p[0] = f.p[1] = f.p[2] = 0;
        f.v[0] = f.v[1] = f.v[2] = 0;
        f.restP[0] = f.restP[1] = f.restP[2] = 0;
        f.alive = 0;
        destruct::integrate(&f, 1, 1.0f / 60.0f, 9.81f, 0.0f);
        checkNear(f.p[1], 0.0, 1e-6, "a dead fragment does not fall");
    }

    // ---------------------------------------------------------- ground
    printf("\nground contact\n");
    {
        destruct::GroundPlane gp;
        gp.height = 10.0f;
        gp.normal[0] = 0; gp.normal[1] = 1; gp.normal[2] = 0;

        destruct::Fragment f;
        f.p[0] = 0; f.p[1] = 9.0f; f.p[2] = 0;
        f.v[0] = 0; f.v[1] = -20.0f; f.v[2] = 0;
        f.restP[0] = f.restP[1] = f.restP[2] = 0;
        f.alive = 1;

        int hit = destruct::clampToGround(&f, 1, gp, 0.5f, 0.0f, 1.0f);
        check(hit == 1, "contact is reported");
        check(f.p[1] >= 10.5f - 1e-3f, "the fragment sits ON the surface");
        check(f.v[1] >= 0.0f, "downward velocity is removed");

        destruct::Fragment s = f;
        s.v[0] = 8.0f; s.v[1] = -1.0f;
        destruct::clampToGround(&s, 1, gp, 0.5f, 0.0f, 1.0f);
        checkNear(s.v[0], 0.0, 1e-3, "full friction halts sliding");

        destruct::Fragment air = f;
        air.p[1] = 50.0f; air.v[1] = -3.0f;
        check(destruct::clampToGround(&air, 1, gp, 0.5f, 0.0f, 1.0f) == 0,
              "an airborne fragment is not in contact");
        checkNear(air.v[1], -3.0, 1e-5, "and its velocity is untouched");
    }
    {
        // Sinking through terrain is the artefact that would make the whole
        // system read as broken, so it must hold under repeated stepping and
        // not merely on the frame of contact.
        destruct::GroundPlane gp;
        gp.height = 0.0f;
        gp.normal[0] = 0; gp.normal[1] = 1; gp.normal[2] = 0;
        destruct::Fragment f;
        f.p[0] = 0; f.p[1] = 5.0f; f.p[2] = 0;
        f.v[0] = f.v[1] = f.v[2] = 0;
        f.restP[0] = f.restP[1] = f.restP[2] = 0;
        f.alive = 1;
        for (int i = 0; i < 600; ++i) {
            destruct::integrate(&f, 1, 1.0f / 60.0f, 9.81f, 0.2f);
            destruct::clampToGround(&f, 1, gp, 0.5f, 0.0f, 0.8f);
        }
        check(f.p[1] >= 0.5f - 1e-3f, "ten seconds of gravity never sinks it");
    }

    // ---------------------------------------------------------- seeding
    printf("\nimpact seeding\n");
    {
        // Fragments near the impact get more energy than distant ones, so the
        // aeroplane comes apart FROM the impact rather than exploding evenly.
        destruct::Fragment f[2];
        for (int i = 0; i < 2; ++i) {
            f[i].p[0] = f[i].p[1] = f[i].p[2] = 0;
            f[i].v[0] = f[i].v[1] = f[i].v[2] = 0;
            f[i].alive = 1;
            f[i].restP[1] = f[i].restP[2] = 0;
        }
        f[0].restP[0] = 1.0f;      // near
        f[1].restP[0] = 20.0f;     // far
        const float impact[3] = { 0.0f, 0.0f, 0.0f };
        const float vel[3]    = { 0.0f, 0.0f, 0.0f };
        destruct::seedImpact(f, 2, impact, vel, 50.0f);
        check(std::fabs(f[0].v[0]) > std::fabs(f[1].v[0]),
              "near fragments get more energy than far ones");
        check(f[0].v[0] > 0.0f, "the impulse radiates away from the impact");
    }
    {
        // The airframe's own velocity is carried by every fragment, or the
        // wreckage would stop dead at the moment of impact.
        destruct::Fragment f;
        f.p[0] = f.p[1] = f.p[2] = 0;
        f.v[0] = f.v[1] = f.v[2] = 0;
        f.restP[0] = 50.0f; f.restP[1] = f.restP[2] = 0;
        f.alive = 1;
        const float impact[3] = { 0.0f, 0.0f, 0.0f };
        const float vel[3]    = { 0.0f, 0.0f, -80.0f };
        destruct::seedImpact(&f, 1, impact, vel, 1.0f);
        check(f.v[2] < -70.0f, "fragments inherit the airframe's velocity");
    }

    // ---------------------------------------------------------- solver
    printf("\nconstraint solver\n");
    {
        destruct::Fragment f[2];
        for (int i = 0; i < 2; ++i) {
            f[i].v[0] = f[i].v[1] = f[i].v[2] = 0;
            f[i].restP[0] = f[i].restP[1] = f[i].restP[2] = 0;
            f[i].alive = 1;
            f[i].p[1] = f[i].p[2] = 0;
        }
        f[0].p[0] = 0.0f;
        f[1].p[0] = 1.3f;                       // rest 1.0 -> 30% strain

        destruct::Link L; L.a = 0; L.b = 1; L.rest = 1.0f;
        unsigned char broken = 0;
        destruct::SolverCfg cfg = destruct::defaultSolverCfg();
        cfg.stiffness = 1.0f;
        cfg.breakStrain = 0.5f;

        int nb = destruct::solveLinks(f, 2, &L, 1, &broken, cfg);
        check(nb == 0, "30% strain under a 50% threshold does not break");
        checkNear(f[1].p[0] - f[0].p[0], 1.0, 0.05, "the link pulls back to rest");

        f[0].p[0] = 0.0f; f[1].p[0] = 2.0f;     // 100% strain
        broken = 0;
        nb = destruct::solveLinks(f, 2, &L, 1, &broken, cfg);
        check(nb == 1, "100% strain over the threshold breaks");
        check(broken == 1, "the break is recorded");

        const float before = f[1].p[0] - f[0].p[0];
        destruct::solveLinks(f, 2, &L, 1, &broken, cfg);
        checkNear(f[1].p[0] - f[0].p[0], before, 1e-4,
                  "a broken link exerts no further force");
    }
    {
        // Load redistribution. A chain whose middle joint is already broken
        // must not transmit force across the gap - that is what makes failure
        // propagate rather than happening everywhere at once.
        destruct::Fragment f[3];
        for (int i = 0; i < 3; ++i) {
            f[i].v[0] = f[i].v[1] = f[i].v[2] = 0;
            f[i].restP[0] = f[i].restP[1] = f[i].restP[2] = 0;
            f[i].alive = 1;
            f[i].p[0] = (float)i; f[i].p[1] = f[i].p[2] = 0;
        }
        f[2].p[0] = 5.0f;
        destruct::Link L[2];
        L[0].a = 0; L[0].b = 1; L[0].rest = 1.0f;
        L[1].a = 1; L[1].b = 2; L[1].rest = 1.0f;
        unsigned char broken[2] = { 0, 1 };
        destruct::SolverCfg cfg = destruct::defaultSolverCfg();

        const float far0 = f[2].p[0];
        destruct::solveLinks(f, 3, L, 2, broken, cfg);
        checkNear(f[2].p[0], far0, 1e-4, "a detached fragment is not dragged back");
        check(destruct::intactLinkCount(broken, 2) == 1, "one joint still holds");
    }
    {
        // A structure under a large impulse must actually come apart, or the
        // whole system is an expensive way to wobble an aeroplane.
        const int N = 8;
        destruct::Fragment f[N];
        for (int i = 0; i < N; ++i) {
            f[i].p[0] = (float)i; f[i].p[1] = f[i].p[2] = 0;
            f[i].restP[0] = (float)i; f[i].restP[1] = f[i].restP[2] = 0;
            f[i].v[0] = f[i].v[1] = f[i].v[2] = 0;
            f[i].alive = 1;
        }
        destruct::Link L[N - 1];
        unsigned char broken[N - 1];
        for (int i = 0; i < N - 1; ++i) {
            L[i].a = i; L[i].b = i + 1; L[i].rest = 1.0f;
            broken[i] = 0;
        }
        const float impact[3] = { 0.0f, 0.0f, 0.0f };
        const float vel[3]    = { 0.0f, 0.0f, 0.0f };
        destruct::seedImpact(f, N, impact, vel, 200.0f);

        destruct::SolverCfg cfg = destruct::defaultSolverCfg();
        int totalBroken = 0;
        for (int step = 0; step < 30; ++step) {
            destruct::integrate(f, N, 1.0f / 60.0f, 9.81f, 0.1f);
            totalBroken += destruct::solveLinks(f, N, L, N - 1, broken, cfg);
        }
        check(totalBroken > 0, "a heavy impact breaks the structure apart");
        check(destruct::intactLinkCount(broken, N - 1) < N - 1,
              "and some joints are gone afterwards");
    }


    // ------------------------------------------------ GPU / CPU agreement
    //
    // The one failure Task 9 calls out by name: if the vertex patch classifies
    // a vertex into a different cell from gridClassify(), the CPU builds its
    // constraint graph over cells the GPU never displaces. Nothing crashes,
    // nothing logs, and the aeroplane just breaks along the wrong seams.
    //
    // So the shader's arithmetic is kept in gpu_layout.h as C and compared
    // against the CPU's here, over the whole box and past every edge of it.
    printf("\ngpu and cpu classify identically\n");
    {
        float bbMin[3] = { -18.0f, -4.0f, -21.0f };
        float bbMax[3] = {  18.0f,  7.0f,  21.0f };
        destruct::Grid g = destruct::gridForBounds(bbMin, bbMax);

        int mismatches = 0, inside = 0, outside = 0;
        // A deterministic sweep rather than random points: a test that samples
        // differently every run reports a different answer every run.
        unsigned seed = 12345u;
        for (int t = 0; t < 20000; ++t) {
            seed = seed * 1664525u + 1013904223u;
            float u = (float)((seed >> 8) & 0xFFFF) / 65535.0f;
            seed = seed * 1664525u + 1013904223u;
            float v = (float)((seed >> 8) & 0xFFFF) / 65535.0f;
            seed = seed * 1664525u + 1013904223u;
            float w = (float)((seed >> 8) & 0xFFFF) / 65535.0f;

            // Deliberately overshoot the box by 20% on every side so the
            // out-of-range rejection is exercised, not just the interior.
            float p[3];
            p[0] = bbMin[0] - 0.2f * (bbMax[0] - bbMin[0]) + u * 1.4f * (bbMax[0] - bbMin[0]);
            p[1] = bbMin[1] - 0.2f * (bbMax[1] - bbMin[1]) + v * 1.4f * (bbMax[1] - bbMin[1]);
            p[2] = bbMin[2] - 0.2f * (bbMax[2] - bbMin[2]) + w * 1.4f * (bbMax[2] - bbMin[2]);

            const int a = destruct::gridClassify(g, p);
            const int b = destruct::gpuCellIndex(g, p);
            if (a != b) ++mismatches;
            if (a >= 0) ++inside; else ++outside;
        }
        check(mismatches == 0, "20000 points classify the same on both paths");
        check(inside  > 1000, "the sweep actually lands inside the box");
        check(outside > 1000, "and the sweep actually lands outside it too");

        // Exact cell boundaries are where a floor-vs-truncate difference would
        // show, so they are hit deliberately rather than left to the sweep.
        int edgeMismatch = 0;
        for (int cx = -1; cx <= g.nx; ++cx) {
            float p[3];
            p[0] = g.min[0] + (float)cx * g.cell;
            p[1] = g.min[1] + 0.5f * g.cell;
            p[2] = g.min[2] + 0.5f * g.cell;
            if (destruct::gridClassify(g, p) != destruct::gpuCellIndex(g, p))
                ++edgeMismatch;
        }
        check(edgeMismatch == 0, "cell boundaries classify the same on both paths");

        // A point below the grid minimum must be rejected, not wrapped. This is
        // the case where truncation toward zero and floor() differ.
        float below[3] = { g.min[0] - 0.5f * g.cell, g.min[1] + 0.1f, g.min[2] + 0.1f };
        check(destruct::gpuCellIndex(g, below) == -1,
              "a point just below the grid is rejected, not wrapped to cell 0");
        check(destruct::gridClassify(g, below) == -1,
              "and the CPU rejects it too");
    }

    // ------------------------------------------------ buffer layout sanity
    printf("\nbuffer layout\n");
    {
        // ---- THE INVARIANTS, NOT THE NUMBER.
        //
        // This asserted kOffOccupancy == 96 and duly failed the moment
        // aircraftFwd and testOffset were added. That is a test of today's
        // arithmetic rather than of anything that must be true, and the honest
        // version states what the layout actually has to satisfy: the fields
        // appear in order, none overlaps the next, and the array is aligned.
        //
        // A field inserted in the middle should pass this. A field that
        // overlaps its neighbour should not, and the old form could not tell
        // the difference.
        check(destruct::kOffAircraftInv == 0,
              "the block starts with the classification matrix");
        check(destruct::kOffAircraftFwd >= destruct::kOffAircraftInv + 64u,
              "the forward matrix clears the inverse one");
        check(destruct::kOffGridMinCell >= destruct::kOffAircraftFwd + 64u,
              "the grid origin clears both matrices");
        check(destruct::kOffGridDim >= destruct::kOffGridMinCell + 16u,
              "the dimensions clear the origin");
        check(destruct::kOffTestOffset >= destruct::kOffGridDim + 16u,
              "the test offset clears the dimensions");
        check(destruct::kOffOccupancy >= destruct::kOffTestOffset + 16u,
              "and occupancy starts after the whole header");
        check((destruct::kOffOccupancy % 16u) == 0,
              "the array is 16-byte aligned, as std430 wants");

        // The member ORDINALS must march with the offsets. SPIR-V addresses
        // members by literal index, so a field added to one and not the other
        // reads the wrong bytes and still validates - the types line up.
        check(destruct::kMemAircraftInv == 0 && destruct::kMemAircraftFwd == 1 &&
              destruct::kMemGridMinCell == 2 && destruct::kMemGridDim == 3 &&
              destruct::kMemTestOffset == 4 && destruct::kMemData == 5,
              "member ordinals are consecutive and in the same order as the offsets");
        check(destruct::kOffDiscard ==
                  destruct::kOffOccupancy + destruct::kMaxCells * 4u,
              "the discard slot sits immediately after the occupancy region");
        check(destruct::kOffXform >= destruct::kOffDiscard + 4u,
              "transforms start after the discard slot, not on top of it");
        check(destruct::kWordOccupancy * 4u == destruct::kOffOccupancy,
              "word and byte offsets agree for occupancy");
        check(destruct::kWordXform * 4u == destruct::kOffXform,
              "word and byte offsets agree for transforms");
        check((destruct::kOffAircraftInv % 16u) == 0 &&
              (destruct::kOffGridMinCell % 16u) == 0 &&
              (destruct::kOffGridDim % 16u) == 0 &&
              (destruct::kOffXform % 16u) == 0,
              "every vector member is 16-byte aligned as std430 requires");

        // A grid big enough to overflow the occupancy region must be clamped
        // rather than allowed to index past it.
        destruct::Grid huge;
        huge.min[0] = huge.min[1] = huge.min[2] = 0.0f;
        huge.cell = 0.4f; huge.nx = 200; huge.ny = 200; huge.nz = 200;
        check(destruct::gpuCellCount(huge) == destruct::kMaxCells,
              "an oversized grid clamps to the buffer rather than overrunning it");
    }


    // -------------------------------------------- the branch-free reject path
    //
    // The shader has no branch: it OpSelects between the real cell and a
    // discard slot and then stores unconditionally. That makes the REJECT path
    // as important to test as the accept path, because a wrong select does not
    // fail loudly - it marks a cell occupied from somewhere in the world, and
    // the aeroplane grows a fragment in mid-air.
    printf("\ndiscovery rejects land in the discard slot\n");
    {
        float bbMin[3] = { -18.0f, -4.0f, -21.0f };
        float bbMax[3] = {  18.0f,  7.0f,  21.0f };
        destruct::Grid g = destruct::gridForBounds(bbMin, bbMax);

        // Dead centre of the box: inside, and must map to its real cell.
        float mid[3] = { 0.0f, 1.5f, 0.0f };
        const int midCell = destruct::gridClassify(g, mid);
        check(midCell >= 0, "the centre of the box is inside the grid");
        check(destruct::gpuStoreIndex(mid, g.min, g.cell, g.nx, g.ny, g.nz, 1)
                  == (uint32_t)midCell,
              "an inside vertex stores to its own cell");

        // Same vertex with discovery off must not touch that cell.
        check(destruct::gpuStoreIndex(mid, g.min, g.cell, g.nx, g.ny, g.nz, 0)
                  == destruct::kDataDiscard,
              "with discovery off even an inside vertex goes to the discard slot");

        // Far outside in every direction.
        float far1[3] = { 1000.0f, 1000.0f, 1000.0f };
        float far2[3] = { -1000.0f, -1000.0f, -1000.0f };
        check(destruct::gpuStoreIndex(far1, g.min, g.cell, g.nx, g.ny, g.nz, 1)
                  == destruct::kDataDiscard,
              "a vertex far above the grid is discarded");
        check(destruct::gpuStoreIndex(far2, g.min, g.cell, g.nx, g.ny, g.nz, 1)
                  == destruct::kDataDiscard,
              "a vertex far below the grid is discarded");

        // The specific trap: a reject must NOT land in cell 0. If it did, the
        // aeroplane's first cell would read occupied from anywhere on earth.
        check(destruct::kDataDiscard != 0,
              "the discard slot is not cell 0");
        bool anyRejectHitsZero = false;
        unsigned seed = 99u;
        for (int t = 0; t < 5000; ++t) {
            seed = seed * 1664525u + 1013904223u;
            float p[3];
            for (int a = 0; a < 3; ++a) {
                seed = seed * 1664525u + 1013904223u;
                const float u = (float)((seed >> 8) & 0xFFFF) / 65535.0f;
                p[a] = -400.0f + u * 800.0f;
            }
            if (destruct::gridClassify(g, p) < 0 &&
                destruct::gpuStoreIndex(p, g.min, g.cell, g.nx, g.ny, g.nz, 1) == 0)
                anyRejectHitsZero = true;
        }
        check(!anyRejectHitsZero,
              "no rejected vertex out of 5000 lands in cell 0");

        // And the discard slot must be inside the buffer but outside both the
        // cells the CPU reads and the transforms.
        check(destruct::kDataDiscard >= destruct::gpuCellCount(g),
              "the discard slot is past every cell this grid uses");
        check(destruct::kDataDiscard < destruct::kDataXform,
              "and before the transform region, so a reject cannot corrupt one");
        check(destruct::kDataXform < destruct::kDataWords,
              "the transform region is inside the buffer");
    }


    // ------------------------------------------------ seed box and refinement
    //
    // X-Plane publishes acf_size_x and acf_size_z and NO height dataref, so the
    // box cannot be read off the aircraft. It is bracketed generously and then
    // measured from the occupancy the GPU reports. These check the bracket
    // really does bracket, and that the measurement narrows correctly.
    printf("\nthe seed box brackets, the refinement measures\n");
    {
        // A 737-ish airframe: 36 m span, 39 m long, gear 2.5 m below datum.
        // acf_size_* are RADII, so these are HALVES of those figures - the
        // bug this pins is exactly the one that read them as full widths.
        destruct::AircraftDims d;
        d.sizeX = 18.0f; d.sizeZ = 19.5f; d.lowestY = -2.5f;
        float bbMin[3], bbMax[3];
        destruct::seedBounds(d, bbMin, bbMax);

        check(bbMin[0] < -18.0f && bbMax[0] > 18.0f,
              "the seed box contains the full span");
        check(bbMin[2] < -19.5f && bbMax[2] > 19.5f,
              "the seed box contains the full length");
        check(bbMin[1] <= -2.5f,
              "the seed box reaches at least to the gear contact point");
        check(bbMax[1] > 8.0f,
              "and high enough to contain a tail");

        // A Pitts: tiny, and proportionally much taller. The height factor is
        // chosen from this extreme rather than from an airliner average.
        destruct::AircraftDims p;
        p.sizeX = 3.05f; p.sizeZ = 2.65f; p.lowestY = -1.0f;
        float pMin[3], pMax[3];
        destruct::seedBounds(p, pMin, pMax);
        check(pMax[1] > 1.8f,
              "a short aerobatic airframe still gets headroom for its fin");

        // ---- THE 747 REGRESSION.
        //
        // The real numbers off the aircraft that exposed the bug. acf_size_x
        // reads 30.0 and acf_size_z 38.5; the .acf planform puts the wingtips
        // at +/-29.63 m, the nose at z 0 and the fin trailing edge at 71.0 m
        // with the datum 5.2 m aft of the nose, and the fin top at 13.31 m.
        //
        // Read as full widths these gave a box of +/-20.25 m and the entire
        // outer wing - 9.4 m of it per side - sat outside the grid. Nothing
        // reported an error: the cells simply were not there to be occupied,
        // which reads identically to a broken transform.
        destruct::AircraftDims b747;
        b747.sizeX = 30.0f; b747.sizeZ = 38.5f; b747.lowestY = -5.2f;
        float jMin[3], jMax[3];
        destruct::seedBounds(b747, jMin, jMax);

        check(jMin[0] <= -29.63f && jMax[0] >= 29.63f,
              "a 747 seed box contains the wingtips the planform reports");
        check(jMax[1] >= 13.31f,
              "and reaches over the top of the fin");
        check(jMin[1] <= -5.2f,
              "and down to the gear contact point");
        check(jMin[2] <= -35.5f && jMax[2] >= 35.5f,
              "and encloses nose to tail");

        // Nothing useful reported at all must not produce a degenerate grid.
        destruct::AircraftDims z;   // all zero
        float zMin[3], zMax[3];
        destruct::seedBounds(z, zMin, zMax);
        bool nonDegenerate = true;
        for (int a = 0; a < 3; ++a)
            if (!(zMax[a] - zMin[a] > 1.0f)) nonDegenerate = false;
        check(nonDegenerate,
              "an aircraft that reports no dimensions still gets a usable box");

        // ---- refinement
        destruct::Grid g = destruct::gridForBounds(bbMin, bbMax);
        std::vector<unsigned char> occ((size_t)g.nx * g.ny * g.nz, 0);

        // Occupy a known interior block and check the measured box encloses it
        // without wandering outside by more than the cell it must round to.
        float realMin[3], realMax[3];
        for (int a = 0; a < 3; ++a) {
            realMin[a] = bbMin[a] + 0.3f * (bbMax[a] - bbMin[a]);
            realMax[a] = bbMin[a] + 0.6f * (bbMax[a] - bbMin[a]);
        }
        // The truth to compare against is the OCCUPIED CELLS, not the region
        // used to choose them: a cell is filled by its centre, so the outermost
        // occupied centre can sit up to half a cell inside realMin/realMax.
        // refineBounds returns cell FACES, and must enclose every centre.
        float cMin[3] = {  1e30f,  1e30f,  1e30f };
        float cMax[3] = { -1e30f, -1e30f, -1e30f };
        int filled = 0;
        for (int zc = 0; zc < g.nz; ++zc)
        for (int yc = 0; yc < g.ny; ++yc)
        for (int xc = 0; xc < g.nx; ++xc) {
            float c[3];
            const int flat = xc + yc * g.nx + zc * g.nx * g.ny;
            destruct::gridCellCentre(g, flat, c);
            if (c[0] >= realMin[0] && c[0] <= realMax[0] &&
                c[1] >= realMin[1] && c[1] <= realMax[1] &&
                c[2] >= realMin[2] && c[2] <= realMax[2]) {
                occ[(size_t)flat] = 1;
                for (int a = 0; a < 3; ++a) {
                    if (c[a] < cMin[a]) cMin[a] = c[a];
                    if (c[a] > cMax[a]) cMax[a] = c[a];
                }
                ++filled;
            }
        }
        check(filled > 0, "the synthetic airframe occupies some cells");

        float rMin[3], rMax[3];
        check(destruct::refineBounds(g, occ.data(), rMin, rMax),
              "refinement succeeds when something is occupied");

        bool encloses = true, tight = true;
        for (int a = 0; a < 3; ++a) {
            if (rMin[a] > cMin[a] || rMax[a] < cMax[a]) encloses = false;
            // Never wider than the truth plus the cell it had to round to on
            // each side, or the refinement is not actually measuring.
            if (rMin[a] < cMin[a] - 2.0f * g.cell ||
                rMax[a] > cMax[a] + 2.0f * g.cell) tight = false;
        }
        check(encloses, "the refined box encloses every occupied cell centre");
        check(tight, "and is tight to within the cell size it rounded to");

        // The refined box must be strictly smaller than the seed, or the seed
        // was not generous and something was clipped.
        bool narrowed = true;
        for (int a = 0; a < 3; ++a)
            if (!(rMax[a] - rMin[a] < bbMax[a] - bbMin[a])) narrowed = false;
        check(narrowed, "pass two narrows the seed rather than matching it");

        // Nothing occupied is a REAL outcome - it means the transform put the
        // aeroplane somewhere other than the grid - and must be reported as
        // failure rather than yielding a box at the origin.
        std::vector<unsigned char> empty((size_t)g.nx * g.ny * g.nz, 0);
        float eMin[3], eMax[3];
        check(!destruct::refineBounds(g, empty.data(), eMin, eMax),
              "an empty occupancy is reported as failure, not as a box at zero");

        // The gate the plan states in these exact terms.
        const float frac = destruct::occupiedFraction(g, occ.data());
        check(frac > 0.0f && frac < 1.0f, "occupied fraction is a real fraction");
        const float fracEmpty = destruct::occupiedFraction(g, empty.data());
        check(fracEmpty == 0.0f, "an empty grid reports zero occupancy");
    }

    // ------------------------------------------- the airframe from the .acf
    //
    // The wings are not in the OBJ files - all 70 exterior objects of a
    // 747-200F fit in a box 22.5 m wide against a real 59.6 - so the geometry
    // comes from the .acf planform instead. These pin the rules that were read
    // off a real aircraft's numbers.
    //
    // Synthetic file rather than a real one: this suite must run on a machine
    // with no X-Plane on it. tools/acf_check.exe does the other half of the
    // job, checking real aeroplanes against their published dimensions.
    printf("\nthe airframe comes out of the .acf planform\n");
    {
        // Segment 4 of the 747-200F, verbatim in feet, plus segment 6 so the
        // chaining rule has something to predict. If semilen were measured
        // across the span rather than along the swept and dihedralled
        // quarter-chord, segment 6's root would land at 52.75 rather than the
        // 67.97 the file states - so this distinguishes the two readings.
        //
        // _geo_xyz data deliberately precedes i_count, because the .acf is
        // sorted ALPHABETICALLY and that is the order a real file has. Reading
        // the counts first and bounds-checking against them silently discarded
        // every body point and produced an aeroplane with no fuselage.
        const char *acf =
            "I\n1200 Version\nACF\n\nPROPERTIES_BEGIN\n"
            "P _wing/4/_part_x -42.659202576\n"
            "P _wing/4/_part_y -2.372622967\n"
            "P _wing/4/_part_z 103.690002441\n"
            "P _wing/4/_Croot 28.799999237\n"
            "P _wing/4/_Ctip 20.000000000\n"
            "P _wing/4/_semilen_SEG 33.099998474\n"
            "P _wing/4/_sweep_design 39.700000763\n"
            "P _wing/4/_dihed_design 6.300000191\n"
            "P _wing/4/_is_right_mult -1.000000000\n"
            "P _wing/5/_part_x 42.659202576\n"
            "P _wing/5/_part_y -2.372622967\n"
            "P _wing/5/_part_z 103.690002441\n"
            "P _wing/5/_Croot 28.799999237\n"
            "P _wing/5/_Ctip 20.000000000\n"
            "P _wing/5/_semilen_SEG 33.099998474\n"
            "P _wing/5/_sweep_design 39.700000763\n"
            "P _wing/5/_dihed_design 6.300000191\n"
            "P _wing/5/_is_right_mult 1.000000000\n"
            // The fin: a segment whose dihedral is 90 degrees.
            "P _wing/11/_part_x 0.000000000\n"
            "P _wing/11/_part_y 11.199999809\n"
            "P _wing/11/_part_z 187.800003052\n"
            "P _wing/11/_Croot 38.029998779\n"
            "P _wing/11/_Ctip 13.079999924\n"
            "P _wing/11/_semilen_SEG 45.689998627\n"
            "P _wing/11/_sweep_design 44.700000763\n"
            "P _wing/11/_dihed_design 90.000000000\n"
            "P _wing/11/_is_right_mult 1.000000000\n"
            // An unused slot: Plane Maker leaves plenty of these.
            "P _wing/30/_part_x 0.000000000\n"
            "P _wing/30/_Croot 0.000000000\n"
            "P _wing/30/_semilen_SEG 0.000000000\n"
            // Two body points, data before counts, exactly as a real file has.
            "P _body/0/_geo_xyz/0,0,0 0.000000000\n"
            "P _body/0/_geo_xyz/0,0,1 -0.540000021\n"
            "P _body/0/_geo_xyz/0,0,2 0.100000001\n"
            "P _body/0/_geo_xyz/9,2,0 6.573009014\n"
            "P _body/0/_geo_xyz/9,2,1 10.235960960\n"
            "P _body/0/_geo_xyz/9,2,2 48.704662323\n"
            "P _body/0/_geo_xyz/i_count 20\n"
            "P _body/0/_geo_xyz/j_count 18\n"
            "P _body/0/_part_x 0.000000000\n"
            "P _body/0/_part_y 0.000000000\n"
            "P _body/0/_part_z 0.000000000\n"
            // Gear: the shorter leg must not win the ground contact.
            "P _gear/0/_gear_x 0.000000000\n"
            "P _gear/0/_gear_y -8.900000000\n"
            "P _gear/0/_gear_z 25.000000000\n"
            "P _gear/0/_leg_len 8.993000000\n"
            "P _gear/0/_tire_radius 2.059999943\n"
            "P _gear/1/_gear_x 6.290000000\n"
            "P _gear/1/_gear_y -11.100000000\n"
            "P _gear/1/_gear_z 114.080000000\n"
            "P _gear/1/_leg_len 6.896000000\n"
            "P _gear/1/_tire_radius 2.049999952\n"
            // The eye, and the sibling key that broke reading it: atoi on
            // "count" is 0, so the COUNT was stored as the x coordinate and
            // the eye read 3 ft off the centreline of an aeroplane whose
            // own file says 0.
            "P acf/_pe_xyz/0 0.000000000\n"
            "P acf/_pe_xyz/1 11.090000153\n"
            "P acf/_pe_xyz/2 21.000000000\n"
            "P acf/_pe_xyz/count 3\n";

        const char *tmp = "build/test_synth.acf";
        FILE *fh = fopen(tmp, "wb");
        check(fh != NULL, "the synthetic .acf can be written");
        if (fh) { fputs(acf, fh); fclose(fh); }

        destruct::Airframe a;
        check(destruct::parseAcf(tmp, a), "the .acf parses");
        check(a.wings.size() == 3,
              "three real surfaces, and the empty slot is not one of them");
        check(a.bodyXyz.size() == 6,
              "body points survive appearing before their own counts");
        check(a.gear.size() == 2, "both gear legs are read");

        // Find segment 5, the right-hand outboard-of-inboard panel.
        const destruct::WingSeg *w5 = NULL, *fin = NULL;
        for (size_t i = 0; i < a.wings.size(); ++i) {
            if (a.wings[i].index == 5)  w5  = &a.wings[i];
            if (a.wings[i].index == 11) fin = &a.wings[i];
        }
        check(w5 != NULL && fin != NULL, "both named segments are present");

        if (w5) {
            checkNear(w5->rootX, 13.002, 0.01, "feet become metres on the way in");
            checkNear(w5->croot, 8.778, 0.01, "and so does the root chord");
            checkNear(w5->side, 1.0, 1e-6, "_is_right_mult gives the side");

            // THE CHAINING RULE. 67.972526 ft is what the real file states for
            // segment 6's root, and this predicts it from segment 4 alone.
            float d[3];
            destruct::spanVector(*w5, d);
            checkNear(w5->rootX + d[0], 67.972526 * 0.3048, 0.01,
                      "semilen runs along the swept, dihedralled span");
            check(d[1] > 0.0f, "dihedral lifts the tip above the root");
            check(d[2] > 0.0f, "sweep carries the tip aft");
        }

        if (fin) {
            // A fin is not a special case: dihedral 90 drives the span into
            // +y on its own, and the thickness must then extrude across the
            // fin in x rather than along it in y.
            float d[3];
            destruct::spanVector(*fin, d);
            checkNear(d[0], 0.0, 0.01, "a fin has no span in x");
            checkNear(d[1], 45.689999 * cosf(44.700001f * 3.14159265f / 180.0f)
                            * 0.3048f, 0.02,
                      "a fin's span goes straight up");

            std::vector<float> fv;
            destruct::wingVertices(*fin, 4, 2, fv);
            float flo[3], fhi[3];
            check(destruct::vertexBounds(fv, flo, fhi),
                  "the fin generates vertices");
            check(fhi[0] - flo[0] > 0.05f,
                  "a fin has thickness across it, not zero width");
            checkNear(fhi[1], (11.199999 + 45.689999
                      * cosf(44.700001f * 3.14159265f / 180.0f)) * 0.3048f,
                      0.05, "and its tip is where the planform says");
        }

        // Both sides come out of a two-sided wing pair.
        std::vector<float> v;
        destruct::airframeVertices(a, 4, 2, v);
        float lo[3], hi[3];
        check(destruct::vertexBounds(v, lo, hi), "the airframe has vertices");
        check(lo[0] < -13.0f && hi[0] > 13.0f,
              "the left and right panels both appear");

        float contact = 0.0f;
        checkNear(a.pilotEye[0], 0.0, 1e-4,
                  "the eye is on the centreline - a count is not coordinate 0");
        checkNear(a.pilotEye[2], 21.0 * 0.3048, 1e-3,
                  "and its station comes through in metres");

        check(destruct::groundContact(a, contact), "the gear gives a floor");
        // Leg 1 reaches -20.046 ft against leg 0's -19.953: the LOWER wins.
        checkNear(contact, -20.045999 * 0.3048, 0.01,
                  "the lowest leg sets the ground, not the first one");

        // ---- THE FRAME, AND HOW IT WAS TOLD FROM THE WRONG ONE.
        //
        // The .acf is about Plane Maker's origin; X-Plane draws about the
        // DEFAULT centre of gravity. On this aeroplane those are 31.83 m apart
        // - nearly half its length - so choosing wrongly does not look like a
        // small error. It looks like the aircraft is not there.
        //
        // The gear was tried first and returned the same constant on every
        // leg, which was persuasive and still wrong to USE: acf_Xarm/Yarm/Zarm
        // are moment arms from the CURRENT centre of gravity, and that moves as
        // fuel burns. It agrees today only because the aeroplane is loaded at
        // its default, so the agreement would have decayed over a long sector
        // rather than failing where anyone would look for it.
        //
        // The pilot's eye is the honest witness: it is published in BOTH
        // frames, so distinguishing them needs no theory about which X-Plane
        // uses.
        const float cgYFeet = -8.01f, cgZFeet = 104.44f;   // the real 747-200F
        float off[3];
        destruct::referencePointOffset(cgYFeet, cgZFeet, off);
        checkNear(off[0], 0.0, 1e-6,
                  "no shift across the aeroplane - PM's origin is on the centreline");
        checkNear(off[1], 2.441, 0.01, "2.44 m vertically");
        checkNear(off[2], -31.833, 0.01, "and 31.83 m along the fuselage");

        // THE MEASUREMENT ITSELF. The .acf puts the eye at z = 21.00 ft and
        // y = 11.09 ft; the sim reports acf_peZ = -25.43 m and acf_peY = 5.82.
        // Checked against the sim's own published numbers rather than against
        // this file's arithmetic, because the two agreeing is the entire claim.
        const float eyeAcf[3] = { 0.0f, 11.09f * 0.3048f, 21.00f * 0.3048f };
        checkNear(eyeAcf[1] + off[1], 5.82, 0.01,
                  "the eye lands where the sim says it is, vertically");
        checkNear(eyeAcf[2] + off[2], -25.43, 0.01,
                  "and along the fuselage - which the PM frame misses by 32 m");

        std::vector<float> shifted = v;
        destruct::applyOffset(shifted, off);
        float slo[3], shi[3];
        destruct::vertexBounds(shifted, slo, shi);
        checkNear(shi[2] - hi[2], -31.833, 0.01,
                  "applying the offset translates the airframe");
        checkNear((shi[2] - slo[2]) - (hi[2] - lo[2]), 0.0, 1e-3,
                  "and does not change its size");
    }

    printf("\n%s: %d failure(s)\n", g_fail ? "FAILED" : "OK", g_fail);
    return g_fail ? 1 : 0;
}
