// Offline tests for the crash destruction maths.
//
// None of this needs X-Plane. Detection, grid classification, integration and
// the constraint solver are arithmetic, and arithmetic that can be tested at a
// command line should never be tested by flying an aeroplane into the ground -
// which is slow, unrepeatable, and cannot be run before a commit.

#include "destruct/trigger.h"
#include "destruct/occupancy.h"
#include "destruct/solver.h"
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

    printf("\n%s: %d failure(s)\n", g_fail ? "FAILED" : "OK", g_fail);
    return g_fail ? 1 : 0;
}
