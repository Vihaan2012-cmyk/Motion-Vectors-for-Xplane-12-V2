// Offline tests for the crash destruction maths.
//
// None of this needs X-Plane. Detection, grid classification, integration and
// the constraint solver are arithmetic, and arithmetic that can be tested at a
// command line should never be tested by flying an aeroplane into the ground -
// which is slow, unrepeatable, and cannot be run before a commit.

#include "destruct/trigger.h"
#include "destruct/occupancy.h"
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

    printf("\n%s: %d failure(s)\n", g_fail ? "FAILED" : "OK", g_fail);
    return g_fail ? 1 : 0;
}
