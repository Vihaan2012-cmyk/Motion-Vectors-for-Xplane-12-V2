#pragma once
#include "body.h"
#include "occupancy.h"
#include <cmath>

namespace destruct {

struct SolverCfg {
    int   iterations;     // Gauss-Seidel passes per step
    float stiffness;      // 0..1, fraction of the error corrected per pass
    float breakStrain;    // |len - rest| / rest above which the joint fails
};

inline SolverCfg defaultSolverCfg()
{
    SolverCfg c;
    c.iterations  = 10;
    c.stiffness   = 0.8f;
    // 35% stretch before a joint lets go. Low enough that an impact tears the
    // airframe rather than deforming it like rubber, high enough that the
    // structure holds together under its own weight while settling.
    c.breakStrain = 0.35f;
    return c;
}

// Position-based dynamics: correct POSITIONS directly and let velocity follow.
//
// WHY NOT SPRINGS
//
// A force-based spring stiff enough to hold an airframe together needs a
// timestep far below the frame rate to stay stable, and this runs inside a
// flight loop callback at whatever rate the sim happens to manage. PBD is
// unconditionally stable at any stiffness. The price is that stiffness becomes
// rate-dependent, which does not matter for wreckage.
//
// WHY BREAKAGE IS PERMANENT
//
// A broken joint never heals, so load redistributes onto whatever is still
// attached. That is what makes failure PROPAGATE - the nose crushes, stress
// runs aft, the tail departs a moment later - instead of the airframe
// disintegrating uniformly, which reads as an explosion rather than a crash.
//
// It is also what makes several hundred fragments look like destruction rather
// than confetti: without joints they are independent debris from frame one.
inline int solveLinks(Fragment *f, int n, Link *links, int linkCount,
                      unsigned char *broken, const SolverCfg &cfg)
{
    int brokeNow = 0;
    for (int it = 0; it < cfg.iterations; ++it) {
        for (int i = 0; i < linkCount; ++i) {
            if (broken[i]) continue;
            Link &L = links[i];
            if (L.a < 0 || L.b < 0 || L.a >= n || L.b >= n) continue;
            Fragment &A = f[L.a];
            Fragment &B = f[L.b];
            if (!A.alive || !B.alive) continue;

            float d[3];
            for (int a = 0; a < 3; ++a) d[a] = B.p[a] - A.p[a];
            const float len = std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
            if (len < 1e-6f) continue;

            const float rest = L.rest > 1e-6f ? L.rest : 1e-6f;
            const float strain = std::fabs(len - rest) / rest;
            if (strain > cfg.breakStrain) {
                broken[i] = 1;
                ++brokeNow;
                continue;
            }

            // Half the correction to each end: the fragments have equal mass,
            // which at uniform cell size they effectively do.
            const float corr = (len - rest) / len * 0.5f * cfg.stiffness;
            for (int a = 0; a < 3; ++a) {
                A.p[a] += d[a] * corr;
                B.p[a] -= d[a] * corr;
            }
        }
    }
    return brokeNow;
}

// How many joints a fragment still has.
//
// Zero means it is loose debris. Useful for deciding what may be culled once
// it has settled, and for reporting whether the structure actually came apart
// rather than merely wobbling.
inline int intactLinkCount(const unsigned char *broken, int linkCount)
{
    int n = 0;
    for (int i = 0; i < linkCount; ++i) if (!broken[i]) ++n;
    return n;
}

}  // namespace destruct
