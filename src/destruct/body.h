#pragma once
#include <cmath>

namespace destruct {

// One fragment.
//
// Position and velocity only - NO ORIENTATION, deliberately.
//
// At sub-metre fragments the eye reads position and separation, not spin, and
// dropping the rotational state halves the transform buffer, removes
// quaternion normalisation from the hot loop, and makes the solver a purely
// positional problem. Position-based dynamics wants positions anyway; adding
// orientation would mean either coupling it into the constraint solve (much
// harder) or integrating it separately and having it disagree with the
// constraints (visibly wrong at the joints).
//
// If tumbling proves necessary it can be derived per fragment from its
// velocity at draw time without ever entering the simulation.
struct Fragment {
    float p[3];        // current position, airframe-local metres
    float v[3];        // velocity
    float restP[3];    // where this fragment sat on the intact airframe
    unsigned char alive;
};

// Semi-implicit Euler: velocity first, then position.
//
// Stable at these step sizes and stiffnesses where explicit Euler is not, and
// an order of magnitude simpler than anything better. Wreckage does not need
// energy conservation; it needs to fall down and stop.
inline void integrate(Fragment *f, int n, float dt, float gravity, float damping)
{
    float k = 1.0f - damping * dt;
    if (k < 0.0f) k = 0.0f;      // damping must never invert the velocity
    for (int i = 0; i < n; ++i) {
        if (!f[i].alive) continue;
        f[i].v[1] -= gravity * dt;
        for (int a = 0; a < 3; ++a) {
            f[i].v[a] *= k;
            f[i].p[a] += f[i].v[a] * dt;
        }
    }
}

// One plane is enough for a debris field. XPLMProbeTerrainXYZ is sampled once
// at the impact point and the wreckage lands within a few tens of metres, over
// which real terrain is flat enough that nobody will measure the difference.
//
// The alternative - probing per fragment per frame - would be 2000 terrain
// probes a frame for an artefact nobody can see.
struct GroundPlane {
    float height;
    float normal[3];
};

// Returns how many fragments are in contact.
//
// restitution 0 because wreckage does not bounce, friction 1 because it does
// not slide. Both are knobs rather than constants because the right value is a
// look, and a look is tuned by eye rather than derived.
//
// Sinking through terrain is the one artefact that would make the whole system
// read as broken rather than as stylised, so the clamp is unconditional: a
// fragment below the plane is placed ON it, every frame, whatever else is
// happening to it.
inline int clampToGround(Fragment *f, int n, const GroundPlane &gp,
                         float radius, float restitution, float friction)
{
    int hits = 0;
    const float floorY = gp.height + radius;
    // ---- RESTING ON THE GROUND IS CONTACT.
    //
    // This tested p[1] >= floorY and skipped, which means a fragment sitting
    // EXACTLY on the surface counted as airborne and never received friction -
    // so settled wreckage slid across the ground forever, frictionless, having
    // been placed there by this very function on the previous frame. The clamp
    // put it at floorY and then refused to recognise it.
    //
    // An epsilon rather than >= : contact is a band, not a plane, and floating
    // point will not land a fragment on an exact value twice running.
    const float contactEps = 1e-3f;
    for (int i = 0; i < n; ++i) {
        if (!f[i].alive) continue;
        if (f[i].p[1] > floorY + contactEps) continue;
        f[i].p[1] = floorY;
        if (f[i].v[1] < 0.0f) f[i].v[1] = -f[i].v[1] * restitution;
        float keep = 1.0f - friction;
        if (keep < 0.0f) keep = 0.0f;
        f[i].v[0] *= keep;
        f[i].v[2] *= keep;
        ++hits;
    }
    return hits;
}

// Seed a fragment field from an impact.
//
// Every fragment starts at its rest position carrying the airframe's velocity,
// plus an impulse that RADIATES from the impact point and falls off with
// distance. That is what makes the aeroplane come apart from where it struck
// rather than exploding uniformly, which reads as a bomb rather than a crash.
//
// The falloff is 1/(1+d) rather than 1/d^2: inverse square puts almost all the
// energy in the two or three fragments nearest the impact and leaves the rest
// of the airframe untouched, which looks like a small explosion inside an
// intact aeroplane.
inline void seedImpact(Fragment *f, int n, const float impactLocal[3],
                       const float airframeVel[3], float energy)
{
    for (int i = 0; i < n; ++i) {
        if (!f[i].alive) continue;
        float d[3];
        float len2 = 0.0f;
        for (int a = 0; a < 3; ++a) {
            d[a] = f[i].restP[a] - impactLocal[a];
            len2 += d[a] * d[a];
        }
        const float len = std::sqrt(len2);
        const float falloff = 1.0f / (1.0f + len);
        for (int a = 0; a < 3; ++a) {
            const float dir = (len > 1e-4f) ? d[a] / len : 0.0f;
            f[i].v[a] = airframeVel[a] + dir * energy * falloff;
        }
    }
}

}  // namespace destruct
