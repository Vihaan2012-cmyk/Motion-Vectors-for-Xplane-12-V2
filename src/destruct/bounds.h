// Where the aeroplane is, in its own frame - seeded from datarefs, then
// measured by the GPU.
//
// THE PROBLEM
//
// Fragmenting an airframe needs its bounding box, and X-Plane does not publish
// one. Searching the shipping binary's string table turns up exactly two
// dimensions: sim/aircraft/view/acf_size_x and acf_size_z. Width and length.
// There is no height dataref at all.
//
// The gear group gives a floor - acf_gear_ynodef minus the leg length and the
// tyre radius is the lowest point the aeroplane touches - but nothing gives the
// top. A 737's fin is about a fifth of its length; a Pitts' is nearer a third.
// Any constant picked here is wrong for some aircraft, and wrong quietly: the
// fin simply does not fragment, which looks like a physics bug rather than a
// bad box.
//
// THE ANSWER: DON'T GUESS THE BOX, MEASURE IT
//
// The occupancy pass already reports which cells the airframe's own vertices
// land in. So the box does not have to be known in advance - it only has to be
// bracketed. Pass one uses a deliberately oversized seed box on a coarse grid
// and asks the GPU which cells were touched; the true extent falls out of the
// answer. Pass two builds the real grid from that.
//
// This means the seed only has to be GENEROUS, never accurate, which is a
// question acf_size_x and acf_size_z can answer. It also means the result is
// correct for an aircraft nobody has tested, which a table of constants never
// would be.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <stdint.h>
#include "grid.h"
#include "gpu_layout.h"

namespace destruct {

// What X-Plane will actually tell us, and nothing it will not.
struct AircraftDims {
    float sizeX = 0.0f;      // acf_size_x, full width
    float sizeZ = 0.0f;      // acf_size_z, full length
    // Lowest point below the aircraft datum, as a NEGATIVE number: gear
    // ynodef minus leg length minus tyre radius. Zero is a legitimate value
    // for a floatplane or a gear-up airframe and is handled by the pad.
    float lowestY = 0.0f;
};

// How much larger than the reported dimensions the seed box is made.
//
// Not tuning for its own sake. acf_size_x and acf_size_z describe the
// AIRFRAME, and the mesh includes things the airframe numbers do not: wingtip
// fences, refuelling probes, rotors, external stores, and on some payware a
// pushback tug parked under the nose. A seed that clips those loses them from
// the box permanently, because pass two only ever narrows.
//
// Overshooting costs a coarse cell or two of resolution in pass one and
// nothing at all afterwards, so it is deliberately lopsided in favour of too
// big.
static const float kSeedPad = 1.35f;

// Height as a multiple of the LONGER horizontal dimension. Only a bracket:
// pass one measures the real value. Chosen from the extremes rather than the
// average - a Pitts is about a third, so a half clears everything with room.
static const float kSeedHeightFactor = 0.5f;

inline void seedBounds(const AircraftDims &d, float bbMin[3], float bbMax[3])
{
    // A degenerate aircraft record must not produce a degenerate grid. Anything
    // that fails to describe itself gets a box big enough for a light single,
    // which discovers its real size in pass one like everything else.
    float sx = d.sizeX > 0.1f ? d.sizeX : 10.0f;
    float sz = d.sizeZ > 0.1f ? d.sizeZ : 10.0f;

    const float halfX = 0.5f * sx * kSeedPad;
    const float halfZ = 0.5f * sz * kSeedPad;
    const float longer = (sx > sz ? sx : sz);

    // Down to whichever is lower: the gear contact point, or a fraction of the
    // hull. lowestY arrives negative; a positive or zero value means the gear
    // datarefs said nothing useful, and the hull term covers it.
    float down = d.lowestY < 0.0f ? d.lowestY : 0.0f;
    const float hullDown = -0.25f * longer * kSeedPad;
    if (hullDown < down) down = hullDown;

    const float up = kSeedHeightFactor * longer * kSeedPad;

    bbMin[0] = -halfX; bbMax[0] = halfX;
    bbMin[1] = down;   bbMax[1] = up;
    bbMin[2] = -halfZ; bbMax[2] = halfZ;
}

// ---- PASS TWO: the box the airframe actually occupies.
//
// Walks the coarse occupancy and returns the tight extent. Returns false when
// nothing was occupied, which is a real and important outcome: it means the
// transform put the aeroplane somewhere other than where the grid was, and
// building a grid from an empty result would silently produce a box at the
// origin with no aircraft in it.
inline bool refineBounds(const Grid &g, const unsigned char *occ,
                         float outMin[3], float outMax[3])
{
    if (!occ) return false;
    const int64_t total = (int64_t)g.nx * g.ny * g.nz;
    if (total <= 0) return false;

    int lo[3] = { g.nx, g.ny, g.nz };
    int hi[3] = { -1, -1, -1 };
    bool any = false;

    for (int z = 0; z < g.nz; ++z) {
        for (int y = 0; y < g.ny; ++y) {
            for (int x = 0; x < g.nx; ++x) {
                const int64_t idx = (int64_t)x + (int64_t)y * g.nx +
                                    (int64_t)z * g.nx * g.ny;
                if (idx >= (int64_t)kMaxCells) continue;
                if (!occ[idx]) continue;
                any = true;
                const int c[3] = { x, y, z };
                for (int a = 0; a < 3; ++a) {
                    if (c[a] < lo[a]) lo[a] = c[a];
                    if (c[a] > hi[a]) hi[a] = c[a];
                }
            }
        }
    }
    if (!any) return false;

    // A cell is occupied if ANY vertex fell in it, so the true surface lies
    // somewhere inside the cell rather than on its edge. Taking the outer face
    // of the outermost occupied cell keeps the whole airframe enclosed; taking
    // the centre would shave half a cell off every side.
    for (int a = 0; a < 3; ++a) {
        outMin[a] = g.min[a] + (float)lo[a] * g.cell;
        outMax[a] = g.min[a] + (float)(hi[a] + 1) * g.cell;
    }
    return true;
}

// What fraction of the coarse grid came back occupied.
//
// The plan's Task 9 gate is expressed in exactly these terms: 5-15% for an
// airliner, below 2% means classification is missing the aircraft, above 40%
// means the box or the transform is wrong and it is catching the world.
inline float occupiedFraction(const Grid &g, const unsigned char *occ)
{
    if (!occ) return 0.0f;
    const int64_t total = (int64_t)g.nx * g.ny * g.nz;
    if (total <= 0) return 0.0f;
    const int64_t n = total > (int64_t)kMaxCells ? (int64_t)kMaxCells : total;
    int64_t hit = 0;
    for (int64_t i = 0; i < n; ++i) if (occ[i]) ++hit;
    return (float)hit / (float)n;
}

} // namespace destruct
