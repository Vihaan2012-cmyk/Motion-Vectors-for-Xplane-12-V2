#pragma once
#include <cmath>

namespace destruct {

// A uniform grid over the airframe's bounding box, in airframe-local metres.
//
// WHY A GRID AND NOT A FRAGMENTED MESH
//
// The obvious approach is to cut the aircraft's geometry into chunks. The
// Vulkan layer cannot: it sees draw calls and vertex buffers, not objects, and
// an aircraft arrives as many draws with X-Plane's own animation transforms
// already applied to gear, flaps and control surfaces. Reconstructing "this is
// a wing" from that is the whole problem, and it would have to be solved again
// for every add-on aircraft.
//
// A grid needs none of that. Classification is arithmetic on a position, so it
// works on any geometry from any aircraft with no authoring. Empty cells cost
// nothing because no vertex ever lands in one, which is why the grid may cover
// the entire bounding box even though an aircraft occupies a few percent of it.
//
// The shader performs the IDENTICAL computation in gridClassify. Keeping the
// two in step is what makes the CPU-side occupancy and constraint graph
// describe the same cells the GPU displaces; if they diverge, the physics acts
// on one set of fragments and the picture shows another.
struct Grid {
    float min[3];
    float cell;
    int   nx, ny, nz;
};

// Cell size is derived from the aircraft's VOLUME against a target cell count,
// then clamped to a sane fragment size.
//
// WHY NOT A FIXED FRAGMENT SIZE
//
// A fixed 1.2 m was the original design and it is wrong across aircraft, which
// a test caught before any of this ran. The arithmetic:
//
//     Cessna  8 x 11 x 3 m   at 1.2 m ->    210 cells -> about  21 fragments
//     A320   38 x 36 x 12 m  at 1.2 m ->  9,600 cells -> about 960 fragments
//
// 21 pieces is not a destroyed aeroplane, it is an aeroplane in 21 pieces.
// Fragment size is the right parameter WITHIN an aircraft - it is what makes
// the debris look like debris - but holding it fixed ACROSS aircraft makes the
// piece count follow the cube of the aircraft's size, which spans two orders of
// magnitude between a trainer and a heavy.
//
// So the target is a cell COUNT, and the resulting cell size is clamped to
// 0.4 - 2.0 m so a small aircraft cannot end up with confetti and a large one
// cannot end up with slabs. Measured against the clamp:
//
//     Cessna   0.40 m (clamped)  ->  3,780 cells -> about   380 fragments
//     A320     0.94 m            -> 20,787 cells -> about 2,000 fragments
//     747      1.63 m            -> 20,640 cells -> about 1,000 fragments
//
// which lands every one of them in the band where destruction reads as
// structural rather than as slabs or as sand.
//
// The 64-per-axis cap is a hard limit on top of that. Both the transform buffer
// and the constraint graph scale with cell count, and a long thin aircraft
// could otherwise exceed the index range on one axis while the volume target
// looked satisfied.
inline Grid gridForBounds(const float bbMin[3], const float bbMax[3],
                          float targetCells = 20000.0f,
                          float minCell = 0.4f, float maxCell = 2.0f)
{
    Grid g;
    for (int i = 0; i < 3; ++i) g.min[i] = bbMin[i];

    float span[3];
    for (int i = 0; i < 3; ++i) {
        span[i] = bbMax[i] - bbMin[i];
        if (span[i] < 0.0f) span[i] = 0.0f;
    }
    const double volume = (double)span[0] * (double)span[1] * (double)span[2];

    float cell = (volume > 1e-6 && targetCells > 1.0f)
               ? (float)std::pow(volume / (double)targetCells, 1.0 / 3.0)
               : minCell;
    if (cell < minCell) cell = minCell;
    if (cell > maxCell) cell = maxCell;

    // The axis cap can still bite a very long aircraft, so grow until it fits.
    for (int guard = 0; guard < 64; ++guard) {
        bool ok = true;
        for (int i = 0; i < 3; ++i)
            if ((int)std::ceil(span[i] / cell) > 64) ok = false;
        if (ok) break;
        cell *= 1.25f;
    }
    g.cell = cell;

    int n[3];
    for (int i = 0; i < 3; ++i) {
        n[i] = (int)std::ceil(span[i] / cell);
        if (n[i] < 1)  n[i] = 1;
        if (n[i] > 64) n[i] = 64;
    }
    g.nx = n[0]; g.ny = n[1]; g.nz = n[2];
    return g;
}

inline int gridCells(const Grid &g) { return g.nx * g.ny * g.nz; }

// Cell index, or -1 when the point lies outside the grid.
inline int gridClassify(const Grid &g, const float local[3])
{
    const int n[3] = { g.nx, g.ny, g.nz };
    int c[3];
    for (int i = 0; i < 3; ++i) {
        float f = (local[i] - g.min[i]) / g.cell;
        if (f < 0.0f) return -1;
        c[i] = (int)f;
        if (c[i] >= n[i]) return -1;
    }
    return c[0] + c[1] * g.nx + c[2] * g.nx * g.ny;
}

// The centre of a cell, in airframe-local metres. This is a fragment's REST
// position: where it sat on the intact airframe, and the pivot its rigid
// transform is expressed about.
inline void gridCellCentre(const Grid &g, int index, float out[3])
{
    if (index < 0 || index >= gridCells(g)) {
        out[0] = out[1] = out[2] = 0.0f;
        return;
    }
    int x =  index % g.nx;
    int y = (index / g.nx) % g.ny;
    int z =  index / (g.nx * g.ny);
    out[0] = g.min[0] + ((float)x + 0.5f) * g.cell;
    out[1] = g.min[1] + ((float)y + 0.5f) * g.cell;
    out[2] = g.min[2] + ((float)z + 0.5f) * g.cell;
}

}  // namespace destruct
