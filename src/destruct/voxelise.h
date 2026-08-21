// Stamp the airframe into the occupancy grid, from its own geometry.
//
// WHY NOT THE GPU PASS THAT ALREADY EXISTS
//
// The occupancy discovery in the layer works - it writes, it reads back, and it
// reported 825 of 20976 cells on a 747. It is also measuring the wrong thing.
// The vertex patch classifies EVERY vertex the sim draws, so a parked aeroplane
// surrounded by terminal buildings marks the buildings as airframe. The measured
// box came out asymmetric in x, -27.4 to +33.2, and a symmetric aeroplane cannot
// do that.
//
// It is not fixable by tightening the classification, because the classification
// is all the shader has: a vertex carries no flag saying which model it came
// from. Isolating the aircraft's draws would mean recognising them by pipeline,
// and X-Plane draws the aeroplane with the same pipelines as everything else.
//
// The .acf gives the geometry directly, on the CPU, before a frame is drawn.
// It cannot include scenery because scenery is not in it.
//
// WHAT "OCCUPIED" MEANS HERE
//
// The SURFACE, not the enclosed volume. An airframe is a shell - skin, spars,
// ribs - and a crash breaks the shell up; filling the interior would seed
// fragments in the middle of an empty fuselage and quadruple the count for
// nothing. The plan wants 800-2000 fragments for an airliner, which is the
// order a shell gives at these cell sizes.
//
// SAMPLING RATE
//
// Every surface is sampled finely enough that consecutive samples are closer
// than half a cell, so a surface cannot pass through a cell without marking it.
// Sampling by vertex instead would leave holes: a 747's loft has 20 stations
// over 70 m, which is 3.5 m apart on a 1.73 m grid - a hull with two empty
// cells between every marked one, which looks like it worked and is not.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "grid.h"
#include "gpu_layout.h"
#include "acf_planform.h"

namespace destruct {

// Marks one point's cell. Points outside the grid are dropped rather than
// clamped: clamping would smear anything that overhangs the box onto its face,
// and the box comes from this same geometry, so an outside point means a bug
// rather than an aircraft that does not fit.
inline bool voxMark(const Grid &g, const float p[3],
                    unsigned char *occ, uint32_t cells, uint32_t &hits)
{
    const int c = gridClassify(g, p);
    if (c < 0 || (uint32_t)c >= cells) return false;
    if (!occ[c]) { occ[c] = 1; ++hits; }
    return true;
}

// Samples the segment a..b at better than half-cell spacing.
inline void voxSegment(const Grid &g, const float a[3], const float b[3],
                       unsigned char *occ, uint32_t cells, uint32_t &hits)
{
    float d[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };
    const float len = sqrtf(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
    int steps = (int)(len / (0.5f * g.cell)) + 1;
    if (steps > 4096) steps = 4096;        // a degenerate part cannot hang this
    for (int i = 0; i <= steps; ++i) {
        const float t = (float)i / (float)steps;
        const float p[3] = { a[0] + d[0]*t, a[1] + d[1]*t, a[2] + d[2]*t };
        voxMark(g, p, occ, cells, hits);
    }
}

// Samples the quad p00-p10-p11-p01 as a lattice. Bilinear rather than two
// triangles: the quads here are planform panels and loft patches, both of which
// are naturally parameterised, and a lattice needs no edge cases.
inline void voxQuad(const Grid &g, const float p00[3], const float p10[3],
                    const float p01[3], const float p11[3],
                    unsigned char *occ, uint32_t cells, uint32_t &hits)
{
    // Step counts from the LONGEST edge on each axis of the parameterisation,
    // so a long thin panel is sampled finely along its length without wasting
    // samples across its width.
    float e0 = 0.0f, e1 = 0.0f;
    for (int a = 0; a < 3; ++a) {
        const float u0 = p10[a] - p00[a], u1 = p11[a] - p01[a];
        const float v0 = p01[a] - p00[a], v1 = p11[a] - p10[a];
        e0 += (u0*u0 > u1*u1 ? u0*u0 : u1*u1);
        e1 += (v0*v0 > v1*v1 ? v0*v0 : v1*v1);
    }
    int nu = (int)(sqrtf(e0) / (0.5f * g.cell)) + 1;
    int nv = (int)(sqrtf(e1) / (0.5f * g.cell)) + 1;
    if (nu > 512) nu = 512;
    if (nv > 512) nv = 512;

    for (int i = 0; i <= nu; ++i) {
        const float u = (float)i / (float)nu;
        for (int j = 0; j <= nv; ++j) {
            const float v = (float)j / (float)nv;
            float p[3];
            for (int a = 0; a < 3; ++a) {
                const float top = p00[a] + (p10[a] - p00[a]) * u;
                const float bot = p01[a] + (p11[a] - p01[a]) * u;
                p[a] = top + (bot - top) * v;
            }
            voxMark(g, p, occ, cells, hits);
        }
    }
}

// ---- ONE FLYING SURFACE, WITH THICKNESS.
//
// Both faces are stamped, offset along the surface normal by half the local
// thickness. A single sheet would give a wing one cell of depth wherever it
// happened to land, so a wing would fragment into a sheet of tiles rather than
// into something with a top and a bottom.
inline void voxWing(const Grid &g, const WingSeg &w, const float off[3],
                    unsigned char *occ, uint32_t cells, uint32_t &hits)
{
    float d[3];
    spanVector(w, d);
    const float dlen = sqrtf(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
    if (dlen <= 1e-6f) return;
    const float sx = d[0]/dlen, sy = d[1]/dlen;

    // n = cross(chordDir, spanDir), chordDir = (0,0,1). Points along +y for a
    // wing and along +x for a fin, so a vertical surface gets its thickness
    // across it rather than along it.
    float nx = -sy, ny = sx, nz = 0.0f;
    const float nl = sqrtf(nx*nx + ny*ny + nz*nz);
    if (nl > 1e-6f) { nx /= nl; ny /= nl; nz /= nl; }
    else { nx = 0.0f; ny = 1.0f; nz = 0.0f; }

    for (int face = -1; face <= 1; face += 2) {
        // Root and tip corners, displaced onto this face.
        const float hr = 0.5f * w.croot * kThicknessRatio * (float)face;
        const float ht = 0.5f * w.ctip  * kThicknessRatio * (float)face;

        const float rLE[3] = { w.rootX + nx*hr + off[0],
                               w.rootY + ny*hr + off[1],
                               w.rootZ + nz*hr + off[2] };
        const float rTE[3] = { rLE[0], rLE[1], rLE[2] + w.croot };
        const float tLE[3] = { w.rootX + d[0] + nx*ht + off[0],
                               w.rootY + d[1] + ny*ht + off[1],
                               w.rootZ + d[2] + nz*ht + off[2] };
        const float tTE[3] = { tLE[0], tLE[1], tLE[2] + w.ctip };

        voxQuad(g, rLE, tLE, rTE, tTE, occ, cells, hits);
    }
}

// ---- THE HULL.
//
// Each loft patch is a quad between two stations and two ring points. Patches
// whose corners the file did not all supply are skipped rather than guessed:
// an unset point is zero, and stitching to the origin would draw a spike from
// the hull to the aircraft datum.
inline void voxBody(const Grid &g, const Airframe::BodyLoft &b, const float off[3],
                    bool mirror, unsigned char *occ, uint32_t cells, uint32_t &hits)
{
    if (b.ni < 2 || b.nj < 2) return;
    const float m = mirror ? -1.0f : 1.0f;

    for (int i = 0; i + 1 < b.ni; ++i) {
        for (int j = 0; j + 1 < b.nj; ++j) {
            const size_t k00 = (size_t)i * b.nj + j;
            const size_t k10 = (size_t)(i + 1) * b.nj + j;
            const size_t k01 = (size_t)i * b.nj + (j + 1);
            const size_t k11 = (size_t)(i + 1) * b.nj + (j + 1);
            if (!b.set[k00] || !b.set[k10] || !b.set[k01] || !b.set[k11]) continue;

            float p00[3], p10[3], p01[3], p11[3];
            const size_t ks[4] = { k00, k10, k01, k11 };
            float *ps[4] = { p00, p10, p01, p11 };
            for (int c = 0; c < 4; ++c) {
                ps[c][0] = m * b.xyz[ks[c]*3 + 0] + off[0];
                ps[c][1] =     b.xyz[ks[c]*3 + 1] + off[1];
                ps[c][2] =     b.xyz[ks[c]*3 + 2] + off[2];
            }
            voxQuad(g, p00, p10, p01, p11, occ, cells, hits);
        }
    }
}

// ---- THE WHOLE AEROPLANE.
//
// off carries the .acf frame into the frame X-Plane draws in - see
// referencePointOffset(). Returns the number of cells marked, so the caller can
// say what it got rather than assume it worked.
inline uint32_t voxeliseAirframe(const Airframe &a, const float off[3],
                                 const Grid &g, unsigned char *occ, uint32_t cells)
{
    if (!occ || cells == 0) return 0;
    memset(occ, 0, cells);
    uint32_t hits = 0;

    for (size_t i = 0; i < a.wings.size(); ++i)
        voxWing(g, a.wings[i], off, occ, cells, hits);

    for (size_t i = 0; i < a.bodies.size(); ++i) {
        voxBody(g, a.bodies[i], off, false, occ, cells, hits);
        // Bodies are stored one side and mirrored by X-Plane. A body ON the
        // centreline mirrors onto itself, which costs repeated marks of cells
        // already marked and no error.
        voxBody(g, a.bodies[i], off, true, occ, cells, hits);
    }

    // ---- AND THEN MAKE IT SOLID.
    //
    // The shell alone tears the aeroplane apart. Displacement is decided PER
    // VERTEX - a vertex moves only when its cell is occupied - and the mesh
    // being drawn is the detailed OBJ, while this occupancy comes from the
    // .acf aero geometry. The two do not coincide: fairings, nacelle detail
    // and panel lines sit in cells the aero surface never passes through.
    //
    // So every triangle spanning a marked and an unmarked cell gets stretched,
    // and a 5 m offset splayed the wings and warped the fuselage instead of
    // lifting the aeroplane. "An airframe is a shell" is true of the STRUCTURE
    // and false of the test that decides which vertices belong to it.
    //
    // Filling between the first and last occupied cell along each axis makes
    // the volume solid. It is not an exact interior - a scanline fill cannot
    // be, with wings and fuselage sharing columns - but everything it adds is
    // inside the aeroplane's own extent, which is the only property that
    // matters here: a vertex of the aircraft should be in an occupied cell.
    //
    // Done on all three axes and unioned, so a gap in the aero hull at a
    // wing-body junction cannot leak the fill along one of them.
    {
        std::vector<unsigned char> solid(occ, occ + cells);

        // Along x, for each (y, z) column.
        for (int z = 0; z < g.nz; ++z)
        for (int y = 0; y < g.ny; ++y) {
            int lo = -1, hi = -1;
            for (int x = 0; x < g.nx; ++x) {
                const int i = x + y*g.nx + z*g.nx*g.ny;
                if ((uint32_t)i < cells && occ[i]) { if (lo < 0) lo = x; hi = x; }
            }
            for (int x = lo; x >= 0 && x <= hi; ++x) {
                const int i = x + y*g.nx + z*g.nx*g.ny;
                if ((uint32_t)i < cells) solid[i] = 1;
            }
        }

        // Along y, for each (x, z) column.
        for (int z = 0; z < g.nz; ++z)
        for (int x = 0; x < g.nx; ++x) {
            int lo = -1, hi = -1;
            for (int y = 0; y < g.ny; ++y) {
                const int i = x + y*g.nx + z*g.nx*g.ny;
                if ((uint32_t)i < cells && occ[i]) { if (lo < 0) lo = y; hi = y; }
            }
            for (int y = lo; y >= 0 && y <= hi; ++y) {
                const int i = x + y*g.nx + z*g.nx*g.ny;
                if ((uint32_t)i < cells) solid[i] = 1;
            }
        }

        // Along z, for each (x, y) column.
        for (int y = 0; y < g.ny; ++y)
        for (int x = 0; x < g.nx; ++x) {
            int lo = -1, hi = -1;
            for (int z = 0; z < g.nz; ++z) {
                const int i = x + y*g.nx + z*g.nx*g.ny;
                if ((uint32_t)i < cells && occ[i]) { if (lo < 0) lo = z; hi = z; }
            }
            for (int z = lo; z >= 0 && z <= hi; ++z) {
                const int i = x + y*g.nx + z*g.nx*g.ny;
                if ((uint32_t)i < cells) solid[i] = 1;
            }
        }

        // ---- ONE CELL OF DILATION.
        //
        // The OBJ mesh overhangs the aero geometry - radome, tailcone, engine
        // nacelles, wingtip devices - so its outermost vertices sit just
        // OUTSIDE the solid volume and would still be left behind. One cell of
        // margin covers that without meaningfully widening the aeroplane.
        std::vector<unsigned char> grown(solid);
        for (int z = 0; z < g.nz; ++z)
        for (int y = 0; y < g.ny; ++y)
        for (int x = 0; x < g.nx; ++x) {
            const int i = x + y*g.nx + z*g.nx*g.ny;
            if ((uint32_t)i >= cells || !solid[i]) continue;
            for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx2 = x+dx, ny2 = y+dy, nz2 = z+dz;
                if (nx2 < 0 || ny2 < 0 || nz2 < 0) continue;
                if (nx2 >= g.nx || ny2 >= g.ny || nz2 >= g.nz) continue;
                const int j = nx2 + ny2*g.nx + nz2*g.nx*g.ny;
                if ((uint32_t)j < cells) grown[j] = 1;
            }
        }

        hits = 0;
        for (uint32_t i = 0; i < cells; ++i) {
            occ[i] = grown[i];
            if (occ[i]) ++hits;
        }
    }

    return hits;
}

// The gear reaches below the lowest airframe vertex and a crash breaks it too,
// so the legs are stamped as lines from their attachment to their contact
// point. Coarse deliberately: a leg is a strut, and its cells are the ones it
// passes through.
inline uint32_t voxeliseGear(const Airframe &a, const float off[3],
                             const Grid &g, unsigned char *occ, uint32_t cells)
{
    uint32_t hits = 0;
    for (size_t i = 0; i < a.gear.size(); ++i) {
        const GearLeg &lg = a.gear[i];
        if (lg.legLen <= 0.0f) continue;      // an unused slot, not a leg
        const float top[3] = { lg.x + off[0], lg.y + off[1], lg.z + off[2] };
        const float bot[3] = { top[0], top[1] - lg.legLen - lg.tyreRad, top[2] };
        voxSegment(g, top, bot, occ, cells, hits);
    }
    return hits;
}

} // namespace destruct
