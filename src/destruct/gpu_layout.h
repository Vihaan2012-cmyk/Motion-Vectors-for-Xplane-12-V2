// The layout of the buffer the vertex shader reads, and the cell arithmetic it
// performs - written once, here, so the CPU and the GPU cannot disagree.
//
// WHY THIS FILE EXISTS
//
// The plan's Task 9 says it outright: if the vertex patch classifies a vertex
// into a different cell from the one gridClassify() would give it, the CPU
// builds a constraint graph over cells the GPU never displaces. Nothing crashes
// and nothing logs. The aeroplane simply breaks along the wrong seams, and the
// only way to notice is to look at it.
//
// Two copies of a formula in two languages is exactly the shape of that bug. So
// gpuCellIndex() below IS the arithmetic the SPIR-V emits, expressed in C so it
// can be run against gridClassify() in tests.ps1 at a command line. If those two
// ever disagree the test fails, rather than the aeroplane looking subtly wrong
// four tasks later.
//
// LAYOUT NOTES
//
// Occupancy is one uint per cell, not one byte. Byte-addressed storage needs
// VK_KHR_8bit_storage, which X-Plane does not enable and which we would have to
// inject into device creation to use - a much more invasive change than
// spending 128 KB. The cost is 4 bytes a cell for a buffer that is read once
// per crash.
//
// The occupancy write is a plain store of 1, not an atomicOr. Every vertex that
// lands in a cell writes the same value, so the race is benign: whoever wins
// writes 1, and 1 is what any of them would have written.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <stdint.h>
#include "grid.h"

namespace destruct {

// Cells the buffer is sized for. gridForBounds targets 20000; 32768 leaves
// headroom and keeps the occupancy region a round 128 KB.
static const uint32_t kMaxCells = 32768;

// Fragments. Matches destructgpu::kMaxFragments; asserted equal there.
static const uint32_t kMaxGpuFragments = 4096;

// ---- BUFFER LAYOUT, in bytes, std430.
//
//   0    mat4  aircraftInv      world/view -> aircraft-local
//   64   vec4  gridMinCell      min.xyz, cell size in w
//   80   ivec4 gridDim          nx, ny, nz, active
//   96   uint  occupancy[kMaxCells]
//   ...  vec4  xform[kMaxGpuFragments]
static const uint32_t kOffAircraftInv = 0;
static const uint32_t kOffGridMinCell = 64;
static const uint32_t kOffGridDim     = 80;
static const uint32_t kOffOccupancy   = 96;

// ---- THE DISCARD SLOT, AND WHY A BRANCH-FREE SHADER NEEDS ONE.
//
// The vertex patch has to skip the occupancy write for any vertex outside the
// grid, and for every vertex at all when discovery is off. The obvious way is
// a branch - and emitting a branch means creating new basic blocks inside a
// function somebody else wrote, splitting the block the injection lands in and
// fixing up its terminator. That is the most invasive thing this injector
// could do to a module, in 15000 pipelines, where a mistake is a black screen.
//
// So the shader is branch-free: it computes the cell index, computes whether
// that index is legal, and OpSelects between the real index and THIS slot. The
// store then happens unconditionally. A rejected vertex writes a 1 into a word
// nothing ever reads.
//
// It has to be its own word rather than a reused cell: writing rejects into
// cell 0 would mark the aeroplane's first cell occupied from anywhere in the
// world, and writing them into the transform region would corrupt a fragment.
//
// Padded to 16 so the transforms stay vector-aligned.
static const uint32_t kOffDiscard     = kOffOccupancy + kMaxCells * 4u;
static const uint32_t kOffXform       = kOffDiscard + 16u;
static const uint32_t kBufferBytes    = kOffXform + kMaxGpuFragments * 16u;

// Word (4-byte) indices, which is how the SPIR-V addresses the block: a
// storage buffer declared as a uint array indexes in words, not bytes, and
// getting that conversion wrong once cost a whole debugging session on the
// velocity buffer.
static const uint32_t kWordOccupancy = kOffOccupancy / 4u;
static const uint32_t kWordDiscard   = kOffDiscard / 4u;
static const uint32_t kWordXform     = kOffXform / 4u;

// The shader addresses one flat uint array that begins at kOffOccupancy, so
// every index it uses is relative to THAT, not to the buffer. Keeping the two
// conversions here rather than in the emitter is the same rule as the cell
// formula: one place, testable.
static const uint32_t kDataWords     = (kBufferBytes - kOffOccupancy) / 4u;
static const uint32_t kDataDiscard   = kWordDiscard - kWordOccupancy;
static const uint32_t kDataXform     = kWordXform   - kWordOccupancy;

// ---- THE CLASSIFICATION THE SHADER PERFORMS.
//
// Deliberately written with the same guards and the same order of operations as
// gridClassify(): subtract, divide, reject negative, truncate, reject high.
//
// The truncation matters. C's (int) cast and SPIR-V's OpConvertFToS both round
// toward zero, which is only the same as floor() for non-negative input - which
// is why the negative case is rejected BEFORE the cast rather than after.
inline int gpuCellIndex(const float local[3], const float gmin[3],
                        float cell, int nx, int ny, int nz)
{
    const int n[3] = { nx, ny, nz };
    int c[3];
    for (int i = 0; i < 3; ++i) {
        const float f = (local[i] - gmin[i]) / cell;
        if (f < 0.0f) return -1;
        c[i] = (int)f;
        if (c[i] >= n[i]) return -1;
    }
    return c[0] + c[1] * nx + c[2] * nx * ny;
}

// Convenience overload against a Grid, so a test can compare the two directly.
inline int gpuCellIndex(const Grid &g, const float local[3])
{
    return gpuCellIndex(local, g.min, g.cell, g.nx, g.ny, g.nz);
}

// ---- WHAT THE SHADER STORES TO, INCLUDING THE REJECT CASE.
//
// Mirrors the OpSelect chain the emitter produces. Given a point and the grid,
// returns the DATA-RELATIVE word index the shader will write a 1 into: the
// cell when the point is inside and discovery is on, the discard slot
// otherwise. Tested against gridClassify so the reject path is covered too,
// not just the accept path.
inline uint32_t gpuStoreIndex(const float local[3], const float gmin[3],
                              float cell, int nx, int ny, int nz, int active)
{
    if (!active) return kDataDiscard;
    const int c = gpuCellIndex(local, gmin, cell, nx, ny, nz);
    if (c < 0) return kDataDiscard;
    if ((uint32_t)c >= kMaxCells) return kDataDiscard;
    return (uint32_t)c;
}

// How many cells a grid actually uses. Anything at or above this in the
// occupancy region is stale from a previous, larger grid.
inline uint32_t gpuCellCount(const Grid &g)
{
    const int64_t n = (int64_t)g.nx * g.ny * g.nz;
    if (n <= 0) return 0;
    return (uint32_t)(n > (int64_t)kMaxCells ? kMaxCells : n);
}

} // namespace destruct
