#pragma once
#include "grid.h"
#include <vector>

namespace destruct {

// Which cells actually contain geometry.
//
// Neither the plugin nor the layer can see the aircraft mesh, so this is not
// computed - it is DISCOVERED. The patched vertex shader sets the bit for its
// own cell during one frame, and one frame of rendering yields the airframe's
// shape. Any aircraft, no authoring, no mesh analysis.
//
// Classification does not need this: a vertex lands where it lands, and an
// empty cell simply never receives one. CONSTRAINTS need it, because joining
// two empty cells would run a load path through thin air, and a fragment with
// no geometry in it would be an invisible body pulling on visible ones.
struct Occupancy {
    std::vector<unsigned char> bit;
};

inline void occupancyInit(Occupancy &o, const Grid &g)
{
    o.bit.assign((size_t)gridCells(g), 0);
}

inline void occupancySet(Occupancy &o, int cellIndex)
{
    if (cellIndex < 0 || (size_t)cellIndex >= o.bit.size()) return;
    o.bit[(size_t)cellIndex] = 1;
}

inline int occupancyCount(const Occupancy &o)
{
    int n = 0;
    for (size_t i = 0; i < o.bit.size(); ++i) if (o.bit[i]) ++n;
    return n;
}

// One breakable joint between two occupied cells.
struct Link { int a, b; float rest; };

// Six-connected: +x, +y and +z only, so each pair is emitted exactly once and
// no deduplication is needed.
//
// Face adjacency rather than 26-connected is deliberate. Diagonals treble the
// graph and stiffen the structure in a way that has to be compensated for in
// the break thresholds, so the cheaper graph is also the easier one to tune.
inline std::vector<Link> buildLinks(const Occupancy &o, const Grid &g)
{
    std::vector<Link> out;
    for (int z = 0; z < g.nz; ++z)
    for (int y = 0; y < g.ny; ++y)
    for (int x = 0; x < g.nx; ++x) {
        const int i = x + y * g.nx + z * g.nx * g.ny;
        if (!o.bit[(size_t)i]) continue;
        const int dx[3] = { 1, 0, 0 };
        const int dy[3] = { 0, 1, 0 };
        const int dz[3] = { 0, 0, 1 };
        for (int d = 0; d < 3; ++d) {
            const int xn = x + dx[d], yn = y + dy[d], zn = z + dz[d];
            if (xn >= g.nx || yn >= g.ny || zn >= g.nz) continue;
            const int j = xn + yn * g.nx + zn * g.nx * g.ny;
            if (!o.bit[(size_t)j]) continue;
            Link L; L.a = i; L.b = j; L.rest = g.cell;
            out.push_back(L);
        }
    }
    return out;
}

// Occupied cells, in index order.
//
// Transforms are allocated for OCCUPIED cells only. An airliner's bounding box
// is mostly air - wings are thin sheets in a large volume - so occupancy runs
// at a few percent, and allocating for every cell would waste most of the
// buffer on fragments that can never be seen.
inline std::vector<int> occupiedCells(const Occupancy &o)
{
    std::vector<int> out;
    for (size_t i = 0; i < o.bit.size(); ++i)
        if (o.bit[i]) out.push_back((int)i);
    return out;
}

}  // namespace destruct
