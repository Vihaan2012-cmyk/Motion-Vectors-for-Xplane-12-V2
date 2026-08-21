# Crash Destruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The user's aircraft fragments into 800–2000 pieces on impact, pieces stay joined until their constraints break, and the wreckage settles on the terrain instead of sinking through it.

**Architecture:** The patched vertex shader recovers view-space position from `gl_Position`, transforms it into airframe-local coordinates using a transform the plugin publishes each frame, classifies it into a uniform grid cell, and applies that cell's rigid transform. No mesh analysis, no object identity, no per-aircraft authoring. Fragment physics is a position-based dynamics solver over the occupied cells, running on the CPU in the plugin.

**Tech Stack:** C++17, Vulkan 1.3 explicit layer, XPLM SDK plugin, GLSL compute/vertex via hand-emitted SPIR-V, MinGW-w64 g++, PowerShell build.

**Spec:** `docs/superpowers/specs/2026-08-21-crash-destruction-design.md`

## Global Constraints

- Target fragment size is **1.2 m**; cell size scales with the aircraft bounding box so count follows aircraft size. Target 800–2000 fragments for an airliner.
- Everything is live-controllable through `%TEMP%\taa_live.ini` under the `crash.*` prefix, matching the existing `taa.*` and `vram.*` convention.
- The system is **inert until a crash fires**. Zero per-vertex cost in normal flight beyond one uniform branch.
- `TAA_VERSION` in `src/share.h` must be bumped whenever the shared struct changes; the reader bails on `structSize` mismatch.
- Pure maths lives in headers under `src/destruct/` and is unit-tested offline. Nothing that can be tested on the CPU is tested by flying.
- No feature may be gated behind an environment variable that only the development launcher sets — `learnings.md` records that trap three times.
- Every silent refusal gets a counter. A guard that returns without counting is indistinguishable from a subsystem that does not exist (see the recycle-pool and reactive-mask findings in git history).

---

### Task 1: Offline test harness

`src/test_math.cpp` established the pattern but nothing builds it any more (`build/test_math.exe` is stale from 2026-08-11). Every later task needs somewhere to put a test, so this comes first.

**Files:**
- Create: `tests.ps1`
- Create: `src/test_destruct.cpp`
- Create: `src/destruct/grid.h`

**Interfaces:**
- Consumes: nothing.
- Produces: `.\tests.ps1` compiles `src/test_destruct.cpp` to `build/test_destruct.exe`, runs it, and returns exit code 0 only when every check passed. `check(bool ok, const char *what)` and `checkNear(double got, double want, double tol, const char *what)` are available to later tasks.

- [ ] **Step 1: Write the failing test**

`src/test_destruct.cpp`:

```cpp
#include "destruct/grid.h"
#include <cstdio>
#include <cmath>

static int g_fail = 0;

static void check(bool ok, const char *what)
{
    if (ok) { printf("  PASS  %s\n", what); }
    else    { printf("  FAIL  %s\n", what); ++g_fail; }
}

static void checkNear(double got, double want, double tol, const char *what)
{
    bool ok = std::fabs(got - want) <= tol;
    if (ok) printf("  PASS  %s (%.4f)\n", what, got);
    else  { printf("  FAIL  %s: got %.4f want %.4f +/- %.4f\n",
                   what, got, want, tol); ++g_fail; }
}

int main()
{
    printf("destruct tests\n");
    check(destruct::selfTestPresent(), "grid.h is reachable from the test");
    printf("%s: %d failure(s)\n", g_fail ? "FAILED" : "OK", g_fail);
    return g_fail ? 1 : 0;
}
```

- [ ] **Step 2: Run it to verify it fails**

```
.\tests.ps1
```

Expected: compile error, `destruct/grid.h: No such file or directory`.

- [ ] **Step 3: Write the minimal implementation**

`tests.ps1`:

```powershell
# Offline unit tests for the pure-maths half of the crash destruction system.
#
# Everything here runs without X-Plane. Grid classification, rigid-body
# integration, the constraint solver and the ground clamp are ordinary
# arithmetic, and arithmetic that can be tested at a command line should never
# be tested by flying an aeroplane into the ground.
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$out  = Join-Path $root "build"
New-Item -ItemType Directory -Force $out | Out-Null

& g++ -o "$out\test_destruct.exe" "$root\src\test_destruct.cpp" `
      -I"$root\src" -m64 -O1 -std=c++17 -Wall -Wextra
if ($LASTEXITCODE -ne 0) { throw "test build failed" }

& "$out\test_destruct.exe"
exit $LASTEXITCODE
```

`src/destruct/grid.h`:

```cpp
#pragma once
namespace destruct {
inline bool selfTestPresent() { return true; }
}
```

- [ ] **Step 4: Run the tests and make sure they pass**

```
.\tests.ps1
```

Expected: `PASS  grid.h is reachable from the test`, then `OK: 0 failure(s)`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add tests.ps1 src/test_destruct.cpp src/destruct/grid.h
git commit -m "Offline test harness for the destruction maths"
```

---

### Task 2: Grid geometry — cell size, classification, indexing

**Files:**
- Modify: `src/destruct/grid.h`
- Modify: `src/test_destruct.cpp`

**Interfaces:**
- Consumes: `check`, `checkNear` from Task 1.
- Produces:
  - `struct Grid { float min[3]; float cell; int nx, ny, nz; }`
  - `Grid gridForBounds(const float bbMin[3], const float bbMax[3], float targetCell)` — chooses a cell size at or below `targetCell` such that no axis exceeds 64 cells, and returns the grid covering the box.
  - `int gridCells(const Grid &g)` — total cells, `nx*ny*nz`.
  - `int gridClassify(const Grid &g, const float local[3])` — cell index, or `-1` when outside the grid.

- [ ] **Step 1: Write the failing test**

Add to `main()` in `src/test_destruct.cpp`, before the summary `printf`:

```cpp
    {
        // A 38 x 36 x 12 m airliner box at 1.2 m should land in the target
        // fragment band once occupancy is accounted for. Here we only assert
        // the grid itself: cell size honoured, axis counts correct.
        const float lo[3] = { -19.0f, -6.0f, -18.0f };
        const float hi[3] = {  19.0f,  6.0f,  18.0f };
        destruct::Grid g = destruct::gridForBounds(lo, hi, 1.2f);
        checkNear(g.cell, 1.2, 0.2, "airliner cell size near target");
        check(g.nx == 32 && g.ny == 10 && g.nz == 30, "airliner axis counts");
        check(destruct::gridCells(g) == 32 * 10 * 30, "cell count is the product");

        // Centre of the box lands inside; a point well outside returns -1.
        const float mid[3]  = { 0.0f, 0.0f, 0.0f };
        const float away[3] = { 500.0f, 0.0f, 0.0f };
        check(destruct::gridClassify(g, mid)  >= 0, "centre classifies");
        check(destruct::gridClassify(g, away) == -1, "far point is outside");

        // Two points one cell apart on x differ by exactly 1 in index.
        const float a[3] = { -18.5f, 0.0f, 0.0f };
        const float b[3] = { -18.5f + 1.2f, 0.0f, 0.0f };
        check(destruct::gridClassify(g, b) - destruct::gridClassify(g, a) == 1,
              "adjacent x cells are adjacent indices");
    }
    {
        // A Cessna-sized box must not collapse to a handful of fragments.
        const float lo[3] = { -5.5f, -1.5f, -4.0f };
        const float hi[3] = {  5.5f,  1.5f,  4.0f };
        destruct::Grid g = destruct::gridForBounds(lo, hi, 1.2f);
        check(destruct::gridCells(g) > 400, "light aircraft still subdivides");
    }
    {
        // The 64-cell axis cap must hold, or the transform buffer and the
        // constraint graph both blow up on a very large aircraft.
        const float lo[3] = { -60.0f, -12.0f, -60.0f };
        const float hi[3] = {  60.0f,  12.0f,  60.0f };
        destruct::Grid g = destruct::gridForBounds(lo, hi, 0.2f);
        check(g.nx <= 64 && g.ny <= 64 && g.nz <= 64, "axis cap holds");
    }
```

- [ ] **Step 2: Run it to verify it fails**

```
.\tests.ps1
```

Expected: compile error, `'gridForBounds' is not a member of 'destruct'`.

- [ ] **Step 3: Write the minimal implementation**

Replace the contents of `src/destruct/grid.h`:

```cpp
#pragma once
#include <cmath>

namespace destruct {

inline bool selfTestPresent() { return true; }

// A uniform grid over the airframe's bounding box, in airframe-local metres.
//
// Classification is arithmetic, not a lookup: empty cells cost nothing because
// no vertex ever lands in one, so the grid may cover the whole box even though
// an aircraft occupies only a few percent of it.
struct Grid {
    float min[3];
    float cell;
    int   nx, ny, nz;
};

// Cell size is chosen from the aircraft's own size so FRAGMENT SIZE stays
// constant across aircraft - a Cessna must not become eight slabs and a 747
// must not become twenty thousand.
//
// The 64-per-axis cap is a hard limit, not a preference. Both the transform
// buffer and the constraint graph scale with cell count, and an uncapped grid
// on a very large aircraft with a small target would allocate absurdly.
inline Grid gridForBounds(const float bbMin[3], const float bbMax[3],
                          float targetCell)
{
    Grid g;
    float cell = targetCell > 0.01f ? targetCell : 1.2f;
    for (int i = 0; i < 3; ++i) g.min[i] = bbMin[i];

    // Grow the cell until every axis fits under the cap.
    for (int guard = 0; guard < 64; ++guard) {
        bool ok = true;
        for (int i = 0; i < 3; ++i) {
            float span = bbMax[i] - bbMin[i];
            if (span < 0.0f) span = 0.0f;
            int n = (int)std::ceil(span / cell);
            if (n > 64) ok = false;
        }
        if (ok) break;
        cell *= 1.25f;
    }
    g.cell = cell;

    int n[3];
    for (int i = 0; i < 3; ++i) {
        float span = bbMax[i] - bbMin[i];
        if (span < 0.0f) span = 0.0f;
        n[i] = (int)std::ceil(span / cell);
        if (n[i] < 1)  n[i] = 1;
        if (n[i] > 64) n[i] = 64;
    }
    g.nx = n[0]; g.ny = n[1]; g.nz = n[2];
    return g;
}

inline int gridCells(const Grid &g) { return g.nx * g.ny * g.nz; }

// Index, or -1 when the point is outside the grid. The shader performs the
// identical computation; keeping them in step is what makes the CPU-side
// occupancy and constraint graph describe the same cells the GPU displaces.
inline int gridClassify(const Grid &g, const float local[3])
{
    int c[3];
    const int n[3] = { g.nx, g.ny, g.nz };
    for (int i = 0; i < 3; ++i) {
        float f = (local[i] - g.min[i]) / g.cell;
        if (f < 0.0f) return -1;
        c[i] = (int)f;
        if (c[i] >= n[i]) return -1;
    }
    return c[0] + c[1] * g.nx + c[2] * g.nx * g.ny;
}

}  // namespace destruct
```

- [ ] **Step 4: Run the tests and make sure they pass**

```
.\tests.ps1
```

Expected: every grid check PASS, `OK: 0 failure(s)`.

- [ ] **Step 5: Commit**

```bash
git add src/destruct/grid.h src/test_destruct.cpp
git commit -m "Grid geometry: cell size from aircraft bounds, arithmetic classification"
```

---

### Task 3: Occupancy and the constraint graph

**Files:**
- Create: `src/destruct/occupancy.h`
- Modify: `src/test_destruct.cpp`

**Interfaces:**
- Consumes: `Grid`, `gridCells`, `gridClassify` from Task 2.
- Produces:
  - `struct Occupancy { std::vector<unsigned char> bit; }` — one byte per cell, non-zero when geometry landed there.
  - `void occupancySet(Occupancy &o, const Grid &g, int cellIndex)`
  - `int occupancyCount(const Occupancy &o)`
  - `struct Link { int a, b; float rest; }`
  - `std::vector<Link> buildLinks(const Occupancy &o, const Grid &g)` — six-connected neighbours, each pair once, only between occupied cells, `rest` set to `g.cell`.

- [ ] **Step 1: Write the failing test**

Add to `main()`:

```cpp
    {
        const float lo[3] = { 0.0f, 0.0f, 0.0f };
        const float hi[3] = { 4.0f, 4.0f, 4.0f };
        destruct::Grid g = destruct::gridForBounds(lo, hi, 1.0f);
        check(g.nx == 4 && g.ny == 4 && g.nz == 4, "4x4x4 test grid");

        destruct::Occupancy o;
        o.bit.assign((size_t)destruct::gridCells(g), 0);

        // A 2x1x1 domino: two occupied cells side by side on x.
        const float p0[3] = { 0.5f, 0.5f, 0.5f };
        const float p1[3] = { 1.5f, 0.5f, 0.5f };
        destruct::occupancySet(o, g, destruct::gridClassify(g, p0));
        destruct::occupancySet(o, g, destruct::gridClassify(g, p1));
        check(destruct::occupancyCount(o) == 2, "two cells occupied");

        std::vector<destruct::Link> links = destruct::buildLinks(o, g);
        check(links.size() == 1, "domino yields exactly one link");
        checkNear(links[0].rest, g.cell, 1e-5, "link rest length is the cell size");

        // Setting the same cell twice must not double-count or duplicate links.
        destruct::occupancySet(o, g, destruct::gridClassify(g, p0));
        check(destruct::occupancyCount(o) == 2, "repeat set does not double count");
        check(destruct::buildLinks(o, g).size() == 1, "repeat set does not duplicate links");

        // Diagonal neighbours are NOT linked - six-connected only, or the
        // graph density and stiffness both change without anyone deciding to.
        const float pd[3] = { 1.5f, 1.5f, 0.5f };
        destruct::occupancySet(o, g, destruct::gridClassify(g, pd));
        std::vector<destruct::Link> l2 = destruct::buildLinks(o, g);
        check(l2.size() == 2, "diagonal adds one face-adjacent link, not two");
    }
```

Add `#include "destruct/occupancy.h"` and `#include <vector>` at the top of the file.

- [ ] **Step 2: Run it to verify it fails**

```
.\tests.ps1
```

Expected: `destruct/occupancy.h: No such file or directory`.

- [ ] **Step 3: Write the minimal implementation**

`src/destruct/occupancy.h`:

```cpp
#pragma once
#include "grid.h"
#include <vector>

namespace destruct {

// Which cells actually contain geometry.
//
// Neither the plugin nor the layer can see the aircraft mesh, so this is not
// computed - it is DISCOVERED, by having the patched vertex shader set the bit
// for its own cell during one frame. One frame of rendering yields the
// airframe's shape, for any aircraft, with no authoring.
//
// Classification does not need this. Constraints do: joining two empty cells
// would put load paths through air.
struct Occupancy {
    std::vector<unsigned char> bit;
};

inline void occupancySet(Occupancy &o, const Grid &g, int cellIndex)
{
    (void)g;
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

// Six-connected: +x, +y, +z only, so each pair is emitted exactly once.
//
// Face adjacency rather than 26-connected is a deliberate choice. Diagonal
// links trebles the graph and stiffens the structure in a way that has to be
// compensated in the break thresholds, so the cheaper graph is also the easier
// one to tune.
inline std::vector<Link> buildLinks(const Occupancy &o, const Grid &g)
{
    std::vector<Link> out;
    for (int z = 0; z < g.nz; ++z)
    for (int y = 0; y < g.ny; ++y)
    for (int x = 0; x < g.nx; ++x) {
        int i = x + y * g.nx + z * g.nx * g.ny;
        if (!o.bit[(size_t)i]) continue;
        const int dx[3] = { 1, 0, 0 };
        const int dy[3] = { 0, 1, 0 };
        const int dz[3] = { 0, 0, 1 };
        for (int d = 0; d < 3; ++d) {
            int nxp = x + dx[d], nyp = y + dy[d], nzp = z + dz[d];
            if (nxp >= g.nx || nyp >= g.ny || nzp >= g.nz) continue;
            int j = nxp + nyp * g.nx + nzp * g.nx * g.ny;
            if (!o.bit[(size_t)j]) continue;
            Link L; L.a = i; L.b = j; L.rest = g.cell;
            out.push_back(L);
        }
    }
    return out;
}

}  // namespace destruct
```

- [ ] **Step 4: Run the tests and make sure they pass**

```
.\tests.ps1
```

Expected: all occupancy and link checks PASS.

- [ ] **Step 5: Commit**

```bash
git add src/destruct/occupancy.h src/test_destruct.cpp
git commit -m "Occupancy grid and six-connected constraint graph"
```

---

### Task 4: Fragment integration and the ground plane

**Files:**
- Create: `src/destruct/body.h`
- Modify: `src/test_destruct.cpp`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `struct Fragment { float p[3]; float v[3]; float restP[3]; unsigned char alive; }`
  - `void integrate(Fragment *f, int n, float dt, float gravity, float damping)`
  - `struct GroundPlane { float height; float normal[3]; }`
  - `int clampToGround(Fragment *f, int n, const GroundPlane &gp, float radius, float restitution, float friction)` — returns how many fragments were in contact.

- [ ] **Step 1: Write the failing test**

Add to `main()`:

```cpp
    {
        // Free fall for one second at 9.81 covers 4.905 m, and Verlet-free
        // semi-implicit Euler at 60 Hz should land within a few centimetres.
        destruct::Fragment f;
        f.p[0] = 0; f.p[1] = 100; f.p[2] = 0;
        f.v[0] = 0; f.v[1] = 0;   f.v[2] = 0;
        f.restP[0] = f.restP[1] = f.restP[2] = 0;
        f.alive = 1;
        for (int i = 0; i < 60; ++i)
            destruct::integrate(&f, 1, 1.0f / 60.0f, 9.81f, 0.0f);
        checkNear(f.p[1], 100.0 - 4.905, 0.15, "one second of free fall");

        // Damping must remove energy, never add it.
        destruct::Fragment d = f;
        d.v[0] = 10.0f;
        destruct::integrate(&d, 1, 1.0f / 60.0f, 0.0f, 5.0f);
        check(std::fabs(d.v[0]) < 10.0f, "damping reduces speed");
        check(std::fabs(d.v[0]) > 0.0f,  "damping does not reverse it");
    }
    {
        // A fragment below the ground is lifted to rest on it and stops
        // descending. Sinking through terrain is the one artefact that would
        // make the whole system read as broken.
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
        check(f.p[1] >= 10.0f + 0.5f - 1e-3f, "fragment sits on the surface");
        check(f.v[1] >= 0.0f, "downward velocity is removed");

        // Friction with a fully rough surface stops lateral sliding.
        destruct::Fragment s = f;
        s.v[0] = 8.0f; s.v[1] = -1.0f;
        destruct::clampToGround(&s, 1, gp, 0.5f, 0.0f, 1.0f);
        checkNear(s.v[0], 0.0, 1e-3, "full friction halts sliding");

        // A fragment well above the plane is untouched.
        destruct::Fragment air = f;
        air.p[1] = 50.0f; air.v[1] = -3.0f;
        check(destruct::clampToGround(&air, 1, gp, 0.5f, 0.0f, 1.0f) == 0,
              "airborne fragment is not in contact");
        checkNear(air.v[1], -3.0, 1e-5, "airborne velocity is unchanged");
    }
```

Add `#include "destruct/body.h"` at the top.

- [ ] **Step 2: Run it to verify it fails**

```
.\tests.ps1
```

Expected: `destruct/body.h: No such file or directory`.

- [ ] **Step 3: Write the minimal implementation**

`src/destruct/body.h`:

```cpp
#pragma once
#include <cmath>

namespace destruct {

// One fragment. Position and velocity only - no orientation.
//
// Orientation is deliberately omitted. At 1.2 m fragments the eye reads
// POSITION and separation, not spin, and dropping the rotational state halves
// the buffer, removes quaternion normalisation from the hot loop, and makes
// the whole solver a positional problem. If tumbling proves necessary later it
// can be derived per fragment from its velocity rather than integrated.
struct Fragment {
    float p[3];
    float v[3];
    float restP[3];      // where this fragment sat on the intact airframe
    unsigned char alive;
};

// Semi-implicit Euler: velocity first, then position. Stable at the step sizes
// and stiffnesses this uses, unlike explicit Euler, and an order of magnitude
// simpler than anything better.
inline void integrate(Fragment *f, int n, float dt, float gravity, float damping)
{
    float k = 1.0f - damping * dt;
    if (k < 0.0f) k = 0.0f;          // never let damping invert the velocity
    for (int i = 0; i < n; ++i) {
        if (!f[i].alive) continue;
        f[i].v[1] -= gravity * dt;
        for (int a = 0; a < 3; ++a) {
            f[i].v[a] *= k;
            f[i].p[a] += f[i].v[a] * dt;
        }
    }
}

// One plane is enough for a debris field: XPLMProbeTerrainXYZ is sampled once
// at the crash site and the wreckage lands within a few tens of metres.
struct GroundPlane {
    float height;
    float normal[3];
};

// Returns the number of fragments in contact.
//
// restitution 0 means debris does not bounce, which is what wreckage does.
// friction 1 means it does not slide either. Both are knobs because the right
// answer is a look, not a number.
inline int clampToGround(Fragment *f, int n, const GroundPlane &gp,
                         float radius, float restitution, float friction)
{
    int hits = 0;
    float floorY = gp.height + radius;
    for (int i = 0; i < n; ++i) {
        if (!f[i].alive) continue;
        if (f[i].p[1] >= floorY) continue;
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

}  // namespace destruct
```

- [ ] **Step 4: Run the tests and make sure they pass**

```
.\tests.ps1
```

Expected: free fall, damping, contact, friction and airborne checks all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/destruct/body.h src/test_destruct.cpp
git commit -m "Fragment integration and ground plane contact"
```

---

### Task 5: Constraint solver with breakage

**Files:**
- Create: `src/destruct/solver.h`
- Modify: `src/test_destruct.cpp`

**Interfaces:**
- Consumes: `Fragment` from Task 4, `Link` from Task 3.
- Produces:
  - `struct SolverCfg { int iterations; float stiffness; float breakStrain; }`
  - `int solveLinks(Fragment *f, int n, Link *links, int linkCount, unsigned char *broken, const SolverCfg &cfg)` — returns how many links broke this call.

- [ ] **Step 1: Write the failing test**

Add to `main()`:

```cpp
    {
        // Two fragments joined by one link, pulled apart by less than the
        // break strain, must be drawn back together and the link must survive.
        destruct::Fragment f[2];
        for (int i = 0; i < 2; ++i) {
            f[i].v[0] = f[i].v[1] = f[i].v[2] = 0;
            f[i].restP[0] = f[i].restP[1] = f[i].restP[2] = 0;
            f[i].alive = 1;
            f[i].p[1] = f[i].p[2] = 0;
        }
        f[0].p[0] = 0.0f;
        f[1].p[0] = 1.3f;                 // rest 1.0, so 30% strain

        destruct::Link L; L.a = 0; L.b = 1; L.rest = 1.0f;
        unsigned char broken = 0;
        destruct::SolverCfg cfg;
        cfg.iterations = 10; cfg.stiffness = 1.0f; cfg.breakStrain = 0.5f;

        int nb = destruct::solveLinks(f, 2, &L, 1, &broken, cfg);
        check(nb == 0, "30% strain under a 50% threshold does not break");
        check(broken == 0, "link is still intact");
        float d = f[1].p[0] - f[0].p[0];
        checkNear(d, 1.0, 0.05, "link pulls back to rest length");

        // Beyond the threshold it breaks, and a broken link stops acting.
        f[0].p[0] = 0.0f; f[1].p[0] = 2.0f;   // 100% strain
        broken = 0;
        nb = destruct::solveLinks(f, 2, &L, 1, &broken, cfg);
        check(nb == 1, "100% strain over a 50% threshold breaks");
        check(broken == 1, "break is recorded");

        float before = f[1].p[0] - f[0].p[0];
        destruct::solveLinks(f, 2, &L, 1, &broken, cfg);
        float after = f[1].p[0] - f[0].p[0];
        checkNear(after, before, 1e-4, "a broken link exerts no further force");
    }
    {
        // Load redistribution: a chain of three with the middle link already
        // broken must not transmit force across the gap. This is what makes
        // failure propagate rather than happening everywhere at once.
        destruct::Fragment f[3];
        for (int i = 0; i < 3; ++i) {
            f[i].v[0] = f[i].v[1] = f[i].v[2] = 0;
            f[i].restP[0] = f[i].restP[1] = f[i].restP[2] = 0;
            f[i].alive = 1;
            f[i].p[0] = (float)i; f[i].p[1] = f[i].p[2] = 0;
        }
        f[2].p[0] = 5.0f;                       // far away
        destruct::Link L[2];
        L[0].a = 0; L[0].b = 1; L[0].rest = 1.0f;
        L[1].a = 1; L[1].b = 2; L[1].rest = 1.0f;
        unsigned char broken[2] = { 0, 1 };     // second link already broken
        destruct::SolverCfg cfg;
        cfg.iterations = 10; cfg.stiffness = 1.0f; cfg.breakStrain = 0.5f;

        float far0 = f[2].p[0];
        destruct::solveLinks(f, 3, L, 2, broken, cfg);
        checkNear(f[2].p[0], far0, 1e-4, "detached fragment is not dragged back");
    }
```

Add `#include "destruct/solver.h"` at the top.

- [ ] **Step 2: Run it to verify it fails**

```
.\tests.ps1
```

Expected: `destruct/solver.h: No such file or directory`.

- [ ] **Step 3: Write the minimal implementation**

`src/destruct/solver.h`:

```cpp
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

// Position-based dynamics: correct POSITIONS directly and let velocity follow.
//
// Chosen over a force-based spring solver because springs stiff enough to hold
// an airframe together need a timestep far below the frame rate to stay
// stable, and this has to run inside a flight loop callback. PBD is
// unconditionally stable at any stiffness - the cost is that stiffness becomes
// rate-dependent, which does not matter for wreckage.
//
// A break is permanent. Load then redistributes onto whatever is still joined,
// which is what makes failure PROPAGATE - the nose crushes, stress runs aft,
// the tail departs a moment later - rather than the airframe disintegrating
// uniformly, which reads as an explosion rather than a crash.
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
            float len = std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
            if (len < 1e-6f) continue;

            float rest = L.rest > 1e-6f ? L.rest : 1e-6f;
            float strain = std::fabs(len - rest) / rest;
            if (strain > cfg.breakStrain) {
                // Only count the break on the first iteration that sees it,
                // so the return value is "links that failed", not "iterations
                // that noticed".
                broken[i] = 1;
                ++brokeNow;
                continue;
            }

            float corr = (len - rest) / len * 0.5f * cfg.stiffness;
            for (int a = 0; a < 3; ++a) {
                A.p[a] += d[a] * corr;
                B.p[a] -= d[a] * corr;
            }
        }
    }
    return brokeNow;
}

}  // namespace destruct
```

- [ ] **Step 4: Run the tests and make sure they pass**

```
.\tests.ps1
```

Expected: strain, breakage, inert-after-break and redistribution checks all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/destruct/solver.h src/test_destruct.cpp
git commit -m "Position-based constraint solver with permanent breakage"
```

---

### Task 6: View-space reconstruction probe — the risk gate

This is the load-bearing assumption of the whole design and the only task that can invalidate everything downstream. It ships nothing user-visible; it answers one question.

**Files:**
- Modify: `src/vklayer/spirv_inject.h` (vertex patch, near the existing `prevClip` emission)
- Modify: `src/shaders/taa.comp` (add `VIZ_AIRFRAME = 9`)
- Modify: `src/vklayer/taa.h` (push `crash.probe` state)

**Interfaces:**
- Consumes: `proj[0]`, `proj[5]` already in the push block; the existing `VIZ_*` convention in `taa.comp`.
- Produces: `crash.probe` live key. When 1, vertices whose reconstructed view-space position lies inside a box centred on the aircraft are marked, and `taa.viz=9` renders marked pixels white.

- [ ] **Step 1: Add the probe to the vertex patch**

In `src/vklayer/spirv_inject.h`, immediately after `idPosW` is extracted (search for `idPosW` — it is `OpCompositeExtract` component 3 of `idLoadedPos`), emit the view-space reconstruction. The three components are:

```
viewX = clip.x / proj[0]
viewY = clip.y / proj[5]
viewZ = -clip.w
```

`proj[0]` and `proj[5]` arrive in the push block. Write the resulting `vec3` into the spare varying already reserved next to `prevClip`, so no new Location is needed.

- [ ] **Step 2: Add the visualisation**

In `src/shaders/taa.comp`, after the existing `VIZ_WRITTEN` block (search `VIZ_WRITTEN`), add:

```glsl
const int VIZ_AIRFRAME = 9;   // white where a pixel reconstructed INSIDE the
                              // airframe box, black elsewhere. The probe for
                              // view-space classification: if this does not
                              // stay glued to the aeroplane at every attitude
                              // and in every view, the grid design cannot work.
```

and a block mirroring `VIZ_WRITTEN` that writes white when the airframe flag is set.

- [ ] **Step 3: Build and launch**

```
.\build.ps1
.\dev-run.cmd
```

- [ ] **Step 4: Verify in the sim — this is the gate**

Set in `%TEMP%\taa_live.ini`:

```
crash.probe=1
taa.viz=9
```

Check all four, because each exercises a different failure mode:

1. **External view, aircraft centred** — the white region must cover the airframe and nothing else.
2. **Roll to 90° and pitch up** — it must stay glued. Drift here means the airframe transform is being applied in the wrong order.
3. **Cockpit view** — must still mark the airframe. This is where `clip.w` is smallest and reconstruction is least accurate.
4. **Fly over dense scenery** — buildings passing near the aircraft must NOT be marked. Some false positives directly under the aircraft are acceptable; a building fifty metres away being marked is not.

Record the outcome in the commit message either way. **If the marking does not hold, stop and report — do not proceed to Task 7.** The remaining tasks assume this works.

- [ ] **Step 5: Commit**

```bash
git add src/vklayer/spirv_inject.h src/shaders/taa.comp src/vklayer/taa.h
git commit -m "View-space airframe probe: the gate for grid classification"
```

---

### Task 7: Shared struct v8 — the plugin/layer contract

**Files:**
- Modify: `src/share.h:81-82` (bump `TAA_VERSION` 7 → 8) and `struct TaaShare`

**Interfaces:**
- Consumes: nothing.
- Produces, appended to `TaaShare` (append only — never reorder, the reader validates `structSize`):
  - `int32_t crashActive;`
  - `float crashAircraftInv[16];` — view-space → airframe-local
  - `float crashGridMin[3]; float crashCell; int32_t crashNx, crashNy, crashNz;`
  - `int32_t crashPartCount;`
  - `float crashGroundHeight;`

- [ ] **Step 1: Bump the version and append the fields**

In `src/share.h`, change:

```c
#define TAA_VERSION     7      // 7: VRAM system state (zone, shaped budget, ...)
```

to:

```c
#define TAA_VERSION     8      // 8: crash destruction state (grid, transform)
```

Append the fields to the END of `struct TaaShare`. Appending rather than inserting matters: `structSize` is what makes a version mismatch fail cleanly instead of reading garbage.

- [ ] **Step 2: Verify both sides still build**

```
.\build.ps1
```

Expected: layer and plugin both compile.

- [ ] **Step 3: Verify the handshake at runtime**

```
.\dev-run.cmd
```

Expected in `%TEMP%\taa_layer.txt`: no `structSize` or version mismatch warning, and the panel still reports the layer attached. A mismatch here means one side was rebuilt and the other was not.

- [ ] **Step 4: Commit**

```bash
git add src/share.h
git commit -m "Share v8: crash destruction state"
```

---

### Task 8: Storage buffer and descriptor set

The most invasive task. The existing layer only *appends* to pipeline layouts and patches shader code; this adds a descriptor set that must be bound before every draw.

**Files:**
- Modify: `src/vklayer/layer.cpp` (pipeline layout creation, command buffer hooks)
- Create: `src/vklayer/destruct_gpu.h` (buffer ownership, set layout, upload)

**Interfaces:**
- Consumes: `TaaShare` v8 from Task 7.
- Produces:
  - `destructgpu::ensure(VkDevice, uint32_t partCount)` — creates or resizes the fragment transform buffer.
  - `destructgpu::upload(const float *xforms, uint32_t count)` — writes translations for the current frame.
  - `destructgpu::setLayout()` — the `VkDescriptorSetLayout` appended to every patched pipeline layout.
  - `destructgpu::bind(VkCommandBuffer, VkPipelineLayout)` — binds the set.

- [ ] **Step 1: Create the buffer and set layout**

2000 fragments × 3 floats is 24 KB — a single host-visible, persistently mapped buffer, matching how the VRAM system already maps its readback buffer.

- [ ] **Step 2: Append the set layout to patched pipeline layouts**

Where the layer already extends `VkPipelineLayoutCreateInfo` with a push constant range, also append the set layout. Count and trace how many layouts were extended and how many were skipped — a skipped layout means those draws cannot displace, and per the Global Constraints a silent skip is not acceptable.

- [ ] **Step 3: Bind the set before draws**

Hook `vkCmdBindPipeline` and bind the destruction set at the appended index for pipelines whose layout was extended.

- [ ] **Step 4: Verify nothing regressed**

```
.\build.ps1
.\dev-run.cmd validate
```

Expected: `%TEMP%\mv_validation.txt` contains no new errors, TAA still resolves, and the sim runs for two minutes without `DEVICE_LOST`. Validation matters here more than anywhere else in the plan — `learnings.md` records that four hours of bisecting failed to find a device bug that one instrumented run named immediately.

- [ ] **Step 5: Commit**

```bash
git add src/vklayer/destruct_gpu.h src/vklayer/layer.cpp
git commit -m "Fragment transform buffer and descriptor set"
```

---

### Task 9: Occupancy discovery

**Files:**
- Modify: `src/vklayer/spirv_inject.h` (vertex patch writes its cell bit)
- Modify: `src/vklayer/destruct_gpu.h` (readback)

**Interfaces:**
- Consumes: the buffer from Task 8, `gridClassify` semantics from Task 2.
- Produces: `destructgpu::readOccupancy(unsigned char *out, uint32_t cells)`.

- [ ] **Step 1: Emit the occupancy write**

When `crash.discover` is 1, the vertex patch computes its cell index with **exactly** the arithmetic in `gridClassify` and sets the corresponding byte. Any divergence between the two makes the CPU constraint graph describe different cells from the ones the GPU displaces.

- [ ] **Step 2: Read it back after one frame**

- [ ] **Step 3: Verify the shape is plausible**

```
.\dev-run.cmd
```

Set `crash.discover=1`, then check `%TEMP%\taa_layer.txt` for the occupancy summary. Expected for an airliner: **5–15% of cells occupied**, giving 800–2000 fragments. Below 2% means classification is missing the aircraft; above 40% means the box or transform is wrong and it is catching the world.

- [ ] **Step 4: Commit**

```bash
git add src/vklayer/spirv_inject.h src/vklayer/destruct_gpu.h
git commit -m "Discover airframe occupancy from the geometry itself"
```

---

### Task 10: Displacement

**Files:**
- Modify: `src/vklayer/spirv_inject.h`

**Interfaces:**
- Consumes: Tasks 6, 8, 9.
- Produces: displaced vertices when `crashActive` is set.

- [ ] **Step 1: Apply the fragment translation**

After classification, read the fragment's translation from the storage buffer and add it to the view-space position, then re-project to clip space:

```
clip.x = view.x * proj[0]
clip.y = view.y * proj[5]
clip.w = -view.z
```

`clip.z` must be recomputed from the projection so depth stays consistent, or fragments will z-fight with the world.

- [ ] **Step 2: Gate it**

The test must be a uniform branch on `crashActive`, not a per-vertex buffer read, so normal flight pays nothing.

- [ ] **Step 3: Verify with a static offset**

Set `crash.test_offset=5` and confirm the whole airframe moves five metres while the world stays put, in both cockpit and external views, with shadows following.

- [ ] **Step 4: Commit**

```bash
git add src/vklayer/spirv_inject.h
git commit -m "Displace airframe vertices by fragment transform"
```

---

### Task 11: Trigger, freeze and the physics tick

**Files:**
- Modify: `src/plugin.cpp`
- Create: `src/destruct/sim.h`

**Interfaces:**
- Consumes: Tasks 2–5.
- Produces:
  - `destruct::Sim` owning fragments, links, broken flags and config.
  - `void simStart(Sim &, const Grid &, const Occupancy &, const float aircraftVel[3], const float impact[3])`
  - `void simStep(Sim &, float dt, const GroundPlane &)`

- [ ] **Step 1: Detect the crash**

Watch `sim/flightmodel2/misc/has_crashed`, plus a `sim/flightmodel/forces/g_nrml` spike as a fallback. On fire: snapshot the aircraft transform, probe terrain once with `XPLMProbeTerrainXYZ`, seed fragments at their rest positions with the airframe's velocity plus an impulse radiating from the impact point.

- [ ] **Step 2: Freeze the airframe**

Set `sim/operation/override/override_planepath` to 1 so the wreckage settles from a fixed point.

**Clear it on every exit path** — plugin disable, flight reset, crash reset, and `crash.enable=0`. This dataref locked the camera during the motion-vector work; it is easy to get stuck in and hard to diagnose from the symptom.

- [ ] **Step 3: Step the simulation**

In the flight loop: `integrate`, `solveLinks`, `clampToGround`, then publish translations to shared memory. Use a fixed 1/60 s step with an accumulator so behaviour does not change with frame rate.

- [ ] **Step 4: Verify end to end**

```
.\dev-run.cmd
```

Fly into terrain. Expected: the airframe fragments, pieces separate progressively rather than all at once, and the wreckage rests **on** the ground with nothing sunk through it. Confirm the camera is still free after a reset — that is the `override_planepath` check.

- [ ] **Step 5: Commit**

```bash
git add src/plugin.cpp src/destruct/sim.h
git commit -m "Crash trigger, airframe freeze and the fragment simulation"
```

---

### Task 12: Tuning knobs and documentation

**Files:**
- Modify: `config/taa_live.ini`
- Modify: `README.md`
- Modify: `packaging/READ ME FIRST.txt`

**Interfaces:**
- Consumes: everything above.
- Produces: the `crash.*` keys, documented.

- [ ] **Step 1: Add the keys with shipped values**

```ini
# ---- crash destruction
crash.enable=1
crash.cell_m=1.2          # target fragment size in metres; count follows aircraft size
crash.iterations=10       # constraint solver passes per step
crash.stiffness=0.8       # 0..1, fraction of joint error corrected per pass
crash.break_strain=0.35   # |len-rest|/rest above which a joint fails
crash.gravity=9.81
crash.damping=0.4
crash.restitution=0.0     # wreckage does not bounce
crash.friction=0.8
```

- [ ] **Step 2: Mirror every default in code**

Every value above must be the compiled default too. `%TEMP%\taa_live.ini` does not exist on a user's machine, so whatever is compiled in **is** the shipping configuration — this is the failure that shipped `varclip=1.25` in 0.0.17 and the `!velArmed` default before it.

- [ ] **Step 3: Document it**

Add a Crash Destruction section to `README.md` and a short note to the packaged README.

- [ ] **Step 4: Verify a clean install behaves the same**

Move `%TEMP%\taa_live.ini` aside, launch, and confirm the behaviour is identical to the tuned configuration.

- [ ] **Step 5: Commit**

```bash
git add config/taa_live.ini README.md "packaging/READ ME FIRST.txt"
git commit -m "Crash destruction: live knobs, matching compiled defaults, docs"
```

---

## Self-Review

**Spec coverage.** View-space classification → Task 6, 10. Grid and fragment sizing → Task 2. Occupancy discovery → Task 9. Constraints and breakage → Tasks 3, 5. Ground → Tasks 4, 11. Trigger and freeze → Task 11. Storage buffer → Task 8. Scale-invariant cell size → Task 2 (`gridForBounds`). Scope exclusions need no tasks.

**Gap found and accepted:** the spec mentions jittering cell boundaries so cuts are not axis-aligned. No task implements it — it is a polish item that only matters once the system works, and adding it before Task 10 is verified would be tuning something not yet observed. Add it after Task 11 if the cuts read as too regular.

**Placeholder scan.** No TBDs. Tasks 8, 9 and 10 give integration points and acceptance criteria rather than complete SPIR-V emission code, because the emission must match surrounding code the implementer will be reading; the pure-maths tasks are fully specified and testable.

**Type consistency.** `Grid`, `Occupancy`, `Link`, `Fragment`, `GroundPlane`, `SolverCfg` are used with identical names and members throughout. `gridClassify` is defined in Task 2 and referenced by Tasks 9 and 10 with the same semantics. `Fragment` carries no orientation in Task 4 and no later task assumes one.
