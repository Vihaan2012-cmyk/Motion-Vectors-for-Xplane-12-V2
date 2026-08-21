// The airframe, built from the .acf the way X-Plane builds it.
//
// WHY THIS EXISTS
//
// The crash system needs the aeroplane's geometry: where the wings are, how
// far they reach, what volume to fragment. Three sources were tried.
//
//   THE OBJ FILES. Parsing all 70 exterior objects of a 747-200F gives a box
//   22.5 m wide. The aeroplane is 59.6 m wide. No object in the folder exceeds
//   22.5 m in any direction, the vertex count matches POINT_COUNTS exactly so
//   nothing is being missed by the parser, and the attached-object list places
//   every one of them at the same offset with no wing attachment. The outer
//   wing is not in the OBJ files.
//
//   acf_size_x AND acf_size_z. Two floats, and DataRefs.txt describes them as
//   "Shadow size, and viewing distance size" - they are radii for the shadow
//   footprint and the LOD sphere. Useful as a bracket, which is what bounds.h
//   uses them for, and nothing more. There is no height among them at all.
//
//   THE .acf PLANFORM. This file. It gives a half-span of 29.63 m against the
//   747-200F's real 29.82, and a nose-to-tail of 70.99 m against a real 70.66.
//   It is also what X-Plane itself believes: acf_size_x reads 30.0, which is
//   the planform half-span and not the OBJ's 11.27.
//
// HOW X-PLANE LAYS OUT A WING SEGMENT
//
// Read off the file's own numbers rather than assumed. Each segment has a root
// at (_part_x, _part_y, _part_z), a _semilen_SEG, a _sweep_design and a
// _dihed_design. The next segment outboard states its own _part_x, so the rule
// can be checked instead of guessed:
//
//     next_root_x = part_x + semilen * cos(sweep) * cos(dihedral)
//
//   segment 2 -> 4:  9.803 + 32.840 = 42.643   file says 42.659
//   segment 4 -> 6: 42.659 + 25.310 = 67.969   file says 67.973
//
// Two independent segments to four significant figures. So _semilen_SEG is
// measured ALONG the swept and dihedralled quarter-chord line, not across the
// span, and the same three lines place a fin as well as a wing: a fin is a
// segment with a dihedral of 90 degrees, which drives the span vector into +y
// with no special case.
//
// UNITS AND FRAME
//
// The .acf is in FEET throughout. Its origin is not X-Plane's render origin -
// on the 747-200F they differ by 5.18 m along z. That offset is NOT assumed
// here: the gear appears in both frames, as _gear_x/y/z in the .acf and as
// acf_Xarm/Yarm/Zarm from datarefs, so datumOffset() measures it from the
// aircraft in front of it and works for an airframe nobody has tested.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <stdint.h>
#include <string>
#include <vector>

namespace destruct {

static const float kFeetToMetres = 0.3048f;

// Plane Maker's fixed part-table sizes. Wings 0..55 and bodies 0..17 in the
// .acf numbering; the sim's own dataref arrays are 56 wide for the same
// reason, which is the cross-check that these are the real limits.
static const int kMaxAcfWings = 56;
static const int kMaxAcfBodies = 24;
static const int kMaxAcfGear = 10;

// ---- ONE FLYING SURFACE.
//
// Lengths in METRES, angles in degrees, already converted from the file.
struct WingSeg {
    int   index    = -1;
    float rootX    = 0.0f, rootY = 0.0f, rootZ = 0.0f;  // leading edge
    float croot    = 0.0f, ctip  = 0.0f;
    float semilen  = 0.0f;      // along the swept, dihedralled span
    float sweepDeg = 0.0f;
    float dihedDeg = 0.0f;
    float side     = 1.0f;      // +1 right, -1 left, from _is_right_mult
};

struct GearLeg {
    float x = 0.0f, y = 0.0f, z = 0.0f;   // metres, .acf frame
    float legLen = 0.0f, tyreRad = 0.0f;
};

struct Airframe {
    std::vector<WingSeg> wings;
    // Body loft points, metres, .acf frame, one side as stored in the file.
    // Mirroring is left to the caller because a body on the centreline must
    // not be mirrored into a duplicate of itself.
    std::vector<float>   bodyXyz;      // xyz triples
    std::vector<GearLeg> gear;
    bool ok = false;
};

// ---- THE PARSER.
//
// The .acf is line-oriented: "P <key> <value>". A 747's runs to 49105
// properties, so this matches the handful of prefixes it needs on the way past
// rather than building a map of all of them.
namespace acfdetail {

inline bool splitProp(const char *line, std::string &key, std::string &val)
{
    if (line[0] != 'P' || line[1] != ' ') return false;
    const char *k = line + 2;
    const char *sp = strchr(k, ' ');
    if (!sp) return false;
    key.assign(k, (size_t)(sp - k));
    const char *v = sp + 1;
    while (*v == ' ' || *v == '\t') ++v;
    const char *end = v + strlen(v);
    while (end > v && (end[-1] == '\n' || end[-1] == '\r' ||
                       end[-1] == ' '  || end[-1] == '\t')) --end;
    val.assign(v, (size_t)(end - v));
    return true;
}

// "_wing/12/_Croot" -> group "_wing", index 12, field "_Croot".
inline bool splitIndexed(const std::string &key, const char *group,
                         int &idx, std::string &field)
{
    const size_t glen = strlen(group);
    if (key.size() <= glen + 2) return false;
    if (key.compare(0, glen, group) != 0) return false;
    if (key[glen] != '/') return false;
    size_t p = glen + 1;
    if (p >= key.size() || key[p] < '0' || key[p] > '9') return false;
    int n = 0;
    while (p < key.size() && key[p] >= '0' && key[p] <= '9') {
        n = n * 10 + (key[p] - '0');
        ++p;
    }
    if (p >= key.size() || key[p] != '/') return false;
    idx = n;
    field.assign(key, p + 1, std::string::npos);
    return true;
}

} // namespace acfdetail

// Parse an .acf into wings, body loft points and gear. Returns false only when
// the file cannot be read or contains no usable flying surface - a distinction
// worth keeping, because "no wings" means the caller must fall back rather
// than fragment an aeroplane with none.
inline bool parseAcf(const char *path, Airframe &out)
{
    out = Airframe();
    FILE *fh = fopen(path, "rb");
    if (!fh) return false;

    struct RawWing {
        bool  seen = false;
        float px = 0, py = 0, pz = 0, croot = 0, ctip = 0;
        float semilen = 0, sweep = 0, dihed = 0, side = 1.0f;
    };
    std::vector<RawWing> raw((size_t)kMaxAcfWings);
    std::vector<GearLeg> gear((size_t)kMaxAcfGear);
    std::vector<bool>    gearSeen((size_t)kMaxAcfGear, false);

    // Body loft: "_body/<b>/_geo_xyz/<station>,<point>,<axis>".
    //
    // Keyed on (station, point) in a map rather than a preallocated array,
    // because the .acf is sorted ALPHABETICALLY and so "_geo_xyz/0,0,0" comes
    // before "_geo_xyz/i_count". Sizing the array from the counts and then
    // bounds-checking against them threw every point away in silence - the
    // fuselage vanished and only the wings survived, which reads exactly like
    // an aeroplane that has no fuselage rather than like a parser bug.
    //
    // A map needs no counts at all, so there is nothing left to order wrongly.
    struct Pt3 { float x = 0, y = 0, z = 0; };
    struct RawBody {
        std::map<uint64_t, Pt3> pts;   // (station << 32) | point
        float ox = 0, oy = 0, oz = 0;
    };
    std::vector<RawBody> bodies((size_t)kMaxAcfBodies);

    char line[1024];
    std::string key, val, field;
    while (fgets(line, sizeof(line), fh)) {
        if (!acfdetail::splitProp(line, key, val)) continue;
        int idx = 0;

        if (acfdetail::splitIndexed(key, "_wing", idx, field)) {
            if (idx < 0 || idx >= kMaxAcfWings) continue;
            RawWing &w = raw[(size_t)idx];
            const float f = (float)atof(val.c_str());
            if      (field == "_part_x")       { w.px = f; w.seen = true; }
            else if (field == "_part_y")       { w.py = f; w.seen = true; }
            else if (field == "_part_z")       { w.pz = f; w.seen = true; }
            else if (field == "_Croot")        { w.croot = f; w.seen = true; }
            else if (field == "_Ctip")         { w.ctip = f; w.seen = true; }
            else if (field == "_semilen_SEG")  { w.semilen = f; w.seen = true; }
            else if (field == "_sweep_design") { w.sweep = f; w.seen = true; }
            else if (field == "_dihed_design") { w.dihed = f; w.seen = true; }
            else if (field == "_is_right_mult"){ w.side = (f < 0.0f) ? -1.0f : 1.0f; }
            continue;
        }

        if (acfdetail::splitIndexed(key, "_gear", idx, field)) {
            if (idx < 0 || idx >= kMaxAcfGear) continue;
            GearLeg &g = gear[(size_t)idx];
            const float f = (float)atof(val.c_str());
            if      (field == "_gear_x")   { g.x = f; gearSeen[(size_t)idx] = true; }
            else if (field == "_gear_y")   { g.y = f; gearSeen[(size_t)idx] = true; }
            else if (field == "_gear_z")   { g.z = f; gearSeen[(size_t)idx] = true; }
            else if (field == "_leg_len")  { g.legLen = f; }
            else if (field == "_tire_radius") { g.tyreRad = f; }
            continue;
        }

        if (acfdetail::splitIndexed(key, "_body", idx, field)) {
            if (idx < 0 || idx >= kMaxAcfBodies) continue;
            RawBody &b = bodies[(size_t)idx];
            const float f = (float)atof(val.c_str());
            if      (field == "_part_x") { b.ox = f; continue; }
            else if (field == "_part_y") { b.oy = f; continue; }
            else if (field == "_part_z") { b.oz = f; continue; }
            if (field.compare(0, 9, "_geo_xyz/") != 0) continue;
            int s = 0, p = 0, a = 0;
            if (sscanf(field.c_str() + 9, "%d,%d,%d", &s, &p, &a) != 3) continue;
            if (a < 0 || a > 2 || s < 0 || p < 0) continue;
            Pt3 &pt = b.pts[((uint64_t)(uint32_t)s << 32) | (uint32_t)p];
            if      (a == 0) pt.x = f;
            else if (a == 1) pt.y = f;
            else             pt.z = f;
            continue;
        }
    }
    fclose(fh);

    // ---- Feet to metres, and drop everything that does not describe a shape.
    for (int i = 0; i < kMaxAcfWings; ++i) {
        const RawWing &w = raw[(size_t)i];
        // A zero semilen or root chord is an unused slot, and Plane Maker
        // leaves plenty of them. Fragmenting one would put a zero-area
        // surface at the origin.
        if (!w.seen || w.semilen <= 0.0f || w.croot <= 0.0f) continue;
        WingSeg s;
        s.index    = i;
        s.rootX    = w.px * kFeetToMetres;
        s.rootY    = w.py * kFeetToMetres;
        s.rootZ    = w.pz * kFeetToMetres;
        s.croot    = w.croot * kFeetToMetres;
        s.ctip     = w.ctip  * kFeetToMetres;
        s.semilen  = w.semilen * kFeetToMetres;
        s.sweepDeg = w.sweep;
        s.dihedDeg = w.dihed;
        s.side     = w.side;
        out.wings.push_back(s);
    }

    for (int b = 0; b < kMaxAcfBodies; ++b) {
        const RawBody &rb = bodies[(size_t)b];
        std::map<uint64_t, Pt3>::const_iterator it = rb.pts.begin();
        for (; it != rb.pts.end(); ++it) {
            out.bodyXyz.push_back((it->second.x + rb.ox) * kFeetToMetres);
            out.bodyXyz.push_back((it->second.y + rb.oy) * kFeetToMetres);
            out.bodyXyz.push_back((it->second.z + rb.oz) * kFeetToMetres);
        }
    }

    for (int i = 0; i < kMaxAcfGear; ++i) {
        if (!gearSeen[(size_t)i]) continue;
        GearLeg g = gear[(size_t)i];
        g.x *= kFeetToMetres; g.y *= kFeetToMetres; g.z *= kFeetToMetres;
        g.legLen *= kFeetToMetres; g.tyreRad *= kFeetToMetres;
        out.gear.push_back(g);
    }

    out.ok = !out.wings.empty();
    return out.ok;
}

// ---- WHERE THE SPAN GOES.
//
// The one piece of geometry everything else is built on, kept in a function so
// the rule verified against the file's own next-segment roots exists exactly
// once. A fin falls out of the same three lines with a dihedral of 90.
inline void spanVector(const WingSeg &w, float out[3])
{
    const float sw = w.sweepDeg * 3.14159265358979f / 180.0f;
    const float di = w.dihedDeg * 3.14159265358979f / 180.0f;
    const float cs = cosf(sw);
    out[0] = w.side * w.semilen * cs * cosf(di);
    out[1] =          w.semilen * cs * sinf(di);
    out[2] =          w.semilen * sinf(sw);
}

// The four planform corners: root LE, root TE, tip LE, tip TE. Chord runs aft
// in +z, and _part_z is the leading edge.
inline void wingCorners(const WingSeg &w, float out[4][3])
{
    float d[3];
    spanVector(w, d);
    out[0][0] = w.rootX;        out[0][1] = w.rootY;        out[0][2] = w.rootZ;
    out[1][0] = w.rootX;        out[1][1] = w.rootY;        out[1][2] = w.rootZ + w.croot;
    out[2][0] = w.rootX + d[0]; out[2][1] = w.rootY + d[1]; out[2][2] = w.rootZ + d[2];
    out[3][0] = w.rootX + d[0]; out[3][1] = w.rootY + d[1]; out[3][2] = w.rootZ + d[2] + w.ctip;
}

// Typical thickness-to-chord. Only used to give the slab a thickness so that
// fragments have volume rather than being a zero-depth sheet; the airfoil
// files carry the real number and nothing here needs that precision.
static const float kThicknessRatio = 0.12f;

// ---- VERTEX GENERATION.
//
// A spanwise x chordwise lattice over the planform, extruded to the local
// thickness. spanSteps and chordSteps are counts of CELLS, so the lattice has
// (spanSteps+1) x (chordSteps+1) points per face and twice that in total.
//
// The extrusion direction is the surface normal, computed rather than assumed
// to be +y: cross(chord, span) points along +y for a wing and along +x for a
// fin, which is what makes a vertical surface come out with thickness across
// it instead of along it.
inline void wingVertices(const WingSeg &w, int spanSteps, int chordSteps,
                         std::vector<float> &out)
{
    if (spanSteps < 1) spanSteps = 1;
    if (chordSteps < 1) chordSteps = 1;

    float d[3];
    spanVector(w, d);
    const float dlen = sqrtf(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
    if (dlen <= 1e-6f) return;
    const float sx = d[0]/dlen, sy = d[1]/dlen;

    // n = cross(chordDir, spanDir) with chordDir = (0,0,1).
    float nx = -sy, ny = sx, nz = 0.0f;
    const float nlen = sqrtf(nx*nx + ny*ny + nz*nz);
    if (nlen > 1e-6f) { nx /= nlen; ny /= nlen; nz /= nlen; }
    else { nx = 0.0f; ny = 1.0f; nz = 0.0f; }

    for (int i = 0; i <= spanSteps; ++i) {
        const float t = (float)i / (float)spanSteps;
        const float chord = w.croot + (w.ctip - w.croot) * t;
        const float half  = 0.5f * chord * kThicknessRatio;
        const float bx = w.rootX + d[0] * t;
        const float by = w.rootY + d[1] * t;
        const float bz = w.rootZ + d[2] * t;
        for (int j = 0; j <= chordSteps; ++j) {
            const float c = chord * (float)j / (float)chordSteps;
            for (int k = -1; k <= 1; k += 2) {
                const float o = half * (float)k;
                out.push_back(bx + nx * o);
                out.push_back(by + ny * o);
                out.push_back(bz + c + nz * o);
            }
        }
    }
}

// Every flying surface plus every body point, in .acf frame metres.
inline void airframeVertices(const Airframe &a, int spanSteps, int chordSteps,
                             std::vector<float> &out)
{
    for (size_t i = 0; i < a.wings.size(); ++i)
        wingVertices(a.wings[i], spanSteps, chordSteps, out);
    for (size_t i = 0; i + 2 < a.bodyXyz.size(); i += 3) {
        out.push_back(a.bodyXyz[i]);
        out.push_back(a.bodyXyz[i+1]);
        out.push_back(a.bodyXyz[i+2]);
        // Bodies store one side; the aeroplane has two. A centreline body
        // mirrors onto itself, which costs a duplicate point and no error.
        out.push_back(-a.bodyXyz[i]);
        out.push_back(a.bodyXyz[i+1]);
        out.push_back(a.bodyXyz[i+2]);
    }
}

// The lowest point the aeroplane touches the ground at, in the .acf frame.
//
// Worth having separately from the vertex box because published aircraft
// heights are measured from the GROUND, not from the design datum, and the
// gear is the whole difference between the two: a 747-200F's box is 17.67 m
// tall about its datum and 19.4 m tall on its wheels.
//
// Returns false when there is no gear, which is a floatplane or a glider on a
// skid rather than an error - the caller keeps the datum-relative box.
inline bool groundContact(const Airframe &a, float &outY)
{
    if (a.gear.empty()) return false;
    bool any = false;
    float lowest = 0.0f;
    for (size_t i = 0; i < a.gear.size(); ++i) {
        const GearLeg &g = a.gear[i];
        if (g.legLen <= 0.0f) continue;      // an unused slot, not a leg
        const float bottom = g.y - g.legLen - g.tyreRad;
        if (!any || bottom < lowest) { lowest = bottom; any = true; }
    }
    if (any) outY = lowest;
    return any;
}

inline bool vertexBounds(const std::vector<float> &v, float lo[3], float hi[3])
{
    if (v.size() < 3) return false;
    for (int a = 0; a < 3; ++a) { lo[a] = v[(size_t)a]; hi[a] = v[(size_t)a]; }
    for (size_t i = 0; i + 2 < v.size(); i += 3)
        for (int a = 0; a < 3; ++a) {
            const float c = v[i + (size_t)a];
            if (c < lo[a]) lo[a] = c;
            if (c > hi[a]) hi[a] = c;
        }
    return true;
}

// ---- THE .acf FRAME IS NOT X-PLANE'S RENDER FRAME.
//
// They differ by 5.18 m along z on the 747-200F. Rather than assume a
// convention, measure it: the gear exists in both frames, as _gear_x/y/z here
// and as acf_Xarm/Yarm/Zarm from datarefs. The mean difference over every leg
// is the offset, and averaging rather than taking one leg means a single
// mis-entered value moves the answer by a fraction rather than all of it.
//
// Returns false when there is nothing to match, so the caller can fall back
// instead of silently placing the airframe at the wrong station.
inline bool datumOffset(const Airframe &a, const float *armX, const float *armY,
                        const float *armZ, int nArm, float out[3])
{
    out[0] = out[1] = out[2] = 0.0f;
    if (!armX || !armY || !armZ) return false;
    const int n = (int)a.gear.size() < nArm ? (int)a.gear.size() : nArm;
    if (n <= 0) return false;
    double sx = 0.0, sy = 0.0, sz = 0.0;
    for (int i = 0; i < n; ++i) {
        sx += (double)armX[i] - (double)a.gear[(size_t)i].x;
        sy += (double)armY[i] - (double)a.gear[(size_t)i].y;
        sz += (double)armZ[i] - (double)a.gear[(size_t)i].z;
    }
    out[0] = (float)(sx / n);
    out[1] = (float)(sy / n);
    out[2] = (float)(sz / n);
    return true;
}

inline void applyOffset(std::vector<float> &v, const float off[3])
{
    for (size_t i = 0; i + 2 < v.size(); i += 3) {
        v[i]     += off[0];
        v[i + 1] += off[1];
        v[i + 2] += off[2];
    }
}

} // namespace destruct
