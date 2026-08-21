// Build the airframe from a real .acf and print the box, so the generator can
// be checked against an aeroplane whose dimensions are public knowledge.
//
//   g++ -std=c++11 -O2 -o build/acf_check.exe tools/acf_check.cpp
//   build/acf_check.exe "<path to .acf>" [expected span] [length] [height]
//
// The unit tests cover the parser with a synthetic file that runs anywhere.
// This covers the thing the unit tests cannot: that the rules read off one
// aeroplane's numbers reproduce that aeroplane.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <stdio.h>
#include <stdlib.h>
#include "../src/destruct/acf_planform.h"

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: acf_check <file.acf> [span] [length] [height]\n");
        return 2;
    }

    destruct::Airframe a;
    if (!destruct::parseAcf(argv[1], a)) {
        fprintf(stderr, "acf_check: could not parse or no flying surfaces: %s\n",
                argv[1]);
        return 1;
    }

    printf("wings %d, body points %d, gear %d\n",
           (int)a.wings.size(), (int)(a.bodyXyz.size() / 3), (int)a.gear.size());

    printf("\n%-5s %9s %9s %9s %8s %8s %8s %7s %7s\n",
           "seg", "root x", "root y", "root z", "croot", "ctip", "semilen",
           "sweep", "dihed");
    for (size_t i = 0; i < a.wings.size(); ++i) {
        const destruct::WingSeg &w = a.wings[i];
        float c[4][3];
        destruct::wingCorners(w, c);
        printf("%-5d %9.2f %9.2f %9.2f %8.2f %8.2f %8.2f %7.1f %7.1f   tip x %8.2f\n",
               w.index, w.rootX, w.rootY, w.rootZ, w.croot, w.ctip, w.semilen,
               w.sweepDeg, w.dihedDeg, c[2][0]);
    }

    std::vector<float> v;
    destruct::airframeVertices(a, 8, 4, v);
    float lo[3], hi[3];
    if (!destruct::vertexBounds(v, lo, hi)) {
        fprintf(stderr, "acf_check: no vertices generated\n");
        return 1;
    }

    const float span = hi[0] - lo[0];
    const float length = hi[2] - lo[2];

    // Published aircraft heights are measured from the ground, so compare
    // like with like: the box is about the design datum, and the gear is the
    // difference. Falls back to the datum-relative height when there is no
    // gear to stand on.
    float contact = 0.0f;
    const bool onWheels = destruct::groundContact(a, contact);
    const float height = onWheels ? (hi[1] - contact) : (hi[1] - lo[1]);
    if (onWheels)
        printf("gear contact %.2f m below datum; height on wheels %.2f m "
               "(datum-relative box is %.2f m)\n",
               -contact, height, hi[1] - lo[1]);

    printf("\n%d vertices\n", (int)(v.size() / 3));
    printf("box  x %8.2f .. %8.2f\n     y %8.2f .. %8.2f\n     z %8.2f .. %8.2f\n",
           lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
    printf("span %.2f  height %.2f  length %.2f  (metres, .acf frame)\n",
           span, height, length);

    // Wings only, which is the number the OBJ files could not produce.
    std::vector<float> wv;
    for (size_t i = 0; i < a.wings.size(); ++i)
        destruct::wingVertices(a.wings[i], 8, 4, wv);
    float wlo[3], whi[3];
    if (destruct::vertexBounds(wv, wlo, whi))
        printf("flying surfaces alone: span %.2f, top of fin %.2f\n",
               whi[0] - wlo[0], whi[1]);

    int rc = 0;
    if (argc >= 5) {
        struct { const char *name; float got, want; } cmp[3] = {
            { "span",   span,   (float)atof(argv[2]) },
            { "length", length, (float)atof(argv[3]) },
            { "height", height, (float)atof(argv[4]) },
        };
        printf("\n%-8s %8s %8s %8s\n", "", "got", "expected", "error");
        for (int i = 0; i < 3; ++i) {
            const float err = cmp[i].want != 0.0f
                ? 100.0f * (cmp[i].got - cmp[i].want) / cmp[i].want : 0.0f;
            const float mag = err < 0.0f ? -err : err;
            printf("%-8s %8.2f %8.2f %7.1f%%  %s\n", cmp[i].name, cmp[i].got,
                   cmp[i].want, err, mag <= 5.0f ? "ok" : "OUT");
            if (mag > 5.0f) rc = 1;
        }
    }
    return rc;
}
