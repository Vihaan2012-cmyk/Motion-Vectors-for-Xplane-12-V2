// Build:  g++ -O2 -std=c++17 -I src -o build/test_fg_flags.exe src/test_fg_flags.cpp
// Run:    build/test_fg_flags.exe      (exit 0 = all pass)
//
// The flags word is the only channel from the layer into FFX's interpolation shader that
// needs no constant-buffer change, so its packing has to be exact: a shifted eps byte lands
// on FFX's own debug bits, and the shader decodes bits 24..31 with 0 meaning "default".
#include "vklayer/fg_flags.h"
#include <cstdio>
static int fails = 0;
static void check(bool ok, const char *what) { std::printf("%s %s\n", ok ? "ok  " : "FAIL", what); if (!ok) ++fails; }
int main()
{
    check(fgTrustFlags(false, false, 0.25f, false) == 0x08000000u, "eps 0.25 alone -> byte 8 in bits 24..31");
    check(fgTrustFlags(true,  true,  0.25f, false) == 0x08030000u, "trust + snap + 0.25 -> 0x08030000");
    check(fgTrustFlags(true,  false, 0.0f,  false) == 0x00010000u, "eps 0 -> byte 0 (shader default), trust only");
    check(fgTrustFlags(true,  true,  10.0f, true)  == 0xFF030004u, "eps clamps to 255, debug view = bit 2");
    check(fgTrustFlags(false, false, 0.03f, false) == 0x01000000u, "0.03 px rounds to 1/32");
    check(fgSnapEpsPx(0x08030000u) == 0.25f, "decode byte 8 -> 0.25");
    check(fgSnapEpsPx(0x00030000u) == 0.25f, "decode byte 0 -> default 0.25");
    check(fgSnapEpsPx(0x20000000u) == 1.0f,  "decode byte 32 -> 1.0");
    std::printf("%d failure(s)\n", fails);
    return fails;
}
