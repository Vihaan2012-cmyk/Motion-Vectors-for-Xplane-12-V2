// Prove the occupancy code is actually EMITTED, not merely that patching a
// module still validates.
//
// A corpus run that passes identically with the feature on and off is exactly
// as consistent with "the feature works" as with "the feature never ran". This
// project has already been burned twice by a measurement that could not
// distinguish those - the availability array that was never written, and the
// crash gate that could never open. So: patch one real module both ways and
// compare the words.
//
//   g++ -O2 -std=c++17 -o emit_check.exe emit_check.cpp -I<src> -static

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>

static void trace(const char *, ...) {}

#include "vklayer/spirv_inject.h"

static std::vector<uint32_t> readSpv(const char *path)
{
    std::ifstream f(path, std::ios::binary);
    std::vector<uint32_t> w;
    if (!f) return w;
    f.seekg(0, std::ios::end);
    const size_t n = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    w.resize(n / 4);
    f.read((char *)w.data(), (std::streamsize)(w.size() * 4));
    return w;
}

int main(int argc, char **argv)
{
    if (argc < 2) { printf("usage: emit_check <module.spv>\n"); return 2; }
    std::vector<uint32_t> in = readSpv(argv[1]);
    if (in.size() < 5) { printf("could not read %s\n", argv[1]); return 2; }

    std::vector<uint32_t> off, on;
    uint32_t l1 = 0, l2 = 0;
    const spvinj::Result r1 = spvinj::inject(in.data(), in.size() * 4, off, &l1, -1);
    const uint64_t emittedBefore = spvinj::occupancyVsCount();
    const spvinj::Result r2 = spvinj::inject(in.data(), in.size() * 4, on,  &l2, 7);
    const uint64_t emittedAfter = spvinj::occupancyVsCount();

    printf("module            %s\n", argv[1]);
    printf("inject off        result %d, %zu words\n", (int)r1, off.size());
    printf("inject on         result %d, %zu words\n", (int)r2, on.size());
    printf("words added       %zd\n", (ptrdiff_t)on.size() - (ptrdiff_t)off.size());
    printf("emit counter      %llu -> %llu\n",
           (unsigned long long)emittedBefore, (unsigned long long)emittedAfter);

    // The decoration that proves a storage buffer was declared at our set.
    bool sawDescriptorSet7 = false, sawBinding0 = false, sawStorageBufferPtr = false;
    size_t i = 5;
    while (i < on.size()) {
        const uint16_t op  = (uint16_t)(on[i] & 0xFFFF);
        const uint16_t len = (uint16_t)(on[i] >> 16);
        if (!len) break;
        if (op == spvinj::OpDecorate && len >= 4) {
            if (on[i + 2] == spvinj::Deco_DescriptorSet && on[i + 3] == 7) sawDescriptorSet7 = true;
            if (on[i + 2] == spvinj::Deco_Binding && on[i + 3] == 0)       sawBinding0 = true;
        }
        if (op == spvinj::OpTypePointer && len >= 4 &&
            on[i + 2] == spvinj::SC_StorageBuffer) sawStorageBufferPtr = true;
        i += len;
    }

    printf("DescriptorSet 7   %s\n", sawDescriptorSet7 ? "present" : "ABSENT");
    printf("Binding 0         %s\n", sawBinding0 ? "present" : "ABSENT");
    printf("StorageBuffer ptr %s\n", sawStorageBufferPtr ? "present" : "ABSENT");

    const bool ok = (emittedAfter == emittedBefore + 1) &&
                    on.size() > off.size() &&
                    sawDescriptorSet7 && sawBinding0 && sawStorageBufferPtr;
    printf("\n%s\n", ok ? "EMISSION CONFIRMED" : "EMISSION DID NOT HAPPEN");
    return ok ? 0 : 1;
}
