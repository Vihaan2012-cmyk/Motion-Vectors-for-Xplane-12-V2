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
    // ---- UNBUFFERED, BECAUSE THIS TOOL CRASHES ON THE WAY OUT.
    //
    // Something in the static teardown segfaults after main returns. With
    // buffered stdout that discards every line the tool printed, so a working
    // run and a broken one both showed nothing at all - and the empty output
    // was read as "inject() crashed", which sent the search into the injector
    // instead of into the exit path.
    //
    // The teardown fault is worth fixing on its own account; this makes the
    // tool's findings survive it in the meantime.
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) { printf("usage: emit_check <module.spv>\n"); return 2; }
    std::vector<uint32_t> in = readSpv(argv[1]);
    if (in.size() < 5) { printf("could not read %s\n", argv[1]); return 2; }

    // ---- REFUSE A MODULE THAT HAS ALREADY BEEN PATCHED.
    //
    // The dumps in %TEMP%\mvspv are POST-injection: they carry Location 28 and
    // 29 decorations this layer put there. Injecting into one of those tests
    // DOUBLE injection, and it passes - 59 of 60 emitted and validated - while
    // saying nothing whatever about the real patch.
    //
    // That is exactly the failure this tool was built to prevent, and it was
    // fed the wrong corpus for a whole evening. So the input is checked rather
    // than assumed: a module already decorated at our varying locations is a
    // corpus error, not a result.
    {
        // The signature is the PUSH CONSTANT BLOCK, not the varying location.
        //
        // Checking Location 14/15 - this tool's defaults, with no device to
        // ask - missed a corpus patched at 28/29, and let a whole evening of
        // validation run against DOUBLE injection. The locations depend on
        // the device; the marker must not.
        //
        // X-Plane declares no push constant ranges in any of the 6855
        // modules this layer has inspected, and the layer adds one to every
        // vertex module it patches. So a vertex module carrying a
        // PushConstant variable is one of ours, whatever locations that run
        // happened to use.
        uint32_t i2 = 5;
        bool alreadyPatched = false;
        while (i2 < in.size()) {
            const uint16_t op = (uint16_t)(in[i2] & 0xFFFF);
            const uint16_t ln = (uint16_t)(in[i2] >> 16);
            if (!ln) break;
            if (op == spvinj::OpVariable && ln >= 4 &&
                in[i2 + 3] == spvinj::SC_PushConstant)
                alreadyPatched = true;
            i2 += ln;
        }
        if (alreadyPatched) {
            printf("module            %s\n", argv[1]);
            printf("\nALREADY PATCHED - this module declares a push constant block.\n"
                   "X-Plane declares none in 6855 modules and this layer adds one to\n"
                   "every vertex module it patches, so this dump was taken AFTER\n"
                   "injection. Injecting again tests DOUBLE injection and proves\n"
                   "nothing about the real patch. Dump originals and re-run.\n");
            return 3;
        }
    }

    std::vector<uint32_t> off, on;
    uint32_t l1 = 0, l2 = 0;
    const spvinj::Result r1 = spvinj::inject(in.data(), in.size() * 4, off, &l1, -1);
    const uint64_t emittedBefore = spvinj::occupancyVsCount();
    const spvinj::Result r2 = spvinj::inject(in.data(), in.size() * 4, on,  &l2, 7);
    const uint64_t emittedAfter = spvinj::occupancyVsCount();

    printf("module            %s\n", argv[1]);

    // A refused injection returns an EMPTY vector, and every check below
    // indexes into it - on[3] for the id bound most obviously. Reading it
    // segfaulted the tool before it printed anything, so a module the injector
    // declined looked identical to a tool that was broken.
    if (on.size() < 5 || off.size() < 5) {
        printf("inject off        result %d, %zu words\n", (int)r1, off.size());
        printf("inject on         result %d, %zu words\n", (int)r2, on.size());
        printf("\nINJECTION REFUSED - nothing to check. This is a statement "
               "about the module, not about the occupancy code.\n");
        return 2;
    }
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

    // ---- THE ONLY CHECK THAT WOULD HAVE CAUGHT THE BUG.
    //
    // Everything above was true of a module that could not possibly work. The
    // occupancy instructions were appended to a vector the emitter had already
    // read, so the patched module declared the storage buffer, decorated it at
    // set 7 binding 0, listed it in the entry point - and never wrote to it.
    //
    // This tool said EMISSION CONFIRMED, spirv-val agreed, and the discard word
    // stayed at zero through 70999 draw-time binds while the search went
    // looking at descriptor binding.
    //
    // A declaration is not a write. Find the store, through an access chain
    // rooted at the storage buffer variable, or report nothing was emitted.
    uint32_t dsVar = 0;
    i = 5;
    while (i < on.size()) {            // the variable our decoration names
        const uint16_t op  = (uint16_t)(on[i] & 0xFFFF);
        const uint16_t len = (uint16_t)(on[i] >> 16);
        if (!len) break;
        if (op == spvinj::OpDecorate && len >= 4 &&
            on[i + 2] == spvinj::Deco_DescriptorSet && on[i + 3] == 7)
            dsVar = on[i + 1];
        i += len;
    }

    bool sawChain = false, sawStore = false;
    uint32_t chainId = 0;
    i = 5;
    while (i < on.size()) {
        const uint16_t op  = (uint16_t)(on[i] & 0xFFFF);
        const uint16_t len = (uint16_t)(on[i] >> 16);
        if (!len) break;
        if (op == spvinj::OpAccessChain && len >= 4 && dsVar && on[i + 3] == dsVar) {
            sawChain = true;
            chainId  = on[i + 2];
        }
        if (op == spvinj::OpStore && len >= 3 && chainId && on[i + 1] == chainId)
            sawStore = true;
        i += len;
    }

    printf("access chain      %s\n", sawChain ? "present" : "ABSENT");
    printf("STORE through it  %s\n", sawStore ? "present" : "ABSENT - nothing writes");

    // ---- WRITE THE PATCHED MODULE OUT FOR spirv-val.
    //
    // A bound check was tried here and removed. It scanned every operand of
    // every instruction and took each for an id, so an OpName's packed ASCII
    // read as id 7628147 and the tool reported PAST THE BOUND on a module that
    // was fine. Telling ids from literals needs per-opcode operand knowledge,
    // which is a SPIR-V grammar table - and spirv-val already has one.
    //
    // So the module is written next to its input and validated by the tool
    // whose job that is. A heuristic that produces false failures is worse than
    // no check: it costs exactly the time a real failure would.
    if (argc >= 3) {
        FILE *fo = fopen(argv[2], "wb");
        if (fo) {
            fwrite(on.data(), 4, on.size(), fo);
            fclose(fo);
            printf("patched module    written to %s (validate with spirv-val)\n",
                   argv[2]);
        } else {
            printf("patched module    could NOT be written to %s\n", argv[2]);
        }
    }

    const bool ok = (emittedAfter == emittedBefore + 1) &&
                    on.size() > off.size() &&
                    sawDescriptorSet7 && sawBinding0 && sawStorageBufferPtr &&
                    sawChain && sawStore;
    printf("\n%s\n", ok ? "EMISSION CONFIRMED - and it writes"
                        : "EMISSION INCOMPLETE");
    return ok ? 0 : 1;
}
