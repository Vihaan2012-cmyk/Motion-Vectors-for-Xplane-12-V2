// Run the injector over X-Plane's ENTIRE shader corpus and validate every
// result, without launching the sim.
//
// The header's opening comment records that the injector was originally
// developed offline against fifteen dumped vertex shaders. There are now 6855
// modules on disk, extracted from the shipping binary, so the same method scales
// to the whole population: patch every one, run spirv-val over the output, and
// report what breaks.
//
// This exists because of light_vis. The "first store to gl_Position" assumption
// held for 488 of 494 vertex shaders and was wrong in exactly one family, which
// no amount of flying would have surfaced - light_vis is a debug overlay, so the
// wrong values it produced would never appear on screen while still being wrong.
// A test that reads every shader finds that; a test that reads the shaders you
// happen to hit does not.
//
//   build: g++ -O2 -std=c++17 -o inject_test.exe tools/inject_test.cpp
//   run:   inject_test.exe "D:\shaders\spv" [spirv-val path]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <fstream>
#include <algorithm>

// The injector emits trace() only from environment-gated dump paths. Give it
// somewhere to go so the header can be included as-is: patching a copy of the
// real thing would be testing the copy.
static void trace(const char *fmt, ...) { (void)fmt; }

#include "../src/vklayer/spirv_inject.h"

namespace fs = std::filesystem;

static bool readSpv(const fs::path &p, std::vector<uint32_t> &out)
{
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return false;
    std::streamsize n = f.tellg();
    if (n <= 0 || (n % 4)) return false;
    f.seekg(0);
    out.resize((size_t)n / 4);
    return (bool)f.read((char *)out.data(), n);
}

static bool writeSpv(const fs::path &p, const std::vector<uint32_t> &w)
{
    std::ofstream f(p, std::ios::binary);
    if (!f) return false;
    f.write((const char *)w.data(), (std::streamsize)w.size() * 4);
    return (bool)f;
}

// OpEntryPoint's execution model, so vertex modules can be told from the rest
// without relying on the family name.
static int entryModel(const std::vector<uint32_t> &w)
{
    if (w.size() < 5 || w[0] != 0x07230203u) return -1;
    size_t i = 5;
    while (i < w.size()) {
        uint16_t op  = (uint16_t)(w[i] & 0xFFFF);
        uint16_t len = (uint16_t)(w[i] >> 16);
        if (!len || i + len > w.size()) break;
        if (op == 15 /* OpEntryPoint */ && len >= 3) return (int)w[i + 1];
        i += len;
    }
    return -1;
}

// Count stores to gl_Position, so the report can name the shaders that made the
// old placement wrong rather than merely asserting they exist.
static size_t countPositionStores(const std::vector<uint32_t> &w)
{
    uint32_t perVertex = 0, directPos = 0, const0 = 0, intType = 0;
    std::vector<uint32_t> chains;
    size_t i = 5;
    while (i < w.size()) {
        uint16_t op  = (uint16_t)(w[i] & 0xFFFF);
        uint16_t len = (uint16_t)(w[i] >> 16);
        if (!len || i + len > w.size()) break;
        const uint32_t *p = &w[i];
        if (op == 72 /* OpMemberDecorate */ && len >= 5 && p[3] == 11 && p[4] == 0)
            perVertex = p[1];
        if (op == 71 /* OpDecorate */ && len >= 4 && p[2] == 11 && p[3] == 0)
            directPos = p[1];
        // MUST be an INTEGER zero. The first version matched any OpConstant
        // whose value word was 0, so it latched onto a float 0.0 declared later
        // and then failed to match a single gl_Position access chain - which
        // made light_vis look clean when it has six of them. An access chain
        // into a struct is indexed by an integer constant; nothing else counts.
        if (op == 21 /* OpTypeInt */ && len >= 4) intType = p[1];
        if (op == 43 /* OpConstant */ && len >= 4 && p[1] == intType && p[3] == 0)
            const0 = p[2];
        i += len;
    }
    // Resolve the gl_PerVertex STRUCT id to its VARIABLE id.
    uint32_t perVertexVar = 0;
    if (perVertex) {
        uint32_t ptr = 0;
        i = 5;
        while (i < w.size()) {
            uint16_t op = (uint16_t)(w[i] & 0xFFFF), len = (uint16_t)(w[i] >> 16);
            if (!len || i + len > w.size()) break;
            const uint32_t *p = &w[i];
            if (op == 32 /* OpTypePointer */ && len >= 4 && p[3] == perVertex) ptr = p[1];
            if (op == 59 /* OpVariable */ && len >= 4 && ptr && p[1] == ptr) perVertexVar = p[2];
            i += len;
        }
    }
    i = 5;
    while (i < w.size()) {
        uint16_t op = (uint16_t)(w[i] & 0xFFFF), len = (uint16_t)(w[i] >> 16);
        if (!len || i + len > w.size()) break;
        const uint32_t *p = &w[i];
        if (op == 65 /* OpAccessChain */ && len >= 5 && perVertexVar &&
            p[3] == perVertexVar && p[4] == const0)
            chains.push_back(p[2]);
        i += len;
    }
    size_t n = 0;
    i = 5;
    while (i < w.size()) {
        uint16_t op = (uint16_t)(w[i] & 0xFFFF), len = (uint16_t)(w[i] >> 16);
        if (!len || i + len > w.size()) break;
        const uint32_t *p = &w[i];
        if (op == 62 /* OpStore */ && len >= 3) {
            if (directPos && p[1] == directPos) ++n;
            for (size_t c = 0; c < chains.size(); ++c)
                if (p[1] == chains[c]) { ++n; break; }
        }
        i += len;
    }
    return n;
}

static size_t countReturns(const std::vector<uint32_t> &w)
{
    size_t n = 0, i = 5;
    while (i < w.size()) {
        uint16_t op = (uint16_t)(w[i] & 0xFFFF), len = (uint16_t)(w[i] >> 16);
        if (!len || i + len > w.size()) break;
        if (op == 253 /* OpReturn */) ++n;
        i += len;
    }
    return n;
}

int main(int argc, char **argv)
{
    const char *root = argc > 1 ? argv[1] : "D:\\shaders\\spv";
    const char *val  = argc > 2 ? argv[2]
                                : "C:\\VulkanSDK\\1.4.357.0\\Bin\\spirv-val.exe";

    // The device this is checked against. Real desktop GPUs report 128
    // components; hard-coding it here keeps the test independent of whatever
    // card happens to be present.
    spvinj::chooseLocations(128, 128);
    spvinj::chooseAttachment(8);
    printf("varyings at Location %u/%u, %u device locations, safe=%s\n\n",
           spvinj::currClipLocation(), spvinj::prevClipLocation(),
           spvinj::deviceLocationCount(),
           spvinj::locationsAreSafe() ? "yes" : "NO");

    fs::path tmp = fs::temp_directory_path() / "mv_inject_test";
    fs::create_directories(tmp);

    size_t nTotal = 0, nVertex = 0, nPatched = 0, nValFail = 0;
    size_t nFragment = 0, nFragPatched = 0, nFragValFail = 0;
    std::map<int, size_t> reasons;
    std::vector<std::string> failures, multiStore;

    std::vector<fs::path> files;
    for (auto &e : fs::recursive_directory_iterator(root))
        if (e.is_regular_file() && e.path().extension() == ".spv")
            files.push_back(e.path());
    std::sort(files.begin(), files.end());

    for (const fs::path &p : files) {
        std::vector<uint32_t> in;
        if (!readSpv(p, in)) continue;
        ++nTotal;
        const int model = entryModel(in);
        // ---- FRAGMENT MODULES GO THROUGH THE FRAGMENT PATCHER.
        //
        // The vertex patcher had corpus validation from the day the corpus
        // existed; the fragment patcher - the half that writes the velocity,
        // relocates colliding outputs, and now emits the transparency gate -
        // never did. Same method: patch every one, validate every result.
        // The attachment index mimics the layer: one past the highest output
        // Location the module already uses.
        if (model == 4 /* Fragment */) {
            ++nFragment;
            uint32_t maxLoc = 0;
            {
                size_t i2 = 5;
                while (i2 < in.size()) {
                    uint16_t op = (uint16_t)(in[i2] & 0xFFFF), ln = (uint16_t)(in[i2] >> 16);
                    if (!ln || i2 + ln > in.size()) break;
                    if (op == 71 /* OpDecorate */ && ln >= 4 && in[i2+2] == 30 /* Location */
                        && in[i2+3] < 16 && in[i2+3] + 1 > maxLoc)
                        maxLoc = in[i2+3] + 1;
                    i2 += ln;
                }
            }
            std::vector<uint32_t> fout;
            spvinj::Result fr = spvinj::injectFragment(in.data(), in.size() * 4,
                                                       fout, maxLoc);
            if (fr != spvinj::INJ_OK) continue;
            ++nFragPatched;
            fs::path o = tmp / ("f_" + p.filename().string());
            if (!writeSpv(o, fout)) continue;
            std::string cmd = std::string("\"\"") + val + "\" \"" +
                              o.string() + "\" 2>&1\"";
            FILE *pipe = _popen(cmd.c_str(), "r");
            std::string msg;
            if (pipe) {
                char buf[512];
                while (fgets(buf, sizeof(buf), pipe)) msg += buf;
                int rc = _pclose(pipe);
                if (rc != 0) {
                    ++nFragValFail;
                    if (failures.size() < 40)
                        failures.push_back("FRAG " + p.filename().string() + ": " + msg);
                }
            }
            fs::remove(o);
            continue;
        }
        if (model != 0 /* Vertex */) continue;
        ++nVertex;

        size_t stores = countPositionStores(in);
        size_t rets   = countReturns(in);
        if (stores > 1) {
            char b[256];
            snprintf(b, sizeof(b), "%s  (%zu stores, %zu returns)",
                     p.filename().string().c_str(), stores, rets);
            multiStore.push_back(b);
        }

        std::vector<uint32_t> out;
        uint32_t loc = 0;
        spvinj::Result r = spvinj::inject(in.data(), in.size() * 4, out, &loc);
        ++reasons[(int)r];
        if (r != spvinj::INJ_OK) continue;
        ++nPatched;

        fs::path o = tmp / p.filename();
        if (!writeSpv(o, out)) continue;
        // MV_KEEP names one module whose patched form is left on disk, so the
        // emitted code can be read rather than inferred from a pass/fail.
        if (const char *k = getenv("MV_KEEP"))
            if (p.filename().string().find(k) != std::string::npos)
                fs::copy_file(o, fs::path(".") / ("patched_" + p.filename().string()),
                              fs::copy_options::overwrite_existing);
        std::string cmd = std::string("\"\"") + val + "\" \"" +
                          o.string() + "\" 2>&1\"";
        FILE *pipe = _popen(cmd.c_str(), "r");
        std::string msg;
        if (pipe) {
            char buf[512];
            while (fgets(buf, sizeof(buf), pipe)) msg += buf;
            int rc = _pclose(pipe);
            if (rc != 0) {
                ++nValFail;
                if (failures.size() < 40)
                    failures.push_back(p.filename().string() + ": " + msg);
            }
        }
        fs::remove(o);
    }

    static const char *kName[] = { "patched", "not-a-vertex-shader",
                                   "no-gl_Position", "never-writes-gl_Position",
                                   "LOCATION-TAKEN", "malformed" };
    printf("modules scanned      %zu\n", nTotal);
    printf("vertex modules       %zu\n", nVertex);
    printf("patched              %zu\n", nPatched);
    printf("spirv-val FAILURES   %zu\n", nValFail);
    printf("fragment modules     %zu\n", nFragment);
    printf("frag patched         %zu\n", nFragPatched);
    printf("frag val FAILURES    %zu\n\n", nFragValFail);
    for (auto &kv : reasons)
        if (kv.first >= 0 && kv.first <= 5)
            printf("  %-26s %zu\n", kName[kv.first], kv.second);

    printf("\nmulti-store gl_Position (%zu) - these are what the first-store\n"
           "assumption got wrong; all must still validate:\n",
           multiStore.size());
    for (auto &s : multiStore) printf("  %s\n", s.c_str());

    printf("\nmodules with >1 OpReturn (fell back to the store splice): %llu\n",
           (unsigned long long)spvinj::multiReturnModules());

    if (!failures.empty()) {
        printf("\n---- VALIDATION FAILURES ----\n");
        for (auto &s : failures) printf("%s\n", s.c_str());
    }
    return (nValFail || nFragValFail) ? 1 : 0;
}
