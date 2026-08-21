// Generate FidelityFX shader permutation headers, because AMD's generator
// cannot be built on this machine.
//
// ---- WHY THIS EXISTS AT ALL.
//
// The FidelityFX SDK ships its GPU code as GLSL and expects a build step to
// turn it into SPIR-V blob tables: for each pass, every combination of that
// pass's permutation options is compiled, deduplicated, reflected, and written
// out as a C header. The backend then does
//
//     const int32_t tableIndex = g_<pass>_IndirectionTable[key.index];
//     return POPULATE_SHADER_BLOB_FFX(g_<pass>_PermutationInfo, tableIndex);
//
// so without those headers nothing links and nothing runs.
//
// AMD's generator is sdk/tools/ffx_shader_compiler. Its CMakeLists contains
//
//     if(MSVC_TOOLSET_VERSION VERSION_LESS 142)
//         message(FATAL_ERROR "Cannot find MSVC toolset version 142 or greater...")
//
// and it links dxguid, agilitysdk and dxc even for the GLSL path. There is no
// Visual Studio on this machine and this layer is built with MinGW, so that
// tool is not available to us. The pieces it uses ARE: AMD bundles a matching
// glslangValidator.exe, and SPIRV-Reflect is vendored in its libs.
//
// So this reproduces the OUTPUT FORMAT exactly, using those same two pieces.
// The format is not invented here - it is transcribed from
// sdk/tools/ffx_shader_compiler/src/ffx_sc.cpp (the key union, the info struct,
// the indirection table) and src/glsl_compiler.cpp (the reflection members, the
// resource arrays, and the descriptor-type classification). Anything that
// disagrees with those two files is a bug here.
//
// ---- WHAT "DEDUPLICATED" MEANS AND WHY THE INDIRECTION TABLE EXISTS.
//
// Most permutation options change nothing for most passes: a pass that never
// reads depth is byte-identical whether INVERTED_DEPTH is 0 or 1. So the table
// of blobs holds only DISTINCT SPIR-V, and a separate indirection table maps
// every possible key to one of them. For an option a pass ignores, both keys
// land on the same entry.
//
// The indirection table must have 2^bits entries whether or not each is
// reachable, because the backend indexes it directly with the packed key.
//
// Copyright (C) 2026 MotionVectors contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Build:
//   g++ -O2 -std=c++17 -o ffx_permute.exe ffx_permute.cpp \
//       <sdk>/tools/ffx_shader_compiler/libs/SPIRV-Reflect/spirv_reflect.c \
//       -I<sdk>/tools/ffx_shader_compiler/libs/SPIRV-Reflect -static

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <fstream>

extern "C" {
#include "spirv_reflect.h"
}

// ---- ONE RESOURCE, AS THE HEADER NEEDS IT.
struct ResInfo {
    std::string name;
    uint32_t    binding = 0;
    uint32_t    count   = 0;
    uint32_t    space   = 0;   // descriptor set
};

// The seven buckets, in the order glsl_compiler.cpp writes them. The order is
// load-bearing: the PermutationInfo struct members and the initialiser are
// positional, so swapping two buckets silently binds the wrong resources.
struct Reflection {
    std::vector<ResInfo> cbv, srvTex, uavTex, srvBuf, uavBuf, sampler, rtas;
};

struct Permutation {
    uint32_t              key = 0;
    std::string           digest;
    std::vector<uint8_t>  spirv;
    Reflection            refl;
};

static std::vector<uint8_t> readFile(const std::string &p)
{
    std::ifstream f(p, std::ios::binary);
    std::vector<uint8_t> v;
    if (!f) return v;
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    if (n <= 0) return v;
    f.seekg(0, std::ios::beg);
    v.resize((size_t)n);
    f.read((char *)v.data(), n);
    return v;
}

// FNV-1a. AMD uses MD5, but the digest is only ever used as a unique C
// identifier and a dedupe key - it never leaves the generated header - so any
// stable hash serves. Written as 16 hex characters to match the shape of the
// names AMD produces.
static std::string digestOf(const std::vector<uint8_t> &b)
{
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < b.size(); ++i) {
        h ^= (uint64_t)b[i];
        h *= 1099511628211ULL;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
    return std::string(buf);
}

// ---- REFLECTION, CLASSIFIED EXACTLY AS glsl_compiler.cpp DOES IT.
//
// The mapping is transcribed rather than reasoned about, because getting it
// wrong produces a header that compiles and binds the wrong descriptor type at
// runtime - which surfaces as corrupted output, not as an error.
static bool reflect(const std::vector<uint8_t> &spirv, Reflection &out,
                    std::string &err)
{
    SpvReflectShaderModule mod;
    SpvReflectResult r = spvReflectCreateShaderModule(spirv.size(), spirv.data(), &mod);
    if (r != SPV_REFLECT_RESULT_SUCCESS) { err = "spvReflectCreateShaderModule failed"; return false; }

    uint32_t n = 0;
    r = spvReflectEnumerateDescriptorSets(&mod, &n, NULL);
    if (r != SPV_REFLECT_RESULT_SUCCESS) { spvReflectDestroyShaderModule(&mod); err = "enumerate sets failed"; return false; }
    std::vector<SpvReflectDescriptorSet *> sets(n);
    if (n) {
        r = spvReflectEnumerateDescriptorSets(&mod, &n, sets.data());
        if (r != SPV_REFLECT_RESULT_SUCCESS) { spvReflectDestroyShaderModule(&mod); err = "enumerate sets failed"; return false; }
    }

    for (size_t s = 0; s < sets.size(); ++s) {
        SpvReflectDescriptorSet *ds = sets[s];
        for (uint32_t b = 0; b < ds->binding_count; ++b) {
            SpvReflectDescriptorBinding *bd = ds->bindings[b];
            ResInfo ri;
            ri.name    = bd->name ? bd->name : "";
            ri.binding = bd->binding;
            ri.count   = bd->count;
            ri.space   = ds->set;

            switch (bd->descriptor_type) {
            case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                out.cbv.push_back(ri); break;
            case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                out.srvTex.push_back(ri); break;
            case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER:
                out.sampler.push_back(ri); break;
            case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                out.uavTex.push_back(ri); break;
            case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                // SRV vs UAV is the readonly flag, not the descriptor type.
                if (bd->resource_type == SPV_REFLECT_RESOURCE_FLAG_SRV)
                    out.srvBuf.push_back(ri);
                else
                    out.uavBuf.push_back(ri);
                break;
            case SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
                out.rtas.push_back(ri); break;
            default:
                // AMD throws here. Refusing loudly is right: an unhandled type
                // means the shader wants something the backend's binding code
                // has no case for, and guessing produces a wrong binding.
                spvReflectDestroyShaderModule(&mod);
                err = "unsupported descriptor type " +
                      std::to_string((int)bd->descriptor_type) + " on '" + ri.name + "'";
                return false;
            }
        }
    }
    spvReflectDestroyShaderModule(&mod);
    return true;
}

static void writeResourceArrays(FILE *fp, const std::string &pname,
                                const std::vector<ResInfo> &v, const char *kind)
{
    if (v.empty()) return;
    fprintf(fp, "static const char* g_%s_%sResourceNames[] = { ", pname.c_str(), kind);
    for (size_t i = 0; i < v.size(); ++i) fprintf(fp, " \"%s\",", v[i].name.c_str());
    fprintf(fp, " };\n");
    fprintf(fp, "static const uint32_t g_%s_%sResourceBindings[] = { ", pname.c_str(), kind);
    for (size_t i = 0; i < v.size(); ++i) fprintf(fp, " %u,", v[i].binding);
    fprintf(fp, " };\n");
    fprintf(fp, "static const uint32_t g_%s_%sResourceCounts[] = { ", pname.c_str(), kind);
    for (size_t i = 0; i < v.size(); ++i) fprintf(fp, " %u,", v[i].count);
    fprintf(fp, " };\n");
    fprintf(fp, "static const uint32_t g_%s_%sResourceSets[] = { ", pname.c_str(), kind);
    for (size_t i = 0; i < v.size(); ++i) fprintf(fp, " %u,", v[i].space);
    fprintf(fp, " };\n\n");
}

static void writeResourceRef(FILE *fp, const std::string &pname,
                             const std::vector<ResInfo> &v, const char *kind)
{
    if (v.empty()) { fprintf(fp, "0, 0, 0, 0, 0, "); return; }
    fprintf(fp, "%u, g_%s_%sResourceNames, g_%s_%sResourceBindings, "
                "g_%s_%sResourceCounts, g_%s_%sResourceSets, ",
            (unsigned)v.size(), pname.c_str(), kind, pname.c_str(), kind,
            pname.c_str(), kind, pname.c_str(), kind);
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    std::string glslang, src, name, outPath, tmpDir;
    std::vector<std::string> includes, defines, permOpts;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char *what) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value after %s\n", what); exit(2); }
            return argv[++i];
        };
        if      (a == "--glslang") glslang = next("--glslang");
        else if (a == "--src")     src     = next("--src");
        else if (a == "--name")    name    = next("--name");
        else if (a == "--out")     outPath = next("--out");
        else if (a == "--tmp")     tmpDir  = next("--tmp");
        else if (a == "-I")        includes.push_back(next("-I"));
        else if (a == "-D")        defines.push_back(next("-D"));
        else if (a == "--permute") permOpts.push_back(next("--permute"));
        else { fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 2; }
    }
    if (glslang.empty() || src.empty() || name.empty() || outPath.empty()) {
        fprintf(stderr,
            "usage: ffx_permute --glslang <exe> --src <file.glsl> --name <shaderName>\n"
            "                   --out <header.h> [--tmp <dir>] [-I <dir>]...\n"
            "                   [-D <NAME=VALUE>]... [--permute <OPTION_NAME>]...\n");
        return 2;
    }
    if (tmpDir.empty()) {
        const char *t = getenv("TEMP");
        tmpDir = std::string(t ? t : ".") + "/ffx_permute_" + name;
    }
    // Each option is one bit, matching AMD's numBits of 1 for {0,1} options.
    if (permOpts.size() > 20) { fprintf(stderr, "too many permutation options\n"); return 2; }
    const uint32_t bits  = (uint32_t)permOpts.size();
    const uint32_t total = 1u << bits;

    {   // The temp directory has to exist before glslang writes into it.
        std::string cmd = "cmd /c if not exist \"" + tmpDir + "\" mkdir \"" + tmpDir + "\" >nul 2>&1";
        (void)system(cmd.c_str());
    }

    std::vector<Permutation>       unique;
    std::map<std::string, uint32_t> digestToIndex;
    std::vector<uint32_t>          indirection(total, 0);

    for (uint32_t key = 0; key < total; ++key) {
        std::string spvPath = tmpDir + "/" + std::to_string(key) + ".spv";

        std::string cmd = "\"" + glslang + "\" -e CS --target-env vulkan1.2 -S comp -Os";
        for (size_t i = 0; i < includes.size(); ++i) cmd += " \"-I" + includes[i] + "\"";
        for (size_t i = 0; i < defines.size();  ++i) cmd += " \"-D" + defines[i]  + "\"";
        for (uint32_t b = 0; b < bits; ++b)
            cmd += " \"-D" + permOpts[b] + "=" + ((key >> b) & 1u ? "1" : "0") + "\"";
        cmd += " -o \"" + spvPath + "\" \"" + src + "\"";
        // glslang prints the source path on success; only the exit code matters.
        cmd += " >nul 2>&1";

        // ---- THE WHOLE COMMAND GETS ONE MORE LAYER OF QUOTES.
        //
        // system() goes through cmd.exe, and cmd strips the OUTER pair of
        // quotes when a command line both starts with a quote and contains
        // more of them. The quoted glslang path then lost its quotes and cmd
        // read the first path component as the program name:
        //
        //     'third_party' is not recognized as an internal or external command
        //
        // Wrapping the entire line in another pair is the documented way out
        // (cmd /c "..."): the outer pair is consumed and everything inside
        // survives intact, including paths with spaces - which every path here
        // has, since this tree lives under "Steam Games".
        const std::string full = "\"" + cmd + "\"";
        const int rc = system(full.c_str());
        std::vector<uint8_t> spirv = readFile(spvPath);
        if (rc != 0 || spirv.empty()) {
            fprintf(stderr, "%s: permutation %u FAILED to compile (rc=%d)\n",
                    name.c_str(), key, rc);
            // Re-run without suppression so the error is visible rather than
            // guessed at - a silent compile failure here would become a
            // missing table entry much later.
            std::string loud = "\"" + cmd.substr(0, cmd.size() - strlen(" >nul 2>&1")) + "\"";
            (void)system(loud.c_str());
            return 1;
        }

        const std::string d = digestOf(spirv);
        std::map<std::string, uint32_t>::iterator it = digestToIndex.find(d);
        if (it != digestToIndex.end()) {
            indirection[key] = it->second;   // byte-identical to one we have
            continue;
        }

        Permutation p;
        p.key    = key;
        p.digest = d;
        p.spirv  = spirv;
        std::string err;
        if (!reflect(spirv, p.refl, err)) {
            fprintf(stderr, "%s: permutation %u reflection failed: %s\n",
                    name.c_str(), key, err.c_str());
            return 1;
        }
        indirection[key] = (uint32_t)unique.size();
        digestToIndex[d] = (uint32_t)unique.size();
        unique.push_back(p);
    }

    FILE *fp = fopen(outPath.c_str(), "wb");
    if (!fp) { fprintf(stderr, "cannot write %s\n", outPath.c_str()); return 1; }

    fprintf(fp, "// GENERATED by tools/ffx_permute.cpp - do not edit.\n"
                "//\n"
                "// Source: %s\n"
                "// %u permutation key(s) over %u option(s), %u distinct blob(s).\n"
                "//\n"
                "// Format matches sdk/tools/ffx_shader_compiler (ffx_sc.cpp and\n"
                "// glsl_compiler.cpp); AMD's generator needs MSVC and cannot be built here.\n"
                "#pragma once\n\n"
                "#include <stdint.h>\n\n",
            src.c_str(), total, bits, (unsigned)unique.size());

    // Blobs and resource tables first: the info table below points at them.
    for (size_t i = 0; i < unique.size(); ++i) {
        const Permutation &p = unique[i];
        const std::string pname = name + "_" + p.digest;

        fprintf(fp, "static const uint32_t g_%s_size = %u;\n",
                pname.c_str(), (unsigned)p.spirv.size());
        fprintf(fp, "static const unsigned char g_%s_data[] = {\n", pname.c_str());
        for (size_t b = 0; b < p.spirv.size(); ++b) {
            fprintf(fp, "0x%02x,", p.spirv[b]);
            if ((b % 16) == 15) fprintf(fp, "\n");
        }
        fprintf(fp, "\n};\n\n");

        writeResourceArrays(fp, pname, p.refl.cbv,     "CBV");
        writeResourceArrays(fp, pname, p.refl.srvTex,  "TextureSRV");
        writeResourceArrays(fp, pname, p.refl.uavTex,  "TextureUAV");
        writeResourceArrays(fp, pname, p.refl.srvBuf,  "BufferSRV");
        writeResourceArrays(fp, pname, p.refl.uavBuf,  "BufferUAV");
        writeResourceArrays(fp, pname, p.refl.sampler, "Sampler");
        writeResourceArrays(fp, pname, p.refl.rtas,    "RTAccelerationStructure");
    }

    // The key union. Bit order is declaration order, which is the order the
    // options were passed - the same order the backend packs them in.
    fprintf(fp, "typedef union %s_PermutationKey {\n    struct {\n", name.c_str());
    for (uint32_t b = 0; b < bits; ++b)
        fprintf(fp, "        uint32_t %s : 1;\n", permOpts[b].c_str());
    fprintf(fp, "    };\n    uint32_t index;\n} %s_PermutationKey;\n\n", name.c_str());

    fprintf(fp, "typedef struct %s_PermutationInfo {\n", name.c_str());
    fprintf(fp, "    const uint32_t       blobSize;\n");
    fprintf(fp, "    const unsigned char* blobData;\n\n");
    static const char *kinds[] = { "ConstantBuffer", "SRVTexture", "UAVTexture",
                                   "SRVBuffer", "UAVBuffer", "Sampler",
                                   "RTAccelerationStructure" };
    static const char *nums[]  = { "numConstantBuffers", "numSRVTextures", "numUAVTextures",
                                   "numSRVBuffers", "numUAVBuffers", "numSamplers",
                                   "numRTAccelerationStructures" };
    static const char *lows[]  = { "constantBuffer", "srvTexture", "uavTexture",
                                   "srvBuffer", "uavBuffer", "sampler",
                                   "rtAccelerationStructure" };
    (void)kinds;
    for (int k = 0; k < 7; ++k) {
        fprintf(fp, "    const uint32_t  %s;\n", nums[k]);
        fprintf(fp, "    const char**    %sNames;\n", lows[k]);
        fprintf(fp, "    const uint32_t* %sBindings;\n", lows[k]);
        fprintf(fp, "    const uint32_t* %sCounts;\n", lows[k]);
        fprintf(fp, "    const uint32_t* %sSpaces;\n", lows[k]);
        if (k != 6) fprintf(fp, "\n");
    }
    fprintf(fp, "} %s_PermutationInfo;\n\n", name.c_str());

    fprintf(fp, "static const uint32_t g_%s_IndirectionTable[] = {\n", name.c_str());
    for (uint32_t i = 0; i < total; ++i) fprintf(fp, "    %u,\n", indirection[i]);
    fprintf(fp, "};\n\n");

    fprintf(fp, "static const %s_PermutationInfo g_%s_PermutationInfo[] = {\n",
            name.c_str(), name.c_str());
    for (size_t i = 0; i < unique.size(); ++i) {
        const Permutation &p = unique[i];
        const std::string pname = name + "_" + p.digest;
        fprintf(fp, "    { g_%s_size, g_%s_data, ", pname.c_str(), pname.c_str());
        writeResourceRef(fp, pname, p.refl.cbv,     "CBV");
        writeResourceRef(fp, pname, p.refl.srvTex,  "TextureSRV");
        writeResourceRef(fp, pname, p.refl.uavTex,  "TextureUAV");
        writeResourceRef(fp, pname, p.refl.srvBuf,  "BufferSRV");
        writeResourceRef(fp, pname, p.refl.uavBuf,  "BufferUAV");
        writeResourceRef(fp, pname, p.refl.sampler, "Sampler");
        writeResourceRef(fp, pname, p.refl.rtas,    "RTAccelerationStructure");
        fprintf(fp, "},\n");
    }
    fprintf(fp, "};\n");
    fclose(fp);

    printf("%-58s %3u keys -> %2u blobs\n", name.c_str(), total, (unsigned)unique.size());
    return 0;
}
