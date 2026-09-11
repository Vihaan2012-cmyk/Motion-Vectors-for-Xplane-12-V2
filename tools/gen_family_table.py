#!/usr/bin/env python3
"""gen_family_table.py - regenerate src/vklayer/mv_family_table.h from X-Plane's shader pack(s).

The injection decides per pipeline what its velocity write should do by looking the VERTEX
shader up in three sorted hash tables (mv_family_table.h):
    kMvZeroVert    screen-fixed overlays: write velocity = (0,0)   (font, manipulators, rain, legacy 2-D)
    kMvGroundVert  ground families: bind with a ZERO near-field threshold (never take the body frame)
    kMvMaskedVert  write no velocity at all (post passes, blits, debug views, particles, ...)
A module can be in more than one table (line3d is ground AND masked). Membership is by FAMILY,
the .xsa archive name inside spv.zip; every vertex-stage module of a member family is listed,
deduplicated by hash (terrain has thousands of permutations that share a few hundred vertex
shaders). Hash = FNV-1a-64 over the SPIR-V words exactly as layer.cpp's mvFragHash does.

The original generator was lost in the 2026-09-08 tree reset; this one was rebuilt from the
table it left behind and is checked by `--validate`, which must reproduce that table's hash sets
from the 12.4.3 pack before it is trusted with a newer one.

Usage:
  python tools/gen_family_table.py --validate <12.4.3 spv.zip or extracted dir>
  python tools/gen_family_table.py --out src/vklayer/mv_family_table.h <pack> [<pack> ...]
  python tools/gen_family_table.py --unpack <dir> <pack>      # write <dir>/<family>/<name>.spv
Packs may be spv.zip files or directories holding the .xsa archives (a `spv/` subfolder is fine).
Several packs are merged: hashes are unique per SPIR-V content, so one table serves every
version whose pack was merged in.
"""
import argparse, io, os, re, struct, sys, zipfile

# ---- policy: family -> tables. Recovered 2026-09-11 from the shipped table's name comments.
ZERO = {"font", "legacy_depth", "legacy_flat", "manip3d", "rain", "rain_draw", "rain_surface"}
GROUND = {"airport_raster", "astronomical", "dome", "fake_terrain", "ground_lights", "line3d",
          "ocean_shading", "planet", "tchotchke", "terrain"}
MASKED = {"acf_map_icon", "airport_raster", "atmosphere", "background_blur", "blit", "blur",
          "cloud_categorize", "cloud_map", "cloud_render_raster", "cube_filter_raster", "debug",
          "depth_resolve", "empty", "ground_lights", "hdr", "histo_debug", "light", "light_vis",
          "line3d", "msaa_categorize", "ocean_meta_data", "ocean_readback", "particle",
          "profile_view", "rain_forces", "rain_normals", "resolve", "scatter_render_atmosphere",
          "sectional2", "shadow_rect", "ssr_mesh", "svt", "volumetric_apply",
          "weather_apply_raster", "wxr_mask"}
# 12.4.4 (public beta 2026-09-10) introduced these archives. None of them is scene geometry:
# lighting compaction, shading-rate control and its debug view, VR/MR blits, a hang probe.
# Defaulted to MASKED (write no velocity; those pixels fall back to depth reprojection, which
# is what unpatched geometry gets anyway). Revisit with a 12.4.4 trace if any of them draws
# world geometry.
MASKED |= {"device_hang", "light_spill_compact", "mixed_reality", "shading_rate",
           "shading_rate_vis", "vr_depth_copy"}

FNV_OFFSET, FNV_PRIME, MASK64 = 1469598103934665603, 1099511628211, (1 << 64) - 1

def fnv1a64_words(data):
    n = len(data) // 4
    h = FNV_OFFSET
    for (w,) in struct.iter_unpack("<I", data[:n * 4]):
        h = ((h ^ w) * FNV_PRIME) & MASK64
    return h

def entry_models(data):
    """Execution models of every OpEntryPoint (0 = Vertex, 4 = Fragment, 5 = Compute)."""
    if len(data) < 20 or struct.unpack_from("<I", data, 0)[0] != 0x07230203:
        return []
    models, pos, n = [], 5, len(data) // 4
    words = struct.unpack("<%dI" % n, data[:n * 4])
    while pos < n:
        wc, op = words[pos] >> 16, words[pos] & 0xFFFF
        if wc == 0: break
        if op == 15: models.append(words[pos + 1])   # OpEntryPoint model id name ...
        if op == 17 or op > 15 and models and op in (16,):  # past the preamble once functions start
            pass
        pos += wc
    return models

def iter_pack(pack):
    """Yield (family, module_name, bytes) for every .spv inside every .xsa of a pack."""
    if os.path.isdir(pack):
        root = os.path.join(pack, "spv") if os.path.isdir(os.path.join(pack, "spv")) else pack
        xsas = [(os.path.splitext(f)[0], open(os.path.join(root, f), "rb").read())
                for f in sorted(os.listdir(root)) if f.endswith(".xsa")]
    else:
        with zipfile.ZipFile(pack) as z:
            xsas = [(os.path.splitext(os.path.basename(n))[0], z.read(n))
                    for n in sorted(z.namelist()) if n.endswith(".xsa")]
    for fam, blob in xsas:
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as inner:
                for n in sorted(inner.namelist()):
                    if n.endswith(".spv"):
                        yield fam, os.path.basename(n), inner.read(n)
        except zipfile.BadZipFile:
            print("  ! %s.xsa is not a zip - skipped" % fam, file=sys.stderr)

def collect(packs):
    """{table: {hash: representative name}}, plus per-family stats."""
    tables = {"kMvZeroVert": {}, "kMvGroundVert": {}, "kMvMaskedVert": {}}
    stats = {}
    for pack in packs:
        for fam, name, data in iter_pack(pack):
            st = stats.setdefault(fam, [0, 0, set()])
            st[0] += 1
            if 0 not in entry_models(data):   # vertex stage only
                continue
            st[1] += 1
            h = fnv1a64_words(data)
            st[2].add(h)
            if fam in ZERO:   tables["kMvZeroVert"].setdefault(h, name)
            if fam in GROUND: tables["kMvGroundVert"].setdefault(h, name)
            if fam in MASKED: tables["kMvMaskedVert"].setdefault(h, name)
    return tables, stats

def read_existing(path):
    s = open(path, encoding="utf-8").read()
    out = {}
    for arr in ("kMvZeroVert", "kMvGroundVert", "kMvMaskedVert"):
        body = re.search(r"static const uint64_t %s\[\] = \{(.*?)\};" % arr, s, re.S).group(1)
        out[arr] = {int(h, 16): n for h, n in re.findall(r"(0x[0-9a-f]+)ull,\s*//\s*(\S+)", body)}
    helpers = s[s.index("// Sorted array; one binary search per pipeline creation."):]
    return out, helpers

def write_header(path, tables, helpers, packs):
    lines = ["// GENERATED by gen_family_table.py from X-Plane's shipped SPIR-V pack",
             "// (Resources/shaders/bin/spv.zip, hashed with the injection's FNV-1a-64).",
             "// Regenerate after an X-Plane update changes the shader pack.",
             "// Policy and reasoning live in the generator's header comment.",
             "// Packs merged: " + ", ".join(os.path.basename(p.rstrip("/\\")) for p in packs),
             "#pragma once", "#include <cstdint>", "#include <cstddef>", ""]
    for arr in ("kMvZeroVert", "kMvGroundVert", "kMvMaskedVert"):
        lines.append("static const uint64_t %s[] = {" % arr)
        for h in sorted(tables[arr]):
            lines.append("    0x%016xull, // %s" % (h, tables[arr][h]))
        lines.append("};")
        lines.append("static const size_t %sCount = sizeof(%s) / sizeof(%s[0]);" % (arr, arr, arr))
        lines.append("")
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + helpers)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packs", nargs="+")
    ap.add_argument("--validate", action="store_true", help="compare against the existing header instead of writing")
    ap.add_argument("--existing", default=os.path.join(os.path.dirname(__file__), "..", "src", "vklayer", "mv_family_table.h"))
    ap.add_argument("--out")
    ap.add_argument("--unpack", help="also write every .spv to <dir>/<family>/ for inject_test")
    a = ap.parse_args()
    if a.unpack:
        for pack in a.packs:
            for fam, name, data in iter_pack(pack):
                d = os.path.join(a.unpack, fam); os.makedirs(d, exist_ok=True)
                open(os.path.join(d, name), "wb").write(data)
        print("unpacked into", a.unpack)
    tables, stats = collect(a.packs)
    for fam in sorted(stats):
        t = [n[3:-4] for n in tables if any(fam == f for f in (fam,)) and fam in {"kMvZeroVert": ZERO, "kMvGroundVert": GROUND, "kMvMaskedVert": MASKED}[n]]
        print("  %-28s modules=%-5d vertex=%-4d unique=%-4d %s" % (fam, stats[fam][0], stats[fam][1], len(stats[fam][2]), "/".join(t) or "-"))
    existing, helpers = read_existing(a.existing)
    for arr in tables:
        new, old = set(tables[arr]), set(existing[arr])
        print("%-14s generated=%-4d existing=%-4d missing_from_generated=%-3d extra_in_generated=%d"
              % (arr, len(new), len(old), len(old - new), len(new - old)))
        for h in sorted(old - new)[:6]: print("      missing: 0x%016x %s" % (h, existing[arr][h]))
        for h in sorted(new - old)[:6]: print("      extra:   0x%016x %s" % (h, tables[arr][h]))
    if a.validate:
        ok = all(set(tables[arr]) == set(existing[arr]) for arr in tables)
        print("VALIDATE:", "OK - the generator reproduces the shipped table" if ok else "MISMATCH")
        sys.exit(0 if ok else 1)
    if a.out:
        write_header(a.out, tables, helpers, a.packs)
        print("wrote", a.out)

if __name__ == "__main__":
    main()
