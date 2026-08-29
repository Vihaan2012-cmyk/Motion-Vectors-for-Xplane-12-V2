#pragma once

// LIVE CONTROLS: change anything without restarting the sim.
//
// This exists because of how the work actually went. Every question - is the
// plumbing right, is the alpha too low, does mode 1 ghost the way it should,
// which pass did we pick - cost a launch, a look, and a kill, and the answer
// arrived four minutes later with the sim's startup in between. A dozen of those
// is an afternoon, and the thing being measured is usually a single number.
//
// Every knob here is read from a text file that is re-read while the sim runs.
// Edit it, save it, and the next frame uses the new value. Nothing is cached in
// a static, nothing needs a relaunch, and the file can be edited from another
// window mid-flight.
//
// PRECEDENCE, and why it is this way round:
//
//     live file  >  environment variable  >  built-in default
//
// The environment still works exactly as it did, so every existing launcher,
// script and habit is unaffected and this is purely additive. The live file wins
// when a key is present because that is the whole point - it is the thing you
// are holding in your hand.
//
// The file is polled rather than watched. A directory-change notification is the
// tidier design and needs a thread, a handle to close, and a failure mode when
// the directory disappears; a poll is a single GetFileAttributesEx every N
// frames comparing a 64-bit timestamp, which is unmeasurable next to a frame,
// and it cannot leak anything.

#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <map>
#include <mutex>

namespace live {

static std::mutex              g_mtx;
static std::map<std::string, std::string> g_kv;
static std::string             g_path;
static uint64_t                g_stamp   = 0;
static uint64_t                g_reloads = 0;
static bool                    g_everSeen = false;

inline const char *path()
{
    if (g_path.empty()) {
        if (const char *p = getenv("TAA_LIVE_FILE")) {
            g_path = p;
        } else {
            const char *t = getenv("TEMP");
            g_path = std::string(t ? t : ".") + "\\taa_live.ini";
        }
    }
    return g_path.c_str();
}

// ---- WRITE A TEMPLATE SO THE FILE IS DISCOVERABLE.
//
// A control file nobody knows the keys of is not a control file. This writes a
// fully commented one the first time the layer starts and the file does not
// exist, so the whole surface is visible by opening it - and every key is
// present but commented out, which means the defaults are documented by the same
// text that changes them.
inline void writeTemplate()
{
    FILE *f = fopen(path(), "r");
    if (f) { fclose(f); return; }        // never overwrite the user's edits
    f = fopen(path(), "w");
    if (!f) return;
    fprintf(f,
"# Motion Vectors - LIVE CONTROLS\n"
"#\n"
"# Edit and save while X-Plane is running. The layer re-reads this file every\n"
"# few frames and applies changes immediately. No restart, ever.\n"
"#\n"
"# Precedence: this file > environment variable > built-in default.\n"
"# A key that is absent or commented out falls through to the environment.\n"
"# Delete a line to hand control back; you do not have to restore its default.\n"
"#\n"
"# ---------------------------------------------------------------- resolve\n"
"# taa.enable      0/1   master switch. 0 leaves the frame completely untouched.\n"
"# taa.mode        0     passthrough - every binding, barrier and dispatch runs\n"
"#                       but the output is the input. IF THE FRAME CHANGES AT\n"
"#                       ALL IN THIS MODE THE FAULT IS PLUMBING, FULL STOP.\n"
"#                 1     reproject only. Ghosting along motion is EXPECTED and\n"
"#                       means reprojection works.\n"
"#                 2     full - neighbourhood statistics on top. Ghosting should\n"
"#                       collapse.\n"
"# taa.alpha       0.1   weight of the current frame. Lower accumulates more.\n"
"# taa.gain        4.0   how hard a rejected history sample pushes alpha to 1.\n"
"# taa.varclip     1.25  variance box width, in standard deviations.\n"
"#\n"
"# ---------------------------------------------------------------- isolate\n"
"# Each of these removes ONE input, so a fault can be attributed instead of\n"
"# guessed. They are the generalisation of taa.mode=0 to every stage.\n"
"#\n"
"# taa.freeze_history  0/1  stop updating history. The picture should freeze in\n"
"#                          place and smear along motion. If it does not, the\n"
"#                          history is not what is being displayed.\n"
"# taa.no_motion       0/1  treat every vector as zero. Reprojection becomes a\n"
"#                          same-pixel fetch, so any remaining ghosting is NOT\n"
"#                          the vectors.\n"
"# taa.no_accum        0/1  output the current frame but keep every binding,\n"
"#                          barrier and dispatch live.\n"
"# taa.force_reset     0/1  drop history every frame. Equivalent to no_accum but\n"
"#                          through the reset path, so the two together separate\n"
"#                          'reset is broken' from 'accumulation is broken'.\n"
"#\n"
"# ---------------------------------------------------------------- look at it\n"
"# taa.viz         0     off, normal image\n"
"#                 1     motion vectors as colour. Red is +x, green is +y, and\n"
"#                       grey is zero. Sky and clouds should be FLAT GREY - they\n"
"#                       have no vectors - and the ground should be a smooth\n"
"#                       gradient with no blocks or discontinuities.\n"
"#                 2     magnitude heatmap. Black 0 px, blue 1, green 4, yellow\n"
"#                       16, red 64+. Should scale as 1/depth.\n"
"#                 3     invalid pixels. RED  = no vector was written here while\n"
"#                       the camera moved (sky, cloud, or a shader we failed to\n"
"#                       patch). BLUE = history reprojected off screen. GREEN =\n"
"#                       accepted. This is the map that says whether the cloud\n"
"#                       and sky rejection is doing what it claims.\n"
"#                 4     history buffer directly, before it is copied back.\n"
"#                 5     blend weight. Black is full accumulation, white is the\n"
"#                       current frame only. Disocclusion edges should glow.\n"
"#                 6     how far the clamp moved history, in sigma.\n"
"# taa.viz_scale   1.0   multiplies the magnitude and weight views.\n"
"#\n"
"# ---------------------------------------------------------------- reports\n"
"# report          1     dump the full state to the log ONCE, then clear itself\n"
"#                       back to 0 in this file. Image census, pass census,\n"
"#                       injection outcomes, every refusal and its reason,\n"
"#                       timings, and the live values in force.\n"
"# report.every    0     dump it automatically every N frames. 0 disables.\n"
"#\n"
"# ---------------------------------------------------------------- vram system\n"
"# vram.enable     1     master switch for the whole VRAM system.\n"
"# vram.shape      1     budget shaping: low-pass + monotone-under-free +\n"
"#                       zone reserve on the budget X-Plane reads.\n"
"# vram.recycle    1     deferred-free pool: freed device-local blocks are\n"
"#                       held and identical re-allocations answered instantly.\n"
"# vram.priority   1     per-category memory priorities via\n"
"#                       VK_EXT_memory_priority / pageable_device_local.\n"
"# vram.governor   1     upload pacing on the transfer-only queue, ORANGE+.\n"
"# vram.reserve_g/y/o/r/c    MB withheld from the reported budget per zone\n"
"#                           (defaults 128/256/384/512/768).\n"
"# vram.upload_o/r/c         MB of uploads released per frame in that zone\n"
"#                           (defaults 64/24/8; _g/_y default 0 = unlimited).\n"
"# vram.upload_max_hold 2    presents a held upload may wait before release.\n"
"# vram.recycle_max_mb 256   pool cap;  vram.recycle_hold_frames 180  age cap.\n"
"# vram.deflate_mb 512       emergency budget cut after an allocation failure,\n"
"# vram.deflate_frames 600   and for how long.\n"
"# vram.teleport_m 2000      camera jump that counts as a teleport.\n"
"# vram.budget_alpha 0.02    low-pass constant per budget query.\n"
"# vram.adaptive 1           frame-time feedback notches the upload budget\n"
"#                           down when frame time degrades while uploading.\n"
"# vram.speed_reserve 0.01   reserve growth per m/frame of camera speed.\n"
"# vram.tex_drop_above       override the zone policy's preload texture cap.\n"
"# vram.tex_streamed_to      override the zone policy's streamed texture cap.\n"
"# vram.upload_cache 1       elide re-uploads of identical texture content;\n"
"#                           also prices eviction-means-disk-reload in bytes.\n"
"# vram.warmup_frames 900    progressive VRAM fill after device creation,\n"
"# vram.warmup_mb 512        starting this much extra reserve, decaying to 0.\n"
"# vram.hold_max_mb 512      governor backpressure: past this, pass through.\n"
"# vram.lookahead 300        frames of usage-trend projection (predict peaks).\n"
"# vram.age_frames 1800      unused-resource window before priority decay;\n"
"#                           doubled for frequently-used resources.\n"
"# vram.bench=1 / =0         open / close a measurement window; closing dumps\n"
"#                           avg, 1%% low, 0.1%% low, VRAM peak, uploads,\n"
"#                           allocs, JIT pipelines, zone residency.\n"
"# vram.trace_every 600      heartbeat cadence in frames; 0 silences it.\n"
"# vram.report=1             one-shot full state dump, clears itself.\n"
"#\n"
"# ---------------------------------------------------------------- examples\n"
"# taa.enable=1\n"
"# taa.mode=2\n"
"# taa.alpha=0.1\n"
"# taa.viz=3\n"
"# report=1\n");
    fclose(f);
}

// Re-read if the file changed. Called once per frame; the common case is one
// GetFileAttributesEx and a 64-bit compare.
// Load the file RIGHT NOW, ignoring the poll interval.
//
// poll() only reads every Nth frame, which is right for a hot-reload but wrong
// for a value that is latched once at startup: the first caller would latch a
// default and keep it for the life of the process however often the file is
// edited. fsr.replace is exactly that kind of key, so it needs a forced read.
inline void loadNow()
{
    // The template has to exist before it can be read. poll() does this on its
    // first call; loadNow() may well BE the first call.
    static bool templated = false;
    if (!templated) { templated = true; writeTemplate(); }
    WIN32_FILE_ATTRIBUTE_DATA fad;
    if (!GetFileAttributesExA(path(), GetFileExInfoStandard, &fad)) return;
    uint64_t stamp = ((uint64_t)fad.ftLastWriteTime.dwHighDateTime << 32) |
                      fad.ftLastWriteTime.dwLowDateTime;
    if (stamp == g_stamp && g_everSeen) return;

    FILE *f = fopen(path(), "r");
    if (!f) return;
    std::map<std::string, std::string> kv;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (*p == ' ' || *p == '\t') ++p;
        if (*p == '#' || *p == ';' || *p == '\n' || *p == '\r' || !*p) continue;
        char *eq = strchr(p, '=');
        if (!eq) continue;
        *eq = 0;
        std::string k(p), v(eq + 1);
        // Trim both ends of both halves: a trailing \r from an editor that
        // writes CRLF would otherwise become part of every value, and a value
        // of "2\r" parses as 2 while a value of "on\r" does not.
        while (!k.empty() && (k.back()==' '||k.back()=='\t')) k.pop_back();
        while (!v.empty() && (v.back()=='\n'||v.back()=='\r'||v.back()==' '||v.back()=='\t'))
            v.pop_back();
        size_t s = v.find_first_not_of(" \t");
        if (s != std::string::npos) v = v.substr(s);
        if (!k.empty()) kv[k] = v;
    }
    fclose(f);

    std::lock_guard<std::mutex> g(g_mtx);
    // Report only what CHANGED. A full dump every reload buries the one line
    // that matters, and the reason for reloading is almost always a single edit.
    for (std::map<std::string, std::string>::iterator it = kv.begin();
         it != kv.end(); ++it) {
        std::map<std::string, std::string>::iterator o = g_kv.find(it->first);
        if (o == g_kv.end() || o->second != it->second)
            trace("LIVE: %s = %s%s", it->first.c_str(), it->second.c_str(),
                  o == g_kv.end() ? "  (new)" : "");
    }
    for (std::map<std::string, std::string>::iterator it = g_kv.begin();
         it != g_kv.end(); ++it)
        if (!kv.count(it->first))
            trace("LIVE: %s removed - falling back to the environment or the "
                  "built-in default", it->first.c_str());

    // The shipped lock silently overrides 31 keys; its state changing is
    // exactly the kind of thing the trace exists to record.
    {
        std::map<std::string, std::string>::iterator ul = kv.find("taa.unlock");
        const bool nowUnlocked =
            ul != kv.end() && !(ul->second == "0" || ul->second == "off" ||
                                ul->second == "false" || ul->second == "no");
        static int last = -1;
        if ((int)nowUnlocked != last) {
            last = (int)nowUnlocked;
            trace(nowUnlocked
                ? "LIVE: taa.unlock=1 - the ini and environment now override "
                  "the shipped tuning table"
                : "LIVE: shipped tuning table IN FORCE - 31 keys locked "
                  "(ini and environment ignored for them; taa.unlock=1 frees)");
        }
    }
    g_kv = kv;
    g_stamp = stamp;
    g_everSeen = true;
    ++g_reloads;
}

inline void poll()
{
    static uint32_t counter = 0;
    static uint32_t everyN = 0;
    if (!everyN) {
        everyN = getenv("TAA_LIVE_POLL") ? (uint32_t)atoi(getenv("TAA_LIVE_POLL")) : 15;
        if (!everyN) everyN = 15;
        writeTemplate();
    }
    if (++counter % everyN) return;
    loadNow();
}


// Clear a one-shot key back to 0 in the file itself, so `report=1` fires once
// rather than every frame until somebody notices. Rewrites the line in place and
// leaves every comment intact.
inline void clearOneShot(const char *key)
{
    FILE *f = fopen(path(), "r");
    if (!f) return;
    std::string all;
    char line[512];
    while (fgets(line, sizeof(line), f)) all += line;
    fclose(f);

    std::string want = std::string(key) + "=";
    size_t at = 0;
    bool changed = false;
    while ((at = all.find(want, at)) != std::string::npos) {
        // Only at the start of a line, and not inside a comment.
        bool lineStart = (at == 0) || all[at-1] == '\n';
        if (lineStart) {
            size_t eol = all.find('\n', at);
            if (eol == std::string::npos) eol = all.size();
            all.replace(at, eol - at, want + "0");
            changed = true;
            break;
        }
        at += want.size();
    }
    if (!changed) return;
    f = fopen(path(), "w");
    if (!f) return;
    fwrite(all.data(), 1, all.size(), f);
    fclose(f);
    // Adopt our own edit so the next poll does not report it as a user change.
    WIN32_FILE_ATTRIBUTE_DATA fad;
    if (GetFileAttributesExA(path(), GetFileExInfoStandard, &fad))
        g_stamp = ((uint64_t)fad.ftLastWriteTime.dwHighDateTime << 32) |
                   fad.ftLastWriteTime.dwLowDateTime;
    std::lock_guard<std::mutex> g(g_mtx);
    g_kv[key] = "0";
}

// ---- THE SHIPPED CONFIGURATION, COMPILED IN.
//
// These are the values the mod is known to look correct with. They live here,
// in the binary, rather than in a file for one reason: the file is not reliable.
// config/taa_live.ini is not even read - the layer resolves its path to
// TAA_LIVE_FILE or %TEMP%	aa_live.ini - so for a long time the "shipped
// config" was a document nobody loaded, and the two had silently diverged.
// A value that matters cannot live somewhere that might not be read.
//
// These OVERRIDE the ini and the environment. That is the opposite of the usual
// precedence and it is deliberate: a tuned value that took measurement to find
// should not be lost to a stale ini, a half-finished experiment, or a file
// someone edited once and forgot.
//
// WHAT IS NOT HERE MATTERS AS MUCH AS WHAT IS. Deliberately absent:
//   - taa.enable, and the effect toggles (ao, contact, sharpen, dilate)
//   - taa.fg and friends
//   - the debug and one-shot switches (viz, force_reset, freeze_history, ...)
// Freezing those would turn every button in the panel into a decoration, and
// the effect toggles are exactly the controls needed to A/B a change. Only the
// deep tuning - the numbers a user has no way to rederive - is locked.
//
// taa.unlock=1 releases all of it, for when tuning is the point.
struct Shipped { const char *k; const char *v; };
// The Lua panel hand-mirrors this table as LOCKED{}; the two drift silently
// unless both sides count. C side asserts here; Lua asserts at load.
#define MV_SHIPPED_COUNT 31
static const Shipped kShipped[] = {
    // Accumulation and clipping.
    { "taa.mode",              "2"      },
    { "taa.alpha",             "0.05"   },
    { "taa.alpha_moving",      "0.35"   },
    { "taa.alpha_moving_px",   "3.0"    },
    { "taa.gain",              "4.0"    },
    { "taa.varclip",           "8.0"    },
    { "taa.moved_dead",        "0.0"    },
    { "taa.moved_eps",         "0.0001" },
    { "taa.novec_alpha",       "0.05"   },
    { "taa.novec_by_vel",      "0"      },
    { "taa.novec_cov",         "-1.0"   },
    { "taa.reactive",          "0"      },
    { "taa.hist_catmull",      "1"      },
    // Jitter and the unjitter shift. smul_y is negative because the viewport
    // height is; this pair was swept, not argued, and must not drift.
    { "taa.jitter_scale",      "1.0"    },
    { "taa.unjitter",          "1"      },
    { "taa.smul_x",            "0.5"    },
    { "taa.smul_y",            "-0.5"   },
    // Velocity conventions.
    { "taa.vel_scale",         "1.0"    },
    { "taa.vel_ypos",          "0"      },
    { "taa.vel_max",           "1.0"    },
    // Pass identification. These are how the layer finds the scene at all.
    { "taa.clear_mode",        "1"      },
    { "taa.mv_pass",           "-1"     },
    { "taa.sticky_colour",     "-1"     },
    { "taa.max_resolves",      "1"      },
    { "taa.quad_needs_depth",  "1"      },
    { "taa.quad_needs_pull",   "1"      },
    { "taa.scene_needs_depth", "1"      },
    // Near field. 0 disarms the near-field select, which is what caused the
    // cockpit shake; 5.0 is the measured working value.
    { "taa.nearfield_m",       "5.0"    },
    { "taa.nearfield_view",    "-1"     },
    // Sampling radius for the AO term. The STRENGTH stays user-facing; only the
    // radius is fixed, because it is a tuned number and not a preference.
    { "taa.ao_radius",         "12.0"   },
    // Measured at -0.5 and it cost about half the frame rate at 4K on 8 GB of
    // VRAM. It stays at 0 unless someone deliberately unlocks and retunes it.
    { "taa.lod_bias",          "0.0"    },
};
static_assert(sizeof(kShipped) / sizeof(kShipped[0]) == MV_SHIPPED_COUNT,
              "kShipped changed size - update MV_SHIPPED_COUNT and the Lua "
              "LOCKED table together");

// Reads g_kv directly and does NOT take the lock - it is called from inside
// lookup(), which already holds it, and std::mutex is not recursive.
// O(1) membership for the shipped table. The 31-strcmp linear scan ran on
// EVERY knob read - dozens per frame - and this map is built once.
inline const std::map<std::string, const char *> &shippedMap()
{
    static std::map<std::string, const char *> m;
    if (m.empty())
        for (size_t n = 0; n < sizeof(kShipped) / sizeof(kShipped[0]); ++n)
            m[kShipped[n].k] = kShipped[n].v;
    return m;
}

inline bool unlockedLocked()
{
    std::map<std::string, std::string>::iterator it = g_kv.find("taa.unlock");
    if (it == g_kv.end()) return false;
    return !(it->second == "0" || it->second == "off" ||
             it->second == "false" || it->second == "no");
}

inline bool lookup(const char *key, std::string &out)
{
    std::lock_guard<std::mutex> g(g_mtx);
    // The shipped table wins, unless the user has explicitly unlocked it.
    // NOTE the lock overrides the ENVIRONMENT as well as the ini - TAA_* vars
    // for locked keys are ignored while locked, by design.
    if (!unlockedLocked()) {
        std::map<std::string, const char *>::const_iterator sh =
            shippedMap().find(key);
        if (sh != shippedMap().end()) { out = sh->second; return true; }
    }
    std::map<std::string, std::string>::iterator it = g_kv.find(key);
    if (it == g_kv.end()) return false;
    out = it->second;
    return true;
}

// `env` may be null for a key with no environment equivalent.
inline int i(const char *key, const char *env, int dflt)
{
    std::string v;
    if (lookup(key, v)) return atoi(v.c_str());
    if (env) if (const char *e = getenv(env)) return atoi(e);
    return dflt;
}

inline float f(const char *key, const char *env, float dflt)
{
    std::string v;
    if (lookup(key, v)) return (float)atof(v.c_str());
    if (env) if (const char *e = getenv(env)) return (float)atof(e);
    return dflt;
}

// Presence-only environment variables (TAA_RESOLVE and friends) are true when
// SET, whatever their value, so the live equivalent has to mean the same thing
// while still being able to say "off" - which the environment cannot.
inline bool onoff(const char *key, const char *env, bool dflt)
{
    std::string v;
    if (lookup(key, v))
        return !(v == "0" || v == "off" || v == "false" || v == "no");
    if (env) if (getenv(env)) return true;
    return dflt;
}

inline uint64_t reloads() { return g_reloads; }

// ---- FOR PER-COMMAND HOOKS, WHERE THE PLAIN ACCESSORS ARE TOO EXPENSIVE.
//
// i()/f()/onoff() each take g_mtx, build temporary std::strings from the char*
// key and do up to three string-keyed map lookups (taa.unlock, the shipped
// table, then g_kv); a key absent from the shipped table also falls through to
// getenv(), a scan of the environment block. That is fine once a frame and
// wrong once a draw - and X-Plane binds pipelines and issues dispatches by the
// hundred per frame.
//
// g_reloads only changes when the ini is actually re-read, so a value stamped
// with it is exactly as fresh as the file. Callers own the two statics, which
// keeps this allocation-free and leaves the "how stale may this be" decision at
// the call site. A race costs one call a one-frame-old value.
inline int iCached(const char *key, const char *env, int dflt,
                   uint64_t &gen, int &slot)
{
    const uint64_t g = reloads();
    if (g != gen) { gen = g; slot = i(key, env, dflt); }
    return slot;
}

} // namespace live
