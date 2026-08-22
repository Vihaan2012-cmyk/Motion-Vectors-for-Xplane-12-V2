#pragma once

#include "taa_live_default.h"

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

#include "../product.h"

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
        if (const char *p = getenv(MV_LIVE_ENV)) {
            g_path = p;
        } else {
            const char *t = getenv("TEMP");
            g_path = std::string(t ? t : ".") + "\\" MV_LIVE_FILE;
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
    // ---- AN EXISTING FILE WITH NO ACTIVE KEYS IS THE OLD DEAD TEMPLATE.
    //
    // Every release up to 1.0.2 wrote a template whose keys were all commented
    // out, and left it in %TEMP% forever. A plain "never overwrite" check would
    // preserve exactly that file for everyone who has already run this mod, so
    // they would keep the compiled defaults - and the ghosting - no matter how
    // many times they reinstalled.
    //
    // A real edit always leaves at least one active key behind. Zero active
    // keys means nothing was ever chosen, so there is no user intent to
    // protect and the file is reseeded. Anything with a single live setting in
    // it is left strictly alone.
    FILE *f = fopen(path(), "r");
    if (f) {
        int active = 0;
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            const char *p = line;
            // Numeric codes, not escapes: a heredoc ate the escapes in
            // this block once and produced unterminated character
            // literals. 9 tab, 10 newline, 13 carriage return.
            while (*p == ' ' || *p == 9) ++p;
            if (*p == '#' || *p == ';' || *p == 10 || *p == 13 || !*p)
                continue;
            if (strchr(p, '=')) { ++active; break; }
        }
        fclose(f);
        if (active) return;              // the user has chosen something: keep it
        trace("LIVE: %s has no active keys - it is the pre-1.0.3 template, "
              "reseeding with the known-good settings", path());
    }
    f = fopen(path(), "w");
    if (!f) return;
    // ---- SEEDED WITH THE TUNED CONFIG, KEYS ACTIVE.
    //
    // This used to write a documented template with every key COMMENTED OUT,
    // which meant a fresh install ran on compiled defaults for all of them.
    // Two of those defaults - taa.mode=0 (passthrough) and taa.jitter_scale=0
    // (no jitter) - made TAA do nothing at all, and the rest were never
    // audited, which is how ghosting shipped afterwards.
    //
    // Auditing forty-eight defaults against a config file is the wrong shape
    // of fix. The file already takes precedence over environment and built-in
    // default, so writing the KNOWN-GOOD file itself makes a fresh install
    // identical to the machine it was tuned on by construction.
    //
    // The user's own edits are still never overwritten: the early return above
    // fires whenever the file already exists.
    fputs(kLiveDefaultIni, f);
    fclose(f);
}

// Re-read if the file changed. Called once per frame; the common case is one
// GetFileAttributesEx and a 64-bit compare.
// ---- READ THE FILE NOW, WHATEVER THE COUNTER SAYS.
//
// poll() is a FRAME-RATE thing: it reads on every fifteenth call and returns
// immediately the other fourteen times. That is right for a hot path and wrong
// for anyone who needs an answer before the first frame exists.
//
// vkCreateDevice is exactly that caller. It has to know whether crash
// destruction is enabled in order to decide whether to build the descriptor
// resources, and it runs long before anything calls poll(). Reading a key
// there without this returned the built-in default, and because the answer was
// then cached for the process, crash.enable=1 in the file could never switch
// anything on - the file was correct, the gate was closed, and nothing said so.
//
// Both entry points share one body so the parse cannot drift between them.
inline void loadNow();

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

    g_kv = kv;
    g_stamp = stamp;
    g_everSeen = true;
    ++g_reloads;
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

inline bool lookup(const char *key, std::string &out)
{
    std::lock_guard<std::mutex> g(g_mtx);
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

} // namespace live
