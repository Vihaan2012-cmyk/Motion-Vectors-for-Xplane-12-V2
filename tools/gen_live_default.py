"""Embed config/taa_live.ini into src/vklayer/taa_live_default.h.

Run from the repository root; build.ps1 runs it before compiling the layer.

WHY THIS IS A FILE AND NOT A HEREDOC
------------------------------------
The escape this needs is a two-character backslash-n inside a C string
literal. Written as a shell heredoc, that collapses into a real newline
before Python ever sees it, and the generated header comes out as

    "# MotionVectors - KNOWN-GOOD LIVE SETTINGS
    "

which is an unterminated string literal. It has cost this project time more
than once. A script file has no shell in the path.
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INI = os.path.join(ROOT, "config", "taa_live.ini")
OUT = os.path.join(ROOT, "src", "vklayer", "taa_live_default.h")

BS = chr(92)          # backslash, never written literally near an escape
QUOTE = chr(34)
NEWLINE_ESCAPE = BS + "n"

HEADER = """// GENERATED from config/taa_live.ini by tools/gen_live_default.py.
// Do not edit: edit config/taa_live.ini and rebuild.
//
// ---- THE SEED IS THE KNOWN-GOOD CONFIG, WITH ITS KEYS ACTIVE.
//
// writeTemplate() used to emit a documented template with every key COMMENTED
// OUT, so a fresh install fell through to compiled defaults for all of them.
// Chasing those defaults one at a time is how ghosting shipped: taa.mode and
// taa.jitter_scale were found and corrected, and whatever else disagreed was
// not, because nobody had audited the other forty-six.
//
// This file takes precedence over both the environment and the built-in
// default, so seeding it with the tuned config makes a fresh install behave
// identically to the machine it was tuned on BY CONSTRUCTION, rather than by an
// audit that has already been wrong once.
#pragma once

static const char *kLiveDefaultIni =
"""


def main():
    if not os.path.exists(INI):
        sys.exit("missing " + INI)

    raw = io.open(INI, "rb").read().decode("utf-8-sig", "replace")
    # Normalise every line ending, including lone carriage returns: a stray CR
    # inside a C string literal terminates the line as far as the preprocessor
    # is concerned, which is the same unterminated-literal error by a different
    # route.
    text = raw.replace(chr(13) + chr(10), chr(10)).replace(chr(13), chr(10))

    parts = [HEADER]
    active = 0
    for line in text.split(chr(10)):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            active += 1
        escaped = line.replace(BS, BS + BS).replace(QUOTE, BS + QUOTE)
        parts.append(QUOTE + escaped + NEWLINE_ESCAPE + QUOTE + chr(10))
    parts.append(";" + chr(10))

    io.open(OUT, "w", encoding="utf-8", newline=chr(10)).write("".join(parts))
    print("  taa_live_default.h: %d active key(s) from config/taa_live.ini" % active)


if __name__ == "__main__":
    main()
