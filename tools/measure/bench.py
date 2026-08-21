"""Turn one bench run into a verdict.

WHAT THIS IS FOR

Every line is OK, BAD, or VOID, and VOID is the important one. This project has
repeatedly reported a number that could not have meant anything - a velocity
field measured while the camera was parked, a detail ratio of 167% from a view
change mid-capture, a clamp distance read through a visualisation that
overwrites the history it is measuring. A bench that prints a number for those
is worse than one that prints nothing, because a number gets believed.

So each section states its own preconditions, checks them, and refuses rather
than guesses. The final block says what the run can and cannot judge.

    python bench.py <capture-dir>

Exit code 0 when nothing is BAD, 1 otherwise. VOID does not fail the run - a
question that could not be asked is not a failing answer.
"""

import os
import re
import sys
import glob

try:
    import numpy as np
    from PIL import Image
except ImportError:
    sys.exit("needs numpy and Pillow")


OK, BAD, VOID = "OK  ", "BAD ", "VOID"
_verdicts = []


def line(state, what, detail=""):
    _verdicts.append(state)
    print("  %s %-38s %s" % (state, what, detail))


def section(name):
    print("")
    print("-- %s" % name)


# --------------------------------------------------------------- image maths
def load(paths):
    return [np.asarray(Image.open(p).convert("L"), dtype=np.float64) for p in paths]


def crop(a):
    """Lower-centre of the frame: ground and airframe, away from the 2-D panel
    which is drawn after the resolve and would dilute every statistic."""
    h, w = a.shape
    return a[int(h * 0.30):int(h * 0.75), int(w * 0.30):]


def flicker(imgs):
    c = [crop(a) for a in imgs]
    if len(c) < 2:
        return None
    return float(np.median([np.mean(np.abs(c[i] - c[i - 1]))
                            for i in range(1, len(c))]))


def detail(imgs):
    c = [crop(a) for a in imgs]
    if not c:
        return None
    return float(np.median([np.mean(np.diff(x, axis=1) ** 2) for x in c]))


def rounds(d, tag):
    """Per-round statistics, so rounds can be compared like with like."""
    out = []
    for r in range(5):
        fs = sorted(glob.glob(os.path.join(d, "park_%s_%d_*.png" % (tag, r))))
        if len(fs) < 2:
            continue
        imgs = load(fs)
        out.append((flicker(imgs), detail(imgs)))
    return out


# --------------------------------------------------------------- trace
def trace_text(d):
    p = os.path.join(d, "trace.txt")
    if not os.path.exists(p):
        return ""
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def last(tx, pattern):
    m = re.findall(pattern, tx)
    return m[-1] if m else None


def main(d):
    tx = trace_text(d)
    print("bench report")
    print("  captures : %s" % d)
    print("  trace    : %d KB" % (len(tx) // 1024))

    # ---- AN ABSENT TRACE IS NOT A PASS.
    #
    # Caught by running this against an empty directory: with no trace, every
    # "did something bad appear?" check found nothing and reported OK. "No
    # refused passes" and "no freeze events" read as a clean bill of health for
    # a run that produced no evidence whatsoever.
    #
    # That is the same error this whole bench exists to prevent, so the checks
    # that can only be answered FROM the trace are VOID without one.
    haveTrace = len(tx) > 2000
    if not haveTrace:
        print("")
        print("-- NO USABLE TRACE")
        line(VOID, "layer trace", "missing or too short - every trace-derived")
        print("       check below is VOID, not passing")

    # ------------------------------------------------------ preconditions
    section("preconditions - nothing below means anything without these")

    if not haveTrace:
        line(VOID, "shader injection armed", "no trace")
    elif "SPIRV INJECT: armed" in tx:
        line(OK, "shader injection armed")
    else:
        line(BAD, "shader injection armed", "not armed - no velocity is produced")

    if not haveTrace:
        line(VOID, "velocity pass armed", "no trace")
    elif re.search(r"VEL: velocity pass ARMED", tx):
        line(OK, "velocity pass armed")
    else:
        line(BAD, "velocity pass armed", "disarmed - the field is empty")

    gates = re.findall(r"TAA GATE: scenePass=1.*?resolved=(\d)", tx) if haveTrace else []
    if not haveTrace:
        line(VOID, "resolve is running", "no trace")
    elif gates:
        duty = 100.0 * sum(1 for g in gates if g == "1") / len(gates)
        line(OK if duty > 5 else BAD, "resolve is running",
             "%.0f%% of sampled scene passes" % duty)
    else:
        line(BAD, "resolve is running", "no scene passes seen at all")

    # ------------------------------------------------------ patching
    section("pipeline patching")

    m = last(tx, r"pipelines by patch outcome - both (\d+), vertex only (\d+), "
                 r"fragment only (\d+), NEITHER (\d+), declined-by-design (\d+) \(of (\d+)\)")
    if m:
        both, vonly, fonly, neither, declined, total = (int(x) for x in m)
        line(OK if both > total * 0.5 else BAD, "pipelines fully patched",
             "%d of %d" % (both, total))
        line(OK if neither == 0 else BAD, "geometry with neither stage", str(neither))
        # Declined must not be zero: zero means the rule stopped excluding the
        # genuine full-screen quads, and a full-screen pass that writes velocity
        # stamps one screen-space vector across the entire frame.
        line(OK if 0 < declined < total * 0.2 else BAD, "declined-by-design",
             "%d (zero would mean quads are being patched)" % declined)
    else:
        line(VOID, "pipeline patch outcome", "not reported in this run")

    # ------------------------------------------------------ passes
    section("velocity target binding")

    bound = sorted(set(re.findall(r"pass shape (\d+x\d+) colour=(\d+) depth=yes -> SCENE", tx))) if haveTrace else []
    if not haveTrace:
        line(VOID, "scene passes bound", "no trace")
    elif bound:
        line(OK if len(bound) >= 2 else BAD, "scene passes bound",
             ", ".join("%s c=%s" % (s, c) for s, c in bound))
    else:
        line(BAD, "scene passes bound", "none - terrain cannot write velocity")

    if haveTrace:
        refused = re.findall(r"MV STICKY: REFUSING pass colour=(\d+)", tx)
        line(OK if not refused else BAD, "passes refused by the colour latch",
             "none" if not refused else ", ".join(sorted(set(refused))))
        freezes = len(re.findall(r"MV FREEZE: no pass bound", tx))
        line(OK if freezes < 5 else BAD, "frames with no velocity target", str(freezes))
    else:
        line(VOID, "passes refused by the colour latch", "no trace")
        line(VOID, "frames with no velocity target", "no trace")

    # ------------------------------------------------------ parked quality
    section("parked image quality (TAA on vs off, interleaved)")

    on, off = rounds(d, "on"), rounds(d, "off")
    n = min(len(on), len(off))
    if n < 3:
        line(VOID, "parked comparison", "fewer than 3 usable rounds")
    else:
        offd = [o[1] for o in off[:n]]
        spread = max(offd) / max(min(offd), 1e-9)
        if spread > 1.3:
            # The TAA-off frames are the control. If the control itself moved,
            # the scene changed during the run and no comparison against it is
            # meaningful. This is the check that "167% detail retained" needed.
            line(VOID, "scene stability during capture",
                 "TAA-off detail moved %.0f%% - scene drifted, comparison void" %
                 ((spread - 1) * 100))
        else:
            line(OK, "scene stability during capture", "control steady")
            fr = np.median([on[i][0] / max(off[i][0], 1e-9) for i in range(n)])
            dr = np.median([on[i][1] / max(off[i][1], 1e-9) for i in range(n)])
            # TAA should be no LESS stable than not running it. Above ~2x it is
            # adding shimmer rather than removing it, which defeats the point.
            line(OK if fr < 2.0 else BAD, "temporal stability vs TAA-off",
                 "%.2fx flicker (want < 2.0)" % fr)
            line(OK if dr > 0.80 else BAD, "detail retained vs TAA-off",
                 "%.0f%% (want > 80%%)" % (dr * 100))

    # ------------------------------------------------------ velocity field
    section("velocity field under motion")

    mags = sorted(glob.glob(os.path.join(d, "mag_*.png")),
                  key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    if not mags:
        line(VOID, "velocity field", "no motion capture in this run (-Quick?)")
    else:
        stats = []
        for p in mags:
            im = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)
            h, w, _ = im.shape
            s = im[int(h * 0.60):, int(w * 0.35):, :]
            lum = 0.299 * s[:, :, 0] + 0.587 * s[:, :, 1] + 0.114 * s[:, :, 2]
            red = np.mean((s[:, :, 0] > 120) & (s[:, :, 1] < 80) & (s[:, :, 2] < 80))
            stats.append((float(np.mean(lum > 12)), float(red), float(np.max(lum))))

        moving = [s for s in stats if s[2] > 20]
        if not moving:
            # The camera never moved, so a black magnitude map means nothing.
            # Reporting "0 coverage" here would be the exact false finding that
            # cost this project an evening.
            line(VOID, "velocity field",
                 "no motion in %d frames - the camera did not move" % len(stats))
        else:
            cov = np.median([s[0] for s in moving]) * 100
            red = np.median([s[1] for s in moving]) * 100
            line(OK if cov > 60 else BAD, "ground writes velocity",
                 "%.0f%% coverage over %d moving frames (want > 60%%)" %
                 (cov, len(moving)))
            # A full-screen pass writing velocity saturates the whole frame.
            line(OK if red < 20 else BAD, "no full-screen stamp",
                 "%.0f%% saturated (want < 20%%)" % red)

    # ------------------------------------------------------ vram
    section("vram")

    m = last(tx, r"zone (\w+)\s+heap ([\d.]+) GB\s+raw budget ([\d.]+) GB\s+"
                 r"usage ([\d.]+) GB\s+shaped ([\d.]+) GB")
    if m:
        zone, heap, raw, usage, shaped = m
        line(OK, "zone", "%s - usage %s of %s GB shaped (heap %s)" %
             (zone, usage, shaped, heap))
    else:
        line(VOID, "zone", "no vram report in this run")

    m = last(tx, r"allocs (\d+)\s+frees (\d+)\s+failures (\d+)\s+worst latency (\d+) us")
    if m:
        allocs, frees, fails, worst = (int(x) for x in m)
        line(OK if fails == 0 else BAD, "allocation failures", str(fails))
        line(OK, "allocation churn",
             "%d allocs, %d frees, worst %.1f ms" % (allocs, frees, worst / 1000.0))
    else:
        line(VOID, "allocations", "no vram report in this run")

    m = last(tx, r"recycle: (\d+) hits\s+(\d+) misses")
    if m:
        hits, misses = int(m[0]), int(m[1])
        if hits + misses == 0:
            # This is the exact reading that hid a dead subsystem: zero and zero
            # means never consulted, not consulted-and-empty.
            line(BAD, "recycle pool", "never consulted (0 hits AND 0 misses)")
        else:
            line(OK, "recycle pool", "%d hits, %d misses" % (hits, misses))
    else:
        line(VOID, "recycle pool", "no vram report in this run")

    m = last(tx, r"recycle skips: dead (\d+)\s+disabled (\d+)\s+pNext (\d+)")
    if m:
        line(OK, "recycle skips", "dead %s, disabled %s, pNext %s" % m)

    # ------------------------------------------------------ performance
    section("performance")

    m = last(tx, r"frame time: avg ([\d.]+) ms\s+1%low ([\d.]+) ms\s+"
                 r"0\.1%low ([\d.]+) ms")
    if m:
        avg, low1, low01 = (float(x) for x in m)
        line(OK, "frame time",
             "avg %.1f ms (%.0f fps), 1%% low %.1f, 0.1%% low %.1f" %
             (avg, 1000.0 / max(avg, 1e-6), low1, low01))
        line(OK if low01 < avg * 2.0 else BAD, "frame consistency",
             "0.1%% low is %.1fx the average (want < 2.0)" % (low01 / max(avg, 1e-6)))
    else:
        m = last(tx, r"FRAME BUDGET: ([\d.]+) fps")
        if m:
            line(OK, "frame rate", "%s fps" % m)
        else:
            line(VOID, "frame time", "no budget line in this run")

    # ------------------------------------------------------ verdict
    print("")
    print("-- verdict")
    bad = _verdicts.count(BAD)
    void = _verdicts.count(VOID)
    print("  %d OK, %d BAD, %d VOID" % (_verdicts.count(OK), bad, void))
    if void:
        print("  VOID means the question could not be asked in this run, not")
        print("  that the answer was bad. Re-run covering those conditions")
        print("  before concluding anything about them.")
    print("  %s" % ("FAIL - see BAD above" if bad else "PASS"))
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
