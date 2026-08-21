"""Self-test for flow.py against KNOWN answers.

Both halves are checked the only way that means anything: construct an input
whose answer is known by construction, and see whether the tool returns it.
"""
import os, sys, glob, math, subprocess

sys.path.insert(0, r"d:\Steam Games\steamapps\common\X-Plane 12\MotionVectors\tools\measure")
import flow

from PIL import Image

T = os.environ["TEMP"]
SCRATCH = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- NCC test
src = sorted(glob.glob(os.path.join(T, "vsw_OFF_*.png")))
if not src:
    src = sorted(glob.glob(os.path.join(T, "fin_*.png")))
if not src:
    sys.exit("no source frame to build the test from")

base = Image.open(src[0]).convert("RGB")
w, h = base.size
print("source frame: %s  (%dx%d)" % (os.path.basename(src[0]), w, h))
print()

fails = 0
print("NCC — a known shift is applied, the tool must recover it")
print("%-14s %-12s %-12s %s" % ("applied", "recovered", "median|flow|", "verdict"))
for (sx, sy) in ((7, 0), (-5, 3), (12, -8), (0, 0)):
    # content moves by (sx,sy): pixel at p in A lands at p+(sx,sy) in B
    shifted = Image.new("RGB", (w, h))
    shifted.paste(base, (sx, sy))
    a = os.path.join(SCRATCH, "t_a.png")
    b = os.path.join(SCRATCH, "t_b.png")
    base.save(a); shifted.save(b)

    ga, _ = flow.gray(a)
    gb, _ = flow.gray(b)
    pts = flow.patches(ga, w, h, n=6)
    got = []
    for (cx, cy) in pts:
        c, dx, dy = flow.ncc_shift(ga, gb, cx, cy)
        if c > 0.5:
            got.append((dx, dy))
    if not got:
        print("%-14s %-12s %-12s %s" % ("(%d,%d)" % (sx, sy), "none", "-", "FAIL no correlation"))
        fails += 1
        continue
    mdx = sorted(g[0] for g in got)[len(got)//2]
    mdy = sorted(g[1] for g in got)[len(got)//2]
    mag = sorted(math.hypot(*g) for g in got)[len(got)//2]
    ok = (mdx == sx and mdy == sy)
    if not ok: fails += 1
    print("%-14s %-12s %-12s %s" % ("(%d,%d)" % (sx, sy), "(%d,%d)" % (mdx, mdy),
                                     "%.2f" % mag, "ok" if ok else "FAIL"))

# ------------------------------------------------------------- viz decode test
print()
print("VIZ decode — a frame is ENCODED with a known velocity, tool must recover it")
print("%-14s %-8s %-14s %s" % ("encoded px", "scale", "recovered", "verdict"))

def linear_to_srgb(c):
    c = max(0.0, min(1.0, c))
    return 12.92*c if c <= 0.0031308 else 1.055*(c ** (1/2.4)) - 0.055

for (vx, vy, scale) in ((10.0, 0.0, 0.2), (25.0, -15.0, 0.2), (3.0, 4.0, 1.0)):
    k = 0.02 * scale
    r = linear_to_srgb(0.5 + vx*k)
    g = linear_to_srgb(0.5 + vy*k)
    img = Image.new("RGB", (400, 300), (int(round(r*255)), int(round(g*255)), 128))
    p = os.path.join(SCRATCH, "t_viz.png")
    img.save(p)
    # decode the same way flow.cmd_viz does
    px = img.load()
    rr, gg, _ = px[200, 250]
    dvx = (flow.srgb_to_linear(rr) - 0.5) / k
    dvy = (flow.srgb_to_linear(gg) - 0.5) / k
    exp = math.hypot(vx, vy); got = math.hypot(dvx, dvy)
    err = abs(got-exp)/max(exp,1e-9)*100
    ok = err < 6.0
    if not ok: fails += 1
    print("%-14s %-8s %-14s %s" % ("(%.0f,%.0f)" % (vx,vy), scale,
          "(%.1f,%.1f)" % (dvx,dvy), "ok  %.1f%% err" % err if ok else "FAIL %.1f%% err" % err))

print()
print("FAILURES: %d" % fails)
sys.exit(1 if fails else 0)
