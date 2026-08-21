"""Ground truth for the velocity field: what the image ACTUALLY did.

WHY THIS EXISTS

Every "the field is wrong by N times" claim in this project has come from
comparing the field against a matrix - that is, against the same belief that
produced it. When the belief is wrong, the check agrees with it. A 46x error
was reported that way and later retracted; the retraction found the assumption
(a field-of-view read as vertical when it is horizontal) sitting in BOTH the
field and the check.

Normalised cross-correlation does not share that assumption. It measures where
a patch of pixels MOVED TO between two rendered frames, using nothing but the
two images. If NCC and the velocity field disagree, the field is wrong,
whatever the matrix says.

  flow.py ncc  <frame_a.png> <frame_b.png>     true displacement, px/frame
  flow.py viz  <viz1_frame.png> <viz_scale>    the field's own claim, px/frame

Patches are chosen by local standard deviation, not by fixed coordinates. A
previous version sampled fixed points, landed on blank fuselage and flat sky,
reported every patch WEAK, and was read as "no flow" rather than "no texture".
"""

import sys
import math

try:
    from PIL import Image
except ImportError:
    sys.exit("needs Pillow")


def gray(path):
    im = Image.open(path).convert("L")
    return im, im.size


def srgb_to_linear(b):
    c = b / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def patches(im, w, h, n=12, half=24, search=40):
    """Pick the n most textured patches inside the ground region."""
    px = im.load()
    x0, x1 = int(w * 0.35), w - half - search - 1
    y0, y1 = int(h * 0.60), h - half - search - 1
    cands = []
    step = max(16, (x1 - x0) // 40)
    for cy in range(y0, y1, step):
        for cx in range(x0, x1, step):
            s = s2 = 0.0
            for dy in range(-half, half + 1, 6):
                for dx in range(-half, half + 1, 6):
                    v = px[cx + dx, cy + dy]
                    s += v
                    s2 += v * v
            k = (2 * (half // 6) + 1) ** 2
            var = s2 / k - (s / k) ** 2
            cands.append((var, cx, cy))
    cands.sort(reverse=True)
    return [(cx, cy) for _, cx, cy in cands[:n]]


def ncc_shift(a, b, cx, cy, half=24, search=40):
    """Best integer (dx,dy) aligning a's patch at (cx,cy) into b."""
    pa, pb = a.load(), b.load()
    ref = []
    for dy in range(-half, half + 1, 2):
        for dx in range(-half, half + 1, 2):
            ref.append(pa[cx + dx, cy + dy])
    n = len(ref)
    ma = sum(ref) / n
    va = math.sqrt(sum((v - ma) ** 2 for v in ref)) or 1e-6
    best, bdx, bdy = -2.0, 0, 0
    for sy in range(-search, search + 1):
        for sx in range(-search, search + 1):
            acc = []
            for dy in range(-half, half + 1, 2):
                row = cy + dy + sy
                for dx in range(-half, half + 1, 2):
                    acc.append(pb[cx + dx + sx, row])
            mb = sum(acc) / n
            vb = math.sqrt(sum((v - mb) ** 2 for v in acc)) or 1e-6
            num = sum((ref[i] - ma) * (acc[i] - mb) for i in range(n))
            c = num / (va * vb)
            if c > best:
                best, bdx, bdy = c, sx, sy
    return best, bdx, bdy


def cmd_ncc(pa, pb):
    a, (w, h) = gray(pa)
    b, _ = gray(pb)
    pts = patches(a, w, h)
    print("%-14s %-9s %-9s %-7s" % ("patch", "dx", "dy", "ncc"))
    mags = []
    for (cx, cy) in pts:
        c, dx, dy = ncc_shift(a, b, cx, cy)
        tag = "" if c > 0.5 else "   WEAK - ignored"
        print("(%5d,%5d) %8d %8d %7.3f%s" % (cx, cy, dx, dy, c, tag))
        if c > 0.5:
            mags.append(math.hypot(dx, dy))
    if mags:
        mags.sort()
        print("\n  patches used : %d" % len(mags))
        print("  median |flow|: %.2f px/frame" % mags[len(mags) // 2])
    else:
        print("\n  no patch correlated - no texture, or motion beyond the search radius")


def cmd_viz(path, scale):
    """Decode VIZ_MOTION: c = 0.5 + vel_px * 0.02 * vizScale, sRGB encoded."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    px = im.load()
    k = 0.02 * scale
    xs = []
    for y in range(int(h * 0.60), h, 4):
        for x in range(int(w * 0.35), w, 4):
            r, g, _ = px[x, y]
            vx = (srgb_to_linear(r) - 0.5) / k
            vy = (srgb_to_linear(g) - 0.5) / k
            xs.append(math.hypot(vx, vy))
    xs.sort()
    n = len(xs)
    print("  samples      : %d" % n)
    print("  p10 |vel|    : %.2f px/frame" % xs[n // 10])
    print("  median |vel| : %.2f px/frame" % xs[n // 2])
    print("  p90 |vel|    : %.2f px/frame" % xs[9 * n // 10])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    if sys.argv[1] == "ncc":
        cmd_ncc(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "viz":
        cmd_viz(sys.argv[2], float(sys.argv[3]))
    else:
        sys.exit(__doc__)
