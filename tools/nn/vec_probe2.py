"""vec_probe2.py - vector and jitter check without a shared mask.

vec_probe.py multiplied both frames by the SAME airframe mask, so the mask's
own silhouette correlated at zero shift and dragged every sub-pixel estimate
toward 0. This one picks two texture-rich windows automatically - the busiest
64 px block that lies entirely on the near field (livery lettering) and the
busiest block entirely on mid-distance terrain - and correlates plain windowed
crops. Per pair it reports the true on-screen shift of each window, the vector
the resolve will use there, and colour-vs-resolved in the same frame (which is
the jitter, if the resolve un-jitters and the draws were jittered).

    python tools/nn/vec_probe2.py E:/NNCap/<session>
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vec_probe import load, luma, shift_between

def busiest(c, ok, bs=64):
    H, W = c.shape; Hb, Wb = H // bs, W // bs
    blk = c[:Hb * bs, :Wb * bs].reshape(Hb, bs, Wb, bs); var = blk.var(axis=(1, 3))
    frac = ok[:Hb * bs, :Wb * bs].reshape(Hb, bs, Wb, bs).mean(axis=(1, 3))
    score = np.where(frac > 0.98, var, -1.0)
    by, bx = np.unravel_index(np.argmax(score), score.shape)
    return by * bs + bs // 2, bx * bs + bs // 2, float(score.max())

def window(cy, cx, H, W, h=192, w=384):
    y0 = int(np.clip(cy - h // 2, 0, H - h)); x0 = int(np.clip(cx - w // 2, 0, W - w))
    return slice(y0, y0 + h), slice(x0, x0 + w)

def main():
    sess = sys.argv[1]
    frames = sorted(glob.glob(os.path.join(sess, "frame_*")))
    metas = [json.load(open(os.path.join(f, "manifest.json"))) for f in frames]
    print("%d frames, view %s" % (len(frames), metas[0].get("view_type")))
    print("pair   | AIRFRAME window: moved px    vel px       colour-vs-resolved | TERRAIN window: moved px    vel px       colour-vs-resolved | manifest jitter  delta")
    for i in range(1, len(frames)):
        m0, m1 = metas[i - 1], metas[i]
        if m1.get("burst", {}).get("index", 1) == 0: continue
        p = m1["planes"]; W, H = p["color"]["w"], p["color"]["h"]
        c0 = luma(load(frames[i - 1], "color", p["color"])); c1 = luma(load(frames[i], "color", p["color"]))
        r1 = luma(load(frames[i], "resolved", p["resolved"]))
        v = load(frames[i], "velocity", p["velocity"])
        d = np.fromfile(os.path.join(frames[i], "depth.bin"), np.float32).reshape(H, W)
        valid = (np.abs(v[..., 0]) < 1.0) & (np.abs(v[..., 1]) < 1.0) & (v[..., 3] > 0.5) & np.isfinite(d)
        dv = d[valid]
        top = np.percentile(dv, 99)                                         # the airframe cluster (reverse-Z: near = large)
        dd = np.nan_to_num(d, nan=-1.0)
        near = valid & (dd >= 0.5 * top)
        far = valid & (dd <= 0.25 * top) & (dd > 0)
        vpx = np.stack([v[..., 0] * W, -v[..., 1] * H], -1)
        out = ["%02d-%02d" % (i - 1, i)]
        for name, ok in (("air", near), ("ter", far)):
            cy, cx, sc = busiest(c1, ok)
            if sc < 0: out.append("%s: no clean block" % name); continue
            ys, xs = window(cy, cx, H, W)
            mx, my, _ = shift_between(c0[ys, xs], c1[ys, xs])
            jx, jy, _ = shift_between(c1[ys, xs], r1[ys, xs])
            mv = vpx[ys, xs][ok[ys, xs]].mean(0)
            out.append("(%+5.2f,%+5.2f)  (%+5.2f,%+5.2f)  (%+5.2f,%+5.2f)" % (mx, my, mv[0], mv[1], jx, jy))
        j0, j1 = m0["jitter"], m1["jitter"]
        out.append("(%+5.2f,%+5.2f)  (%+5.2f,%+5.2f)" % (j1["x"], j1["y"], j1["x"] - j0["x"], j1["y"] - j0["y"]))
        print(" | ".join(out))
    print("\nmoved: how the window's content shifted from the previous frame (px, +x right, +y down).\n"
          "vel: the injected vector there, sign convention as the trainer calibrated it (backward = -moved if exact).\n"
          "colour-vs-resolved: shift of the raw sample against the un-jittered output; equals the jitter when jitter is live.")

if __name__ == "__main__":
    main()
