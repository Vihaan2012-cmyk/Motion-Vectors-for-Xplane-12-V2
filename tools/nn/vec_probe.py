"""vec_probe.py - does the injected velocity match how the airframe actually moved?

For each consecutive frame pair in one capture burst: find the airframe (the
pixels whose vector is small - body path in chase view), measure its true
on-screen shift between the two colour planes by phase correlation, and compare
with the shift the velocity plane predicts. The residual is the vector bias the
resolve fetches history with. Reports both jitter-corrected and raw so the
convention question answers itself.

    python tools/nn/vec_probe.py E:/NNCap/<session>
"""
import glob, json, os, sys
import numpy as np

def load(fd, name, p):
    a = np.fromfile(os.path.join(fd, name + ".bin"), np.float16).reshape(p["h"], p["w"], p["c"]).astype(np.float32)
    return np.nan_to_num(a)

def luma(c): return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]

def shift_between(a, b):
    """Shift (dx, dy) that moves a onto b, sub-pixel via parabolic peak fit."""
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    A = np.fft.fft2((a - a.mean()) * win); B = np.fft.fft2((b - b.mean()) * win)
    R = A * np.conj(B); R /= np.abs(R) + 1e-9
    r = np.real(np.fft.ifft2(R))
    iy, ix = np.unravel_index(np.argmax(r), r.shape)
    def sub(m, c, n):
        l, rr = m[(c - 1) % n], m[(c + 1) % n]; d = (l - 2 * m[c] + rr)
        return c + (0.5 * (l - rr) / d if abs(d) > 1e-9 else 0.0)
    fy = sub(r[:, ix], iy, r.shape[0]); fx = sub(r[iy, :], ix, r.shape[1])
    if fy > r.shape[0] / 2: fy -= r.shape[0]
    if fx > r.shape[1] / 2: fx -= r.shape[1]
    return -fx, -fy, float(r.max())

def main():
    sess = sys.argv[1]
    frames = sorted(glob.glob(os.path.join(sess, "frame_*")))
    metas = [json.load(open(os.path.join(f, "manifest.json"))) for f in frames]
    print("%d frames, view %s" % (len(frames), metas[0].get("view_type")))
    print("%-6s %-22s %-22s %-22s %-22s %s" % ("pair", "measured px", "vel px (sx,sy=+,-)", "jitter delta px", "resid (jit-corr)", "peak"))
    for i in range(1, len(frames)):
        m0, m1 = metas[i - 1], metas[i]
        if m1.get("burst", {}).get("index", 1) == 0: continue          # burst boundary, not consecutive
        p = m1["planes"]; W, H = p["color"]["w"], p["color"]["h"]
        c0 = luma(load(frames[i - 1], "color", p["color"])); c1 = luma(load(frames[i], "color", p["color"]))
        v = load(frames[i], "velocity", p["velocity"])
        vpx = np.stack([v[..., 0] * W, -v[..., 1] * H], -1)                  # trainer-calibrated signs (+1, -1)
        valid = (np.abs(v[..., 0]) < 1.0) & (np.abs(v[..., 1]) < 1.0) & (v[..., 3] > 0.5)
        body = valid & (np.hypot(vpx[..., 0], vpx[..., 1]) < 2.0)
        if "depth" in p:                                                    # keep only the near field: the airframe, not far terrain
            d = load(frames[i], "depth", p["depth"])[..., 0] if p["depth"]["dtype"] == "f16" else \
                np.fromfile(os.path.join(frames[i], "depth.bin"), np.float32).reshape(p["depth"]["h"], p["depth"]["w"])
            # X-Plane depth is reverse-Z (measured: near = 1, far = 0) whatever the manifest flag says.
            valid &= np.isfinite(d); body &= np.isfinite(d)
            dv = d[valid]; near_cut = np.percentile(dv, 88)
            body &= np.nan_to_num(d, nan=-1.0) >= near_cut
            if i == 1: print("depth dtype %s nan%% %.1f  valid p1/p50/p99" % (p["depth"]["dtype"], 100.0 * (~np.isfinite(d)).mean()) + " %.5f %.5f %.5f  near cut %.5f" % (np.percentile(dv, 1), np.percentile(dv, 50), np.percentile(dv, 99), near_cut)); pass
            if False: print(": %.5f %.5f %.5f  near cut %.5f" % (np.percentile(dv, 1), np.percentile(dv, 50), np.percentile(dv, 99), near_cut))
        ys, xs = np.where(body)
        if len(ys) < 5000: print("%02d-%02d  airframe mask too small (%d px)" % (i - 1, i, len(ys))); continue
        y0, y1 = np.percentile(ys, [2, 98]).astype(int); x0, x1 = np.percentile(xs, [2, 98]).astype(int)
        y0, y1, x0, x1 = max(y0, 0), min(y1, H), max(x0, 0), min(x1, W)
        mask = body[y0:y1, x0:x1].astype(np.float32)
        a = c0[y0:y1, x0:x1] * mask; b = c1[y0:y1, x0:x1] * mask
        dx, dy, pk = shift_between(a, b)
        r1 = luma(load(frames[i], "resolved", p["resolved"]))[y0:y1, x0:x1] * mask    # same frame: colour vs resolved = the jitter, if any
        jx, jy, _ = shift_between(b, r1)
        mv = vpx[y0:y1, x0:x1][body[y0:y1, x0:x1]].mean(0)
        j0, j1 = m0["jitter"], m1["jitter"]; jd = (j1["x"] - j0["x"], j1["y"] - j0["y"])
        print("%02d-%02d  (%+6.2f,%+6.2f)        (%+6.2f,%+6.2f)        (%+6.2f,%+6.2f)        (%+6.2f,%+6.2f)        %.2f  crop %dx%d  colour-vs-resolved (%+5.2f,%+5.2f) manifest jitter (%+5.2f,%+5.2f)" %
              (i - 1, i, dx, dy, mv[0], mv[1], jd[0], jd[1], dx - jd[0] + mv[0], dy - jd[1] + mv[1], pk, x1 - x0, y1 - y0, jx, jy, j1["x"], j1["y"]))
    print("\nread: 'measured' is how the airframe crop actually moved frame-to-frame (px, +x right, +y down);\n"
          "'vel px' is the vector the resolve will use; 'resid' is measured minus jitter delta plus vel -\n"
          "zero means history is fetched exactly right, a constant offset is a bias, a value tracking\n"
          "'measured' means the vector missed the motion entirely.")

if __name__ == "__main__":
    main()
