#!/usr/bin/env python3
"""pack_aa.py - training crops for the 1x (native-resolution) anti-aliasing net.

The captured resolve is the best single 1440p frame on disk, so a net trained toward it can
only tie it. A burst holds a better teacher: consecutive raw jittered rasters of the same
scene (the sweep steps the clock BETWEEN bursts, never inside one) with exact per-pixel motion
vectors between them. Warping every earlier raster of the burst onto a frame and averaging is a
multi-sample supersampled truth - one interpolation per sample instead of the resolve's
recursive re-fetch blur - from data already on disk.

Conventions, MEASURED on the 2026-09-11 capture (scratch jitstatic/jitab/jitnf):
  * a static burst shifts by exactly -(j_b - j_a) between consecutive rasters, so the raster of
    frame i holds the scene displaced by -j_i: the unjittered scene at pixel centre x is
    raster_i fetched at x + j_i (manifest jitter, pixels). That is the shader's S = uv + S.
  * the injected vectors are jitter-free and (prevNDC - currNDC) * 0.5 = a UV offset;
    tc.calibrate_warp picks their screen sign. Warping by them and THEN applying each frame's
    own +j_i offset aligns jittered geometry ("B" below).
  * passes the injector never touches (night light billboards at least) are NOT jittered:
    for them the warp alone aligns and the offset must stay the target's +j_t ("A").
    The choice is made per pixel from a box-smoothed agreement test, so a night city and
    the daytime terrain both come out aligned.

Per burst the LAST K=3 frames become one sequence:
  raw   (K,3,S,S)  unjittered bicubic fetch of the frame's own raster: aliased single sample = the input
  gt    (K,3,S,S)  accumulated truth: mean over burst frames 0..t warped into t; a sample is dropped
                   where its warp leaves the screen, crosses an unwritten vector, or falls outside
                   the 3x3 tonemapped range of the current frame by more than --tol (disocclusion);
                   the current frame always counts, so no pixel needs a fallback
  ns    (K,S,S)    samples per pixel behind gt;  fb (K,S,S) fraction of them taken under "B"
  aux, vel, hist0, dims  as pack_captures.py (train_recurrent.seq_tensors)
  jit   (K,2)      the manifest jitter j_t in pixels = the shader's S; flip negates x
  sign  (2,)       the calibrated velocity screen sign, so the trainer's warp_crop matches

    python tools/nn/pack_aa.py --root E:/NNCap_fresh --out E:/NNCap/packed_aa --crops 4 --size 384
    python tools/nn/pack_aa.py --check          # per-pair A/B/C errors on a few bursts, no packing

Copyright (C) 2026 MotionVectors contributors. SPDX: GPL-3.0-or-later
"""
import argparse, glob, json, os, random, sys, time
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quick_train import tm, luma
import train_captures as tc
import train_recurrent as tr

K = 3

def raster_linear(f, m):
    c = tc.load_plane(f, m, "color").astype(np.float32)[..., :3]
    return np.clip(np.nan_to_num(c, nan=0.0, posinf=64.0, neginf=0.0), 0.0, 64.0)

class Frame:
    """One burst frame's rasters on the device: colour (3,H,W), vel (2,H,W) in UV, valid (1,H,W), jitter (UV, px)."""
    def __init__(self, f, m, dev):
        self.col = torch.from_numpy(raster_linear(f, m).transpose(2, 0, 1)).to(dev)
        v = tc.load_plane(f, m, "velocity").astype(np.float32)
        vx = np.nan_to_num(v[..., 0]); vy = np.nan_to_num(v[..., 1])
        bad = (np.abs(vx) > tc.VEL_SENTINEL) | (np.abs(vy) > tc.VEL_SENTINEL)
        vx = np.where(bad, 0.0, vx).astype(np.float32); vy = np.where(bad, 0.0, vy).astype(np.float32)
        self.vel = torch.from_numpy(np.stack([vx, vy], 0)).to(dev)
        self.valid = torch.from_numpy((~bad).astype(np.float32)[None]).to(dev)
        self.H, self.W = self.col.shape[-2:]
        j = m.get("jitter", {}) or {}
        sc = float(j.get("scale", 1.0))
        self.jpx = (float(j.get("x", 0.0)) * sc, float(j.get("y", 0.0)) * sc)
        self.juv = (self.jpx[0] / self.W, self.jpx[1] / self.H)

def sample(img, gu, gv, mode="bilinear"):
    """img (C,H,W) sampled at UV grids (H,W); returns (C,H,W) and the in-bounds mask (H,W)."""
    grid = torch.stack([gu * 2 - 1, gv * 2 - 1], dim=-1).unsqueeze(0)
    out = F.grid_sample(img.unsqueeze(0), grid, mode=mode, padding_mode="zeros", align_corners=False)[0]
    inb = ((gu >= 0) & (gu <= 1) & (gv >= 0) & (gv <= 1)).float()
    return out, inb

def centres(H, W, dev):
    uu = (torch.arange(W, dtype=torch.float32, device=dev) + 0.5) / W
    vv = (torch.arange(H, dtype=torch.float32, device=dev) + 0.5) / H
    return uu.view(1, W).expand(H, W).contiguous(), vv.view(H, 1).expand(H, W).contiguous()

def box(e, win):
    return F.avg_pool2d(e[None, None], win, 1, win // 2, count_include_pad=False)[0, 0]

def accumulate(frames, t, tol, dev, mode="bicubic", win=31):
    """Truth for frame t: mean of frames 0..t warped into t.
    Returns (truth (3,H,W), count (H,W), raw (3,H,W), fracB (H,W))."""
    ft = frames[t]; H, W = ft.H, ft.W
    U, V = centres(H, W, dev)
    sx, sy = tc._SIGN
    tx, ty = ft.juv
    raw, _ = sample(ft.col, U + tx, V + ty, mode); raw = raw.clamp_min(0.0)
    cur_bl, _ = sample(ft.col, U + tx, V + ty, "bilinear")
    ctm = tm(cur_bl.unsqueeze(0))[0]
    # acceptance range: the RASTER's own 3x3 min/max around the unjittered position (nearest), not the
    # resampled current frame - a bilinear fetch narrows the range and rejects legitimate edge samples
    rtm = tm(ft.col.unsqueeze(0))
    lo = sample(-F.max_pool2d(-rtm, 3, 1, 1)[0], U + tx, V + ty, "nearest")[0] - tol
    hi = sample(F.max_pool2d(rtm, 3, 1, 1)[0], U + tx, V + ty, "nearest")[0] + tol
    acc = raw.clone(); cnt = torch.ones(H, W, device=dev); nb = torch.zeros(H, W, device=dev)
    # displacement (UV, unjittered coordinates) from frame t's pixel centre back to frame i, composed step by step
    du = torch.zeros(H, W, device=dev); dv = torch.zeros(H, W, device=dev); ok = torch.ones(H, W, device=dev)
    for i in range(t - 1, -1, -1):
        nxt = frames[i + 1]; nx, ny = nxt.juv
        # vectors of frame i+1, read in its own raster at the unjittered position, carry pixel(i+1) -> pixel(i)
        step, inb1 = sample(nxt.vel, U + du + nx, V + dv + ny)
        vval, _ = sample(nxt.valid, U + du + nx, V + dv + ny)
        du = du + sx * step[0]; dv = dv + sy * step[1]
        ok = ok * inb1 * (vval[0] > 0.5).float()
        fi = frames[i]; ix, iy = fi.juv
        wA, inbA = sample(fi.col, U + du + tx, V + dv + ty, "bilinear")     # A: not jittered (lights) - target offset
        wB, inbB = sample(fi.col, U + du + ix, V + dv + iy, "bilinear")     # B: jittered geometry - its own offset
        eA = box((tm(wA.unsqueeze(0))[0] - ctm).abs().mean(0), win)
        eB = box((tm(wB.unsqueeze(0))[0] - ctm).abs().mean(0), win)
        useB = (eB < eA).float()
        w_bl = wB * useB + wA * (1.0 - useB); inb = inbB * useB + inbA * (1.0 - useB)
        wtm = tm(w_bl.unsqueeze(0))[0]
        agree = ((wtm >= lo) & (wtm <= hi)).all(0).float()
        m = ok * inb * agree
        if mode == "bilinear":
            w = w_bl
        else:
            cA = sample(fi.col, U + du + tx, V + dv + ty, mode)[0]; cB = sample(fi.col, U + du + ix, V + dv + iy, mode)[0]
            w = cB * useB + cA * (1.0 - useB)
        acc = acc + w.clamp_min(0.0) * m; cnt = cnt + m; nb = nb + useB * m
    return acc / cnt, cnt, raw, nb / (cnt - 1.0).clamp_min(1.0)

def check(bl, dev, log=print):
    """Per consecutive pair: alignment error with the warp alone (A), warp + own jitter offset (B), warp + the
    wrong sign (C). Jittered geometry wants B, unjittered passes A; C must never win."""
    picks = [bl[i][2] for i in range(1, len(bl), max(1, len(bl) // 6))][:6]
    sx, sy = tc._SIGN
    for frames in picks:
        n = len(frames); fr = [Frame(f, m, dev) for f, m in frames]
        H, W = fr[0].H, fr[0].W; U, V = centres(H, W, dev); row = []
        for b in range(1, n):
            a = b - 1; ctm = tm(fr[b].col.unsqueeze(0))
            gx = (ctm[0, :, :, 2:] - ctm[0, :, :, :-2]).abs().mean(0); gy = (ctm[0, :, 2:, :] - ctm[0, :, :-2, :]).abs().mean(0)
            edge = torch.zeros(H, W, device=dev); edge[:, 1:-1] += gx; edge[1:-1, :] += gy
            emask = (edge > 0.05).float() * fr[b].valid[0]
            def err_at(dx, dy):
                s_, inb = sample(fr[a].col, U + sx * fr[b].vel[0] + dx / W, V + sy * fr[b].vel[1] + dy / H); mm = inb * emask
                return float(((tm(s_.unsqueeze(0)) - ctm).abs().mean(1)[0] * mm).sum() / mm.sum().clamp_min(1))
            djx, djy = fr[b].jpx[0] - fr[a].jpx[0], fr[b].jpx[1] - fr[a].jpx[1]
            eA, eB, eC = err_at(0.0, 0.0), err_at(-djx, -djy), err_at(djx, djy)
            row.append("%s %.4f/%.4f/%.4f" % ("A" if eA < eB else "B", eA, eB, eC) if eC > min(eA, eB) else "C!! %.4f/%.4f/%.4f" % (eA, eB, eC))
        log("%s: %s" % (os.path.basename(frames[-1][0]), " | ".join(row)))

def bursts(root, min_len):
    out = []
    for sess in sorted(glob.glob(os.path.join(root, "*"))):
        cur, cur_id = [], None
        for f in sorted(glob.glob(os.path.join(sess, "frame_*"))):
            mp = os.path.join(f, "manifest.json")
            if not os.path.exists(mp): continue
            m = json.load(open(mp))
            if not all(k in m["planes"] for k in ("color", "resolved", "normal", "velocity")): continue
            if not tr.capture_ok(m): continue
            b = m.get("burst") or {}
            if cur and b.get("id") == cur_id and b.get("index") == len(cur): cur.append((f, m))
            else:
                if len(cur) >= max(min_len, K + 1): out.append((os.path.basename(sess), cur_id, cur))
                cur, cur_id = [(f, m)], b.get("id")
        if len(cur) >= max(min_len, K + 1): out.append((os.path.basename(sess), cur_id, cur))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="E:/NNCap_fresh"); ap.add_argument("--out", default="E:/NNCap/packed_aa")
    ap.add_argument("--crops", type=int, default=4); ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--tol", type=float, default=0.03, help="tonemapped slack around the current frame's 3x3 range for a warped sample to count")
    ap.add_argument("--win", type=int, default=31, help="box window of the per-pixel A/B decision")
    ap.add_argument("--min-len", type=int, default=6, help="skip bursts shorter than this (few samples = weak truth)")
    ap.add_argument("--held", type=float, default=0.1); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    roots = [r.strip() for r in a.root.split(";")]
    tc.calibrate_warp(tc.list_frames(roots))
    bl = [b for r in roots for b in bursts(r, a.min_len)]
    if not bl: print("no bursts of length >= %d under %s" % (a.min_len, a.root)); sys.exit(2)
    lens = [len(b[2]) for b in bl]
    print("bursts: %d, lengths %d..%d (mean %.1f)" % (len(bl), min(lens), max(lens), sum(lens) / len(lens)), flush=True)
    if a.check: check(bl, dev); return
    if a.limit: bl = bl[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    groups = sorted(set("%s/%s" % (s, i) for s, i, _ in bl))
    held = set(random.Random(7).sample(groups, max(1, int(len(groups) * a.held))))
    print("packing %d bursts (%d held out) -> %s, device %s" % (len(bl), len(held), a.out, dev), flush=True)
    t0 = time.time(); written = 0; S = a.size; nsamp = []; fracb = []
    for bi, (sess, bid, fl) in enumerate(bl):
        tag = "held" if "%s/%s" % (sess, bid) in held else "train"
        if all(os.path.exists(os.path.join(a.out, "seq%05d_c%d_%s.npz" % (bi, c, tag))) for c in range(a.crops)): continue
        n = len(fl); win = fl[n - K:]; prev = fl[n - K - 1][0]
        _, aux, vel, hist0 = tr.seq_tensors(win, prev)
        frames = [Frame(f, m, dev) for f, m in fl]
        truths, counts, raws, fbs, jits = [], [], [], [], []
        for k in range(K):
            t = n - K + k
            truth, cnt, raw, fb = accumulate(frames, t, a.tol, dev, win=a.win)
            truths.append(truth.cpu()); counts.append(cnt.cpu()); raws.append(raw.cpu()); fbs.append(fb.cpu())
            nsamp.append(float(cnt.mean())); fracb.append(float(fb.mean()))
            jits.append([frames[t].jpx[0], frames[t].jpx[1]])
        del frames
        gt = torch.stack(truths); ns = torch.stack(counts); raw = torch.stack(raws); fb = torch.stack(fbs); jit = torch.tensor(jits, dtype=torch.float32)
        _, _, H, W = gt.shape
        rng = random.Random(1000 + bi)
        for c in range(a.crops):
            for _ in range(8):
                y, x = rng.randrange(H - S + 1), rng.randrange(W - S + 1)
                if float(luma(tm(gt[-1:, :, y:y + S, x:x + S])).mean()) >= tr.LUMA_FLOOR: break
            np.savez(os.path.join(a.out, "seq%05d_c%d_%s.npz" % (bi, c, tag)),
                     gt=gt[:, :, y:y + S, x:x + S].numpy().astype(np.float16),
                     raw=raw[:, :, y:y + S, x:x + S].numpy().astype(np.float16),
                     ns=ns[:, y:y + S, x:x + S].numpy().astype(np.float16),
                     fb=fb[:, y:y + S, x:x + S].numpy().astype(np.float16),
                     aux=aux[:, :, y:y + S, x:x + S].numpy().astype(np.float16),
                     vel=vel[:, :, y:y + S, x:x + S].numpy().astype(np.float16),
                     hist0=hist0[:, y:y + S, x:x + S].numpy().astype(np.float16),
                     jit=jit.numpy(), sign=np.array(tc._SIGN, dtype=np.float32), dims=np.array([W, H], dtype=np.float32))
            written += 1
        if bi % 10 == 0:
            print("  %d/%d bursts, %d crops, mean samples/pixel %.1f, B fraction %.2f, %.0fs" % (bi + 1, len(bl), written, sum(nsamp) / max(len(nsamp), 1), sum(fracb) / max(len(fracb), 1), time.time() - t0), flush=True)
    print("done: %d crops in %.0fs, mean samples/pixel %.1f, B fraction %.2f" % (written, time.time() - t0, sum(nsamp) / max(len(nsamp), 1), sum(fracb) / max(len(fracb), 1)), flush=True)

if __name__ == "__main__":
    main()
