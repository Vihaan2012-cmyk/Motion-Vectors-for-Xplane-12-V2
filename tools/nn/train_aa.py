#!/usr/bin/env python3
"""train_aa.py - the 1x (native-resolution) anti-aliasing net on pack_aa.py crops.

Same ResidualSR weights and recurrent unroll as train_recurrent.py, but the input is the
frame's own unjittered raster fetch (aliased, one sample) instead of a synthetic degrade, the
ratio plane is 1.0, the jitter planes carry each frame's REAL jitter (per sample, not per
batch), and the target is the burst-accumulated supersampled truth. The quiz measures the PSNR
gain of the output over the raw fetch against that truth, the Laplacian ratio, and the temporal
consistency of the net's own stream. The upscaler's cross-phase stability test does not apply:
the jitter is fixed per captured frame.

    python tools/nn/train_aa.py --packed E:/NNCap/packed_aa --ch 8,16,24 --rounds 24
    python tools/nn/train_aa.py --packed E:/NNCap/packed_aa --quiz --ckpt tools/nn/ckpt/aa_budget.pt

Copyright (C) 2026 MotionVectors contributors. SPDX: GPL-3.0-or-later
"""
import argparse, glob, os, random, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quick_train import ResidualSR, tm, luma, lap_energy, psnr
import train_captures as tc
import train_recurrent as tr

K = 3
IN_CH = 3 + 3 + 7 + 4      # tm(raw) rgb, jitter x/y + ratio planes, aux(6)+dz, warped history rgb + valid

class AANet(ResidualSR):
    """ResidualSR with per-sample jitter planes (jit: (B,2) pixels) and the ratio plane fixed at 1.0."""
    def forward(self, base_lin, jit, ratio=1.0, aux=None):
        B, _, H, W = base_lin.shape
        x = tm(base_lin)
        jx = jit[:, 0].to(x.dtype).view(B, 1, 1, 1).expand(B, 1, H, W)
        jy = jit[:, 1].to(x.dtype).view(B, 1, 1, 1).expand(B, 1, H, W)
        rp = torch.full((B, 1, H, W), ratio - 1.5, device=x.device, dtype=x.dtype)
        x = torch.cat([x, jx, jy, rp], dim=1)
        if aux is not None: x = torch.cat([x, aux.to(x.dtype)], dim=1)
        e0 = self.e0(x); e1 = self.e1(F.avg_pool2d(e0, 2)); e2 = self.e2(F.avg_pool2d(e1, 2))
        d1 = self.d1(torch.cat([F.interpolate(e2, scale_factor=2, mode="bilinear", align_corners=False), e1], 1))
        d0 = self.d0(torch.cat([F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False), e0], 1))
        res = torch.tanh(self.out(d0)) * (0.5 + base_lin.abs())
        return base_lin + res, res

KEYS = ("gt", "raw", "aux", "vel", "hist0", "jit", "dims")

class AACrops(torch.utils.data.Dataset):
    def __init__(self, files, n=4000, seed=0):
        self.files, self.n, self.seed = files, n, seed
    def __len__(self): return self.n
    def __getitem__(self, k):
        rng = random.Random(self.seed * 100003 + k)
        z = np.load(self.files[rng.randrange(len(self.files))])
        g, r, a, v, h, j, d = [torch.from_numpy(z[key]) for key in KEYS]
        if rng.random() < 0.5:
            g = g.flip(-1); r = r.flip(-1); a = a.flip(-1); v = v.flip(-1); h = h.flip(-1)
            a[:, 1] = -a[:, 1]; v[:, 0] = -v[:, 0]; j = j.clone(); j[:, 0] = -j[:, 0]
        return g, r, a, v, h, j, d

def unroll(model, raw, aux, vel, hist0, dims, jit, detach_hist=False):
    outs, valids = [], []
    hist = hist0
    for t in range(K):
        base = raw[:, t]
        hin = torch.cat([tm(hist[:, :3]) * hist[:, 3:4], hist[:, 3:4]], 1)
        if t > 0:
            wd, vd = tr.warp_crop(aux[:, t - 1, 0:1], vel[:, t], dims)
            dz = (aux[:, t, 0:1] - wd).abs() * vd
        else:
            dz = torch.zeros_like(aux[:, t, 0:1])
        ax = torch.cat([aux[:, t], dz], 1)
        out, _ = model(base.float(), jit[:, t], 1.0, torch.cat([ax, hin], 1).float())
        out = out.float(); outs.append(out); valids.append(hist[:, 3:4])
        if t + 1 < K:
            src = out.detach() if detach_hist else out
            w, valid = tr.warp_crop(src, vel[:, t + 1], dims)
            hist = torch.cat([w * valid, valid], 1)
    return outs, valids

def quiz(model, held, device, log=None):
    model.eval(); dpsnr, lapr, tcons, rmean, nan, hl1 = [], [], [], [], 0, []
    with torch.no_grad():
        for item in held:
            gt, raw, aux, vel, h0, jit, dims = [x.to(device).unsqueeze(0).float() for x in item]
            if float(luma(tm(gt[:, -1])).mean()) < tr.LUMA_FLOOR or float(lap_energy(gt[:, -1])) < 5e-3: continue
            # a crop whose raw fetch already IS the truth (every sample rejected, or flat) carries no AA
            # information and turns any change into a -60 dB "loss"; it is not a test of anything
            if psnr(raw[:, -1], gt[:, -1]) > 50.0: continue
            outs, valids = unroll(model, raw, aux, vel, h0, dims, jit, True)
            o, b, g = outs[-1], raw[:, -1], gt[:, -1]
            if not torch.isfinite(o).all(): nan += 1
            dpsnr.append(psnr(o, g) - psnr(b, g))
            lapr.append(float(lap_energy(o) / lap_energy(g).clamp_min(1e-8)))
            w, valid = tr.warp_crop(outs[-2], vel[:, -1], dims)
            tcons.append(float(((tm(o) - tm(w)).abs() * valid).sum() / valid.sum().clamp_min(1) / 3))
            rmean.append(float((o - b).abs().mean())); hl1.append(float((tm(o) - tm(g)).abs().mean()))
    rows = []
    def row(name, val, ok, thr): rows.append((name, val, ok, thr)); return ok
    ok = True
    ok &= row("psnr_gain_db  (out vs raw fetch, mean)", float(np.mean(dpsnr)), np.mean(dpsnr) >= 1.0, ">= 1.0")
    row("psnr_gain_med (median)", float(np.median(dpsnr)), True, "info")
    ok &= row("psnr_gain_min (worst case)", float(np.min(dpsnr)), np.min(dpsnr) >= 0.0, ">= 0.0")
    ok &= row("lap_ratio     (|lap out| / |lap truth|)", float(np.mean(lapr)), 0.85 <= np.mean(lapr) <= 1.15, "0.85..1.15")
    row("temporal      (|out_t - warp(out_t-1)| on valid px)", float(np.mean(tcons)), True, "info")
    ok &= row("residual_mean (|out - raw|)", float(np.mean(rmean)), np.mean(rmean) < 0.15, "< 0.15")
    ok &= row("finite        (NaN/Inf outputs)", float(nan), nan == 0, "== 0")
    model.train()
    lines = ["QUIZ %s (%d crops)" % ("PASS" if ok else "FAIL", len(dpsnr))] + ["  %s  %-52s %10.4f   want %s" % ("PASS" if r[2] else "FAIL", r[0], r[1], r[3]) for r in rows] + ["  held-out L1 %.5f" % float(np.mean(hl1))]
    print("\n".join(lines), flush=True)
    if log: log.write("\n".join(lines) + "\n"); log.flush()
    return bool(ok), float(np.mean(hl1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", default="E:/NNCap/packed_aa"); ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ch", default="8,16,24"); ap.add_argument("--quiz", action="store_true")
    ap.add_argument("--wlap", type=float, default=1.0); ap.add_argument("--wmse", type=float, default=4.0)
    ap.add_argument("--quiz_every", type=int, default=2)
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt", "aa_budget.pt"))
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(a.ckpt), exist_ok=True)
    log = open(os.path.splitext(a.ckpt)[0] + ".log", "a")
    def logp(s): print(s, flush=True); log.write(s + "\n"); log.flush()
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    files = sorted(glob.glob(os.path.join(a.packed, "*.npz")))
    train_f = [f for f in files if f.endswith("_train.npz")]; held_f = [f for f in files if f.endswith("_held.npz")]
    logp("aa crops: %d train, %d held-out (%s)" % (len(train_f), len(held_f), a.packed))
    if not train_f or not held_f: logp("no crops - run pack_aa.py first"); sys.exit(2)
    z0 = np.load(files[0]); tc._SIGN[0], tc._SIGN[1] = [float(s) for s in z0["sign"]]
    logp("warp sign from the pack: (%+.0f, %+.0f)" % (tc._SIGN[0], tc._SIGN[1]))
    held = [tuple(torch.from_numpy(np.load(f)[k]) for k in KEYS) for f in held_f]
    ch = tuple(int(c) for c in a.ch.split(","))
    model = AANet(ch=ch, in_ch=IN_CH).to(device).to(memory_format=torch.channels_last)
    if os.path.exists(a.ckpt):
        try: model.load_state_dict(torch.load(a.ckpt, map_location=device)); logp("loaded " + a.ckpt)
        except Exception as e: logp("could not load %s (%s) - fresh" % (a.ckpt, e))
    logp("train crops %d, held-out %d, channels %s, params %d" % (len(train_f), len(held_f), ch, sum(p.numel() for p in model.parameters())))
    if a.quiz: ok, _ = quiz(model, held, device, log); sys.exit(0 if ok else 1)
    ds = AACrops(train_f, n=a.rounds * a.steps * a.batch)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.rounds * a.steps)
    amp = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    it = iter(dl); t0 = time.time(); best_held, stale = 1e9, 0
    for rnd in range(a.rounds):
        run = 0.0
        for step in range(a.steps):
            gt, raw, aux, vel, h0, jit, dims = [x.to(device, non_blocking=True).float() for x in next(it)]
            gt = gt.contiguous(); raw = raw.contiguous(); aux = aux.contiguous()
            with torch.autocast("cuda", dtype=amp, enabled=device == "cuda"):
                outs, valids = unroll(model, raw, aux, vel, h0, dims, jit)
                l1 = sum((tm(outs[t]) - tm(gt[:, t])).abs().mean() for t in range(K)) / K
                mse = sum(((tm(outs[t]) - tm(gt[:, t])) ** 2).mean() for t in range(K)) / K
                llap = sum((tr.lap_map(outs[t]) - tr.lap_map(gt[:, t])).abs().mean() for t in range(K)) / K
                tcons = 0.0
                for t in range(1, K):
                    w, valid = tr.warp_crop(outs[t - 1].detach(), vel[:, t], dims)
                    tcons = tcons + ((tm(outs[t]) - tm(w)).abs() * valid).sum() / valid.sum().clamp_min(1) / 3
                tcons = tcons / (K - 1)
                wst = 0.1 + 0.4 * rnd / max(a.rounds - 1, 1)
                loss = l1 + a.wmse * mse + a.wlap * llap + wst * tcons
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            run += float(loss.detach())
            if step % 50 == 0:
                print("round %d step %d loss %.4f (l1 %.4f lap %.4f tcons %.4f) %.0fs" % (rnd, step, float(loss), float(l1), float(llap), float(tcons), time.time() - t0), flush=True)
        logp("round %d done, mean loss %.4f" % (rnd, run / a.steps))
        q = max(a.quiz_every, 1)
        if (rnd % q) == q - 1 or rnd == a.rounds - 1:
            ok, hl1 = quiz(model, held, device, log)
        else:
            model.eval()
            with torch.no_grad():
                hl = []
                for item in held[:60]:
                    gt, raw, aux, vel, h0, jit, dims = [x.to(device).unsqueeze(0).float() for x in item]
                    o = unroll(model, raw, aux, vel, h0, dims, jit, True)[0][-1]; hl.append(float((tm(o) - tm(gt[:, -1])).abs().mean()))
            model.train(); hl1 = float(np.mean(hl))
        if hl1 < best_held:
            best_held, stale = hl1, 0; torch.save(model.state_dict(), a.ckpt); logp("  saved (best held-out L1 %.5f)" % hl1)
        else:
            stale += 1
        logp("round %d: held-out L1 %.5f  best %.5f  stale %d" % (rnd, hl1, best_held, stale))
    logp("done: best held-out L1 %.5f -> %s" % (best_held, a.ckpt))

if __name__ == "__main__":
    main()
