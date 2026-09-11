"""train_captures.py - train + quiz the SR/AA residual net on REAL captured frames,
TEMPORAL: the previous frame of a burst, warped by the current velocity field,
is an input alongside the current low-res frame.

Data: sessions written by the layer's capture harness (docs/nn/capture-format.md).
Ground truth = the TAA-resolved plane (anti-aliased, render res, linear HDR).
Input = that truth pushed through the sim's forward model (plugin-exact Halton
jitter + non-integer downsample), the frame's own depth / normal / velocity
magnitude at the same jittered positions, and - when the frame is not the
first of its burst - the PREVIOUS frame's resolved output warped into this
frame by this frame's velocity, plus a validity mask (0 where the vector is
the unwritten sentinel or points off-screen). Frames without a predecessor get
a zero history and mask 0, so the net learns the single-frame fallback too.

The warp's sign convention is not assumed: at start the loader tries the four
sign combinations on a real burst pair and keeps the one that best matches the
current frame (logged), so a flipped axis can never silently ruin the run.

    python tools/nn/train_captures.py            # D:/NNCap and E:/NNCap
    python tools/nn/train_captures.py --quiz     # score the saved checkpoint

Copyright (C) 2026 MotionVectors contributors. SPDX: GPL-3.0-or-later
"""
import argparse, glob, json, os, random, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quick_train import (tm, luma, lap_energy, psnr, jitter_px, degrade, baseline_up, ResidualSR)

CROPS_PER_FRAME = 8      # a 150 MB frame is decoded once and cropped eight times
VEL_SENTINEL = 0.05      # |v| in UV units above this = unwritten (the 64-px clamp)

# ---------------------------------------------------------------- frames
def load_plane(frame_dir, meta, name):
    p = meta["planes"][name]
    dt = {"f16": np.float16, "f32": np.float32, "u8": np.uint8, "u16": np.uint16, "u32": np.uint32}[p["dtype"]]
    return np.fromfile(os.path.join(frame_dir, name + ".bin"), dtype=dt).reshape(p["h"], p["w"], p["c"])

def list_frames(roots):
    """[(dir, meta, prev_dir_or_None)] - prev is the previous frame of the same burst."""
    out = []
    for root in roots:
        for sess in sorted(glob.glob(os.path.join(root, "*"))):
            frames = []
            for f in sorted(glob.glob(os.path.join(sess, "frame_*"))):
                mp = os.path.join(f, "manifest.json")
                if not os.path.exists(mp): continue
                m = json.load(open(mp))
                if all(k in m["planes"] for k in ("resolved", "normal", "velocity")): frames.append((f, m))
            for i, (f, m) in enumerate(frames):
                prev = None
                b = m.get("burst")
                if b and b["index"] > 0 and i > 0:
                    pf, pm = frames[i - 1]; pb = pm.get("burst")
                    if pb and pb["id"] == b["id"] and pb["index"] == b["index"] - 1: prev = pf
                out.append((f, m, prev))
    return out

def resolved_linear(frame_dir, meta):
    res = load_plane(frame_dir, meta, "resolved").astype(np.float32)[..., :3]
    return np.clip(np.nan_to_num(res, nan=0.0, posinf=64.0, neginf=0.0), 0.0, 64.0)

def warp(prev, vx, vy, sx, sy):
    """prev (3,H,W) sampled at uv + (sx*vx, sy*vy); vel is in UV units (NDC*0.5)."""
    _, H, W = prev.shape
    uu = (torch.arange(W, dtype=torch.float32) + 0.5) / W; vv = (torch.arange(H, dtype=torch.float32) + 0.5) / H
    gu = uu.view(1, W).expand(H, W) + sx * vx; gv = vv.view(H, 1).expand(H, W) + sy * vy
    grid = torch.stack([gu * 2 - 1, gv * 2 - 1], dim=-1).unsqueeze(0)
    return F.grid_sample(prev.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0], \
           ((gu >= 0) & (gu <= 1) & (gv >= 0) & (gv <= 1)).float()

_SIGN = [1.0, -1.0]      # decided at start by calibrate_warp()

def calibrate_warp(frames, log=print):
    """Pick the (sx, sy) that makes warp(prev) match current best on a real pair."""
    pairs = [(f, m, p) for (f, m, p) in frames if p is not None]
    if not pairs: log("warp: no burst pairs found - history stays zero"); return
    f, m, p = pairs[len(pairs) // 2]
    cur = torch.from_numpy(resolved_linear(f, m).transpose(2, 0, 1)); prv = torch.from_numpy(resolved_linear(p, json.load(open(os.path.join(p, "manifest.json")))).transpose(2, 0, 1))
    vel = load_plane(f, m, "velocity").astype(np.float32); vx = torch.from_numpy(vel[..., 0]); vy = torch.from_numpy(vel[..., 1])
    bad = (vx.abs() > VEL_SENTINEL) | (vy.abs() > VEL_SENTINEL); vx = torch.where(bad, 0.0, vx); vy = torch.where(bad, 0.0, vy)
    best = None
    ref = float((tm(cur.unsqueeze(0)) - tm(prv.unsqueeze(0))).abs().mean())
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            w, mask = warp(prv, vx, vy, sx, sy)
            err = float(((tm(w.unsqueeze(0)) - tm(cur.unsqueeze(0))).abs().mean(1)[0] * mask).sum() / mask.sum().clamp_min(1))
            log("warp sign (%+.0f,%+.0f): L1 %.5f   (unwarped %.5f)" % (sx, sy, err, ref))
            if best is None or err < best[0]: best = (err, sx, sy)
    _SIGN[0], _SIGN[1] = best[1], best[2]
    log("warp: using (sx, sy) = (%+.0f, %+.0f)" % (best[1], best[2]))

def frame_tensors(frame_dir, meta, prev_dir):
    """gt (3,H,W) linear, aux (4,H,W), hist (4,H,W): warped previous resolved (linear) + validity."""
    gt = resolved_linear(frame_dir, meta)
    vel_ = load_plane(frame_dir, meta, "velocity").astype(np.float32)
    dep = np.clip(1.0 - np.log1p(np.clip(np.nan_to_num(vel_[..., 2], nan=65504.0, posinf=65504.0, neginf=0.0), 0.0, 65504.0)) / np.log1p(20000.0), 0.0, 1.0)   # log depth: 60 m -> 0.59, 2 km -> 0.23, 13 km -> 0.04, sky -> 0   # clip-w depth, see train_recurrent
    nrm = load_plane(frame_dir, meta, "normal").astype(np.float32)[..., :2]
    vel = load_plane(frame_dir, meta, "velocity").astype(np.float32)
    d = np.sqrt(np.clip(np.nan_to_num(dep, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))
    vx = np.nan_to_num(vel[..., 0]); vy = np.nan_to_num(vel[..., 1])
    bad = (np.abs(vx) > VEL_SENTINEL) | (np.abs(vy) > VEL_SENTINEL)
    vmag = np.clip(np.sqrt(vx ** 2 + vy ** 2) * 50.0, 0.0, 1.0); vmag[bad] = 1.0
    nrm = np.clip(np.nan_to_num(nrm), -1.0, 1.0)
    aux = np.stack([d, nrm[..., 0], nrm[..., 1], vmag], axis=0)
    H, W = gt.shape[:2]
    if prev_dir is not None:
        prv = torch.from_numpy(resolved_linear(prev_dir, json.load(open(os.path.join(prev_dir, "manifest.json")))).transpose(2, 0, 1))
        vxt = torch.from_numpy(np.where(bad, 0.0, vx).astype(np.float32)); vyt = torch.from_numpy(np.where(bad, 0.0, vy).astype(np.float32))
        w, mask = warp(prv, vxt, vyt, _SIGN[0], _SIGN[1])
        mask = mask * torch.from_numpy((~bad).astype(np.float32))
        hist = torch.cat([w * mask, mask.unsqueeze(0)], dim=0)
    else:
        hist = torch.zeros(4, H, W)
    return torch.from_numpy(np.ascontiguousarray(gt.transpose(2, 0, 1))), torch.from_numpy(np.ascontiguousarray(aux)), hist

class CaptureCrops(torch.utils.data.Dataset):
    def __init__(self, frames, size=512, n=4000, seed=0):
        self.frames, self.size, self.n, self.seed = frames, size, n, seed
        self.cache = {}
    def __len__(self): return self.n
    def _frame(self, i):
        if i not in self.cache:
            if len(self.cache) > 2: self.cache.pop(next(iter(self.cache)))
            self.cache[i] = frame_tensors(*self.frames[i])
        return self.cache[i]
    def __getitem__(self, k):
        rng = random.Random(self.seed * 100003 + k)
        fi = random.Random(self.seed * 7919 + k // CROPS_PER_FRAME).randrange(len(self.frames))   # 8 crops share a frame
        gt, aux, hist = self._frame(fi)
        _, H, W = gt.shape
        y, x = rng.randrange(H - self.size + 1), rng.randrange(W - self.size + 1)
        g = gt[:, y:y + self.size, x:x + self.size].clone(); a = aux[:, y:y + self.size, x:x + self.size].clone(); h = hist[:, y:y + self.size, x:x + self.size].clone()
        if rng.random() < 0.5: g = g.flip(-1); a = a.flip(-1); h = h.flip(-1); a[1] = -a[1]
        return g, a, h

# ---------------------------------------------------------------- pairs
def aux_down_up(aux, ratio, jit, S):
    B, C, _, _ = aux.shape
    s = int(round(S / ratio)); jx, jy = jit
    ii = torch.arange(s, device=aux.device, dtype=aux.dtype)
    gx = ((ii + 0.5 + jx) * ratio) / S * 2 - 1; gy = ((ii + 0.5 + jy) * ratio) / S * 2 - 1
    grid = torch.stack(torch.meshgrid(gy, gx, indexing="ij")[::-1], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    lo = F.grid_sample(aux, grid, mode="nearest", padding_mode="border", align_corners=False)
    oo = torch.arange(S, device=aux.device, dtype=aux.dtype)
    ux = (((oo + 0.5) / ratio - jx) / s) * 2 - 1; uy = (((oo + 0.5) / ratio - jy) / s) * 2 - 1
    grid2 = torch.stack(torch.meshgrid(uy, ux, indexing="ij")[::-1], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    return F.grid_sample(lo, grid2, mode="nearest", padding_mode="border", align_corners=False)

def make_pair(gt, aux, hist, ratio, idx, phases=8):
    jit = jitter_px(idx % phases)
    lo = degrade(gt, ratio, jit, phases)
    hin = torch.cat([tm(hist[:, :3]) * hist[:, 3:4], hist[:, 3:4]], dim=1)     # history tonemapped, masked
    return baseline_up(lo, gt.shape[-1], ratio, jit), torch.cat([aux_down_up(aux, ratio, jit, gt.shape[-1]), hin], dim=1), jit

def quiz(model, held, device, log=None):
    model.eval(); rows = []
    dpsnr, lapr, stab_m, stab_b, rmean, hl1, nan = [], [], [], [], [], [], 0
    with torch.no_grad():
        for gt, aux, hist in held:
            gt = gt.to(device).unsqueeze(0).float(); aux = aux.to(device).unsqueeze(0).float(); hist = hist.to(device).unsqueeze(0).float()
            if float(lap_energy(gt)) < 1e-3: continue
            for ratio in (1.3, 1.5, 2.0):
                ba, aa, ja = make_pair(gt, aux, hist, ratio, 3); bb, ab, jb = make_pair(gt, aux, hist, ratio, 5)
                oa, ra = model(ba, ja, ratio, aa); ob, _ = model(bb, jb, ratio, ab)
                if not torch.isfinite(oa).all(): nan += 1
                dpsnr.append(psnr(oa, gt) - psnr(ba, gt))
                lapr.append(float(lap_energy(oa) / lap_energy(gt).clamp_min(1e-8)))
                stab_m.append(float((tm(oa) - tm(ob)).abs().mean())); stab_b.append(float((tm(ba) - tm(bb)).abs().mean()))
                rmean.append(float(ra.abs().mean())); hl1.append(float((tm(oa) - tm(gt)).abs().mean()))
    def row(name, val, ok, thr): rows.append((name, val, ok, thr)); return ok
    ok = True
    ok &= row("psnr_gain_db  (out vs baseline, mean)", float(np.mean(dpsnr)), np.mean(dpsnr) >= 1.0, ">= 1.0")
    ok &= row("psnr_gain_min (worst case)", float(np.min(dpsnr)), np.min(dpsnr) >= 0.0, ">= 0.0")
    ok &= row("lap_ratio     (|lap out| / |lap truth|)", float(np.mean(lapr)), 0.85 <= np.mean(lapr) <= 1.15, "0.85..1.15")
    ok &= row("stability     (out delta across phases / baseline delta)", float(np.mean(stab_m) / max(np.mean(stab_b), 1e-8)), np.mean(stab_m) <= 0.6 * np.mean(stab_b), "<= 0.60")
    ok &= row("residual_mean (|r|, additive sanity)", float(np.mean(rmean)), np.mean(rmean) < 0.15, "< 0.15")
    ok &= row("finite        (NaN/Inf outputs)", float(nan), nan == 0, "== 0")
    lines = ["QUIZ %s" % ("PASS" if ok else "FAIL")] + ["  %s  %-52s %10.4f   want %s" % ("PASS" if r[2] else "FAIL", r[0], r[1], r[3]) for r in rows]
    lines.append("  held-out L1 %.5f" % float(np.mean(hl1)))
    print("\n".join(lines), flush=True)
    if log: log.write("\n".join(lines) + "\n"); log.flush()
    model.train()
    return bool(ok), rows, float(np.mean(hl1))

# ---------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/NNCap;E:/NNCap", help="capture roots, ';'-separated")
    ap.add_argument("--rounds", type=int, default=12); ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=6); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--quiz", action="store_true")
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt", "sr_temporal.pt"))
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(a.ckpt), exist_ok=True)
    log = open(os.path.join(os.path.dirname(a.ckpt), "quiz_temporal.log"), "a")
    def logp(s): print(s, flush=True); log.write(s + "\n"); log.flush()
    torch.manual_seed(0); np.random.seed(0); random.seed(0)

    frames = list_frames([r.strip() for r in a.root.split(";")])
    npair = sum(1 for f in frames if f[2] is not None)
    logp("captured frames: %d, with a burst predecessor: %d (roots: %s)" % (len(frames), npair, a.root))
    if len(frames) < 8: sys.exit("too few frames")
    calibrate_warp(frames, logp)
    held_frames = frames[9::10]; train_frames = [f for i, f in enumerate(frames) if i % 10 != 9]
    hd = CaptureCrops(held_frames, n=32, seed=777); held = [hd[i] for i in range(32)]
    train = CaptureCrops(train_frames, n=a.rounds * a.steps * a.batch, seed=1)
    dl = torch.utils.data.DataLoader(train, batch_size=a.batch, shuffle=False, num_workers=a.workers,
                                     persistent_workers=a.workers > 0, pin_memory=device == "cuda", prefetch_factor=4 if a.workers > 0 else None)
    logp("train frames %d, held-out frames %d, workers %d" % (len(train_frames), len(held_frames), a.workers))

    model = ResidualSR(in_ch=14).to(device)          # 6 + depth/nx/ny/|v| + history rgb/mask
    logp("params: %d" % sum(p.numel() for p in model.parameters()))
    if os.path.exists(a.ckpt): model.load_state_dict(torch.load(a.ckpt, map_location=device)); logp("loaded " + a.ckpt)
    if a.quiz: ok, _, _ = quiz(model, held, device, log); sys.exit(0 if ok else 1)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.rounds * a.steps)
    amp = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    it = iter(dl); passes = 0; t0 = time.time(); best_held, stale = 1e9, 0
    for rnd in range(a.rounds):
        run = 0.0; run_l1 = 0.0
        for step in range(a.steps):
            gt, aux, hist = next(it); gt = gt.to(device, non_blocking=True); aux = aux.to(device, non_blocking=True); hist = hist.to(device, non_blocking=True)
            ratio = random.uniform(1.3, 2.0); ia, ib = random.randrange(8), random.randrange(8)
            with torch.autocast("cuda", dtype=amp, enabled=device == "cuda"):
                ba, aa, ja = make_pair(gt, aux, hist, ratio, ia); bb, ab, jb = make_pair(gt, aux, hist, ratio, ib)
                oa, ra = model(ba.float(), ja, ratio, aa.float()); ob, _ = model(bb.float(), jb, ratio, ab.float())
                oa, ob = oa.float(), ob.float()
                l1 = (tm(oa) - tm(gt)).abs().mean() + (tm(ob) - tm(gt)).abs().mean()
                llap = (lap_energy(oa) - lap_energy(gt)).abs()
                lstab = (tm(oa) - tm(ob)).abs().mean()
                wst = 0.25 + 0.75 * rnd / max(a.rounds - 1, 1)
                loss = l1 + 0.5 * llap + wst * lstab
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            run += float(loss.detach()); run_l1 += float(l1.detach()) * 0.5
            if step % 100 == 0:
                print("round %d step %d loss %.4f (l1 %.4f lap %.4f stab %.4f) %.0fs" % (rnd, step, float(loss), float(l1), float(llap), float(lstab), time.time() - t0), flush=True)
        logp("round %d done, mean loss %.4f" % (rnd, run / a.steps))
        ok, _, held_l1 = quiz(model, held, device, log)
        train_l1 = run_l1 / a.steps
        if held_l1 < best_held - 1e-4:
            best_held, stale = held_l1, 0; torch.save(model.state_dict(), a.ckpt); logp("  saved (best held-out L1 %.5f)" % held_l1)
        else: stale += 1
        logp("round %d: train L1 %.5f  held-out L1 %.5f  gap %+.5f  stale %d" % (rnd, train_l1, held_l1, held_l1 - train_l1, stale))
        if stale >= 2 and held_l1 > 2.0 * train_l1 and held_l1 < best_held * 1.05:
            logp("MEMORISING: held-out flat for %d rounds at %.1fx train - stopping; more frames (new places, aircraft), not more steps." % (stale, held_l1 / max(train_l1, 1e-9))); break
        passes = passes + 1 if ok else 0
        if passes >= 2: logp("PERFECT: quiz passed twice in a row after round %d" % rnd); break
    log.close()

if __name__ == "__main__":
    main()
