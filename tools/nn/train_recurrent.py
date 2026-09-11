"""train_recurrent.py - RECURRENT training of the SR/AA residual net on burst captures.

What changes against train_captures.py:
  * Unrolled recurrence. Each item is K consecutive burst frames at one crop
    window. Step 0 sees the warped previous TAA output (what train_captures
    trained on); every later step sees the net's OWN previous output warped by
    that frame's velocity - exactly what it will see at inference.
  * Temporal-consistency loss: |out_t - warp(out_{t-1})| on pixels the warp
    reaches, lighting held constant inside a burst by the capture sweep. This
    is the metrics suite's static-delta, measured on the net's own stream.
  * Brightness floor: crops whose tonemapped mean luminance is under 0.03 are
    resampled (train) or skipped (quiz) - a near-black night crop let a tiny
    residual swing PSNR by -25 dB and taught nothing.
  * Per-ratio quiz: PSNR gain at 1.3x / 1.5x / 2.0x separately.
  * Width flag: --ch 32,64,96 (default, ~400k params) or 16,32,48 (the old net).

    python tools/nn/train_recurrent.py                 # D:/NNCap and E:/NNCap
    python tools/nn/train_recurrent.py --quiz
    python tools/nn/watch.py tools/nn/ckpt/train_recurrent.log

Copyright (C) 2026 MotionVectors contributors. SPDX: GPL-3.0-or-later
"""
import argparse, glob, json, os, random, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quick_train import (tm, luma, lap_energy, psnr, jitter_px, degrade, baseline_up, ResidualSR)
from train_captures import (load_plane, resolved_linear, warp, calibrate_warp, aux_down_up, VEL_SENTINEL, _SIGN)
import train_captures as tc

K = 3                    # unrolled steps
CROPS_PER_SEQ = 6        # one decoded sequence -> this many crops
LUMA_FLOOR = 0.03

# ---------------------------------------------------------------- sequences
def capture_ok(m):
    """Reject frames whose resolved plane is not real history: the resolve off, or an
    experimental accumulation flag on (accum_exp shipped with a=1 everywhere on 2026-09-10)."""
    live = m.get("live", {}) or {}
    if str(live.get("taa.enable", "1")) == "0": return False
    if str(live.get("taa.accum_exp", "0")) == "1": return False
    if str(live.get("taa.conf_exp", "0")) == "1": return False
    if "gbuf_0" not in m.get("planes", {}): return False        # no pass-end snapshot: the normal plane is the scrambled resolve-time copy
    return True

def list_sequences(roots):
    """[(frames[K] (dir, meta), prev_of_first or None, group)] from bursts.
    group = session + burst id: the held-out split is made on it, never on sequence
    index, so no held-out window shares a burst with a training window."""
    seqs = []
    for root in roots:
        for sess in sorted(glob.glob(os.path.join(root, "*"))):
            fr = []
            for f in sorted(glob.glob(os.path.join(sess, "frame_*"))):
                mp = os.path.join(f, "manifest.json")
                if not os.path.exists(mp): continue
                m = json.load(open(mp))
                if not all(k in m["planes"] for k in ("resolved", "normal", "velocity")): continue
                if not capture_ok(m): continue
                fr.append((f, m))
            i = 0
            while i < len(fr):
                b = fr[i][1].get("burst")
                if not b: seqs.append(([fr[i]] * K, None, "%s/single/%d" % (os.path.basename(sess), i))); i += 1; continue
                j = i
                # consecutive in the burst AND consecutive in the frame counter (a paused burst resumes minutes later)
                while (j + 1 < len(fr) and fr[j + 1][1].get("burst", {}).get("id") == b["id"]
                       and fr[j + 1][1]["burst"]["index"] == fr[j][1]["burst"]["index"] + 1
                       and fr[j + 1][1].get("frame", 0) == fr[j][1].get("frame", -2) + 1): j += 1
                run = fr[i:j + 1]; group = "%s/burst/%s" % (os.path.basename(sess), b["id"])
                for s in range(0, len(run) - K + 1, K):        # non-overlapping windows of K
                    prev = run[s - 1][0] if s > 0 else None
                    seqs.append((run[s:s + K], prev, group))
                i = j + 1
    return seqs

def group_hash(group):
    h = 0
    for ch in group: h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h

def held_groups(seqs):
    """Deterministic ~10 % held-out by burst group; a tiny set still holds out its highest-hash group."""
    groups = sorted(set(s[2] for s in seqs))
    held = set(g for g in groups if (group_hash(g) % 10) == 9)
    if not held and groups: held = {max(groups, key=group_hash)}
    return held

def seq_tensors(frames, prev_dir):
    """gt (K,3,H,W), aux (K,6,H,W: depth, nrm.x, nrm.y, |vel|, valid, coverage), vel (K,3,H,W: vx, vy, valid), hist0 (4,H,W)."""
    gts, auxs, vels = [], [], []
    for t, (f, m) in enumerate(frames):
        gt = resolved_linear(f, m)
        nrm = np.clip(np.nan_to_num(load_plane(f, m, "normal").astype(np.float32)[..., :2]), -1, 1)
        vel = load_plane(f, m, "velocity").astype(np.float32)
        # Depth from the injected clip-w in velocity.z (metres; 65504 = nothing drawn). depth.bin is
        # the resolve-time R32 copy, which reads X-Plane's Z-compressed depth raw (every other
        # column zero, measured 2026-09-09); the velocity target is ours and clean in every build.
        dep = np.clip(1.0 - np.log1p(np.clip(np.nan_to_num(vel[..., 2], nan=65504.0, posinf=65504.0, neginf=0.0), 0.0, 65504.0)) / np.log1p(20000.0), 0.0, 1.0)   # log depth: 60 m -> 0.59, 2 km -> 0.23, 13 km -> 0.04, sky -> 0
        vx = np.nan_to_num(vel[..., 0]); vy = np.nan_to_num(vel[..., 1])
        bad = (np.abs(vx) > VEL_SENTINEL) | (np.abs(vy) > VEL_SENTINEL)
        vx = np.where(bad, 0.0, vx).astype(np.float32); vy = np.where(bad, 0.0, vy).astype(np.float32)
        d = np.clip(dep, 0.0, 1.0).astype(np.float32)          # near = 1, far = 0, sky = 0
        vmag = np.clip(np.sqrt(vx ** 2 + vy ** 2) * 50.0, 0.0, 1.0); vmag[bad] = 1.0
        same = frames[0][0] == f and t > 0            # padded single frame: no motion, history invalid
        valid = (~bad).astype(np.float32); cov = np.clip(np.nan_to_num(vel[..., 3]), 0.0, 1.0).astype(np.float32)
        gts.append(gt.transpose(2, 0, 1)); auxs.append(np.stack([d, nrm[..., 0], nrm[..., 1], vmag, valid, cov], 0))
        vels.append(np.stack([vx, vy, np.zeros_like(vx) if same else (~bad).astype(np.float32)], 0))
    gt = torch.from_numpy(np.ascontiguousarray(np.stack(gts))); aux = torch.from_numpy(np.ascontiguousarray(np.stack(auxs)))
    vel = torch.from_numpy(np.ascontiguousarray(np.stack(vels)))
    H, W = gt.shape[-2:]
    if prev_dir is not None:
        prv = torch.from_numpy(resolved_linear(prev_dir, json.load(open(os.path.join(prev_dir, "manifest.json")))).transpose(2, 0, 1))
        w, mask = warp(prv, vel[0, 0], vel[0, 1], _SIGN[0], _SIGN[1]); mask = mask * vel[0, 2]
        hist0 = torch.cat([w * mask, mask.unsqueeze(0)], 0)
    else:
        hist0 = torch.zeros(4, H, W)
    return gt, aux, vel, hist0

class SeqCrops(torch.utils.data.Dataset):
    def __init__(self, seqs, size=384, n=4000, seed=0):
        self.seqs, self.size, self.n, self.seed = seqs, size, n, seed; self.cache = {}
    def __len__(self): return self.n
    def _seq(self, i):
        if i not in self.cache:
            if len(self.cache) > 1: self.cache.pop(next(iter(self.cache)))
            self.cache[i] = seq_tensors(*self.seqs[i])
        return self.cache[i]
    def __getitem__(self, k):
        rng = random.Random(self.seed * 100003 + k)
        si = random.Random(self.seed * 7919 + k // CROPS_PER_SEQ).randrange(len(self.seqs))
        gt, aux, vel, hist0 = self._seq(si)
        _, _, H, W = gt.shape; S = self.size
        for _ in range(8):                                             # brightness floor
            y, x = rng.randrange(H - S + 1), rng.randrange(W - S + 1)
            if float(luma(tm(gt[-1:, :, y:y + S, x:x + S])).mean()) >= LUMA_FLOOR: break
        g = gt[:, :, y:y + S, x:x + S].clone(); a = aux[:, :, y:y + S, x:x + S].clone()
        v = vel[:, :, y:y + S, x:x + S].clone(); h = hist0[:, y:y + S, x:x + S].clone()
        if rng.random() < 0.5:
            g = g.flip(-1); a = a.flip(-1); v = v.flip(-1); h = h.flip(-1); a[:, 1] = -a[:, 1]; v[:, 0] = -v[:, 0]
        return g, a, v, h, torch.tensor([W, H], dtype=torch.float32)

class PackedCrops(torch.utils.data.Dataset):
    """Crops written by pack_captures.py: one ~10 MB .npz per item, flips at load time."""
    def __init__(self, files, n=4000, seed=0):
        self.files, self.n, self.seed = files, n, seed
    def __len__(self): return self.n
    def __getitem__(self, k):
        rng = random.Random(self.seed * 100003 + k)
        z = np.load(self.files[rng.randrange(len(self.files))])
        g = torch.from_numpy(z["gt"]); a = torch.from_numpy(z["aux"])            # fp16 stays fp16 until the GPU
        v = torch.from_numpy(z["vel"]); h = torch.from_numpy(z["hist0"])
        if rng.random() < 0.5:
            g = g.flip(-1); a = a.flip(-1); v = v.flip(-1); h = h.flip(-1); a[:, 1] = -a[:, 1]; v[:, 0] = -v[:, 0]
        return g, a, v, h, torch.from_numpy(z["dims"])

# Per-pixel Laplacian of tonemapped luma. The scalar-energy match this replaced
# rewarded high-frequency energy ANYWHERE (round 2 packed: lap ratio 1.55, PSNR
# -1.2 dB); a map match rewards edges only where the truth has them.
_LAP = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
def lap_map(c):
    y = luma(tm(c))
    return F.conv2d(y, _LAP.to(y.device, y.dtype), padding=1)

# ---------------------------------------------------------------- warp of the net's own output, inside the crop
def warp_crop(prev_out, vel, dims):
    """prev_out (B,3,S,S) sampled at crop uv + full-frame displacement; returns (warped, valid)."""
    B, _, S, _ = prev_out.shape
    vx = vel[:, 0] * _SIGN[0] * dims[:, 0].view(B, 1, 1) / S; vy = vel[:, 1] * _SIGN[1] * dims[:, 1].view(B, 1, 1) / S
    u = (torch.arange(S, device=prev_out.device, dtype=prev_out.dtype) + 0.5) / S
    gu = u.view(1, 1, S).expand(B, S, S) + vx; gv = u.view(1, S, 1).expand(B, S, S) + vy
    grid = torch.stack([gu * 2 - 1, gv * 2 - 1], dim=-1)
    w = F.grid_sample(prev_out, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    valid = ((gu >= 0) & (gu <= 1) & (gv >= 0) & (gv <= 1)).to(prev_out.dtype) * vel[:, 2]
    return w, valid.unsqueeze(1)

def unroll(model, gt, aux, vel, hist0, dims, ratio, phases, detach_hist=False):
    """Run K steps; returns lists of (out, base, hist_valid) per step."""
    outs, bases, valids = [], [], []
    hist = hist0
    for t in range(K):
        jit = jitter_px(phases[t] % 8)
        lo = degrade(gt[:, t], ratio, jit, 8)
        base = baseline_up(lo, gt.shape[-1], ratio, jit)
        hin = torch.cat([tm(hist[:, :3]) * hist[:, 3:4], hist[:, 3:4]], 1)
        # depth consistency: |depth_t - warp(depth_t-1)| on pixels the warp reaches; 0 at step 0.
        if t > 0:
            wd, vd = warp_crop(aux[:, t - 1, 0:1], vel[:, t], dims)
            dz = (aux[:, t, 0:1] - wd).abs() * vd
        else:
            dz = torch.zeros_like(aux[:, t, 0:1])
        ax = torch.cat([aux[:, t], dz], 1)
        out, _ = model(base.float(), jit, ratio, torch.cat([aux_down_up(ax, ratio, jit, gt.shape[-1]), hin], 1).float())
        out = out.float(); outs.append(out); bases.append(base); valids.append(hist[:, 3:4])
        if t + 1 < K:
            src = out.detach() if detach_hist else out
            w, valid = warp_crop(src, vel[:, t + 1], dims)
            hist = torch.cat([w * valid, valid], 1)
    return outs, bases, valids

def quiz(model, held, device, log=None):
    model.eval(); rows = []; per = {1.3: [], 1.5: [], 2.0: []}
    dpsnr, lapr, stab_m, stab_b, rmean, tcons, nan = [], [], [], [], [], [], 0
    with torch.no_grad():
        for gt, aux, vel, h0, dims in held:
            gt, aux, vel, h0, dims = [x.to(device).unsqueeze(0).float() for x in (gt, aux, vel, h0, dims)]
            if float(luma(tm(gt[:, -1])).mean()) < LUMA_FLOOR or float(lap_energy(gt[:, -1])) < 1e-3: continue
            for ratio in (1.3, 1.5, 2.0):
                oa, ba, _ = unroll(model, gt, aux, vel, h0, dims, ratio, [3, 4, 5], True)
                ob, bb, _ = unroll(model, gt, aux, vel, h0, dims, ratio, [5, 6, 7], True)
                o, b, g = oa[-1], ba[-1], gt[:, -1]
                if not torch.isfinite(o).all(): nan += 1
                gain = psnr(o, g) - psnr(b, g); dpsnr.append(gain); per[ratio].append(gain)
                lapr.append(float(lap_energy(o) / lap_energy(g).clamp_min(1e-8)))
                stab_m.append(float((tm(o) - tm(ob[-1])).abs().mean())); stab_b.append(float((tm(b) - tm(bb[-1])).abs().mean()))
                rmean.append(float((o - b).abs().mean()))
                w, valid = warp_crop(oa[-2], vel[:, -1], dims)
                tcons.append(float(((tm(o) - tm(w)).abs() * valid).sum() / valid.sum().clamp_min(1) / 3))
    def row(name, val, ok, thr): rows.append((name, val, ok, thr)); return ok
    ok = True
    ok &= row("psnr_gain_db  (out vs baseline, mean)", float(np.mean(dpsnr)), np.mean(dpsnr) >= 1.0, ">= 1.0")
    ok &= row("psnr_gain_min (worst case)", float(np.min(dpsnr)), np.min(dpsnr) >= 0.0, ">= 0.0")
    for r in (1.3, 1.5, 2.0): row("  psnr_gain @%.1fx" % r, float(np.mean(per[r])) if per[r] else 0.0, True, "info")
    ok &= row("lap_ratio     (|lap out| / |lap truth|)", float(np.mean(lapr)), 0.85 <= np.mean(lapr) <= 1.15, "0.85..1.15")
    ok &= row("stability     (out delta across phases / baseline delta)", float(np.mean(stab_m) / max(np.mean(stab_b), 1e-8)), np.mean(stab_m) <= 0.6 * np.mean(stab_b), "<= 0.60")
    row("temporal      (|out_t - warp(out_t-1)| on valid px)", float(np.mean(tcons)), True, "info")
    ok &= row("residual_mean (|out - baseline|)", float(np.mean(rmean)), np.mean(rmean) < 0.15, "< 0.15")
    ok &= row("finite        (NaN/Inf outputs)", float(nan), nan == 0, "== 0")
    lines = ["QUIZ %s" % ("PASS" if ok else "FAIL")] + ["  %s  %-52s %10.4f   want %s" % ("PASS" if r[2] else "FAIL", r[0], r[1], r[3]) for r in rows]
    print("\n".join(lines), flush=True)
    if log: log.write("\n".join(lines) + "\n"); log.flush()
    model.train()
    return bool(ok), rows

# ---------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/NNCap;E:/NNCap"); ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ch", default="32,64,96"); ap.add_argument("--quiz", action="store_true")
    ap.add_argument("--packed", default="", help="dir of pack_captures.py crops; empty = decode frames on the fly")
    ap.add_argument("--wlap", type=float, default=1.0, help="per-pixel Laplacian weight; raise when lap_ratio < 0.85")
    ap.add_argument("--wmse", type=float, default=4.0)
    ap.add_argument("--compile", action="store_true", help="torch.compile the net (falls back to eager if the backend is unavailable)")
    ap.add_argument("--quiz_every", type=int, default=2, help="full quiz every N rounds; the held-out L1 gate runs every round")
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt", "sr_recurrent.pt"))
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(a.ckpt), exist_ok=True)
    log = open(os.path.join(os.path.dirname(a.ckpt), "quiz_recurrent.log"), "a")
    def logp(s): print(s, flush=True); log.write(s + "\n"); log.flush()
    torch.manual_seed(0); np.random.seed(0); random.seed(0)

    roots = [r.strip() for r in a.root.split(";")]
    if a.packed:
        files = sorted(glob.glob(os.path.join(a.packed, "seq*_train.npz"))); hfiles = sorted(glob.glob(os.path.join(a.packed, "seq*_held.npz")))
        if len(files) < 8: sys.exit("no packed crops under %s - run pack_captures.py" % a.packed)
        if len(hfiles) < 1: sys.exit("no held-out crops under %s - re-run pack_captures.py (it holds out at least one burst)" % a.packed)
        calibrate_warp(tc.list_frames(roots), logp)      # the sign the packed history was warped with
        logp("packed crops: %d train, %d held-out (%s)" % (len(files), len(hfiles), a.packed))
        hd = PackedCrops(hfiles, n=32, seed=777); held = [hd[i] for i in range(32)]
        train = PackedCrops(files, n=a.rounds * a.steps * a.batch, seed=1)
        train_s = files; held_s = hfiles
    else:
        calibrate_warp(tc.list_frames(roots), logp)
        seqs = list_sequences(roots)
        nb = sum(1 for s in seqs if s[0][0][1].get("burst"))
        logp("sequences of %d: %d (%d from bursts, %d padded singles)" % (K, len(seqs), nb, len(seqs) - nb))
        if len(seqs) < 8: sys.exit("too few sequences")
        hg = held_groups(seqs); held_s = [s for s in seqs if s[2] in hg]; train_s = [s for s in seqs if s[2] not in hg]
        hd = SeqCrops(held_s, n=32, seed=777); held = [hd[i] for i in range(32)]
        train = SeqCrops(train_s, n=a.rounds * a.steps * a.batch, seed=1)
    dl = torch.utils.data.DataLoader(train, batch_size=a.batch, shuffle=False, num_workers=a.workers,
                                     persistent_workers=a.workers > 0, pin_memory=device == "cuda", prefetch_factor=4 if a.workers > 0 else None)
    ch = tuple(int(c) for c in a.ch.split(","))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    IN_CH = 3 + 3 + 7 + 4      # tm(base) rgb, jitter x/y + ratio planes, aux(6)+dz, warped history rgb + valid
    model = ResidualSR(ch=ch, in_ch=IN_CH).to(device).to(memory_format=torch.channels_last)
    logp("train sequences %d, held-out %d, channels %s, params %d" % (len(train_s), len(held_s), ch, sum(p.numel() for p in model.parameters())))
    if os.path.exists(a.ckpt):
        try: model.load_state_dict(torch.load(a.ckpt, map_location=device)); logp("loaded " + a.ckpt)
        except Exception as e: logp("checkpoint %s does not fit this input layout (%s) - starting fresh" % (a.ckpt, str(e)[:80]))
    raw_model = model                                   # state_dict / save always go through the uncompiled module
    if a.compile and device == "cuda":
        try:
            cm = torch.compile(model)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                cm(torch.zeros(1, 3, 64, 64, device=device), (0.0, 0.0), 1.5, torch.zeros(1, IN_CH - 6, 64, 64, device=device))
            model = cm; logp("torch.compile: on")
        except Exception as e:
            logp("torch.compile unavailable on this setup (%s) - eager" % str(e).splitlines()[-1][:120])
    if a.quiz: ok, _ = quiz(model, held, device, log); sys.exit(0 if ok else 1)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.rounds * a.steps)
    amp = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    it = iter(dl); passes = 0; t0 = time.time(); best_held, stale = 1e9, 0
    for rnd in range(a.rounds):
        run = 0.0; run_l1 = 0.0
        for step in range(a.steps):
            gt, aux, vel, h0, dims = [x.to(device, non_blocking=True).float() for x in next(it)]
            gt = gt.contiguous(); aux = aux.contiguous()
            ratio = random.uniform(1.3, 2.0); phases = [random.randrange(8) for _ in range(K)]
            with torch.autocast("cuda", dtype=amp, enabled=device == "cuda"):
                outs, bases, valids = unroll(model, gt, aux, vel, h0, dims, ratio, phases)
                l1 = sum((tm(outs[t]) - tm(gt[:, t])).abs().mean() for t in range(K)) / K
                mse = sum(((tm(outs[t]) - tm(gt[:, t])) ** 2).mean() for t in range(K)) / K
                llap = sum((lap_map(outs[t]) - lap_map(gt[:, t])).abs().mean() for t in range(K)) / K
                tcons = 0.0
                for t in range(1, K):
                    w, valid = warp_crop(outs[t - 1].detach(), vel[:, t], dims)
                    tcons = tcons + ((tm(outs[t]) - tm(w)).abs() * valid).sum() / valid.sum().clamp_min(1) / 3
                tcons = tcons / (K - 1)
                wst = 0.1 + 0.4 * rnd / max(a.rounds - 1, 1)          # 0.1 -> 0.5, never dominant
                loss = l1 + a.wmse * mse + a.wlap * llap + wst * tcons
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            run += float(loss.detach()); run_l1 += float(l1.detach())
            if step % 50 == 0:
                print("round %d step %d loss %.4f (l1 %.4f lap %.4f tcons %.4f) %.0fs" % (rnd, step, float(loss), float(l1), float(llap), float(tcons), time.time() - t0), flush=True)
        logp("round %d done, mean loss %.4f" % (rnd, run / a.steps))
        if (rnd % max(a.quiz_every, 1)) == max(a.quiz_every, 1) - 1 or rnd == a.rounds - 1:
            ok, rows = quiz(model, held, device, log)
        else:
            ok = False
        train_l1 = run_l1 / a.steps
        hl = float(np.mean([float((tm(unroll(model, *[x.to(device).unsqueeze(0).float() for x in h], 1.5, [3, 4, 5], True)[0][-1]) - tm(h[0].to(device).unsqueeze(0).float()[:, -1])).abs().mean()) for h in held[:8]])) if True else 0.0
        if hl < best_held - 1e-4:
            best_held, stale = hl, 0; torch.save(raw_model.state_dict(), a.ckpt); logp("  saved (best held-out L1 %.5f)" % hl)
        else: stale += 1
        logp("round %d: train L1 %.5f  held-out L1 %.5f  gap %+.5f  stale %d" % (rnd, train_l1, hl, hl - train_l1, stale))
        if stale >= 2 and hl > 2.0 * train_l1 and hl < best_held * 1.05:
            logp("MEMORISING: held-out flat for %d rounds at %.1fx train - stopping; more frames, not more steps." % (stale, hl / max(train_l1, 1e-9))); break
        passes = passes + 1 if ok else 0
        if passes >= 2: logp("PERFECT: quiz passed twice in a row after round %d" % rnd); break
    log.close()

if __name__ == "__main__":
    main()
