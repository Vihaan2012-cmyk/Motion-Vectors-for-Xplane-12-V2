"""quick_train.py - train + quiz the SR/AA residual net on X-Plane's OWN content.

Data: X-Plane's stock textures (Resources/default scenery, Resources/bitmaps,
Laminar aircraft) - the render base, no addons. Each crop is a 512x512 ground
truth at OUTPUT resolution; the INPUT is that truth pushed through the same
forward model the sim applies: a jittered raster at a lower resolution
(Halton(2,3) jitter exactly as plugin.cpp publishes it: halton(index+1) - 0.5,
in low-res pixels) at a random non-integer ratio in [1.3, 2.0]. The net sees the
bilinear un-jittered upsample (the baseline, what taau/EASU would hand it) plus
the jitter and ratio as constant planes, and predicts an ADDITIVE residual. The
last conv is zero-initialised, so a fresh model IS the baseline (spec rule 1).

Quiz: a fixed held-out set is scored the way metrics.comp scores the sim -
PSNR gain over the baseline, Laplacian energy ratio to truth (not blurrier,
not sharper than truth), stability across two jitter phases of the SAME truth
(the S_STATIC_DELTA analogue), residual sanity. Every criterion is a PASS/FAIL
line with its threshold; the loop trains, quizzes, repeats, and stops when
every line passes twice in a row.

    python tools/nn/quick_train.py                # train+quiz loop on the 4060
    python tools/nn/quick_train.py --quiz         # quiz the saved checkpoint only
    python tools/nn/quick_train.py --rounds 8 --steps 600

Copyright (C) 2026 MotionVectors contributors. SPDX: GPL-3.0-or-later
"""
import argparse, glob, math, os, random, sys, time, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

XP_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
STOCK_AIRCRAFT = ["Cessna 172 SP", "Boeing 737-800", "Airbus A330-300", "Cirrus SR22",
                  "Cirrus Vision SF50", "Beechcraft Baron 58", "Beechcraft King Air C90B",
                  "Cessna Citation X", "Aero-Works Aerolite 103", "BETA Technologies Alia-250",
                  "Grumman F-14 Tomcat", "McDonnell Douglas MD-82", "Lancair Evolution",
                  "Van's RV-10", "Piper PA-18", "Robinson R22", "Sikorsky S-76C", "Stinson L-5",
                  "Schleicher ASK 21", "F-4 Phantom"]
SKIP_NAME = ("_NML", "_NRM", "_MAT", "_nml", "_nrm", "_mat", "normal", "NORMAL")

# ---------------------------------------------------------------- data
def discover(max_images=600, min_side=512, seed=0):
    roots = [os.path.join(XP_ROOT, "Resources", "default scenery"),
             os.path.join(XP_ROOT, "Resources", "bitmaps")]
    roots += [os.path.join(XP_ROOT, "Aircraft", "Laminar Research", a) for a in STOCK_AIRCRAFT]
    files = []
    for r in roots:
        if not os.path.isdir(r): continue
        for ext in ("png", "dds", "jpg"):
            files += glob.glob(os.path.join(r, "**", "*." + ext), recursive=True)
    files = [f for f in files if not any(s in os.path.basename(f) for s in SKIP_NAME)]
    random.Random(seed).shuffle(files)
    keep = []
    for f in files:
        try:
            with Image.open(f) as im:
                if min(im.size) >= min_side: keep.append(f)
        except Exception:
            pass
        if len(keep) >= max_images: break
    return keep

def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)

def load_linear(path):
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return srgb_to_linear(a)

class Crops(torch.utils.data.Dataset):
    """Random 512 crops of stock textures, in LINEAR light, as (3,H,W) tensors."""
    def __init__(self, files, size=512, n=4000, seed=0, exposure=(0.6, 2.5)):
        self.files, self.size, self.n, self.seed, self.exposure = files, size, n, seed, exposure
        self.cache = {}
    def __len__(self): return self.n
    def __getitem__(self, i):
        rng = random.Random(self.seed * 100003 + i)
        f = self.files[rng.randrange(len(self.files))]
        if f not in self.cache:
            if len(self.cache) > 64: self.cache.pop(next(iter(self.cache)))
            self.cache[f] = load_linear(f)
        img = self.cache[f]
        # ---- MEMORISATION IS A DATA PROBLEM. Run 3 learned 600 textures by
        # heart (train L1 0.0065 vs held-out 0.0177). Every crop is now a new
        # image: random source scale (crop a bigger window, shrink it), 90-degree
        # rotations, flips, per-channel gain, and a 35% chance of blending with
        # a second crop from another file - novel content every step.
        c = self._crop(img, rng)
        if rng.random() < 0.35:
            g = self.files[rng.randrange(len(self.files))]
            if g not in self.cache:
                if len(self.cache) > 64: self.cache.pop(next(iter(self.cache)))
                self.cache[g] = load_linear(g)
            a = rng.uniform(0.25, 0.75)
            c = a * c + (1 - a) * self._crop(self.cache[g], rng)
        gain = np.array([rng.uniform(0.7, 1.3) for _ in range(3)], dtype=np.float32)
        c = np.ascontiguousarray(c * gain * rng.uniform(*self.exposure))   # HDR-ish exposure spread
        return torch.from_numpy(c).permute(2, 0, 1)

    def _crop(self, img, rng):
        h, w, _ = img.shape
        s = rng.uniform(1.0, min(2.0, h / self.size, w / self.size))     # source scale
        win = int(self.size * s)
        y, x = rng.randrange(h - win + 1), rng.randrange(w - win + 1)
        c = img[y:y + win, x:x + win]
        if win != self.size:
            c = np.asarray(Image.fromarray(np.clip(c * 255.0, 0, 255).astype(np.uint8))
                           .resize((self.size, self.size), Image.LANCZOS), dtype=np.float32) / 255.0
        c = np.rot90(c, rng.randrange(4))
        if rng.random() < 0.5: c = c[:, ::-1]
        return np.ascontiguousarray(c, dtype=np.float32)

# ---------------------------------------------------------------- forward model
def halton(index, base):
    """Radical inverse, float32 order of operations as plugin.cpp taaHalton."""
    f, r = np.float32(1.0), np.float32(0.0)
    i = int(index)
    while i > 0:
        f = np.float32(f / np.float32(base))
        r = np.float32(r + f * np.float32(i % base))
        i //= base
    return np.float32(r)

def jitter_px(index):
    """The plugin: jitterX = halton(index+1, 2) - 0.5, jitterY = halton(index+1, 3) - 0.5."""
    return float(halton(index + 1, 2) - np.float32(0.5)), float(halton(index + 1, 3) - np.float32(0.5))

def luma(c):  # taa.comp luma
    return 0.2126 * c[:, 0:1] + 0.7152 * c[:, 1:2] + 0.0722 * c[:, 2:3]

def tm(c):    # taa.comp tm: c / (1 + luma)
    return c / (1.0 + luma(c))

def degrade(gt, ratio, jit, phases=8):
    """gt (B,3,S,S) linear -> low-res jittered raster proxy (B,3,s,s).
    Raster footprint ~ Gaussian(sigma = 0.5*ratio px of GT) then point-sample at
    low-res pixel centres shifted by the jitter (in LOW-res pixels)."""
    B, _, S, _ = gt.shape
    s = int(round(S / ratio))
    sigma = 0.5 * ratio
    k = int(2 * math.ceil(2 * sigma) + 1)
    xs = torch.arange(k, device=gt.device, dtype=gt.dtype) - (k - 1) / 2
    g = torch.exp(-0.5 * (xs / sigma) ** 2); g = g / g.sum()
    pre = F.conv2d(F.pad(gt, (k // 2,) * 4, mode="reflect"), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    pre = F.conv2d(F.pad(pre, (k // 2,) * 4, mode="reflect"), g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)
    jx, jy = jit
    # low-res pixel i samples GT position ((i + 0.5 + jx) * ratio) in GT pixels -> normalised [-1,1]
    ii = torch.arange(s, device=gt.device, dtype=gt.dtype)
    gx = ((ii + 0.5 + jx) * ratio) / S * 2 - 1
    gy = ((ii + 0.5 + jy) * ratio) / S * 2 - 1
    grid = torch.stack(torch.meshgrid(gy, gx, indexing="ij")[::-1], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    return F.grid_sample(pre, grid, mode="bilinear", padding_mode="border", align_corners=False)

def baseline_up(lo, S, ratio, jit):
    """Un-jittered bilinear upsample of the low-res raster to SxS (the taau/EASU stand-in)."""
    B, _, s, _ = lo.shape
    jx, jy = jit
    oo = torch.arange(S, device=lo.device, dtype=lo.dtype)
    # output pixel o (centre o+0.5 in GT px) lies at low-res coordinate (o+0.5)/ratio - jx (pixel units)
    gx = (((oo + 0.5) / ratio - jx) / s) * 2 - 1
    gy = (((oo + 0.5) / ratio - jy) / s) * 2 - 1
    grid = torch.stack(torch.meshgrid(gy, gx, indexing="ij")[::-1], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
    return F.grid_sample(lo, grid, mode="bilinear", padding_mode="border", align_corners=False)

# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, padding=1); self.c2 = nn.Conv2d(co, co, 3, padding=1)
    def forward(self, x):
        x = F.leaky_relu(self.c1(x), 0.1); return F.leaky_relu(self.c2(x), 0.1)

class ResidualSR(nn.Module):
    """U-Net-lite, additive residual on the baseline. Input: tm(baseline) RGB + jitter x,y + ratio planes."""
    def __init__(self, ch=(16, 32, 48), in_ch=6):
        super().__init__()
        self.in_ch = in_ch
        self.e0 = Block(in_ch, ch[0]); self.e1 = Block(ch[0], ch[1]); self.e2 = Block(ch[1], ch[2])
        self.d1 = Block(ch[2] + ch[1], ch[1]); self.d0 = Block(ch[1] + ch[0], ch[0])
        self.out = nn.Conv2d(ch[0], 3, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # fresh model == baseline
    def forward(self, base_lin, jit, ratio, aux=None):
        B, _, H, W = base_lin.shape
        x = tm(base_lin)
        planes = torch.stack([torch.full((B, H, W), v, device=x.device, dtype=x.dtype)
                              for v in (jit[0], jit[1], ratio - 1.5)], dim=1)
        x = torch.cat([x, planes], dim=1)
        if aux is not None: x = torch.cat([x, aux.to(x.dtype)], dim=1)   # depth / normal / |vel| at output res
        e0 = self.e0(x); e1 = self.e1(F.avg_pool2d(e0, 2)); e2 = self.e2(F.avg_pool2d(e1, 2))
        d1 = self.d1(torch.cat([F.interpolate(e2, scale_factor=2, mode="bilinear", align_corners=False), e1], 1))
        d0 = self.d0(torch.cat([F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False), e0], 1))
        # Bounded residual: at most +/- the local level plus a little. Unbounded,
        # a high LR ran the output into the tonemap's saturation (run 2: |r| 9e4
        # and climbing, gradients gone). Additive still holds and zero-init still
        # means "exactly the baseline".
        res = torch.tanh(self.out(d0)) * (0.5 + base_lin.abs())
        return base_lin + res, res

# ---------------------------------------------------------------- metrics (metrics.comp analogues)
LAP = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
def lap_energy(c):
    y = luma(tm(c))
    return F.conv2d(y, LAP.to(y.device, y.dtype), padding=1).abs().mean()

def psnr(a, b):
    m = ((tm(a) - tm(b)) ** 2).mean().clamp_min(1e-12)
    return float(10 * torch.log10(1.0 / m))

def make_pair(gt, ratio, idx, phases):
    jit = jitter_px(idx % phases)
    lo = degrade(gt, ratio, jit, phases)
    return baseline_up(lo, gt.shape[-1], ratio, jit), jit

def quiz(model, held, device, log=None):
    """Score against the held-out truths. Returns (all_pass, rows)."""
    model.eval(); rows = []
    dpsnr, lapr, stab_m, stab_b, rmean, nan, hl1 = [], [], [], [], [], 0, []
    # fp32 throughout: the quiz is the yardstick, and fp16 clamps (1e-8, 1e-12)
    # underflow to zero and turn a flat crop into an infinite ratio.
    with torch.no_grad():
        for gt in held:
            gt = gt.to(device).unsqueeze(0).float()
            if float(lap_energy(gt)) < 1e-3: continue          # flat crop: nothing to measure
            for ratio in (1.3, 1.5, 2.0):
                ba, ja = make_pair(gt, ratio, 3, 8); bb, jb = make_pair(gt, ratio, 5, 8)
                oa, ra = model(ba, ja, ratio); ob, _ = model(bb, jb, ratio)
                if not torch.isfinite(oa).all(): nan += 1
                dpsnr.append(psnr(oa, gt) - psnr(ba, gt))
                lapr.append(float(lap_energy(oa) / lap_energy(gt).clamp_min(1e-8)))
                stab_m.append(float((tm(oa) - tm(ob)).abs().mean()))
                stab_b.append(float((tm(ba) - tm(bb)).abs().mean()))
                rmean.append(float(ra.abs().mean()))
                hl1.append(float((tm(oa) - tm(gt)).abs().mean()))
    def row(name, val, ok, thr): rows.append((name, val, ok, thr)); return ok
    ok = True
    ok &= row("psnr_gain_db  (out vs baseline, mean)", float(np.mean(dpsnr)), np.mean(dpsnr) >= 1.0, ">= 1.0")
    ok &= row("psnr_gain_min (worst case)", float(np.min(dpsnr)), np.min(dpsnr) >= 0.0, ">= 0.0")
    ok &= row("lap_ratio     (|lap out| / |lap truth|)", float(np.mean(lapr)), 0.85 <= np.mean(lapr) <= 1.15, "0.85..1.15")
    ok &= row("stability     (out delta across jitter phases / baseline delta)",
              float(np.mean(stab_m) / max(np.mean(stab_b), 1e-8)), np.mean(stab_m) <= 0.6 * np.mean(stab_b), "<= 0.60")
    ok &= row("residual_mean (|r|, additive sanity)", float(np.mean(rmean)), np.mean(rmean) < 0.15, "< 0.15")
    ok &= row("finite        (NaN/Inf outputs)", float(nan), nan == 0, "== 0")
    model.train()
    lines = ["QUIZ %s" % ("PASS" if ok else "FAIL")] + \
            ["  %s  %-52s %10.4f   want %s" % ("PASS" if r[2] else "FAIL", r[0], r[1], r[3]) for r in rows]
    lines.append("  held-out L1 %.5f" % float(np.mean(hl1)))
    print("\n".join(lines), flush=True)
    if log: log.write("\n".join(lines) + "\n"); log.flush()
    return bool(ok), rows, float(np.mean(hl1))

# ---------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=12); ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=6); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--quiz", action="store_true"); ap.add_argument("--images", type=int, default=5000)
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt", "sr_quick.pt"))
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(a.ckpt), exist_ok=True)
    log = open(os.path.join(os.path.dirname(a.ckpt), "quiz.log"), "a")
    torch.manual_seed(0); np.random.seed(0); random.seed(0)

    files = discover(a.images)
    print("stock textures usable: %d" % len(files), flush=True)
    if len(files) < 20: sys.exit("too few stock textures found under %s" % XP_ROOT)
    nh = max(40, len(files) // 20)
    held = [Crops(files[-nh:], n=24, seed=777)[i] for i in range(24)]      # held-out: last 5% of files, fixed crops
    train = Crops(files[:-nh], n=a.rounds * a.steps * a.batch, seed=1)
    dl = torch.utils.data.DataLoader(train, batch_size=a.batch, shuffle=False, num_workers=0)

    model = ResidualSR().to(device)
    print("params: %d" % sum(p.numel() for p in model.parameters()), flush=True)
    if os.path.exists(a.ckpt):
        model.load_state_dict(torch.load(a.ckpt, map_location=device)); print("loaded", a.ckpt)
    if a.quiz:
        ok, _, _ = quiz(model, held, device, log); sys.exit(0 if ok else 1)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.rounds * a.steps)
    # bf16 autocast: same range as fp32, so no GradScaler and no silently
    # skipped optimizer steps (fp16 + scaler left round 0 with a zero residual).
    amp = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and amp == torch.float16))
    it = iter(dl); passes = 0; t0 = time.time()
    # ---- MEMORISATION WATCH. Train L1 (what it fits) vs held-out L1 (what it
    # generalises to). Held-out flat for two rounds while train keeps falling =
    # it is learning the crops, not the forward model. Stop and say so: the
    # cure is more content, not more steps.
    best_held, stale, prev_train = 1e9, 0, 1e9
    for rnd in range(a.rounds):
        run = 0.0; run_l1 = 0.0
        for step in range(a.steps):
            gt = next(it).to(device)
            ratio = random.uniform(1.3, 2.0); phases = 8
            ia, ib = random.randrange(phases), random.randrange(phases)
            with torch.autocast("cuda", dtype=amp, enabled=device == "cuda"):
                ba, ja = make_pair(gt, ratio, ia, phases); bb, jb = make_pair(gt, ratio, ib, phases)
                oa, ra = model(ba.float(), ja, ratio); ob, _ = model(bb.float(), jb, ratio)
                oa, ob = oa.float(), ob.float()
                l1 = (tm(oa) - tm(gt)).abs().mean() + (tm(ob) - tm(gt)).abs().mean()
                llap = (lap_energy(oa) - lap_energy(gt)).abs()
                lstab = (tm(oa) - tm(ob)).abs().mean()            # S_STATIC_DELTA analogue
                # Stability ramps 0.25 -> 1.0 over the rounds: at 2.0 from step
                # one the cheapest answer was a zero residual (round 11: |r| 0.003,
                # lap ratio 0.49, no PSNR gain). Let it learn detail first.
                wst = 0.25 + 0.75 * rnd / max(a.rounds - 1, 1)
                loss = l1 + 0.5 * llap + wst * lstab
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += float(loss.detach()); run_l1 += float(l1.detach()) * 0.5
            if step % 100 == 0:
                print("round %d step %d loss %.4f (l1 %.4f lap %.4f stab %.4f) %.0fs" %
                      (rnd, step, float(loss), float(l1), float(llap), float(lstab), time.time() - t0), flush=True)
        print("round %d done, mean loss %.4f" % (rnd, run / a.steps), flush=True)
        log.write("round %d mean loss %.4f\n" % (rnd, run / a.steps))
        ok, _, held_l1 = quiz(model, held, device, log)
        train_l1 = run_l1 / a.steps
        # Keep only the BEST checkpoint by held-out L1. Run 2 diverged and
        # overwrote a good round with garbage; a worse round never saves now.
        if held_l1 < best_held - 1e-4:
            best_held, stale = held_l1, 0
            torch.save(model.state_dict(), a.ckpt); print("  saved (best held-out L1 %.5f)" % held_l1, flush=True)
        else: stale += 1
        msg = "round %d: train L1 %.5f  held-out L1 %.5f  gap %+.5f  stale %d" % (rnd, train_l1, held_l1, held_l1 - train_l1, stale)
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
        # Memorising means "near its best on held-out and still fitting train";
        # a diverged model (held-out far above best) is not memorising.
        # Gap-based: run 3 sat at train 0.0065 vs held-out 0.0177 for eleven
        # rounds (2.7x) while train L1 only jittered, so a "train still falling"
        # test never fired. Memorising is the GAP, not the slope.
        if stale >= 2 and held_l1 > 2.0 * train_l1 and held_l1 < best_held * 1.05:
            m = "MEMORISING: held-out flat for %d rounds while train L1 still falls - stopping. Needs more content (real captures), not more steps." % stale
            print(m, flush=True); log.write(m + "\n"); break
        prev_train = train_l1
        passes = passes + 1 if ok else 0
        if passes >= 2:
            print("PERFECT: quiz passed twice in a row after round %d" % rnd, flush=True); break
    log.close()

if __name__ == "__main__":
    main()
