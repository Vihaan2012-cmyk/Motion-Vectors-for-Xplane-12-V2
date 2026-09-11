"""pack_captures.py - turn capture sessions into fast training crops (one-time).

Each K-frame sequence (same grouping as train_recurrent.list_sequences) is
decoded once, its step-0 history warped once at full resolution, and cut into
CROPS random windows that pass the brightness floor. Every crop is one .npz
of fp16 arrays (~10 MB) - what the loader reads per item instead of three
150 MB frames. Flips still happen at load time, so nothing is lost.

    python tools/nn/pack_captures.py                      # D:/NNCap + E:/NNCap -> E:/NNCap/packed
    python tools/nn/pack_captures.py --out E:/NNCap/packed --crops 8 --size 384

Copyright (C) 2026 MotionVectors contributors. SPDX: GPL-3.0-or-later
"""
import argparse, os, random, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quick_train import tm, luma
import train_captures as tc
import train_recurrent as tr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/NNCap;E:/NNCap"); ap.add_argument("--out", default="E:/NNCap/packed")
    ap.add_argument("--crops", type=int, default=8); ap.add_argument("--size", type=int, default=384)
    a = ap.parse_args()
    roots = [r.strip() for r in a.root.split(";")]
    tc.calibrate_warp(tc.list_frames(roots))
    seqs = tr.list_sequences(roots)
    os.makedirs(a.out, exist_ok=True)
    print("sequences: %d -> %s (%d crops each, %dpx)" % (len(seqs), a.out, a.crops, a.size), flush=True)
    t0 = time.time(); written = 0; S = a.size
    hg = tr.held_groups(seqs)
    print("held-out groups: %d of %d" % (len(hg), len(set(s[2] for s in seqs))), flush=True)
    for si, (frames, prev, group) in enumerate(seqs):
        tag = "held" if group in hg else "train"
        if all(os.path.exists(os.path.join(a.out, "seq%05d_c%d_%s.npz" % (si, c, tag))) for c in range(a.crops)): continue
        gt, aux, vel, hist0 = tr.seq_tensors(frames, prev)
        _, _, H, W = gt.shape
        rng = random.Random(1000 + si)
        for c in range(a.crops):
            for _ in range(8):
                y, x = rng.randrange(H - S + 1), rng.randrange(W - S + 1)
                if float(luma(tm(gt[-1:, :, y:y + S, x:x + S])).mean()) >= tr.LUMA_FLOOR: break
            np.savez(os.path.join(a.out, "seq%05d_c%d_%s.npz" % (si, c, tag)),
                     gt=gt[:, :, y:y + S, x:x + S].numpy().astype(np.float16),
                     aux=aux[:, :, y:y + S, x:x + S].numpy().astype(np.float16),
                     vel=vel[:, :, y:y + S, x:x + S].numpy().astype(np.float16),
                     hist0=hist0[:, y:y + S, x:x + S].numpy().astype(np.float16),
                     dims=np.array([W, H], dtype=np.float32))
            written += 1
        if si % 20 == 0: print("  %d/%d sequences, %d crops, %.0fs" % (si + 1, len(seqs), written, time.time() - t0), flush=True)
    print("done: %d crops written in %.0fs" % (written, time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
