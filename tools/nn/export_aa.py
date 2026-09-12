#!/usr/bin/env python3
"""export_aa.py - write an AA-net checkpoint as the layer's nn_aa.bin weight blob.

Layout (little-endian):
  char[4] "MVNN", u32 version=1, u32 in_ch, u32 nlayers
  nlayers x (u32 cout, u32 cin, u32 ksize)            -- ksize is 3 for every layer of this net
  then, per layer in order: float32 w[cout][cin][3][3], float32 b[cout]

Layer order matches src/vklayer/nnaa.h: e0.c1 e0.c2 e1.c1 e1.c2 e2.c1 e2.c2 d1.c1 d1.c2 d0.c1 d0.c2 out.
Concat inputs keep torch's order: d1 = cat(up(e2), e1), d0 = cat(up(d1), e0).

    python tools/nn/export_aa.py --ckpt tools/nn/ckpt/aa_budget.pt --out config/nn_aa.bin

Copyright (C) 2026 MotionVectors contributors. SPDX: GPL-3.0-or-later
"""
import argparse, os, struct, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_aa import AANet, IN_CH

ORDER = ["e0.c1", "e0.c2", "e1.c1", "e1.c2", "e2.c1", "e2.c2", "d1.c1", "d1.c2", "d0.c1", "d0.c2", "out"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt", "aa_budget.pt"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config", "nn_aa.bin"))
    ap.add_argument("--ch", default="8,16,24")
    a = ap.parse_args()
    ch = tuple(int(c) for c in a.ch.split(","))
    model = AANet(ch=ch, in_ch=IN_CH); model.load_state_dict(torch.load(a.ckpt, map_location="cpu")); model.eval()
    sd = model.state_dict()
    layers = []
    for name in ORDER:
        w = sd[name + ".weight"].float().numpy(); b = sd[name + ".bias"].float().numpy()
        assert w.shape[2] == 3 and w.shape[3] == 3, name
        layers.append((name, w, b))
    out = bytearray(b"MVNN" + struct.pack("<III", 1, IN_CH, len(layers)))
    for name, w, b in layers: out += struct.pack("<III", w.shape[0], w.shape[1], 3)
    nfloat = 0
    for name, w, b in layers:
        out += np.ascontiguousarray(w, dtype="<f4").tobytes(); out += np.ascontiguousarray(b, dtype="<f4").tobytes()
        nfloat += w.size + b.size
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "wb").write(bytes(out))
    print("wrote %s: %d layers, %d floats (%.1f KB) from %s" % (a.out, len(layers), nfloat, len(out) / 1024.0, a.ckpt))
    for name, w, b in layers: print("  %-6s cout %2d cin %2d" % (name, w.shape[0], w.shape[1]))

if __name__ == "__main__":
    main()
