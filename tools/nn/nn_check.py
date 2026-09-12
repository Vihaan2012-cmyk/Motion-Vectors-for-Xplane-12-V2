#!/usr/bin/env python3
"""nn_check.py - prove the in-layer AA net IS the torch model.

Reads %TEMP%\\nn_aa_dump.bin (written once by the layer with taa.nn_dump=1): the six-layer
input map the resolve wrote, the last conv's output (the pre-activation residual r) and the
history image after the post pass. Rebuilds the 17-channel tensor exactly as nnaa.h lays it
out, runs the same checkpoint through train_aa.AANet, and reports the difference against the
shader's r and against the final image (where x < split). fp16 storage between layers puts
the expected agreement around 1e-2 in r; anything larger is a layout, order or convention
error, and the per-channel input statistics printed first say which plane is off.

    python tools/nn/nn_check.py --ckpt tools/nn/ckpt/aa_budget.pt [--dump %TEMP%/nn_aa_dump.bin]

Copyright (C) 2026 MotionVectors contributors. SPDX: GPL-3.0-or-later
"""
import argparse, os, struct, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_aa import AANet, IN_CH
from quick_train import tm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(os.environ.get("TEMP", "."), "nn_aa_dump.bin"))
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt", "aa_budget.pt"))
    ap.add_argument("--ch", default="8,16,24"); ap.add_argument("--split", type=float, default=0.5)
    a = ap.parse_args()
    raw = open(a.dump, "rb").read()
    magic, w, h, planes = struct.unpack("<IIII", raw[:16])
    assert magic == 0x444E564D and planes >= 8, "not an nnaa dump"
    px = np.frombuffer(raw[16:], dtype=np.float16).reshape(planes, h, w, 4).astype(np.float32)
    L = [px[i] for i in range(6)]; r_gpu = px[6][..., :3]; final = px[7][..., :3]
    # the 17 net channels in nnaa.h order
    chans = [L[0][..., 0], L[0][..., 1], L[0][..., 2], L[0][..., 3],
             L[1][..., 0], L[1][..., 1], L[1][..., 2], L[1][..., 3],
             L[2][..., 0], L[2][..., 1], L[2][..., 2], L[2][..., 3],
             L[3][..., 0], L[3][..., 1], L[3][..., 2], L[3][..., 3], L[4][..., 0]]
    names = ["tm.r", "tm.g", "tm.b", "jit.x", "jit.y", "ratio-1.5", "depth", "nrm.x", "nrm.y", "|vel|", "valid", "cov", "dz", "htm.r", "htm.g", "htm.b", "hvalid"]
    print("dump %dx%d; input channel statistics (mean / min / max):" % (w, h))
    for n, c in zip(names, chans): print("  %-9s %9.4f %9.4f %9.4f" % (n, c.mean(), c.min(), c.max()))
    base = L[4][..., 1:4]; shade = L[5][..., :3]
    x = torch.from_numpy(np.stack(chans, 0))[None]                       # (1,17,H,W)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AANet(ch=tuple(int(c) for c in a.ch.split(",")), in_ch=IN_CH).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev)); model.eval()
    # AANet.forward builds tm(base) + jitter/ratio planes itself; feed it the dump's base and jitter, and the 14 aux channels
    jit = torch.tensor([[float(chans[3].mean()), float(chans[4].mean())]])
    aux = x[:, 6:]                                                         # depth .. hvalid (11 channels)
    with torch.no_grad():
        out, res = model(torch.from_numpy(base.transpose(2, 0, 1))[None].to(dev), jit.to(dev), 1.0, aux.to(dev))
        # the shader's r is the pre-tanh conv output: recover it from res = tanh(r) * (0.5 + |base|)
        b = torch.from_numpy(base.transpose(2, 0, 1))[None].to(dev)
        r_ref = torch.atanh((res / (0.5 + b.abs())).clamp(-0.999, 0.999))[0].permute(1, 2, 0).cpu().numpy()
        o_ref = (out[0].permute(1, 2, 0).cpu().numpy() * shade).clip(0, None)
    d = np.abs(r_gpu - r_ref); m = np.abs(r_ref) < 3.0            # atanh is ill-conditioned near saturation
    print("residual r: shader vs torch  mean |d| %.5f  max %.4f  (over %.1f%% of pixels with |r|<3); |r| mean %.4f" % (d[m].mean(), d[m].max(), 100 * m.mean(), np.abs(r_ref).mean()))
    xs = int(w * a.split)
    df = np.abs(final[:, :xs] - o_ref[:, :xs]); dt = np.abs(tm(torch.from_numpy(final[None].transpose(0, 3, 1, 2)))[0].permute(1, 2, 0).numpy()[:, :xs] - tm(torch.from_numpy(o_ref[None].transpose(0, 3, 1, 2)))[0].permute(1, 2, 0).numpy()[:, :xs])
    print("final image (x < split): mean |d| linear %.5f  tonemapped %.5f  max tonemapped %.4f" % (df.mean(), dt.mean(), dt.max()))
    print("right of the split (the resolve): mean |final - base*shade| tonemapped %.5f (should be small but nonzero: the resolve is base blended with history)" % (
        np.abs(tm(torch.from_numpy(final[None].transpose(0, 3, 1, 2)))[0].permute(1, 2, 0).numpy()[:, xs:] - tm(torch.from_numpy((base * shade)[None].transpose(0, 3, 1, 2)))[0].permute(1, 2, 0).numpy()[:, xs:]).mean()))
    verdict = "PASS" if d[m].mean() < 0.02 and dt.mean() < 0.005 else "FAIL"
    print("VERDICT %s (want residual mean |d| < 0.02 and final tonemapped mean |d| < 0.005)" % verdict)

if __name__ == "__main__":
    main()
