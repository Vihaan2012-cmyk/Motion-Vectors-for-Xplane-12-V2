"""gbuf_view.py - decode every plane of one capture frame into a labelled contact
sheet + per-channel stats, so the G-buffer attachments can be named by eye
(albedo / material / emissive / ...). numpy only; never touches the GPU.

    python tools/nn/gbuf_view.py                          # first non-cockpit frame it finds
    python tools/nn/gbuf_view.py D:/NNCap/<session>/frame_000123

Copyright (C) 2026 MotionVectors contributors. SPDX: GPL-3.0-or-later
"""
import glob, json, os, sys
import numpy as np
from PIL import Image, ImageDraw

here = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(here, "out"); os.makedirs(OUT, exist_ok=True)

def ufloat(u, mbits):            # 5-bit exponent, mbits mantissa, bias 15 (B10G11R11)
    e = ((u >> mbits) & 0x1F).astype(np.float32); m = (u & ((1 << mbits) - 1)).astype(np.float32) / (1 << mbits)
    v = np.where(e == 0, m * 2.0 ** -14, (1 + m) * 2.0 ** (e - 15)).astype(np.float32)
    return np.where(e == 31, np.nan, v)                     # e=31 is Inf/NaN in the spec

def decode(path, p):
    w, h, fmt = p["w"], p["h"], p["format"]
    if p["dtype"] == "u8":  return np.fromfile(path, np.uint8).reshape(h, w, p["c"]).astype(np.float32) / 255.0
    if p["dtype"] == "f16": return np.fromfile(path, np.float16).reshape(h, w, p["c"]).astype(np.float32)
    u = np.fromfile(path, np.uint32).reshape(h, w)
    if fmt == "A2B10G10R10_UNORM":      # bits 9..0 R, 19..10 G, 29..20 B, 31..30 A
        return np.stack([(u & 1023) / 1023.0, ((u >> 10) & 1023) / 1023.0, ((u >> 20) & 1023) / 1023.0, (u >> 30) / 3.0], -1).astype(np.float32)
    if fmt == "B10G11R11_UFLOAT":       # bits 10..0 R(11), 21..11 G(11), 31..22 B(10)
        return np.stack([ufloat(u & 0x7FF, 6), ufloat((u >> 11) & 0x7FF, 6), ufloat(u >> 22, 5)], -1)
    raise ValueError(fmt)

def pick_frame():
    for m in sorted(glob.glob("D:/NNCap/*/frame_*/manifest.json") + glob.glob("E:/NNCap/*/frame_*/manifest.json")):
        j = json.load(open(m))
        if j.get("view_type") != 1026: return os.path.dirname(m), j
    raise SystemExit("no non-cockpit frame found")

def main():
    if len(sys.argv) > 1: fd = sys.argv[1]; j = json.load(open(os.path.join(fd, "manifest.json")))
    else: fd, j = pick_frame()
    print("frame:", fd, "view", j.get("view_type"), "sun_view", j.get("sun_view"))
    tiles = []
    for name, p in j["planes"].items():
        a = decode(os.path.join(fd, name + ".bin"), p)
        c = a.shape[2]
        print("%-9s %-20s c=%d" % (name, p["format"], c))
        for k in range(c):
            ch = a[..., k]; fin = ch[np.isfinite(ch)]
            print("   ch%d nan%% %.1f mean %.4f std %.4f p1 %.4f p99 %.4f max %.4f distinct %d" %
                  (k, 100.0 * (1 - fin.size / ch.size), fin.mean(), fin.std(), np.percentile(fin, 1), np.percentile(fin, 99), fin.max(), min(len(np.unique(fin[::97])), 99999)))
        s = a[::4, ::4]
        if name == "velocity": rgb = np.stack([np.abs(s[..., 0]) * 20, np.abs(s[..., 1]) * 20, np.zeros_like(s[..., 0])], -1)
        elif c == 2: rgb = np.stack([(s[..., 0] + 1) * 0.5, (s[..., 1] + 1) * 0.5, np.zeros_like(s[..., 0])], -1)
        elif p["dtype"] == "f16" or "UFLOAT" in p["format"]: rgb = s[..., :3] / (1 + s[..., :3])
        else: rgb = s[..., :3]
        rgb = np.nan_to_num(rgb)
        im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)); ImageDraw.Draw(im).text((6, 6), "%s %s" % (name, p["format"]), fill=(255, 255, 0))
        tiles.append(im)
        if c == 4 and p["dtype"] != "f16":                       # alpha of packed planes as its own tile
            al = Image.fromarray((np.clip(s[..., 3], 0, 1) * 255).astype(np.uint8)).convert("RGB"); ImageDraw.Draw(al).text((6, 6), name + " .a", fill=(255, 255, 0)); tiles.append(al)
    tw, th = tiles[0].size; cols = 3; rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (tw * cols, th * rows))
    for i, t in enumerate(tiles): sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    sheet.save(os.path.join(OUT, "gbuf_sheet.png")); print("wrote", os.path.join(OUT, "gbuf_sheet.png"), sheet.size)

if __name__ == "__main__":
    main()
