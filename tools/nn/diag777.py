#!/usr/bin/env python3
"""diag777.py - what the FlightFactor 777 does to the motion-vector field, from one capture flight.

Inputs: a capture session written by the layer (nn.capture, tag ff777) and the diagnostics folder
tools/ff777_suite.ps1 filled beside it (mv_pid_table_<pid>.txt, shaders/pipelines.csv, shaders/*.spv,
the layer trace). For the last frame of every burst it rebuilds the velocity the injected vertex
shaders SHOULD have written from the captured depth and the frame's matrices (world reprojection
and body-frame reprojection, both in the manifest), classifies every pixel

    WORLD      matches the world reprojection      (terrain, buildings, the world seen through glass)
    BODY       matches the body-frame reprojection (cockpit, airframe)
    ZERO       no motion written where the world moved (an unpatched or masked draw claimed the pixel)
    UNWRITTEN  the sentinel: nothing drew here (sky, clouds) or the writer was masked
    OTHER      a vector that fits neither matrix (wrong matrix, clamp, garbage)

measures where the resolve SMEARS (its output leaves the current frame's 3x3 tonemapped range), and,
in TAA_MV_PID runs, reads the pipeline id the velocity target carries in its depth channel so the
smearing pixels name their writers. Pipeline ids join to vertex/fragment module hashes through the
pid table, to the dumped .spv files, and to X-Plane's own shader names by hashing the shipped pack.

    python tools/nn/diag777.py --session E:/NNCap/<stamp>_ff777 --diag E:/NNCap/ff777_diag_<stamp> --xplane "D:/Steam Games/steamapps/common/X-Plane 12"

Writes <diag>/report.md and <diag>/sheets/*.png. Run it on a 737 session first: the report must say
WORLD dominates and the smear fraction is small - that is the calibration of the matrix convention.

Copyright (C) 2026 MotionVectors contributors. SPDX: GPL-3.0-or-later
"""
import argparse, glob, json, os, re, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_captures as tc

INTERIOR = {1000, 1017, 1026}       # X-Plane interior view types; everything else is treated as external
CLASSES = ["WORLD", "BODY", "ZERO", "UNWRITTEN", "OTHER"]
CLASS_RGB = np.array([[40, 160, 40], [60, 90, 220], [230, 40, 40], [30, 30, 30], [230, 190, 30]], dtype=np.uint8)

def tm(c):
    l = 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]
    return c / (1.0 + l[..., None])

def mat_apply(M, clip, colmajor):
    """M: 16 floats; clip: (H,W,4). Returns (H,W,4)."""
    m = np.array(M, dtype=np.float64).reshape(4, 4)
    m = m.T if colmajor else m          # column-major storage: element [c*4+r] -> reshape gives rows=c
    return clip @ m.T

def predict(M, clip, ndc, colmajor):
    p = mat_apply(M, clip, colmajor)
    w = p[..., 3:4]; w = np.where(np.abs(w) < 1e-6, 1e-6, w)
    pn = p[..., :2] / w
    return 0.5 * (pn - ndc)             # (prevNDC - currNDC) * 0.5, the stored convention

def load_depth(f, m):
    if "depth" not in m["planes"]: return None
    d = tc.load_plane(f, m, "depth").astype(np.float32)[..., 0]
    d = np.nan_to_num(d)
    # the resolve-time depth copy reads X-Plane's Z-compressed depth raw: every other column can be
    # zero (measured 2026-09-09). Fill such columns from their neighbours.
    zero_cols = (d == 0).mean(axis=0) > 0.9
    if zero_cols.any() and not zero_cols.all():
        good = np.where(~zero_cols)[0]
        d = np.stack([np.interp(np.arange(d.shape[1]), good, row[good]) for row in d], axis=0).astype(np.float32)
    if (d == 0).mean() > 0.99: return None
    return d

def burst_frames(session):
    frames = []
    for f in sorted(glob.glob(os.path.join(session, "frame_*"))):
        mp = os.path.join(f, "manifest.json")
        if os.path.exists(mp): frames.append((f, json.load(open(mp))))
    bursts = collections.OrderedDict()
    for f, m in frames:
        b = (m.get("burst") or {}).get("id", -1); bursts.setdefault(b, []).append((f, m))
    return [v for v in bursts.values() if len(v) >= 2]

def parse_pid_table(diag):
    vs, pipes = {}, {}
    for p in glob.glob(os.path.join(diag, "mv_pid_table_*.txt")):
        for line in open(p, errors="replace"):
            mm = re.match(r"vs (\d+) ([0-9a-f]{16})", line)
            if mm: vs[int(mm.group(1))] = mm.group(2); continue
            mm = re.match(r"PIPE vs=(\d+) frag=([0-9a-f]{16}) (.*)", line)
            if mm:
                pid = int(mm.group(1)); rest = mm.group(3)
                d = dict(re.findall(r"(\w+)=(-?\d+)", rest)); d["frag"] = mm.group(2); d["flags"] = re.sub(r"\S+=\S+ ?", "", rest).strip()
                pipes.setdefault(pid, []).append(d)
    return vs, pipes

def shader_names(xplane, diag):
    names = {}
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        import gen_family_table as g
        for pack in (os.path.join(xplane, "Resources", "shaders", "bin", "spv"), os.path.join(xplane, "Resources", "shaders", "bin", "spv.zip")):
            if os.path.exists(pack):
                for fam, mod, data in g.iter_pack(pack): names.setdefault("%016x" % g.fnv1a64_words(data), "%s/%s" % (fam, mod))
                break
    except Exception as e:
        print("  (shader names unavailable: %s)" % e)
    # the family table's own comments name the classified families too
    tbl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "vklayer", "mv_family_table.h")
    if os.path.exists(tbl):
        for line in open(tbl):
            mm = re.match(r"\s*0x([0-9a-f]{16})ull, // (\S+)", line)
            if mm: names.setdefault(mm.group(1), mm.group(2))
    return names

def analyse_burst(frames, calib):
    f, m = frames[-1]
    vel = np.nan_to_num(tc.load_plane(f, m, "velocity").astype(np.float32), nan=-2000.0)
    col = np.clip(np.nan_to_num(tc.load_plane(f, m, "color").astype(np.float32)[..., :3]), 0, 64)
    res = tc.resolved_linear(f, m)
    H, W = vel.shape[:2]
    vx, vy = vel[..., 0], vel[..., 1]
    unwritten = (vx < -1000) | (vy < -1000) | (np.abs(vx) > tc.VEL_SENTINEL) | (np.abs(vy) > tc.VEL_SENTINEL)
    pid_mode = int(m.get("pid_mode", 0)) == 1
    depth = load_depth(f, m)
    if depth is None and not pid_mode:
        # clip-w rides in velocity.z when no pid tag replaces it
        w = np.clip(np.nan_to_num(vel[..., 2], nan=0.0, posinf=65504.0), 0, 65504)
    elif depth is not None:
        proj = m["proj"]; A, B = proj[10], proj[14]
        vz = -B / (depth + A + 1e-12)
        w = np.abs(vz)
    else:
        w = None
    out = {"frame": os.path.basename(f), "view": int(m.get("view_type", 0)), "W": W, "H": H, "unwritten": float(unwritten.mean()),
           "body_valid": int(m.get("body_valid", 0)), "near_field_m": m.get("near_field_m"), "cockpit_pass": m.get("cockpit_pass"), "pid_mode": pid_mode}
    u = (np.arange(W) + 0.5) / W; v = (np.arange(H) + 0.5) / H
    ndc = np.stack(np.meshgrid(2 * u - 1, 1 - 2 * v, indexing="xy"), axis=-1)
    cls = np.full((H, W), 3, dtype=np.uint8)          # UNWRITTEN
    predW = None
    if w is not None:
        # THE MATRIX ACTS ON (clip.x, clip.y, clip.w, 1), NOT ON A CLIP-SPACE POINT. Measured on
        # the 737 captures 2026-09-12: this form, column-major, fits 96% of written pixels within
        # 0.75 px; a true clip vector (x, y, z, w) fits 0%. It is the injector's own K-form - the
        # varying's z is dead, the matrix reads w in slot 2 and 1 in slot 3.
        clip = np.concatenate([ndc * w[..., None], w[..., None], np.ones_like(w)[..., None]], axis=-1).astype(np.float64)
        written = ~unwritten
        # calibrate the matrix storage order and the stored sign per axis once, on the pixels that fit best
        if calib.get("cm") is None:
            best = None
            for cm in (True, False):
                pw = predict(m["reproj"], clip, ndc, cm)
                for sx in (1, -1):
                    for sy in (1, -1):
                        d = np.abs(np.stack([sx * pw[..., 0] - vx, sy * pw[..., 1] - vy], -1)) * np.array([W, H])
                        fit = float(((d.max(-1) < 0.75) & written).sum())
                        if best is None or fit > best[0]: best = (fit, cm, sx, sy)
            calib["cm"], calib["sx"], calib["sy"] = best[1], best[2], best[3]
            out["calib"] = "colmajor=%s sign=(%+d,%+d) fit=%.1f%%" % (best[1], best[2], best[3], 100 * best[0] / max(written.sum(), 1))
        sgn = np.array([calib["sx"], calib["sy"]])
        predW = predict(m["reproj"], clip, ndc, calib["cm"]) * sgn
        dW = (np.abs(predW - vel[..., :2]) * np.array([W, H])).max(-1)
        velpx = (np.abs(vel[..., :2]) * np.array([W, H])).max(-1)
        predpx = (np.abs(predW) * np.array([W, H])).max(-1)
        cls[written & (dW < 0.75)] = 0
        if int(m.get("body_valid", 0)) and "body" in m:
            predB = predict(m["body"], clip, ndc, calib["cm"]) * sgn
            dB = (np.abs(predB - vel[..., :2]) * np.array([W, H])).max(-1)
            cls[written & (dW >= 0.75) & (dB < 0.75)] = 1
        zero = written & (cls == 3) & (velpx < 0.25) & (predpx > 1.0)
        cls[zero] = 2
        cls[written & (cls == 3)] = 4
        far = written & (w > 30.0)          # world motion: pixels beyond cockpit distance
        out["camera_px"] = float(np.median(predpx[far])) if far.any() else 0.0
        out["pred_max_px"] = float(np.percentile(predpx[far], 99)) if far.any() else 0.0
    # smear: the resolve outside the current frame's 3x3 tonemapped range
    ct = tm(col); rt = tm(res)
    lo = ct.copy(); hi = ct.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            s = np.roll(np.roll(ct, dy, 0), dx, 1); lo = np.minimum(lo, s); hi = np.maximum(hi, s)
    smear = ((rt < lo - 0.04) | (rt > hi + 0.04)).any(-1)
    out["smear"] = float(smear.mean()); out["smear_top"] = float(smear[: H // 2].mean()); out["smear_bottom"] = float(smear[H // 2:].mean())
    out["classes"] = {c: float((cls == i).mean()) for i, c in enumerate(CLASSES)}
    out["classes_in_smear"] = {c: float(((cls == i) & smear).sum() / max(smear.sum(), 1)) for i, c in enumerate(CLASSES)}
    pids = None
    if pid_mode:
        tag = np.nan_to_num(vel[..., 2], nan=0.0)
        pid = np.floor(tag).astype(np.int64); half = (tag - pid) >= 0.5
        pids = {}
        for mask, key in ((smear & ~unwritten, "smear"), (~unwritten, "all")):
            c = collections.Counter(zip(pid[mask].tolist(), half[mask].tolist()))
            pids[key] = c.most_common(12)
        out["pids"] = pids
    return out, cls, smear, col, (pid if pid_mode else None)

def sheet(path, col, cls, smear, pid):
    from PIL import Image
    def srgb(c):
        t = tm(c); return (np.clip(t, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    a = srgb(col); b = CLASS_RGB[cls]
    c = np.repeat((smear * 255).astype(np.uint8)[..., None], 3, axis=2)
    tiles = [a, b, c]
    if pid is not None:
        rng = np.random.RandomState(3); pal = rng.randint(30, 255, size=(4096, 3)).astype(np.uint8); pal[0] = 0
        tiles.append(pal[np.clip(pid, 0, 4095)])
    img = np.concatenate([t[::2, ::2] for t in tiles], axis=1)
    Image.fromarray(img).save(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True); ap.add_argument("--diag", required=True)
    ap.add_argument("--xplane", default="D:/Steam Games/steamapps/common/X-Plane 12"); ap.add_argument("--max-bursts", type=int, default=400)
    a = ap.parse_args()
    bursts = burst_frames(a.session)[: a.max_bursts]
    if not bursts: print("no bursts in", a.session); sys.exit(2)
    vs, pipes = parse_pid_table(a.diag)
    names = shader_names(a.xplane, a.diag)
    csv = {}
    pcsv = os.path.join(a.diag, "shaders", "pipelines.csv")
    if os.path.exists(pcsv):
        for line in list(open(pcsv))[1:]:
            p = line.strip().split(",")
            if len(p) == 8: csv.setdefault(int(p[1]), []).append(p)
    os.makedirs(os.path.join(a.diag, "sheets"), exist_ok=True)
    calib = {}; rows = []; worst = {"interior": [], "external": []}
    print("bursts: %d" % len(bursts))
    for i, fr in enumerate(bursts):
        try:
            out, cls, smear, col, pid = analyse_burst(fr, calib)
        except Exception as e:
            print("  burst %d failed: %s" % (i, e)); continue
        out["kind"] = "interior" if out["view"] in INTERIOR else "external"
        rows.append(out)
        worst[out["kind"]].append((out["smear"], i, cls, smear, col, pid))
        if "calib" in out: print("  matrix calibration:", out["calib"])
        if i % 20 == 0: print("  %d/%d %s view %d smear %.3f classes %s" % (i + 1, len(bursts), out["kind"], out["view"], out["smear"], {k: round(v, 3) for k, v in out["classes"].items()}))
    for kind in worst:
        for s, i, cls, smear, col, pid in sorted(worst[kind], key=lambda t: -t[0])[:3]:
            sheet(os.path.join(a.diag, "sheets", "%s_burst%03d_smear%.3f.png" % (kind, i, s)), col, cls, smear, pid)

    L = ["# FF 777 diagnostic report", "", "session: `%s`  bursts analysed: %d" % (a.session, len(rows)),
         "matrix calibration: %s" % next((r["calib"] for r in rows if "calib" in r), "n/a"), ""]
    for kind in ("interior", "external"):
        rs = [r for r in rows if r["kind"] == kind]
        if not rs: L += ["## %s views: no bursts" % kind, ""]; continue
        L += ["## %s views (%d bursts, view types %s)" % (kind, len(rs), sorted(set(r["view"] for r in rs))), ""]
        mean = lambda k: np.mean([r[k] for r in rs if k in r]) if any(k in r for r in rs) else float("nan")
        L += ["| metric | mean | worst |", "|---|---|---|"]
        for k, lab in (("smear", "resolve smear fraction"), ("smear_top", "smear, top half"), ("smear_bottom", "smear, bottom half"), ("unwritten", "unwritten fraction"), ("camera_px", "median world motion px"), ("pred_max_px", "99th pct world motion px")):
            vals = [r[k] for r in rs if k in r]
            if vals: L.append("| %s | %.4f | %.4f |" % (lab, np.mean(vals), np.max(vals)))
        L += ["", "| class | share of frame | share of smear pixels |", "|---|---|---|"]
        for c in CLASSES:
            L.append("| %s | %.3f | %.3f |" % (c, np.mean([r["classes"][c] for r in rs]), np.mean([r["classes_in_smear"][c] for r in rs])))
        L += ["", "body_valid %s, near_field_m %s, cockpit_pass %s, pid_mode %s" % (
            sorted(set(r["body_valid"] for r in rs)), sorted(set(str(r["near_field_m"]) for r in rs)), sorted(set(str(r["cockpit_pass"]) for r in rs)), sorted(set(r["pid_mode"] for r in rs))), ""]
        if any("pids" in r for r in rs):
            agg = collections.Counter(); tot = collections.Counter()
            for r in rs:
                for (p, half), n in r["pids"]["smear"]: agg[(p, half)] += n
                for (p, half), n in r["pids"]["all"]: tot[(p, half)] += n
            L += ["### writers under the smear (pipeline id, +0.5 = no depth write)", "", "| pid | smear px | frame px | vertex module | fragment / flags |", "|---|---|---|---|---|"]
            for (p, half), n in agg.most_common(15):
                vh = vs.get(p, "?"); vn = names.get(vh, vh)
                pl = pipes.get(p, [])
                desc = "; ".join("%s blend%s depthW%s colour%s %s" % (names.get(d.get("frag", ""), d.get("frag", "?"))[:40], d.get("blend0", "?"), d.get("depthWrite", "?"), d.get("colour", "?"), d.get("flags", "")) for d in pl[:2]) or "-"
                L.append("| %d%s | %d | %d | %s | %s |" % (p, "+0.5" if half else "", n, tot[(p, half)], vn, desc))
            L.append("")
    L += ["## reading the report", "",
          "- WORLD dominant and smear small: the field is right; any 777 problem is elsewhere (clamp, history).",
          "- smear pixels mostly ZERO or OTHER with blended, no-depth-write writers: a glass/overlay draw claims the pixel with a wrong vector - mask that pipeline.",
          "- smear pixels mostly UNWRITTEN: the writer is masked already; the resolve's no-vector path (matrix reprojection or 'never move') is what smears.",
          "- smear mostly WORLD with 99th-pct motion near or above 64 px: the vector clamp / velMax is the limit, not the writer.",
          "- BODY on the world (terrain classed BODY in interior views): the near-field select took the body matrix for world geometry; check near_field_m and cockpit_pass.",
          "", "sheets/: colour | class map (green WORLD, blue BODY, red ZERO, black UNWRITTEN, yellow OTHER) | smear mask | pid map"]
    open(os.path.join(a.diag, "report.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L[:40]))
    print("... report written to", os.path.join(a.diag, "report.md"))

if __name__ == "__main__":
    main()
