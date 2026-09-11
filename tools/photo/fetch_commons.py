#!/usr/bin/env python3
"""fetch_commons.py - build a licence-clean aerial photo reference set from Wikimedia Commons.

Walks aerial-photography categories through the official MediaWiki API (breadth-first,
depth-limited), keeps only files whose licence permits reuse without share-alike or
non-commercial strings (public domain, CC0, CC BY), records licence + author + source per
file for attribution, and downloads a 4096 px rendition when the original is huge. Not a
scraper: one API call at a time, a real User-Agent, resumable, and a byte cap.

Domain filters (the reference set is for a photoreal pass, so it must look like a
photograph taken from a modern aircraft): files dated before --min-year are skipped,
Historic American Engineering / Buildings Survey records (HAER/HABS, black-and-white film)
are skipped by title, credit and category, and every download is checked for colour -
grayscale files are deleted on arrival. --clean applies the same rules to what is already
on disk and rewrites the manifest.

    python tools/photo/fetch_commons.py --out E:/PhotoRef --max-gb 50
    python tools/photo/fetch_commons.py --out E:/PhotoRef --clean

Output: <out>/commons/<category>/<file>, <out>/manifest.jsonl (one line per file: url, licence,
author, credit, category, dims, bytes, sha1), <out>/fetch.log.

Copyright (C) 2026 MotionVectors contributors. SPDX: GPL-3.0-or-later
"""
import argparse, json, os, re, sys, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://commons.wikimedia.org/w/api.php"
UA = "MotionVectors-PhotoRef/0.2 (X-Plane neural rendering research; https://github.com/Vihaan2012-cmyk/Motion-Vectors-for-Xplane-12-V2)"
SEEDS = [
    "Aerial photographs of Los Angeles",
    "Aerial photographs of Los Angeles County, California",
    "Aerial photographs of California",
    "Aerial photographs of San Francisco",
    "Aerial photographs of San Diego",
    "Aerial photographs of Nevada",
    "Aerial photographs of Arizona",
    "Aerial photographs of airports in the United States",
    "Aerial photographs of coasts",
    "Aerial photographs of mountains",
    "Aerial photographs of the United States",
    "Aerial photographs of airports",
    "Aerial photographs of cities",
    "Oblique aerial photographs",
    "Aerial photographs of Europe",
]
SKIP_CAT = re.compile(r"map|diagram|drawing|painting|lithograph|postcard|engraving|model|logo|chart|scan|historic american|HAER|HABS|black and white|monochrome|\b19[0-8]\d\b|\b18\d\d\b", re.I)
HAER_RX = re.compile(r"\bHAER\b|\bHABS\b|Historic American|LCCN199|LC-DIG-h|Survey \(Library of Congress\)", re.I)
OK_MIME = {"image/jpeg", "image/png", "image/tiff"}

def log(f, msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True); f.write(line + "\n"); f.flush()

def api(params, tries=5):
    params = dict(params); params["format"] = "json"; params["formatversion"] = "2"
    url = API + "?" + urllib.parse.urlencode(params)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            time.sleep(2.0 * (k + 1))
            if k == tries - 1: raise
    return None

def licence_ok(meta):
    s = (meta.get("LicenseShortName", {}).get("value") or "").strip().lower()
    if not s: return False, s
    if s.startswith("public domain") or s.startswith("pd") or s in ("cc0", "cc0 1.0", "no restrictions", "pdm", "cc-zero", "cc zero"): return True, s
    if s.startswith("cc by") or s.startswith("cc-by"):
        tail = s.replace("cc by", "").replace("cc-by", "")
        if any(("-" + b) in tail or (" " + b) in tail for b in ("sa", "nc", "nd")): return False, s
        return True, s
    return False, s

def year_of(date):
    m = re.search(r"(1[89]\d\d|20\d\d)", date or "")
    return int(m.group(1)) if m else None

def strip_html(s):
    return re.sub(r"<[^>]+>", "", s or "")

def is_gray(path):
    """Mean channel spread on a 128 px thumbnail; under 6/255 is monochrome (film, scans)."""
    try:
        from PIL import Image
        im = Image.open(path); im.draft("RGB", (256, 256)); im = im.convert("RGB").resize((128, 128))
        b = im.tobytes(); n = len(b) // 3
        d = sum(abs(b[3*i] - b[3*i+1]) + abs(b[3*i+1] - b[3*i+2]) for i in range(n)) / max(n, 1)
        return d < 6.0
    except Exception:
        return False

def resolve_seed(cat, logf):
    """Return the category name as it exists on Commons, searching when the guess is wrong."""
    r = api(dict(action="query", prop="categoryinfo", titles="Category:" + cat))
    pages = (r or {}).get("query", {}).get("pages", [])
    if pages and not pages[0].get("missing") and (pages[0].get("categoryinfo") or {}).get("files", 0) + (pages[0].get("categoryinfo") or {}).get("subcats", 0) > 0:
        return cat
    r = api(dict(action="query", list="search", srnamespace="14", srsearch=cat, srlimit="5"))
    for hit in (r or {}).get("query", {}).get("search", []):
        t = hit["title"][9:]
        if "aerial" in t.lower():
            log(logf, "seed %r -> %r" % (cat, t)); return t
    log(logf, "seed %r not found on Commons - skipped" % cat)
    return None

def members(cat):
    files, subs, cont = [], [], {}
    while True:
        r = api(dict(action="query", list="categorymembers", cmtitle="Category:" + cat, cmtype="file|subcat", cmlimit="500", **cont))
        if not r: break
        for m in r.get("query", {}).get("categorymembers", []):
            t = m["title"]
            if t.startswith("File:"): files.append(t)
            elif t.startswith("Category:"): subs.append(t[9:])
        cont = r.get("continue", {})
        if not cont: break
        time.sleep(0.05)
    return files, subs

def infos(titles):
    r = api(dict(action="query", prop="imageinfo", titles="|".join(titles), iiprop="url|size|mime|sha1|extmetadata", iiurlwidth="4096", iiextmetadatafilter="LicenseShortName|Artist|Credit|DateTimeOriginal|ImageDescription|LicenseUrl|Attribution"))
    out = []
    for p in (r or {}).get("query", {}).get("pages", []):
        ii = (p.get("imageinfo") or [None])[0]
        if ii: out.append((p.get("pageid"), p["title"], ii))
    return out

def fetch(url, dst, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r, open(dst + ".part", "wb") as f:
                n = 0
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b); n += len(b)
            os.replace(dst + ".part", dst)
            return n
        except Exception:
            time.sleep(2.0 * (k + 1))
    return 0

def rejected(rec, min_year):
    y = year_of(rec.get("date"))
    if y is not None and y < min_year: return "year %d" % y
    if HAER_RX.search((rec.get("title") or "") + " " + (rec.get("credit") or "") + " " + (rec.get("category") or "")): return "HAER/HABS"
    return None

def clean(out, min_year, logf):
    man_path = os.path.join(out, "manifest.jsonl")
    recs = [json.loads(l) for l in open(man_path, encoding="utf-8")] if os.path.exists(man_path) else []
    keep, why = [], {}
    for r in recs:
        reason = rejected(r, min_year)
        if not reason and os.path.exists(r["file"]) and is_gray(r["file"]): reason = "grayscale"
        if not reason and not os.path.exists(r["file"]): reason = "missing"
        if reason:
            why[reason] = why.get(reason, 0) + 1
            try: os.remove(r["file"])
            except Exception: pass
        else:
            keep.append(r)
    with open(man_path, "w", encoding="utf-8") as f:
        for r in keep: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(logf, "clean: kept %d of %d (%.2f GB); removed %s" % (len(keep), len(recs), sum(r["bytes"] for r in keep) / 1e9, why or "nothing"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="E:/PhotoRef"); ap.add_argument("--max-gb", type=float, default=50.0)
    ap.add_argument("--depth", type=int, default=3); ap.add_argument("--min-width", type=int, default=1400)
    ap.add_argument("--min-year", type=int, default=1995)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--seeds", default="")
    ap.add_argument("--clean", action="store_true", help="apply the domain filters to what is on disk and exit")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "fetch.log"), "a", encoding="utf-8")
    if a.clean:
        clean(a.out, a.min_year, logf); return
    man_path = os.path.join(a.out, "manifest.jsonl")
    done, total = set(), 0
    if os.path.exists(man_path):
        for line in open(man_path, encoding="utf-8"):
            try:
                r = json.loads(line); done.add(r["pageid"]); total += r["bytes"]
            except Exception: pass
    man = open(man_path, "a", encoding="utf-8")
    cap = a.max_gb * 1e9
    seeds = [s.strip() for s in a.seeds.split(";") if s.strip()] or SEEDS
    log(logf, "start: %d already fetched (%.1f GB), cap %.0f GB, %d seeds, depth %d, min year %d" % (len(done), total / 1e9, a.max_gb, len(seeds), a.depth, a.min_year))
    seen_cat, queue = set(), []
    for sd in seeds:
        r = resolve_seed(sd, logf)
        if r and r not in [q[0] for q in queue]: queue.append((r, 0))
    nok = nskip = ngray = 0
    pool = ThreadPoolExecutor(max_workers=a.workers)
    while queue and total < cap:
        cat, d = queue.pop(0)
        if cat in seen_cat: continue
        seen_cat.add(cat)
        try: files, subs = members(cat)
        except Exception as e:
            log(logf, "category %s: %s" % (cat, e)); continue
        if d < a.depth:
            for s in subs:
                if not SKIP_CAT.search(s) and s not in seen_cat: queue.append((s, d + 1))
        if not files: continue
        catdir = os.path.join(a.out, "commons", re.sub(r"[^A-Za-z0-9._-]+", "_", cat)[:80])
        pending, got = {}, 0
        def drain(block):
            nonlocal total, got, nok, ngray
            for fu in [f for f in list(pending) if block or f.done()]:
                rec = pending.pop(fu); n = fu.result()
                if n <= 0: continue
                if is_gray(rec["file"]):
                    ngray += 1
                    try: os.remove(rec["file"])
                    except Exception: pass
                    continue
                rec["bytes"] = n; total += n; got += 1; nok += 1; done.add(rec["pageid"])
                man.write(json.dumps(rec, ensure_ascii=False) + "\n"); man.flush()
        for i in range(0, len(files), 50):
            if total >= cap: break
            try: batch = infos(files[i:i + 50])
            except Exception as e:
                log(logf, "imageinfo failed: %s" % e); continue
            time.sleep(0.05)
            submitted = 0
            for pageid, title, ii in batch:
                if pageid in done: continue
                meta = ii.get("extmetadata", {}) or {}
                ok, lic = licence_ok(meta)
                if not ok or ii.get("mime") not in OK_MIME or (ii.get("width") or 0) < a.min_width:
                    nskip += 1; continue
                use_orig = ii.get("mime") == "image/jpeg" and (ii.get("size") or 0) <= 15_000_000
                url = ii["url"] if use_orig else ii.get("thumburl") or ii["url"]
                name = re.sub(r"[^A-Za-z0-9._-]+", "_", title[5:])[:120]
                if not use_orig and not name.lower().endswith((".jpg", ".jpeg")): name += ".jpg"
                dst = os.path.join(catdir, name)
                rec = dict(pageid=pageid, title=title, category=cat, url=url, source="https://commons.wikimedia.org/wiki/" + urllib.parse.quote(title),
                           licence=lic, licence_url=(meta.get("LicenseUrl", {}).get("value") or ""),
                           author=strip_html(meta.get("Artist", {}).get("value"))[:200], credit=strip_html(meta.get("Credit", {}).get("value"))[:200],
                           attribution=strip_html(meta.get("Attribution", {}).get("value"))[:200], date=(meta.get("DateTimeOriginal", {}).get("value") or "")[:40],
                           width=ii.get("width"), height=ii.get("height"), mime=ii.get("mime"), sha1=ii.get("sha1"), file=dst, rendition="original" if use_orig else "4096px")
                if rejected(rec, a.min_year): nskip += 1; continue
                os.makedirs(catdir, exist_ok=True)
                pending[pool.submit(fetch, url, dst)] = rec; submitted += 1
            drain(False)
            while len(pending) > a.workers * 4: time.sleep(0.5); drain(False)
            log(logf, "%-48s batch %3d/%-3d submitted %2d | total %5d files %.2f GB | gray dropped %d | queue %d" % (cat[:48], i // 50 + 1, (len(files) + 49) // 50, submitted, len(done), total / 1e9, ngray, len(queue)))
        drain(True)
        log(logf, "%-48s done: fetched %d of %d files" % (cat[:48], got, len(files)))
    pool.shutdown(wait=False, cancel_futures=True)
    log(logf, "done: %d files on disk, %.1f GB, %d skipped by licence/size/date/HAER, %d grayscale dropped, %d categories" % (len(done), total / 1e9, nskip, ngray, len(seen_cat)))

if __name__ == "__main__":
    main()
