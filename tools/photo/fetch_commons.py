#!/usr/bin/env python3
"""fetch_commons.py - build a licence-clean aerial photo reference set from Wikimedia Commons.

Walks aerial-photography categories through the official MediaWiki API (breadth-first,
depth-limited), keeps only files whose licence permits reuse without share-alike or
non-commercial strings (public domain, CC0, CC BY), records licence + author + source per
file for attribution, and downloads a 4096 px rendition when the original is huge. Not a
scraper: one API call at a time, a real User-Agent, resumable, and a byte cap.

    python tools/photo/fetch_commons.py --out E:/PhotoRef --max-gb 50
    python tools/photo/fetch_commons.py --out E:/PhotoRef --max-gb 50 --depth 2 --seeds "Aerial photographs of Los Angeles"

Output: <out>/commons/<category>/<file>, <out>/manifest.jsonl (one line per file: url, licence,
author, credit, category, dims, bytes, sha1), <out>/fetch.log.

Copyright (C) 2026 MotionVectors contributors. SPDX: GPL-3.0-or-later
"""
import argparse, hashlib, json, os, re, sys, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://commons.wikimedia.org/w/api.php"
UA = "MotionVectors-PhotoRef/0.1 (X-Plane neural rendering research; https://github.com/Vihaan2012-cmyk/Motion-Vectors-for-Xplane-12-V2)"
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
SKIP_CAT = re.compile(r"map|diagram|drawing|painting|lithograph|postcard|engraving|model|logo|chart|scan|\b19[0-4]\d\b|\b18\d\d\b", re.I)
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
        except Exception as e:
            time.sleep(2.0 * (k + 1))
            if k == tries - 1: raise
    return None

def licence_ok(meta):
    s = (meta.get("LicenseShortName", {}).get("value") or "").strip().lower()
    if not s: return False, s
    if s.startswith("public domain") or s.startswith("pd") or s in ("cc0", "cc0 1.0", "no restrictions", "pdm", "cc-zero", "cc zero"): return True, s
    if s.startswith("cc by") or s.startswith("cc-by"):
        bad = ("sa", "nc", "nd")
        tail = s.replace("cc by", "").replace("cc-by", "")
        if any(("-" + b) in tail or (" " + b) in tail for b in bad): return False, s
        return True, s
    return False, s

def members(cat):
    """(files, subcats) of a category, all pages."""
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
        time.sleep(0.2)
    return files, subs

def infos(titles):
    """imageinfo for up to 50 File: titles."""
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="E:/PhotoRef"); ap.add_argument("--max-gb", type=float, default=50.0)
    ap.add_argument("--depth", type=int, default=3); ap.add_argument("--min-width", type=int, default=1400)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--seeds", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "fetch.log"), "a", encoding="utf-8")
    man_path = os.path.join(a.out, "manifest.jsonl")
    done = set()
    if os.path.exists(man_path):
        for line in open(man_path, encoding="utf-8"):
            try: done.add(json.loads(line)["pageid"])
            except Exception: pass
    total = sum(json.loads(l)["bytes"] for l in open(man_path, encoding="utf-8")) if os.path.exists(man_path) else 0
    man = open(man_path, "a", encoding="utf-8")
    cap = a.max_gb * 1e9
    seeds = [s.strip() for s in a.seeds.split(";") if s.strip()] or SEEDS
    log(logf, "start: %d already fetched (%.1f GB), cap %.0f GB, %d seeds, depth %d" % (len(done), total / 1e9, a.max_gb, len(seeds), a.depth))
    seen_cat, queue = set(), [(s, 0) for s in seeds]
    nfiles = nok = nskip = 0
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
        jobs = []
        for i in range(0, len(files), 50):
            try: batch = infos(files[i:i + 50])
            except Exception as e:
                log(logf, "imageinfo failed: %s" % e); continue
            time.sleep(0.2)
            for pageid, title, ii in batch:
                nfiles += 1
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
                           author=re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value") or "")[:200],
                           credit=re.sub(r"<[^>]+>", "", meta.get("Credit", {}).get("value") or "")[:200],
                           attribution=re.sub(r"<[^>]+>", "", meta.get("Attribution", {}).get("value") or "")[:200],
                           date=(meta.get("DateTimeOriginal", {}).get("value") or "")[:40],
                           width=ii.get("width"), height=ii.get("height"), mime=ii.get("mime"), sha1=ii.get("sha1"), file=dst, rendition="original" if use_orig else "4096px")
                jobs.append((rec, url, dst))
        if not jobs:
            log(logf, "%-60s files %4d, nothing new (licence/size filtered)" % (cat[:60], len(files))); continue
        os.makedirs(catdir, exist_ok=True)
        futs = {pool.submit(fetch, url, dst): rec for rec, url, dst in jobs}
        got = 0
        for fu in as_completed(futs):
            rec = futs[fu]; n = fu.result()
            if n <= 0: continue
            rec["bytes"] = n; total += n; got += 1; nok += 1; done.add(rec["pageid"])
            man.write(json.dumps(rec, ensure_ascii=False) + "\n"); man.flush()
            if total >= cap: break
        log(logf, "%-60s files %4d -> fetched %3d | total %d files %.1f GB | queue %d" % (cat[:60], len(files), got, nok + len(done) - nok, total / 1e9, len(queue)))
    pool.shutdown(wait=False, cancel_futures=True)
    log(logf, "done: %d files on disk, %.1f GB, %d skipped by licence/size, %d categories" % (len(done), total / 1e9, nskip, len(seen_cat)))

if __name__ == "__main__":
    main()
