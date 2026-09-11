//! photoref-fetch - licence-clean aerial photo reference set from Wikimedia Commons.
//!
//! Walks aerial-photography categories through the MediaWiki API (breadth-first, depth-limited),
//! keeps only public domain / CC0 / CC BY files (no SA, NC, ND), rejects anything dated before
//! --min-year, HAER/HABS survey scans, and grayscale images (checked on arrival), downloads the
//! original up to 40 MB or a 2560 px rendition, and appends one JSON line per file to
//! manifest.jsonl (url, licence, author, credit, source, dims, bytes) for attribution.
//! Metadata calls and downloads run concurrently; one writer owns the manifest; resumable;
//! byte-capped. Not a scraper: the official API, a real User-Agent, bounded concurrency.
//!
//!   photoref-fetch --out E:/PhotoRef --max-gb 50 --depth 3
//!
//! Copyright (C) 2026 MotionVectors contributors. SPDX-License-Identifier: GPL-3.0-or-later

use anyhow::{anyhow, Result};
use futures::stream::{self, StreamExt};
use regex::Regex;
use serde_json::{json, Value};
use std::collections::{HashSet, VecDeque};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinSet;

const API: &str = "https://commons.wikimedia.org/w/api.php";
const UA: &str = "MotionVectors-PhotoRef/0.3 (X-Plane neural rendering research; https://github.com/Vihaan2012-cmyk/Motion-Vectors-for-Xplane-12-V2)";
const SEEDS: &[&str] = &[
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
];

struct Args {
    out: PathBuf,
    max_bytes: u64,
    depth: u32,
    min_year: i32,
    min_width: u64,
    workers: usize,
    meta: usize,
    seeds: Vec<String>,
}

fn parse_args() -> Args {
    let mut a = Args { out: PathBuf::from("E:/PhotoRef"), max_bytes: 50_000_000_000, depth: 3, min_year: 1995, min_width: 1400, workers: 12, meta: 6, seeds: vec![] };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i + 1 < argv.len() {
        let v = argv[i + 1].clone();
        match argv[i].as_str() {
            "--out" => a.out = PathBuf::from(v),
            "--max-gb" => a.max_bytes = (v.parse::<f64>().unwrap_or(50.0) * 1e9) as u64,
            "--depth" => a.depth = v.parse().unwrap_or(3),
            "--min-year" => a.min_year = v.parse().unwrap_or(1995),
            "--min-width" => a.min_width = v.parse().unwrap_or(1400),
            "--workers" => a.workers = v.parse().unwrap_or(12),
            "--meta" => a.meta = v.parse().unwrap_or(6),
            "--seeds" => a.seeds = v.split(';').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
            _ => {}
        }
        i += 2;
    }
    if a.seeds.is_empty() { a.seeds = SEEDS.iter().map(|s| s.to_string()).collect(); }
    a
}

struct Log { file: Mutex<std::fs::File> }
impl Log {
    fn say(&self, msg: &str) {
        let line = format!("{} {}", chrono::Local::now().format("%H:%M:%S"), msg);
        println!("{line}");
        if let Ok(mut f) = self.file.lock() { let _ = writeln!(f, "{line}"); let _ = f.flush(); }
    }
}

async fn api(client: &reqwest::Client, params: Vec<(String, String)>) -> Result<Value> {
    let mut q = params;
    q.push(("format".into(), "json".into()));
    q.push(("formatversion".into(), "2".into()));
    let mut last = anyhow!("no attempt");
    for k in 0..5u64 {
        match client.get(API).query(&q).send().await {
            Ok(r) if r.status().is_success() => match r.json::<Value>().await {
                Ok(v) => return Ok(v),
                Err(e) => last = e.into(),
            },
            Ok(r) => last = anyhow!("HTTP {}", r.status()),
            Err(e) => last = e.into(),
        }
        tokio::time::sleep(Duration::from_secs(2 * (k + 1))).await;
    }
    Err(last)
}

fn p(k: &str, v: impl Into<String>) -> (String, String) { (k.to_string(), v.into()) }

async fn members(client: &reqwest::Client, cat: &str) -> Result<(Vec<String>, Vec<String>)> {
    let (mut files, mut subs) = (Vec::new(), Vec::new());
    let mut cont: Vec<(String, String)> = Vec::new();
    loop {
        let mut params = vec![p("action", "query"), p("list", "categorymembers"), p("cmtitle", format!("Category:{cat}")), p("cmtype", "file|subcat"), p("cmlimit", "500")];
        params.extend(cont.iter().cloned());
        let v = api(client, params).await?;
        if let Some(arr) = v["query"]["categorymembers"].as_array() {
            for m in arr {
                let t = m["title"].as_str().unwrap_or("");
                if let Some(f) = t.strip_prefix("File:") { files.push(format!("File:{f}")); }
                else if let Some(c) = t.strip_prefix("Category:") { subs.push(c.to_string()); }
            }
        }
        cont.clear();
        if let Some(c) = v["continue"].as_object() {
            for (k, val) in c { if let Some(s) = val.as_str() { cont.push((k.clone(), s.to_string())); } }
        }
        if cont.is_empty() { break; }
    }
    Ok((files, subs))
}

async fn infos(client: reqwest::Client, titles: Vec<String>) -> Result<Vec<(i64, String, Value)>> {
    let v = api(&client, vec![p("action", "query"), p("prop", "imageinfo"), p("titles", titles.join("|")), p("iiprop", "url|size|mime|sha1|extmetadata"), p("iiextmetadatafilter", "LicenseShortName|Artist|Credit|DateTimeOriginal|LicenseUrl|Attribution")]).await?;
    let mut out = Vec::new();
    if let Some(pages) = v["query"]["pages"].as_array() {
        for page in pages {
            if let Some(ii) = page["imageinfo"].as_array().and_then(|a| a.first()) {
                out.push((page["pageid"].as_i64().unwrap_or(0), page["title"].as_str().unwrap_or("").to_string(), ii.clone()));
            }
        }
    }
    Ok(out)
}

async fn resolve_seed(client: &reqwest::Client, cat: &str, log: &Log) -> Option<String> {
    if let Ok(v) = api(client, vec![p("action", "query"), p("prop", "categoryinfo"), p("titles", format!("Category:{cat}"))]).await {
        if let Some(pg) = v["query"]["pages"].as_array().and_then(|a| a.first()) {
            let n = pg["categoryinfo"]["files"].as_u64().unwrap_or(0) + pg["categoryinfo"]["subcats"].as_u64().unwrap_or(0);
            if !pg["missing"].as_bool().unwrap_or(false) && n > 0 { return Some(cat.to_string()); }
        }
    }
    if let Ok(v) = api(client, vec![p("action", "query"), p("list", "search"), p("srnamespace", "14"), p("srsearch", cat), p("srlimit", "5")]).await {
        if let Some(hits) = v["query"]["search"].as_array() {
            for h in hits {
                if let Some(t) = h["title"].as_str().and_then(|t| t.strip_prefix("Category:")) {
                    if t.to_lowercase().contains("aerial") { log.say(&format!("seed {cat:?} -> {t:?}")); return Some(t.to_string()); }
                }
            }
        }
    }
    log.say(&format!("seed {cat:?} not found on Commons - skipped"));
    None
}

fn licence_ok(short: &str) -> bool {
    let s = short.trim().to_lowercase();
    if s.is_empty() { return false; }
    if s.starts_with("public domain") || s.starts_with("pd") || ["cc0", "cc0 1.0", "no restrictions", "pdm", "cc-zero", "cc zero"].contains(&s.as_str()) { return true; }
    if s.starts_with("cc by") || s.starts_with("cc-by") {
        let tail = s.replacen("cc by", "", 1).replacen("cc-by", "", 1);
        for b in ["sa", "nc", "nd"] { if tail.contains(&format!("-{b}")) || tail.contains(&format!(" {b}")) { return false; } }
        return true;
    }
    false
}

fn is_gray(path: &Path) -> bool {
    let img = match image::ImageReader::open(path).and_then(|r| r.with_guessed_format()) { Ok(r) => match r.decode() { Ok(i) => i, Err(_) => return false }, Err(_) => return false };
    let t = img.thumbnail(128, 128).to_rgb8();
    let (mut s, mut n) = (0u64, 0u64);
    for px in t.pixels() {
        let [r, g, b] = px.0;
        s += (r as i32 - g as i32).unsigned_abs() as u64 + (g as i32 - b as i32).unsigned_abs() as u64;
        n += 1;
    }
    (s as f64 / n.max(1) as f64) < 6.0
}

async fn fetch(client: reqwest::Client, url: String, dst: PathBuf) -> Result<u64> {
    let part = PathBuf::from(format!("{}.part", dst.display()));
    let mut last = anyhow!("no attempt");
    for k in 0..4u64 {
        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                let mut f = tokio::fs::File::create(&part).await?;
                let mut n = 0u64;
                let mut st = resp.bytes_stream();
                let mut ok = true;
                while let Some(chunk) = st.next().await {
                    match chunk {
                        Ok(c) => { tokio::io::AsyncWriteExt::write_all(&mut f, &c).await?; n += c.len() as u64; }
                        Err(e) => { last = e.into(); ok = false; break; }
                    }
                }
                drop(f);
                if ok { tokio::fs::rename(&part, &dst).await?; return Ok(n); }
            }
            Ok(r) => last = anyhow!("HTTP {}", r.status()),
            Err(e) => last = e.into(),
        }
        tokio::time::sleep(Duration::from_secs(2 * (k + 1))).await;
    }
    let _ = tokio::fs::remove_file(&part).await;
    Err(last)
}

#[tokio::main]
async fn main() -> Result<()> {
    let a = parse_args();
    std::fs::create_dir_all(&a.out)?;
    let log = Arc::new(Log { file: Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(a.out.join("fetch.log"))?) });
    let man_path = a.out.join("manifest.jsonl");
    let mut done: HashSet<i64> = HashSet::new();
    let mut total: u64 = 0;
    if man_path.exists() {
        for line in std::io::BufReader::new(std::fs::File::open(&man_path)?).lines().map_while(Result::ok) {
            if let Ok(v) = serde_json::from_str::<Value>(&line) {
                if let Some(id) = v["pageid"].as_i64() { if done.insert(id) { total += v["bytes"].as_u64().unwrap_or(0); } }
            }
        }
    }
    let done = Arc::new(Mutex::new(done));
    let total = Arc::new(AtomicU64::new(total));
    let ngray = Arc::new(AtomicU64::new(0));
    let cap = a.max_bytes;
    let year_rx = Regex::new(r"(1[89]\d\d|20\d\d)")?;
    let haer_rx = Regex::new(r"(?i)\bHAER\b|\bHABS\b|Historic American|LCCN199|LC-DIG-h|Survey \(Library of Congress\)")?;
    let skip_cat = Regex::new(r"(?i)map|diagram|drawing|painting|lithograph|postcard|engraving|model|logo|chart|scan|historic american|HAER|HABS|black and white|monochrome|\b19[0-8]\d\b|\b18\d\d\b")?;
    let html_rx = Regex::new(r"<[^>]+>")?;
    let name_rx = Regex::new(r"[^A-Za-z0-9._-]+")?;
    let client = reqwest::Client::builder().user_agent(UA).timeout(Duration::from_secs(180)).build()?;

    // one writer owns the manifest
    let (tx, mut rx) = mpsc::unbounded_channel::<Value>();
    let writer_path = man_path.clone();
    let writer = tokio::spawn(async move {
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&writer_path).expect("manifest");
        while let Some(v) = rx.recv().await { let _ = writeln!(f, "{}", v); let _ = f.flush(); }
    });

    log.say(&format!("start: {} already fetched ({:.2} GB), cap {:.0} GB, {} seeds, depth {}, min year {}, {} downloads / {} metadata in flight",
        done.lock().unwrap().len(), total.load(Ordering::Relaxed) as f64 / 1e9, cap as f64 / 1e9, a.seeds.len(), a.depth, a.min_year, a.workers, a.meta));

    let mut queue: VecDeque<(String, u32)> = VecDeque::new();
    for s in &a.seeds { if let Some(r) = resolve_seed(&client, s, &log).await { if !queue.iter().any(|q| q.0 == r) { queue.push_back((r, 0)); } } }
    let mut seen: HashSet<String> = HashSet::new();
    let sem = Arc::new(Semaphore::new(a.workers));
    let mut nskip: u64 = 0;

    while let Some((cat, d)) = queue.pop_front() {
        if total.load(Ordering::Relaxed) >= cap { break; }
        if !seen.insert(cat.clone()) { continue; }
        let (files, subs) = match members(&client, &cat).await { Ok(x) => x, Err(e) => { log.say(&format!("category {cat}: {e}")); continue; } };
        if d < a.depth { for s in subs { if !skip_cat.is_match(&s) && !seen.contains(&s) { queue.push_back((s, d + 1)); } } }
        if files.is_empty() { continue; }
        let catdir = a.out.join("commons").join(name_rx.replace_all(&cat, "_").chars().take(80).collect::<String>());
        let nbatches = (files.len() + 49) / 50;
        let chunks: Vec<Vec<String>> = files.chunks(50).map(|c| c.to_vec()).collect();
        let mut batches = stream::iter(chunks.into_iter().enumerate()).map(|(bi, c)| { let cl = client.clone(); async move { (bi, infos(cl, c).await) } }).buffer_unordered(a.meta);
        let mut tasks: JoinSet<()> = JoinSet::new();
        let mut got_cat = 0u64;
        while let Some((bi, res)) = batches.next().await {
            if total.load(Ordering::Relaxed) >= cap { break; }
            let batch = match res { Ok(b) => b, Err(e) => { log.say(&format!("imageinfo failed: {e}")); continue; } };
            let mut submitted = 0u32;
            for (pageid, title, ii) in batch {
                if done.lock().unwrap().contains(&pageid) { continue; }
                let meta = &ii["extmetadata"];
                let lic = meta["LicenseShortName"]["value"].as_str().unwrap_or("").to_string();
                let width = ii["width"].as_u64().unwrap_or(0);
                let mime = ii["mime"].as_str().unwrap_or("").to_string();
                if !licence_ok(&lic) || !["image/jpeg", "image/png", "image/tiff"].contains(&mime.as_str()) || width < a.min_width { nskip += 1; continue; }
                let date = meta["DateTimeOriginal"]["value"].as_str().unwrap_or("").chars().take(40).collect::<String>();
                if let Some(m) = year_rx.captures(&date) { if m[1].parse::<i32>().unwrap_or(9999) < a.min_year { nskip += 1; continue; } }
                let credit = html_rx.replace_all(meta["Credit"]["value"].as_str().unwrap_or(""), "").chars().take(200).collect::<String>();
                if haer_rx.is_match(&format!("{title} {credit} {cat}")) { nskip += 1; continue; }
                // JPEG originals only, straight off storage. A non-JPEG original would need a
                // 2560 px rendition from the thumbnail backend, which renders on demand, takes
                // seconds per file and is rate-limited per client - it was the whole bottleneck
                // (measured 2026-09-11: originals at 15 MB/s, the pipeline at 9 files/min). Those
                // files are also the wrong domain: PNG satellite composites and TIFF scans.
                let size = ii["size"].as_u64().unwrap_or(0);
                if mime != "image/jpeg" || size > 40_000_000 { nskip += 1; continue; }
                let use_orig = true;
                let url = ii["url"].as_str().unwrap_or("").to_string();
                if url.is_empty() { continue; }
                let name = name_rx.replace_all(&title[5..], "_").chars().take(120).collect::<String>();
                let dst = catdir.join(&name);
                let rec = json!({
                    "pageid": pageid, "title": title, "category": cat, "url": url,
                    "source": format!("https://commons.wikimedia.org/wiki/{}", title.replace(' ', "_")),
                    "licence": lic.to_lowercase(), "licence_url": meta["LicenseUrl"]["value"].as_str().unwrap_or(""),
                    "author": html_rx.replace_all(meta["Artist"]["value"].as_str().unwrap_or(""), "").chars().take(200).collect::<String>(),
                    "credit": credit, "attribution": html_rx.replace_all(meta["Attribution"]["value"].as_str().unwrap_or(""), "").chars().take(200).collect::<String>(),
                    "date": date, "width": width, "height": ii["height"].as_u64().unwrap_or(0), "mime": mime, "sha1": ii["sha1"].as_str().unwrap_or(""),
                    "file": dst.to_string_lossy(), "rendition": if use_orig { "original" } else { "2560px" }
                });
                done.lock().unwrap().insert(pageid);
                std::fs::create_dir_all(&catdir)?;
                let (cl, sem, tx, total, ngray, url2) = (client.clone(), sem.clone(), tx.clone(), total.clone(), ngray.clone(), url.clone());
                tasks.spawn(async move {
                    let _permit = sem.acquire_owned().await;
                    match fetch(cl, url2, dst.clone()).await {
                        Ok(n) => {
                            let gray = tokio::task::spawn_blocking(move || is_gray(&dst)).await.unwrap_or(false);
                            if gray { ngray.fetch_add(1, Ordering::Relaxed); let _ = std::fs::remove_file(rec["file"].as_str().unwrap_or("")); }
                            else { let mut r = rec; r["bytes"] = json!(n); total.fetch_add(n, Ordering::Relaxed); let _ = tx.send(r); }
                        }
                        Err(_) => {}
                    }
                });
                submitted += 1; got_cat += 1;
            }
            // keep the download backlog bounded so the byte cap is respected
            while tasks.len() > a.workers * 4 { tasks.join_next().await; }
            log.say(&format!("{:<48} batch {:>3}/{:<3} submitted {:>2} | total {:>5} files {:.2} GB | gray dropped {} | queue {}",
                cat.chars().take(48).collect::<String>(), bi + 1, nbatches, submitted, done.lock().unwrap().len(), total.load(Ordering::Relaxed) as f64 / 1e9, ngray.load(Ordering::Relaxed), queue.len()));
        }
        while tasks.join_next().await.is_some() {}
        log.say(&format!("{:<48} done: submitted {} of {} files", cat.chars().take(48).collect::<String>(), got_cat, files.len()));
    }
    drop(tx);
    let _ = writer.await;
    log.say(&format!("done: {} files, {:.2} GB, {} skipped by licence/size/date/HAER, {} grayscale dropped, {} categories",
        done.lock().unwrap().len(), total.load(Ordering::Relaxed) as f64 / 1e9, nskip, ngray.load(Ordering::Relaxed), seen.len()));
    Ok(())
}
