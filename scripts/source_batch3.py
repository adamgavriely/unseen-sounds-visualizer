"""Batch-3 sourcing: ~150+ diverse unseen/mixed candidates for the benchmark.

Three sources, per-source quotas so no single origin dominates:
  yt   — yt-dlp section-grabs (~28 s) from top search hits across ~70 distinct
         scene queries (one clip per video, --download-archive prevents repeats)
  wm   — Wikimedia Commons API video search across varied terms (CC), whole
         file if small, middle 28 s cut locally
  ia   — Internet Archive color travelogues/home-movies, mp4 derivative
         segment-grab over HTTP (works on IA mp4s)

Every candidate passes the quality screen learned from Adam's BAD tags:
video+audio streams, >=15 s, mean_volume > -50 dB, not a still image, not B&W.
Survivors land in data/input/benchmark/unsorted/ as b3_<theme>.mp4 and are
logged with source+license in benchmark/sources_batch3.json.

Usage: python scripts/source_batch3.py [yt|wm|ia|all]
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNSORTED = ROOT / "data" / "input" / "benchmark" / "unsorted"
STAGE = ROOT / "data" / "work" / "batch3_staging"
LOG = ROOT / "benchmark" / "sources_batch3.json"
ARCHIVE = STAGE / "yt_archive.txt"
HDRS = {"User-Agent": "MscFinalProject/0.1 (academic research; benchmark curation)"}
CLIP_SEC = 28

# ---------------------------------------------------------------- quality screen
def _probe(f: Path):
    try:
        o = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration:stream=codec_type", "-of", "json", str(f)],
                           capture_output=True, text=True, timeout=60).stdout
        d = json.loads(o or "{}")
        dur = float(d.get("format", {}).get("duration", 0) or 0)
        types = [s.get("codec_type") for s in d.get("streams", [])]
        return dur, "video" in types, "audio" in types
    except Exception:
        return 0, False, False


def _meanvol(f: Path):
    o = subprocess.run(["ffmpeg", "-i", str(f), "-af", "volumedetect", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=120).stderr
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", o)
    return float(m.group(1)) if m else -99.0


def _frame(f: Path, t: float, td: str, tag: str):
    from PIL import Image
    fp = Path(td) / f"{tag}.jpg"
    subprocess.run(["ffmpeg", "-y", "-ss", f"{t:.1f}", "-i", str(f), "-frames:v", "1",
                    "-vf", "scale=160:-1", str(fp)], capture_output=True, timeout=60)
    return Image.open(fp) if fp.exists() else None


def screen(f: Path) -> str | None:
    """Return a rejection reason, or None if the clip is good."""
    from PIL import ImageChops
    dur, hasv, hasa = _probe(f)
    if not hasv:
        return "no video"
    if not hasa:
        return "no audio"
    if dur < 15:
        return f"short {dur:.0f}s"
    if _meanvol(f) < -50:
        return "near-silent"
    with tempfile.TemporaryDirectory() as td:
        a, b = _frame(f, dur * 0.3, td, "a"), _frame(f, dur * 0.7, td, "b")
        if a and b and a.size == b.size:
            if statistics.mean(ImageChops.difference(
                    a.convert("L"), b.convert("L")).getdata()) < 2.0:
                return "still image"
        if a and statistics.mean(a.convert("HSV").split()[1].getdata()) < 14:
            return "black&white"
    return None


# ---------------------------------------------------------------- helpers
def _log(entry: dict):
    log = json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else []
    log.append(entry)
    LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


def _accept(tmp: Path, name: str, meta: dict) -> bool:
    why = screen(tmp)
    if why:
        print(f"    reject {name}: {why}")
        tmp.unlink(missing_ok=True)
        return False
    dest = UNSORTED / f"{name}.mp4"
    if dest.exists():
        tmp.unlink(missing_ok=True)
        return False
    if tmp.suffix != ".mp4":
        subprocess.run(["ffmpeg", "-y", "-i", str(tmp), "-c:v", "libx264", "-preset",
                        "veryfast", "-crf", "26", "-c:a", "aac", str(dest)],
                       capture_output=True, timeout=300)
        tmp.unlink(missing_ok=True)
    else:
        tmp.replace(dest)
    if not dest.exists():
        return False
    _log({"file": dest.name, **meta})
    print(f"    OK  {dest.name}")
    return True


def _cut_middle(src: Path, out: Path):
    dur, _, _ = _probe(src)
    start = max(0.0, dur / 2 - CLIP_SEC / 2)
    subprocess.run(["ffmpeg", "-y", "-ss", f"{start:.1f}", "-i", str(src),
                    "-t", str(CLIP_SEC), "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "26", "-c:a", "aac", str(out)],
                   capture_output=True, timeout=300)


# ---------------------------------------------------------------- source: YouTube
YT_QUERIES = [
    # street life / walks — many countries
    ("walk_lagos", "lagos nigeria street walk market sounds"),
    ("walk_hanoi", "hanoi old quarter walking tour motorbikes"),
    ("walk_cairo", "cairo street walk khan el khalili"),
    ("walk_mexicocity", "mexico city centro walking tour street"),
    ("walk_mumbai_rain", "mumbai monsoon rain street walk"),
    ("walk_seoul_night", "seoul night market street walk"),
    ("walk_naples", "naples italy narrow streets walking scooters"),
    ("walk_havana", "havana cuba street walk old cars"),
    ("walk_athens", "athens plaka walking tour street musicians"),
    ("walk_saopaulo", "sao paulo downtown street walk"),
    # markets / bazaars
    ("market_marrakech", "marrakech souk walking tour"),
    ("market_taipei_night", "taipei night market food walk"),
    ("market_istanbul_spice", "istanbul spice bazaar walk"),
    ("market_fish_tokyo", "tsukiji fish market walk"),
    ("market_floating", "bangkok floating market boat ride"),
    ("market_delhi", "delhi chandni chowk market walk"),
    # water / harbors / boats
    ("harbor_fishing", "fishing harbor boats seagulls morning"),
    ("ferry_ride", "ferry ride deck sea sound"),
    ("kayak_river", "kayaking river rapids sounds"),
    ("beach_boardwalk", "beach boardwalk summer crowd walk"),
    ("canal_amsterdam", "amsterdam canal boat ride"),
    ("venice_gondola", "venice gondola ride sounds ambient"),
    # nature / rural
    ("farm_sheep", "sheep farm ambient sounds new zealand"),
    ("jungle_hike", "jungle hike borneo sounds birds"),
    ("waterfall_hike", "waterfall hike trail sound"),
    ("forest_rain", "rain forest walk sound"),
    ("desert_camel", "desert camel ride wind"),
    ("savanna_safari", "safari game drive sounds animals"),
    ("mountain_stream", "mountain stream hiking sounds"),
    ("rice_field", "rice field village sounds vietnam"),
    # weather
    ("storm_street", "thunderstorm city street heavy rain"),
    ("snowstorm_walk", "snowstorm blizzard walk city"),
    ("wind_coast", "strong wind coastal cliff walk"),
    ("hail_storm", "hailstorm on street video sound"),
    # transport (non-train-heavy)
    ("airport_terminal", "airport terminal walk announcements"),
    ("tram_lisbon", "lisbon tram 28 ride sounds"),
    ("tuktuk_ride", "tuk tuk ride bangkok traffic"),
    ("cablecar_ride", "cable car ride sounds mountain"),
    ("motorbike_traffic", "vietnam motorbike traffic crossing street"),
    ("bus_ride_city", "city bus ride ambient sounds"),
    # events / people
    ("carnival_parade", "carnival parade street drums brazil"),
    ("festival_lantern", "lantern festival crowd walk"),
    ("street_performer", "street performer crowd watching plaza"),
    ("stadium_entrance", "football stadium entrance crowd chants outside"),
    ("protest_march", "protest march chanting street"),
    ("wedding_street", "street wedding procession music india"),
    ("church_square", "church square bells pigeons people"),
    ("playground_park", "playground park children playing sounds"),
    ("basketball_court", "outdoor basketball court pickup game sounds"),
    ("skatepark", "skatepark sounds skating ambient"),
    # work / industry
    ("construction_site", "construction site sounds excavator"),
    ("workshop_blacksmith", "blacksmith workshop hammering"),
    ("sawmill_wood", "sawmill woodworking sounds"),
    ("car_wash", "car wash from inside sounds"),
    ("barbershop", "barbershop ambient clippers sounds"),
    ("kitchen_restaurant", "restaurant kitchen sounds cooking service"),
    ("farm_tractor", "tractor plowing field sounds"),
    ("port_cranes", "container port cranes loading sounds"),
    # indoor / misc
    ("cafe_terrace", "cafe terrace street ambient people"),
    ("library_hall", "library hall quiet ambience pages"),
    ("arcade_games", "arcade game hall sounds walk"),
    ("bowling_alley", "bowling alley sounds strikes"),
    ("swimming_pool", "indoor swimming pool echo sounds"),
    ("zoo_walk", "zoo walking tour animal sounds"),
    ("aquarium_walk", "aquarium walk ambient"),
    ("night_crickets", "village night crickets dogs distant"),
    ("campfire_night", "campfire night sounds crackling talking"),
    ("fireworks_festival", "fireworks festival crowd watching"),
    ("marina_masts", "marina sailboat masts clinking wind"),
    ("food_court", "food court mall lunchtime ambience"),
]


def source_yt(target: int = 70):
    STAGE.mkdir(parents=True, exist_ok=True)
    got = 0
    for name, query in YT_QUERIES:
        if got >= target:
            break
        tmp = STAGE / f"yt_{name}.mp4"
        tmp.unlink(missing_ok=True)
        print(f"  [yt] {name}: {query}")
        r = subprocess.run(
            ["yt-dlp", f"ytsearch1:{query}",
             "--download-sections", f"*180-{180 + CLIP_SEC}",
             "-f", "mp4[height<=720]/best[height<=720]/best",
             "--no-playlist", "--force-keyframes-at-cuts",
             "--match-filter", "duration>240",
             "--download-archive", str(ARCHIVE),
             "--print-to-file", "%(webpage_url)s|%(title)s", str(STAGE / f"yt_{name}.meta"),
             "--no-warnings", "-o", str(tmp)],
            capture_output=True, text=True, timeout=420)
        if not tmp.exists():
            print(f"    skip ({(r.stderr or '')[:90].strip()})")
            continue
        meta_f = STAGE / f"yt_{name}.meta"
        url, title = ("", "")
        if meta_f.exists():
            parts = meta_f.read_text(encoding="utf-8").strip().split("|", 1)
            url = parts[0]
            title = parts[1] if len(parts) > 1 else ""
        if _accept(tmp, f"b3_{name}", {"source": "youtube", "url": url, "title": title,
                                       "license": "research use; not redistributed"}):
            got += 1
    print(f"[yt] accepted {got}")
    return got


# ---------------------------------------------------------------- source: Commons
WM_TERMS = ["walking tour city", "street market video", "harbor boats",
            "traffic street sound", "festival crowd", "rain street",
            "waterfall river", "construction site", "farm animals video",
            "airport", "playground", "beach waves people", "protest demonstration",
            "village street", "fountain square", "birds park city"]


def _wm_api(params: dict) -> dict:
    qs = urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(
        f"https://commons.wikimedia.org/w/api.php?{qs}", headers=HDRS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def source_wm(target: int = 45):
    STAGE.mkdir(parents=True, exist_ok=True)
    got = 0
    for term in WM_TERMS:
        if got >= target:
            break
        print(f"  [wm] search: {term}")
        try:
            data = _wm_api({"action": "query", "generator": "search",
                            "gsrsearch": f"{term} filetype:video", "gsrnamespace": "6",
                            "gsrlimit": "8", "prop": "videoinfo",
                            "viprop": "url|size|derivatives"})
        except Exception as e:
            print(f"    api error {type(e).__name__}")
            continue
        pages = (data.get("query") or {}).get("pages", {})
        per_term = 0
        for p in pages.values():
            if got >= target or per_term >= 3:
                break
            vi = (p.get("videoinfo") or [{}])[0]
            dur = float(vi.get("duration") or 0)
            if dur < 15:
                continue
            # smallest derivative >= 360p, else original if < 90 MB
            url, size = None, 0
            for d in sorted(vi.get("derivatives", []),
                            key=lambda d: int(d.get("height") or 0)):
                if int(d.get("height") or 0) >= 360:
                    url = d.get("src")
                    break
            if not url:
                if int(vi.get("size") or 0) > 90 * 1024 * 1024:
                    continue
                url = vi.get("url")
            title = p.get("title", "").replace("File:", "")
            slug = re.sub(r"[^a-z0-9]+", "_", title.lower())[:32].strip("_")
            tmp = STAGE / f"wm_{slug}.vid"
            try:
                req = urllib.request.Request(url, headers=HDRS)
                with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        if f.tell() > 120 * 1024 * 1024:
                            raise IOError("too big")
            except Exception as e:
                tmp.unlink(missing_ok=True)
                print(f"    dl fail {slug}: {type(e).__name__}")
                continue
            cut = STAGE / f"wm_{slug}.mp4"
            _cut_middle(tmp, cut)
            tmp.unlink(missing_ok=True)
            if cut.exists() and _accept(cut, f"b3_wm_{slug}",
                                        {"source": "wikimedia", "title": title,
                                         "url": f"https://commons.wikimedia.org/wiki/File:{urllib.parse.quote(title)}",
                                         "license": "CC (verify on Commons page)"}):
                got += 1
                per_term += 1
    print(f"[wm] accepted {got}")
    return got


# ---------------------------------------------------------------- source: IA
IA_QUERIES = [
    'travelogue color city', 'home movie color vacation sound',
    'street scenes color 16mm sound', 'documentary color village daily life',
    'promotional film color 1970s city', 'nature film color sound',
]


def source_ia(target: int = 40):
    STAGE.mkdir(parents=True, exist_ok=True)
    got = 0
    for q in IA_QUERIES:
        if got >= target:
            break
        print(f"  [ia] search: {q}")
        qs = urllib.parse.urlencode({
            "q": f'{q} AND mediatype:(movies)', "fl[]": "identifier",
            "rows": "12", "output": "json"})
        try:
            req = urllib.request.Request(
                f"https://archive.org/advancedsearch.php?{qs}", headers=HDRS)
            with urllib.request.urlopen(req, timeout=30) as r:
                docs = json.loads(r.read())["response"]["docs"]
        except Exception as e:
            print(f"    search error {type(e).__name__}")
            continue
        per_q = 0
        for doc in docs:
            if got >= target or per_q >= 3:
                break
            ident = doc["identifier"]
            try:
                req = urllib.request.Request(
                    f"https://archive.org/metadata/{ident}", headers=HDRS)
                with urllib.request.urlopen(req, timeout=30) as r:
                    meta = json.loads(r.read())
                mp4s = [f for f in meta.get("files", [])
                        if f.get("name", "").endswith(".mp4")]
                if not mp4s:
                    continue
                f0 = mp4s[0]
                length = float(f0.get("length") or 0)
                if length < 120:
                    continue
                url = f"https://archive.org/download/{ident}/{urllib.parse.quote(f0['name'])}"
            except Exception:
                continue
            # two spaced scenes max per film
            for si, frac in enumerate((0.3, 0.7)):
                if got >= target or per_q >= 2:
                    break
                start = length * frac
                slug = re.sub(r"[^a-z0-9]+", "_", ident.lower())[:28].strip("_")
                cut = STAGE / f"ia_{slug}_{si}.mp4"
                subprocess.run(["ffmpeg", "-y", "-ss", f"{start:.0f}", "-i", url,
                                "-t", str(CLIP_SEC), "-c:v", "libx264", "-preset",
                                "veryfast", "-crf", "26", "-c:a", "aac", str(cut)],
                               capture_output=True, timeout=300)
                if cut.exists() and cut.stat().st_size > 500_000 and _accept(
                        cut, f"b3_ia_{slug}_{si}",
                        {"source": "internet_archive", "item": ident, "url": url,
                         "license": "public domain / verify item page"}):
                    got += 1
                    per_q += 1
                else:
                    cut.unlink(missing_ok=True)
    print(f"[ia] accepted {got}")
    return got


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    UNSORTED.mkdir(parents=True, exist_ok=True)
    total = 0
    if which in ("yt", "all"):
        total += source_yt()
    if which in ("wm", "all"):
        total += source_wm()
    if which in ("ia", "all"):
        total += source_ia()
    print(f"\nbatch3 total accepted: {total}")
