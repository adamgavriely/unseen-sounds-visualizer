"""AudioSet-Strong clips for Adam's tagging tool (docs/tagger_selection_rule.md point 2, final spec from Adam 2026-09-30:
AudioSet-Strong is the only source of new clips; 50 clips a001..a050; >= 3 labelled sound types; no BEATs keep rule;
up to 25 "striking" clips first).

Pool: AudioSet-Strong EVALUATION split minus every YouTube id in the fresh set's exclusion union
(audioset_fresh.exclusion: 415 builder exclusions, the 415's 500, repo scan, 'as_' stubs), minus the 500 ids drawn
for the fresh set (incl. missing), minus any eval id named anywhere under benchmark/, data/, docs/, tagger/ (text and
file names; label TSVs skipped). The fresh-set clips are never used.
Eligible: strong labels have >= 3 distinct depictable non-speech, non-music sound types (the BEATs suggest filter
applied to the ontology names, families grouped by labels.same_source).
Order: sorted eligible list, random.Random(20260930).shuffle. Pass 1 walks the STRIKING eligible clips (an event
whose name, family or ontology ancestor is on STRIKING) in that order until 25 are kept; pass 2 walks the remaining
eligible clips (not yet tried) in the same order until 50 are kept. A candidate is kept iff the labelled 10 s
downloads, re-encodes (<= 480p, H.264/AAC, faststart), is not a still image / slideshow and is not silent.
Clips are numbered in selection order. BEATs suggestions (tagger_beats_suggest.py) are computed for the kept clips
and recorded, never used to filter.

    python benchmark/gold/tagger_audioset.py --scratch <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config  # noqa: E402

config.use_scored()
for _k in ("FLEXSED_BAR", "FLEXSED_VETO", "PANNS_VETO"):
    setattr(config, _k, 0.0)
from src.labels import is_salient_nonspeech, canonical, is_music, is_descendant, same_source  # noqa: E402
from benchmark.gold.audioset_slice import load_labels  # noqa: E402
import benchmark.gold.audioset_fresh as F  # noqa: E402
from benchmark.gold.tagger_beats_suggest import sound_groups, NOT_TAGGED  # noqa: E402

SEED, N_KEEP, N_STRIKING, MIN_TYPES = 20260930, 50, 25, 3
OUT = _ROOT / "benchmark" / "gold" / "tagger_audioset_sources.json"
DEST = _ROOT / "data" / "work" / "tagger_new" / "audioset"
FRESH = _ROOT / "benchmark" / "gold" / "audioset_fresh.json"
BEATS_WORK = _ROOT / "data" / "work" / "tagger_beats"
TH = {"mean_db_gt": -45.0, "motion_ge": 2.0, "slideshow": "fraction of 0.5-s frame pairs with mean gray diff < 1.0 >= 0.9",
      "min_duration_s": 8.0, "strong_types_ge": MIN_TYPES}
# Adam's striking list mapped to AudioSet-Strong display names (strong-only names such as "Glass shatter" are not in
# the ontology tree, so exact names are listed too; the three "(siren)" vehicles sit under Emergency vehicle in the
# ontology, listed by name). Match: the exact name or an ontology ancestor on the list (NOT labels.canonical, which
# would add Beep -> Alarm, Bang -> Explosion, Rumble -> Thunder).
STRIKING = {"Explosion", "Gunshot, gunfire", "Fireworks", "Burst, pop", "Boom", "Siren", "Alarm",
            "Smoke detector, smoke alarm", "Fire alarm", "Car alarm", "Glass shatter", "Shatter", "Screaming",
            "Crying, sobbing", "Baby cry, infant cry", "Thunder", "Smash, crash", "Vehicle horn, car horn, honking",
            "Vehicle horn, car horn, honking, toot", "Air horn, truck horn", "Doorbell", "Telephone bell ringing",
            "Ringtone", "Bark", "Growling", "Breaking", "Crockery breaking and smashing",
            "Ambulance (siren)", "Police car (siren)", "Fire engine, fire truck (siren)"}
BOT = re.compile(r"sign in to confirm|not a bot|429|too many requests|rate.?limit", re.I)


def depictable_types(events):
    fams = set()
    for name, _s, _e in events:
        if not is_salient_nonspeech(name):
            continue
        fam = canonical(name)
        if NOT_TAGGED.match(fam) or is_music(fam) or is_descendant(fam, "Speech") or is_descendant(fam, "Music"):
            continue
        fams.add(fam)
    return sound_groups(fams, same_source)


def striking_hit(name):
    for x in sorted(STRIKING):
        if name == x or is_descendant(name, x):
            return x
    return None


def striking_events(events):
    out = []
    for name, s, e in sorted(events, key=lambda z: (z[1], z[0])):
        x = striking_hit(name)
        if x:
            out.append({"name": name, "onset": round(s, 3), "offset": round(e, 3), "matched": x})
    return out


def build_pool(clips):
    F.SCAN_DIRS = ("benchmark", "data", "docs", "tagger")
    F.TEXT_EXT = F.TEXT_EXT | {".js", ".tex", ".htm", ".xml", ".ts"}
    F.SKIP_FILES = F.SKIP_FILES | {OUT.name}
    ex, info = F.exclusion(clips)
    fr = json.loads(FRESH.read_text(encoding="utf-8"))
    fresh = {F.ytid(s) for s in fr["strata"]} | {F.ytid(c["id"]) for c in fr["clips"]} | {F.ytid(s) for s in fr["missing"]}
    eval_y = {F.ytid(s) for s in clips}
    info["fresh_500"] = len(fresh & eval_y)
    ex |= fresh
    info["excluded_eval_ytids_total"] = len(ex & eval_y)
    pool = sorted(s for s in clips if F.ytid(s) not in ex)
    info |= {"eval_segments": len(clips), "eval_ytids": len(eval_y), "pool_segments": len(pool)}
    return pool, info


def fetch(seg, raw_dir: Path):
    y, ms = seg.rsplit("_", 1)
    start = int(ms) / 1000.0
    dst = raw_dir / "raw.mp4"
    for p in raw_dir.glob("raw*"):
        p.unlink()
    cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "-f", "bv*[height<=480]+ba/b[height<=480]/b",
           "--download-sections", f"*{start:.1f}-{start + 10:.1f}", "--force-keyframes-at-cuts",
           "--merge-output-format", "mp4", "-o", str(dst), f"https://www.youtube.com/watch?v={y}"]
    try:
        r = subprocess.run(cmd, timeout=300, capture_output=True, text=True, errors="ignore")
        err = (r.stderr or "")
    except subprocess.TimeoutExpired:
        return None, "download timeout"
    if dst.exists():
        return dst, None
    if BOT.search(err):
        return None, "BLOCKED: " + err.strip().splitlines()[-1][:160]
    last = err.strip().splitlines()[-1][:160] if err.strip() else "no file"
    return None, "unavailable: " + last


def reencode(src: Path, dst: Path):
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", "10",
           "-vf", "scale=-2:'min(480,trunc(ih/2)*2)'", "-c:v", "libx264", "-preset", "medium", "-crf", "22",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, capture_output=True, timeout=300)
    return dst.exists()


def probe(f: Path):
    o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name",
                        "-of", "json", str(f)], capture_output=True, text=True).stdout
    d = json.loads(o or "{}")
    return float(d.get("format", {}).get("duration", 0) or 0), {s.get("codec_type"): s.get("codec_name") for s in d.get("streams", [])}


def mean_db(f: Path):
    o = subprocess.run(["ffmpeg", "-i", str(f), "-af", "volumedetect", "-f", "null", "-"],
                       capture_output=True, text=True, errors="ignore").stderr or ""
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", o)
    return float(m.group(1)) if m else -99.0


def frames(f: Path, fps=2):
    import numpy as np
    o = subprocess.run(["ffmpeg", "-v", "error", "-i", str(f), "-vf", f"fps={fps},scale=160:90,format=gray",
                        "-f", "rawvideo", "-"], capture_output=True).stdout
    a = np.frombuffer(o, dtype=np.uint8)
    n = a.size // (160 * 90)
    return a[: n * 160 * 90].reshape(n, 90, 160).astype(float)


def qc(f: Path):
    dur, st = probe(f)
    if "video" not in st:
        return "no video", {}
    if "audio" not in st:
        return "no audio", {}
    if dur < TH["min_duration_s"]:
        return f"short {dur:.1f}s", {}
    db = mean_db(f)
    fr = frames(f)
    if len(fr) < 4:
        return "unreadable frames", {}
    i, j = int(len(fr) * 0.3), int(len(fr) * 0.7)
    motion = float(abs(fr[i] - fr[j]).mean())
    pairs = [float(abs(fr[k + 1] - fr[k]).mean()) for k in range(len(fr) - 1)]
    static = sum(p < 1.0 for p in pairs) / len(pairs)
    m = {"duration": round(dur, 2), "mean_db": db, "motion": round(motion, 2), "static_frac": round(static, 2),
         "codecs": st}
    if db <= TH["mean_db_gt"]:
        return f"silent (mean {db} dB)", m
    if motion < TH["motion_ge"]:
        return f"still image (motion {motion:.2f})", m
    if static >= 0.9:
        return f"slideshow (static pairs {static:.2f})", m
    return None, m


def run_beats(folder: Path, out: Path, device: str):
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    cmd = [sys.executable, str(_ROOT / "benchmark" / "gold" / "tagger_beats_suggest.py"), "--videos", str(folder),
           "--out", str(out), "--device", device]
    r = subprocess.run(cmd, env=env, cwd=str(_ROOT), capture_output=True, text=True, errors="ignore")
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])
    return json.loads(out.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    scratch = Path(a.scratch)
    raw_dir, cand_dir = scratch / "raw", scratch / "cand"
    for d in (raw_dir, cand_dir):
        d.mkdir(parents=True, exist_ok=True)
    cache_p = scratch / "cache.json"      # seg -> {"reason": str|None, "qc": {...}, "cand": file}  (download/QC outcome)
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if cache_p.exists() else {}

    clips = load_labels()
    pool, info = build_pool(clips)
    info["unmapped_mids"] = len({n for ev in clips.values() for n, _s, _e in ev if n.startswith("/")})
    types = {s: depictable_types(clips[s]) for s in pool}
    eligible = sorted(s for s in pool if len(types[s]) >= MIN_TYPES)
    order = list(eligible)
    random.Random(SEED).shuffle(order)
    strike = {s: striking_events(clips[s]) for s in order}
    info["eligible_segments"] = len(eligible)
    info["eligible_striking"] = sum(bool(v) for v in strike.values())
    order_sha = hashlib.sha256("\n".join(order).encode()).hexdigest()
    print(f"[as] {info}", flush=True)

    tried, selected, blocked_run = [], [], 0      # tried: [{seg, pass, reason}] in walk order

    def attempt(seg, pas):
        nonlocal blocked_run
        c = cache.get(seg)
        if c is None:
            f, err = fetch(seg, raw_dir)
            if f is None:
                c = {"reason": err, "qc": {}}
                blocked_run = blocked_run + 1 if err.startswith("BLOCKED") else 0
                if blocked_run >= 5:
                    cache_p.write_text(json.dumps(cache, indent=1), encoding="utf-8")
                    sys.exit("[as] 5 blocked downloads in a row: stop (tooling block)")
            else:
                blocked_run = 0
                cand = cand_dir / f"c{len(cache) + 1:04d}.mp4"
                while cand.exists():
                    cand = cand.with_name(f"x{cand.name}")
                if not reencode(f, cand):
                    c = {"reason": "re-encode failed", "qc": {}}
                else:
                    reason, m = qc(cand)
                    c = {"reason": reason, "qc": m, "cand": cand.name}
                    if reason:
                        cand.unlink()
            cache[seg] = c
            cache_p.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        tried.append({"segment_id": seg, "pass": pas, "reason": c["reason"]})
        print(f"[as] {pas} #{len(tried)} {seg}: {c['reason'] or 'KEPT'} ({len(selected) + (c['reason'] is None)} kept)",
              flush=True)
        if c["reason"] is None:
            selected.append((seg, pas))

    for seg in order:                                   # pass 1: striking first
        if len(selected) >= N_STRIKING:
            break
        if strike[seg]:
            attempt(seg, "striking")
    done = {t["segment_id"] for t in tried}
    for seg in order:                                   # pass 2: fill from the remaining eligible, same order
        if len(selected) >= N_KEEP:
            break
        if seg not in done:
            attempt(seg, "fill")

    if DEST.exists():
        for p in DEST.glob("a*.mp4"):
            p.unlink()
    DEST.mkdir(parents=True, exist_ok=True)
    for n, (seg, _pas) in enumerate(selected, 1):
        shutil.copy2(cand_dir / cache[seg]["cand"], DEST / f"a{n:03d}.mp4")
    beats_out = scratch / "beats_final.json"
    if beats_out.exists():
        beats_out.unlink()
    beats = run_beats(DEST, beats_out, a.device)

    out_clips = []
    for n, (seg, pas) in enumerate(selected, 1):
        name = f"a{n:03d}"
        y, ms = seg.rsplit("_", 1)
        ev = [{"name": nm, "onset": round(s, 3), "offset": round(e, 3)} for nm, s, e in
              sorted(clips[seg], key=lambda x: (x[1], x[2], x[0]))]
        b = beats[f"{name}.mp4"]
        out_clips.append({"clip": name, "segment_id": seg, "youtube_id": y, "start": int(ms) / 1000.0,
                          "selected_in": pas, "striking": bool(strike[seg]), "striking_events": strike[seg],
                          "strong_events": ev, "strong_types": types[seg], "n_strong_types": len(types[seg]),
                          "n_sounds": b["n_sounds"], "beats_sounds": b["sounds"], "beats_suggestions": b["suggestions"],
                          "qc": cache[seg]["qc"]})
    skipped = [t for t in tried if t["reason"]]
    reasons = {}
    for s in skipped:
        k = re.sub(r"[:(].*", "", s["reason"]).strip()
        reasons[k] = reasons.get(k, 0) + 1
    doc = {"rule": ("docs/tagger_selection_rule.md point 2 with Adam's final spec (2026-09-30): AudioSet-Strong is the "
                    "only source of new clips; 50 clips a001..a050; eligible = strong labels with >= 3 distinct "
                    "depictable non-speech, non-music sound types (same_source grouping); striking clips first (an "
                    "event on the STRIKING list, by name, family or ontology ancestor) in seed order until 25 kept, "
                    "then the remaining eligible clips in the same order until 50; kept iff downloadable, not a still "
                    "image / slideshow, not silent. No BEATs keep rule (BEATs suggestions recorded for the tool). "
                    "Numbered in selection order. Strong labels are private: only for checking Adam's tagging."),
           "seed": SEED, "n_keep_target": N_KEEP, "n_striking_target": N_STRIKING, "striking_list": sorted(STRIKING),
           "pool": info, "order_sha256": order_sha, "thresholds": TH, "builder": "benchmark/gold/tagger_audioset.py",
           "tried": len(tried), "kept": len(out_clips), "kept_striking": sum(c["striking"] for c in out_clips),
           "pool_exhausted": len(out_clips) < N_KEEP, "drop_counts": reasons, "clips": out_clips, "skipped": skipped}
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[as] done: kept {len(out_clips)} ({doc['kept_striking']} striking) / tried {len(tried)}; drops {reasons}",
          flush=True)


if __name__ == "__main__":
    main()
