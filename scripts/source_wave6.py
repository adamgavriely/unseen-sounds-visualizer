"""Wave 6: the last unseen / mixed candidates (2026-09-22, Adam: "I still need 7 unseen and 2 mixed,
preferably with less music and more sounds; complex scenes where the pipeline can shine").

Real footage only (no film scores): raw / no-commentary / POV / bodycam / dashcam uploads of places
where several sound events happen and some sources are out of frame. Per query: the first 2 hits
not in the archive, two 16-20 s cuts each (35 % and 65 % of the upload). Keep rule, audio only
(BEATs; no gate, no VLM): at least TWO distinct drawable sound families rated >= 2, and music
below 0.5 for the clip. Clips: data/input/benchmark/unsorted/m6_<kind>_<n><a|b><1|2>.mp4;
families: benchmark/gold/wave6_audio.json; sources: benchmark/sources_wave6.json.

Usage: python scripts/source_wave6.py [n_wanted]      (default 40)
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from scripts.source_batch3 import _accept, STAGE, _clip_len
import scripts.source_batch3 as b3
from scripts.source_mixed2 import audio_families

ARCHIVE = STAGE / "yt_archive.txt"
LOG = ROOT / "benchmark" / "sources_wave6.json"
AUDIO = ROOT / "benchmark" / "gold" / "wave6_audio.json"
BENCH = ROOT / "data" / "input" / "benchmark"
FMT = "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b"

QUERIES = [
    ("dashcam_city", "dashcam city driving raw no music horns sirens"),
    ("dashcam_near_miss", "dashcam near miss compilation no music"),
    ("bodycam_call", "police bodycam footage traffic stop raw"),
    ("bodycam_fire", "firefighter bodycam raw audio no music"),
    ("ems_ride", "ambulance ride along raw footage"),
    ("er_corridor", "hospital emergency room walkthrough raw"),
    ("airport_terminal", "airport terminal walk announcements no talking"),
    ("train_platform", "train station platform ambience raw footage arrivals"),
    ("subway_ride", "subway ride pov doors announcements no music"),
    ("market_pov", "walking through busy market pov no talking"),
    ("night_market", "night market walk pov raw audio"),
    ("harbor_work", "fishing harbor unloading raw footage"),
    ("shipyard", "shipyard work raw audio no music"),
    ("construction_pov", "construction site pov walk no music"),
    ("demolition", "building demolition raw footage crowd"),
    ("farm_yard", "farm yard morning animals raw audio no music"),
    ("zoo_walk", "zoo walk pov animals raw audio no commentary"),
    ("wildlife_cam", "trail camera footage animals sounds"),
    ("backyard_cam", "backyard night camera animals raw"),
    ("home_kitchen_pov", "cooking pov kitchen raw audio no music family"),
    ("workshop_pov", "workshop pov building no talking no music"),
    ("garage_repair", "garage repair raw audio air tools"),
    ("street_riot", "protest clashes raw footage no commentary"),
    ("stadium_tunnel", "stadium tunnel players entrance raw audio"),
    ("school_hall", "school hallway bell raw footage"),
    ("hotel_lobby", "hotel lobby ambience elevator raw"),
    ("storm_house", "storm hitting house raw footage no music"),
    ("flood_street", "flash flood street raw footage"),
    ("earthquake_cam", "earthquake caught on camera indoor raw"),
    ("car_chase_news", "news helicopter police chase raw feed"),
    ("crash_test", "crash test facility raw footage"),
    ("rocket_launch_crowd", "rocket launch crowd reaction raw audio"),
    ("air_show", "air show flyover crowd raw audio no music"),
    ("motor_race_pit", "pit lane raw audio race no commentary"),
    ("boxing_gym", "boxing gym raw audio training"),
    ("kindergarten", "kindergarten playground raw audio"),
    ("dog_shelter", "dog shelter walkthrough raw audio"),
    ("cattle_auction", "cattle auction raw footage"),
    ("carnival", "carnival rides raw footage screams"),
    ("ferry_deck", "ferry deck departure horn gulls raw"),
]


def _run(args, timeout=420):
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def _pick(query: str, k: int = 2):
    r = _run(["yt-dlp", f"ytsearch10:{query}", "--simulate", "--no-warnings", "--download-archive", str(ARCHIVE),
              "--match-filter", "duration>120 & duration<5400", "--print", "%(id)s\t%(duration)s\t%(title)s"], timeout=180)
    hits = []
    for line in r.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3 and parts[1] not in ("NA", ""):
            hits.append((parts[0], float(parts[1]), parts[2]))
    return hits[:k]


def _grab(name: str, vid: str, start: float, sec: int) -> Path | None:
    raw = STAGE / f"{name}.mp4"
    raw.unlink(missing_ok=True)
    _run(["yt-dlp", f"https://www.youtube.com/watch?v={vid}", "--download-sections", f"*{start:.0f}-{start + sec:.0f}",
          "-f", FMT, "--merge-output-format", "mp4", "--force-keyframes-at-cuts", "--no-warnings", "-o", str(raw)])
    return raw if raw.exists() else None


def music_level(video: Path) -> float:
    """max BEATs confidence for Music over the clip (audio only)"""
    from src.stage1_audio_extraction import extract_audio
    from src.stage4_audio_event_detection import detect_events
    with tempfile.TemporaryDirectory() as td:
        media = extract_audio(video, Path(td) / "a.wav", config.SAMPLE_RATE)
        events = detect_events(Path(media.wav_path), threshold=0.1, min_dur=config.AED_MIN_DUR, device="cuda", model="beats")
    return max((float(e.confidence) for e in events if e.label == "Music"), default=0.0)


def _exists(name: str) -> bool:
    return any((BENCH / d / f"{name}.mp4").exists() for d in ("unsorted", "_dropped", "_bad", "mixed", "seen_ambient", "unseen_ambient", "no_ambient"))


def main(want_n: int) -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    b3.LOG = LOG
    audio = json.loads(AUDIO.read_text(encoding="utf-8")) if AUDIO.exists() else {}
    qs = QUERIES[:]
    random.shuffle(qs)
    kept = 0
    for kind, query in qs:
        if kept >= want_n:
            break
        i = QUERIES.index((kind, query)) + 1
        try:
            hits = _pick(query)
        except Exception as e:
            print(f"  {kind}: search failed ({type(e).__name__})", flush=True); continue
        if not hits:
            print(f"  {kind}: no hits", flush=True); continue
        for h, (vid, dur, title) in enumerate(hits):
            # mark the video in the archive so a re-run picks other videos
            with ARCHIVE.open("a", encoding="utf-8") as f:
                f.write(f"youtube {vid}\n")
            for c, frac in ((1, 0.35), (2, 0.65)):
                name = f"m6_{kind}_{i}{'ab'[h]}{c}"
                if _exists(name):
                    continue
                sec = _clip_len()
                start = max(30.0, frac * dur - sec / 2)
                raw = _grab(name, vid, start, sec)
                if raw is None:
                    print(f"  {name}: skip (no download)", flush=True); continue
                try:
                    mus = music_level(raw)
                    fams = audio_families(raw) if mus < 0.5 else []
                except Exception as e:
                    print(f"  {name}: audio failed ({type(e).__name__})", flush=True); raw.unlink(missing_ok=True); continue
                if mus >= 0.5 or len(fams) < 2:
                    print(f"  {name}: drop (music {mus:.2f}, families {[f['family'] for f in fams]})", flush=True); raw.unlink(missing_ok=True); continue
                if _accept(raw, name, {"source": "youtube", "url": f"https://www.youtube.com/watch?v={vid}", "title": title,
                                       "targeted": "wave6 (unseen/mixed, real footage, audio check)", "kind": kind,
                                       "cut": f"{int(frac * 100)}%", "music": round(mus, 2), "license": "research use; not redistributed"}):
                    kept += 1
                    audio[name] = fams
                    AUDIO.write_text(json.dumps(audio, indent=1, ensure_ascii=False), encoding="utf-8")
                    print(f"  {name}: KEEP ({kept}) music {mus:.2f} {[f['family'] for f in fams]}", flush=True)
    print(f"done: {kept} kept")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
