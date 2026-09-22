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

ARCHIVE = STAGE / "yt_archive.txt"
LOG = ROOT / "benchmark" / "sources_wave6.json"
AUDIO = ROOT / "benchmark" / "gold" / "wave6_audio.json"
BENCH = ROOT / "data" / "input" / "benchmark"
FAST = "--fast" in sys.argv
FMT = "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b"

QUERIES = [
    ("street_walk_events", "city walk 4k no talking binaural sirens horns"),
    ("night_walk", "night city walk no talking ambient sound"),
    ("market_walk", "market walk no talking ambient sound vendors"),
    ("harbor_walk", "harbor walk no talking boats gulls ambient"),
    ("train_station_walk", "train station walk no talking announcements trains"),
    ("subway_pov", "subway ride pov no talking doors announcements"),
    ("bus_ride", "bus ride pov no talking city sounds"),
    ("airport_walk", "airport walk no talking announcements"),
    ("construction_ambience", "construction site ambience raw sound no talking"),
    ("workshop_ambience", "workshop ambience no talking tools"),
    ("farm_ambience", "farm ambience no talking animals morning"),
    ("village_walk", "village walk no talking animals bells"),
    ("zoo_walk", "zoo walk no talking animals sounds"),
    ("kennel_ambience", "dog kennel ambience barking raw"),
    ("stable_ambience", "horse stable ambience raw sound"),
    ("kitchen_restaurant", "restaurant kitchen ambience raw no music"),
    ("cafe_ambience", "cafe ambience raw sound no music dishes"),
    ("school_ambience", "school corridor ambience bell raw"),
    ("hospital_ambience", "hospital corridor ambience raw sound"),
    ("carnival_walk", "carnival walk no talking rides screams"),
    ("beach_walk_events", "beach walk no talking kids dogs gulls"),
    ("park_walk_events", "park walk no talking playground dogs"),
    ("forest_walk_events", "forest walk no talking birds woodpecker chainsaw"),
    ("river_walk", "river walk no talking boats birds"),
    ("storm_footage", "storm footage raw sound thunder wind no music"),
    ("flood_footage", "flood footage raw sound no music"),
    ("fireworks_street", "fireworks street raw sound crowd dogs"),
    ("dashcam_raw", "dashcam raw audio city horns sirens no music"),
    ("bodycam_fire", "firefighter helmet cam raw audio"),
    ("air_show_raw", "air show raw sound crowd no music"),
    ("race_track_raw", "race track raw sound pit lane no commentary"),
    ("ship_deck_raw", "ship deck raw sound horn engine gulls"),
    ("logging_raw", "logging chainsaw raw sound forest"),
    ("blacksmith_raw", "blacksmith raw sound hammering no talking"),
    ("garage_raw", "mechanic garage raw sound air tools"),
    ("demolition_raw", "demolition raw sound crowd"),
    ("protest_raw", "protest raw sound drums whistles sirens no commentary"),
    ("playground_raw", "playground raw sound kids"),
    ("cattle_raw", "cattle farm raw sound cows tractor"),
    ("poultry_raw", "chicken farm raw sound rooster"),
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


def audio_check(video: Path):
    """one BEATs pass: (music max, speech seconds, drawable families rated >= 2)"""
    from src.stage1_audio_extraction import extract_audio
    from src.stage4_audio_event_detection import detect_events
    from benchmark.gold.build_tool import importance_of
    from src.labels import is_salient_nonspeech, canonical
    config.LABEL_FILTER = "depictable"
    with tempfile.TemporaryDirectory() as td:
        media = extract_audio(video, Path(td) / "a.wav", config.SAMPLE_RATE)
        events = detect_events(Path(media.wav_path), threshold=0.1, min_dur=config.AED_MIN_DUR, device="cuda", model="beats")
    mus = max((float(e.confidence) for e in events if e.label == "Music"), default=0.0)
    speech_s = sum(e.end - e.start for e in events if e.label == "Speech" and e.confidence >= 0.5)
    fams = {}
    for e in events:
        if e.confidence < 0.25 or not is_salient_nonspeech(e.label):
            continue
        fam = canonical(e.label)
        if importance_of(e.label.lower(), fam) < 2:
            continue
        if fam not in fams or e.confidence > fams[fam]["conf"]:
            fams[fam] = {"family": fam, "detail": e.label, "conf": round(float(e.confidence), 2), "start": round(e.start, 1), "end": round(e.end, 1)}
    return mus, speech_s, sorted(fams.values(), key=lambda f: -f["conf"])


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
                # one cheap audio pass (~5 s): interviews / news (speech most of the clip), music beds and
                # clips with fewer than two drawable sound events are dropped (Adam: "interviews or news, no sounds")
                try:
                    mus, speech_s, fams = audio_check(raw)
                except Exception as e:
                    print(f"  {name}: audio failed ({type(e).__name__})", flush=True); raw.unlink(missing_ok=True); continue
                if mus >= 0.5 or speech_s > 0.5 * sec or len(fams) < 2:
                    print(f"  {name}: drop (music {mus:.2f}, speech {speech_s:.0f}s, families {[f['family'] for f in fams]})", flush=True); raw.unlink(missing_ok=True); continue
                if _accept(raw, name, {"source": "youtube", "url": f"https://www.youtube.com/watch?v={vid}", "title": title,
                                       "targeted": "wave6 (unseen/mixed, real footage, audio check)", "kind": kind,
                                       "cut": f"{int(frac * 100)}%", "music": round(mus, 2), "license": "research use; not redistributed"}):
                    kept += 1
                    audio[name] = fams
                    AUDIO.write_text(json.dumps(audio, indent=1, ensure_ascii=False), encoding="utf-8")
                    print(f"  {name}: KEEP ({kept}) music {mus:.2f} {[f['family'] for f in fams]}", flush=True)
                    if kept % 4 == 0:                          # the tool sees new clips every few keeps
                        subprocess.run([sys.executable, str(ROOT / "benchmark" / "gold" / "build_tool.py")], capture_output=True)
    print(f"done: {kept} kept")


if __name__ == "__main__":
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    main(int(nums[0]) if nums else 40)
