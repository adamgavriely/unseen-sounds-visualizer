"""Wave 7: 10-s cuts from film TRAILERS (2026-09-22, Adam: "try to get maybe 10 seconds from trailers").

Trailers pack many sound events into a minute, so short cuts often hold an off-screen sound
(siren, blast, scream, knock) next to an on-screen one. For each film: the official trailer
(60-200 s), four 10-s cuts at 30 / 50 / 70 / 85 %, each kept only if BEATs hears a drawable,
non-speech, non-music sound (scripts/source_mixed2.audio_families). Detector only; no gate.
Clips land in data/input/benchmark/unsorted/ as t1_<film>_<k>.mp4 (quality check with a 10-s
floor), sources in benchmark/sources_trailers.json.

Usage: python scripts/source_trailers.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.source_batch3 as b3
from scripts.source_batch3 import STAGE, _probe, _meanvol, _frame
from scripts.source_mixed2 import audio_families

LOG = ROOT / "benchmark" / "sources_trailers.json"
AUDIO = ROOT / "benchmark" / "gold" / "trailers_audio.json"
BENCH = ROOT / "data" / "input" / "benchmark"
SEC = 10
FRACS = {"a": 0.30, "b": 0.50, "c": 0.70, "d": 0.85}
FMT = "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b"

FILMS = [
    # thriller / horror
    "the others", "panic room", "hereditary", "bird box", "10 cloverfield lane", "the babadook", "the strangers",
    "green room", "barbarian", "smile 2022", "talk to me 2022", "the invisible man 2020", "us 2019", "it 2017",
    "the ring 2002", "insidious", "sinister", "the descent", "a quiet place part ii", "evil dead rise",
    # war / action
    "the pianist", "band of brothers", "all quiet on the western front 2022", "the outpost 2020", "lone survivor",
    "13 hours", "american sniper", "civil war 2024", "the covenant 2023", "extraction 2020", "top gun maverick",
    "mission impossible fallout", "the batman 2022", "nobody 2021", "ambulance 2022",
    # drama / crime
    "the pianist", "room 2015", "take shelter", "cast away", "titanic", "the revenant", "sound of metal", "coda",
    "prisoners", "wind river", "hell or high water", "uncut gems", "good time 2017", "the town 2010",
    # disaster / survival
    "the wave 2015", "greenland", "san andreas", "twisters 2024", "the impossible", "society of the snow",
    "everest 2015", "deepwater horizon", "crawl 2019", "the swarm 2020",
    # sci-fi
    "cloverfield", "a quiet place day one", "nope", "war of the worlds 2005", "arrival", "annihilation", "prey 2022",
    "65 2023", "godzilla minus one", "dune part two",
    # tv
    "chernobyl hbo", "the bear season 2", "stranger things 4", "the last of us hbo", "shogun 2024", "masters of the air",
]


def _run(args, timeout=420):
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def screen10(f: Path):
    """the batch-3 quality screen with a 10-s floor"""
    from PIL import ImageChops
    import statistics, tempfile
    dur, hasv, hasa = _probe(f)
    if not hasv: return "no video"
    if not hasa: return "no audio"
    if dur < 9.5: return f"short {dur:.0f}s"
    if _meanvol(f) < -50: return "near-silent"
    with tempfile.TemporaryDirectory() as td:
        a, b = _frame(f, dur * 0.3, td, "a"), _frame(f, dur * 0.7, td, "b")
        if a and b and a.size == b.size and statistics.mean(ImageChops.difference(a.convert("L"), b.convert("L")).getdata()) < 2.0:
            return "still image"
        if a and statistics.mean(a.convert("HSV").split()[1].getdata()) < 14:
            return "black&white"
    return None


def _pick(query: str):
    r = _run(["yt-dlp", f"ytsearch5:{query}", "--simulate", "--no-warnings",
              "--match-filter", "duration>55 & duration<200", "--print", "%(id)s\t%(duration)s\t%(title)s"], timeout=180)
    for line in r.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3 and parts[1] not in ("NA", ""):
            return parts[0], float(parts[1]), parts[2]
    return None


def _grab(name: str, vid: str, start: float) -> Path | None:
    raw = STAGE / f"{name}.mp4"
    raw.unlink(missing_ok=True)
    _run(["yt-dlp", f"https://www.youtube.com/watch?v={vid}", "--download-sections", f"*{start:.0f}-{start + SEC:.0f}",
          "-f", FMT, "--merge-output-format", "mp4", "--force-keyframes-at-cuts", "--no-warnings", "-o", str(raw)])
    return raw if raw.exists() else None


def _exists(name: str) -> bool:
    return any((BENCH / d / f"{name}.mp4").exists() for d in ("unsorted", "_dropped", "_bad", "mixed", "seen_ambient", "unseen_ambient", "no_ambient"))


def main() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    b3.LOG = LOG
    b3.screen = screen10                     # _accept calls the module-level screen
    audio = json.loads(AUDIO.read_text(encoding="utf-8")) if AUDIO.exists() else {}
    kept = 0
    seen = set()
    for film in FILMS:
        if film in seen:
            continue
        seen.add(film)
        slug = "".join(ch if ch.isalnum() else "_" for ch in film).strip("_")[:28]
        base = f"t1_{slug}"
        if all(_exists(base + k) for k in FRACS):
            continue
        hit = _pick(f"{film} official trailer")
        if not hit:
            print(f"  {base}: no trailer found", flush=True); continue
        vid, dur, title = hit
        for suf, frac in FRACS.items():
            name = base + suf
            if _exists(name):
                continue
            start = max(5.0, frac * dur - SEC / 2)
            raw = _grab(name, vid, start)
            if raw is None:
                print(f"  {name}: skip (no download)", flush=True); continue
            try:
                fams = audio_families(raw)
            except Exception as e:
                print(f"  {name}: audio failed ({type(e).__name__})", flush=True); raw.unlink(missing_ok=True); continue
            if not fams:
                print(f"  {name}: drop (no drawable sound)", flush=True); raw.unlink(missing_ok=True); continue
            if b3._accept(raw, name, {"source": "youtube", "url": f"https://www.youtube.com/watch?v={vid}", "title": title,
                                      "targeted": "trailers (10-s cuts, audio check only)", "film": film, "cut": f"{int(frac * 100)}%",
                                      "license": "research use; not redistributed"}):
                kept += 1
                audio[name] = fams
                AUDIO.write_text(json.dumps(audio, indent=1, ensure_ascii=False), encoding="utf-8")
                print(f"  {name}: KEEP ({kept}) {[f['family'] for f in fams]}", flush=True)
    print(f"done: {kept} kept")


if __name__ == "__main__":
    main()
