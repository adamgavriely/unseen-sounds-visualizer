"""Wave 4 of mixed candidates: by STRUCTURE (2026-09-22, Adam: "none of m2/m3 are mixed;
consult one Fable with the problem statement; find at least 30 more").

The panel's rule: rank by structure, not topic, and never name the off-screen event in the
query -- name the setting that guarantees the structure.
  A "many stations, one in frame": the same action at many spots, camera on one (kennels,
    goat barns at feeding time, bird colonies, zoo feedings, dog parks, shooting ranges,
    bowling lanes). Any cut works: the in-frame animal/shooter is the visible sound, the
    others are off screen.
  B "one subject + a distant event that repeats for minutes": dog + fireworks, hail on the
    windshield + thunder, firefighter helmet cam + other units' sirens, airsoft / re-enactment
    GoPro + return fire, duck blind + geese in the air.
  C "single targeted event" (doorbell, phone, alarm, knock): only works for a known scene ->
    search the scene by name and cut around the middle of the upload.
No screening of any kind (Adam): quality check only (video + audio, >= 15 s, not a still,
colour). Clips land in data/input/benchmark/unsorted/ as m4_<kind>_<n>.mp4, logged in
benchmark/sources_mixed4.json. No real combat footage: re-enactment, milsim, films only.

Usage: python scripts/source_mixed4.py [n_wanted]     (default 40)
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.source_batch3 import _accept, STAGE, _clip_len
import scripts.source_batch3 as b3

ARCHIVE = STAGE / "yt_archive.txt"
LOG = ROOT / "benchmark" / "sources_mixed4.json"

# (kind, query, cut): cut "random" = 60-200 s into a long video; "mid" = the middle of a short
# upload (a named scene). Two queries per kind; high-hit kinds get two hits each (see PER_QUERY).
QUERIES = [
    # A: many stations, one in frame
    ("kennel", "animal shelter kennel walkthrough", "random"),
    ("kennel", "dog rescue kennels tour raw footage", "random"),
    ("goat_barn", "goat farm feeding time morning chores", "random"),
    ("sheep_barn", "sheep barn lambing season no commentary", "random"),
    ("bird_colony", "gannet colony ambient no music", "random"),
    ("penguin_colony", "penguin colony rookery raw footage", "random"),
    ("zoo_sealion", "sea lion feeding zoo full", "random"),
    ("safari", "safari game drive hyena morning", "random"),
    ("dog_park", "dog park afternoon no commentary", "random"),
    ("dog_daycare", "dog daycare playgroup cam", "random"),
    ("clay_shoot", "clay shooting competition full round", "random"),
    ("range_gopro", "shooting range day GoPro", "random"),
    ("bowling", "bowling league night full game", "random"),
    ("table_tennis_hall", "table tennis club hall training", "random"),
    ("construction", "construction site raw footage no talking", "random"),
    ("frame_raising", "timber frame raising day", "random"),
    ("frog_pond", "bullfrog pond night camera", "random"),
    # B: one subject + a repeating distant event
    ("dog_fireworks", "dog scared of fireworks new year", "random"),
    ("dog_bonfire", "bonfire night garden dog reaction", "random"),
    ("hail_dashcam", "hail storm car dashcam thunder", "random"),
    ("storm_chasers", "storm chasers hail core intercept", "random"),
    ("fire_helmetcam", "firefighter helmet cam structure fire", "random"),
    ("fire_bodycam", "fire department bodycam working fire", "random"),
    ("airsoft", "airsoft milsim GoPro full game", "random"),
    ("reenactment", "civil war reenactment cannon fire spectator view", "random"),
    ("live_fire", "live fire exercise combined arms raw", "random"),
    ("mortar_range", "mortar range training day", "random"),
    ("duck_hunt", "duck hunting opening day blind GoPro", "random"),
    ("goose_hunt", "goose hunt field full hunt", "random"),
    ("dog_doorbell_cam", "ring doorbell dog barking delivery", "random"),
    # C: named scenes, cut around the middle of the upload
    ("film_omaha", "saving private ryan omaha beach scene", "mid"),
    ("film_blackhawk", "black hawk down first crash scene", "mid"),
    ("film_1917", "1917 trench run scene", "mid"),
    ("sitcom_doorbell", "friends doorbell scene", "mid"),
    ("film_home_alone", "home alone kevin cooking scene", "mid"),
    ("film_quiet_place", "a quiet place basement scene", "mid"),
    ("film_conjuring", "the conjuring hide and clap scene", "mid"),
    ("film_alarm_clock", "alarm clock wakes up movie scene", "mid"),
    ("film_phone_dinner", "phone rings during dinner movie scene", "mid"),
    ("film_explosion_far", "distant explosion reaction movie scene", "mid"),
]
PER_QUERY = 2          # hits per query (two different videos)
FMT = "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b"


def _run(args, timeout=420):
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def _pick(query: str, min_dur: int, k: int) -> list[tuple[str, float, str]]:
    """(id, duration, title) of up to k search hits not in the archive and long enough"""
    r = _run(["yt-dlp", f"ytsearch12:{query}", "--simulate", "--no-warnings", "--download-archive", str(ARCHIVE),
              "--match-filter", f"duration>{min_dur} & duration<7200", "--print", "%(id)s\t%(duration)s\t%(title)s"], timeout=180)
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
          "-f", FMT, "--merge-output-format", "mp4", "--force-keyframes-at-cuts", "--download-archive", str(ARCHIVE),
          "--no-warnings", "-o", str(raw)])
    return raw if raw.exists() else None


def main(want_n: int) -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    b3.LOG = LOG
    got = 0
    for i, (kind, query, cut) in enumerate(QUERIES, 1):
        if got >= want_n:
            break
        sec = _clip_len()
        min_dur = 150 if cut == "random" else 60
        try:
            hits = _pick(query, min_dur, PER_QUERY)
        except Exception as e:
            print(f"  {kind}: search failed ({type(e).__name__})", flush=True); continue
        if not hits:
            print(f"  {kind}: no hits ({query})", flush=True); continue
        for j, (vid, dur, title) in enumerate(hits, 1):
            name = f"m4_{kind}_{i}{'abc'[j - 1]}"
            if any((ROOT / "data" / "input" / "benchmark" / d / f"{name}.mp4").exists()
                   for d in ("unsorted", "mixed", "seen_ambient", "unseen_ambient", "no_ambient", "_dropped", "_bad")):
                continue
            if cut == "random":
                start = random.uniform(60, max(61, min(dur - 45, 200)))
            else:
                start = max(20, dur / 2 - sec / 2)         # the middle of a named scene
            print(f"  {name}: {title[:60]} ({dur:.0f} s, cut {start:.0f}-{start + sec:.0f})", flush=True)
            raw = _grab(name, vid, start, sec)
            if raw is None:
                print("    skip (no download)", flush=True); continue
            if _accept(raw, name, {"source": "youtube", "url": f"https://www.youtube.com/watch?v={vid}", "title": title,
                                   "targeted": "mixed4 (structure, no screening)", "kind": kind, "cut": cut,
                                   "license": "research use; not redistributed"}):
                got += 1
                print(f"    KEEP {got}/{want_n}", flush=True)
    print(f"done: {got} kept")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
