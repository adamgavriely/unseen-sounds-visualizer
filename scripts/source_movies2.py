"""Wave 5: UNSEEN candidates from films (2026-09-22, Adam: "find more videos that are unseen audios,
preferably from movies, 2-3 scenes per movie, ~15 s each, diverse genres, scenes with more than one
sound; at least 50").

Scenes chosen for off-screen sound design: a creature, a knock, a phone, a siren, shelling, a train,
a crowd, a storm, something heard before it is seen. One YouTube upload per query (the scene by
name), one 15-s cut at ~40 % of the upload (past the intro, before the outro), no screening beyond
the quality check (video + audio, >= 15 s, not a still, colour). Clips land in
data/input/benchmark/unsorted/ as m5_<genre>_<film>_<n>.mp4, logged in benchmark/sources_movies2.json.
Adam's ticks decide what each clip is.

Usage: python scripts/source_movies2.py [n_wanted]     (default 60)
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.source_batch3 import _accept, STAGE
import scripts.source_batch3 as b3

ARCHIVE = STAGE / "yt_archive.txt"
LOG = ROOT / "benchmark" / "sources_movies2.json"
SEC = 15
FMT = "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b"

# (genre, film, scene query) -- 2-3 scenes per film where a sound is heard before / without its source
QUERIES = [
    # horror / thriller
    ("horror", "quiet_place", "a quiet place basement flooding scene"),
    ("horror", "quiet_place", "a quiet place nail scene stairs"),
    ("horror", "quiet_place", "a quiet place cornfield silo scene"),
    ("horror", "conjuring", "the conjuring hide and clap scene"),
    ("horror", "conjuring", "the conjuring music box scene"),
    ("horror", "dont_breathe", "don't breathe basement scene lights out"),
    ("horror", "dont_breathe", "don't breathe dog chase scene"),
    ("horror", "hush", "hush 2016 window scene"),
    ("horror", "jaws", "jaws barrels scene"),
    ("horror", "jaws", "jaws beach attack scene"),
    ("horror", "alien", "alien 1979 air duct scene"),
    ("horror", "signs", "signs baby monitor scene"),
    ("horror", "signs", "signs basement scene"),
    ("horror", "shining", "the shining typewriter scene"),
    ("horror", "it_follows", "it follows beach scene"),
    ("horror", "the_descent", "the descent cave crawl scene"),
    # war / action
    ("war", "private_ryan", "saving private ryan omaha beach scene"),
    ("war", "private_ryan", "saving private ryan sniper scene bell tower"),
    ("war", "dunkirk", "dunkirk beach scene planes"),
    ("war", "dunkirk", "dunkirk mole scene stuka"),
    ("war", "1917", "1917 trench run scene"),
    ("war", "1917", "1917 farmhouse plane crash scene"),
    ("war", "black_hawk_down", "black hawk down first crash scene"),
    ("war", "hurt_locker", "the hurt locker sniper scene desert"),
    ("war", "fury", "fury 2014 tiger tank battle scene"),
    ("war", "hacksaw_ridge", "hacksaw ridge night scene"),
    ("action", "heat", "heat 1995 bank shootout scene"),
    ("action", "sicario", "sicario border crossing scene"),
    ("action", "sicario", "sicario tunnel scene"),
    ("action", "no_country", "no country for old men motel scene"),
    ("action", "no_country", "no country for old men coin toss scene"),
    ("action", "children_of_men", "children of men car ambush scene"),
    ("action", "mad_max", "mad max fury road storm scene"),
    ("action", "bourne", "bourne ultimatum waterloo station scene"),
    # drama
    ("drama", "godfather", "the godfather restaurant scene train"),
    ("drama", "godfather", "the godfather baptism scene"),
    ("drama", "taxi_driver", "taxi driver apartment scene sirens"),
    ("drama", "the_conversation", "the conversation opening scene union square"),
    ("drama", "blow_out", "blow out recording scene bridge"),
    ("drama", "there_will_be_blood", "there will be blood oil derrick explosion scene"),
    ("drama", "sound_of_metal", "sound of metal hearing loss scene"),
    ("drama", "coda", "coda 2021 dinner scene"),
    ("drama", "the_father", "the father 2020 apartment scene"),
    ("drama", "marriage_story", "marriage story argument scene"),
    ("drama", "manchester", "manchester by the sea fire scene"),
    ("drama", "roma", "roma 2018 beach scene"),
    ("drama", "roma", "roma 2018 earthquake scene hospital"),
    ("drama", "parasite", "parasite flood scene basement"),
    ("drama", "parasite", "parasite garden party scene"),
    # western / classic
    ("western", "once_upon_west", "once upon a time in the west opening station scene"),
    ("western", "unforgiven", "unforgiven saloon scene rain"),
    ("classic", "rear_window", "rear window courtyard scene"),
    ("classic", "rear_window", "rear window night scene scream"),
    ("classic", "psycho", "psycho shower scene"),
    ("classic", "the_birds", "the birds school playground scene"),
    ("classic", "the_birds", "the birds attic scene"),
    ("classic", "north_by_northwest", "north by northwest crop duster scene"),
    # comedy
    ("comedy", "home_alone", "home alone furnace scene"),
    ("comedy", "home_alone", "home alone kevin cooking scene"),
    ("comedy", "ferris", "ferris bueller doorbell scene"),
    ("comedy", "mr_bean", "mr bean hotel scene"),
    ("comedy", "hot_fuzz", "hot fuzz swan scene"),
    # sci-fi
    ("scifi", "arrival", "arrival first contact scene"),
    ("scifi", "interstellar", "interstellar docking scene"),
    ("scifi", "blade_runner", "blade runner 2049 los angeles scene"),
    ("scifi", "gravity", "gravity debris scene"),
    ("scifi", "war_of_worlds", "war of the worlds 2005 first attack scene"),
    ("scifi", "war_of_worlds", "war of the worlds ferry scene"),
    # disaster
    ("disaster", "impossible", "the impossible tsunami scene"),
    ("disaster", "twister", "twister 1996 drive-in scene"),
    ("disaster", "san_andreas", "san andreas earthquake scene"),
    ("disaster", "the_wave", "the wave 2015 tsunami scene"),
    ("disaster", "deepwater", "deepwater horizon blowout scene"),
    # sports / music-free crowd
    ("sports", "rocky", "rocky training run scene"),
    ("sports", "rush", "rush 2013 rain race scene"),
    ("sports", "ford_v_ferrari", "ford v ferrari le mans night scene"),
    # kids / family (live action)
    ("family", "et", "e.t. bike chase scene"),
    ("family", "jurassic", "jurassic park t-rex scene glass of water"),
    ("family", "jurassic", "jurassic park raptors kitchen scene"),
    ("family", "paddington", "paddington bathroom flood scene"),
]


def _run(args, timeout=420):
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def _pick(query: str):
    """(id, duration, title) of the first search hit not in the archive, 45 s - 15 min long"""
    r = _run(["yt-dlp", f"ytsearch5:{query}", "--simulate", "--no-warnings", "--download-archive", str(ARCHIVE),
              "--match-filter", "duration>45 & duration<900", "--print", "%(id)s\t%(duration)s\t%(title)s"], timeout=180)
    for line in r.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3 and parts[1] not in ("NA", ""):
            return parts[0], float(parts[1]), parts[2]
    return None


def _grab(name: str, vid: str, start: float) -> Path | None:
    raw = STAGE / f"{name}.mp4"
    raw.unlink(missing_ok=True)
    _run(["yt-dlp", f"https://www.youtube.com/watch?v={vid}", "--download-sections", f"*{start:.0f}-{start + SEC:.0f}",
          "-f", FMT, "--merge-output-format", "mp4", "--force-keyframes-at-cuts", "--download-archive", str(ARCHIVE),
          "--no-warnings", "-o", str(raw)])
    return raw if raw.exists() else None


def main(want_n: int) -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    b3.LOG = LOG
    got = 0
    for i, (genre, film, query) in enumerate(QUERIES, 1):
        if got >= want_n:
            break
        name = f"m5_{genre}_{film}_{i}"
        if any((ROOT / "data" / "input" / "benchmark" / d / f"{name}.mp4").exists()
               for d in ("unsorted", "mixed", "seen_ambient", "unseen_ambient", "no_ambient", "_dropped", "_bad")):
            continue
        try:
            hit = _pick(query)
        except Exception as e:
            print(f"  {name}: search failed ({type(e).__name__})", flush=True); continue
        if not hit:
            print(f"  {name}: no hit ({query})", flush=True); continue
        vid, dur, title = hit
        start = max(20.0, 0.4 * dur - SEC / 2)          # past the intro, before the outro
        print(f"  {name}: {title[:60]} ({dur:.0f} s, cut {start:.0f}-{start + SEC:.0f})", flush=True)
        raw = _grab(name, vid, start)
        if raw is None:
            print("    skip (no download)", flush=True); continue
        if _accept(raw, name, {"source": "youtube", "url": f"https://www.youtube.com/watch?v={vid}", "title": title,
                               "targeted": "movies2 (unseen candidates, no screening)", "genre": genre, "film": film,
                               "query": query, "license": "research use; not redistributed"}):
            got += 1
            print(f"    KEEP {got}/{want_n}", flush=True)
    print(f"done: {got} kept")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
