"""Wave 3 of mixed candidates: by GENRE, no screening (2026-09-21 night, Adam: "find 20 more
candidates with no screening, just using reasoning on which type of videos could that be").

A Fable panel of one reasoned about which kinds of video naturally have the mixed structure --
a camera fixed on one continuous sound-maker inside a busy shared place, so a random 20-s cut
catches a visible sound and a clear off-screen event: village cooking (rooster, motorbike horn
off camera), farm chores (cows, dog, tractor), dog groomer / daycare (other dogs bark), family
cooking vlogs (baby, doorbell, microwave), shop-dog workshops, mechanic garages, street-food
long takes, fishing boats, renovation crews, bird feeders, barista shifts, vet clinics, toddler
playrooms, goat farms. Avoid: scripted drama (foley, fast cuts), performer + traffic (texture),
night security cams (IR, mono), sports (crowd = texture, commentary = speech).

Only the quality screen runs (video + audio, >= 15 s, not silent, not a still, colour); no
audio model, no VLM. Clips land in data/input/benchmark/unsorted/ as m3_<kind>_<n>.mp4 and are
logged in benchmark/sources_mixed3.json.

Usage: python scripts/source_mixed3.py [n_wanted]     (default 22)
"""
from __future__ import annotations

import random
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.source_batch3 import screen, _accept, STAGE, _clip_len
import scripts.source_batch3 as b3

ARCHIVE = STAGE / "yt_archive.txt"
LOG = ROOT / "benchmark" / "sources_mixed3.json"

# (kind, query) -- ranked by the panel; two queries per kind
QUERIES = [
    ("village_cooking", "village cooking outdoor fire full video"),
    ("village_cooking", "village life cooking rooster dog countryside vlog"),
    ("farm_chores", "morning farm chores vlog milking feeding animals"),
    ("farm_chores", "homestead daily chores full day vlog"),
    ("dog_groomer", "dog groomer full groom vlog busy salon"),
    ("dog_daycare", "dog daycare day in the life"),
    ("family_cooking", "mom of 3 day in the life cooking cleaning vlog"),
    ("family_cooking", "cooking dinner with kids at home vlog"),
    ("shop_dog_workshop", "blacksmith forging shop dog vlog"),
    ("shop_dog_workshop", "woodworking shop day vlog building"),
    ("mechanic_garage", "mechanic shop day in the life busy garage"),
    ("mechanic_garage", "diesel shop vlog full day repairs"),
    ("street_food", "street food vendor long video Hanoi"),
    ("street_food", "Indian street food full process busy market"),
    ("fishing_boat", "commercial fishing boat vlog hauling nets"),
    ("harbor_market", "harbor fish market unloading catch"),
    ("renovation_crew", "house renovation vlog crew working full day"),
    ("framing_crew", "framing crew vlog job site"),
    ("bird_feeder", "backyard bird feeder live camera daytime"),
    ("squirrel_feeder", "squirrel feeder camera long video backyard"),
    ("barista_shift", "barista POV shift vlog busy cafe"),
    ("bakery_shift", "bakery morning shift vlog"),
    ("vet_clinic", "vet clinic day in the life vlog"),
    ("vet_tech", "veterinary technician shift vlog"),
    ("toddler_play", "toddler playing at home family vlog"),
    ("kids_living_room", "kids playing living room daily vlog"),
    ("goat_farm", "goat farm daily routine vlog"),
    ("backyard_chickens", "backyard chickens morning routine"),
]


def _download(name: str, query: str, deep: bool) -> tuple[Path | None, str, str]:
    raw = STAGE / f"{name}.mp4"
    raw.unlink(missing_ok=True)
    meta = STAGE / f"{name}.meta"
    meta.unlink(missing_ok=True)
    sec = _clip_len()
    start = random.choice((90, 150, 240))            # well inside a long video
    subprocess.run(
        ["yt-dlp", f"ytsearch{12 if deep else 6}:{query}", "--max-downloads", "1"]
        + (["--playlist-items", "7:12"] if deep else [])
        + ["--download-sections", f"*{start}-{start + sec}",
           "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b",
           "--merge-output-format", "mp4", "--force-keyframes-at-cuts",
           "--match-filter", f"duration>{start + 90} & duration<7200",
           "--download-archive", str(ARCHIVE),
           "--print-to-file", "%(webpage_url)s|%(title)s", str(meta),
           "--no-warnings", "-o", str(raw)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=420)
    url, title = "", ""
    if meta.exists():
        parts = meta.read_text(encoding="utf-8").strip().split("|", 1)
        url, title = parts[0], (parts[1] if len(parts) > 1 else "")
    return (raw if raw.exists() else None), url, title


def main(want_n: int) -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    b3.LOG = LOG
    got = 0
    for deep in (False, True):
        for i, (kind, query) in enumerate(QUERIES, 1):
            if got >= want_n:
                break
            name = f"m3_{kind}_{i}" + ("b" if deep else "")
            if any((ROOT / "data" / "input" / "benchmark" / d / f"{name}.mp4").exists()
                   for d in ("unsorted", "mixed", "seen_ambient", "unseen_ambient", "no_ambient", "_dropped", "_bad")):
                continue
            print(f"  {name}: {query}", flush=True)
            raw, url, title = _download(name, query, deep)
            if raw is None:
                print("    skip (no download)", flush=True)
                continue
            if _accept(raw, name, {"source": "youtube", "url": url, "title": title,
                                   "targeted": "mixed3 (genre, no screening)", "kind": kind,
                                   "license": "research use; not redistributed"}):
                got += 1
                print(f"    KEEP {got}/{want_n}: {title[:70]}", flush=True)
    print(f"done: {got} kept")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 22)
