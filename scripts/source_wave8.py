"""Wave 8: Fable's NEW targets (2026-09-22 night) -- videos where the unseen / mixed structure is
sustained, or where the clip IS the event: bodycam / dashcam incidents, single-take documentary
moments, live-stream intrusions, pets and kids reacting to off-screen sounds, TV/film scenes not
used before, sports moments. Films/series already used are excluded. Real combat footage
(trench assaults, shootings) is left out.

Per query: the top 2 uploads (30 s - 15 min), two 15-s cuts each (40 % and 60 % of the upload;
a short upload under 90 s gets one cut at its middle). No audio check, no VLM (Adam: "just give
me scenes"). Clips land in data/input/benchmark/unsorted/ as w8_<slug>_<k>.mp4, sources in
benchmark/sources_wave8.json.

Usage: python scripts/source_wave8.py
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

import scripts.source_batch3 as b3
from scripts.source_batch3 import STAGE, _accept, _clip_len

LOG = ROOT / "benchmark" / "sources_wave8.json"
BENCH = ROOT / "data" / "input" / "benchmark"
FMT = "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b"

# (slug, query, expected U/M) -- Fable, 2026-09-22 night
QUERIES = [
    ("bodycam_breach_dog", "police bodycam breaching ram door dog inside", "M"),
    ("ring_distant_boom", "Ring doorbell camera explosion heard distant boom", "U"),
    ("dashcam_sonic_boom", "dashcam sonic boom jet flyover reaction", "U"),
    ("chaser_hail_siren", "storm chaser windshield hail tornado siren", "M"),
    ("helmetcam_interior", "firefighter helmet cam interior attack raw", "M"),
    ("firetruck_cab_view", "fire truck cab view responding intersection", "M"),
    ("dashcam_crossing_horn", "dashcam railroad crossing near miss horn", "M"),
    ("gopro_rear_ended", "GoPro motorcycle rear-ended at light", "M"),
    ("reporter_shelling", "war reporter live shot incoming shelling ducks", "U"),
    ("kyiv_balcony_siren", "Kyiv balcony air raid siren explosions phone video", "U"),
    ("protest_flashbang", "protest reporter flash-bang tear gas raw", "M"),
    ("deer_rut_hide", "red deer rut hide stag roaring rival unseen", "M"),
    ("safari_lion_roar", "safari vehicle lion roar heard hidden bush", "U"),
    ("elephant_thicket", "elephant trumpet from thicket hikers freeze", "U"),
    ("glacier_calving", "glacier calving cruise boom before ice falls", "U"),
    ("streamer_earthquake", "streamer earthquake live reaction desk", "U"),
    ("streamer_doorbell_dog", "streamer doorbell interrupts dog goes crazy", "U"),
    ("warzone_stream_dog", "Warzone stream dog barking background", "M"),
    ("streamer_fireworks", "streamer fireworks outside window mid stream", "U"),
    ("streamer_smoke_alarm", "streamer smoke alarm goes off mid stream", "U"),
    ("irl_stream_crash", "IRL stream walking crash behind camera", "M"),
    ("dog_howl_ambulance", "dog howls at passing ambulance siren", "M"),
    ("cat_knock_door", "cat reacts to knock at door", "U"),
    ("baby_thunder", "baby startled by thunder", "U"),
    ("dog_fireworks_window", "dog barks at fireworks window", "M"),
    ("dog_vacuum_next_room", "dog hides vacuum next room", "U"),
    ("horse_gunshot", "horse spooks gunshot nearby hunting", "M"),
    ("kids_backfire", "kids jump at car backfire street", "U"),
    ("zone_of_interest_garden", "Zone of Interest garden pool scene", "U"),
    ("true_detective_tracking", "True Detective S1E4 six-minute tracking shot", "M"),
    ("duel_diner_horn", "Duel 1971 diner scene truck horn", "U"),
    ("bcs_bagman_ambush", "Better Call Saul Bagman desert ambush", "M"),
    ("oppenheimer_trinity", "Oppenheimer Trinity delayed shockwave", "U"),
    ("son_of_saul_opening", "Son of Saul opening long take", "U"),
    ("generation_kill_mortars", "Generation Kill convoy night mortars", "M"),
    ("golf_plane_overhead", "golf broadcast tee shot plane overhead", "M"),
    ("boxing_ringside_bell", "boxing ringside camera round bell", "M"),
    ("rally_spectator_unseen", "rally spectator car passes next car unseen", "M"),
]


def _run(args, timeout=420):
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def _hits(query: str, k: int = 2):
    r = _run(["yt-dlp", f"ytsearch6:{query}", "--simulate", "--no-warnings",
              "--match-filter", "duration>30 & duration<900", "--print", "%(id)s\t%(duration)s\t%(title)s"], timeout=180)
    out = []
    for line in r.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3 and parts[1] not in ("NA", "") and "compilation" not in parts[2].lower():
            out.append((parts[0], float(parts[1]), parts[2]))
    return out[:k]


def _grab(name: str, vid: str, start: float, sec: int) -> Path | None:
    raw = STAGE / f"{name}.mp4"
    raw.unlink(missing_ok=True)
    _run(["yt-dlp", f"https://www.youtube.com/watch?v={vid}", "--download-sections", f"*{start:.0f}-{start + sec:.0f}",
          "-f", FMT, "--merge-output-format", "mp4", "--force-keyframes-at-cuts", "--no-warnings", "-o", str(raw)])
    return raw if raw.exists() else None


def _exists(name: str) -> bool:
    return any((BENCH / d / f"{name}.mp4").exists() for d in ("unsorted", "_dropped", "_bad", "mixed", "seen_ambient", "unseen_ambient", "no_ambient"))


def main() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    b3.LOG = LOG
    kept = 0
    for slug, query, expect in QUERIES:
        try:
            hits = _hits(query)
        except Exception as e:
            print(f"  {slug}: search failed ({type(e).__name__})", flush=True); continue
        if not hits:
            print(f"  {slug}: no hits", flush=True); continue
        for j, (vid, dur, title) in enumerate(hits, 1):
            sec = _clip_len()
            cuts = [0.5] if dur < 90 else [0.4, 0.6]
            for k, frac in enumerate(cuts, 1):
                name = f"w8_{slug}_{j}{'abc'[k - 1]}"
                if _exists(name):
                    continue
                start = max(5.0, frac * dur - sec / 2)
                raw = _grab(name, vid, start, sec)
                if raw is None:
                    print(f"  {name}: skip (no download)", flush=True); continue
                if _accept(raw, name, {"source": "youtube", "url": f"https://www.youtube.com/watch?v={vid}", "title": title,
                                       "targeted": f"wave8 ({expect})", "query": query, "cut": f"{int(frac * 100)}%",
                                       "license": "research use; not redistributed"}):
                    kept += 1
                    print(f"  {name}: KEEP ({kept}) {title[:60]}", flush=True)
    print(f"done: {kept} kept")


if __name__ == "__main__":
    main()
