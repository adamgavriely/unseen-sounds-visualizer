"""Source clips whose sound is STRUCTURALLY impossible to see.

Adam (2026-08-23): "aren't there videos of nature where we hear water but not see
it? people stepping on dry sticks but we only see them walking and we have to
understand something?" That reframes the search: stop asking WHICH sound, ask WHY
it cannot be on screen. Five structural patterns, in decreasing reliability:

  A. the camera operator's OWN sounds -- footsteps, breathing, clothing. A camera
     never shows who holds it, so in POV/handheld footage these are guaranteed
     off-screen. This is the strongest bet available and the one Adam described.
  B. occluded by the environment -- a stream behind trees, a sound through a wall
     or window, something around the corner.
  C. distance -- distant thunder, dogs across a valley, a town's bells: audible
     but too far to resolve visually.
  D. behind the camera -- someone or something approaching from out of frustum.
  E. sound carries MATERIAL rather than object -- footsteps on gravel vs snow vs
     dry sticks. Even with the walker in frame, "a twig snapped, something is
     near" is information only the audio provides. This is the deepest
     accessibility case: not a missing object, a missing meaning.

Queries below target A, B, C and E; D is not reliably searchable.

Usage: python scripts/source_occluded.py [n]
"""

from __future__ import annotations

import random
import statistics
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
from scripts.source_batch3 import screen, _accept, STAGE

WINDOW = 240           # seconds of the scene to fetch and search
ARCHIVE = STAGE / "yt_archive.txt"
MIN_EVENT_CONF = 0.20   # ambient textures score lower than bangs

# Sounds a film puts off-screen ON PURPOSE, and that a captioner would bracket.
NARRATIVE = {
    # A/E: the operator's own movement -- guaranteed off-screen in POV footage
    "Walk, footsteps", "Run", "Shuffle", "Wood", "Crumpling, crinkling", "Crack",
    "Breathing", "Rustling leaves", "Snow", "Squeak", "Creak", "Sanding",
    # B: occluded water / weather
    "Stream", "Waterfall", "Water", "Gurgling", "Trickle, dribble", "Drip",
    "Raindrop", "Rain on surface", "Rain",
    # C: distance
    "Thunder", "Thunderstorm", "Church bell", "Bell", "Wind",
    # kept from the film set for the distant-city cases
    "Siren", "Ambulance (siren)", "Police car (siren)", "Fire engine, fire truck (siren)",
    "Civil defense siren", "Emergency vehicle", "Air horn, truck horn",
    "Explosion", "Boom", "Bang", "Eruption", "Artillery fire", "Machine gun",
    "Gunshot, gunfire", "Fusillade", "Cap gun",
    "Glass", "Shatter", "Breaking", "Smash, crash",
    "Screaming", "Yell", "Shout", "Children shouting", "Whimper", "Crying, sobbing",
    "Helicopter", "Fixed-wing aircraft, airplane", "Jet engine", "Aircraft",
    "Alarm", "Fire alarm", "Car alarm", "Smoke detector, smoke alarm", "Alarm clock",
    "Thunder", "Thunderstorm", "Rain", "Wind",
    "Dog", "Bark", "Bow-wow", "Growling", "Howl",
    "Train horn", "Train whistle", "Vehicle horn, car horn, honking",
    "Skidding", "Tire squeal", "Crowd", "Cheering", "Applause",
    "Doorbell", "Knock", "Slam", "Squeak", "Creak",
    "Sewing machine", "Telephone bell ringing", "Ringtone", "Telephone",
}
MUSIC_HINTS = {"Music", "Background music", "Theme music", "Soundtrack music"}

QUERIES = [
    # --- A: the operator's own footsteps (guaranteed off-screen) ---------------
    ("oc_pov_forest_walk", "pov walking forest trail footsteps leaves"),
    ("oc_pov_dry_leaves", "walking through dry leaves pov autumn forest"),
    ("oc_pov_snow_walk", "pov walking in snow crunching footsteps"),
    ("oc_pov_gravel_path", "pov walking gravel path countryside"),
    ("oc_pov_wooden_boardwalk", "pov walking wooden boardwalk forest"),
    ("oc_pov_night_forest", "pov night walk forest flashlight sounds"),
    ("oc_pov_mountain_trail", "pov hiking mountain trail footsteps wind"),
    ("oc_pov_beach_walk", "pov walking on beach sand waves"),
    ("oc_pov_city_walk_rain", "pov walking rain umbrella city footsteps"),
    ("oc_pov_stairs_climb", "pov climbing stairs breathing footsteps"),
    ("oc_pov_bushcraft", "bushcraft pov walking woods twigs breaking"),
    ("oc_pov_hunting_stalk", "hunting stalk pov quiet forest footsteps"),
    # --- B: occluded by terrain / walls ---------------------------------------
    ("oc_stream_behind_trees", "forest walk stream sound hidden behind trees"),
    ("oc_waterfall_approach", "hiking towards waterfall sound before seeing it"),
    ("oc_river_through_forest", "river heard walking forest path"),
    ("oc_indoor_rain_window", "sitting indoors rain outside window sound"),
    ("oc_indoor_street_window", "apartment interior street sounds through window"),
    ("oc_neighbour_through_wall", "neighbours noise through wall apartment"),
    ("oc_around_the_corner", "walking around corner alley sounds ahead"),
    ("oc_cave_water_drip", "cave walk water dripping echo unseen"),
    ("oc_tunnel_sounds_ahead", "walking tunnel sounds ahead echo"),
    ("oc_behind_hedge_road", "country lane hedge cars passing unseen"),
    # --- C: distance ----------------------------------------------------------
    ("oc_distant_thunder_field", "distant thunder rolling countryside calm"),
    ("oc_distant_dogs_night", "village night distant dogs barking"),
    ("oc_distant_church_bells", "countryside distant church bells morning"),
    ("oc_distant_traffic_hill", "hilltop view distant traffic hum city"),
    ("oc_distant_train_valley", "valley distant train horn countryside"),
    ("oc_distant_fireworks", "distant fireworks heard from far neighbourhood"),
    ("oc_distant_construction", "distant construction noise residential street"),
    ("oc_distant_airplane_field", "open field distant airplane passing overhead"),
    ("oc_distant_sirens_city", "rooftop distant sirens city night"),
    # --- E: material / meaning carried only by sound ---------------------------
    ("oc_twig_snap_forest", "twig snapping forest something approaching"),
    ("oc_ice_cracking_lake", "walking frozen lake ice cracking sounds"),
    ("oc_creaking_floorboards", "old house creaking floorboards walking"),
    ("oc_wind_before_storm", "wind picking up before storm trees"),
    ("oc_rain_starting_roof", "rain starting on roof sound indoors"),
    ("oc_footsteps_behind", "footsteps behind you following night street"),
]


def _detect(wav: Path):
    from src.stage4_audio_event_detection import detect_events
    return detect_events(wav, threshold=0.05, min_dur=0.1)


def _pick(events):
    """Strongest narrative event, not right at the clip edge."""
    best = None
    for e in events:
        if e.label not in NARRATIVE or e.confidence < MIN_EVENT_CONF:
            continue
        if e.start < 4.0:
            continue
        if best is None or e.confidence > best.confidence:
            best = e
    return best


def _score_drowned(events, ev) -> bool:
    """A film score is expected; reject only if it buries the event entirely."""
    m = max((e.confidence for e in events if e.label in MUSIC_HINTS), default=0.0)
    return m >= 0.90 and m > ev.confidence * 2.0


def _is_bw(f: Path, at: float) -> bool:
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "f.jpg"
        subprocess.run(["ffmpeg", "-y", "-ss", f"{at:.1f}", "-i", str(f), "-frames:v", "1",
                        "-vf", "scale=160:-1", str(fp)], capture_output=True, timeout=60)
        if fp.exists():
            return statistics.mean(Image.open(fp).convert("HSV").split()[1].getdata()) < 14
    return False


def main(want: int = 30):
    STAGE.mkdir(parents=True, exist_ok=True)
    from src.stage1_audio_extraction import extract_audio
    got = 0
    qs = QUERIES[:]
    random.shuffle(qs)
    for name, query in qs:
        if got >= want:
            break
        raw = STAGE / f"{name}_raw.mp4"
        raw.unlink(missing_ok=True)
        print(f"  [mv] {name}: {query}", flush=True)
        for rank in (1, 2, 3):
            subprocess.run(
                ["yt-dlp", f"ytsearch{rank}:{query}", "--playlist-items", str(rank),
                 "--download-sections", f"*0-{WINDOW}",
                 "-f", "mp4[height<=720]/best[height<=720]/best",
                 "--force-keyframes-at-cuts",
                 "--match-filter", "duration>40 & duration<5400",
                 "--download-archive", str(ARCHIVE),
                 "--print-to-file", "%(webpage_url)s|%(title)s", str(STAGE / f"{name}.meta"),
                 "--no-warnings", "-o", str(raw)],
                capture_output=True, text=True, timeout=420)
            if raw.exists():
                break
        if not raw.exists():
            print("    skip (no download)")
            continue
        try:
            with tempfile.TemporaryDirectory() as td:
                media = extract_audio(raw, Path(td) / "a.wav", config.SAMPLE_RATE)
                events = _detect(Path(media.wav_path))
        except Exception as e:
            print(f"    skip (audio {type(e).__name__})")
            raw.unlink(missing_ok=True)
            continue
        ev = _pick(events)
        if ev is None:
            print("    reject: no clear narrative sound event")
            raw.unlink(missing_ok=True)
            continue
        if _score_drowned(events, ev):
            print("    reject: score buries the event")
            raw.unlink(missing_ok=True)
            continue
        if _is_bw(raw, ev.start):
            print("    reject: black & white")
            raw.unlink(missing_ok=True)
            continue
        length = random.randint(16, 20)
        start = max(0.0, ev.start - length * 0.45)   # keep the on-screen reaction after it
        cut = STAGE / f"{name}.mp4"
        subprocess.run(["ffmpeg", "-y", "-ss", f"{start:.1f}", "-i", str(raw),
                        "-t", str(length), "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "26", "-c:a", "aac", str(cut)],
                       capture_output=True, timeout=300)
        raw.unlink(missing_ok=True)
        if not cut.exists():
            continue
        meta_f = STAGE / f"{name}.meta"
        url, title = "", ""
        if meta_f.exists():
            parts = meta_f.read_text(encoding="utf-8").strip().split("|", 1)
            url = parts[0]
            title = parts[1] if len(parts) > 1 else ""
        print(f"    event: {ev.label} @{ev.start:.1f}s conf {ev.confidence:.2f}  |  {title[:50]}")
        if _accept(cut, name, {"source": "youtube(movie scene)", "url": url, "title": title,
                               "event_label": ev.label, "event_conf": round(ev.confidence, 3),
                               "license": "research use; not redistributed"}):
            got += 1
    print(f"\nmovie scenes accepted: {got}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
