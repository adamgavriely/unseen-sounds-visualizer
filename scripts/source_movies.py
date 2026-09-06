"""Source MOVIE/TV SCENES where an off-screen sound carries narrative meaning.

Rationale (Adam, 2026-08-22): scraped ambience walks are weak benchmark cases --
a city soundscape is diffuse and a DHH viewer loses little. The cases where a DHH
viewer genuinely BENEFITS are authored ones: film sound design deliberately puts
meaningful sound off-screen (a siren approaching a crime scene, an explosion behind
the characters, a scream down the corridor) and shows the reaction on screen. That
is exactly "heard but not seen", and it is what captioners bracket as [siren wails].

Differences from source_events.py:
  * queries target scripted scenes, not reaction/CCTV footage;
  * the target list is NARRATIVE sounds (siren, explosion, gunshot, scream, glass,
    helicopter, alarm), not household clatter;
  * a musical score is tolerated -- it is intrinsic to the medium and Stage 5 treats
    music as non-salient anyway -- provided the target event is clearly present;
  * a higher event-confidence bar (0.25), because we want unambiguous events.

Usage: python scripts/source_movies.py [n_wanted]
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

WINDOW = 300           # seconds of the scene to fetch and search
ARCHIVE = STAGE / "yt_archive.txt"
MIN_EVENT_CONF = 0.25

# Sounds a film puts off-screen ON PURPOSE, and that a captioner would bracket.
NARRATIVE = {
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
    # crime / police -- sirens approaching from off-screen (Adam's example)
    ("mv_crime_scene_sirens", "crime scene police sirens arriving movie scene"),
    ("mv_police_raid_scene", "police raid apartment scene sirens"),
    ("mv_arrest_street_scene", "arrest street scene police cars movie"),
    ("mv_getaway_chase_sirens", "getaway car chase sirens police movie scene"),
    ("mv_detective_crime_scene", "detective arrives crime scene movie"),
    ("mv_hostage_standoff", "hostage standoff police negotiation scene"),
    ("mv_prison_alarm_scene", "prison break alarm scene movie"),
    ("mv_bank_robbery_alarm", "bank robbery alarm scene movie"),
    # war / explosions off-screen
    ("mv_war_bombardment", "war movie scene shelling bombardment soldiers"),
    ("mv_trench_scene", "trench warfare scene explosions"),
    ("mv_air_raid_scene", "air raid siren scene movie civilians"),
    ("mv_battle_aftermath", "battle aftermath scene movie soldiers"),
    ("mv_helicopter_arrival", "helicopter arrives scene movie soldiers"),
    ("mv_bomb_defusal_scene", "bomb defusal tense scene movie"),
    ("mv_evacuation_scene", "evacuation scene movie explosions distant"),
    # thriller / horror -- off-screen threat
    ("mv_horror_door_creak", "horror movie scene door creaking footsteps"),
    ("mv_horror_scream_offscreen", "horror movie scream off screen reaction"),
    ("mv_haunted_house_noises", "haunted house strange noises scene"),
    ("mv_stalker_footsteps", "thriller footsteps approaching scene tension"),
    ("mv_basement_scene_noise", "basement scene strange noise movie"),
    ("mv_attic_noise_scene", "attic noise scene horror movie"),
    ("mv_phone_ringing_tense", "tense scene phone ringing movie"),
    ("mv_glass_break_intruder", "intruder breaks window scene movie"),
    # disaster / emergency
    ("mv_earthquake_scene", "earthquake scene movie building shaking"),
    ("mv_fire_alarm_evacuate", "fire alarm building evacuation scene movie"),
    ("mv_storm_scene_house", "storm scene movie house thunder"),
    ("mv_tornado_scene", "tornado approaching scene movie"),
    ("mv_shipwreck_scene", "ship sinking scene movie alarms"),
    ("mv_plane_emergency", "plane emergency scene movie alarms"),
    ("mv_hospital_emergency", "hospital emergency room scene monitors"),
    # drama with meaningful offscreen ambience
    ("mv_courtroom_outside", "courtroom scene crowd outside movie"),
    ("mv_funeral_rain_scene", "funeral scene rain movie"),
    ("mv_train_station_farewell", "train station farewell scene movie"),
    ("mv_apartment_street_noise", "apartment window city street noise scene movie"),
    ("mv_diner_scene_ambience", "diner scene movie ambience conversation"),
    ("mv_school_corridor_bell", "school corridor scene bell students movie"),
    ("mv_protest_scene_movie", "protest riot scene movie crowd"),
    ("mv_market_chase_scene", "market chase scene movie crowd"),
    # tv / series scenes
    ("tv_police_procedural", "police procedural scene sirens series"),
    ("tv_medical_drama_emergency", "medical drama emergency scene series"),
    ("tv_war_series_scene", "war series battle scene"),
    ("tv_crime_series_stakeout", "crime series stakeout scene street"),
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
        length = random.randint(16, 25)
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
