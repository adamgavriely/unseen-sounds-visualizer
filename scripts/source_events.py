"""Source DISCRETE off-screen sound EVENTS (glass shatter, sudden bark, crash,
alarm) -- the strongest "heard but not seen" cases, unlike continuous ambience.

Strategy differs from scripts/source_batch3.py: we cannot guess when the event
happens, so we download a several-minute window, run PANNs over it to LOCATE an
impulsive target sound, then cut a 16-25 s clip positioned so the event lands
early-ish and the on-screen REACTION is included. Clips whose soundtrack is
dominated by music are rejected (Adam BAD-tagged music-over-video before).

Usage: python scripts/source_events.py [n_wanted]
"""
from __future__ import annotations

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
from scripts.source_batch3 import screen, _accept, STAGE   # reuse the quality screen

WINDOW = 240          # seconds of video to fetch and search for the event
ARCHIVE = STAGE / "yt_archive.txt"       # shared: never reuse a source video

# Impulsive, narratively-meaningful sounds a captioner would mark [glass shatters]
TARGET = {
    "Glass", "Shatter", "Breaking", "Smash, crash", "Crack",
    "Bark", "Bow-wow", "Growling", "Howl", "Meow",
    "Explosion", "Boom", "Bang", "Gunshot, gunfire", "Firecracker", "Fireworks",
    "Alarm", "Car alarm", "Fire alarm", "Smoke detector, smoke alarm",
    "Doorbell", "Ding-dong", "Knock", "Slam", "Thump, thud", "Clatter",
    "Screaming", "Shout", "Yell", "Squeal", "Skidding", "Tire squeal",
    "Vehicle horn, car horn, honking", "Siren", "Thunder", "Thunderstorm",
    "Splash, splatter", "Whip", "Sneeze",
}
MUSIC_HINTS = {"Music", "Background music", "Theme music", "Soundtrack music"}

QUERIES = [
    # something breaks / falls off-camera
    ("glass_drop_reaction", "glass breaking accident caught on camera reaction"),
    ("dish_smash_kitchen", "plates dropped kitchen crash reaction"),
    ("shelf_collapse", "shelf collapse store crash caught camera"),
    ("window_break_street", "window breaking street security camera"),
    ("vase_knocked_cat", "cat knocks over glass breaking"),
    # animals startle someone / bark off-screen
    ("dog_bark_startle", "dog barks suddenly scares person reaction"),
    ("dog_barks_doorbell", "doorbell rings dogs barking reaction home camera"),
    ("dog_bark_vlog", "vlog dog barking off camera interrupts"),
    ("rooster_surprise", "rooster crows surprise reaction video"),
    ("goose_chase", "goose honking chases person reaction"),
    # alarms / horns / sirens off-screen
    ("smoke_alarm_kitchen", "smoke alarm goes off cooking reaction"),
    ("car_alarm_street", "car alarm going off street video"),
    ("fire_alarm_office", "fire alarm goes off office reaction"),
    ("siren_interrupt", "ambulance siren passes interview street"),
    ("horn_startle", "car horn honks startles pedestrian"),
    # impacts / crashes heard off-camera
    ("crash_offscreen", "car crash sound caught on camera bystanders react"),
    ("dashcam_crash_sound", "dashcam hears crash reaction"),
    ("bodycam_noise", "bodycam loud bang officers react"),
    ("construction_bang", "construction loud bang workers react"),
    ("tree_fall_crash", "tree falls loud crash people watching"),
    # weather / sudden nature events
    ("thunder_startle", "thunder clap startles people reaction outdoor"),
    ("hail_sudden", "sudden hailstorm starts people running"),
    ("wave_surprise", "big wave surprises people on rocks"),
    ("avalanche_distant", "avalanche sound skiers reaction"),
    # celebration bangs
    ("firework_startle", "firework bang startles crowd reaction"),
    ("balloon_pop_reaction", "balloon pops startles reaction"),
    ("champagne_pop", "champagne cork pop celebration reaction"),
    ("confetti_cannon", "confetti cannon bang surprise reaction"),
    # doors / knocks / household
    ("door_slam_wind", "door slams shut wind reaction video"),
    ("knock_door_night", "knocking at door night camera reaction"),
    ("something_falls_night", "noise at night camera something falls"),
    # public spaces
    ("street_shout_offscreen", "street interview interrupted shouting off camera"),
    ("news_live_interrupted", "news reporter interrupted loud noise live"),
    ("classroom_bell", "school bell rings students react"),
    ("stadium_roar_offscreen", "fans react to goal noise outside stadium"),
    ("skate_bail_crash", "skateboard crash sound bystanders react"),
    # wave 2 -- camera types whose audio is raw (no added soundtrack)
    ("ring_doorbell_dog", "ring doorbell camera dog barking inside"),
    ("cctv_glass_break", "cctv footage glass break shop"),
    ("security_cam_crash", "security camera car crash sound street"),
    ("bodycam_dog_bark", "bodycam dog barking officer"),
    ("livestream_noise", "live stream interrupted loud noise streamer reacts"),
    ("baby_monitor_noise", "baby monitor dog barking wakes"),
    ("zoom_call_noise", "video call interrupted dog barking background"),
    ("news_live_bang", "live news broadcast loud bang startles anchor"),
    ("field_report_siren", "field report interrupted siren passing"),
    ("classroom_fire_drill", "fire alarm drill school students leave"),
    ("kitchen_pan_drop", "pan drops kitchen loud clatter"),
    ("garage_door_slam", "garage door loud slam camera"),
    ("thunder_dog_scared", "dog scared of thunder reaction home video"),
    ("cat_knocks_glass", "cat pushes glass off table breaks"),
    ("bird_window_hit", "bird hits window sound reaction"),
]


def _detect(wav: Path):
    from src.stage4_audio_event_detection import detect_events
    return detect_events(wav, threshold=0.05, min_dur=0.1)


def _pick_event(events):
    """Best impulsive target event: short, confident, not at the very start."""
    best = None
    for e in events:
        if e.label not in TARGET:
            continue
        if (e.end - e.start) > 4.0 or e.confidence < 0.15 or e.start < 3.0:
            continue
        if best is None or e.confidence > best.confidence:
            best = e
    return best


def _music_dominant(events, ev=None) -> bool:
    """Reject only when music really drowns the clip: loud music AND louder than
    the event itself. (0.55 with no comparison rejected 7/17 usable clips.)"""
    m = max((e.confidence for e in events if e.label in MUSIC_HINTS), default=0.0)
    if ev is None:
        return m >= 0.85
    return m >= 0.70 and m > ev.confidence * 1.3


def main(want: int = 15):
    STAGE.mkdir(parents=True, exist_ok=True)
    from src.stage1_audio_extraction import extract_audio
    got = 0
    queries = QUERIES[:]
    random.shuffle(queries)
    for name, query in queries:
        if got >= want:
            break
        raw = STAGE / f"ev_{name}_raw.mp4"
        raw.unlink(missing_ok=True)
        print(f"  [ev] {name}: {query}", flush=True)
        for rank in (1, 2, 3):          # a single search hit misses too often
            subprocess.run(
                ["yt-dlp", f"ytsearch{rank}:{query}", "--playlist-items", str(rank),
                 "--download-sections", f"*0-{WINDOW}",
                 "-f", "mp4[height<=720]/best[height<=720]/best",
                 "--force-keyframes-at-cuts",
                 "--match-filter", "duration>30 & duration<7200",
                 "--download-archive", str(ARCHIVE),
                 "--print-to-file", "%(webpage_url)s|%(title)s",
                 str(STAGE / f"ev_{name}.meta"),
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
            print(f"    skip (audio: {type(e).__name__})")
            raw.unlink(missing_ok=True)
            continue
        ev = _pick_event(events)
        if ev is None:
            print("    reject: no discrete target event found")
            raw.unlink(missing_ok=True)
            continue
        if _music_dominant(events, ev):
            print("    reject: music drowns the event")
            raw.unlink(missing_ok=True)
            continue
        # place the event ~40% in, so the on-screen reaction after it is captured
        length = random.randint(16, 20)
        start = max(0.0, ev.start - length * 0.4)
        cut = STAGE / f"ev_{name}.mp4"
        subprocess.run(["ffmpeg", "-y", "-ss", f"{start:.1f}", "-i", str(raw),
                        "-t", str(length), "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "26", "-c:a", "aac", str(cut)],
                       capture_output=True, timeout=300)
        raw.unlink(missing_ok=True)
        if not cut.exists():
            continue
        meta_f = STAGE / f"ev_{name}.meta"
        url, title = "", ""
        if meta_f.exists():
            parts = meta_f.read_text(encoding="utf-8").strip().split("|", 1)
            url = parts[0]
            title = parts[1] if len(parts) > 1 else ""
        print(f"    event: {ev.label} @{ev.start:.1f}s conf {ev.confidence:.2f}")
        if _accept(cut, f"ev_{name}", {"source": "youtube", "url": url, "title": title,
                                       "event_label": ev.label,
                                       "event_conf": round(ev.confidence, 3),
                                       "license": "research use; not redistributed"}):
            got += 1
    print(f"\nevent clips accepted: {got}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
