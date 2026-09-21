"""VLM-only pre-screen of the wave-2 mixed candidates (2026-09-21 night, Adam: "use the VLM
without the gate, so it is not corrupted; just find candidates").

Nothing of the pipeline's gate runs here: no stage-5 rule, no visibility vote, no cascade.
Per clip: 8 frames spread over the clip + the plain list of sounds BEATs heard (label, time,
confidence >= 0.25, speech and music left out). One question to the VLM: for each listed sound,
is its source visible in the frames (yes / no / unsure)? Sounds of importance 1 (traffic hum,
wind, rain, crowd noise...) are ignored for the verdict.
Verdict: "mixed" = at least one sound visible and one not visible (both importance >= 2);
"unseen" = only not-visible; "seen" = only visible; "empty" = nothing of importance >= 2.

A pre-screen only: it finds videos that MIGHT contain two or more sounds, some seen and some
unseen, so Adam looks at those first. Every tag is Adam's.

Usage (BIU, GPU):  python scripts/screen_mixed2.py data/input/benchmark/unsorted 'm2_*.mp4'
Output:            benchmark/gold/mixed2_vlm.json  {stem: {verdict, unseen: [...], seen: [...], unsure: [...]}}
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from benchmark.gold.build_tool import importance_of
from src.labels import canonical, is_music, SPEECH_LABELS, is_salient_nonspeech
config.LABEL_FILTER = "depictable"     # the label list only (music, speech, silence, environment out); not the gate
from src.stage1_audio_extraction import extract_audio
from src.stage2_video_understanding import _sample_frames
from src.stage4_audio_event_detection import detect_events
from src.stage5_cross_modal_analysis.reason import _load, _ask, unload

OUT = ROOT / "benchmark" / "gold" / "mixed2_vlm.json"
VLM = "Qwen/Qwen3.8-27B"
AUDIO_BAR = 0.25
FRAMES = 8


def heard(video: Path) -> list[dict]:
    """what BEATs heard, one row per label (best span), speech/music out"""
    with tempfile.TemporaryDirectory() as td:
        media = extract_audio(video, Path(td) / "a.wav", config.SAMPLE_RATE)
        events = detect_events(Path(media.wav_path), threshold=config.AED_THRESHOLD, min_dur=config.AED_MIN_DUR,
                               model="beats", device="cuda")
    best: dict[str, dict] = {}
    for e in events:
        if e.confidence < AUDIO_BAR or e.label in SPEECH_LABELS or is_music(e.label) or e.label in ("Speech", "Music")                 or not is_salient_nonspeech(e.label):
            continue
        fam = canonical(e.label)
        imp = importance_of(e.label.lower(), fam)
        row = best.get(e.label)
        if row is None or e.confidence > row["conf"]:
            best[e.label] = {"label": e.label, "family": fam, "conf": round(float(e.confidence), 2),
                             "start": round(e.start, 1), "end": round(e.end, 1), "importance": imp}
    rows = sorted(best.values(), key=lambda r: -r["conf"])
    return rows[:10]


PROMPT = ("These {n} frames are taken in order from a short video. The soundtrack of the video contains "
          "these sounds (found by an audio model; some may be wrong):\n{sounds}\n"
          "For EACH numbered sound, decide whether the thing that makes it is visible in the frames "
          "(the source is on screen while it sounds), or not visible (the source is off screen, out of frame, "
          "or hidden), or unsure. Answer with one line per sound, exactly in the form\n"
          "1: visible | not visible | unsure\n"
          "and nothing else.")


def ask_vlm(video: Path, sounds: list[dict], mdl, proc) -> tuple[dict[int, str], str]:
    frames = _sample_frames(video, FRAMES)
    listing = "\n".join(f"{i + 1}. {s['label'].lower()} at {s['start']:.0f}-{s['end']:.0f} s" for i, s in enumerate(sounds))
    text = _ask(mdl, proc, PROMPT.format(n=len(frames), sounds=listing), images=frames, max_new=12 * len(sounds) + 8)
    ans: dict[int, str] = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s*[:.)-]\s*(.+)", line)
        if not m:
            continue
        v = m.group(2).strip().lower()
        ans[int(m.group(1))] = "not visible" if "not" in v else "unsure" if "unsure" in v else "visible" if "visible" in v else "unsure"
    return ans, text


def screen(video: Path, mdl, proc) -> dict:
    sounds = heard(video)
    out = {"verdict": "empty", "unseen": [], "seen": [], "unsure": [], "raw": ""}
    if not sounds:
        return out
    ans, out["raw"] = ask_vlm(video, sounds, mdl, proc)
    for i, s in enumerate(sounds, 1):
        s = dict(s, vlm=ans.get(i, "unsure"))
        {"visible": out["seen"], "not visible": out["unseen"]}.get(s["vlm"], out["unsure"]).append(s)
    big = lambda rows: [r for r in rows if r["importance"] >= 2]
    u, v = big(out["unseen"]), big(out["seen"])
    out["verdict"] = "mixed" if u and v else "unseen" if u else "seen" if v else "empty"
    return out


def main(folder: str, pattern: str = "m2_*.mp4") -> None:
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    videos = sorted(Path(folder).glob(pattern))
    todo = [v for v in videos if v.stem not in out]
    print(f"[screen] {len(videos)} clips, {len(videos) - len(todo)} already done", flush=True)
    if not todo:
        return
    mdl, proc = _load(VLM, "cuda")
    try:
        for i, v in enumerate(todo, 1):
            try:
                out[v.stem] = screen(v, mdl, proc)
            except Exception as e:
                out[v.stem] = {"verdict": "error", "error": f"{type(e).__name__}: {e}"[:300], "unseen": [], "seen": [], "unsure": []}
            OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
            r = out[v.stem]
            print(f"  {i}/{len(todo)} {v.stem}: {r['verdict']}  unseen={[x['label'] for x in r['unseen']]}  "
                  f"seen={[x['label'] for x in r['seen']]}  unsure={[x['label'] for x in r['unsure']]}", flush=True)
    finally:
        unload()
    from collections import Counter
    print("[screen] verdicts:", dict(Counter(r["verdict"] for r in out.values())), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "input" / "benchmark" / "unsorted"),
         sys.argv[2] if len(sys.argv) > 2 else "m2_*.mp4")
