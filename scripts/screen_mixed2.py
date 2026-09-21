"""VLM screen for the wave-2 mixed candidates (2026-09-21 night, Adam: "I checked 5 and it
feels not mixed. Use the VLM to check which ones actually are candidates for mixed").

Runs the real gate of the v4b3 row (stages 1, 2, 4, 5 + the Qwen3.8-27B per-sound visibility
vote; no speech, no pictures) on every video in a folder and sorts each drawable sound of
importance >= 2 into
    unseen  -- the gate would draw it (source not visible),
    seen    -- the gate held it back because the source is on screen.
Verdict per clip: "mixed" (both lists non-empty), "unseen", "seen" or "empty".

A pre-screen only: it finds videos that MIGHT contain two or more sounds, some seen and some
unseen, so Adam looks at those first. Every tag is Adam's.

Usage (BIU, GPU):  python scripts/screen_mixed2.py data/input/benchmark/unsorted 'm2_*.mp4'
Output:            benchmark/gold/mixed2_vlm.json  {stem: {verdict, unseen: [...], seen: [...]}}
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
config.use_v4("59")          # the v4b3 gate: BEATs, depictable filter, Qwen3.8-27B visibility
config.DEVICE = "cuda"
config.TRANSCRIBE = False

from benchmark.gold.build_tool import importance_of
from src.labels import canonical
from src.stage1_audio_extraction import extract_audio
from src.stage2_video_understanding import analyze
from src.stage4_audio_event_detection import detect_events
from src.stage5_cross_modal_analysis import plan_augmentations, reason

OUT = ROOT / "benchmark" / "gold" / "mixed2_vlm.json"


def screen(video: Path, work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    media = extract_audio(video, work / "audio.wav", config.SAMPLE_RATE)
    scene = analyze(video, backend=config.VIDEO_BACKEND, num_frames=config.NUM_FRAMES, model=config.VIDEO_MODEL,
                    vlm_model=config.VLM_MODEL, siglip_model=config.SIGLIP_MODEL, owl_model=config.OWL_MODEL,
                    owl_threshold=config.OWL_THRESHOLD, sam3_model=getattr(config, "SAM3_MODEL", "facebook/sam3"),
                    sam3_threshold=getattr(config, "SAM3_THRESHOLD", 0.5), siglip_threshold=config.SIGLIP_THRESHOLD,
                    device=config.DEVICE, threshold=config.VISIBILITY_THRESHOLD)
    events = detect_events(Path(media.wav_path), threshold=config.AED_THRESHOLD, min_dur=config.AED_MIN_DUR,
                           model=config.AED_MODEL, device=config.DEVICE)
    specs = plan_augmentations(scene, [], events, threshold=config.AED_THRESHOLD, gate_enabled=config.GATE_ENABLED,
                               display_threshold=config.DISPLAY_THRESHOLD, augment_threshold=config.AUGMENT_THRESHOLD)
    if any(s.augment for s in specs):
        reason.decide_subjects(video, specs, segments=[], model=config.VLM_MODEL, device=config.DEVICE,
                               display_threshold=config.DISPLAY_THRESHOLD)
    unseen, seen = [], []
    for s in specs:
        if s.confidence < config.DISPLAY_THRESHOLD:
            continue
        fam = canonical(s.event_label)
        if importance_of(s.event_label.lower(), fam) < 2:
            continue
        row = {"label": s.event_label, "family": fam, "conf": round(float(s.confidence), 2),
               "start": round(s.start, 1), "end": round(s.end, 1), "reason": s.reason[:160]}
        if s.augment:
            unseen.append(row)
        elif "visible" in s.reason and not s.reason.startswith("below display threshold"):
            seen.append(row)
    verdict = "mixed" if unseen and seen else "unseen" if unseen else "seen" if seen else "empty"
    return {"verdict": verdict, "unseen": unseen, "seen": seen, "visible_entities": list(scene.visible_entities)[:20]}


def main(folder: str, pattern: str = "m2_*.mp4") -> None:
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    videos = sorted(Path(folder).glob(pattern))
    print(f"[screen] {len(videos)} clips, {sum(v.stem in out for v in videos)} already done", flush=True)
    for i, v in enumerate(videos, 1):
        if v.stem in out:
            continue
        try:
            out[v.stem] = screen(v, ROOT / "data" / "work" / "mixed2_screen" / v.stem)
        except Exception as e:
            out[v.stem] = {"verdict": "error", "error": f"{type(e).__name__}: {e}"[:300], "unseen": [], "seen": []}
        OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        r = out[v.stem]
        print(f"  {i}/{len(videos)} {v.stem}: {r['verdict']}  unseen={[x['family'] for x in r['unseen']]}  "
              f"seen={[x['family'] for x in r['seen']]}", flush=True)
    from collections import Counter
    print("[screen] verdicts:", dict(Counter(r["verdict"] for r in out.values())), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "input" / "benchmark" / "unsorted"),
         sys.argv[2] if len(sys.argv) > 2 else "m2_*.mp4")
