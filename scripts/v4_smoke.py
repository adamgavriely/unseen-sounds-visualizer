"""Smoke test of the v4 context stages on one clip: SAM 3 (stage 2) and Granite Speech 4.1
(stage 3). Prints what each returns and its peak VRAM; a failure in one does not stop the other.

    python scripts/v4_smoke.py --clip data/input/benchmark/mixed/ly_ambulance_(siren)_-yPSgCn.mp4
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config
config.DEVICE = "cuda"


def vram():
    import torch
    return f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak"


def part_sam3(video: Path, td: Path):
    from src.stage2_video_understanding.sam3 import analyze_video_sam3
    t0 = time.time()
    sc = analyze_video_sam3(video, num_frames=4, device="cuda", threshold=config.SAM3_THRESHOLD)
    print(f"[sam3] {time.time() - t0:.0f}s {vram()} visible={sc.visible_entities} scores={sc.raw.get('scores')} err={sc.raw.get('error')}")


def part_granite(video: Path, td: Path):
    from src.stage1_audio_extraction import extract_audio
    from src.stage3_speech_recognition import transcribe
    media = extract_audio(video, td / "audio.wav", config.SAMPLE_RATE)
    t0 = time.time()
    segs = transcribe(Path(media.wav_path), model_size="ibm-granite/granite-speech-4.1-2b", device="cuda")
    print(f"[granite] {time.time() - t0:.0f}s {vram()} {len(segs)} segment(s)")
    for s in segs:
        print(f"[granite]   {s.start:5.1f}-{s.end:5.1f}  {s.text[:100]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["sam3", "granite"])
    ap.add_argument("--clip", default=str(_ROOT / "data" / "input" / "benchmark" / "mixed" / "ly_ambulance_(siren)_-yPSgCn.mp4"))
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        for p in a.parts:
            print(f"=== {p} ===", flush=True)
            try:
                {"sam3": part_sam3, "granite": part_granite}[p](Path(a.clip), Path(td))
            except Exception:
                print(f"[{p}] FAILED:\n" + traceback.format_exc()[-1500:], flush=True)
            import torch, gc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
