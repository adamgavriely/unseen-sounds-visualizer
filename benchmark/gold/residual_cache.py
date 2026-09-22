"""BEATs over the speech-removed residual (the completeness test Adam approved, 2026-09-23).

The masking diagnosis was that at every onset where BEATs scored a needed sound below 0.05, its own
top labels were Speech 0.58-0.81 or Music 0.48-0.58. Removing the speech and tagging what is left
tests that directly:

    residual = mix - DeepFilterNet3(mix)              (benchmark/gold/speech_residual.py)

The residual is tagged as an extra VIEW and OR-ed into the union, so it can only add labels. The
go/no-go was written before the run: adopt only at onset-recall >= 0.62 with <= 2.0 false labels per
clip and no rise on the quiet clips.

    python benchmark/gold/residual_cache.py            # GPU
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config

RES = _ROOT / "data" / "work" / "residual_wav"
OUT = _ROOT / "benchmark" / "gold" / "beats_res_fw"


def main():
    from src.stage4_audio_event_detection.beats_infer import infer_beats
    OUT.mkdir(parents=True, exist_ok=True)
    wavs = sorted(RES.glob("*/audio.wav"))
    print(f"[residual] {len(wavs)} residual clips", flush=True)
    for i, w in enumerate(wavs, 1):
        stem = w.parent.name
        f = OUT / f"{stem}.npz"
        if f.exists():
            continue
        fw, times, labels = infer_beats(w, "cuda")
        np.savez_compressed(f, fw=np.asarray(fw, dtype=np.float32),
                            times=np.asarray(times, dtype=np.float64), labels=np.array(labels))
        print(f"[{i}] {stem} {np.asarray(fw).shape}", flush=True)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
