"""Round 15 amendment O (docs/prereg_round13_detector_push.md): Whisper-AT frame tags as F8's third vote.
Whisper-AT (pip whisper-at 0.5, official large-v2 checkpoint + AudioSet tagging head) on each clip's 16-kHz audio at
at_time_res 0.4 s. Saved in the F8 format: fw [T, 527] = 1 / rank of the class at that step (1 = top class), times = step
start, labels = AudioSet names; raw logits kept as `logits`. Runs in ~/venvs/wat (msproj + whisper-at). No gold is read.

    python benchmark/gold/wat_cache.py --split dev|dev2|test2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
WORK = _ROOT / "data" / "work"
WAV = {"dev": WORK / "devcand" / "wav16", "dev2": WORK / "r13dev2" / "wav16", "test2": WORK / "r13test2" / "wav16"}
OUT = WORK / "wat_cache"
RES = 0.4


def main():
    import torch
    import whisper_at as w
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=sorted(WAV))
    a = ap.parse_args()
    labels = json.load(open(os.path.join(os.path.dirname(w.__file__), "assets", "label_name_dict.json")))["en"]
    assert len(labels) == 527
    out = OUT / a.split
    out.mkdir(parents=True, exist_ok=True)
    model = w.load_model("large-v2", device="cuda")
    for wp in sorted(WAV[a.split].glob("*.wav")):
        dst = out / f"{wp.stem}.npz"
        if dst.exists():
            continue
        audio = w.load_audio(str(wp))
        res = model.transcribe(audio, at_time_res=RES, fp16=True)
        lg = res["audio_tag"]
        lg = lg.float().cpu().numpy() if torch.is_tensor(lg) else np.asarray(lg, np.float32)
        order = np.argsort(-lg, axis=1)
        rank = np.empty_like(order)
        rows = np.arange(lg.shape[0])[:, None]
        rank[rows, order] = np.arange(1, lg.shape[1] + 1)[None, :]
        fw = (1.0 / rank).astype(np.float32)
        times = np.arange(lg.shape[0]) * RES
        np.savez_compressed(dst, fw=fw, times=times.astype(np.float64), labels=np.array(labels), logits=lg.astype(np.float32))
        top = [labels[i] for i in order[:, 0]]
        print(f"[wat] {a.split} {wp.stem}: {lg.shape[0]} steps ({len(audio) / 16000:.1f} s); top1 {sorted(set(top))[:6]}", flush=True)


if __name__ == "__main__":
    main()
