"""More independent opinions for stage 4 (2026-09-23).

The PANNs veto worked because PANNs is a genuinely different model from FlexSED -- different
architecture, different training -- so the two fail in different places. That argument does not stop
at one extra model. This caches two further AudioSet taggers over the gold clips so an n-of-m
agreement rule can be tested:

  AST   MIT/ast-finetuned-audioset-10-10-0.4593   spectrogram transformer, AudioSet mAP ~0.459
  CED   mispeech/ced-base                          consistent ensemble distillation, mAP ~0.496

Both emit the same AudioSet 527-label space as PANNs and BEATs, so their scores are directly
comparable per canonical family. Neither is frame-level; for a VETO that does not matter, because
the question asked is only "does this model hear this family anywhere in the clip".

    python benchmark/gold/extra_taggers.py --model ast   [--shard i --of n]
    python benchmark/gold/extra_taggers.py --model ced
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

WIN = 10.0          # seconds per window; both models are trained on ~10 s
HOP = 5.0


def windows(y, sr):
    n = int(WIN * sr); h = int(HOP * sr)
    if len(y) <= n:
        return [np.pad(y, (0, max(0, n - len(y))))]
    return [y[i:i + n] for i in range(0, len(y) - n + 1, h)] or [y[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ast", "ced"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()
    import librosa
    import torch
    from benchmark.gold.detector_dry import clip_path, wav_for, gold_clips
    out = _ROOT / "benchmark" / "gold" / f"{a.model}_fw"
    out.mkdir(parents=True, exist_ok=True)
    if a.model == "ast":
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        fe = ASTFeatureExtractor.from_pretrained(name)
        mdl = ASTForAudioClassification.from_pretrained(name).to("cuda").eval()
        labels = [mdl.config.id2label[i] for i in range(mdl.config.num_labels)]
        sr = 16000
    else:
        from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
        name = "mispeech/ced-base"
        fe = AutoFeatureExtractor.from_pretrained(name, trust_remote_code=True)
        mdl = AutoModelForAudioClassification.from_pretrained(name, trust_remote_code=True).to("cuda").eval()
        labels = [mdl.config.id2label[i] for i in range(mdl.config.num_labels)]
        sr = 16000
    names = gold_clips()[a.shard::max(1, a.of)]
    print(f"[{a.model}] {len(names)} clips, {len(labels)} labels", flush=True)
    for i, nm in enumerate(names, 1):
        stem = Path(nm).stem
        f = out / f"{stem}.npz"
        if f.exists():
            continue
        p = clip_path(nm)
        if p is None:
            print("missing", nm, flush=True); continue
        y, _ = librosa.load(str(wav_for(p)), sr=sr, mono=True)
        rows = []
        for w in windows(y, sr):
            inp = fe(w, sampling_rate=sr, return_tensors="pt")
            inp = {k: v.to("cuda") for k, v in inp.items()}
            with torch.inference_mode():
                logits = mdl(**inp).logits[0]
            rows.append(torch.sigmoid(logits).float().cpu().numpy())
        fw = np.stack(rows)                         # [windows, 527]
        np.savez_compressed(f, fw=fw.astype(np.float32),
                            times=np.arange(len(rows), dtype=np.float64) * HOP,
                            labels=np.array(labels))
        print(f"[{i}] {stem} {fw.shape}", flush=True)
    print("done ->", out)


if __name__ == "__main__":
    main()
