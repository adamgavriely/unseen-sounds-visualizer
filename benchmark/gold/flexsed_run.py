"""FlexSED on the gold clips (amendment 7, 2026-09-22): an open-vocabulary detector as a second
opinion where BEATs is blind.

Why this one. At all nine onsets where BEATs scored the needed sound below 0.05, its top labels are
**Speech (0.58-0.81) or Music (0.48-0.58)** and the target is nowhere in the top six: the sound is
masked, not missing from the vocabulary. A model that is asked about one label at a time cannot be
out-voted that way, and FlexSED (Dasheng SSL encoder + CLAP text queries, trained on AudioSet-Strong,
PSDS1 0.448) is exactly that: a text-queried frame-level detector.

It is queried with the project's own depictable family vocabulary (benchmark/gold/depictable_vocab.json,
215 families) -- the labels a picture could be drawn of -- which is fixed in advance and never looks
at the gold labels. Frame probabilities are cached per clip; nothing is decided here.

    python benchmark/gold/flexsed_run.py --out data/work/flexsed_cache     # GPU, one pass
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

VOCAB = _ROOT / "benchmark" / "gold" / "depictable_vocab.json"
GOLD = _ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"
FLEXSED = Path(os.environ.get("FLEXSED_ROOT", Path.home() / "FlexSED"))
FPS = 25.0                      # FlexSED's frame rate (api.plot_and_save_multi uses sr=25)


def clips():
    d = json.loads(GOLD.read_text(encoding="utf-8"))
    return [c["clip"] for c in d["clips"] if isinstance(c, dict) and c.get("done") and not c.get("bad")]


def clip_path(name: str):
    for sub in ("mixed", "seen_ambient", "unseen_ambient", "no_ambient", "unsorted", "_dropped"):
        p = _ROOT / "data" / "input" / "benchmark" / sub / name
        if p.exists():
            return p
    for p in (_ROOT / "data" / "input" / "audioset_strong").glob(name + ".*"):
        return p
    return None


def wav_for(p: Path, work: Path) -> Path:
    work.mkdir(parents=True, exist_ok=True)
    w = work / (p.stem + ".wav")
    if not w.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(p), "-ac", "1", "-ar", "16000", str(w)], check=True)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_ROOT / "data" / "work" / "flexsed_cache"))
    ap.add_argument("--batch", type=int, default=64, help="queries per forward pass")
    a = ap.parse_args()
    sys.path.insert(0, str(FLEXSED))
    os.chdir(FLEXSED)                      # the model config path is relative to the repo
    from api import FlexSED
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))["families"]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    work = _ROOT / "data" / "work" / "gold_wav_flat"
    m = FlexSED(device="cuda")
    for i, name in enumerate(clips(), 1):
        stem = Path(name).stem
        f = out / f"{stem}.npz"
        if f.exists():
            continue
        p = clip_path(name)
        if p is None:
            print("missing", name, flush=True); continue
        wav = wav_for(p, work)
        parts = []
        for k in range(0, len(vocab), a.batch):
            preds = m.run_inference(str(wav), vocab[k:k + a.batch])     # [n_events, 1, T]
            parts.append(preds.squeeze(1).numpy().astype(np.float16))
        fw = np.concatenate(parts, axis=0)                              # [n_vocab, T]
        np.savez_compressed(f, fw=fw, labels=np.array(vocab), fps=FPS)
        print(f"[{i}] {stem} {fw.shape}", flush=True)
    print("done ->", out)


if __name__ == "__main__":
    main()
