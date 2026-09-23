"""Compute FlexSED scores for ONE wav and write them to the cache the pipeline already reads.

Why a subprocess. FlexSED ships its own top-level `src` package and its model config path is
relative to its repository, so importing it requires removing this project from `sys.path` and
chdir-ing into the repo (see benchmark/gold/flexsed_run.py). Both are destructive to a long-running
process, which is exactly what a ComfyUI server is. Running it as a short-lived subprocess keeps
that surgery contained, and the parent only ever sees a .npz appear.

This is what lets the demo accept a video it has never seen: everything else in stage 4 works from
the audio directly, but FlexSED's scores were pre-computed for the benchmark, so a new clip has no
cache entry until this runs.

    python comfyui_nodes/flexsed_worker.py --wav /path/audio.wav --stem my_clip
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
VOCAB = _ROOT / "benchmark" / "gold" / "depictable_vocab.json"
FLEXSED = Path(os.environ.get("FLEXSED_ROOT", Path.home() / "FlexSED"))
FPS = 25.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=24)
    a = ap.parse_args()
    out = Path(a.out) if a.out else (_ROOT / "data" / "work" / "flexsed_cache")
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{a.stem}.npz"
    if dst.exists():
        print(f"[flexsed] {a.stem}: cached already")
        return

    queries = json.loads(VOCAB.read_text(encoding="utf-8"))["families"]
    # the project root must leave sys.path before FlexSED is imported: our regular `src` package
    # always beats its namespace package, whatever the path order
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _ROOT]
    sys.modules.pop("src", None)
    local_clap = Path.home() / "clap-htsat-unfused-st"
    if local_clap.exists():
        # transformers >= 5 refuses torch.load on torch < 2.6 and laion/clap-htsat-unfused ships
        # only a .bin, so a locally converted safetensors copy is pointed at instead
        import transformers
        _clap, _tok = transformers.ClapTextModelWithProjection, transformers.AutoTokenizer
        _from_clap, _from_tok = _clap.from_pretrained, _tok.from_pretrained
        _clap.from_pretrained = staticmethod(lambda n, *x, **k: _from_clap(str(local_clap) if "clap" in str(n) else n, *x, **k))
        _tok.from_pretrained = staticmethod(lambda n, *x, **k: _from_tok(str(local_clap) if "clap" in str(n) else n, *x, **k))
    sys.path.insert(0, str(FLEXSED))
    os.chdir(FLEXSED)
    import torch
    from api import FlexSED
    m = FlexSED(device="cuda" if torch.cuda.is_available() else "cpu")

    # a remainder shorter than a second breaks the mel front-end's padding; zero-pad it
    _split = m.split_audio_fixed

    def split(audio, sr, chunk_duration=10.0):
        parts = []
        for c in _split(audio, sr, chunk_duration):
            if len(c) < sr:
                c = np.pad(c, (0, sr - len(c)))
            parts.append(c)
        return parts

    m.split_audio_fixed = split

    chunks = []
    for k in range(0, len(queries), a.batch):
        with torch.inference_mode():
            preds = m.run_inference(a.wav, queries[k:k + a.batch])
        chunks.append(preds.detach().squeeze(1).numpy().astype(np.float16))
        del preds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    fw = np.concatenate(chunks, axis=0)
    np.savez_compressed(dst, fw=fw, labels=np.array(queries), fps=FPS)
    print(f"[flexsed] {a.stem}: {fw.shape} -> {dst}")


if __name__ == "__main__":
    main()
