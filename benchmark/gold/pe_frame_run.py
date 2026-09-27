"""PE-A-Frame frame scores (amendment 24, docs/prereg_v4.md): a text-queried frame-level detector whose encoder is
independent of FlexSED's (Dasheng + CLAP), cached exactly like FlexSED so stage-4 code reads it unchanged.

Model: facebook/pe-a-frame-large (Apache-2.0; PE-AV, arXiv 2512.19687), `transformers.PeAudioFrameLevelModel`.
Queries: the 215 depictable families (benchmark/gold/depictable_vocab.json), worded "The sound of {family}" -- FlexSED's
wording -- fixed in advance, never looking at gold. Audio: 48 kHz mono, scored in 30-s pieces; one frame per 40 ms
(hop 1920 = 25 fps). Score = sigmoid(model logit, with the model's own learned scale and bias). Output per clip:
npz fw [n_labels, T] float16, labels, fps -- the FlexSED cache layout. Nothing is decided here.

    python benchmark/gold/pe_frame_run.py --out data/work/pe_frame_cache                       # the gold clips
    python benchmark/gold/pe_frame_run.py --clip-dir data/input/audioset_calib --out benchmark/audioset_calib_windows/pe_frame
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

VOCAB = _ROOT / "benchmark" / "gold" / "depictable_vocab.json"
MODEL = "facebook/pe-a-frame-large"
SR, HOP, FPS, PIECE_S = 48_000, 1920, 25.0, 30.0


def load_48k(p: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(p), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_ROOT / "data" / "work" / "pe_frame_cache"))
    ap.add_argument("--clip-dir", default=None, help="score this folder of clips instead of the gold set")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N clips (smoke test)")
    a = ap.parse_args()
    import torch
    from transformers import AutoProcessor, PeAudioFrameLevelModel

    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))["families"]
    queries = [f"The sound of {f.lower()}" for f in vocab]
    if a.clip_dir:
        d = Path(a.clip_dir)
        allp = sorted(p for p in d.iterdir() if p.suffix.lower() in (".mp4", ".wav", ".mkv", ".webm", ".m4a"))
        paths = allp[a.shard::max(1, a.of)]
    else:
        from benchmark.gold.flexsed_run import clips, clip_path
        paths = [clip_path(n) for n in clips()[a.shard::max(1, a.of)]]
        paths = [p for p in paths if p]
    if a.limit:
        paths = paths[:a.limit]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(MODEL)
    model = PeAudioFrameLevelModel.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev).eval()
    with torch.inference_mode():
        tok = proc.tokenizer(queries, padding=True, return_tensors="pt").to(dev)
        text = model.get_text_audio_embeds(tok["input_ids"], tok.get("attention_mask"))          # [L, D]
        scale = model.text_audio_logit_scale.float(); bias = model.text_audio_logit_bias.float()
    print(f"[pe] {len(paths)} clips, {len(queries)} queries, device {dev}, logit scale {scale.item():.3f} bias {bias.item():.3f}", flush=True)
    for i, p in enumerate(paths, 1):
        dst = out / f"{p.stem}.npz"
        if dst.exists():
            continue
        wav = load_48k(p)
        piece = int(PIECE_S * SR)
        cols = []
        for s in range(0, max(1, len(wav)), piece):
            w = wav[s:s + piece]
            if len(w) < HOP:
                w = np.pad(w, (0, HOP - len(w)))
            feats = proc.feature_extractor(w, sampling_rate=SR, return_tensors="pt")
            pm = feats.get("padding_mask")
            with torch.inference_mode():
                aud = model.get_audio_embeds(feats["input_values"].to(dev), pm.to(dev) if pm is not None else None)
                logit = (aud[0].float() @ text.float().T) * scale + bias                        # [T, L]
            n = int(np.ceil(len(w) / HOP))
            cols.append(torch.sigmoid(logit)[:n].cpu().numpy())
        fw = np.concatenate(cols, axis=0).T.astype(np.float16)                                 # [L, T]
        np.savez_compressed(dst, fw=fw, labels=np.array(vocab), fps=FPS)
        if i % 20 == 0 or i == len(paths):
            print(f"[pe] {i}/{len(paths)} {p.stem} {fw.shape} max {float(fw.max()):.3f}", flush=True)
    print("done ->", out)


if __name__ == "__main__":
    main()
