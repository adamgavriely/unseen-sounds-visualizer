"""Speech removal before tagging (the last detector idea, 2026-09-22; Adam: "run it when free just
for completeness").

Both reviewers ranked it last and capped its value at about two sounds, because the masking that
hides those sounds is a property of BEATs' clip-level tagging, not of the waveform — FlexSED, asked
one label at a time, already recovers seven of the nine. The test is run anyway, on the declared
terms: the SPEECH is removed (not the music — HTDemucs puts ambient sound in the same stem as the
music, which is why the earlier "views" attempt was uninformative), by inverting a speech enhancer:

    residual = mix - DeepFilterNet3(mix)

and the residual is tagged by both detectors as an extra *view*, OR-ed into the union, so it can only
add labels. Go/no-go, written before the run: adopt only at onset-recall >= 0.62 with <= 2.0 false
labels per clip and no rise on the quiet clips.

    python benchmark/gold/speech_residual.py --out data/work/residual_wav       # GPU
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_ROOT / "data" / "work" / "residual_wav"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()
    import soundfile as sf
    from df.enhance import enhance, init_df, load_audio
    from benchmark.gold.flexsed_run import clips, clip_path, wav_for
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    work = _ROOT / "data" / "work" / "gold_wav_flat"
    model, df_state, _ = init_df()
    sr = df_state.sr()
    for i, name in enumerate(clips()[a.shard::max(1, a.of)], 1):
        stem = Path(name).stem
        # the residual keeps the pipeline's layout, so the detectors read it like any other clip
        dst = out / stem / "audio.wav"
        if dst.exists():
            continue
        p = clip_path(name)
        if p is None:
            print("missing", name, flush=True); continue
        wav = wav_for(p, work)
        audio, _ = load_audio(str(wav), sr=sr)
        speech = enhance(model, df_state, audio)
        res = (audio - speech).squeeze().cpu().numpy() if hasattr(speech, "cpu") else np.asarray(audio - speech).squeeze()
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dst), res.astype(np.float32), sr)
        print(f"[{i}] {stem} {res.shape[-1] / sr:.1f}s", flush=True)
    print("done ->", out)


if __name__ == "__main__":
    main()
