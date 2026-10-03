"""SSLAM (ICLR 2025; ta012/SSLAM_AS2M_Finetuned, MIT) as a long-window tagger: a ViT-B trained
on audio MIXTURES, fine-tuned on AudioSet-2M (527 classes, mAP 50.2). Used in the last
detector attempt (docs/history/preregistrations/prereg_detector_v5.md, release v1.2.0) only to confirm long candidates that PSED
proposed; it never adds sounds. Scores per 10-s window, hop HOP; times = window centres.

    from src.stage4_audio_event_detection.sslam_infer import cache_clips
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

MODEL_ID = "ta012/SSLAM_AS2M_Finetuned"
SR = 16000
WIN = 10.0
HOP = 2.0
TARGET = 1024                # mel frames for 10 s
NORM_MEAN, NORM_STD = -4.268, 4.569

_MODEL = None


def load_model(device="cuda"):
    global _MODEL
    if _MODEL is None:
        from transformers import AutoModel
        _MODEL = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).eval().to(device)
    return _MODEL


LABELS_CSV = Path.home() / "SSLAM" / "SSLAM_Inference" / "inference" / "labels.csv"   # index, mid, display name (AudioSet order)


def labels():
    import csv
    rows = list(csv.reader(open(LABELS_CSV, encoding="utf-8")))
    return [r[2] for r in rows]


def mel(wav: np.ndarray, device):
    import torch, torchaudio
    x = torch.from_numpy(wav).float().to(device)
    x = x - x.mean()
    m = torchaudio.compliance.kaldi.fbank(x.unsqueeze(0), htk_compat=True, sample_frequency=SR, use_energy=False,
                                          window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10).unsqueeze(0)
    n = m.shape[1]
    if n < TARGET:
        m = torch.nn.ZeroPad2d((0, 0, 0, TARGET - n))(m)
    else:
        m = m[:, :TARGET, :]
    m = (m - NORM_MEAN) / (NORM_STD * 2)
    return m.unsqueeze(0)      # [1, 1, T, F]


def score_file(model, wav_path: Path, device="cuda"):
    import torch, librosa
    audio, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    n = len(audio); win = int(WIN * SR); hop = int(HOP * SR)
    starts = list(range(0, max(1, n - win // 2), hop)) or [0]
    fws, times = [], []
    with torch.no_grad():
        for s in starts:
            chunk = audio[s:s + win]
            if len(chunk) < win:
                chunk = np.pad(chunk, (0, win - len(chunk)))
            pred = torch.sigmoid(model(mel(chunk, device))).float().cpu().numpy()[0]
            fws.append(pred); times.append(s / SR + WIN / 2)
    return np.stack(fws), np.array(times)


def cache_clips(sources, device="cuda", out_dir: Path = None, names=None) -> int:
    import tempfile
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(device)
    names = names or labels()
    done = 0
    with tempfile.TemporaryDirectory() as td:
        for src in sources:
            src = Path(src); dst = out_dir / f"{src.stem}.npz"
            if dst.exists():
                continue
            wav = Path(td) / "a.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SR), str(wav)], check=True)
            fw, times = score_file(model, wav, device)
            np.savez_compressed(dst, fw=fw.astype(np.float16), times=times.astype(np.float32), labels=np.array(names)); done += 1
    return done
