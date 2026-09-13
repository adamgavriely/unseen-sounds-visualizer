"""Dump PretrainedSED frame-level probabilities for a wav, for the timing comparison.

Run inside the `psed` env with the PretrainedSED checkout on sys.path:
    python scripts/psed_dump.py <wav> <out.npz> [--model BEATs]

Writes probs (frames x classes) at the model's frame rate, the frame rate, and the class
names (AudioSet Strong display names). Everything else -- decoding into events, thresholds
-- is done in the comparison script so all four methods share one decoder.
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import librosa

ap = argparse.ArgumentParser()
ap.add_argument("wav")
ap.add_argument("out")
ap.add_argument("--model", default="BEATs")
ap.add_argument("--psed", default=str(Path.home() / "PretrainedSED"))
args = ap.parse_args()
sys.path.insert(0, args.psed)
import os
os.chdir(args.psed)                       # checkpoints resolve relative to the repo

from models.prediction_wrapper import PredictionsWrapper
from models.beats.BEATs_wrapper import BEATsWrapper
from data_util import audioset_classes

model = PredictionsWrapper(BEATsWrapper(), checkpoint="BEATs_strong_1").eval()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(dev)

sr, seg = 16_000, 10 * 16_000
wav, _ = librosa.load(args.wav, sr=sr, mono=True)
x = torch.from_numpy(wav[None, :]).to(dev)
n = x.shape[1]
chunks = []
for i in range(n // seg + (n % seg != 0)):
    c = x[:, i * seg:(i + 1) * seg]
    if c.shape[1] < seg:
        c = torch.nn.functional.pad(c, (0, seg - c.shape[1]))
    with torch.no_grad():
        y, _ = model(model.mel_forward(c))
    chunks.append(y)
y = torch.sigmoid(torch.cat(chunks, dim=2).float())[0].cpu().numpy().T   # (frames, classes)
frames_total = y.shape[0]
fps = frames_total / (len(chunks) * 10.0)
y = y[: int(round(n / sr * fps))]                                          # drop padding
np.savez(args.out, probs=y, fps=fps, classes=np.array(audioset_classes.as_strong_train_classes))
print("wrote", args.out, y.shape, "fps", fps)
