"""PretrainedSED "BEATs-strong" as the Stage 4 detector (v4, docs/prereg_psed.md).

Schmid et al., "Effective Pre-Training of Audio Transformers for Sound Event Detection"
(CP-JKU, ICASSP 2025): the same BEATs backbone we ship, fine-tuned FRAME BY FRAME on
AudioSet-Strong (human-timed labels) with distillation -- PSDS1 46.5 against 36.5 for a
plain BEATs, the recognised fixed-vocabulary state of the art with public weights (the
2026 boundary-aware successor, 49.6, has none). It outputs a probability every ~40 ms for
447 AudioSet-Strong classes, so the 2-s sliding window and the occlusion onset refinement
are not needed; the hysteresis span rule runs on its frames unchanged.

Class names are AudioSet-Strong's; 41 of the 447 are renamed or new relative to the
ontology names the rest of the pipeline uses (labels.py families, the depiction query),
so they are mapped back to the nearest ontology label here (STRONG_TO_ONTOLOGY); a name
with no counterpart is kept as it is and is simply its own family.

Environment: the PretrainedSED checkout (~/PretrainedSED) and its own conda env `psed`
(numpy < 2), so like FLAM it runs as a cached pre-pass; the pipeline reads the cache and
only computes on the fly when the checkout is importable.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

import config

SR = 16000
SEG = 10 * SR                     # the model takes 10-s inputs
PSED_ROOT = Path(os.environ.get("PSED_ROOT", Path.home() / "PretrainedSED"))
CHECKPOINT = "BEATs_strong_1"
CACHE_DIR = config.WORK_DIR / "psed_cache"

STRONG_TO_ONTOLOGY = {
    "Alert": "Alarm", "Bathroom sounds": "Domestic sounds, home sounds", "Bicycle, tricycle": "Bicycle",
    "Blender, food processor": "Blender", "Canidae, wild dogs, wolves": "Canidae, dogs, wolves",
    "Carbon monoxide detector, CO detector": "Smoke detector, smoke alarm",
    "Crockery breaking and smashing": "Shatter", "Dong, bong": "Bell", "Ducks, geese, waterfowl": "Duck",
    "Electric rotor drone, quadcopter": "Helicopter", "Error signal": "Beep, bleep", "Glass chink, clink": "Chink, clink",
    "Glass shatter": "Shatter", "Gurgling, bubbling": "Gurgling", "Keypress tone": "Beep, bleep",
    "Kitchen and dining room sounds": "Dishes, pots, and pans", "Lock": "Door", "Mechanical bell": "Bell",
    "Pant (dog)": "Dog", "Power saw, circular saw, table saw": "Sawing", "Ringing tone, ringback tone": "Telephone",
    "Shower": "Water tap, faucet", "Stomp, stamp": "Walk, footsteps", "Stream, river": "Stream",
    "Tire squeal, skidding": "Skidding", "Vehicle horn, car horn, honking, toot": "Vehicle horn, car horn, honking",
    "White noise, pink noise": "White noise",
}

_MODEL = None


def load_model(device: str = "cuda"):
    global _MODEL
    if _MODEL is None:
        import torch
        sys.path.insert(0, str(PSED_ROOT))
        cwd = os.getcwd(); os.chdir(PSED_ROOT)          # checkpoints resolve relative to the repo
        try:
            from models.prediction_wrapper import PredictionsWrapper
            from models.beats.BEATs_wrapper import BEATsWrapper
            from data_util import audioset_classes
            model = PredictionsWrapper(BEATsWrapper(), checkpoint=CHECKPOINT).eval().to(torch.device(device))
        finally:
            os.chdir(cwd)
        names = [STRONG_TO_ONTOLOGY.get(n, n) for n in audioset_classes.as_strong_train_classes]
        _MODEL = (model, names, device)
    return _MODEL


def wav16(src: Path, dst: Path) -> Path:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1",
                    "-ar", str(SR), str(dst)], check=True)
    return dst


def score_file(model, wav: Path, device: str = "cuda"):
    """framewise[frames, classes] probabilities, times[frames]."""
    import torch, librosa
    audio, _ = librosa.load(str(wav), sr=SR, mono=True)
    x = torch.from_numpy(audio[None, :]).to(device)
    n = x.shape[1]
    chunks = []
    for i in range(n // SEG + (n % SEG != 0)):
        c = x[:, i * SEG:(i + 1) * SEG]
        if c.shape[1] < SEG:
            c = torch.nn.functional.pad(c, (0, SEG - c.shape[1]))
        with torch.no_grad():
            y, _ = model(model.mel_forward(c))
        chunks.append(y)
    y = torch.sigmoid(torch.cat(chunks, dim=2).float())[0].cpu().numpy().T     # (frames, classes)
    fps = y.shape[0] / (len(chunks) * 10.0)
    y = y[: int(round(n / SR * fps))]
    return y, np.arange(y.shape[0]) / fps


def cache_path(stem: str) -> Path:
    return CACHE_DIR / f"{stem}.npz"


def cache_clips(sources, device: str = "cuda", out_dir: Path = None) -> int:
    """Pre-pass (env `psed`): raw probabilities per source video/audio, keyed by stem."""
    import tempfile
    out_dir = out_dir or CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    model, names, _ = load_model(device)
    done = 0
    with tempfile.TemporaryDirectory() as td:
        for src in sources:
            src = Path(src); dst = out_dir / f"{src.stem}.npz"
            if dst.exists():
                continue
            fw, times = score_file(model, wav16(src, Path(td) / "a.wav"), device)
            np.savez_compressed(dst, fw=fw.astype(np.float16), times=times.astype(np.float32), labels=np.array(names))
            done += 1
    return done


def infer_psed(wav_path: Path, device: str = "cuda") -> Tuple[np.ndarray, np.ndarray, list]:
    stem = Path(wav_path).parent.name
    p = cache_path(stem)
    if p.exists():
        z = np.load(p, allow_pickle=False)
        return z["fw"].astype(np.float32), z["times"].astype(np.float64), [str(x) for x in z["labels"]]
    model, names, _ = load_model(device)
    fw, times = score_file(model, Path(wav_path), device)
    return fw, times, list(names)
