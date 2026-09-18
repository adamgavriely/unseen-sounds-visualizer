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
# the five backbones of the paper, all fine-tuned the same way on AudioSet-Strong; the
# paper's best number is their average (docs/prereg_psed_ensemble.md)
BACKBONES = {"BEATs": ("models.beats.BEATs_wrapper", "BEATsWrapper", "BEATs_strong_1"),
             "ATST-F": ("models.atstframe.ATSTF_wrapper", "ATSTWrapper", "ATST-F_strong_1"),
             "fpasst": ("models.frame_passt.fpasst_wrapper", "FPaSSTWrapper", "fpasst_strong_1"),
             "M2D": ("models.m2d.M2D_wrapper", "M2DWrapper", "M2D_strong_1"),
             "ASIT": ("models.asit.ASIT_wrapper", "ASiTWrapper", "ASIT_strong_1")}
ENSEMBLE_CACHE = config.WORK_DIR / "psed_ens_cache"

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

_MODELS = {}


def load_model(device: str = "cuda", backbone: str = "BEATs"):
    """(model, names, device) for one of BACKBONES; loaded once per process."""
    key = (backbone, device)
    if key not in _MODELS:
        import torch, importlib
        sys.path.insert(0, str(PSED_ROOT))
        cwd = os.getcwd(); os.chdir(PSED_ROOT)          # checkpoints resolve relative to the repo
        # PretrainedSED has its own top-level `config` module (RESOURCES_FOLDER ...), which
        # clashes with ours: swap ours out of sys.modules while its code imports
        ours = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "config"}
        try:
            from models.prediction_wrapper import PredictionsWrapper
            from data_util import audioset_classes
            mod, cls, ckpt = BACKBONES[backbone]
            wrapper = getattr(importlib.import_module(mod), cls)()
            extra = {"embed_dim": wrapper.m2d.cfg.feature_d} if backbone == "M2D" else {}   # as in the paper's inference.py
            model = PredictionsWrapper(wrapper, checkpoint=ckpt, **extra).eval().to(torch.device(device))
        finally:
            os.chdir(cwd)
            sys.modules.pop("config", None)
            sys.modules.update(ours)
        names = [STRONG_TO_ONTOLOGY.get(n, n) for n in audioset_classes.as_strong_train_classes]
        _MODELS[key] = (model, names, device)
    return _MODELS[key]


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


def cache_clips(sources, device: str = "cuda", out_dir: Path = None, backbone: str = "BEATs") -> int:
    """Pre-pass (env `psed`): raw probabilities per source video/audio, keyed by stem."""
    import tempfile
    out_dir = out_dir or CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    model, names, _ = load_model(device, backbone)
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


SETTING = config.ROOT / "benchmark" / "psed_setting.json"


_BAR = None


def chosen_bar() -> float:
    """PSED's own bar, read ONCE per process and then frozen (a run must not change bar
    mid-way because a calibration file appeared): the AudioSet-Strong calibration
    (benchmark/detector_calib.json, 0.15) if present, else the DCASE choice (0.20).
    config.PSED_BAR, if set, overrides both (used to pin a run's bar explicitly)."""
    global _BAR
    if _BAR is None:
        import json
        forced = getattr(config, "PSED_BAR", None)
        calib = config.ROOT / "benchmark" / "detector_calib.json"
        if forced:
            _BAR = float(forced)
        elif calib.exists():
            _BAR = float(json.loads(calib.read_text(encoding="utf-8"))["bars"]["psed"])
        elif SETTING.exists():
            _BAR = float(json.loads(SETTING.read_text(encoding="utf-8"))["bar_chosen_on_dcase"])
        else:
            _BAR = 0.20
        print(f"       [stage4] PSED bar frozen for this run: {_BAR:.2f}", flush=True)
    return _BAR


def rescale(fw: np.ndarray, bar: float = None, ship: float = None) -> np.ndarray:
    """Map PSED's bar onto config.DISPLAY_THRESHOLD, piecewise-linearly (bar -> ship, 1 -> 1,
    0 -> 0), so the hysteresis rule, the gate and every downstream bar stay as they are."""
    bar = chosen_bar() if bar is None else bar
    ship = float(config.DISPLAY_THRESHOLD if ship is None else ship)
    s = fw.astype(np.float32)
    lo = ship * s / bar
    hi = ship + (1 - ship) * (s - bar) / (1 - bar)
    return np.clip(np.where(s < bar, lo, hi), 0, 1)


ENS_SETTING = config.ROOT / "benchmark" / "psed_ensemble_setting.json"


def ensemble_of(stem: str, root: Path):
    """Mean of the five backbones' cached probabilities for one clip (root/<backbone>/<stem>.npz);
    None if any backbone is missing. Frame rates differ slightly between backbones, so
    each is put on the first one's time grid by nearest frame."""
    parts = []
    for b in BACKBONES:
        p = root / b / f"{stem}.npz"
        if not p.exists():
            return None
        z = np.load(p, allow_pickle=False)
        parts.append((z["fw"].astype(np.float32), z["times"].astype(np.float64), [str(x) for x in z["labels"]]))
    fw0, t0, labels = parts[0]
    acc = np.zeros_like(fw0)
    for fw, t, lab in parts:
        assert lab == labels
        idx = np.clip(np.searchsorted(t, t0), 0, len(t) - 1)
        acc += fw[idx]
    return acc / len(parts), t0, labels


def infer_psed(wav_path: Path, device: str = "cuda") -> Tuple[np.ndarray, np.ndarray, list]:
    stem = Path(wav_path).parent.name
    p = cache_path(stem)
    if p.exists():
        z = np.load(p, allow_pickle=False)
        fw, times, labels = z["fw"].astype(np.float32), z["times"].astype(np.float64), [str(x) for x in z["labels"]]
    else:
        model, labels, _ = load_model(device)
        fw, times = score_file(model, Path(wav_path), device)
        labels = list(labels)
    return rescale(fw), times, labels
