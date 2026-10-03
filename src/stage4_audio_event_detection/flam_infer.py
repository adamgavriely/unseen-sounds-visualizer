"""FLAM as the Stage 4 detector (docs/history/preregistrations/prereg_v4.md §4, release v1.2.0; the first attempt is
docs/history/preregistrations/prereg_flam.md (release v1.2.0) and benchmark/flam_detector.py (release v1.2.0)).

FLAM (Adobe, ICML 2025; ``openflam`` v1-base) scores any text query per audio frame. It
is run with the fixed descriptive vocabulary in flam_queries.py, one query per AudioSet
label, so the output has the same shape as BEATs' -- framewise[frames, Q], times, labels
-- and ``_extract_events`` works unchanged.

Calibration. FLAM's scores are not on BEATs' scale and differ per query, so each query
has its own bar, chosen once on DCASE dev-train audio (benchmark/flam_v2.py, release v1.2.0) and stored
in benchmark/flam_calibration.json. Scores are rescaled piecewise-linearly so that the
query's bar lands on config.DISPLAY_THRESHOLD and 1 stays 1; the shipping hysteresis rule
(bar, half the bar, 0.5 s) then runs on the rescaled scores.

Two environments. openflam pins torch 2.7 and clashes with the transformers the rest of
the pipeline needs, so FLAM runs in the ``sota`` conda env as a pre-pass that caches one
npz per clip (``cache_clips``); the pipeline reads the cache and only computes on the fly
when openflam happens to be importable.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Tuple

import numpy as np

import config
from .flam_queries import LABELS, TEXTS

SR = 48000
WIN = 480000                # 10 s: FLAM's input length
CALIBRATION = config.ROOT / "benchmark" / "flam_calibration.json"
CACHE_DIR = config.WORK_DIR / "flam_cache"

_MODEL = None


def load_model(device: str = "cuda"):
    global _MODEL
    if _MODEL is None:
        import openflam
        _MODEL = openflam.OpenFLAM(model_name="v1-base",
                                   default_ckpt_path=str(Path.home() / ".cache" / "openflam")).to(device).eval()
    return _MODEL


def wav48(src: Path, dst: Path) -> Path:
    """Mono 48 kHz wav straight from the source file (a 16 kHz wav would lose the top
    two octaves FLAM was trained on)."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1",
                    "-ar", str(SR), str(dst)], check=True)
    return dst


def score_file(model, wav: Path, texts=TEXTS, batch: int = 96, device: str = "cuda"):
    """Raw framewise[frames, Q] in [0, 1] (the "unbiased" local similarity is a
    probability), times[frames]."""
    import torch, librosa
    audio, _ = librosa.load(str(wav), sr=SR, mono=True)
    n = len(audio)
    fws, times = [], []
    with torch.inference_mode():
        for start in range(0, max(1, n), WIN):
            chunk = audio[start:start + WIN]
            valid = len(chunk) / SR
            if valid < 0.5:
                break
            x = torch.from_numpy(np.pad(chunk, (0, WIN - len(chunk)))).float().unsqueeze(0).to(device)
            cols = []
            for i in range(0, len(texts), batch):
                q = list(texts[i:i + batch])
                sim = model.get_local_similarity(x.repeat(len(q), 1), q, method="unbiased")   # [q, frames]
                cols.append(sim.float().clamp(0, 1).cpu().numpy().T)
            fw = np.concatenate(cols, axis=1)
            T = fw.shape[0]; hop = 10.0 / T
            keep = int(np.ceil(valid / hop))
            fws.append(fw[:keep]); times.append(start / SR + np.arange(keep) * hop)
    return np.concatenate(fws), np.concatenate(times)


def load_calibration() -> dict:
    if CALIBRATION.exists():
        return json.loads(CALIBRATION.read_text(encoding="utf-8"))["bars"]
    return {}


def rescale(fw: np.ndarray, labels, calib: dict, bar: float = None) -> np.ndarray:
    """Per column: the query's bar -> `bar` (DISPLAY_THRESHOLD), 1 -> 1, 0 -> 0, linear
    on each side. A label with no calibration entry is left as it is."""
    bar = float(config.DISPLAY_THRESHOLD if bar is None else bar)
    out = np.empty_like(fw, dtype=np.float32)
    for c, lab in enumerate(labels):
        t = calib.get(lab)
        s = fw[:, c].astype(np.float32)
        if t is None or not (0 < t < 1):
            out[:, c] = s; continue
        lo = bar * s / t
        hi = bar + (1 - bar) * (s - t) / (1 - t)
        out[:, c] = np.where(s < t, lo, hi)
    return np.clip(out, 0, 1)


def cache_path(stem: str) -> Path:
    return CACHE_DIR / f"{stem}.npz"


def cache_clips(sources, device: str = "cuda", out_dir: Path = None) -> int:
    """Pre-pass (env `sota`): raw scores for each source video/audio, keyed by its stem."""
    import tempfile
    out_dir = out_dir or CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(device)
    done = 0
    with tempfile.TemporaryDirectory() as td:
        for src in sources:
            src = Path(src); dst = out_dir / f"{src.stem}.npz"
            if dst.exists():
                continue
            fw, times = score_file(model, wav48(src, Path(td) / "a.wav"), device=device)
            np.savez_compressed(dst, fw=fw.astype(np.float16), times=times.astype(np.float32),
                                labels=np.array(LABELS))
            done += 1
    return done


def infer_flam(wav_path: Path, device: str = "cuda") -> Tuple[np.ndarray, np.ndarray, list]:
    """(framewise[frames, Q] rescaled by the calibration, times, labels) for the clip
    whose work dir holds `wav_path` (data/work/.../<stem>/audio.wav)."""
    stem = Path(wav_path).parent.name
    p = cache_path(stem)
    if p.exists():
        z = np.load(p, allow_pickle=False)
        fw, times, labels = z["fw"].astype(np.float32), z["times"].astype(np.float64), [str(x) for x in z["labels"]]
    else:
        fw, times = score_file(load_model(device), Path(wav_path), device=device)
        labels = list(LABELS)
    calib = load_calibration()
    if not calib:
        print("       [stage4] FLAM: no calibration file, raw scores in use")
    return rescale(fw, labels, calib), times, labels
