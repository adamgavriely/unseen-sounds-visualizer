"""FlexSED (open-vocabulary SED) as a second detector, read from a cache.

Why it is here (measured 2026-09-22 on Adam's gold set): at every onset where BEATs scored a needed
sound below 0.05, BEATs' own top labels were Speech 0.58-0.81 or Music 0.48-0.58 -- the sound was
masked, not out of vocabulary. FlexSED is queried one label at a time (Dasheng SSL encoder + CLAP
text encoder, trained on AudioSet-Strong), so a loud voice cannot out-vote a quiet hammer. On the
nine masked sounds it lifts 0.006-0.044 to 0.49-0.85 and recovers six of them.

The frame probabilities for the project's 215 depictable family names are computed once per clip by
benchmark/gold/flexsed_run.py (a separate environment: the FlexSED repo has its own `src` package)
and cached as data/work/flexsed_cache/<stem>.npz. This module only reads that cache and hands the
scores to the same span extractor BEATs uses, so the union is one line in detect_events.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

import config

CACHE_DIR = config.WORK_DIR / "flexsed_cache"


def cache_path(stem: str) -> Path:
    return CACHE_DIR / f"{stem}.npz"


def available(wav_path: Path) -> bool:
    return cache_path(Path(wav_path).parent.name).exists()


def infer_flexsed(wav_path: Path, device: str = "cuda") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """(framewise[frames, n_labels], times[frames], labels) from the cache; keyed like the PSED
    cache, by the work directory's name (data/work/<tag>/<stem>/audio.wav -> <stem>)."""
    stem = Path(wav_path).parent.name
    p = cache_path(stem)
    if not p.exists():
        raise FileNotFoundError(f"no FlexSED cache for {stem}; run benchmark/gold/flexsed_run.py")
    z = np.load(p, allow_pickle=False)
    fw = z["fw"].astype(np.float32).T                     # stored [n_labels, T]
    labels = [str(x) for x in z["labels"]]
    times = np.arange(fw.shape[0], dtype=np.float64) / float(z["fps"])
    return fw, times, labels
