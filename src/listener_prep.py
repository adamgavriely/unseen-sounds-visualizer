"""On-the-spot inputs for the shipped listener rescue (TO1+F7F8) on a clip that has no precomputed answers.

The shipped stage 4 reads, per clip, the audio-LLM answers (Qwen3-Omni yes/no + variants, Audio Flamingo Next) and the DASM
scores. On the benchmark these were built by benchmark/gold/tagger_prep.py; `slurm/run_best.sh` runs that harness on any
folder. This module runs the SAME harness, step by step, for one new clip (a one-clip split named after it), so the
answers are identical to the benchmark's by construction (checked: shipcheck, 5 DEV clips, identical pictures). It is
slow (a scored render of the clip, then two audio LLMs and DASM, one model at a time), but needs no manual step.

    from src.listener_prep import ensure_listener_inputs
    split = ensure_listener_inputs(Path("clip.mp4"))       # then config.set_listener_split(split)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
STEPS = ("flexsed", "render", "wav16", "beats", "panns", "stage4", "stage5", "lpool", "qwen", "afn", "dasm")


def split_name(video: Path) -> str:
    return "live_" + re.sub(r"[^A-Za-z0-9_]", "_", video.stem)


def ready(split: str, stem: str) -> bool:
    g = _ROOT / "benchmark" / "gold"
    need = [g / f"{split}_listener.json", g / f"{split}_listener_v.json", g / f"{split}_listener_afn.json",
            _ROOT / "data" / "work" / f"dasm_{split}" / f"{stem}.npz"]
    return all(p.exists() for p in need)


def ensure_listener_inputs(video: Path) -> str:
    """build (once) the listener answers and DASM scores of this clip; returns the split name for set_listener_split"""
    video = Path(video).resolve()
    split, stem = split_name(video), video.stem
    if ready(split, stem):
        return split
    d = _ROOT / "data" / "input" / f"tagger_{split}"
    d.mkdir(parents=True, exist_ok=True)
    link = d / video.name
    if not link.exists():
        try:
            link.symlink_to(video)
        except OSError:                                  # no symlink rights (Windows): copy
            import shutil
            shutil.copy2(video, link)
    (_ROOT / "benchmark" / "gold" / f"{split}_stems.txt").write_text(stem + "\n", encoding="utf-8")
    env = {**os.environ, "TG_EXTRA_SPLITS": split}
    prep = str(_ROOT / "benchmark" / "gold" / "tagger_prep.py")
    for step in STEPS:
        cmd = [sys.executable, prep, step, "--split", split] + (["--arms", "B0r"] if step in ("stage4", "stage5") else [])
        print(f"       [listener-prep] {split}: {step}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(_ROOT), env=env)
    if not ready(split, stem):
        raise RuntimeError(f"[listener-prep] {split}: inputs still missing after the harness ran")
    return split
