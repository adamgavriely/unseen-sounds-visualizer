"""Model inputs for one new clip (listener answers, DASM and FineLAP scores), built by the same steps as the benchmark
(benchmark/gold/clip_prep.py).

The shipped stage 4 reads, per clip, the audio-LLM answers (Qwen3-Omni yes/no + variants, Audio Flamingo Next) and the DASM
scores. This module runs clip_prep.py step by step for one new clip (a one-clip split named after it), so the
answers are built exactly as the benchmark's. It is
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


def ready(split: str, stem: str, flap: bool = True) -> bool:
    g = _ROOT / "benchmark" / "gold"
    need = [g / f"{split}_listener.json", g / f"{split}_listener_v.json", g / f"{split}_listener_afn.json",
            _ROOT / "data" / "work" / f"dasm_{split}" / f"{stem}.npz", g / f"{split}_listener_p4.json",
            g / f"{split}_listener_p1v4.json"] + ([_ROOT / "data" / "work" / f"finelap_{split}" / f"{stem}.npz"] if flap else [])
    return all(p.exists() for p in need)


def finelap(split: str, stem: str) -> None:
    """FineLAP frame scores of the clip's rescue families (benchmark/gold/finelap_screen.py, the same
    rule as DEV/TEST). FineLAP needs transformers 4.51 -> its own venv (FINELAP_PYTHON, default ~/venv_flap)."""
    if (_ROOT / "data" / "work" / f"finelap_{split}" / f"{stem}.npz").exists():
        return
    py = os.environ.get("FINELAP_PYTHON") or str(Path.home() / "venv_flap" / "bin" / "python")
    if not Path(py).exists():
        raise RuntimeError(f"[listener-prep] FineLAP venv not found at {py} (set FINELAP_PYTHON)")
    print(f"       [listener-prep] {split}: finelap", flush=True)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    subprocess.run([py, str(_ROOT / "benchmark" / "gold" / "finelap_screen.py"), "split", split, stem],
                   check=True, cwd=str(_ROOT), env=env)


def dasm_queries(env=None) -> None:
    """once per machine: DASM's MGA-CLAP text embeddings of the 215 families (data/work/dasm_text_queries.pt), which the
    `dasm` step reads. Run in its own process: the embedder takes this project off the import path."""
    from src.stage4_audio_event_detection.dasm_infer import QFILE
    if QFILE.exists():
        return
    print("       [listener-prep] DASM text queries (once)", flush=True)
    subprocess.run([sys.executable, str(_ROOT / "src" / "stage4_audio_event_detection" / "dasm_infer.py")],
                   check=True, cwd=str(_ROOT), env=env)


def ensure_listener_inputs(video: Path) -> str:
    """build (once) the listener answers, DASM and FineLAP scores of this clip; returns the split name for set_listener_split"""
    video = Path(video).resolve()
    split, stem = split_name(video), video.stem
    if ready(split, stem):
        return split
    if ready(split, stem, flap=False):                    # built before the FineLAP step existed
        finelap(split, stem)
        return split
    d = _ROOT / "data" / "input" / f"prep_{split}"
    d.mkdir(parents=True, exist_ok=True)
    link = d / video.name
    if not link.exists():
        try:
            link.symlink_to(video)
        except OSError:                                  # no symlink rights (Windows): copy
            import shutil
            shutil.copy2(video, link)
    (_ROOT / "benchmark" / "gold" / f"{split}_stems.txt").write_text(stem + "\n", encoding="utf-8")
    env = {**os.environ, "PREP_EXTRA_SPLITS": split}
    dasm_queries(env)
    prep = str(_ROOT / "benchmark" / "gold" / "clip_prep.py")
    for step in STEPS:
        cmd = [sys.executable, prep, step, "--split", split] + (["--arms", "B0r"] if step in ("stage4", "stage5") else [])
        print(f"       [listener-prep] {split}: {step}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(_ROOT), env=env)
    # DASM-only spans (P4) and both listeners' answers on them
    w = _ROOT / "data" / "work"
    p4 = _ROOT / "benchmark" / "gold" / f"{split}_listener_p4.json"
    dr = str(_ROOT / "benchmark" / "gold" / "dasm_rescue.py")
    print(f"       [listener-prep] {split}: dasm rescue pool + listeners", flush=True)
    subprocess.run([sys.executable, dr, "pool", split, str(w / f"r13{split}" / "stage4.json"), str(w / f"dasm_{split}"),
                    str(w / f"r13{split}" / "wav16"), str(p4)], check=True, cwd=str(_ROOT), env=env)
    subprocess.run([sys.executable, dr, "listen", str(p4)], check=True, cwd=str(_ROOT), env=env)
    # open listener inventory: Qwen V4 on the P1 cuts + both listeners' family lists
    gd = _ROOT / "benchmark" / "gold"
    print(f"       [listener-prep] {split}: P1 open inventory", flush=True)
    subprocess.run([sys.executable, str(gd / "listener_p1v4.py"),
                    f"{split}:{gd / f'{split}_listener_v.json'}:{gd / f'{split}_listener_afn.json'}:{w / f'r13{split}' / 'wav16'}:"
                    f"{gd / f'{split}_listener_p1v4.json'}"], check=True, cwd=str(_ROOT), env=env)
    finelap(split, stem)
    if not ready(split, stem):
        raise RuntimeError(f"[listener-prep] {split}: inputs still missing after the harness ran")
    return split
