"""Where does a sound START? Four timing methods on the same clips, one decoder.

    A  BEATs, 2 s window, hop 0.25, stamped 0.5 s before the window end (shipping)
    B  BEATs, 0.5 s window, hop 0.125, stamped at the window centre
    C  PANNs CNN14 DecisionLevelMax, 10 ms frames (the old detector, timing only)
    D  PretrainedSED BEATs-strong, 40 ms frames, fine-tuned on AudioSet Strong

Adam: "the sizzling starts a second late and the alarm a bit early -- is it the sync?"
It is: a 2 s window cannot say where inside it a sound begins, and the error's sign
depends on how the sound starts. This script puts the four candidates side by side on the
clips he judged, as a plot per clip and a table of onsets, so the choice is made once on
evidence rather than by moving an offset.

    python scripts/timing_compare.py <clip.mp4> [<clip.mp4> ...]  --psed-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config
from src.stage4_audio_event_detection import _extract_events, _infer
from src.stage4_audio_event_detection import beats_infer as B
from src.labels import canonical, is_salient_nonspeech

THR = 0.35
MIN_DUR = 0.5


def family_events(framewise, times, names, thr=THR, min_dur=MIN_DUR):
    """Events collapsed to families: first onset and last offset per family."""
    ev = _extract_events(framewise, times, list(names), thr, None, min_dur)
    fam = {}
    for e in ev:
        if not is_salient_nonspeech(e.label):
            continue
        f = canonical(e.label)
        s, t, c = fam.get(f, (1e9, -1, 0))
        fam[f] = (min(s, e.start), max(t, e.end), max(c, e.confidence))
    return fam


def method_A(wav, device):
    return B.infer_beats(wav, device, window=2.0, hop=0.25)


def method_B(wav, device):
    old = B.STAMP_OFFSET
    B.STAMP_OFFSET = 0.25            # window centre for a 0.5 s window
    try:
        return B.infer_beats(wav, device, window=0.5, hop=0.125)
    finally:
        B.STAMP_OFFSET = old


def method_C(wav, device):
    return _infer(wav, device)


def method_D(npz):
    d = np.load(npz, allow_pickle=True)
    probs, fps, classes = d["probs"], float(d["fps"]), list(d["classes"])
    times = np.arange(probs.shape[0]) / fps
    return probs, times, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--psed-dir", required=True, help="dir with <stem>.npz from psed_dump.py")
    ap.add_argument("--out", default=str(_ROOT / "data" / "output" / "timing"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = {}
    for clip in a.clips:
        clip = Path(clip)
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "a.wav"
            subprocess.run(["ffmpeg", "-y", "-i", str(clip), "-ac", "1", "-ar", "16000",
                            str(wav), "-loglevel", "error"], check=True)
            methods = {}
            methods["A beats 2s"] = family_events(*method_A(wav, a.device))
            methods["B beats 0.5s"] = family_events(*method_B(wav, a.device))
            try:
                methods["C panns 10ms"] = family_events(*method_C(wav, a.device))
            except Exception as e:
                print("  PANNs unavailable:", e)
            npz = Path(a.psed_dir) / (clip.stem + ".npz")
            if npz.exists():
                methods["D psed-strong 40ms"] = family_events(*method_D(npz))
        # families worth plotting: present in A (what ships) or in D (the candidate)
        fams = sorted({f for m in ("A beats 2s", "D psed-strong 40ms") if m in methods
                       for f in methods[m]},
                      key=lambda f: -max(methods[m].get(f, (0, 0, 0))[2] for m in methods))[:5]
        report[clip.stem] = {f: {m: methods[m].get(f) for m in methods} for f in fams}

        fig, ax = plt.subplots(figsize=(11, 0.9 * len(fams) * len(methods) / 2 + 1.5))
        y = 0
        ticks, labels = [], []
        colours = {"A beats 2s": "#888", "B beats 0.5s": "#4a90d9",
                   "C panns 10ms": "#e0a030", "D psed-strong 40ms": "#2eaa60"}
        for f in fams:
            for m in methods:
                v = methods[m].get(f)
                if v:
                    ax.barh(y, v[1] - v[0], left=v[0], height=0.8, color=colours[m])
                    ax.text(v[1] + 0.1, y, f"{v[2]:.2f}", va="center", fontsize=7)
                ticks.append(y); labels.append(f"{f[:18]}  [{m}]")
                y += 1
            y += 0.6
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("seconds"); ax.set_title(clip.stem)
        ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
        fig.tight_layout(); fig.savefig(out / f"{clip.stem}.png", dpi=130); plt.close(fig)
        print(f"[{clip.stem}]")
        for f in fams:
            print(f"  {f[:20]:20s} " + "  ".join(
                f"{m.split()[0]}:{v[0]:5.2f}-{v[1]:5.2f}" if (v := methods[m].get(f)) else f"{m.split()[0]}:  --  "
                for m in methods))
    (out / "timing_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("plots ->", out)


if __name__ == "__main__":
    main()
