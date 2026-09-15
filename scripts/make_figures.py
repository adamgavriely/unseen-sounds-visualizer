"""Two report figures, from cached decisions and the benchmark clips (no models).

  fig_pipeline.pdf   the seven stages as boxes, with what flows between them
  fig_examples.pdf   three test clips end to end: frames, the detector's confidence over
                     time, the three visibility votes, and what the panel showed --
                     a correct silence (win), a wrong silence (loss), a correct picture.

    python scripts/make_figures.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
OUT = _ROOT / "docs" / "report"


# ----------------------------------------------------------------------------- pipeline
def pipeline():
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.set_xlim(0, 11.2); ax.set_ylim(0, 4.3); ax.axis("off")
    xs = [0.15, 2.45, 4.75, 7.05, 9.35]
    w, h = 1.55, 1.2
    top = [(xs[0], "1  Audio\nFFmpeg"),
           (xs[1], "4  Detect sounds\nBEATs, 527 classes\n2 s windows, bar 0.35"),
           (xs[2], "5  Source visible?\nQwen2.5-VL-7B\n3 votes per 5 s"),
           (xs[3], "5  Depict + merge\nevent phrase + place\nontology dedup"),
           (xs[4], "6  Draw + panel\nFLUX.1-schnell\n≤ 3 slots, ≥ 1.5 s")]
    low = [(xs[1], "3  Speech\nWhisper → hint only"),
           (xs[2], "2  Objects in clip\nOWLv2 → candidates"),
           (xs[3], "7  Evaluate\ndescribe → reference\n→ judge 0–4")]

    yt, yl = 2.7, 0.5
    for x, t in top:
        ax.add_patch(FancyBboxPatch((x, yt), w, h, boxstyle="round,pad=0.04", fc="#f2f2f2", ec="#444", lw=0.9))
        ax.text(x + w / 2, yt + h / 2, t, ha="center", va="center", fontsize=7)
    for x, t in low:
        ax.add_patch(FancyBboxPatch((x, yl), w, h, boxstyle="round,pad=0.04", fc="#fafafa", ec="#777", lw=0.8))
        ax.text(x + w / 2, yl + h / 2, t, ha="center", va="center", fontsize=7)

    def arrow(p0, p1, txt="", tx=None, ty=None):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#333",
                                     shrinkA=1, shrinkB=1))
        if txt:
            ax.text(tx if tx is not None else (p0[0] + p1[0]) / 2, ty if ty is not None else (p0[1] + p1[1]) / 2,
                    txt, ha="center", va="bottom", fontsize=5.9, color="#333", linespacing=0.95)
    ym = yt + h / 2
    for a, b, t in zip(xs[:-1], xs[1:], ["wav", "sounds"+chr(10)+"+ times", "unseen"+chr(10)+"sounds", "phrases"]):
        arrow((a + w, ym), (b, ym), t, ty=ym + 0.08)
    arrow((xs[1] + w * 0.75, yl + h), (xs[2] + 0.15, yt), "speech near a sound:"+chr(10)+"priority", tx=3.15, ty=1.95)
    arrow((xs[2] + w / 2, yl + h), (xs[2] + w / 2, yt), "defer to VLM", tx=xs[2] + w / 2 + 0.55, ty=2.05)
    arrow((xs[4] + w / 2, yt), (xs[3] + w, yl + h / 2), "augmented"+chr(10)+"video", tx=10.3, ty=1.6)
    ax.text(xs[0], 4.2, "video in", fontsize=7.5, va="top")
    ax.text(xs[4] + w / 2, 4.2, "video + side panel out", fontsize=7.5, va="top", ha="center")
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "fig_pipeline.pdf")
    print("->", OUT / "fig_pipeline.pdf")


# ----------------------------------------------------------------------------- examples
EXAMPLES = [
    # (clip, label of interest, headline, panel image or None, judge scores gated/blind)
    ("un_police_car_siren_WEROVWDp.mp4", "Siren",
     "correct silence: the siren's police car is on screen", None, (4, 3)),
    ("ambient_citywalk_nyc_2627.mp4", "Vehicle",
     "wrong silence: cars visible, the off-screen traffic still needed a picture", None, (0, 3)),
    ("ly_ambulance_(siren)_-yPSgCn.mp4", "Siren",
     "correct picture: siren heard, no ambulance in frame", "fig_src/ambulance_aug0.png", (4, 3)),
]
VOTE = {True: "yes", False: "no", None: "split"}


def frames_at(video: Path, times, td: Path):
    imgs = []
    for i, t in enumerate(times):
        fp = td / f"f{i}.jpg"
        subprocess.run(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
                        "-vf", "scale=320:-1", "-q:v", "4", str(fp), "-loglevel", "error"], check=True)
        imgs.append(Image.open(fp).convert("RGB").copy())
    return imgs


def examples():
    from benchmark.gate_dev_sweep import _find_clip
    from src.labels import canonical
    n = len(EXAMPLES)
    fig = plt.figure(figsize=(7.2, 8.4))
    gs = fig.add_gridspec(2 * n, 6, width_ratios=[1, 1, 1, 1, 1.8, 1.05],
                          height_ratios=[1.0, 0.42] * n, hspace=0.6, wspace=0.08)
    heads = []
    with tempfile.TemporaryDirectory() as td:
        for row, (clip, label, headline, panel, scores) in enumerate(EXAMPLES):
            r0 = 2 * row
            rec = json.loads((_ROOT / "benchmark" / "gate_votes" / "test" / (clip + ".json")).read_text(encoding="utf-8"))
            snd = next(s_ for s_ in rec["sounds"] if s_["label"] == label)
            video = _find_clip(clip)
            dur = rec["duration"]
            times = [snd["start"] + (snd["end"] - snd["start"]) * f for f in (0.05, 0.35, 0.65, 0.95)]
            imgs = frames_at(video, times, Path(td))
            for k, (im, t) in enumerate(zip(imgs, times)):
                ax = fig.add_subplot(gs[r0, k]); ax.imshow(im); ax.axis("off")
                ax.set_title(f"{t:.1f} s", fontsize=6.5, pad=2)
            z = np.load(_ROOT / "benchmark" / "beats_windows" / "test" / (clip + ".npz"), allow_pickle=False)
            labels = list(z["labels"]); fw = z["fw"].astype(np.float32); tt = z["times"]
            cols = [i for i, l in enumerate(labels) if canonical(l) == label or l == label]
            curve = fw[:, cols].max(axis=1) if cols else np.zeros(len(tt))
            ax = fig.add_subplot(gs[r0, 4])
            ax.fill_between(tt, 0, curve, color="#7a9cc6", alpha=0.6)
            ax.axhline(0.35, ls=":", c="k", lw=0.7)
            ax.axvspan(snd["start"], snd["end"], color="#ffd27f", alpha=0.35)
            ax.set_xlim(0, dur); ax.set_ylim(0, 1); ax.set_yticks([0, 0.35, 1]); ax.tick_params(labelsize=6)
            ax.set_title(f"BEATs '{label}' peak {snd['confidence']:.2f}", fontsize=6.8, pad=2)
            ax.set_xlabel("s", fontsize=6, labelpad=1)
            ax = fig.add_subplot(gs[r0, 5]); ax.axis("off")
            if panel:
                ax.imshow(Image.open(OUT / panel).convert("RGB"))
                ax.set_title("panel: picture shown", fontsize=6.8, pad=2)
            else:
                ax.add_patch(FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.02", fc="white", ec="#999", lw=0.8))
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.text(0.5, 0.5, "panel: empty", ha="center", va="center", fontsize=7, color="#555")
                ax.set_title("panel", fontsize=6.8, pad=2)
            # votes row, spanning the width
            ax = fig.add_subplot(gs[r0 + 1, :]); ax.axis("off")
            lines = [f"stretch {st['start']:4.1f}–{st['end']:4.1f} s:  name → {VOTE[st['name']]:5s} ({st['named'][:22]:22s})  "
                     f"a/b → {VOTE[st['ab']]:5s}  describe → {VOTE[st['desc']]:5s}  ⇒ {'VISIBLE' if st['verdict'] else 'not visible'}"
                     for st in snd["stretches"][:4]]
            verdict = ("silenced (visible in every stretch)" if all(st["verdict"] for st in snd["stretches"])
                       else "shown for the whole sound")
            ax.text(0.0, 1.0, chr(10).join(lines) + f"{chr(10)}→ {verdict};  judge: gated {scores[0]}, blind {scores[1]}",
                    transform=ax.transAxes, fontsize=6.0, va="top", family="monospace")
            heads.append((fig.axes[-6 - 1 + 0] if False else fig.axes[len(fig.axes) - 7], headline, row))
    for ax0, headline, row in heads:
        y = ax0.get_position().y1 + 0.06
        fig.text(0.01, y, f"({'abc'[row]}) {headline}", fontsize=8, va="bottom", weight="bold")
    fig.savefig(OUT / "fig_examples.pdf", bbox_inches="tight")
    print("->", OUT / "fig_examples.pdf")


if __name__ == "__main__":
    pipeline()
    examples()
