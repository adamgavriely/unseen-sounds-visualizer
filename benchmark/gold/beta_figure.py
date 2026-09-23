"""The figure Adam asked for: whether this system helps depends on the viewer's beta.

Nothing here chooses beta. It draws the cost of every system against the price of a wrong picture,
marks the landmarks that are DERIVED rather than chosen, and shades the band where the project's
claim holds. The x-axis stops at the point above which no system of this shape can win, because
beyond it the question is not a question.

    python benchmark/gold/beta_figure.py --subset dev
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.error_taxonomy import GOLD

W_MISS = 4.0
BMAX = 4.30          # above this even a perfect sound list loses to silence: derived, not chosen


def cost(rows, b):
    return float(np.mean([W_MISS * r["miss"] + b * (r["visible"] + r["cross"] + r["phantom"]) for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="dev")
    a = ap.parse_args()
    config.use_v4("59")
    gold = S.load_gold([GOLD])
    subs = S.subsets_of(gold)
    stems = sorted(set(subs[a.subset]) & set(subs["bench"]))
    w = _ROOT / "data" / "work"

    def rows(tag, system, sts):
        return [S.score_clip(gold[s], S.load_pictures(w / f"protocol_{system}_{tag}", s, system) or []) for s in sts]

    from benchmark.gold.oracle_gate import pictures as oracle_pictures
    from benchmark.gold.gate_gold import OUT_DIR, decide
    files = {f.stem: f for f in (OUT_DIR / "Qwen38-27B").glob("*.json")}

    def oracle(sts):
        out = []
        for st in sts:
            snds = gold[st]; f = files.get(st); got = None
            if f is not None:
                by = {(c["label"], round(float(c["start"]), 2)): c
                      for c in json.loads(f.read_text(encoding="utf-8"))["sounds"]}
                sil, ok = set(), True
                for i, s in enumerate(snds):
                    c = by.get((s["label"], round(s["start"], 2)))
                    if c is None:
                        ok = False; break
                    if decide(c["stretches"], "majority"):
                        sil.add(i)
                if ok:
                    got = oracle_pictures(snds, sil)
            out.append(S.score_clip(snds, got if got is not None else []))
        return out

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    betas = np.linspace(0, BMAX, 200)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), sharey=True)
    for ax, (name, sts) in zip(axes, [("all clips", stems),
                                      ("off-screen ('unseen') clips only", [s for s in stems if S.category(gold[s]) == "unseen"])]):
        R = {"ours": rows("dev_sym_v30", "proposed", sts),
             "sensitive": rows("dev_fam_v30", "proposed", sts),
             "blind": rows("dev_sym_v30", "blind_a2i", sts),
             "oracle": oracle(sts),
             "silence": [S.score_clip(gold[s], []) for s in sts]}
        C = {k: np.array([cost(v, b) for b in betas]) for k, v in R.items()}
        # where is our line below silence?
        below = C["ours"] < C["silence"]
        if below.any():
            hi = betas[below].max()
            ax.axvspan(0.35, min(hi, BMAX), color="#2ca02c", alpha=0.10, lw=0)
            ax.text(min(hi, BMAX) / 2 + 0.15, 0.22, "cheaper than showing nothing", fontsize=8.5,
                    color="#1b6b1b", ha="center")
        ax.plot(betas, C["ours"], color="#0b6efd", lw=2.8, label="Ours — conservative (adopted)")
        ax.plot(betas, C["sensitive"], color="#6f42c1", lw=2.0, label="Ours — sensitive (per-family bars)")
        ax.plot(betas, C["blind"], color="#d62728", lw=1.6, ls="--", label="Blind (no visibility gate)")
        ax.plot(betas, C["silence"], color="#2ca02c", lw=2.0, label="Silence (subtitles today)")
        ax.plot(betas, C["oracle"], color="#0b6efd", lw=1.3, ls=":", label="Our gate, perfect sound list")
        for x, txt, st in ((0.35, "0.35  gate starts paying", "-"),
                           (0.95, "0.95  blind = silence", "-"),
                           (1.37, "1.37  ours = silence", "-"),
                           (2.00, "2.0  rubric (asserted)", "--")):
            ax.axvline(x, color="0.35" if st == "-" else "0.55", lw=0.9, ls=st, alpha=0.8)
            ax.text(x + 0.045, 2.02, txt, rotation=90, fontsize=7.2, va="bottom", color="0.28")
        ax.set_xlim(0, BMAX); ax.set_ylim(0, 5.0)
        ax.set_xlabel("β — cost of one wrong picture\n(a missed needed sound always costs 4)")
        ax.set_title(name, fontsize=11)
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("cost per clip to the viewer  (lower is better)")
    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.95)
    fig.suptitle("Whether this system helps depends on how a deaf viewer prices a wrong picture",
                 fontsize=12.5, y=0.99)
    fig.text(0.5, 0.005,
             "β is not chosen here. Below 0.35 a visibility gate costs more than it saves; above 4.30 (the right edge) even our gate "
             "given the annotator's perfect sound list loses to showing nothing.\n"
             "All-clips crossing β = 1.37, 95% CI [0.71, 2.19] — which contains the rubric's asserted 2.0, so neither side of it is "
             "established. On off-screen clips the crossing is β = 3.33 (CI [1.33, 10.0], 14 clips).",
             ha="center", fontsize=7.8, color="0.3")
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    out = _ROOT / "benchmark" / "gold" / f"beta_figure_{a.subset}.png"
    fig.savefig(out, dpi=190)
    print("->", out)


if __name__ == "__main__":
    main()
