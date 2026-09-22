"""The cost curve: every system's cost to a deaf viewer, over the whole range of "how bad is a wrong
picture?" (Adam, 2026-09-23).

Equal-weight F1 prices a picture of a sound that is not there exactly like a sound left undrawn, and
cannot see the 74 of 139 clips that contain no needed sound at all. This figure replaces the single
number with the trade-off itself:

    cost per clip = w_miss x (needed sounds rated >= 2 with no picture) + beta x (pictures that are
                    false alarms)

with beta swept from 0 (a wrong picture is free) to 4 (a wrong picture is as bad as a missed danger
sound). Where a system's line is lowest, that system is the right choice for a viewer with that
trade-off. The declared operating point (beta = 2, w_miss = 4) comes from this project's own judging
rubric (benchmark/gate_dev_sweep.py, September 2026), not from this result.

Systems drawn: OURS (the cross-modal gate), BLIND (same pipeline, gate off -- every detected sound is
drawn), CAPTION (the sound names as text), SILENCE (show nothing -- what a deaf viewer has today),
and ORACLE-GATE (the gate given a perfect sound list) as the upper bound the gate could reach.

    python benchmark/gold/cost_curve.py --tag v4b4 [--weighted]
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

GOLD = _ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"
W_MISS = 4.0
BETAS = np.linspace(0, 4, 41)


def cost_of(rows, beta, weighted):
    """mean cost per clip; `weighted` prices a missed sound by the annotator's importance (2 or 3)"""
    out = []
    for r in rows:
        miss = 2.0 * r["w_miss"] if weighted else W_MISS * r["miss"]
        out.append(miss + beta * (r["visible"] + r["cross"] + r["phantom"]))
    return float(np.mean(out)) if out else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v4b4")
    ap.add_argument("--subset", default="bench")
    ap.add_argument("--weighted", action="store_true", help="price a missed sound by its importance")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    config.use_v4("59")
    gold = S.load_gold([GOLD])
    subs = S.subsets_of(gold)
    w = _ROOT / "data" / "work"
    rows = {k: [] for k in ("ours", "blind", "caption", "silence", "oracle")}
    from benchmark.gold.oracle_gate import pictures as oracle_pictures
    from benchmark.gold.gate_gold import OUT_DIR, decide
    files = {f.stem: f for f in (OUT_DIR / "Qwen38-27B").glob("*.json")}
    stems = []
    for stem in sorted(subs[a.subset]):
        snds = gold[stem]
        p = {"ours": S.load_pictures(w / f"protocol_proposed_{a.tag}", stem, "proposed"),
             "blind": S.load_pictures(w / f"protocol_blind_a2i_{a.tag}", stem, "blind_a2i"),
             "caption": S.load_pictures(w / f"protocol_audio_caption_{a.tag}", stem, "audio_caption"),
             "silence": []}
        if p["ours"] is None or p["blind"] is None:
            continue
        if p["caption"] is None:
            p["caption"] = p["blind"]                      # the caption row shares the detector's list
        # the oracle line: the same gate verdicts applied to the annotator's own sound list
        orc = None
        f = files.get(stem)
        if f is not None:
            cached = json.loads(f.read_text(encoding="utf-8"))["sounds"]
            bykey = {(c["label"], round(float(c["start"]), 2)): c for c in cached}
            sil, ok = set(), True
            for i, s in enumerate(snds):
                c = bykey.get((s["label"], round(s["start"], 2)))
                if c is None:
                    ok = False; break
                if decide(c["stretches"], "majority"):
                    sil.add(i)
            if ok:
                orc = oracle_pictures(snds, sil)
        stems.append(stem)
        for k in ("ours", "blind", "caption", "silence"):
            rows[k].append(S.score_clip(snds, p[k]))
        rows["oracle"].append(S.score_clip(snds, orc) if orc is not None else S.score_clip(snds, p["ours"]))
    curves = {k: [cost_of(v, b, a.weighted) for b in BETAS] for k, v in rows.items()}
    out = {"tag": a.tag, "subset": a.subset, "clips": len(stems), "weighted": a.weighted,
           "betas": [float(b) for b in BETAS], "curves": curves}
    j = Path(a.out or (_ROOT / "benchmark" / "gold" / f"cost_curve_{a.tag}{'_w' if a.weighted else ''}.json"))
    j.write_text(json.dumps(out, indent=1), encoding="utf-8")
    # the figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"ours": ("#0b6efd", 2.6, "-", "Ours (cross-modal gate)"),
             "blind": ("#d62728", 2.0, "-", "Blind (draw every detected sound)"),
             "caption": ("#7f7f7f", 1.6, "--", "Caption (sound names as text)"),
             "silence": ("#2ca02c", 2.0, "-", "Silence (show nothing: today's subtitles)"),
             "oracle": ("#0b6efd", 1.4, ":", "Gate with a perfect sound list (upper bound)")}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for k in ("silence", "caption", "blind", "oracle", "ours"):
        c, lw, ls, lab = style[k]
        ax.plot(BETAS, curves[k], color=c, lw=lw, ls=ls, label=lab)
    ax.axvline(2.0, color="k", lw=0.8, alpha=0.5)
    ax.annotate("declared operating point\n(rubric: miss 4, wrong picture 2)", xy=(2.0, ax.get_ylim()[1] * 0.95),
                xytext=(2.15, ax.get_ylim()[1] * 0.95), fontsize=8, va="top", color="0.25")
    ax.set_xlabel("cost of one wrong picture  (a missed needed sound costs 4)")
    ax.set_ylabel("cost per clip  (lower is better)")
    ax.set_title(f"What each system costs a deaf viewer — {len(stems)} clips"
                 + (" (misses weighted by importance)" if a.weighted else ""))
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    png = j.with_suffix(".png")
    fig.savefig(png, dpi=190)
    print("->", png)
    for k in ("ours", "blind", "caption", "silence", "oracle"):
        i2 = int(np.argmin(np.abs(BETAS - 2.0)))
        print(f"  {style[k][3]:46s} cost at beta=2: {curves[k][i2]:5.2f}")


if __name__ == "__main__":
    main()
