"""Secondary, post-hoc cost-sensitivity analysis of the clip-level gate decision.

The pre-registered judge score prices a withheld picture at about 4 points and a
redundant one at about 1 -- a property of the rubric, not of deaf viewers, and the
reason the gated system only ties a baseline that draws everything. This analysis keeps
the headline untouched and asks how the comparison depends on that ratio.

Per clip and system: SHOWED = the system produced at least one augmentation (for the
caption baseline: a caption other than "no notable non-speech sound"); DUE = the human
tag is unseen_ambient or mixed. A clip is a miss (DUE and not SHOWED), redundant (SHOWED
and not DUE) or correct. Score

    S(r) = 1 - (misses + r * redundant) / N,   r = cost of a redundant picture
                                                   relative to a missed one, swept.

The judge's implied r is about 0.25. The crossover is the smallest r at which the gated
system's S(r) is at least the blind baseline's.

Caveat printed with the table: this metric scores a clip as correct if the system showed
at least one picture when the clip contains an off-screen sound, or none when it does
not. It does not assess whether the picture was shown at the right moment, for the right
sound, or with the right content; the pre-registered judge score covers those.

    python scripts/cost_sensitivity.py v3_grounded
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
DUE_TAGS = {"unseen_ambient", "mixed"}
SYSTEMS = ["proposed", "blind_a2i", "audio_caption"]
RS = np.round(np.arange(0.10, 1.0001, 0.05), 2)
CAPTION_SILENT = "no notable non-speech sound"


def showed(rec: dict) -> bool:
    if rec["system"] == "audio_caption":
        return not rec["description"].strip().lower().startswith(CAPTION_SILENT)
    return rec.get("n_augmentations", 1) > 0


def load(tag: str):
    rows = json.loads((_ROOT / "benchmark" / f"protocol_results_{tag}.json").read_text(encoding="utf-8"))
    by = {}
    for r in rows:
        by.setdefault(r["clip"], {})[r["system"]] = r
    clips = sorted(c for c, v in by.items() if len(v) == 3)
    due = np.array([by[c]["proposed"]["human_tag"] in DUE_TAGS for c in clips])
    show = {s: np.array([showed(by[c][s]) for c in clips]) for s in SYSTEMS}
    return clips, due, show


def s_of(due, show, r):
    miss = (due & ~show).sum()
    red = (~due & show).sum()
    return 1.0 - (miss + r * red) / len(due)


def main(tag: str):
    clips, due, show = load(tag)
    n = len(clips)
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, n, n) for _ in range(1000)]
    out = {"tag": tag, "n": n, "n_due": int(due.sum()), "systems": {}, "r": RS.tolist()}
    print(f"=== {tag}: {n} clips, {due.sum()} with a picture due ===")
    for s in SYSTEMS:
        sh = show[s]
        tp = int((due & sh).sum()); fp = int((~due & sh).sum()); fn = int((due & ~sh).sum()); tn = int((~due & ~sh).sum())
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        curve = [s_of(due, sh, r) for r in RS]
        ci = [np.percentile([s_of(due[b], sh[b], r) for b in boots], [2.5, 97.5]).tolist() for r in RS]
        out["systems"][s] = {"showed": int(sh.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                             "precision": prec, "recall": rec, "S": curve, "S_ci": ci}
        print(f"  {s:14s} showed {sh.sum():3d}  hit {tp:2d} miss {fn:2d} redundant {fp:2d} correct-silence {tn:2d}  "
              f"P {prec:.2f} R {rec:.2f}  S(0.25)={curve[3]:.3f}  S(0.5)={curve[8]:.3f}  S(1)={curve[-1]:.3f}")
    # paired difference gated - blind, with CI, and the crossover
    diff = [s_of(due, show["proposed"], r) - s_of(due, show["blind_a2i"], r) for r in RS]
    dci = [np.percentile([s_of(due[b], show["proposed"][b], r) - s_of(due[b], show["blind_a2i"][b], r) for b in boots],
                         [2.5, 97.5]).tolist() for r in RS]
    cross = next((float(r) for r, d in zip(RS, diff) if d >= 0), None)
    sig = next((float(r) for r, c in zip(RS, dci) if c[0] > 0), None)
    out["gated_minus_blind"] = {"diff": diff, "ci": dci, "crossover_r": cross, "ci_excludes_zero_from_r": sig}
    print(f"  gated - blind: crossover at r = {cross}; CI excludes zero from r = {sig}")
    for r, d, c in zip(RS, diff, dci):
        if float(r) in (0.1, 0.25, 0.5, 0.75, 1.0):
            print(f"    r={r:.2f}  diff {d:+.3f}  [{c[0]:+.3f}, {c[1]:+.3f}]")
    (_ROOT / "benchmark" / f"cost_sensitivity_{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        names = {"proposed": "gated (ours)", "blind_a2i": "blind audio-to-image", "audio_caption": "audio caption"}
        for s in SYSTEMS:
            c = np.array(out["systems"][s]["S_ci"])
            ax.plot(RS, out["systems"][s]["S"], label=names[s])
            ax.fill_between(RS, c[:, 0], c[:, 1], alpha=0.12)
        ax.axvline(0.25, ls=":", c="k", lw=0.8)
        ax.text(0.26, ax.get_ylim()[0] + 0.01, "judge's implied r", fontsize=7, va="bottom")
        if cross is not None:
            ax.axvline(cross, ls="--", c="grey", lw=0.8)
            ax.text(cross + 0.01, ax.get_ylim()[1] - 0.02, f"crossover r = {cross:.2f}", fontsize=7, va="top")
        ax.set_xlabel("r = cost of a redundant picture relative to a withheld one")
        ax.set_ylabel("S(r) = 1 − (missed + r·redundant)/N")
        ax.legend(fontsize=7, loc="lower left")
        fig.tight_layout()
        fig.savefig(_ROOT / "docs" / "report" / f"fig_cost_sensitivity_{tag}.pdf")
        print("figure ->", _ROOT / "docs" / "report" / f"fig_cost_sensitivity_{tag}.pdf")
    except Exception as e:
        print("no figure:", e)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v3_grounded")
