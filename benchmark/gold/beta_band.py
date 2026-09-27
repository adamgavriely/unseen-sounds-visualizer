"""Where is ours significantly better, as a function of the price of a wrong picture (beta; a miss costs 4)?
Per-clip misses and wrong pictures from docs/inspector/data.json (the official scorer's classes); paired clip bootstrap,
2000 draws, seed 0, 95 % CI of (ours - silence) and (ours - blind) at each beta. POST HOC (asked by Adam 28 Sept after the
tables were closed); no multiplicity correction across beta; DEV was used for tuning, so DEV and DEV+TEST lean optimistic.

    python benchmark/gold/beta_band.py     # -> benchmark/gold/beta_band.json, beta_band.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = _ROOT / "docs" / "inspector" / "data.json"
OUT = _ROOT / "benchmark" / "gold" / "beta_band"
BETAS = np.round(np.arange(0, 4.01, 0.1), 2)


def counts(cl, n):
    s = cl["systems"][n]
    miss = sum(1 for i, x in enumerate(cl["sounds"]) if x["needed"] and x["importance"] >= 2 and s["sounds"][i]["outcome"] == "miss")
    return miss, sum(1 for p in s["pictures"] if p["class"].startswith("wrong"))


def band(clips):
    O = np.array([counts(c, "ours") for c in clips], float)
    B = np.array([counts(c, "blind") for c in clips], float)
    S = np.array([[sum(1 for x in c["sounds"] if x["needed"] and x["importance"] >= 2), 0] for c in clips], float)
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(clips), len(clips)) for _ in range(2000)]
    res = {"silence": [], "blind": []}
    for beta in BETAS:
        cost = lambda M: 4 * M[:, 0] + beta * M[:, 1]
        for name, other in (("silence", S), ("blind", B)):
            dd = cost(O) - cost(other)
            bs = np.array([dd[i].mean() for i in idx])
            res[name].append([float(beta), float(dd.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    return res


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.loads(DATA.read_text(encoding="utf-8"))
    sets = {"TEST (60 clips)": ["TEST"], "DEV (49)": ["DEV"], "DEV + TEST (109)": ["DEV", "TEST"]}
    out = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (name, sel) in zip(axes, sets.items()):
        cl = [c for c in d["clips"] if c["split"] in sel]
        r = band(cl); out[name] = r
        for key, col, lab in (("silence", "#2a9d3a", "ours − show nothing"), ("blind", "#d62828", "ours − draw every sound")):
            a = np.array(r[key])
            ax.plot(a[:, 0], a[:, 1], color=col, lw=2.2, label=lab)
            ax.fill_between(a[:, 0], a[:, 2], a[:, 3], color=col, alpha=0.18)
        both = [b for b, s, bl in zip(BETAS, r["silence"], r["blind"]) if s[3] < 0 and bl[3] < 0]
        if both:
            ax.axvspan(min(both) - 0.05, max(both) + 0.05, color="#1f6feb", alpha=0.12, label="ours significantly better than both")
        ax.axhline(0, color="#555", lw=0.8); ax.axvline(2, color="#999", lw=0.8, ls=":")
        ax.set_title(name); ax.set_xlabel("cost of one wrong picture (a miss costs 4)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("cost difference per clip (below 0 = ours better)")
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.suptitle("Where ours is significantly better (95 % CI band below 0) — post hoc, per-clip bootstrap", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".png"), dpi=150)
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for name, r in out.items():
        both = [b for b, s, bl in zip(BETAS, r["silence"], r["blind"]) if s[3] < 0 and bl[3] < 0]
        sil = [b for b, s in zip(BETAS, r["silence"]) if s[3] < 0]
        bli = [b for b, s in zip(BETAS, r["blind"]) if s[3] < 0]
        print(f"{name}: better than silence for beta <= {max(sil) if sil else '-'}; better than blind for beta >= "
              f"{min(bli) if bli else '-'}; better than both: {(min(both), max(both)) if both else 'none'}")


if __name__ == "__main__":
    main()
