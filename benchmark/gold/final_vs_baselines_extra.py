"""Two extra views of the final system vs the proposal's baselines (same rows as final_vs_baselines.py):

  * F1 per system with a paired clip-bootstrap 95 % interval of the F1 difference (10,000 draws, seed 0, two-sided p);
  * hits by importance level (2 = an event one can say in a sentence, 3 = danger or a key moment).

Show nothing has F1 = 0 and no hits. Run from the scoring checkout:
    python benchmark/gold/final_vs_baselines_extra.py      -> benchmark/gold/final_vs_baselines_extra.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import final_vs_baselines as FB

OUT = _ROOT / "benchmark" / "gold" / "final_vs_baselines_extra.json"


def counts(rows):
    """per-clip arrays: hits, misses, wrong pictures, hits and needed sounds of importance 3"""
    h = np.array([r["hit"] for r in rows], float)
    m = np.array([r["miss"] for r in rows], float)
    w = np.array([r["visible"] + r["cross"] + r["phantom"] for r in rows], float)
    h3 = np.array([r["w_hit"] - 2 * r["hit"] for r in rows], float)     # importance is 2 or 3 for scored sounds
    n3 = np.array([r["n3"] for r in rows], float)
    return h, m, w, h3, n3


def f1(h, m, w):
    return 2 * h / (2 * h + m + w) if (2 * h + m + w) else 0.0


def f1_boot(a, b, n=10000, seed=0):
    ha, ma, wa = a[:3]; hb, mb, wb = b[:3]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ha), size=(n, len(ha)))
    fa = 2 * ha[idx].sum(1) / (2 * ha[idx].sum(1) + ma[idx].sum(1) + wa[idx].sum(1))
    fb = 2 * hb[idx].sum(1) / (2 * hb[idx].sum(1) + mb[idx].sum(1) + wb[idx].sum(1))
    d = fa - fb
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((d >= 0).mean(), (d <= 0).mean()))
    point = f1(ha.sum(), ma.sum(), wa.sum()) - f1(hb.sum(), mb.sum(), wb.sum())
    return [float(point), float(lo), float(hi), float(p)]


def summarise(rows):
    c = {s: counts(rows[s]) for s in FB.SYSTEMS}
    out = {"f1": {}, "by_importance": {}}
    for s, (h, m, w, h3, n3) in c.items():
        out["f1"][s] = f1(h.sum(), m.sum(), w.sum())
        needed = h.sum() + m.sum()
        out["by_importance"][s] = {"importance_3": [int(h3.sum()), int(n3.sum())],
                                   "importance_2": [int(h.sum() - h3.sum()), int(needed - n3.sum())]}
    out["f1"]["show_nothing"] = 0.0
    out["f1_diff_final_vs_audio_to_image"] = f1_boot(c["proposed"], c["blind_a2i"])
    return out


def main():
    out = {"arm": FB.ARM, "development": summarise(FB.dev_rows()), "test": summarise(FB.test_rows())}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
