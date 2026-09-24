"""Paired comparison of judge scores between systems, by gold subset (clip bootstrap, 2000 draws, seed 0).

    python benchmark/gold/judge_compare.py benchmark/judge_direct_v4b4.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from benchmark.gold import score_per_sound as S  # noqa: E402
from benchmark.gold.error_taxonomy import GOLD  # noqa: E402


def main(path, subsets=("all", "dev", "test", "sliceB", "cat_unseen", "cat_mixed", "cat_seen", "cat_no_ambient")):
    g = S.load_gold([GOLD])
    subs = S.subsets_of(g)
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    sc = {(r["clip"], r["system"]): r["score"] for r in rows if r.get("score") is not None and not r.get("repeat")}
    for sub in subsets:
        pool = g if sub == "all" else subs.get(sub, [])
        clips = [c for c in pool if (c, "proposed") in sc]
        if not clips:
            continue
        cells = []
        for other in ("blind_a2i", "audio_caption"):
            cl = [c for c in clips if (c, other) in sc]
            d = np.array([sc[(c, "proposed")] - sc[(c, other)] for c in cl], float)
            rng = np.random.default_rng(0)
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            star = "*" if lo > 0 or hi < 0 else " "
            cells.append(f"vs {other[:7]:7s} {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]{star}")
        mean = np.mean([sc[(c, "proposed")] for c in clips])
        print(f"{sub:15s} n={len(clips):3d}  ours {mean:.2f}   " + "   ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1])
