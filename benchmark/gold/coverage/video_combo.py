"""Video droppers together (Adam 10 Oct): dense-frame "sudden change" on short pictures (dense_sync.json) OR screen
reasoning "source off screen" margin (screen_margins.json). Grid: dense bar in {off, 0.5, 0.94, 1.74, 2.725} x off-screen
bar in {off, -6.14, -3.15, -1.637, -0.5}; clip-grouped 5-fold CV (seed 0) on all 158 clips by onset cost and by "no hit lost".
    python benchmark/gold/coverage/video_combo.py -> video_combo.md"""
import itertools, json, random, sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import GOLD
from benchmark.gold.coverage.screen_reason import base_pics

HERE = Path(__file__).resolve().parent
GRID = list(itertools.product((None, 0.5, 0.94, 1.74, 2.725), (None, -6.14, -3.15, -1.637, -0.5)))


def main():
    gold = S.load_gold([GOLD]); pics = base_pics(); allc = list(pics)
    D = json.loads((HERE / "dense_sync.json").read_text()); M = json.loads((HERE / "screen_margins.json").read_text())

    def drop(st, p, g):
        k = f"{st}|{p[0]}|{p[1]:.3f}"; d, m = D.get(k), M.get(k)
        a = g[0] is not None and d is not None and d["change"] is not None and d["len"] <= 2.5 and d["change"] > g[0]
        b = g[1] is not None and m is not None and m["off"] is not None and m["off"] < g[1]
        return a or b
    memo = {}
    row = lambda st, g: memo.setdefault((st, g), S.score_clip(gold[st], [p for p in pics[st] if not drop(st, p, g)]))
    hits = lambda rr: sum(x["hit"] for x in rr)
    wr = lambda rr: sum(x["visible"] + x["cross"] + x["phantom"] for x in rr)
    cost = lambda rr: (4 * sum(x["miss"] for x in rr) + 2 * wr(rr)) / len(rr)
    L = ["# Dense-frame change OR off-screen margin, all 158 clips", "", "| dense bar | off-screen bar | hits | wrong | cost |", "|---|---|---|---|---|"]
    for g in GRID:
        rr = [row(st, g) for st in allc]; L.append(f"| {g[0]} | {g[1]} | {hits(rr)} | {wr(rr)} | {cost(rr):.3f} |")
    base = [row(st, (None, None)) for st in allc]
    for mode in ("cost", "no hit lost"):
        rng = random.Random(0); sh = allc[:]; rng.shuffle(sh); oof, ch = {}, []
        for k in range(5):
            te = set(sh[k::5]); tr = [c for c in allc if c not in te]
            if mode == "cost":
                best = min(GRID, key=lambda g: cost([row(st, g) for st in tr]))
            else:
                h0 = hits([row(st, (None, None)) for st in tr])
                best = min([g for g in GRID if hits([row(st, g) for st in tr]) >= h0], key=lambda g: wr([row(st, g) for st in tr]))
            ch.append(best)
            for st in te:
                oof[st] = row(st, best)
        rr = [oof[st] for st in allc]
        d = np.array([S.viewer_cost([oof[st]]) - S.viewer_cost([row(st, (None, None))]) for st in allc])
        idx = np.random.default_rng(0).integers(0, len(d), (100000, len(d))); m = d[idx].mean(1)
        L += ["", f"CV by {mode}: choices {ch}; out of fold hits {hits(rr)}, wrong {wr(rr)}, cost {cost(rr):.3f}; "
              f"paired vs v1.4 {d.mean():+.3f} [{np.percentile(m, 2.5):+.3f}, {np.percentile(m, 97.5):+.3f}], p {min(1, 2 * min((m >= 0).mean(), (m <= 0).mean())):.3f}"]
    (HERE / "video_combo.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
