"""Steadier video droppers (Adam 10 Oct): average the "sudden change" margin of the three dense-frame runs (12 frames
wording 1, 16 frames wording 1, 12 frames wording 2), each divided by its own standard deviation over short pictures,
then drop a short picture when the mean > t, OR when the screen-reasoning "off screen" margin < u. Bars from quantiles;
clip-grouped 5-fold CV (seed 0) on all 158 clips by onset cost; paired clip bootstrap vs v1.4.
    python benchmark/gold/coverage/video_vote.py -> video_vote.md"""
import itertools, json, random, sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import GOLD
from benchmark.gold.coverage.screen_reason import base_pics

HERE = Path(__file__).resolve().parent


def main():
    gold = S.load_gold([GOLD]); pics = base_pics(); allc = list(pics)
    runs = [json.loads((HERE / f"dense_sync{t}.json").read_text()) for t in ("", "_f16", "_w2")]
    M = json.loads((HERE / "screen_margins.json").read_text())
    sd = [np.std([v["change"] for v in R.values() if v["change"] is not None and v["len"] <= 2.5]) for R in runs]
    vote = {}
    for k in runs[0]:
        xs = [R[k]["change"] / s for R, s in zip(runs, sd) if k in R and R[k]["change"] is not None]
        if xs and runs[0][k]["len"] <= 2.5:
            vote[k] = float(np.mean(xs))
    dv = sorted(vote.values()); ov = sorted(v["off"] for v in M.values() if v["off"] is not None)
    GRID = list(itertools.product([None] + [float(x) for x in np.quantile(dv, [0.5, 0.6, 0.7, 0.8, 0.9])],
                                  [None] + [float(x) for x in np.quantile(ov, [0.05, 0.1, 0.15, 0.2, 0.3])]))

    def drop(st, p, g):
        k = f"{st}|{p[0]}|{p[1]:.3f}"
        return (g[0] is not None and k in vote and vote[k] > g[0]) or (g[1] is not None and k in M and M[k]["off"] is not None and M[k]["off"] < g[1])
    memo = {}
    row = lambda st, g: memo.setdefault((st, g), S.score_clip(gold[st], [p for p in pics[st] if not drop(st, p, g)]))
    hits = lambda rr: sum(x["hit"] for x in rr)
    wr = lambda rr: sum(x["visible"] + x["cross"] + x["phantom"] for x in rr)
    cost = lambda rr: (4 * sum(x["miss"] for x in rr) + 2 * wr(rr)) / len(rr)
    L = ["# Averaged dense-frame vote OR off-screen margin, all 158 clips", "", "| dense vote bar | off-screen bar | hits | wrong | cost |", "|---|---|---|---|---|"]
    for g in GRID:
        rr = [row(st, g) for st in allc]
        L.append(f"| {g[0] if g[0] is None else round(g[0], 3)} | {g[1] if g[1] is None else round(g[1], 3)} | {hits(rr)} | {wr(rr)} | {cost(rr):.3f} |")
    for name, grid in (("vote only", [g for g in GRID if g[1] is None]), ("off-screen only", [g for g in GRID if g[0] is None]), ("both", GRID)):
        rng = random.Random(0); sh = allc[:]; rng.shuffle(sh); oof, ch = {}, []
        for k in range(5):
            te = set(sh[k::5]); tr = [c for c in allc if c not in te]
            best = min(grid, key=lambda g: cost([row(st, g) for st in tr])); ch.append(best)
            for st in te:
                oof[st] = row(st, best)
        rr = [oof[st] for st in allc]
        d = np.array([S.viewer_cost([oof[st]]) - S.viewer_cost([row(st, (None, None))]) for st in allc])
        m = d[np.random.default_rng(0).integers(0, len(d), (100000, len(d)))].mean(1)
        L += ["", f"{name}: out of fold hits {hits(rr)}, wrong {wr(rr)}, cost {cost(rr):.3f}; vs v1.4 {d.mean():+.3f} "
              f"[{np.percentile(m, 2.5):+.3f}, {np.percentile(m, 97.5):+.3f}], p {min(1, 2 * min((m >= 0).mean(), (m <= 0).mean())):.3f}"]
    (HERE / "video_vote.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L[-6:]))


if __name__ == "__main__":
    main()
