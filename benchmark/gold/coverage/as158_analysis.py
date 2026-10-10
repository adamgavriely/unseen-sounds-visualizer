"""PREREG_beats_strong.md, A (wrong-dropper) and B (proposer ceiling), on as158_feats.json (as158_feats.py, cluster).
    python benchmark/gold/coverage/as158_analysis.py -> as158_analysis.md"""
import json, random, sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import GOLD, stems
from benchmark.gold.coverage.screen_reason import base_pics

HERE = Path(__file__).resolve().parent
MODELS = ("BEATs", "ATST-F", "fpasst", "M2D", "ASIT")


def main():
    F = json.loads((HERE / "as158_feats.json").read_text())
    gold = S.load_gold([GOLD]); pics = base_pics(); allc = stems("dev") + stems("test")
    for f in F["pics"]:
        for w in ("in", "on"):
            xs = [f[f"{m}_{w}"] for m in MODELS if f[f"{m}_{w}"] is not None]
            f[f"mean_{w}"] = float(np.mean(xs)) if xs else None
    feat = {(f["clip"], f["label"], round(f["start"], 3)): f for f in F["pics"]}
    keys = ["BEATs_in", "BEATs_on", "mean_in", "mean_on"]
    L = ["# BEATs-strong (PretrainedSED) on the 158 benchmark clips", "", "## A. Wrong-dropper on v1.7 pictures", "",
         "Family probability per picture (median, and the lowest hit):", "", "| feature | hits median | lowest hit | wrongs median | wrongs below the lowest hit |", "|---|---|---|---|---|"]
    for k in keys:
        h = [f[k] for f in F["pics"] if f["cls"] == "hit" and f[k] is not None]
        w = [f[k] for f in F["pics"] if f["cls"] == "wrong" and f[k] is not None]
        L.append(f"| {k} | {np.median(h):.3f} | {min(h):.3f} | {np.median(w):.3f} | {sum(x < min(h) for x in w)} of {len(w)} |")
    L += ["", "No family class in the 447 (feature empty, never dropped): " + ", ".join(sorted({f['label'] for f in F['pics'] if not f['has_family']})), ""]
    grid = [None]
    for k in keys:
        v = np.array([f[k] for f in F["pics"] if f[k] is not None])
        grid += [(k, float(x)) for x in np.unique(np.round(np.quantile(v, np.linspace(0.02, 0.4, 12)), 4))]

    def drop(st, p, g):
        f = feat.get((st, p[0], round(p[1], 3)))
        return g is not None and f is not None and f[g[0]] is not None and f[g[0]] < g[1]
    memo = {}
    row = lambda st, g: memo.setdefault((st, g), S.score_clip(gold[st], [p for p in pics[st] if not drop(st, p, g)]))
    hits = lambda rr: sum(x["hit"] for x in rr)
    wr = lambda rr: sum(x["visible"] + x["cross"] + x["phantom"] for x in rr)
    cost = lambda rr: (4 * sum(x["miss"] for x in rr) + 2 * wr(rr)) / len(rr)
    r0 = [row(st, None) for st in allc]
    L += [f"v1.7: {hits(r0)} hits, {wr(r0)} wrong, cost {cost(r0):.3f}", "", "| feature | bar | hits | wrong | cost |", "|---|---|---|---|---|"]
    for g in grid[1:]:
        rr = [row(st, g) for st in allc]
        if wr(rr) < wr(r0):
            L.append(f"| {g[0]} | {g[1]} | {hits(rr)} | {wr(rr)} | {cost(rr):.3f} |")
    rng = random.Random(0); sh = allc[:]; rng.shuffle(sh); oof, ch = {}, []
    for k in range(5):
        te = set(sh[k::5]); tr = [c for c in allc if c not in te]
        best = min(grid, key=lambda g: cost([row(st, g) for st in tr])); ch.append(best)
        for st in te:
            oof[st] = row(st, best)
    rr = [oof[st] for st in allc]
    L += ["", f"CV choices {ch}; out of fold {hits(rr)} hits, {wr(rr)} wrong, cost {cost(rr):.3f}",
          f"Pass (0 hits lost, >= 2 wrongs removed out of fold): {'YES' if hits(rr) >= hits(r0) and wr(r0) - wr(rr) >= 2 else 'NO'}", ""]

    # B. proposer ceiling
    need = [g for g in F["gold"] if g["needed"] and g["importance"] >= 2]
    miss = [g for g in need if not g["hit"]]
    L += ["## B. BEATs-strong as an extra detector (upper bound)", "",
          f"needed sounds {len(need)}; v1.7 misses {len(miss)}; of these, {sum(g['has_family'] for g in miss)} have a family class among the 447.", "",
          "| bar t | recoverable misses | runs | extra runs where an on-screen sound of the family plays | other extra runs | 4 x rec - 2 x other | pass |", "|---|---|---|---|---|---|---|"]
    for t in ("0.2", "0.3", "0.4", "0.5", "0.6"):
        rec = 0; on_scr = other = n = 0
        for st in allc:
            runs = F["runs"][st][t]; n += len(runs)
            for g in [g for g in miss if g["clip"] == st]:
                rec += any(S.same_family(r[0], g["label"]) and S.in_window(r[1], g["start"], S.EARLY, S.LATE) for r in runs)
            for lab, a, b, mx in runs:
                gs = [g for g in gold[st] if S.same_family(lab, g["label"])]
                if any(S.in_window(a, g["start"], S.EARLY, S.LATE) and g["needed"] for g in gs):
                    continue
                if any(S.same_family(lab, p[0]) and p[1] - 0.5 <= a <= p[2] for p in pics[st]):
                    continue
                if any((S.in_window(a, g["start"], S.EARLY, S.LATE) or g["start"] <= a <= g["end"]) and not g["needed"] for g in gs):
                    on_scr += 1
                else:
                    other += 1
        L.append(f"| {t} | {rec} | {n} | {on_scr} | {other} | {4 * rec - 2 * other} | {'YES' if rec >= 8 and 4 * rec - 2 * other > 30 else 'no'} |")
    L += ["", "Recoverable: a v1.7-missed needed sound with a BEATs-strong run of its family starting -0.5..+1.0 s of its onset.",
          "Extra runs: runs that match no needed onset and no v1.7 picture of their family; before any veto or gate (upper bound of wrongs)."]
    (HERE / "as158_analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
