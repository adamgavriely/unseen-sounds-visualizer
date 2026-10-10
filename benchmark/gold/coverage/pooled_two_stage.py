"""All 158 clips (Adam 10 Oct): finder + dropper. Finder = every candidate burst (BEATs, FlexSED, band, DASM; Step 3b
bursts). Dropper = gradient-boosted trees on the 24 Step-4 features + flash at the onset (Step 11) + texture flag + the
family's hit rate on the training clips (target encoding, inside the fold). Pictures = kept bursts (family, start, end),
with the adopted hold. Clip-grouped 5-fold CV (seed 0): every probability is out of fold. Curve over the bar t:
hits vs wrong; scored with importance 2-3 (as always) and with importance ignored (all needed sounds count).

    python benchmark/gold/coverage/pooled_two_stage.py   (cluster CPU from ~/wt_slice) -> pooled_two_stage.md
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.coverage import score_coverage as V
from benchmark.gold.coverage import keep_score as K
from benchmark.gold.coverage.pooled_keep import table
from benchmark.gold.coverage.pooled_hold import hold, J
from benchmark.gold.coverage.step11_policies import TEXTURE

HERE = Path(__file__).resolve().parent
HOLD = (0.5, None, False, 0.0, "extend")


def good(b, gold, imp):
    return any(S.same_family(x["label"], b["family"]) and x["needed"] and x["importance"] >= imp
               and S.in_window(b["start"], x["start"], S.EARLY, S.LATE) for x in gold[b["clip"]])


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    gold = S.load_gold([V.GOLD])
    dev, test = V.stems("dev"), V.stems("test")
    allc = dev + test
    rows = table("dev") + [r for r in table("test") if r[0]["clip"] in set(test)]
    B = [b for b, _ in rows]
    fl = {**json.loads((HERE / "flashes_dev.json").read_text()), **json.loads((HERE / "flashes_test.json").read_text())}
    ev = {**json.loads((HERE / "evidence_dev_all.json").read_text()), **json.loads((HERE / "evidence_test_all.json").read_text())}
    X0 = np.array([x for _, x in rows])
    flash = np.array([float(any(abs(t - b["start"]) <= 0.3 for t in fl.get(b["clip"], {}).get("flashes", []))) for b in B])
    tex = np.array([float(b["family"] in TEXTURE) for b in B])
    fams = [b["family"] for b in B]
    dur = {}
    for st in allc:
        ts = [float(m["t"][-1]) for c in ev.get(st, {}).get("labels", {}).values() for m in (c or {}).values() if m and m.get("t")]
        dur[st] = max(ts) if ts else 10.0
    out = {}
    for imp in (2, 1):
        S.MIN_IMPORTANCE = imp
        y = np.array([good(b, gold, imp) for b in B])
        rng = random.Random(0); sh = allc[:]; rng.shuffle(sh)
        p = np.zeros(len(B))
        for k in range(5):
            te = set(sh[k::5])
            tr = np.array([b["clip"] not in te for b in B])
            rate = {}
            for f in set(fams):
                idx = [i for i in range(len(B)) if tr[i] and fams[i] == f]
                rate[f] = (y[idx].sum() + 1) / (len(idx) + 10)            # smoothed hit rate of the family, training only
            fr = np.array([rate.get(f, 1 / 10) for f in fams])
            X = np.hstack([X0, flash[:, None], tex[:, None], fr[:, None]])
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=20, random_state=0)
            m.fit(X[tr], y[tr]); p[~tr] = m.predict_proba(X[~tr])[:, 1]
        byclip = {}
        for i, b in enumerate(B):
            byclip.setdefault(b["clip"], []).append(i)
        pts = []
        for t in np.unique(np.round(np.quantile(p, np.linspace(0.5, 0.999, 80)), 4)):
            rr = []
            for st in allc:
                pics = sorted([(B[i]["family"], B[i]["start"], B[i]["end"]) for i in byclip.get(st, []) if p[i] >= t], key=lambda x: x[1])
                rr.append(V.score_clip_v2(gold[st], hold(pics, ev.get(st, {"labels": {}}), HOLD, dur[st])))
            a = V.aggregate(rr)
            pts.append((float(t), a["hits"], a["needed"], a["wrong"], a["onset_cost"], a["cost_cov"]))
        out[imp] = pts
    S.MIN_IMPORTANCE = 2
    L = ["# All 158 clips: finder (every candidate) + learned dropper, out-of-fold", ""]
    for imp, pts in out.items():
        L += [f"## needed sounds = importance >= {imp} ({pts[0][2]} sounds)", "",
              "| bar | hits | wrong | onset cost | cost_cov |", "|---|---|---|---|---|"]
        for t, h, n, w, oc, cc in pts[::4]:
            L.append(f"| {t:.3f} | {h} | {w} | {oc:.3f} | {cc:.3f} |")
        best = min(pts, key=lambda q: q[4])
        L += ["", f"Lowest onset cost: bar {best[0]:.3f}: {best[1]} hits / {best[3]} wrong, cost {best[4]:.3f}. "
              "Most hits at <= 34 wrong: " + str(max(((q[1], q[3]) for q in pts if q[3] <= 34), default=None)), ""]
    (HERE / "pooled_two_stage.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
