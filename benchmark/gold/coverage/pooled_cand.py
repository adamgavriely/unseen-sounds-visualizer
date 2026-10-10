"""All 158 clips: finder + dropper on RAW candidates (their own starts), Adam + Fable 10 Oct.

Finder: every candidate in the decision trail (BEATs / FlexSED / band / DASM; Speech / Music left out).
Dropper: gradient-boosted trees; features = the candidate's own (origin, peak, length, start in clip, whether the frozen
rule chain drew it, which rule step dropped it) + the 24 Step-4 features of the burst it belongs to + flash at its start
+ texture flag + its family's hit rate on the training clips. Kept candidates of one family closer than 2.5 s join one
picture (earliest kept start, latest end); then the adopted hold. Clip-grouped 5-fold CV (seed 0), all out of fold.
Reported: the hits / wrong curve over the bar; importance 2-3 and importance ignored; next to the current system.

    python benchmark/gold/coverage/pooled_cand.py   (cluster CPU from ~/wt_slice) -> pooled_cand.md
"""
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.coverage import score_coverage as V
from benchmark.gold.coverage import keep_score as K
from benchmark.gold.coverage.pooled_keep import table
from benchmark.gold.coverage.pooled_hold import hold
from benchmark.gold.coverage.step11_policies import TEXTURE
from src.labels import canonical, is_music

HERE = Path(__file__).resolve().parent
HOLD = (0.5, None, False, 0.0, "extend")
ORIG = ["beats", "flexsed", "flexsed band", "dasm"]
GAP = 2.5


def peak_of(c):
    for r in c.get("trail", []):
        m = re.search(r"peak (\d+(?:\.\d+)?)", str(r.get("value", "")))
        if m:
            return float(m.group(1))
    return np.nan


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    gold = S.load_gold([V.GOLD]); dev, test = V.stems("dev"), V.stems("test"); allc = dev + test
    s = (_ROOT / "docs" / "decision_trail" / "data.js").read_text(encoding="utf-8")
    dj = json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))
    burst_rows = {b["id"] + "|" + b["clip"]: x for b, x in table("dev") + table("test")}
    mem = {}
    for suf in ("dev", "test"):
        for b in json.loads((HERE / f"verify_items_{suf}.json").read_text(encoding="utf-8"))["bursts"]:
            for m in b["members"]:
                mem[(b["clip"], m)] = b["id"] + "|" + b["clip"]
    fl = {**json.loads((HERE / "flashes_dev.json").read_text()), **json.loads((HERE / "flashes_test.json").read_text())}
    ev = {**json.loads((HERE / "evidence_dev_all.json").read_text()), **json.loads((HERE / "evidence_test_all.json").read_text())}
    steps = sorted({c.get("at") or "" for cl in dj["clips"] for c in cl["cands"]})
    C, F = [], []
    for cl in dj["clips"]:
        st = cl["clip"]
        if st not in set(allc):
            continue
        dur = float(cl.get("dur") or 10.0)
        for c in cl["cands"]:
            fam = canonical(c["label"])
            if fam in ("Speech", "Music") or is_music(c["label"]):
                continue
            bx = burst_rows.get(mem.get((st, c["id"])), [np.nan] * len(K.FEATS))
            f = [ORIG.index(c["origin"]) if c["origin"] in ORIG else -1, peak_of(c), c["end"] - c["start"], c["start"] / max(dur, 1.0),
                 float(c.get("fate") == "drawn"), steps.index(c.get("at") or ""),
                 float(any(abs(t - c["start"]) <= 0.3 for t in fl.get(st, {}).get("flashes", []))), float(fam in TEXTURE)] + list(bx)
            C.append({"clip": st, "family": fam, "start": float(c["start"]), "end": float(c["end"]), "dur": dur}); F.append(f)
    X0 = np.array(F, dtype=float)
    fams = [c["family"] for c in C]
    byclip = {}
    for i, c in enumerate(C):
        byclip.setdefault(c["clip"], []).append(i)
    L = ["# All 158 clips: finder (raw candidates) + learned dropper, out of fold", "",
         f"{len(C)} candidates. Current system (CURRENT_SYSTEM.md): 59 hits / 34 wrong, onset cost 2.076 (importance 2-3).", ""]
    for imp in (2, 1):
        S.MIN_IMPORTANCE = imp
        y = np.array([any(S.same_family(x["label"], c["family"]) and x["needed"] and x["importance"] >= imp
                          and S.in_window(c["start"], x["start"], S.EARLY, S.LATE) for x in gold[c["clip"]]) for c in C])
        rng = random.Random(0); sh = allc[:]; rng.shuffle(sh)
        p = np.zeros(len(C))
        for k in range(5):
            te = set(sh[k::5]); tr = np.array([c["clip"] not in te for c in C])
            rate = {}
            for f in set(fams):
                idx = [i for i in range(len(C)) if tr[i] and fams[i] == f]
                rate[f] = (y[idx].sum() + 1) / (len(idx) + 10)
            X = np.hstack([X0, np.array([rate.get(f, 0.1) for f in fams])[:, None]])
            m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, min_samples_leaf=20, random_state=0)
            m.fit(X[tr], y[tr]); p[~tr] = m.predict_proba(X[~tr])[:, 1]
        pts = []
        for t in np.unique(np.round(np.quantile(p, np.linspace(0.80, 0.999, 70)), 4)):
            rr = []
            for st in allc:
                kept = sorted((C[i] for i in byclip.get(st, []) if p[i] >= t), key=lambda c: (c["family"], c["start"]))
                pics, cur = [], None
                for c in kept:
                    if cur and cur[0] == c["family"] and c["start"] - cur[2] <= GAP:
                        cur[2] = max(cur[2], c["end"])
                    else:
                        cur = [c["family"], c["start"], c["end"]]; pics.append(cur)
                pics = sorted((tuple(x) for x in pics), key=lambda x: x[1])
                rr.append(V.score_clip_v2(gold[st], hold(pics, ev.get(st, {"labels": {}}), HOLD, C[byclip[st][0]]["dur"] if st in byclip else 10.0)))
            a = V.aggregate(rr)
            pts.append((float(t), a["hits"], a["needed"], a["wrong"], a["onset_cost"], a["cost_cov"]))
        L += [f"## needed = importance >= {imp} ({pts[0][2]} sounds; candidates in window: {int(y.sum())})", "",
              "| bar | hits | wrong | onset cost | cost_cov |", "|---|---|---|---|---|"]
        for q in pts[::3]:
            L.append(f"| {q[0]:.3f} | {q[1]} | {q[3]} | {q[4]:.3f} | {q[5]:.3f} |")
        best = min(pts, key=lambda q: q[4])
        L += ["", f"Lowest onset cost: {best[1]} hits / {best[3]} wrong, cost {best[4]:.3f} (bar {best[0]:.3f}).",
              "Most hits at <= 34 wrong: " + str(max(((q[1], q[3]) for q in pts if q[3] <= 34), default=None)) +
              "; fewest wrong at >= 59 hits: " + str(min(((q[3], q[1]) for q in pts if q[1] >= 59), default=None)), ""]
    S.MIN_IMPORTANCE = 2
    (HERE / "pooled_cand.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
