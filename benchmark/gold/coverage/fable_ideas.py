"""Five display / selection ideas (Adam 10 Oct, in this order), each on top of the current system and of the ideas
before it that helped, all chosen by clip-grouped 5-fold CV on the 158 clips (seed 0) and reported out of fold.

Current system (CURRENT_SYSTEM.md) before hold: a/b gate + flash + texture ban; then the adopted hold.
1. restart at each raw onset: a picture is split at every raw candidate onset of its family inside it that is >= g s after
   the last start. g in {off, 1, 2, 3}; candidates: those the frozen chain drew, or any.
2. evidence-bound ends: the end rule (pooled_hold.hold) in {adopted (FlexSED 0.5, extend), FlexSED 0.5 / 0.4 "both" with
   tau 0 / 0.5}. Chosen by J; the other ideas by onset cost.
3. parent picture on sibling disputes: when a raw candidate of another family starts within w s of a picture's start and
   both share a parent that is not a top-level category, the picture shows the parent. w in {off, 0.5, 1.0}.
4. within-clip contrast dropper: logistic regression on each picture's burst features plus their z-score inside the clip;
   trained on hit pictures vs other-sound / no-sound pictures only; a picture is removed below bar b (chosen on the
   training clips). Needs the TEST Omni answers; skipped if they are missing.
5. best-timed guess: a picture's start becomes the onset of one raw candidate of its family within [start - 0.5,
   start + 2]: {current (earliest), strongest peak, first FlexSED onset}.

    python benchmark/gold/coverage/fable_ideas.py   (cluster CPU from ~/wt_slice) -> fable_ideas.md
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
from benchmark.gold.coverage.pooled_hold import hold, J
from benchmark.gold.coverage.step11_policies import policy, TEXTURE
from src.labels import _common_parent, canonical, ancestors

HERE = Path(__file__).resolve().parent
ADOPTED_HOLD = (0.5, None, False, 0.0, "extend")
TOP = {"Sounds of things", "Animal", "Natural sounds", "Human sounds", "Source-ambiguous sounds", "Channel, environment and background", "Music"}


def peak_of(c):
    for r in c.get("trail", []):
        m = re.search(r"peak (\d+(?:\.\d+)?)", str(r.get("value", "")))
        if m:
            return float(m.group(1))
    return 0.0


def main():
    gold = S.load_gold([V.GOLD]); dev, test = V.stems("dev"), V.stems("test"); allc = dev + test
    s = (_ROOT / "docs" / "decision_trail" / "data.js").read_text(encoding="utf-8")
    dj = json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))
    cands = {c["clip"]: c["cands"] for c in dj["clips"]}
    Dc = json.loads((HERE / "step2_pics_dev.json").read_text(encoding="utf-8"))["cells"]["SHIP8+MD3+WW5+SL|AB-m"]["clips"]
    Tc = json.loads((HERE / "final_pics_test.json").read_text(encoding="utf-8"))["cells"]["SHIP8+MD3+WW5+SL|AB-m"]["clips"]
    fl = {**json.loads((HERE / "flashes_dev.json").read_text()), **json.loads((HERE / "flashes_test.json").read_text())}
    ev = {**json.loads((HERE / "evidence_dev_all.json").read_text()), **json.loads((HERE / "evidence_test_all.json").read_text())}
    dur = {}
    for st in allc:
        ts = [float(m["t"][-1]) for c in ev.get(st, {}).get("labels", {}).values() for m in (c or {}).values() if m and m.get("t")]
        dur[st] = max(ts) if ts else 10.0
    base = {st: [x for x in policy(st, [tuple(p) for p in (Dc if st in dev else Tc)[st]["pics_none"]], "F", {}, {}, fl) if x[0] not in TEXTURE]
            for st in allc}

    # ---- the five transformations (pictures before hold unless stated)
    def restart(st, pics, prm):
        if prm is None:
            return pics
        g, src = prm
        out = []
        for lab, a, b in pics:
            ons = sorted(c["start"] for c in cands[st] if S.same_family(c["label"], lab) and a < c["start"] < b
                         and (src == "any" or c.get("fate") == "drawn"))
            starts = [a]
            for o in ons:
                if o - starts[-1] >= g:
                    starts.append(o)
            for i, s0 in enumerate(starts):
                out.append((lab, s0, starts[i + 1] if i + 1 < len(starts) else b))
        return out

    def parent(st, pics, w):
        if w is None:
            return pics
        out = []
        for lab, a, b in pics:
            new = lab
            for c in cands[st]:
                if abs(c["start"] - a) <= w and canonical(c["label"]) != canonical(lab) and not S.same_family(c["label"], lab):
                    cp = _common_parent(lab, c["label"])
                    if cp and cp not in TOP and len(ancestors(cp)) >= 1:
                        new = cp; break
            out.append((new, a, b))
        return out

    def best_time(st, pics, mode):
        if mode == "current":
            return pics
        out = []
        for lab, a, b in pics:
            cs = [c for c in cands[st] if S.same_family(c["label"], lab) and a - 0.5 <= c["start"] <= a + 2.0]
            if mode == "strongest" and cs:
                a2 = max(cs, key=peak_of)["start"]
            elif mode == "flexfirst" and [c for c in cs if c["origin"].startswith("flexsed")]:
                a2 = min(c["start"] for c in cs if c["origin"].startswith("flexsed"))
            else:
                a2 = a
            out.append((lab, a2, max(b, a2 + 0.5)))
        return out

    # contrast dropper inputs
    have_omni = (HERE / "omni_verify_test.json").exists()
    feat_of = {}
    if have_omni:
        from benchmark.gold.coverage.pooled_keep import table
        rows = table("dev") + [r for r in table("test") if r[0]["clip"] in set(test)]
        for b, x in rows:
            feat_of.setdefault(b["clip"], []).append((b, np.array(x, float)))

    def pic_feats(st, pics):
        F = []
        for lab, a, b in pics:
            m = [x for bb, x in feat_of.get(st, []) if S.same_family(lab, bb["family"]) and bb["end"] > a - 0.5 and bb["start"] < b + 0.5]
            F.append(np.nanmax(np.vstack(m), axis=0) if m else np.full(len(K.FEATS), np.nan))
        F = np.array(F) if F else np.zeros((0, len(K.FEATS)))
        if len(F):
            mu, sd = np.nanmean(F, 0), np.nanstd(F, 0) + 1e-6
            F = np.hstack([F, (F - mu) / sd, np.full((len(F), 1), len(F))])
        return F

    def pipeline(st, cfg, drop=None):
        p = base[st]
        p = restart(st, p, cfg["restart"])
        p = parent(st, p, cfg["parent"])
        p = best_time(st, p, cfg["time"])
        if drop is not None and p:
            keep = drop(st, p)
            p = [x for x, k in zip(p, keep) if k]
        p = sorted(p, key=lambda x: x[1])
        return V.score_clip_v2(gold[st], hold(p, ev.get(st, {"labels": {}}), cfg["hold"], dur[st]))

    rng = random.Random(0); sh = allc[:]; rng.shuffle(sh)
    folds = [set(sh[k::5]) for k in range(5)]
    cfg0 = {"restart": None, "parent": None, "time": "current", "hold": ADOPTED_HOLD}
    L = ["# Five ideas, in order, clip-grouped 5-fold CV on all 158 clips (out of fold)", ""]

    def cv(key, options, crit):
        """choose cfg[key] per fold on the training clips; return out-of-fold rows and choices"""
        rows, chosen = {}, []
        for te in folds:
            tr = [c for c in allc if c not in te]
            def score(o):
                cfg = dict(cur); cfg[key] = o
                rr = [pipeline(st, cfg) for st in tr]
                return J(rr)[0] if crit == "J" else V.aggregate(rr)["onset_cost"]
            best = min(options, key=score)
            chosen.append(best)
            cfg = dict(cur); cfg[key] = best
            for st in te:
                rows[st] = pipeline(st, cfg)
        return [rows[st] for st in allc], chosen

    cur = dict(cfg0)
    ref = [pipeline(st, cur) for st in allc]
    jr, ar = J(ref)
    L.append(f"Current system: {ar['hits']} hits / {ar['wrong']} wrong, onset cost {ar['onset_cost']:.3f}, J {jr:.3f}, cover {ar['hit_cov']:.2f}, "
             f"|end err| {ar['end_abs_med']:.2f} s, start err {ar['start_med']:+.2f} s")
    L += ["", "| idea | options | chosen per fold | out-of-fold hits / wrong | onset cost | J | adopted? |", "|---|---|---|---|---|---|---|"]
    ideas = [("1 restart at raw onsets", "restart", [None, (1.0, "drawn"), (2.0, "drawn"), (3.0, "drawn"), (1.0, "any"), (2.0, "any"), (3.0, "any")], "cost"),
             ("2 evidence-bound ends", "hold", [ADOPTED_HOLD, (0.5, None, False, 0.0, "both"), (0.5, None, False, 0.5, "both"),
                                                (0.4, None, False, 0.0, "both"), (0.4, None, False, 0.5, "both")], "J"),
             ("3 parent on sibling disputes", "parent", [None, 0.5, 1.0], "cost"),
             ("5 best-timed guess", "time", ["current", "strongest", "flexfirst"], "cost")]
    for name, key, opts, crit in ideas:
        rr, ch = cv(key, opts, crit)
        j, a = J(rr)
        better = (j < jr - 1e-9) if crit == "J" else (a["onset_cost"] < ar["onset_cost"] - 1e-9)
        L.append(f"| {name} | {opts} | {ch} | {a['hits']} / {a['wrong']} | {a['onset_cost']:.3f} | {j:.3f} | {'yes' if better else 'no'} |")
        if better:
            from collections import Counter
            cur[key] = Counter(map(str, ch)).most_common(1)[0][0]
            cur[key] = next(o for o in opts if str(o) == cur[key])
            ref = [pipeline(st, cur) for st in allc]; jr, ar = J(ref)
    # idea 4: contrast dropper
    if have_omni:
        from sklearn.linear_model import LogisticRegression
        rows, bars = {}, []
        for te in folds:
            tr = [c for c in allc if c not in te]
            Xs, ys = [], []
            for st in tr:
                p = sorted(best_time(st, parent(st, restart(st, base[st], cur["restart"]), cur["parent"]), cur["time"]), key=lambda x: x[1])
                if not p:
                    continue
                F = pic_feats(st, p)
                from benchmark.gold.inspector_data import classify
                _, pp = classify(gold[st], p)
                for f, q in zip(F, pp):
                    if q["class"].startswith("hit"):
                        Xs.append(f); ys.append(1)
                    elif q["class"] in ("wrong: a different sound", "wrong: no such sound"):
                        Xs.append(f); ys.append(0)
            Xs = np.array(Xs); med = np.nanmedian(Xs, 0); med = np.where(np.isnan(med), 0, med)
            fill = lambda Z: np.where(np.isnan(Z), med, Z)
            mu, sd = fill(Xs).mean(0), fill(Xs).std(0) + 1e-6
            m = LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced").fit((fill(Xs) - mu) / sd, ys)
            def dropper(b):
                return lambda st, p: [pr >= b for pr in m.predict_proba((fill(pic_feats(st, p)) - mu) / sd)[:, 1]]
            bb = min([0.0, 0.1, 0.2, 0.3, 0.4], key=lambda b: V.aggregate([pipeline(st, cur, dropper(b)) for st in tr])["onset_cost"])
            bars.append(bb)
            for st in te:
                rows[st] = pipeline(st, cur, dropper(bb))
        rr = [rows[st] for st in allc]; j, a = J(rr)
        L.append(f"| 4 contrast dropper | bar 0-0.4 | {bars} | {a['hits']} / {a['wrong']} | {a['onset_cost']:.3f} | {j:.3f} | "
                 f"{'yes' if a['onset_cost'] < ar['onset_cost'] - 1e-9 else 'no'} |")
    else:
        L.append("| 4 contrast dropper | - | - | - | - | - | skipped: TEST Omni answers not ready |")
    L += ["", f"Final config: {cur}"]
    (HERE / "fable_ideas.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
