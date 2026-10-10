"""All 158 clips: Step 4's keep-score (24 features) trained on DEV + TEST bursts, used only to REMOVE pictures of the
current system (CURRENT_SYSTEM.md: a/b gate + flash + texture ban + hold). Clip-grouped 5-fold CV (seed 0): the model
and the removal bar t (chosen to minimise J on the training clips, t in 0, 0.02, ..., 0.5) never see the held-out clips.

    python benchmark/gold/coverage/pooled_keep.py   (cluster CPU from ~/wt_slice) -> pooled_keep.md
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
from benchmark.gold.coverage.step11_policies import policy, TEXTURE
from benchmark.gold.coverage.pooled_hold import hold, J

HERE = Path(__file__).resolve().parent
HOLD = (0.5, None, False, 0.0, "extend")
BARS = [round(x, 2) for x in np.arange(0.0, 0.501, 0.02)]


def table(suf):
    it = json.loads((HERE / f"verify_items_{suf}.json").read_text(encoding="utf-8"))
    bf = json.loads((HERE / f"burst_features_{suf}.json").read_text(encoding="utf-8"))
    om = json.loads((HERE / f"omni_verify_{suf}.json").read_text(encoding="utf-8"))
    gd = json.loads((HERE / f"gate_{suf}.json").read_text(encoding="utf-8"))
    secs = {}
    for s in it["seconds"]:
        if s["id"] in om["v3"]:
            secs.setdefault(s["burst"], []).append((s["t"], om["v3"][s["id"]]))
    rows = []
    for b in it["bursts"]:
        f = dict(bf[b["id"]])
        f.update(length=b["end"] - b["start"], n_members=len(b["members"]), o_beats="beats" in b["origins"],
                 o_flexsed="flexsed" in b["origins"], o_band="flexsed band" in b["origins"], o_dasm="dasm" in b["origins"])
        rec = [g for g in gd.get(b["clip"], {"gate": []})["gate"] if S.same_family(g["label"], b["family"]) and g["end"] > b["start"] and g["start"] < b["end"]]
        st = [v for g in rec for v in g["stretches"]]
        f["gate_maj"] = float(np.mean([v["seen"] for v in st])) if st else None
        f["gate_ab"] = float(np.mean([(v["ab"] if v["ab"] is not None else v["seen"]) for v in st])) if st else None
        r = om["v1"].get(b["id"])
        f["omni_p_own"] = float(np.mean([r[o]["p"][r[o]["options"].index(b["family"])] for o in ("fwd", "rev")])) if r else None
        f["omni_p_none"] = float(np.mean([r[o]["p"][-1] for o in ("fwd", "rev")])) if r else None
        f["omni_onset_visible"] = om["v2"].get(b["id"])
        inside = [p for t, p in secs.get(b["id"], []) if t < b["end"]] or [p for _, p in secs.get(b["id"], [])[:1]]
        f["omni_still_heard"] = float(np.mean(inside)) if inside else None
        rows.append((b, [np.nan if f.get(k) is None else float(f[k]) for k in K.FEATS]))
    return rows


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    gold = S.load_gold([V.GOLD])
    dev, test = V.stems("dev"), V.stems("test")
    allc = dev + test
    rows = table("dev") + [r for r in table("test") if r[0]["clip"] in set(test)]
    B = [b for b, _ in rows]
    X = np.array([x for _, x in rows]); y = np.array([K.good(b, gold) for b in B])
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
    byclip = {}
    for i, b in enumerate(B):
        byclip.setdefault(b["clip"], []).append(i)

    def final(st, p, t):
        keep = []
        for pic in base[st]:
            m = [p[i] for i in byclip.get(st, []) if S.same_family(pic[0], B[i]["family"]) and B[i]["end"] > pic[1] - 0.5 and B[i]["start"] < pic[2] + 0.5]
            if not m or max(m) >= t:
                keep.append(pic)
        return V.score_clip_v2(gold[st], hold(keep, ev.get(st, {"labels": {}}), HOLD, dur[st]))

    rng = random.Random(0)
    sh = allc[:]; rng.shuffle(sh)
    oos = {}
    L = ["# All 158 clips: keep-score trained on DEV + TEST bursts, used only to remove current-system pictures", "",
         "| fold | bar t chosen on training clips | held-out J: current -> filtered | hits / wrong |", "|---|---|---|---|"]
    for k in range(5):
        te = set(sh[k::5]); tr = [c for c in allc if c not in te]
        tri = [i for i, b in enumerate(B) if b["clip"] not in te]
        # inner out-of-fold probabilities on the training clips (to choose t without fitting t on the model's own fit)
        g = np.array([B[i]["clip"] for i in tri])
        p_in = np.zeros(len(B))
        from sklearn.model_selection import GroupKFold
        for a, c in GroupKFold(5).split(X[tri], y[tri], g):
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20, random_state=0)
            m.fit(X[tri][a], y[tri][a]); p_in[np.array(tri)[c]] = m.predict_proba(X[tri][c])[:, 1]
        tbest = min(BARS, key=lambda t: J([final(st, p_in, t) for st in tr])[0])
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20, random_state=0)
        m.fit(X[tri], y[tri])
        p = np.zeros(len(B)); tei = [i for i, b in enumerate(B) if b["clip"] in te]
        p[tei] = m.predict_proba(X[tei])[:, 1]
        r0 = [final(st, p, 0.0) for st in te]; r1 = [final(st, p, tbest) for st in te]
        for st, r in zip(te, r1):
            oos[st] = (final(st, p, 0.0), r)
        j0, a0 = J(r0); j1, a1 = J(r1)
        L.append(f"| {k + 1}/5 | {tbest} | {j0:.3f} -> {j1:.3f} | {a0['hits']}/{a0['wrong']} -> {a1['hits']}/{a1['wrong']} |")
    j0, a0 = J([oos[st][0] for st in allc]); j1, a1 = J([oos[st][1] for st in allc])
    L += ["", f"All 158 clips out of sample: J {j0:.3f} -> {j1:.3f}; hits/wrong {a0['hits']}/{a0['wrong']} -> {a1['hits']}/{a1['wrong']}; "
          f"onset cost {a0['onset_cost']:.3f} -> {a1['onset_cost']:.3f}; cost_cov {a0['cost_cov']:.3f} -> {a1['cost_cov']:.3f}"]
    (HERE / "pooled_keep.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
