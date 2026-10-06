"""Step 4 (PREREG_step4_keep_score.md): one keep-score from all saved answers, clip-grouped 10-fold CV on DEV. CPU, local.

    python benchmark/gold/coverage/keep_score.py   -> keep_score_dev.json, keep_score_dev.md
"""
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.coverage import score_coverage as V

HERE = Path(__file__).resolve().parent
FEATS = ["beats_max", "beats_mean", "flex_max", "flex_mean", "dasm_max", "beats_clip", "flex_clip", "dasm_clip", "speech_music",
         "length", "n_members", "o_beats", "o_flexsed", "o_band", "o_dasm", "listener_q", "af_v4", "af_yn",
         "gate_maj", "gate_ab", "omni_p_own", "omni_p_none", "omni_onset_visible", "omni_still_heard"]
AB = "SHIP8+MD3+WW5+SL|AB-m"


def table():
    it = json.loads((HERE / "verify_items_dev.json").read_text(encoding="utf-8"))
    bf = json.loads((HERE / "burst_features_dev.json").read_text(encoding="utf-8"))
    om = json.loads((HERE / "omni_verify_dev.json").read_text(encoding="utf-8"))
    gd = json.loads((HERE / "gate_dev.json").read_text(encoding="utf-8"))
    secs = {}
    for s in it["seconds"]:
        if s["id"] in om["v3"]:
            secs.setdefault(s["burst"], []).append((s["t"], om["v3"][s["id"]]))
    rows = []
    for b in it["bursts"]:
        f = dict(bf[b["id"]])
        f.update(length=b["end"] - b["start"], n_members=len(b["members"]), o_beats="beats" in b["origins"],
                 o_flexsed="flexsed" in b["origins"], o_band="flexsed band" in b["origins"], o_dasm="dasm" in b["origins"])
        rec = [g for g in gd[b["clip"]]["gate"] if S.same_family(g["label"], b["family"]) and g["end"] > b["start"] and g["start"] < b["end"]]
        st = [v for g in rec for v in g["stretches"]]
        f["gate_maj"] = float(np.mean([v["seen"] for v in st])) if st else None
        f["gate_ab"] = float(np.mean([(v["ab"] if v["ab"] is not None else v["seen"]) for v in st])) if st else None
        r = om["v1"][b["id"]]
        f["omni_p_own"] = float(np.mean([r[o]["p"][r[o]["options"].index(b["family"])] for o in ("fwd", "rev")]))
        f["omni_p_none"] = float(np.mean([r[o]["p"][-1] for o in ("fwd", "rev")]))
        f["omni_onset_visible"] = om["v2"].get(b["id"])
        inside = [p for t, p in secs.get(b["id"], []) if t < b["end"]] or [p for _, p in secs.get(b["id"], [])[:1]]
        f["omni_still_heard"] = float(np.mean(inside)) if inside else None
        rows.append((b, [np.nan if f[k] is None else float(f[k]) for k in FEATS]))
    return rows


def good(b, gold):
    return any(S.same_family(x["label"], b["family"]) and x["needed"] and x["importance"] >= S.MIN_IMPORTANCE
               and S.in_window(b["start"], x["start"], S.EARLY, S.LATE) for x in gold[b["clip"]])


def oof(X, y, groups, kind):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=10).split(X, y, groups):
        if kind == "trees":
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20, random_state=0)
            m.fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
        else:
            med = np.nanmedian(X[tr], axis=0); med = np.where(np.isnan(med), 0.0, med)
            fill = lambda Z: np.where(np.isnan(Z), med, Z)
            mu, sd = fill(X[tr]).mean(0), fill(X[tr]).std(0) + 1e-9
            m = LogisticRegression(C=1.0, max_iter=2000)
            m.fit((fill(X[tr]) - mu) / sd, y[tr]); p[te] = m.predict_proba((fill(X[te]) - mu) / sd)[:, 1]
    return p


def score(pics_by_clip, gold, dev):
    return V.aggregate([V.score_clip_v2(gold[st], pics_by_clip.get(st, [])) for st in dev])


def curve(make, probs, gold, dev):
    ths = sorted(set(np.round(probs, 6)) | {0.0, 1.0 / 3, 1.01})
    pts = []
    for t in ths:
        a = score(make(t), gold, dev)
        pts.append({"th": float(t), "hits": a["hits"], "wrong": a["wrong"], "onset_cost": a["onset_cost"], "cost_cov": a["cost_cov"]})
    return pts


def at_wrong(pts, w):
    ok = [p for p in pts if p["wrong"] <= w]
    return max(ok, key=lambda p: (p["hits"], -p["wrong"])) if ok else None


def main():
    gold = S.load_gold([V.GOLD])
    dev = V.stems("dev")
    rows = table()
    B = [b for b, _ in rows]
    X = np.array([x for _, x in rows])
    y = np.array([good(b, gold) for b in B])
    groups = np.array([b["clip"] for b in B])
    abp = json.loads((HERE / "step2_pics_dev.json").read_text(encoding="utf-8"))["cells"][AB]["clips"]
    res = {"n": len(B), "good": int(y.sum()), "features": FEATS, "models": {}}
    L = [f"# Step 4: keep-score, DEV (71 clips), clip-grouped 10-fold CV", "", f"{len(B)} bursts, {int(y.sum())} good (would be hits).", ""]
    # representation baseline of (b): the bursts the frozen system draws, rendered as bursts
    rep = {}
    for b in B:
        if b["drawn"]:
            rep.setdefault(b["clip"], []).append((b["family"], b["start"], b["end"]))
    ra = score(rep, gold, dev)
    L += [f"Variant (b) representation baseline (bursts the frozen system draws, as bursts): {ra['hits']} hits / {ra['wrong']} wrong, "
          f"onset cost {ra['onset_cost']:.3f}, cost_cov {ra['cost_cov']:.3f}", ""]
    res["rep_baseline"] = ra
    L += ["| model | variant | AUROC (good) | hits @ wrong<=5 | @ <=10 | @ <=15 | at p > 1/3: hits / wrong / onset cost / cost_cov |",
          "|---|---|---|---|---|---|---|"]
    from benchmark.gold.coverage.omni_probe import auroc
    for kind in ("trees", "logistic"):
        p = oof(X, y, groups, kind)
        bp = {b["id"]: float(pp) for b, pp in zip(B, p)}

        def pic_p(st, pic):
            m = [bp[b["id"]] for b in B if b["clip"] == st and S.same_family(pic[0], b["family"]) and b["end"] > pic[1] - 0.5 and b["start"] < pic[2] + 0.5]
            return max(m) if m else 1.0

        ab = {st: [(tuple(pc), pic_p(st, pc)) for pc in abp[st]["pics_none"]] for st in dev}
        make_a = lambda t: {st: [pc for pc, pp in v if pp > t] for st, v in ab.items()}
        byc = {}
        for b in B:
            byc.setdefault(b["clip"], []).append(((b["family"], b["start"], b["end"]), bp[b["id"]]))
        make_b = lambda t: {st: [pc for pc, pp in byc.get(st, []) if pp > t] for st in dev}
        res["models"][kind] = {"auroc": auroc(p, y), "unmatched_ab_pictures": sum(1 for v in ab.values() for _, pp in v if pp == 1.0)}
        for var, mk in (("a reorder-only", make_a), ("b replace vetoes", make_b)):
            pts = curve(mk, p if var.startswith("b") else np.array([pp for v in ab.values() for _, pp in v]), gold, dev)
            third = score(mk(1.0 / 3), gold, dev)
            w = {k: at_wrong(pts, k) for k in (5, 10, 15)}
            res["models"][kind][var] = {"curve": pts, "at_wrong": w, "p_third": third}
            fw = lambda q: "-" if q is None else f"{q['hits']} ({q['wrong']} w)"
            L.append(f"| {kind} | {var} | {res['models'][kind]['auroc']:.3f} | {fw(w[5])} | {fw(w[10])} | {fw(w[15])} | "
                     f"{third['hits']} / {third['wrong']} / {third['onset_cost']:.3f} / {third['cost_cov']:.3f} |")
    # clear-win check against the a/b candidate (32 hits / 16 wrong)
    win = []
    for kind, m in res["models"].items():
        for var in ("a reorder-only", "b replace vetoes"):
            for q in m[var]["curve"]:
                if (q["hits"] >= 35 and q["wrong"] <= 16) or (q["hits"] >= 32 and q["wrong"] <= 13):
                    win.append((kind, var, q["hits"], q["wrong"], round(q["th"], 3)))
    res["clear_wins"] = win
    L += ["", "Reference: frozen 29 hits / 15 wrong (onset 2.056, cost_cov 2.509); a/b candidate 32 / 16 (1.915, 2.421).",
          "Clear win = >= 35 hits at <= 16 wrong, or >= 32 hits at <= 13 wrong (any threshold on the CV curve).",
          f"Clear wins: {win if win else 'NONE -- the keep-score does not beat the a/b candidate by a clear margin.'}"]
    (HERE / "keep_score_dev.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    (HERE / "keep_score_dev.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
