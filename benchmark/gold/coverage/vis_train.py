"""Step 12 v1 (PREREG_step12_visible_source_head.md): train the visibility head on AVATAR, check bar A on DEV, then bar B.
Cluster CPU, run from ~/wt_slice (repo slice with gold), features from ~/MscProj_tg/scratch_cov/s12/feats_items_*of4.json.

    python benchmark/gold/coverage/vis_train.py   -> benchmark/gold/coverage/vis_head_dev.md / .json
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.coverage import score_coverage as V
from benchmark.gold.coverage.omni_probe import auroc

HERE = Path(__file__).resolve().parent
S12 = Path.home() / "MscProj_tg" / "scratch_cov" / "s12"
F = ["h0_max", "h0_mean", "h0_peak", "h1_max", "h1_mean", "h1_peak", "box_score", "box_in_out", "motion_in_out", "flash"]


def vec(f):
    return [np.nan if f.get(k) is None else float(f[k]) for k in F]


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    items = {x["k"]: x for x in json.loads((S12 / "items.json").read_text(encoding="utf-8"))}
    feats = {}
    for f in glob.glob(str(S12 / "feats_items_*of4.json")):
        feats.update({int(k): v for k, v in json.loads(Path(f).read_text()).items() if "error" not in v})
    tr = [k for k in feats if items[k]["src"] == "avatar"]
    Xtr = np.array([vec(feats[k]) for k in tr]); ytr = np.array([items[k]["y"] for k in tr])
    med = np.nanmedian(Xtr, axis=0)
    fill = lambda X: np.where(np.isnan(X), med, X)
    mu, sd = fill(Xtr).mean(0), fill(Xtr).std(0) + 1e-9
    Z = lambda X: (fill(X) - mu) / sd
    models = {"logistic": lambda: LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
              "mlp": lambda: MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=2000, random_state=0)}
    cv = {}
    for nm, mk in models.items():
        p = np.zeros(len(ytr))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, ytr):
            m = mk().fit(Z(Xtr[a]), ytr[a]); p[b] = m.predict_proba(Z(Xtr[b]))[:, 1]
        cv[nm] = (roc_auc_score(ytr, p), p)
    best = max(cv, key=lambda k: cv[k][0])
    pcv = cv[best][1]
    # s: the lowest score with >= 90% precision for "visible" on the training CV predictions
    order = np.argsort(-pcv)
    prec = np.cumsum(ytr[order]) / (np.arange(len(order)) + 1)
    ok = np.where(prec >= 0.9)[0]
    s_bar = float(pcv[order][ok.max()]) if len(ok) else 1.0
    model = models[best]().fit(Z(Xtr), ytr)
    dev = [k for k in feats if items[k]["src"] == "dev"]
    pdev = dict(zip(dev, model.predict_proba(Z(np.array([vec(feats[k]) for k in dev])))[:, 1]))
    # bar A on gate records
    gold = S.load_gold([V.GOLD])
    gd = json.loads((HERE / "gate_dev.json").read_text(encoding="utf-8"))
    rows = []
    for st, r in gd.items():
        for gi, g in enumerate(r["gate"]):
            m = [x for x in gold[st] if S.same_family(x["label"], g["label"]) and x["end"] > g["start"] and x["start"] < g["end"]]
            if not m:
                continue
            if any(x["visible"] for x in m):
                y = True
            elif all(x["needed"] for x in m):
                y = False
            else:
                continue
            sc = [pdev[k] for k in dev if items[k]["clip"] == st and items[k]["gate"] == gi]
            stt = g["stretches"]
            rows.append({"y": y, "head": float(np.mean(sc)) if sc else np.nan,
                         "maj": float(np.mean([v["seen"] for v in stt])) if stt else 0.0,
                         "ab": float(np.mean([(v["ab"] if v["ab"] is not None else v["seen"]) for v in stt])) if stt else 0.0})
    rows = [r for r in rows if not np.isnan(r["head"])]
    y = [r["y"] for r in rows]
    aA = auroc([r["head"] for r in rows], y)
    L = ["# Step 12 v1: trained visibility head (DEV)", "",
         f"Training: AVATAR {len(tr)} frames ({int(ytr.sum())} on-screen, {len(tr) - int(ytr.sum())} off-screen). 5-fold CV AUROC: "
         + ", ".join(f"{k} {v[0]:.3f}" for k, v in cv.items()) + f"; chosen {best}; cut bar s (90% precision on CV) {s_bar:.3f}.", "",
         f"Bar A (DEV gate records with a visibility label, {sum(y)} visible / {len(y) - sum(y)} not):", "",
         "| score | AUROC |", "|---|---|",
         f"| trained head (mean over seconds) | {aA:.3f} |",
         f"| current gate (share of stretches seen) | {auroc([r['maj'] for r in rows], y):.3f} |",
         f"| a/b rule | {auroc([r['ab'] for r in rows], y):.3f} |", "",
         f"Bar A (>= 0.85): {'PASS' if aA >= 0.85 else 'FAIL'}" + ("" if aA >= 0.85 else " -> bar B not run; version 2 (Synchformer, CAV-MAE) is the declared next step.")]
    (HERE / "vis_head_dev.json").write_text(json.dumps({"cv": {k: v[0] for k, v in cv.items()}, "best": best, "s": s_bar, "auroc_A": aA,
                                                        "dev_scores": {str(k): float(v) for k, v in pdev.items()}}, indent=1), encoding="utf-8")
    (HERE / "vis_head_dev.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
