"""Step 14 bar 1 / bar 2 (PREREG_step14_audiosep_verifier.md). Cluster CPU from ~/wt_slice.
    python benchmark/gold/coverage/audiosep_eval.py  (needs audiosep_dev.json next to it) -> audiosep_dev.md"""
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


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    gold = S.load_gold([V.GOLD]); dev = V.stems("dev")
    it = json.loads((HERE / "verify_items_dev.json").read_text(encoding="utf-8"))["bursts"]
    f = json.loads((HERE / "audiosep_dev.json").read_text())
    B = [b for b in it if b["id"] in f and "error" not in f[b["id"]]]
    y = np.array([any(S.same_family(x["label"], b["family"]) and x["end"] > b["start"] - 0.5 and x["start"] < b["end"] + 0.5
                      for x in gold[b["clip"]]) for b in B])
    X = np.array([[np.log(f[b["id"]]["f1_frac"] + 1e-6), np.log(f[b["id"]]["f2_in_out"] + 1e-6), f[b["id"]]["f3_clap_diff"]] for b in B])
    g = np.array([b["clip"] for b in B])
    p = np.zeros(len(y))
    for a, c in GroupKFold(10).split(X, y, g):
        mu, sd = X[a].mean(0), X[a].std(0) + 1e-9
        m = LogisticRegression(max_iter=2000).fit((X[a] - mu) / sd, y[a]); p[c] = m.predict_proba((X[c] - mu) / sd)[:, 1]
    a3, al = auroc(X[:, 2], y), auroc(p, y)
    L = ["# Step 14: AudioSep verifier (DEV)", "", f"{len(B)} bursts, {int(y.sum())} real.", "",
         "| score | AUROC real vs not |", "|---|---|",
         f"| (1) stem energy fraction in span | {auroc(X[:, 0], y):.3f} |", f"| (2) stem in span vs outside | {auroc(X[:, 1], y):.3f} |",
         f"| (3) CLAP(stem) - CLAP(residual) | {a3:.3f} |", f"| logistic (1)-(3), clip-grouped 10-fold | {al:.3f} |", "",
         f"Bar 1 (>= 0.80): {'PASS' if max(a3, al) >= 0.80 else 'FAIL'}"]
    (HERE / "audiosep_dev.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
