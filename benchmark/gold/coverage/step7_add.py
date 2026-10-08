"""Step 7 C (PREREG_step7_add_and_gate_combo.md): keep every a/b-candidate picture, add dropped bursts with a high
out-of-fold keep-score. DEV only, CPU.   python benchmark/gold/coverage/step7_add.py -> step7_add_dev.md"""
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.coverage import score_coverage as V
from benchmark.gold.coverage import keep_score as K

HERE = Path(__file__).resolve().parent


def main():
    gold = S.load_gold([V.GOLD])
    dev = V.stems("dev")
    rows = K.table()
    B = [b for b, _ in rows]
    X = np.array([x for _, x in rows])
    y = np.array([K.good(b, gold) for b in B])
    g = np.array([b["clip"] for b in B])
    ab = json.loads((HERE / "step2_pics_dev.json").read_text(encoding="utf-8"))["cells"][K.AB]["clips"]
    base = {st: [tuple(p) for p in ab[st]["pics_none"]] for st in dev}
    dropped = [i for i, b in enumerate(B) if not any(S.same_family(p[0], b["family"]) and p[2] > b["start"] and p[1] < b["end"]
                                                   for p in base[b["clip"]])]
    L = ["# Step 7 C: add-only rescue on top of the a/b candidate (DEV, out-of-fold keep-score)", "",
         f"{len(dropped)} dropped bursts can be added ({int(y[dropped].sum())} of them would be hits).", "",
         "| model | best at wrong <= 16 | <= 18 | <= 20 | clear win |", "|---|---|---|---|---|"]
    res = {}
    for kind in ("trees", "logistic"):
        p = K.oof(X, y, g, kind)
        order = sorted(dropped, key=lambda i: -p[i])
        pts = []
        for k in range(0, min(len(order), 60) + 1):
            add = {}
            for i in order[:k]:
                add.setdefault(B[i]["clip"], []).append((B[i]["family"], B[i]["start"], B[i]["end"]))
            a = K.score({st: base[st] + add.get(st, []) for st in dev}, gold, dev)
            pts.append({"k": k, "p_min": float(p[order[k - 1]]) if k else None, "hits": a["hits"], "wrong": a["wrong"],
                        "onset_cost": a["onset_cost"], "cost_cov": a["cost_cov"]})
        best = lambda w: max((q for q in pts if q["wrong"] <= w), key=lambda q: (q["hits"], -q["wrong"]))
        win = [q for q in pts if (q["hits"] >= 35 and q["wrong"] <= 16) or (q["hits"] >= 32 and q["wrong"] <= 13)]
        f = lambda q: f"{q['hits']} / {q['wrong']} (k {q['k']}, cost {q['onset_cost']:.3f})"
        L.append(f"| {kind} | {f(best(16))} | {f(best(18))} | {f(best(20))} | {'YES ' + str([(q['hits'], q['wrong']) for q in win]) if win else 'no'} |")
        res[kind] = pts
    (HERE / "step7_add_dev.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (HERE / "step7_add_dev.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
