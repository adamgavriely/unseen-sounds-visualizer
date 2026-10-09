"""All 158 clips (DEV 71 + TEST 87; Adam 9 Oct: optimise on all clips). Existing rule combinations, pooled, with a
clip bootstrap against the frozen system and the DEV / TEST split shown. Cluster CPU from ~/wt_slice.
    python benchmark/gold/coverage/pooled_eval.py -> pooled_dev_test.md"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.coverage import score_coverage as V
from benchmark.gold.coverage.step11_policies import policy, TEXTURE

HERE = Path(__file__).resolve().parent


def main():
    gold = S.load_gold([V.GOLD])
    dev, test = V.stems("dev"), V.stems("test")
    D = json.loads((HERE / "step2_pics_dev.json").read_text(encoding="utf-8"))["cells"]
    T = json.loads((HERE / "final_pics_test.json").read_text(encoding="utf-8"))["cells"]
    fl = {**json.loads((HERE / "flashes_dev.json").read_text()), **json.loads((HERE / "flashes_test.json").read_text())}
    base = {}
    for rule in ("M", "AB-m", "AB-s"):
        c = f"SHIP8+MD3+WW5+SL|{rule}"
        base[rule] = {**{st: [tuple(p) for p in D[c]["clips"][st]["pics_none"]] for st in dev},
                      **{st: [tuple(p) for p in T[c]["clips"][st]["pics_none"]] for st in test}}
    def apply(rule, flash, tex):
        out = {}
        for st, pics in base[rule].items():
            p = policy(st, pics, "F", {}, {}, fl) if flash else pics
            if tex:
                p = [x for x in p if x[0] not in TEXTURE]
            out[st] = p
        return out
    allc = dev + test
    ref = [S.viewer_cost([V.score_clip_v2(gold[st], base["M"][st])]) for st in allc]
    L = ["# All 158 clips: existing rules pooled (DEV 71 + TEST 87)", "",
         "| gate | flash | texture ban | hits /124 | wrong | onset cost | cost_cov | DEV hits/wrong | TEST hits/wrong | d cost vs frozen [95% CI], p |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    rng = np.random.default_rng(0)
    for rule, flash, tex in itertools.product(("M", "AB-m", "AB-s"), (False, True), (False, True)):
        P = apply(rule, flash, tex)
        rows = {st: V.score_clip_v2(gold[st], P[st]) for st in allc}
        A = V.aggregate([rows[st] for st in allc])
        a_d = V.aggregate([rows[st] for st in dev]); a_t = V.aggregate([rows[st] for st in test])
        d = np.array([S.viewer_cost([rows[st]]) for st in allc]) - np.array(ref)
        m = d[rng.integers(0, len(d), (50000, len(d)))].mean(1)
        p = min(1.0, 2 * min((m >= 0).mean(), (m <= 0).mean()))
        L.append(f"| {rule} | {'yes' if flash else '-'} | {'yes' if tex else '-'} | {A['hits']} | {A['wrong']} | {A['onset_cost']:.3f} | {A['cost_cov']:.3f} | "
                 f"{a_d['hits']}/{a_d['wrong']} | {a_t['hits']}/{a_t['wrong']} | {d.mean():+.3f} [{np.percentile(m, 2.5):+.3f}, {np.percentile(m, 97.5):+.3f}], {p:.2f} |")
    (HERE / "pooled_dev_test.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
