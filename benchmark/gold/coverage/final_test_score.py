"""FINAL_TEST_RUN.md: score the frozen system, the a/b rule, and a/b + flash on the 87 TEST clips, once. Cluster CPU
from ~/wt_slice; inputs final_pics_test.json, flashes_test.json next to this file."""
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.coverage import score_coverage as V
from benchmark.gold.coverage.step11_policies import policy

HERE = Path(__file__).resolve().parent


def main():
    gold = S.load_gold([V.GOLD]); test = V.stems("test")
    cells = json.loads((HERE / "final_pics_test.json").read_text(encoding="utf-8"))["cells"]
    fl = json.loads((HERE / "flashes_test.json").read_text(encoding="utf-8"))
    pics = {"frozen": {st: [tuple(p) for p in cells["SHIP8+MD3+WW5+SL|M"]["clips"][st]["pics_none"]] for st in test},
            "a/b": {st: [tuple(p) for p in cells["SHIP8+MD3+WW5+SL|AB-m"]["clips"][st]["pics_none"]] for st in test}}
    pics["a/b + flash"] = {st: policy(st, pics["a/b"][st], "F", {}, {}, fl) for st in test}
    rows = {k: [V.score_clip_v2(gold[st], v[st]) for st in test] for k, v in pics.items()}
    L = ["# Final TEST run (87 clips), scored once", "", "| system | hits | wrong (vis / other / none) | onset cost | cost_cov |", "|---|---|---|---|---|"]
    for k, rr in rows.items():
        A = V.aggregate(rr)
        L.append(f"| {k} | {A['hits']}/{A['needed']} | {A['wrong']} ({sum(r['visible'] for r in rr)} / {sum(r['cross'] for r in rr)} / "
                 f"{sum(r['phantom'] for r in rr)}) | {A['onset_cost']:.3f} | {A['cost_cov']:.3f} |")
    base = np.array([S.viewer_cost([r]) for r in rows["frozen"]])
    rng = np.random.default_rng(0)
    for k in ("a/b", "a/b + flash"):
        d = np.array([S.viewer_cost([r]) for r in rows[k]]) - base
        m = d[rng.integers(0, len(d), (100000, len(d)))].mean(1)
        p = min(1.0, 2 * min((m >= 0).mean(), (m <= 0).mean()))
        L.append(f"\n{k} minus frozen, onset cost per clip: {d.mean():+.3f} [{np.percentile(m, 2.5):+.3f}, {np.percentile(m, 97.5):+.3f}], p {p:.3f}")
    L.append("\nFlash rule changes on TEST: " + "; ".join(f"{st}: {sorted(set(pics['a/b'][st]) - set(pics['a/b + flash'][st]))}" for st in test
                                                        if pics["a/b"][st] != pics["a/b + flash"][st]))
    (HERE / "final_test.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
