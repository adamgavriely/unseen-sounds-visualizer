"""Per-step statistics of the shipped stack (Fable review, 2026-09-30): B0r -> TO1+F7F8 -> +N2b -> +DR2 -> +K-V4 on merged
DEV (71) and merged TEST (88). Per-clip costs are RECOMPUTED from the saved pictures of runs already scored (no new TEST
exposure, no new decision). Paired bootstrap (DCC.boot, 2000, seed 0) of each consecutive step and of each arm vs B0r; Holm
over the four TEST reads.

    python benchmark/gold/stack_stats.py      # from ~/MscProj_tg
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold import dev_candidates_check as DCC

ARMS = ["B0r", "TO1+F7F8", "TO1F7F8+N2b", "SHIP+DR2", "SHIP2+KV4"]
NAMES = ["B0r", "TO1+F7F8", "+N2b", "+DR2", "+K-V4"]
WORK = _ROOT / "data" / "work"


def gold_tagger(stems):
    from benchmark.gold import tagger_prep as T
    d = json.loads(T.TAGGER_GOLD.read_text(encoding="utf-8"))
    keep = set(stems)
    d["clips"] = [c for c in d["clips"] if isinstance(c, dict) and Path(str(c.get("clip", ""))).stem in keep]
    tmp = WORK / "stack_stats_gold_tmp.json"
    DCC.dump(tmp, d)
    g = T._REAL_LOAD_GOLD([tmp])
    tmp.unlink()
    return g


def pics(root, arm, stems, R):
    with R.flags({k: R.arm_cfg(arm)[k] for k in R.DISPLAY_KEYS}):
        return {st: S.load_pictures(root / f"{arm}_proposed", st, "proposed") or [] for st in stems}


def main():
    from benchmark.gold import round13_dev as R
    real = S.load_gold                                    # before tagger_prep installs its gold stub
    parts = {}
    # DEV 49
    g1, st1 = DCC.dev_stems()
    parts["dev49"] = (g1, st1, {a: pics(WORK / "r13", a, st1, R) for a in ARMS})
    # old TEST 60 (r16final; B0r and TO1+F7F8 also there)
    tst = [x.strip() for x in (_ROOT / "benchmark" / "gold" / "test_stems.txt").read_text(encoding="utf-8").split() if x.strip()]
    gT = real([_ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"])
    parts["test60"] = (gT, tst, {a: pics(WORK / "r16final", a, tst, R) for a in ARMS})
    # tagger parts (tagger_prep's own folders)
    from benchmark.gold import tagger_prep as T
    for split, name in (("dev2", "dev22"), ("test2", "test28")):
        _D, R2, st = T.configure(split)
        g = gold_tagger(st)
        st = [s for s in st if s in g]
        parts[name] = (g, st, {a: pics(T.out(split), a, st, R2) for a in ARMS})
    res = {}
    for set_name, keys in (("DEV", ("dev49", "dev22")), ("TEST", ("test60", "test28"))):
        rows = {a: [] for a in ARMS}
        for k in keys:
            g, st, P = parts[k]
            for a in ARMS:
                rows[a] += [S.score_clip(g[s], P[a][s]) for s in st]
        cost = {a: np.array([DCC.clip_cost(r) for r in rows[a]]) for a in ARMS}
        met = {a: DCC.metrics(rows[a]) for a in ARMS}
        steps = [(NAMES[i + 1], DCC.boot(cost[ARMS[i + 1]] - cost[ARMS[i]])) for i in range(len(ARMS) - 1)]
        vsb = [(NAMES[i], DCC.boot(cost[ARMS[i]] - cost["B0r"])) for i in range(1, len(ARMS))]
        res[set_name] = {"clips": len(cost["B0r"]), "rows": {n: {k: met[a][k] for k in ("hits", "misses", "wrong", "visible", "cross",
                                                                                        "phantom", "viewer_cost")}
                                                          for n, a in zip(NAMES, ARMS)},
                         "steps": steps, "vs_B0r": vsb}
    ps = sorted([(n, d[3]) for n, d in res["TEST"]["vs_B0r"]], key=lambda x: x[1])
    m, holm, run = len(ps), {}, 0.0
    for i, (n, p) in enumerate(ps):
        run = max(run, min(1.0, (m - i) * p))
        holm[n] = round(run, 3)
    res["TEST"]["holm_vs_B0r"] = holm
    (_ROOT / "benchmark" / "gold" / "stack_stats.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for set_name in ("DEV", "TEST"):
        r = res[set_name]
        print(f"== {set_name} ({r['clips']} clips)")
        for n in NAMES:
            x = r["rows"][n]
            print(f"  {n:10s} hits {x['hits']} wrong {x['wrong']} ({x['visible']}/{x['cross']}/{x['phantom']}) cost {x['viewer_cost']:.3f}")
        for n, d in r["steps"]:
            print(f"  step {n:8s} d {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] p {d[3]:.3f}")
        for n, d in r["vs_B0r"]:
            print(f"  vs B0r {n:8s} d {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] p {d[3]:.3f}"
                  + (f"  Holm {r['holm_vs_B0r'][n]}" if set_name == "TEST" else ""))


if __name__ == "__main__":
    main()
