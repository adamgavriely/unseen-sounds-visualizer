"""Round 53c WEAK-WITNESS-3 step 1 (docs/prereg_round13_detector_push.md "Round 53c WEAK-WITNESS-3"), CPU, held-out 415: the no-ear
DASM bar b0. Rows = weakwitness2_415.json (Round 53 span set + whole-clip Omni / AFN ear proxy). b0 = highest edge e in {0.05..0.60}
with cumulative precision (no-ear spans, DASM < e) <= 0.20; STOP unless 0.1 < b0 < 0.6.

    python benchmark/gold/weakwitness3_415.py      # from ~/MscProj_tg -> weakwitness3_415.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")

from benchmark.gold import heldout_a4_screen as H  # noqa: E402

GOLD = _ROOT / "benchmark" / "gold"
CAP = 0.20
EDGES = [round(0.05 * k, 2) for k in range(1, 13)]
OUT = GOLD / "weakwitness3_415.json"


def main():
    rows = json.loads((GOLD / "weakwitness2_415.json").read_text(encoding="utf-8"))["rows"]
    S = lambda rs: H.summ([(r["family"], r["class"]) for r in rs])
    none = [r for r in rows if r["ears"] == 0]
    bins, lo = [], 0.0
    for hi in EDGES + [9.0]:
        bins.append({"bin": [lo, hi if hi < 9 else None], **S([r for r in none if lo <= r["dasm_win"] < hi])})
        lo = hi
    cum = [{"edge": e, **S([r for r in none if r["dasm_win"] < e])} for e in EDGES]
    ok = [x["edge"] for x in cum if x["precision"] is not None and x["precision"] <= CAP]
    b0 = max(ok) if ok else None
    go = b0 is not None and 0.1 < b0 < 0.6
    res = {"what": __doc__.strip().splitlines()[0], "proxy": "whole-clip Omni / AFN lists stand in for P1 listener / AF V4",
           "no_ear": S(none), "no_ear_bins": bins, "no_ear_cumulative_below_edge": cum, "b0": b0, "go": go,
           "drop_415_at_b0": S([r for r in none if b0 and r["dasm_win"] < b0]),
           "drop_415_round53_both_only_0.35": S([r for r in rows if r["dasm_win"] < 0.35 and r["ears"] < 2])}
    print("no-ear", res["no_ear"])
    for x in bins:
        print("bin", x["bin"], x["n"], x["correct"], x["precision"])
    for x in cum:
        print("below", x["edge"], x["n"], x["correct"], x["precision"])
    print("b0", b0, "go", go, "\ndrop at b0", res["drop_415_at_b0"], "\nR53 drop", res["drop_415_round53_both_only_0.35"])
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
