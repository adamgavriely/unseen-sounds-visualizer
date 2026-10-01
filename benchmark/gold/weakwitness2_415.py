"""Round 53b WEAK-WITNESS-2 step 1 (docs/prereg_round13_detector_push.md "Round 53b WEAK-WITNESS-2"), CPU, held-out 415: the one-ear
DASM bar b1. Span set = Round 53 step 1 (weakwitness_415.json rows: raw BEATs spans, bar 0.35, low 0.175, min 0.3 s, depictable,
family DASM max over [start - 0.5, end + 0.5], Round 42 class). Ear proxy (whole-clip lists, NOT the span-level V4 of DEV):
Qwen3-Omni list (heldout_a4/listen/<id>.json items -> expect_a_screen.map_item) and AFN list (agree_ears/heldout/<id>.json families).
one-ear = family named by exactly one list. b1 = highest edge e in {0.05..0.60} with cumulative precision (one-ear spans, DASM < e)
<= 0.20; STOP unless 0.1 < b1 < 0.35.

    python benchmark/gold/weakwitness2_415.py      # from ~/MscProj_tg (after weakwitness_415.py) -> weakwitness2_415.json
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
OUT = GOLD / "weakwitness2_415.json"


def main():
    from benchmark.gold import expect_a_screen as A
    R = json.loads((GOLD / "weakwitness_415.json").read_text(encoding="utf-8"))
    rows = R["rows"]
    qw, af, miss = {}, {}, {"omni": [], "afn": []}
    for c in H.ids():
        p, q = GOLD / "heldout_a4" / "listen" / f"{c}.json", GOLD / "agree_ears" / "heldout" / f"{c}.json"
        if p.exists():
            qw[c] = {f for f in (A.map_item(i) for i in json.loads(p.read_text(encoding="utf-8"))["items"]) if f}
        else:
            miss["omni"].append(c)
        if q.exists():
            af[c] = set(json.loads(q.read_text(encoding="utf-8"))["families"])
        else:
            miss["afn"].append(c)
    assert not any(miss.values()), {k: (len(v), v[:3]) for k, v in miss.items()}
    for r in rows:
        r["omni"], r["afn"] = r["family"] in qw[r["clip"]], r["family"] in af[r["clip"]]
        r["ears"] = int(r["omni"]) + int(r["afn"])

    def S(rs):
        return H.summ([(r["family"], r["class"]) for r in rs])

    by_ears = {k: S([r for r in rows if r["ears"] == k]) for k in (0, 1, 2)}
    one = [r for r in rows if r["ears"] == 1]
    bins, lo = [], 0.0
    for hi in EDGES + [9.0]:
        bins.append({"bin": [lo, hi if hi < 9 else None], **S([r for r in one if lo <= r["dasm_win"] < hi])})
        lo = hi
    cum = [{"edge": e, **S([r for r in one if r["dasm_win"] < e])} for e in EDGES]
    ok = [x["edge"] for x in cum if x["precision"] is not None and x["precision"] <= CAP]
    b1 = max(ok) if ok else None
    go = b1 is not None and 0.1 < b1 < 0.35
    # reported: what the tiered rule at b1 would drop on the 415 vs Round 53's both-only rule (proxy ears)
    def dropped(rs, t):
        return S([r for r in rs if r["dasm_win"] < 0.35 and r["ears"] < 2 and not (r["ears"] == 1 and t is not None and r["dasm_win"] >= t)])
    res = {"what": __doc__.strip().splitlines()[0], "clips": len(qw), "cap": CAP, "edges": EDGES,
           "proxy": "whole-clip Omni / AFN lists stand in for span-level Qwen V4 / AF V4",
           "all_spans": S(rows), "by_ears": by_ears,
           "one_ear_split": {"omni_only": S([r for r in one if r["omni"]]), "afn_only": S([r for r in one if r["afn"]])},
           "one_ear_bins": bins, "one_ear_cumulative_below_edge": cum, "b1": b1, "go": go,
           "drop_415_round53_both_only": dropped(rows, None), "drop_415_tiered_b1": dropped(rows, b1) if b1 else None,
           "rows": rows}
    print("all", res["all_spans"], "\nby ears", by_ears, "\none-ear split", res["one_ear_split"])
    for x in bins:
        print("bin", x["bin"], x["n"], x["correct"], x["precision"])
    for x in cum:
        print("below", x["edge"], x["n"], x["correct"], x["precision"])
    print("b1", b1, "go", go, "\ndrop R53", res["drop_415_round53_both_only"], "\ndrop tiered", res["drop_415_tiered_b1"])
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
