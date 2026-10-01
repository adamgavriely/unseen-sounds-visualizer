"""Round 53 WEAK-WITNESS step 1 (docs/prereg_round13_detector_push.md "Round 53 WEAK-WITNESS"), CPU, held-out 415 AudioSet-Strong
clips: precision of raw BEATs spans binned by their family's DASM max over [start - 0.5, end + 0.5] (the DASM_LOCAL_VETO window).

Raw BEATs spans as shipped MD3 / Round 52: _extract_events(threshold 0.35 display bar, min span 0.3 s, low 0.175 = AED_THRESHOLD,
hysteresis 1.0), one span per BEATs label, family = canonical(label), depictable families only (expect_screen.FAMILIES). Spans whose
family has no DASM column (or no DASM frame in the window) are excluded (the veto keeps them; counted). Correctness = Round 42
(heldout_a4_screen.classify: a same_family strong event starts in [onset - 0.5, onset + 1.0]).
b = the single highest edge e in {0.05 .. 0.60} whose cumulative precision (spans with DASM < e) is <= 0.20; STOP unless 0.1 < b < 0.4.
No both-ears keep can be simulated here (no P1 listener cache on the 415).

    python benchmark/gold/weakwitness_415.py      # from ~/MscProj_tg -> benchmark/gold/weakwitness_415.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")

from benchmark.gold import heldout_a4_screen as H  # noqa: E402

HALF, CAP, TARGET = 0.5, 0.20, 0.375
EDGES = [round(0.05 * k, 2) for k in range(1, 13)]          # 0.05 .. 0.60
OUT = _ROOT / "benchmark" / "gold" / "weakwitness_415.json"


def main():
    import config
    from benchmark.gold.score_per_sound import same_family
    from benchmark.gold import expect_screen as E
    from benchmark.gold import expect_a4_screen as A4
    from benchmark.gold import prior415_screen as P
    from benchmark.gold import dev_candidates_check as DCC
    from src.labels import canonical
    from src.stage4_audio_event_detection import _extract_events
    config.use_shipped()
    flags = {k: getattr(config, k, None) for k in ("AED_THRESHOLD", "AED_HYSTERESIS", "AED_MIN_DUR", "AED_RELEASE", "DISPLAY_THRESHOLD")}
    assert flags["AED_THRESHOLD"] == 0.175 and flags["AED_HYSTERESIS"] == 1.0 and flags["AED_MIN_DUR"] == 0.3, flags
    assert flags["AED_RELEASE"] is None and flags["DISPLAY_THRESHOLD"] == 0.35, flags
    dep = set(E.FAMILIES)
    Hj = json.loads(H.HELDOUT.read_text(encoding="utf-8"))
    ev_of = {c["id"]: c["events"] for c in Hj["clips"]}
    cl = H.ids()
    miss = {"beats": [c for c in cl if not (H.BEATS_HELD / f"{c}.npz").exists()],
            "dasm": [c for c in cl if not (H.DASM_HELD / f"{c}.npz").exists()]}
    assert not any(miss.values()), {k: v[:3] for k, v in miss.items() if v}
    ref, rows, nocol = [], [], []
    for c in cl:
        bfw, bt, blabs = DCC.load_fr(H.BEATS_HELD / f"{c}.npz")
        dfr = DCC.load_fr(H.DASM_HELD / f"{c}.npz")
        for lab, a, _b in P.beats_runs(bfw, bt, blabs):          # reference: min 0.5, display bar (Round 42: 754 / 283)
            f = canonical(lab)
            if f in dep:
                ref.append((f, H.classify(f, a, ev_of[c], same_family)))
        for e in _extract_events(bfw, bt, blabs, 0.35, None, 0.3, low=0.175):
            f = canonical(e.label)
            if f not in dep:
                continue
            a, b = float(e.start), float(e.end)
            k = H.classify(f, a, ev_of[c], same_family)
            v = A4.dasm_max(dfr, f, a - HALF, b + HALF)
            r = {"clip": c, "label": e.label, "family": f, "onset": round(a, 2), "end": round(b, 2),
                 "conf": round(float(e.confidence), 3), "dasm_win": None if v is None else round(v, 4), "class": k}
            (nocol if v is None else rows).append(r)

    def S(rs):
        return H.summ([(r["family"], r["class"]) for r in rs])

    bins = []
    lo = 0.0
    for hi in EDGES + [9.0]:
        bins.append({"bin": [lo, hi if hi < 9 else None], **S([r for r in rows if lo <= r["dasm_win"] < hi])})
        lo = hi
    cum = [{"edge": e, **S([r for r in rows if r["dasm_win"] < e])} for e in EDGES]
    ok = [x["edge"] for x in cum if x["precision"] is not None and x["precision"] <= CAP]
    b = max(ok) if ok else None
    go = b is not None and 0.1 < b < 0.4
    res = {"what": __doc__.strip().splitlines()[0], "clips": len(cl), "flags": flags,
           "rule": "correct iff a same_family strong event starts in [onset-0.5, onset+1.0] (heldout_a4_screen.classify)",
           "params": {"window": "[start-0.5, end+0.5]", "cap": CAP, "edges": EDGES,
                      "beats": "threshold 0.35, low 0.175, min span 0.3 s, one span per BEATs label"},
           "reference_raw_beats_min0.5": H.summ(ref),
           "spans_with_dasm": S(rows), "spans_no_dasm_column": S(nocol),
           "bins": bins, "cumulative_below_edge": cum, "b": b, "go": go,
           "per_family_below_0.35": H.per_family([(r["family"], r["class"]) for r in rows if r["dasm_win"] < 0.35]),
           "rows": rows, "rows_no_dasm": nocol}
    print("reference", res["reference_raw_beats_min0.5"], "\nspans with DASM", res["spans_with_dasm"],
          "\nno DASM column", res["spans_no_dasm_column"], flush=True)
    for x in bins:
        print("bin", x["bin"], x["n"], x["correct"], x["precision"])
    for x in cum:
        print("below", x["edge"], x["n"], x["correct"], x["precision"])
    print("b", b, "go", go)
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
