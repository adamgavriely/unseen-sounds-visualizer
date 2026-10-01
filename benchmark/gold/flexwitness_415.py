"""Round 58b FLEX-WITNESS step 1 (docs/prereg_round13_detector_push.md "Round 58b FLEX-WITNESS"), CPU, held-out 415.
Span set = Round 53 step 1 (weakwitness_415.json rows), ears = Round 53b whole-clip list proxy (weakwitness2_415.json rows: omni,
afn, ears). FlexSED own-family max over [onset - 0.5, end + 0.5] (flexsed_heldout, columns with canonical(column) == family).
GO iff one-ear spans with FlexSED >= 0.5 have precision >= 0.375 AND one-ear spans with FlexSED < 0.5 (incl. no column) <= 0.20.

    python benchmark/gold/flexwitness_415.py      # from ~/MscProj_tg (after weakwitness2_415.py) -> flexwitness_415.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")

from benchmark.gold import dev_candidates_check as DCC  # noqa: E402
from benchmark.gold import heldout_a4_screen as H  # noqa: E402
from src.labels import canonical  # noqa: E402

GOLD = _ROOT / "benchmark" / "gold"
BAR, HI_MIN, LO_MAX, DASM_B = 0.5, 0.375, 0.20, 0.35
OUT = GOLD / "flexwitness_415.json"


def main():
    rows = json.loads((GOLD / "weakwitness2_415.json").read_text(encoding="utf-8"))["rows"]
    assert len(rows) == 754, len(rows)
    frs = {}
    for r in rows:
        c = r["clip"]
        if c not in frs:
            frs[c] = DCC.load_fr(H.FLEX_HELD / f"{c}.npz")
        fw, t, labs = frs[c]
        cols = [i for i, l in enumerate(labs) if canonical(l) == r["family"]]
        m = (t >= r["onset"] - 0.5 - 1e-9) & (t <= r["end"] + 0.5 + 1e-9)
        r["flex_win"] = round(float(fw[m][:, cols].max()), 3) if cols and m.any() else None

    def S(rs):
        return H.summ([(r["family"], r["class"]) for r in rs])

    hi = lambda r: r["flex_win"] is not None and r["flex_win"] >= BAR
    one = [r for r in rows if r["ears"] == 1]
    res = {"what": __doc__.strip().splitlines()[0], "bar": BAR, "proxy": "whole-clip Omni / AFN lists stand in for span-level V4",
           "all_spans": S(rows), "no_flex_column": S([r for r in rows if r["flex_win"] is None])}
    res["one_ear_flex_ge"] = S([r for r in one if hi(r)])
    res["one_ear_flex_lt"] = S([r for r in one if not hi(r)])
    res["one_ear_no_column"] = S([r for r in one if r["flex_win"] is None])
    res["one_ear_flex_lt_with_column"] = S([r for r in one if r["flex_win"] is not None and r["flex_win"] < BAR])
    lo1 = [r for r in one if r["dasm_win"] < DASM_B]
    res["reported_one_ear_dasm_lt_0.35"] = {"flex_ge": S([r for r in lo1 if hi(r)]), "flex_lt": S([r for r in lo1 if not hi(r)])}
    res["reported_by_ears"] = {k: {"flex_ge": S([r for r in rows if r["ears"] == k and hi(r)]),
                                   "flex_lt": S([r for r in rows if r["ears"] == k and not hi(r)])} for k in (0, 1, 2)}
    bins, lo = [], 0.0
    for e in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 9.0]:
        bins.append({"bin": [lo, e if e < 9 else None], **S([r for r in one if r["flex_win"] is not None and lo <= r["flex_win"] < e])})
        lo = e
    res["one_ear_bins"] = bins
    g, l = res["one_ear_flex_ge"]["precision"], res["one_ear_flex_lt"]["precision"]
    res["go"] = g is not None and l is not None and g >= HI_MIN and l <= LO_MAX
    # reported: what the rule would drop on the 415 (proxy ears) vs Round 53's both-only
    drop = lambda rs, fw: S([r for r in rs if r["dasm_win"] < DASM_B and r["ears"] < 2 and not (fw and r["ears"] == 1 and hi(r))])
    res["drop_415_round53"], res["drop_415_flexwitness"] = drop(rows, False), drop(rows, True)
    for k, v in res.items():
        if k not in ("rows",):
            print(k, v)
    res["rows"] = rows
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
