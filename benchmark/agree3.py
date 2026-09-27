"""All-three-agree check (docs/prereg_v4.md, 2026-09-28): cell E where each span E adds over shipped is kept only if
FlexSED >= 0.5, PE-A-Frame >= 5.20 and the listener says yes (score > 0) for its family within 1 s. The 280 (fit set).

    python benchmark/agree3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark import audioset_stage4_report as R
from benchmark import detector_round2 as D
from benchmark import listener_round as L

F_MIN, PE_MIN = 0.5, 5.196442604064941


def main():
    R.use_set("calib")
    cl = D.usable()
    pool = json.loads(L.POOL["calib"].read_text(encoding="utf-8"))["items"]
    listen = {k: v["score"] for k, v in pool.items() if "score" in v}
    spec = D.cells()["E"]

    def run(cid):
        base = D.stack(cid)
        ev, add = L.added(cid, base, spec)
        b, f, p, pe = D.caches(cid)
        drop = set()
        for e in add:
            ok = D.peak_near(f, e) >= F_MIN and D.peak_near(pe, e) >= PE_MIN and listen.get(L.key(cid, e), -1e9) > 0
            if not ok:
                drop.add(id(e))
        return [e for e in ev if id(e) not in drop]
    base = L.run_fn(cl, lambda cid: D.stack(cid))[1]
    cell = L.run_fn(cl, run)[1]
    kept = sum(1 for k, v in pool.items() if "E" in v["cells"] and "score" in v)
    ok = cell["C_overlap"] < base["C_overlap"] and cell["C_onset"] < base["C_onset"]
    res = {"shipped": base, "E+3": cell, "passes_fit_rule": ok}
    (_ROOT / "benchmark" / "agree3.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k in ("shipped", "E+3"):
        v = res[k]
        print(f"{k:8s} C-overlap {v['C_overlap']:.3f} C-onset {v['C_onset']:.3f} recall {v['recall_overlap']:.1%} false/min {v['fp_per_min']:.2f}")
    print("passes the fit rule:", ok)


if __name__ == "__main__":
    main()
