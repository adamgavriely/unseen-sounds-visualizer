"""Round 40c EXPECT-A2 (docs/prereg_round13_detector_push.md "Round 40c EXPECT-A2"): the Round 40b Omni replies, map-only
(expect_a_screen.map_item: frozen word map or exact depictable family / label name, no cosine matcher), the FIRST TWO distinct
families each clip names, then as 40b: not already drawn, earliest weak-bar onset, shipped gate (the 40b cached answer re-used),
2-s picture. CPU only.

    TG_ARMS=SHIP8 python benchmark/gold/expect_a2_screen.py cands    # -> expect_a2/cands.json (+ cached gate files copied)
    TG_ARMS=SHIP8 python benchmark/gold/expect_a2_screen.py score    # -> expect_a2_screen.json
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")
from benchmark.gold import expect_screen as E
from benchmark.gold import expect_a_screen as A

SRC = _ROOT / "benchmark" / "gold" / "expect_a"                 # Round 40b listen replies + gate cache
E.DIR = _ROOT / "benchmark" / "gold" / "expect_a2"
E.OUT = _ROOT / "benchmark" / "gold" / "expect_a2_screen.json"
DIR = E.DIR
MAX_FAM = 2


def cmd_cands():
    from benchmark.gold import gbtp_screen as G
    from benchmark.gold import detached_add_screen as D
    from benchmark.gold.score_per_sound import same_family
    from src.labels import canonical
    assert D.BARS == {"flex": 0.3, "beats": 0.175, "dasm": 0.575}, D.BARS
    rows = []; missing = {m: 0 for m in D.BARS}
    tally = {"items": 0, "named": 0, "kept2": 0, "already_drawn": 0, "no_run": 0, "candidate": 0, "gate_cached": 0, "gate_uncached": 0}
    (DIR / "gate").mkdir(parents=True, exist_ok=True)
    for pt, st, g, pics in E.parts():
        lf = SRC / "listen" / f"{st}.json"
        if not lf.exists():
            print("missing listen output", pt, st); continue
        L = json.loads(lf.read_text(encoding="utf-8"))
        items = A.items_of(L["text"]); tally["items"] += len(items)
        fams, how = [], {}
        for it in items:
            f = A.map_item(it)
            if f and f not in fams:
                fams.append(f); how[f] = f"map:{it}"
        tally["named"] += len(fams)
        fams = fams[:MAX_FAM]; tally["kept2"] += len(fams)
        cs = G.caches(pt, st)
        for m in D.BARS:
            missing[m] += cs[m] is None
        drawn = [l for l, a, b, r in pics]
        for F in fams:
            row = {"part": pt, "clip": st, "family": F, "how": how[F], "omni_match": True}
            already = [l for l in drawn if canonical(l) == F or same_family(l, F)]
            if already:
                tally["already_drawn"] += 1; row["outcome"] = "already drawn"; row["drawn_as"] = already
            else:
                evs = {mm: D.evidence(fr, F) for mm, fr in cs.items()}
                found = {}
                for mm in D.BARS:
                    rr = D.runs_of(evs[mm], D.BARS[mm])
                    if rr:
                        found[mm] = min(s for s, e in rr)
                if not found:
                    tally["no_run"] += 1; row["outcome"] = "no run at weak bars"
                else:
                    tally["candidate"] += 1
                    src = min(found, key=found.get)
                    row.update({"outcome": "candidate", "onset": round(found[src], 2), "from": src,
                                "onsets": {k: round(v, 2) for k, v in found.items()}})
                    k = E.gate_key(row) + ".json"
                    if (SRC / "gate" / k).exists():
                        shutil.copy(SRC / "gate" / k, DIR / "gate" / k); tally["gate_cached"] += 1
                    else:
                        tally["gate_uncached"] += 1
            rows.append(row)
    (DIR / "cands.json").write_text(json.dumps({"tally": tally, "caches_missing": missing, "rows": rows}, indent=1), encoding="utf-8")
    print(tally, "caches missing", missing)
    for r in rows:
        if r["outcome"] == "candidate":
            print(f"   {r['part']:4s} {r['clip']} {r['family']} ({r['how']}) onset {r['onset']} ({r['from']})")


if __name__ == "__main__":
    {"cands": cmd_cands, "gate": E.cmd_gate, "score": E.cmd_score}[sys.argv[1]]()
