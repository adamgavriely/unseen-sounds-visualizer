"""Round 56 TWIN-SHORT, merged DEV: changed pictures of arm A vs arm B (weakwitness_dev.diff's loop, own output file).

    TG_ARMS="SHIP8+MD3 SHIP8+MD3+TS" python benchmark/gold/twinshort_dev.py diff SHIP8+MD3 SHIP8+MD3+TS   # from ~/MscProj_tg
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8+MD3")

from benchmark.gold import weakwitness_dev as W  # noqa: E402


def diff(A, B):
    res = []
    for part, (gold, P, _c, _s) in W.parts([A, B]).items():
        for st in P[A]:
            ca, cb = W.classify(gold[st], P[A][st]), W.classify(gold[st], P[B][st])
            ka = {(l, round(a, 2), round(b, 2), k) for l, a, b, k in ca}
            kb = {(l, round(a, 2), round(b, 2), k) for l, a, b, k in cb}
            if ka != kb:
                res.append({"part": part, "clip": st, "only_" + A: sorted(ka - kb, key=lambda x: x[1]),
                            "only_" + B: sorted(kb - ka, key=lambda x: x[1])})
                print(part, st, "\n   -", sorted(ka - kb, key=lambda x: x[1]), "\n   +", sorted(kb - ka, key=lambda x: x[1]), flush=True)
    p = _ROOT / "benchmark" / "gold" / "twinshort_diff.json"
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(p)


if __name__ == "__main__":
    diff(sys.argv[2], sys.argv[3])
