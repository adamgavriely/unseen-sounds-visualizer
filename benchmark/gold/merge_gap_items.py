"""Per-item diff behind merge_gap_sens (Adam 2026-10-01: which sound changes at each gap?). Re-reads the saved SHIP8 pictures
at one MERGE_GAP and writes every SHIP8 hit / wrong / miss (ledger_ship8.items). Reported, not selected on.

    TG_ARMS=SHIP8 python benchmark/gold/merge_gap_items.py dev --gap 2.5   -> merge_gap_sens/items_<set>_<gap>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import merge_gap_sens as MG
from benchmark.gold import score_per_sound as S
from benchmark.gold.ledger_ship8 import items

NAME, LOG = {}, []


def main():
    st_set, gap = sys.argv[1], float(sys.argv[3])
    lp, sc = S.load_pictures, S.score_clip

    def load(d, st, *a, **k):
        r = lp(d, st, *a, **k)
        if r is not None:
            NAME[id(r)] = (str(d), st)
        return r

    def score(gold, pics, *a, **k):
        LOG.append((gold, pics))
        return sc(gold, pics, *a, **k)

    S.load_pictures, S.score_clip = load, score
    rows = (MG.dev if st_set == "dev" else MG.test)(gap)
    S.load_pictures, S.score_clip = lp, sc
    out = []
    for gold, pics in LOG:
        d, st = NAME.get(id(pics), ("?", "?"))
        if "B0r" in Path(d).name:
            continue
        hits = sc(gold, pics)["hit"]
        out.append({"clip": st, "dir": Path(d).name, "hit": hits, "pics": [[l, round(a, 2), round(b, 2)] for l, a, b in pics],
                    "items": items(st, gold, pics)})
    (MG.OUTD / f"items_{st_set}_{gap}.json").write_text(json.dumps({"rows": rows, "clips": out}, indent=1, default=float),
                                                      encoding="utf-8")
    print(st_set, gap, rows.get("SHIP8"))


if __name__ == "__main__":
    main()
