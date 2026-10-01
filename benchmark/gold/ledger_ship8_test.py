"""Per-item ledger of the shipped pipeline (SHIP8 = TEST arm SHIP7+K4AD) on merged TEST: every wrong picture and every
missed needed sound, with the scorer's own matching (ledger_ship8.items, checked against score_clip's counts).
A reporting re-read of the already-spent TEST (no selection); written for the Round 46 gold-scope recheck.

    python benchmark/gold/ledger_ship8_test.py      (from ~/MscProj_tg)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import final_test as FT
from benchmark.gold import score_per_sound as S
from benchmark.gold.ledger_ship8 import items

ARM = "SHIP7+K4AD"
NAME, LOG = {}, []


def main():
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
    FT.set_tag("ledger_ship8_test_tmp")
    for p in (FT.OUT, FT.STARTED, FT.MD):
        p.unlink(missing_ok=True)
    FT.score(ARM)
    S.load_pictures, S.score_clip = lp, sc
    rows = []
    for gold, pics in LOG:
        d, st = NAME.get(id(pics), ("?", "?"))
        if f"{ARM}_proposed" not in d:
            continue
        its = items(st, gold, pics)
        r = sc(gold, pics)
        assert sum(x["kind"] == "wrong" for x in its) == r["visible"] + r["cross"] + r["phantom"], st
        for x in its:
            x["part"] = "test2" if "test2" in d or "tagger" in d else "test"
        rows += its
    for p in (FT.OUT, FT.STARTED, FT.MD):
        p.unlink(missing_ok=True)
    wrong = [x for x in rows if x["kind"] == "wrong"]; miss = [x for x in rows if x["kind"] == "miss"]
    print(f"{ARM} TEST: wrong {len(wrong)}, misses {len(miss)}")
    (_ROOT / "benchmark" / "gold" / "ledger_ship8_test.json").write_text(
        json.dumps({"arm": ARM, "misses": miss, "wrong": wrong}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
