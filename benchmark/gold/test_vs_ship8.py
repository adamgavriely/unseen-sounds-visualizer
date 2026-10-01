"""TEST read of one arm against the CURRENT shipped SHIP8 (TEST renders SHIP7+K4AD read with SHIP8's display keys: MERGE_GAP
2.5 + GROUP), merged TEST (old 60 + tagger 28), paired clip bootstrap. Reported, not selected on.
    python benchmark/gold/test_vs_ship8.py SHIP8+MD3        (from ~/MscProj_tg) -> test_vs_ship8_<arm>.json"""
import json
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import numpy as np
from benchmark.gold import final_test as FT
from benchmark.gold import score_per_sound as S

ARM, BASE = sys.argv[1], "SHIP7+K4AD"


def main():
    from benchmark.gold import dev_candidates_check as DCC
    F, T, R, _a, fin = FT.old_setup(ARM)
    disp = {k: R.arm_cfg("SHIP8")[k] for k in R.DISPLAY_KEYS}
    stems1 = list(T.STEMS)
    P1 = {}
    for a in (BASE, ARM):
        with R.flags(disp if a == BASE else {k: R.arm_cfg(a)[k] for k in R.DISPLAY_KEYS}):
            P1[a] = {st: S.load_pictures(fin / f"{a}_proposed", st, "proposed") or [] for st in stems1}
    from benchmark.gold import tagger_prep as TP
    for k, v in FT._ARMS0.items():
        R.ARMS[k] = dict(v)
    TP._ORIG.clear()
    _D2, R2, stems2 = TP.configure("test2")
    o2 = TP.out("test2")
    P2 = {}
    for a in (BASE, ARM):
        with R2.flags(disp if a == BASE else {k: R2.arm_cfg(a)[k] for k in R2.DISPLAY_KEYS}):
            P2[a] = {st: S.load_pictures(o2 / f"{a}_proposed", st, "proposed") or [] for st in stems2}
    S.load_gold = FT._REAL_LOAD_GOLD
    gold1 = S.load_gold([F.GOLD])
    d = json.loads(TP.TAGGER_GOLD.read_text(encoding="utf-8"))
    d["clips"] = [c for c in d.get("clips", []) if isinstance(c, dict) and Path(str(c.get("clip", ""))).stem in set(stems2)]
    tmp = o2 / "test2_gold_only.json"
    DCC.dump(tmp, d)
    gold2 = S.load_gold([tmp])
    res, cost = {"arm": ARM, "rows": {}}, {}
    for a in (BASE, ARM):
        rr = [S.score_clip(gold1[st], P1[a][st]) for st in stems1] + [S.score_clip(gold2[st], P2[a][st]) for st in stems2 if st in gold2]
        m = DCC.metrics(rr)
        res["rows"][a] = {k: m[k] for k in ("hits", "misses", "wrong", "visible", "cross", "phantom", "viewer_cost")}
        cost[a] = [DCC.clip_cost(r) for r in rr]
    res["d_vs_ship8"] = DCC.boot(np.subtract(cost[ARM], cost[BASE]))
    (_ROOT / "benchmark" / "gold" / f"test_vs_ship8_{ARM.replace('+', '_')}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
