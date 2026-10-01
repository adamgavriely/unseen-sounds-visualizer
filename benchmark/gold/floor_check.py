"""Parity check (1 Oct): config.use_shipped sets PICTURE_MIN_CONF 0.40, every scored arm uses None. Score SHIP8+MD3 with and
without the floor on merged DEV (and TEST via test_vs_ship8's loaders is not touched). Reported, not selected on.
    TG_ARMS="SHIP8+MD3" python benchmark/gold/floor_check.py"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import merged_dev as M
from benchmark.gold import round13_dev as R
A = "SHIP8+MD3"
for floor in ([None if sys.argv[1] == "none" else float(sys.argv[1])]):
    R.ARMS[A]["PICTURE_MIN_CONF"] = floor
    d1 = M.rows_dev([A]); d2 = M.rows_dev2([A])
    m = DCC.metrics([r for _s, r in d1[A]] + [r for _s, r in d2[A]])
    print("floor", floor, {k: m[k] for k in ("hits", "wrong", "visible", "cross", "phantom", "viewer_cost")}, flush=True)
