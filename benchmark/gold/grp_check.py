"""Round 47 shipped form check: SHIP8 + src GROUP (cache from grp_screen ask_*, GROUP_MAX_GAP 4.0) through the scorer's own
display path, merged DEV and TEST at MERGE_GAP 2.5.   TG_ARMS=SHIP8 python benchmark/gold/grp_check.py dev|test"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import merge_gap_sens as MG
from benchmark.gold import round13_dev as R
G = {"GROUP_ASK": True, "GROUP_MAX_GAP": 4.0,
     "GROUP_CACHE": str(_ROOT / "benchmark" / "gold" / "grp" / "group_answers_bench.json")}
_orig = R.arm_cfg
R.arm_cfg = lambda a: {**_orig(a), **(G if a in ("SHIP8", "SHIP7+K4AD") else {})}
print(sys.argv[1], (MG.dev if sys.argv[1] == "dev" else MG.test)(2.5)["SHIP8"])
