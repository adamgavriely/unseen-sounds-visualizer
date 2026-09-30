"""Does the runner (slurm/run_best.sh, split `shipcheck`: 5 DEV clips re-run from scratch through tagger_prep) give the same
pictures as the DEV harness run of the same arms (data/work/r13/<arm>_proposed)? Bar: identical (label, start, end) on
every clip for B0r and TO1+F7F8. No gold is read.

    TG_EXTRA_SPLITS=shipcheck python benchmark/gold/shipcheck_compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import round13_dev as R
from benchmark.gold import score_per_sound as S
from benchmark.gold import tagger_prep as T

SYS = "proposed"


def pics(root, arm, st):
    with R.flags({k: R.arm_cfg(arm)[k] for k in R.DISPLAY_KEYS}):
        return [(p[0], round(p[1], 2), round(p[2], 2)) for p in (S.load_pictures(root / f"{arm}_{SYS}", st, SYS) or [])]


def main():
    dev_root = R.R13
    stems = T.stems_of("shipcheck")
    new_root = T.out("shipcheck")
    bad = 0
    for arm in ("B0r", "TO1+F7F8"):
        for st in stems:
            a, b = pics(dev_root, arm, st), pics(new_root, arm, st)
            same = a == b
            bad += not same
            print(f"[{arm:9s}] {st:32s} {'SAME' if same else 'DIFF'}  dev {a}" + ("" if same else f"\n{'':44s}new {b}"))
    print("shipcheck:", "PASS (identical pictures)" if not bad else f"FAIL ({bad} clip-arm pairs differ)")


if __name__ == "__main__":
    main()
