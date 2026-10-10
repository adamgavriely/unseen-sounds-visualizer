"""Fable idea 5: how small a difference can 158 clips show? Clip-level paired bootstrap of onset cost, v1.4 vs v1.3.11,
and the width of a single-system 95 % interval.   python benchmark/gold/coverage/noise_floor.py -> noise_floor.md"""
import json, sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import decide, pictures, stems, IN, GOLD

HERE = Path(__file__).resolve().parent


def run(v14):
    config.use_shipped(); config.MAX_AFTER_END = None
    if not v14:
        config.VISIBILITY_RULE, config.FLASH_RULE, config.PICTURE_BAN, config.HOLD_FLEXSED = "majority", False, None, None
    config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
    config.FLASH_CACHE, config.HOLD_CURVES, config.HOLD_FLEXSED_DIR = str(IN / "flashes.json"), str(IN / "flexsed_curves.json"), str(IN / "x")
    st5 = json.loads((IN / "stage5_specs.json").read_text()); dur = json.loads((IN / "durations.json").read_text())
    return [S.viewer_cost([S.score_clip(GOLD_SET[st], pictures(decide(st5[st]["P"], st5[st]["B"], st5[st]["gate"]), float(dur[st] or 10), st))])
            for st in stems("dev") + stems("test")]


def main():
    global GOLD_SET
    GOLD_SET = S.load_gold([GOLD])
    a, b = np.array(run(True)), np.array(run(False))
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(a), (100000, len(a)))
    d = (a - b)[idx].mean(1); m = a[idx].mean(1)
    L = ["# Noise floor of the 158-clip benchmark", "",
         f"v1.4 cost {a.mean():.3f} (95 % interval {np.percentile(m, 2.5):.3f} .. {np.percentile(m, 97.5):.3f}, width {np.percentile(m, 97.5) - np.percentile(m, 2.5):.3f})",
         f"v1.4 - v1.3.11: {d.mean():+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}], p {min(1, 2 * min((d >= 0).mean(), (d <= 0).mean())):.3f}",
         f"one hit = {4 / len(a):.3f} cost; one wrong = {2 / len(a):.3f} cost; paired-difference interval half-width = "
         f"{(np.percentile(d, 97.5) - np.percentile(d, 2.5)) / 2:.3f}"]
    (HERE / "noise_floor.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
