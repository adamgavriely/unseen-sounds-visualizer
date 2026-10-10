"""Sibling relabel, upper bound (Fable idea 2, check before training): for each v1.4 cross-trigger picture (wrong family
while a real sound plays), is there a needed, still-missed gold sound whose onset window holds the picture start? If so,
the right label would turn one wrong into one hit. Lists the label pairs.   python benchmark/gold/coverage/relabel_ceiling.py"""
import json, sys
from collections import Counter
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import decide, pictures, stems, IN, GOLD

HERE = Path(__file__).resolve().parent


def main():
    gold = S.load_gold([GOLD]); config.use_shipped(); config.MAX_AFTER_END = None
    config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
    config.FLASH_CACHE, config.HOLD_CURVES, config.HOLD_FLEXSED_DIR = str(IN / "flashes.json"), str(IN / "flexsed_curves.json"), str(IN / "x")
    st5 = json.loads((IN / "stage5_specs.json").read_text()); dur = json.loads((IN / "durations.json").read_text())
    pairs, n_cross, gain = Counter(), 0, 0
    for st in stems("dev") + stems("test"):
        pics = pictures(decide(st5[st]["P"], st5[st]["B"], st5[st]["gate"]), float(dur[st] or 10), st)
        base = S.score_clip(gold[st], pics)
        n_cross += base["cross"]
        for i, (lab, a, b) in enumerate(pics):
            if any(S.same_family(lab, g["label"]) and S.in_window(a, g["start"], S.EARLY, S.LATE) for g in gold[st]):
                continue
            for g in gold[st]:
                if not (g["needed"] and g["importance"] >= 2 and S.in_window(a, g["start"], S.EARLY, S.LATE)):
                    continue
                alt = pics[:i] + [(g["label"], a, b)] + pics[i + 1:]
                r = S.score_clip(gold[st], alt)
                if r["hit"] > base["hit"]:
                    gain += 1; pairs[(lab, g["label"])] += 1; break
    L = ["# Sibling relabel: upper bound on v1.4, all 158 clips", "", f"cross-trigger wrongs: {n_cross}; a relabel would turn {gain} of them into hits", "",
         "| picture label | gold label | n |", "|---|---|---|"] + [f"| {a} | {b} | {n} |" for (a, b), n in pairs.most_common()]
    (HERE / "relabel_ceiling.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
