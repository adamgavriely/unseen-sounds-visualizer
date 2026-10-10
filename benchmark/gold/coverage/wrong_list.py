"""Every v1.4 wrong picture on all 158 clips (Adam 10 Oct: focus on dropping wrongs): class (visible / cross / phantom),
picture label and time, and the gold sounds playing then (label, needed, importance).
    python benchmark/gold/coverage/wrong_list.py -> wrong_list.md"""
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
    L = ["# v1.4 wrong pictures, all 158 clips", "", "| clip | class | picture | start-end s | gold playing (needed, importance) |", "|---|---|---|---|---|"]
    n = Counter(); fam = Counter()
    for st in stems("dev") + stems("test"):
        pics = pictures(decide(st5[st]["P"], st5[st]["B"], st5[st]["gate"]), float(dur[st] or 10), st)
        for p in pics:
            r = S.score_clip(gold[st], [p])
            if r["hit"] or r["dup"] or r["collision"] or r["dontcare"]:
                continue
            cls = "visible" if r["visible"] else "cross" if r["cross"] else "phantom" if r["phantom"] else None
            if cls is None:
                continue
            # a picture alone may be "visible" only if no needed same-family sound; check in full context
            full = S.score_clip(gold[st], pics); 
            lab, a, b = p
            play = [f"{g['label']} ({'N' if g['needed'] else 'on'}{g['importance']})" for g in gold[st]
                    if S.in_window(a, g["start"], S.EARLY, S.LATE) or g["start"] <= a <= g["end"]]
            n[cls] += 1; fam[lab] += 1
            L.append(f"| {st} | {cls} | {lab} | {a:.1f}-{b:.1f} | {'; '.join(play) or '-'} |")
    L += ["", f"totals {dict(n)}", "", "by picture label: " + ", ".join(f"{k} {v}" for k, v in fam.most_common())]
    (HERE / "wrong_list.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
