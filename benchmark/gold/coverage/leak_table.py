"""Leak table (Fable, 10 Oct; logging only): for each needed sound (importance 2-3) v1.4 misses, is there a raw
candidate of its family whose start is in the hit window (-0.5..+1.0 s)? If so, at which step was the best such
candidate dropped (decision trail, docs/decision_trail/data.js), or, if the trail kept it, which v1.4 rule removed it.
    python benchmark/gold/coverage/leak_table.py -> leak_table.md"""
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
    s = (_ROOT / "docs" / "decision_trail" / "data.js").read_text(encoding="utf-8")
    dj = {c["clip"]: c for c in json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))["clips"]}
    where, rows, nmiss, nocand = Counter(), [], 0, 0
    for st in stems("dev") + stems("test"):
        pics = pictures(decide(st5[st]["P"], st5[st]["B"], st5[st]["gate"]), float(dur[st] or 10), st)
        for g in gold[st]:
            if not (g["needed"] and g["importance"] >= 2):
                continue
            if any(S.same_family(l, g["label"]) and S.in_window(a, g["start"], S.EARLY, S.LATE) for l, a, b in pics):
                continue
            nmiss += 1
            cs = [c for c in dj[st]["cands"] if S.same_family(c["label"], g["label"]) and S.in_window(c["start"], g["start"], S.EARLY, S.LATE)]
            if not cs:
                nocand += 1; continue
            kept = [c for c in cs if c["fate"] != "dropped"]
            if kept:
                step = "kept by the trail; removed by a v1.4 rule (a/b gate, flash, texture ban) or merged"
            else:
                order = [x["step"] for x in dj[st]["cands"][0]["trail"]]
                step = max(cs, key=lambda c: len(c["trail"]))["at"]
            where[step] += 1; rows.append((st, g["label"], step))
    L = ["# Leak table: where v1.4 loses needed sounds that a raw candidate had, all 158 clips", "",
         f"needed sounds missed: {nmiss}; with no in-window raw candidate of the family: {nocand}; with one: {nmiss - nocand}", "",
         "| step that dropped the latest-surviving candidate | sounds |", "|---|---|"] + [f"| {k} | {n} |" for k, n in where.most_common()] + \
        ["", "| clip | gold | step |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
    tot, good = Counter(), Counter()
    for st in stems("dev") + stems("test"):
        for c in dj[st]["cands"]:
            if c["fate"] == "dropped":
                tot[c["at"]] += 1
                good[c["at"]] += any(g["needed"] and S.same_family(c["label"], g["label"]) and S.in_window(c["start"], g["start"], S.EARLY, S.LATE) for g in gold[st])
    L += ["", "Undoing a step pays only if at least 1 in 3 of its drops is a real needed sound (one hit = two wrongs).", "",
          "| step | candidates dropped | in a needed sound's hit window | share |", "|---|---|---|---|"] +          [f"| {k} | {n} | {good[k]} | {good[k] / n:.0%} |" for k, n in tot.most_common() if n >= 20]
    (HERE / "leak_table.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
