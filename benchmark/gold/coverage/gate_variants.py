"""In-between on-screen rules on the stored votes (all 158 clips, clip-grouped 5-fold CV, onset cost), on top of the
other v1.4 rules. A stretch is "seen" under:
  M      majority of name / a-b / describe (v1.3.11)
  AB     a-b decides, split -> majority (v1.4)
  AB|ND  a-b says seen, or name AND describe both say seen
  AB|N   a-b says seen, or name says seen
  AB&M   a-b says seen and the majority says seen (split -> majority)
A sound is silenced iff every stretch is seen.   python benchmark/gold/coverage/gate_variants.py -> gate_variants.md"""
import copy, json, random, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import pictures, stems, IN, GOLD, TOL

HERE = Path(__file__).resolve().parent
RULES = {
    "M": lambda v: v["seen"],
    "AB": lambda v: v["seen"] if v["ab"] is None else bool(v["ab"]),
    "AB|ND": lambda v: (bool(v["ab"]) if v["ab"] is not None else v["seen"]) or (v["name"] is True and v["desc"] is True),
    "AB|N": lambda v: (bool(v["ab"]) if v["ab"] is not None else v["seen"]) or v["name"] is True,
    "AB&M": lambda v: v["seen"] and (bool(v["ab"]) if v["ab"] is not None else True),
}


def decide(P, B, gates, seen):
    P = copy.deepcopy(P); drawn = set()
    for i, (s, t) in enumerate(zip(P, B)):
        if not t.get("augment") or t["event_label"] != s["event_label"]:
            continue
        g = [x for x in gates if x["label"] == s["event_label"] and abs(x["start"] - s["start"]) < TOL and abs(x["end"] - s["end"]) < TOL]
        if not g or not g[0]["stretches"]:
            continue
        silence = all(seen(v) for v in g[0]["stretches"])
        gated = (not s.get("augment")) and str(s.get("reason", "")).startswith("source visible on screen")
        if gated and not silence:
            P[i] = dict(t); drawn.add(s["event_label"])
        elif s.get("augment") and g[0]["res"] == "pass" and silence:
            P[i] = dict(s); P[i]["augment"] = False
    for i, (s, t) in enumerate(zip(P, B)):
        if not s.get("augment") and t.get("augment") and str(s.get("reason", "")).startswith("a kind of "):
            if s["reason"][len("a kind of "):].split(",")[0] in drawn:
                P[i] = dict(t)
    return P


def main():
    gold = S.load_gold([GOLD]); config.use_shipped(); config.MAX_AFTER_END = None
    config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
    config.FLASH_CACHE, config.HOLD_CURVES, config.HOLD_FLEXSED_DIR = str(IN / "flashes.json"), str(IN / "flexsed_curves.json"), str(IN / "x")
    st5 = json.loads((IN / "stage5_specs.json").read_text()); dur = json.loads((IN / "durations.json").read_text())
    allc = stems("dev") + stems("test")
    rows = {(r, st): S.score_clip(gold[st], pictures(decide(st5[st]["P"], st5[st]["B"], st5[st]["gate"], f), float(dur[st] or 10), st))
            for r, f in RULES.items() for st in allc}
    summ = lambda rr: (S.aggregate(rr)["hits"], sum(x["visible"] + x["cross"] + x["phantom"] for x in rr), S.viewer_cost(rr))
    rng = random.Random(0); sh = allc[:]; rng.shuffle(sh); oof, ch = {}, []
    for k in range(5):
        te = set(sh[k::5]); tr = [c for c in allc if c not in te]
        best = min(RULES, key=lambda r: S.viewer_cost([rows[(r, st)] for st in tr])); ch.append(best)
        for st in te:
            oof[st] = rows[(best, st)]
    L = ["# In-between on-screen rules, all 158 clips", "", "| rule | hits | wrong | cost |", "|---|---|---|---|"]
    for r in RULES:
        h, w, c = summ([rows[(r, st)] for st in allc]); L.append(f"| {r} | {h} | {w} | {c:.3f} |")
    h, w, c = summ([oof[st] for st in allc])
    L += ["", f"CV choices: {ch}; out of fold {h} / {w}, cost {c:.3f}"]
    (HERE / "gate_variants.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
