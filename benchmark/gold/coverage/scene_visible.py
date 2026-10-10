"""Wrong-dropping R1 (Adam 10 Oct: focus on dropping wrongs; Fable): suppress pictures whose sound may be on screen.
On the stored stage-5 plan after the v1.4 a/b decision, a drawn sound is turned off when
  own:  any of its own gate stretches is seen (by the a/b answer, or by any of name / a/b / describe), or
  other: its time span overlaps a different sound that the gate found on screen (res drop, "source visible").
Variants: none, own_ab, own_any, other, own_ab+other, own_any+other. Two objectives, clip-grouped 5-fold CV (seed 0) on
all 158 clips: onset cost (w=2) and w=4 (one hit = one wrong, the "fewer wrongs" profile).
    python benchmark/gold/coverage/scene_visible.py -> scene_visible.md"""
import copy, json, random, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import decide, pictures, stems, IN, GOLD

HERE = Path(__file__).resolve().parent
GRID = [None, "own_ab", "own_any", "other", "own_ab+other", "own_any+other"]
TOL = 0.05


def spans(s):
    return s.get("spans") or [[s["start"], s["end"]]]


def apply(P, gates, v):
    if v is None:
        return P
    P = copy.deepcopy(P)
    vis = [s for s in P if not s.get("augment") and str(s.get("reason", "")).startswith("source visible on screen")]
    for s in P:
        if not s.get("augment"):
            continue
        g = [x for x in gates if x["label"] == s["event_label"] and abs(x["start"] - s["start"]) < TOL and abs(x["end"] - s["end"]) < TOL]
        st = g[0]["stretches"] if g else []
        off = False
        if "own_ab" in v:
            off |= any(x["ab"] for x in st)
        if "own_any" in v:
            off |= any(x["seen"] or x["name"] or x["ab"] or x["desc"] for x in st)
        if "other" in v:
            off |= any(o["event_label"] != s["event_label"] and any(a < s["end"] and b > s["start"] for a, b in spans(o)) for o in vis)
        if off:
            s["augment"] = False
    return P


def main():
    gold = S.load_gold([GOLD]); config.use_shipped(); config.MAX_AFTER_END = None
    config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
    config.FLASH_CACHE, config.HOLD_CURVES, config.HOLD_FLEXSED_DIR = str(IN / "flashes.json"), str(IN / "flexsed_curves.json"), str(IN / "x")
    st5 = json.loads((IN / "stage5_specs.json").read_text()); dur = json.loads((IN / "durations.json").read_text())
    allc = stems("dev") + stems("test")
    memo = {}
    row = lambda st, v: memo.setdefault((st, v), S.score_clip(gold[st], pictures(apply(decide(st5[st]["P"], st5[st]["B"], st5[st]["gate"]), st5[st]["gate"], v), float(dur[st] or 10), st)))
    wr = lambda rr: sum(x["visible"] + x["cross"] + x["phantom"] for x in rr)
    cost = lambda rr, w: (4 * sum(x["miss"] for x in rr) + w * wr(rr)) / len(rr)
    L = ["# Scene-visible suppress (R1), all 158 clips", "", "| variant | hits | wrong | cost (w=2) | cost (w=4) |", "|---|---|---|---|---|"]
    for v in GRID:
        rr = [row(st, v) for st in allc]
        L.append(f"| {v or 'v1.4'} | {sum(x['hit'] for x in rr)} | {wr(rr)} | {cost(rr, 2):.3f} | {cost(rr, 4):.3f} |")
    for w in (2, 4):
        rng = random.Random(0); sh = allc[:]; rng.shuffle(sh); oof, ch = {}, []
        for k in range(5):
            te = set(sh[k::5]); tr = [c for c in allc if c not in te]
            best = min(GRID, key=lambda v: cost([row(st, v) for st in tr], w)); ch.append(best)
            for st in te:
                oof[st] = row(st, best)
        rr = [oof[st] for st in allc]
        L += ["", f"CV by w={w}: choices {ch}; out of fold hits {sum(x['hit'] for x in rr)}, wrong {wr(rr)}, cost w=2 {cost(rr, 2):.3f}, w=4 {cost(rr, 4):.3f}"]
    (HERE / "scene_visible.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
