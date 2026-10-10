"""Sibling-confusion drop-only veto (Fable idea 8; Adam 10 Oct: keep working). On top of v1.4, all 158 clips.
For each picture, the overlapping candidate burst(s) of its family with Qwen3-Omni's closed choice {family, AudioSet
siblings, none} (Step 3b, mean of the two option orders). The picture is removed when Omni's top option is a different
family (not "none") ahead of the own family by more than m, and the burst's strongest detector peak (BEATs / FlexSED
max in the burst) is below p. m in {0, 0.2, 0.4}, p in {0.5, 0.7, 1.01}, or off; clip-grouped 5-fold CV on onset cost.
Never relabels (relabelling failed in Step 8).   python benchmark/gold/coverage/sibling_drop.py -> sibling_drop.md"""
import itertools, json, random, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import decide, pictures, stems, IN, GOLD

HERE = Path(__file__).resolve().parent
GRID = [None] + list(itertools.product((0.0, 0.2, 0.4), (0.5, 0.7, 1.01)))


def main():
    gold = S.load_gold([GOLD]); config.use_shipped(); config.MAX_AFTER_END = None
    config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
    config.FLASH_CACHE, config.HOLD_CURVES, config.HOLD_FLEXSED_DIR = str(IN / "flashes.json"), str(IN / "flexsed_curves.json"), str(IN / "x")
    st5 = json.loads((IN / "stage5_specs.json").read_text()); dur = json.loads((IN / "durations.json").read_text())
    allc = stems("dev") + stems("test")
    bursts, om, bf = {}, {}, {}
    for suf in ("dev", "test"):
        for b in json.loads((HERE / f"verify_items_{suf}.json").read_text())["bursts"]:
            bursts.setdefault(b["clip"], []).append(b)
        om.update(json.loads((HERE / f"omni_verify_{suf}.json").read_text())["v1"])
        bf.update({k + "|" + suf: v for k, v in json.loads((HERE / f"burst_features_{suf}.json").read_text()).items()})
    suf_of = {st: ("dev" if st in set(stems("dev")) else "test") for st in allc}
    pics = {st: pictures(decide(st5[st]["P"], st5[st]["B"], st5[st]["gate"]), float(dur[st] or 10), st) for st in allc}

    def verdict(st, lab, a, b):
        """(margin of the best other sibling over the own family, max detector peak) of the best-matching burst"""
        best = None
        for bu in bursts.get(st, []):
            if not (S.same_family(lab, bu["family"]) and bu["end"] > a - 0.5 and bu["start"] < b + 0.5) or bu["id"] not in om:
                continue
            r = om[bu["id"]]
            P = {}
            for o in ("fwd", "rev"):
                for opt, p in zip(r[o]["options"], r[o]["p"]):
                    P[opt] = P.get(opt, 0.0) + p / 2
            own = P.get(bu["family"], 0.0)
            others = [v for k, v in P.items() if k not in (bu["family"], "none of these")]
            f = bf.get(bu["id"] + "|" + suf_of[st], {})
            peak = max([x for x in (f.get("beats_max"), f.get("flex_max")) if x is not None] or [0.0])
            m = (max(others) if others else 0.0) - own
            if best is None or m < best[0]:
                best = (m, peak)
        return best

    V = {st: [verdict(st, *p) for p in pics[st]] for st in allc}

    def apply(st, g):
        if g is None:
            return pics[st]
        m, p = g
        return [x for x, v in zip(pics[st], V[st]) if not (v and v[0] > m and v[1] < p)]

    memo = {}
    row = lambda st, g: memo.setdefault((st, g), S.score_clip(gold[st], apply(st, g)))
    summ = lambda rr: (S.aggregate(rr)["hits"], sum(x["visible"] + x["cross"] + x["phantom"] for x in rr), S.viewer_cost(rr))
    rng = random.Random(0); sh = allc[:]; rng.shuffle(sh); oof, ch = {}, []
    for k in range(5):
        te = set(sh[k::5]); tr = [c for c in allc if c not in te]
        best = min(GRID, key=lambda g: S.viewer_cost([row(st, g) for st in tr])); ch.append(best)
        for st in te:
            oof[st] = row(st, best)
    L = ["# Sibling-confusion drop-only veto, all 158 clips", "", f"v1.4: {summ([row(st, None) for st in allc])}",
         f"CV choices {ch}; out of fold: {summ([oof[st] for st in allc])}", "", "| m, p | hits | wrong | cost |", "|---|---|---|---|"]
    for g in GRID[1:]:
        h, w, c = summ([row(st, g) for st in allc]); L.append(f"| {g} | {h} | {w} | {c:.3f} |")
    (HERE / "sibling_drop.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
