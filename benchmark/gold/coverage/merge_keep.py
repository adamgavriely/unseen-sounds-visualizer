"""Family-merge keep (Adam 10 Oct: keep working). The family-merge step folds a candidate into another family's sound;
6 needed sounds are lost there (frozen trail). Letting all of them through gave +4 hits / +19 wrong (filter_bypass.md).
Here only merged-away candidates with strong evidence of their OWN family come back: burst DASM max >= d or FlexSED max
>= f (Step 4 burst features). Added as pictures as in filter_bypass.py, on top of v1.4; (d, f) chosen by clip-grouped
5-fold CV on onset cost.   python benchmark/gold/coverage/merge_keep.py -> merge_keep.md"""
import itertools, json, random, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import decide, pictures, stems, IN, GOLD
from benchmark.gold.coverage.step11_policies import TEXTURE
from src.labels import canonical, is_music

HERE = Path(__file__).resolve().parent
GRID = [None] + list(itertools.product((0.35, 0.5, 0.7, 9), (0.5, 0.7, 0.9, 9)))


def main():
    gold = S.load_gold([GOLD]); config.use_shipped(); config.MAX_AFTER_END = None
    config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
    config.FLASH_CACHE, config.HOLD_CURVES, config.HOLD_FLEXSED_DIR = str(IN / "flashes.json"), str(IN / "flexsed_curves.json"), str(IN / "x")
    st5 = json.loads((IN / "stage5_specs.json").read_text()); dur = json.loads((IN / "durations.json").read_text())
    allc = stems("dev") + stems("test"); dev = set(stems("dev"))
    s = (_ROOT / "docs" / "decision_trail" / "data.js").read_text(encoding="utf-8")
    dj = json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))
    merged = {c["clip"]: [x for x in c["cands"] if x.get("at") == "family_merge" and canonical(x["label"]) not in ("Speech", "Music")
                          and not is_music(x["label"]) and canonical(x["label"]) not in TEXTURE] for c in dj["clips"]}
    feat, mem = {}, {}
    for suf in ("dev", "test"):
        bf = json.loads((HERE / f"burst_features_{suf}.json").read_text())
        for b in json.loads((HERE / f"verify_items_{suf}.json").read_text())["bursts"]:
            for m in b["members"]:
                mem[(b["clip"], m)] = bf.get(b["id"], {})
    base = {st: pictures(decide(st5[st]["P"], st5[st]["B"], st5[st]["gate"]), float(dur[st] or 10), st) for st in allc}

    def apply(st, g):
        if g is None:
            return base[st]
        d, f = g
        pics = list(base[st])
        for x in sorted(merged.get(st, []), key=lambda x: x["start"]):
            fe = mem.get((st, x["id"]), {})
            strong = (fe.get("dasm_max") or 0) >= d or (fe.get("flex_max") or 0) >= f
            fam = canonical(x["label"])
            if strong and not any(S.same_family(p[0], fam) and p[2] > x["start"] - 0.5 and p[1] < x["end"] + 0.5 for p in pics):
                pics.append((fam, x["start"], max(x["end"], x["start"] + 1.5)))
        return sorted(pics, key=lambda p: p[1])

    memo = {}
    row = lambda st, g: memo.setdefault((st, g), S.score_clip(gold[st], apply(st, g)))
    summ = lambda rr: (S.aggregate(rr)["hits"], sum(x["visible"] + x["cross"] + x["phantom"] for x in rr), S.viewer_cost(rr))
    rng = random.Random(0); sh = allc[:]; rng.shuffle(sh); oof, ch = {}, []
    for k in range(5):
        te = set(sh[k::5]); tr = [c for c in allc if c not in te]
        best = min(GRID, key=lambda g: S.viewer_cost([row(st, g) for st in tr])); ch.append(best)
        for st in te:
            oof[st] = row(st, best)
    L = ["# Family-merge keep, all 158 clips", "", f"merged-away candidates: {sum(len(v) for st, v in merged.items() if st in set(allc))}",
         f"v1.4: {summ([row(st, None) for st in allc])}", f"CV choices {ch}; out of fold: {summ([oof[st] for st in allc])}", "",
         "| DASM >= d, FlexSED >= f | hits | wrong | cost |", "|---|---|---|---|"]
    for g in GRID[1:]:
        h, w, c = summ([row(st, g) for st in allc]); L.append(f"| {g} | {h} | {w} | {c:.3f} |")
    (HERE / "merge_keep.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
