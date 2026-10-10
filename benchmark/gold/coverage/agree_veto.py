"""Agreement veto (Adam 10 Oct: keep working). Each single veto (OWLv2 object, energy rise, Omni sibling preference,
weak detector peak) removed about one hit per wrong picture. Here a picture is removed only when at least k of them agree.
Signals per v1.4 picture: owl >= 0.3 (owl_scores.json), full-band rise < 0 dB (salience.py), Omni sibling margin > 0.2
(sibling_drop.py), burst peak (BEATs / FlexSED max) < 0.5. k in {2, 3, 4} or off, chosen by clip-grouped 5-fold CV on
onset cost, all 158 clips.   python benchmark/gold/coverage/agree_veto.py   (cluster CPU from ~/wt_slice) -> agree_veto.md"""
import json, random, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import decide, pictures, stems, IN, GOLD
from benchmark.gold.coverage import salience as SAL

HERE = Path(__file__).resolve().parent
GRID = [None, 2, 3, 4]


def main():
    gold = S.load_gold([GOLD]); config.use_shipped(); config.MAX_AFTER_END = None
    config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
    config.FLASH_CACHE, config.HOLD_CURVES, config.HOLD_FLEXSED_DIR = str(IN / "flashes.json"), str(IN / "flexsed_curves.json"), str(IN / "x")
    st5 = json.loads((IN / "stage5_specs.json").read_text()); dur = json.loads((IN / "durations.json").read_text())
    allc = stems("dev") + stems("test"); dev = set(stems("dev"))
    pics = {st: pictures(decide(st5[st]["P"], st5[st]["B"], st5[st]["gate"]), float(dur[st] or 10), st) for st in allc}
    owl = json.loads((HERE / "owl_scores.json").read_text())
    bursts, om, bf = {}, {}, {}
    for suf in ("dev", "test"):
        for b in json.loads((HERE / f"verify_items_{suf}.json").read_text())["bursts"]:
            bursts.setdefault(b["clip"], []).append(b)
        om.update(json.loads((HERE / f"omni_verify_{suf}.json").read_text())["v1"])
        bf.update({k + "|" + suf: v for k, v in json.loads((HERE / f"burst_features_{suf}.json").read_text()).items()})
    sig = {}
    for st in allc:
        rises = SAL.rises(st, pics[st])
        suf = "dev" if st in dev else "test"
        out = []
        for (lab, a, b), r in zip(pics[st], rises):
            m, peak = None, None
            for bu in bursts.get(st, []):
                if not (S.same_family(lab, bu["family"]) and bu["end"] > a - 0.5 and bu["start"] < b + 0.5) or bu["id"] not in om:
                    continue
                P = {}
                for o in ("fwd", "rev"):
                    for opt, p in zip(om[bu["id"]][o]["options"], om[bu["id"]][o]["p"]):
                        P[opt] = P.get(opt, 0.0) + p / 2
                others = [v for k, v in P.items() if k not in (bu["family"], "none of these")]
                mm = (max(others) if others else 0.0) - P.get(bu["family"], 0.0)
                f = bf.get(bu["id"] + "|" + suf, {})
                pk = max([x for x in (f.get("beats_max"), f.get("flex_max")) if x is not None] or [0.0])
                m = mm if m is None else min(m, mm); peak = pk if peak is None else max(peak, pk)
            n = int(owl.get(f"{st}|{lab}|{a:.3f}", 0.0) >= 0.3) + int(r["full"] < 0.0) + int(m is not None and m > 0.2) + int(peak is not None and peak < 0.5)
            out.append(n)
        sig[st] = out
    apply = lambda st, k: pics[st] if k is None else [p for p, n in zip(pics[st], sig[st]) if n < k]
    memo = {}
    row = lambda st, g: memo.setdefault((st, g), S.score_clip(gold[st], apply(st, g)))
    summ = lambda rr: (S.aggregate(rr)["hits"], sum(x["visible"] + x["cross"] + x["phantom"] for x in rr), S.viewer_cost(rr))
    rng = random.Random(0); sh = allc[:]; rng.shuffle(sh); oof, ch = {}, []
    for k in range(5):
        te = set(sh[k::5]); tr = [c for c in allc if c not in te]
        best = min(GRID, key=lambda g: S.viewer_cost([row(st, g) for st in tr])); ch.append(best)
        for st in te:
            oof[st] = row(st, best)
    L = ["# Agreement veto, all 158 clips", "", f"v1.4: {summ([row(st, None) for st in allc])}", f"CV choices {ch}; out of fold: {summ([oof[st] for st in allc])}",
         "", "| k signals | hits | wrong | cost |", "|---|---|---|---|"]
    for g in GRID[1:]:
        h, w, c = summ([row(st, g) for st in allc]); L.append(f"| {g} | {h} | {w} | {c:.3f} |")
    (HERE / "agree_veto.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
