"""Wrong-dropping stack (Adam 10 Oct): start from R1 + R3 (scene_visible own_ab, then weapon pictures dropped when Omni's
closed choice prefers another option, m = 0) and add one more dropper: the energy rise at the picture start
(salience.py; band full / hi, drop when rise < r dB) or a weak detector peak (BEATs / FlexSED max under the picture < p).
Clip-grouped 5-fold CV (seed 0) on all 158 clips by w=2 and w=4.
    python benchmark/gold/coverage/wrong_stack.py   (cluster CPU from ~/wt_slice) -> wrong_stack.md"""
import itertools, json, random, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import decide, pictures, stems, IN, GOLD
from benchmark.gold.coverage import scene_visible as SV, salience as SAL

HERE = Path(__file__).resolve().parent
WEAPON = ("Gunshot", "Explosion", "Machine gun", "Fireworks", "Artillery")
GRID = [None] + [("rise", b, r) for b, r in itertools.product(("full", "hi"), (-3.0, 0.0, 2.0, 4.0))] + [("peak", None, p) for p in (0.3, 0.4, 0.5, 0.6)]


def main():
    gold = S.load_gold([GOLD]); config.use_shipped(); config.MAX_AFTER_END = None
    config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
    config.FLASH_CACHE, config.HOLD_CURVES, config.HOLD_FLEXSED_DIR = str(IN / "flashes.json"), str(IN / "flexsed_curves.json"), str(IN / "x")
    st5 = json.loads((IN / "stage5_specs.json").read_text()); dur = json.loads((IN / "durations.json").read_text())
    allc = stems("dev") + stems("test"); dev = set(stems("dev"))
    bursts, om, bf = {}, {}, {}
    for suf in ("dev", "test"):
        for b in json.loads((HERE / f"verify_items_{suf}.json").read_text())["bursts"]:
            bursts.setdefault(b["clip"], []).append(b)
        om.update(json.loads((HERE / f"omni_verify_{suf}.json").read_text())["v1"])
        bf.update({k + "|" + suf: v for k, v in json.loads((HERE / f"burst_features_{suf}.json").read_text()).items()})
    base, feat = {}, {}
    for st in allc:
        pics = pictures(SV.apply(decide(st5[st]["P"], st5[st]["B"], st5[st]["gate"]), st5[st]["gate"], "own_ab"), float(dur[st] or 10), st)
        keep, fs = [], []
        suf = "dev" if st in dev else "test"
        for lab, a, b in pics:
            P, n, peak = {}, 0, None
            for bu in bursts.get(st, []):
                if not (S.same_family(lab, bu["family"]) and bu["end"] > a - 0.5 and bu["start"] < b + 0.5):
                    continue
                f = bf.get(bu["id"] + "|" + suf, {})
                pk = max([x for x in (f.get("beats_max"), f.get("flex_max")) if x is not None] or [0.0])
                peak = pk if peak is None else max(peak, pk)
                if bu["id"] not in om:
                    continue
                n += 1
                for o in ("fwd", "rev"):
                    for opt, p in zip(om[bu["id"]][o]["options"], om[bu["id"]][o]["p"]):
                        P[opt] = P.get(opt, 0.0) + p / 2
            if n and lab.startswith(WEAPON):
                own = max([v for k, v in P.items() if S.same_family(k, lab)] or [0.0])
                if max([v for k, v in P.items() if not S.same_family(k, lab)] or [0.0]) > own:
                    continue
            keep.append((lab, a, b)); fs.append(peak)
        base[st] = keep
        feat[st] = list(zip(SAL.rises(st, keep), fs))

    def apply(st, g):
        if g is None:
            return base[st]
        kind, band, t = g
        if kind == "rise":
            return [p for p, (r, _) in zip(base[st], feat[st]) if r[band] >= t]
        return [p for p, (_, pk) in zip(base[st], feat[st]) if pk is None or pk >= t]

    memo = {}
    row = lambda st, g: memo.setdefault((st, g), S.score_clip(gold[st], apply(st, g)))
    wr = lambda rr: sum(x["visible"] + x["cross"] + x["phantom"] for x in rr)
    cost = lambda rr, w: (4 * sum(x["miss"] for x in rr) + w * wr(rr)) / len(rr)
    L = ["# Wrong-dropping stack on top of R1 + R3, all 158 clips", "", "| added dropper | hits | wrong | cost (w=2) | cost (w=4) |", "|---|---|---|---|---|"]
    for g in GRID:
        rr = [row(st, g) for st in allc]
        L.append(f"| {g or 'R1 + R3'} | {sum(x['hit'] for x in rr)} | {wr(rr)} | {cost(rr, 2):.3f} | {cost(rr, 4):.3f} |")
    for w in (2, 4):
        rng = random.Random(0); sh = allc[:]; rng.shuffle(sh); oof, ch = {}, []
        for k in range(5):
            te = set(sh[k::5]); tr = [c for c in allc if c not in te]
            best = min(GRID, key=lambda g: cost([row(st, g) for st in tr], w)); ch.append(best)
            for st in te:
                oof[st] = row(st, best)
        rr = [oof[st] for st in allc]
        L += ["", f"CV by w={w}: choices {ch}; out of fold hits {sum(x['hit'] for x in rr)}, wrong {wr(rr)}, cost w=2 {cost(rr, 2):.3f}, w=4 {cost(rr, 4):.3f}"]
    (HERE / "wrong_stack.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
