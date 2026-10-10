"""Per-family hold bars (Adam 10 Oct: keep working). v1.4 holds every picture while FlexSED >= 0.5. Here each
canonical sound family with >= 3 pictures on the training clips gets its own bar from {off, 0.3, 0.4, 0.5, 0.6, 0.7}
(the others keep 0.5), chosen per family on the training clips by J = cost_cov + 0.5 x (wrong + stale s) per clip;
clip-grouped 5-fold CV on all 158 clips, scored out of fold. Hits / wrong never change (ends only).

    python benchmark/gold/coverage/family_hold.py   (cluster CPU from ~/wt_slice) -> family_hold.md
"""
import json, random, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.coverage import score_coverage as V
from benchmark.gold.coverage.pooled_hold import hold, J
from benchmark.gold.v14_score import decide, pictures, stems, IN, GOLD
from src.labels import canonical

HERE = Path(__file__).resolve().parent
BARS = [None, 0.3, 0.4, 0.5, 0.6, 0.7]


def main():
    gold = S.load_gold([GOLD]); config.use_shipped(); config.MAX_AFTER_END = None
    config.HOLD_FLEXSED = None                       # pictures before the hold; the hold is applied below
    config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
    config.FLASH_CACHE = str(IN / "flashes.json")
    st5 = json.loads((IN / "stage5_specs.json").read_text()); dur = json.loads((IN / "durations.json").read_text())
    curves = json.loads((IN / "flexsed_curves.json").read_text())
    allc = stems("dev") + stems("test")
    base = {st: pictures(decide(st5[st]["P"], st5[st]["B"], st5[st]["gate"]), float(dur[st] or 10), st) for st in allc}
    held = {}
    for st in allc:
        for b in BARS:
            held[(st, b)] = base[st] if b is None else hold(base[st], curves.get(st, {"labels": {}}), (b, None, False, 0.0, "extend"), float(dur[st] or 10))

    def build(st, bars):
        out = []
        for i, p in enumerate(base[st]):
            b = bars.get(canonical(p[0]), 0.5)
            out.append(held[(st, b)][i])
        return out

    fams = sorted({canonical(p[0]) for st in allc for p in base[st]})
    memo = {}

    def rows(bars, clips):
        k = tuple(sorted(bars.items(), key=str))
        return [memo.setdefault((k, st), V.score_clip_v2(gold[st], build(st, bars))) for st in clips]

    rng = random.Random(0); sh = allc[:]; rng.shuffle(sh); oof, chosen = {}, []
    for k in range(5):
        te = set(sh[k::5]); tr = [c for c in allc if c not in te]
        bars = {}
        for f in fams:
            if sum(1 for st in tr for p in base[st] if canonical(p[0]) == f) < 3:
                continue
            bars[f] = min(BARS, key=lambda b: J(rows({**bars, f: b}, tr))[0])
        chosen.append({f: b for f, b in bars.items() if b != 0.5})
        for st, r in zip(sorted(te), rows(bars, sorted(te))):
            oof[st] = r
    j0, a0 = J(rows({}, allc)); j1, a1 = J([oof[st] for st in allc])
    L = ["# Per-family hold bars, all 158 clips (out of fold)", "",
         f"v1.4 (bar 0.5 for all): J {j0:.3f}, cost_cov {a0['cost_cov']:.3f}, cover {a0['hit_cov']:.2f}, |end err| {a0['end_abs_med']:.2f} s, "
         f"wrong s/clip {a0['wrong_s_per_clip']:.2f}, stale s/clip {a0['stale_s_per_clip']:.2f}",
         f"per-family bars: J {j1:.3f}, cost_cov {a1['cost_cov']:.3f}, cover {a1['hit_cov']:.2f}, |end err| {a1['end_abs_med']:.2f} s, "
         f"wrong s/clip {a1['wrong_s_per_clip']:.2f}, stale s/clip {a1['stale_s_per_clip']:.2f}", "", "Bars that differ from 0.5, per fold:"] + \
        [f"- fold {i + 1}: {c}" for i, c in enumerate(chosen)]
    (HERE / "family_hold.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
