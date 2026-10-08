"""Step 6 A (PREREG_step6_...): each ear's bar chosen ON DEV = the highest bar in 0.01 .. 0.30 at which it hears >= 34 of
the 58 needed sounds; reports burst AUROC (bar-free), needed heard and spans per clip at that bar.
    python benchmark/gold/coverage/ear_tier1_devbar.py EAR_DIR [EAR_DIR ...]   -> ear_tier1_devbar_dev.md"""
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.coverage import score_coverage as V
from benchmark.gold.coverage import keep_score as K
from benchmark.gold.coverage.omni_probe import auroc
from benchmark.gold.coverage.ear_tier1 import load, fam_curve, spans
from benchmark.gold.coverage.build_mixtures import drawable

HERE = Path(__file__).resolve().parent
NEED = 34


def main():
    gold = S.load_gold([V.GOLD])
    dev = V.stems("dev")
    rows = K.table()
    y = np.array([K.good(b, gold) for b, _ in rows])
    needed = [(st, g) for st in dev for g in gold[st] if g["needed"] and g["importance"] >= S.MIN_IMPORTANCE]
    L = ["# Step 6 A: ears with the bar chosen on DEV (highest bar hearing >= 34 / 58 needed sounds)", "",
         "| ear | DEV bar | needed heard | spans per clip (drawable families) | burst AUROC |", "|---|---|---|---|---|"]
    for d in sys.argv[1:]:
        d = Path(d)
        fr = {st: load(d / "dev" / f"{st}.npz") for st in dev}
        sc = []
        for b, _ in rows:
            c, t = fam_curve(fr[b["clip"]], b["family"])
            m = (t >= b["start"]) & (t <= b["end"])
            sc.append(float(c[m].max()) if m.any() else float(c[int(np.argmin(np.abs(t - b["start"])))]))
        curves = {}
        for st, g in needed:
            curves[(st, g["label"])] = fam_curve(fr[st], g["label"])
        pick = None
        for bar in np.round(np.arange(0.30, 0.0099, -0.01), 2):
            heard = sum(any(S.in_window(a, g["start"], S.EARLY, S.LATE) for a, _ in spans(*curves[(st, g["label"])], bar))
                        for st, g in needed)
            if heard >= NEED:
                pick = (bar, heard); break
        if pick is None:
            L.append(f"| {d.name} | none in 0.01-0.30 reaches {NEED} | - | - | {auroc(sc, y):.3f} |"); continue
        bar, heard = pick
        n = 0
        for st in dev:
            fw, t, labs = fr[st]
            for i, lab in enumerate(labs):
                if drawable(lab) and fw[:, i].max() >= bar:
                    n += len(spans(fw[:, i], t, bar))
        L.append(f"| {d.name} | {bar:.2f} | {heard} / {len(needed)} | {n / len(dev):.1f} | {auroc(sc, y):.3f} |")
    (HERE / "ear_tier1_devbar_dev.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
