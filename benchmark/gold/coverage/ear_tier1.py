"""Step 5 tier 1 (PREREG_step5_detector_finetune.md): does a new ear hear the DEV bursts and needed sounds better?

Bar of each frame ear (no DEV): on the 415 held-out AudioSet-Strong clips, the bar in 0.05 .. 0.60 (step 0.05) with the
best frame-level micro F1 over drawable classes (40-ms frames; truth = inside a labelled event of the class). The
pipeline's BEATs ear keeps its shipped bar 0.175.

    python benchmark/gold/coverage/ear_tier1.py EAR_DIR [EAR_DIR ...]   (each EAR_DIR has dev/ and heldout/ npz)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.coverage import score_coverage as V
from benchmark.gold.coverage import keep_score as K
from benchmark.gold.coverage.omni_probe import auroc
from benchmark.gold.coverage.build_mixtures import drawable

HERE = Path(__file__).resolve().parent


def load(p):
    z = np.load(p)
    return z["fw"].astype(np.float32), z["times"].astype(np.float64), [str(x) for x in z["labels"]]


def fam_curve(fr, fam):
    fw, t, labs = fr
    cols = [i for i, c in enumerate(labs) if S.same_family(c, fam)]
    return (fw[:, cols].max(axis=1), t) if cols else (np.zeros(len(t)), t)


def calibrate(heldout_dir, heldout_json):
    clips = {c["id"]: c for c in json.loads(Path(heldout_json).read_text(encoding="utf-8"))["clips"]}
    bars = np.arange(0.05, 0.601, 0.05)
    tp, fp, fn = np.zeros(len(bars)), np.zeros(len(bars)), np.zeros(len(bars))
    for f in sorted(Path(heldout_dir).glob("*.npz")):
        c = clips.get(f.stem)
        if c is None:
            continue
        fw, t, labs = load(f)
        keep = [i for i, l in enumerate(labs) if drawable(l)]
        truth = np.zeros((len(t), len(labs)), bool)
        for e in c["events"]:
            for i in keep:
                if S.same_family(labs[i], e["label"]) or labs[i] == e["label"]:
                    truth[(t >= e["start"]) & (t <= e["end"]), i] = True
        for k, b in enumerate(bars):
            pr = fw[:, keep] >= b
            tr = truth[:, keep]
            tp[k] += (pr & tr).sum(); fp[k] += (pr & ~tr).sum(); fn[k] += (~pr & tr).sum()
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return float(bars[int(np.argmax(f1))]), dict(zip([round(b, 2) for b in bars], [round(x, 4) for x in f1]))


def spans(curve, t, bar, min_dur=0.3):
    on = curve >= bar
    out, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if t[min(j, len(t) - 1)] - t[i] >= min_dur or j == len(on):
                out.append((t[i], t[min(j, len(t) - 1)]))
            i = j
        else:
            i += 1
    return out


def main():
    gold = S.load_gold([V.GOLD])
    dev = V.stems("dev")
    rows = K.table()
    y = np.array([K.good(b, gold) for b, _ in rows])
    needed = [(st, g) for st in dev for g in gold[st] if g["needed"] and g["importance"] >= S.MIN_IMPORTANCE]
    L = ["# Step 5 tier 1: ears on DEV", "", "| ear | bar | burst AUROC (good vs not) | needed sounds with a span starting in the onset window |",
         "|---|---|---|---|"]
    i_b = K.FEATS.index("beats_max")
    L.append(f"| pipeline BEATs (reference) | 0.175 | {auroc([-1 if np.isnan(r[i_b]) else r[i_b] for _, r in rows], y):.3f} | see note |")
    res = {}
    for d in sys.argv[1:]:
        d = Path(d)
        bar, f1s = calibrate(d / "heldout", os.environ.get("HELDOUT_JSON", _ROOT / "benchmark" / "gold" / "audioset_heldout.json")) if (d / "heldout").exists() else (0.15, {})
        fr = {st: load(d / "dev" / f"{st}.npz") for st in dev}
        sc = []
        for b, _ in rows:
            c, t = fam_curve(fr[b["clip"]], b["family"])
            m = (t >= b["start"]) & (t <= b["end"])
            sc.append(float(c[m].max()) if m.any() else float(c[int(np.argmin(np.abs(t - b["start"])))]))
        heard = 0
        for st, g in needed:
            c, t = fam_curve(fr[st], g["label"])
            heard += any(S.in_window(a, g["start"], S.EARLY, S.LATE) for a, _ in spans(c, t, bar))
        a = auroc(sc, y)
        res[d.name] = {"bar": bar, "f1_by_bar": f1s, "auroc": a, "needed_heard": heard, "needed": len(needed)}
        L.append(f"| {d.name} | {bar:.2f} | {a:.3f} | {heard} / {len(needed)} |")
    L += ["", "Reference AUROCs on the same bursts (Step 4 features): BEATs 0.710, FlexSED 0.809, DASM 0.885."]
    (HERE / "ear_tier1_dev.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (HERE / "ear_tier1_dev.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
