"""Exploratory (slice B, not the test clips): simple combinations of BEATs (2-s windows) and
PretrainedSED (40-ms frames) from the cached scores. Each detector's scores are first put
on the pipeline scale (its own bar -> 0.35); the two are then combined on a common 0.25-s
grid: OR (either fires), MAX of scores, MEAN of scores, AND (both fire).
"""
import sys, json, numpy as np
sys.path.insert(0, ".")
from pathlib import Path
from benchmark import audioset_detector_eval as E
from src.stage4_audio_event_detection.psed_infer import rescale as psed_rescale, chosen_bar
from src.labels import canonical

OUT = E.WIN / "fusion"
GRID = 0.25


def common(labels_a, labels_b):
    """the union of the two label sets, matched by exact name"""
    return sorted(set(labels_a) | set(labels_b))


def on_grid(fw, times, labels, T, all_labels):
    """max score per grid cell per label; missing labels -> 0"""
    idx = {l: i for i, l in enumerate(labels)}
    g = np.zeros((len(T), len(all_labels)), np.float32)
    cell = np.minimum((times / GRID).astype(int), len(T) - 1)
    for j, l in enumerate(all_labels):
        if l in idx:
            np.maximum.at(g[:, j], cell, fw[:, idx[l]])
    return g


def build(mode):
    out = OUT / mode; out.mkdir(parents=True, exist_ok=True)
    for c in E.clips():
        pb, pp = E.WIN / "beats" / f"{c['id']}.npz", E.WIN / "psed" / f"{c['id']}.npz"
        if not (pb.exists() and pp.exists()):
            continue
        fb, tb, lb = E._load(pb); fp, tp, lp = E._load(pp)
        fp = psed_rescale(fp)
        n = int(np.ceil(max(tb.max(), tp.max()) / GRID)) + 1
        T = np.arange(n) * GRID
        L = common(lb, lp)
        gb, gp = on_grid(fb, tb, lb, T, L), on_grid(fp, tp, lp, T, L)
        if mode == "max":   g = np.maximum(gb, gp)
        elif mode == "mean": g = (gb + gp) / 2
        elif mode == "or":   g = np.where((gb >= 0.35) | (gp >= 0.35), np.maximum(gb, gp), np.minimum(gb, gp))
        elif mode == "and":  g = np.where((gb >= 0.35) & (gp >= 0.35), np.maximum(gb, gp), np.minimum(gb, gp) * 0.5)
        np.savez_compressed(out / f"{c['id']}.npz", fw=g.astype(np.float16), times=T.astype(np.float32), labels=np.array(L))


if __name__ == "__main__":
    print(f"{'detector':10s} {'masked-conseq':>13} {'conseq':>7} {'all':>6} {'FP/min':>7} {'onsetMAE':>9}")
    for name, bar in (("beats", 0.35), ("psed", chosen_bar())):
        r = E.evaluate_one(name, bar)
        print(f"{name:10s} {r['masked_conseq_recall']:13.1%} {r['conseq_recall']:7.1%} {r['all_recall']:6.1%} {r['fp_per_min']:7.2f} {r['onset_mae']:9.2f}")
    for mode in ("or", "max", "mean", "and"):
        build(mode)
        r = E.evaluate_one(f"fusion/{mode}", 0.35)
        print(f"{mode:10s} {r['masked_conseq_recall']:13.1%} {r['conseq_recall']:7.1%} {r['all_recall']:6.1%} {r['fp_per_min']:7.2f} {r['onset_mae']:9.2f}")
