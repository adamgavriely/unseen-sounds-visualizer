"""Diagnostic (DEV only, no selection): every needed DEV sound (importance >= 2), whether the final system and the blind arm
show it, the likely reason for a miss, and each detector's best same-family score around its onset.

Reasons, in this order: hit (ours) / removed by the gate (blind shows it in the window, ours does not) / timing (a
same-family picture overlaps the sound but does not start in [onset - 0.5, onset + 1.0] s) / detected, not drawn (the
detector's events have the family near the onset but no picture was shown: label filter, display) / never detected.
The hit test here is the onset window with family match, without the scorer's one-to-one bookkeeping (a diagnostic).

    python benchmark/gold/dev_miss_table.py        # cluster (needs the DEV renders) -> docs/dev_miss_table_2026-09-27.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.audioset_stage4_report import load
from src.labels import canonical

TAG = "dev_monocap_v31"
WORK = _ROOT / "data" / "work"
CACHES = {"BEATs": _ROOT / "benchmark" / "gold" / "beats_fw", "FlexSED": _ROOT / "data" / "work" / "flexsed_cache",
          "PANNs": _ROOT / "benchmark" / "gold" / "panns_fw", "PE-A-Frame": _ROOT / "data" / "work" / "pe_frame_cache"}
OUT = _ROOT / "docs" / "dev_miss_table_2026-09-27.md"


def best(cache, stem, fam, a, b, pe=False):
    p = cache / f"{stem}.npz"
    if not p.exists():
        return None
    fw, ts, labs = load(p)
    if pe:
        fw = fw - np.median(fw, axis=1, keepdims=True)
    m = (ts >= a) & (ts <= b)
    cols = [i for i, l in enumerate(labs) if canonical(l) == fam]
    if not cols or not m.any():
        return 0.0
    return float(fw[m][:, cols].max())


def main():
    gold = S.load_gold([_ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"])
    dev = sorted(S.subsets_of(gold)["dev"])
    rows = []
    for stem in dev:
        ours = S.load_pictures(WORK / f"protocol_proposed_{TAG}", stem, "proposed") or []
        blind = S.load_pictures(WORK / f"protocol_blind_a2i_{TAG}", stem, "blind_a2i") or []
        evf = WORK / f"protocol_blind_a2i_{TAG}" / stem / "events.json"
        events = json.loads(evf.read_text(encoding="utf-8")) if evf.exists() else []
        for g in gold[stem]:
            if not g["needed"] or g["importance"] < 2:
                continue
            fam, on = canonical(g["label"]), g["start"]
            inwin = lambda pics: any(S.same_family(l, g["label"]) and S.in_window(a, on, S.EARLY, S.LATE) for l, a, b in pics)
            if inwin(ours):
                why = "hit"
            elif inwin(blind):
                why = "removed by the gate"
            elif any(S.same_family(l, g["label"]) and b > g["start"] and a < g["end"] for l, a, b in blind):
                why = "timing"
            elif any(S.same_family(e["label"], g["label"]) and e["end"] >= on - 0.5 and e["start"] <= on + 1.0 for e in events):
                why = "detected, not drawn"
            else:
                why = "never detected"
            sc = {k: best(c, stem, fam, on - 0.5, on + 1.0, pe=(k == "PE-A-Frame")) for k, c in CACHES.items()}
            rows.append({"clip": stem, "sound": g["label"], "onset": on, "importance": g["importance"], "why": why, **sc})
    order = ["hit", "removed by the gate", "timing", "detected, not drawn", "never detected"]
    rows.sort(key=lambda r: (order.index(r["why"]), r["clip"], r["onset"]))
    counts = {k: sum(r["why"] == k for r in rows) for k in order}
    f = lambda x: "–" if x is None else f"{x:.2f}"
    lines = [f"# DEV-49: every needed sound (importance ≥ 2) and why it is missed — `{TAG}`, diagnostic, 2026-09-27", "",
             f"{len(rows)} needed sounds: " + ", ".join(f"{k} {v}" for k, v in counts.items()) + ".", "",
             "Scores = best same-family frame score in [onset − 0.5, onset + 1.0] s. Shipped bars: BEATs 0.35, FlexSED 0.8, "
             "PANNs veto 0.05. PE-A-Frame = logit minus the frame's median (amendment 24); not used by the shipped system.", "",
             "| reason | clip | sound | onset s | imp. | BEATs | FlexSED | PANNs | PE-A-Frame |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['why']} | {r['clip']} | {r['sound']} | {r['onset']:.1f} | {r['importance']} | {f(r['BEATs'])} | "
                     f"{f(r['FlexSED'])} | {f(r['PANNs'])} | {f(r['PE-A-Frame'])} |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(counts, "->", OUT)


if __name__ == "__main__":
    main()
