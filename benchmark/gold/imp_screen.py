"""Round 36 IMP step 1 (docs/prereg_round13_detector_push.md "Round 36 IMP"), CPU only, candidate level: peaks of DASM's
impact-type labels (union column = max over the 5 exact-name columns, >= 0.575, runs merged over gaps <= LISTEN_RUN_GAP) on
merged DEV, each peak put in ONE bucket (first that applies) by the scorer's picture-start window [g.start - 0.5, g.start + 1.0]:
(a) SHIP8 ledger miss, (b) needed importance >= 2 sound SHIP8 already hits, (c) needed importance-1, (d) seen sound, (e) nothing.
Trigger for step 2: distinct (a) misses >= 3 and (b)+(d)+(e) peaks <= 3 x that. Nothing in src/ or config.py is edited.

    python benchmark/gold/imp_screen.py            # DASM caches under cross_group.PARTS roots (data/work/...)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import score_per_sound as S
from src.labels import canonical
from src.stage4_audio_event_detection import _runs, LISTEN_RUN_GAP

BAR = 0.575
WORK = _ROOT / "data" / "work"
PARTS = {"dev": WORK / "devcand" / "dasm_cache", "dev2": WORK / "dasm_dev2"}      # = cross_group.PARTS[..]["dasm"]
IMPACT = ["Specific impact sounds", "Thump, thud", "Knock", "Tap", "Hammer"]       # exact names present in DASM's list
ASKED_ABSENT = ["Bang", "Slam", "Smash, crash", "Whack, thwack", "Clang"]
LEDGER = _ROOT / "benchmark" / "gold" / "ledger_ship8.json"
OUT = _ROOT / "benchmark" / "gold" / "imp_screen.json"


def stems():
    gold, dev = DCC.dev_stems()
    dev2 = [l.strip() for l in (_ROOT / "benchmark" / "gold" / "dev2_stems.txt").read_text().split() if l.strip()]
    return gold, [("dev", s) for s in dev] + [("dev2", s) for s in dev2]


def peaks(fr):
    fw, ft, labs = fr
    cols = {l: labs.index(l) for l in IMPACT}
    assert len(cols) == len(IMPACT), [l for l in IMPACT if l not in labs]
    union = fw[:, list(cols.values())].max(axis=1).astype(np.float64)
    rr, dt = _runs(union, ft, BAR, LISTEN_RUN_GAP)
    out = []
    for i, j in rr:
        k = i + int(np.argmax(union[i:j]))
        firing = {l: round(float(fw[i:j, c].max()), 3) for l, c in cols.items() if fw[i:j, c].max() >= BAR}
        out.append({"t": round(float(ft[k]), 2), "run": [round(float(ft[i]), 2), round(float(ft[j - 1] + dt), 2)],
                    "peak": round(float(union[k]), 3), "labels": firing})
    return out


def main():
    gold, ST = stems()
    led = json.load(open(LEDGER, encoding="utf-8"))
    assert led["arm"] == "SHIP8"
    miss_keys = {(m["clip"], canonical(m["sound"]), round(float(m["at"]), 2)) for m in led["misses"]}
    rows, missing = [], []
    for pt, st in ST:
        p = PARTS[pt] / f"{st}.npz"
        if not p.exists():
            missing.append(st); continue
        g = gold[st]
        for pk in peaks(DCC.load_fr(p)):
            t = pk["t"]
            near = [x for x in g if x["start"] - S.EARLY <= t <= x["start"] + S.LATE]
            key = lambda x: (st, canonical(x["label"]), round(float(x["start"]), 2))
            b, who = "e_nothing", None
            for cand, tag in ((lambda x: x["needed"] and key(x) in miss_keys, "a_miss"),
                              (lambda x: x["needed"] and x["importance"] >= 2, "b_needed_hit"),
                              (lambda x: x["needed"], "c_needed_imp1"),
                              (lambda x: not x["needed"], "d_seen")):
                m = [x for x in near if cand(x)]
                if m:
                    b, who = tag, [(x["label"], x["start"], x["importance"]) for x in m]; break
            rows.append({"part": pt, "clip": st, **pk, "bucket": b, "gold": who})
    assert not missing, missing
    n = {b: sum(r["bucket"] == b for r in rows) for b in ["a_miss", "b_needed_hit", "c_needed_imp1", "d_seen", "e_nothing"]}
    distinct = sorted({(r["clip"], w[0], w[1]) for r in rows if r["bucket"] == "a_miss" for w in r["gold"]})
    elsewhere = n["b_needed_hit"] + n["d_seen"] + n["e_nothing"]
    trig = len(distinct) >= 3 and elsewhere <= 3 * len(distinct)
    perlab = {l: sum(l in r["labels"] for r in rows) for l in IMPACT}
    print(f"{len(ST)} clips, {len(rows)} peaks >= {BAR}; per label (runs where it fires): {perlab}")
    print(f"buckets: {n}; distinct SHIP8 misses with a peak: {len(distinct)}; elsewhere (b+d+e) {elsewhere} "
          f"-> {'TRIGGER step 2' if trig else 'STOP'}")
    for r in rows:
        if r["bucket"] == "a_miss":
            print(f"  MISS {r['clip']} t={r['t']} peak {r['peak']} {r['labels']} -> gold {r['gold']} "
                  f"(family {[canonical(w[0]) for w in r['gold']]})")
    ledger_impact = [m for m in led["misses"] if canonical(m["sound"]) in
                     {"Whack, thwack", "Clang", "Hammer", "Explosion", "Dishes", "Knock", "Tap", "Thump, thud"}]
    print(f"ledger misses that look impact-like: {len(ledger_impact)}; of them with a peak: "
          f"{sum((m['clip'], m['sound'], m['at']) in {(c, s, a) for c, s, a in distinct} for m in ledger_impact)}")
    for m in ledger_impact:
        print(f"    {m['clip']} {m['sound']} {m['at']} -> {'PEAK' if (m['clip'], m['sound'], m['at']) in set(distinct) else 'no peak'}")
    OUT.write_text(json.dumps({"bar": BAR, "gap": LISTEN_RUN_GAP, "impact_labels": IMPACT, "asked_absent": ASKED_ABSENT,
                               "n_clips": len(ST), "buckets": n, "per_label": perlab, "distinct_misses": distinct,
                               "elsewhere": elsewhere, "trigger": trig, "peaks": rows}, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
