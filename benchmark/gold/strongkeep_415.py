"""Round 52 STRONG-KEEP step 1 (docs/prereg_round13_detector_push.md "Round 52 STRONG-KEEP"), CPU, held-out 415 AudioSet-Strong
clips: precision of the spans the two FlexSED-based vetoes would remove although BEATs is very sure of them.

Raw BEATs spans as shipped MD3: _extract_events(threshold 0.35 display bar, min span 0.3 s, low 0.175 = AED_THRESHOLD, hysteresis
1.0), one span per BEATs label (as the 754/283 reference counts), family = canonical(label), depictable families only
(expect_screen.FAMILIES). Kept: span max (e.confidence, the value STRONG_BEATS_KEEP compares) >= 0.7 AND EITHER
  (i)  cross-detector veto: FlexSED's clip-wide max over the family's queries < 0.3 (FLEXSED_VETO; peak.get(key, 1.0) -> a family
       FlexSED has no query for is never vetoed, so it is not in the set)
  (ii) mirror veto: over FlexSED frames ft in [start, end) (nearest frame to the midpoint if none), the top query is another
       canonical family >= 0.7 while the own family's max < 0.4 (MIRROR_VETO 0.7, MIRROR_OWN_MAX 0.4; a family with no
       FlexSED query is never touched -> also requires a column)
Correctness = Round 42 (heldout_a4_screen.classify: a same_family strong event starts in [onset - 0.5, onset + 1.0]).
Bar: precision >= 0.375 (raw BEATs on the 415, heldout_a4_screen.json, recomputed here as a loader check).

    python benchmark/gold/strongkeep_415.py      # from ~/MscProj_tg -> benchmark/gold/strongkeep_415.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")

from benchmark.gold import heldout_a4_screen as H  # noqa: E402

SK, VETO, MIRROR, OWN_MAX = 0.7, 0.3, 0.7, 0.4
BAR = 0.375
OUT = _ROOT / "benchmark" / "gold" / "strongkeep_415.json"


def main():
    import config
    from benchmark.gold.score_per_sound import same_family
    from benchmark.gold import expect_screen as E
    from benchmark.gold import prior415_screen as P
    from benchmark.gold import dev_candidates_check as DCC
    from src.labels import canonical
    from src.stage4_audio_event_detection import _extract_events
    config.use_shipped()
    flags = {k: getattr(config, k, None) for k in ("AED_THRESHOLD", "AED_HYSTERESIS", "AED_MIN_DUR", "AED_RELEASE",
                                                   "DISPLAY_THRESHOLD", "FLEXSED_VETO", "MIRROR_VETO", "MIRROR_OWN_MAX")}
    assert flags["AED_THRESHOLD"] == 0.175 and flags["AED_HYSTERESIS"] == 1.0 and flags["AED_MIN_DUR"] == 0.3, flags
    assert flags["AED_RELEASE"] is None and flags["DISPLAY_THRESHOLD"] == 0.35, flags
    dep = set(E.FAMILIES)
    Hj = json.loads(H.HELDOUT.read_text(encoding="utf-8"))
    ev_of = {c["id"]: c["events"] for c in Hj["clips"]}
    cl = H.ids()
    miss = {"flexsed": [c for c in cl if not (H.FLEX_HELD / f"{c}.npz").exists()],
            "beats": [c for c in cl if not (H.BEATS_HELD / f"{c}.npz").exists()]}
    assert not any(miss.values()), {k: v[:3] for k, v in miss.items() if v}
    ref, strong, rows = [], [], []
    for c in cl:
        bfw, bt, blabs = DCC.load_fr(H.BEATS_HELD / f"{c}.npz")
        ffw, ft, flabs = DCC.load_fr(H.FLEX_HELD / f"{c}.npz")
        ft = np.asarray(ft)
        ffams = [canonical(l) for l in flabs]
        peak = {}
        for i, f in enumerate(ffams):
            peak[f] = max(peak.get(f, 0.0), float(ffw[:, i].max()))
        for lab, a, _b in P.beats_runs(bfw, bt, blabs):          # reference: min 0.5, display bar (Round 42: 754 / 283)
            f = canonical(lab)
            if f in dep:
                ref.append((f, H.classify(f, a, ev_of[c], same_family)))
        for e in _extract_events(bfw, bt, blabs, 0.35, None, 0.3, low=0.175):
            f = canonical(e.label)
            if f not in dep or float(e.confidence) < SK:
                continue
            a, b = float(e.start), float(e.end)
            k = H.classify(f, a, ev_of[c], same_family)
            strong.append((f, k))
            own = [i for i, g in enumerate(ffams) if g == f]
            v1 = bool(own) and peak[f] < VETO
            v2, top_f, top_v, own_v = False, None, None, None
            if own:
                m = (ft >= a) & (ft < b)
                if not m.any():
                    m = np.zeros(len(ft), bool); m[int(np.argmin(np.abs(ft - 0.5 * (a + b))))] = True
                pk = ffw[m].max(axis=0)
                top = int(np.argmax(pk))
                top_f, top_v, own_v = ffams[top], float(pk[top]), float(pk[own].max())
                v2 = top_f != f and top_v >= MIRROR and own_v < OWN_MAX
            if v1 or v2:
                rows.append({"clip": c, "label": e.label, "family": f, "onset": round(a, 2), "end": round(b, 2),
                             "conf": round(float(e.confidence), 3), "flex_clip_max": round(peak[f], 3),
                             "mirror_top": top_f, "mirror_top_v": None if top_v is None else round(top_v, 3),
                             "mirror_own_v": None if own_v is None else round(own_v, 3),
                             "i_cross": v1, "ii_mirror": v2, "class": k})

    def S(rs):
        return H.summ([(r["family"], r["class"]) for r in rs])

    kept = S(rows)
    res = {"what": __doc__.strip().splitlines()[0], "clips": len(cl), "flags": flags,
           "rule": "correct iff a same_family strong event starts in [onset-0.5, onset+1.0] (heldout_a4_screen.classify)",
           "params": {"strong_keep": SK, "flexsed_veto": VETO, "mirror_bar": MIRROR, "mirror_own_max": OWN_MAX,
                      "beats": "threshold 0.35, low 0.175, min span 0.3 s, one span per BEATs label",
                      "column_rule": "both (i) and (ii) need a FlexSED query of the family (as the two vetoes in stage 4)"},
           "bar": BAR,
           "set_i_or_ii": kept,
           "i_only": S([r for r in rows if r["i_cross"] and not r["ii_mirror"]]),
           "ii_only": S([r for r in rows if r["ii_mirror"] and not r["i_cross"]]),
           "both": S([r for r in rows if r["i_cross"] and r["ii_mirror"]]),
           "set_per_family": H.per_family([(r["family"], r["class"]) for r in rows]),
           "raw_beats_ge0.7_min0.3": H.summ(strong),
           "raw_beats_ge0.7_per_family": {f: x for f, x in H.per_family(strong).items() if x["n"] >= H.MIN_DET},
           "reference_raw_beats_min0.5": H.summ(ref),
           "passes": kept["precision"] is not None and kept["precision"] >= BAR,
           "rows": rows}
    for k in ("set_i_or_ii", "i_only", "ii_only", "both", "raw_beats_ge0.7_min0.3", "reference_raw_beats_min0.5"):
        print(k, res[k], flush=True)
    print("passes", res["passes"])
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
