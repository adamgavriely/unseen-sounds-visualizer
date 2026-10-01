"""Round 56 TWIN-SHORT step 1 (docs/prereg_round13_detector_push.md "Round 56 TWIN-SHORT"), CPU, held-out 415 AudioSet-Strong
clips: precision of the twin-short unions (case 1) and short FlexSED twins (case 2) that src twin_short() admits.

Shipped flags (config.use_shipped: AED_THRESHOLD 0.175, hysteresis 1.0, AED_MIN_DUR 0.3, FLEXSED_BAR 0.8, TWIN_MAX, display 0.35).
Correctness = Round 42 (heldout_a4_screen.classify). Depictable families only. Guard: G = case-1 unions at partner bar 0.5 with
confidence >= 0.35; STOP iff n >= 3 and precision < 0.375.

    python benchmark/gold/twinshort_415.py      # from ~/MscProj_tg -> benchmark/gold/twinshort_415.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")

from benchmark.gold import heldout_a4_screen as H  # noqa: E402

BAND, LITERAL, BAR, DV = 0.5, 0.8, 0.375, 0.08392333984375
OUT = _ROOT / "benchmark" / "gold" / "twinshort_415.json"


def main():
    import config
    from benchmark.gold.score_per_sound import same_family
    from benchmark.gold import expect_screen as E
    from benchmark.gold import prior415_screen as P
    from benchmark.gold import expect_a4_screen as A4
    from benchmark.gold import dev_candidates_check as DCC
    from src.labels import canonical
    from src.stage4_audio_event_detection import twin_short
    config.use_shipped()
    flags = {k: getattr(config, k, None) for k in ("AED_THRESHOLD", "AED_HYSTERESIS", "AED_MIN_DUR", "FLEXSED_BAR",
                                                   "FLEXSED_FAMILY_BARS", "IMPULSE_MIN_SPAN", "TWIN_MAX", "DISPLAY_THRESHOLD",
                                                   "AED_RELEASE", "DASM_CLIP_VETO", "TWIN_SHORT")}
    assert flags["AED_THRESHOLD"] == 0.175 and flags["AED_HYSTERESIS"] == 1.0 and flags["AED_MIN_DUR"] == 0.3, flags
    assert flags["FLEXSED_BAR"] == 0.8 and flags["FLEXSED_FAMILY_BARS"] is None and flags["IMPULSE_MIN_SPAN"] is None, flags
    assert flags["TWIN_MAX"] and flags["DISPLAY_THRESHOLD"] == 0.35 and flags["AED_RELEASE"] is None, flags
    assert abs(flags["DASM_CLIP_VETO"] - DV) < 1e-12 and flags["TWIN_SHORT"] is None, flags
    md, fbar, disp = 0.3, 0.8, 0.35
    dep = set(E.FAMILIES)
    Hj = json.loads(H.HELDOUT.read_text(encoding="utf-8"))
    ev_of = {c["id"]: c["events"] for c in Hj["clips"]}
    cl = H.ids()
    miss = {k: [c for c in cl if not (d / f"{c}.npz").exists()]
            for k, d in (("flexsed", H.FLEX_HELD), ("beats", H.BEATS_HELD), ("dasm", H.DASM_HELD))}
    assert not any(miss.values()), {k: v[:3] for k, v in miss.items() if v}
    ref, rows = [], []
    for c in cl:
        bfw, bt, blabs = DCC.load_fr(H.BEATS_HELD / f"{c}.npz")
        ffw, ft, flabs = DCC.load_fr(H.FLEX_HELD / f"{c}.npz")
        dfr = DCC.load_fr(H.DASM_HELD / f"{c}.npz")
        for lab, a, _b in P.beats_runs(bfw, bt, blabs):
            f = canonical(lab)
            if f in dep:
                ref.append((f, H.classify(f, a, ev_of[c], same_family)))
        for band in (BAND, LITERAL):
            c1, c2 = twin_short(bfw, bt, blabs, ffw, ft, flabs, md, fbar, band, twin_max=True, disp=disp)
            for case, evs in (("case1", c1), ("case2", c2 if band == BAND else [])):
                for e in evs:
                    f = canonical(e.label)
                    if f not in dep:
                        continue
                    dclip = A4.dasm_max(dfr, f, -1e9, 1e9)
                    rows.append({"clip": c, "band": band, "case": case, "sub": getattr(e, "ts_case", "shortF"),
                                 "label": e.label, "family": f, "onset": round(float(e.start), 2), "end": round(float(e.end), 2),
                                 "conf": round(float(e.confidence), 3), "beats": getattr(e, "ts_beats", None),
                                 "partner_peak": getattr(e, "ts_partner_peak", None),
                                 "dasm_clip": None if dclip is None else round(dclip, 4),
                                 "class": H.classify(f, float(e.start), ev_of[c], same_family)})

    def S(rs):
        return H.summ([(r["family"], r["class"]) for r in rs])

    c1 = [r for r in rows if r["case"] == "case1" and r["band"] == BAND]
    G = [r for r in c1 if r["conf"] >= disp]
    lit = [r for r in rows if r["case"] == "case1" and r["band"] == LITERAL and r["conf"] >= disp]
    g = S(G)
    stop = g["n"] >= H.MIN_DET and g["precision"] < BAR
    res = {"what": __doc__.strip().splitlines()[0], "clips": len(cl), "flags": flags,
           "rule": "correct iff a same_family strong event starts in [onset-0.5, onset+1.0] (heldout_a4_screen.classify)",
           "guard": {"set": "case-1 unions, partner band 0.5, conf >= 0.35", "bar": BAR, "min_n": H.MIN_DET},
           "G": g, "STOP": bool(stop), "judged": g["n"] >= H.MIN_DET,
           "G_short_short": S([r for r in G if r["sub"] == "short+short"]),
           "G_shortB_longF": S([r for r in G if r["sub"] == "shortB+longF"]),
           "G_dasm_clip_below_DV": S([r for r in G if r["dasm_clip"] is not None and r["dasm_clip"] < DV]),
           "case1_all_incl_subdisplay": S(c1),
           "literal_partner_0.8_G": S(lit),
           "case2_shortF": S([r for r in rows if r["case"] == "case2"]),
           "G_per_family": H.per_family([(r["family"], r["class"]) for r in G]),
           "loader_check_raw_beats_min0.5_display": H.summ(ref),
           "rows": rows}
    for k in ("G", "STOP", "G_short_short", "G_shortB_longF", "G_dasm_clip_below_DV", "case1_all_incl_subdisplay",
              "literal_partner_0.8_G", "case2_shortF", "loader_check_raw_beats_min0.5_display"):
        print(k, res[k], flush=True)
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
