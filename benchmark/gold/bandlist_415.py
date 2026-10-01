"""bandlist_415 (CPU diagnostic, held-out 415 AudioSet-Strong clips): precision of FlexSED runs in the band 0.5 <= max < 0.8
of a depictable family, and of the subsets whose family a whole-clip list names.

Reuses Round 42 HELDOUT-A4 exactly (benchmark/gold/heldout_a4_screen.py): its clip list, FlexSED cache, DASM cache (Round 6),
its correctness rule `classify` (correct iff a same_family strong event starts in [onset - 0.5, onset + 1.0]), the depictable
families (expect_screen.FAMILIES) and canonical(label) as the run family. Runs are formed as PRIOR415 flex_runs forms them
(_extract_events under the shipped flags), but with threshold = low = 0.5, then kept only if the run max < 0.8.

  (a) band runs, minimum length 0.5 s (as PRIOR415) and with no minimum
  (b) (a) whose family the Qwen3-Omni whole-clip list names (heldout_a4/listen, expect_a_screen.items_of / map_item, all items)
  (c) (a) whose family the Audio Flamingo Next whole-clip list names (agree_ears/heldout, same map)
  (d) (b) AND DASM >= 0.575 for that family within run onset +- 0.5 s (expect_a4_screen.dasm_max)
  reference: raw FlexSED >= 0.8 runs (PRIOR415 flex_runs), same rule (Round 42 reported 117/297 = 0.394)

    TG_ARMS=SHIP8 python benchmark/gold/bandlist_415.py      # -> benchmark/gold/bandlist_415.json
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

LO, HI = 0.5, 0.8
AFN_DIR = _ROOT / "benchmark" / "gold" / "agree_ears" / "heldout"
OUT = _ROOT / "benchmark" / "gold" / "bandlist_415.json"


def fams_of(text, A):
    out = []
    for it in A.items_of(text):
        f = A.map_item(it)
        if f and f not in out:
            out.append(f)
    return out


def main():
    from benchmark.gold.score_per_sound import same_family
    from benchmark.gold import expect_screen as E
    from benchmark.gold import expect_a_screen as A
    from benchmark.gold import expect_a4_screen as A4
    from benchmark.gold import prior415_screen as P
    from benchmark.gold import dev_candidates_check as DCC
    from src.labels import canonical
    from src.stage4_audio_event_detection import _extract_events
    flags = P.shipped_flags()
    dep = set(E.FAMILIES)
    Hj = json.loads(H.HELDOUT.read_text(encoding="utf-8"))
    ev_of = {c["id"]: c["events"] for c in Hj["clips"]}
    cl = H.ids()
    afn_missing = [c for c in cl if not (AFN_DIR / f"{c}.json").exists()]
    rows, ref = [], []
    afn_map_mismatch = 0
    for c in cl:
        omni = fams_of(json.loads((H.DIR / "listen" / f"{c}.json").read_text(encoding="utf-8"))["text"], A)
        afn = None
        if c not in afn_missing:
            R = json.loads((AFN_DIR / f"{c}.json").read_text(encoding="utf-8"))
            afn = fams_of(R["text"], A)
            afn_map_mismatch += afn != R["families"]
        fw, t, labs = DCC.load_fr(H.FLEX_HELD / f"{c}.npz")
        dfr = None
        for lab, a, b in P.flex_runs(fw, t, labs):
            f = canonical(lab)
            if f in dep:
                ref.append({"clip": c, "family": f, "onset": a, "end": b, "class": H.classify(f, a, ev_of[c], same_family),
                            "omni": f in omni, "afn": None if afn is None else f in afn})
        for e in _extract_events(fw, t, labs, LO, None, 0.0, low=LO):
            f = canonical(e.label)
            if f not in dep or e.confidence >= HI:
                continue
            a, b = float(e.start), float(e.end)
            if dfr is None:
                dfr = DCC.load_fr(H.DASM_HELD / f"{c}.npz")
            v = A4.dasm_max(dfr, f, a - H.HALF, a + H.HALF)
            rows.append({"clip": c, "family": f, "onset": round(a, 2), "end": round(b, 2), "len": round(b - a, 2),
                         "max": round(float(e.confidence), 3), "omni": f in omni, "afn": None if afn is None else f in afn,
                         "dasm_onset": None if v is None else round(v, 3),
                         "class": H.classify(f, a, ev_of[c], same_family)})

    def S(rs):
        return H.summ([(r["family"], r["class"]) for r in rs])

    res = {"what": __doc__.strip().splitlines()[0], "clips": len(cl), "flags": flags,
           "rule": "correct iff a same_family strong event starts in [onset-0.5, onset+1.0] (heldout_a4_screen.classify)",
           "band": [LO, HI], "dasm_bar": H.DASM_BAR, "dasm_half": H.HALF,
           "afn_lists": {"dir": str(AFN_DIR.relative_to(_ROOT)), "missing": len(afn_missing), "map_mismatch_vs_cache": afn_map_mismatch}}
    for name, mind in (("min0.5", 0.5), ("nomin", 0.0)):
        R = [r for r in rows if r["len"] >= mind - 1e-9]
        b = [r for r in R if r["omni"]]
        res[name] = {"a_band": S(R), "b_omni": S(b), "c_afn": S([r for r in R if r["afn"]]),
                     "d_omni_dasm": S([r for r in b if r["dasm_onset"] is not None and r["dasm_onset"] >= H.DASM_BAR]),
                     "omni_and_afn": S([r for r in b if r["afn"]]),
                     "a_per_family": H.per_family([(r["family"], r["class"]) for r in R]),
                     "b_per_family": H.per_family([(r["family"], r["class"]) for r in b])}
        print(name, {k: v for k, v in res[name].items() if "per_family" not in k}, flush=True)
    res["reference_flexsed_ge0.8"] = {"all": S(ref), "omni": S([r for r in ref if r["omni"]]), "afn": S([r for r in ref if r["afn"]])}
    print("reference", res["reference_flexsed_ge0.8"])
    res["band_rows"] = rows
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
