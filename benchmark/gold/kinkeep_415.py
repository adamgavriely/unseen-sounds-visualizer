"""Round 52b KIN-KEEP step 1 (docs/prereg_round13_detector_push.md "Round 52b KIN-KEEP"), CPU, held-out 415 AudioSet-Strong clips:
which ontology closeness level (if any) separates the Round 52 proxy spans the two FlexSED vetoes would remove into a kept set
that is precise and a rest that is not.

Proxy spans exactly as benchmark/gold/strongkeep_415.py (raw BEATs, threshold 0.35, low 0.175, min span 0.3 s, one span per BEATs
label, depictable families, conf >= 0.7, AND (i) cross-detector veto OR (ii) mirror veto would remove it). Added per span: the top
FlexSED query of a DIFFERENT canonical family over [start, end) (nearest frame to the midpoint if none), its label and score.
Kept by level L iff that score >= 0.7 (MIRROR_VETO) and kin_labels(span label, query label, L) (below; the step-1 run used the
same function placed in src.stage4_audio_event_detection, removed after the STOP).
Level qualifies iff precision(K) >= 0.375 with n(K) >= 10 AND precision(rest) < 0.375; pick the qualifying level with the higher
precision(K), tie -> "parent"; none -> STOP.

    python benchmark/gold/kinkeep_415.py      # from ~/MscProj_tg -> benchmark/gold/kinkeep_415.json
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
BAR, MIN_N = 0.375, 10
LEVELS = ("nonroot", "parent")
OUT = _ROOT / "benchmark" / "gold" / "kinkeep_415.json"


def kin_labels(a: str, b: str, level: str) -> bool:
    """are two labels ontology relatives? "nonroot": their deepest common AudioSet ancestor (itself included) exists and is
    not a top-level root; "parent": that ancestor is each label itself or its direct parent"""
    from src.labels import _common_parent, ancestors
    c = _common_parent(a, b)
    if not c or not ancestors(c):                        # none, or a top-level root (no parent)
        return False
    if level == "nonroot":
        return True
    if level == "parent":
        near = lambda x: c == x or (ancestors(x)[:1] == [c])
        return near(a) and near(b)
    raise ValueError(f"KIN_KEEP level {level!r}")


def main():
    import config
    from benchmark.gold.score_per_sound import same_family
    from benchmark.gold import expect_screen as E
    from benchmark.gold import dev_candidates_check as DCC
    from src.labels import canonical
    from src.stage4_audio_event_detection import _extract_events
    config.use_shipped()
    dep = set(E.FAMILIES)
    Hj = json.loads(H.HELDOUT.read_text(encoding="utf-8"))
    ev_of = {c["id"]: c["events"] for c in Hj["clips"]}
    rows = []
    for c in H.ids():
        bfw, bt, blabs = DCC.load_fr(H.BEATS_HELD / f"{c}.npz")
        ffw, ft, flabs = DCC.load_fr(H.FLEX_HELD / f"{c}.npz")
        ft = np.asarray(ft)
        ffams = [canonical(l) for l in flabs]
        peak = {}
        for i, f in enumerate(ffams):
            peak[f] = max(peak.get(f, 0.0), float(ffw[:, i].max()))
        for e in _extract_events(bfw, bt, blabs, 0.35, None, 0.3, low=0.175):
            f = canonical(e.label)
            if f not in dep or float(e.confidence) < SK:
                continue
            a, b = float(e.start), float(e.end)
            own = [i for i, g in enumerate(ffams) if g == f]
            if not own:
                continue                                    # both vetoes need a FlexSED query of the family
            m = (ft >= a) & (ft < b)
            if not m.any():
                m = np.zeros(len(ft), bool); m[int(np.argmin(np.abs(ft - 0.5 * (a + b))))] = True
            pk = ffw[m].max(axis=0)
            top = int(np.argmax(pk))
            v1 = peak[f] < VETO
            v2 = ffams[top] != f and float(pk[top]) >= MIRROR and float(pk[own].max()) < OWN_MAX
            if not (v1 or v2):
                continue
            other = [i for i, g in enumerate(ffams) if g != f]
            o = max(other, key=lambda i: float(pk[i]))
            r = {"clip": c, "label": e.label, "family": f, "onset": round(a, 2), "end": round(b, 2),
                 "conf": round(float(e.confidence), 3), "i_cross": v1, "ii_mirror": v2,
                 "comp_label": flabs[o], "comp_v": round(float(pk[o]), 3),
                 "class": H.classify(f, a, ev_of[c], same_family)}
            for L in LEVELS:
                r[f"kin_{L}"] = float(pk[o]) >= MIRROR and kin_labels(e.label, flabs[o], L)
            rows.append(r)

    def S(rs):
        return H.summ([(r["family"], r["class"]) for r in rs])

    lv = {}
    for L in LEVELS:
        K, Rr = S([r for r in rows if r[f"kin_{L}"]]), S([r for r in rows if not r[f"kin_{L}"]])
        ok = (K["precision"] is not None and K["n"] >= MIN_N and K["precision"] >= BAR
              and (Rr["precision"] is None or Rr["precision"] < BAR))
        lv[L] = {"kept": K, "rest": Rr, "qualifies": ok}
    q = [L for L in LEVELS if lv[L]["qualifies"]]
    pick = None
    if q:
        best = max(lv[L]["kept"]["precision"] for L in q)
        pick = "parent" if ("parent" in q and lv["parent"]["kept"]["precision"] == best) else \
            next(L for L in q if lv[L]["kept"]["precision"] == best)
    res = {"what": __doc__.strip().splitlines()[0], "clips": len(H.ids()), "proxy_set": S(rows),
           "levels": lv, "pick": pick, "verdict": "GO" if pick else "STOP",
           "kin_rows": [r for r in rows if r["kin_nonroot"] or r["kin_parent"]], "rows": rows}
    print("proxy", res["proxy_set"])
    for L in LEVELS:
        print(L, lv[L])
    print("pick", pick, res["verdict"])
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
