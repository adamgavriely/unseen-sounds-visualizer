"""The merged TEST (Adam, 2026-09-30; docs/prereg_round13_detector_push.md "Merge of the tagger set" and "Merged-DEV
selection"): old TEST (60 clips, gold_AG test_bench) + the tagger TEST part (benchmark/gold/test2_stems.txt, tagger_AG),
scored ONCE as one set for the merged-DEV pick vs the old shipped config B0 (= B0r, the scored render).

  dasm      DASM frame scores (round-6 scorer, as the DEV / tagger caches) for the old TEST clips -> data/work/dasm_test
  stage     old TEST: stage 4 / stage 5 / gates of B0r and the pick through round13_dev (r13_test_final.setup: TEST
            redirects, gold stubbed), outputs in data/work/r16final (r13final, the round-13 exposure, is left untouched)
  score     THE exposure (both parts; the tagger TEST part must have been run by tagger_prep stage4/stage5 first).
            Refuses if benchmark/gold/final_test.json or its .started marker exists. Rule (prereg, primary = proposed):
            better iff hits do not drop AND wrong rises by <= 2 x hits gained AND d cost < 0 with one-sided p < 0.05;
            worse iff d cost > 0 with lower bound > 0 or the hits/wrong rule fails; same otherwise.

    python benchmark/gold/final_test.py dasm
    python benchmark/gold/final_test.py stage --arm TO1+F7F8
    python benchmark/gold/final_test.py score --arm TO1+F7F8
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S

_REAL_LOAD_GOLD = S.load_gold
G = _ROOT / "benchmark" / "gold"
OUT, STARTED, MD = G / "final_test.json", G / "final_test.json.started", G / "final_test.md"


def set_tag(tag):
    """a later, separately recorded TEST read (Adam 30 Sept 07:50: every new DEV best is also reported on TEST)"""
    global OUT, STARTED, MD
    if tag:
        OUT, STARTED, MD = G / f"final_test_{tag}.json", G / f"final_test_{tag}.json.started", G / f"final_test_{tag}.md"
WORKD = _ROOT / "data" / "work"
DASM_TEST = WORKD / "dasm_test"
CMAP = [f"dev_listener_v.json={G / 'test_listener_v.json'}", f"dev_listener_afn.json={G / 'test_listener_afn.json'}",
        f"dev_listener_kimi.json={G / 'test_listener_kimi.json'}", f"dasm_cache={DASM_TEST}",
        f"dev_listener_p4.json={G / 'test_listener_p4.json'}", f"dev_listener_p1v4.json={G / 'test_listener_p1v4.json'}",
        f"dev_listener_v4b.json={G / 'test_listener_v4b.json'}"]


def guard():
    for p in (OUT, STARTED):
        if p.exists():
            raise SystemExit(f"ONE EXPOSURE: {p} exists; refusing")


_ARMS0 = {}


def old_setup(arm):
    from benchmark.gold import round13_dev as R0
    for k in ("B0r", "B1", arm):                              # every setup starts from the DEV originals
        if k in R0.ARMS:
            _ARMS0.setdefault(k, dict(R0.ARMS[k]))
            R0.ARMS[k] = dict(_ARMS0[k])
    from benchmark.gold import r13_test_final as F
    a = argparse.Namespace(arm=arm, flags=None, cache_map=CMAP, stage5_keys=None, dry_run=False)
    T, R, arms, fin = F.setup(a)
    fin = WORKD / "r16final"                                  # leave r13final (the round-13 exposure) as it is
    fin.mkdir(parents=True, exist_ok=True)
    if not (fin / "ask_memo.json").exists() and (T.OUT / "ask_memo.json").exists():
        shutil.copy(T.OUT / "ask_memo.json", fin / "ask_memo.json")
    R.R13, R.STAGE4, R.MEMO = fin, fin / "stage4.json", fin / "ask_memo.json"
    arms = ["B0r", arm]
    return F, T, R, arms, fin


def dasm():
    from benchmark.gold import r13_test_prep as T              # TEST redirects of dev_candidates_check (gold stubbed)
    from benchmark.gold import dev_candidates_check as DCC
    DCC.DASM_DIR = DASM_TEST
    DCC.dasm()


def stage(arm):
    F, T, R, arms, fin = old_setup(arm)
    R.stage4(arms)
    R.stage5(arms)
    g = F.gates(T, R, arms, fin)
    print("[stage] old TEST gates pass:", g["pass"], flush=True)


def score(arm):
    from benchmark.gold import dev_candidates_check as DCC
    guard()
    F, T, R, arms, fin = old_setup(arm)
    g = F.gates(T, R, arms, fin)
    if not g["pass"]:
        raise SystemExit(f"old TEST gates failed ({g}); gold NOT read")
    stems1 = list(T.STEMS)
    P1 = {"B0r": {st: S.load_pictures(fin / f"B0r_proposed", st, "proposed") or [] for st in stems1}}
    with R.flags({k: R.arm_cfg(arm)[k] for k in R.DISPLAY_KEYS}):
        P1[arm] = {st: S.load_pictures(fin / f"{arm}_proposed", st, "proposed") or [] for st in stems1}
    # tagger TEST part (tagger_prep's own outputs; its configure() re-points round13_dev, so pictures are read first)
    from benchmark.gold import tagger_prep as TP
    for k, v in _ARMS0.items():                               # tagger_prep maps from the DEV originals too
        R.ARMS[k] = dict(v)
    TP._ORIG.clear()
    _D2, R2, stems2 = TP.configure("test2")
    o2 = TP.out("test2")
    P2 = {"B0r": {st: S.load_pictures(o2 / "B0r_proposed", st, "proposed") or [] for st in stems2}}
    with R2.flags({k: R2.arm_cfg(arm)[k] for k in R2.DISPLAY_KEYS}):
        P2[arm] = {st: S.load_pictures(o2 / f"{arm}_proposed", st, "proposed") or [] for st in stems2}
    miss = [st for st in stems2 if not (o2 / f"{arm}_proposed" / st / "augmentations.json").exists()]
    if miss:
        raise SystemExit(f"tagger TEST part incomplete for {arm}: {miss}; gold NOT read")
    STARTED.write_text(json.dumps({"arm": arm, "time": datetime.now().isoformat()}), encoding="utf-8")
    S.load_gold = _REAL_LOAD_GOLD
    gold1 = S.load_gold([F.GOLD])
    assert sorted(S.subsets_of(gold1)["test_bench"]) == sorted(stems1)
    d = json.loads(TP.TAGGER_GOLD.read_text(encoding="utf-8"))
    keep = set(stems2)
    d["clips"] = [c for c in d.get("clips", []) if isinstance(c, dict) and Path(str(c.get("clip", ""))).stem in keep]
    tmp = o2 / "test2_gold_only.json"
    DCC.dump(tmp, d)
    gold2 = S.load_gold([tmp])
    rows, cost, parts = {}, {}, {}
    for n in ("B0r", arm):
        r1 = [S.score_clip(gold1[st], P1[n][st]) for st in stems1]
        r2 = [S.score_clip(gold2[st], P2[n][st]) for st in stems2 if st in gold2]
        rows[n] = DCC.metrics(r1 + r2)
        parts[n] = {"old_test": DCC.metrics(r1), "tagger_test": DCC.metrics(r2)}
        cost[n] = [DCC.clip_cost(r) for r in r1 + r2]
    d_ = DCC.boot(np.subtract(cost[arm], cost["B0r"]))
    v, rule = F.verdict(rows[arm], rows["B0r"], d_)
    res = {"arm": arm, "clips": {"old_test": len(stems1), "tagger_test": len([s for s in stems2 if s in gold2])},
           "rows": rows, "parts": parts, "delta_vs_B0r": d_, "verdict": v, "rule_hits_wrong": rule, "gates_old": g}
    DCC.dump(OUT, res)
    lines = [f"# Merged TEST (one exposure): {arm} vs B0r", "",
             f"clips: old TEST {res['clips']['old_test']} + tagger TEST {res['clips']['tagger_test']}", "",
             "| arm | hits | misses | wrong (v / c / p) | cost | old TEST hits / wrong | tagger TEST hits / wrong |",
             "|---|---|---|---|---|---|---|"]
    for n in ("B0r", arm):
        x, p = rows[n], parts[n]
        lines.append(f"| {n} | {x['hits']} | {x['misses']} | {x['wrong']} ({x['visible']} / {x['cross']} / {x['phantom']}) | "
                     f"{x['viewer_cost']:.3f} | {p['old_test']['hits']} / {p['old_test']['wrong']} | "
                     f"{p['tagger_test']['hits']} / {p['tagger_test']['wrong']} |")
    lines += ["", f"d cost vs B0r: {d_[0]:+.3f} [{d_[1]:+.3f}, {d_[2]:+.3f}], one-sided p {d_[3]:.3f}; verdict: **{v}**"]
    MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(MD.read_text(encoding="utf-8"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("dasm", "stage", "score"))
    ap.add_argument("--arm", default="TO1+F7F8")
    ap.add_argument("--tag", default=None, help="record a later TEST read in final_test_<tag>.json (reported, not selected on)")
    a = ap.parse_args()
    set_tag(a.tag)
    {"dasm": dasm, "stage": lambda: stage(a.arm), "score": lambda: score(a.arm)}[a.step]()


if __name__ == "__main__":
    main()
