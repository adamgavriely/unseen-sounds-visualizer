"""Confirmation set 1, REVISED (docs/prereg_round13_detector_push.md): the ONE scoring of TEST2 (the sealed half of the tagger
set, benchmark/gold/test2_stems.txt; later batches' TEST2 clips are appended to that file before this runs).

Arms (stage 4/5 through round13_dev on the split's caches, via tagger_prep.configure("test2"); resumable, gold-free):
  B0r  primary baseline (the scored config), gates D0 (stage 4 == render trace) and D5 (B0r == render)
  B1   shipped stage 4 (BEATs self-veto 0.1218, PANNs off)
  C1   TO1+F7F8 (frozen candidate); C2 only if given with --c2 (the amendment-G arm, if the coordinator rules it in)
Then, and only then, the TEST2 tags are read: tagger_AG.json filtered to the TEST2 stems ON PARSE (no other clip kept),
score_per_sound on both systems, paired clip bootstrap 2000 seed 0 of the per-clip cost difference vs B0r (and vs B1),
one-sided p = share of draws >= 0, Holm over the candidates (primary = proposed). Verdict per candidate (prereg):
  better iff hits do not drop AND wrong <= B0r wrong + 2 x hits gained AND d cost < 0 with Holm-adjusted one-sided p < 0.05
  worse  iff d cost > 0 with lower 95 % bound > 0 (or the hits/wrong rule fails with d cost > 0)
  same   otherwise
ONE EXPOSURE: refuses to run if benchmark/gold/test2_final.json or its .started marker exists; the marker is written
immediately before the tags are read. --dry-run: everything up to the tag read, which is stubbed to raise.

    python benchmark/gold/test2_final.py stage4 [--c2 NAME]
    python benchmark/gold/test2_final.py stage5 [--c2 NAME]
    python benchmark/gold/test2_final.py score  [--c2 NAME]      # THE exposure
    python benchmark/gold/test2_final.py score --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import tagger_prep as T              # installs the load_gold stub; T._REAL_LOAD_GOLD kept
from benchmark.gold import score_per_sound as S

OUT = T.GOLDD / "test2_final.json"
MD = T.GOLDD / "test2_final.md"
STARTED = OUT.with_suffix(".started")
SPLIT = "test2"
PRIMARY = "proposed"


def guard():
    for p in (OUT, STARTED):
        if p.exists():
            raise SystemExit(f"ONE EXPOSURE: {p} exists -- TEST2 has already been scored (or scoring started); refusing")


def setup(c2):
    DCC, R, stems = T.configure(SPLIT)
    cands = [T.C1]
    if c2:
        if c2 not in R.ARMS:
            raise SystemExit(f"unknown arm {c2}")
        cmap = {"dev_listener.json": T.lcache(SPLIT), "dev_listener_v.json": T.lcache(SPLIT, "_v"),
                "dev_listener_afn.json": T.lcache(SPLIT, "_afn"), "dasm_cache": DCC.DASM_DIR,
                "flexsed_extra_dev": T.WORK / f"flexsed_extra_{SPLIT}"}
        new = {}
        for k, v in R.ARMS[c2].items():                       # the same DEV -> split mapping as tagger_prep.configure
            if isinstance(v, str) and Path(v).name in cmap:
                v = str(cmap[Path(v).name])
            if isinstance(v, str) and any(b in v.replace("\\", "/") for b in
                                          ("dev_listener", "/devcand", "flexsed_extra_dev", "test_listener", "r13test/")):
                raise SystemExit(f"{c2}: flag {k} = {v} still points at a DEV/TEST file")
            new[k] = v
        R.ARMS[c2] = new
        cands.append(c2)
    arms = ["B0r", "B1"] + cands
    print(f"[setup] TEST2 {len(stems)} clips; arms {arms}; " + "; ".join(f"{a} = {R.ARMS[a]}" for a in cands), flush=True)
    return DCC, R, stems, arms, cands


def gates(DCC, R, stems, arms):
    o = T.out(SPLIT)
    s4 = json.loads(R.STAGE4.read_text(encoding="utf-8"))
    g = {"complete": {}, "d0": {}, "d5": {}}
    for a in arms:
        for sysn in T.SYSTEMS:
            g["complete"][f"{a}|{sysn}"] = {"stage4": len(s4["arms"].get(f"{a}|{sysn}", {})),
                                            "stage5": sum((o / f"{a}_{sysn}" / st / "augmentations.json").exists() for st in stems),
                                            "of": len(stems)}
    d0 = {k: v for k, v in s4["d0"].items() if k.split("|", 1)[1] in set(stems)}
    g["d0"] = {"pass": sum(v["pass"] for v in d0.values()), "of": len(d0), "fails": [k for k, v in d0.items() if not v["pass"]]}
    for sysn in T.SYSTEMS:
        bad = [st for st in stems
               if DCC.pics_sig(S.load_pictures(o / f"B0r_{sysn}", st, sysn) or [])
               != DCC.pics_sig(S.load_pictures(DCC.scored_dir(sysn), st, sysn) or [])
               or DCC.spec_sig(o / f"B0r_{sysn}", st) != DCC.spec_sig(DCC.scored_dir(sysn), st)]
        g["d5"][sysn] = {"pass": len(stems) - len(bad), "of": len(stems), "differ": bad}
    g["pass"] = bool(all(v["stage4"] == v["stage5"] == v["of"] for v in g["complete"].values())
                     and g["d0"]["of"] == 2 * len(stems) and not g["d0"]["fails"]
                     and all(not v["differ"] for v in g["d5"].values()))
    print(f"[gates] complete {g['complete']}\n[gates] D0 {g['d0']['pass']}/{g['d0']['of']}; D5 "
          + ", ".join("%s %d/%d" % (s, v["pass"], v["of"]) for s, v in g["d5"].items()) + f"; pass {g['pass']}", flush=True)
    return g


def holm_adjusted(ps):
    """Holm step-down adjusted p-values (one-sided p per candidate)"""
    order = sorted(ps, key=lambda k: ps[k])
    m, run, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k]))
        out[k] = run
    return out


def verdict(x, y, d, p_adj):
    gain = x["hits"] - y["hits"]
    rule = gain >= 0 and x["wrong"] <= y["wrong"] + 2 * gain
    if rule and d[0] < 0 and p_adj < 0.05:
        return "better", rule
    if d[0] > 0 and d[1] > 0:
        return "worse", rule
    return "same", rule


def score(a, DCC, R, stems, arms, cands):
    g = gates(DCC, R, stems, arms)
    if not g["pass"]:
        raise SystemExit("gates failed: tags NOT read, nothing scored")
    if a.dry_run:
        try:
            S.load_gold([T.TAGGER_GOLD])                     # tagger_prep's stub
        except RuntimeError as e:
            print(f"[dry-run] gates pass; tag loading stub raised as expected ({e}); nothing scored", flush=True)
            return
        raise SystemExit("dry-run: the stub did not raise")
    guard()
    STARTED.write_text(json.dumps({"arms": arms, "flags": {c: R.ARMS[c] for c in cands}, "stems": stems,
                                   "time": datetime.now().isoformat()}), encoding="utf-8")
    keep = set(stems)
    d = json.loads(T.TAGGER_GOLD.read_text(encoding="utf-8"))
    d["clips"] = [c for c in d.get("clips", []) if isinstance(c, dict) and Path(str(c.get("clip", ""))).stem in keep]
    tmp = T.out(SPLIT) / "test2_gold_only.json"
    DCC.dump(tmp, d)
    del d
    gold = T._REAL_LOAD_GOLD([tmp])
    tmp.unlink()
    assert set(gold) <= keep
    sc = [st for st in stems if st in gold]
    o = T.out(SPLIT)
    s4 = json.loads(R.STAGE4.read_text(encoding="utf-8"))
    names = ["B0"] + arms
    res = {"plan": "docs/prereg_round13_detector_push.md (Confirmation set 1, REVISED): TEST2, one exposure", "stems": stems,
           "scored_clips": sc, "no_tags": sorted(keep - set(gold)), "candidates": cands,
           "flags": {x: R.ARMS[x] for x in arms}, "gates": g, "rows": {}, "heard": {}, "delta_vs_B0r": {}, "delta_vs_B1": {},
           "holm": {}, "verdict": {}, "rule_hits_wrong": {}, "needed_changes": {}, "primary_system": PRIMARY}
    for sysn in T.SYSTEMS:
        P = {"B0": {st: S.load_pictures(DCC.scored_dir(sysn), st, sysn) or [] for st in sc}}
        for x in arms:
            with R.flags({k: R.arm_cfg(x)[k] for k in R.DISPLAY_KEYS}):
                P[x] = {st: S.load_pictures(o / f"{x}_{sysn}", st, sysn) or [] for st in sc}
        rows = {n: [S.score_clip(gold[st], P[n][st]) for st in sc] for n in names}
        R_ = {n: DCC.metrics(r) for n, r in rows.items()}
        res["rows"][sysn] = R_
        res["heard"][sysn] = {x: sum(S.score_clip(gold[st], DCC.heard_pics(s4["arms"][f"{x}|{sysn}"][st], 0.35))["hit"]
                                     for st in sc) for x in arms}
        cost = {n: [DCC.clip_cost(r) for r in rr] for n, rr in rows.items()}
        res["delta_vs_B0r"][sysn] = {n: DCC.boot(np.subtract(cost[n], cost["B0r"])) for n in names if n != "B0r"}
        res["delta_vs_B1"][sysn] = {n: DCC.boot(np.subtract(cost[n], cost["B1"])) for n in names if n != "B1"}
        padj = holm_adjusted({c: res["delta_vs_B0r"][sysn][c][3] for c in cands})
        res["holm"][sysn] = {c: {"p": res["delta_vs_B0r"][sysn][c][3], "p_holm": padj[c]} for c in cands}
        res["verdict"][sysn], res["rule_hits_wrong"][sysn], res["needed_changes"][sysn] = {}, {}, {}
        for c in cands:
            v, rule = verdict(R_[c], R_["B0r"], res["delta_vs_B0r"][sysn][c], padj[c])
            res["verdict"][sysn][c], res["rule_hits_wrong"][sysn][c] = v, rule
            gained, lost = [], []
            for st in sc:
                for (gg, a0), (_g, a1) in zip(DCC.needed_hit(gold[st], P["B0r"][st]), DCC.needed_hit(gold[st], P[c][st])):
                    if a1 and not a0:
                        gained.append([st, gg["label"], gg["start"]])
                    if a0 and not a1:
                        lost.append([st, gg["label"], gg["start"]])
            res["needed_changes"][sysn][c] = {"gained": gained, "lost": lost}
    DCC.dump(OUT, res)
    MD.write_text(markdown(res, names), encoding="utf-8")
    print(MD.read_text(encoding="utf-8"), flush=True)
    print(f"-> {OUT}\n-> {MD}", flush=True)


def markdown(res, names):
    L = []
    for sysn in T.SYSTEMS:
        R_, H = res["rows"][sysn], res["heard"][sysn]
        n = R_["B0r"]["hits"] + R_["B0r"]["misses"]
        L.append(f"*{'ours (with gate)' if sysn == 'proposed' else 'blind (no gate)'}, TEST2 {len(res['scored_clips'])} clips, "
                 f"{n} needed sounds; d = paired clip bootstrap 2000, seed 0; p one-sided, Holm over {res['candidates']}*\n")
        L.append("| row | heard | hits / %d | misses | wrong (vis / cross / phantom) | cost | d cost vs B0r [95 %% CI], p (Holm) "
                 "| d cost vs B1 [95 %% CI] |" % n)
        L.append("|---|---|---|---|---|---|---|---|")
        for k in names:
            x = R_[k]; d = res["delta_vs_B0r"][sysn].get(k); e = res["delta_vs_B1"][sysn].get(k)
            ph = res["holm"][sysn].get(k, {}).get("p_holm")
            f = "—" if d is None else f"{d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}], p {d[3]:.3f}" + (f" ({ph:.3f})" if ph is not None else "")
            f1 = "—" if e is None else f"{e[0]:+.2f} [{e[1]:+.2f}, {e[2]:+.2f}]"
            L.append(f"| {k} | {H.get(k)} | {x['hits']} | {x['misses']} | {x['wrong']} ({x['visible']} / {x['cross']} / "
                     f"{x['phantom']}) | {x['viewer_cost']:.2f} | {f} | {f1} |")
        for c in res["candidates"]:
            L.append(f"\n**{c} ({sysn}{', primary' if sysn == res['primary_system'] else ''}): {res['verdict'][sysn][c]}** "
                     f"(hits/wrong rule {'holds' if res['rule_hits_wrong'][sysn][c] else 'fails'}); needed gained "
                     f"{res['needed_changes'][sysn][c]['gained']}, lost {res['needed_changes'][sysn][c]['lost']}")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("stage4", "stage5", "gates", "score"))
    ap.add_argument("--c2", default=None, help="the amendment-G arm, only if ruled in as C2")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    guard()
    DCC, R, stems, arms, cands = setup(a.c2)
    if a.step == "stage4":
        R.stage4(arms)
    elif a.step == "stage5":
        R.stage5(arms)
    elif a.step == "gates":
        gates(DCC, R, stems, arms)
    else:
        score(a, DCC, R, stems, arms, cands)


if __name__ == "__main__":
    main()
