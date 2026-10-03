"""Scores one detector variant on the 60 TEST clips, once (the ONE TEST exposure).

One job, 60 TEST clips (benchmark/gold/test_stems.txt), three stage-4 configs through the pipeline's own code
(dev_harness.build -> fuse_flexsed) and the same stage-5 path as DEV (dev_harness.stage5: scored gate answers reused,
other questions memoised; the memo starts from data/work/r13test/ask_memo.json, B0r's TEST prep run):
  B0r  the scored config (PANNs veto 0.05), flags off; gate D0 (stage 4 == scored trace) and D5 (== scored render)
  B1   the shipped stage 4: BEATs self-veto 0.1218, PANNs veto off
  ARM  the one candidate picked by the DEV selection rule (--arm NAME from dev_harness.ARMS, or --flags JSON)
A listener arm reads the gold-free TEST cache (benchmark/gold/test_listener.json) in place of dev_listener.json.
Then (only then) TEST gold is loaded and all three are scored with score_per_sound, both systems, as dev_harness.score:
heard by stage 4, hits, misses, wrong (visible / cross / phantom), cost; paired clip bootstrap 2000, seed 0, of the
per-clip cost difference vs B0r and vs B1; one-sided p = share of draws >= 0. Verdict (prereg), primary = proposed:
  better iff hits do not drop AND wrong rises by <= 2 x hits gained AND d cost < 0 with one-sided p < 0.05
  worse  iff d cost > 0 with lower 95 % bound > 0, OR the hits/wrong rule fails
  same   otherwise
ONE EXPOSURE: refuses to run if benchmark/gold/test_harness.json or its .started marker exists; the marker is written
immediately before gold is read. --dry-run: stages 4/5 into data/work/r13final_dry, gold loading stubbed to raise.

    python benchmark/gold/test_harness.py --arm R13-1 stage4
    python benchmark/gold/test_harness.py --arm R13-1 stage5
    python benchmark/gold/test_harness.py --arm R13-1 score          # THE exposure (reads TEST gold)
    python benchmark/gold/test_harness.py devcheck-b1                # DEV only: fuse_flexsed B1 == devcand B1 rows

(design record: release v1.2.0)
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

_REAL_LOAD_GOLD = S.load_gold                             # kept before test_harness_prep installs its raise-stub

OUT = _ROOT / "benchmark" / "gold" / "test_harness.json"
MD = _ROOT / "benchmark" / "gold" / "test_harness.md"
STARTED = OUT.with_suffix(".started")
GOLD = _ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"
TEST_CACHE = _ROOT / "benchmark" / "gold" / "test_listener.json"
CACHE_MAP = {"dev_listener.json": str(TEST_CACHE)}        # DEV cache file name -> TEST file (extend with --cache-map)
B1_FLAGS = {"BEATS_SELF_VETO": 0.1218, "PANNS_VETO": 0.0}
SYS_PRIMARY = "proposed"


def guard():
    for p in (OUT, STARTED):
        if p.exists():
            raise SystemExit(f"ONE EXPOSURE: {p} exists -- the TEST run has already happened (or started); refusing")


# ============================================================================= set-up (TEST redirects + arms)
def setup(a):
    from benchmark.gold import test_harness_prep as T        # TEST redirects + gold stub (restored only in score, real run)
    from benchmark.gold import dev_harness as R
    fin = T.WORK / ("r13final_dry" if a.dry_run else "r13final")
    R.R13 = fin
    R.STAGE4, R.MEMO = fin / "stage4.json", fin / "ask_memo.json"      # R.PANNS_DIR stays data/work/r13test/panns
    fin.mkdir(parents=True, exist_ok=True)
    if not R.MEMO.exists() and (T.OUT / "ask_memo.json").exists():
        shutil.copy(T.OUT / "ask_memo.json", R.MEMO)                  # a copy: the prep memo is not written
    R.ARMS.setdefault("B1", dict(B1_FLAGS))
    assert R.ARMS["B1"] == B1_FLAGS, R.ARMS["B1"]
    if a.flags:
        R.ARMS[a.arm] = json.loads(a.flags)
    if a.arm not in R.ARMS:
        raise SystemExit(f"unknown arm {a.arm}; give --flags")
    cmap = dict(CACHE_MAP)
    for m in a.cache_map or []:
        k, v = m.split("=", 1); cmap[k] = v
    flags = {}
    for k, v in R.ARMS[a.arm].items():
        if isinstance(v, str) and Path(v).name in cmap:
            v = cmap[Path(v).name]
        if isinstance(v, str) and ("dev_" in Path(v).name or "/devcand" in v.replace("\\", "/")):
            raise SystemExit(f"arm flag {k} = {v} points at a DEV file; map it with --cache-map")
        flags[k] = v
    R.ARMS[a.arm] = flags
    for k in a.stage5_keys or []:
        R.BASE.setdefault(k, getattr(__import__("config"), k, None))
        if k not in R.STAGE5_KEYS:
            R.STAGE5_KEYS = tuple(R.STAGE5_KEYS) + (k,)
    cfg = R.arm_cfg(a.arm)
    if cfg.get("LISTENER_RESCUE"):
        p = Path(cfg["LISTENER_CACHE"])
        d = json.loads(p.read_text(encoding="utf-8"))
        its = [x for x in d["items"] if x["clip"] in set(T.STEMS)]
        assert its and all(x.get("score") is not None for x in its), f"{p}: TEST items missing scores"
        assert {x["clip"] for x in d["items"]} <= set(T.STEMS), f"{p}: not a TEST cache"
        assert not any("gold" in x for x in d["items"]), f"{p}: has a gold field"
        print(f"[setup] listener cache {p}: {len(its)} scored TEST items", flush=True)
    arms = ["B0r", "B1"] + ([a.arm] if a.arm not in ("B0r", "B1") else [])
    print(f"[setup] {'DRY RUN ' if a.dry_run else ''}out {fin}; arms {arms}; {a.arm} = {R.ARMS[a.arm]}", flush=True)
    return T, R, arms, fin


# ============================================================================= gates (structural, no gold)
def gates(T, R, arms, fin):
    from benchmark.gold import dev_candidates_check as DCC
    stems = T.STEMS
    s4 = json.loads(R.STAGE4.read_text(encoding="utf-8"))
    g = {"complete": {}, "d0": {}, "d5": {}}
    for a in arms:
        for sysn in DCC.SYSTEMS:
            g["complete"][f"{a}|{sysn}"] = {"stage4": len(s4["arms"].get(f"{a}|{sysn}", {})),
                                            "stage5": sum((fin / f"{a}_{sysn}" / st / "augmentations.json").exists() for st in stems),
                                            "of": len(stems)}
    d0 = s4["d0"]
    g["d0"] = {"pass": sum(v["pass"] for v in d0.values()), "of": len(d0), "fails": [k for k, v in d0.items() if not v["pass"]]}
    for sysn in DCC.SYSTEMS:
        bad = [st for st in stems
               if DCC.pics_sig(S.load_pictures(fin / f"B0r_{sysn}", st, sysn) or [])
               != DCC.pics_sig(S.load_pictures(DCC.scored_dir(sysn), st, sysn) or [])
               or DCC.spec_sig(fin / f"B0r_{sysn}", st) != DCC.spec_sig(DCC.scored_dir(sysn), st)]
        g["d5"][sysn] = {"pass": len(stems) - len(bad), "of": len(stems), "differ": bad}
    g["listener"] = {k: {f: (len(v[f]) if isinstance(v[f], list) else v[f]) for f in v}
                     for k, v in s4.get("listener", {}).items()} if s4.get("listener") else {}
    print(f"[gates] complete {g['complete']}", flush=True)
    d5 = ", ".join("%s %d/%d" % (s, v["pass"], v["of"]) for s, v in g["d5"].items())
    print(f"[gates] D0 {g['d0']['pass']}/{g['d0']['of']} fails {g['d0']['fails']}; D5 {d5}", flush=True)
    ok = (all(v["stage4"] == v["of"] == v["stage5"] for v in g["complete"].values()) and not g["d0"]["fails"]
          and all(not v["differ"] for v in g["d5"].values()))
    g["pass"] = bool(ok)
    return g


# ============================================================================= THE exposure
def verdict(x, y, d):
    """prereg: x = arm row, y = B0r row, d = boot(arm - B0r) [mean, lo, hi, one-sided p]"""
    gain = x["hits"] - y["hits"]
    rule = gain >= 0 and x["wrong"] - y["wrong"] <= 2 * gain
    if rule and d[0] < 0 and d[3] < 0.05:
        return "better", rule
    if (d[0] > 0 and d[1] > 0) or not rule:
        return "worse", rule
    return "same", rule


def score(a, T, R, arms, fin):
    from benchmark.gold import dev_candidates_check as DCC
    g = gates(T, R, arms, fin)
    if not g["pass"]:
        raise SystemExit(f"gates failed ({g}); gold NOT read, nothing scored")
    if a.dry_run:
        try:
            S.load_gold([GOLD])                          # still test_harness_prep's stub
        except RuntimeError as e:
            print(f"[dry-run] stages 4/5 complete, gates pass; gold stub raised as expected ({e}); nothing scored", flush=True)
            return
        raise SystemExit("dry-run: the gold stub did not raise")
    guard()
    STARTED.write_text(json.dumps({"arm": a.arm, "flags": R.ARMS[a.arm], "time": datetime.now().isoformat()}), encoding="utf-8")
    S.load_gold = _REAL_LOAD_GOLD
    gold = S.load_gold([GOLD])
    stems = T.STEMS
    test = sorted(S.subsets_of(gold)["test_bench"])   # the frozen TEST table subset (test minus slice B); was "test" (30 slice-B clips too)
    assert test == sorted(stems), "gold test_bench subset != test_stems.txt"
    s4 = json.loads(R.STAGE4.read_text(encoding="utf-8"))
    names = ["B0", "B0r", "B1", a.arm] if a.arm not in ("B0r", "B1") else ["B0", "B0r", "B1"]
    res = {"plan": "docs/history/preregistrations/prereg_round13_detector_push.md, release v1.2.0 (TEST decision, one exposure)", "arm": a.arm,
           "flags": R.ARMS[a.arm], "b1_flags": R.ARMS["B1"], "base": R.BASE, "clips": len(stems), "gates": g,
           "rows": {}, "heard": {}, "delta_vs_B0r": {}, "delta_vs_B1": {}, "verdict": {}, "rule_hits_wrong": {},
           "needed_changes": {}, "picture_changes": {}, "primary_system": SYS_PRIMARY}
    for sysn in DCC.SYSTEMS:
        P = {"B0": {st: S.load_pictures(DCC.scored_dir(sysn), st, sysn) or [] for st in stems}}
        for n in names[1:]:
            P[n] = {st: S.load_pictures(fin / f"{n}_{sysn}", st, sysn) or [] for st in stems}
        rows = {n: [S.score_clip(gold[st], P[n][st]) for st in stems] for n in names}
        res["rows"][sysn] = {n: DCC.metrics(r) for n, r in rows.items()}
        heard = {n: sum(S.score_clip(gold[st], DCC.heard_pics(s4["arms"][f"{n}|{sysn}"][st], 0.35))["hit"] for st in stems)
                 for n in names[1:]}
        heard["B0"] = heard["B0r"]
        res["heard"][sysn] = heard
        cost = {n: [DCC.clip_cost(r) for r in rr] for n, rr in rows.items()}
        res["delta_vs_B0r"][sysn] = {n: DCC.boot(np.subtract(cost[n], cost["B0r"])) for n in names if n != "B0r"}
        res["delta_vs_B1"][sysn] = {n: DCC.boot(np.subtract(cost[n], cost["B1"])) for n in names if n != "B1"}
        if a.arm in names[3:]:
            v, rule = verdict(res["rows"][sysn][a.arm], res["rows"][sysn]["B0r"], res["delta_vs_B0r"][sysn][a.arm])
            res["verdict"][sysn], res["rule_hits_wrong"][sysn] = v, rule
            gained, lost = [], []
            for st in stems:
                for (gg, a0), (_g2, a1) in zip(DCC.needed_hit(gold[st], P["B0r"][st]), DCC.needed_hit(gold[st], P[a.arm][st])):
                    if a1 and not a0:
                        gained.append([st, gg["label"], gg["start"]])
                    if a0 and not a1:
                        lost.append([st, gg["label"], gg["start"]])
            res["needed_changes"][sysn] = {"gained": gained, "lost": lost}
            add, rem = DCC.diff_pics(P["B0r"], P[a.arm])
            res["picture_changes"][sysn] = {
                "appeared": [list(x) + [R.pic_type(gold[x[0]], P[a.arm][x[0]], R._find(P[a.arm][x[0]], x))] for x in add],
                "disappeared": [list(x) + [R.pic_type(gold[x[0]], P["B0r"][x[0]], R._find(P["B0r"][x[0]], x))] for x in rem]}
    DCC.dump(OUT, res)
    MD.write_text(markdown(res, names), encoding="utf-8")
    print(MD.read_text(encoding="utf-8"), flush=True)
    print(f"-> {OUT}\n-> {MD}", flush=True)


def markdown(res, names):
    L = []
    for sysn in ("proposed", "blind_a2i"):
        R_, H = res["rows"][sysn], res["heard"][sysn]
        n = R_["B0r"]["hits"] + R_["B0r"]["misses"]
        L.append(f"*{'ours (with gate)' if sysn == 'proposed' else 'blind (no gate)'}, TEST {res['clips']} clips, {n} needed "
                 f"sounds; d = paired clip bootstrap 2000, seed 0; p = one-sided (share of draws >= 0)*\n")
        L.append("| row | heard by stage 4 | hits / %d | misses | wrong (vis / cross / phantom) | cost | d cost vs B0r [95 %% CI], p "
                 "| d cost vs B1 [95 %% CI], p |" % n)
        L.append("|---|---|---|---|---|---|---|---|")
        for k in names:
            x = R_[k]
            f = lambda d: "—" if d is None else f"{d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}], p {d[3]:.3f}"
            L.append(f"| {k} | {H.get(k)} | {x['hits']} | {x['misses']} | {x['wrong']} ({x['visible']} / {x['cross']} / "
                     f"{x['phantom']}) | {x['viewer_cost']:.2f} | {f(res['delta_vs_B0r'][sysn].get(k))} | "
                     f"{f(res['delta_vs_B1'][sysn].get(k))} |")
        if sysn in res["verdict"]:
            L.append(f"\n**Verdict ({sysn}{', primary' if sysn == res['primary_system'] else ''}): {res['verdict'][sysn]}** "
                     f"(hits/wrong rule {'holds' if res['rule_hits_wrong'][sysn] else 'fails'}). Needed gained "
                     f"{res['needed_changes'][sysn]['gained']}; lost {res['needed_changes'][sysn]['lost']}.\n")
        L.append("")
    return "\n".join(L)


# ============================================================================= DEV check of the B1 path (no TEST)
def devcheck_b1():
    """B1 through dev_harness.build (fuse_flexsed, BEATS_SELF_VETO 0.1218, PANNs 0) == devcand's B1 stage-4 rows on DEV"""
    from benchmark.gold import dev_candidates_check as DCC
    from benchmark.gold import dev_harness as R
    R.ARMS.setdefault("B1", dict(B1_FLAGS))
    ref = json.loads((DCC.DC / "stage4.json").read_text(encoding="utf-8"))["arms"]
    stems = sorted(ref["B1|proposed"])
    sig = lambda rr: sorted((r["label"], round(r["pre_start"], 3), round(r["start"], 3), round(r["end"], 3),
                             round(r["conf"], 4), r["origin"]) for r in rr)
    bad, live = [], 0
    for st in stems:
        C = {"beats": DCC.load_fr(DCC.BEATS_DIR / f"{st}.npz"), "flex": DCC.load_fr(DCC.FLEX_DIR / f"{st}.npz")}
        for sysn in DCC.SYSTEMS:
            tr = json.loads((DCC.scored_dir(sysn) / st / "onset_trace.json").read_text(encoding="utf-8"))
            rows, _i = R.build(st, sysn, "B1", C, tr)
            live += sum(r["refine"] == "live" for r in rows)
            if sig(rows) != sig(ref[f"B1|{sysn}"][st]):
                bad.append(f"{sysn}|{st}")
    print(f"[devcheck-b1] {2 * len(stems) - len(bad)} / {2 * len(stems)} equal to devcand B1; live {live}; differ {bad}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("stage4", "stage5", "score", "gates", "devcheck-b1"))
    ap.add_argument("--arm", default="R13-1")
    ap.add_argument("--flags", default=None, help="JSON flags for an arm not in dev_harness.ARMS (or to override one)")
    ap.add_argument("--cache-map", nargs="*", help="DEVFILENAME=TESTPATH for a listener-variant cache")
    ap.add_argument("--stage5-keys", nargs="*", help="extra arm flags read after stage 4")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.step == "devcheck-b1":
        return devcheck_b1()
    guard()
    T, R, arms, fin = setup(a)
    if a.step == "stage4":
        R.stage4(arms)
    elif a.step == "stage5":
        R.stage5(arms)
    elif a.step == "gates":
        gates(T, R, arms, fin)
    else:
        score(a, T, R, arms, fin)


if __name__ == "__main__":
    main()
