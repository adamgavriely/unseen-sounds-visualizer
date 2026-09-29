"""Round 14 follow-up: the stricter listener verifiers on the BEATs weak band (pool P3), exactly as they were run on the FlexSED
runs (pool P2, amendment A / C of docs/prereg_round13_detector_push.md).

Candidates (per split): the cache's pool P3 (benchmark/gold/dev_listener.json, test_listener.json): BEATs label runs >= 0.175
with peak < 0.35, depictable only. DEV P3 excludes runs covered by a same-family B0r stage-4 span; TEST P3 has no coverage
filter (test_listener.json meta). Same audio cut as the cache scored them: [cut_start - 1, cut_end + 1] clipped, >= 1 s
(cut_start/cut_end = the run itself for P3), from the same wav16 files.
Verifiers (code reused, not copied):
  Qwen3-Omni   listener_variants.score() unchanged (V1 MC, V2 paired cut, V3 localisation, V4 open inventory, V12 = V1 and V2);
               V3 is run only because that loop reaches V4 through it
  AF Next      listener_afnext.run() unchanged (V4 open inventory, same prompt + matching; YN0 sanity flag)
Null control: the item's cached null_family on the same cut. No gold is read by pool / score / merge (score_per_sound.load_gold
raises). The DEV report reads DEV gold (score_per_sound.score_clip per span alone) -- DEV only; TEST has no report.

    python benchmark/gold/listener_p3.py pool      # CPU: P3 depictable -> *_p3_pool.json and *_p3_q.json (distractors, controls)
    python benchmark/gold/listener_p3.py qwen      # GPU: Qwen3-Omni V1-V4, DEV then TEST -> *_p3_q.json (resumable)
    python benchmark/gold/listener_p3.py afn       # GPU: Audio Flamingo Next V4 + YN, DEV then TEST -> *_p3_afn.json (resumable)
    python benchmark/gold/listener_p3.py merge     # CPU: -> dev_listener_p3.json / test_listener_p3.json
    python benchmark/gold/listener_p3.py report    # CPU: DEV screen (DEV gold)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold import dev_candidates_check as C
from benchmark.gold import listener_variants as LV

GOLD = _ROOT / "benchmark" / "gold"
SPLITS = ("dev", "test")
SRC = {"dev": GOLD / "dev_listener.json", "test": GOLD / "test_listener.json"}
POOL = {s: GOLD / f"{s}_listener_p3_pool.json" for s in SPLITS}
PART_Q = {s: GOLD / f"{s}_listener_p3_q.json" for s in SPLITS}
PART_A = {s: GOLD / f"{s}_listener_p3_afn.json" for s in SPLITS}
OUT = {s: GOLD / f"{s}_listener_p3.json" for s in SPLITS}
KEEP = ("clip", "pool", "family", "label", "start", "end", "cut_start", "cut_end", "run_len", "peak", "peak_t", "depictable",
        "null_family", "gold", "score", "null_score")
TIER_B = 0.26                                           # screen only: BEATs peak (NOT the pipeline TIER's FlexSED 0.6)


def key(x):
    return (x["clip"], x["family"], round(float(x["start"]), 3), round(float(x["end"]), 3))


# ============================================================================= candidates
def build(split, force=False):
    import soundfile as sf
    S.load_gold = LV._no_gold
    if PART_Q[split].exists() and not force:
        d = json.loads(PART_Q[split].read_text(encoding="utf-8"))
        if any("accept" in x for x in d["items"]):
            raise SystemExit(f"{PART_Q[split]} already has scores; pass --force to rebuild")
    src = json.loads(SRC[split].read_text(encoding="utf-8"))
    wavd = LV.SPLITS[split]["wav"]
    out = []
    for x in src["items"]:
        if x["pool"] != "P3" or not x["depictable"]:
            continue
        it = {k: x[k] for k in KEEP if k in x}
        it["cached_score"], it["cached_null_score"] = it.pop("score", None), it.pop("null_score", None)
        it["run_start"], it["run_end"] = x["start"], x["end"]
        it["variants"] = ["V1", "V2", "V3", "V4"]
        out.append(it)
    ks = [key(x) for x in out]
    assert len(set(ks)) == len(ks), "P3 key (clip, family, start, end) not unique"
    O = LV.Onto()
    flex, durs = {}, {}
    for it in out:                                        # listener_variants.build's distractor / control loop
        for tag, fam in (("x", it["family"]), ("null", it["null_family"])):
            ds, s_ = LV.distractors(O, it["clip"], fam, it["start"])
            it[f"v1_{tag}_options"], it[f"v1_{tag}_source"] = ds, s_
        if it["clip"] not in flex:
            flex[it["clip"]] = C.load_fr(LV.FLEX_DIR / f"{it['clip']}.npz")
            durs[it["clip"]] = float(sf.info(str(wavd / f"{it['clip']}.wav")).duration)
        dur = durs[it["clip"]]
        a = max(0.0, it["cut_start"] - 1.0); b = min(dur, it["cut_end"] + 1.0)
        it["run_audio"] = [a, max(b, a + 1.0)]
        for tag, fam in (("x", it["family"]), ("null", it["null_family"])):
            it[f"v2_{tag}_ctrl"], it[f"v2_{tag}_flex_col"] = LV.control(flex[it["clip"]], fam, it, dur)
        it["v3_cut"] = [max(0.0, it["run_start"] - 3.0), min(dur, it["run_end"] + 3.0)]
    nocol = sum(not x["v2_x_flex_col"] for x in out)
    meta = {"what": "P3 (BEATs weak band) verified by the amendment-A Qwen variants and the amendment-C AF Next V4",
            "split": split, "source_cache": str(SRC[split]), "source_P3": src["_meta"].get("P3"),
            "coverage_filter": "DEV: runs covered by a same-family B0r stage-4 span excluded; TEST: none (cache meta)",
            "candidates": "cache pool P3 with depictable True", "model": LV.MODEL,
            "audio": f"[cut_start - 1, cut_end + 1] clipped, >= 1 s, from {wavd} (the cache's P3 cut)",
            "V1": LV.V1_Q, "V2": "listener_round.QUESTION on the run cut minus the same on the control cut "
            f"(FlexSED(X) max < {LV.CTRL_BAR}, same length, nearest, not overlapping the run cut; no FlexSED column for X: "
            "only the overlap condition)", "V3": LV.V3_Q, "V4": LV.V4_Q,
            "rules": {"V1": "p(X) > 0.5 and p(X) > 2 x max(other)", "V2": "s_run > 3 and s_run - s_ctrl > 2",
                      "V3": "run_start - 0.5 <= t <= run_end + 0.5", "V4": "some line matches X", "V12": "V1 and V2"},
            "v2_x_no_flexsed_column": nocol, "n": len(out), "clips": len({x["clip"] for x in out})}
    C.dump(POOL[split], {"_meta": meta, "items": out})
    C.dump(PART_Q[split], {"_meta": meta, "items": json.loads(json.dumps(out))})
    print(f"[pool] {split}: {len(out)} P3 depictable items on {meta['clips']} clips; cuts "
          f"{len({(x['clip'], round(x['run_audio'][0], 3), round(x['run_audio'][1], 3)) for x in out})}; V1 sources x "
          f"{LV.count(out, 'v1_x_source')}; control found x {sum(x['v2_x_ctrl'] is not None for x in out)} null "
          f"{sum(x['v2_null_ctrl'] is not None for x in out)}; no FlexSED column for X {nocol}", flush=True)
    for x in out[:3]:
        print(f"   {x['clip']} {x['family']} {x['start']:.2f}-{x['end']:.2f} peak {x['peak']:.3f} audio {x['run_audio']} "
              f"options {x['v1_x_options']} ctrl {x['v2_x_ctrl']} null {x['null_family']}", flush=True)


# ============================================================================= GPU
def score_qwen():
    LV.SPLITS = {s: {**LV.SPLITS[s], "out": PART_Q[s]} for s in SPLITS}      # listener_variants.score() on the P3 file
    LV.score()
    for s in SPLITS:
        summary_q(json.loads(PART_Q[s].read_text(encoding="utf-8"))["items"], s)


def summary_q(items, split):
    its = [x for x in items if "accept" in x]
    print(f"[summary] {split} P3 Qwen n {len(its)}: " + ", ".join(
        f"{r} {sum(x['accept'][r] for x in its)} (null {sum(x['null_accept'][r] for x in its)})"
        for r in ("V1", "V2", "V3", "V4", "V12")), flush=True)


def score_afn():
    from benchmark.gold import listener_afnext as AF
    S.load_gold = LV._no_gold
    M = AF.load_model()
    for split in SPLITS:
        outp = PART_A[split]
        if outp.exists():
            d = json.loads(outp.read_text(encoding="utf-8"))
            items, meta = d["items"], d["_meta"]
        else:
            items = [{k: x[k] for k in AF.KEEP if k in x}
                     for x in json.loads(POOL[split].read_text(encoding="utf-8"))["items"]]
            meta = AF.make_meta(M, split)
            meta["candidates"] = f"{POOL[split]} (cache pool P3, depictable)"
        AF.run(M, items, split, outp, meta)
        its = [x for x in items if "accept" in x]
        print(f"[summary] {split} P3 AF n {len(its)}: V4 {sum(x['accept']['V4'] for x in its)} (null "
              f"{sum(x['null_accept']['V4'] for x in its)}), YN>0 {sum(x['accept']['YN0'] for x in its)} (null "
              f"{sum(x['null_accept']['YN0'] for x in its)})", flush=True)


# ============================================================================= merge
AF_FIELDS = ("afn_v4_text", "afn_v4_x", "afn_v4_null", "afn_yn_x", "afn_yn_null", "afn_yn_x_top1", "afn_yn_null_top1")


def merge():
    for split in SPLITS:
        dq = json.loads(PART_Q[split].read_text(encoding="utf-8"))
        da = json.loads(PART_A[split].read_text(encoding="utf-8"))
        a = {key(x): x for x in da["items"]}
        assert len(a) == len(da["items"]) == len(dq["items"]), (len(a), len(da["items"]), len(dq["items"]))
        out = []
        for x in dq["items"]:
            y = a[key(x)]
            assert "accept" in x and "accept" in y, ("unscored", key(x))
            it = dict(x)
            it["afn_accept"], it["afn_null_accept"] = y["accept"], y["null_accept"]
            for f in AF_FIELDS:
                it[f] = y.get(f)
            for tag, q, f in (("x", x["accept"], y["accept"]), ("null", x["null_accept"], y["null_accept"])):
                it["screen" if tag == "x" else "null_screen"] = {
                    "QV4": q["V4"], "AFV4": f["V4"], "BOTH_V4": q["V4"] and f["V4"],
                    "TIER_B026": q["V4"] and (x["peak"] >= TIER_B or f["V4"])}
            out.append(it)
        meta = {"what": "P3 (BEATs weak band, depictable) with the Qwen3-Omni amendment-A flags (accept / null_accept: V1 V2 V3 "
                "V4 V12, as dev_listener_v.json) and the AF Next amendment-C flags (afn_accept / afn_null_accept: V4 YN0, as "
                "dev_listener_afn.json's accept); key (clip, family, start, end)",
                "screen": {"QV4": "Qwen V4", "AFV4": "AF V4", "BOTH_V4": "Qwen V4 and AF V4",
                           "TIER_B026": f"BEATs peak >= {TIER_B}: Qwen V4 alone, else Qwen V4 and AF V4 (screen only; the "
                           "pipeline TIER uses a FlexSED peak of 0.6)"},
                "qwen_meta": dq["_meta"], "afn_meta": da["_meta"]}
        C.dump(OUT[split], {"_meta": meta, "items": out})
        print(f"[merge] {split}: {len(out)} items -> {OUT[split]}; " + ", ".join(
            f"{r} {sum(x['screen'][r] for x in out)} (null {sum(x['null_screen'][r] for x in out)})"
            for r in ("QV4", "AFV4", "BOTH_V4", "TIER_B026")) + f"; Qwen V12 {sum(x['accept']['V12'] for x in out)}",
            flush=True)


# ============================================================================= DEV report (DEV gold)
def missed_needed(gold):
    """needed sounds the current best arm (round 14 TO1+F7F8, proposed) misses on DEV: B0r misses in dev_heard_dropped.json
    minus the arm's gained list in round13_dev.json (needed_changes; lost = [])"""
    hd = json.loads((GOLD / "dev_heard_dropped.json").read_text(encoding="utf-8"))["rows"]
    nc = json.loads((GOLD / "round13_dev.json").read_text(encoding="utf-8"))["needed_changes"]["proposed"]["TO1+F7F8"]
    assert not nc["lost"], nc["lost"]
    gained = {(c, l, round(o, 1)) for c, l, o in nc["gained"]}
    out = []
    for r in hd:
        if r["hit_B0r"] or (r["clip"], r["label"], round(r["onset"], 1)) in gained:
            continue
        g = [x for x in gold[r["clip"]] if S.same_family(x["label"], r["label"]) and abs(x["start"] - r["onset"]) <= 0.06
             and x["needed"]]
        assert g, (r["clip"], r["label"], r["onset"])
        out.append({"clip": r["clip"], "label": g[0]["label"], "onset": g[0]["start"], "depictable": r.get("depictable"),
                    "beats": (r.get("evidence_window") or {}).get("beats")})
    return out


def report():
    d = json.loads(OUT["dev"].read_text(encoding="utf-8"))
    its = d["items"]
    gold = S.load_gold([GOLD / "annotations" / "gold_AG.json"])       # DEV clips only (report)
    miss = missed_needed(gold)
    print(f"[dev] P3 depictable n {len(its)}; cached gold {LV.count(its, 'gold')}; best arm TO1+F7F8 misses {len(miss)} needed")
    classes = ("hit", "visible", "cross", "phantom", "dontcare")

    def cls(x):
        r = S.score_clip(gold[x["clip"]], [(x["label"], x["start"], x["end"])])
        return next((c for c in classes if r[c]), "none?")

    def once(xs):
        best = {}
        for x in sorted(xs, key=lambda x: (x["start"], x["end"])):
            best.setdefault((x["clip"], x["family"]), x)
        return list(best.values())

    def recovers(xs):
        got = []
        for m in miss:
            sp = [x for x in xs if x["clip"] == m["clip"] and S.same_family(x["label"], m["label"])
                  and S.in_window(x["start"], m["onset"], S.EARLY, S.LATE)]
            if sp:
                got.append(f"{m['clip']} {m['label']} {m['onset']:.1f} (BEATs {sp[0]['peak']:.2f})")
        return got
    res = {}
    rules = {"Qwen V4": "QV4", "AF V4": "AFV4", "both": "BOTH_V4", f"TIER-like (BEATs peak >= {TIER_B})": "TIER_B026",
             "Qwen V12": None}
    print(f"[dev] pool upper bound (every P3 span): recovers {recovers(its)}")
    for name, r in rules.items():
        acc = [x for x in its if (x["screen"][r] if r else x["accept"]["V12"])]
        nul = sum((x["null_screen"][r] if r else x["null_accept"]["V12"]) for x in its)
        for tag, xs in (("all", acc), ("once", once(acc))):
            c = {k: 0 for k in classes + ("none?",)}
            for x in xs:
                c[cls(x)] += 1
            rec = recovers(xs)
            res[f"{name}|{tag}"] = {"n": len(xs), **c, "recovers": rec, "null_accepts": nul}
            print(f"[dev] {name:32s} {tag:4s} n {len(xs):3d}: hit {c['hit']} visible {c['visible']} cross {c['cross']} "
                  f"phantom {c['phantom']} dontcare {c['dontcare']}; null accepts {nul}/{len(its)}; recovers {rec}")
    print("[dev] currently missed needed sounds and their P3 spans (Qwen V4 / AF V4 / Qwen V12, AF text):")
    for m in miss:
        sp = [x for x in its if x["clip"] == m["clip"] and S.same_family(x["label"], m["label"])
              and S.in_window(x["start"], m["onset"], S.EARLY, S.LATE)]
        s = "; ".join(f"{x['label']} {x['start']:.2f} pk {x['peak']:.2f} Q{int(x['accept']['V4'])} A{int(x['afn_accept']['V4'])} "
                      f"V12{int(x['accept']['V12'])} q'{x.get('v4_text', '')[:40]!s}' a'{(x.get('afn_v4_text') or '')[:40]}'"
                      for x in sp) or "-"
        print(f"   {m['clip']} {m['label']} {m['onset']:.1f} dep {m['depictable']} beats {m['beats']}: {s}")
    C.dump(GOLD / "dev_listener_p3_report.json", {"missed": miss, "rules": res})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("pool", "qwen", "afn", "merge", "report"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.step == "pool":
        build("dev", a.force); build("test", a.force)
    elif a.step == "qwen":
        score_qwen()
    elif a.step == "afn":
        score_afn()
    elif a.step == "merge":
        merge()
    else:
        report()


if __name__ == "__main__":
    main()
