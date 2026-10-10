"""Root causes of the 34 v1.4 wrong pictures (158 clips) and three counter-rules read from the decision trail.

Each v1.4 picture (benchmark/gold/v14_score.py decide + pictures, as wrong_list.py) is linked to
  - its stage-5 plan entry: an augmented entry of the same family, the one whose burst ("spans") holds the picture start
    (else the nearest burst); k = the index of that burst;
  - its gate record (stage5_specs.json "gate": same label, same start);
  - its stage-4 candidates in docs/decision_trail/data.js: same family, reached the gate step, overlapping
    [picture start - 0.5 s, picture end].
Rules (all online: they read only one video's own trail):
  R2d re-trigger lock: the plan reason starts "stage 2 saw the source somewhere in the clip", at least one gate stretch
      answered name = yes, and the picture starts in burst k > 0 -> drop the picture (draw only the first burst).
  R2c (reference) as R2d without the stage-2 condition.
  R11 unverifiable label: every candidate's DASM check reads "n/a (no DASM query" and its FlexSED mirror reads
      "FlexSED has no query" -> no second detector can check the family -> drop.
  R7  (reference) one listener only: the open lists were asked (k4a_inventory not "not asked") and exactly one of the
      Qwen3-Omni list / Audio Flamingo list names the family (any candidate) -> drop.
  R7b R7, and no candidate came from FlexSED (origin "flexsed" / "flexsed band").
Reported: every on/off combination of {R2d, R11, R7b} on all 158 clips (hits, wrong, onset cost = (4 miss + 2 wrong) /
clips); clip-grouped 5-fold CV (random.Random(0) shuffle of stems('dev') + stems('test'), folds sh[k::5]) choosing the
combination by cost on the training clips; paired clip bootstrap (100000, seed 0) of out-of-fold cost vs v1.4.
The rules were found by reading the wrongs of all 158 clips (TEST included): the CV is a sanity check, not a held-out test.
    python benchmark/gold/coverage/wrong_types.py -> wrong_types.md"""
import itertools
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import decide, pictures, stems, IN, GOLD

HERE = Path(__file__).resolve().parent
TRAIL = _ROOT / "docs" / "decision_trail" / "data.js"

TAXONOMY = """| type | n | wrongs |
|---|---|---|
| D. sibling-label confusion (a real sound plays, wrong family) | 11 | Gunshot/Explosion x5 (m4_film_1917_33a 1.8, m4_live_fire_26a 2.5, m4_clay_shoot_11a, tg_d103 1.8, tg_d088 Explosion); mv_air_raid_scene Shofar for siren; tg_d001 Honk for duck; tg_d045 Siren for Ambulance (siren); tg_d107 Screaming for parrot; tg_d146 Rowboat for water; mv_protest_scene_movie Glass 4.8 |
| A. visible source, onset matched, gate missed | 7 | a/b flips: ambient_transport_subway_10800 Train, mv_protest_scene_movie Baby cry, tg_d110 Chink; tg_d128 Laughter, un_driving_motorcycle Explosion, m4_film_1917_33a Explosion 10.9, w8_dashcam_ambulance_behind_1a Siren 0.2 |
| B. re-trigger: later burst of a sound whose source the gate named in some stretch | 6 | as_explosion_XJ8lc3I6 8.2, tg_d088 Thunder 13.2, m4_film_1917_33a Gunshot 11.5, m4_live_fire_26a 8.6, w8_dashcam_ambulance_behind_1a Siren 11.0, tg_d103 Gunshot 8.8 |
| C. right family, picture start outside -0.5..+1 s of a long needed sound | 5 | b3_golf_course Bird, ly_applause Crowd (1.9 s early), tg_d031 Bird, w8_hide_wolves_howl_1a Baby cry, w8_hide_wolves_howl_1a Human locomotion |
| C'. same family as an on-screen sound, late | 2 | w8_dashcam_avalanche_road_2b Siren (Alarm), w8_dashcam_ambulance_behind_1a Shofar |
| E. phantom | 3 | un_hair_dryer Computer keyboard, m4_dog_doorbell_cam_30a Dog, tg_d023 Bee |

Not separable from the trail: C (rescued or two-listener confirmed, good DASM), A without a flip (gate 0 seen),
E Dog / keyboard (strong peaks, DASM >= 0.59, both listeners name them)."""


def split_list(s):
    return [x.strip() for x in re.split(r",\s(?=[A-Z])", s) if x.strip()]


def features(st, P, gate, pics, trail_clip):
    out = []
    for lab, a, b in pics:
        best, bd, bk = None, 1e9, 0
        for s in P:
            if not s.get("augment") or not S.same_family(lab, s["event_label"]):
                continue
            for k, (x, y) in enumerate(s.get("spans") or [[s["start"], s["end"]]]):
                d = 0.0 if x - 0.01 <= a <= y + 0.01 else min(abs(a - x), abs(a - y))
                if d < bd:
                    best, bd, bk = s, d, k
        fam = best["event_label"] if best else lab
        g = [x for x in gate if best and x["label"] == fam and abs(x["start"] - best["start"]) < 0.011]
        stretches = g[0]["stretches"] if g else []
        cands = [c for c in trail_clip["cands"] if S.same_family(c["label"], fam) and any(t["step"] == "gate" for t in c["trail"])
                 and c["start"] <= b and c["end"] >= a - 0.5]
        q = af = asked = False
        no_dasm = no_flex = bool(cands)
        flex_origin = False
        for c in cands:
            flex_origin |= "flexsed" in c["origin"]
            steps = {t["step"]: (t.get("value") or "") + " " + (t.get("note") or "") for t in c["trail"]}
            dv = steps.get("dasm_local_veto", steps.get("dasm_vote", ""))
            no_dasm &= "n/a (no DASM query" in dv
            no_flex &= "FlexSED has no query" in steps.get("mirror_veto", "")
            v = steps.get("k4a_inventory", "")
            if v and "not asked" not in v:
                asked = True
                m = re.search(r"Qwen list: (.*?)(?:; AF list: (.*))?\s*$", v)
                if m:
                    q |= any(S.same_family(x, fam) for x in split_list(m.group(1)))
                    af |= any(S.same_family(x, fam) for x in split_list(m.group(2) or ""))
        r7 = asked and (q != af)
        stage2 = bool(best) and str(best.get("reason", "")).startswith("stage 2 saw the source somewhere")
        name_yes = any(x.get("name") for x in stretches)
        out.append({"pic": (lab, a, b),
                    "R2d": bk > 0 and stage2 and name_yes, "R2c": bk > 0 and name_yes,
                    "R11": no_dasm and no_flex, "R7": r7, "R7b": r7 and not flex_origin})
    return out


def main():
    gold = S.load_gold([GOLD]); config.use_shipped(); config.MAX_AFTER_END = None
    config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
    config.FLASH_CACHE, config.HOLD_CURVES, config.HOLD_FLEXSED_DIR = str(IN / "flashes.json"), str(IN / "flexsed_curves.json"), str(IN / "x")
    st5 = json.loads((IN / "stage5_specs.json").read_text(encoding="utf-8")); dur = json.loads((IN / "durations.json").read_text(encoding="utf-8"))
    s = TRAIL.read_text(encoding="utf-8"); trail = {c["clip"]: c for c in json.loads(s[s.index("=") + 1:].strip().rstrip(";"))["clips"]}
    clips = stems("dev") + stems("test")
    F = {}
    for st in clips:
        r = st5[st]; P = decide(r["P"], r["B"], r["gate"])
        F[st] = features(st, P, r["gate"], pictures(P, float(dur[st] or 10), st), trail[st])

    def rows_for(rules, which):
        return {st: S.score_clip(gold[st], [f["pic"] for f in F[st] if not any(f[x] for x in rules)]) for st in which}

    def summ(rows):
        a = S.aggregate(list(rows.values()))
        return a["hits"], a["visible"] + a["cross"] + a["phantom"], a["viewer_cost"]

    base = rows_for((), clips)
    L = ["# v1.4 wrong pictures: root causes and trail counter-rules (158 clips)", "", "## Root-cause types (34 wrongs)", "", TAXONOMY, "",
         "## Single rules (all 158 clips)", "", "| rule | hits | wrong | cost | wrongs removed | hits lost |", "|---|---|---|---|---|---|"]
    h0, w0, c0 = summ(base)
    L.append(f"| v1.4 | {h0} | {w0} | {c0:.3f} | - | - |")
    for name in ("R2d", "R2c", "R11", "R7", "R7b"):
        h, w, c = summ(rows_for((name,), clips))
        L.append(f"| {name} | {h} | {w} | {c:.3f} | {w0 - w} | {h0 - h} |")
    combos = [tuple(x for x, on in zip(("R2d", "R11", "R7b"), bits) if on) for bits in itertools.product((0, 1), repeat=3)]
    L += ["", "R7 / R7b read the listener lists over all matched candidates (a family named in any candidate's list counts). "
          "The first exploration let the last candidate decide; that also dropped m4_film_1917_33a Explosion 10.9 (wrong) "
          "and, for R7 only, the w8_kids_fire_alarm_school_1b Alarm hit (R7b combination then 57 / 19)."]
    L += ["", "## Combinations of {R2d, R11, R7b} (all 158 clips; DEV / TEST hits-wrong)", "",
          "| rules | hits | wrong | cost | DEV | TEST |", "|---|---|---|---|---|---|"]
    for cb in combos:
        h, w, c = summ(rows_for(cb, clips)); hd, wd, _ = summ(rows_for(cb, stems("dev"))); ht, wt, _ = summ(rows_for(cb, stems("test")))
        L.append(f"| {'+'.join(cb) or 'none (v1.4)'} | {h} | {w} | {c:.3f} | {hd}/{wd} | {ht}/{wt} |")
    # clip-grouped 5-fold CV
    sh = list(clips); random.Random(0).shuffle(sh); folds = [sh[k::5] for k in range(5)]
    oof = {}; L += ["", "## Clip-grouped 5-fold CV (choice by training cost)", "", "| fold | test clips | chosen | train cost |", "|---|---|---|---|"]
    for k, fold in enumerate(folds):
        train = [c for c in clips if c not in fold]
        costs = [(summ(rows_for(cb, train))[2], i, cb) for i, cb in enumerate(combos)]
        tc, _, cb = min(costs)
        oof.update(rows_for(cb, fold))
        L.append(f"| {k} | {len(fold)} | {'+'.join(cb) or 'none'} | {tc:.3f} |")
    h, w, c = summ(oof)
    L += ["", f"Out-of-fold: {h} hits, {w} wrong, cost {c:.3f} (v1.4 {h0} / {w0} / {c0:.3f})"]
    pc = lambda r: S.COST_MISS * r["miss"] + S.COST_FA * (r["visible"] + r["cross"] + r["phantom"])
    d = np.array([pc(oof[st]) - pc(base[st]) for st in clips])
    rng = np.random.default_rng(0); idx = rng.integers(0, len(d), size=(100000, len(d)))
    bm = d[idx].mean(axis=1); lo, hi = np.percentile(bm, [2.5, 97.5])
    L += [f"Paired clip bootstrap (100000, seed 0), out-of-fold minus v1.4 cost per clip: {d.mean():+.3f} "
          f"[95% CI {lo:+.3f}, {hi:+.3f}], P(diff >= 0) = {float((bm >= 0).mean()):.4f}", "",
          "The rules were found by reading the wrongs of all 158 clips (TEST included); the CV is a sanity check, not a held-out test."]
    (HERE / "wrong_types.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
