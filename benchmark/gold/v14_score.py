"""Reproduce the v1.6 / v1.5 / v1.4 / v1.3.11 numbers from a plain clone, CPU only, in about a minute.

Inputs (benchmark/gold/v14/): the stage-5 outputs of the 158 benchmark clips (each sound's plan with and without the
on-screen check, and the check's per-stretch votes), the flashes found in each video, FlexSED's family curves, the
BEATs / FlexSED / DASM family curves the v1.6 onset rules read (detector_curves.json, from the stage-4 caches), clip
durations, and the grouping / event-check answers the display reads. The script applies the system's own code:
the on-screen decision of config.VISIBILITY_RULE to the stored votes, then src/stage6_visual_augmentation's display
(_display_spans + _assign_rows) under config.use_shipped(), and scores the pictures with score_per_sound.py.

    python benchmark/gold/v14_score.py            # v1.6 (config.use_shipped(): v1.5 + the three onset rules)
    python benchmark/gold/v14_score.py --v1.5     # the same inputs with the three v1.6 rules switched off
    python benchmark/gold/v14_score.py --v1.4     # ... and the two v1.5 rules
    python benchmark/gold/v14_score.py --v1.3.11  # ... and the v1.4 rules
"""
import copy
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S

IN = Path(__file__).resolve().parent / "v14"
GOLD = Path(__file__).resolve().parent / "annotations" / "gold_AG.json"
TOL = 0.011


def stems(name):
    return [x.strip() for x in (Path(__file__).resolve().parent / f"{name}_stems.txt").read_text(encoding="utf-8").splitlines() if x.strip()]


def decide(P, B, gates):
    """the on-screen decision of config.VISIBILITY_RULE on the stored votes ("majority" keeps the stored plan P;
    "ab": a stretch is seen iff its a/b answer says so, a split falls back to the stored majority verdict). A sound
    the rule draws is planned as in B (the same plan without the check); a "kind of X" sound follows a drawn X."""
    if getattr(config, "VISIBILITY_RULE", "majority") != "ab":
        return P
    P = copy.deepcopy(P)
    if getattr(config, "REPEAT_LOCK", False):           # v1.5: as reason.py, on the stored gate votes
        for s in P:
            sp = s.get("spans") or []
            g = [x for x in gates if x["label"] == s["event_label"] and abs(x["start"] - s["start"]) < TOL and abs(x["end"] - s["end"]) < TOL]
            if (s.get("augment") and len(sp) > 1 and str(s.get("reason", "")).startswith("stage 2 saw the source somewhere")
                    and g and any(v.get("name") for v in g[0]["stretches"])):
                s["spans"] = sp[:1]
    drawn = set()
    seen = lambda v: v["seen"] if v["ab"] is None else bool(v["ab"])
    for i, (s, t) in enumerate(zip(P, B)):
        if not t.get("augment") or t["event_label"] != s["event_label"]:
            continue
        g = [x for x in gates if x["label"] == s["event_label"] and abs(x["start"] - s["start"]) < TOL and abs(x["end"] - s["end"]) < TOL]
        if not g or not g[0]["stretches"]:
            continue
        silence = all(seen(v) for v in g[0]["stretches"])
        gated = (not s.get("augment")) and str(s.get("reason", "")).startswith("source visible on screen")
        if gated and not silence:
            P[i] = dict(t); drawn.add(s["event_label"])
        elif s.get("augment") and g[0]["res"] == "pass" and silence:
            P[i] = dict(s); P[i]["augment"] = False
    for i, (s, t) in enumerate(zip(P, B)):
        if not s.get("augment") and t.get("augment") and str(s.get("reason", "")).startswith("a kind of "):
            if s["reason"][len("a kind of "):].split(",")[0] in drawn:
                P[i] = dict(t)
    return P


def pictures(specs, dur, stem):
    from src.types import AugmentationSpec
    from src.stage6_visual_augmentation import _display_spans, _assign_rows
    objs = [AugmentationSpec(index=s.get("index", 0), event_label=s["event_label"], start=float(s["start"]), end=float(s["end"]),
                             augment=bool(s.get("augment")), confidence=float(s.get("confidence", 0)), image_path=s.get("image_path"),
                             talked_about=bool(s.get("talked_about")), spans=[tuple(x) for x in s.get("spans", [])],
                             breaks=[tuple(x) for x in s.get("breaks", [])]) for s in specs]
    placed, _ = _assign_rows(_display_spans(objs, dur, require_image=True, clip=stem))
    return [(lab, float(a), float(b)) for _, lab, a, b, _ in placed]


def main():
    gold = S.load_gold([GOLD])                   # before use_shipped(): its label filter must not touch the gold set
    config.use_shipped()
    if "--v1.3.11" in sys.argv:
        config.VISIBILITY_RULE, config.FLASH_RULE, config.PICTURE_BAN, config.HOLD_FLEXSED = "majority", False, None, None
    if "--v1.3.11" in sys.argv or "--v1.4" in sys.argv:
        config.REPEAT_LOCK, config.UNVERIFIABLE_BAN = False, None
    if "--v1.3.11" in sys.argv or "--v1.4" in sys.argv or "--v1.5" in sys.argv:
        config.COONSET_CONTEST, config.WEAK_NO_RISE, config.BEATS_NO_RISE = None, None, False
    config.MAX_AFTER_END = None                  # as the benchmark scoring (score_per_sound.load_pictures)
    config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
    config.FLASH_CACHE, config.HOLD_CURVES = str(IN / "flashes.json"), str(IN / "flexsed_curves.json")
    config.HOLD_FLEXSED_DIR = str(IN / "no_npz_here")      # offline: the family curves above, not the FlexSED cache
    config.ONSET_CURVES = str(IN / "detector_curves.json")  # offline: v1.6 onset rules read these, not the stage-4 caches
    st5 = json.loads((IN / "stage5_specs.json").read_text(encoding="utf-8"))
    dur = json.loads((IN / "durations.json").read_text(encoding="utf-8"))
    for name in ("dev", "test"):
        rows = []
        for st in stems(name):
            r = st5[st]
            rows.append(S.score_clip(gold[st], pictures(decide(r["P"], r["B"], r["gate"]), float(dur[st] or 10.0), st)))
        a = S.aggregate(rows)
        wrong = a["visible"] + a["cross"] + a["phantom"]
        print(f"{name:4s} ({len(rows)} clips): {a['hits']} / {a['needed']} hits, {wrong} wrong, cost {a['viewer_cost']:.3f}")


if __name__ == "__main__":
    main()
