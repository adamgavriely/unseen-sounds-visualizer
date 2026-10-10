"""v1.4 check: the pipeline's own display code (src/stage6_visual_augmentation, config.use_shipped() v1.4 keys) on the
frozen system's stage-5 outputs re-decided by the a/b rule, all 158 clips, against the analysis numbers
(CURRENT_SYSTEM.md). Cluster CPU from ~/wt_slice (this checkout's src and config).

    python benchmark/gold/coverage/v14_parity.py -> v14_parity.md
"""
import copy
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.coverage import score_coverage as V

HERE = Path(__file__).resolve().parent
R13 = Path.home() / "MscProj_r13" / "data" / "work"
TOL = 0.011


def redecide(P, B, gates):
    """the a/b rule on the stored votes, both directions (as step2_gate_build, rule AB-m)"""
    P = copy.deepcopy(P)
    restored = set()
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
            P[i] = dict(t); restored.add(s["event_label"])
        elif s.get("augment") and g[0]["res"] == "pass" and silence:
            P[i] = dict(s); P[i]["augment"] = False
    for i, (s, t) in enumerate(zip(P, B)):
        if not s.get("augment") and t.get("augment") and str(s.get("reason", "")).startswith("a kind of "):
            if s["reason"][len("a kind of "):].split(",")[0] in restored:
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
    gold = S.load_gold([V.GOLD]); dev, test = V.stems("dev"), V.stems("test")
    config.use_shipped()
    # the frozen arm's display caches, as dump_pictures.py reads them; flashes and FlexSED frames from their caches
    config.MERGE_GAP, config.PICTURE_MIN_CONF, config.GROUP_ASK, config.GROUP_MAX_GAP = 2.5, None, True, 8.0
    config.GROUP_CACHE, config.DEPICT_EVENT, config.DEPICT_CACHE = str(R13 / "group_answers.json"), True, str(R13 / "depict_answers.json")
    config.MAX_AFTER_END = None
    fl = {**json.loads((HERE / "flashes_dev.json").read_text()), **json.loads((HERE / "flashes_test.json").read_text())}
    (HERE / "_flashes_all.json").write_text(json.dumps(fl))
    config.FLASH_CACHE = str(HERE / "_flashes_all.json")
    config.HOLD_FLEXSED_DIR = str(Path.home() / "MscProj" / "data" / "work" / "flexsed_cache")
    pf = json.loads((HERE / "pics_frozen.json").read_text(encoding="utf-8"))["clips"]
    res, L = {}, ["# v1.4 pipeline code vs the analysis (use_shipped v1.4 keys, frozen stage-5 outputs, a/b re-decision)", ""]
    for name, stems, gfile in (("DEV", dev, "gate_dev.json"), ("TEST", test, "gate_test.json")):
        gd = json.loads((HERE / gfile).read_text(encoding="utf-8"))
        rows = []
        for st in stems:
            r = gd[st]
            specs = redecide(r["P"], r["B"], r["gate"])
            dur = float(pf[st]["dur"]) if pf[st].get("dur") else None
            rows.append(V.score_clip_v2(gold[st], pictures(specs, dur or 10.0, st)))
        res[name] = rows
        a = V.aggregate(rows)
        L.append(f"- {name}: {a['hits']}/{a['needed']} hits, {a['wrong']} wrong, onset cost {a['onset_cost']:.3f}, cost_cov {a['cost_cov']:.3f}, "
                 f"cover {a['hit_cov']:.2f}, |end err| {a['end_abs_med']:.2f} s")
    a = V.aggregate(res["DEV"] + res["TEST"])
    L += [f"- all 158: {a['hits']}/{a['needed']} hits, {a['wrong']} wrong, onset cost {a['onset_cost']:.3f}, cost_cov {a['cost_cov']:.3f}",
          "", "Analysis (CURRENT_SYSTEM.md): DEV 31 / 12, TEST 28 / 22, all 59 / 34, onset cost 2.076."]
    (HERE / "v14_parity.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
