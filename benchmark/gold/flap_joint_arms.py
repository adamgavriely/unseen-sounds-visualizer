"""Round 32 arms (docs/prereg_round13_detector_push.md, "Round 32"): FineLAP in DASM's two seats (F8 span vote, DV clip veto)
through LISTENER_DASM_DIR = data/work/finelap_as_dasm (finelap_full.py build), bars from the P1 rule (span 0.329, clip bar
from benchmark/gold/finelap_full.json). No edit to src/ or config.py: the arms are registered here and the round-13 /
tagger harnesses run them.

    python benchmark/gold/flap_joint_arms.py dev      # from ~/MscProj_r13 (GPU): stage4, stage5, score on DEV 49
    python benchmark/gold/flap_joint_arms.py dev2     # from ~/MscProj_tg (GPU): stage4, stage5, gates on the tagger DEV part
    python benchmark/gold/flap_joint_arms.py merged   # from ~/MscProj_tg (CPU): merged-DEV scoring + CV selection
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import round13_dev as R

G = _ROOT / "benchmark" / "gold"
WORK = _ROOT / "data" / "work"
SPAN_BAR = 0.32922908663749695
AS_DASM = WORK / "finelap_as_dasm"
ALL_ARMS = ["SHIP6+FLR", "SHIP6+FLR+F1", "SHIP6+DV2"]
ARMS = [a for a in os.environ.get("FJ_ARMS", "SHIP6+FLR SHIP6+FLR+F1").split() if a in ALL_ARMS]
ALL = ["B0r", "SHIP6"] + ALL_ARMS


def patch_dv2():
    """amendment DV2: after fuse_flexsed (DV, BTP, CONT, PANNs veto, rescue), a non-rescued span whose family FineLAP clip
    max < FINELAP_CLIP_VETO2 is dropped unless Qwen P1 (F7's rule) or Audio Flamingo P1 V4 keeps it -- DV's own keep. The
    harness calls the name R.fuse_flexsed, so the wrapper is the pipeline's code plus this one veto; src/ is not edited."""
    from src.labels import canonical
    from src.stage4_audio_event_detection import listener_p1_lookup, _af_p1_accepts
    orig = R.fuse_flexsed

    def wrapped(*a, **k):
        events, flex_ids, ffw = orig(*a, **k)
        bar = getattr(config, "FINELAP_CLIP_VETO2", None)
        clip = getattr(config, "_CURRENT_CLIP", None)
        f = AS_DASM / f"{clip}.npz"
        if not bar or clip is None or not f.exists():
            return events, flex_ids, ffw
        z = np.load(f, allow_pickle=True)
        dpk = {}
        for i, lab in enumerate([str(x) for x in z["labels"]]):
            dpk[canonical(lab)] = max(dpk.get(canonical(lab), 0.0), float(z["fw"][:, i].max()))
        lp1 = listener_p1_lookup(getattr(config, "LISTENER_VCACHE", None), getattr(config, "LISTENER_CACHE", None), clip)
        gone = [e for e in events if not getattr(e, "rescued", False) and dpk.get(canonical(e.label), 1.0) < float(bar)
                and not lp1(e.label, e.start, e.end)[0] and not _af_p1_accepts(e)]
        events = [e for e in events if e not in gone]
        print(f"       [stage4] DV2 FineLAP clip veto ({bar}): dropped {len(gone)} span(s) {[(e.label, round(e.start, 2)) for e in gone]}", flush=True)
        return events, flex_ids, ffw
    R.fuse_flexsed = wrapped


def register():
    clip_bar = json.loads((G / "finelap_full.json").read_text(encoding="utf-8"))["clip_bar"]
    R.ARMS["SHIP6+FLR"] = {**R.ARMS["SHIP6"], "LISTENER_DASM_DIR": str(WORK / "finelap_as_dasm"), "LISTENER_DASM_BAR": SPAN_BAR,
                           "LISTENER_DASM_PAD": 0.08, "DASM_CLIP_VETO": float(clip_bar)}
    R.ARMS["SHIP6+FLR+F1"] = {**R.ARMS["SHIP6+FLR"], "LISTENER_NEW_TYPE_ONCE": True}
    R.ARMS["SHIP6+DV2"] = {**R.ARMS["SHIP6"], "FINELAP_CLIP_VETO2": float(clip_bar)}
    R.BASE.setdefault("FINELAP_CLIP_VETO2", None)
    patch_dv2()
    os.environ["TG_ARMS"] = " ".join(["SHIP6"] + ALL_ARMS)
    return clip_bar


def dev():
    register()
    R.stage4(ARMS)
    R.stage5(ARMS)
    R.score()


def dev2():
    register()
    from benchmark.gold import tagger_prep as T
    DCC, R2, stems = T.configure("dev2")
    for arm in ARMS:
        cfg = R2.arm_cfg(arm)
        miss = [str(Path(cfg["LISTENER_DASM_DIR"]) / f"{s}.npz") for s in stems if not (Path(cfg["LISTENER_DASM_DIR"]) / f"{s}.npz").exists()]
        miss += [p for p in (cfg.get("LISTENER_CACHE"), cfg.get("LISTENER_VCACHE"), cfg.get("LISTENER_AFCACHE")) if p and not Path(p).exists()]
        if miss:
            raise SystemExit(f"{arm}: inputs missing, not run: {miss[:5]} ({len(miss)})")
    R2.stage4(ARMS)
    R2.stage5(ARMS)
    T.gates("dev2", ["B0r", "SHIP6"] + ARMS)


def merged():
    clip_bar = register()
    from benchmark.gold import merged_dev as M
    from benchmark.gold import dev_candidates_check as DCC
    from benchmark.gold.flap_joint_sim import cv, passes, BASE
    have = [a for a in ALL if all((R.R13 / f"{a}_proposed" / st / "augmentations.json").exists() for st in DCC.dev_stems()[1])]
    dev_rows = M.rows_dev(have)
    dev2_rows = M.rows_dev2(have)
    ALL[:] = have
    res = {"clip_bar": clip_bar, "rows": {}, "per_part": {}, "verdict": {}, "hits_lost": {}}
    C = {}
    part = np.array([0] * len(dev_rows["B0r"]) + [1] * len(dev2_rows["B0r"]))
    for n in ALL:
        rr = [r for _s, r in dev_rows[n]] + [r for _s, r in dev2_rows[n]]
        res["rows"][n] = DCC.metrics(rr)
        res["per_part"][n] = {"dev": DCC.metrics([r for _s, r in dev_rows[n]]), "dev2": DCC.metrics([r for _s, r in dev2_rows[n]])}
        C[n] = np.array([DCC.clip_cost(r) for r in rr])
    base = res["rows"]["SHIP6"]
    BASE.update({"hits": base["hits"], "wrong": base["wrong"], "cost": base["viewer_cost"]})
    for n in [a for a in ALL_ARMS if a in ALL]:
        lost = [(pt, st) for pt, rows0, rows1 in (("dev", dev_rows["SHIP6"], dev_rows[n]), ("dev2", dev2_rows["SHIP6"], dev2_rows[n]))
                for (st, r0), (_s, r1) in zip(rows0, rows1) if r1["hit"] < r0["hit"]]
        x = res["rows"][n]
        res["hits_lost"][n] = lost
        res["verdict"][n] = passes({"hits": x["hits"], "wrong": x["wrong"], "cost": x["viewer_cost"]}, lost)
        res.setdefault("delta_vs_SHIP6", {})[n] = DCC.boot(C[n] - C["SHIP6"])
    res["cv"] = cv({n: C[n] for n in ["SHIP6"] + [a for a in ALL_ARMS if a in ALL]}, part, "SHIP6")
    for n in ALL:
        x = res["rows"][n]; pp = res["per_part"][n]
        print(f"MERGED DEV {n:13s} hits {x['hits']}/{x['hits'] + x['misses']} wrong {x['wrong']} ({x['visible']}/{x['cross']}/{x['phantom']}) "
              f"cost {x['viewer_cost']:.3f} | DEV {pp['dev']['hits']}/{pp['dev']['wrong']} | DEV2 {pp['dev2']['hits']}/{pp['dev2']['wrong']}"
              + (f" | {res['verdict'][n]} lost {res['hits_lost'][n]} d {res['delta_vs_SHIP6'][n]}" if n in res["verdict"] else ""))
    print("CV:", json.dumps(res["cv"]))
    (G / "flap_joint_arms.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    {"dev": dev, "dev2": dev2, "merged": merged}[sys.argv[1]]()
