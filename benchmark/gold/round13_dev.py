"""Round 13 detector push on DEV (docs/prereg_round13_detector_push.md). DEV only: no TEST, no slice-B clip.

Every arm runs the PIPELINE's own stage-4 code (src.stage4_audio_event_detection.fuse_flexsed / attach_breaks, flags in
config.py, all off = the scored behaviour) on the cached detector scores of the 49 DEV clips, then the same stage-5 path
as benchmark/gold/dev_candidates_check.py (gate answers of the scored run reused, every other question memoised), and
is scored with benchmark/gold/score_per_sound.py. Arms are built on top of B0r (the scored config, PANNs veto 0.05).

    python benchmark/gold/round13_dev.py panns     # GPU: PANNs CNN14 clip peaks on each DEV audio.wav (the veto input)
    python benchmark/gold/round13_dev.py stage4    # GPU only for occlusion onsets of new/changed BEATs spans; gates D0
    python benchmark/gold/round13_dev.py stage5    # GPU: stage 5 per arm and system
    python benchmark/gold/round13_dev.py score     # CPU: gate D5, table, bootstrap vs B0r, gained/lost, appeared/disappeared
    python benchmark/gold/round13_dev.py stage4 --offline   # local check: flags off == devcand B0r rows (PANNs from trace)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import score_per_sound as S
from src.labels import canonical
from src.stage4_audio_event_detection import (TRACE, LISTENER_STATS, _extract_events, attach_breaks, fuse_flexsed,
                                              listener_from_cache, listener_from_vcache, filter_rescued,
                                              listener_p1_lookup)
from src.types import AudioEvent

WORK = DCC.WORK
R13 = WORK / "r13"
STAGE4, MEMO, PANNS_DIR = R13 / "stage4.json", R13 / "ask_memo.json", R13 / "panns"
OUT = _ROOT / "benchmark" / "gold" / "round13_dev.json"
LISTENER = _ROOT / "benchmark" / "gold" / "dev_listener.json"
SYSTEMS = DCC.SYSTEMS

# the stage-4 part of B0 (config.use_scored(), job logs of the scored DEV run), set explicitly in every arm
BASE = {"AED_MODEL": "beats", "AED_THRESHOLD": 0.175, "DISPLAY_THRESHOLD": 0.35, "AED_MIN_DUR": 0.5,
        "AED_HYSTERESIS": 1.0, "AED_RELEASE": None, "FLEXSED_BAR": 0.8, "FLEXSED_FAMILY_BARS": None,
        "FLEXSED_VETO": 0.3, "PANNS_VETO": 0.05, "BEATS_SELF_VETO": 0.0, "FLEXSED_CORROB": None,
        "UNION_WEAK_TWIN": "absorb", "UNION_START": "min", "BEATS_LOWBAND_CORROB": None, "ONSET_CAM": True,
        "ONSET_MONOTONE": True, "MAX_SPAN": None, "MERGE_START": "earliest",
        "TWIN_MAX": False, "MIRROR_VETO": None, "MIRROR_OWN_MAX": 0.4, "IMPULSE_MIN_SPAN": None,
        "RETRIGGER": None, "RETRIGGER_RAW": False,
        "LISTENER_RESCUE": False, "LISTENER_CACHE": None, "LISTENER_LO": 0.4, "LISTENER_TH": 0.0, "LISTENER_BEATS_TH": None,
        "LISTENER_RULE": None, "LISTENER_VCACHE": None,
        "LISTENER_NEW_TYPE_ONCE": False, "LISTENER_LOCAL_WINNER": False, "LISTENER_SCENE_FIT": False,
        "LISTENER_SHADOW": False, "LISTENER_SHADOW_S": 0.2, "LISTENER_EDGE": False, "LISTENER_EDGE_S": 0.3,
        "LISTENER_CONFIRMED_MIRROR": False, "LISTENER_DASM_VOTE": False, "LISTENER_DASM_DIR": None,
        "LISTENER_DASM_BAR": 0.575, "LISTENER_DASM_PAD": 0.5,
        "LABEL_FILTER": "depictable"}
RT = (1.5, 0.4, 0.175)
R1_ = {"TWIN_MAX": True}
R6_ = {"RETRIGGER": RT, "RETRIGGER_RAW": True}
ARMS = {
    "B0r": {},
    "R13-1": dict(R1_),
    "R13-2b08": {"MIRROR_VETO": 0.8},
    "R13-2b07": {"MIRROR_VETO": 0.7},
    "R13-5": {"IMPULSE_MIN_SPAN": 0.2},
    "R13-6": dict(R6_),
    "R13-6ev": {"RETRIGGER": RT},                      # diagnostic: the evidence retrigger alone (no raw-label split)
    "STACK12b08": {**R1_, "MIRROR_VETO": 0.8},
    "STACK12b07": {**R1_, "MIRROR_VETO": 0.7},
    "STACK1256b08": {**R1_, "MIRROR_VETO": 0.8, "IMPULSE_MIN_SPAN": 0.2, **R6_},
    "STACK1256b07": {**R1_, "MIRROR_VETO": 0.7, "IMPULSE_MIN_SPAN": 0.2, **R6_},
}
# R13-3 (coordinator, 2026-09-29): listener rescue, LO x TH grid, each alone and + R13-1; + the P3 BEATs band (score > 3)
for _lo in (0.4, 0.5):
    for _th in (0, 2, 3):
        _n = f"L{int(_lo * 10):02d}t{_th}"
        _c = {"LISTENER_RESCUE": True, "LISTENER_CACHE": str(LISTENER), "LISTENER_LO": _lo, "LISTENER_TH": float(_th)}
        ARMS[_n] = dict(_c)
        ARMS[_n + "+1"] = {**_c, **R1_}
        ARMS[_n + "+P3"] = {**_c, "LISTENER_BEATS_TH": 3.0}
        ARMS[_n + "+1+P3"] = {**_c, **R1_, "LISTENER_BEATS_TH": 3.0}
# amendment A (2026-09-29): the rescue decided by one stricter listener rule (variants cache), LO 0.5; each + R13-1
VCACHE = _ROOT / "benchmark" / "gold" / "dev_listener_v.json"
for _r in ("V1", "V2", "V3", "V4", "V12"):
    _c = {"LISTENER_RESCUE": True, "LISTENER_RULE": _r, "LISTENER_VCACHE": str(VCACHE), "LISTENER_LO": 0.5}
    ARMS[f"LR-{_r}"] = dict(_c)
    ARMS[f"LR-{_r}+1"] = {**_c, **R1_}
# Round 14 (2026-09-29): precision filters on rescued spans, on the two amendment-A bases
for _b in ("LR-V4+1", "LR-V12+1"):
    for _fn, _f in (("F1", {"LISTENER_NEW_TYPE_ONCE": True}), ("F4", {"LISTENER_LOCAL_WINNER": True}),
                    ("F1F4", {"LISTENER_NEW_TYPE_ONCE": True, "LISTENER_LOCAL_WINNER": True}),
                    ("F1F4F3", {"LISTENER_NEW_TYPE_ONCE": True, "LISTENER_LOCAL_WINNER": True, "LISTENER_SCENE_FIT": True})):
        ARMS[f"{_b}+{_fn}"] = {**ARMS[_b], **_f}
R14 = [a for a in ARMS if a.startswith("LR-V4+1+") or a.startswith("LR-V12+1+")]
for _a in list(R14):                                  # addendum: F5 / F6 on the best round-14 arm (picked in the job)
    for _fn, _f in (("F5", {"LISTENER_SHADOW": True}), ("F6", {"LISTENER_EDGE": True}),
                    ("F5F6", {"LISTENER_SHADOW": True, "LISTENER_EDGE": True})):
        ARMS[f"{_a}+{_fn}"] = {**ARMS[_a], **_f}
R14b = [a for a in ARMS if a.startswith(("LR-V4+1+", "LR-V12+1+"))]
_F7 = {"MIRROR_VETO": 0.7, "LISTENER_CONFIRMED_MIRROR": True, "LISTENER_CACHE": str(LISTENER)}
_F8 = {"LISTENER_DASM_VOTE": True, "LISTENER_DASM_DIR": str(DCC.DASM_DIR), "LISTENER_DASM_BAR": DCC.F["DASM_G"]}
for _a in list(R14b):                                 # amendment B on the best arm after F1-F6 (picked in the job)
    for _fn, _f in (("F7", _F7), ("F8", _F8), ("F7F8", {**_F7, **_F8})):
        ARMS[f"{_a}+{_fn}"] = {**ARMS[_a], **_f}
STAGE5_KEYS = ("RETRIGGER_RAW", "LISTENER_SCENE_FIT")      # arm flags read after stage 4


@contextlib.contextmanager
def flags(d):
    old = {k: getattr(config, k, None) for k in d}
    try:
        for k, v in d.items():
            setattr(config, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(config, k, v)


def arm_cfg(arm):
    return {**BASE, **ARMS[arm]}


# ============================================================================= R13-3 hook (listener; not run yet)
def load_listener():
    """benchmark/gold/dev_listener.json (written by the listener job): per-span scores, keyed (clip, family, start, end).
    Accepted shapes: {"spans": [{"clip", "label"|"family", "start", "end", "score"}...]} or a list of such rows."""
    if not LISTENER.exists():
        return None
    d = json.loads(LISTENER.read_text(encoding="utf-8"))
    rows = d.get("spans", d.get("rows", [])) if isinstance(d, dict) else d
    out = {}
    for r in rows:
        fam = canonical(r.get("family") or r.get("label"))
        out[(r["clip"], fam, round(float(r["start"]), 2), round(float(r["end"]), 2))] = float(r["score"])
    return out


def listener_score(L, st, e, tol=0.05):
    if not L:
        return None
    k = (st, canonical(e.label), round(e.start, 2), round(e.end, 2))
    if k in L:
        return L[k]
    for (c, f, a, b), v in L.items():
        if c == st and f == k[1] and abs(a - e.start) <= tol and abs(b - e.end) <= tol:
            return v
    return None


# ============================================================================= PANNs (the veto's input)
def panns():
    from src.stage4_audio_event_detection import _infer
    _g, stems = DCC.dev_stems()
    PANNS_DIR.mkdir(parents=True, exist_ok=True)
    for st in stems:
        dst = PANNS_DIR / f"{st}.npz"
        if dst.exists():
            continue
        fw, _t, labs = _infer(DCC.wav_of(st), "cuda")
        np.savez_compressed(dst, peak=fw.max(axis=0).astype(np.float32), labels=np.array(labs))
        print(f"[panns] {st} {fw.shape}", flush=True)


def panns_provider(st):
    """PANNs (framewise, times, labels) for the veto: the clip peak as one frame (the veto reads only the clip max)"""
    def f():
        z = np.load(PANNS_DIR / f"{st}.npz")
        return z["peak"][None, :].astype(np.float32), np.array([0.0]), [str(x) for x in z["labels"]]
    return f


def trace_provider(tr):
    """offline only: a PANNs stand-in that keeps exactly the FlexSED-only families the scored run's veto kept"""
    def f():
        sig = lambda n: {(x["label"], x["start"], x["end"]) for x in tr if x["step"] == n}
        only = sig("union") & sig("flexsed_raw")                 # FlexSED-only spans (a twin never enters union)
        kept = {canonical(l) for (l, a, b) in only & sig("veto")}
        dropped = {canonical(l) for (l, a, b) in only - sig("veto")}
        assert not kept & dropped, (kept & dropped)
        labs = sorted(kept | dropped | {"_"})
        pk = np.array([[1.0 if l in kept else 0.0 for l in labs]], np.float32)
        return pk, np.array([0.0]), labs
    return f


# ============================================================================= stage 4 (the pipeline's code)
def build(st, sysn, arm, C, tr, offline=False):
    """one arm's stage-4 spans on one clip, through fuse_flexsed; refinement reused from the scored trace when the
    (label, start, end) equals a trace veto-step span, else marked live"""
    step = lambda n: [x for x in tr if x["step"] == n]
    Bfr, Ffr = C["beats"], C["flex"]
    cfg = arm_cfg(arm)
    info = {}
    with flags(cfg):
        TRACE.clear()
        events = _extract_events(Bfr[0], Bfr[1], Bfr[2], config.AED_THRESHOLD, None, config.AED_MIN_DUR,
                                 low=config.AED_THRESHOLD * float(config.AED_HYSTERESIS))
        TRACE.extend({"step": "extract", "label": e.label, "start": round(e.start, 3), "end": round(e.end, 3),
                      "conf": round(e.confidence, 3)} for e in events)
        prov = trace_provider(tr) if offline else panns_provider(st)
        lp1 = (listener_p1_lookup(config.LISTENER_VCACHE, config.LISTENER_CACHE, st)
               if config.LISTENER_CONFIRMED_MIRROR else None)
        lis = None
        if config.LISTENER_RESCUE:
            lis = (listener_from_vcache(config.LISTENER_VCACHE, st) if config.LISTENER_RULE
                   else listener_from_cache(config.LISTENER_CACHE, st))
        events, flex_ids, ffw = fuse_flexsed(events, Bfr[0], Bfr[1], Bfr[2], Ffr[0], Ffr[1], Ffr[2],
                                             config.AED_MIN_DUR, backend="BEATs", panns=prov, listener=lis,
                                             listener_p1=lp1)
        if lp1 is not None:
            info["f7"] = list(LISTENER_STATS.get("f7", []))
        if lis is not None:
            info["listener"] = json.loads(json.dumps(LISTENER_STATS))
        mine = {n: [(x["label"], x["start"], x["end"]) for x in TRACE if x["step"] == n]
                for n in ("extract", "flexsed_raw", "union")}
        mine["veto"] = [(e.label, e.start, e.end) for e in events]
        info["mirror_dropped"] = [[x["label"], x["start"], x["end"]] for x in TRACE if x["step"] == "mirror_veto"]
        if arm == "B0r":
            info["d0"] = {n: DCC.same_list(mine[n], [DCC.sp(x) for x in step(n)]) for n in mine}
            info["d0"]["pass"] = all(info["d0"].values())
        veto_tr = [DCC.sp(x) for x in step("veto")]
        refine_tr = step("refine")
        assert len(refine_tr) == len(veto_tr), st
        rows, evs = [], []
        for e in events:
            o = "flex" if id(e) in flex_ids else "tagger"
            r = {"label": e.label, "start": float(e.start), "end": float(e.end), "conf": float(e.confidence), "origin": o,
                 "rescued": bool(getattr(e, "rescued", False)),
                 "pre_start": float(e.start)}
            if o == "flex":
                r["refine"] = "none (frame-level)"
            else:
                m = [i for i, v in enumerate(veto_tr) if DCC.close(v, (e.label, e.start, e.end))]
                if m:
                    r["start"] = float(refine_tr[m[0]]["start"]); r["refine"] = "trace"
                else:
                    r["refine"] = "live"
            rows.append(r)
        info["ffw"] = ffw
    return rows, info


def filter_rows(rows, arm, C, ffw, res, k, st):
    """round 14: the pipeline's filter_rescued on the refined rows (as detect_events, after refinement)"""
    cfg = arm_cfg(arm)
    if not cfg.get("LISTENER_RESCUE"):
        return rows
    with flags(cfg):
        evs = [AudioEvent(r["label"], r["start"], r["end"], r["conf"], rescued=r.get("rescued", False)) for r in rows]
        dasm = None
        if cfg.get("LISTENER_DASM_VOTE") and cfg.get("LISTENER_DASM_DIR"):
            f = Path(cfg["LISTENER_DASM_DIR"]) / f"{st}.npz"
            dasm = DCC.load_fr(f) if f.exists() else None
        kept, dropped = filter_rescued(evs, ffw, C["flex"][1], C["flex"][2], dasm=dasm)
    keep = {id(e) for e in kept}
    if any(dropped.values()):
        res.setdefault("r14_dropped", {})[f"{k}|{st}"] = dropped
    return [r for r, e in zip(rows, evs) if id(e) in keep]


def add_breaks(rows, arm, C, ffw):
    cfg = arm_cfg(arm)
    if not cfg.get("RETRIGGER"):
        return
    with flags(cfg):
        evs = [AudioEvent(r["label"], r["start"], r["end"], r["conf"]) for r in rows]
        attach_breaks(evs, C["beats"][0], C["beats"][1], C["beats"][2], ffw, C["flex"][1], C["flex"][2])
    for r, e in zip(rows, evs):
        r["breaks"] = [list(b) for b in e.breaks]


def stage4(arms, offline=False):
    if offline:
        loc = Path(os.environ["R13_LOCAL"])
        DCC.WORK = loc
        DCC.BEATS_DIR = loc / "j2_dev_beats"
    else:
        from src.stage4_audio_event_detection import _refine_onsets_cam
        from src.stage4_audio_event_detection import beats_infer as B
        B._MODEL = None
    gold, stems = DCC.dev_stems()
    ref = json.loads((DCC.DC / "stage4.json").read_text(encoding="utf-8")) if (DCC.DC / "stage4.json").exists() else None
    if offline and ref is None:
        ref = json.loads((_ROOT / "data" / "work" / "devcand" / "stage4.json").read_text(encoding="utf-8"))
    res = {} if offline or not STAGE4.exists() else json.loads(STAGE4.read_text(encoding="utf-8"))
    res.setdefault("arms", {}); res.setdefault("d0", {}); res.setdefault("conf_eq", {}); res.setdefault("mirror", {})
    res.setdefault("cfg", {}).update({a: {k: v for k, v in arm_cfg(a).items()} for a in arms})
    for st in stems:
        C = {"beats": DCC.load_fr(DCC.BEATS_DIR / f"{st}.npz"), "flex": DCC.load_fr(DCC.FLEX_DIR / f"{st}.npz")}
        for sysn in SYSTEMS:
            tr = json.loads((DCC.scored_dir(sysn) / st / "onset_trace.json").read_text(encoding="utf-8"))
            for arm in arms:
                k = f"{arm}|{sysn}"
                res["arms"].setdefault(k, {})
                if st in res["arms"][k]:
                    continue
                rows, info = build(st, sysn, arm, C, tr, offline)
                if arm == "B0r":
                    res["d0"][f"{sysn}|{st}"] = info["d0"]
                    if not info["d0"]["pass"]:
                        print(f"[D0 FAIL] {sysn} {st}: {info['d0']}", flush=True)
                    if ref is not None:                  # stricter: label/start/end/conf/origin/refined start vs devcand
                        sig = lambda rr: sorted((r["label"], round(r["pre_start"], 3), round(r["start"], 3), round(r["end"], 3),
                                                 round(r["conf"], 4), r["origin"]) for r in rr)
                        res["conf_eq"][f"{sysn}|{st}"] = sig(rows) == sig(ref["arms"][f"B0r|{sysn}"][st])
                if "f7" in info:
                    res.setdefault("f7", {})[f"{k}|{st}"] = info["f7"]
                if "listener" in info:
                    res.setdefault("listener", {})[f"{k}|{st}"] = info["listener"]
                if info["mirror_dropped"]:
                    res["mirror"][f"{k}|{st}"] = info["mirror_dropped"]
                live = [r for r in rows if r["refine"] == "live"]
                if live and not offline:
                    with flags(arm_cfg(arm)):
                        ev = [AudioEvent(r["label"], r["pre_start"], r["end"], r["conf"]) for r in live]
                        out = _refine_onsets_cam(DCC.wav_of(st), ev, C["beats"][2], "cuda", skip_ids=set())
                    for r, e in zip(live, out):
                        r["start"] = float(e.start)
                rows = filter_rows(rows, arm, C, info["ffw"], res, k, st)
                add_breaks(rows, arm, C, info["ffw"])
                res["arms"][k][st] = rows
        if not offline:
            DCC.dump(STAGE4, res)
        print(f"[stage4] {st} done", flush=True)
    d0f = [k for k, v in res["d0"].items() if not v["pass"]]
    ce = [k for k, v in res["conf_eq"].items() if not v]
    print(f"[D0] {len(res['d0']) - len(d0f)} / {len(res['d0'])} pass; fails {d0f}", flush=True)
    print(f"[conf-eq vs devcand B0r] {len(res['conf_eq']) - len(ce)} / {len(res['conf_eq'])} equal; differ {ce}", flush=True)
    live = {k: sum(1 for rr in v.values() for r in rr if r["refine"] == "live") for k, v in res["arms"].items()}
    print(f"[live refinements] {live}", flush=True)
    if offline:
        DCC.dump(Path(os.environ["R13_LOCAL"]) / "stage4_offline.json", res)
    else:
        DCC.dump(STAGE4, res)


# ============================================================================= stage 5 (DCC's path, events with breaks)
def stage5(arms):
    from benchmark.run_protocol import configure
    from src.stage5_cross_modal_analysis import plan_augmentations, reason
    from src.types import SceneContext, SpeechSegment
    changed = config.use_scored()
    config.DEVICE = "cuda"; config.TRANSCRIBE = True
    print("[cfg] use_scored:", {k: v[1] for k, v in changed.items()}, flush=True)
    for k, v in (("FLEXSED_BAR", 0.8), ("FLEXSED_VETO", 0.3), ("PANNS_VETO", 0.05), ("ONSET_MONOTONE", True),
                 ("MAX_SPAN", None), ("VLM_MODEL", "Qwen/Qwen3.8-27B"), ("VLM_THINKING", False),
                 ("LABEL_FILTER", "depictable"), ("KINSHIP_DIRECTED", False), ("PICTURE_MIN_CONF", None),
                 ("BEATS_SELF_VETO", 0.0), ("RETRIGGER_RAW", False), ("LISTENER_SCENE_FIT", False)):
        assert getattr(config, k, None) == v, (k, getattr(config, k, None), v)
    if not MEMO.exists() and (DCC.DC / "ask_memo.json").exists():
        MEMO.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(DCC.DC / "ask_memo.json", MEMO)      # a copy: devcand's memo is not written
    DCC.MEMO = MEMO
    _g, stems = DCC.dev_stems()
    s4 = json.loads(STAGE4.read_text(encoding="utf-8"))
    R = DCC.Reuse()
    base = {k: getattr(config, k) for k in ("DISPLAY_THRESHOLD", "AUGMENT_THRESHOLD", "AED_THRESHOLD") + STAGE5_KEYS}
    for sysn in ("blind_a2i", "proposed"):
        configure(sysn)
        for arm in arms:
            for k, v in base.items():
                setattr(config, k, v)
            for k in STAGE5_KEYS:
                setattr(config, k, arm_cfg(arm)[k])
            root = R13 / f"{arm}_{sysn}"
            logp = root / "_stage5_log.json"
            log = json.loads(logp.read_text(encoding="utf-8")) if logp.exists() else {}
            for st in stems:
                d = root / st
                if (d / "augmentations.json").exists():
                    continue
                src = DCC.scored_dir(sysn) / st
                media = json.loads((src / "media.json").read_text(encoding="utf-8"))
                scene = SceneContext(**json.loads((src / "scene.json").read_text(encoding="utf-8")))
                segments = [SpeechSegment(**x) for x in json.loads((src / "segments.json").read_text(encoding="utf-8"))]
                events = [AudioEvent(r["label"], r["start"], r["end"], r["conf"],
                                     breaks=[tuple(b) for b in r.get("breaks", [])], rescued=bool(r.get("rescued", False)))
                          for r in s4["arms"][f"{arm}|{sysn}"][st]]
                gv = src / "gate_votes.json"
                R.votes = json.loads(gv.read_text(encoding="utf-8")) if gv.exists() else []
                R.stats = {}
                print(f"[stage5] {arm} {sysn} {st}: {len(events)} events", flush=True)
                specs = plan_augmentations(scene, segments, events, threshold=config.AED_THRESHOLD,
                                           gate_enabled=config.GATE_ENABLED, display_threshold=config.DISPLAY_THRESHOLD,
                                           augment_threshold=config.AUGMENT_THRESHOLD)
                votes = []
                if getattr(config, "DEPICTION_REASONING", True) and any(s.augment for s in specs):
                    reason.decide_subjects(Path(media["video_path"]), specs, segments=segments, model=config.VLM_MODEL,
                                           device=config.DEVICE, display_threshold=config.DISPLAY_THRESHOLD)
                    votes = list(reason.VOTE_LOG)
                for s in specs:
                    if s.augment:
                        s.image_path = DCC.PLACEHOLDER; s.backend = "placeholder"
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy(src / "media.json", d / "media.json")
                (d / "gate_votes.json").write_text(json.dumps(votes, indent=1), encoding="utf-8")
                (d / "augmentations.json").write_text(json.dumps([s.to_dict() for s in specs], indent=1), encoding="utf-8")
                log[st] = R.stats
                DCC.dump(logp, log)
                R.save()
    for k, v in base.items():
        setattr(config, k, v)
    R.save()
    print("[stage5] done", flush=True)


# ============================================================================= scoring
def pic_type(gold_clip, pics, pic):
    """what one picture counts as: the change in score_clip's counts when it is added (the miss table's rule)"""
    a = S.score_clip(gold_clip, pics)
    rest = list(pics); rest.remove(pic)
    b = S.score_clip(gold_clip, rest)
    for k in ("visible", "cross", "phantom", "hit", "dup", "dontcare"):
        if a[k] - b[k] > 0:
            return k
    return "other"


def score():
    gold, stems = DCC.dev_stems()
    s4 = json.loads(STAGE4.read_text(encoding="utf-8"))
    dc4 = json.loads((DCC.DC / "stage4.json").read_text(encoding="utf-8"))
    arms = [a for a in ARMS if all(DCC.complete(R13 / f"{a}_{s}", stems) for s in SYSTEMS)]
    res = {"plan": "docs/prereg_round13_detector_push.md", "base": BASE, "arms": {a: ARMS[a] for a in arms},
           "d0": {}, "conf_eq": {}, "d5": {}, "rows": {}, "heard": {}, "delta_vs_B0r": {}, "delta_vs_B1": {},
           "eligible": {}, "needed_changes": {}, "picture_changes": {}, "stage5": {}, "mirror_dropped": s4.get("mirror", {}),
           "twin_blast": {}, "breaks": {}, "listener": {}}
    for k, v in s4.get("listener", {}).items():                  # per arm and system: rescues and cache coverage
        arm_sys = k.rsplit("|", 1)[0]
        agg = res["listener"].setdefault(arm_sys, {"a_added": 0, "b_kept": 0, "c_added": 0, "a_asked": 0, "a_missing": 0,
                                                   "b_asked": 0, "b_missing": 0, "c_asked": 0, "c_missing": 0,
                                                   "b_missing_list": [], "a_missing_list": []})
        for f in ("a_added", "b_kept", "c_added"):
            agg[f] += len(v[f])
        for f in ("a_asked", "a_missing", "b_asked", "b_missing", "c_asked", "c_missing"):
            agg[f] += v[f]
        agg["b_missing_list"] += [[k.rsplit("|", 1)[1]] + x for x in v.get("b_missing_list", [])]
        agg["a_missing_list"] += [[k.rsplit("|", 1)[1]] + x for x in v.get("a_missing_list", [])]
    d0 = s4["d0"]
    res["d0"] = {"pass": sum(v["pass"] for v in d0.values()), "of": len(d0), "fails": [k for k, v in d0.items() if not v["pass"]]}
    ce = s4.get("conf_eq", {})
    res["conf_eq"] = {"equal": sum(bool(v) for v in ce.values()), "of": len(ce), "differ": [k for k, v in ce.items() if not v]}
    for sysn in SYSTEMS:
        # R13-1 blast radius: twin-absorbed BEATs spans whose conf crosses the display bar
        b0 = s4["arms"][f"B0r|{sysn}"]
        if f"R13-1|{sysn}" in s4["arms"]:
            up = []
            for st in stems:
                for r0, r1 in zip(b0[st], s4["arms"][f"R13-1|{sysn}"][st]):
                    if r0["conf"] < 0.35 <= r1["conf"]:
                        up.append([st, r0["label"], round(r0["start"], 2), round(r0["conf"], 3), round(r1["conf"], 3)])
            res["twin_blast"][sysn] = up
        res["breaks"][sysn] = {a: sum(len(r.get("breaks", [])) > 0 for rr in s4["arms"][f"{a}|{sysn}"].values() for r in rr)
                               for a in arms}
        P = {"B0": {st: S.load_pictures(DCC.scored_dir(sysn), st, sysn) or [] for st in stems},
             "B1": {st: S.load_pictures(DCC.DC / f"B1_{sysn}", st, sysn) or [] for st in stems}}
        for a in arms:
            P[a] = {st: S.load_pictures(R13 / f"{a}_{sysn}", st, sysn) or [] for st in stems}
            lg = R13 / f"{a}_{sysn}" / "_stage5_log.json"
            st5 = json.loads(lg.read_text(encoding="utf-8")) if lg.exists() else {}
            tot = {}
            for v in st5.values():
                for k2 in ("gate_reused", "gate_live", "ask_memo", "ask_live"):
                    tot[k2] = tot.get(k2, 0) + v.get(k2, 0)
            res["stage5"][f"{a}|{sysn}"] = tot
        bad = [st for st in stems if DCC.pics_sig(P["B0r"][st]) != DCC.pics_sig(P["B0"][st])
               or DCC.spec_sig(R13 / f"B0r_{sysn}", st) != DCC.spec_sig(DCC.scored_dir(sysn), st)]
        res["d5"][sysn] = {"pass": len(stems) - len(bad), "of": len(stems), "differ": bad}
        rows = {n: [S.score_clip(gold[st], P[n][st]) for st in stems] for n in P}
        res["rows"][sysn] = {n: DCC.metrics(r) for n, r in rows.items()}
        heard = {}
        for n in arms:
            heard[n] = sum(S.score_clip(gold[st], DCC.heard_pics(s4["arms"][f"{n}|{sysn}"][st], 0.35))["hit"] for st in stems)
        heard["B1"] = sum(S.score_clip(gold[st], DCC.heard_pics(dc4["arms"][f"B1|{sysn}"][st], 0.35))["hit"] for st in stems)
        heard["B0"] = heard.get("B0r")
        res["heard"][sysn] = heard
        cost = {n: [DCC.clip_cost(r) for r in rr] for n, rr in rows.items()}
        res["delta_vs_B0r"][sysn] = {n: DCC.boot(np.subtract(cost[n], cost["B0r"])) for n in P if n != "B0r"}
        res.setdefault("halves", {})[sysn] = {}
        for h, idx in (("A", range(0, len(stems), 2)), ("B", range(1, len(stems), 2))):
            idx = list(idx)
            res["halves"][sysn][h] = {n: {**DCC.metrics([rows[n][i] for i in idx]),
                                          "delta_vs_B0r": DCC.boot(np.subtract([cost[n][i] for i in idx],
                                                                               [cost["B0r"][i] for i in idx]))}
                                      for n in P}
        res["delta_vs_B1"][sysn] = {n: DCC.boot(np.subtract(cost[n], cost["B1"])) for n in P if n != "B1"}
        R_ = res["rows"][sysn]
        res["eligible"][sysn] = {}
        for n in arms:
            if n == "B0r":
                continue
            x, y = R_[n], R_["B0r"]
            gain = x["hits"] - y["hits"]
            res["eligible"][sysn][n] = bool(gain >= 1 and x["wrong"] <= y["wrong"] + 2 * gain
                                            and x["viewer_cost"] < y["viewer_cost"] - 1e-12)
        res["needed_changes"][sysn], res["picture_changes"][sysn] = {}, {}
        for n in [a for a in arms if a != "B0r"] + ["B1"]:
            gained, lost = [], []
            for st in stems:
                h0 = DCC.needed_hit(gold[st], P["B0r"][st]); h1 = DCC.needed_hit(gold[st], P[n][st])
                for (g, a0), (_g2, a1) in zip(h0, h1):
                    if a1 and not a0:
                        gained.append([st, g["label"], g["start"]])
                    if a0 and not a1:
                        lost.append([st, g["label"], g["start"]])
            res["needed_changes"][sysn][n] = {"gained": gained, "lost": lost}
            add, rem = DCC.diff_pics(P["B0r"], P[n])
            res["picture_changes"][sysn][n] = {
                "appeared": [list(x) + [pic_type(gold[x[0]], P[n][x[0]], _find(P[n][x[0]], x))] for x in add],
                "disappeared": [list(x) + [pic_type(gold[x[0]], P["B0r"][x[0]], _find(P["B0r"][x[0]], x))] for x in rem]}
        print(f"[D5 {sysn}] {res['d5'][sysn]['pass']}/{len(stems)}; differ {bad}", flush=True)
        for n in ["B0", "B1"] + arms:
            x = R_[n]; dd = res["delta_vs_B0r"][sysn].get(n)
            print(f"DEV {sysn:9s} {n:13s} heard {heard.get(n)} hits {x['hits']}/{x['hits'] + x['misses']} wrong {x['wrong']} "
                  f"({x['visible']}/{x['cross']}/{x['phantom']}) cost {x['viewer_cost']:.3f}"
                  + (f"  d vs B0r {dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}]" if dd else "")
                  + (f"  eligible {res['eligible'][sysn].get(n)}" if n in res['eligible'][sysn] else ""), flush=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"-> {OUT}", flush=True)


def _find(pics, x):
    st, lab, a, b = x
    for p in pics:
        if p[0] == lab and round(p[1], 2) == a and round(p[2], 2) == b:
            return p
    raise KeyError(x)


def best14(which="F5F6"):
    """the best round-14 arm so far (lowest DEV cost at beta 2, ties fewer wrong) and its next extra arms: which = F5F6
    (addendum, among the 8 filter arms) or F7F8 (amendment B, among all round-14 arms incl. F5/F6)"""
    r = json.loads(OUT.read_text(encoding="utf-8"))["rows"]["proposed"]
    pool = R14 if which == "F5F6" else R14b
    c = [a for a in pool if a in r]
    b = min(c, key=lambda a: (r[a]["viewer_cost"], r[a]["wrong"]))
    ext = ("F5", "F6", "F5F6") if which == "F5F6" else ("F7", "F8", "F7F8")
    print(" ".join(f"{b}+{x}" for x in ext))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("panns", "stage4", "stage5", "score", "best14"))
    ap.add_argument("--arms", nargs="+", default=[a for a in ARMS if not a.endswith(("+F5", "+F6", "+F5F6", "+F7", "+F8",
                                                                                      "+F7F8"))])
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--which", default="F5F6")
    a = ap.parse_args()
    {"panns": panns, "stage4": lambda: stage4(a.arms, a.offline), "stage5": lambda: stage5(a.arms),
     "score": score, "best14": lambda: best14(a.which)}[a.step]()


if __name__ == "__main__":
    main()
