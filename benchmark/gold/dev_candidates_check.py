"""Replay helpers for the scoring harness: the scored render's folders, DASM scores (dasm()), stage-4/5 replay, metrics,
boot, holm. The earlier detector-candidate arms (EATR, I4, R6, R7) need modules in release v1.2.0.

    python benchmark/gold/dev_candidates_check.py dasm     # GPU: DASM frame scores, 10-s pieces
    python benchmark/gold/dev_candidates_check.py stage4   # GPU (occlusion onsets of new spans only): gate D0, spans of every arm
    python benchmark/gold/dev_candidates_check.py stage5   # GPU: stage 5 per arm and system, gate answers reused
    python benchmark/gold/dev_candidates_check.py score    # CPU: gate D5, the table, bootstrap, Holm, ship rule

(design record: release v1.2.0)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from src.labels import canonical, is_descendant
from src.stage4_audio_event_detection import _extract_events
from src.types import AudioEvent

WORK = _ROOT / "data" / "work"
TAG = "dev_monocap_v31"
SYSTEMS = ("proposed", "blind_a2i")
DC = WORK / "devcand"
BEATS_DIR, FLEX_DIR = WORK / "j2_dev_beats", WORK / "flexsed_cache"
EAT_DIR, DASM_DIR, VLM_JSON = DC / "eat_cache", DC / "dasm_cache", DC / "i6_vlm.json"
STAGE4, MEMO = DC / "stage4.json", DC / "ask_memo.json"
OUT = _ROOT / "benchmark" / "gold" / "dev_candidates_check.json"
PLACEHOLDER = str(_ROOT / "README.md")          # any existing file: a shown picture, as a placeholder counts in the scored run
TOL = 0.01
ARMS = ["B0r", "B1", "EATR", "D1", "I4", "I6"]   # arms that go through stage 5 (job 31330563)
EXTRA = ["R1", "R6", "R7"]                       # amendment 1: round 10's rescue cells (job_devcand_extra.sh, release v1.2.0)
CANDS = ["EATR", "D1", "I4", "I6", "I7"] + EXTRA
RELEASE_ONLY = ("EATR", "I4", "R6", "R7")       # arms that import benchmark/detector_round5/8.py (release v1.2.0)
PARA_DIR, WAV16 = DC / "para", DC / "wav16"
NAMES = {"B0": "B0 scored render (PANNs veto)", "B0r": "B0 repro (this code)", "B1": "B1 shipped stack (self-veto)",
         "EATR": "EAT-R", "D1": "DASM D1", "I4": "I4 parent emission", "I6": "I6 VLM scene prior", "I7": "I7 contrast veto",
         "R1": "R1 DASM-agreed band", "R6": "R6 I4 confirmed", "R7": "R7 I6 confirmed"}
FLANK, I7_MARGIN = 3.0, 0.2


def frozen():
    """the candidates' values. Originally read from the rounds' own result files (benchmark/detector_round5.json fit EAT-R,
    detector_round6.json bars, detector_round10.py; release v1.2.0); this is that dict, printed and pasted unchanged."""
    return {"BAND": 0.4, "R1_BAR": 0.35937499999999994, "R1_PAD": 0.5, "R2_ADMIT": 0.5,
            "AED": 0.175, "DISP": 0.35, "FBAR": 0.8, "FVETO": 0.3, "B_SELF": 0.1218,
            "EAT_AED": 0.2575, "EAT_DISP": 0.515, "EAT_B": 0.1658935546875,
            "DASM_G": 0.575, "DASM_V": 0.08392333984375, "I6_BAR": 0.5, "I7_MARGIN": 0.2, "FLANK": 3.0}


F = frozen()
key = lambda e: canonical(e.label)


def same(a, b):
    return a == b or canonical(a) == canonical(b) or is_descendant(a, b) or is_descendant(b, a)


def dev_stems():
    gold = S.load_gold([_ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"])
    return gold, sorted(S.subsets_of(gold)["dev"])


def scored_dir(sysn):
    return WORK / f"protocol_{sysn}_{TAG}"


def wav_of(st):
    return scored_dir("proposed") / st / "audio.wav"


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(p)


# ============================================================================= caches (GPU)
def dasm():
    """round 6's scorer and queries; 10-s pieces, the last piece = the clip's final 10 s (no zero-padded piece)"""
    import librosa
    import torch
    from src.stage4_audio_event_detection import dasm_infer as R6     # round 6's DASM block (detector_round6.py, v1.2.0)
    _g, stems = dev_stems()
    DASM_DIR.mkdir(parents=True, exist_ok=True)
    wavs = {st: str(wav_of(st).resolve()) for st in stems}
    out = DASM_DIR.resolve()
    q = torch.load(str(R6.QFILE))
    voc = R6.vocab()
    assert q["vocab"] == voc and q["queries"] == [R6.dasm_text(x) for x in voc], "query file differs from the vocab"
    E = q["embeds"]
    d = R6._Dasm("cuda")                       # takes this project off sys.path (foreign repo imports): nothing of ours after
    print(f"[dasm] load strict=True {d.load_info}", flush=True)
    n, SR = int(round(R6.CLIP_S * R6.SR)), R6.SR
    log = {"load": d.load_info, "clips": {}}
    for st in stems:
        dst = out / f"{st}.npz"
        if dst.exists():
            continue
        a, _ = librosa.load(wavs[st], sr=SR, mono=True)
        L = len(a)
        fws, ts = [], []
        if L <= n:
            fw, _at = d.score(a, E); fws.append(fw); ts.append(np.arange(fw.shape[1]) / R6.FPS)
        else:
            k = 0
            while (k + 1) * n <= L:
                fw, _at = d.score(a[k * n:(k + 1) * n], E)
                assert fw.shape[1] == 500
                fws.append(fw); ts.append(k * R6.CLIP_S + np.arange(500) / R6.FPS); k += 1
            if k * n < L:                                   # remainder: score the final 10 s, keep only its new frames
                s0 = (L - n) / SR
                fw, _at = d.score(a[L - n:], E)
                t = s0 + np.arange(fw.shape[1]) / R6.FPS
                keep = t >= k * R6.CLIP_S - 1e-9
                fws.append(fw[:, keep]); ts.append(t[keep])
        fw = np.concatenate(fws, axis=1).T                   # [T, 215]
        t = np.concatenate(ts)
        np.savez_compressed(dst, fw=fw.astype(np.float32), times=t.astype(np.float64), labels=np.array(voc))
        log["clips"][st] = {"seconds": L / SR, "frames": int(fw.shape[0])}
        print(f"[dasm] {st} {L / SR:.2f} s -> {fw.shape}", flush=True)
    (out / "_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")


# ============================================================================= stage 4
def load_fr(p):
    z = np.load(p)
    fw = z["fw"].astype(np.float32)
    labs = [str(x) for x in z["labels"]]
    if "times" in z:
        return fw, z["times"].astype(np.float64), labs
    fw = fw.T
    return fw, np.arange(fw.shape[0], dtype=np.float64) / float(z["fps"]), labs


def ext(fr, bar):
    return _extract_events(fr[0], fr[1], fr[2], bar, None, config.AED_MIN_DUR,
                           low=bar * float(getattr(config, "AED_HYSTERESIS", 1.0)))


def clip_peak(fr):
    out = {}
    for i, lab in enumerate(fr[2]):
        k = canonical(lab); out[k] = max(out.get(k, 0.0), float(fr[0][:, i].max()))
    return out


def sp(x):
    return (x["label"], float(x["start"]), float(x["end"]))


def close(a, b):
    return a[0] == b[0] and abs(a[1] - b[1]) <= TOL and abs(a[2] - b[2]) <= TOL


def same_list(a, b):
    a, b = sorted(a), sorted(b)
    return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))


def twin(events, new):
    """stage 4's union ("absorb", UNION_START min): a new span with a same-family tagger span within 1 s pulls its start
    earlier; otherwise it is fresh (only-this-detector)"""
    fresh = []
    for e in new:
        tw = [b for b in events if key(b) == key(e) and b.start - 1.0 <= e.end and e.start - 1.0 <= b.end]
        if tw:
            for b in tw:
                b.start = min(b.start, e.start)
        else:
            fresh.append(e)
    return fresh


def flex_i6(Ffr, listed):
    fw, ts, labs = Ffr
    out = []
    for j, lab in enumerate(labs):
        bar = F["I6_BAR"] if lab in listed else F["FBAR"]
        out += ext((fw[:, [j]], ts, [lab]), bar)
    out.sort(key=lambda e: (e.start, -e.confidence))
    return out


def contrast(fr, label, a, b):
    fw, ts, labs = fr
    cols = [i for i, l in enumerate(labs) if canonical(l) == canonical(label)]
    if not cols:
        return None
    s = fw[:, cols].max(axis=1)
    ins = (ts >= a) & (ts < b)
    fl = ((ts >= a - FLANK) & (ts < a)) | ((ts >= b) & (ts < b + FLANK))
    if not fl.any() or not ins.any():
        return None
    return float(s[ins].mean() - s[fl].mean())


def _win(fr, t0, t1):
    ts = fr[1]
    m = (ts >= t0) & (ts <= t1)
    if not m.any():
        m = np.zeros(len(ts), bool); m[int(np.argmin(np.abs(ts - 0.5 * (t0 + t1))))] = True
    return m


def _cols(labs, label, match):
    if match:                                        # family matching: detector_round8 (release v1.2.0)
        from benchmark import detector_round8 as M
        return M.fam_cols(labs, label)
    return [i for i, l in enumerate(labs) if canonical(l) == canonical(label)]


def r1(C, e, match=False):
    """round 10's R1: DASM same family >= 0.359375 at some frame in [start - 0.5, end + 0.5] (nearest frame if none)"""
    fr = C["dasm"]
    cols = _cols(fr[2], e.label, match)
    if not cols:
        return False
    m = _win(fr, e.start - F["R1_PAD"], e.end + F["R1_PAD"])
    return float(fr[0][m][:, cols].max()) >= F["R1_BAR"]


def r2(C, e, match=False):
    """round 10's R2: both FlexSED paraphrases >= 0.5 in the span (some matching family)"""
    from benchmark import detector_round8 as M
    fr = C["para"]
    fams = [l.split("||")[0] for l in fr[2][::2]]
    for f in [f for f in fams if (M.same(f, e.label) if match else canonical(f) == canonical(e.label))]:
        c1, c2 = fr[2].index(f"{f}||p1"), fr[2].index(f"{f}||p2")
        if M.peak_in(fr, [c1], e.start, e.end) >= F["R2_ADMIT"] and M.peak_in(fr, [c2], e.start, e.end) >= F["R2_ADMIT"]:
            return True
    return False


def band_cands(C):
    """round 10's band candidates: FlexSED 0.4 spans, peak < 0.8, no BEATs 0.175 twin within 1 s, BEATs self-veto"""
    bev = ext(C["beats"], F["AED"])
    bp = clip_peak(C["beats"])
    out = []
    for e in ext(C["flex"], F["BAND"]):
        if e.confidence >= F["FBAR"]:
            continue
        if any(key(y) == key(e) and y.start - 1.0 <= e.end and e.start - 1.0 <= y.end for y in bev):
            continue
        if bp.get(key(e), 1.0) < F["B_SELF"]:
            continue
        out.append(e)
    return out


def build(st, sysn, arm, C, vlm_ans, d0, stat=None):
    """one arm's stage-4 output on one clip: events after the vetoes (pipeline order), tagged; refinement is done later"""
    tr = json.loads((scored_dir(sysn) / st / "onset_trace.json").read_text(encoding="utf-8"))
    step = lambda n: [x for x in tr if x["step"] == n]
    Bfr, Ffr = C["beats"], C["flex"]
    tagfr = C["eat"] if arm == "EATR" else Bfr
    events = ext(tagfr, F["EAT_AED"] if arm == "EATR" else F["AED"])
    if arm == "B0r":                                       # gate D0
        d0["extract"] = same_list([(e.label, e.start, e.end) for e in events], [sp(x) for x in step("extract")])
    origin = {id(e): "tagger" for e in events}
    if arm in ("I4", "R6"):
        from benchmark import detector_round8 as R8
        par = R8.parent_spans(Bfr)
        if arm == "R6":                                  # confirmed by R1 OR R2 (E._same columns), else removed
            ok = [p for p in par if r1(C, p, True) or r2(C, p, True)]
            if stat is not None:
                stat["considered"] += len(par); stat["admitted"] += len(ok)
            par = ok
        origin.update({id(e): "parent" for e in par})
        events = events + par
    if arm == "D1":
        fev = ext(C["dasm"], F["DASM_G"])
    elif arm in ("I6", "R7"):
        fev = flex_i6(Ffr, set(vlm_ans.get(st, {}).get("families", [])))
        if arm == "R7":                                  # lowered-bar spans confirmed by R1 OR R2 (own columns), else removed
            low = [e for e in fev if e.confidence < F["FBAR"]]
            okid = {id(e) for e in low if r1(C, e) or r2(C, e)}
            if stat is not None:
                stat["considered"] += len(low); stat["admitted"] += len(okid)
            fev = [e for e in fev if e.confidence >= F["FBAR"] or id(e) in okid]
    else:
        fev = ext(Ffr, F["FBAR"])
    if arm == "B0r":
        d0["flexsed_raw"] = same_list([(e.label, e.start, e.end) for e in fev], [sp(x) for x in step("flexsed_raw")])
    fresh = twin(events, fev)
    only = {id(e) for e in fresh}
    origin.update({id(e): ("dasm" if arm == "D1" else "flex") for e in fresh})
    events = events + fresh
    if arm == "B0r":
        d0["union"] = same_list([(e.label, e.start, e.end) for e in events], [sp(x) for x in step("union")])
    vfr, vbar = (C["dasm"], F["DASM_V"]) if arm == "D1" else (Ffr, F["FVETO"])
    pk = clip_peak(vfr)
    events = [e for e in events if pk.get(key(e), 1.0) >= vbar]
    veto_tr = [sp(x) for x in step("veto")]
    if arm == "B0r":                                        # PANNs veto: which second-detector-only spans the run kept
        events = [e for e in events if id(e) not in only or any(close((e.label, e.start, e.end), v) for v in veto_tr)]
        d0["veto"] = same_list([(e.label, e.start, e.end) for e in events], veto_tr)
    else:
        spk = clip_peak(tagfr)
        b = F["EAT_B"] if arm == "EATR" else F["B_SELF"]
        events = [e for e in events if id(e) not in only or spk.get(key(e), 1.0) >= b]
    if arm == "R1":                                      # admitted band candidates are added as FlexSED-only spans
        cs = band_cands(C)
        ok = [e for e in cs if r1(C, e)]
        if stat is not None:
            stat["considered"] += len(cs); stat["admitted"] += len(ok)
        origin.update({id(e): "band" for e in ok})
        events = events + ok
    refine_tr = [x for x in step("refine")]
    assert len(refine_tr) == len(veto_tr), st
    rows = []
    for e in events:
        o = origin[id(e)]
        r = {"label": e.label, "start": float(e.start), "end": float(e.end), "conf": float(e.confidence), "origin": o,
             "pre_start": float(e.start)}
        if o in ("flex", "dasm", "band"):
            r["refine"] = "none (frame-level)"
            if o == "flex":
                c = contrast(Ffr, e.label, e.start, e.end)
                r["i7_contrast"] = c
                r["i7_drop"] = bool(c is not None and c < I7_MARGIN)
        else:
            m = [i for i, v in enumerate(veto_tr) if close(v, (e.label, e.start, e.end))] if arm != "EATR" else []
            if m:
                r["start"] = float(refine_tr[m[0]]["start"]); r["refine"] = "trace"
            else:
                r["refine"] = "live"
        rows.append(r)
    return rows


def stage4(arms=None):
    arms = arms or ARMS
    old = [a for a in arms if a in RELEASE_ONLY]
    if old:
        raise SystemExit(f"[stage4] arms {old} need benchmark/detector_round5.py / detector_round8.py, which are in release "
                         f"v1.2.0 (not in this repository); run them from that release or pass --arms without them")
    from src.stage4_audio_event_detection import _refine_onsets_cam
    from src.stage4_audio_event_detection import beats_infer as B
    config.ONSET_MONOTONE = True                            # the scored run's MONO=1
    gold, stems = dev_stems()
    vlm_ans = json.loads(VLM_JSON.read_text(encoding="utf-8"))
    res = json.loads(STAGE4.read_text(encoding="utf-8")) if STAGE4.exists() else {}
    res["frozen"] = F
    res.setdefault("d0", {}); res.setdefault("arms", {}); res.setdefault("filter", {})
    for sysn in SYSTEMS:
        for arm in arms:
            res["arms"].setdefault(f"{arm}|{sysn}", {})
    todo_eat = []
    B._MODEL = None
    for st in stems:
        C = {"beats": load_fr(BEATS_DIR / f"{st}.npz"), "flex": load_fr(FLEX_DIR / f"{st}.npz"),
             "eat": load_fr(EAT_DIR / f"{st}.npz"), "dasm": load_fr(DASM_DIR / f"{st}.npz")}
        if any(a in ("R6", "R7") for a in arms):
            C["para"] = load_fr(PARA_DIR / f"{st}.npz")
        for sysn in SYSTEMS:
            for arm in arms:
                if st in res["arms"][f"{arm}|{sysn}"]:
                    continue
                d0 = {}
                stat = res["filter"].setdefault(f"{arm}|{sysn}", {"considered": 0, "admitted": 0})
                rows = build(st, sysn, arm, C, vlm_ans, d0, stat)
                if arm == "B0r":
                    d0["pass"] = all(d0.values())
                    res["d0"][f"{sysn}|{st}"] = d0
                    if not d0["pass"]:
                        print(f"[D0 FAIL] {sysn} {st}: {d0}", flush=True)
                if arm == "EATR":
                    todo_eat.append((st, sysn, rows)); continue
                live = [r for r in rows if r["refine"] == "live"]
                if live:                                         # occlusion onsets (BEATs) of new/changed tagger spans
                    ev = [AudioEvent(r["label"], r["pre_start"], r["end"], r["conf"]) for r in live]
                    out = _refine_onsets_cam(wav_of(st), ev, C["beats"][2], "cuda", skip_ids=set())
                    for r, e in zip(live, out):
                        r["start"] = float(e.start)
                res["arms"][f"{arm}|{sysn}"][st] = rows
        dump(STAGE4, res)
        print(f"[stage4] {st} done", flush=True)
    if todo_eat:                                                 # EAT-R: the same occlusion with EAT as the model
        from benchmark import detector_round5 as R5
        import torch
        score, _info = R5.eat_model("cuda")

        class EatAsBeats:
            def extract_features(self, x, padding_mask=None):
                return torch.from_numpy(score(x.detach().cpu().numpy())), None
        for st, sysn, rows in todo_eat:
            eat_labels = load_fr(EAT_DIR / f"{st}.npz")[2]
            B._MODEL = (EatAsBeats(), eat_labels, "cuda")
            live = [r for r in rows if r["refine"] == "live"]
            ev = [AudioEvent(r["label"], r["pre_start"], r["end"], r["conf"]) for r in live]
            out = _refine_onsets_cam(wav_of(st), ev, eat_labels, "cuda", skip_ids=set()) if ev else []
            for r, e in zip(live, out):
                r["start"] = float(e.start)
            res["arms"][f"EATR|{sysn}"][st] = rows
        B._MODEL = None
        dump(STAGE4, res)
    fails = [k for k, v in res["d0"].items() if not v["pass"]]
    print(f"[D0] {len(res['d0']) - len(fails)} / {len(res['d0'])} clip x system pass; fails {fails}", flush=True)


# ============================================================================= stage 5
class Reuse:
    """gate answers of the scored run (per label and stretch) + a memo of every other stage-5 question"""
    def __init__(self):
        from src.stage5_cross_modal_analysis import reason
        import src.stage2_video_understanding as S2
        self.reason, self.S2 = reason, S2
        self.memo = json.loads(MEMO.read_text(encoding="utf-8")) if MEMO.exists() else {}
        self.votes, self.last_t = [], None
        self.stats = {}
        self._ask, self._vis, self._sfa = reason._ask, reason._sound_is_visible, S2._sample_frames_at
        reason._ask = self.ask
        reason._sound_is_visible = self.vis
        S2._sample_frames_at = self.sfa

    def bump(self, k):
        self.stats[k] = self.stats.get(k, 0) + 1

    @staticmethod
    def _img(i):
        try:
            return hashlib.md5(i.tobytes()).hexdigest() + f"{i.size}"
        except Exception:
            return hashlib.md5(np.asarray(i).tobytes()).hexdigest()

    def ask(self, mdl, proc, prompt, *a, **kw):
        images = kw.get("images", a[0] if a else None)
        max_new = kw.get("max_new", a[1] if len(a) > 1 else 48)
        k = hashlib.sha1((prompt + "|" + str(max_new) + "|" + "|".join(self._img(i) for i in (images or []))).encode("utf-8")).hexdigest()
        if k in self.memo:
            self.bump("ask_memo"); return self.memo[k]
        ans = self._ask(mdl, proc, prompt, *a, **kw)
        self.memo[k] = ans; self.bump("ask_live")
        return ans

    def sfa(self, video, times, *a, **kw):
        self.last_t = list(times)
        return self._sfa(video, times, *a, **kw)

    def vis(self, label, frames, mdl, proc, device="cpu"):
        if frames and self.last_t:
            a, b = self.last_t[0] + 1.0, self.last_t[-1] - 1.0
            for v in self.votes:
                if v["label"] == label and abs(v["stretch"][0] - a) <= TOL and abs(v["stretch"][1] - b) <= TOL:
                    self.reason.LAST_VOTES = {k: v.get(k) for k in ("name", "ab", "desc", "named")}
                    self.bump("gate_reused"); self.stats.setdefault("gate_reused_list", []).append([label, a, b])
                    return bool(v["seen"]), v.get("named", "")
        self.bump("gate_live")
        a_b = [round(self.last_t[0] + 1.0, 2), round(self.last_t[-1] - 1.0, 2)] if self.last_t else None
        self.stats.setdefault("gate_live_list", []).append([label] + (a_b or []))
        return self._vis(label, frames, mdl, proc, device)

    def save(self):
        dump(MEMO, self.memo)


def stage5(arms):
    from benchmark.run_protocol import configure
    from src.stage5_cross_modal_analysis import plan_augmentations, reason
    from src.types import SceneContext, SpeechSegment
    changed = config.use_scored()
    config.DEVICE = "cuda"; config.TRANSCRIBE = True
    print("[cfg] use_scored:", {k: v[1] for k, v in changed.items()}, flush=True)
    for k, v in (("FLEXSED_BAR", 0.8), ("FLEXSED_VETO", 0.3), ("PANNS_VETO", 0.05), ("ONSET_MONOTONE", True), ("MAX_SPAN", None),
                 ("VLM_MODEL", "Qwen/Qwen3.8-27B"), ("VLM_THINKING", False), ("LABEL_FILTER", "depictable"),
                 ("KINSHIP_DIRECTED", False), ("PICTURE_MIN_CONF", None)):
        assert getattr(config, k, None) == v, (k, getattr(config, k, None), v)
    _g, stems = dev_stems()
    s4 = json.loads(STAGE4.read_text(encoding="utf-8"))
    R = Reuse()
    base = {k: getattr(config, k) for k in ("DISPLAY_THRESHOLD", "AUGMENT_THRESHOLD", "AED_THRESHOLD")}
    for sysn in ("blind_a2i", "proposed"):          # the pipeline without gate first: no VLM
        configure(sysn)
        for arm in arms:
            for k, v in base.items():
                setattr(config, k, v)
            if arm == "EATR":                           # the BAR= path of the release v1.2.0 job script
                config.DISPLAY_THRESHOLD = config.AUGMENT_THRESHOLD = F["EAT_DISP"]; config.AED_THRESHOLD = F["EAT_AED"]
            root = DC / f"{arm}_{sysn}"
            logp = root / "_stage5_log.json"
            log = json.loads(logp.read_text(encoding="utf-8")) if logp.exists() else {}
            for st in stems:
                d = root / st
                if (d / "augmentations.json").exists():
                    continue
                src = scored_dir(sysn) / st
                media = json.loads((src / "media.json").read_text(encoding="utf-8"))
                scene = SceneContext(**json.loads((src / "scene.json").read_text(encoding="utf-8")))
                segments = [SpeechSegment(**x) for x in json.loads((src / "segments.json").read_text(encoding="utf-8"))]
                events = [AudioEvent(r["label"], r["start"], r["end"], r["conf"]) for r in s4["arms"][f"{arm}|{sysn}"][st]]
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
                        s.image_path = PLACEHOLDER; s.backend = "placeholder"
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy(src / "media.json", d / "media.json")
                (d / "gate_votes.json").write_text(json.dumps(votes, indent=1), encoding="utf-8")
                (d / "augmentations.json").write_text(json.dumps([s.to_dict() for s in specs], indent=1), encoding="utf-8")
                log[st] = R.stats
                dump(logp, log)
                R.save()
    for k, v in base.items():
        setattr(config, k, v)
    R.save()
    print("[stage5] done", flush=True)


# ============================================================================= scoring
def pics_sig(pics):
    return sorted((l, round(a, 2), round(b, 2)) for l, a, b in (pics or []))


def spec_sig(d, st):
    p = d / st / "augmentations.json"
    if not p.exists():
        return None
    return sorted((s["event_label"], [[round(a, 2), round(b, 2)] for a, b in (s.get("spans") or [[s["start"], s["end"]]])])
                  for s in json.loads(p.read_text(encoding="utf-8")) if s.get("augment"))


def simulate_i7(root, stems, sysn, events_by_stem, log):
    """J2's rule with only the I7 test: drop a picture span iff every refined event of the same family overlapping it is a
    FlexSED-only span that I7 drops; a picture with no span left is not shown"""
    tmp = Path(tempfile.mkdtemp(prefix="i7_"))
    changes = []
    for st in stems:
        d = root / st
        if not (d / "augmentations.json").exists():
            continue
        (tmp / st).mkdir()
        if (d / "media.json").exists():
            shutil.copy(d / "media.json", tmp / st / "media.json")
        specs = json.loads((d / "augmentations.json").read_text(encoding="utf-8"))
        rows = events_by_stem[st]
        for s in specs:
            if s.get("augment") and s.get("image_path") and not Path(s["image_path"]).exists():
                s["image_path"] = PLACEHOLDER
            if not s.get("augment"):
                continue
            keep = []
            for span in s.get("spans") or [[s["start"], s["end"]]]:
                ev = [r for r in rows if same(r["label"], s["event_label"]) and min(span[1], r["end"]) - max(span[0], r["start"]) > 0]
                if not ev:
                    log["span_unmapped"] += 1; keep.append(span); continue
                if all(r.get("i7_drop", False) for r in ev):
                    changes.append((st, s["event_label"], [round(x, 2) for x in span]))
                else:
                    keep.append(span)
            if not keep:
                s["augment"] = False; log["pictures_removed"] += 1
            s["spans"] = keep
        (tmp / st / "augmentations.json").write_text(json.dumps(specs), encoding="utf-8")
    return tmp, changes


def metrics(rows):
    a = S.aggregate(rows)
    m = {k: a[k] for k in ("hits", "misses", "visible", "cross", "phantom", "dup", "F1", "P", "R", "viewer_cost", "coverage")}
    m["wrong"] = a["visible"] + a["cross"] + a["phantom"]
    return m


def clip_cost(r):
    return S.COST_MISS * r["miss"] + S.COST_FA * (r["visible"] + r["cross"] + r["phantom"])


def boot(d, n=2000, seed=0):
    """paired clip bootstrap of the mean per-clip difference (the draws of S.paired_ci / detector_round8.boot8);
    returns [mean, lo, hi, one-sided p = share of draws with mean >= 0]"""
    rng = np.random.default_rng(seed); d = np.asarray(d, float)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m >= 0).mean())]


def holm(ps, alpha=0.025):
    order = sorted(ps, key=lambda k: ps[k])
    out, stop = {}, False
    for i, k in enumerate(order):
        thr = alpha / (len(order) - i)
        rej = (not stop) and ps[k] <= thr
        stop = stop or not rej
        out[k] = {"p": ps[k], "threshold": thr, "rejected": bool(rej)}
    return out


def diff_pics(a, b):
    """pictures in b not in a (added) and in a not in b (removed), per clip, by label and time"""
    add, rem = [], []
    for st in a:
        A, Bq = pics_sig(a[st]), pics_sig(b[st])
        add += [(st,) + x for x in Bq if x not in A]
        rem += [(st,) + x for x in A if x not in Bq]
    return add, rem


def needed_hit(gold_clip, pics):
    """per needed sound (importance >= MIN_IMPORTANCE): True iff a same-family picture starts in its onset window (the
    official rule's matching, which also covers same-family sounds in the window); checked against score_clip's hits"""
    out = []
    for g in gold_clip:
        if not g["needed"] or g["importance"] < S.MIN_IMPORTANCE:
            continue
        out.append((g, any(S.same_family(l, g["label"]) and S.in_window(a, g["start"], S.EARLY, S.LATE) for l, a, _b in pics)))
    return out


def heard_pics(rows, disp):
    """the stage-4 spans a viewer could get: at or above the display bar, salient non-speech (depictable filter, as the run)"""
    from src.labels import is_salient_nonspeech
    old = config.LABEL_FILTER
    config.LABEL_FILTER = "depictable"
    try:
        return [(r["label"], r["start"], r["end"]) for r in rows if r["conf"] >= disp and is_salient_nonspeech(r["label"])]
    finally:
        config.LABEL_FILTER = old


def band_flag(fr, g):
    """round 8's heard-but-dropped flag on DEV: same-family FlexSED reaches 0.4-0.8 within [start - 1, end + 1]"""
    fw, ts, labs = fr
    cols = [i for i, l in enumerate(labs) if same(l, g["label"])]
    m = (ts >= g["start"] - 1.0) & (ts <= g["end"] + 1.0)
    if not cols or not m.any():
        return False
    v = float(fw[m][:, cols].max())
    return 0.4 <= v < 0.8


def complete(root, stems):
    return all((root / st / "augmentations.json").exists() for st in stems)


def score():
    gold, stems = dev_stems()
    s4 = json.loads(STAGE4.read_text(encoding="utf-8"))
    arms = [a for a in ARMS + EXTRA if all(complete(DC / f"{a}_{s}", stems) for s in SYSTEMS)
            and all(f"{a}|{s}" in s4["arms"] for s in SYSTEMS)]
    cands = [c for c in CANDS if c in arms or c == "I7"]
    res = {"plan": "docs/history/analyses/dev_candidates_check_2026-09-28.md (amendment 1: confirmatory; release v1.2.0)", "frozen": F, "clips": len(stems),
           "arms_scored": arms, "candidates": cands, "needed": None, "d0": {}, "d5": {}, "stage5": {}, "rows": {},
           "delta_vs_B1": {}, "delta_vs_B0": {}, "holm_vs_B1": {}, "ship_rule": {}, "verdict": {}, "changes_vs_B1": {},
           "i7": {}, "heard_stage4": {}, "hbd": {}, "filter": s4.get("filter", {})}
    d0 = s4["d0"]
    res["d0"] = {"pass": sum(v["pass"] for v in d0.values()), "of": len(d0), "fails": {k: v for k, v in d0.items() if not v["pass"]}}
    refine = {}
    for k, v in s4["arms"].items():
        cnt = {}
        for rows in v.values():
            for r in rows:
                cnt[r["refine"]] = cnt.get(r["refine"], 0) + 1
            cnt["live_moved"] = cnt.get("live_moved", 0) + sum(1 for r in rows if r["refine"] == "live" and r["start"] != r["pre_start"])
        refine[k] = cnt
    res["stage4_refine"] = refine
    flex = {st: load_fr(FLEX_DIR / f"{st}.npz") for st in stems}
    for sysn in SYSTEMS:
        P, rows = {}, {}
        P["B0"] = {st: S.load_pictures(scored_dir(sysn), st, sysn) or [] for st in stems}
        for arm in arms:
            P[arm] = {st: S.load_pictures(DC / f"{arm}_{sysn}", st, sysn) or [] for st in stems}
            lg = DC / f"{arm}_{sysn}" / "_stage5_log.json"
            st5 = json.loads(lg.read_text(encoding="utf-8")) if lg.exists() else {}
            tot = {}
            for v in st5.values():
                for k2 in ("gate_reused", "gate_live", "ask_memo", "ask_live"):
                    tot[k2] = tot.get(k2, 0) + v.get(k2, 0)
            tot["gate_live_list"] = [[st] + x for st, v in st5.items() for x in v.get("gate_live_list", [])]
            res["stage5"][f"{arm}|{sysn}"] = tot
        bad = [st for st in stems if pics_sig(P["B0r"][st]) != pics_sig(P["B0"][st])
               or spec_sig(DC / f"B0r_{sysn}", st) != spec_sig(scored_dir(sysn), st)]
        res["d5"][sysn] = {"pass": len(stems) - len(bad), "of": len(stems), "differ": bad,
                           "detail": {st: {"scored": pics_sig(P["B0"][st]), "repro": pics_sig(P["B0r"][st])} for st in bad}}
        for base, root, arm4 in (("B1", DC / f"B1_{sysn}", "B1"), ("B0", scored_dir(sysn), "B0r")):
            log = {"span_unmapped": 0, "pictures_removed": 0}
            tmp, ch = simulate_i7(root, stems, sysn, s4["arms"][f"{arm4}|{sysn}"], log)
            name = "I7" if base == "B1" else "I7_on_B0"
            P[name] = {st: S.load_pictures(tmp, st, sysn) or [] for st in stems}
            shutil.rmtree(tmp, ignore_errors=True)
            fo = [r for rr in s4["arms"][f"{arm4}|{sysn}"].values() for r in rr if r["origin"] == "flex"]
            res["i7"][f"{name}|{sysn}"] = {"removed_spans": ch, "log": log, "flex_only_spans": len(fo),
                                           "flex_only_dropped": sum(1 for r in fo if r.get("i7_drop"))}
        for name, pp in P.items():
            rows[name] = [S.score_clip(gold[st], pp[st]) for st in stems]
        res["rows"][sysn] = {name: metrics(r) for name, r in rows.items()}
        res["needed"] = res["rows"][sysn]["B0"]["hits"] + res["rows"][sysn]["B0"]["misses"]
        # needed sounds heard by stage 4 (before the gate; I7 = B1's stage 4 minus its I7 drops)
        heard = {}
        for name in arms + ["I7"]:
            src4 = "B1" if name == "I7" else name
            disp = F["EAT_DISP"] if name == "EATR" else F["DISP"]
            hr = []
            for st in stems:
                r4 = [r for r in s4["arms"][f"{src4}|{sysn}"][st] if not (name == "I7" and r.get("i7_drop"))]
                hr.append(S.score_clip(gold[st], heard_pics(r4, disp)))
            heard[name] = sum(r["hit"] for r in hr)
        heard["B0"] = heard["B0r"]
        res["heard_stage4"][sysn] = heard
        # heard-but-dropped group: needed sounds B1 misses (this system) with FlexSED 0.4-0.8 nearby; rescued by each arm
        grp = {st: [g for g, h in needed_hit(gold[st], P["B1"][st]) if not h and band_flag(flex[st], g)] for st in stems}
        chk = sum(h for st in stems for _g, h in needed_hit(gold[st], P["B1"][st]))
        hb = {"group": sum(len(v) for v in grp.values()), "per_sound_hits_check_B1": [chk, res["rows"][sysn]["B1"]["hits"]],
              "group_list": [[st, g["label"], g["start"]] for st, v in grp.items() for g in v]}
        for name in P:
            hb[name] = sum(1 for st in stems for g, h in needed_hit(gold[st], P[name][st]) if h and g in grp[st])
        res["hbd"][sysn] = hb
        cost = {name: [clip_cost(r) for r in rr] for name, rr in rows.items()}
        res["delta_vs_B1"][sysn] = {c: boot(np.subtract(cost[c], cost["B1"])) for c in cands + ["B0", "B0r"]}
        res["delta_vs_B0"][sysn] = {c: boot(np.subtract(cost[c], cost["B0"])) for c in cands + ["B1", "B0r", "I7_on_B0"]}
        res["holm_vs_B1"][sysn] = holm({c: res["delta_vs_B1"][sysn][c][3] for c in cands})
        R_ = res["rows"][sysn]
        res["ship_rule"][sysn], res["verdict"][sysn] = {}, {}
        for c in cands:
            sr = {}
            for bname in ("B1", "B0"):
                cc = "I7_on_B0" if (c == "I7" and bname == "B0") else c
                x, y = R_[cc], R_[bname]
                gain = x["hits"] - y["hits"]
                ok = gain >= 0 and (x["wrong"] - y["wrong"]) <= 2 * max(0, gain)
                sr[bname] = {"hits": [y["hits"], x["hits"]], "wrong": [y["wrong"], x["wrong"]], "pass": bool(ok)}
            res["ship_rule"][sysn][c] = sr
            d = res["delta_vs_B1"][sysn][c]
            rej = res["holm_vs_B1"][sysn][c]["rejected"]
            if rej and d[2] < 0 and sr["B1"]["pass"]:
                v = "better"
            elif (d[0] > 0 and d[1] > 0) or not sr["B1"]["pass"]:
                v = "worse"
            else:
                v = "same"
            res["verdict"][sysn][c] = v
        if sysn == "proposed":
            for c in cands + ["B0"]:
                add, rem = diff_pics(P["B1"], P[c])
                res["changes_vs_B1"][c] = {"added": add, "removed": rem}
        print(f"[D5 {sysn}] {res['d5'][sysn]['pass']}/{len(stems)} clips reproduce the scored render; differ: {bad}", flush=True)
        for name in ["B0", "B0r", "B1"] + cands + ["I7_on_B0"]:
            x = res["rows"][sysn][name]
            dd = res["delta_vs_B1"][sysn].get(name)
            print(f"DEV {sysn:9s} {name:9s} heard {heard.get(name, '-')} hits {x['hits']}/{x['hits'] + x['misses']} "
                  f"rescued {hb.get(name, '-')}/{hb['group']} wrong {x['wrong']} (vis {x['visible']}, cross {x['cross']}, "
                  f"phantom {x['phantom']}) dup {x['dup']} F1 {x['F1']:.3f} cost {x['viewer_cost']:.2f}"
                  + (f"  dcost vs B1 {dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}] p {dd[3]:.3f}" if dd else "")
                  + (f"  -> {res['verdict'][sysn][name]}" if name in res["verdict"][sysn] else ""), flush=True)
        print(f"[holm {sysn}] {res['holm_vs_B1'][sysn]}", flush=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("dasm", "stage4", "stage5", "score"))
    ap.add_argument("--arms", nargs="+", default=ARMS)
    a = ap.parse_args()
    {"dasm": dasm, "stage4": lambda: stage4(a.arms),
     "score": score}.get(a.step, lambda: stage5(a.arms))()


if __name__ == "__main__":
    main()
