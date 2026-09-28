"""Detector round 10 (docs/prereg_round10_rescue.md, 2026-09-28): a discriminator for sounds FlexSED hears at 0.4-0.8.

Band candidates (label-free): FlexSED spans at 0.4 (low 0.4) with peak < 0.8, no BEATs 0.175 twin within 1 s (discarded,
not merged), BEATs self-veto b 0.1218. Cells (admitted candidates are added to the shipped stack's shown spans):
  R0  all candidates (reference, never picked)
  R1  DASM same family >= 0.575 * 0.5 / 0.8 in [start - 0.5, end + 0.5]
  R2  both FlexSED paraphrases >= 0.5 in the span
  R3  FlexSED >= 0.5 in the span in >= 2 of 3 perturbed versions (pitch +1, pitch -1, 0.95x duration)
  R4  R1 AND R2         R5  R1 OR R2
  R6  round 8's I4 parent spans kept only if R1 OR R2 (E._same columns)
  R7  round 8's I6 lowered-bar spans (peak < 0.8) kept only if R1 OR R2

    python benchmark/detector_round10.py paraphrases                 # writes benchmark/round10_paraphrases.json (once)
    python benchmark/detector_round10.py prep --set calib|heldout    # CPU: candidates, 16-kHz wavs, views, GPU work list
    (GPU) python benchmark/round10_flexsed.py --work <list> --shard i --of n
    python benchmark/detector_round10.py check                       # caches complete and aligned (both sets)
    python benchmark/detector_round10.py fit                         # gate 0 / 0b on the 280, R0-R7, picks
    python benchmark/detector_round10.py heldout                     # the picks on the 415 + Holm (read the setup audit first)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark import detector_round8 as M
from src.labels import canonical
from src.stage4_audio_event_detection import _extract_events

R, D = M.R, M.D
OUT = _ROOT / "benchmark" / "detector_round10.json"
PARA = _ROOT / "benchmark" / "round10_paraphrases.json"
ONTO = _ROOT / "src" / "audioset_ontology.json"
BAND, FBAR, ADMIT = 0.4, 0.8, 0.5
DASM_BAR, R1_PAD = 0.575 * 0.5 / 0.8, 0.5
VIEWS = {"p1u": ("pitch", 1.0, 1.0), "p1d": ("pitch", -1.0, 1.0), "s95": ("stretch", 0.95, 0.95)}   # kind, arg, time scale
CELLS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
R8_EXPECT = {"I4": (2.9214285714285713, 3.807142857142857), "I6": (2.9285714285714284, 3.842857142857143)}
key, same = M.key, M.same


# ============================================================================= paraphrases (written once, before scoring)
def first_sentence(d):
    d = " ".join(d.split())
    for mt in re.finditer(r"\.\s", d):
        head = d[:mt.start()]
        if re.search(r"(\be\.g|\bi\.e|\betc)$", head):
            continue
        return head + "."
    return d if d.endswith(".") else d + "."


def paraphrases():
    assert not PARA.exists(), f"{PARA} exists: written once, not rewritten"
    fams = json.loads(M.VOCAB.read_text(encoding="utf-8"))["families"]
    onto = json.loads(ONTO.read_text(encoding="utf-8"))
    out = {}
    for f in fams:
        ent = next(x for x in onto if canonical(x["name"]) == f)
        parts = [re.sub(r"^(and|or)\s+", "", p.strip()) for p in f.split(", ")]
        p2 = " or ".join(p for p in parts if p)
        p2 = p2[0].upper() + p2[1:] + " can be heard in this recording."
        out[f] = {"ontology_name": ent["name"], "p1": first_sentence(ent["description"]), "p2": p2}
    PARA.write_text(json.dumps({"_template": {
        "p1": "first sentence of the family's AudioSet description (first ontology entry whose canonical name = family), as is",
        "p2": "family name split at ', ', leading and/or removed, joined with ' or ', first letter upper-case, + ' can be heard in this recording.'",
        "encoding": "full sentences given to FlexSED's CLAP text tower without the 'The sound of' wrapper"},
        "families": out}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(out)} families -> {PARA}")


# ============================================================================= band candidates (label-free)
def beats_raw(x):
    return _extract_events(x.b[0], x.b[1], x.b[2], M.AED, None, M.MIN_DUR, low=M.AED * M.HYS)


def band_cands(x):
    fw, ts, labs = x.f
    bev = beats_raw(x)
    bp = D.clip_peak(x.b)
    out = []
    for e in _extract_events(fw, ts, labs, BAND, None, M.MIN_DUR, low=BAND * M.HYS):
        if e.confidence >= FBAR:
            continue
        if any(key(y) == key(e) and y.start - 1.0 <= e.end and e.start - 1.0 <= y.end for y in bev):
            continue
        if bp.get(key(e), 1.0) < M.B:
            continue
        out.append(e)
    return out


def cands_path():
    return R.E.WIN / "round10_cands.json"


def para_dir():
    return R.E.WIN / "round10_para"


def view_dir(v):
    return R.E.WIN / f"round10_view_{v}"


def wav_dir(set_name):
    return _ROOT / "data" / "work" / "round10_wav" / set_name


def prep(set_name):
    """CPU: candidates, the original 16-kHz wav (flexsed_run's ffmpeg command), the 3 views for candidate clips, work list"""
    import librosa
    import soundfile as sf
    cl, xs = M.load_set(set_name)
    fams = json.loads(M.VOCAB.read_text(encoding="utf-8"))["families"]
    para = json.loads(PARA.read_text(encoding="utf-8"))["families"]
    cands, work = {}, []
    wd = wav_dir(set_name); wd.mkdir(parents=True, exist_ok=True)
    for x in xs:
        cs = band_cands(x)
        cands[x.cid] = [[e.label, e.start, e.end, e.confidence] for e in cs]
        wav = wd / f"{x.cid}.wav"
        if not wav.exists():
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(R.E.VIDEOS / f"{x.cid}.mp4"), "-ac", "1", "-ar", "16000",
                            str(wav)], check=True)
        item = {"id": x.cid, "wav": str(wav), "flex_cache": str(R.FLEX / f"{x.cid}.npz"), "para_out": str(para_dir() / f"{x.cid}.npz")}
        if cs:
            item["views"] = {}
            y = None
            for v, (kind, arg, scale) in VIEWS.items():
                vw = wd / v / f"{x.cid}__{v}.wav"
                if not vw.exists():
                    if y is None:
                        y, _ = librosa.load(str(wav), sr=16000, mono=True)
                    z = librosa.effects.pitch_shift(y, sr=16000, n_steps=arg) if kind == "pitch" else librosa.effects.time_stretch(y, rate=1.0 / arg)
                    vw.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(vw), np.clip(z, -1.0, 1.0).astype(np.float32), 16000, subtype="PCM_16")
                item["views"][v] = {"wav": str(vw), "out": str(view_dir(v) / f"{x.cid}.npz"), "time_scale": scale}
        work.append(item)
    cands_path().write_text(json.dumps(cands, indent=0), encoding="utf-8")
    wl = _ROOT / "data" / "work" / f"round10_work_{set_name}.json"
    wl.write_text(json.dumps({"set": set_name, "families": fams, "paraphrases": para, "clips": work,
                              "log_dir": str(R.E.WIN)}, indent=0), encoding="utf-8")
    n = sum(1 for v in cands.values() if v)
    print(f"[prep] {set_name}: {len(xs)} clips, {sum(len(v) for v in cands.values())} band candidates in {n} clips -> {wl}", flush=True)


# ============================================================================= caches for the tests
class X(M.Clip):
    def get(self, name):
        if name in self._x:
            return self._x[name]
        if name == "dasm":
            v = R.load(R.E.WIN / "dasm_cache" / f"{self.cid}.npz")
        elif name == "para":
            v = R.load(para_dir() / f"{self.cid}.npz")
        elif name in VIEWS:
            v = R.load(view_dir(name) / f"{self.cid}.npz")
        else:
            return super().get(name)
        self._x[name] = v
        return v


def load_set(set_name):
    R.use_set(set_name)
    cl = D.usable()
    return cl, [X(c, set_name) for c in cl]


def check(log):
    res = {}
    for s in ("calib", "heldout"):
        cl, xs = load_set(s)
        cands = json.loads(cands_path().read_text(encoding="utf-8"))
        miss, bad = [], []
        for x in xs:
            try:
                d = x.get("dasm")
                T = x.f[0].shape[0]
                if list(d[2]) != list(x.f[2]) or d[0].shape[0] > 500 or abs(d[0].shape[0] - 2 * min(T, 250)) > 2:
                    bad.append((x.cid, "dasm", d[0].shape))
            except FileNotFoundError:
                miss.append((x.cid, "dasm"))
            try:
                p = x.get("para")
                if p[0].shape != (x.f[0].shape[0], 430):
                    bad.append((x.cid, "para", p[0].shape))
            except FileNotFoundError:
                miss.append((x.cid, "para"))
            if cands.get(x.cid):
                for v, (_k, _a, sc) in VIEWS.items():
                    try:
                        w = x.get(v)
                    except FileNotFoundError:
                        miss.append((x.cid, v)); continue
                    T = x.f[0].shape[0]
                    # clarification 1 (prereg): a 10.007-10.01 s clip's cache has 25 padded tail frames (275); the stretched version
                    # (9.51 s) has none, so it is compared with the frames that carry audio, min(T, 250)
                    ok = list(w[2]) == list(x.f[2]) and (w[0].shape[0] == T if sc == 1.0 else abs(w[0].shape[0] - sc * min(T, 250)) <= 2)
                    if not ok:
                        bad.append((x.cid, v, w[0].shape, T))
        res[s] = {"clips": len(cl), "cand_clips": sum(1 for x in xs if cands.get(x.cid)), "missing": len(miss),
                  "missing_first": [list(map(str, m)) for m in miss[:5]], "misaligned": len(bad), "bad_first": [list(map(str, b)) for b in bad[:5]]}
        gl = sorted(R.E.WIN.glob("round10_gpu_*.json"))
        res[s]["gpu_logs"] = [json.loads(p.read_text(encoding="utf-8")) for p in gl]
        print(f"[check {s}] {res[s]['clips']} clips, {res[s]['cand_clips']} with candidates, missing {res[s]['missing']}, "
              f"misaligned {res[s]['misaligned']} {res[s]['bad_first']}; gpu logs {[(g['shard'], g['c2'], [round(c['max_diff'], 5) for c in g['c1']]) for g in res[s]['gpu_logs']]}", flush=True)
    log["check"] = res
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")
    return all(r["missing"] == 0 and r["misaligned"] == 0 for r in res.values())


# ============================================================================= the tests
def in_span(fr, cols, t0, t1):
    return M.peak_in(fr, cols, t0, t1)


def r1(x, e, match=False):
    fr = x.get("dasm")
    cols = M.fam_cols(fr[2], e.label) if match else M.cols_for(fr[2], key(e))
    if not cols:
        return False
    fw, ts, _l = fr
    m = (ts >= e.start - R1_PAD) & (ts <= e.end + R1_PAD)
    if not m.any():
        m = np.zeros(len(ts), bool); m[int(np.argmin(np.abs(ts - 0.5 * (e.start + e.end))))] = True
    return float(fw[m][:, cols].max()) >= DASM_BAR


def r2(x, e, match=False):
    fr = x.get("para")
    fams = [l.split("||")[0] for l in fr[2][::2]]
    targets = [f for f in fams if (same(f, e.label) if match else canonical(f) == key(e))]
    for f in targets:
        c1, c2 = fr[2].index(f"{f}||p1"), fr[2].index(f"{f}||p2")
        if in_span(fr, [c1], e.start, e.end) >= ADMIT and in_span(fr, [c2], e.start, e.end) >= ADMIT:
            return True
    return False


def r3(x, e):
    n = 0
    for v in VIEWS:
        fr = x.get(v)
        cols = M.cols_for(fr[2], key(e))
        if cols and in_span(fr, cols, e.start, e.end) >= ADMIT:
            n += 1
    return n >= 2


TEST = {"R0": lambda x, e: True, "R1": r1, "R2": r2, "R3": r3,
        "R4": lambda x, e: r1(x, e) and r2(x, e), "R5": lambda x, e: r1(x, e) or r2(x, e)}


def conf15(x, e, match):
    return r1(x, e, match) or r2(x, e, match)


# ============================================================================= the stack with I4 / I6 filters (R6, R7)
def stack10(x, i4=None, i6=None, stat=None):
    """= detector_round8.stack8 (no option / i4 / i6), with the R6 / R7 confirmation. i4: None | 'all' | 'conf';
    i6: None | ('all' | 'conf', listed families)"""
    events = beats_raw(x)
    if i4:
        ps = M.parent_spans(x.b)
        if i4 == "conf":
            ok = [p for p in ps if conf15(x, p, True)]
            if stat is not None:
                stat["considered"] += len(ps); stat["admitted"] += len(ok); stat["_adm"] += ok
            ps = ok
        events += ps
    if i6:
        mode, listed = i6
        fev = M.flex_spans(x.f, {"i6": listed})
        if mode == "conf":
            low = [e for e in fev if e.confidence < FBAR]
            ok = [e for e in low if conf15(x, e, False)]
            if stat is not None:
                stat["considered"] += len(low); stat["admitted"] += len(ok); stat["_adm"] += ok
            okid = {id(e) for e in ok}
            fev = [e for e in fev if e.confidence >= FBAR or id(e) in okid]
    else:
        fev = M.flex_spans(x.f, {})
    fresh = []
    for e in fev:
        tw = [y for y in events if key(y) == key(e) and y.start - 1.0 <= e.end and e.start - 1.0 <= y.end]
        if tw:
            for y in tw:
                y.start = min(y.start, e.start)
        else:
            fresh.append(e)
    flex_ids = {id(e) for e in fresh}
    events = events + fresh
    fp_ = D.clip_peak(x.f)
    events = [e for e in events if fp_.get(key(e), 1.0) >= M.FVETO]
    bp_ = D.clip_peak(x.b)
    events = [e for e in events if id(e) not in flex_ids or bp_.get(key(e), 1.0) >= M.B]
    return [e for e in events if e.confidence >= M.DISP]


def run_band(cell, xs, cands, stat):
    out = []
    for x in xs:
        base = M.stack8(x)
        cs = [e for e in band_cands(x)]
        assert [[e.label, e.start, e.end, e.confidence] for e in cs] == cands[x.cid], f"candidates changed on {x.cid}"
        ok = [e for e in cs if TEST[cell](x, e)]
        stat["considered"] += len(cs); stat["admitted"] += len(ok); stat["_adm"] += [(x, e) for e in ok]
        out.append(base + ok)
    return out


def run_cell(cell, xs, cands, vlm, stat):
    if cell in TEST:
        return run_band(cell, xs, cands, stat)
    out = []
    for x in xs:
        s0 = len(stat["_adm"])
        if cell == "R6":
            ev = stack10(x, i4="conf", stat=stat)
        else:
            ev = stack10(x, i6=("conf", vlm[x.cid]["families"]), stat=stat)
        stat["_adm"][s0:] = [(x, e) for e in stat["_adm"][s0:]]
        out.append(ev)
    return out


def adm_counts(stat):
    """admitted spans (salient, as scored) that overlap a gold event of their family (true) or not (false)"""
    adm = [(x, e) for x, e in stat.pop("_adm") if D._ev([e])]
    stat["admitted_scored"] = len(adm)
    stat["admitted_false"] = sum(M.is_false(e, x.c) for x, e in adm)
    stat["admitted_true"] = stat["admitted_scored"] - stat["admitted_false"]
    return stat


# ============================================================================= gates
def gate0b(xs, vlm, log):
    for x in xs:
        assert M.sig(stack10(x)) == M.sig(M.stack8(x)), f"stack10 != stack8 on {x.cid}"
        assert M.sig(stack10(x, i4="all")) == M.sig(M.stack8(x, {"i4": True})), f"stack10 i4 != stack8 i4 on {x.cid}"
        L = vlm[x.cid]["families"]
        assert M.sig(stack10(x, i6=("all", L))) == M.sig(M.stack8(x, {"i6": L})), f"stack10 i6 != stack8 i6 on {x.cid}"
    res = {}
    for k, kw in (("I4", {"i4": "all"}), ("I6", None)):
        evs = [stack10(x, **kw) if kw else stack10(x, i6=("all", vlm[x.cid]["families"])) for x in xs]
        _r, _rd, S = M.score(xs, evs)
        res[k] = [S["C_overlap"], S["C_onset"]]
        assert abs(S["C_overlap"] - R8_EXPECT[k][0]) < 1e-9 and abs(S["C_onset"] - R8_EXPECT[k][1]) < 1e-9, f"{k} != round 8: {res[k]}"
    log["gate0b"] = {"span_for_span": True, "round8_costs": res}
    print(f"[gate0b] stack10 = stack8 (none / i4 / i6) span for span; I4 / I6 costs = round 8: {res}", flush=True)


def vlm_answers(set_name):
    return json.loads((R.E.WIN / "round8_vlm.json").read_text(encoding="utf-8"))


def one_set(set_name, cells, log, key_):
    cl, xs, base_evs, brows, brows_d, Sb = M.gate_set(set_name, log)
    xs = [X(x.c, set_name) for x in xs]                 # same clips, round-10 cache access
    vlm = vlm_answers(set_name)
    if key_ == "fit":
        gate0b(xs, vlm, log)
    cands = json.loads(cands_path().read_text(encoding="utf-8"))
    flags = M.hbd_flags(xs, base_evs)
    base = (base_evs, brows, brows_d, flags)
    Rs = log.setdefault(key_, {})
    Rs["baseline"] = Sb
    Rs["band_pool"] = len(flags)
    Rs["candidates"] = {"n": sum(len(v) for k, v in cands.items() if k in {x.cid for x in xs}),
                        "clips": sum(1 for x in xs if cands.get(x.cid))}
    for cell in cells:
        stat = {"considered": 0, "admitted": 0, "_adm": []}
        evs = run_cell(cell, xs, cands, vlm, stat)
        rows, _rd, S = M.score(xs, evs, base)
        S["filter"] = adm_counts(stat)
        if key_ == "heldout":
            S["pass"] = bool(S["dC_overlap"][2] < 0)
            d = [a["C_overlap"] - b["C_overlap"] for a, b in zip(rows, brows)]
            S["strata"] = {s: {"n": len(ix), "dC_overlap": M.boot8([d[i] for i in ix]) if ix else None}
                           for s, ix in ((s, [i for i, x in enumerate(xs) if x.c.get("stratum") == s]) for s in ("complex", "random"))}
        Rs[cell] = S
        M.show(cell, S)
        print(f"   filter {S['filter']}" + (f" -> {'PASS' if S['pass'] else 'fail'}" if key_ == "heldout" else ""), flush=True)
        OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")
    return Rs, Sb


def fit(log):
    Rs, Sb = one_set("calib", ["R0"] + CELLS, log, "fit")
    log["picks"] = [k for k in CELLS if Rs[k]["C_overlap"] < Sb["C_overlap"] and Rs[k]["C_onset"] < Sb["C_onset"]]
    print(f"[picks] to the 415: {log['picks']}", flush=True)
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


def heldout(log):
    if not log.get("picks"):
        log["heldout"] = "no cell picked on the 280"; print(log["heldout"])
        OUT.write_text(json.dumps(log, indent=1), encoding="utf-8"); return
    H, _Sb = one_set("heldout", ["R0"] + log["picks"], log, "heldout")
    H["holm"] = M.holm({k: H[k]["dC_overlap"][3] for k in log["picks"]})
    for k in log["picks"]:
        H[k]["pass_and_holm"] = bool(H[k]["pass"] and H["holm"][k]["reject"])
    print(f"[holm] {H['holm']}", flush=True)
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


# ============================================================================= reach diagnosis (lead, after the HOLD; 280 only, descriptive)
def cands_opt(x, twin="any", min_dur=None, high=False, sv=True):
    """band candidates with one rule changed. twin: 'any' (shipped: any BEATs 0.175 twin discards) | 'shown' (only a twin
    with confidence >= 0.35 discards); min_dur; high: keep 0.4-spans that also reach >= 0.8; sv: BEATs self-veto"""
    fw, ts, labs = x.f
    bev = beats_raw(x)
    bp = D.clip_peak(x.b)
    out = []
    for e in _extract_events(fw, ts, labs, BAND, None, M.MIN_DUR if min_dur is None else min_dur, low=BAND * M.HYS):
        if e.confidence >= FBAR and not high:
            continue
        tw = [y for y in bev if key(y) == key(e) and y.start - 1.0 <= e.end and e.start - 1.0 <= y.end]
        if (twin == "any" and tw) or (twin == "shown" and any(y.confidence >= M.DISP for y in tw)):
            continue
        if sv and bp.get(key(e), 1.0) < M.B:
            continue
        out.append(e)
    return out


def hits(ev, g):
    return [e for e in ev if same(e.label, g["label"]) and R.E._overlap_ok(e.start, e.end, g["start"], g["end"])]


def reach(log):
    cl, xs, base_evs, brows, brows_d, Sb = M.gate_set("calib", log)
    xs = [X(x.c, "calib") for x in xs]
    flags = M.hbd_flags(xs, base_evs)
    cands = json.loads(cands_path().read_text(encoding="utf-8"))
    ev_by = {}
    for i, gi in flags:
        ev_by.setdefault(i, []).append(gi)
    reasons, rows = {}, []
    for i, gis in ev_by.items():
        x = xs[i]
        cs = cands_opt(x)
        assert [[e.label, e.start, e.end, e.confidence] for e in cs] == cands[x.cid]
        fw, ts, labs = x.f
        bev = beats_raw(x)
        bp = D.clip_peak(x.b)
        for gi in gis:
            g = x.c["events"][gi]
            if hits(cs, g):
                r = "reachable (a candidate hits it)"
            else:
                cols = M.fam_cols(labs, g["label"])
                raw = hits(_extract_events(fw[:, cols], ts, [labs[c] for c in cols], BAND, None, M.MIN_DUR, low=BAND * M.HYS), g)
                if not raw:
                    inside = (ts >= g["start"]) & (ts <= g["end"])
                    if not (inside.any() and fw[inside][:, cols].max() >= BAND):
                        r = "FlexSED >= 0.4 only outside the event (in the +-1 s margin)"
                    elif hits(_extract_events(fw[:, cols], ts, [labs[c] for c in cols], BAND, None, 0.0, low=BAND * M.HYS), g):
                        r = "span shorter than 0.5 s"
                    else:
                        r = "other (0.4 span overlaps the event too little)"
                else:
                    stage = []
                    for e in raw:                      # how far each hitting raw span gets through the rules
                        if e.confidence >= FBAR:
                            stage.append((0, "span also reaches >= 0.8 (not a band span)")); continue
                        tw = [y for y in bev if key(y) == key(e) and y.start - 1.0 <= e.end and e.start - 1.0 <= y.end]
                        if tw:
                            shown = any(y.confidence >= M.DISP for y in tw)
                            stage.append((1, "BEATs twin within 1 s, " + ("shown (>= 0.35)" if shown else "weak (< 0.35, not shown)"))); continue
                        if bp.get(key(e), 1.0) < M.B:
                            stage.append((2, "BEATs self-veto")); continue
                        stage.append((3, "other"))
                    r = max(stage)[1]
            reasons[r] = reasons.get(r, 0) + 1
            rows.append({"clip": x.cid, "label": g["label"], "start": g["start"], "end": g["end"], "reason": r})
    base_n = sum(len(v) for v in cands.values())
    changes = {"weak twin kept (only a shown twin discards)": {"twin": "shown"}, "min duration 0.25 s": {"min_dur": 0.25},
               "keep 0.4 spans that reach >= 0.8": {"high": True}, "self-veto off": {"sv": False},
               "all four": {"twin": "shown", "min_dur": 0.25, "high": True, "sv": False}}
    ch = {}
    for name, kw in changes.items():
        n = nf = reach_n = 0
        for i, x in enumerate(xs):
            cs = cands_opt(x, **kw)
            n += len(cs); nf += sum(M.is_false(e, x.c) for e in D._ev(cs))
            reach_n += sum(bool(hits(cs, x.c["events"][gi])) for gi in ev_by.get(i, []))
        ch[name] = {"reachable_of_band": reach_n, "candidates": n, "extra_candidates": n - base_n, "candidates_false": nf}
        print(f"[reach] {name}: reachable {reach_n}/{len(flags)}, candidates {n} (+{n - base_n}), false {nf}", flush=True)
    log["reach"] = {"band_pool": len(flags), "base_candidates": base_n,
                    "base_candidates_false": sum(M.is_false(e, x.c) for x in xs for e in D._ev(cands_opt(x))),
                    "reasons": reasons, "changes": ch, "events": rows}
    print(f"[reach] reasons {json.dumps(reasons, indent=1)}", flush=True)
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("paraphrases", "prep", "check", "fit", "heldout", "reach"))
    ap.add_argument("--set", choices=("calib", "heldout"))
    a = ap.parse_args()
    if a.step == "paraphrases":
        paraphrases(); return
    if a.step == "prep":
        prep(a.set); return
    log = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    if a.step == "check":
        ok = check(log); sys.exit(0 if ok else "check failed")
    if a.step in ("fit", "heldout") and not check(log):
        sys.exit("caches incomplete or misaligned: stop")
    if a.step == "reach":
        reach(log); return
    {"fit": fit, "heldout": heldout}[a.step](log)


if __name__ == "__main__":
    main()
