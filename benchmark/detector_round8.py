"""Detector round 8 (docs/history/preregistrations/prereg_round8_ideas.md, release v1.2.0, 2026-09-28): new ideas for sounds a detector hears but the stack drops.

Cells (all a priori; see the prereg): I2 repetition-conditioned FlexSED bar (6 cells), I3 per-clip normalised BEATs,
I4 ontology parent emission, I6 VLM scene prior, I7 FlexSED local-contrast veto, I9 ontology-matched vetoes,
I10 speech/music guard on BEATs-only spans, I5 multi-scale BEATs, I8 band-limited views; timing cells I1-i/ii/iii.
Baseline = the shipped stack (detector_round5.run, slot 0 BEATs, AED 0.175, display 0.35, self-veto b 0.1218).

    python benchmark/detector_round8.py views  --set calib|heldout     # LP < 300 Hz / HP > 4 kHz wavs (CPU)
    python benchmark/detector_round8.py beats  --set calib|heldout     # GPU: BEATs 1-s windows + BEATs on the two views
    python benchmark/detector_round8.py vlm    --set calib|heldout     # GPU: Qwen3.8-27B scene prior
    python benchmark/detector_round8.py check                          # caches complete and aligned (both sets)
    python benchmark/detector_round8.py gate                           # gate 0 on both sets
    python benchmark/detector_round8.py fit    [--cells ...]           # the 280: refits, cells, picks
    python benchmark/detector_round8.py heldout                        # the picks on the 415 + Holm
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark import audioset_stage4_report as R
from benchmark import detector_round2 as D
from benchmark import detector_round4 as R4
from benchmark import detector_round5 as R5
from src.labels import canonical, _parents, is_salient_nonspeech, is_music
from src.stage4_audio_event_detection import _extract_events

OUT = _ROOT / "benchmark" / "detector_round8.json"
VOCAB = _ROOT / "benchmark" / "gold" / "depictable_vocab.json"
B = R5.B_SHIPPED
AED, DISP, HYS, MIN_DUR = D.AED, D.DISP, D.HYS, config.AED_MIN_DUR
FBAR, FVETO = R4.FBAR, R4.FVETO
SR = 16000
EXPECT = {"calib": {"C_overlap": 3.0357142857142856, "C_onset": 3.85, "recall_overlap": 0.5133928571428572, "fp": 207, "shown": 564},
          "heldout": {"C_overlap": 1.927710843373494, "C_onset": 2.207228915662651, "recall_overlap": 0.4970760233918129, "fp": 228,
                      "shown": 672}}
key = lambda e: canonical(e.label)
same = lambda a, b: R.E._same(a, b)
LOGIT035 = float(np.log(0.35 / 0.65))
I2_W = {"5s": 5.0, "10s": 10.0, "clip": float("inf")}
I2_BARS = {"b05": 0.5, "b04": 0.4}
RP_CELLS = ["I2-b05-5s", "I2-b05-10s", "I2-b05-clip", "I2-b04-5s", "I2-b04-10s", "I2-b04-clip",
            "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10"]
T_CELLS = ["I1-i", "I1-ii", "I1-iii"]


# ============================================================================= paths
def views_dir(set_name, v):
    return _ROOT / "data" / "work" / "round8_views" / set_name / v


def flex_view_dir(set_name, v):
    return _ROOT / "data" / "work" / f"round8_flexsed_{set_name}_{v}"


def beats_cache_dir(which):                     # which: "1s" | "lp" | "hp"  (inside the set's windows folder)
    return R.E.WIN / f"round8_beats_{which}"


def vlm_path():
    return R.E.WIN / "round8_vlm.json"


# ============================================================================= GPU / cache steps
def load_audio(mp4):
    import librosa
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vn", "-ac", "1", "-ar", str(SR), str(wav)], check=True)
        audio, _ = librosa.load(str(wav), sr=SR, mono=True)
    return audio


def views(set_name):
    """8th-order Butterworth, zero phase: low-pass < 300 Hz and high-pass > 4 kHz, as <id>__lp.wav / <id>__hp.wav"""
    import soundfile as sf
    from scipy.signal import butter, sosfiltfilt
    R.use_set(set_name)
    sos = {"lp": butter(8, 300, "lowpass", fs=SR, output="sos"), "hp": butter(8, 4000, "highpass", fs=SR, output="sos")}
    for v in sos:
        views_dir(set_name, v).mkdir(parents=True, exist_ok=True)
    n = 0
    for c in D.usable():
        dst = {v: views_dir(set_name, v) / f"{c['id']}__{v}.wav" for v in sos}
        if all(p.exists() for p in dst.values()):
            continue
        a = load_audio(R.E.VIDEOS / f"{c['id']}.mp4")
        for v, s in sos.items():
            y = np.clip(sosfiltfilt(s, a), -1.0, 1.0).astype(np.float32)
            sf.write(str(dst[v]), y, SR, subtype="PCM_16")
        n += 1
    print(f"[views] {set_name}: {n} clips written -> {views_dir(set_name, 'lp').parent}", flush=True)


def windows(audio, window, hop, stamp):
    """infer_beats' windowing with a free window / stamp offset (lead pad = window - hop, reflect)"""
    n_win = int(round(window * SR)); n_hop = int(round(hop * SR))
    lead = n_win - n_hop
    audio = np.pad(audio, (lead, 0), mode="reflect" if len(audio) > lead else "constant")
    if len(audio) < n_win:
        audio = np.pad(audio, (0, n_win - len(audio)))
    starts = list(range(0, len(audio) - n_win + 1, n_hop))
    if not starts or starts[-1] + n_win < len(audio):
        starts.append(max(0, len(audio) - n_win))
    times = np.array([(s + n_win - lead) / SR - stamp for s in starts])
    keep = times >= 0.0
    return np.stack([audio[s:s + n_win] for s in starts]).astype(np.float32)[keep], times[keep]


def beats(set_name, device="cuda", batch=64):
    """BEATs on 1-s windows (hop 0.25, stamp 0.25 s before the window end) and on the two band views (2-s, as shipped)"""
    import librosa
    import torch
    from src.stage4_audio_event_detection.beats_infer import _load
    R.use_set(set_name)
    model, names, dev = _load(device)

    def score(chunks):
        out = []
        with torch.no_grad():
            for i in range(0, len(chunks), batch):
                x = torch.from_numpy(chunks[i:i + batch]).float().to(dev)
                p, _ = model.extract_features(x, padding_mask=torch.zeros(x.shape, dtype=torch.bool, device=dev))
                out.append(p.float().cpu().numpy())
        return np.concatenate(out, 0)
    for w in ("1s", "lp", "hp"):
        beats_cache_dir(w).mkdir(parents=True, exist_ok=True)
    n = 0
    for c in D.usable():
        cid = c["id"]
        _bf, bt, bl = R.load(R.E.WIN / "beats" / f"{cid}.npz")
        assert list(bl) == list(names)
        dst = beats_cache_dir("1s") / f"{cid}.npz"
        if not dst.exists():
            ch, t = windows(load_audio(R.E.VIDEOS / f"{cid}.mp4"), 1.0, 0.25, 0.25)
            np.savez_compressed(dst, fw=score(ch).astype(np.float16), times=t.astype(np.float32), labels=np.array(names))
        for v in ("lp", "hp"):
            dst = beats_cache_dir(v) / f"{cid}.npz"
            if dst.exists():
                continue
            a, _ = librosa.load(str(views_dir(set_name, v) / f"{cid}__{v}.wav"), sr=SR, mono=True)
            ch, t = windows(a, 2.0, 0.25, 0.5)
            assert len(t) == len(bt) and np.allclose(t, bt, atol=1e-4), f"window mismatch {cid} {v}"
            np.savez_compressed(dst, fw=score(ch).astype(np.float16), times=t.astype(np.float32), labels=np.array(names))
        n += 1
    print(f"[beats] {set_name}: {n} clips done", flush=True)


VLM_MODEL = "Qwen/Qwen3.8-27B"
VLM_PROMPT = ("These are 4 frames from one short video, in time order. Below is a numbered list of sound sources. Which of them "
              "could plausibly be heard in this place, on screen or off screen? Include every source that is plausible for this "
              "kind of place, not only what you can see. Answer with the numbers only, separated by commas. If none is plausible, "
              "answer: none.\n\n")
# amendment 2 (2026-09-28, before any I6 cost): try 1 wrote explanations and 6 of its first 10 answers were cut at the
# token budget, so the lists were incomplete. This format line is appended after the numbered list; nothing else changes.
VLM_FORMAT = ("\n\nReply with ONE line that contains only the numbers, separated by commas (for example: 3, 17, 102). "
              "Do not write any words, names or explanations.")
VLM_MAX_NEW = 1024


def vlm(set_name):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from src.stage2_video_understanding.vlm import _frames
    R.use_set(set_name)
    fams = json.loads(VOCAB.read_text(encoding="utf-8"))["families"]
    assert len(fams) == 215
    prompt = VLM_PROMPT + "\n".join(f"{i}. {f}" for i, f in enumerate(fams, 1)) + VLM_FORMAT
    proc = AutoProcessor.from_pretrained(VLM_MODEL)
    mdl = AutoModelForImageTextToText.from_pretrained(VLM_MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    out = json.loads(vlm_path().read_text(encoding="utf-8")) if vlm_path().exists() else {}
    out.setdefault("_meta", {"model": VLM_MODEL, "prompt": prompt, "max_new": VLM_MAX_NEW, "thinking": False, "frames": 4})
    todo = [c for c in D.usable() if c["id"] not in out]
    for i, c in enumerate(todo, 1):
        imgs = _frames(R.E.VIDEOS / f"{c['id']}.mp4", 4)
        content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": prompt}]
        try:
            text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
        except TypeError:
            text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=imgs, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            g = mdl.generate(**inputs, max_new_tokens=VLM_MAX_NEW, do_sample=False)
        new = g[:, inputs["input_ids"].shape[1]:]
        ans = proc.batch_decode(new, skip_special_tokens=True)[0].strip()
        nums = sorted({int(x) for x in re.findall(r"\d+", ans) if 1 <= int(x) <= 215})
        out[c["id"]] = {"answer": ans, "n_frames": len(imgs), "cut": bool(new.shape[1] >= VLM_MAX_NEW),
                        "families": [fams[k - 1] for k in nums]}
        if i % 10 == 0 or i == len(todo):
            vlm_path().write_text(json.dumps(out, indent=1), encoding="utf-8")
            print(f"[vlm] {set_name} {i}/{len(todo)}  last: {len(nums)} families, cut {out[c['id']]['cut']}", flush=True)
    vlm_path().write_text(json.dumps(out, indent=1), encoding="utf-8")


# ============================================================================= caches for scoring
def fr_max(frs):
    fw = frs[0][0].copy()
    for f in frs[1:]:
        assert f[0].shape == fw.shape and list(f[2]) == list(frs[0][2])
        fw = np.maximum(fw, f[0])
    return fw, frs[0][1], frs[0][2]


class Clip:
    """lazy per-clip caches"""
    def __init__(self, c, set_name):
        self.c, self.cid, self.set, self.win = c, c["id"], set_name, R.E.WIN
        self.b, self.f, self.p = R4.load3(self.cid)
        self._x = {}

    def get(self, name):
        if name not in self._x:
            cid = self.cid
            if name in ("b1s", "blp", "bhp"):
                v = self._x[name] = R.load(self.win / f"round8_beats_{name[1:]}" / f"{cid}.npz")
            elif name in ("flp", "fhp"):
                v = self._x[name] = R.load(flex_view_dir(self.set, name[1:]) / f"{cid}__{name[1:]}.npz")
            elif name == "b8":
                v = self._x[name] = fr_max([self.b, self.get("blp"), self.get("bhp")])
            elif name == "f8":
                v = self._x[name] = fr_max([self.f, self.get("flp"), self.get("fhp")])
            elif name == "z":                                        # I3: logit minus the frame's median logit
                p = np.clip(self.b[0].astype(np.float64), 1e-6, 1 - 1e-6)
                L = np.log(p / (1 - p))
                v = self._x[name] = L - np.median(L, axis=1, keepdims=True)
            else:
                raise KeyError(name)
        return self._x[name]


def load_set(set_name):
    R.use_set(set_name)
    cl = D.usable()
    return cl, [Clip(c, set_name) for c in cl]


# ============================================================================= the stack
def cols_for(labs, k):
    return [i for i, l in enumerate(labs) if canonical(l) == k]


def fam_cols(labs, label):
    return [i for i, l in enumerate(labs) if same(l, label)]


def flex_spans(F, o, anchors=None):
    """FlexSED spans: 0.8 by default; I6 per-family bar; I2 repetition-conditioned re-extraction"""
    fw, ts, labs = F
    if o.get("i6") is None and o.get("i2") is None:
        return _extract_events(fw, ts, labs, FBAR, None, MIN_DUR, low=FBAR * HYS)
    out = []
    listed = set(o["i6"]) if o.get("i6") is not None else None
    for j, lab in enumerate(labs):
        bar = 0.5 if (listed is not None and lab in listed) else FBAR
        base = _extract_events(fw[:, [j]], ts, [lab], bar, None, MIN_DUR, low=bar * HYS)
        if o.get("i2") is not None and anchors and canonical(lab) in anchors:
            beta, W = o["i2"]
            med = float(np.median(fw[:, j]))
            an = anchors[canonical(lab)]
            for s in _extract_events(fw[:, [j]], ts, [lab], beta, None, MIN_DUR, low=beta * HYS):
                ok_w = any(s.start <= a1 + W and s.end >= a0 - W for a0, a1 in an)
                if not ok_w or s.confidence - med < 0.3:
                    continue
                if s.confidence >= FBAR:                 # contains 0.8 span(s): replace them (the weak start becomes the onset)
                    base = [e for e in base if not (e.start >= s.start - 1e-9 and e.end <= s.end + 1e-9)]
                    base.append(s)
                else:
                    base.append(s)
        out += base
    out.sort(key=lambda e: (e.start, -e.confidence))
    return out


def parent_spans(b):
    """I4: 1 - prod(1 - p_child) over direct children; emitted (display-level) where no child / P itself reaches 0.35"""
    fw, ts, labs = b
    idx = {l: i for i, l in enumerate(labs)}
    kids = {}
    for ch, par in _parents().items():
        if ch in idx and par in idx:
            kids.setdefault(par, []).append(idx[ch])
    out = []
    for P, ch in kids.items():
        if len(ch) < 2:
            continue
        s = 1.0 - np.prod(1.0 - fw[:, ch], axis=1)
        if s.max() < DISP:
            continue
        for e in _extract_events(s[:, None], ts, [P], AED, None, MIN_DUR, low=AED * HYS):
            if e.confidence < DISP:
                continue
            m = (ts >= e.start) & (ts < e.end)
            if m.any() and fw[m][:, ch + [idx[P]]].max() >= DISP:
                continue
            out.append(e)
    return out


def peak_in(fr, cols, t0, t1):
    fw, ts, _l = fr
    m = (ts >= t0) & (ts <= t1)
    if not m.any():
        m = np.zeros(len(ts), bool); m[int(np.argmin(np.abs(ts - 0.5 * (t0 + t1))))] = True
    return float(fw[m][:, cols].max())


def clip_peak_same(fr, label, cache):
    """max clip score over the detector's labels matching `label` by E._same; None if none match"""
    fw, _t, labs = fr
    k = (id(fr), label)
    if k not in cache:
        cols = fam_cols(labs, label)
        cache[k] = float(fw[:, cols].max()) if cols else None
    return cache[k]


def stack8(x, o=None, tag=False):
    """the shipped stack with options. x: Clip. Returns shown spans (and, with tag, origin info)."""
    o = o or {}
    Bsrc = o.get("bsrc", x.b)                       # BEATs span source
    Bv = o.get("bveto", x.b)                        # BEATs for the self-veto
    F = o.get("fsrc", x.f)                          # FlexSED span source and clip veto
    events = _extract_events(Bsrc[0], Bsrc[1], Bsrc[2], AED, None, MIN_DUR, low=AED * HYS)
    if o.get("i4"):
        events += parent_spans(x.b)
    fev = flex_spans(F, o, o.get("anchors"))
    fresh, twinned = [], {}
    for e in fev:
        tw = [y for y in events if key(y) == key(e) and y.start - 1.0 <= e.end and e.start - 1.0 <= y.end]
        if tw:
            for y in tw:
                if o.get("i1") == "i":
                    if id(y) not in twinned:
                        y.start = e.start
                else:
                    y.start = min(y.start, e.start)
                twinned[id(y)] = True
        else:
            fresh.append(e)
    flex_ids = {id(e) for e in fresh}
    events = events + fresh
    cache = {}
    if o.get("i9"):
        fpk = lambda e: clip_peak_same(F, e.label, cache)
        bpk = lambda e: clip_peak_same(Bv, e.label, cache)
        events = [e for e in events if (fpk(e) is None or fpk(e) >= FVETO)]
        events = [e for e in events if id(e) not in flex_ids or bpk(e) is None or bpk(e) >= B]
    else:
        fp_ = D.clip_peak(F)
        events = [e for e in events if fp_.get(key(e), 1.0) >= FVETO]
        bp_ = D.clip_peak(Bv)
        events = [e for e in events if id(e) not in flex_ids or bp_.get(key(e), 1.0) >= B]
    events = [e for e in events if e.confidence >= DISP]
    if o.get("i7"):
        fw, ts, labs = F
        keep = []
        for e in events:
            if id(e) in flex_ids:
                cols = cols_for(labs, key(e))
                s = fw[:, cols].max(axis=1)
                ins = (ts >= e.start) & (ts < e.end)
                fl = ((ts >= e.start - 3) & (ts < e.start)) | ((ts >= e.end) & (ts < e.end + 3))
                if ins.any() and fl.any() and s[ins].mean() - s[fl].mean() < 0.2:
                    continue
            keep.append(e)
        events = keep
    if o.get("i10"):
        bl = x.b[2]
        sm = [bl.index("Speech"), bl.index("Music")]
        keep = []
        for e in events:
            if id(e) not in flex_ids and id(e) not in twinned and peak_in(x.b, sm, e.start, e.end) >= 0.3:
                cols = fam_cols(x.f[2], e.label)
                ok = (peak_in(x.f, cols, e.start, e.end) >= 0.5) if cols else (e.confidence >= 0.5)
                if not ok:
                    continue
            keep.append(e)
        events = keep
    if o.get("i1") == "ii":
        events = split_dips(events, x.f)
    if o.get("i1") == "iii":
        m = o["shift"]
        events = [dataclasses.replace(e, start=float(min(max(e.start - m, 0.0), e.end - 0.25))) for e in events]
    if tag:
        return events, {"flex": flex_ids, "twinned": set(twinned)}
    return events


def split_dips(events, f):
    fw, ts, labs = f
    dt = 1.0 / 25.0
    nmin = int(np.ceil(0.3 / dt - 1e-9))
    out = []
    for e in events:
        cols = cols_for(labs, key(e))
        m = np.where((ts >= e.start) & (ts < e.end))[0]
        if not cols or len(m) < nmin + 2:
            out.append(e); continue
        s = fw[m][:, cols].max(axis=1)
        cuts, i, n = [], 0, len(s)
        while i < n:
            if s[i] < FBAR / 2:
                j = i
                while j < n and s[j] < FBAR / 2:
                    j += 1
                if j - i >= nmin and (s[:i] >= FBAR).any() and j < n and (s[j:] >= FBAR).any():
                    cuts.append((float(ts[m[i]]), float(ts[m[j]])))
                i = j
            else:
                i += 1
        if not cuts:
            out.append(e); continue
        bounds, st = [], e.start
        for a, b_ in cuts:
            bounds.append((st, a)); st = b_
        bounds.append((st, e.end))
        out += [dataclasses.replace(e, start=a, end=b_) for a, b_ in bounds if b_ - a >= MIN_DUR]
    return out


def base_run(xs):
    """round 5's shipped stack (the reference) on the clips"""
    return [R5.finish(R5.pre(x.b, x.f, AED), B, DISP) for x in xs]


def sig(ev):
    return sorted((e.label, round(e.start, 6), round(e.end, 6), round(e.confidence, 6)) for e in ev)


# ============================================================================= scoring
@contextlib.contextmanager
def label_filter(mode):
    old = getattr(config, "LABEL_FILTER", "lists")
    config.LABEL_FILTER = mode
    try:
        yield
    finally:
        config.LABEL_FILTER = old


def boot8(d, n=2000, seed=0):
    """= detector_round2.boot (same draws), plus the one-sided p = share of draws with mean >= 0"""
    rng = np.random.default_rng(seed); d = np.asarray(d, float)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m >= 0).mean())]


def is_false(e, c):
    return not any(same(e.label, g["label"]) and min(e.end, g["end"]) - max(e.start, g["start"]) > 0 for g in c["events"])


def conseq(c):
    return [g for g in c["events"] if is_salient_nonspeech(g["label"]) and not is_music(g["label"]) and g["consequential"]]


def hit(ev, g):
    return any(same(e.label, g["label"]) and R.E._overlap_ok(e.start, e.end, g["start"], g["end"]) for e in ev)


def hbd_flags(xs, base_evs):
    """baseline misses with FlexSED 0.4-0.8 near them (the audit's flag ii); [(clip index, gold index)]"""
    out = []
    for i, (x, ev) in enumerate(zip(xs, base_evs)):
        ev = D._ev(ev)
        fw, ts, labs = x.f
        for gi, g in enumerate(x.c["events"]):
            if not (is_salient_nonspeech(g["label"]) and not is_music(g["label"]) and g["consequential"]) or hit(ev, g):
                continue
            cols = fam_cols(labs, g["label"])
            if not cols:
                continue
            m = (ts >= g["start"] - 1.0) & (ts <= g["end"] + 1.0)
            if m.any() and 0.4 <= float(fw[m][:, cols].max()) < 0.8:
                out.append((i, gi))
    return out


def span_changes(xs, base_evs, evs):
    """false/true spans removed, gross new false spans (salient spans, primary filter)"""
    fr = tr = nf = 0
    for x, eb, ec in zip(xs, base_evs, evs):
        eb, ec = D._ev(eb), D._ev(ec)
        ovl = lambda a, bs: any(b.label == a.label and min(a.end, b.end) - max(a.start, b.start) > 0 for b in bs)
        for e in eb:
            if not ovl(e, ec):
                fr += is_false(e, x.c); tr += not is_false(e, x.c)
        fb = [e for e in eb if is_false(e, x.c)]
        nf += sum(1 for e in ec if is_false(e, x.c) and not ovl(e, fb))
    return {"false_removed": fr, "true_removed": tr, "new_false_gross": nf}


def score(xs, evs, base=None):
    """summary (primary 'lists' filter + secondary 'depictable'), deltas vs base = (base_evs, rows, rows_dep, flags)"""
    cl = [x.c for x in xs]
    rows = [D.clip_cost(c, ev) for c, ev in zip(cl, evs)]
    S = R4.summary(cl, rows)
    S["shown"] = int(sum(len(D._ev(ev)) for ev in evs))
    with label_filter("depictable"):
        rows_d = [D.clip_cost(c, ev) for c, ev in zip(cl, evs)]
        Sd = R4.summary(cl, rows_d)
        Sd["shown"] = int(sum(len(D._ev(ev)) for ev in evs))
    S["depictable"] = Sd
    if base is not None:
        base_evs, brows, brows_d, flags = base
        S["dC_overlap"] = boot8([a["C_overlap"] - b["C_overlap"] for a, b in zip(rows, brows)])
        S["dC_onset"] = boot8([a["C_onset"] - b["C_onset"] for a, b in zip(rows, brows)])
        Sd["dC_overlap"] = boot8([a["C_overlap"] - b["C_overlap"] for a, b in zip(rows_d, brows_d)])
        Sd["dC_onset"] = boot8([a["C_onset"] - b["C_onset"] for a, b in zip(rows_d, brows_d)])
        rec = sum(hit(D._ev(evs[i]), xs[i].c["events"][gi]) for i, gi in flags)
        S["heard_but_dropped"] = {"pool": len(flags), "recovered": int(rec)}
        S["recovered_all"] = int(sum(b["miss_overlap"] - a["miss_overlap"] for a, b in zip(rows, brows)))
        S.update(span_changes(xs, base_evs, evs))
        S["new_false_net"] = S["fp"] - sum(r["fp"] for r in brows)
    return rows, rows_d, S


def show(name, S):
    hb = S.get("heard_but_dropped", {})
    f = lambda v: "" if v is None else f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}] p {v[3]:.3f}"
    print(f"{name:12s} C-ov {S['C_overlap']:.3f} C-on {S['C_onset']:.3f} rec {S['recall_overlap']:.1%} on-rec {S['recall_onset']:.1%} "
          f"fp {S['fp']} ({S['fp_per_min']:.2f}/min) shown {S['shown']} | dC-ov {f(S.get('dC_overlap'))} dC-on {f(S.get('dC_onset'))} | "
          f"HBD {hb.get('recovered')}/{hb.get('pool')} rec-all {S.get('recovered_all')} new-false {S.get('new_false_gross')} "
          f"(net {S.get('new_false_net')}) removed false {S.get('false_removed')} true {S.get('true_removed')} | "
          f"depictable C-ov {S['depictable']['C_overlap']:.3f} C-on {S['depictable']['C_onset']:.3f}", flush=True)


# ============================================================================= gate 0
def gate_set(set_name, log):
    cl, xs = load_set(set_name)
    ref = base_run(xs)
    mine = [stack8(x) for x in xs]
    for x, a, b_ in zip(xs, ref, mine):
        assert sig(a) == sig(b_), f"stack8 differs from round 5 on {x.cid}"
    rows, rows_d, S = score(xs, ref)
    exp = EXPECT[set_name]
    ok = all(abs(S[k] - v) <= 1e-9 for k, v in exp.items())
    S5 = json.loads((_ROOT / "benchmark" / "detector_round5.json").read_text(encoding="utf-8"))[f"gate1_{set_name}"]["baseline_shipped_b"]
    ok5 = all(abs(S[k] - S5[k]) <= 1e-9 for k in ("C_overlap", "C_onset", "recall_overlap", "recall_onset", "fp_per_min"))
    log.setdefault("gate0", {})[set_name] = {"clips": len(cl), "baseline": S, "span_for_span": True, "expected_ok": ok, "round5_ok": ok5}
    show(f"base {set_name}", S)
    print(f"[gate0 {set_name}] {len(cl)} clips; span-for-span equal to round 5: True; expected numbers: {ok}; = round-5 gate 1: {ok5}")
    if not (ok and ok5):
        OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")
        sys.exit("gate 0 failed: stop")
    return cl, xs, ref, rows, rows_d, S


def gate(log):
    for s in ("calib", "heldout"):
        gate_set(s, log)
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


# ============================================================================= checks of the new caches
def check(log):
    res = {}
    for s in ("calib", "heldout"):
        cl, xs = load_set(s)
        miss, bad = [], []
        vj = json.loads(vlm_path().read_text(encoding="utf-8")) if vlm_path().exists() else {}
        for x in xs:
            for n in ("b1s", "blp", "bhp", "flp", "fhp"):
                try:
                    fr = x.get(n)
                except FileNotFoundError:
                    miss.append((x.cid, n)); continue
                ref = x.f if n.startswith("f") else x.b
                if n == "b1s":
                    ok = len(np.intersect1d(np.round(fr[1], 3), np.round(ref[1], 3))) >= len(ref[1]) - 1
                else:
                    ok = fr[0].shape == ref[0].shape and np.allclose(fr[1], ref[1], atol=1e-4) and list(fr[2]) == list(ref[2])
                if not ok:
                    bad.append((x.cid, n, fr[0].shape, ref[0].shape))
            if x.cid not in vj:
                miss.append((x.cid, "vlm"))
        cut = sum(v.get("cut", False) for k, v in vj.items() if k != "_meta")
        nf = [len(v["families"]) for k, v in vj.items() if k != "_meta"]
        res[s] = {"clips": len(cl), "missing": len(miss), "missing_first": miss[:5], "misaligned": len(bad), "bad_first": [list(map(str, b)) for b in bad[:5]],
                  "vlm_cut": int(cut), "vlm_families_median": float(np.median(nf)) if nf else None,
                  "vlm_families_range": [int(min(nf)), int(max(nf))] if nf else None}
        print(f"[check {s}] {res[s]}", flush=True)
    log["check"] = res
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


# ============================================================================= cell options
def i2_anchors(ev):
    an = {}
    for e in ev:
        an.setdefault(key(e), []).append((e.start, e.end))
    return an


def i3_src(x, d):
    z = x.get("z")
    q = 1.0 / (1.0 + np.exp(-(z - d + LOGIT035)))
    return q.astype(np.float32), x.b[1], x.b[2]


def i5_src(x, d1):
    fw2, t2, labs = x.b
    fw1, t1, _l = x.get("b1s")
    p1 = np.minimum(1.0, fw1 * (0.35 / d1))
    out = fw2.copy()
    j = {round(float(t), 3): i for i, t in enumerate(t1)}
    for i, t in enumerate(t2):
        k = j.get(round(float(t), 3))
        if k is not None:
            out[i] = np.maximum(out[i], p1[k])
    return out, t2, labs


def opts(cell, x, base_ev, prm):
    if cell.startswith("I2-"):
        _, b_, w = cell.split("-")
        return {"i2": (I2_BARS[b_], I2_W[w]), "anchors": i2_anchors(base_ev)}
    if cell == "I3":
        return {"bsrc": i3_src(x, prm["I3_d"])}
    if cell == "I4":
        return {"i4": True}
    if cell == "I5":
        s = i5_src(x, prm["I5_d1"]); return {"bsrc": s, "bveto": s}
    if cell == "I6":
        return {"i6": prm["vlm"][x.set][x.cid]["families"]}
    if cell == "I7":
        return {"i7": True}
    if cell == "I8":
        return {"bsrc": x.get("b8"), "bveto": x.get("b8"), "fsrc": x.get("f8")}
    if cell == "I9":
        return {"i9": True}
    if cell == "I10":
        return {"i10": True}
    if cell == "I1-i":
        return {"i1": "i"}
    if cell == "I1-ii":
        return {"i1": "ii"}
    if cell == "I1-iii":
        return {"i1": "iii", "shift": prm["I1_m"]}
    raise KeyError(cell)


def run_cell(cell, xs, base_evs, prm):
    return [stack8(x, opts(cell, x, be, prm)) for x, be in zip(xs, base_evs)]


# ============================================================================= refits on the 280
def fit_i3(xs, grid=None):
    grid = grid if grid is not None else [round(0.05 * i, 2) for i in range(401)]
    res = []
    for d in grid:
        fp = sum(D.clip_cost(x.c, stack8(x, {"bsrc": i3_src(x, d)}))["fp"] for x in xs)
        res.append((d, fp))
    return res


def pick_matched(res, target):
    return min(res, key=lambda r: (abs(r[1] - target), -r[0]))[0]


def fit_i5(xs):
    """d1 so that BEATs-1s ALONE gives the false-span count closest to BEATs-2s ALONE (0.175 / 0.35)"""
    def alone(fr, d):
        return [e for e in _extract_events(fr[0], fr[1], fr[2], d / 2, None, MIN_DUR, low=d / 2 * HYS) if e.confidence >= d]
    target = sum(D.clip_cost(x.c, alone(x.b, DISP))["fp"] for x in xs)
    res = [(round(0.35 + 0.01 * i, 2), sum(D.clip_cost(x.c, alone(x.get("b1s"), 0.35 + 0.01 * i))["fp"] for x in xs)) for i in range(66)]
    return target, res


def fit_i1(xs, base_evs):
    errs = []
    for x, ev in zip(xs, base_evs):
        ev = D._ev(ev)
        for g in conseq(x.c):
            m = [e for e in ev if same(e.label, g["label"]) and R.E._overlap_ok(e.start, e.end, g["start"], g["end"])]
            if m:
                best = min(m, key=lambda e: (-(min(e.end, g["end"]) - max(e.start, g["start"])), e.start))
                errs.append(best.start - g["start"])
    return float(np.median(errs)), len(errs)


# ============================================================================= steps
def fit(log, only=None):
    cl, xs, base_evs, brows, brows_d, Sb = gate_set("calib", log)
    flags = hbd_flags(xs, base_evs)
    base = (base_evs, brows, brows_d, flags)
    F = log.setdefault("fit", {})
    F["baseline"] = Sb
    F["heard_but_dropped_pool"] = len(flags)
    prm = log.setdefault("params", {})
    todo = list(only or RP_CELLS + T_CELLS)
    # ---- refits (written before the cells' costs)
    if "I3_d" not in prm and "I3" in todo:
        r = fit_i3(xs)
        d = pick_matched(r, Sb["fp"])
        if d in (r[0][0], r[-1][0]):                        # on a grid edge: extend by 10 in that direction (disclosed)
            ext = ([round(r[-1][0] + 0.05 * i, 2) for i in range(1, 201)] if d == r[-1][0]
                   else [round(r[0][0] - 0.05 * i, 2) for i in range(1, 201)])
            r = sorted(r + fit_i3(xs, ext)); d = pick_matched(r, Sb["fp"]); prm["I3_grid_extended"] = True
        prm["I3_d"] = d; prm["I3_grid_fp"] = r
        print(f"[refit] I3 d = {d} (fp {dict(r)[d]} vs baseline {Sb['fp']})", flush=True)
    if "I5_d1" not in prm and "I5" in todo and all((x.win / "round8_beats_1s" / f"{x.cid}.npz").exists() for x in xs):
        tgt, r = fit_i5(xs)
        prm["I5_d1"] = pick_matched(r, tgt); prm["I5_target_fp_2s_alone"] = tgt; prm["I5_grid_fp"] = r
        print(f"[refit] I5 d1 = {prm['I5_d1']} (1-s alone fp {dict(r)[prm['I5_d1']]} vs 2-s alone {tgt})", flush=True)
    if "I1_m" not in prm:
        m, n = fit_i1(xs, base_evs)
        prm["I1_m"] = m; prm["I1_m_n"] = n
        print(f"[refit] I1-iii m = {m:+.3f} s (median signed onset error, {n} hits)", flush=True)
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")
    prm_run = dict(prm)
    prm_run["vlm"] = {"calib": json.loads(vlm_path().read_text(encoding="utf-8"))} if vlm_path().exists() else {}
    for cell in todo:
        if cell == "I5" and "I5_d1" not in prm:
            print("[fit] I5 skipped: no 1-s caches yet"); continue
        if cell == "I6" and not all(x.cid in prm_run["vlm"].get("calib", {}) for x in xs):
            print("[fit] I6 skipped: VLM cache incomplete"); continue
        if cell == "I8" and not all((flex_view_dir("calib", v) / f"{x.cid}__{v}.npz").exists()
                                    and (x.win / f"round8_beats_{v}" / f"{x.cid}.npz").exists() for x in xs for v in ("lp", "hp")):
            print("[fit] I8 skipped: view caches incomplete"); continue
        evs = run_cell(cell, xs, base_evs, prm_run)
        _r, _rd, S = score(xs, evs, base)
        F[cell] = S; show(cell, S)
        OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")
    picks(log)


def picks(log):
    F, Sb = log["fit"], log["fit"]["baseline"]
    below = lambda k: k in F and F[k]["C_overlap"] < Sb["C_overlap"] and F[k]["C_onset"] < Sb["C_onset"]
    sent = []
    for b_ in I2_BARS:
        el = [f"I2-{b_}-{w}" for w in I2_W if below(f"I2-{b_}-{w}")]
        if el:
            sent.append(min(el, key=lambda k: (F[k]["C_overlap"], F[k]["C_onset"], list(I2_W).index(k.split("-")[2]))))
    sent += [k for k in RP_CELLS if not k.startswith("I2-") and below(k)]
    t_sent = [k for k in T_CELLS if k in F and F[k]["C_onset"] < Sb["C_onset"] and F[k]["C_overlap"] <= Sb["C_overlap"] + 0.05]
    log["picks"] = {"recall_precision": sent, "timing": t_sent,
                    "scored_280": [k for k in RP_CELLS + T_CELLS if k in F]}
    print(f"[picks] to the 415: recall/precision {sent}; timing {t_sent}", flush=True)
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


def holm(ps, alpha=0.025):
    order = sorted(ps, key=lambda k: ps[k])
    m, out, stop = len(order), {}, False
    for i, k in enumerate(order):
        thr = alpha / (m - i)
        rej = (not stop) and ps[k] <= thr
        stop = stop or not rej
        out[k] = {"p": ps[k], "threshold": thr, "reject": bool(rej)}
    return out


def heldout(log):
    P = log["picks"]
    cells = P["recall_precision"] + P["timing"]
    if not cells:
        log["heldout"] = "no cell picked on the 280"; OUT.write_text(json.dumps(log, indent=1), encoding="utf-8"); print(log["heldout"]); return
    cl, xs, base_evs, brows, brows_d, Sb = gate_set("heldout", log)
    base = (base_evs, brows, brows_d, hbd_flags(xs, base_evs))
    prm = dict(log["params"])
    prm["vlm"] = {"heldout": json.loads(vlm_path().read_text(encoding="utf-8"))} if vlm_path().exists() else {}
    H = log.setdefault("heldout", {})
    H["baseline"] = Sb
    for cell in cells:
        evs = run_cell(cell, xs, base_evs, prm)
        _r, _rd, S = score(xs, evs, base)
        if cell in P["timing"]:
            S["pass"] = bool(S["dC_onset"][2] < 0 and S["dC_overlap"][2] < 0.05)
        else:
            S["pass"] = bool(S["dC_overlap"][2] < 0)
        idx = {s: [i for i, x in enumerate(xs) if x.c.get("stratum") == s] for s in ("complex", "random")}
        d_ov = [a["C_overlap"] - b["C_overlap"] for a, b in zip(_r, brows)]
        S["strata"] = {s: {"n": len(ix), "dC_overlap": boot8([d_ov[i] for i in ix]) if ix else None} for s, ix in idx.items()}
        H[cell] = S; show(cell, S); print(f"   -> {'PASS' if S['pass'] else 'fail'}", flush=True)
        OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")
    if P["recall_precision"]:
        H["holm_recall_precision"] = holm({k: H[k]["dC_overlap"][3] for k in P["recall_precision"]})
    if P["timing"]:
        H["holm_timing"] = holm({k: H[k]["dC_onset"][3] for k in P["timing"]})
    print(f"[holm] {H.get('holm_recall_precision')} {H.get('holm_timing')}")
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("views", "beats", "vlm", "check", "gate", "fit", "picks", "heldout"))
    ap.add_argument("--set", choices=("calib", "heldout"))
    ap.add_argument("--cells", nargs="*")
    a = ap.parse_args()
    if a.step == "views":
        views(a.set); return
    if a.step == "beats":
        beats(a.set); return
    if a.step == "vlm":
        vlm(a.set); return
    log = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    if a.step == "fit":
        fit(log, a.cells)
    else:
        {"check": check, "gate": gate, "picks": picks, "heldout": heldout}[a.step](log)


if __name__ == "__main__":
    main()
