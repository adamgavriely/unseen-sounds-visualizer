"""Detector round 11 (docs/prereg_round11_masker.md, 2026-09-28): is a FlexSED 0.4-0.8 band candidate's evidence CAUSED by
the speech / music masker?

Cells: M1 / M1m / M1b foreign-masker dose-response, M2 argmax margin, M3 masker-envelope correlation, M5 two-regime shape,
M7 per-clip cap on the better of M1 / M2, C = M1 AND M2 + cap, W time reversal (transient families); A0 = all candidates
(reference). Reuses detector_round8 (stack, gate 0, scoring, bootstrap, Holm).

    python benchmark/detector_round11.py pool                  # CPU: the foreign speech / music pool from the 280
    python benchmark/detector_round11.py cands --set calib     # CPU: band candidates (caches only, no gold)
    python benchmark/detector_round11.py infer --set calib     # GPU: FlexSED re-run on orig / spk / mus / rev views
    python benchmark/detector_round11.py fit                   # gate 0, gate R0, all cells on the 280, picks
    python benchmark/detector_round11.py heldout               # the picks on the 415 (+ Holm)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark import detector_round8 as M
from src.labels import canonical, _parents, is_music, is_descendant, is_salient_nonspeech, SPEECH_LABELS
from src.stage4_audio_event_detection import _extract_events

R, D = M.R, M.D
key = M.key
OUT = _ROOT / "benchmark" / "detector_round11.json"
SR = 16000
FPS = 25.0
BAND, FBAR = 0.4, M.FBAR
M2_MARGIN, M3_R = 0.15, 0.7
M5_RISE, M5_WIN, M5_PROM, M5_CV = 0.3, 0.5, 0.2, 0.2
W_DROP = 0.2
MASK_BAR = 0.3
POOL_BARS = [(0.6, 0.1, 0.1), (0.5, 0.2, 0.2)]          # (main >=, other <, salient <); second = the one fallback
POOL_MIN_LEN, POOL_MIN_S, POOL_MIN_CLIPS, FADE = 2.0, 120.0, 10, 0.02
EXTRA_Q = ["Speech", "Music"]
TRANSIENT = ["Basketball bounce", "Burst, pop", "Camera", "Single-lens reflex camera", "Chink, clink", "Chop", "Clapping",
             "Clunk", "Coin (dropping)", "Cough", "Crack", "Dishes", "Dog", "Door", "Doorbell", "Ding-dong",
             "Drawer open or close", "Explosion", "Finger snapping", "Fireworks", "Glass", "Gunshot", "Hammer", "Hiccup",
             "Knock", "Snap", "Sneeze", "Sonic boom", "Specific impact sounds", "Splinter", "Tap", "Thump, thud", "Thunk"]
CELLS = ["M1", "M1m", "M1b", "M2", "M3", "M5", "M7", "C", "W"]
FLEXSED = Path(os.environ.get("FLEXSED_ROOT", Path.home() / "FlexSED"))


def rdir(set_name):
    return _ROOT / "data" / "work" / f"round11_flexsed_{set_name}"


def load_log():
    return json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}


def save_log(log):
    OUT.write_text(json.dumps(log, indent=1, default=float), encoding="utf-8")


def decode(mp4, wav):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-ac", "1", "-ar", str(SR), str(wav)], check=True)


# ============================================================================= band candidates (= round 10's definition)
def band_cands(x):
    fw, ts, labs = x.f
    bev = _extract_events(x.b[0], x.b[1], x.b[2], M.AED, None, M.MIN_DUR, low=M.AED * M.HYS)
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


def cands(set_name, log):
    cl, xs = M.load_set(set_name)
    out = {x.cid: [[e.label, e.start, e.end, e.confidence] for e in band_cands(x)] for x in xs}
    out = {k: v for k, v in out.items() if v}
    n = sum(len(v) for v in out.values())
    ntr = sum(canonical(c[0]) in TRANSIENT for v in out.values() for c in v)
    log.setdefault("cands", {})[set_name] = {"clips_total": len(cl), "clips_with": len(out), "n": n, "n_transient": ntr,
                                             "clips_with_transient": sum(any(canonical(c[0]) in TRANSIENT for c in v) for v in out.values()),
                                             "list": out}
    print(f"[cands {set_name}] {n} candidates in {len(out)} / {len(cl)} clips; transient {ntr}", flush=True)
    save_log(log)


# ============================================================================= the masker pool (from the 280)
DCASE = _ROOT / "data" / "dcase2025_task3"


def dcase_speech():
    """amendment 1: speech stretches from the DCASE2025 Task 3 Stereo SELD dev set (classes 0 / 1 only, >= 2 s)"""
    import csv
    st = []
    for wav in sorted((DCASE / "stereo_dev").glob("*/*.wav")):
        meta = DCASE / "metadata_dev" / wav.parent.name / f"{wav.stem}.csv"
        if not meta.exists():
            continue
        fr = {}
        with open(meta, newline="") as fh:
            for row in csv.DictReader(fh):
                fr.setdefault(int(row["frame"]), set()).add(int(row["class"]))
        ok = lambda k: k in fr and fr[k] and fr[k] <= {0, 1}
        ks = sorted(fr)
        i = 0
        while i < len(ks):
            if not ok(ks[i]):
                i += 1; continue
            j = i
            while j + 1 < len(ks) and ks[j + 1] == ks[j] + 1 and ok(ks[j + 1]):
                j += 1
            t0, t1 = ks[i] * 0.1, ks[j] * 0.1
            if t1 - t0 >= POOL_MIN_LEN - 1e-9:
                st.append([wav.stem, round(t0, 4), round(t1, 4), str(wav.relative_to(_ROOT)).replace("\\", "/")])
            i = j + 1
    tot = sum(b - a for _c, a, b, _p in st)
    ncl = len({s[0] for s in st})
    ok_pool = tot >= POOL_MIN_S and ncl >= POOL_MIN_CLIPS
    print(f"[pool speech, DCASE] {len(st)} stretches, {tot:.1f} s, {ncl} files -> ok {ok_pool}", flush=True)
    return {"source": "DCASE2025 Task3 Stereo SELD dev (amendment 1)", "seconds": round(tot, 2), "clips": ncl,
            "stretches": st, "ok": ok_pool}


def pool(log):
    R.use_set("calib")
    cl = D.usable()
    res = {}
    for kind, main, other in (("speech", "Speech", "Music"), ("music", "Music", "Speech")):
        for lvl, (b_main, b_other, b_sal) in enumerate(POOL_BARS):
            st = []
            for c in cl:
                fw, ts, labs = R.load(R.E.WIN / "beats" / f"{c['id']}.npz")
                im, io = labs.index(main), labs.index(other)
                sal = [i for i, l in enumerate(labs) if is_salient_nonspeech(l) and not is_music(l)]
                ok = (fw[:, im] >= b_main) & (fw[:, io] < b_other) & (fw[:, sal].max(axis=1) < b_sal)
                gold = [g for g in c["events"] if is_salient_nonspeech(g["label"]) and not is_music(g["label"])]
                i, n = 0, len(ok)
                while i < n:
                    if not ok[i]:
                        i += 1; continue
                    j = i
                    while j < n and ok[j]:
                        j += 1
                    t0, t1 = float(ts[i]), float(ts[j - 1])
                    if t1 - t0 >= POOL_MIN_LEN and not any(g["start"] < t1 + 0.5 and g["end"] > t0 - 0.5 for g in gold):
                        st.append([c["id"], round(t0, 4), round(t1, 4), f"data/input/audioset_calib/{c['id']}.mp4"])
                    i = j
            tot = sum(b - a for _c, a, b, _p in st)
            ncl = len({s[0] for s in st})
            ok_pool = tot >= POOL_MIN_S and ncl >= POOL_MIN_CLIPS
            res[kind] = {"source": "the 280", "bars": [b_main, b_other, b_sal], "fallback": lvl > 0, "seconds": round(tot, 2),
                         "clips": ncl, "stretches": st, "ok": ok_pool}
            print(f"[pool {kind}] bars {b_main}/{b_other}/{b_sal}: {len(st)} stretches, {tot:.1f} s, {ncl} clips -> ok {ok_pool}",
                  flush=True)
            if ok_pool:
                break
    res["speech_280_failed"] = {k: v for k, v in res["speech"].items() if k != "stretches"}
    res["speech"] = dcase_speech()                                     # amendment 1
    log["pool"] = res
    save_log(log)


def pool_audio(stretches):
    """cut every stretch from its source (16 kHz mono), RMS 1, 20-ms fades; [(src id, array)]"""
    import librosa
    import tempfile
    out, cache = [], {}
    with tempfile.TemporaryDirectory() as td:
        for cid, t0, t1, rel in stretches:
            if cid not in cache:
                src = _ROOT / rel
                if src.suffix.lower() == ".mp4":
                    w = Path(td) / f"{cid}.wav"
                    decode(src, w)
                    src = w
                cache[cid], _ = librosa.load(str(src), sr=SR, mono=True)
            a = cache[cid][int(round(t0 * SR)):int(round(t1 * SR))].astype(np.float64)
            rms = float(np.sqrt(np.mean(a ** 2))) if len(a) else 0.0
            if rms < 1e-6:
                continue
            a = a / rms
            nf = int(FADE * SR)
            ramp = np.linspace(0.0, 1.0, nf)
            a[:nf] *= ramp; a[-nf:] *= ramp[::-1]
            out.append((cid, a))
    return out


def foreign(cid, n, pieces, m):
    rng = np.random.default_rng([0, zlib.crc32(cid.encode()), m])
    el = [a for src, a in pieces if src != cid]
    order = rng.permutation(len(el))
    buf, tot = [], 0
    while tot < n:
        for k in order:
            buf.append(el[k]); tot += len(el[k])
            if tot >= n:
                break
    return np.concatenate(buf)[:n]


# ============================================================================= GPU: FlexSED re-run
def infer(set_name, log, shard=0, of=1, reverse=False):
    import librosa
    import soundfile as sf
    import tempfile
    vocab = json.loads(M.VOCAB.read_text(encoding="utf-8"))["families"]
    todo = sorted(log["cands"][set_name]["list"].items())[shard::of]
    if reverse:                                  # a second worker from the other end (each clip is skipped once done)
        todo = todo[::-1]
    R.use_set(set_name)
    vids = R.E.VIDEOS
    P = log["pool"]
    pieces = {k: pool_audio(P[k]["stretches"]) if P[k]["ok"] else None for k in ("speech", "music")}
    out = rdir(set_name); out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="r11_", dir=str(_ROOT / "data" / "work")))
    # ---- flexsed_run.py's setup, copied (the project's `src` must not shadow FlexSED's)
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _ROOT]
    for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
        sys.modules.pop(k, None)
    local_clap = Path.home() / "clap-htsat-unfused-st"
    if local_clap.exists():
        import transformers
        _clap, _tok = transformers.ClapTextModelWithProjection, transformers.AutoTokenizer
        _from_clap, _from_tok = _clap.from_pretrained, _tok.from_pretrained
        _clap.from_pretrained = staticmethod(lambda name, *x, **k: _from_clap(str(local_clap) if "clap" in str(name) else name, *x, **k))
        _tok.from_pretrained = staticmethod(lambda name, *x, **k: _from_tok(str(local_clap) if "clap" in str(name) else name, *x, **k))
    sys.path.insert(0, str(FLEXSED))
    os.chdir(FLEXSED)
    import torch
    from api import FlexSED
    mdl = FlexSED(device="cuda")
    _split = mdl.split_audio_fixed

    def split(audio, sr, chunk_duration=10.0):
        o = []
        for c in _split(audio, sr, chunk_duration):
            if len(c) < sr:
                c = np.pad(c, (0, sr - len(c)))
            o.append(c)
        return o
    mdl.split_audio_fixed = split
    _clap, _memo = mdl.clap, {}

    def clap(**inputs):                          # same CLAP text output, computed once per query text
        k = tuple(inputs["input_ids"][0].tolist())
        if k not in _memo:
            _memo[k] = _clap(**inputs)
        return _memo[k]
    mdl.clap = clap

    def run(wav, queries, batch=24):
        parts = []
        for k in range(0, len(queries), batch):
            with torch.inference_mode():
                p = mdl.run_inference(str(wav), queries[k:k + batch])
            parts.append(p.detach().squeeze(1).float().numpy())
            del p
            torch.cuda.empty_cache()
        return np.concatenate(parts, axis=0)

    def save(stem, fw, sm=None):
        d = {"fw": fw.astype(np.float32), "labels": np.array(vocab), "fps": FPS}
        if sm is not None:
            d["sm"] = sm.astype(np.float32)
        np.savez_compressed(out / f"{stem}.npz", **d)

    try:
        for i, (cid, cs) in enumerate(todo, 1):
            need_rev = any(canonical(c[0]) in TRANSIENT for c in cs)
            views = ["orig", "spk", "mus"] + (["rev"] if need_rev else [])
            if all((out / f"{cid}__{v}.npz").exists() for v in views):
                continue
            w0 = tmp / f"{cid}.wav"
            decode(vids / f"{cid}.mp4", w0)
            a, _ = librosa.load(str(w0), sr=SR)
            rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
            if not (out / f"{cid}__orig.npz").exists():
                save(f"{cid}__orig", run(w0, vocab), run(w0, EXTRA_Q))
            for v, kind, m in (("spk", "speech", 0), ("mus", "music", 1)):
                if (out / f"{cid}__{v}.npz").exists() or pieces[kind] is None or rms < 1e-4:
                    continue
                f = foreign(cid, len(a), pieces[kind], m)
                f = f * (rms / max(1e-12, float(np.sqrt(np.mean(f ** 2)))))
                wv = tmp / f"{cid}__{v}.wav"
                sf.write(str(wv), (a.astype(np.float64) + f).astype(np.float32), SR, subtype="FLOAT")
                save(f"{cid}__{v}", run(wv, vocab))
                wv.unlink()
            if need_rev and not (out / f"{cid}__rev.npz").exists():
                wv = tmp / f"{cid}__rev.wav"
                sf.write(str(wv), np.ascontiguousarray(a[::-1]).astype(np.float32), SR, subtype="FLOAT")
                save(f"{cid}__rev", run(wv, vocab))
                wv.unlink()
            w0.unlink()
            print(f"[infer {set_name}] {i}/{len(todo)} {cid} views {views} rms {rms:.4f}", flush=True)
    finally:
        for p in tmp.glob("*"):
            p.unlink()
        tmp.rmdir()
    print(f"[infer {set_name}] done -> {out}", flush=True)


# ============================================================================= features per candidate
def ld(p):
    z = np.load(p, allow_pickle=False)
    fw = z["fw"].astype(np.float32).T                          # [T, 215]
    sm = z["sm"].astype(np.float32).T if "sm" in z else None     # [T, 2]
    return fw, sm


def span_idx(ts, e):
    m = (ts >= e.start) & (ts <= e.end)
    if not m.any():
        m = np.zeros(len(ts), bool); m[int(np.argmin(np.abs(ts - 0.5 * (e.start + e.end))))] = True
    return np.where(m)[0]


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def competitors(labs, lab):
    """M2's competitor columns for candidate label `lab`"""
    par = _parents()
    p = par.get(lab)
    k = canonical(lab)
    out = []
    for j, l in enumerate(labs):
        if canonical(l) == k or M.same(l, lab):
            continue
        if p is not None and par.get(l) == p:
            continue
        if l in SPEECH_LABELS or any(is_descendant(l, s) for s in SPEECH_LABELS) or is_music(l):
            continue
        out.append(j)
    return out


def features(x, e, rd, gate):
    fw, ts, labs = x.f
    cols = M.cols_for(labs, key(e))
    ix = span_idx(ts, e)
    s = fw[:, cols].max(axis=1)
    f = {"label": e.label, "start": e.start, "end": e.end, "peak": e.confidence}
    # M2 on the cached scores
    comp = competitors(labs, e.label)
    f["m2_margin"] = float(np.mean(s[ix] - fw[ix][:, comp].max(axis=1))) if comp else float(np.mean(s[ix]))
    # re-run views
    o_fw, o_sm = ld(rd / f"{x.cid}__orig.npz")
    assert o_fw.shape == fw.shape, f"frame count {x.cid}: {o_fw.shape} vs {fw.shape}"
    if x.cid not in gate:
        gate[x.cid] = float(np.abs(o_fw - fw).max())
    so = o_fw[:, cols].max(axis=1)
    mo = float(so[ix].mean())
    f["mean_orig"] = mo
    for v in ("spk", "mus"):
        p = rd / f"{x.cid}__{v}.npz"
        f[f"d_{v}"] = (float(ld(p)[0][:, cols].max(axis=1)[ix].mean()) - mo) if p.exists() else None
    # M3: cached family curve vs re-run Speech / Music query curves
    f["r_s"], f["r_m"] = pearson(s[ix], o_sm[ix, 0]), pearson(s[ix], o_sm[ix, 1])
    # M5
    rise = max(float(s[i] - s[(ts >= ts[i] - M5_WIN) & (ts <= ts[i])].min()) for i in ix)
    from scipy.signal import peak_prominences
    pk = int(ix[int(np.argmax(s[ix]))])
    f["rise"], f["prom"] = rise, float(peak_prominences(s, [pk])[0][0])
    mu = float(s[ix].mean())
    f["cv"] = float(s[ix].std() / mu) if mu > 0 else float("inf")
    # W
    f["transient"] = canonical(e.label) in TRANSIENT
    f["d_rev"] = None
    p = rd / f"{x.cid}__rev.npz"
    if f["transient"] and p.exists():
        r_fw = ld(p)[0][:, cols].max(axis=1)
        L = x_len(x)
        j = np.clip(np.rint((L - ts - 1.0 / FPS) * FPS).astype(int), 0, len(r_fw) - 1)
        f["d_rev"] = float(r_fw[j][ix].mean()) - mo            # rev - orig (admit iff orig - rev >= 0.2)
    # masked
    bl = x.b[2]
    f["masked"] = M.peak_in(x.b, [bl.index("Speech"), bl.index("Music")], e.start, e.end) >= MASK_BAR
    f["true"] = not M.is_false(e, x.c)
    return f


_LEN = {}


def x_len(x):
    """clip length in seconds (ffprobe of the mp4's audio, cached)"""
    if x.cid not in _LEN:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration", "-of",
                            "default=nw=1:nk=1", str(R.E.VIDEOS / f"{x.cid}.mp4")], capture_output=True, text=True)
        try:
            _LEN[x.cid] = float(r.stdout.strip().splitlines()[0])
        except (ValueError, IndexError):
            _LEN[x.cid] = len(x.f[1]) / FPS
    return _LEN[x.cid]


# ============================================================================= rules
def m1(f):
    return f["d_spk"] is not None and f["d_spk"] < 0


def m1m(f):
    return f["d_mus"] is not None and f["d_mus"] < 0


def m3(f):
    return f["r_s"] <= M3_R and f["r_m"] <= M3_R


def m5(f):
    return (f["rise"] > M5_RISE and f["prom"] > M5_PROM) or (f["cv"] < M5_CV and m3(f))


RULE = {"A0": lambda f: True, "M1": m1, "M1m": m1m, "M1b": lambda f: m1(f) and m1m(f),
        "M2": lambda f: f["m2_margin"] >= M2_MARGIN, "M3": m3, "M5": m5,
        "C": lambda f: m1(f) and f["m2_margin"] >= M2_MARGIN,
        "W": lambda f: f["transient"] and f["d_rev"] is not None and -f["d_rev"] >= W_DROP}


def cap(items):
    """items: [(event, feat)] of one clip -> the one with the largest M2 margin"""
    if not items:
        return []
    return [min(items, key=lambda t: (-t[1]["m2_margin"], -t[1]["peak"], t[1]["start"]))]


# ============================================================================= scoring
def prepare(set_name, log):
    cl, xs, base_evs, brows, brows_d, Sb = M.gate_set(set_name, log)
    flags = M.hbd_flags(xs, base_evs)
    base = (base_evs, brows, brows_d, flags)
    rd = rdir(set_name)
    ref = log["cands"][set_name]["list"]
    gate, C = {}, []
    for x in xs:
        cs = band_cands(x)
        assert [[e.label, e.start, e.end, e.confidence] for e in cs] == ref.get(x.cid, []), f"candidates changed on {x.cid}"
        C.append([(e, features(x, e, rd, gate)) for e in cs])
    worst = max(gate.values()) if gate else 0.0
    log.setdefault("gateR0", {})[set_name] = {"clips": len(gate), "max_abs_diff": worst,
                                              "median_max_abs_diff": float(np.median(list(gate.values()))) if gate else 0.0,
                                              "pass": bool(worst < 0.02)}
    print(f"[gate R0 {set_name}] {log['gateR0'][set_name]}", flush=True)
    if worst >= 0.02:
        save_log(log); sys.exit("gate R0 failed: stop")
    return xs, base_evs, base, flags, C, Sb


def masked_ev(x, t0, t1):
    bl = x.b[2]
    return M.peak_in(x.b, [bl.index("Speech"), bl.index("Music")], t0, t1) >= MASK_BAR


def run_cell(cell, xs, base_evs, C, best=None):
    evs, adm = [], []
    for x, be, cs in zip(xs, base_evs, C):
        if cell in ("M7", "C"):
            r = RULE[best] if cell == "M7" else RULE["C"]
            ok = cap([(e, f) for e, f in cs if r(f)])
        else:
            ok = [(e, f) for e, f in cs if RULE[cell](f)]
        adm += [f for _e, f in ok]
        evs.append(list(be) + [e for e, _f in ok])
    return evs, adm


def extras(xs, base_evs, evs, flags, adm, C):
    """masked / unmasked splits, candidate table"""
    out = {}
    pool_m = [masked_ev(xs[i], xs[i].c["events"][gi]["start"], xs[i].c["events"][gi]["end"]) for i, gi in flags]
    rec = [M.hit(D._ev(evs[i]), xs[i].c["events"][gi]) for i, gi in flags]
    out["hbd"] = {"pool_masked": int(sum(pool_m)), "pool_unmasked": int(len(pool_m) - sum(pool_m)),
                  "rescued_masked": int(sum(r and m for r, m in zip(rec, pool_m))),
                  "rescued_unmasked": int(sum(r and not m for r, m in zip(rec, pool_m)))}
    nm = nu = 0
    for x, be, ev in zip(xs, base_evs, evs):
        add = ev[len(be):]
        for e in D._ev(add):
            if M.is_false(e, x.c):
                if masked_ev(x, e.start, e.end):
                    nm += 1
                else:
                    nu += 1
    out["false_added"] = {"masked": nm, "unmasked": nu}
    allf = [f for cs in C for _e, f in cs]
    tab = lambda fs: {f"{'true' if t else 'false'}_{'masked' if m else 'unmasked'}":
                      sum(1 for f in fs if f["true"] == t and f["masked"] == m) for t in (True, False) for m in (True, False)}
    out["candidates"] = {"considered": len(allf), "admitted": len(adm), "considered_split": tab(allf), "admitted_split": tab(adm)}
    return out


def m1_table(C):
    allf = [f for cs in C for _e, f in cs]
    out = {}
    for v in ("spk", "mus"):
        t = {}
        for tr in (True, False):
            for m in (True, False):
                g = [f for f in allf if f["true"] == tr and f["masked"] == m and f[f"d_{v}"] is not None]
                t[f"{'true' if tr else 'false'}_{'masked' if m else 'unmasked'}"] = {
                    "n": len(g), "drop": sum(f[f"d_{v}"] < 0 for f in g),
                    "median_delta": float(np.median([f[f"d_{v}"] for f in g])) if g else None}
        out[v] = t
    return out


def score_cell(cell, xs, base_evs, base, flags, C, best=None, strata=False):
    evs, adm = run_cell(cell, xs, base_evs, C, best)
    rows, _rd, S = M.score(xs, evs, base)
    assert S["true_removed"] == 0 and S["false_removed"] == 0, f"{cell} removed spans"
    S.update(extras(xs, base_evs, evs, flags, adm, C))
    if strata:
        d = [a["C_overlap"] - b["C_overlap"] for a, b in zip(rows, base[1])]
        S["strata"] = {s: {"n": len(ix), "dC_overlap": M.boot8([d[i] for i in ix]) if ix else None}
                       for s, ix in ((s, [i for i, x in enumerate(xs) if x.c.get("stratum") == s]) for s in ("complex", "random"))}
    M.show(cell, S)
    print(f"   hbd {S['hbd']} false added {S['false_added']} cands {S['candidates']['admitted']}/{S['candidates']['considered']} "
          f"adm {S['candidates']['admitted_split']}", flush=True)
    return S


def fit(log):
    xs, base_evs, base, flags, C, Sb = prepare("calib", log)
    F = log.setdefault("fit", {})
    F["baseline"] = Sb
    F["feature_rows"] = [dict(f, clip=x.cid) for x, cs in zip(xs, C) for _e, f in cs]
    F["m1_table"] = m1_table(C)
    print(f"[M1 table] {json.dumps(F['m1_table'])}", flush=True)
    for cell in ["A0", "M1", "M1m", "M1b", "M2", "M3", "M5"]:
        F[cell] = score_cell(cell, xs, base_evs, base, flags, C); save_log(log)
    b1, b2 = F["M1"], F["M2"]
    best = "M1" if (b1["C_overlap"], b1["C_onset"]) < (b2["C_overlap"], b2["C_onset"]) else "M2"
    log["m7_base"] = best
    print(f"[M7] on top of {best}", flush=True)
    F["M7"] = score_cell("M7", xs, base_evs, base, flags, C, best); save_log(log)
    for cell in ["C", "W"]:
        F[cell] = score_cell(cell, xs, base_evs, base, flags, C); save_log(log)
    log["picks"] = [k for k in CELLS if F[k]["C_overlap"] < Sb["C_overlap"] and F[k]["C_onset"] < Sb["C_onset"]]
    print(f"[picks] to the 415: {log['picks']}", flush=True)
    save_log(log)


def heldout(log):
    if not log.get("picks"):
        log["heldout"] = "no cell picked on the 280"; print(log["heldout"]); save_log(log); return
    audit = _ROOT / "docs" / "setup_audit_2026-09-28.md"
    assert audit.exists(), "setup audit not written yet: the 415 waits"
    xs, base_evs, base, flags, C, Sb = prepare("heldout", log)
    H = log.setdefault("heldout", {})
    H["baseline"] = Sb
    H["m1_table"] = m1_table(C)
    for cell in ["A0"] + log["picks"]:
        S = score_cell(cell, xs, base_evs, base, flags, C, log.get("m7_base"), strata=True)
        if cell != "A0":
            S["pass"] = bool(S["dC_overlap"][2] < 0)
            print(f"   -> {'PASS' if S['pass'] else 'fail'}", flush=True)
        H[cell] = S; save_log(log)
    H["holm"] = M.holm({k: H[k]["dC_overlap"][3] for k in log["picks"]})
    print(f"[holm] {H['holm']}", flush=True)
    save_log(log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("pool", "cands", "infer", "fit", "heldout"))
    ap.add_argument("--set", choices=("calib", "heldout"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    log = load_log()
    if a.step == "pool":
        pool(log)
    elif a.step == "cands":
        cands(a.set, log)
    elif a.step == "infer":
        infer(a.set, log, a.shard, a.of, a.reverse)
    else:
        {"fit": fit, "heldout": heldout}[a.step](log)


if __name__ == "__main__":
    main()
