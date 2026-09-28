"""Detector round 7 (docs/prereg_round7_samaudio.md, 2026-09-28): SAM-Audio removes speech and music, the shipped
detectors (BEATs + FlexSED, shipped bars and vetoes, no refit) listen again to what is left, and the residual view's
spans are added to the shipped stack. Cost C, clip sets and bootstrap as amendment 24 (detector_round2.py); baseline =
round 5's shipped stack at b = 0.1218 (detector_round5.run).

    python benchmark/detector_round7.py sep --set calib|heldout [--shard i --of n]   # env sota, GPU (SAM-Audio)
    python benchmark/detector_round7.py qc                                            # env msproj, GPU: picks the route
    python benchmark/detector_round7.py cache-beats --set calib|heldout --route res|ident
    python benchmark/detector_round7.py cache-flex  --set calib|heldout --route res|ident   (own process: FlexSED's src)
    python benchmark/detector_round7.py identity     # gate: the route with r := x reproduces the shipped spans
    python benchmark/detector_round7.py fit          # baseline gate + S1, S2 on the 280 + the pick
    python benchmark/detector_round7.py heldout      # the pick on the 415 (only if there is one)

The module top imports only the standard library and numpy: `sep` runs in the sota env, `cache-flex` must import
FlexSED's own top-level `src` package; project modules are imported inside the steps that need them.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
# R7_TAG=7 (default): round 7 as run. R7_TAG=7b: the retry pre-registered in the same doc (descriptive queries,
# predict_spans=True with PE-A-Frame-large, erase check); own json, own folders.
TAG = os.environ.get("R7_TAG", "7")
assert TAG in ("7", "7b"), TAG
SFX = "r" + TAG                                          # folder suffix: r7 / r7b
OUT = _ROOT / "benchmark" / f"detector_round{TAG}.json"
SEP = _ROOT / "data" / "work" / f"{SFX}_samaudio"      # SEP/<set>/{arith,stem,ident,qc}/<id>.{wav,json}
MODEL_ID = os.environ.get("R7_MODEL", "facebook/sam-audio-large")
QUERIES = ("speech", "music") if TAG == "7" else ("a person talking", "background music")
PREDICT_SPANS = TAG == "7b"
ERASE_DROP, ERASE_SHARE, ERASE_MIN_ORIG = 0.3, 0.25, 0.3
SR_SEP, SR_DET = 48000, 16000
HOP48 = 1920                                             # SAM-Audio codec hop at 48 kHz
B_SHIPPED = 0.1218
N_QC, N_IDENT = 10, 20
EXPECT = {"calib": {"C_overlap": 3.0357, "C_onset": 3.85, "recall_overlap": 0.5134, "fp_per_min": 4.4357},
          "heldout": {"C_overlap": 1.9277, "C_onset": 2.2072, "recall_overlap": 0.4971, "fp_per_min": 3.2964}}
TOL = {"C_overlap": 0.0005, "C_onset": 0.0005, "recall_overlap": 0.0005, "fp_per_min": 0.005}
VOCAB = _ROOT / "benchmark" / "gold" / "depictable_vocab.json"
FLEXSED = Path(os.environ.get("FLEXSED_ROOT", Path.home() / "FlexSED"))


def win_dir(set_name):
    return _ROOT / "benchmark" / f"audioset_{set_name}_windows"


def vid_dir(set_name):
    return _ROOT / "data" / "input" / f"audioset_{set_name}"


def beats_dir(set_name, route):
    return win_dir(set_name) / f"beats_{SFX}{route}"


def flex_dir(set_name, route):
    return _ROOT / "data" / "work" / f"flexsed_{set_name}_{SFX}{route}"


def load_log():
    return json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}


def save_log(log):
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


# ============================================================================= separation (env sota)
def ffmpeg_read(mp4, sr):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(mp4), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def write16(dst, x):
    """float32 48 kHz -> mono 16 kHz 16-bit wav through ffmpeg (the original caches' route: ffmpeg -> 16 kHz wav)"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as fh:
        fh.write(np.asarray(x, np.float32).tobytes()); tmp = fh.name
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "f32le", "-ar", str(SR_SEP), "-ac", "1", "-i", tmp,
                        "-ac", "1", "-ar", str(SR_DET), str(dst)], check=True)
    finally:
        os.unlink(tmp)


def fit_len(y, n):
    """trim or zero-pad at the END to n samples; never shift"""
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def xcorr_lag(x, t, maxlag=2400):
    if np.sqrt(np.mean(t ** 2)) < 1e-5:
        return None
    n = 1 << int(np.ceil(np.log2(2 * len(x))))
    c = np.fft.irfft(np.fft.rfft(x, n) * np.conj(np.fft.rfft(t, n)), n)
    lags = np.r_[np.arange(0, maxlag + 1), np.arange(-maxlag, 0)]
    vals = np.r_[c[:maxlag + 1], c[-maxlag:]]
    return int(lags[int(np.argmax(vals))])


def stub_torchcodec():
    """torchcodec cannot load FFmpeg in the sota env; the processor imports it but is given tensors, so it is never called"""
    import importlib.machinery as M
    tc = types.ModuleType("torchcodec"); dec = types.ModuleType("torchcodec.decoders")
    tc.__spec__ = M.ModuleSpec("torchcodec", None, is_package=True); dec.__spec__ = M.ModuleSpec("torchcodec.decoders", None)

    class _No:
        def __init__(self, *a, **k):
            raise RuntimeError("torchcodec is stubbed in round 7 (audio is passed as tensors)")
    dec.AudioDecoder = dec.VideoDecoder = _No; tc.decoders = dec
    sys.modules["torchcodec"] = tc; sys.modules["torchcodec.decoders"] = dec


def sep(set_name, shard=0, of=1, limit=None):
    import torch
    stub_torchcodec()
    from sam_audio import SAMAudio, SAMAudioProcessor
    ids = sorted(p.stem for p in (win_dir(set_name) / "beats").glob("*.npz") if (vid_dir(set_name) / f"{p.stem}.mp4").exists())
    ids = ids[shard::max(1, of)][:limit]
    out = SEP / set_name
    for d in ("arith", "stem", "ident", "qc"):
        (out / d).mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    t0 = time.time()
    model = SAMAudio.from_pretrained(MODEL_ID, visual_ranker=None, text_ranker=None,
                                     span_predictor="pe-a-frame-large" if PREDICT_SPANS else None).to(dev).eval()
    proc = SAMAudioProcessor.from_pretrained(MODEL_ID)
    assert proc.audio_sampling_rate == SR_SEP, proc.audio_sampling_rate
    print(f"[sep] {MODEL_ID} loaded in {time.time() - t0:.0f} s; {torch.cuda.get_device_name(0)}; set {set_name} shard {shard}/{of}: "
          f"{len(ids)} clips; queries {QUERIES}; predict_spans {PREDICT_SPANS}", flush=True)

    def separate(a, q):
        torch.manual_seed(0)
        batch = proc(descriptions=[q], audios=[torch.from_numpy(np.ascontiguousarray(a))[None]]).to(dev)
        r = model.separate(batch, predict_spans=PREDICT_SPANS, reranking_candidates=1)
        t, s = r.target[0].float().cpu().numpy(), r.residual[0].float().cpu().numpy()
        return t, s

    done = 0
    for cid in ids:
        qj = out / "qc" / f"{cid}.json"
        if qj.exists():
            continue
        ts0 = time.time()
        x = ffmpeg_read(vid_dir(set_name) / f"{cid}.mp4", SR_SEP)
        n = len(x)
        t_s, st_s = separate(x, QUERIES[0])
        raw = {"speech_target": len(t_s), "speech_stem": len(st_s)}
        t_s, st_s = fit_len(t_s, n), fit_len(st_s, n)
        r1 = x - t_s                                                          # arithmetic route
        t_m, _ = separate(r1, QUERIES[1])
        raw["music_target_arith"] = len(t_m)
        t_m = fit_len(t_m, n)
        r_arith = r1 - t_m
        _, st_m = separate(st_s, QUERIES[1])                                  # stem route
        raw["music_stem_stem"] = len(st_m)
        r_stem = fit_len(st_m, n)
        write16(out / "ident" / f"{cid}.wav", x)
        write16(out / "arith" / f"{cid}.wav", r_arith)
        write16(out / "stem" / f"{cid}.wav", r_stem)
        rms = lambda v: float(np.sqrt(np.mean(np.square(v, dtype=np.float64))))
        qc = {"n_in": n, "raw_len": raw, "len_ok": all(abs(v - n) <= HOP48 for v in raw.values()),
              "lag_speech": xcorr_lag(x, t_s), "lag_music": xcorr_lag(r1, t_m),
              "rms": {"x": rms(x), "t_speech": rms(t_s), "t_music": rms(t_m), "r_arith": rms(r_arith), "stem_speech": rms(st_s),
                      "r_stem": rms(r_stem)},
              "peak": {"x": float(np.abs(x).max()), "r_arith": float(np.abs(r_arith).max()), "r_stem": float(np.abs(r_stem).max())},
              "nan": bool(np.isnan(r_arith).any() or np.isnan(r_stem).any()), "secs": round(time.time() - ts0, 2),
              "model": MODEL_ID, "queries": list(QUERIES), "predict_spans": PREDICT_SPANS}
        qj.write_text(json.dumps(qc), encoding="utf-8")
        done += 1
        if done <= 3 or done % 20 == 0:
            print(f"[sep] {done} {cid} {qc['secs']} s rms x {qc['rms']['x']:.4f} t_s {qc['rms']['t_speech']:.4f} t_m {qc['rms']['t_music']:.4f} "
                  f"arith {qc['rms']['r_arith']:.4f} stem {qc['rms']['r_stem']:.4f} lag {qc['lag_speech']}/{qc['lag_music']}", flush=True)
    print(f"[sep] {set_name} shard {shard}/{of}: {done} clips separated now, {time.time() - t0:.0f} s", flush=True)


# ============================================================================= project imports (env msproj)
def proj():
    sys.path.insert(0, str(_ROOT))
    import config  # noqa: F401
    from benchmark import audioset_stage4_report as R
    from benchmark import detector_round2 as D
    from benchmark import detector_round3 as R3
    from benchmark import detector_round4 as R4
    from benchmark import detector_round5 as R5
    return R, D, R3, R4, R5


def qc_clips(R, D):
    R.use_set("calib")
    out = []
    for c in D.usable():
        fw, _t, labs = R.load(R.E.WIN / "beats" / f"{c['id']}.npz")
        pk = dict(zip(labs, fw.max(axis=0)))
        if max(pk["Speech"], pk["Music"]) >= 0.6:
            out.append(c)
        if len(out) == N_QC:
            break
    return out


WINDOW_MISMATCH = []


def beats_one(wav, ref_times):
    """infer_beats on a 16 kHz wav; a window grid that differs from the original cache's is counted and reported (the
    stack reads each cache's own times), not an error"""
    from src.stage4_audio_event_detection.beats_infer import infer_beats
    fw, times, labels = infer_beats(wav, "cuda")
    if not (len(times) == len(ref_times) and np.allclose(times, ref_times, atol=1e-4)):
        WINDOW_MISMATCH.append(Path(wav).stem)
        print(f"[beats] window grid differs from the original cache on {Path(wav).stem}: {len(times)} vs {len(ref_times)}", flush=True)
    return fw, times, labels


def qc(log):
    """BEATs on both routes' residuals for the QC clips; the declared checks; the route (prereg: QC gate)"""
    R, D, *_ = proj()
    import librosa
    cl = qc_clips(R, D)
    assert len(cl) == N_QC, len(cl)
    res = {"clips": [c["id"] for c in cl]}
    from src.labels import is_salient_nonspeech, is_music
    for route in ("arith", "stem"):
        rows, erase_all = [], []
        dst = beats_dir("calib", f"qc_{route}"); dst.mkdir(parents=True, exist_ok=True)
        for c in cl:
            cid = c["id"]
            q = json.loads((SEP / "calib" / "qc" / f"{cid}.json").read_text(encoding="utf-8"))
            ofw, ot, ol = R.load(R.E.WIN / "beats" / f"{cid}.npz")
            wav = SEP / "calib" / route / f"{cid}.wav"
            fw, t, labs = beats_one(wav, ot)
            np.savez_compressed(dst / f"{cid}.npz", fw=fw.astype(np.float16), times=np.asarray(t, np.float32), labels=np.array(labs))
            assert labs == ol
            si, mi = ol.index("Speech"), ol.index("Music")
            opk, rpk = ofw.max(axis=0), fw.max(axis=0)
            dom = si if opk[si] >= opk[mi] else mi
            drop = float(opk[dom] - rpk[dom])
            xin, _ = librosa.load(str(SEP / "calib" / "ident" / f"{cid}.wav"), sr=16000, mono=True)
            xr, _ = librosa.load(str(wav), sr=16000, mono=True)
            n = min(len(xin), len(xr))
            m = np.zeros(n, bool)
            for k in np.flatnonzero(np.maximum(ofw[:, si], ofw[:, mi]) >= 0.5):
                a, b = int(max(0, (ot[k] - 1.5) * 16000)), int(min(n, (ot[k] + 0.5) * 16000))
                m[a:b] = True
            ratio = float(np.sqrt(np.mean(xr[:n][m] ** 2)) / max(1e-9, np.sqrt(np.mean(xin[:n][m] ** 2)))) if m.any() else None
            erase = []
            for g in c["events"]:
                if not (is_salient_nonspeech(g["label"]) and not is_music(g["label"])):
                    continue
                ge = types.SimpleNamespace(start=g["start"], end=g["end"], label=g["label"])
                po = D.peak_near((ofw, ot, ol), ge)
                if po >= ERASE_MIN_ORIG:
                    pr = D.peak_near((fw, t, labs), ge)
                    erase.append({"label": g["label"], "start": g["start"], "orig": float(po), "res": float(pr),
                                  "drop": float(po - pr), "erased": bool(po - pr > ERASE_DROP)})
            erase_all.extend(erase)
            rows.append({"id": cid, "len_ok": bool(q["len_ok"] and len(xr) == len(xin)), "lag_speech": q["lag_speech"],
                         "lag_music": q["lag_music"], "dominant": ol[dom], "orig_clipmax": float(opk[dom]),
                         "res_clipmax": float(rpk[dom]), "drop": drop, "drop_ok": drop >= 0.3, "rms_ratio_speechy": ratio})
        a_ok = sum(r["len_ok"] for r in rows) == N_QC
        b_n = sum(r["drop_ok"] for r in rows)
        ratios = [r["rms_ratio_speechy"] for r in rows if r["rms_ratio_speechy"] is not None]
        c_med = float(np.median(ratios)) if ratios else None
        passed = bool(a_ok and b_n >= 6 and c_med is not None and c_med < 1.0)
        res[route] = {"rows": rows, "a_len_ok_10_of_10": a_ok, "b_drop_ok": b_n, "c_median_rms_ratio": c_med, "pass": passed}
        print(f"[qc] {route}: len ok {a_ok}; dominant class falls >= 0.3 on {b_n}/10; median RMS ratio in speech/music "
              f"windows {c_med}; gate 1 PASS {passed}", flush=True)
        if TAG == "7b":                                      # gate 2: the erase check (prereg, round 7b)
            n_er = sum(e["erased"] for e in erase_all)
            ok2 = bool(erase_all) and n_er <= ERASE_SHARE * len(erase_all)
            res[route]["erase"] = {"events": erase_all, "counted": len(erase_all), "erased": n_er,
                                   "median_drop": float(np.median([e["drop"] for e in erase_all])) if erase_all else None,
                                   "pass": ok2}
            res[route]["gate1_pass"] = passed
            passed = bool(passed and ok2)
            res[route]["pass"] = passed
            print(f"[qc] {route}: erase check: {n_er}/{len(erase_all)} counted gold events lose > {ERASE_DROP} "
                  f"(allowed <= {ERASE_SHARE:.0%}); median drop {res[route]['erase']['median_drop']}; gate 2 PASS {ok2}", flush=True)
            for e in erase_all:
                print(f"   erase {e}", flush=True)
        for r in rows:
            print(f"   {r}", flush=True)
    route = "arith" if res["arith"]["pass"] else ("stem" if res["stem"]["pass"] else None)
    res["route"] = route
    log["qc"] = res
    save_log(log)
    print(f"[qc] route chosen by the declared rule: {route}", flush=True)
    if route is None:
        sys.exit(f"QC: neither route passes -> round {TAG} stops here (as pre-registered)")


def src_route(set_name, route, log):
    if route == "ident":
        return "ident"
    r = log.get("qc", {}).get("route")
    assert r in ("arith", "stem"), "run qc first (no route chosen)"
    return r


def cache_ids(set_name, route):
    R, D, *_ = proj()
    R.use_set(set_name)
    cl = D.usable()
    return cl[:N_IDENT] if route == "ident" else cl


def cache_beats(set_name, route, log):
    R, D, *_ = proj()
    srt = src_route(set_name, route, log)
    cl = cache_ids(set_name, route)
    R.use_set(set_name)
    dst = beats_dir(set_name, route); dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for c in cl:
        f = dst / f"{c['id']}.npz"
        if f.exists():
            continue
        _ofw, ot, ol = R.load(R.E.WIN / "beats" / f"{c['id']}.npz")
        fw, t, labs = beats_one(SEP / set_name / srt / f"{c['id']}.wav", ot)
        assert labs == ol
        np.savez_compressed(f, fw=fw.astype(np.float16), times=np.asarray(t, np.float32), labels=np.array(labs)); n += 1
    print(f"[cache-beats] {set_name} {route} (audio route {srt}): {len(cl)} clips, {n} scored now -> {dst}; window-grid "
          f"mismatches vs original cache: {len(WINDOW_MISMATCH)} {WINDOW_MISMATCH[:10]}", flush=True)


def cache_flex(set_name, route, log, batch=24):
    """FlexSED exactly as benchmark/gold/flexsed_run.py (215 families, no prompt ensemble, batch 24, inference_mode, the
    1-s tail pad), but fed the residual wav path directly (flexsed_run would reuse data/work/gold_wav_flat/<stem>.wav)"""
    srt = src_route(set_name, route, log)
    # clip ids first (project imports), then drop the project's `src` so FlexSED's own package can be imported
    ids = [c["id"] for c in cache_ids(set_name, route)]
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _ROOT]
    for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
        sys.modules.pop(k)
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))["families"]
    assert len(vocab) == 215
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
    m = FlexSED(device="cuda")
    _split = m.split_audio_fixed

    def split(audio, sr, chunk_duration=10.0):
        out = []
        for c in _split(audio, sr, chunk_duration):
            if len(c) < sr:
                c = np.pad(c, (0, sr - len(c)))
            out.append(c)
        return out
    m.split_audio_fixed = split
    dst = flex_dir(set_name, route); dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for cid in ids:
        f = dst / f"{cid}.npz"
        if f.exists():
            continue
        wav = SEP / set_name / srt / f"{cid}.wav"
        parts = []
        for k in range(0, len(vocab), batch):
            with torch.inference_mode():
                preds = m.run_inference(str(wav), vocab[k:k + batch])
            parts.append(preds.detach().squeeze(1).numpy().astype(np.float16))
            del preds
            torch.cuda.empty_cache()
        fw = np.concatenate(parts, axis=0)
        np.savez_compressed(f, fw=fw, labels=np.array(vocab), fps=25.0); n += 1
        if n <= 2 or n % 50 == 0:
            print(f"[cache-flex] {n} {cid} {fw.shape}", flush=True)
    print(f"[cache-flex] {set_name} {route} (audio route {srt}): {len(ids)} clips, {n} scored now -> {dst}", flush=True)


# ============================================================================= scoring (env msproj, CPU)
def identity(log):
    R, D, _R3, R4, R5 = proj()
    R.use_set("calib")
    cl = D.usable()[:N_IDENT]
    same, dmax_b, dmax_f, rows = 0, [], [], []
    for c in cl:
        cid = c["id"]
        ob, of_, _p = R4.load3(cid)
        ib, if_ = R.load(beats_dir("calib", "ident") / f"{cid}.npz"), R.load(flex_dir("calib", "ident") / f"{cid}.npz")
        e_o = R5.finish(R5.pre(ob, of_, D.AED), B_SHIPPED, D.DISP)
        e_i = R5.finish(R5.pre(ib, if_, D.AED), B_SHIPPED, D.DISP)
        ok = R4.same_spans(e_o, e_i)
        same += ok
        db = float(np.abs(ob[0] - ib[0]).max()) if ob[0].shape == ib[0].shape else float("inf")
        df = float(np.abs(of_[0] - if_[0]).max()) if of_[0].shape == if_[0].shape else float("inf")
        dmax_b.append(db); dmax_f.append(df)
        rows.append({"id": cid, "same_spans": bool(ok), "beats_max_abs_diff": db, "flexsed_max_abs_diff": df,
                     "n_orig": len(e_o), "n_ident": len(e_i)})
    passed = same >= 18
    log["identity"] = {"clips": len(cl), "same_spans": same, "pass": passed, "beats_max_abs_diff_median": float(np.median(dmax_b)),
                       "beats_max_abs_diff_max": float(np.max(dmax_b)), "flexsed_max_abs_diff_median": float(np.median(dmax_f)),
                       "flexsed_max_abs_diff_max": float(np.max(dmax_f)), "rows": rows}
    save_log(log)
    print(f"[identity] same shown spans on {same}/{len(cl)} clips (gate >= 18); frame |d| max BEATs median "
          f"{np.median(dmax_b):.4f} / max {np.max(dmax_b):.4f}; FlexSED median {np.median(dmax_f):.4f} / max {np.max(dmax_f):.4f} "
          f"-> {'PASS' if passed else 'FAIL'}", flush=True)
    if not passed:
        sys.exit("identity gate failed: route bug, stop (no cost read)")


def views(set_name):
    """per clip: (shipped spans, residual-view spans, original BEATs cache, gold clip) -- every clip in D.usable()"""
    R, D, _R3, R4, R5 = proj()
    R.use_set(set_name)
    cl = D.usable()
    miss = [c["id"] for c in cl if not (beats_dir(set_name, "res") / f"{c['id']}.npz").exists()
            or not (flex_dir(set_name, "res") / f"{c['id']}.npz").exists()]
    assert not miss, f"missing residual caches: {miss[:5]} ... ({len(miss)})"
    out = []
    for c in cl:
        ob, of_, _p = R4.load3(c["id"])
        rb = R.load(beats_dir(set_name, "res") / f"{c['id']}.npz")
        rf = R.load(flex_dir(set_name, "res") / f"{c['id']}.npz")
        ship = R5.finish(R5.pre(ob, of_, D.AED), B_SHIPPED, D.DISP)
        res = R5.finish(R5.pre(rb, rf, D.AED), B_SHIPPED, D.DISP)
        out.append((c, ship, res, ob, of_))
    return cl, out


def union(ship, res, ob, gate):
    """residual span with a shipped twin (same family, within 1 s) -> absorbed, shipped span untouched; S2 also needs the
    ORIGINAL BEATs Speech or Music >= 0.3 inside the span (round 3's speechy)"""
    from benchmark.detector_round3 import speechy
    from src.labels import canonical
    key = lambda e: canonical(e.label)
    added = []
    for e in res:
        if any(key(x) == key(e) and x.start - 1.0 <= e.end and e.start - 1.0 <= x.end for x in ship):
            continue
        if gate and not speechy(ob, e):
            continue
        added.append(e)
    return list(ship) + added, added


def unheard_flags(per):
    """per clip, per consequential gold index: (missed by shipped, unheard) -- the prereg's definition"""
    R, D, *_ = proj()
    from src.labels import is_salient_nonspeech, is_music
    flags = []
    for c, ship, _res, ob, of_ in per:
        ev = D._ev(ship)
        fl = {}
        for gi, g in enumerate(c["events"]):
            if not (is_salient_nonspeech(g["label"]) and not is_music(g["label"]) and g["consequential"]):
                continue
            hit = any(R.E._same(e.label, g["label"]) and R.E._overlap_ok(e.start, e.end, g["start"], g["end"]) for e in ev)
            ge = types.SimpleNamespace(start=g["start"], end=g["end"], label=g["label"])
            unheard = (not hit) and D.peak_near(ob, ge) < 0.35 and D.peak_near(of_, ge) < 0.8
            fl[gi] = (not hit, unheard)
        flags.append(fl)
    return flags


def score_cell(cl, per, gate, flags):
    R, D, _R3, R4, _R5 = proj()
    rows, evs, add_n, add_fp, rec_all, rec_unh, true_removed, true_added = [], [], 0, 0, 0, 0, 0, 0
    for (c, ship, res, ob, _f), fl in zip(per, flags):
        ev, added = union(ship, res, ob, gate)
        evs.append(ev); rows.append(D.clip_cost(c, ev))
        sa = D._ev(added)
        add_n += len(sa)
        false = lambda e: not any(R.E._same(e.label, g["label"]) and min(e.end, g["end"]) - max(e.start, g["start"]) > 0 for g in c["events"])
        add_fp += sum(false(e) for e in sa)
        kept = {id(e) for e in ev}
        true_removed += sum(1 for e in D._ev(ship) if not false(e) and id(e) not in kept)   # 0 by construction (union only adds)
        true_added += sum(1 for e in sa if not false(e))
        for gi, (missed, unh) in fl.items():
            g = c["events"][gi]
            if missed and any(R.E._same(e.label, g["label"]) and R.E._overlap_ok(e.start, e.end, g["start"], g["end"]) for e in sa):
                rec_all += 1; rec_unh += unh
    S = R4.summary(cl, rows)
    S.update({"shown": int(sum(len(D._ev(e)) for e in evs)), "residual_only_added": add_n, "residual_only_false": add_fp,
              "recovered_shipped_misses": rec_all, "recovered_unheard": rec_unh,
              "true_spans_removed": true_removed, "true_spans_added": true_added})
    return rows, S


def base_gate(set_name, cl, per):
    R, D, _R3, R4, _R5 = proj()
    rows = [D.clip_cost(c, ship) for c, ship, *_ in per]
    S = R4.summary(cl, rows)
    S["shown"] = int(sum(len(D._ev(ship)) for _c, ship, *_ in per))
    ok = all(abs(S[k] - v) <= TOL[k] for k, v in EXPECT[set_name].items())
    print(f"[baseline {set_name}] {len(cl)} clips: C-ov {S['C_overlap']:.4f} C-on {S['C_onset']:.4f} rec {S['recall_overlap']:.4f} "
          f"on-rec {S['recall_onset']:.4f} fp/min {S['fp_per_min']:.4f} -> expected {EXPECT[set_name]}: {'OK' if ok else 'MISMATCH'}", flush=True)
    return rows, S, ok


def unheard_summary(flags):
    missed = sum(m for fl in flags for m, _u in fl.values())
    unh = sum(u for fl in flags for _m, u in fl.values())
    n = sum(len(fl) for fl in flags)
    return {"consequential": n, "shipped_misses": missed, "unheard": unh, "unheard_share_of_misses": unh / max(1, missed)}


def show(name, S):
    print(f"{name:9s} C-ov {S['C_overlap']:.3f} C-on {S['C_onset']:.3f} rec {S['recall_overlap']:.1%} on-rec {S['recall_onset']:.1%} "
          f"fp {S['fp']} ({S['fp_per_min']:.2f}/min) shown {S['shown']} res-only {S.get('residual_only_added')} "
          f"(false {S.get('residual_only_false')}) recovered {S.get('recovered_shipped_misses')} (unheard {S.get('recovered_unheard')}) "
          f"true spans removed/added {S.get('true_spans_removed')}/{S.get('true_spans_added')} "
          f"dC-ov {S.get('dC_overlap')} dC-on {S.get('dC_onset')}", flush=True)


CELLS = {"S1": False, "S2": True}


def secondary(cl, per, names):
    """Secondary rows (prereg note of 2026-09-28, from the detector audit): the same spans scored with the shipped label
    filter config.LABEL_FILTER = "depictable" (gold and spans); the primary ("lists") and the pick are unchanged."""
    import config
    _R, D, *_ = proj()
    old = config.LABEL_FILTER
    config.LABEL_FILTER = "depictable"
    try:
        base_rows = [D.clip_cost(c, ship) for c, ship, *_ in per]
        R4 = proj()[3]
        Sb = R4.summary(cl, base_rows)
        Sb["shown"] = int(sum(len(D._ev(ship)) for _c, ship, *_ in per))
        flags = unheard_flags(per)
        out = {"label_filter": "depictable", "baseline": Sb, "unheard": unheard_summary(flags)}
        show("shipped/dep", Sb)
        for name in names:
            rows, S = score_cell(cl, per, CELLS[name], flags)
            S["dC_overlap"] = D.boot([x["C_overlap"] - y["C_overlap"] for x, y in zip(rows, base_rows)])
            S["dC_onset"] = D.boot([x["C_onset"] - y["C_onset"] for x, y in zip(rows, base_rows)])
            out[name] = S; show(f"{name}/dep", S)
    finally:
        config.LABEL_FILTER = old
    return out


def fit(log):
    assert log.get("identity", {}).get("pass"), "identity gate first"
    _R, D, *_ = proj()
    cl, per = views("calib")
    base_rows, Sb, ok = base_gate("calib", cl, per)
    if not ok:
        sys.exit("baseline gate failed: stop")
    flags = unheard_flags(per)
    log["fit"] = {"clips": len(cl), "route": log["qc"]["route"], "model": MODEL_ID, "baseline": Sb, "unheard": unheard_summary(flags)}
    print(f"[unheard 280] {log['fit']['unheard']}", flush=True)
    show("shipped", Sb)
    for name, gate in CELLS.items():
        rows, S = score_cell(cl, per, gate, flags)
        S["dC_overlap"] = D.boot([x["C_overlap"] - y["C_overlap"] for x, y in zip(rows, base_rows)])
        S["dC_onset"] = D.boot([x["C_onset"] - y["C_onset"] for x, y in zip(rows, base_rows)])
        log["fit"][name] = S; show(name, S)
    elig = [k for k in CELLS if log["fit"][k]["C_overlap"] < Sb["C_overlap"] and log["fit"][k]["C_onset"] < Sb["C_onset"]]
    pick = min(elig, key=lambda k: (log["fit"][k]["C_overlap"], log["fit"][k]["C_onset"], k)) if elig else None
    log["eligible"] = elig; log["pick"] = pick
    save_log(log)
    print(f"eligible (C-overlap AND C-onset below shipped): {elig}; PICK: {pick}", flush=True)
    log["fit_depictable"] = secondary(cl, per, list(CELLS))
    save_log(log)


def heldout(log):
    pick = log.get("pick")
    if not pick:
        log["heldout"] = "not scored: no eligible cell on the 280 (as pre-registered); the shipped stack stays"
        print(log["heldout"]); save_log(log); return
    _R, D, *_ = proj()
    cl, per = views("heldout")
    base_rows, Sb, ok = base_gate("heldout", cl, per)
    if not ok:
        sys.exit("baseline gate failed on the 415: stop")
    flags = unheard_flags(per)
    rows, S = score_cell(cl, per, CELLS[pick], flags)
    d_ov = [x["C_overlap"] - y["C_overlap"] for x, y in zip(rows, base_rows)]
    S["dC_overlap"] = D.boot(d_ov)
    S["dC_onset"] = D.boot([x["C_onset"] - y["C_onset"] for x, y in zip(rows, base_rows)])
    strata = {}
    for s in ("complex", "random"):
        idx = [i for i, c in enumerate(cl) if c.get("stratum") == s]
        strata[s] = {"n": len(idx), "dC_overlap": D.boot([d_ov[i] for i in idx]) if idx else None,
                     "dfp": float(np.mean([rows[i]["fp"] - base_rows[i]["fp"] for i in idx])) if idx else None}
    S["strata"] = strata
    S["pass"] = bool(S["dC_overlap"][2] < 0)
    log["heldout"] = {"pick": pick, "clips": len(cl), "baseline": Sb, "cell": S, "unheard": unheard_summary(flags)}
    save_log(log)
    show("shipped", Sb); show(pick, S)
    print(f"[heldout] {pick}: dC-overlap {S['dC_overlap']} -> {'PASS' if S['pass'] else 'fail'}; strata {strata}; "
          f"unheard {log['heldout']['unheard']}", flush=True)
    log["heldout_depictable"] = secondary(cl, per, [pick])
    save_log(log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("sep", "qc", "cache-beats", "cache-flex", "identity", "fit", "heldout"))
    ap.add_argument("--set", choices=("calib", "heldout"))
    ap.add_argument("--route", choices=("res", "ident"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.step == "sep":
        sep(a.set, a.shard, a.of, a.limit); return
    log = load_log()
    if a.step == "qc":
        qc(log)
    elif a.step == "cache-beats":
        cache_beats(a.set, a.route, log)
    elif a.step == "cache-flex":
        cache_flex(a.set, a.route, log)
    else:
        {"identity": identity, "fit": fit, "heldout": heldout}[a.step](log)


if __name__ == "__main__":
    main()
