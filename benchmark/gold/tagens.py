"""Round 63 TAG-ENS (docs/prereg_round13_detector_push.md "Round 63 TAG-ENS"): EAT-large + SSLAM on BEATs' 2-s / 0.25-s
windows, per-label quantile matching onto BEATs' scale on 415 half A (label-free), span source = mean(BEATs, calibrated).

    python benchmark/gold/tagens.py manifest --set heldout|dev|dev2   # CPU, msproj; dev from ~/MscProj_r13, dev2 from ~/MscProj_tg
    python benchmark/gold/tagens.py cache --set S --model eat|sslam    # GPU; eat in msproj, sslam in sota (or msproj)
    python benchmark/gold/tagens.py fit                                # CPU: quantiles on half A + half-A inclusion check
    python benchmark/gold/tagens.py step1                              # CPU: half B, GO / STOP (exit 0 / 3)
    TG_ARMS="A B" python benchmark/gold/tagens.py diff A B             # CPU, from ~/MscProj_tg: changed DEV pictures
"""
from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

HOME_PROJ = Path(os.environ.get("MSCPROJ_MAIN_CHECKOUT", "/home/dsi/adamg/MscProj"))   # cluster checkout with the caches
CACHE = HOME_PROJ / "data" / "work" / "tagens"
HELDOUT = _ROOT / "benchmark" / "gold" / "audioset_heldout.json"
BEATS_HELD = HOME_PROJ / "benchmark" / "audioset_heldout_windows" / "beats"
CALIB_JSON = _ROOT / "benchmark" / "gold" / "tagens_calib.json"
CALIB_NPZ = CALIB_JSON.with_suffix(".npz")
OUT1 = _ROOT / "benchmark" / "gold" / "tagens_415.json"
MID_NAMES = _ROOT / "src" / "audioset_mid_names.json"
EAT_LABELS = _ROOT / "benchmark" / "round5_eat_labels.csv"
SSLAM_LABELS = Path.home() / "SSLAM" / "SSLAM_Inference" / "inference" / "labels.csv"
TAGGERS = ("eat", "sslam")
SEED, N_A, LEVELS = 63, 207, np.linspace(0.0, 1.0, 2001)
AED, DISP, MIN_DUR, EARLY, LATE = 0.175, 0.35, 0.3, 0.5, 1.0
SR, WINDOW, HOP, STAMP_OFFSET = 16000, 2.0, 0.25, 0.5


# ----------------------------------------------------------------------------- windows (copy of infer_beats / detector_round5)
def windows(audio):
    n_win = int(round(WINDOW * SR)); n_hop = int(round(HOP * SR))
    lead = n_win - n_hop
    audio = np.pad(audio, (lead, 0), mode="reflect" if len(audio) > lead else "constant")
    if len(audio) < n_win:
        audio = np.pad(audio, (0, n_win - len(audio)))
    starts = list(range(0, len(audio) - n_win + 1, n_hop))
    if not starts or starts[-1] + n_win < len(audio):
        starts.append(max(0, len(audio) - n_win))
    times = np.array([(s + n_win - lead) / SR - STAMP_OFFSET for s in starts])
    keep = times >= 0.0
    chunks = np.stack([audio[s:s + n_win] for s in starts]).astype(np.float32)
    return chunks[keep], times[keep]


def _names(model):
    mid_names = json.loads(MID_NAMES.read_text("utf-8"))
    src = EAT_LABELS if model == "eat" else SSLAM_LABELS
    rows = list(csv.reader(src.open(encoding="utf-8")))
    rows = [r for r in rows if r and r[0].strip().isdigit()]            # skip a header row if any
    assert len(rows) == 527, (model, len(rows))
    return [mid_names.get(r[1], r[1]) for r in rows]


_M: dict = {}


def _scorer(model, device="cuda"):
    if model in _M:
        return _M[model]
    import torch
    if model == "eat":
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        snap = Path(snapshot_download("worstchan/EAT-large_epoch20_finetune_AS2M"))
        pkg = types.ModuleType("eat_hf"); pkg.__path__ = [str(snap)]; sys.modules["eat_hf"] = pkg
        from eat_hf.eat_model import EAT
        cfg = types.SimpleNamespace(**json.loads((snap / "config.json").read_text("utf-8")))
        cfg.img_size = tuple(cfg.img_size)
        m = EAT(cfg)
        sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in load_file(str(snap / "model.safetensors")).items()}
        m.load_state_dict(sd, strict=True)
        m = m.to(device).eval()

        def score(chunks):                                  # detector_round5.eat_model's score, unchanged
            import torchaudio
            mels = []
            for w in chunks:
                x = torch.from_numpy(w).float(); x = x - x.mean()
                mel = torchaudio.compliance.kaldi.fbank(x.unsqueeze(0), htk_compat=True, sample_frequency=16000, use_energy=False,
                                                        window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10)
                mel = torch.nn.functional.pad(mel, (0, 0, 0, 208 - mel.shape[0]))
                mels.append((mel - (-4.268)) / (4.569 * 2))
            with torch.no_grad():
                return torch.sigmoid(m(torch.stack(mels).unsqueeze(1).to(device))).float().cpu().numpy()
    else:
        from src.stage4_audio_event_detection import sslam_infer as SI
        m = SI.load_model(device)

        def score(chunks):                                  # sslam_infer.mel pads each 2-s window to 1024 frames
            with torch.no_grad():
                x = torch.cat([SI.mel(w, device) for w in chunks], dim=0)
                return torch.sigmoid(m(x)).float().cpu().numpy()
    _M[model] = (score, _names(model))
    return _M[model]


def score_to(model, wav: Path, times, out: Path, beats_labels=None, batch=32, audio=None):
    """one tagger pass over BEATs' windows of `wav`; columns in BEATs' label order; saved fp16 (the calibration's input format)"""
    score, names = _scorer(model)
    if audio is None:
        import librosa
        audio, _ = librosa.load(str(wav), sr=SR, mono=True)
    chunks, t = windows(audio)
    assert len(t) == len(times) and np.allclose(t, times, atol=1e-3), f"window mismatch {wav}"
    fw = np.concatenate([score(chunks[i:i + batch]) for i in range(0, len(chunks), batch)], axis=0)
    bl = list(beats_labels) if beats_labels is not None else json.loads(CALIB_JSON.read_text("utf-8"))["labels"]
    idx = {n: i for i, n in reversed(list(enumerate(names)))}
    missing = [n for n in bl if n not in idx]
    assert not missing, f"{model}: BEATs names not found {missing[:5]}"
    fw = fw[:, [idx[n] for n in bl]]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, fw=fw.astype(np.float16), times=np.asarray(times, np.float32), labels=np.array(bl))


def _load_fr(p):
    z = np.load(p)
    return z["fw"].astype(np.float32), z["times"].astype(np.float64), [str(x) for x in z["labels"]]


# ----------------------------------------------------------------------------- manifest / cache
def cmd_manifest(set_name):
    rows = []
    if set_name == "heldout":
        H = json.loads(HELDOUT.read_text(encoding="utf-8"))
        for c in H["clips"]:
            rows.append({"clip": c["id"], "media": str((HOME_PROJ / "data" / "input" / "audioset_heldout" / f"{c['id']}.mp4")),
                         "beats": str(BEATS_HELD / f"{c['id']}.npz")})
    else:
        from benchmark.gold import dev_candidates_check as DCC
        if set_name == "dev":
            _g, stems = DCC.dev_stems()
        else:
            from benchmark.gold import tagger_prep as TP
            DCC, _R, stems = TP.configure(set_name)
        for st in stems:
            rows.append({"clip": st, "media": str(DCC.wav_of(st).resolve()), "beats": str((DCC.BEATS_DIR / f"{st}.npz").resolve())})
    miss = [r for r in rows if not (Path(r["media"]).exists() and Path(r["beats"]).exists())]
    assert not miss, miss[:3]
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"manifest_{set_name}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"[manifest] {set_name}: {len(rows)} clips", flush=True)


def cmd_decode(set_name):
    """16-kHz mono float32 audio per clip, exactly as score_to loads it in msproj, for a tagger run in another env"""
    import librosa
    rows = json.loads((CACHE / f"manifest_{set_name}.json").read_text(encoding="utf-8"))
    (CACHE / "audio").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for r in rows:
            dst = CACHE / "audio" / f"{r['clip']}.npy"
            if dst.exists():
                continue
            wav = Path(r["media"])
            if wav.suffix != ".wav":
                wav = Path(td) / "a.wav"
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", r["media"], "-vn", "-ac", "1", "-ar", str(SR), str(wav)], check=True)
            audio, _ = librosa.load(str(wav), sr=SR, mono=True)
            _c, t = windows(audio)
            assert np.allclose(t, _load_fr(r["beats"])[1], atol=1e-3), r["clip"]
            np.save(dst, audio.astype(np.float32))
    print(f"[decode] {set_name}: {len(rows)} clips", flush=True)


def cmd_cache(set_name, model):
    rows = json.loads((CACHE / f"manifest_{set_name}.json").read_text(encoding="utf-8"))
    secs, n = [], 0
    with tempfile.TemporaryDirectory() as td:
        for r in rows:
            dst = CACHE / model / f"{r['clip']}.npz"
            if dst.exists():
                continue
            _bf, bt, bl = _load_fr(r["beats"])
            wav = Path(r["media"])
            npy = CACHE / "audio" / f"{r['clip']}.npy"         # decoded once in msproj (the sota env decodes differently)
            if npy.exists():
                t0 = time.time()
                score_to(model, wav, bt, dst, beats_labels=bl, audio=np.load(npy))
                secs += [time.time() - t0] if n else []
                n += 1
                continue
            if wav.suffix != ".wav":
                wav = Path(td) / "a.wav"
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", r["media"], "-vn", "-ac", "1", "-ar", str(SR), str(wav)], check=True)
            t0 = time.time()
            score_to(model, wav, bt, dst, beats_labels=bl)
            if n:                                             # the first clip includes the model load
                secs.append(time.time() - t0)
            n += 1
    log = CACHE / "timing.json"
    T = json.loads(log.read_text(encoding="utf-8")) if log.exists() else {}
    if secs:
        T[f"{set_name}|{model}"] = {"clips": n, "mean_s_per_clip": round(float(np.mean(secs)), 3)}
        log.write_text(json.dumps(T, indent=1), encoding="utf-8")
    print(f"[cache] {set_name} {model}: {n} scored now; {T.get(f'{set_name}|{model}')}", flush=True)


# ----------------------------------------------------------------------------- 415: split, fit, scoring
def split():
    ids = [c["id"] for c in json.loads(HELDOUT.read_text(encoding="utf-8"))["clips"]]
    assert len(ids) == 415
    random.Random(SEED).shuffle(ids)
    return ids[:N_A], ids[N_A:]


def frames(cid):
    B = _load_fr(BEATS_HELD / f"{cid}.npz")
    T = {m: _load_fr(CACHE / m / f"{cid}.npz")[0] for m in TAGGERS}
    for m, x in T.items():
        assert x.shape == B[0].shape, (cid, m)
    return B, T


def calibrate(T, q, m):
    qT, qB = q[f"{m}_qT"], q[f"{m}_qB"]
    return np.stack([np.interp(T[:, c], qT[:, c], qB[:, c]) for c in range(T.shape[1])], axis=1)


def spans(fw, t, labs):
    from src.labels import canonical
    from src.stage4_audio_event_detection import _extract_events
    from benchmark.gold import expect_screen as E
    dep = set(E.FAMILIES)
    out = []
    for e in _extract_events(fw, t, labs, DISP, None, MIN_DUR, low=AED):
        f = canonical(e.label)
        if f in dep:
            out.append((f, float(e.start)))
    return out


def evaluate(ids, source):
    """source(cid) -> (fw, t, labs); Round 42 precision + onset recall of depictable strong events"""
    from src.labels import canonical
    from benchmark.gold.score_per_sound import same_family
    from benchmark.gold import expect_screen as E
    dep = set(E.FAMILIES)
    ev_of = {c["id"]: c["events"] for c in json.loads(HELDOUT.read_text(encoding="utf-8"))["clips"]}
    n = cor = ne = rec = 0
    for cid in ids:
        sp = spans(*source(cid))
        evs = ev_of[cid]
        for f, a in sp:
            n += 1
            cor += any(same_family(f, ev["label"]) and ev["start"] - EARLY <= a <= ev["start"] + LATE for ev in evs)
        for ev in evs:
            if canonical(ev["label"]) in dep:
                ne += 1
                rec += any(same_family(f, ev["label"]) and ev["start"] - EARLY <= a <= ev["start"] + LATE for f, a in sp)
    return {"spans": n, "correct": cor, "precision": round(cor / n, 4) if n else None,
            "events": ne, "recalled": rec, "recall": round(rec / ne, 4) if ne else None}


def cmd_fit():
    A, B = split()
    bl = None
    BA, TA = [], {m: [] for m in TAGGERS}
    for cid in A:
        (fw, _t, labs), T = frames(cid)
        bl = bl or labs
        assert labs == bl
        BA.append(fw)
        for m in TAGGERS:
            TA[m].append(T[m])
    BA = np.concatenate(BA)
    q = {}
    for m in TAGGERS:
        X = np.concatenate(TA[m])
        q[f"{m}_qT"] = np.quantile(X, LEVELS, axis=0).astype(np.float32)
        q[f"{m}_qB"] = np.quantile(BA, LEVELS, axis=0).astype(np.float32)
    np.savez_compressed(CALIB_NPZ, **q)
    # half-A inclusion check (pre-registered): each calibrated tagger alone vs BEATs alone on half A
    base = evaluate(A, lambda c: _load_fr(BEATS_HELD / f"{c}.npz"))
    alone, kept = {}, []
    for m in TAGGERS:
        def src(c, m=m):
            (fw, t, labs), T = frames(c)
            return calibrate(T[m], q, m), t, labs
        alone[m] = evaluate(A, src)
        ok = (alone[m]["precision"] >= base["precision"] - 0.05 and abs(alone[m]["spans"] - base["spans"]) <= 0.25 * base["spans"])
        alone[m]["included"] = bool(ok)
        if ok:
            kept.append(m)
    meta = {"round": "Round 63 TAG-ENS", "seed": SEED, "half_A": A, "half_B": B, "levels": len(LEVELS), "taggers": kept,
            "cache": str(CACHE), "labels": bl, "half_A_beats": base, "half_A_alone": alone}
    CALIB_JSON.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[fit] half A BEATs {base}", flush=True)
    for m in TAGGERS:
        print(f"[fit] half A {m} alone (calibrated) {alone[m]}", flush=True)
    print(f"[fit] taggers kept: {kept}", flush=True)
    if not kept:
        print("[fit] STOP: both taggers fail the half-A inclusion check", flush=True)
        sys.exit(3)


def cmd_step1():
    meta = json.loads(CALIB_JSON.read_text(encoding="utf-8"))
    q = dict(np.load(CALIB_NPZ))
    kept = meta["taggers"]
    _A, B = split()
    assert B == meta["half_B"]

    def ens(c):
        (fw, t, labs), T = frames(c)
        return np.mean([fw] + [calibrate(T[m], q, m) for m in kept], axis=0), t, labs
    base = evaluate(B, lambda c: _load_fr(BEATS_HELD / f"{c}.npz"))
    te = evaluate(B, ens)
    c1 = abs(te["spans"] - base["spans"]) <= 0.10 * base["spans"]
    c2 = te["precision"] >= 0.42
    c3 = te["recall"] >= base["recall"]
    go = bool(c1 and c2 and c3)
    res = {"round": "Round 63 TAG-ENS step 1 (415 half B)", "taggers": kept, "beats": base, "tag_ens": te,
           "count_within_10pct": bool(c1), "precision_ge_0.42": bool(c2), "recall_ge_beats": bool(c3), "GO": go}
    OUT1.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1), flush=True)
    sys.exit(0 if go else 3)


def cmd_diff(A, B):
    from benchmark.gold import weakwitness_dev as W
    res = []
    for part, (gold, P, _c, _s) in W.parts([A, B]).items():
        for st in P[A]:
            ca, cb = W.classify(gold[st], P[A][st]), W.classify(gold[st], P[B][st])
            ka = {(l, round(a, 2), round(b, 2), k) for l, a, b, k in ca}
            kb = {(l, round(a, 2), round(b, 2), k) for l, a, b, k in cb}
            if ka != kb:
                res.append({"part": part, "clip": st, "only_" + A: sorted(ka - kb, key=lambda x: x[1]),
                            "only_" + B: sorted(kb - ka, key=lambda x: x[1])})
                print(part, st, "\n   -", sorted(ka - kb, key=lambda x: x[1]), "\n   +", sorted(kb - ka, key=lambda x: x[1]), flush=True)
    p = _ROOT / "benchmark" / "gold" / "tagens_diff.json"
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(p)


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {a[i].lstrip("-"): a[i + 1] for i in range(1, len(a) - 1) if a[i].startswith("--")}
    if a[0] == "manifest":
        cmd_manifest(kw["set"])
    elif a[0] == "decode":
        cmd_decode(kw["set"])
    elif a[0] == "cache":
        cmd_cache(kw["set"], kw["model"])
    elif a[0] == "fit":
        cmd_fit()
    elif a[0] == "step1":
        cmd_step1()
    elif a[0] == "diff":
        cmd_diff(a[1], a[2])
