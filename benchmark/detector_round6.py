"""Detector round 6 (docs/prereg_round6_dasm.md, 2026-09-28): DASM (Detect Any Sound Model, ACM MM 2025) in place of
FlexSED (D1) or next to it (D2) in the shipped stage-4 stack (BEATs + FlexSED 0.8 + FlexSED clip veto 0.3 + BEATs
self-veto b 0.1218, PANNs off). Cost C, clip sets and bootstrap as amendment 24 (detector_round2.py).

    python benchmark/detector_round6.py embed                    # GPU: MGA-CLAP text embeddings of the 215 queries
    python benchmark/detector_round6.py cache --set calib|heldout  # GPU: DASM frame scores -> <WIN>/dasm_cache
    python benchmark/detector_round6.py check                    # c4: every clip cached, 215 labels == FlexSED's
    python benchmark/detector_round6.py fit                      # gate, v and g on the 280, D1 + D2, the pick
    python benchmark/detector_round6.py heldout                  # the pick on the 415 (only if there is one)
    python benchmark/detector_round6.py secondary                # reported only: LABEL_FILTER lists vs depictable, span churn

DASM code: github.com/cai525/Transformer4SED (~/Transformer4SED); MGA-CLAP: github.com/Ming-er/MGA-CLAP
(~/Transformer4SED/third_parties/MGA-CLAP); weights: HF CPF2/detect_any_sound (text_query), MGA-CLAP model.pt (official
Google Drive link), google-bert/bert-base-uncased (~/bert-base-uncased). The embed / cache steps import those repos'
own `src` / `models` / `tools` packages, so they take the project off sys.path first (as benchmark/gold/flexsed_run.py).
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark import audioset_stage4_report as R
from benchmark import detector_round2 as D
from benchmark import detector_round4 as R4
from benchmark import detector_round5 as R5
from src.labels import canonical
from src.stage4_audio_event_detection import _extract_events

OUT = _ROOT / "benchmark" / "detector_round6.json"
VOCAB = _ROOT / "benchmark" / "gold" / "depictable_vocab.json"
WAVCAPS = _ROOT / "benchmark" / "round6_wavcaps_ids.json"
QFILE = _ROOT / "data" / "work" / "dasm_text_queries.pt"
CACHE = "dasm_cache"
T4 = Path(os.environ.get("T4SED_ROOT", Path.home() / "Transformer4SED"))
MGA = T4 / "third_parties" / "MGA-CLAP"
BERT = Path(os.environ.get("BERT_DIR", Path.home() / "bert-base-uncased"))
DASM_W = T4 / "pretrained_model" / "detect_any_sound" / "text_query" / "as_full_text_query_best_model.pt"
DASM_CFG = T4 / "pretrained_model" / "detect_any_sound" / "text_query" / "config.yaml"
SR, CLIP_S, FPS, TEMP_W = 32000, 10, 50.0, 0.5
NB_QUERIES = ["alarm", "sirens", "air raid sirens", "dog", "music", "bird"]     # the official notebook's example list
B_SHIPPED, FVETO, FBAR = R5.B_SHIPPED, R4.FVETO, R4.FBAR
G_GRID = [round(0.05 + 0.005 * i, 3) for i in range(181)]
key = lambda e: canonical(e.label)


def vocab():
    return json.loads(VOCAB.read_text(encoding="utf-8"))["families"]


def dasm_text(x):
    return "sound of " + x.lower()                     # the official notebook's template


# ----------------------------------------------------------------------------- foreign-repo imports
def _foreign(root: Path):
    """take this project off the import path and make `root` the repo whose `src`/`models`/`tools` packages are used"""
    for k in list(sys.modules):
        if k in ("src", "models", "tools") or k.startswith(("src.", "models.", "tools.")):
            del sys.modules[k]
    bad = {_ROOT.resolve(), (_ROOT / "benchmark").resolve()}
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() not in bad]
    sys.path.insert(0, str(root))
    os.chdir(root)


def _stub(name, **attrs):
    """an import-only dependency that the inference path never calls (wandb, loguru, ... in MGA-CLAP's tools/utils)"""
    try:
        importlib.import_module(name)
        return False
    except ImportError:
        pass
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        n = ".".join(parts[:i])
        if n not in sys.modules:
            sys.modules[n] = types.ModuleType(n)
        if i > 1:
            setattr(sys.modules[".".join(parts[:i - 1])], parts[i - 1], sys.modules[n])
    for k, v in attrs.items():
        setattr(sys.modules[name], k, v)
    return True


# ----------------------------------------------------------------------------- step 1: text embeddings (GPU)
def embed(device="cuda"):
    import torch
    import torch.nn.functional as F
    import yaml
    voc = vocab()
    assert len(voc) == 215
    queries = [dasm_text(x) for x in voc]
    nb = [dasm_text(x) for x in NB_QUERIES]
    _foreign(MGA)
    # the real transformers / accelerate chain is imported before the stubs exist: accelerate probes
    # importlib.util.find_spec("wandb") at import time and a stub has no __spec__
    import accelerate  # noqa: F401
    from transformers import BertModel, BertTokenizer  # noqa: F401
    stubs = [n for n, a in (("wandb", {}), ("loguru", {"logger": logging.getLogger("loguru")}),
                            ("sentence_transformers", {"util": types.SimpleNamespace()}), ("sed_scores_eval", {}),
                            ("sed_scores_eval.utils.scores", {"create_score_dataframe": None}), ("ruamel.yaml", {}))
             if _stub(n, **a)]
    print(f"[embed] import-only stubs: {stubs}", flush=True)
    from models import text_encoder as TE
    TE.MODELS[str(BERT)] = TE.MODELS["bert-base-uncased"]    # the same BertModel / BertTokenizer, read from a local copy
    from models.ase_model import ASE
    cfg = yaml.safe_load(open("settings/inference_sed.yaml", encoding="utf-8"))
    cfg["text_encoder_args"]["type"] = str(BERT)
    clap = ASE(cfg)
    sd = torch.load(str(MGA / "pretrained_models" / "models" / "model.pt"), map_location="cpu")["model"]
    res = clap.load_state_dict(sd, strict=False)                 # strict=False as the official notebook
    bad = [k for k in res.missing_keys if k.startswith(("text_encoder.", "word_proj.", "codebook"))]
    print(f"[embed] MGA-CLAP load: {len(res.missing_keys)} missing (text side: {bad}), {len(res.unexpected_keys)} unexpected "
          f"{res.unexpected_keys[:8]}", flush=True)
    assert not bad, "c1 failed: text side of MGA-CLAP not fully loaded"
    clap = clap.to(device).eval()

    def enc(q):
        with torch.no_grad():
            _, we, am = clap.encode_text([q])                    # one query at a time (prereg): no batch padding
            return F.normalize(clap.msc(we, clap.codebook, am), dim=-1)[0].float().cpu()
    E = torch.stack([enc(q) for q in queries])
    NB = torch.stack([enc(q) for q in nb])
    assert E.shape == (215, 1024), E.shape
    QFILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vocab": voc, "queries": queries, "embeds": E, "nb_queries": nb, "nb_embeds": NB,
                "mga_missing": list(res.missing_keys), "mga_unexpected": list(res.unexpected_keys), "stubs": stubs}, str(QFILE))
    print(f"[embed] {E.shape} -> {QFILE}; first queries {queries[:3]}", flush=True)


# ----------------------------------------------------------------------------- step 2: DASM frame scores (GPU)
class _Dasm:
    def __init__(self, device):
        import torch
        import yaml
        self.torch, self.device = torch, device
        _foreign(T4)
        import torch._dynamo  # noqa: F401  (probes find_spec("pandas") at import time; a stub has no __spec__)
        import timm, torchaudio, torchlibrosa  # noqa: F401,E401
        _stub("pandas")                                        # src/codec/encoder.py imports it; not used for inference
        from src.models.detect_any_sound.detect_any_sound_htast import DASM_HTSAT
        from src.codec.encoder import Encoder
        from src.preprocess.feats_extraction import pad_wav, to_mono
        self.pad_wav, self.to_mono = pad_wav, to_mono
        cfg = yaml.safe_load(open(DASM_CFG, encoding="utf-8"))
        m = DASM_HTSAT(**cfg["DASM_HTSAT"]["init_kwargs"])      # the HTS-AT backbone is loaded strict=True inside
        res = m.load_state_dict(torch.load(str(DASM_W), map_location="cpu"), strict=True)
        self.load_info = {"missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys)}
        self.m = m.to(device).eval()
        f = cfg["feature"]
        self.enc = Encoder([], audio_len=f["audio_max_len"], frame_len=f["win_length"], frame_hop=f["hopsize"],
                           net_pooling=f["net_subsample"], sr=f["sr"])
        assert self.enc.sr == SR and self.enc.audio_len == CLIP_S and self.enc.n_frames == 500
        assert abs(self.enc.n_frames / CLIP_S - FPS) < 1e-9
        self.base = self.m.at_query                              # nn.Parameter [407, 1024] (not a ParameterList)
        assert isinstance(self.base, torch.nn.Parameter) and self.base.shape[0] == 407, type(self.base)
        self.extract = self.m.get_feature_extractor()

    def score(self, wav, qemb):
        """wav: mono float32 at 32 kHz; qemb [Q, 1024]. Returns (fw [Q, T] with T = frames before the pad, at_out [Q])"""
        torch = self.torch
        base = self.base.detach()
        query = torch.cat([base, qemb.to(base.device, base.dtype)]).to(self.device)
        nb = base.shape[0]
        att = torch.ones(query.shape[0], query.shape[0], dtype=torch.bool, device=self.device)
        att[:, :nb] = False
        att.fill_diagonal_(False)                               # the notebook's get_att_mask
        n = int(round(CLIP_S * SR))
        # clarification (prereg): up to 11 s the clip is cut to 10 s as the official waveform_modification does (a
        # 10.01-s clip would otherwise add a 10-ms piece whose one zero-padded frame scores ~0.17 for most families);
        # only a longer clip (none in the two sets) is scored in 10-s pieces
        pieces = [wav[i:i + n] for i in range(0, len(wav), n)] if len(wav) > n + SR else [wav[:n]]
        fws, ats = [], []
        for p in pieces:
            w, pad_mask = self.pad_wav(self.to_mono(p), n, self.enc)
            x = torch.from_numpy(w).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                mel = self.extract(x)
                strong, _weak, other = self.m(input=mel, temp_w=TEMP_W, pad_mask=pad_mask.unsqueeze(0).to(self.device),
                                              query=query, query_type="text", tgt_mask=att)
            keep = int((~pad_mask).sum())
            fws.append(strong[0, nb:, :keep].float().cpu().numpy())
            ats.append(other["at_out"][0, nb:].float().cpu().numpy())
        return np.concatenate(fws, axis=1), np.max(np.stack(ats), axis=0)


def cache(set_name, device="cuda"):
    import librosa
    import torch
    R.use_set(set_name)
    out = (R.E.WIN / CACHE).resolve(); out.mkdir(parents=True, exist_ok=True)
    clips = [(c["id"], (R.E.VIDEOS / f"{c['id']}.mp4").resolve()) for c in R.E.clips()]
    q = torch.load(str(QFILE))
    voc = vocab()
    assert q["vocab"] == voc and q["queries"] == [dasm_text(x) for x in voc], "query file differs from the vocab"
    E = q["embeds"]
    d = _Dasm(device)
    print(f"[cache] {set_name}: DASM load strict=True {d.load_info}; {len(clips)} clips -> {out}", flush=True)
    log = {"set": set_name, "load": d.load_info}
    with tempfile.TemporaryDirectory() as td:
        def wav_of(src):
            wp = Path(td) / "a.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(SR), str(wp)],
                           check=True)
            a, _ = librosa.load(str(wp), sr=SR, mono=True)
            return a
        if set_name == "calib":                                  # c2 batch independence, c3 notebook examples
            a = wav_of(clips[0][1])
            fw, at = d.score(a, E)
            c2 = {}
            for j in (0, 100, 214):
                f1, a1 = d.score(a, E[j:j + 1])
                c2[voc[j]] = {"max_abs_frame": float(np.abs(f1[0] - fw[j]).max()), "abs_at": float(abs(a1[0] - at[j]))}
            ok2 = all(v["max_abs_frame"] < 1e-4 and v["abs_at"] < 1e-4 for v in c2.values())
            print(f"[c2] batch independence on {clips[0][0]}: {c2} -> {'OK' if ok2 else 'FAIL'}", flush=True)
            log["c2"] = {"clip": clips[0][0], "diff": c2, "pass": ok2}
            if not ok2:
                (out / "_round6_cache_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
                sys.exit("c2 failed: stop")
            c3 = {}
            for ex in sorted((T4 / "docs" / "DASM" / "examples").glob("*.wav")):
                a3, _ = librosa.load(str(ex), sr=SR, mono=True)
                f3, at3 = d.score(a3, q["nb_embeds"])
                c3[ex.name] = {NB_QUERIES[i]: {"at_out": round(float(at3[i]), 3), "frame_max": round(float(f3[i].max()), 3)}
                               for i in range(len(NB_QUERIES))}
                print(f"[c3] {ex.name}: {c3[ex.name]}", flush=True)
            log["c3"] = c3
        n = 0
        for cid, src in clips:
            dst = out / f"{cid}.npz"
            if dst.exists():
                continue
            if not src.exists():
                print("missing video", cid, flush=True); continue
            fw, at = d.score(wav_of(src), E)
            assert fw.shape[0] == 215
            np.savez_compressed(dst, fw=fw.astype(np.float16), labels=np.array(voc), fps=FPS, at_out=at.astype(np.float32))
            n += 1
    log["scored_now"] = n
    (out / "_round6_cache_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"[cache] {set_name}: {n} clips scored now -> {out}", flush=True)


# ----------------------------------------------------------------------------- c4
def check(log):
    voc = vocab(); ok = True
    for set_name in ("calib", "heldout"):
        R.use_set(set_name)
        cl = D.usable()
        miss, bad_t, bad_l = [], [], []
        for c in cl:
            p = R.E.WIN / CACHE / f"{c['id']}.npz"
            if not p.exists():
                miss.append(c["id"]); continue
            fw, t, labs = R.load(p)
            ff, _ft, fl = R.load(R.FLEX / f"{c['id']}.npz")
            # clarification (prereg c4): a clip whose audio is shorter than 10 s has fewer frames, so the frame count
            # is checked against FlexSED's for the same clip (50 vs 25 fps: 2x, +-2 for rounding; FlexSED capped at 10 s)
            bad_t += [c["id"]] if (fw.shape[1] != 215 or fw.shape[0] > 500 or abs(fw.shape[0] - 2 * min(ff.shape[0], 250)) > 2
                                   or abs(t[1] - t[0] - 1 / FPS) > 1e-9) else []
            bad_l += [c["id"]] if not (labs == voc == fl) else []
        g = {"clips": len(cl), "missing": len(miss), "bad_frames": bad_t[:5], "n_bad_frames": len(bad_t),
             "labels_differ_from_flexsed": bad_l[:5], "n_labels_differ": len(bad_l)}
        print(f"[c4] {set_name}: {g}")
        log[f"c4_{set_name}"] = g
        ok &= not miss and not bad_t and not bad_l
    log["c4_pass"] = bool(ok)
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")
    if not ok:
        sys.exit("c4 failed: stop")


# ----------------------------------------------------------------------------- the stack with DASM
def load6(cid):
    b, f, p = R4.load3(cid)
    return b, f, p, R.load(R.E.WIN / CACHE / f"{cid}.npz")


def _ext(fr, bar):
    return _extract_events(fr[0], fr[1], fr[2], bar, None, config.AED_MIN_DUR, low=bar * D.HYS)


def _twin(events, new):
    """the shipped twin rule: a new span with a same-family span in `events` within 1 s extends that span's start
    (earlier start kept); otherwise it is returned as fresh (only-this-model)"""
    fresh = []
    for e in new:
        tw = [x for x in events if key(x) == key(e) and x.start - 1.0 <= e.end and e.start - 1.0 <= x.end]
        if tw:
            for x in tw:
                x.start = min(x.start, e.start)
        else:
            fresh.append(e)
    return fresh


def stack6(cc, cell, g=None, v=None, b=B_SHIPPED):
    """cell 'ship' (the round-5 baseline), 'D1' (DASM in place of FlexSED) or 'D2' (BEATs + FlexSED + DASM).
    Returns (shown spans, ids of FlexSED-only spans, ids of DASM-only spans)."""
    bb, f, _p, dm = cc
    events = _ext(bb, D.AED)
    bpk = D.clip_peak(bb)
    flex_ids, dasm_ids = set(), set()
    if cell in ("ship", "D2"):
        fresh = _twin(events, _ext(f, FBAR))
        flex_ids = {id(e) for e in fresh}
        events = events + fresh
    if cell in ("D1", "D2"):
        fresh = _twin(events, _ext(dm, g))                   # D2: against BEATs and FlexSED-only spans (pre-veto pool)
        dasm_ids = {id(e) for e in fresh}
        events = events + fresh
    if cell in ("ship", "D2"):
        fpk = D.clip_peak(f)
        events = [e for e in events if fpk.get(key(e), 1.0) >= FVETO]                 # FlexSED clip veto, every span
    if cell in ("D1", "D2"):
        dpk = D.clip_peak(dm)
        scope = (lambda e: True) if cell == "D1" else (lambda e: id(e) in dasm_ids)   # D2: DASM-only spans only
        events = [e for e in events if not scope(e) or dpk.get(key(e), 1.0) >= v]     # DASM clip veto
    only = flex_ids | dasm_ids
    events = [e for e in events if id(e) not in only or bpk.get(key(e), 1.0) >= b]  # BEATs self-veto
    out = [e for e in events if id(e) in dasm_ids or e.confidence >= D.DISP]         # DASM-only: floor = g (prereg)
    return out, flex_ids, dasm_ids


def run6(cl, ccs, cell, g=None, v=None):
    evs, n_flex, n_dasm = [], 0, 0
    for cc in ccs:
        ev, fi, di = stack6(cc, cell, g, v)
        evs.append(ev)
        shown = D._ev(ev)
        n_flex += sum(id(e) in fi for e in shown); n_dasm += sum(id(e) in di for e in shown)
    rows = [D.clip_cost(c, ev) for c, ev in zip(cl, evs)]
    S = R4.summary(cl, rows)
    S.update({"cell": cell, "g": g, "v": v, "b": B_SHIPPED, "shown": int(sum(len(D._ev(ev)) for ev in evs)),
              "flex_only_shown": n_flex, "dasm_only_shown": n_dasm})
    return evs, rows, S


def show(name, S):
    ee = S.get("end_err", {})
    print(f"{name:9s} C-ov {S['C_overlap']:.3f} C-on {S['C_onset']:.3f} rec {S['recall_overlap']:.1%} on-rec {S['recall_onset']:.1%} "
          f"fp {S['fp']} ({S['fp_per_min']:.2f}/min) shown {S['shown']} flex-only {S['flex_only_shown']} dasm-only "
          f"{S['dasm_only_shown']} g {S['g']} v {S['v']} dC-ov {S.get('dC_overlap')} dC-on {S.get('dC_onset')} "
          f"end-err {ee.get('median_baseline')} -> {ee.get('median_cell')} d {ee.get('delta_baseline_minus_cell')}", flush=True)


def gate(set_name, log):
    """the round-6 baseline == round 5's free-bar stack at the shipped b, span for span, and its numbers"""
    R.use_set(set_name)
    cl = D.usable()
    miss = [c["id"] for c in cl if not (R.E.WIN / CACHE / f"{c['id']}.npz").exists()]
    assert not miss, f"missing DASM caches: {miss[:5]} ({len(miss)})"
    ccs = [load6(c["id"]) for c in cl]
    evs, rows, S = run6(cl, ccs, "ship")
    for c, cc, e in zip(cl, ccs, evs):
        assert R4.same_spans(e, R5.finish(R5.pre(cc[0], cc[1], D.AED), B_SHIPPED, D.DISP)), c["id"]
    exp = R5.EXPECT_BASE[set_name] if set_name == "heldout" else {"C_overlap": 3.036, "C_onset": 3.850,
                                                                  "recall_overlap": 0.513, "fp_per_min": 4.44}
    tol = {"C_overlap": 0.0005, "C_onset": 0.0005, "recall_overlap": 0.0005, "fp_per_min": 0.005}
    ok = all(abs(S[k] - x) <= tol[k] for k, x in exp.items())
    print(f"[gate {set_name}] {len(cl)} clips; baseline == round-5 stack span for span; C-ov {S['C_overlap']:.4f} C-on "
          f"{S['C_onset']:.4f} rec {S['recall_overlap']:.4f} fp/min {S['fp_per_min']:.4f} shown {S['shown']} "
          f"flex-only {S['flex_only_shown']} -> expected {exp}: {'OK' if ok else 'MISMATCH'}", flush=True)
    log[f"gate_{set_name}"] = {"clips": len(cl), "baseline": S, "ok": ok}
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")
    if not ok:
        sys.exit("gate failed: stop")
    return cl, ccs, evs, rows, S


def fit_v(ccs):
    """the DASM veto bar keeping the same share of (clip x family) cells as FlexSED's 0.3 (prereg step 1)"""
    fcm = np.concatenate([cc[1][0].max(axis=0) for cc in ccs])
    dcm = np.concatenate([cc[3][0].max(axis=0) for cc in ccs])
    assert len(fcm) == len(dcm) == 215 * len(ccs)
    s_f = float((fcm >= FVETO).mean())
    k = int(np.floor(s_f * len(dcm) + 0.5))
    v = float(np.sort(dcm)[::-1][k - 1])
    return v, {"cells": int(len(dcm)), "flex_share_at_0.3": s_f, "k": k, "v": v, "dasm_share_at_v": float((dcm >= v).mean()),
               "dasm_clipmax_pct": {q: float(np.percentile(dcm, q)) for q in (50, 90, 99)},
               "flex_clipmax_pct": {q: float(np.percentile(fcm, q)) for q in (50, 90, 99)}}


def fit(log):
    assert log.get("c4_pass"), "run check first"
    cl, ccs, base_evs, base_rows, Sb = gate("calib", log)
    n_f = Sb["flex_only_shown"]
    v, vinfo = fit_v(ccs)
    print(f"[fit] v: {vinfo}", flush=True)
    grid = []
    for g in G_GRID:
        _e, _r, Sg = run6(cl, ccs, "D1", g, v)
        grid.append({"g": g, "dasm_only_shown": Sg["dasm_only_shown"], "shown": Sg["shown"]})
    g = min(grid, key=lambda x: (abs(x["dasm_only_shown"] - n_f), -x["g"]))["g"]
    print(f"[fit] N_F (FlexSED-only shown spans in the baseline) = {n_f}; g = {g} "
          f"(DASM-only shown {[x['dasm_only_shown'] for x in grid if x['g'] == g][0]})", flush=True)
    log["bars"] = {"N_F": n_f, "v": v, "v_info": vinfo, "g": g, "g_grid": grid}
    log["fit"] = {"baseline": Sb}
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")
    show("baseline", Sb)
    for cell in ("D1", "D2"):
        evs, rows, S = run6(cl, ccs, cell, g, v)
        log["fit"][cell] = R5.compare(cl, base_evs, base_rows, evs, rows, S)
        show(cell, S)
        OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")
    elig = [k for k in ("D1", "D2") if log["fit"][k]["C_overlap"] < Sb["C_overlap"] and log["fit"][k]["C_onset"] < Sb["C_onset"]]
    pick = min(elig, key=lambda k: (log["fit"][k]["C_overlap"], log["fit"][k]["C_onset"], k)) if elig else None
    log["eligible"] = elig; log["pick"] = pick
    print(f"eligible (C-overlap AND C-onset below baseline): {elig}; PICK: {pick}", flush=True)
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


def heldout(log):
    pick = log.get("pick")
    if not pick:
        log["heldout"] = "not scored: no eligible cell on the 280 (as pre-registered); FlexSED stays"
        print(log["heldout"]); OUT.write_text(json.dumps(log, indent=1), encoding="utf-8"); return
    cl, ccs, base_evs, base_rows, Sb = gate("heldout", log)
    g, v = log["bars"]["g"], log["bars"]["v"]                          # frozen on the 280
    evs, rows, S = run6(cl, ccs, pick, g, v)
    S = R5.compare(cl, base_evs, base_rows, evs, rows, S)
    d_ov = [x["C_overlap"] - y["C_overlap"] for x, y in zip(rows, base_rows)]
    wc = set(json.loads(WAVCAPS.read_text(encoding="utf-8"))["heldout"])
    groups = {"complex": lambda c: c.get("stratum") == "complex", "random": lambda c: c.get("stratum") == "random",
              "in_wavcaps": lambda c: c["id"] in wc, "not_in_wavcaps": lambda c: c["id"] not in wc}
    strata = {}
    for s, f in groups.items():
        idx = [i for i, c in enumerate(cl) if f(c)]
        strata[s] = {"n": len(idx), "dC_overlap": D.boot([d_ov[i] for i in idx]) if idx else None,
                     "dfp": float(np.mean([rows[i]["fp"] - base_rows[i]["fp"] for i in idx])) if idx else None}
    S["strata"] = strata
    S["pass"] = bool(S["dC_overlap"][2] < 0)
    log["heldout"] = {"pick": pick, "baseline": Sb, "cell": S}
    show("baseline", Sb); show(pick, S)
    print(f"[heldout] {pick}: dC-overlap {S['dC_overlap']} -> {'PASS' if S['pass'] else 'fail'}; strata {strata}", flush=True)
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------- secondary rows (note added after the result)
def _true(c, e):
    return any(R.E._same(e.label, g["label"]) and min(e.end, g["end"]) - max(e.start, g["start"]) > 0 for g in c["events"])


def churn(cl, base_evs, evs):
    """spans removed / added against the baseline: a shown span (after the label filter) is 'removed' when no same-family
    cell span overlaps it, 'added' when no same-family baseline span overlaps it; split by true (overlaps a gold event of
    its family) vs false"""
    out = {"true_removed": 0, "false_removed": 0, "true_added": 0, "false_added": 0}
    ov = lambda a, b: key(a) == key(b) and min(a.end, b.end) - max(a.start, b.start) > 0
    for c, eb, ec in zip(cl, base_evs, evs):
        sb, sc = D._ev(eb), D._ev(ec)
        for e in sb:
            if not any(ov(e, x) for x in sc):
                out["true_removed" if _true(c, e) else "false_removed"] += 1
        for e in sc:
            if not any(ov(e, x) for x in sb):
                out["true_added" if _true(c, e) else "false_added"] += 1
    out["net_true"] = out["true_added"] - out["true_removed"]
    out["net_false"] = out["false_added"] - out["false_removed"]
    return out


def secondary(log):
    g, v, pick = log["bars"]["g"], log["bars"]["v"], log["pick"]
    res = {"note": "reported only; frozen bars; D2 not scored on the 415 (only the pick goes there)"}
    for set_name, cells in (("calib", ("D1", "D2")), ("heldout", (pick,) if pick else ())):
        R.use_set(set_name)
        cl = D.usable()
        ccs = [load6(c["id"]) for c in cl]
        base_evs = [stack6(cc, "ship")[0] for cc in ccs]
        cell_evs = {k: [stack6(cc, k, g, v)[0] for cc in ccs] for k in cells}
        res[set_name] = {}
        for filt in ("lists", "depictable"):
            config.LABEL_FILTER = filt
            base_rows = [D.clip_cost(c, ev) for c, ev in zip(cl, base_evs)]
            Sb = R4.summary(cl, base_rows); Sb["shown"] = int(sum(len(D._ev(e)) for e in base_evs))
            res[set_name][filt] = {"baseline": Sb}
            print(f"[secondary {set_name} {filt}] baseline C-ov {Sb['C_overlap']:.3f} C-on {Sb['C_onset']:.3f} rec "
                  f"{Sb['recall_overlap']:.1%} fp {Sb['fp']} ({Sb['fp_per_min']:.2f}/min) conseq {Sb['n_conseq']} shown {Sb['shown']}", flush=True)
            for k, evs in cell_evs.items():
                rows = [D.clip_cost(c, ev) for c, ev in zip(cl, evs)]
                S = R4.summary(cl, rows); S["shown"] = int(sum(len(D._ev(e)) for e in evs))
                S["dC_overlap"] = D.boot([x["C_overlap"] - y["C_overlap"] for x, y in zip(rows, base_rows)])
                S["dC_onset"] = D.boot([x["C_onset"] - y["C_onset"] for x, y in zip(rows, base_rows)])
                S["churn"] = churn(cl, base_evs, evs)
                res[set_name][filt][k] = S
                print(f"[secondary {set_name} {filt}] {k} C-ov {S['C_overlap']:.3f} dC-ov {tuple(round(x, 3) for x in S['dC_overlap'])} "
                      f"C-on {S['C_onset']:.3f} dC-on {tuple(round(x, 3) for x in S['dC_onset'])} rec {S['recall_overlap']:.1%} "
                      f"on-rec {S['recall_onset']:.1%} fp {S['fp']} ({S['fp_per_min']:.2f}/min) shown {S['shown']} churn {S['churn']}", flush=True)
        config.LABEL_FILTER = "lists"
    log["secondary"] = res
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("embed", "cache", "check", "fit", "heldout", "secondary"))
    ap.add_argument("--set", choices=("calib", "heldout"))
    a = ap.parse_args()
    if a.step == "embed":
        embed(); return
    if a.step == "cache":
        cache(a.set); return
    log = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    {"check": check, "fit": fit, "heldout": heldout, "secondary": secondary}[a.step](log)


if __name__ == "__main__":
    main()
