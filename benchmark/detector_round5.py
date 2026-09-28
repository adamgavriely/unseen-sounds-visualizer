"""Detector round 5 (docs/prereg_v4.md, 2026-09-28): does a newer AudioSet tagger beat BEATs as the stage-4 framewise tagger?

  A  EAT-large fine-tuned on AS2M      HF worstchan/EAT-large_epoch20_finetune_AS2M (github.com/cwx-worst-one/EAT)
  B  Dasheng-base AudioSet fine-tuned  Zenodo dasheng_audioset_mAP497.pt (github.com/RicherMans/Dasheng)

Each replaces BEATs framewise in the shipped stack (BEATs + FlexSED 0.8 + FlexSED clip veto 0.3 + self-veto, PANNs off),
scored on exactly BEATs' 2-s / 0.25-s windows. Cost C, clip sets and bootstrap as amendment 24 (detector_round2.py).

    python benchmark/detector_round5.py cache --model eat|dasheng --set calib|heldout   # GPU
    python benchmark/detector_round5.py check      # gates 2-3 (windows, labels, order) on both sets
    python benchmark/detector_round5.py fit        # gate 1 + four cells on the 280 + the pick
    python benchmark/detector_round5.py heldout    # the pick on the 415 (only if there is one)
"""
from __future__ import annotations

import argparse
import csv
import json
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
from src.labels import canonical
from src.stage4_audio_event_detection import _extract_events

OUT = _ROOT / "benchmark" / "detector_round5.json"
R4_JSON = _ROOT / "benchmark" / "detector_round4.json"
EAT_LABELS = _ROOT / "benchmark" / "round5_eat_labels.csv"      # official EAT inference/labels.csv (index, mid, name)
MID_NAMES = _ROOT / "src" / "audioset_mid_names.json"
CACHE = {"eat": "eat_cache", "dasheng": "dasheng_cache"}
MODELS = ("eat", "dasheng")
B_SHIPPED = 0.1218                   # config.py use_shipped (the baseline bar)
S0 = 0.3413                          # round 4's kept share (57 / 167) -- the b-refit rule
D_GRID = [round(0.05 + 0.005 * i, 3) for i in range(181)]
EXPECT_BASE = {"calib": {"C_overlap": 3.043, "C_onset": 3.857, "recall_overlap": 0.513, "fp_per_min": 4.46},
               "heldout": {"C_overlap": 1.928, "C_onset": 2.207, "recall_overlap": 0.497, "fp_per_min": 3.30}}
TOL = {"C_overlap": 0.0005, "C_onset": 0.0005, "recall_overlap": 0.0005, "fp_per_min": 0.005}
key = lambda e: canonical(e.label)

# ----------------------------------------------------------------------------- windows (copied from infer_beats)
SR, WINDOW, HOP, STAMP_OFFSET = 16000, 2.0, 0.25, 0.5


def windows(audio):
    """exactly src/stage4_audio_event_detection/beats_infer.infer_beats' windowing: (chunks [n, 32000], times [n])"""
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


def label_names():
    """index -> display name for both candidates: mid from EAT's official labels.csv, name via the ontology json (BEATs' route)"""
    mids = [row[1] for row in csv.reader(EAT_LABELS.open(encoding="utf-8"))]
    assert len(mids) == 527
    mid_names = json.loads(MID_NAMES.read_text("utf-8"))
    return mids, [mid_names.get(m, m) for m in mids]


# ----------------------------------------------------------------------------- models (fp32, no autocast)
def eat_model(device):
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    snap = Path(snapshot_download("worstchan/EAT-large_epoch20_finetune_AS2M"))
    pkg = types.ModuleType("eat_hf"); pkg.__path__ = [str(snap)]; sys.modules["eat_hf"] = pkg
    from eat_hf.eat_model import EAT                   # the repo's own class; no AutoModel (remote code targets transformers 4.51)
    cfg = types.SimpleNamespace(**json.loads((snap / "config.json").read_text("utf-8")))
    cfg.img_size = tuple(cfg.img_size)
    m = EAT(cfg)
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in load_file(str(snap / "model.safetensors")).items()}
    m.load_state_dict(sd, strict=True)
    assert m.mode == "finetune" and m.head.out_features == 527
    m = m.to(device).eval()

    def score(chunks):
        import torchaudio
        mels = []
        for w in chunks:
            x = torch.from_numpy(w).float()
            x = x - x.mean()
            mel = torchaudio.compliance.kaldi.fbank(x.unsqueeze(0), htk_compat=True, sample_frequency=16000, use_energy=False,
                                                    window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10)
            assert mel.shape[0] == 198, mel.shape
            mel = torch.nn.functional.pad(mel, (0, 0, 0, 208 - mel.shape[0]))   # zero-pad to 208 frames (13 patches)
            mels.append((mel - (-4.268)) / (4.569 * 2))
        x = torch.stack(mels).unsqueeze(1).to(device)           # [B, 1, 208, 128]
        with torch.no_grad():
            logits = m(x)                                        # CLS -> fc_norm -> head: LOGITS
        return torch.sigmoid(logits).float().cpu().numpy()       # sigmoid applied once, here
    return score, {"weights": "worstchan/EAT-large_epoch20_finetune_AS2M", "snapshot": snap.name, "load": "strict=True"}


def dasheng_model(device):
    import torch
    import dasheng

    class DashengAudiosetClassifier(torch.nn.Module):        # the official README's classifier
        def __init__(self):
            super().__init__()
            self.dashengmodel = dasheng.dasheng_base()
            self.classifier = torch.nn.Sequential(torch.nn.LayerNorm(self.dashengmodel.embed_dim),
                                                  torch.nn.Linear(self.dashengmodel.embed_dim, 527))

        def load(self, state_dict):
            # dasheng's load_state_dict override returns None, so the key audit is done by hand
            own = set(self.dashengmodel.state_dict().keys())
            self.dashengmodel.load_state_dict(state_dict, strict=False)
            self.classifier.load_state_dict({k.replace("outputlayer.", ""): v for k, v in state_dict.items() if "outputlayer" in k})
            return types.SimpleNamespace(missing_keys=sorted(own - set(state_dict)), unexpected_keys=sorted(set(state_dict) - own))

        def forward(self, x):
            return self.classifier(self.dashengmodel(x).mean(1)).sigmoid()   # sigmoid ALREADY inside: do not re-apply

    mdl = DashengAudiosetClassifier()
    ck = torch.hub.load_state_dict_from_url("https://zenodo.org/records/13315686/files/dasheng_audioset_mAP497.pt?download=1",
                                            map_location="cpu")
    res = mdl.load(ck)
    missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
    assert not missing and all(k.startswith("outputlayer.") for k in unexpected), (missing, unexpected)
    mdl = mdl.to(device).eval()

    def score(chunks):
        with torch.no_grad():
            return mdl(torch.from_numpy(np.asarray(chunks)).float().to(device)).float().cpu().numpy()
    return score, {"weights": "zenodo 13315686 dasheng_audioset_mAP497.pt", "missing": missing, "unexpected": unexpected}


def cache(model, set_name, device="cuda", batch=64):
    import librosa
    R.use_set(set_name)
    out = R.E.WIN / CACHE[model]; out.mkdir(parents=True, exist_ok=True)
    score, info = (eat_model if model == "eat" else dasheng_model)(device)
    print(f"[cache] {model} {set_name}: {info}", flush=True)
    _mids, names = label_names()
    n = 0
    with tempfile.TemporaryDirectory() as td:
        for c in R.E.clips():
            dst = out / f"{c['id']}.npz"
            bp = R.E.WIN / "beats" / f"{c['id']}.npz"
            if dst.exists() or not bp.exists():
                continue
            wav = Path(td) / "a.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(R.E.VIDEOS / f"{c['id']}.mp4"), "-vn", "-ac", "1",
                            "-ar", "16000", str(wav)], check=True)
            audio, _ = librosa.load(str(wav), sr=SR, mono=True)
            chunks, times = windows(audio)
            _bf, bt, _bl = R.load(bp)
            assert len(bt) == len(times) and np.allclose(bt, times, atol=1e-4), f"window mismatch on {c['id']}"
            fw = np.concatenate([score(chunks[i:i + batch]) for i in range(0, len(chunks), batch)], axis=0)
            assert fw.shape == (len(times), 527)
            np.savez_compressed(dst, fw=fw.astype(np.float16), times=np.asarray(times, np.float32), labels=np.array(names)); n += 1
    print(f"[cache] {model} {set_name}: {n} clips scored now -> {out}", flush=True)


# ----------------------------------------------------------------------------- gates 2-3
def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def check(log):
    mids, names = label_names()
    from panns_inference import config as pc
    g = {"eat_csv_equals_panns_order": list(pc.ids) == mids}
    print(f"[labels] EAT labels.csv mids == panns_inference class_labels_indices order: {g['eat_csv_equals_panns_order']}")
    ok = g["eat_csv_equals_panns_order"]
    for set_name in ("calib", "heldout"):
        R.use_set(set_name)
        cl = [c for c in R.E.clips() if (R.E.WIN / "beats" / f"{c['id']}.npz").exists()]
        for model in MODELS:
            bad, missing, name_ok = 0, 0, True
            for c in cl:
                p = R.E.WIN / CACHE[model] / f"{c['id']}.npz"
                if not p.exists():
                    missing += 1; continue
                fw, t, labs = R.load(p)
                _bf, bt, bl = R.load(R.E.WIN / "beats" / f"{c['id']}.npz")
                bad += not (len(t) == len(bt) and np.allclose(t, bt, atol=1e-4))
                name_ok &= sorted(labs) == sorted(bl) and labs == names
            g[f"{model}_{set_name}"] = {"clips": len(cl), "missing": missing, "time_mismatch": bad, "names_equal_beats": bool(name_ok)}
            print(f"[windows/labels] {model} {set_name}: {g[f'{model}_{set_name}']}")
            ok &= missing == 0 and bad == 0 and name_ok
    R.use_set("calib")                                       # order check: the 280, scores only
    cl = [c for c in R.E.clips() if (R.E.WIN / "beats" / f"{c['id']}.npz").exists()]
    for model in MODELS:
        B, C = [], []
        for c in cl:
            bf, _bt, bl = R.load(R.E.WIN / "beats" / f"{c['id']}.npz")
            cf, _ct, _cl = R.load(R.E.WIN / CACHE[model] / f"{c['id']}.npz")
            bmax = dict(zip(bl, bf.max(axis=0)))
            B.append([bmax[n] for n in names]); C.append(cf.max(axis=0))
        B, C = np.array(B), np.array(C)                      # [clips, 527], both in the candidate's index order
        # clarification 2 (prereg): the Spearman margin was mis-specified (a shared per-clip activity level lifts every
        # column pairing); gate = per-clip top-1 agreement with BEATs, true order >= 0.30 and >= 10x each +-1 shifted order
        rho = {s: float(np.nanmedian([spearman(np.roll(C, s, axis=1)[:, j], B[:, j]) for j in range(527)])) for s in (0, 1, -1)}
        pear = {s: float(np.nanmedian([np.corrcoef(np.roll(C, s, axis=1)[:, j], B[:, j])[0, 1] for j in range(527)])) for s in (0, 1, -1)}
        top1 = {s: float(np.mean(np.roll(C, s, axis=1).argmax(1) == B.argmax(1))) for s in (0, 1, -1)}
        passed = top1[0] >= 0.30 and top1[0] >= 10 * max(top1[1], top1[-1])
        g[f"{model}_order"] = {"top1_true": top1[0], "top1_shift+1": top1[1], "top1_shift-1": top1[-1], "pass": passed,
                               "descriptive": {"median_rho": rho, "median_pearson": pear, "old_rho_margin_gate_pass": rho[0] - max(rho[1], rho[-1]) >= 0.2}}
        print(f"[order] {model}: {g[f'{model}_order']}")
        ok &= passed
    g["pass"] = bool(ok)
    log["gates_2_3"] = g
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")
    if not ok:
        sys.exit("gate 2/3 failed: stop")


# ----------------------------------------------------------------------------- the stack with free bars
def load5(cid, model):
    b, f, p = R4.load3(cid)
    return b, f, p, R.load(R.E.WIN / CACHE[model] / f"{cid}.npz") if model else None


def pre(tag, f, aed):
    """round 4's stack_tagged up to the self-veto, with the tagger in slot 0 and AED bar `aed` (hysteresis 1.0).
    Returns (events after the FlexSED clip veto, ids of FlexSED-only spans, tagger clip-max per family)."""
    events = _extract_events(tag[0], tag[1], tag[2], aed, None, config.AED_MIN_DUR, low=aed * D.HYS)
    fev = _extract_events(f[0], f[1], f[2], R4.FBAR, None, config.AED_MIN_DUR, low=R4.FBAR * D.HYS)
    fresh = []
    for e in fev:
        tw = [x for x in events if key(x) == key(e) and x.start - 1.0 <= e.end and e.start - 1.0 <= x.end]
        if tw:
            for x in tw:
                x.start = min(x.start, e.start)
        else:
            fresh.append(e)
    flex_ids = {id(e) for e in fresh}
    events = events + fresh
    fpk = D.clip_peak(f)
    events = [e for e in events if fpk.get(key(e), 1.0) >= R4.FVETO]
    return events, flex_ids, D.clip_peak(tag)


def pool_vals(pres):
    return np.array([tpk.get(key(e), 1.0) for ev, fids, tpk in pres for e in ev if id(e) in fids], float)


def fit_b(pv):
    k = int(np.floor(S0 * len(pv) + 0.5))
    return (float(np.sort(pv)[::-1][k - 1]) if k > 0 else float("inf")), k


def finish(pr, b, disp):
    ev, fids, tpk = pr
    ev = [e for e in ev if id(e) not in fids or tpk.get(key(e), 1.0) >= b]
    return [e for e in ev if e.confidence >= disp]


def run(cl, cc, slot, aed, disp, b=None):
    """one arm over a clip set. slot 0 = BEATs, 3 = the candidate. b None -> refit by the round-4 share rule."""
    pres = [pre(x[slot], x[1], aed) for x in cc]
    pv = pool_vals(pres)
    k = None
    if b is None:
        b, k = fit_b(pv)
    evs = [finish(pr, b, disp) for pr in pres]
    rows = [D.clip_cost(c, ev) for c, ev in zip(cl, evs)]
    S = R4.summary(cl, rows)
    S.update({"b": b, "k": k, "pool": int(len(pv)), "kept_share": float((pv >= b).mean()) if len(pv) else None,
              "aed": aed, "disp": disp, "shown": int(sum(len(D._ev(ev)) for ev in evs))})
    return evs, rows, S


def compare(cl, base_evs, base_rows, evs, rows, S):
    pairs, e_b, e_c = [], [], []
    for c, eb, ec in zip(cl, base_evs, evs):
        a, b = R4.end_errors(c, eb), R4.end_errors(c, ec)
        e_b += list(a.values()); e_c += list(b.values())
        pairs.append([(a[k], b[k]) for k in a if k in b])
    d, lo, hi = R4.boot_med(pairs)
    npair = sum(len(p) for p in pairs)
    allp = np.array([x for p in pairs for x in p]) if npair else np.zeros((0, 2))
    S["end_err"] = {"paired_n": npair, "median_baseline": float(np.median(allp[:, 0])) if npair else None,
                    "median_cell": float(np.median(allp[:, 1])) if npair else None, "delta_baseline_minus_cell": [d, lo, hi],
                    "per_arm": {"baseline": [len(e_b), float(np.median(e_b)) if e_b else None],
                                "cell": [len(e_c), float(np.median(e_c)) if e_c else None]}}
    S["dC_overlap"] = D.boot([x["C_overlap"] - y["C_overlap"] for x, y in zip(rows, base_rows)])
    S["dC_onset"] = D.boot([x["C_onset"] - y["C_onset"] for x, y in zip(rows, base_rows)])
    return S


def show(name, S):
    ee = S.get("end_err", {})
    print(f"{name:12s} C-ov {S['C_overlap']:.3f} C-on {S['C_onset']:.3f} rec {S['recall_overlap']:.1%} on-rec {S['recall_onset']:.1%} "
          f"fp {S['fp']} ({S['fp_per_min']:.2f}/min) shown {S['shown']} pool {S['pool']} b {S['b']:.4f} disp {S['disp']} "
          f"dC-ov {S.get('dC_overlap')} dC-on {S.get('dC_onset')} end-err {ee.get('median_baseline')} -> {ee.get('median_cell')} "
          f"d {ee.get('delta_baseline_minus_cell')}", flush=True)


def usable5():
    cl = D.usable()
    miss = [(c["id"], m) for c in cl for m in MODELS if not (R.E.WIN / CACHE[m] / f"{c['id']}.npz").exists()]
    assert not miss, f"missing candidate caches: {miss[:5]} ... ({len(miss)})"
    return cl


def gate1(set_name, log):
    """round 4's tagged stack == detector_round2.stack (asserted inside R4.base), PANNs chain numbers, and the self-veto baseline"""
    cl, per, rows, S = R4.base(set_name)
    ok = R4.check_shipped(set_name, cl, rows, S)
    assert [c["id"] for c in cl] == [c["id"] for c in usable5()], "clip set differs"
    r4 = json.loads(R4_JSON.read_text(encoding="utf-8"))
    b4 = r4["test2_b"]["b"]
    cc = [load5(c["id"], None) for c in cl]
    ev4, _r, S4 = run(cl, cc, 0, D.AED, D.DISP, b=b4)
    ref = r4["test2_calib" if set_name == "calib" else "test2_heldout"]["self_veto"]
    ok4 = all(abs(S4[k] - ref[k]) <= 1e-9 for k in ("C_overlap", "C_onset", "recall_overlap", "fp_per_min"))
    for (c, _cc, _ev, _f, _p), e in zip(per, ev4):             # the free-bar stack == round 4's stack_tagged at the same bars
        assert R4.same_spans(e, R4.stack_tagged(c["id"], veto=("beats", b4))[0]), c["id"]
    evb, rowsb, Sb = run(cl, cc, 0, D.AED, D.DISP, b=B_SHIPPED)
    exp = EXPECT_BASE[set_name]
    okb = all(abs(Sb[k] - v) <= TOL[k] for k, v in exp.items())
    print(f"[gate1 {set_name}] PANNs chain {'OK' if ok else 'MISMATCH'}; self-veto at round-4 b {b4:.8f}: {'OK' if ok4 else 'MISMATCH'} "
          f"(C-ov {S4['C_overlap']:.4f}); baseline at shipped b {B_SHIPPED}: C-ov {Sb['C_overlap']:.4f} C-on {Sb['C_onset']:.4f} "
          f"rec {Sb['recall_overlap']:.4f} fp/min {Sb['fp_per_min']:.4f} -> expected {exp}: {'OK' if okb else 'MISMATCH'}; "
          f"same as round-4 b: {S4['C_overlap'] == Sb['C_overlap'] and S4['fp'] == Sb['fp'] and S4['recall_overlap'] == Sb['recall_overlap']}")
    log[f"gate1_{set_name}"] = {"clips": len(cl), "panns_chain": S, "panns_chain_ok": ok, "self_veto_round4_b": S4, "round4_b_ok": ok4,
                                "baseline_shipped_b": Sb, "baseline_ok": okb}
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")
    if not (ok and ok4):                  # okb is reported, not a stop: the clarification expects 0.1218 may drop the span on b
        sys.exit("gate 1 failed: stop")
    return cl, evb, rowsb, Sb


def fit(log):
    assert log.get("gates_2_3", {}).get("pass"), "run check first"
    cl, base_evs, base_rows, Sb = gate1("calib", log)
    log["fit"] = {"baseline": Sb}
    show("baseline", Sb)
    for model in MODELS:
        cc = [load5(c["id"], model) for c in cl]
        evs, rows, S = run(cl, cc, 3, D.AED, D.DISP)                     # primary: same bars, b refitted
        name = f"{'EAT' if model == 'eat' else 'Dasheng'}-P"
        log["fit"][name] = compare(cl, base_evs, base_rows, evs, rows, S); show(name, S)
        OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")
        grid = []                                                        # secondary: d refitted to the baseline's shown count
        for d in D_GRID:
            _e, _r, Sd = run(cl, cc, 3, d / 2, d)
            grid.append((d, Sd["shown"], Sd["b"]))
        dbest = min(grid, key=lambda g: (abs(g[1] - Sb["shown"]), -g[0]))[0]
        evs, rows, S = run(cl, cc, 3, dbest / 2, dbest)
        name = name[:-1] + "R"
        log["fit"][name] = compare(cl, base_evs, base_rows, evs, rows, S)
        log["fit"][name]["d_grid"] = [{"d": g[0], "shown": g[1], "b": g[2]} for g in grid]
        show(name, S)
        OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")
    cells = {k: v for k, v in log["fit"].items() if k != "baseline"}
    order = ["EAT-P", "EAT-R", "Dasheng-P", "Dasheng-R"]
    elig = [k for k in order if cells[k]["C_overlap"] < Sb["C_overlap"] and cells[k]["C_onset"] < Sb["C_onset"]]
    prim = lambda k: 0 if k.endswith("-P") else 1
    pick = min(elig, key=lambda k: (cells[k]["C_overlap"], cells[k]["C_onset"], prim(k))) if elig else None
    log["eligible"] = elig; log["pick"] = pick
    print(f"eligible (C-overlap AND C-onset below baseline): {elig}; PICK: {pick}")
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


def heldout(log):
    pick = log.get("pick")
    if not pick:
        log["heldout"] = "not scored: no eligible cell on the 280 (as pre-registered); BEATs stays"
        print(log["heldout"]); OUT.write_text(json.dumps(log, indent=1), encoding="utf-8"); return
    cl, base_evs, base_rows, Sb = gate1("heldout", log)
    fr = log["fit"][pick]
    model = "eat" if pick.startswith("EAT") else "dasheng"
    cc = [load5(c["id"], model) for c in cl]
    evs, rows, S = run(cl, cc, 3, fr["aed"], fr["disp"], b=fr["b"])     # frozen d and b
    S = compare(cl, base_evs, base_rows, evs, rows, S)
    d_ov = [x["C_overlap"] - y["C_overlap"] for x, y in zip(rows, base_rows)]
    strata = {}
    for s in ("complex", "random"):
        idx = [i for i, c in enumerate(cl) if c.get("stratum") == s]
        strata[s] = {"n": len(idx), "dC_overlap": D.boot([d_ov[i] for i in idx]) if idx else None,
                     "dfp": float(np.mean([rows[i]["fp"] - base_rows[i]["fp"] for i in idx])) if idx else None}
    S["strata"] = strata
    S["pass"] = bool(S["dC_overlap"][2] < 0)
    log["heldout"] = {"pick": pick, "baseline": Sb, "cell": S}
    show("baseline", Sb); show(pick, S)
    print(f"[heldout] {pick}: dC-overlap {S['dC_overlap']} -> {'PASS' if S['pass'] else 'fail'}; strata {strata}")
    OUT.write_text(json.dumps(log, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("cache", "check", "fit", "heldout"))
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--set", choices=("calib", "heldout"))
    a = ap.parse_args()
    if a.step == "cache":
        cache(a.model, a.set); return
    log = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    {"check": check, "fit": fit, "heldout": heldout}[a.step](log)


if __name__ == "__main__":
    main()
