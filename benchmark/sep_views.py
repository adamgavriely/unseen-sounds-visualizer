"""V4, amendment 2 (docs/prereg_detector_v5.md): cleaned-audio views for PSED.

HTDemucs (htdemucs_ft, Meta; stems vocals / drums / bass / other) gives view B = original
minus vocals ("speech removed"), view C = original minus drums minus bass ("music removed";
"other" is kept because it holds ambient sounds as well as melodic instruments) and view
D = original minus vocals, drums and bass; each view = 0.9 * residual + 0.1 * original. PSED
scores A (original) and the three views. (Amendment 3: HTDemucs replaces SAM-Audio, whose
2023-era dependencies and hidden 15-GB reranker could not be made to run on the cluster.) A candidate span is kept iff its best PSED family score
over the four versions >= one tau, chosen at 2.6 false spans/min on the calibration set.
Two candidate sources are declared: 'orig' (loose boxes on A) and 'union' (loose boxes on
any version, merged per family). Nothing class-specific.

    python -m benchmark.sep_views --separate --set calib [--limit 20] [--twice]   # env msproj, GPU
    python -m benchmark.sep_views --psed --set calib                               # env psed, GPU
    python -m benchmark.sep_views --prompt-check                                   # env msproj, CPU (first 20 calib clips)
    python -m benchmark.sep_views --select                                         # env msproj, CPU -> benchmark/sep_v4_setting.json
    python -m benchmark.sep_views --sliceb                                         # env msproj, CPU, once
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark import audioset_detector_eval as E

SEP_ROOT = _ROOT / "data" / "work" / "sep_views"
MODEL_ID = "htdemucs_ft"
STEMS_REMOVED = {"B": ("vocals",), "C": ("drums", "bass"), "D": ("vocals", "drums", "bass")}
VIEWS = ["B", "C", "D"]
VERSIONS = ["A"] + VIEWS
BLEND = 0.9
SR = 16000
LOOSE = 0.05
FP_TARGET = 2.6
GRID = [round(x, 2) for x in np.arange(0.05, 0.96, 0.01)]
MIN_DUR = 0.5
OUT = _ROOT / "benchmark" / "sep_v4_setting.json"
BASE_GRID = [0.15, 0.20, 0.25, 0.30, 0.35]     # PSED's own bar inside the rule (correction, amendment 3): at 0.15 alone PSED
                                                # already exceeds the 2.6/min target on the calibration set (5.4/min), so an
                                                # additive rule can only meet the target if the base bar rises, as in fusion_v5


# ----------------------------------------------------------------------------- separation (env msproj, GPU)
def separate(which: str, limit: int = 0, twice: bool = False):
    import torch, soundfile as sf, librosa
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    E.use_set(which)
    out = SEP_ROOT / which; out.mkdir(parents=True, exist_ok=True)
    qc_f = out / "qc.json"
    qc = json.loads(qc_f.read_text(encoding="utf-8")) if qc_f.exists() else {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(MODEL_ID).to(dev).eval()
    osr = model.samplerate; names = list(model.sources)
    clips = E.clips()[:limit] if limit else E.clips()
    with tempfile.TemporaryDirectory() as td:
        for i, c in enumerate(clips, 1):
            if all((out / f"{c['id']}__{v}.wav").exists() for v in VIEWS) and c["id"] in qc:
                continue
            wav = Path(td) / "a.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(E.VIDEOS / f"{c['id']}.mp4"), "-vn", "-ac", "1",
                            "-ar", str(SR), str(wav)], check=True)
            orig16, _ = librosa.load(str(wav), sr=SR, mono=True)
            orig, _ = librosa.load(str(wav), sr=osr, mono=True)

            def run(seed=0):
                torch.manual_seed(seed)
                x = torch.from_numpy(orig).float()[None, None].repeat(1, 2, 1).to(dev)      # (batch, stereo, time)
                ref = x.mean(1, keepdim=True); mean, std = ref.mean(), ref.std() + 1e-8
                with torch.inference_mode():
                    st = apply_model(model, (x - mean) / std, shifts=0, split=True, overlap=0.25, progress=False)[0]
                return (st * std + mean).mean(1).cpu().numpy()                                    # (stems, time) mono

            stems = run(0)
            n = min(len(orig), stems.shape[1]); orig = orig[:n]; stems = stems[:, :n]
            S = {nm: stems[k] for k, nm in enumerate(names)}
            views = {v: orig - sum(S[nm] for nm in STEMS_REMOVED[v]) for v in VIEWS}
            rms = lambda a: float(np.sqrt(np.mean(a ** 2)) + 1e-9)
            rec = stems.sum(0)
            entry = {"sr_model": int(osr), "len_ratio": n / max(1, len(orig16) * osr // SR), "rms_orig": rms(orig),
                     "rms": {v: rms(views[v]) for v in VIEWS}, "rms_stem": {nm: rms(S[nm]) for nm in names},
                     "nan": bool(np.isnan(stems).any()),
                     "recon_err": float(np.mean(np.abs(orig - rec)) / (np.mean(np.abs(orig)) + 1e-9))}
            k = min(n, osr * 3); xc = np.correlate(orig[:k], rec[:k], "full"); entry["lag_samples"] = int(np.argmax(xc) - (k - 1))
            if twice and i == 1:
                entry["determinism_maxdiff"] = float(np.abs(run(0)[:, :n] - stems).max())
            for v in VIEWS:
                x = BLEND * views[v] + (1 - BLEND) * orig
                x16 = librosa.resample(x.astype(np.float32), orig_sr=osr, target_sr=SR)[: len(orig16)]
                if len(x16) < len(orig16):
                    x16 = np.pad(x16, (0, len(orig16) - len(x16)))
                entry.setdefault("peak", {})[v] = float(np.abs(x16).max())
                sf.write(str(out / f"{c['id']}__{v}.wav"), np.clip(x16, -1, 1), SR, subtype="PCM_16")
            qc[c["id"]] = entry
            if i % 20 == 0 or i == len(clips):
                qc_f.write_text(json.dumps(qc), encoding="utf-8")
                print(f"[sep] {which}: {i}/{len(clips)}", flush=True)
    qc_f.write_text(json.dumps(qc), encoding="utf-8")
    print(f"[sep] {which}: done, {len(qc)} clips -> {out}")


# ----------------------------------------------------------------------------- PSED on the views (env psed, GPU)
def psed_views(which: str):
    from src.stage4_audio_event_detection.psed_infer import cache_clips
    E.use_set(which)
    for v in VIEWS:
        items = [SEP_ROOT / which / f"{c['id']}__{v}.wav" for c in E.clips()]
        items = [p for p in items if p.exists()]
        raw = E.WIN / f"psed_{v}_raw"; outdir = E.WIN / f"psed_{v}"; outdir.mkdir(parents=True, exist_ok=True)
        n = cache_clips(items, out_dir=raw)
        for f in raw.glob("*.npz"):
            dst = outdir / f.name.replace(f"__{v}", "")
            if not dst.exists():
                dst.write_bytes(f.read_bytes())
        print(f"[psed] {which}/{v}: {len(items)} views, {n} scored now -> {outdir}", flush=True)


# ----------------------------------------------------------------------------- shared analysis helpers (CPU)
def load(which: str):
    """[(clip, {version: (fw, times, labels)})] for clips cached in every version"""
    E.use_set(which)
    out = []
    for c in E.clips():
        d = {}
        for v in VERSIONS:
            p = E.WIN / ("psed" if v == "A" else f"psed_{v}") / f"{c['id']}.npz"
            if not p.exists():
                d = None; break
            d[v] = E._load(p)
        if d:
            out.append((c, d))
    return out


def boxes(fw, t, lab, bar):
    from src.stage4_audio_event_detection import _extract_events
    from src.labels import is_salient_nonspeech, is_music
    return [(e.label, e.start, e.end, e.confidence) for e in _extract_events(fw, t, lab, bar, None, MIN_DUR, low=bar * 0.5)
            if is_salient_nonspeech(e.label) and not is_music(e.label)]


_FAM = {}


def fam_idx(labels, label):
    key = (id(labels), label)
    if key not in _FAM:
        _FAM[key] = [i for i, l in enumerate(labels) if E._same(l, label)]
    return _FAM[key]


def span_score(d, v, label, a, b):
    fw, t, lab = d[v]
    m = (t >= a) & (t <= b)
    idx = fam_idx(lab, label)
    return float(fw[m][:, idx].max()) if m.any() and idx else 0.0


def gold_events(c):
    from src.labels import is_salient_nonspeech, is_music
    return [g for g in c["events"] if is_salient_nonspeech(g["label"]) and not is_music(g["label"])]


def is_true(box, gold):
    lab, a, b = box[0], box[1], box[2]
    return any(E._same(lab, g["label"]) and E._overlap_ok(a, b, g["start"], g["end"]) for g in gold)


def candidates(d, source: str):
    """loose candidate spans (label, start, end) from A only, or merged per family over all versions"""
    if source == "orig":
        return [(b[0], b[1], b[2]) for b in boxes(*d["A"], LOOSE)]
    allb = [(b[0], b[1], b[2]) for v in VERSIONS for b in boxes(*d[v], LOOSE)]
    merged = []
    for lab, a, b in sorted(allb, key=lambda x: (x[0], x[1])):
        for m in merged:
            if E._same(m[0], lab) and min(m[2], b) - max(m[1], a) > 0:
                m[1], m[2] = min(m[1], a), max(m[2], b); break
        else:
            merged.append([lab, a, b])
    return [tuple(m) for m in merged]


def scored(data, source: str):
    """per clip: [(label, start, end, {version: score}, max_score)] for every candidate"""
    out = []
    for c, d in data:
        rows = []
        for lab, a, b in candidates(d, source):
            s = {v: span_score(d, v, lab, a, b) for v in VERSIONS}
            rows.append((lab, a, b, s, max(s.values())))
        out.append((c, rows))
    return out


def run_rule(sc, tau, base_bar, single_view=None):
    """detections: candidates with (A-score >= base_bar) or (score >= tau); score = max over
    views, or one view's score when single_view is given. Returns per-clip stats."""
    per_clip = []
    for c, rows in sc:
        gold = gold_events(c)
        dets = []
        for lab, a, b, s, mx in rows:
            v = s[single_view] if single_view else mx
            if s["A"] >= base_bar or v >= tau:
                dets.append((lab, a, b, s))
        hm = tm = hc = tc = ha = ta = 0
        for g in gold:
            m = any(E._same(x[0], g["label"]) and E._overlap_ok(x[1], x[2], g["start"], g["end"]) for x in dets)
            ta += 1; ha += m
            if g["consequential"]:
                tc += 1; hc += m
                if g["masked"]:
                    tm += 1; hm += m
        fp = sum(1 for x in dets if not is_true(x, c["events"]))
        per_clip.append({"id": c["id"], "hm": hm, "tm": tm, "hc": hc, "tc": tc, "ha": ha, "ta": ta, "fp": fp,
                         "minutes": c["duration"] / 60.0, "dets": dets})
    return per_clip


def summarise(pc):
    S = lambda k: sum(x[k] for x in pc)
    return {"masked": S("hm") / max(1, S("tm")), "conseq": S("hc") / max(1, S("tc")), "all": S("ha") / max(1, S("ta")),
            "fp": S("fp") / max(1e-6, S("minutes")), "n_masked": S("tm"), "n_conseq": S("tc")}


def loosest_tau(sc, base_bar, single_view=None):
    for tau in GRID:
        r = summarise(run_rule(sc, tau, base_bar, single_view))
        if r["fp"] <= FP_TARGET:
            return tau, r
    return GRID[-1], summarise(run_rule(sc, GRID[-1], base_bar, single_view))


def psed_bar():
    f = _ROOT / "benchmark" / "detector_calib.json"
    return float(json.loads(f.read_text(encoding="utf-8"))["bars"]["psed"]) if f.exists() else 0.15


# ----------------------------------------------------------------------------- prompt check (first 20 calib clips)
def prompt_check(n_clips: int = 20):
    data = load("calib")[:n_clips]
    lost = {v: 0 for v in ("B", "C")}; tot = 0
    for c, d in data:
        for g in gold_events(c):
            if not g["consequential"]:
                continue
            tot += 1
            a = span_score(d, "A", g["label"], g["start"], g["end"])
            for v in ("B", "C"):
                if a - span_score(d, v, g["label"], g["start"], g["end"]) > 0.3:
                    lost[v] += 1
    res = {"clips": len(data), "consequential_events": tot, "lost_gt_0.3": lost,
           "share": {v: lost[v] / max(1, tot) for v in lost}, "over_removal": any(lost[v] / max(1, tot) > 0.25 for v in lost)}
    print("[prompt-check]", json.dumps(res))
    (_ROOT / "benchmark" / "sep_v4_prompt_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


# ----------------------------------------------------------------------------- selection (calibration set)
def select():
    from sklearn.metrics import roc_auc_score
    data = load("calib")
    bb = psed_bar()
    print(f"[select] {len(data)} calibration clips with A+B+C+D; PSED bar {bb:.2f}")
    rng = np.random.default_rng(0); idx = rng.permutation(len(data)); half = len(data) // 2
    halves = [[data[i] for i in idx[:half]], [data[i] for i in idx[half:]]]
    res = {"when": datetime.now().isoformat(timespec="minutes"), "fp_target": FP_TARGET, "loose": LOOSE, "psed_bar": bb,
           "grid_step": 0.01, "clips": len(data)}
    # control: PSED alone (its own bar, and the loosest bar under the target on this set)
    sc_orig = scored(data, "orig")
    ctrl = summarise(run_rule(sc_orig, 9.9, bb))
    ctrl_tau, ctrl_at = loosest_tau(sc_orig, 9.9, "A")
    res["control"] = {"at_psed_bar": ctrl, "loosest_A": {"tau": ctrl_tau, **ctrl_at}}
    print(f"[select] control PSED@{bb:.2f}: masked {ctrl['masked']:.1%} conseq {ctrl['conseq']:.1%} all {ctrl['all']:.1%} FP {ctrl['fp']:.2f}")
    print(f"[select] control PSED at the loosest bar under {FP_TARGET}/min: bar {ctrl_tau:.2f} masked {ctrl_at['masked']:.1%} conseq {ctrl_at['conseq']:.1%} all {ctrl_at['all']:.1%} FP {ctrl_at['fp']:.2f}")
    # AUROC on candidate boxes (orig source): A alone vs max over views
    y, sA, sM = [], [], []
    for c, rows in sc_orig:
        gold = gold_events(c)
        for lab, a, b, s, mx in rows:
            y.append(int(is_true((lab, a, b), gold))); sA.append(s["A"]); sM.append(mx)
    res["auroc"] = {"boxes": len(y), "true": int(sum(y)), "A": roc_auc_score(y, sA), "max_views": roc_auc_score(y, sM)}
    for v in VIEWS:
        res["auroc"][v] = roc_auc_score(y, [r[3][v] for c, rows in sc_orig for r in rows])
    print(f"[select] AUROC on {len(y)} candidate boxes: A {res['auroc']['A']:.3f}  max-views {res['auroc']['max_views']:.3f}  "
          + "  ".join(f"{v} {res['auroc'][v]:.3f}" for v in VIEWS))
    # the two declared variants + single-view diagnostics; base bar x tau grid, loosest pair under the target
    rows = []
    for source in ("orig", "union"):
        sc = sc_orig if source == "orig" else scored(data, "union")
        sc_h = [[x for x in sc if x[0]["id"] in {c["id"] for c, _ in h}] for h in halves]
        sc_h_ctrl = [[x for x in sc_orig if x[0]["id"] in {c["id"] for c, _ in h}] for h in halves]   # control = PSED on A, same for both sources
        for name, sv in [("max", None)] + [(v, v) for v in VIEWS]:
            def best_pair(scx):
                best = None
                for bb_ in BASE_GRID:
                    tau, r = loosest_tau(scx, bb_, sv)
                    if r["fp"] <= FP_TARGET and (best is None or r["conseq"] > best[2]["conseq"]):
                        best = (bb_, tau, r)
                return best or (BASE_GRID[-1], GRID[-1], summarise(run_rule(scx, GRID[-1], BASE_GRID[-1], sv)))
            b_full, tau, full = best_pair(sc)
            hh = []
            for k in (0, 1):
                bA, tA, _ = best_pair(sc_h[k])
                rB = summarise(run_rule(sc_h[1 - k], tA, bA, sv))
                cB = summarise(run_rule(sc_h_ctrl[1 - k], ctrl_tau, 9.9, "A"))
                hh.append({"base_chosen": bA, "tau_chosen": tA, "held": rB, "control_held": cB})
            row = {"source": source, "score": name, "base_bar": b_full, "tau": tau, **full, "halves": hh}
            rows.append(row)
            print(f"[select] {source:5s} {name:3s} base={b_full:.2f} tau={tau:.2f}: masked {full['masked']:.1%} conseq {full['conseq']:.1%} all {full['all']:.1%} FP {full['fp']:.2f}"
                  + " | held: " + " / ".join(f"conseq {h['held']['conseq']:.1%} (ctrl {h['control_held']['conseq']:.1%}) FP {h['held']['fp']:.2f}" for h in hh))
    res["rows"] = rows
    # go/no-go and winner between the two declared variants (score = max over views)
    def ok(r):
        gain = [h["held"]["conseq"] - h["control_held"]["conseq"] for h in r["halves"]]
        return (r["fp"] <= FP_TARGET and all(h["held"]["fp"] <= FP_TARGET for h in r["halves"])
                and min(gain) >= 0.02 and r["all"] >= ctrl_at["all"] - 0.01)
    auroc_ok = res["auroc"]["max_views"] >= res["auroc"]["A"] + 0.02
    cands = [r for r in rows if r["score"] == "max" and ok(r)]
    win = None
    if auroc_ok and cands:
        top = max(cands, key=lambda r: min(h["held"]["conseq"] for h in r["halves"]))
        near = [r for r in cands if min(h["held"]["conseq"] for h in top["halves"]) - min(h["held"]["conseq"] for h in r["halves"]) < 0.02]
        win = next((r for r in near if r["source"] == "orig"), top)
    res["go"] = {"auroc_ok": auroc_ok, "variants_ok": [r["source"] for r in cands], "winner": win}
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[select] AUROC gate {'passed' if auroc_ok else 'FAILED'}; winner: {win['source'] + ' tau ' + str(win['tau']) if win else 'none -> V4 not run on slice B'}")
    print("->", OUT)


# ----------------------------------------------------------------------------- slice B, once
def sliceb():
    s = json.loads(OUT.read_text(encoding="utf-8"))
    win = s["go"]["winner"]
    if not win:
        print("[sliceB] no winner on calibration; per the pre-registration V4 is not run on slice B"); return
    data = load("sliceB"); bb = s["psed_bar"]
    print(f"[sliceB] {len(data)} clips with A+B+C+D")
    sc = scored(data, win["source"])
    cb = s["control"]["loosest_A"]["tau"]
    ctrl_pc = run_rule(sc, cb, 9.9, "A"); v4_pc = run_rule(sc, win["tau"], win["base_bar"])
    ctrl, v4 = summarise(ctrl_pc), summarise(v4_pc)
    print(f"[sliceB] PSED@{cb:.2f} (loosest bar under the target on calibration): masked {ctrl['masked']:.1%} all {ctrl['all']:.1%} FP {ctrl['fp']:.2f}")
    print(f"[sliceB] V4 {win['source']} tau {win['tau']:.2f}: masked {v4['masked']:.1%} all {v4['all']:.1%} FP {v4['fp']:.2f}")
    # paired clip bootstrap of the masked-recall difference
    rng = np.random.default_rng(0); n = len(ctrl_pc); diffs = []
    for _ in range(1000):
        ix = rng.integers(0, n, n)
        tm = sum(ctrl_pc[i]["tm"] for i in ix)
        if tm == 0: continue
        diffs.append((sum(v4_pc[i]["hm"] for i in ix) - sum(ctrl_pc[i]["hm"] for i in ix)) / tm)
    ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    # 2x2 event agreement, top-clip removal, per-view attribution of rescued detections
    def event_hits(pc, sc):
        out = []
        for x, (c, rows) in zip(pc, sc):
            for g in gold_events(c):
                if g["consequential"] and g["masked"]:
                    out.append(any(E._same(d_[0], g["label"]) and E._overlap_ok(d_[1], d_[2], g["start"], g["end"]) for d_ in x["dets"]))
        return out
    hc, hv = event_hits(ctrl_pc, sc), event_hits(v4_pc, sc)
    both = sum(a and b for a, b in zip(hc, hv)); only_v4 = sum((not a) and b for a, b in zip(hc, hv))
    only_c = sum(a and not b for a, b in zip(hc, hv)); neither = sum((not a) and (not b) for a, b in zip(hc, hv))
    top = max(range(n), key=lambda i: ctrl_pc[i]["tm"] - ctrl_pc[i]["hm"])
    drop = lambda pc: summarise([x for i, x in enumerate(pc) if i != top])
    attrib = {v: 0 for v in VERSIONS}
    for x in v4_pc:
        for lab, a, b, sc_ in x["dets"]:
            if sc_["A"] < win["base_bar"]:
                attrib[max(sc_, key=sc_.get)] += 1
    curve = {name: [] for name in ["A"] + ["max"] + VIEWS}
    for tau in GRID[::5]:
        curve["A"].append((tau,) + tuple(summarise(run_rule(sc, tau, 9.9, "A"))[k] for k in ("masked", "fp")))
        curve["max"].append((tau,) + tuple(summarise(run_rule(sc, tau, bb))[k] for k in ("masked", "fp")))
        for v in VIEWS:
            curve[v].append((tau,) + tuple(summarise(run_rule(sc, tau, bb, v))[k] for k in ("masked", "fp")))
    passed = bool(v4["masked"] >= 0.70 and v4["fp"] <= FP_TARGET and ci[0] > 0)
    res = {"control": ctrl, "v4": v4, "diff_masked": v4["masked"] - ctrl["masked"], "ci95_diff_clip_bootstrap": ci,
           "agreement": {"both": both, "only_v4": only_v4, "only_psed": only_c, "neither": neither},
           "top_clip_removed": {"id": ctrl_pc[top]["id"], "control": drop(ctrl_pc), "v4": drop(v4_pc)},
           "rescued_by_view": attrib, "curves": curve, "passed": passed}
    s["sliceB"] = res
    OUT.write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(f"[sliceB] diff {res['diff_masked']:+.1%}  CI95 [{ci[0]:+.1%}, {ci[1]:+.1%}]  agreement {res['agreement']}  rescued by {attrib}")
    print(f"[sliceB] pass rule (>= 70% at <= 2.6/min and CI above 0): {'PASSED' if passed else 'FAILED'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--separate", action="store_true"); ap.add_argument("--psed", action="store_true")
    ap.add_argument("--prompt-check", action="store_true"); ap.add_argument("--select", action="store_true")
    ap.add_argument("--sliceb", action="store_true")
    ap.add_argument("--set", choices=("calib", "sliceB"), default="calib")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--twice", action="store_true")
    a = ap.parse_args()
    if a.separate: separate(a.set, a.limit, a.twice)
    if a.psed: psed_views(a.set)
    if a.prompt_check: prompt_check()
    if a.select: select()
    if a.sliceb: sliceb()


if __name__ == "__main__":
    main()
