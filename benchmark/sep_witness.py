"""Speech removal as a second witness for PSED (docs/prereg_detector_v5.md, variant V4; design
by Fable 2026-09-19): SAM-Audio (Meta, Dec 2025) with the prompt "speech" gives a residual;
view B = 0.9 * residual + 0.1 * original (Focus-Then-Listen 2026: pure stems hurt frozen
models); PSED is run on view B; a candidate PSED proposed on the original is accepted if
p_orig >= tau OR (p_orig >= tau_low AND p_B >= tau). Nothing class-specific; music is NOT
removed (sirens, alarms, whistles are tonal). A 1.0 view is kept as a diagnostic only.

    python -m benchmark.sep_witness --separate --set calib      # env samaudio, GPU: writes view-B wavs + speech RMS
    python -m benchmark.sep_witness --separate --set sliceB
    python -m benchmark.sep_witness --psed --set calib          # env psed, GPU: PSED on the views -> audioset_*_windows/psed_sep09, psed_sep10
    python -m benchmark.sep_witness --analyse                   # env msproj: AUROC / delta / witness recall on the calibration set
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

SEP_ROOT = _ROOT / "data" / "work" / "sep_samaudio"
MODEL_ID = "facebook/sam-audio-base"
PROMPT = "speech"
BLENDS = {"psed_sep09": 0.9, "psed_sep10": 1.0}
SR = 16000


# ----------------------------------------------------------------------------- separation (env samaudio)
def separate(which: str):
    import torch, soundfile as sf, librosa
    from sam_audio import SAMAudio, SAMAudioProcessor
    E.use_set(which)
    out = SEP_ROOT / which; out.mkdir(parents=True, exist_ok=True)
    model = SAMAudio.from_pretrained(MODEL_ID).eval().cuda()
    proc = SAMAudioProcessor.from_pretrained(MODEL_ID)
    stats = {}
    stats_f = out / "speech_rms.json"
    if stats_f.exists():
        stats = json.loads(stats_f.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as td:
        for i, c in enumerate(E.clips(), 1):
            dst09, dst10 = out / f"{c['id']}_b09.wav", out / f"{c['id']}_b10.wav"
            if dst09.exists() and dst10.exists() and c["id"] in stats:
                continue
            wav = Path(td) / "a.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(E.VIDEOS / f"{c['id']}.mp4"), "-vn", "-ac", "1", "-ar", str(SR), str(wav)], check=True)
            batch = proc(audios=[str(wav)], descriptions=[PROMPT])
            with torch.inference_mode():
                res = model.separate(batch, predict_spans=False, reranking_candidates=1)
            osr = proc.audio_sampling_rate
            target = res.target[0].float().cpu().numpy().reshape(-1)
            residual = res.residual[0].float().cpu().numpy().reshape(-1)
            orig, _ = librosa.load(str(wav), sr=osr, mono=True)
            n = min(len(orig), len(residual), len(target))
            orig, residual, target = orig[:n], residual[:n], target[:n]
            for name, b in (("_b09", 0.9), ("_b10", 1.0)):
                view = b * residual + (1 - b) * orig
                sf.write(str(out / f"{c['id']}{name}.wav"), view, osr)
            # per-second speech loudness (RMS of the speech stem) for the loudness split
            sec = int(osr)
            stats[c["id"]] = {"speech_rms": [float(np.sqrt(np.mean(target[k:k + sec] ** 2)) + 1e-9) for k in range(0, n, sec)],
                              "orig_rms": [float(np.sqrt(np.mean(orig[k:k + sec] ** 2)) + 1e-9) for k in range(0, n, sec)],
                              "sr": osr, "lag_check": float(np.argmax(np.correlate(orig[:osr * 3], residual[:osr * 3], "full")) - (osr * 3 - 1))}
            if i % 20 == 0:
                print(f"[sep] {which}: {i} clips", flush=True)
                stats_f.write_text(json.dumps(stats), encoding="utf-8")
    stats_f.write_text(json.dumps(stats), encoding="utf-8")
    print(f"[sep] {which}: done, {len(stats)} clips -> {out}")


# ----------------------------------------------------------------------------- PSED on the views (env psed)
def psed_views(which: str):
    from src.stage4_audio_event_detection.psed_infer import cache_clips
    E.use_set(which)
    for name, b in BLENDS.items():
        suffix = "_b09" if b == 0.9 else "_b10"
        items = [SEP_ROOT / which / f"{c['id']}{suffix}.wav" for c in E.clips()]
        items = [p for p in items if p.exists()]
        outdir = E.WIN / name
        # cache_clips keys by stem: strip the suffix so the file name matches the clip id
        outdir.mkdir(parents=True, exist_ok=True)
        tmp = E.WIN / (name + "_raw")
        n = cache_clips(items, out_dir=tmp)
        for f in tmp.glob("*.npz"):
            (outdir / f.name.replace(suffix, "")).write_bytes(f.read_bytes())
        print(f"[psed] {which}/{name}: {len(items)} views, {n} scored now -> {outdir}", flush=True)


# ----------------------------------------------------------------------------- analysis (env msproj)
def analyse():
    from sklearn.metrics import roc_auc_score
    from benchmark.fusion_v5 import boxes, LOOSE, gold_events, is_true, load_set, FP_TARGET, GRID, psed_bar
    E.use_set("calib")
    data = load_set("calib", ["psed", "psed_sep09", "psed_sep10"])
    stats = json.loads((SEP_ROOT / "calib" / "speech_rms.json").read_text(encoding="utf-8"))
    print(f"[analyse] {len(data)} calibration clips with all three views")
    rows = []
    for c, d in data:
        fw, t, lab = d["psed"]
        gold = gold_events(c)
        for box in boxes(fw, t, lab, LOOSE):
            lab_i = lab.index(box[0])
            m = (t >= box[1]) & (t <= box[2])
            def best(view):
                fw2, t2, lab2 = d[view]
                m2 = (t2 >= box[1]) & (t2 <= box[2])
                return float(fw2[m2, lab2.index(box[0])].max()) if m2.any() else 0.0
            sp = stats.get(c["id"], {}).get("speech_rms", []); orm = stats.get(c["id"], {}).get("orig_rms", [])
            k0, k1 = int(box[1]), min(int(box[2]) + 1, len(sp))
            loud = float(np.mean(sp[k0:k1]) / (np.mean(orm[k0:k1]) + 1e-9)) if sp and k1 > k0 else 0.0
            rows.append({"true": is_true(box, gold), "len": box[2] - box[1], "p": box[3], "p09": best("psed_sep09"), "p10": best("psed_sep10"), "speech_share": loud})
    y = [int(r["true"]) for r in rows]
    res = {"boxes": len(rows), "true": int(sum(y)), "auroc": {}}
    for k in ("p", "p09", "p10"):
        res["auroc"][k] = roc_auc_score(y, [r[k] for r in rows])
    res["auroc"]["max_p_p09"] = roc_auc_score(y, [max(r["p"], r["p09"]) for r in rows])
    print("[analyse] AUROC true-vs-false boxes: " + "  ".join(f"{k} {v:.3f}" for k, v in res["auroc"].items()))
    for name, sel in (("true", lambda r: r["true"]), ("false", lambda r: not r["true"])):
        d09 = [r["p09"] - r["p"] for r in rows if sel(r)]
        res[f"delta09_{name}"] = {"median": float(np.median(d09)), "share_up": float(np.mean([x > 0 for x in d09]))}
        print(f"[analyse] delta (view 0.9 - original) on {name} boxes: median {np.median(d09):+.3f}, share up {np.mean([x > 0 for x in d09]):.0%}")
    # by speech loudness third
    q = np.quantile([r["speech_share"] for r in rows], [1 / 3, 2 / 3])
    for lo, hi, name in ((0, q[0], "quiet-speech"), (q[0], q[1], "mid"), (q[1], 9e9, "loud-speech")):
        sub = [r for r in rows if lo <= r["speech_share"] < hi and r["true"]]
        if sub:
            print(f"[analyse] {name:12s} true boxes n={len(sub):4d}: median delta {np.median([r['p09'] - r['p'] for r in sub]):+.3f}")
    # the witness rule at the false-alarm target (split-half)
    from benchmark.fusion_v5 import run_variant
    res["witness"] = witness_select(data)
    (_ROOT / "benchmark" / "sep_witness_calib.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("->", _ROOT / "benchmark" / "sep_witness_calib.json")


def witness_run(data, tau, tau_low, view="psed_sep09"):
    """PSED candidates at tau_low on the original; keep if p_orig >= tau or p_view >= tau (same family, same span)"""
    from benchmark.fusion_v5 import boxes, gold_events, is_true
    hits_m = tot_m = hits_all = tot_all = fp = 0; minutes = 0.0
    for c, d in data:
        fw, t, lab = d["psed"]; fw2, t2, lab2 = d[view]
        gold = gold_events(c); minutes += c["duration"] / 60.0
        dets = []
        for box in boxes(fw, t, lab, tau_low):
            if box[3] >= tau:
                dets.append(box); continue
            m2 = (t2 >= box[1]) & (t2 <= box[2])
            if m2.any() and float(fw2[m2, lab2.index(box[0])].max()) >= tau:
                dets.append(box)
        for g in gold:
            m = any(E._same(x[0], g["label"]) and E._overlap_ok(x[1], x[2], g["start"], g["end"]) for x in dets)
            tot_all += 1; hits_all += m
            if g["consequential"] and g["masked"]:
                tot_m += 1; hits_m += m
        fp += sum(1 for x in dets if not is_true(x, c["events"]))
    return hits_m / max(1, tot_m), hits_all / max(1, tot_all), fp / max(1e-6, minutes)


def witness_select(data):
    from benchmark.fusion_v5 import FP_TARGET, GRID, LOOSE
    rng = np.random.default_rng(0); idx = rng.permutation(len(data)); half = len(data) // 2
    A = [data[i] for i in idx[:half]]; B = [data[i] for i in idx[half:]]
    def best(dd):
        for tau in GRID:
            r = witness_run(dd, tau, LOOSE)
            if r[2] <= FP_TARGET:
                return (tau,) + r
        return (GRID[-1],) + witness_run(dd, GRID[-1], LOOSE)
    def ctrl(dd):
        from benchmark.fusion_v5 import run_variant, psed_bar
        for tau in GRID:
            r = run_variant(dd, "psed", "beats", tau, 0.0, psed_bar())
            if r[2] <= FP_TARGET:
                return (tau,) + r
        return (GRID[-1],) + run_variant(dd, "psed", "beats", GRID[-1], 0.0, psed_bar())
    full_w, full_c = best(data), ctrl(data)
    tA = best(A)[0]; rB = witness_run(B, tA, LOOSE); cA = ctrl(A)[0]
    from benchmark.fusion_v5 import run_variant, psed_bar
    cB = run_variant(B, "psed", "beats", cA, 0.0, psed_bar())
    out = {"witness_full": {"tau": full_w[0], "masked": full_w[1], "all": full_w[2], "fp": full_w[3]},
           "control_full": {"tau": full_c[0], "masked": full_c[1], "all": full_c[2], "fp": full_c[3]},
           "split_half": {"witness_B_at_A": {"tau": tA, "masked": rB[0], "all": rB[1], "fp": rB[2]},
                          "control_B_at_A": {"tau": cA, "masked": cB[0], "all": cB[1], "fp": cB[2]}}}
    print(f"[witness] full: witness tau {full_w[0]:.2f} masked {full_w[1]:.1%} all {full_w[2]:.1%} FP {full_w[3]:.2f} | control tau {full_c[0]:.2f} masked {full_c[1]:.1%} all {full_c[2]:.1%} FP {full_c[3]:.2f}")
    print(f"[witness] split A->B: witness masked {rB[0]:.1%} all {rB[1]:.1%} FP {rB[2]:.2f} | control masked {cB[0]:.1%} all {cB[1]:.1%} FP {cB[2]:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--separate", action="store_true")
    ap.add_argument("--psed", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--set", choices=("calib", "sliceB"), default="calib")
    a = ap.parse_args()
    if a.separate: separate(a.set)
    if a.psed: psed_views(a.set)
    if a.analyse: analyse()


if __name__ == "__main__":
    main()
