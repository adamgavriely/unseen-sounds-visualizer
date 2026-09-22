"""Stage-4 dry run on the per-sound gold (amendment 5, docs/prereg_v4.md): the BLIND system's
shown set (detector -> label filter -> families -> display timeline, no gate, no pictures) for
BEATs and PretrainedSED at display bars 0.25-0.40, scored with the per-sound rules of
benchmark/gold/score_per_sound.py. A diagnostic of what each detector finds and fires, on
DEV (the 54 old judge clips), slice B and everything; the row-level verdict stays the arm rule.

    python benchmark/gold/detector_dry.py --infer            # BEATs frame-wise cache (GPU), once
    python benchmark/gold/detector_dry.py                    # score (needs the PSED cache for the psed rows)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from src.types import SceneContext, AugmentationSpec
from benchmark.gold import score_per_sound as S

GOLD = _ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"
FW = _ROOT / "benchmark" / "gold" / "beats_fw"          # frame-wise BEATs cache, one npz per clip
WAV = _ROOT / "data" / "work" / "gold_wav"
PSED = _ROOT / "data" / "work" / "psed_cache"
OUT = _ROOT / "benchmark" / "gold" / "detector_dry.json"
BARS = (0.25, 0.30, 0.35, 0.40)
JUDGE100 = _ROOT / "benchmark" / "gold" / "judge100.txt"  # the 100 frozen judge clips (stems)


def clip_path(name: str):
    for d in ("mixed", "seen_ambient", "unseen_ambient", "no_ambient", "unsorted", "_dropped"):
        p = _ROOT / "data" / "input" / "benchmark" / d / name
        if p.exists():
            return p
    for p in (_ROOT / "data" / "input" / "audioset_strong").glob(name + ".*"):
        return p
    return None


def gold_clips():
    d = json.loads(GOLD.read_text(encoding="utf-8"))
    return [c["clip"] for c in d["clips"] if isinstance(c, dict) and c.get("done") and not c.get("bad")]


def wav_for(p: Path) -> Path:
    """data/work/gold_wav/<stem>/audio.wav -- the PSED cache is keyed by the wav's parent folder
    name (psed_infer.infer_psed), as in a pipeline work dir"""
    (WAV / p.stem).mkdir(parents=True, exist_ok=True)
    w = WAV / p.stem / "audio.wav"
    if not w.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(p), "-ac", "1", "-ar", str(config.SAMPLE_RATE), str(w)], check=True)
    return w


def infer_all(device="cuda", det="beats"):
    """frame-wise cache for BEATs (v3/v4 detector) or PANNs CNN14 (v1, the "other" detector)"""
    from src.stage4_audio_event_detection.beats_infer import infer_beats
    from src.stage4_audio_event_detection import _infer as infer_panns
    fw_dir = FW if det == "beats" else FW.with_name("panns_fw")
    fw_dir.mkdir(parents=True, exist_ok=True)
    for name in gold_clips():
        p = clip_path(name)
        if p is None:
            print("missing", name); continue
        out = fw_dir / (p.stem + ".npz")
        if out.exists():
            continue
        fw, times, labels = (infer_beats if det == "beats" else infer_panns)(wav_for(p), device)
        np.savez_compressed(out, fw=np.asarray(fw, dtype=np.float32), times=np.asarray(times, dtype=np.float64), labels=np.array(labels))
        print(name, np.asarray(fw).shape, flush=True)


_CAM = {}


def _memo_cam():
    """occlusion_onset is the slow part (BEATs re-run per event); the same event recurs at
    every bar, so its refinement is computed once per (clip, window, class)"""
    from src.stage4_audio_event_detection import beats_infer as B
    if getattr(B, "_memo", False):
        return
    raw = B.occlusion_onset

    def memo(audio, sr, window_start, class_idx, device="cpu", **kw):
        key = (len(audio), round(float(window_start), 3), int(class_idx))
        if key not in _CAM:
            _CAM[key] = raw(audio, sr, window_start, class_idx, device, **kw)
        return _CAM[key]
    B.occlusion_onset = memo; B._memo = True


def events_for(stem: str, det: str, bar: float, wav: Path):
    from src.stage4_audio_event_detection import _extract_events, _refine_onsets_cam
    if det in ("beats", "panns"):
        z = np.load((FW if det == "beats" else FW.with_name("panns_fw")) / f"{stem}.npz")
        fw, times, labels = z["fw"], z["times"], [str(x) for x in z["labels"]]
    else:
        # PSED: its own raw bar is the operating point (0.15 = the AudioSet-Strong calibration of the
        # declared arm); rescale() maps that bar onto the display bar 0.35, so "bar" here is PSED's
        # raw bar and the display side stays at 0.35 (the pipeline's V4["4"] behaviour)
        from src.stage4_audio_event_detection import psed_infer as PI
        PI._BAR = float(bar); config.PSED_BAR = float(bar); config.DISPLAY_THRESHOLD = 0.35
        fw, times, labels = PI.infer_psed(wav, "cpu")
        bar = 0.35
    if det == "corr":
        # two-detector corroboration (Fables C+D, 2026-09-22): a BEATs event is kept only if
        # PretrainedSED fires the same family overlapping it; audio only, applied to both systems
        be = events_for(stem, "beats", bar, wav)
        ps = events_for(stem, "psed", 0.15, wav)     # PSED at its declared calibration
        config.AED_MODEL = "beats"; config.ONSET_CAM = True
        return [e for e in be if any(S.same_family(e.label, f.label) and f.start - 1.0 <= e.end and e.start - 1.0 <= f.end for f in ps)]
    thr = 0.5 * bar                                 # config.AED_THRESHOLD = 0.5 * DISPLAY_THRESHOLD
    ev = _extract_events(fw, times, labels, thr, None, config.AED_MIN_DUR, low=thr * float(getattr(config, "AED_HYSTERESIS", 1.0)))
    if det == "beats" and getattr(config, "ONSET_CAM", True):
        ev = _refine_onsets_cam(wav, ev, labels, "cuda")
    cap = getattr(config, "MAX_SPAN", None)
    if cap:
        for e in ev:
            e.end = min(e.end, e.start + float(cap))
    return ev


def blind_pictures(events, duration: float, bar: float):
    """the blind system's on-screen spans, exactly as the scorer reads a render (no images needed)"""
    from src.stage5_cross_modal_analysis import plan_augmentations
    from src.stage6_visual_augmentation import _display_spans, _assign_rows
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        specs = plan_augmentations(SceneContext(), [], events, threshold=0.5 * bar, gate_enabled=False,
                                   display_threshold=bar, augment_threshold=bar)
    spans = _display_spans(specs, duration, require_image=False)
    placed, _ = _assign_rows(spans)
    return [(lab, float(a), float(b)) for _, lab, a, b, _ in placed]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--detectors", nargs="+", default=["beats", "psed"])
    ap.add_argument("--bars", nargs="+", type=float, default=list(BARS))
    a = ap.parse_args()
    if a.infer:
        for det in a.detectors:
            if det in ("beats", "panns"):
                infer_all(det=det)
        return
    _memo_cam()
    config.use_v4("59")                             # depictable filter, 8-s cap (v4b4)
    gold = S.load_gold([GOLD])
    judge = set(JUDGE100.read_text().split()) if JUDGE100.exists() else set()
    results = {}
    for det in a.detectors:
        if det == "psed":
            config.AED_MODEL = "psed"; config.ONSET_CAM = False
        else:
            config.AED_MODEL = "beats"; config.ONSET_CAM = True
        for bar in a.bars:
            rows = {}
            for name in gold_clips():
                p = clip_path(name)
                if p is None or p.stem not in gold:
                    continue
                if det in ("psed", "corr") and not (PSED / f"{p.stem}.npz").exists():
                    continue
                if det == "panns" and not (FW.with_name("panns_fw") / f"{p.stem}.npz").exists():
                    continue
                wav = wav_for(p)
                import soundfile as sf
                dur = sf.info(str(wav)).duration
                ev = events_for(p.stem, det, bar, wav)
                pics = blind_pictures(ev, dur, 0.35 if det == "psed" else bar)
                rows[p.stem] = S.score_clip(gold[p.stem], pics)
            subsets = {"all": list(rows), "dev54": [s for s in rows if s in judge],
                       "sliceB": [s for s in rows if clip_path(s + ".mp4") is None and not (s.endswith(".webm"))],
                       "bench103": [s for s in rows if not (clip_path(s + ".mp4") is None and not s.endswith(".webm"))]}
            for sub, stems in subsets.items():
                if not stems or sub not in a.subsets:
                    continue
                agg = S.aggregate([rows[s] for s in stems])
                results[f"{det}@{bar}@{sub}"] = agg
                print(f"[{det} bar {bar:.2f} {sub:8s}] clips {agg['clips']:3d} needed {agg['needed']:3d} | P {agg['P']:.2f} R {agg['R']:.2f} F1 {agg['F1']:.2f} "
                      f"| hits {agg['hits']} miss {agg['misses']} visible {agg['visible']} cross {agg['cross']} phantom {agg['phantom']} dup {agg['dup']} | clean {agg['clean_acc']}")
    OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("->", OUT)


if __name__ == "__main__":
    main()
