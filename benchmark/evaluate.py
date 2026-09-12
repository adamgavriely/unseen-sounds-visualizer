"""Benchmark evaluation: score the gate against Adam's hand tags.

Ground truth (benchmark/tags.json): unseen_ambient & mixed = the system SHOULD
augment; seen_ambient & no_ambient = it should NOT. The prediction replicates
the pipeline gate (PANNs + family consolidation + the Stage-2 visibility backend).

Model outputs are cached (benchmark/eval_cache.json) so threshold sweeps and
re-scoring are instant after the first pass.

Usage:
    python -m benchmark.evaluate            # analyze + score + sweep
    python -m benchmark.evaluate --rescore  # score from cache only (no models)
"""
from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from src.labels import is_salient_nonspeech, consolidate_families, min_confidence

BENCH = _ROOT / "data" / "input" / "benchmark"
TAGS = _ROOT / "benchmark" / "tags.json"
# per-backend files so the CLIP-gate and VLM-gate runs can be compared, not overwritten
_SFX = "" if config.VIDEO_BACKEND == "clip" else f"_{config.VIDEO_BACKEND}"
# and per detector: BEATs and PANNs are calibrated differently, so their caches and
# their sweeps must not be mixed
_BEATS = "beats" in str(getattr(config, "AED_MODEL", "")).lower()
if _BEATS:
    _SFX += "_beats"
CACHE = _ROOT / "benchmark" / f"eval_cache{_SFX}.json"
RESULTS = _ROOT / "benchmark" / f"eval_results{_SFX}.json"
EXT = {".webm", ".ogv", ".mp4"}
CATEGORIES = ["unseen_ambient", "mixed", "seen_ambient", "no_ambient"]
SHOULD_AUGMENT = {"unseen_ambient": True, "mixed": True,
                  "seen_ambient": False, "no_ambient": False}


def _find_clip(basename: str) -> Path | None:
    for folder in CATEGORIES + ["unsorted"]:
        p = BENCH / folder / basename
        if p.exists():
            return p
    return None


def _analyze(path: Path) -> dict:
    """Run stages 1+4+2 once; return the cacheable raw signals."""
    from src.stage1_audio_extraction import extract_audio
    from src.stage4_audio_event_detection import detect_events
    from src.stage2_video_understanding import analyze
    from src.labels import canonical
    with tempfile.TemporaryDirectory() as td:
        media = extract_audio(path, Path(td) / "a.wav", config.SAMPLE_RATE)
        events = detect_events(Path(media.wav_path), threshold=config.AED_THRESHOLD,
                               min_dur=config.AED_MIN_DUR)
        # VLM: ask only about sounds actually heard (faster, less hallucination)
        cands = sorted({canonical(e.label) for e in events}) \
            if config.VIDEO_BACKEND == "vlm" else None
        scene = analyze(path, backend=config.VIDEO_BACKEND, num_frames=config.NUM_FRAMES,
                        model=config.VIDEO_MODEL, vlm_model=config.VLM_MODEL, siglip_model=config.SIGLIP_MODEL, owl_model=config.OWL_MODEL,
                        owl_threshold=config.OWL_THRESHOLD,
                        siglip_threshold=config.SIGLIP_THRESHOLD,
                        device=config.DEVICE, threshold=config.VISIBILITY_THRESHOLD,
                        candidates=cands)
    # Cache the RAW per-concept visibility scores as well as the thresholded list.
    # Storing only visible_entities meant the visibility threshold could never be
    # swept -- the one knob on the module responsible for ~76% of the errors.
    return {"duration": media.duration,
            "events": [[e.label, e.start, e.end, e.confidence] for e in events],
            "visible": scene.visible_entities,
            "vis_scores": (scene.raw or {}).get("scores", {}),
            "backend": config.VIDEO_BACKEND}


def _predict(entry: dict, display_threshold: float,
             augment_threshold: float | None = None,
             vis_threshold: float | None = None) -> dict:
    """Replicate the gate from cached signals -> clip-level decision."""
    from src.types import AudioEvent
    aug_thr = config.AUGMENT_THRESHOLD if augment_threshold is None else augment_threshold
    events = [AudioEvent(*e) for e in entry["events"]]
    salient = [e for e in consolidate_families(
                   [x for x in events if is_salient_nonspeech(x.label)])
               if e.confidence >= min_confidence(e.label, display_threshold)]
    scores = entry.get("vis_scores") or {}
    if scores and vis_threshold is not None:
        visible = {k.lower() for k, v in scores.items() if v >= vis_threshold}
    else:
        visible = {v.lower() for v in entry["visible"]}
    # asymmetric: an off-screen (augment) claim needs the higher bar
    offscreen = [e.label for e in salient
                 if e.label.lower() not in visible
                 and e.confidence >= min_confidence(e.label, aug_thr)]
    seen = [e.label for e in salient if e.label.lower() in visible]
    if offscreen and seen:
        cat = "mixed"
    elif offscreen:
        cat = "unseen_ambient"
    elif seen:
        cat = "seen_ambient"
    else:
        cat = "no_ambient"
    return {"category": cat, "augment": bool(offscreen),
            "offscreen": offscreen, "seen": seen}


def _score(tags: dict, cache: dict, display_threshold: float,
           augment_threshold: float | None = None,
           vis_threshold: float | None = None) -> dict:
    tp = fp = tn = fn = 0
    confusion = {t: {p: 0 for p in CATEGORIES} for t in CATEGORIES}
    errors = []
    for key, val in tags.items():
        tag = val.get("tag")
        if tag not in SHOULD_AUGMENT:
            continue
        base = key.split("/", 1)[1]
        if base not in cache:
            continue
        pred = _predict(cache[base], display_threshold, augment_threshold,
                        vis_threshold)
        confusion[tag][pred["category"]] += 1
        gt, decided = SHOULD_AUGMENT[tag], pred["augment"]
        if gt and decided:
            tp += 1
        elif gt and not decided:
            fn += 1
            errors.append({"clip": base, "truth": tag, "pred": pred["category"],
                           "kind": "missed augment"})
        elif not gt and decided:
            fp += 1
            errors.append({"clip": base, "truth": tag, "pred": pred["category"],
                           "kind": "false augment", "offscreen": pred["offscreen"]})
        else:
            tn += 1
    n = tp + fp + tn + fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"display_threshold": display_threshold,
            "augment_threshold": config.AUGMENT_THRESHOLD if augment_threshold is None
            else augment_threshold, "n_clips": n,
            "accuracy": (tp + tn) / n if n else 0.0,
            "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "confusion": confusion, "errors": errors}


def main():
    tags = json.loads(TAGS.read_text(encoding="utf-8"))
    usable = {k: v for k, v in tags.items() if v.get("tag") in SHOULD_AUGMENT}
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

    if "--rescore" not in sys.argv:
        todo = []
        for key in usable:
            base = key.split("/", 1)[1]
            if base not in cache:
                p = _find_clip(base)
                if p:
                    todo.append((base, p))
        print(f"[evaluate] {len(usable)} tagged clips; {len(todo)} to analyze "
              f"({len(cache)} cached)")
        for i, (base, p) in enumerate(todo, 1):
            try:
                cache[base] = _analyze(p)
            except Exception as e:
                print(f"  ! {base}: {type(e).__name__}: {e}")
                continue
            print(f"  [{i}/{len(todo)}] {base}")
            CACHE.write_text(json.dumps(cache), encoding="utf-8")  # save as we go

    # score at the configured thresholds + 2D sweep (display x augment)
    result = _score(usable, cache, config.DISPLAY_THRESHOLD, config.AUGMENT_THRESHOLD)
    # PANNs lives at 0.08-0.30; BEATs' real sounds sat at 0.30+ on the demos and its
    # phantoms at 0.28-, so its grid is shifted up.
    grid_d = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40) if _BEATS else (0.08, 0.10, 0.12, 0.15)
    grid_a = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50) if _BEATS else (0.12, 0.15, 0.20, 0.25, 0.30)
    sweep = [_score(usable, cache, d, a) for d in grid_d for a in grid_a if a >= d]
    RESULTS.write_text(json.dumps({"main": result, "sweep": [
        {k: s[k] for k in ("display_threshold", "augment_threshold",
                           "accuracy", "precision", "recall", "f1")}
        for s in sweep]}, indent=2), encoding="utf-8")

    r = result
    print(f"\n=== gating accuracy @ display_threshold={r['display_threshold']} "
          f"({r['n_clips']} clips) ===")
    print(f"  accuracy {r['accuracy']:.2%}  precision {r['precision']:.2%}  "
          f"recall {r['recall']:.2%}  F1 {r['f1']:.2%}")
    print(f"  tp {r['tp']}  fp {r['fp']}  tn {r['tn']}  fn {r['fn']}")
    print("  confusion (rows=truth, cols=pred):")
    header = "               " + "".join(f"{c[:12]:>14}" for c in CATEGORIES)
    print(header)
    for t in CATEGORIES:
        row = "".join(f"{r['confusion'][t][p]:>14}" for p in CATEGORIES)
        print(f"  {t[:13]:13}{row}")
    print("\n  sweep (display/augment thresholds -> F1):")
    for s in sweep:
        print(f"    d={s['display_threshold']:.2f} a={s['augment_threshold']:.2f} -> "
              f"acc {s['accuracy']:.2%} P {s['precision']:.2%} "
              f"R {s['recall']:.2%} F1 {s['f1']:.2%}")
    print(f"\n  full results (incl. {len(r['errors'])} errors) -> {RESULTS}")


if __name__ == "__main__":
    main()
