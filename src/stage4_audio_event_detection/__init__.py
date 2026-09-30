"""Stage 4 - Audio Event Detection.

Detect and classify non-speech (and speech) sounds with time boundaries over the
AudioSet ontology (527 classes), using PANNs CNN14 (DecisionLevelMax) for
frame-level Sound Event Detection. Current strategy (per project decision
2026-08-06): DETECT EVERYTHING with a low threshold and plot it; deciding which
sounds matter is deferred to later gating.

Real implementation. Requires: torch, panns-inference, librosa, matplotlib, and
the CNN14 SED checkpoint at ~/panns_data/ (auto-fetched once, ~327 MB). Degrades
gracefully to [] if any of that is missing so the skeleton still runs.

See docs/project_notes.tex sec:models (PANNs executive summary).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import json

import numpy as np

import config
from src.types import AudioEvent

# Onset provenance (2026-09-24). Three reviewers gave three different accounts of which step moves
# an onset earlier, and none of them reconciles with the cached detector scores end to end: the
# adopted run emitted a Siren anchor of 2.60 s and a Laughter span that neither cache produces.
# Nothing outside the pipeline can recover an intermediate start, so the pipeline records its own:
# one row per (step, label, start, end), reset per clip, dumped to the work dir as onset_trace.json.
TRACE: List[dict] = []


def trace(step: str, events, note: str = "") -> None:
    for e in events:
        TRACE.append({"step": step, "label": getattr(e, "label", ""),
                      "start": round(float(getattr(e, "start", 0.0)), 3),
                      "end": round(float(getattr(e, "end", 0.0)), 3),
                      "conf": round(float(getattr(e, "confidence", 0.0)), 3), "note": note})


def trace_spans(step: str, rows, note: str = "") -> None:
    """Same log, for stages that carry (label, start, end) rather than an AudioEvent."""
    for label, a, b in rows:
        TRACE.append({"step": step, "label": label, "start": round(float(a), 3),
                      "end": round(float(b), 3), "conf": 0.0, "note": note})


_SED = None          # lazy-loaded model singleton (avoid reloading per call)
_PANNS_SR = 32000    # PANNs operates at 32 kHz
_PANNS_FPS = 100.0   # framewise output rate (hop 320 @ 32 kHz)


def _get_sed(device: str = "cpu"):
    global _SED
    if _SED is None:
        from panns_inference import SoundEventDetection
        _SED = SoundEventDetection(checkpoint_path=None, device=device)
    return _SED


def _infer(wav_path: Path, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray, list]:
    """Return (framewise[frames,527], times[frames], labels)."""
    import librosa
    from panns_inference.config import labels
    audio, _ = librosa.load(str(wav_path), sr=_PANNS_SR, mono=True)
    framewise = _get_sed(device).inference(audio[None, :])[0]   # (frames, 527)
    times = np.arange(framewise.shape[0]) / _PANNS_FPS
    return framewise, times, list(labels)


def _extract_events(framewise, times, labels, threshold, top_k, min_dur,
                    low: float = None) -> List[AudioEvent]:
    """Turn framewise probabilities into contiguous (label, start, end) spans.

    Double threshold (hysteresis), the standard SED post-processing: a span counts only
    if it reaches `threshold` somewhere, but it extends through any contiguous stretch
    above `low`. A helicopter approaching scored 0.15-0.42 for 1.5 s before crossing
    0.35, and without this the picture came 1.5 s after the ear heard it.
    """
    peaks = framewise.max(axis=0)
    classes = np.where(peaks >= threshold)[0]
    if top_k:
        classes = classes[np.argsort(peaks[classes])[::-1][:top_k]]
    dt = (times[1] - times[0]) if len(times) > 1 else 1.0 / _PANNS_FPS
    low = threshold if low is None else min(low, threshold)
    events: List[AudioEvent] = []
    for c in classes:
        active = framewise[:, c] >= low
        strong = framewise[:, c] >= threshold
        i, n = 0, len(active)
        while i < n:
            if not active[i]:
                i += 1
                continue
            j = i
            while j < n and active[j]:
                j += 1
            if not strong[i:j].any():
                i = j
                continue
            start, end = float(times[i]), float(times[j - 1] + dt)
            # Release (2026-09-24, docs/onset_timing.md): the span ends where the score falls
            # below `low`, but a sustained sound dips below it and keeps going -- measured on the
            # rendered panel the picture leaves 2.30 s before the sound stops. With AED_RELEASE
            # set, the end extends through any following stretch above that absolute score,
            # tolerating gaps shorter than AED_RELEASE_GAP.
            rel = getattr(config, "AED_RELEASE", None)
            if rel:
                gap_frames = int(round(float(getattr(config, "AED_RELEASE_GAP", 1.0)) / dt))
                k, quiet = j, 0
                while k < n and quiet <= gap_frames:
                    quiet = 0 if framewise[k, c] >= float(rel) else quiet + 1
                    k += 1
                k -= quiet                       # step back off the trailing quiet stretch
                if k > j:
                    end = float(times[min(k, n - 1)] + dt)
            if end - start >= min_dur:
                events.append(AudioEvent(label=labels[c], start=start, end=end,
                                         confidence=float(framewise[i:j, c].max())))
            i = j
    events.sort(key=lambda e: (e.start, -e.confidence))
    return events


def plot_timeline(framewise, times, labels, out_png: Path,
                  top_k: int = 15, threshold: float = 0.2) -> None:
    """Save a piano-roll heatmap of the top-K detected sound classes over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    peaks = framewise.max(axis=0)
    order = np.argsort(peaks)[::-1][:top_k]
    order = order[peaks[order] > 0.01]
    if len(order) == 0:
        return
    M = framewise[:, order].T                      # (K, frames)
    names = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, max(3, 0.42 * len(order))))
    im = ax.imshow(M, aspect="auto", origin="upper", vmin=0, vmax=1, cmap="magma",
                   extent=[float(times[0]), float(times[-1]), len(order) - 0.5, -0.5])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("time (s)")
    ax.set_title(f"Detected sound events over time (PANNs SED, threshold={threshold})")
    fig.colorbar(im, ax=ax, label="probability", shrink=0.85)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def _refine_onsets_cam(wav_path: Path, events, labels, device: str, skip_ids=None):
    """Move each BEATs event's start to where silencing the window's opening starts to
    cost the class its evidence (beats_infer.occlusion_onset). The sliding-window stamp
    is within about a second; this is within a cut (80 ms) for abrupt sounds and marks
    the 10%-of-evidence point for sounds that fade in. The end is left alone: a picture
    lingering a little is cheap, a picture arriving late or early is what the viewer
    notices.
    """
    import librosa
    from src.stage4_audio_event_detection import beats_infer as B
    audio, _ = librosa.load(str(wav_path), sr=B.SR, mono=True)
    idx = {l: i for i, l in enumerate(labels)}
    mono = bool(getattr(config, "ONSET_MONOTONE", False))
    skip_ids = skip_ids or set()
    out = []
    for e in events:
        c = idx.get(e.label)
        if c is None:
            out.append(e); continue
        # A span FlexSED raised is already frame-level (25 fps). Treating its start as a BEATs
        # sliding-window stamp and searching [start-1.5, start+0.5] is what produced the Siren at
        # 1.10 (= 2.60 - 1.50) and the Applause at 0.00 (= max(0, 1.12 - 1.50)).
        if mono and id(e) in skip_ids:
            out.append(e); continue
        # the first window that fired ends STAMP_OFFSET after the stamp
        w0 = e.start + B.STAMP_OFFSET - B.WINDOW
        try:
            t, ramp = B.occlusion_onset(audio, B.SR, w0, c, device)
        except Exception as ex:
            print(f"       [stage4] onset refinement failed for {e.label}: {ex}")
            t = None
        if mono and t is not None:
            t = min(max(float(t), e.start), e.end)      # sharpen inside the anchor, never before it
        if t is not None and t < e.end:
            e = AudioEvent(e.label, max(0.0, float(t)), e.end, e.confidence)
        out.append(e)
    return out


def detect_events(wav_path: Path, threshold: float = 0.2, top_k: int = None,
                  min_dur: float = 0.2, device: str = "cpu", model: str = "",
                  plot_path: Path = None, plot_top_k: int = 15) -> List[AudioEvent]:
    # "beats" (default, see config.AED_MODEL) or anything else for PANNs. Same 527
    # labels either way, so nothing downstream cares which one ran.
    # "flam" (v4, docs/prereg_v4.md): frame-level language-audio model over a fixed
    # descriptive vocabulary, one query per AudioSet label, per-query calibration.
    m = (model or "").lower()
    backend = "PSED" if "psed" in m else "FLAM" if "flam" in m else "BEATs" if "beats" in m else "PANNs"
    try:
        if backend == "PSED":
            # v4: PretrainedSED BEATs-strong, frame-level (docs/prereg_psed.md)
            from src.stage4_audio_event_detection.psed_infer import infer_psed
            framewise, times, labels = infer_psed(Path(wav_path), device)
        elif backend == "FLAM":
            from src.stage4_audio_event_detection.flam_infer import infer_flam
            framewise, times, labels = infer_flam(Path(wav_path), device)
        elif backend == "BEATs":
            from src.stage4_audio_event_detection.beats_infer import infer_beats
            framewise, times, labels = infer_beats(Path(wav_path), device)
        else:
            framewise, times, labels = _infer(Path(wav_path), device)
    except Exception as e:  # missing package/checkpoint -> keep the pipeline runnable
        print(f"       [stage4] {backend} unavailable ({type(e).__name__}: {e}); "
              f"returning no events.")
        return []

    low = threshold * float(getattr(config, "AED_HYSTERESIS", 1.0))
    events = _extract_events(framewise, times, labels, threshold, top_k, min_dur, low=low)
    TRACE.clear()
    trace("extract", events, backend)
    flex_ids = set()          # spans FlexSED raised alone: frame-level already, never re-anchored
    ffw = ftimes = flabels = None
    # amendment 8 (2026-09-22): the open-vocabulary second detector, added as a UNION with its own
    # bar -- BEATs is deaf to sounds that speech or music masks, FlexSED is asked one label at a
    # time and hears them (docs/GOLD_RERUN_2026-09-22.md sec 10). Both bars are set on Adam's DEV
    # half; a clip with no cache falls back to BEATs alone.
    fbar = float(getattr(config, "FLEXSED_BAR", 0) or 0)
    if fbar > 0:
        from src.stage4_audio_event_detection import flexsed_infer as FX
        try:
            ffw, ftimes, flabels = FX.infer_flexsed(Path(wav_path), device)
            if getattr(config, "FLEXSED_EXTRA", False):
                ffw, ftimes, flabels = add_flexsed_extra(ffw, ftimes, flabels, Path(wav_path).parent.name)
            config._CURRENT_CLIP = Path(wav_path).parent.name       # round 18 N2b: the clip id for per-clip caches
            events, flex_ids, ffw = fuse_flexsed(events, framewise, times, labels, ffw, ftimes, flabels, min_dur,
                                                 backend=backend, panns=lambda: _infer(Path(wav_path), device),
                                                 listener=_pipeline_listener(wav_path),
                                                 listener_p1=_pipeline_listener_p1(wav_path),
                                                 extra=(load_extra_evidence(Path(wav_path).parent.name)
                                                        if getattr(config, "TIER_SPECIFIC", False) else None))
            from src.labels import canonical as _cn
            _dr = dasm_rescue_events(Path(wav_path).parent.name, {_cn(e.label) for e in events})   # round 19 DR / 20 DR2
            events = events + _dr
            flex_ids |= {id(e) for e in _dr}
        except FileNotFoundError as e:
            print(f"       [stage4] FlexSED cache missing ({e}); BEATs alone", flush=True)
    trace("veto", events, "after the cross-detector and PANNs vetoes")
    if backend == "BEATs" and getattr(config, "ONSET_CAM", True):
        events = _refine_onsets_cam(Path(wav_path), events, labels, device,
                                    skip_ids=flex_ids)
        trace("refine", events, "after occlusion onset refinement")
    cap = getattr(config, "MAX_SPAN", None)      # v4ab3/v4b3: a picture never stays longer than this (docs/prereg_v4.md)
    if cap:
        for e in events:
            e.end = min(e.end, e.start + float(cap))
        trace("cap", events, f"MAX_SPAN {cap}")
    if getattr(config, "ONSET_RELOC", False):              # round 14 amendment I1
        events, _log = relocate_onsets(events, framewise, times, labels, ffw, ftimes, flabels,
                                       extra=load_extra_evidence(Path(wav_path).parent.name))
    if getattr(config, "LISTENER_RESCUE", False):          # round 14: precision filters on the rescued spans
        events, _dropped = filter_rescued(events, ffw, ftimes, flabels, dasm=_pipeline_dasm(wav_path))
    if getattr(config, "CO_ONSET_ARB", False) or getattr(config, "RELABEL_2L", False):   # round 17 R3 / R1
        events, _r17 = post_rules(events, ffw, ftimes, flabels, Path(wav_path).parent.name,
                                  origin={id(e): ("flex" if id(e) in flex_ids else "tagger") for e in events},
                                  listener_p1=_pipeline_listener_p1(wav_path))
    if getattr(config, "RETRIGGER", None):
        attach_breaks(events, framewise, times, labels, ffw, ftimes, flabels)
    n_classes = len({e.label for e in events})
    print(f"       [stage4] {backend} SED: {len(events)} event span(s) over "
          f"{n_classes} class(es) (threshold={threshold}).")
    if plot_path is not None:
        plot_timeline(framewise, times, labels, Path(plot_path),
                      top_k=plot_top_k, threshold=threshold)
        print(f"       [stage4] timeline plot -> {plot_path}")
    return events


def _require_caches(wav_path):
    """LISTENER_REQUIRE_CACHES (set by use_shipped when the listener rescue ships): a clip with no listener answers or no
    DASM scores must not quietly fall back to the plain detector -- stop instead. The answers come from slurm/run_best.sh."""
    if not getattr(config, "LISTENER_REQUIRE_CACHES", False):
        return
    clip = Path(wav_path).parent.name
    miss = []
    for k in ("LISTENER_CACHE", "LISTENER_VCACHE", "LISTENER_AFCACHE"):
        p = getattr(config, k, None)
        if not p or not all(Path(x).exists() for x in str(p).split(";")):
            miss.append(f"{k} (no file)")
        elif not any(x.get("clip") == clip for x in _cache_items(p)):
            miss.append(f"{k} (no items for {clip})")
    d = getattr(config, "LISTENER_DASM_DIR", None)
    if getattr(config, "LISTENER_DASM_VOTE", False) and not (d and (Path(d) / f"{clip}.npz").exists()):
        miss.append(f"LISTENER_DASM_DIR (no {clip}.npz)")
    pv = getattr(config, "RELABEL_P1V4", None)
    if getattr(config, "KEEP_NEEDS_V4", False) and not (pv and Path(pv).exists()):
        miss.append("RELABEL_P1V4 (no file)")
    p4 = getattr(config, "DASM_P4_CACHE", None)
    if getattr(config, "DASM_RESCUE", False) and not (p4 and Path(p4).exists()):
        miss.append("DASM_P4_CACHE (no file)")
    if miss:
        raise RuntimeError(f"[stage4] {clip}: the shipped listener rescue needs precomputed answers "
                           f"(run slurm/run_best.sh on this clip first): missing {miss}")


def _pipeline_listener(wav_path):
    """R13-3 in the pipeline: the listener cache (config.LISTENER_CACHE) for this clip (its work dir name = the clip id)"""
    if not getattr(config, "LISTENER_RESCUE", False):
        return None
    _require_caches(wav_path)
    if getattr(config, "LISTENER_RULE", None):
        vpath = getattr(config, "LISTENER_VCACHE", None)
        return listener_from_vcache(vpath, Path(wav_path).parent.name) if vpath else None
    path = getattr(config, "LISTENER_CACHE", None)
    return listener_from_cache(path, Path(wav_path).parent.name) if path else None


def listener_p1_lookup(vcache, cache, clip: str, tol: float = 0.02):
    """Round 14 F7: does the listener accept this BEATs span's family? The P1 item of the span in the variants cache
    (rule V4 if it has one, else V12), else the yes/no score > 3 of its P1 item in the R13-3 cache. P1 starts are the
    refined starts, so an item matches on family and end, with its start inside [start - tol, end]; the nearest start
    wins. Returns (accepted, how) with how in {"V4", "V12", "yesno", "missing"}."""
    from src.labels import canonical

    def items(path, pool="P1"):
        if not path:
            return []
        return [x for x in _cache_items(path) if x.get("clip") == clip and x.get("pool") == pool]
    V, Y = items(vcache), items(cache)

    def find(its, label, start, end):
        fam = canonical(label)
        c = [x for x in its if x["family"] == fam and abs(x["end"] - end) <= tol and start - tol <= x["start"] <= end]
        if not c:
            return None
        same = [x for x in c if x.get("label") == label] or c
        return min(same, key=lambda x: abs(x["start"] - start))

    def look(label, start, end):
        it = find(V, label, start, end)
        acc = (it or {}).get("accept") or {}
        for r in ("V4", "V12"):
            if r in acc:
                return bool(acc[r]), r
        it = find(Y, label, start, end)
        if it is not None and it.get("score") is not None:
            return float(it["score"]) > 3.0, "yesno"
        return False, "missing"
    return look


def _pipeline_listener_p1(wav_path):
    if not getattr(config, "LISTENER_CONFIRMED_MIRROR", False):
        return None
    return listener_p1_lookup(getattr(config, "LISTENER_VCACHE", None), getattr(config, "LISTENER_CACHE", None),
                              Path(wav_path).parent.name)


def _pipeline_dasm(wav_path):
    """Round 14 F8: DASM frame scores (fw [T, Q], times, labels) of this clip from config.LISTENER_DASM_DIR/<clip>.npz"""
    d = getattr(config, "LISTENER_DASM_DIR", None)
    if not (getattr(config, "LISTENER_DASM_VOTE", False) and d):
        return None
    f = Path(d) / (Path(wav_path).parent.name + ".npz")
    if not f.exists():
        return None
    z = np.load(f)
    return z["fw"].astype(np.float32), z["times"].astype(np.float64), [str(x) for x in z["labels"]]


def add_flexsed_extra(ffw, ftimes, flabels, clip: str):
    """Round 14 amendment D (FLEXSED_EXTRA): append the group-a ("a_folded") extra FlexSED queries of
    config.FLEXSED_EXTRA_QUERIES as more columns, read from config.FLEXSED_EXTRA_DIR/<clip>.npz (fw [Q, frames], labels,
    fps). Label = the query's AudioSet label; every later step treats them like the 215. The frame rate must equal the
    main cache's; a frame-count difference of a few frames is trimmed to the shorter one. No file = unchanged."""
    d = getattr(config, "FLEXSED_EXTRA_DIR", None)
    qf = getattr(config, "FLEXSED_EXTRA_QUERIES", None)
    f = Path(d) / f"{clip}.npz" if d else None
    if f is None or not f.exists() or not qf:
        print(f"       [stage4] FlexSED extra queries: no cache for {clip}; main queries only", flush=True)
        return ffw, ftimes, flabels
    want = [q["query"] for q in json.loads(Path(qf).read_text(encoding="utf-8"))["queries"]
            if q.get("bucket") == "a_folded"]
    z = np.load(f)
    efw = z["fw"].astype(np.float32).T                     # [frames, Q]
    elab = [str(x) for x in z["labels"]]
    fps = float(z["fps"])
    ft = np.asarray(ftimes, float)
    main_fps = 1.0 / float(ft[1] - ft[0]) if len(ft) > 1 else fps
    assert abs(fps - main_fps) < 1e-3, (clip, fps, main_fps)
    cols = [elab.index(q) for q in want if q in elab and q not in flabels]
    n = min(len(ft), efw.shape[0])
    if abs(len(ft) - efw.shape[0]) > 2:
        print(f"       [stage4] FlexSED extra: frame counts differ ({len(ft)} vs {efw.shape[0]}) for {clip}", flush=True)
    out = np.concatenate([np.asarray(ffw)[:n], efw[:n][:, cols]], axis=1)
    return out, ft[:n], list(flabels) + [elab[i] for i in cols]


def load_extra_evidence(clip: str):
    """amendment I: the specific (group-a, folded) FlexSED queries of this clip as EVIDENCE only (no spans): (fw [T, Q],
    times, labels) from config.FLEXSED_EXTRA_DIR / FLEXSED_EXTRA_QUERIES, or None if absent"""
    d, qf = getattr(config, "FLEXSED_EXTRA_DIR", None), getattr(config, "FLEXSED_EXTRA_QUERIES", None)
    f = Path(d) / f"{clip}.npz" if d else None
    if f is None or not f.exists() or not qf:
        return None
    want = [q["query"] for q in json.loads(Path(qf).read_text(encoding="utf-8"))["queries"] if q.get("bucket") == "a_folded"]
    z = np.load(f)
    efw = z["fw"].astype(np.float32).T
    elab = [str(x) for x in z["labels"]]
    cols = [elab.index(q) for q in want if q in elab]
    return efw[:, cols], np.arange(efw.shape[0], dtype=np.float64) / float(z["fps"]), [elab[i] for i in cols]


def _family_peak(fam, a, b, fw_, ft_, flabels, extra=None):
    """max FlexSED score in [a, b) over the family's own queries and its specific child queries (amendment I5)"""
    from src.labels import canonical
    best = 0.0
    for fw, ft, fl in [(fw_, ft_, flabels)] + ([extra] if extra is not None else []):
        fw, ft = np.asarray(fw), np.asarray(ft)
        cols = [i for i, l in enumerate(fl) if canonical(l) == fam]
        m = (ft >= a) & (ft < b)
        if cols and m.any():
            best = max(best, float(fw[m][:, cols].max()))
    return best


def _family_evidence(fam, grid, framewise, times, labels, ffw, ftimes, flabels, extra=None):
    """amendment I1: the family's evidence on the FlexSED grid = max of its BEATs columns (a window's score held over
    [t, t + hop)) and its FlexSED family / specific queries"""
    from src.labels import canonical
    ev = np.zeros(len(grid), np.float32)
    tt = np.asarray(times, float)
    tc = [i for i, l in enumerate(labels) if canonical(l) == fam]
    if tc and len(tt):
        k = np.clip(np.searchsorted(tt, grid, side="right") - 1, 0, len(tt) - 1)
        hop = (tt[1] - tt[0]) if len(tt) > 1 else 0.25
        ok = (grid >= tt[0]) & (grid < tt[-1] + hop)
        ev = np.maximum(ev, np.where(ok, np.asarray(framewise)[:, tc].max(axis=1)[k], 0.0))
    for fw, ft, fl in [(ffw, ftimes, flabels)] + ([extra] if extra is not None else []):
        cols = [i for i, l in enumerate(fl) if canonical(l) == fam]
        if not cols:
            continue
        fw, ft = np.asarray(fw), np.asarray(ft)
        n = min(len(ft), len(grid))
        ev[:n] = np.maximum(ev[:n], fw[:n][:, cols].max(axis=1))
    return ev


def relocate_onsets(events, framewise, times, labels, ffw, ftimes, flabels, extra=None):
    """Round 14 amendment I1 (ONSET_RELOC): inside each span, its START moves to the frame of the steepest rise of the family's
    evidence within the span's first 3 s, searched after the last dip below half the span's evidence peak (no dip -> from
    the start); a later rise after a dip >= 1.5 s below half the peak splits the span in two. Only later, never earlier;
    no new sound. Returns the new event list and a log."""
    from src.labels import canonical
    grid = np.asarray(ftimes, float)
    if len(grid) < 2:
        return events, []
    dt = float(grid[1] - grid[0])
    memo, out, log = {}, [], []
    for e in events:
        fam = canonical(e.label)
        if fam not in memo:
            memo[fam] = _family_evidence(fam, grid, framewise, times, labels, ffw, ftimes, flabels, extra)
        ev = memo[fam]
        idx = np.where((grid >= e.start - 1e-6) & (grid < e.end))[0]
        if len(idx) < 3 or float(ev[idx].max()) <= 0:
            out.append(e); continue
        half = 0.5 * float(ev[idx].max())
        pieces, cur, low = [], [idx[0]], 0
        for k in idx[1:]:                                   # split at dips >= 1.5 s below half the peak
            if ev[k] < half:
                low += 1
            else:
                if low * dt >= 1.5 and len(cur) > low:
                    pieces.append(cur[:len(cur) - low]); cur = []
                low = 0
            cur.append(k)
        pieces.append(cur)
        segs = []
        for pi, p in enumerate(pieces):
            p = np.asarray(p)
            w = p[grid[p] < grid[p[0]] + 3.0]
            dips = [k for k in w if ev[k] < half]
            s0 = dips[-1] if dips else w[0]
            cand = w[w >= s0]
            if len(cand) >= 2:
                d = np.diff(ev[cand])
                j = int(cand[int(np.argmax(d)) + 1]) if float(d.max()) > 0 else int(cand[0])
            else:
                j = int(cand[0])
            st = max(float(e.start), float(grid[j]))
            en = float(e.end) if pi == len(pieces) - 1 else float(grid[p[-1]] + dt)
            segs.append((st, en))
        new = []
        for k, (st, en) in enumerate(segs):
            if k < len(segs) - 1:
                nxt = segs[k + 1][0]
                en = min(en, nxt)
            if en - st <= dt:
                continue
            ne = AudioEvent(e.label, st, en, e.confidence, rescued=getattr(e, "rescued", False),
                            arbiter=getattr(e, "arbiter", False), breaks=list(getattr(e, "breaks", []) or []),
                            agree=getattr(e, "agree", False))
            new.append(ne)
        if not new:
            out.append(e); continue
        if len(new) > 1 or abs(new[0].start - e.start) > 1e-6:
            log.append([e.label, round(e.start, 2), round(e.end, 2), [[round(x.start, 2), round(x.end, 2)] for x in new]])
        out += new
    return out, log


def _impulse_cols(flabels) -> list:
    """R13-5: the FlexSED queries of the impulsive families (config.IMPULSE_FAMILIES, matched on the query name)"""
    fams = set(getattr(config, "IMPULSE_FAMILIES", ()) or ())
    return [i for i, lab in enumerate(flabels) if lab in fams]


def _mirror_veto(events, keep_ids, ffw, ftimes, flabels, bar: float, own_max: float):
    """R13-2 (round 13, docs/prereg_round13_detector_push.md): a BEATs-only span (no same-family FlexSED twin) is dropped
    when FlexSED's top query in the span's time window is a DIFFERENT (canonical) family scoring >= `bar` while the
    span's own family scores < `own_max` there. A family FlexSED has no query for is never touched (as the veto above)."""
    from src.labels import canonical
    fams = [canonical(l) for l in flabels]
    ft = np.asarray(ftimes)
    out, dropped = [], []
    for e in events:
        if id(e) in keep_ids:
            out.append(e); continue
        own = [i for i, f in enumerate(fams) if f == canonical(e.label)]
        if not own:
            out.append(e); continue
        m = (ft >= e.start) & (ft < e.end)
        if not m.any():
            m = np.zeros(len(ft), bool); m[int(np.argmin(np.abs(ft - 0.5 * (e.start + e.end))))] = True
        pk = ffw[m].max(axis=0)
        top = int(np.argmax(pk))
        if fams[top] != canonical(e.label) and float(pk[top]) >= bar and float(pk[own].max()) < own_max:
            dropped.append(e)
            continue
        out.append(e)
    return out, dropped


def family_absences(fam: str, framewise, times, labels, ffw, ftimes, flabels, gap: float, flex_low: float,
                    tag_low: float) -> list:
    """R13-6: the interior stretches >= `gap` s where the family has no evidence at all -- every FlexSED query of the
    family < flex_low AND every tagger column of the family < tag_low (a tagger window's score held over [t, t + hop)) --
    and that the family's evidence then returns (a new onset). Grid: FlexSED frames if present, else the tagger's."""
    from src.labels import canonical
    tt = np.asarray(times, float)
    tcols = [i for i, l in enumerate(labels) if canonical(l) == fam]
    if ffw is not None and len(ftimes) > 1:
        grid = np.asarray(ftimes, float)
        fcols = [i for i, l in enumerate(flabels) if canonical(l) == fam]
    else:
        grid, fcols = tt, []
    on = np.zeros(len(grid), bool)
    if fcols:
        on |= np.asarray(ffw)[:, fcols].max(axis=1) >= flex_low
    if tcols and len(tt):
        k = np.clip(np.searchsorted(tt, grid, side="right") - 1, 0, len(tt) - 1)
        hop = (tt[1] - tt[0]) if len(tt) > 1 else 0.25
        ok = (grid >= tt[0]) & (grid < tt[-1] + hop)
        on |= (np.asarray(framewise)[:, tcols].max(axis=1)[k] >= tag_low) & ok
    dt = (grid[1] - grid[0]) if len(grid) > 1 else 0.04
    out, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            i += 1; continue
        j = i
        while j < n and not on[j]:
            j += 1
        if i > 0 and j < n and (grid[j] - grid[i]) >= gap - 1e-6:       # evidence before and after: interior
            out.append((round(float(grid[i]), 3), round(float(grid[j]), 3)))
        i = j
    return out


def attach_breaks(events, framewise, times, labels, ffw, ftimes, flabels) -> None:
    """R13-6 (config.RETRIGGER = (gap_s, flexsed_low, tagger_low)): record on every event its family's evidence
    absences, so no later merge (families, dedup, display) joins two appearances across one."""
    from src.labels import canonical
    gap, fl, tl = (float(x) for x in config.RETRIGGER)
    memo = {}
    for e in events:
        fam = canonical(e.label)
        if fam not in memo:
            memo[fam] = family_absences(fam, framewise, times, labels, ffw, ftimes, flabels, gap, fl, tl)
        e.breaks = list(memo[fam])


def fuse_flexsed(events, framewise, times, labels, ffw, ftimes, flabels, min_dur, backend: str = "BEATs",
                 panns=None, listener=None, listener_p1=None, extra=None):
    """The union of the tagger's spans with FlexSED's, and the vetoes (amendments 8, 10, 11, 16, 22; round 4; round 13).

    Pure given its inputs, so a benchmark can run the exact pipeline rule on cached scores. `events` are the tagger's
    extracted spans; (ffw, ftimes, flabels) are FlexSED's frame scores; `panns` is a callable returning PANNs
    (framewise, times, labels) for the clip, called at most once. Returns (events, flex_ids, ffw), where flex_ids are the
    ids of the spans FlexSED raised alone and ffw the (possibly per-family rescaled) FlexSED scores. `listener` (R13-3,
    config.LISTENER_RESCUE) is a lookup(label, start, end, contain=False) -> the cached listener item or None."""
    from src.labels import canonical
    fbar = float(getattr(config, "FLEXSED_BAR", 0) or 0)
    flex_ids = set()
    _pc = {}
    _reset_listener_stats()
    listen_on = bool(getattr(config, "LISTENER_RESCUE", False)) and listener is not None
    lth = float(getattr(config, "LISTENER_TH", 0.0))

    def _panns():
        if "v" not in _pc:
            _pc["v"] = panns() if panns is not None else None
            if _pc["v"] is None:
                raise RuntimeError("no PANNs provider")
        return _pc["v"]

    # Amendment 11: the score scale is family-dependent, so one bar cannot serve every
    # family -- at the 78 needed gold sounds FlexSED knows, its score at the sound has
    # median 0.75 and a bar of 0.8 refuses 59% of them. Each family's own bar was fitted on
    # the 280-clip AudioSet-Strong calibration set (zero id overlap with the gold), and is
    # applied here by rescaling that family's column so its own bar lands exactly on fbar.
    # Everything downstream -- the hysteresis, the span extractor, the veto -- is unchanged,
    # and the veto therefore scales with the bar at the ratio frozen in the prereg.
    from src.labels import canonical
    fambars = getattr(config, "FLEXSED_FAMILY_BARS", None)
    if fambars:
        bars = json.loads(Path(fambars).read_text(encoding="utf-8"))["bars"]
        ffw = ffw.copy()
        n = 0
        for i, lab in enumerate(flabels):
            b = bars.get(canonical(lab))
            if b and b > 0 and abs(b - fbar) > 1e-9:
                # clipped at 1.0: the rescaling exists to move the THRESHOLD, and the
                # peak value survives as spec.confidence, which ranks sounds for the panel's
                # limited rows. A family scaled by 0.8/0.15 would otherwise reach 5.3 and
                # outrank every unscaled sound for a slot.
                ffw[:, i] = np.minimum(ffw[:, i] * (fbar / float(b)), 1.0)
                n += 1
        print(f"       [stage4] per-family bars applied to {n} of {len(flabels)} queries", flush=True)
    fev = _extract_events(ffw, ftimes, flabels, fbar, None, min_dur,
                          low=fbar * float(getattr(config, "AED_HYSTERESIS", 1.0)))
    # Round 13, R13-5 (docs/prereg_round13_detector_push.md): an impulsive sound (gunshot, gasp, hammer, explosion,
    # knock) is short by physics, so the 0.5-s minimum span cuts it; for those FlexSED queries the minimum is
    # IMPULSE_MIN_SPAN instead (same bar, same vetoes). None = off.
    imp = getattr(config, "IMPULSE_MIN_SPAN", None)
    if imp:
        cols = _impulse_cols(flabels)
        if cols:
            names = {flabels[i] for i in cols}
            short = _extract_events(ffw[:, cols], ftimes, [flabels[i] for i in cols], fbar, None,
                                    float(imp) - 1e-6, low=fbar * float(getattr(config, "AED_HYSTERESIS", 1.0)))
            fev = [e for e in fev if e.label not in names] + short
            fev.sort(key=lambda e: (e.start, -e.confidence))
    # a family both detectors report at the same moment keeps the earlier onset
    def key(e):
        return canonical(e.label)
    twinned = set()          # tagger spans that absorbed a same-family FlexSED span
    twin_max = bool(getattr(config, "TWIN_MAX", False))
    disp_bar = float(getattr(config, "DISPLAY_THRESHOLD", 0.35))
    fresh = []
    for e in fev:
        twin = [b for b in events if key(b) == key(e) and b.start - 1.0 <= e.end and e.start - 1.0 <= b.end]
        # Amendment 22, cell F (bug found 2026-09-27): a BEATs twin too weak to be shown (peak below the display
        # bar) used to absorb a FlexSED span above FlexSED's own bar, and the merged sound was then dropped at the
        # display bar -- on DEV 6 strong FlexSED detections vanished this way. With UNION_WEAK_TWIN = "ignore" only a
        # displayable BEATs twin absorbs; otherwise the FlexSED span stays FlexSED-only (and faces the PANNs veto).
        if str(getattr(config, "UNION_WEAK_TWIN", "absorb")) == "ignore":
            twin = [b for b in twin if b.confidence >= float(getattr(config, "DISPLAY_THRESHOLD", 0.35))]
        if twin:
            # UNION_START (2026-09-24, docs/NIGHT_REPORT_2026-09-24.md): the trace shows this
            # step moving 77 starts earlier on DEV by a median of 4 s, the worst by 14 s -- a long
            # FlexSED span pulling a BEATs start to near the clip's beginning. "min" is as shipped;
            # "bounded" lets a twin pull the start at most as far as the twin tolerance itself
            # (1 s), since a start 4-14 s earlier is by definition a different moment; "beats"
            # keeps BEATs' own start and lets FlexSED only confirm.
            rule = str(getattr(config, "UNION_START", "min"))
            for b in twin:
                twinned.add(id(b))
                # Round 13, R13-1: the merged span keeps the stronger side's evidence, each side normalised by its own
                # bar and expressed on the tagger's display scale -- displayable if BEATs >= the display bar OR FlexSED
                # >= its own bar (the Cricket that BEATs gave 0.31 and FlexSED 0.92). Off = BEATs' conf, as before.
                if twin_max and fbar > 0:
                    b.confidence = min(1.0, disp_bar * max(b.confidence / disp_bar, e.confidence / fbar))
                if rule == "beats":
                    continue
                if rule == "bounded":
                    b.start = max(min(b.start, e.start), b.start - 1.0)
                else:
                    b.start = min(b.start, e.start)
        else:
            fresh.append(e)
    # Detector round 2026-09-27 (docs/prereg_v4.md amendment 22): at a LOWER FlexSED bar, a span FlexSED
    # raised alone is admitted only if another detector rises for the same family near it in time --
    # BEATs >= b or PANNs >= p within `win` seconds of the span. Off unless FLEXSED_CORROB = (b, p, win).
    corr = getattr(config, "FLEXSED_CORROB", None)
    if corr and fresh:
        b_min, p_min, win = (float(x) for x in corr)
        try:
            cfw, ct, cl = _panns()
        except Exception as e3:
            cfw = None
            print(f"       [stage4] corroboration: PANNs unavailable ({type(e3).__name__}); BEATs only", flush=True)

        def _near(fw_, ts_, labs_, e_, thr):
            m = (np.asarray(ts_) >= e_.start - win) & (np.asarray(ts_) <= e_.end + win)
            cols = [i for i, lab_ in enumerate(labs_) if canonical(lab_) == key(e_)]
            return bool(cols) and bool(m.any()) and float(np.asarray(fw_)[m][:, cols].max()) >= thr
        n0 = len(fresh)
        fresh = [e for e in fresh if _near(framewise, times, labels, e, b_min)
                 or (cfw is not None and _near(cfw, ct, cl, e, p_min))]
        print(f"       [stage4] corroboration (BEATs {b_min} / PANNs {p_min} within {win} s): kept {len(fresh)} of {n0} FlexSED-only span(s)", flush=True)
    print(f"       [stage4] FlexSED (bar {fbar}): {len(fev)} span(s), {len(fresh)} new family/moment(s)", flush=True)
    trace("flexsed_raw", fev, "FlexSED spans as extracted")
    events = events + fresh
    flex_ids |= {id(e) for e in fresh}
    trace("union", events, "after the twin rule kept the earlier start")
    # Round 13, R13-2: the mirror of amendment 10 at the moment level. Where BEATs raised a span alone and FlexSED, asked
    # about every family at that moment, says it is clearly something else (top query of another family >= b) and not
    # this family (< MIRROR_OWN_MAX), the span is dropped (the laundromat's washing-machine rumble that BEATs called
    # Vehicle while FlexSED said Train). None = off.
    mb = getattr(config, "MIRROR_VETO", None)
    if mb:
        events, dropped = _mirror_veto(events, flex_ids | twinned, ffw, ftimes, flabels, float(mb),
                                       float(getattr(config, "MIRROR_OWN_MAX", 0.4)))
        # Round 14 amendment B, F7 (LISTENER_CONFIRMED_MIRROR): swap, don't add -- a span the mirror veto drops is kept
        # when the listener accepts its family (listener_p1: rule V4, else V12, else yes/no score > 3)
        if getattr(config, "LISTENER_CONFIRMED_MIRROR", False) and listener_p1 is not None and dropped:
            back = []
            for e in dropped:
                ok, how = listener_p1(e.label, e.start, e.end)
                if ok and getattr(config, "KEEP_NEEDS_V4", False):
                    ok = _v4_names(e)                              # round 21 K-V4
                LISTENER_STATS.setdefault("f7", []).append([e.label, round(e.start, 2), round(e.end, 2), ok, how])
                if ok:
                    back.append(e)
            events = events + back
            dropped = [e for e in dropped if e not in back]
        trace("mirror_veto", dropped, f"R13-2 dropped (b {mb})")
        print(f"       [stage4] mirror veto (b {mb}): dropped {len(dropped)} BEATs-only span(s)", flush=True)
    # Round 16 N2 (MASKED_WEAK_VETO): a weak BEATs-only span (conf < 0.5) that sits under speech or music (BEATs Speech or
    # Music >= 0.3 inside it) is mostly a phantom (280/415 audit): dropped unless FlexSED backs it (a twin >= its bar) or the
    # listener accepts its family (F7's P1 rule). Constants from the audit, not from DEV.
    if getattr(config, "MASKED_WEAK_VETO", False):
        sm = [i for i, lab_ in enumerate(labels) if lab_ in ("Speech", "Music")]
        gone = []
        if sm:
            T = np.asarray(times)
            tw = locals().get("twinned", set())
            for e in [e for e in events if id(e) not in flex_ids and id(e) not in tw and e.confidence < 0.5]:
                m = (T >= e.start - 1e-6) & (T <= e.end + 1e-6)
                if getattr(config, "MASKED_WEAK_NEED_MASK", True) and (
                        not m.any() or float(np.asarray(framewise)[m][:, sm].max()) < 0.3):
                    continue                                     # round 18 N2c drops the masking condition
                if getattr(config, "MASKED_WEAK_PANNS", False):
                    # round 21 N2e: the trigger is "PANNs does not reach its clip-veto bar for the family inside the span"
                    try:
                        pfw, pt, pl = _panns()
                        pc = [i for i, lab_ in enumerate(pl) if canonical(lab_) == canonical(e.label)]
                        pm = (np.asarray(pt) >= e.start - 1e-6) & (np.asarray(pt) <= e.end + 1e-6)
                        if pc and pm.any() and float(np.asarray(pfw)[pm][:, pc].max()) >= float(getattr(config, "PANNS_VETO", 0.05) or 0.05):
                            continue
                    except RuntimeError:
                        continue
                ok, how = listener_p1(e.label, e.start, e.end) if listener_p1 is not None else (False, "missing")
                if ok and getattr(config, "KEEP_NEEDS_V4", False):
                    ok = _v4_names(e)                              # round 21 K-V4
                if ok:
                    continue
                if getattr(config, "MASKED_WEAK_AF", False) and _af_p1_accepts(e):   # round 18 N2b: or AF V4 on the P1 cut
                    continue
                if getattr(config, "MASKED_WEAK_MISSING_KEEP", False) and how == "missing":
                    continue                                     # round 21 N2c-D: no second opinion was asked -> keep
                if getattr(config, "MASKED_WEAK_DASM_KEEP", False) and _dasm_keeps(e):
                    continue                                     # round 21 N2c-D: DASM >= F8's bar in the span +- 0.5 s
                gone.append(e)
        if getattr(config, "FLEX_ONLY_CONFIRM", False):
            # round 18 N2d: a FlexSED-only span (no BEATs span of its family) is kept only if BEATs hears its family at all
            # inside it (>= the tagger's own bar) or either listener accepts it (Qwen P1 rule / AF V4 on the P1 cut)
            from src.labels import canonical as _can
            T = np.asarray(times)
            ab = float(getattr(config, "AED_THRESHOLD", 0.175))
            for e in [e for e in events if id(e) in flex_ids and not getattr(e, "rescued", False)]:
                cols = [i for i, lab_ in enumerate(labels) if _can(lab_) == _can(e.label)]
                m = (T >= e.start - 1e-6) & (T <= e.end + 1e-6)
                if cols and m.any() and float(np.asarray(framewise)[m][:, cols].max()) >= ab:
                    continue
                if listener_p1 is not None and listener_p1(e.label, e.start, e.end)[0]:
                    continue
                if _af_p1_accepts(e):
                    continue
                gone.append(e)
        events = [e for e in events if e not in gone]
        trace("masked_weak_veto", gone, "N2 dropped")
        print(f"       [stage4] N2 masked weak BEATs: dropped {len(gone)} span(s)", flush=True)
    # Amendment 10 (2026-09-23): the second detector also carries the DISagreement. Where
    # BEATs names a family FlexSED never hears anywhere in the clip, the taxonomy shows the
    # sound is usually not there at all (Whale 0.13, Horse 0.13, Cat 0.00, Telephone 0.00).
    # The veto is deliberately one-sided -- it is applied only to labels FlexSED did not
    # itself raise (its own bar is higher than tau), so the sounds the union was adopted to
    # recover can never be deleted by it.
    dv = getattr(config, "DASM_CLIP_VETO", None)
    if dv:
        # round 28 DV: a span whose family DASM never reaches the calibrated clip bar is dropped unless a listener keeps it
        d, clip = getattr(config, "LISTENER_DASM_DIR", None), getattr(config, "_CURRENT_CLIP", None)
        f = Path(d) / f"{clip}.npz" if d and clip else None
        if f is not None and f.exists():
            z = np.load(f, allow_pickle=True)
            dpk = {}
            for i, lab_ in enumerate([str(x) for x in z["labels"]]):
                dpk[canonical(lab_)] = max(dpk.get(canonical(lab_), 0.0), float(z["fw"][:, i].max()))
            gone = []
            for e in events:
                if getattr(e, "rescued", False) or dpk.get(canonical(e.label), 1.0) >= float(dv):
                    continue
                if listener_p1 is not None and listener_p1(e.label, e.start, e.end)[0]:
                    continue
                if _af_p1_accepts(e):
                    continue
                gone.append(e)
            events = [e for e in events if e not in gone]
            print(f"       [stage4] DASM clip veto ({dv}): dropped {len(gone)} span(s)", flush=True)
    lv = getattr(config, "DASM_LOCAL_VETO", None)
    if lv:
        # round 29 DV-L / DV-G: the DASM test inside the span +- 0.5 s; keep by either listener ("either") or by both ("both")
        d, clip = getattr(config, "LISTENER_DASM_DIR", None), getattr(config, "_CURRENT_CLIP", None)
        f = Path(d) / f"{clip}.npz" if d and clip else None
        if f is not None and f.exists():
            z = np.load(f, allow_pickle=True)
            dl, dfw, dt = [str(x) for x in z["labels"]], z["fw"], z["times"]
            mode = getattr(config, "DASM_LOCAL_KEEP", "either")
            gone = []
            for e in events:
                if getattr(e, "rescued", False):
                    continue
                cols = [i for i, l in enumerate(dl) if canonical(l) == canonical(e.label)]
                m = (dt >= e.start - 0.5) & (dt <= e.end + 0.5)
                if not cols or not m.any() or float(dfw[m][:, cols].max()) >= float(lv):
                    continue
                if mode == "both":
                    if _v4_names_qwen(e) and _af_p1_accepts(e):
                        continue
                else:
                    if listener_p1 is not None and listener_p1(e.label, e.start, e.end)[0]:
                        continue
                    if _af_p1_accepts(e):
                        continue
                gone.append(e)
            events = [e for e in events if e not in gone]
            print(f"       [stage4] DASM local veto ({lv}, keep {mode}): dropped {len(gone)} span(s)", flush=True)
    kall = getattr(config, "KEEP_NEEDS_V4_ALL", None)
    if kall:
        # round 30 K4A: K-V4 on every drawn non-rescued span -- dropped when both open-inventory listeners were asked on its
        # P1 cut and neither names its family ("exact"), or no named family is it or a kind of it ("onto"); unasked -> kept
        from src.labels import is_descendant
        disp = float(getattr(config, "DISPLAY_THRESHOLD", 0.35))
        gone = []
        for e in events:
            if getattr(e, "rescued", False) or e.confidence < disp:
                continue
            q, a = _p1v4_lists(e)
            if q is None or a is None:
                continue
            fam = canonical(e.label)
            names = set(q) | set(a)
            if fam in names or (kall == "onto" and any(is_descendant(n, fam) for n in names)):
                continue
            gone.append(e)
        events = [e for e in events if e not in gone]
        print(f"       [stage4] K4A ({kall}): dropped {len(gone)} span(s)", flush=True)
    veto = float(getattr(config, "FLEXSED_VETO", 0) or 0)
    if veto > 0:
        peak = {}
        for i, lab in enumerate(flabels):
            peak[canonical(lab)] = max(peak.get(canonical(lab), 0.0), float(ffw[:, i].max()))
        before = len(events)
        events = [e for e in events if peak.get(key(e), 1.0) >= veto]
        print(f"       [stage4] cross-detector veto (tau {veto}): dropped {before - len(events)} span(s)", flush=True)
    # Amendment 16 (2026-09-23): the veto above is one-sided -- it asks FlexSED about
    # labels BEATs raised alone. Nothing asked about labels FLEXSED raised alone, and after
    # the first veto those became the largest error class, dominated by sustained textures
    # (Insect, Bicycle, Ice cream van, Power tool) that FlexSED holds above its bar for much
    # of a clip. A THIRD model settles them: PANNs CNN14, a different architecture trained
    # differently, scores median 0.424 where FlexSED is right and 0.026 where it is wrong.
    # The guard is the same as before -- a span BOTH detectors raised is never touched -- so
    # this can only remove spans that rest on one model's word alone.
    # Round 4 (docs/prereg_v4.md, 2026-09-28): BEATs' own clip-max for the family settles FlexSED-only spans
    # in place of PANNs -- the bar b keeps the same share PANNs kept on the 280; held-out 415 dC +0.005
    # [-0.048, +0.067], recall 49.7 vs 49.1 %. Same identity guard as the PANNs veto below.
    bveto = float(getattr(config, "BEATS_SELF_VETO", 0) or 0)
    if bveto > 0 and backend == "BEATs":
        bpeak = {}
        for i, lab in enumerate(labels):
            k = canonical(lab)
            bpeak[k] = max(bpeak.get(k, 0.0), float(framewise[:, i].max()))
        flex_only = {id(e) for e in fresh}
        before3 = len(events)
        events = [e for e in events if id(e) not in flex_only or bpeak.get(key(e), 1.0) >= bveto]
        print(f"       [stage4] BEATs self-veto (b {bveto}) on FlexSED-only spans: "
              f"dropped {before3 - len(events)} span(s)", flush=True)
    veto2 = float(getattr(config, "PANNS_VETO", 0) or 0)
    keep_b = set()
    if veto2 > 0:
        try:
            pfw, _pt, plabels = _panns()
            ppeak = {}
            for i, lab in enumerate(plabels):
                k = canonical(lab)
                ppeak[k] = max(ppeak.get(k, 0.0), float(pfw[:, i].max()))
            # by IDENTITY, not by family name: keying on the canonical family would also
            # drop BEATs' OWN span of that family whenever FlexSED raised the same family
            # elsewhere in the clip without merging, which is exactly the asymmetry the
            # guard exists to prevent.
            flex_only = {id(e) for e in fresh}          # raised by FlexSED, no BEATs twin
            before2 = len(events)
            if listen_on:
                # R13-3 (b): a FlexSED >= bar span the PANNs clip veto would drop is kept when the listener, asked about
                # the FlexSED 0.4-run that contains it, says the family is there (score > LISTENER_TH)
                for e in events:
                    if id(e) in flex_only and ppeak.get(key(e), 1.0) < veto2:
                        it = listener(e.label, e.start, e.end, contain=True)
                        LISTENER_STATS["b_asked"] += 1
                        if it is None:
                            LISTENER_STATS["b_missing"] += 1
                            LISTENER_STATS["b_missing_list"].append([e.label, round(e.start, 2), round(e.end, 2)])
                        elif _accepted(it, lth):
                            keep_b.add(id(e)); e.agree = bool((it.get("accept") or {}).get("AGREE_V4", False))
                            LISTENER_STATS["b_kept"].append([e.label, round(e.start, 2), round(e.end, 2), it.get("score")])
                        elif _arbiter_candidate(it):
                            keep_b.add(id(e)); e.arbiter = True
                            LISTENER_STATS.setdefault("arbiter", []).append([e.label, round(e.start, 2), round(e.end, 2), "b"])
            # Round 14 amendment G5 (PANNS_VETO_SKIP_ABOVE): a FlexSED span this strong is not put to the PANNs clip veto
            skip = getattr(config, "PANNS_VETO_SKIP_ABOVE", None)
            events = [e for e in events
                      if id(e) not in flex_only or ppeak.get(key(e), 1.0) >= veto2 or id(e) in keep_b
                      or (skip is not None and e.confidence >= float(skip))]
            print(f"       [stage4] PANNs veto (tau2 {veto2}) on FlexSED-only spans: "
                  f"dropped {before2 - len(events)} span(s)", flush=True)
        except Exception as e2:
            print(f"       [stage4] PANNs veto unavailable ({type(e2).__name__}: {e2}); skipped", flush=True)
    # Amendment 22, tier 3 (Adam, 2026-09-27: "take all sounds above X, then for sounds below X try Y"):
    # a BEATs span whose peak is in the band [AED_THRESHOLD, DISPLAY_THRESHOLD) -- heard, but too weak to be
    # shown -- is promoted to the display bar when FlexSED (>= f) or PANNs (>= p) rises for the same family
    # within `win` seconds of it. Off unless BEATS_LOWBAND_CORROB = (f, p, win).
    low = getattr(config, "BEATS_LOWBAND_CORROB", None)
    if low:
        f_min, p_min, win = (float(x) for x in low)
        disp = float(getattr(config, "DISPLAY_THRESHOLD", 0.35))
        try:
            lfw, lt, ll = _panns()
        except Exception:
            lfw = None

        def _near2(fw_, ts_, labs_, e_, thr):
            m = (np.asarray(ts_) >= e_.start - win) & (np.asarray(ts_) <= e_.end + win)
            cols = [i for i, lab_ in enumerate(labs_) if canonical(lab_) == key(e_)]
            return bool(cols) and bool(m.any()) and float(np.asarray(fw_)[m][:, cols].max()) >= thr
        promoted = 0
        for e in events:
            if id(e) in flex_ids or e.confidence >= disp:
                continue
            if _near2(ffw, ftimes, flabels, e, f_min) or (lfw is not None and _near2(lfw, lt, ll, e, p_min)):
                e.confidence = disp; promoted += 1
        print(f"       [stage4] low-band corroboration (FlexSED {f_min} / PANNs {p_min} within {win} s): "
              f"promoted {promoted} weak BEATs span(s) to the display bar", flush=True)
    if listen_on:
        for e in events:                                     # (b) kept by the listener = rescued
            if id(e) in keep_b:
                e.rescued = True
        events = _listener_band(events, flex_ids, framewise, times, labels, ffw, ftimes, flabels, fbar, listener, lth,
                                extra=extra)
        print(f"       [stage4] listener rescue: kept {len(LISTENER_STATS['b_kept'])} vetoed, added "
              f"{len(LISTENER_STATS['a_added'])} band + {len(LISTENER_STATS['c_added'])} BEATs-band span(s); cache misses "
              f"a {LISTENER_STATS['a_missing']}/{LISTENER_STATS['a_asked']} b {LISTENER_STATS['b_missing']}/"
              f"{LISTENER_STATS['b_asked']} c {LISTENER_STATS['c_missing']}/{LISTENER_STATS['c_asked']}", flush=True)
    return events, flex_ids, ffw


# ----------------------------------------------------------------------------- R13-3: listener rescue
LISTENER_STATS: dict = {}
LISTEN_RUN_BAR, LISTEN_RUN_GAP, LISTEN_SHORT = 0.4, 0.24, 0.5


def _reset_listener_stats():
    LISTENER_STATS.clear()
    LISTENER_STATS.update({"b_asked": 0, "b_missing": 0, "b_kept": [], "b_missing_list": [],
                           "a_asked": 0, "a_missing": 0, "a_added": [], "a_missing_list": [],
                           "c_asked": 0, "c_missing": 0, "c_added": [], "c_missing_list": [],
                           })


def _runs(col, ts, bar, gap_s):
    """frames >= bar, runs separated by <= gap_s merged; (i, j) frame ranges (as benchmark/gold/dev_listener.py)"""
    dt = float(ts[1] - ts[0]) if len(ts) > 1 else 0.04
    on = col >= bar
    out, i, n = [], 0, len(on)
    while i < n:
        if not on[i]:
            i += 1; continue
        j = i
        while j < n and on[j]:
            j += 1
        if out and (i - out[-1][1]) * dt <= gap_s + 1e-6:
            out[-1] = (out[-1][0], j)
        else:
            out.append((i, j))
        i = j
    return out, dt


def _arbiter_candidate(it) -> bool:
    """Round 14 amendment F, LISTENER_ARBITER: a band candidate the rule rejects but Qwen V4 AND Audio Flamingo V4 accept
    while V12 says no; it is kept as rescued and marked `arbiter`, and stage 5 asks the VLM arbiter question about it."""
    if not getattr(config, "LISTENER_ARBITER", False):
        return False
    acc = it.get("accept") or {}
    return bool(acc.get("AGREE_V4", False)) and not bool(acc.get("V12", False))


def _tier(acc: dict, peak: float) -> bool:
    """amendment E TIER: peak >= 0.6 -> Qwen V4 (amendment K3, TIER_HIGH_OR: Qwen V4 OR AF V4); below -> Qwen V4 AND AF V4"""
    v4, af = bool(acc.get("V4", False)), bool(acc.get("AF_V4", False))
    if peak >= float(getattr(config, "TIER_SPLIT", 0.6)):          # round 25: the split is a config value (was 0.6)
        return (v4 or af) if getattr(config, "TIER_HIGH_OR", False) else v4
    return v4 and af


def _accepted(it, lth: float) -> bool:
    """R13-3 decision for one cached item: the named variant's accept flag (config.LISTENER_RULE, amendment A) or, with no
    rule, the yes/no score > LISTENER_TH"""
    rule = getattr(config, "LISTENER_RULE", None)
    if rule:
        return bool((it.get("accept") or {}).get(rule, False))
    return float(it["score"]) > lth


def _cache_items(path):
    """the items of one listener cache, or of several joined by ';' (a base cache + a supplement, e.g. the K2 runs)"""
    out = []
    for p in str(path).split(";"):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        out += list(d["items"] if isinstance(d, dict) else d)
    return out


def listener_from_vcache(path, clip: str, tol: float = 0.02):
    """Amendment A: the variants cache (benchmark/gold/listener_variants.py; items with clip, pool P2 / PV, family, label,
    start, end, cut_start, cut_end, accept{rule: bool}). lookup(label, start, end, contain=False): (a) the P2 run with this
    (family, start, end); contain=True (b): the PV item of this vetoed span (same family, start, end). P1 is never used."""
    from src.labels import canonical
    items = [x for x in _cache_items(path) if x.get("clip") == clip and x.get("pool") in ("P2", "PV")]
    # Round 14 amendment C: the second listener (Audio Flamingo Next, LISTENER_AFCACHE, same keys) -- rule AGREE_V4 =
    # Qwen V4 AND AF V4, AGREE_V12 = Qwen V12 AND AF V4; an item AF did not score is not accepted by AGREE
    afp = getattr(config, "LISTENER_AFCACHE", None)
    qbp = getattr(config, "LISTENER_V4B_CACHE", None)
    qb = ({(x.get("pool"), x["family"], x.get("label"), round(x["start"], 2), round(x["end"], 2)): x
           for x in _cache_items(qbp) if x.get("clip") == clip} if qbp else None)
    kcp, kfield = getattr(config, "LISTENER_KCACHE", None), getattr(config, "LISTENER_KFIELD", "accept_norm")
    kc = None
    if kcp:
        kc = {(x.get("pool"), x["family"], x.get("label"), round(x["start"], 2), round(x["end"], 2)): x
              for x in _cache_items(kcp) if x.get("clip") == clip}
    af = {}
    if afp:
        for x in _cache_items(afp):
            if x.get("clip") == clip:
                af[(x.get("pool"), x["family"], x.get("label"), round(x["start"], 2), round(x["end"], 2))] = x

    def look(label, start, end, contain=False):
        fam = canonical(label)
        pool = "PV" if contain else "P2"
        c = [x for x in items if x["pool"] == pool and x["family"] == fam
             and abs(x["start"] - start) <= tol and abs(x["end"] - end) <= tol]
        if not c:
            return None
        same = [x for x in c if x.get("label") == label]
        it = (same or c)[0]
        if getattr(config, "FIX_CTRL", False) and it.get("v2_x_ctrl") is None and "V1" in (it.get("accept") or {}):
            # amendment F, FIX-CTRL: no control window (the family is >= 0.2 across the whole clip) -> the paired-cut leg V2
            # cannot be evaluated and counts as neutral: V12 = V1 alone
            it = {**it, "accept": {**it["accept"], "V12": bool(it["accept"]["V1"])}}
        if afp:
            a = af.get((it["pool"], it["family"], it.get("label"), round(it["start"], 2), round(it["end"], 2)))
            av4 = bool(((a or {}).get("accept") or {}).get("V4", False))
            acc = dict(it.get("accept") or {})
            acc["AGREE_V4"] = bool(acc.get("V4", False)) and av4
            acc["AF_V4"] = av4
            acc["AGREE_V12"] = bool(acc.get("V12", False)) and av4
            # amendment E: TIER = Qwen V4, and below a FlexSED run peak of 0.6 also AF V4; QV4_AFYN = Qwen V4 AND AF yes/no > 0
            pk = float(it.get("peak", 1.0) if it.get("peak") is not None else 1.0)
            if qb is not None:                                   # round 27 QE: the Qwen leg = V4 OR the V4b prompt
                b = qb.get((it["pool"], it["family"], it.get("label"), round(it["start"], 2), round(it["end"], 2)))
                acc["V4"] = bool(acc.get("V4", False)) or bool(((b or {}).get("accept") or {}).get("V4B", False))
            acc["TIER"] = _tier(acc, pk)
            if getattr(config, "TIER_2OF3_DASM", False):       # round 22 TD: or two of {Qwen V4, AF V4, DASM >= bar}
                from src.types import AudioEvent as _AE
                dv = _dasm_keeps(_AE(it["family"], float(it["start"]), float(it["end"]), 0.0))
                acc["TIER"] = acc["TIER"] or (int(bool(acc.get("V4", False))) + int(av4) + int(dv)) >= 2
            ayn = (a or {}).get("afn_yn_x")
            acc["QV4_AFYN"] = bool(acc.get("V4", False)) and ayn is not None and float(ayn) > 0
            acc["AF_missing"] = a is None
            # Round 16 N3: a third open-inventory listener (Kimi-Audio, LISTENER_KCACHE, flag LISTENER_KFIELD) -- TIER3: at peak
            # >= 0.6 at least 2 of {Qwen V4, AF V4, Kimi V4}; below, all 3
            if kc is not None:
                k = kc.get((it["pool"], it["family"], it.get("label"), round(it["start"], 2), round(it["end"], 2)))
                kv4 = bool(((k or {}).get(kfield) or {}).get("V4", False))
                acc["KIMI_V4"] = kv4
                votes = int(bool(acc.get("V4", False))) + int(av4) + int(kv4)
                acc["TIER3"] = votes >= 2 if pk >= float(getattr(config, "TIER_SPLIT", 0.6)) else votes == 3
            it = {**it, "accept": acc}
        return it
    return look


def listener_from_cache(path, clip: str, tol: float = 0.02):
    """R13-3: per-span listener scores (benchmark/gold/dev_listener.json: items with clip, family, label, start, end,
    cut_start, cut_end, score). Returns lookup(label, start, end, contain=False) -> the cached item or None: the same
    (family, start, end) within tol, or with contain=True the P2 run of that family that contains [start, end]."""
    from src.labels import canonical
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    items = [x for x in (d["items"] if isinstance(d, dict) else d) if x.get("clip") == clip and x.get("score") is not None]

    def look(label, start, end, contain=False):
        fam = canonical(label)
        if contain:
            c = [x for x in items if x.get("pool") == "P2" and x["family"] == fam
                 and x["start"] <= start + tol and x["end"] >= end - tol]
        else:
            c = [x for x in items if x["family"] == fam and abs(x["start"] - start) <= tol and abs(x["end"] - end) <= tol]
        if not c:
            return None
        same = [x for x in c if x.get("label") == label]
        return max(same or c, key=lambda x: float(x["score"]))
    return look


def _family_match(a: str, b: str) -> bool:
    """amendment F / H, FIX-FAM: two query labels are one family if their canonical families are equal or one is an
    ontology ancestor or descendant of the other (the pipeline's canonical family / score_per_sound.same_family; siblings
    such as Dog and Cat are NOT one family; amendment H corrected an earlier sibling reading)"""
    from src.labels import canonical, is_descendant
    if a == b or canonical(a) == canonical(b) or is_descendant(a, b) or is_descendant(b, a):
        return True
    ca, cb = canonical(a), canonical(b)
    return is_descendant(ca, cb) or is_descendant(cb, ca) or is_descendant(a, cb) or is_descendant(b, ca)


def dasm_rescue_events(clip: str, present=None):
    """Round 19 DR (DASM_RESCUE): DASM runs no B0r stage-4 span touched (config.DASM_P4_CACHE, benchmark/gold/dasm_rescue.py)
    that BOTH open-inventory listeners name (Qwen V4 and Audio Flamingo V4) become rescued spans (label = the family,
    confidence = the DASM peak). The base's rescue filters (ONCE, F8) then apply as to any rescued span."""
    from src.types import AudioEvent
    p = getattr(config, "DASM_P4_CACHE", None)
    if not (getattr(config, "DASM_RESCUE", False) and p):
        return []
    out = []
    for x in _cache_items(p):
        if x.get("clip") == clip and x.get("qwen_v4") and (x.get("accept") or {}).get("V4"):
            if getattr(config, "DASM_RESCUE_NEW_ONLY", False) and present is not None and x["family"] in present:
                continue                                              # round 20 DR2: new families only
            out.append(AudioEvent(x["family"], float(x["start"]), float(x["end"]), float(x["peak"]), rescued=True, agree=True))
    if out:
        print(f"       [stage4] DASM rescue: {len(out)} span(s) {[(e.label, round(e.start, 2)) for e in out]}", flush=True)
    return out


_P1V4 = {}


def _v4_names(e) -> bool:
    """round 21 K-V4: an open-inventory listener names e's family on its P1 cut (Qwen V4 family list in config.RELABEL_P1V4,
    or Audio Flamingo V4 in LISTENER_AFCACHE)"""
    from src.labels import canonical
    p, clip = getattr(config, "RELABEL_P1V4", None), getattr(config, "_CURRENT_CLIP", None)
    if p and clip is not None:
        key = (p, clip)
        if key not in _P1V4:
            _P1V4[key] = [x for x in _cache_items(p) if x.get("clip") == clip]
        fam = canonical(e.label)
        c = [x for x in _P1V4[key] if x["family"] == fam and abs(x["end"] - e.end) <= 0.02 and e.start - 0.02 <= x["start"] <= e.end]
        if c and fam in (min(c, key=lambda x: abs(x["start"] - e.start)).get("qwen_fams") or []):
            return True
    return _af_p1_accepts(e)


def _p1v4_lists(e):
    """round 30 K4A: (qwen_fams, af_fams) of e's P1 cut in config.RELABEL_P1V4 (None where not asked)"""
    from src.labels import canonical
    p, clip = getattr(config, "RELABEL_P1V4", None), getattr(config, "_CURRENT_CLIP", None)
    if not p or clip is None:
        return None, None
    key = (p, clip)
    if key not in _P1V4:
        _P1V4[key] = [x for x in _cache_items(p) if x.get("clip") == clip]
    fam = canonical(e.label)
    c = [x for x in _P1V4[key] if x["family"] == fam and abs(x["end"] - e.end) <= 0.02 and e.start - 0.02 <= x["start"] <= e.end]
    if not c:
        return None, None
    it = min(c, key=lambda x: abs(x["start"] - e.start))
    return it.get("qwen_fams"), it.get("af_fams")


def _v4_names_qwen(e) -> bool:
    """Qwen V4 (P1 open inventory, config.RELABEL_P1V4) names e's family on its P1 cut"""
    from src.labels import canonical
    p, clip = getattr(config, "RELABEL_P1V4", None), getattr(config, "_CURRENT_CLIP", None)
    if not p or clip is None:
        return False
    key = (p, clip)
    if key not in _P1V4:
        _P1V4[key] = [x for x in _cache_items(p) if x.get("clip") == clip]
    fam = canonical(e.label)
    c = [x for x in _P1V4[key] if x["family"] == fam and abs(x["end"] - e.end) <= 0.02 and e.start - 0.02 <= x["start"] <= e.end]
    return bool(c) and fam in (min(c, key=lambda x: abs(x["start"] - e.start)).get("qwen_fams") or [])


_DASMC = {}


def _dasm_keeps(e) -> bool:
    """round 21 N2c-D: DASM gives e's family >= LISTENER_DASM_BAR (0.575) within the span +- 0.5 s (config.LISTENER_DASM_DIR)"""
    from src.labels import canonical
    d, clip = getattr(config, "LISTENER_DASM_DIR", None), getattr(config, "_CURRENT_CLIP", None)
    if not d or clip is None:
        return False
    f = Path(d) / f"{clip}.npz"
    if f not in _DASMC:
        _DASMC[f] = np.load(f, allow_pickle=True) if f.exists() else None
    z = _DASMC[f]
    if z is None:
        return False
    fw, t, L = z["fw"], z["times"], [str(x) for x in z["labels"]]
    cols = [i for i, l in enumerate(L) if canonical(l) == canonical(e.label)]
    m = (t >= e.start - 0.5) & (t <= e.end + 0.5)
    return bool(cols) and bool(m.any()) and float(fw[m][:, cols].max()) >= float(getattr(config, "LISTENER_DASM_BAR", 0.575))


_AFP1 = {}


def _af_p1_accepts(e) -> bool:
    """round 18 N2b: Audio Flamingo Next V4 accepts this span's family on its P1 cut (config.LISTENER_AFCACHE, P1 items;
    same match as listener_p1_lookup: family, end within 0.02 s, start inside the span)"""
    from src.labels import canonical
    p = getattr(config, "LISTENER_AFCACHE", None)
    clip = getattr(config, "_CURRENT_CLIP", None)
    if not p or clip is None:
        return False
    key = (p, clip)
    if key not in _AFP1:
        _AFP1[key] = [x for x in _cache_items(p) if x.get("clip") == clip and x.get("pool") == "P1"]
    fam = canonical(e.label)
    c = [x for x in _AFP1[key] if x["family"] == fam and abs(x["end"] - e.end) <= 0.02 and e.start - 0.02 <= x["start"] <= e.end]
    return bool(c) and bool(((min(c, key=lambda x: abs(x["start"] - e.start)).get("accept")) or {}).get("V4", False))


def post_rules(events, ffw, ftimes, flabels, clip: str, origin=None, listener_p1=None):
    """Round 17 (docs/prereg_round13_detector_push.md), after the rescue filters; both flags default off.
    R3 CO_ONSET_ARB: two display-level spans of different families starting within +-0.3 s -> keep the one with the higher
      FlexSED family peak inside its own span, unless the listener (listener_p1, F7's rule) accepts both.
    R1 RELABEL_2L: a display-level tagger-origin, non-rescued span whose P1 cut neither open-inventory listener (Qwen V4 and
      Audio Flamingo V4 family lists, config.RELABEL_P1V4) names as its own family, while both name one same depictable
      family F -> the span is relabelled F (first such F in Qwen's order). origin: {id(event): "tagger"|"flex"}.
    Returns (events, log)."""
    from src.labels import canonical
    log = {"R3": [], "R1": []}
    disp = float(getattr(config, "DISPLAY_THRESHOLD", 0.35))
    origin = origin or {}

    def fpeak(e):
        cols = [i for i, lab in enumerate(flabels) if canonical(lab) == canonical(e.label)]
        m = (np.asarray(ftimes) >= e.start - 1e-6) & (np.asarray(ftimes) <= e.end + 1e-6)
        return float(np.asarray(ffw)[m][:, cols].max()) if cols and m.any() else 0.0   # ffw is [frames, queries]
    if getattr(config, "CO_ONSET_ARB", False) and ffw is not None:
        vis = sorted([e for e in events if e.confidence >= disp], key=lambda e: e.start)
        drop = set()
        for i, a in enumerate(vis):
            for b in vis[i + 1:]:
                if b.start - a.start > 0.3:
                    break
                if id(a) in drop or id(b) in drop or canonical(a.label) == canonical(b.label):
                    continue
                if listener_p1 is not None and listener_p1(a.label, a.start, a.end)[0] and listener_p1(b.label, b.start, b.end)[0]:
                    continue
                pa, pb = fpeak(a), fpeak(b)
                lo = b if pa >= pb else a
                drop.add(id(lo)); log["R3"].append([lo.label, round(lo.start, 2), round(min(pa, pb), 3), round(max(pa, pb), 3)])
        events = [e for e in events if id(e) not in drop]
    path = getattr(config, "RELABEL_P1V4", None)
    if getattr(config, "RELABEL_2L", False) and path:
        its = [x for x in _cache_items(path) if x.get("clip") == clip]
        for e in events:
            if e.confidence < disp or getattr(e, "rescued", False) or origin.get(id(e), "tagger") != "tagger":
                continue
            fam = canonical(e.label)
            c = [x for x in its if x["family"] == fam and abs(x["end"] - e.end) <= 0.02 and e.start - 0.02 <= x["start"] <= e.end]
            if not c:
                continue
            it = min(c, key=lambda x: abs(x["start"] - e.start))
            q, a = it.get("qwen_fams") or [], it.get("af_fams")
            if a is None or fam in q or fam in a:
                continue
            both = [f for f in q if f in a and f != fam]
            if both:
                log["R1"].append([e.label, round(e.start, 2), both[0]])
                e.label = both[0]
    return events, log


def filter_rescued(events, ffw, ftimes, flabels, dasm=None):
    """Round 14 precision filters on RESCUED spans only (docs/prereg_round13_detector_push.md, Round 14 + addendum), run
    after onset refinement so the B0 onsets are final. Order: F4, F6, F5, F8, then F1 (F1 picks among the survivors).
    F4 LISTENER_LOCAL_WINNER: at the rescued span's peak frame (its own FlexSED query's highest frame inside the span), its
      family must be the top-scoring depictable FlexSED query.
    F6 LISTENER_EDGE: drop it if that peak lies in the first or last LISTENER_EDGE_S (0.3) s of the clip.
    F5 LISTENER_SHADOW: drop it if the peak lies within +-LISTENER_SHADOW_S (0.2) s of the onset of a B0 picture candidate
      (a non-rescued, drawable span at or above the display bar) of a different family whose score (conf) is higher.
    F8 LISTENER_DASM_VOTE (amendment B): keep it only if DASM gives its family >= LISTENER_DASM_BAR within the span +- 0.5 s.
    F1 LISTENER_NEW_TYPE_ONCE: rescue only a family with no B0 picture in the clip -- in stage 4, no non-rescued span of
      that family at or above the display bar -- and at most one rescued span per family per clip (highest FlexSED peak).
    Returns (events, dropped) with dropped = {filter: [[label, start, end, why], ...]}."""
    from src.labels import canonical, is_salient_nonspeech
    f4 = bool(getattr(config, "LISTENER_LOCAL_WINNER", False))
    f1 = bool(getattr(config, "LISTENER_NEW_TYPE_ONCE", False))
    f5 = bool(getattr(config, "LISTENER_SHADOW", False))
    f6 = bool(getattr(config, "LISTENER_EDGE", False))
    f8 = bool(getattr(config, "LISTENER_DASM_VOTE", False))
    once = bool(getattr(config, "LISTENER_ONCE", False))
    dropped = {"F4": [], "F6": [], "F5": [], "F8": [], "F1": [], "ONCE": []}
    resc = [e for e in events if getattr(e, "rescued", False)]
    if not (f4 or f1 or f5 or f6 or f8 or once) or not resc or ffw is None:
        return events, dropped
    fw_, ft_ = np.asarray(ffw), np.asarray(ftimes)
    dt = float(ft_[1] - ft_[0]) if len(ft_) > 1 else 0.04
    clip_end = float(ft_[-1] + dt)
    col = {l: i for i, l in enumerate(flabels)}
    old = config.LABEL_FILTER
    config.LABEL_FILTER = "depictable"                     # the pipeline's label filter (use_scored / use_shipped)
    try:
        dep = np.array([bool(is_salient_nonspeech(l)) for l in flabels])
        disp = float(getattr(config, "DISPLAY_THRESHOLD", 0.35))
        b0 = [e for e in events if not getattr(e, "rescued", False) and e.confidence >= disp
              and is_salient_nonspeech(e.label)]
    finally:
        config.LABEL_FILTER = old

    def peak(e):
        c = col.get(e.label)
        if c is None:
            return None, float(e.confidence), 0.5 * (e.start + e.end)
        m = (ft_ >= e.start) & (ft_ < e.end)
        if not m.any():
            m = np.zeros(len(ft_), bool); m[int(np.argmin(np.abs(ft_ - 0.5 * (e.start + e.end))))] = True
        idx = np.where(m)[0]
        k = int(idx[np.argmax(fw_[idx, c])])
        return k, float(fw_[k, c]), float(ft_[k] + dt / 2)
    P = {id(e): peak(e) for e in resc}
    drop = set()
    sig = lambda e: [e.label, round(e.start, 2), round(e.end, 2)]
    if f4:
        for e in resc:
            k = P[id(e)][0]
            if k is None or not dep.any():
                continue
            top = int(np.argmax(np.where(dep, fw_[k], -1.0)))
            if not (_family_match(flabels[top], e.label) if getattr(config, "FIX_FAM", False)
                    else canonical(flabels[top]) == canonical(e.label)):
                drop.add(id(e)); dropped["F4"].append(sig(e) + [flabels[top]])
    if f6:
        edge = float(getattr(config, "LISTENER_EDGE_S", 0.3))
        for e in resc:
            if id(e) in drop:
                continue
            pt = P[id(e)][2]
            if pt < edge or pt > clip_end - edge:
                drop.add(id(e)); dropped["F6"].append(sig(e) + [round(pt, 2)])
    if f5:
        win = float(getattr(config, "LISTENER_SHADOW_S", 0.2))
        for e in resc:
            if id(e) in drop:
                continue
            _k, pk, pt = P[id(e)]
            sh = [o for o in b0 if canonical(o.label) != canonical(e.label) and abs(pt - o.start) <= win
                  and o.confidence > pk]
            if sh:
                drop.add(id(e)); dropped["F5"].append(sig(e) + [sh[0].label])
    if f8:
        # amendment B, F8: a third vote -- DASM must give the same family >= LISTENER_DASM_BAR somewhere in the span +- 0.5 s;
        # with no DASM scores for the clip the span is kept (counted as "no DASM")
        bar = float(getattr(config, "LISTENER_DASM_BAR", 0.575))
        pad = float(getattr(config, "LISTENER_DASM_PAD", 0.5))
        for e in resc:
            if id(e) in drop:
                continue
            if getattr(config, "F8_BYPASS_BOTH", False) and getattr(e, "agree", False):
                dropped.setdefault("F8_bypassed", []).append(sig(e)); continue   # amendment K1: two audio LLMs outvote one SED
            if dasm is None:
                dropped.setdefault("F8_no_dasm", []).append(sig(e)); continue
            dfw, dts, dl = dasm
            cols = [i for i, l in enumerate(dl) if canonical(l) == canonical(e.label)]
            m = (np.asarray(dts) >= e.start - pad) & (np.asarray(dts) <= e.end + pad)
            v = float(np.asarray(dfw)[m][:, cols].max()) if cols and m.any() else 0.0
            rk = getattr(config, "LISTENER_DASM_RANK", None)
            if rk and cols and m.any():
                # Round 16 N4: rank readout -- the family is among DASM's top-rk queries at some frame of the span +- pad
                sub = np.asarray(dfw)[m]
                kth = np.sort(sub, axis=1)[:, -int(rk)]
                v = 1.0 if bool((sub[:, cols].max(axis=1) >= kth).any()) else 0.0
                bar = 0.5
            if v < bar:
                drop.add(id(e)); dropped["F8"].append(sig(e) + [round(v, 3) if cols else "no DASM query"])
    if f1:
        have = {canonical(e.label) for e in events if not getattr(e, "rescued", False) and e.confidence >= disp}
        best = {}
        for e in resc:
            if id(e) in drop:
                continue
            fam = canonical(e.label)
            if fam in have:
                drop.add(id(e)); dropped["F1"].append(sig(e) + ["B0 has the family"])
                continue
            pk = P[id(e)][1]
            if getattr(config, "FIX_EARLY", False):             # amendment F, FIX-EARLY: the earliest, not the strongest
                pk = -float(e.start)
            if fam not in best or pk > best[fam][0] or (pk == best[fam][0] and e.start < best[fam][1].start):
                if fam in best:
                    o = best[fam][1]; drop.add(id(o)); dropped["F1"].append(sig(o) + ["not the strongest"])
                best[fam] = (pk, e)
            else:
                drop.add(id(e)); dropped["F1"].append(sig(e) + ["not the strongest"])
    if once:
        # amendment E, ONCE (F1b): at most one rescued span per family per clip, the EARLIEST (the first onset is the one
        # a viewer needs); no "already shown" condition
        first = {}
        gap = getattr(config, "ONCE_GAP", None)                 # round 22 ONCE-G: a later rescue > gap s away is a new event
        for e in sorted([e for e in resc if id(e) not in drop], key=lambda e: (e.start, e.end)):
            fam = canonical(e.label)
            if fam in first and (gap is None or e.start - first[fam].start <= float(gap)):
                drop.add(id(e)); dropped["ONCE"].append(sig(e) + ["not the first"])
            else:
                first[fam] = e
    return [e for e in events if id(e) not in drop], dropped


def _listener_band(events, flex_ids, framewise, times, labels, ffw, ftimes, flabels, fbar, listener, lth, extra=None):
    """R13-3 (a): a FlexSED 0.4-run (gaps <= 0.24 s merged) with peak in [LISTENER_LO, bar) and no same-family span over it
    becomes a FlexSED-only span (its run, or the listener's 1-s cut when shorter than 0.5 s) when the listener's score >
    LISTENER_TH; added after the PANNs clip veto, so exempt from it. (c) with LISTENER_BEATS_TH set: a BEATs run >=
    AED_THRESHOLD with peak below the display bar and no same-family span over it is added at the display bar when its
    score > LISTENER_BEATS_TH. A run with no cached score is not rescued (counted in LISTENER_STATS)."""
    from src.labels import canonical
    lo = float(getattr(config, "LISTENER_LO", 0.4))
    fw_, ft_ = np.asarray(ffw), np.asarray(ftimes)

    k2 = bool(getattr(config, "RESCUE_COVERED", False))
    disp = float(getattr(config, "DISPLAY_THRESHOLD", 0.35))

    def covered(fam, a, b):
        # amendment K2 (RESCUE_COVERED): a same-family BEATs span below the display bar no longer hides a band run
        return any(canonical(e.label) == fam and min(b, e.end) - max(a, e.start) > 0
                   and not (k2 and id(e) not in flex_ids and e.confidence < disp) for e in events)
    add = []
    for c, lab in enumerate(flabels):
        fam = canonical(lab)
        rr, dt = _runs(fw_[:, c], ft_, LISTEN_RUN_BAR, LISTEN_RUN_GAP)
        for i, j in rr:
            pk = float(fw_[i:j, c].max())
            if not (lo <= pk < fbar):
                continue
            a, b = float(ft_[i]), float(ft_[j - 1] + dt)
            if covered(fam, a, b):
                continue
            LISTENER_STATS["a_asked"] += 1
            it = listener(lab, a, b)
            if it is None:
                LISTENER_STATS["a_missing"] += 1
                LISTENER_STATS["a_missing_list"].append([lab, round(a, 2), round(b, 2), round(pk, 3)])
                continue
            if getattr(config, "TIER_SPECIFIC", False) and (it.get("accept") or {}).get("TIER") is not None:
                # amendment I5: the tier's peak = max over the family's queries and its specific child queries in the run
                spk = max(pk, _family_peak(fam, a, b, fw_, ft_, flabels, extra))
                acc = dict(it["accept"])
                acc["TIER"] = _tier(acc, spk)
                it = {**it, "accept": acc}
            arb = False
            if not _accepted(it, lth) and _arbiter_candidate(it):
                arb = True
                LISTENER_STATS.setdefault("arbiter", []).append([lab, round(a, 2), round(b, 2), "a"])
            if arb or _accepted(it, lth):
                s, e_ = (a, b) if b - a >= LISTEN_SHORT else (float(it["cut_start"]), float(it["cut_end"]))
                add.append(AudioEvent(lab, s, e_, pk, rescued=True, arbiter=arb,
                                      agree=bool((it.get("accept") or {}).get("AGREE_V4", False))))
                LISTENER_STATS["a_added"].append([lab, round(s, 2), round(e_, 2), round(pk, 3), it.get("score")])
    events = events + add
    flex_ids |= {id(e) for e in add}
    bth = getattr(config, "LISTENER_BEATS_TH", None)
    if bth is not None:
        disp = float(getattr(config, "DISPLAY_THRESHOLD", 0.35))
        tlo = float(getattr(config, "AED_THRESHOLD", 0.175))
        bw, bt = np.asarray(framewise), np.asarray(times)
        seen, addc = set(), []
        for c, lab in enumerate(labels):
            fam = canonical(lab)
            rr, dt = _runs(bw[:, c], bt, tlo, 0.0)
            for i, j in rr:
                pk = float(bw[i:j, c].max())
                if pk >= disp:
                    continue
                a, b = float(bt[i]), float(bt[j - 1] + dt)
                if covered(fam, a, b) or (fam, round(a, 3), round(b, 3)) in seen:
                    continue
                seen.add((fam, round(a, 3), round(b, 3)))
                LISTENER_STATS["c_asked"] += 1
                it = listener(lab, a, b)
                if it is None:
                    LISTENER_STATS["c_missing"] += 1
                    LISTENER_STATS["c_missing_list"].append([lab, round(a, 2), round(b, 2), round(pk, 3)])
                    continue
                if float(it["score"]) > float(bth):
                    addc.append(AudioEvent(lab, a, b, disp))
                    LISTENER_STATS["c_added"].append([lab, round(a, 2), round(b, 2), round(pk, 3), float(it["score"])])
        events = events + addc
    return events
