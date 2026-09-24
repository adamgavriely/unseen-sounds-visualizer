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
    # amendment 8 (2026-09-22): the open-vocabulary second detector, added as a UNION with its own
    # bar -- BEATs is deaf to sounds that speech or music masks, FlexSED is asked one label at a
    # time and hears them (docs/GOLD_RERUN_2026-09-22.md sec 10). Both bars are set on Adam's DEV
    # half; a clip with no cache falls back to BEATs alone.
    fbar = float(getattr(config, "FLEXSED_BAR", 0) or 0)
    if fbar > 0:
        from src.stage4_audio_event_detection import flexsed_infer as FX
        try:
            ffw, ftimes, flabels = FX.infer_flexsed(Path(wav_path), device)
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
            # a family both detectors report at the same moment keeps the earlier onset
            def key(e):
                return canonical(e.label)
            fresh = []
            for e in fev:
                twin = [b for b in events if key(b) == key(e) and b.start - 1.0 <= e.end and e.start - 1.0 <= b.end]
                if twin:
                    for b in twin:
                        b.start = min(b.start, e.start)
                else:
                    fresh.append(e)
            print(f"       [stage4] FlexSED (bar {fbar}): {len(fev)} span(s), {len(fresh)} new family/moment(s)", flush=True)
            trace("flexsed_raw", fev, "FlexSED spans as extracted")
            events = events + fresh
            flex_ids |= {id(e) for e in fresh}
            trace("union", events, "after the twin rule kept the earlier start")
            # Amendment 10 (2026-09-23): the second detector also carries the DISagreement. Where
            # BEATs names a family FlexSED never hears anywhere in the clip, the taxonomy shows the
            # sound is usually not there at all (Whale 0.13, Horse 0.13, Cat 0.00, Telephone 0.00).
            # The veto is deliberately one-sided -- it is applied only to labels FlexSED did not
            # itself raise (its own bar is higher than tau), so the sounds the union was adopted to
            # recover can never be deleted by it.
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
            veto2 = float(getattr(config, "PANNS_VETO", 0) or 0)
            if veto2 > 0:
                try:
                    pfw, _pt, plabels = _infer(Path(wav_path), device)
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
                    events = [e for e in events
                              if id(e) not in flex_only or ppeak.get(key(e), 1.0) >= veto2]
                    print(f"       [stage4] PANNs veto (tau2 {veto2}) on FlexSED-only spans: "
                          f"dropped {before2 - len(events)} span(s)", flush=True)
                except Exception as e2:
                    print(f"       [stage4] PANNs veto unavailable ({type(e2).__name__}: {e2}); skipped", flush=True)
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
    n_classes = len({e.label for e in events})
    print(f"       [stage4] {backend} SED: {len(events)} event span(s) over "
          f"{n_classes} class(es) (threshold={threshold}).")
    if plot_path is not None:
        plot_timeline(framewise, times, labels, Path(plot_path),
                      top_k=plot_top_k, threshold=threshold)
        print(f"       [stage4] timeline plot -> {plot_path}")
    return events
