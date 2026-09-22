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

import numpy as np

import config
from src.types import AudioEvent

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


def _refine_onsets_cam(wav_path: Path, events, labels, device: str):
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
    out = []
    for e in events:
        c = idx.get(e.label)
        if c is None:
            out.append(e); continue
        # the first window that fired ends STAMP_OFFSET after the stamp
        w0 = e.start + B.STAMP_OFFSET - B.WINDOW
        try:
            t, ramp = B.occlusion_onset(audio, B.SR, w0, c, device)
        except Exception as ex:
            print(f"       [stage4] onset refinement failed for {e.label}: {ex}")
            t = None
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
    # amendment 8 (2026-09-22): the open-vocabulary second detector, added as a UNION with its own
    # bar -- BEATs is deaf to sounds that speech or music masks, FlexSED is asked one label at a
    # time and hears them (docs/GOLD_RERUN_2026-09-22.md sec 10). Both bars are set on Adam's DEV
    # half; a clip with no cache falls back to BEATs alone.
    fbar = float(getattr(config, "FLEXSED_BAR", 0) or 0)
    if fbar > 0:
        from src.stage4_audio_event_detection import flexsed_infer as FX
        try:
            ffw, ftimes, flabels = FX.infer_flexsed(Path(wav_path), device)
            fev = _extract_events(ffw, ftimes, flabels, fbar, None, min_dur,
                                  low=fbar * float(getattr(config, "AED_HYSTERESIS", 1.0)))
            # a family both detectors report at the same moment keeps the earlier onset
            from src.labels import canonical
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
            events = events + fresh
        except FileNotFoundError as e:
            print(f"       [stage4] FlexSED cache missing ({e}); BEATs alone", flush=True)
    if backend == "BEATs" and getattr(config, "ONSET_CAM", True):
        events = _refine_onsets_cam(Path(wav_path), events, labels, device)
    cap = getattr(config, "MAX_SPAN", None)      # v4ab3/v4b3: a picture never stays longer than this (docs/prereg_v4.md)
    if cap:
        for e in events:
            e.end = min(e.end, e.start + float(cap))
    n_classes = len({e.label for e in events})
    print(f"       [stage4] {backend} SED: {len(events)} event span(s) over "
          f"{n_classes} class(es) (threshold={threshold}).")
    if plot_path is not None:
        plot_timeline(framewise, times, labels, Path(plot_path),
                      top_k=plot_top_k, threshold=threshold)
        print(f"       [stage4] timeline plot -> {plot_path}")
    return events
