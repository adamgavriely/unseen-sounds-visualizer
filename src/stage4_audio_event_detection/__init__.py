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


def _extract_events(framewise, times, labels, threshold, top_k, min_dur) -> List[AudioEvent]:
    """Turn framewise probabilities into contiguous (label, start, end) spans."""
    peaks = framewise.max(axis=0)
    classes = np.where(peaks >= threshold)[0]
    if top_k:
        classes = classes[np.argsort(peaks[classes])[::-1][:top_k]]
    dt = (times[1] - times[0]) if len(times) > 1 else 1.0 / _PANNS_FPS
    events: List[AudioEvent] = []
    for c in classes:
        active = framewise[:, c] >= threshold
        i, n = 0, len(active)
        while i < n:
            if not active[i]:
                i += 1
                continue
            j = i
            while j < n and active[j]:
                j += 1
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


def detect_events(wav_path: Path, threshold: float = 0.2, top_k: int = None,
                  min_dur: float = 0.2, device: str = "cpu", model: str = "",
                  plot_path: Path = None, plot_top_k: int = 15) -> List[AudioEvent]:
    # "beats" (default, see config.AED_MODEL) or anything else for PANNs. Same 527
    # labels either way, so nothing downstream cares which one ran.
    backend = "BEATs" if "beats" in (model or "").lower() else "PANNs"
    try:
        if backend == "BEATs":
            from src.stage4_audio_event_detection.beats_infer import infer_beats
            framewise, times, labels = infer_beats(Path(wav_path), device)
        else:
            framewise, times, labels = _infer(Path(wav_path), device)
    except Exception as e:  # missing package/checkpoint -> keep the pipeline runnable
        print(f"       [stage4] {backend} unavailable ({type(e).__name__}: {e}); "
              f"returning no events.")
        return []

    events = _extract_events(framewise, times, labels, threshold, top_k, min_dur)
    n_classes = len({e.label for e in events})
    print(f"       [stage4] {backend} SED: {len(events)} event span(s) over "
          f"{n_classes} class(es) (threshold={threshold}).")
    if plot_path is not None:
        plot_timeline(framewise, times, labels, Path(plot_path),
                      top_k=plot_top_k, threshold=threshold)
        print(f"       [stage4] timeline plot -> {plot_path}")
    return events
