"""BEATs as the Stage 4 detector: the same 527 AudioSet labels as PANNs, better at them.

Why this exists. Adam went through eight demo clips and eleven of his fourteen
complaints were the detector: a siren labelled "truck horn", a car squeak "dog", a
stutter "hiccup", an engine rotor "printer", a fire alarm "whistle", a crying baby
"sheep", and a glass shatter missed outright. Stage 5 then drew the wrong label
faithfully. PANNs CNN14 is 2019 and scores 0.431 mAP on AudioSet; BEATs iter3+ (Chen
et al., Microsoft, 2022) scores 0.486 with the SAME label set, so everything downstream
-- the ontology deduplication, the families, the gate -- carries over untouched.
config.py has said "-> BEATs for quality later" since the first commit.

BEATs is a clip-level tagger, not a frame-level one, so time spans come from a sliding
window: every WINDOW seconds of audio scored as one clip, stepped by HOP. The result
is shaped exactly like PANNs' framewise output so ``_extract_events`` and the timeline
plot work unchanged; the only difference is the time resolution (HOP, not 10 ms).

Weights: the official checkpoints were published on a SharePoint link, which is not
reachable from the cluster, so they come from an ungated mirror on the Hugging Face
hub. The file is the one named in the BEATs README as the strongest release.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np

REPO = "WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2"
FILE = "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
SR = 16000
WINDOW = 2.0    # seconds scored as one clip
HOP = 0.25      # seconds between windows; the time resolution of the spans
# Where in the window a detection is stamped. Stamped at the centre (offset 1.0) the
# picture came up to a second BEFORE the sound; stamped at the end (offset 0) it came a
# second or two AFTER, because a sound has to fill enough of the window to score. Half a
# second from the end is the compromise measured on the demo sets; it is a number, not a
# principle, and a shorter window would shrink both errors at a cost in accuracy.
STAMP_OFFSET = 0.5

_MODEL = None   # (model, names, device)


def _load(device: str):
    global _MODEL
    if _MODEL is None or _MODEL[2] != device:
        import torch
        from huggingface_hub import hf_hub_download
        from .beats.BEATs import BEATs, BEATsConfig
        ckpt = torch.load(hf_hub_download(REPO, FILE), map_location="cpu",
                          weights_only=False)
        model = BEATs(BEATsConfig(ckpt["cfg"]))
        model.load_state_dict(ckpt["model"])
        model = model.to(device).eval()
        # the checkpoint maps output index -> AudioSet "mid"; the ontology maps mid ->
        # display name, which is the same string PANNs uses, so labels.py needs nothing
        mid_names = json.loads(
            (Path(__file__).resolve().parents[1] / "audioset_mid_names.json")
            .read_text("utf-8"))
        label_dict = ckpt["label_dict"]
        names = [mid_names.get(label_dict[i], label_dict[i]) for i in range(len(label_dict))]
        _MODEL = (model, names, device)
    return _MODEL


def infer_beats(wav_path: Path, device: str = "cpu",
                window: float = WINDOW, hop: float = HOP,
                batch: int = 32) -> Tuple[np.ndarray, np.ndarray, list]:
    """(framewise[windows, 527], times[windows], labels) -- PANNs-shaped."""
    import librosa
    import torch
    model, names, dev = _load(device)
    audio, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    n_win = int(round(window * SR))
    n_hop = int(round(hop * SR))
    # Lead-in padding, so the first windows end inside the first two seconds. Without it
    # the earliest possible stamp is the window length, and nothing in the first two
    # seconds of any clip could ever be detected -- barks and birdsong at the start of
    # two demo clips simply did not exist to the pipeline.
    lead = n_win - n_hop
    # Mirrored, not zeros: a window of silence with an abrupt onset reads to BEATs as a
    # transient -- the first half-second of a jungle clip scored "Vehicle 0.41" and drew
    # a truck horn. A reflection of the clip's own opening is plausible audio.
    audio = np.pad(audio, (lead, 0), mode="reflect" if len(audio) > lead else "constant")
    if len(audio) < n_win:
        audio = np.pad(audio, (0, n_win - len(audio)))
    starts = list(range(0, len(audio) - n_win + 1, n_hop))
    if not starts or starts[-1] + n_win < len(audio):
        starts.append(max(0, len(audio) - n_win))      # cover the tail
    probs = []
    with torch.no_grad():
        for b in range(0, len(starts), batch):
            chunk = np.stack([audio[s:s + n_win] for s in starts[b:b + batch]])
            x = torch.from_numpy(chunk).float().to(dev)
            mask = torch.zeros(x.shape, dtype=torch.bool, device=dev)
            p, _ = model.extract_features(x, padding_mask=mask)
            probs.append(p.float().cpu().numpy())
    framewise = np.concatenate(probs, axis=0)           # (windows, 527)
    # Stamped at the window END, not its centre. A score for [s, s+2] says the sound
    # is somewhere in those two seconds; stamping it at s+1 put a sound from the last
    # half of the window on screen up to a second before it happened, and Adam saw the
    # glass shatter before he heard it in three of five clips. Stamped at s+2 the onset
    # is never early -- at worst it is late by one hop -- and the tail lingers by up to
    # a window, which the display's minimum dwell was going to do anyway.
    times = np.array([(s + n_win - lead) / SR - STAMP_OFFSET for s in starts])
    keep = times >= 0.0
    return framewise[keep], times[keep], list(names)


# ---------------------------------------------------------------------------
# WHEN, from the model that decided WHAT: prefix-silencing occlusion.
#
# The first idea was a class activation map on BEATs' 160 ms tokens. It cannot work, and
# the reason is worth keeping: BEATs was fine-tuned only through the MEAN of its tokens,
# so nothing ever asked a token to be local, and after 12 full-attention layers every
# token carries roughly the clip logit. On three clips the CAM moved every onset by
# exactly 0 or exactly -1.5 s -- the fingerprint of a flat curve. (Fable, second opinion,
# 2026-09-14, on reading the code.)
#
# Occlusion is model-agnostic and immune to that. Take the first 2 s window that fired
# the class. Make 26 copies, silencing the first t seconds for t = 0, 0.08, ..., 2.0,
# and score them in one batched forward. Silencing audio BEFORE the onset changes
# nothing, so the class logit stays flat; once the cut passes the onset, evidence is
# removed and the logit falls. The onset is the first cut that removes a tenth of the
# evidence (measured in logit space, where mean pooling makes the fall roughly linear).
# Class-conditional, so a voice or a bark in the same window does not pull it.
CUT = 0.08           # seconds per occlusion step; 26 steps span the 2 s window
EVIDENCE_MIN = 1.0   # logit drop from full window to fully silenced, below which the
                     # window carries no evidence for the class and nothing is decided
ONSET_FRAC = 0.10    # onset = first cut removing this fraction of the evidence
RAMP_FRAC = 0.50     # a sound whose 50% point trails its 10% point by > RAMP_SEC is a ramp
RAMP_SEC = 0.5


def occlusion_onset(audio, sr, window_start: float, class_idx: int, device: str = "cpu",
                    window: float = WINDOW, debug: bool = False):
    """Where inside [window_start, window_start+window] does class `class_idx` begin?

    Returns (onset_time, ramp) or (None, False) when the window carries no evidence.
    The end is never touched: a picture lingering a little is cheap.
    """
    import numpy as np
    import torch
    model, _, dev = _load(device)
    a, b = int(round(window_start * sr)), int(round((window_start + window) * sr))
    seg = audio[max(0, a):b]
    if a < 0:
        seg = np.pad(seg, (-a, 0), mode="reflect" if len(seg) > -a else "constant")
    if len(seg) < b - a:
        seg = np.pad(seg, (0, b - a - len(seg)))
    n = len(seg)
    steps = int(round(window / CUT)) + 1
    fade = int(0.010 * sr)
    batch = np.zeros((steps, n), dtype=np.float32)
    for i in range(steps):
        cut = min(n, int(round(i * CUT * sr)))
        x = seg.copy()
        x[:cut] = 0.0
        if 0 < cut < n - fade:                      # 10 ms ramp at the cut, no click
            x[cut:cut + fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        batch[i] = x
    xb = torch.from_numpy(batch).to(dev)
    with torch.no_grad():
        mask = torch.zeros(xb.shape, dtype=torch.bool, device=dev)
        p, _ = model.extract_features(xb, padding_mask=mask)
    p = p[:, class_idx].float().clamp(1e-6, 1 - 1e-6).cpu().numpy()
    L = np.log(p / (1 - p))
    L0, Ls = L[0], L[-1]
    if L0 - Ls < EVIDENCE_MIN:
        return None, False
    Lm = np.array([np.median(L[max(0, i - 1):i + 2]) for i in range(len(L))])
    span = L0 - Ls
    below = Lm < L0 - ONSET_FRAC * span
    if not below.any():
        return None, False
    i = int(np.argmax(below))
    i50 = int(np.argmax(Lm < L0 - RAMP_FRAC * span)) if (Lm < L0 - RAMP_FRAC * span).any() else i
    ramp = (i50 - i) * CUT > RAMP_SEC
    onset = window_start if i <= 1 else window_start + (i - 0.5) * CUT
    if debug:
        return onset, ramp, L
    return onset, ramp


def unload() -> None:
    global _MODEL
    _MODEL = None
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
