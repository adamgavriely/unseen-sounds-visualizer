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
HOP = 0.5       # seconds between windows; the time resolution of the spans

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
    times = np.array([s / SR + window / 2 for s in starts])   # window centres
    return framewise, times, list(names)


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
