"""A second opinion on every detection: does CLAP hear what BEATs named?

Why. Reading the gate's cached decisions on the development split (2026-09-14), about
eighty of the hundred pictures the VLM could not silence were confident phantoms -- a
whale at a Christmas market (0.36), roaring cats at a helicopter arrival (0.51), an
electric toothbrush in an alarm clip (0.73), glass at 0.87 in a bombardment. The gate
correctly finds no visible source for them and the rubric charges a point each. The
place-plausibility veto that would have caught some was a hand rule and was 2 right / 2
wrong; this is model-based: a second audio model must agree, and it is asked about the
same two seconds BEATs heard.

How. CLAP scores the detection's 2 s window against all 527 AudioSet names ("the sound
of <name>"), the names are collapsed to FAMILIES through the ontology (max score per
family: 'Vehicle', 'Car' and 'Motor vehicle (road)' are one competitor, not three), and
the detection survives only if its own family (or one in its ontology subtree) is among CLAP's top-k
families for that window. k is set once on DCASE 2025 gold events as the smallest value
that keeps 95 % of gold recall, and must be at most 30 of the 328 families or the check
is declared useless (benchmark/calibrate_clap_k.py) -- the constant comes from gold, never from
the phantoms -- and then frozen (config.CLAP_TOP_K). A dropped detection does not exist
downstream: it is neither shown nor gated. (Design reviewed by Fable, 2026-09-14.)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

CLAP_MODEL = "laion/clap-htsat-fused"
CLAP_SR = 48000
PROMPT = "the sound of {}"

_STATE: dict = {}


def family(label: str) -> str:
    """The label's ancestor two below the ontology root, or the label itself when its
    chain is shorter: Bark -> Dog, Car -> Motor vehicle (road), Bird vocalization -> Bird,
    Whale vocalization -> itself, Fire alarm -> itself (Alarm is a root child)."""
    from src.labels import ancestors
    chain = [label] + list(ancestors(label))
    return chain[-3] if len(chain) >= 3 else chain[0]


def _audioset_names() -> List[str]:
    """The 527 AudioSet display names BEATs and PANNs both emit."""
    try:
        from panns_inference.config import labels
        return list(labels)
    except Exception:
        # every ontology name minus the abstract roots; a superset of the 527
        names = json.loads((Path(__file__).resolve().parents[1] / "audioset_mid_names.json")
                           .read_text("utf-8"))
        roots = {"Human sounds", "Animal", "Natural sounds", "Sounds of things", "Music",
                 "Source-ambiguous sounds", "Channel, environment and background"}
        return sorted(set(names.values()) - roots)


def _load(device: str):
    if "model" in _STATE:
        return _STATE
    import torch
    from transformers import ClapModel, ClapProcessor
    from src.text_similarity import _as_tensor
    proc = ClapProcessor.from_pretrained(CLAP_MODEL)
    mdl = ClapModel.from_pretrained(CLAP_MODEL).to(device).eval()
    names: List[str] = _audioset_names()
    fams = [family(n) for n in names]
    with torch.no_grad():
        embs = []
        for i in range(0, len(names), 64):
            t = proc(text=[PROMPT.format(n.lower()) for n in names[i:i + 64]],
                     return_tensors="pt", padding=True).to(device)
            e = _as_tensor(mdl.get_text_features(**t))
            embs.append(e / e.norm(dim=-1, keepdim=True))
        E = torch.cat(embs, 0)
    _STATE.update(proc=proc, model=mdl, names=names, families=fams, text=E, device=device,
                  as_tensor=_as_tensor)
    return _STATE


def unload():
    _STATE.clear()


MAX_WINDOW = 10.0   # CLAP-fused takes up to 10 s; a detection is scored on its own burst


def family_scores(audio: np.ndarray, sr: int, start: float, device: str = "cpu",
                  window: float = MAX_WINDOW) -> Dict[str, float]:
    """CLAP score per family for [start, start + window] of ``audio`` -- the detection's
    own span, capped at 10 s, rather than its first two seconds: CLAP was built for
    clips, and a sustained sound is what it recognises."""
    import librosa
    import torch
    st = _load(device)
    a = int(max(0.0, start) * sr)
    seg = audio[a:a + int(window * sr)]
    if len(seg) < sr // 2:
        return {}
    if sr != CLAP_SR:
        seg = librosa.resample(seg.astype(np.float32), orig_sr=sr, target_sr=CLAP_SR)
    with torch.no_grad():
        inp = st["proc"](audio=[seg], sampling_rate=CLAP_SR, return_tensors="pt").to(device)
        ea = st["as_tensor"](st["model"].get_audio_features(**inp))
        ea = ea / ea.norm(dim=-1, keepdim=True)
        sims = (ea @ st["text"].T)[0].cpu().numpy()
    out: Dict[str, float] = {}
    for f, s in zip(st["families"], sims):
        if s > out.get(f, -9.0):
            out[f] = float(s)
    return out


def _subtree_families(label: str) -> List[str]:
    """The label's own family plus the family of every ontology descendant: a parent
    label such as 'Vehicle' is heard by CLAP as one of its children ('Motor vehicle
    (road)', 'Rail transport'), and 'same family' means the subtree. Cached."""
    if label in _SUBTREE:
        return _SUBTREE[label]
    from src.labels import is_descendant
    st = _load_names_only()
    fams = {family(label)}
    for n in st:
        if is_descendant(n, label):
            fams.add(family(n))
    _SUBTREE[label] = sorted(fams)
    return _SUBTREE[label]


_SUBTREE: Dict[str, List[str]] = {}


def _load_names_only() -> List[str]:
    if "names" in _STATE:
        return _STATE["names"]
    return _audioset_names()


def family_rank(label: str, scores: Dict[str, float]) -> Optional[int]:
    """1 = CLAP's best family for the window; None when the window was too short.
    A label's rank is the best rank over the families of its ontology subtree."""
    if not scores:
        return None
    mine = max((scores[f] for f in _subtree_families(label) if f in scores), default=None)
    if mine is None:
        return len(scores) + 1
    return 1 + sum(1 for v in scores.values() if v > mine)


def agrees(label: str, audio: np.ndarray, sr: int, start: float, end: float, top_k: int,
           device: str = "cpu") -> Tuple[bool, Optional[int]]:
    r = family_rank(label, family_scores(audio, sr, start, device,
                                         window=min(MAX_WINDOW, max(2.0, end - start))))
    return (r is None or r <= top_k), r
