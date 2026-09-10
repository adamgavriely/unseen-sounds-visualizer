"""Text-embedding similarity, used wherever the pipeline needs "do these two phrases
mean the same thing?".

It replaces two string-matching heuristics that could not generalise:

* deduplication used to compare depiction strings for equality, so *Laughter*,
  *Snicker* and *Chuckle* each took a slot of their own with three near-identical
  pictures beside them;
* validation used to ask the VLM a yes/no question about whether a phrase depicted a
  sound, and the VLM approved "Man on motorcycle drives past silver van" for
  *Laughter*.

A synonym table would fix the clips we have already looked at and nothing else. An
embedding generalises: it scores an unseen sound the same way it scores a familiar
one, which is the requirement -- the system has to work on future videos, not on the
three demos on the desk.

SigLIP's text tower is the encoder because it is already a dependency (Stage 2) and
because it was trained on exactly this kind of string: a short phrase describing what
is in a picture. Our depictions are short phrases describing what is in a picture.
"""
from __future__ import annotations

from typing import List, Sequence

_ENC = None
_MODEL = "google/siglip-base-patch16-224"


def _encoder(model: str = "", device: str = "cpu"):
    global _ENC
    if _ENC is None:
        from transformers import AutoModel, AutoTokenizer
        repo = model or _MODEL
        tok = AutoTokenizer.from_pretrained(repo)
        mdl = AutoModel.from_pretrained(repo).to(device).eval()
        _ENC = (mdl, tok, device)
    return _ENC


def unload() -> None:
    global _ENC
    if _ENC is None:
        return
    _ENC = None
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _as_tensor(out):
    """The embedding tensor, whichever shape of return value the library used.

    ``get_text_features`` returns a bare tensor on transformers 4.x and a
    ``BaseModelOutputWithPooling`` on 5.x. The cluster runs 5.16 and this machine runs
    4.56, and the 5.x path failed with ``'BaseModelOutputWithPooling' object has no
    attribute 'norm'`` -- caught only because every similarity came back as exactly
    0.000 in a job log. Both are supported rather than pinning a version, because the
    difference is a return type and not a behaviour.
    """
    import torch
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("pooler_output", "text_embeds", "last_hidden_state"):
        val = getattr(out, attr, None)
        if isinstance(val, torch.Tensor):
            return val if val.dim() == 2 else val[:, 0]
    if isinstance(out, (tuple, list)) and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError("no embedding tensor in " + type(out).__name__)


def embed(texts: Sequence[str], model: str = "", device: str = "cpu"):
    """L2-normalised text embeddings, one row per input. Returns a torch tensor."""
    import torch
    mdl, tok, dev = _encoder(model, device)
    batch = tok([t or " " for t in texts], padding="max_length", truncation=True,
                max_length=64, return_tensors="pt").to(dev)
    with torch.no_grad():
        feats = _as_tensor(mdl.get_text_features(**batch)).float()
    return feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def similarity(a: Sequence[str], b: Sequence[str], model: str = "",
               device: str = "cpu") -> List[List[float]]:
    """Cosine similarity matrix between every phrase in `a` and every phrase in `b`."""
    ea, eb = embed(list(a) + list(b), model, device).split([len(a), len(b)])
    return (ea @ eb.T).tolist()


def pairwise(texts: Sequence[str], model: str = "", device: str = "cpu"):
    """Cosine similarity of every phrase against every other."""
    return similarity(texts, texts, model, device)
