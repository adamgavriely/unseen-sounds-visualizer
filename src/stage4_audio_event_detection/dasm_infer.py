"""DASM (Detect Any Sound Model, ACM MM 2025) frame scorer with MGA-CLAP text queries: the scorer block of detector
round 6 (benchmark/detector_round6.py, release v1.2.0), moved here unchanged except the depth of _ROOT.

Used by benchmark/gold/dev_candidates_check.dasm() (and so by clip_prep.py's `dasm` step and src/listener_prep.py).
The text embeddings of the 215 depictable families are computed once (GPU) into QFILE:

    python src/stage4_audio_event_detection/dasm_infer.py      # -> data/work/dasm_text_queries.pt

DASM code: github.com/cai525/Transformer4SED (~/Transformer4SED, or T4SED_ROOT); MGA-CLAP: github.com/Ming-er/MGA-CLAP
(~/Transformer4SED/third_parties/MGA-CLAP); weights: HF CPF2/detect_any_sound (text_query), MGA-CLAP model.pt (official
Google Drive link), google-bert/bert-base-uncased (~/bert-base-uncased, or BERT_DIR). embed() and _Dasm import those
repos' own `src` / `models` / `tools` packages, so they take this project off sys.path first (_foreign): call them in a
process of their own, or as the last thing that needs this project's modules.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import types
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
VOCAB = _ROOT / "benchmark" / "gold" / "depictable_vocab.json"
QFILE = _ROOT / "data" / "work" / "dasm_text_queries.pt"
T4 = Path(os.environ.get("T4SED_ROOT", Path.home() / "Transformer4SED"))
MGA = T4 / "third_parties" / "MGA-CLAP"
BERT = Path(os.environ.get("BERT_DIR", Path.home() / "bert-base-uncased"))
DASM_W = T4 / "pretrained_model" / "detect_any_sound" / "text_query" / "as_full_text_query_best_model.pt"
DASM_CFG = T4 / "pretrained_model" / "detect_any_sound" / "text_query" / "config.yaml"
SR, CLIP_S, FPS, TEMP_W = 32000, 10, 50.0, 0.5
NB_QUERIES = ["alarm", "sirens", "air raid sirens", "dog", "music", "bird"]     # the official notebook's example list


def vocab():
    return json.loads(VOCAB.read_text(encoding="utf-8"))["families"]


def dasm_text(x):
    return "sound of " + x.lower()                     # the official notebook's template


# ----------------------------------------------------------------------------- foreign-repo imports
def _foreign(root: Path):
    """take this project off the import path and make `root` the repo whose `src`/`models`/`tools` packages are used"""
    for k in list(sys.modules):
        if k in ("src", "models", "tools") or k.startswith(("src.", "models.", "tools.")):
            del sys.modules[k]
    bad = {_ROOT.resolve(), (_ROOT / "benchmark").resolve()}
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() not in bad]
    sys.path.insert(0, str(root))
    os.chdir(root)


def _stub(name, **attrs):
    """an import-only dependency that the inference path never calls (wandb, loguru, ... in MGA-CLAP's tools/utils)"""
    try:
        importlib.import_module(name)
        return False
    except ImportError:
        pass
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        n = ".".join(parts[:i])
        if n not in sys.modules:
            sys.modules[n] = types.ModuleType(n)
        if i > 1:
            setattr(sys.modules[".".join(parts[:i - 1])], parts[i - 1], sys.modules[n])
    for k, v in attrs.items():
        setattr(sys.modules[name], k, v)
    return True


# ----------------------------------------------------------------------------- step 1: text embeddings (GPU)
def embed(device="cuda"):
    import torch
    import torch.nn.functional as F
    import yaml
    voc = vocab()
    assert len(voc) == 215
    queries = [dasm_text(x) for x in voc]
    nb = [dasm_text(x) for x in NB_QUERIES]
    _foreign(MGA)
    # the real transformers / accelerate chain is imported before the stubs exist: accelerate probes
    # importlib.util.find_spec("wandb") at import time and a stub has no __spec__
    import accelerate  # noqa: F401
    from transformers import BertModel, BertTokenizer  # noqa: F401
    stubs = [n for n, a in (("wandb", {}), ("loguru", {"logger": logging.getLogger("loguru")}),
                            ("sentence_transformers", {"util": types.SimpleNamespace()}), ("sed_scores_eval", {}),
                            ("sed_scores_eval.utils.scores", {"create_score_dataframe": None}), ("ruamel.yaml", {}))
             if _stub(n, **a)]
    print(f"[embed] import-only stubs: {stubs}", flush=True)
    from models import text_encoder as TE
    TE.MODELS[str(BERT)] = TE.MODELS["bert-base-uncased"]    # the same BertModel / BertTokenizer, read from a local copy
    from models.ase_model import ASE
    cfg = yaml.safe_load(open("settings/inference_sed.yaml", encoding="utf-8"))
    cfg["text_encoder_args"]["type"] = str(BERT)
    clap = ASE(cfg)
    sd = torch.load(str(MGA / "pretrained_models" / "models" / "model.pt"), map_location="cpu")["model"]
    res = clap.load_state_dict(sd, strict=False)                 # strict=False as the official notebook
    bad = [k for k in res.missing_keys if k.startswith(("text_encoder.", "word_proj.", "codebook"))]
    print(f"[embed] MGA-CLAP load: {len(res.missing_keys)} missing (text side: {bad}), {len(res.unexpected_keys)} unexpected "
          f"{res.unexpected_keys[:8]}", flush=True)
    assert not bad, "c1 failed: text side of MGA-CLAP not fully loaded"
    clap = clap.to(device).eval()

    def enc(q):
        with torch.no_grad():
            _, we, am = clap.encode_text([q])                    # one query at a time (prereg): no batch padding
            return F.normalize(clap.msc(we, clap.codebook, am), dim=-1)[0].float().cpu()
    E = torch.stack([enc(q) for q in queries])
    NB = torch.stack([enc(q) for q in nb])
    assert E.shape == (215, 1024), E.shape
    QFILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vocab": voc, "queries": queries, "embeds": E, "nb_queries": nb, "nb_embeds": NB,
                "mga_missing": list(res.missing_keys), "mga_unexpected": list(res.unexpected_keys), "stubs": stubs}, str(QFILE))
    print(f"[embed] {E.shape} -> {QFILE}; first queries {queries[:3]}", flush=True)


# ----------------------------------------------------------------------------- step 2: DASM frame scores (GPU)
class _Dasm:
    def __init__(self, device):
        import torch
        import yaml
        self.torch, self.device = torch, device
        _foreign(T4)
        import torch._dynamo  # noqa: F401  (probes find_spec("pandas") at import time; a stub has no __spec__)
        import timm, torchaudio, torchlibrosa  # noqa: F401,E401
        _stub("pandas")                                        # DASM's own encoder module imports it; not used for inference
        from src.models.detect_any_sound.detect_any_sound_htast import DASM_HTSAT
        from src.codec.encoder import Encoder
        from src.preprocess.feats_extraction import pad_wav, to_mono
        self.pad_wav, self.to_mono = pad_wav, to_mono
        cfg = yaml.safe_load(open(DASM_CFG, encoding="utf-8"))
        m = DASM_HTSAT(**cfg["DASM_HTSAT"]["init_kwargs"])      # the HTS-AT backbone is loaded strict=True inside
        res = m.load_state_dict(torch.load(str(DASM_W), map_location="cpu"), strict=True)
        self.load_info = {"missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys)}
        self.m = m.to(device).eval()
        f = cfg["feature"]
        self.enc = Encoder([], audio_len=f["audio_max_len"], frame_len=f["win_length"], frame_hop=f["hopsize"],
                           net_pooling=f["net_subsample"], sr=f["sr"])
        assert self.enc.sr == SR and self.enc.audio_len == CLIP_S and self.enc.n_frames == 500
        assert abs(self.enc.n_frames / CLIP_S - FPS) < 1e-9
        self.base = self.m.at_query                              # nn.Parameter [407, 1024] (not a ParameterList)
        assert isinstance(self.base, torch.nn.Parameter) and self.base.shape[0] == 407, type(self.base)
        self.extract = self.m.get_feature_extractor()

    def score(self, wav, qemb):
        """wav: mono float32 at 32 kHz; qemb [Q, 1024]. Returns (fw [Q, T] with T = frames before the pad, at_out [Q])"""
        torch = self.torch
        base = self.base.detach()
        query = torch.cat([base, qemb.to(base.device, base.dtype)]).to(self.device)
        nb = base.shape[0]
        att = torch.ones(query.shape[0], query.shape[0], dtype=torch.bool, device=self.device)
        att[:, :nb] = False
        att.fill_diagonal_(False)                               # the notebook's get_att_mask
        n = int(round(CLIP_S * SR))
        # clarification (prereg): up to 11 s the clip is cut to 10 s as the official waveform_modification does (a
        # 10.01-s clip would otherwise add a 10-ms piece whose one zero-padded frame scores ~0.17 for most families);
        # only a longer clip (none in the two sets) is scored in 10-s pieces
        pieces = [wav[i:i + n] for i in range(0, len(wav), n)] if len(wav) > n + SR else [wav[:n]]
        fws, ats = [], []
        for p in pieces:
            w, pad_mask = self.pad_wav(self.to_mono(p), n, self.enc)
            x = torch.from_numpy(w).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                mel = self.extract(x)
                strong, _weak, other = self.m(input=mel, temp_w=TEMP_W, pad_mask=pad_mask.unsqueeze(0).to(self.device),
                                              query=query, query_type="text", tgt_mask=att)
            keep = int((~pad_mask).sum())
            fws.append(strong[0, nb:, :keep].float().cpu().numpy())
            ats.append(other["at_out"][0, nb:].float().cpu().numpy())
        return np.concatenate(fws, axis=1), np.max(np.stack(ats), axis=0)


if __name__ == "__main__":
    embed()
