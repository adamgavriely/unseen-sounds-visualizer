"""Detector round 10 GPU worker (docs/prereg_round10_rescue.md): FlexSED re-queries for R2 and R3.

  R2  the 2 paraphrases of every family (benchmark/round10_paraphrases.json), on the clip's original 16-kHz audio,
      all clips -> <WIN>/round10_para/<id>.npz   (fw [T, 430] float16, times, labels "family||p1" / "family||p2")
  R3  the 215 family queries (FlexSED's own template) on the 3 perturbed versions (pitch +1 / -1, 0.95x duration),
      only clips with a band candidate -> <WIN>/round10_view_<v>/<id>.npz  (fw [T', 215], times mapped back)

Standalone on purpose: the FlexSED repo has its own top-level `src` package, so this file imports nothing from the project
(as benchmark/gold/flexsed_run.py, whose model set-up it copies). The work list is written by
`detector_round10.py prep` (json: set, clips with the original wav, the view wavs, the output paths).

The scoring copies FlexSED's api.run_inference (text -> CLAP embedding one query at a time, 10-s chunks, the remainder
zero-padded to 1 s as flexsed_run.py, peak normalisation, sigmoid), except that the query text is given as is, so the
paraphrases are not wrapped in "The sound of ...". Checks (a failure stops the job):
  c1  the copy with the 215 families wrapped exactly as run_inference reproduces the cached FlexSED scores on this
      shard's first 2 clips (max abs diff < 5e-3)
  c2  one paraphrase alone vs inside its batch (first clip), max abs diff < 1e-3

    python benchmark/round10_flexsed.py --work <worklist.json> --shard i --of n
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
FLEXSED = Path(os.environ.get("FLEXSED_ROOT", Path.home() / "FlexSED"))
FPS = 25.0
C1_TOL, C2_TOL = 5e-3, 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--batch", type=int, default=24)
    a = ap.parse_args()
    W = json.loads(Path(a.work).read_text(encoding="utf-8"))
    fams = W["families"]
    para = W["paraphrases"]                                  # {family: {"p1":..., "p2":...}}
    ptexts, plabels = [], []
    for f in fams:
        for k in ("p1", "p2"):
            ptexts.append(para[f][k]); plabels.append(f"{f}||{k}")
    ftexts = [f"The sound of {f.replace('_', ' ').capitalize()}" for f in fams]     # = api.run_inference's wrapper
    clips = W["clips"][a.shard::max(1, a.of)]

    # ---- FlexSED set-up, as benchmark/gold/flexsed_run.py
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _ROOT]
    sys.modules.pop("src", None)
    local_clap = Path.home() / "clap-htsat-unfused-st"
    if local_clap.exists():
        import transformers
        _clap, _tok = transformers.ClapTextModelWithProjection, transformers.AutoTokenizer
        _from_clap, _from_tok = _clap.from_pretrained, _tok.from_pretrained
        _clap.from_pretrained = staticmethod(lambda name, *x, **k: _from_clap(str(local_clap) if "clap" in str(name) else name, *x, **k))
        _tok.from_pretrained = staticmethod(lambda name, *x, **k: _from_tok(str(local_clap) if "clap" in str(name) else name, *x, **k))
    sys.path.insert(0, str(FLEXSED))
    os.chdir(FLEXSED)
    import librosa
    import torch
    from api import FlexSED
    m = FlexSED(device="cuda")
    _split = m.split_audio_fixed

    def split(audio, sr, chunk_duration=10.0):
        out = []
        for c in _split(audio, sr, chunk_duration):
            if len(c) < sr:
                c = np.pad(c, (0, sr - len(c)))
            out.append(c)
        return out

    def embed(texts):
        """api.run_inference lines 61-71, with the text as given"""
        out = []
        with torch.no_grad():
            for t in texts:
                inputs = m.tokenizer([t], padding=True, return_tensors="pt")
                out.append(m.clap(**inputs).text_embeds.unsqueeze(1))
        return torch.cat(out, dim=1).to(m.device)             # [1, n, d]

    def score(wav, Q, batch=a.batch):
        """api.run_inference lines 73-94 over query batches -> [n, T] float32"""
        audio, sr = librosa.load(str(wav), sr=16000)
        chunks = split(audio, sr, 10)
        parts = []
        for k in range(0, Q.shape[1], batch):
            preds_list = []
            with torch.inference_mode():
                for chunk in chunks:
                    x = torch.tensor(np.array([chunk])).to(m.device)
                    x = x / (torch.max(torch.abs(x)) + 1e-9)
                    mel = m.model.forward_to_spec(x)
                    preds_list.append(torch.sigmoid(m.model(mel, Q[:, k:k + batch])).cpu())
            parts.append(torch.cat(preds_list, dim=2).squeeze(1).float().numpy())
            torch.cuda.empty_cache()
        return np.concatenate(parts, axis=0)

    QP, QF = embed(ptexts), embed(ftexts)
    log = {"shard": a.shard, "of": a.of, "c1": [], "c2": None}
    # ---- c2: query independence (first clip)
    if clips:
        w0 = clips[0]["wav"]
        alone = score(w0, QP[:, :1], batch=1)[0]
        inb = score(w0, QP[:, :a.batch])[0]
        d2 = float(np.abs(alone - inb).max())
        log["c2"] = d2
        print(f"[c2] one paraphrase alone vs in batch: max diff {d2:.2e}", flush=True)
        assert d2 < C2_TOL, "c2 failed: stop"
    n_new = 0
    for i, c in enumerate(clips):
        # ---- c1 on the first 2 clips of the shard
        if i < 2:
            z = np.load(c["flex_cache"], allow_pickle=False)
            ref = z["fw"].astype(np.float32)
            assert [str(s) for s in z["labels"]] == fams
            mine = score(c["wav"], QF)
            ok = mine.shape == ref.shape
            d1 = float(np.abs(mine - ref).max()) if ok else float("inf")
            log["c1"].append({"id": c["id"], "shape_ok": ok, "max_diff": d1})
            print(f"[c1] {c['id']} shape {mine.shape} vs {ref.shape}: max diff {d1:.2e}", flush=True)
            assert ok and d1 < C1_TOL, "c1 failed: stop"
        dst = Path(c["para_out"])
        if not dst.exists():
            fw = score(c["wav"], QP)                          # [430, T]
            T = fw.shape[1]
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dst, fw=fw.T.astype(np.float16), times=(np.arange(T) / FPS).astype(np.float64),
                                labels=np.array(plabels))
            n_new += 1
        for v, spec in c.get("views", {}).items():
            dst = Path(spec["out"])
            if dst.exists():
                continue
            fw = score(spec["wav"], QF)                       # [215, T']
            T = fw.shape[1]
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dst, fw=fw.T.astype(np.float16),
                                times=(np.arange(T) / FPS / float(spec["time_scale"])).astype(np.float64),
                                labels=np.array(fams))
        if (i + 1) % 10 == 0 or i + 1 == len(clips):
            print(f"[{i + 1}/{len(clips)}] {c['id']}", flush=True)
    lp = Path(W["log_dir"]) / f"round10_gpu_{W['set']}_{a.shard}of{a.of}.json"
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"done {W['set']} shard {a.shard}/{a.of}: {len(clips)} clips, {n_new} new paraphrase caches", flush=True)


if __name__ == "__main__":
    main()
