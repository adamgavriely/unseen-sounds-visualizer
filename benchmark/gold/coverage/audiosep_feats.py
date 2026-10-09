"""Step 14 (PREREG_step14_audiosep_verifier.md): AudioSep target separation of each DEV burst's family; three features.
Cluster GPU, env ~/venvs/audiosep, run from ~/AudioSep (its modules), items from ~/MscProj_tg/benchmark/gold/coverage.

    python ~/MscProj_tg/benchmark/gold/coverage/audiosep_feats.py   -> ~/MscProj_tg/scratch_cov/audiosep_dev.json (resumes)
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "AudioSep"))
from pipeline import build_audiosep

TG = Path.home() / "MscProj_tg"
ITEMS = TG / "benchmark" / "gold" / "coverage" / "verify_items_dev.json"
OUT = TG / "scratch_cov" / "audiosep_dev.json"
SR = 32000


def clip_audio(path):
    with tempfile.TemporaryDirectory() as td:
        w = Path(td) / "a.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), str(w)], check=True)
        import soundfile as sf
        x, _ = sf.read(str(w), dtype="float32")
    return x


def main():
    it = json.loads(ITEMS.read_text(encoding="utf-8"))
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    dev = "cuda"
    m = build_audiosep("config/audiosep_base.yaml", "checkpoint/audiosep_base_4M_steps.ckpt", dev)
    qe = m.query_encoder
    cache = {}
    tcache = {}
    for n, b in enumerate(it["bursts"]):
        if b["id"] in done:
            continue
        st = b["clip"]
        if st not in cache:
            cache.clear(); cache[st] = clip_audio(TG / it["videos"][st])
        x = cache[st]
        a, e = b["cut1"]
        i0, i1 = int(a * SR), max(int(e * SR), int(a * SR) + SR)
        mix = x[i0:i1]
        if len(mix) < SR // 2:
            done[b["id"]] = {"error": "short"}; continue
        text = b["family"].lower()
        with torch.no_grad():
            if text not in tcache:
                tcache[text] = qe.get_query_embed(modality="text", text=[text], device=dev)
            cond = tcache[text]
            stem = m.ss_model({"mixture": torch.tensor(mix)[None, None, :].to(dev), "condition": cond})["waveform"]
            stem = stem.squeeze().float().cpu().numpy()[: len(mix)]
        res = mix[: len(stem)] - stem
        s0, s1 = int((b["start"] - a) * SR), int((b["end"] - a) * SR)
        s0, s1 = max(0, s0), max(s0 + SR // 10, min(len(stem), s1))
        en = lambda v: float((v ** 2).sum()) + 1e-9
        inside_stem, inside_mix = stem[s0:s1], mix[s0:s1]
        outside_stem = np.concatenate([stem[:s0], stem[s1:]])
        f1 = en(inside_stem) / en(inside_mix)
        f2 = (en(inside_stem) / max(1, len(inside_stem))) / (en(outside_stem) / max(1, len(outside_stem)) if len(outside_stem) else 1e-9)
        with torch.no_grad():
            pair = torch.tensor(np.stack([inside_stem, res[s0:s1]]))
            emb = qe.get_query_embed(modality="audio", audio=pair.to(dev), device=dev)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            te = torch.nn.functional.normalize(cond, dim=-1)
            cs = (emb @ te[0]).cpu().numpy()
        done[b["id"]] = {"f1_frac": f1, "f2_in_out": float(f2), "f3_clap_diff": float(cs[0] - cs[1]),
                         "clap_stem": float(cs[0]), "clap_res": float(cs[1])}
        if n % 50 == 0:
            OUT.write_text(json.dumps(done))
            print(n, len(it["bursts"]), flush=True)
    OUT.write_text(json.dumps(done))
    print("AUDIOSEP_DONE", len(done))


if __name__ == "__main__":
    main()
