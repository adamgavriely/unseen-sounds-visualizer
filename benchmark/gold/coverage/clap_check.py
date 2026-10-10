"""Fable idea 1, first check: for needed sounds (importance 2-3) that NO detector candidate overlaps, does a zero-shot
text-queried audio model (LAION CLAP, laion/clap-htsat-unfused) put the gold family in its top 3 / top 10 over the
depictable vocabulary, on the 2-s cut starting 0.5 s before the onset? Cluster GPU/CPU from ~/wt_slice.
    python benchmark/gold/coverage/clap_check.py -> clap_check.md"""
import json, sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import stems, GOLD
from src.labels import canonical

HERE = Path(__file__).resolve().parent
WAV = Path.home() / "MscProj" / "data" / "work" / "gold_wav_flat"
VOCAB = Path.home() / "MscProj_r13" / "benchmark" / "gold" / "depictable_vocab.json"
SNAP = str(Path.home() / ".cache/huggingface/hub/models--laion--clap-htsat-unfused/snapshots/79b58ed25fc00386262a2bea4b19fd21dc4310a0")


def main():
    import torch, soundfile as sf, librosa
    from transformers import ClapModel, ClapProcessor
    gold = S.load_gold([GOLD]); allc = stems("dev") + stems("test")
    s = (_ROOT / "docs" / "decision_trail" / "data.js").read_text(encoding="utf-8")
    dj = json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))
    cands = {c["clip"]: c["cands"] for c in dj["clips"]}
    items = [(st, g) for st in allc for g in gold[st] if g["needed"] and g["importance"] >= 2
             and not any(S.same_family(c["label"], g["label"]) and c["end"] > g["start"] - 0.5 and c["start"] < g["end"] + 0.5 for c in cands[st])]
    fams = sorted(json.loads(VOCAB.read_text())["families"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = ClapModel.from_pretrained(SNAP, use_safetensors=True).to(dev).eval(); p = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    with torch.no_grad():
        te = m.get_text_features(**p(text=[f"the sound of {f.lower()}" for f in fams], return_tensors="pt", padding=True).to(dev)); te = te if torch.is_tensor(te) else te.pooler_output
        te = torch.nn.functional.normalize(te, dim=-1)
    L = ["# Zero-shot CLAP on needed sounds no detector heard", "", f"{len(items)} sounds; vocabulary {len(fams)} families", "",
         "| clip | gold | rank of the gold family | top 3 |", "|---|---|---|---|"]
    top3 = top10 = 0
    for st, g in items:
        x, sr = sf.read(str(WAV / f"{st}.wav"), dtype="float32", always_2d=True); x = x.mean(1)
        x = librosa.resample(x, orig_sr=sr, target_sr=48000)
        a = max(0.0, g["start"] - 0.5); seg = x[int(a * 48000):int((a + 2.0) * 48000)]
        with torch.no_grad():
            ae = m.get_audio_features(**p(audio=[seg], sampling_rate=48000, return_tensors="pt").to(dev)); ae = ae if torch.is_tensor(ae) else ae.pooler_output
            sim = (torch.nn.functional.normalize(ae, dim=-1) @ te.T)[0].cpu().numpy()
        order = list(np.argsort(-sim))
        ranks = [order.index(i) + 1 for i, f in enumerate(fams) if S.same_family(f, g["label"])]
        r = min(ranks) if ranks else None
        top3 += r is not None and r <= 3; top10 += r is not None and r <= 10
        L.append(f"| {st} | {g['label']} | {r} | {', '.join(fams[i] for i in order[:3])} |")
    L += ["", f"gold family in top 3: {top3} / {len(items)}; in top 10: {top10} / {len(items)}"]
    (HERE / "clap_check.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
