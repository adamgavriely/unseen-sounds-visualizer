"""Round 17 R1 (docs/prereg_round13_detector_push.md): Qwen3-Omni V4 open inventory on the P1 cuts (the drawn BEATs spans),
same prompt, decoding and cut as listener_variants.score's V4, plus the family parse of both listeners' V4 answers (Qwen and
Audio Flamingo Next): for every depictable family, the V4 matcher (word match on match_names, or mpnet cosine > COS). No gold.

    python benchmark/gold/listener_p1v4.py SPLIT:VCACHE:AFCACHE:WAVDIR:OUT [...]
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import listener_variants as LV
from benchmark.gold import dev_candidates_check as C


def lines_of(txt):
    ls = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip() for ln in (txt or "").splitlines()]
    return [ln for ln in ls if ln]


def main():
    import soundfile as sf
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from sentence_transformers import SentenceTransformer
    LV.S.load_gold = LV._no_gold
    proc = Qwen3OmniMoeProcessor.from_pretrained(LV.MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(LV.MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass
    emb = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    O = LV.Onto()
    fams = sorted(json.loads((_ROOT / "benchmark" / "gold" / "depictable_vocab.json").read_text(encoding="utf-8"))["families"])
    names = {f: LV.match_names(O, f) for f in fams}
    femb = emb.encode([f.lower() for f in fams], normalize_embeddings=True, convert_to_numpy=True)

    def gen(w):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": w}, {"type": "text", "text": LV.V4_Q}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[w], return_tensors="pt", padding=True, use_audio_in_video=False)
        inp = inp.to(model.thinker.device).to(torch.bfloat16)
        with torch.inference_mode():
            out = model.thinker.generate(**inp, max_new_tokens=LV.V4_NEW, do_sample=False)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def fam_parse(txt):
        ls = lines_of(txt)
        if not ls:
            return []
        e = emb.encode(ls, normalize_embeddings=True, convert_to_numpy=True)
        cos = e @ femb.T                                   # [lines, fams]
        got = []
        for li, ln in enumerate(ls):                       # Qwen's order: line by line
            for j, f in enumerate(fams):
                word = any(re.search(r"\b" + re.escape(nm) + r"(?:s|es)?\b", ln.lower()) for nm in names[f])
                if (word or cos[li, j] > LV.COS) and f not in got:
                    got.append(f)
        return got

    for spec in sys.argv[1:]:
        split, vc, afc, wavdir, outp = spec.split(":")
        items = [x for x in json.loads(Path(vc).read_text(encoding="utf-8"))["items"] if x.get("pool") == "P1"]
        af = {(x["clip"], x["pool"], x["family"], x.get("label"), round(x["start"], 2), round(x["end"], 2)): x
              for x in json.loads(Path(afc).read_text(encoding="utf-8"))["items"] if x.get("pool") == "P1"}
        out, cache, v4c = [], {}, {}
        t0 = time.time()
        for n, it in enumerate(sorted(items, key=lambda x: (x["clip"], x["run_audio"][0])), 1):
            st = it["clip"]
            if st not in cache:
                w, sr = sf.read(str(Path(wavdir) / f"{st}.wav"), dtype="float32")
                assert sr == LV.SR and w.ndim == 1
                cache.clear(); cache[st] = w
            w = cache[st]
            a, b = it["run_audio"]
            i, j = int(a * LV.SR), int(b * LV.SR)
            seg = w[i:max(j, i + LV.SR)]
            k = (st, round(a, 3), round(b, 3))
            if k not in v4c:
                v4c[k] = gen(seg)
            q = v4c[k]
            a_it = af.get((st, it["pool"], it["family"], it.get("label"), round(it["start"], 2), round(it["end"], 2)))
            at = (a_it or {}).get("afn_v4_text")
            out.append({"clip": st, "pool": "P1", "family": it["family"], "label": it.get("label"), "start": it["start"],
                        "end": it["end"], "run_audio": it["run_audio"], "qwen_v4_text": q, "qwen_fams": fam_parse(q),
                        "af_v4_text": at, "af_fams": fam_parse(at) if at is not None else None})
            if n % 50 == 0:
                print(f"[p1v4] {split} {n}/{len(items)} ({time.time() - t0:.0f} s)", flush=True)
        C.dump(Path(outp), {"_meta": {"split": split, "vcache": vc, "afcache": afc, "prompt": LV.V4_Q, "cos": LV.COS},
                            "items": out})
        print(f"[p1v4] {split}: {len(out)} P1 items -> {outp}", flush=True)


if __name__ == "__main__":
    main()
