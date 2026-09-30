"""Round 29 V4D (docs/prereg_round13_detector_push.md): the unchanged V4 prompt on P2 / PV cuts whose loud frames (RMS above
the cut's 85th percentile, 20-ms frames) are ducked by 20 dB with 20-ms fades, same model / cut / greedy decoding / 64 tokens / V4 matcher as listener_variants V4. Output keeps the item keys and adds
`v4b_text` and accept {"V4B": bool}. No gold.

    python benchmark/gold/listener_v4b.py VCACHE:WAVDIR:OUT [...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import listener_variants as LV
from benchmark.gold import listener_afnext as AF
from benchmark.gold import dev_candidates_check as C

V4B_Q = None                                             # the prompt is the unchanged V4_Q


def duck(w, sr=16000, q=85, db=20.0):
    import numpy as np
    n = int(0.02 * sr)
    if len(w) < 2 * n:
        return w
    k = len(w) // n
    rms = np.sqrt((w[:k * n].reshape(k, n) ** 2).mean(axis=1) + 1e-12)
    loud = rms > np.percentile(rms, q)
    g = np.where(loud, 10 ** (-db / 20), 1.0)
    gain = np.repeat(g, n)
    gain = np.concatenate([gain, np.full(len(w) - len(gain), gain[-1])])
    ker = np.ones(n) / n
    gain = np.convolve(gain, ker, mode="same")               # 20-ms fades
    return (w * gain).astype(np.float32)


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

    def gen(w):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": w}, {"type": "text", "text": LV.V4_Q}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[w], return_tensors="pt", padding=True, use_audio_in_video=False)
        inp = inp.to(model.thinker.device).to(torch.bfloat16)
        with torch.inference_mode():
            out = model.thinker.generate(**inp, max_new_tokens=LV.V4_NEW, do_sample=False)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    for spec in sys.argv[1:]:
        vc, wavdir, outp = spec.split(":")
        items = [x for x in json.loads(Path(vc).read_text(encoding="utf-8"))["items"] if x.get("pool") in ("P2", "PV")]
        out, cache, gc, t0 = [], {}, {}, time.time()
        for it in sorted(items, key=lambda x: (x["clip"], x["run_audio"][0])):
            st = it["clip"]
            if st not in cache:
                w, sr = sf.read(str(Path(wavdir) / f"{st}.wav"), dtype="float32")
                cache.clear(); cache[st] = w
            w = cache[st]
            k = (st, round(it["run_audio"][0], 3), round(it["run_audio"][1], 3))
            if k not in gc:
                i, j = int(it["run_audio"][0] * LV.SR), int(it["run_audio"][1] * LV.SR)
                gc[k] = gen(duck(w[i:max(j, i + LV.SR)]))
            m, ok = AF.v4_match(O, emb, gc[k], it["family"])
            out.append({kk: it[kk] for kk in ("clip", "pool", "family", "label", "start", "end", "run_audio") if kk in it}
                       | {"peak": it.get("peak"), "v4d_text": gc[k], "v4d_match": m, "accept": {"V4D": bool(ok)}})
        C.dump(Path(outp), {"_meta": {"prompt": "V4_Q on ducked audio", "vcache": vc}, "items": out})
        print(f"[v4d] {outp}: {len(out)} items, V4D accepts {sum(x['accept']['V4D'] for x in out)}, "
              f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
