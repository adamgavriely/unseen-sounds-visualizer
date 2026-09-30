"""Round 31 PTC (docs/prereg_round13_detector_push.md, "Round 31 PTC"): peak-tight cut for both ears. For every P2 / PV
candidate whose listener cut (run_audio) is longer than PTC_MIN_CUT s, Qwen3-Omni V4 and Audio Flamingo Next V4 (same prompt,
greedy, 64 tokens, same matcher as listener_variants / listener_afnext) are asked again on a PTC_WIN-s window centred on the
family's FlexSED peak frame inside [start, end] (data/work/flexsed_cache/<clip>.npz, the item's own query column; no column
-> the span midpoint; clipped to the clip and shifted to keep the window length). Output keeps the item keys and adds
`ptc_win`, `ptc_v4_text`, `ptc_afn_text` and accept {"PTC_V4": bool, "PTC_AF": bool}. No gold.

    python benchmark/gold/listener_ptc.py VCACHE:WAVDIR:OUT [...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import listener_variants as LV
from benchmark.gold import listener_afnext as AF
from benchmark.gold import dev_candidates_check as C

PTC_MIN_CUT, PTC_WIN = 4.0, 3.0
FLEX_DIR = C.FLEX_DIR


def window(it, flex, dur):
    """(a, b) of the tight window: centred on the item's own FlexSED query peak inside [start, end], else the midpoint"""
    a, b = float(it["start"]), float(it["end"])
    c = None
    if flex is not None:
        fw, ts, labs = flex
        if it["label"] in labs:
            col = labs.index(it["label"])
            dt = float(ts[1] - ts[0]) if len(ts) > 1 else 0.0
            m = (ts + dt > a) & (ts < b)
            if m.any():
                idx = np.flatnonzero(m)
                k = idx[int(np.argmax(fw[idx, col]))]
                c = float(ts[k] + dt / 2)
    if c is None:
        c = (a + b) / 2
    wa = max(0.0, c - PTC_WIN / 2)
    wb = wa + PTC_WIN
    if wb > dur:
        wb = dur
        wa = max(0.0, wb - PTC_WIN)
    return wa, wb


def select(vc):
    items = [x for x in json.loads(Path(vc).read_text(encoding="utf-8"))["items"] if x.get("pool") in ("P2", "PV")]
    return [x for x in items if float(x["run_audio"][1]) - float(x["run_audio"][0]) > PTC_MIN_CUT]


def main():
    import soundfile as sf
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from sentence_transformers import SentenceTransformer
    LV.S.load_gold = LV._no_gold
    specs = [s.split(":") for s in sys.argv[1:]]
    O = LV.Onto()
    work = {}
    for vc, wavdir, outp in specs:
        items = select(vc)
        flex, wavs = {}, {}
        out = []
        for it in sorted(items, key=lambda x: (x["clip"], x["start"])):
            st = it["clip"]
            if st not in wavs:
                w, sr = sf.read(str(Path(wavdir) / f"{st}.wav"), dtype="float32")
                assert sr == LV.SR and w.ndim == 1, (st, sr, w.shape)
                wavs[st] = w
                p = FLEX_DIR / f"{st}.npz"
                flex[st] = C.load_fr(p) if p.exists() else None
            wa, wb = window(it, flex[st], len(wavs[st]) / LV.SR)
            out.append({kk: it[kk] for kk in ("clip", "pool", "family", "label", "start", "end", "run_audio") if kk in it}
                       | {"peak": it.get("peak"), "ptc_win": [round(wa, 3), round(wb, 3)],
                          "ptc_col": bool(flex[st] is not None and it["label"] in flex[st][2])})
        work[outp] = (vc, wavdir, out, wavs)
        print(f"[ptc] {vc}: {len(items)} long-cut P2/PV items of {len(json.loads(Path(vc).read_text(encoding='utf-8'))['items'])}",
              flush=True)

    # --- ear 1: Qwen3-Omni V4 ---
    t0 = time.time()
    proc = Qwen3OmniMoeProcessor.from_pretrained(LV.MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(LV.MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass
    emb = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    print(f"[ptc] Qwen loaded in {time.time() - t0:.0f} s", flush=True)

    def qgen(w):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": w}, {"type": "text", "text": LV.V4_Q}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[w], return_tensors="pt", padding=True, use_audio_in_video=False)
        inp = inp.to(model.thinker.device).to(torch.bfloat16)
        with torch.inference_mode():
            o = model.thinker.generate(**inp, max_new_tokens=LV.V4_NEW, do_sample=False)
        return proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    for outp, (vc, wavdir, out, wavs) in work.items():
        gc, t1 = {}, time.time()
        for it in out:
            k = (it["clip"], *it["ptc_win"])
            if k not in gc:
                w = wavs[it["clip"]]
                gc[k] = qgen(w[int(it["ptc_win"][0] * LV.SR):int(it["ptc_win"][1] * LV.SR)])
            it["ptc_v4_text"] = gc[k]
            m, ok = AF.v4_match(O, emb, gc[k], it["family"])
            it["ptc_v4_x"] = m
            it["accept"] = {"PTC_V4": bool(ok)}
        C.dump(Path(outp), {"_meta": {"round": "31 PTC", "min_cut": PTC_MIN_CUT, "win": PTC_WIN, "prompt": LV.V4_Q,
                                      "vcache": vc, "ears": ["qwen"]}, "items": out})
        print(f"[ptc] qwen {outp}: {len(out)} items ({len(gc)} windows), PTC_V4 accepts "
              f"{sum(x['accept']['PTC_V4'] for x in out)}, {time.time() - t1:.0f} s", flush=True)
    del model, proc
    torch.cuda.empty_cache()

    # --- ear 2: Audio Flamingo Next V4 ---
    M = AF.load_model()
    for outp, (vc, wavdir, out, wavs) in work.items():
        gc, t1 = {}, time.time()
        for it in out:
            k = (it["clip"], *it["ptc_win"])
            if k not in gc:
                w = wavs[it["clip"]]
                gc[k] = M["gen"](w[int(it["ptc_win"][0] * LV.SR):int(it["ptc_win"][1] * LV.SR)], LV.V4_Q, LV.V4_NEW)
            it["ptc_afn_text"] = gc[k]
            m, ok = AF.v4_match(O, M["emb"], gc[k], it["family"])
            it["ptc_afn_x"] = m
            it["accept"]["PTC_AF"] = bool(ok)
        C.dump(Path(outp), {"_meta": {"round": "31 PTC", "min_cut": PTC_MIN_CUT, "win": PTC_WIN, "prompt": LV.V4_Q,
                                      "vcache": vc, "ears": ["qwen", "afn"]}, "items": out})
        print(f"[ptc] afn {outp}: PTC_AF accepts {sum(x['accept']['PTC_AF'] for x in out)}, {time.time() - t1:.0f} s",
              flush=True)
    print("DONE ptc", flush=True)


if __name__ == "__main__":
    main()
