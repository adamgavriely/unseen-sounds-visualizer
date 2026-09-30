"""Round 19 DR (docs/prereg_round13_detector_push.md): DASM as a third ear. P4 pool = DASM runs (family >= 0.575, gaps
<= 0.5 s merged) that no B0r stage-4 span of the same family touches within +-0.5 s; both open-inventory listeners (Qwen3-Omni
V4, Audio Flamingo Next V4; same prompt / decoding / matcher) answer each run's cut. No gold is read.

    python benchmark/gold/dasm_rescue.py pool NAME STAGE4_JSON DASM_DIR WAV_DIR OUT_JSON     # CPU
    python benchmark/gold/dasm_rescue.py listen OUT_JSON[:WAV_DIR] ...                        # GPU (Qwen, then AF)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import dev_candidates_check as C

BAR, GAP, PAD = 0.575, 0.5, 0.5


def runs(v, t, bar):
    on = v >= bar
    out, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            out.append([float(t[i]), float(t[j]) + (t[1] - t[0]), float(v[i:j + 1].max())])
            i = j + 1
        else:
            i += 1
    merged = []
    for r in out:
        if merged and r[0] - merged[-1][1] <= GAP:
            merged[-1][1] = r[1]; merged[-1][2] = max(merged[-1][2], r[2])
        else:
            merged.append(r)
    return merged


def pool(name, s4p, ddir, wavdir, outp):
    import soundfile as sf
    from src.labels import canonical
    s4 = json.loads(Path(s4p).read_text(encoding="utf-8"))["arms"]
    items = []
    for f in sorted(Path(ddir).glob("*.npz")):
        st = f.stem
        rows = s4.get("B0r|proposed", {}).get(st)
        if rows is None:
            continue
        rows = rows + s4.get("B0r|blind_a2i", {}).get(st, [])
        z = np.load(f, allow_pickle=True)
        fw, t, labs = z["fw"], z["times"], [str(x) for x in z["labels"]]
        dur = float(sf.info(str(Path(wavdir) / f"{st}.wav")).duration)
        fams = {}
        for i, l in enumerate(labs):
            fams.setdefault(canonical(l), []).append(i)
        for fam, cols in fams.items():
            v = fw[:, cols].max(axis=1)
            for a, b, pk in runs(v, t, BAR):
                if any(canonical(r["label"]) == fam and r["start"] <= b + PAD and r["end"] >= a - PAD for r in rows):
                    continue
                ra = [max(0.0, a - 1.0), min(dur, max(b + 1.0, a + 1.0))]
                items.append({"clip": st, "pool": "P4", "family": fam, "label": fam, "start": round(a, 3), "end": round(b, 3),
                              "peak": round(pk, 3), "run_audio": ra, "cut_start": ra[0], "cut_end": ra[1],
                              "run_len": round(b - a, 3), "null_family": fam})
    C.dump(Path(outp), {"_meta": {"split": name, "bar": BAR, "gap": GAP, "pad": PAD, "wav": str(wavdir)}, "items": items})
    print(f"[pool] {name}: {len(items)} P4 runs in {len({x['clip'] for x in items})} clips -> {outp}", flush=True)


def listen(specs):
    import soundfile as sf
    import torch
    from benchmark.gold import listener_variants as LV
    from benchmark.gold import listener_afnext as AF
    LV.S.load_gold = LV._no_gold
    jobs = []
    for sp in specs:
        outp = Path(sp)
        d = json.loads(outp.read_text(encoding="utf-8"))
        jobs.append((outp, d))
    # Qwen3-Omni V4
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from sentence_transformers import SentenceTransformer
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
    for outp, d in jobs:
        wavdir = Path(d["_meta"]["wav"])
        cache, t0 = {}, time.time()
        for it in d["items"]:
            if "qwen_v4_text" in it:
                continue
            st = it["clip"]
            if st not in cache:
                w, sr = sf.read(str(wavdir / f"{st}.wav"), dtype="float32")
                cache.clear(); cache[st] = w
            w = cache[st]
            i, j = int(it["run_audio"][0] * LV.SR), int(it["run_audio"][1] * LV.SR)
            txt = gen(w[i:max(j, i + LV.SR)])
            it["qwen_v4_text"] = txt
            m, ok = AF.v4_match(O, emb, txt, it["family"])
            it["qwen_v4_match"], it["qwen_v4"] = m, bool(ok)
        C.dump(outp, d)
        print(f"[qwen] {outp.name}: {len(d['items'])} items in {time.time() - t0:.0f} s", flush=True)
    del model
    torch.cuda.empty_cache()
    # Audio Flamingo Next V4 (listener_afnext.run, unchanged; its accept.V4 = AF's V4 on the item's family)
    M = AF.load_model()
    for outp, d in jobs:
        LV.SPLITS = {"p4": {"wav": Path(d["_meta"]["wav"])}}
        AF.run(M, d["items"], "p4", outp, d["_meta"])
    for outp, d in jobs:
        its = d["items"]
        both = sum(1 for x in its if x.get("qwen_v4") and (x.get("accept") or {}).get("V4"))
        print(f"[listen] {outp.name}: {len(its)} runs, Qwen V4 {sum(bool(x.get('qwen_v4')) for x in its)}, "
              f"AF V4 {sum(bool((x.get('accept') or {}).get('V4')) for x in its)}, both {both}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "pool":
        pool(*sys.argv[2:7])
    else:
        listen(sys.argv[2:])
