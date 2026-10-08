"""Step 10 (PREREG_step10_v4fix.md): re-ask the Qwen3-Omni V4 open-inventory question with fixed decoding and rebuild the
three caches that read it. Cluster GPU, run from ~/MscProj_r13 (old module names), env msproj.

Bug: listener_variants.gen / listener_p1v4.gen / dasm_rescue.gen call thinker.generate without an end-of-turn stop, so
after the first 1-2 names the model runs on ("...\\nAssistant\\nMechanical hiss\\nAssistant..." or "with\\nwith...") until
64 tokens; the open list is then 1-2 items. Fix: eos_token_id = [<|im_end|>, <|endoftext|>], repetition_penalty 1.1,
same prompt, same 64 tokens, greedy; repeated lines and role words ("assistant", "user") removed.

    python ~/MscProj_tg/benchmark/gold/coverage/v4fix.py count            # CPU: degenerate answers in the old caches
    python ~/MscProj_tg/benchmark/gold/coverage/v4fix.py run dev dev2      # GPU -> ~/MscProj_tg/scratch_cov/v4fix/<split>/
"""
import copy
import json
import re
import sys
import time
from pathlib import Path

_ROOT = Path.cwd()
sys.path.insert(0, str(_ROOT))
OUTROOT = Path.home() / "MscProj_tg" / "scratch_cov" / "v4fix"
G_R13 = Path.home() / "MscProj_r13" / "benchmark" / "gold"
G_TG = Path.home() / "MscProj_tg" / "benchmark" / "gold"
G_MAIN = Path.home() / "MscProj" / "benchmark" / "gold"
SPLITS = {
    "dev": {"v": G_R13 / "dev_listener_v.json", "p1v4": G_R13 / "dev_listener_p1v4.json", "p4": G_R13 / "dev_listener_p4.json",
            "wav": Path.home() / "MscProj" / "data" / "work" / "devcand" / "wav16"},
    "dev2": {"v": G_TG / "dev2_listener_v.json", "p1v4": G_TG / "dev2_listener_p1v4.json", "p4": G_TG / "dev2_listener_p4.json",
             "wav": Path.home() / "MscProj" / "data" / "work" / "r13dev2" / "wav16"},
    "test": {"v": G_MAIN / "test_listener_v.json", "p1v4": G_MAIN / "test_listener_p1v4.json", "p4": G_MAIN / "test_listener_p4.json",
             "wav": Path.home() / "MscProj" / "data" / "work" / "r13test" / "wav16"},
    "test2": {"v": G_TG / "test2_listener_v.json", "p1v4": G_TG / "test2_listener_p1v4.json", "p4": G_TG / "test2_listener_p4.json",
              "wav": Path.home() / "MscProj" / "data" / "work" / "r13test2" / "wav16"},
}
ROLE = re.compile(r"^\s*(assistant|user|system)\s*:?\s*$", re.I)


def degenerate(t):
    ls = [l.strip() for l in (t or "").splitlines() if l.strip()]
    return bool(ls) and (any(ROLE.match(l) for l in ls) or len(set(ls)) < len(ls) / 2)


def clean(t):
    out = []
    for l in (t or "").splitlines():
        s = l.strip()
        if not s or ROLE.match(s) or s.lower() in (x.lower() for x in out):
            continue
        out.append(s)
    return "\n".join(out)


def count():
    for sp, c in SPLITS.items():
        for k in ("v", "p1v4", "p4"):
            if not c[k].exists():
                print(sp, k, "missing"); continue
            it = json.loads(c[k].read_text(encoding="utf-8"))["items"]
            key = "v4_text" if k == "v" else "qwen_v4_text"
            tx = [x[key] for x in it if key in x]
            print(f"{sp:5s} {k:5s} items {len(it):5d}  Qwen V4 answers {len(tx):5d}  degenerate {sum(map(degenerate, tx)):5d}")


def run(splits):
    import soundfile as sf
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from sentence_transformers import SentenceTransformer
    from benchmark.gold import listener_variants as LV
    from benchmark.gold import listener_afnext as AF
    LV.S.load_gold = LV._no_gold
    proc = Qwen3OmniMoeProcessor.from_pretrained(LV.MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(LV.MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass
    tok = proc.tokenizer
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    emb = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    O = LV.Onto()
    fams_all = sorted(json.loads((G_R13 / "depictable_vocab.json").read_text(encoding="utf-8"))["families"])
    names_all = {f: LV.match_names(O, f) for f in fams_all}
    femb = emb.encode([f.lower() for f in fams_all], normalize_embeddings=True, convert_to_numpy=True)

    def gen(w):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": w}, {"type": "text", "text": LV.V4_Q}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[w], return_tensors="pt", padding=True, use_audio_in_video=False)
        inp = inp.to(model.thinker.device).to(torch.bfloat16)
        with torch.inference_mode():
            out = model.thinker.generate(**inp, max_new_tokens=LV.V4_NEW, do_sample=False, eos_token_id=eos,
                                         pad_token_id=eos[-1], repetition_penalty=1.1)
        return clean(proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip())

    def lines_of(txt):
        ls = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip() for ln in (txt or "").splitlines()]
        return [ln for ln in ls if ln]

    def v_match(txt, fam):                               # listener_variants.score's V4 block, unchanged
        lines = lines_of(txt)
        names = LV.match_names(O, fam)
        hit_word = [ln for ln in lines if any(re.search(r"\b" + re.escape(nm) + r"(?:s|es)?\b", ln.lower()) for nm in names)]
        cos = []
        if lines:
            e = emb.encode(lines + [fam.lower()], normalize_embeddings=True, convert_to_numpy=True)
            cos = [float(x) for x in e[:-1] @ e[-1]]
        hit_cos = [ln for ln, c in zip(lines, cos) if c > LV.COS and ln not in hit_word]
        return {"names": names, "matched_word": hit_word, "matched_cos": hit_cos, "cos": cos}, bool(hit_word or hit_cos)

    def fam_parse(txt):                                  # listener_p1v4.fam_parse, unchanged
        ls = lines_of(txt)
        if not ls:
            return []
        e = emb.encode(ls, normalize_embeddings=True, convert_to_numpy=True)
        cos = e @ femb.T
        got = []
        for li, ln in enumerate(ls):
            for j, f in enumerate(fams_all):
                word = any(re.search(r"\b" + re.escape(nm) + r"(?:s|es)?\b", ln.lower()) for nm in names_all[f])
                if (word or cos[li, j] > LV.COS) and f not in got:
                    got.append(f)
        return got

    for sp in splits:
        c = SPLITS[sp]
        od = OUTROOT / sp
        od.mkdir(parents=True, exist_ok=True)
        cache, wcache, t0 = {}, {}, time.time()

        def answer(clip, a, b):
            k = (clip, round(a, 3), round(b, 3))
            if k not in cache:
                if clip not in wcache:
                    w, sr = sf.read(str(c["wav"] / f"{clip}.wav"), dtype="float32")
                    assert sr == LV.SR and w.ndim == 1
                    wcache.clear(); wcache[clip] = w
                w = wcache[clip]
                i, j = int(a * LV.SR), int(b * LV.SR)
                cache[k] = gen(w[i:max(j, i + LV.SR)])
            return cache[k]

        d = json.loads(c["v"].read_text(encoding="utf-8"))
        for it in d["items"]:
            if "v4_text" not in it:
                continue
            txt = answer(it["clip"], *it["run_audio"])
            it["v4_text_old"], it["v4_text"] = it["v4_text"], txt
            for tag, fam in (("x", it["family"]), ("null", it["null_family"])):
                m, ok = v_match(txt, fam)
                it[f"v4_{tag}"] = m
                (it["accept"] if tag == "x" else it["null_accept"])["V4"] = ok
        (od / c["v"].name).write_text(json.dumps(d), encoding="utf-8")
        print(f"[v4fix] {sp} v: {len(cache)} answers ({time.time() - t0:.0f} s)", flush=True)
        d = json.loads(c["p1v4"].read_text(encoding="utf-8"))
        for it in d["items"]:
            txt = answer(it["clip"], *it["run_audio"])
            it["qwen_v4_text_old"], it["qwen_v4_text"] = it["qwen_v4_text"], txt
            it["qwen_fams"] = fam_parse(txt)
        (od / c["p1v4"].name).write_text(json.dumps(d), encoding="utf-8")
        print(f"[v4fix] {sp} p1v4 done ({time.time() - t0:.0f} s)", flush=True)
        if c["p4"].exists():
            d = json.loads(c["p4"].read_text(encoding="utf-8"))
            for it in d["items"]:
                if "qwen_v4_text" not in it:
                    continue
                txt = answer(it["clip"], *it["run_audio"])
                it["qwen_v4_text_old"], it["qwen_v4_text"] = it["qwen_v4_text"], txt
                m, ok = AF.v4_match(O, emb, txt, it["family"])
                it["qwen_v4_match"], it["qwen_v4"] = m, bool(ok)
            (od / c["p4"].name).write_text(json.dumps(d), encoding="utf-8")
        print(f"[v4fix] {sp} done: {len(cache)} answers, degenerate now {sum(map(degenerate, cache.values()))} "
              f"({time.time() - t0:.0f} s)", flush=True)
    print("V4FIX_DONE")


if __name__ == "__main__":
    if sys.argv[1] == "count":
        count()
    else:
        run(sys.argv[2:])
