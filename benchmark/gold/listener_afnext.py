"""Audio Flamingo Next (nvidia/audio-flamingo-next-hf) as a second listener, on exactly the
listener_variants candidates (the items of dev_listener_v.json / test_listener_v.json: P2 peak >= 0.5,
PV, P1 depictable conf < 0.6), same audio cut (item["run_audio"] from the same wav16 files, at least 1 s).
  V4  open inventory: listener_variants.V4_Q, greedy, 64 new tokens; one generation per cut, shared by X and the null;
      same matching rule as listener_variants.score (match_names word match, else all-mpnet-base-v2 cosine > 0.6)
  YN  benchmark.listener_round.QUESTION on the same cut; score = max logit(yes ids) - max logit(no ids)
Null control: the item's cached null_family on the same cut (as listener_variants). Run on all three pools.
No gold is read by score (score_per_sound.load_gold raises); the DEV report reads only the cached 'gold' field
(dev_listener.json, copied into dev_listener_v.json), never the gold annotations. TEST: features only, no report.

    python benchmark/gold/listener_afnext.py smoke [--n 5]   # GPU: 5 DEV items -> *_afn_smoke.json, prints texts/ids
    python benchmark/gold/listener_afnext.py score           # GPU: one model load, DEV then TEST (resumable)
    python benchmark/gold/listener_afnext.py both            # GPU: smoke, then score, one model load
    python benchmark/gold/listener_afnext.py report          # CPU: DEV V4 / AGREE counts by cached gold class, 8 misses, nulls

(design record: release v1.2.0)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold import dev_candidates_check as C
from benchmark.gold import listener_variants as LV
from benchmark.listener_round import QUESTION

AFN = "nvidia/audio-flamingo-next-hf"
GOLD = _ROOT / "benchmark" / "gold"
OUT = {"dev": GOLD / "dev_listener_afn.json", "test": GOLD / "test_listener_afn.json"}
SMOKE = GOLD / "dev_listener_afn_smoke.json"
SR = LV.SR
KEEP = ("clip", "pool", "family", "label", "start", "end", "run_start", "run_end", "cut_start", "cut_end", "run_audio",
        "run_len", "peak", "conf", "origin", "depictable", "null_family", "run_from_cache")


def key(x):
    return (x["clip"], x["family"], x["label"], round(float(x["start"]), 3), round(float(x["end"]), 3), x["pool"])


def cut(w, a, b):                                                   # listener_variants.score's cut(), unchanged
    i = int(a * SR); j = int(b * SR)
    return w[i:max(j, i + SR)]


def v4_match(O, emb, txt, fam):                                     # listener_variants.score lines 405-415, unchanged
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip() for ln in txt.splitlines()]
    lines = [ln for ln in lines if ln]
    names = LV.match_names(O, fam)
    hit_word = [ln for ln in lines if any(re.search(r"\b" + re.escape(nm) + r"(?:s|es)?\b", ln.lower()) for nm in names)]
    cos = []
    if lines:
        e = emb.encode(lines + [fam.lower()], normalize_embeddings=True, convert_to_numpy=True)
        cos = [float(x) for x in e[:-1] @ e[-1]]
    hit_cos = [ln for ln, c in zip(lines, cos) if c > LV.COS and ln not in hit_word]
    return {"names": names, "matched_word": hit_word, "matched_cos": hit_cos, "cos": cos}, bool(hit_word or hit_cos)


def base_items(split):
    d = json.loads(LV.SPLITS[split]["out"].read_text(encoding="utf-8"))
    return [{k: x[k] for k in KEEP if k in x} for x in d["items"]]


def load_model():
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(AFN)
    cls = getattr(transformers, AutoConfig.from_pretrained(AFN).architectures[0])   # MusicFlamingoForConditionalGeneration
    model = cls.from_pretrained(AFN, dtype=torch.bfloat16, device_map="auto").eval()
    emb = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    tok = proc.tokenizer
    first = lambda w: tok.encode(w, add_special_tokens=False)[0]
    yes_ids = sorted({first(w) for w in ("yes", "Yes", " yes", " Yes")})
    no_ids = sorted({first(w) for w in ("no", "No", " no", " No")})
    info = {"model": AFN, "class": type(model).__name__, "processor": type(proc).__name__,
            "transformers": transformers.__version__, "yes_ids": yes_ids, "no_ids": no_ids,
            "yes_tokens": [tok.decode([i]) for i in yes_ids], "no_tokens": [tok.decode([i]) for i in no_ids],
            "load_seconds": round(time.time() - t0, 1)}
    print(f"[afn] loaded {info}", flush=True)
    dev = next(model.parameters()).device

    def prep(w, q):
        conv = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=[text], audio=[w], return_tensors="pt")
        inp = {k: v for k, v in inp.items() if k != "num_audio_tokens"}
        inp = {k: v.to(dev) for k, v in inp.items()}
        inp["input_features"] = inp["input_features"].to(model.dtype)
        return inp, text

    def last(w, q):
        inp, _ = prep(w, q)
        with torch.inference_mode():
            return model(**inp).logits[0, -1].float()

    def yesno(w, fam):
        lg = last(w, QUESTION.format(fam.lower()))
        return float(lg[yes_ids].max() - lg[no_ids].max()), int(lg.argmax())

    def gen(w, q, n):
        inp, _ = prep(w, q)
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=n, do_sample=False, use_cache=True)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    return {"proc": proc, "model": model, "emb": emb, "tok": tok, "info": info, "prep": prep, "yesno": yesno, "gen": gen}


def run(M, items, split, outp, meta, every=100):
    import soundfile as sf
    O = LV.Onto()
    d = {"_meta": meta, "items": items}
    todo = sorted([x for x in items if "accept" not in x], key=lambda x: (x["clip"], x["cut_start"]))
    cache, v4cache = {}, {}
    t1 = time.time()
    print(f"[afn] {split}: {len(todo)} items to do", flush=True)

    def wav(st):
        if st not in cache:
            w, sr = sf.read(str(LV.SPLITS[split]["wav"] / f"{st}.wav"), dtype="float32")
            assert sr == SR and w.ndim == 1, (st, sr, w.shape)
            cache.clear(); cache[st] = w
        return cache[st]

    for n, it in enumerate(todo, 1):
        w = wav(it["clip"])
        seg = cut(w, *it["run_audio"])
        k = (it["clip"], round(it["run_audio"][0], 3), round(it["run_audio"][1], 3))
        if k not in v4cache:
            v4cache[k] = M["gen"](seg, LV.V4_Q, LV.V4_NEW)
        it["afn_v4_text"] = v4cache[k]
        it["accept"], it["null_accept"] = {}, {}
        for tag, fam in (("x", it["family"]), ("null", it["null_family"])):
            acc = it["accept"] if tag == "x" else it["null_accept"]
            m, ok = v4_match(O, M["emb"], it["afn_v4_text"], fam)
            it[f"afn_v4_{tag}"] = m
            acc["V4"] = ok
            s, top = M["yesno"](seg, fam)
            it[f"afn_yn_{tag}"] = s
            it[f"afn_yn_{tag}_top1"] = top
            acc["YN0"] = bool(s > 0)                                  # sanity flag only, not a preregistered rule
        if n == 10:
            print(f"[afn] {split} 10 items in {time.time() - t1:.0f} s", flush=True)
        if n % every == 0:
            C.dump(outp, d)
            print(f"[afn] {split} {n}/{len(todo)} ({time.time() - t1:.0f} s)", flush=True)
    meta["seconds_" + split] = round(time.time() - t1, 1)
    C.dump(outp, d)
    print(f"[afn] {split} done {len(todo)} in {time.time() - t1:.0f} s -> {outp}", flush=True)
    return d


def make_meta(M, split):
    return {"amendment": "round 14 C (docs/history/preregistrations/prereg_round13_detector_push.md, release v1.2.0)", "split": split,
            "candidates": str(LV.SPLITS[split]["out"]) + " (all items: P2 peak >= 0.5, PV, P1 depictable conf < 0.6)",
            "audio": "item run_audio = [cut_start - 1, cut_end + 1] clipped, >= 1 s, from " + str(LV.SPLITS[split]["wav"]),
            "V4": LV.V4_Q, "V4_decoding": f"greedy (do_sample=False), max_new_tokens {LV.V4_NEW}, no repetition_penalty "
            "(the model card example uses 1.2; left out to match the Qwen V4 decoding)", "V4_rule": "some line matches X "
            "(listener_variants match_names word match, else all-mpnet-base-v2 cosine > 0.6)",
            "YN": QUESTION, "YN_score": "max logit over yes ids - max logit over no ids at the first answer position",
            "chat": "content [audio, text]; the model's chat template inserts its own default system prompt (Audio Flamingo-Next)",
            "null": "the item's cached null_family on the same cut", **M["info"]}


def smoke(n, M=None):
    import torch
    S.load_gold = LV._no_gold
    M = M or load_model()
    items = base_items("dev")
    # 5 items: spread over pools and clips
    pick, seen = [], set()
    for p in ("P2", "PV", "P1", "P2", "P2"):
        c = [x for x in items if x["pool"] == p and x["clip"] not in seen]
        c = c[len(c) // 3] if c else None
        if c:
            pick.append(c); seen.add(c["clip"])
    pick = pick[:n]
    import soundfile as sf
    w, _ = sf.read(str(LV.SPLITS["dev"]["wav"] / f"{pick[0]['clip']}.wav"), dtype="float32")
    seg = cut(w, *pick[0]["run_audio"])
    inp, text = M["prep"](seg, QUESTION.format(pick[0]["family"].lower()))
    na = int((inp["input_ids"] == M["proc"].audio_token_id).sum())
    print(f"[smoke] prompt text:\n{text}\n[smoke] seg {len(seg) / SR:.2f} s; <sound> tokens in input_ids {na}; keys "
          f"{ {k: tuple(v.shape) for k, v in inp.items()} }", flush=True)
    assert na > 0
    d = run(M, pick, "dev", SMOKE, make_meta(M, "dev"), every=1000)
    tok = M["tok"]
    for x in d["items"]:
        print(f"\n[smoke] {x['clip']} {x['pool']} {x['family']} {x['start']:.2f}-{x['end']:.2f} cut {x['run_audio']} "
              f"null {x['null_family']}", flush=True)
        print(f"  V4 text: {x['afn_v4_text']!r}")
        print(f"  V4 x {x['accept']['V4']} {x['afn_v4_x']['matched_word']} {x['afn_v4_x']['matched_cos']}; null "
              f"{x['null_accept']['V4']} {x['afn_v4_null']['matched_word']} {x['afn_v4_null']['matched_cos']}")
        print(f"  YN x {x['afn_yn_x']:+.2f} (top1 {tok.decode([x['afn_yn_x_top1']])!r}); null {x['afn_yn_null']:+.2f} "
              f"(top1 {tok.decode([x['afn_yn_null_top1']])!r})", flush=True)
    print(f"[smoke] max GPU mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    return M


def score(M=None):
    S.load_gold = LV._no_gold
    M = M or load_model()
    for split in ("dev", "test"):
        outp = OUT[split]
        if outp.exists():
            d = json.loads(outp.read_text(encoding="utf-8"))
            items, meta = d["items"], d["_meta"]
        else:
            items, meta = base_items(split), make_meta(M, split)
        run(M, items, split, outp, meta)
        summary(items, split)


def summary(items, split):
    its = [x for x in items if "accept" in x]
    for p in ("P2", "PV", "P1"):
        pp = [x for x in its if x["pool"] == p]
        if pp:
            print(f"[summary] {split} {p} n {len(pp)}: V4 {sum(x['accept']['V4'] for x in pp)} (null "
                  f"{sum(x['null_accept']['V4'] for x in pp)}), YN>0 {sum(x['accept']['YN0'] for x in pp)} (null "
                  f"{sum(x['null_accept']['YN0'] for x in pp)})", flush=True)


# ============================================================================= DEV report (cached gold field only)
def report():
    S.load_gold = LV._no_gold
    afn = json.loads(OUT["dev"].read_text(encoding="utf-8"))["items"]
    qv = {key(x): x for x in json.loads(LV.SPLITS["dev"]["out"].read_text(encoding="utf-8"))["items"]}
    cache = {(x["clip"], x["label"], round(x["start"], 3), round(x["end"], 3)): x.get("gold")
             for x in json.loads((GOLD / "dev_listener.json").read_text(encoding="utf-8"))["items"] if x["pool"] == "P2"}
    its = [x for x in afn if "accept" in x]
    assert len(its) == len(afn) == len(qv), (len(its), len(afn), len(qv))
    for x in its:
        q = qv[key(x)]
        x["q"], x["qn"] = q.get("accept", {}), q.get("null_accept", {})
        x["gold"] = cache.get((x["clip"], x["label"], round(x["start"], 3), round(x["end"], 3))) if x["pool"] == "P2" else None
    rules = {"AF V4": lambda x, a, qa: a["V4"], "AF YN>0": lambda x, a, qa: a["YN0"],
             "Qwen V4": lambda x, a, qa: qa.get("V4", False), "Qwen V12": lambda x, a, qa: qa.get("V12", False),
             "AGREE V4 (Qwen V4 & AF V4)": lambda x, a, qa: qa.get("V4", False) and a["V4"],
             "AGREE V12 (Qwen V12 & AF V4)": lambda x, a, qa: qa.get("V12", False) and a["V4"]}
    acc = lambda x, r: rules[r](x, x["accept"], x["q"])
    nacc = lambda x, r: rules[r](x, x["null_accept"], x["qn"])
    p2 = [x for x in its if x["pool"] == "P2"]
    assert all(x["gold"] in ("hit_needed", "none", "other_gold") for x in p2)
    tot = {k: sum(x["gold"] == k for x in p2) for k in ("hit_needed", "none", "other_gold")}
    print(f"[dev] P2 peak >= 0.5: n {len(p2)}, by cached gold {tot}")
    for r in rules:
        by = {k: sum(1 for x in p2 if x["gold"] == k and acc(x, r)) for k in tot}
        print(f"[dev] P2 {r}: accepted hit_needed {by['hit_needed']}/{tot['hit_needed']}, none {by['none']}/{tot['none']}, "
              f"other_gold {by['other_gold']}/{tot['other_gold']}")
    for p in ("PV", "P1"):
        pp = [x for x in its if x["pool"] == p]
        print(f"[dev] {p} n {len(pp)}: " + ", ".join(f"{r} {sum(acc(x, r) for x in pp)}" for r in rules))
    # the 8 R13-3 hit misses: onsets as listed in listener_variants.MISSES (within 0.06 s of gold; gold file not read)
    cand = [x for x in its if x["pool"] in ("P2", "PV")]
    for r in rules:
        kept = []
        for st, lab, on in LV.MISSES:
            ok = False
            for x in cand:
                if x["clip"] != st or not acc(x, r) or not S.same_family(x["label"], lab):
                    continue
                eff = x["start"] if (x["pool"] == "PV" or x["run_len"] >= 0.5) else x["cut_start"]
                ok = ok or S.in_window(eff, on, S.EARLY, S.LATE)
            kept.append(ok)
        print(f"[dev] {r} keeps {sum(kept)}/8 R13-3 hits: {[f'{m[1]} {m[2]:.1f}' for m, k in zip(LV.MISSES, kept) if k]}")
    clips = sorted({x["clip"] for x in its})
    half = {c: "AB"[i % 2] for i, c in enumerate(clips)}
    for r in rules:
        s = []
        for h in "AB":
            hh = [x for x in cand if half[x["clip"]] == h]
            s.append(f"{h} {np.mean([nacc(x, r) for x in hh]):.1%} (n {len(hh)})")
        print(f"[dev] {r} null accept-rate P2+PV by half: {', '.join(s)}")
    hn = [x for x in p2 if x["gold"] in ("hit_needed", "none")]
    y = np.array([x["gold"] == "hit_needed" for x in hn])
    if y.any() and (~y).any():
        from benchmark.listener_round import auroc
        print(f"[dev] P2 AF YN score AUROC hit_needed {int(y.sum())} vs none {int((~y).sum())}: "
              f"{auroc(np.array([x['afn_yn_x'] for x in hn]), y):.3f}; Qwen R13-3 score (cached): "
              f"{auroc(np.array([qv[key(x)].get('cached_score') or 0.0 for x in hn]), y):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("smoke", "score", "both", "report"))
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    if a.step == "smoke":
        smoke(a.n)
    elif a.step == "score":
        score()
    elif a.step == "both":                                          # smoke, then the full run in the same allocation
        score(smoke(a.n))
    else:
        report()


if __name__ == "__main__":
    main()
