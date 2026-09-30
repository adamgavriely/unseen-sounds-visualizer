"""Round 16 N3 (docs/prereg_round13_detector_push.md): Kimi-Audio-7B-Instruct (moonshotai/Kimi-Audio-7B-Instruct, official
weights + the official kimia_infer code, github.com/MoonshotAI/Kimi-Audio) as a THIRD open-inventory listener ("V4"), on
exactly the P2 and PV items of the amendment-A files that Qwen3-Omni (listener_variants.py) and Audio Flamingo Next
(listener_afnext.py) already answered. Structure mirrors listener_afnext.py.
  V4  listener_variants.V4_Q, greedy (text_temperature 0), at most V4_NEW = 64 text tokens, text output only (no audio
      generation, speech detokenizer not downloaded / not loaded); one generation per run cut, shared by X and the null.
      Message order as the official README: [text instruction, audio].
      Audio = item["run_audio"] (the same cut Qwen V4 and AF V4 heard: [cut_start - 1, cut_end + 1] clipped, >= 1 s), written to
      a temporary 16 kHz wav because kimia_infer only takes a file path.
  match: listener_afnext.v4_match (= listener_variants.score lines 405-415), unchanged: match_names word match, else
      all-mpnet-base-v2 cosine > LV.COS (0.6). The code tree (ontology) is the one that scored the split (--root per split).
Two steps, two Python environments (Kimi needs transformers 4.x + flash-attn; the matching must use the msproj stack that
scored Qwen/AF):
    ~/venvs/kimi/bin/python benchmark/gold/listener_kimi.py gen dev [--smoke 5]   # GPU, kimi venv: writes kimi_v4_text
    python benchmark/gold/listener_kimi.py match dev [--smoke]                    # msproj: accept {"V4"}, null_accept, counts
No gold is read: output items drop every gold field; the DEV sanity print is accept counts only. TEST splits: answers + accept
flags only, no counts against anything.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

HOME = Path.home()
KIMI = "moonshotai/Kimi-Audio-7B-Instruct"
POOLS = ("P2", "PV")
KEEP = ("clip", "pool", "family", "label", "start", "end", "run_start", "run_end", "cut_start", "cut_end", "run_audio",
        "run_len", "peak", "conf", "origin", "depictable", "null_family", "run_from_cache")
W = HOME / "MscProj" / "data" / "work"
G13, GTG, GM = (HOME / r / "benchmark" / "gold" for r in ("MscProj_r13", "MscProj_tg", "MscProj"))
CFG = {   # root = the code tree whose listener_variants / ontology scored the split's Qwen V4
    "dev": {"root": HOME / "MscProj_r13", "items": G13 / "dev_listener_v.json", "afn": G13 / "dev_listener_afn.json",
            "wav": W / "devcand" / "wav16", "out": G13 / "dev_listener_kimi.json"},
    "dev2": {"root": HOME / "MscProj_tg", "items": GTG / "dev2_listener_v.json", "afn": GTG / "dev2_listener_afn.json",
             "wav": W / "r13dev2" / "wav16", "out": GTG / "dev2_listener_kimi.json", "stems": GTG / "dev2_stems.txt"},
    "test": {"root": HOME / "MscProj", "items": GM / "test_listener_v.json", "afn": GM / "test_listener_afn.json",
             "wav": W / "r13test" / "wav16", "out": GM / "test_listener_kimi.json", "test": True},
    "test2": {"root": HOME / "MscProj_tg", "items": GTG / "test2_listener_v.json", "afn": GTG / "test2_listener_afn.json",
              "wav": W / "r13test2" / "wav16", "out": GTG / "test2_listener_kimi.json", "stems": GTG / "test2_stems.txt",
              "test": True},
}


def lv(split):
    root = CFG[split]["root"].resolve()
    sys.path.insert(0, str(root))
    from benchmark.gold import listener_variants as LV
    assert Path(LV.__file__).resolve().is_relative_to(root), (LV.__file__, root)
    LV.S.load_gold = LV._no_gold
    return LV


def outp(split, smoke):
    o = CFG[split]["out"]
    return o.with_name(o.stem + "_smoke.json") if smoke else o


def dump(p, obj):                                                     # dev_candidates_check.dump, unchanged
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(p)


def key(x):
    return (x["clip"], x["family"], x["label"], round(float(x["start"]), 3), round(float(x["end"]), 3), x["pool"])


def cut(w, a, b, sr):                                                 # listener_variants.score's cut(), unchanged
    i = int(a * sr); j = int(b * sr)
    return w[i:max(j, i + sr)]


def base_items(split):
    cfg = CFG[split]
    d = json.loads(cfg["items"].read_text(encoding="utf-8"))
    items = [{k: x[k] for k in KEEP if k in x} for x in d["items"] if x["pool"] in POOLS]
    if "stems" in cfg:                                                 # the tagger files must cover every listed clip
        stems = [s.strip() for s in cfg["stems"].read_text(encoding="utf-8").split() if s.strip()]
        have = {x["clip"] for x in d["items"]}
        miss = [s for s in stems if s not in have]
        print(f"[kimi] {split}: {len(stems)} stems, {len(have)} clips with items, not covered {miss}", flush=True)
    return items


def pick_smoke(items, n):
    pick, seen = [], set()
    for p in ("P2", "PV", "P2", "PV", "P2", "P2", "P2"):
        c = [x for x in items if x["pool"] == p and x["clip"] not in seen]
        if c:
            x = c[len(c) // 3]; pick.append(x); seen.add(x["clip"])
    return pick[:n]


# ============================================================================= step 1: generation (kimi venv, GPU)
def gen(split, smoke, order="text_first"):
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download
    from kimia_infer.api.kimia import KimiAudio
    LV = lv(split)
    SR = LV.SR
    cfg, op = CFG[split], outp(split, smoke)
    if op.exists():
        d = json.loads(op.read_text(encoding="utf-8"))
    else:
        items = base_items(split)
        if smoke:
            items = pick_smoke(items, smoke)
        d = {"_meta": {}, "items": items}
    meta, items = d["_meta"], d["items"]
    t0 = time.time()
    path = snapshot_download(KIMI, local_files_only=True)
    model = KimiAudio(model_path=path, load_detokenizer=False)
    import transformers
    meta.update({"prereg": "round 16 N3 (docs/prereg_round13_detector_push.md)", "split": split, "model": KIMI,
                 "snapshot": path, "code": "github.com/MoonshotAI/Kimi-Audio kimia_infer (KimiAudio, load_detokenizer=False)",
                 "transformers": transformers.__version__, "torch": torch.__version__,
                 "candidates": str(cfg["items"]) + f" pools {POOLS}", "audio": "item run_audio from " + str(cfg["wav"]),
                 "V4": LV.V4_Q, "V4_decoding": f"output_type text, text_temperature 0 (greedy), max_new_tokens {LV.V4_NEW}, "
                 "text_repetition_penalty 1.0 (README default)",
                 "chat": {"text_first": "[user text V4_Q, user audio] (README order)",
                          "audio_first": "[user audio, user text V4_Q] (the Qwen / AF order)"}[order],
                 "load_seconds": round(time.time() - t0, 1)})
    print(f"[kimi] loaded in {meta['load_seconds']} s from {path}", flush=True)
    params = dict(output_type="text", text_temperature=0.0, text_top_k=5, audio_temperature=0.0, audio_top_k=5,
                  text_repetition_penalty=1.0, text_repetition_window_size=16, audio_repetition_penalty=1.0,
                  audio_repetition_window_size=64, max_new_tokens=LV.V4_NEW)
    tmpd = Path(tempfile.mkdtemp(prefix="kimi_", dir=os.environ.get("TMPDIR")))
    todo = sorted([x for x in items if "kimi_v4_text" not in x], key=lambda x: (x["clip"], x["cut_start"]))
    print(f"[kimi] {split}: {len(items)} items, {len(todo)} to do -> {op}", flush=True)
    cache, v4cache = {}, {}
    t1 = time.time()
    try:
        for n, it in enumerate(todo, 1):
            if it["clip"] not in cache:
                w, sr = sf.read(str(cfg["wav"] / f"{it['clip']}.wav"), dtype="float32")
                assert sr == SR and w.ndim == 1, (it["clip"], sr, w.shape)
                cache.clear(); cache[it["clip"]] = w
            w = cache[it["clip"]]
            k = (it["clip"], round(it["run_audio"][0], 3), round(it["run_audio"][1], 3))
            if k not in v4cache:
                seg = cut(w, *it["run_audio"], SR)
                fp = tmpd / "seg.wav"
                sf.write(str(fp), seg, SR, subtype="FLOAT")
                msgs = [{"role": "user", "message_type": "text", "content": LV.V4_Q},
                        {"role": "user", "message_type": "audio", "content": str(fp)}]
                if order == "audio_first":
                    msgs = msgs[::-1]
                _, txt = model.generate(msgs, **params)
                v4cache[k] = txt.strip()
            it["kimi_v4_text"] = v4cache[k]
            if smoke:
                print(f"[smoke] {it['clip']} {it['pool']} {it['family']} {it['start']:.2f}-{it['end']:.2f} cut "
                      f"{it['run_audio']}\n  V4 text: {it['kimi_v4_text']!r}", flush=True)
            if n == 10:
                print(f"[kimi] {split} 10 items in {time.time() - t1:.0f} s", flush=True)
            if n % 50 == 0:
                dump(op, d)
                print(f"[kimi] {split} {n}/{len(todo)} ({time.time() - t1:.0f} s, {len(v4cache)} cuts)", flush=True)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    meta["seconds_gen"] = round(time.time() - t1, 1) + meta.get("seconds_gen", 0.0)
    meta["unique_cuts"] = len({(x["clip"], round(x["run_audio"][0], 3), round(x["run_audio"][1], 3)) for x in items})
    dump(op, d)
    print(f"[kimi] {split} gen done {len(todo)} in {time.time() - t1:.0f} s; max GPU mem "
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB -> {op}", flush=True)


# ============================================================================= step 2: matching (msproj, the Qwen/AF stack)
def v4_match(LV, O, emb, txt, fam):                                   # listener_afnext.v4_match, unchanged
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


def norm(txt):
    """Kimi answers V4 with an AudioSet-style tag string ('Bird,Wild_animals,Animal') instead of one sound per line: one tag
    per line, underscores to spaces. Only for the *_norm fields; accept / null_accept use the raw text (the literal rule)."""
    return txt.replace("_", " ").replace(",", "\n")


def match(split, smoke):
    import torch
    from sentence_transformers import SentenceTransformer
    LV = lv(split)
    cfg, op = CFG[split], outp(split, smoke)
    d = json.loads(op.read_text(encoding="utf-8"))
    items = d["items"]
    assert all("kimi_v4_text" in x for x in items), "gen not finished"
    O = LV.Onto()
    emb = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda" if torch.cuda.is_available() else "cpu")
    for it in items:
        it["accept"], it["null_accept"], it["accept_norm"], it["null_accept_norm"] = {}, {}, {}, {}
        for tag, fam in (("x", it["family"]), ("null", it["null_family"])):
            m, ok = v4_match(LV, O, emb, it["kimi_v4_text"], fam)
            it[f"kimi_v4_{tag}"] = m
            (it["accept"] if tag == "x" else it["null_accept"])["V4"] = ok
            m, ok = v4_match(LV, O, emb, norm(it["kimi_v4_text"]), fam)
            it[f"kimi_v4_{tag}_norm"] = m
            (it["accept_norm"] if tag == "x" else it["null_accept_norm"])["V4"] = ok
    d["_meta"].update({"V4_rule": "some line matches X (listener_afnext.v4_match: match_names word match, else "
                       f"all-mpnet-base-v2 cosine > {LV.COS})", "match_root": str(cfg["root"]),
                       "match_device": str(emb.device), "null": "the item's cached null_family on the same text",
                       "V4_norm": "accept_norm / null_accept_norm: the same v4_match on the Kimi text with ',' -> newline and "
                       "'_' -> space (Kimi answers with an AudioSet tag string, not one sound per line)"})
    dump(op, d)
    print(f"[match] {split}: {len(items)} items -> {op}", flush=True)
    # rule-identity check: the same function on the Qwen / AF texts must give their stored V4 flags (flags only, no gold)
    qv = [x for x in json.loads(cfg["items"].read_text(encoding="utf-8"))["items"] if x["pool"] in POOLS]
    af = [x for x in json.loads(cfg["afn"].read_text(encoding="utf-8"))["items"] if x["pool"] in POOLS] \
        if cfg["afn"].exists() else []
    for name, src, fld in (("Qwen", qv, "v4_text"), ("AF", af, "afn_v4_text")):
        src = [x for x in src if fld in x and "V4" in x.get("accept", {})]
        bad = sum(v4_match(LV, O, emb, x[fld], x["family"])[1] != x["accept"]["V4"] for x in src)
        print(f"[check] {split} {name} V4 re-derived with this code: {len(src) - bad}/{len(src)} identical", flush=True)
    if cfg.get("test"):
        return                                                         # TEST: answers + flags only, no counts
    qk = {key(x): x["accept"]["V4"] for x in qv if "V4" in x.get("accept", {})}
    ak = {key(x): x["accept"]["V4"] for x in af if "V4" in x.get("accept", {})}
    for (fa, fn), p in [(f, p) for f in (("accept", "null_accept"), ("accept_norm", "null_accept_norm"))
                        for p in POOLS + ("P2+PV",)]:
        pp = [x for x in items if p == "P2+PV" or x["pool"] == p]
        both = [x for x in pp if key(x) in qk and key(x) in ak]
        k_ = [x[fa]["V4"] for x in both]; q_ = [qk[key(x)] for x in both]; a_ = [ak[key(x)] for x in both]
        s = [a + b + c for a, b, c in zip(k_, q_, a_)]
        print(f"[dev] {split} Kimi {fa} {p} n {len(pp)} (with Qwen+AF {len(both)}): Kimi V4 {sum(k_)}, Qwen V4 {sum(q_)}, AF V4 "
              f"{sum(a_)}; Kimi&Qwen {sum(a and b for a, b in zip(k_, q_))}, Kimi&AF {sum(a and b for a, b in zip(k_, a_))}, "
              f"Qwen&AF {sum(a and b for a, b in zip(q_, a_))}; >=2 of 3 {sum(v >= 2 for v in s)}, all 3 "
              f"{sum(v == 3 for v in s)}; Kimi null accept {sum(x[fn]['V4'] for x in pp)}", flush=True)
    empty = sum(not x["kimi_v4_text"].strip() for x in items)
    print(f"[dev] {split} empty Kimi texts {empty}; mean lines "
          f"{sum(len([l for l in x['kimi_v4_text'].splitlines() if l.strip()]) for x in items) / max(1, len(items)):.2f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("gen", "match"))
    ap.add_argument("split", choices=tuple(CFG))
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--order", choices=("text_first", "audio_first"), default="text_first")
    a = ap.parse_args()
    if a.step == "gen":
        gen(a.split, a.smoke, a.order)
    else:
        match(a.split, a.smoke)


if __name__ == "__main__":
    main()
