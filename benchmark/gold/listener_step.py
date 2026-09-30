"""Rounds 25-26 candidate screen (docs/prereg_round13_detector_push.md): Step-Audio-2-mini (stepfun-ai/Step-Audio-2-mini,
official weights, revision e36fdd5d71e0ea22f09dd94bbab9bfc544ca1e36, + the official inference code github.com/stepfun-ai/
Step-Audio2 stepaudio2.StepAudio2, commit 76e272b56c3917a8d7188f18bbb5a65dfc8a0845) as a further open-inventory listener
("V4"), on exactly the P2 and PV items of the amendment-A files that Qwen3-Omni (listener_variants.py), Audio Flamingo Next
(listener_afnext.py) and Kimi-Audio (listener_kimi.py) already answered. Structure mirrors listener_kimi.py (round 16 N3).
  V4  listener_variants.V4_Q, greedy (do_sample False), at most V4_NEW = 64 new tokens, text output only (token2wav not
      downloaded / not loaded, no <tts_start>); one generation per run cut, shared by X and the null.
      Chat as the official README "multi-modal inputs" example: [system "You are a helpful assistant.",
      human [text V4_Q, audio], assistant None] (text_first; --order audio_first puts the audio before the text).
      Audio = item["run_audio"] (the same cut Qwen / AF / Kimi V4 heard), written to a temporary 16 kHz wav because the
      official apply_chat_template only takes a file path.
      The official __call__ drops the last generated token (it assumes EOS); when the 64-token cap is hit a real token is lost.
      Kept as the official code returns it; step_v4_ntok / step_v4_capped record when it happens.
  match: listener_afnext.v4_match (= listener_variants.score lines 405-415), unchanged: match_names word match, else
      all-mpnet-base-v2 cosine > LV.COS (0.6). The code tree (ontology) is the one that scored the split (--root per split).
Two steps, two Python environments (Step needs transformers 4.49 per its README; the matching must use the msproj stack that
scored Qwen/AF):
    ~/venvs/stepaudio/bin/python benchmark/gold/listener_step.py gen dev [--smoke 5]   # GPU: writes step_v4_text
    python benchmark/gold/listener_step.py match dev [--smoke]                         # msproj: accept {"V4"}, counts
No gold is read: output items drop every gold field; the DEV print is accept counts only (Step vs Qwen V4 vs AF V4 vs Kimi V4).
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
STEP = "stepfun-ai/Step-Audio-2-mini"
STEP_REV = "e36fdd5d71e0ea22f09dd94bbab9bfc544ca1e36"
STEP_CODE = HOME / "Step-Audio2"                                      # git clone github.com/stepfun-ai/Step-Audio2
STEP_CODE_COMMIT = "76e272b56c3917a8d7188f18bbb5a65dfc8a0845"
SYSTEM = "You are a helpful assistant."
POOLS = ("P2", "PV")
KEEP = ("clip", "pool", "family", "label", "start", "end", "run_start", "run_end", "cut_start", "cut_end", "run_audio",
        "run_len", "peak", "conf", "origin", "depictable", "null_family", "run_from_cache")
W = HOME / "MscProj" / "data" / "work"
G13, GTG, GM = (HOME / r / "benchmark" / "gold" for r in ("MscProj_r13", "MscProj_tg", "MscProj"))
CFG = {   # root = the code tree whose listener_variants / ontology scored the split's Qwen V4
    "dev": {"root": HOME / "MscProj_r13", "items": G13 / "dev_listener_v.json", "afn": G13 / "dev_listener_afn.json",
            "kimi": G13 / "dev_listener_kimi.json", "wav": W / "devcand" / "wav16", "out": G13 / "dev_listener_step.json"},
    "dev2": {"root": HOME / "MscProj_tg", "items": GTG / "dev2_listener_v.json", "afn": GTG / "dev2_listener_afn.json",
             "kimi": GTG / "dev2_listener_kimi.json", "wav": W / "r13dev2" / "wav16", "out": GTG / "dev2_listener_step.json",
             "stems": GTG / "dev2_stems.txt"},
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
        print(f"[step] {split}: {len(stems)} stems, {len(have)} clips with items, not covered {miss}", flush=True)
    return items


def pick_smoke(items, n):
    pick, seen = [], set()
    for p in ("P2", "PV", "P2", "PV", "P2", "P2", "P2"):
        c = [x for x in items if x["pool"] == p and x["clip"] not in seen]
        if c:
            x = c[len(c) // 3]; pick.append(x); seen.add(x["clip"])
    return pick[:n]


# ============================================================================= step 1: generation (stepaudio venv, GPU)
def gen(split, smoke, order="text_first"):
    sys.path.insert(0, str(STEP_CODE))                                 # before lv(): the official code imports bare `utils`
    import stepaudio2
    import utils as step_utils
    assert Path(step_utils.__file__).resolve().parent == STEP_CODE.resolve(), step_utils.__file__
    import soundfile as sf
    import torch
    import transformers
    from huggingface_hub import snapshot_download
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
    path = snapshot_download(STEP, revision=STEP_REV, local_files_only=True)
    model = stepaudio2.StepAudio2(path)
    meta.update({"prereg": "rounds 25-26 candidate screen (docs/prereg_round13_detector_push.md)", "split": split,
                 "model": STEP, "revision": STEP_REV, "snapshot": path,
                 "code": f"github.com/stepfun-ai/Step-Audio2 stepaudio2.StepAudio2 (commit {STEP_CODE_COMMIT}), text only, "
                 "token2wav not loaded", "transformers": transformers.__version__, "torch": torch.__version__,
                 "candidates": str(cfg["items"]) + f" pools {POOLS}", "audio": "item run_audio from " + str(cfg["wav"]),
                 "V4": LV.V4_Q, "V4_decoding": f"do_sample False (greedy), max_new_tokens {LV.V4_NEW}; official __call__ "
                 "drops the last generated token (assumes EOS), so a capped answer loses one token (step_v4_capped)",
                 "chat": {"text_first": f"[system {SYSTEM!r}, human [text V4_Q, audio], assistant None] (README multi-modal "
                          "example order)",
                          "audio_first": f"[system {SYSTEM!r}, human [audio, text V4_Q], assistant None] (the Qwen / AF "
                          "order)"}[order],
                 "untested_alternative": "instruction in the system slot (README audio_caption_test style)",
                 "load_seconds": round(time.time() - t0, 1)})
    print(f"[step] loaded in {meta['load_seconds']} s from {path}", flush=True)
    tmpd = Path(tempfile.mkdtemp(prefix="step_", dir=os.environ.get("TMPDIR")))
    todo = sorted([x for x in items if "step_v4_text" not in x], key=lambda x: (x["clip"], x["cut_start"]))
    print(f"[step] {split}: {len(items)} items, {len(todo)} to do -> {op}", flush=True)
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
                content = [{"type": "text", "text": LV.V4_Q}, {"type": "audio", "audio": str(fp)}]
                if order == "audio_first":
                    content = content[::-1]
                msgs = [{"role": "system", "content": SYSTEM}, {"role": "human", "content": content},
                        {"role": "assistant", "content": None}]
                with torch.inference_mode():
                    toks, txt, _ = model(msgs, max_new_tokens=LV.V4_NEW, do_sample=False)
                v4cache[k] = (txt.strip(), len(toks))
            it["step_v4_text"], it["step_v4_ntok"] = v4cache[k]
            it["step_v4_capped"] = it["step_v4_ntok"] >= LV.V4_NEW - 1
            if smoke:
                print(f"[smoke] {it['clip']} {it['pool']} {it['family']} {it['start']:.2f}-{it['end']:.2f} cut "
                      f"{it['run_audio']}\n  V4 text ({it['step_v4_ntok']} tok): {it['step_v4_text']!r}", flush=True)
            if n == 10:
                print(f"[step] {split} 10 items in {time.time() - t1:.0f} s", flush=True)
            if n % 50 == 0:
                dump(op, d)
                print(f"[step] {split} {n}/{len(todo)} ({time.time() - t1:.0f} s, {len(v4cache)} cuts)", flush=True)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    meta["seconds_gen"] = round(time.time() - t1, 1) + meta.get("seconds_gen", 0.0)
    meta["unique_cuts"] = len({(x["clip"], round(x["run_audio"][0], 3), round(x["run_audio"][1], 3)) for x in items})
    dump(op, d)
    print(f"[step] {split} gen done {len(todo)} in {time.time() - t1:.0f} s; max GPU mem "
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
    """listener_kimi.norm, unchanged: tag-style answers ('Bird,Wild_animals') to one tag per line, underscores to spaces.
    Only for the *_norm fields; accept / null_accept use the raw text (the literal rule)."""
    return txt.replace("_", " ").replace(",", "\n")


def match(split, smoke):
    import torch
    from sentence_transformers import SentenceTransformer
    LV = lv(split)
    cfg, op = CFG[split], outp(split, smoke)
    d = json.loads(op.read_text(encoding="utf-8"))
    items = d["items"]
    assert all("step_v4_text" in x for x in items), "gen not finished"
    O = LV.Onto()
    emb = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda" if torch.cuda.is_available() else "cpu")
    for it in items:
        it["accept"], it["null_accept"], it["accept_norm"], it["null_accept_norm"] = {}, {}, {}, {}
        for tag, fam in (("x", it["family"]), ("null", it["null_family"])):
            m, ok = v4_match(LV, O, emb, it["step_v4_text"], fam)
            it[f"step_v4_{tag}"] = m
            (it["accept"] if tag == "x" else it["null_accept"])["V4"] = ok
            m, ok = v4_match(LV, O, emb, norm(it["step_v4_text"]), fam)
            it[f"step_v4_{tag}_norm"] = m
            (it["accept_norm"] if tag == "x" else it["null_accept_norm"])["V4"] = ok
    d["_meta"].update({"V4_rule": "some line matches X (listener_afnext.v4_match: match_names word match, else "
                       f"all-mpnet-base-v2 cosine > {LV.COS})", "match_root": str(cfg["root"]),
                       "match_device": str(emb.device), "null": "the item's cached null_family on the same text",
                       "V4_norm": "accept_norm / null_accept_norm: the same v4_match on the Step text with ',' -> newline and "
                       "'_' -> space (listener_kimi.norm)"})
    dump(op, d)
    print(f"[match] {split}: {len(items)} items -> {op}", flush=True)
    # rule-identity check: the same function on the Qwen / AF / Kimi texts must give their stored V4 flags (flags only, no gold)
    rd = lambda p: [x for x in json.loads(p.read_text(encoding="utf-8"))["items"] if x["pool"] in POOLS] if p.exists() else []
    qv, af, km = rd(cfg["items"]), rd(cfg["afn"]), rd(cfg["kimi"])
    for name, src, fld in (("Qwen", qv, "v4_text"), ("AF", af, "afn_v4_text"), ("Kimi", km, "kimi_v4_text")):
        src = [x for x in src if fld in x and "V4" in x.get("accept", {})]
        bad = sum(v4_match(LV, O, emb, x[fld], x["family"])[1] != x["accept"]["V4"] for x in src)
        print(f"[check] {split} {name} V4 re-derived with this code: {len(src) - bad}/{len(src)} identical", flush=True)
    qk = {key(x): x["accept"]["V4"] for x in qv if "V4" in x.get("accept", {})}
    ak = {key(x): x["accept"]["V4"] for x in af if "V4" in x.get("accept", {})}
    kk = {key(x): x["accept"]["V4"] for x in km if "V4" in x.get("accept", {})}
    kkn = {key(x): x["accept_norm"]["V4"] for x in km if "V4" in x.get("accept_norm", {})}
    counts = {}
    for (fa, fn), p in [(f, p) for f in (("accept", "null_accept"), ("accept_norm", "null_accept_norm"))
                        for p in POOLS + ("P2+PV",)]:
        pp = [x for x in items if p == "P2+PV" or x["pool"] == p]
        both = [x for x in pp if key(x) in qk and key(x) in ak]
        s_ = [x[fa]["V4"] for x in both]; q_ = [qk[key(x)] for x in both]; a_ = [ak[key(x)] for x in both]
        v = [a + b + c for a, b, c in zip(s_, q_, a_)]
        c = {"n": len(pp), "n_with_qwen_af": len(both), "step": sum(s_), "qwen": sum(q_), "af": sum(a_),
             "step_and_qwen": sum(a and b for a, b in zip(s_, q_)), "step_and_af": sum(a and b for a, b in zip(s_, a_)),
             "qwen_and_af": sum(a and b for a, b in zip(q_, a_)),
             "step_only": sum(a and not (b or z) for a, b, z in zip(s_, q_, a_)),
             "ge2_of_3": sum(t >= 2 for t in v), "all3": sum(t == 3 for t in v),
             "step_null_accept": sum(x[fn]["V4"] for x in pp)}
        kd = kkn if fa == "accept_norm" else kk
        kb = [x for x in both if key(x) in kd]
        c.update({"n_with_kimi": len(kb), "kimi": sum(kd[key(x)] for x in kb),
                  "step_and_kimi": sum(x[fa]["V4"] and kd[key(x)] for x in kb)})
        counts[f"{fa} {p}"] = c
        print(f"[dev] {split} Step {fa} {p} n {len(pp)} (with Qwen+AF {len(both)}): Step V4 {c['step']}, Qwen V4 {c['qwen']}, "
              f"AF V4 {c['af']}; Step&Qwen {c['step_and_qwen']}, Step&AF {c['step_and_af']}, Qwen&AF {c['qwen_and_af']}; "
              f"Step only {c['step_only']}; >=2 of 3 {c['ge2_of_3']}, all 3 {c['all3']}; Step null accept "
              f"{c['step_null_accept']}; Kimi V4 {c['kimi']} (Kimi {'norm' if fa == 'accept_norm' else 'raw'}), "
              f"Step&Kimi {c['step_and_kimi']}", flush=True)
    empty = sum(not x["step_v4_text"].strip() for x in items)
    capped = sum(bool(x.get("step_v4_capped")) for x in items)
    cjk = sum(bool(re.search(r"[一-鿿]", x["step_v4_text"])) for x in items)
    ml = sum(len([ln for ln in x["step_v4_text"].splitlines() if ln.strip()]) for x in items) / max(1, len(items))
    d["_meta"]["counts_no_gold"] = counts
    d["_meta"]["text_stats"] = {"empty": empty, "capped": capped, "with_cjk": cjk, "mean_lines": round(ml, 2)}
    dump(op, d)
    print(f"[dev] {split} empty Step texts {empty}; capped at {LV.V4_NEW} {capped}; with Chinese characters {cjk}; "
          f"mean lines {ml:.2f}", flush=True)


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
