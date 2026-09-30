"""Round 34 MOSS (docs/prereg_round13_detector_push.md, "Round 34 MOSS"): MOSS-Audio-8B-Thinking (OpenMOSS-Team/MOSS-Audio-8B-Thinking,
Apache-2.0, arXiv 2606.01802; official src/ code from github.com/OpenMOSS/MOSS-Audio) as a listener on exactly the P2 / PV items
that Qwen3-Omni (listener_variants.py) and Audio Flamingo Next (listener_afnext.py) already answered, plus the P1 items of
*_listener_p1v4.json (the sanity set). Structure mirrors listener_kimi.py: two steps, two environments.
  gen   (~/venv_moss = msproj + transformers 4.57.1, GPU): the unchanged V4 question (V4_Q below == listener_variants.V4_Q,
        asserted in the match step) on item["run_audio"] (the same cut Qwen V4 / AF V4 heard), the processor's default prompt
        (audio first, then the text = the Qwen / AF order), GREEDY. Thinking cannot be switched off for audio requests (the
        official usage guide), so: up to MAX_NEW tokens; if no </think> came out, the prompt is re-built with the partial
        output + "\\n</think>\\n\\n" and at most V4_NEW = 64 answer tokens follow (forced close, flagged). The final answer =
        the text after the last </think> (the whole output when the model wrote no <think> block). Stored: moss_raw,
        moss_v4_text, moss_think_tokens, moss_forced.
  match (msproj): listener_afnext.v4_match unchanged on moss_v4_text -> accept {"V4"}; null_accept on the item's null_family.
The gen step imports NOTHING from the project tree (the MOSS code has its own top-level `src` package).

    ~/venv_moss/bin/python benchmark/gold/listener_moss.py gen dev|dev2|p1 [--smoke 5]
    python benchmark/gold/listener_moss.py match dev|dev2|p1 [--smoke]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HOME = Path.home()
MOSS = "OpenMOSS-Team/MOSS-Audio-8B-Thinking"
MOSS_CODE = HOME / "MOSS-Audio"
V4_Q = "List every distinct non-speech sound you hear in this recording, one per line, most prominent first."
V4_NEW, MAX_NEW, SR = 64, 2048, 16000
POOLS = ("P2", "PV")
KEEP = ("clip", "pool", "family", "label", "start", "end", "run_start", "run_end", "cut_start", "cut_end", "run_audio",
        "run_len", "peak", "conf", "origin", "depictable", "null_family", "run_from_cache")
W = HOME / "MscProj" / "data" / "work"
G13, GTG = (HOME / r / "benchmark" / "gold" for r in ("MscProj_r13", "MscProj_tg"))
CFG = {   # root = the code tree whose listener_variants / ontology scored the split's Qwen V4
    "dev": {"root": HOME / "MscProj_r13", "items": G13 / "dev_listener_v.json", "afn": G13 / "dev_listener_afn.json",
            "wav": W / "devcand" / "wav16", "out": GTG / "dev_listener_moss.json", "pools": POOLS},
    "dev2": {"root": HOME / "MscProj_tg", "items": GTG / "dev2_listener_v.json", "afn": GTG / "dev2_listener_afn.json",
             "wav": W / "r13dev2" / "wav16", "out": GTG / "dev2_listener_moss.json", "pools": POOLS},
    "p1": {"root": HOME / "MscProj_r13", "items": G13 / "dev_listener_p1v4.json", "afn": None,
           "wav": W / "devcand" / "wav16", "out": GTG / "dev_listener_p1_moss.json", "pools": ("P1",)},
}


def outp(split, smoke):
    o = CFG[split]["out"]
    return o.with_name(o.stem + "_smoke.json") if smoke else o


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(p)


def key(x):
    return (x["clip"], x["family"], x["label"], round(float(x["start"]), 3), round(float(x["end"]), 3), x["pool"])


def cut(w, a, b):                                                     # listener_variants.score's cut(), unchanged
    i = int(a * SR); j = int(b * SR)
    return w[i:max(j, i + SR)]


def base_items(split):
    cfg = CFG[split]
    d = json.loads(cfg["items"].read_text(encoding="utf-8"))
    items = [{k: x[k] for k in KEEP if k in x} for x in d["items"] if x["pool"] in cfg["pools"]]
    assert all("run_audio" in x for x in items), split
    return items


def pick_smoke(items, n):
    pick, seen = [], set()
    for p in ("P2", "PV", "P2", "PV", "P2", "P2", "P2", "P1", "P1", "P1", "P1", "P1"):
        c = [x for x in items if x["pool"] == p and x["clip"] not in seen]
        if c:
            x = c[len(c) // 3]; pick.append(x); seen.add(x["clip"])
    longest = max(items, key=lambda x: x["run_audio"][1] - x["run_audio"][0])
    if longest not in pick:
        pick.append(longest)
    return pick[:n]


def split_think(raw):
    """(answer, think_text, forced_missing): answer = text after the LAST </think>; no </think> -> ("", raw, True)."""
    if "</think>" in raw:
        head, _, tail = raw.rpartition("</think>")
        return tail.strip(), head.replace("<think>", "", 1).strip(), False
    return "", raw.replace("<think>", "", 1).strip(), True


# ============================================================================= step 1: generation (venv_moss, GPU)
def gen(split, smoke):
    import soundfile as sf
    import torch
    sys.path.insert(0, str(MOSS_CODE))
    from src.modeling_moss_audio import MossAudioModel                 # the MOSS `src`, not the project's
    from src.processing_moss_audio import MossAudioProcessor
    from huggingface_hub import snapshot_download
    import transformers
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
    path = snapshot_download(MOSS, local_files_only=True)
    model = MossAudioModel.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda:0",
                                           attn_implementation="sdpa").eval()
    proc = MossAudioProcessor.from_pretrained(path, trust_remote_code=True, enable_time_marker=True)
    tok = proc._base_tokenizer
    meta.update({"prereg": "round 34 MOSS (docs/prereg_round13_detector_push.md)", "split": split, "model": MOSS,
                 "snapshot": path, "code": str(MOSS_CODE) + " src/ (MossAudioModel, MossAudioProcessor, enable_time_marker=True)",
                 "transformers": transformers.__version__, "torch": torch.__version__,
                 "candidates": str(cfg["items"]) + f" pools {cfg['pools']}", "audio": "item run_audio from " + str(cfg["wav"]),
                 "V4": V4_Q, "prompt": "processor default prompt: system 'You are a helpful assistant.', user = audio then V4_Q",
                 "V4_decoding": f"greedy (do_sample False), max_new_tokens {MAX_NEW} incl. thinking; no </think> -> forced "
                 f"close ('\\n</think>\\n\\n' appended, <= {V4_NEW} answer tokens); answer = text after the last </think>",
                 "load_seconds": round(time.time() - t0, 1)})
    print(f"[moss] loaded in {meta['load_seconds']} s from {path}", flush=True)
    close_ids = tok.encode("\n</think>\n\n", add_special_tokens=False)

    def run(seg):
        inp = proc(text=V4_Q, audios=[seg], return_tensors="pt").to(model.device)
        inp["audio_data"] = inp["audio_data"].to(model.dtype)
        inp["audio_input_mask"] = inp["input_ids"] == proc.audio_token_id
        n0 = inp["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False, num_beams=1, use_cache=True)
        new = out[0, n0:]
        raw = tok.decode(new, skip_special_tokens=False)
        raw = re.sub(r"<\|im_end\|>.*$", "", raw, flags=re.S).replace("<|endoftext|>", "")
        ans, think, forced = split_think(raw)
        nthink = int(len(new)) if forced else len(tok.encode(think, add_special_tokens=False))
        if forced:                                                    # thinking budget hit: close it and ask for the answer
            ids = torch.cat([out[0], torch.tensor(close_ids, device=out.device)])[None]
            mask = torch.ones_like(ids)
            am = torch.zeros_like(ids, dtype=torch.bool); am[0, :n0] = inp["audio_input_mask"][0]
            with torch.inference_mode():
                out2 = model.generate(input_ids=ids, attention_mask=mask, audio_data=inp["audio_data"],
                                      audio_data_seqlens=inp["audio_data_seqlens"], audio_input_mask=am,
                                      max_new_tokens=V4_NEW, do_sample=False, num_beams=1, use_cache=True)
            ans = tok.decode(out2[0, ids.shape[1]:], skip_special_tokens=True).strip()
            raw = raw + "\n</think>\n\n" + ans
        return raw, ans, nthink, forced

    todo = sorted([x for x in items if "moss_v4_text" not in x], key=lambda x: (x["clip"], x["run_audio"][0]))
    print(f"[moss] {split}: {len(items)} items, {len(todo)} to do -> {op}", flush=True)
    cache, v4cache = {}, {}
    t1 = time.time()
    for n, it in enumerate(todo, 1):
        if it["clip"] not in cache:
            w, sr = sf.read(str(cfg["wav"] / f"{it['clip']}.wav"), dtype="float32")
            assert sr == SR and w.ndim == 1, (it["clip"], sr, w.shape)
            cache.clear(); cache[it["clip"]] = w
        w = cache[it["clip"]]
        k = (it["clip"], round(it["run_audio"][0], 3), round(it["run_audio"][1], 3))
        if k not in v4cache:
            v4cache[k] = run(cut(w, *it["run_audio"]))
        it["moss_raw"], it["moss_v4_text"], it["moss_think_tokens"], it["moss_forced"] = v4cache[k]
        if smoke:
            print(f"[smoke] {it['clip']} {it['pool']} {it['family']} {it['start']:.2f}-{it['end']:.2f} cut {it['run_audio']} "
                  f"think {it['moss_think_tokens']} forced {it['moss_forced']} ({time.time() - t1:.0f} s)\n"
                  f"  V4 text: {it['moss_v4_text']!r}", flush=True)
        if n == 10:
            print(f"[moss] {split} 10 items in {time.time() - t1:.0f} s", flush=True)
        if n % 20 == 0:
            dump(op, d)
            print(f"[moss] {split} {n}/{len(todo)} ({time.time() - t1:.0f} s, {len(v4cache)} cuts)", flush=True)
    meta["seconds_gen"] = round(time.time() - t1, 1) + meta.get("seconds_gen", 0.0)
    meta["unique_cuts"] = len({(x["clip"], round(x["run_audio"][0], 3), round(x["run_audio"][1], 3)) for x in items})
    dump(op, d)
    print(f"[moss] {split} gen done {len(todo)} in {time.time() - t1:.0f} s; forced {sum(x['moss_forced'] for x in items)}; "
          f"max GPU mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB -> {op}", flush=True)


# ============================================================================= step 2: matching (msproj, the Qwen/AF stack)
def lv(split):
    root = CFG[split]["root"].resolve()
    sys.path.insert(0, str(root))
    from benchmark.gold import listener_variants as LV
    from benchmark.gold import listener_afnext as AF
    assert Path(LV.__file__).resolve().is_relative_to(root), (LV.__file__, root)
    assert LV.V4_Q == V4_Q and LV.V4_NEW == V4_NEW and LV.SR == SR
    LV.S.load_gold = LV._no_gold
    return LV, AF


def match(split, smoke):
    import torch
    from sentence_transformers import SentenceTransformer
    LV, AF = lv(split)
    cfg, op = CFG[split], outp(split, smoke)
    d = json.loads(op.read_text(encoding="utf-8"))
    items = d["items"]
    assert all("moss_v4_text" in x for x in items), "gen not finished"
    O = LV.Onto()
    emb = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda" if torch.cuda.is_available() else "cpu")
    for it in items:
        it["accept"], it["null_accept"] = {}, {}
        for tag, fam in (("x", it["family"]), ("null", it.get("null_family"))):
            if not fam:
                continue
            m, ok = AF.v4_match(O, emb, it["moss_v4_text"], fam)
            it[f"moss_v4_{tag}"] = m
            (it["accept"] if tag == "x" else it["null_accept"])["V4"] = ok
    d["_meta"].update({"V4_rule": "some answer line matches X (listener_afnext.v4_match: match_names word match, else "
                       f"all-mpnet-base-v2 cosine > {LV.COS})", "match_root": str(cfg["root"]),
                       "null": "the item's cached null_family on the same text"})
    dump(op, d)
    print(f"[match] {split}: {len(items)} items -> {op}", flush=True)
    src = json.loads(cfg["items"].read_text(encoding="utf-8"))["items"]
    qv = [x for x in src if x["pool"] in cfg["pools"]]
    af = [x for x in json.loads(cfg["afn"].read_text(encoding="utf-8"))["items"] if x["pool"] in cfg["pools"]] \
        if cfg["afn"] and cfg["afn"].exists() else []
    for name, s, fld in (("Qwen", qv, "v4_text"), ("AF", af, "afn_v4_text")):   # rule identity, flags only, no gold
        s = [x for x in s if fld in x and "V4" in x.get("accept", {})]
        bad = sum(AF.v4_match(O, emb, x[fld], x["family"])[1] != x["accept"]["V4"] for x in s)
        if s:
            print(f"[check] {split} {name} V4 re-derived with this code: {len(s) - bad}/{len(s)} identical", flush=True)
    print(f"[dev] {split} n {len(items)}: MOSS V4 {sum(x['accept'].get('V4', False) for x in items)}, null accept "
          f"{sum(x['null_accept'].get('V4', False) for x in items)}, forced {sum(x.get('moss_forced', False) for x in items)}, "
          f"empty answers {sum(not x['moss_v4_text'].strip() for x in items)}, mean think tokens "
          f"{sum(x.get('moss_think_tokens', 0) for x in items) / max(1, len(items)):.0f}, mean answer lines "
          f"{sum(len([l for l in x['moss_v4_text'].splitlines() if l.strip()]) for x in items) / max(1, len(items)):.2f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("gen", "match"))
    ap.add_argument("split", choices=tuple(CFG))
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    (gen if a.step == "gen" else match)(a.split, a.smoke)


if __name__ == "__main__":
    main()
