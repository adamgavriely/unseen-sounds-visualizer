"""Round 31 SPOT (docs/prereg_round13_detector_push.md): SpotSound-A (LoRA `Loie/SpotSound` merged into
nvidia/audio-flamingo-3-hf) asked, on the exact cut the listeners heard, whether the candidate's family is present (EXIST)
and when (GROUND). SPOT-yes = EXIST not "no" AND a GROUND interval overlaps the candidate run +- 0.5 s.

    python benchmark/gold/spotsound_screen.py smoke OUT.json [--n 5]     # GPU, cluster: first N items of each part
    python benchmark/gold/spotsound_screen.py run OUT.json               # GPU, cluster: all P2/PV + P1 items, resumable
    python benchmark/gold/spotsound_screen.py screen RAW.json            # CPU: gold classes, GO/STOP -> spotsound_screen.json
Cuts are zero-padded to whole seconds (the timestamp-interleaved prompt has 25 audio tokens per second).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from src.labels import canonical

G = _ROOT / "benchmark" / "gold"
H = Path(os.path.expanduser("~"))
W = H / "MscProj" / "data" / "work"
PARTS = {  # part: (items json, wav dir, pools, gold annotations for the screen)
    "dev": (H / "MscProj_r13/benchmark/gold/dev_listener_v.json", W / "devcand/wav16", ("P2", "PV"), "gold_AG.json"),
    "dev2": (H / "MscProj_tg/benchmark/gold/dev2_listener_v.json", W / "r13dev2/wav16", ("P2", "PV"), "gold_AG.json"),
    "p1": (H / "MscProj_r13/benchmark/gold/dev_listener.json", W / "devcand/wav16", ("P1",), "gold_AG.json"),
}
BASE = "nvidia/audio-flamingo-3-hf"
ADAPTER = "Loie/SpotSound"
SR = 16000
NEW = 64
EXIST_Q = ("This is a sequence of audio stream. Your task is to identify whether the sound event in the query occurs. "
           "The query is: {q}. Answer: ")
GROUND_Q = ("This is a sequence of audio stream. Your task is to identify the temporal window (start and end timestamps) "
            "when the given query appears. The query is: {q} Answer: ")        # inference.py: prompt + phrase + ' Answer: '
SLACK = 0.5


def query(label):
    return canonical(label).lower()


def cut_of(it, n_clip):
    if "run_audio" in it:                                             # P2/PV: listener_v4d.py w[i:max(j, i + SR)]
        a, b = it["run_audio"]
        return a, max(b, a + 1.0)
    a = max(0.0, it["cut_start"] - 1.0)                               # P1: dev_listener.py line 206
    b = min(n_clip / SR, it["cut_end"] + 1.0)
    return a, max(b, a + 1.0)


def parse(txt):
    iv = [(float(a), float(b)) for a, b in re.findall(r"[Ff]rom\s+(\d+(?:\.\d+)?)\s*(?:seconds?|s)?\s*to\s+(\d+(?:\.\d+)?)", txt)]
    iv += [(float(a), float(b)) for a, b in re.findall(r"\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)", txt)]
    return [(a, b) for a, b in iv if b > a]                          # "from 0.000s to 0.000s" = no interval


def exist_yes(txt):
    return not txt.strip().lower().startswith("no")


def overlaps(ivs, cut_a, s, e):
    return any(cut_a + a <= e + SLACK and cut_a + b >= s - SLACK for a, b in ivs)


# ---------------------------------------------------------------- GPU
def load():
    import glob
    import torch
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download
    from transformers import AudioFlamingo3ForConditionalGeneration, AudioFlamingo3Processor

    class TSProc(AudioFlamingo3Processor):                            # SpotSound processor/af3.py, timestamp interleave
        def replace_audio_token(self, audio_inputs, audio_idx, **kw):
            n = int(audio_inputs["num_audio_tokens"][audio_idx])
            assert n % 25 == 0, n
            return "".join(f"timestamp: {t} seconds; feature: " + self.audio_token * 25 for t in range(n // 25))

    proc = TSProc.from_pretrained(BASE)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
    ad = snapshot_download(ADAPTER, allow_patterns=["adapter_*"])
    cfg = json.loads(Path(ad, "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(glob.glob(f"{ad}/adapter_model.safetensors")[0])
    params = dict(model.named_parameters())
    n = 0
    with torch.no_grad():
        for k in sd:
            if not k.endswith("lora_A.weight"):
                continue
            m = re.search(r"layers\.(\d+)\.(mlp|self_attn)\.(\w+_proj)\.lora_A", k)
            suf = f"layers.{m.group(1)}.{m.group(2)}.{m.group(3)}.weight"
            hits = [p for p in params if p.endswith(suf) and "language_model" in p]
            assert len(hits) == 1, (k, hits)
            A, B = sd[k].float(), sd[k.replace("lora_A", "lora_B")].float()
            P = params[hits[0]]
            P.add_((scale * (B.to(P.device) @ A.to(P.device))).to(P.dtype))
            n += 1
    assert n * 2 == len(sd), (n, len(sd))
    print(f"[spot] merged {n} LoRA pairs (scale {scale}) into {BASE}", flush=True)
    return proc, model


def gen(proc, model, w, text):
    import torch
    conv = [{"role": "user", "content": [{"type": "text", "text": text}, {"type": "audio", "audio": w}]}]
    inp = proc.apply_chat_template(conv, tokenize=True, add_generation_prompt=True, return_dict=True)
    inp = inp.to(model.device)
    inp["input_features"] = inp["input_features"].to(model.dtype)
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=NEW, do_sample=False)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def run(outp, n_smoke=None):
    import numpy as np
    import soundfile as sf
    proc, model = load()
    outp = Path(outp)
    res = json.loads(outp.read_text()) if outp.exists() else {"_meta": {}, "items": []}
    res["_meta"] = {"base": BASE, "adapter": ADAPTER, "exist_q": EXIST_Q, "ground_q": GROUND_Q, "new_tokens": NEW,
                    "parts": {p: str(v[0]) for p, v in PARTS.items()}, "pad": "zero-pad cut to whole seconds"}
    done = {(x["part"], x["clip"], x["pool"], x["label"], x["start"], x["end"]) for x in res["items"]}
    cache, t0 = {}, time.time()
    for part, (ip, wd, pools, _) in PARTS.items():
        items = [x for x in json.loads(ip.read_text(encoding="utf-8"))["items"] if x.get("pool") in pools]
        items = sorted(items, key=lambda x: (x["clip"], x["start"]))
        if n_smoke:
            items = items[:n_smoke]
        wav = {}
        for it in items:
            if (part, it["clip"], it["pool"], it["label"], it["start"], it["end"]) in done:
                continue
            if it["clip"] not in wav:
                wav.clear(); wav[it["clip"]] = sf.read(str(wd / f"{it['clip']}.wav"), dtype="float32")[0]
            w = wav[it["clip"]]
            a, b = cut_of(it, len(w))
            seg = w[int(a * SR):int(b * SR)]
            L = int(np.ceil(len(seg) / SR - 1e-6)) * SR
            seg = np.concatenate([seg, np.zeros(L - len(seg), np.float32)])
            rec = {k: it[k] for k in ("clip", "pool", "family", "label", "start", "end", "null_family", "gold", "peak")
                   if k in it} | {"part": part, "cut": [a, b], "accept_V4": bool(it.get("accept", {}).get("V4"))}
            for tag, lab in (("x", it["label"]), ("null", it.get("null_family"))):
                if not lab:
                    continue
                q = query(lab)
                for qn, tmpl in (("exist", EXIST_Q), ("ground", GROUND_Q)):
                    ck = (it["clip"], round(a, 3), round(b, 3), q, qn)
                    if ck not in cache:
                        cache[ck] = gen(proc, model, seg, tmpl.format(q=q))
                    rec[f"{tag}_{qn}"] = cache[ck]
                rec[f"{tag}_query"] = q
            res["items"].append(rec)
            if n_smoke:
                print(json.dumps(rec)[:600], flush=True)
            if len(res["items"]) % 50 == 0:
                outp.write_text(json.dumps(res, indent=0), encoding="utf-8")
                print(f"[spot] {len(res['items'])} items, {time.time() - t0:.0f} s", flush=True)
    outp.write_text(json.dumps(res, indent=0), encoding="utf-8")
    print(f"[spot] DONE {len(res['items'])} items, {len(cache)} generations, {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------- CPU screen
def flags(x, tag="x"):
    if f"{tag}_exist" not in x:
        return None
    ivs = parse(x[f"{tag}_ground"])
    a = x["cut"][0]
    ex = exist_yes(x[f"{tag}_exist"])
    return {"yes": ex and overlaps(ivs, a, x["start"], x["end"]), "any": ex and bool(ivs), "exist": ex,
            "ground_only": overlaps(ivs, a, x["start"], x["end"])}


def screen(rawp):
    from benchmark.gold import score_per_sound as S
    from benchmark.gold import dev_listener as L
    raw = json.loads(Path(rawp).read_text(encoding="utf-8"))["items"]
    golds = {p: S.load_gold([G / "annotations" / v[3]]) for p, v in PARTS.items()}
    out = {"a_added": {"needed": [], "other": []}, "b_removed": {"needed_lost": [], "other_removed": []},
           "c": {"spot": {"needed": 0, "other": 0}, "v4": {"needed": 0, "other": 0}},
           "side": {k: {"needed_add": 0, "other_add": 0, "needed_lost": 0, "other_removed": 0} for k in ("any", "ground_only")},
           "null_yes": {"cand": [0, 0], "p1": [0, 0]}, "p1": {}, "dropped_not_in_gold": 0, "n_cand": 0,
           "unparsed_ground": 0}
    p1 = {}
    for x in raw:
        f = flags(x)
        fn = flags(x, "null")
        if x["part"] == "p1":
            g = x.get("gold")
            p1.setdefault(g, [0, 0, 0])
            p1[g][0] += 1; p1[g][1] += f["yes"]; p1[g][2] += f["any"]
            if fn:
                out["null_yes"]["p1"][0] += 1; out["null_yes"]["p1"][1] += fn["yes"]
            continue
        gold = golds[x["part"]]
        if x["clip"] not in gold:
            out["dropped_not_in_gold"] += 1
            continue
        out["n_cand"] += 1
        if f["exist"] and not parse(x["x_ground"]):
            out["unparsed_ground"] += 1
        cls = "needed" if L.gold_class(gold[x["clip"]], x["label"], x["start"], x["end"]) == "hit_needed" else "other"
        s = f"{x['part']} {x['clip']} {x['pool']} {x['label']} {x['start']:.1f}-{x['end']:.1f}"
        v4 = x["accept_V4"]
        if f["yes"] and not v4:
            out["a_added"][cls].append(s)
        if v4 and not f["yes"]:
            out["b_removed"]["needed_lost" if cls == "needed" else "other_removed"].append(s)
        out["c"]["spot"][cls] += f["yes"]; out["c"]["v4"][cls] += v4
        for k in ("any", "ground_only"):
            sd = out["side"][k]
            if f[k] and not v4:
                sd[f"{cls}_add"] += 1
            if v4 and not f[k]:
                sd["needed_lost" if cls == "needed" else "other_removed"] += 1
        if fn:
            out["null_yes"]["cand"][0] += 1; out["null_yes"]["cand"][1] += fn["yes"]
    out["p1"] = {str(g): {"n": v[0], "spot_yes": v[1], "spot_any": v[2], "yes_rate": round(v[1] / v[0], 3)}
                 for g, v in p1.items()}
    na, oa = len(out["a_added"]["needed"]), len(out["a_added"]["other"])
    nl, orm = len(out["b_removed"]["needed_lost"]), len(out["b_removed"]["other_removed"])
    out["go_a"] = na >= 2 and oa <= na
    out["go_b"] = orm >= 1 and orm >= 3 * nl
    print(json.dumps({k: v for k, v in out.items() if k not in ("a_added", "b_removed")}, indent=1))
    print(f"(a) added needed {na}, other {oa} -> {'GO' if out['go_a'] else 'STOP'}")
    for s in out["a_added"]["needed"]:
        print("    needed+", s)
    print(f"(b) V4 accepts SPOT-no: needed lost {nl}, other removed {orm} -> {'GO' if out['go_b'] else 'STOP'}")
    (G / "spotsound_screen.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode in ("run", "smoke"):
        n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else (5 if mode == "smoke" else None)
        run(sys.argv[2], n)
    else:
        screen(sys.argv[2])
