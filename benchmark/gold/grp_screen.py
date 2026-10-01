"""Round 47 GROUP (Adam's idea, 2026-10-01): should two same-sound pictures a few seconds apart be ONE picture?
SHIP8's saved pictures are read at the shipped MERGE_GAP 2.5; every pair of consecutive same-family pictures whose gap is
(0, MAX_GAP] s is a candidate. GRP-A: Qwen3-Omni hears the stretch between them and says "same" continuing sound or a "new"
event (both option orders must say same). GRP-L: Qwen3-Omni, text only, says whether that sound type is usually one continuing
sound or separate events (both orders). A merged pair becomes one picture (first start, second end). Pre-registered in
docs/prereg_round13_detector_push.md (Round 47).

    TG_ARMS=SHIP8 python benchmark/gold/grp_screen.py pairs dev     (CPU)  -> grp/pairs_dev.json
    TG_ARMS=SHIP8 python benchmark/gold/grp_screen.py ask dev       (GPU)  -> grp/ask_dev.json
    TG_ARMS=SHIP8 python benchmark/gold/grp_screen.py score dev     (CPU)  -> grp/score_dev.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S

GAP, MAX_GAP, PRE, POST = 2.5, 8.0, 1.0, 1.5
DIR = _ROOT / "benchmark" / "gold" / "grp"
WAV = {"dev": _ROOT / "data" / "work" / "devcand" / "wav16", "dev2": _ROOT / "data" / "work" / "r13dev2" / "wav16",
       "test": _ROOT / "data" / "work" / "r13test" / "wav16", "test2": _ROOT / "data" / "work" / "r13test2" / "wav16"}
A_Q = ("You hear a {lab} sound near the start and again near the end of this recording. Is the {lab} near the end {o1}, or "
       "{o2}? Answer with exactly one word: {w1} or {w2}.")
A_OPT = {"same": "the same continuing sound as at the start (for example one alarm or engine with a short pause)",
         "new": "a new, separate event (for example a second bark, knock or shot)"}
L_Q = ("Think of the sound '{lab}'. Is it usually {o1}, or {o2}? Answer with exactly one word: {w1} or {w2}.")
L_OPT = {"continuing": "one continuing sound that can pause for a few seconds (like an alarm, engine, rain or a crowd)",
         "separate": "a series of separate short events (like barks, knocks, shots or footsteps)"}


def read_set(which):
    """SHIP8 pictures at GAP and gold, per clip, through merge_gap_sens's loaders (the scorer's own reading)"""
    from benchmark.gold import merge_gap_sens as MG
    name, log = {}, []
    lp, sc = S.load_pictures, S.score_clip

    def load(d, st, *a, **k):
        r = lp(d, st, *a, **k)
        if r is not None:
            name[id(r)] = (str(d), st)
        return r

    def score(gold, pics, *a, **k):
        log.append((gold, pics))
        return sc(gold, pics, *a, **k)

    S.load_pictures, S.score_clip = load, score
    rows = (MG.dev if which == "dev" else MG.test)(GAP)
    S.load_pictures, S.score_clip = lp, sc
    out = []
    for gold, pics in log:
        d, st = name.get(id(pics), ("?", "?"))
        if "B0r" in Path(d).name:
            continue
        part = ("dev2" if "dev2" in d or "tagger" in d else "dev") if which == "dev" else ("test2" if "test2" in d else "test")
        out.append({"clip": st, "part": part, "gold": gold, "pics": [[l, float(a), float(b)] for l, a, b in pics]})
    return rows["SHIP8"], out


def pairs_of(pics):
    ps = sorted(pics, key=lambda p: p[1])
    out = []
    for i, p in enumerate(ps):
        nxt = [q for q in ps[i + 1:] if S.same_family(q[0], p[0])]
        if nxt and 0 < nxt[0][1] - p[2] <= MAX_GAP:
            out.append({"lab": p[0], "a": [p[1], p[2]], "b": [nxt[0][1], nxt[0][2]], "gap": round(nxt[0][1] - p[2], 2)})
    return out


def cmd_pairs(which):
    base, clips = read_set(which)
    for c in clips:
        c["pairs"] = pairs_of(c["pics"])
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / f"pairs_{which}.json").write_text(json.dumps({"base": base, "clips": clips}, indent=1, default=float), encoding="utf-8")
    print(which, "base", base, "pairs", sum(len(c["pairs"]) for c in clips))


def cmd_ask(which):
    import soundfile as sf
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from benchmark.gold import listener_variants as LV
    clips = json.loads((DIR / f"pairs_{which}.json").read_text(encoding="utf-8"))["clips"]
    proc = Qwen3OmniMoeProcessor.from_pretrained(LV.MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(LV.MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass

    def gen(content, audio=None):
        conv = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[audio] if audio is not None else None, return_tensors="pt", padding=True,
                   use_audio_in_video=False).to(model.thinker.device)
        inp = inp.to(torch.bfloat16) if audio is not None else inp
        with torch.inference_mode():
            out = model.thinker.generate(**inp, max_new_tokens=4, do_sample=False)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().lower()

    res, labs = {"audio": {}, "label": {}}, set()
    for c in clips:
        if not c["pairs"]:
            continue
        w, sr = sf.read(str(WAV[c["part"]] / f"{c['clip']}.wav"), dtype="float32")
        assert sr == LV.SR, (c["clip"], sr)
        for p in c["pairs"]:
            labs.add(p["lab"])
            s, e = max(0.0, p["a"][1] - PRE), min(len(w) / sr, p["b"][0] + POST)
            cut = w[int(s * sr):int(e * sr)]
            ans = []
            for w1, w2 in (("same", "new"), ("new", "same")):
                q = A_Q.format(lab=p["lab"].lower(), o1=A_OPT[w1], o2=A_OPT[w2], w1=w1, w2=w2)
                ans.append(gen([{"type": "audio", "audio": cut}, {"type": "text", "text": q}], cut))
            k = f"{c['clip']}|{p['lab']}|{p['a'][0]:.2f}"
            res["audio"][k] = ans
            print(which, k, p["gap"], ans, flush=True)
    for lab in sorted(labs):
        ans = []
        for w1, w2 in (("continuing", "separate"), ("separate", "continuing")):
            q = L_Q.format(lab=lab.lower(), o1=L_OPT[w1], o2=L_OPT[w2], w1=w1, w2=w2)
            ans.append(gen([{"type": "text", "text": q}]))
        res["label"][lab] = ans
        print("label", lab, ans, flush=True)
    (DIR / f"ask_{which}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def merged(pics, pairs, keep):
    ps = sorted([list(p) for p in pics], key=lambda p: p[1])
    for p in pairs:
        if not keep(p):
            continue
        first = next((q for q in ps if q[0] == p["lab"] and abs(q[1] - p["a"][0]) < 1e-6), None)
        second = next((q for q in ps if S.same_family(q[0], p["lab"]) and abs(q[1] - p["b"][0]) < 1e-6), None)
        if first is None or second is None:
            continue
        first[2] = max(first[2], second[2])
        ps.remove(second)
    return [tuple(p) for p in ps]


def cmd_score(which):
    from benchmark.gold import dev_candidates_check as DCC
    clips = json.loads((DIR / f"pairs_{which}.json").read_text(encoding="utf-8"))["clips"]
    ask = json.loads((DIR / f"ask_{which}.json").read_text(encoding="utf-8"))
    yes = lambda a, w: all(x.startswith(w) for x in a)
    A = lambda c: lambda p: yes(ask["audio"].get(f"{c['clip']}|{p['lab']}|{p['a'][0]:.2f}", []), "same")
    L = lambda c: lambda p: yes(ask["label"].get(p["lab"], []), "continuing")
    arms = {"base": lambda c: lambda p: False, "GRP-A": A, "GRP-L": L, "GRP-LA": lambda c: lambda p: A(c)(p) and L(c)(p)}
    res, cost = {"rows": {}, "parts": {}, "merged": {}}, {}
    for n, f in arms.items():
        rr, per = [], {}
        for c in clips:
            pics = merged(c["pics"], c["pairs"], f(c))
            r = S.score_clip(c["gold"], pics)
            rr.append(r); per.setdefault(c["part"], []).append(r)
            if n != "base":
                res["merged"].setdefault(n, []).extend(f"{c['clip']} {p['lab']} {p['a'][0]:.2f}+{p['b'][0]:.2f} (gap {p['gap']})"
                                                       for p in c["pairs"] if f(c)(p))
        m = DCC.metrics(rr)
        res["rows"][n] = {k: m[k] for k in ("hits", "misses", "wrong", "visible", "cross", "phantom", "viewer_cost")}
        res["parts"][n] = {pt: DCC.metrics(v)["viewer_cost"] for pt, v in per.items()}
        cost[n] = [DCC.clip_cost(r) for r in rr]
    import numpy as np
    res["d_vs_base"] = {n: DCC.boot(np.subtract(cost[n], cost["base"])) for n in arms if n != "base"}
    (DIR / f"score_{which}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    for n, x in res["rows"].items():
        print(f"{which} {n:7s} {x['hits']}/{x['hits'] + x['misses']} wrong {x['wrong']} ({x['visible']}/{x['cross']}/{x['phantom']})"
              f" cost {x['viewer_cost']:.3f} parts {res['parts'][n]} d {res['d_vs_base'].get(n)}")
    for n, v in res["merged"].items():
        print(n, "merged:", v)


if __name__ == "__main__":
    {"pairs": cmd_pairs, "ask": cmd_ask, "score": cmd_score}[sys.argv[1]](sys.argv[2])
