"""Round 51 GRP-P (docs/prereg_round13_detector_push.md "Round 51 GRP-P", written before any number of it): keep GRP-A (Omni
"same"/"new", both orders, picture-gap cap 8 s) and add a pause question on the same cut (seg1 end - 1.0 ... seg2 start + 1.5 s):
"Does the {lab} sound stop completely for more than 2 seconds before it is heard again?" (yes/no, both option orders).
GRP-P merges a pair iff same/same AND no/no. Nothing in src/ or config.py is edited.

Step 1 (held-out 415 AudioSet-Strong clips, calibration): every pair of consecutive same-label strong segments (overlapping or
touching same-label segments are first joined) with gap 0 < g <= 8 s; truth = g > 2.0. Bar: accuracy >= 0.75 on BOTH sides of
2 s (both orders must agree; disagreement or a non yes/no answer counts wrong). `score415` exits 3 below the bar.
Step 2 (merged DEV, only if step 1 passes): the SHIP8+MD3 candidate pairs exactly as the shipped group.apply sees them (captured by
wrapping group.apply while the scorer runs), pause question on the same cuts, then base (shipped cache) vs GRP-P scored through the
shipped display path (group._answers wrapped: a pair merges only if the cached answers are same/same and the pause answers no/no).

    python benchmark/gold/grpp_screen.py pairs415                    (CPU) -> grpp/pairs_415.json
    python benchmark/gold/grpp_screen.py ask415                      (GPU) -> grpp/ask_415.json (resumable)
    python benchmark/gold/grpp_screen.py score415                    (CPU) -> grpp/score_415.json ; exit 3 below the bar
    TG_ARMS="SHIP8+MD3" python benchmark/gold/grpp_screen.py pairsdev   (CPU) -> grpp/pairs_dev.json (+ base reproduction)
    TG_ARMS="SHIP8+MD3" python benchmark/gold/grpp_screen.py askdev     (GPU) -> grpp/ask_dev.json
    TG_ARMS="SHIP8+MD3" python benchmark/gold/grpp_screen.py scoredev   (CPU) -> grpp/score_dev.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

PRE, POST, MAX_GAP, T = 1.0, 1.5, 8.0, 2.0
BAR = 0.75
P_Q = ("Does the {lab} sound stop completely for more than 2 seconds before it is heard again? "
       "Answer with exactly one word: {w1} or {w2}.")
ORDERS = (("yes", "no"), ("no", "yes"))
DIR = _ROOT / "benchmark" / "gold" / "grpp"
HELDOUT = _ROOT / "benchmark" / "gold" / "audioset_heldout.json"
WAV415 = _ROOT / "benchmark" / "gold" / "heldout_a4" / "wav16"
WAVDEV = {"dev": _ROOT / "data" / "work" / "devcand" / "wav16", "dev2": _ROOT / "data" / "work" / "r13dev2" / "wav16"}
ARM = "SHIP8+MD3"
BINS = (("<=1", 0.0, 1.0), ("1-2", 1.0, 2.0), ("2-4", 2.0, 4.0), ("4-8", 4.0, 8.0), ("fuzzy 1.5-2.5", 1.5, 2.5))


def _dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    os.replace(tmp, path)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================================ Omni
def load_omni():
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from benchmark.gold import listener_variants as LV
    proc = Qwen3OmniMoeProcessor.from_pretrained(LV.MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(LV.MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass

    def gen(content, audio):
        conv = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[audio], return_tensors="pt", padding=True, use_audio_in_video=False)
        inp = inp.to(model.thinker.device).to(torch.bfloat16)
        with torch.inference_mode():
            out = model.thinker.generate(**inp, max_new_tokens=4, do_sample=False)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().lower()
    return gen, LV.SR


def pause_answers(gen, w, sr, lab, a_end, b_start):
    """both option orders on the GRP cut: a_end - PRE ... b_start + POST"""
    s, e = max(0.0, a_end - PRE), min(len(w) / sr, b_start + POST)
    cut = w[int(s * sr):int(e * sr)]
    out = []
    for w1, w2 in ORDERS:
        q = P_Q.format(lab=lab.lower(), w1=w1, w2=w2)
        out.append(gen([{"type": "audio", "audio": cut}, {"type": "text", "text": q}], cut))
    return out


def verdict(ans):
    """'yes' / 'no' if both orders agree on it, else None (no answer)"""
    v = ["yes" if x.startswith("yes") else "no" if x.startswith("no") else "?" for x in ans]
    return v[0] if len(v) == 2 and v[0] == v[1] and v[0] != "?" else None


# ============================================================================ step 1: held-out 415
def k415(p):
    return f"{p['clip']}|{p['lab']}|{p['a'][0]:.2f}|{p['b'][0]:.2f}"


def cmd_pairs415():
    from src.labels import SPEECH_LABELS, is_music
    H = _load(HELDOUT)
    assert len(H["clips"]) == 415
    pairs, joined, beyond = [], 0, 0
    for c in H["clips"]:
        by = {}
        for e in c["events"]:
            by.setdefault(e["label"], []).append(e)
        for lab, ev in by.items():
            ev = sorted(ev, key=lambda e: (float(e["start"]), float(e["end"])))
            segs = []                                        # join overlapping / touching same-label segments
            for e in ev:
                a, b, m = float(e["start"]), float(e["end"]), bool(e.get("masked"))
                if segs and a <= segs[-1][1]:
                    segs[-1][1] = max(segs[-1][1], b); segs[-1][2] = segs[-1][2] or m; joined += 1
                else:
                    segs.append([a, b, m])
            for x, y in zip(segs, segs[1:]):
                g = round(y[0] - x[1], 3)
                if g > MAX_GAP:
                    beyond += 1
                    continue
                pairs.append({"clip": c["id"], "stratum": c.get("stratum"), "lab": lab, "a": [x[0], x[1]], "b": [y[0], y[1]],
                              "gap": g, "truth": "yes" if g > T else "no", "masked": x[2] or y[2],
                              "speech_music": lab in SPEECH_LABELS or is_music(lab)})
    _dump(DIR / "pairs_415.json", {"rule": "consecutive same-label strong segments, overlapping/touching joined, 0 < gap <= 8",
                                   "joined": joined, "gap_over_8": beyond, "pairs": pairs})
    print(f"pairs415 {len(pairs)} (joined {joined}, >8 s {beyond}); g<=2 {sum(p['gap'] <= T for p in pairs)}, "
          f"g>2 {sum(p['gap'] > T for p in pairs)}", flush=True)


def cmd_ask415():
    import soundfile as sf
    pairs = _load(DIR / "pairs_415.json")["pairs"]
    out = DIR / "ask_415.json"
    res = _load(out) if out.exists() else {"prompt": P_Q, "orders": ORDERS, "pre": PRE, "post": POST, "answers": {}}
    todo = [p for p in pairs if k415(p) not in res["answers"]]
    print(f"ask415: {len(todo)} of {len(pairs)} pairs to ask", flush=True)
    if not todo:
        return
    gen, SR = load_omni()
    wav, n = {}, 0
    for p in todo:
        if p["clip"] not in wav:
            w, sr = sf.read(str(WAV415 / f"{p['clip']}.wav"), dtype="float32")
            assert sr == SR, (p["clip"], sr)
            wav = {p["clip"]: w}                              # pairs are grouped by clip; keep one clip in memory
        ans = pause_answers(gen, wav[p["clip"]], SR, p["lab"], p["a"][1], p["b"][0])
        res["answers"][k415(p)] = ans
        n += 1
        if n % 20 == 0 or n < 5:
            print(f"{k415(p)} gap {p['gap']} truth {p['truth']} -> {ans}", flush=True)
        if n % 100 == 0:
            _dump(out, res)
    _dump(out, res)
    print(f"ask415 done: {len(res['answers'])} answered", flush=True)


def summ(rows):
    n = len(rows)
    ok = sum(r["correct"] for r in rows)
    agree = sum(r["v"] is not None for r in rows)
    yes = sum(r["v"] == "yes" for r in rows)
    o1 = sum(r["o1"] == r["truth"] for r in rows)
    o2 = sum(r["o2"] == r["truth"] for r in rows)
    f = lambda x: round(x / n, 3) if n else None
    return {"n": n, "correct": ok, "acc": f(ok), "agree": f(agree), "agreed_yes": f(yes),
            "acc_order1_alone": f(o1), "acc_order2_alone": f(o2)}


def cmd_score415():
    P = _load(DIR / "pairs_415.json")
    A = _load(DIR / "ask_415.json")["answers"]
    rows = []
    for p in P["pairs"]:
        ans = A[k415(p)]
        v = verdict(ans)
        one = ["yes" if x.startswith("yes") else "no" if x.startswith("no") else "?" for x in ans]
        rows.append({**p, "ans": ans, "v": v, "o1": one[0], "o2": one[1], "correct": v == p["truth"]})
    res = {"what": "Round 51 GRP-P step 1: pause question on the held-out 415 strong-label pairs", "prompt": P_Q,
           "pairs": len(rows), "bar": BAR, "bins": {}, "sides": {}}
    for name, lo, hi in BINS:
        res["bins"][name] = summ([r for r in rows if lo < r["gap"] <= hi])
    res["sides"]["g<=2 (truth no)"] = summ([r for r in rows if r["gap"] <= T])
    res["sides"]["g>2 (truth yes)"] = summ([r for r in rows if r["gap"] > T])
    sub = {"no speech/music": [r for r in rows if not r["speech_music"]], "unmasked": [r for r in rows if not r["masked"]]}
    res["disclosure"] = {k: {"g<=2": summ([r for r in v if r["gap"] <= T]), "g>2": summ([r for r in v if r["gap"] > T])}
                         for k, v in sub.items()}
    a, b = res["sides"]["g<=2 (truth no)"]["acc"] or 0, res["sides"]["g>2 (truth yes)"]["acc"] or 0
    res["pass"] = bool(a >= BAR and b >= BAR)
    res["rows"] = rows
    _dump(DIR / "score_415.json", res)
    for k, v in {**res["bins"], **res["sides"]}.items():
        print(f"{k:18s} {v}", flush=True)
    for k, v in res["disclosure"].items():
        print("disclosure", k, v, flush=True)
    print(f"BAR (>= {BAR} both sides): {'PASS' if res['pass'] else 'FAIL'} ({a} / {b})", flush=True)
    sys.exit(0 if res["pass"] else 3)


# ============================================================================ step 2: merged DEV
MODE = {"m": "base", "part": None, "rec": None, "pause": {}}


def setup_dev():
    """the scorer path of floor_check.py none, with group.apply / group._answers wrapped"""
    assert ARM in os.environ.get("TG_ARMS", "").split(), "run with TG_ARMS=\"SHIP8+MD3\""
    import config
    from benchmark.gold import merged_dev as M
    from benchmark.gold import round13_dev as R
    from src.stage6_visual_augmentation import group as G
    R.ARMS[ARM]["PICTURE_MIN_CONF"] = None
    cfg = R.arm_cfg(ARM)
    assert cfg["GROUP_ASK"] and float(cfg["GROUP_MAX_GAP"]) == MAX_GAP, {k: cfg[k] for k in R.DISPLAY_KEYS}
    orig_apply, orig_ans = G.apply, G._answers

    def answers(clip):
        ans = orig_ans(clip)
        if MODE["m"] == "none":
            return {}
        if MODE["m"] == "grpp":
            out = {}
            for k, v in ans.items():
                p = MODE["pause"].get(f"{clip}|{k}")
                out[k] = list(v) + (["same" if str(x).lower().startswith("no") else f"pause:{x}" for x in p]
                                    if p else ["pause:missing"])
            return out
        return ans

    def apply(spans, clip):
        if MODE["rec"] is not None and getattr(config, "GROUP_ASK", False):
            cached = orig_ans(clip or getattr(config, "GROUP_CLIP", None))
            ps = G.pairs(sorted(spans, key=lambda p: p[1]), float(getattr(config, "GROUP_MAX_GAP", 4.0)))
            for a, b in ps:
                MODE["rec"].append({"clip": clip, "part": MODE["part"], "lab": a[0], "a": [float(a[1]), float(a[2])],
                                    "b_lab": b[0], "b": [float(b[1]), float(b[2])], "gap": round(float(b[1]) - float(a[2]), 2),
                                    "key": G.key(a[0], a[1]), "cache_file": str(getattr(config, "GROUP_CACHE", None)),
                                    "group_answers": cached.get(G.key(a[0], a[1]))})
        return orig_apply(spans, clip)

    G._answers, G.apply = answers, apply
    return M


def score_modes(M, modes):
    """per part, per mode: [(stem, score_clip row)]; every DEV call before any DEV2 call (tagger_prep.configure moves R.R13)"""
    out = {m: {} for m in modes}
    for part, fn in (("dev", M.rows_dev), ("dev2", M.rows_dev2)):
        for m in modes:
            MODE["m"], MODE["part"] = m, part
            out[m][part] = fn([ARM])[ARM]
    MODE["rec"] = None
    return out


def row_of(rr):
    from benchmark.gold import dev_candidates_check as DCC
    m = DCC.metrics(rr)
    return {k: m[k] for k in ("hits", "misses", "wrong", "visible", "cross", "phantom", "viewer_cost")}


def fmt(x):
    return f"{x['hits']}/{x['hits'] + x['misses']}, {x['wrong']} wrong ({x['visible']}/{x['cross']}/{x['phantom']}), cost {x['viewer_cost']:.3f}"


def cmd_pairsdev():
    M = setup_dev()
    MODE["rec"] = []
    rec = MODE["rec"]
    sc = score_modes(M, ["base"])
    MODE["rec"] = None
    base = row_of([r for p in ("dev", "dev2") for _s, r in sc["base"][p]])
    for p in rec:
        assert (WAVDEV[p["part"]] / f"{p['clip']}.wav").exists(), p
    _dump(DIR / "pairs_dev.json", {"arm": ARM, "max_gap": MAX_GAP, "base": base, "pairs": rec})
    print(f"pairsdev base {ARM}: {fmt(base)}; candidate pairs {len(rec)}", flush=True)
    for p in rec:
        print("  ", p["part"], p["clip"], p["key"], "gap", p["gap"], "GRP-A", p["group_answers"], flush=True)


def kdev(p):
    return f"{p['clip']}|{p['key']}"


def cmd_askdev():
    import soundfile as sf
    pairs = _load(DIR / "pairs_dev.json")["pairs"]
    gen, SR = load_omni()
    res = {"prompt": P_Q, "orders": ORDERS, "answers": {}}
    for p in pairs:
        w, sr = sf.read(str(WAVDEV[p["part"]] / f"{p['clip']}.wav"), dtype="float32")
        assert sr == SR, (p["clip"], sr)
        res["answers"][kdev(p)] = pause_answers(gen, w, SR, p["lab"], p["a"][1], p["b"][0])
        print("askdev", kdev(p), "gap", p["gap"], "GRP-A", p["group_answers"], "pause", res["answers"][kdev(p)], flush=True)
    _dump(DIR / "ask_dev.json", res)


def cmd_scoredev():
    import numpy as np
    from benchmark.gold import dev_candidates_check as DCC
    P = _load(DIR / "pairs_dev.json")
    MODE["pause"] = _load(DIR / "ask_dev.json")["answers"]
    M = setup_dev()
    sc = score_modes(M, ["none", "base", "grpp"])
    res = {"what": "Round 51 GRP-P step 2: merged DEV, SHIP8+MD3 (GRP-A shipped) vs GRP-P", "rows": {}, "parts": {}}
    flat = {m: [(p, s, r) for p in ("dev", "dev2") for s, r in sc[m][p]] for m in sc}
    for m, rr in flat.items():
        res["rows"][m] = row_of([r for _p, _s, r in rr])
        res["parts"][m] = {p: row_of([r for s, r in sc[m][p]]) for p in ("dev", "dev2")}
    B, G = res["rows"]["base"], res["rows"]["grpp"]
    hb = {(p, s): r["hit"] for p, s, r in flat["base"]}
    hg = {(p, s): r["hit"] for p, s, r in flat["grpp"]}
    lost = {f"{p}:{s}": hb[(p, s)] - hg[(p, s)] for (p, s) in hb if hg[(p, s)] < hb[(p, s)]}
    gained = {f"{p}:{s}": hg[(p, s)] - hb[(p, s)] for (p, s) in hb if hg[(p, s)] > hb[(p, s)]}
    cb = [DCC.clip_cost(r) for _p, _s, r in flat["base"]]
    cg = [DCC.clip_cost(r) for _p, _s, r in flat["grpp"]]
    res["d_grpp_vs_base"] = DCC.boot(np.subtract(cg, cb))
    res["hits_lost_clips"], res["hits_gained_clips"] = lost, gained
    merged = {}
    for p in P["pairs"]:
        a = p["group_answers"] or []
        pa = MODE["pause"].get(kdev(p), [])
        merged[kdev(p)] = {"gap": p["gap"], "GRP-A": a, "pause": pa,
                           "base_merges": bool(a) and all(str(x).lower().startswith("same") for x in a),
                           "grpp_merges": bool(a) and all(str(x).lower().startswith("same") for x in a)
                           and verdict(pa) == "no"}
    res["pairs"] = merged
    tie = (B["hits"], B["wrong"], B["visible"], B["cross"], B["phantom"]) == (G["hits"], G["wrong"], G["visible"], G["cross"], G["phantom"]) \
        and abs(B["viewer_cost"] - G["viewer_cost"]) < 1e-9
    n_lost = sum(lost.values())
    main = G["hits"] >= 28 and not lost and G["viewer_cost"] < B["viewer_cost"] - 1e-12
    fewer = G["viewer_cost"] < B["viewer_cost"] - 1e-12 and (B["wrong"] - G["wrong"]) >= 3 * n_lost and n_lost <= 3
    res["rule"] = {"tie": tie, "main_rule": main, "fewer_pictures_clause": fewer,
                   "tie_definition": "identical merged-DEV hits, wrong (visible/cross/phantom) and cost at w = 2",
                   "verdict": "GO (main rule)" if main else "GO (fewer-pictures clause)" if fewer
                   else "TIE -> GRP-P by the pre-registered tie-break" if tie else "STOP (GRP-A stays)"}
    _dump(DIR / "score_dev.json", res)
    for m in ("none", "base", "grpp"):
        print(f"merged DEV {m:5s} {fmt(res['rows'][m])}  parts dev {res['parts'][m]['dev']['viewer_cost']:.3f} "
              f"dev2 {res['parts'][m]['dev2']['viewer_cost']:.3f}", flush=True)
    for k, v in merged.items():
        print("  pair", k, v, flush=True)
    print("d grpp-base", res["d_grpp_vs_base"], "lost", lost, "gained", gained, flush=True)
    print("VERDICT", res["rule"], flush=True)


if __name__ == "__main__":
    {"pairs415": cmd_pairs415, "ask415": cmd_ask415, "score415": cmd_score415,
     "pairsdev": cmd_pairsdev, "askdev": cmd_askdev, "scoredev": cmd_scoredev}[sys.argv[1]]()
