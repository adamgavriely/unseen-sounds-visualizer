"""Round 66 NAME-ALL (docs/prereg_round13_detector_push.md "Round 66 NAME-ALL"): forced top-3 naming (no escape) + describe
noun phrases -> Round 50 grounding + crop (ViCrop rel-att crop where nothing grounds) -> null-calibrated logit crop margin
m = [s(Q+) - s0(Q+)] - [s(Q-) - s0(Q-)] (Q- = Round 50L's NOT twin, chosen by the (A) twin check). Steps 0-1 on gate-gold.

    python benchmark/gold/nameall.py run      # GPU, ~/MscProj -> gate_gold/nameall_Qwen38-27B/
    python benchmark/gold/nameall.py score    # CPU, ~/MscProj -> benchmark/gold/nameall_gold.json (exit 0 = step 1 PASS)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

N_PROMPT = ("A {label} sound is heard while these frames are on screen. List the 3 visible things most likely to be making it, "
            "most likely first, one short noun phrase per line. Always give 3, even if none fits well.")
CROP_Q = ("This is a close-up cut from a video frame. Is this {phrase} making the {label} sound right now? "
          "Answer yes or no.")
CROP_NQ = ("This is a close-up cut from a video frame. Is this {phrase} NOT making the {label} sound right now? "
           "Answer yes or no.")
GENERIC = "Describe this image briefly."
NEG = float("-inf")
WITNESS = {("as_explosion_XJ8lc3I6", None), ("b3_golf_course", "Whack, thwack"), ("bell_miami", "Bell"), ("b3_pet_shop", "Bird"),
           ("b3_aviary_birds", "Bird"), ("ambient_market_marrakech_3102", "Motorcycle"), ("ambient_weather_storm_7200", "Rain")}

DETS = set("a an the two three four five several some many his her their its one another this that these those few".split())
STOPW = set(("in on with at of to from by near under over behind through down up into onto across along around while and or but "
             "as is are was were be being been has have had which who whom whose there for against beside next during between "
             "inside outside above below beneath toward towards out off it they he she").split())


def parse_n(reply):
    from src.stage5_cross_modal_analysis.reason import _clean_phrase
    lines = [l.strip() for l in reply.splitlines() if l.strip()]
    if len(lines) < 3 and lines and re.search(r"[,;]", lines[0]):
        lines = [x.strip() for x in re.split(r"[,;]", lines[0]) if x.strip()] + lines[1:]
    out = []
    for l in lines[:3]:
        l = re.sub(r"^\s*(\d+\s*[\.\):]|[-*•])\s*", "", l).strip().strip("*").strip()
        p = _clean_phrase(l, max_words=5)
        if p:
            out.append(p)
    return out


def noun_phrases(text):
    toks = re.findall(r"[a-z][a-z'-]*|[,.;:!?]", (text or "").lower())
    out, i = [], 0
    while i < len(toks):
        if toks[i] in DETS:
            j, ph = i + 1, []
            while j < len(toks) and len(ph) < 4:
                w = toks[j]
                if not re.match(r"[a-z]", w) or w in STOPW or w in DETS or (ph and w.endswith("ing")):
                    break
                ph.append(w); j += 1
            if ph:
                out.append(" ".join(ph))
            i = j
        else:
            i += 1
    return out


def _norm(p):
    return re.sub(r"^(a|an|the)\s+", "", p.lower().strip())


def candidates(n_reply, desc_reply):
    seen, out = set(), []
    for src, ps in (("N", parse_n(n_reply)), ("desc", noun_phrases(desc_reply))):
        for p in ps:
            k = _norm(p)
            if k and k not in seen:
                seen.add(k); out.append({"phrase": p, "src": src})
    return out[:4]


def vicrop(R, img, lab):
    """ViCrop rel-att (2502.17422): last-token attention to the image tokens under Q+ ("the thing") / under a generic prompt,
    mean over heads and the middle third of the decoder layers, 3x3-smoothed; box 40 % x 40 % around the arg-max cell."""
    import torch
    import torch.nn.functional as F
    m = R.mdl
    pad = R.proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    merge = int(getattr(R.proc.image_processor, "merge_size", 2))
    prev = getattr(m.config, "_attn_implementation", "sdpa")

    def att(prompt):
        _, inp = R.inputs(prompt, [img])
        pos = (inp["input_ids"][0] == pad).nonzero().flatten()
        t, gh, gw = [int(v) for v in inp["image_grid_thw"][0]]
        h, w = gh // merge, gw // merge
        assert len(pos) == t * h * w, (len(pos), t, h, w)
        with torch.inference_mode():
            out = m(**inp, output_attentions=True)
        A = out.attentions
        sel = A[len(A) // 3: 2 * len(A) // 3]
        a = torch.stack([x[0, :, -1, pos].float().mean(0) for x in sel]).mean(0).reshape(t, h, w).mean(0)
        del out
        return a, h, w

    try:
        m.set_attn_implementation("eager")
        aq, h, w = att(CROP_Q.format(phrase="thing", label=lab))
        ag, _, _ = att(GENERIC)
    finally:
        m.set_attn_implementation(prev)
    rel = aq / (ag + 1e-6)
    rel = F.avg_pool2d(rel[None, None], 3, stride=1, padding=1, count_include_pad=False)[0, 0]
    i, j = divmod(int(rel.argmax()), w)
    W, H = img.size
    cx, cy = (j + 0.5) / w * W, (i + 0.5) / h * H
    bw, bh = 0.4 * W, 0.4 * H
    x0, y0 = max(0.0, min(W - bw, cx - bw / 2)), max(0.0, min(H - bh, cy - bh / 2))
    return [x0, y0, x0 + bw, y0 + bh], [i, j, h, w]


def run():
    import config
    from PIL import Image
    from src.stage2_video_understanding import _sample_frames_at
    from src.stage5_cross_modal_analysis import reason
    from benchmark.gold import gate_gold as G
    from benchmark.gold.sign_gate import gate_times
    from benchmark.gold.box_gate import parse_reprompt, to_pixels
    from benchmark.gold.som_gate import crops
    from benchmark.gold.humanbox_gate import GROUND_Q, CROP_Q as Q50
    from benchmark.gold.logit_gate import Reader
    assert Q50 == CROP_Q
    config.use_v4("5")
    config.VLM_THINKING = False
    judge = set(G.JUDGE100.read_text().split())
    names = {stem: name for name, stem, _ in G.gold_sounds()}
    OUT = G.OUT_DIR / "nameall_Qwen38-27B"
    OUT.mkdir(parents=True, exist_ok=True)
    s0f = OUT / "_s0.json"
    S0 = json.loads(s0f.read_text(encoding="utf-8")) if s0f.exists() else {}
    grey = Image.new("RGB", (224, 224), (128, 128, 128))
    R = None
    for f in sorted((G.OUT_DIR / "Qwen38-27B").glob("*.json")):
        if f.stem not in judge or (OUT / f.name).exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        p = G.clip_path(names.get(f.stem, d["clip"]))
        R = R or Reader()
        for s in d["sounds"]:
            lab = s["label"]
            if lab not in S0:
                S0[lab] = [R.s(CROP_Q.format(phrase="this thing", label=lab), [grey])[0],
                           R.s(CROP_NQ.format(phrase="this thing", label=lab), [grey])[0]]
                s0f.write_text(json.dumps(S0, indent=1), encoding="utf-8")
            s0q, s0n = S0[lab]
            for st in s["stretches"]:
                fr = _sample_frames_at(p, gate_times(st["start"], st["end"])) if p else []
                rec = {"n_frames": len(fr), "cands": [], "vicrop": None, "gen": 0, "prefill": 0}
                st["na"] = rec
                if len(fr) < 2:
                    continue
                rec["n_reply"] = reason._ask(R.mdl, R.proc, N_PROMPT.format(label=lab), images=fr, max_new=40)
                rec["gen"] += 1
                rec["desc_reply"] = reason._ask(R.mdl, R.proc, reason.DESCRIBE_PROMPT, images=fr, max_new=48)   # online: the gate's own call
                for c in candidates(rec["n_reply"], rec["desc_reply"]):
                    reply = reason._ask(R.mdl, R.proc, GROUND_Q.format(n=len(fr), label=lab, phrase=c["phrase"]), images=fr, max_new=64)
                    rec["gen"] += 1
                    status, parsed = parse_reprompt(reply, len(fr))
                    c.update({"status": status, "reply": reply})
                    if status == "box":
                        k, bb = parsed
                        px = to_pixels(bb, fr[k - 1])
                        cr = crops(fr[k - 1], [px])
                        if cr:
                            sq = R.s(CROP_Q.format(phrase=c["phrase"], label=lab), [cr[0]])
                            sn = R.s(CROP_NQ.format(phrase=c["phrase"], label=lab), [cr[0]])
                            rec["prefill"] += 2
                            c.update({"frame": k, "bbox_px": [round(v, 1) for v in px], "sQ": sq[0], "sN": sn[0],
                                      "yn": [sq[2], sn[2]], "m": (sq[0] - s0q) - (sn[0] - s0n)})
                        else:
                            c["status"] = "degenerate"
                    rec["cands"].append(c)
                if not any("m" in c for c in rec["cands"]):
                    try:
                        box, cell = vicrop(R, fr[2], lab)
                        cr = crops(fr[2], [box])
                        rec["prefill"] += 2
                        if cr:
                            sq = R.s(CROP_Q.format(phrase="thing", label=lab), [cr[0]])
                            sn = R.s(CROP_NQ.format(phrase="thing", label=lab), [cr[0]])
                            rec["prefill"] += 2
                            rec["vicrop"] = {"box": [round(v, 1) for v in box], "cell": cell, "sQ": sq[0], "sN": sn[0],
                                             "m": (sq[0] - s0q) - (sn[0] - s0n)}
                    except Exception as e:                    # counted as no crop
                        rec["vicrop"] = {"error": repr(e)[:300]}
                ms = [c["m"] for c in rec["cands"] if "m" in c]
                vm = (rec["vicrop"] or {}).get("m")
                rec["rank1"] = ms[0] if ms else vm
                rec["max"] = max(ms) if ms else vm
                print(f.stem, lab, st["start"], "| N", [c["phrase"] for c in rec["cands"]],
                      [(c.get("status"), None if "m" not in c else round(c["m"], 2)) for c in rec["cands"]],
                      "| vicrop", None if rec["vicrop"] is None else rec["vicrop"].get("m", rec["vicrop"].get("error")), flush=True)
        (OUT / f.name).write_text(json.dumps(d, indent=1), encoding="utf-8")


def auroc(p, n):
    return sum((a > b) + 0.5 * (a == b) for a in p for b in n) / (len(p) * len(n))


def A_of(sts, agg):
    v = sorted([((st.get("na") or {}).get(agg) if (st.get("na") or {}).get(agg) is not None else NEG) for st in sts], reverse=True)
    return v[len(v) // 2]


def score():
    import statistics
    from benchmark.gold import gate_gold as G
    from benchmark.gold.box_gate import gold_index, majority
    gold = gold_index()
    OUT = G.OUT_DIR / "nameall_Qwen38-27B"
    files = sorted(x for x in OUT.glob("*.json") if not x.name.startswith("_"))
    assert len(files) == 49, len(files)
    S, cov, n_st, vic, vic_err, yn, nreads, gens, pref = [], 0, 0, 0, 0, 0, 0, {}, {}
    crop_m = ([], [])
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for s in d["sounds"]:
            g = gold.get((f.stem, s["label"], round(s["start"], 2)))
            for st in s["stretches"]:
                na = st.get("na") or {}
                gens[f.stem] = gens.get(f.stem, 0) + na.get("gen", 0)
                pref[f.stem] = pref.get(f.stem, 0) + na.get("prefill", 0)
            if g is None or g["importance"] < 2:
                continue
            sts = s["stretches"]
            for st in sts:
                na = st.get("na") or {}
                n_st += 1
                cov += na.get("max") is not None
                v = na.get("vicrop") or {}
                vic += "m" in v; vic_err += "error" in v
                for c in na.get("cands", []):
                    if "m" in c:
                        yn += sum(c["yn"]); nreads += 2
                        crop_m[0 if g["seen"] else 1].append(c["m"])
            S.append({"clip": f.stem, "label": s["label"], "start": s["start"], "seen": bool(g["seen"]),
                      "ship": all(majority(st) for st in sts), "A_rank1": A_of(sts, "rank1"), "A_max": A_of(sts, "max"),
                      "stretches": [{"cands": [(c["phrase"], c["src"], c.get("status"), None if "m" not in c else round(c["m"], 3))
                                               for c in (st.get("na") or {}).get("cands", [])],
                                     "vicrop": ((st.get("na") or {}).get("vicrop") or {}).get("m"),
                                     "rank1": (st.get("na") or {}).get("rank1"), "max": (st.get("na") or {}).get("max")}
                                    for st in sts]})
    base = {"seen": sum(x["seen"] for x in S), "seen_sil": sum(x["seen"] and x["ship"] for x in S),
            "needed": sum(not x["seen"] for x in S), "needed_kept": sum(not x["seen"] and not x["ship"] for x in S)}
    assert base == {"seen": 41, "seen_sil": 16, "needed": 38, "needed_kept": 33}, base
    au = {a: auroc([x[f"A_{a}"] for x in S if x["seen"]], [x[f"A_{a}"] for x in S if not x["seen"]]) for a in ("rank1", "max")}
    agg = "max" if au["max"] > au["rank1"] else "rank1"
    stop = au[agg] < 0.65
    au_crop = auroc(*crop_m) if crop_m[0] and crop_m[1] else None
    print(f"base {base}; stretches {n_st}, with a crop {cov} (target >= 120), ViCrop crops {vic} (errors {vic_err}); "
          f"yes/no argmax {yn}/{nreads}; per-crop AUROC {au_crop}")
    print(f"STEP 0: AUROC rank-1 {au['rank1']:.3f} | max {au['max']:.3f} -> picked {agg} -> {'STOP' if stop else 'GO'}")
    K = f"A_{agg}"
    fin = sorted({x[K] for x in S if x[K] != NEG})
    mids = [(a + b) / 2 for a, b in zip(fin, fin[1:])]
    if fin:
        mids = [fin[0] - 1.0] + mids + [fin[-1] + 1.0]

    def tally(lo, hi):
        sil = kept = 0
        for x in S:
            a = x[K]
            pred = (x["ship"] and not (a != NEG and a < lo)) or a > hi
            sil += x["seen"] and pred; kept += (not x["seen"]) and (not pred)
        return sil, kept

    def bar(sil, kept):
        return (sil >= 19 and kept >= 32) or (sil >= 15 and kept >= 35)

    best, rep = None, None
    for lo in [NEG] + mids:
        for hi in mids + [float("inf")]:
            if lo > hi:
                continue
            sil, kept = tally(lo, hi)
            if bar(sil, kept):
                key = (sil, kept, hi, -lo)
                if best is None or key > best[0]:
                    best = (key, {"t_lo": lo, "t_hi": hi, "silenced": sil, "kept": kept})
    for t in mids + [float("inf")]:
        sil = sum(x["seen"] and x[K] > t for x in S); kept = sum((not x["seen"]) and not (x[K] > t) for x in S)
        if bar(sil, kept) and (rep is None or (sil, kept, t) > rep[0]):
            rep = ((sil, kept, t), {"t": t, "silenced": sil, "kept": kept})
    sel = None if stop else (best[1] if best else None)
    step1 = "not run (step 0 STOP)" if stop else ("PASS" if sel else "FAIL")
    print("STEP 1 two-sided:", step1, sel if not stop else (best[1] if best else None), "| REPLACE (report):", rep[1] if rep else None)
    if sel or (best and not stop):
        lo, hi = best[1]["t_lo"], best[1]["t_hi"]
        for x in S:
            a = x[K]
            pred = (x["ship"] and not (a != NEG and a < lo)) or a > hi
            if pred != x["ship"]:
                print("   flip", x["clip"], x["label"], x["start"], "seen" if x["seen"] else "NEEDED", "silenced" if pred else "kept", a)
    for x in S:
        if any(x["clip"] == c and (l is None or l == x["label"]) for c, l in WITNESS):
            print("witness", x["clip"], x["label"], x["start"], "seen" if x["seen"] else "NEEDED", "ship-silenced" if x["ship"] else
                  "ship-kept", "A_rank1", x["A_rank1"], "A_max", x["A_max"], [(y["cands"], y["vicrop"]) for y in x["stretches"]][:3])
    gv, pv = sorted(gens.values()), sorted(pref.values())
    print(f"cost per gate-gold clip: generations (naming + grounding) min {gv[0]} median {statistics.median(gv)} max {gv[-1]}; "
          f"prefill passes min {pv[0]} median {statistics.median(pv)} max {pv[-1]}")
    out = _ROOT / "benchmark" / "gold" / "nameall_gold.json"
    out.write_text(json.dumps({"base": base, "coverage": [cov, n_st], "vicrop": [vic, vic_err], "yn_argmax": [yn, nreads],
                               "auroc": au, "auroc_per_crop": au_crop, "agg": agg, "stop": stop, "step1": step1,
                               "two_sided_best": best[1] if best else None, "replace_best": rep[1] if rep else None,
                               "cost_gen": gens, "cost_prefill": pref, "sounds": S, "q": [N_PROMPT, CROP_Q, CROP_NQ]},
                              indent=1, default=float), encoding="utf-8")
    print(out)
    sys.exit(0 if sel else 3)


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1]]()
