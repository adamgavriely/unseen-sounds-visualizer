"""Round 66 amendment (docs/prereg_round13_detector_push.md "Round 66 amendment"): external proposers B (OWLv2 label box) and
A (SSL-SaN audio-driven box) read with NAME-ALL's crop margin; arm V PRODUCTION-VETO (shipped named phrase, 6 frames); Round 62b
PRIOR (image-free per-label prior on Round 62's d). The binding Round 66 proposer selection is in `score` here.

    python benchmark/gold/nameall_ext.py run      # GPU, ~/MscProj -> gate_gold/nameall_ext_Qwen38-27B/
    python benchmark/gold/nameall_ext.py score    # CPU, ~/MscProj (after nameall.py run) -> benchmark/gold/nameall_ext_gold.json
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from benchmark.gold.nameall import CROP_Q, CROP_NQ, NEG, auroc    # noqa: E402

VQ = "Is this {named} making the {label} sound right now? Answer yes or no."
VNQ = "Is this {named} NOT making the {label} sound right now? Answer yes or no."
V_NAMED = (("bell_miami", "Bell"), ("b3_golf_course", "Whack, thwack"), ("ly_ambulance", None), ("as_explosion_XJ8lc3I6", "Gunshot, gunfire"))


def s_text(R, prompt):
    import torch
    text = R.proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": prompt}]}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
    inp = R.proc(text=[text], return_tensors="pt").to(R.mdl.device)
    with torch.inference_mode():
        lg = R.mdl(**inp).logits[0, -1].float()
    return float(lg[R.yes].max() - lg[R.no].max())


def run():
    import config
    import numpy as np
    import soundfile as sf
    import torch
    from PIL import Image
    from src.stage2_video_understanding import _sample_frames_at
    from src.stage2_video_understanding.owl import _load as owl_load
    from benchmark.gold import gate_gold as G
    from benchmark.gold import gate_sslsan as SS
    from benchmark.gold import detector_dry as DD
    from benchmark.gold.sign_gate import gate_times
    from benchmark.gold.box_gate import majority
    from benchmark.gold.som_gate import crops, phrase_of, OWL, BOX_BAR
    from benchmark.gold.logit_gate import Reader, prompts as p62
    config.use_v4("5"); config.VLM_THINKING = False
    judge = set(G.JUDGE100.read_text().split())
    names = {stem: name for name, stem, _ in G.gold_sounds()}
    OUT = G.OUT_DIR / "nameall_ext_Qwen38-27B"
    OUT.mkdir(parents=True, exist_ok=True)
    cf = OUT / "_label_consts.json"
    C = json.loads(cf.read_text(encoding="utf-8")) if cf.exists() else {}
    grey1 = Image.new("RGB", (224, 224), (128, 128, 128))
    grey6 = [grey1.copy() for _ in range(6)]
    R = Reader()
    omdl, oproc = owl_load(OWL, "cuda")
    try:                                     # SSL-SaN repo + checkpoint are no longer on the cluster (arm A dropped, disclosed)
        sm = SS.load_model("cuda")
    except Exception as e:
        print("ARM A DROPPED: SSL-SaN cannot load:", repr(e)[:200], flush=True)
        sm = None

    def consts(lab):
        if lab not in C:
            c = {"crop_s0": [R.s(CROP_Q.format(phrase="this thing", label=lab), [grey1])[0],
                             R.s(CROP_NQ.format(phrase="this thing", label=lab), [grey1])[0]],
                 "v_s0": [R.s(VQ.format(named="thing", label=lab), grey6)[0], R.s(VNQ.format(named="thing", label=lab), grey6)[0]],
                 "prior62": {q: s_text(R, qs[0]) - s_text(R, qs[1]) for q, qs in p62(lab).items()}}
            C[lab] = c
            cf.write_text(json.dumps(C, indent=1), encoding="utf-8")
        return C[lab]

    def crop_m(lab, phrase, img, box):
        cr = crops(img, [box])
        if not cr:
            return None
        c0 = consts(lab)["crop_s0"]
        sq, sn = R.s(CROP_Q.format(phrase=phrase, label=lab), [cr[0]]), R.s(CROP_NQ.format(phrase=phrase, label=lab), [cr[0]])
        return {"box": [round(v, 1) for v in box], "sQ": sq[0], "sN": sn[0], "m": (sq[0] - c0[0]) - (sn[0] - c0[1])}

    for f in sorted((G.OUT_DIR / "Qwen38-27B").glob("*.json")):
        if f.stem not in judge or (OUT / f.name).exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        p = G.clip_path(names.get(f.stem, d["clip"]))
        w = None
        if p is not None:
            w, sr = sf.read(str(DD.wav_for(p)), dtype="float32")
            assert sr == SS.SR, sr
            w = w.mean(1) if w.ndim > 1 else w
        for s in d["sounds"]:
            lab = s["label"]
            k = consts(lab)
            ph = phrase_of(lab)
            for st in s["stretches"]:
                fr = _sample_frames_at(p, gate_times(st["start"], st["end"])) if p else []
                ext = {"B": None, "A": None, "V": None}
                st["ext"] = ext
                if len(fr) < 2:
                    continue
                if ph:                                                         # B: OWLv2 label box, best over 6 frames
                    best = None
                    for i, img in enumerate(fr):
                        inp = oproc(text=[[ph]], images=img, return_tensors="pt").to("cuda")
                        with torch.no_grad():
                            o = omdl(**inp)
                        r = oproc.post_process_grounded_object_detection(o, threshold=BOX_BAR,
                                                                         target_sizes=torch.tensor([[img.height, img.width]]).to("cuda"))[0]
                        for sc, bx in zip(r["scores"], r["boxes"]):
                            if best is None or float(sc) > best[0]:
                                best = (float(sc), i, [float(v) for v in bx])
                    if best:
                        ext["B"] = crop_m(lab, ph, fr[best[1]], best[2])
                        if ext["B"]:
                            ext["B"].update({"score": best[0], "frame": best[1], "phrase": ph})
                a, b = float(st["start"]), float(st["end"])                    # A: SSL-SaN audio-driven box
                seg = w[int(max(0.0, a - 1.0) * SS.SR):int((b + 1.0) * SS.SR)] if w is not None else []
                if sm is not None and len(seg) >= SS.SR // 2:
                    M = SS.cosine_maps(sm, fr, SS.spectrogram(seg), "cuda")    # (n, 14, 14)
                    fi = int(M.flatten(1).max(1).values.argmax())
                    i, j = divmod(int(M[fi].argmax()), 14)
                    W, H = fr[fi].size
                    box = [max(0.0, (j - 1) / 14 * W), max(0.0, (i - 1) / 14 * H), min(W, (j + 2) / 14 * W), min(H, (i + 2) / 14 * H)]
                    ext["A"] = crop_m(lab, "thing", fr[fi], box)
                    if ext["A"]:
                        ext["A"].update({"frame": fi, "cell": [i, j], "peak": float(M[fi].max())})
                named = str(st.get("named") or "").strip()
                if majority(st) and named and named.lower() != "nothing":      # V: shipped named phrase, 6 frames
                    sq, sn = R.s(VQ.format(named=named, label=lab), fr), R.s(VNQ.format(named=named, label=lab), fr)
                    ext["V"] = {"named": named, "sQ": sq[0], "sN": sn[0], "yn": [sq[2], sn[2]],
                                "m": (sq[0] - k["v_s0"][0]) - (sn[0] - k["v_s0"][1])}
                print(f.stem, lab, st["start"], {x: None if not ext[x] else round(ext[x]["m"], 2) for x in "BAV"}, flush=True)
        (OUT / f.name).write_text(json.dumps(d, indent=1), encoding="utf-8")


def kth(vals):
    v = sorted([x if x is not None else NEG for x in vals], reverse=True)
    return v[len(v) // 2]


def two_sided(S, K):
    """Round 66 step 1 on per-sound key K -> best {'t_lo','t_hi','silenced','kept'} or None, plus REPLACE best"""
    fin = sorted({x[K] for x in S if x[K] != NEG})
    mids = [(a + b) / 2 for a, b in zip(fin, fin[1:])]
    if fin:
        mids = [fin[0] - 1.0] + mids + [fin[-1] + 1.0]
    bar = lambda s_, k_: (s_ >= 19 and k_ >= 32) or (s_ >= 15 and k_ >= 35)
    best = rep = None
    for lo in [NEG] + mids:
        for hi in mids + [float("inf")]:
            if lo > hi:
                continue
            sil = kept = 0
            for x in S:
                a = x[K]
                pred = (x["ship"] and not (a != NEG and a < lo)) or a > hi
                sil += x["seen"] and pred; kept += (not x["seen"]) and (not pred)
            if bar(sil, kept) and (best is None or (sil, kept, hi, -lo) > best[0]):
                best = ((sil, kept, hi, -lo), {"t_lo": lo, "t_hi": hi, "silenced": sil, "kept": kept})
    for t in mids + [float("inf")]:
        sil = sum(x["seen"] and x[K] > t for x in S); kept = sum((not x["seen"]) and not (x[K] > t) for x in S)
        if bar(sil, kept) and (rep is None or (sil, kept, t) > rep[0]):
            rep = ((sil, kept, t), {"t": t, "silenced": sil, "kept": kept})
    return (best[1] if best else None), (rep[1] if rep else None)


def score():
    from benchmark.gold import gate_gold as G
    from benchmark.gold.box_gate import gold_index, majority
    from benchmark.gold.logit_gate import pick_t
    gold = gold_index()
    EXT, NA, L62 = G.OUT_DIR / "nameall_ext_Qwen38-27B", G.OUT_DIR / "nameall_Qwen38-27B", G.OUT_DIR / "logit_Qwen38-27B"
    C = json.loads((EXT / "_label_consts.json").read_text(encoding="utf-8"))
    files = sorted(x for x in EXT.glob("*.json") if not x.name.startswith("_"))
    assert len(files) == 49, len(files)
    S, Vst, cnt = [], [], {"B": 0, "A": 0, "NA": 0, "union": 0, "stretches": 0}
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        na = json.loads((NA / f.name).read_text(encoding="utf-8"))
        lg = json.loads((L62 / f.name).read_text(encoding="utf-8"))
        for s, sn, sl in zip(d["sounds"], na["sounds"], lg["sounds"]):
            assert (s["label"], s["start"]) == (sn["label"], sn["start"]) == (sl["label"], sl["start"])
            g = gold.get((f.stem, s["label"], round(s["start"], 2)))
            if g is None or g["importance"] < 2:
                continue
            per = {"NA_rank1": [], "NA_max": [], "B": [], "A": [], "union": []}
            for st, stn in zip(s["stretches"], sn["stretches"]):
                e, n = st.get("ext") or {}, stn.get("na") or {}
                b = (e.get("B") or {}).get("m"); a = (e.get("A") or {}).get("m")
                per["NA_rank1"].append(n.get("rank1")); per["NA_max"].append(n.get("max"))
                per["B"].append(b); per["A"].append(a)
                u = [x for x in (n.get("max"), b, a) if x is not None]
                per["union"].append(max(u) if u else None)
                cnt["stretches"] += 1; cnt["B"] += b is not None; cnt["A"] += a is not None
                cnt["NA"] += n.get("max") is not None; cnt["union"] += bool(u)
                if e.get("V"):
                    Vst.append({"clip": f.stem, "label": s["label"], "start": s["start"], "stretch": st["start"], "seen": bool(g["seen"]),
                                "named": e["V"]["named"], "m": e["V"]["m"]})
            pri = C[s["label"]]["prior62"]
            dprime = {q: statistics.fmean([x["lg"][q]["d"] - pri[q] for x in sl["stretches"] if x.get("lg")]) for q in "ab"}
            vm = [((st.get("ext") or {}).get("V") or {}).get("m") for st in s["stretches"]]
            rec = {"clip": f.stem, "label": s["label"], "start": s["start"], "seen": bool(g["seen"]), "vm": vm,
                   "named": [st.get("named") for st in s["stretches"]],
                   "ship": all(majority(st) for st in s["stretches"]), "per": per, "d62p_a": dprime["a"], "d62p_b": dprime["b"],
                   "votes": [[st.get("name"), st.get("ab"), st.get("desc"), st.get("named")] for st in s["stretches"]]}
            for k, v in per.items():
                rec["A_" + k] = kth(v)
            S.append(rec)
    base = (sum(x["seen"] and x["ship"] for x in S), sum((not x["seen"]) and (not x["ship"]) for x in S))
    assert base == (16, 33), base
    au = {k: auroc([x["A_" + k] for x in S if x["seen"]], [x["A_" + k] for x in S if not x["seen"]])
          for k in ("NA_rank1", "NA_max", "B", "A", "union")}
    na_agg = "NA_max" if au["NA_max"] > au["NA_rank1"] else "NA_rank1"
    cands = [na_agg, "B", "A", "union"]
    order = {na_agg: 0, "B": 1, "A": 2, "union": 3}
    pick = max(cands, key=lambda k: (au[k], -order[k]))
    stop = au[pick] < 0.65
    print(f"coverage (stretches with a crop / {cnt['stretches']}): NAME-ALL {cnt['NA']}, B {cnt['B']}, A {cnt['A']}, union {cnt['union']}")
    print("STEP 0 AUROC:", {k: round(v, 3) for k, v in au.items()}, "| NAME-ALL agg", na_agg, "| picked", pick, "->", "STOP" if stop else "GO")
    best, rep = two_sided(S, "A_" + pick)
    step1 = "not run (step 0 STOP)" if stop else ("PASS" if best else "FAIL")
    print("STEP 1 (two-sided) on", pick, ":", step1, best, "| REPLACE (report)", rep)
    if best:
        for x in S:
            a = x["A_" + pick]
            pred = (x["ship"] and not (a != NEG and a < best["t_lo"])) or a > best["t_hi"]
            if pred != x["ship"]:
                print("   flip", x["clip"], x["label"], x["start"], "seen" if x["seen"] else "NEEDED", "silenced" if pred else "kept", round(a, 3))
    # ---- arm V
    av = auroc([v["m"] for v in Vst if v["seen"]], [v["m"] for v in Vst if not v["seen"]]) if Vst else None
    vstop = av is None or av < 0.65
    ms = sorted({v["m"] for v in Vst})
    tl = [NEG] + [(a + b) / 2 for a, b in zip(ms, ms[1:])] + ([ms[-1] + 1.0] if ms else [])
    nothing = lambda nm: str(nm or "").strip().lower() in ("", "nothing")

    def v_tally(t):
        sil = kept = 0
        for x in S:
            pred = x["ship"] and not any(nothing(nm) for nm in x["named"]) and all(m is None or m >= t for m in x["vm"])
            sil += x["seen"] and pred; kept += (not x["seen"]) and (not pred)
        return sil, kept

    vbest = None
    for t in tl:
        sil, kept = v_tally(t)
        if ((sil >= 19 and kept >= 32) or (sil >= 15 and kept >= 35)) and (vbest is None or (sil, kept, -t) > vbest[0]):
            vbest = ((sil, kept, -t), {"t_lo": t, "silenced": sil, "kept": kept})
    fixonly = v_tally(NEG)
    vstep1 = "not run (step 0 STOP)" if vstop else ("PASS" if vbest else "FAIL")
    print(f"ARM V: shipped-seen named stretches {len(Vst)} (seen-sound {sum(v['seen'] for v in Vst)} / needed-sound "
          f"{sum(not v['seen'] for v in Vst)}); AUROC {av if av is None else round(av, 3)} -> {'STOP' if vstop else 'GO'}; "
          f"FIX_GATE alone {fixonly}; step 1 {vstep1} {vbest[1] if vbest else None}")
    for c, l in V_NAMED:
        print("   V named", c, l, [(v["label"], v["stretch"], v["named"], round(v["m"], 3), "seen" if v["seen"] else "NEEDED")
                                    for v in Vst if v["clip"].startswith(c) and (l is None or v["label"] == l)])
    for v in sorted(Vst, key=lambda v: v["m"])[:12]:
        print("   V low", v["clip"], v["label"], v["stretch"], v["named"], round(v["m"], 3), "seen" if v["seen"] else "NEEDED")
    # ---- Round 62b PRIOR
    p62 = {}
    for q in "ab":
        A = auroc([x[f"d62p_{q}"] for x in S if x["seen"]], [x[f"d62p_{q}"] for x in S if not x["seen"]])
        r = {"auroc": A, "stop": A < 0.65}
        if not r["stop"]:
            r["step1_add_seen"] = pick_t([(x["seen"], x["ship"], x[f"d62p_{q}"]) for x in S], True)[0]
        p62[q] = r
    print("ROUND 62b PRIOR:", {q: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for q, r in p62.items()})
    W = {("as_explosion_XJ8lc3I6", None), ("b3_golf_course", "Whack, thwack"), ("bell_miami", "Bell"), ("b3_pet_shop", "Bird"),
         ("b3_aviary_birds", "Bird"), ("ambient_market_marrakech_3102", "Motorcycle"), ("ambient_weather_storm_7200", "Rain")}
    for x in S:
        if any(x["clip"] == c and (l is None or l == x["label"]) for c, l in W):
            print("witness", x["clip"], x["label"], x["start"], "seen" if x["seen"] else "NEEDED", "ship-sil" if x["ship"] else "ship-kept",
                  {k: (None if x["A_" + k] == NEG else round(x["A_" + k], 2)) for k in ("NA_rank1", "NA_max", "B", "A", "union")})
    out = _ROOT / "benchmark" / "gold" / "nameall_ext_gold.json"
    out.write_text(json.dumps({"coverage": cnt, "auroc": au, "na_agg": na_agg, "picked": pick, "stop": stop, "step1": step1,
                               "two_sided_best": best, "replace_best": rep, "V": {"auroc": av, "stop": vstop, "step1": vstep1,
                               "best": vbest[1] if vbest else None, "fix_gate_only": fixonly, "stretches": Vst},
                               "prior62": p62, "sounds": S}, indent=1, default=float), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1]]()
