"""Round 58 TIGHT-CUT (docs/prereg_round13_detector_push.md "Round 58 TIGHT-CUT"; panelist 1's P3,
docs/review/panel_2026-10-01/p1_detection.md). The listener hears [cut - 1, cut + 1] s; for a refused band run the pad can let a
louder neighbour win. Rule: re-ask ONLY refused band runs on [run start - 0.25, run end] (no trailing pad, at least 0.6 s:
short windows grow BACKWARD; only when the start clips at 0 does the end move, to 0.6 s), same models (Qwen3-Omni, AF Next),
same V4 prompt and matcher (listener_variants.V4_Q, listener_afnext.v4_match); accept iff TIER(padded) OR TIER(tight), each
TIER computed on ONE cut at the run's cached peak (peak >= 0.6: Qwen V4; below: Qwen V4 AND AF V4). Downstream unchanged.

    held q|af                GPU  held-out 415 band runs (FlexSED 0.4 runs, gaps <= 0.24 merged, 0.5 <= peak < 0.8,
                                  depictable families), seeded subsample of 800 (random.Random(58)): padded + tight V4,
                                  one model per job -> tightcut_415/held_<model>.json
    devask q|af              GPU  DEV / DEV2 refused band items (P2, 0.5 <= peak < 0.8, TIER False) ->
                                  tightcut_ask_<split>_<model>.json
    gate                     CPU  step-1 verdict -> tightcut_415.json; exit 3 on STOP
    build                    CPU  override caches tightcut/{v,afn}_<split>.json + replay assert
    r13|tg|merged|floor|diff CPU/GPU  the arm SHIP8+MD3+TC registered, then round13_dev / tagger_prep / merged_dev /
                                  floor_check_arm / changed pictures (twinshort_dev pattern)
Run from ~/MscProj_tg (held, devask, gate, build, tg, merged, floor, diff) or ~/MscProj_r13 (r13). TEST is never read.
"""
from __future__ import annotations

import json
import os
import runpy
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

HOME = Path.home()
TG, R13 = HOME / "MscProj_tg", HOME / "MscProj_r13"
OUTD = TG / "benchmark" / "gold"
SHARDS = OUTD / "tightcut_415"
OVR = OUTD / "tightcut"
ARM, BASE = "SHIP8+MD3+TC", "SHIP8+MD3"
PRE, MINW, SPLIT_PK, LO, FBAR, MIN_CUT, LSHORT = 0.25, 0.6, 0.6, 0.5, 0.8, 1.0, 0.5
GUARD, NMIN, DASM_BAR, DASM_PAD, N_HELD = 0.60, 5, 0.575, 0.5, 800
SR = 16000
CACHES = {"dev": {"v": R13 / "benchmark/gold/dev_listener_v.json", "afn": R13 / "benchmark/gold/dev_listener_afn.json",
                  "wav": HOME / "MscProj/data/work/devcand/wav16"},
          "dev2": {"v": TG / "benchmark/gold/dev2_listener_v.json", "afn": TG / "benchmark/gold/dev2_listener_afn.json",
                   "wav": HOME / "MscProj/data/work/r13dev2/wav16"}}


def tight(s, e, dur):
    a = max(0.0, min(s - PRE, e - MINW))
    b = e if e - a >= MINW - 1e-9 else min(dur, a + MINW)
    return a, b


def padded(s, e, pk_t, dur):
    """dev_listener P2 cut (1-s cut centred on the peak when the run is < 1 s) and its [cut - 1, cut + 1] audio"""
    if e - s < MIN_CUT:
        ca = min(max(0.0, pk_t - MIN_CUT / 2), max(0.0, dur - MIN_CUT))
        cs, ce = ca, ca + MIN_CUT
    else:
        cs, ce = s, e
    a0 = max(0.0, cs - 1.0); b0 = min(dur, ce + 1.0)
    return cs, ce, [a0, max(b0, a0 + 1.0)]


def tier(q, af, pk):
    return bool(q) if pk >= SPLIT_PK else bool(q) and bool(af)


def _key(x):
    return (x["clip"], x.get("pool"), x["family"], x.get("label"), round(x["start"], 2), round(x["end"], 2))


# ============================================================================= models (one at a time)
def qwen_model():
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from sentence_transformers import SentenceTransformer
    from benchmark.gold import listener_variants as LV
    proc = Qwen3OmniMoeProcessor.from_pretrained(LV.MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(LV.MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass
    emb = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")

    def prep(w, q):                                               # listener_variants.score prep(), unchanged
        conv = [{"role": "user", "content": [{"type": "audio", "audio": w}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[w], return_tensors="pt", padding=True, use_audio_in_video=False)
        return inp.to(model.thinker.device).to(torch.bfloat16) if hasattr(inp, "to") else inp

    def gen(w, q, n):                                             # listener_variants.score gen(), unchanged
        inp = prep(w, q)
        with torch.inference_mode():
            out = model.thinker.generate(**inp, max_new_tokens=n, do_sample=False)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    return {"gen": gen, "emb": emb, "model": model}


def afn_model():
    from benchmark.gold import listener_afnext as LA
    return LA.load_model()


def free(M):
    import gc
    import torch
    M.clear(); gc.collect(); torch.cuda.empty_cache()


def ask_all(items, wav_of, fields, model):
    """fields: [(item field with [a, b] window, out prefix)]; V4 text + match for ONE model (q = Qwen3-Omni, af = AF Next)"""
    import soundfile as sf
    from benchmark.gold import listener_variants as LV
    from benchmark.gold import listener_afnext as LA
    O = LV.Onto()
    for tag, load in [(t, l) for t, l in (("q", qwen_model), ("af", afn_model)) if t == model]:
        t0 = time.time()
        M = load()
        print(f"[ask] {tag} loaded in {time.time() - t0:.0f} s; {len(items)} items", flush=True)
        cache, txt = {}, {}
        for n, it in enumerate(sorted(items, key=lambda x: (x["clip"], x["start"])), 1):
            if it["clip"] not in cache:
                w, sr = sf.read(str(wav_of(it["clip"])), dtype="float32")
                assert sr == SR and w.ndim == 1, (it["clip"], sr, w.shape)
                cache.clear(); cache[it["clip"]] = w
            w = cache[it["clip"]]
            for fld, pre in fields:
                if f"{pre}_{tag}_v4" in it:
                    continue
                a, b = it[fld]
                seg = LA.cut(w, a, b) if fld == "pad_audio" else w[int(a * SR):int(b * SR)]   # tight: no forward extension
                if fld != "pad_audio":
                    assert len(seg) >= int(MINW * SR) - 2 or b - a < MINW, (it["clip"], a, b, len(seg))
                k = (it["clip"], fld, round(a, 3), round(b, 3))
                if k not in txt:
                    try:
                        txt[k] = M["gen"](seg, LV.V4_Q, LV.V4_NEW)
                    except Exception as ex:                  # pre-registered fallback: zero-pad (silence) to 1 s
                        print(f"[ask] {tag} {it['clip']} {a:.2f}-{b:.2f} ({len(seg) / SR:.2f} s) failed ({ex}); zero-pad",
                              flush=True)
                        txt[k] = M["gen"](np.concatenate([seg, np.zeros(max(0, SR - len(seg)), np.float32)]),
                                          LV.V4_Q, LV.V4_NEW)
                        it[f"{pre}_{tag}_zeropad"] = True
                m, ok = LA.v4_match(O, M["emb"], txt[k], it["family"])
                it[f"{pre}_{tag}_text"], it[f"{pre}_{tag}_match"], it[f"{pre}_{tag}_v4"] = txt[k], m, ok
            if n <= 2:
                print(f"[ask] {tag} smoke {it['clip']} {it['label']} {it['start']:.2f}-{it['end']:.2f} " +
                      " ".join(f"{p}={it[f][0]:.2f}-{it[f][1]:.2f}:{it[f'{p}_{tag}_v4']}:{it[f'{p}_{tag}_text'][:60]!r}"
                               for f, p in fields), flush=True)
            if n % 100 == 0:
                print(f"[ask] {tag} {n}/{len(items)} ({time.time() - t0:.0f} s)", flush=True)
                yield None
        free(M)
    yield None


# ============================================================================= step 1: held-out 415 (GPU, sharded)
def held_items():
    import soundfile as sf
    from benchmark.gold import heldout_a4_screen as H
    from benchmark.gold import dev_candidates_check as DCC
    from benchmark.gold import dev_listener as L
    from benchmark.gold import expect_screen as E
    from src.labels import canonical
    dep = set(E.FAMILIES)
    out = []
    for c in H.ids():
        fw, ts, labs = DCC.load_fr(H.FLEX_HELD / f"{c}.npz")
        dur = float(sf.info(str(H.WAV / f"{c}.wav")).duration)
        for j, lab in enumerate(labs):
            fam = canonical(lab)
            if fam not in dep:
                continue
            rr, dt = L.runs(fw[:, j], ts, L.FLEX_BAR, L.FLEX_GAP)
            for i, k in rr:
                pk = float(fw[i:k, j].max())
                if not (LO <= pk < FBAR):
                    continue
                s, e = float(ts[i]), float(ts[k - 1] + dt)
                pk_t = float(ts[i + int(np.argmax(fw[i:k, j]))] + dt / 2)
                cs, ce, pa = padded(s, e, pk_t, dur)
                out.append({"clip": c, "label": lab, "family": fam, "start": s, "end": e, "peak": pk, "cut_start": cs,
                            "cut_end": ce, "pad_audio": pa, "tight_audio": list(tight(s, e, dur)), "dur": dur,
                            "onset": s if e - s >= LSHORT else cs, "span": [s, e] if e - s >= LSHORT else [cs, ce]})
    return out


def held_sample():
    """the pre-registered seeded subsample of the 415 band runs (random.Random(58), N_HELD runs, in enumeration order)"""
    import random
    allr = held_items()
    idx = sorted(random.Random(58).sample(range(len(allr)), min(N_HELD, len(allr))))
    return [allr[i] for i in idx], len(allr)


def held(model):
    from benchmark.gold import heldout_a4_screen as H
    from benchmark.gold import listener_variants as LV
    from benchmark.gold import score_per_sound as S
    S.load_gold = LV._no_gold
    SHARDS.mkdir(parents=True, exist_ok=True)
    p = SHARDS / f"held_{model}.json"
    items, nall = held_sample()
    if p.exists():
        done = {(_k(x)): x for x in json.loads(p.read_text(encoding="utf-8"))["items"]}
        items = [done.get(_k(x), x) for x in items]
    print(f"[held] {model}: {len(items)} of {nall} band runs, {len({x['clip'] for x in items})} clips", flush=True)
    meta = {"round": "58 TIGHT-CUT step 1", "model": model, "n": len(items), "of": nall, "sample": "random.Random(58)"}
    for _ in ask_all(items, lambda c: H.WAV / f"{c}.wav", [("pad_audio", "pad"), ("tight_audio", "tight")], model):
        p.write_text(json.dumps({"_meta": meta, "items": items}, indent=0), encoding="utf-8")
    print(f"[held] {model} done -> {p}", flush=True)


def _k(x):
    return (x["clip"], x["label"], round(x["start"], 3), round(x["end"], 3))


def gate():
    from benchmark.gold import heldout_a4_screen as H
    from benchmark.gold import expect_a4_screen as A4
    from benchmark.gold import dev_candidates_check as DCC
    from benchmark.gold.score_per_sound import same_family
    Hj = json.loads(H.HELDOUT.read_text(encoding="utf-8"))
    ev_of = {c["id"]: c["events"] for c in Hj["clips"]}
    ref, nall = held_sample()
    got = {_k(x): dict(x) for x in ref}
    for m in ("q", "af"):
        for x in json.loads((SHARDS / f"held_{m}.json").read_text(encoding="utf-8"))["items"]:
            got[_k(x)].update({k: v for k, v in x.items() if k.startswith(("pad_" + m + "_", "tight_" + m + "_"))})
    miss = [k for k in (_k(x) for x in ref) if not all(f in got[k] for f in ("pad_q_v4", "pad_af_v4", "tight_q_v4",
                                                                                "tight_af_v4"))]
    assert not miss, f"{len(miss)} band runs not asked: {miss[:3]}"
    rows = []
    dcache = {}
    for x in (got[_k(r)] for r in ref):
        tp = tier(x["pad_q_v4"], x["pad_af_v4"], x["peak"])
        tt = tier(x["tight_q_v4"], x["tight_af_v4"], x["peak"])
        if x["clip"] not in dcache:
            f = H.DASM_HELD / f"{x['clip']}.npz"
            dcache.clear(); dcache[x["clip"]] = DCC.load_fr(f) if f.exists() else None
        dm = A4.dasm_max(dcache[x["clip"]], x["family"], x["span"][0] - DASM_PAD, x["span"][1] + DASM_PAD)
        cls = H.classify(x["family"], x["onset"], ev_of[x["clip"]], same_family)
        ovl = any(same_family(x["family"], ev["label"]) and min(x["end"], ev["end"]) > max(x["start"], ev["start"])
                  for ev in ev_of[x["clip"]])
        rows.append({"clip": x["clip"], "label": x["label"], "family": x["family"], "start": round(x["start"], 2),
                     "end": round(x["end"], 2), "peak": round(x["peak"], 3), "tier_pad": tp, "tier_tight": tt,
                     "legs": [x["pad_q_v4"], x["pad_af_v4"], x["tight_q_v4"], x["tight_af_v4"]],
                     "dasm": None if dm is None else round(dm, 3), "class": cls, "overlap": ovl})

    def P(rs, f="class"):
        n = len(rs); c = sum((r[f] == "correct") if f == "class" else bool(r[f]) for r in rs)
        return {"n": n, "correct": c, "precision": round(c / n, 3) if n else None}
    new = [r for r in rows if not r["tier_pad"] and r["tier_tight"]]
    newf8 = [r for r in new if r["dasm"] is not None and r["dasm"] >= DASM_BAR]
    res = {"all band runs": P(rows), "padded accepted (shipped rescue)": P([r for r in rows if r["tier_pad"]]),
           "padded refused": P([r for r in rows if not r["tier_pad"]]),
           "NEW = refused padded, accepted tight (guard)": P(new),
           "NEW, overlap rule (reported)": P(new, "overlap"),
           "NEW with DASM >= 0.575 (F8 survivors, reported)": P(newf8),
           "NEW by peak >= 0.6": P([r for r in new if r["peak"] >= SPLIT_PK]),
           "NEW by peak < 0.6": P([r for r in new if r["peak"] < SPLIT_PK]),
           "padded accepted, tight refused (lost if tight replaced padded; not the rule)":
               P([r for r in rows if r["tier_pad"] and not r["tier_tight"]])}
    g = res["NEW = refused padded, accepted tight (guard)"]
    if g["n"] < NMIN:
        verdict = f"NOT JUDGED (n {g['n']} < {NMIN}); no STOP, DEV runs (disclosed)"
    elif g["precision"] >= GUARD:
        verdict = f"GO (precision {g['precision']} >= {GUARD})"
    else:
        verdict = f"STOP (precision {g['precision']} < {GUARD})"
    out = {"_meta": {"round": "58 TIGHT-CUT step 1", "guard": GUARD, "nmin": NMIN, "verdict": verdict, "sample": len(ref),
                     "of": nall}, "summary": res,
           "new": new, "rows": rows}
    (OUTD / "tightcut_415.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(f"[gate] {k}: {v}", flush=True)
    print(f"[gate] VERDICT {verdict}", flush=True)
    if verdict.startswith("STOP"):
        sys.exit(3)


# ============================================================================= DEV / DEV2 asks (GPU)
def refused(split):
    """P2 band items of the base cache whose TIER (at the cached peak) is a refusal: [(item, af item)]"""
    c = CACHES[split]
    V = [x for x in json.loads(c["v"].read_text(encoding="utf-8"))["items"] if x.get("pool") == "P2"]
    A = {_key(x): x for x in json.loads(c["afn"].read_text(encoding="utf-8"))["items"]}
    out = []
    for x in V:
        pk = float(x.get("peak", 1.0) if x.get("peak") is not None else 1.0)
        if not (LO <= pk < FBAR):
            continue
        af = A.get(_key(x))
        assert af is not None, _key(x)
        if not tier((x.get("accept") or {}).get("V4"), (af.get("accept") or {}).get("V4"), pk):
            out.append((x, af))
    return out


def ask_path(split, model):
    return OUTD / f"tightcut_ask_{split}_{model}.json"


def load_ask(split):
    out = {}
    for m in ("q", "af"):
        for x in json.loads(ask_path(split, m).read_text(encoding="utf-8"))["items"]:
            out.setdefault(_key(x), dict(x)).update({k: v for k, v in x.items() if k.startswith("tight_" + m + "_")})
    return out


def devask(model):
    import soundfile as sf
    from benchmark.gold import listener_variants as LV
    from benchmark.gold import score_per_sound as S
    S.load_gold = LV._no_gold
    for split in ("dev", "dev2"):
        p = ask_path(split, model)
        if p.exists() and all(f"tight_{model}_v4" in x for x in json.loads(p.read_text(encoding="utf-8"))["items"]):
            print(f"[devask] {split}: done before", flush=True); continue
        items = []
        for x, _af in refused(split):
            dur = float(sf.info(str(CACHES[split]["wav"] / f"{x['clip']}.wav")).duration)
            items.append({"clip": x["clip"], "pool": "P2", "family": x["family"], "label": x.get("label"), "start": x["start"],
                          "end": x["end"], "peak": x["peak"], "run_audio": x.get("run_audio"),
                          "tight_audio": list(tight(float(x["start"]), float(x["end"]), dur)), "dur": dur})
        meta = {"round": "58 TIGHT-CUT", "split": split, "n": len(items), "rule": "refused P2 band items (TIER False)"}
        print(f"[devask] {split}: {len(items)} refused band items", flush=True)
        wd = CACHES[split]["wav"]
        for _ in ask_all(items, lambda c, wd=wd: wd / f"{c}.wav", [("tight_audio", "tight")], model):
            p.write_text(json.dumps({"_meta": meta, "items": items}, indent=0), encoding="utf-8")


# ============================================================================= override caches + replay (CPU)
def ovr(split, kind):
    return OVR / f"{kind}_{split}.json"


def build():
    import shutil
    OVR.mkdir(parents=True, exist_ok=True)
    rep = {}
    for split in ("dev", "dev2"):
        c = CACHES[split]
        ask = load_ask(split)
        assert all("tight_af_v4" in x and "tight_q_v4" in x for x in ask.values()), split
        ref = {_key(x) for x, _a in refused(split)}
        assert ref == set(ask), (split, len(ref), len(ask))
        V = json.loads(c["v"].read_text(encoding="utf-8"))
        A = json.loads(c["afn"].read_text(encoding="utf-8"))
        aidx = {_key(x): i for i, x in enumerate(A["items"])}
        flips = []
        for x in V["items"]:
            k = _key(x)
            if x.get("pool") != "P2" or k not in ask:
                continue
            t = ask[k]
            pk = float(x["peak"])
            if not tier(t["tight_q_v4"], t["tight_af_v4"], pk):
                continue
            # TIER(tight) True: set exactly the legs _tier reads at this peak to the tight answers
            x["accept"] = {**x["accept"], "V4": True}
            x["tightcut"] = {"q_text": t["tight_q_text"], "af_text": t["tight_af_text"], "window": t["tight_audio"]}
            if pk < SPLIT_PK:
                ai = A["items"][aidx[k]]
                ai["accept"] = {**ai["accept"], "V4": True}
                ai["tightcut"] = {"af_text": t["tight_af_text"], "window": t["tight_audio"]}
            flips.append([x["clip"], x.get("label"), round(x["start"], 2), round(x["end"], 2), round(pk, 3),
                          t["tight_q_text"][:80], t["tight_af_text"][:80]])
        V.setdefault("_meta", {})["tightcut"] = f"Round 58: {len(flips)} refused band items accepted on the tight cut"
        A.setdefault("_meta", {})["tightcut"] = V["_meta"]["tightcut"]
        ovr(split, "v").write_text(json.dumps(V), encoding="utf-8")
        ovr(split, "afn").write_text(json.dumps(A), encoding="utf-8")
        rep[split] = {"refused": len(ask), "accepted_tight": len(flips), "flips": flips}
        print(f"[build] {split}: refused {len(ask)}, accepted on the tight cut {len(flips)}", flush=True)
        for f in flips:
            print("   ", f, flush=True)
        replay(split, ask)
    (OUTD / "tightcut_build.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")


def replay(split, ask):
    """the stage-4 lookup on the override caches: TIER == TIER(pad) OR TIER(tight) for re-asked items; every other P2 / PV
    accept dict identical to the base caches"""
    import config
    from src import stage4_audio_event_detection as S4
    c = CACHES[split]
    V = [x for x in json.loads(c["v"].read_text(encoding="utf-8"))["items"] if x.get("pool") in ("P2", "PV")]
    old = config.LISTENER_AFCACHE if hasattr(config, "LISTENER_AFCACHE") else None
    n, bad = 0, []
    for clip in sorted({x["clip"] for x in V}):
        config.LISTENER_AFCACHE = str(c["afn"]); lb = S4.listener_from_vcache(str(c["v"]), clip)
        config.LISTENER_AFCACHE = str(ovr(split, "afn")); lo = S4.listener_from_vcache(str(ovr(split, "v")), clip)
        for x in V:
            if x["clip"] != clip:
                continue
            ctn = x["pool"] == "PV"
            b = lb(x["label"], x["start"], x["end"], contain=ctn)
            o = lo(x["label"], x["start"], x["end"], contain=ctn)
            if b is None or b["start"] != x["start"] or b.get("label") != x.get("label"):
                continue                                     # duplicate keys: the lookup's first match is another item
            n += 1
            k = _key(x)
            if k in ask:
                t = ask[k]
                want = bool(b["accept"]["TIER"]) or tier(t["tight_q_v4"], t["tight_af_v4"], float(x["peak"]))
                if bool(o["accept"]["TIER"]) != want or b["accept"]["TIER"]:
                    bad.append(("asked", k, b["accept"]["TIER"], o["accept"]["TIER"], want))
            elif o["accept"] != b["accept"]:
                bad.append(("other", k))
    config.LISTENER_AFCACHE = old
    assert not bad, bad[:5]
    print(f"[replay] {split}: {n} P2/PV lookups checked, override TIER == pad OR tight on re-asked, others identical",
          flush=True)


# ============================================================================= the arm (registered, then the existing steps)
def register(split):
    from benchmark.gold import round13_dev as R
    cfg = R.arm_cfg(BASE)
    assert cfg["LISTENER_RULE"] == "TIER" and not cfg.get("TIER_SPECIFIC") and not cfg.get("F8_BYPASS_BOTH"), cfg
    assert not cfg.get("TIER_HIGH_OR") and not cfg.get("TIER_2OF3_DASM") and not cfg.get("LISTENER_V4B_CACHE"), cfg
    assert not cfg.get("LISTENER_KCACHE") and float(cfg["TIER_SPLIT"]) == SPLIT_PK and float(cfg["LISTENER_LO"]) == LO
    assert float(cfg["FLEXSED_BAR"]) == FBAR, cfg["FLEXSED_BAR"]
    R.ARMS[ARM] = {**R.ARMS[BASE], "LISTENER_VCACHE": str(ovr(split, "v")), "LISTENER_AFCACHE": str(ovr(split, "afn"))}
    os.environ["TG_ARMS"] = " ".join(dict.fromkeys(os.environ.get("TG_ARMS", "").split() + [BASE, ARM]))
    return R


def diff():
    from benchmark.gold import weakwitness_dev as W
    register("dev2")
    res = []
    for part, (gold, P, _c, _s) in W.parts([BASE, ARM]).items():
        for st in P[BASE]:
            ca, cb = W.classify(gold[st], P[BASE][st]), W.classify(gold[st], P[ARM][st])
            ka = {(l, round(a, 2), round(b, 2), k) for l, a, b, k in ca}
            kb = {(l, round(a, 2), round(b, 2), k) for l, a, b, k in cb}
            if ka != kb:
                res.append({"part": part, "clip": st, "only_base": sorted(ka - kb, key=lambda x: x[1]),
                            "only_arm": sorted(kb - ka, key=lambda x: x[1])})
                print(part, st, "\n   -", sorted(ka - kb, key=lambda x: x[1]), "\n   +", sorted(kb - ka, key=lambda x: x[1]),
                      flush=True)
    p = OUTD / "tightcut_diff.json"
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[diff] {len(res)} clips changed -> {p}", flush=True)


def main():
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "held":                                  # held q|af
        return held(rest[0])
    if cmd == "devask":                                # devask q|af
        return devask(rest[0])
    if cmd in ("gate", "build", "diff"):
        return {"gate": gate, "build": build, "diff": diff}[cmd]()
    if cmd == "r13":                                   # from ~/MscProj_r13: round13_dev.py <step> --arms ...
        R = register("dev")
        sys.argv = ["round13_dev.py"] + rest
        return R.main()
    if cmd == "tg":                                    # from ~/MscProj_tg: tagger_prep.py <step> --split dev2 --arms ...
        register("dev2")
        from benchmark.gold import tagger_prep as T
        sys.argv = ["tagger_prep.py"] + rest
        return T.main()
    if cmd == "merged":
        register("dev2")
        sys.argv = ["merged_dev.py"] + rest
        return runpy.run_path(str(_ROOT / "benchmark" / "gold" / "merged_dev.py"), run_name="__main__")
    if cmd == "floor":                                 # floor_check_arm.py <none|floor> <arm>
        register("dev2")
        sys.argv = ["floor_check_arm.py"] + rest
        return runpy.run_path(str(_ROOT / "benchmark" / "gold" / "floor_check_arm.py"), run_name="__main__")
    raise SystemExit(f"unknown step {cmd}")


if __name__ == "__main__":
    main()
