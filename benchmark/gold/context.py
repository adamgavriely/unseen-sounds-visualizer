"""Round 59 CONTEXT (docs/prereg_round13_detector_push.md "Round 59 CONTEXT"; Adam, 1 Oct 17:16: "when you are not sure about
a sound, look 5 seconds before or after and try to understand what the sound was"). For an uncertain candidate the listeners
REFUSED, Qwen3-Omni (thinker, greedy) gets the audio of [onset - 5, onset + 5] s (clamped to the clip) AND 8 video frames
of the same window (as one video, fps = 8 / window length; image fallback fixed by a smoke call on a non-candidate 415 clip)
and an open question (QUESTION below, k = onset - window start, one decimal). Its answer is split by expect_a_screen.items_of
and mapped by the frozen expect_a_screen.map_item; the FIRST item that maps gives the context family; context-accept iff it
equals the candidate's family.

    ask 415 <shard 0|1>       GPU  the Round 58 seeded 415 sample (tightcut.held_sample(), 800 band runs), half per shard
    ask dev|dev2              GPU  DEV / DEV2 refused candidates: P2 band items (0.5 <= peak < 0.8) and PV items whose
                                   cached TIER (at the cached peak) is False -> context_ask_<split>.json
    gate                      CPU  step 1 verdict -> context_415.json; exit 3 on STOP
    build                     CPU  override caches context/{v,afn}_<split>.json (+ replay assert) and context/f8bypass.json
    r13|tg|merged|floor|diff  CPU  arms SHIP8+MD3+CTX (rule a) and SHIP8+MD3+CTXB (b: also bypasses F8) registered at run
                                   time beside SHIP8+MD3, then round13_dev / tagger_prep / merged_dev / floor_check_arm / diff
Run from ~/MscProj_tg (r13 from ~/MscProj_r13). TEST is never read.
"""
from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from benchmark.gold import tightcut as TC  # noqa: E402  (helpers + the R58 415 sample; nothing run at import)

HOME = Path.home()
TG, R13 = TC.TG, TC.R13
OUTD = TC.OUTD
OVR = OUTD / "context"
ARM, ARMB, BASE = "SHIP8+MD3+CTX", "SHIP8+MD3+CTXB", "SHIP8+MD3"
HALF, NFR, NEW = 5.0, 8, 64
GUARD, NMIN, DASM_BAR, DASM_PAD = 0.60, 5, 0.575, 0.5
SR = 16000
QUESTION = ("A sound starts about {k} seconds into this clip. Using what you hear and what you see anywhere in the clip, what "
            "is making that sound? Answer with a short sound name.")
VID = {"dev": [HOME / "MscProj/data/input/gold_dev", HOME / "MscProj/data/input/benchmark/seen_ambient",
               HOME / "MscProj/data/input/benchmark/unseen_ambient"],
       "dev2": [TG / "data/input/tagger_dev2"],
       "415": [HOME / "MscProj/data/input/audioset_heldout"]}


def video_of(split, clip):
    for d in VID[split]:
        for ext in (".mp4", ".webm", ".mkv"):
            p = d / f"{clip}{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(f"no video for {split} {clip}")


def window(onset, dur):
    a, b = max(0.0, onset - HALF), min(dur, onset + HALF)
    return a, b, round(onset - a, 1)


def frames(video, a, b, dur):
    from PIL import Image
    out = []
    with tempfile.TemporaryDirectory() as td:
        for i in range(NFR):
            t = min(a + (b - a) * (i + 0.5) / NFR, max(0.0, dur - 0.05))
            fp = Path(td) / f"f{i}.jpg"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                            "-vf", "scale=640:-2", str(fp)], capture_output=True, timeout=120)
            if fp.exists():
                out.append(Image.open(fp).convert("RGB").copy())
    if len(out) != NFR:                               # pad by repeating the last frame (counted)
        assert out, (video, a, b)
        out += [out[-1]] * (NFR - len(out))
    return out


def fam_of(answer):
    from benchmark.gold import expect_a_screen as A
    for it in A.items_of(answer):
        f = A.map_item(it)
        if f:
            return f
    return None


# ============================================================================= the model
def omni():
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from benchmark.gold import listener_variants as LV
    proc = Qwen3OmniMoeProcessor.from_pretrained(LV.MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(LV.MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass
    state = {"mode": None}

    def run(seg, imgs, q, mode, fps):
        if mode == "video":
            vid = np.stack([np.asarray(im) for im in imgs])
            conv = [{"role": "user", "content": [{"type": "video", "video": "frames"}, {"type": "audio", "audio": "a"},
                                                 {"type": "text", "text": q}]}]
            text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            inp = proc(text=text, audio=[seg], videos=[vid], return_tensors="pt", padding=True, use_audio_in_video=False,
                       fps=fps, do_sample_frames=False)
        else:
            conv = [{"role": "user", "content": [{"type": "image", "image": "f"} for _ in imgs] +
                     [{"type": "audio", "audio": "a"}, {"type": "text", "text": q}]}]
            text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            inp = proc(text=text, audio=[seg], images=imgs, return_tensors="pt", padding=True, use_audio_in_video=False)
        inp = inp.to(model.thinker.device)
        for k in list(inp.keys()):
            if torch.is_floating_point(inp[k]):
                inp[k] = inp[k].to(torch.bfloat16)
        with torch.inference_mode():
            out = model.thinker.generate(**inp, max_new_tokens=NEW, do_sample=False)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def smoke():
        """the input mode, fixed once on a NON-candidate: the first 415 clip, window [0, 10]"""
        import soundfile as sf
        from benchmark.gold import heldout_a4_screen as H
        c = sorted(H.ids())[0]
        w, sr = sf.read(str(H.WAV / f"{c}.wav"), dtype="float32")
        dur = len(w) / SR
        a, b, k = window(min(5.0, dur / 2), dur)
        imgs = frames(video_of("415", c), a, b, dur)
        q = QUESTION.format(k=k)
        try:
            ans = run(w[int(a * SR):int(b * SR)], imgs, q, "video", NFR / (b - a))
            state["mode"] = "video"
        except Exception as ex:
            print(f"[smoke] video input failed ({type(ex).__name__}: {ex}); images", flush=True)
            ans = run(w[int(a * SR):int(b * SR)], imgs, q, "images", None)
            state["mode"] = "images"
        print(f"[smoke] {c} mode={state['mode']} answer={ans!r}", flush=True)

    smoke()
    return {"ask": lambda seg, imgs, q, fps: run(seg, imgs, q, state["mode"], fps), "mode": state}


def ask_items(M, items, split, wav_of, save):
    import soundfile as sf
    wc, fc = {}, {}
    t0 = time.time()
    for n, it in enumerate(items, 1):
        if "ctx_answer" in it:
            continue
        c = it["clip"]
        if c not in wc:
            wc.clear(); fc.clear()
            w, sr = sf.read(str(wav_of(c)), dtype="float32")
            assert sr == SR and w.ndim == 1, (c, sr, w.shape)
            wc[c] = w
        w = wc[c]
        dur = len(w) / SR
        a, b, k = window(float(it["onset"]), dur)
        key = (round(a, 3), round(b, 3))
        if key not in fc:
            fc[key] = frames(video_of(split, c), a, b, dur)
        q = QUESTION.format(k=k)
        ans = M["ask"](w[int(a * SR):int(b * SR)], fc[key], q, NFR / (b - a))
        f = fam_of(ans)
        it.update({"ctx_window": [round(a, 3), round(b, 3)], "ctx_k": k, "ctx_answer": ans, "ctx_family": f,
                   "ctx_accept": f == it["family"], "ctx_mode": M["mode"]["mode"]})
        if n <= 3:
            print(f"[ask] {split} {c} {it['family']} onset {it['onset']:.2f} k={k} -> {ans!r} ({f})", flush=True)
        if n % 50 == 0:
            print(f"[ask] {split} {n}/{len(items)} ({time.time() - t0:.0f} s)", flush=True)
            save()
    save()


# ============================================================================= candidates
def onset_span(s, e, cs, ce):
    return (s, [s, e]) if e - s >= TC.LSHORT else (cs, [cs, ce])


def refused(split):
    """refused candidates of the base caches: P2 band items (0.5 <= peak < 0.8) and PV items, TIER False at the cached peak"""
    c = TC.CACHES[split]
    V = [x for x in json.loads(c["v"].read_text(encoding="utf-8"))["items"] if x.get("pool") in ("P2", "PV")]
    A = {TC._key(x): x for x in json.loads(c["afn"].read_text(encoding="utf-8"))["items"]}
    out = []
    for x in V:
        pk = float(x.get("peak", 1.0) if x.get("peak") is not None else 1.0)
        if x["pool"] == "P2" and not (TC.LO <= pk < TC.FBAR):
            continue
        af = A.get(TC._key(x))
        if not TC.tier((x.get("accept") or {}).get("V4"), ((af or {}).get("accept") or {}).get("V4"), pk):
            out.append((x, af))
    return out


def ask_path(split):
    return OUTD / f"context_ask_{split}.json"


def ask(which, shard=None):
    import soundfile as sf
    from benchmark.gold import listener_variants as LV
    from benchmark.gold import score_per_sound as S
    from benchmark.gold import heldout_a4_screen as H
    S.load_gold = LV._no_gold
    t0 = time.time()
    M = omni()
    print(f"[ask] Qwen3-Omni loaded + smoke in {time.time() - t0:.0f} s; mode {M['mode']['mode']}", flush=True)
    if which == "415":
        (OUTD / "context_415").mkdir(parents=True, exist_ok=True)
        p = OUTD / "context_415" / f"shard{shard}.json"
        ref, nall = TC.held_sample()
        items = [dict(x) for i, x in enumerate(ref) if i % 2 == int(shard)]
        if p.exists():
            done = {TC._k(x): x for x in json.loads(p.read_text(encoding="utf-8"))["items"]}
            items = [done.get(TC._k(x), x) for x in items]
        meta = {"round": "59 CONTEXT step 1", "shard": shard, "n": len(items), "question": QUESTION}
        save = lambda: p.write_text(json.dumps({"_meta": meta, "items": items}, indent=0), encoding="utf-8")
        ask_items(M, items, "415", lambda c: H.WAV / f"{c}.wav", save)
        print(f"[ask] 415 shard {shard} done -> {p}", flush=True)
        return
    for split in [which]:                              # "dev" or "dev2"
        p = ask_path(split)
        items = []
        for x, _af in refused(split):
            on, sp = onset_span(float(x["start"]), float(x["end"]), float(x["cut_start"]), float(x["cut_end"])) \
                if x["pool"] == "P2" else (float(x["start"]), [float(x["start"]), float(x["end"])])
            items.append({"clip": x["clip"], "pool": x["pool"], "family": x["family"], "label": x.get("label"),
                          "start": x["start"], "end": x["end"], "peak": x.get("peak"), "onset": on, "span": sp})
        if p.exists():
            done = {TC._key(x): x for x in json.loads(p.read_text(encoding="utf-8"))["items"]}
            items = [done.get(TC._key(x), x) for x in items]
        items.sort(key=lambda x: (x["clip"], x["onset"]))
        meta = {"round": "59 CONTEXT", "split": split, "n": len(items), "question": QUESTION,
                "rule": "refused P2 band items and PV items (TIER False at the cached peak)"}
        print(f"[ask] {split}: {len(items)} refused candidates "
              f"({sum(x['pool'] == 'P2' for x in items)} P2, {sum(x['pool'] == 'PV' for x in items)} PV)", flush=True)
        wd = TC.CACHES[split]["wav"]
        save = lambda p=p, items=items, meta=meta: p.write_text(json.dumps({"_meta": meta, "items": items}, indent=0),
                                                                encoding="utf-8")
        ask_items(M, items, split, lambda c, wd=wd: wd / f"{c}.wav", save)


# ============================================================================= step 1 gate (CPU)
def gate():
    from benchmark.gold import heldout_a4_screen as H
    from benchmark.gold import expect_a_screen as A
    from benchmark.gold import expect_a4_screen as A4
    from benchmark.gold import dev_candidates_check as DCC
    from benchmark.gold.bandlist_415 import fams_of, AFN_DIR
    from benchmark.gold.score_per_sound import same_family
    Hj = json.loads(H.HELDOUT.read_text(encoding="utf-8"))
    ev_of = {c["id"]: c["events"] for c in Hj["clips"]}
    ref, nall = TC.held_sample()
    got = {}
    for s in (0, 1):
        for x in json.loads((OUTD / "context_415" / f"shard{s}.json").read_text(encoding="utf-8"))["items"]:
            got[TC._k(x)] = x
    miss = [TC._k(x) for x in ref if "ctx_answer" not in got.get(TC._k(x), {})]
    assert not miss, f"{len(miss)} not asked: {miss[:3]}"
    r58 = {}
    for m in ("q", "af"):                                   # Round 58's padded asks on the same 800 (reported row only)
        f = TC.SHARDS / f"held_{m}.json"
        if f.exists():
            for x in json.loads(f.read_text(encoding="utf-8"))["items"]:
                if f"pad_{m}_v4" in x:
                    r58.setdefault(TC._k(x), {})[m] = x[f"pad_{m}_v4"]
    lists, dcache, rows = {}, {}, []
    for r in ref:
        x = got[TC._k(r)]
        c = x["clip"]
        if c not in lists:
            om = fams_of(json.loads((H.DIR / "listen" / f"{c}.json").read_text(encoding="utf-8"))["text"], A)
            af = AFN_DIR / f"{c}.json"
            lists[c] = (om, fams_of(json.loads(af.read_text(encoding="utf-8"))["text"], A) if af.exists() else None)
        om, af = lists[c]
        conf = x["family"] in om or (af is not None and x["family"] in af)
        if c not in dcache:
            f = H.DASM_HELD / f"{c}.npz"
            dcache.clear(); dcache[c] = DCC.load_fr(f) if f.exists() else None
        dm = A4.dasm_max(dcache[c], x["family"], x["span"][0] - DASM_PAD, x["span"][1] + DASM_PAD)
        pr = r58.get(TC._k(r), {})
        pad_tier = TC.tier(pr["q"], pr["af"], x["peak"]) if "q" in pr and "af" in pr else None
        rows.append({"clip": c, "label": x["label"], "family": x["family"], "start": round(x["start"], 2),
                     "end": round(x["end"], 2), "onset": round(x["onset"], 2), "peak": round(x["peak"], 3),
                     "confirmed_by_lists": conf, "af_list_missing": af is None, "pad_tier_r58": pad_tier,
                     "answer": x["ctx_answer"], "ctx_family": x["ctx_family"], "ctx_accept": x["ctx_accept"],
                     "ctx_same_family": bool(x["ctx_family"]) and same_family(x["family"], x["ctx_family"]),
                     "ctx_mode": x["ctx_mode"], "dasm": None if dm is None else round(dm, 3),
                     "class": H.classify(x["family"], x["onset"], ev_of[c], same_family)})

    def P(rs):
        n = len(rs); k = sum(r["class"] == "correct" for r in rs)
        return {"n": n, "correct": k, "precision": round(k / n, 3) if n else None}
    unc = [r for r in rows if not r["confirmed_by_lists"]]
    acc = [r for r in unc if r["ctx_accept"]]
    res = {"all 800 band runs (base rate)": P(rows),
           "unconfirmed by whole-clip lists (guard population, base rate)": P(unc),
           "GUARD: unconfirmed AND context-accept": P(acc),
           "unconfirmed, context-refused": P([r for r in unc if not r["ctx_accept"]]),
           "unconfirmed AND context-accept AND DASM >= 0.575 (F8 survivors, rule a)":
               P([r for r in acc if r["dasm"] is not None and r["dasm"] >= DASM_BAR]),
           "unconfirmed AND context-accept by peak >= 0.6": P([r for r in acc if r["peak"] >= TC.SPLIT_PK]),
           "unconfirmed AND context-accept by peak < 0.6": P([r for r in acc if r["peak"] < TC.SPLIT_PK]),
           "unconfirmed AND context same_family (reported, not the rule)": P([r for r in unc if r["ctx_same_family"]]),
           "R58 padded-TIER refused (reported)": P([r for r in rows if r["pad_tier_r58"] is False]),
           "R58 padded-TIER refused AND context-accept (reported)":
               P([r for r in rows if r["pad_tier_r58"] is False and r["ctx_accept"]]),
           "R58 padded-TIER accepted = shipped rescue (reported)": P([r for r in rows if r["pad_tier_r58"] is True]),
           "all 800 AND context-accept (reported)": P([r for r in rows if r["ctx_accept"]])}
    g = res["GUARD: unconfirmed AND context-accept"]
    if g["n"] < NMIN:
        verdict = f"NOT JUDGED (n {g['n']} < {NMIN}); no STOP, DEV runs (disclosed)"
    elif g["precision"] >= GUARD:
        verdict = f"GO (precision {g['precision']} >= {GUARD})"
    else:
        verdict = f"STOP (precision {g['precision']} < {GUARD})"
    modes = sorted({r["ctx_mode"] for r in rows})
    out = {"_meta": {"round": "59 CONTEXT step 1", "guard": GUARD, "nmin": NMIN, "verdict": verdict, "modes": modes,
                     "sample": len(ref), "of": nall}, "summary": res, "rows": rows}
    (OUTD / "context_415.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(f"[gate] {k}: {v}", flush=True)
    print(f"[gate] modes {modes}; VERDICT {verdict}", flush=True)
    if verdict.startswith("STOP"):
        sys.exit(3)


# ============================================================================= override caches + replay (CPU)
def ovr(split, kind):
    return OVR / f"{kind}_{split}.json"


def load_ask(split):
    return {TC._key(x): x for x in json.loads(ask_path(split).read_text(encoding="utf-8"))["items"]}


def build():
    OVR.mkdir(parents=True, exist_ok=True)
    rep, byp = {}, {}
    for split in ("dev", "dev2"):
        c = TC.CACHES[split]
        ask_ = load_ask(split)
        assert all("ctx_answer" in x for x in ask_.values()), split
        ref = {TC._key(x) for x, _a in refused(split)}
        assert ref == set(ask_), (split, len(ref), len(ask_))
        V = json.loads(c["v"].read_text(encoding="utf-8"))
        A = json.loads(c["afn"].read_text(encoding="utf-8"))
        aidx = {TC._key(x): i for i, x in enumerate(A["items"])}
        flips = []
        for x in V["items"]:
            k = TC._key(x)
            if x.get("pool") not in ("P2", "PV") or k not in ask_ or not ask_[k]["ctx_accept"]:
                continue
            t = ask_[k]
            pk = float(x.get("peak", 1.0) if x.get("peak") is not None else 1.0)
            x["accept"] = {**x["accept"], "V4": True}            # the legs _tier reads at this peak
            x["context"] = {"answer": t["ctx_answer"], "window": t["ctx_window"], "k": t["ctx_k"]}
            if pk < TC.SPLIT_PK:
                ai = A["items"][aidx[k]]
                ai["accept"] = {**ai["accept"], "V4": True}
                ai["context"] = x["context"]
            byp.setdefault(x["clip"], []).append([x["family"], round(t["span"][0], 3), round(t["span"][1], 3)])
            flips.append([x["clip"], x["pool"], x.get("label"), round(x["start"], 2), round(x["end"], 2), round(pk, 3),
                          t["ctx_answer"][:80]])
        V.setdefault("_meta", {})["context"] = f"Round 59: {len(flips)} refused candidates context-accepted"
        A.setdefault("_meta", {})["context"] = V["_meta"]["context"]
        ovr(split, "v").write_text(json.dumps(V), encoding="utf-8")
        ovr(split, "afn").write_text(json.dumps(A), encoding="utf-8")
        rep[split] = {"refused": len(ask_), "context_accepted": len(flips), "flips": flips,
                      "answers": [[x["clip"], x["pool"], x["family"], round(x["onset"], 2), x["ctx_answer"], x["ctx_family"]]
                                  for x in ask_.values()]}
        print(f"[build] {split}: refused {len(ask_)}, context-accepted {len(flips)}", flush=True)
        for f in flips:
            print("   ", f, flush=True)
        replay(split, ask_)
    (OVR / "f8bypass.json").write_text(json.dumps(byp, indent=1), encoding="utf-8")
    (OUTD / "context_build.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")


def replay(split, ask_):
    """stage-4 lookup on the override caches: TIER == TIER(base) OR context-accept on asked items; others identical"""
    import config
    from src import stage4_audio_event_detection as S4
    c = TC.CACHES[split]
    V = [x for x in json.loads(c["v"].read_text(encoding="utf-8"))["items"] if x.get("pool") in ("P2", "PV")]
    old = getattr(config, "LISTENER_AFCACHE", None)
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
                continue
            n += 1
            k = TC._key(x)
            if k in ask_:
                want = bool(b["accept"]["TIER"]) or bool(ask_[k]["ctx_accept"])
                if bool(o["accept"]["TIER"]) != want or b["accept"]["TIER"]:
                    bad.append(("asked", k, b["accept"]["TIER"], o["accept"]["TIER"], want))
            elif o["accept"] != b["accept"]:
                bad.append(("other", k))
    config.LISTENER_AFCACHE = old
    assert not bad, bad[:5]
    print(f"[replay] {split}: {n} P2/PV lookups checked, override TIER == base OR context on asked, others identical",
          flush=True)


# ============================================================================= arms (registered at run time)
def register(split):
    from benchmark.gold import round13_dev as R
    cfg = R.arm_cfg(BASE)
    assert cfg["LISTENER_RULE"] == "TIER" and not cfg.get("TIER_SPECIFIC") and not cfg.get("F8_BYPASS_BOTH"), cfg
    assert not cfg.get("TIER_HIGH_OR") and not cfg.get("TIER_2OF3_DASM") and not cfg.get("LISTENER_V4B_CACHE"), cfg
    assert not cfg.get("LISTENER_KCACHE") and float(cfg["TIER_SPLIT"]) == TC.SPLIT_PK
    assert float(cfg["LISTENER_LO"]) == TC.LO and float(cfg["FLEXSED_BAR"]) == TC.FBAR, cfg
    assert cfg.get("LISTENER_DASM_VOTE"), cfg
    R.ARMS[ARM] = {**R.ARMS[BASE], "LISTENER_VCACHE": str(ovr(split, "v")), "LISTENER_AFCACHE": str(ovr(split, "afn"))}
    R.ARMS[ARMB] = {**R.ARMS[ARM], "CONTEXT_F8_BYPASS": str(OVR / "f8bypass.json")}
    os.environ["TG_ARMS"] = " ".join(dict.fromkeys(os.environ.get("TG_ARMS", "").split() + [BASE, ARM, ARMB]))
    return R


def diff():
    from benchmark.gold import weakwitness_dev as W
    from src.labels import canonical
    register("dev2")
    ans = {}
    for split in ("dev", "dev2"):
        for x in load_ask(split).values():
            ans.setdefault(x["clip"], []).append([x["family"], round(x["onset"], 2), x["pool"], x["ctx_answer"],
                                                  x["ctx_family"], x["ctx_accept"]])
    res = []
    for arm in (ARM, ARMB):
        for part, (gold, P, _c, _s) in W.parts([BASE, arm]).items():
            for st in P[BASE]:
                ca, cb = W.classify(gold[st], P[BASE][st]), W.classify(gold[st], P[arm][st])
                ka = {(l, round(a, 2), round(b, 2), k) for l, a, b, k in ca}
                kb = {(l, round(a, 2), round(b, 2), k) for l, a, b, k in cb}
                if ka != kb:
                    fams = {canonical(x[0]) for x in ka ^ kb}
                    ctx = [a_ for a_ in ans.get(st, []) if a_[0] in fams]
                    res.append({"arm": arm, "part": part, "clip": st, "only_base": sorted(ka - kb, key=lambda x: x[1]),
                                "only_arm": sorted(kb - ka, key=lambda x: x[1]), "context_answers": ctx})
                    print(arm, part, st, "\n   -", sorted(ka - kb, key=lambda x: x[1]), "\n   +",
                          sorted(kb - ka, key=lambda x: x[1]), "\n   ctx", ctx, flush=True)
    p = OUTD / "context_diff.json"
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[diff] {len(res)} clip changes -> {p}", flush=True)


def main():
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "ask":                                    # ask 415 <0|1> | ask dev | ask dev2
        return ask(rest[0], rest[1] if len(rest) > 1 else None)
    if cmd in ("gate", "build", "diff"):
        return {"gate": gate, "build": build, "diff": diff}[cmd]()
    if cmd == "r13":
        R = register("dev")
        sys.argv = ["round13_dev.py"] + rest
        return R.main()
    if cmd == "tg":
        register("dev2")
        from benchmark.gold import tagger_prep as T
        sys.argv = ["tagger_prep.py"] + rest
        return T.main()
    if cmd == "merged":
        register("dev2")
        sys.argv = ["merged_dev.py"] + rest
        return runpy.run_path(str(_ROOT / "benchmark" / "gold" / "merged_dev.py"), run_name="__main__")
    if cmd == "floor":
        register("dev2")
        sys.argv = ["floor_check_arm.py"] + rest
        return runpy.run_path(str(_ROOT / "benchmark" / "gold" / "floor_check_arm.py"), run_name="__main__")
    raise SystemExit(f"unknown step {cmd}")


if __name__ == "__main__":
    main()
