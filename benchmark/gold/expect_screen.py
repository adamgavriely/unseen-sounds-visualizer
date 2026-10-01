"""Round 40 EXPECT screen (docs/prereg_round13_detector_push.md "Round 40 EXPECT"): the scene VLM proposes families one would
expect to HEAR off-screen in this place, the open-inventory listener (Qwen3-Omni, whole clip, unchanged V4 question) must name
the same family, the family must not already be drawn by SHIP8 in the clip, its onset is the earliest weak-bar run in any frame
cache, and the shipped visibility gate at that onset decides. Added pictures are 2 s. Scored dbr_screen-style on gold_AG.json vs
SHIP8 (28/58, 21 (6/13/2), 2.282). Nothing in src/ or config.py is edited.

    TG_ARMS=SHIP8 python benchmark/gold/expect_screen.py propose   # GPU: Qwen3.8-27B, 8 frames per clip -> expect/propose/
    TG_ARMS=SHIP8 python benchmark/gold/expect_screen.py listen    # GPU: Qwen3-Omni V4 on the whole clip -> expect/listen/
    TG_ARMS=SHIP8 python benchmark/gold/expect_screen.py cands     # CPU: intersection, already-drawn, onsets -> expect/cands.json
    TG_ARMS=SHIP8 python benchmark/gold/expect_screen.py gate      # GPU: shipped gate at the onset -> expect/gate/
    TG_ARMS=SHIP8 python benchmark/gold/expect_screen.py score     # CPU -> expect_screen.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")

B_ARM = "SHIP8"
MODEL = "Qwen/Qwen3.8-27B"                                       # the shipped gate VLM
BASE = {"hits": 28, "wrong": 21, "cost": 2.282}
DIR = _ROOT / "benchmark" / "gold" / "expect"
OUT = _ROOT / "benchmark" / "gold" / "expect_screen.json"
VOCAB = json.loads((_ROOT / "benchmark" / "gold" / "depictable_vocab.json").read_text(encoding="utf-8"))
FAMILIES = list(VOCAB["families"])
N_FRAMES, MAX_PROPOSE, PIC_LEN = 8, 5, 2.0
PROPOSE_Q = ("These {n} frames are spread over a whole video. Think about the place and what is going on. Name at most "
             "{k} sounds a person standing there would most likely HEAR but whose maker is NOT visible in any of the "
             "frames (off-screen sources only: do not name anything you can see). Choose names ONLY from this list, "
             "copied exactly:\n{families}\n"
             "Answer with a JSON list of strings only, most likely first, for example [\"Bird\", \"Vehicle\"]. "
             "If nothing off-screen is likely, answer [].")
WAV = {"dev": _ROOT / "data" / "work" / "devcand" / "wav16", "dev2": _ROOT / "data" / "work" / "r13dev2" / "wav16"}


def clip_file(pt, st):
    from benchmark.gold.imp_v_screen import clip_file as cf
    return cf(pt, st)


def parts():
    from benchmark.gold import btp_screen as B
    B.ARM = B_ARM
    return B.parts()


def stems_only():
    """[(part, stem)] of merged DEV; the gold is loaded by btp_screen.parts but never read by the GPU stages"""
    return [(pt, st) for pt, st, g, pics in parts()]


# ----------------------------------------------------------------------------- 1. propose (VLM)
_LOW = {f.lower(): f for f in FAMILIES}


def parse_families(reply: str):
    """-> (kept families (<= MAX_PROPOSE, deduped), discarded strings)"""
    from src.labels import canonical
    m = re.search(r"\[.*?\]", reply, re.S)
    items = []
    if m:
        try:
            items = [str(x) for x in json.loads(m.group(0))]
        except Exception:
            items = [x.strip().strip('"\'') for x in m.group(0).strip("[]").split(",")]
    else:                                                       # amendment 1: whole-word family names, order of first appearance
        low = reply.lower(); hits = []
        for f in FAMILIES:
            m2 = re.search(r"(?<![a-z])" + re.escape(f.lower()) + r"(?![a-z])", low)
            if m2:
                hits.append((m2.start(), f))
        items = [f for _p, f in sorted(hits)]
    kept, disc = [], []
    for it in items:
        it = it.strip()
        if not it:
            continue
        f = _LOW.get(it.lower())
        if f is None:
            c = canonical(it)
            f = c if c in _LOW.values() and it in VOCAB["labels"] else None
        if f is None:
            disc.append(it)
        elif f not in kept:
            kept.append(f)
    return kept[:MAX_PROPOSE], disc


def cmd_propose(device="cuda"):
    from src.stage5_cross_modal_analysis import reason
    from src.stage2_video_understanding import _sample_frames_at
    from src.stage1_audio_extraction import media_duration
    d = DIR / "propose"; d.mkdir(parents=True, exist_ok=True)
    todo = [(pt, st) for pt, st in stems_only() if not (d / f"{st}.json").exists()]
    print(f"{len(todo)} clips to propose", flush=True)
    if not todo:
        return
    mdl, proc = reason._load(MODEL, device)
    q = PROPOSE_Q.format(n=N_FRAMES, k=MAX_PROPOSE, families=", ".join(FAMILIES))
    for pt, st in todo:
        vp = Path(clip_file(pt, st))
        dur = float(media_duration(vp) or 0.0)
        times = [dur * (i + 0.5) / N_FRAMES for i in range(N_FRAMES)]
        win = _sample_frames_at(vp, times)
        reply = reason._ask(mdl, proc, q, images=win, max_new=96)
        fams, disc = parse_families(reply)
        rec = {"part": pt, "clip": st, "duration": round(dur, 2), "times": [round(t, 2) for t in times], "n_frames": len(win),
               "reply": reply, "families": fams, "discarded": disc}
        (d / f"{st}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"{pt:4s} {st} -> {fams} (discarded {disc})", flush=True)


# ----------------------------------------------------------------------------- 2. listen (Omni, whole clip, V4_Q)
def cmd_listen():
    import soundfile as sf
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from benchmark.gold import listener_variants as LV
    d = DIR / "listen"; d.mkdir(parents=True, exist_ok=True)
    todo = [(pt, st) for pt, st in stems_only() if not (d / f"{st}.json").exists()]      # stems first: parts() loads the gold
    LV.S.load_gold = LV._no_gold                                                          # ...and nothing after this may
    print(f"{len(todo)} clips to listen", flush=True)
    if not todo:
        return
    proc = Qwen3OmniMoeProcessor.from_pretrained(LV.MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(LV.MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass
    for pt, st in todo:
        w, sr = sf.read(str(WAV[pt] / f"{st}.wav"), dtype="float32")
        assert sr == LV.SR, (st, sr)
        conv = [{"role": "user", "content": [{"type": "audio", "audio": w}, {"type": "text", "text": LV.V4_Q}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[w], return_tensors="pt", padding=True, use_audio_in_video=False)
        inp = inp.to(model.thinker.device).to(torch.bfloat16)
        with torch.inference_mode():
            out = model.thinker.generate(**inp, max_new_tokens=LV.V4_NEW, do_sample=False)
        txt = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        rec = {"part": pt, "clip": st, "seconds": round(len(w) / sr, 2), "prompt": LV.V4_Q, "max_new": LV.V4_NEW, "v4_text": txt}
        (d / f"{st}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"{pt:4s} {st} -> {txt!r}", flush=True)


# ----------------------------------------------------------------------------- 3. cands (CPU)
def cmd_cands():
    from sentence_transformers import SentenceTransformer
    from benchmark.gold import listener_variants as LV
    from benchmark.gold import listener_afnext as AF
    from benchmark.gold import gbtp_screen as G
    from benchmark.gold import detached_add_screen as D
    from benchmark.gold.score_per_sound import same_family
    from src.labels import canonical
    assert D.BARS == {"flex": 0.3, "beats": 0.175, "dasm": 0.575}, D.BARS
    O = LV.Onto()
    emb = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    rows = []; missing = {m: 0 for m in D.BARS}
    tally = {"proposed": 0, "both": 0, "already_drawn": 0, "no_run": 0, "candidate": 0}
    for pt, st, g, pics in parts():
        pf = DIR / "propose" / f"{st}.json"; lf = DIR / "listen" / f"{st}.json"
        if not (pf.exists() and lf.exists()):
            print("missing stage output", pt, st); continue
        P = json.loads(pf.read_text(encoding="utf-8")); L = json.loads(lf.read_text(encoding="utf-8"))
        P["families"], P["discarded"] = parse_families(P["reply"])          # amendment 1: re-parse from the stored reply
        pf.write_text(json.dumps(P, indent=1), encoding="utf-8")
        cs = G.caches(pt, st)
        for m in D.BARS:
            missing[m] += cs[m] is None
        drawn = [l for l, a, b, r in pics]
        for F in P["families"]:
            tally["proposed"] += 1
            m, ok = AF.v4_match(O, emb, L["v4_text"], F)
            row = {"part": pt, "clip": st, "family": F, "omni_match": bool(ok), "matched": m["matched_word"] + m["matched_cos"]}
            if not ok:
                row["outcome"] = "omni does not name it"
            else:
                tally["both"] += 1
                already = [l for l in drawn if canonical(l) == F or same_family(l, F)]
                if already:
                    tally["already_drawn"] += 1; row["outcome"] = "already drawn"; row["drawn_as"] = already
                else:
                    evs = {mm: D.evidence(fr, F) for mm, fr in cs.items()}
                    found = {}
                    for mm in D.BARS:
                        rr = D.runs_of(evs[mm], D.BARS[mm])
                        if rr:
                            found[mm] = min(s for s, e in rr)
                    if not found:
                        tally["no_run"] += 1; row["outcome"] = "no run at weak bars"
                        row["no_evidence"] = [mm for mm in D.BARS if evs[mm] is None]
                    else:
                        tally["candidate"] += 1
                        src = min(found, key=found.get)
                        row.update({"outcome": "candidate", "onset": round(found[src], 2), "from": src,
                                    "onsets": {k: round(v, 2) for k, v in found.items()}})
            rows.append(row)
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / "cands.json").write_text(json.dumps({"tally": tally, "caches_missing": missing, "rows": rows}, indent=1), encoding="utf-8")
    print(tally, "caches missing", missing)
    for r in rows:
        if r["outcome"] == "candidate":
            print(f"   {r['part']:4s} {r['clip']} {r['family']} onset {r['onset']} ({r['from']}, {r['onsets']})")


# ----------------------------------------------------------------------------- 4. gate (VLM)
def gate_key(r):
    return f"{r['clip']}_{r['family'].replace(' ', '_').replace(',', '').replace('.', '')}_{r['onset']:.2f}"


def cmd_gate(device="cuda"):
    from src.stage5_cross_modal_analysis import reason
    from src.stage2_video_understanding import _sample_frames_at
    d = DIR / "gate"; d.mkdir(parents=True, exist_ok=True)
    rows = [r for r in json.loads((DIR / "cands.json").read_text(encoding="utf-8"))["rows"] if r["outcome"] == "candidate"]
    todo = [r for r in rows if not (d / f"{gate_key(r)}.json").exists()]
    print(f"{len(rows)} candidates, {len(todo)} to gate", flush=True)
    if not todo:
        return
    mdl, proc = reason._load(MODEL, device)
    for r in todo:
        vp = Path(clip_file(r["part"], r["clip"]))
        a = r["onset"]
        times = [a - 1.0 + 0.4 * t for t in range(6)]
        win = _sample_frames_at(vp, times)
        seen, named = reason._sound_is_visible(r["family"], win, mdl, proc, device)
        rec = {**r, "times": [round(t, 2) for t in times], "n_frames": len(win), "seen": bool(seen), "named": named,
               **{k: v for k, v in reason.LAST_VOTES.items() if k != "named"}}
        (d / f"{gate_key(r)}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"{r['part']:4s} {r['clip']} {r['family']} @{a} -> seen={seen} {reason.LAST_VOTES}", flush=True)


# ----------------------------------------------------------------------------- 5. score (CPU)
def cmd_score():
    from benchmark.gold import btp_screen as B
    from benchmark.gold import gbtp_screen as G
    from benchmark.gold import score_per_sound as S
    from benchmark.gold.cross_group import classify
    P = parts()
    base = {"dev": [], "dev2": []}
    for pt, st, g, pics in P:
        base[pt].append(S.score_clip(g, [p[:3] for p in pics]))
    Bm = {"merged": B.summ(base["dev"] + base["dev2"]), "dev": B.summ(base["dev"]), "dev2": B.summ(base["dev2"])}
    print(f"BASE SHIP8: merged {B.fmt(Bm['merged'])} | DEV {B.fmt(Bm['dev'])} | DEV2 {B.fmt(Bm['dev2'])}")
    assert (Bm["merged"]["hits"], Bm["merged"]["wrong"]) == (BASE["hits"], BASE["wrong"]), Bm["merged"]
    assert abs(Bm["merged"]["cost"] - BASE["cost"]) < 0.001, Bm["merged"]
    n_clips = len(P)
    C = json.loads((DIR / "cands.json").read_text(encoding="utf-8"))
    gates = {}
    for f in (DIR / "gate").glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8")); gates[(r["clip"], r["family"])] = r
    rows = {"dev": [], "dev2": []}; added = []; dropped = []; lost = []; unasked = 0
    oc = lambda r: f"h{r['hit']} v{r['visible']} c{r['cross']} p{r['phantom']} d{r['dup']}"
    for pt, st, g, pics in P:
        old = [p[:3] for p in pics]; new = list(old); adds = []
        for r in C["rows"]:
            if r["clip"] != st or r["part"] != pt or r["outcome"] != "candidate":
                continue
            gt = gates.get((st, r["family"]))
            if gt is None:
                unasked += 1; continue
            if gt["seen"]:
                dropped.append({"part": pt, "clip": st, "family": r["family"], "onset": r["onset"], "votes": [gt.get(k) for k in ("name", "ab", "desc")]})
                continue
            new.append((r["family"], r["onset"], r["onset"] + PIC_LEN)); adds.append((r, gt))
        r0 = S.score_clip(g, old); r1 = S.score_clip(g, new)
        rows[pt].append(r1)
        if adds:
            c1 = {(l, round(a, 3)): k for l, a, b, k, _ in classify(g, new)}
            for r, gt in adds:
                added.append({"part": pt, "clip": st, "family": r["family"], "start": r["onset"], "end": round(r["onset"] + PIC_LEN, 2),
                              "from": r["from"], "votes": [gt.get(k) for k in ("name", "ab", "desc")],
                              "class": c1[(r["family"], round(r["onset"], 3))], "clip_before": oc(r0), "clip_after": oc(r1)})
        if r1["hit"] < r0["hit"]:
            lost.append((pt, st, r0["hit"] - r1["hit"]))
    X = {"merged": B.summ(rows["dev"] + rows["dev2"]), "dev": B.summ(rows["dev"]), "dev2": B.summ(rows["dev2"])}
    gain = X["merged"]["hits"] - BASE["hits"]
    c1w = {w: G.cost_w(X["merged"], n_clips, w) for w in (1, 2)}
    c0w = {w: G.cost_w(Bm["merged"], n_clips, w) for w in (1, 2)}
    main_go = X["merged"]["hits"] >= BASE["hits"] and not lost and c1w[2] < c0w[2]
    fa_ok = all(X[pt][k] <= Bm[pt][k] for pt in ("dev", "dev2") for k in ("cross", "phantom"))
    more_go = gain > 0 and fa_ok and c1w[1] < c0w[1]
    print(f"EXPECT: merged {B.fmt(X['merged'])} | DEV {B.fmt(X['dev'])} | DEV2 {B.fmt(X['dev2'])}  added {len(added)}  "
          f"gate-dropped {len(dropped)}  unasked {unasked}  hits lost {lost}  tally {C['tally']}")
    print(f"  main rule: {'GO' if main_go else 'STOP'}   more-hits rule: {'PASS' if more_go else 'FAIL'}   "
          f"cost w=1 {c0w[1]:.3f} -> {c1w[1]:.3f}, w=2 {c0w[2]:.3f} -> {c1w[2]:.3f}")
    for d in added:
        print(f"   {d['part']:4s} {d['clip']} {d['family']} +{d['start']}-{d['end']} ({d['from']}, votes {d['votes']}): {d['class']}   "
              f"clip {d['clip_before']} -> {d['clip_after']}")
    for d in dropped:
        print(f"   {d['part']:4s} {d['clip']} {d['family']} @{d['onset']} gate seen {d['votes']} -> dropped")
    res = {"base": Bm, "rows": X, "added": added, "gate_dropped": dropped, "hits_lost": lost, "unasked": unasked,
           "tally": C["tally"], "caches_missing": C["caches_missing"],
           "cost": {str(w): {"base": round(c0w[w], 3), "new": round(c1w[w], 3)} for w in (1, 2)},
           "main_GO": main_go, "more_hits_PASS": more_go}
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    {"propose": cmd_propose, "listen": cmd_listen, "cands": cmd_cands, "gate": cmd_gate, "score": cmd_score}[sys.argv[1]]()
