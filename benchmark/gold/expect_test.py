"""Round 40e TEST read (docs/prereg_round13_detector_push.md "Round 40e TEST read (reported, not selected on)"): EXPECT-A4 exactly
as frozen on the merged TEST (old TEST 60 under data/work/r16final arm SHIP7+K4AD + tagger TEST 28 under tagger_prep.out("test2")),
base = the shipped SHIP8 TEST pictures (23/42/29 (4/20/5) 2.568 must reproduce). Gold is read ONLY in `score`.

    TG_ARMS=SHIP8 python benchmark/gold/expect_test.py parts    # CPU: the clip list + picture counts (no gold)
    TG_ARMS=SHIP8 python benchmark/gold/expect_test.py listen   # GPU msproj: Qwen3-Omni list -> expect_test/listen/
    TG_ARMS=SHIP8 python benchmark/gold/expect_test.py cands    # CPU: map, first two, not already drawn -> expect_test/pairs.json
    python benchmark/gold/expect_test.py flap                  # GPU venv_flap: FineLAP -> expect_test/flap/<clip>.npz
    TG_ARMS=SHIP8 python benchmark/gold/expect_test.py dasm     # CPU: FineLAP onset + DASM check -> expect_test/cands.json
    TG_ARMS=SHIP8 python benchmark/gold/expect_test.py gate     # GPU msproj: shipped gate, live -> expect_test/gate/
    TG_ARMS=SHIP8 python benchmark/gold/expect_test.py score    # CPU: gold read here only -> expect_test.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")

ARM = "SHIP7+K4AD"                                               # the shipped SHIP8 TEST render's arm name
DIR = _ROOT / "benchmark" / "gold" / "expect_test"
OUT = _ROOT / "benchmark" / "gold" / "expect_test.json"
WORKD = _ROOT / "data" / "work"
BASE = {"hits": 23, "misses": 42, "wrong": 29, "cost": 2.568}
WAV = {"test": WORKD / "r13test" / "wav16", "test2": WORKD / "r13test2" / "wav16"}
DASM = {"test": WORKD / "dasm_test", "test2": WORKD / "dasm_test2"}
FLAP_BAR, DASM_BAR, HALF, PIC_LEN, GAP, FRAME = 0.329, 0.575, 0.5, 2.0, 0.24, 0.16


# ---------------------------------------------------------------- the merged TEST without gold (final_test.score's loading, verbatim)
_PARTS = None


def parts():
    """[(part, stem, pics [(label, start, end)], clip video path)]; gold never read"""
    global _PARTS
    if _PARTS is not None:
        return _PARTS
    from benchmark.gold import score_per_sound as S
    from benchmark.gold import final_test as FT
    F, T, R, arms, fin = FT.old_setup(ARM)
    stems1 = list(T.STEMS)
    with R.flags({k: R.arm_cfg(ARM)[k] for k in R.DISPLAY_KEYS}):
        P1 = {st: S.load_pictures(fin / f"{ARM}_proposed", st, "proposed") or [] for st in stems1}
    from benchmark.gold import tagger_prep as TP
    for k, v in FT._ARMS0.items():
        R.ARMS[k] = dict(v)
    TP._ORIG.clear()
    _D2, R2, stems2 = TP.configure("test2")
    o2 = TP.out("test2")
    with R2.flags({k: R2.arm_cfg(ARM)[k] for k in R2.DISPLAY_KEYS}):
        P2 = {st: S.load_pictures(o2 / f"{ARM}_proposed", st, "proposed") or [] for st in stems2}
    miss = [st for st in stems2 if not (o2 / f"{ARM}_proposed" / st / "augmentations.json").exists()]
    assert not miss, miss
    from benchmark.gold.imp_v_screen import clip_file
    out = [("test", st, [tuple(p[:3]) for p in P1[st]], Path(clip_file("dev", st))) for st in stems1]
    out += [("test2", st, [tuple(p[:3]) for p in P2[st]], TP.CLIPS / f"{st}.mp4") for st in stems2]
    _PARTS = out
    return out


def cmd_parts():
    P = parts()
    print(len(P), "clips;", sum(len(p[2]) for p in P), "SHIP8 pictures;", sum(not p[3].exists() for p in P), "clip files missing;",
          sum(not (WAV[pt] / f"{st}.wav").exists() for pt, st, _p, _c in P), "wavs missing;",
          sum(not (DASM[pt] / f"{st}.npz").exists() for pt, st, _p, _c in P), "DASM caches missing")


# ---------------------------------------------------------------- listen (GPU msproj)
def cmd_listen():
    import soundfile as sf
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from benchmark.gold import listener_variants as LV
    from benchmark.gold import expect_a_screen as A
    d = DIR / "listen"; d.mkdir(parents=True, exist_ok=True)
    todo = [(pt, st) for pt, st, _p, _c in parts() if not (d / f"{st}.json").exists()]
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
        conv = [{"role": "user", "content": [{"type": "audio", "audio": w}, {"type": "text", "text": A.LIST_Q}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[w], return_tensors="pt", padding=True, use_audio_in_video=False)
        inp = inp.to(model.thinker.device).to(torch.bfloat16)
        with torch.inference_mode():
            out = model.thinker.generate(**inp, max_new_tokens=A.MAX_NEW, do_sample=False, repetition_penalty=A.REP_PEN)
        txt = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        rec = {"part": pt, "clip": st, "seconds": round(len(w) / sr, 2), "prompt": A.LIST_Q, "max_new": A.MAX_NEW,
               "repetition_penalty": A.REP_PEN, "text": txt, "items": A.items_of(txt)}
        (d / f"{st}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"{pt:5s} {st} -> {txt[:120]!r}", flush=True)


# ---------------------------------------------------------------- cands (CPU): map, first two, not already drawn
def cmd_cands():
    from benchmark.gold import expect_a_screen as A
    from benchmark.gold.score_per_sound import same_family
    from src.labels import canonical
    rows = []; tally = {"items": 0, "named": 0, "kept2": 0, "already_drawn": 0, "pair": 0}
    for pt, st, pics, _c in parts():
        L = json.loads((DIR / "listen" / f"{st}.json").read_text(encoding="utf-8"))
        items = A.items_of(L["text"]); tally["items"] += len(items)
        fams, how = [], {}
        for it in items:
            f = A.map_item(it)
            if f and f not in fams:
                fams.append(f); how[f] = f"map:{it}"
        tally["named"] += len(fams)
        fams = fams[:2]; tally["kept2"] += len(fams)                               # first two, as Round 40c
        drawn = [l for l, a, b in pics]
        for F in fams:
            row = {"part": pt, "clip": st, "family": F, "how": how[F]}
            already = [l for l in drawn if canonical(l) == F or same_family(l, F)]
            if already:
                tally["already_drawn"] += 1; row["outcome"] = "already drawn"; row["drawn_as"] = already
            else:
                tally["pair"] += 1; row["outcome"] = "pair"
            rows.append(row)
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / "pairs.json").write_text(json.dumps({"tally": tally, "rows": rows}, indent=1), encoding="utf-8")
    print(tally)


# ---------------------------------------------------------------- flap (GPU venv_flap; no project imports beyond finelap_screen)
def cmd_flap():
    import torch
    from transformers import AutoModel
    from benchmark.gold import finelap_screen as FS
    d = DIR / "flap"; d.mkdir(parents=True, exist_ok=True)
    rows = [r for r in json.loads((DIR / "pairs.json").read_text(encoding="utf-8"))["rows"] if r["outcome"] == "pair"]
    want = {}
    for r in rows:
        want.setdefault((r["part"], r["clip"]), set()).add(r["family"])
    todo = {k: v for k, v in want.items() if not (d / f"{k[1]}.npz").exists()}
    print(f"{len(want)} clips, {len(rows)} pairs, {len(todo)} clips to score", flush=True)
    if not todo:
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("AndreasXi/FineLAP", trust_remote_code=True).to(dev).eval()
    for (pt, st), fams in sorted(todo.items()):
        ph = sorted(fams)
        fs, fe, sc = FS.frame_scores(model, FS._load(WAV[pt] / f"{st}.wav"), ph, dev)
        np.savez(d / f"{st}.npz", fs=fs, fe=fe, scores=sc, labels=np.array(ph))
        print(f"[flap] {pt} {st}: {ph} -> max {sc.max(axis=0).round(3).tolist()}", flush=True)


# ---------------------------------------------------------------- dasm (CPU): FineLAP onset, DASM confirmation
def cmd_dasm():
    from benchmark.gold import expect_a3_screen as A3
    from benchmark.gold import expect_a4_screen as A4
    from benchmark.gold import dev_candidates_check as DCC
    from src.stage4_audio_event_detection import _runs, LISTEN_RUN_GAP
    assert abs(LISTEN_RUN_GAP - GAP) < 1e-9 and A3.BAR == FLAP_BAR and A4.BAR == DASM_BAR
    rows = [r for r in json.loads((DIR / "pairs.json").read_text(encoding="utf-8"))["rows"] if r["outcome"] == "pair"]
    out = []; tally = {"pairs": len(rows), "below_flap": 0, "no_dasm_column": 0, "below_dasm": 0, "candidate": 0}
    for r in rows:
        pt, st, F = r["part"], r["clip"], r["family"]
        z = np.load(DIR / "flap" / f"{st}.npz")
        ts, g = A3.grid_scores(z, F)
        r["flap_max"] = round(float(g.max()), 3)
        rr, dt = _runs(g, ts, FLAP_BAR, GAP)
        if not rr:
            tally["below_flap"] += 1; r["outcome"] = "below FineLAP bar"; out.append(r); continue
        best = max(rr, key=lambda ij: (float(g[ij[0]:ij[1]].max()), -ij[0]))
        onset = float(ts[best[0]])
        r.update({"onset": round(onset, 2), "from": "flap", "run": [round(onset, 2), round(float(ts[best[1] - 1]) + dt, 2)]})
        p = DASM[pt] / f"{st}.npz"
        fr = DCC.load_fr(p) if p.exists() else None
        v = A4.dasm_max(fr, F, onset - HALF, onset + HALF)
        r["dasm_onset"] = None if v is None else round(v, 3)
        if v is None:
            tally["no_dasm_column"] += 1; r["outcome"] = "no DASM column"
        elif v < DASM_BAR:
            tally["below_dasm"] += 1; r["outcome"] = "below DASM bar"
        else:
            tally["candidate"] += 1; r["outcome"] = "candidate"
        out.append(r)
    (DIR / "cands.json").write_text(json.dumps({"tally": tally, "caches_missing": {}, "rows": out}, indent=1), encoding="utf-8")
    print(tally)
    for r in out:
        if r["outcome"] == "candidate":
            print(f"   {r['part']:5s} {r['clip']} {r['family']} flap {r['flap_max']} onset {r['onset']} dasm {r['dasm_onset']}")


# ---------------------------------------------------------------- gate (GPU msproj), live
def cmd_gate(device="cuda"):
    from src.stage5_cross_modal_analysis import reason
    from src.stage2_video_understanding import _sample_frames_at
    from benchmark.gold import expect_screen as E
    d = DIR / "gate"; d.mkdir(parents=True, exist_ok=True)
    rows = [r for r in json.loads((DIR / "cands.json").read_text(encoding="utf-8"))["rows"] if r["outcome"] == "candidate"]
    todo = [r for r in rows if not (d / f"{E.gate_key(r)}.json").exists()]
    print(f"{len(rows)} candidates, {len(todo)} to gate", flush=True)
    if not todo:
        return
    clip_of = {(pt, st): c for pt, st, _p, c in parts()}
    mdl, proc = reason._load(E.MODEL, device)
    for r in todo:
        a = r["onset"]
        times = [a - 1.0 + 0.4 * t for t in range(6)]
        win = _sample_frames_at(clip_of[(r["part"], r["clip"])], times)
        seen, named = reason._sound_is_visible(r["family"], win, mdl, proc, device)
        rec = {**r, "times": [round(t, 2) for t in times], "n_frames": len(win), "seen": bool(seen), "named": named,
               **{k: v for k, v in reason.LAST_VOTES.items() if k != "named"}}
        (d / f"{E.gate_key(r)}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"{r['part']:5s} {r['clip']} {r['family']} @{a} -> seen={seen} {reason.LAST_VOTES}", flush=True)


# ---------------------------------------------------------------- score (CPU): gold read here only
def cmd_score():
    from benchmark.gold import score_per_sound as S
    from benchmark.gold import final_test as FT
    from benchmark.gold import dev_candidates_check as DCC
    from benchmark.gold import expect_screen as E
    from benchmark.gold.cross_group import classify
    P = parts()
    from benchmark.gold import tagger_prep as TP
    S.load_gold = FT._REAL_LOAD_GOLD
    gold1 = S.load_gold([FT.G / "annotations" / "gold_AG.json"])
    stems1 = [st for pt, st, _p, _c in P if pt == "test"]; stems2 = [st for pt, st, _p, _c in P if pt == "test2"]
    assert sorted(S.subsets_of(gold1)["test_bench"]) == sorted(stems1)
    o2 = TP.out("test2")
    dg = json.loads(TP.TAGGER_GOLD.read_text(encoding="utf-8"))
    dg["clips"] = [c for c in dg.get("clips", []) if isinstance(c, dict) and Path(str(c.get("clip", ""))).stem in set(stems2)]
    tmp = o2 / "test2_gold_only.json"; DCC.dump(tmp, dg)
    gold2 = S.load_gold([tmp])
    gold = {("test", st): gold1[st] for st in stems1} | {("test2", st): gold2[st] for st in stems2 if st in gold2}
    C = json.loads((DIR / "cands.json").read_text(encoding="utf-8"))
    gates = {}
    for f in (DIR / "gate").glob("*.json"):
        g = json.loads(f.read_text(encoding="utf-8")); gates[(g["clip"], g["family"])] = g
    r0s, r1s, c0, c1, added, dropped, lost, unasked = [], [], [], [], [], [], [], 0
    pr = {"base": {"test": [], "test2": []}, "new": {"test": [], "test2": []}}
    oc = lambda r: f"h{r['hit']} v{r['visible']} c{r['cross']} p{r['phantom']} d{r['dup']}"
    for pt, st, pics, _c in P:
        if (pt, st) not in gold:
            continue
        g = gold[(pt, st)]
        new = list(pics); adds = []
        for r in C["rows"]:
            if (r["clip"], r["part"], r["outcome"]) != (st, pt, "candidate"):
                continue
            gt = gates.get((st, r["family"]))
            if gt is None:
                unasked += 1; continue
            if gt["seen"]:
                dropped.append({"part": pt, "clip": st, "family": r["family"], "onset": r["onset"], "votes": [gt.get(k) for k in ("name", "ab", "desc")]})
                continue
            new.append((r["family"], r["onset"], r["onset"] + PIC_LEN)); adds.append((r, gt))
        r0 = S.score_clip(g, pics); r1 = S.score_clip(g, new)
        r0s.append(r0); r1s.append(r1); c0.append(DCC.clip_cost(r0)); c1.append(DCC.clip_cost(r1))
        pr["base"][pt].append(r0); pr["new"][pt].append(r1)
        if adds:
            cl = {(l, round(a, 3)): k for l, a, b, k, _ in classify(g, new)}
            for r, gt in adds:
                added.append({"part": pt, "clip": st, "family": r["family"], "start": r["onset"], "flap_max": r["flap_max"],
                              "dasm": r["dasm_onset"], "votes": [gt.get(k) for k in ("name", "ab", "desc")],
                              "class": cl[(r["family"], round(r["onset"], 3))], "clip_before": oc(r0), "clip_after": oc(r1)})
        if r1["hit"] < r0["hit"]:
            lost.append((pt, st, r0["hit"] - r1["hit"]))
    m0, m1 = DCC.metrics(r0s), DCC.metrics(r1s)
    n = len(r0s)
    cw = lambda m, w: (4 * m["misses"] + w * m["visible"] + 2 * m["cross"] + 2 * m["phantom"]) / n
    assert (m0["hits"], m0["misses"], m0["wrong"]) == (BASE["hits"], BASE["misses"], BASE["wrong"]), m0
    assert abs(m0["viewer_cost"] - BASE["cost"]) < 0.001, m0["viewer_cost"]
    d_ = DCC.boot(np.subtract(c1, c0))
    fmt = lambda m: f"{m['hits']}/{m['hits'] + m['misses']} wrong {m['wrong']} ({m['visible']}/{m['cross']}/{m['phantom']}) cost {m['viewer_cost']:.3f}"
    print(f"BASE SHIP8 TEST: {fmt(m0)}  (old TEST {fmt(DCC.metrics(pr['base']['test']))} | tagger TEST {fmt(DCC.metrics(pr['base']['test2']))})")
    print(f"EXPECT-A4 TEST:  {fmt(m1)}  (old TEST {fmt(DCC.metrics(pr['new']['test']))} | tagger TEST {fmt(DCC.metrics(pr['new']['test2']))})")
    print(f"  cost w=2 {cw(m0, 2):.3f} -> {cw(m1, 2):.3f}; w=1 {cw(m0, 1):.3f} -> {cw(m1, 1):.3f}; d cost {d_[0]:+.3f} [{d_[1]:+.3f}, {d_[2]:+.3f}] one-sided p {d_[3]:.3f}")
    print(f"  added {len(added)}  gate-dropped {len(dropped)}  unasked {unasked}  hits lost {lost}  tally {C['tally']}")
    for a in added:
        print(f"   {a['part']:5s} {a['clip']} {a['family']} +{a['start']} (flap {a['flap_max']}, dasm {a['dasm']}, votes {a['votes']}): {a['class']}  clip {a['clip_before']} -> {a['clip_after']}")
    for dd in dropped:
        print(f"   {dd['part']:5s} {dd['clip']} {dd['family']} @{dd['onset']} gate seen {dd['votes']} -> dropped")
    res = {"arm": "SHIP8+EXPECT-A4 (TEST read)", "clips": n, "base": m0, "new": m1,
           "parts": {k: {pt: DCC.metrics(v[pt]) for pt in v} for k, v in pr.items()},
           "cost": {"w2": [round(cw(m0, 2), 3), round(cw(m1, 2), 3)], "w1": [round(cw(m0, 1), 3), round(cw(m1, 1), 3)]},
           "delta_cost_vs_base": d_, "added": added, "gate_dropped": dropped, "hits_lost": lost, "unasked": unasked, "tally": C["tally"]}
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    {"parts": cmd_parts, "listen": cmd_listen, "cands": cmd_cands, "flap": cmd_flap, "dasm": cmd_dasm, "gate": cmd_gate,
     "score": cmd_score}[sys.argv[1]]()
