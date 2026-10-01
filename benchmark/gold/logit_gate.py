"""Round 62 LOGIT-GATE (docs/prereg_round13_detector_push.md "Round 62 LOGIT-GATE"). For a yes/no question: ONE forward pass
of the shipped gate VLM (Qwen3.8-27B), no generation; s = max logit(yes ids) - max logit(no ids) at the last prompt position;
d = s(Q) - s(notQ) with a fixed opposite-polarity twin (high d = visible). Questions (a) source-visible, (b) sign/effect with
Fable D's fixed lexicon. Per sound / spec: D = mean d over its gate stretches.

    python benchmark/gold/logit_gate.py sanity       # GPU, ~/MscProj: template tail, top-5, greedy check (exit 4 = harness wrong)
    python benchmark/gold/logit_gate.py gold         # GPU, ~/MscProj: gate-gold d per stretch -> gate_gold/logit_Qwen38-27B/
    python benchmark/gold/logit_gate.py score_gold   # CPU, ~/MscProj: step 0 + step 1 (exit 0 = step-1 PASS, 3 = FAIL / STOP)
    ARM=SHIP8+MD3+WW5 python benchmark/gold/logit_gate.py dev        # GPU, ~/MscProj_tg -> benchmark/gold/logit_pics/
    ARM=SHIP8+MD3+WW5 python benchmark/gold/logit_gate.py score_dev  # CPU, ~/MscProj_tg -> benchmark/gold/logit_gate_dev.json
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", os.environ.get("ARM", "SHIP8+MD3+WW5"))

MODEL = "Qwen/Qwen3.8-27B"
G = _ROOT / "benchmark" / "gold"
GOLD_OUT = G / "gate_gold" / "logit_Qwen38-27B"
GOLD_SUMMARY = G / "logit_gate_gold.json"           # copied to ~/MscProj_tg by the job (step 2 reads question + t)
DEV_CACHE = G / "logit_pics"
DEV_OUT = G / "logit_gate_dev.json"
NAMED = "as_explosion_XJ8lc3I6"
BASE_GOLD = {"seen": 41, "seen_sil": 16, "needed": 38, "needed_kept": 33}
BASE_D = {"tag": "D", "hits": 29, "wrong": 14, "cost": 2.028, "cls": (6, 6, 2)}
AUROC_STOP = 0.65

QA = ("Is the thing making the {label} sound visible in these frames? Answer yes or no.",
      "Is the thing making the {label} sound NOT visible in these frames? Answer yes or no.")
QB = ("Is {sign} visible in these frames? Answer yes or no.",
      "Is {sign} NOT visible in these frames? Answer yes or no.")
LEXICON = {"thunder": "a lightning flash", "water": "water flowing or splashing", "laughter": "a person visibly laughing",
           "explosion": "a burst of fire or light", "vehicle": "a vehicle moving"}


def sign_of(label):
    return LEXICON.get(label.strip().lower(), f"the visible effect of {label}")


def prompts(label):
    return {"a": [q.format(label=label) for q in QA], "b": [q.format(sign=sign_of(label)) for q in QB]}


class Reader:
    def __init__(self, device="cuda"):
        from src.stage5_cross_modal_analysis import reason
        self.mdl, self.proc = reason._load(MODEL, device)
        tok = self.proc.tokenizer
        self.tok = tok
        self.yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")})
        self.no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")})
        print("yes ids", self.yes, "no ids", self.no, flush=True)

    def inputs(self, prompt, frames):
        content = [{"type": "image"} for _ in frames] + [{"type": "text", "text": prompt}]
        text = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        return text, self.proc(text=[text], images=frames, return_tensors="pt").to(self.mdl.device)

    def logits(self, inp):
        import torch
        with torch.inference_mode():
            return self.mdl(**inp).logits[0, -1].float()

    def s(self, prompt, frames):
        _, inp = self.inputs(prompt, frames)
        lg = self.logits(inp)
        am = int(lg.argmax())
        return float(lg[self.yes].max() - lg[self.no].max()), am, am in self.yes or am in self.no

    def pair(self, qs, frames):
        sq, aq, okq = self.s(qs[0], frames)
        sn, an, okn = self.s(qs[1], frames)
        return {"sQ": sq, "sN": sn, "d": sq - sn, "argmax": [self.tok.decode([aq]), self.tok.decode([an])], "yn_argmax": [okq, okn]}

    def both(self, label, frames):
        return {k: self.pair(qs, frames) for k, qs in prompts(label).items()}


# ---------------------------------------------------------------- gate-gold (~/MscProj)
def _gold_items():
    import config
    from benchmark.gold import gate_gold as GG
    config.use_v4("5")
    judge = set(GG.JUDGE100.read_text().split())
    names = {stem: name for name, stem, _ in GG.gold_sounds()}
    src = GG.OUT_DIR / "Qwen38-27B"
    return GG, [(f, names) for f in sorted(src.glob("*.json")) if f.stem in judge]


def cmd_sanity():
    import torch
    from benchmark.gold.sign_gate import gate_times
    from src.stage2_video_understanding import _sample_frames_at
    GG, items = _gold_items()
    f, names = items[0]
    d = json.loads(f.read_text(encoding="utf-8"))
    s, st = d["sounds"][0], d["sounds"][0]["stretches"][0]
    frames = _sample_frames_at(GG.clip_path(names.get(f.stem, d["clip"])), gate_times(st["start"], st["end"]))
    R = Reader()
    ok_all = True
    for k, qs in prompts(s["label"]).items():
        for q in qs:
            text, inp = R.inputs(q, frames)
            print(f"SANITY [{k}] {f.stem} {s['label']} prompt={q!r}")
            print("  template tail:", repr(text[-160:]))
            lg = R.logits(inp)
            top = torch.topk(lg, 5)
            print("  top5 at read position:", [(R.tok.decode([int(i)]), round(float(v), 2)) for v, i in zip(top.values, top.indices)])
            with torch.inference_mode():
                out = R.mdl.generate(**inp, max_new_tokens=4, do_sample=False)
            new = out[0, inp["input_ids"].shape[1]:]
            print("  greedy 4 tokens:", [R.tok.decode([int(t)]) for t in new], "| s =", round(float(lg[R.yes].max() - lg[R.no].max()), 3))
            ok = int(new[0]) == int(lg.argmax())
            ok_all &= ok
            print("  first greedy token == argmax at read position:", ok, flush=True)
    print("SANITY", "GO" if ok_all else "STOP (harness wrong)")
    sys.exit(0 if ok_all else 4)


def cmd_gold():
    from benchmark.gold.sign_gate import gate_times
    from src.stage2_video_understanding import _sample_frames_at
    GG, items = _gold_items()
    GOLD_OUT.mkdir(parents=True, exist_ok=True)
    R = None
    for f, names in items:
        if (GOLD_OUT / f.name).exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        p = GG.clip_path(names.get(f.stem, d["clip"]))
        if p is None:
            print("missing", f.stem); continue
        R = R or Reader()
        for s in d["sounds"]:
            for st in s["stretches"]:
                fr = _sample_frames_at(p, gate_times(st["start"], st["end"]))
                st["lg"] = R.both(s["label"], fr) if len(fr) >= 2 else None
                print(f.stem, s["label"], st["start"], {k: round(v["d"], 2) for k, v in (st["lg"] or {}).items()}, flush=True)
        (GOLD_OUT / f.name).write_text(json.dumps(d, indent=1), encoding="utf-8")


def auroc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def mean_d(sts, q):
    v = [st["lg"][q]["d"] for st in sts if st.get("lg")]
    return statistics.fmean(v) if v else None


def pick_t(rows, add_seen):
    """rows: [(seen, shipped_silenced, D)] -> (best or None, curve). Rule: maximise silenced under the bar; ties -> kept -> t."""
    vals = sorted({r[2] for r in rows if r[2] is not None})
    cands = [(a + b) / 2 for a, b in zip(vals, vals[1:])] + [float("inf")]
    if vals:
        cands.append(vals[0] - 1.0)
    curve = []
    for t in cands:
        sil = kept = 0
        for seen, ship, D in rows:
            pred = (D is not None and D > t) or (add_seen and ship)
            sil += seen and pred
            kept += (not seen) and (not pred)
        bar = (sil >= 19 and kept >= 32) or (sil >= 15 and kept >= 35)
        curve.append({"t": t, "silenced": sil, "kept": kept, "bar": bar})
    ok = [c for c in curve if c["bar"]]
    best = max(ok, key=lambda c: (c["silenced"], c["kept"], c["t"])) if ok else None
    return best, curve


def cmd_score_gold():
    from benchmark.gold.box_gate import gold_index, majority
    gold = gold_index()
    files = sorted(GOLD_OUT.glob("*.json"))
    assert len(files) == 49, len(files)
    sounds, n_st, n_yn, n_none = [], 0, 0, 0
    per_stretch = {"a": ([], []), "b": ([], [])}
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for s in d["sounds"]:
            g = gold.get((f.stem, s["label"], round(s["start"], 2)))
            if g is None or g["importance"] < 2:
                continue
            sts = s["stretches"]
            for st in sts:
                n_st += 1
                if not st.get("lg"):
                    n_none += 1; continue
                n_yn += sum(st["lg"]["a"]["yn_argmax"]) + sum(st["lg"]["b"]["yn_argmax"])
                for q in "ab":
                    per_stretch[q][0 if g["seen"] else 1].append(st["lg"][q]["d"])
            sounds.append({"clip": f.stem, "label": s["label"], "start": s["start"], "seen": bool(g["seen"]),
                           "shipped_silenced": all(majority(st) for st in sts), "n_stretches": len(sts),
                           "D_a": mean_d(sts, "a"), "D_b": mean_d(sts, "b"),
                           "votes": [[st.get("name"), st.get("ab"), st.get("desc")] for st in sts]})
    base = {"seen": sum(x["seen"] for x in sounds), "seen_sil": sum(x["seen"] and x["shipped_silenced"] for x in sounds),
            "needed": sum(not x["seen"] for x in sounds), "needed_kept": sum(not x["seen"] and not x["shipped_silenced"] for x in sounds)}
    print("base shipped majority:", base)
    assert base == BASE_GOLD, base
    print(f"stretches {n_st} (no frames {n_none}); argmax is a yes/no id on {n_yn}/{4 * (n_st - n_none)} reads")
    res = {"base": base, "stretches": n_st, "no_frames": n_none, "yn_argmax_share": n_yn / max(1, 4 * (n_st - n_none)),
           "questions": {"a": QA, "b": QB, "lexicon": LEXICON}, "q": {}}
    for q in "ab":
        k = f"D_{q}"
        A = auroc([x[k] for x in sounds if x["seen"] and x[k] is not None], [x[k] for x in sounds if not x["seen"] and x[k] is not None])
        As = auroc(*per_stretch[q])
        rows = [(x["seen"], x["shipped_silenced"], x[k]) for x in sounds]
        add, add_curve = pick_t(rows, True)
        rep, rep_curve = pick_t(rows, False)
        stop = A is None or A < AUROC_STOP
        res["q"][q] = {"auroc_sound": A, "auroc_stretch": As, "stop": stop, "add_seen": add, "replace": rep,
                       "add_curve_top": sorted(add_curve, key=lambda c: -c["t"])[:12],
                       "replace_curve_bar": [c for c in rep_curve if c["bar"]][:20]}
        print(f"[{q}] AUROC per sound {A:.3f} | per stretch {As:.3f} -> {'STOP' if stop else 'GO'} | ADD-seen best {add} | "
              f"replace best {rep}")
    go = [q for q in "ab" if not res["q"][q]["stop"]]
    best = max(go, key=lambda q: (res["q"][q]["auroc_sound"], q == "a")) if go else None
    sel = res["q"][best]["add_seen"] if best else None
    res["best_question"], res["t"] = best, (sel["t"] if sel else None)
    res["step1"] = "PASS" if sel else ("STOP (AUROC)" if not best else "FAIL (no t meets the bar)")
    # ADD-seen flips at the chosen t (or, on FAIL, the shipped-kept sounds ranked by D of the best / (a) question)
    qq = best or "a"
    kk = f"D_{qq}"
    res["shipped_kept_ranked"] = sorted([[x["clip"], x["label"], x["start"], "seen" if x["seen"] else "NEEDED",
                                          None if x[kk] is None else round(x[kk], 3)]
                                         for x in sounds if not x["shipped_silenced"]],
                                        key=lambda r: -(r[4] if r[4] is not None else -1e9))
    res["named"] = [x for x in sounds if x["clip"] == NAMED]
    res["sounds"] = sounds
    print("step 1:", res["step1"], "| question", best, "| t", res["t"])
    print("shipped-kept sounds ranked by", kk, "(top 12):")
    for r in res["shipped_kept_ranked"][:12]:
        print("   ", r)
    print("named", NAMED, [(x["label"], x["seen"], x["shipped_silenced"], x["D_a"], x["D_b"]) for x in res["named"]])
    GOLD_SUMMARY.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(GOLD_SUMMARY)
    sys.exit(0 if sel else 3)


# ---------------------------------------------------------------- DEV arm D (~/MscProj_tg)
def _arm():
    from benchmark.gold import sign_screen as SS
    SS.ARM = os.environ.get("ARM", "SHIP8+MD3+WW5")
    return SS


def cmd_dev():
    from benchmark.gold.imp_v_screen import clip_file
    from benchmark.gold.sign2_screen import _frames
    from benchmark.gold.sign_screen import key
    SS = _arm()
    DEV_CACHE.mkdir(parents=True, exist_ok=True)
    A = SS.asked(SS.parts())
    todo = [a for a in A if not (DEV_CACHE / f"{key(a[0], a[1], a[3], a[4])}.json").exists()]
    print(len(A), "drawn specs,", len(todo), "to ask; fallback specs", sum(a[6] == "fallback" for a in A), flush=True)
    R = None
    for pt, st, w, lab, a0, sts, src in todo:
        R = R or Reader()
        vp = Path(clip_file(pt, st))
        res = []
        for a, b in sts:
            fr = _frames(vp, a, b)
            res.append({"stretch": [a, b], "n_frames": len(fr), "lg": R.both(lab, fr) if len(fr) >= 2 else None})
        rec = {"part": pt, "clip": st, "label": lab, "start": a0, "source": src, "stretches": res,
               "D_a": mean_d(res, "a"), "D_b": mean_d(res, "b")}
        (DEV_CACHE / f"{key(pt, st, lab, a0)}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(pt, st, lab, a0, src, "D_a", rec["D_a"], "D_b", rec["D_b"], flush=True)


def cmd_score_dev():
    from benchmark.gold import score_per_sound as S
    from benchmark.gold.cross_group import classify
    from benchmark.gold.sign_screen import key
    SS = _arm()
    summ = json.loads(GOLD_SUMMARY.read_text(encoding="utf-8"))
    assert summ["step1"] == "PASS", summ["step1"]
    q, t = summ["best_question"], float(summ["t"])
    specs = json.loads((G / "explain_d_specs_D.json").read_text(encoding="utf-8"))
    recs = [json.loads((DEV_CACHE / f"{key(*x)}.json").read_text(encoding="utf-8")) for x in specs]
    assert len(recs) == 41, len(recs)
    Dk = f"D_{q}"
    mask, Dof = {}, {}
    for r in recs:
        Dof[(r["part"], r["clip"], r["label"], round(float(r["start"]), 3))] = r
        if r[Dk] is not None and r[Dk] > t:
            mask.setdefault((r["part"], r["clip"]), set()).add((r["label"], round(float(r["start"]), 3)))
    rows = SS.parts({k: frozenset(v) for k, v in mask.items()})          # parts() once per process
    base, new, lost, changed = [], [], [], []
    for pt, st, g, w, pics, pics1 in rows:
        r0, r1 = S.score_clip(g, pics), S.score_clip(g, pics1)
        base.append(r0); new.append(r1)
        if pics != pics1:
            cl = {(l, round(a, 3)): k for l, a, b, k, _ in classify(g, pics)}
            changed.append({"part": pt, "clip": st,
                            "removed": [[l, round(a, 2), round(b, 2), cl.get((l, round(a, 3)))] for l, a, b in pics if (l, a, b) not in pics1],
                            "added": [[l, round(a, 2), round(b, 2)] for l, a, b in pics1 if (l, a, b) not in pics],
                            "silenced_specs": [[l, a0, round(Dof[(pt, st, l, a0)]["D_a"] or 0, 3), round(Dof[(pt, st, l, a0)]["D_b"] or 0, 3)]
                                               for l, a0 in sorted(mask.get((pt, st), ()))]})
        if r1["hit"] < r0["hit"]:
            lost.append((pt, st, r0["hit"] - r1["hit"]))
    Bm, M = SS.summ_fmt(base)[0], SS.summ_fmt(new)[0]
    from benchmark.gold import btp_screen as Bt
    K = BASE_D
    assert (Bm["hits"], Bm["wrong"]) == (K["hits"], K["wrong"]) and abs(Bm["cost"] - K["cost"]) < 0.001, Bm
    ln = sum(x[2] for x in lost); removed_wrong = K["wrong"] - M["wrong"]
    cheaper = M["cost"] < K["cost"]
    unit = (removed_wrong >= 1) if ln == 0 else (removed_wrong >= 2 * ln)
    main = M["hits"] >= K["hits"] and not lost and cheaper
    few = cheaper and M["wrong"] <= K["wrong"] - 3 * ln and M["hits"] >= 26
    verdict = "PASS" if (cheaper and unit and (main or few)) else "FAIL"
    print(f"BASE D {Bt.fmt(Bm)} | LOGIT-GATE ({q}, t={t:.3f}) {Bt.fmt(M)}: lost {lost} wrong removed {removed_wrong} "
          f"cheaper {cheaper} unit {unit} main {main} few {few} -> {verdict}")
    for c in changed:
        print("   changed", c)
    named = [c for c in changed if c["clip"] == NAMED] or "unchanged"
    print("named", NAMED, named, [(r["label"], r["start"], r["D_a"], r["D_b"]) for r in recs if r["clip"] == NAMED])
    per_clip = {}
    for r in recs:
        per_clip[r["clip"]] = per_clip.get(r["clip"], 0) + 2 * sum(1 for x in r["stretches"] if x["lg"])
    pv = sorted(per_clip.values())
    print(f"cost: prefill passes per video (chosen question, drawn sounds) min {pv[0]} median {statistics.median(pv)} max {pv[-1]}")
    table = sorted([[r["part"], r["clip"], r["label"], r["start"], r["source"], len(r["stretches"]), r["D_a"], r["D_b"]] for r in recs],
                   key=lambda x: -(x[6 if q == "a" else 7] or -1e9))
    DEV_OUT.write_text(json.dumps({"question": q, "t": t, "base": Bm, "row": M, "ref": K, "hits_lost": lost,
                                   "wrong_removed": removed_wrong, "cheaper": cheaper, "unit": unit, "main_rule": main,
                                   "fewer_pictures": few, "verdict": verdict, "changed": changed, "named": named,
                                   "passes_per_video": per_clip, "specs_by_D": table}, indent=1, default=float), encoding="utf-8")
    print(DEV_OUT)


SCENE_TWIN = "Could the sound of {label} NOT plausibly be heard in this scene? Answer yes or no."


def cmd_scene():
    """Round 62 secondary SCENE-RECHECK: the DEV / DEV2 rows of D's memoised F3 asks re-read through the logit harness."""
    from src.stage5_cross_modal_analysis import reason
    from src.stage2_video_understanding import _sample_frames_at
    mp = Path.home() / "MscProj" / "data" / "work" / "scenemargin" / "videos.json"
    devclips = set(json.loads(mp.read_text(encoding="utf-8")))
    rows = []
    for ln in mp.with_name(mp.name + ".answers.jsonl").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            x = json.loads(ln)
            if x["key"][0] in devclips:                       # non-DEV rows skipped unread
                rows.append(x)
    print(len(rows), "DEV / DEV2 asks", flush=True)
    R = Reader()
    out, changed = [], []
    for x in rows:
        label = str(x["key"][1]).split(",")[0].split("(")[0].strip().lower()     # the family stage 4 passed (key), not e.label
        q, tw = reason.SCENE_FIT_PROMPT.format(label=label), SCENE_TWIN.format(label=label)
        sts = []
        for a_ in x["answers"]:
            a, b = a_["stretch"]
            lo, hi = a - 1.0, b + 1.0
            fr = _sample_frames_at(Path(x["video"]), [lo + (hi - lo) * t / 5 for t in range(6)])
            if not fr:
                sts.append({"stretch": [a, b], "s": None}); continue
            s, am, _ = R.s(q, fr)
            st_, _, _ = R.s(tw, fr)
            sts.append({"stretch": [a, b], "stored": a_["answer"], "s": s, "argmax": R.tok.decode([am]), "d_twin": s - st_})
        y, n = sum(1 for z in sts if z["s"] is not None and z["s"] > 0), sum(1 for z in sts if z["s"] is not None and z["s"] < 0)
        v = None if y + n == 0 else y > n
        rec = {"key": x["key"], "label": label, "stored_verdict": x["verdict"], "logit_verdict": v, "stretches": sts}
        out.append(rec)
        if v != x["verdict"]:
            changed.append(rec)
        print(x["key"], "stored", x["verdict"], [z.get("stored") for z in sts], "| logit", v,
              [(round(z["s"], 2), z["argmax"], round(z["d_twin"], 2)) for z in sts if z["s"] is not None], flush=True)
    print("SCENE-RECHECK:", len(rows), "asks;", len(changed), "verdicts differ:", [(c["key"], c["stored_verdict"], c["logit_verdict"]) for c in changed])
    (G / "logit_scene_recheck.json").write_text(json.dumps({"asks": out, "changed": changed, "twin": SCENE_TWIN,
                                                             "prompt": reason.SCENE_FIT_PROMPT}, indent=1, default=float), encoding="utf-8")


def cmd_scene_row():
    """Round 62 secondary: D's merged-DEV row with mv_protest Glass 4.75 returned (logit F3 = credible). In the saved arms the
    only mv_protest difference between B (SHIP8+MD3, span kept) and D (span dropped) is that Glass span (scenemargin_diff.json),
    so D's mv_protest pictures are replaced by B's work-root pictures placed under D's display flags."""
    from benchmark.gold import score_per_sound as S
    from benchmark.gold import round13_dev as R13
    from benchmark.gold import btp_screen as Bt
    from benchmark.gold.cross_group import classify
    SS = _arm()
    clip = "mv_protest_scene_movie"
    with R13.flags({k: R13.arm_cfg(SS.ARM)[k] for k in R13.DISPLAY_KEYS}):     # before parts(): it re-points R13 to DEV2
        pB = SS.placed(R13.R13 / "SHIP8+MD3_proposed", clip)
    rows = SS.parts()
    base, new = [], []
    for pt, st, g, w, pics, _ in rows:
        base.append(S.score_clip(g, pics))
        if pt == "dev" and st == clip:
            print("D  ", [x[:4] for x in classify(g, pics)])
            print("D+G", [x[:4] for x in classify(g, pB)])
            new.append(S.score_clip(g, pB))
        else:
            new.append(S.score_clip(g, pics))
    Bm, M = SS.summ_fmt(base)[0], SS.summ_fmt(new)[0]
    assert (Bm["hits"], Bm["wrong"]) == (BASE_D["hits"], BASE_D["wrong"]) and abs(Bm["cost"] - BASE_D["cost"]) < 0.001, Bm
    print(f"SCENE-RECHECK row: D {Bt.fmt(Bm)} -> D with Glass 4.75 back {Bt.fmt(M)}")
    f = G / "logit_scene_recheck.json"
    d = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    d["row"] = {"D": Bm, "D_glass_back": M}
    f.write_text(json.dumps(d, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    {"scene_row": cmd_scene_row, "scene": cmd_scene, "sanity": cmd_sanity, "gold": cmd_gold, "score_gold": cmd_score_gold, "dev": cmd_dev, "score_dev": cmd_score_dev}[sys.argv[1]]()
