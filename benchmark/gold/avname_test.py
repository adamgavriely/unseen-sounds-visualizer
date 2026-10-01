"""Round 44 TEST read (docs/prereg_round13_detector_push.md "Round 44 TEST read"): AVNAME variant (b), frozen, on the merged TEST
(expect_test.parts: shipped SHIP8 TEST pictures; base 23/42/29 (4/20/5) 2.568 must reproduce). Every placed picture is asked the
Round 44 question on its audio cut + 8 frames; a picture is dropped iff its answer is NONE or unmapped (avname_screen matcher).
Gold is read ONLY in `score`. Reported, not selected on.

    TG_ARMS=SHIP8 python benchmark/gold/avname_test.py ask      # GPU msproj -> avname/test_answers.json (no gold)
    TG_ARMS=SHIP8 python benchmark/gold/avname_test.py score    # CPU: gold read here only -> avname_test.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")
from benchmark.gold import avname_screen as A
from benchmark.gold import expect_test as ET
from src.labels import canonical

ANS = A.DIR / "test_answers.json"
OUT = _ROOT / "benchmark" / "gold" / "avname_test.json"


def items():
    """[(part, clip, label, start, clip path)]; gold never read"""
    return [(pt, st, lab, float(a), cp) for pt, st, pics, cp in ET.parts() for lab, a, _b in pics]


def cmd_ask():
    import soundfile as sf
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from src.stage2_video_understanding import _sample_frames_at
    its = items()
    A.DIR.mkdir(parents=True, exist_ok=True)
    ans = json.loads(ANS.read_text(encoding="utf-8")) if ANS.exists() else {}
    todo = [x for x in its if A.key(*x[:4]) not in ans]
    print(f"{len(its)} placed TEST pictures, {len(todo)} to ask", flush=True)
    if not todo:
        return
    for pt, st, _l, _a, cp in todo:
        assert (ET.WAV[pt] / f"{st}.wav").exists(), (pt, st)
        assert cp.exists(), (pt, st, cp)
    t0 = time.time()
    proc = Qwen3OmniMoeProcessor.from_pretrained(A.MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(A.MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass
    print(f"[ask] model loaded in {time.time() - t0:.0f} s", flush=True)
    conv = [{"role": "user", "content": [{"type": "video", "video": "x"}, {"type": "audio", "audio": "x"}, {"type": "text", "text": A.Q}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    wcache = {}

    def wav(pt, st):
        k = (pt, st)
        if k not in wcache:
            w, sr = sf.read(str(ET.WAV[pt] / f"{st}.wav"), dtype="float32")
            assert sr == A.SR and w.ndim == 1, (st, sr, w.shape)
            wcache.clear(); wcache[k] = w
        return wcache[k]

    def cut(w, a):                                              # avname_screen.cmd_ask.cut, verbatim
        n = int(round((A.PRE + A.POST) * A.SR)); seg = np.zeros(n, dtype=np.float32)
        i = int(round((a - A.PRE) * A.SR))
        lo, hi = max(i, 0), min(i + n, len(w))
        if hi > lo:
            seg[lo - i:hi - i] = w[lo:hi]
        return seg

    def resize(im):
        s = A.LONG / max(im.size)
        return im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s)))) if s < 1 else im

    t1 = time.time()
    for n, (pt, st, lab, a, cp) in enumerate(todo, 1):
        seg = cut(wav(pt, st), a)
        times = [a - A.PRE + (i + 0.5) * (A.PRE + A.POST) / A.N_FRAMES for i in range(A.N_FRAMES)]
        frames = [resize(im) for im in _sample_frames_at(Path(cp), times)]
        assert frames, (pt, st, a)
        meta = [{"fps": A.FPS, "duration": len(frames) / A.FPS, "total_num_frames": len(frames)}]
        inp = proc(text=text, audio=[seg], videos=[frames], video_metadata=meta, return_tensors="pt", padding=True,
                   use_audio_in_video=False, cap_pixels_per_frame=False)
        inp = inp.to(model.thinker.device).to(torch.bfloat16)
        with torch.inference_mode():
            out = model.thinker.generate(**inp, max_new_tokens=A.MAX_NEW, do_sample=False)
        raw = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        ans[A.key(pt, st, lab, a)] = {"part": pt, "clip": st, "label": lab, "family": canonical(lab), "start": round(a, 3),
                                      "n_frames": len(frames), "frame_size": list(frames[0].size), "tokens": int(inp["input_ids"].shape[1]),
                                      "raw": raw, "answer": A.norm(raw)}
        if n <= 3 or n % 20 == 0:
            print(f"[ask] {n}/{len(todo)} {pt} {st} {lab} @{a:.2f}: {raw!r} ({time.time() - t1:.0f} s)", flush=True)
            ANS.write_text(json.dumps(ans, indent=1), encoding="utf-8")
    ANS.write_text(json.dumps(ans, indent=1), encoding="utf-8")
    print(f"[ask] done {len(todo)} in {time.time() - t1:.0f} s -> {ANS}", flush=True)


def cmd_score():
    from benchmark.gold import score_per_sound as S
    from benchmark.gold import final_test as FT
    from benchmark.gold import dev_candidates_check as DCC
    from benchmark.gold import tagger_prep as TP
    from benchmark.gold.cross_group import classify
    P = ET.parts()
    ans = json.loads(ANS.read_text(encoding="utf-8"))
    tab = A.build_matcher()
    S.load_gold = FT._REAL_LOAD_GOLD                           # expect_test.cmd_score's gold loading, verbatim
    gold1 = S.load_gold([FT.G / "annotations" / "gold_AG.json"])
    stems1 = [st for pt, st, _p, _c in P if pt == "test"]; stems2 = [st for pt, st, _p, _c in P if pt == "test2"]
    assert sorted(S.subsets_of(gold1)["test_bench"]) == sorted(stems1)
    o2 = TP.out("test2")
    dg = json.loads(TP.TAGGER_GOLD.read_text(encoding="utf-8"))
    dg["clips"] = [c for c in dg.get("clips", []) if isinstance(c, dict) and Path(str(c.get("clip", ""))).stem in set(stems2)]
    tmp = o2 / "test2_gold_only.json"; DCC.dump(tmp, dg)
    gold2 = S.load_gold([tmp])
    gold = {("test", st): gold1[st] for st in stems1} | {("test2", st): gold2[st] for st in stems2 if st in gold2}
    tally = {"pictures": 0, "same_family": 0, "other_family": 0, "NONE": 0, "unmapped": 0, "ambiguous": 0}
    r0s, r1s, c0, c1, dropped, lost = [], [], [], [], [], []
    pr = {"base": {"test": [], "test2": []}, "new": {"test": [], "test2": []}}
    oc = lambda r: f"h{r['hit']} v{r['visible']} c{r['cross']} p{r['phantom']} d{r['dup']}"
    for pt, st, pics, _c in P:
        if (pt, st) not in gold:
            continue
        g = gold[(pt, st)]
        cl0 = {(l, round(a, 3)): k for l, a, b, k, _ in classify(g, list(pics))}
        new = []
        for lab, a, b in pics:
            x = ans[A.key(pt, st, lab, a)]
            f, nf = A.map_answer(tab, x["answer"])
            tally["pictures"] += 1; tally["ambiguous"] += nf > 1
            tally["NONE" if f == "NONE" else "unmapped" if f is None else "same_family" if f == canonical(lab) else "other_family"] += 1
            if f is None or f == "NONE":
                dropped.append({"part": pt, "clip": st, "label": lab, "start": round(a, 2), "answer": x["answer"], "raw": x["raw"],
                                "class": cl0[(lab, round(a, 3))]})
            else:
                new.append((lab, a, b))
        r0 = S.score_clip(g, list(pics)); r1 = S.score_clip(g, new)
        r0s.append(r0); r1s.append(r1); c0.append(DCC.clip_cost(r0)); c1.append(DCC.clip_cost(r1))
        pr["base"][pt].append(r0); pr["new"][pt].append(r1)
        for d in dropped:
            if d["part"] == pt and d["clip"] == st and "clip_after" not in d:
                d["clip_before"], d["clip_after"] = oc(r0), oc(r1)
        if r1["hit"] < r0["hit"]:
            lost.append((pt, st, r0["hit"] - r1["hit"]))
    m0, m1 = DCC.metrics(r0s), DCC.metrics(r1s)
    n = len(r0s)
    cw = lambda m, w: (4 * m["misses"] + w * m["visible"] + 2 * m["cross"] + 2 * m["phantom"]) / n
    assert (m0["hits"], m0["misses"], m0["wrong"]) == (ET.BASE["hits"], ET.BASE["misses"], ET.BASE["wrong"]), m0
    assert abs(m0["viewer_cost"] - ET.BASE["cost"]) < 0.001, m0["viewer_cost"]
    d_ = DCC.boot(np.subtract(c1, c0))
    fmt = lambda m: f"{m['hits']}/{m['hits'] + m['misses']} wrong {m['wrong']} ({m['visible']}/{m['cross']}/{m['phantom']}) cost {m['viewer_cost']:.3f}"
    print(f"answers: {tally}")
    print(f"BASE SHIP8 TEST:  {fmt(m0)}  (old TEST {fmt(DCC.metrics(pr['base']['test']))} | tagger TEST {fmt(DCC.metrics(pr['base']['test2']))})")
    print(f"AVNAME(b) TEST:   {fmt(m1)}  (old TEST {fmt(DCC.metrics(pr['new']['test']))} | tagger TEST {fmt(DCC.metrics(pr['new']['test2']))})")
    print(f"  cost w=2 {cw(m0, 2):.3f} -> {cw(m1, 2):.3f}; w=1 {cw(m0, 1):.3f} -> {cw(m1, 1):.3f}; d cost {d_[0]:+.3f} [{d_[1]:+.3f}, {d_[2]:+.3f}] one-sided p {d_[3]:.3f}")
    print(f"  dropped {len(dropped)}  hits lost {lost}")
    for d in dropped:
        print(f"   {d['part']:5s} {d['clip']} {d['label']} @{d['start']} answer {d['answer']!r} -> dropped: {d['class']}  clip {d['clip_before']} -> {d['clip_after']}")
    res = {"arm": "SHIP8 + AVNAME(b) (TEST read)", "clips": n, "base": m0, "new": m1, "answers": tally,
           "parts": {k: {pt: DCC.metrics(v[pt]) for pt in v} for k, v in pr.items()},
           "cost": {"w2": [round(cw(m0, 2), 3), round(cw(m1, 2), 3)], "w1": [round(cw(m0, 1), 3), round(cw(m1, 1), 3)]},
           "delta_cost_vs_base": d_, "dropped": dropped, "hits_lost": lost}
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    {"ask": cmd_ask, "score": cmd_score}[sys.argv[1]]()
